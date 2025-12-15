import {
  FnContext,
  Image,
  createSolidImage,
  getPrevImage,
  getPixel,
  setPixel,
  cloneImage,
  hexToRgb,
  rgbToHsl,
  hslToRgb,
  initWebGL,
  createShaderProgram,
  getOldImage,
  createPlaceholderImage,
  emeraldReady,
  bgRemovalReady,
} from "./helpers.js";

function composite(
  ctx: FnContext,
  old: Image,
  srcX: number,
  srcY: number,
  srcW: number,
  srcH: number,
  dstX: number,
  dstY: number,
  dstW: number,
  dstH: number,
  rot: number,
  blend: number,
): Image {
  const prev = getPrevImage(ctx);
  const out = cloneImage(prev);

  // Map integer values (1-68) to normalized 0-1 range
  const norm = (n: number) => Math.max(0, Math.min(1, (n - 1) / 67));

  // Source crop region in old image
  // X and Y use full range, W and H use remaining space after offset
  const sourceCropX = Math.floor(norm(srcX) * old.width);
  const sourceCropY = Math.floor(norm(srcY) * old.height);
  const sourceCropW = Math.max(
    1,
    Math.round(norm(srcW) * (old.width - sourceCropX)),
  );
  const sourceCropH = Math.max(
    1,
    Math.round(norm(srcH) * (old.height - sourceCropY)),
  );

  // Destination region in output image
  // X and Y use full range, W and H use remaining space after offset
  const destX = Math.floor(norm(dstX) * ctx.width);
  const destY = Math.floor(norm(dstY) * ctx.height);
  const destW = Math.max(1, Math.round(norm(dstW) * (ctx.width - destX)));
  const destH = Math.max(1, Math.round(norm(dstH) * (ctx.height - destY)));

  // Rotation angle (0-360 degrees)
  const rotation = norm(rot) * 2 * Math.PI;

  // Blend mode (0-15)
  const NUM_BLEND_MODES = 16;
  const blendMode = (blend - 1) % NUM_BLEND_MODES;

  // Blend mode functions: (base, top) => result (all values 0-255)
  const blendFuncs: ((b: number, t: number) => number)[] = [
    // 0: Normal - replace
    (b, t) => t,
    // 1: XOR
    (b, t) => b ^ t,
    // 2: NAND
    (b, t) => 255 - (b & t),
    // 3: AND
    (b, t) => b & t,
    // 4: OR
    (b, t) => b | t,
    // 5: Multiply
    (b, t) => (b * t) / 255,
    // 6: Screen
    (b, t) => 255 - ((255 - b) * (255 - t)) / 255,
    // 7: Overlay
    (b, t) =>
      b < 128 ? (2 * b * t) / 255 : 255 - (2 * (255 - b) * (255 - t)) / 255,
    // 8: Darken
    (b, t) => Math.min(b, t),
    // 9: Lighten
    (b, t) => Math.max(b, t),
    // 10: Difference
    (b, t) => Math.abs(b - t),
    // 11: Exclusion
    (b, t) => b + t - (2 * b * t) / 255,
    // 12: Add (clamped)
    (b, t) => Math.min(255, b + t),
    // 13: Subtract (clamped)
    (b, t) => Math.max(0, b - t),
    // 14: Hard Light
    (b, t) =>
      t < 128 ? (2 * b * t) / 255 : 255 - (2 * (255 - b) * (255 - t)) / 255,
    // 15: Soft Light
    (b, t) => {
      const tb = t / 255,
        bb = b / 255;
      return Math.round(
        (tb < 0.5
          ? bb - (1 - 2 * tb) * bb * (1 - bb)
          : bb +
            (2 * tb - 1) *
              (bb < 0.25
                ? ((16 * bb - 12) * bb + 4) * bb
                : Math.sqrt(bb) - bb)) * 255,
      );
    },
  ];

  const blendFunc = blendFuncs[blendMode];

  const destCenterX = destX + destW / 2;
  const destCenterY = destY + destH / 2;
  const cosR = Math.cos(-rotation);
  const sinR = Math.sin(-rotation);

  // Calculate bounding box of rotated rectangle
  const halfW = destW / 2;
  const halfH = destH / 2;
  const corners = [
    [-halfW, -halfH],
    [halfW, -halfH],
    [halfW, halfH],
    [-halfW, halfH],
  ];

  let minX = Infinity,
    maxX = -Infinity,
    minY = Infinity,
    maxY = -Infinity;
  for (const [cx, cy] of corners) {
    const rx = cx * Math.cos(rotation) - cy * Math.sin(rotation) + destCenterX;
    const ry = cx * Math.sin(rotation) + cy * Math.cos(rotation) + destCenterY;
    minX = Math.min(minX, rx);
    maxX = Math.max(maxX, rx);
    minY = Math.min(minY, ry);
    maxY = Math.max(maxY, ry);
  }

  const startX = Math.max(0, Math.floor(minX));
  const endX = Math.min(ctx.width, Math.ceil(maxX));
  const startY = Math.max(0, Math.floor(minY));
  const endY = Math.min(ctx.height, Math.ceil(maxY));

  for (let py = startY; py < endY; py++) {
    for (let px = startX; px < endX; px++) {
      // Inverse rotation to find source coordinates
      const relX = px - destCenterX;
      const relY = py - destCenterY;
      const rotX = relX * cosR - relY * sinR;
      const rotY = relX * sinR + relY * cosR;

      // Map to normalized coordinates in destination rect
      const normX = (rotX + halfW) / destW;
      const normY = (rotY + halfH) / destH;

      // Check if within the destination rectangle (0-1 range)
      if (normX < 0 || normX >= 1 || normY < 0 || normY >= 1) continue;

      // Map to source coordinates with bilinear sampling
      const srcPxF = sourceCropX + normX * sourceCropW;
      const srcPyF = sourceCropY + normY * sourceCropH;

      const x0 = Math.floor(srcPxF);
      const y0 = Math.floor(srcPyF);
      const x1 = Math.min(old.width - 1, x0 + 1);
      const y1 = Math.min(old.height - 1, y0 + 1);
      const fx = srcPxF - x0;
      const fy = srcPyF - y0;

      const [r00, g00, b00] = getPixel(old, x0, y0);
      const [r10, g10, b10] = getPixel(old, x1, y0);
      const [r01, g01, b01] = getPixel(old, x0, y1);
      const [r11, g11, b11] = getPixel(old, x1, y1);

      // Bilinear interpolation for source pixel
      const srcR = Math.round(
        r00 * (1 - fx) * (1 - fy) +
          r10 * fx * (1 - fy) +
          r01 * (1 - fx) * fy +
          r11 * fx * fy,
      );
      const srcG = Math.round(
        g00 * (1 - fx) * (1 - fy) +
          g10 * fx * (1 - fy) +
          g01 * (1 - fx) * fy +
          g11 * fx * fy,
      );
      const srcB = Math.round(
        b00 * (1 - fx) * (1 - fy) +
          b10 * fx * (1 - fy) +
          b01 * (1 - fx) * fy +
          b11 * fx * fy,
      );

      // Get base pixel from prev
      const [baseR, baseG, baseB] = getPixel(prev, px, py);

      // Apply blend mode
      const r = Math.round(blendFunc(baseR, srcR));
      const g = Math.round(blendFunc(baseG, srcG));
      const b = Math.round(blendFunc(baseB, srcB));

      setPixel(out, px, py, r, g, b);
    }
  }

  return out;
}

export { composite };
