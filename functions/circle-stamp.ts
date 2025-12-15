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

function circleStamp(
  ctx: FnContext,
  old: Image,
  offX: number,
  offY: number,
  size: number,
  blend: string,
): Image {
  const prev = getPrevImage(ctx);
  const out = cloneImage(prev);

  const norm = (n: number) => Math.max(0, Math.min(1, (n - 0.5) / 67));

  // Source: full circle from center of old image
  const srcCenterX = old.width / 2;
  const srcCenterY = old.height / 2;
  const srcRadius = Math.min(old.width, old.height) / 2;

  // Destination position and size
  const dstX = norm(offX) * ctx.width;
  const dstY = norm(offY) * ctx.height;
  const dstRadius = Math.max(1, norm(size) * Math.min(ctx.width, ctx.height));

  const blendFuncs: Record<string, (b: number, t: number) => number> = {
    normal: (b, t) => b,
    xor: (b, t) => b ^ t,
    nand: (b, t) => 255 - (b & t),
    and: (b, t) => b & t,
    or: (b, t) => b | t,
    multiply: (b, t) => (b * t) / 255,
    screen: (b, t) => 255 - ((255 - b) * (255 - t)) / 255,
    overlay: (b, t) =>
      b < 128 ? (2 * b * t) / 255 : 255 - (2 * (255 - b) * (255 - t)) / 255,
    darken: (b, t) => Math.min(b, t),
    lighten: (b, t) => Math.max(b, t),
    difference: (b, t) => Math.abs(b - t),
    exclusion: (b, t) => b + t - (2 * b * t) / 255,
    add: (b, t) => Math.min(255, b + t),
    subtract: (b, t) => Math.max(0, b - t),
    hardlight: (b, t) =>
      t < 128 ? (2 * b * t) / 255 : 255 - (2 * (255 - b) * (255 - t)) / 255,
    softlight: (b, t) => {
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
  };

  const blendFunc = blendFuncs[blend] || blendFuncs["normal"];

  // Scale factor from destination to source
  const scale = srcRadius / dstRadius;

  // Iterate over bounding box of destination circle
  const startX = Math.max(0, Math.floor(dstX - dstRadius));
  const endX = Math.min(ctx.width, Math.ceil(dstX + dstRadius));
  const startY = Math.max(0, Math.floor(dstY - dstRadius));
  const endY = Math.min(ctx.height, Math.ceil(dstY + dstRadius));

  for (let py = startY; py < endY; py++) {
    for (let px = startX; px < endX; px++) {
      const dx = px - dstX;
      const dy = py - dstY;
      const distSq = dx * dx + dy * dy;

      if (distSq > dstRadius * dstRadius) continue;

      // Map to source coordinates
      const srcPxF = srcCenterX + dx * scale;
      const srcPyF = srcCenterY + dy * scale;

      // Bilinear interpolation
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

      const [baseR, baseG, baseB] = getPixel(prev, px, py);

      const r = Math.round(blendFunc(srcR, baseR));
      const g = Math.round(blendFunc(srcG, baseG));
      const b = Math.round(blendFunc(srcB, baseB));

      setPixel(out, px, py, r, g, b);
    }
  }

  return out;
}

export { circleStamp };
