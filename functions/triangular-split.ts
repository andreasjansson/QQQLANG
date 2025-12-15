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

function triangularSplit(ctx: FnContext, n: number): Image {
  const prev = getPrevImage(ctx);
  const out = cloneImage(prev);

  const cellSize = Math.max(20, n * 4);
  const cols = Math.max(2, Math.floor(ctx.width / cellSize));
  const rows = Math.max(2, Math.floor(ctx.height / cellSize));
  const totalTriangles = cols * rows * 2;

  for (let row = 0; row < rows; row++) {
    for (let col = 0; col < cols; col++) {
      const x0 = Math.floor((col / cols) * ctx.width);
      const y0 = Math.floor((row / rows) * ctx.height);
      const x1 = Math.floor(((col + 1) / cols) * ctx.width);
      const y1 = Math.floor(((row + 1) / rows) * ctx.height);
      const cellW = x1 - x0;
      const cellH = y1 - y0;

      const triIndex1 = (row * cols + col) * 2;
      const triIndex2 = triIndex1 + 1;

      const hueShift1 = (triIndex1 * 137.5) % 360;
      const hueShift2 = (triIndex2 * 137.5) % 360;
      const lightMod1 = triIndex1 % 2 === 0 ? 0.15 : -0.15;
      const lightMod2 = triIndex2 % 2 === 0 ? 0.15 : -0.05;

      for (let y = y0; y < y1; y++) {
        for (let x = x0; x < x1; x++) {
          const localX = (x - x0) / cellW;
          const localY = (y - y0) / cellH;

          const [r, g, b] = getPixel(prev, x, y);
          const [h, s, l] = rgbToHsl(r, g, b);
          const isUpperTriangle = localX + localY < 1;

          const hueShift = isUpperTriangle ? hueShift1 : hueShift2;
          const lightMod = isUpperTriangle ? lightMod1 : lightMod2;
          const newL = Math.max(0, Math.min(1, l + lightMod));

          const [nr, ng, nb] = hslToRgb(
            (h + hueShift) % 360,
            Math.min(1, s * 1.3),
            newL,
          );
          setPixel(out, x, y, nr, ng, nb);
        }
      }
    }
  }

  return out;
}

export { triangularSplit };
