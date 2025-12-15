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

function diagonalPixelate(ctx: FnContext, n: number): Image {
  const prev = getPrevImage(ctx);
  const out = createSolidImage(ctx.width, ctx.height, "#000000");

  const cellSize = Math.max(2, Math.min(n + 1, 50));

  for (let by = 0; by < ctx.height; by += cellSize) {
    for (let bx = 0; bx < ctx.width; bx += cellSize) {
      let sumR = 0,
        sumG = 0,
        sumB = 0,
        count = 0;
      let maxSat = 0;
      let mostSatR = 0,
        mostSatG = 0,
        mostSatB = 0;

      for (let y = by; y < by + cellSize && y < ctx.height; y++) {
        for (let x = bx; x < bx + cellSize && x < ctx.width; x++) {
          const [r, g, b] = getPixel(prev, x, y);
          sumR += r;
          sumG += g;
          sumB += b;
          count++;

          const [h, s, l] = rgbToHsl(r, g, b);
          if (s > maxSat) {
            maxSat = s;
            mostSatR = r;
            mostSatG = g;
            mostSatB = b;
          }
        }
      }

      const avgR = Math.round(sumR / count);
      const avgG = Math.round(sumG / count);
      const avgB = Math.round(sumB / count);

      for (let y = by; y < by + cellSize && y < ctx.height; y++) {
        for (let x = bx; x < bx + cellSize && x < ctx.width; x++) {
          const localX = x - bx;
          const localY = y - by;
          const isTopLeft = localX + localY < cellSize;

          if (isTopLeft) {
            setPixel(out, x, y, avgR, avgG, avgB);
          } else {
            setPixel(out, x, y, mostSatR, mostSatG, mostSatB);
          }
        }
      }
    }
  }

  return out;
}

export { diagonalPixelate };
