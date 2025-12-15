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

function lemniscate(ctx: FnContext, n: number): Image {
  const prev = getPrevImage(ctx);
  const out = createSolidImage(ctx.width, ctx.height, "#000000");

  const strength = Math.max(0.1, n / 5);
  const cx = ctx.width / 2;
  const cy = ctx.height / 2;
  const scale = Math.min(ctx.width, ctx.height) / 4;

  for (let y = 0; y < ctx.height; y++) {
    for (let x = 0; x < ctx.width; x++) {
      const nx = (x - cx) / scale;
      const ny = (y - cy) / scale;

      const r2 = nx * nx + ny * ny;
      const denom = r2 + 1;

      const lemnX = (nx * (r2 - 1)) / denom;
      const lemnY = (ny * (r2 + 1)) / denom;

      const sx = cx + (nx + (lemnX - nx) * strength * 0.3) * scale;
      const sy = cy + (ny + (lemnY - ny) * strength * 0.3) * scale;

      const [r, g, b] = getPixel(prev, Math.floor(sx), Math.floor(sy));
      setPixel(out, x, y, r, g, b);
    }
  }

  return out;
}

export { lemniscate };
