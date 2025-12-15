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

function stipple(ctx: FnContext, c: string): Image {
  const prev = getPrevImage(ctx);
  const out = cloneImage(prev);
  const [cr, cg, cb] = hexToRgb(c);

  for (let y = 0; y < ctx.height; y++) {
    for (let x = 0; x < ctx.width; x++) {
      const [r, g, b] = getPixel(prev, x, y);
      const luminance = Math.floor(r * 0.299 + g * 0.587 + b * 0.114);
      const divisor = 1 + Math.floor(luminance / 32);

      if ((x * 13 + y * 7) % divisor === 0) {
        setPixel(out, x, y, cr, cg, cb);
      }
    }
  }

  return out;
}

export { stipple };
