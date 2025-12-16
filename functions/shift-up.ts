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

function verticalShift(ctx: FnContext, amount: number): Image {
  const prev = getPrevImage(ctx);
  const out = createSolidImage(ctx.width, ctx.height, "#000000");

  const normalizedAmount = ((amount - 1) / 67) * 2 - 1;
  const shift = Math.floor((normalizedAmount * ctx.height) / 2);

  for (let y = 0; y < ctx.height; y++) {
    for (let x = 0; x < ctx.width; x++) {
      const srcY = (((y - shift) % ctx.height) + ctx.height) % ctx.height;
      const [r, g, b] = getPixel(prev, x, srcY);
      setPixel(out, x, y, r, g, b);
    }
  }

  return out;
}

export { verticalShift };
