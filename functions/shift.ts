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

function horizontalShift(ctx: FnContext, amount: number): Image {
  const prev = getPrevImage(ctx);
  const out = createSolidImage(ctx.width, ctx.height, "#000000");

  const normalizedAmount = ((amount - 1) / 67) * 2 - 1;
  const shift = Math.floor((normalizedAmount * ctx.width) / 2);

  for (let y = 0; y < ctx.height; y++) {
    for (let x = 0; x < ctx.width; x++) {
      const srcX = (((x - shift) % ctx.width) + ctx.width) % ctx.width;
      const [r, g, b] = getPixel(prev, srcX, y);
      setPixel(out, x, y, r, g, b);
    }
  }

  return out;
}

export { horizontalShift };
