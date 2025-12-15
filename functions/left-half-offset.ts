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

function leftHalfOffset(ctx: FnContext): Image {
  const prev = getPrevImage(ctx);
  const out = createSolidImage(ctx.width, ctx.height, "#000000");

  const offset = Math.floor(ctx.height * 0.2);
  const midX = Math.floor(ctx.width / 2);

  for (let y = 0; y < ctx.height; y++) {
    for (let x = 0; x < ctx.width; x++) {
      if (x < midX) {
        const srcY = (y + offset) % ctx.height;
        const [r, g, b] = getPixel(prev, x, srcY);
        setPixel(out, x, y, r, g, b);
      } else {
        const [r, g, b] = getPixel(prev, x, y);
        setPixel(out, x, y, r, g, b);
      }
    }
  }

  return out;
}

export { leftHalfOffset };
