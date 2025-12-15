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

function skewLeft(ctx: FnContext): Image {
  const prev = getPrevImage(ctx);
  const out = createSolidImage(ctx.width, ctx.height, "#000000");

  const skewAmount = (Math.tan((20 * Math.PI) / 180) * ctx.height) / 2;

  for (let y = 0; y < ctx.height; y++) {
    const rowSkew = skewAmount * (1 - (2 * y) / ctx.height);
    for (let x = 0; x < ctx.width; x++) {
      const sx = (((x + rowSkew) % ctx.width) + ctx.width) % ctx.width;
      const [r, g, b] = getPixel(prev, Math.floor(sx), y);
      setPixel(out, x, y, r, g, b);
    }
  }

  return out;
}

export { skewLeft };
