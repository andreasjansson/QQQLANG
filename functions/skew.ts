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

function skew(ctx: FnContext, amount: number): Image {
  const prev = getPrevImage(ctx);
  const out = createSolidImage(ctx.width, ctx.height, "#000000");

  const normalizedAmount = ((amount - 1) / 67) * 2 - 1;
  const maxSkewAngle = 45;
  const skewAngle = normalizedAmount * maxSkewAngle;
  const skewAmount = (Math.tan((skewAngle * Math.PI) / 180) * ctx.height) / 2;

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

export { skew };
