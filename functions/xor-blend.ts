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

function xorBlend(ctx: FnContext, old: Image): Image {
  const prev = getPrevImage(ctx);
  const out = createSolidImage(ctx.width, ctx.height, "#000000");

  for (let y = 0; y < ctx.height; y++) {
    for (let x = 0; x < ctx.width; x++) {
      const [r1, g1, b1] = getPixel(prev, x, y);
      const [r2, g2, b2] = getPixel(old, x, y);

      setPixel(out, x, y, r1 ^ r2, g1 ^ g2, b1 ^ b2);
    }
  }

  return out;
}

export { xorBlend };
