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

function shiftedStripes(ctx: FnContext, n: number): Image {
  const prev = getPrevImage(ctx);
  const out = createSolidImage(ctx.width, ctx.height, "#000000");

  const stripeHeight = Math.max(2, n);

  for (let y = 0; y < ctx.height; y++) {
    const stripeIdx = Math.floor(y / stripeHeight);
    const isOdd = stripeIdx % 2 === 1;
    const shift = isOdd ? stripeIdx * 5 : 0;

    for (let x = 0; x < ctx.width; x++) {
      const srcX = (((x + shift) % ctx.width) + ctx.width) % ctx.width;
      const [r, g, b] = getPixel(prev, srcX, y);
      setPixel(out, x, y, r, g, b);
    }
  }

  return out;
}

export { shiftedStripes };
