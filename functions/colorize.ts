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

function colorize(ctx: FnContext, c: string): Image {
  const prev = getPrevImage(ctx);
  const out = createSolidImage(ctx.width, ctx.height, "#000000");
  const [tr, tg, tb] = hexToRgb(c);

  for (let y = 0; y < ctx.height; y++) {
    for (let x = 0; x < ctx.width; x++) {
      const [r, g, b] = getPixel(prev, x, y);
      const luminance = r * 0.299 + g * 0.587 + b * 0.114;
      const factor = Math.pow(luminance / 255, 0.6);

      const tintedR = tr * factor;
      const tintedG = tg * factor;
      const tintedB = tb * factor;

      const [h, s, l] = rgbToHsl(tintedR, tintedG, tintedB);
      const boostedS = Math.min(1, s * 1.5);
      const [finalR, finalG, finalB] = hslToRgb(h, boostedS, l);

      setPixel(out, x, y, finalR, finalG, finalB);
    }
  }

  return out;
}

export { colorize };
