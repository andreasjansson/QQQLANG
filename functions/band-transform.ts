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

function bandTransform(ctx: FnContext, n: number): Image {
  const prev = getPrevImage(ctx);
  const out = createSolidImage(ctx.width, ctx.height, "#000000");
  const bands = Math.max(2, n);
  const bandHeight = ctx.height / bands;

  for (let y = 0; y < ctx.height; y++) {
    const bandIdx = Math.floor(y / bandHeight);
    const isOdd = bandIdx % 2 === 1;

    for (let x = 0; x < ctx.width; x++) {
      const [r, g, b] = getPixel(prev, x, y);

      if (isOdd) {
        const [h, s, l] = rgbToHsl(r, g, b);
        const [nr, ng, nb] = hslToRgb((h + 180) % 360, s, l);
        setPixel(out, x, y, nr, ng, nb);
      } else {
        const [h, s, l] = rgbToHsl(r, g, b);
        const [nr, ng, nb] = hslToRgb(h, 1 - s, l);
        setPixel(out, x, y, nr, ng, nb);
      }
    }
  }

  return out;
}

export { bandTransform };
