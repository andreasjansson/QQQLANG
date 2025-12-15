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

function waveChromatic(ctx: FnContext, n: number): Image {
  const prev = getPrevImage(ctx);
  const out = createSolidImage(ctx.width, ctx.height, "#000000");

  const waveN = Math.max(1, n);

  for (let y = 0; y < ctx.height; y++) {
    const amplitude = Math.sin(y * 0.05) * waveN * 8;

    for (let x = 0; x < ctx.width; x++) {
      const srcX = Math.floor(
        (((x + amplitude) % ctx.width) + ctx.width) % ctx.width,
      );

      const srcXR = Math.floor(
        (((x + amplitude + waveN) % ctx.width) + ctx.width) % ctx.width,
      );
      const srcXB = Math.floor(
        (((x + amplitude - waveN) % ctx.width) + ctx.width) % ctx.width,
      );

      const [rr] = getPixel(prev, srcXR, y);
      const [, gg] = getPixel(prev, srcX, y);
      const [, , bb] = getPixel(prev, srcXB, y);

      setPixel(out, x, y, rr, gg, bb);
    }
  }

  return out;
}

export { waveChromatic };
