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

function rotate90(ctx: FnContext): Image {
  const prev = getPrevImage(ctx);
  const out = createSolidImage(ctx.width, ctx.height, "#000000");

  for (let y = 0; y < ctx.height; y++) {
    for (let x = 0; x < ctx.width; x++) {
      const sx = (y * prev.width) / ctx.height;
      const sy = ((ctx.width - 1 - x) * prev.height) / ctx.width;
      const [r, g, b] = getPixel(prev, Math.floor(sx), Math.floor(sy));
      setPixel(out, x, y, r, g, b);
    }
  }

  return out;
}

export { rotate90 };
