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

function scanlines(ctx: FnContext): Image {
  const prev = getPrevImage(ctx);
  const out = createSolidImage(ctx.width, ctx.height, "#000000");
  const spacing = 2;

  for (let y = 0; y < ctx.height; y++) {
    for (let x = 0; x < ctx.width; x++) {
      let [r, g, b] = getPixel(prev, x, y);

      if (y % spacing === 0) {
        r = Math.floor(r * 0.5);
        g = Math.floor(g * 0.5);
        b = Math.floor(b * 0.5);
      }

      if (y % (spacing * 2) === 0) {
        const srcY = Math.max(0, y - spacing);
        [r, g, b] = getPixel(prev, x, srcY);
      }

      setPixel(out, x, y, r, g, b);
    }
  }

  return out;
}

export { scanlines };
