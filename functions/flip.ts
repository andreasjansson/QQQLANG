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

function flip(ctx: FnContext, n: number): Image {
  const prev = getPrevImage(ctx);
  const out = createSolidImage(ctx.width, ctx.height, "#000000");

  const horizontal = n % 2 === 0;

  for (let y = 0; y < ctx.height; y++) {
    for (let x = 0; x < ctx.width; x++) {
      const srcX = horizontal ? ctx.width - 1 - x : x;
      const srcY = horizontal ? y : ctx.height - 1 - y;
      const [r, g, b] = getPixel(prev, srcX, srcY);
      setPixel(out, x, y, r, g, b);
    }
  }

  return out;
}

export { flip };
