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

function tripleRotate(ctx: FnContext): Image {
  const prev = getPrevImage(ctx);
  const out = createSolidImage(ctx.width, ctx.height, "#000000");

  const stripWidth = Math.floor(ctx.width / 3);

  for (let y = 0; y < ctx.height; y++) {
    for (let x = 0; x < stripWidth; x++) {
      const srcX = Math.floor((y / ctx.height) * stripWidth);
      const srcY = Math.floor(((stripWidth - 1 - x) / stripWidth) * ctx.height);
      const [r, g, b] = getPixel(prev, srcX, srcY);
      setPixel(out, x, y, r, g, b);
    }
  }

  for (let y = 0; y < ctx.height; y++) {
    for (let x = stripWidth; x < stripWidth * 2; x++) {
      const srcX = Math.floor(
        ((x - stripWidth) / stripWidth) * stripWidth + stripWidth,
      );
      const srcY = y;
      const [r, g, b] = getPixel(prev, srcX, srcY);
      setPixel(out, x, y, r, g, b);
    }
  }

  for (let y = 0; y < ctx.height; y++) {
    for (let x = stripWidth * 2; x < ctx.width; x++) {
      const localX = x - stripWidth * 2;
      const srcX = Math.floor(
        ((ctx.height - 1 - y) / ctx.height) * (ctx.width - stripWidth * 2) +
          stripWidth * 2,
      );
      const srcY = Math.floor(
        (localX / (ctx.width - stripWidth * 2)) * ctx.height,
      );
      const [r, g, b] = getPixel(prev, srcX, srcY);
      setPixel(out, x, y, r, g, b);
    }
  }

  return out;
}

export { tripleRotate };
