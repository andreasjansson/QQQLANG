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

function prism(ctx: FnContext): Image {
  const prev = getPrevImage(ctx);
  const out = createSolidImage(ctx.width, ctx.height, "#000000");

  const hw = Math.floor(ctx.width / 2);
  const hh = Math.floor(ctx.height / 2);

  const temp = createSolidImage(hw, hh, "#000000");

  for (let y = 0; y < hh; y++) {
    for (let x = 0; x < hw; x++) {
      const sx = (x / hw) * prev.width;
      const sy = (y / hh) * prev.height;
      const [r, g, b] = getPixel(prev, Math.floor(sx), Math.floor(sy));

      if (x / hw + y / hh > 1) {
        setPixel(temp, x, y, 255 - r, 255 - g, 255 - b);
      } else {
        setPixel(temp, x, y, r, g, b);
      }
    }
  }

  for (let qy = 0; qy < 2; qy++) {
    for (let qx = 0; qx < 2; qx++) {
      for (let y = 0; y < hh; y++) {
        for (let x = 0; x < hw; x++) {
          const outX = qx * hw + x;
          const outY = qy * hh + y;
          if (outX < ctx.width && outY < ctx.height) {
            const srcX = qx === 1 ? hw - 1 - x : x;
            const srcY = qy === 1 ? hh - 1 - y : y;
            const [r, g, b] = getPixel(temp, srcX, srcY);
            setPixel(out, outX, outY, r, g, b);
          }
        }
      }
    }
  }

  return out;
}

export { prism };
