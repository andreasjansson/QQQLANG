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

function zoom(ctx: FnContext): Image {
  const prev = getPrevImage(ctx);
  const out = createSolidImage(ctx.width, ctx.height, "#000000");

  const cx = ctx.width / 2;
  const cy = ctx.height / 2;
  const zoom = 1.2;

  for (let y = 0; y < ctx.height; y++) {
    for (let x = 0; x < ctx.width; x++) {
      const sx = cx + (x - cx) / zoom;
      const sy = cy + (y - cy) / zoom;
      const [r, g, b] = getPixel(prev, Math.floor(sx), Math.floor(sy));
      setPixel(out, x, y, r, g, b);
    }
  }

  return out;
}

export { zoom };
