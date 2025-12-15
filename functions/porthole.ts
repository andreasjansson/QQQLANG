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

function porthole(ctx: FnContext, old: Image): Image {
  const prev = getPrevImage(ctx);
  const out = createSolidImage(ctx.width, ctx.height, "#000000");

  const cx = ctx.width / 2;
  const cy = ctx.height / 2;
  const radius = Math.min(cx, cy) * 0.9;

  for (let y = 0; y < ctx.height; y++) {
    for (let x = 0; x < ctx.width; x++) {
      const dx = x - cx;
      const dy = y - cy;
      const dist = Math.sqrt(dx * dx + dy * dy);

      if (dist < radius) {
        const [r, g, b] = getPixel(prev, x, y);
        setPixel(out, x, y, r, g, b);
      } else {
        const [r, g, b] = getPixel(old, x, y);
        setPixel(out, x, y, r, g, b);
      }
    }
  }

  return out;
}

export { porthole };
