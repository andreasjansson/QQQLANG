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

function semicircleReflect(ctx: FnContext): Image {
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

      if (y < cy && dist < radius) {
        const [r, g, b] = getPixel(prev, x, y);
        setPixel(out, x, y, r, g, b);
      } else if (y >= cy) {
        const mirrorY = 2 * cy - y;
        const mirrorDx = x - cx;
        const mirrorDy = mirrorY - cy;
        const mirrorDist = Math.sqrt(mirrorDx * mirrorDx + mirrorDy * mirrorDy);

        if (mirrorDist < radius) {
          const wave = Math.sin(x * 0.1) * 10;
          const srcY = Math.floor(mirrorY + wave);
          const [r, g, b] = getPixel(
            prev,
            x,
            Math.max(0, Math.min(ctx.height - 1, srcY)),
          );
          setPixel(out, x, y, r, g, b);
        }
      }
    }
  }

  return out;
}

export { semicircleReflect };
