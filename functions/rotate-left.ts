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

function rotate(ctx: FnContext, amount: number): Image {
  const prev = getPrevImage(ctx);
  const out = createSolidImage(ctx.width, ctx.height, "#000000");

  const normalizedAmount = ((amount - 1) / 67) * 2 - 1;
  const maxRotation = 90;
  const degrees = normalizedAmount * maxRotation;
  const angle = (-degrees * Math.PI) / 180;
  const cos = Math.cos(angle);
  const sin = Math.sin(angle);
  const cx = ctx.width / 2;
  const cy = ctx.height / 2;

  for (let y = 0; y < ctx.height; y++) {
    for (let x = 0; x < ctx.width; x++) {
      const dx = x - cx;
      const dy = y - cy;
      const srcX = cx + dx * cos - dy * sin;
      const srcY = cy + dx * sin + dy * cos;
      const [r, g, b] = getPixel(prev, Math.floor(srcX), Math.floor(srcY));
      setPixel(out, x, y, r, g, b);
    }
  }

  return out;
}

export { rotate };
