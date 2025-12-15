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

function swirl(ctx: FnContext, n: number): Image {
  const prev = getPrevImage(ctx);
  const out = createSolidImage(ctx.width, ctx.height, "#000000");

  const maxRotation = n * 20 * (Math.PI / 180);
  const cx = ctx.width / 2;
  const cy = ctx.height / 2;
  const maxR = Math.sqrt(cx * cx + cy * cy);

  for (let y = 0; y < ctx.height; y++) {
    for (let x = 0; x < ctx.width; x++) {
      const dx = x - cx;
      const dy = y - cy;
      const dist = Math.sqrt(dx * dx + dy * dy);
      const normDist = dist / maxR;

      const falloff = (1 - normDist) * (1 - normDist);
      const rotation = maxRotation * falloff;

      const cos = Math.cos(rotation);
      const sin = Math.sin(rotation);
      const sx = cx + dx * cos - dy * sin;
      const sy = cy + dx * sin + dy * cos;

      const [r, g, b] = getPixel(prev, Math.floor(sx), Math.floor(sy));
      setPixel(out, x, y, r, g, b);
    }
  }

  return out;
}

export { swirl };
