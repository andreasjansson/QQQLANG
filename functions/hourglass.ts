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

function hourglass(ctx: FnContext): Image {
  const prev = getPrevImage(ctx);
  const out = createSolidImage(ctx.width, ctx.height, "#000000");

  const cx = ctx.width / 2;
  const cy = ctx.height / 2;

  for (let y = 0; y < ctx.height; y++) {
    for (let x = 0; x < ctx.width; x++) {
      const nx = (x - cx) / cx;
      const ny = (y - cy) / cy;

      const hourglassWidth = Math.abs(ny);
      const inHourglass = Math.abs(nx) < hourglassWidth;

      const [pr, pg, pb] = getPixel(prev, x, y);

      const distFromCenter = Math.sqrt(nx * nx + ny * ny);
      const angle = Math.atan2(ny, nx);

      if (inHourglass) {
        const gr = Math.floor((1 - Math.abs(ny)) * 255);
        const gg = Math.floor(
          (Math.abs(nx) / Math.max(0.01, hourglassWidth)) * 255,
        );
        const gb = Math.floor(distFromCenter * 180);

        const nr = 255 - (pr & gr);
        const ng = 255 - (pg & gg);
        const nb = 255 - (pb & gb);

        setPixel(out, x, y, nr, ng, nb);
      } else {
        const gr = Math.floor(((angle + Math.PI) / (Math.PI * 2)) * 255);
        const gg = Math.floor((1 - distFromCenter) * 200 + 55);
        const gb = Math.floor(((nx + 1) / 2) * 255);

        const nr = (pr + gr) % 256;
        const ng = Math.abs(pg - gg);
        const nb = pb ^ gb;

        setPixel(out, x, y, nr, ng, nb);
      }
    }
  }

  return out;
}

export { hourglass };
