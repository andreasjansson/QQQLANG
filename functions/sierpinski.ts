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

function sierpinski(ctx: FnContext, old: Image, size: number): Image {
  const prev = getPrevImage(ctx);
  const out = createSolidImage(ctx.width, ctx.height, "#000000");

  const popcount = (n: number): number => {
    let count = 0;
    while (n) {
      count += n & 1;
      n >>= 1;
    }
    return count;
  };

  const resolution = Math.pow(2, Math.floor((size - 1) / 10) + 2);

  const v0x = ctx.width / 2,
    v0y = 0;
  const v1x = 0,
    v1y = ctx.height;
  const v2x = ctx.width,
    v2y = ctx.height;

  const denom = (v1y - v2y) * (v0x - v2x) + (v2x - v1x) * (v0y - v2y);

  for (let y = 0; y < ctx.height; y++) {
    for (let x = 0; x < ctx.width; x++) {
      const baryA = ((v1y - v2y) * (x - v2x) + (v2x - v1x) * (y - v2y)) / denom;
      const baryB = ((v2y - v0y) * (x - v2x) + (v0x - v2x) * (y - v2y)) / denom;
      const baryC = 1 - baryA - baryB;

      const [pr, pg, pb] = getPixel(prev, x, y);

      if (baryA < 0 || baryB < 0 || baryC < 0) {
        setPixel(out, x, y, pr, pg, pb);
        continue;
      }

      const ai = Math.floor(baryA * resolution);
      const bi = Math.floor(baryB * resolution);
      const ci = Math.floor(baryC * resolution);

      const overlap = (ai & bi) | (bi & ci) | (ai & ci);
      const level = popcount(overlap) % 6;

      const [or, og, ob] = getPixel(old, x, y);

      let r: number, g: number, b: number;

      switch (level) {
        case 0: {
          r = or;
          g = og;
          b = ob;
          break;
        }
        case 1: {
          r = 255 - pr;
          g = 255 - pg;
          b = 255 - pb;
          break;
        }
        case 2: {
          const [h, s, l] = rgbToHsl(pr, pg, pb);
          [r, g, b] = hslToRgb((h + 120) % 360, s, l);
          break;
        }
        case 3: {
          const [h, s, l] = rgbToHsl(pr, pg, pb);
          [r, g, b] = hslToRgb((h + 240) % 360, s, l);
          break;
        }
        case 4: {
          const gray = Math.round(pr * 0.299 + pg * 0.587 + pb * 0.114);
          const contrast = gray < 128 ? gray * 0.5 : 128 + (gray - 128) * 1.5;
          r = g = b = Math.max(0, Math.min(255, Math.round(contrast)));
          break;
        }
        case 5: {
          const [h, s, l] = rgbToHsl(pr, pg, pb);
          [r, g, b] = hslToRgb((h + 60) % 360, Math.min(1, s * 1.5), l);
          break;
        }
        default:
          r = pr;
          g = pg;
          b = pb;
      }

      setPixel(out, x, y, r, g, b);
    }
  }

  return out;
}

export { sierpinski };
