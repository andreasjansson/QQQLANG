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

function cond(
  ctx: FnContext,
  condImg: Image,
  trueImg: Image,
  falseImg: Image,
  channel: string,
  thresholdN: number,
): Image {
  const out = createSolidImage(ctx.width, ctx.height, "#000000");

  const normalizedThreshold = (thresholdN - 1) / 67;

  for (let y = 0; y < ctx.height; y++) {
    for (let x = 0; x < ctx.width; x++) {
      const [cr, cg, cb] = getPixel(condImg, x, y);

      let value: number;

      switch (channel) {
        case "hue": {
          const [h] = rgbToHsl(cr, cg, cb);
          value = h / 360;
          break;
        }
        case "saturation": {
          const [, s] = rgbToHsl(cr, cg, cb);
          value = s;
          break;
        }
        case "lightness": {
          const [, , l] = rgbToHsl(cr, cg, cb);
          value = l;
          break;
        }
        case "red":
          value = cr / 255;
          break;
        case "green":
          value = cg / 255;
          break;
        case "blue":
          value = cb / 255;
          break;
        default:
          value = (cr * 0.299 + cg * 0.587 + cb * 0.114) / 255;
      }

      const sourceImg = value >= normalizedThreshold ? trueImg : falseImg;
      const [r, g, b] = getPixel(sourceImg, x, y);
      setPixel(out, x, y, r, g, b);
    }
  }

  return out;
}

export { cond };
