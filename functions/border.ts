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

function border(ctx: FnContext, style: string, c: string): Image {
  const prev = getPrevImage(ctx);
  const out = cloneImage(prev);
  const [tr, tg, tb] = hexToRgb(c);

  const cx = ctx.width / 2;
  const cy = ctx.height / 2;

  const isSolid = style.endsWith("-solid");

  const computeDistance = (x: number, y: number): number => {
    const nx = (x - cx) / cx;
    const ny = (y - cy) / cy;

    if (style.startsWith("circular")) {
      return Math.sqrt(nx * nx + ny * ny);
    } else if (style.startsWith("horizontal")) {
      return Math.abs(ny);
    } else if (style.startsWith("vertical")) {
      return Math.abs(nx);
    } else if (style.startsWith("rectangular")) {
      return Math.max(Math.abs(nx), Math.abs(ny));
    } else if (style.startsWith("diamond")) {
      return (Math.abs(nx) + Math.abs(ny)) * 0.7;
    } else if (style.startsWith("hexagon")) {
      const ax = Math.abs(nx);
      const ay = Math.abs(ny);
      const hexDist = Math.max(ax, ax * 0.5 + ay * 0.866) * 0.8;
      const topBottomBorder = Math.abs(ny) > 0.85 ? 10 : 0;
      return Math.max(hexDist, topBottomBorder);
    } else if (style.startsWith("sine-horizontal")) {
      return Math.abs(ny - 0.3 * Math.sin(nx * Math.PI * 3));
    } else if (style.startsWith("sine-vertical")) {
      return Math.abs(nx - 0.3 * Math.sin(ny * Math.PI * 3));
    } else if (style.startsWith("triangle-horizontal")) {
      const t = nx * 1.5;
      const triWave = 0.3 * (2 * Math.abs(2 * (t - Math.floor(t + 0.5))) - 1);
      return Math.abs(ny - triWave);
    } else if (style.startsWith("triangle-vertical")) {
      const t = ny * 1.5;
      const triWave = 0.3 * (2 * Math.abs(2 * (t - Math.floor(t + 0.5))) - 1);
      return Math.abs(nx - triWave);
    }
    return Math.sqrt(nx * nx + ny * ny);
  };

  for (let y = 0; y < ctx.height; y++) {
    for (let x = 0; x < ctx.width; x++) {
      const dist = computeDistance(x, y);

      let vignette: number;
      if (isSolid) {
        vignette = dist < 0.7 ? 1 : 0;
      } else {
        vignette = Math.max(0, 1 - Math.pow(dist / 0.7, 2));
      }

      const idx = (y * ctx.width + x) * 4;
      out.data[idx] = Math.round(
        out.data[idx] * vignette + tr * (1 - vignette),
      );
      out.data[idx + 1] = Math.round(
        out.data[idx + 1] * vignette + tg * (1 - vignette),
      );
      out.data[idx + 2] = Math.round(
        out.data[idx + 2] * vignette + tb * (1 - vignette),
      );
    }
  }

  return out;
}

export { border };
