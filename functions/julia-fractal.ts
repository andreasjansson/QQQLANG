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

function juliaFractal(ctx: FnContext, n: number): Image {
  const prev = getPrevImage(ctx);
  const out = createSolidImage(ctx.width, ctx.height, "#000000");

  const zoomPower = Math.max(1, n);
  const scale = Math.pow(0.7, zoomPower);
  const maxIterations = 10 + zoomPower * 5;

  const cReal = -0.7;
  const cImag = 0.27015;

  const centerReal = 0.15;
  const centerImag = 0.6;

  const aspect = ctx.height / ctx.width;

  for (let py = 0; py < ctx.height; py++) {
    for (let px = 0; px < ctx.width; px++) {
      let zReal = centerReal + (px / ctx.width - 0.5) * 3 * scale;
      let zImag = centerImag + (py / ctx.height - 0.5) * 3 * scale * aspect;

      let iteration = 0;
      let zReal2 = zReal * zReal;
      let zImag2 = zImag * zImag;

      while (iteration < maxIterations && zReal2 + zImag2 < 256) {
        zImag = 2 * zReal * zImag + cImag;
        zReal = zReal2 - zImag2 + cReal;
        zReal2 = zReal * zReal;
        zImag2 = zImag * zImag;
        iteration++;
      }

      if (iteration === maxIterations) {
        const [pr, pg, pb] = getPixel(prev, px, py);
        setPixel(out, px, py, pr, pg, pb);
      } else {
        const log_zn = Math.log(zReal2 + zImag2) / 2;
        const nu = Math.log(log_zn / Math.log(2)) / Math.log(2);
        const smoothed = iteration + 1 - nu;

        const t = (smoothed % 20) / 20;
        const sampleX = Math.floor(t * ctx.width);
        const sampleY = py;

        const [pr, pg, pb] = getPixel(prev, sampleX, sampleY);
        const brightness = 0.85 + 0.3 * t;
        setPixel(
          out,
          px,
          py,
          Math.min(255, Math.round(pr * brightness)),
          Math.min(255, Math.round(pg * brightness)),
          Math.min(255, Math.round(pb * brightness)),
        );
      }
    }
  }

  return out;
}

export { juliaFractal };
