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
import FFT from "fft.js";

function fftOverflow(ctx: FnContext, n: number): Image {
  const prev = getPrevImage(ctx);

  const multiplier = 1.5 + n * 0.8;

  const nextPow2 = (v: number) => {
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    return v + 1;
  };

  const fftW = nextPow2(ctx.width);
  const fftH = nextPow2(ctx.height);

  const fftRow = new FFT(fftW);
  const fftCol = new FFT(fftH);

  const processChannel = (
    channel: Float32Array,
    mult: number,
    phaseShift: number,
  ): Float32Array => {
    const data = new Float64Array(fftW * fftH * 2);

    for (let y = 0; y < ctx.height; y++) {
      for (let x = 0; x < ctx.width; x++) {
        data[(y * fftW + x) * 2] = channel[y * ctx.width + x];
      }
    }

    const rowIn = fftRow.createComplexArray();
    const rowOut = fftRow.createComplexArray();
    for (let y = 0; y < fftH; y++) {
      for (let x = 0; x < fftW; x++) {
        rowIn[x * 2] = data[(y * fftW + x) * 2];
        rowIn[x * 2 + 1] = data[(y * fftW + x) * 2 + 1];
      }
      fftRow.transform(rowOut, rowIn);
      for (let x = 0; x < fftW; x++) {
        data[(y * fftW + x) * 2] = rowOut[x * 2];
        data[(y * fftW + x) * 2 + 1] = rowOut[x * 2 + 1];
      }
    }

    const colIn = fftCol.createComplexArray();
    const colOut = fftCol.createComplexArray();
    for (let x = 0; x < fftW; x++) {
      for (let y = 0; y < fftH; y++) {
        colIn[y * 2] = data[(y * fftW + x) * 2];
        colIn[y * 2 + 1] = data[(y * fftW + x) * 2 + 1];
      }
      fftCol.transform(colOut, colIn);
      for (let y = 0; y < fftH; y++) {
        data[(y * fftW + x) * 2] = colOut[y * 2];
        data[(y * fftW + x) * 2 + 1] = colOut[y * 2 + 1];
      }
    }

    const cx = fftW / 2;
    const cy = fftH / 2;
    const maxFreqDist = Math.sqrt(cx * cx + cy * cy);
    const wrapLimit = (255 * fftW * fftH) / 4;

    for (let y = 0; y < fftH; y++) {
      for (let x = 0; x < fftW; x++) {
        const i = (y * fftW + x) * 2;
        const re = data[i];
        const im = data[i + 1];
        const mag = Math.sqrt(re * re + im * im);
        let phase = Math.atan2(im, re);

        const dx = x < cx ? x : x - fftW;
        const dy = y < cy ? y : y - fftH;
        const freqDist = Math.sqrt(dx * dx + dy * dy) / maxFreqDist;

        let newMag = mag * mult;
        while (newMag > wrapLimit) {
          newMag = Math.abs(newMag - wrapLimit * 2);
        }

        phase += phaseShift * freqDist;

        data[i] = newMag * Math.cos(phase);
        data[i + 1] = newMag * Math.sin(phase);
      }
    }

    for (let x = 0; x < fftW; x++) {
      for (let y = 0; y < fftH; y++) {
        colIn[y * 2] = data[(y * fftW + x) * 2];
        colIn[y * 2 + 1] = data[(y * fftW + x) * 2 + 1];
      }
      fftCol.inverseTransform(colOut, colIn);
      for (let y = 0; y < fftH; y++) {
        data[(y * fftW + x) * 2] = colOut[y * 2];
        data[(y * fftW + x) * 2 + 1] = colOut[y * 2 + 1];
      }
    }

    for (let y = 0; y < fftH; y++) {
      for (let x = 0; x < fftW; x++) {
        rowIn[x * 2] = data[(y * fftW + x) * 2];
        rowIn[x * 2 + 1] = data[(y * fftW + x) * 2 + 1];
      }
      fftRow.inverseTransform(rowOut, rowIn);
      for (let x = 0; x < fftW; x++) {
        data[(y * fftW + x) * 2] = rowOut[x * 2];
      }
    }

    const result = new Float32Array(ctx.width * ctx.height);
    for (let y = 0; y < ctx.height; y++) {
      for (let x = 0; x < ctx.width; x++) {
        result[y * ctx.width + x] = data[(y * fftW + x) * 2];
      }
    }

    return result;
  };

  const rIn = new Float32Array(ctx.width * ctx.height);
  const gIn = new Float32Array(ctx.width * ctx.height);
  const bIn = new Float32Array(ctx.width * ctx.height);

  for (let y = 0; y < ctx.height; y++) {
    for (let x = 0; x < ctx.width; x++) {
      const [r, g, b] = getPixel(prev, x, y);
      const idx = y * ctx.width + x;
      rIn[idx] = r;
      gIn[idx] = g;
      bIn[idx] = b;
    }
  }

  const rOut = processChannel(rIn, multiplier, 0);
  const gOut = processChannel(gIn, multiplier * 1.1, Math.PI * 0.1);
  const bOut = processChannel(bIn, multiplier * 0.9, -Math.PI * 0.1);

  const out = createSolidImage(ctx.width, ctx.height, "#000000");
  for (let y = 0; y < ctx.height; y++) {
    for (let x = 0; x < ctx.width; x++) {
      const idx = y * ctx.width + x;
      let r = rOut[idx];
      let g = gOut[idx];
      let b = bOut[idx];

      r = ((r % 256) + 256) % 256;
      g = ((g % 256) + 256) % 256;
      b = ((b % 256) + 256) % 256;

      setPixel(out, x, y, Math.round(r), Math.round(g), Math.round(b));
    }
  }

  return out;
}

export { fftOverflow };
