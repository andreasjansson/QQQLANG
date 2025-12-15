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

function dither(ctx: FnContext, mode: string): Image {
  const prev = getPrevImage(ctx);
  const out = createSolidImage(ctx.width, ctx.height, "#000000");

  const modes = [
    "ordered-5level",
    "bayer-bw",
    "threshold-bw",
    "ordered-2bit",
    "floyd-rgb",
    "floyd-bw",
    "atkinson-4level",
    "atkinson-bw",
    "stucki-6level",
    "burkes",
    "sierra",
    "random-bw",
    "cluster-2bit",
    "bluenoise-bw",
    "bayer2x2-2bit",
    "noise-2bit",
  ];
  const ditherMode = modes.indexOf(mode);
  const seed = ctx.images.length * 137.5;

  const hash = (x: number, y: number) => {
    const v = Math.sin(x * 127.1 + y * 311.7 + seed * 113.3) * 43758.5453;
    return v - Math.floor(v);
  };

  const bayer8x8 = [
    [0, 32, 8, 40, 2, 34, 10, 42],
    [48, 16, 56, 24, 50, 18, 58, 26],
    [12, 44, 4, 36, 14, 46, 6, 38],
    [60, 28, 52, 20, 62, 30, 54, 22],
    [3, 35, 11, 43, 1, 33, 9, 41],
    [51, 19, 59, 27, 49, 17, 57, 25],
    [15, 47, 7, 39, 13, 45, 5, 37],
    [63, 31, 55, 23, 61, 29, 53, 21],
  ];

  const bayer2x2 = [
    [0, 2],
    [3, 1],
  ];

  if (ditherMode === 0) {
    const levels = 5;
    const spread = 85;

    for (let y = 0; y < ctx.height; y++) {
      for (let x = 0; x < ctx.width; x++) {
        const [r, g, b] = getPixel(prev, x, y);

        const tr = (bayer8x8[y % 8][x % 8] / 64.0 - 0.5) * spread;
        const tg = (bayer8x8[(y + 4) % 8][(x + 4) % 8] / 64.0 - 0.5) * spread;
        const tb = (bayer8x8[(y + 2) % 8][(x + 2) % 8] / 64.0 - 0.5) * spread;

        const quantize = (v: number, threshold: number) => {
          const adjusted = v + threshold;
          const step = 255 / (levels - 1);
          return Math.max(0, Math.min(255, Math.round(adjusted / step) * step));
        };

        setPixel(out, x, y, quantize(r, tr), quantize(g, tg), quantize(b, tb));
      }
    }
  } else if (ditherMode === 1) {
    for (let y = 0; y < ctx.height; y++) {
      for (let x = 0; x < ctx.width; x++) {
        const [r, g, b] = getPixel(prev, x, y);
        const gray = r * 0.299 + g * 0.587 + b * 0.114;
        const threshold = (bayer8x8[y % 8][x % 8] / 64.0) * 255;
        const val = gray > threshold ? 255 : 0;
        setPixel(out, x, y, val, val, val);
      }
    }
  } else if (ditherMode === 2) {
    for (let y = 0; y < ctx.height; y++) {
      for (let x = 0; x < ctx.width; x++) {
        const [r, g, b] = getPixel(prev, x, y);
        const gray = r * 0.299 + g * 0.587 + b * 0.114;
        const val = gray > 128 ? 255 : 0;
        setPixel(out, x, y, val, val, val);
      }
    }
  } else if (ditherMode === 3) {
    for (let y = 0; y < ctx.height; y++) {
      for (let x = 0; x < ctx.width; x++) {
        const [r, g, b] = getPixel(prev, x, y);
        const threshold = (bayer2x2[y % 2][x % 2] / 4.0) * 255;

        const qr = r > threshold ? 255 : 0;
        const qg = g > threshold ? 255 : 0;
        const qb = b > threshold ? 255 : 0;

        setPixel(out, x, y, qr, qg, qb);
      }
    }
  } else if (ditherMode === 4) {
    const errorR = new Float32Array(ctx.width * ctx.height);
    const errorG = new Float32Array(ctx.width * ctx.height);
    const errorB = new Float32Array(ctx.width * ctx.height);

    for (let y = 0; y < ctx.height; y++) {
      for (let x = 0; x < ctx.width; x++) {
        const idx = y * ctx.width + x;
        const [r, g, b] = getPixel(prev, x, y);

        const nr = Math.max(0, Math.min(255, r + errorR[idx]));
        const ng = Math.max(0, Math.min(255, g + errorG[idx]));
        const nb = Math.max(0, Math.min(255, b + errorB[idx]));

        const qr = nr > 128 ? 255 : 0;
        const qg = ng > 128 ? 255 : 0;
        const qb = nb > 128 ? 255 : 0;

        setPixel(out, x, y, qr, qg, qb);

        const er = nr - qr;
        const eg = ng - qg;
        const eb = nb - qb;

        if (x + 1 < ctx.width) {
          errorR[idx + 1] += (er * 7) / 16;
          errorG[idx + 1] += (eg * 7) / 16;
          errorB[idx + 1] += (eb * 7) / 16;
        }
        if (y + 1 < ctx.height) {
          if (x > 0) {
            errorR[idx + ctx.width - 1] += (er * 3) / 16;
            errorG[idx + ctx.width - 1] += (eg * 3) / 16;
            errorB[idx + ctx.width - 1] += (eb * 3) / 16;
          }
          errorR[idx + ctx.width] += (er * 5) / 16;
          errorG[idx + ctx.width] += (eg * 5) / 16;
          errorB[idx + ctx.width] += (eb * 5) / 16;
          if (x + 1 < ctx.width) {
            errorR[idx + ctx.width + 1] += (er * 1) / 16;
            errorG[idx + ctx.width + 1] += (eg * 1) / 16;
            errorB[idx + ctx.width + 1] += (eb * 1) / 16;
          }
        }
      }
    }
  } else if (ditherMode === 5) {
    const error = new Float32Array(ctx.width * ctx.height);

    for (let y = 0; y < ctx.height; y++) {
      for (let x = 0; x < ctx.width; x++) {
        const idx = y * ctx.width + x;
        const [r, g, b] = getPixel(prev, x, y);
        const gray = r * 0.299 + g * 0.587 + b * 0.114;

        const ng = Math.max(0, Math.min(255, gray + error[idx]));
        const qg = ng > 128 ? 255 : 0;

        setPixel(out, x, y, qg, qg, qg);

        const e = ng - qg;

        if (x + 1 < ctx.width) {
          error[idx + 1] += (e * 7) / 16;
        }
        if (y + 1 < ctx.height) {
          if (x > 0) {
            error[idx + ctx.width - 1] += (e * 3) / 16;
          }
          error[idx + ctx.width] += (e * 5) / 16;
          if (x + 1 < ctx.width) {
            error[idx + ctx.width + 1] += (e * 1) / 16;
          }
        }
      }
    }
  } else if (ditherMode === 6) {
    const errorR = new Float32Array(ctx.width * ctx.height);
    const errorG = new Float32Array(ctx.width * ctx.height);
    const errorB = new Float32Array(ctx.width * ctx.height);

    for (let y = 0; y < ctx.height; y++) {
      for (let x = 0; x < ctx.width; x++) {
        const idx = y * ctx.width + x;
        const [r, g, b] = getPixel(prev, x, y);

        const nr = Math.max(0, Math.min(255, r + errorR[idx]));
        const ng = Math.max(0, Math.min(255, g + errorG[idx]));
        const nb = Math.max(0, Math.min(255, b + errorB[idx]));

        const qr = Math.round(nr / 85) * 85;
        const qg = Math.round(ng / 85) * 85;
        const qb = Math.round(nb / 85) * 85;

        setPixel(out, x, y, qr, qg, qb);

        const er = nr - qr;
        const eg = ng - qg;
        const eb = nb - qb;

        if (x + 1 < ctx.width) {
          errorR[idx + 1] += (er * 1) / 8;
          errorG[idx + 1] += (eg * 1) / 8;
          errorB[idx + 1] += (eb * 1) / 8;
        }
        if (x + 2 < ctx.width) {
          errorR[idx + 2] += (er * 1) / 8;
          errorG[idx + 2] += (eg * 1) / 8;
          errorB[idx + 2] += (eb * 1) / 8;
        }
        if (y + 1 < ctx.height) {
          if (x > 0) {
            errorR[idx + ctx.width - 1] += (er * 1) / 8;
            errorG[idx + ctx.width - 1] += (eg * 1) / 8;
            errorB[idx + ctx.width - 1] += (eb * 1) / 8;
          }
          errorR[idx + ctx.width] += (er * 1) / 8;
          errorG[idx + ctx.width] += (eg * 1) / 8;
          errorB[idx + ctx.width] += (eb * 1) / 8;
          if (x + 1 < ctx.width) {
            errorR[idx + ctx.width + 1] += (er * 1) / 8;
            errorG[idx + ctx.width + 1] += (eg * 1) / 8;
            errorB[idx + ctx.width + 1] += (eb * 1) / 8;
          }
        }
        if (y + 2 < ctx.height) {
          errorR[idx + ctx.width * 2] += (er * 1) / 8;
          errorG[idx + ctx.width * 2] += (eg * 1) / 8;
          errorB[idx + ctx.width * 2] += (eb * 1) / 8;
        }
      }
    }
  } else if (ditherMode === 7) {
    const error = new Float32Array(ctx.width * ctx.height);

    for (let y = 0; y < ctx.height; y++) {
      for (let x = 0; x < ctx.width; x++) {
        const idx = y * ctx.width + x;
        const [r, g, b] = getPixel(prev, x, y);
        const gray = r * 0.299 + g * 0.587 + b * 0.114;

        const ng = Math.max(0, Math.min(255, gray + error[idx]));
        const qg = ng > 128 ? 255 : 0;

        setPixel(out, x, y, qg, qg, qg);

        const e = ng - qg;

        if (x + 1 < ctx.width) {
          error[idx + 1] += (e * 1) / 8;
        }
        if (x + 2 < ctx.width) {
          error[idx + 2] += (e * 1) / 8;
        }
        if (y + 1 < ctx.height) {
          if (x > 0) {
            error[idx + ctx.width - 1] += (e * 1) / 8;
          }
          error[idx + ctx.width] += (e * 1) / 8;
          if (x + 1 < ctx.width) {
            error[idx + ctx.width + 1] += (e * 1) / 8;
          }
        }
        if (y + 2 < ctx.height) {
          error[idx + ctx.width * 2] += (e * 1) / 8;
        }
      }
    }
  } else if (ditherMode === 8) {
    const errorR = new Float32Array(ctx.width * ctx.height);
    const errorG = new Float32Array(ctx.width * ctx.height);
    const errorB = new Float32Array(ctx.width * ctx.height);

    for (let y = 0; y < ctx.height; y++) {
      for (let x = 0; x < ctx.width; x++) {
        const idx = y * ctx.width + x;
        const [r, g, b] = getPixel(prev, x, y);

        const nr = Math.max(0, Math.min(255, r + errorR[idx]));
        const ng = Math.max(0, Math.min(255, g + errorG[idx]));
        const nb = Math.max(0, Math.min(255, b + errorB[idx]));

        const qr = Math.round(nr / 51) * 51;
        const qg = Math.round(ng / 51) * 51;
        const qb = Math.round(nb / 51) * 51;

        setPixel(out, x, y, qr, qg, qb);

        const er = nr - qr;
        const eg = ng - qg;
        const eb = nb - qb;

        const distribute = (dx: number, dy: number, factor: number) => {
          if (
            x + dx >= 0 &&
            x + dx < ctx.width &&
            y + dy >= 0 &&
            y + dy < ctx.height
          ) {
            const tidx = idx + dy * ctx.width + dx;
            errorR[tidx] += er * factor;
            errorG[tidx] += eg * factor;
            errorB[tidx] += eb * factor;
          }
        };

        distribute(1, 0, 8 / 42);
        distribute(2, 0, 4 / 42);
        distribute(-2, 1, 2 / 42);
        distribute(-1, 1, 4 / 42);
        distribute(0, 1, 8 / 42);
        distribute(1, 1, 4 / 42);
        distribute(2, 1, 2 / 42);
        distribute(-2, 2, 1 / 42);
        distribute(-1, 2, 2 / 42);
        distribute(0, 2, 4 / 42);
        distribute(1, 2, 2 / 42);
        distribute(2, 2, 1 / 42);
      }
    }
  } else if (ditherMode === 9) {
    const errorR = new Float32Array(ctx.width * ctx.height);
    const errorG = new Float32Array(ctx.width * ctx.height);
    const errorB = new Float32Array(ctx.width * ctx.height);

    for (let y = 0; y < ctx.height; y++) {
      for (let x = 0; x < ctx.width; x++) {
        const idx = y * ctx.width + x;
        const [r, g, b] = getPixel(prev, x, y);

        const nr = Math.max(0, Math.min(255, r + errorR[idx]));
        const ng = Math.max(0, Math.min(255, g + errorG[idx]));
        const nb = Math.max(0, Math.min(255, b + errorB[idx]));

        const qr = Math.round(nr / 51) * 51;
        const qg = Math.round(ng / 51) * 51;
        const qb = Math.round(nb / 51) * 51;

        setPixel(out, x, y, qr, qg, qb);

        const er = nr - qr;
        const eg = ng - qg;
        const eb = nb - qb;

        const distribute = (dx: number, dy: number, factor: number) => {
          if (
            x + dx >= 0 &&
            x + dx < ctx.width &&
            y + dy >= 0 &&
            y + dy < ctx.height
          ) {
            const tidx = idx + dy * ctx.width + dx;
            errorR[tidx] += er * factor;
            errorG[tidx] += eg * factor;
            errorB[tidx] += eb * factor;
          }
        };

        distribute(1, 0, 8 / 32);
        distribute(2, 0, 4 / 32);
        distribute(-2, 1, 2 / 32);
        distribute(-1, 1, 4 / 32);
        distribute(0, 1, 8 / 32);
        distribute(1, 1, 4 / 32);
        distribute(2, 1, 2 / 32);
      }
    }
  } else if (ditherMode === 10) {
    const errorR = new Float32Array(ctx.width * ctx.height);
    const errorG = new Float32Array(ctx.width * ctx.height);
    const errorB = new Float32Array(ctx.width * ctx.height);

    for (let y = 0; y < ctx.height; y++) {
      for (let x = 0; x < ctx.width; x++) {
        const idx = y * ctx.width + x;
        const [r, g, b] = getPixel(prev, x, y);

        const nr = Math.max(0, Math.min(255, r + errorR[idx]));
        const ng = Math.max(0, Math.min(255, g + errorG[idx]));
        const nb = Math.max(0, Math.min(255, b + errorB[idx]));

        const qr = Math.round(nr / 51) * 51;
        const qg = Math.round(ng / 51) * 51;
        const qb = Math.round(nb / 51) * 51;

        setPixel(out, x, y, qr, qg, qb);

        const er = nr - qr;
        const eg = ng - qg;
        const eb = nb - qb;

        const distribute = (dx: number, dy: number, factor: number) => {
          if (
            x + dx >= 0 &&
            x + dx < ctx.width &&
            y + dy >= 0 &&
            y + dy < ctx.height
          ) {
            const tidx = idx + dy * ctx.width + dx;
            errorR[tidx] += er * factor;
            errorG[tidx] += eg * factor;
            errorB[tidx] += eb * factor;
          }
        };

        distribute(1, 0, 5 / 32);
        distribute(2, 0, 3 / 32);
        distribute(-2, 1, 2 / 32);
        distribute(-1, 1, 4 / 32);
        distribute(0, 1, 5 / 32);
        distribute(1, 1, 4 / 32);
        distribute(2, 1, 2 / 32);
        distribute(-1, 2, 2 / 32);
        distribute(0, 2, 3 / 32);
        distribute(1, 2, 2 / 32);
      }
    }
  } else if (ditherMode === 11) {
    for (let y = 0; y < ctx.height; y++) {
      for (let x = 0; x < ctx.width; x++) {
        const [r, g, b] = getPixel(prev, x, y);
        const gray = r * 0.299 + g * 0.587 + b * 0.114;
        const noise = (hash(x, y) - 0.5) * 100;
        const val = gray + noise > 128 ? 255 : 0;
        setPixel(out, x, y, val, val, val);
      }
    }
  } else if (ditherMode === 12) {
    const cluster = [
      [12, 5, 6, 13],
      [4, 0, 1, 7],
      [11, 3, 2, 8],
      [15, 10, 9, 14],
    ];

    for (let y = 0; y < ctx.height; y++) {
      for (let x = 0; x < ctx.width; x++) {
        const [r, g, b] = getPixel(prev, x, y);
        const threshold = (cluster[y % 4][x % 4] / 16.0) * 255;

        const qr = r > threshold ? 255 : 0;
        const qg = g > threshold ? 255 : 0;
        const qb = b > threshold ? 255 : 0;

        setPixel(out, x, y, qr, qg, qb);
      }
    }
  } else if (ditherMode === 13) {
    const blueNoise = (x: number, y: number): number => {
      const v1 = hash(x * 0.7 + seed, y * 0.7 + seed);
      const v2 = hash(x * 1.3 + seed + 100, y * 1.3 + seed + 100);
      const v3 = hash(x * 2.1 + seed + 200, y * 2.1 + seed + 200);
      return (v1 + v2 + v3) / 3;
    };

    for (let y = 0; y < ctx.height; y++) {
      for (let x = 0; x < ctx.width; x++) {
        const [r, g, b] = getPixel(prev, x, y);
        const gray = r * 0.299 + g * 0.587 + b * 0.114;
        const threshold = blueNoise(x, y) * 255;
        const val = gray > threshold ? 255 : 0;
        setPixel(out, x, y, val, val, val);
      }
    }
  } else if (ditherMode === 14) {
    for (let y = 0; y < ctx.height; y++) {
      for (let x = 0; x < ctx.width; x++) {
        const [r, g, b] = getPixel(prev, x, y);
        const threshold = (bayer2x2[y % 2][x % 2] / 4.0) * 255;

        const qr = r > threshold ? 255 : 0;
        const qg = g > threshold ? 255 : 0;
        const qb = b > threshold ? 255 : 0;

        setPixel(out, x, y, qr, qg, qb);
      }
    }
  } else if (ditherMode === 15) {
    for (let y = 0; y < ctx.height; y++) {
      for (let x = 0; x < ctx.width; x++) {
        const [r, g, b] = getPixel(prev, x, y);
        const noise = (hash(x, y) - 0.5) * 180;

        const qr = r + noise > 128 ? 255 : 0;
        const qg = g + noise > 128 ? 255 : 0;
        const qb = b + noise > 128 ? 255 : 0;

        setPixel(out, x, y, qr, qg, qb);
      }
    }
  }

  return out;
}

export { dither };
