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

function moire(ctx: FnContext, n: number): Image {
  const prev = getPrevImage(ctx);
  const out = createSolidImage(ctx.width, ctx.height, "#000000");

  const hash = (i: number) => {
    const x = Math.sin(i * 127.1 + n * 311.7) * 43758.5453;
    return x - Math.floor(x);
  };

  const scale = 12 + hash(0) * 30;
  const freqRatio = 1.02 + hash(1) * 0.1;
  const amp1 = 15 + hash(2) * 80;
  const amp2 = 15 + hash(3) * 80;
  const freq1 = 0.006 + hash(4) * 0.025;
  const angle1 = hash(5) * Math.PI;
  const angle2 = angle1 + (hash(6) - 0.5) * 0.3;
  const phase = hash(7) * Math.PI * 2;
  const hueShift = Math.floor(hash(8) * 360);
  const harmonic1 = 0.3 + hash(9) * 0.7;
  const harmonic2 = hash(10) * 0.5;
  const scaleRatio = 0.95 + hash(11) * 0.1;
  const crossAmp = hash(12) * 40;
  const crossFreq = 0.01 + hash(13) * 0.02;

  const cos_a1 = Math.cos(angle1),
    sin_a1 = Math.sin(angle1);
  const cos_a2 = Math.cos(angle2),
    sin_a2 = Math.sin(angle2);

  for (let y = 0; y < ctx.height; y++) {
    for (let x = 0; x < ctx.width; x++) {
      const rx1 = x * cos_a1 + y * sin_a1;
      const ry1 = -x * sin_a1 + y * cos_a1;
      const rx2 = x * cos_a2 + y * sin_a2;
      const ry2 = -x * sin_a2 + y * cos_a2;

      const wobble1 =
        Math.sin(rx1 * freq1) * amp1 +
        Math.sin(rx1 * freq1 * 2.1) * amp1 * harmonic1 +
        Math.sin(ry1 * crossFreq) * crossAmp;
      const wobble2 =
        Math.sin(rx2 * freq1 * freqRatio + phase) * amp2 +
        Math.sin(rx2 * freq1 * freqRatio * 2.3 + phase) * amp2 * harmonic2 +
        Math.sin(ry2 * crossFreq * 1.1) * crossAmp;

      const wave1 = ry1 + wobble1;
      const wave2 = ry2 + wobble2;

      const line1 = Math.floor(wave1 / scale) % 2;
      const line2 = Math.floor(wave2 / (scale * scaleRatio)) % 2;

      const moire = line1 !== line2;

      const [r, g, b] = getPixel(prev, x, y);

      if (moire) {
        setPixel(out, x, y, 255 - r, 255 - g, 255 - b);
      } else {
        setPixel(out, x, y, r, g, b);
      }
    }
  }

  return out;
}

export { moire };
