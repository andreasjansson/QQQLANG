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
import * as tf from "@tensorflow/tfjs";

async function cppn(ctx: FnContext, strengthN: number): Promise<Image> {
  const prev = getPrevImage(ctx);
  const { width, height } = ctx;

  const strength = (strengthN - 1) / 67;
  const weightScale = 0.5 + strength * 1.5;

  function seededNormal(seed: number): () => number {
    let hasSpare = false;
    let spare = 0;
    return () => {
      if (hasSpare) {
        hasSpare = false;
        return spare;
      }
      seed = (seed * 1103515245 + 12345) & 0x7fffffff;
      const u1 = seed / 0x7fffffff;
      seed = (seed * 1103515245 + 12345) & 0x7fffffff;
      const u2 = seed / 0x7fffffff;
      const mag = Math.sqrt(-2.0 * Math.log(u1 + 1e-10));
      spare = mag * Math.sin(2.0 * Math.PI * u2);
      hasSpare = true;
      return mag * Math.cos(2.0 * Math.PI * u2);
    };
  }

  function createWeightTensor(
    rows: number,
    cols: number,
    rng: () => number,
    scale: number,
  ): tf.Tensor2D {
    const data = new Float32Array(rows * cols);
    for (let i = 0; i < data.length; i++) data[i] = rng() * scale;
    return tf.tensor2d(data, [rows, cols]);
  }

  function createBiasTensor(
    size: number,
    rng: () => number,
    scale: number,
  ): tf.Tensor2D {
    const data = new Float32Array(size);
    for (let i = 0; i < size; i++) data[i] = rng() * scale;
    return tf.tensor2d(data, [1, size]);
  }

  const rng = seededNormal(ctx.images.length);
  const coordScale = 8.0 + strength * 8.0;
  const netSize = 9;
  const zDim = 8;
  const outDim = 4; // dx, dy displacement + saturation, value modulation
  const displacementStrength = 0.1 + strength * 0.2;

  // Derive z from input image for determinism
  let sumR = 0,
    sumG = 0,
    sumB = 0;
  const sampleStep = Math.max(1, Math.floor(Math.sqrt((width * height) / 100)));
  let sampleCount = 0;
  for (let sy = 0; sy < height; sy += sampleStep) {
    for (let sx = 0; sx < width; sx += sampleStep) {
      const idx = (sy * width + sx) * 4;
      sumR += prev.data[idx];
      sumG += prev.data[idx + 1];
      sumB += prev.data[idx + 2];
      sampleCount++;
    }
  }
  const avgR = sumR / sampleCount / 255;
  const avgG = sumG / sampleCount / 255;
  const avgB = sumB / sampleCount / 255;

  const zRng = seededNormal(Math.floor((avgR + avgG + avgB) * 10000) + 12345);
  const zData = new Float32Array(zDim);
  for (let i = 0; i < zDim; i++) zData[i] = zRng() * 0.5;

  const W_z = createWeightTensor(zDim, netSize, rng, weightScale);
  const B_z = createBiasTensor(netSize, rng, weightScale);
  const W_x = createWeightTensor(1, netSize, rng, weightScale);
  const W_y = createWeightTensor(1, netSize, rng, weightScale);
  const W_r = createWeightTensor(1, netSize, rng, weightScale);

  const W_h0 = createWeightTensor(netSize, netSize, rng, weightScale);
  const B_h0 = createBiasTensor(netSize, rng, weightScale);
  const W_h1 = createWeightTensor(netSize, netSize, rng, weightScale);
  const B_h1 = createBiasTensor(netSize, rng, weightScale);
  const W_h2 = createWeightTensor(netSize, netSize, rng, weightScale);
  const B_h2 = createBiasTensor(netSize, rng, weightScale);
  const W_h3 = createWeightTensor(netSize, netSize, rng, weightScale);
  const B_h3 = createBiasTensor(netSize, rng, weightScale);
  const W_h4 = createWeightTensor(netSize, netSize, rng, weightScale);
  const B_h4 = createBiasTensor(netSize, rng, weightScale);

  const W_out = createWeightTensor(netSize, outDim, rng, weightScale);
  const B_out = createBiasTensor(outDim, rng, weightScale);

  const outputTensor = tf.tidy(() => {
    const nPoints = width * height;

    const xMat: number[] = [];
    const yMat: number[] = [];
    const rMat: number[] = [];

    for (let py = 0; py < height; py++) {
      const yVal =
        (coordScale * (py - (height - 1) / 2.0)) / (height - 1) / 0.5;
      for (let px = 0; px < width; px++) {
        const xVal =
          (coordScale * (px - (width - 1) / 2.0)) / (width - 1) / 0.5;
        xMat.push(xVal);
        yMat.push(yVal);
        rMat.push(Math.sqrt(xVal * xVal + yVal * yVal));
      }
    }

    const xTensor = tf.tensor2d(xMat, [nPoints, 1]);
    const yTensor = tf.tensor2d(yMat, [nPoints, 1]);
    const rTensor = tf.tensor2d(rMat, [nPoints, 1]);

    const zScaled = tf.tensor2d(zData, [1, zDim]).mul(coordScale);
    const zBroadcast = tf.tile(zScaled, [nPoints, 1]);

    const Uz = tf.add(tf.matMul(zBroadcast, W_z), B_z);
    const Ux = tf.matMul(xTensor, W_x);
    const Uy = tf.matMul(yTensor, W_y);
    const Ur = tf.matMul(rTensor, W_r);
    const U = tf.add(tf.add(Uz, Ux), tf.add(Uy, Ur));

    let H = tf.tanh(U) as tf.Tensor2D;
    H = tf.tanh(tf.add(tf.matMul(H, W_h0), B_h0)) as tf.Tensor2D;
    H = tf.tanh(tf.add(tf.matMul(H, W_h1), B_h1)) as tf.Tensor2D;
    H = tf.tanh(tf.add(tf.matMul(H, W_h2), B_h2)) as tf.Tensor2D;
    H = tf.tanh(tf.add(tf.matMul(H, W_h3), B_h3)) as tf.Tensor2D;
    H = tf.tanh(tf.add(tf.matMul(H, W_h4), B_h4)) as tf.Tensor2D;

    const output = tf.tanh(tf.add(tf.matMul(H, W_out), B_out));

    return output;
  });

  const cpnnData = await outputTensor.data();
  outputTensor.dispose();

  W_z.dispose();
  B_z.dispose();
  W_x.dispose();
  W_y.dispose();
  W_r.dispose();
  W_h0.dispose();
  B_h0.dispose();
  W_h1.dispose();
  B_h1.dispose();
  W_h2.dispose();
  B_h2.dispose();
  W_h3.dispose();
  B_h3.dispose();
  W_h4.dispose();
  B_h4.dispose();
  W_out.dispose();
  B_out.dispose();

  const out = createSolidImage(width, height, "#000000");
  for (let py = 0; py < height; py++) {
    for (let px = 0; px < width; px++) {
      const i = py * width + px;
      const dx = cpnnData[i * outDim] * displacementStrength * width;
      const dy = cpnnData[i * outDim + 1] * displacementStrength * height;

      // CPPN outputs in [-1, 1], map to modulation factor [0.95, 1.05]
      const sMod = cpnnData[i * outDim + 2] * 0.05 + 1.0;
      const vMod = cpnnData[i * outDim + 3] * 0.05 + 1.0;

      const srcX = Math.max(0, Math.min(width - 1, Math.round(px + dx)));
      const srcY = Math.max(0, Math.min(height - 1, Math.round(py + dy)));
      const srcIdx = (srcY * width + srcX) * 4;

      const r = prev.data[srcIdx];
      const g = prev.data[srcIdx + 1];
      const b = prev.data[srcIdx + 2];

      // Convert to HSV
      const max = Math.max(r, g, b);
      const min = Math.min(r, g, b);
      const d = max - min;
      let h = 0;
      const s = max === 0 ? 0 : d / max;
      const v = max / 255;

      if (d !== 0) {
        if (max === r) h = ((g - b) / d + (g < b ? 6 : 0)) / 6;
        else if (max === g) h = ((b - r) / d + 2) / 6;
        else h = ((r - g) / d + 4) / 6;
      }

      // Modulate S and V
      const newS = Math.min(1, Math.max(0, s * sMod));
      const newV = Math.min(1, Math.max(0, v * vMod));

      // Convert back to RGB
      const hi = Math.floor(h * 6) % 6;
      const f = h * 6 - Math.floor(h * 6);
      const p = newV * (1 - newS);
      const q = newV * (1 - f * newS);
      const t = newV * (1 - (1 - f) * newS);

      let rOut: number, gOut: number, bOut: number;
      switch (hi) {
        case 0:
          rOut = newV;
          gOut = t;
          bOut = p;
          break;
        case 1:
          rOut = q;
          gOut = newV;
          bOut = p;
          break;
        case 2:
          rOut = p;
          gOut = newV;
          bOut = t;
          break;
        case 3:
          rOut = p;
          gOut = q;
          bOut = newV;
          break;
        case 4:
          rOut = t;
          gOut = p;
          bOut = newV;
          break;
        default:
          rOut = newV;
          gOut = p;
          bOut = q;
          break;
      }

      const outIdx = i * 4;
      out.data[outIdx] = Math.round(rOut * 255);
      out.data[outIdx + 1] = Math.round(gOut * 255);
      out.data[outIdx + 2] = Math.round(bOut * 255);
      out.data[outIdx + 3] = 255;
    }
  }

  return out;
}

export { cppn };
