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
  formatFunctionHelp,
  breakLigatures,
  wrapText,
  generateIntroPage,
  numToChar,
  generateCharacterRefLines,
  getPageChar,
  generateAllHelpPages,
  generateIndexPage,
  sinetSession,
  sinetInferenceInProgress,
  SINET_INPUT_SIZE,
  SINET_MEAN,
  SINET_STD,
} from "./helpers.js";
import * as ort from "onnxruntime-web";

async function bgRemove(ctx: FnContext, old: Image): Promise<Image> {
  const prev = getPrevImage(ctx);

  if (!sinetSession) {
    console.warn("SINet model not ready, returning prev image");
    return cloneImage(prev);
  }

  if (sinetInferenceInProgress) {
    console.warn("SINet inference already in progress, returning prev image");
    return cloneImage(prev);
  }

  sinetInferenceInProgress = true;
  const startTime = performance.now();

  // Resize input to 224x224 for SINet
  const inputCanvas = document.createElement("canvas");
  inputCanvas.width = SINET_INPUT_SIZE;
  inputCanvas.height = SINET_INPUT_SIZE;
  const inputCtx = inputCanvas.getContext("2d")!;

  // Draw prev image scaled to 224x224
  const prevCanvas = document.createElement("canvas");
  prevCanvas.width = prev.width;
  prevCanvas.height = prev.height;
  const prevCtx = prevCanvas.getContext("2d")!;
  const prevImageData = new ImageData(
    new Uint8ClampedArray(prev.data),
    prev.width,
    prev.height,
  );
  prevCtx.putImageData(prevImageData, 0, 0);
  inputCtx.drawImage(prevCanvas, 0, 0, SINET_INPUT_SIZE, SINET_INPUT_SIZE);

  const resizedData = inputCtx.getImageData(
    0,
    0,
    SINET_INPUT_SIZE,
    SINET_INPUT_SIZE,
  );

  // Preprocess: normalize with mean/std, convert to CHW format
  const inputTensor = new Float32Array(3 * SINET_INPUT_SIZE * SINET_INPUT_SIZE);
  for (let y = 0; y < SINET_INPUT_SIZE; y++) {
    for (let x = 0; x < SINET_INPUT_SIZE; x++) {
      const pixelIdx = (y * SINET_INPUT_SIZE + x) * 4;
      const r = resizedData.data[pixelIdx];
      const g = resizedData.data[pixelIdx + 1];
      const b = resizedData.data[pixelIdx + 2];

      // Normalize: (pixel - mean) / (std * 255)
      const tensorIdx = y * SINET_INPUT_SIZE + x;
      inputTensor[0 * SINET_INPUT_SIZE * SINET_INPUT_SIZE + tensorIdx] =
        (r - SINET_MEAN[0]) / (SINET_STD[0] * 255);
      inputTensor[1 * SINET_INPUT_SIZE * SINET_INPUT_SIZE + tensorIdx] =
        (g - SINET_MEAN[1]) / (SINET_STD[1] * 255);
      inputTensor[2 * SINET_INPUT_SIZE * SINET_INPUT_SIZE + tensorIdx] =
        (b - SINET_MEAN[2]) / (SINET_STD[2] * 255);
    }
  }

  // Run inference
  const feeds = {
    data: new ort.Tensor("float32", inputTensor, [
      1,
      3,
      SINET_INPUT_SIZE,
      SINET_INPUT_SIZE,
    ]),
  };
  const results = await sinetSession.run(feeds);

  // Get output mask
  const outputNames = Object.keys(results);
  const outputTensor = results[outputNames[0]];
  const outputData = outputTensor.data as Float32Array;

  console.log("SINet output dims:", outputTensor.dims);
  console.log("SINet output sample values:", outputData.slice(0, 10));

  // Create mask at model size
  const maskCanvas = document.createElement("canvas");
  maskCanvas.width = SINET_INPUT_SIZE;
  maskCanvas.height = SINET_INPUT_SIZE;
  const maskCtx = maskCanvas.getContext("2d")!;
  const maskImageData = maskCtx.createImageData(
    SINET_INPUT_SIZE,
    SINET_INPUT_SIZE,
  );

  // Handle different output formats
  const outputChannels = outputTensor.dims[1] as number;
  const spatialSize = SINET_INPUT_SIZE * SINET_INPUT_SIZE;

  for (let i = 0; i < spatialSize; i++) {
    let alpha: number;
    if (outputChannels === 2) {
      // Model is SINet_Softmax - already outputs probabilities
      // Channel 0 = background, Channel 1 = foreground
      const fg = outputData[spatialSize + i]; // foreground is second channel
      alpha = fg; // Already 0-1 from softmax
    } else {
      // Single channel - assume it's already a probability
      alpha = outputData[i];
    }

    // Clamp and convert to 0-255
    alpha = Math.max(0, Math.min(1, alpha));
    const alphaInt = Math.round(alpha * 255);

    // Set RGBA - use white with alpha for proper scaling
    maskImageData.data[i * 4] = 255;
    maskImageData.data[i * 4 + 1] = 255;
    maskImageData.data[i * 4 + 2] = 255;
    maskImageData.data[i * 4 + 3] = alphaInt;
  }

  console.log(
    "Mask alpha sample:",
    Array.from(maskImageData.data.slice(0, 40)).filter((_, i) => i % 4 === 3),
  );

  // Scale mask to original size
  maskCtx.putImageData(maskImageData, 0, 0);

  const outputCanvas = document.createElement("canvas");
  outputCanvas.width = ctx.width;
  outputCanvas.height = ctx.height;
  const outputCtx = outputCanvas.getContext("2d")!;

  // Draw background (old image)
  const oldImageData = new ImageData(
    new Uint8ClampedArray(old.data),
    old.width,
    old.height,
  );
  const oldCanvas = document.createElement("canvas");
  oldCanvas.width = old.width;
  oldCanvas.height = old.height;
  const oldCtx = oldCanvas.getContext("2d")!;
  oldCtx.putImageData(oldImageData, 0, 0);
  outputCtx.drawImage(oldCanvas, 0, 0, ctx.width, ctx.height);

  // Apply mask to foreground and draw
  // Scale mask to output size
  const scaledMaskCanvas = document.createElement("canvas");
  scaledMaskCanvas.width = ctx.width;
  scaledMaskCanvas.height = ctx.height;
  const scaledMaskCtx = scaledMaskCanvas.getContext("2d")!;
  scaledMaskCtx.drawImage(maskCanvas, 0, 0, ctx.width, ctx.height);
  const scaledMaskData = scaledMaskCtx.getImageData(
    0,
    0,
    ctx.width,
    ctx.height,
  );

  // Create foreground with alpha from mask
  const fgCanvas = document.createElement("canvas");
  fgCanvas.width = ctx.width;
  fgCanvas.height = ctx.height;
  const fgCtx = fgCanvas.getContext("2d")!;
  fgCtx.drawImage(prevCanvas, 0, 0, ctx.width, ctx.height);
  const fgData = fgCtx.getImageData(0, 0, ctx.width, ctx.height);

  for (let i = 0; i < fgData.data.length; i += 4) {
    // Sharpen the mask - boost alpha values to reduce transparency
    const rawAlpha = scaledMaskData.data[i + 3];
    // Apply a curve to make edges sharper: values above threshold become more opaque
    //const sharpened = rawAlpha < 10 ? 0 : Math.min(255, Math.round(rawAlpha * 1.5 + 50));
    const sharpened = rawAlpha;
    fgData.data[i + 3] = sharpened;
  }

  fgCtx.putImageData(fgData, 0, 0);
  outputCtx.drawImage(fgCanvas, 0, 0);

  const finalData = outputCtx.getImageData(0, 0, ctx.width, ctx.height);

  const endTime = performance.now();
  console.log(`SINet inference took ${(endTime - startTime).toFixed(1)}ms`);

  sinetInferenceInProgress = false;

  return {
    width: ctx.width,
    height: ctx.height,
    data: new Uint8ClampedArray(finalData.data),
  };
}

export { bgRemove };
