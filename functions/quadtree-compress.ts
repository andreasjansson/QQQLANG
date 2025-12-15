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

function quadtreeCompress(ctx: FnContext, n: number): Image {
  const prev = getPrevImage(ctx);
  const { width, height } = ctx;
  const out = createSolidImage(width, height, "#000000");

  // Map n (1-68): A=minimal compression, ~=maximal compression
  // Doubled range for stronger effect at high values
  const compressionLevel = ((n - 1) / 67) * 2;

  // Base threshold - higher means more merging
  const baseThreshold = compressionLevel * 3000;

  // Non-linear exponent: larger blocks need lower variance to merge
  // Lower exponent = less aggressive detail preservation
  const exponent = 0.5 + compressionLevel * 0.3;

  const maxBlockSize = Math.max(width, height);

  // Calculate color variance of a block
  const getBlockVariance = (
    x0: number,
    y0: number,
    w: number,
    h: number,
  ): number => {
    let sumR = 0,
      sumG = 0,
      sumB = 0;
    let sumR2 = 0,
      sumG2 = 0,
      sumB2 = 0;
    let count = 0;

    const endX = Math.min(x0 + w, width);
    const endY = Math.min(y0 + h, height);

    for (let y = y0; y < endY; y++) {
      for (let x = x0; x < endX; x++) {
        const [r, g, b] = getPixel(prev, x, y);
        sumR += r;
        sumG += g;
        sumB += b;
        sumR2 += r * r;
        sumG2 += g * g;
        sumB2 += b * b;
        count++;
      }
    }

    if (count === 0) return 0;

    const varR = sumR2 / count - (sumR / count) ** 2;
    const varG = sumG2 / count - (sumG / count) ** 2;
    const varB = sumB2 / count - (sumB / count) ** 2;

    return varR + varG + varB;
  };

  // Get average color of a block
  const getBlockAverage = (
    x0: number,
    y0: number,
    w: number,
    h: number,
  ): [number, number, number] => {
    let sumR = 0,
      sumG = 0,
      sumB = 0;
    let count = 0;

    const endX = Math.min(x0 + w, width);
    const endY = Math.min(y0 + h, height);

    for (let y = y0; y < endY; y++) {
      for (let x = x0; x < endX; x++) {
        const [r, g, b] = getPixel(prev, x, y);
        sumR += r;
        sumG += g;
        sumB += b;
        count++;
      }
    }

    if (count === 0) return [0, 0, 0];
    return [sumR / count, sumG / count, sumB / count];
  };

  // Fill a block with a color
  const fillBlock = (
    x0: number,
    y0: number,
    w: number,
    h: number,
    r: number,
    g: number,
    b: number,
  ): void => {
    const endX = Math.min(x0 + w, width);
    const endY = Math.min(y0 + h, height);

    for (let y = y0; y < endY; y++) {
      for (let x = x0; x < endX; x++) {
        setPixel(out, x, y, Math.round(r), Math.round(g), Math.round(b));
      }
    }
  };

  // Recursive quadtree decomposition
  const processBlock = (x0: number, y0: number, w: number, h: number): void => {
    if (w <= 0 || h <= 0 || x0 >= width || y0 >= height) return;

    // Minimum block size of 1
    if (w <= 1 && h <= 1) {
      const [r, g, b] = getPixel(prev, x0, y0);
      setPixel(out, x0, y0, r, g, b);
      return;
    }

    const variance = getBlockVariance(x0, y0, w, h);
    const blockSize = Math.max(w, h);

    // Non-linear threshold: large blocks need VERY low variance to merge
    // Small blocks can merge with higher variance
    // This ensures fine detail is preserved even at high compression
    const sizeRatio = blockSize / maxBlockSize;
    const effectiveThreshold =
      baseThreshold * Math.pow(1 - sizeRatio, exponent);

    // If block is uniform enough, fill with average
    if (variance < effectiveThreshold) {
      const [r, g, b] = getBlockAverage(x0, y0, w, h);
      fillBlock(x0, y0, w, h, r, g, b);
    } else {
      // Subdivide into 4 quadrants
      const halfW = Math.floor(w / 2);
      const halfH = Math.floor(h / 2);

      if (halfW === 0 && halfH === 0) {
        const [r, g, b] = getPixel(prev, x0, y0);
        setPixel(out, x0, y0, r, g, b);
        return;
      }

      processBlock(x0, y0, Math.max(1, halfW), Math.max(1, halfH));
      processBlock(x0 + halfW, y0, w - halfW, Math.max(1, halfH));
      processBlock(x0, y0 + halfH, Math.max(1, halfW), h - halfH);
      processBlock(x0 + halfW, y0 + halfH, w - halfW, h - halfH);
    }
  };

  // Start with the full image
  processBlock(0, 0, width, height);

  return out;
}

export { quadtreeCompress };
