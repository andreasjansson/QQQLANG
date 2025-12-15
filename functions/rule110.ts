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

function rule110(ctx: FnContext, n: number): Image {
  const prev = getPrevImage(ctx);
  const { width, height } = ctx;

  // Invert input image first so the state-based inversion doesn't result in a fully inverted output
  const inverted = cloneImage(prev);
  for (let i = 0; i < inverted.data.length; i += 4) {
    inverted.data[i] = 255 - inverted.data[i];
    inverted.data[i + 1] = 255 - inverted.data[i + 1];
    inverted.data[i + 2] = 255 - inverted.data[i + 2];
  }

  const CHUNK_SIZE = 2;
  const chunksX = Math.ceil(width / CHUNK_SIZE);
  const chunksY = Math.ceil(height / CHUNK_SIZE);

  // Rule 110 lookup table: index = (left << 2) | (center << 1) | right
  // 111→0, 110→1, 101→1, 100→0, 011→1, 010→1, 001→1, 000→0
  const rule110 = [0, 1, 1, 1, 0, 1, 1, 0];

  const iterations = Math.max(1, n * 4);

  // Compute chunk luminance averages from original (not inverted) for CA state
  const chunkLuminance = new Float32Array(chunksX * chunksY);
  for (let cy = 0; cy < chunksY; cy++) {
    for (let cx = 0; cx < chunksX; cx++) {
      let sum = 0,
        count = 0;
      const startX = cx * CHUNK_SIZE;
      const startY = cy * CHUNK_SIZE;
      const endX = Math.min(startX + CHUNK_SIZE, width);
      const endY = Math.min(startY + CHUNK_SIZE, height);

      for (let y = startY; y < endY; y++) {
        for (let x = startX; x < endX; x++) {
          const [r, g, b] = getPixel(prev, x, y);
          sum += r * 0.299 + g * 0.587 + b * 0.114;
          count++;
        }
      }
      chunkLuminance[cy * chunksX + cx] = sum / count;
    }
  }

  // Compute threshold from image statistics (median of chunk luminances)
  const sortedLuminance = Array.from(chunkLuminance).sort((a, b) => a - b);
  const threshold = sortedLuminance[Math.floor(sortedLuminance.length / 2)];

  // Binarize chunk states using adaptive threshold
  let chunkStates = new Uint8Array(chunksX * chunksY);
  for (let i = 0; i < chunkLuminance.length; i++) {
    chunkStates[i] = chunkLuminance[i] > threshold ? 1 : 0;
  }

  // Evolve chunk states using Rule 110 (horizontal neighbors per row)
  for (let gen = 0; gen < iterations; gen++) {
    const newStates = new Uint8Array(chunksX * chunksY);

    for (let cy = 0; cy < chunksY; cy++) {
      for (let cx = 0; cx < chunksX; cx++) {
        const left = chunkStates[cy * chunksX + ((cx - 1 + chunksX) % chunksX)];
        const center = chunkStates[cy * chunksX + cx];
        const right = chunkStates[cy * chunksX + ((cx + 1) % chunksX)];

        const index = (left << 2) | (center << 1) | right;
        newStates[cy * chunksX + cx] = rule110[index];
      }
    }

    chunkStates = newStates;
  }

  // Render output: blend inverted image content with state-based inversion
  const out = createSolidImage(width, height, "#000000");

  for (let cy = 0; cy < chunksY; cy++) {
    for (let cx = 0; cx < chunksX; cx++) {
      const state = chunkStates[cy * chunksX + cx];
      const startX = cx * CHUNK_SIZE;
      const startY = cy * CHUNK_SIZE;
      const endX = Math.min(startX + CHUNK_SIZE, width);
      const endY = Math.min(startY + CHUNK_SIZE, height);

      // Source chunk for visual content - pull from neighbor based on state
      const srcCx =
        state === 1 ? (cx - 1 + chunksX) % chunksX : (cx + 1) % chunksX;
      const srcStartX = srcCx * CHUNK_SIZE;

      for (let y = startY; y < endY; y++) {
        for (let x = startX; x < endX; x++) {
          const localX = x - startX;
          const srcX = Math.min(srcStartX + localX, width - 1);

          const [r, g, b] = getPixel(inverted, srcX, y);

          if (state === 1) {
            setPixel(out, x, y, r, g, b);
          } else {
            // Invert for state 0 - since input is already inverted, this restores original colors
            setPixel(out, x, y, 255 - r, 255 - g, 255 - b);
          }
        }
      }
    }
  }

  return out;
}

export { rule110 };
