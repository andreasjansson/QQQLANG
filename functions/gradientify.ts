import {
  FnContext,
  Image,
  createSolidImage,
  getPrevImage,
  getPixel,
  setPixel,
  cloneImage,
  rgbToHsl,
  hslToRgb,
} from "./helpers.js";

function gradientify(ctx: FnContext): Image {
  const prev = getPrevImage(ctx);
  const out = cloneImage(prev);
  const { width, height } = prev;

  // Step 1: Compute gradient magnitude for each pixel
  const gradientMag = new Float32Array(width * height);

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const idx = y * width + x;
      const [r, g, b] = getPixel(prev, x, y);

      const [r1, g1, b1] = x > 0 ? getPixel(prev, x - 1, y) : [r, g, b];
      const [r2, g2, b2] =
        x < width - 1 ? getPixel(prev, x + 1, y) : [r, g, b];
      const [r3, g3, b3] = y > 0 ? getPixel(prev, x, y - 1) : [r, g, b];
      const [r4, g4, b4] =
        y < height - 1 ? getPixel(prev, x, y + 1) : [r, g, b];

      const gxR = (r2 - r1) / 2;
      const gxG = (g2 - g1) / 2;
      const gxB = (b2 - b1) / 2;
      const gyR = (r4 - r3) / 2;
      const gyG = (g4 - g3) / 2;
      const gyB = (b4 - b3) / 2;

      const mag = Math.sqrt(
        gxR * gxR +
          gxG * gxG +
          gxB * gxB +
          gyR * gyR +
          gyG * gyG +
          gyB * gyB,
      );
      gradientMag[idx] = mag;
    }
  }

  // Step 2: Create flatness mask with threshold
  const gradientThreshold = 12;
  const isFlat = new Uint8Array(width * height);
  for (let i = 0; i < width * height; i++) {
    isFlat[i] = gradientMag[i] < gradientThreshold ? 1 : 0;
  }

  // Step 3: Connected components using union-find for flat regions with color similarity
  const parent = new Int32Array(width * height);
  const rank = new Uint8Array(width * height);
  for (let i = 0; i < width * height; i++) {
    parent[i] = i;
    rank[i] = 0;
  }

  function find(x: number): number {
    if (parent[x] !== x) {
      parent[x] = find(parent[x]);
    }
    return parent[x];
  }

  function union(x: number, y: number): void {
    const px = find(x);
    const py = find(y);
    if (px === py) return;
    if (rank[px] < rank[py]) {
      parent[px] = py;
    } else if (rank[px] > rank[py]) {
      parent[py] = px;
    } else {
      parent[py] = px;
      rank[px]++;
    }
  }

  const colorThreshold = 25;

  function colorSimilar(
    r1: number,
    g1: number,
    b1: number,
    r2: number,
    g2: number,
    b2: number,
  ): boolean {
    const dr = r1 - r2;
    const dg = g1 - g2;
    const db = b1 - b2;
    return Math.sqrt(dr * dr + dg * dg + db * db) < colorThreshold;
  }

  // Union adjacent flat pixels with similar colors
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const idx = y * width + x;
      if (!isFlat[idx]) continue;

      const [r, g, b] = getPixel(prev, x, y);

      // Check right neighbor
      if (x < width - 1) {
        const nidx = y * width + (x + 1);
        if (isFlat[nidx]) {
          const [nr, ng, nb] = getPixel(prev, x + 1, y);
          if (colorSimilar(r, g, b, nr, ng, nb)) {
            union(idx, nidx);
          }
        }
      }

      // Check bottom neighbor
      if (y < height - 1) {
        const nidx = (y + 1) * width + x;
        if (isFlat[nidx]) {
          const [nr, ng, nb] = getPixel(prev, x, y + 1);
          if (colorSimilar(r, g, b, nr, ng, nb)) {
            union(idx, nidx);
          }
        }
      }
    }
  }

  // Step 4: Gather region statistics
  const regionStats = new Map<
    number,
    {
      sumR: number;
      sumG: number;
      sumB: number;
      minX: number;
      maxX: number;
      minY: number;
      maxY: number;
      count: number;
      sumX: number;
      sumY: number;
    }
  >();

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const idx = y * width + x;
      if (!isFlat[idx]) continue;

      const root = find(idx);
      const [r, g, b] = getPixel(prev, x, y);

      if (!regionStats.has(root)) {
        regionStats.set(root, {
          sumR: 0,
          sumG: 0,
          sumB: 0,
          minX: x,
          maxX: x,
          minY: y,
          maxY: y,
          count: 0,
          sumX: 0,
          sumY: 0,
        });
      }

      const stats = regionStats.get(root)!;
      stats.sumR += r;
      stats.sumG += g;
      stats.sumB += b;
      stats.minX = Math.min(stats.minX, x);
      stats.maxX = Math.max(stats.maxX, x);
      stats.minY = Math.min(stats.minY, y);
      stats.maxY = Math.max(stats.maxY, y);
      stats.count++;
      stats.sumX += x;
      stats.sumY += y;
    }
  }

  // Step 5: Apply gradients to regions
  const minRegionSize = Math.max(50, (width * height) / 500);
  const hueShiftAmount = 15;
  const lightnessShiftAmount = 0.08;

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const idx = y * width + x;
      if (!isFlat[idx]) continue;

      const root = find(idx);
      const stats = regionStats.get(root);
      if (!stats || stats.count < minRegionSize) continue;

      const [r, g, b] = getPixel(prev, x, y);
      const [h, s, l] = rgbToHsl(r, g, b);

      // Calculate gradient direction based on region shape
      const regionWidth = stats.maxX - stats.minX + 1;
      const regionHeight = stats.maxY - stats.minY + 1;
      const centerX = stats.sumX / stats.count;
      const centerY = stats.sumY / stats.count;

      // Use diagonal gradient direction based on region aspect ratio
      let gradientPos: number;
      if (regionWidth > regionHeight * 1.5) {
        // Wide region: horizontal gradient
        gradientPos =
          regionWidth > 1 ? (x - stats.minX) / (regionWidth - 1) : 0.5;
      } else if (regionHeight > regionWidth * 1.5) {
        // Tall region: vertical gradient
        gradientPos =
          regionHeight > 1 ? (y - stats.minY) / (regionHeight - 1) : 0.5;
      } else {
        // Square-ish region: radial gradient from center
        const dx = x - centerX;
        const dy = y - centerY;
        const maxDist =
          Math.sqrt(regionWidth * regionWidth + regionHeight * regionHeight) /
          2;
        gradientPos =
          maxDist > 0
            ? Math.min(1, Math.sqrt(dx * dx + dy * dy) / maxDist)
            : 0.5;
      }

      // Apply subtle hue shift
      const hueShift = (gradientPos - 0.5) * hueShiftAmount;
      let newH = h + hueShift;
      if (newH < 0) newH += 360;
      if (newH >= 360) newH -= 360;

      // Apply subtle lightness variation
      const lightnessShift = (gradientPos - 0.5) * lightnessShiftAmount;
      const newL = Math.max(0, Math.min(1, l + lightnessShift));

      // Slightly boost saturation in gradients
      const newS = Math.min(1, s * 1.05);

      const [newR, newG, newB] = hslToRgb(newH, newS, newL);
      setPixel(out, x, y, newR, newG, newB);
    }
  }

  // Step 6: Feathering - create smooth transition at edges of gradient regions
  const featherRadius = 3;
  const feathered = cloneImage(out);

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const idx = y * width + x;

      // Check if this pixel is near a boundary between gradient and non-gradient
      let isNearBoundary = false;
      let gradientCount = 0;
      let nonGradientCount = 0;

      for (
        let dy = -featherRadius;
        dy <= featherRadius && !isNearBoundary;
        dy++
      ) {
        for (let dx = -featherRadius; dx <= featherRadius; dx++) {
          const nx = x + dx;
          const ny = y + dy;
          if (nx < 0 || nx >= width || ny < 0 || ny >= height) continue;

          const nidx = ny * width + nx;
          const nRoot = isFlat[nidx] ? find(nidx) : -1;
          const nStats = nRoot >= 0 ? regionStats.get(nRoot) : null;
          const nIsGradient = nStats && nStats.count >= minRegionSize;

          if (nIsGradient) {
            gradientCount++;
          } else {
            nonGradientCount++;
          }
        }
      }

      if (gradientCount > 0 && nonGradientCount > 0) {
        // This pixel is near a boundary - blend
        const blendFactor = gradientCount / (gradientCount + nonGradientCount);
        const [origR, origG, origB] = getPixel(prev, x, y);
        const [gradR, gradG, gradB] = getPixel(out, x, y);

        const finalR = Math.round(
          origR * (1 - blendFactor) + gradR * blendFactor,
        );
        const finalG = Math.round(
          origG * (1 - blendFactor) + gradG * blendFactor,
        );
        const finalB = Math.round(
          origB * (1 - blendFactor) + gradB * blendFactor,
        );

        setPixel(feathered, x, y, finalR, finalG, finalB);
      }
    }
  }

  return feathered;
}

export { gradientify };
