import {
  FnContext,
  Image,
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
  const gradientThreshold = 15;
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

  const colorThreshold = 30;

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

      if (x < width - 1) {
        const nidx = y * width + (x + 1);
        if (isFlat[nidx]) {
          const [nr, ng, nb] = getPixel(prev, x + 1, y);
          if (colorSimilar(r, g, b, nr, ng, nb)) {
            union(idx, nidx);
          }
        }
      }

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

  // Step 4: Gather region statistics including mean color and bounding box
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
    }
  }

  // Step 5: Apply gradients to regions
  const minRegionSize = Math.max(30, (width * height) / 800);
  const lightnessRange = 0.45;
  const saturationBoost = 1.8;

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const idx = y * width + x;
      if (!isFlat[idx]) continue;

      const root = find(idx);
      const stats = regionStats.get(root);
      if (!stats || stats.count < minRegionSize) continue;

      // Calculate mean color for region
      const meanR = stats.sumR / stats.count;
      const meanG = stats.sumG / stats.count;
      const meanB = stats.sumB / stats.count;
      const [h, s, l] = rgbToHsl(meanR, meanG, meanB);

      // Calculate gradient position: top-left (0) to bottom-right (1)
      // Angle slightly to the left means more weight on Y than X
      const regionWidth = stats.maxX - stats.minX + 1;
      const regionHeight = stats.maxY - stats.minY + 1;
      
      const normX = regionWidth > 1 ? (x - stats.minX) / (regionWidth - 1) : 0.5;
      const normY = regionHeight > 1 ? (y - stats.minY) / (regionHeight - 1) : 0.5;
      
      // Angled gradient: more vertical than horizontal (0.3x + 0.7y)
      const gradientPos = normX * 0.3 + normY * 0.7;

      // Lightness: high at top, low at bottom
      const lightnessShift = (0.5 - gradientPos) * lightnessRange;
      const newL = Math.max(0.05, Math.min(0.95, l + lightnessShift));

      // Boost saturation
      const newS = Math.min(1, s * saturationBoost);

      const [newR, newG, newB] = hslToRgb(h, newS, newL);
      setPixel(out, x, y, newR, newG, newB);
    }
  }

  return out;
}

export { gradientify };
