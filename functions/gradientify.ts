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

function hueDiff(h1: number, h2: number): number {
  // Circular distance for hue (0-360)
  const diff = Math.abs(h1 - h2);
  return Math.min(diff, 360 - diff);
}

function gradientify(ctx: FnContext): Image {
  const prev = getPrevImage(ctx);
  const out = cloneImage(prev);
  const { width, height } = prev;

  // Pre-compute HSL for all pixels
  const hslData = new Float32Array(width * height * 3);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const idx = y * width + x;
      const [r, g, b] = getPixel(prev, x, y);
      const [h, s, l] = rgbToHsl(r, g, b);
      hslData[idx * 3] = h;
      hslData[idx * 3 + 1] = s;
      hslData[idx * 3 + 2] = l;
    }
  }

  function getHsl(x: number, y: number): [number, number, number] {
    const idx = y * width + x;
    return [hslData[idx * 3], hslData[idx * 3 + 1], hslData[idx * 3 + 2]];
  }

  // Step 1: Compute gradient magnitude in HSL space
  const gradientMag = new Float32Array(width * height);

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const idx = y * width + x;
      const [h, s, l] = getHsl(x, y);

      const [h1, s1, l1] = x > 0 ? getHsl(x - 1, y) : [h, s, l];
      const [h2, s2, l2] = x < width - 1 ? getHsl(x + 1, y) : [h, s, l];
      const [h3, s3, l3] = y > 0 ? getHsl(x, y - 1) : [h, s, l];
      const [h4, s4, l4] = y < height - 1 ? getHsl(x, y + 1) : [h, s, l];

      // Hue gradient (weighted by saturation - hue matters less when desaturated)
      const avgS = (s + s1 + s2 + s3 + s4) / 5;
      const hueWeight = avgS * 0.5; // Scale hue importance by saturation
      
      const gxH = hueDiff(h2, h1) / 2 * hueWeight;
      const gyH = hueDiff(h4, h3) / 2 * hueWeight;
      
      // Saturation gradient (0-1 scale, multiply by 100 to match lightness scale)
      const gxS = (s2 - s1) / 2 * 100;
      const gyS = (s4 - s3) / 2 * 100;
      
      // Lightness gradient (0-1 scale, multiply by 100)
      const gxL = (l2 - l1) / 2 * 100;
      const gyL = (l4 - l3) / 2 * 100;

      const mag = Math.sqrt(
        gxH * gxH + gyH * gyH +
        gxS * gxS + gyS * gyS +
        gxL * gxL + gyL * gyL
      );
      gradientMag[idx] = mag;
    }
  }

  // Step 2: Create flatness mask with threshold
  const gradientThreshold = 1.5;
  const isFlatRaw = new Uint8Array(width * height);
  for (let i = 0; i < width * height; i++) {
    isFlatRaw[i] = gradientMag[i] < gradientThreshold ? 1 : 0;
  }

  // Erode the flatness mask to remove small isolated flat patches
  // A pixel stays flat only if most of its neighbors are also flat
  const erosionRadius = 5;
  const erosionThreshold = 0.7; // 70% of neighbors must be flat
  const isFlat = new Uint8Array(width * height);
  
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const idx = y * width + x;
      if (!isFlatRaw[idx]) {
        isFlat[idx] = 0;
        continue;
      }
      
      let flatCount = 0;
      let totalCount = 0;
      
      for (let dy = -erosionRadius; dy <= erosionRadius; dy++) {
        for (let dx = -erosionRadius; dx <= erosionRadius; dx++) {
          const nx = x + dx;
          const ny = y + dy;
          if (nx < 0 || nx >= width || ny < 0 || ny >= height) continue;
          
          totalCount++;
          if (isFlatRaw[ny * width + nx]) {
            flatCount++;
          }
        }
      }
      
      isFlat[idx] = (flatCount / totalCount) >= erosionThreshold ? 1 : 0;
    }
  }

  // Step 3: Connected components using union-find for flat regions with HSL color similarity
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

  function hslSimilar(
    h1: number, s1: number, l1: number,
    h2: number, s2: number, l2: number,
  ): boolean {
    // Lightness difference
    const dL = Math.abs(l1 - l2);
    if (dL > 0.02) return false;
    
    // Saturation difference - be strict here
    const dS = Math.abs(s1 - s2);
    if (dS > 0.02) return false;
    
    // If one is gray and one is colored, don't merge
    const isGray1 = s1 < 0.05;
    const isGray2 = s2 < 0.05;
    if (isGray1 !== isGray2) return false;
    
    // Hue difference (only matters if both have decent saturation)
    if (!isGray1 && !isGray2) {
      const dH = hueDiff(h1, h2);
      if (dH > 5) return false;
    }
    
    return true;
  }

  // Union adjacent flat pixels with similar colors
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const idx = y * width + x;
      if (!isFlat[idx]) continue;

      const [h, s, l] = getHsl(x, y);

      if (x < width - 1) {
        const nidx = y * width + (x + 1);
        if (isFlat[nidx]) {
          const [nh, ns, nl] = getHsl(x + 1, y);
          if (hslSimilar(h, s, l, nh, ns, nl)) {
            union(idx, nidx);
          }
        }
      }

      if (y < height - 1) {
        const nidx = (y + 1) * width + x;
        if (isFlat[nidx]) {
          const [nh, ns, nl] = getHsl(x, y + 1);
          if (hslSimilar(h, s, l, nh, ns, nl)) {
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
      sumH_x: number; // For circular mean of hue
      sumH_y: number;
      sumS: number;
      sumL: number;
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
      const [h, s, l] = getHsl(x, y);

      if (!regionStats.has(root)) {
        regionStats.set(root, {
          sumH_x: 0,
          sumH_y: 0,
          sumS: 0,
          sumL: 0,
          minX: x,
          maxX: x,
          minY: y,
          maxY: y,
          count: 0,
        });
      }

      const stats = regionStats.get(root)!;
      // Circular mean for hue
      const hRad = (h / 360) * 2 * Math.PI;
      stats.sumH_x += Math.cos(hRad) * s; // Weight by saturation
      stats.sumH_y += Math.sin(hRad) * s;
      stats.sumS += s;
      stats.sumL += l;
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
      const meanH = ((Math.atan2(stats.sumH_y, stats.sumH_x) / (2 * Math.PI)) * 360 + 360) % 360;
      const meanS = stats.sumS / stats.count;
      const meanL = stats.sumL / stats.count;

      // Calculate gradient position: top-left (0) to bottom-right (1)
      const regionWidth = stats.maxX - stats.minX + 1;
      const regionHeight = stats.maxY - stats.minY + 1;
      
      const normX = regionWidth > 1 ? (x - stats.minX) / (regionWidth - 1) : 0.5;
      const normY = regionHeight > 1 ? (y - stats.minY) / (regionHeight - 1) : 0.5;
      
      // Angled gradient: more vertical than horizontal (0.3x + 0.7y)
      const gradientPos = normX * 0.3 + normY * 0.7;

      // Lightness: high at top, low at bottom
      const lightnessShift = (0.5 - gradientPos) * lightnessRange;
      const newL = Math.max(0.05, Math.min(0.95, meanL + lightnessShift));

      // Boost saturation
      const newS = Math.min(1, meanS * saturationBoost);

      const [newR, newG, newB] = hslToRgb(meanH, newS, newL);
      setPixel(out, x, y, newR, newG, newB);
    }
  }

  return out;
}

export { gradientify };
