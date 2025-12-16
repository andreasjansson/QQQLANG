import {
  FnContext,
  Image,
  createSolidImage,
  getPrevImage,
  getPixel,
  setPixel,
} from "./helpers.js";

function voronoi(ctx: FnContext, old: Image, n: number): Image {
  const prev = getPrevImage(ctx);
  const out = createSolidImage(ctx.width, ctx.height, "#000000");

  // Use golden ratio for non-repeating patterns
  const phi = 1.618033988749895;
  const sqrt2 = 1.4142135623730951;
  const sqrt3 = 1.7320508075688772;

  // Grid size varies with n, using prime-ish ratios for interesting tiling
  const baseGrid = 5 + Math.round(((n - 1) * 10) / 67);
  const gridCols = baseGrid;
  // Use golden ratio to avoid alignment with screen aspect
  const gridRows = Math.round(baseGrid * phi * 0.7);

  const cellW = ctx.width / gridCols;
  const cellH = ctx.height / gridRows;

  // Different pattern modes based on n
  const patternMode = n % 8;

  // Jitter amount varies with n
  const jitterAmount = 0.3 + ((n % 23) / 23) * 0.4;

  // Wave parameters for organic distortion
  const waveFreqX = 2 + (n % 7);
  const waveFreqY = 3 + ((n * 3) % 5);
  const waveAmp = cellW * (0.1 + ((n % 11) / 11) * 0.3);

  // Rotation angle using irrational multiple
  const angle = n * phi * 0.1;
  const cos_a = Math.cos(angle);
  const sin_a = Math.sin(angle);

  // Phase offsets using irrational numbers to avoid screen alignment
  const phaseX = (n * phi) % 1;
  const phaseY = (n * sqrt2) % 1;

  const cx = ctx.width / 2;
  const cy = ctx.height / 2;

  // Generate seeds with various patterns
  const seeds: [number, number][] = [];

  // Deterministic pseudo-random based on index
  const hash = (i: number, j: number): number => {
    const x = Math.sin(i * 12.9898 + j * 78.233) * 43758.5453;
    return x - Math.floor(x);
  };

  for (let row = -1; row <= gridRows + 1; row++) {
    for (let col = -1; col <= gridCols + 1; col++) {
      // Base position with phase offset
      let baseX = (col + 0.5 + phaseX) * cellW;
      let baseY = (row + 0.5 + phaseY) * cellH;

      // Apply pattern-specific jitter
      const h1 = hash(col + n, row);
      const h2 = hash(row + n, col * 2 + 1);

      switch (patternMode) {
        case 0:
          // Hexagonal-ish offset
          if (row % 2 === 0) {
            baseX += cellW * 0.5;
          }
          baseX += (h1 - 0.5) * cellW * jitterAmount;
          baseY += (h2 - 0.5) * cellH * jitterAmount;
          break;

        case 1:
          // Spiral jitter - distance from center affects offset
          const distFromCenter = Math.sqrt(
            (baseX - cx) ** 2 + (baseY - cy) ** 2
          );
          const spiralAngle = distFromCenter * 0.01 + n * 0.1;
          baseX += Math.cos(spiralAngle) * cellW * jitterAmount * h1;
          baseY += Math.sin(spiralAngle) * cellH * jitterAmount * h2;
          break;

        case 2:
          // Wave-based displacement
          baseX +=
            Math.sin((row * sqrt3 + phaseX * 10) * 0.5) * cellW * jitterAmount;
          baseY +=
            Math.cos((col * phi + phaseY * 10) * 0.7) * cellH * jitterAmount;
          break;

        case 3:
          // Triangular grid approximation
          if ((row + col) % 2 === 0) {
            baseX += cellW * 0.25;
            baseY += cellH * 0.15;
          } else {
            baseX -= cellW * 0.25;
            baseY -= cellH * 0.15;
          }
          baseX += (h1 - 0.5) * cellW * jitterAmount * 0.5;
          baseY += (h2 - 0.5) * cellH * jitterAmount * 0.5;
          break;

        case 4:
          // Radial pattern - seeds pushed outward from center
          const dx0 = baseX - cx;
          const dy0 = baseY - cy;
          const r0 = Math.sqrt(dx0 * dx0 + dy0 * dy0) + 1;
          const push = Math.sin(r0 * 0.02 + n * 0.3) * cellW * 0.3;
          baseX += (dx0 / r0) * push;
          baseY += (dy0 / r0) * push;
          baseX += (h1 - 0.5) * cellW * jitterAmount * 0.7;
          baseY += (h2 - 0.5) * cellH * jitterAmount * 0.7;
          break;

        case 5:
          // Diagonal waves
          const diag = (col + row) * 0.5;
          baseX += Math.sin(diag * phi) * cellW * jitterAmount;
          baseY += Math.cos(diag * sqrt2) * cellH * jitterAmount;
          break;

        case 6:
          // Clustered - some cells get multiple nearby seeds
          baseX += (h1 - 0.5) * cellW * jitterAmount * 1.2;
          baseY += (h2 - 0.5) * cellH * jitterAmount * 1.2;
          // Add extra seed nearby sometimes
          if (h1 > 0.7 && col >= 0 && row >= 0) {
            seeds.push([
              baseX + cellW * 0.2 * (h2 - 0.5),
              baseY + cellH * 0.2 * (h1 - 0.5),
            ]);
          }
          break;

        case 7:
        default:
          // Organic noise-like displacement
          const noise1 =
            Math.sin(col * 1.3 + n) * Math.cos(row * 0.7) +
            Math.sin(row * 1.1 + col * 0.9);
          const noise2 =
            Math.cos(row * 1.5 + n) * Math.sin(col * 0.8) +
            Math.cos(col * 1.2 + row * 0.6);
          baseX += noise1 * cellW * jitterAmount * 0.5;
          baseY += noise2 * cellH * jitterAmount * 0.5;
          break;
      }

      // Apply global rotation around center
      const dx = baseX - cx;
      const dy = baseY - cy;
      const rotX = cx + dx * cos_a - dy * sin_a;
      const rotY = cy + dx * sin_a + dy * cos_a;

      seeds.push([rotX, rotY]);
    }
  }

  // Distance function varies with n
  const distMode = Math.floor(n / 8) % 6;

  const distance = (x: number, y: number, sx: number, sy: number): number => {
    const dx = x - sx;
    const dy = y - sy;

    switch (distMode) {
      case 0:
        // Standard Euclidean
        return dx * dx + dy * dy;

      case 1:
        // Manhattan - creates diamond shapes
        return Math.abs(dx) + Math.abs(dy);

      case 2:
        // Chebyshev - creates square shapes
        return Math.max(Math.abs(dx), Math.abs(dy));

      case 3:
        // Weighted - creates elongated cells
        return dx * dx * 1.5 + dy * dy * 0.7;

      case 4:
        // Minkowski p=3 - rounder squares
        return Math.pow(Math.abs(dx), 3) + Math.pow(Math.abs(dy), 3);

      case 5:
      default:
        // Mixed - organic blobs
        const euclidean = Math.sqrt(dx * dx + dy * dy);
        const manhattan = Math.abs(dx) + Math.abs(dy);
        return euclidean * 0.6 + manhattan * 0.4;
    }
  };

  // Add wave distortion to lookup coordinates
  const waveDistortX = (x: number, y: number): number => {
    return (
      x + Math.sin((y / ctx.height) * waveFreqY * Math.PI * 2 + n) * waveAmp
    );
  };

  const waveDistortY = (x: number, y: number): number => {
    return (
      y + Math.sin((x / ctx.width) * waveFreqX * Math.PI * 2 + n * phi) * waveAmp
    );
  };

  for (let y = 0; y < ctx.height; y++) {
    for (let x = 0; x < ctx.width; x++) {
      // Apply wave distortion to the lookup point
      const wx = waveDistortX(x, y);
      const wy = waveDistortY(x, y);

      let minDist = Infinity;
      let closestIdx = 0;

      for (let i = 0; i < seeds.length; i++) {
        const dist = distance(wx, wy, seeds[i][0], seeds[i][1]);
        if (dist < minDist) {
          minDist = dist;
          closestIdx = i;
        }
      }

      const src = closestIdx % 2 === 0 ? prev : old;
      const [r, g, b] = getPixel(src, x, y);
      setPixel(out, x, y, r, g, b);
    }
  }

  return out;
}

export { voronoi };
