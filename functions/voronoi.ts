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

  const gridCols = 6 + Math.round(((n - 1) * 11) / 67);
  const gridRows = 6 + Math.round(((n - 1) * 11) / 67);
  const cellW = ctx.width / gridCols;
  const cellH = ctx.height / gridRows;
  
  // Use angles that create more interference patterns
  // Small angles relative to grid alignment create interesting wrap effects
  const angle = (n * Math.PI) / 68;  // Smaller divisor = more variation per n
  const cos_a = Math.cos(angle);
  const sin_a = Math.sin(angle);
  const offsetX = (n % 17) / 17;
  const offsetY = ((n * 7) % 17) / 17;
  const cx = ctx.width / 2;
  const cy = ctx.height / 2;

  // More extension = more wrapping interference
  const extend = 2 + (n % 3);
  
  const seeds: [number, number][] = [];
  for (let row = -extend; row < gridRows + extend; row++) {
    for (let col = -extend; col < gridCols + extend; col++) {
      const baseX = (col + 0.5 + offsetX) * cellW;
      const baseY = (row + 0.5 + offsetY) * cellH;
      const dx = baseX - cx;
      const dy = baseY - cy;
      const rotX = cx + dx * cos_a - dy * sin_a;
      const rotY = cy + dx * sin_a + dy * cos_a;
      seeds.push([
        ((rotX % ctx.width) + ctx.width) % ctx.width,
        ((rotY % ctx.height) + ctx.height) % ctx.height,
      ]);
    }
  }

  for (let y = 0; y < ctx.height; y++) {
    for (let x = 0; x < ctx.width; x++) {
      let minDist = Infinity;
      let closestIdx = 0;

      for (let i = 0; i < seeds.length; i++) {
        const dx = x - seeds[i][0];
        const dy = y - seeds[i][1];
        const dist = dx * dx + dy * dy;
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
