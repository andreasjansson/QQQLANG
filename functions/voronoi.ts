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

  const numSeeds = 36 + Math.round(((n - 1) * 100) / 67);
  
  // Use golden angle for seed placement - creates natural spiraling patterns
  // that tile interestingly without being a boring grid
  const goldenAngle = Math.PI * (3 - Math.sqrt(5)); // ~137.5 degrees
  
  const cx = ctx.width / 2;
  const cy = ctx.height / 2;
  const maxRadius = Math.sqrt(cx * cx + cy * cy);
  
  // n controls the spiral tightness and rotation
  const spiralTightness = 0.5 + (n % 17) / 17;
  const globalRotation = (n * Math.PI) / 34;
  
  const seeds: [number, number][] = [];
  
  for (let i = 0; i < numSeeds; i++) {
    // Fermat spiral with golden angle - creates sunflower-like seed pattern
    const angle = i * goldenAngle + globalRotation;
    const radius = maxRadius * Math.sqrt(i / numSeeds) * spiralTightness + 
                   maxRadius * (1 - spiralTightness) * (i / numSeeds);
    
    const x = cx + radius * Math.cos(angle);
    const y = cy + radius * Math.sin(angle);
    
    seeds.push([x, y]);
  }
  
  // Add edge seeds to fill corners
  const edgeCount = Math.ceil(Math.sqrt(numSeeds));
  for (let i = 0; i < edgeCount; i++) {
    const t = i / edgeCount;
    seeds.push([t * ctx.width, 0]);
    seeds.push([t * ctx.width, ctx.height]);
    seeds.push([0, t * ctx.height]);
    seeds.push([ctx.width, t * ctx.height]);
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
