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

function neon(ctx: FnContext): Image {
  const prev = getPrevImage(ctx);
  const out = createSolidImage(ctx.width, ctx.height, "#000000");

  const scale = Math.min(ctx.width, ctx.height);
  const numLights = 15;

  const seed = ctx.images.length * 137.5;
  const hash = (n: number) => {
    const x = Math.sin(n + seed) * 43758.5453;
    return x - Math.floor(x);
  };

  const lights: {
    x: number;
    y: number;
    r: number;
    g: number;
    b: number;
    size: number;
  }[] = [];
  for (let i = 0; i < numLights; i++) {
    const px = hash(i * 127.1) * ctx.width;
    const py = hash(i * 311.7) * ctx.height;
    const colorAngle = hash(i * 74.3) * Math.PI * 2;
    const size = 0.03 + hash(i * 191.3) * 0.04;
    lights.push({
      x: px,
      y: py,
      r: Math.cos(colorAngle) * 0.5 + 0.5,
      g: Math.cos(colorAngle + (Math.PI * 2) / 3) * 0.5 + 0.5,
      b: Math.cos(colorAngle + (Math.PI * 4) / 3) * 0.5 + 0.5,
      size,
    });
  }

  const glowRadius = 8;

  for (let y = 0; y < ctx.height; y++) {
    for (let x = 0; x < ctx.width; x++) {
      const [pr, pg, pb] = getPixel(prev, x, y);
      const brightness = (pr + pg + pb) / (255 * 3);

      let tr = (pr / 255) * 0.3;
      let tg = (pg / 255) * 0.3;
      let tb = (pb / 255) * 0.3;

      if (brightness > 0.3) {
        let glowSum = 0;
        let glowR = 0,
          glowG = 0,
          glowB = 0;
        for (let dy = -glowRadius; dy <= glowRadius; dy += 2) {
          for (let dx = -glowRadius; dx <= glowRadius; dx += 2) {
            const sx = Math.max(0, Math.min(ctx.width - 1, x + dx));
            const sy = Math.max(0, Math.min(ctx.height - 1, y + dy));
            const [sr, sg, sb] = getPixel(prev, sx, sy);
            const sBright = (sr + sg + sb) / (255 * 3);
            if (sBright > 0.3) {
              const dist = Math.sqrt(dx * dx + dy * dy);
              const weight = Math.max(0, 1 - dist / glowRadius);
              glowR += (sr / 255) * weight * sBright;
              glowG += (sg / 255) * weight * sBright;
              glowB += (sb / 255) * weight * sBright;
              glowSum += weight;
            }
          }
        }
        if (glowSum > 0) {
          tr += (glowR / glowSum) * brightness * 1.5;
          tg += (glowG / glowSum) * brightness * 1.5;
          tb += (glowB / glowSum) * brightness * 1.5;
        }
      }

      for (const light of lights) {
        const dx = (x - light.x) / scale;
        const dy = (y - light.y) / scale;
        const distSq = dx * dx + dy * dy;
        const glow = (light.size * light.size) / Math.max(0.0001, distSq);
        const cappedGlow = Math.min(1, glow);
        tr += cappedGlow * light.r;
        tg += cappedGlow * light.g;
        tb += cappedGlow * light.b;
      }

      setPixel(
        out,
        x,
        y,
        Math.min(255, Math.floor(tr * 255)),
        Math.min(255, Math.floor(tg * 255)),
        Math.min(255, Math.floor(tb * 255)),
      );
    }
  }

  return out;
}

export { neon };
