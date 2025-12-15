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

function fur(ctx: FnContext): Image {
  const prev = getPrevImage(ctx);
  const out = cloneImage(prev);

  const hash = (x: number, y: number, seed: number) => {
    const v = Math.sin(x * 127.1 + y * 311.7 + seed * 113.3) * 43758.5453;
    return v - Math.floor(v);
  };

  const n = 60;
  const baseLen = 8 + n * 5;
  const chaos = 0.5 + n * 0.15;
  const cx = ctx.width / 2;
  const cy = ctx.height / 2;

  for (let y = 0; y < ctx.height; y++) {
    for (let x = 0; x < ctx.width; x++) {
      const [r, g, b] = getPixel(prev, x, y);
      const [h, s, l] = rgbToHsl(r, g, b);

      if (s < 0.03 && l < 0.03) continue;
      if (l > 0.97) continue;

      const hueAngle = (h / 360) * Math.PI * 2;

      const turbulence = (hash(x, y, 1) - 0.5) * Math.PI * chaos;
      const swirl = Math.sin(x * 0.02) * Math.cos(y * 0.02) * 0.5;

      const angle = hueAngle + turbulence + swirl;

      const lenNoise = 0.3 + hash(x, y, 2) * 0.7;
      const len = baseLen * s * (0.2 + l * 0.8) * lenNoise;

      if (len < 2) continue;

      const dx = Math.cos(angle);
      const dy = Math.sin(angle);

      for (let i = 1; i <= len; i++) {
        const wobble = Math.sin(i * 0.5) * hash(x, y, 3) * 2;
        const px = Math.floor(x + dx * i + dy * wobble);
        const py = Math.floor(y + dy * i - dx * wobble);

        if (px >= 0 && px < ctx.width && py >= 0 && py < ctx.height) {
          const fade = 1 - i / len;
          const brightness = fade * fade;
          const idx = (py * ctx.width + px) * 4;

          out.data[idx] = Math.min(
            255,
            Math.floor(
              out.data[idx] * (1 - brightness * 0.8) + r * brightness * 0.8,
            ),
          );
          out.data[idx + 1] = Math.min(
            255,
            Math.floor(
              out.data[idx + 1] * (1 - brightness * 0.8) + g * brightness * 0.8,
            ),
          );
          out.data[idx + 2] = Math.min(
            255,
            Math.floor(
              out.data[idx + 2] * (1 - brightness * 0.8) + b * brightness * 0.8,
            ),
          );
        }
      }
    }
  }

  return out;
}

export { fur };
