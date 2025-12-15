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

function pointillism(ctx: FnContext, n: number): Image {
  const prev = getPrevImage(ctx);
  const out = createSolidImage(ctx.width, ctx.height, "#000000");
  const radius = Math.max(2, Math.floor(n / 2) + 1);
  const diameter = radius * 2;

  for (let cy = radius; cy < ctx.height; cy += diameter) {
    for (let cx = radius; cx < ctx.width; cx += diameter) {
      const [r, g, b] = getPixel(prev, cx, cy);
      const [h, s, l] = rgbToHsl(r, g, b);
      const [nr, ng, nb] = hslToRgb(h, Math.min(1, s + 0.1), l);

      for (let dy = -radius; dy <= radius; dy++) {
        for (let dx = -radius; dx <= radius; dx++) {
          const dist = Math.sqrt(dx * dx + dy * dy);

          if (dist <= radius) {
            const px = cx + dx;
            const py = cy + dy;

            if (px >= 0 && px < ctx.width && py >= 0 && py < ctx.height) {
              const edge = radius - 0.5;
              const alpha =
                dist < edge ? 1 : Math.max(0, 1 - (dist - edge) * 2);

              if (alpha > 0) {
                const [br, bg, bb] = getPixel(out, px, py);
                const finalR = Math.round(br * (1 - alpha) + nr * alpha);
                const finalG = Math.round(bg * (1 - alpha) + ng * alpha);
                const finalB = Math.round(bb * (1 - alpha) + nb * alpha);
                setPixel(out, px, py, finalR, finalG, finalB);
              }
            }
          }
        }
      }
    }
  }

  return out;
}

export { pointillism };
