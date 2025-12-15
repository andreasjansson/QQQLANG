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

function verticalSplit(ctx: FnContext, old: Image): Image {
  const prev = getPrevImage(ctx);
  const out = createSolidImage(ctx.width, ctx.height, "#000000");

  const midX = Math.floor(ctx.width / 2);
  const blendWidth = Math.floor(ctx.width * 0.1);

  for (let y = 0; y < ctx.height; y++) {
    const waveOffset = Math.sin(y * 0.05) * 15;
    const effectiveMidX = midX + waveOffset;

    for (let x = 0; x < ctx.width; x++) {
      const [pr, pg, pb] = getPixel(prev, x, y);
      const [or, og, ob] = getPixel(old, x, y);

      if (x < effectiveMidX - blendWidth) {
        setPixel(out, x, y, pr, pg, pb);
      } else if (x > effectiveMidX + blendWidth) {
        setPixel(out, x, y, or, og, ob);
      } else {
        const t = (x - (effectiveMidX - blendWidth)) / (blendWidth * 2);

        const screenR = 255 - ((255 - pr) * (255 - or)) / 255;
        const screenG = 255 - ((255 - pg) * (255 - og)) / 255;
        const screenB = 255 - ((255 - pb) * (255 - ob)) / 255;

        const diffR = Math.abs(pr - or);
        const diffG = Math.abs(pg - og);
        const diffB = Math.abs(pb - ob);

        const xorR = pr ^ or;
        const xorG = pg ^ og;
        const xorB = pb ^ ob;

        const band = Math.floor(y / 20) % 3;

        let r: number, g: number, b: number;

        if (band === 0) {
          if (t < 0.5) {
            const localT = t * 2;
            r = pr * (1 - localT) + screenR * localT;
            g = pg * (1 - localT) + screenG * localT;
            b = pb * (1 - localT) + screenB * localT;
          } else {
            const localT = (t - 0.5) * 2;
            r = screenR * (1 - localT) + or * localT;
            g = screenG * (1 - localT) + og * localT;
            b = screenB * (1 - localT) + ob * localT;
          }
        } else if (band === 1) {
          const centerDist = Math.abs(t - 0.5) * 2;
          const diffWeight = 1 - centerDist;
          r = (pr * (1 - t) + or * t) * (1 - diffWeight) + diffR * diffWeight;
          g = (pg * (1 - t) + og * t) * (1 - diffWeight) + diffG * diffWeight;
          b = (pb * (1 - t) + ob * t) * (1 - diffWeight) + diffB * diffWeight;
        } else {
          const xorWeight = Math.sin(t * Math.PI) * 0.7;
          const baseR = pr * (1 - t) + or * t;
          const baseG = pg * (1 - t) + og * t;
          const baseB = pb * (1 - t) + ob * t;
          r = baseR * (1 - xorWeight) + xorR * xorWeight;
          g = baseG * (1 - xorWeight) + xorG * xorWeight;
          b = baseB * (1 - xorWeight) + xorB * xorWeight;
        }

        setPixel(out, x, y, Math.round(r), Math.round(g), Math.round(b));
      }
    }
  }

  return out;
}

export { verticalSplit };
