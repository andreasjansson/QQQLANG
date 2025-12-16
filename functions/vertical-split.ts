import {
  FnContext,
  Image,
  createSolidImage,
  getPrevImage,
  getPixel,
  setPixel,
  getOldImage,
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

        const band = Math.floor(y / 20) % 3;

        let r: number, g: number, b: number;

        // Base linear blend
        const baseR = pr * (1 - t) + or * t;
        const baseG = pg * (1 - t) + og * t;
        const baseB = pb * (1 - t) + ob * t;

        if (band === 0) {
          // Screen blend - but normalized so same image = no change
          const screenR = 255 - ((255 - pr) * (255 - or)) / 255;
          const screenG = 255 - ((255 - pg) * (255 - og)) / 255;
          const screenB = 255 - ((255 - pb) * (255 - ob)) / 255;
          
          // When pr=or, screen gives: 255 - (255-pr)^2/255, which is brighter than pr
          // Normalize: subtract the "self-screen" brightness boost
          const selfScreenR = 255 - ((255 - pr) * (255 - pr)) / 255;
          const selfScreenG = 255 - ((255 - pg) * (255 - pg)) / 255;
          const selfScreenB = 255 - ((255 - pb) * (255 - pb)) / 255;
          
          const boostR = selfScreenR - pr;
          const boostG = selfScreenG - pg;
          const boostB = selfScreenB - pb;
          
          // Blend weight peaks at center
          const blendWeight = Math.sin(t * Math.PI);
          
          r = baseR + (screenR - baseR - boostR) * blendWeight;
          g = baseG + (screenG - baseG - boostG) * blendWeight;
          b = baseB + (screenB - baseB - boostB) * blendWeight;
        } else if (band === 1) {
          // Overlay blend - normalized
          const overlayR = pr < 128 
            ? (2 * pr * or) / 255 
            : 255 - (2 * (255 - pr) * (255 - or)) / 255;
          const overlayG = pg < 128 
            ? (2 * pg * og) / 255 
            : 255 - (2 * (255 - pg) * (255 - og)) / 255;
          const overlayB = pb < 128 
            ? (2 * pb * ob) / 255 
            : 255 - (2 * (255 - pb) * (255 - ob)) / 255;
          
          const blendWeight = Math.sin(t * Math.PI) * 0.5;
          
          // Overlay with same image = original, so just blend toward overlay result
          r = baseR + (overlayR - baseR) * blendWeight;
          g = baseG + (overlayG - baseG) * blendWeight;
          b = baseB + (overlayB - baseB) * blendWeight;
        } else {
          // Soft light blend
          const softR = or < 128
            ? pr - (255 - 2 * or) * pr * (255 - pr) / (255 * 255)
            : pr + (2 * or - 255) * (Math.sqrt(pr / 255) * 255 - pr) / 255;
          const softG = og < 128
            ? pg - (255 - 2 * og) * pg * (255 - pg) / (255 * 255)
            : pg + (2 * og - 255) * (Math.sqrt(pg / 255) * 255 - pg) / 255;
          const softB = ob < 128
            ? pb - (255 - 2 * ob) * pb * (255 - pb) / (255 * 255)
            : pb + (2 * ob - 255) * (Math.sqrt(pb / 255) * 255 - pb) / 255;
          
          const blendWeight = Math.sin(t * Math.PI) * 0.5;
          
          r = baseR + (softR - baseR) * blendWeight;
          g = baseG + (softG - baseG) * blendWeight;
          b = baseB + (softB - baseB) * blendWeight;
        }

        r = Math.max(0, Math.min(255, r));
        g = Math.max(0, Math.min(255, g));
        b = Math.max(0, Math.min(255, b));

        setPixel(out, x, y, Math.round(r), Math.round(g), Math.round(b));
      }
    }
  }

  return out;
}

export { verticalSplit };
