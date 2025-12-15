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

function hslShift(ctx: FnContext, n: number): Image {
  const prev = getPrevImage(ctx);
  const w = ctx.width;
  const h = ctx.height;

  const hueAmount = 90 + (n % 12) * 15;
  const satAmount = 0.5 + (n % 8) * 0.1;
  const lightAmount = 0.06 + (n % 8) * 0.01;

  const baseAngle = n * 0.5;
  const angleH = baseAngle;
  const angleS = baseAngle + (Math.PI * 2) / 3;
  const angleL = baseAngle + (Math.PI * 4) / 3;

  const dirHX = Math.cos(angleH);
  const dirHY = Math.sin(angleH);
  const dirSX = Math.cos(angleS);
  const dirSY = Math.sin(angleS);
  const dirLX = Math.cos(angleL);
  const dirLY = Math.sin(angleL);

  const cx = w / 2;
  const cy = h / 2;
  const maxDist = Math.sqrt(cx * cx + cy * cy);

  const out = createSolidImage(w, h, "#000000");

  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const [origR, origG, origB] = getPixel(prev, x, y);
      let [oh, os, ol] = rgbToHsl(origR, origG, origB);

      const rx = x - cx;
      const ry = y - cy;

      const tH = (rx * dirHX + ry * dirHY) / maxDist;
      const tS = (rx * dirSX + ry * dirSY) / maxDist;
      const tL = (rx * dirLX + ry * dirLY) / maxDist;

      const hueShift = tH * hueAmount;
      let nh = (oh + hueShift + 360) % 360;

      const satMod = 1 + Math.abs(tS) * satAmount;
      let ns = Math.min(1, os * satMod);

      const midtoneFactor = 1 - Math.pow(Math.abs(ol - 0.5) * 2, 2);
      const lightShift = tL * lightAmount * midtoneFactor;
      let nl = Math.max(0.05, Math.min(0.95, ol + lightShift));

      const [r, g, b] = hslToRgb(nh, ns, nl);
      setPixel(out, x, y, r, g, b);
    }
  }

  return out;
}

export { hslShift };
