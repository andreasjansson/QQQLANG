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

function rgbToYuv(r: number, g: number, b: number): [number, number, number] {
  const rn = r / 255;
  const gn = g / 255;
  const bn = b / 255;

  const y = 0.299 * rn + 0.587 * gn + 0.114 * bn;
  const u = -0.14713 * rn - 0.28886 * gn + 0.436 * bn;
  const v = 0.615 * rn - 0.51499 * gn - 0.10001 * bn;

  return [y, u, v];
}

function yuvToRgb(y: number, u: number, v: number): [number, number, number] {
  const r = y + 1.13983 * v;
  const g = y - 0.39465 * u - 0.58060 * v;
  const b = y + 2.03211 * u;

  return [
    Math.max(0, Math.min(255, Math.round(r * 255))),
    Math.max(0, Math.min(255, Math.round(g * 255))),
    Math.max(0, Math.min(255, Math.round(b * 255))),
  ];
}

function hslShift(ctx: FnContext, n: number): Image {
  const prev = getPrevImage(ctx);
  const w = ctx.width;
  const h = ctx.height;

  const yAmount = 0.15 + (n % 8) * 0.03;
  const uAmount = 0.2 + (n % 10) * 0.05;
  const vAmount = 0.2 + (n % 10) * 0.05;

  const baseAngle = n * 0.5;
  const angleY = baseAngle;
  const angleU = baseAngle + (Math.PI * 2) / 3;
  const angleV = baseAngle + (Math.PI * 4) / 3;

  const dirYX = Math.cos(angleY);
  const dirYY = Math.sin(angleY);
  const dirUX = Math.cos(angleU);
  const dirUY = Math.sin(angleU);
  const dirVX = Math.cos(angleV);
  const dirVY = Math.sin(angleV);

  const cx = w / 2;
  const cy = h / 2;
  const maxDist = Math.sqrt(cx * cx + cy * cy);

  const out = createSolidImage(w, h, "#000000");

  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const [origR, origG, origB] = getPixel(prev, x, y);
      let [oy, ou, ov] = rgbToYuv(origR, origG, origB);

      const rx = x - cx;
      const ry = y - cy;

      const tY = (rx * dirYX + ry * dirYY) / maxDist;
      const tU = (rx * dirUX + ry * dirUY) / maxDist;
      const tV = (rx * dirVX + ry * dirVY) / maxDist;

      const yShift = tY * yAmount;
      let ny = Math.max(0, Math.min(1, oy + yShift));

      const uShift = tU * uAmount;
      let nu = Math.max(-0.436, Math.min(0.436, ou + uShift));

      const vShift = tV * vAmount;
      let nv = Math.max(-0.615, Math.min(0.615, ov + vShift));

      const [r, g, b] = yuvToRgb(ny, nu, nv);
      setPixel(out, x, y, r, g, b);
    }
  }

  return out;
}

export { hslShift };
