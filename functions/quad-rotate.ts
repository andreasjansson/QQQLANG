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

function quadRotate(ctx: FnContext): Image {
  const prev = getPrevImage(ctx);
  const out = createSolidImage(ctx.width, ctx.height, "#000000");

  const hw = Math.floor(ctx.width / 2);
  const hh = Math.floor(ctx.height / 2);

  const rotatePoint = (
    x: number,
    y: number,
    cx: number,
    cy: number,
    angle: number,
  ): [number, number] => {
    const cos = Math.cos(angle);
    const sin = Math.sin(angle);
    const dx = x - cx;
    const dy = y - cy;
    return [cx + dx * cos - dy * sin, cy + dx * sin + dy * cos];
  };

  for (let y = 0; y < ctx.height; y++) {
    for (let x = 0; x < ctx.width; x++) {
      const qx = x < hw ? 0 : 1;
      const qy = y < hh ? 0 : 1;
      const quadrant = qy * 2 + qx;
      const angles = [0, Math.PI / 2, Math.PI, (Math.PI * 3) / 2];
      const angle = angles[quadrant];

      const localCx = qx === 0 ? hw / 2 : hw + hw / 2;
      const localCy = qy === 0 ? hh / 2 : hh + hh / 2;

      const [srcX, srcY] = rotatePoint(x, y, localCx, localCy, -angle);
      const [r, g, b] = getPixel(prev, Math.floor(srcX), Math.floor(srcY));
      setPixel(out, x, y, r, g, b);
    }
  }

  return out;
}

export { quadRotate };
