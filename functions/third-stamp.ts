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

function thirdStamp(
  ctx: FnContext,
  old: Image,
  oldThird: number,
  prevThird: number,
): Image {
  const prev = getPrevImage(ctx);
  const out = cloneImage(prev);

  const oldIdx = (((oldThird - 1) % 3) + 3) % 3;
  const prevIdx = (((prevThird - 1) % 3) + 3) % 3;

  const thirdWidth = Math.floor(ctx.width / 3);

  const oldStartX = oldIdx * thirdWidth;
  const oldEndX = oldIdx === 2 ? ctx.width : (oldIdx + 1) * thirdWidth;

  const prevStartX = prevIdx * thirdWidth;
  const prevEndX = prevIdx === 2 ? ctx.width : (prevIdx + 1) * thirdWidth;

  const oldWidth = oldEndX - oldStartX;
  const prevWidth = prevEndX - prevStartX;

  for (let y = 0; y < ctx.height; y++) {
    for (let x = prevStartX; x < prevEndX; x++) {
      const srcX =
        oldStartX + Math.floor(((x - prevStartX) / prevWidth) * oldWidth);
      const [r, g, b] = getPixel(old, srcX, y);
      setPixel(out, x, y, r, g, b);
    }
  }

  return out;
}

export { thirdStamp };
