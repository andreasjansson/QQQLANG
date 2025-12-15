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

function variableCheckerboard(ctx: FnContext, old: Image): Image {
  const prev = getPrevImage(ctx);
  const out = createSolidImage(ctx.width, ctx.height, "#000000");

  for (let y = 0; y < ctx.height; y++) {
    for (let x = 0; x < ctx.width; x++) {
      const nx = x / ctx.width;
      const ny = y / ctx.height;

      const squareSize = Math.max(2, Math.floor(2 + (nx + ny) * 50));

      const gridX = Math.floor(x / squareSize);
      const gridY = Math.floor(y / squareSize);

      const useOld = (gridX + gridY) % 2 === 0;
      const src = useOld ? old : prev;

      const [r, g, b] = getPixel(src, x, y);
      setPixel(out, x, y, r, g, b);
    }
  }

  return out;
}

export { variableCheckerboard };
