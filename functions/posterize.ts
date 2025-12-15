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

function posterize(ctx: FnContext): Image {
  const prev = getPrevImage(ctx);
  const out = cloneImage(prev);

  const levels = 4;

  for (let i = 0; i < out.data.length; i += 4) {
    out.data[i] =
      Math.floor((out.data[i] / 256) * levels) * (255 / (levels - 1));
    out.data[i + 1] =
      Math.floor((out.data[i + 1] / 256) * levels) * (255 / (levels - 1));
    out.data[i + 2] =
      Math.floor((out.data[i + 2] / 256) * levels) * (255 / (levels - 1));
  }

  return out;
}

export { posterize };
