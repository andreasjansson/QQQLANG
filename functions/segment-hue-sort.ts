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

function segmentHueSort(ctx: FnContext): Image {
  const prev = getPrevImage(ctx);
  const out = createSolidImage(ctx.width, ctx.height, "#000000");
  const w = ctx.width;
  const h = ctx.height;

  const threshold = 12;

  const labels = new Int32Array(w * h);
  labels.fill(-1);

  const colorDist = (
    x1: number,
    y1: number,
    x2: number,
    y2: number,
  ): number => {
    const [r1, g1, b1] = getPixel(prev, x1, y1);
    const [r2, g2, b2] = getPixel(prev, x2, y2);
    return Math.abs(r1 - r2) + Math.abs(g1 - g2) + Math.abs(b1 - b2);
  };

  let currentLabel = 0;

  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const idx = y * w + x;
      if (labels[idx] !== -1) continue;

      const queue: [number, number][] = [[x, y]];
      labels[idx] = currentLabel;

      while (queue.length > 0) {
        const [cx, cy] = queue.shift()!;

        const neighbors = [
          [cx - 1, cy],
          [cx + 1, cy],
          [cx, cy - 1],
          [cx, cy + 1],
        ];

        for (const [nx, ny] of neighbors) {
          if (nx < 0 || nx >= w || ny < 0 || ny >= h) continue;
          const nidx = ny * w + nx;
          if (labels[nidx] !== -1) continue;

          if (colorDist(cx, cy, nx, ny) < threshold) {
            labels[nidx] = currentLabel;
            queue.push([nx, ny]);
          }
        }
      }

      currentLabel++;
    }
  }

  const segments = new Map<
    number,
    { x: number; y: number; r: number; g: number; b: number; hue: number }[]
  >();

  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const label = labels[y * w + x];
      const [r, g, b] = getPixel(prev, x, y);
      const [hue] = rgbToHsl(r, g, b);

      if (!segments.has(label)) {
        segments.set(label, []);
      }
      segments.get(label)!.push({ x, y, r, g, b, hue });
    }
  }

  for (const [, pixels] of segments) {
    pixels.sort((a, b) => a.hue - b.hue);

    const positions = pixels.map((p) => ({ x: p.x, y: p.y }));
    positions.sort((a, b) => (a.x !== b.x ? a.x - b.x : a.y - b.y));

    for (let i = 0; i < pixels.length; i++) {
      const color = pixels[i];
      const pos = positions[i];
      setPixel(out, pos.x, pos.y, color.r, color.g, color.b);
    }
  }

  return out;
}

export { segmentHueSort };
