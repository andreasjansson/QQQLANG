import type {
  CharDef,
  ArgDef,
  FnContext,
  Image,
  ArgType,
  IntType,
  ColorType,
  IndexType,
  ChoiceType,
  OpInfo,
} from "./functions/helpers.js";

import {
  INT,
  COLOR,
  INDEX,
  Choice,
  createSolidImage,
  getPrevImage,
  getPixel,
  setPixel,
  cloneImage,
  hexToRgb,
  rgbToHsl,
  hslToRgb,
  getOldImage,
  UPLOAD_CHAR,
  UPLOAD_COUNT,
  isIndexedUpload,
  isInvalidUpload,
  isAnyUpload,
  getUploadIndex,
  getUploadChar,
  getInvalidUploadChar,
  createPlaceholderImage,
  emeraldReady,
  bgRemovalReady,
} from "./functions/helpers.js";

import { spheres } from "./functions/spheres.js";
import { border } from "./functions/border.js";
import { concentricHue } from "./functions/concentric-hue.js";
import { drip } from "./functions/drip.js";
import { emerald } from "./functions/emerald.js";
import { fftOverflow } from "./functions/fft-overflow.js";
import { grayscaleColorize } from "./functions/grayscale-colorize.js";
import { hourglass } from "./functions/hourglass.js";
import { invertEdges } from "./functions/invert-edges.js";
import { juliaFractal } from "./functions/julia-fractal.js";
import { kaleidoscope } from "./functions/kaleidoscope.js";
import { lissajous } from "./functions/lissajous.js";
import { moire } from "./functions/moire.js";
import { neon } from "./functions/neon.js";
import { oilSlick } from "./functions/oil-slick.js";
import { diagonalPixelate } from "./functions/diagonal-pixelate.js";
import { prism } from "./functions/prism.js";
import { room } from "./functions/room.js";
import { sierpinski } from "./functions/sierpinski.js";
import { tiles } from "./functions/tiles.js";
import { dither } from "./functions/dither.js";
import { voronoi } from "./functions/voronoi.js";
import { swirl } from "./functions/swirl.js";
import { cppn } from "./functions/cppn.js";
import { hslShift } from "./functions/yuv-shift.js";
import { zoomBlur } from "./functions/zoom-blur.js";
import { bgRemove } from "./functions/bg-remove.js";
import { colorize } from "./functions/colorize.js";
import { thirdStamp } from "./functions/third-stamp.js";
import { tripleRotate } from "./functions/triple-rotate.js";
import { quadRotate } from "./functions/quad-rotate.js";
import { triangularSplit } from "./functions/triangular-split.js";
import { posterize } from "./functions/posterize.js";
import { chromatic } from "./functions/chromatic.js";
import { lemniscate } from "./functions/lemniscate.js";
import { xorBlend } from "./functions/xor-blend.js";
import { shift } from "./functions/shift.js";
import { rotate90 } from "./functions/rotate-90.js";
import { shiftUp } from "./functions/shift-up.js";
import { godrays } from "./functions/godrays.js";
import { bandTransform } from "./functions/band-transform.js";
import { insert } from "./functions/insert.js";
import { segmentHueSort } from "./functions/segment-hue-sort.js";
import { flip } from "./functions/flip.js";
import { quadtreeCompress } from "./functions/quadtree-compress.js";
import { variableCheckerboard } from "./functions/variable-checkerboard.js";
import { shearRadial } from "./functions/shear-radial.js";
import { blur } from "./functions/blur.js";
import { fur } from "./functions/fur.js";
import { zoom } from "./functions/zoom.js";
import { stipple } from "./functions/stipple.js";
import { blend } from "./functions/blend.js";
import { pointillism } from "./functions/pointillism.js";
import { circleStamp } from "./functions/circle-stamp.js";
import { porthole } from "./functions/porthole.js";
import { semicircleReflect } from "./functions/semicircle-reflect.js";
import { shiftedStripes } from "./functions/shifted-stripes.js";
import { help } from "./functions/help.js";
import { cond } from "./functions/cond.js";
import { rotate } from "./functions/rotate.js";
import { composite } from "./functions/composite.js";
import { leftHalfOffset } from "./functions/left-half-offset.js";
import { scanlines } from "./functions/scanlines.js";
import { rule110 } from "./functions/rule110.js";
import { skew } from "./functions/skew.js";
import { verticalSplit } from "./functions/vertical-split.js";
import { sharpen } from "./functions/sharpen.js";
import { waveChromatic } from "./functions/wave-chromatic.js";

export const characterDefs: Record<string, CharDef> = {
  A: {
    color: "#78A10F",
    number: 1,
    fn: spheres,
    args: [],
    functionName: "spheres",
    documentation:
      "Flips prev horizontally, then renders as texture on two 3D spheres with lighting.",
  },

  B: {
    color: "#8B4513",
    number: 2,
    fn: border,
    args: [
      {
        type: Choice(
          "circular-solid",
          "circular-blur",
          "horizontal-solid",
          "horizontal-blur",
          "vertical-solid",
          "vertical-blur",
          "rectangular-solid",
          "rectangular-blur",
          "diamond-solid",
          "diamond-blur",
          "hexagon-solid",
          "hexagon-blur",
          "sine-horizontal-solid",
          "sine-horizontal-blur",
          "sine-vertical-solid",
          "sine-vertical-blur",
          "triangle-horizontal-solid",
          "triangle-horizontal-blur",
          "triangle-vertical-solid",
          "triangle-vertical-blur",
        ),
        documentation: "Border shape style",
      },
      { type: COLOR, documentation: "Border color" },
    ],
    functionName: "border",
    documentation:
      "Border effect with various shapes: circular, horizontal, vertical, rectangular, diamond, hexagon, sine waves, and triangle waves.",
  },

  C: {
    color: "#FF6B35",
    number: 3,
    fn: concentricHue,
    args: [{ type: INT, documentation: "Number of concentric circles" }],
    functionName: "concentric-hue",
    documentation: "Alternating original and hue-shifted concentric circles.",
  },

  D: {
    color: "#FF1493",
    number: 4,
    fn: drip,
    args: [],
    functionName: "drip",
    documentation: "Metaball-based dripping water drops effect.",
  },

  E: {
    color: "#50C878",
    number: 5,
    fn: emerald,
    args: [],
    functionName: "emerald",
    documentation: "Renders reflective 3D emeralds in symmetric pattern.",
  },

  F: {
    color: "#FFD700",
    number: 6,
    fn: fftOverflow,
    args: [{ type: INT, documentation: "FFT multiplier strength" }],
    functionName: "fft-overflow",
    documentation: "2D FFT with magnitude overflow and chromatic phase shifts.",
  },

  G: {
    color: "#9370DB",
    number: 7,
    fn: grayscaleColorize,
    args: [{ type: INT, documentation: "Number of posterize colors" }],
    functionName: "grayscale-colorize",
    documentation: "Converts to grayscale then applies rainbow palette.",
  },

  H: {
    color: "#DC143C",
    number: 8,
    fn: hourglass,
    args: [],
    functionName: "hourglass",
    documentation: "Hourglass gradient with bitwise color blending.",
  },

  I: {
    color: "#00FF7F",
    number: 9,
    fn: invertEdges,
    args: [],
    functionName: "invert-edges",
    documentation: "Inverts colors then adds Sobel edge detection.",
  },

  J: {
    color: "#FF8C00",
    number: 10,
    fn: juliaFractal,
    args: [{ type: INT, documentation: "Fractal zoom depth" }],
    functionName: "julia-fractal",
    documentation: "Julia set fractal masking the previous image.",
  },

  K: {
    color: "#9966FF",
    number: 11,
    fn: kaleidoscope,
    args: [{ type: INT, documentation: "Number of kaleidoscope segments" }],
    functionName: "kaleidoscope",
    documentation: "N-way kaleidoscope effect with zoom.",
  },

  L: {
    color: "#20B2AA",
    number: 12,
    fn: lissajous,
    args: [
      { type: INDEX, documentation: "Old image for tube texture" },
      { type: INT, documentation: "Rotation angle multiplier" },
    ],
    functionName: "lissajous",
    documentation: "3D Lissajous tube with textured surface.",
  },

  M: {
    color: "#FF69B4",
    number: 13,
    fn: moire,
    args: [{ type: INT, documentation: "Pattern complexity seed" }],
    functionName: "moire",
    documentation: "Moiré interference pattern with color zones.",
  },

  N: {
    color: "#8A2BE2",
    number: 14,
    fn: neon,
    args: [],
    functionName: "neon",
    documentation: "Neon glow effect on bright edges.",
  },

  O: {
    color: "#FF6347",
    number: 15,
    fn: oilSlick,
    args: [{ type: INT, documentation: "Warp intensity and depth" }],
    functionName: "oil-slick",
    documentation: "Domain warping with iridescent lighting.",
  },

  P: {
    color: "#4682B4",
    number: 16,
    fn: diagonalPixelate,
    args: [{ type: INT, documentation: "Pixel cell size" }],
    functionName: "pixelate",
    documentation:
      "Pixelate with diagonal split using average/saturated colors.",
  },

  Q: {
    color: "#32CD32",
    number: 17,
    fn: prism,
    args: [],
    functionName: "quad-prism",
    documentation: "Negative prism with diagonal inversion and mirroring.",
  },

  R: {
    color: "#DA70D6",
    number: 18,
    fn: room,
    args: [],
    functionName: "room",
    documentation: "3D room with textured walls, ceiling, and floor.",
  },

  S: {
    color: "#87CEEB",
    number: 19,
    fn: sierpinski,
    args: [
      { type: INDEX, documentation: "Old image for triangle interior" },
      { type: INT, documentation: "Fractal detail level (A-~)" },
    ],
    functionName: "sierpinski",
    documentation: "Sierpiński triangle fractal with color effects.",
  },

  T: {
    color: "#F0E68C",
    number: 20,
    fn: tiles,
    args: [
      {
        type: INT,
        documentation: "Building height multiplier (A=short, ~=tall)",
      },
    ],
    functionName: "tiles",
    documentation:
      "Grid of 3D tiles covering entire canvas, heights based on seed with multiplier.",
  },

  U: {
    color: "#DDA0DD",
    number: 21,
    fn: dither,
    args: [
      {
        type: Choice(
          "ordered-5level",
          "bayer-bw",
          "threshold-bw",
          "ordered-2bit",
          "floyd-rgb",
          "floyd-bw",
          "atkinson-4level",
          "atkinson-bw",
          "stucki-6level",
          "burkes",
          "sierra",
          "random-bw",
          "cluster-2bit",
          "bluenoise-bw",
          "bayer2x2-2bit",
          "noise-2bit",
        ),
        documentation: "Dithering algorithm",
      },
    ],
    functionName: "dither",
    documentation:
      "Apply one of 16 dithering algorithms, from subtle to aggressive 2-color modes.",
  },

  V: {
    color: "#40E0D0",
    number: 22,
    fn: voronoi,
    args: [
      { type: INDEX, documentation: "Old image to alternate with" },
      { type: INT, documentation: "Pattern shape" },
    ],
    functionName: "voronoi",
    documentation:
      "Voronoi cells alternating between current and old image, with variable pattern shape.",
  },

  W: {
    color: "#EE82EE",
    number: 23,
    fn: swirl,
    args: [{ type: INT, documentation: "Rotation multiplier (×20°)" }],
    functionName: "whirl",
    documentation: "Swirl distortion from center with quadratic falloff.",
  },

  X: {
    color: "#F5DEB3",
    number: 24,
    fn: cppn,
    args: [{ type: INT, documentation: "Effect strength" }],
    functionName: "cppn",
    documentation:
      "Compositional Pattern Producing Network warps and modulates saturation/value using a neural network. Fully deterministic based on input.",
  },

  Y: {
    color: "#98FB98",
    number: 25,
    fn: hslShift,
    args: [
      {
        type: INT,
        documentation: "Controls angle and intensity of color shifts",
      },
    ],
    functionName: "yuv-shift",
    documentation:
      "YUV color shift with three gradient directions 120° apart: luminance, blue chrominance, and red chrominance shifts.",
  },

  Z: {
    color: "#AFEEEE",
    number: 26,
    fn: zoomBlur,
    args: [{ type: INT, documentation: "Blur strength multiplier (×4px)" }],
    functionName: "zoom-blur",
    documentation: "Radial motion blur from center with sharp center.",
  },

  "0": {
    color: "#E6E6FA",
    number: 27,
    fn: bgRemove,
    args: [
      { type: INDEX, documentation: "Background image to composite behind" },
    ],
    functionName: "bg-remove",
    documentation:
      "Removes background from prev image using ML, composites on specified background.",
  },

  "1": {
    color: "#FFA07A",
    number: 28,
    fn: colorize,
    args: [
      { type: COLOR, documentation: "Tint color applied based on luminance" },
    ],
    functionName: "colorize",
    documentation:
      "Tints the image with the specified color using gamma-corrected luminance and boosted saturation.",
  },

  "2": {
    color: "#98D8C8",
    number: 29,
    fn: thirdStamp,
    args: [
      { type: INDEX, documentation: "Old image to extract third from" },
      {
        type: INT,
        documentation: "Old image third (1=left, 2=mid, 3=right, cycling)",
      },
      {
        type: INT,
        documentation:
          "Current image third to replace (1=left, 2=mid, 3=right, cycling)",
      },
    ],
    functionName: "third-stamp",
    documentation:
      "Replace a vertical third of current image with a third from old image.",
  },

  "3": {
    color: "#F7DC6F",
    number: 30,
    fn: tripleRotate,
    args: [],
    functionName: "triple-rotate",
    documentation: "Three vertical strips with different rotations.",
  },

  "4": {
    color: "#BB8FCE",
    number: 31,
    fn: quadRotate,
    args: [],
    functionName: "quad-rotate",
    documentation: "Four quadrants each rotated 0°, 90°, 180°, 270°.",
  },

  "5": {
    color: "#85C1E9",
    number: 32,
    fn: triangularSplit,
    args: [{ type: INT, documentation: "Cell size multiplier" }],
    functionName: "triangular-split",
    documentation: "Triangular grid with hue shifts and lightness variation.",
  },

  "6": {
    color: "#F1948A",
    number: 33,
    fn: posterize,
    args: [],
    functionName: "posterize",
    documentation: "Posterize to 4 levels per channel.",
  },

  "7": {
    color: "#82E0AA",
    number: 34,
    fn: chromatic,
    args: [],
    functionName: "chromatic",
    documentation: "Chromatic aberration with RGB channel shifts.",
  },

  "8": {
    color: "#F8C471",
    number: 35,
    fn: lemniscate,
    args: [{ type: INT, documentation: "Distortion strength" }],
    functionName: "lemniscate",
    documentation: "Infinity-loop lemniscate distortion.",
  },

  "9": {
    color: "#D7BDE2",
    number: 36,
    fn: xorBlend,
    args: [{ type: INDEX, documentation: "Old image to XOR with" }],
    functionName: "xor-blend",
    documentation: "XOR blend creating glitchy digital artifacts.",
  },

  "<": {
    color: "#E74C3C",
    number: 37,
    fn: shiftLeft,
    args: [],
    functionName: "shift-left",
    documentation: "Horizontal shift 1/3 width left with wraparound.",
  },

  ">": {
    color: "#3498DB",
    number: 38,
    fn: rotate90,
    args: [],
    functionName: "rotate-90",
    documentation: "Rotate 90 degrees clockwise.",
  },

  "^": {
    color: "#2ECC71",
    number: 39,
    fn: shiftUp,
    args: [],
    functionName: "shift-up",
    documentation: "Vertical shift 1/3 height up with wraparound.",
  },

  "!": {
    color: "#FF4500",
    number: 40,
    fn: godrays,
    args: [],
    functionName: "godrays",
    documentation: "Volumetric light scattering from center.",
  },

  '"': {
    color: "#9932CC",
    number: 41,
    fn: bandTransform,
    args: [{ type: INT, documentation: "Number of horizontal bands" }],
    functionName: "band-transform",
    documentation:
      "Horizontal bands with alternating hue/saturation transforms.",
  },

  "#": {
    color: "#228B22",
    number: 42,
    fn: insert,
    args: [{ type: INDEX, documentation: "Old image index to insert" }],
    functionName: "insert",
    documentation: "Replaces current image with specified old image.",
  },

  $: {
    color: "#FFD700",
    number: 43,
    fn: segmentHueSort,
    args: [],
    functionName: "segment-hue-sort",
    documentation:
      "Color-based segmentation, then sorts pixels by hue within each segment.",
  },

  "%": {
    color: "#8B0000",
    number: 44,
    fn: flip,
    args: [
      {
        type: INT,
        documentation: "Flip direction (even=horizontal, odd=vertical)",
      },
    ],
    functionName: "flip",
    documentation:
      "Flips image horizontally or vertically based on argument parity.",
  },

  "&": {
    color: "#4169E1",
    number: 45,
    fn: quadtreeCompress,
    args: [
      {
        type: INT,
        documentation:
          "Compression level (A=minimal/detailed, ~=maximal/geometric blocks)",
      },
    ],
    functionName: "quadtree-compress",
    documentation:
      "Adaptive quadtree compression - detailed areas keep resolution while uniform areas become large blocks, creating geometric patterns.",
  },

  "'": {
    color: "#FF1493",
    number: 46,
    fn: variableCheckerboard,
    args: [{ type: INDEX, documentation: "Old image to checkerboard with" }],
    functionName: "variable-checkerboard",
    documentation:
      "Checkerboard blend with increasing square size from corner to corner.",
  },

  "(": {
    color: "#00CED1",
    number: 47,
    fn: shearRadial,
    args: [
      {
        type: INT,
        documentation: "Center X offset (A=left, M=center, ~=right)",
      },
      {
        type: INT,
        documentation: "Center Y offset (A=bottom, M=center, ~=top)",
      },
      {
        type: INT,
        documentation: "Radial/shear strength (A=barrel, M=none, ~=pincushion)",
      },
    ],
    functionName: "shear-radial",
    documentation:
      "Combined shear and radial distortion. Shear amount couples to horizontal offset and radial strength.",
  },

  ")": {
    color: "#FF69B4",
    number: 48,
    fn: blur,
    args: [{ type: INT, documentation: "Blur radius (A=subtle, ~=heavy)" }],
    functionName: "blur",
    documentation:
      "Gaussian blur with adjustable radius using two-pass convolution.",
  },

  "*": {
    color: "#FFD700",
    number: 49,
    fn: fur,
    args: [],
    functionName: "fur",
    documentation:
      "Fur/hair strands growing from pixels based on hue and noise.",
  },

  "+": {
    color: "#32CD32",
    number: 50,
    fn: zoom,
    args: [],
    functionName: "zoom",
    documentation: "Zoom in 1.2× from center.",
  },

  ",": {
    color: "#BA55D3",
    number: 51,
    fn: stipple,
    args: [{ type: COLOR, documentation: "Stipple dot color" }],
    functionName: "stipple",
    documentation: "Stipple dots at luminance-based positions.",
  },

  "-": {
    color: "#FF7F50",
    number: 52,
    fn: blend,
    args: [
      { type: INDEX, documentation: "Old image to blend with" },
      {
        type: Choice(
          "multiply",
          "screen",
          "overlay",
          "darken",
          "lighten",
          "dodge",
          "burn",
          "hardlight",
          "softlight",
          "difference",
          "exclusion",
          "add",
          "subtract",
          "xor",
          "and",
          "or",
          "nand",
          "nor",
          "xnor",
          "average",
          "divide",
          "grain-extract",
          "grain-merge",
          "vivid",
          "linear",
          "pin",
          "hardmix",
          "hue",
          "saturation",
          "color",
          "luminosity",
          "replace-dark-third",
          "replace-mid-third",
          "replace-light-third",
        ),
        documentation: "Blend mode",
      },
    ],
    functionName: "blend",
    documentation: "Blend old image with current using specified mode.",
  },

  ".": {
    color: "#20B2AA",
    number: 53,
    fn: pointillism,
    args: [{ type: INT, documentation: "Dot radius base (mod 8 + 2)" }],
    functionName: "pointillism",
    documentation: "Pointillism effect with saturated circular dots.",
  },

  "/": {
    color: "#CD853F",
    number: 54,
    fn: circleStamp,
    args: [
      { type: INDEX, documentation: "Old image source" },
      { type: INT, documentation: "X position (A=left, 7=center, ~=right)" },
      { type: INT, documentation: "Y position (A=top, 7=center, ~=bottom)" },
      { type: INT, documentation: "Circle size (A=tiny, ~=full)" },
      {
        type: Choice(
          "normal",
          "xor",
          "nand",
          "and",
          "or",
          "multiply",
          "screen",
          "overlay",
          "darken",
          "lighten",
          "difference",
          "exclusion",
          "add",
          "subtract",
          "hardlight",
          "softlight",
        ),
        documentation: "Blend mode",
      },
    ],
    functionName: "circle-stamp",
    documentation: "Stamp circular region from old image center onto current.",
  },

  ":": {
    color: "#6B8E23",
    number: 55,
    fn: porthole,
    args: [{ type: INDEX, documentation: "Old image for background" }],
    functionName: "porthole",
    documentation:
      "Circular window showing current image with old image as background.",
  },

  ";": {
    color: "#DB7093",
    number: 56,
    fn: semicircleReflect,
    args: [],
    functionName: "semicircle-reflect",
    documentation:
      "Top semicircle preserved, bottom reflected with wave distortion.",
  },

  "=": {
    color: "#5F9EA0",
    number: 57,
    fn: shiftedStripes,
    args: [{ type: INT, documentation: "Stripe height in pixels" }],
    functionName: "shifted-stripes",
    documentation: "Horizontal stripes with alternating shifts.",
  },

  "?": {
    color: "#D2691E",
    number: 58,
    fn: help,
    args: [
      {
        type: INT,
        documentation: "Page number (A=intro, B+=reference, #=history)",
      },
    ],
    functionName: "help",
    documentation: "Display help text or image history table.",
  },

  "@": {
    color: "#7B68EE",
    number: 59,
    fn: cond,
    args: [
      {
        type: INDEX,
        documentation: "Condition image sampled for threshold comparison",
      },
      {
        type: INDEX,
        documentation: "Source image when condition >= threshold",
      },
      { type: INDEX, documentation: "Source image when condition < threshold" },
      {
        type: Choice("hue", "saturation", "lightness", "red", "green", "blue"),
        documentation: "Color channel to extract from condition image",
      },
      { type: INT, documentation: "Threshold (A=0%, ~=100% of channel range)" },
    ],
    functionName: "cond",
    documentation:
      "Per-pixel conditional: extracts channel from condition image, outputs true-image pixel where value >= threshold, otherwise false-image pixel.",
  },

  "[": {
    color: "#48D1CC",
    number: 60,
    fn: rotateLeft,
    args: [],
    functionName: "rotate-left",
    documentation: "Rotate 20° counter-clockwise.",
  },

  "\\": {
    color: "#C71585",
    number: 61,
    fn: composite,
    args: [
      { type: INDEX, documentation: "Old image source" },
      { type: INT, documentation: "Source X (normalized 0-1)" },
      { type: INT, documentation: "Source Y (normalized 0-1)" },
      { type: INT, documentation: "Source width (normalized 0-1)" },
      { type: INT, documentation: "Source height (normalized 0-1)" },
      { type: INT, documentation: "Dest X (normalized 0-1)" },
      { type: INT, documentation: "Dest Y (normalized 0-1)" },
      { type: INT, documentation: "Dest width (normalized 0-1)" },
      { type: INT, documentation: "Dest height (normalized 0-1)" },
      { type: INT, documentation: "Rotation (normalized 0-1 → 0-360°)" },
      {
        type: INT,
        documentation:
          "Blend mode (mod 16: normal, xor, nand, and, or, multiply, screen, overlay, darken, lighten, diff, excl, add, sub, hard, soft)",
      },
    ],
    functionName: "composite",
    documentation: "Composite transformed region from old image onto current.",
  },

  "]": {
    color: "#00FA9A",
    number: 62,
    fn: leftHalfOffset,
    args: [],
    functionName: "left-half-offset",
    documentation: "Shift left half vertically by 20% with wraparound.",
  },

  _: {
    color: "#708090",
    number: 63,
    fn: scanlines,
    args: [],
    functionName: "scanlines",
    documentation: "CRT scanline effect with darkening and displacement.",
  },

  "`": {
    color: "#6495ED",
    number: 64,
    fn: rule110,
    args: [{ type: INT, documentation: "Number of generations (×8)" }],
    functionName: "rule110",
    documentation:
      "Rule 110 cellular automaton - a Turing-complete 1D CA applied horizontally to each row.",
  },

  "{": {
    color: "#DC143C",
    number: 65,
    fn: skewLeft,
    args: [],
    functionName: "skew-left",
    documentation: "Skew 20° left with wraparound (top left, bottom right).",
  },

  "|": {
    color: "#00BFFF",
    number: 66,
    fn: verticalSplit,
    args: [{ type: INDEX, documentation: "Old image for right half" }],
    functionName: "vertical-split",
    documentation:
      "Vertical split with wavy blend zone using multiple blend modes.",
  },

  "}": {
    color: "#9400D3",
    number: 67,
    fn: skewRight,
    args: [],
    functionName: "skew-right",
    documentation: "Skew 20° right with wraparound (top right, bottom left).",
  },

  "~": {
    color: "#FF6347",
    number: 68,
    fn: waveChromatic,
    args: [{ type: INT, documentation: "Wave amplitude and chromatic shift" }],
    functionName: "wave-chromatic",
    documentation: "Horizontal wave distortion with chromatic aberration.",
  },
};

// Re-export types and utilities for runtime.ts
export {
  CharDef,
  INT,
  COLOR,
  INDEX,
  Choice,
  ArgDef,
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
  getOldImage,
  ArgType,
  IntType,
  ColorType,
  IndexType,
  ChoiceType,
  OpInfo,
  UPLOAD_CHAR,
  UPLOAD_COUNT,
  isIndexedUpload,
  isInvalidUpload,
  isAnyUpload,
  getUploadIndex,
  getUploadChar,
  getInvalidUploadChar,
  createPlaceholderImage,
  emeraldReady,
  bgRemovalReady,
};
