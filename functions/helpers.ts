import * as THREE from "three";
import * as tf from "@tensorflow/tfjs";
import FFT from "fft.js";
import { EffectComposer } from "three/examples/jsm/postprocessing/EffectComposer.js";
import { RenderPass } from "three/examples/jsm/postprocessing/RenderPass.js";
import { UnrealBloomPass } from "three/examples/jsm/postprocessing/UnrealBloomPass.js";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import * as ort from "onnxruntime-web";

// Set TensorFlow.js to use WebGL backend
tf.setBackend("webgl");

// Configure ONNX Runtime to use CDN for WASM files
ort.env.wasm.wasmPaths =
  "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.23.2/dist/";

export interface Image {
  width: number;
  height: number;
  data: Uint8ClampedArray;
}

export interface OpInfo {
  identifier: string;
  type: "solid" | "uploaded-image" | "function";
}

export interface FnContext {
  width: number;
  height: number;
  images: Image[];
  currentIndex: number;
  opInfos: OpInfo[];
}

export class ArgType {
  constructor(public name: string) {}
}

export class IntType extends ArgType {
  constructor() {
    super("int");
  }
}

export class ColorType extends ArgType {
  constructor() {
    super("color");
  }
}

export class IndexType extends ArgType {
  constructor() {
    super("index");
  }
}

export class ChoiceType extends ArgType {
  constructor(public choices: string[]) {
    super("choice");
  }
}

export const INT = new IntType();
export const COLOR = new ColorType();
export const INDEX = new IndexType();
export function Choice(...choices: string[]): ChoiceType {
  return new ChoiceType(choices);
}

export interface ArgDef {
  type: ArgType;
  documentation: string;
}

export interface CharDef {
  color: string;
  number: number;
  fn: (ctx: FnContext, ...args: any[]) => Image | Promise<Image>;
  args: ArgDef[];
  functionName: string;
  documentation: string;
}

// Upload character constants - must match build-font.py
export const UPLOAD_CHAR = "□"; // U+25A1 - unassigned upload placeholder
export const UPLOAD_COUNT = 256;
export const UPLOAD_REGULAR_BASE = 0xe200; // U+E200 to U+E2FF: valid upload □
export const UPLOAD_INVALID_BASE = 0xe300; // U+E300 to U+E3FF: invalid upload ■

// Get the upload character for a given index (0-255)
export function getUploadChar(index: number): string {
  if (index < 0 || index >= UPLOAD_COUNT) {
    throw new Error(`Upload index ${index} out of range [0, ${UPLOAD_COUNT})`);
  }
  return String.fromCodePoint(UPLOAD_REGULAR_BASE + index);
}

// Get the invalid upload character for a given index (0-255)
export function getInvalidUploadChar(index: number): string {
  if (index < 0 || index >= UPLOAD_COUNT) {
    throw new Error(`Upload index ${index} out of range [0, ${UPLOAD_COUNT})`);
  }
  return String.fromCodePoint(UPLOAD_INVALID_BASE + index);
}

// Check if a character is a valid indexed upload (U+E200 to U+E2FF)
export function isIndexedUpload(char: string | undefined | null): boolean {
  if (!char) return false;
  const code = char.codePointAt(0);
  if (code === undefined) return false;
  return (
    code >= UPLOAD_REGULAR_BASE && code < UPLOAD_REGULAR_BASE + UPLOAD_COUNT
  );
}

// Check if a character is an invalid indexed upload (U+E300 to U+E3FF)
export function isInvalidUpload(char: string | undefined | null): boolean {
  if (!char) return false;
  const code = char.codePointAt(0);
  if (code === undefined) return false;
  return (
    code >= UPLOAD_INVALID_BASE && code < UPLOAD_INVALID_BASE + UPLOAD_COUNT
  );
}

// Check if a character is any kind of upload (valid, invalid, or unassigned □)
export function isAnyUpload(char: string | undefined | null): boolean {
  if (!char) return false;
  return char === UPLOAD_CHAR || isIndexedUpload(char) || isInvalidUpload(char);
}

// Get the upload index from an indexed upload character (valid or invalid)
export function getUploadIndex(char: string | undefined | null): number | null {
  if (!char) return null;
  const code = char.codePointAt(0);
  if (code === undefined) return null;

  if (
    code >= UPLOAD_REGULAR_BASE &&
    code < UPLOAD_REGULAR_BASE + UPLOAD_COUNT
  ) {
    return code - UPLOAD_REGULAR_BASE;
  }
  if (
    code >= UPLOAD_INVALID_BASE &&
    code < UPLOAD_INVALID_BASE + UPLOAD_COUNT
  ) {
    return code - UPLOAD_INVALID_BASE;
  }
  return null;
}

export function createPlaceholderImage(width: number, height: number): Image {
  const data = new Uint8ClampedArray(width * height * 4);
  const checkSize = 16;
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const i = (y * width + x) * 4;
      const isLight =
        (Math.floor(x / checkSize) + Math.floor(y / checkSize)) % 2 === 0;
      const gray = isLight ? 128 : 96;
      data[i] = gray;
      data[i + 1] = gray;
      data[i + 2] = gray;
      data[i + 3] = 255;
    }
  }
  return { width, height, data };
}

let glCanvas: HTMLCanvasElement | null = null;
let gl: WebGLRenderingContext | null = null;
let glTexture: WebGLTexture | null = null;
let glFramebuffer: WebGLFramebuffer | null = null;

export function initWebGL(width: number, height: number): WebGLRenderingContext {
  if (!glCanvas || glCanvas.width !== width || glCanvas.height !== height) {
    glCanvas = document.createElement("canvas");
    glCanvas.width = width;
    glCanvas.height = height;
    gl = glCanvas.getContext("webgl", {
      premultipliedAlpha: false,
      preserveDrawingBuffer: true,
    });
    if (!gl) throw new Error("WebGL not supported");

    glTexture = gl.createTexture();
    glFramebuffer = gl.createFramebuffer();
  }
  return gl!;
}

function createShaderProgram(
  gl: WebGLRenderingContext,
  vertSource: string,
  fragSource: string,
): WebGLProgram {
  const vertShader = gl.createShader(gl.VERTEX_SHADER)!;
  gl.shaderSource(vertShader, vertSource);
  gl.compileShader(vertShader);

  const fragShader = gl.createShader(gl.FRAGMENT_SHADER)!;
  gl.shaderSource(fragShader, fragSource);
  gl.compileShader(fragShader);

  const program = gl.createProgram()!;
  gl.attachShader(program, vertShader);
  gl.attachShader(program, fragShader);
  gl.linkProgram(program);

  return program;
}

export function createSolidImage(
  width: number,
  height: number,
  color: string,
): Image {
  const [r, g, b] = hexToRgb(color);
  const data = new Uint8ClampedArray(width * height * 4);
  for (let i = 0; i < width * height; i++) {
    data[i * 4] = r;
    data[i * 4 + 1] = g;
    data[i * 4 + 2] = b;
    data[i * 4 + 3] = 255;
  }
  return { width, height, data };
}

function hexToRgb(hex: string): [number, number, number] {
  const h = hex.replace("#", "");
  return [
    parseInt(h.substring(0, 2), 16),
    parseInt(h.substring(2, 4), 16),
    parseInt(h.substring(4, 6), 16),
  ];
}

function cloneImage(img: Image): Image {
  return {
    width: img.width,
    height: img.height,
    data: new Uint8ClampedArray(img.data),
  };
}

function getPrevImage(ctx: FnContext): Image {
  if (ctx.images.length === 0) {
    return createSolidImage(ctx.width, ctx.height, "#000000");
  }
  return ctx.images[ctx.images.length - 1];
}

export function getOldImage(ctx: FnContext, j: number): Image {
  if (ctx.images.length === 0)
    return createSolidImage(ctx.width, ctx.height, "#000000");
  // j is 1-based (A=1, B=2, etc.), convert to 0-based index
  // A (1) -> index 0 (initial black), B (2) -> index 1, etc.
  const adjusted = Math.abs(j) - 1;
  const idx =
    ((adjusted % ctx.images.length) + ctx.images.length) % ctx.images.length;
  return ctx.images[idx];
}

function getPixel(
  img: Image,
  x: number,
  y: number,
): [number, number, number, number] {
  const cx = Math.max(0, Math.min(img.width - 1, Math.floor(x)));
  const cy = Math.max(0, Math.min(img.height - 1, Math.floor(y)));
  const i = (cy * img.width + cx) * 4;
  return [img.data[i], img.data[i + 1], img.data[i + 2], img.data[i + 3]];
}

function setPixel(
  img: Image,
  x: number,
  y: number,
  r: number,
  g: number,
  b: number,
  a: number = 255,
): void {
  if (x < 0 || x >= img.width || y < 0 || y >= img.height) return;
  const i = (y * img.width + x) * 4;
  img.data[i] = r;
  img.data[i + 1] = g;
  img.data[i + 2] = b;
  img.data[i + 3] = a;
}

function rgbToHsl(r: number, g: number, b: number): [number, number, number] {
  r /= 255;
  g /= 255;
  b /= 255;
  const max = Math.max(r, g, b);
  const min = Math.min(r, g, b);
  const l = (max + min) / 2;
  let h = 0,
    s = 0;

  if (max !== min) {
    const d = max - min;
    s = l > 0.5 ? d / (2 - max - min) : d / (max + min);
    if (max === r) h = ((g - b) / d + (g < b ? 6 : 0)) / 6;
    else if (max === g) h = ((b - r) / d + 2) / 6;
    else h = ((r - g) / d + 4) / 6;
  }

  return [h * 360, s, l];
}

function hslToRgb(h: number, s: number, l: number): [number, number, number] {
  h = h / 360;
  let r, g, b;

  if (s === 0) {
    r = g = b = l;
  } else {
    const hue2rgb = (p: number, q: number, t: number) => {
      if (t < 0) t += 1;
      if (t > 1) t -= 1;
      if (t < 1 / 6) return p + (q - p) * 6 * t;
      if (t < 1 / 2) return q;
      if (t < 2 / 3) return p + (q - p) * (2 / 3 - t) * 6;
      return p;
    };
    const q = l < 0.5 ? l * (1 + s) : l + s - l * s;
    const p = 2 * l - q;
    r = hue2rgb(p, q, h + 1 / 3);
    g = hue2rgb(p, q, h);
    b = hue2rgb(p, q, h - 1 / 3);
  }

  return [Math.round(r * 255), Math.round(g * 255), Math.round(b * 255)];
}

let emeraldScene: THREE.Scene | null = null;
let emeraldRenderer: THREE.WebGLRenderer | null = null;
let emeraldCamera: THREE.PerspectiveCamera | null = null;
let emeraldModel: THREE.Group | null = null;
let emeraldModelLoaded = false;
let emeraldLoadPromise: Promise<void> | null = null;
let emeraldComposer: EffectComposer | null = null;

function initEmeraldScene(width: number, height: number) {
  if (
    !emeraldRenderer ||
    emeraldRenderer.domElement.width !== width ||
    emeraldRenderer.domElement.height !== height
  ) {
    if (emeraldRenderer) {
      emeraldRenderer.dispose();
    }

    emeraldRenderer = new THREE.WebGLRenderer({
      alpha: true,
      antialias: true,
      premultipliedAlpha: false,
      preserveDrawingBuffer: true,
      powerPreference: "high-performance",
    });
    emeraldRenderer.setSize(width, height);
    emeraldRenderer.setPixelRatio(1);
    emeraldRenderer.setClearColor(0x000000, 0);
    emeraldRenderer.toneMapping = THREE.NoToneMapping;
    emeraldRenderer.outputColorSpace = THREE.SRGBColorSpace;

    // Setup bloom for sparkle effect
    emeraldComposer = new EffectComposer(emeraldRenderer);
  }

  if (!emeraldScene) {
    emeraldScene = new THREE.Scene();
  }

  if (!emeraldCamera) {
    emeraldCamera = new THREE.PerspectiveCamera(45, width / height, 0.1, 100);
    emeraldCamera.position.set(0, 2, 5);
    emeraldCamera.lookAt(0, 0, 0);
  } else {
    emeraldCamera.aspect = width / height;
    emeraldCamera.updateProjectionMatrix();
  }
}

function loadEmeraldModel(): Promise<void> {
  if (emeraldLoadPromise) {
    return emeraldLoadPromise;
  }

  emeraldLoadPromise = new Promise((resolve, reject) => {
    const loader = new GLTFLoader();
    loader.load(
      "./emerald.glb",
      (gltf) => {
        emeraldModel = gltf.scene;

        const box = new THREE.Box3().setFromObject(emeraldModel);
        const center = box.getCenter(new THREE.Vector3());
        const size = box.getSize(new THREE.Vector3());
        const maxDim = Math.max(size.x, size.y, size.z);
        const scale = 2.0 / maxDim;

        emeraldModel.position.set(
          -center.x * scale,
          -center.y * scale,
          -center.z * scale,
        );
        emeraldModel.scale.setScalar(scale);

        emeraldModelLoaded = true;
        resolve();
      },
      undefined,
      (error) => {
        console.error("Error loading emerald model:", error);
        reject(error);
      },
    );
  });

  return emeraldLoadPromise;
}

// Start loading immediately when module loads
export const emeraldReady = loadEmeraldModel();

// SINet background removal model
// Model: https://github.com/anilsathyan7/Portrait-Segmentation/tree/master/SINet
// 86.9K params, ~350KB, runs at 100 FPS on mobile
let sinetSession: ort.InferenceSession | null = null;
let sinetInferenceInProgress = false;
const SINET_INPUT_SIZE = 320;
const SINET_MEAN = [102.890434, 111.25247, 126.91212];
const SINET_STD = [62.93292, 62.82138, 66.355705];

async function loadSinetModel(): Promise<void> {
  try {
    console.log("Loading SINet model...");
    sinetSession = await ort.InferenceSession.create("./sinet_224.onnx", {
      executionProviders: ["wasm"],
      graphOptimizationLevel: "all",
    });
    console.log("SINet model loaded successfully");
    console.log("Input names:", sinetSession.inputNames);
    console.log("Output names:", sinetSession.outputNames);
  } catch (error) {
    console.error("Failed to load SINet model:", error);
  }
}

export const bgRemovalReady = loadSinetModel();

function breakLigatures(text: string): string {
  // Insert zero-width non-joiner between common ligature pairs
  return text
    .replace(/ff/g, "f\u200Cf")
    .replace(/fi/g, "f\u200Ci")
    .replace(/fl/g, "f\u200Cl")
    .replace(/ffi/g, "f\u200Cf\u200Ci")
    .replace(/ffl/g, "f\u200Cf\u200Cl");
}

function wrapText(text: string, maxWidth: number): string[] {
  const lines: string[] = [];
  const paragraphs = text.split("\n");

  for (const para of paragraphs) {
    if (para.length === 0) {
      lines.push("");
      continue;
    }

    const words = para.split(" ");
    let currentLine = "";

    for (const word of words) {
      if (currentLine.length === 0) {
        currentLine = word;
      } else if (currentLine.length + 1 + word.length <= maxWidth) {
        currentLine += " " + word;
      } else {
        lines.push(currentLine);
        currentLine = word;
      }
    }

    if (currentLine.length > 0) {
      lines.push(currentLine);
    }
  }

  return lines;
}

function generateIntroPage(charsPerLine: number): string[] {
  const introText = `QQQLANG: A syntax-free programming language for image synthesis

In QQQLANG, any string of visible uppercase ascii characters is a valid program.

Each character has three properties:
* An integer ('A'=1, 'B'=2, [...], '}'=67, '~'=68)
* A color
* A function

Functions can take zero or more arguments. If a function takes arguments, the characters that follow are interpreted as arguments. Otherwise characters are interpreted as functions. The exception is the first character of the program string which sets an initial solid color.

For example, the program 'ABCD' has the following interpretation:

* 'A' sets the intial color to #78A10F
* 'B' is the 'border' function that creates a circular gradient around the edges. It takes one argument, the border color.
* 'C' becomes the argument to 'B', the color of 'C' is #FF6B35
* 'D' is the 'drip' function, which creates a water drop effect. It takes no arguments.

If the program string ends before the last function has had arguments defined, it will use its own number and color as default arguments. For example, the programs 'AL', 'ALL', and 'ALLL' are equivalent.

The question mark character '?' is also a function that displays help text. '?1' and '??' show the first page of help, and '?A', '?B', etc. show subsequent pages of help text.

Some functions take an image index as an argument, and uses that old image in some way. '?#' shows the history of images the the characters to use to retrieve each image.
`;

  return wrapText(introText, charsPerLine);
}

function numToChar(num: number): string {
  if (num >= 1 && num <= 26)
    return String.fromCharCode("A".charCodeAt(0) + num - 1);
  if (num >= 27 && num <= 36)
    return String.fromCharCode("0".charCodeAt(0) + num - 27);
  const symbols = "<>^!\"#$%&'()*+,-./:;=?@[\\]_`{|}~";
  const idx = num - 37;
  if (idx >= 0 && idx < symbols.length) return symbols[idx];
  return "?";
}

export function formatFunctionHelp(
  char: string,
  def: CharDef,
  charsPerLine: number = 80,
): string[] {
  const lines: string[] = [];

  // First line: C fn-name — documentation
  const firstLine = `${char} ${def.functionName} — ${def.documentation}`;
  lines.push(...wrapText(firstLine, charsPerLine));

  // Arg lines: (n) argDoc (A=x, B=y, ...)
  for (let i = 0; i < def.args.length; i++) {
    const arg = def.args[i];
    let argLine = `(${i + 1}) ${arg.documentation}`;

    if (arg.type instanceof ChoiceType) {
      const choices = arg.type.choices;
      const mappings = choices
        .map((choice, idx) => `${numToChar(idx + 1)}=${choice}`)
        .join(", ");
      argLine += ` (${mappings})`;
    }

    lines.push(...wrapText(argLine, charsPerLine));
  }

  return lines;
}

function generateCharacterRefLines(
  char: string,
  def: CharDef,
  charsPerLine: number,
): string[] {
  return formatFunctionHelp(char, def, charsPerLine);
}

function getPageChar(pageNum: number): string {
  if (pageNum <= 0) return "?";
  if (pageNum === 1) return "?";
  if (pageNum <= 26)
    return String.fromCharCode("A".charCodeAt(0) + pageNum - 1);
  return "?";
}

interface HelpPagesResult {
  pages: string[][];
  introPageCount: number;
  refPageCount: number;
}

function generateAllHelpPages(
  charsPerLine: number,
  linesPerPage: number,
  defs: Record<string, CharDef>,
): HelpPagesResult {
  const pages: string[][] = [];

  const introLines = generateIntroPage(charsPerLine);

  let introPage: string[] = [];
  for (let i = 0; i < introLines.length; i++) {
    if (introPage.length >= linesPerPage - 2) {
      pages.push(introPage);
      introPage = [];
    }
    introPage.push(introLines[i]);
  }
  if (introPage.length > 0) {
    pages.push(introPage);
  }

  const introPageCount = pages.length;

  const chars = Object.keys(defs).sort(
    (a, b) => defs[a].number - defs[b].number,
  );

  let currentPage: string[] = [];
  currentPage.push("=== CHARACTER REFERENCE ===");
  currentPage.push("");
  let linesUsed = 2;

  for (const char of chars) {
    const def = defs[char];
    const charLines = generateCharacterRefLines(char, def, charsPerLine);

    if (linesUsed + charLines.length + 1 > linesPerPage - 2) {
      pages.push(currentPage);
      currentPage = [];
      currentPage.push("=== CHARACTER REFERENCE (continued) ===");
      currentPage.push("");
      linesUsed = 2;
    }

    currentPage.push(...charLines);
    currentPage.push("");
    linesUsed += charLines.length + 1;
  }

  if (currentPage.length > 2) {
    pages.push(currentPage);
  }

  const refPageCount = pages.length - introPageCount;
  const totalPages = pages.length;

  for (let i = 0; i < pages.length; i++) {
    const pageNum = i + 1;
    const nextPageChar = getPageChar(pageNum + 1);
    pages[i].push("");
    if (pageNum < totalPages) {
      pages[i].push(
        `[Page ${pageNum}/${totalPages}, type '?${nextPageChar}' for next page]`,
      );
    } else {
      pages[i].push(`[Page ${pageNum}/${totalPages}]`);
    }
  }

  return { pages, introPageCount, refPageCount };
}

function generateIndexPage(
  introPageCount: number,
  refPageCount: number,
): string[] {
  const lines: string[] = [];
  lines.push("=== QQQLANG HELP INDEX ===");
  lines.push("");
  lines.push("Available pages:");
  lines.push("");

  // Introduction pages
  if (introPageCount === 1) {
    lines.push("?? or ?A - Introduction");
  } else {
    const lastIntroChar = getPageChar(introPageCount);
    lines.push(`??/?A-?${lastIntroChar} - Introduction`);
  }

  // Character reference pages
  if (refPageCount > 0) {
    const firstRefPage = introPageCount + 1;
    const lastRefPage = introPageCount + refPageCount;
    const firstRefChar = getPageChar(firstRefPage);
    const lastRefChar = getPageChar(lastRefPage);

    if (refPageCount === 1) {
      lines.push(`?${firstRefChar} - Character reference`);
    } else {
      lines.push(`?${firstRefChar}-?${lastRefChar} - Character reference`);
    }
  }

  // Image history page
  lines.push("?# - Image history");

  lines.push("");
  lines.push("Enter a valid page code to view help.");
  lines.push("Invalid page codes show this index.");

  return lines;
}

// Export items not already exported
export {
  initWebGL,
  createShaderProgram,
  hexToRgb,
  cloneImage,
  getPrevImage,
  getPixel,
  setPixel,
  rgbToHsl,
  hslToRgb,
  breakLigatures,
  wrapText,
  generateIntroPage,
  numToChar,
  generateCharacterRefLines,
  getPageChar,
  generateAllHelpPages,
  generateIndexPage,
};
