import * as THREE from 'three';
import FFT from 'fft.js';
import { EffectComposer } from 'three/examples/jsm/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/examples/jsm/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/examples/jsm/postprocessing/UnrealBloomPass.js';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';

export interface Image {
  width: number;
  height: number;
  data: Uint8ClampedArray;
}

export interface OpInfo {
  identifier: string;
  type: 'solid' | 'uploaded-image' | 'function';
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
  constructor() { super('int'); }
}

export class ColorType extends ArgType {
  constructor() { super('color'); }
}

export class IndexType extends ArgType {
  constructor() { super('index'); }
}

export class ChoiceType extends ArgType {
  constructor(public choices: string[]) { super('choice'); }
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
  fn: (ctx: FnContext, ...args: any[]) => Image;
  args: ArgDef[];
  functionName: string;
  documentation: string;
}

export const UPLOAD_CHAR = '🖼';

export function createPlaceholderImage(width: number, height: number): Image {
  const data = new Uint8ClampedArray(width * height * 4);
  const checkSize = 16;
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const i = (y * width + x) * 4;
      const isLight = (Math.floor(x / checkSize) + Math.floor(y / checkSize)) % 2 === 0;
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

function initWebGL(width: number, height: number): WebGLRenderingContext {
  if (!glCanvas || glCanvas.width !== width || glCanvas.height !== height) {
    glCanvas = document.createElement('canvas');
    glCanvas.width = width;
    glCanvas.height = height;
    gl = glCanvas.getContext('webgl', { 
      premultipliedAlpha: false,
      preserveDrawingBuffer: true 
    });
    if (!gl) throw new Error('WebGL not supported');
    
    glTexture = gl.createTexture();
    glFramebuffer = gl.createFramebuffer();
  }
  return gl!;
}

function createShaderProgram(gl: WebGLRenderingContext, vertSource: string, fragSource: string): WebGLProgram {
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

export function createSolidImage(width: number, height: number, color: string): Image {
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
  const h = hex.replace('#', '');
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
    return createSolidImage(ctx.width, ctx.height, '#000000');
  }
  return ctx.images[ctx.images.length - 1];
}

export function getOldImage(ctx: FnContext, j: number): Image {
  if (ctx.images.length === 0) return createSolidImage(ctx.width, ctx.height, '#000000');
  if (ctx.images.length === 1) return ctx.images[0];
  const idx = Math.abs(j) % ctx.images.length;
  return ctx.images[idx];
}

function getPixel(img: Image, x: number, y: number): [number, number, number, number] {
  const cx = Math.max(0, Math.min(img.width - 1, Math.floor(x)));
  const cy = Math.max(0, Math.min(img.height - 1, Math.floor(y)));
  const i = (cy * img.width + cx) * 4;
  return [img.data[i], img.data[i + 1], img.data[i + 2], img.data[i + 3]];
}

function setPixel(img: Image, x: number, y: number, r: number, g: number, b: number, a: number = 255): void {
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
  let h = 0, s = 0;
  
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
      if (t < 1/6) return p + (q - p) * 6 * t;
      if (t < 1/2) return q;
      if (t < 2/3) return p + (q - p) * (2/3 - t) * 6;
      return p;
    };
    const q = l < 0.5 ? l * (1 + s) : l + s - l * s;
    const p = 2 * l - q;
    r = hue2rgb(p, q, h + 1/3);
    g = hue2rgb(p, q, h);
    b = hue2rgb(p, q, h - 1/3);
  }
  
  return [Math.round(r * 255), Math.round(g * 255), Math.round(b * 255)];
}

function fnA(ctx: FnContext): Image {
  const prev = getPrevImage(ctx);
  const gl = initWebGL(ctx.width, ctx.height);
  
  gl.activeTexture(gl.TEXTURE0);
  gl.bindTexture(gl.TEXTURE_2D, null);
  gl.activeTexture(gl.TEXTURE1);
  gl.bindTexture(gl.TEXTURE_2D, null);
  gl.activeTexture(gl.TEXTURE0);
  
  const vertexShader = `
    attribute vec2 position;
    varying vec2 vUV;
    void main() {
      vUV = position * 0.5 + 0.5;
      gl_Position = vec4(position, 0.0, 1.0);
    }
  `;
  
  const fragmentShader = `
    precision mediump float;
    uniform sampler2D texture;
    uniform vec2 resolution;
    varying vec2 vUV;
    
    vec3 renderSphere(vec2 uv, vec2 center, float radius, sampler2D tex) {
      vec2 p = (uv - center) / radius;
      float d = length(p);
      
      if (d > 1.0) return vec3(-1.0);
      
      float z = sqrt(1.0 - d * d);
      vec3 normal = normalize(vec3(p.x, -p.y, z));
      vec3 lightDir = normalize(vec3(-0.3, 0.3, 1.0));
      
      float diffuse = max(dot(normal, lightDir), 0.0);
      float ambient = 0.3;
      float lighting = ambient + diffuse * 0.7;
      
      vec2 texCoord = vec2(
        atan(normal.x, normal.z) / (2.0 * 3.14159) + 0.5,
        acos(normal.y) / 3.14159
      );
      
      vec3 color = texture2D(tex, texCoord).rgb;
      return color * lighting;
    }
    
    void main() {
      vec2 uv = gl_FragCoord.xy / resolution;
      vec3 bg = texture2D(texture, vec2(1.0 - uv.x, 1.0 - uv.y)).rgb;
      
      vec2 topRight = vec2(0.75, 0.75);
      vec2 bottomLeft = vec2(0.25, 0.25);
      float radius = 0.15;
      
      vec3 sphere1 = renderSphere(uv, topRight, radius, texture);
      vec3 sphere2 = renderSphere(uv, bottomLeft, radius, texture);
      
      vec3 color = bg;
      if (sphere1.x >= 0.0) color = sphere1;
      if (sphere2.x >= 0.0) color = sphere2;
      
      gl_FragColor = vec4(color, 1.0);
    }
  `;
  
  const program = createShaderProgram(gl, vertexShader, fragmentShader);
  gl.useProgram(program);
  
  const texture = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, texture);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, prev.width, prev.height, 0, gl.RGBA, gl.UNSIGNED_BYTE, prev.data);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  
  const vertices = new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]);
  const buffer = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
  gl.bufferData(gl.ARRAY_BUFFER, vertices, gl.STATIC_DRAW);
  
  const positionLoc = gl.getAttribLocation(program, 'position');
  gl.enableVertexAttribArray(positionLoc);
  gl.vertexAttribPointer(positionLoc, 2, gl.FLOAT, false, 0, 0);
  
  const textureLoc = gl.getUniformLocation(program, 'texture');
  const resolutionLoc = gl.getUniformLocation(program, 'resolution');
  
  gl.uniform1i(textureLoc, 0);
  gl.uniform2f(resolutionLoc, ctx.width, ctx.height);
  
  gl.viewport(0, 0, ctx.width, ctx.height);
  gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  
  const pixels = new Uint8ClampedArray(ctx.width * ctx.height * 4);
  gl.readPixels(0, 0, ctx.width, ctx.height, gl.RGBA, gl.UNSIGNED_BYTE, pixels);
  
  const flipped = new Uint8ClampedArray(ctx.width * ctx.height * 4);
  for (let y = 0; y < ctx.height; y++) {
    for (let x = 0; x < ctx.width; x++) {
      const srcIdx = ((ctx.height - 1 - y) * ctx.width + x) * 4;
      const dstIdx = (y * ctx.width + x) * 4;
      flipped[dstIdx] = pixels[srcIdx];
      flipped[dstIdx + 1] = pixels[srcIdx + 1];
      flipped[dstIdx + 2] = pixels[srcIdx + 2];
      flipped[dstIdx + 3] = pixels[srcIdx + 3];
    }
  }
  
  gl.deleteTexture(texture);
  gl.deleteBuffer(buffer);
  gl.deleteProgram(program);
  
  return { width: ctx.width, height: ctx.height, data: flipped };
}

function fnB(ctx: FnContext, old: Image): Image {
  const prev = getPrevImage(ctx);
  const out = createSolidImage(ctx.width, ctx.height, '#000000');
  
  const seeds: [number, number][] = [];
  for (let i = 0; i < 36; i++) {
    seeds.push([
      (i * 47) % ctx.width,
      (i * 89) % ctx.height
    ]);
  }
  
  for (let y = 0; y < ctx.height; y++) {
    for (let x = 0; x < ctx.width; x++) {
      let minDist = Infinity;
      let closestIdx = 0;
      
      for (let i = 0; i < seeds.length; i++) {
        const dx = x - seeds[i][0];
        const dy = y - seeds[i][1];
        const dist = dx * dx + dy * dy;
        if (dist < minDist) {
          minDist = dist;
          closestIdx = i;
        }
      }
      
      const src = closestIdx % 2 === 0 ? prev : old;
      const [r, g, b] = getPixel(src, x, y);
      setPixel(out, x, y, r, g, b);
    }
  }
  
  return out;
}

function fnC(ctx: FnContext, n: number): Image {
  const prev = getPrevImage(ctx);
  const out = createSolidImage(ctx.width, ctx.height, '#000000');
  
  const rings = Math.max(2, Math.min(n, 50));
  const cx = ctx.width / 2;
  const cy = ctx.height / 2;
  const maxRadius = Math.sqrt(cx * cx + cy * cy);
  
  for (let y = 0; y < ctx.height; y++) {
    for (let x = 0; x < ctx.width; x++) {
      const dx = x - cx;
      const dy = y - cy;
      const dist = Math.sqrt(dx * dx + dy * dy);
      
      const ringIdx = Math.floor((dist / maxRadius) * rings);
      const [r, g, b] = getPixel(prev, x, y);
      
      if (ringIdx % 2 === 0) {
        setPixel(out, x, y, r, g, b);
      } else {
        const hueShift = (360 / rings) * ringIdx;
        const [h, s, l] = rgbToHsl(r, g, b);
        const [nr, ng, nb] = hslToRgb((h + hueShift) % 360, s, l);
        setPixel(out, x, y, nr, ng, nb);
      }
    }
  }
  
  return out;
}

function fn5(ctx: FnContext, n: number): Image {
  const prev = getPrevImage(ctx);
  const out = cloneImage(prev);
  
  const cellSize = Math.max(20, n * 4);
  const cols = Math.max(2, Math.floor(ctx.width / cellSize));
  const rows = Math.max(2, Math.floor(ctx.height / cellSize));
  const totalTriangles = cols * rows * 2;
  
  for (let row = 0; row < rows; row++) {
    for (let col = 0; col < cols; col++) {
      const x0 = Math.floor((col / cols) * ctx.width);
      const y0 = Math.floor((row / rows) * ctx.height);
      const x1 = Math.floor(((col + 1) / cols) * ctx.width);
      const y1 = Math.floor(((row + 1) / rows) * ctx.height);
      const cellW = x1 - x0;
      const cellH = y1 - y0;
      
      const triIndex1 = (row * cols + col) * 2;
      const triIndex2 = triIndex1 + 1;
      
      const hueShift1 = (triIndex1 * 137.5) % 360;
      const hueShift2 = (triIndex2 * 137.5) % 360;
      const lightMod1 = (triIndex1 % 2 === 0) ? 0.15 : -0.15;
      const lightMod2 = (triIndex2 % 2 === 0) ? 0.15 : -0.05;
      
      for (let y = y0; y < y1; y++) {
        for (let x = x0; x < x1; x++) {
          const localX = (x - x0) / cellW;
          const localY = (y - y0) / cellH;
          
          const [r, g, b] = getPixel(prev, x, y);
          const [h, s, l] = rgbToHsl(r, g, b);
          const isUpperTriangle = localX + localY < 1;
          
          const hueShift = isUpperTriangle ? hueShift1 : hueShift2;
          const lightMod = isUpperTriangle ? lightMod1 : lightMod2;
          const newL = Math.max(0, Math.min(1, l + lightMod));
          
          const [nr, ng, nb] = hslToRgb((h + hueShift) % 360, Math.min(1, s * 1.3), newL);
          setPixel(out, x, y, nr, ng, nb);
        }
      }
    }
  }
  
  return out;
}

function fnF(ctx: FnContext, n: number): Image {
  const prev = getPrevImage(ctx);
  const out = createSolidImage(ctx.width, ctx.height, '#000000');
  
  const zoomPower = Math.max(1, n);
  const scale = Math.pow(0.7, zoomPower);
  const maxIterations = 10 + zoomPower * 5;
  
  const cReal = -0.7;
  const cImag = 0.27015;
  
  const centerReal = 0.15;
  const centerImag = 0.6;
  
  const aspect = ctx.height / ctx.width;
  
  for (let py = 0; py < ctx.height; py++) {
    for (let px = 0; px < ctx.width; px++) {
      let zReal = centerReal + ((px / ctx.width) - 0.5) * 3 * scale;
      let zImag = centerImag + ((py / ctx.height) - 0.5) * 3 * scale * aspect;
      
      let iteration = 0;
      let zReal2 = zReal * zReal;
      let zImag2 = zImag * zImag;
      
      while (iteration < maxIterations && zReal2 + zImag2 < 256) {
        zImag = 2 * zReal * zImag + cImag;
        zReal = zReal2 - zImag2 + cReal;
        zReal2 = zReal * zReal;
        zImag2 = zImag * zImag;
        iteration++;
      }
      
      if (iteration === maxIterations) {
        const [pr, pg, pb] = getPixel(prev, px, py);
        setPixel(out, px, py, pr, pg, pb);
      } else {
        const log_zn = Math.log(zReal2 + zImag2) / 2;
        const nu = Math.log(log_zn / Math.log(2)) / Math.log(2);
        const smoothed = iteration + 1 - nu;
        
        const t = (smoothed % 20) / 20;
        const sampleX = Math.floor(t * ctx.width);
        const sampleY = py;
        
        const [pr, pg, pb] = getPixel(prev, sampleX, sampleY);
        const brightness = 0.85 + 0.3 * t;
        setPixel(out, px, py, 
          Math.min(255, Math.round(pr * brightness)),
          Math.min(255, Math.round(pg * brightness)),
          Math.min(255, Math.round(pb * brightness)));
      }
    }
  }
  
  return out;
}

function fnG(ctx: FnContext, n: number): Image {
  const prev = getPrevImage(ctx);
  const out = cloneImage(prev);
  
  const histogram = new Array(256).fill(0);
  for (let i = 0; i < out.data.length; i += 4) {
    const gray = Math.round(out.data[i] * 0.299 + out.data[i + 1] * 0.587 + out.data[i + 2] * 0.114);
    out.data[i] = gray;
    out.data[i + 1] = gray;
    out.data[i + 2] = gray;
    histogram[gray]++;
  }
  
  const numColors = Math.max(2, Math.min(n, 16));
  const colors: [number, number, number][] = [];
  
  for (let i = 0; i < numColors; i++) {
    const hue = (i / numColors) * 360;
    colors.push(hslToRgb(hue, 0.7, 0.5));
  }
  
  for (let i = 0; i < out.data.length; i += 4) {
    const gray = out.data[i];
    const colorIndex = Math.floor((gray / 256) * numColors);
    const clampedIndex = Math.min(colorIndex, numColors - 1);
    
    out.data[i] = colors[clampedIndex][0];
    out.data[i + 1] = colors[clampedIndex][1];
    out.data[i + 2] = colors[clampedIndex][2];
  }
  
  return out;
}

function fnH(ctx: FnContext): Image {
  const prev = getPrevImage(ctx);
  const out = createSolidImage(ctx.width, ctx.height, '#000000');
  
  const cx = ctx.width / 2;
  const cy = ctx.height / 2;
  
  for (let y = 0; y < ctx.height; y++) {
    for (let x = 0; x < ctx.width; x++) {
      const nx = (x - cx) / cx;
      const ny = (y - cy) / cy;
      
      const hourglassWidth = Math.abs(ny);
      const inHourglass = Math.abs(nx) < hourglassWidth;
      
      const [pr, pg, pb] = getPixel(prev, x, y);
      
      const distFromCenter = Math.sqrt(nx * nx + ny * ny);
      const angle = Math.atan2(ny, nx);
      
      if (inHourglass) {
        const gr = Math.floor((1 - Math.abs(ny)) * 255);
        const gg = Math.floor((Math.abs(nx) / Math.max(0.01, hourglassWidth)) * 255);
        const gb = Math.floor(distFromCenter * 180);
        
        const nr = 255 - (pr & gr);
        const ng = 255 - (pg & gg);
        const nb = 255 - (pb & gb);
        
        setPixel(out, x, y, nr, ng, nb);
      } else {
        const gr = Math.floor(((angle + Math.PI) / (Math.PI * 2)) * 255);
        const gg = Math.floor((1 - distFromCenter) * 200 + 55);
        const gb = Math.floor(((nx + 1) / 2) * 255);
        
        const nr = (pr + gr) % 256;
        const ng = Math.abs(pg - gg);
        const nb = pb ^ gb;
        
        setPixel(out, x, y, nr, ng, nb);
      }
    }
  }
  
  return out;
}

function fnI(ctx: FnContext): Image {
  const prev = getPrevImage(ctx);
  const out = cloneImage(prev);
  
  for (let i = 0; i < out.data.length; i += 4) {
    out.data[i] = 255 - out.data[i];
    out.data[i + 1] = 255 - out.data[i + 1];
    out.data[i + 2] = 255 - out.data[i + 2];
  }
  
  const sobelX = [-1, 0, 1, -2, 0, 2, -1, 0, 1];
  const sobelY = [-1, -2, -1, 0, 0, 0, 1, 2, 1];
  
  for (let y = 1; y < ctx.height - 1; y++) {
    for (let x = 1; x < ctx.width - 1; x++) {
      let gx = 0, gy = 0;
      
      for (let ky = -1; ky <= 1; ky++) {
        for (let kx = -1; kx <= 1; kx++) {
          const px = getPixel(prev, x + kx, y + ky);
          const gray = px[0] * 0.299 + px[1] * 0.587 + px[2] * 0.114;
          const kernelIdx = (ky + 1) * 3 + (kx + 1);
          gx += gray * sobelX[kernelIdx];
          gy += gray * sobelY[kernelIdx];
        }
      }
      
      const magnitude = Math.sqrt(gx * gx + gy * gy);
      
      if (magnitude > 50) {
        setPixel(out, x, y, 255, 255, 255);
      }
    }
  }
  
  return out;
}

function fnJ(ctx: FnContext, old: Image): Image {
  const prev = getPrevImage(ctx);
  const out = createSolidImage(ctx.width, ctx.height, '#000000');
  
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

function fnK(ctx: FnContext, n: number): Image {
  const prev = getPrevImage(ctx);
  const out = createSolidImage(ctx.width, ctx.height, '#000000');
  
  const segments = Math.max(1, Math.min(n + 2, 18));
  const cx = ctx.width / 2;
  const cy = ctx.height / 2;
  const angleStep = (Math.PI * 2) / segments;
  const zoom = 1.1;
  
  for (let y = 0; y < ctx.height; y++) {
    for (let x = 0; x < ctx.width; x++) {
      const dx = (x - cx) * zoom;
      const dy = (y - cy) * zoom;
      let angle = Math.atan2(dy, dx);
      if (angle < 0) angle += Math.PI * 2;
      
      const segmentAngle = angle % angleStep;
      const mirroredAngle = segmentAngle > angleStep / 2 ? angleStep - segmentAngle : segmentAngle;
      
      const r = Math.sqrt(dx * dx + dy * dy);
      const sx = cx + r * Math.cos(mirroredAngle);
      const sy = cy + r * Math.sin(mirroredAngle);
      
      const [pr, pg, pb] = getPixel(prev, Math.floor(sx), Math.floor(sy));
      setPixel(out, x, y, pr, pg, pb);
    }
  }
  
  return out;
}

function fnL(ctx: FnContext, old: Image, rot: number): Image {
  const prev = getPrevImage(ctx);
  const gl = initWebGL(ctx.width, ctx.height);
  
  gl.activeTexture(gl.TEXTURE0);
  gl.bindTexture(gl.TEXTURE_2D, null);
  gl.activeTexture(gl.TEXTURE1);
  gl.bindTexture(gl.TEXTURE_2D, null);
  gl.activeTexture(gl.TEXTURE0);
  
  const rotation = rot * 0.3;
  
  const vertexShader = `
    attribute vec3 aPosition;
    attribute vec3 aNormal;
    attribute vec2 aTexCoord;
    
    uniform mat4 uProjection;
    uniform mat4 uModelView;
    uniform mat3 uNormalMatrix;
    
    varying vec3 vNormal;
    varying vec3 vPosition;
    varying vec2 vTexCoord;
    
    void main() {
      vPosition = (uModelView * vec4(aPosition, 1.0)).xyz;
      vNormal = uNormalMatrix * aNormal;
      vTexCoord = aTexCoord;
      gl_Position = uProjection * vec4(vPosition, 1.0);
    }
  `;
  
  const fragmentShader = `
    precision highp float;
    
    uniform sampler2D uTexture;
    uniform vec3 uLightPos;
    uniform vec3 uLightPos2;
    
    varying vec3 vNormal;
    varying vec3 vPosition;
    varying vec2 vTexCoord;
    
    void main() {
      vec3 normal = normalize(vNormal);
      vec3 viewDir = normalize(-vPosition);
      
      // Light 1 - main light
      vec3 lightDir1 = normalize(uLightPos - vPosition);
      vec3 halfDir1 = normalize(lightDir1 + viewDir);
      float diff1 = max(dot(normal, lightDir1), 0.0);
      float spec1 = pow(max(dot(normal, halfDir1), 0.0), 32.0);
      
      // Light 2 - fill light
      vec3 lightDir2 = normalize(uLightPos2 - vPosition);
      vec3 halfDir2 = normalize(lightDir2 + viewDir);
      float diff2 = max(dot(normal, lightDir2), 0.0);
      float spec2 = pow(max(dot(normal, halfDir2), 0.0), 32.0);
      
      float ambient = 0.35;
      float diffuse = diff1 * 0.6 + diff2 * 0.35;
      float specular = (spec1 * 0.8 + spec2 * 0.4);
      
      vec2 tiledCoord = fract(vTexCoord);
      vec3 texColor = texture2D(uTexture, tiledCoord).rgb;
      vec3 color = texColor * (ambient + diffuse) + vec3(1.0) * specular;
      
      gl_FragColor = vec4(color, 1.0);
    }
  `;
  
  const bgVertShader = `
    attribute vec2 aPosition;
    varying vec2 vUV;
    void main() {
      vUV = aPosition * 0.5 + 0.5;
      gl_Position = vec4(aPosition, 0.999, 1.0);
    }
  `;
  
  const bgFragShader = `
    precision highp float;
    uniform sampler2D uBgTexture;
    varying vec2 vUV;
    void main() {
      gl_FragColor = texture2D(uBgTexture, vec2(vUV.x, 1.0 - vUV.y));
    }
  `;
  
  const tubeProgram = createShaderProgram(gl, vertexShader, fragmentShader);
  const bgProgram = createShaderProgram(gl, bgVertShader, bgFragShader);
  
  const prevTexture = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, prevTexture);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, prev.width, prev.height, 0, gl.RGBA, gl.UNSIGNED_BYTE, prev.data);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  
  const oldTexture = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, oldTexture);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, old.width, old.height, 0, gl.RGBA, gl.UNSIGNED_BYTE, old.data);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  
  const positions: number[] = [];
  const normals: number[] = [];
  const texCoords: number[] = [];
  
  const a = 3, b = 2, c = 1;
  const tubeRadius = 0.08;
  const segments = 400;
  const radialSegments = 12;
  
  const cross = (a: number[], b: number[]): number[] => [
    a[1]*b[2] - a[2]*b[1],
    a[2]*b[0] - a[0]*b[2],
    a[0]*b[1] - a[1]*b[0]
  ];
  const normalize = (v: number[]): number[] => {
    const len = Math.sqrt(v[0]**2 + v[1]**2 + v[2]**2);
    if (len < 0.0001) return [1, 0, 0];
    return [v[0]/len, v[1]/len, v[2]/len];
  };
  const dot = (a: number[], b: number[]): number => a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
  
  const getPoint = (t: number): number[] => [
    Math.sin(a * t) * 0.7,
    Math.sin(b * t) * 0.7,
    Math.sin(c * t) * 0.5
  ];
  
  const getTangent = (t: number): number[] => {
    const eps = 0.0001;
    const p1 = getPoint(t - eps);
    const p2 = getPoint(t + eps);
    return normalize([p2[0]-p1[0], p2[1]-p1[1], p2[2]-p1[2]]);
  };
  
  const frames: {point: number[], normal: number[], binormal: number[]}[] = [];
  
  let prevNormal = [1, 0, 0];
  const firstTan = getTangent(0);
  if (Math.abs(dot(prevNormal, firstTan)) > 0.9) {
    prevNormal = [0, 1, 0];
  }
  prevNormal = normalize(cross(cross(firstTan, prevNormal), firstTan));
  
  for (let i = 0; i <= segments; i++) {
    const t = (i / segments) * Math.PI * 2;
    const point = getPoint(t);
    const tangent = getTangent(t);
    
    let normal = normalize(cross(cross(tangent, prevNormal), tangent));
    const binormal = normalize(cross(tangent, normal));
    
    frames.push({ point, normal, binormal });
    prevNormal = normal;
  }
  
  const firstFrame = frames[0];
  const lastFrame = frames[segments];
  const twistAngle = Math.atan2(
    dot(lastFrame.normal, firstFrame.binormal),
    dot(lastFrame.normal, firstFrame.normal)
  );
  
  for (let i = 0; i <= segments; i++) {
    const correction = -twistAngle * (i / segments);
    const cos_c = Math.cos(correction), sin_c = Math.sin(correction);
    const f = frames[i];
    const newNormal = [
      f.normal[0] * cos_c + f.binormal[0] * sin_c,
      f.normal[1] * cos_c + f.binormal[1] * sin_c,
      f.normal[2] * cos_c + f.binormal[2] * sin_c
    ];
    const newBinormal = [
      -f.normal[0] * sin_c + f.binormal[0] * cos_c,
      -f.normal[1] * sin_c + f.binormal[1] * cos_c,
      -f.normal[2] * sin_c + f.binormal[2] * cos_c
    ];
    f.normal = newNormal;
    f.binormal = newBinormal;
  }
  
  for (let i = 0; i < segments; i++) {
    const f0 = frames[i];
    const f1 = frames[i + 1];
    
    for (let r = 0; r < radialSegments; r++) {
      const angle0 = (r / radialSegments) * Math.PI * 2;
      const angle1 = ((r + 1) / radialSegments) * Math.PI * 2;
      
      const cos0 = Math.cos(angle0), sin0 = Math.sin(angle0);
      const cos1 = Math.cos(angle1), sin1 = Math.sin(angle1);
      
      const v00 = [
        f0.point[0] + (f0.normal[0] * cos0 + f0.binormal[0] * sin0) * tubeRadius,
        f0.point[1] + (f0.normal[1] * cos0 + f0.binormal[1] * sin0) * tubeRadius,
        f0.point[2] + (f0.normal[2] * cos0 + f0.binormal[2] * sin0) * tubeRadius
      ];
      const v01 = [
        f0.point[0] + (f0.normal[0] * cos1 + f0.binormal[0] * sin1) * tubeRadius,
        f0.point[1] + (f0.normal[1] * cos1 + f0.binormal[1] * sin1) * tubeRadius,
        f0.point[2] + (f0.normal[2] * cos1 + f0.binormal[2] * sin1) * tubeRadius
      ];
      const v10 = [
        f1.point[0] + (f1.normal[0] * cos0 + f1.binormal[0] * sin0) * tubeRadius,
        f1.point[1] + (f1.normal[1] * cos0 + f1.binormal[1] * sin0) * tubeRadius,
        f1.point[2] + (f1.normal[2] * cos0 + f1.binormal[2] * sin0) * tubeRadius
      ];
      const v11 = [
        f1.point[0] + (f1.normal[0] * cos1 + f1.binormal[0] * sin1) * tubeRadius,
        f1.point[1] + (f1.normal[1] * cos1 + f1.binormal[1] * sin1) * tubeRadius,
        f1.point[2] + (f1.normal[2] * cos1 + f1.binormal[2] * sin1) * tubeRadius
      ];
      
      const n00 = [f0.normal[0] * cos0 + f0.binormal[0] * sin0, f0.normal[1] * cos0 + f0.binormal[1] * sin0, f0.normal[2] * cos0 + f0.binormal[2] * sin0];
      const n01 = [f0.normal[0] * cos1 + f0.binormal[0] * sin1, f0.normal[1] * cos1 + f0.binormal[1] * sin1, f0.normal[2] * cos1 + f0.binormal[2] * sin1];
      const n10 = [f1.normal[0] * cos0 + f1.binormal[0] * sin0, f1.normal[1] * cos0 + f1.binormal[1] * sin0, f1.normal[2] * cos0 + f1.binormal[2] * sin0];
      const n11 = [f1.normal[0] * cos1 + f1.binormal[0] * sin1, f1.normal[1] * cos1 + f1.binormal[1] * sin1, f1.normal[2] * cos1 + f1.binormal[2] * sin1];
      
      const u0 = i / segments * 3;
      const u1 = (i + 1) / segments * 3;
      const v0 = r / radialSegments;
      const v1 = (r + 1) / radialSegments;
      
      positions.push(...v00, ...v10, ...v11, ...v00, ...v11, ...v01);
      normals.push(...n00, ...n10, ...n11, ...n00, ...n11, ...n01);
      texCoords.push(u0, v0, u1, v0, u1, v1, u0, v0, u1, v1, u0, v1);
    }
  }
  
  gl.enable(gl.DEPTH_TEST);
  gl.viewport(0, 0, ctx.width, ctx.height);
  gl.clearColor(0, 0, 0, 1);
  gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
  
  gl.useProgram(bgProgram);
  gl.activeTexture(gl.TEXTURE0);
  gl.bindTexture(gl.TEXTURE_2D, prevTexture);
  gl.uniform1i(gl.getUniformLocation(bgProgram, 'uBgTexture'), 0);
  
  const bgVerts = new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]);
  const bgBuffer = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, bgBuffer);
  gl.bufferData(gl.ARRAY_BUFFER, bgVerts, gl.STATIC_DRAW);
  const bgPosLoc = gl.getAttribLocation(bgProgram, 'aPosition');
  gl.enableVertexAttribArray(bgPosLoc);
  gl.vertexAttribPointer(bgPosLoc, 2, gl.FLOAT, false, 0, 0);
  gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  gl.disableVertexAttribArray(bgPosLoc);
  
  gl.useProgram(tubeProgram);
  
  const posLoc = gl.getAttribLocation(tubeProgram, 'aPosition');
  const normLoc = gl.getAttribLocation(tubeProgram, 'aNormal');
  const texLoc = gl.getAttribLocation(tubeProgram, 'aTexCoord');
  
  const posBuffer = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, posBuffer);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(positions), gl.STATIC_DRAW);
  
  const normBuffer = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, normBuffer);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(normals), gl.STATIC_DRAW);
  
  const texBuffer = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, texBuffer);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(texCoords), gl.STATIC_DRAW);
  
  gl.bindBuffer(gl.ARRAY_BUFFER, posBuffer);
  gl.enableVertexAttribArray(posLoc);
  gl.vertexAttribPointer(posLoc, 3, gl.FLOAT, false, 0, 0);
  
  gl.bindBuffer(gl.ARRAY_BUFFER, normBuffer);
  gl.enableVertexAttribArray(normLoc);
  gl.vertexAttribPointer(normLoc, 3, gl.FLOAT, false, 0, 0);
  
  gl.bindBuffer(gl.ARRAY_BUFFER, texBuffer);
  gl.enableVertexAttribArray(texLoc);
  gl.vertexAttribPointer(texLoc, 2, gl.FLOAT, false, 0, 0);
  
  const aspect = ctx.width / ctx.height;
  const fov = Math.PI / 3.5;
  const near = 0.1, far = 10.0;
  const f = 1.0 / Math.tan(fov / 2);
  const projection = new Float32Array([
    f / aspect, 0, 0, 0,
    0, f, 0, 0,
    0, 0, (far + near) / (near - far), -1,
    0, 0, (2 * far * near) / (near - far), 0
  ]);
  
  const angleY = -55 * Math.PI / 180 + rotation;
  const angleX = 0.3;
  const cy = Math.cos(angleY), sy = Math.sin(angleY);
  const cx = Math.cos(angleX), sx = Math.sin(angleX);
  const modelView = new Float32Array([
    cy, sy * sx, sy * cx, 0,
    0, cx, -sx, 0,
    -sy, cy * sx, cy * cx, 0,
    0, 0, -2.2, 1
  ]);
  
  const normalMatrix = new Float32Array([
    cy, sy * sx, sy * cx,
    0, cx, -sx,
    -sy, cy * sx, cy * cx
  ]);
  
  gl.uniformMatrix4fv(gl.getUniformLocation(tubeProgram, 'uProjection'), false, projection);
  gl.uniformMatrix4fv(gl.getUniformLocation(tubeProgram, 'uModelView'), false, modelView);
  gl.uniformMatrix3fv(gl.getUniformLocation(tubeProgram, 'uNormalMatrix'), false, normalMatrix);
  gl.uniform3f(gl.getUniformLocation(tubeProgram, 'uLightPos'), 3.0, 3.0, 3.0);
  gl.uniform3f(gl.getUniformLocation(tubeProgram, 'uLightPos2'), -2.0, 1.0, 2.0);
  
  gl.activeTexture(gl.TEXTURE1);
  const tubeTexture = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, tubeTexture);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, old.width, old.height, 0, gl.RGBA, gl.UNSIGNED_BYTE, old.data);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  gl.uniform1i(gl.getUniformLocation(tubeProgram, 'uTexture'), 1);
  
  gl.drawArrays(gl.TRIANGLES, 0, positions.length / 3);
  
  const pixels = new Uint8ClampedArray(ctx.width * ctx.height * 4);
  gl.readPixels(0, 0, ctx.width, ctx.height, gl.RGBA, gl.UNSIGNED_BYTE, pixels);
  
  const flipped = new Uint8ClampedArray(ctx.width * ctx.height * 4);
  for (let y = 0; y < ctx.height; y++) {
    for (let x = 0; x < ctx.width; x++) {
      const srcIdx = ((ctx.height - 1 - y) * ctx.width + x) * 4;
      const dstIdx = (y * ctx.width + x) * 4;
      flipped[dstIdx] = pixels[srcIdx];
      flipped[dstIdx + 1] = pixels[srcIdx + 1];
      flipped[dstIdx + 2] = pixels[srcIdx + 2];
      flipped[dstIdx + 3] = pixels[srcIdx + 3];
    }
  }
  
  gl.disable(gl.DEPTH_TEST);
  gl.disableVertexAttribArray(posLoc);
  gl.disableVertexAttribArray(normLoc);
  gl.disableVertexAttribArray(texLoc);
  gl.activeTexture(gl.TEXTURE0);
  gl.bindTexture(gl.TEXTURE_2D, null);
  gl.activeTexture(gl.TEXTURE1);
  gl.bindTexture(gl.TEXTURE_2D, null);
  gl.activeTexture(gl.TEXTURE0);
  gl.deleteTexture(tubeTexture);
  gl.deleteTexture(prevTexture);
  gl.deleteTexture(oldTexture);
  gl.deleteBuffer(bgBuffer);
  gl.deleteBuffer(posBuffer);
  gl.deleteBuffer(normBuffer);
  gl.deleteBuffer(texBuffer);
  gl.deleteProgram(tubeProgram);
  gl.deleteProgram(bgProgram);
  gl.useProgram(null);
  
  return { width: ctx.width, height: ctx.height, data: flipped };
}

function fnM(ctx: FnContext, n: number): Image {
  const prev = getPrevImage(ctx);
  const out = createSolidImage(ctx.width, ctx.height, '#000000');
  
  const hash = (i: number) => {
    const x = Math.sin(i * 127.1 + n * 311.7) * 43758.5453;
    return x - Math.floor(x);
  };
  
  const scale = 12 + hash(0) * 30;
  const freqRatio = 1.02 + hash(1) * 0.1;
  const amp1 = 15 + hash(2) * 80;
  const amp2 = 15 + hash(3) * 80;
  const freq1 = 0.006 + hash(4) * 0.025;
  const angle1 = hash(5) * Math.PI;
  const angle2 = angle1 + (hash(6) - 0.5) * 0.3;
  const phase = hash(7) * Math.PI * 2;
  const hueShift = Math.floor(hash(8) * 360);
  const harmonic1 = 0.3 + hash(9) * 0.7;
  const harmonic2 = hash(10) * 0.5;
  const scaleRatio = 0.95 + hash(11) * 0.1;
  const crossAmp = hash(12) * 40;
  const crossFreq = 0.01 + hash(13) * 0.02;
  
  const cos_a1 = Math.cos(angle1), sin_a1 = Math.sin(angle1);
  const cos_a2 = Math.cos(angle2), sin_a2 = Math.sin(angle2);
  
  for (let y = 0; y < ctx.height; y++) {
    for (let x = 0; x < ctx.width; x++) {
      const rx1 = x * cos_a1 + y * sin_a1;
      const ry1 = -x * sin_a1 + y * cos_a1;
      const rx2 = x * cos_a2 + y * sin_a2;
      const ry2 = -x * sin_a2 + y * cos_a2;
      
      const wobble1 = Math.sin(rx1 * freq1) * amp1 
                    + Math.sin(rx1 * freq1 * 2.1) * amp1 * harmonic1
                    + Math.sin(ry1 * crossFreq) * crossAmp;
      const wobble2 = Math.sin(rx2 * freq1 * freqRatio + phase) * amp2
                    + Math.sin(rx2 * freq1 * freqRatio * 2.3 + phase) * amp2 * harmonic2
                    + Math.sin(ry2 * crossFreq * 1.1) * crossAmp;
      
      const wave1 = ry1 + wobble1;
      const wave2 = ry2 + wobble2;
      
      const line1 = Math.floor(wave1 / scale) % 2;
      const line2 = Math.floor(wave2 / (scale * scaleRatio)) % 2;
      
      const moire = line1 !== line2;
      
      const [r, g, b] = getPixel(prev, x, y);
      
      if (moire) {
        setPixel(out, x, y, 255 - r, 255 - g, 255 - b);
      } else {
        setPixel(out, x, y, r, g, b);
      }
    }
  }
  
  return out;
}

function fnN(ctx: FnContext): Image {
  const prev = getPrevImage(ctx);
  const out = createSolidImage(ctx.width, ctx.height, '#000000');
  
  const scale = Math.min(ctx.width, ctx.height);
  const numLights = 15;
  
  const seed = ctx.images.length * 137.5;
  const hash = (n: number) => {
    const x = Math.sin(n + seed) * 43758.5453;
    return x - Math.floor(x);
  };
  
  const lights: { x: number; y: number; r: number; g: number; b: number; size: number }[] = [];
  for (let i = 0; i < numLights; i++) {
    const px = hash(i * 127.1) * ctx.width;
    const py = hash(i * 311.7) * ctx.height;
    const colorAngle = hash(i * 74.3) * Math.PI * 2;
    const size = 0.03 + hash(i * 191.3) * 0.04;
    lights.push({
      x: px, y: py,
      r: Math.cos(colorAngle) * 0.5 + 0.5,
      g: Math.cos(colorAngle + Math.PI * 2 / 3) * 0.5 + 0.5,
      b: Math.cos(colorAngle + Math.PI * 4 / 3) * 0.5 + 0.5,
      size
    });
  }
  
  const glowRadius = 8;
  
  for (let y = 0; y < ctx.height; y++) {
    for (let x = 0; x < ctx.width; x++) {
      const [pr, pg, pb] = getPixel(prev, x, y);
      const brightness = (pr + pg + pb) / (255 * 3);
      
      let tr = (pr / 255) * 0.3;
      let tg = (pg / 255) * 0.3;
      let tb = (pb / 255) * 0.3;
      
      if (brightness > 0.3) {
        let glowSum = 0;
        let glowR = 0, glowG = 0, glowB = 0;
        for (let dy = -glowRadius; dy <= glowRadius; dy += 2) {
          for (let dx = -glowRadius; dx <= glowRadius; dx += 2) {
            const sx = Math.max(0, Math.min(ctx.width - 1, x + dx));
            const sy = Math.max(0, Math.min(ctx.height - 1, y + dy));
            const [sr, sg, sb] = getPixel(prev, sx, sy);
            const sBright = (sr + sg + sb) / (255 * 3);
            if (sBright > 0.3) {
              const dist = Math.sqrt(dx * dx + dy * dy);
              const weight = Math.max(0, 1 - dist / glowRadius);
              glowR += (sr / 255) * weight * sBright;
              glowG += (sg / 255) * weight * sBright;
              glowB += (sb / 255) * weight * sBright;
              glowSum += weight;
            }
          }
        }
        if (glowSum > 0) {
          tr += (glowR / glowSum) * brightness * 1.5;
          tg += (glowG / glowSum) * brightness * 1.5;
          tb += (glowB / glowSum) * brightness * 1.5;
        }
      }
      
      for (const light of lights) {
        const dx = (x - light.x) / scale;
        const dy = (y - light.y) / scale;
        const distSq = dx * dx + dy * dy;
        const glow = (light.size * light.size) / Math.max(0.0001, distSq);
        const cappedGlow = Math.min(1, glow);
        tr += cappedGlow * light.r;
        tg += cappedGlow * light.g;
        tb += cappedGlow * light.b;
      }
      
      setPixel(out, x, y,
        Math.min(255, Math.floor(tr * 255)),
        Math.min(255, Math.floor(tg * 255)),
        Math.min(255, Math.floor(tb * 255))
      );
    }
  }
  
  return out;
}

function fnO(ctx: FnContext, n: number): Image {
  const prev = getPrevImage(ctx);
  
  const multiplier = 1.5 + n * 0.8;
  
  const nextPow2 = (v: number) => {
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    return v + 1;
  };
  
  const fftW = nextPow2(ctx.width);
  const fftH = nextPow2(ctx.height);
  
  const fftRow = new FFT(fftW);
  const fftCol = new FFT(fftH);
  
  const processChannel = (channel: Float32Array, mult: number, phaseShift: number): Float32Array => {
    const data = new Float64Array(fftW * fftH * 2);
    
    for (let y = 0; y < ctx.height; y++) {
      for (let x = 0; x < ctx.width; x++) {
        data[(y * fftW + x) * 2] = channel[y * ctx.width + x];
      }
    }
    
    const rowIn = fftRow.createComplexArray();
    const rowOut = fftRow.createComplexArray();
    for (let y = 0; y < fftH; y++) {
      for (let x = 0; x < fftW; x++) {
        rowIn[x * 2] = data[(y * fftW + x) * 2];
        rowIn[x * 2 + 1] = data[(y * fftW + x) * 2 + 1];
      }
      fftRow.transform(rowOut, rowIn);
      for (let x = 0; x < fftW; x++) {
        data[(y * fftW + x) * 2] = rowOut[x * 2];
        data[(y * fftW + x) * 2 + 1] = rowOut[x * 2 + 1];
      }
    }
    
    const colIn = fftCol.createComplexArray();
    const colOut = fftCol.createComplexArray();
    for (let x = 0; x < fftW; x++) {
      for (let y = 0; y < fftH; y++) {
        colIn[y * 2] = data[(y * fftW + x) * 2];
        colIn[y * 2 + 1] = data[(y * fftW + x) * 2 + 1];
      }
      fftCol.transform(colOut, colIn);
      for (let y = 0; y < fftH; y++) {
        data[(y * fftW + x) * 2] = colOut[y * 2];
        data[(y * fftW + x) * 2 + 1] = colOut[y * 2 + 1];
      }
    }
    
    const cx = fftW / 2;
    const cy = fftH / 2;
    const maxFreqDist = Math.sqrt(cx * cx + cy * cy);
    const wrapLimit = 255 * fftW * fftH / 4;
    
    for (let y = 0; y < fftH; y++) {
      for (let x = 0; x < fftW; x++) {
        const i = (y * fftW + x) * 2;
        const re = data[i];
        const im = data[i + 1];
        const mag = Math.sqrt(re * re + im * im);
        let phase = Math.atan2(im, re);
        
        const dx = (x < cx ? x : x - fftW);
        const dy = (y < cy ? y : y - fftH);
        const freqDist = Math.sqrt(dx * dx + dy * dy) / maxFreqDist;
        
        let newMag = mag * mult;
        while (newMag > wrapLimit) {
          newMag = Math.abs(newMag - wrapLimit * 2);
        }
        
        phase += phaseShift * freqDist;
        
        data[i] = newMag * Math.cos(phase);
        data[i + 1] = newMag * Math.sin(phase);
      }
    }
    
    for (let x = 0; x < fftW; x++) {
      for (let y = 0; y < fftH; y++) {
        colIn[y * 2] = data[(y * fftW + x) * 2];
        colIn[y * 2 + 1] = data[(y * fftW + x) * 2 + 1];
      }
      fftCol.inverseTransform(colOut, colIn);
      for (let y = 0; y < fftH; y++) {
        data[(y * fftW + x) * 2] = colOut[y * 2];
        data[(y * fftW + x) * 2 + 1] = colOut[y * 2 + 1];
      }
    }
    
    for (let y = 0; y < fftH; y++) {
      for (let x = 0; x < fftW; x++) {
        rowIn[x * 2] = data[(y * fftW + x) * 2];
        rowIn[x * 2 + 1] = data[(y * fftW + x) * 2 + 1];
      }
      fftRow.inverseTransform(rowOut, rowIn);
      for (let x = 0; x < fftW; x++) {
        data[(y * fftW + x) * 2] = rowOut[x * 2];
      }
    }
    
    const result = new Float32Array(ctx.width * ctx.height);
    for (let y = 0; y < ctx.height; y++) {
      for (let x = 0; x < ctx.width; x++) {
        result[y * ctx.width + x] = data[(y * fftW + x) * 2];
      }
    }
    
    return result;
  };
  
  const rIn = new Float32Array(ctx.width * ctx.height);
  const gIn = new Float32Array(ctx.width * ctx.height);
  const bIn = new Float32Array(ctx.width * ctx.height);
  
  for (let y = 0; y < ctx.height; y++) {
    for (let x = 0; x < ctx.width; x++) {
      const [r, g, b] = getPixel(prev, x, y);
      const idx = y * ctx.width + x;
      rIn[idx] = r;
      gIn[idx] = g;
      bIn[idx] = b;
    }
  }
  
  const rOut = processChannel(rIn, multiplier, 0);
  const gOut = processChannel(gIn, multiplier * 1.1, Math.PI * 0.1);
  const bOut = processChannel(bIn, multiplier * 0.9, -Math.PI * 0.1);
  
  const out = createSolidImage(ctx.width, ctx.height, '#000000');
  for (let y = 0; y < ctx.height; y++) {
    for (let x = 0; x < ctx.width; x++) {
      const idx = y * ctx.width + x;
      let r = rOut[idx];
      let g = gOut[idx];
      let b = bOut[idx];
      
      r = ((r % 256) + 256) % 256;
      g = ((g % 256) + 256) % 256;
      b = ((b % 256) + 256) % 256;
      
      setPixel(out, x, y, Math.round(r), Math.round(g), Math.round(b));
    }
  }
  
  return out;
}

function fnP(ctx: FnContext, n: number): Image {
  const prev = getPrevImage(ctx);
  const out = createSolidImage(ctx.width, ctx.height, '#000000');
  
  const cellSize = Math.max(2, Math.min(n + 1, 50));
  
  for (let by = 0; by < ctx.height; by += cellSize) {
    for (let bx = 0; bx < ctx.width; bx += cellSize) {
      let sumR = 0, sumG = 0, sumB = 0, count = 0;
      let maxSat = 0;
      let mostSatR = 0, mostSatG = 0, mostSatB = 0;
      
      for (let y = by; y < by + cellSize && y < ctx.height; y++) {
        for (let x = bx; x < bx + cellSize && x < ctx.width; x++) {
          const [r, g, b] = getPixel(prev, x, y);
          sumR += r;
          sumG += g;
          sumB += b;
          count++;
          
          const [h, s, l] = rgbToHsl(r, g, b);
          if (s > maxSat) {
            maxSat = s;
            mostSatR = r;
            mostSatG = g;
            mostSatB = b;
          }
        }
      }
      
      const avgR = Math.round(sumR / count);
      const avgG = Math.round(sumG / count);
      const avgB = Math.round(sumB / count);
      
      for (let y = by; y < by + cellSize && y < ctx.height; y++) {
        for (let x = bx; x < bx + cellSize && x < ctx.width; x++) {
          const localX = x - bx;
          const localY = y - by;
          const isTopLeft = localX + localY < cellSize;
          
          if (isTopLeft) {
            setPixel(out, x, y, avgR, avgG, avgB);
          } else {
            setPixel(out, x, y, mostSatR, mostSatG, mostSatB);
          }
        }
      }
    }
  }
  
  return out;
}

let emeraldScene: THREE.Scene | null = null;
let emeraldRenderer: THREE.WebGLRenderer | null = null;
let emeraldCamera: THREE.PerspectiveCamera | null = null;
let emeraldModel: THREE.Group | null = null;
let emeraldModelLoaded = false;
let emeraldLoadPromise: Promise<void> | null = null;
let emeraldComposer: EffectComposer | null = null;

function initEmeraldScene(width: number, height: number) {
  if (!emeraldRenderer || emeraldRenderer.domElement.width !== width || emeraldRenderer.domElement.height !== height) {
    if (emeraldRenderer) {
      emeraldRenderer.dispose();
    }
    
    emeraldRenderer = new THREE.WebGLRenderer({ 
      alpha: true, 
      antialias: true,
      premultipliedAlpha: false,
      preserveDrawingBuffer: true,
      powerPreference: 'high-performance',
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
      './emerald.glb',
      (gltf) => {
        emeraldModel = gltf.scene;
        
        const box = new THREE.Box3().setFromObject(emeraldModel);
        const center = box.getCenter(new THREE.Vector3());
        const size = box.getSize(new THREE.Vector3());
        const maxDim = Math.max(size.x, size.y, size.z);
        const scale = 2.0 / maxDim;
        
        emeraldModel.position.set(-center.x * scale, -center.y * scale, -center.z * scale);
        emeraldModel.scale.setScalar(scale);
        
        emeraldModelLoaded = true;
        resolve();
      },
      undefined,
      (error) => {
        console.error('Error loading emerald model:', error);
        reject(error);
      }
    );
  });
  
  return emeraldLoadPromise;
}

// Start loading immediately when module loads
export const emeraldReady = loadEmeraldModel();

function fnE(ctx: FnContext): Image {
  const prev = getPrevImage(ctx);
  
  if (!emeraldModelLoaded || !emeraldModel) {
    throw new Error('Emerald model not loaded - await emeraldReady before rendering');
  }
  
  initEmeraldScene(ctx.width, ctx.height);
  
  while (emeraldScene!.children.length > 0) {
    emeraldScene!.remove(emeraldScene!.children[0]);
  }
  
  // Create background texture - preserve original colors
  const bgTexture = new THREE.DataTexture(
    prev.data,
    prev.width,
    prev.height,
    THREE.RGBAFormat
  );
  bgTexture.colorSpace = THREE.SRGBColorSpace;
  bgTexture.needsUpdate = true;
  bgTexture.flipY = true;
  emeraldScene!.background = bgTexture;

  // Create environment map for reflections
  const pmremGenerator = new THREE.PMREMGenerator(emeraldRenderer!);
  pmremGenerator.compileEquirectangularShader();
  const envRT = pmremGenerator.fromEquirectangular(bgTexture);
  emeraldScene!.environment = envRT.texture;

  // Low ambient for more contrast
  const ambientLight = new THREE.AmbientLight(0xffffff, 0.15);
  emeraldScene!.add(ambientLight);
  
  // Strong key light for highlights
  const keyLight = new THREE.DirectionalLight(0xffffff, 5.0);
  keyLight.position.set(5, 8, 10);
  emeraldScene!.add(keyLight);
  
  // Weak fill light - keeps shadows darker
  const fillLight = new THREE.DirectionalLight(0xeeffee, 0.8);
  fillLight.position.set(-5, 3, 8);
  emeraldScene!.add(fillLight);
  
  // Rim lights for edge highlights from multiple angles
  const rimLight = new THREE.DirectionalLight(0xffffff, 2.5);
  rimLight.position.set(0, -2, 8);
  emeraldScene!.add(rimLight);
  
  const rimLight2 = new THREE.DirectionalLight(0xffffff, 2.0);
  rimLight2.position.set(-6, 0, -2);
  emeraldScene!.add(rimLight2);
  
  const rimLight3 = new THREE.DirectionalLight(0xffffff, 2.0);
  rimLight3.position.set(6, 0, -2);
  emeraldScene!.add(rimLight3);
  
  // Seeded random for deterministic light positions based on image count
  const seed = ctx.images.length * 137.5;
  const hash = (n: number) => {
    const x = Math.sin(n + seed) * 43758.5453;
    return x - Math.floor(x);
  };
  
  // Dramatic point lights - fewer but more intense for contrast
  const numLights = 8;
  for (let i = 0; i < numLights; i++) {
    const angle = hash(i * 127.1) * Math.PI * 2;
    const elevation = hash(i * 311.7) * Math.PI * 0.5 + 0.3;
    const distance = 3 + hash(i * 74.3) * 5;
    
    const px = Math.cos(angle) * Math.cos(elevation) * distance;
    const py = Math.sin(elevation) * distance + 2;
    const pz = Math.sin(angle) * Math.cos(elevation) * distance + 4;
    
    const intensity = 15.0 + hash(i * 191.3) * 25.0;
    const light = new THREE.PointLight(0xffffff, intensity, 30);
    light.decay = 2;
    light.position.set(px, py, pz);
    emeraldScene!.add(light);
  }

  // Glass emerald material - pronounced edges with sheen
  const emeraldMaterial = new THREE.MeshPhysicalMaterial({
    color: new THREE.Color(0.3, 0.95, 0.5),
    metalness: 0.0,
    roughness: 0.0,
    transmission: 0.92,
    thickness: 0.4,
    ior: 1.3,
    envMapIntensity: 0.25,
    clearcoat: 1.0,
    clearcoatRoughness: 0.0,
    transparent: true,
    side: THREE.DoubleSide,
    flatShading: true,
    attenuationColor: new THREE.Color(0.0, 0.75, 0.25),
    attenuationDistance: 0.4,
    specularIntensity: 1.5,
    specularColor: new THREE.Color(1, 1, 1),
    reflectivity: 0.3,
    sheen: 0.5,
    sheenRoughness: 0.2,
    sheenColor: new THREE.Color(0.8, 1.0, 0.9),
  });
  
  // Corner positions extracted from the emerald geometry
  // Girdle corners (8 points around y ≈ 0.07)
  const girdleCorners = [
    [0.0, 0.064, -0.323],    // front
    [0.227, 0.069, -0.226],  // front-right
    [0.322, 0.063, 0.0],     // right
    [0.224, 0.072, 0.228],   // back-right
    [0.0, 0.064, 0.322],     // back
    [-0.227, 0.07, 0.226],   // back-left
    [-0.322, 0.07, 0.0],     // left (inferred)
    [-0.225, 0.069, -0.227], // front-left
  ];
  
  // Crown corners (upper facet intersections around y ≈ 0.176)
  const crownCorners = [
    [-0.169, 0.176, -0.092],
    [-0.089, 0.176, 0.174],
    [0.169, 0.176, -0.092],  // mirrored
    [0.089, 0.176, 0.174],   // mirrored
    [0.0, 0.176, -0.18],     // front center
    [0.0, 0.176, 0.18],      // back center
    [-0.15, 0.176, 0.0],     // left center
    [0.15, 0.176, 0.0],      // right center
  ];
  
  // Create subtle sparkle sprite texture
  const sparkleCanvas = document.createElement('canvas');
  sparkleCanvas.width = 64;
  sparkleCanvas.height = 64;
  const sctx = sparkleCanvas.getContext('2d')!;
  const cx = 32, cy = 32;
  
  // Soft subtle glow
  const gradient = sctx.createRadialGradient(cx, cy, 0, cx, cy, 32);
  gradient.addColorStop(0, 'rgba(255, 255, 255, 0.7)');
  gradient.addColorStop(0.15, 'rgba(255, 255, 255, 0.3)');
  gradient.addColorStop(0.4, 'rgba(255, 255, 255, 0.08)');
  gradient.addColorStop(1, 'rgba(255, 255, 255, 0)');
  sctx.fillStyle = gradient;
  sctx.fillRect(0, 0, 64, 64);
  
  // Very subtle cross rays
  sctx.globalCompositeOperation = 'lighter';
  const rayGradient = sctx.createLinearGradient(0, cy, 64, cy);
  rayGradient.addColorStop(0, 'rgba(255,255,255,0)');
  rayGradient.addColorStop(0.35, 'rgba(255,255,255,0.08)');
  rayGradient.addColorStop(0.5, 'rgba(255,255,255,0.2)');
  rayGradient.addColorStop(0.65, 'rgba(255,255,255,0.08)');
  rayGradient.addColorStop(1, 'rgba(255,255,255,0)');
  sctx.fillStyle = rayGradient;
  sctx.fillRect(0, cy-1, 64, 2);
  
  const rayGradientV = sctx.createLinearGradient(cx, 0, cx, 64);
  rayGradientV.addColorStop(0, 'rgba(255,255,255,0)');
  rayGradientV.addColorStop(0.35, 'rgba(255,255,255,0.08)');
  rayGradientV.addColorStop(0.5, 'rgba(255,255,255,0.2)');
  rayGradientV.addColorStop(0.65, 'rgba(255,255,255,0.08)');
  rayGradientV.addColorStop(1, 'rgba(255,255,255,0)');
  sctx.fillStyle = rayGradientV;
  sctx.fillRect(cx-1, 0, 2, 64);
  
  const sparkleTexture = new THREE.CanvasTexture(sparkleCanvas);
  
  const createSparkleMaterial = () => new THREE.SpriteMaterial({
    map: sparkleTexture,
    color: 0xffffff,
    transparent: true,
    opacity: 0.6,
    blending: THREE.AdditiveBlending,
    depthTest: false,
    depthWrite: false,
  });

  const addEmerald = (x: number, y: number, scale: number, logGeometry: boolean = false) => {
    const gem = emeraldModel!.clone();
    
    gem.traverse((child) => {
      if (child instanceof THREE.Mesh) {
        const geom = child.geometry.clone();
        geom.computeVertexNormals();
        child.geometry = geom;
        child.material = emeraldMaterial;
        child.renderOrder = 1;
        
        // Log geometry info for the first emerald
        if (logGeometry) {
          const positions = geom.attributes.position;
          const normals = geom.attributes.normal;
          
          console.log('=== EMERALD GEOMETRY ===');
          console.log('Vertex count:', positions.count);
          console.log('Triangle count:', positions.count / 3);
          
          // Find unique vertices and their positions
          const uniqueVerts = new Map<string, {pos: number[], count: number, indices: number[]}>();
          
          for (let i = 0; i < positions.count; i++) {
            const px = positions.getX(i).toFixed(3);
            const py = positions.getY(i).toFixed(3);
            const pz = positions.getZ(i).toFixed(3);
            const key = `${px},${py},${pz}`;
            
            if (!uniqueVerts.has(key)) {
              uniqueVerts.set(key, {pos: [parseFloat(px), parseFloat(py), parseFloat(pz)], count: 0, indices: []});
            }
            uniqueVerts.get(key)!.count++;
            uniqueVerts.get(key)!.indices.push(i);
          }
          
          console.log('Unique vertex positions:', uniqueVerts.size);
          
          // Sort by how many triangles share this vertex (corners have more)
          const sorted = [...uniqueVerts.entries()].sort((a, b) => b[1].count - a[1].count);
          
          console.log('\nTop 20 most-shared vertices (likely corners):');
          sorted.slice(0, 20).forEach(([key, data], i) => {
            console.log(`  ${i+1}. [${data.pos.join(', ')}] shared by ${data.count} triangles`);
          });
          
          // Also log bounding box
          geom.computeBoundingBox();
          const bb = geom.boundingBox!;
          console.log('\nBounding box:');
          console.log('  min:', bb.min.x.toFixed(3), bb.min.y.toFixed(3), bb.min.z.toFixed(3));
          console.log('  max:', bb.max.x.toFixed(3), bb.max.y.toFixed(3), bb.max.z.toFixed(3));
        }
      }
    });
    
    gem.scale.setScalar(scale * 3.0);
    gem.position.set(x, y, 0);
    emeraldScene!.add(gem);
    
    // Add sparkle sprites at corner positions
    const allCorners = [...girdleCorners, ...crownCorners];
    const scaleFactor = scale * 3.0;
    
    allCorners.forEach((corner, i) => {
      // Only show sparkles on front-facing corners (positive z)
      if (corner[2] > 0) {
        const sprite = new THREE.Sprite(createSparkleMaterial());
        const sparkleSize = 0.03 + (i % 3) * 0.01;
        sprite.scale.set(sparkleSize * scaleFactor, sparkleSize * scaleFactor, 1);
        // Position at the corner, pushed forward a bit to be visible
        sprite.position.set(
          x + corner[0] * scaleFactor,
          y + corner[1] * scaleFactor,
          corner[2] * scaleFactor + 0.02
        );
        sprite.renderOrder = 10;
        emeraldScene!.add(sprite);
      }
    });
  };
  
  addEmerald(0, 0, 1.0, false);  // Main emerald
  addEmerald(-2.5, 0, 0.5);
  addEmerald(2.5, 0, 0.5);
  addEmerald(-1.5, 1.2, 0.35);
  addEmerald(1.5, 1.2, 0.35);
  addEmerald(-1.5, -1.2, 0.35);
  addEmerald(1.5, -1.2, 0.35);
  
  // Setup bloom passes
  emeraldComposer!.passes = [];
  const renderPass = new RenderPass(emeraldScene!, emeraldCamera!);
  emeraldComposer!.addPass(renderPass);
  
  const bloomPass = new UnrealBloomPass(
    new THREE.Vector2(ctx.width, ctx.height),
    0.3,
    0.15,
    0.97
  );
  emeraldComposer!.addPass(bloomPass);
  
  // Render multiple times - transmission needs multiple passes to converge
  for (let i = 0; i < 6; i++) {
    emeraldComposer!.render();
  }
  
  const glContext = emeraldRenderer!.getContext();
  const pixels = new Uint8ClampedArray(ctx.width * ctx.height * 4);
  glContext.readPixels(0, 0, ctx.width, ctx.height, glContext.RGBA, glContext.UNSIGNED_BYTE, pixels);
  
  const flipped = new Uint8ClampedArray(ctx.width * ctx.height * 4);
  for (let y = 0; y < ctx.height; y++) {
    for (let x = 0; x < ctx.width; x++) {
      const srcIdx = ((ctx.height - 1 - y) * ctx.width + x) * 4;
      const dstIdx = (y * ctx.width + x) * 4;
      flipped[dstIdx] = pixels[srcIdx];
      flipped[dstIdx + 1] = pixels[srcIdx + 1];
      flipped[dstIdx + 2] = pixels[srcIdx + 2];
      flipped[dstIdx + 3] = pixels[srcIdx + 3];
    }
  }
  
  // Clean up
  bgTexture.dispose();
  envRT.texture.dispose();
  pmremGenerator.dispose();
  
  return { width: ctx.width, height: ctx.height, data: flipped };
}

function fnR(ctx: FnContext): Image {
  const prev = getPrevImage(ctx);
  const gl = initWebGL(ctx.width, ctx.height);
  
  gl.activeTexture(gl.TEXTURE0);
  gl.bindTexture(gl.TEXTURE_2D, null);
  gl.activeTexture(gl.TEXTURE1);
  gl.bindTexture(gl.TEXTURE_2D, null);
  gl.activeTexture(gl.TEXTURE0);
  
  const vertexShader = `
    attribute vec2 position;
    varying vec2 vUV;
    void main() {
      vUV = vec2(position.x * 0.5 + 0.5, 1.0 - (position.y * 0.5 + 0.5));
      gl_Position = vec4(position, 0.0, 1.0);
    }
  `;
  
  const fragmentShader = `
    precision highp float;
    uniform sampler2D uTexture;
    uniform vec2 uResolution;
    varying vec2 vUV;
    
    #define PI 3.14159265359
    
    // Ray-plane intersection
    float intersectPlane(vec3 ro, vec3 rd, vec3 planeNormal, float planeD) {
      float denom = dot(rd, planeNormal);
      if (abs(denom) < 0.0001) return -1.0;
      return -(dot(ro, planeNormal) + planeD) / denom;
    }
    
    void main() {
      vec2 uv = vUV;
      float aspect = uResolution.x / uResolution.y;
      
      // Camera setup - looking into the room
      vec3 ro = vec3(0.0, 0.0, 2.5); // camera position
      vec2 screenPos = (uv - 0.5) * 2.0;
      screenPos.x *= aspect;
      vec3 rd = normalize(vec3(screenPos * 0.8, -1.0)); // ray direction
      
      // Room dimensions
      float roomSize = 2.0;
      float roomDepth = 3.0;
      
      // Light position (center of room)
      vec3 lightPos = vec3(0.0, 0.0, -roomDepth * 0.4);
      
      // Find closest intersection
      float tMin = 1000.0;
      vec3 hitNormal = vec3(0.0);
      vec2 texCoord = vec2(0.0);
      
      // Back wall (z = -roomDepth)
      float t = intersectPlane(ro, rd, vec3(0.0, 0.0, 1.0), roomDepth);
      if (t > 0.0 && t < tMin) {
        vec3 hit = ro + rd * t;
        if (abs(hit.x) < roomSize && abs(hit.y) < roomSize) {
          tMin = t;
          hitNormal = vec3(0.0, 0.0, 1.0);
          texCoord = vec2(hit.x / roomSize * 0.5 + 0.5, hit.y / roomSize * 0.5 + 0.5);
        }
      }
      
      // Left wall (x = -roomSize)
      t = intersectPlane(ro, rd, vec3(1.0, 0.0, 0.0), roomSize);
      if (t > 0.0 && t < tMin) {
        vec3 hit = ro + rd * t;
        if (hit.z > -roomDepth && hit.z < ro.z && abs(hit.y) < roomSize) {
          tMin = t;
          hitNormal = vec3(1.0, 0.0, 0.0);
          texCoord = vec2((hit.z + roomDepth) / (ro.z + roomDepth), hit.y / roomSize * 0.5 + 0.5);
        }
      }
      
      // Right wall (x = roomSize)
      t = intersectPlane(ro, rd, vec3(-1.0, 0.0, 0.0), roomSize);
      if (t > 0.0 && t < tMin) {
        vec3 hit = ro + rd * t;
        if (hit.z > -roomDepth && hit.z < ro.z && abs(hit.y) < roomSize) {
          tMin = t;
          hitNormal = vec3(-1.0, 0.0, 0.0);
          texCoord = vec2((hit.z + roomDepth) / (ro.z + roomDepth), hit.y / roomSize * 0.5 + 0.5);
        }
      }
      
      // Floor (y = -roomSize)
      t = intersectPlane(ro, rd, vec3(0.0, 1.0, 0.0), roomSize);
      if (t > 0.0 && t < tMin) {
        vec3 hit = ro + rd * t;
        if (hit.z > -roomDepth && hit.z < ro.z && abs(hit.x) < roomSize) {
          tMin = t;
          hitNormal = vec3(0.0, 1.0, 0.0);
          texCoord = vec2(hit.x / roomSize * 0.5 + 0.5, 1.0 - (hit.z + roomDepth) / (ro.z + roomDepth));
        }
      }
      
      // Ceiling (y = roomSize)
      t = intersectPlane(ro, rd, vec3(0.0, -1.0, 0.0), roomSize);
      if (t > 0.0 && t < tMin) {
        vec3 hit = ro + rd * t;
        if (hit.z > -roomDepth && hit.z < ro.z && abs(hit.x) < roomSize) {
          tMin = t;
          hitNormal = vec3(0.0, -1.0, 0.0);
          texCoord = vec2(hit.x / roomSize * 0.5 + 0.5, (hit.z + roomDepth) / (ro.z + roomDepth));
        }
      }
      
      vec3 color = vec3(0.0);
      
      if (tMin < 1000.0) {
        vec3 hitPos = ro + rd * tMin;
        
        // Sample texture
        texCoord = clamp(texCoord, 0.0, 1.0);
        vec3 texColor = texture2D(uTexture, texCoord).rgb;
        
        // Lighting
        vec3 toLight = lightPos - hitPos;
        float lightDist = length(toLight);
        vec3 lightDir = toLight / lightDist;
        
        // Diffuse
        float diff = max(dot(hitNormal, lightDir), 0.0);
        
        // Attenuation
        float attenuation = 1.0 / (1.0 + 0.1 * lightDist + 0.05 * lightDist * lightDist);
        
        // Ambient
        float ambient = 0.25;
        
        color = texColor * (ambient + diff * attenuation * 1.5);
      }
      
      gl_FragColor = vec4(color, 1.0);
    }
  `;
  
  const program = createShaderProgram(gl, vertexShader, fragmentShader);
  gl.useProgram(program);
  
  const texture = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, texture);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, prev.width, prev.height, 0, gl.RGBA, gl.UNSIGNED_BYTE, prev.data);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  
  const vertices = new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]);
  const buffer = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
  gl.bufferData(gl.ARRAY_BUFFER, vertices, gl.STATIC_DRAW);
  
  const positionLoc = gl.getAttribLocation(program, 'position');
  gl.enableVertexAttribArray(positionLoc);
  gl.vertexAttribPointer(positionLoc, 2, gl.FLOAT, false, 0, 0);
  
  gl.uniform1i(gl.getUniformLocation(program, 'uTexture'), 0);
  gl.uniform2f(gl.getUniformLocation(program, 'uResolution'), ctx.width, ctx.height);
  
  gl.viewport(0, 0, ctx.width, ctx.height);
  gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  
  const pixels = new Uint8ClampedArray(ctx.width * ctx.height * 4);
  gl.readPixels(0, 0, ctx.width, ctx.height, gl.RGBA, gl.UNSIGNED_BYTE, pixels);
  
  const flipped = new Uint8ClampedArray(ctx.width * ctx.height * 4);
  for (let y = 0; y < ctx.height; y++) {
    for (let x = 0; x < ctx.width; x++) {
      const srcIdx = ((ctx.height - 1 - y) * ctx.width + x) * 4;
      const dstIdx = (y * ctx.width + x) * 4;
      flipped[dstIdx] = pixels[srcIdx];
      flipped[dstIdx + 1] = pixels[srcIdx + 1];
      flipped[dstIdx + 2] = pixels[srcIdx + 2];
      flipped[dstIdx + 3] = pixels[srcIdx + 3];
    }
  }
  
  gl.deleteTexture(texture);
  gl.deleteBuffer(buffer);
  gl.deleteProgram(program);
  
  return { width: ctx.width, height: ctx.height, data: flipped };
}

function fnS(ctx: FnContext, old: Image, size: number): Image {
  const prev = getPrevImage(ctx);
  const out = createSolidImage(ctx.width, ctx.height, '#000000');
  
  const popcount = (n: number): number => {
    let count = 0;
    while (n) {
      count += n & 1;
      n >>= 1;
    }
    return count;
  };
  
  const resolution = Math.pow(2, Math.floor((size - 1) / 10) + 2);
  
  const v0x = ctx.width / 2, v0y = 0;
  const v1x = 0, v1y = ctx.height;
  const v2x = ctx.width, v2y = ctx.height;
  
  const denom = (v1y - v2y) * (v0x - v2x) + (v2x - v1x) * (v0y - v2y);
  
  for (let y = 0; y < ctx.height; y++) {
    for (let x = 0; x < ctx.width; x++) {
      const baryA = ((v1y - v2y) * (x - v2x) + (v2x - v1x) * (y - v2y)) / denom;
      const baryB = ((v2y - v0y) * (x - v2x) + (v0x - v2x) * (y - v2y)) / denom;
      const baryC = 1 - baryA - baryB;
      
      const [pr, pg, pb] = getPixel(prev, x, y);
      
      if (baryA < 0 || baryB < 0 || baryC < 0) {
        setPixel(out, x, y, pr, pg, pb);
        continue;
      }
      
      const ai = Math.floor(baryA * resolution);
      const bi = Math.floor(baryB * resolution);
      const ci = Math.floor(baryC * resolution);
      
      const overlap = (ai & bi) | (bi & ci) | (ai & ci);
      const level = popcount(overlap) % 6;
      
      const [or, og, ob] = getPixel(old, x, y);
      
      let r: number, g: number, b: number;
      
      switch (level) {
        case 0: {
          r = or; g = og; b = ob;
          break;
        }
        case 1: {
          r = 255 - pr; g = 255 - pg; b = 255 - pb;
          break;
        }
        case 2: {
          const [h, s, l] = rgbToHsl(pr, pg, pb);
          [r, g, b] = hslToRgb((h + 120) % 360, s, l);
          break;
        }
        case 3: {
          const [h, s, l] = rgbToHsl(pr, pg, pb);
          [r, g, b] = hslToRgb((h + 240) % 360, s, l);
          break;
        }
        case 4: {
          const gray = Math.round(pr * 0.299 + pg * 0.587 + pb * 0.114);
          const contrast = gray < 128 ? gray * 0.5 : 128 + (gray - 128) * 1.5;
          r = g = b = Math.max(0, Math.min(255, Math.round(contrast)));
          break;
        }
        case 5: {
          const [h, s, l] = rgbToHsl(pr, pg, pb);
          [r, g, b] = hslToRgb((h + 60) % 360, Math.min(1, s * 1.5), l);
          break;
        }
        default:
          r = pr; g = pg; b = pb;
      }
      
      setPixel(out, x, y, r, g, b);
    }
  }
  
  return out;
}

function fnSkewLeft(ctx: FnContext): Image {
  const prev = getPrevImage(ctx);
  const out = createSolidImage(ctx.width, ctx.height, '#000000');
  
  const skewAmount = Math.tan(20 * Math.PI / 180) * ctx.height / 2;
  
  for (let y = 0; y < ctx.height; y++) {
    const rowSkew = skewAmount * (1 - 2 * y / ctx.height);
    for (let x = 0; x < ctx.width; x++) {
      const sx = ((x + rowSkew) % ctx.width + ctx.width) % ctx.width;
      const [r, g, b] = getPixel(prev, Math.floor(sx), y);
      setPixel(out, x, y, r, g, b);
    }
  }
  
  return out;
}

function fnSkewRight(ctx: FnContext): Image {
  const prev = getPrevImage(ctx);
  const out = createSolidImage(ctx.width, ctx.height, '#000000');
  
  const skewAmount = Math.tan(20 * Math.PI / 180) * ctx.height / 2;
  
  for (let y = 0; y < ctx.height; y++) {
    const rowSkew = skewAmount * (2 * y / ctx.height - 1);
    for (let x = 0; x < ctx.width; x++) {
      const sx = ((x + rowSkew) % ctx.width + ctx.width) % ctx.width;
      const [r, g, b] = getPixel(prev, Math.floor(sx), y);
      setPixel(out, x, y, r, g, b);
    }
  }
  
  return out;
}

function fnT(ctx: FnContext, n: number): Image {
  const prev = getPrevImage(ctx);
  
  const baseGridSize = 8;
  const heightMultiplier = 0.2 + (n / 68) * 2.0;
  const seed = ctx.images.length * 137.5;
  const hash = (i: number) => {
    const x = Math.sin(i + seed) * 43758.5453;
    return x - Math.floor(x);
  };
  
  const aspect = ctx.width / ctx.height;
  const approxCellSize = Math.min(ctx.width, ctx.height) / baseGridSize;
  const cols = Math.max(1, Math.round(ctx.width / approxCellSize));
  const rows = Math.max(1, Math.round(ctx.height / approxCellSize));
  
  const renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
  renderer.setSize(ctx.width, ctx.height);
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.VSMShadowMap;
  
  const scene = new THREE.Scene();
  
  const bgTexture = new THREE.DataTexture(prev.data, prev.width, prev.height, THREE.RGBAFormat);
  bgTexture.needsUpdate = true;
  bgTexture.flipY = true;
  scene.background = bgTexture;
  
  const fov = 50;
  const camera = new THREE.PerspectiveCamera(fov, aspect, 0.1, 100);
  const frustumHeight = 2;
  const frustumWidth = frustumHeight * aspect;
  const camZ = frustumHeight / (2 * Math.tan((fov * Math.PI / 180) / 2));
  camera.position.set(0, 0, camZ);
  camera.lookAt(0, 0, 0);
  
  const ambient = new THREE.AmbientLight(0xffffff, 0.5);
  scene.add(ambient);
  
  const light = new THREE.DirectionalLight(0xffffff, 2);
  light.position.set(3, 3, 5);
  light.castShadow = true;
  light.shadow.mapSize.width = 2048;
  light.shadow.mapSize.height = 2048;
  const d = Math.max(frustumWidth, frustumHeight);
  light.shadow.camera.left = -d;
  light.shadow.camera.right = d;
  light.shadow.camera.top = d;
  light.shadow.camera.bottom = -d;
  light.shadow.camera.near = 0.1;
  light.shadow.camera.far = 20;
  light.shadow.radius = 4;
  scene.add(light);
  
  const cellWidth = frustumWidth / cols;
  const cellHeight = frustumHeight / rows;
  
  for (let row = 0; row < rows; row++) {
    for (let col = 0; col < cols; col++) {
      const idx = row * cols + col;
      const depth = (0.05 + hash(idx * 127.1) * 0.4) * heightMultiplier;
      const cx = (col + 0.5) * cellWidth - frustumWidth / 2;
      const cy = (row + 0.5) * cellHeight - frustumHeight / 2;
      
      const geometry = new THREE.BoxGeometry(cellWidth, cellHeight, depth);
      
      const uvs = geometry.attributes.uv;
      const positions = geometry.attributes.position;
      const normals = geometry.attributes.normal;
      const texX0 = col / cols, texX1 = (col + 1) / cols;
      const texY0 = 1 - (row + 1) / rows, texY1 = 1 - row / rows;
      
      for (let i = 0; i < uvs.count; i++) {
        if (normals.getZ(i) > 0.9) {
          uvs.setXY(i, positions.getX(i) > 0 ? texX1 : texX0, positions.getY(i) > 0 ? texY1 : texY0);
        }
      }
      
      const hue = hash(idx * 311.7);
      const sideColor = new THREE.Color().setHSL(hue, 0.2, 0.55);
      
      const topMat = new THREE.MeshStandardMaterial({ map: bgTexture, roughness: 0.4 });
      const sideMat = new THREE.MeshStandardMaterial({ color: sideColor, roughness: 0.7 });
      
      const box = new THREE.Mesh(geometry, [sideMat, sideMat, sideMat, sideMat, topMat, sideMat]);
      box.position.set(cx, cy, depth / 2);
      box.castShadow = true;
      box.receiveShadow = true;
      scene.add(box);
    }
  }
  
  renderer.render(scene, camera);
  
  const gl = renderer.getContext();
  const pixels = new Uint8ClampedArray(ctx.width * ctx.height * 4);
  gl.readPixels(0, 0, ctx.width, ctx.height, gl.RGBA, gl.UNSIGNED_BYTE, pixels);
  
  const flipped = new Uint8ClampedArray(ctx.width * ctx.height * 4);
  for (let y = 0; y < ctx.height; y++) {
    for (let x = 0; x < ctx.width; x++) {
      const src = ((ctx.height - 1 - y) * ctx.width + x) * 4;
      const dst = (y * ctx.width + x) * 4;
      flipped[dst] = pixels[src];
      flipped[dst + 1] = pixels[src + 1];
      flipped[dst + 2] = pixels[src + 2];
      flipped[dst + 3] = pixels[src + 3];
    }
  }
  
  bgTexture.dispose();
  scene.traverse((obj) => {
    if (obj instanceof THREE.Mesh) {
      obj.geometry.dispose();
      if (Array.isArray(obj.material)) obj.material.forEach(m => m.dispose());
      else obj.material.dispose();
    }
  });
  renderer.dispose();
  
  return { width: ctx.width, height: ctx.height, data: flipped };
}

function fnU(ctx: FnContext, n: number): Image {
  const prev = getPrevImage(ctx);
  const w = ctx.width;
  const h = ctx.height;
  
  const hueAmount = 90 + (n % 12) * 15;
  const satAmount = 0.5 + (n % 8) * 0.1;
  const lightAmount = 0.06 + (n % 8) * 0.01;
  
  const baseAngle = n * 0.5;
  const angleH = baseAngle;
  const angleS = baseAngle + Math.PI * 2 / 3;
  const angleL = baseAngle + Math.PI * 4 / 3;
  
  const dirHX = Math.cos(angleH);
  const dirHY = Math.sin(angleH);
  const dirSX = Math.cos(angleS);
  const dirSY = Math.sin(angleS);
  const dirLX = Math.cos(angleL);
  const dirLY = Math.sin(angleL);
  
  const cx = w / 2;
  const cy = h / 2;
  const maxDist = Math.sqrt(cx * cx + cy * cy);
  
  const out = createSolidImage(w, h, '#000000');
  
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

function fnV(ctx: FnContext, c: string): Image {
  const prev = getPrevImage(ctx);
  const out = cloneImage(prev);
  const [tr, tg, tb] = hexToRgb(c);
  
  const cx = ctx.width / 2;
  const cy = ctx.height / 2;
  const maxR = Math.sqrt(cx * cx + cy * cy) * 0.7;
  
  for (let y = 0; y < ctx.height; y++) {
    for (let x = 0; x < ctx.width; x++) {
      const dx = x - cx;
      const dy = y - cy;
      const dist = Math.sqrt(dx * dx + dy * dy);
      
      const vignette = Math.max(0, 1 - Math.pow(dist / maxR, 2));
      const darken = vignette;
      const tint = 1 - vignette;
      
      const idx = (y * ctx.width + x) * 4;
      out.data[idx] = Math.round(out.data[idx] * darken + tr * tint * 0.5);
      out.data[idx + 1] = Math.round(out.data[idx + 1] * darken + tg * tint * 0.5);
      out.data[idx + 2] = Math.round(out.data[idx + 2] * darken + tb * tint * 0.5);
    }
  }
  
  return out;
}

function fnW(ctx: FnContext, n: number): Image {
  const prev = getPrevImage(ctx);
  const out = createSolidImage(ctx.width, ctx.height, '#000000');
  
  const maxRotation = (n * 20) * (Math.PI / 180);
  const cx = ctx.width / 2;
  const cy = ctx.height / 2;
  const maxR = Math.sqrt(cx * cx + cy * cy);
  
  for (let y = 0; y < ctx.height; y++) {
    for (let x = 0; x < ctx.width; x++) {
      const dx = x - cx;
      const dy = y - cy;
      const dist = Math.sqrt(dx * dx + dy * dy);
      const normDist = dist / maxR;
      
      const falloff = (1 - normDist) * (1 - normDist);
      const rotation = maxRotation * falloff;
      
      const cos = Math.cos(rotation);
      const sin = Math.sin(rotation);
      const sx = cx + dx * cos - dy * sin;
      const sy = cy + dx * sin + dy * cos;
      
      const [r, g, b] = getPixel(prev, Math.floor(sx), Math.floor(sy));
      setPixel(out, x, y, r, g, b);
    }
  }
  
  return out;
}

function fnX(ctx: FnContext, symbol: string, c: string): Image {
  const prev = getPrevImage(ctx);
  const out = cloneImage(prev);
  const [cr, cg, cb] = hexToRgb(c);
  
  const size = Math.min(ctx.width, ctx.height) * 0.9;
  const tempCanvas = document.createElement('canvas');
  tempCanvas.width = ctx.width;
  tempCanvas.height = ctx.height;
  const tempCtx = tempCanvas.getContext('2d')!;
  
  tempCtx.fillStyle = 'black';
  tempCtx.fillRect(0, 0, ctx.width, ctx.height);
  
  tempCtx.fillStyle = 'white';
  tempCtx.font = `${size}px serif`;
  tempCtx.textAlign = 'center';
  tempCtx.textBaseline = 'alphabetic';
  
  const metrics = tempCtx.measureText(symbol);
  const actualHeight = metrics.actualBoundingBoxAscent + metrics.actualBoundingBoxDescent;
  const yOffset = metrics.actualBoundingBoxAscent - actualHeight / 2;
  
  tempCtx.fillText(symbol, ctx.width / 2, ctx.height / 2 + yOffset);
  
  const maskData = tempCtx.getImageData(0, 0, ctx.width, ctx.height);
  
  let avgR = 0, avgG = 0, avgB = 0;
  for (let i = 0; i < prev.data.length; i += 4) {
    avgR += prev.data[i];
    avgG += prev.data[i + 1];
    avgB += prev.data[i + 2];
  }
  const numPixels = prev.data.length / 4;
  avgR = Math.round(avgR / numPixels);
  avgG = Math.round(avgG / numPixels);
  avgB = Math.round(avgB / numPixels);
  
  for (let y = 0; y < ctx.height; y++) {
    const t = y / ctx.height;
    
    const gr = cr * (1 - t) + avgR * t;
    const gg = cg * (1 - t) + avgG * t;
    const gb = cb * (1 - t) + avgB * t;
    
    for (let x = 0; x < ctx.width; x++) {
      const maskIdx = (y * ctx.width + x) * 4;
      if (maskData.data[maskIdx] > 128) {
        const [pr, pg, pb] = getPixel(prev, x, y);
        const nr = Math.round(gr * 0.9 + pr * 0.1);
        const ng = Math.round(gg * 0.9 + pg * 0.1);
        const nb = Math.round(gb * 0.9 + pb * 0.1);
        setPixel(out, x, y, nr, ng, nb);
      }
    }
  }
  
  return out;
}

function fnY(ctx: FnContext, n: number): Image {
  const prev = getPrevImage(ctx);
  const out = createSolidImage(ctx.width, ctx.height, '#000000');
  
  const sections = Math.max(2, Math.min(n, 36));
  const cx = ctx.width / 2;
  const cy = ctx.height / 2;
  
  for (let y = 0; y < ctx.height; y++) {
    for (let x = 0; x < ctx.width; x++) {
      const dx = x - cx;
      const dy = y - cy;
      let angle = Math.atan2(dy, dx);
      if (angle < 0) angle += Math.PI * 2;
      
      const sectionIdx = Math.floor(angle / (Math.PI * 2) * sections);
      const hueShift = (sectionIdx / sections) * 360;
      
      const [r, g, b] = getPixel(prev, x, y);
      const [h, s, l] = rgbToHsl(r, g, b);
      const [nr, ng, nb] = hslToRgb((h + hueShift) % 360, s, l);
      
      setPixel(out, x, y, nr, ng, nb);
    }
  }
  
  return out;
}

function fnZ(ctx: FnContext, n: number): Image {
  const prev = getPrevImage(ctx);
  const out = createSolidImage(ctx.width, ctx.height, '#000000');
  
  const blurAmount = n * 4;
  const cx = ctx.width / 2;
  const cy = ctx.height / 2;
  const maxR = Math.sqrt(cx * cx + cy * cy);
  const sharpRadius = 0.2;
  
  for (let y = 0; y < ctx.height; y++) {
    for (let x = 0; x < ctx.width; x++) {
      const dx = x - cx;
      const dy = y - cy;
      const dist = Math.sqrt(dx * dx + dy * dy);
      const normDist = dist / maxR;
      
      if (normDist < sharpRadius) {
        const [r, g, b] = getPixel(prev, x, y);
        setPixel(out, x, y, r, g, b);
      } else {
        const blurStrength = Math.min(1, (normDist - sharpRadius) / (1 - sharpRadius));
        const samples = Math.max(1, Math.floor(blurStrength * blurAmount / 2));
        
        let sumR = 0, sumG = 0, sumB = 0;
        for (let i = 0; i <= samples; i++) {
          const t = i / samples;
          const sx = cx + dx * (1 - t * blurStrength * 0.5);
          const sy = cy + dy * (1 - t * blurStrength * 0.5);
          const [r, g, b] = getPixel(prev, Math.floor(sx), Math.floor(sy));
          sumR += r;
          sumG += g;
          sumB += b;
        }
        
        setPixel(out, x, y, 
          Math.round(sumR / (samples + 1)),
          Math.round(sumG / (samples + 1)),
          Math.round(sumB / (samples + 1))
        );
      }
    }
  }
  
  return out;
}

function fnQ(ctx: FnContext): Image {
  const prev = getPrevImage(ctx);
  const out = createSolidImage(ctx.width, ctx.height, '#000000');
  
  const hw = Math.floor(ctx.width / 2);
  const hh = Math.floor(ctx.height / 2);
  
  const temp = createSolidImage(hw, hh, '#000000');
  
  for (let y = 0; y < hh; y++) {
    for (let x = 0; x < hw; x++) {
      const sx = (x / hw) * prev.width;
      const sy = (y / hh) * prev.height;
      const [r, g, b] = getPixel(prev, Math.floor(sx), Math.floor(sy));
      
      if (x / hw + y / hh > 1) {
        setPixel(temp, x, y, 255 - r, 255 - g, 255 - b);
      } else {
        setPixel(temp, x, y, r, g, b);
      }
    }
  }
  
  for (let qy = 0; qy < 2; qy++) {
    for (let qx = 0; qx < 2; qx++) {
      for (let y = 0; y < hh; y++) {
        for (let x = 0; x < hw; x++) {
          const outX = qx * hw + x;
          const outY = qy * hh + y;
          if (outX < ctx.width && outY < ctx.height) {
            const srcX = qx === 1 ? hw - 1 - x : x;
            const srcY = qy === 1 ? hh - 1 - y : y;
            const [r, g, b] = getPixel(temp, srcX, srcY);
            setPixel(out, outX, outY, r, g, b);
          }
        }
      }
    }
  }
  
  return out;
}

function fn0(ctx: FnContext, old: Image): Image {
  const prev = getPrevImage(ctx);
  const out = createSolidImage(ctx.width, ctx.height, '#000000');
  
  for (let y = 0; y < ctx.height; y++) {
    for (let x = 0; x < ctx.width; x++) {
      const [pr, pg, pb] = getPixel(prev, x, y);
      const [or, og, ob] = getPixel(old, x, y);
      
      const nr = pr < 128 ? (2 * pr * or) / 255 : 255 - (2 * (255 - pr) * (255 - or)) / 255;
      const ng = pg < 128 ? (2 * pg * og) / 255 : 255 - (2 * (255 - pg) * (255 - og)) / 255;
      const nb = pb < 128 ? (2 * pb * ob) / 255 : 255 - (2 * (255 - pb) * (255 - ob)) / 255;
      
      setPixel(out, x, y, Math.round(nr), Math.round(ng), Math.round(nb));
    }
  }
  
  return out;
}

function fn1(ctx: FnContext): Image {
  const prev = getPrevImage(ctx);
  const out = createSolidImage(ctx.width, ctx.height, '#000000');
  
  const barStart = Math.floor(ctx.width / 3);
  const barEnd = Math.floor(ctx.width * 2 / 3);
  
  for (let y = 0; y < ctx.height; y++) {
    for (let x = 0; x < ctx.width; x++) {
      const [r, g, b] = getPixel(prev, x, y);
      
      if (x >= barStart && x < barEnd) {
        let sr = 0, sg = 0, sb = 0;
        const kernel = [
          [0, -1, 0],
          [-1, 5, -1],
          [0, -1, 0]
        ];
        for (let ky = -1; ky <= 1; ky++) {
          for (let kx = -1; kx <= 1; kx++) {
            const [pr, pg, pb] = getPixel(prev, x + kx, y + ky);
            const weight = kernel[ky + 1][kx + 1];
            sr += pr * weight;
            sg += pg * weight;
            sb += pb * weight;
          }
        }
        sr = Math.max(0, Math.min(255, sr));
        sg = Math.max(0, Math.min(255, sg));
        sb = Math.max(0, Math.min(255, sb));
        
        const contrast = 1.3;
        sr = Math.max(0, Math.min(255, ((sr / 255 - 0.5) * contrast + 0.5) * 255));
        sg = Math.max(0, Math.min(255, ((sg / 255 - 0.5) * contrast + 0.5) * 255));
        sb = Math.max(0, Math.min(255, ((sb / 255 - 0.5) * contrast + 0.5) * 255));
        
        setPixel(out, x, y, Math.round(sr), Math.round(sg), Math.round(sb));
      } else {
        const gray = Math.round(r * 0.299 + g * 0.587 + b * 0.114);
        setPixel(out, x, y, gray, gray, gray);
      }
    }
  }
  
  return out;
}

function fn2(ctx: FnContext, old: Image, oldThird: number, prevThird: number): Image {
  const prev = getPrevImage(ctx);
  const out = cloneImage(prev);
  
  const oldIdx = ((oldThird - 1) % 3 + 3) % 3;
  const prevIdx = ((prevThird - 1) % 3 + 3) % 3;
  
  const thirdWidth = Math.floor(ctx.width / 3);
  
  const oldStartX = oldIdx * thirdWidth;
  const oldEndX = oldIdx === 2 ? ctx.width : (oldIdx + 1) * thirdWidth;
  
  const prevStartX = prevIdx * thirdWidth;
  const prevEndX = prevIdx === 2 ? ctx.width : (prevIdx + 1) * thirdWidth;
  
  const oldWidth = oldEndX - oldStartX;
  const prevWidth = prevEndX - prevStartX;
  
  for (let y = 0; y < ctx.height; y++) {
    for (let x = prevStartX; x < prevEndX; x++) {
      const srcX = oldStartX + Math.floor((x - prevStartX) / prevWidth * oldWidth);
      const [r, g, b] = getPixel(old, srcX, y);
      setPixel(out, x, y, r, g, b);
    }
  }
  
  return out;
}

function fn3(ctx: FnContext): Image {
  const prev = getPrevImage(ctx);
  const out = createSolidImage(ctx.width, ctx.height, '#000000');
  
  const stripWidth = Math.floor(ctx.width / 3);
  
  for (let y = 0; y < ctx.height; y++) {
    for (let x = 0; x < stripWidth; x++) {
      const srcX = Math.floor((y / ctx.height) * stripWidth);
      const srcY = Math.floor(((stripWidth - 1 - x) / stripWidth) * ctx.height);
      const [r, g, b] = getPixel(prev, srcX, srcY);
      setPixel(out, x, y, r, g, b);
    }
  }
  
  for (let y = 0; y < ctx.height; y++) {
    for (let x = stripWidth; x < stripWidth * 2; x++) {
      const srcX = Math.floor(((x - stripWidth) / stripWidth) * stripWidth + stripWidth);
      const srcY = y;
      const [r, g, b] = getPixel(prev, srcX, srcY);
      setPixel(out, x, y, r, g, b);
    }
  }
  
  for (let y = 0; y < ctx.height; y++) {
    for (let x = stripWidth * 2; x < ctx.width; x++) {
      const localX = x - stripWidth * 2;
      const srcX = Math.floor(((ctx.height - 1 - y) / ctx.height) * (ctx.width - stripWidth * 2) + stripWidth * 2);
      const srcY = Math.floor((localX / (ctx.width - stripWidth * 2)) * ctx.height);
      const [r, g, b] = getPixel(prev, srcX, srcY);
      setPixel(out, x, y, r, g, b);
    }
  }
  
  return out;
}

function fn4(ctx: FnContext): Image {
  const prev = getPrevImage(ctx);
  const out = createSolidImage(ctx.width, ctx.height, '#000000');
  
  const hw = Math.floor(ctx.width / 2);
  const hh = Math.floor(ctx.height / 2);
  
  const rotatePoint = (x: number, y: number, cx: number, cy: number, angle: number): [number, number] => {
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
      const angles = [0, Math.PI / 2, Math.PI, Math.PI * 3 / 2];
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

function fnD(ctx: FnContext): Image {
  const prev = getPrevImage(ctx);
  const gl = initWebGL(ctx.width, ctx.height);
  
  gl.activeTexture(gl.TEXTURE0);
  gl.bindTexture(gl.TEXTURE_2D, null);
  gl.activeTexture(gl.TEXTURE1);
  gl.bindTexture(gl.TEXTURE_2D, null);
  gl.activeTexture(gl.TEXTURE0);
  
  const n = 7;
  const numDrips = Math.max(1, n * 3);
  const dripStrength = Math.max(0.5, Math.min(n * 0.2 + 0.8, 2.0));
  const seed = ctx.images.length * 137.5;
  
  const vertexShader = `
    attribute vec2 position;
    varying vec2 vUV;
    void main() {
      vUV = vec2(position.x * 0.5 + 0.5, 1.0 - (position.y * 0.5 + 0.5));
      gl_Position = vec4(position, 0.0, 1.0);
    }
  `;
  
  const fragmentShader = `
    precision highp float;
    uniform sampler2D uTexture;
    uniform vec2 uResolution;
    uniform float uStrength;
    uniform int uNumDrips;
    uniform float uSeed;
    varying vec2 vUV;
    
    #define MAX_DRIPS 100
    #define MAX_STEPS 64
    #define WALL_Z 0.0
    #define EPS 0.001
    
    float hash(float n) {
      return fract(sin(n) * 43758.5453123);
    }
    
    // Smooth min for organic blending
    float smin(float a, float b, float k) {
      float h = clamp(0.5 + 0.5 * (b - a) / k, 0.0, 1.0);
      return mix(b, a, h) - k * h * (1.0 - h);
    }
    
    // Simple sphere SDF
    float sdSphere(vec3 p, vec3 center, float r) {
      return length(p - center) - r;
    }
    
    // Scene SDF - randomly distributed clusters of 4 metaballs each
    float sceneSDF(vec3 p, int numClusters, float strength) {
      float d = 1000.0;
      
      for (int i = 0; i < MAX_DRIPS; i++) {
        if (i >= numClusters) break;
        
        float fi = float(i);
        
        // Random cluster center position across full screen (seeded by image count)
        float clusterX = (hash(fi * 127.1 + uSeed) * 2.0 - 1.0) * 1.5;
        float clusterY = (hash(fi * 311.7 + uSeed) * 2.0 - 1.0) * 1.2;
        
        // Cluster size
        float clusterSize = 0.15 + hash(fi * 74.3 + uSeed) * 0.15;
        
        // 4 balls per cluster (fixed to avoid dynamic loop issues)
        for (int j = 0; j < 4; j++) {
          float fj = float(j);
          float localSeed = fi * 100.0 + fj + uSeed;
          
          // Random offset from cluster center
          float offX = (hash(localSeed * 123.4) - 0.5) * clusterSize;
          float offY = (hash(localSeed * 234.5) - 0.5) * clusterSize * 1.5; // More vertical spread
          
          float x = clusterX + offX;
          float y = clusterY + offY;
          
          // Varying ball sizes within cluster
          float r = (0.06 + hash(localSeed * 345.6) * 0.08) * strength * 0.6;
          
          float z = 0.05 + r * 0.3;
          
          vec3 center = vec3(x, y, z);
          float sphere = sdSphere(p, center, r);
          
          // Smooth blend to merge balls in cluster
          d = smin(d, sphere, 0.12);
        }
      }
      
      return d;
    }
    
    // Calculate normal via gradient
    vec3 calcNormal(vec3 p, int numDrips, float strength) {
      vec2 e = vec2(0.001, 0.0);
      return normalize(vec3(
        sceneSDF(p + e.xyy, numDrips, strength) - sceneSDF(p - e.xyy, numDrips, strength),
        sceneSDF(p + e.yxy, numDrips, strength) - sceneSDF(p - e.yxy, numDrips, strength),
        sceneSDF(p + e.yyx, numDrips, strength) - sceneSDF(p - e.yyx, numDrips, strength)
      ));
    }
    
    void main() {
      vec2 uv = vUV;
      float aspect = uResolution.x / uResolution.y;
      
      // Ray origin and direction (orthographic-ish camera)
      vec3 ro = vec3((uv.x * 2.0 - 1.0) * aspect, uv.y * 2.0 - 1.0, 2.0);
      vec3 rd = vec3(0.0, 0.0, -1.0);
      
      // Raymarch to find metaball surface
      float t = 0.0;
      float d = 0.0;
      bool hit = false;
      vec3 p;
      
      for (int i = 0; i < MAX_STEPS; i++) {
        p = ro + rd * t;
        d = sceneSDF(p, uNumDrips, uStrength);
        
        if (d < EPS) {
          hit = true;
          break;
        }
        if (t > 4.0) break;
        
        t += d * 0.8;
      }
      
      // Sample wall texture
      vec4 wallColor = texture2D(uTexture, uv);
      
      if (hit) {
        vec3 normal = calcNormal(p, uNumDrips, uStrength);
        
        // Light from top (Y is flipped in screen coords)
        vec3 lightDir = normalize(vec3(0.0, -1.0, 0.6));
        vec3 viewDir = -rd;
        
        // Fresnel - edges reflect more
        float fresnel = pow(1.0 - max(dot(normal, viewDir), 0.0), 3.0);
        
        // Specular highlight - bright and sharp
        vec3 reflectDir = reflect(-lightDir, normal);
        float spec = pow(max(dot(reflectDir, viewDir), 0.0), 120.0);
        
        // Refraction - water has IOR ~1.33, so ratio is 1.0/1.33 ≈ 0.75
        vec3 refracted = refract(rd, normal, 0.75);
        vec2 refractUV = uv + refracted.xy * 0.06;
        refractUV = clamp(refractUV, 0.0, 1.0);
        vec4 refractColor = texture2D(uTexture, refractUV);
        
        // Clear water droplet - mostly shows refracted background
        vec3 dropletColor = refractColor.rgb;
        
        // Slight caustic brightening where light focuses
        float caustic = max(dot(normal, lightDir), 0.0);
        dropletColor *= 1.0 + caustic * 0.15;
        
        // Add crisp specular highlight
        dropletColor += vec3(1.0) * spec * 1.5;
        
        // Subtle rim highlight from fresnel
        dropletColor += vec3(1.0) * fresnel * 0.2;
        
        // Very subtle shadow directly under droplet on wall
        vec2 shadowUV = uv + vec2(0.01, 0.02);
        float shadowMask = smoothstep(0.05, 0.0, d) * 0.15;
        vec3 wallWithShadow = wallColor.rgb * (1.0 - shadowMask);
        
        // Droplet is almost fully transparent, just refracts
        gl_FragColor = vec4(dropletColor, 1.0);
      } else {
        gl_FragColor = wallColor;
      }
    }
  `;
  
  const program = createShaderProgram(gl, vertexShader, fragmentShader);
  gl.useProgram(program);
  
  const texture = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, texture);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, prev.width, prev.height, 0, gl.RGBA, gl.UNSIGNED_BYTE, prev.data);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  
  const vertices = new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]);
  const buffer = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
  gl.bufferData(gl.ARRAY_BUFFER, vertices, gl.STATIC_DRAW);
  
  const positionLoc = gl.getAttribLocation(program, 'position');
  gl.enableVertexAttribArray(positionLoc);
  gl.vertexAttribPointer(positionLoc, 2, gl.FLOAT, false, 0, 0);
  
  gl.uniform1i(gl.getUniformLocation(program, 'uTexture'), 0);
  gl.uniform2f(gl.getUniformLocation(program, 'uResolution'), ctx.width, ctx.height);
  gl.uniform1f(gl.getUniformLocation(program, 'uStrength'), dripStrength);
  gl.uniform1i(gl.getUniformLocation(program, 'uNumDrips'), numDrips);
  gl.uniform1f(gl.getUniformLocation(program, 'uSeed'), seed);
  
  gl.viewport(0, 0, ctx.width, ctx.height);
  gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  
  const pixels = new Uint8ClampedArray(ctx.width * ctx.height * 4);
  gl.readPixels(0, 0, ctx.width, ctx.height, gl.RGBA, gl.UNSIGNED_BYTE, pixels);
  
  const flipped = new Uint8ClampedArray(ctx.width * ctx.height * 4);
  for (let y = 0; y < ctx.height; y++) {
    for (let x = 0; x < ctx.width; x++) {
      const srcIdx = ((ctx.height - 1 - y) * ctx.width + x) * 4;
      const dstIdx = (y * ctx.width + x) * 4;
      flipped[dstIdx] = pixels[srcIdx];
      flipped[dstIdx + 1] = pixels[srcIdx + 1];
      flipped[dstIdx + 2] = pixels[srcIdx + 2];
      flipped[dstIdx + 3] = pixels[srcIdx + 3];
    }
  }
  
  gl.deleteTexture(texture);
  gl.deleteBuffer(buffer);
  gl.deleteProgram(program);
  
  return { width: ctx.width, height: ctx.height, data: flipped };
}

function fn6(ctx: FnContext): Image {
  const prev = getPrevImage(ctx);
  const out = cloneImage(prev);
  
  const levels = 4;
  
  for (let i = 0; i < out.data.length; i += 4) {
    out.data[i] = Math.floor(out.data[i] / 256 * levels) * (255 / (levels - 1));
    out.data[i + 1] = Math.floor(out.data[i + 1] / 256 * levels) * (255 / (levels - 1));
    out.data[i + 2] = Math.floor(out.data[i + 2] / 256 * levels) * (255 / (levels - 1));
  }
  
  return out;
}

function fn7(ctx: FnContext): Image {
  const prev = getPrevImage(ctx);
  const out = createSolidImage(ctx.width, ctx.height, '#000000');
  
  for (let y = 0; y < ctx.height; y++) {
    for (let x = 0; x < ctx.width; x++) {
      const [r] = getPixel(prev, x - 4, y);
      const [, g] = getPixel(prev, x, y);
      const [, , b] = getPixel(prev, x + 4, y);
      setPixel(out, x, y, r, g, b);
    }
  }
  
  return out;
}

function fn8(ctx: FnContext, n: number): Image {
  const prev = getPrevImage(ctx);
  const out = createSolidImage(ctx.width, ctx.height, '#000000');
  
  const strength = Math.max(0.1, n / 5);
  const cx = ctx.width / 2;
  const cy = ctx.height / 2;
  const scale = Math.min(ctx.width, ctx.height) / 4;
  
  for (let y = 0; y < ctx.height; y++) {
    for (let x = 0; x < ctx.width; x++) {
      const nx = (x - cx) / scale;
      const ny = (y - cy) / scale;
      
      const r2 = nx * nx + ny * ny;
      const denom = r2 + 1;
      
      const lemnX = nx * (r2 - 1) / denom;
      const lemnY = ny * (r2 + 1) / denom;
      
      const sx = cx + (nx + (lemnX - nx) * strength * 0.3) * scale;
      const sy = cy + (ny + (lemnY - ny) * strength * 0.3) * scale;
      
      const [r, g, b] = getPixel(prev, Math.floor(sx), Math.floor(sy));
      setPixel(out, x, y, r, g, b);
    }
  }
  
  return out;
}

function fn9(ctx: FnContext, old: Image): Image {
  const prev = getPrevImage(ctx);
  const out = createSolidImage(ctx.width, ctx.height, '#000000');
  
  for (let y = 0; y < ctx.height; y++) {
    for (let x = 0; x < ctx.width; x++) {
      const [r1, g1, b1] = getPixel(prev, x, y);
      const [r2, g2, b2] = getPixel(old, x, y);
      
      setPixel(out, x, y, r1 ^ r2, g1 ^ g2, b1 ^ b2);
    }
  }
  
  return out;
}

function fnLessThan(ctx: FnContext): Image {
  const prev = getPrevImage(ctx);
  const out = createSolidImage(ctx.width, ctx.height, '#000000');
  const shift = Math.floor(ctx.width / 8);
  
  for (let y = 0; y < ctx.height; y++) {
    for (let x = 0; x < ctx.width; x++) {
      const srcX = (x + shift) % ctx.width;
      const [r, g, b] = getPixel(prev, srcX, y);
      setPixel(out, x, y, r, g, b);
    }
  }
  
  return out;
}

function fnGreaterThan(ctx: FnContext): Image {
  const prev = getPrevImage(ctx);
  const out = createSolidImage(ctx.width, ctx.height, '#000000');
  
  for (let y = 0; y < ctx.height; y++) {
    for (let x = 0; x < ctx.width; x++) {
      const sx = y * prev.width / ctx.height;
      const sy = (ctx.width - 1 - x) * prev.height / ctx.width;
      const [r, g, b] = getPixel(prev, Math.floor(sx), Math.floor(sy));
      setPixel(out, x, y, r, g, b);
    }
  }
  
  return out;
}

function fnCaret(ctx: FnContext): Image {
  const prev = getPrevImage(ctx);
  const out = createSolidImage(ctx.width, ctx.height, '#000000');
  const shift = Math.floor(ctx.height / 3);
  
  for (let y = 0; y < ctx.height; y++) {
    for (let x = 0; x < ctx.width; x++) {
      const srcY = (y + shift) % ctx.height;
      const [r, g, b] = getPixel(prev, x, srcY);
      setPixel(out, x, y, r, g, b);
    }
  }
  
  return out;
}

function fnExclaim(ctx: FnContext): Image {
  const prev = getPrevImage(ctx);
  const out = cloneImage(prev);
  
  const hash = (x: number, y: number, seed: number) => {
    const v = Math.sin(x * 127.1 + y * 311.7 + seed * 113.3) * 43758.5453;
    return v - Math.floor(v);
  };
  
  const n = 60;
  const baseLen = 8 + n * 5;
  const chaos = 0.5 + n * 0.15;
  const cx = ctx.width / 2;
  const cy = ctx.height / 2;
  
  for (let y = 0; y < ctx.height; y++) {
    for (let x = 0; x < ctx.width; x++) {
      const [r, g, b] = getPixel(prev, x, y);
      const [h, s, l] = rgbToHsl(r, g, b);
      
      if (s < 0.03 && l < 0.03) continue;
      if (l > 0.97) continue;
      
      const hueAngle = (h / 360) * Math.PI * 2;
      
      const turbulence = (hash(x, y, 1) - 0.5) * Math.PI * chaos;
      const swirl = Math.sin(x * 0.02) * Math.cos(y * 0.02) * 0.5;
      
      const angle = hueAngle + turbulence + swirl;
      
      const lenNoise = 0.3 + hash(x, y, 2) * 0.7;
      const len = baseLen * s * (0.2 + l * 0.8) * lenNoise;
      
      if (len < 2) continue;
      
      const dx = Math.cos(angle);
      const dy = Math.sin(angle);
      
      for (let i = 1; i <= len; i++) {
        const wobble = Math.sin(i * 0.5) * hash(x, y, 3) * 2;
        const px = Math.floor(x + dx * i + dy * wobble);
        const py = Math.floor(y + dy * i - dx * wobble);
        
        if (px >= 0 && px < ctx.width && py >= 0 && py < ctx.height) {
          const fade = 1 - (i / len);
          const brightness = fade * fade;
          const idx = (py * ctx.width + px) * 4;
          
          out.data[idx] = Math.min(255, Math.floor(out.data[idx] * (1 - brightness * 0.8) + r * brightness * 0.8));
          out.data[idx + 1] = Math.min(255, Math.floor(out.data[idx + 1] * (1 - brightness * 0.8) + g * brightness * 0.8));
          out.data[idx + 2] = Math.min(255, Math.floor(out.data[idx + 2] * (1 - brightness * 0.8) + b * brightness * 0.8));
        }
      }
    }
  }
  
  return out;
}

function fnDoubleQuote(ctx: FnContext, n: number): Image {
  const prev = getPrevImage(ctx);
  const out = createSolidImage(ctx.width, ctx.height, '#000000');
  const bands = Math.max(2, n);
  const bandHeight = ctx.height / bands;
  
  for (let y = 0; y < ctx.height; y++) {
    const bandIdx = Math.floor(y / bandHeight);
    const isOdd = bandIdx % 2 === 1;
    
    for (let x = 0; x < ctx.width; x++) {
      const [r, g, b] = getPixel(prev, x, y);
      
      if (isOdd) {
        const [h, s, l] = rgbToHsl(r, g, b);
        const [nr, ng, nb] = hslToRgb((h + 180) % 360, s, l);
        setPixel(out, x, y, nr, ng, nb);
      } else {
        const [h, s, l] = rgbToHsl(r, g, b);
        const [nr, ng, nb] = hslToRgb(h, 1 - s, l);
        setPixel(out, x, y, nr, ng, nb);
      }
    }
  }
  
  return out;
}

function fnHash(ctx: FnContext, n: number): Image {
  const prev = getPrevImage(ctx);
  const out = createSolidImage(ctx.width, ctx.height, '#000000');
  
  const gridSize = Math.max(2, n + 2);
  
  const getCellBounds = (gx: number, gy: number): [number, number, number, number] => {
    const x0 = Math.floor(gx * ctx.width / gridSize);
    const x1 = Math.floor((gx + 1) * ctx.width / gridSize);
    const y0 = Math.floor(gy * ctx.height / gridSize);
    const y1 = Math.floor((gy + 1) * ctx.height / gridSize);
    return [x0, y0, x1, y1];
  };
  
  const cells: { gx: number; gy: number; hue: number }[] = [];
  
  for (let gy = 0; gy < gridSize; gy++) {
    for (let gx = 0; gx < gridSize; gx++) {
      let sumR = 0, sumG = 0, sumB = 0, count = 0;
      const [x0, y0, x1, y1] = getCellBounds(gx, gy);
      
      for (let y = y0; y < y1; y++) {
        for (let x = x0; x < x1; x++) {
          const [r, g, b] = getPixel(prev, x, y);
          sumR += r;
          sumG += g;
          sumB += b;
          count++;
        }
      }
      
      const avgR = count > 0 ? sumR / count : 0;
      const avgG = count > 0 ? sumG / count : 0;
      const avgB = count > 0 ? sumB / count : 0;
      const [hue] = rgbToHsl(avgR, avgG, avgB);
      
      cells.push({ gx, gy, hue });
    }
  }
  
  const sortedIndices = cells.map((_, i) => i).sort((a, b) => cells[a].hue - cells[b].hue);
  
  for (let i = 0; i < sortedIndices.length; i++) {
    const srcIdx = sortedIndices[i];
    const srcCell = cells[srcIdx];
    const [sx0, sy0, sx1, sy1] = getCellBounds(srcCell.gx, srcCell.gy);
    const srcW = sx1 - sx0;
    const srcH = sy1 - sy0;
    
    const targetGX = i % gridSize;
    const targetGY = Math.floor(i / gridSize);
    const [tx0, ty0, tx1, ty1] = getCellBounds(targetGX, targetGY);
    const targetW = tx1 - tx0;
    const targetH = ty1 - ty0;
    
    for (let ty = ty0; ty < ty1; ty++) {
      for (let tx = tx0; tx < tx1; tx++) {
        const srcX = sx0 + Math.floor((tx - tx0) * srcW / targetW);
        const srcY = sy0 + Math.floor((ty - ty0) * srcH / targetH);
        const [r, g, b] = getPixel(prev, srcX, srcY);
        setPixel(out, tx, ty, r, g, b);
      }
    }
  }
  
  return out;
}

function fnDollar(ctx: FnContext): Image {
  const prev = getPrevImage(ctx);
  const out = createSolidImage(ctx.width, ctx.height, '#000000');
  const w = ctx.width;
  const h = ctx.height;
  
  const threshold = 12;
  
  const labels = new Int32Array(w * h);
  labels.fill(-1);
  
  const colorDist = (x1: number, y1: number, x2: number, y2: number): number => {
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
          [cx - 1, cy], [cx + 1, cy], [cx, cy - 1], [cx, cy + 1]
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
  
  const segments = new Map<number, { x: number; y: number; r: number; g: number; b: number; hue: number }[]>();
  
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
    
    const positions = pixels.map(p => ({ x: p.x, y: p.y }));
    positions.sort((a, b) => a.x !== b.x ? a.x - b.x : a.y - b.y);
    
    for (let i = 0; i < pixels.length; i++) {
      const color = pixels[i];
      const pos = positions[i];
      setPixel(out, pos.x, pos.y, color.r, color.g, color.b);
    }
  }
  
  return out;
}

function fnPercent(ctx: FnContext, n: number): Image {
  const prev = getPrevImage(ctx);
  const out = createSolidImage(ctx.width, ctx.height, '#000000');
  
  const horizontal = n % 2 === 0;
  
  for (let y = 0; y < ctx.height; y++) {
    for (let x = 0; x < ctx.width; x++) {
      const srcX = horizontal ? ctx.width - 1 - x : x;
      const srcY = horizontal ? y : ctx.height - 1 - y;
      const [r, g, b] = getPixel(prev, srcX, srcY);
      setPixel(out, x, y, r, g, b);
    }
  }
  
  return out;
}

function fnAmpersand(ctx: FnContext, mode: string): Image {
  const prev = getPrevImage(ctx);
  const out = createSolidImage(ctx.width, ctx.height, '#000000');
  
  const modes = [
    'ordered-5level','bayer-bw','threshold-bw','ordered-2bit','floyd-rgb','floyd-bw',
    'atkinson-4level','atkinson-bw','stucki-6level','burkes','sierra','random-bw',
    'cluster-2bit','bluenoise-bw','bayer2x2-2bit','noise-2bit'
  ];
  const ditherMode = modes.indexOf(mode);
  const seed = ctx.images.length * 137.5;
  
  const hash = (x: number, y: number) => {
    const v = Math.sin(x * 127.1 + y * 311.7 + seed * 113.3) * 43758.5453;
    return v - Math.floor(v);
  };
  
  const bayer8x8 = [
    [ 0, 32,  8, 40,  2, 34, 10, 42],
    [48, 16, 56, 24, 50, 18, 58, 26],
    [12, 44,  4, 36, 14, 46,  6, 38],
    [60, 28, 52, 20, 62, 30, 54, 22],
    [ 3, 35, 11, 43,  1, 33,  9, 41],
    [51, 19, 59, 27, 49, 17, 57, 25],
    [15, 47,  7, 39, 13, 45,  5, 37],
    [63, 31, 55, 23, 61, 29, 53, 21]
  ];
  
  const bayer2x2 = [
    [0, 2],
    [3, 1]
  ];
  
  if (ditherMode === 0) {
    const levels = 5;
    const spread = 85;
    
    for (let y = 0; y < ctx.height; y++) {
      for (let x = 0; x < ctx.width; x++) {
        const [r, g, b] = getPixel(prev, x, y);
        
        const tr = (bayer8x8[y % 8][x % 8] / 64.0 - 0.5) * spread;
        const tg = (bayer8x8[(y + 4) % 8][(x + 4) % 8] / 64.0 - 0.5) * spread;
        const tb = (bayer8x8[(y + 2) % 8][(x + 2) % 8] / 64.0 - 0.5) * spread;
        
        const quantize = (v: number, threshold: number) => {
          const adjusted = v + threshold;
          const step = 255 / (levels - 1);
          return Math.max(0, Math.min(255, Math.round(adjusted / step) * step));
        };
        
        setPixel(out, x, y, quantize(r, tr), quantize(g, tg), quantize(b, tb));
      }
    }
  } else if (ditherMode === 1) {
    for (let y = 0; y < ctx.height; y++) {
      for (let x = 0; x < ctx.width; x++) {
        const [r, g, b] = getPixel(prev, x, y);
        const gray = r * 0.299 + g * 0.587 + b * 0.114;
        const threshold = (bayer8x8[y % 8][x % 8] / 64.0) * 255;
        const val = gray > threshold ? 255 : 0;
        setPixel(out, x, y, val, val, val);
      }
    }
  } else if (ditherMode === 2) {
    for (let y = 0; y < ctx.height; y++) {
      for (let x = 0; x < ctx.width; x++) {
        const [r, g, b] = getPixel(prev, x, y);
        const gray = r * 0.299 + g * 0.587 + b * 0.114;
        const val = gray > 128 ? 255 : 0;
        setPixel(out, x, y, val, val, val);
      }
    }
  } else if (ditherMode === 3) {
    for (let y = 0; y < ctx.height; y++) {
      for (let x = 0; x < ctx.width; x++) {
        const [r, g, b] = getPixel(prev, x, y);
        const threshold = (bayer2x2[y % 2][x % 2] / 4.0) * 255;
        
        const qr = r > threshold ? 255 : 0;
        const qg = g > threshold ? 255 : 0;
        const qb = b > threshold ? 255 : 0;
        
        setPixel(out, x, y, qr, qg, qb);
      }
    }
  } else if (ditherMode === 4) {
    const errorR = new Float32Array(ctx.width * ctx.height);
    const errorG = new Float32Array(ctx.width * ctx.height);
    const errorB = new Float32Array(ctx.width * ctx.height);
    
    for (let y = 0; y < ctx.height; y++) {
      for (let x = 0; x < ctx.width; x++) {
        const idx = y * ctx.width + x;
        const [r, g, b] = getPixel(prev, x, y);
        
        const nr = Math.max(0, Math.min(255, r + errorR[idx]));
        const ng = Math.max(0, Math.min(255, g + errorG[idx]));
        const nb = Math.max(0, Math.min(255, b + errorB[idx]));
        
        const qr = nr > 128 ? 255 : 0;
        const qg = ng > 128 ? 255 : 0;
        const qb = nb > 128 ? 255 : 0;
        
        setPixel(out, x, y, qr, qg, qb);
        
        const er = nr - qr;
        const eg = ng - qg;
        const eb = nb - qb;
        
        if (x + 1 < ctx.width) {
          errorR[idx + 1] += er * 7/16;
          errorG[idx + 1] += eg * 7/16;
          errorB[idx + 1] += eb * 7/16;
        }
        if (y + 1 < ctx.height) {
          if (x > 0) {
            errorR[idx + ctx.width - 1] += er * 3/16;
            errorG[idx + ctx.width - 1] += eg * 3/16;
            errorB[idx + ctx.width - 1] += eb * 3/16;
          }
          errorR[idx + ctx.width] += er * 5/16;
          errorG[idx + ctx.width] += eg * 5/16;
          errorB[idx + ctx.width] += eb * 5/16;
          if (x + 1 < ctx.width) {
            errorR[idx + ctx.width + 1] += er * 1/16;
            errorG[idx + ctx.width + 1] += eg * 1/16;
            errorB[idx + ctx.width + 1] += eb * 1/16;
          }
        }
      }
    }
  } else if (ditherMode === 5) {
    const error = new Float32Array(ctx.width * ctx.height);
    
    for (let y = 0; y < ctx.height; y++) {
      for (let x = 0; x < ctx.width; x++) {
        const idx = y * ctx.width + x;
        const [r, g, b] = getPixel(prev, x, y);
        const gray = r * 0.299 + g * 0.587 + b * 0.114;
        
        const ng = Math.max(0, Math.min(255, gray + error[idx]));
        const qg = ng > 128 ? 255 : 0;
        
        setPixel(out, x, y, qg, qg, qg);
        
        const e = ng - qg;
        
        if (x + 1 < ctx.width) {
          error[idx + 1] += e * 7/16;
        }
        if (y + 1 < ctx.height) {
          if (x > 0) {
            error[idx + ctx.width - 1] += e * 3/16;
          }
          error[idx + ctx.width] += e * 5/16;
          if (x + 1 < ctx.width) {
            error[idx + ctx.width + 1] += e * 1/16;
          }
        }
      }
    }
  } else if (ditherMode === 6) {
    const errorR = new Float32Array(ctx.width * ctx.height);
    const errorG = new Float32Array(ctx.width * ctx.height);
    const errorB = new Float32Array(ctx.width * ctx.height);
    
    for (let y = 0; y < ctx.height; y++) {
      for (let x = 0; x < ctx.width; x++) {
        const idx = y * ctx.width + x;
        const [r, g, b] = getPixel(prev, x, y);
        
        const nr = Math.max(0, Math.min(255, r + errorR[idx]));
        const ng = Math.max(0, Math.min(255, g + errorG[idx]));
        const nb = Math.max(0, Math.min(255, b + errorB[idx]));
        
        const qr = Math.round(nr / 85) * 85;
        const qg = Math.round(ng / 85) * 85;
        const qb = Math.round(nb / 85) * 85;
        
        setPixel(out, x, y, qr, qg, qb);
        
        const er = nr - qr;
        const eg = ng - qg;
        const eb = nb - qb;
        
        if (x + 1 < ctx.width) {
          errorR[idx + 1] += er * 1/8;
          errorG[idx + 1] += eg * 1/8;
          errorB[idx + 1] += eb * 1/8;
        }
        if (x + 2 < ctx.width) {
          errorR[idx + 2] += er * 1/8;
          errorG[idx + 2] += eg * 1/8;
          errorB[idx + 2] += eb * 1/8;
        }
        if (y + 1 < ctx.height) {
          if (x > 0) {
            errorR[idx + ctx.width - 1] += er * 1/8;
            errorG[idx + ctx.width - 1] += eg * 1/8;
            errorB[idx + ctx.width - 1] += eb * 1/8;
          }
          errorR[idx + ctx.width] += er * 1/8;
          errorG[idx + ctx.width] += eg * 1/8;
          errorB[idx + ctx.width] += eb * 1/8;
          if (x + 1 < ctx.width) {
            errorR[idx + ctx.width + 1] += er * 1/8;
            errorG[idx + ctx.width + 1] += eg * 1/8;
            errorB[idx + ctx.width + 1] += eb * 1/8;
          }
        }
        if (y + 2 < ctx.height) {
          errorR[idx + ctx.width * 2] += er * 1/8;
          errorG[idx + ctx.width * 2] += eg * 1/8;
          errorB[idx + ctx.width * 2] += eb * 1/8;
        }
      }
    }
  } else if (ditherMode === 7) {
    const error = new Float32Array(ctx.width * ctx.height);
    
    for (let y = 0; y < ctx.height; y++) {
      for (let x = 0; x < ctx.width; x++) {
        const idx = y * ctx.width + x;
        const [r, g, b] = getPixel(prev, x, y);
        const gray = r * 0.299 + g * 0.587 + b * 0.114;
        
        const ng = Math.max(0, Math.min(255, gray + error[idx]));
        const qg = ng > 128 ? 255 : 0;
        
        setPixel(out, x, y, qg, qg, qg);
        
        const e = ng - qg;
        
        if (x + 1 < ctx.width) {
          error[idx + 1] += e * 1/8;
        }
        if (x + 2 < ctx.width) {
          error[idx + 2] += e * 1/8;
        }
        if (y + 1 < ctx.height) {
          if (x > 0) {
            error[idx + ctx.width - 1] += e * 1/8;
          }
          error[idx + ctx.width] += e * 1/8;
          if (x + 1 < ctx.width) {
            error[idx + ctx.width + 1] += e * 1/8;
          }
        }
        if (y + 2 < ctx.height) {
          error[idx + ctx.width * 2] += e * 1/8;
        }
      }
    }
  } else if (ditherMode === 8) {
    const errorR = new Float32Array(ctx.width * ctx.height);
    const errorG = new Float32Array(ctx.width * ctx.height);
    const errorB = new Float32Array(ctx.width * ctx.height);
    
    for (let y = 0; y < ctx.height; y++) {
      for (let x = 0; x < ctx.width; x++) {
        const idx = y * ctx.width + x;
        const [r, g, b] = getPixel(prev, x, y);
        
        const nr = Math.max(0, Math.min(255, r + errorR[idx]));
        const ng = Math.max(0, Math.min(255, g + errorG[idx]));
        const nb = Math.max(0, Math.min(255, b + errorB[idx]));
        
        const qr = Math.round(nr / 51) * 51;
        const qg = Math.round(ng / 51) * 51;
        const qb = Math.round(nb / 51) * 51;
        
        setPixel(out, x, y, qr, qg, qb);
        
        const er = nr - qr;
        const eg = ng - qg;
        const eb = nb - qb;
        
        const distribute = (dx: number, dy: number, factor: number) => {
          if (x + dx >= 0 && x + dx < ctx.width && y + dy >= 0 && y + dy < ctx.height) {
            const tidx = idx + dy * ctx.width + dx;
            errorR[tidx] += er * factor;
            errorG[tidx] += eg * factor;
            errorB[tidx] += eb * factor;
          }
        };
        
        distribute(1, 0, 8/42);
        distribute(2, 0, 4/42);
        distribute(-2, 1, 2/42);
        distribute(-1, 1, 4/42);
        distribute(0, 1, 8/42);
        distribute(1, 1, 4/42);
        distribute(2, 1, 2/42);
        distribute(-2, 2, 1/42);
        distribute(-1, 2, 2/42);
        distribute(0, 2, 4/42);
        distribute(1, 2, 2/42);
        distribute(2, 2, 1/42);
      }
    }
  } else if (ditherMode === 9) {
    const errorR = new Float32Array(ctx.width * ctx.height);
    const errorG = new Float32Array(ctx.width * ctx.height);
    const errorB = new Float32Array(ctx.width * ctx.height);
    
    for (let y = 0; y < ctx.height; y++) {
      for (let x = 0; x < ctx.width; x++) {
        const idx = y * ctx.width + x;
        const [r, g, b] = getPixel(prev, x, y);
        
        const nr = Math.max(0, Math.min(255, r + errorR[idx]));
        const ng = Math.max(0, Math.min(255, g + errorG[idx]));
        const nb = Math.max(0, Math.min(255, b + errorB[idx]));
        
        const qr = Math.round(nr / 51) * 51;
        const qg = Math.round(ng / 51) * 51;
        const qb = Math.round(nb / 51) * 51;
        
        setPixel(out, x, y, qr, qg, qb);
        
        const er = nr - qr;
        const eg = ng - qg;
        const eb = nb - qb;
        
        const distribute = (dx: number, dy: number, factor: number) => {
          if (x + dx >= 0 && x + dx < ctx.width && y + dy >= 0 && y + dy < ctx.height) {
            const tidx = idx + dy * ctx.width + dx;
            errorR[tidx] += er * factor;
            errorG[tidx] += eg * factor;
            errorB[tidx] += eb * factor;
          }
        };
        
        distribute(1, 0, 8/32);
        distribute(2, 0, 4/32);
        distribute(-2, 1, 2/32);
        distribute(-1, 1, 4/32);
        distribute(0, 1, 8/32);
        distribute(1, 1, 4/32);
        distribute(2, 1, 2/32);
      }
    }
  } else if (ditherMode === 10) {
    const errorR = new Float32Array(ctx.width * ctx.height);
    const errorG = new Float32Array(ctx.width * ctx.height);
    const errorB = new Float32Array(ctx.width * ctx.height);
    
    for (let y = 0; y < ctx.height; y++) {
      for (let x = 0; x < ctx.width; x++) {
        const idx = y * ctx.width + x;
        const [r, g, b] = getPixel(prev, x, y);
        
        const nr = Math.max(0, Math.min(255, r + errorR[idx]));
        const ng = Math.max(0, Math.min(255, g + errorG[idx]));
        const nb = Math.max(0, Math.min(255, b + errorB[idx]));
        
        const qr = Math.round(nr / 51) * 51;
        const qg = Math.round(ng / 51) * 51;
        const qb = Math.round(nb / 51) * 51;
        
        setPixel(out, x, y, qr, qg, qb);
        
        const er = nr - qr;
        const eg = ng - qg;
        const eb = nb - qb;
        
        const distribute = (dx: number, dy: number, factor: number) => {
          if (x + dx >= 0 && x + dx < ctx.width && y + dy >= 0 && y + dy < ctx.height) {
            const tidx = idx + dy * ctx.width + dx;
            errorR[tidx] += er * factor;
            errorG[tidx] += eg * factor;
            errorB[tidx] += eb * factor;
          }
        };
        
        distribute(1, 0, 5/32);
        distribute(2, 0, 3/32);
        distribute(-2, 1, 2/32);
        distribute(-1, 1, 4/32);
        distribute(0, 1, 5/32);
        distribute(1, 1, 4/32);
        distribute(2, 1, 2/32);
        distribute(-1, 2, 2/32);
        distribute(0, 2, 3/32);
        distribute(1, 2, 2/32);
      }
    }
  } else if (ditherMode === 11) {
    for (let y = 0; y < ctx.height; y++) {
      for (let x = 0; x < ctx.width; x++) {
        const [r, g, b] = getPixel(prev, x, y);
        const gray = r * 0.299 + g * 0.587 + b * 0.114;
        const noise = (hash(x, y) - 0.5) * 100;
        const val = gray + noise > 128 ? 255 : 0;
        setPixel(out, x, y, val, val, val);
      }
    }
  } else if (ditherMode === 12) {
    const cluster = [
      [12, 5, 6, 13],
      [4, 0, 1, 7],
      [11, 3, 2, 8],
      [15, 10, 9, 14]
    ];
    
    for (let y = 0; y < ctx.height; y++) {
      for (let x = 0; x < ctx.width; x++) {
        const [r, g, b] = getPixel(prev, x, y);
        const threshold = (cluster[y % 4][x % 4] / 16.0) * 255;
        
        const qr = r > threshold ? 255 : 0;
        const qg = g > threshold ? 255 : 0;
        const qb = b > threshold ? 255 : 0;
        
        setPixel(out, x, y, qr, qg, qb);
      }
    }
  } else if (ditherMode === 13) {
    const blueNoise = (x: number, y: number): number => {
      const v1 = hash(x * 0.7 + seed, y * 0.7 + seed);
      const v2 = hash(x * 1.3 + seed + 100, y * 1.3 + seed + 100);
      const v3 = hash(x * 2.1 + seed + 200, y * 2.1 + seed + 200);
      return (v1 + v2 + v3) / 3;
    };
    
    for (let y = 0; y < ctx.height; y++) {
      for (let x = 0; x < ctx.width; x++) {
        const [r, g, b] = getPixel(prev, x, y);
        const gray = r * 0.299 + g * 0.587 + b * 0.114;
        const threshold = blueNoise(x, y) * 255;
        const val = gray > threshold ? 255 : 0;
        setPixel(out, x, y, val, val, val);
      }
    }
  } else if (ditherMode === 14) {
    for (let y = 0; y < ctx.height; y++) {
      for (let x = 0; x < ctx.width; x++) {
        const [r, g, b] = getPixel(prev, x, y);
        const threshold = (bayer2x2[y % 2][x % 2] / 4.0) * 255;
        
        const qr = r > threshold ? 255 : 0;
        const qg = g > threshold ? 255 : 0;
        const qb = b > threshold ? 255 : 0;
        
        setPixel(out, x, y, qr, qg, qb);
      }
    }
  } else if (ditherMode === 15) {
    for (let y = 0; y < ctx.height; y++) {
      for (let x = 0; x < ctx.width; x++) {
        const [r, g, b] = getPixel(prev, x, y);
        const noise = (hash(x, y) - 0.5) * 180;
        
        const qr = r + noise > 128 ? 255 : 0;
        const qg = g + noise > 128 ? 255 : 0;
        const qb = b + noise > 128 ? 255 : 0;
        
        setPixel(out, x, y, qr, qg, qb);
      }
    }
  }
  
  return out;
}

function fnApostrophe(ctx: FnContext, n: number): Image {
  const prev = getPrevImage(ctx);
  const out = cloneImage(prev);
  const spacing = n + 2;
  
  for (let x = 0; x < ctx.width; x++) {
    if (x % spacing !== 0) continue;
    
    let colLuminance = 0;
    for (let y = 0; y < ctx.height; y++) {
      const [r, g, b] = getPixel(prev, x, y);
      colLuminance += r * 0.299 + g * 0.587 + b * 0.114;
    }
    colLuminance /= ctx.height;
    
    const streakLength = Math.floor((colLuminance / 255) * 100);
    const startY = 0;
    
    for (let y = startY; y < startY + streakLength && y < ctx.height; y++) {
      const idx = (y * ctx.width + x) * 4;
      out.data[idx] = Math.min(255, Math.round(out.data[idx] * 0.5 + 127));
      out.data[idx + 1] = Math.min(255, Math.round(out.data[idx + 1] * 0.5 + 127));
      out.data[idx + 2] = Math.min(255, Math.round(out.data[idx + 2] * 0.5 + 127));
    }
  }
  
  return out;
}

function fnOpenParen(ctx: FnContext, n: number): Image {
  const prev = getPrevImage(ctx);
  const out = createSolidImage(ctx.width, ctx.height, '#000000');
  
  const strength = n / 10;
  const cx = ctx.width / 2;
  const cy = ctx.height / 2;
  const maxR = Math.sqrt(cx * cx + cy * cy);
  
  for (let y = 0; y < ctx.height; y++) {
    for (let x = 0; x < ctx.width; x++) {
      const dx = x - cx;
      const dy = y - cy;
      const dist = Math.sqrt(dx * dx + dy * dy);
      const normDist = dist / maxR;
      
      const pinchFactor = 1 + strength * (1 - normDist);
      const sx = cx + dx * pinchFactor;
      const sy = cy + dy * pinchFactor;
      
      let [r, g, b] = getPixel(prev, Math.floor(sx), Math.floor(sy));
      
      const brighten = (1 - normDist) * strength * 30;
      r = Math.min(255, r + brighten);
      g = Math.min(255, g + brighten);
      b = Math.min(255, b + brighten);
      
      setPixel(out, x, y, Math.round(r), Math.round(g), Math.round(b));
    }
  }
  
  return out;
}

function fnCloseParen(ctx: FnContext, n: number): Image {
  const prev = getPrevImage(ctx);
  const out = createSolidImage(ctx.width, ctx.height, '#000000');
  
  const strength = n / 10;
  const cx = ctx.width / 2;
  const cy = ctx.height / 2;
  const maxR = Math.sqrt(cx * cx + cy * cy);
  
  for (let y = 0; y < ctx.height; y++) {
    for (let x = 0; x < ctx.width; x++) {
      const dx = x - cx;
      const dy = y - cy;
      const dist = Math.sqrt(dx * dx + dy * dy);
      const normDist = dist / maxR;
      
      const bulgeFactor = 1 - strength * normDist * 0.5;
      const sx = cx + dx * bulgeFactor;
      const sy = cy + dy * bulgeFactor;
      
      let [r, g, b] = getPixel(prev, Math.floor(sx), Math.floor(sy));
      
      const darken = normDist * strength * 50;
      r = Math.max(0, r - darken);
      g = Math.max(0, g - darken);
      b = Math.max(0, b - darken);
      
      setPixel(out, x, y, Math.round(r), Math.round(g), Math.round(b));
    }
  }
  
  return out;
}

function fnAsterisk(ctx: FnContext): Image {
  const prev = getPrevImage(ctx);
  const gl = initWebGL(ctx.width, ctx.height);
  
  gl.activeTexture(gl.TEXTURE0);
  gl.bindTexture(gl.TEXTURE_2D, null);
  gl.activeTexture(gl.TEXTURE1);
  gl.bindTexture(gl.TEXTURE_2D, null);
  gl.activeTexture(gl.TEXTURE0);
  
  const vertexShader = `
    attribute vec2 position;
    varying vec2 vUV;
    void main() {
      vUV = vec2(position.x * 0.5 + 0.5, 1.0 - (position.y * 0.5 + 0.5));
      gl_Position = vec4(position, 0.0, 1.0);
    }
  `;
  
  const occlusionFragShader = `
    precision highp float;
    uniform sampler2D uTexture;
    uniform vec2 uLightPos;
    uniform float uLightRadius;
    varying vec2 vUV;
    
    void main() {
      vec3 texColor = texture2D(uTexture, vUV).rgb;
      float brightness = dot(texColor, vec3(0.299, 0.587, 0.114));
      
      // Distance from light center
      vec2 diff = vUV - uLightPos;
      float dist = length(diff);
      
      // Light source glow (bright center)
      float lightGlow = exp(-dist * dist / (uLightRadius * uLightRadius * 2.0));
      
      // Threshold bright areas as potential light sources
      float threshold = step(0.5, brightness);
      
      // Combine: light center + bright pixels from image
      vec3 lightColor = vec3(1.0, 0.95, 0.8);
      vec3 result = lightGlow * lightColor + threshold * texColor * 0.8;
      
      gl_FragColor = vec4(result, 1.0);
    }
  `;
  
  const godrayFragShader = `
    precision highp float;
    uniform sampler2D uOcclusionTexture;
    uniform sampler2D uSceneTexture;
    uniform vec2 uLightPos;
    uniform float uExposure;
    uniform float uDecay;
    uniform float uDensity;
    uniform float uWeight;
    varying vec2 vUV;
    
    #define NUM_SAMPLES 80
    
    void main() {
      vec2 texCoord = vUV;
      vec2 deltaTexCoord = (texCoord - uLightPos);
      deltaTexCoord *= 1.0 / float(NUM_SAMPLES) * uDensity;
      
      float illuminationDecay = 1.0;
      vec3 godrayColor = vec3(0.0);
      
      vec2 sampleCoord = texCoord;
      
      for (int i = 0; i < NUM_SAMPLES; i++) {
        sampleCoord -= deltaTexCoord;
        vec3 sampleColor = texture2D(uOcclusionTexture, sampleCoord).rgb;
        sampleColor *= illuminationDecay * uWeight;
        godrayColor += sampleColor;
        illuminationDecay *= uDecay;
      }
      
      godrayColor *= uExposure;
      
      // Get original scene
      vec3 sceneColor = texture2D(uSceneTexture, vUV).rgb;
      
      // Blend godrays additively with scene
      vec3 finalColor = sceneColor + godrayColor;
      
      gl_FragColor = vec4(finalColor, 1.0);
    }
  `;
  
  const occlusionProgram = createShaderProgram(gl, vertexShader, occlusionFragShader);
  const godrayProgram = createShaderProgram(gl, vertexShader, godrayFragShader);
  
  const vertices = new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]);
  const buffer = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
  gl.bufferData(gl.ARRAY_BUFFER, vertices, gl.STATIC_DRAW);
  
  const sceneTexture = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, sceneTexture);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, prev.width, prev.height, 0, gl.RGBA, gl.UNSIGNED_BYTE, prev.data);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  
  const occlusionTexture = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, occlusionTexture);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, ctx.width, ctx.height, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  
  const framebuffer = gl.createFramebuffer();
  
  const lightX = 0.5;
  const lightY = 0.5;
  const lightRadius = 0.08;
  
  // Pass 1: Render occlusion texture (light sources)
  gl.useProgram(occlusionProgram);
  
  const occPosLoc = gl.getAttribLocation(occlusionProgram, 'position');
  gl.enableVertexAttribArray(occPosLoc);
  gl.vertexAttribPointer(occPosLoc, 2, gl.FLOAT, false, 0, 0);
  
  gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);
  gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, occlusionTexture, 0);
  gl.viewport(0, 0, ctx.width, ctx.height);
  
  gl.activeTexture(gl.TEXTURE0);
  gl.bindTexture(gl.TEXTURE_2D, sceneTexture);
  gl.uniform1i(gl.getUniformLocation(occlusionProgram, 'uTexture'), 0);
  gl.uniform2f(gl.getUniformLocation(occlusionProgram, 'uLightPos'), lightX, lightY);
  gl.uniform1f(gl.getUniformLocation(occlusionProgram, 'uLightRadius'), lightRadius);
  
  gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  
  // Pass 2: Apply god rays and combine with scene
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  gl.viewport(0, 0, ctx.width, ctx.height);
  
  gl.useProgram(godrayProgram);
  
  const grPosLoc = gl.getAttribLocation(godrayProgram, 'position');
  gl.enableVertexAttribArray(grPosLoc);
  gl.vertexAttribPointer(grPosLoc, 2, gl.FLOAT, false, 0, 0);
  
  gl.activeTexture(gl.TEXTURE0);
  gl.bindTexture(gl.TEXTURE_2D, occlusionTexture);
  gl.uniform1i(gl.getUniformLocation(godrayProgram, 'uOcclusionTexture'), 0);
  
  gl.activeTexture(gl.TEXTURE1);
  gl.bindTexture(gl.TEXTURE_2D, sceneTexture);
  gl.uniform1i(gl.getUniformLocation(godrayProgram, 'uSceneTexture'), 1);
  
  gl.uniform2f(gl.getUniformLocation(godrayProgram, 'uLightPos'), lightX, lightY);
  gl.uniform1f(gl.getUniformLocation(godrayProgram, 'uExposure'), 0.15);
  gl.uniform1f(gl.getUniformLocation(godrayProgram, 'uDecay'), 0.96);
  gl.uniform1f(gl.getUniformLocation(godrayProgram, 'uDensity'), 0.85);
  gl.uniform1f(gl.getUniformLocation(godrayProgram, 'uWeight'), 0.4);
  
  gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  
  const pixels = new Uint8ClampedArray(ctx.width * ctx.height * 4);
  gl.readPixels(0, 0, ctx.width, ctx.height, gl.RGBA, gl.UNSIGNED_BYTE, pixels);
  
  const flipped = new Uint8ClampedArray(ctx.width * ctx.height * 4);
  for (let y = 0; y < ctx.height; y++) {
    for (let x = 0; x < ctx.width; x++) {
      const srcIdx = ((ctx.height - 1 - y) * ctx.width + x) * 4;
      const dstIdx = (y * ctx.width + x) * 4;
      flipped[dstIdx] = pixels[srcIdx];
      flipped[dstIdx + 1] = pixels[srcIdx + 1];
      flipped[dstIdx + 2] = pixels[srcIdx + 2];
      flipped[dstIdx + 3] = pixels[srcIdx + 3];
    }
  }
  
  gl.deleteTexture(sceneTexture);
  gl.deleteTexture(occlusionTexture);
  gl.deleteFramebuffer(framebuffer);
  gl.deleteBuffer(buffer);
  gl.deleteProgram(occlusionProgram);
  gl.deleteProgram(godrayProgram);
  
  return { width: ctx.width, height: ctx.height, data: flipped };
}

function fnPlus(ctx: FnContext): Image {
  const prev = getPrevImage(ctx);
  const out = createSolidImage(ctx.width, ctx.height, '#000000');
  
  const cx = ctx.width / 2;
  const cy = ctx.height / 2;
  const zoom = 1.2;
  
  for (let y = 0; y < ctx.height; y++) {
    for (let x = 0; x < ctx.width; x++) {
      const sx = cx + (x - cx) / zoom;
      const sy = cy + (y - cy) / zoom;
      const [r, g, b] = getPixel(prev, Math.floor(sx), Math.floor(sy));
      setPixel(out, x, y, r, g, b);
    }
  }
  
  return out;
}

function fnComma(ctx: FnContext, c: string): Image {
  const prev = getPrevImage(ctx);
  const out = cloneImage(prev);
  const [cr, cg, cb] = hexToRgb(c);
  
  for (let y = 0; y < ctx.height; y++) {
    for (let x = 0; x < ctx.width; x++) {
      const [r, g, b] = getPixel(prev, x, y);
      const luminance = Math.floor(r * 0.299 + g * 0.587 + b * 0.114);
      const divisor = 1 + Math.floor(luminance / 32);
      
      if ((x * 13 + y * 7) % divisor === 0) {
        setPixel(out, x, y, cr, cg, cb);
      }
    }
  }
  
  return out;
}

function fnMinus(ctx: FnContext): Image {
  const prev = getPrevImage(ctx);
  const out = createSolidImage(ctx.width, ctx.height, '#000000');
  const spacing = 2;
  
  for (let y = 0; y < ctx.height; y++) {
    for (let x = 0; x < ctx.width; x++) {
      let [r, g, b] = getPixel(prev, x, y);
      
      if (y % spacing === 0) {
        r = Math.floor(r * 0.5);
        g = Math.floor(g * 0.5);
        b = Math.floor(b * 0.5);
      }
      
      if (y % (spacing * 2) === 0) {
        const srcY = Math.max(0, y - spacing);
        [r, g, b] = getPixel(prev, x, srcY);
      }
      
      setPixel(out, x, y, r, g, b);
    }
  }
  
  return out;
}

function fnDot(ctx: FnContext, n: number): Image {
  const prev = getPrevImage(ctx);
  const out = createSolidImage(ctx.width, ctx.height, '#000000');
  const radius = (n % 8) + 2;
  const diameter = radius * 2;
  
  for (let cy = radius; cy < ctx.height; cy += diameter) {
    for (let cx = radius; cx < ctx.width; cx += diameter) {
      const [r, g, b] = getPixel(prev, cx, cy);
      const [h, s, l] = rgbToHsl(r, g, b);
      const [nr, ng, nb] = hslToRgb(h, Math.min(1, s + 0.1), l);
      
      for (let dy = -radius; dy <= radius; dy++) {
        for (let dx = -radius; dx <= radius; dx++) {
          if (dx * dx + dy * dy <= radius * radius) {
            setPixel(out, cx + dx, cy + dy, nr, ng, nb);
          }
        }
      }
    }
  }
  
  return out;
}

function fnSlash(ctx: FnContext, old: Image, offX: number, offY: number, size: number, blend: number): Image {
  const prev = getPrevImage(ctx);
  const out = cloneImage(prev);
  
  const norm = (n: number) => Math.max(0, Math.min(1, (n - 1) / 67));
  
  // Source: full circle from center of old image
  const srcCenterX = old.width / 2;
  const srcCenterY = old.height / 2;
  const srcRadius = Math.min(old.width, old.height) / 2;
  
  // Destination position and size
  const dstX = norm(offX) * ctx.width;
  const dstY = norm(offY) * ctx.height;
  const dstRadius = Math.max(1, norm(size) * Math.min(ctx.width, ctx.height));
  
  // Blend mode (0-15)
  const NUM_BLEND_MODES = 16;
  const blendMode = (blend - 1) % NUM_BLEND_MODES;
  
  const blendFuncs: ((b: number, t: number) => number)[] = [
    (b, t) => t,
    (b, t) => b ^ t,
    (b, t) => 255 - (b & t),
    (b, t) => b & t,
    (b, t) => b | t,
    (b, t) => (b * t) / 255,
    (b, t) => 255 - ((255 - b) * (255 - t)) / 255,
    (b, t) => b < 128 ? (2 * b * t) / 255 : 255 - (2 * (255 - b) * (255 - t)) / 255,
    (b, t) => Math.min(b, t),
    (b, t) => Math.max(b, t),
    (b, t) => Math.abs(b - t),
    (b, t) => b + t - (2 * b * t) / 255,
    (b, t) => Math.min(255, b + t),
    (b, t) => Math.max(0, b - t),
    (b, t) => t < 128 ? (2 * b * t) / 255 : 255 - (2 * (255 - b) * (255 - t)) / 255,
    (b, t) => {
      const tb = t / 255, bb = b / 255;
      return Math.round((tb < 0.5 ? bb - (1 - 2 * tb) * bb * (1 - bb) : bb + (2 * tb - 1) * (bb < 0.25 ? ((16 * bb - 12) * bb + 4) * bb : Math.sqrt(bb) - bb)) * 255);
    },
  ];
  
  const blendFunc = blendFuncs[blendMode];
  
  // Scale factor from destination to source
  const scale = srcRadius / dstRadius;
  
  // Iterate over bounding box of destination circle
  const startX = Math.max(0, Math.floor(dstX - dstRadius));
  const endX = Math.min(ctx.width, Math.ceil(dstX + dstRadius));
  const startY = Math.max(0, Math.floor(dstY - dstRadius));
  const endY = Math.min(ctx.height, Math.ceil(dstY + dstRadius));
  
  for (let py = startY; py < endY; py++) {
    for (let px = startX; px < endX; px++) {
      const dx = px - dstX;
      const dy = py - dstY;
      const distSq = dx * dx + dy * dy;
      
      if (distSq > dstRadius * dstRadius) continue;
      
      // Map to source coordinates
      const srcPxF = srcCenterX + dx * scale;
      const srcPyF = srcCenterY + dy * scale;
      
      // Bilinear interpolation
      const x0 = Math.floor(srcPxF);
      const y0 = Math.floor(srcPyF);
      const x1 = Math.min(old.width - 1, x0 + 1);
      const y1 = Math.min(old.height - 1, y0 + 1);
      const fx = srcPxF - x0;
      const fy = srcPyF - y0;
      
      const [r00, g00, b00] = getPixel(old, x0, y0);
      const [r10, g10, b10] = getPixel(old, x1, y0);
      const [r01, g01, b01] = getPixel(old, x0, y1);
      const [r11, g11, b11] = getPixel(old, x1, y1);
      
      const srcR = Math.round(r00 * (1 - fx) * (1 - fy) + r10 * fx * (1 - fy) + r01 * (1 - fx) * fy + r11 * fx * fy);
      const srcG = Math.round(g00 * (1 - fx) * (1 - fy) + g10 * fx * (1 - fy) + g01 * (1 - fx) * fy + g11 * fx * fy);
      const srcB = Math.round(b00 * (1 - fx) * (1 - fy) + b10 * fx * (1 - fy) + b01 * (1 - fx) * fy + b11 * fx * fy);
      
      const [baseR, baseG, baseB] = getPixel(prev, px, py);
      
      const r = Math.round(blendFunc(baseR, srcR));
      const g = Math.round(blendFunc(baseG, srcG));
      const b = Math.round(blendFunc(baseB, srcB));
      
      setPixel(out, px, py, r, g, b);
    }
  }
  
  return out;
}

function fnColon(ctx: FnContext, old: Image): Image {
  const prev = getPrevImage(ctx);
  const out = createSolidImage(ctx.width, ctx.height, '#000000');
  
  const cx = ctx.width / 2;
  const cy = ctx.height / 2;
  const radius = Math.min(cx, cy) * 0.9;
  
  for (let y = 0; y < ctx.height; y++) {
    for (let x = 0; x < ctx.width; x++) {
      const dx = x - cx;
      const dy = y - cy;
      const dist = Math.sqrt(dx * dx + dy * dy);
      
      if (dist < radius) {
        const [r, g, b] = getPixel(old, x, y);
        setPixel(out, x, y, r, g, b);
      } else {
        const [r, g, b] = getPixel(prev, x, y);
        setPixel(out, x, y, r, g, b);
      }
    }
  }
  
  return out;
}

function fnSemicolon(ctx: FnContext): Image {
  const prev = getPrevImage(ctx);
  const out = createSolidImage(ctx.width, ctx.height, '#000000');
  
  const cx = ctx.width / 2;
  const cy = ctx.height / 2;
  const radius = Math.min(cx, cy) * 0.9;
  
  for (let y = 0; y < ctx.height; y++) {
    for (let x = 0; x < ctx.width; x++) {
      const dx = x - cx;
      const dy = y - cy;
      const dist = Math.sqrt(dx * dx + dy * dy);
      
      if (y < cy && dist < radius) {
        const [r, g, b] = getPixel(prev, x, y);
        setPixel(out, x, y, r, g, b);
      } else if (y >= cy) {
        const mirrorY = 2 * cy - y;
        const mirrorDx = x - cx;
        const mirrorDy = mirrorY - cy;
        const mirrorDist = Math.sqrt(mirrorDx * mirrorDx + mirrorDy * mirrorDy);
        
        if (mirrorDist < radius) {
          const wave = Math.sin(x * 0.1) * 10;
          const srcY = Math.floor(mirrorY + wave);
          const [r, g, b] = getPixel(prev, x, Math.max(0, Math.min(ctx.height - 1, srcY)));
          setPixel(out, x, y, r, g, b);
        }
      }
    }
  }
  
  return out;
}

function fnEquals(ctx: FnContext, n: number): Image {
  const prev = getPrevImage(ctx);
  const out = createSolidImage(ctx.width, ctx.height, '#000000');
  
  const stripeHeight = Math.max(2, n);
  
  for (let y = 0; y < ctx.height; y++) {
    const stripeIdx = Math.floor(y / stripeHeight);
    const isOdd = stripeIdx % 2 === 1;
    const shift = isOdd ? stripeIdx * 5 : 0;
    
    for (let x = 0; x < ctx.width; x++) {
      const srcX = ((x + shift) % ctx.width + ctx.width) % ctx.width;
      const [r, g, b] = getPixel(prev, srcX, y);
      setPixel(out, x, y, r, g, b);
    }
  }
  
  return out;
}

function fnQuestion(ctx: FnContext): Image {
  const prev = getPrevImage(ctx);
  const out = createSolidImage(ctx.width, ctx.height, '#000000');
  
  const corners = [
    getPixel(prev, 0, 0),
    getPixel(prev, ctx.width - 1, 0),
    getPixel(prev, 0, ctx.height - 1),
    getPixel(prev, ctx.width - 1, ctx.height - 1)
  ];
  
  let sum = 0;
  for (const [r, g, b] of corners) {
    sum += r + g + b;
  }
  const effect = sum % 4;
  
  for (let y = 0; y < ctx.height; y++) {
    for (let x = 0; x < ctx.width; x++) {
      const [r, g, b] = getPixel(prev, x, y);
      
      if (effect === 0) {
        const sr = r < 128 ? r : 255 - r;
        const sg = g < 128 ? g : 255 - g;
        const sb = b < 128 ? b : 255 - b;
        setPixel(out, x, y, sr * 2, sg * 2, sb * 2);
      } else if (effect === 1) {
        let gx = 0, gy = 0;
        for (let ky = -1; ky <= 1; ky++) {
          for (let kx = -1; kx <= 1; kx++) {
            const [pr, pg, pb] = getPixel(prev, x + kx, y + ky);
            const gray = pr * 0.299 + pg * 0.587 + pb * 0.114;
            const weight = (kx === 0 && ky === 0) ? 8 : -1;
            gx += gray * weight;
          }
        }
        const embossVal = Math.max(0, Math.min(255, 128 + gx / 4));
        setPixel(out, x, y, embossVal, embossVal, embossVal);
      } else if (effect === 2) {
        const sobelX = [-1, 0, 1, -2, 0, 2, -1, 0, 1];
        const sobelY = [-1, -2, -1, 0, 0, 0, 1, 2, 1];
        let gx = 0, gy = 0;
        for (let ky = -1; ky <= 1; ky++) {
          for (let kx = -1; kx <= 1; kx++) {
            const [pr, pg, pb] = getPixel(prev, x + kx, y + ky);
            const gray = pr * 0.299 + pg * 0.587 + pb * 0.114;
            const kidx = (ky + 1) * 3 + (kx + 1);
            gx += gray * sobelX[kidx];
            gy += gray * sobelY[kidx];
          }
        }
        const mag = Math.min(255, Math.sqrt(gx * gx + gy * gy));
        setPixel(out, x, y, mag, mag, mag);
      } else {
        const levels = 4;
        const pr = Math.floor(r / 256 * levels) * (255 / (levels - 1));
        const pg = Math.floor(g / 256 * levels) * (255 / (levels - 1));
        const pb = Math.floor(b / 256 * levels) * (255 / (levels - 1));
        setPixel(out, x, y, pr, pg, pb);
      }
    }
  }
  
  return out;
}

function fnOpenBracket(ctx: FnContext): Image {
  const prev = getPrevImage(ctx);
  const out = createSolidImage(ctx.width, ctx.height, '#000000');
  
  const angle = -20 * Math.PI / 180;
  const cos = Math.cos(angle);
  const sin = Math.sin(angle);
  const cx = ctx.width / 2;
  const cy = ctx.height / 2;
  
  for (let y = 0; y < ctx.height; y++) {
    for (let x = 0; x < ctx.width; x++) {
      const dx = x - cx;
      const dy = y - cy;
      const srcX = cx + dx * cos - dy * sin;
      const srcY = cy + dx * sin + dy * cos;
      const [r, g, b] = getPixel(prev, Math.floor(srcX), Math.floor(srcY));
      setPixel(out, x, y, r, g, b);
    }
  }
  
  return out;
}

function fnBackslash(
  ctx: FnContext,
  old: Image,
  srcX: number,
  srcY: number,
  srcW: number,
  srcH: number,
  dstX: number,
  dstY: number,
  dstW: number,
  dstH: number,
  rot: number,
  blend: number
): Image {
  const prev = getPrevImage(ctx);
  const out = cloneImage(prev);
  
  // Map integer values (1-68) to normalized 0-1 range
  const norm = (n: number) => Math.max(0, Math.min(1, (n - 1) / 67));
  
  // Source crop region in old image
  // X and Y use full range, W and H use remaining space after offset
  const sourceCropX = Math.floor(norm(srcX) * old.width);
  const sourceCropY = Math.floor(norm(srcY) * old.height);
  const sourceCropW = Math.max(1, Math.round(norm(srcW) * (old.width - sourceCropX)));
  const sourceCropH = Math.max(1, Math.round(norm(srcH) * (old.height - sourceCropY)));
  
  // Destination region in output image
  // X and Y use full range, W and H use remaining space after offset
  const destX = Math.floor(norm(dstX) * ctx.width);
  const destY = Math.floor(norm(dstY) * ctx.height);
  const destW = Math.max(1, Math.round(norm(dstW) * (ctx.width - destX)));
  const destH = Math.max(1, Math.round(norm(dstH) * (ctx.height - destY)));
  
  // Rotation angle (0-360 degrees)
  const rotation = norm(rot) * 2 * Math.PI;
  
  // Blend mode (0-15)
  const NUM_BLEND_MODES = 16;
  const blendMode = (blend - 1) % NUM_BLEND_MODES;
  
  // Blend mode functions: (base, top) => result (all values 0-255)
  const blendFuncs: ((b: number, t: number) => number)[] = [
    // 0: Normal - replace
    (b, t) => t,
    // 1: XOR
    (b, t) => b ^ t,
    // 2: NAND
    (b, t) => 255 - (b & t),
    // 3: AND
    (b, t) => b & t,
    // 4: OR
    (b, t) => b | t,
    // 5: Multiply
    (b, t) => (b * t) / 255,
    // 6: Screen
    (b, t) => 255 - ((255 - b) * (255 - t)) / 255,
    // 7: Overlay
    (b, t) => b < 128 ? (2 * b * t) / 255 : 255 - (2 * (255 - b) * (255 - t)) / 255,
    // 8: Darken
    (b, t) => Math.min(b, t),
    // 9: Lighten
    (b, t) => Math.max(b, t),
    // 10: Difference
    (b, t) => Math.abs(b - t),
    // 11: Exclusion
    (b, t) => b + t - (2 * b * t) / 255,
    // 12: Add (clamped)
    (b, t) => Math.min(255, b + t),
    // 13: Subtract (clamped)
    (b, t) => Math.max(0, b - t),
    // 14: Hard Light
    (b, t) => t < 128 ? (2 * b * t) / 255 : 255 - (2 * (255 - b) * (255 - t)) / 255,
    // 15: Soft Light
    (b, t) => {
      const tb = t / 255, bb = b / 255;
      return Math.round((tb < 0.5 ? bb - (1 - 2 * tb) * bb * (1 - bb) : bb + (2 * tb - 1) * (bb < 0.25 ? ((16 * bb - 12) * bb + 4) * bb : Math.sqrt(bb) - bb)) * 255);
    },
  ];
  
  const blendFunc = blendFuncs[blendMode];
  
  const destCenterX = destX + destW / 2;
  const destCenterY = destY + destH / 2;
  const cosR = Math.cos(-rotation);
  const sinR = Math.sin(-rotation);
  
  // Calculate bounding box of rotated rectangle
  const halfW = destW / 2;
  const halfH = destH / 2;
  const corners = [
    [-halfW, -halfH],
    [halfW, -halfH],
    [halfW, halfH],
    [-halfW, halfH]
  ];
  
  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
  for (const [cx, cy] of corners) {
    const rx = cx * Math.cos(rotation) - cy * Math.sin(rotation) + destCenterX;
    const ry = cx * Math.sin(rotation) + cy * Math.cos(rotation) + destCenterY;
    minX = Math.min(minX, rx);
    maxX = Math.max(maxX, rx);
    minY = Math.min(minY, ry);
    maxY = Math.max(maxY, ry);
  }
  
  const startX = Math.max(0, Math.floor(minX));
  const endX = Math.min(ctx.width, Math.ceil(maxX));
  const startY = Math.max(0, Math.floor(minY));
  const endY = Math.min(ctx.height, Math.ceil(maxY));
  
  for (let py = startY; py < endY; py++) {
    for (let px = startX; px < endX; px++) {
      // Inverse rotation to find source coordinates
      const relX = px - destCenterX;
      const relY = py - destCenterY;
      const rotX = relX * cosR - relY * sinR;
      const rotY = relX * sinR + relY * cosR;
      
      // Map to normalized coordinates in destination rect
      const normX = (rotX + halfW) / destW;
      const normY = (rotY + halfH) / destH;
      
      // Check if within the destination rectangle (0-1 range)
      if (normX < 0 || normX >= 1 || normY < 0 || normY >= 1) continue;
      
      // Map to source coordinates with bilinear sampling
      const srcPxF = sourceCropX + normX * sourceCropW;
      const srcPyF = sourceCropY + normY * sourceCropH;
      
      const x0 = Math.floor(srcPxF);
      const y0 = Math.floor(srcPyF);
      const x1 = Math.min(old.width - 1, x0 + 1);
      const y1 = Math.min(old.height - 1, y0 + 1);
      const fx = srcPxF - x0;
      const fy = srcPyF - y0;
      
      const [r00, g00, b00] = getPixel(old, x0, y0);
      const [r10, g10, b10] = getPixel(old, x1, y0);
      const [r01, g01, b01] = getPixel(old, x0, y1);
      const [r11, g11, b11] = getPixel(old, x1, y1);
      
      // Bilinear interpolation for source pixel
      const srcR = Math.round(r00 * (1 - fx) * (1 - fy) + r10 * fx * (1 - fy) + r01 * (1 - fx) * fy + r11 * fx * fy);
      const srcG = Math.round(g00 * (1 - fx) * (1 - fy) + g10 * fx * (1 - fy) + g01 * (1 - fx) * fy + g11 * fx * fy);
      const srcB = Math.round(b00 * (1 - fx) * (1 - fy) + b10 * fx * (1 - fy) + b01 * (1 - fx) * fy + b11 * fx * fy);
      
      // Get base pixel from prev
      const [baseR, baseG, baseB] = getPixel(prev, px, py);
      
      // Apply blend mode
      const r = Math.round(blendFunc(baseR, srcR));
      const g = Math.round(blendFunc(baseG, srcG));
      const b = Math.round(blendFunc(baseB, srcB));
      
      setPixel(out, px, py, r, g, b);
    }
  }
  
  return out;
}

function fnCloseBracket(ctx: FnContext): Image {
  const prev = getPrevImage(ctx);
  const out = createSolidImage(ctx.width, ctx.height, '#000000');
  
  const offset = Math.floor(ctx.height * 0.2);
  const midX = Math.floor(ctx.width / 2);
  
  for (let y = 0; y < ctx.height; y++) {
    for (let x = 0; x < ctx.width; x++) {
      if (x < midX) {
        const srcY = (y + offset) % ctx.height;
        const [r, g, b] = getPixel(prev, x, srcY);
        setPixel(out, x, y, r, g, b);
      } else {
        const [r, g, b] = getPixel(prev, x, y);
        setPixel(out, x, y, r, g, b);
      }
    }
  }
  
  return out;
}

function fnBlend(ctx: FnContext, modeName: string): Image {
  const prev = getPrevImage(ctx);
  const prev1 = ctx.images.length >= 2 ? ctx.images[ctx.images.length - 2] : prev;
  const out = createSolidImage(ctx.width, ctx.height, '#000000');
  
  for (let y = 0; y < ctx.height; y++) {
    for (let x = 0; x < ctx.width; x++) {
      const [br, bg, bb] = getPixel(prev1, x, y);
      const [tr, tg, tb] = getPixel(prev, x, y);
      
      let r: number, g: number, b: number;
      
      switch (modeName) {
        case 'multiply':
          r = (br * tr) / 255;
          g = (bg * tg) / 255;
          b = (bb * tb) / 255;
          break;
          
        case 'screen':
          r = 255 - ((255 - br) * (255 - tr)) / 255;
          g = 255 - ((255 - bg) * (255 - tg)) / 255;
          b = 255 - ((255 - bb) * (255 - tb)) / 255;
          break;
          
        case 'overlay':
          r = br < 128 ? (2 * br * tr) / 255 : 255 - (2 * (255 - br) * (255 - tr)) / 255;
          g = bg < 128 ? (2 * bg * tg) / 255 : 255 - (2 * (255 - bg) * (255 - tg)) / 255;
          b = bb < 128 ? (2 * bb * tb) / 255 : 255 - (2 * (255 - bb) * (255 - tb)) / 255;
          break;
          
        case 'darken':
          r = Math.min(br, tr);
          g = Math.min(bg, tg);
          b = Math.min(bb, tb);
          break;
          
        case 'lighten':
          r = Math.max(br, tr);
          g = Math.max(bg, tg);
          b = Math.max(bb, tb);
          break;
          
        case 'dodge':
          r = tr === 255 ? 255 : Math.min(255, (br * 255) / (255 - tr));
          g = tg === 255 ? 255 : Math.min(255, (bg * 255) / (255 - tg));
          b = tb === 255 ? 255 : Math.min(255, (bb * 255) / (255 - tb));
          break;
          
        case 'burn':
          r = tr === 0 ? 0 : Math.max(0, 255 - ((255 - br) * 255) / tr);
          g = tg === 0 ? 0 : Math.max(0, 255 - ((255 - bg) * 255) / tg);
          b = tb === 0 ? 0 : Math.max(0, 255 - ((255 - bb) * 255) / tb);
          break;
          
        case 'hardlight':
          r = tr < 128 ? (2 * br * tr) / 255 : 255 - (2 * (255 - br) * (255 - tr)) / 255;
          g = tg < 128 ? (2 * bg * tg) / 255 : 255 - (2 * (255 - bg) * (255 - tg)) / 255;
          b = tb < 128 ? (2 * bb * tb) / 255 : 255 - (2 * (255 - bb) * (255 - tb)) / 255;
          break;
          
        case 'softlight': {
          const softLight = (b: number, t: number) => {
            const tb = t / 255, bb = b / 255;
            return (tb < 0.5 
              ? bb - (1 - 2 * tb) * bb * (1 - bb) 
              : bb + (2 * tb - 1) * (bb < 0.25 ? ((16 * bb - 12) * bb + 4) * bb : Math.sqrt(bb) - bb)) * 255;
          };
          r = softLight(br, tr);
          g = softLight(bg, tg);
          b = softLight(bb, tb);
          break;
        }
          
        case 'difference':
          r = Math.abs(br - tr);
          g = Math.abs(bg - tg);
          b = Math.abs(bb - tb);
          break;
          
        case 'exclusion':
          r = br + tr - (2 * br * tr) / 255;
          g = bg + tg - (2 * bg * tg) / 255;
          b = bb + tb - (2 * bb * tb) / 255;
          break;
          
        case 'add':
          r = Math.min(255, br + tr);
          g = Math.min(255, bg + tg);
          b = Math.min(255, bb + tb);
          break;
          
        case 'subtract':
          r = Math.max(0, br - tr);
          g = Math.max(0, bg - tg);
          b = Math.max(0, bb - tb);
          break;
          
        case 'xor':
          r = br ^ tr;
          g = bg ^ tg;
          b = bb ^ tb;
          break;
          
        case 'and':
          r = br & tr;
          g = bg & tg;
          b = bb & tb;
          break;
          
        case 'or':
          r = br | tr;
          g = bg | tg;
          b = bb | tb;
          break;
          
        case 'nand':
          r = 255 - (br & tr);
          g = 255 - (bg & tg);
          b = 255 - (bb & tb);
          break;
          
        case 'nor':
          r = 255 - (br | tr);
          g = 255 - (bg | tg);
          b = 255 - (bb | tb);
          break;
          
        case 'xnor':
          r = 255 - (br ^ tr);
          g = 255 - (bg ^ tg);
          b = 255 - (bb ^ tb);
          break;
          
        case 'average':
          r = (br + tr) / 2;
          g = (bg + tg) / 2;
          b = (bb + tb) / 2;
          break;
          
        case 'divide':
          r = tr === 0 ? 255 : Math.min(255, (br * 255) / tr);
          g = tg === 0 ? 255 : Math.min(255, (bg * 255) / tg);
          b = tb === 0 ? 255 : Math.min(255, (bb * 255) / tb);
          break;
          
        case 'grain-extract':
          r = Math.max(0, Math.min(255, br - tr + 128));
          g = Math.max(0, Math.min(255, bg - tg + 128));
          b = Math.max(0, Math.min(255, bb - tb + 128));
          break;
          
        case 'grain-merge':
          r = Math.max(0, Math.min(255, br + tr - 128));
          g = Math.max(0, Math.min(255, bg + tg - 128));
          b = Math.max(0, Math.min(255, bb + tb - 128));
          break;
          
        case 'vivid':
          r = tr < 128 
            ? (tr === 0 ? 0 : Math.max(0, 255 - ((255 - br) * 255) / (2 * tr)))
            : (tr === 255 ? 255 : Math.min(255, (br * 255) / (2 * (255 - tr))));
          g = tg < 128 
            ? (tg === 0 ? 0 : Math.max(0, 255 - ((255 - bg) * 255) / (2 * tg)))
            : (tg === 255 ? 255 : Math.min(255, (bg * 255) / (2 * (255 - tg))));
          b = tb < 128 
            ? (tb === 0 ? 0 : Math.max(0, 255 - ((255 - bb) * 255) / (2 * tb)))
            : (tb === 255 ? 255 : Math.min(255, (bb * 255) / (2 * (255 - tb))));
          break;
          
        case 'linear':
          r = Math.max(0, Math.min(255, br + 2 * tr - 255));
          g = Math.max(0, Math.min(255, bg + 2 * tg - 255));
          b = Math.max(0, Math.min(255, bb + 2 * tb - 255));
          break;
          
        case 'pin':
          r = tr < 128 ? Math.min(br, 2 * tr) : Math.max(br, 2 * tr - 255);
          g = tg < 128 ? Math.min(bg, 2 * tg) : Math.max(bg, 2 * tg - 255);
          b = tb < 128 ? Math.min(bb, 2 * tb) : Math.max(bb, 2 * tb - 255);
          break;
          
        case 'hardmix':
          r = br + tr < 255 ? 0 : 255;
          g = bg + tg < 255 ? 0 : 255;
          b = bb + tb < 255 ? 0 : 255;
          break;
          
        case 'hue': {
          const [bh, bs, bl] = rgbToHsl(br, bg, bb);
          const [th] = rgbToHsl(tr, tg, tb);
          [r, g, b] = hslToRgb(th, bs, bl);
          break;
        }
          
        case 'saturation': {
          const [bh, , bl] = rgbToHsl(br, bg, bb);
          const [, ts] = rgbToHsl(tr, tg, tb);
          [r, g, b] = hslToRgb(bh, ts, bl);
          break;
        }
          
        case 'color': {
          const [, , bl] = rgbToHsl(br, bg, bb);
          const [th, ts] = rgbToHsl(tr, tg, tb);
          [r, g, b] = hslToRgb(th, ts, bl);
          break;
        }
          
        case 'luminosity': {
          const [bh, bs] = rgbToHsl(br, bg, bb);
          const [, , tl] = rgbToHsl(tr, tg, tb);
          [r, g, b] = hslToRgb(bh, bs, tl);
          break;
        }
        
        case 'replace-dark-third': {
          const [, , bl] = rgbToHsl(br, bg, bb);
          if (bl < 0.333) {
            r = tr; g = tg; b = tb;
          } else {
            r = br; g = bg; b = bb;
          }
          break;
        }
        
        case 'replace-mid-third': {
          const [, , bl] = rgbToHsl(br, bg, bb);
          if (bl >= 0.333 && bl < 0.667) {
            r = tr; g = tg; b = tb;
          } else {
            r = br; g = bg; b = bb;
          }
          break;
        }
        
        case 'replace-light-third': {
          const [, , bl] = rgbToHsl(br, bg, bb);
          if (bl >= 0.667) {
            r = tr; g = tg; b = tb;
          } else {
            r = br; g = bg; b = bb;
          }
          break;
        }
          
        default:
          r = tr; g = tg; b = tb;
      }
      
      setPixel(out, x, y, Math.round(r), Math.round(g), Math.round(b));
    }
  }
  
  return out;
}

function fnImageHistory(ctx: FnContext): Image {
  const out = createSolidImage(ctx.width, ctx.height, '#000000');
  
  const tempCanvas = document.createElement('canvas');
  tempCanvas.width = ctx.width;
  tempCanvas.height = ctx.height;
  const tempCtx = tempCanvas.getContext('2d')!;
  
  tempCtx.fillStyle = '#000000';
  tempCtx.fillRect(0, 0, ctx.width, ctx.height);
  
  // Skip the initial black placeholder at index 0
  const numImages = ctx.images.length - 1;
  if (numImages === 0) {
    tempCtx.fillStyle = '#00FF00';
    tempCtx.font = '16px monospace';
    tempCtx.fillText('No images in history', 10, 30);
    const imageData = tempCtx.getImageData(0, 0, ctx.width, ctx.height);
    out.data.set(imageData.data);
    return out;
  }
  
  const margin = 1;
  const availWidth = ctx.width - margin * 2;
  const availHeight = ctx.height - margin * 2;
  
  // Calculate optimal layout
  let bestLayout = { cols: 1, rows: numImages, thumbSize: 10, fontSize: 6 };
  let bestScore = 0;
  
  for (let cols = 1; cols <= Math.min(10, numImages); cols++) {
    const rows = Math.ceil(numImages / cols);
    
    // Each cell needs: thumbnail + text space
    const cellWidth = availWidth / cols;
    const cellHeight = availHeight / rows;
    
    // Reserve space for text (1 line: #n [C] AAA) - minimal, plus bottom margin
    const textHeight = Math.min(cellHeight * 0.15, 14);
    const fontSize = Math.max(6, Math.min(10, textHeight));
    const bottomMargin = 3;
    
    // Thumbnail gets remaining space with minimal padding
    const thumbSize = Math.min(cellWidth - 2, cellHeight - textHeight - bottomMargin - 2);
    
    if (thumbSize > 8 && fontSize >= 6) {
      // Score based on thumbnail size and font readability
      const score = thumbSize * fontSize;
      if (score > bestScore) {
        bestScore = score;
        bestLayout = { cols, rows, thumbSize, fontSize };
      }
    }
  }
  
  const { cols, thumbSize, fontSize } = bestLayout;
  const cellWidth = availWidth / cols;
  const cellHeight = availHeight / Math.ceil(numImages / cols);
  
  tempCtx.fillStyle = '#00FF00';
  tempCtx.font = `${fontSize}px monospace`;
  
  // Number to character mapping (same as in characterDefs)
  const numToChar = (num: number): string => {
    if (num >= 1 && num <= 26) return String.fromCharCode('A'.charCodeAt(0) + num - 1);
    if (num >= 27 && num <= 36) return String.fromCharCode('0'.charCodeAt(0) + num - 27);
    const symbols = '<>^!"#$%&\'()*+,-./:;=?@[\\]_`{|}~';
    const idx = num - 37;
    if (idx >= 0 && idx < symbols.length) return symbols[idx];
    return '?';
  };
  
  for (let displayIdx = 0; displayIdx < numImages; displayIdx++) {
    const col = displayIdx % cols;
    const row = Math.floor(displayIdx / cols);
    
    const x = margin + col * cellWidth;
    const y = margin + row * cellHeight;
    
    // Actual image index is displayIdx + 1 (skip index 0)
    const i = displayIdx + 1;
    const img = ctx.images[i];
    const accessKey = numToChar(i);
    
    // Get operation info for this image
    const opInfo = ctx.opInfos[i];
    const prevOpIdentifier = i > 1 ? ctx.opInfos[i - 1].identifier : '';
    const opChars = opInfo.identifier.substring(prevOpIdentifier.length);
    const displayOp = opChars || '?';
    
    // Draw thumbnail
    const thumbX = x + (cellWidth - thumbSize) / 2;
    const thumbY = y + 1;
    
    // Create temp canvas for thumbnail
    const thumbCanvas = document.createElement('canvas');
    thumbCanvas.width = thumbSize;
    thumbCanvas.height = thumbSize;
    const thumbCtx = thumbCanvas.getContext('2d')!;
    
    // Draw image scaled to thumbnail
    const srcSize = Math.min(img.width, img.height);
    const srcX = (img.width - srcSize) / 2;
    const srcY = (img.height - srcSize) / 2;
    
    const srcCanvas = document.createElement('canvas');
    srcCanvas.width = img.width;
    srcCanvas.height = img.height;
    const srcCtx = srcCanvas.getContext('2d')!;
    const srcImageData = new ImageData(img.data, img.width, img.height);
    srcCtx.putImageData(srcImageData, 0, 0);
    
    thumbCtx.drawImage(srcCanvas, srcX, srcY, srcSize, srcSize, 0, 0, thumbSize, thumbSize);
    
    // Draw thumbnail border
    tempCtx.strokeStyle = '#00FF00';
    tempCtx.lineWidth = 1;
    tempCtx.strokeRect(thumbX, thumbY, thumbSize, thumbSize);
    
    // Draw thumbnail
    tempCtx.drawImage(thumbCanvas, thumbX, thumbY);
    
    // Draw text below thumbnail on single line
    // Display with 1-based numbering to match access keys
    const textY = thumbY + thumbSize + fontSize + 1;
    tempCtx.fillStyle = '#00FF00';
    tempCtx.textAlign = 'center';
    tempCtx.fillText(`#${i} [${accessKey}] ${displayOp}`, thumbX + thumbSize / 2, textY);
  }
  
  const imageData = tempCtx.getImageData(0, 0, ctx.width, ctx.height);
  out.data.set(imageData.data);
  
  return out;
}

function fnBacktick(ctx: FnContext, n: number): Image {
  const prev = getPrevImage(ctx);
  const { width, height } = ctx;
  
  const sobelX = [-1, 0, 1, -2, 0, 2, -1, 0, 1];
  const sobelY = [-1, -2, -1, 0, 0, 0, 1, 2, 1];
  
  const gradX = new Float32Array(width * height);
  const gradY = new Float32Array(width * height);
  const gradMag = new Float32Array(width * height);
  
  let maxMag = 1;
  
  for (let y = 1; y < height - 1; y++) {
    for (let x = 1; x < width - 1; x++) {
      let gx = 0, gy = 0;
      
      for (let ky = -1; ky <= 1; ky++) {
        for (let kx = -1; kx <= 1; kx++) {
          const [r, g, b] = getPixel(prev, x + kx, y + ky);
          const gray = r * 0.299 + g * 0.587 + b * 0.114;
          const kidx = (ky + 1) * 3 + (kx + 1);
          gx += gray * sobelX[kidx];
          gy += gray * sobelY[kidx];
        }
      }
      
      const idx = y * width + x;
      gradX[idx] = gx;
      gradY[idx] = gy;
      const mag = Math.sqrt(gx * gx + gy * gy);
      gradMag[idx] = mag;
      if (mag > maxMag) maxMag = mag;
    }
  }
  
  const iterations = Math.max(5, Math.min(n * 2 + 8, 30));
  const baseStrength = 4.0 + n * 1.5;
  const threshold = 8;
  
  let current = cloneImage(prev);
  
  for (let iter = 0; iter < iterations; iter++) {
    const next = createSolidImage(width, height, '#000000');
    const iterDecay = 1 - (iter / iterations) * 0.35;
    const iterStrength = baseStrength * iterDecay;
    
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const idx = y * width + x;
        const mag = gradMag[idx];
        
        if (mag < threshold) {
          const [r, g, b] = getPixel(current, x, y);
          setPixel(next, x, y, r, g, b);
          continue;
        }
        
        const normMag = mag / maxMag;
        const scale = normMag * iterStrength;
        const dx = (gradX[idx] / mag) * scale;
        const dy = (gradY[idx] / mag) * scale;
        
        const srcX = x - dx;
        const srcY = y - dy;
        
        const [r1, g1, b1] = getPixel(current, Math.floor(srcX), Math.floor(srcY));
        const [r2, g2, b2] = getPixel(current, x, y);
        
        const blend = Math.min(0.95, normMag * 0.6 + 0.35);
        setPixel(next, x, y,
          Math.round(r1 * blend + r2 * (1 - blend)),
          Math.round(g1 * blend + g2 * (1 - blend)),
          Math.round(b1 * blend + b2 * (1 - blend))
        );
      }
    }
    
    current = next;
  }
  
  return current;
}

function fnOpenBrace(ctx: FnContext): Image {
  const prev = getPrevImage(ctx);
  const gl = initWebGL(ctx.width, ctx.height);
  
  gl.activeTexture(gl.TEXTURE0);
  gl.bindTexture(gl.TEXTURE_2D, null);
  gl.activeTexture(gl.TEXTURE1);
  gl.bindTexture(gl.TEXTURE_2D, null);
  gl.activeTexture(gl.TEXTURE0);
  
  const vertexShader = `
    attribute vec2 position;
    varying vec2 vUV;
    void main() {
      vUV = vec2(position.x * 0.5 + 0.5, 1.0 - (position.y * 0.5 + 0.5));
      gl_Position = vec4(position, 0.0, 1.0);
    }
  `;
  
  const fragmentShader = `
    precision mediump float;
    uniform sampler2D texture;
    uniform vec2 resolution;
    varying vec2 vUV;
    
    void main() {
      vec2 uv = vUV;
      
      if (uv.x < 0.5) {
        float localX = uv.x * 2.0;
        float angle = (localX - 0.5) * 3.14159 * 0.4;
        float z = cos(angle);
        float newX = 0.5 + sin(angle) * 0.4;
        
        float shade = 0.6 + 0.4 * z;
        vec2 sampleUV = vec2(newX * 0.5, uv.y);
        vec3 color = texture2D(texture, sampleUV).rgb * shade;
        gl_FragColor = vec4(color, 1.0);
      } else {
        vec3 color = texture2D(texture, uv).rgb;
        gl_FragColor = vec4(color, 1.0);
      }
    }
  `;
  
  const program = createShaderProgram(gl, vertexShader, fragmentShader);
  gl.useProgram(program);
  
  const texture = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, texture);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, prev.width, prev.height, 0, gl.RGBA, gl.UNSIGNED_BYTE, prev.data);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  
  const vertices = new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]);
  const buffer = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
  gl.bufferData(gl.ARRAY_BUFFER, vertices, gl.STATIC_DRAW);
  
  const positionLoc = gl.getAttribLocation(program, 'position');
  gl.enableVertexAttribArray(positionLoc);
  gl.vertexAttribPointer(positionLoc, 2, gl.FLOAT, false, 0, 0);
  
  gl.uniform1i(gl.getUniformLocation(program, 'texture'), 0);
  gl.uniform2f(gl.getUniformLocation(program, 'resolution'), ctx.width, ctx.height);
  
  gl.viewport(0, 0, ctx.width, ctx.height);
  gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  
  const pixels = new Uint8ClampedArray(ctx.width * ctx.height * 4);
  gl.readPixels(0, 0, ctx.width, ctx.height, gl.RGBA, gl.UNSIGNED_BYTE, pixels);
  
  const flipped = new Uint8ClampedArray(ctx.width * ctx.height * 4);
  for (let y = 0; y < ctx.height; y++) {
    for (let x = 0; x < ctx.width; x++) {
      const srcIdx = ((ctx.height - 1 - y) * ctx.width + x) * 4;
      const dstIdx = (y * ctx.width + x) * 4;
      flipped[dstIdx] = pixels[srcIdx];
      flipped[dstIdx + 1] = pixels[srcIdx + 1];
      flipped[dstIdx + 2] = pixels[srcIdx + 2];
      flipped[dstIdx + 3] = pixels[srcIdx + 3];
    }
  }
  
  gl.deleteTexture(texture);
  gl.deleteBuffer(buffer);
  gl.deleteProgram(program);
  
  return { width: ctx.width, height: ctx.height, data: flipped };
}

function fnPipe(ctx: FnContext, old: Image): Image {
  const prev = getPrevImage(ctx);
  const out = createSolidImage(ctx.width, ctx.height, '#000000');
  
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
        
        const screenR = 255 - ((255 - pr) * (255 - or)) / 255;
        const screenG = 255 - ((255 - pg) * (255 - og)) / 255;
        const screenB = 255 - ((255 - pb) * (255 - ob)) / 255;
        
        const diffR = Math.abs(pr - or);
        const diffG = Math.abs(pg - og);
        const diffB = Math.abs(pb - ob);
        
        const xorR = pr ^ or;
        const xorG = pg ^ og;
        const xorB = pb ^ ob;
        
        const band = Math.floor(y / 20) % 3;
        
        let r: number, g: number, b: number;
        
        if (band === 0) {
          if (t < 0.5) {
            const localT = t * 2;
            r = pr * (1 - localT) + screenR * localT;
            g = pg * (1 - localT) + screenG * localT;
            b = pb * (1 - localT) + screenB * localT;
          } else {
            const localT = (t - 0.5) * 2;
            r = screenR * (1 - localT) + or * localT;
            g = screenG * (1 - localT) + og * localT;
            b = screenB * (1 - localT) + ob * localT;
          }
        } else if (band === 1) {
          const centerDist = Math.abs(t - 0.5) * 2;
          const diffWeight = 1 - centerDist;
          r = (pr * (1 - t) + or * t) * (1 - diffWeight) + diffR * diffWeight;
          g = (pg * (1 - t) + og * t) * (1 - diffWeight) + diffG * diffWeight;
          b = (pb * (1 - t) + ob * t) * (1 - diffWeight) + diffB * diffWeight;
        } else {
          const xorWeight = Math.sin(t * Math.PI) * 0.7;
          const baseR = pr * (1 - t) + or * t;
          const baseG = pg * (1 - t) + og * t;
          const baseB = pb * (1 - t) + ob * t;
          r = baseR * (1 - xorWeight) + xorR * xorWeight;
          g = baseG * (1 - xorWeight) + xorG * xorWeight;
          b = baseB * (1 - xorWeight) + xorB * xorWeight;
        }
        
        setPixel(out, x, y, Math.round(r), Math.round(g), Math.round(b));
      }
    }
  }
  
  return out;
}

function fnCloseBrace(ctx: FnContext): Image {
  const prev = getPrevImage(ctx);
  const gl = initWebGL(ctx.width, ctx.height);
  
  gl.activeTexture(gl.TEXTURE0);
  gl.bindTexture(gl.TEXTURE_2D, null);
  gl.activeTexture(gl.TEXTURE1);
  gl.bindTexture(gl.TEXTURE_2D, null);
  gl.activeTexture(gl.TEXTURE0);
  
  const vertexShader = `
    attribute vec2 position;
    varying vec2 vUV;
    void main() {
      vUV = vec2(position.x * 0.5 + 0.5, 1.0 - (position.y * 0.5 + 0.5));
      gl_Position = vec4(position, 0.0, 1.0);
    }
  `;
  
  const fragmentShader = `
    precision mediump float;
    uniform sampler2D texture;
    uniform vec2 resolution;
    varying vec2 vUV;
    
    void main() {
      vec2 uv = vUV;
      
      if (uv.x > 0.5) {
        float localX = (uv.x - 0.5) * 2.0;
        float angle = (localX - 0.5) * 3.14159 * 0.4;
        float z = cos(angle);
        float newX = 0.5 + sin(angle) * 0.4;
        
        float shade = 0.6 + 0.4 * z;
        vec2 sampleUV = vec2(0.5 + newX * 0.5, uv.y);
        vec3 color = texture2D(texture, sampleUV).rgb * shade;
        gl_FragColor = vec4(color, 1.0);
      } else {
        vec3 color = texture2D(texture, uv).rgb;
        gl_FragColor = vec4(color, 1.0);
      }
    }
  `;
  
  const program = createShaderProgram(gl, vertexShader, fragmentShader);
  gl.useProgram(program);
  
  const texture = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, texture);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, prev.width, prev.height, 0, gl.RGBA, gl.UNSIGNED_BYTE, prev.data);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  
  const vertices = new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]);
  const buffer = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
  gl.bufferData(gl.ARRAY_BUFFER, vertices, gl.STATIC_DRAW);
  
  const positionLoc = gl.getAttribLocation(program, 'position');
  gl.enableVertexAttribArray(positionLoc);
  gl.vertexAttribPointer(positionLoc, 2, gl.FLOAT, false, 0, 0);
  
  gl.uniform1i(gl.getUniformLocation(program, 'texture'), 0);
  gl.uniform2f(gl.getUniformLocation(program, 'resolution'), ctx.width, ctx.height);
  
  gl.viewport(0, 0, ctx.width, ctx.height);
  gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  
  const pixels = new Uint8ClampedArray(ctx.width * ctx.height * 4);
  gl.readPixels(0, 0, ctx.width, ctx.height, gl.RGBA, gl.UNSIGNED_BYTE, pixels);
  
  const flipped = new Uint8ClampedArray(ctx.width * ctx.height * 4);
  for (let y = 0; y < ctx.height; y++) {
    for (let x = 0; x < ctx.width; x++) {
      const srcIdx = ((ctx.height - 1 - y) * ctx.width + x) * 4;
      const dstIdx = (y * ctx.width + x) * 4;
      flipped[dstIdx] = pixels[srcIdx];
      flipped[dstIdx + 1] = pixels[srcIdx + 1];
      flipped[dstIdx + 2] = pixels[srcIdx + 2];
      flipped[dstIdx + 3] = pixels[srcIdx + 3];
    }
  }
  
  gl.deleteTexture(texture);
  gl.deleteBuffer(buffer);
  gl.deleteProgram(program);
  
  return { width: ctx.width, height: ctx.height, data: flipped };
}

function fnOilSlick(ctx: FnContext, warpN: number, iridN: number): Image {
  const prev = getPrevImage(ctx);
  const gl = initWebGL(ctx.width, ctx.height);
  
  gl.activeTexture(gl.TEXTURE0);
  gl.bindTexture(gl.TEXTURE_2D, null);
  gl.activeTexture(gl.TEXTURE1);
  gl.bindTexture(gl.TEXTURE_2D, null);
  gl.activeTexture(gl.TEXTURE0);
  
  const seed = ctx.images.length * 137.5 + warpN * 17.3 + iridN * 7.1;
  const depth = 1 + Math.floor(iridN / 4);
  const warpStrength = 0.05 + warpN * 0.025;
  const patternScale = 1.5;
  
  const vertexShader = `
    attribute vec2 position;
    varying vec2 vUV;
    void main() {
      vUV = vec2(position.x * 0.5 + 0.5, 1.0 - (position.y * 0.5 + 0.5));
      gl_Position = vec4(position, 0.0, 1.0);
    }
  `;
  
  const fragmentShader = `
    precision highp float;
    uniform sampler2D uTexture;
    uniform vec2 uResolution;
    uniform float uSeed;
    uniform int uDepth;
    uniform float uWarpStrength;
    uniform float uPatternScale;
    varying vec2 vUV;
    
    // Hash function for deterministic noise
    float hash(vec2 p, float seed) {
      return fract(sin(dot(p + seed * 0.1, vec2(127.1, 311.7))) * 43758.5453);
    }
    
    // Smooth noise
    float noise(vec2 p, float seed) {
      vec2 i = floor(p);
      vec2 f = fract(p);
      f = f * f * (3.0 - 2.0 * f);
      
      float a = hash(i, seed);
      float b = hash(i + vec2(1.0, 0.0), seed);
      float c = hash(i + vec2(0.0, 1.0), seed);
      float d = hash(i + vec2(1.0, 1.0), seed);
      
      return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
    }
    
    // FBM for organic patterns
    float fbm(vec2 p, float seed) {
      float value = 0.0;
      float amplitude = 0.5;
      float frequency = 1.0;
      for (int i = 0; i < 4; i++) {
        value += amplitude * noise(p * frequency, seed + float(i) * 100.0);
        amplitude *= 0.5;
        frequency *= 2.0;
      }
      return value;
    }
    
    // Domain warping effect - creates swirling patterns
    vec2 warpEffect(vec2 p, float i, float seed) {
      float angle = fbm(p * 0.8 + seed * 0.01, seed) * 6.28;
      float magnitude = fbm(p * 0.5 + seed * 0.02 + 50.0, seed + 100.0);
      return vec2(cos(angle), sin(angle)) * magnitude / (i * 0.3 + 1.0);
    }
    
    void main() {
      vec2 uv = vUV;
      float aspect = uResolution.x / uResolution.y;
      
      // Noise coordinates with offset to avoid symmetry
      vec2 noiseCoord = uv * 4.0 + vec2(uSeed * 0.1 + 5.7, uSeed * 0.07 + 3.2);
      noiseCoord.x *= aspect;
      
      // Calculate warp displacement from noise
      vec2 warp = vec2(0.0);
      for (int i = 1; i < 20; i++) {
        if (i >= uDepth) break;
        float fi = float(i);
        warp += warpEffect(noiseCoord + warp, fi, uSeed) * uWarpStrength;
      }
      
      // Apply warp as UV offset
      vec2 warpedUV = uv + warp * 0.15;
      warpedUV = clamp(warpedUV, 0.0, 1.0);
      
      // Sample texture at warped position
      vec3 texColor = texture2D(uTexture, warpedUV).rgb;
      
      // Oil slick lighting overlay based on noise
      float h = fbm(noiseCoord + warp, uSeed);
      float hx = fbm(noiseCoord + warp + vec2(0.05, 0.0), uSeed);
      float hy = fbm(noiseCoord + warp + vec2(0.0, 0.05), uSeed);
      
      // Fake normal from height field
      vec3 normal = normalize(vec3((h - hx) * 8.0, (h - hy) * 8.0, 1.0));
      
      // Light from top-left
      vec3 lightDir = normalize(vec3(0.5, 0.6, 1.0));
      float diffuse = max(dot(normal, lightDir), 0.0);
      
      // Specular highlight
      vec3 viewDir = vec3(0.0, 0.0, 1.0);
      vec3 reflectDir = reflect(-lightDir, normal);
      float spec = pow(max(dot(viewDir, reflectDir), 0.0), 24.0);
      
      // Lighting adjustment
      float lighting = 0.85 + diffuse * 0.2;
      float highlight = spec * 0.25;
      
      vec3 color = texColor * lighting + vec3(highlight);
      
      gl_FragColor = vec4(color, 1.0);
    }
  `;
  
  const program = createShaderProgram(gl, vertexShader, fragmentShader);
  gl.useProgram(program);
  
  const texture = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, texture);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, prev.width, prev.height, 0, gl.RGBA, gl.UNSIGNED_BYTE, prev.data);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  
  const vertices = new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]);
  const buffer = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
  gl.bufferData(gl.ARRAY_BUFFER, vertices, gl.STATIC_DRAW);
  
  const positionLoc = gl.getAttribLocation(program, 'position');
  gl.enableVertexAttribArray(positionLoc);
  gl.vertexAttribPointer(positionLoc, 2, gl.FLOAT, false, 0, 0);
  
  gl.uniform1i(gl.getUniformLocation(program, 'uTexture'), 0);
  gl.uniform2f(gl.getUniformLocation(program, 'uResolution'), ctx.width, ctx.height);
  gl.uniform1f(gl.getUniformLocation(program, 'uSeed'), seed);
  gl.uniform1i(gl.getUniformLocation(program, 'uDepth'), depth);
  gl.uniform1f(gl.getUniformLocation(program, 'uWarpStrength'), warpStrength);
  gl.uniform1f(gl.getUniformLocation(program, 'uPatternScale'), patternScale);
  
  gl.viewport(0, 0, ctx.width, ctx.height);
  gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  
  const pixels = new Uint8ClampedArray(ctx.width * ctx.height * 4);
  gl.readPixels(0, 0, ctx.width, ctx.height, gl.RGBA, gl.UNSIGNED_BYTE, pixels);
  
  const flipped = new Uint8ClampedArray(ctx.width * ctx.height * 4);
  for (let y = 0; y < ctx.height; y++) {
    for (let x = 0; x < ctx.width; x++) {
      const srcIdx = ((ctx.height - 1 - y) * ctx.width + x) * 4;
      const dstIdx = (y * ctx.width + x) * 4;
      flipped[dstIdx] = pixels[srcIdx];
      flipped[dstIdx + 1] = pixels[srcIdx + 1];
      flipped[dstIdx + 2] = pixels[srcIdx + 2];
      flipped[dstIdx + 3] = pixels[srcIdx + 3];
    }
  }
  
  gl.deleteTexture(texture);
  gl.deleteBuffer(buffer);
  gl.deleteProgram(program);
  
  return { width: ctx.width, height: ctx.height, data: flipped };
}

function fnTilde(ctx: FnContext, n: number): Image {
  const prev = getPrevImage(ctx);
  const out = createSolidImage(ctx.width, ctx.height, '#000000');
  
  const waveN = Math.max(1, n);
  
  for (let y = 0; y < ctx.height; y++) {
    const amplitude = Math.sin(y * 0.05) * waveN * 8;
    
    for (let x = 0; x < ctx.width; x++) {
      const srcX = Math.floor(((x + amplitude) % ctx.width + ctx.width) % ctx.width);
      
      const srcXR = Math.floor(((x + amplitude + waveN) % ctx.width + ctx.width) % ctx.width);
      const srcXB = Math.floor(((x + amplitude - waveN) % ctx.width + ctx.width) % ctx.width);
      
      const [rr] = getPixel(prev, srcXR, y);
      const [, gg] = getPixel(prev, srcX, y);
      const [, , bb] = getPixel(prev, srcXB, y);
      
      setPixel(out, x, y, rr, gg, bb);
    }
  }
  
  return out;
}

function fnHoles(ctx: FnContext, old: Image): Image {
  const prev = getPrevImage(ctx);
  const out = createSolidImage(ctx.width, ctx.height, '#000000');
  
  for (let y = 0; y < ctx.height; y++) {
    for (let x = 0; x < ctx.width; x++) {
      const [pr, pg, pb] = getPixel(prev, x, y);
      const [h, s, l] = rgbToHsl(pr, pg, pb);
      
      const midSat = s >= 0.35 && s <= 0.65;
      const midVal = l >= 0.35 && l <= 0.65;
      
      if (midSat || midVal) {
        const [or, og, ob] = getPixel(old, x, y);
        setPixel(out, x, y, or, og, ob);
      } else {
        setPixel(out, x, y, pr, pg, pb);
      }
    }
  }
  
  return out;
}

function wrapText(text: string, maxWidth: number): string[] {
  const lines: string[] = [];
  const paragraphs = text.split('\n');
  
  for (const para of paragraphs) {
    if (para.length === 0) {
      lines.push('');
      continue;
    }
    
    const words = para.split(' ');
    let currentLine = '';
    
    for (const word of words) {
      if (currentLine.length === 0) {
        currentLine = word;
      } else if (currentLine.length + 1 + word.length <= maxWidth) {
        currentLine += ' ' + word;
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

function generateCharacterRefLines(char: string, def: CharDef, charsPerLine: number): string[] {
  const lines: string[] = [];
  
  const argsStr = def.args.length > 0 ? ` [${def.args.map(a => a.type.name).join(', ')}]` : '';
  const header = `${char} (${def.number}) ${def.color} - ${def.functionName}${argsStr}`;
  lines.push(header);
  
  const docLines = wrapText('  ' + def.documentation, charsPerLine);
  lines.push(...docLines);
  
  for (let i = 0; i < def.args.length; i++) {
    const arg = def.args[i];
    const argLine = `  (${i + 1}) :${arg.type.name} -- ${arg.documentation}`;
    const argDocLines = wrapText(argLine, charsPerLine);
    lines.push(...argDocLines);
  }
  
  return lines;
}

function getPageChar(pageNum: number): string {
  if (pageNum <= 0) return '?';
  if (pageNum === 1) return '?';
  if (pageNum <= 26) return String.fromCharCode('A'.charCodeAt(0) + pageNum - 1);
  return '?';
}

function generateAllHelpPages(charsPerLine: number, linesPerPage: number, defs: Record<string, CharDef>): string[][] {
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
  
  const chars = Object.keys(defs).sort((a, b) => defs[a].number - defs[b].number);
  
  let currentPage: string[] = [];
  currentPage.push('=== CHARACTER REFERENCE ===');
  currentPage.push('');
  let linesUsed = 2;
  
  for (const char of chars) {
    const def = defs[char];
    const charLines = generateCharacterRefLines(char, def, charsPerLine);
    
    if (linesUsed + charLines.length + 1 > linesPerPage - 2) {
      pages.push(currentPage);
      currentPage = [];
      currentPage.push('=== CHARACTER REFERENCE (continued) ===');
      currentPage.push('');
      linesUsed = 2;
    }
    
    currentPage.push(...charLines);
    currentPage.push('');
    linesUsed += charLines.length + 1;
  }
  
  if (currentPage.length > 2) {
    pages.push(currentPage);
  }
  
  const totalPages = pages.length;
  for (let i = 0; i < pages.length; i++) {
    const pageNum = i + 1;
    const nextPageChar = getPageChar(pageNum + 1);
    pages[i].push('');
    if (pageNum < totalPages) {
      pages[i].push(`[Page ${pageNum}/${totalPages}, type '?${nextPageChar}' for next page]`);
    } else {
      pages[i].push(`[Page ${pageNum}/${totalPages}]`);
    }
  }
  
  return pages;
}

function generateIndexPage(numPages: number): string[] {
  const lines: string[] = [];
  lines.push('=== QQQLANG HELP INDEX ===');
  lines.push('');
  lines.push('Available pages:');
  lines.push('');
  lines.push('?? or ?A - Introduction to QQQLANG');
  
  for (let i = 2; i <= Math.min(numPages, 26); i++) {
    const char = String.fromCharCode('A'.charCodeAt(0) + i - 1);
    if (i === 2) {
      lines.push(`?${char} - Character reference`);
    } else {
      lines.push(`?${char} - Character reference (continued)`);
    }
  }
  
  lines.push('');
  lines.push('Enter a valid page code to view help.');
  lines.push('Invalid page codes show this index.');
  
  return lines;
}

function fnHelp(ctx: FnContext, pageArg: number): Image {
  // If pageArg is 42 (the '#' character), show image history
  if (pageArg === 42) {
    return fnImageHistory(ctx);
  }
  
  const out = createSolidImage(ctx.width, ctx.height, '#000000');
  
  const tempCanvas = document.createElement('canvas');
  tempCanvas.width = ctx.width;
  tempCanvas.height = ctx.height;
  const tempCtx = tempCanvas.getContext('2d')!;
  
  tempCtx.fillStyle = '#000000';
  tempCtx.fillRect(0, 0, ctx.width, ctx.height);
  
  const marginFraction = 0.03;
  const margin = Math.max(10, Math.floor(Math.min(ctx.width, ctx.height) * marginFraction));
  
  const baseFontSize = Math.min(ctx.width, ctx.height) * 0.025;
  const fontSize = Math.max(8, Math.min(16, baseFontSize));
  const lineHeight = Math.floor(fontSize * 1.4);
  
  tempCtx.font = `${fontSize}px monospace`;
  tempCtx.fillStyle = '#00FF00';
  
  const charWidth = tempCtx.measureText('M').width;
  const charsPerLine = Math.max(20, Math.floor((ctx.width - margin * 2) / charWidth));
  const linesPerPage = Math.max(5, Math.floor((ctx.height - margin * 2) / lineHeight));
  
  let page: number;
  if (pageArg === 58 || pageArg === 1) {
    page = 1;
  } else {
    page = pageArg;
  }
  
  const pages = generateAllHelpPages(charsPerLine, linesPerPage, characterDefs);
  
  let lines: string[];
  if (page >= 1 && page <= pages.length) {
    lines = pages[page - 1];
  } else {
    lines = generateIndexPage(pages.length);
  }
  
  let y = margin + fontSize;
  for (let i = 0; i < Math.min(lines.length, linesPerPage); i++) {
    tempCtx.fillText(lines[i], margin, y);
    y += lineHeight;
  }
  
  const imageData = tempCtx.getImageData(0, 0, ctx.width, ctx.height);
  out.data.set(imageData.data);
  
  return out;
}

export const characterDefs: Record<string, CharDef> = {
  'A': {
    color: '#78A10F',
    number: 1,
    fn: fnA,
    args: [],
    functionName: "flip-and-sphere",
    documentation: "Flips prev horizontally, then renders as texture on two 3D spheres with lighting."
  },
  
  'B': {
    color: '#8B4513',
    number: 2,
    fn: fnV,
    args: [{ type: COLOR, documentation: "Border tint color" }],
    functionName: "border",
    documentation: "Circular gradient darkening edges with color tint."
  },
  
  'C': {
    color: '#FF6B35',
    number: 3,
    fn: fnC,
    args: [{ type: INT, documentation: "Number of concentric circles" }],
    functionName: "concentric-hue",
    documentation: "Alternating original and hue-shifted concentric circles."
  },
  
  'D': {
    color: '#FF1493',
    number: 4,
    fn: fnD,
    args: [],
    functionName: "drip",
    documentation: "Metaball-based dripping water drops effect."
  },
  
  'E': {
    color: '#50C878',
    number: 5,
    fn: fnE,
    args: [],
    functionName: "emerald",
    documentation: "Renders reflective 3D emeralds in symmetric pattern."
  },
  
  'F': {
    color: '#FFD700',
    number: 6,
    fn: fnO,
    args: [{ type: INT, documentation: "FFT multiplier strength" }],
    functionName: "fft-overflow",
    documentation: "2D FFT with magnitude overflow and chromatic phase shifts."
  },
  
  'G': {
    color: '#9370DB',
    number: 7,
    fn: fnG,
    args: [{ type: INT, documentation: "Number of posterize colors" }],
    functionName: "grayscale-colorize",
    documentation: "Converts to grayscale then applies rainbow palette."
  },
  
  'H': {
    color: '#DC143C',
    number: 8,
    fn: fnH,
    args: [],
    functionName: "hourglass",
    documentation: "Hourglass gradient with bitwise color blending."
  },
  
  'I': {
    color: '#00FF7F',
    number: 9,
    fn: fnI,
    args: [],
    functionName: "invert-edges",
    documentation: "Inverts colors then adds Sobel edge detection."
  },
  
  'J': {
    color: '#FF8C00',
    number: 10,
    fn: fnF,
    args: [{ type: INT, documentation: "Fractal zoom depth" }],
    functionName: "julia-fractal",
    documentation: "Julia set fractal masking the previous image."
  },
  
  'K': {
    color: '#4B0082',
    number: 11,
    fn: fnK,
    args: [{ type: INT, documentation: "Number of kaleidoscope segments" }],
    functionName: "kaleidoscope",
    documentation: "N-way kaleidoscope effect with zoom."
  },
  
  'L': {
    color: '#20B2AA',
    number: 12,
    fn: fnL,
    args: [
      { type: INDEX, documentation: "Old image for tube texture" },
      { type: INT, documentation: "Rotation angle multiplier" }
    ],
    functionName: "lissajous",
    documentation: "3D Lissajous tube with textured surface."
  },
  
  'M': {
    color: '#FF69B4',
    number: 13,
    fn: fnM,
    args: [{ type: INT, documentation: "Pattern complexity seed" }],
    functionName: "moire",
    documentation: "Moiré interference pattern with color zones."
  },
  
  'N': {
    color: '#8A2BE2',
    number: 14,
    fn: fnN,
    args: [],
    functionName: "neon",
    documentation: "Neon glow effect on bright edges."
  },
  
  'O': {
    color: '#FF6347',
    number: 15,
    fn: fnOilSlick,
    args: [
      { type: INT, documentation: "Warp strength" },
      { type: INT, documentation: "Iteration depth" }
    ],
    functionName: "oil-slick",
    documentation: "Domain warping with iridescent lighting."
  },
  
  'P': {
    color: '#4682B4',
    number: 16,
    fn: fnP,
    args: [{ type: INT, documentation: "Pixel cell size" }],
    functionName: "diagonal-pixelate",
    documentation: "Pixelate with diagonal split using average/saturated colors."
  },
  
  'Q': {
    color: '#32CD32',
    number: 17,
    fn: fnQ,
    args: [],
    functionName: "prism",
    documentation: "Negative prism with diagonal inversion and mirroring."
  },
  
  'R': {
    color: '#DA70D6',
    number: 18,
    fn: fnR,
    args: [],
    functionName: "room",
    documentation: "3D room with textured walls, ceiling, and floor."
  },
  
  'S': {
    color: '#87CEEB',
    number: 19,
    fn: fnS,
    args: [
      { type: INDEX, documentation: "Old image for triangle interior" },
      { type: INT, documentation: "Fractal detail level (A-~)" }
    ],
    functionName: "sierpinski",
    documentation: "Sierpiński triangle fractal with color effects."
  },
  
  'T': {
    color: '#F0E68C',
    number: 20,
    fn: fnT,
    args: [{ type: INT, documentation: "Building height multiplier (A=short, ~=tall)" }],
    functionName: "city",
    documentation: "Grid of 3D buildings covering entire canvas, heights based on seed with multiplier."
  },
  
  'U': {
    color: '#DDA0DD',
    number: 21,
    fn: fnU,
    args: [{ type: INT, documentation: "Intensity and angle multiplier" }],
    functionName: "undertone",
    documentation: "HSL shift with three gradient directions for hue, saturation, and lightness."
  },
  
  'V': {
    color: '#40E0D0',
    number: 22,
    fn: fnB,
    args: [{ type: INDEX, documentation: "Old image to alternate with" }],
    functionName: "voronoi",
    documentation: "36 voronoi cells alternating between current and old image."
  },
  
  'W': {
    color: '#EE82EE',
    number: 23,
    fn: fnW,
    args: [{ type: INT, documentation: "Rotation multiplier (×20°)" }],
    functionName: "swirl",
    documentation: "Swirl distortion from center with quadratic falloff."
  },
  
  'X': {
    color: '#F5DEB3',
    number: 24,
    fn: fnX,
    args: [
      { type: Choice('★','●','■','▲','◆','♥','✦','⬡','✚','◐','☽','⚡','∞','☀','✿','⬢','◯','△','□','◇','♦','♣','♠','⬟','⬠','▽','◁','▷','⊕','⊗','⊛','⊚','▣','▤','▥','▦','▧','▨','▩','⬣','⬤','◉','◎','◈','◊','○','◌','◍','◢','◣','◤','◥','♯','♮','♩','♪','✶','✴','✳','✲','✱','✰','✯','✮'), documentation: "Unicode shape (cycles through 64 symbols)" },
      { type: COLOR, documentation: "Shape color" }
    ],
    functionName: "shape-overlay",
    documentation: "Draws unicode shape at center with gradient to average color."
  },
  
  'Y': {
    color: '#98FB98',
    number: 25,
    fn: fnY,
    args: [{ type: INT, documentation: "Number of radial sections" }],
    functionName: "radial-hue",
    documentation: "Radial sections with progressive hue rotation."
  },
  
  'Z': {
    color: '#AFEEEE',
    number: 26,
    fn: fnZ,
    args: [{ type: INT, documentation: "Blur strength multiplier (×4px)" }],
    functionName: "zoom-blur",
    documentation: "Radial motion blur from center with sharp center."
  },
  
  '0': {
    color: '#E6E6FA',
    number: 27,
    fn: fn0,
    args: [{ type: INDEX, documentation: "Old image to overlay blend" }],
    functionName: "overlay",
    documentation: "Overlay blend mode darkening darks and lightening lights."
  },
  
  '1': {
    color: '#FFA07A',
    number: 28,
    fn: fn1,
    args: [],
    functionName: "center-bar",
    documentation: "Middle third sharpened and boosted, rest desaturated."
  },
  
  '2': {
    color: '#98D8C8',
    number: 29,
    fn: fn2,
    args: [
      { type: INDEX, documentation: "Old image to extract third from" },
      { type: INT, documentation: "Old image third (1=left, 2=mid, 3=right, cycling)" },
      { type: INT, documentation: "Current image third to replace (1=left, 2=mid, 3=right, cycling)" }
    ],
    functionName: "third-stamp",
    documentation: "Replace a vertical third of current image with a third from old image."
  },
  
  '3': {
    color: '#F7DC6F',
    number: 30,
    fn: fn3,
    args: [],
    functionName: "triple-rotate",
    documentation: "Three vertical strips with different rotations."
  },
  
  '4': {
    color: '#BB8FCE',
    number: 31,
    fn: fn4,
    args: [],
    functionName: "quad-rotate",
    documentation: "Four quadrants each rotated 0°, 90°, 180°, 270°."
  },
  
  '5': {
    color: '#85C1E9',
    number: 32,
    fn: fn5,
    args: [{ type: INT, documentation: "Cell size multiplier" }],
    functionName: "triangular-split",
    documentation: "Triangular grid with hue shifts and lightness variation."
  },
  
  '6': {
    color: '#F1948A',
    number: 33,
    fn: fn6,
    args: [],
    functionName: "posterize",
    documentation: "Posterize to 4 levels per channel."
  },
  
  '7': {
    color: '#82E0AA',
    number: 34,
    fn: fn7,
    args: [],
    functionName: "chromatic",
    documentation: "Chromatic aberration with RGB channel shifts."
  },
  
  '8': {
    color: '#F8C471',
    number: 35,
    fn: fn8,
    args: [{ type: INT, documentation: "Distortion strength" }],
    functionName: "lemniscate",
    documentation: "Infinity-loop lemniscate distortion."
  },
  
  '9': {
    color: '#D7BDE2',
    number: 36,
    fn: fn9,
    args: [{ type: INDEX, documentation: "Old image to XOR with" }],
    functionName: "xor-blend",
    documentation: "XOR blend creating glitchy digital artifacts."
  },
  
  '<': {
    color: '#E74C3C',
    number: 37,
    fn: fnLessThan,
    args: [],
    functionName: "shift-left",
    documentation: "Horizontal shift 1/3 width left with wraparound."
  },
  
  '>': {
    color: '#3498DB',
    number: 38,
    fn: fnGreaterThan,
    args: [],
    functionName: "rotate-90",
    documentation: "Rotate 90 degrees clockwise."
  },
  
  '^': {
    color: '#2ECC71',
    number: 39,
    fn: fnCaret,
    args: [],
    functionName: "shift-up",
    documentation: "Vertical shift 1/3 height up with wraparound."
  },
  
  '!': {
    color: '#FF4500',
    number: 40,
    fn: fnAsterisk,
    args: [],
    functionName: "godrays",
    documentation: "Volumetric light scattering from center."
  },
  
  '"': {
    color: '#9932CC',
    number: 41,
    fn: fnDoubleQuote,
    args: [{ type: INT, documentation: "Number of horizontal bands" }],
    functionName: "band-transform",
    documentation: "Horizontal bands with alternating hue/saturation transforms."
  },
  
  '#': {
    color: '#228B22',
    number: 42,
    fn: (ctx: FnContext, old: Image) => cloneImage(old),
    args: [{ type: INDEX, documentation: "Old image index to insert" }],
    functionName: "insert",
    documentation: "Replaces current image with specified old image."
  },
  
  '$': {
    color: '#FFD700',
    number: 43,
    fn: fnDollar,
    args: [],
    functionName: "segment-hue-sort",
    documentation: "Color-based segmentation, then sorts pixels by hue within each segment."
  },
  
  '%': {
    color: '#8B0000',
    number: 44,
    fn: fnPercent,
    args: [{ type: INT, documentation: "Flip direction (even=horizontal, odd=vertical)" }],
    functionName: "flip",
    documentation: "Flips image horizontally or vertically based on argument parity."
  },
  
  '&': {
    color: '#4169E1',
    number: 45,
    fn: fnAmpersand,
    args: [{ type: Choice('ordered-5level','bayer-bw','threshold-bw','ordered-2bit','floyd-rgb','floyd-bw','atkinson-4level','atkinson-bw','stucki-6level','burkes','sierra','random-bw','cluster-2bit','bluenoise-bw','bayer2x2-2bit','noise-2bit'), documentation: "Dithering algorithm" }],
    functionName: "dither",
    documentation: "Apply one of 16 dithering algorithms, from subtle to aggressive 2-color modes."
  },
  
  "'": {
    color: '#FF1493',
    number: 46,
    fn: fnJ,
    args: [{ type: INDEX, documentation: "Old image to checkerboard with" }],
    functionName: "variable-checkerboard",
    documentation: "Checkerboard blend with increasing square size from corner to corner."
  },
  
  '(': {
    color: '#00CED1',
    number: 47,
    fn: fnOpenParen,
    args: [{ type: INT, documentation: "Pinch strength (÷10)" }],
    functionName: "pinch",
    documentation: "Pinch distortion toward center with brightening."
  },
  
  ')': {
    color: '#FF69B4',
    number: 48,
    fn: fnCloseParen,
    args: [{ type: INT, documentation: "Bulge strength (÷10)" }],
    functionName: "bulge",
    documentation: "Bulge distortion from center with darkening."
  },
  
  '*': {
    color: '#FFD700',
    number: 49,
    fn: fnExclaim,
    args: [],
    functionName: "fur",
    documentation: "Fur/hair strands growing from pixels based on hue and noise."
  },
  
  '+': {
    color: '#32CD32',
    number: 50,
    fn: fnPlus,
    args: [],
    functionName: "zoom",
    documentation: "Zoom in 1.2× from center."
  },
  
  ',': {
    color: '#BA55D3',
    number: 51,
    fn: fnComma,
    args: [{ type: COLOR, documentation: "Stipple dot color" }],
    functionName: "stipple",
    documentation: "Stipple dots at luminance-based positions."
  },
  
  '-': {
    color: '#708090',
    number: 52,
    fn: fnMinus,
    args: [],
    functionName: "scanlines",
    documentation: "CRT scanline effect with darkening and displacement."
  },
  
  '.': {
    color: '#20B2AA',
    number: 53,
    fn: fnDot,
    args: [{ type: INT, documentation: "Dot radius base (mod 8 + 2)" }],
    functionName: "pointillism",
    documentation: "Pointillism effect with saturated circular dots."
  },
  
  '/': {
    color: '#CD853F',
    number: 54,
    fn: fnSlash,
    args: [
      { type: INDEX, documentation: "Old image source" },
      { type: INT, documentation: "X position (normalized 0-1)" },
      { type: INT, documentation: "Y position (normalized 0-1)" },
      { type: INT, documentation: "Circle size (normalized 0-1)" },
      { type: INT, documentation: "Blend mode (mod 16: normal, xor, nand, and, or, multiply, screen, overlay, darken, lighten, diff, excl, add, sub, hard, soft)" }
    ],
    functionName: "circle-stamp",
    documentation: "Stamp circular region from old image center onto current."
  },
  
  ':': {
    color: '#6B8E23',
    number: 55,
    fn: fnColon,
    args: [{ type: INDEX, documentation: "Old image to show in porthole" }],
    functionName: "porthole",
    documentation: "Circular window showing old image with current image outside."
  },
  
  ';': {
    color: '#DB7093',
    number: 56,
    fn: fnSemicolon,
    args: [],
    functionName: "semicircle-reflect",
    documentation: "Top semicircle preserved, bottom reflected with wave distortion."
  },
  
  '=': {
    color: '#5F9EA0',
    number: 57,
    fn: fnEquals,
    args: [{ type: INT, documentation: "Stripe height in pixels" }],
    functionName: "shifted-stripes",
    documentation: "Horizontal stripes with alternating shifts."
  },
  
  '?': {
    color: '#D2691E',
    number: 58,
    fn: fnHelp,
    args: [{ type: INT, documentation: "Page number (A=intro, B+=reference, #=history)" }],
    functionName: "help",
    documentation: "Display help text or image history table."
  },
  
  '@': {
    color: '#7B68EE',
    number: 59,
    fn: fnHoles,
    args: [{ type: INDEX, documentation: "Old image to reveal in midtones" }],
    functionName: "midtone-reveal",
    documentation: "Show old image where saturation/lightness is mid-range."
  },
  
  '[': {
    color: '#48D1CC',
    number: 60,
    fn: fnOpenBracket,
    args: [],
    functionName: "rotate-left",
    documentation: "Rotate 20° counter-clockwise."
  },
  
  '\\': {
    color: '#C71585',
    number: 61,
    fn: fnBackslash,
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
      { type: INT, documentation: "Blend mode (mod 16: normal, xor, nand, and, or, multiply, screen, overlay, darken, lighten, diff, excl, add, sub, hard, soft)" }
    ],
    functionName: "composite",
    documentation: "Composite transformed region from old image onto current."
  },
  
  ']': {
    color: '#00FA9A',
    number: 62,
    fn: fnCloseBracket,
    args: [],
    functionName: "left-half-offset",
    documentation: "Shift left half vertically by 20% with wraparound."
  },
  
  '_': {
    color: '#FF7F50',
    number: 63,
    fn: fnBlend,
    args: [{ type: Choice('multiply','screen','overlay','darken','lighten','dodge','burn','hardlight','softlight','difference','exclusion','add','subtract','xor','and','or','nand','nor','xnor','average','divide','grain-extract','grain-merge','vivid','linear','pin','hardmix','hue','saturation','color','luminosity','replace-dark-third','replace-mid-third','replace-light-third'), documentation: "Blend mode" }],
    functionName: "blend",
    documentation: "Blend current with previous using specified mode."
  },
  
  '`': {
    color: '#6495ED',
    number: 64,
    fn: fnBacktick,
    args: [{ type: INT, documentation: "Iterations and smear strength" }],
    functionName: "gradient-smear",
    documentation: "Iterative gradient-based pixel smearing creating directional streaks."
  },
  
  '{': {
    color: '#DC143C',
    number: 65,
    fn: fnSkewLeft,
    args: [],
    functionName: "skew-left",
    documentation: "Skew 20° left with wraparound (top left, bottom right)."
  },
  
  '|': {
    color: '#00BFFF',
    number: 66,
    fn: fnPipe,
    args: [{ type: INDEX, documentation: "Old image for right half" }],
    functionName: "vertical-split",
    documentation: "Vertical split with wavy blend zone using multiple blend modes."
  },
  
  '}': {
    color: '#9400D3',
    number: 67,
    fn: fnSkewRight,
    args: [],
    functionName: "skew-right",
    documentation: "Skew 20° right with wraparound (top right, bottom left)."
  },
  
  '~': {
    color: '#FF6347',
    number: 68,
    fn: fnTilde,
    args: [{ type: INT, documentation: "Wave amplitude and chromatic shift" }],
    functionName: "wave-chromatic",
    documentation: "Horizontal wave distortion with chromatic aberration."
  },
};
