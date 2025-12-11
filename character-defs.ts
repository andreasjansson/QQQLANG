import * as THREE from 'three';
import { EffectComposer } from 'three/examples/jsm/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/examples/jsm/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/examples/jsm/postprocessing/UnrealBloomPass.js';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';

export interface Image {
  width: number;
  height: number;
  data: Uint8ClampedArray;
}

export interface FnContext {
  width: number;
  height: number;
  images: Image[];
  currentIndex: number;
}

export interface CharDef {
  color: string;
  number: number;
  fn: (ctx: FnContext, ...args: any[]) => Image;
  arity: number;
  argTypes: ('int' | 'color')[];
  functionName: string;
  documentation: string;
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

function getOldImage(ctx: FnContext, j: number): Image {
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
      vec3 bg = texture2D(texture, uv).rgb;
      
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

function fnB(ctx: FnContext, j: number): Image {
  const prev = getPrevImage(ctx);
  const old = getOldImage(ctx, j);
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

function fnJ(ctx: FnContext, j: number): Image {
  const prev = getPrevImage(ctx);
  const old = getOldImage(ctx, j);
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

function fnL(ctx: FnContext, j: number, rot: number): Image {
  const prev = getPrevImage(ctx);
  const old = getOldImage(ctx, j);
  const gl = initWebGL(ctx.width, ctx.height);
  
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
  const out = createSolidImage(ctx.width, ctx.height, '#000000');
  
  const strength = Math.max(0.0001, Math.min(0.0002 + (n - 1) * 0.05, 8));
  const cx = ctx.width / 2;
  const cy = ctx.height / 2;
  
  for (let y = 0; y < ctx.height; y++) {
    for (let x = 0; x < ctx.width; x++) {
      const dx = x - cx;
      const dy = y - cy;
      
      const normX = dx / cx;
      const normY = dy / cy;
      const normR = Math.min(1, Math.sqrt(normX * normX + normY * normY));
      
      if (normR < 0.001) {
        const [pr, pg, pb] = getPixel(prev, Math.floor(cx), Math.floor(cy));
        setPixel(out, x, y, pr, pg, pb);
        continue;
      }
      
      const falloff = 1 - normR;
      const factor = 1 + (strength - 1) * falloff;
      
      const sx = cx + dx * factor;
      const sy = cy + dy * factor;
      
      const [pr, pg, pb] = getPixel(prev, Math.floor(sx), Math.floor(sy));
      setPixel(out, x, y, pr, pg, pb);
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
          texCoord = vec2(1.0 - (hit.z + roomDepth) / (ro.z + roomDepth), hit.y / roomSize * 0.5 + 0.5);
        }
      }
      
      // Floor (y = -roomSize)
      t = intersectPlane(ro, rd, vec3(0.0, 1.0, 0.0), roomSize);
      if (t > 0.0 && t < tMin) {
        vec3 hit = ro + rd * t;
        if (hit.z > -roomDepth && hit.z < ro.z && abs(hit.x) < roomSize) {
          tMin = t;
          hitNormal = vec3(0.0, 1.0, 0.0);
          texCoord = vec2(hit.x / roomSize * 0.5 + 0.5, (hit.z + roomDepth) / (ro.z + roomDepth));
        }
      }
      
      // Ceiling (y = roomSize)
      t = intersectPlane(ro, rd, vec3(0.0, -1.0, 0.0), roomSize);
      if (t > 0.0 && t < tMin) {
        vec3 hit = ro + rd * t;
        if (hit.z > -roomDepth && hit.z < ro.z && abs(hit.x) < roomSize) {
          tMin = t;
          hitNormal = vec3(0.0, -1.0, 0.0);
          texCoord = vec2(hit.x / roomSize * 0.5 + 0.5, 1.0 - (hit.z + roomDepth) / (ro.z + roomDepth));
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
  const gl = initWebGL(ctx.width, ctx.height);
  
  const numCubes = Math.max(1, n + 1);
  const seed = ctx.images.length * 137.5 + n * 17.0;
  const hash = (i: number) => {
    const x = Math.sin(i + seed) * 43758.5453;
    return x - Math.floor(x);
  };
  
  interface CubeData {
    cx: number; cy: number; hw: number; hh: number; depth: number;
  }
  const cubesData: CubeData[] = [];
  for (let i = 0; i < numCubes; i++) {
    cubesData.push({
      cx: hash(i * 127.1),
      cy: hash(i * 311.7),
      hw: 0.03 + hash(i * 74.3) * 0.08,
      hh: 0.025 + hash(i * 183.9) * 0.06,
      depth: 0.1 + hash(i * 271.3) * 0.2
    });
  }
  
  const lightDirX = -0.3;
  const lightDirY = -0.4;
  
  const vertexShader = `
    attribute vec3 aPosition;
    attribute vec3 aNormal;
    attribute vec2 aTexCoord;
    
    uniform mat4 uProjection;
    uniform mat4 uView;
    uniform mat4 uModel;
    
    varying vec3 vNormal;
    varying vec2 vTexCoord;
    varying vec3 vWorldPos;
    
    void main() {
      vec4 worldPos = uModel * vec4(aPosition, 1.0);
      vWorldPos = worldPos.xyz;
      vNormal = mat3(uModel) * aNormal;
      vTexCoord = aTexCoord;
      gl_Position = uProjection * uView * worldPos;
    }
  `;
  
  const fragmentShader = `
    precision highp float;
    
    uniform sampler2D uTexture;
    uniform vec3 uLightDir;
    
    varying vec3 vNormal;
    varying vec2 vTexCoord;
    varying vec3 vWorldPos;
    
    void main() {
      vec3 normal = normalize(vNormal);
      vec3 lightDir = normalize(uLightDir);
      
      float ambient = 0.5;
      float diffuse = max(dot(normal, lightDir), 0.0) * 0.5;
      float lighting = ambient + diffuse;
      
      vec3 color = texture2D(uTexture, vTexCoord).rgb * lighting;
      gl_FragColor = vec4(color, 1.0);
    }
  `;
  
  const program = createShaderProgram(gl, vertexShader, fragmentShader);
  
  const texture = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, texture);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, prev.width, prev.height, 0, gl.RGBA, gl.UNSIGNED_BYTE, prev.data);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  
  function createBox(cx: number, cy: number, hw: number, hh: number, depth: number): { vertices: number[], normals: number[], texCoords: number[] } {
    const x0 = cx - hw, x1 = cx + hw;
    const y0 = cy - hh, y1 = cy + hh;
    const z0 = 0, z1 = depth;
    
    const u0 = cx - hw, u1 = cx + hw;
    const v0 = cy - hh, v1 = cy + hh;
    
    const vertices: number[] = [];
    const normals: number[] = [];
    const texCoords: number[] = [];
    
    // Top face - textured with image at this position
    vertices.push(x0,y0,z1, x1,y0,z1, x1,y1,z1, x0,y0,z1, x1,y1,z1, x0,y1,z1);
    for(let i=0;i<6;i++) normals.push(0,0,1);
    texCoords.push(u0,v0, u1,v0, u1,v1, u0,v0, u1,v1, u0,v1);
    
    // Right face (+x)
    vertices.push(x1,y0,z0, x1,y0,z1, x1,y1,z1, x1,y0,z0, x1,y1,z1, x1,y1,z0);
    for(let i=0;i<6;i++) normals.push(1,0,0);
    texCoords.push(u1,v0, u1,v0, u1,v1, u1,v0, u1,v1, u1,v1);
    
    // Left face (-x)
    vertices.push(x0,y0,z1, x0,y0,z0, x0,y1,z0, x0,y0,z1, x0,y1,z0, x0,y1,z1);
    for(let i=0;i<6;i++) normals.push(-1,0,0);
    texCoords.push(u0,v0, u0,v0, u0,v1, u0,v0, u0,v1, u0,v1);
    
    // Front face (+y)
    vertices.push(x0,y1,z1, x1,y1,z1, x1,y1,z0, x0,y1,z1, x1,y1,z0, x0,y1,z0);
    for(let i=0;i<6;i++) normals.push(0,1,0);
    texCoords.push(u0,v1, u1,v1, u1,v1, u0,v1, u1,v1, u0,v1);
    
    // Back face (-y)
    vertices.push(x0,y0,z0, x1,y0,z0, x1,y0,z1, x0,y0,z0, x1,y0,z1, x0,y0,z1);
    for(let i=0;i<6;i++) normals.push(0,-1,0);
    texCoords.push(u0,v0, u1,v0, u1,v0, u0,v0, u1,v0, u0,v0);
    
    return { vertices, normals, texCoords };
  }
  
  const allVertices: number[] = [];
  const allNormals: number[] = [];
  const allTexCoords: number[] = [];
  
  for (const c of cubesData) {
    const box = createBox(c.cx, c.cy, c.hw, c.hh, c.depth);
    allVertices.push(...box.vertices);
    allNormals.push(...box.normals);
    allTexCoords.push(...box.texCoords);
  }
  
  gl.useProgram(program);
  gl.disable(gl.DEPTH_TEST);
  
  const bgVertices = new Float32Array([0,0,0, 1,0,0, 1,1,0, 0,0,0, 1,1,0, 0,1,0]);
  const bgNormals = new Float32Array([0,0,1, 0,0,1, 0,0,1, 0,0,1, 0,0,1, 0,0,1]);
  const bgTexCoords = new Float32Array([0,0, 1,0, 1,1, 0,0, 1,1, 0,1]);
  
  const posLoc = gl.getAttribLocation(program, 'aPosition');
  const normLoc = gl.getAttribLocation(program, 'aNormal');
  const texLoc = gl.getAttribLocation(program, 'aTexCoord');
  
  const identity = new Float32Array([1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1]);
  const ortho = new Float32Array([2,0,0,0, 0,2,0,0, 0,0,-1,0, -1,-1,0,1]);
  
  gl.uniformMatrix4fv(gl.getUniformLocation(program, 'uProjection'), false, ortho);
  gl.uniformMatrix4fv(gl.getUniformLocation(program, 'uView'), false, identity);
  gl.uniformMatrix4fv(gl.getUniformLocation(program, 'uModel'), false, identity);
  gl.uniform3f(gl.getUniformLocation(program, 'uLightDir'), lightDirX, lightDirY, 1.0);
  gl.uniform1i(gl.getUniformLocation(program, 'uTexture'), 0);
  
  const bgPosBuf = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, bgPosBuf);
  gl.bufferData(gl.ARRAY_BUFFER, bgVertices, gl.STATIC_DRAW);
  gl.enableVertexAttribArray(posLoc);
  gl.vertexAttribPointer(posLoc, 3, gl.FLOAT, false, 0, 0);
  
  const bgNormBuf = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, bgNormBuf);
  gl.bufferData(gl.ARRAY_BUFFER, bgNormals, gl.STATIC_DRAW);
  gl.enableVertexAttribArray(normLoc);
  gl.vertexAttribPointer(normLoc, 3, gl.FLOAT, false, 0, 0);
  
  const bgTexBuf = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, bgTexBuf);
  gl.bufferData(gl.ARRAY_BUFFER, bgTexCoords, gl.STATIC_DRAW);
  gl.enableVertexAttribArray(texLoc);
  gl.vertexAttribPointer(texLoc, 2, gl.FLOAT, false, 0, 0);
  
  gl.viewport(0, 0, ctx.width, ctx.height);
  gl.clearColor(0, 0, 0, 1);
  gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
  gl.drawArrays(gl.TRIANGLES, 0, 6);
  
  gl.enable(gl.DEPTH_TEST);
  gl.depthFunc(gl.LESS);
  gl.clear(gl.DEPTH_BUFFER_BIT);
  
  const aspect = ctx.width / ctx.height;
  const fov = Math.PI / 2.5;
  const near = 0.1, far = 10.0;
  const f = 1.0 / Math.tan(fov / 2);
  const perspective = new Float32Array([
    f/aspect, 0, 0, 0,
    0, f, 0, 0,
    0, 0, (far+near)/(near-far), -1,
    0, 0, (2*far*near)/(near-far), 0
  ]);
  
  const camZ = 0.5 / Math.tan(fov / 2);
  const view = new Float32Array([
    1, 0, 0, 0,
    0, 1, 0, 0,
    0, 0, 1, 0,
    -0.5, -0.5, -camZ, 1
  ]);
  
  gl.uniformMatrix4fv(gl.getUniformLocation(program, 'uProjection'), false, perspective);
  gl.uniformMatrix4fv(gl.getUniformLocation(program, 'uView'), false, view);
  
  const posBuf = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, posBuf);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(allVertices), gl.STATIC_DRAW);
  gl.enableVertexAttribArray(posLoc);
  gl.vertexAttribPointer(posLoc, 3, gl.FLOAT, false, 0, 0);
  
  const normBuf = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, normBuf);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(allNormals), gl.STATIC_DRAW);
  gl.enableVertexAttribArray(normLoc);
  gl.vertexAttribPointer(normLoc, 3, gl.FLOAT, false, 0, 0);
  
  const texBuf = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, texBuf);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(allTexCoords), gl.STATIC_DRAW);
  gl.enableVertexAttribArray(texLoc);
  gl.vertexAttribPointer(texLoc, 2, gl.FLOAT, false, 0, 0);
  
  gl.drawArrays(gl.TRIANGLES, 0, allVertices.length / 3);
  
  const pixels = new Uint8ClampedArray(ctx.width * ctx.height * 4);
  gl.readPixels(0, 0, ctx.width, ctx.height, gl.RGBA, gl.UNSIGNED_BYTE, pixels);
  
  const flipped = new Uint8ClampedArray(ctx.width * ctx.height * 4);
  for (let y = 0; y < ctx.height; y++) {
    for (let x = 0; x < ctx.width; x++) {
      const srcIdx = (y * ctx.width + x) * 4;
      const dstIdx = (y * ctx.width + x) * 4;
      flipped[dstIdx] = pixels[srcIdx];
      flipped[dstIdx + 1] = pixels[srcIdx + 1];
      flipped[dstIdx + 2] = pixels[srcIdx + 2];
      flipped[dstIdx + 3] = pixels[srcIdx + 3];
    }
  }
  
  gl.disableVertexAttribArray(posLoc);
  gl.disableVertexAttribArray(normLoc);
  gl.disableVertexAttribArray(texLoc);
  gl.disable(gl.DEPTH_TEST);
  
  gl.deleteTexture(texture);
  gl.deleteBuffer(bgPosBuf);
  gl.deleteBuffer(bgNormBuf);
  gl.deleteBuffer(bgTexBuf);
  gl.deleteBuffer(posBuf);
  gl.deleteBuffer(normBuf);
  gl.deleteBuffer(texBuf);
  gl.deleteProgram(program);
  
  return { width: ctx.width, height: ctx.height, data: flipped };
}

function fnU(ctx: FnContext, n: number): Image {
  const prev = getPrevImage(ctx);
  const out = createSolidImage(ctx.width, ctx.height, '#000000');
  
  const amplitude = n * 5;
  const frequency = Math.max(1, n);
  
  for (let y = 0; y < ctx.height; y++) {
    let rowLuminance = 0;
    for (let x = 0; x < ctx.width; x++) {
      const [r, g, b] = getPixel(prev, x, y);
      rowLuminance += r * 0.299 + g * 0.587 + b * 0.114;
    }
    rowLuminance /= ctx.width;
    const phase = (rowLuminance / 255) * Math.PI * 2;
    
    for (let x = 0; x < ctx.width; x++) {
      const wave = Math.sin((y / ctx.height) * Math.PI * 2 * frequency + phase) * amplitude;
      const sx = ((x + wave) % ctx.width + ctx.width) % ctx.width;
      const [r, g, b] = getPixel(prev, Math.floor(sx), y);
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

function fnX(ctx: FnContext, k: number, c: string): Image {
  const prev = getPrevImage(ctx);
  const out = cloneImage(prev);
  const [cr, cg, cb] = hexToRgb(c);
  
  const shapes = [
    '★', '●', '■', '▲', '◆', '♥', '✦', '⬡', '✚', '◐', '☽', '⚡', '∞', '☀', '✿', '⬢',
    '◯', '△', '□', '◇', '♦', '♣', '♠', '⬟', '⬠', '▽', '◁', '▷', '⊕', '⊗', '⊛', '⊚',
    '▣', '▤', '▥', '▦', '▧', '▨', '▩', '⬣', '⬤', '◉', '◎', '◈', '◊', '○', '◌', '◍',
    '◢', '◣', '◤', '◥', '♯', '♮', '♩', '♪', '✶', '✴', '✳', '✲', '✱', '✰', '✯', '✮'
  ];
  
  const shapeIdx = Math.abs(k) % 64;
  const symbol = shapes[shapeIdx];
  
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
            const [r, g, b] = getPixel(temp, x, y);
            setPixel(out, outX, outY, r, g, b);
          }
        }
      }
    }
  }
  
  return out;
}

function fn0(ctx: FnContext, j: number): Image {
  const prev = getPrevImage(ctx);
  const old = getOldImage(ctx, j);
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

function fn2(ctx: FnContext): Image {
  const prev = getPrevImage(ctx);
  const out = createSolidImage(ctx.width, ctx.height, '#000000');
  
  const len = ctx.images.length;
  const img3 = len >= 3 ? ctx.images[len - 3] : (len >= 1 ? ctx.images[0] : prev);
  const img2 = len >= 2 ? ctx.images[len - 2] : prev;
  const img1 = prev;
  
  const third1 = Math.floor(ctx.width / 3);
  const third2 = Math.floor(ctx.width * 2 / 3);
  
  for (let y = 0; y < ctx.height; y++) {
    for (let x = 0; x < ctx.width; x++) {
      let src: Image;
      if (x < third1) {
        src = img3;
      } else if (x < third2) {
        src = img2;
      } else {
        src = img1;
      }
      const [r, g, b] = getPixel(src, x, y);
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

function fn9(ctx: FnContext, j: number): Image {
  const prev = getPrevImage(ctx);
  const old = getOldImage(ctx, j);
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

function fnExclaim(ctx: FnContext, n: number): Image {
  const prev = getPrevImage(ctx);
  const out = cloneImage(prev);
  const opacity = 0.3;
  const factor = n + 17;
  
  for (let y = 0; y < ctx.height; y++) {
    for (let x = 0; x < ctx.width; x++) {
      const brightness = ((x * factor) ^ (y * 31) ^ (x * y)) % 256;
      const idx = (y * ctx.width + x) * 4;
      out.data[idx] = Math.round(out.data[idx] * (1 - opacity) + brightness * opacity);
      out.data[idx + 1] = Math.round(out.data[idx + 1] * (1 - opacity) + brightness * opacity);
      out.data[idx + 2] = Math.round(out.data[idx + 2] * (1 - opacity) + brightness * opacity);
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
  
  const cellSize = Math.max(1, n + 1);
  const corners = [
    getPixel(prev, 0, 0),
    getPixel(prev, ctx.width - 1, 0),
    getPixel(prev, 0, ctx.height - 1),
    getPixel(prev, ctx.width - 1, ctx.height - 1)
  ];
  const palette = corners.map(([r, g, b]) => [r, g, b] as [number, number, number]);
  
  const findClosest = (r: number, g: number, b: number): [number, number, number] => {
    let minDist = Infinity;
    let closest = palette[0];
    for (const [pr, pg, pb] of palette) {
      const dist = (r - pr) ** 2 + (g - pg) ** 2 + (b - pb) ** 2;
      if (dist < minDist) {
        minDist = dist;
        closest = [pr, pg, pb];
      }
    }
    return closest;
  };
  
  const tempData = new Float32Array(ctx.width * ctx.height * 3);
  for (let y = 0; y < ctx.height; y++) {
    for (let x = 0; x < ctx.width; x++) {
      const [r, g, b] = getPixel(prev, x, y);
      const idx = (y * ctx.width + x) * 3;
      tempData[idx] = r;
      tempData[idx + 1] = g;
      tempData[idx + 2] = b;
    }
  }
  
  for (let y = 0; y < ctx.height; y += cellSize) {
    for (let x = 0; x < ctx.width; x += cellSize) {
      const idx = (y * ctx.width + x) * 3;
      const oldR = tempData[idx];
      const oldG = tempData[idx + 1];
      const oldB = tempData[idx + 2];
      
      const [newR, newG, newB] = findClosest(oldR, oldG, oldB);
      
      for (let cy = 0; cy < cellSize && y + cy < ctx.height; cy++) {
        for (let cx = 0; cx < cellSize && x + cx < ctx.width; cx++) {
          setPixel(out, x + cx, y + cy, newR, newG, newB);
        }
      }
      
      const errR = oldR - newR;
      const errG = oldG - newG;
      const errB = oldB - newB;
      
      const distribute = (dx: number, dy: number, factor: number) => {
        const nx = x + dx * cellSize;
        const ny = y + dy * cellSize;
        if (nx >= 0 && nx < ctx.width && ny >= 0 && ny < ctx.height) {
          const nidx = (ny * ctx.width + nx) * 3;
          tempData[nidx] += errR * factor;
          tempData[nidx + 1] += errG * factor;
          tempData[nidx + 2] += errB * factor;
        }
      };
      
      distribute(1, 0, 7 / 16);
      distribute(-1, 1, 3 / 16);
      distribute(0, 1, 5 / 16);
      distribute(1, 1, 1 / 16);
    }
  }
  
  return out;
}

function fnAmpersand(ctx: FnContext, j: number): Image {
  const prev = getPrevImage(ctx);
  const old = getOldImage(ctx, j);
  const out = createSolidImage(ctx.width, ctx.height, '#000000');
  
  for (let y = 0; y < ctx.height; y++) {
    for (let x = 0; x < ctx.width; x++) {
      const [pr, pg, pb] = getPixel(prev, x, y);
      const [or, og, ob] = getPixel(old, x, y);
      
      const luminance = pr * 0.299 + pg * 0.587 + pb * 0.114;
      const [oh, os, ol] = rgbToHsl(or, og, ob);
      const [nr, ng, nb] = hslToRgb(oh, os, luminance / 255);
      
      setPixel(out, x, y, nr, ng, nb);
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

function fnAsterisk(ctx: FnContext, n: number): Image {
  const prev = getPrevImage(ctx);
  const out = cloneImage(prev);
  
  const numRays = Math.max(2, n);
  const cx = ctx.width / 2;
  const cy = ctx.height / 2;
  const maxR = Math.sqrt(cx * cx + cy * cy);
  const opacity = 0.6;
  
  for (let y = 0; y < ctx.height; y++) {
    for (let x = 0; x < ctx.width; x++) {
      const dx = x - cx;
      const dy = y - cy;
      let angle = Math.atan2(dy, dx);
      if (angle < 0) angle += Math.PI * 2;
      
      const rayAngle = (Math.PI * 2) / numRays;
      const nearestRay = Math.round(angle / rayAngle) * rayAngle;
      const angleDiff = Math.abs(angle - nearestRay);
      
      const rayWidth = 0.05;
      if (angleDiff < rayWidth) {
        const softness = 1 - angleDiff / rayWidth;
        const sampleX = Math.floor(cx + Math.cos(nearestRay) * (ctx.width / numRays));
        const [sr, sg, sb] = getPixel(prev, sampleX, Math.floor(cy));
        
        const idx = (y * ctx.width + x) * 4;
        const blend = softness * opacity;
        out.data[idx] = Math.round(out.data[idx] * (1 - blend) + sr * blend);
        out.data[idx + 1] = Math.round(out.data[idx + 1] * (1 - blend) + sg * blend);
        out.data[idx + 2] = Math.round(out.data[idx + 2] * (1 - blend) + sb * blend);
      }
    }
  }
  
  return out;
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

function fnMinus(ctx: FnContext, n: number): Image {
  const prev = getPrevImage(ctx);
  const out = createSolidImage(ctx.width, ctx.height, '#000000');
  const spacing = Math.max(2, n);
  
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

function fnSlash(ctx: FnContext, c: string): Image {
  const prev = getPrevImage(ctx);
  const out = cloneImage(prev);
  const [cr, cg, cb] = hexToRgb(c);
  
  for (let y = 0; y < ctx.height; y++) {
    for (let x = 0; x < ctx.width; x++) {
      const [r, g, b] = getPixel(prev, x, y);
      const luminance = (r * 0.299 + g * 0.587 + b * 0.114) / 255;
      const spacing = Math.max(2, Math.floor(2 + luminance * 15));
      
      const diag = (x + y) % spacing;
      if (diag === 0) {
        setPixel(out, x, y, cr, cg, cb);
      }
    }
  }
  
  return out;
}

function fnColon(ctx: FnContext, n: number): Image {
  const prev = getPrevImage(ctx);
  const out = createSolidImage(ctx.width, ctx.height, '#000000');
  
  const numCircles = Math.max(1, Math.min(n, 10));
  const circleRadius = Math.min(ctx.width / (numCircles * 3), ctx.height / 4);
  const blurRadius = 5;
  
  const blurred = createSolidImage(ctx.width, ctx.height, '#000000');
  for (let y = 0; y < ctx.height; y++) {
    for (let x = 0; x < ctx.width; x++) {
      let sr = 0, sg = 0, sb = 0, count = 0;
      for (let ky = -blurRadius; ky <= blurRadius; ky++) {
        for (let kx = -blurRadius; kx <= blurRadius; kx++) {
          const [pr, pg, pb] = getPixel(prev, x + kx, y + ky);
          sr += pr;
          sg += pg;
          sb += pb;
          count++;
        }
      }
      setPixel(blurred, x, y, Math.round(sr / count), Math.round(sg / count), Math.round(sb / count));
    }
  }
  
  const circleCenters: [number, number][] = [];
  for (let i = 0; i < numCircles; i++) {
    const cx = Math.floor((i + 0.5) * ctx.width / numCircles);
    const cy = Math.floor(ctx.height / 2);
    circleCenters.push([cx, cy]);
  }
  
  for (let y = 0; y < ctx.height; y++) {
    for (let x = 0; x < ctx.width; x++) {
      let inCircle = false;
      for (const [cx, cy] of circleCenters) {
        const dx = x - cx;
        const dy = y - cy;
        if (dx * dx + dy * dy < circleRadius * circleRadius) {
          const srcX = cx + dx * 0.5;
          const srcY = cy + dy * 0.5;
          const [r, g, b] = getPixel(prev, Math.floor(srcX), Math.floor(srcY));
          setPixel(out, x, y, r, g, b);
          inCircle = true;
          break;
        }
      }
      if (!inCircle) {
        const [r, g, b] = getPixel(blurred, x, y);
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

function fnTornLeft(ctx: FnContext, j: number): Image {
  const prev = getPrevImage(ctx);
  const old = getOldImage(ctx, j);
  const out = createSolidImage(ctx.width, ctx.height, '#000000');
  
  const boundary = ctx.width / 3;
  
  for (let y = 0; y < ctx.height; y++) {
    const tear = boundary + Math.sin(y * 0.3) * 20 + Math.sin(y * 0.7) * 10;
    
    for (let x = 0; x < ctx.width; x++) {
      if (x < tear) {
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

function fnBackslash(ctx: FnContext, c: string): Image {
  const prev = getPrevImage(ctx);
  const out = cloneImage(prev);
  const [cr, cg, cb] = hexToRgb(c);
  
  for (let y = 0; y < ctx.height; y++) {
    for (let x = 0; x < ctx.width; x++) {
      const [r, g, b] = getPixel(prev, x, y);
      const luminance = (r * 0.299 + g * 0.587 + b * 0.114) / 255;
      const spacing = Math.max(2, Math.floor(2 + luminance * 15));
      
      const diag = ((ctx.width - 1 - x) + y) % spacing;
      if (diag === 0) {
        setPixel(out, x, y, cr, cg, cb);
      }
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

function fnTornRight(ctx: FnContext, j: number): Image {
  const prev = getPrevImage(ctx);
  const old = getOldImage(ctx, j);
  const out = createSolidImage(ctx.width, ctx.height, '#000000');
  
  const boundary = ctx.width * 2 / 3;
  
  for (let y = 0; y < ctx.height; y++) {
    const tear = boundary + Math.sin(y * 0.3) * 20 + Math.sin(y * 0.7) * 10;
    
    for (let x = 0; x < ctx.width; x++) {
      if (x > tear) {
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

function fnUnderscore(ctx: FnContext, n: number): Image {
  const prev = getPrevImage(ctx);
  const out = cloneImage(prev);
  
  const reflectHeight = Math.floor(ctx.height * n * 0.05);
  const startY = ctx.height - reflectHeight;
  
  for (let y = startY; y < ctx.height; y++) {
    const mirrorY = 2 * startY - y - 1;
    const wave = Math.sin(y * 0.1) * 5;
    
    for (let x = 0; x < ctx.width; x++) {
      if (mirrorY >= 0 && mirrorY < ctx.height) {
        const srcX = Math.floor(((x + wave) % ctx.width + ctx.width) % ctx.width);
        const [r, g, b] = getPixel(prev, srcX, mirrorY);
        const idx = (y * ctx.width + x) * 4;
        out.data[idx] = Math.round(out.data[idx] * 0.5 + r * 0.5);
        out.data[idx + 1] = Math.round(out.data[idx + 1] * 0.5 + g * 0.5);
        out.data[idx + 2] = Math.round(out.data[idx + 2] * 0.5 + b * 0.5);
      }
    }
  }
  
  return out;
}

function fnBacktick(ctx: FnContext, n: number): Image {
  const prev = getPrevImage(ctx);
  const out = createSolidImage(ctx.width, ctx.height, '#000000');
  
  const glitchN = Math.max(1, n);
  
  for (let y = 0; y < ctx.height; y++) {
    const shouldGlitch = (y * 17) % 23 < glitchN;
    const shiftAmount = shouldGlitch ? ((y * 31) % (glitchN * 20)) : 0;
    
    for (let x = 0; x < ctx.width; x++) {
      if (shouldGlitch) {
        const srcXR = ((x - shiftAmount - glitchN) % ctx.width + ctx.width) % ctx.width;
        const srcXG = ((x - shiftAmount) % ctx.width + ctx.width) % ctx.width;
        const srcXB = ((x - shiftAmount + glitchN) % ctx.width + ctx.width) % ctx.width;
        
        const [rr] = getPixel(prev, srcXR, y);
        const [, gg] = getPixel(prev, srcXG, y);
        const [, , bb] = getPixel(prev, srcXB, y);
        
        setPixel(out, x, y, rr, gg, bb);
      } else {
        const [r, g, b] = getPixel(prev, x, y);
        setPixel(out, x, y, r, g, b);
      }
    }
  }
  
  return out;
}

function fnOpenBrace(ctx: FnContext): Image {
  const prev = getPrevImage(ctx);
  const gl = initWebGL(ctx.width, ctx.height);
  
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

function fnPipe(ctx: FnContext, j: number): Image {
  const prev = getPrevImage(ctx);
  const old = getOldImage(ctx, j);
  const out = createSolidImage(ctx.width, ctx.height, '#000000');
  
  const midX = Math.floor(ctx.width / 2);
  const blendWidth = 5;
  
  for (let y = 0; y < ctx.height; y++) {
    for (let x = 0; x < ctx.width; x++) {
      if (x < midX - blendWidth) {
        const [r, g, b] = getPixel(prev, x, y);
        setPixel(out, x, y, r, g, b);
      } else if (x > midX + blendWidth) {
        const [r, g, b] = getPixel(old, x, y);
        setPixel(out, x, y, r, g, b);
      } else {
        const t = (x - (midX - blendWidth)) / (blendWidth * 2);
        const [pr, pg, pb] = getPixel(prev, x, y);
        const [or, og, ob] = getPixel(old, x, y);
        setPixel(out, x, y,
          Math.round(pr * (1 - t) + or * t),
          Math.round(pg * (1 - t) + og * t),
          Math.round(pb * (1 - t) + ob * t)
        );
      }
    }
  }
  
  return out;
}

function fnCloseBrace(ctx: FnContext): Image {
  const prev = getPrevImage(ctx);
  const gl = initWebGL(ctx.width, ctx.height);
  
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

export const characterDefs: Record<string, CharDef> = {
  'A': {
    color: '#78A10F',
    number: 1,
    fn: fnA,
    arity: 0,
    argTypes: [],
    functionName: "sphere-overlay",
    documentation: "Renders prev as texture on two 3D spheres with lighting in top-right and bottom-left quadrants"
  },
  
  'B': {
    color: '#8B4513',
    number: 2,
    fn: fnV,
    arity: 1,
    argTypes: ['color'],
    functionName: "border",
    documentation: "Circular gradient darkening edges, tinted toward color c"
  },
  
  'C': {
    color: '#FF6B35',
    number: 3,
    fn: fnC,
    arity: 1,
    argTypes: ['int'],
    functionName: "concentric-hue",
    documentation: "Creates n concentric circles alternating between original and hue-shifted versions"
  },
  
  'D': {
    color: '#FF1493',
    number: 4,
    fn: fnD,
    arity: 0,
    argTypes: [],
    functionName: "drip",
    documentation: "Metaball-based dripping effect with blobby liquid drips"
  },
  
  'E': {
    color: '#50C878',
    number: 5,
    fn: fnE,
    arity: 0,
    argTypes: [],
    functionName: "emerald",
    documentation: "Renders bright reflective 3D emeralds in symmetric pattern with large center emerald"
  },
  
  'F': {
    color: '#FFD700',
    number: 6,
    fn: fnF,
    arity: 1,
    argTypes: ['int'],
    functionName: "julia-fractal",
    documentation: "Julia set fractal masks prev - inside set shows prev, outside darkened; n controls zoom level"
  },
  
  'G': {
    color: '#9370DB',
    number: 7,
    fn: fnG,
    arity: 1,
    argTypes: ['int'],
    functionName: "grayscale-colorize",
    documentation: "Converts to grayscale then applies n-color rainbow palette based on brightness"
  },
  
  'H': {
    color: '#DC143C',
    number: 8,
    fn: fnH,
    arity: 0,
    argTypes: [],
    functionName: "hourglass",
    documentation: "Hourglass gradient: NAND blend inside hourglass shape, XOR/add/diff blend outside, creates colors from gradients"
  },
  
  'I': {
    color: '#00FF7F',
    number: 9,
    fn: fnI,
    arity: 0,
    argTypes: [],
    functionName: "invert-edges",
    documentation: "Inverts prev colors, then applies Sobel edge detection and draws edges in white"
  },
  
  'J': {
    color: '#FF8C00',
    number: 10,
    fn: fnJ,
    arity: 1,
    argTypes: ['int'],
    functionName: "variable-checkerboard",
    documentation: "Checkerboard blend where square size increases from top-left (2px) to bottom-right (52px)"
  },
  
  'K': {
    color: '#4B0082',
    number: 11,
    fn: fnK,
    arity: 1,
    argTypes: ['int'],
    functionName: "kaleidoscope",
    documentation: "Creates n-way kaleidoscope effect centered on image with 1.1x zoom"
  },
  
  'L': {
    color: '#20B2AA',
    number: 12,
    fn: fnL,
    arity: 2,
    argTypes: ['int', 'int'],
    functionName: "lissajous",
    documentation: "3D Lissajous tube textured with prev, old_image as background; j=old image, rot=rotation"
  },
  
  'M': {
    color: '#FF69B4',
    number: 13,
    fn: fnM,
    arity: 1,
    argTypes: ['int'],
    functionName: "moire",
    documentation: "Moiré interference pattern: 3 overlapping line grids create 8 zones with different hue/saturation/lightness shifts"
  },
  
  'N': {
    color: '#8A2BE2',
    number: 14,
    fn: fnN,
    arity: 0,
    argTypes: [],
    functionName: "neon",
    documentation: "Neon glow effect - edges glow in their original hue, scattered light points add ambiance"
  },
  
  'O': {
    color: '#FF6347',
    number: 15,
    fn: fnO,
    arity: 1,
    argTypes: ['int'],
    functionName: "fisheye-morph",
    documentation: "Fisheye distortion that brightens center and darkens edges with strength n"
  },
  
  'P': {
    color: '#4682B4',
    number: 16,
    fn: fnP,
    arity: 1,
    argTypes: ['int'],
    functionName: "diagonal-pixelate",
    documentation: "Pixelates with cell size n+1, diagonal split: top-left = average, bottom-right = most saturated"
  },
  
  'Q': {
    color: '#32CD32',
    number: 17,
    fn: fnQ,
    arity: 0,
    argTypes: [],
    functionName: "inverted-tile",
    documentation: "Splits prev diagonally, inverts bottom-right half, then tiles in four quadrants"
  },
  
  'R': {
    color: '#DA70D6',
    number: 18,
    fn: fnR,
    arity: 0,
    argTypes: [],
    functionName: "room",
    documentation: "3D room with three walls, ceiling, and floor textured with prev, lit from center"
  },
  
  'S': {
    color: '#87CEEB',
    number: 19,
    fn: fnTornLeft,
    arity: 1,
    argTypes: ['int'],
    functionName: "split-left",
    documentation: "Left third shows old_image at index j, right two-thirds show prev, torn-paper edge using sin waves"
  },
  
  'T': {
    color: '#F0E68C',
    number: 20,
    fn: fnT,
    arity: 1,
    argTypes: ['int'],
    functionName: "cubes",
    documentation: "n+1 3D cubes protrude from prev plane toward camera, tops textured from prev, sides use edge pixels, lit from front"
  },
  
  'U': {
    color: '#DDA0DD',
    number: 21,
    fn: fnU,
    arity: 1,
    argTypes: ['int'],
    functionName: "undulate",
    documentation: "Vertical wave distortion with amplitude n*5px, frequency n cycles, phase based on luminance"
  },
  
  'V': {
    color: '#40E0D0',
    number: 22,
    fn: fnB,
    arity: 1,
    argTypes: ['int'],
    functionName: "voronoi",
    documentation: "Breaks image into 36 voronoi cells, alternating between prev and old image"
  },
  
  'W': {
    color: '#EE82EE',
    number: 23,
    fn: fnW,
    arity: 1,
    argTypes: ['int'],
    functionName: "swirl",
    documentation: "Swirl distortion from center, rotation = n*45°, quadratic falloff"
  },
  
  'X': {
    color: '#F5DEB3',
    number: 24,
    fn: fnX,
    arity: 2,
    argTypes: ['int', 'color'],
    functionName: "shape-overlay",
    documentation: "Draws unicode shape k%64 at center in color c (★●■▲◆♥✦⬡✚◐☽⚡∞☀✿⬢...)"
  },
  
  'Y': {
    color: '#98FB98',
    number: 25,
    fn: fnY,
    arity: 1,
    argTypes: ['int'],
    functionName: "radial-hue",
    documentation: "Splits into n radial sections, each hue-rotated by i*(360/n)°"
  },
  
  'Z': {
    color: '#AFEEEE',
    number: 26,
    fn: fnZ,
    arity: 1,
    argTypes: ['int'],
    functionName: "zoom-blur",
    documentation: "Radial motion blur from center, blur amount n*4px, center 20% stays sharp"
  },
  
  '0': {
    color: '#E6E6FA',
    number: 27,
    fn: fn0,
    arity: 1,
    argTypes: ['int'],
    functionName: "overlay",
    documentation: "Overlay blend mode: combines prev with old_image at index j, darkening darks and lightening lights"
  },
  
  '1': {
    color: '#FFA07A',
    number: 28,
    fn: fn1,
    arity: 0,
    argTypes: [],
    functionName: "center-bar",
    documentation: "Middle third of image shows prev sharpened and contrast-boosted, rest is desaturated"
  },
  
  '2': {
    color: '#98D8C8',
    number: 29,
    fn: fn2,
    arity: 0,
    argTypes: [],
    functionName: "time-echo",
    documentation: "Left third shows 3rd previous image, middle shows 2nd previous, right shows prev - recent history triptych"
  },
  
  '3': {
    color: '#F7DC6F',
    number: 30,
    fn: fn3,
    arity: 0,
    argTypes: [],
    functionName: "triple-rotate",
    documentation: "Divides into 3 vertical strips; left rotated 90° CW, middle unchanged, right rotated 90° CCW"
  },
  
  '4': {
    color: '#BB8FCE',
    number: 31,
    fn: fn4,
    arity: 0,
    argTypes: [],
    functionName: "quad-rotate",
    documentation: "Divides into 2x2 quadrants, each rotated (0°, 90°, 180°, 270°)"
  },
  
  '5': {
    color: '#85C1E9',
    number: 32,
    fn: fn5,
    arity: 1,
    argTypes: ['int'],
    functionName: "triangular-split",
    documentation: "Splits prev into triangles based on cell width, each with hue shift and lightness variation"
  },
  
  '6': {
    color: '#F1948A',
    number: 33,
    fn: fn6,
    arity: 0,
    argTypes: [],
    functionName: "posterize",
    documentation: "Posterizes prev to 4 levels per channel"
  },
  
  '7': {
    color: '#82E0AA',
    number: 34,
    fn: fn7,
    arity: 0,
    argTypes: [],
    functionName: "chromatic",
    documentation: "Chromatic aberration: R shifted left 4px, G centered, B shifted right 4px"
  },
  
  '8': {
    color: '#F8C471',
    number: 35,
    fn: fn8,
    arity: 1,
    argTypes: ['int'],
    functionName: "lemniscate",
    documentation: "Infinity-loop/lemniscate distortion centered on image with strength n/5, pixels flow along curve"
  },
  
  '9': {
    color: '#D7BDE2',
    number: 36,
    fn: fn9,
    arity: 1,
    argTypes: ['int'],
    functionName: "xor-blend",
    documentation: "XORs prev with old_image at index j, creating glitchy digital artifacts"
  },
  
  '<': {
    color: '#E74C3C',
    number: 37,
    fn: fnLessThan,
    arity: 0,
    argTypes: [],
    functionName: "shift-left",
    documentation: "Shifts prev 1/3 of the width to the left with wraparound"
  },
  
  '>': {
    color: '#3498DB',
    number: 38,
    fn: fnGreaterThan,
    arity: 0,
    argTypes: [],
    functionName: "rotate-90",
    documentation: "Rotates prev 90 degrees clockwise"
  },
  
  '^': {
    color: '#2ECC71',
    number: 39,
    fn: fnCaret,
    arity: 0,
    argTypes: [],
    functionName: "shift-up",
    documentation: "Shifts prev 1/3 of the height up with wraparound"
  },
  
  '!': {
    color: '#FF4500',
    number: 40,
    fn: fnExclaim,
    arity: 1,
    argTypes: ['int'],
    functionName: "hash-overlay",
    documentation: "Overlays deterministic hash pattern where brightness = ((x*(n+17)) ^ (y*31) ^ (x*y)) % 256 at 30% opacity"
  },
  
  '"': {
    color: '#9932CC',
    number: 41,
    fn: fnDoubleQuote,
    arity: 1,
    argTypes: ['int'],
    functionName: "band-transform",
    documentation: "Splits into n horizontal bands; odd bands hue-rotated 180°, even bands have saturation inverted"
  },
  
  '#': {
    color: '#228B22',
    number: 42,
    fn: fnHash,
    arity: 1,
    argTypes: ['int'],
    functionName: "hue-sort-tiles",
    documentation: "Divides image into tiles and sorts them left-to-right, top-to-bottom by average hue, grid size = n+2 (min 2)"
  },
  
  '$': {
    color: '#FFD700',
    number: 43,
    fn: fnDollar,
    arity: 0,
    argTypes: [],
    functionName: "segment-hue-sort",
    documentation: "Color-based flood-fill segmentation, then sorts pixels within each segment by hue vertically"
  },
  
  '%': {
    color: '#8B0000',
    number: 44,
    fn: fnPercent,
    arity: 1,
    argTypes: ['int'],
    functionName: "floyd-steinberg",
    documentation: "Floyd-Steinberg dithers prev to 4-corner color palette with cell size n+1"
  },
  
  '&': {
    color: '#4169E1',
    number: 45,
    fn: fnAmpersand,
    arity: 1,
    argTypes: ['int'],
    functionName: "hue-sat-transfer",
    documentation: "Output has prev's luminance but hue/saturation from old_image at index j"
  },
  
  "'": {
    color: '#FF1493',
    number: 46,
    fn: fnApostrophe,
    arity: 1,
    argTypes: ['int'],
    functionName: "vertical-streaks",
    documentation: "Vertical streaks at x positions where x % (n+2) == 0, length based on column luminance, 50% white"
  },
  
  '(': {
    color: '#00CED1',
    number: 47,
    fn: fnOpenParen,
    arity: 1,
    argTypes: ['int'],
    functionName: "pinch",
    documentation: "Pinch distortion toward center with strength n/10, brightens pixels near center"
  },
  
  ')': {
    color: '#FF69B4',
    number: 48,
    fn: fnCloseParen,
    arity: 1,
    argTypes: ['int'],
    functionName: "bulge",
    documentation: "Bulge distortion from center with strength n/10, darkens pixels near edge"
  },
  
  '*': {
    color: '#FFD700',
    number: 49,
    fn: fnAsterisk,
    arity: 1,
    argTypes: ['int'],
    functionName: "rays",
    documentation: "n rays emanate from center, color sampled from prev at angle*width/n, soft edges at 60% opacity"
  },
  
  '+': {
    color: '#32CD32',
    number: 50,
    fn: fnPlus,
    arity: 0,
    argTypes: [],
    functionName: "zoom",
    documentation: "Zooms in 1.2x from center, can be repeated for more zoom"
  },
  
  ',': {
    color: '#BA55D3',
    number: 51,
    fn: fnComma,
    arity: 1,
    argTypes: ['color'],
    functionName: "stipple",
    documentation: "Stipples prev with dots of color c at positions where (x*13+y*7) % (1+luminance/32) == 0"
  },
  
  '-': {
    color: '#708090',
    number: 52,
    fn: fnMinus,
    arity: 1,
    argTypes: ['int'],
    functionName: "scanlines",
    documentation: "Every nth row darkened 50%, every (n*2)th row samples from n pixels above"
  },
  
  '.': {
    color: '#20B2AA',
    number: 53,
    fn: fnDot,
    arity: 1,
    argTypes: ['int'],
    functionName: "pointillism",
    documentation: "Rebuilds prev from circles of radius (n%8)+2, color from center with +10% saturation"
  },
  
  '/': {
    color: '#CD853F',
    number: 54,
    fn: fnSlash,
    arity: 1,
    argTypes: ['color'],
    functionName: "diagonal-lines",
    documentation: "Diagonal lines top-left to bottom-right in color c, spacing based on luminance (bright=sparse)"
  },
  
  ':': {
    color: '#6B8E23',
    number: 55,
    fn: fnColon,
    arity: 1,
    argTypes: ['int'],
    functionName: "circular-zoom",
    documentation: "n circular regions evenly spaced horizontally show prev at 2x zoom, rest box-blurred with radius 5"
  },
  
  ';': {
    color: '#DB7093',
    number: 56,
    fn: fnSemicolon,
    arity: 0,
    argTypes: [],
    functionName: "semicircle-reflect",
    documentation: "Semicircle arc (top half) crops prev, bottom half is reflection with wave distortion"
  },
  
  '=': {
    color: '#5F9EA0',
    number: 57,
    fn: fnEquals,
    arity: 1,
    argTypes: ['int'],
    functionName: "shifted-stripes",
    documentation: "Horizontal stripes of height n, even stripes are prev, odd stripes shifted left by stripe_index*5 pixels"
  },
  
  '?': {
    color: '#D2691E',
    number: 58,
    fn: fnQuestion,
    arity: 0,
    argTypes: [],
    functionName: "corner-effect",
    documentation: "Effect selected by (sum of corner pixel values) % 4: [solarize, emboss, edge-detect, posterize]"
  },
  
  '@': {
    color: '#7B68EE',
    number: 59,
    fn: fnA,
    arity: 0,
    argTypes: [],
    functionName: "sphere-overlay-alt",
    documentation: "Same as A: prev rendered on two 3D spheres with lighting in top-right and bottom-left quadrants"
  },
  
  '[': {
    color: '#48D1CC',
    number: 60,
    fn: fnOpenBracket,
    arity: 0,
    argTypes: [],
    functionName: "rotate-left",
    documentation: "Rotates prev 20 degrees counter-clockwise"
  },
  
  '\\': {
    color: '#C71585',
    number: 61,
    fn: fnBackslash,
    arity: 1,
    argTypes: ['color'],
    functionName: "diagonal-lines-reverse",
    documentation: "Diagonal lines from top-right to bottom-left in color c, spacing based on luminance"
  },
  
  ']': {
    color: '#00FA9A',
    number: 62,
    fn: fnCloseBracket,
    arity: 0,
    argTypes: [],
    functionName: "left-half-offset",
    documentation: "Offsets the left half of the image vertically by 20% with wraparound"
  },
  
  '_': {
    color: '#FF7F50',
    number: 63,
    fn: fnUnderscore,
    arity: 1,
    argTypes: ['int'],
    functionName: "bottom-reflect",
    documentation: "Bottom n*5% of image is reflected and overlaid with 50% opacity, with horizontal wave distortion"
  },
  
  '`': {
    color: '#6495ED',
    number: 64,
    fn: fnBacktick,
    arity: 1,
    argTypes: ['int'],
    functionName: "glitch",
    documentation: "Glitch effect: horizontal strips shifted right with RGB separation of n pixels"
  },
  
  '{': {
    color: '#DC143C',
    number: 65,
    fn: fnTornLeft,
    arity: 1,
    argTypes: ['int'],
    functionName: "torn-left",
    documentation: "Left third shows old_image, right two-thirds show prev, torn-paper edge using sin waves"
  },
  
  '|': {
    color: '#00BFFF',
    number: 66,
    fn: fnPipe,
    arity: 1,
    argTypes: ['int'],
    functionName: "vertical-split",
    documentation: "Left half is prev, right half is old_image, center column is 10px blend"
  },
  
  '}': {
    color: '#9400D3',
    number: 67,
    fn: fnTornRight,
    arity: 1,
    argTypes: ['int'],
    functionName: "torn-right",
    documentation: "Right third shows old_image, left two-thirds show prev, with torn-paper edge"
  },
  
  '~': {
    color: '#FF6347',
    number: 68,
    fn: fnTilde,
    arity: 1,
    argTypes: ['int'],
    functionName: "wave-chromatic",
    documentation: "Horizontal wave distortion with amplitude = sin(y*0.05)*n*8, chromatic aberration (R +n px, B -n px)"
  },
};
