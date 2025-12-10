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
  if (ctx.images.length <= 1) return getPrevImage(ctx);
  const idx = Math.abs(j) % (ctx.images.length - 1);
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

function fnD(ctx: FnContext, n: number): Image {
  const prev = getPrevImage(ctx);
  const out = cloneImage(prev);
  
  const divisions = Math.max(1, Math.min(n + 1, 10));
  const totalTriangles = divisions * divisions * 2;
  
  for (let row = 0; row < divisions; row++) {
    for (let col = 0; col < divisions; col++) {
      const x0 = Math.floor((col / divisions) * ctx.width);
      const y0 = Math.floor((row / divisions) * ctx.height);
      const x1 = Math.floor(((col + 1) / divisions) * ctx.width);
      const y1 = Math.floor(((row + 1) / divisions) * ctx.height);
      
      const triIndex = (row * divisions + col) * 2;
      const hue1 = ((triIndex * 137.5) % 360);
      const hue2 = (((triIndex + 1) * 137.5) % 360);
      
      const sat1 = 0.6 + (triIndex % 3) * 0.15;
      const sat2 = 0.6 + ((triIndex + 1) % 3) * 0.15;
      
      const light1 = 0.4 + (triIndex % 5) * 0.1;
      const light2 = 0.4 + ((triIndex + 1) % 5) * 0.1;
      
      for (let y = y0; y < y1; y++) {
        for (let x = x0; x < x1; x++) {
          const localX = (x - x0) / (x1 - x0);
          const localY = (y - y0) / (y1 - y0);
          
          const [r, g, b] = getPixel(prev, x, y);
          const [h, s, l] = rgbToHsl(r, g, b);
          
          const isUpperTriangle = localX + localY < 1;
          const hue = isUpperTriangle ? hue1 : hue2;
          const sat = isUpperTriangle ? sat1 : sat2;
          const light = isUpperTriangle ? light1 : light2;
          
          const avgLuminance = l;
          const finalLight = light * 0.7 + avgLuminance * 0.3;
          
          const [nr, ng, nb] = hslToRgb(hue, sat, finalLight);
          setPixel(out, x, y, nr, ng, nb);
        }
      }
    }
  }
  
  return out;
}

function fnE(ctx: FnContext, c: string, n: number): Image {
  const prev = getPrevImage(ctx);
  const out = cloneImage(prev);
  const [cr, cg, cb] = hexToRgb(c);
  
  const numWaves = Math.max(1, Math.min(n, 20));
  
  for (let i = 0; i < numWaves; i++) {
    const period = ctx.width / (1 + i * 0.3);
    const amplitude = ctx.height * 0.45 * Math.sin(i * 0.8);
    const thickness = Math.max(2, Math.floor(4 + Math.sin(i * 0.5) * 3));
    const phase = Math.PI * Math.cos(i * 0.6);
    
    let prevY = -1;
    
    for (let x = 0; x < ctx.width; x++) {
      const waveY = Math.floor(ctx.height / 2 + Math.sin((x / period) * Math.PI * 2 + phase) * amplitude);
      
      if (prevY !== -1) {
        const y1 = Math.min(prevY, waveY);
        const y2 = Math.max(prevY, waveY);
        
        for (let y = y1; y <= y2; y++) {
          for (let dy = -Math.floor(thickness / 2); dy <= Math.floor(thickness / 2); dy++) {
            const drawY = y + dy;
            if (drawY >= 0 && drawY < ctx.height && x >= 0 && x < ctx.width) {
              const idx = (drawY * ctx.width + x) * 4;
              const alpha = 0.4;
              out.data[idx] = Math.round(out.data[idx] * (1 - alpha) + cr * alpha);
              out.data[idx + 1] = Math.round(out.data[idx + 1] * (1 - alpha) + cg * alpha);
              out.data[idx + 2] = Math.round(out.data[idx + 2] * (1 - alpha) + cb * alpha);
            }
          }
        }
      } else {
        for (let dy = -Math.floor(thickness / 2); dy <= Math.floor(thickness / 2); dy++) {
          const drawY = waveY + dy;
          if (drawY >= 0 && drawY < ctx.height) {
            const idx = (drawY * ctx.width + x) * 4;
            const alpha = 0.4;
            out.data[idx] = Math.round(out.data[idx] * (1 - alpha) + cr * alpha);
            out.data[idx + 1] = Math.round(out.data[idx + 1] * (1 - alpha) + cg * alpha);
            out.data[idx + 2] = Math.round(out.data[idx + 2] * (1 - alpha) + cb * alpha);
          }
        }
      }
      
      prevY = waveY;
    }
  }
  
  return out;
}

function fnF(ctx: FnContext, n: number): Image {
  const prev = getPrevImage(ctx);
  const out = cloneImage(prev);
  
  const maxIterations = Math.max(10, Math.min(n * 10, 100));
  
  const centerPixel = getPixel(prev, Math.floor(ctx.width / 2), Math.floor(ctx.height / 2));
  const cReal = (centerPixel[0] / 255) * 2 - 1;
  const cImag = (centerPixel[1] / 255) * 2 - 1;
  
  for (let py = 0; py < ctx.height; py++) {
    for (let px = 0; px < ctx.width; px++) {
      let zReal = (px / ctx.width) * 3 - 1.5;
      let zImag = (py / ctx.height) * 3 - 1.5;
      
      let iteration = 0;
      while (iteration < maxIterations && zReal * zReal + zImag * zImag < 4) {
        const zRealTemp = zReal * zReal - zImag * zImag + cReal;
        zImag = 2 * zReal * zImag + cImag;
        zReal = zRealTemp;
        iteration++;
      }
      
      const intensity = iteration / maxIterations;
      const idx = (py * ctx.width + px) * 4;
      
      out.data[idx] = Math.min(255, out.data[idx] + intensity * 255);
      out.data[idx + 1] = Math.min(255, out.data[idx + 1] + intensity * 200);
      out.data[idx + 2] = Math.min(255, out.data[idx + 2] + intensity * 150);
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

function fnH(ctx: FnContext, j: number): Image {
  const prev = getPrevImage(ctx);
  const old = getOldImage(ctx, j);
  const out = createSolidImage(ctx.width, ctx.height, '#000000');
  
  for (let y = 0; y < ctx.height; y++) {
    const stripHeight = 5 + Math.sin(y * 0.1) * 20;
    const stripIndex = Math.floor(y / stripHeight);
    const useOld = stripIndex % 2 === 1;
    
    for (let x = 0; x < ctx.width; x++) {
      const src = useOld ? old : prev;
      const [r, g, b] = getPixel(src, x, y);
      setPixel(out, x, y, r, g, b);
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

function fnL(ctx: FnContext, c: string): Image {
  const prev = getPrevImage(ctx);
  const out = cloneImage(prev);
  const [cr, cg, cb] = hexToRgb(c);
  
  const occupied = new Set<string>();
  const candidates = new Set<string>();
  
  const startX = Math.floor(ctx.width / 2);
  const startY = 0;
  occupied.add(`${startX},${startY}`);
  
  const dirs = [[-1, 0], [1, 0], [0, -1], [0, 1], [-1, -1], [1, -1], [-1, 1], [1, 1]];
  for (const [dx, dy] of dirs) {
    const nx = startX + dx;
    const ny = startY + dy;
    if (nx >= 0 && nx < ctx.width && ny >= 0 && ny < ctx.height) {
      candidates.add(`${nx},${ny}`);
    }
  }
  
  const seededRandom = (seed: number) => {
    let state = seed;
    return () => {
      state = (state * 1103515245 + 12345) & 0x7fffffff;
      return state / 0x7fffffff;
    };
  };
  const rand = seededRandom(12345);
  
  let segmentCount = 0;
  const maxSegments = 500;
  
  while (candidates.size > 0 && segmentCount < maxSegments) {
    const candidateArray = Array.from(candidates);
    const chosen = candidateArray[Math.floor(rand() * candidateArray.length)];
    const [x, y] = chosen.split(',').map(Number);
    
    occupied.add(chosen);
    candidates.delete(chosen);
    
    setPixel(out, x, y, cr, cg, cb);
    segmentCount++;
    
    for (const [dx, dy] of dirs) {
      const nx = x + dx;
      const ny = y + dy;
      const key = `${nx},${ny}`;
      if (nx >= 0 && nx < ctx.width && ny >= 0 && ny < ctx.height && 
          !occupied.has(key) && !candidates.has(key)) {
        candidates.add(key);
      }
    }
  }
  
  return out;
}

function fnM(ctx: FnContext, j: number): Image {
  const prev = getPrevImage(ctx);
  const old = getOldImage(ctx, j);
  const out = createSolidImage(ctx.width, ctx.height, '#000000');
  
  const cx = ctx.width / 2;
  const cy = ctx.height / 2;
  const maxRadius = Math.sqrt(cx * cx + cy * cy);
  
  for (let y = 0; y < ctx.height; y++) {
    for (let x = 0; x < ctx.width; x++) {
      const dx = x - cx;
      const dy = y - cy;
      const radius = Math.sqrt(dx * dx + dy * dy);
      const angle = Math.atan2(dy, dx);
      
      const spiralValue = radius + angle * 20;
      const useOld = Math.floor(spiralValue / 30) % 2 === 0;
      
      const src = useOld ? old : prev;
      const [r, g, b] = getPixel(src, x, y);
      setPixel(out, x, y, r, g, b);
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
    fn: fnB,
    arity: 1,
    argTypes: ['int'],
    functionName: "voronoi-blend",
    documentation: "Breaks image into 36 voronoi cells, alternating between prev and old image"
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
    arity: 1,
    argTypes: ['int'],
    functionName: "triangular-split",
    documentation: "Splits prev into (n+1)² triangles, each colorized with a hue based on its index"
  },
  
  'E': {
    color: '#00CED1',
    number: 5,
    fn: fnE,
    arity: 2,
    argTypes: ['color', 'int'],
    functionName: "sinusoidal-waves",
    documentation: "Superimposes n sinusoidal waves of varying thickness and period in color c"
  },
  
  'F': {
    color: '#FFD700',
    number: 6,
    fn: fnF,
    arity: 1,
    argTypes: ['int'],
    functionName: "julia-fractal",
    documentation: "Draws Julia set fractal using prev's center pixel as c parameter, screen blended"
  },
  
  'G': {
    color: '#9370DB',
    number: 7,
    fn: fnG,
    arity: 1,
    argTypes: ['int'],
    functionName: "gradient-map",
    documentation: "Converts to grayscale then applies gradient map using n colors from histogram"
  },
  
  'H': {
    color: '#DC143C',
    number: 8,
    fn: fnH,
    arity: 1,
    argTypes: ['int'],
    functionName: "horizontal-strips",
    documentation: "Horizontal strips alternate between prev and old_image, strip height varies with sin"
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
    arity: 1,
    argTypes: ['color'],
    functionName: "lichtenberg",
    documentation: "Draws Lichtenberg figures from top using DLA simulation (max 500 segments) in color c"
  },
  
  'M': {
    color: '#FF69B4',
    number: 13,
    fn: fnM,
    arity: 1,
    argTypes: ['int'],
    functionName: "spiral-interleave",
    documentation: "Interleaves prev and old_image in a spiral pattern from center outward"
  },
};
