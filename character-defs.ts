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

function fnL(ctx: FnContext, c: string, recursionDepth: number, angleVariation: number): Image {
  const prev = getPrevImage(ctx);
  const out = cloneImage(prev);
  const [cr, cg, cb] = hexToRgb(c);
  
  const maxDepth = Math.max(2, Math.min(recursionDepth, 15));
  const angleVar = Math.max(0.1, Math.min(angleVariation * 0.15, Math.PI));
  
  const seededRandom = (seed: number) => {
    let state = seed;
    return () => {
      state = (state * 1103515245 + 12345) & 0x7fffffff;
      return state / 0x7fffffff;
    };
  };
  const rand = seededRandom(12345);
  
  const drawLine = (x1: number, y1: number, x2: number, y2: number, thickness: number) => {
    const dx = x2 - x1;
    const dy = y2 - y1;
    const dist = Math.sqrt(dx * dx + dy * dy);
    const steps = Math.ceil(dist);
    
    for (let i = 0; i <= steps; i++) {
      const t = i / steps;
      const x = Math.floor(x1 + dx * t);
      const y = Math.floor(y1 + dy * t);
      
      for (let r = 0; r < thickness; r++) {
        for (let angle = 0; angle < Math.PI * 2; angle += 0.5) {
          const px = Math.floor(x + Math.cos(angle) * r);
          const py = Math.floor(y + Math.sin(angle) * r);
          if (px >= 0 && px < ctx.width && py >= 0 && py < ctx.height) {
            setPixel(out, px, py, cr, cg, cb);
          }
        }
      }
    }
  };
  
  const growBranch = (x: number, y: number, angle: number, length: number, thickness: number, depth: number) => {
    if (depth > maxDepth || length < 8) return;
    
    const angleVariation = (rand() - 0.5) * angleVar * 0.5;
    const actualAngle = angle + angleVariation;
    
    const endX = x + Math.cos(actualAngle) * length;
    const endY = y + Math.sin(actualAngle) * length;
    
    if (endX < 0 || endX >= ctx.width || endY < 0 || endY >= ctx.height) return;
    
    drawLine(x, y, endX, endY, thickness);
    
    const branchProbability = Math.min(0.9, 0.3 + maxDepth * 0.05);
    const baseBranches = depth === 0 ? Math.min(4, 1 + Math.floor(maxDepth / 4)) : 1;
    const extraBranches = rand() < branchProbability ? 1 : 0;
    const numBranches = baseBranches + extraBranches;
    
    for (let i = 0; i < numBranches; i++) {
      const branchPoint = 0.4 + rand() * 0.5;
      const branchX = x + Math.cos(actualAngle) * length * branchPoint;
      const branchY = y + Math.sin(actualAngle) * length * branchPoint;
      
      const spreadAngle = (i - (numBranches - 1) / 2) * angleVar * 0.8;
      const branchAngle = actualAngle + spreadAngle + (rand() - 0.5) * angleVar * 0.5;
      const branchLength = length * (0.4 + rand() * 0.35);
      const branchThickness = Math.max(1, thickness - 1);
      
      growBranch(branchX, branchY, branchAngle, branchLength, branchThickness, depth + 1);
    }
  };
  
  const topY = Math.floor(ctx.height * 0.05);
  const centerX = Math.floor(ctx.width / 2);
  const mainLength = ctx.height * 0.7;
  const mainAngle = Math.PI / 2;
  
  growBranch(centerX, topY, mainAngle, mainLength, 8, 0);
  
  return out;
}

function fnM(ctx: FnContext, spiralEffect: number, j: number): Image {
  const prev = getPrevImage(ctx);
  const old = getOldImage(ctx, j);
  const out = createSolidImage(ctx.width, ctx.height, '#000000');
  
  const blockSize = Math.max(2, Math.min(spiralEffect + 2, 50));
  const spiralTightness = Math.max(5, spiralEffect * 5);
  
  const cx = ctx.width / 2;
  const cy = ctx.height / 2;
  
  for (let by = 0; by < ctx.height; by += blockSize) {
    for (let bx = 0; bx < ctx.width; bx += blockSize) {
      const centerX = bx + blockSize / 2;
      const centerY = by + blockSize / 2;
      
      const dx = centerX - cx;
      const dy = centerY - cy;
      const radius = Math.sqrt(dx * dx + dy * dy);
      const angle = Math.atan2(dy, dx);
      
      const spiralValue = radius + angle * spiralTightness;
      const bandIndex = Math.floor(spiralValue / spiralTightness);
      const useOld = bandIndex % 2 === 0;
      
      if (useOld) {
        for (let y = by; y < by + blockSize && y < ctx.height; y++) {
          for (let x = bx; x < bx + blockSize && x < ctx.width; x++) {
            const [r, g, b] = getPixel(old, x, y);
            setPixel(out, x, y, r, g, b);
          }
        }
      } else {
        let sumR = 0, sumG = 0, sumB = 0, count = 0;
        for (let y = by; y < by + blockSize && y < ctx.height; y++) {
          for (let x = bx; x < bx + blockSize && x < ctx.width; x++) {
            const [r, g, b] = getPixel(prev, x, y);
            sumR += r;
            sumG += g;
            sumB += b;
            count++;
          }
        }
        
        const avgR = Math.round(sumR / count);
        const avgG = Math.round(sumG / count);
        const avgB = Math.round(sumB / count);
        
        for (let y = by; y < by + blockSize && y < ctx.height; y++) {
          for (let x = bx; x < bx + blockSize && x < ctx.width; x++) {
            setPixel(out, x, y, avgR, avgG, avgB);
          }
        }
      }
    }
  }
  
  return out;
}

function fnN(ctx: FnContext, j: number): Image {
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
    precision mediump float;
    uniform sampler2D texture;
    uniform vec2 resolution;
    varying vec2 vUV;
    
    void main() {
      vec2 uv = vUV;
      float wave = sin(uv.x * 20.0) * 0.1;
      float shade = 0.7 + 0.3 * cos(uv.x * 20.0);
      
      vec2 distortedUV = vec2(uv.x, uv.y + wave * (0.5 - abs(uv.y - 0.5)));
      vec3 color = texture2D(texture, distortedUV).rgb * shade;
      
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

function fnS(ctx: FnContext, n: number): Image {
  const prev = getPrevImage(ctx);
  const out = createSolidImage(ctx.width, ctx.height, '#000000');
  
  const skewAmount = (n / 20) * ctx.width * 0.5;
  
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

function fnT(ctx: FnContext, n: number): Image {
  const prev = getPrevImage(ctx);
  const gl = initWebGL(ctx.width, ctx.height);
  
  const numDrawers = Math.max(1, n + 1);
  
  // Vertex shader with MVP transformation
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
  
  // Fragment shader with lighting
  const fragmentShader = `
    precision highp float;
    
    uniform sampler2D uTexture;
    uniform vec3 uLightDir;
    uniform int uFaceType; // 0 = front, 1 = side, 2 = top/bottom
    
    varying vec3 vNormal;
    varying vec2 vTexCoord;
    varying vec3 vWorldPos;
    
    void main() {
      vec3 normal = normalize(vNormal);
      vec3 lightDir = normalize(uLightDir);
      
      float ambient = 0.3;
      float diffuse = max(dot(normal, lightDir), 0.0) * 0.7;
      float lighting = ambient + diffuse;
      
      vec3 color = texture2D(uTexture, vTexCoord).rgb * lighting;
      gl_FragColor = vec4(color, 1.0);
    }
  `;
  
  const program = createShaderProgram(gl, vertexShader, fragmentShader);
  
  // Create texture from prev image
  const texture = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, texture);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, prev.width, prev.height, 0, gl.RGBA, gl.UNSIGNED_BYTE, prev.data);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  
  // Helper to create box geometry
  function createBox(cx: number, cy: number, hw: number, hh: number, depth: number): { vertices: number[], normals: number[], texCoords: number[] } {
    const x0 = cx - hw, x1 = cx + hw;
    const y0 = cy - hh, y1 = cy + hh;
    const z0 = 0, z1 = depth;
    
    // Texture coords based on box position in image
    const u0 = cx - hw, u1 = cx + hw;
    const v0 = cy - hh, v1 = cy + hh;
    
    const vertices: number[] = [];
    const normals: number[] = [];
    const texCoords: number[] = [];
    
    // Front face (z = z1, facing +z towards camera)
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
    
    // Top face (+y)
    vertices.push(x0,y1,z1, x1,y1,z1, x1,y1,z0, x0,y1,z1, x1,y1,z0, x0,y1,z0);
    for(let i=0;i<6;i++) normals.push(0,1,0);
    texCoords.push(u0,v1, u1,v1, u1,v1, u0,v1, u1,v1, u0,v1);
    
    // Bottom face (-y)
    vertices.push(x0,y0,z0, x1,y0,z0, x1,y0,z1, x0,y0,z0, x1,y0,z1, x0,y0,z1);
    for(let i=0;i<6;i++) normals.push(0,-1,0);
    texCoords.push(u0,v0, u1,v0, u1,v0, u0,v0, u1,v0, u0,v0);
    
    return { vertices, normals, texCoords };
  }
  
  // Seeded random
  const hash = (n: number) => {
    const x = Math.sin(n) * 43758.5453;
    return x - Math.floor(x);
  };
  
  // Generate all boxes
  const allVertices: number[] = [];
  const allNormals: number[] = [];
  const allTexCoords: number[] = [];
  
  for (let i = 0; i < numDrawers; i++) {
    const cx = hash(i * 127.1);
    const cy = hash(i * 311.7);
    const hw = 0.03 + hash(i * 74.3) * 0.08;
    const hh = 0.025 + hash(i * 183.9) * 0.06;
    const depth = 0.3 + hash(i * 271.3) * 0.9;
    
    const box = createBox(cx, cy, hw, hh, depth);
    allVertices.push(...box.vertices);
    allNormals.push(...box.normals);
    allTexCoords.push(...box.texCoords);
  }
  
  // First render the background quad
  gl.useProgram(program);
  gl.disable(gl.DEPTH_TEST);
  
  const bgVertices = new Float32Array([0,0,0, 1,0,0, 1,1,0, 0,0,0, 1,1,0, 0,1,0]);
  const bgNormals = new Float32Array([0,0,1, 0,0,1, 0,0,1, 0,0,1, 0,0,1, 0,0,1]);
  const bgTexCoords = new Float32Array([0,0, 1,0, 1,1, 0,0, 1,1, 0,1]);
  
  const posLoc = gl.getAttribLocation(program, 'aPosition');
  const normLoc = gl.getAttribLocation(program, 'aNormal');
  const texLoc = gl.getAttribLocation(program, 'aTexCoord');
  
  // Identity matrices for background
  const identity = new Float32Array([1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1]);
  const ortho = new Float32Array([2,0,0,0, 0,2,0,0, 0,0,-1,0, -1,-1,0,1]);
  
  gl.uniformMatrix4fv(gl.getUniformLocation(program, 'uProjection'), false, ortho);
  gl.uniformMatrix4fv(gl.getUniformLocation(program, 'uView'), false, identity);
  gl.uniformMatrix4fv(gl.getUniformLocation(program, 'uModel'), false, identity);
  gl.uniform3f(gl.getUniformLocation(program, 'uLightDir'), 0.3, 0.5, 1.0);
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
  
  // Now render boxes with depth testing
  gl.enable(gl.DEPTH_TEST);
  gl.depthFunc(gl.LESS);
  gl.clear(gl.DEPTH_BUFFER_BIT);
  
  // Perspective projection with wider FOV
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
  
  // View matrix - camera positioned to see full 0-1 range
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
  
  // Read pixels
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
  
  // Cleanup - disable vertex attributes before deleting buffers
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
  
  const maxRotation = (n * 45) * (Math.PI / 180);
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
  
  for (let y = 0; y < ctx.height; y++) {
    for (let x = 0; x < ctx.width; x++) {
      const maskIdx = (y * ctx.width + x) * 4;
      if (maskData.data[maskIdx] > 128) {
        setPixel(out, x, y, cr, cg, cb);
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
      
      if (x + y > hw) {
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

function fn0(ctx: FnContext, n: number): Image {
  const prev = getPrevImage(ctx);
  const out = cloneImage(prev);
  
  const cx = ctx.width / 2;
  const cy = ctx.height / 2;
  const maxR = Math.sqrt(cx * cx + cy * cy);
  const opacity = Math.max(0.1, Math.min((n * 10) / 100, 1));
  
  for (let y = 0; y < ctx.height; y++) {
    for (let x = 0; x < ctx.width; x++) {
      const dx = x - cx;
      const dy = y - cy;
      const dist = Math.sqrt(dx * dx + dy * dy);
      const gradient = 1 - (dist / maxR);
      const gradientValue = gradient * 255;
      
      const idx = (y * ctx.width + x) * 4;
      out.data[idx] = Math.min(255, Math.round(out.data[idx] * (1 - opacity) + (out.data[idx] * gradientValue / 255) * opacity + gradientValue * opacity * 0.5));
      out.data[idx + 1] = Math.min(255, Math.round(out.data[idx + 1] * (1 - opacity) + (out.data[idx + 1] * gradientValue / 255) * opacity + gradientValue * opacity * 0.5));
      out.data[idx + 2] = Math.min(255, Math.round(out.data[idx + 2] * (1 - opacity) + (out.data[idx + 2] * gradientValue / 255) * opacity + gradientValue * opacity * 0.5));
    }
  }
  
  return out;
}

function fn1(ctx: FnContext, n: number): Image {
  const prev = getPrevImage(ctx);
  const out = createSolidImage(ctx.width, ctx.height, '#000000');
  
  const barWidthPercent = Math.max(1, Math.min(n, 100)) / 100;
  const barHalfWidth = Math.floor((ctx.width * barWidthPercent) / 2);
  const barStart = Math.floor(ctx.width / 2) - barHalfWidth;
  const barEnd = Math.floor(ctx.width / 2) + barHalfWidth;
  
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

function fn2(ctx: FnContext, n: number): Image {
  const prev = getPrevImage(ctx);
  const out = cloneImage(prev);
  
  const levels = Math.max(2, Math.min(n, 32));
  const step = 256 / levels;
  
  let sumR = 0, sumG = 0, sumB = 0;
  const pixelCount = ctx.width * ctx.height;
  
  for (let i = 0; i < out.data.length; i += 4) {
    sumR += out.data[i];
    sumG += out.data[i + 1];
    sumB += out.data[i + 2];
    
    out.data[i] = Math.floor(out.data[i] / step) * step;
    out.data[i + 1] = Math.floor(out.data[i + 1] / step) * step;
    out.data[i + 2] = Math.floor(out.data[i + 2] / step) * step;
  }
  
  const avgR = sumR / pixelCount;
  const avgG = sumG / pixelCount;
  const avgB = sumB / pixelCount;
  const compR = 255 - avgR;
  const compG = 255 - avgG;
  const compB = 255 - avgB;
  
  for (let y = 0; y < ctx.height; y++) {
    for (let x = 0; x < ctx.width; x++) {
      const t = x / ctx.width;
      const idx = (y * ctx.width + x) * 4;
      
      const gradR = avgR * (1 - t) + compR * t;
      const gradG = avgG * (1 - t) + compG * t;
      const gradB = avgB * (1 - t) + compB * t;
      
      out.data[idx] = Math.min(255, Math.round(out.data[idx] * 0.7 + gradR * 0.3));
      out.data[idx + 1] = Math.min(255, Math.round(out.data[idx + 1] * 0.7 + gradG * 0.3));
      out.data[idx + 2] = Math.min(255, Math.round(out.data[idx + 2] * 0.7 + gradB * 0.3));
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
  const feather = 5;
  
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
      
      const distToHorizSeam = Math.abs(y - hh);
      const distToVertSeam = Math.abs(x - hw);
      const minDist = Math.min(distToHorizSeam, distToVertSeam);
      
      if (minDist < feather) {
        const blend = minDist / feather;
        const [or, og, ob] = getPixel(prev, x, y);
        setPixel(out, x, y,
          Math.round(r * blend + or * (1 - blend)),
          Math.round(g * blend + og * (1 - blend)),
          Math.round(b * blend + ob * (1 - blend))
        );
      } else {
        setPixel(out, x, y, r, g, b);
      }
    }
  }
  
  return out;
}

function fn5(ctx: FnContext, n: number): Image {
  const prev = getPrevImage(ctx);
  const out = createSolidImage(ctx.width, ctx.height, '#000000');
  
  const cellSize = Math.max(10, n * 10);
  const h = cellSize * Math.sqrt(3) / 2;
  
  for (let y = 0; y < ctx.height; y++) {
    for (let x = 0; x < ctx.width; x++) {
      const row = Math.floor(y / h);
      const isOddRow = row % 2 === 1;
      const offsetX = isOddRow ? cellSize / 2 : 0;
      const col = Math.floor((x + offsetX) / cellSize);
      
      const triX = (x + offsetX) - col * cellSize;
      const triY = y - row * h;
      
      const isUpper = triY < h * (1 - triX / cellSize) && triY < h * triX / cellSize;
      const isLower = triY > h * triX / cellSize || triY > h * (1 - triX / cellSize);
      
      let centerX: number, centerY: number;
      
      if (isUpper || (!isUpper && !isLower && triY < h / 2)) {
        centerX = col * cellSize + cellSize / 2 - offsetX;
        centerY = row * h + h / 3;
      } else {
        centerX = col * cellSize + cellSize / 2 - offsetX;
        centerY = row * h + 2 * h / 3;
      }
      
      const [sr, sg, sb] = getPixel(prev, Math.floor(centerX), Math.floor(centerY));
      
      const edgeDist = Math.min(
        Math.abs(triX),
        Math.abs(triX - cellSize),
        Math.abs(triY),
        Math.abs(triY - h)
      );
      
      if (edgeDist < 2) {
        setPixel(out, x, y, 255 - sr, 255 - sg, 255 - sb);
      } else {
        setPixel(out, x, y, sr, sg, sb);
      }
    }
  }
  
  return out;
}

function fn6(ctx: FnContext, n: number): Image {
  const prev = getPrevImage(ctx);
  const out = createSolidImage(ctx.width, ctx.height, '#000000');
  
  const hexRadius = Math.max(5, n * 5);
  const hexWidth = hexRadius * 2;
  const hexHeight = hexRadius * Math.sqrt(3);
  
  const pixelToHex = (px: number, py: number): [number, number] => {
    const q = (2 / 3 * px) / hexRadius;
    const r = (-1 / 3 * px + Math.sqrt(3) / 3 * py) / hexRadius;
    return [q, r];
  };
  
  const hexRound = (q: number, r: number): [number, number] => {
    const s = -q - r;
    let rq = Math.round(q);
    let rr = Math.round(r);
    let rs = Math.round(s);
    
    const qDiff = Math.abs(rq - q);
    const rDiff = Math.abs(rr - r);
    const sDiff = Math.abs(rs - s);
    
    if (qDiff > rDiff && qDiff > sDiff) {
      rq = -rr - rs;
    } else if (rDiff > sDiff) {
      rr = -rq - rs;
    }
    
    return [rq, rr];
  };
  
  const hexToPixel = (q: number, r: number): [number, number] => {
    const x = hexRadius * (3 / 2 * q);
    const y = hexRadius * (Math.sqrt(3) / 2 * q + Math.sqrt(3) * r);
    return [x, y];
  };
  
  const hexAverages = new Map<string, [number, number, number, number]>();
  
  for (let y = 0; y < ctx.height; y++) {
    for (let x = 0; x < ctx.width; x++) {
      const [q, r] = pixelToHex(x, y);
      const [hq, hr] = hexRound(q, r);
      const key = `${hq},${hr}`;
      
      const [pr, pg, pb] = getPixel(prev, x, y);
      
      if (!hexAverages.has(key)) {
        hexAverages.set(key, [0, 0, 0, 0]);
      }
      const avg = hexAverages.get(key)!;
      avg[0] += pr;
      avg[1] += pg;
      avg[2] += pb;
      avg[3]++;
    }
  }
  
  for (let y = 0; y < ctx.height; y++) {
    for (let x = 0; x < ctx.width; x++) {
      const [q, r] = pixelToHex(x, y);
      const [hq, hr] = hexRound(q, r);
      const key = `${hq},${hr}`;
      
      const [centerX, centerY] = hexToPixel(hq, hr);
      const dist = Math.sqrt((x - centerX) ** 2 + (y - centerY) ** 2);
      
      const avg = hexAverages.get(key);
      if (avg && avg[3] > 0) {
        const ar = Math.round(avg[0] / avg[3]);
        const ag = Math.round(avg[1] / avg[3]);
        const ab = Math.round(avg[2] / avg[3]);
        
        if (dist > hexRadius - 1.5) {
          setPixel(out, x, y, Math.max(0, ar - 30), Math.max(0, ag - 30), Math.max(0, ab - 30));
        } else {
          setPixel(out, x, y, ar, ag, ab);
        }
      }
    }
  }
  
  return out;
}

function fn7(ctx: FnContext, n: number): Image {
  const prev = getPrevImage(ctx);
  const out = cloneImage(prev);
  
  const numRects = Math.max(1, Math.min(n, 20));
  
  for (let i = 1; i <= numRects; i++) {
    const rx = Math.floor(i * ctx.width / (numRects + 2));
    const ry = Math.floor(i * ctx.height / (numRects + 4));
    const rw = Math.floor(ctx.width / (numRects + 2));
    const rh = Math.floor(ctx.height / (numRects + 4));
    
    const hueRotation = (i * 360 / numRects);
    
    for (let y = ry; y < ry + rh && y < ctx.height; y++) {
      for (let x = rx; x < rx + rw && x < ctx.width; x++) {
        const [r, g, b] = getPixel(prev, x, y);
        const [h, s, l] = rgbToHsl(r, g, b);
        const [nr, ng, nb] = hslToRgb((h + hueRotation) % 360, s, l);
        setPixel(out, x, y, nr, ng, nb);
      }
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

function fn9(ctx: FnContext, n: number): Image {
  const prev = getPrevImage(ctx);
  const out = createSolidImage(ctx.width, ctx.height, '#000000');
  
  const mode = Math.abs(n) % 9;
  
  const blend = (a: number, b: number, mode: number): number => {
    const an = a / 255;
    const bn = b / 255;
    let result: number;
    
    switch (mode) {
      case 0: result = an * bn; break;
      case 1: result = 1 - (1 - an) * (1 - bn); break;
      case 2: result = an < 0.5 ? 2 * an * bn : 1 - 2 * (1 - an) * (1 - bn); break;
      case 3: result = Math.min(an, bn); break;
      case 4: result = Math.max(an, bn); break;
      case 5: result = bn === 0 ? 0 : Math.min(1, an / (1 - bn)); break;
      case 6: result = bn === 1 ? 1 : Math.max(0, 1 - (1 - an) / bn); break;
      case 7: result = bn < 0.5 ? 2 * an * bn : 1 - 2 * (1 - an) * (1 - bn); break;
      case 8: result = bn < 0.5 
        ? an - (1 - 2 * bn) * an * (1 - an) 
        : an + (2 * bn - 1) * (Math.sqrt(an) - an); break;
      default: result = an;
    }
    
    return Math.round(Math.max(0, Math.min(1, result)) * 255);
  };
  
  for (let y = 0; y < ctx.height; y++) {
    for (let x = 0; x < ctx.width; x++) {
      const [r, g, b] = getPixel(prev, x, y);
      setPixel(out, x, y, blend(r, r, mode), blend(g, g, mode), blend(b, b, mode));
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
    functionName: "grayscale-colorize",
    documentation: "Converts to grayscale then applies n-color rainbow palette based on brightness"
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
    arity: 3,
    argTypes: ['color', 'int', 'int'],
    functionName: "lichtenberg",
    documentation: "Draws Lichtenberg figures with recursive branching in color c, recursion depth (3-12), and angle variation (1-30)"
  },
  
  'M': {
    color: '#FF69B4',
    number: 13,
    fn: fnM,
    arity: 2,
    argTypes: ['int', 'int'],
    functionName: "spiral-interleave",
    documentation: "Interleaves prev and old_image in jagged pixelated spiral, spiral effect (5-100), old image index"
  },
  
  'N': {
    color: '#8A2BE2',
    number: 14,
    fn: fnN,
    arity: 1,
    argTypes: ['int'],
    functionName: "xor-blend",
    documentation: "XORs prev with old_image at index j, creating glitchy digital artifacts"
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
    functionName: "corrugated",
    documentation: "Image transformed with sinusoidal shape in z dimension, looks vertically corrugated"
  },
  
  'S': {
    color: '#87CEEB',
    number: 19,
    fn: fnS,
    arity: 1,
    argTypes: ['int'],
    functionName: "skew-left",
    documentation: "Skews image left by amount based on n, with wraparound"
  },
  
  'T': {
    color: '#F0E68C',
    number: 20,
    fn: fnT,
    arity: 1,
    argTypes: ['int'],
    functionName: "cubes",
    documentation: "3D cubes emerge from image in deterministic chaotic patterns, light from top-right"
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
    fn: fnV,
    arity: 1,
    argTypes: ['color'],
    functionName: "vignette-tint",
    documentation: "Vignette with radius 0.7, darkened edges tinted toward color c"
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
    functionName: "radial-light",
    documentation: "Radial gradient from center (white) to edges (black) multiplied with prev at n*10% opacity, then added back"
  },
  
  '1': {
    color: '#FFA07A',
    number: 28,
    fn: fn1,
    arity: 1,
    argTypes: ['int'],
    functionName: "center-bar",
    documentation: "Vertical bar at center (width = n% of image) shows prev sharpened and contrast-boosted, rest is desaturated"
  },
  
  '2': {
    color: '#98D8C8',
    number: 29,
    fn: fn2,
    arity: 1,
    argTypes: ['int'],
    functionName: "posterize-gradient",
    documentation: "Posterizes prev to n levels per channel, then adds gradient map from average color to its complement"
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
    documentation: "Divides into 2x2 quadrants, each rotated (0°, 90°, 180°, 270°), seams blended with 5px feather"
  },
  
  '5': {
    color: '#85C1E9',
    number: 32,
    fn: fn5,
    arity: 1,
    argTypes: ['int'],
    functionName: "triangular-tile",
    documentation: "Triangular tiling with cell size n*10, tiles filled from prev at center, edges drawn in inverted colors"
  },
  
  '6': {
    color: '#F1948A',
    number: 33,
    fn: fn6,
    arity: 1,
    argTypes: ['int'],
    functionName: "hexagonal-pixelate",
    documentation: "Hexagonal pixelation with cell radius n*5, filled with region's average color, edges 1px darker"
  },
  
  '7': {
    color: '#82E0AA',
    number: 34,
    fn: fn7,
    arity: 1,
    argTypes: ['int'],
    functionName: "hue-rectangles",
    documentation: "n rectangles placed diagonally, each filled with prev hue-rotated by i*(360/n)°"
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
    functionName: "self-blend",
    documentation: "Applies blend mode n%9 of prev with itself: multiply, screen, overlay, darken, lighten, color-dodge, color-burn, hard-light, soft-light"
  },
};
