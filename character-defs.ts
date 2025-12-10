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
  
  const strength = Math.max(0.02, Math.min(0.03 + n / 22, 5));
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
};
