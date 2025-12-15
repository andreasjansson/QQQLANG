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

function oilSlick(ctx: FnContext, n: number): Image {
  const prev = getPrevImage(ctx);
  const gl = initWebGL(ctx.width, ctx.height);

  gl.activeTexture(gl.TEXTURE0);
  gl.bindTexture(gl.TEXTURE_2D, null);
  gl.activeTexture(gl.TEXTURE1);
  gl.bindTexture(gl.TEXTURE_2D, null);
  gl.activeTexture(gl.TEXTURE0);

  const seed = ctx.images.length * 137.5 + n * 24.4;
  const depth = 1 + Math.floor(n / 4);
  const warpStrength = 0.05 + n * 0.025;
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
  gl.texImage2D(
    gl.TEXTURE_2D,
    0,
    gl.RGBA,
    prev.width,
    prev.height,
    0,
    gl.RGBA,
    gl.UNSIGNED_BYTE,
    prev.data,
  );
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);

  const vertices = new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]);
  const buffer = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
  gl.bufferData(gl.ARRAY_BUFFER, vertices, gl.STATIC_DRAW);

  const positionLoc = gl.getAttribLocation(program, "position");
  gl.enableVertexAttribArray(positionLoc);
  gl.vertexAttribPointer(positionLoc, 2, gl.FLOAT, false, 0, 0);

  gl.uniform1i(gl.getUniformLocation(program, "uTexture"), 0);
  gl.uniform2f(
    gl.getUniformLocation(program, "uResolution"),
    ctx.width,
    ctx.height,
  );
  gl.uniform1f(gl.getUniformLocation(program, "uSeed"), seed);
  gl.uniform1i(gl.getUniformLocation(program, "uDepth"), depth);
  gl.uniform1f(gl.getUniformLocation(program, "uWarpStrength"), warpStrength);
  gl.uniform1f(gl.getUniformLocation(program, "uPatternScale"), patternScale);

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

export { oilSlick };
