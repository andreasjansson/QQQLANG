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

function blur(ctx: FnContext, n: number): Image {
  const prev = getPrevImage(ctx);
  const gl = initWebGL(ctx.width, ctx.height);

  gl.activeTexture(gl.TEXTURE0);
  gl.bindTexture(gl.TEXTURE_2D, null);

  // Blur radius based on n (1-68 maps to radius 1-50)
  const radius = Math.max(1, Math.min(Math.floor(n * 0.75) + 1, 50));

  const vertexShader = `
    attribute vec2 position;
    varying vec2 vUV;
    void main() {
      vUV = vec2(position.x * 0.5 + 0.5, position.y * 0.5 + 0.5);
      gl_Position = vec4(position, 0.0, 1.0);
    }
  `;

  // Two-pass Gaussian blur for efficiency
  const blurFragShader = `
    precision highp float;
    uniform sampler2D uTexture;
    uniform vec2 uResolution;
    uniform vec2 uDirection;
    uniform float uRadius;
    varying vec2 vUV;
    
    void main() {
      vec2 texelSize = 1.0 / uResolution;
      vec3 result = vec3(0.0);
      float weightSum = 0.0;
      
      // Gaussian weights approximation
      for (float i = -50.0; i <= 50.0; i += 1.0) {
        if (abs(i) > uRadius) continue;
        
        float weight = exp(-(i * i) / (2.0 * uRadius * uRadius / 4.0));
        vec2 offset = uDirection * texelSize * i;
        result += texture2D(uTexture, vUV + offset).rgb * weight;
        weightSum += weight;
      }
      
      gl_FragColor = vec4(result / weightSum, 1.0);
    }
  `;

  const program = createShaderProgram(gl, vertexShader, blurFragShader);

  // Create textures for two-pass blur
  const srcTexture = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, srcTexture);
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

  const tempTexture = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, tempTexture);
  gl.texImage2D(
    gl.TEXTURE_2D,
    0,
    gl.RGBA,
    ctx.width,
    ctx.height,
    0,
    gl.RGBA,
    gl.UNSIGNED_BYTE,
    null,
  );
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);

  const framebuffer = gl.createFramebuffer();

  const vertices = new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]);
  const buffer = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
  gl.bufferData(gl.ARRAY_BUFFER, vertices, gl.STATIC_DRAW);

  gl.useProgram(program);

  const positionLoc = gl.getAttribLocation(program, "position");
  gl.enableVertexAttribArray(positionLoc);
  gl.vertexAttribPointer(positionLoc, 2, gl.FLOAT, false, 0, 0);

  gl.uniform2f(
    gl.getUniformLocation(program, "uResolution"),
    ctx.width,
    ctx.height,
  );
  gl.uniform1f(gl.getUniformLocation(program, "uRadius"), radius);

  // Pass 1: Horizontal blur (src -> temp)
  gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);
  gl.framebufferTexture2D(
    gl.FRAMEBUFFER,
    gl.COLOR_ATTACHMENT0,
    gl.TEXTURE_2D,
    tempTexture,
    0,
  );
  gl.viewport(0, 0, ctx.width, ctx.height);

  gl.activeTexture(gl.TEXTURE0);
  gl.bindTexture(gl.TEXTURE_2D, srcTexture);
  gl.uniform1i(gl.getUniformLocation(program, "uTexture"), 0);
  gl.uniform2f(gl.getUniformLocation(program, "uDirection"), 1.0, 0.0);

  gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);

  // Pass 2: Vertical blur (temp -> screen)
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  gl.viewport(0, 0, ctx.width, ctx.height);

  gl.bindTexture(gl.TEXTURE_2D, tempTexture);
  gl.uniform2f(gl.getUniformLocation(program, "uDirection"), 0.0, 1.0);

  gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);

  const pixels = new Uint8ClampedArray(ctx.width * ctx.height * 4);
  gl.readPixels(0, 0, ctx.width, ctx.height, gl.RGBA, gl.UNSIGNED_BYTE, pixels);

  gl.deleteTexture(srcTexture);
  gl.deleteTexture(tempTexture);
  gl.deleteFramebuffer(framebuffer);
  gl.deleteBuffer(buffer);
  gl.deleteProgram(program);

  return { width: ctx.width, height: ctx.height, data: pixels };
}

export { blur };
