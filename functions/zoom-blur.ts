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

function zoomBlur(ctx: FnContext, n: number): Image {
  const prev = getPrevImage(ctx);
  const gl = initWebGL(ctx.width, ctx.height);

  gl.activeTexture(gl.TEXTURE0);
  gl.bindTexture(gl.TEXTURE_2D, null);

  // Match original: blurAmount = n * 4, used for sample count scaling
  const blurAmount = n * 4.0;

  const vertexShader = `
    attribute vec2 position;
    varying vec2 vUV;
    void main() {
      vUV = vec2(position.x * 0.5 + 0.5, position.y * 0.5 + 0.5);
      gl_Position = vec4(position, 0.0, 1.0);
    }
  `;

  const fragmentShader = `
    precision highp float;
    uniform sampler2D uTexture;
    uniform vec2 uResolution;
    uniform float uBlurAmount;
    varying vec2 vUV;
    
    #define MAX_SAMPLES 64
    
    void main() {
      vec2 center = vec2(0.5, 0.5);
      vec2 delta = vUV - center;
      
      // Account for aspect ratio
      float aspect = uResolution.x / uResolution.y;
      vec2 scaledDelta = vec2(delta.x * aspect, delta.y);
      
      float dist = length(scaledDelta);
      float maxR = length(vec2(0.5 * aspect, 0.5));
      float normDist = dist / maxR;
      
      // Match original: sharp center at 0.2 radius
      float sharpRadius = 0.2;
      
      if (normDist < sharpRadius) {
        gl_FragColor = texture2D(uTexture, vUV);
        return;
      }
      
      // Linear falloff from sharpRadius to edge (matching original)
      float blurStrength = min(1.0, (normDist - sharpRadius) / (1.0 - sharpRadius));
      
      // Number of samples scales with blur strength and amount
      int samples = int(max(1.0, blurStrength * uBlurAmount / 2.0));
      
      vec3 color = vec3(0.0);
      float sampleCount = 0.0;
      
      for (int i = 0; i < MAX_SAMPLES; i++) {
        if (i > samples) break;
        
        float t = float(i) / max(1.0, float(samples));
        // Sample from current position toward center
        // Original: sx = cx + dx * (1 - t * blurStrength * 0.5)
        // Which means: samplePos = center + delta * (1 - t * blurStrength * 0.5)
        vec2 samplePos = center + delta * (1.0 - t * blurStrength * 0.5);
        
        color += texture2D(uTexture, samplePos).rgb;
        sampleCount += 1.0;
      }
      
      gl_FragColor = vec4(color / sampleCount, 1.0);
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
  gl.uniform1f(gl.getUniformLocation(program, "uBlurAmount"), blurAmount);

  gl.viewport(0, 0, ctx.width, ctx.height);
  gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);

  const pixels = new Uint8ClampedArray(ctx.width * ctx.height * 4);
  gl.readPixels(0, 0, ctx.width, ctx.height, gl.RGBA, gl.UNSIGNED_BYTE, pixels);

  gl.deleteTexture(texture);
  gl.deleteBuffer(buffer);
  gl.deleteProgram(program);

  return { width: ctx.width, height: ctx.height, data: pixels };
}

export { zoomBlur };
