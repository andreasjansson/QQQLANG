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

function shearRadial(
  ctx: FnContext,
  cxParam: number,
  cyParam: number,
  kParam: number,
): Image {
  const prev = getPrevImage(ctx);
  const gl = initWebGL(ctx.width, ctx.height);

  gl.activeTexture(gl.TEXTURE0);
  gl.bindTexture(gl.TEXTURE_2D, null);

  // Map parameters:
  // cx (1-68) -> center x offset -0.5 to 0.5
  // cy (1-68) -> center y offset -0.5 to 0.5
  // k (1-68) -> radial strength -4.0 to 4.0
  const cx = (cxParam - 34.5) / 67;
  const cy = (cyParam - 34.5) / 67;
  const k = ((kParam - 34.5) / 34.5) * 4.0;

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
    uniform float uCx;
    uniform float uCy;
    uniform float uK;
    varying vec2 vUV;
    
    // Mirror UV and return how many times we've mirrored (for fade calculation)
    vec2 mirrorUV(vec2 uv, out float dist) {
      // Track how far outside [0,1] we are
      float dx = 0.0;
      float dy = 0.0;
      
      if (uv.x < 0.0) dx = -uv.x;
      else if (uv.x > 1.0) dx = uv.x - 1.0;
      
      if (uv.y < 0.0) dy = -uv.y;
      else if (uv.y > 1.0) dy = uv.y - 1.0;
      
      dist = max(dx, dy);
      
      // Mirror by reflecting coordinates
      vec2 m = mod(uv, 2.0);
      if (m.x > 1.0) m.x = 2.0 - m.x;
      if (m.y > 1.0) m.y = 2.0 - m.y;
      if (m.x < 0.0) m.x = -m.x;
      if (m.y < 0.0) m.y = -m.y;
      
      return m;
    }
    
    void main() {
      // Convert to centered coords (-0.5 to 0.5)
      float x = vUV.x - 0.5;
      float y = vUV.y - 0.5;
      
      // Shear amount derived from center offset
      float s = uK * uCx;
      
      // Shear first
      float x_s = x + s * y;
      float y_s = y;
      
      // Radial centered at (cx, cy)
      float dx = x_s - uCx;
      float dy = y_s - uCy;
      float r2 = dx * dx + dy * dy;
      
      float denom = 1.0 + uK * r2;
      float xPrime = uCx + dx / denom;
      float yPrime = uCy + dy / denom;
      
      // Convert back to UV coords
      vec2 sampleUV = vec2(xPrime + 0.5, yPrime + 0.5);
      
      // Mirror and get distance outside bounds
      float dist;
      vec2 mirroredUV = mirrorUV(sampleUV, dist);
      
      // Flip upside down and sample the texture
      vec4 color = texture2D(uTexture, vec2(mirroredUV.x, 1.0 - mirroredUV.y));
      
      // Fade to black based on distance outside original bounds
      float fade = 1.0 - smoothstep(0.0, 0.5, dist);
      
      gl_FragColor = vec4(color.rgb * fade, 1.0);
    }
  `;

  const program = createShaderProgram(gl, vertexShader, fragmentShader);
  gl.useProgram(program);

  const vertices = new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]);
  const buffer = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
  gl.bufferData(gl.ARRAY_BUFFER, vertices, gl.STATIC_DRAW);

  const posLoc = gl.getAttribLocation(program, "position");
  gl.enableVertexAttribArray(posLoc);
  gl.vertexAttribPointer(posLoc, 2, gl.FLOAT, false, 0, 0);

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

  gl.uniform1i(gl.getUniformLocation(program, "uTexture"), 0);
  gl.uniform1f(gl.getUniformLocation(program, "uCx"), cx);
  gl.uniform1f(gl.getUniformLocation(program, "uCy"), cy);
  gl.uniform1f(gl.getUniformLocation(program, "uK"), k);

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

  gl.disableVertexAttribArray(posLoc);
  gl.deleteTexture(texture);
  gl.deleteBuffer(buffer);
  gl.deleteProgram(program);

  return { width: ctx.width, height: ctx.height, data: flipped };
}

export { shearRadial };
