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

function spheres(ctx: FnContext): Image {
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
    
    vec3 renderSphere(vec2 uv, vec2 center, float radius, sampler2D tex, float rotation) {
      vec2 p = (uv - center) / radius;
      float d = length(p);
      
      if (d > 1.0) return vec3(-1.0);
      
      float z = sqrt(1.0 - d * d);
      vec3 normal = normalize(vec3(p.x, -p.y, z));
      vec3 lightDir1 = normalize(vec3(-0.6, -0.6, 1.0));
      vec3 lightDir2 = normalize(vec3(0.15, 0.8, -0.3));
      vec3 viewDir = vec3(0.0, 0.0, 1.0);
      vec3 reflectDir1 = reflect(-lightDir1, normal);
      vec3 reflectDir2 = reflect(-lightDir2, normal);
      
      float diffuse1 = max(dot(normal, lightDir1), 0.0);
      float diffuse2 = max(dot(normal, lightDir2), 0.0);
      float specular1 = pow(max(dot(viewDir, reflectDir1), 0.0), 4.0);
      float specular2 = pow(max(dot(viewDir, reflectDir2), 0.0), 8.0);
      float ambient = 0.6;
      float lighting = ambient + diffuse1 * 0.3 + diffuse2 * 0.15;
      
      float theta = atan(normal.x, normal.z) + rotation;
      float phi = acos(normal.y);
      vec2 texCoord = vec2(
        theta / (2.0 * 3.14159) + 0.5,
        1.0 - (phi - 0.3) / (3.14159 - 0.6)
      );
      
      vec3 color = texture2D(tex, texCoord).rgb;
      return color * lighting + vec3(1.0) * specular1 * 0.4 + vec3(1.0) * specular2 * 0.15;
    }
    
    void main() {
      vec2 uv = gl_FragCoord.xy / resolution;
      vec3 bg = texture2D(texture, vec2(uv.x, 1.0 - uv.y)).rgb;
      
      vec2 topRight = vec2(0.75, 0.75);
      vec2 bottomLeft = vec2(0.25, 0.25);
      float radius = 0.195;
      float rotationAmount = 0.3 * 2.0 * 3.14159;
      
      vec3 sphere1 = renderSphere(uv, topRight, radius, texture, -rotationAmount);
      vec3 sphere2 = renderSphere(uv, bottomLeft, radius, texture, rotationAmount);
      
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

  const textureLoc = gl.getUniformLocation(program, "texture");
  const resolutionLoc = gl.getUniformLocation(program, "resolution");

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

export { spheres };
