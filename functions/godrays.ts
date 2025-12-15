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

function godrays(ctx: FnContext): Image {
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

  const occlusionProgram = createShaderProgram(
    gl,
    vertexShader,
    occlusionFragShader,
  );
  const godrayProgram = createShaderProgram(gl, vertexShader, godrayFragShader);

  const vertices = new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]);
  const buffer = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
  gl.bufferData(gl.ARRAY_BUFFER, vertices, gl.STATIC_DRAW);

  const sceneTexture = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, sceneTexture);
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

  const occlusionTexture = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, occlusionTexture);
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

  const lightX = 0.5;
  const lightY = 0.5;
  const lightRadius = 0.08;

  // Pass 1: Render occlusion texture (light sources)
  gl.useProgram(occlusionProgram);

  const occPosLoc = gl.getAttribLocation(occlusionProgram, "position");
  gl.enableVertexAttribArray(occPosLoc);
  gl.vertexAttribPointer(occPosLoc, 2, gl.FLOAT, false, 0, 0);

  gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);
  gl.framebufferTexture2D(
    gl.FRAMEBUFFER,
    gl.COLOR_ATTACHMENT0,
    gl.TEXTURE_2D,
    occlusionTexture,
    0,
  );
  gl.viewport(0, 0, ctx.width, ctx.height);

  gl.activeTexture(gl.TEXTURE0);
  gl.bindTexture(gl.TEXTURE_2D, sceneTexture);
  gl.uniform1i(gl.getUniformLocation(occlusionProgram, "uTexture"), 0);
  gl.uniform2f(
    gl.getUniformLocation(occlusionProgram, "uLightPos"),
    lightX,
    lightY,
  );
  gl.uniform1f(
    gl.getUniformLocation(occlusionProgram, "uLightRadius"),
    lightRadius,
  );

  gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);

  // Pass 2: Apply god rays and combine with scene
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  gl.viewport(0, 0, ctx.width, ctx.height);

  gl.useProgram(godrayProgram);

  const grPosLoc = gl.getAttribLocation(godrayProgram, "position");
  gl.enableVertexAttribArray(grPosLoc);
  gl.vertexAttribPointer(grPosLoc, 2, gl.FLOAT, false, 0, 0);

  gl.activeTexture(gl.TEXTURE0);
  gl.bindTexture(gl.TEXTURE_2D, occlusionTexture);
  gl.uniform1i(gl.getUniformLocation(godrayProgram, "uOcclusionTexture"), 0);

  gl.activeTexture(gl.TEXTURE1);
  gl.bindTexture(gl.TEXTURE_2D, sceneTexture);
  gl.uniform1i(gl.getUniformLocation(godrayProgram, "uSceneTexture"), 1);

  gl.uniform2f(
    gl.getUniformLocation(godrayProgram, "uLightPos"),
    lightX,
    lightY,
  );
  gl.uniform1f(gl.getUniformLocation(godrayProgram, "uExposure"), 0.15);
  gl.uniform1f(gl.getUniformLocation(godrayProgram, "uDecay"), 0.96);
  gl.uniform1f(gl.getUniformLocation(godrayProgram, "uDensity"), 0.85);
  gl.uniform1f(gl.getUniformLocation(godrayProgram, "uWeight"), 0.4);

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

export { godrays };
