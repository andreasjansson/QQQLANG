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

function room(ctx: FnContext): Image {
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
          texCoord = vec2((hit.z + roomDepth) / (ro.z + roomDepth), hit.y / roomSize * 0.5 + 0.5);
        }
      }
      
      // Floor (y = -roomSize)
      t = intersectPlane(ro, rd, vec3(0.0, 1.0, 0.0), roomSize);
      if (t > 0.0 && t < tMin) {
        vec3 hit = ro + rd * t;
        if (hit.z > -roomDepth && hit.z < ro.z && abs(hit.x) < roomSize) {
          tMin = t;
          hitNormal = vec3(0.0, 1.0, 0.0);
          texCoord = vec2(hit.x / roomSize * 0.5 + 0.5, 1.0 - (hit.z + roomDepth) / (ro.z + roomDepth));
        }
      }
      
      // Ceiling (y = roomSize)
      t = intersectPlane(ro, rd, vec3(0.0, -1.0, 0.0), roomSize);
      if (t > 0.0 && t < tMin) {
        vec3 hit = ro + rd * t;
        if (hit.z > -roomDepth && hit.z < ro.z && abs(hit.x) < roomSize) {
          tMin = t;
          hitNormal = vec3(0.0, -1.0, 0.0);
          texCoord = vec2(hit.x / roomSize * 0.5 + 0.5, (hit.z + roomDepth) / (ro.z + roomDepth));
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

export { room };
