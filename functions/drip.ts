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

function drip(ctx: FnContext): Image {
  const prev = getPrevImage(ctx);
  const gl = initWebGL(ctx.width, ctx.height);

  gl.activeTexture(gl.TEXTURE0);
  gl.bindTexture(gl.TEXTURE_2D, null);
  gl.activeTexture(gl.TEXTURE1);
  gl.bindTexture(gl.TEXTURE_2D, null);
  gl.activeTexture(gl.TEXTURE0);

  const n = 7;
  const numDrips = Math.max(1, n * 3);
  const dripStrength = Math.max(0.5, Math.min(n * 0.2 + 0.8, 2.0));
  const seed = ctx.images.length * 137.5;

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
    uniform float uStrength;
    uniform int uNumDrips;
    uniform float uSeed;
    varying vec2 vUV;
    
    #define MAX_DRIPS 100
    #define MAX_STEPS 64
    #define WALL_Z 0.0
    #define EPS 0.001
    
    float hash(float n) {
      return fract(sin(n) * 43758.5453123);
    }
    
    // Smooth min for organic blending
    float smin(float a, float b, float k) {
      float h = clamp(0.5 + 0.5 * (b - a) / k, 0.0, 1.0);
      return mix(b, a, h) - k * h * (1.0 - h);
    }
    
    // Simple sphere SDF
    float sdSphere(vec3 p, vec3 center, float r) {
      return length(p - center) - r;
    }
    
    // Scene SDF - randomly distributed clusters of 4 metaballs each
    float sceneSDF(vec3 p, int numClusters, float strength) {
      float d = 1000.0;
      
      for (int i = 0; i < MAX_DRIPS; i++) {
        if (i >= numClusters) break;
        
        float fi = float(i);
        
        // Random cluster center position across full screen (seeded by image count)
        float clusterX = (hash(fi * 127.1 + uSeed) * 2.0 - 1.0) * 1.5;
        float clusterY = (hash(fi * 311.7 + uSeed) * 2.0 - 1.0) * 1.2;
        
        // Cluster size
        float clusterSize = 0.15 + hash(fi * 74.3 + uSeed) * 0.15;
        
        // 4 balls per cluster (fixed to avoid dynamic loop issues)
        for (int j = 0; j < 4; j++) {
          float fj = float(j);
          float localSeed = fi * 100.0 + fj + uSeed;
          
          // Random offset from cluster center
          float offX = (hash(localSeed * 123.4) - 0.5) * clusterSize;
          float offY = (hash(localSeed * 234.5) - 0.5) * clusterSize * 1.5; // More vertical spread
          
          float x = clusterX + offX;
          float y = clusterY + offY;
          
          // Varying ball sizes within cluster
          float r = (0.06 + hash(localSeed * 345.6) * 0.08) * strength * 0.6;
          
          float z = 0.05 + r * 0.3;
          
          vec3 center = vec3(x, y, z);
          float sphere = sdSphere(p, center, r);
          
          // Smooth blend to merge balls in cluster
          d = smin(d, sphere, 0.12);
        }
      }
      
      return d;
    }
    
    // Calculate normal via gradient
    vec3 calcNormal(vec3 p, int numDrips, float strength) {
      vec2 e = vec2(0.001, 0.0);
      return normalize(vec3(
        sceneSDF(p + e.xyy, numDrips, strength) - sceneSDF(p - e.xyy, numDrips, strength),
        sceneSDF(p + e.yxy, numDrips, strength) - sceneSDF(p - e.yxy, numDrips, strength),
        sceneSDF(p + e.yyx, numDrips, strength) - sceneSDF(p - e.yyx, numDrips, strength)
      ));
    }
    
    void main() {
      vec2 uv = vUV;
      float aspect = uResolution.x / uResolution.y;
      
      // Ray origin and direction (orthographic-ish camera)
      vec3 ro = vec3((uv.x * 2.0 - 1.0) * aspect, uv.y * 2.0 - 1.0, 2.0);
      vec3 rd = vec3(0.0, 0.0, -1.0);
      
      // Raymarch to find metaball surface
      float t = 0.0;
      float d = 0.0;
      bool hit = false;
      vec3 p;
      
      for (int i = 0; i < MAX_STEPS; i++) {
        p = ro + rd * t;
        d = sceneSDF(p, uNumDrips, uStrength);
        
        if (d < EPS) {
          hit = true;
          break;
        }
        if (t > 4.0) break;
        
        t += d * 0.8;
      }
      
      // Sample wall texture
      vec4 wallColor = texture2D(uTexture, uv);
      
      if (hit) {
        vec3 normal = calcNormal(p, uNumDrips, uStrength);
        
        // Light from top (Y is flipped in screen coords)
        vec3 lightDir = normalize(vec3(0.0, -1.0, 0.6));
        vec3 viewDir = -rd;
        
        // Fresnel - edges reflect more
        float fresnel = pow(1.0 - max(dot(normal, viewDir), 0.0), 3.0);
        
        // Specular highlight - bright and sharp
        vec3 reflectDir = reflect(-lightDir, normal);
        float spec = pow(max(dot(reflectDir, viewDir), 0.0), 120.0);
        
        // Refraction - water has IOR ~1.33, so ratio is 1.0/1.33 ≈ 0.75
        vec3 refracted = refract(rd, normal, 0.75);
        vec2 refractUV = uv + refracted.xy * 0.06;
        refractUV = clamp(refractUV, 0.0, 1.0);
        vec4 refractColor = texture2D(uTexture, refractUV);
        
        // Clear water droplet - mostly shows refracted background
        vec3 dropletColor = refractColor.rgb;
        
        // Slight caustic brightening where light focuses
        float caustic = max(dot(normal, lightDir), 0.0);
        dropletColor *= 1.0 + caustic * 0.15;
        
        // Add crisp specular highlight
        dropletColor += vec3(1.0) * spec * 1.5;
        
        // Subtle rim highlight from fresnel
        dropletColor += vec3(1.0) * fresnel * 0.2;
        
        // Very subtle shadow directly under droplet on wall
        vec2 shadowUV = uv + vec2(0.01, 0.02);
        float shadowMask = smoothstep(0.05, 0.0, d) * 0.15;
        vec3 wallWithShadow = wallColor.rgb * (1.0 - shadowMask);
        
        // Droplet is almost fully transparent, just refracts
        gl_FragColor = vec4(dropletColor, 1.0);
      } else {
        gl_FragColor = wallColor;
      }
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
  gl.uniform1f(gl.getUniformLocation(program, "uStrength"), dripStrength);
  gl.uniform1i(gl.getUniformLocation(program, "uNumDrips"), numDrips);
  gl.uniform1f(gl.getUniformLocation(program, "uSeed"), seed);

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

export { drip };
