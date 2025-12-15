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

function lissajous(ctx: FnContext, old: Image, rot: number): Image {
  const prev = getPrevImage(ctx);
  const gl = initWebGL(ctx.width, ctx.height);

  gl.activeTexture(gl.TEXTURE0);
  gl.bindTexture(gl.TEXTURE_2D, null);
  gl.activeTexture(gl.TEXTURE1);
  gl.bindTexture(gl.TEXTURE_2D, null);
  gl.activeTexture(gl.TEXTURE0);

  const rotation = rot * 0.3;

  const vertexShader = `
    attribute vec3 aPosition;
    attribute vec3 aNormal;
    attribute vec2 aTexCoord;
    
    uniform mat4 uProjection;
    uniform mat4 uModelView;
    uniform mat3 uNormalMatrix;
    
    varying vec3 vNormal;
    varying vec3 vPosition;
    varying vec2 vTexCoord;
    
    void main() {
      vPosition = (uModelView * vec4(aPosition, 1.0)).xyz;
      vNormal = uNormalMatrix * aNormal;
      vTexCoord = aTexCoord;
      gl_Position = uProjection * vec4(vPosition, 1.0);
    }
  `;

  const fragmentShader = `
    precision highp float;
    
    uniform sampler2D uTexture;
    uniform vec3 uLightPos;
    uniform vec3 uLightPos2;
    
    varying vec3 vNormal;
    varying vec3 vPosition;
    varying vec2 vTexCoord;
    
    void main() {
      vec3 normal = normalize(vNormal);
      vec3 viewDir = normalize(-vPosition);
      
      // Light 1 - main light
      vec3 lightDir1 = normalize(uLightPos - vPosition);
      vec3 halfDir1 = normalize(lightDir1 + viewDir);
      float diff1 = max(dot(normal, lightDir1), 0.0);
      float spec1 = pow(max(dot(normal, halfDir1), 0.0), 32.0);
      
      // Light 2 - fill light
      vec3 lightDir2 = normalize(uLightPos2 - vPosition);
      vec3 halfDir2 = normalize(lightDir2 + viewDir);
      float diff2 = max(dot(normal, lightDir2), 0.0);
      float spec2 = pow(max(dot(normal, halfDir2), 0.0), 32.0);
      
      float ambient = 0.35;
      float diffuse = diff1 * 0.6 + diff2 * 0.35;
      float specular = (spec1 * 0.8 + spec2 * 0.4);
      
      vec2 tiledCoord = fract(vTexCoord);
      vec3 texColor = texture2D(uTexture, tiledCoord).rgb;
      vec3 color = texColor * (ambient + diffuse) + vec3(1.0) * specular;
      
      gl_FragColor = vec4(color, 1.0);
    }
  `;

  const bgVertShader = `
    attribute vec2 aPosition;
    varying vec2 vUV;
    void main() {
      vUV = aPosition * 0.5 + 0.5;
      gl_Position = vec4(aPosition, 0.999, 1.0);
    }
  `;

  const bgFragShader = `
    precision highp float;
    uniform sampler2D uBgTexture;
    varying vec2 vUV;
    void main() {
      gl_FragColor = texture2D(uBgTexture, vec2(vUV.x, 1.0 - vUV.y));
    }
  `;

  const tubeProgram = createShaderProgram(gl, vertexShader, fragmentShader);
  const bgProgram = createShaderProgram(gl, bgVertShader, bgFragShader);

  const prevTexture = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, prevTexture);
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

  const oldTexture = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, oldTexture);
  gl.texImage2D(
    gl.TEXTURE_2D,
    0,
    gl.RGBA,
    old.width,
    old.height,
    0,
    gl.RGBA,
    gl.UNSIGNED_BYTE,
    old.data,
  );
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);

  const positions: number[] = [];
  const normals: number[] = [];
  const texCoords: number[] = [];

  const a = 3,
    b = 2,
    c = 1;
  const tubeRadius = 0.08;
  const segments = 400;
  const radialSegments = 12;

  const cross = (a: number[], b: number[]): number[] => [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
  const normalize = (v: number[]): number[] => {
    const len = Math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2);
    if (len < 0.0001) return [1, 0, 0];
    return [v[0] / len, v[1] / len, v[2] / len];
  };
  const dot = (a: number[], b: number[]): number =>
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2];

  const getPoint = (t: number): number[] => [
    Math.sin(a * t) * 0.7,
    Math.sin(b * t) * 0.7,
    Math.sin(c * t) * 0.5,
  ];

  const getTangent = (t: number): number[] => {
    const eps = 0.0001;
    const p1 = getPoint(t - eps);
    const p2 = getPoint(t + eps);
    return normalize([p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2]]);
  };

  const frames: { point: number[]; normal: number[]; binormal: number[] }[] =
    [];

  let prevNormal = [1, 0, 0];
  const firstTan = getTangent(0);
  if (Math.abs(dot(prevNormal, firstTan)) > 0.9) {
    prevNormal = [0, 1, 0];
  }
  prevNormal = normalize(cross(cross(firstTan, prevNormal), firstTan));

  for (let i = 0; i <= segments; i++) {
    const t = (i / segments) * Math.PI * 2;
    const point = getPoint(t);
    const tangent = getTangent(t);

    let normal = normalize(cross(cross(tangent, prevNormal), tangent));
    const binormal = normalize(cross(tangent, normal));

    frames.push({ point, normal, binormal });
    prevNormal = normal;
  }

  const firstFrame = frames[0];
  const lastFrame = frames[segments];
  const twistAngle = Math.atan2(
    dot(lastFrame.normal, firstFrame.binormal),
    dot(lastFrame.normal, firstFrame.normal),
  );

  for (let i = 0; i <= segments; i++) {
    const correction = -twistAngle * (i / segments);
    const cos_c = Math.cos(correction),
      sin_c = Math.sin(correction);
    const f = frames[i];
    const newNormal = [
      f.normal[0] * cos_c + f.binormal[0] * sin_c,
      f.normal[1] * cos_c + f.binormal[1] * sin_c,
      f.normal[2] * cos_c + f.binormal[2] * sin_c,
    ];
    const newBinormal = [
      -f.normal[0] * sin_c + f.binormal[0] * cos_c,
      -f.normal[1] * sin_c + f.binormal[1] * cos_c,
      -f.normal[2] * sin_c + f.binormal[2] * cos_c,
    ];
    f.normal = newNormal;
    f.binormal = newBinormal;
  }

  for (let i = 0; i < segments; i++) {
    const f0 = frames[i];
    const f1 = frames[i + 1];

    for (let r = 0; r < radialSegments; r++) {
      const angle0 = (r / radialSegments) * Math.PI * 2;
      const angle1 = ((r + 1) / radialSegments) * Math.PI * 2;

      const cos0 = Math.cos(angle0),
        sin0 = Math.sin(angle0);
      const cos1 = Math.cos(angle1),
        sin1 = Math.sin(angle1);

      const v00 = [
        f0.point[0] +
          (f0.normal[0] * cos0 + f0.binormal[0] * sin0) * tubeRadius,
        f0.point[1] +
          (f0.normal[1] * cos0 + f0.binormal[1] * sin0) * tubeRadius,
        f0.point[2] +
          (f0.normal[2] * cos0 + f0.binormal[2] * sin0) * tubeRadius,
      ];
      const v01 = [
        f0.point[0] +
          (f0.normal[0] * cos1 + f0.binormal[0] * sin1) * tubeRadius,
        f0.point[1] +
          (f0.normal[1] * cos1 + f0.binormal[1] * sin1) * tubeRadius,
        f0.point[2] +
          (f0.normal[2] * cos1 + f0.binormal[2] * sin1) * tubeRadius,
      ];
      const v10 = [
        f1.point[0] +
          (f1.normal[0] * cos0 + f1.binormal[0] * sin0) * tubeRadius,
        f1.point[1] +
          (f1.normal[1] * cos0 + f1.binormal[1] * sin0) * tubeRadius,
        f1.point[2] +
          (f1.normal[2] * cos0 + f1.binormal[2] * sin0) * tubeRadius,
      ];
      const v11 = [
        f1.point[0] +
          (f1.normal[0] * cos1 + f1.binormal[0] * sin1) * tubeRadius,
        f1.point[1] +
          (f1.normal[1] * cos1 + f1.binormal[1] * sin1) * tubeRadius,
        f1.point[2] +
          (f1.normal[2] * cos1 + f1.binormal[2] * sin1) * tubeRadius,
      ];

      const n00 = [
        f0.normal[0] * cos0 + f0.binormal[0] * sin0,
        f0.normal[1] * cos0 + f0.binormal[1] * sin0,
        f0.normal[2] * cos0 + f0.binormal[2] * sin0,
      ];
      const n01 = [
        f0.normal[0] * cos1 + f0.binormal[0] * sin1,
        f0.normal[1] * cos1 + f0.binormal[1] * sin1,
        f0.normal[2] * cos1 + f0.binormal[2] * sin1,
      ];
      const n10 = [
        f1.normal[0] * cos0 + f1.binormal[0] * sin0,
        f1.normal[1] * cos0 + f1.binormal[1] * sin0,
        f1.normal[2] * cos0 + f1.binormal[2] * sin0,
      ];
      const n11 = [
        f1.normal[0] * cos1 + f1.binormal[0] * sin1,
        f1.normal[1] * cos1 + f1.binormal[1] * sin1,
        f1.normal[2] * cos1 + f1.binormal[2] * sin1,
      ];

      const u0 = (i / segments) * 3;
      const u1 = ((i + 1) / segments) * 3;
      const v0 = r / radialSegments;
      const v1 = (r + 1) / radialSegments;

      positions.push(...v00, ...v10, ...v11, ...v00, ...v11, ...v01);
      normals.push(...n00, ...n10, ...n11, ...n00, ...n11, ...n01);
      texCoords.push(u0, v0, u1, v0, u1, v1, u0, v0, u1, v1, u0, v1);
    }
  }

  gl.enable(gl.DEPTH_TEST);
  gl.viewport(0, 0, ctx.width, ctx.height);
  gl.clearColor(0, 0, 0, 1);
  gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

  gl.useProgram(bgProgram);
  gl.activeTexture(gl.TEXTURE0);
  gl.bindTexture(gl.TEXTURE_2D, prevTexture);
  gl.uniform1i(gl.getUniformLocation(bgProgram, "uBgTexture"), 0);

  const bgVerts = new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]);
  const bgBuffer = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, bgBuffer);
  gl.bufferData(gl.ARRAY_BUFFER, bgVerts, gl.STATIC_DRAW);
  const bgPosLoc = gl.getAttribLocation(bgProgram, "aPosition");
  gl.enableVertexAttribArray(bgPosLoc);
  gl.vertexAttribPointer(bgPosLoc, 2, gl.FLOAT, false, 0, 0);
  gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  gl.disableVertexAttribArray(bgPosLoc);

  gl.useProgram(tubeProgram);

  const posLoc = gl.getAttribLocation(tubeProgram, "aPosition");
  const normLoc = gl.getAttribLocation(tubeProgram, "aNormal");
  const texLoc = gl.getAttribLocation(tubeProgram, "aTexCoord");

  const posBuffer = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, posBuffer);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(positions), gl.STATIC_DRAW);

  const normBuffer = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, normBuffer);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(normals), gl.STATIC_DRAW);

  const texBuffer = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, texBuffer);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(texCoords), gl.STATIC_DRAW);

  gl.bindBuffer(gl.ARRAY_BUFFER, posBuffer);
  gl.enableVertexAttribArray(posLoc);
  gl.vertexAttribPointer(posLoc, 3, gl.FLOAT, false, 0, 0);

  gl.bindBuffer(gl.ARRAY_BUFFER, normBuffer);
  gl.enableVertexAttribArray(normLoc);
  gl.vertexAttribPointer(normLoc, 3, gl.FLOAT, false, 0, 0);

  gl.bindBuffer(gl.ARRAY_BUFFER, texBuffer);
  gl.enableVertexAttribArray(texLoc);
  gl.vertexAttribPointer(texLoc, 2, gl.FLOAT, false, 0, 0);

  const aspect = ctx.width / ctx.height;
  const fov = Math.PI / 3.5;
  const near = 0.1,
    far = 10.0;
  const f = 1.0 / Math.tan(fov / 2);
  const projection = new Float32Array([
    f / aspect,
    0,
    0,
    0,
    0,
    f,
    0,
    0,
    0,
    0,
    (far + near) / (near - far),
    -1,
    0,
    0,
    (2 * far * near) / (near - far),
    0,
  ]);

  const angleY = (-55 * Math.PI) / 180 + rotation;
  const angleX = 0.3;
  const cy = Math.cos(angleY),
    sy = Math.sin(angleY);
  const cx = Math.cos(angleX),
    sx = Math.sin(angleX);
  const modelView = new Float32Array([
    cy,
    sy * sx,
    sy * cx,
    0,
    0,
    cx,
    -sx,
    0,
    -sy,
    cy * sx,
    cy * cx,
    0,
    0,
    0,
    -2.2,
    1,
  ]);

  const normalMatrix = new Float32Array([
    cy,
    sy * sx,
    sy * cx,
    0,
    cx,
    -sx,
    -sy,
    cy * sx,
    cy * cx,
  ]);

  gl.uniformMatrix4fv(
    gl.getUniformLocation(tubeProgram, "uProjection"),
    false,
    projection,
  );
  gl.uniformMatrix4fv(
    gl.getUniformLocation(tubeProgram, "uModelView"),
    false,
    modelView,
  );
  gl.uniformMatrix3fv(
    gl.getUniformLocation(tubeProgram, "uNormalMatrix"),
    false,
    normalMatrix,
  );
  gl.uniform3f(gl.getUniformLocation(tubeProgram, "uLightPos"), 3.0, 3.0, 3.0);
  gl.uniform3f(
    gl.getUniformLocation(tubeProgram, "uLightPos2"),
    -2.0,
    1.0,
    2.0,
  );

  gl.activeTexture(gl.TEXTURE1);
  const tubeTexture = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, tubeTexture);
  gl.texImage2D(
    gl.TEXTURE_2D,
    0,
    gl.RGBA,
    old.width,
    old.height,
    0,
    gl.RGBA,
    gl.UNSIGNED_BYTE,
    old.data,
  );
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  gl.uniform1i(gl.getUniformLocation(tubeProgram, "uTexture"), 1);

  gl.drawArrays(gl.TRIANGLES, 0, positions.length / 3);

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

  gl.disable(gl.DEPTH_TEST);
  gl.disableVertexAttribArray(posLoc);
  gl.disableVertexAttribArray(normLoc);
  gl.disableVertexAttribArray(texLoc);
  gl.activeTexture(gl.TEXTURE0);
  gl.bindTexture(gl.TEXTURE_2D, null);
  gl.activeTexture(gl.TEXTURE1);
  gl.bindTexture(gl.TEXTURE_2D, null);
  gl.activeTexture(gl.TEXTURE0);
  gl.deleteTexture(tubeTexture);
  gl.deleteTexture(prevTexture);
  gl.deleteTexture(oldTexture);
  gl.deleteBuffer(bgBuffer);
  gl.deleteBuffer(posBuffer);
  gl.deleteBuffer(normBuffer);
  gl.deleteBuffer(texBuffer);
  gl.deleteProgram(tubeProgram);
  gl.deleteProgram(bgProgram);
  gl.useProgram(null);

  return { width: ctx.width, height: ctx.height, data: flipped };
}

export { lissajous };
