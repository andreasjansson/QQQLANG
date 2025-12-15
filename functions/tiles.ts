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
import * as THREE from "three";

function tiles(ctx: FnContext, n: number): Image {
  const prev = getPrevImage(ctx);

  const baseGridSize = 8;
  const heightMultiplier = 0.2 + (n / 68) * 2.0;
  const seed = ctx.images.length * 137.5;
  const hash = (i: number) => {
    const x = Math.sin(i + seed) * 43758.5453;
    return x - Math.floor(x);
  };

  const aspect = ctx.width / ctx.height;
  const approxCellSize = Math.min(ctx.width, ctx.height) / baseGridSize;
  const cols = Math.max(1, Math.round(ctx.width / approxCellSize));
  const rows = Math.max(1, Math.round(ctx.height / approxCellSize));

  const renderer = new THREE.WebGLRenderer({
    antialias: true,
    preserveDrawingBuffer: true,
  });
  renderer.setSize(ctx.width, ctx.height);
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.VSMShadowMap;

  const scene = new THREE.Scene();

  const bgTexture = new THREE.DataTexture(
    prev.data,
    prev.width,
    prev.height,
    THREE.RGBAFormat,
  );
  bgTexture.colorSpace = THREE.SRGBColorSpace;
  bgTexture.minFilter = THREE.LinearFilter;
  bgTexture.magFilter = THREE.LinearFilter;
  bgTexture.needsUpdate = true;
  scene.background = bgTexture;

  const fov = 50;
  const camera = new THREE.PerspectiveCamera(fov, aspect, 0.1, 100);
  const frustumHeight = 2;
  const frustumWidth = frustumHeight * aspect;
  const camZ = frustumHeight / (2 * Math.tan((fov * Math.PI) / 180 / 2));
  camera.position.set(0, 0, camZ);
  camera.lookAt(0, 0, 0);

  const ambient = new THREE.AmbientLight(0xffffff, 0.5);
  scene.add(ambient);

  const light = new THREE.DirectionalLight(0xffffff, 2);
  light.position.set(0.5, 0.5, 8);
  light.castShadow = true;
  light.shadow.mapSize.width = 1536;
  light.shadow.mapSize.height = 1536;
  const d = Math.max(frustumWidth, frustumHeight);
  light.shadow.camera.left = -d;
  light.shadow.camera.right = d;
  light.shadow.camera.top = d;
  light.shadow.camera.bottom = -d;
  light.shadow.camera.near = 0.1;
  light.shadow.camera.far = 20;
  light.shadow.radius = 2;
  light.shadow.bias = -0.0005;
  scene.add(light);

  const light2 = new THREE.DirectionalLight(0xffffff, 1.0);
  light2.position.set(-8, 2, 4);
  light2.castShadow = true;
  light2.shadow.mapSize.width = 1536;
  light2.shadow.mapSize.height = 1536;
  light2.shadow.camera.left = -d;
  light2.shadow.camera.right = d;
  light2.shadow.camera.top = d;
  light2.shadow.camera.bottom = -d;
  light2.shadow.camera.near = 0.1;
  light2.shadow.camera.far = 30;
  light2.shadow.radius = 2;
  light2.shadow.bias = -0.0005;
  scene.add(light2);

  const cellWidth = frustumWidth / cols;
  const cellHeight = frustumHeight / rows;

  for (let row = 0; row < rows; row++) {
    for (let col = 0; col < cols; col++) {
      const idx = row * cols + col;
      const depth = (0.001 + hash(idx * 127.1) * 0.4) * heightMultiplier;
      const cx = (col + 0.5) * cellWidth - frustumWidth / 2;
      const cy = (row + 0.5) * cellHeight - frustumHeight / 2;

      const geometry = new THREE.BoxGeometry(cellWidth, cellHeight, depth);

      const uvs = geometry.attributes.uv;
      const positions = geometry.attributes.position;
      const normals = geometry.attributes.normal;
      const texX0 = col / cols,
        texX1 = (col + 1) / cols;
      const texY0 = 1 - (row + 1) / rows,
        texY1 = 1 - row / rows;
      const halfW = cellWidth / 2,
        halfH = cellHeight / 2;

      for (let i = 0; i < uvs.count; i++) {
        const nx = normals.getX(i),
          ny = normals.getY(i),
          nz = normals.getZ(i);
        const py = positions.getY(i);
        const px = positions.getX(i);

        if (nz > 0.9) {
          // Top face - map to texture region
          uvs.setXY(i, px > 0 ? texX1 : texX0, py > 0 ? texY0 : texY1);
        } else if (nx > 0.9) {
          // Right side - use right edge column
          uvs.setXY(i, texX1, py > 0 ? texY0 : texY1);
        } else if (nx < -0.9) {
          // Left side - use left edge column
          uvs.setXY(i, texX0, py > 0 ? texY0 : texY1);
        } else if (ny > 0.9) {
          // Front side - use top edge row
          uvs.setXY(i, px > 0 ? texX1 : texX0, texY0);
        } else if (ny < -0.9) {
          // Back side - use bottom edge row
          uvs.setXY(i, px > 0 ? texX1 : texX0, texY1);
        }
      }

      const mat = new THREE.MeshStandardMaterial({
        map: bgTexture,
        roughness: 0.4,
      });

      const box = new THREE.Mesh(geometry, mat);
      box.position.set(cx, cy, depth / 2);
      box.castShadow = true;
      box.receiveShadow = true;
      scene.add(box);
    }
  }

  renderer.render(scene, camera);

  const gl = renderer.getContext();
  const pixels = new Uint8ClampedArray(ctx.width * ctx.height * 4);
  gl.readPixels(0, 0, ctx.width, ctx.height, gl.RGBA, gl.UNSIGNED_BYTE, pixels);

  const flipped = new Uint8ClampedArray(ctx.width * ctx.height * 4);
  for (let y = 0; y < ctx.height; y++) {
    for (let x = 0; x < ctx.width; x++) {
      const src = ((ctx.height - 1 - y) * ctx.width + x) * 4;
      const dst = (y * ctx.width + x) * 4;
      flipped[dst] = pixels[src];
      flipped[dst + 1] = pixels[src + 1];
      flipped[dst + 2] = pixels[src + 2];
      flipped[dst + 3] = pixels[src + 3];
    }
  }

  bgTexture.dispose();
  scene.traverse((obj) => {
    if (obj instanceof THREE.Mesh) {
      obj.geometry.dispose();
      if (Array.isArray(obj.material)) obj.material.forEach((m) => m.dispose());
      else obj.material.dispose();
    }
  });
  renderer.dispose();

  return { width: ctx.width, height: ctx.height, data: flipped };
}

export { tiles };
