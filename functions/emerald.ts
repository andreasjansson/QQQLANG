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
  formatFunctionHelp,
  breakLigatures,
  wrapText,
  generateIntroPage,
  numToChar,
  generateCharacterRefLines,
  getPageChar,
  generateAllHelpPages,
  generateIndexPage,
  emeraldScene,
  emeraldRenderer,
  emeraldCamera,
  emeraldModel,
  emeraldModelLoaded,
  emeraldLoadPromise,
  emeraldComposer,
  initEmeraldScene,
} from "./helpers.js";
import * as THREE from "three";
import { RenderPass } from "three/examples/jsm/postprocessing/RenderPass.js";
import { UnrealBloomPass } from "three/examples/jsm/postprocessing/UnrealBloomPass.js";

function emerald(ctx: FnContext): Image {
  const prev = getPrevImage(ctx);

  if (!emeraldModelLoaded || !emeraldModel) {
    throw new Error(
      "Emerald model not loaded - await emeraldReady before rendering",
    );
  }

  initEmeraldScene(ctx.width, ctx.height);

  while (emeraldScene!.children.length > 0) {
    emeraldScene!.remove(emeraldScene!.children[0]);
  }

  // Create background texture - preserve original colors
  const bgTexture = new THREE.DataTexture(
    prev.data,
    prev.width,
    prev.height,
    THREE.RGBAFormat,
  );
  bgTexture.colorSpace = THREE.SRGBColorSpace;
  bgTexture.needsUpdate = true;
  bgTexture.flipY = true;
  emeraldScene!.background = bgTexture;

  // Create environment map for reflections
  const pmremGenerator = new THREE.PMREMGenerator(emeraldRenderer!);
  pmremGenerator.compileEquirectangularShader();
  const envRT = pmremGenerator.fromEquirectangular(bgTexture);
  emeraldScene!.environment = envRT.texture;

  // Low ambient for more contrast
  const ambientLight = new THREE.AmbientLight(0xffffff, 0.15);
  emeraldScene!.add(ambientLight);

  // Strong key light for highlights
  const keyLight = new THREE.DirectionalLight(0xffffff, 5.0);
  keyLight.position.set(5, 8, 10);
  emeraldScene!.add(keyLight);

  // Weak fill light - keeps shadows darker
  const fillLight = new THREE.DirectionalLight(0xeeffee, 0.8);
  fillLight.position.set(-5, 3, 8);
  emeraldScene!.add(fillLight);

  // Rim lights for edge highlights from multiple angles
  const rimLight = new THREE.DirectionalLight(0xffffff, 2.5);
  rimLight.position.set(0, -2, 8);
  emeraldScene!.add(rimLight);

  const rimLight2 = new THREE.DirectionalLight(0xffffff, 2.0);
  rimLight2.position.set(-6, 0, -2);
  emeraldScene!.add(rimLight2);

  const rimLight3 = new THREE.DirectionalLight(0xffffff, 2.0);
  rimLight3.position.set(6, 0, -2);
  emeraldScene!.add(rimLight3);

  // Seeded random for deterministic light positions based on image count
  const seed = ctx.images.length * 137.5;
  const hash = (n: number) => {
    const x = Math.sin(n + seed) * 43758.5453;
    return x - Math.floor(x);
  };

  // Dramatic point lights - fewer but more intense for contrast
  const numLights = 8;
  for (let i = 0; i < numLights; i++) {
    const angle = hash(i * 127.1) * Math.PI * 2;
    const elevation = hash(i * 311.7) * Math.PI * 0.5 + 0.3;
    const distance = 3 + hash(i * 74.3) * 5;

    const px = Math.cos(angle) * Math.cos(elevation) * distance;
    const py = Math.sin(elevation) * distance + 2;
    const pz = Math.sin(angle) * Math.cos(elevation) * distance + 4;

    const intensity = 15.0 + hash(i * 191.3) * 25.0;
    const light = new THREE.PointLight(0xffffff, intensity, 30);
    light.decay = 2;
    light.position.set(px, py, pz);
    emeraldScene!.add(light);
  }

  // Glass emerald material - pronounced edges with sheen
  const emeraldMaterial = new THREE.MeshPhysicalMaterial({
    color: new THREE.Color(0.3, 0.95, 0.5),
    metalness: 0.0,
    roughness: 0.0,
    transmission: 0.92,
    thickness: 0.4,
    ior: 1.3,
    envMapIntensity: 0.25,
    clearcoat: 1.0,
    clearcoatRoughness: 0.0,
    transparent: true,
    side: THREE.DoubleSide,
    flatShading: true,
    attenuationColor: new THREE.Color(0.0, 0.75, 0.25),
    attenuationDistance: 0.4,
    specularIntensity: 1.5,
    specularColor: new THREE.Color(1, 1, 1),
    reflectivity: 0.3,
    sheen: 0.5,
    sheenRoughness: 0.2,
    sheenColor: new THREE.Color(0.8, 1.0, 0.9),
  });

  // Corner positions extracted from the emerald geometry
  // Girdle corners (8 points around y ≈ 0.07)
  const girdleCorners = [
    [0.0, 0.064, -0.323], // front
    [0.227, 0.069, -0.226], // front-right
    [0.322, 0.063, 0.0], // right
    [0.224, 0.072, 0.228], // back-right
    [0.0, 0.064, 0.322], // back
    [-0.227, 0.07, 0.226], // back-left
    [-0.322, 0.07, 0.0], // left (inferred)
    [-0.225, 0.069, -0.227], // front-left
  ];

  // Crown corners (upper facet intersections around y ≈ 0.176)
  const crownCorners = [
    [-0.169, 0.176, -0.092],
    [-0.089, 0.176, 0.174],
    [0.169, 0.176, -0.092], // mirrored
    [0.089, 0.176, 0.174], // mirrored
    [0.0, 0.176, -0.18], // front center
    [0.0, 0.176, 0.18], // back center
    [-0.15, 0.176, 0.0], // left center
    [0.15, 0.176, 0.0], // right center
  ];

  // Create subtle sparkle sprite texture
  const sparkleCanvas = document.createElement("canvas");
  sparkleCanvas.width = 64;
  sparkleCanvas.height = 64;
  const sctx = sparkleCanvas.getContext("2d")!;
  const cx = 32,
    cy = 32;

  // Soft subtle glow
  const gradient = sctx.createRadialGradient(cx, cy, 0, cx, cy, 32);
  gradient.addColorStop(0, "rgba(255, 255, 255, 0.7)");
  gradient.addColorStop(0.15, "rgba(255, 255, 255, 0.3)");
  gradient.addColorStop(0.4, "rgba(255, 255, 255, 0.08)");
  gradient.addColorStop(1, "rgba(255, 255, 255, 0)");
  sctx.fillStyle = gradient;
  sctx.fillRect(0, 0, 64, 64);

  // Very subtle cross rays
  sctx.globalCompositeOperation = "lighter";
  const rayGradient = sctx.createLinearGradient(0, cy, 64, cy);
  rayGradient.addColorStop(0, "rgba(255,255,255,0)");
  rayGradient.addColorStop(0.35, "rgba(255,255,255,0.08)");
  rayGradient.addColorStop(0.5, "rgba(255,255,255,0.2)");
  rayGradient.addColorStop(0.65, "rgba(255,255,255,0.08)");
  rayGradient.addColorStop(1, "rgba(255,255,255,0)");
  sctx.fillStyle = rayGradient;
  sctx.fillRect(0, cy - 1, 64, 2);

  const rayGradientV = sctx.createLinearGradient(cx, 0, cx, 64);
  rayGradientV.addColorStop(0, "rgba(255,255,255,0)");
  rayGradientV.addColorStop(0.35, "rgba(255,255,255,0.08)");
  rayGradientV.addColorStop(0.5, "rgba(255,255,255,0.2)");
  rayGradientV.addColorStop(0.65, "rgba(255,255,255,0.08)");
  rayGradientV.addColorStop(1, "rgba(255,255,255,0)");
  sctx.fillStyle = rayGradientV;
  sctx.fillRect(cx - 1, 0, 2, 64);

  const sparkleTexture = new THREE.CanvasTexture(sparkleCanvas);

  const createSparkleMaterial = () =>
    new THREE.SpriteMaterial({
      map: sparkleTexture,
      color: 0xffffff,
      transparent: true,
      opacity: 0.6,
      blending: THREE.AdditiveBlending,
      depthTest: false,
      depthWrite: false,
    });

  const addEmerald = (
    x: number,
    y: number,
    scale: number,
    logGeometry: boolean = false,
  ) => {
    const gem = emeraldModel!.clone();

    gem.traverse((child) => {
      if (child instanceof THREE.Mesh) {
        const geom = child.geometry.clone();
        geom.computeVertexNormals();
        child.geometry = geom;
        child.material = emeraldMaterial;
        child.renderOrder = 1;

        // Log geometry info for the first emerald
        if (logGeometry) {
          const positions = geom.attributes.position;
          const normals = geom.attributes.normal;

          console.log("=== EMERALD GEOMETRY ===");
          console.log("Vertex count:", positions.count);
          console.log("Triangle count:", positions.count / 3);

          // Find unique vertices and their positions
          const uniqueVerts = new Map<
            string,
            { pos: number[]; count: number; indices: number[] }
          >();

          for (let i = 0; i < positions.count; i++) {
            const px = positions.getX(i).toFixed(3);
            const py = positions.getY(i).toFixed(3);
            const pz = positions.getZ(i).toFixed(3);
            const key = `${px},${py},${pz}`;

            if (!uniqueVerts.has(key)) {
              uniqueVerts.set(key, {
                pos: [parseFloat(px), parseFloat(py), parseFloat(pz)],
                count: 0,
                indices: [],
              });
            }
            uniqueVerts.get(key)!.count++;
            uniqueVerts.get(key)!.indices.push(i);
          }

          console.log("Unique vertex positions:", uniqueVerts.size);

          // Sort by how many triangles share this vertex (corners have more)
          const sorted = [...uniqueVerts.entries()].sort(
            (a, b) => b[1].count - a[1].count,
          );

          console.log("\nTop 20 most-shared vertices (likely corners):");
          sorted.slice(0, 20).forEach(([key, data], i) => {
            console.log(
              `  ${i + 1}. [${data.pos.join(", ")}] shared by ${data.count} triangles`,
            );
          });

          // Also log bounding box
          geom.computeBoundingBox();
          const bb = geom.boundingBox!;
          console.log("\nBounding box:");
          console.log(
            "  min:",
            bb.min.x.toFixed(3),
            bb.min.y.toFixed(3),
            bb.min.z.toFixed(3),
          );
          console.log(
            "  max:",
            bb.max.x.toFixed(3),
            bb.max.y.toFixed(3),
            bb.max.z.toFixed(3),
          );
        }
      }
    });

    gem.scale.setScalar(scale * 3.0);
    gem.position.set(x, y, 0);
    emeraldScene!.add(gem);

    // Add sparkle sprites at corner positions
    const allCorners = [...girdleCorners, ...crownCorners];
    const scaleFactor = scale * 3.0;

    allCorners.forEach((corner, i) => {
      // Only show sparkles on front-facing corners (positive z)
      if (corner[2] > 0) {
        const sprite = new THREE.Sprite(createSparkleMaterial());
        const sparkleSize = 0.03 + (i % 3) * 0.01;
        sprite.scale.set(
          sparkleSize * scaleFactor,
          sparkleSize * scaleFactor,
          1,
        );
        // Position at the corner, pushed forward a bit to be visible
        sprite.position.set(
          x + corner[0] * scaleFactor,
          y + corner[1] * scaleFactor,
          corner[2] * scaleFactor + 0.02,
        );
        sprite.renderOrder = 10;
        emeraldScene!.add(sprite);
      }
    });
  };

  addEmerald(0, 0, 1.0, false); // Main emerald
  addEmerald(-2.5, 0, 0.5);
  addEmerald(2.5, 0, 0.5);
  addEmerald(-1.5, 1.2, 0.35);
  addEmerald(1.5, 1.2, 0.35);
  addEmerald(-1.5, -1.2, 0.35);
  addEmerald(1.5, -1.2, 0.35);

  // Setup bloom passes
  emeraldComposer!.passes = [];
  const renderPass = new RenderPass(emeraldScene!, emeraldCamera!);
  emeraldComposer!.addPass(renderPass);

  const bloomPass = new UnrealBloomPass(
    new THREE.Vector2(ctx.width, ctx.height),
    0.3,
    0.15,
    0.97,
  );
  emeraldComposer!.addPass(bloomPass);

  // Render multiple times - transmission needs multiple passes to converge
  for (let i = 0; i < 6; i++) {
    emeraldComposer!.render();
  }

  const glContext = emeraldRenderer!.getContext();
  const pixels = new Uint8ClampedArray(ctx.width * ctx.height * 4);
  glContext.readPixels(
    0,
    0,
    ctx.width,
    ctx.height,
    glContext.RGBA,
    glContext.UNSIGNED_BYTE,
    pixels,
  );

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

  // Clean up
  bgTexture.dispose();
  envRT.texture.dispose();
  pmremGenerator.dispose();

  return { width: ctx.width, height: ctx.height, data: flipped };
}

export { emerald };
