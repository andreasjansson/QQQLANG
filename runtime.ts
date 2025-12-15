import { characterDefs } from "./character-defs.js";
import type {
  Image,
  FnContext,
  CharDef,
  OpInfo,
  ArgType,
  IntType,
  ColorType,
  IndexType,
  ChoiceType,
  ArgDef,
} from "./functions/helpers.js";
import {
  createSolidImage,
  createPlaceholderImage,
  getOldImage,
  UPLOAD_CHAR,
  UPLOAD_COUNT,
  isIndexedUpload,
  isInvalidUpload,
  isAnyUpload,
  getUploadIndex,
  getUploadChar,
  getInvalidUploadChar,
} from "./functions/helpers.js";

interface UploadedImageRef {
  type: "uploaded";
  index: number; // This is now the upload character index (0-255), not positional
}

interface ParsedSolidColor {
  type: "solid";
  identifier: string;
  color: string;
}

interface ParsedUploadedImage {
  type: "uploaded-image";
  identifier: string;
  uploadIndex: number; // The upload character index (0-255)
}

interface ParsedFunction {
  type: "function";
  identifier: string;
  fnDef: CharDef;
  args: (number | string | UploadedImageRef)[];
}

type ParsedOp = ParsedSolidColor | ParsedUploadedImage | ParsedFunction;

class LRUCache<K, V> {
  private cache = new Map<K, V>();
  private maxSize: number;

  constructor(maxSize: number = 100) {
    this.maxSize = maxSize;
  }

  get(key: K): V | undefined {
    const value = this.cache.get(key);
    if (value !== undefined) {
      this.cache.delete(key);
      this.cache.set(key, value);
    }
    return value;
  }

  set(key: K, value: V): void {
    if (this.cache.has(key)) {
      this.cache.delete(key);
    }
    this.cache.set(key, value);
    if (this.cache.size > this.maxSize) {
      const firstKey = this.cache.keys().next().value;
      if (firstKey !== undefined) {
        this.cache.delete(firstKey);
      }
    }
  }

  clear(): void {
    this.cache.clear();
  }
}

// Uploaded images are now stored by their character index (0-255), not positionally
// This allows copy/paste to preserve image identity
const uploadedImages: Map<number, { blob: Blob; hash: string | null }> =
  new Map();
const uploadedImagesCache: Map<number, Image> = new Map();
let uploadedCacheWidth = 0;
let uploadedCacheHeight = 0;
let nextUploadIndex = 0; // Track next available index

export function clearUploadedImages(): void {
  uploadedImages.clear();
  uploadedImagesCache.clear();
  uploadedCacheWidth = 0;
  uploadedCacheHeight = 0;
  nextUploadIndex = 0;
}

// Add a new uploaded image and return its assigned index
export function addUploadedImage(
  blob: Blob,
  hash: string | null = null,
): number {
  const index = nextUploadIndex++;
  if (index >= UPLOAD_COUNT) {
    throw new Error(`Maximum upload count (${UPLOAD_COUNT}) exceeded`);
  }
  uploadedImages.set(index, { blob, hash });
  imageCache.clear();
  return index;
}

// Set an uploaded image at a specific index (used for URL loading and paste remapping)
export function setUploadedImage(
  index: number,
  blob: Blob,
  hash: string | null = null,
): void {
  if (index < 0 || index >= UPLOAD_COUNT) {
    throw new Error(`Upload index ${index} out of range [0, ${UPLOAD_COUNT})`);
  }
  uploadedImages.set(index, { blob, hash });
  uploadedImagesCache.delete(index);
  if (index >= nextUploadIndex) {
    nextUploadIndex = index + 1;
  }
  imageCache.clear();
}

// Get the hash for an uploaded image (for URL serialization)
export function getUploadedImageHash(index: number): string | null {
  return uploadedImages.get(index)?.hash ?? null;
}

// Get all upload indices that are currently in use
export function getUsedUploadIndices(): number[] {
  return Array.from(uploadedImages.keys()).sort((a, b) => a - b);
}

// Find the next available upload index
export function getNextAvailableIndex(): number {
  for (let i = 0; i < UPLOAD_COUNT; i++) {
    if (!uploadedImages.has(i)) {
      return i;
    }
  }
  throw new Error(`Maximum upload count (${UPLOAD_COUNT}) exceeded`);
}

// Check if an upload index has an associated image
export function hasUploadedImage(index: number): boolean {
  return uploadedImages.has(index);
}

// Get the blob for an uploaded image (for thumbnail generation)
export function getUploadedBlob(index: number): Blob | null {
  return uploadedImages.get(index)?.blob ?? null;
}

export function getUploadedImageCount(): number {
  return uploadedImages.size;
}

function loadBlobToImage(
  blob: Blob,
  width: number,
  height: number,
): Promise<Image> {
  return new Promise((resolve) => {
    const img = new window.Image();
    img.onload = () => {
      const tempCanvas = document.createElement("canvas");
      tempCanvas.width = width;
      tempCanvas.height = height;
      const tempCtx = tempCanvas.getContext("2d")!;
      tempCtx.drawImage(img, 0, 0, width, height);
      const imageData = tempCtx.getImageData(0, 0, width, height);
      URL.revokeObjectURL(img.src);
      resolve({
        width,
        height,
        data: new Uint8ClampedArray(imageData.data),
      });
    };
    img.onerror = () => {
      URL.revokeObjectURL(img.src);
      resolve(createPlaceholderImage(width, height));
    };
    img.src = URL.createObjectURL(blob);
  });
}

export async function preloadUploadedImages(
  width: number,
  height: number,
): Promise<void> {
  if (uploadedCacheWidth === width && uploadedCacheHeight === height) {
    let allCached = true;
    for (const index of uploadedImages.keys()) {
      if (!uploadedImagesCache.has(index)) {
        allCached = false;
        break;
      }
    }
    if (allCached) return;
  }

  if (uploadedCacheWidth !== width || uploadedCacheHeight !== height) {
    uploadedImagesCache.clear();
    uploadedCacheWidth = width;
    uploadedCacheHeight = height;
  }

  const promises = Array.from(uploadedImages.entries()).map(
    async ([index, source]) => {
      if (!uploadedImagesCache.has(index)) {
        uploadedImagesCache.set(
          index,
          await loadBlobToImage(source.blob, width, height),
        );
      }
    },
  );
  await Promise.all(promises);
}

export function getUploadedImage(
  index: number,
  width: number,
  height: number,
): Image {
  if (!uploadedImages.has(index)) {
    return createPlaceholderImage(width, height);
  }

  const cached = uploadedImagesCache.get(index);
  if (
    cached &&
    uploadedCacheWidth === width &&
    uploadedCacheHeight === height
  ) {
    return cached;
  }

  return createPlaceholderImage(width, height);
}

// Count indexed uploads in a program string
export function getUploadCount(program: string): number {
  const chars = [...program];
  return chars.filter((c) => isIndexedUpload(c)).length;
}

// Get all upload indices used in a program string
export function getUploadIndicesInProgram(program: string): number[] {
  const indices: number[] = [];
  for (const char of program) {
    const idx = getUploadIndex(char);
    if (idx !== null) {
      indices.push(idx);
    }
  }
  return indices;
}

interface ParseResult {
  ops: ParsedOp[];
  invalidUploadIndices: Set<number>; // Upload character indices (0-255) in invalid positions
}

// Check if a character is a valid program character
function isValidProgramChar(char: string): boolean {
  const code = char.codePointAt(0)!;
  const isAscii = code > 32 && code < 127;
  const isIdxUpload = isIndexedUpload(char);
  const isInvUpload = isInvalidUpload(char);
  const isUnassigned = char === UPLOAD_CHAR;
  return isAscii || isIdxUpload || isInvUpload || isUnassigned;
}

function parseProgram(program: string): ParseResult {
  const chars = [...program].filter(isValidProgramChar);

  if (chars.length === 0) {
    return { ops: [], invalidUploadIndices: new Set() };
  }

  const ops: ParsedOp[] = [];
  const invalidUploadIndices = new Set<number>();

  const firstChar = chars[0];
  const firstUploadIdx = getUploadIndex(firstChar);

  if (firstUploadIdx !== null) {
    ops.push({
      type: "uploaded-image",
      identifier: firstChar,
      uploadIndex: firstUploadIdx,
    });
  } else if (firstChar === UPLOAD_CHAR) {
    ops.push({
      type: "solid",
      identifier: firstChar,
      color: "#000000",
    });
  } else {
    const firstDef = characterDefs[firstChar];
    const firstColor = firstDef ? firstDef.color : "#000000";
    ops.push({
      type: "solid",
      identifier: firstChar,
      color: firstColor,
    });
  }

  let i = 1;
  while (i < chars.length) {
    const char = chars[i];
    const uploadIdx = getUploadIndex(char);

    if (uploadIdx !== null || char === UPLOAD_CHAR) {
      if (uploadIdx !== null) {
        invalidUploadIndices.add(uploadIdx);
      }
      i++;
      continue;
    }

    const def = characterDefs[char];

    if (!def) {
      i++;
      continue;
    }

    const args: (number | string | UploadedImageRef)[] = [];
    let argsConsumed = 0;

    for (let argIdx = 0; argIdx < def.args.length; argIdx++) {
      const argDef = def.args[argIdx];
      const argType = argDef.type;
      let nextCharIdx = i + 1 + argsConsumed;

      while (nextCharIdx < chars.length) {
        const nextChar = chars[nextCharIdx];
        const nextUploadIdx = getUploadIndex(nextChar);
        const isUpload = nextUploadIdx !== null || nextChar === UPLOAD_CHAR;

        if (isUpload && !(argType instanceof IndexType)) {
          if (nextUploadIdx !== null) {
            invalidUploadIndices.add(nextUploadIdx);
          }
          argsConsumed++;
          nextCharIdx = i + 1 + argsConsumed;
        } else {
          break;
        }
      }

      if (nextCharIdx < chars.length) {
        const argChar = chars[nextCharIdx];
        const argUploadIdx = getUploadIndex(argChar);

        if (argUploadIdx !== null) {
          args.push({ type: "uploaded", index: argUploadIdx });
          argsConsumed++;
        } else if (argChar === UPLOAD_CHAR) {
          if (
            argType instanceof IntType ||
            argType instanceof IndexType ||
            argType instanceof ChoiceType
          ) {
            args.push(def.number);
          } else {
            args.push(def.color);
          }
          argsConsumed++;
        } else {
          const charDef = characterDefs[argChar];

          if (charDef) {
            if (
              argType instanceof IntType ||
              argType instanceof IndexType ||
              argType instanceof ChoiceType
            ) {
              args.push(charDef.number);
            } else {
              args.push(charDef.color);
            }
            argsConsumed++;
          } else {
            if (
              argType instanceof IntType ||
              argType instanceof IndexType ||
              argType instanceof ChoiceType
            ) {
              args.push(def.number);
            } else {
              args.push(def.color);
            }
          }
        }
      } else {
        if (
          argType instanceof IntType ||
          argType instanceof IndexType ||
          argType instanceof ChoiceType
        ) {
          args.push(def.number);
        } else {
          args.push(def.color);
        }
      }
    }

    const endIndex = i + 1 + argsConsumed;
    const identifier = chars.slice(0, endIndex).join("");

    ops.push({
      type: "function",
      identifier,
      fnDef: def,
      args,
    });

    i += 1 + argsConsumed;
  }

  return { ops, invalidUploadIndices };
}

const imageCache = new LRUCache<string, Image>(100);
let lastWidth = 0;
let lastHeight = 0;
let lastUploadCount = 0;

export async function runProgram(
  program: string,
  width: number,
  height: number,
): Promise<Image[]> {
  const currentUploadCount = uploadedImages.size;

  if (
    width !== lastWidth ||
    height !== lastHeight ||
    currentUploadCount !== lastUploadCount
  ) {
    imageCache.clear();
    lastWidth = width;
    lastHeight = height;
    lastUploadCount = currentUploadCount;
  }

  const { ops } = parseProgram(program);

  if (ops.length === 0) {
    return [createSolidImage(width, height, "#000000")];
  }

  const images: Image[] = [createSolidImage(width, height, "#000000")];
  const opInfos: OpInfo[] = [{ identifier: "", type: "solid" }];

  for (let opIdx = 0; opIdx < ops.length; opIdx++) {
    const op = ops[opIdx];

    const cached = imageCache.get(op.identifier);
    if (cached) {
      images.push(cached);
      opInfos.push({
        identifier: op.identifier,
        type: op.type,
      });
      continue;
    }

    let result: Image;

    if (op.type === "solid") {
      result = createSolidImage(width, height, op.color);
    } else if (op.type === "uploaded-image") {
      result = getUploadedImage(op.uploadIndex, width, height);
    } else {
      const ctx: FnContext = {
        width,
        height,
        images: [...images],
        currentIndex: images.length,
        opInfos: [...opInfos],
      };

      const resolvedArgs = op.args.map((arg, idx) => {
        const argDef = op.fnDef.args[idx];
        const argType = argDef.type;
        if (argType instanceof IndexType) {
          if (typeof arg === "object" && arg.type === "uploaded") {
            return getUploadedImage(arg.index, width, height);
          } else if (typeof arg === "number") {
            return getOldImage(ctx, arg);
          }
        } else if (argType instanceof ChoiceType) {
          if (typeof arg === "number") {
            const choiceIndex = (arg - 1) % argType.choices.length;
            return argType.choices[choiceIndex];
          }
        }
        return arg;
      });

      const fnResult = op.fnDef.fn(ctx, ...resolvedArgs);
      result = fnResult instanceof Promise ? await fnResult : fnResult;
    }

    images.push(result);
    opInfos.push({
      identifier: op.identifier,
      type: op.type,
    });
    imageCache.set(op.identifier, result);
  }

  return images;
}

export async function getFinalImage(
  program: string,
  width: number,
  height: number,
): Promise<Image> {
  const images = await runProgram(program, width, height);
  return images[images.length - 1];
}

export function getParsedOperations(program: string): ParsedOp[] {
  return parseProgram(program).ops;
}

export function getInvalidUploadIndices(program: string): Set<number> {
  return parseProgram(program).invalidUploadIndices;
}

export function getExpectedNextType(
  program: string,
): "function" | "int" | "color" | "index" | "initial" {
  if (!program || program.length === 0) {
    return "initial";
  }

  const { ops } = parseProgram(program);
  if (ops.length === 0) {
    return "initial";
  }

  const lastOp = ops[ops.length - 1];

  if (lastOp.type === "solid" || lastOp.type === "uploaded-image") {
    return "function";
  }

  if (lastOp.type === "function") {
    const def = lastOp.fnDef;
    const prevIdentifier = ops.length > 1 ? ops[ops.length - 2].identifier : "";
    const currentOpChars = [
      ...lastOp.identifier.substring(prevIdentifier.length),
    ];
    const argsProvided = currentOpChars.length - 1;

    if (argsProvided < def.args.length) {
      const argType = def.args[argsProvided].type;
      if (argType instanceof IntType) return "int";
      if (argType instanceof ColorType) return "color";
      if (argType instanceof IndexType) return "index";
      if (argType instanceof ChoiceType) return "int";
    }

    return "function";
  }

  return "function";
}

export function getExpectedTypeAtPosition(
  program: string,
  cursorPosition: number,
): "function" | "int" | "color" | "index" | "initial" {
  const beforeCursor = program.substring(0, cursorPosition);
  return getExpectedNextType(beforeCursor);
}
