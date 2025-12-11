import { characterDefs, createSolidImage, createPlaceholderImage, getOldImage, Image, FnContext, CharDef, UPLOAD_CHAR } from './character-defs.js';

interface UploadedImageRef {
  type: 'uploaded';
  index: number;
}

interface ParsedSolidColor {
  type: 'solid';
  identifier: string;
  color: string;
}

interface ParsedUploadedImage {
  type: 'uploaded-image';
  identifier: string;
  uploadIndex: number;
}

interface ParsedFunction {
  type: 'function';
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

export const uploadedImages: Image[] = [];

export function clearUploadedImages(): void {
  uploadedImages.length = 0;
}

export function addUploadedImage(img: Image): number {
  const index = uploadedImages.length;
  uploadedImages.push(img);
  return index;
}

export function setUploadedImage(index: number, img: Image): void {
  uploadedImages[index] = img;
}

export function getUploadCount(program: string): number {
  const chars = [...program];
  return chars.filter(c => c === UPLOAD_CHAR).length;
}

function isUploadChar(char: string): boolean {
  return char === UPLOAD_CHAR;
}

interface ParseResult {
  ops: ParsedOp[];
  invalidUploadIndices: Set<number>;
}

function parseProgram(program: string): ParseResult {
  console.log(`\n=== PARSING: "${program}" ===`);
  const chars = [...program].filter(c => {
    const code = c.codePointAt(0)!;
    return (code > 32 && code < 127) || c === UPLOAD_CHAR;
  });
  console.log(`Filtered chars: "${chars.join('')}"`);
  
  if (chars.length === 0) {
    console.log('Empty program, returning []');
    return { ops: [], invalidUploadIndices: new Set() };
  }

  const ops: ParsedOp[] = [];
  const invalidUploadIndices = new Set<number>();
  let uploadIndexCounter = 0;
  
  const firstChar = chars[0];
  
  if (isUploadChar(firstChar)) {
    console.log(`[0] First char is upload -> uploaded image ${uploadIndexCounter}`);
    ops.push({
      type: 'uploaded-image',
      identifier: firstChar,
      uploadIndex: uploadIndexCounter++
    });
  } else {
    const firstDef = characterDefs[firstChar];
    const firstColor = firstDef ? firstDef.color : '#000000';
    
    console.log(`[0] First char '${firstChar}' -> solid color ${firstColor}`);
    ops.push({
      type: 'solid',
      identifier: firstChar,
      color: firstColor
    });
  }

  let i = 1;
  while (i < chars.length) {
    const char = chars[i];
    
    if (isUploadChar(char)) {
      console.log(`[${i}] '${char}' is upload char in function position -> INVALID, skipping`);
      invalidUploadIndices.add(uploadIndexCounter++);
      i++;
      continue;
    }
    
    const def = characterDefs[char];
    
    if (!def) {
      console.log(`[${i}] '${char}' undefined, skipping`);
      i++;
      continue;
    }

    const args: (number | string | UploadedImageRef)[] = [];
    let argsConsumed = 0;

    console.log(`[${i}] '${char}' -> ${def.functionName}, arity ${def.arity}`);

    for (let argIdx = 0; argIdx < def.arity; argIdx++) {
      const argType = def.argTypes[argIdx];
      let nextCharIdx = i + 1 + argsConsumed;
      
      // Skip any upload chars in non-index positions
      while (nextCharIdx < chars.length && isUploadChar(chars[nextCharIdx]) && argType !== 'index') {
        console.log(`  arg[${argIdx}] (${argType}): upload char invalid here -> SKIPPING`);
        invalidUploadIndices.add(uploadIndexCounter++);
        argsConsumed++;
        nextCharIdx = i + 1 + argsConsumed;
      }
      
      if (nextCharIdx < chars.length) {
        const argChar = chars[nextCharIdx];
        
        if (isUploadChar(argChar)) {
          // argType must be 'index' here (others were skipped above)
          args.push({ type: 'uploaded', index: uploadIndexCounter++ });
          console.log(`  arg[${argIdx}] (${argType}): upload char -> uploaded image`);
          argsConsumed++;
        } else {
          const argDef = characterDefs[argChar];
          
          if (argDef) {
            if (argType === 'int' || argType === 'index') {
              args.push(argDef.number);
              console.log(`  arg[${argIdx}] (${argType}): '${argChar}' -> ${argDef.number}`);
            } else {
              args.push(argDef.color);
              console.log(`  arg[${argIdx}] (${argType}): '${argChar}' -> ${argDef.color}`);
            }
            argsConsumed++;
          } else {
            if (argType === 'int' || argType === 'index') {
              args.push(def.number);
              console.log(`  arg[${argIdx}] (${argType}): '${argChar}' undefined, using default ${def.number}`);
            } else {
              args.push(def.color);
              console.log(`  arg[${argIdx}] (${argType}): '${argChar}' undefined, using default ${def.color}`);
            }
          }
        }
      } else {
        if (argType === 'int' || argType === 'index') {
          args.push(def.number);
          console.log(`  arg[${argIdx}] (${argType}): EOF, using default ${def.number}`);
        } else {
          args.push(def.color);
          console.log(`  arg[${argIdx}] (${argType}): EOF, using default ${def.color}`);
        }
      }
    }

    const endIndex = i + 1 + argsConsumed;
    const identifier = chars.slice(0, endIndex).join('');
    
    console.log(`  identifier: "${identifier}"`);
    
    ops.push({
      type: 'function',
      identifier,
      fnDef: def,
      args
    });
    
    i += 1 + argsConsumed;
  }

  console.log(`Parsed into ${ops.length} operations, ${invalidUploadIndices.size} invalid uploads`);
  return { ops, invalidUploadIndices };
}

const imageCache = new LRUCache<string, Image>(100);
let lastWidth = 0;
let lastHeight = 0;
let lastUploadCount = 0;

export function runProgram(program: string, width: number, height: number): Image[] {
  console.log(`\n=== EXECUTION: ${width}x${height} ===`);
  
  const currentUploadCount = uploadedImages.length;
  
  if (width !== lastWidth || height !== lastHeight || currentUploadCount !== lastUploadCount) {
    console.log(`Dimensions or uploads changed, clearing cache`);
    imageCache.clear();
    lastWidth = width;
    lastHeight = height;
    lastUploadCount = currentUploadCount;
  }

  const { ops } = parseProgram(program);
  
  if (ops.length === 0) {
    console.log('No operations, returning black image');
    return [createSolidImage(width, height, '#000000')];
  }

  const images: Image[] = [createSolidImage(width, height, '#000000')];
  let cacheHits = 0;
  let cacheMisses = 0;
  
  for (let opIdx = 0; opIdx < ops.length; opIdx++) {
    const op = ops[opIdx];
    console.log(`\n[Op ${opIdx}] identifier="${op.identifier}"`);
    
    const cached = imageCache.get(op.identifier);
    if (cached) {
      console.log(`  ✓ CACHE HIT`);
      images.push(cached);
      cacheHits++;
      continue;
    }

    console.log(`  ✗ CACHE MISS`);
    cacheMisses++;

    let result: Image;
    
    if (op.type === 'solid') {
      console.log(`  Creating solid image: ${op.color}`);
      result = createSolidImage(width, height, op.color);
    } else if (op.type === 'uploaded-image') {
      console.log(`  Using uploaded image ${op.uploadIndex}`);
      if (op.uploadIndex < uploadedImages.length) {
        result = uploadedImages[op.uploadIndex];
      } else {
        console.log(`  Upload ${op.uploadIndex} not found, using placeholder`);
        result = createPlaceholderImage(width, height);
      }
    } else {
      console.log(`  Executing function: ${op.fnDef.functionName} with args:`, op.args);
      const ctx: FnContext = {
        width,
        height,
        images: [...images],
        currentIndex: images.length,
      };
      
      const resolvedArgs = op.args.map((arg, idx) => {
        const argType = op.fnDef.argTypes[idx];
        if (argType === 'index') {
          if (typeof arg === 'object' && arg.type === 'uploaded') {
            if (arg.index < uploadedImages.length) {
              return uploadedImages[arg.index];
            } else {
              return createPlaceholderImage(width, height);
            }
          } else if (typeof arg === 'number') {
            return getOldImage(ctx, arg);
          }
        }
        return arg;
      });
      
      result = op.fnDef.fn(ctx, ...resolvedArgs);
    }
    
    images.push(result);
    imageCache.set(op.identifier, result);
    console.log(`  ✓ Cached result for "${op.identifier}"`);
  }

  console.log(`\n=== EXECUTION COMPLETE ===`);
  console.log(`Cache hits: ${cacheHits}, Cache misses: ${cacheMisses}`);
  console.log(`Total images: ${images.length}`);
  return images;
}

export function getFinalImage(program: string, width: number, height: number): Image {
  const images = runProgram(program, width, height);
  return images[images.length - 1];
}

export function getParsedOperations(program: string): ParsedOp[] {
  return parseProgram(program);
}

export function getExpectedNextType(program: string): 'function' | 'int' | 'color' | 'index' | 'initial' {
  if (!program || program.length === 0) {
    return 'initial';
  }
  
  const ops = parseProgram(program);
  if (ops.length === 0) {
    return 'initial';
  }
  
  const lastOp = ops[ops.length - 1];
  
  if (lastOp.type === 'solid' || lastOp.type === 'uploaded-image') {
    return 'function';
  }
  
  if (lastOp.type === 'function') {
    const def = lastOp.fnDef;
    const prevIdentifier = ops.length > 1 ? ops[ops.length - 2].identifier : '';
    const currentOpChars = [...lastOp.identifier.substring(prevIdentifier.length)];
    const argsProvided = currentOpChars.length - 1;
    
    if (argsProvided < def.arity) {
      return def.argTypes[argsProvided] as 'int' | 'color' | 'index';
    }
    
    return 'function';
  }
  
  return 'function';
}
