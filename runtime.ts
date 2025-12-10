import { characterDefs, createSolidImage, Image, FnContext, CharDef } from './character-defs.js';

interface ParsedSolidColor {
  type: 'solid';
  identifier: string;
  color: string;
}

interface ParsedFunction {
  type: 'function';
  identifier: string;
  fnDef: CharDef;
  args: (number | string)[];
}

type ParsedOp = ParsedSolidColor | ParsedFunction;

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
      this.cache.delete(firstKey);
    }
  }

  clear(): void {
    this.cache.clear();
  }
}

function parseProgram(program: string): ParsedOp[] {
  console.log(`\n=== PARSING: "${program}" ===`);
  const chars = program.split('').filter(c => c.charCodeAt(0) > 32 && c.charCodeAt(0) < 127);
  console.log(`Filtered chars: "${chars.join('')}"`);
  
  if (chars.length === 0) {
    console.log('Empty program, returning []');
    return [];
  }

  const ops: ParsedOp[] = [];
  
  const firstChar = chars[0];
  const firstDef = characterDefs[firstChar];
  const firstColor = firstDef ? firstDef.color : '#000000';
  
  console.log(`[0] First char '${firstChar}' -> solid color ${firstColor}`);
  ops.push({
    type: 'solid',
    identifier: firstChar,
    color: firstColor
  });

  let i = 1;
  while (i < chars.length) {
    const char = chars[i];
    const def = characterDefs[char];
    
    if (!def) {
      console.log(`[${i}] '${char}' undefined, skipping`);
      i++;
      continue;
    }

    const args: (number | string)[] = [];
    let argsConsumed = 0;

    console.log(`[${i}] '${char}' -> ${def.functionName}, arity ${def.arity}`);

    for (let argIdx = 0; argIdx < def.arity; argIdx++) {
      const argType = def.argTypes[argIdx];
      const nextCharIdx = i + 1 + argIdx;
      
      if (nextCharIdx < chars.length) {
        const argChar = chars[nextCharIdx];
        const argDef = characterDefs[argChar];
        
        if (argDef) {
          if (argType === 'int') {
            args.push(argDef.number);
            console.log(`  arg[${argIdx}] (${argType}): '${argChar}' -> ${argDef.number}`);
          } else {
            args.push(argDef.color);
            console.log(`  arg[${argIdx}] (${argType}): '${argChar}' -> ${argDef.color}`);
          }
          argsConsumed++;
        } else {
          if (argType === 'int') {
            args.push(def.number);
            console.log(`  arg[${argIdx}] (${argType}): '${argChar}' undefined, using default ${def.number}`);
          } else {
            args.push(def.color);
            console.log(`  arg[${argIdx}] (${argType}): '${argChar}' undefined, using default ${def.color}`);
          }
        }
      } else {
        if (argType === 'int') {
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

  console.log(`Parsed into ${ops.length} operations`);
  return ops;
}

const imageCache = new LRUCache<string, Image>(100);
let lastWidth = 0;
let lastHeight = 0;

export function runProgram(program: string, width: number, height: number): Image[] {
  if (width !== lastWidth || height !== lastHeight) {
    imageCache.clear();
    lastWidth = width;
    lastHeight = height;
  }

  const ops = parseProgram(program);
  
  if (ops.length === 0) {
    return [createSolidImage(width, height, '#000000')];
  }

  const images: Image[] = [];
  
  for (let opIdx = 0; opIdx < ops.length; opIdx++) {
    const op = ops[opIdx];
    
    const cached = imageCache.get(op.identifier);
    if (cached) {
      images.push(cached);
      continue;
    }

    let result: Image;
    
    if (op.type === 'solid') {
      result = createSolidImage(width, height, op.color);
    } else {
      const ctx: FnContext = {
        width,
        height,
        images: [...images],
        currentIndex: images.length,
      };
      
      result = op.fnDef.fn(ctx, ...op.args);
    }
    
    images.push(result);
    imageCache.set(op.identifier, result);
  }

  return images;
}

export function getFinalImage(program: string, width: number, height: number): Image {
  const images = runProgram(program, width, height);
  return images[images.length - 1];
}
