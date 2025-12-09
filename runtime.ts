import { characterDefs, createSolidImage, Image, FnContext, CharDef } from './character-defs';

export function runProgram(program: string, width: number, height: number): Image[] {
  const chars = program.split('').filter(c => c.charCodeAt(0) > 32 && c.charCodeAt(0) < 127);
  
  if (chars.length === 0) {
    return [createSolidImage(width, height, '#000000')];
  }

  const images: Image[] = [];
  let i = 0;

  const firstChar = chars[i];
  const firstDef = characterDefs[firstChar];
  if (firstDef) {
    images.push(createSolidImage(width, height, firstDef.color));
  } else {
    images.push(createSolidImage(width, height, '#000000'));
  }
  i++;

  while (i < chars.length) {
    const char = chars[i];
    const def = characterDefs[char];
    
    if (!def) {
      i++;
      continue;
    }

    const args: (number | string)[] = [];
    let argsConsumed = 0;

    for (let argIdx = 0; argIdx < def.arity; argIdx++) {
      const argType = def.argTypes[argIdx];
      const nextCharIdx = i + 1 + argIdx;
      
      if (nextCharIdx < chars.length) {
        const argChar = chars[nextCharIdx];
        const argDef = characterDefs[argChar];
        
        if (argDef) {
          if (argType === 'int') {
            args.push(argDef.number);
          } else {
            args.push(argDef.color);
          }
          argsConsumed++;
        } else {
          if (argType === 'int') {
            args.push(def.number);
          } else {
            args.push(def.color);
          }
        }
      } else {
        if (argType === 'int') {
          args.push(def.number);
        } else {
          args.push(def.color);
        }
      }
    }

    const ctx: FnContext = {
      width,
      height,
      images: [...images],
      currentIndex: images.length,
    };

    const result = def.fn(ctx, ...args);
    images.push(result);
    
    i += 1 + argsConsumed;
  }

  return images;
}

export function getFinalImage(program: string, width: number, height: number): Image {
  const images = runProgram(program, width, height);
  return images[images.length - 1];
}
