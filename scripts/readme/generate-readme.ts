import { chromium, type Page } from "playwright";
import * as fs from "fs";
import * as path from "path";
import { fileURLToPath } from "url";
import sharp from "sharp";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const PROJECT_ROOT = path.resolve(__dirname, "../..");
const ASSETS_DIR = path.join(PROJECT_ROOT, "assets");
const README_PATH = path.join(PROJECT_ROOT, "README.md");
const CHARACTER_DEFS_PATH = path.join(PROJECT_ROOT, "character-defs.ts");

const UPLOAD_CHAR = String.fromCodePoint(0x2600); // ☀ U+2600
const UPLOAD_HASH = "Lh8lX-CEM_8ykW3QtaeIyw";
const SWATCH_SIZE = 16;

const GALLERY = [
  "AF`H+F++++++++++++++++++++++{4}HX~WDCBY5N$UA77",
  "☀CWc022HqJzLmt018fuUEUALBL++8.&&FF((((-I^XA,V#F.ANH-BFVL0((((D",
  "AFFGERE",
  "AVWNA.~}5YMDJA@FGIBJ8G-K~H:JJSX.<=|J<E||%%",
  "KL;SWW6}#ALLL+>{3>>>Q1D(..!=E#F0R2FBBXH",
  "FSEDSH2H%A8@",
  "WEXO6}655::{#FAYF-J3",
  "A5XWF}JD55T=(665",
  "1QQ(FX6JERHSQ3WF%%1B-WQ",
];

function hexToRgb(hex: string): { r: number; g: number; b: number } {
  const h = hex.replace("#", "");
  return {
    r: parseInt(h.substring(0, 2), 16),
    g: parseInt(h.substring(2, 4), 16),
    b: parseInt(h.substring(4, 6), 16),
  };
}

async function generateColorSwatch(
  color: string,
  outputPath: string
): Promise<void> {
  const rgb = hexToRgb(color);
  await sharp({
    create: {
      width: SWATCH_SIZE,
      height: SWATCH_SIZE,
      channels: 3,
      background: rgb,
    },
  })
    .png()
    .toFile(outputPath);
}

interface CharDef {
  color: string;
  number: number;
  functionName: string;
  documentation: string;
  example: string;
  args: { type: { choices?: string[] }; documentation: string }[];
}

function parseCharacterDefs(): Record<string, CharDef> {
  const content = fs.readFileSync(CHARACTER_DEFS_PATH, "utf-8");
  
  // Find the characterDefs object
  const startMatch = content.match(/export const characterDefs[^{]*\{/);
  if (!startMatch) throw new Error("Could not find characterDefs");
  
  const startIdx = startMatch.index! + startMatch[0].length;
  
  // Find matching closing brace, skipping string contents
  let braceCount = 1;
  let endIdx = startIdx;
  while (braceCount > 0 && endIdx < content.length) {
    const ch = content[endIdx];
    if (ch === '"' || ch === "'") {
      // Skip string
      const quote = ch;
      endIdx++;
      while (endIdx < content.length) {
        if (content[endIdx] === '\\') {
          endIdx += 2; // Skip escaped char
        } else if (content[endIdx] === quote) {
          endIdx++;
          break;
        } else {
          endIdx++;
        }
      }
    } else if (ch === '{') {
      braceCount++;
      endIdx++;
    } else if (ch === '}') {
      braceCount--;
      endIdx++;
    } else {
      endIdx++;
    }
  }
  
  const defsContent = content.substring(startIdx, endIdx - 1);
  const chars: Record<string, CharDef> = {};
  
  // Split by top-level entries - look for pattern like `  X: {` or `  "X": {` or `  $: {`
  // Handle: unquoted A-Z0-9$_, double-quoted single chars, single-quoted single chars, escaped backslash
  const entryRegex = /^  (?:([A-Z0-9$_])|\s*"(\\\\|.)"|'(.)'):\s*\{/gm;
  let match;
  const entries: { char: string; startIdx: number }[] = [];
  
  while ((match = entryRegex.exec(defsContent)) !== null) {
    let char = match[1] || match[2] || match[3];
    if (char === "\\\\") char = "\\";
    entries.push({ char, startIdx: match.index });
  }
  
  for (let i = 0; i < entries.length; i++) {
    const entry = entries[i];
    const nextStart = i + 1 < entries.length ? entries[i + 1].startIdx : defsContent.length;
    const entryContent = defsContent.substring(entry.startIdx, nextStart);
    
    const colorMatch = entryContent.match(/color:\s*["']([^"']+)["']/);
    const numberMatch = entryContent.match(/number:\s*(\d+)/);
    const functionNameMatch = entryContent.match(/functionName:\s*["']([^"']+)["']/);
    // Match documentation that comes after functionName (function-level doc, not arg docs)
    const fnNameIdx = entryContent.indexOf('functionName:');
    const docAfterFnName = fnNameIdx >= 0 ? entryContent.substring(fnNameIdx) : entryContent;
    const docMatch = docAfterFnName.match(/documentation:\s*\n?\s*["']([^"']+)["']/);
    // Match example with correct closing quote (double quote)
    const exampleMatch = entryContent.match(/example:\s*"([^"]+)"/);
    
    if (!colorMatch || !numberMatch || !functionNameMatch || !docMatch) continue;
    
    const args: CharDef["args"] = [];
    const argsMatch = entryContent.match(/args:\s*\[([\s\S]*?)\],?\s*functionName:/);
    if (argsMatch) {
      const argsContent = argsMatch[1];
      
      // Find each arg by looking for `documentation: "..."` patterns
      const docRegex = /documentation:\s*["']([^"']+)["']/g;
      let docMatch2;
      let argIndex = 0;
      
      while ((docMatch2 = docRegex.exec(argsContent)) !== null) {
        const doc = docMatch2[1];
        
        // Look backwards from this doc to find the type
        const beforeDoc = argsContent.substring(0, docMatch2.index);
        
        let choices: string[] | undefined;
        
        // Check if there's a Choice before this doc
        const lastChoiceIdx = beforeDoc.lastIndexOf("Choice(");
        const lastTypeIdx = beforeDoc.lastIndexOf("type:");
        
        if (lastChoiceIdx > lastTypeIdx - 20 && lastChoiceIdx !== -1) {
          // Extract the Choice content
          let parenCount = 0;
          let choiceEnd = lastChoiceIdx;
          for (let j = lastChoiceIdx; j < argsContent.length; j++) {
            if (argsContent[j] === '(') parenCount++;
            else if (argsContent[j] === ')') {
              parenCount--;
              if (parenCount === 0) {
                choiceEnd = j + 1;
                break;
              }
            }
          }
          const choiceContent = argsContent.substring(lastChoiceIdx + 7, choiceEnd - 1);
          choices = choiceContent
            .split(",")
            .map(s => s.trim().replace(/["'\n\s]/g, ""))
            .filter(s => s.length > 0);
        }
        
        args.push({ type: { choices }, documentation: doc });
        argIndex++;
      }
    }
    
    let example = exampleMatch ? exampleMatch[1] : entry.char;
    example = example.replace(/\\\\/g, "\\");
    
    chars[entry.char] = {
      color: colorMatch[1],
      number: parseInt(numberMatch[1]),
      functionName: functionNameMatch[1],
      documentation: docMatch[1],
      example,
      args,
    };
  }
  
  return chars;
}

function numToChar(num: number): string {
  if (num >= 1 && num <= 26)
    return String.fromCharCode("A".charCodeAt(0) + num - 1);
  if (num >= 27 && num <= 36)
    return String.fromCharCode("0".charCodeAt(0) + num - 27);
  const symbols = "<>^!\"#$%&'()*+,-./:;=?@[\\]_`{|}~";
  const idx = num - 37;
  if (idx >= 0 && idx < symbols.length) return symbols[idx];
  return "?";
}

async function captureExampleImage(
  page: Page,
  program: string,
  outputPath: string,
  useUploadPrefix: boolean = true
): Promise<void> {
  // Build URL with optional upload char + hash prefix
  const fullProgram = useUploadPrefix ? UPLOAD_CHAR + UPLOAD_HASH + program : program;
  // encodeURIComponent doesn't encode single quotes, so do it manually
  const encoded = encodeURIComponent(fullProgram).replace(/'/g, "%27");
  const url = `http://localhost:5173/?p=${encoded}`;
  
  await page.goto(url, { waitUntil: "networkidle" });
  await page.waitForTimeout(2000);

  const canvas = await page.$("#canvas");
  if (!canvas) throw new Error("Canvas not found");

  await canvas.screenshot({ path: outputPath });
  console.log(`  Captured: ${path.basename(outputPath)}`);
}

async function captureGalleryImage(
  browser: ReturnType<typeof chromium.launch> extends Promise<infer T> ? T : never,
  program: string,
  outputPath: string,
  debugMode: boolean
): Promise<void> {
  const context = await browser.newContext({
    viewport: { width: 1512, height: 797 },
    ignoreHTTPSErrors: true,
  });
  const page = await context.newPage();

  const encoded = encodeURIComponent(program).replace(/'/g, "%27");
  const url = `http://localhost:5173/?p=${encoded}`;
  
  await page.goto(url, { waitUntil: "networkidle" });
  // Wait for render + help text to disappear
  await page.waitForTimeout(4000);

  const canvas = await page.$("#canvas");
  if (!canvas) throw new Error("Canvas not found");

  await canvas.screenshot({ path: outputPath });
  console.log(`  Captured: ${path.basename(outputPath)}`);
  
  await context.close();
}

function generateReadme(chars: Record<string, CharDef>): string {
  const sortedChars = Object.entries(chars).sort(
    (a, b) => a[1].number - b[1].number
  );

  let readme = `[![QQQLANG Logo](public/logo.png)](https://qqqlang.com)

# QQQLANG: A syntax-free programming language for image synthesis

**https://qqqlang.com**

In QQQLANG, any string of visible uppercase ASCII characters is a valid program.

Each character has three properties:
- An integer ('A'=1, 'B'=2, [...], '}'=67, '~'=68)
- A color
- A function

Functions can take zero or more arguments. If a function takes arguments, the characters that follow are interpreted as arguments. Otherwise characters are interpreted as functions. The exception is the first character of the program string which sets an initial solid color.

For example, the program \`ABCD\` has the following interpretation:

- \`A\` sets the initial color to #78A10F
- \`B\` is the 'border' function that creates a circular gradient around the edges. It takes one argument, the border color.
- \`C\` becomes the argument to 'B', the color of 'C' is #FF6B35
- \`D\` is the 'drip' function, which creates a water drop effect. It takes no arguments.

If the program string ends before the last function has had arguments defined, it will use its own number and color as default arguments. For example, the programs \`AL\`, \`ALL\`, and \`ALLL\` are equivalent.

The question mark character \`?\` is also a function that displays help text. \`?1\` and \`??\` show the first page of help, and \`?A\`, \`?B\`, etc. show subsequent pages of help text.

Some functions take an image index as an argument, and uses that old image in some way. \`?#\` shows the history of images and the characters to use to retrieve each image.

---

# About

QQQLANG is built by me, Andreas Jansson, and is MIT licensed. The code is on github.com/andreasjansson/qqqlang.

QQQ is short for QQQEJOTTONO, a word that my three-year old son wrote on a label maker. He then went on to write fifty or so other words, until the label roll ran out. I thought it'd be nice if these labels could be treated like code.

So QQQLANG is really a language designed for three year olds. It's a Turing complete* stack-based language that can accept any string of characters as a valid program, because each character is either a function name or an argument, depending on context.

(* Turing complete because it includes a Rule 110 function)

There are 68 functions, some are normal image editing functions like '1' (colorize), and some are weird, like 'L' (3D Lissajous tubes) or 'V' (overlay another image in the stack in Voronoi patterns).

The output images are completely deterministic given the program string and canvas size. You can share a qqqlang.com URL to replicate and fork the image.

QQQLANG is both an image synthesis and editing language. You can upload an image as the starting image, or as arguments to functions that take image inputs. You can also paste images from the clipboard, or paste image URLs.

The language is complete and won't change (other than bug fixes). But anyone can fork the language and add new functions as a different language. It would be both fun and possible to build languages like QQQ-AUDIO, QQQ-VIDEO, QQQ-3D, etc.

---

# Gallery

<table>
<tr>
` + GALLERY.slice(0, 3).map((prog, i) => `<td><a href="https://qqqlang.com/?p=${encodeURIComponent(prog).replace(/'/g, "%27")}"><img src="assets/gallery-${i}.png" width="256"></a></td>`).join("\n") + `
</tr>
<tr>
` + GALLERY.slice(3, 6).map((prog, i) => `<td><a href="https://qqqlang.com/?p=${encodeURIComponent(prog).replace(/'/g, "%27")}"><img src="assets/gallery-${i + 3}.png" width="256"></a></td>`).join("\n") + `
</tr>
<tr>
` + GALLERY.slice(6, 9).map((prog, i) => `<td><a href="https://qqqlang.com/?p=${encodeURIComponent(prog).replace(/'/g, "%27")}"><img src="assets/gallery-${i + 6}.png" width="256"></a></td>`).join("\n") + `
</tr>
</table>

---

# Character Reference

`;

  for (const [char, def] of sortedChars) {
    const escapedChar = char === "\\" ? "\\\\" : char;
    const displayChar = char === "`" ? "\\`" : escapedChar;

    const safeFilename = def.number.toString().padStart(2, "0");
    const fullProgram = UPLOAD_CHAR + UPLOAD_HASH + def.example;
    const encodedProgram = encodeURIComponent(fullProgram).replace(/'/g, "%27");
    const qqqlangUrl = `https://qqqlang.com/?p=${encodedProgram}`;
    
    readme += `## \`${displayChar}\` — number ${def.number}, color ![${def.color}](assets/${safeFilename}-color.png)\n\n`;
    readme += `<a href="${qqqlangUrl}"><img align="right" width="384" src="assets/${safeFilename}-example.png"></a>\n\n`;

    readme += `**Function:** \`${def.functionName}\` — ${def.documentation}\n\n`;

    if (def.args.length > 0) {
      readme += `**Arguments:**\n`;
      for (let i = 0; i < def.args.length; i++) {
        const arg = def.args[i];
        let argLine = `${i + 1}. ${arg.documentation}`;
        if (arg.type.choices && arg.type.choices.length > 0) {
          const mappings = arg.type.choices
            .map((choice, idx) => `${numToChar(idx + 1)}=${choice}`)
            .join(", ");
          argLine += ` (${mappings})`;
        }
        readme += `   ${argLine}\n`;
      }
      readme += "\n";
    }

    readme += `<br clear="right">\n\n---\n\n`;
  }

  readme += `## License

MIT

## Author

Andreas Jansson ([@andreasjansson](https://github.com/andreasjansson))
`;

  return readme;
}

async function main() {
  // Parse --char X argument for single character mode
  const charArgIdx = process.argv.indexOf("--char");
  const singleChar = charArgIdx !== -1 ? process.argv[charArgIdx + 1] : null;
  const skipImages = process.argv.includes("--skip-images");
  const galleryOnly = process.argv.includes("--gallery-only");
  const skipGallery = process.argv.includes("--skip-gallery");
  const debugMode = process.argv.includes("--debug");

  console.log("Parsing character definitions...");
  const chars = parseCharacterDefs();
  console.log(`Found ${Object.keys(chars).length} character definitions`);

  if (!fs.existsSync(ASSETS_DIR)) {
    fs.mkdirSync(ASSETS_DIR, { recursive: true });
  }

  if (!galleryOnly) {
    console.log("\nGenerating color swatches...");
    for (const [char, def] of Object.entries(chars)) {
      const safeFilename = def.number.toString().padStart(2, "0");
      const swatchPath = path.join(ASSETS_DIR, `${safeFilename}-color.png`);
      await generateColorSwatch(def.color, swatchPath);
    }
    console.log(`Generated ${Object.keys(chars).length} color swatches`);
  }

  const needsBrowser = !skipImages || galleryOnly;
  const captureGallery = !skipImages && !skipGallery || galleryOnly;
  const captureExamples = !skipImages && !galleryOnly;

  if (needsBrowser && (captureGallery || captureExamples)) {
    console.log("\nLaunching browser...");
    const browser = await chromium.launch({ headless: !debugMode });
    const context = await browser.newContext({
      viewport: { width: 768, height: 512 },
      ignoreHTTPSErrors: true,
    });
    const page = await context.newPage();

    console.log("Navigating to localhost:5173...");
    await page.goto("http://localhost:5173/", {
      waitUntil: "networkidle",
    });

    await page.waitForTimeout(3000);

    if (captureGallery) {
      console.log("\nCapturing gallery images...");
      for (let i = 0; i < GALLERY.length; i++) {
        const program = GALLERY[i];
        const outputPath = path.join(ASSETS_DIR, `gallery-${i}.png`);
        console.log(`  Gallery ${i}: ${program.substring(0, 30)}...`);
        await captureExampleImage(page, program, outputPath, false);
      }
    }

    if (captureExamples) {
      console.log("\nCapturing example images...");
      
      let charsToProcess: [string, CharDef][];
      if (singleChar) {
        if (!chars[singleChar]) {
          console.error(`Character '${singleChar}' not found`);
          process.exit(1);
        }
        charsToProcess = [[singleChar, chars[singleChar]]];
      } else {
        charsToProcess = Object.entries(chars).sort(
          (a, b) => a[1].number - b[1].number
        );
      }

      for (const [char, def] of charsToProcess) {
        const safeFilename = def.number.toString().padStart(2, "0");
        const outputPath = path.join(ASSETS_DIR, `${safeFilename}-example.png`);

        console.log(`  Processing '${char}' (${def.functionName})...`);
        await captureExampleImage(page, def.example, outputPath);
      }
    }

    if (debugMode) {
      console.log("\nDebug mode: keeping browser open for 10 minutes...");
      await page.waitForTimeout(10 * 60 * 1000);
    }
    
    await browser.close();
  }

  if (!singleChar && !galleryOnly) {
    console.log("\nGenerating README.md...");
    const readme = generateReadme(chars);
    fs.writeFileSync(README_PATH, readme);
    console.log(`Wrote ${README_PATH}`);
  }

  console.log("\nDone!");
}

main().catch((err) => {
  console.error("Error:", err);
  process.exit(1);
});
