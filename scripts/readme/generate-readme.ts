import { chromium, type Page } from "playwright";
import * as fs from "fs";
import * as path from "path";
import { fileURLToPath } from "url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const PROJECT_ROOT = path.resolve(__dirname, "../..");
const ASSETS_DIR = path.join(PROJECT_ROOT, "assets");
const README_PATH = path.join(PROJECT_ROOT, "README.md");
const CHARACTER_DEFS_PATH = path.join(PROJECT_ROOT, "character-defs.ts");

const BASE_IMAGE_URL =
  "https://replicate.delivery/pbxt/NV0JLz4NfRmXPOkVrzjiASCfJvsea419i9agH2EuPJlHjG9h/0_1.webp";

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
  
  console.log(`  Found ${entries.length} entries: ${entries.map(e => e.char).join(', ')}`);
  
  // Check for missing numbers
  const foundNumbers = new Set<number>();
  
  for (let i = 0; i < entries.length; i++) {
    const entry = entries[i];
    const nextStart = i + 1 < entries.length ? entries[i + 1].startIdx : defsContent.length;
    const entryContent = defsContent.substring(entry.startIdx, nextStart);
    
    const colorMatch = entryContent.match(/color:\s*["']([^"']+)["']/);
    const numberMatch = entryContent.match(/number:\s*(\d+)/);
    const functionNameMatch = entryContent.match(/functionName:\s*["']([^"']+)["']/);
    const docMatch = entryContent.match(/documentation:\s*\n?\s*["']([^"']+)["']/);
    const exampleMatch = entryContent.match(/example:\s*["'](.+?)["']/);
    
    if (!colorMatch || !numberMatch || !functionNameMatch || !docMatch) continue;
    
    const args: CharDef["args"] = [];
    const argsMatch = entryContent.match(/args:\s*\[([\s\S]*?)\],?\s*(?:functionName|$)/);
    if (argsMatch) {
      const argsContent = argsMatch[1];
      // Find each arg object
      const argObjRegex = /\{\s*type:\s*([\w(][^}]*?),\s*documentation:\s*["']([^"']+)["']\s*\}/g;
      let argMatch;
      while ((argMatch = argObjRegex.exec(argsContent)) !== null) {
        const typeStr = argMatch[1];
        const doc = argMatch[2];
        
        let choices: string[] | undefined;
        if (typeStr.startsWith("Choice(")) {
          // Extract choices - they may span multiple lines
          const choiceStart = argsContent.indexOf("Choice(", argMatch.index);
          if (choiceStart !== -1) {
            let parenCount = 0;
            let choiceEnd = choiceStart;
            for (let j = choiceStart; j < argsContent.length; j++) {
              if (argsContent[j] === '(') parenCount++;
              else if (argsContent[j] === ')') {
                parenCount--;
                if (parenCount === 0) {
                  choiceEnd = j + 1;
                  break;
                }
              }
            }
            const choiceContent = argsContent.substring(choiceStart + 7, choiceEnd - 1);
            choices = choiceContent
              .split(",")
              .map(s => s.trim().replace(/["'\n\s]/g, ""))
              .filter(s => s.length > 0);
          }
        }
        
        args.push({ type: { choices }, documentation: doc });
      }
    }
    
    let example = exampleMatch ? exampleMatch[1] : entry.char;
    example = example.replace(/\\\\/g, "\\");
    
    const num = parseInt(numberMatch[1]);
    foundNumbers.add(num);
    chars[entry.char] = {
      color: colorMatch[1],
      number: num,
      functionName: functionNameMatch[1],
      documentation: docMatch[1],
      example,
      args,
    };
  }
  
  // Report missing numbers
  for (let i = 1; i <= 68; i++) {
    if (!foundNumbers.has(i)) {
      console.log(`  Missing number ${i}`);
    }
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
  outputPath: string
): Promise<void> {
  await page.evaluate(() => {
    const input = document.getElementById("program-input") as HTMLInputElement;
    input.value = "";
    input.dispatchEvent(new Event("input", { bubbles: true }));
  });

  await page.waitForTimeout(200);

  const input = await page.$("#program-input");
  await input?.focus();

  await page.evaluate((url) => {
    const input = document.getElementById("program-input") as HTMLInputElement;
    input.focus();

    const clipboardData = new DataTransfer();
    clipboardData.setData("text/plain", url);

    const pasteEvent = new ClipboardEvent("paste", {
      bubbles: true,
      cancelable: true,
      clipboardData: clipboardData,
    });

    input.dispatchEvent(pasteEvent);
  }, BASE_IMAGE_URL);

  await page.waitForTimeout(2000);

  await page.evaluate((prog) => {
    const input = document.getElementById("program-input") as HTMLInputElement;
    input.value = input.value + prog;
    input.dispatchEvent(new Event("input", { bubbles: true }));
  }, program);

  await page.waitForTimeout(1000);

  const canvas = await page.$("#canvas");
  if (!canvas) throw new Error("Canvas not found");

  await canvas.screenshot({ path: outputPath });
  console.log(`  Captured: ${path.basename(outputPath)}`);
}

function generateReadme(chars: Record<string, CharDef>): string {
  const sortedChars = Object.entries(chars).sort(
    (a, b) => a[1].number - b[1].number
  );

  let readme = `![QQQLANG Logo](public/logo.png)

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

# Character Reference

`;

  for (const [char, def] of sortedChars) {
    const escapedChar = char === "\\" ? "\\\\" : char;
    const displayChar = char === "`" ? "\\`" : escapedChar;

    readme += `## \`${displayChar}\` (number ${def.number}, color <span style="color:${def.color}">${def.color}</span>)\n\n`;

    const safeFilename = def.number.toString().padStart(2, "0");
    readme += `![Example for ${displayChar}](assets/${safeFilename}-example.png)\n\n`;

    readme += `**Function:** \`${def.functionName}\` — ${def.documentation}\n\n`;

    if (def.args.length > 0) {
      readme += `**Arguments:**\n`;
      for (let i = 0; i < def.args.length; i++) {
        const arg = def.args[i];
        let argLine = `${i + 1}. ${arg.documentation}`;
        if (arg.type.choices && arg.type.choices.length > 0) {
          const mappings = arg.type.choices
            .map((choice, idx) => `${numToChar(idx + 1)}=${choice}`)
            .slice(0, 8)
            .join(", ");
          const suffix = arg.type.choices.length > 8 ? ", ..." : "";
          argLine += ` (${mappings}${suffix})`;
        }
        readme += `   ${argLine}\n`;
      }
      readme += "\n";
    }

    readme += `**Example:** \`${def.example}\`\n\n`;
    readme += `---\n\n`;
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

  console.log("Parsing character definitions...");
  const chars = parseCharacterDefs();
  console.log(`Found ${Object.keys(chars).length} character definitions`);

  if (!fs.existsSync(ASSETS_DIR)) {
    fs.mkdirSync(ASSETS_DIR, { recursive: true });
  }

  if (!skipImages) {
    console.log("\nLaunching browser...");
    const browser = await chromium.launch({ headless: true });
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

    await browser.close();
  }

  if (!singleChar) {
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
