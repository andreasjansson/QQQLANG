import { chromium, type Browser, type Page } from "playwright";
import * as fs from "fs";
import * as path from "path";
import { fileURLToPath } from "url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const PROJECT_ROOT = path.resolve(__dirname, "../..");
const ASSETS_DIR = path.join(PROJECT_ROOT, "assets");
const CHARACTER_DEFS_PATH = path.join(PROJECT_ROOT, "character-defs.ts");
const README_PATH = path.join(PROJECT_ROOT, "README.md");

const BASE_IMAGE_URL =
  "https://replicate.delivery/pbxt/NV0JLz4NfRmXPOkVrzjiASCfJvsea419i9agH2EuPJlHjG9h/0_1.webp";

interface CharDef {
  color: string;
  number: number;
  functionName: string;
  documentation: string;
  example: string;
  args: ArgDef[];
}

interface ArgDef {
  type: string;
  documentation: string;
  choices?: string[];
}

function parseCharacterDefs(): Record<string, CharDef> {
  const content = fs.readFileSync(CHARACTER_DEFS_PATH, "utf-8");

  const chars: Record<string, CharDef> = {};

  // Find all top-level keys in characterDefs object
  // Match patterns like: 'X': {, "X": {, X: {, "\\": {, '"': {
  const keyPatterns = [
    /'"':\s*\{/g, // Single quote character: '"':
    /"\\\\?":\s*\{/g, // Backslash: "\\":
    /"([^"\\])"|'([^'\\])':\s*\{/g, // Other single chars in quotes
    /([A-Z0-9]):\s*\{/g, // Unquoted alphanumeric
  ];

  // Instead, let's find all character definitions by looking for the pattern
  // We'll scan for lines that look like character keys
  const lines = content.split("\n");
  let inCharacterDefs = false;
  let braceDepth = 0;
  let currentChar: string | null = null;
  let currentObjStart = -1;

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];

    if (line.includes("export const characterDefs")) {
      inCharacterDefs = true;
      continue;
    }

    if (!inCharacterDefs) continue;

    // Check for a new character key at depth 1
    if (braceDepth === 1) {
      // Match various key patterns
      let keyMatch =
        line.match(/^\s*"\\\\?":\s*\{/) || // "\\":
        line.match(/^\s*'"':\s*\{/) || // '"':
        line.match(/^\s*"([^"\\])":\s*\{/) || // "X":
        line.match(/^\s*'([^'\\])':\s*\{/) || // 'X':
        line.match(/^\s*([A-Z0-9$_]):\s*\{/); // X:

      if (keyMatch) {
        // Extract the character
        if (line.includes('"\\\\":') || line.includes('"\\":"')) {
          currentChar = "\\";
        } else if (line.includes("'\"':")) {
          currentChar = '"';
        } else {
          // Extract from the match
          const fullMatch = keyMatch[0];
          if (fullMatch.includes('"')) {
            const m = fullMatch.match(/"([^"\\])"/);
            currentChar = m ? m[1] : null;
          } else if (fullMatch.includes("'")) {
            const m = fullMatch.match(/'([^'\\])'/);
            currentChar = m ? m[1] : null;
          } else {
            const m = fullMatch.match(/([A-Z0-9$_]):/);
            currentChar = m ? m[1] : null;
          }
        }
        currentObjStart = i;
      }
    }

    // Track brace depth
    for (const c of line) {
      if (c === "{") braceDepth++;
      else if (c === "}") {
        braceDepth--;

        // If we just closed a character definition
        if (braceDepth === 1 && currentChar !== null) {
          const objContent = lines.slice(currentObjStart, i + 1).join("\n");

          const colorMatch = objContent.match(/color:\s*["']([^"']+)["']/);
          const numberMatch = objContent.match(/number:\s*(\d+)/);
          const functionNameMatch = objContent.match(
            /functionName:\s*["']([^"']+)["']/
          );
          const documentationMatch = objContent.match(
            /documentation:\s*\n?\s*["']([^"']+)["']/
          );
          const exampleMatch = objContent.match(
            /example:\s*["']([^"']+)["']/
          );

          if (colorMatch && numberMatch && functionNameMatch && documentationMatch) {
            const args: ArgDef[] = [];
            const argsMatch = objContent.match(/args:\s*\[([\s\S]*?)\]/);
            if (argsMatch) {
              const argsContent = argsMatch[1];
              const argRegex =
                /\{\s*(?:type:\s*([^,}]+),\s*documentation:\s*["']([^"']+)["']|documentation:\s*["']([^"']+)["'],\s*type:\s*([^,}]+))/g;
              let argMatch;
              while ((argMatch = argRegex.exec(argsContent)) !== null) {
                const typeStr = (argMatch[1] || argMatch[4] || "").trim();
                const doc = argMatch[2] || argMatch[3];

                let type = typeStr;
                let choices: string[] | undefined;

                if (typeStr.startsWith("Choice(")) {
                  type = "choice";
                  const fullChoiceMatch = argsContent
                    .substring(argMatch.index)
                    .match(/Choice\(([\s\S]*?)\)/);
                  if (fullChoiceMatch) {
                    choices = fullChoiceMatch[1]
                      .split(",")
                      .map((s) => s.trim().replace(/["'\n\s]/g, ""))
                      .filter((s) => s.length > 0);
                  }
                }

                args.push({ type, documentation: doc, choices });
              }
            }

            // Handle escaped backslash in example
            let example = exampleMatch ? exampleMatch[1] : currentChar;
            if (example.includes("\\\\")) {
              example = example.replace(/\\\\/g, "\\");
            }

            chars[currentChar] = {
              color: colorMatch[1],
              number: parseInt(numberMatch[1]),
              functionName: functionNameMatch[1],
              documentation: documentationMatch[1],
              example,
              args,
            };
          }

          currentChar = null;
        }

        if (braceDepth === 0) {
          inCharacterDefs = false;
        }
      }
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
  // Clear input and paste the base image URL to upload it
  await page.evaluate(() => {
    const input = document.getElementById("program-input") as HTMLInputElement;
    input.value = "";
    input.dispatchEvent(new Event("input", { bubbles: true }));
  });

  await page.waitForTimeout(200);

  // Focus the input and paste the image URL
  const input = await page.$("#program-input");
  await input?.focus();

  // Paste the URL - this should trigger the image upload
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

  // Wait for image to load
  await page.waitForTimeout(2000);

  // Now type the example program
  await page.evaluate((prog) => {
    const input = document.getElementById("program-input") as HTMLInputElement;
    // Append to existing value (which should have the upload char)
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
        if (arg.choices && arg.choices.length > 0) {
          const mappings = arg.choices
            .map((choice, idx) => `${numToChar(idx + 1)}=${choice}`)
            .slice(0, 8)
            .join(", ");
          const suffix = arg.choices.length > 8 ? ", ..." : "";
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
  console.log("Parsing character definitions...");
  const chars = parseCharacterDefs();
  console.log(`Found ${Object.keys(chars).length} character definitions`);

  // Debug: show what we parsed
  for (const [char, def] of Object.entries(chars).sort(
    (a, b) => a[1].number - b[1].number
  )) {
    console.log(`  ${def.number}: '${char}' -> ${def.functionName}, example: ${def.example}`);
  }

  if (!fs.existsSync(ASSETS_DIR)) {
    fs.mkdirSync(ASSETS_DIR, { recursive: true });
  }

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
  const sortedChars = Object.entries(chars).sort(
    (a, b) => a[1].number - b[1].number
  );

  for (const [char, def] of sortedChars) {
    const safeFilename = def.number.toString().padStart(2, "0");
    const outputPath = path.join(ASSETS_DIR, `${safeFilename}-example.png`);

    console.log(`  Processing '${char}' (${def.functionName})...`);
    await captureExampleImage(page, def.example, outputPath);
  }

  await browser.close();

  console.log("\nGenerating README.md...");
  const readme = generateReadme(chars);
  fs.writeFileSync(README_PATH, readme);
  console.log(`Wrote ${README_PATH}`);

  console.log("\nDone!");
}

main().catch((err) => {
  console.error("Error:", err);
  process.exit(1);
});
