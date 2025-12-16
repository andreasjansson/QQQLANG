import { chromium, type Page } from "playwright";
import * as fs from "fs";
import * as path from "path";
import { fileURLToPath } from "url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const PROJECT_ROOT = path.resolve(__dirname, "../..");
const ASSETS_DIR = path.join(PROJECT_ROOT, "assets");
const README_PATH = path.join(PROJECT_ROOT, "README.md");

const BASE_IMAGE_URL =
  "https://replicate.delivery/pbxt/NV0JLz4NfRmXPOkVrzjiASCfJvsea419i9agH2EuPJlHjG9h/0_1.webp";

// Import characterDefs directly
import { characterDefs, ChoiceType } from "../../character-defs.js";

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

function generateReadme(): string {
  const sortedChars = Object.entries(characterDefs).sort(
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
        if (arg.type instanceof ChoiceType) {
          const choices = arg.type.choices;
          const mappings = choices
            .map((choice, idx) => `${numToChar(idx + 1)}=${choice}`)
            .slice(0, 8)
            .join(", ");
          const suffix = choices.length > 8 ? ", ..." : "";
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
  console.log("Loading character definitions...");
  const chars = Object.keys(characterDefs);
  console.log(`Found ${chars.length} character definitions`);

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
  const sortedChars = Object.entries(characterDefs).sort(
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
  const readme = generateReadme();
  fs.writeFileSync(README_PATH, readme);
  console.log(`Wrote ${README_PATH}`);

  console.log("\nDone!");
}

main().catch((err) => {
  console.error("Error:", err);
  process.exit(1);
});
