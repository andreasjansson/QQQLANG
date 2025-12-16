import {
  FnContext,
  Image,
  createSolidImage,
  CharDef,
  ChoiceType,
} from "./helpers.js";
import { characterDefs } from "../character-defs.js";

function breakLigatures(text: string): string {
  return text
    .replace(/ff/g, "f\u200Cf")
    .replace(/fi/g, "f\u200Ci")
    .replace(/fl/g, "f\u200Cl")
    .replace(/ffi/g, "f\u200Cf\u200Ci")
    .replace(/ffl/g, "f\u200Cf\u200Cl");
}

function wrapText(text: string, maxWidth: number): string[] {
  const lines: string[] = [];
  const paragraphs = text.split("\n");

  for (const para of paragraphs) {
    if (para.length === 0) {
      lines.push("");
      continue;
    }

    const words = para.split(" ");
    let currentLine = "";

    for (const word of words) {
      if (currentLine.length === 0) {
        currentLine = word;
      } else if (currentLine.length + 1 + word.length <= maxWidth) {
        currentLine += " " + word;
      } else {
        lines.push(currentLine);
        currentLine = word;
      }
    }

    if (currentLine.length > 0) {
      lines.push(currentLine);
    }
  }

  return lines;
}

export const introText = `QQQLANG: A syntax-free programming language for image synthesis

In QQQLANG, any string of visible uppercase ascii characters is a valid program.

Each character has three properties:
* An integer ('A'=1, 'B'=2, [...], '}'=67, '~'=68)
* A color
* A function

Functions can take zero or more arguments. If a function takes arguments, the characters that follow are interpreted as arguments. Otherwise characters are interpreted as functions. The exception is the first character of the program string which sets an initial solid color.

For example, the program 'ABCD' has the following interpretation:

* 'A' sets the intial color to #78A10F
* 'B' is the 'border' function that creates a circular gradient around the edges. It takes one argument, the border color.
* 'C' becomes the argument to 'B', the color of 'C' is #FF6B35
* 'D' is the 'drip' function, which creates a water drop effect. It takes no arguments.

If the program string ends before the last function has had arguments defined, it will use its own number and color as default arguments. For example, the programs 'AL', 'ALL', and 'ALLL' are equivalent.

The question mark character '?' is also a function that displays help text. '?1' and '??' show the first page of help, and '?A', '?B', etc. show subsequent pages of help text.

Some functions take an image index as an argument, and uses that old image in some way. '?#' shows the history of images the the characters to use to retrieve each image.
`;

export const aboutText = `=== ABOUT QQQLANG ===

QQQLANG is built by me, Andreas Jansson, and is MIT licensed. The code is on github.com/andreasjansson/qqqlang.

QQQ is short for QQQEJOTTONO, a word that my three-year old son wrote on a label maker. He then went on to write fifty or so other words, until the label roll ran out. I thought it'd be nice if these labels could be treated like code.

So QQQLANG is really a language designed for three year olds. It's a Turing complete* stack-based language that can accept any string of characters as a valid program, because each character is either a function name or an argument, depending on context.

(* Turing complete because it includes a Rule 110 function)

There are 68 functions, some are normal image editing functions like '1' (colorize), and some are weird, like 'L' (3D Lissajous tubes) or 'V' (overlay another image in the stack in Voronoi patterns).

The output images are completely deterministic given the program string and canvas size. You can share a qqqlang.com URL to replicate and fork the image.

QQQLANG is both an image synthesis and editing language. You can upload an image as the starting image, or as an argument to functions that take image inputs. You can also paste images from the clipboard, or paste image URLs.

This project is finished now and the language won't change (other than bug fixes). But anyone can fork the language and add new functions as a new language. It would be both fun and possible to build languages like QQQ-AUDIO, QQQ-VIDEO, QQQ-3D, etc.
`;

function generateIntroPage(charsPerLine: number): string[] {
  return wrapText(introText, charsPerLine);
}

function generateAboutPage(charsPerLine: number): string[] {
  return wrapText(aboutText, charsPerLine);
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

export function formatFunctionHelp(
  char: string,
  def: CharDef,
  charsPerLine: number = 80,
): string[] {
  const lines: string[] = [];

  const firstLine = `${char} ${def.functionName} — ${def.documentation}`;
  lines.push(...wrapText(firstLine, charsPerLine));

  for (let i = 0; i < def.args.length; i++) {
    const arg = def.args[i];
    let argLine = `(${i + 1}) ${arg.documentation}`;

    if (arg.type instanceof ChoiceType) {
      const choices = arg.type.choices;
      const mappings = choices
        .map((choice, idx) => `${numToChar(idx + 1)}=${choice}`)
        .join(", ");
      argLine += ` (${mappings})`;
    }

    lines.push(...wrapText(argLine, charsPerLine));
  }

  return lines;
}

function generateCharacterRefLines(
  char: string,
  def: CharDef,
  charsPerLine: number,
): string[] {
  return formatFunctionHelp(char, def, charsPerLine);
}

function getPageChar(pageNum: number): string {
  if (pageNum <= 0) return "?";
  if (pageNum === 1) return "?";
  if (pageNum <= 26)
    return String.fromCharCode("A".charCodeAt(0) + pageNum - 1);
  return "?";
}

interface HelpPagesResult {
  pages: string[][];
  introPageCount: number;
  aboutPageCount: number;
  refPageCount: number;
}

function generateAllHelpPages(
  charsPerLine: number,
  linesPerPage: number,
  defs: Record<string, CharDef>,
): HelpPagesResult {
  const pages: string[][] = [];

  const introLines = generateIntroPage(charsPerLine);

  let introPage: string[] = [];
  for (let i = 0; i < introLines.length; i++) {
    if (introPage.length >= linesPerPage - 2) {
      pages.push(introPage);
      introPage = [];
    }
    introPage.push(introLines[i]);
  }
  if (introPage.length > 0) {
    pages.push(introPage);
  }

  const introPageCount = pages.length;

  const aboutLines = generateAboutPage(charsPerLine);

  let aboutPage: string[] = [];
  for (let i = 0; i < aboutLines.length; i++) {
    if (aboutPage.length >= linesPerPage - 2) {
      pages.push(aboutPage);
      aboutPage = [];
    }
    aboutPage.push(aboutLines[i]);
  }
  if (aboutPage.length > 0) {
    pages.push(aboutPage);
  }

  const aboutPageCount = pages.length - introPageCount;

  const chars = Object.keys(defs).sort(
    (a, b) => defs[a].number - defs[b].number,
  );

  let currentPage: string[] = [];
  currentPage.push("=== CHARACTER REFERENCE ===");
  currentPage.push("");
  let linesUsed = 2;

  for (const char of chars) {
    const def = defs[char];
    const charLines = generateCharacterRefLines(char, def, charsPerLine);

    if (linesUsed + charLines.length + 1 > linesPerPage - 2) {
      pages.push(currentPage);
      currentPage = [];
      currentPage.push("=== CHARACTER REFERENCE (continued) ===");
      currentPage.push("");
      linesUsed = 2;
    }

    currentPage.push(...charLines);
    currentPage.push("");
    linesUsed += charLines.length + 1;
  }

  if (currentPage.length > 2) {
    pages.push(currentPage);
  }

  const refPageCount = pages.length - introPageCount - aboutPageCount;
  const totalPages = pages.length;

  for (let i = 0; i < pages.length; i++) {
    const pageNum = i + 1;
    const nextPageChar = getPageChar(pageNum + 1);
    pages[i].push("");
    if (pageNum < totalPages) {
      pages[i].push(
        `[Page ${pageNum}/${totalPages}, type '?${nextPageChar}' for next page]`,
      );
    } else {
      pages[i].push(`[Page ${pageNum}/${totalPages}]`);
    }
  }

  return { pages, introPageCount, aboutPageCount, refPageCount };
}

function generateIndexPage(
  introPageCount: number,
  aboutPageCount: number,
  refPageCount: number,
): string[] {
  const lines: string[] = [];
  lines.push("=== QQQLANG HELP INDEX ===");
  lines.push("");
  lines.push("Available pages:");
  lines.push("");

  if (introPageCount === 1) {
    lines.push("?? or ?A - Introduction");
  } else {
    const lastIntroChar = getPageChar(introPageCount);
    lines.push(`??/?A-?${lastIntroChar} - Introduction`);
  }

  if (aboutPageCount > 0) {
    const firstAboutPage = introPageCount + 1;
    const lastAboutPage = introPageCount + aboutPageCount;
    const firstAboutChar = getPageChar(firstAboutPage);
    const lastAboutChar = getPageChar(lastAboutPage);

    if (aboutPageCount === 1) {
      lines.push(`?${firstAboutChar} - About`);
    } else {
      lines.push(`?${firstAboutChar}-?${lastAboutChar} - About`);
    }
  }

  if (refPageCount > 0) {
    const firstRefPage = introPageCount + aboutPageCount + 1;
    const lastRefPage = introPageCount + aboutPageCount + refPageCount;
    const firstRefChar = getPageChar(firstRefPage);
    const lastRefChar = getPageChar(lastRefPage);

    if (refPageCount === 1) {
      lines.push(`?${firstRefChar} - Character reference`);
    } else {
      lines.push(`?${firstRefChar}-?${lastRefChar} - Character reference`);
    }
  }

  lines.push("?# - Image history");

  lines.push("");
  lines.push("Enter a valid page code to view help.");
  lines.push("Invalid page codes show this index.");

  return lines;
}

function imageHistory(ctx: FnContext): Image {
  const out = createSolidImage(ctx.width, ctx.height, "#000000");

  const tempCanvas = document.createElement("canvas");
  tempCanvas.width = ctx.width;
  tempCanvas.height = ctx.height;
  const tempCtx = tempCanvas.getContext("2d")!;

  tempCtx.fillStyle = "#141414";
  tempCtx.fillRect(0, 0, ctx.width, ctx.height);

  // Window size is 68 (the number of character functions)
  const windowSize = 68;
  const totalImages = ctx.images.length;
  const startOffset = Math.max(0, totalImages - windowSize);
  const numImages = Math.min(windowSize, totalImages);

  if (ctx.images.length === 0) {
    tempCtx.fillStyle = "#E8E4DC";
    tempCtx.font = "300 16px Inconsolata, monospace";
    tempCtx.fillText("No images in history", 10, 30);
    const imageData = tempCtx.getImageData(0, 0, ctx.width, ctx.height);
    out.data.set(imageData.data);
    return out;
  }

  const margin = 1;
  const hGap = 5;
  const availWidth = ctx.width - margin * 2;
  const availHeight = ctx.height - margin * 2;

  let bestLayout = { cols: 1, rows: numImages, thumbSize: 10, fontSize: 6 };
  let bestScore = 0;

  for (let cols = 1; cols <= Math.min(10, numImages); cols++) {
    const rows = Math.ceil(numImages / cols);

    const cellWidth = (availWidth - (cols - 1) * hGap) / cols;
    const cellHeight = availHeight / rows;

    const textHeight = Math.min(cellHeight * 0.15, 14);
    const fontSize = Math.max(6, Math.min(10, textHeight));
    const bottomMargin = 3;

    const thumbSize = Math.min(
      cellWidth - 2,
      cellHeight - textHeight - bottomMargin - 2,
    );

    if (thumbSize > 8 && fontSize >= 6) {
      const score = thumbSize * fontSize;
      if (score > bestScore) {
        bestScore = score;
        bestLayout = { cols, rows, thumbSize, fontSize };
      }
    }
  }

  const { cols, thumbSize, fontSize } = bestLayout;
  const cellWidth = (availWidth - (cols - 1) * hGap) / cols;
  const cellHeight = availHeight / Math.ceil(numImages / cols);

  tempCtx.fillStyle = "#E8E4DC";
  tempCtx.font = `300 ${fontSize}px Inconsolata, monospace`;

  for (let windowPos = 0; windowPos < numImages; windowPos++) {
    const col = windowPos % cols;
    const row = Math.floor(windowPos / cols);

    const x = margin + col * (cellWidth + hGap);
    const y = margin + row * cellHeight;

    const absoluteIndex = startOffset + windowPos;
    const img = ctx.images[absoluteIndex];
    const accessKey = numToChar(windowPos + 1);

    const opInfo = ctx.opInfos[absoluteIndex];
    const prevOpIdentifier =
      absoluteIndex > 0 ? ctx.opInfos[absoluteIndex - 1].identifier : "";
    const opChars = opInfo.identifier.substring(prevOpIdentifier.length);
    const displayOp = absoluteIndex === 0 ? "(init)" : opChars || "?";

    const thumbX = x + (cellWidth - thumbSize) / 2;
    const thumbY = y + 1;

    const thumbCanvas = document.createElement("canvas");
    thumbCanvas.width = thumbSize;
    thumbCanvas.height = thumbSize;
    const thumbCtx = thumbCanvas.getContext("2d")!;

    const srcSize = Math.min(img.width, img.height);
    const srcX = (img.width - srcSize) / 2;
    const srcY = (img.height - srcSize) / 2;

    const srcCanvas = document.createElement("canvas");
    srcCanvas.width = img.width;
    srcCanvas.height = img.height;
    const srcCtx = srcCanvas.getContext("2d")!;
    const srcImageData = new ImageData(
      new Uint8ClampedArray(img.data),
      img.width,
      img.height,
    );
    srcCtx.putImageData(srcImageData, 0, 0);

    thumbCtx.drawImage(
      srcCanvas,
      srcX,
      srcY,
      srcSize,
      srcSize,
      0,
      0,
      thumbSize,
      thumbSize,
    );

    tempCtx.strokeStyle = "#E8E4DC";
    tempCtx.lineWidth = 1;
    tempCtx.strokeRect(thumbX, thumbY, thumbSize, thumbSize);

    tempCtx.drawImage(thumbCanvas, thumbX, thumbY);

    const textY = thumbY + thumbSize + fontSize + 1;
    tempCtx.fillStyle = "#E8E4DC";
    tempCtx.textAlign = "center";
    tempCtx.fillText(`[${accessKey}] ${displayOp}`, thumbX + thumbSize / 2, textY);
  }

  const imageData = tempCtx.getImageData(0, 0, ctx.width, ctx.height);
  out.data.set(imageData.data);

  return out;
}

function help(ctx: FnContext, pageArg: number): Image {
  // If pageArg is 42 (the '#' character), show image history
  if (pageArg === 42) {
    return imageHistory(ctx);
  }

  const out = createSolidImage(ctx.width, ctx.height, "#000000");

  const tempCanvas = document.createElement("canvas");
  tempCanvas.width = ctx.width;
  tempCanvas.height = ctx.height;
  const tempCtx = tempCanvas.getContext("2d")!;

  tempCtx.fillStyle = "#141414";
  tempCtx.fillRect(0, 0, ctx.width, ctx.height);

  const marginFraction = 0.025;
  const margin = Math.max(
    8,
    Math.floor(Math.min(ctx.width, ctx.height) * marginFraction),
  );

  let page: number;
  if (pageArg === 58 || pageArg === 1) {
    page = 1;
  } else {
    page = pageArg;
  }

  // Iteratively find the largest font size that fits the content
  const minFontSize = 8;
  const maxFontSize = 16;
  let bestFontSize = minFontSize;
  let bestLines: string[] = [];
  let bestCharsPerLine = 40;
  let bestLinesPerPage = 20;

  for (let testSize = maxFontSize; testSize >= minFontSize; testSize -= 1) {
    tempCtx.font = `300 ${testSize}px Inconsolata, monospace`;
    const charWidth = tempCtx.measureText("M").width;
    const lineHeight = Math.floor(testSize * 1.25);

    const charsPerLine = Math.max(
      20,
      Math.floor((ctx.width - margin * 2) / charWidth),
    );
    const linesPerPage = Math.max(
      5,
      Math.floor((ctx.height - margin * 2) / lineHeight),
    );

    // Generate pages at this size
    const { pages, introPageCount, aboutPageCount, refPageCount } = generateAllHelpPages(
      charsPerLine,
      linesPerPage,
      characterDefs,
    );

    let lines: string[];
    if (page >= 1 && page <= pages.length) {
      lines = pages[page - 1];
    } else {
      lines = generateIndexPage(introPageCount, aboutPageCount, refPageCount);
    }

    // Check if content fits
    if (lines.length <= linesPerPage) {
      bestFontSize = testSize;
      bestLines = lines;
      bestCharsPerLine = charsPerLine;
      bestLinesPerPage = linesPerPage;
      break;
    }
  }

  // If we couldn't fit even at min size, just use min size
  if (bestLines.length === 0) {
    tempCtx.font = `300 ${minFontSize}px Inconsolata, monospace`;
    const charWidth = tempCtx.measureText("M").width;
    const lineHeight = Math.floor(minFontSize * 1.25);
    const charsPerLine = Math.max(
      20,
      Math.floor((ctx.width - margin * 2) / charWidth),
    );
    const linesPerPage = Math.max(
      5,
      Math.floor((ctx.height - margin * 2) / lineHeight),
    );
    const { pages, introPageCount, aboutPageCount, refPageCount } = generateAllHelpPages(
      charsPerLine,
      linesPerPage,
      characterDefs,
    );
    if (page >= 1 && page <= pages.length) {
      bestLines = pages[page - 1];
    } else {
      bestLines = generateIndexPage(introPageCount, aboutPageCount, refPageCount);
    }
    bestFontSize = minFontSize;
    bestLinesPerPage = linesPerPage;
  }

  const lineHeight = Math.floor(bestFontSize * 1.25);
  tempCtx.font = `300 ${bestFontSize}px Inconsolata, monospace`;
  tempCtx.fillStyle = "#E8E4DC";

  let y = margin + bestFontSize;
  for (let i = 0; i < Math.min(bestLines.length, bestLinesPerPage); i++) {
    tempCtx.fillText(breakLigatures(bestLines[i]), margin, y);
    y += lineHeight;
  }

  const imageData = tempCtx.getImageData(0, 0, ctx.width, ctx.height);
  out.data.set(imageData.data);

  return out;
}

export { help };
