#!/usr/bin/env node
/**
 * Downloads pre-built opentype.js with COLR support from the demo site.
 * This is needed because:
 * 1. The npm published version (1.3.4) doesn't have COLR support
 * 2. COLR support exists in master but hasn't been released
 * 3. The maintainers host pre-built dist files at https://opentype.js.org/dist/
 */

import fs from 'fs';
import path from 'path';
import https from 'https';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const PROJECT_DIR = path.join(__dirname, '..');
const OPENTYPE_DIR = path.join(PROJECT_DIR, 'node_modules', 'opentype.js');
const DIST_DIR = path.join(OPENTYPE_DIR, 'dist');
const DIST_FILE = path.join(DIST_DIR, 'opentype.mjs');
const DIST_FILE_JS = path.join(DIST_DIR, 'opentype.js');

const OPENTYPE_MJS_URL = 'https://opentype.js.org/dist/opentype.mjs';
const OPENTYPE_JS_URL = 'https://opentype.js.org/dist/opentype.js';

function download(url, dest) {
  return new Promise((resolve, reject) => {
    const file = fs.createWriteStream(dest);
    https.get(url, (response) => {
      if (response.statusCode === 301 || response.statusCode === 302) {
        download(response.headers.location, dest).then(resolve).catch(reject);
        return;
      }
      if (response.statusCode !== 200) {
        reject(new Error(`Failed to download ${url}: ${response.statusCode}`));
        return;
      }
      response.pipe(file);
      file.on('finish', () => {
        file.close();
        resolve();
      });
    }).on('error', (err) => {
      fs.unlink(dest, () => {});
      reject(err);
    });
  });
}

async function setup() {
  // Check if already downloaded
  if (fs.existsSync(DIST_FILE) && fs.existsSync(DIST_FILE_JS)) {
    const stats = fs.statSync(DIST_FILE);
    if (stats.size > 400000) { // Should be ~480KB
      console.log('opentype.js with COLR support already exists, skipping download');
      return;
    }
  }

  console.log('Downloading opentype.js with COLR support from opentype.js.org...');

  // Ensure directories exist
  if (!fs.existsSync(OPENTYPE_DIR)) {
    fs.mkdirSync(OPENTYPE_DIR, { recursive: true });
  }
  if (!fs.existsSync(DIST_DIR)) {
    fs.mkdirSync(DIST_DIR, { recursive: true });
  }

  // Download the ESM version
  console.log(`Downloading ${OPENTYPE_MJS_URL}...`);
  await download(OPENTYPE_MJS_URL, DIST_FILE);
  const mjsSize = fs.statSync(DIST_FILE).size;
  console.log(`✓ Downloaded opentype.mjs (${Math.round(mjsSize / 1024)}KB)`);

  // Download the UMD version
  console.log(`Downloading ${OPENTYPE_JS_URL}...`);
  await download(OPENTYPE_JS_URL, DIST_FILE_JS);
  const jsSize = fs.statSync(DIST_FILE_JS).size;
  console.log(`✓ Downloaded opentype.js (${Math.round(jsSize / 1024)}KB)`);

  // Create a minimal package.json if it doesn't exist
  const pkgJsonPath = path.join(OPENTYPE_DIR, 'package.json');
  if (!fs.existsSync(pkgJsonPath)) {
    fs.writeFileSync(pkgJsonPath, JSON.stringify({
      name: 'opentype.js',
      version: '2.0.0-beta',
      main: './dist/opentype.js',
      module: './dist/opentype.mjs'
    }, null, 2));
  }

  console.log('✓ opentype.js setup complete with COLR support!');
}

setup().catch(err => {
  console.error('Error setting up opentype.js:', err);
  process.exit(1);
});
