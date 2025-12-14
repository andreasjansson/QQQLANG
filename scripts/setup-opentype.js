#!/usr/bin/env node
/**
 * Downloads and builds opentype.js from GitHub source
 * This is needed because npm respects the `files` field even for git installs,
 * which excludes the src/ folder from opentype.js
 */

import { execSync } from 'child_process';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const PROJECT_DIR = path.join(__dirname, '..');
const OPENTYPE_DIR = path.join(PROJECT_DIR, 'node_modules', 'opentype.js');
const DIST_FILE = path.join(OPENTYPE_DIR, 'dist', 'opentype.js');

async function setup() {
  // Check if already built
  if (fs.existsSync(DIST_FILE)) {
    console.log('opentype.js dist already exists, skipping build');
    return;
  }

  console.log('Setting up opentype.js with COLR support...');

  // Clone repo to temp location
  const tempDir = path.join(PROJECT_DIR, '.opentype-temp');
  
  if (fs.existsSync(tempDir)) {
    fs.rmSync(tempDir, { recursive: true });
  }

  console.log('Cloning opentype.js repository...');
  execSync('git clone --depth 1 https://github.com/opentypejs/opentype.js.git .opentype-temp', {
    cwd: PROJECT_DIR,
    stdio: 'inherit'
  });

  console.log('Installing opentype.js dependencies...');
  execSync('npm install', {
    cwd: tempDir,
    stdio: 'inherit'
  });

  console.log('Building opentype.js...');
  execSync('npm run build', {
    cwd: tempDir,
    stdio: 'inherit'
  });

  // Copy dist folder to node_modules/opentype.js
  const srcDist = path.join(tempDir, 'dist');
  const dstDist = path.join(OPENTYPE_DIR, 'dist');

  if (!fs.existsSync(OPENTYPE_DIR)) {
    fs.mkdirSync(OPENTYPE_DIR, { recursive: true });
  }

  console.log('Copying dist to node_modules/opentype.js...');
  fs.cpSync(srcDist, dstDist, { recursive: true });

  // Also copy package.json if needed
  if (!fs.existsSync(path.join(OPENTYPE_DIR, 'package.json'))) {
    fs.copyFileSync(
      path.join(tempDir, 'package.json'),
      path.join(OPENTYPE_DIR, 'package.json')
    );
  }

  // Cleanup temp dir
  console.log('Cleaning up...');
  fs.rmSync(tempDir, { recursive: true });

  console.log('opentype.js setup complete!');
}

setup().catch(err => {
  console.error('Error setting up opentype.js:', err);
  process.exit(1);
});
