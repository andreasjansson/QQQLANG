#!/usr/bin/env node
/**
 * Downloads and builds opentype.js from GitHub source
 * This is needed because:
 * 1. npm respects the `files` field even for git installs, excluding src/
 * 2. The COLR support in opentype.js master hasn't been published to npm yet
 */

import { execSync } from 'child_process';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const PROJECT_DIR = path.join(__dirname, '..');
const OPENTYPE_DIR = path.join(PROJECT_DIR, 'node_modules', 'opentype.js');
const DIST_FILE = path.join(OPENTYPE_DIR, 'dist', 'opentype.mjs');

async function setup() {
  // Check if already built
  if (fs.existsSync(DIST_FILE)) {
    console.log('opentype.js dist already exists, skipping build');
    return;
  }

  console.log('Setting up opentype.js with COLR support from GitHub master...');

  // Remove existing incomplete install
  if (fs.existsSync(OPENTYPE_DIR)) {
    console.log('Removing existing opentype.js installation...');
    fs.rmSync(OPENTYPE_DIR, { recursive: true });
  }

  // Clone directly into node_modules
  console.log('Cloning opentype.js repository...');
  execSync('git clone --depth 1 https://github.com/opentypejs/opentype.js.git', {
    cwd: path.join(PROJECT_DIR, 'node_modules'),
    stdio: 'inherit'
  });

  // Verify src folder exists
  const srcFolder = path.join(OPENTYPE_DIR, 'src');
  if (!fs.existsSync(srcFolder)) {
    throw new Error('src folder not found after clone - something went wrong');
  }
  console.log('✓ src folder exists');

  // List contents to verify
  const srcContents = fs.readdirSync(srcFolder);
  console.log('src/ contents:', srcContents.slice(0, 10).join(', '), srcContents.length > 10 ? '...' : '');

  // Check for tables folder
  const tablesFolder = path.join(srcFolder, 'tables');
  if (fs.existsSync(tablesFolder)) {
    const tableContents = fs.readdirSync(tablesFolder);
    console.log('src/tables/ contents:', tableContents.join(', '));
    
    // Look for COLR
    const hasColr = tableContents.some(f => f.toLowerCase().includes('colr'));
    console.log('Has COLR table:', hasColr ? '✓ YES' : '✗ NO');
  }

  console.log('Installing opentype.js dependencies...');
  execSync('npm install', {
    cwd: OPENTYPE_DIR,
    stdio: 'inherit'
  });

  console.log('Building opentype.js...');
  execSync('npm run build', {
    cwd: OPENTYPE_DIR,
    stdio: 'inherit'
  });

  // Verify dist was created
  if (!fs.existsSync(DIST_FILE)) {
    throw new Error('Build failed - dist/opentype.mjs not created');
  }

  console.log('✓ opentype.js setup complete with COLR support!');
}

setup().catch(err => {
  console.error('Error setting up opentype.js:', err);
  process.exit(1);
});
