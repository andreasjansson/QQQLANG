#!/usr/bin/env python3
"""Generate color swatch images for each character in QQQLANG."""

import re
import sys
from pathlib import Path
from PIL import Image

PROJECT_ROOT = Path(__file__).parent.parent.parent
CHARACTER_DEFS_PATH = PROJECT_ROOT / "character-defs.ts"
ASSETS_DIR = PROJECT_ROOT / "assets"

SWATCH_SIZE = 32


def parse_colors():
    """Parse character colors from character-defs.ts"""
    content = CHARACTER_DEFS_PATH.read_text()
    
    colors = {}
    
    # Find color and number for each character
    # Look for patterns like: color: "#XXXXXX", ... number: N,
    pattern = r'color:\s*["\']([^"\']+)["\'].*?number:\s*(\d+)'
    
    for match in re.finditer(pattern, content, re.DOTALL):
        color = match.group(1)
        number = int(match.group(2))
        colors[number] = color
    
    return colors


def hex_to_rgb(hex_color):
    """Convert hex color to RGB tuple."""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


def generate_swatch(color_hex, output_path):
    """Generate a small square color swatch image."""
    rgb = hex_to_rgb(color_hex)
    img = Image.new('RGB', (SWATCH_SIZE, SWATCH_SIZE), rgb)
    img.save(output_path, 'PNG')


def main():
    print("Generating color swatches...")
    
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    
    colors = parse_colors()
    print(f"Found {len(colors)} colors")
    
    for number, color in sorted(colors.items()):
        filename = f"{number:02d}-color.png"
        output_path = ASSETS_DIR / filename
        generate_swatch(color, output_path)
        print(f"  Generated {filename} ({color})")
    
    print("Done!")


if __name__ == "__main__":
    main()
