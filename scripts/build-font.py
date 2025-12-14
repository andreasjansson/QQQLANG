#!/usr/bin/env python3
"""
Build a custom Inconsolata font for QQQLANG with:
1. Colored glyphs using COLR/CPAL tables
2. Wider □ (U+25A1) character to match 1em for thumbnails
3. Ligatures for function+argument combinations
"""

import json
import re
import os
import sys
from pathlib import Path

try:
    from fontTools.ttLib import TTFont
    from fontTools.colorLib.builder import buildCOLR, buildCPAL
    from fontTools.feaLib.builder import addOpenTypeFeatures
    from fontTools.ttLib.tables import otTables
except ImportError:
    print("Please install fonttools: pip install fonttools")
    sys.exit(1)

SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
CHARACTER_DEFS_PATH = PROJECT_DIR / "character-defs.ts"
INPUT_FONT_PATH = PROJECT_DIR / "fonts" / "Inconsolata-Regular.ttf"
OUTPUT_FONT_PATH = PROJECT_DIR / "fonts" / "QQQLANG.ttf"

UPLOAD_CHAR = "□"  # U+25A1


def parse_character_defs():
    """Parse character-defs.ts to extract colors and arities."""
    content = CHARACTER_DEFS_PATH.read_text()
    
    # Find the characterDefs object
    pattern = r"'([^']+)':\s*\{[^}]*color:\s*'([^']+)'[^}]*number:\s*(\d+)[^}]*args:\s*\[(.*?)\]"
    
    chars = {}
    for match in re.finditer(pattern, content, re.DOTALL):
        char = match.group(1)
        color = match.group(2)
        number = int(match.group(3))
        args_str = match.group(4)
        
        # Count arguments by counting { type: patterns
        arity = len(re.findall(r'\{\s*type:', args_str))
        
        chars[char] = {
            'color': color,
            'number': number,
            'arity': arity
        }
    
    return chars


def hex_to_rgba(hex_color):
    """Convert hex color to RGBA tuple (0-255)."""
    hex_color = hex_color.lstrip('#')
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return (r, g, b, 255)


def build_color_font(char_defs):
    """Build the custom color font."""
    print(f"Loading font from {INPUT_FONT_PATH}")
    font = TTFont(INPUT_FONT_PATH)
    
    # Get glyph names
    glyph_order = font.getGlyphOrder()
    cmap = font.getBestCmap()
    
    # Build color palette
    colors = []
    color_index_map = {}
    
    # Add default color (white) at index 0
    colors.append((255, 255, 255, 255))
    color_index_map['#FFFFFF'] = 0
    
    # Add colors for each character
    for char, info in char_defs.items():
        color = info['color'].upper()
        if color not in color_index_map:
            color_index_map[color] = len(colors)
            colors.append(hex_to_rgba(color))
    
    print(f"Built palette with {len(colors)} colors")
    
    # Build COLR layers - map each glyph to its color
    color_layers = {}
    
    for char, info in char_defs.items():
        if len(char) == 1:
            codepoint = ord(char)
            if codepoint in cmap:
                glyph_name = cmap[codepoint]
                color = info['color'].upper()
                color_idx = color_index_map[color]
                # Each colored glyph references itself with a color
                color_layers[glyph_name] = [(glyph_name, color_idx)]
    
    print(f"Created color layers for {len(color_layers)} glyphs")
    
    # Build CPAL table (Color Palette)
    font['CPAL'] = buildCPAL([colors])
    
    # Build COLR table (Color Glyph)
    font['COLR'] = buildCOLR(color_layers)
    
    # Widen the □ character to match 1em
    widen_upload_char(font)
    
    # Save the font
    OUTPUT_FONT_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving font to {OUTPUT_FONT_PATH}")
    font.save(OUTPUT_FONT_PATH)
    print("Done!")


def widen_upload_char(font):
    """Make the □ character wider to match 1em (same as font's UPM)."""
    cmap = font.getBestCmap()
    codepoint = ord(UPLOAD_CHAR)
    
    if codepoint not in cmap:
        print(f"Warning: {UPLOAD_CHAR} (U+{codepoint:04X}) not found in font")
        return
    
    glyph_name = cmap[codepoint]
    
    # Get the units per em - this is what 1em equals in font units
    units_per_em = font['head'].unitsPerEm
    
    # Get current advance width
    hmtx = font['hmtx']
    current_width, lsb = hmtx[glyph_name]
    
    print(f"□ glyph '{glyph_name}': current width={current_width}, target width={units_per_em} (1em)")
    
    # Set new width to 1em
    # Center the glyph by adjusting LSB
    width_diff = units_per_em - current_width
    new_lsb = lsb + width_diff // 2
    
    hmtx[glyph_name] = (units_per_em, new_lsb)
    print(f"Set □ width to {units_per_em} (1em)")


def main():
    print("Building QQQLANG custom font...")
    
    if not INPUT_FONT_PATH.exists():
        print(f"Error: Input font not found at {INPUT_FONT_PATH}")
        print("Please download Inconsolata-Regular.ttf and place it in fonts/")
        sys.exit(1)
    
    char_defs = parse_character_defs()
    print(f"Parsed {len(char_defs)} character definitions")
    
    build_color_font(char_defs)


if __name__ == "__main__":
    main()
