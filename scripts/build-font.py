#!/usr/bin/env python3
"""
Build a custom QQQLANG font based on Inconsolata with:
1. COLR/CPAL color tables for colored characters
2. Context-dependent bolding via GSUB calt feature (function chars are bold, args are not)
3. Context-dependent spacing via GPOS (extra space after complete function calls)
4. Wider □ (U+25A1) character for uploads

Strategy:
- Load variable Inconsolata font and create instances at regular and bold weights
- Create bold variants of all QQQLANG characters in PUA (Private Use Area)
- Create spaced variants (with extra advance width) in PUA
- Use OpenType GSUB 'calt' feature with chained contextual substitution rules to:
  - Make function-position characters bold
  - Keep argument-position characters regular
  - Add spacing after complete function calls
"""

import re
import sys
from pathlib import Path
from fontTools.ttLib import TTFont
from fontTools.varLib.mutator import instantiateVariableFont
from fontTools.colorLib.builder import buildCOLR, buildCPAL

PROJECT_DIR = Path(__file__).parent.parent
CHARACTER_DEFS_PATH = PROJECT_DIR / "character-defs.ts"
INPUT_FONT_PATH = PROJECT_DIR / "fonts" / "Inconsolata-Variable.ttf"
OUTPUT_FONT_PATH = PROJECT_DIR / "public" / "QQQLANG.ttf"

UPLOAD_CHAR = "□"
PUA_START = 0xE000
FUNCTION_GAP = 100  # Extra advance width units after complete function calls (~10px at 22px font)

# Weight axis values for Inconsolata variable font
REGULAR_WEIGHT = 400
BOLD_WEIGHT = 700


def parse_character_defs():
    """Parse character definitions from character-defs.ts"""
    content = CHARACTER_DEFS_PATH.read_text()
    
    # Match patterns like: 'A': { color: '#78A10F', number: 1, ... args: [...], functionName: "..."
    pattern = r"'([^']+)':\s*\{[^}]*color:\s*'([^']+)'[^}]*number:\s*(\d+)[^}]*args:\s*\[([\s\S]*?)\][^}]*functionName"
    
    chars = {}
    for match in re.finditer(pattern, content):
        char = match.group(1)
        color = match.group(2)
        number = int(match.group(3))
        args_str = match.group(4)
        arity = len(re.findall(r'\{\s*type:', args_str))
        chars[char] = {"color": color, "number": number, "arity": arity}
    
    return chars


def hex_to_rgb(hex_color):
    """Convert hex color to RGB tuple (0-255 range)"""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


def create_calt_feature(font, char_defs, glyph_name_map, bold_glyph_map, spaced_glyph_map):
    """
    Create the 'calt' feature for contextual alternates using fontTools' feaLib.
    
    Strategy:
    - All glyphs start as regular
    - Use chained contextual substitution to:
      1. Make function chars bold
      2. Keep argument chars regular  
      3. Add spacing after the last char of each complete function call
    
    We process patterns for each arity level separately.
    """
    from fontTools.feaLib.builder import addOpenTypeFeatures
    from io import StringIO
    
    qqqlang_chars = sorted([c for c in char_defs.keys() if len(c) == 1])
    
    # Group by arity
    arity_groups = {}
    for char, info in char_defs.items():
        if len(char) != 1:
            continue
        arity = info['arity']
        if arity not in arity_groups:
            arity_groups[arity] = []
        arity_groups[arity].append(char)
    
    # Build feature file content
    fea_lines = []
    
    # Define glyph classes
    all_regular = [glyph_name_map[c] for c in qqqlang_chars if c in glyph_name_map]
    all_bold = [bold_glyph_map[c] for c in qqqlang_chars if c in bold_glyph_map]
    all_spaced = [spaced_glyph_map[c] for c in qqqlang_chars if c in spaced_glyph_map]
    
    fea_lines.append(f"@regular = [{' '.join(all_regular)}];")
    fea_lines.append(f"@bold = [{' '.join(all_bold)}];")
    fea_lines.append(f"@spaced = [{' '.join(all_spaced)}];")
    fea_lines.append("")
    
    # Define classes per arity
    for arity, chars in sorted(arity_groups.items()):
        regular_glyphs = [glyph_name_map[c] for c in chars if c in glyph_name_map]
        bold_glyphs = [bold_glyph_map[c] for c in chars if c in bold_glyph_map]
        if regular_glyphs:
            fea_lines.append(f"@fn_arity{arity}_regular = [{' '.join(regular_glyphs)}];")
        if bold_glyphs:
            fea_lines.append(f"@fn_arity{arity}_bold = [{' '.join(bold_glyphs)}];")
    fea_lines.append("")
    
    # Lookup: regular -> bold (for function positions)
    fea_lines.append("lookup toBold {")
    for char in qqqlang_chars:
        if char in glyph_name_map and char in bold_glyph_map:
            fea_lines.append(f"    sub {glyph_name_map[char]} by {bold_glyph_map[char]};")
    fea_lines.append("} toBold;")
    fea_lines.append("")
    
    # Lookup: regular -> spaced (for end of function call - arity 0 or last arg)
    fea_lines.append("lookup toSpaced {")
    for char in qqqlang_chars:
        if char in glyph_name_map and char in spaced_glyph_map:
            fea_lines.append(f"    sub {glyph_name_map[char]} by {spaced_glyph_map[char]};")
    fea_lines.append("} toSpaced;")
    fea_lines.append("")
    
    # Lookup: bold -> spaced (alternative path)
    fea_lines.append("lookup boldToSpaced {")
    for char in qqqlang_chars:
        if char in bold_glyph_map and char in spaced_glyph_map:
            fea_lines.append(f"    sub {bold_glyph_map[char]} by {spaced_glyph_map[char]};")
    fea_lines.append("} boldToSpaced;")
    fea_lines.append("")
    
    # Now create the calt feature with contextual rules
    fea_lines.append("feature calt {")
    
    # For arity 0: fn -> bold+spaced
    # These are standalone functions that need spacing after them
    if 0 in arity_groups:
        fea_lines.append("    # Arity 0: standalone functions get bold+spaced")
        for char in arity_groups[0]:
            if char in glyph_name_map and char in spaced_glyph_map:
                # Convert to spaced (which is bold weight)
                fea_lines.append(f"    sub {glyph_name_map[char]}' lookup toSpaced;")
    fea_lines.append("")
    
    # For arity > 0: fn -> bold, args stay regular, last arg -> spaced
    # We need to match the full pattern and apply substitutions
    
    for arity in sorted(arity_groups.keys()):
        if arity == 0:
            continue
            
        chars = arity_groups[arity]
        fea_lines.append(f"    # Arity {arity}: function + {arity} argument(s)")
        
        for char in chars:
            if char not in glyph_name_map:
                continue
            
            reg_glyph = glyph_name_map[char]
            
            # Build the pattern: fn arg1 arg2 ... argN
            # We want: fn' -> bold, arg1...argN-1 stay regular, argN' -> spaced
            
            # Pattern with N arguments following
            # sub fn' @regular @regular ... @regular by bold_fn;  (just fn to bold)
            # Then separately: sub @fn_arityN_bold @regular ... @regular' by spaced; (last arg to spaced)
            
            # First rule: make function bold when followed by N regular chars
            if arity == 1:
                fea_lines.append(f"    sub {reg_glyph}' lookup toBold @regular;")
            else:
                args_pattern = " @regular" * arity
                fea_lines.append(f"    sub {reg_glyph}' lookup toBold{args_pattern};")
        
        fea_lines.append("")
        
        # Second set of rules: make the last argument spaced
        # We look for: bold_fn followed by (arity-1) regular chars, then the last regular char
        bold_glyphs_for_arity = [bold_glyph_map[c] for c in chars if c in bold_glyph_map]
        if bold_glyphs_for_arity:
            fea_lines.append(f"    # Arity {arity}: space the last argument")
            class_name = f"@fn_arity{arity}_bold"
            
            if arity == 1:
                # Pattern: bold_fn regular' -> bold_fn spaced
                fea_lines.append(f"    sub {class_name} @regular' lookup toSpaced;")
            else:
                # Pattern: bold_fn regular regular ... regular' -> spaced
                middle_args = " @regular" * (arity - 1)
                fea_lines.append(f"    sub {class_name}{middle_args} @regular' lookup toSpaced;")
        
        fea_lines.append("")
    
    fea_lines.append("} calt;")
    
    # Join and apply
    fea_code = "\n".join(fea_lines)
    
    print("Generated feature code:")
    print("-" * 40)
    for i, line in enumerate(fea_lines[:50]):
        print(f"{i+1:3}: {line}")
    if len(fea_lines) > 50:
        print(f"... ({len(fea_lines) - 50} more lines)")
    print("-" * 40)
    
    # Apply the feature code to the font
    addOpenTypeFeatures(font, StringIO(fea_code))


def build_font():
    print("Building QQQLANG custom font...")
    
    if not INPUT_FONT_PATH.exists():
        print(f"Error: Input font not found at {INPUT_FONT_PATH}")
        sys.exit(1)
    
    char_defs = parse_character_defs()
    print(f"Parsed {len(char_defs)} character definitions")
    
    # Count by arity
    arity_counts = {}
    for info in char_defs.values():
        arity = info['arity']
        arity_counts[arity] = arity_counts.get(arity, 0) + 1
    
    for arity, count in sorted(arity_counts.items()):
        print(f"  {count} chars with arity {arity}")
    
    # Load the variable font
    print(f"Loading font from {INPUT_FONT_PATH}")
    var_font = TTFont(INPUT_FONT_PATH)
    
    # Create regular weight instance
    print("Creating regular weight instance...")
    regular_font = instantiateVariableFont(var_font, {"wght": REGULAR_WEIGHT})
    
    # Create bold weight instance  
    print("Creating bold weight instance...")
    bold_font = instantiateVariableFont(var_font, {"wght": BOLD_WEIGHT})
    
    # We'll work with the regular font as our base
    font = regular_font
    
    # Get the glyph order and cmap
    glyph_order = font.getGlyphOrder()
    cmap = font.getBestCmap()
    
    # Map characters to glyph names
    glyph_name_map = {}  # char -> regular glyph name
    for char in char_defs.keys():
        if len(char) != 1:
            continue
        codepoint = ord(char)
        if codepoint in cmap:
            glyph_name_map[char] = cmap[codepoint]
    
    print(f"Found {len(glyph_name_map)} characters in font")
    
    # Create bold variants in PUA
    bold_glyph_map = {}  # char -> bold PUA glyph name
    spaced_glyph_map = {}  # char -> spaced PUA glyph name
    
    pua_index = PUA_START
    
    # Get glyf table for glyph manipulation
    glyf = font['glyf']
    bold_glyf = bold_font['glyf']
    hmtx = font['hmtx']
    bold_hmtx = bold_font['hmtx']
    
    print("Creating bold and spaced variants in Private Use Area...")
    
    qqqlang_chars = sorted([c for c in char_defs.keys() if len(c) == 1])
    
    # Collect new glyphs to add
    new_glyphs = []  # List of (glyph_name, glyph_data, width, lsb, codepoint)
    
    for char in qqqlang_chars:
        if char not in glyph_name_map:
            continue
            
        regular_glyph_name = glyph_name_map[char]
        
        # Create bold variant
        bold_codepoint = pua_index
        pua_index += 1
        bold_glyph_name = f"uni{bold_codepoint:04X}"
        bold_glyph_map[char] = bold_glyph_name
        
        # Get the bold glyph data
        if regular_glyph_name in bold_glyf.glyphs:
            bold_glyph_data = bold_glyf[regular_glyph_name]
            if regular_glyph_name in bold_hmtx.metrics:
                width, lsb = bold_hmtx.metrics[regular_glyph_name]
            else:
                width, lsb = hmtx.metrics[regular_glyph_name]
        else:
            bold_glyph_data = glyf[regular_glyph_name]
            width, lsb = hmtx.metrics[regular_glyph_name]
        
        new_glyphs.append((bold_glyph_name, bold_glyph_data, width, lsb, bold_codepoint))
        
        # Create spaced variant (bold + extra advance width)
        spaced_codepoint = pua_index
        pua_index += 1
        spaced_glyph_name = f"uni{spaced_codepoint:04X}"
        spaced_glyph_map[char] = spaced_glyph_name
        
        # Use the same bold glyph data but with extra width
        new_glyphs.append((spaced_glyph_name, bold_glyph_data, width + FUNCTION_GAP, lsb, spaced_codepoint))
    
    # Now add all new glyphs at once
    for glyph_name, glyph_data, width, lsb, codepoint in new_glyphs:
        glyph_order.append(glyph_name)
        glyf.glyphs[glyph_name] = glyph_data
        hmtx.metrics[glyph_name] = (width, lsb)
        cmap[codepoint] = glyph_name
    
    # Update glyph order in font and glyf table
    font.setGlyphOrder(glyph_order)
    glyf.glyphOrder = glyph_order
    
    print(f"Created {len(bold_glyph_map)} bold variants")
    print(f"Created {len(spaced_glyph_map)} spaced variants")
    
    # Update cmap table
    for table in font['cmap'].tables:
        if hasattr(table, 'cmap'):
            table.cmap.update(cmap)
    
    # Build COLR/CPAL color tables
    print("Building color tables...")
    
    # Collect unique colors
    colors = []
    color_to_index = {}
    
    for char, info in char_defs.items():
        if len(char) != 1:
            continue
        color = info['color'].upper()
        if color not in color_to_index:
            color_to_index[color] = len(colors)
            rgb = hex_to_rgb(color)
            # CPAL expects colors in 0-1 range
            colors.append((rgb[0] / 255, rgb[1] / 255, rgb[2] / 255, 1.0))
    
    # Build CPAL (Color Palette)
    cpal = buildCPAL([colors])
    font['CPAL'] = cpal
    
    # Build COLR (Color Glyph) - map each glyph to its color layer
    color_glyphs = {}
    
    for char, info in char_defs.items():
        if len(char) != 1:
            continue
        
        color = info['color'].upper()
        color_index = color_to_index[color]
        
        # Regular glyph
        if char in glyph_name_map:
            glyph_name = glyph_name_map[char]
            color_glyphs[glyph_name] = [(glyph_name, color_index)]
        
        # Bold variant
        if char in bold_glyph_map:
            bold_name = bold_glyph_map[char]
            color_glyphs[bold_name] = [(bold_name, color_index)]
        
        # Spaced variant
        if char in spaced_glyph_map:
            spaced_name = spaced_glyph_map[char]
            color_glyphs[spaced_name] = [(spaced_name, color_index)]
    
    colr = buildCOLR(color_glyphs)
    font['COLR'] = colr
    
    print(f"Added {len(colors)} colors to palette")
    print(f"Added {len(color_glyphs)} colored glyphs")
    
    # Create GSUB calt feature for contextual substitution
    print("Building GSUB contextual alternates feature...")
    create_calt_feature(font, char_defs, glyph_name_map, bold_glyph_map, spaced_glyph_map)
    
    # Update font names
    print("Updating font names...")
    name_table = font['name']
    
    # Update relevant name records
    for record in name_table.names:
        if record.nameID in (1, 4, 6):  # Family, Full name, PostScript name
            if record.nameID == 1:
                new_name = "QQQLANG"
            elif record.nameID == 4:
                new_name = "QQQLANG"
            elif record.nameID == 6:
                new_name = "QQQLANG"
            
            record.string = new_name
    
    # Save the font
    OUTPUT_FONT_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving font to {OUTPUT_FONT_PATH}")
    font.save(str(OUTPUT_FONT_PATH))
    
    print("Done!")


if __name__ == "__main__":
    build_font()
