#!/usr/bin/env python3
"""
Build a custom QQQLANG font based on Inconsolata with:
1. COLR/CPAL color tables for colored characters
2. Context-dependent bolding via GSUB calt feature
3. Context-dependent spacing after complete function calls

Rules:
- First character is always bold (initial color)
- Function characters are bold
- Arguments are regular weight
- After complete function calls, add extra spacing

Strategy using multiple GSUB lookup passes:
1. Make ALL characters bold unconditionally
2. For each arity N>0: find bold function, un-bold its N arguments
   - Arguments 1 to N-1 become regular
   - Argument N becomes regular_spaced (adds gap after complete call)
3. For arity 0: bold functions become bold_spaced (adds gap after complete call)

This works because after pass 1, everything is bold. Then pass 2 converts
arguments to regular (looking for bold function + bold args). Characters
that remain bold are functions. Pass 3 adds spacing.
"""

import re
import sys
from pathlib import Path
from fontTools.ttLib import TTFont
from fontTools.varLib.instancer import instantiateVariableFont
from fontTools.colorLib.builder import buildCOLR, buildCPAL

PROJECT_DIR = Path(__file__).parent.parent
CHARACTER_DEFS_PATH = PROJECT_DIR / "character-defs.ts"
INPUT_FONT_PATH = PROJECT_DIR / "fonts" / "Inconsolata-Variable.ttf"
OUTPUT_FONT_PATH = PROJECT_DIR / "public" / "QQQLANG.ttf"

PUA_START = 0xE000
FUNCTION_GAP = 100  # Extra advance width units after complete function calls

REGULAR_WEIGHT = 400
BOLD_WEIGHT = 700


def parse_character_defs():
    """Parse character definitions from character-defs.ts"""
    content = CHARACTER_DEFS_PATH.read_text()
    
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
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


def cleanup_variable_font_tables(font):
    """Remove variable font tables that cause issues with new glyphs."""
    for table_name in ['HVAR', 'VVAR', 'MVAR']:
        if table_name in font:
            del font[table_name]
            print(f"  Removed {table_name} table")
    
    if 'GDEF' in font:
        table = font['GDEF']
        if hasattr(table.table, 'VarStore'):
            del table.table.VarStore


def build_font():
    print("Building QQQLANG custom font...")
    
    if not INPUT_FONT_PATH.exists():
        print(f"Error: Input font not found at {INPUT_FONT_PATH}")
        sys.exit(1)
    
    char_defs = parse_character_defs()
    print(f"Parsed {len(char_defs)} character definitions")
    
    arity_map = {}
    for char, info in char_defs.items():
        if len(char) == 1:
            arity_map[char] = info['arity']
    
    arity_counts = {}
    for arity in arity_map.values():
        arity_counts[arity] = arity_counts.get(arity, 0) + 1
    
    for arity, count in sorted(arity_counts.items()):
        print(f"  {count} chars with arity {arity}")
    
    print(f"Loading font from {INPUT_FONT_PATH}")
    var_font = TTFont(INPUT_FONT_PATH)
    
    print("Creating regular weight instance...")
    regular_font = instantiateVariableFont(var_font, {"wght": REGULAR_WEIGHT})
    
    print("Creating bold weight instance...")
    bold_font = instantiateVariableFont(var_font, {"wght": BOLD_WEIGHT})
    
    font = regular_font
    
    glyph_order = list(font.getGlyphOrder())
    cmap = dict(font.getBestCmap())
    
    glyph_name_map = {}
    for char in char_defs.keys():
        if len(char) != 1:
            continue
        codepoint = ord(char)
        if codepoint in cmap:
            glyph_name_map[char] = cmap[codepoint]
    
    qqqlang_chars = sorted([c for c in char_defs.keys() if len(c) == 1 and c in glyph_name_map])
    print(f"Found {len(qqqlang_chars)} characters in font")
    
    glyf = font['glyf']
    bold_glyf = bold_font['glyf']
    hmtx = font['hmtx']
    bold_hmtx = bold_font['hmtx']
    
    print("Creating glyph variants in Private Use Area...")
    
    # Variants:
    # - bold: bold weight, normal spacing (for functions with args following)
    # - bold_spaced: bold weight, extra spacing (for arity-0 functions)
    # - regular_spaced: regular weight, extra spacing (for last argument)
    
    bold_glyph_map = {}
    bold_spaced_glyph_map = {}
    regular_spaced_glyph_map = {}
    
    pua_index = PUA_START
    new_glyphs = []
    
    for char in qqqlang_chars:
        regular_glyph_name = glyph_name_map[char]
        regular_glyph_data = glyf[regular_glyph_name]
        regular_width, regular_lsb = hmtx.metrics[regular_glyph_name]
        
        if regular_glyph_name in bold_glyf.glyphs:
            bold_glyph_data = bold_glyf[regular_glyph_name]
            if regular_glyph_name in bold_hmtx.metrics:
                bold_width, bold_lsb = bold_hmtx.metrics[regular_glyph_name]
            else:
                bold_width, bold_lsb = regular_width, regular_lsb
        else:
            bold_glyph_data = regular_glyph_data
            bold_width, bold_lsb = regular_width, regular_lsb
        
        # Bold variant
        bold_codepoint = pua_index
        pua_index += 1
        bold_name = f"uni{bold_codepoint:04X}"
        bold_glyph_map[char] = bold_name
        new_glyphs.append((bold_name, bold_glyph_data, bold_width, bold_lsb, bold_codepoint))
        
        # Bold+spaced variant
        bold_spaced_codepoint = pua_index
        pua_index += 1
        bold_spaced_name = f"uni{bold_spaced_codepoint:04X}"
        bold_spaced_glyph_map[char] = bold_spaced_name
        new_glyphs.append((bold_spaced_name, bold_glyph_data, bold_width + FUNCTION_GAP, bold_lsb, bold_spaced_codepoint))
        
        # Regular+spaced variant
        regular_spaced_codepoint = pua_index
        pua_index += 1
        regular_spaced_name = f"uni{regular_spaced_codepoint:04X}"
        regular_spaced_glyph_map[char] = regular_spaced_name
        new_glyphs.append((regular_spaced_name, regular_glyph_data, regular_width + FUNCTION_GAP, regular_lsb, regular_spaced_codepoint))
    
    for glyph_name, glyph_data, width, lsb, codepoint in new_glyphs:
        glyph_order.append(glyph_name)
        glyf.glyphs[glyph_name] = glyph_data
        hmtx.metrics[glyph_name] = (width, lsb)
        cmap[codepoint] = glyph_name
    
    font.setGlyphOrder(glyph_order)
    glyf.glyphOrder = glyph_order
    
    print(f"Created {len(bold_glyph_map)} bold variants")
    print(f"Created {len(bold_spaced_glyph_map)} bold+spaced variants")
    print(f"Created {len(regular_spaced_glyph_map)} regular+spaced variants")
    
    for table in font['cmap'].tables:
        if hasattr(table, 'cmap'):
            table.cmap.update(cmap)
    
    cleanup_variable_font_tables(font)
    
    # Build COLR/CPAL
    print("Building color tables...")
    
    colors = []
    color_to_index = {}
    
    for char, info in char_defs.items():
        if len(char) != 1:
            continue
        color = info['color'].upper()
        if color not in color_to_index:
            color_to_index[color] = len(colors)
            rgb = hex_to_rgb(color)
            colors.append((rgb[0] / 255, rgb[1] / 255, rgb[2] / 255, 1.0))
    
    cpal = buildCPAL([colors])
    font['CPAL'] = cpal
    
    color_glyphs = {}
    
    for char, info in char_defs.items():
        if len(char) != 1:
            continue
        
        color = info['color'].upper()
        color_index = color_to_index[color]
        
        if char in glyph_name_map:
            glyph_name = glyph_name_map[char]
            color_glyphs[glyph_name] = [(glyph_name, color_index)]
        
        if char in bold_glyph_map:
            name = bold_glyph_map[char]
            color_glyphs[name] = [(name, color_index)]
        
        if char in bold_spaced_glyph_map:
            name = bold_spaced_glyph_map[char]
            color_glyphs[name] = [(name, color_index)]
        
        if char in regular_spaced_glyph_map:
            name = regular_spaced_glyph_map[char]
            color_glyphs[name] = [(name, color_index)]
    
    colr = buildCOLR(color_glyphs)
    font['COLR'] = colr
    
    print(f"Added {len(colors)} colors to palette")
    print(f"Added {len(color_glyphs)} colored glyphs")
    
    # Build GSUB
    print("Building GSUB contextual substitution rules...")
    build_gsub_feature(font, char_defs, qqqlang_chars, arity_map,
                       glyph_name_map, bold_glyph_map, bold_spaced_glyph_map, regular_spaced_glyph_map)
    
    print("Updating font names...")
    name_table = font['name']
    for record in name_table.names:
        if record.nameID in (1, 4, 6):
            record.string = "QQQLANG"
    
    OUTPUT_FONT_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving font to {OUTPUT_FONT_PATH}")
    font.save(str(OUTPUT_FONT_PATH))
    
    print("Done!")


def build_gsub_feature(font, char_defs, qqqlang_chars, arity_map,
                       glyph_name_map, bold_glyph_map, bold_spaced_glyph_map, regular_spaced_glyph_map):
    """
    Build GSUB calt feature using multiple lookup passes:
    
    1. Make ALL characters bold (unconditionally)
    2. For each arity N (high to low): un-bold arguments of bold functions
       - Last arg becomes regular_spaced
       - Other args become regular
    3. For arity 0: bold → bold_spaced (add spacing)
    
    After pass 1, everything is bold. Pass 2 converts arguments to regular
    (by looking for bold_fn + bold_args). Characters that remain bold after
    pass 2 are functions. Pass 3 adds spacing to arity-0 functions.
    """
    from fontTools.feaLib.builder import addOpenTypeFeatures
    from io import StringIO
    
    by_arity = {}
    for char in qqqlang_chars:
        arity = arity_map.get(char, 0)
        if arity not in by_arity:
            by_arity[arity] = []
        by_arity[arity].append(char)
    
    max_arity = max(by_arity.keys()) if by_arity else 0
    print(f"  Max arity: {max_arity}")
    
    fea_lines = []
    
    # Define glyph classes
    all_regular = [glyph_name_map[c] for c in qqqlang_chars]
    all_bold = [bold_glyph_map[c] for c in qqqlang_chars]
    all_bold_spaced = [bold_spaced_glyph_map[c] for c in qqqlang_chars]
    all_regular_spaced = [regular_spaced_glyph_map[c] for c in qqqlang_chars]
    
    fea_lines.append(f"@regular = [{' '.join(all_regular)}];")
    fea_lines.append(f"@bold = [{' '.join(all_bold)}];")
    fea_lines.append(f"@bold_spaced = [{' '.join(all_bold_spaced)}];")
    fea_lines.append(f"@regular_spaced = [{' '.join(all_regular_spaced)}];")
    
    # @any includes all variants (for lookahead that doesn't care about variant)
    all_any = all_regular + all_bold + all_bold_spaced + all_regular_spaced
    fea_lines.append(f"@any = [{' '.join(all_any)}];")
    fea_lines.append("")
    
    # Classes by arity - only need bold class for matching functions
    for arity in sorted(by_arity.keys()):
        if arity == 0:
            continue  # arity 0 handled separately
        chars = by_arity[arity]
        bold = [bold_glyph_map[c] for c in chars]
        fea_lines.append(f"@fn{arity}_bold = [{' '.join(bold)}];")
    
    # Arity 0 bold class (for spacing pass)
    if 0 in by_arity:
        chars = by_arity[0]
        bold = [bold_glyph_map[c] for c in chars]
        fea_lines.append(f"@fn0_bold = [{' '.join(bold)}];")
    fea_lines.append("")
    
    # Lookup 1: regular → bold (make everything bold)
    fea_lines.append("lookup make_bold {")
    for char in qqqlang_chars:
        fea_lines.append(f"    sub {glyph_name_map[char]} by {bold_glyph_map[char]};")
    fea_lines.append("} make_bold;")
    fea_lines.append("")
    
    # Lookup 2: bold → regular (for non-last arguments, arity > 1)
    fea_lines.append("lookup bold_to_regular {")
    for char in qqqlang_chars:
        fea_lines.append(f"    sub {bold_glyph_map[char]} by {glyph_name_map[char]};")
    fea_lines.append("} bold_to_regular;")
    fea_lines.append("")
    
    # Lookup 3: bold → regular_spaced (for last argument)
    fea_lines.append("lookup bold_to_regular_spaced {")
    for char in qqqlang_chars:
        fea_lines.append(f"    sub {bold_glyph_map[char]} by {regular_spaced_glyph_map[char]};")
    fea_lines.append("} bold_to_regular_spaced;")
    fea_lines.append("")
    
    # Lookup 4: bold → bold_spaced (for arity-0 functions, add spacing)
    fea_lines.append("lookup bold_to_bold_spaced {")
    for char in qqqlang_chars:
        fea_lines.append(f"    sub {bold_glyph_map[char]} by {bold_spaced_glyph_map[char]};")
    fea_lines.append("} bold_to_bold_spaced;")
    fea_lines.append("")
    
    # The calt feature with ordered lookups
    fea_lines.append("feature calt {")
    
    # Pass 1: Make all regular chars bold
    fea_lines.append("    # Pass 1: Make all chars bold")
    fea_lines.append("    lookup pass1 {")
    fea_lines.append("        sub @regular' lookup make_bold;")
    fea_lines.append("    } pass1;")
    fea_lines.append("")
    
    # Pass 2: Un-bold arguments (process from highest arity to lowest)
    fea_lines.append("    # Pass 2: Un-bold arguments of functions")
    
    for arity in sorted([a for a in by_arity.keys() if a > 0], reverse=True):
        fea_lines.append(f"    # Arity {arity}")
        fea_lines.append(f"    lookup pass2_arity{arity} {{")
        
        fn_class = f"@fn{arity}_bold"
        
        if arity == 1:
            # Single arg: fn arg' → fn regular_spaced_arg
            fea_lines.append(f"        sub {fn_class} @bold' lookup bold_to_regular_spaced;")
        else:
            # Multiple args: first N-1 become regular, last becomes regular_spaced
            # We need multiple rules, processed left-to-right
            
            # First arg (and middle args): fn arg1' [args...] → regular
            # Pattern: fn_bold @bold' @any @any ... (arity-1 @any after the target)
            for arg_pos in range(arity - 1):
                # Match: fn, then arg_pos @any chars, then @bold', then remaining @any chars
                before = " @any" * arg_pos
                after = " @any" * (arity - 1 - arg_pos)
                fea_lines.append(f"        sub {fn_class}{before} @bold' lookup bold_to_regular{after};")
            
            # Last arg: fn @any... @bold' → regular_spaced
            before = " @any" * (arity - 1)
            fea_lines.append(f"        sub {fn_class}{before} @bold' lookup bold_to_regular_spaced;")
        
        fea_lines.append(f"    }} pass2_arity{arity};")
        fea_lines.append("")
    
    # Pass 3: Add spacing to arity-0 functions (still bold after pass 2)
    if 0 in by_arity:
        fea_lines.append("    # Pass 3: Add spacing to arity-0 functions")
        fea_lines.append("    lookup pass3_spacing {")
        fea_lines.append("        sub @fn0_bold' lookup bold_to_bold_spaced;")
        fea_lines.append("    } pass3_spacing;")
    
    fea_lines.append("} calt;")
    
    fea_code = "\n".join(fea_lines)
    
    rule_count = len([l for l in fea_lines if l.strip().startswith('sub ')])
    print(f"  Generated {rule_count} substitution rules")
    print(f"  Feature code: {len(fea_code)} bytes")
    
    # Debug output
    print("  Feature code preview:")
    for i, line in enumerate(fea_lines[:80]):
        print(f"    {line}")
    if len(fea_lines) > 80:
        print(f"    ... ({len(fea_lines) - 80} more lines)")
    
    addOpenTypeFeatures(font, StringIO(fea_code))


if __name__ == "__main__":
    build_font()
