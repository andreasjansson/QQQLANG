#!/usr/bin/env python3
"""
Build a custom QQQLANG font based on Inconsolata with:
1. COLR/CPAL color tables for colored characters
2. Context-dependent bolding via GSUB calt feature (function chars are bold, args are not)
3. Context-dependent spacing via GPOS (extra space after complete function calls)

Strategy:
- First character is always bold (initial color)
- Function characters are bold
- Arguments are regular weight
- After a complete function call (fn + all its args), add extra spacing

We use OpenType GSUB ligature-like substitution to handle this:
- Create PUA variants: bold, bold+spaced, regular+spaced
- Build explicit substitution rules for each possible function pattern
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


def build_font():
    print("Building QQQLANG custom font...")
    
    if not INPUT_FONT_PATH.exists():
        print(f"Error: Input font not found at {INPUT_FONT_PATH}")
        sys.exit(1)
    
    char_defs = parse_character_defs()
    print(f"Parsed {len(char_defs)} character definitions")
    
    # Group by arity
    arity_map = {}  # char -> arity
    for char, info in char_defs.items():
        if len(char) == 1:
            arity_map[char] = info['arity']
    
    arity_counts = {}
    for arity in arity_map.values():
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
    glyph_order = list(font.getGlyphOrder())
    cmap = dict(font.getBestCmap())
    
    # Map characters to glyph names
    glyph_name_map = {}  # char -> regular glyph name
    for char in char_defs.keys():
        if len(char) != 1:
            continue
        codepoint = ord(char)
        if codepoint in cmap:
            glyph_name_map[char] = cmap[codepoint]
    
    qqqlang_chars = sorted([c for c in char_defs.keys() if len(c) == 1 and c in glyph_name_map])
    print(f"Found {len(qqqlang_chars)} characters in font")
    
    # Get tables for glyph manipulation
    glyf = font['glyf']
    bold_glyf = bold_font['glyf']
    hmtx = font['hmtx']
    bold_hmtx = bold_font['hmtx']
    
    print("Creating glyph variants in Private Use Area...")
    
    # Create PUA variants:
    # - bold: bold weight, normal spacing (for function chars with args following)
    # - bold_spaced: bold weight, extra spacing (for arity-0 functions, or function when it's the last in a call)
    # - regular_spaced: regular weight, extra spacing (for last argument of a function)
    
    bold_glyph_map = {}        # char -> bold PUA glyph name
    bold_spaced_glyph_map = {} # char -> bold+spaced PUA glyph name  
    regular_spaced_glyph_map = {}  # char -> regular+spaced PUA glyph name
    
    pua_index = PUA_START
    new_glyphs = []
    
    for char in qqqlang_chars:
        regular_glyph_name = glyph_name_map[char]
        regular_glyph_data = glyf[regular_glyph_name]
        regular_width, regular_lsb = hmtx.metrics[regular_glyph_name]
        
        # Get bold glyph data
        if regular_glyph_name in bold_glyf.glyphs:
            bold_glyph_data = bold_glyf[regular_glyph_name]
            if regular_glyph_name in bold_hmtx.metrics:
                bold_width, bold_lsb = bold_hmtx.metrics[regular_glyph_name]
            else:
                bold_width, bold_lsb = regular_width, regular_lsb
        else:
            bold_glyph_data = regular_glyph_data
            bold_width, bold_lsb = regular_width, regular_lsb
        
        # 1. Bold variant (bold weight, normal spacing)
        bold_codepoint = pua_index
        pua_index += 1
        bold_name = f"uni{bold_codepoint:04X}"
        bold_glyph_map[char] = bold_name
        new_glyphs.append((bold_name, bold_glyph_data, bold_width, bold_lsb, bold_codepoint))
        
        # 2. Bold+spaced variant (bold weight, extra spacing)
        bold_spaced_codepoint = pua_index
        pua_index += 1
        bold_spaced_name = f"uni{bold_spaced_codepoint:04X}"
        bold_spaced_glyph_map[char] = bold_spaced_name
        new_glyphs.append((bold_spaced_name, bold_glyph_data, bold_width + FUNCTION_GAP, bold_lsb, bold_spaced_codepoint))
        
        # 3. Regular+spaced variant (regular weight, extra spacing)
        regular_spaced_codepoint = pua_index
        pua_index += 1
        regular_spaced_name = f"uni{regular_spaced_codepoint:04X}"
        regular_spaced_glyph_map[char] = regular_spaced_name
        new_glyphs.append((regular_spaced_name, regular_glyph_data, regular_width + FUNCTION_GAP, regular_lsb, regular_spaced_codepoint))
    
    # Add all new glyphs
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
    
    # Update cmap table
    for table in font['cmap'].tables:
        if hasattr(table, 'cmap'):
            table.cmap.update(cmap)
    
    # Build COLR/CPAL color tables
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
        
        # Regular glyph
        if char in glyph_name_map:
            glyph_name = glyph_name_map[char]
            color_glyphs[glyph_name] = [(glyph_name, color_index)]
        
        # Bold variant
        if char in bold_glyph_map:
            bold_name = bold_glyph_map[char]
            color_glyphs[bold_name] = [(bold_name, color_index)]
        
        # Bold+spaced variant
        if char in bold_spaced_glyph_map:
            name = bold_spaced_glyph_map[char]
            color_glyphs[name] = [(name, color_index)]
        
        # Regular+spaced variant
        if char in regular_spaced_glyph_map:
            name = regular_spaced_glyph_map[char]
            color_glyphs[name] = [(name, color_index)]
    
    colr = buildCOLR(color_glyphs)
    font['COLR'] = colr
    
    print(f"Added {len(colors)} colors to palette")
    print(f"Added {len(color_glyphs)} colored glyphs")
    
    # Build GSUB feature
    print("Building GSUB contextual substitution rules...")
    build_gsub_feature(font, char_defs, qqqlang_chars, arity_map,
                       glyph_name_map, bold_glyph_map, bold_spaced_glyph_map, regular_spaced_glyph_map)
    
    # Update font names
    print("Updating font names...")
    name_table = font['name']
    
    for record in name_table.names:
        if record.nameID in (1, 4, 6):
            record.string = "QQQLANG"
    
    # Save the font
    OUTPUT_FONT_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving font to {OUTPUT_FONT_PATH}")
    font.save(str(OUTPUT_FONT_PATH))
    
    print("Done!")


def cleanup_variable_font_tables(font):
    """
    Remove or clean up variable font tables that can cause issues
    when adding new glyphs to an instanced font.
    """
    # Tables that may have VarIdxMap or other glyph-indexed data
    tables_to_check = ['MVAR', 'HVAR', 'VVAR', 'GDEF']
    
    for table_name in tables_to_check:
        if table_name in font:
            table = font[table_name]
            # For GDEF, we need to be careful - it may have LigCaretList etc.
            # But if it has VarIdxMap references, those will break
            if table_name == 'GDEF' and hasattr(table.table, 'VarStore'):
                # Remove the VarStore to avoid glyph indexing issues
                del table.table.VarStore
                if hasattr(table.table, 'GlyphClassDef'):
                    # Keep GlyphClassDef but remove VarStore
                    pass
            elif table_name in ['HVAR', 'VVAR', 'MVAR']:
                # These are specifically for variable fonts, safe to remove
                del font[table_name]
                print(f"  Removed {table_name} table")


def build_gsub_feature(font, char_defs, qqqlang_chars, arity_map,
                       glyph_name_map, bold_glyph_map, bold_spaced_glyph_map, regular_spaced_glyph_map):
    """
    Build GSUB calt feature for contextual substitution.
    
    The strategy is to enumerate all possible complete function calls and create
    explicit substitution rules. Since we're dealing with a small character set
    and max arity of ~11, this is tractable.
    
    Rules (processed in order):
    1. First, we need to handle the sequence-based transformations
    2. Use multiple lookups applied in sequence
    
    The key insight: we process left-to-right. When we see a function character,
    we need to know if it's in "function position" (start of program, or after
    a complete function call) vs "argument position".
    
    Approach:
    - We'll use a state machine implemented via chained contextual substitution
    - The "state" is encoded in which variant of the glyph we're using
    - Regular glyphs = in argument position or initial
    - After processing: bold = function, regular = arg, spaced = end of call
    
    Simpler approach for now:
    - Just handle spacing after complete calls
    - Make ALL function characters bold (can't easily distinguish first char of program)
    
    Even simpler first pass:
    - Enumerate all complete function call patterns
    - Sub the function char to bold, and last char to spaced variant
    """
    from fontTools.feaLib.builder import addOpenTypeFeatures
    from io import StringIO
    
    # Group functions by arity
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
    fea_lines.append("")
    
    # Classes by arity - for functions
    for arity in sorted(by_arity.keys()):
        chars = by_arity[arity]
        reg = [glyph_name_map[c] for c in chars]
        bold = [bold_glyph_map[c] for c in chars]
        bold_sp = [bold_spaced_glyph_map[c] for c in chars]
        fea_lines.append(f"@fn{arity}_reg = [{' '.join(reg)}];")
        fea_lines.append(f"@fn{arity}_bold = [{' '.join(bold)}];")
        fea_lines.append(f"@fn{arity}_bold_spaced = [{' '.join(bold_sp)}];")
    fea_lines.append("")
    
    # Lookups for individual substitutions
    
    # Lookup: regular -> bold
    fea_lines.append("lookup regular_to_bold {")
    for char in qqqlang_chars:
        fea_lines.append(f"    sub {glyph_name_map[char]} by {bold_glyph_map[char]};")
    fea_lines.append("} regular_to_bold;")
    fea_lines.append("")
    
    # Lookup: regular -> bold_spaced
    fea_lines.append("lookup regular_to_bold_spaced {")
    for char in qqqlang_chars:
        fea_lines.append(f"    sub {glyph_name_map[char]} by {bold_spaced_glyph_map[char]};")
    fea_lines.append("} regular_to_bold_spaced;")
    fea_lines.append("")
    
    # Lookup: regular -> regular_spaced
    fea_lines.append("lookup regular_to_regular_spaced {")
    for char in qqqlang_chars:
        fea_lines.append(f"    sub {glyph_name_map[char]} by {regular_spaced_glyph_map[char]};")
    fea_lines.append("} regular_to_regular_spaced;")
    fea_lines.append("")
    
    # Lookup: bold -> bold_spaced
    fea_lines.append("lookup bold_to_bold_spaced {")
    for char in qqqlang_chars:
        fea_lines.append(f"    sub {bold_glyph_map[char]} by {bold_spaced_glyph_map[char]};")
    fea_lines.append("} bold_to_bold_spaced;")
    fea_lines.append("")
    
    # The calt feature - contextual rules
    # 
    # Strategy: We'll use multiple passes
    # Pass 1: Mark function chars as bold when followed by enough args
    # Pass 2: Mark last char of complete calls as spaced
    #
    # The tricky part is that we need to process patterns correctly.
    # We'll enumerate explicit patterns for each arity.
    
    fea_lines.append("feature calt {")
    
    # For clarity and correctness, we'll generate explicit rules for each arity
    # processing from highest arity down to lowest (greedy matching)
    
    # First: Arity 0 - function itself is bold+spaced (complete call = just the function)
    if 0 in by_arity:
        fea_lines.append("    # Arity 0: function char -> bold_spaced")
        for char in by_arity[0]:
            fea_lines.append(f"    sub {glyph_name_map[char]}' lookup regular_to_bold_spaced;")
        fea_lines.append("")
    
    # For arity >= 1: function -> bold, last arg -> regular_spaced
    # We process from high to low arity for proper precedence
    for arity in sorted([a for a in by_arity.keys() if a > 0], reverse=True):
        fea_lines.append(f"    # Arity {arity}: fn -> bold, arg{arity} -> regular_spaced")
        
        # We need to be careful here. The pattern is:
        # fn_char arg1 arg2 ... argN
        # We want: fn_char' -> bold (when followed by N @regular)
        #          argN' -> regular_spaced (when preceded by bold_fn and N-1 regular)
        
        # First rule: fn_char -> bold when followed by exactly arity @regular chars
        # BUT: we need to ensure the chars after are actually "regular" (not already transformed)
        # AND: the last one should become spaced
        
        # The approach: 
        # Rule 1: fn followed by arity regulars -> make fn bold
        # Rule 2: bold_fn followed by (arity-1) regulars, then 1 regular -> make last regular spaced
        
        fn_class = f"@fn{arity}_reg"
        fn_bold_class = f"@fn{arity}_bold"
        
        # Rule 1: Transform function char to bold
        if arity == 1:
            fea_lines.append(f"    sub {fn_class}' lookup regular_to_bold @regular;")
        else:
            args = " @regular" * arity
            fea_lines.append(f"    sub {fn_class}' lookup regular_to_bold{args};")
        
        # Rule 2: Transform last argument to spaced
        # We look for the bold function followed by args
        if arity == 1:
            fea_lines.append(f"    sub {fn_bold_class} @regular' lookup regular_to_regular_spaced;")
        else:
            preceding_args = " @regular" * (arity - 1)
            fea_lines.append(f"    sub {fn_bold_class}{preceding_args} @regular' lookup regular_to_regular_spaced;")
        
        fea_lines.append("")
    
    fea_lines.append("} calt;")
    
    # Join feature code
    fea_code = "\n".join(fea_lines)
    
    # Print summary
    rule_count = len([l for l in fea_lines if l.strip().startswith('sub ')])
    print(f"  Generated {rule_count} substitution rules")
    print(f"  Feature code: {len(fea_code)} bytes")
    
    # Debug: print the feature code
    if len(fea_lines) <= 100:
        print("  Generated feature code:")
        for line in fea_lines:
            print(f"    {line}")
    else:
        print("  First 50 lines of feature code:")
        for line in fea_lines[:50]:
            print(f"    {line}")
        print(f"    ... ({len(fea_lines) - 50} more lines)")
    
    # Apply the feature code to the font
    addOpenTypeFeatures(font, StringIO(fea_code))
    
    # Clean up variable font tables that may cause issues with new glyphs
    cleanup_variable_font_tables(font)


if __name__ == "__main__":
    build_font()
