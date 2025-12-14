#!/usr/bin/env python3
"""
Build a custom QQQLANG font based on Inconsolata with:
1. COLR/CPAL color tables for colored characters
2. Context-dependent bolding via GSUB calt feature
3. Context-dependent spacing after complete function calls

Rules:
- First character is ALWAYS bold (initial color) - special case!
- Function characters are bold
- Arguments are regular weight
- After complete function calls, add extra spacing

Strategy using multiple GSUB lookup passes:
1a. Make ALL characters "bold_first" (special initial variant)
1b. Convert any bold_first preceded by bold_first/bold → bold
    (This leaves only the FIRST character as bold_first)
2. For each arity N>0: un-bold arguments of bold functions
   - Each arg position is handled separately (don't require all args present)
   - Args become regular (not spaced yet)
3. For complete calls: add spacing to last arg
   - For arity N: if fn_bold followed by N @any chars, last one → spaced
4. For arity 0: bold functions → bold_spaced
5. Convert bold_first → bold_spaced (first char gets spacing)
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
FUNCTION_GAP = 150  # ~15px at 22px font size

REGULAR_WEIGHT = 400
BOLD_WEIGHT = 700

UPLOAD_CHAR = '□'  # U+25A1 - special character for uploaded images


def parse_character_defs():
    content = CHARACTER_DEFS_PATH.read_text()
    pattern = r"'([^']+)':\s*\{[^}]*color:\s*'([^']+)'[^}]*number:\s*(\d+)[^}]*args:\s*\[([\s\S]*?)\][^}]*functionName"
    
    chars = {}
    for match in re.finditer(pattern, content):
        char = match.group(1)
        # Handle escape sequences from TypeScript source
        if char == '\\\\':
            char = '\\'
        elif char == "\\'":
            char = "'"
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
    for table_name in ['HVAR', 'VVAR', 'MVAR']:
        if table_name in font:
            del font[table_name]
            print(f"  Removed {table_name} table")
    
    if 'GDEF' in font:
        table = font['GDEF']
        if hasattr(table.table, 'VarStore'):
            del table.table.VarStore


def create_upload_char_variants(font, glyph_order, glyf, hmtx, cmap, pua_start):
    """
    Create the □ (upload) character and all its variants (bold_first, bold, 
    bold_spaced, regular_spaced).
    
    The □ character is used for uploaded images and should follow the same
    spacing rules as other characters - it acts like an arity-0 function.
    
    Returns a dict with glyph names for each variant.
    """
    from fontTools.ttLib.tables._g_l_y_f import Glyph
    
    upload_codepoint = ord(UPLOAD_CHAR)
    units_per_em = font['head'].unitsPerEm
    
    # Check if □ already exists in the font
    if upload_codepoint in cmap:
        upload_glyph_name = cmap[upload_codepoint]
        print(f"  Found existing □ glyph: {upload_glyph_name}")
    else:
        # Create a simple square outline for □
        upload_glyph_name = 'uni25A1'
        
        stroke_width = int(units_per_em * 0.06)
        size = int(units_per_em * 0.7)
        left = int((units_per_em - size) / 2)
        right = left + size
        bottom = 0
        top = size
        
        inner_left = left + stroke_width
        inner_right = right - stroke_width
        inner_bottom = bottom + stroke_width
        inner_top = top - stroke_width
        
        # Create glyph
        glyph = Glyph()
        glyph.numberOfContours = 2
        
        # Outer square (clockwise), inner square (counter-clockwise for hole)
        glyph.coordinates = [
            (left, bottom), (left, top), (right, top), (right, bottom),
            (inner_left, inner_bottom), (inner_right, inner_bottom), 
            (inner_right, inner_top), (inner_left, inner_top)
        ]
        glyph.flags = [1] * 8  # All on-curve points
        glyph.endPtsOfContours = [3, 7]
        
        glyph_order.append(upload_glyph_name)
        glyf.glyphs[upload_glyph_name] = glyph
        hmtx.metrics[upload_glyph_name] = (units_per_em, left)
        cmap[upload_codepoint] = upload_glyph_name
        
        print(f"  Created □ glyph: {upload_glyph_name}")
    
    # Get the existing glyph metrics (use same for all variants since □ has no bold)
    upload_width, upload_lsb = hmtx.metrics[upload_glyph_name]
    upload_glyph_data = glyf[upload_glyph_name]
    
    result = {
        'regular': upload_glyph_name,
    }
    
    pua_index = pua_start
    
    # Create bold_first variant
    bf_codepoint = pua_index
    pua_index += 1
    bf_name = f"uni{bf_codepoint:04X}"
    glyph_order.append(bf_name)
    glyf.glyphs[bf_name] = upload_glyph_data
    hmtx.metrics[bf_name] = (upload_width, upload_lsb)
    cmap[bf_codepoint] = bf_name
    result['bold_first'] = bf_name
    
    # Create bold variant
    bold_codepoint = pua_index
    pua_index += 1
    bold_name = f"uni{bold_codepoint:04X}"
    glyph_order.append(bold_name)
    glyf.glyphs[bold_name] = upload_glyph_data
    hmtx.metrics[bold_name] = (upload_width, upload_lsb)
    cmap[bold_codepoint] = bold_name
    result['bold'] = bold_name
    
    # Create bold_spaced variant
    bs_codepoint = pua_index
    pua_index += 1
    bs_name = f"uni{bs_codepoint:04X}"
    glyph_order.append(bs_name)
    glyf.glyphs[bs_name] = upload_glyph_data
    hmtx.metrics[bs_name] = (upload_width + FUNCTION_GAP, upload_lsb)
    cmap[bs_codepoint] = bs_name
    result['bold_spaced'] = bs_name
    
    # Create regular_spaced variant
    rs_codepoint = pua_index
    pua_index += 1
    rs_name = f"uni{rs_codepoint:04X}"
    glyph_order.append(rs_name)
    glyf.glyphs[rs_name] = upload_glyph_data
    hmtx.metrics[rs_name] = (upload_width + FUNCTION_GAP, upload_lsb)
    cmap[rs_codepoint] = rs_name
    result['regular_spaced'] = rs_name
    
    # Update font
    font.setGlyphOrder(glyph_order)
    glyf.glyphOrder = glyph_order
    
    for table in font['cmap'].tables:
        if hasattr(table, 'cmap'):
            table.cmap.update(cmap)
    
    print(f"  Created □ variants: {result}")
    
    return result


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
    # - bold_first: bold weight, normal spacing (for first char before conversion)
    # - bold: bold weight, normal spacing (for functions with args following)
    # - bold_spaced: bold weight, extra spacing (for complete arity-0 calls and first char)
    # - regular_spaced: regular weight, extra spacing (for last argument of complete calls)
    
    bold_first_glyph_map = {}
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
        
        # Bold_first variant (for initial char detection)
        bf_codepoint = pua_index
        pua_index += 1
        bf_name = f"uni{bf_codepoint:04X}"
        bold_first_glyph_map[char] = bf_name
        new_glyphs.append((bf_name, bold_glyph_data, bold_width, bold_lsb, bf_codepoint))
        
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
    
    print(f"Created {len(bold_first_glyph_map)} bold_first variants")
    print(f"Created {len(bold_glyph_map)} bold variants")
    print(f"Created {len(bold_spaced_glyph_map)} bold+spaced variants")
    print(f"Created {len(regular_spaced_glyph_map)} regular+spaced variants")
    
    for table in font['cmap'].tables:
        if hasattr(table, 'cmap'):
            table.cmap.update(cmap)
    
    # Create □ upload character and all its variants
    upload_variants = create_upload_char_variants(
        font, glyph_order, glyf, hmtx, cmap, pua_index
    )
    
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
        
        # All variants get the same color
        for gmap in [glyph_name_map, bold_first_glyph_map, bold_glyph_map, 
                     bold_spaced_glyph_map, regular_spaced_glyph_map]:
            if char in gmap:
                name = gmap[char]
                color_glyphs[name] = [(name, color_index)]
    
    colr = buildCOLR(color_glyphs)
    font['COLR'] = colr
    
    print(f"Added {len(colors)} colors to palette")
    print(f"Added {len(color_glyphs)} colored glyphs")
    
    # Build GSUB
    print("Building GSUB contextual substitution rules...")
    build_gsub_feature(font, char_defs, qqqlang_chars, arity_map,
                       glyph_name_map, bold_first_glyph_map, bold_glyph_map, 
                       bold_spaced_glyph_map, regular_spaced_glyph_map,
                       upload_variants)
    
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
                       glyph_name_map, bold_first_glyph_map, bold_glyph_map,
                       bold_spaced_glyph_map, regular_spaced_glyph_map,
                       upload_glyph_name=None, upload_spaced_glyph_name=None):
    """
    Build GSUB calt feature.
    
    Key insight: The first character is special (initial color). We detect it by:
    1. Converting all chars to bold_first
    2. Converting any bold_first PRECEDED by bold_first/bold to bold
    3. Only the first char remains bold_first (nothing precedes it)
    
    Then bold_first is NOT in @fn_bold, so it won't be treated as a function
    that consumes arguments.
    
    The □ (upload) character is handled specially - it's not in the char_defs
    but needs spacing when used as an initial image.
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
    all_bold_first = [bold_first_glyph_map[c] for c in qqqlang_chars]
    all_bold = [bold_glyph_map[c] for c in qqqlang_chars]
    all_bold_spaced = [bold_spaced_glyph_map[c] for c in qqqlang_chars]
    all_regular_spaced = [regular_spaced_glyph_map[c] for c in qqqlang_chars]
    
    fea_lines.append(f"@regular = [{' '.join(all_regular)}];")
    fea_lines.append(f"@bold_first = [{' '.join(all_bold_first)}];")
    fea_lines.append(f"@bold = [{' '.join(all_bold)}];")
    fea_lines.append(f"@bold_spaced = [{' '.join(all_bold_spaced)}];")
    fea_lines.append(f"@regular_spaced = [{' '.join(all_regular_spaced)}];")
    
    # @any includes all variants
    all_any = all_regular + all_bold_first + all_bold + all_bold_spaced + all_regular_spaced
    fea_lines.append(f"@any = [{' '.join(all_any)}];")
    
    # @preceded_by for detecting non-first chars (bold_first or bold)
    all_preceded = all_bold_first + all_bold
    fea_lines.append(f"@preceded_by = [{' '.join(all_preceded)}];")
    fea_lines.append("")
    
    # Classes by arity - @fn_bold does NOT include bold_first!
    for arity in sorted(by_arity.keys()):
        chars = by_arity[arity]
        bold = [bold_glyph_map[c] for c in chars]
        fea_lines.append(f"@fn{arity}_bold = [{' '.join(bold)}];")
    fea_lines.append("")
    
    # === LOOKUPS ===
    
    # regular → bold_first
    fea_lines.append("lookup regular_to_bold_first {")
    for char in qqqlang_chars:
        fea_lines.append(f"    sub {glyph_name_map[char]} by {bold_first_glyph_map[char]};")
    fea_lines.append("} regular_to_bold_first;")
    fea_lines.append("")
    
    # bold_first → bold (for non-first chars)
    fea_lines.append("lookup bold_first_to_bold {")
    for char in qqqlang_chars:
        fea_lines.append(f"    sub {bold_first_glyph_map[char]} by {bold_glyph_map[char]};")
    fea_lines.append("} bold_first_to_bold;")
    fea_lines.append("")
    
    # bold → regular (for arguments)
    fea_lines.append("lookup bold_to_regular {")
    for char in qqqlang_chars:
        fea_lines.append(f"    sub {bold_glyph_map[char]} by {glyph_name_map[char]};")
    fea_lines.append("} bold_to_regular;")
    fea_lines.append("")
    
    # regular → regular_spaced (for last arg of complete call)
    fea_lines.append("lookup regular_to_regular_spaced {")
    for char in qqqlang_chars:
        fea_lines.append(f"    sub {glyph_name_map[char]} by {regular_spaced_glyph_map[char]};")
    fea_lines.append("} regular_to_regular_spaced;")
    fea_lines.append("")
    
    # bold → bold_spaced (for arity-0 functions)
    fea_lines.append("lookup bold_to_bold_spaced {")
    for char in qqqlang_chars:
        fea_lines.append(f"    sub {bold_glyph_map[char]} by {bold_spaced_glyph_map[char]};")
    fea_lines.append("} bold_to_bold_spaced;")
    fea_lines.append("")
    
    # bold_first → bold_spaced (for first char, which is complete)
    fea_lines.append("lookup bold_first_to_bold_spaced {")
    for char in qqqlang_chars:
        fea_lines.append(f"    sub {bold_first_glyph_map[char]} by {bold_spaced_glyph_map[char]};")
    fea_lines.append("} bold_first_to_bold_spaced;")
    fea_lines.append("")
    
    # === PASS LOOKUPS (defined outside feature for explicit ordering) ===
    
    # Pass 1a: Convert all regular to bold_first
    fea_lines.append("lookup pass1a {")
    fea_lines.append("    sub @regular' lookup regular_to_bold_first;")
    fea_lines.append("} pass1a;")
    fea_lines.append("")
    
    # Pass 1b: Any bold_first preceded by @preceded_by becomes bold
    fea_lines.append("lookup pass1b {")
    fea_lines.append("    sub @preceded_by @bold_first' lookup bold_first_to_bold;")
    fea_lines.append("} pass1b;")
    fea_lines.append("")
    
    # Pass 2: Un-bold arguments
    #
    # We use a SINGLE lookup with all rules, ordered by pattern length (longest first).
    # This ensures longer patterns match before shorter ones at each position.
    #
    # For example, with arity-2 functions:
    #   sub @fn2_bold @any @bold' ...  (3-glyph pattern for arg2)
    #   sub @fn2_bold @bold' ...        (2-glyph pattern for arg1)
    #
    # When scanning "L L L L L" at position 1:
    #   - First try 3-glyph rule: matches positions 1,2,3 → substitute position 3
    #   - Skip to position 4
    #   - At position 4: try 3-glyph rule (needs pos 5,6) - no match
    #   - Try 2-glyph rule (needs pos 5) - if exists, match
    #
    # The key is that within a single lookup, after a match, we skip the ENTIRE
    # matched sequence before trying rules again.
    
    fea_lines.append("lookup pass2_unbold_args {")
    
    # Generate rules ordered by total pattern length (longest first)
    # Pattern length = 1 (fn) + arg_pos (number of @any + @bold)
    pass2_rules = []
    for arg_pos in range(max_arity, 0, -1):
        for arity in sorted([a for a in by_arity.keys() if a >= arg_pos], reverse=True):
            fn_class = f"@fn{arity}_bold"
            preceding = " @any" * (arg_pos - 1)
            pattern_len = 1 + arg_pos  # fn + args
            rule = f"    sub {fn_class}{preceding} @bold' lookup bold_to_regular;"
            pass2_rules.append((pattern_len, arity, arg_pos, rule))
    
    # Sort by pattern length (descending), then by arity (descending)
    pass2_rules.sort(key=lambda x: (-x[0], -x[1]))
    
    for _, _, _, rule in pass2_rules:
        fea_lines.append(rule)
    
    fea_lines.append("} pass2_unbold_args;")
    fea_lines.append("")
    
    pass2_lookup_names = ["pass2_unbold_args"]
    
    # Pass 3: Add spacing to last arg of COMPLETE calls
    fea_lines.append("lookup pass3_spacing {")
    for arity in sorted([a for a in by_arity.keys() if a > 0], reverse=True):
        fn_class = f"@fn{arity}_bold"
        preceding_any = " @any" * (arity - 1)
        fea_lines.append(f"    sub {fn_class}{preceding_any} @regular' lookup regular_to_regular_spaced;")
    fea_lines.append("} pass3_spacing;")
    fea_lines.append("")
    
    # Pass 4: Arity-0 functions get spacing
    if 0 in by_arity:
        fea_lines.append("lookup pass4_arity0 {")
        fea_lines.append("    sub @fn0_bold' lookup bold_to_bold_spaced;")
        fea_lines.append("} pass4_arity0;")
        fea_lines.append("")
    
    # Pass 5: First char (bold_first) gets spacing
    fea_lines.append("lookup pass5_first {")
    fea_lines.append("    sub @bold_first' lookup bold_first_to_bold_spaced;")
    fea_lines.append("} pass5_first;")
    fea_lines.append("")
    
    # Pass 6: Upload character □ gets spacing (it acts like an initial image)
    has_upload_char = upload_glyph_name and upload_spaced_glyph_name
    if has_upload_char:
        fea_lines.append("lookup upload_to_spaced {")
        fea_lines.append(f"    sub {upload_glyph_name} by {upload_spaced_glyph_name};")
        fea_lines.append("} upload_to_spaced;")
        fea_lines.append("")
        
        fea_lines.append("lookup pass6_upload {")
        fea_lines.append(f"    sub {upload_glyph_name}' lookup upload_to_spaced;")
        fea_lines.append("} pass6_upload;")
        fea_lines.append("")
    
    # === FEATURE (references lookups in order) ===
    fea_lines.append("feature calt {")
    fea_lines.append("    lookup pass1a;")
    fea_lines.append("    lookup pass1b;")
    for lookup_name in pass2_lookup_names:
        fea_lines.append(f"    lookup {lookup_name};")
    fea_lines.append("    lookup pass3_spacing;")
    if 0 in by_arity:
        fea_lines.append("    lookup pass4_arity0;")
    fea_lines.append("    lookup pass5_first;")
    fea_lines.append("} calt;")
    
    fea_code = "\n".join(fea_lines)
    
    rule_count = len([l for l in fea_lines if l.strip().startswith('sub ')])
    print(f"  Generated {rule_count} substitution rules")
    print(f"  Feature code: {len(fea_code)} bytes")
    
    # Write feature code to file for inspection
    fea_file = PROJECT_DIR / "debug_feature.fea"
    fea_file.write_text(fea_code)
    print(f"  Wrote feature code to {fea_file}")
    
    # Print key lookups for debugging
    print("\n  === KEY LOOKUPS ===")
    in_lookup = None
    for line in fea_lines:
        if line.startswith("lookup pass"):
            in_lookup = line
            print(f"  {line}")
        elif in_lookup and line.startswith("} "):
            in_lookup = None
        elif in_lookup and "sub " in line:
            print(f"  {line}")
    
    addOpenTypeFeatures(font, StringIO(fea_code))


if __name__ == "__main__":
    build_font()
