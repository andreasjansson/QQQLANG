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
FUNCTION_GAP = 0

REGULAR_WEIGHT = 400
BOLD_WEIGHT = 700

# Upload character configuration
# Using Mathematical Alphanumeric Symbols (U+1D400-U+1D7FF) because:
# - Chrome has GSUB issues with PUA codepoints
# - Hangul codepoints cause HarfBuzz to detect script as Hang, breaking GSUB
# - Math Alphanumeric is detected as Latin script and works in Chrome
UPLOAD_CHAR = '□'  # U+25A1 - unassigned upload placeholder
UPLOAD_COUNT = 256
UPLOAD_REGULAR_BASE = 0x1D400  # U+1D400 to U+1D4FF: valid upload □
UPLOAD_INVALID_BASE = 0x1D500  # U+1D500 to U+1D5FF: invalid upload ■
UPLOAD_GSUB_BASE = 0x1D600    # U+1D600+: upload GSUB variants (bold_first, regular_spaced)


def parse_character_defs():
    content = CHARACTER_DEFS_PATH.read_text()
    
    # Match character definitions more robustly by finding the key and then parsing the object
    # Pattern matches both quoted ("X": {) and unquoted (X: {) keys
    # Unquoted: alphanumeric or $ or _ (valid JS identifiers that don't need quotes)
    # Quoted: string literal followed by colon (for symbols that need quotes)
    pattern = r"(?:(['\"])([^'\"]+)\1|([A-Z0-9$_])):\s*\{"
    
    chars = {}
    for match in re.finditer(pattern, content):
        # Extract character - either from quoted string (group 2) or unquoted identifier (group 3)
        char = match.group(2) if match.group(2) else match.group(3)
        
        # Handle escape sequences from TypeScript source
        if char == '\\\\':
            char = '\\'
        elif char == "\\'":
            char = "'"
        
        # Find the matching closing brace by counting brace depth
        start_pos = match.end()
        brace_count = 1
        pos = start_pos
        
        while pos < len(content) and brace_count > 0:
            if content[pos] == '{':
                brace_count += 1
            elif content[pos] == '}':
                brace_count -= 1
            pos += 1
        
        if brace_count != 0:
            continue  # Couldn't find matching brace
        
        obj_content = content[start_pos:pos-1]
        
        # Extract color
        color_match = re.search(r'color:\s*["\']([^"\']+)["\']', obj_content)
        if not color_match:
            continue
        color = color_match.group(1)
        
        # Extract number
        number_match = re.search(r'number:\s*(\d+)', obj_content)
        if not number_match:
            continue
        number = int(number_match.group(1))
        
        # Extract args array and count entries
        args_match = re.search(r'args:\s*\[([\s\S]*?)\]', obj_content)
        if args_match:
            args_str = args_match.group(1)
            # Count objects in args array by counting { that start an argument object
            # We look for patterns like "{ type:" or "{\n    type:"
            arity = len(re.findall(r'\{\s*type:', args_str))
        else:
            arity = 0
        
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


def create_indexed_upload_chars(font, glyph_order, glyf, hmtx, cmap, pua_start):
    """
    Create 256 indexed upload characters, each with variants for GSUB.
    
    For each index i (0-255):
    - regular □ at U+E200+i: Base codepoint in text, used as INDEX argument
    - invalid ■ at U+E300+i: Filled square for invalid positions (JS replaces)
    - bold_first □ in PUA: For GSUB first-char detection
    - regular_spaced □ in PUA: For last arg of complete call or first char
    
    Upload chars can only be:
    1. First character (initial image) - uses bold_first → regular_spaced
    2. INDEX argument - uses regular or regular_spaced
    
    They CANNOT be functions, so no bold/bold_spaced variants needed.
    
    Returns dict with:
    - 'all_regular': list of all 256 regular glyph names
    - 'all_bold_first': list of all 256 bold_first glyph names
    - 'all_regular_spaced': list of all 256 regular_spaced glyph names
    - 'all_invalid': list of all 256 invalid glyph names
    """
    from fontTools.ttLib.tables._g_l_y_f import Glyph, GlyphCoordinates
    from fontTools.ttLib.tables import ttProgram
    
    units_per_em = font['head'].unitsPerEm
    
    # Create □ outline glyph (hollow square)
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
    
    outline_glyph = Glyph()
    outline_glyph.numberOfContours = 2
    outline_glyph.coordinates = GlyphCoordinates([
        (left, bottom), (left, top), (right, top), (right, bottom),
        (inner_left, inner_bottom), (inner_right, inner_bottom), 
        (inner_right, inner_top), (inner_left, inner_top)
    ])
    outline_glyph.flags = [1] * 8
    outline_glyph.endPtsOfContours = [3, 7]
    outline_glyph.program = ttProgram.Program()
    outline_glyph.xMin = left
    outline_glyph.yMin = bottom
    outline_glyph.xMax = right
    outline_glyph.yMax = top
    
    # Create ■ filled glyph (solid square)
    filled_glyph = Glyph()
    filled_glyph.numberOfContours = 1
    filled_glyph.coordinates = GlyphCoordinates([
        (left, bottom), (left, top), (right, top), (right, bottom)
    ])
    filled_glyph.flags = [1] * 4
    filled_glyph.endPtsOfContours = [3]
    filled_glyph.program = ttProgram.Program()
    filled_glyph.xMin = left
    filled_glyph.yMin = bottom
    filled_glyph.xMax = right
    filled_glyph.yMax = top
    
    width = units_per_em
    lsb = left
    
    all_regular = []
    all_bold_first = []
    all_regular_spaced = []
    all_invalid = []
    
    # Use dedicated range for upload GSUB variants to avoid collision with invalid upload range
    pua_index = UPLOAD_GSUB_BASE
    
    print(f"  Creating {UPLOAD_COUNT} indexed upload characters...")
    
    for i in range(UPLOAD_COUNT):
        # Regular □ at U+E200+i
        reg_codepoint = UPLOAD_REGULAR_BASE + i
        reg_name = f"upload_{i}"
        glyph_order.append(reg_name)
        glyf.glyphs[reg_name] = outline_glyph
        hmtx.metrics[reg_name] = (width, lsb)
        cmap[reg_codepoint] = reg_name
        all_regular.append(reg_name)
        
        # Invalid ■ at U+E300+i
        inv_codepoint = UPLOAD_INVALID_BASE + i
        inv_name = f"upload_{i}_invalid"
        glyph_order.append(inv_name)
        glyf.glyphs[inv_name] = filled_glyph
        hmtx.metrics[inv_name] = (width, lsb)
        cmap[inv_codepoint] = inv_name
        all_invalid.append(inv_name)
        
        # Bold_first □ in PUA (for GSUB)
        bf_codepoint = pua_index
        pua_index += 1
        bf_name = f"upload_{i}_bf"
        glyph_order.append(bf_name)
        glyf.glyphs[bf_name] = outline_glyph
        hmtx.metrics[bf_name] = (width, lsb)
        cmap[bf_codepoint] = bf_name
        all_bold_first.append(bf_name)
        
        # Regular_spaced □ in PUA (for GSUB)
        rs_codepoint = pua_index
        pua_index += 1
        rs_name = f"upload_{i}_rs"
        glyph_order.append(rs_name)
        glyf.glyphs[rs_name] = outline_glyph
        hmtx.metrics[rs_name] = (width + FUNCTION_GAP, lsb)
        cmap[rs_codepoint] = rs_name
        all_regular_spaced.append(rs_name)
    
    # Update font
    font.setGlyphOrder(glyph_order)
    glyf.glyphOrder = glyph_order
    
    for table in font['cmap'].tables:
        if hasattr(table, 'cmap'):
            table.cmap.update(cmap)
    
    print(f"  Created {UPLOAD_COUNT} regular □ glyphs (U+AC00-U+ACFF)")
    print(f"  Created {UPLOAD_COUNT} invalid ■ glyphs (U+AD00-U+ADFF)")
    print(f"  Created {UPLOAD_COUNT} bold_first variants")
    print(f"  Created {UPLOAD_COUNT} regular_spaced variants")
    
    return {
        'all_regular': all_regular,
        'all_bold_first': all_bold_first,
        'all_regular_spaced': all_regular_spaced,
        'all_invalid': all_invalid,
    }


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
    
    # Helper to create a glyph with a dot underneath
    from fontTools.ttLib.tables._g_l_y_f import Glyph, GlyphCoordinates
    from fontTools.ttLib.tables import ttProgram
    
    def create_glyph_with_dot(original_glyph, color_hex):
        """Create a composite glyph with original character and a colored dot underneath"""
        from fontTools.pens.t2CharStringPen import T2CharStringPen
        from fontTools.pens.ttGlyphPen import TTGlyphPen
        
        # Create a new glyph with dot underneath
        units_per_em = font['head'].unitsPerEm
        
        # Dot parameters
        dot_radius = int(units_per_em * 0.06)  # Small dot
        # Center dot based on actual character bounds, not em-square
        if hasattr(original_glyph, 'xMin') and hasattr(original_glyph, 'xMax'):
            dot_center_x = (original_glyph.xMin + original_glyph.xMax) // 2
        else:
            dot_center_x = units_per_em // 2  # Fallback to em-square center
        dot_center_y = int(-units_per_em * 0.15)  # Below baseline
        
        # Create circular dot with 8 points
        import math
        dot_glyph = Glyph()
        dot_glyph.numberOfContours = 1
        
        points = []
        for i in range(8):
            angle = 2 * math.pi * i / 8
            x = dot_center_x + int(dot_radius * math.cos(angle))
            y = dot_center_y + int(dot_radius * math.sin(angle))
            points.append((x, y))
        
        dot_glyph.coordinates = GlyphCoordinates(points)
        dot_glyph.flags = [1] * 8  # All on-curve points
        dot_glyph.endPtsOfContours = [7]
        dot_glyph.program = ttProgram.Program()
        
        # Calculate bounds including dot
        dot_glyph.xMin = dot_center_x - dot_radius
        dot_glyph.yMin = dot_center_y - dot_radius
        dot_glyph.xMax = dot_center_x + dot_radius
        dot_glyph.yMax = dot_center_y + dot_radius
        
        # Combine with original glyph
        if hasattr(original_glyph, 'numberOfContours') and original_glyph.numberOfContours > 0:
            combined = Glyph()
            combined.numberOfContours = original_glyph.numberOfContours + 1
            
            # Combine coordinates
            orig_coords = list(original_glyph.coordinates)
            dot_coords = list(dot_glyph.coordinates)
            combined.coordinates = GlyphCoordinates(orig_coords + dot_coords)
            
            # Combine flags
            combined.flags = list(original_glyph.flags) + list(dot_glyph.flags)
            
            # Combine contour end points (offset dot contour indices)
            orig_end_pts = list(original_glyph.endPtsOfContours)
            dot_end_pt = len(orig_coords) + 7  # Dot ends at 8th point after original
            combined.endPtsOfContours = orig_end_pts + [dot_end_pt]
            
            combined.program = ttProgram.Program()
            
            # Update bounds
            combined.xMin = min(original_glyph.xMin, dot_glyph.xMin)
            combined.yMin = min(original_glyph.yMin, dot_glyph.yMin)
            combined.xMax = max(original_glyph.xMax, dot_glyph.xMax)
            combined.yMax = max(original_glyph.yMax, dot_glyph.yMax)
            
            return combined
        else:
            return dot_glyph
    
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
        
        # Get color for this character
        char_color = char_defs[char]['color']
        
        # Bold_first variant (for initial char detection) - now with dot instead of bold
        bf_codepoint = pua_index
        pua_index += 1
        bf_name = f"uni{bf_codepoint:04X}"
        bold_first_glyph_map[char] = bf_name
        # Use regular glyph with dot underneath instead of bold
        bf_glyph_with_dot = create_glyph_with_dot(regular_glyph_data, char_color)
        new_glyphs.append((bf_name, bf_glyph_with_dot, regular_width, regular_lsb, bf_codepoint))
        
        # Bold variant (actually regular with dot - used for functions)
        bold_codepoint = pua_index
        pua_index += 1
        bold_name = f"uni{bold_codepoint:04X}"
        bold_glyph_map[char] = bold_name
        # Use regular glyph with dot instead of bold
        bold_glyph_with_dot = create_glyph_with_dot(regular_glyph_data, char_color)
        new_glyphs.append((bold_name, bold_glyph_with_dot, regular_width, regular_lsb, bold_codepoint))
        
        # Bold+spaced variant (actually regular with dot, kept for spacing infrastructure)
        bold_spaced_codepoint = pua_index
        pua_index += 1
        bold_spaced_name = f"uni{bold_spaced_codepoint:04X}"
        bold_spaced_glyph_map[char] = bold_spaced_name
        # Use regular glyph with dot (same as bold_first) for consistency
        bs_glyph_with_dot = create_glyph_with_dot(regular_glyph_data, char_color)
        new_glyphs.append((bold_spaced_name, bs_glyph_with_dot, regular_width + FUNCTION_GAP, regular_lsb, bold_spaced_codepoint))
        
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
    
    # Create 256 indexed upload characters with variants
    upload_chars = create_indexed_upload_chars(
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
                       upload_chars)
    
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
                       upload_chars=None):
    """
    Build GSUB calt feature.
    
    Key insight: The first character is special (initial color). We detect it by:
    1. Converting all chars to bold_first
    2. Converting any bold_first PRECEDED by bold_first/bold to bold
    3. Only the first char remains bold_first (nothing precedes it)
    
    Then bold_first is NOT in @fn_bold, so it won't be treated as a function
    that consumes arguments.
    
    Upload characters (□) can only appear as:
    1. First character (initial image) - bold_first → regular_spaced
    2. INDEX argument to a function - regular or regular_spaced
    
    They CANNOT be functions, so they don't need bold/bold_spaced variants.
    Upload bold_first converts directly to regular (not bold) when not first.
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
    
    # Define glyph classes for regular characters
    all_regular = [glyph_name_map[c] for c in qqqlang_chars]
    all_bold_first = [bold_first_glyph_map[c] for c in qqqlang_chars]
    all_bold = [bold_glyph_map[c] for c in qqqlang_chars]
    all_bold_spaced = [bold_spaced_glyph_map[c] for c in qqqlang_chars]
    all_regular_spaced = [regular_spaced_glyph_map[c] for c in qqqlang_chars]
    
    # Upload characters have their own variants (no bold/bold_spaced)
    upload_regular = upload_chars['all_regular'] if upload_chars else []
    upload_bold_first = upload_chars['all_bold_first'] if upload_chars else []
    upload_regular_spaced = upload_chars['all_regular_spaced'] if upload_chars else []
    
    # Add upload variants to the main classes
    all_regular.extend(upload_regular)
    all_bold_first.extend(upload_bold_first)
    all_regular_spaced.extend(upload_regular_spaced)
    # Note: uploads don't have bold or bold_spaced variants
    
    fea_lines.append(f"@regular = [{' '.join(all_regular)}];")
    fea_lines.append(f"@bold_first = [{' '.join(all_bold_first)}];")
    fea_lines.append(f"@bold = [{' '.join(all_bold)}];")
    fea_lines.append(f"@bold_spaced = [{' '.join(all_bold_spaced)}];")
    fea_lines.append(f"@regular_spaced = [{' '.join(all_regular_spaced)}];")
    
    # Upload-specific classes (uploads don't have bold/bold_spaced)
    if upload_chars:
        fea_lines.append(f"@upload_regular = [{' '.join(upload_regular)}];")
        fea_lines.append(f"@upload_bold_first = [{' '.join(upload_bold_first)}];")
        fea_lines.append(f"@upload_regular_spaced = [{' '.join(upload_regular_spaced)}];")
    
    # @any includes all variants (uploads don't have bold/bold_spaced)
    all_any = all_regular + all_bold_first + all_bold + all_bold_spaced + all_regular_spaced
    fea_lines.append(f"@any = [{' '.join(all_any)}];")
    
    # @preceded_by for detecting non-first chars (bold_first or bold)
    # Upload bold_first is included so it gets converted when not first
    all_preceded = all_bold_first + all_bold
    fea_lines.append(f"@preceded_by = [{' '.join(all_preceded)}];")
    fea_lines.append("")
    
    # Classes by arity - @fn_bold does NOT include bold_first or uploads!
    # Uploads cannot be functions, only first char or INDEX arguments
    for arity in sorted(by_arity.keys()):
        chars = by_arity[arity]
        bold = [bold_glyph_map[c] for c in chars]
        fea_lines.append(f"@fn{arity}_bold = [{' '.join(bold)}];")
    fea_lines.append("")
    
    # === LOOKUPS ===
    
    # regular → bold_first (for all chars including uploads)
    fea_lines.append("lookup regular_to_bold_first {")
    for char in qqqlang_chars:
        fea_lines.append(f"    sub {glyph_name_map[char]} by {bold_first_glyph_map[char]};")
    if upload_chars:
        for i in range(UPLOAD_COUNT):
            fea_lines.append(f"    sub {upload_regular[i]} by {upload_bold_first[i]};")
    fea_lines.append("} regular_to_bold_first;")
    fea_lines.append("")
    
    # bold_first → bold (for non-first regular chars only, not uploads)
    fea_lines.append("lookup bold_first_to_bold {")
    for char in qqqlang_chars:
        fea_lines.append(f"    sub {bold_first_glyph_map[char]} by {bold_glyph_map[char]};")
    fea_lines.append("} bold_first_to_bold;")
    fea_lines.append("")
    
    # upload_bold_first → upload_regular (uploads skip bold state)
    if upload_chars:
        fea_lines.append("lookup upload_bold_first_to_regular {")
        for i in range(UPLOAD_COUNT):
            fea_lines.append(f"    sub {upload_bold_first[i]} by {upload_regular[i]};")
        fea_lines.append("} upload_bold_first_to_regular;")
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
    if upload_chars:
        for i in range(UPLOAD_COUNT):
            fea_lines.append(f"    sub {upload_regular[i]} by {upload_regular_spaced[i]};")
    fea_lines.append("} regular_to_regular_spaced;")
    fea_lines.append("")
    
    # bold → bold_spaced (for arity-0 functions, not uploads)
    fea_lines.append("lookup bold_to_bold_spaced {")
    for char in qqqlang_chars:
        fea_lines.append(f"    sub {bold_glyph_map[char]} by {bold_spaced_glyph_map[char]};")
    fea_lines.append("} bold_to_bold_spaced;")
    fea_lines.append("")
    
    # bold_first → bold_spaced (for first regular char)
    # NOTE: Currently FUNCTION_GAP = 0, so bold_spaced has same spacing as bold.
    # We keep this infrastructure in case we want to add spacing after first char later.
    # The "bold" variants now actually use regular weight + dot, not bold weight.
    fea_lines.append("lookup bold_first_to_bold_spaced {")
    for char in qqqlang_chars:
        fea_lines.append(f"    sub {bold_first_glyph_map[char]} by {bold_spaced_glyph_map[char]};")
    fea_lines.append("} bold_first_to_bold_spaced;")
    fea_lines.append("")
    
    # upload_bold_first → upload_regular_spaced (for first upload char)
    if upload_chars:
        fea_lines.append("lookup upload_bold_first_to_regular_spaced {")
        for i in range(UPLOAD_COUNT):
            fea_lines.append(f"    sub {upload_bold_first[i]} by {upload_regular_spaced[i]};")
        fea_lines.append("} upload_bold_first_to_regular_spaced;")
        fea_lines.append("")
    
    # === PASS LOOKUPS (defined outside feature for explicit ordering) ===
    
    # Pass 1a: Convert all regular to bold_first
    fea_lines.append("lookup pass1a {")
    fea_lines.append("    sub @regular' lookup regular_to_bold_first;")
    fea_lines.append("} pass1a;")
    fea_lines.append("")
    
    # Pass 1b: Any bold_first preceded by @preceded_by becomes bold (regular chars)
    # Upload bold_first goes directly to regular (skips bold state)
    fea_lines.append("lookup pass1b {")
    fea_lines.append("    sub @preceded_by @bold_first' lookup bold_first_to_bold;")
    if upload_chars:
        fea_lines.append("    sub @preceded_by @upload_bold_first' lookup upload_bold_first_to_regular;")
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
    # Regular chars: bold_first → bold_spaced
    # Upload chars: bold_first → regular_spaced (uploads don't have bold_spaced)
    fea_lines.append("lookup pass5_first {")
    fea_lines.append("    sub @bold_first' lookup bold_first_to_bold_spaced;")
    if upload_chars:
        fea_lines.append("    sub @upload_bold_first' lookup upload_bold_first_to_regular_spaced;")
    fea_lines.append("} pass5_first;")
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
