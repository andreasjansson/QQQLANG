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

import json
import re
import sys
from pathlib import Path
from fontTools.ttLib import TTFont
from fontTools.varLib.mutator import instantiateVariableFont
from fontTools.ttLib.tables import otTables
from fontTools.colorLib.builder import buildCOLR, buildCPAL
from fontTools.pens.t2CharStringPen import T2CharStringPen
from fontTools.fontBuilder import FontBuilder

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


def build_gsub_rules(char_defs, glyph_name_map, bold_glyph_map, spaced_glyph_map):
    """
    Build GSUB calt rules for context-dependent substitution.
    
    The logic:
    - At the start of text or after a complete function call, the next char is a function (bold + possibly spaced)
    - After a function char, the next N chars are arguments (regular), where N = arity
    - After all arguments, the last argument gets extra spacing
    
    We implement this using chained contextual substitution (GSUB lookup type 6).
    """
    qqqlang_chars = [c for c in char_defs.keys() if len(c) == 1]
    
    # Group characters by arity for rule generation
    arity_groups = {}
    for char, info in char_defs.items():
        if len(char) != 1:
            continue
        arity = info['arity']
        if arity not in arity_groups:
            arity_groups[arity] = []
        arity_groups[arity].append(char)
    
    rules = []
    
    # For arity 0 functions: substitute to bold+spaced variant
    # Rule: any_qqqlang_char -> bold_spaced_char (when it's arity 0)
    for char in arity_groups.get(0, []):
        if char in bold_glyph_map and char in spaced_glyph_map:
            # Use spaced variant (which is also bold)
            rules.append({
                'type': 'single',
                'input': [glyph_name_map[char]],
                'output': [spaced_glyph_map[char]],
            })
    
    # For arity > 0 functions, we need contextual rules
    # This is complex because we need to:
    # 1. Make the function char bold (but not spaced, since args follow)
    # 2. Keep argument chars regular
    # 3. Make the last argument spaced
    
    # Since OpenType calt processes left-to-right and we can't "look back" easily,
    # we use a different approach: enumerate all possible patterns
    
    return rules, arity_groups


def create_calt_feature(font, char_defs, glyph_name_map, bold_glyph_map, spaced_glyph_map):
    """
    Create the 'calt' feature for contextual alternates.
    
    Strategy: Use reverse chaining single substitution (GSUB lookup type 8)
    which processes from end to start, allowing us to:
    1. First pass: identify and mark last-argument positions with spacing
    2. Second pass: substitute function chars to bold
    
    Actually, let's use a simpler approach with multiple lookups:
    - Lookup 1: Bold substitution (all qqqlang chars -> bold variants by default)
    - Lookup 2: De-bold substitution for argument positions (bold -> regular)
    - Lookup 3: Add spacing to end-of-function positions
    """
    gsub = font['GSUB'].table
    
    qqqlang_chars = sorted([c for c in char_defs.keys() if len(c) == 1])
    
    # Get all glyph names
    all_regular = [glyph_name_map[c] for c in qqqlang_chars if c in glyph_name_map]
    all_bold = [bold_glyph_map[c] for c in qqqlang_chars if c in bold_glyph_map]
    all_spaced = [spaced_glyph_map[c] for c in qqqlang_chars if c in spaced_glyph_map]
    
    # Create glyph classes
    # Class for all regular QQQLANG chars
    # Class for all bold QQQLANG chars
    
    lookups = []
    
    # === LOOKUP 0: Single substitution - all regular chars to bold ===
    lookup0 = otTables.Lookup()
    lookup0.LookupType = 1  # Single substitution
    lookup0.LookupFlag = 0
    lookup0.SubTableCount = 1
    lookup0.SubTable = [otTables.SingleSubst()]
    lookup0.SubTable[0].mapping = {}
    
    for char in qqqlang_chars:
        if char in glyph_name_map and char in bold_glyph_map:
            lookup0.SubTable[0].mapping[glyph_name_map[char]] = bold_glyph_map[char]
    
    lookups.append(lookup0)
    
    # === LOOKUP 1: Contextual substitution - de-bold argument positions ===
    # For each function with arity > 0, we need rules like:
    # For arity 1: FN_CHAR ARG -> FN_CHAR arg (de-bold the arg)
    # For arity 2: FN_CHAR ARG1 ARG2 -> FN_CHAR arg1 arg2 (de-bold both args)
    # etc.
    
    # Group by arity
    arity_groups = {}
    for char, info in char_defs.items():
        if len(char) != 1:
            continue
        arity = info['arity']
        if arity not in arity_groups:
            arity_groups[arity] = []
        arity_groups[arity].append(char)
    
    # We'll create chained contextual substitution rules
    # GSUB Lookup Type 6: Chaining Contextual Substitution
    
    lookup1 = otTables.Lookup()
    lookup1.LookupType = 6  # Chaining contextual substitution
    lookup1.LookupFlag = 0
    lookup1.SubTable = []
    
    # Create a nested lookup for de-bolding (bold -> regular)
    debold_lookup = otTables.Lookup()
    debold_lookup.LookupType = 1  # Single substitution
    debold_lookup.LookupFlag = 0
    debold_lookup.SubTableCount = 1
    debold_lookup.SubTable = [otTables.SingleSubst()]
    debold_lookup.SubTable[0].mapping = {}
    
    for char in qqqlang_chars:
        if char in bold_glyph_map and char in glyph_name_map:
            debold_lookup.SubTable[0].mapping[bold_glyph_map[char]] = glyph_name_map[char]
    
    lookups.append(debold_lookup)
    debold_lookup_index = 1
    
    # Create a nested lookup for adding spacing (bold -> spaced)
    space_lookup = otTables.Lookup()
    space_lookup.LookupType = 1  # Single substitution
    space_lookup.LookupFlag = 0
    space_lookup.SubTableCount = 1
    space_lookup.SubTable = [otTables.SingleSubst()]
    space_lookup.SubTable[0].mapping = {}
    
    for char in qqqlang_chars:
        if char in bold_glyph_map and char in spaced_glyph_map:
            space_lookup.SubTable[0].mapping[bold_glyph_map[char]] = spaced_glyph_map[char]
        if char in glyph_name_map and char in spaced_glyph_map:
            space_lookup.SubTable[0].mapping[glyph_name_map[char]] = spaced_glyph_map[char]
    
    lookups.append(space_lookup)
    space_lookup_index = 2
    
    # Now create chained contextual rules for each arity
    # Format 3 (Coverage-based) is most flexible
    
    for arity, fn_chars in arity_groups.items():
        if arity == 0:
            # Arity 0: just needs spacing (handled separately)
            # Rule: [any bold arity-0 char] -> apply space_lookup
            subtable = otTables.ChainContextSubst()
            subtable.Format = 3
            
            # Backtrack: nothing
            subtable.BacktrackCount = 0
            subtable.BacktrackCoverage = []
            
            # Input: one glyph from arity-0 bold chars
            arity0_bold_glyphs = [bold_glyph_map[c] for c in fn_chars if c in bold_glyph_map]
            if not arity0_bold_glyphs:
                continue
                
            input_coverage = otTables.Coverage()
            input_coverage.glyphs = arity0_bold_glyphs
            subtable.InputCount = 1
            subtable.InputCoverage = [input_coverage]
            
            # Lookahead: nothing
            subtable.LookAheadCount = 0
            subtable.LookAheadCoverage = []
            
            # Substitution: apply space lookup to position 0
            subtable.SubstCount = 1
            rec = otTables.SubstLookupRecord()
            rec.SequenceIndex = 0
            rec.LookupListIndex = space_lookup_index
            subtable.SubstLookupRecord = [rec]
            
            lookup1.SubTable.append(subtable)
            
        else:
            # Arity > 0: need to de-bold arguments and add spacing to last arg
            # Rule: [bold fn char] [bold arg1] ... [bold argN] 
            #       -> [bold fn char] [regular arg1] ... [spaced argN]
            
            # Get bold glyphs for these function chars
            fn_bold_glyphs = [bold_glyph_map[c] for c in fn_chars if c in bold_glyph_map]
            if not fn_bold_glyphs:
                continue
            
            # All bold glyphs (for arguments)
            all_bold_glyphs = [bold_glyph_map[c] for c in qqqlang_chars if c in bold_glyph_map]
            
            # Create rule for de-bolding each argument position
            for arg_pos in range(arity):
                subtable = otTables.ChainContextSubst()
                subtable.Format = 3
                
                # Backtrack: the function char + previous args
                # (we're looking at arg_pos, so backtrack is fn + args 0..arg_pos-1)
                backtrack_coverages = []
                
                # First backtrack is the function char
                fn_cov = otTables.Coverage()
                fn_cov.glyphs = fn_bold_glyphs
                backtrack_coverages.append(fn_cov)
                
                # Then previous argument positions (as bold chars that might have been de-bolded)
                # Actually, by the time we process arg_pos, previous args are still bold
                # because we process left-to-right... hmm, this is tricky
                
                # Let's use a different approach: process all args at once per function type
                # Skip for now and handle differently
                
            # Actually, let's use a simpler approach:
            # Create one rule per function character that matches the full pattern
            # and applies multiple substitutions
            
            for fn_char in fn_chars:
                if fn_char not in bold_glyph_map:
                    continue
                    
                subtable = otTables.ChainContextSubst()
                subtable.Format = 3
                
                # Backtrack: nothing (or could be any non-qqqlang or start)
                subtable.BacktrackCount = 0
                subtable.BacktrackCoverage = []
                
                # Input: fn_char + arity args
                input_coverages = []
                
                # First input: the specific function char (bold)
                fn_cov = otTables.Coverage()
                fn_cov.glyphs = [bold_glyph_map[fn_char]]
                input_coverages.append(fn_cov)
                
                # Following inputs: any bold qqqlang char (the arguments)
                for i in range(arity):
                    arg_cov = otTables.Coverage()
                    arg_cov.glyphs = all_bold_glyphs
                    input_coverages.append(arg_cov)
                
                subtable.InputCount = 1 + arity
                subtable.InputCoverage = input_coverages
                
                # Lookahead: nothing
                subtable.LookAheadCount = 0
                subtable.LookAheadCoverage = []
                
                # Substitutions:
                # - Position 0 (fn char): keep bold (no substitution needed)
                # - Positions 1 to arity-1: de-bold
                # - Position arity (last arg): de-bold AND add spacing
                subst_records = []
                
                for i in range(1, arity):
                    # De-bold this argument
                    rec = otTables.SubstLookupRecord()
                    rec.SequenceIndex = i
                    rec.LookupListIndex = debold_lookup_index
                    subst_records.append(rec)
                
                # Last argument: de-bold then space (need two lookups, but can only apply one per position)
                # Solution: create a combined lookup that goes bold -> spaced (not bold)
                # Actually our space_lookup already maps bold -> spaced, so just use that
                rec = otTables.SubstLookupRecord()
                rec.SequenceIndex = arity  # Last argument position
                rec.LookupListIndex = space_lookup_index
                subst_records.append(rec)
                
                subtable.SubstCount = len(subst_records)
                subtable.SubstLookupRecord = subst_records
                
                lookup1.SubTable.append(subtable)
    
    if lookup1.SubTable:
        lookup1.SubTableCount = len(lookup1.SubTable)
        lookups.insert(1, lookup1)  # Insert after the initial bold substitution
        # Adjust indices
        debold_lookup_index = 2
        space_lookup_index = 3
    
    # Update the lookup list
    if not hasattr(gsub, 'LookupList') or gsub.LookupList is None:
        gsub.LookupList = otTables.LookupList()
        gsub.LookupList.Lookup = []
    
    base_index = len(gsub.LookupList.Lookup)
    gsub.LookupList.Lookup.extend(lookups)
    
    # Create or update calt feature
    calt_feature = otTables.FeatureRecord()
    calt_feature.FeatureTag = 'calt'
    calt_feature.Feature = otTables.Feature()
    calt_feature.Feature.FeatureParams = None
    calt_feature.Feature.LookupListIndex = list(range(base_index, base_index + len(lookups)))
    calt_feature.Feature.LookupCount = len(lookups)
    
    # Add feature to feature list
    if not hasattr(gsub, 'FeatureList') or gsub.FeatureList is None:
        gsub.FeatureList = otTables.FeatureList()
        gsub.FeatureList.FeatureRecord = []
    
    gsub.FeatureList.FeatureRecord.append(calt_feature)
    gsub.FeatureList.FeatureCount = len(gsub.FeatureList.FeatureRecord)
    
    # Add feature to all scripts/languages
    if hasattr(gsub, 'ScriptList') and gsub.ScriptList:
        for script_record in gsub.ScriptList.ScriptRecord:
            script = script_record.Script
            if script.DefaultLangSys:
                if script.DefaultLangSys.FeatureIndex is None:
                    script.DefaultLangSys.FeatureIndex = []
                script.DefaultLangSys.FeatureIndex.append(len(gsub.FeatureList.FeatureRecord) - 1)
                script.DefaultLangSys.FeatureCount = len(script.DefaultLangSys.FeatureIndex)
            if script.LangSysRecord:
                for lang_sys_record in script.LangSysRecord:
                    if lang_sys_record.LangSys.FeatureIndex is None:
                        lang_sys_record.LangSys.FeatureIndex = []
                    lang_sys_record.LangSys.FeatureIndex.append(len(gsub.FeatureList.FeatureRecord) - 1)
                    lang_sys_record.LangSys.FeatureCount = len(lang_sys_record.LangSys.FeatureIndex)


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
    
    # Ensure GSUB table exists
    if 'GSUB' not in font:
        font['GSUB'] = TTFont()['GSUB']
        font['GSUB'].table = otTables.GSUB()
        font['GSUB'].table.Version = 0x00010000
        
        # Create empty script list
        font['GSUB'].table.ScriptList = otTables.ScriptList()
        font['GSUB'].table.ScriptList.ScriptRecord = []
        
        # Add DFLT script
        dflt_script = otTables.ScriptRecord()
        dflt_script.ScriptTag = 'DFLT'
        dflt_script.Script = otTables.Script()
        dflt_script.Script.DefaultLangSys = otTables.DefaultLangSys()
        dflt_script.Script.DefaultLangSys.ReqFeatureIndex = 0xFFFF
        dflt_script.Script.DefaultLangSys.FeatureIndex = []
        dflt_script.Script.DefaultLangSys.FeatureCount = 0
        dflt_script.Script.LangSysRecord = []
        dflt_script.Script.LangSysCount = 0
        font['GSUB'].table.ScriptList.ScriptRecord.append(dflt_script)
        font['GSUB'].table.ScriptList.ScriptCount = 1
        
        # Create empty feature list
        font['GSUB'].table.FeatureList = otTables.FeatureList()
        font['GSUB'].table.FeatureList.FeatureRecord = []
        font['GSUB'].table.FeatureList.FeatureCount = 0
        
        # Create empty lookup list
        font['GSUB'].table.LookupList = otTables.LookupList()
        font['GSUB'].table.LookupList.Lookup = []
        font['GSUB'].table.LookupList.LookupCount = 0
    
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
