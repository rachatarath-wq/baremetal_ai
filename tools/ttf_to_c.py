#!/usr/bin/env python3
"""
ttf_to_c.py -- render a TrueType/OpenType font (ASCII + Thai) into a C header.

This is the font-*upgrade* half of the bare-metal Thai rendering solution: it
takes a real TTF/TTC (e.g. NotoSansThai.ttf, or the macOS-bundled Krungthep.ttf
/ Thonburi.ttc), renders the printable ASCII range (U+0020..U+007E) and the whole
Thai Unicode block (U+0E00..U+0E7F) at native resolution, and emits a
bare-metal-friendly C header (`font.h`) containing:

  * a packed 1-bit bitmap for every glyph (one byte per 8 pixels, MSB = leftmost
    pixel, matching the existing 8x16 font's convention), and
  * per-glyph metadata: advance width, ink bounding box, and the cell offsets
    that encode the glyph's position relative to the shared baseline.

The reason this matters for Thai: Thai is written with *combining marks* --
upper vowels (sara i/ii/ue/uee), lower vowels (sara u/uu, phinthu) and tone
marks -- that stack on top of / below the base consonant.  If every glyph is
rendered on the SAME baseline inside a fixed cell, then drawing a zero-width
combining mark transparently over the cell of the preceding consonant (without
advancing the cursor) makes the mark land exactly on the consonant.

How it works, concretely:

  1. Pick a cell of (cell_w x cell_h) pixels with a fixed baseline `baseline`
     pixels down from the cell top (default 16x32, baseline 24 -- the same 3:1
     above/below ratio the existing hand-drawn Thai font uses).

  2. Auto-fit the FreeType pixel size: shrink `size` from cell_h downward until
        ascent  <= baseline
        descent <= cell_h - baseline
        advance <= cell_w
     so nothing is clipped.  (A real Thai glyph is ~0.67x the em wide, so the
     width constraint is usually the binding one.  The widest uppercase Latin
     letters 'M'/'W' are ~1.0x the em; in a narrow cell they are allowed to
     clip at the edge rather than forcing the whole font tiny.)

  3. Draw every code point with Pillow's anchor "ls" (left + *baseline*), which
     makes FreeType position the glyph by its typographic baseline rather than
     its top.  All glyphs share that baseline, so marks and consonants align
     vertically for free.

  4. Horizontal placement: spacing glyphs start at a single centring shift so the
     monospace advance box is centred in the cell.  A combining mark is drawn
     with the pen one advance further right (as if a consonant preceded it), so
     the mark's negative left bearing reaches back and lands on the consonant.
     Because the shift is identical, marks and consonants align when overlaid.

  5. Pack each cell to 1 bit/pixel and dump `font.h`.

Usage:
    python3 ttf_to_c.py <font.ttf|font.ttc> [options]

Options:
    --index N      face index inside a .ttc collection        (default 0)
    --cell-w N     glyph cell width, pixels                   (default 16)
    --cell-h N     glyph cell height, pixels                  (default 32)
    --baseline N   baseline from the cell top, pixels         (default 3/4 * cell-h)
    --out PATH     output header path                         (default font.h)
    --name PREFIX  macro prefix                               (default HRFONT)
    --no-center    left-align glyphs at x=0 instead of centring

Example (crisper 24x48 font):
    python3 ttf_to_c.py NotoSansThai.ttf --cell-w 24 --cell-h 48 --baseline 36
"""

import argparse
import sys
import os
import datetime

from PIL import Image, ImageDraw, ImageFont

# --------------------------------------------------------------------------- #
# Code point ranges and Thai combining-mark classification.
#
# This mirrors run.c's thai_is_combining_mark() EXACTLY, so the generated font
# and the C renderer agree on which code points are zero-width overlays:
#   U+0E31 (mai han-akat), U+0E34..0E3A (sara i/ii/ue/uee/u/uu, phinthu),
#   U+0E47..0E4E (tone marks + thanthakhat + nikhahit + yamakkan).
# --------------------------------------------------------------------------- #
ASCII_FIRST = 0x20
ASCII_LAST = 0x7E
THAI_BASE = 0x0E00
THAI_END = 0x0E7F


def is_thai_combining(cp):
    if cp == 0x0E31:
        return True
    if 0x0E34 <= cp <= 0x0E3A:
        return True
    if 0x0E47 <= cp <= 0x0E4E:
        return True
    return False


# Every code point the header will contain, in array order.
def code_points():
    return list(range(ASCII_FIRST, ASCII_LAST + 1)) + \
           list(range(THAI_BASE, THAI_END + 1))


def index_of(cp):
    """Inverse of code_points(): array index for a code point, or -1."""
    if ASCII_FIRST <= cp <= ASCII_LAST:
        return cp - ASCII_FIRST
    if THAI_BASE <= cp <= THAI_END:
        return (ASCII_LAST - ASCII_FIRST + 1) + (cp - THAI_BASE)
    return -1


# --------------------------------------------------------------------------- #
# Font size auto-fit.
#
# Pillow's ImageFont.truetype(path, size) sizes the em square to `size` px.
# getmetrics() returns (ascent, descent): the font-wide extents measured from
# the baseline.  A glyph is fully inside the cell iff
#       ascent  <= baseline            (top of the tallest ascender is >= cell top)
#       descent <= cell_h - baseline   (bottom of the deepest descender is <= cell bottom)
# and its advance fits the cell width.
# --------------------------------------------------------------------------- #
def fit_font(path, index, cell_w, cell_h, baseline):
    size = cell_h
    last = None
    while size >= 6:
        try:
            f = ImageFont.truetype(path, size, index=index)
        except Exception:
            size -= 1
            continue
        asc, desc = f.getmetrics()
        # Use the Thai consonant advance as the representative text advance.
        # Thai consonants, digits and lowercase Latin all share this width in
        # the fonts we target.  The only outliers are the wide uppercase letters
        # ('M'/'W' are ~1.0x the em), which are left to clip gracefully at the
        # cell edge rather than shrinking the whole font down to fit them.
        adv = f.getlength("ก")   # U+0E01 KO KAI
        if asc <= baseline and desc <= cell_h - baseline and adv <= cell_w:
            return f, size, asc, desc, adv
        last = (f, size, asc, desc, adv)
        size -= 1

    # Could not satisfy every constraint even at the smallest sane size: keep the
    # last attempt and let the caller clip (a warning is printed below).
    return last if last is not None else \
        (ImageFont.truetype(path, 6, index=index), 6, 0, 0, 0)


# --------------------------------------------------------------------------- #
# Rendering one glyph into its cell.
# --------------------------------------------------------------------------- #
def render_glyph(font, ch, cell_w, cell_h, baseline, pen_x):
    """Return (bitmap_bytes, adv, w, h, x0, y0)."""
    img = Image.new("L", (cell_w, cell_h), 0)  # black background
    d = ImageDraw.Draw(img)
    # anchor "ls" = left + baseline: FreeType places the glyph by its baseline.
    # `pen_x` is the horizontal pen position.  Combining marks are rendered with
    # the pen one advance to the right (as if a consonant preceded them), so
    # their negative left bearing reaches back onto the consonant.
    d.text((pen_x, baseline), ch, font=font, fill=255, anchor="ls")

    # Ink bounding box (None when the glyph is blank, e.g. a missing codepoint).
    bbox = img.getbbox()
    if bbox is None:
        w = h = x0 = y0 = 0
    else:
        x0, y0, x1, y1 = bbox
        w = x1 - x0
        h = y1 - y0

    # advance: zero for combining marks, otherwise the fixed cell width.  This is
    # a monospaced grid, so every spacing glyph advances exactly one full cell;
    # the font's natural (fractional) advance is instead absorbed by the centring.
    adv = 0 if is_thai_combining(ord(ch)) else cell_w

    # Pack to 1 bit/pixel, row-major, MSB = leftmost pixel.
    row_bytes = (cell_w + 7) // 8
    buf = bytearray(row_bytes * cell_h)
    px = img.load()
    for y in range(cell_h):
        base = y * row_bytes
        for x in range(cell_w):
            if px[x, y] >= 128:
                buf[base + (x >> 3)] |= (0x80 >> (x & 7))
    return buf, adv, w, h, x0, y0


# --------------------------------------------------------------------------- #
# Header emission.
# --------------------------------------------------------------------------- #
def emit_header(out, name, path, index, cell_w, cell_h, baseline, size,
                glyphs, blanks, stdint=False):
    # Integer types for the struct fields.  gnu-efi has no <stdint.h> in its
    # freestanding include path, so default to its UINT16/UINT8/INT8; pass
    # --stdint to emit stdint.h types for a hosted build instead.
    if stdint:
        t_cp, t_adv, t_w, t_h, t_x, t_y = \
            "uint16_t", "uint8_t", "uint8_t", "uint8_t", "int8_t", "int8_t"
    else:
        t_cp, t_adv, t_w, t_h, t_x, t_y = \
            "UINT16", "UINT8", "UINT8", "UINT8", "INT8", "INT8"

    row_bytes = (cell_w + 7) // 8
    glyph_bytes = row_bytes * cell_h
    count = len(glyphs)

    lines = []
    lines.append("/*")
    lines.append(" * font.h -- auto-generated by tools/ttf_to_c.py; DO NOT EDIT.")
    lines.append(" *")
    lines.append(" *   source font : %s (face index %d)" % (path, index))
    lines.append(" *   generated   : %s" % datetime.datetime.now().isoformat(" ", "seconds"))
    lines.append(" *   cell        : %d x %d px, baseline %d px from top" %
                 (cell_w, cell_h, baseline))
    lines.append(" *   FreeType px : %d (auto-fitted to the cell)" % size)
    lines.append(" *   glyphs      : %d (ASCII 0x%02X..0x%02X + Thai 0x%04X..0x%04X)"
                 % (count, ASCII_FIRST, ASCII_LAST, THAI_BASE, THAI_END))
    if blanks:
        lines.append(" *   blank glyphs: %d -> %s" %
                     (len(blanks), " ".join("U+%04X" % cp for cp in blanks)))
    lines.append(" *")
    lines.append(" * Layout: every glyph is drawn on the SAME baseline inside its cell, so a")
    lines.append(" * zero-width Thai combining mark (adv == 0) drawn transparently over the")
    lines.append(" * preceding consonant's cell stacks on it correctly.")
    lines.append(" */")
    lines.append("")
    lines.append("#ifndef %s_DATA_H" % name)
    lines.append("#define %s_DATA_H" % name)
    lines.append("")
    lines.append("#define %s_W            %d   /* glyph cell width, px */" % (name, cell_w))
    lines.append("#define %s_H            %d   /* glyph cell height, px */" % (name, cell_h))
    lines.append("#define %s_BASE         %d   /* shared baseline, px from cell top */" % (name, baseline))
    lines.append("#define %s_ROW_BYTES    %d   /* ((W+7)/8) bytes per bitmap row */" % (name, row_bytes))
    lines.append("#define %s_GLYPH_BYTES  %d   /* ROW_BYTES * H */" % (name, glyph_bytes))
    lines.append("#define %s_COUNT        %d" % (name, count))
    lines.append("#define %s_ASCII_FIRST  0x%02X" % (name, ASCII_FIRST))
    lines.append("#define %s_ASCII_LAST   0x%02X" % (name, ASCII_LAST))
    lines.append("#define %s_THAI_BASE    0x%04X" % (name, THAI_BASE))
    lines.append("#define %s_THAI_END     0x%04X" % (name, THAI_END))
    lines.append("")
    lines.append("typedef struct {")
    lines.append("    %s cp;    /* Unicode code point */" % t_cp)
    lines.append("    %s adv;   /* horizontal advance, px (0 = zero-width combining) */" % t_adv)
    lines.append("    %s w;     /* ink bounding-box width, px */" % t_w)
    lines.append("    %s h;     /* ink bounding-box height, px */" % t_h)
    lines.append("    %s x0;    /* ink bbox left edge within the cell, px (signed) */" % t_x)
    lines.append("    %s y0;    /* ink bbox top edge within the cell, px (signed) */" % t_y)
    lines.append("} %s_glyph_t;" % name)
    lines.append("")
    lines.append("/* metadata, ordered ASCII(0x%02X..0x%02X) then Thai(0x%04X..0x%04X);"
                 % (ASCII_FIRST, ASCII_LAST, THAI_BASE, THAI_END))
    lines.append(" * direct index: ascii -> cp-0x%02X, thai -> %d+(cp-0x%04X)."
                 % (ASCII_FIRST, ASCII_LAST - ASCII_FIRST + 1, THAI_BASE))
    lines.append(" */")
    lines.append("static const %s_glyph_t %s_glyphs[%s_COUNT] = {" % (name, name, name))
    for g in glyphs:
        cp, adv, w, h, x0, y0, _bmp = g
        ch = chr(cp) if cp >= 0x20 else "?"
        lines.append("    { 0x%04X, %3d, %3d, %3d, %4d, %4d }, /* U+%04X %s */"
                     % (cp, adv, w, h, x0, y0, cp, ch))
    lines.append("};")
    lines.append("")
    lines.append("/* packed 1-bit bitmaps, MSB = leftmost pixel, %s_GLYPH_BYTES per glyph */"
                 % name)
    lines.append("static const uint8_t %s_bitmap[%s_COUNT * %s_GLYPH_BYTES] = {"
                 % (name, name, name))
    for gi, g in enumerate(glyphs):
        bmp = g[6]
        lines.append("    /* glyph %d: U+%04X %s */" % (gi, g[0], chr(g[0]) if g[0] >= 0x20 else "?"))
        for i in range(0, len(bmp), 12):
            chunk = bmp[i:i + 12]
            lines.append("    " + ", ".join("0x%02X" % b for b in chunk) + ",")
    lines.append("};")
    lines.append("")
    lines.append("#endif /* %s_DATA_H */" % name)
    lines.append("")

    with open(out, "w") as f:
        f.write("\n".join(lines))


# --------------------------------------------------------------------------- #
def main(argv):
    ap = argparse.ArgumentParser(description="Render a TTF/TTC into a C header.")
    ap.add_argument("font", help="path to a .ttf or .ttc file")
    ap.add_argument("--index", type=int, default=0, help="face index in a .ttc")
    ap.add_argument("--cell-w", type=int, default=16, help="cell width px")
    ap.add_argument("--cell-h", type=int, default=32, help="cell height px")
    ap.add_argument("--baseline", type=int, default=None,
                    help="baseline from cell top (default 3/4 * cell-h)")
    ap.add_argument("--out", default="font.h", help="output header path")
    ap.add_argument("--name", default="HRFONT", help="macro prefix")
    ap.add_argument("--no-center", action="store_true",
                    help="left-align glyphs at x=0 instead of centring")
    ap.add_argument("--stdint", action="store_true",
                    help="emit stdint.h types (uint16_t/...) instead of gnu-efi types")
    args = ap.parse_args(argv)

    if not os.path.isfile(args.font):
        sys.exit("error: font not found: %s" % args.font)

    cell_w, cell_h = args.cell_w, args.cell_h
    baseline = args.baseline if args.baseline is not None else (3 * cell_h) // 4

    font, size, asc, desc, max_adv = fit_font(
        args.font, args.index, cell_w, cell_h, baseline)
    if asc > baseline or desc > cell_h - baseline or max_adv > cell_w:
        print("warning: font does not fit cell %dx%d baseline %d "
              "(ascent=%d descent=%d advance=%.1f); glyphs may be clipped"
              % (cell_w, cell_h, baseline, asc, desc, max_adv), file=sys.stderr)

    # Horizontal placement.  `mono_adv` is the Thai consonant advance (the grid
    # pitch for this monospaced layout); spacing glyphs are centred by it, and a
    # single shared shift preserves mark<->consonant alignment.
    mono_adv = int(round(font.getlength("ก")))   # U+0E01 KO KAI
    if args.no_center:
        x_shift = 0
    else:
        x_shift = max(0, (cell_w - mono_adv) // 2)

    glyphs = []
    blanks = []
    for cp in code_points():
        ch = chr(cp)
        # Combining marks are drawn with the pen advanced past a base consonant
        # (pen = x_shift + mono_adv) so their negative left bearing reaches back
        # over it; spacing glyphs start at x_shift.  Both share the same cell
        # origin at overlay time, so marks land on the consonant.
        pen_x = x_shift + (mono_adv if is_thai_combining(cp) else 0)
        buf, adv, w, h, x0, y0 = render_glyph(
            font, ch, cell_w, cell_h, baseline, pen_x)
        glyphs.append((cp, adv, w, h, x0, y0, bytes(buf)))
        if w == 0 and h == 0 and adv != 0:
            blanks.append(cp)   # spacing glyph with no ink (missing codepoint)

    emit_header(args.out, args.name, args.font, args.index, cell_w, cell_h,
                baseline, size, glyphs, blanks, args.stdint)

    print("wrote %s: %d glyphs, %d x %d px cells, baseline %d, FreeType size %d%s"
          % (args.out, len(glyphs), cell_w, cell_h, baseline, size,
             (" (%d blank glyphs)" % len(blanks)) if blanks else ""))


if __name__ == "__main__":
    main(sys.argv[1:])
