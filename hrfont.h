/*
 * hrfont.h -- high-resolution Unicode (ASCII + Thai) framebuffer text renderer.
 *
 * This is the renderer half of the TTF -> C font upgrade.  It consumes the
 * glyph data produced by tools/ttf_to_c.py (see font.h) and draws one UTF-8
 * character at a time straight into the 32-bit GOP framebuffer, with correct
 * handling of Thai combining marks.
 *
 * The trick that makes Thai work is the *baseline*.  Every glyph in font.h is
 * rendered on the SAME baseline inside its fixed cell, so:
 *
 *   - a base consonant (advance != 0) is drawn at the cursor and the cursor is
 *     advanced by one cell;
 *   - a zero-width combining mark (advance == 0: sara i/ii/ue/uee/u/uu, phinthu,
 *     mai han-akat, and the tone marks) is drawn *transparently* one cell back
 *     -- over the base consonant -- WITHOUT advancing the cursor.  Because both
 *     glyphs share a baseline and a cell origin, the mark lands exactly on the
 *     consonant (upper marks above it, lower marks below it).
 *
 * Integration into run.c:
 *   1. Make the five framebuffer globals visible here (drop `static` on
 *      gFbBase / gFbWidth / gFbHeight / gFbScanLine / gBgColor in run.c).
 *   2. `#include "hrfont.h"` AFTER <efi.h>/<efilib.h> (it needs UINT32/UINT8
 *      and NULL, and pulls in font.h for the glyph data).
 *   3. Call draw_utf8_char() instead of the old 8x16 path.  `color` must already
 *      be in the framebuffer's native pixel format (use run.c's fb_make_color()).
 *
 * Cursor coordinates are PIXELS: (cursor_x, cursor_y) is the top-left of the
 * current glyph cell.  draw_utf8_char() advances the cursor, wraps to the next
 * line at the right edge, and consumes exactly one UTF-8 character from
 * *utf8_str.  It does NOT scroll; the caller may detect cursor_y overflow and
 * scroll the framebuffer as run.c's fb_scroll() does.
 */

#ifndef HRFONT_RENDERER_H
#define HRFONT_RENDERER_H

#include "font.h"   /* HRFONT_W/H/BASE, HRFONT_glyphs[], HRFONT_bitmap[] */

/* ----------------------------------------------------------------------------
 * Framebuffer state owned by run.c.  These are the only globals the renderer
 * touches; every pixel write goes through them so the whole thing stays
 * self-contained and portable across GOP pixel formats (the caller passes a
 * pre-formatted `color` word).
 * ------------------------------------------------------------------------- */
extern UINT8  *gFbBase;        /* framebuffer base (device memory) */
extern UINT32  gFbWidth;       /* visible width, px */
extern UINT32  gFbHeight;      /* visible height, px */
extern UINT32  gFbScanLine;    /* PixelsPerScanLine (>= width) */
extern UINT32  gBgColor;       /* background colour word */

/* Write one 32-bit pixel, clipped to the framebuffer.  `color` is already in
 * the framebuffer's native format. */
static inline void hrf_put_pixel(int x, int y, UINT32 color) {
    if (gFbBase == NULL) return;
    if (x < 0 || y < 0) return;
    if ((UINT32)x >= gFbWidth || (UINT32)y >= gFbHeight) return;
    UINT32 *p = (UINT32 *)(gFbBase + (UINTN)y * gFbScanLine * 4 + (UINTN)x * 4);
    *p = color;
}

/* Blit one full-cell 1-bit glyph bitmap (MSB = leftmost pixel, one byte per 8
 * pixels, HRFONT_ROW_BYTES bytes per row) at pixel (x, y).
 *
 *   transparent == 0 : cleared pixels are painted with `bg` (full overwrite,
 *                      used for spacing glyphs so backspace/redraw erase);
 *   transparent != 0 : cleared pixels are left untouched (used for zero-width
 *                      combining marks overlaid on the base consonant). */
static inline void hrf_blit(const UINT8 *bmp, int x, int y,
                            UINT32 fg, UINT32 bg, int transparent) {
    for (int row = 0; row < HRFONT_H; row++) {
        const UINT8 *r = bmp + (UINTN)row * HRFONT_ROW_BYTES;
        for (int col = 0; col < HRFONT_W; col++) {
            if (r[col >> 3] & (UINT8)(0x80 >> (col & 7)))
                hrf_put_pixel(x + col, y + row, fg);
            else if (!transparent)
                hrf_put_pixel(x + col, y + row, bg);
        }
    }
}

/* O(1) glyph lookup by direct index: ASCII 0x20..0x7E first, then the Thai block
 * 0x0E00..0x0E7F.  Returns NULL for anything else. */
static inline const HRFONT_glyph_t *hrf_lookup(UINT32 cp) {
    if (cp >= HRFONT_ASCII_FIRST && cp <= HRFONT_ASCII_LAST)
        return &HRFONT_glyphs[cp - HRFONT_ASCII_FIRST];
    if (cp >= HRFONT_THAI_BASE && cp <= HRFONT_THAI_END)
        return &HRFONT_glyphs[(HRFONT_ASCII_LAST - HRFONT_ASCII_FIRST + 1)
                              + (cp - HRFONT_THAI_BASE)];
    return NULL;
}

/* Decode exactly one UTF-8 code point from *s, advancing *s past its bytes.
 * Invalid sequences are replaced with U+FFFD and advanced by one byte. */
static inline UINT32 hrf_decode_utf8(const char **s) {
    const UINT8 *p = (const UINT8 *)*s;
    UINT8 b = *p;
    UINT32 cp;
    int need;

    if (b < 0x80) {                       /* ASCII */
        *s = (const char *)(p + 1);
        return b;
    }
    if ((b & 0xE0) == 0xC0) {             /* 110xxxxx : 2 bytes */
        cp = b & 0x1F; need = 1;
    } else if ((b & 0xF0) == 0xE0) {      /* 1110xxxx : 3 bytes (Thai) */
        cp = b & 0x0F; need = 2;
    } else if ((b & 0xF8) == 0xF0) {      /* 11110xxx : 4 bytes */
        cp = b & 0x07; need = 3;
    } else {                              /* invalid lead byte */
        *s = (const char *)(p + 1);
        return 0xFFFD;
    }
    for (int i = 0; i < need; i++) {
        b = p[1 + i];
        if ((b & 0xC0) != 0x80) {         /* truncated / invalid */
            *s = (const char *)(p + 1 + i);
            return 0xFFFD;
        }
        cp = (cp << 6) | (b & 0x3F);
    }
    *s = (const char *)(p + 1 + need);
    return cp;
}

/* Draw one UTF-8 character at the cursor and update the cursor.
 *
 *   cursor_x / cursor_y : in-out, pixel position of the current cell's top-left.
 *   color               : foreground colour word (framebuffer-native format).
 *   utf8_str            : in-out, points at the next UTF-8 byte; advanced past
 *                         the character just drawn.
 *
 * Behaviour:
 *   - '\n' -> next line, '\r' -> column 0, '\t' -> +4 cells, '\b' -> back one
 *     cell (control characters move the cursor and draw nothing);
 *   - spacing glyph -> opaque blit, then cursor_x += one cell (wrap at right);
 *   - Thai combining mark (adv == 0) -> transparent blit one cell back, cursor
 *     unchanged, so the mark stacks on the preceding consonant. */
void draw_utf8_char(int *cursor_x, int *cursor_y, UINT32 color,
                    const char **utf8_str) {
    UINT32 cp = hrf_decode_utf8(utf8_str);

    /* control characters drive the cursor, not a glyph */
    if (cp == '\n') { *cursor_x = 0; *cursor_y += HRFONT_H; return; }
    if (cp == '\r') { *cursor_x = 0; return; }
    if (cp == '\t') { *cursor_x += HRFONT_W * 4; return; }
    if (cp == '\b') { if (*cursor_x >= HRFONT_W) *cursor_x -= HRFONT_W; return; }

    const HRFONT_glyph_t *g = hrf_lookup(cp);
    if (g == NULL) g = hrf_lookup('?');   /* unsupported -> '?' */

    const UINT8 *bmp =
        HRFONT_bitmap + (UINTN)(g - HRFONT_glyphs) * HRFONT_GLYPH_BYTES;

    if (g->adv != 0) {
        /* spacing glyph: wrap to the next line if it would not fit */
        if ((UINT32)*cursor_x + (UINT32)g->adv > gFbWidth) {
            *cursor_x = 0;
            *cursor_y += HRFONT_H;
        }
        hrf_blit(bmp, *cursor_x, *cursor_y, color, gBgColor, 0);
        *cursor_x += g->adv;
    } else {
        /* zero-width combining mark: overlay the base consonant, which is one
         * cell back (g->adv == HRFONT_W for every spacing glyph), and leave the
         * cursor untouched so the next base glyph still lands after it. */
        int tx = *cursor_x - HRFONT_W;
        if (tx < 0) tx = 0;
        hrf_blit(bmp, tx, *cursor_y, color, 0, /*transparent=*/1);
    }
}

/* Convenience: draw a whole NUL-terminated UTF-8 string starting at the given
 * pixel cursor. */
static inline void hrf_puts(const char *s, int *cx, int *cy, UINT32 color) {
    while (*s) draw_utf8_char(cx, cy, color, &s);
}

#endif /* HRFONT_RENDERER_H */
