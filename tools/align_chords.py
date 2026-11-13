#!/usr/bin/env python3
"""
align_chords.py -- put chord names exactly above the right syllable in a lyric.

The songbook renders every character on a fixed monospace grid: one cell per
*visible* character.  Thai combining marks -- tone marks (่ ้ ๊ ๋), upper/lower
vowels (ิ ี ึ ื ุ ู), ั, ็ -- are ZERO-width overlays, so they do NOT advance the
cursor.  Counting columns by hand is error-prone; this script does it for you,
mirroring run.c's cell-width rule (combining = 0, everything else = 1).

Usage:
    python3 tools/align_chords.py "<lyric>" "Chord@anchor" ["Chord@anchor" ...]

    Each token is "Chord@anchor": the chord name and the lyric substring it
    should sit above (usually the word the chord falls on).  The chord is placed
    at the column where that substring first appears.

Example:
    python3 tools/align_chords.py "หากว่าเธอผ่านมาได้ยินเพลงนี้" "Am@หาก" "G@ได้" "F@เพลง"
    >Am           G   F
"""
import sys


def is_combining(cp):
    # must match run.c's thai_is_combining_mark() exactly
    return cp == 0x0E31 or 0x0E34 <= cp <= 0x0E3A or 0x0E47 <= cp <= 0x0E4E


def cells(text):
    """Visible columns `text` occupies (combining marks count as 0)."""
    return sum(0 if is_combining(ord(c)) else 1 for c in text)


def main(argv):
    if len(argv) < 3:
        print(__doc__)
        return 2

    lyric = argv[1]
    chords = []
    for tok in argv[2:]:
        if "@" in tok:
            chord, anchor = tok.split("@", 1)
        else:
            chord, anchor = tok, ""
        chords.append((chord, anchor))

    # column -> chord (leftmost chord wins if two land on the same column)
    col_chord = {}
    for chord, anchor in chords:
        if anchor:
            idx = lyric.find(anchor)
            if idx < 0:
                print("anchor not found: %r" % anchor, file=sys.stderr)
                return 1
            col = cells(lyric[:idx])
        else:
            col = 0
        if col not in col_chord:
            col_chord[col] = chord

    out = []
    pos = 0  # current column, after the last emitted chord
    for col in sorted(col_chord):
        if col < pos:
            print("warning: chord %r overlaps a previous chord (col %d)"
                  % (col_chord[col], col), file=sys.stderr)
            continue
        out.append(" " * (col - pos))
        out.append(col_chord[col])
        pos = col + len(col_chord[col])

    print(">" + "".join(out))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
