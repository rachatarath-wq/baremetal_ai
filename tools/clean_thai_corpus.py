#!/usr/bin/env python3
"""
clean_thai_corpus.py — strip Wikipedia navigation/reference boilerplate from a
raw Thai corpus produced by build_thai_corpus.py.

The raw corpus is a stream of blank-line-separated chunks (an article title sits
on the line before its first paragraph; later paragraphs are their own chunks).
prep_thai.py treats each chunk as one BOS..EOS training document, so this cleaner
works on chunks too and leaves the format unchanged.

Three kinds of non-language chrome are removed:

  1. Navigation/portal chunks — a chunk whose first non-empty line is a known
     nav title (หน้าหลัก, ป้ายบอกทาง, ศาลาประชาคม, ...) is dropped whole.
  2. Trailing reference sections — a standalone "อ้างอิง" / "ดูเพิ่ม" /
     "หมายเหตุ" / "เชิงอรรถ" / "รายการอ้างอิง" / "แหล่งข้อมูลอื่น" /
     "บรรณานุกรม" / "หมวดหมู่" / "แม่แบบ" header marks the start of a citation /
     link list, not prose.  The chunk is truncated at the first such header.
  3. Stray citation lines — bare URLs and "หมวดหมู่:..."/"แม่แบบ:..." lines.

Nothing here tries to be a full MediaWiki parser; it just removes the biggest
sources of boilerplate so the tokenizer sees mostly real Thai prose.

Usage:
    python clean_thai_corpus.py <input> <output>
"""
import re
import sys

# First non-empty line of a chunk == pure-navigation/portal document -> drop.
NAV_TITLES = {
    "หน้าหลัก", "ป้ายบอกทาง", "ศาลาประชาคม", "กระดานข่าว",
    "ปุจฉา-วิสัชนา", "แผนกช่วยเหลือ", "ข่าวไซต์", "สภากาแฟ",
    "สร้างบทความใหม่", "ภาษาอื่น", "Local Embassy",
    "การนำทาง", "สถานีย่อย",
}

# Standalone header lines that mark the start of a trailing reference /
# navigation block.  Matched after strip() + trimming wiki "=" / ":" marks.
REF_HEADERS = (
    "อ้างอิง",
    "รายการอ้างอิง",
    "เชิงอรรถ",
    "บรรณานุกรม",
    "แหล่งข้อมูลอื่น",
    "ดูเพิ่ม",
    "หมายเหตุ",
    "หนังสืออ่านเพิ่มเติม",
    "อ่านเพิ่ม",
    "หมวดหมู่",
    "แม่แบบ",
)

URL_RE = re.compile(r"(?:https?://|www\.)")
CATEGORY_RE = re.compile(r"^\s*(?:หมวดหมู่|แม่แบบ)\s*:")
REF_MARKER_RE = re.compile(r"^\s*\[\d+\]\s*$")  # bare "[12]" citation markers

MIN_DOC_CHARS = 20  # drop chunks shorter than this after cleaning


def is_ref_header(line: str) -> bool:
    s = line.strip().strip("=: \t​")
    return s in REF_HEADERS


def clean_chunk(chunk: str):
    """Return the cleaned chunk text, or None if the whole chunk should be dropped."""
    lines = [ln.rstrip() for ln in chunk.split("\n")]

    # Navigation chunk: first non-empty line is a nav title.
    first = next((ln.strip() for ln in lines if ln.strip()), "")
    if first in NAV_TITLES:
        return None

    kept = []
    for ln in lines:
        s = ln.strip()
        if is_ref_header(s):
            break  # drop this header and everything after it
        if URL_RE.search(ln) or CATEGORY_RE.match(s) or REF_MARKER_RE.match(s):
            continue
        kept.append(ln)

    body = "\n".join(kept).strip()
    if len(body) < MIN_DOC_CHARS:
        return None
    return body


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    src, dst = sys.argv[1], sys.argv[2]

    with open(src, encoding="utf-8") as f:
        raw = f.read()

    chunks = raw.replace("\r\n", "\n").replace("\r", "\n").split("\n\n")

    n_in = 0
    n_out = 0
    dropped_nav = 0
    dropped_stub = 0
    truncated = 0

    with open(dst, "w", encoding="utf-8") as out:
        for chunk in chunks:
            chunk = chunk.strip()
            if not chunk:
                continue
            n_in += 1

            body = clean_chunk(chunk)
            if body is None:
                first = next((ln.strip() for ln in chunk.split("\n") if ln.strip()), "")
                if first in NAV_TITLES:
                    dropped_nav += 1
                else:
                    dropped_stub += 1
                continue
            if body != chunk.strip():
                truncated += 1
            out.write(body + "\n\n")
            n_out += 1

    print(f"chunks in: {n_in:,}  ->  kept {n_out:,} "
          f"(dropped nav {dropped_nav:,}, stub/short {dropped_stub:,}, truncated {truncated:,})")


if __name__ == "__main__":
    main()
