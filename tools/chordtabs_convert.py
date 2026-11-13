#!/usr/bin/env python3
"""
chordtabs_convert.py -- แปลงเพลงจาก chordtabs.in.th (หรือ text คอร์ดแบบไทยทั่วไป)
เป็น format ของสมุดเพลงในโปรเจกต์นี้ (@title / @artist / @album / บรรทัด >คอร์ด / @end)

--- โหมด 1: chordtabs.in.th (URL หรือไฟล์ .html ที่ save มา) ---

    python3 tools/chordtabs_convert.py https://chordtabs.in.th/73028/ -o songs/0005.txt
    python3 tools/chordtabs_convert.py saved.html -o songs/0005.txt

    chordtabs.in.th แสดงคอร์ดเป็น *รูป* (chord diagram) ไม่ใช่ตัวอักษร ตัว script จึง
    ดึงสิ่งที่อยู่ในหน้าได้จริง: ชื่อเพลง, ศิลปิน, รายการคอร์ดที่ใช้ (เรียงตามที่ขึ้นในหน้า),
    และเนื้อเพลง (ลอกเครื่องหมาย * กับ ** ที่ใช้แบ่ง verse/chorus ออกให้)
    คอร์ดไม่สามารถจัดตำแหน่งให้อัตโนมัติได้ (มันเป็นรูป) ผลลัพธ์จึงเป็น "เนื้อเพลง + บรรทัด
    # คอร์ดที่ใช้:" แล้วคุณค่อยเอาไปวางคอร์ดด้วย tools/align_chords.py

--- โหมด 2: text คอร์ดแบบ inline [คอร์ด] (ที่หลายเว็บ/ที่คนก็อปกันมาใช้) ---

    python3 tools/chordtabs_convert.py --inline song.txt --title "ชื่อเพลง" --artist "ศิลปิน"

    บรรทัดแบบ  [C]ฉันรักเธอ [G]เสมอมา   จะถูกแปลงเป็น
        >C            G
        ฉันรักเธอ เสมอมา
    หัวข้อท่อน (Intro: / Verse: / Chorus: / ท่อนฮุค: ...) กลายเป็น # comment
    บรรทัดที่เป็นคอร์ดล้วน (เช่น  "Intro: C G Am F") กลายเป็นบรรทัด > คอร์ด

ผลลัพธ์ทุกโหมด print ออก stdout (หรือเขียน -o FILE) ใน format นี้:
    @title ...
    @artist ...
    @album ไม่ระบุ
    >chord line
    lyric line
    @end
"""
import sys
import re
import html
import argparse
import urllib.request

COMBINING = None


def is_combining(cp):
    # ต้องตรงกับ run.c  thai_is_combining_mark() เป๊ะ:
    # ไม้หันอากาศ (ั U+0E31), สระบน/ล่าง (U+0E34..0E3A), วรรณยุกต์ (U+0E47..0E4E)
    return cp == 0x0E31 or 0x0E34 <= cp <= 0x0E3A or 0x0E47 <= cp <= 0x0E4E


def cells(text):
    """ความกว้างเป็นช่อง ๆ (combining = 0) ตรงกับ align_chords.py และ run.c"""
    return sum(0 if is_combining(ord(c)) else 1 for c in text)


# ---------------------------------------------------------------------------
# โหมด 1: chordtabs.in.th
# ---------------------------------------------------------------------------

def fetch(url):
    req = urllib.request.Request(
        url, headers={"User-Agent": "Mozilla/5.0 (chordtabs-converter)"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read().decode("utf-8", errors="replace")


def parse_page_title(t):
    """'ชื่อคอร์ด | คอร์ด ชื่อ ศิลปิน' -> (ชื่อ, ศิลปิน)"""
    t = t.strip()
    title, artist = "", ""
    if "|" in t:
        left, right = t.split("|", 1)
        title = re.sub(r"คอร์ด\s*$", "", left).strip()
        right = re.sub(r"^คอร์ด\s*", "", right.strip()).strip()
        if title and right.startswith(title):
            artist = right[len(title):].strip()
        else:
            artist = right
    else:
        title = re.sub(r"คอร์ด\s*$", "", t).strip()
    return title, artist


def extract_title_artist(page):
    artist = title = ""
    m = re.search(r'class=["\']chordheader["\'][^>]*>\s*คอร์ดเพลง\s*([^<]+)',
                  page, re.I)
    if m:
        artist = html.unescape(m.group(1)).strip()
    m = re.search(r"<a\b[^>]*>\s*คอร์ดเพลง\s*([^<]+)</a>", page, re.I)
    if m:
        title = html.unescape(m.group(1)).strip()
    if not title or not artist:
        t = re.search(r"<title[^>]*>(.*?)</title>", page, re.S | re.I)
        if t:
            t2, a2 = parse_page_title(html.unescape(t.group(1)))
            title = title or t2
            artist = artist or a2
    if not title:
        m = re.search(r"<h3[^>]*>\s*เนื้อเพลง\s*([^<]+)", page, re.I)
        if m:
            title = html.unescape(m.group(1)).strip()
    return title, artist


def extract_chords(page):
    """รายการคอร์ดที่ใช้ เรียงตามที่ปรากฏในหน้า (จาก alt='chord-X')"""
    names = []
    for m in re.finditer(r'alt=["\']chord-([^"\']+)["\']', page, re.I):
        n = m.group(1).strip()
        if n and n not in names:
            names.append(n)
    return names


def extract_lyrics(page):
    # หน้า chordtabs มี div id="divlyric" สองตัว (ตัวแรกเป็นรูปโน้ต) ให้เลือกตัวที่
    # มีคำว่า "เนื้อเพลง" (ตัวจริง) เท่านั้น
    m = None
    for mm in re.finditer(r'id=["\']divlyric["\'][^>]*>(.*?)</div>',
                          page, re.S | re.I):
        if "เนื้อเพลง" in mm.group(1):
            m = mm
            break
    if not m:
        # fallback: เส้นที่ขึ้นต้นด้วย * (ใช้กับหน้าเก่า / text ที่แปะมา)
        return _lyrics_from_text(strip_tags(page))
    block = m.group(1)
    block = re.sub(r"<h3[^>]*>.*?</h3>", "", block, flags=re.S | re.I)
    lyrics = []
    for part in re.split(r"<br\s*/?>", block, flags=re.I):
        txt = re.sub(r"<[^>]+>", "", part)
        txt = html.unescape(txt).replace("\xa0", " ").strip()
        if not txt:
            continue
        if re.match(r"\(\s*\*", txt):      # ( * , ** ) เครื่องหมาย repeat
            continue
        txt = re.sub(r"^\*{1,2}\s*", "", txt).strip()
        if txt:
            lyrics.append(txt)
    return lyrics


def _lyrics_from_text(text):
    lyrics = []
    for raw in text.splitlines():
        t = raw.strip()
        if not t or re.match(r"\(\s*\*", t):
            continue
        t = re.sub(r"^\*{1,2}\s*", "", t).strip()
        if t:
            lyrics.append(t)
    return lyrics


def strip_tags(page):
    t = re.sub(r"<script.*?</script>", " ", page, flags=re.S | re.I)
    t = re.sub(r"<style.*?</style>", " ", t, flags=re.S | re.I)
    t = re.sub(r"<[^>]+>", "\n", t)
    return html.unescape(t)


# ---------------------------------------------------------------------------
# โหมด 2: inline [คอร์ด]
# ---------------------------------------------------------------------------

CHORDISH = re.compile(
    r"^[A-Ga-g](?:#|b)?(?:m|maj|min|dim|aug|sus|add)?[0-9]*(?:/[A-Ga-g](?:#|b)?)?$")


def looks_like_chord(tok):
    return bool(CHORDISH.match(tok.strip(".,:;()")))


def convert_inline_line(line):
    """'[C]ฉันรัก [G]เธอ' -> (chord_line, lyric_line)"""
    lyric_chars = []
    chords = []          # (col, name)
    col = 0
    i = 0
    n = len(line)
    while i < n:
        c = line[i]
        if c == "[":
            j = line.find("]", i + 1)
            if j > i:
                name = line[i + 1:j].strip()
                if name:
                    chords.append((col, name))
                i = j + 1
                continue
        lyric_chars.append(c)
        if not is_combining(ord(c)):
            col += 1
        i += 1

    lyric = "".join(lyric_chars).strip()

    chord_line = ""
    pos = 0
    for c, name in chords:
        if c < pos:
            continue               # คอร์ดซ้อนกัน ข้ามตัวหลังไป
        chord_line += " " * (c - pos) + name
        pos = c + len(name)
    return chord_line.rstrip(), lyric


def looks_like_section_label(line):
    return bool(re.match(
        r"^(intro|verse|chorus|bridge|pre-?chorus|hook|outro|solo|interlude|"
        r"instrumental|ท่อน|อินโทร|โซโล|เอาท์โทร|ดนตรี)\b",
        line.strip(), re.I))


def convert_inline(text, title, artist):
    out = ["@title " + (title or "ไม่ระบุ"),
           "@artist " + (artist or "ไม่ระบุ"),
           "@album ไม่ระบุ"]
    for raw in text.splitlines():
        line = raw.rstrip("\r\n")
        s = line.strip()
        if not s:
            continue

        # บรรทัดคอร์ดล้วน เช่น  "C G Am F"  หรือ  "Intro: C G Am F"
        label = ""
        body = line
        if ":" in line:
            head, tail = line.split(":", 1)
            if looks_like_section_label(head):
                label = head.strip()
                body = tail
        toks = body.split()
        if toks and all(looks_like_chord(t) for t in toks):
            if label:
                out.append("# " + label)
            out.append(">" + " ".join(toks))
            continue

        # หัวข้อท่อนล้วน  เช่น  "Chorus:"
        if looks_like_section_label(s) and ("[" not in line):
            out.append("# " + s.rstrip(":。.:"))
            continue

        chord_line, lyric = convert_inline_line(line)
        if chord_line:
            out.append(">" + chord_line)
        if lyric:
            out.append(lyric)

    out.append("@end")
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------------------

def main(argv):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", nargs="?", help="URL / .html file (โหมด chordtabs) หรือ - เพื่ออ่าน stdin")
    ap.add_argument("--inline", action="store_true",
                    help="โหมด inline [คอร์ด] (อ่านจาก text file หรือ stdin)")
    ap.add_argument("--title", help="ชื่อเพลง (สำหรับ --inline)")
    ap.add_argument("--artist", help="ศิลปิน (สำหรับ --inline)")
    ap.add_argument("-o", "--out", help="เขียนลงไฟล์แทน stdout")
    args = ap.parse_args(argv)

    if args.inline:
        text = sys.stdin.read() if args.source in (None, "-") else open(
            args.source, encoding="utf-8").read()
        result = convert_inline(text, args.title, args.artist)
    else:
        src = args.source
        if not src:
            ap.error("ต้องระบุ URL/ไฟล์ .html หรือใช้ --inline")
        if src.startswith(("http://", "https://")):
            page = fetch(src)
        else:
            page = open(src, encoding="utf-8", errors="replace").read()

        title, artist = extract_title_artist(page)
        chords = extract_chords(page)
        lyrics = extract_lyrics(page)

        out = ["@title " + (title or "ไม่ระบุ"),
               "@artist " + (artist or "ไม่ระบุ"),
               "@album ไม่ระบุ"]
        if chords:
            out.append("# คอร์ดที่ใช้: " + " ".join(chords))
            out.append("# (วางคอร์ดลงบนเนื้อด้วย tools/align_chords.py)")
        out.extend(lyrics)
        out.append("@end")
        result = "\n".join(out) + "\n"

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(result)
        print("wrote", args.out, file=sys.stderr)
    else:
        sys.stdout.write(result)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
