#!/usr/bin/env python3
"""
expand_facts.py — rule-based Thai fact extraction from the clean corpus
(AUTO_IMPROVE_PLAN.md, แผน B).

Thai has no spaces between words, so a full NER/template pipeline needs a word
segmenter (pythainlp).  This extractor deliberately goes *high-precision,
low-recall* instead: it only trusts the two numeric anchors that survive cleanly
in unsegmented Thai — population ("X มีประชากร N คน") and area
("X มีพื้นที่ N ตารางกิโลเมตร").  The subject is the Thai run immediately before
the anchor, required to start at a space/punctuation boundary (Thai Wikipedia
leaves spaces around clause boundaries, so a clean subject is often
space-delimited).  A noise filter rejects subjects that swallowed a leading
function word, and everything is deduplicated.

Other patterns were tried and rejected as too noisy without segmentation:
"เป็นเมืองหลวงของ" (mostly "เคย/กลายเป็นเมืองหลวง…"), "สูง … เมตร" (matches
"ความสูง" of anything), "ยาว … กิโลเมตร", "มี … จังหวัด" (subject rarely
space-delimited).  The output is still meant to be *eyeballed once*.

Each emitted fact uses the same format build_qa_corpus.py already parses:

    คำถาม: <q1>
    คำถาม: <q2>
    คำตอบ: <answer>
    # key: <value>

Usage:
    python tools/expand_facts.py --input thai_corpus_clean.txt --output qa_thai_auto.txt
                                 [--max-facts 150]
"""
import argparse
import re

THAI = "฀-๿"

# A subject that *starts* with these function words has swallowed the front of
# the sentence — reject it.
NOISY_LEAD = (
    "ใน", "ซึ่ง", "และ", "ของ", "ที่", "ว่า", "กับ", "โดย", "แต่", "เมื่อ",
    "เป็น", "ส่วน", "จาก", "รวม", "ทั้ง", "มี", "หรือ", "ปัจจุบัน", "ขณะ",
    "ตั้ง", "ทั่ว", "รอง", "ระบุ", "หาก", "ได้", "ตัว", "จะ", "นับ", "เมือง",
)
NOISY_EXACT = {"นอกจากนี้", "โดยเฉพาะ", "ปัจจุบัน", "สำหรับ", "โดยทั่วไป", "ซึ่ง", "อย่างไรก็ตาม",
               "ตัวนคร", "เมืองนี้", "จะ", "ได้แก่"}
NOISY_TAIL = ("ซึ่ง", "และ", "ของ", "ที่", "หรือ", "ว่า", "โดย", "แต่", "กับ")

# subject = a 2..30 char Thai run that starts at a non-Thai boundary
SUBJ = rf"(?<![฀-๿])([{THAI}]{{2,30}})"

# Each pattern: (label, compiled regex, question templates, answer template, key template).
PATTERNS = [
    (
        "population",
        re.compile(rf"{SUBJ}มีประชากร\s*([\d,.]+)\s*คน"),
        ["{subj}มีประชากรกี่คน", "{subj}มีประชากรเท่าไร"],
        "{subj}มีประชากร {num} คน",
        "{num}",
    ),
    (
        "area",
        re.compile(rf"{SUBJ}มีพื้นที่\s*([\d,.]+)\s*ตารางกิโลเมตร"),
        ["{subj}มีพื้นที่เท่าไร", "{subj}มีพื้นที่ทั้งหมดเท่าไร"],
        "{subj}มีพื้นที่ {num} ตารางกิโลเมตร",
        "{num}",
    ),
]


def clean_subject(s):
    s = s.strip()
    if not s or len(s) < 2 or len(s) > 24:
        return None
    if s in NOISY_EXACT:
        return None
    for w in NOISY_LEAD:
        if s.startswith(w):
            return None
    for w in NOISY_TAIL:
        if s.endswith(w):
            return None
    return s


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default="thai_corpus_clean.txt")
    ap.add_argument("--output", default="qa_thai_auto.txt")
    ap.add_argument("--max-facts", type=int, default=150)
    args = ap.parse_args()

    with open(args.input, encoding="utf-8") as f:
        text = f.read()

    facts = {}   # normalized answer -> (questions, answer, key)
    for label, rx, qtemplates, atemplate, ktemplate in PATTERNS:
        for m in rx.finditer(text):
            subj, num = clean_subject(m.group(1)), m.group(2)
            if not subj:
                continue
            ctx = {"subj": subj, "num": num}
            answer = atemplate.format(**ctx)
            key = ktemplate.format(**ctx)
            questions = [q.format(**ctx) for q in qtemplates]
            norm = re.sub(r"\s+", "", answer)
            facts.setdefault(norm, (questions, answer, key))

    items = list(facts.values())[: args.max_facts]
    with open(args.output, "w", encoding="utf-8") as out:
        out.write("# ข้อเท็จจริงที่สกัดอัตโนมัติจาก corpus (Plan B) — ตรวจทานด้วยมือก่อนเทรน\n\n")
        for questions, answer, key in items:
            for q in questions:
                out.write(f"คำถาม: {q}\n")
            out.write(f"คำตอบ: {answer}\n")
            out.write(f"# key: {key}\n\n")

    print(f"extracted {len(items)} unique facts -> {args.output}")


if __name__ == "__main__":
    main()
