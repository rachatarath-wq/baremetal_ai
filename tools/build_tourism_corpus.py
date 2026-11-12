#!/usr/bin/env python3
"""
build_tourism_corpus.py — stream Thai Wikipedia and keep only tourism-relevant
articles (filtered by title keyword), writing a raw UTF-8 corpus in the same
blank-line-separated format build_thai_corpus.py produces.

Why filter on a fresh stream instead of reusing thai_corpus_500.txt: the raw
corpus has no reliable article separator (paragraph breaks use the same "\\n\\n"
as article boundaries), so titles can't be recovered from it. Filtering the live
stream keeps each title + text together correctly.

Usage:
    python build_tourism_corpus.py --output thai_corpus_tourism.txt --limit-mb 120
"""
import argparse
import os

from datasets import load_dataset
from tqdm import tqdm

DATASET = "wikimedia/wikipedia"
CONFIG  = "20231101.th"

REDIRECT_PREFIXES = ("#REDIRECT", "#redirect", "#เปลี่ยนทาง")

# Title substrings that mark a tourism-relevant article. "จังหวัด" alone pulls in
# all 77 province articles (the backbone of Thai tourism content). The rest cover
# attraction types + a few major destinations whose titles lack "จังหวัด".
TOURISM_KEYWORDS = (
    "จังหวัด", "กรุงเทพมหานคร", "พัทยา", "หัวหิน", "สมุย", "พีพี", "เขาหลัก",
    "เกาะ", "อุทยาน", "น้ำตก", "ชายหาด", "หาด", "อ่าว", "ถ้ำ",
    "ดอย", "ภูเขา", "วัด", "พระราชวัง", "พิพิธภัณฑ์", "ตลาด",
    "ท่องเที่ยว", "โบราณสถาน", "อุทยานประวัติศาสตร์",
)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--output", default="thai_corpus_tourism.txt")
    ap.add_argument("--limit-mb", type=float, default=120.0)
    ap.add_argument("--min-text-len", type=int, default=20)
    args = ap.parse_args()

    target_bytes = int(args.limit_mb * 1024 * 1024)

    print(f"Loading {DATASET} config={CONFIG!r} (streaming, title filter) ...")
    ds = load_dataset(DATASET, CONFIG, split="train", streaming=True)

    written = 0
    n_kept = 0
    n_skipped = 0

    with open(args.output, "w", encoding="utf-8") as f, \
         tqdm(total=target_bytes, unit="B", unit_scale=True,
              desc=os.path.basename(args.output)) as pbar:

        for ex in ds:
            title = (ex.get("title") or "").strip()
            text  = (ex.get("text")  or "").strip()

            if (not text or len(text) < args.min_text_len
                    or text.startswith(REDIRECT_PREFIXES)):
                n_skipped += 1
                continue

            if not any(k in title for k in TOURISM_KEYWORDS):
                n_skipped += 1
                continue

            block = f"{title}\n{text}\n\n"
            block_bytes = len(block.encode("utf-8"))
            if written + block_bytes > target_bytes:
                break

            f.write(block)
            written += block_bytes
            n_kept += 1
            pbar.update(block_bytes)

    print(f"\nDone: kept {n_kept} tourism articles -> {written / 1024 / 1024:.2f} MB "
          f"({n_skipped} skipped) in {args.output}")


if __name__ == "__main__":
    main()
