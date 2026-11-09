#!/usr/bin/env python3
"""
build_thai_corpus.py — download Thai Wikipedia and write a raw UTF-8 corpus
capped at ~100 MB.

Output format (blank line between documents, which matches prep_thai.py's
--split_on blank):

    <title>
    <article text>
                        <- blank line
    <title>
    <article text>
                        <- blank line
    ...

Usage:
    python build_thai_corpus.py --output thai_corpus.txt --limit-mb 100

Requires (both ship native arm64 wheels, so no extra setup on Apple Silicon):
    pip install datasets tqdm
"""
import argparse
import os

from datasets import load_dataset
from tqdm import tqdm

DATASET = "wikimedia/wikipedia"
CONFIG  = "20231101.th"          # verified: the only Thai config on the Hub

# Thai Wikipedia redirects start with these; they carry no real text.
REDIRECT_PREFIXES = ("#REDIRECT", "#redirect", "#เปลี่ยนทาง")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--output", default="thai_corpus.txt")
    ap.add_argument("--config", default=CONFIG)
    ap.add_argument("--limit-mb", type=float, default=100.0,
                    help="stop writing once the output reaches ~this size")
    ap.add_argument("--min-text-len", type=int, default=20,
                    help="skip articles whose body is shorter than this")
    args = ap.parse_args()

    target_bytes = int(args.limit_mb * 1024 * 1024)

    # streaming=True fetches shards lazily, so we can stop at the size cap
    # without downloading the whole (~GB-scale) Thai Wikipedia dump first.
    print(f"Loading {DATASET} config={args.config!r} (streaming) ...")
    ds = load_dataset(DATASET, args.config, split="train", streaming=True)

    written = 0          # bytes written == current UTF-8 file size
    n_articles = 0
    n_skipped = 0

    with open(args.output, "w", encoding="utf-8") as f, \
         tqdm(total=target_bytes, unit="B", unit_scale=True,
              desc=os.path.basename(args.output)) as pbar:

        for ex in ds:
            title = (ex.get("title") or "").strip()
            text  = (ex.get("text")  or "").strip()

            # Redirects and stubs are noise, not language — skip them.
            if (not text or len(text) < args.min_text_len
                    or text.startswith(REDIRECT_PREFIXES)):
                n_skipped += 1
                continue

            block = f"{title}\n{text}\n\n"
            block_bytes = len(block.encode("utf-8"))

            # Graceful stop: don't write a block that would blow past the cap,
            # so the file lands just under the target instead of overshooting.
            if written + block_bytes > target_bytes:
                break

            f.write(block)
            written += block_bytes
            n_articles += 1
            pbar.update(block_bytes)     # progress = bytes toward the 100 MB target

    print(f"\nDone: {n_articles} articles -> {written / 1024 / 1024:.2f} MB "
          f"({n_skipped} skipped) in {args.output}")


if __name__ == "__main__":
    main()
