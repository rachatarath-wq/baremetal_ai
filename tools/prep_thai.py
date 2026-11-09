#!/usr/bin/env python3
"""
prep_thai.py — tokenize a raw Thai UTF-8 corpus into llama2.c's uint16 format.

Usage:
    python prep_thai.py --input thai_corpus.txt --out_dir data \
        --tokenizer tokenizer.model --val_split 0.1 --seed 42

Outputs (both are flat little-endian uint16 arrays, no header):
    data/train.bin    [BOS] tok tok ... [EOS] [BOS] ... [EOS]
    data/val.bin      same format, held-out documents
"""
import os
import argparse
import random
import numpy as np
import sentencepiece as spm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True, help='raw UTF-8 Thai corpus')
    ap.add_argument('--out_dir', default='data', help='where train.bin/val.bin go')
    ap.add_argument('--tokenizer', default='tokenizer.model', help='Llama 2 sp model')
    ap.add_argument('--val_split', type=float, default=0.1, help='fraction for validation')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--split_on', default='blank', choices=['blank', 'line'],
                    help="document boundary: 'blank' = paragraphs, 'line' = one line each")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # 1. Load the standard Llama 2 SentencePiece tokenizer (vocab must be 32000
    #    to match both train.py's vocab_size and your existing tokenizer.bin).
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer)
    assert sp.vocab_size() == 32000, f"expected vocab 32000, got {sp.vocab_size()}"
    bos, eos = sp.bos_id(), sp.eos_id()   # 1 and 2 for Llama 2

    # 2. Read the whole corpus as UTF-8, normalise line endings.
    with open(args.input, encoding='utf-8') as f:
        text = f.read()
    text = text.replace('\r\n', '\n').replace('\r', '\n')

    # 3. Split into documents (each becomes one BOS..EOS training example).
    docs = [d.strip() for d in (text.split('\n\n') if args.split_on == 'blank'
                                else text.split('\n'))]
    docs = [d for d in docs if d]          # drop empty lines/paragraphs
    assert docs, "no non-empty documents found"

    # 4. Deterministic document-level train/val split (no leakage across a doc).
    rng = random.Random(args.seed)
    order = list(range(len(docs)))
    rng.shuffle(order)
    n_val = max(1, int(len(docs) * args.val_split))
    val_set = set(order[:n_val])

    # 5. Encode each document as [BOS] + ids + [EOS], concatenating everything.
    def encode_all(doc_list):
        ids = []
        for d in doc_list:
            ids.append(bos)
            ids.extend(sp.encode(d, out_type=int))
            ids.append(eos)
        return ids

    train_ids = encode_all([d for i, d in enumerate(docs) if i not in val_set])
    val_ids   = encode_all([d for i, d in enumerate(docs) if i in val_set])

    # 6. Write flat uint16 arrays (native/little-endian), exactly what
    #    np.memmap(..., dtype=np.uint16) in train.py expects.
    np.array(train_ids, dtype=np.uint16).tofile(os.path.join(args.out_dir, 'train.bin'))
    np.array(val_ids,   dtype=np.uint16).tofile(os.path.join(args.out_dir, 'val.bin'))

    # Sanity: decode the first document back to text to confirm Thai round-trips.
    sample = sp.decode(train_ids[:30])
    print(f"docs: {len(docs)}  (train {len(docs)-len(val_set)}, val {len(val_set)})")
    print(f"train tokens: {len(train_ids):,}   val tokens: {len(val_ids):,}")
    print(f"wrote {args.out_dir}/train.bin and {args.out_dir}/val.bin")
    print(f"sample decode: {sample!r}")


if __name__ == '__main__':
    main()
