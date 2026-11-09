#!/usr/bin/env python3
"""
build_qa_corpus.py — turn a fact file (qa_thai.txt) into a repeated Q&A
training corpus, in the same blank-line-separated chunk format prep_thai.py
expects.

The fact file format (one fact per blank-line block):

    คำถาม: ภาษาไทยมีพยัญชนะกี่ตัว
    คำถาม: พยัญชนะไทยมีกี่ตัว
    คำตอบ: ภาษาไทยมีพยัญชนะ 44 ตัว

Every "คำถาม:" line is paired with every "คำตอบ:" line in the block, and each
pair is emitted in several surface forms so the model sees the pattern with
structural cues ("คำถาม:/คำตอบ:") and without (bare question, next line the
answer).  Each emitted chunk is repeated `--repeat` times so a small fact set
gets enough coverage for a 15M model to memorise it.

Usage:
    python build_qa_corpus.py [--input qa_thai.txt] [--output thai_qa.txt]
                              [--repeat 100]
"""
import argparse
import sys

Q_PREFIX = "คำถาม:"
A_PREFIX = "คำตอบ:"
ALT_Q_PREFIX = "ถาม:"
ALT_A_PREFIX = "ตอบ:"


def parse_facts(path):
    """Return a list of (questions:list[str], answers:list[str])."""
    facts = []
    qs, ans = [], []
    with open(path, encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                if qs or ans:
                    facts.append((qs, ans))
                    qs, ans = [], []
                continue
            if line.startswith(Q_PREFIX):
                qs.append(line[len(Q_PREFIX):].strip())
            elif line.startswith(A_PREFIX):
                ans.append(line[len(A_PREFIX):].strip())
            elif line.startswith("//"):
                continue
        if qs or ans:
            facts.append((qs, ans))
    return facts


def forms(q, a):
    """Return the surface forms for one (question, answer) pair."""
    return [
        f"{Q_PREFIX} {q}\n{A_PREFIX} {a}",
        f"{ALT_Q_PREFIX} {q}\n{ALT_A_PREFIX} {a}",
        f"{q}\n{a}",
    ]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default="qa_thai.txt")
    ap.add_argument("--output", default="thai_qa.txt")
    ap.add_argument("--repeat", type=int, default=100,
                    help="copies of each surface form per question")
    args = ap.parse_args()

    facts = parse_facts(args.input)
    n_pairs = 0
    n_chunks = 0

    with open(args.output, "w", encoding="utf-8") as out:
        for qs, ans in facts:
            if not qs or not ans:
                print(f"warning: block with missing Q or A skipped (Q={qs}, A={ans})",
                      file=sys.stderr)
                continue
            for q in qs:
                for a in ans:
                    n_pairs += 1
                    for text in forms(q, a):
                        for _ in range(args.repeat):
                            out.write(text + "\n\n")
                            n_chunks += 1

    print(f"facts: {len(facts)}  question/answer pairs: {n_pairs}  "
          f"chunks: {n_chunks:,}  (--repeat {args.repeat}) -> {args.output}")


if __name__ == "__main__":
    main()
