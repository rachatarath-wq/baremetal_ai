#!/usr/bin/env python3
"""
build_kb.py — build kb.txt (tab-separated "question<TAB>answer", UTF-8) from
qa_thai_all.txt, for the bare-metal knowledge-base lookup layer
(LEARNING_SYSTEM_PLAN.md, ชั้นที่ 1).

Each fact in qa_thai_all.txt has several "คำถาม:" lines (alternate phrasings)
and one or more "คำตอบ:" lines.  We emit one kb.txt line per question, paired
with that fact's first answer, so the device can answer any stored phrasing
verbatim instead of re-generating (and hallucinating) it.

Usage:
    tools/venv/bin/python tools/build_kb.py          # writes kb.txt in repo root
    tools/venv/bin/python tools/build_kb.py -o kb.txt
"""
import argparse
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def parse(path):
    """qa_thai_all.txt -> list of (questions, first_answer)."""
    facts = []
    questions = []
    answer = None

    def flush():
        nonlocal answer
        if questions and answer:
            facts.append((questions[:], answer))
        questions.clear()
        answer = None

    with open(path, encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\n").rstrip("\r").strip()
            if not line:
                flush()
                continue
            if line.startswith("#"):
                continue
            if line.startswith("คำถาม:"):
                questions.append(line[len("คำถาม:"):].strip())
            elif line.startswith("คำตอบ:"):
                a = line[len("คำตอบ:"):].strip()
                if answer is None:
                    answer = a
        flush()
    return facts


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--qa", default=os.path.join(ROOT, "qa_thai_all.txt"))
    ap.add_argument("-o", "--output", default=os.path.join(ROOT, "kb.txt"))
    args = ap.parse_args()

    facts = parse(args.qa)
    n = 0
    with open(args.output, "w", encoding="utf-8") as out:
        for questions, answer in facts:
            for q in questions:
                out.write(f"{q}\t{answer}\n")
                n += 1
    print(f"{n} Q/A lines from {len(facts)} facts -> {args.output}")


if __name__ == "__main__":
    main()
