#!/usr/bin/env python3
"""
check_qa.py — fast Python sanity check + automatic grading: does a trained
checkpoint answer the Thai Q&A facts correctly?  (No QEMU / export needed.)

Loads a llama2.c checkpoint (ckpt.pt), runs greedy generation on the first
question of each fact in qa_thai.txt, then grades each answer against the
fact's "# key:" phrase(s): if the generated text contains any key, it is marked
PASS.  Prints a per-question PASS/FAIL table and a "N/total" score.

Also exposes grade() so tools/auto_improve.py can reuse the same grading logic
in-process (it returns the per-fact results instead of just printing them).

Usage:
    tools/venv/bin/python tools/check_qa.py \
        --checkpoint llama2c/out_thai_v2/ckpt.pt \
        --device cpu --max-new-tokens 40
"""
import argparse
import os
import re
import sys

import torch
import sentencepiece as spm

# import model.py from the llama2c/ dir (sibling of this script's repo)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "llama2c"))
from model import Transformer, ModelArgs  # noqa: E402


def load_facts(path):
    """Parse qa_thai.txt -> list of (first_question, keys) per fact."""
    facts = []
    questions = []
    keys = []

    def flush():
        if questions and keys:
            facts.append((questions[0], list(keys)))  # copy, not a live reference
        questions.clear()
        keys.clear()

    with open(path, encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                flush()
                continue
            if line.startswith("# key:"):
                keys = [k.strip() for k in line[len("# key:"):].split("|") if k.strip()]
                continue
            if line.startswith("#"):
                continue
            if line.startswith("คำถาม:"):
                questions.append(line[len("คำถาม:"):].strip())
        flush()
    return facts


def normalize(s):
    return re.sub(r"\s+", "", s)


def grade(checkpoint, qa, tokenizer, device="cpu", max_new_tokens=40, temperature=0.0):
    """Load a checkpoint and grade every fact in `qa`.

    Returns a dict:
        iter_num   : training iteration the checkpoint was saved at
        results    : list of {index, question, keys, answer, ok}
        n_pass     : number of PASS facts
        total      : number of facts
    """
    ckpt = torch.load(checkpoint, map_location="cpu")
    model_args = ckpt["model_args"]
    model = Transformer(ModelArgs(**model_args))
    model.load_state_dict(ckpt["model"])
    model = model.to(device)
    model.eval()

    sp = spm.SentencePieceProcessor(model_file=tokenizer)
    bos = sp.bos_id()

    facts = load_facts(qa)
    results = []
    for i, (q, keys) in enumerate(facts, 1):
        ids = [bos] + sp.encode(q, out_type=int)
        idx = torch.tensor([ids], dtype=torch.long, device=device)
        out = model.generate(idx, max_new_tokens, temperature=temperature)
        new_tokens = out[0].tolist()[len(ids):]
        answer = sp.decode(new_tokens).strip()

        norm = normalize(answer)
        ok = any(normalize(k) in norm for k in keys)
        results.append({"index": i, "question": q, "keys": keys,
                        "answer": answer, "ok": ok})

    n_pass = sum(1 for r in results if r["ok"])
    return {
        "iter_num": ckpt.get("iter_num"),
        "dim": model_args["dim"],
        "layers": model_args["n_layers"],
        "results": results,
        "n_pass": n_pass,
        "total": len(facts),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default=os.path.join(ROOT, "llama2c", "out_thai_v2", "ckpt.pt"))
    ap.add_argument("--qa", default=os.path.join(ROOT, "qa_thai_all.txt"))
    ap.add_argument("--tokenizer", default=os.path.join(ROOT, "tokenizer.model"))
    ap.add_argument("--max-new-tokens", type=int, default=40)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--temperature", type=float, default=0.0, help="0.0 = greedy")
    args = ap.parse_args()

    report = grade(args.checkpoint, args.qa, args.tokenizer,
                   device=args.device, max_new_tokens=args.max_new_tokens,
                   temperature=args.temperature)

    print(f"loaded {args.checkpoint}  (iter {report['iter_num']}, "
          f"dim={report['dim']} layers={report['layers']})")

    for r in report["results"]:
        mark = "PASS" if r["ok"] else "FAIL"
        print(f"\n[{r['index']:2d}] {mark}  key={r['keys']}")
        print(f"     ถาม: {r['question']}")
        print(f"     ตอบ: {r['answer']}")

    print(f"\n===== ผล: {report['n_pass']}/{report['total']} ถูก =====")


if __name__ == "__main__":
    main()
