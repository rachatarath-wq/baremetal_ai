#!/usr/bin/env python3
"""
auto_improve.py — the "auto-boost loop" (AUTO_IMPROVE_PLAN.md, แผน A).

Turns the manual "grade → boost failed fact → rebuild QA corpus → finetune"
cycle into one command.  Each round:

    1. grade the checkpoint with check_qa.grade()  → find FAIL facts
    2. if everything passes, stop
    3. collect the failed facts' "# key:" phrases into a *persistent* boost set
       (so fixing a new fact does not un-memorise an old one)
    4. rebuild thai_qa.txt with those facts repeated more (build_qa_corpus --boost)
    5. concatenate corpus + thai_qa.txt → combined corpus, re-tokenise (prep_thai)
    6. finetune (train.py --init_from resume --finetune) for --finetune-iters steps
    7. loop back to grading

Stop conditions: every fact passes, or --max-rounds is reached.

Usage (from the repo root):
    tools/venv/bin/python tools/auto_improve.py \
        --checkpoint llama2c/out_thai_v2/ckpt.pt \
        --max-rounds 5 --finetune-iters 500 \
        [--dry-run]   # plan only, no training
"""
import argparse
import os
import shlex
import subprocess
import sys

# make `import check_qa` work when this script is run as tools/auto_improve.py
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import check_qa  # noqa: E402


def run(cmd, cwd, dry_run):
    print(f"  $ cd {cwd} && {shlex.join(cmd)}")
    if dry_run:
        return
    subprocess.run(cmd, cwd=cwd, check=True)


def concat(corpus, qa_corpus, combined):
    """cat corpus + qa_corpus into combined, with a blank-line boundary."""
    with open(combined, "w", encoding="utf-8") as out:
        with open(corpus, encoding="utf-8") as f:
            out.write(f.read().rstrip())
        out.write("\n\n")
        with open(qa_corpus, encoding="utf-8") as f:
            out.write(f.read())
    print(f"  combined {corpus} + {qa_corpus} -> {combined}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    # files
    ap.add_argument("--qa", default=os.path.join(ROOT, "qa_thai_all.txt"))
    ap.add_argument("--corpus", default=os.path.join(ROOT, "thai_corpus_clean.txt"))
    ap.add_argument("--qa-corpus", default=os.path.join(ROOT, "thai_qa.txt"))
    ap.add_argument("--combined", default=os.path.join(ROOT, "thai_corpus_v2.txt"))
    ap.add_argument("--tokenizer", default=os.path.join(ROOT, "tokenizer.model"))
    ap.add_argument("--data-dir", default=os.path.join(ROOT, "data"))
    ap.add_argument("--checkpoint",
                    default=os.path.join(ROOT, "llama2c", "out_thai_v2", "ckpt.pt"))
    # loop control
    ap.add_argument("--max-rounds", type=int, default=5)
    ap.add_argument("--finetune-iters", type=int, default=500,
                    help="training steps per finetune round (~45 min on MPS)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print every command but run none")
    # grading
    ap.add_argument("--grade-device", default="cpu")
    ap.add_argument("--max-new-tokens", type=int, default=40)
    # corpus build
    ap.add_argument("--repeat", type=int, default=100)
    ap.add_argument("--boost-mult", type=float, default=3.0)
    # training
    ap.add_argument("--train-device", default="mps")
    ap.add_argument("--learning-rate", type=float, default=1e-3)
    ap.add_argument("--no-caffeinate", action="store_true",
                    help="do not wrap the train step in `caffeinate -i`")
    args = ap.parse_args()

    out_dir = os.path.dirname(args.checkpoint)
    py = sys.executable
    boost_keys = set()

    report = check_qa.grade(args.checkpoint, args.qa, args.tokenizer,
                            device=args.grade_device,
                            max_new_tokens=args.max_new_tokens)

    for rnd in range(1, args.max_rounds + 1):
        results = report["results"]
        fails = [r for r in results if not r["ok"]]
        print(f"\n=== round {rnd}: {report['n_pass']}/{report['total']} pass "
              f"(iter {report['iter_num']}) ===")
        for r in fails:
            print(f"    FAIL [{r['index']}] key={r['keys']}  ตอบ={r['answer']}")

        if not fails:
            print(f"✅ all {report['total']}/{report['total']} facts pass — done.")
            break

        for r in fails:
            boost_keys.update(r["keys"])
        print(f"  boost keys now: {sorted(boost_keys)}")

        if rnd == args.max_rounds:
            print(f"⚠️  reached --max-rounds {args.max_rounds}, still failing "
                  f"{len(fails)} fact(s).")
            break

        # rebuild QA corpus with the persistent boost set
        run([py, os.path.join(ROOT, "tools", "build_qa_corpus.py"),
             "--input", args.qa, "--output", args.qa_corpus,
             "--repeat", str(args.repeat),
             "--boost", ",".join(sorted(boost_keys)),
             "--boost-mult", str(args.boost_mult)], cwd=ROOT, dry_run=args.dry_run)

        # concatenate clean corpus + QA corpus, then re-tokenise
        if not args.dry_run:
            concat(args.corpus, args.qa_corpus, args.combined)
        run([py, os.path.join(ROOT, "tools", "prep_thai.py"),
             "--input", args.combined, "--out_dir", args.data_dir,
             "--tokenizer", args.tokenizer], cwd=ROOT, dry_run=args.dry_run)

        # finetune: resume + restart LR schedule, train finetune_iters more steps
        warmup = max(10, args.finetune_iters // 2)
        target_iter = report["iter_num"] + args.finetune_iters
        cmd = [py, os.path.join(ROOT, "llama2c", "train.py"),
               "--data_dir", args.data_dir, "--out_dir", out_dir,
               "--init_from", "resume", "--finetune",
               "--device", args.train_device, "--dtype", "float32",
               "--batch_size", "16", "--gradient_accumulation_steps", "16",
               "--max_iters", str(target_iter),
               "--lr_decay_iters", str(args.finetune_iters),
               "--warmup_iters", str(warmup),
               "--learning_rate", str(args.learning_rate),
               "--eval_interval", str(args.finetune_iters),
               "--eval_iters", "100", "--log_interval", "20"]
        if sys.platform == "darwin" and not args.no_caffeinate:
            cmd = ["caffeinate", "-i"] + cmd
        run(cmd, cwd=ROOT, dry_run=args.dry_run)

        # re-grade the newly finetuned checkpoint
        report = check_qa.grade(args.checkpoint, args.qa, args.tokenizer,
                                device=args.grade_device,
                                max_new_tokens=args.max_new_tokens)

    # final summary
    results = report["results"]
    fails = [r for r in results if not r["ok"]]
    print(f"\n===== สรุป: {report['n_pass']}/{report['total']} ถูก "
          f"(iter {report['iter_num']}, boost keys {sorted(boost_keys) or 'none'}) =====")
    if fails:
        for r in fails:
            print(f"  ยังผิด [{r['index']}] key={r['keys']}: {r['answer']}")
        sys.exit(1)


if __name__ == "__main__":
    main()
