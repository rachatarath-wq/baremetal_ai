# Pre-training a 15M Thai model from scratch (bare-metal UEFI engine)

End-to-end pipeline for training your own TinyLlama-sized Thai model and exporting
it to the Q8_0 v2 binary that `run.c` (the UEFI payload) loads.

```
Thai Wikipedia ──► thai_corpus.txt ──► train.bin / val.bin ──► train.py ──► ckpt.pt ──► export.py ──► stories15M_thai_q80.bin ──► run.c
                    (tools/build_       (uint16, tools/prep_      (MPS)                   (Q8_0 v2)
                     thai_corpus.py)     thai.py)
```

The target architecture is **identical** to the existing `stories15M_q80.bin`, so the
same `export.py --version 2` command and the same `tokenizer.bin` are reused unchanged:

| | |
|---|---|
| `dim` | 288 |
| `hidden_dim` | 768 (4 × dim) |
| `n_layers` | 6 |
| `n_heads` | 6 |
| `n_kv_heads` | 6 |
| `vocab_size` | 32000 (Llama 2 tokenizer) |
| `seq_len` / `max_seq_len` | 256 |
| `multiple_of` | 32 |
| `head_size` | 48 (dim / n_heads) |

---

## 0. Prerequisites

```bash
pip install torch numpy sentencepiece datasets tqdm
```

The Python side of llama2.c lives in `llama2c/`: `model.py` + `export.py` are the
upstream `karpathy/llama2.c` `master` files, and `train.py` is a local wrapper that
reads your `data/*.bin` (see the version note below).

> **⚠️ Version drift — resolved.** Upstream `karpathy/llama2.c` `train.py` does **not**
> read local `train.bin`/`val.bin` — it streams TinyStories from HuggingFace via
> `tinystories.Task` + `configurator.py`. So this repo ships `llama2c/train.py`
> (a `np.memmap` loader for your `data/train.bin`/`val.bin`) while importing the upstream
> `llama2c/model.py` + `llama2c/export.py` unchanged. Your `run.c` (magic `0x616b3432`,
> version 2, Q8_0) matches `export.py --version 2` exactly — no commit pinning needed.

---

## 1. Build the corpus (~100 MB of Thai Wikipedia)

`tools/build_thai_corpus.py` streams the Thai Wikipedia split
(`wikimedia/wikipedia`, config `20231101.th`) and writes `title` + `text` per article,
blank-line separated, stopping just under the size cap.

```bash
python3 tools/build_thai_corpus.py --output thai_corpus.txt --limit-mb 100
```

- `streaming=True` inside the script means it stops at 100 MB without downloading the
  whole (~GB) dump first.
- Redirects and stubs are skipped automatically.

---

## 2. Tokenize into `train.bin` / `val.bin`

`tools/prep_thai.py` encodes each document as `[BOS] tok … tok [EOS]` and writes a flat
little-endian `uint16` array (no header) — exactly what `train.py` reads via
`np.memmap(..., dtype=np.uint16)`.

```bash
python3 tools/prep_thai.py \
    --input thai_corpus.txt \
    --out_dir data \
    --tokenizer tokenizer.model
```

- `tokenizer.model` is the **Llama 2 SentencePiece model** (vocab 32000) from the
  llama2.c repo root. It must be this one — the ids must match `vocab_size=32000` and
  your existing `tokenizer.bin`.
- Outputs: `data/train.bin` and `data/val.bin` (the default directory `train.py` reads).

---

## 3. Train (Apple Silicon, MPS)

Use the bundled `llama2c/train.py` (a `np.memmap` loader for your `data/*.bin`). First a
quick smoke test to confirm the pipeline and see your tokens/sec:

```bash
cd llama2c
../tools/venv/bin/python train.py --device=mps --dtype=float32 \
    --max_iters=200 --lr_decay_iters=200 --warmup_iters=10 \
    --eval_interval=100 --log_interval=20
```

Then the real run. The 100 MB corpus is only ~36M tokens, so a 15M model starts
overfitting after ~10 epochs (~5,500 iters at the default batch 64×4). Start with a few
thousand iters:

```bash
cd llama2c
../tools/venv/bin/python train.py --device=mps --dtype=float32 \
    --max_iters=3000 --lr_decay_iters=3000
```

- **`--dtype=float32`** — the safe choice on MPS (bfloat16/float16 can be unstable here).
- No `torch.compile` — it is not supported on MPS (the script does not use it).
- Architecture defaults are already the 15M config (`--dim=288 --n_layers=6 --n_heads=6
  --n_kv_heads=6 --vocab_size=32000 --max_seq_len=256 --multiple_of=32`), so no flags needed.
- **`--max_iters` / `--lr_decay_iters`** are the main knobs. More data beats more
  iterations: raise `build_thai_corpus.py --limit-mb` before adding more iters.

The final checkpoint is `llama2c/out_thai/ckpt.pt`.

---

## 4. Export to Q8_0 v2

```bash
cd llama2c
../tools/venv/bin/python export.py ../stories15M_thai_q80.bin --checkpoint out_thai/ckpt.pt --version 2
```

- `--version 2` selects **Q8_0** (grouped int8 weights + fp32 scales/norms) and writes
  the `0x616b3432` magic, `version=2`, the 7-field config, the `shared_classifier` byte,
  and `group_size` — the exact header `run.c` hard-checks.
- The positional argument is the **output** filename; `--checkpoint` is the input `.pt`.
  If your checkout's `export.py` uses a different arg order, check `python3 export.py --help`.
- **Reuse your existing `tokenizer.bin`** — the tokenizer is unchanged.

---

## 5. Deploy to the UEFI drive

1. Copy `stories15M_thai_q80.bin` to the FAT32 root (next to `tokenizer.bin`).
2. In `run.c`, point at it:

   ```c
   CHAR16 *checkpoint_name = L"\\stories15M_thai_q80.bin";
   ```

3. `make` and boot. `run.c` reads `dim/hidden_dim/n_layers/n_heads/n_kv_heads/`
   `vocab_size/seq_len/shared_classifier/group_size` all from the header, so no other
   code change is needed.

---

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `export.py` errors: "…divisible by group_size" | `dim=288` is not a multiple of the default `group_size=64`. Use a value that divides 288 (e.g. 32) — the same one your existing `stories15M_q80.bin` used. The header records it; `run.c` reads it automatically. |
| `sentencepiece` can't load `tokenizer.model` | Grab it from the llama2.c repo root (it's bundled there). |
| Wrong / out-of-range token ids | `tokenizer.model` must be the Llama 2 one (32000). A different SentencePiece model yields ids ≥ 32000 that don't match `run.c`. |
| `train.py` can't find `data/train.bin` | The classic `train.py` reads from `data/`. Run `prep_thai.py` with `--out_dir data` (the default). |
| `--device=mps` slow or wrong | Keep `--dtype=float32` and `--compile=False`; MPS + bfloat16/compile are unreliable in this codebase. |
| Model overfits | Too few documents / too many iters for a small corpus. Grow the corpus (raise `--limit-mb`) or lower `--max_iters`. |

---

## Files

| File | Purpose |
|---|---|
| `tools/build_thai_corpus.py` | Download Thai Wikipedia → `thai_corpus.txt` (~100 MB cap) |
| `tools/prep_thai.py` | Tokenize corpus → `data/train.bin`, `data/val.bin` (uint16) |
| `llama2c/` | llama2.c Python side — `model.py`/`export.py` (upstream) + `train.py` (memmap loader) |
| `USAGE_THAI.md` | Running the small model + Thai (Kedmanee) keyboard input on the engine |
| `README.md` | The UEFI/`run.c` port itself (build, deploy, math) |
