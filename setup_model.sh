#!/usr/bin/env bash
#
# setup_model.sh — build tinyllama_q80.bin + tokenizer.bin for the bare-metal
# UEFI llama2.c port, then install them into fat_usb/.
#
# IMPORTANT: llama2.c's `export.py --version 2` already emits the Q8_0-quantized
#            .bin directly. There is NO separate `quantize.py` step.
#
# Usage:
#   ./setup_model.sh                  # build + install + delete the ~4.4 GB HF download
#   CLEANUP_MODEL=0 ./setup_model.sh  # build + install + KEEP the HF download
#
# Requirements:
#   - python3, git
#   - ~7 GB free disk space (the HF model is ~4.4 GB fp32)
#   - ~8 GB RAM during export (a 1.1B fp32 model is loaded into memory)
#
# Outputs (installed into fat_usb/):
#   fat_usb/tinyllama_q80.bin   (~1.2 GB Q8_0 checkpoint)
#   fat_usb/tokenizer.bin       (~1.7 MB, 32000-vocab)

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOOLS_DIR="$ROOT/tools"
LLAMA2C_DIR="$TOOLS_DIR/llama2.c"
VENV_DIR="$TOOLS_DIR/venv"
MODEL_REPO="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
MODEL_DIR="$TOOLS_DIR/TinyLlama-1.1B-Chat-v1.0"

FAT_USB="$ROOT/fat_usb"
OUT_BIN="$ROOT/tinyllama_q80.bin"
TOK_BIN="$ROOT/tokenizer.bin"

# Set to 0 to keep the ~4.4 GB HF checkpoint after the build.
CLEANUP_MODEL="${CLEANUP_MODEL:-1}"

echo "==> [1/6] Cloning llama2.c ..."
mkdir -p "$TOOLS_DIR"
if [ ! -d "$LLAMA2C_DIR/.git" ]; then
    git clone --depth 1 https://github.com/karpathy/llama2.c "$LLAMA2C_DIR"
fi

echo "==> [2/6] Creating Python venv and installing deps ..."
if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR"
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
python -m pip install --quiet --upgrade pip
python -m pip install --quiet torch transformers sentencepiece huggingface_hub

echo "==> [3/6] Downloading $MODEL_REPO ..."
if [ ! -f "$MODEL_DIR/config.json" ]; then
    # Use the library API instead of the `huggingface-cli` console script (which
    # only ships with the `huggingface_hub[cli]` extra on recent versions).
   # python -c 'import sys; from huggingface_hub import snapshot_download; snapshot_download(repo_id=sys.argv[1], local_dir=sys.argv[2])' "$MODEL_REPO" "$MODEL_DIR"
    python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='$MODEL_REPO', local_dir='$MODEL_DIR')"
fi

echo "==> [4/6] Exporting Q8_0 weights (export.py --version 2) ..."
( cd "$LLAMA2C_DIR" && python export.py "$OUT_BIN" --version 2 --hf "$MODEL_DIR" )

echo "==> [5/6] Building tokenizer.bin ..."
# tokenizer.py writes tokenizer.bin next to the input tokenizer.model
( cd "$LLAMA2C_DIR" && python tokenizer.py --tokenizer-model "$MODEL_DIR/tokenizer.model" )
mv -f "$MODEL_DIR/tokenizer.bin" "$TOK_BIN"

echo "==> [6/6] Installing into $FAT_USB ..."
mkdir -p "$FAT_USB"
cp -f "$OUT_BIN" "$FAT_USB/tinyllama_q80.bin"
cp -f "$TOK_BIN"  "$FAT_USB/tokenizer.bin"

if [ "$CLEANUP_MODEL" = "1" ]; then
    echo "==> Cleaning up the ~4.4 GB HF model download ..."
    rm -rf "$MODEL_DIR"
fi

echo ""
echo "Done."
echo "  checkpoint : $FAT_USB/tinyllama_q80.bin ($(du -h "$FAT_USB/tinyllama_q80.bin" | cut -f1))"
echo "  tokenizer  : $FAT_USB/tokenizer.bin ($(du -h "$FAT_USB/tokenizer.bin" | cut -f1))"
echo "  (also kept at $OUT_BIN and $TOK_BIN)"
