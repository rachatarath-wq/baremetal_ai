#!/usr/bin/env bash
#
# sync_models.sh — copy Q8_0 model checkpoints (*_q80.bin) from the repo root
# into fat_usb/ so that create_usb.sh (/createboot) flashes them.
#
# Idempotent: only copies a file when its content actually changed (cmp).
# Skips tinyllama_q80.bin (1.1 GB) because it cannot fit on the 1 GB USB.
#
# Usage:
#   ./sync_models.sh              # sync repo root -> fat_usb/ (staging only)
#   ./sync_models.sh /dev/disk6   # also push onto a mounted USB, then eject (no reformat)
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$SCRIPT_DIR"
DST="$SCRIPT_DIR/fat_usb"
VOL_NAME="BAREMETAL"

# sync_to <src_dir> <dst_dir>  — copy model files that changed.
sync_to() {
    local src="$1" dst="$2" name src_bin
    local copied=0 skipped=0
    mkdir -p "$dst"

    # 1) Q8_0 model checkpoints.
    for src_bin in "$src"/*_q80.bin; do
        [[ -e "$src_bin" ]] || continue
        name="$(basename "$src_bin")"
        if [[ "$name" == "tinyllama_q80.bin" ]]; then
            echo "  SKIP $name (1.1 GB — too big for the 1 GB USB)"
            skipped=$((skipped+1))
            continue
        fi
        if cmp -s "$src_bin" "$dst/$name" 2>/dev/null; then
            echo "  ok    $name (unchanged)"
        else
            cp -p "$src_bin" "$dst/$name"
            echo "  copy  $name"
            copied=$((copied+1))
        fi
    done

    # NOTE: tokenizer.bin and kb.txt are intentionally NOT synced here. They are
    # managed by the build flow (tools/build_kb.py -> fat_usb/kb.txt) and may be
    # edited directly in fat_usb/, so this script must never overwrite them.

    echo "  -> $copied copied, $skipped skipped"
}

echo "Syncing models: $SRC -> $DST"
sync_to "$SRC" "$DST"

# Optional: also push straight onto a plugged-in USB without reformatting.
DISK_ARG="${1:-}"
if [[ -n "$DISK_ARG" ]]; then
    DISK="$(printf '%s' "$DISK_ARG" | sed -E 's#^/dev/##; s#^r##')"
    if [[ ! "$DISK" =~ ^disk[0-9]+$ ]]; then
        echo "ERROR: '$DISK_ARG' is not a whole-disk identifier (expected /dev/diskX)."
        exit 1
    fi
    MOUNT="/Volumes/$VOL_NAME"
    if [[ ! -d "$MOUNT" ]]; then
        echo ""
        echo "Mounting /dev/${DISK}s1 ..."
        diskutil mount "/dev/${DISK}s1" >/dev/null
    fi
    if [[ -d "$MOUNT" ]]; then
        echo ""
        echo "Pushing models onto mounted USB at $MOUNT ..."
        sync_to "$SRC" "$MOUNT"
        sync
        diskutil eject "/dev/$DISK"
        echo "Done — USB ejected."
    else
        echo "ERROR: could not mount /dev/${DISK}s1 (is the USB plugged in?)."
        exit 1
    fi
fi
