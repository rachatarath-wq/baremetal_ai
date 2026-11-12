#!/usr/bin/env bash
#
# create_usb.sh — format a USB drive as MS-DOS FAT32 (MBR) and copy the
# bare-metal UEFI AI files (EFI/BOOT/BOOTX64.EFI, the model .bin, tokenizer.bin,
# kb.txt) from ./fat_usb onto it, then sync + eject.
#
# Usage:
#   diskutil list                    # find your USB (e.g. /dev/disk4)
#   ./create_usb.sh /dev/disk4
#
# Safety: refuses to erase internal/system disks, requires typing "YES" to proceed.
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="$SCRIPT_DIR/fat_usb"
VOL_NAME="BAREMETAL"
VOLUME="/Volumes/$VOL_NAME"
FILESYSTEM="MS-DOS FAT32"
SCHEME="MBR"

# ---------------------------------------------------------------------------
# 1. Argument parsing
# ---------------------------------------------------------------------------
DISK_ARG="${1:-}"
if [[ -z "$DISK_ARG" ]]; then
    echo "Usage: $0 /dev/diskX"
    echo "  example: $0 /dev/disk4"
    echo ""
    echo "Find your USB drive with:  diskutil list"
    echo "  (match it by SIZE and the '(external, physical)' tag, then note its disk number)"
    exit 1
fi

# Normalize: accept /dev/disk4, /dev/rdisk4, rdisk4, or disk4  ->  disk4
DISK="$(printf '%s' "$DISK_ARG" | sed -E 's#^/dev/##; s#^r##')"
if [[ ! "$DISK" =~ ^disk[0-9]+$ ]]; then
    echo "ERROR: '$DISK_ARG' is not a whole-disk identifier (expected /dev/diskX)."
    echo "       If you passed a partition like disk4s1, use the whole disk instead."
    exit 1
fi

BLOCK="/dev/$DISK"

# ---------------------------------------------------------------------------
# 2. Safety checks
# ---------------------------------------------------------------------------
INFO="$(mktemp)"
trap 'rm -f "$INFO"' EXIT

if ! diskutil info -plist "$BLOCK" > "$INFO" 2>/dev/null; then
    echo "ERROR: cannot query $BLOCK. Does it exist?"
    echo "       List disks with:  diskutil list"
    exit 1
fi

# Read properties from the plist (PlistBuddy ships with macOS).
pget() { /usr/libexec/PlistBuddy -c "Print :$1" "$INFO" 2>/dev/null || true; }

INTERNAL="$(pget Internal)"
REMOVABLE="$(pget RemovableMedia)"
[[ -z "$REMOVABLE" ]] && REMOVABLE="$(pget Removable)"
EJECTABLE="$(pget Ejectable)"
BUS="$(pget BusProtocol)"
SIZE_BYTES="$(pget DiskSize)"
[[ -z "$SIZE_BYTES" ]] && SIZE_BYTES="$(pget TotalSize)"
[[ -z "$SIZE_BYTES" ]] && SIZE_BYTES="$(pget Size)"
MEDIA_NAME="$(pget MediaName)"

# Hard fail: never erase an internal/system disk.
if [[ "$INTERNAL" == "true" ]]; then
    echo "ERROR: $BLOCK is an INTERNAL disk. Refusing to erase a system drive."
    exit 1
fi
if [[ -z "$INTERNAL" ]]; then
    echo "WARNING: could not determine whether $BLOCK is internal. Review carefully."
fi

# Soft warning: flag disks that don't look removable/external.
EXT_OK=0
[[ "$REMOVABLE" == "true" ]] && EXT_OK=1
[[ "$EJECTABLE" == "true" ]] && EXT_OK=1
for proto in USB Thunderbolt FireWire; do
    if [[ "$BUS" == *"$proto"* ]]; then
        EXT_OK=1
    fi
done
if [[ "$EXT_OK" -ne 1 ]]; then
    echo "WARNING: $BLOCK does not look removable/external"
    echo "         (BusProtocol='$BUS', Removable='$REMOVABLE', Ejectable='$EJECTABLE')."
    echo "         Proceeding only because Internal != true."
fi

# Human-readable size helper.
hr() { awk -v b="${1:-0}" 'BEGIN{ if (b >= 1000000000) printf "%.2f GB\n", b/1000000000; else printf "%.1f MB\n", b/1000000 }'; }

# Source directory must exist.
if [[ ! -d "$SRC_DIR" ]]; then
    echo "ERROR: source directory not found: $SRC_DIR"
    exit 1
fi
SRC_BYTES="$(du -sk "$SRC_DIR" | awk '{print $1*1024}')"

# Make sure the target is big enough for the files.
if [[ -n "$SIZE_BYTES" && "$SIZE_BYTES" -lt "$SRC_BYTES" ]]; then
    echo "ERROR: target disk ($(hr "$SIZE_BYTES")) is smaller than the files to copy ($(hr "$SRC_BYTES"))."
    exit 1
fi

# ---------------------------------------------------------------------------
# 3. Confirmation
# ---------------------------------------------------------------------------
echo ""
echo "=================== CREATE USB ==================="
echo "  Target disk : $BLOCK"
echo "  Media name  : ${MEDIA_NAME:-?}"
echo "  Size        : $(hr "${SIZE_BYTES:-0}")"
echo "  Bus         : ${BUS:-?}  (Removable=${REMOVABLE:-?}, Ejectable=${EJECTABLE:-?})"
echo "  Filesystem  : $FILESYSTEM, $SCHEME partition map, volume '$VOL_NAME'"
echo "  Source      : $SRC_DIR  ($(hr "$SRC_BYTES"))"
echo "=================================================="
echo ""
echo "This will PERMANENTLY ERASE everything on $BLOCK."
read -r -p "Type YES (uppercase) to continue: " CONFIRM
if [[ "$CONFIRM" != "YES" ]]; then
    echo "Aborted. Nothing was changed."
    exit 1
fi

# ---------------------------------------------------------------------------
# 4. Format
# ---------------------------------------------------------------------------
echo ""
echo "==> Formatting $BLOCK as $FILESYSTEM ($SCHEME) ..."
diskutil eraseDisk "$FILESYSTEM" "$VOL_NAME" "$SCHEME" "$BLOCK"

# Wait for the volume to appear.
for _ in $(seq 1 20); do
    if [[ -d "$VOLUME" ]]; then
        break
    fi
    sleep 1
done
if [[ ! -d "$VOLUME" ]]; then
    echo "ERROR: $VOLUME did not mount after formatting."
    exit 1
fi

# ---------------------------------------------------------------------------
# 5. Copy (with progress for the 1.1 GB .bin)
# ---------------------------------------------------------------------------
echo "==> Copying files to $VOLUME ..."
if command -v rsync >/dev/null 2>&1; then
    rsync -r --progress --no-perms --no-owner --no-group --no-times \
          --exclude '.DS_Store' --exclude '._*' \
          --exclude '.Spotlight-V100' --exclude '.Trashes' --exclude '.fseventsd' \
          "$SRC_DIR/" "$VOLUME/"
else
    cp -Rp "$SRC_DIR/." "$VOLUME/"
fi

echo "==> Verifying ..."
# Boot-critical files: the app, the default checkpoint, and the tokenizer.
# kb.txt is optional (run.c skips KB lookup if it is missing).
for f in "EFI/BOOT/BOOTX64.EFI" "stories15M_thai_q80.bin" "tokenizer.bin"; do
    if [[ ! -e "$VOLUME/$f" ]]; then
        echo "ERROR: expected file not found after copy: $f"
    fi
done
[[ -e "$VOLUME/kb.txt" ]] || echo "NOTE: kb.txt not present — the KB layer will be skipped."
ls -la "$VOLUME" "$VOLUME/EFI/BOOT" 2>/dev/null || true

# ---------------------------------------------------------------------------
# 6. Sync + eject
# ---------------------------------------------------------------------------
echo "==> Flushing writes and ejecting ..."
sync
sleep 1
diskutil eject "$BLOCK"

echo ""
echo "Done. $BLOCK is ready — boot a PC in UEFI mode and pick the 'UEFI:' entry."
