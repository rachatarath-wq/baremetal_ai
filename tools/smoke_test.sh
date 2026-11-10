#!/usr/bin/env bash
# smoke_test.sh — verify the 3 learning layers (KB lookup / 'ไม่ทราบ' / 'จำไว้:')
# work in a RUNNING, already-booted QEMU guest.
#
# Prereqs:
#   1. make install  &&  tools/venv/bin/python tools/build_kb.py  &&  cp kb.txt fat_usb/
#   2. In another terminal:  make qemu-headless   (wait ~50s for boot)
#
# Usage:  ./tools/smoke_test.sh [qmp-socket]     # default /tmp/qemu_bm.sock
set -u
SOCK="${1:-/tmp/qemu_bm.sock}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="$ROOT/tools/venv/bin/python"
TYPE="$ROOT/tools/qemu_type.py"
OCR="$ROOT/tools/ocr.swift"

if [ ! -S "$SOCK" ]; then
  echo "error: $SOCK not found — is 'make qemu-headless' running and booted?"
  exit 1
fi

# type + screendump + OCR -> echo the OCR text (whitespace-normalised)
ocr_shot() {
  local ppm="$1"
  local png="${ppm%.ppm}.png"
  sips -s format png "$ppm" --out "$png" >/dev/null 2>&1
  swift "$OCR" "$png" 2>/dev/null | tr -d ' \n\r\t'
}

# fuzzy substring check on OCR text
check() {
  local desc="$1" want="$2" got="$3"
  local w g
  w="$(printf '%s' "$want" | tr -d ' \n\r\t')"
  g="$(printf '%s' "$got"  | tr -d ' \n\r\t')"
  if printf '%s' "$g" | grep -qF "$w"; then echo "PASS  $desc"
  else echo "FAIL  $desc  (expected '$want')"; fi
}

echo "== smoke test กับ $SOCK =="

# --- ชั้น 1: รู้จัก -> ตอบจาก KB (guest เริ่มที่ EN หลัง boot) ---
"$PY" "$TYPE" "$SOCK" --start-mode en --th "พ่อเราชื่ออะไร" --shot /tmp/st1.ppm >/dev/null
t1="$(ocr_shot /tmp/st1.ppm)"
check "ชั้น 1 รู้จัก -> ตอบ verbatim" "คุณรชต" "$t1"

# --- ชั้น 2: ไม่รู้ -> 'ไม่ทราบ' (guest ตอนนี้เป็น TH) ---
"$PY" "$TYPE" "$SOCK" --start-mode th --th "ทำไมท้องฟ้าถึงเป็นสีฟ้า" --shot /tmp/st2.ppm >/dev/null
t2="$(ocr_shot /tmp/st2.ppm)"
check "ชั้น 2 ไม่รู้ -> 'ไม่ทราบ'" "ไม่ทราบ" "$t2"

# --- ชั้น 3: จำไว้ -> append ลง kb.txt (check ไฟล์จริง ไม่ผ่าน OCR) ---
BEFORE="$(wc -l < "$ROOT/fat_usb/kb.txt" 2>/dev/null || echo 0)"
"$PY" "$TYPE" "$SOCK" --start-mode th \
  --segments "th:จำไว้,en:: ,th:ดาวพุธมีสีอะไร,en: = ,th:สีเหลือง" \
  --shot /tmp/st3.ppm >/dev/null
AFTER="$(wc -l < "$ROOT/fat_usb/kb.txt" 2>/dev/null || echo 0)"
if [ "$AFTER" -gt "$BEFORE" ] && grep -qF "ดาวพุธมีสีอะไร" "$ROOT/fat_usb/kb.txt"; then
  echo "PASS  ชั้น 3 จำไว้ -> append ลง kb.txt ($BEFORE -> $AFTER บรรทัด)"
else
  echo "FAIL  ชั้น 3 จำไว้ -> ไม่เห็นบรรทัดใหม่ใน fat_usb/kb.txt"
fi

# --- ชั้น 3: ถามข้อใหม่ทันที -> ตอบได้ (guest ตอนนี้เป็น TH) ---
"$PY" "$TYPE" "$SOCK" --start-mode th --th "ดาวพุธมีสีอะไร" --shot /tmp/st4.ppm >/dev/null
t4="$(ocr_shot /tmp/st4.ppm)"
check "ชั้น 3 ถามข้อใหม่ -> ตอบทันที" "สีเหลือง" "$t4"

echo "== เสร็จ (PASS/FAIL ดูด้านบน; ถ้า FAIL เปิด /tmp/st*.png ดูเอง) =="
