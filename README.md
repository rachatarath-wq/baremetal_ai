# Bare-metal AI — a Thai LLM that boots from USB with no OS
# AI บน bare-metal — โมเดลภาษาไทยที่บูตจาก USB โดยไม่มี OS

A small **Thai language model** that runs directly on **UEFI firmware** — no operating
system, no libc, no kernel. The whole thing compiles to a single `BOOTX64.EFI` image that
firmware loads and executes straight from a USB stick. Turn a PC on, and the AI is on
screen before any OS would have started.

**โมเดลภาษาไทยขนาดเล็ก**ที่รันบน **เฟิร์มแวร์ UEFI** โดยตรง — ไม่มีระบบปฏิบัติการ ไม่มี
libc ไม่มีเคอร์เนล ทั้งหมดคอมไพล์เป็นไฟล์เดียว `BOOTX64.EFI` ที่เฟิร์มแวร์โหลดและรันจาก
USB ทันที เปิดเครื่องปุ๊บ AI ก็โผล่บนจอก่อนที่ OS ใด ๆ จะบูตด้วยซ้ำ

It is a from-scratch port of Andrej Karpathy's
[llama2.c](https://github.com/karpathy/llama2.c) `run.c` to a bare-metal UEFI payload.

พอร์ตจาก [llama2.c](https://github.com/karpathy/llama2.c) ของ Andrej Karpathy
(`run.c`) มาเป็น UEFI payload แบบ bare-metal ตั้งแต่ศูนย์

---

## What it can do · ความสามารถ

- 💬 **Chat with a Thai LLM** (llama2.c architecture, Q8_0 quantized, reads its config
  entirely from the model file header — no recompile to swap models).
  **คุยกับ LLM ภาษาไทย** (สถาปัตยกรรม llama2.c, Q8_0 — อ่าน config จาก header ของไฟล์
  โมเดลล้วน ๆ สลับโมเดลได้โดยไม่ต้องคอมไพล์ใหม่)
- ⌨️ **Type Thai** with the Kedmanee keyboard layout (`F1` toggles EN/TH), rendered to the
  framebuffer with combining-mark handling.
  **พิมพ์ภาษาไทย**ด้วยเลย์เอาต์เกษมณี (`F1` สลับ EN/TH) วาดลง framebuffer พร้อมจัดการ
  สระ/วรรณยุกต์ซ้อน
- 🖱️ **A real GUI** — mouse cursor, a chat window, slash commands, and input history.
  **GUI จริง ๆ** — เคอร์เซอร์เมาส์ หน้าต่างแชท slash command และประวัติการพิมพ์
- 🧠 **A 3-layer learning system**: verbatim knowledge-base lookup (`kb.txt`), an
  "I don't know" answer instead of hallucinating, and an on-device `remember:` command
  that survives reboot.
  **ระบบเรียนรู้ 3 ชั้น**: ค้นคำตอบจาก knowledge base (`kb.txt`) แบบคำต่อคำ ตอบ
  "ไม่ทราบ" แทนการมั่ว และคำสั่ง `จำไว้:` บนเครื่องที่จำได้ข้ามการรีบูต
- 🎮 **Five games** drawn pixel-by-pixel to the framebuffer: Pac-Man, Snake, Tic-tac-toe,
  Pong, and Tetris (`/games` picker).
  **เกม 5 เกม**วาดทีละพิกเซลลง framebuffer: Pac-Man, Snake, XO, Pong และ Tetris
  (เมนู `/games`)
- 🔀 **Hot model switching** — `/listmodel` and `/loadmodel` let one USB carry many
  domain models and swap between them at runtime.
  **สลับโมเดลตอนรัน** — `/listmodel` + `/loadmodel` ให้ USB อันเดียวใส่โมเดลโดเมนได้
  หลายตัวและสลับใช้ทันที

---

## Quick start · เริ่มต้นใช้งาน

Requirements: macOS (or Linux) with `x86_64-w64-mingw32-gcc` (mingw-w64) and
[gnu-efi](https://sourceforge.net/projects/gnu-efi/).

ต้องการ: macOS (หรือ Linux) พร้อม `x86_64-w64-mingw32-gcc` (mingw-w64) และ
[gnu-efi](https://sourceforge.net/projects/gnu-efi/)

```bash
make            # builds build/BOOTX64.EFI
make install    # copies it to fat_usb/EFI/BOOT/BOOTX64.EFI
```

Then flash it to a USB drive (formats FAT32, copies the app + model + tokenizer, ejects):

แล้วแฟลชลง USB (ฟอร์แมต FAT32 + copy ตัว app + โมเดล + tokenizer แล้ว eject):

```bash
./create_usb.sh /dev/diskX     # find your USB with: diskutil list
# or, in Claude Code: /createboot /dev/diskX
```

Boot the USB in **UEFI mode** and pick the `UEFI:` entry.
บูต USB ในโหมด **UEFI** แล้วเลือก entry `UEFI:`

> ⚠️ `create_usb.sh` refuses to erase internal/system disks and requires you to type `YES`.
> ⚠️ `create_usb.sh` ปฏิเสธการลบดิสก์ภายใน/ระบบ และต้องพิมพ์ `YES` ยืนยันก่อน

---

## Models · โมเดล

All models are Q8_0-quantized `.bin` files read from the FAT32 root at boot. The default
checkpoint is `stories15M_thai_q80.bin`; use `/loadmodel` to switch.

โมเดลทั้งหมดเป็นไฟล์ `.bin` แบบ Q8_0 ที่อ่านจาก root ของ FAT32 ตอนบูต ตัว default คือ
`stories15M_thai_q80.bin` — ใช้ `/loadmodel` เพื่อสลับ

| File | What | QA accuracy |
|------|------|-------------|
| `stories15M_thai_q80.bin` | Thai factual-QA (general) | 14/14 |
| `stories15M_thai_tourism_q80.bin` | Thai **tourism** domain model | 16/18 |
| `stories15M_q80.bin` | English baseline (llama2.c `stories15M`) | — |

Config (read from the header, no hardcoded dims): `dim=288 · 6 layers · 6 heads ·
vocab=32000 · seq_len=256` ≈ **15M params**.

Config (อ่านจาก header ไม่ hardcode dim): `dim=288 · 6 เลเยอร์ · 6 heads · vocab=32000 ·
seq_len=256` ≈ **15M พารามิเตอร์**

### The "domain model library" approach · แนวคิด "ห้องสมุดโมเดลเฉพาะทาง"

Instead of one big expensive model, the project accumulates **many small 15M
domain-specific models** (tourism, storytelling, math tutor, …) and lets you pick one at
runtime. Each is ~17 MB, so a 1 GB USB holds dozens. See
[`PROTOTYPE.md`](PROTOTYPE.md) for the scaling/budget rationale (15M → 110M → 7B).

แทนที่จะเทรนโมเดลใหญ่ตัวเดียว โปรเจกต์นี้สะสม **โมเดล 15M เฉพาะทางหลายตัว** (ท่องเที่ยว,
นิทาน, สอนคณิต, …) แล้วเลือกใช้ตอนรัน แต่ละตัว ~17 MB → USB 1 GB ใส่ได้หลายสิบตัว
ดูเหตุผลเรื่อง scale/งบใน [`PROTOTYPE.md`](PROTOTYPE.md) (15M → 110M → 7B)

---

## Training your own model · เทรนโมเดลเอง

The full pipeline is documented in [`TRAIN_THAI.md`](TRAIN_THAI.md) (Thai). In short:
pipeline ทั้งหมดอยู่ใน [`TRAIN_THAI.md`](TRAIN_THAI.md) สรุปสั้น ๆ:

```bash
# 1. Build a corpus (streams Thai Wikipedia)
tools/venv/bin/python tools/build_thai_corpus.py --limit-mb 500
# 2. Clean it
tools/venv/bin/python tools/clean_thai_corpus.py thai_corpus_500.txt
# 3. Tokenize -> uint16 train.bin / val.bin
tools/venv/bin/python tools/prep_thai.py
# 4. Train (MPS float32; 15M runs on an 8 GB Mac)
cd llama2c && ../tools/venv/bin/python train.py --device=mps --dtype=float32 ...
# 5. Export to Q8_0 v2 .bin
../tools/venv/bin/python export.py stories15M_thai_new_q80.bin --version 2 --checkpoint out/ckpt.pt
```

A domain model is the same thing plus a curated Q&A facts file (see
`tourism_qa.txt` and `tools/build_tourism_corpus.py`).
โมเดลโดเมนคือของเดียวกันบวกไฟล์ Q&A ข้อเท็จจริงที่คัดมา (ดู `tourism_qa.txt` และ
`tools/build_tourism_corpus.py`)

---

## Project structure · โครงสร้างโปรเจกต์

| Path | Purpose |
|------|---------|
| `run.c` | The entire bare-metal port (freestanding; only `<efi.h>` + `<efilib.h>`) |
| `Makefile` | mingw-w64 + gnu-efi cross-compile recipe |
| `create_usb.sh` | Format + copy + eject a bootable USB |
| `sync_models.sh` | Copy `*_q80.bin` models into `fat_usb/` (auto-run by `create_usb.sh`) |
| `llama2c/` | Training/export code (`train.py`, `export.py`, `model.py`) |
| `tools/` | Corpus builder/cleaner, tokenizer, KB builder, QEMU test harness |
| `fat_usb/` | Staging directory that mirrors the USB layout |
| `kb.txt` | Knowledge-base facts (built from `qa_thai_all.txt` by `tools/build_kb.py`) |

---

## How it works (technical notes) · เบื้องหลังการทำงาน

**No libc.** Every dependency is reimplemented in `run.c`:
**ไม่มี libc** — ทุก dependency เขียนใหม่เองใน `run.c`:

- `expf`/`sqrtf`/`sinf`/`cosf` are hand-written (Taylor series + Newton–Raphson + IEEE-754
  bit tricks); `powf` is removed entirely — the single RoPE frequency becomes
  `expf(-(head_dim/head_size) * ln(10000))`.
  `expf`/`sqrtf`/`sinf`/`cosf` เขียนเอง (อนุกรม Taylor + Newton–Raphson + bit-trick IEEE-754);
  ตัด `powf` ทิ้งทั้งหมด — ความถี่ RoPE ตัวเดียวเปลี่ยนเป็น `expf(-(head_dim/head_size) * ln(10000))`
- **No AVX** — `-mavx`/`-march=native` are forbidden because UEFI never enables `XCR0`,
  so AVX would fault. The image stays on the x86-64 SSE2 baseline.
  **ห้ามใช้ AVX** — เพราะ UEFI ไม่เปิด `XCR0` ทำให้ AVX fault ภาพทั้งหมดจึงอยู่บน baseline SSE2
- File I/O goes through the UEFI `SIMPLE_FILE_SYSTEM_PROTOCOL`; weights are page-allocated
  with `AllocatePages`, the tokenizer with `AllocatePool`.
  อ่านไฟล์ผ่าน `SIMPLE_FILE_SYSTEM_PROTOCOL` ของ UEFI; น้ำหนักโมเดล allocate แบบ page ด้วย
  `AllocatePages` ส่วน tokenizer ใช้ `AllocatePool`
- `efi_main` replaces `main()`; `InitializeLib(...)` is called first.
  `efi_main` แทนที่ `main()` และเรียก `InitializeLib(...)` ก่อนเสมอ

**Cross-compile flags** (all required for a clean mingw + gnu-efi build):
**แฟล็ก cross-compile** (จำเป็นทั้งหมดเพื่อให้ mingw + gnu-efi บิลด์ผ่าน):

```makefile
CC      = x86_64-w64-mingw32-gcc
CFLAGS  = -O3 -Wall -ffreestanding -fno-stack-protector -fno-builtin \
          -DGNU_EFI_USE_MS_ABI -Ignu-efi/inc -Ignu-efi/inc/protocol
LDFLAGS = -nostdlib -Wl,--subsystem,10 -Wl,--entry,efi_main -Wl,--dynamicbase
```

**Gotcha — gnu-efi `Print` format widths:** `%u`/`%d` read 32-bit values; for 64-bit use
`%lu`/`%ld`. `%a` = `CHAR8*`, `%s` = `CHAR16*`, `%r` = `EFI_STATUS`.
**ข้อควรระวัง — ความกว้าง format ของ `Print`:** `%u`/`%d` อ่านค่า 32-bit; ค่า 64-bit ต้องใช้
`%lu`/`%ld` · `%a` = `CHAR8*`, `%s` = `CHAR16*`, `%r` = `EFI_STATUS`

**Verified image:** PE32+ (`0x020b`), subsystem `0x0a` (EFI application), `.reloc` present,
empty import table (no DLL deps).
**ตรวจสอบแล้ว:** PE32+ (`0x020b`), subsystem `0x0a` (EFI application), มี `.reloc`, import
table ว่าง (ไม่พึ่ง DLL)

---

## Documentation · เอกสาร

Some docs are in Thai (the project's working language):
เอกสารส่วนใหญ่เป็นภาษาไทย (ภาษาที่ใช้ทำงานของโปรเจกต์):

| Doc | Language | Covers |
|-----|----------|--------|
| [`USAGE_THAI.md`](USAGE_THAI.md) | 🇹🇭 Thai | Usage, Kedmanee key table, QEMU testing |
| [`GUIDE_THAI.md`](GUIDE_THAI.md) | 🇹🇭 Thai | Overall Thai-language guide |
| [`TRAIN_THAI.md`](TRAIN_THAI.md) | 🇹🇭 Thai | Training pipeline in detail |
| [`COMMANDS_THAI.md`](COMMANDS_THAI.md) | 🇹🇭 Thai | Slash commands |
| [`PROTOTYPE.md`](PROTOTYPE.md) | 🇹🇭 Thai | Status, roadmap, scaling/budget decisions |
| [`PENDING.md`](PENDING.md) | 🇹🇭 Thai | Open tasks / notes |

---

## License · สัญญาอนุญาต

[MIT](LICENSE) — © 2026 rachatarath. Fork it, extend it, ship it.
[MIT](LICENSE) — © 2026 rachatarath. เอาไป fork ต่อยอด ปล่อยต่อได้เลย
