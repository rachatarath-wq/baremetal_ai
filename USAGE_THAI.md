# คู่มือการใช้งาน — Bare-metal UEFI LLM (โมเดลเล็ก + พิมพ์ภาษาไทย)

คู่มือนี้ครอบคลุม 3 เรื่องที่พึ่งทำเสร็จ:

1. วิธีใช้ **โมเดลเล็ก** `stories15M_q80.bin` (ทดสอบเร็ว, 16 MiB)
2. วิธี**แสดงผล**และ**พิมพ์ภาษาไทย**บนเครื่องจริง/ใน QEMU (เลย์เอาต์ Kedmanee + ปุ่ม F1 สลับภาษา)
3. วิธี**ทดสอบด้วย QEMU** บน macOS (Apple Silicon)

ดูรายละเอียดการ build/install ระดับล่างได้ที่ [`README.md`](README.md)

---

## 1. โมเดลเล็ก `stories15M_q80.bin`

โมเดลเล็กใช้สำหรับทดสอบเร็ว (ไม่ต้องรอโหลด/เจนเนอเรทนานเท่า TinyLlama-1.1B)

| รายการ | ค่า |
|--------|-----|
| ชื่อไฟล์ | `stories15M_q80.bin` |
| พารามิเตอร์ | ~15M |
| ขนาดไฟล์ | ~16 MiB (17,101,696 ไบต์) |
| Config | `dim=288 hidden=768 layers=6 heads=6 kv_heads=6 vocab=32000 seq_len=256` |
| Format | llama2.c **version 2**, Q8_0 (magic `0x616b3432`) |
| Tokenizer | Llama 2 `tokenizer.bin` (vocab 32000) — ใช้ร่วมกับ TinyLlama ได้ |

### เปลี่ยนไปใช้โมเดลเล็ก (ใน `run.c`)

```c
CHAR16 *checkpoint_name = L"\\stories15M_q80.bin";  /* ~บรรทัด 1822 */
CHAR16 *tokenizer_name   = L"\\tokenizer.bin";
int steps = 256;
```

ถ้าจะกลับไปใช้โมเดลใหญ่ ให้เปลี่ยน `checkpoint_name` เป็น `L"\\tinyllama_1.1b_q80.bin"` (หรือชื่อไฟล์จริง) และเพิ่ม `steps`

### ใช้บนเครื่องจริง (USB)

1. ฟอร์แมต USB เป็น FAT32 (GPT + EFI System Partition)
2. วาง `stories15M_q80.bin` และ `tokenizer.bin` ไว้ที่ **root** ของ drive
3. `make install` (copy `BOOTX64.EFI` ไป `EFI/BOOT/`)
4. บูต USB ในโหมด UEFI

**ความเร็วอ้างอิง:** ใน QEMU แบบ TCG บน Apple Silicon ได้ ~14.6 tok/s — บนเครื่องจริงจะเร็วกว่านี้มาก

---

## 2. ภาษาไทย: แสดงผล + พิมพ์ (Kedmanee)

### สถานะการแสดงผล

- ฟอนต์ bitmap **8×16** มีครบทั้ง ASCII (`font8x16_ascii`) และ Thai (`font8x16_thai`)
- รองรับ UTF-8 decode + สระ/วรรณยุกต์ที่เป็น combining marks (U+0E31, U+0E34–U+0E3A, U+0E47–U+0E4E)
  โดยวาดแบบ transparent (ทับซ้อนตัวอักษรหลัก ไม่เลื่อนเคอร์เซอร์)
- ใช้ได้ทันทีกับข้อความภาษาไทยที่ฝังในโค้ด (เช่น banner `>> ระบบแสดงผลภาษาไทยพร้อมใช้งาน … <<`)

### การพิมพ์ภาษาไทยด้วยคีย์บอร์ด

- **ปุ่ม F1** สลับระหว่างโหมด **EN** ↔ **TH** (จะ echo `[EN]` / `[TH]` ให้เห็น)
- ในโหมด TH จะ map แป้น US-QWERTY → ตัวอักษรไทย ตามเลย์เอาต์ **Kedmanee (TIS 820-2538)**
- ผลลัพธ์เก็บเป็น UTF-8 ใน buffer → ส่งเข้า tokenizer ได้ตรง
- Backspace ลบได้ทั้งตัวอักษรไทยหลายไบต์ (ลบครบทั้ง codepoint)

**ตัวอย่าง:** พิมพ์ `wmp` ในโหมด TH → ได้คำว่า **ไทย** (ไ=w, ท=m, ย=p)

### ตาราง mapping Kedmanee (US → ไทย)

ตัวเล็ก (shift ไม่กด) / ตัวใหญ่ (shift กด):

```
      ไม่กด Shift                         กด Shift
  ` → _      1 → ๅ  2 → /  3 → -       ~ → %      ! → +  @ → ๑  # → ๒
  4 → ภ      5 → ถ  6 → ุ  7 → ึ       $ → ๓      % → ๔  ^ → ู  & → ฿
  8 → ค      9 → ต  0 → จ  - → ข       * → ๕      ( → ๖  ) → ๗  _ → ๘
  = → ช                                + → ๙

  q → ๆ      w → ไ  e → ำ  r → พ       Q → ๐      W → "  E → ฎ  R → ฑ
  t → ะ      y → ั  u → ี  i → ร       T → ธ      Y → ํ  U → ๊  I → ณ
  o → น      p → ย  [ → บ  ] → ล       O → ฯ      P → ญ  { → ฐ  } → ,
  \ → ฃ                               | → ฅ

  a → ฟ      s → ห  d → ก  f → ด       A → ฤ      S → ฆ  D → ฏ  F → โ
  g → เ      h → ้  j → ่  k → า       G → ฌ      H → ็  J → ๋  K → ษ
  l → ส      ; → ว  ' → ง             L → ศ      : → ซ  " → .

  z → ผ      x → ป  c → แ  v → อ       Z → (      X → )  C → ฉ  V → ฮ
  b → ิ      n → ื  m → ท  , → ม       B → ฺ      N → ์  M → ?  < → ฒ
  . → ใ      / → ฝ                    > → ฬ      ? → ฦ
```

ตารางเต็มอยู่ใน `gKedmanee[128]` ใน `run.c` (index = ASCII code)

**คำที่ตรวจสอบแล้ว (verify ใน Python):**

| คำ | แป้นที่พิมพ์ | Codepoints |
|----|-------------|------------|
| ภาษา | `4kKk` | ภ(0E20) า(0E32) ษ(0E29) า(0E32) |
| ไทย | `wmp` | ไ(0E44) ท(0E17) ย(0E22) |
| สวัสดี | `l;ylfu` | ส(0E2A) ว(0E27) ั(0E31) ส(0E2A) ด(0E14) ี(0E35) |
| ใช้ | `.=h` | ใ(0E43) ช(0E0A) ้(0E49) |

---

## 3. ทดสอบด้วย QEMU (Apple Silicon / macOS)

### 3.1 จุดสำคัญที่เจอมาแล้ว (ประหยัดเวลา)

- **PS/2 keyboard injection ใช้ไม่ได้กับ OVMF** — HMP `sendkey` และ QMP `input-send-event`
  กดแล้ว app มองไม่เห็น (ค้างที่ "Enter prompt") ต้องใช้ **q35 + USB keyboard** แทน
- บน `-machine q35` drive ต้องต่อผ่าน **AHCI** อย่างชัดเจน (ไม่มี PIIX)
- QEMU Homebrew build นี้ `screendump` ออกเป็น **PPM** (ไม่มี libpng) → แปลงด้วย `sips`
- OCR ต้อง crop+zoom ตัวอักษร 2–3× ก่อน ไม่งั้นได้ผลลัพธ์ว่าง

### 3.2 คำสั่ง launch (ทำงานแล้ว)

```bash
qemu-system-x86_64 \
  -machine q35 -m 1G -net none -bios OVMF.fd \
  -device ich9-ahci,id=ahci \
  -drive file=fat:rw:/tmp/baremetal_test,format=raw,if=none,id=disk0 \
  -device ide-hd,drive=disk0,bus=ahci.0 \
  -device qemu-xhci,id=xhci \
  -device usb-kbd,bus=xhci.0 \
  -monitor tcp:127.0.0.1:45454,server,nowait \
  -qmp tcp:127.0.0.1:45455,server,nowait \
  -vnc 127.0.0.1:1
```

- `/tmp/baremetal_test/` ต้องมี `EFI/BOOT/BOOTX64.EFI`, `stories15M_q80.bin`, `tokenizer.bin`
- monitor ที่พอร์ต 45454, QMP ที่ 45455, VNC display `:1`

### 3.3 Inject คีย์ผ่าน QMP (พิมพ์ภาษาไทยอัตโนมัติ)

Schema ของ QEMU 11.x — `key` เป็น **union object** ไม่ใช่ string เปล่า:

```json
{"execute":"input-send-event","arguments":{"events":[
  {"type":"key","data":{"key":{"type":"qcode","data":"f1"},"down":true}},
  {"type":"key","data":{"key":{"type":"qcode","data":"f1"},"down":false}}
]}}
```

ขั้นตอน: กด `f1` (สลับเป็น TH) → กด `w` `m` `p` → จะเห็นคำว่า **ไทย** บนจอ

### 3.4 จับภาพ + ตรวจผล

```bash
# จับภาพ (PPM)
echo 'screendump /tmp/shot.ppm' | nc 127.0.0.1 45454
# แปลงเป็น PNG
sips -s format png /tmp/shot.ppm --out /tmp/shot.png
# OCR (Swift Vision) — crop+zoom บริเวณข้อความก่อน
swift /tmp/ocr.swift
```

**ผลตรวจจริง (2026-11-04):** พิมพ์ `wmp` ในโหมด TH แล้ว template-match glyph บน framebuffer
เทียบกับ bitmap ฟอนต์ ได้ตรง **ไ(464,112) ท(472,112) ย(480,112)** เป๊ะ — ยืนยันว่า input ภาษาไทย
ทำงานครบวงจร (toggle → mapping → UTF-8 → render)

---

## 4. ขยายตัวอักษร (font scaling)

ขยายตัวอักษรด้วย **scale factor** — ฟอนต์ bitmap ยังเป็น 8×16 เหมือนเดิม แต่ตอนวาดจะขยาย
แต่ละพิกเซลเป็นบล็อก `FONT_SCALE × FONT_SCALE` (nearest-neighbor)

```c
#define FONT_SCALE 2              /* 1 = 8x16, 2 = 16x32, 3 = 24x48, ... */
#define CELL_W (FONT_W * FONT_SCALE)
#define CELL_H (FONT_H * FONT_SCALE)
```

- `FONT_SCALE = 2` → เซลล์ 16×32 (ค่าเริ่มต้นปัจจุบัน)
- อยากใหญ่/เล็กกว่านี้ แก้เลขใน `#define FONT_SCALE` ที่เดียว แล้ว `make` ใหม่
- กระทบทุกจุดอัตโนมัติ: `DrawGlyph`, `DrawGlyphTransparent` (สระ/วรรณยุกต์), scroll,
  backspace, และจำนวนคอลัมน์/แถว (`gCols`/`gRows`)

**ตรวจแล้ว (2026-11-04):** บนจอ 2048×2048 pitch ระหว่างบรรทัด = 32px (จากเดิม 16px)
ยืนยันว่า 2× ทำงานถูกต้อง

---

## 5. หมายเหตุ

- `README.md` เขียนไว้ตอนใช้ `stories15M.bin` + "ASCII-only" — ไฟล์นี้เป็นคู่มือปัจจุบัน
