# คู่มือทดสอบระบบเรียนรู้ (LEARNING_SYSTEM_TEST.md)

ขั้นตอนตรวจสอบว่า AI ภาษาไทยบน bare-metal **ตอบข้อเท็จจริงถูก ไม่มั่ว และ "จำไว้" ได้จริง** ตามแผน `LEARNING_SYSTEM_PLAN.md` (ชั้น 1/2/3)

> เขียน 2026-11-10 — สำหรับ branch `feature/thai-qa-corpus` · อ่านคู่กับ `LEARNING_SYSTEM_PLAN.md`
> ตัวโมเดล (llama2.c 15M) ตอนนี้ถูก **ปิด generate** แล้ว = โหมด Q&A ล้วน (KB + ไม่ทราบ + จำไว้)

---

## ภาพรวม 3 ชั้นที่ต้องเทส

| ชั้น | พฤติกรรม | ตัวชี้วัด "ผ่าน" |
|---|---|---|
| **1. KB lookup** | คำถามที่อยู่ใน `kb.txt` → ตอบ verbatim (ไม่ generate) | ตอบตรงกับไฟล์ `kb.txt` |
| **2. router + "ไม่ทราบ"** | คำถามที่ไม่อยู่ใน KB → ตอบ `ไม่ทราบ` (ไม่เดามั่ว) | เห็นคำว่า `ไม่ทราบ` |
| **3. "จำไว้:"** | `จำไว้: คำถาม = คำตอบ` → append ลง `kb.txt` + จำได้ทันที + อยู่ข้าม reboot | ไฟล์ `fat_usb/kb.txt` เพิ่มบรรทัด + ถามซ้ำตอบได้ |

---

## 0. สิ่งที่ต้องมี

- ระบบ macOS + Homebrew QEMU (`qemu-system-x86_64`), Swift (Vision OCR)
- ไฟล์ใน `fat_usb/`: `EFI/BOOT/BOOTX64.EFI`, `stories15M_thai_q80.bin`, `tokenizer.bin`, `kb.txt`
- QEMU ต้องรันด้วย **q35 + USB keyboard** (`-device qemu-xhci -device usb-kbd`) — PS/2 ส่งคีย์ไม่ถึง OVMF ConIn (ดู `USAGE_THAI.md` §3.1)

---

## 1. ตั้งค่าครั้งเดียว (build + deploy)

```bash
cd /Volumes/backup2568/project/baremetal_ai
make install                                          # build run.c → fat_usb/EFI/BOOT/BOOTX64.EFI
tools/venv/bin/python tools/build_kb.py               # สร้าง kb.txt จาก qa_thai_all.txt
cp kb.txt fat_usb/kb.txt                              # เอา kb.txt ลง volume ที่ QEMU อ่าน
```

- `make install` ต้องรันใหม่ **เฉพาะตอนแก้ `run.c`** (ตอนแก้ข้อมูลใน `qa_thai_all.txt` แค่รัน `build_kb.py` + copy `kb.txt` ใหม่ ก็พอ ไม่ต้อง rebuild)

---

## 2. เปิด QEMU

มี 2 โหมด:

### 2.1 Headless (ใช้กับ smoke test / QMP อัตโนมัติ)

```bash
make qemu-headless     # terminal A — ไม่มีหน้าต่าง, เปิด QMP socket /tmp/qemu_bm.sock
```

- รอ **~50 วินาที** ให้โหลดโมเดล 17MB จนเห็น `Enter prompt [EN] (F1 toggles EN/TH)` (ดูผ่าน screendump ได้)
- QEMU จะอยู่ foreground — เปิด terminal B ไว้พิมพ์คำสั่งเทส

### 2.2 Interactive (มีหน้าต่าง, พิมพ์เอง)

```bash
make qemu             # เปิดหน้าต่าง, พิมพ์คีย์ตรงเข้า guest ได้
```

- กด **F1** สลับโหมด EN ↔ TH; พิมพ์ภาษาไทยโดยกดคีย์ US ตามผัง Kedmanee (หรือใช้ `qemu_type.py` ตาม §3)

---

## 3. ทดสอบอัตโนมัติ (รันคำสั่งเดียว)

หลังจาก QEMU boot เสร็จ (`Enter prompt [EN]`) รัน:

```bash
./tools/smoke_test.sh          # ใช้ socket เริ่มต้น /tmp/qemu_bm.sock
```

มันจะพิมพ์ `PASS`/`FAIL` 4 ข้อ:

```
== smoke test กับ /tmp/qemu_bm.sock ==
PASS  ชั้น 1 รู้จัก -> ตอบ verbatim
PASS  ชั้น 2 ไม่รู้ -> 'ไม่ทราบ'
PASS  ชั้น 3 จำไว้ -> append ลง kb.txt (225 -> 226 บรรทัด)
PASS  ชั้น 3 ถามข้อใหม่ -> ตอบทันที
```

> สคริปต์ตรวจชั้น 3 ด้วยการ grep ไฟล์ `fat_usb/kb.txt` จริง (ไม่ผ่าน OCR) จึงเชื่อถือได้ 100%
> ชั้น 1/2/4 ตรวจผ่าน OCR — ถ้าขึ้น `FAIL` ให้เปิด `/tmp/st*.png` ดูเองก่อนสรุป (OCR ไทยเพี้ยนได้ ดู §6)

---

## 4. ทดสอบทีละขั้น (manual — ดูผลเอง)

หลักการ: พิมพ์คำถาม → จับภาพ → อ่านข้อความจากภาพ

**อ่านภาพ** (ใช้ซ้ำทุกขั้น):

```bash
sips -s format png /tmp/shot.ppm --out /tmp/shot.png >/dev/null 2>&1
swift tools/ocr.swift /tmp/shot.png
```

### 4.1 ชั้น 1 — รู้จัก → ตอบจาก KB

```bash
tools/venv/bin/python tools/qemu_type.py /tmp/qemu_bm.sock \
  --start-mode en --th "พ่อเราชื่ออะไร" --shot /tmp/shot.ppm
```

**คาดหวัง**: `ผู้พัฒนา , contact@example.com` (ข้อความเดียวกับ `kb.txt`)

### 4.2 ชั้น 2 — ไม่รู้ → ตอบ "ไม่ทราบ"

```bash
tools/venv/bin/python tools/qemu_type.py /tmp/qemu_bm.sock \
  --start-mode th --th "ทำไมท้องฟ้าถึงเป็นสีฟ้า" --shot /tmp/shot.ppm
```

**คาดหวัง**: `ไม่ทราบ` (ไม่ generate ข้อความเดา)

### 4.3 ชั้น 3 — "จำไว้:" สอนข้อใหม่

```bash
tools/venv/bin/python tools/qemu_type.py /tmp/qemu_bm.sock --start-mode th \
  --segments "th:จำไว้,en:: ,th:ดาวพุธมีสีอะไร,en: = ,th:สีเหลือง" \
  --shot /tmp/shot.ppm
```

**คาดหวัง**: `จำไว้แล้ว: สีเหลือง` และ `fat_usb/kb.txt` มีบรรทัดใหม่ `ดาวพุธมีสีอะไร<TAB>สีเหลือง`

> ทำไมต้อง `--segments`: `:` และ `=` ในโหมด TH ถูกแมปเป็น ซ/ช ตามผัง Kedmanee
> จึงต้องสลับเป็นโหมด EN ตอนพิมพ์ `:` และ `=` (เครื่องมือสลับ F1 ให้อัตโนมัติ)

### 4.4 ชั้น 3 — ถามข้อที่เพิ่งสอน (จำได้ทันที ไม่ต้องเทรน)

```bash
tools/venv/bin/python tools/qemu_type.py /tmp/qemu_bm.sock \
  --start-mode th --th "ดาวพุธมีสีอะไร" --shot /tmp/shot.ppm
```

**คาดหวัง**: `สีเหลือง`

---

## 5. ทดสอบ "จำได้ข้าม reboot"

ข้อที่ "จำไว้" ถูก append ลง `fat_usb/kb.txt` (FAT จริง) จึงควรอยู่รอดการบูตใหม่:

```bash
# terminal A: หยุด QEMU แล้วเปิดใหม่ (จำลอง reboot)
pkill -f qemu-system-x86_64
make qemu-headless        # รอ ~50 วิ
```

```bash
# terminal B: ถามข้อที่เคยสอนไว้
tools/venv/bin/python tools/qemu_type.py /tmp/qemu_bm.sock \
  --start-mode en --th "ดาวพุธมีสีอะไร" --shot /tmp/shot.ppm
```

**คาดหวัง**:
- จอบูตแสดง `kb: 226 entries loaded` (ไม่ใช่ 225) = ข้อใหม่ถูกอ่านกลับมาจาก flash แล้ว
- ถามแล้วตอบ `สีเหลือง`

---

## 6. ข้อควรรู้ / แก้ปัญหา

| อาการ | สาเหตุ / วิธีแก้ |
|---|---|
| OCR อ่านเพี้ยน (`จำ`→`จ่า`, `ด`→`ต`, `สีเหลือง`→`สีเ หลือง`) | ปกติสำหรับ OCR ไทย — เปิด `/tmp/*.png` ดูเอง; ตัวตรวจที่เชื่อถือได้คือ grep ไฟล์ `kb.txt` |
| พิมพ์แล้วได้ตัวอักษรผิด (ธ→ะ, `:`→`;`) | คีย์ที่ต้องกด shift (ธ ็ ฯ) ต้อง "กด shift ค้าง" ตอนกดคีย์ — `qemu_type.py` เวอร์ชันปัจจุบันทำถูกแล้ว |
| QEMU หลุดเงียบ ๆ (exit 0) ตอนเทส | `-no-reboot` ทำให้ guest reset = exit; ถ้าสงสัย crash ให้เปิดใหม่ด้วย `-d cpu_reset` แล้วดูว่า boot ปกติมีแค่ 2 `CPU Reset` block |
| `--start-mode` ผิด (พิมพ์ไทยแต่ได้อังกฤษ/สลับ) | guest จำโหมด EN/TH ค้างไว้ข้ามคำสั่ง — หลัง boot ใช้ `en`; หลังพิมพ์ไทยไปแล้วใช้ `th` |
| ข้อที่ "จำไว้" ค้างใน `fat_usb/kb.txt` | มันไม่ใช่ source (`qa_thai_all.txt`) — reset ได้ด้วย `cp kb.txt fat_usb/kb.txt` |
| กด Enter เปล่า = จบโปรแกรม | app return → guest reset (ใน `-no-reboot` จะทำให้ QEMU exit) — ไม่ใช่ crash |

---

## 7. ผลการเทสจริง (อ้างอิง 2026-11-10)

รันบน QEMU q35 + usb-kbd, boot ~50 วิ, โมเดล `stories15M_thai_q80.bin`:

```
kb: 225 entries loaded (41357 bytes)      ← บูตแรก
Enter prompt [EN] ... [TH]พ่อเราชื่ออะไร
ผู้พัฒนา , contact@example.com  ← ชั้น 1 ✓
Enter prompt [TH] ... [TH]ทำไมท้องฟ้าถึงเป็นสีฟ้า
ไม่ทราบ                                   ← ชั้น 2 ✓
Enter prompt [TH] ... จำไว้: ดาวพุธมีสีอะไร = สีเหลือง
จำไว้แล้ว: สีเหลือง                        ← ชั้น 3 ✓
Enter prompt [TH] ... [TH]ดาวพุธมีสีอะไร
สีเหลือง                                  ← ถามซ้ำทันที ✓

(reboot)
kb: 226 entries loaded (41425 bytes)      ← ข้อใหม่ถูกอ่านกลับมาจาก flash ✓
Enter prompt [EN] ... [TH]ดาวพุธมีสีอะไร
สีเหลือง                                  ← อยู่ข้าม reboot ✓
```
