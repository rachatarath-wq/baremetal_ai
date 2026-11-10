# สถานะงาน + งานค้าง (PENDING.md)

> อัปเดต 2026-11-10 — เก็บไว้สำหรับ resume ต่อ · branch `feature/thai-qa-corpus`
> อ่านคู่กับ `LEARNING_SYSTEM_PLAN.md` (แผน) และ `LEARNING_SYSTEM_TEST.md` (วิธีเทส)

---

## สถานะปัจจุบัน (เสร็จแล้ว ✓)

| งาน | สถานะ | commit |
|---|---|---|
| แผนระบบเรียนรู้ 3 ชั้น | ✓ | `1281750` |
| ชั้น 1: KB lookup (`kb.txt` → ตอบ verbatim) | ✓ | `e3fbf3d` |
| ชั้น 2: router + "ไม่ทราบ" (ไม่ generate) | ✓ | `6bc3296` |
| ชั้น 3: "จำไว้:" (append + จำทันที + ข้าม reboot) | ✓ | `6bc3296` |
| เครื่องมือเทส `qemu_type.py` / `ppm_crop.py` | ✓ | `6bc3296` |
| คู่มือเทส + smoke test + OCR | ✓ | `e67dca5` |
| เทส QEMU ครบ (4 ข้อ + persistence) | ✓ | — |

ผลเทสจริง: ชั้น 1 ตอบ `คุณรชต รัตนวสวัต , rachata.rath@gmail.com` · ชั้น 2 ตอบ `ไม่ทราบ`
· ชั้น 3 สอน `ดาวพุธมีสีอะไร = สีเหลือง` → ถามซ้ำได้ → reboot แล้ว `kb: 225 → 226 entries` ยังตอบได้

---

## งานค้าง (ยังไม่ทำ)

### 1. `qa_thai_all.txt` ยังไม่ commit ⚠️
การแก้ของคุณเอง (ยังไม่ commit): เพิ่มอีเมล `rachata.rath@gmail.com` ในคำตอบ + เพิ่มคำถาม `ติดต่อคนคิด`

```diff
 คำถาม: คนคิดเอไอภาษาไทยชื่ออะไร
 คำถาม: พ่อเราชื่ออะไร
-คำตอบ: คุณรชต รัตนวสวัต
+คำถาม: ติดต่อคนคิด
+คำตอบ: คุณรชต รัตนวสวัต , rachata.rath@gmail.com
```

- **ต้องตัดสินใจ**: commit ไหม (เป็นข้อมูลส่วนตัวของคุณเอง)
- ถ้า commit → ต้องรัน `tools/venv/bin/python tools/build_kb.py && cp kb.txt fat_usb/kb.txt` ใหม่ (kb.txt เปลี่ยนแล้ว)

### 2. ตัดสินใจเรื่อง `generate()` — ตอนนี้ปิดอยู่
ตัวโมเดล 17MB **ยังโหลดตอนบูต (~40 วิ)** แต่ `generate()` ไม่ถูกเรียก = เสียเวลาโหลดโดยเปล่า
ทางเลือก (ดู `LEARNING_SYSTEM_PLAN.md` §3):

| ทางเลือก | ข้อดี | ข้อเสีย |
|---|---|---|
| (ก) คง Q&A ล้วน (สถานะปัจจุบัน) | เรียบง่าย ไม่มั่ว | โหลดโมเดลเปล่า ~40 วิ |
| (ข) เปิด generate + ขึ้นต้น "ไม่แน่ใจ: …" | ได้คำตอบแม้ไม่อยู่ใน KB | ต้องแก้ `run.c` กลับไปเรียก generate |
| (ค) ไม่โหลดโมเดลเลย (KB-only) | บูตไวขึ้นมาก | เลิกใช้โมเดลถาวร |

### 3. แผนขั้น 4 — consolidate (นับความถี่ → host `/auto-improve`)
- ข้อที่ถูกถาม/ถูกสอน **บ่อย** → ส่งกลับ host → `/auto-improve` finetune ยัดลง weights
- ยังไม่เริ่ม · อยู่ในฝั่ง host (Mac) · ความยาก: ง่าย

### 4. แผนขั้น 5 — RAG ระยะไกล (optional, ข้ามได้)
- network stack + query ฐานความรู้/API แทน KB local · ยังไม่เริ่ม · ความยาก: ยาก

### 5. จุกจิก / ข้อสังเกต
- **`-no-reboot`** ทำให้ guest reset = QEMU exit (กด Enter เปล่า = จบโปรแกรม = reset = exit) — ไม่ใช่ crash (ยืนยันด้วย `-d cpu_reset` แล้ว)
- **`smoke_test.sh` ทิ้ง fact ทดสอบ** `ดาวพุธมีสีอะไร` ไว้ใน `fat_usb/kb.txt` → reset ด้วย `cp kb.txt fat_usb/kb.txt`
- **OCR ไทยเพี้ยนได้** (`จำ`→`จ่า`, `สีเหลือง`→`สีเ หลือง`) — ตรวจขั้น 3 ด้วย grep ไฟล์ `kb.txt` ถึงจะเชื่อถือได้

---

## วิธี resume

```bash
cd /Volumes/backup2568/project/baremetal_ai
make install                                          # ถ้าแก้ run.c
tools/venv/bin/python tools/build_kb.py && cp kb.txt fat_usb/kb.txt   # ถ้าแก้ qa_thai_all.txt
make qemu-headless                                     # terminal A (รอ ~50 วิ)
./tools/smoke_test.sh                                  # terminal B — เทส 4 ข้อ
```

- เริ่มจากข้อ **1** (commit `qa_thai_all.txt`?) และ **2** (ตัดสินใจ `generate()`) ก่อน
- ข้อ 3–4 เป็นของที่ทำต่อได้หลังจากนั้น
