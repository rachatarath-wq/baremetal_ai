# ขั้นตอนทดสอบโมเดลไทย (TEST_THAI.md)

เอกสารขั้นตอนตรวจสอบว่าโมเดลที่เทรนใหม่ (clean corpus + ชุด Q&A) **ตอบคำถามข้อเท็จจริงภาษาไทยได้ถูกต้อง** ทั้งแบบเร็ว (Python) และแบบจริง (bare-metal QEMU)

> เขียน 2026-11-09 — สำหรับ branch `feature/thai-qa-corpus` · อ่านคู่กับ `TRAINING_PLAN.md`

---

## ภาพรวม pipeline

```
corpus → prep → train ──▶ export → install → QEMU → ถาม
                          (ทางลัด: check_qa.py โหลด ckpt.pt ตรง ๆ)
```

---

## 0. สิ่งที่ต้องมีก่อน

- เทรนเสร็จแล้ว checkpoint: `llama2c/out_thai_v2/ckpt.pt`
- Tokenizer: `tokenizer.model` (ใช้ตอน export เป็น `tokenizer.bin`)
- โมเดลเป้าหมายบน FAT: `fat_usb/stories15M_thai_q80.bin` (ชื่อที่ `run.c` อ่านตอนบูต)

---

## 1. Export checkpoint → โมเดล Q8

```bash
cd /Volumes/backup2568/project/baremetal_ai/llama2c
../tools/venv/bin/python export.py ../stories15M_thai_q80.bin --version 2 --checkpoint out_thai_v2/ckpt.pt
```

**ตรวจผล**: magic `0x616b3432` version 2, dim=288 hidden=768 layers=6 heads=6
vocab=32000 seq=256, shared_classifier=1, group_size=32, ขนาดไฟล์ **17,101,696 bytes**

---

## 2. ใส่โมเดลใหม่ลง fat_usb

```bash
cd /Volumes/backup2568/project/baremetal_ai
cp stories15M_thai_q80.bin fat_usb/stories15M_thai_q80.bin
```

> ไม่ต้อง rebuild `BOOTX64.EFI` — `run.c` อ่านโมเดลจากชื่อไฟล์ (`\stories15M_thai_q80.bin`) ตอนบูต
> ถ้าแก้ `run.c` / `hrfont.h` / `font.h` ค่อย `make && make install`

---

## 3. ทดสอบบน QEMU (interactive — ตัวจริง)

```bash
make qemu        # เปิดหน้าต่าง
```

1. รอ OVMF บูต → เห็นแบนเนอร์ `>> ระบบแสดงผลภาษาไทยพร้อมใช้งาน: สวัสดี Bare-Metal AI <<`
2. ที่ prompt `Enter prompt [EN]`: กด **F1** สลับเป็นโหมด **TH**
3. พิมพ์คำถาม (หรือใช้ `คำถาม: ...` นำหน้า) แล้ว Enter
4. ดูคำตอบที่ model generate ออกมา (empty line = จบโปรแกรม)

---

## 4. ทดสอบ headless + จับภาพ (ไม่มีหน้าต่าง)

```bash
# terminal A — รัน QEMU headless (foreground)
make qemu-headless

# terminal B — จับ framebuffer
python3 tools/screendump.py /tmp/qemu_bm.sock /tmp/shot.ppm
python3 -c "from PIL import Image; Image.open('/tmp/shot.ppm').convert('RGB').save('/tmp/shot.png')"
```

> ใช้สำหรับเช็ค**การเรนเดอร์** (แบนเนอร์/ฟอนต์) — การป้อนคำถามไทยผ่านคีย์บอร์ดในโหมด headless ทำได้ยาก
> ถ้าอยากเทสต์ Q&A แบบ headless อัตโนมัติ: hardcode คำถามใน `run.c` ชั่วคราวแล้ว `make && make install`

---

## 5. ชุดคำถามทดสอบ (สุ่มจาก 14 ข้อใน `qa_thai.txt`)

| พิมพ์ | ควรตอบ |
|---|---|
| ภาษาไทยมีพยัญชนะกี่ตัว | ภาษาไทยมีพยัญชนะ 44 ตัว |
| ภาษาไทยมีตัวอักษรทั้งหมดกี่ตัว | ภาษาไทยมีพยัญชนะ 44 ตัว สระ 21 รูป และวรรณยุกต์ 4 รูป รวมประมาณ 69 ตัวอักษร |
| ภาษาไทยมีสระกี่รูป | ภาษาไทยมีสระ 21 รูป |
| ภาษาไทยมีสระกี่เสียง | ภาษาไทยมีสระ 32 เสียง |
| ภาษาไทยมีวรรณยุกต์กี่รูป | ภาษาไทยมีวรรณยุกต์ 4 รูป |
| ภาษาไทยมีเสียงวรรณยุกต์กี่เสียง | ภาษาไทยมีเสียงวรรณยุกต์ 5 เสียง |
| พยัญชนะไทยตัวแรกคือตัวอะไร | พยัญชนะไทยตัวแรกคือ ก ไก่ |
| ภาษาไทยจัดอยู่ในตระกูลภาษาใด | ภาษาไทยจัดอยู่ในตระกูลภาษาไท-กะได |
| เมืองหลวงของประเทศไทยคือที่ไหน | เมืองหลวงของประเทศไทยคือกรุงเทพมหานคร |
| ประเทศไทยมีกี่จังหวัด | ประเทศไทยมี 77 จังหวัด |

---

## 6. ทางลัด: เช็คเร็วด้วย Python (ไม่ต้องรอ QEMU)

โหลด `ckpt.pt` ตรง ๆ แล้ว generate ทุกคำถามใน `qa_thai.txt` (ข้อแรกของแต่ละ fact) — เร็วกว่ามาก เหมาะเช็คระหว่างรอ/ก่อน export:

```bash
tools/venv/bin/python tools/check_qa.py \
  --checkpoint llama2c/out_thai_v2/ckpt.pt \
  --device cpu --max-new-tokens 40
```

ตัวอย่างผลจาก checkpoint **เก่า** (iter 1250, เทรน prose ล้วน — ก่อนเติม Q&A):
```
ถาม: ภาษาไทยมีพยัญชนะกี่ตัว
ตอบ: เป็นภาษาที่มีความหมายต่าง ๆ
```
หลังเทรนใหม่คาดหวัง:
```
ถาม: ภาษาไทยมีพยัญชนะกี่ตัว
ตอบ: ภาษาไทยมีพยัญชนะ 44 ตัว
```

---

## จุดที่ต้องรู้ / gotchas

- **ป้อนคีย์ไทยเข้า QEMU**: ใช้ `make qemu` (q35 + usb-kbd) พิมพ์มือ; PS/2 key ไม่ถึง OVMF ConIn
- **รูปแบบ prompt**: เทรนไว้ 3 แบบ — `คำถาม: Q\nคำตอบ: A`, `ถาม: Q\nตอบ: A`, และแบบเปล่า `Q\nA` → พิมพ์แบบไหนก็ควรตอบได้
- **บน M3**: QEMU ใช้ TCG (ไม่มี HVF สำหรับ x86_64), vvfat ต้อง `rw` (ตรงตาม Makefile แล้ว)
- **ขนาดโมเดล**: 15M Q8 = 17MB, เทรน fp32 ~60MB — โหลดเข้า RAM 512MB ใน QEMU ได้สบาย
