# คู่มือการใช้ Slash Commands — ระบบ Train แบบรวมศูนย์

คู่มือนี้สอนใช้ **คำสั่ง slash** ที่ช่วยรัน pipeline เทรนโมเดลภาษาไทยได้แบบไม่ต้องจำคำสั่งยาว ๆ
เพียงพิมพ์ `/<ชื่อคำสั่ง>` ใน Claude Code ตอนอยู่ที่โฟลเดอร์โปรเจกต์นี้

---

## 1. คำสั่ง slash คืออะไร

คำสั่ง slash (`/train`, `/corpus`, ...) คือ **ทางลัด** ที่เราสร้างไว้ให้ Claude Code ทำงานซ้ำ ๆ ให้
อัตโนมัติ ไฟล์คำสั่งเก็บอยู่ในโฟลเดอร์ `.claude/commands/` (1 ไฟล์ = 1 คำสั่ง)

> เมื่อคุณพิมพ์ `/train` → Claude Code จะอ่านไฟล์ `.claude/commands/train.md`
> แล้วทำตามขั้นตอนที่เขียนไว้ (รันคำสั่ง, ตั้งค่าพารามิเตอร์, ตรวจผล) ให้อัตโนมัติ

---

## 2. รายการคำสั่งทั้งหมด

| คำสั่ง | หน้าที่ | ลำดับ pipeline |
|--------|---------|----------------|
| `/guide` | แสดงคู่มือภาพรวม + รายการคำสั่ง | — |
| `/corpus` | สร้าง corpus ไทยจาก Wikipedia → `thai_corpus.txt` | 1 |
| `/prep` | tokenize → `data/train.bin` + `val.bin` | 2 |
| `/train` | เทรนโมเดลไทย (รันยาวใน background) | 3 |
| `/train-status` | ดูความคืบหน้าของ train ที่รันอยู่ | — |
| `/auto-improve` | วน loop อัตโนมัติ: grade → boost → finetune | — |
| `/export-model` | export checkpoint → Q8_0 v2 | 4 |
| `/build` | คอมไพล์ `run.c` → `BOOTX64.EFI` | — |
| `/deploy` | ชี้ `run.c` ไปที่โมเดล + ลง USB | 5 |
| `/createboot` | ฟอร์แมต USB จริง + ลงไฟล์ boot สำหรับเครื่องจริง | 5 |

**Pipeline เต็ม:** `/corpus` → `/prep` → `/train` → `/export-model` → `/deploy`

---

## 3. วิธีใช้แต่ละคำสั่ง

### `/guide`
แสดงสรุปภาพรวมโปรเจกต์ (จาก `GUIDE_THAI.md`) + รายการคำสั่งทั้งหมด
```
/guide
```

### `/corpus [limit-mb]`
สร้าง corpus ไทยจาก Wikipedia (default 100 MB)
```
/corpus          → 100 MB (default)
/corpus 150      → 150 MB
```

### `/prep`
tokenize `thai_corpus.txt` เป็น `data/train.bin` + `data/val.bin`
```
/prep
```

### `/train [max_iters|resume]`
เทรนโมเดลไทยบน Apple Silicon (MPS) — รันยาวใน background
```
/train           → เทรน 3000 iters (~5 ชม.)
/train 2000      → เทรน 2000 iters
/train resume    → เทรนต่อจาก ckpt เดิม
```
> มีการตั้งค่าอัตโนมัติให้ถูกต้องแล้ว: `batch_size=16` (กัน OOM บน RAM 8 GB),
> `float32` (ปลอดภัยสุดบน MPS), `caffeinate` (กันเครื่องหลับ), checkpoint ทุก 250 iters

### `/train-status`
ดู loss / step / throughput / ETA ของ train ที่รันอยู่
```
/train-status
```

### `/auto-improve`
วน loop อัตโนมัติ: ตรวจข้อเท็จจริง → boost ข้อที่พลาด → finetune → วนใหม่ (แผน A)
```
/auto-improve              → dry-run (โชว์แผน ไม่เทรน)
/auto-improve 5            → รันจริง 5 รอบ
```

### `/export-model`
export `ckpt.pt` → `stories15M_thai_q80.bin` (Q8_0) พร้อมตรวจ header ให้อัตโนมัติ
```
/export-model
```

### `/build`
คอมไพล์ `run.c` → `build/BOOTX64.EFI` + copy ไป `fat_usb/EFI/BOOT/`
```
/build
```

### `/deploy [model-name]`
ชี้ `run.c` ไปที่โมเดลที่เลือก แล้ว build ลง USB
```
/deploy                            → stories15M_thai_q80.bin (default)
/deploy stories15M_q80.bin          → โมเดลเล็กอังกฤษ
/deploy tinyllama_q80.bin           → โมเดลใหญ่
```

### `/createboot [disk]`
ฟอร์แมต USB จริงเป็น FAT32 แล้วลงไฟล์ boot ทั้งหมด (สำหรับบูตเครื่องจริง)
```
/createboot              → ไม่ระบุ disk → โชว์ diskutil list ให้เลือก
/createboot /dev/disk4   → ลง USB ตัว disk4
```
> ต่างจาก `/deploy` ที่แค่ build ลงโฟลเดอร์ `fat_usb/` (ใช้กับ QEMU) — `/createboot`
> จะฟอร์แมต USB จริง + copy ไฟล์ + eject ให้พร้อมบูตเครื่องจริงทันที

---

## 4. ตัวอย่าง workflow เต็ม (เทรนโมเดลไทยตั้งแต่ศูนย์)

```
1. /corpus          # สร้าง corpus ไทย (~100 MB)
2. /prep            # tokenize → data/train.bin + val.bin
3. /train 3000      # เทรน ~5 ชม. (รันใน background)
4. /train-status    # เช็คความคืบหน้าเป็นระยะ
5. /export-model    # export เป็น Q8_0
6. /deploy          # ชี้ run.c + build ลง USB
7. (วางไฟล์โมเดล + tokenizer.bin ที่ root USB แล้วบูต UEFI)
```

---

## 5. วิธีแก้ไข / เพิ่มคำสั่งเอง

1. สร้างไฟล์ใหม่ใน `.claude/commands/` เช่น `.claude/commands/mytest.md`
2. โครงสร้างไฟล์:
   ```markdown
   ---
   description: คำอธิบายสั้น ๆ (โชว์ใน autocomplete)
   argument-hint: "[argument]"
   ---

   ขั้นตอนที่ให้ Claude Code ทำเมื่อเรียกใช้คำสั่งนี้
   ```
3. ชื่อไฟล์ = ชื่อคำสั่ง (`mytest.md` → `/mytest`)

---

## 6. หมายเหตุ

- **`/help` ใช้ชื่อไม่ได้** เพราะเป็นคำสั่ง built-in ของ Claude Code → ใช้ `/guide` แทน
- คำสั่งเป็น **project-scoped** (ใช้ได้เฉพาะตอนอยู่ที่โฟลเดอร์โปรเจกต์นี้)
  ถ้าอยากใช้ทุกโปรเจกต์ → ย้ายไป `~/.claude/commands/`
- ถ้าพิมพ์แล้วไม่ขึ้น autocomplete → รีสตาร์ท session (ปกติ auto-pickup)
- ดูคู่มือภาพรวมโปรเจกต์ได้ที่ [`GUIDE_THAI.md`](GUIDE_THAI.md)
