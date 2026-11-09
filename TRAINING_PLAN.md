# แผนทำให้โมเดลฉลาดขึ้น (resume later)

บันทึกสถานะ + แผนแก้ปัญหา "โมเดลตอบคำถาม factual ผิด" — เขียน 2026-11-09

---

## 1. สถานะที่ทำเสร็จแล้ว (done)

- ✅ ฟอนต์ high-res ไทย (`font.h` + `hrfont.h`) ต่อเป็น renderer ตัวจริงแล้ว
- ✅ Thai combining marks (สระบน/ล่าง/วรรณยุกต์) ซ้อนถูกต้อง — ยืนยันแล้วทั้งแบบตรวจพิกเซล + QEMU
- ✅ QEMU smoke test ผ่าน: บูต OVMF → โหลดโมเดล → เรนเดอร์แบนเนอร์ไทยถูก
- ✅ Makefile มี target ใหม่: `make qemu` (หน้าต่าง), `make qemu-headless` (headless + QMP)
- ✅ `tools/screendump.py` — ตัวจับภาพ QMP แบบถาวร

---

## 2. ปัญหาที่พบ

ทดสอบถาม **"ภาษาไทยมีกี่ตัวอักษร"** → โมเดลตอบ **"เป็นการปกครองของพระองค์"** (ผิด)

คำตอบที่ถูกควรเป็น: **พยัญชนะ 44 ตัว + สระ 21 รูป (32 เสียง) + วรรณยุกต์ 4 รูป = ~69 ตัว**

---

## 3. วินิจฉัยสาเหตุ (ไม่ใช่ bug)

โมเดลคือ llama2.c **15M พารามิเตอร์** เทรนบน Thai Wikipedia **prose** แค่ **3000 iters (~197M token)**

| ปัจจัย | ค่าปัจจุบัน | ปัญหา |
|--------|-------------|-------|
| ขนาดโมเดล | `dim=288 layers=6 heads=6 seq=256` (~17M param) | เล็กเกินจะ "จำ" ข้อเท็จจริง |
| ข้อมูล | `thai_corpus.txt` 104MB (Wikipedia ล้วน) | เป็น prose + navigation boilerplate ไม่ใช่ Q&A |
| จำนวนเทรน | 3000 iters | น้อยมาก (~ไม่กี่ epoch) |
| hyperparam | `min_lr=0`, ไม่มี warmup | ต่ำกว่ามาตรฐาน |

หัวใจ: **Generative LLM = ตัวทำนาย token ถัดไป ไม่ใช่ฐานข้อมูล** — คำถาม factual ต้องมี pattern "คำถาม→คำตอบ" ซ้ำ ๆ ใน training data ถึงจะตอบถูก

---

## 4. คันโยกทำให้ฉลาดขึ้น (เรียงตาม impact)

### 1) ขยายโมเดล (impact สูงสุด, แพงสุด)
```
--dim=768 --n_layers=12 --n_heads=12 --n_kv_heads=12   # 15M -> 110M
```
- Q8: 17MB → ~110MB | เทรน fp32: ~60MB → ~440MB
- ⚠️ ต้องเช็ค RAM ของเครื่อง bare-metal ปลายทางก่อน

### 2) ล้าง/เปลี่ยนข้อมูล (ถูกสุด, ได้ผลชัดกับคำถามเฉพาะ)
- ตัด boilerplate: `หน้าหลัก`, `ป้ายบอกทาง`, `ศาลาประชาคม`, `กระดานข่าว`, `ปุจฉา-วิสัชนา`, `Local Embassy` …
- เพิ่มข้อมูล **Q&A / FAQ / หนังสือเรียน / สารานุกรมไทย**
- ข้อเท็จจริงที่อยากให้ตอบ ต้องมีประโยคนั้น **ซ้ำหลายครั้ง** ใน data

### 3) เทรนนานขึ้น
```
--max_iters=3000  ->  50000..100000+   # หลาย epoch
```

### 4) แก้ hyperparameters
```
min_lr        = learning_rate / 10   # (~5e-5) แทน 0
warmup_iters  = 2000
```

---

## 5. ข้อควรรู้

- LLM generative **hallucinate ได้เสมอ** กับคำถาม factual ที่ data ไม่มี
- อยากได้ความรู้เฉพาะชุด (เช่น เรื่องภาษาไทย 10–20 ข้อ) → โมเดล 15M ก็พอ ถ้าเทรนบน Q&A นั้นซ้ำ ๆ (คือการจำ pattern)
- อยากได้ "ฉลาดทั่วไป" → ต้อง 100M–1B+ เกินกำลัง bare-metal เล็ก → ทางออกจริงคือ **RAG / lookup**

---

## 6. ตัวเลือกถัดไป (ยังไม่ได้ทำ — รอตัดสินใจ)

1. **ล้าง corpus + เติม Q&A ไทย** (เร็ว ถูก เห็นผลเฉพาะคำถามที่เตรียมไว้)
2. **ขยับเป็น 110M** แล้วเทรนใหม่ (ต้องเช็ค RAM target ก่อน)
3. **ปรับ hyperparams + เทรนนานขึ้น** บน config เดิม

---

## 7. ข้อมูลอ้างอิง (ใช้ resume)

### ไฟล์/พาธสำคัญ
- โมเดล: `stories15M_thai_q80.bin` (17MB, Q8_0) — checkpoint_name ใน `run.c`
- Tokenizer: `tokenizer.bin` (433KB, SentencePiece 32000 vocab)
- Corpus: `thai_corpus.txt` (104MB, ~496k บรรทัด, Thai Wikipedia)
- Train script: `tools/llama2.c/train.py` (+ `configurator.py`)
- Tokenizer script: `tools/llama2.c/tokenizer.py` (SentencePiece)
- Build corpus: `tools/build_thai_corpus.py`, `tools/prep_thai.py`

### Model config (ใน train.py, บรรทัด ~48–62)
```
batch_size = 128          # micro-batch (ตอนรันจริงใช้ 16)
max_seq_len = 256
vocab_size = 32000
dim = 288
n_layers = 6
n_heads = 6
n_kv_heads = 6
dropout = 0.0
learning_rate = 5e-4
max_iters = 100000       # default (ตอนรันจริงใช้ 3000)
lr_decay_iters = max_iters
min_lr = 0.0             # ← ควรเป็น learning_rate/10
```

### คำสั่งเทรนที่รันอยู่ (background `bzn3rme4y`)
```
caffeinate -i ../tools/venv/bin/python -u train.py \
  --device=mps --dtype=float32 \
  --batch_size=16 --gradient_accumulation_steps=16 \
  --max_iters=3000 --lr_decay_iters=3000 \
  --eval_interval=250 --eval_iters=100 --log_interval=20
```
> ⚠️ run นี้เทรนบน Wikipedia prose → **จะไม่ทำให้ตอบคำถาม factual ถูกขึ้น** อย่าคาดหวัง

### QEMU test (ทำเสร็จแล้ว)
```
make qemu                     # หน้าต่าง พิมพ์โต้ตอบ (F1 สลับ EN/TH)
make qemu-headless            # headless + QMP socket
python3 tools/screendump.py /tmp/qemu_bm.sock /tmp/shot.ppm   # จับภาพ
```
- บน M3: ใช้ TCG (ไม่มี hvf สำหรับ x86_64), vvfat ต้อง `rw` ไม่ใช่ `ro`
