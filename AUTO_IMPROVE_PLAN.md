# แผน: AI ไทยที่เรียนรู้ + ปรับปรุงโมเดลอัตโนมัติ

> เขียน 2026-11-09 — บริบทของ branch `feature/thai-qa-corpus` · โมเดล llama2.c 15M (dim=288, 6 layers, 6 heads, vocab 32000, seq_len 256) รันเป็น UEFI app บน bare-metal
>
> เอกสารนี้เก็บไว้สำหรับ**ทำต่อในอนาคต** — เป็น roadmap ไม่ใช่ขั้นตอนที่ทำเสร็จแล้ว

---

## 1. หลักการ: "ปรับปรุงโมเดลอัตโนมัติ" คือ loop นี้

```
เก็บข้อมูล → ทำความสะอาด → เทรน → ประเมิน → เจอจุดพลาด → เก็บข้อมูลเพิ่ม → เทรนใหม่ → (วนซ้ำ)
```

LLM สมัยใหม่**ไม่**อัปเดตน้ำหนักสด ๆ ตอนรัน (online learning) เพราะทำให้เกิด
**catastrophic forgetting** (ลืมของเก่า) และไม่เสถียร การพัฒนาจริงคือ
**retrain / fine-tune แบบออฟไลน์ด้วยข้อมูลชุดใหม่** เสมอ

สิ่งที่ทำไปแล้วใน branch นี้คือ loop นี้แบบ manual ครบทุกขั้น:

| ขั้น | เครื่องมือปัจจุบัน | สถานะ |
|---|---|---|
| เก็บข้อมูล | `tools/build_thai_corpus.py` (Wikipedia) | ✅ |
| ทำความสะอาด | `tools/clean_thai_corpus.py` | ✅ |
| สร้าง Q/A + boost | `tools/build_qa_corpus.py` + `qa_thai.txt` | ✅ (มือ) |
| เทรน | `llama2c/train.py` (+ `--finetune` ที่เพิ่งเพิ่ม) | ✅ |
| ประเมิน | `tools/check_qa.py` (auto-grade ด้วย `# key:`) | ✅ |
| เจอจุดพลาดแล้วเติม | `tools/auto_improve.py` (boost อัตโนมัติ) | ✅ (แผน A) |

**ตัวเชื่อมขั้นให้เป็น loop เดียวเสร็จแล้ว — `tools/auto_improve.py` (ดูแผน A ด้านล่าง)**

---

## 2. สิ่งที่ทำได้จริงที่สเกลของเรา

| ส่วนของ loop | ทำได้ไหม | หมายเหตุ |
|---|---|---|
| ประเมินอัตโนมัติ | ✅ | `check_qa.py` คือ seed — ขยายเป็น "จับว่าข้อไหนพลาด" |
| เก็บข้อมูลอัตโนมัติ | ✅ | สกัด (entity, คุณสมบัติ, ค่า) จาก Wikipedia → Q/A |
| เติมเฉพาะข้อที่พลาด | ✅ | แบบที่เพิ่งทำกับ "69" → ทำให้เป็น script |
| retrain อัตโนมัติ | ✅ | `--finetune` (resume + LR ใหม่) เปิดทางไว้แล้ว |
| เทรนบนตัว EFI (on-device) | ❌ | bare-metal 512MB ทำได้แค่ *inference*; เทรนต้องรันบน Mac (PyTorch/MPS) |

**สรุป**: "auto-improve" = **orchestrator ตัวเดียว** ที่วน 4 ขั้นบน host Mac แล้ว deploy โมเดลใหม่ลง EFI เมื่อคะแนนดีขึ้น

---

## 3. สามแผนต่อยอด (เรียงตามความคุ้ม)

### แผน A — Auto-boost loop ✅ เสร็จแล้ว (`tools/auto_improve.py` + `/auto-improve`)

ต่อยอดจากสิ่งที่ทำมือวันนี้ให้เป็น "กดปุ่มเดียว":

```
check_qa.py (หา FAIL) → script เติม fact ที่พลาดซ้ำหนักขึ้น
  → build_qa_corpus.py + prep_thai.py → train.py --finetune
  → วนจนครบ X/X หรือหมดงบเวลา
```

- **ของที่เขียนไปแล้ว**: `tools/auto_improve.py` (orchestrator) + `build_qa_corpus.py --boost <fact>` (เพิ่ม repeat เฉพาะ fact ที่พลาด) + `check_qa.grade()` (ใช้ซ้ำ) + final-save ใน `train.py`
- **จุดตัดสินใจที่ทำแล้ว**: เกณฑ์หยุด (pass ครบ / `--max-rounds`) และ boost keys แบบ **union สะสม** เพื่อกันไม่ให้ overfit จนลืม fact เดิม

### แผน B — Fact auto-expansion ✅ เสร็จแล้ว (`tools/expand_facts.py` → `qa_thai_auto.txt`)

สกัดข้อเท็จจริงจาก `thai_corpus_clean.txt` อัตโนมัติ → Q/A (ทำเสร็จ: **56 ข้อ**):

- **ทำได้จริง (high-precision)**: จับ pattern ที่มี **ตัวเลขเป็นจุดยึด** — `X มีประชากร N คน`
  และ `X มีพื้นที่ N ตารางกิโลเมตร` (ประธานสกัดจาก Thai run ก่อนจุดยึด + noise filter + dedup)
- **พยายามแล้วแต่ทิ้ง** (noise เกินไปหากไม่มี Thai word segmenter): `X เป็นเมืองหลวงของ Y`
  (เจอแต่ "เคย/กลายเป็นเมืองหลวง…"), `สูง … เมตร`, `ยาว … กิโลเมตร`, `มี … จังหวัด`
- **fact-verify**: อาศัยว่า pattern มาจากประโยคใน corpus โดยตรง + ตรวจทานด้วยมือหนึ่งรอบ
  (เหลือ headroom: ติดตั้ง `pythainlp` แล้วค่อยทำ NER + template จริงจัง)

### แผน C — Teacher distillation ✅ เสร็จแล้ว (`qa_thai_big.txt` — 39 ข้อ แต่งโดย Claude)

- ครู (Claude) เขียน `qa_thai_big.txt`: เมืองหลวง 15 ประเทศ + ข้อเท็จจริงโลก (แม่น้ำไนล์,
  เอเวอเรสต์, ดาวเคราะห์ 8 ดวง…) + ภูมิศาสตร์/ประวัติศาสตร์/วัฒนธรรมไทย
- ทั้งหมดตรวจทานความถูกต้องแล้ว ต้นทุน = 0 (ไม่ต้องเรียก API ครู)
- รวมเป็น `qa_thai_all.txt` (14 + 39 + 56 = **109 ข้อ**) → ใช้เป็นชุดเทรน/ให้คะแนนของ loop

---

## 4. ข้อจำกัดที่ต้องจำไว้ (กันคาดหวังเกิน)

- **"เข้าใจภาษาไทย" ของโมเดล 15M = ความคล่อง + การจำ pattern** ไม่ใช่ความเข้าใจ
  เชิงความหมาย มันตอบถูกเพราะเห็นคำถาม-คำตอบซ้ำในเทรน ไม่ได้ "ใช้เหตุผล" กับข้อเท็จจริงใหม่
- **ความจุมีเพดาน** — 15M params จดจำ fact ได้หลักสิบถึงหลักร้อยข้อ ยิ่งเพิ่มยิ่งชนกัน
  (interference) ถ้าอยาก "รู้" จริงต้องขยายพารามิเตอร์ (50–100M+) ซึ่ง bare-metal RAM
  พอไหวแต่เทรนช้าลง
- **RLHF / preference แบบ OpenAI** สเกลเรา overkill — ใช้เกณฑ์ให้คะแนนง่าย ๆ
  (แบบ `# key:`) ก็เพียงพอ
- **on-device เรียนไม่ได้** — ตัว EFI เป็นแค่ inference; loop ทั้งหมดรันบน Mac

---

## 5. ลำดับที่แนะนำ

1. **แผน A** (auto-boost loop) — ✅ เสร็จแล้ว: `tools/auto_improve.py` + `build_qa_corpus.py --boost` + `/auto-improve`
2. **แผน B** (fact auto-expansion) — ✅ เสร็จแล้ว: `tools/expand_facts.py` → `qa_thai_auto.txt` (56 ข้อ)
3. **แผน C** (teacher distillation) — ✅ เสร็จแล้ว: `qa_thai_big.txt` (39 ข้อ)

> ทั้งสามแผนเสร็จแล้ว → ชุดข้อเท็จจริงรวม `qa_thai_all.txt` (109 ข้อ) พร้อมเทรนผ่าน `/auto-improve`
> อ่านคู่กับ `TRAINING_PLAN.md` (แผนเทรน), `TEST_THAI.md` (ขั้นทดสอบ), `GUIDE_THAI.md`
