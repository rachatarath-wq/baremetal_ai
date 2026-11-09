---
description: รัน auto-boost loop (grade → boost → finetune → วน) — แผน A
argument-hint: "[dry-run|max-rounds]"
---

รัน **auto-improve loop** (จาก `AUTO_IMPROVE_PLAN.md` แผน A) อัตโนมัติ โดยให้คะแนน
เทียบกับ `qa_thai_all.txt` (109 ข้อ = 14 หลัก + 39 Plan C + 56 Plan B):

```
check_qa (หา FAIL) → build_qa_corpus --boost (เพิ่ม repeat เฉพาะ fact ที่พลาด)
  → prep_thai → train.py --finetune → วนจนครบ หรือหมด --max-rounds
```

**ตรวจแผนก่อน (dry-run — ไม่เทรน แค่โชว์ว่า fact ไหนพลาด + จะรันคำสั่งอะไร):**
```
cd /Volumes/backup2568/project/baremetal_ai
tools/venv/bin/python tools/auto_improve.py --dry-run --max-rounds 5
```

**รันจริง (เทรนยาว ~45 นาที/รอบ บน MPS — ควร run_in_background):**
```
cd /Volumes/backup2568/project/baremetal_ai
tools/venv/bin/python tools/auto_improve.py \
  --checkpoint llama2c/out_thai_v2/ckpt.pt \
  --max-rounds 5 --finetune-iters 500
```

**สำคัญ:**
- ห้ามแก้ `ckpt.pt` ระหว่าง loop — มัน resume จาก checkpoint เดิม แล้ว finetune ต่อเอง
- loop เก็บ boost keys แบบสะสม (union) เพื่อไม่ให้ลืม fact เดิมที่เคยแก้
- พารามิเตอร์หลัก: `--max-rounds` (จำนวนรอบ), `--finetune-iters` (step/รอบ),
  `--boost-mult` (คูณ repeat เฉพาะ fact ที่พลาด, default 3x)
- รายละเอียดไฟล์/ข้อจำกัด อ่าน `AUTO_IMPROVE_PLAN.md`
