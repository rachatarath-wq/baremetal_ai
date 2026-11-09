---
description: เทรนโมเดลภาษาไทยบน MPS (รันยาวใน background) (step 3)
argument-hint: "[max_iters|resume]"
---

เทรนโมเดลไทยจาก `data/train.bin` (ต้องรัน `/prep` ก่อน)

**ข้อจำกัดเครื่อง (RAM 8 GB) — สำคัญมาก:**
- ต้องใช้ `--batch_size=16 --gradient_accumulation_steps=16` (effective batch ยัง = 256)
- **ห้าม** `--batch_size=64` → จะ OOM ตอน backward (logits tensor 64×256×32000 = 2.1 GB)
- ใช้ `--dtype=float32` (ปลอดภัยสุดบน MPS)

**parse argument จากผู้ใช้:**
- `resume` → เพิ่ม `--init_from=resume` (ต่อจาก ckpt เดิม)
- ตัวเลข N → `max_iters=N` (default 3000)
- ไม่ส่งอะไร → max_iters=3000

รันแบบ background + unbuffered + กันเครื่องหลับ (ต้อง `run_in_background: true`):
```
cd /Volumes/backup2568/project/baremetal_ai/llama2c
caffeinate -i ../tools/venv/bin/python -u train.py \
  --device=mps --dtype=float32 \
  --batch_size=16 --gradient_accumulation_steps=16 \
  --max_iters=N --lr_decay_iters=N \
  --eval_interval=250 --eval_iters=100 --log_interval=20 \
  [--init_from=resume]
```

report ให้ผู้ใช้: throughput (~12k tok/s), ETA (~5 ชม. ที่ 3000 iters),
checkpoint อยู่ที่ `llama2c/out_thai/ckpt.pt` (save ทุก 250 iters, resume ได้)
