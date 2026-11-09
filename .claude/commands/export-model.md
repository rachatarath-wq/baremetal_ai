---
description: export checkpoint → Q8_0 v2 (stories15M_thai_q80.bin) (step 4)
---

export `llama2c/out_thai/ckpt.pt` เป็น `stories15M_thai_q80.bin` (Q8_0, version 2)

รัน (จาก folder llama2c):
```
cd /Volumes/backup2568/project/baremetal_ai/llama2c
../tools/venv/bin/python export.py ../stories15M_thai_q80.bin --version 2 --checkpoint out_thai/ckpt.pt
```

จบแล้วตรวจ header ให้ตรงกับ run.c:
- magic = 0x616b3432, version = 2
- dim=288 hidden=768 layers=6 heads=6 kv_heads=6 vocab=32000 seq=256
- shared_classifier=1, group_size=32 (export.py จะ backoff จาก 64 → 32 อัตโนมัติ)
- ขนาดไฟล์ = 17,101,696 bytes
