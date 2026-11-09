---
description: tokenize thai_corpus.txt → data/train.bin + val.bin (step 2)
---

tokenize `thai_corpus.txt` เป็น flat uint16 data files (ต้องรัน `/corpus` ก่อน)

รัน:
```
cd /Volumes/backup2568/project/baremetal_ai
tools/venv/bin/python tools/prep_thai.py --input thai_corpus.txt --out_dir data --tokenizer tokenizer.model
```

จบแล้วตรวจว่ามี `data/train.bin` และ `data/val.bin` เกิดขึ้น แล้ว report จำนวน token (train.bin ไบต์ ÷ 2 = จำนวน token)
