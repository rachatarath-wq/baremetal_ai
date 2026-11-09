---
description: สร้าง Thai corpus จาก Wikipedia → thai_corpus.txt (step 1)
argument-hint: "[limit-mb]"
---

สร้าง corpus ภาษาไทยจาก Wikipedia เป็นไฟล์ `thai_corpus.txt`

- ถ้าผู้ใช้ส่ง argument ตัวเลข (เช่น `/corpus 150`) ใช้เป็น limit-mb
- ถ้าไม่ส่ง ใช้ default 100

รัน:
```
cd /Volumes/backup2568/project/baremetal_ai
tools/venv/bin/python tools/build_thai_corpus.py --output thai_corpus.txt --limit-mb <N>
```

จบแล้ว report ขนาดไฟล์ `thai_corpus.txt` ที่ได้
