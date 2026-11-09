---
description: แสดงคู่มือภาพรวมโปรเจกต์ + รายการคำสั่ง train ทั้งหมด
---

อ่านไฟล์ `GUIDE_THAI.md` ใน root ของโปรเจกต์ แล้วสรุปเป็นภาษาไทยสั้น ๆ ให้ผู้ใช้ พร้อมลิสต์คำสั่ง slash ที่มีในระบบ train นี้:

- `/corpus` — สร้าง corpus ไทยจาก Wikipedia (step 1)
- `/prep` — tokenize เป็น data/train.bin + val.bin (step 2)
- `/train` — เทรนโมเดลไทย (รันยาวใน background) (step 3)
- `/train-status` — ดูความคืบหน้าของ train ที่รันอยู่
- `/export-model` — export checkpoint → Q8_0 v2 (step 4)
- `/build` — คอมไพล์ run.c → BOOTX64.EFI
- `/deploy` — ชี้ run.c ไปที่โมเดล + make install (step 5)

ลำดับ pipeline เต็ม: `/corpus` → `/prep` → `/train` → `/export-model` → `/deploy`
