---
description: ชี้ run.c ไปที่โมเดลที่เลือก + make install (step 5)
argument-hint: "[model-name]"
---

เปลี่ยน `checkpoint_name` ใน `run.c` ให้ชี้ไปที่โมเดลที่เลือก แล้ว build ลง USB

**ตัวเลือกโมเดล** (default = stories15M_thai_q80.bin):
- `stories15M_q80.bin` — โมเดลเล็กอังกฤษ (16 MB)
- `stories15M_thai_q80.bin` — โมเดลไทย (16 MB)
- `tinyllama_q80.bin` — โมเดลใหญ่ (1.1 GB)

ขั้นตอน:
1. ถ้าผู้ใช้ระบุชื่อโมเดล → ใช้ชื่อนั้น; ถ้าไม่ → stories15M_thai_q80.bin
2. แก้บรรทัดใน `run.c` (~บรรทัด 1906):
   `CHAR16 *checkpoint_name = L"\\<model>";`
3. `make && make install`
4. เตือนผู้ใช้: ต้องวางไฟล์โมเดล + tokenizer.bin ไว้ที่ root ของ USB (FAT32) ด้วย
