---
description: ฟอร์แมต USB เป็น FAT32 + ลงไฟล์ boot สำหรับเครื่องจริง
argument-hint: "[disk]"
---

สร้าง **boot USB สำหรับเครื่องจริง** โดยฟอร์แมต USB เป็น FAT32 (MBR) แล้ว copy ไฟล์
ทั้งหมดจาก `fat_usb/` ลงไป (EFI/BOOT/BOOTX64.EFI + โมเดล .bin + tokenizer.bin + kb.txt)

ขั้นตอน:

1. **ระบุ disk เป้าหมาย**
   - ผู้ใช้พิมพ์ `/createboot /dev/disk4` (หรือ `disk4`) → ใช้ค่านั้น
   - ไม่ระบุ → รัน `diskutil list` แล้วให้ผู้ใช้เลือก USB
     (เลือกโดยดูขนาด + แท็ก `(external, physical)` เพื่อกันเผลอเลือก disk ระบบ)

2. **build ให้สด** → `make install`
   - build + copy `BOOTX64.EFI` ล่าสุดลง `fat_usb/EFI/BOOT/`
   - ถ้าผู้ใช้เพิ่งแก้ Q&A/kb → รัน `tools/venv/bin/python tools/build_kb.py`
     แล้ว `cp kb.txt fat_usb/kb.txt` ก่อน (ปกติ `fat_usb/kb.txt` มีอยู่แล้ว)

3. **ลง USB** → `./create_usb.sh <disk>`
   - ตัว script มี safety check (กันลบ disk ระบบ/internal) และขอพิมพ์ `YES` ยืนยัน
     ก่อนฟอร์แมต แล้ว format + copy + verify + eject ให้อัตโนมัติ

4. **สรุปผล** — บอกผู้ใช้ว่าบูตเครื่องจริงยังไง:
   เปิดเครื่องจริงในโหมด UEFI แล้วเลือก entry `UEFI:` (ชื่อ USB ที่เพิ่งสร้าง)
