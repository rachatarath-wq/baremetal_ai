---
description: ดูความคืบหน้าของ train ที่รันอยู่
---

ดู background task ที่รัน train อยู่:

- ถ้ามี task train รันอยู่ → อ่านไฟล์ output ของ task นั้น (tail 20 บรรทัดสุดท้าย) แล้วสรุป: step ล่าสุด, loss, lr, throughput, และ ETA
- ถ้าไม่มี train กำลังรัน → บอกผู้ใช้ว่ายังไม่มี train รันอยู่ และวิธีเริ่มคือ `/train`

หมายเหตุ: ไฟล์ output ของ background task อยู่ใต้ `/private/tmp/claude-*/.../tasks/*.output`
