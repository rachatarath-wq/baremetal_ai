# คู่มือสมุดเพลง + การสร้าง USB จาก Release Image

**Songbook (สมุดเพลง) manual & bootable-image guide** — how to download the release
image and make a bootable USB, plus how to use the on-device songbook (เนื้อ + คอร์ด),
add/edit/delete/search songs, and convert chord tabs from the web into this project's
format.

คู่มือนี้ครอบคลุม 4 เรื่อง:

1. ดาวน์โหลด **release image** แล้วสร้าง USB ที่บูตได้เอง (ไม่ต้องคอมไพล์)
2. วิธีใช้ **สมุดเพลง** (เลือกเพลง / ค้นหา / แก้ / เพิ่ม / ลบ / นำเข้า)
3. **รูปแบบไฟล์เพลง** (`@title` / `@artist` / บรรทัด `>คอร์ด` / `@end`)
4. **เครื่องมือแปลงคอร์ด** จากเว็บ (chordtabs.in.th และ text คอร์ดทั่วไป)

ดูรายละเอียด build จาก source / พิมพ์ไทย / ทดสอบ QEMU ได้ที่ [`README.md`](README.md),
[`GUIDE_THAI.md`](GUIDE_THAI.md), [`USAGE_THAI.md`](USAGE_THAI.md)

---

## 1. ดาวน์โหลด Release Image และสร้าง USB บูตได้

ในหน้า **GitHub Releases** ของ repo จะมีไฟล์ `BOOTAI-v*.img` (เป็น disk image แบบ
MBR + FAT32 ที่บรรจุ `EFI/BOOT/BOOTX64.EFI`, โมเดลเริ่มต้น, tokenizer, และเพลงตัวอย่าง
ไว้ครบ) ดาวน์โหลดไฟล์ `.img` นี้ แล้วเขียนลง USB ตาม OS ที่คุณใช้

> ⚠️ **คำเตือนก่อนเขียนลง USB:** คำสั่งด้านล่างจะ **ลบข้อมูลทั้งหมดใน USB** ให้
> เช็คชื่อ device ให้ถูกต้องก่อนทุกครั้ง (เช่น `/dev/disk6`) ห้ามเขียนทับฮาร์ดดิสก์
> ของเครื่องเด็ดขาด

### macOS

```bash
# 1) หา USB ของคุณ (ดูขนาด + คำว่า "external, physical")
diskutil list

# 2) eject ก่อน (อย่าลืมเปลี่ยน disk6 เป็น USB จริงของคุณ)
diskutil unmountDisk /dev/disk6

# 3) เขียน image ลง USB แบบ raw (rdisk = เร็ว, อย่าใช้ผิด device!)
sudo dd if=BOOTAI-v1.0.0.img of=/dev/rdisk6 bs=1m

# 4) eject หลังเขียนเสร็จ
diskutil eject /dev/disk6
```

หรือใช้ **balenaEtcher** (GUI) — เลือกไฟล์ `.img` เลือก USB กด Flash ก็เสร็จ (ง่ายสุด)

### Windows

ใช้ **Rufus** หรือ **balenaEtcher**:
- Rufus: เลือก Device = USB → "Boot selection" = `BOOTAI-v*.img` → Start
- หรือ balenaEtcher: เลือก `.img` → เลือก USB → Flash

(Windows ไม่มี `dd` ในตัว ถ้าชอบ CLI ใช้ `balena-cli` หรือ WSL ก็ได้)

### Linux

```bash
# หา USB:  lsblk   (เช่น /dev/sdb)
sudo dd if=BOOTAI-v1.0.0.img of=/dev/sdb bs=1M status=progress oflag=sync
sync
```

### วิธีบูต

1. เสียบ USB ที่เขียนเสร็จแล้ว
2. เปิดเครื่อง กดปุ่มเลือก boot menu (F12 / F8 / Esc / F11 ต่างกันตามเครื่อง)
3. เลือกรายการ **UEFI: (ชื่อ USB)** — ต้องเป็น UEFI ไม่ใช่ Legacy/BIOS
4. AI จะโผล่บนจอก่อนเข้าระบบปฏิบัติการใด ๆ

> **ไฟล์ใน image:**
> ```
> EFI/BOOT/BOOTX64.EFI      ← ตัวโปรแกรม (UEFI application)
> stories15M_thai_q80.bin   ← โมเดลภาษาไทยเริ่มต้น
> tokenizer.bin             ← tokenizer (Llama 2 vocab)
> kb.txt                    ← ฐานความรู้ (ถาม-ตอบ, ไม่บังคับ)
> songs/                    ← สมุดเพลง (ตัวอย่าง)
> import/                   ← โฟลเดอร์สำหรับนำเข้าเพลงเพิ่ม
> songmarks.txt             ← ที่คั่นหน้าที่เคยเปิด (สร้างเองอัตโนมัติ)
> ```
> โมเดล `.bin` เพิ่ม (เช่น `stories15M_thai_tourism_q80.bin`) ใส่ไว้ root ของ USB แล้วใช้
> คำสั่ง `/loadmodel` เพื่อสลับได้ทันที

---

## 2. ใช้สมุดเพลง (Songbook)

พิมพ์ `/songs` (หรือกดลูกศรขึ้น/ลงเพื่อเลือกเพลง) แล้ว Enter เพื่อเปิดสมุดเพลง

### หน้าจอเลือกเพลง

| ปุ่ม | การทำงาน |
|------|-----------|
| พิมพ์ข้อความ | ค้นหาเพลง (ค้นจาก **ชื่อ** หรือ **ศิลปิน** ก็ได้) |
| `Enter` | เปิดเพลงที่เลือก |
| `N` | เพิ่มเพลงใหม่ (เข้า editor) |
| `E` | แก้เพลงที่เลือก |
| `Del` | ลบเพลงที่เลือก (ต้องกด Enter ยืนยัน) |
| `I` | นำเข้าเพลงจากโฟลเดอร์ `import/` |
| `Esc` | กลับไปหน้าแชท |

### หน้าจอเนื้อเพลง

| ปุ่ม | การทำงาน |
|------|-----------|
| `↑` / `↓` | เลื่อนทีละบรรทัด |
| `PageUp` / `PageDown` | เลื่อนทีละหน้า |
| `E` | แก้เพลงนี้ |
| `Del` | ลบเพลงนี้ |
| `Esc` | กลับ (บันทึกที่คั่นหน้าให้อัตโนมัติ) |

- บรรทัดที่ขึ้นต้นด้วย `>` จะเป็น **คอร์ด** (สีเขียว/สีเน้น)
- บรรทัดธรรมดาเป็น **เนื้อร้อง** (สีขาว)
- แถวล่างแสดง "บรรทัด N/M" = ตำแหน่งบรรทัดปัจจุบัน/ทั้งหมด

### Editor (เพิ่ม / แก้เพลง)

- 3 แถวแรกคือ **ชื่อเพลง / ศิลปิน / อัลบั้ม**
- แถวถัด ๆ ไปคือเนื้อ + คอร์ด (คอร์ดพิมพ์นำหน้าด้วย `>`)
- `Enter` แก้แถวที่เลือก (เลื่อนเคอร์เซอร์/ลบ/แทรกกลางบรรทัดแบบ UTF-8 ปลอดภัย)
- `S` บันทึกทั้งเล่มกลับลง USB
- `Esc` ออกจาก editor

### พิมพ์ภาษาไทย

- ปุ่ม **`F1`** สลับโหมด แป้นพิมพ์อังกฤษ ↔ ไทย (เลย์เอาต์ **เกษมณี**)
- มุมล่างซ้ายแสดงสถานะ `EN` / `TH`

---

## 3. รูปแบบไฟล์เพลง

ไฟล์เพลง 1 เพลง = 1 ไฟล์ `.txt` ในโฟลเดอร์ `songs/` (หรือหลายเพลงรวมกันในไฟล์เดียว)

```
@title ชื่อเพลง
@artist ชื่อศิลปิน
@album ชื่ออัลบั้ม
>C      G      Am     F          ← บรรทัดคอร์ด เริ่มด้วย > แล้วตามด้วยคอร์ด+ช่องว่าง
เห็นเธอครั้งแรก ใจมันสั่นไหว        ← บรรทัดเนื้อ (ไม่มี >)
# (Instrumental)                 ← บรรทัด comment เริ่มด้วย # จะถูกข้าม
@end                             ← จบเพลงนี้
```

กฎที่ต้องรู้:

- **จัดคอร์ดให้ตรงเนื้อด้วย "ช่องว่าง"** — จอแสดงผลเป็น monospace ทุกตัวอักษร (ไทย/
  ลาติน) กว้าง 1 ช่องเท่ากัน ดังนั้นกด space จัดคอร์ดให้ตรงพยางค์ที่ต้องการได้เลย
- **สระ/วรรณยุกต์ลอย** (`ั ิ ี ึ ื ุ ู ่ ้ ๊ ๋ ็ ์ ำ ฯลฯ`) **ไม่นับช่อง** — ตัวอักษรไทย
  ประกอบกัน (เช่น `ใจ` = 1 ช่อง) เครื่องมือ `align_chords.py` ช่วยนับให้ถูกต้อง
- ชื่อเพลง/ศิลปิน/อัลบั้ม อยู่หลัง `@title` / `@artist` / `@album`
- ค้นหาได้ทั้งชื่อและศิลปิน

---

## 4. เครื่องมือแปลงคอร์ดจากเว็บ

ในโฟลเดอร์ `tools/`:

### `chordtabs_convert.py` — แปลงจาก chordtabs.in.th หรือ text คอร์ดทั่วไป

```bash
# โหมด 1: chordtabs.in.th (URL หรือไฟล์ .html ที่ save ไว้)
python3 tools/chordtabs_convert.py https://chordtabs.in.th/73028/ -o songs/0005.txt
python3 tools/chordtabs_convert.py saved.html -o songs/0005.txt

# โหมด 2: text คอร์ดแบบ inline [คอร์ด]
python3 tools/chordtabs_convert.py --inline song.txt --title "ชื่อเพลง" --artist "ศิลปิน"
```

- chordtabs.in.th แสดงคอร์ดเป็น **รูป** (ไม่ใช่ตัวอักษร) script จึงดึงได้เฉพาะชื่อ/ศิลปิน/
  รายการคอร์ด + เนื้อ แล้วให้คุณวางคอร์ดเองด้วย `align_chords.py`
- text แบบ `[C]ฉันรักเธอ [G]เสมอมา` จะถูกแปลงเป็นบรรทัด `>C` + เนื้ออัตโนมัติ

### `align_chords.py` — วางชื่อคอร์ดให้ตรงพยางค์ (นับช่องให้อัตโนมัติ)

```bash
python3 tools/align_chords.py "หากว่าเธอผ่านมาได้ยินเพลงนี้" "Am@หาก" "G@ได้" "F@เพลง"
# >Am           G   F
```

รูปแบบ `คอร์ด@คำยึด` — คอร์ดจะถูกวางที่ช่องที่คำนั้นเริ่มต้น (สระ/วรรณยุกต์ไม่นับช่อง
ตรงกับ run.c เป๊ะ)

---

## 5. Build จาก source (สำหรับ developer)

ถ้าไม่อยากใช้ release image แต่จะ compile เอง:

```bash
make            # สร้าง build/BOOTX64.EFI (ต้องมี x86_64-w64-mingw32-gcc + gnu-efi)
make install    # copy ลง fat_usb/ พร้อม songs/
make qemu       # ทดสอบใน QEMU แบบมีหน้าต่าง
```

รายละเอียดเพิ่มเติม (dependency, การ flash USB ด้วย `create_usb.sh`, การฝึกโมเดล)
อ่านได้ที่ [`README.md`](README.md) และ [`GUIDE_THAI.md`](GUIDE_THAI.md)

---

## หมายเหตุเรื่องลิขสิทธิ์

เพลงตัวอย่างใน repo เป็นเพลง **แต่งเอง (demo)** เท่านั้น เนื้อร้อง/คอร์ดของเพลงจริงที่มี
ลิขสิทธิ์ (เช่น เพลงของศิลปินชื่อดัง) **ไม่ควร commit ขึ้น repo สาธารณะ** — ถ้าต้องการ
ใช้ส่วนตัว ให้เก็บเป็นไฟล์ใน `songs/` ของเครื่องตัวเอง โดยไม่ `git add` ขึ้นไป
