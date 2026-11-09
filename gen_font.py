
from PIL import Image, ImageDraw, ImageFont
import sys

# เรียกใช้ฟอนต์ Thonburi จากระบบ macOS โดยตรง
font_path = "/System/Library/Fonts/Thonburi.ttc"
try:
    font = ImageFont.truetype(font_path, 12)
except IOError:
    print("ไม่พบฟอนต์ กรุณาตรวจสอบพาธ")
    sys.exit()

# ปริ้นต์หัว Array
print("static const unsigned char font8x16_thai[128][16] = {")

for cp in range(0x0E00, 0x0E80):
    char = chr(cp)
    idx = cp - 0x0E00  # คำนวณ Index (0-127)
    
    img = Image.new('1', (8, 16), color=0)
    draw = ImageDraw.Draw(img)

    # ขยับแกน Y ขึ้นเล็กน้อยเพื่อให้มีพื้นที่สำหรับสระล่าง
    draw.text((0, -2), char, font=font, fill=1)

    hex_rows = []
    for y in range(16):
        row_val = 0
        for x in range(8):
            if img.getpixel((x, y)):
                row_val |= (1 << (7 - x))
        hex_rows.append(f"0x{row_val:02X}")

    # จัดกลุ่มโค้ด Hex ทีละ 4 ตัว (เพื่อให้อ่านง่ายเหมือนตัวอย่างของคุณ)
    g1 = ",".join(hex_rows[0:4])
    g2 = ",".join(hex_rows[4:8])
    g3 = ",".join(hex_rows[8:12])
    g4 = ",".join(hex_rows[12:16])
    
    # ปริ้นต์ผลลัพธ์ออกมาในรูปแบบที่ต้องการ: [index] = { ... }, /* char */
    print(f"    [{idx}] = {{{g1}, {g2}, {g3}, {g4}}}, /* {char} */")

# ปริ้นต์ปิดท้าย Array
print("};")