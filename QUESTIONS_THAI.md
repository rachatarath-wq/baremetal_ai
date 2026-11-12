# คำถามท่องเที่ยว · Tourism prompts

ชุดคำถามสำหรับทดสอบโมเดลโดเมน **ท่องเที่ยวไทย** (`stories15M_thai_tourism_q80.bin`).
พิมพ์คำถาม (ตัวหนา) ลงใน AI เพื่อเช็คว่าตอบตรงกับ "คำตอบที่ควรได้" หรือไม่

Prompts to test the **Thai tourism** domain model (`stories15M_thai_tourism_q80.bin`).
Type a question (in bold) and check the answer against the expected one.

> วิธีใช้: บูต USB → `/loadmodel` เลือก `stories15M_thai_tourism_q80.bin` → กด `F1` เป็นโหมดไทย → พิมพ์คำถาม
> How to use: boot the USB → `/loadmodel` → pick `stories15M_thai_tourism_q80.bin` → press `F1` for Thai → type a question.

---

## สถานที่ / Places

- **สถานที่ท่องเที่ยวที่มีชื่อเสียงที่สุดของประเทศไทยคือที่ไหน**
  *(Where are Thailand's most famous destinations?)*
  → กรุงเทพมหานคร, เชียงใหม่, ภูเก็ต, อยุธยา, พัทยา, เกาะสมุย, หมู่เกาะพีพี

- **เมืองหลวงของประเทศไทยคือเมืองอะไร**
  *(What is the capital of Thailand?)*
  → กรุงเทพมหานคร

- **กรุงเทพมหานครมีสถานที่ท่องเที่ยวอะไรบ้าง**
  *(What are Bangkok's attractions?)*
  → วัดพระแก้ว, วัดโพธิ์, พระบรมมหาราชวัง, วัดอรุณราชวราราม, ถนนข้าวสาร

- **เชียงใหม่มีสถานที่ท่องเที่ยวอะไรขึ้นชื่อ**
  *(What is Chiang Mai famous for?)*
  → ดอยสุเทพ, วัดพระธาตุดอยสุเทพ, ประตูท่าแพ, ถนนคนเดิน

- **อยุธยาเป็นมรดกโลกหรือไม่**
  *(Is Ayutthaya a World Heritage site?)*
  → อุทยานประวัติศาสตร์พระนครศรีอยุธยา (มรดกโลกยูเนสโก)

- **สุโขทัยมีชื่อเสียงเรื่องอะไร**
  *(What is Sukhothai famous for?)*
  → อุทยานประวัติศาสตร์สุโขทัย (มรดกโลก), เมืองหลวงแห่งแรกของไทย

- **ตลาดน้ำที่มีชื่อเสียงของไทยคือตลาดอะไร**
  *(What is Thailand's famous floating market?)*
  → ตลาดน้ำดำเนินสะดวก, ตลาดน้ำอัมพวา

## ทะเล / Sea & islands

- **ภูเก็ตมีชื่อเสียงเรื่องอะไร**
  *(What is Phuket famous for?)*
  → ชายหาดสวย เช่น หาดป่าตอง, หาดกะรน, เกาะพีพี

- **พัทยาอยู่จังหวัดอะไร**
  *(Which province is Pattaya in?)*
  → ชลบุรี

- **เกาะสมุยอยู่จังหวัดอะไร**
  *(Which province is Koh Samui in?)*
  → สุราษฎร์ธานี

- **หมู่เกาะพีพีอยู่จังหวัดอะไร**
  *(Which province are the Phi Phi Islands in?)*
  → กระบี่

- **กระบี่มีชื่อเสียงเรื่องอะไร**
  *(What is Krabi famous for?)*
  → ทะเลสวย, หน้าผาหินปูน, เกาะพีพี, อ่าวนาง

## ธรรมชาติ / Nature

- **น้ำตกที่มีชื่อเสียงของประเทศไทยมีที่ไหนบ้าง**
  *(What are Thailand's famous waterfalls?)*
  → น้ำตกเอราวัณ, น้ำตกทีลอซู, น้ำตกแม่ยะ

- **ฤดูที่เหมาะแก่การท่องเที่ยวทะเลในไทยคือฤดูอะไร**
  *(Which season is best for the sea?)*
  → ฤดูหนาว (พฤศจิกายน–กุมภาพันธ์)

## ข้อมูลทั่วไป / General

- **ประเทศไทยมีกี่จังหวัด** ⚠️
  *(How many provinces does Thailand have?)*
  → 77 จังหวัด — *โมเดลยังตอบผิดบ่อย*

- **อาหารไทยยอดนิยมมีอะไรบ้าง** ⚠️
  *(What are popular Thai dishes?)*
  → ต้มยำกุ้ง, ผัดไทย, ส้มตำ, แกงเขียวหวาน, ข้าวเหนียวมะม่วง — *โมเดลยังสับสน*

- **เงินตราที่ใช้ในประเทศไทยคืออะไร**
  *(What currency does Thailand use?)*
  → บาท

- **ประเทศไทยตั้งอยู่ในภูมิภาคใด**
  *(Which region is Thailand in?)*
  → เอเชียตะวันออกเฉียงใต้

---

## คะแนนปัจจุบัน / Current score

`stories15M_thai_tourism_q80.bin` ตอบถูก **16/18** (เหลือ 2 ข้อเครื่องหมาย ⚠️)
เทียบกับโมเดลทั่วไป `stories15M_thai_q80.bin` ที่ได้ **3/18**
