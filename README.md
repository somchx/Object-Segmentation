# รายงานการทดลอง: Object Segmentation และ Detection
## CI 7204 / CI 7306 — การประมวลผลและวิเคราะห์ภาพ | สถาบันบัณฑิตพัฒนบริหารศาสตร์ (NIDA)

### 🎓 ข้อมูลผู้จัดทำ
- **รหัสนักศึกษา:** 6710421004
- **ชื่อ:** ชนัญญา เอี่ยมประโคน

### 🏛️ สถาบัน
- **สถาบันบัณฑิตพัฒนบริหารศาสตร์ (NIDA)**
- **คณะ:** คณะสถิติประยุกต์
- **สาขา:** วิทยาการคอมพิวเตอร์และระบบสารสนเทศ (CSIS)

---

โปรเจกต์นี้เป็นการทดลองใช้ **เทคนิค Image Processing แบบดั้งเดิม** เพื่อทำ Object Segmentation และ Detection  โดยใช้ **OpenCV** และ **Python** ทั้งหมด 40 กรณีทดสอบ แบ่งตามระดับความยากและผลลัพธ์ที่ได้

---

## 🛠️ วิธีการและขั้นตอนการทำงาน

Pipeline หลักที่ใช้ในการทดลองนี้เป็น Computer Vision แบบ Traditional ประกอบด้วย 5 ขั้นตอนหลัก ดังนี้

1. **Preprocessing (การเตรียมภาพ):** ลด Noise ด้วย Gaussian/Bilateral Blur และเพิ่ม Contrast ด้วย CLAHE
2. **Feature Extraction (การดึงคุณลักษณะ):**
    - **Color (สี):** ใช้ HSV สำหรับการทำ Hue-based Masking และ LAB สำหรับการแยกสีที่ดูคล้ายกันแต่ต่างกันในเชิง Perceptual
    - **Texture (พื้นผิว):** ใช้ Gabor Filter เพื่อตรวจจับรูปแบบทิศทาง เช่น ขนสัตว์หรือผ้า
    - **Edge (ขอบ):** ใช้ Canny และ Sobel เพื่อหารอยต่อของ Intensity
3. **Refinement (การปรับปรุง Mask):** ใช้ Morphological Operations (Opening, Closing, Dilation) เพื่อเชื่อมช่องว่างและกำจัด Noise blob
4. **Segmentation (การแบ่งกลุ่ม):** ใช้ Watershed Algorithm สำหรับวัตถุที่ทับกัน และ GrabCut สำหรับการดึง Foreground แบบ Iterative
5. **Detection (การตรวจจับ):** สร้าง Bounding Box จากการวิเคราะห์ Contour และกรองขนาดด้วย Area Filtering

---

## 📂 ประเภทของกรณีทดสอบ

### 1. ✅ กรณีที่ง่ายและสำเร็จ (E1–E10)
กรณีที่วัตถุมีสีสด ตัดกับพื้นหลังอย่างชัดเจน
- **ผลลัพธ์:** แม่นยำใกล้เคียง 100% ด้วย **HSV Color Masking** แบบง่าย
- **ตัวอย่าง:** แอปเปิ้ลสีแดง, กล้วย, บาสเก็ตบอล, ลูกเทนนิส

### 2. ✅ กรณีที่ยากแต่สำเร็จ (D1–D10)
กรณีที่มีความท้าทาย เช่น Contrast ต่ำ วัตถุใกล้เคียงกับพื้นหลัง หรือวัตถุทับซ้อนกัน
- **ผลลัพธ์:** สำเร็จได้โดยการผสมเทคนิคหลายอย่างเข้าด้วยกัน เช่น **Distance Transform + Watershed** สำหรับมะนาว หรือ **LAB Fusion + GrabCut** สำหรับหมีขั้วโลก
- **ตัวอย่าง:** หมีขั้วโลกในหิมะ, กิ้งก่าพรางตัว, ขวดแก้วใส

### 3. ❌ กรณีที่ล้มเหลวตามคาด (FE1–FE10)
กรณีที่วัตถุกลืนหายไปกับสิ่งแวดล้อมอย่างสมบูรณ์แบบ (Evolutionary Camouflage) — ตั้งใจให้ล้มเหลว
- **ผลลัพธ์:** ล้มเหลวตามเป้าหมาย แสดงให้เห็นว่าการประมวลผลในระดับ Pixel ไม่เพียงพอ หากไม่มี Semantic Knowledge หรือข้อมูล 3D (Depth)
- **ตัวอย่าง:** ชุดทหารลายพราง, ปลาแบน (Flounder), แมลงกิ่งไม้

### 4. ❌ กรณีที่ล้มเหลวโดยไม่คาดคิด (FU1–FU10)
กรณีที่ตามหลักการควรแยกได้ แต่กลับล้มเหลวเพราะ Noise จากสภาพแวดล้อม
- **ผลลัพธ์:** ล้มเหลวเชิงเทคนิค เช่น เกิดจาก **Specular Highlights** (แก้วกาแฟ), **Wood Grain Noise** (เหรียญ), หรือ **Water Scattering** (ใต้น้ำ)
- **ข้อสังเกต:** แสดงให้เห็นว่า Threshold ที่ต้อง Tune มือนั้นเปราะบางมากต่อความแปรปรวนของแสงในโลกจริง

---

## 🔍 วิเคราะห์เชิงลึก: รายละเอียดเทคนิคของทั้ง 40 กรณี

### 🟢 กลุ่มที่ 1: ง่ายและสำเร็จ (E1–E10)
**แนวทางหลัก:** High-Saturation Color Masking — กรณีเหล่านี้ได้เปรียบตรงที่วัตถุมีสีสดโดดเด่น แยกออกจากพื้นหลังได้ง่าย

*   **E1 — แอปเปิ้ลแดง:** ใช้ HSV Dual-Band Masking จับทั้งสองขอบของช่วงสีแดงใน Hue ความ Contrast สูงกับพื้นหลังสีขาวทำให้ Contour Detection แม่นยำมาก
    ![E1](output/E1.png)
*   **E2 — กล้วย:** อาศัยสีเหลืองสดสว่างของกล้วย ใช้ Morphological Closing เพื่อเชื่อมช่องว่างเล็กน้อยที่เกิดจาก Specular Highlight บนผิวกล้วย
    ![E2](output/E2.png)
*   **E3 — บาสเก็ตบอล:** สีส้มของบาสเก็ตบอลมีความเฉพาะเจาะจงมาก ใช้ Hue Band แคบๆ แยกออกจากพื้นสนามหรือพื้นหลังสีกลางได้ง่าย
    ![E3](output/E3.png)
*   **E4 — ลูกเทนนิส:** สีเหลือง-เขียว Fluorescent (Hue 30-80) มีช่วงความยาวคลื่นที่เฉพาะ แทบไม่ปรากฏในพื้นหลังธรรมชาติ ทำให้ได้ Mask ที่สะอาดมาก
    ![E4](output/E4.png)
*   **E5 — ถังดับเพลิง:** แยกสีแดงเข้มอุตสาหกรรมออกจากผนังสีอ่อนด้วย HSV Range ที่แข็งแกร่ง เสริมด้วย Area Filtering กรอง Noise ออก
    ![E5](output/E5.png)
*   **E6 — เมฆ:** เมฆเป็น Achromatic (ไม่มีสี) จึงใช้ Color Masking ปกติไม่ได้ แก้ด้วยวิธี Otsu's Thresholding บนภาพ Grayscale เพื่อแยกเมฆขาวสว่างออกจากฟ้าสีเข้มกว่า
    ![E6](output/E6.png)
*   **E7 — ฟักทอง:** สีส้มสดบนพื้นหลังมืด ใช้ Morphological Opening เพื่อตัดก้านสีเขียนออก เหลือแค่ส่วนผลไม้
    ![E7](output/E7.png)
*   **E8 — กุหลาบ:** แยก Hue สีชมพู/Magenta ออกจากใบสีเขียวรอบๆ ได้สำเร็จด้วยการกำหนด Hue Range ให้แม่น
    ![E8](output/E8.png)
*   **E9 — แตงโม:** โฟกัสที่การ Segment เนื้อสีแดงข้างใน ใช้ Red Mask เพื่อข้ามผ่านเปลือกสีเขียวและสิ่งรอบข้าง
    ![E9](output/E9.png)
*   **E10 — ไฟจราจร:** โฟกัสที่สัญญาณไฟสีเขียว Intensity สูง กรองด้วยค่า Saturation และ Value สูงพร้อมกัน เพื่อ Isolate หลอดไฟที่กำลังติดอยู่ได้แม่นยำ
    ![E10](output/E10.png)

---

### 🟡 กลุ่มที่ 2: ยากแต่สำเร็จ (D1–D10)
**แนวทางหลัก:** Advanced Preprocessing & Multi-stage Algorithms — กรณีเหล่านี้มีความท้าทาย เช่น Contrast ต่ำ วัตถุซ้อนทับกัน หรือ Color Cue ที่จาง

*   **D1 — จานสีขาว:** ปัญหา White-on-White ความต่างของ Intensity เล็กน้อยที่ขอบจาน ถูกจับได้ด้วย Canny แล้วส่งต่อให้ GrabCut ดึงจานออกจากผ้าปูโต๊ะสีขาว
    ![D1](output/D1.png)
*   **D2 — เห็ด:** เห็ดสีน้ำตาลบนพื้นป่า ใช้ LAB a-channel (แกนแดง-เขียว) แยก Tone อุ่นของเห็ดออกจากดินและใบไม้ที่มีโทนเย็นกว่า
    ![D2](output/D2.png)
*   **D3 — มะนาว:** ปัญหาวัตถุชนกัน ใช้ Distance Transform เพื่อหา "ยอด" ของมะนาวแต่ละลูก แล้วใช้เป็น Seed ให้ Watershed Algorithm แบ่งขอบที่ทับกันออก
    ![D3](output/D3.png)
*   **D4 — หมีขั้วโลก:** ขนขาวบนหิมะขาว ใช้ CLAHE ขยาย Texture ของขน และ LAB b-channel แยก Tone อุ่นของหมีออกจากหิมะที่มีโทนเย็น-ฟ้า
    ![D4](output/D4.png)
*   **D5 — กิ้งก่า:** พระเอกด้านพรางตัว ใช้ Sobel Gradient Magnitude เพื่อตรวจจับ Texture ความถี่สูงของเกล็ด ซึ่งแตกต่างจาก Texture เรียบๆ ของใบไม้
    ![D5](output/D5.png)
*   **D6 — แมวในพื้นหลังรก:** ใช้ Bilateral Filtering ลด Noise พื้นหลังโดยยังรักษาขอบของแมวไว้ ทำให้ Canny สามารถหา Silhouette ที่ต่อเนื่องได้
    ![D6](output/D6.png)
*   **D7 — ม้าลาย:** ลายทาง Contrast สูง ใช้ Sobel Operator จับ Gradient สนาม Density สูงของลายขาว-ดำ เพื่อระบุตำแหน่งของสัตว์
    ![D7](output/D7.png)
*   **D8 — ป้ายจราจร:** ป้ายถูกบดบังด้วยกิ่งไม้บางส่วน หลัง HSV Yellow Masking ใช้ Morphological Closing ขนาดใหญ่ "ซ่อม" รูปทรงป้ายที่หายไปข้างหลังกิ่ง
    ![D8](output/D8.png)
*   **D9 — ภูเขาในหมอก:** Contrast ต่ำมากเพราะหมอกหนา ใช้ CLAHE ทำ Dehazing เพื่อปรับ Brightness เฉพาะพื้นที่ ทำให้ขอบฟ้าของภูเขาจับได้ด้วย Canny
    ![D9](output/D9.png)
*   **D10 — ขวดแก้ว:** ปัญหา Transparency ใช้ GrabCut ในโหมด Rectangle เพื่อ Iterate จากรูปแบบการหักเหของแสงที่ขอบขวด จนได้ Foreground Mask ที่แน่นหนา
    ![D10](output/D10.png)

---

### 🔴 กลุ่มที่ 3: ล้มเหลวตามคาด (FE1–FE10)
**แนวทางหลัก:** Evolutionary Camouflage (ไม่มี Signal เลย) — วัตถุเหล่านี้วิวัฒนาการมาเพื่อให้มีสี ขอบ และ Texture เหมือนกับสิ่งแวดล้อมอย่างสมบูรณ์แบบ

*   **FE1 — ทหารลายพราง:** ลวดลายออกแบบมาเพื่อทำลาย Silhouette และจับคู่กับความถี่ของสีและ Texture ในป่า ไม่มี Signal เหลือสำหรับ Traditional CV เลย
    ![FE1](output/FE1.png)
*   **FE2 — นกฮูกพรางตัว:** ขนของนกฮูกเลียนแบบลาย Texture ของเปลือกไม้ได้สมบูรณ์แบบมาก แม้แต่ Gabor Texture Analysis ก็มองเห็นภาพทั้งภาพเป็น Texture เดียวกัน
    ![FE2](output/FE2.png)
*   **FE3 — ปลาในปะการัง:** สีสันของปลาแม้จะสวยงามแต่กระจัดกระจาย ตรงกับปะการังรอบๆ Algorithm แตกปลาออกเป็น Blob สีเล็กๆ มากมาย แทนที่จะเป็นวัตถุชิ้นเดียว
    ![FE3](output/FE3.png)
*   **FE4 — แมลงกิ่งไม้:** ทั้งสีและ "รูปทรงเรขาคณิต" เลียนแบบกิ่งไม้ ไม่มีความแตกต่างที่แยกแยะได้ระหว่างขาของแมลงกับกิ่งไม้รอบๆ
    ![FE4](output/FE4.png)
*   **FE5 — ร่มสีแดงในกองมะเขือเทศ:** Hue, Saturation และ Value เหมือนกันหมด Algorithm รวมร่มเข้ากับมะเขือเทศโดยแยกไม่ออก
    ![FE5](output/FE5.png)
*   **FE6 — ชุดสีผิว:** สีของชุดตรงกับโทนผิวของผู้สวมใส่พอดี ทุก Color Space ไม่มีขอบที่ตรวจจับได้ระหว่างคนและเสื้อผ้าเลย
    ![FE6](output/FE6.png)
*   **FE7 — ฉากกลางคืน:** Underexposure รุนแรงมาก ข้อมูลภาพอยู่ที่ "Noise Floor" (ความเข้มใกล้ศูนย์) ไม่มีวิธีการ Enhancement ใดกู้รูปทรงวัตถุกลับมาได้
    ![FE7](output/FE7.png)
*   **FE8 — เต่าทะเล:** ลวดลายบนกระดองเต่าเลียนแบบ Texture ของปะการัง Canny ตรวจจับลาย Chaotic ของปะการังและกระดองเป็น Edge Field ต่อเนื่องกัน
    ![FE8](output/FE8.png)
*   **FE9 — ปลาแบน Flounder:** ปลาแบนที่ฝังตัวอยู่ในทราย เลียนแบบสีและ Granularity ของทรายได้สมบูรณ์แบบ จนกลายเป็นส่วนหนึ่งของพื้นหลัง
    ![FE9](output/FE9.png)
*   **FE10 — ฝูงชนหนาแน่น:** มนุษย์ที่ทับซ้อนกันในกลุ่มแน่น หากไม่มี Semantic Understanding ว่า "ร่างกายมนุษย์" คืออะไร Traditional CV จะมองเห็นเป็นก้อน Texture ขนาดใหญ่ก้อนเดียว
    ![FE10](output/FE10.png)

---

### 🟠 กลุ่มที่ 4: ล้มเหลวโดยไม่คาดคิด (FU1–FU10)
**แนวทางหลัก:** Environmental Noise & Global Threshold Failure — กรณีเหล่านี้ตามหลักการควรทำได้ แต่กลับพ่ายแพ้ให้กับ Artifacts ของแสงหรือวัสดุโดยเฉพาะ

*   **FU1 — ไข่ต้มสุก:** ไข่ขาวบนจานขาว แสงสว่างคุณภาพสูงทำให้ไม่มี Shadow ที่ขอบ Pixel ขอบมี Intensity เหมือนกันหมด ไม่มี Intensity Gradient เหลือเลย
    ![FU1](output/FU1.png)
*   **FU2 — เหรียญบนไม้:** Reflection บนผิวโลหะ เหรียญสะท้อน Grain ของไม้โต๊ะ ทำให้มันดู "โปร่งใส" ต่อ Algorithm
    ![FU2](output/FU2.png)
*   **FU3 — แก้วกาแฟสีขาว:** Specular Highlights จากแสงไฟที่สะท้อนบนผิวเซรามิก สว่างกว่าตัวแก้วเองด้วยซ้ำ ทำให้เกิด "รู" ใน Mask ตรงจุดที่แสงกระทบ
    ![FU3](output/FU3.png)
*   **FU4 — แมวดำบนโซฟาดำ:** ทั้งคู่อยู่ในช่วง Intensity ต่ำสุด 5% (Value 0-20) Contrast Enhancement ล้มเหลวเพราะข้อมูลทั้งหมดอัดแน่นอยู่ในช่วงแคบมาก
    ![FU4](output/FU4.png)
*   **FU5 — ขนมปังทั้งก้อน:** ขนมปังสีน้ำตาลบนกระดานสีน้ำตาล เงาที่ขนมปังทอดทิ้งมีขอบคมกว่าขนมปังเอง ทำให้ Canny วิ่งตามเงาแทนที่จะวิ่งตามก้อนขนมปัง
    ![FU5](output/FU5.png)
*   **FU6 — นกฮูกหิมะ:** คล้ายหมีขั้วโลก แต่ขนอ่อนนุ่มกว่า (Downy Feather) ทำให้ Canny หาวง Closed Loop ของ Silhouette ร่างกายไม่ได้
    ![FU6](output/FU6.png)
*   **FU7 — ควันโรงงาน:** Transparency และ Diffusion ควันไม่มี "ขอบ" ที่ชัดเจน ขอบมันคือ Gradient ของความโปร่งใสที่ค่อยๆ กลืนหายไปกับพื้นหลังเมฆ
    ![FU7](output/FU7.png)
*   **FU8 — เก้าอี้ไม้:** Material Match ไม้ของเก้าอี้และพื้นมี Grain Direction และสีเหมือนกัน ทำให้ Morphological Operations รวมทั้งสองเข้าด้วยกัน
    ![FU8](output/FU8.png)
*   **FU9 — ใต้น้ำ:** Light Scattering (Haze) น้ำทำหน้าที่เป็น Blue-Green Filter และ Diffuser ทำลาย Contrast และทำให้ขอบต่างๆ เบลอจนตรวจจับไม่ได้
    ![FU9](output/FU9.png)
*   **FU10 — แมวสีส้ม:** Color และ Texture ทับซ้อนกัน ลายของแมวตรงกับทิศทางการทอผ้าของโซฟา ทำให้ Gabor และ Edge Analysis เห็นเป็น Texture plane เดียวกัน
    ![FU10](output/FU10.png)

---

## 📊 ตารางสรุป: 40 กรณีทดสอบ

| # | กรณี | เทคนิคที่ใช้ | พารามิเตอร์สำคัญ |
|---|------|-------------|-------------------|
| **E1** | แอปเปิ้ลแดง | HSV Dual-Band Red + Morphology | Hue[0-10, 160-180], Sat[120-255] |
| **E2** | กล้วย | HSV Yellow Range + Morphology | Hue[20-35], Sat[105-255] |
| **E3** | บาสเก็ตบอล | HSV Orange Range + Morphology | Hue[5-25], Sat[180-255] |
| **E4** | ลูกเทนนิส | HSV Yellow-Green + Morphology | Hue[30-80], Sat[60-255] |
| **E5** | ถังดับเพลิง | HSV Dual-Band Red + Large Morphology | Hue[0-12, 158-180], Sat[120-255] |
| **E6** | เมฆ | Grayscale + Gaussian Blur + Otsu | Blur(5x5), Otsu auto |
| **E7** | ฟักทอง | HSV Orange + Morphological Open/Close | Hue[6-18], Sat[130-255] |
| **E8** | กุหลาบ | HSV Pink Range + Morphological Ops | Hue[170-175], Sat[120-200] |
| **E9** | แตงโม | HSV Dual-Band Red + Large Morph Close | Hue[0-12, 165-180] |
| **E10** | ไฟจราจร | HSV Bright Green (High Value) | Hue[45-90], Val[180-255] |
| **D1** | จานสีขาว | Canny Edge Detection + GrabCut Seed | Canny(40,110), 10 iterations |
| **D2** | เห็ด | LAB a-channel Threshold + Morphology | A > 145 (pink/brown) |
| **D3** | มะนาว | HSV + Distance Transform + Watershed | Dist Threshold(0.65xMax) |
| **D4** | หมีขั้วโลก | CLAHE + LAB Fusion + GrabCut Refinement | CLAHE(2.5), GrabCut(5 iter) |
| **D5** | กิ้งก่า | HSV Threshold + Sobel Gradient | Threshold(18) |
| **D6** | แมว (ซับซ้อน) | Canny + Contour Fill + Large Morphology | Canny(20,80), Close(15x15) |
| **D7** | ม้าลาย | Sobel Gradient Magnitude + Threshold | Threshold(52) |
| **D8** | ป้ายจราจร | HSV Yellow + Morphological Close | Hue[170-180], Sat[90-255] |
| **D9** | ภูเขาในหมอก | CLAHE + Low-Threshold Canny | Canny(15,50), CLAHE(4.0) |
| **D10** | ขวดแก้ว | Canny + Morph Close + GrabCut | 5 iterations, Rect mode |
| **FE1** | ทหารลายพราง | HSV + Canny | ล้มเหลว (Pattern matching bg) |
| **FE2** | นกฮูกพรางตัว | HSV + Canny + Gabor Texture | ล้มเหลว (Texture mimicry) |
| **FE3** | ปลาในปะการัง | RGB / HSV Colorphase Masking | ล้มเหลว (Complex background) |
| **FE4** | แมลงกิ่งไม้ | HSV + Gabor Filter Bank | ล้มเหลว (Shape/Texture) |
| **FE5** | ร่มสีแดง | HSV Red Band Masking | ล้มเหลว (Color mimicry) |
| **FE6** | ชุดสีผิว | HSV Skin-Tone Range | ล้มเหลว (Color matching skin) |
| **FE7** | ฉากกลางคืน | Low-Intensity Thresholding | ล้มเหลว (Extreme darkness) |
| **FE8** | เต่าทะเล | HSV + Gabor Texture Energy | ล้มเหลว (Pattern overlap) |
| **FE9** | ปลาแบน Flounder | Gabor Texture + HSV | ล้มเหลว (Perfect matching) |
| **FE10** | ฝูงชนหนาแน่น | Canny Edge Density | ล้มเหลว (Semantic overlap) |
| **FU1** | ไข่ต้มสุก | HSV + Canny + LAB + Adaptive | ล้มเหลว (White on White) |
| **FU2** | เหรียญบนไม้ | CLAHE + Canny + Morphology | ล้มเหลว (Wood Grain Noise) |
| **FU3** | แก้วกาแฟสีขาว | Canny + Otsu + Adaptive | ล้มเหลว (Specular Highlights) |
| **FU4** | แมวดำ | Dark HSV + Canny + Gabor + CLAHE | ล้มเหลว (Same Dark Color) |
| **FU5** | ขนมปังทั้งก้อน | Brown HSV Mask + Canny | ล้มเหลว (Shadow/Grain) |
| **FU6** | นกฮูกหิมะ | HSV + LAB b-ch + Canny + Gabor | ล้มเหลว (White on White) |
| **FU7** | ควันโรงงาน | HSV + Gabor + Sobel Gradient | ล้มเหลว (Transparency/Clouds) |
| **FU8** | เก้าอี้ไม้ | HSV + Canny + LAB + Adaptive | ล้มเหลว (Material/Texture match) |
| **FU9** | ใต้น้ำ | LAB + Canny + HSV + CLAHE | ล้มเหลว (Scattering/Cast) |
| **FU10** | แมวสีส้ม | HSV + Sobel + Canny + LAB | ล้มเหลว (Color/Texture overlap) |

---

### 💡 สิ่งที่เรียนรู้จากการทดลองนี้

1. **วิธีอิงสี (HSV)** ได้ผลดีที่สุดเมื่อวัตถุมี *Hue ที่เฉพาะและสดใส* แตกต่างจากพื้นหลังอย่างชัดเจน
2. **วิธีอิงขอบ (Canny, Sobel)** ได้ผลเมื่อวัตถุมี *Intensity Discontinuity* ที่ชัดเจนที่ขอบ
3. **Morphological Operations** เป็น Post-processor ที่สำคัญมาก ช่วยเชื่อมช่องว่างและกำจัด Noise
4. **Watershed + Distance Transform** แยกวัตถุที่ชนกันหรือทับซ้อนกันและมีสีเดียวกันออกจากกันได้
5. **CLAHE** ช่วยกู้ภาพที่ Contrast ต่ำหรือมีหมอกได้ โดยการ Normalize ความสว่างแบบ Local
6. **Gabor Filter** ตรวจจับขอบที่อิง Texture ได้ในกรณีที่ข้อมูลสีไม่เพียงพอ
7. **LAB Color Space** แยกสีที่ต่างกันในเชิง Perceptual ได้ดีกว่า HSV/RGB ในบางกรณี
8. **ข้อจำกัดพื้นฐานของ Traditional Method** คือ ขาด Semantic Understanding — ไม่สามารถ "เข้าใจ" ได้ว่าวัตถุในภาพคืออะไรหรืออยู่ที่ไหนในเชิงความหมาย
