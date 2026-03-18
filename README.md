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
**แนวทางหลัก:** กรณีนี้นักศึกษาสามารถใช้วิธีทาง Color, Image Thresholding หรือ Edge detection ร่วมกับ Morphological Operations เพื่อตรวจจับวัตถุได้โดยง่าย

#### E1 – Red Apple on White Background
- **เครดิตที่มาของภาพ (Source):** [คลิกเพื่อดูภาพจากแแหล่งที่มา](https://share.google/74CVNllQzVKbdnTBs)
- **วัตถุเป้าหมายที่ต้องการตรวจจับ:** Red Apple on White Background
- **เทคนิคและพารามิเตอร์ที่ใช้:** HSV Color Thresholding + Morphology + Contour
- **รูปภาพอินพุต (Input) และ รูปภาพเอาต์พุต (Output):**
  > **หมายเหตุ:** เอาต์พุตที่แสดงด้านล่างเป็นการรวมภาพ Input ดั้งเดิมไว้ในหน้าต่างเดียวกันทั้งหมดเพื่อเปรียบเทียบผลลัพธ์การทำ Object Segmentation แล้ว
  ![E1](output/E1.png)

#### E2 – Yellow Banana on White/Light Background
- **เครดิตที่มาของภาพ (Source):** [คลิกเพื่อดูภาพจากแแหล่งที่มา](https://share.google/ISvh2g2j8AdKsbL3J)
- **วัตถุเป้าหมายที่ต้องการตรวจจับ:** Yellow Banana on White/Light Background
- **เทคนิคและพารามิเตอร์ที่ใช้:** HSV Color Thresholding + Morphology + Contour
- **รูปภาพอินพุต (Input) และ รูปภาพเอาต์พุต (Output):**
  > **หมายเหตุ:** เอาต์พุตที่แสดงด้านล่างเป็นการรวมภาพ Input ดั้งเดิมไว้ในหน้าต่างเดียวกันทั้งหมดเพื่อเปรียบเทียบผลลัพธ์การทำ Object Segmentation แล้ว
  ![E2](output/E2.png)

#### E3 – Orange Basketball on White Background
- **เครดิตที่มาของภาพ (Source):** [คลิกเพื่อดูภาพจากแแหล่งที่มา](https://share.google/jlWQTePSda25oOeFM)
- **วัตถุเป้าหมายที่ต้องการตรวจจับ:** Orange Basketball on White Background
- **เทคนิคและพารามิเตอร์ที่ใช้:** HSV Color Thresholding + Morphology + Contour
- **รูปภาพอินพุต (Input) และ รูปภาพเอาต์พุต (Output):**
  > **หมายเหตุ:** เอาต์พุตที่แสดงด้านล่างเป็นการรวมภาพ Input ดั้งเดิมไว้ในหน้าต่างเดียวกันทั้งหมดเพื่อเปรียบเทียบผลลัพธ์การทำ Object Segmentation แล้ว
  ![E3](output/E3.png)

#### E4 – (Improved): Tennis Ball Segmentation
- **เครดิตที่มาของภาพ (Source):** [คลิกเพื่อดูภาพจากแแหล่งที่มา](https://share.google/F2xmjs120eakDCnYS)
- **วัตถุเป้าหมายที่ต้องการตรวจจับ:** (Improved): Tennis Ball Segmentation
- **เทคนิคและพารามิเตอร์ที่ใช้:** HSV + Blur + Hole Filling + Largest Component
- **รูปภาพอินพุต (Input) และ รูปภาพเอาต์พุต (Output):**
  > **หมายเหตุ:** เอาต์พุตที่แสดงด้านล่างเป็นการรวมภาพ Input ดั้งเดิมไว้ในหน้าต่างเดียวกันทั้งหมดเพื่อเปรียบเทียบผลลัพธ์การทำ Object Segmentation แล้ว
  ![E4](output/E4.png)

#### E5 – Red Fire Extinguisher on Light Background
- **เครดิตที่มาของภาพ (Source):** [คลิกเพื่อดูภาพจากแแหล่งที่มา](https://pin.it/7KnUPlNKW)
- **วัตถุเป้าหมายที่ต้องการตรวจจับ:** Red Fire Extinguisher on Light Background
- **เทคนิคและพารามิเตอร์ที่ใช้:** HSV Dual-Band Red Threshold + Morphology + Contour
- **รูปภาพอินพุต (Input) และ รูปภาพเอาต์พุต (Output):**
  > **หมายเหตุ:** เอาต์พุตที่แสดงด้านล่างเป็นการรวมภาพ Input ดั้งเดิมไว้ในหน้าต่างเดียวกันทั้งหมดเพื่อเปรียบเทียบผลลัพธ์การทำ Object Segmentation แล้ว
  ![E5](output/E5.png)

#### E6 – Blue Sky — Detect Clouds (White Region)
- **เครดิตที่มาของภาพ (Source):** [คลิกเพื่อดูภาพจากแแหล่งที่มา](https://pin.it/5pd8jz05R)
- **วัตถุเป้าหมายที่ต้องการตรวจจับ:** Blue Sky — Detect Clouds (White Region)
- **เทคนิคและพารามิเตอร์ที่ใช้:** Grayscale + Gaussian Blur + Otsu's Threshold + Morphology
- **รูปภาพอินพุต (Input) และ รูปภาพเอาต์พุต (Output):**
  > **หมายเหตุ:** เอาต์พุตที่แสดงด้านล่างเป็นการรวมภาพ Input ดั้งเดิมไว้ในหน้าต่างเดียวกันทั้งหมดเพื่อเปรียบเทียบผลลัพธ์การทำ Object Segmentation แล้ว
  ![E6](output/E6.png)

#### E7 – Orange Pumpkin on Dark Background
- **เครดิตที่มาของภาพ (Source):** [คลิกเพื่อดูภาพจากแแหล่งที่มา](https://share.google/1os9xZLPmsSH5waUx)
- **วัตถุเป้าหมายที่ต้องการตรวจจับ:** Orange Pumpkin on Dark Background
- **เทคนิคและพารามิเตอร์ที่ใช้:** HSV Color Thresholding + Morphology + Contour
- **รูปภาพอินพุต (Input) และ รูปภาพเอาต์พุต (Output):**
  > **หมายเหตุ:** เอาต์พุตที่แสดงด้านล่างเป็นการรวมภาพ Input ดั้งเดิมไว้ในหน้าต่างเดียวกันทั้งหมดเพื่อเปรียบเทียบผลลัพธ์การทำ Object Segmentation แล้ว
  ![E7](output/E7.png)

#### E8 – Pink/Red Rose on Dark Background
- **เครดิตที่มาของภาพ (Source):** [คลิกเพื่อดูภาพจากแแหล่งที่มา](https://share.google/yOfLQPJVZJewNaQRr)
- **วัตถุเป้าหมายที่ต้องการตรวจจับ:** Pink/Red Rose on Dark Background
- **เทคนิคและพารามิเตอร์ที่ใช้:** HSV Color Thresholding + Morphology + Contour
- **รูปภาพอินพุต (Input) และ รูปภาพเอาต์พุต (Output):**
  > **หมายเหตุ:** เอาต์พุตที่แสดงด้านล่างเป็นการรวมภาพ Input ดั้งเดิมไว้ในหน้าต่างเดียวกันทั้งหมดเพื่อเปรียบเทียบผลลัพธ์การทำ Object Segmentation แล้ว
  ![E8](output/E8.png)

#### E9 – Watermelon (Green External, Red Internal) — Segment Red Part
- **เครดิตที่มาของภาพ (Source):** [คลิกเพื่อดูภาพจากแแหล่งที่มา](https://share.google/5UtSq3gco7WGckMiq)
- **วัตถุเป้าหมายที่ต้องการตรวจจับ:** Watermelon (Green External, Red Internal) — Segment Red Part
- **เทคนิคและพารามิเตอร์ที่ใช้:** HSV Dual-Band Red Threshold + Morphology + Contour
- **รูปภาพอินพุต (Input) และ รูปภาพเอาต์พุต (Output):**
  > **หมายเหตุ:** เอาต์พุตที่แสดงด้านล่างเป็นการรวมภาพ Input ดั้งเดิมไว้ในหน้าต่างเดียวกันทั้งหมดเพื่อเปรียบเทียบผลลัพธ์การทำ Object Segmentation แล้ว
  ![E9](output/E9.png)

#### E10 – Green Traffic Light Signal
- **เครดิตที่มาของภาพ (Source):** [คลิกเพื่อดูภาพจากแแหล่งที่มา](https://share.google/NEttFWjaG8fqE4jtF)
- **วัตถุเป้าหมายที่ต้องการตรวจจับ:** Green Traffic Light Signal
- **เทคนิคและพารามิเตอร์ที่ใช้:** HSV Bright Green Thresholding + Morphology + Contour
- **รูปภาพอินพุต (Input) และ รูปภาพเอาต์พุต (Output):**
  > **หมายเหตุ:** เอาต์พุตที่แสดงด้านล่างเป็นการรวมภาพ Input ดั้งเดิมไว้ในหน้าต่างเดียวกันทั้งหมดเพื่อเปรียบเทียบผลลัพธ์การทำ Object Segmentation แล้ว
  ![E10](output/E10.png)

### 🟡 กลุ่มที่ 2: ยากแต่สำเร็จ (D1–D10)
**แนวทางหลัก:** กรณีเหล่านี้มีความท้าทายจาก Contrast ต่ำ หรือความคล้ายกับพื้นหลัง แต่สามารถแก้ได้ด้วยการใช้ Advanced Filters, Fusion หรือ Iterative Segmentation (GrabCut, Watershed)

#### D1 – White Plate on a High-Brightness Tablecloth
- **เครดิตที่มาของภาพ (Source):** [คลิกเพื่อดูภาพจากแแหล่งที่มา](https://share.google/ZepDCx9BEFsNNx0wv)
- **วัตถุเป้าหมายที่ต้องการตรวจจับ:** White Plate on a High-Brightness Tablecloth
- **เทคนิคและพารามิเตอร์ที่ใช้:** Bilateral Filter, Canny Edge Detection, Hough Circle Transform, GrabCut Refinement
- **ความยากที่พบ (Difficulty):** Low Contrast, Hard Color Discrimination, Bright Background Blends with Object
- **รูปภาพอินพุต (Input) และ รูปภาพเอาต์พุต (Output):**
  > **หมายเหตุ:** เอาต์พุตที่แสดงด้านล่างเป็นการรวมภาพ Input ดั้งเดิมไว้ในหน้าต่างเดียวกันทั้งหมดเพื่อเปรียบเทียบผลลัพธ์การทำ Object Segmentation แล้ว
  ![D1](output/D1.png)

#### D2 – Brown Mushroom on Forest Floor
- **เครดิตที่มาของภาพ (Source):** [คลิกเพื่อดูภาพจากแแหล่งที่มา](https://share.google/9n7sXo5h3a8qj1v0C)
- **วัตถุเป้าหมายที่ต้องการตรวจจับ:** Brown Mushroom on Forest Floor
- **เทคนิคและพารามิเตอร์ที่ใช้:** LAB Color Thresholding + Morphology + Contour Scoring
- **ความยากที่พบ (Difficulty):** Low Contrast, Complex Background, Multiple Similar-Colored Objects
- **รูปภาพอินพุต (Input) และ รูปภาพเอาต์พุต (Output):**
  > **หมายเหตุ:** เอาต์พุตที่แสดงด้านล่างเป็นการรวมภาพ Input ดั้งเดิมไว้ในหน้าต่างเดียวกันทั้งหมดเพื่อเปรียบเทียบผลลัพธ์การทำ Object Segmentation แล้ว
  ![D2](output/D2.png)

#### D3 – Lemon Segmentation (Traditional Approach)
- **เครดิตที่มาของภาพ (Source):** [คลิกเพื่อดูภาพจากแแหล่งที่มา](https://share.google/d5VEueqApN872IKVPg)
- **วัตถุเป้าหมายที่ต้องการตรวจจับ:** Lemon Segmentation (Traditional Approach)
- **เทคนิคและพารามิเตอร์ที่ใช้:** HSV Color Masking, Distance Transform, Watershed Segmentation
- **ความยากที่พบ (Difficulty):** Overlapping Objects, Specular Highlights, Similar Colors Between Instances
- **รูปภาพอินพุต (Input) และ รูปภาพเอาต์พุต (Output):**
  > **หมายเหตุ:** เอาต์พุตที่แสดงด้านล่างเป็นการรวมภาพ Input ดั้งเดิมไว้ในหน้าต่างเดียวกันทั้งหมดเพื่อเปรียบเทียบผลลัพธ์การทำ Object Segmentation แล้ว
  ![D3](output/D3.png)

#### D4 – Polar Bear in Snow
- **เครดิตที่มาของภาพ (Source):** [คลิกเพื่อดูภาพจากแแหล่งที่มา](https://share.google/rzxLTsiFlIdtgwTL8)
- **วัตถุเป้าหมายที่ต้องการตรวจจับ:** Polar Bear in Snow
- **เทคนิคและพารามิเตอร์ที่ใช้:** Gaussian Blur + Otsu, Low-threshold Canny, LAB b-channel + Morphological Refinement
- **ความยากที่พบ (Difficulty):** Near-zero color/intensity difference between white fur and white snow
- **รูปภาพอินพุต (Input) และ รูปภาพเอาต์พุต (Output):**
  > **หมายเหตุ:** เอาต์พุตที่แสดงด้านล่างเป็นการรวมภาพ Input ดั้งเดิมไว้ในหน้าต่างเดียวกันทั้งหมดเพื่อเปรียบเทียบผลลัพธ์การทำ Object Segmentation แล้ว
  ![D4](output/D4.png)

#### D5 – Camouflaged Chameleon on Leafy Background
- **เครดิตที่มาของภาพ (Source):** [คลิกเพื่อดูภาพจากแแหล่งที่มา](https://share.google/B2vSogKAA37Rc7quN)
- **วัตถุเป้าหมายที่ต้องการตรวจจับ:** Camouflaged Chameleon on Leafy Background
- **เทคนิคและพารามิเตอร์ที่ใช้:** Branch-Color Exclusion + LAB Sobel Gradient + Saturation-Weighted Canny
- **ความยากที่พบ (Difficulty):** Highly Camouflaged Pattern Closely Matching the Leafy Background,
- **รูปภาพอินพุต (Input) และ รูปภาพเอาต์พุต (Output):**
  > **หมายเหตุ:** เอาต์พุตที่แสดงด้านล่างเป็นการรวมภาพ Input ดั้งเดิมไว้ในหน้าต่างเดียวกันทั้งหมดเพื่อเปรียบเทียบผลลัพธ์การทำ Object Segmentation แล้ว
  ![D5](output/D5.png)

#### D6 – Cat on Complex Background
- **เครดิตที่มาของภาพ (Source):** [คลิกเพื่อดูภาพจากแแหล่งที่มา](https://share.google/qivBOHSNhPhA88eA6)
- **วัตถุเป้าหมายที่ต้องการตรวจจับ:** Cat on Complex Background
- **เทคนิคและพารามิเตอร์ที่ใช้:** Gaussian Blur + Canny Edge Detection + Contour Fill + Morphological Closing
- **ความยากที่พบ (Difficulty):** Fur Texture Blends with Background, Color Thresholding Unreliable
- **รูปภาพอินพุต (Input) และ รูปภาพเอาต์พุต (Output):**
  > **หมายเหตุ:** เอาต์พุตที่แสดงด้านล่างเป็นการรวมภาพ Input ดั้งเดิมไว้ในหน้าต่างเดียวกันทั้งหมดเพื่อเปรียบเทียบผลลัพธ์การทำ Object Segmentation แล้ว
  ![D6](output/D6.png)

#### D7 – Zebra on Natural Background
- **เครดิตที่มาของภาพ (Source):** [คลิกเพื่อดูภาพจากแแหล่งที่มา](https://pin.it/4XFJiQZDw)
- **วัตถุเป้าหมายที่ต้องการตรวจจับ:** Zebra on Natural Background
- **เทคนิคและพารามิเตอร์ที่ใช้:** Sobel Gradient Magnitude + Morphology + Geometric Candidate Scoring
- **ความยากที่พบ (Difficulty):** Black-and-White Stripes Blend with Sky and Grass Textures,
- **รูปภาพอินพุต (Input) และ รูปภาพเอาต์พุต (Output):**
  > **หมายเหตุ:** เอาต์พุตที่แสดงด้านล่างเป็นการรวมภาพ Input ดั้งเดิมไว้ในหน้าต่างเดียวกันทั้งหมดเพื่อเปรียบเทียบผลลัพธ์การทำ Object Segmentation แล้ว
  ![D7](output/D7.png)

#### D8 – Street Sign (Partially Occluded by Tree Branch)
- **เครดิตที่มาของภาพ (Source):** [คลิกเพื่อดูภาพจากแแหล่งที่มา](https://share.google/BUaYqUT3vca4MYiaP)
- **วัตถุเป้าหมายที่ต้องการตรวจจับ:** Street Sign (Partially Occluded by Tree Branch)
- **เทคนิคและพารามิเตอร์ที่ใช้:** HSV red segmentation + morphological reconstruction
- **ความยากที่พบ (Difficulty):** object partially occluded; irregular occlusion pattern
- **รูปภาพอินพุต (Input) และ รูปภาพเอาต์พุต (Output):**
  > **หมายเหตุ:** เอาต์พุตที่แสดงด้านล่างเป็นการรวมภาพ Input ดั้งเดิมไว้ในหน้าต่างเดียวกันทั้งหมดเพื่อเปรียบเทียบผลลัพธ์การทำ Object Segmentation แล้ว
  ![D8](output/D8.png)

#### D9 – Foggy Landscape — Detect Mountain/Horizon
- **เครดิตที่มาของภาพ (Source):** [คลิกเพื่อดูภาพจากแแหล่งที่มา](https://pin.it/5Zv5rxRCh)
- **วัตถุเป้าหมายที่ต้องการตรวจจับ:** Foggy Landscape — Detect Mountain/Horizon
- **เทคนิคและพารามิเตอร์ที่ใช้:** CLAHE + Canny with low thresholds + Probabilistic Hough
- **ความยากที่พบ (Difficulty):** low contrast due to fog, unclear boundaries
- **รูปภาพอินพุต (Input) และ รูปภาพเอาต์พุต (Output):**
  > **หมายเหตุ:** เอาต์พุตที่แสดงด้านล่างเป็นการรวมภาพ Input ดั้งเดิมไว้ในหน้าต่างเดียวกันทั้งหมดเพื่อเปรียบเทียบผลลัพธ์การทำ Object Segmentation แล้ว
  ![D9](output/D9.png)

#### D10 – Transparent Glass Bottle
- **เครดิตที่มาของภาพ (Source):** [คลิกเพื่อดูภาพจากแแหล่งที่มา](https://share.google/iNFBvuNFjRGHVs5OR)
- **วัตถุเป้าหมายที่ต้องการตรวจจับ:** Transparent Glass Bottle
- **เทคนิคและพารามิเตอร์ที่ใช้:** Canny + GrabCut (graph-cut, NOT ML — no model training)
- **ความยากที่พบ (Difficulty):** transparent object reflects surroundings; no clear color
- **รูปภาพอินพุต (Input) และ รูปภาพเอาต์พุต (Output):**
  > **หมายเหตุ:** เอาต์พุตที่แสดงด้านล่างเป็นการรวมภาพ Input ดั้งเดิมไว้ในหน้าต่างเดียวกันทั้งหมดเพื่อเปรียบเทียบผลลัพธ์การทำ Object Segmentation แล้ว
  ![D10](output/D10.png)

### 🔴 กลุ่มที่ 3: ล้มเหลวตามคาด (FE1–FE10)
**แนวทางหลัก:** กรณีที่วัตถุถูกออกแบบหรือวิวัฒนาการมาเพื่อพรางตัว (Evolutionary Camouflage) ทำให้วิธีระดับ Pixel ไม่มีทางทำได้สำเร็จ

#### FE1 – Military Camouflage Soldier
- **เครดิตที่มาของภาพ (Source):** [คลิกเพื่อดูภาพจากแแหล่งที่มา](https://share.google/2cmeFjak8cuMdqPm2)
- **วัตถุเป้าหมายที่ต้องการตรวจจับ:** Military Camouflage Soldier
- **การทดลองที่ได้ลองทำมาแล้ว:**
  - Attempt 1: HSV green-brown range (common camouflage colors)
  - Attempt 2: Canny edges
  - Attempt 3: Otsu thresholding
- **สาเหตุที่ไม่สำเร็จ:**  (ตามที่คาดไว้) camouflage is specifically designed to defeat color/texture detection
- **ความคิดเห็น (เทคนิคความสามารถพิเศษที่จำเป็น):** จำเป็นต้องใช้ความสามารถด้าน Semantic Understanding เช่น Model Deep Learning (CNN, Transformers) ที่ได้มีการฝึกสอน (Train) จากชุดข้อมูล เพื่อให้สามารถเข้าใจบริบทของภาพโดยรวมได้ (Context)
- **รูปภาพอินพุต (Input) และ รูปภาพเอาต์พุต (Output):**
  > **หมายเหตุ:** เอาต์พุตที่แสดงด้านล่างเป็นการรวมภาพ Input ดั้งเดิมไว้ในหน้าต่างเดียวกันทั้งหมดเพื่อเปรียบเทียบผลลัพธ์การทำ Object Segmentation แล้ว
  ![FE1](output/FE1.png)

#### FE2 – Camouflaged Owl on Tree Bark
- **เครดิตที่มาของภาพ (Source):** [คลิกเพื่อดูภาพจากแแหล่งที่มา](https://share.google/fWHlxeQOQuFpYvyUI)
- **วัตถุเป้าหมายที่ต้องการตรวจจับ:** Camouflaged Owl on Tree Bark
- **การทดลองที่ได้ลองทำมาแล้ว:**
  - Attempt 1: HSV brown-gray range (bark and owl feather tones)
  - Attempt 2: Canny edges
  - Attempt 3: Otsu thresholding
  - Attempt 4: Multi-Scale Gabor (bark vs feather texture)
- **สาเหตุที่ไม่สำเร็จ:**  (ตามที่คาดไว้) owl feather patterns evolved to mimic bark texture and color
- **ความคิดเห็น (เทคนิคความสามารถพิเศษที่จำเป็น):** จำเป็นต้องใช้ความสามารถด้าน Semantic Understanding เช่น Model Deep Learning (CNN, Transformers) ที่ได้มีการฝึกสอน (Train) จากชุดข้อมูล เพื่อให้สามารถเข้าใจบริบทของภาพโดยรวมได้ (Context)
- **รูปภาพอินพุต (Input) และ รูปภาพเอาต์พุต (Output):**
  > **หมายเหตุ:** เอาต์พุตที่แสดงด้านล่างเป็นการรวมภาพ Input ดั้งเดิมไว้ในหน้าต่างเดียวกันทั้งหมดเพื่อเปรียบเทียบผลลัพธ์การทำ Object Segmentation แล้ว
  ![FE2](output/FE2.png)

#### FE3 – Reef Fish School in Coral Aquarium
- **เครดิตที่มาของภาพ (Source):** [คลิกเพื่อดูภาพจากแแหล่งที่มา](https://share.google/RF1RnDgvHY22gkzJG)
- **วัตถุเป้าหมายที่ต้องการตรวจจับ:** Reef Fish School in Coral Aquarium
- **การทดลองที่ได้ลองทำมาแล้ว:**
  - Attempt 1: HSV yellow-orange range (targets the dominant yellow/orange fish)
  - Attempt 2: Canny edges
  - Attempt 3: Otsu thresholding
  - Attempt 4: Multi-color HSV union mask
- **สาเหตุที่ไม่สำเร็จ:**  (ตามที่คาดไว้) dozens of fish species with colors identical to corals,
- **ความคิดเห็น (เทคนิคความสามารถพิเศษที่จำเป็น):** จำเป็นต้องใช้ความสามารถด้าน Semantic Understanding เช่น Model Deep Learning (CNN, Transformers) ที่ได้มีการฝึกสอน (Train) จากชุดข้อมูล เพื่อให้สามารถเข้าใจบริบทของภาพโดยรวมได้ (Context)
- **รูปภาพอินพุต (Input) และ รูปภาพเอาต์พุต (Output):**
  > **หมายเหตุ:** เอาต์พุตที่แสดงด้านล่างเป็นการรวมภาพ Input ดั้งเดิมไว้ในหน้าต่างเดียวกันทั้งหมดเพื่อเปรียบเทียบผลลัพธ์การทำ Object Segmentation แล้ว
  ![FE3](output/FE3.png)

#### FE4 – Stick Insect on Tree Bark
- **เครดิตที่มาของภาพ (Source):** [คลิกเพื่อดูภาพจากแแหล่งที่มา](https://share.google/hPa7sxKQzODfSvhnw)
- **วัตถุเป้าหมายที่ต้องการตรวจจับ:** Stick Insect on Tree Bark
- **การทดลองที่ได้ลองทำมาแล้ว:**
  - Attempt 1: HSV brown range
  - Attempt 2: Gabor filter
- **สาเหตุที่ไม่สำเร็จ:**  (ตามที่คาดไว้) insect matches bark texture perfectly by evolution
- **ความคิดเห็น (เทคนิคความสามารถพิเศษที่จำเป็น):** จำเป็นต้องใช้ความสามารถด้าน Semantic Understanding เช่น Model Deep Learning (CNN, Transformers) ที่ได้มีการฝึกสอน (Train) จากชุดข้อมูล เพื่อให้สามารถเข้าใจบริบทของภาพโดยรวมได้ (Context)
- **รูปภาพอินพุต (Input) และ รูปภาพเอาต์พุต (Output):**
  > **หมายเหตุ:** เอาต์พุตที่แสดงด้านล่างเป็นการรวมภาพ Input ดั้งเดิมไว้ในหน้าต่างเดียวกันทั้งหมดเพื่อเปรียบเทียบผลลัพธ์การทำ Object Segmentation แล้ว
  ![FE4](output/FE4.png)

#### FE5 – Red Umbrella Camouflaged Among Tomatoes
- **เครดิตที่มาของภาพ (Source):** [คลิกเพื่อดูภาพจากแแหล่งที่มา](https://share.google/AMAnD6s2Rt7kPsmQl)
- **วัตถุเป้าหมายที่ต้องการตรวจจับ:** Red Umbrella Camouflaged Among Tomatoes
- **การทดลองที่ได้ลองทำมาแล้ว:**
  - Attempt 1: HSV red-orange mask
  - Attempt 2: Shape analysis — find elongated contours (folded umbrella = tall cylinder)
  - Attempt 3: Texture variance — smooth surface (umbrella) vs bumpy surface (tomato)
- **สาเหตุที่ไม่สำเร็จ:**  (ตามที่คาดไว้) the umbrella was intentionally designed in tomato-red color,
- **ความคิดเห็น (เทคนิคความสามารถพิเศษที่จำเป็น):** จำเป็นต้องใช้ความสามารถด้าน Semantic Understanding เช่น Model Deep Learning (CNN, Transformers) ที่ได้มีการฝึกสอน (Train) จากชุดข้อมูล เพื่อให้สามารถเข้าใจบริบทของภาพโดยรวมได้ (Context)
- **รูปภาพอินพุต (Input) และ รูปภาพเอาต์พุต (Output):**
  > **หมายเหตุ:** เอาต์พุตที่แสดงด้านล่างเป็นการรวมภาพ Input ดั้งเดิมไว้ในหน้าต่างเดียวกันทั้งหมดเพื่อเปรียบเทียบผลลัพธ์การทำ Object Segmentation แล้ว
  ![FE5](output/FE5.png)

#### FE6 – Nude Dress Blending with Skin Tone
- **เครดิตที่มาของภาพ (Source):** [คลิกเพื่อดูภาพจากแแหล่งที่มา](https://share.google/Owag6kXFSi8xBo9fD)
- **วัตถุเป้าหมายที่ต้องการตรวจจับ:** Nude Dress Blending with Skin Tone
- **การทดลองที่ได้ลองทำมาแล้ว:**
  - Attempt 1: HSV skin-tone mask
  - Attempt 2: YCrCb skin detection (standard skin segmentation method)
  - Attempt 3: Canny edge segmentation
  - Attempts to find the fabric-to-skin boundary via gradient —
  - Attempt 4: Texture variance — fabric (embellishments + wrinkles) vs smooth skin
- **สาเหตุที่ไม่สำเร็จ:**  (ตามที่คาดไว้) the pale pink / nude dress shares the same hue, saturation,
- **ความคิดเห็น (เทคนิคความสามารถพิเศษที่จำเป็น):** จำเป็นต้องใช้ความสามารถด้าน Semantic Understanding เช่น Model Deep Learning (CNN, Transformers) ที่ได้มีการฝึกสอน (Train) จากชุดข้อมูล เพื่อให้สามารถเข้าใจบริบทของภาพโดยรวมได้ (Context)
- **รูปภาพอินพุต (Input) และ รูปภาพเอาต์พุต (Output):**
  > **หมายเหตุ:** เอาต์พุตที่แสดงด้านล่างเป็นการรวมภาพ Input ดั้งเดิมไว้ในหน้าต่างเดียวกันทั้งหมดเพื่อเปรียบเทียบผลลัพธ์การทำ Object Segmentation แล้ว
  ![FE6](output/FE6.png)

#### FE7 – Night Scene — Detect Car Body
- **เครดิตที่มาของภาพ (Source):** [คลิกเพื่อดูภาพจากแแหล่งที่มา](https://share.google/AGMcw5aU4iuqoEnfZ)
- **วัตถุเป้าหมายที่ต้องการตรวจจับ:** Night Scene — Detect Car Body
- **การทดลองที่ได้ลองทำมาแล้ว:**
  - Attempt 1: CLAHE + Canny
  - Attempt 2: Bright-pixel threshold (headlights / taillights only)
  - Attempt 3: Otsu thresholding on CLAHE-enhanced image
  - Attempts a global brightness split after contrast enhancement —
- **สาเหตุที่ไม่สำเร็จ:**  (ตามที่คาดไว้) extreme darkness eliminates all color information;
- **ความคิดเห็น (เทคนิคความสามารถพิเศษที่จำเป็น):** เทคนิคที่อาจจะช่วยได้คือการใช้เซ็นเซอร์ภาพอื่น เช่น Infrared หรือ Thermal Camera เนื่องจากแสงในช่วงสายตาปกติมีไม่เพียงพอในการสร้าง Intensity ให้ตรวจจับ
- **รูปภาพอินพุต (Input) และ รูปภาพเอาต์พุต (Output):**
  > **หมายเหตุ:** เอาต์พุตที่แสดงด้านล่างเป็นการรวมภาพ Input ดั้งเดิมไว้ในหน้าต่างเดียวกันทั้งหมดเพื่อเปรียบเทียบผลลัพธ์การทำ Object Segmentation แล้ว
  ![FE7](output/FE7.png)

#### FE8 – Sea Turtle on Coral Reef
- **เครดิตที่มาของภาพ (Source):** [คลิกเพื่อดูภาพจากแแหล่งที่มา](https://pin.it/57y6wVlCt)
- **วัตถุเป้าหมายที่ต้องการตรวจจับ:** Sea Turtle on Coral Reef
- **การทดลองที่ได้ลองทำมาแล้ว:**
  - Attempt 1: HSV coral-brown mask (target the turtle's shell tone)
  - Attempt 2: Canny edge detection + morphological closing
  - Attempt 3: Multi-angle Gabor filter (detect scute pattern on shell)
  - Attempt 4: GrabCut with center ROI (assume turtle occupies the image centre)
- **สาเหตุที่ไม่สำเร็จ:**  (ตามที่คาดไว้) the turtle's shell pattern and warm-brown tone overlap completely
- **ความคิดเห็น (เทคนิคความสามารถพิเศษที่จำเป็น):** จำเป็นต้องใช้ความสามารถด้าน Semantic Understanding เช่น Model Deep Learning (CNN, Transformers) ที่ได้มีการฝึกสอน (Train) จากชุดข้อมูล เพื่อให้สามารถเข้าใจบริบทของภาพโดยรวมได้ (Context)
- **รูปภาพอินพุต (Input) และ รูปภาพเอาต์พุต (Output):**
  > **หมายเหตุ:** เอาต์พุตที่แสดงด้านล่างเป็นการรวมภาพ Input ดั้งเดิมไว้ในหน้าต่างเดียวกันทั้งหมดเพื่อเปรียบเทียบผลลัพธ์การทำ Object Segmentation แล้ว
  ![FE8](output/FE8.png)

#### FE9 – Flounder Fish on Sandy Bottom
- **เครดิตที่มาของภาพ (Source):** [คลิกเพื่อดูภาพจากแแหล่งที่มา](https://pin.it/BystUjijX)
- **วัตถุเป้าหมายที่ต้องการตรวจจับ:** Flounder Fish on Sandy Bottom
- **การทดลองที่ได้ลองทำมาแล้ว:**
  - Attempt 1: HSV sandy/tan color mask
  - Attempt 2: Gabor at various scales
- **สาเหตุที่ไม่สำเร็จ:**  (ตามที่คาดไว้) flatfish evolved perfect color/texture mimicry of sand
- **ความคิดเห็น (เทคนิคความสามารถพิเศษที่จำเป็น):** จำเป็นต้องใช้ความสามารถด้าน Semantic Understanding เช่น Model Deep Learning (CNN, Transformers) ที่ได้มีการฝึกสอน (Train) จากชุดข้อมูล เพื่อให้สามารถเข้าใจบริบทของภาพโดยรวมได้ (Context)
- **รูปภาพอินพุต (Input) และ รูปภาพเอาต์พุต (Output):**
  > **หมายเหตุ:** เอาต์พุตที่แสดงด้านล่างเป็นการรวมภาพ Input ดั้งเดิมไว้ในหน้าต่างเดียวกันทั้งหมดเพื่อเปรียบเทียบผลลัพธ์การทำ Object Segmentation แล้ว
  ![FE9](output/FE9.png)

#### FE10 – Dense Crowd of People
- **เครดิตที่มาของภาพ (Source):** [คลิกเพื่อดูภาพจากแแหล่งที่มา](https://share.google/Z33aTWqH3gDI1ypEf)
- **วัตถุเป้าหมายที่ต้องการตรวจจับ:** Dense Crowd of People
- **การทดลองที่ได้ลองทำมาแล้ว:**
  - Attempt 1: Adaptive threshold
  - Attempt 2: Canny
- **สาเหตุที่ไม่สำเร็จ:**  (ตามที่คาดไว้) no way to separate individual people without semantic knowledge
- **ความคิดเห็น (เทคนิคความสามารถพิเศษที่จำเป็น):** ต้องใช้วิธี Instance Segmentation ผ่าน Deep Learning ที่มีการทำ Object Counting หรือ Heatmap เพื่อก้าวข้ามข้อจำกัดของการตรวจจับ Edge ที่ซ้อนทับกันหนาแน่น
- **รูปภาพอินพุต (Input) และ รูปภาพเอาต์พุต (Output):**
  > **หมายเหตุ:** เอาต์พุตที่แสดงด้านล่างเป็นการรวมภาพ Input ดั้งเดิมไว้ในหน้าต่างเดียวกันทั้งหมดเพื่อเปรียบเทียบผลลัพธ์การทำ Object Segmentation แล้ว
  ![FE10](output/FE10.png)

### 🟠 กลุ่มที่ 4: ล้มเหลวโดยไม่คาดคิด (FU1–FU10)
**แนวทางหลัก:** กรณีที่คาดว่าควรจะสำเร็จแต่ล้มเหลว เนื่องจากข้อจำกัดที่ไม่คาดคิดจากสภาพแวดล้อม แสง และข้อบกพร่องของ Traditional Image Processing

#### FU1 – Peeled Hard-Boiled Egg on White Plate
- **เครดิตที่มาของภาพ (Source):** [คลิกเพื่อดูภาพจากแแหล่งที่มา](https://share.google/8O2a5qIGxSqxw7mED)
- **วัตถุเป้าหมายที่ต้องการตรวจจับ:** Peeled Hard-Boiled Egg on White Plate
- **การทดลองที่ได้ลองทำมาแล้ว:**
  - Attempt 1: Brightness threshold (egg white = very high brightness)
  - Attempt 2: Canny low threshold (chase the faint contact shadow at egg edge)
  - Attempt 3: LAB b-channel (egg has a faint warm yellow tint vs cool plate white)
  - Attempt 4: Adaptive threshold (exploit local contrast between egg and plate)
- **สาเหตุที่ไม่สำเร็จ:** egg has a clear oval shape + subtle contact shadow → contour should find it easily => กลายเป็นว่า: egg white and white plate share identical brightness, hue, and saturation;
- **ความคิดเห็น (เทคนิคความสามารถพิเศษที่จำเป็น):** จำเป็นต้องใช้ความสามารถด้าน Semantic Understanding เช่น Model Deep Learning (CNN, Transformers) ที่ได้มีการฝึกสอน (Train) จากชุดข้อมูล เพื่อให้สามารถเข้าใจบริบทของภาพโดยรวมได้ (Context)
- **รูปภาพอินพุต (Input) และ รูปภาพเอาต์พุต (Output):**
  > **หมายเหตุ:** เอาต์พุตที่แสดงด้านล่างเป็นการรวมภาพ Input ดั้งเดิมไว้ในหน้าต่างเดียวกันทั้งหมดเพื่อเปรียบเทียบผลลัพธ์การทำ Object Segmentation แล้ว
  ![FU1](output/FU1.png)

#### FU2 – Coin on Wooden Table
- **เครดิตที่มาของภาพ (Source):** [คลิกเพื่อดูภาพจากแแหล่งที่มา](https://share.google/BKo5lqWmNLlk5mAbF)
- **วัตถุเป้าหมายที่ต้องการตรวจจับ:** Coin on Wooden Table
- **การทดลองที่ได้ลองทำมาแล้ว:**
  - Attempt 1: CLAHE + GaussianBlur (enhance local contrast before edge detection)
  - Attempt 2: Canny Edge Detection (low=30, high=100)
  - Attempt 3: Morphological Closing to reconnect broken edges
  - Attempt 4: Contour + Bounding Box
- **สาเหตุที่ไม่สำเร็จ:** coin has a distinct circular shape + metallic sheen → Canny/contour should detect it => กลายเป็นว่า: coin's metallic reflection matches wood tone; grain edges override coin boundary
- **ความคิดเห็น (เทคนิคความสามารถพิเศษที่จำเป็น):** จำเป็นต้องใช้ความสามารถด้าน Semantic Understanding เช่น Model Deep Learning (CNN, Transformers) ที่ได้มีการฝึกสอน (Train) จากชุดข้อมูล เพื่อให้สามารถเข้าใจบริบทของภาพโดยรวมได้ (Context)
- **รูปภาพอินพุต (Input) และ รูปภาพเอาต์พุต (Output):**
  > **หมายเหตุ:** เอาต์พุตที่แสดงด้านล่างเป็นการรวมภาพ Input ดั้งเดิมไว้ในหน้าต่างเดียวกันทั้งหมดเพื่อเปรียบเทียบผลลัพธ์การทำ Object Segmentation แล้ว
  ![FU2](output/FU2.png)

#### FU3 – White Coffee Mug on White Table
- **เครดิตที่มาของภาพ (Source):** [คลิกเพื่อดูภาพจากแแหล่งที่มา](https://share.google/LiKJfGXf1xejAhGE2)
- **วัตถุเป้าหมายที่ต้องการตรวจจับ:** White Coffee Mug on White Table
- **การทดลองที่ได้ลองทำมาแล้ว:**
  - Attempt 1: Canny (expected to detect cup boundary and rim)
  - Attempt 2: Otsu Thresholding (automatic foreground/background separation)
  - Attempt 3: Adaptive Threshold (capture local contrast between cup and table)
- **สาเหตุที่ไม่สำเร็จ:** simple Canny should find the cup boundary. => กลายเป็นว่า: strong specular highlight + white surface = boundary disappears.
- **ความคิดเห็น (เทคนิคความสามารถพิเศษที่จำเป็น):** จำเป็นต้องใช้ความสามารถด้าน Semantic Understanding เช่น Model Deep Learning (CNN, Transformers) ที่ได้มีการฝึกสอน (Train) จากชุดข้อมูล เพื่อให้สามารถเข้าใจบริบทของภาพโดยรวมได้ (Context)
- **รูปภาพอินพุต (Input) และ รูปภาพเอาต์พุต (Output):**
  > **หมายเหตุ:** เอาต์พุตที่แสดงด้านล่างเป็นการรวมภาพ Input ดั้งเดิมไว้ในหน้าต่างเดียวกันทั้งหมดเพื่อเปรียบเทียบผลลัพธ์การทำ Object Segmentation แล้ว
  ![FU3](output/FU3.png)

#### FU4 – Black Cat on Black Sofa
- **เครดิตที่มาของภาพ (Source):** [คลิกเพื่อดูภาพจากแแหล่งที่มา](https://pin.it/7Kg6lwfTk)
- **วัตถุเป้าหมายที่ต้องการตรวจจับ:** Black Cat on Black Sofa
- **การทดลองที่ได้ลองทำมาแล้ว:**
  - Attempt 1: Dark region threshold (cat = very dark pixels)
  - Attempt 2: Canny edge (find cat body boundary)
  - Attempt 3: Gabor texture filter (cat fur vs. sofa fabric frequency)
  - Attempt 4: CLAHE + Otsu on LAB L-channel (local contrast enhancement)
- **สาเหตุที่ไม่สำเร็จ:** Cat has distinct body shape + fur texture → contour/texture will work => กลายเป็นว่า: Cat and sofa are same dark color; texture difference too subtle
- **ความคิดเห็น (เทคนิคความสามารถพิเศษที่จำเป็น):** จำเป็นต้องใช้ความสามารถด้าน Semantic Understanding เช่น Model Deep Learning (CNN, Transformers) ที่ได้มีการฝึกสอน (Train) จากชุดข้อมูล เพื่อให้สามารถเข้าใจบริบทของภาพโดยรวมได้ (Context)
- **รูปภาพอินพุต (Input) และ รูปภาพเอาต์พุต (Output):**
  > **หมายเหตุ:** เอาต์พุตที่แสดงด้านล่างเป็นการรวมภาพ Input ดั้งเดิมไว้ในหน้าต่างเดียวกันทั้งหมดเพื่อเปรียบเทียบผลลัพธ์การทำ Object Segmentation แล้ว
  ![FU4](output/FU4.png)

#### FU5 – Brown Bread Loaf on Brown Wooden Board
- **เครดิตที่มาของภาพ (Source):** [คลิกเพื่อดูภาพจากแแหล่งที่มา](https://share.google/gHKlAsBZ7vuxqUJR9)
- **วัตถุเป้าหมายที่ต้องการตรวจจับ:** Brown Bread Loaf on Brown Wooden Board
- **การทดลองที่ได้ลองทำมาแล้ว:**
  - Attempt 1: Brown HSV range (isolate bread crust color)
  - Attempt 2: Canny + Morphological Close (find loaf outline)
  - Attempt 3: Contour bounding box on closed edges
- **สาเหตุที่ไม่สำเร็จ:** distinct shape (rounded loaf) will be found by contours. => กลายเป็นว่า: grain color of wood matches crust; shadows fragment the contour.
- **ความคิดเห็น (เทคนิคความสามารถพิเศษที่จำเป็น):** จำเป็นต้องใช้ความสามารถด้าน Semantic Understanding เช่น Model Deep Learning (CNN, Transformers) ที่ได้มีการฝึกสอน (Train) จากชุดข้อมูล เพื่อให้สามารถเข้าใจบริบทของภาพโดยรวมได้ (Context)
- **รูปภาพอินพุต (Input) และ รูปภาพเอาต์พุต (Output):**
  > **หมายเหตุ:** เอาต์พุตที่แสดงด้านล่างเป็นการรวมภาพ Input ดั้งเดิมไว้ในหน้าต่างเดียวกันทั้งหมดเพื่อเปรียบเทียบผลลัพธ์การทำ Object Segmentation แล้ว
  ![FU5](output/FU5.png)

#### FU6 – Snowy Owl on Snowy Branch
- **เครดิตที่มาของภาพ (Source):** [คลิกเพื่อดูภาพจากแแหล่งที่มา](https://pin.it/69DeTK2bj)
- **วัตถุเป้าหมายที่ต้องการตรวจจับ:** Snowy Owl on Snowy Branch
- **การทดลองที่ได้ลองทำมาแล้ว:**
  - Attempt 1: HSV brightness threshold (owl = bright white)
  - Attempt 2: LAB b-channel threshold (feathers slightly warm vs cool snow)
  - Attempt 3: Canny edge (find owl body outline)
  - Attempt 4: Gabor texture filter (feathers vs smooth snow surface)
- **สาเหตุที่ไม่สำเร็จ:** Owl has a distinct round body + facial disc → shape cues should work => กลายเป็นว่า: White feathers and white snow are identical in every color space
- **ความคิดเห็น (เทคนิคความสามารถพิเศษที่จำเป็น):** จำเป็นต้องใช้ความสามารถด้าน Semantic Understanding เช่น Model Deep Learning (CNN, Transformers) ที่ได้มีการฝึกสอน (Train) จากชุดข้อมูล เพื่อให้สามารถเข้าใจบริบทของภาพโดยรวมได้ (Context)
- **รูปภาพอินพุต (Input) และ รูปภาพเอาต์พุต (Output):**
  > **หมายเหตุ:** เอาต์พุตที่แสดงด้านล่างเป็นการรวมภาพ Input ดั้งเดิมไว้ในหน้าต่างเดียวกันทั้งหมดเพื่อเปรียบเทียบผลลัพธ์การทำ Object Segmentation แล้ว
  ![FU6](output/FU6.png)

#### FU7 – Factory Smoke Blending with Clouds at Sunset
- **เครดิตที่มาของภาพ (Source):** [คลิกเพื่อดูภาพจากแแหล่งที่มา](https://share.google/VJnwMAARFGjwkW7b4)
- **วัตถุเป้าหมายที่ต้องการตรวจจับ:** Factory Smoke Blending with Clouds at Sunset
- **การทดลองที่ได้ลองทำมาแล้ว:**
  - Attempt 1: HSV gray/white mask (smoke = low saturation, mid-high brightness)
  - Attempt 2: Low-saturation threshold (smoke & clouds share low-sat zone)
  - Attempt 3: Multi-scale Gabor (turbulent smoke vs. softer cloud texture)
  - Attempt 4: Sobel gradient magnitude (find plume boundary edges)
- **สาเหตุที่ไม่สำเร็จ:** Smoke plume rises from chimney → distinct upward shape + low saturation boundary => กลายเป็นว่า: Sunset light tints both smoke and clouds the same orange-gold; boundaries dissolve
- **ความคิดเห็น (เทคนิคความสามารถพิเศษที่จำเป็น):** จำเป็นต้องใช้ความสามารถด้าน Semantic Understanding เช่น Model Deep Learning (CNN, Transformers) ที่ได้มีการฝึกสอน (Train) จากชุดข้อมูล เพื่อให้สามารถเข้าใจบริบทของภาพโดยรวมได้ (Context)
- **รูปภาพอินพุต (Input) และ รูปภาพเอาต์พุต (Output):**
  > **หมายเหตุ:** เอาต์พุตที่แสดงด้านล่างเป็นการรวมภาพ Input ดั้งเดิมไว้ในหน้าต่างเดียวกันทั้งหมดเพื่อเปรียบเทียบผลลัพธ์การทำ Object Segmentation แล้ว
  ![FU7](output/FU7.png)

#### FU8 – Wooden Chair Blending with Wooden Table & Floor
- **เครดิตที่มาของภาพ (Source):** [คลิกเพื่อดูภาพจากแแหล่งที่มา](https://share.google/YjglcuzoyJGwoEgNZ)
- **วัตถุเป้าหมายที่ต้องการตรวจจับ:** Wooden Chair Blending with Wooden Table & Floor
- **การทดลองที่ได้ลองทำมาแล้ว:**
  - Attempt 1: HSV warm-brown mask (target chair wood tone)
  - Attempt 2: Canny edge detection (chair legs and back = strong geometric lines)
  - Attempt 3: LAB L-channel threshold (separate lit chair from dark shadows)
  - Attempt 4: Adaptive threshold (recover local contrast under mixed lighting)
  - Attempts to find chair structure via local brightness differences —
- **สาเหตุที่ไม่สำเร็จ:** chair has a distinct geometric structure (legs, back, seat) → => กลายเป็นว่า: chair, table, and floor share identical warm-brown wood tones;
- **ความคิดเห็น (เทคนิคความสามารถพิเศษที่จำเป็น):** จำเป็นต้องใช้ความสามารถด้าน Semantic Understanding เช่น Model Deep Learning (CNN, Transformers) ที่ได้มีการฝึกสอน (Train) จากชุดข้อมูล เพื่อให้สามารถเข้าใจบริบทของภาพโดยรวมได้ (Context)
- **รูปภาพอินพุต (Input) และ รูปภาพเอาต์พุต (Output):**
  > **หมายเหตุ:** เอาต์พุตที่แสดงด้านล่างเป็นการรวมภาพ Input ดั้งเดิมไว้ในหน้าต่างเดียวกันทั้งหมดเพื่อเปรียบเทียบผลลัพธ์การทำ Object Segmentation แล้ว
  ![FU8](output/FU8.png)

#### FU9 – Person / Dolphin Underwater (Poor Visibility)
- **เครดิตที่มาของภาพ (Source):** [คลิกเพื่อดูภาพจากแแหล่งที่มา](https://share.google/OKHplb2fxDBVBgewQ)
- **วัตถุเป้าหมายที่ต้องการตรวจจับ:** Person / Dolphin Underwater (Poor Visibility)
- **การทดลองที่ได้ลองทำมาแล้ว:**
  - Attempt 1: LAB L-channel dark region (subject = darker than water background)
  - Attempt 2: Canny (find subject body outline vs. water)
  - Attempt 3: HSV blue-green suppression (isolate non-water hue regions)
  - Attempt 4: CLAHE + Adaptive threshold (recover local contrast from haze)
- **สาเหตุที่ไม่สำเร็จ:** LAB dark-region mask + Canny should outline the subject vs. water => กลายเป็นว่า: Water scattering creates uniform blue-green cast; caustics dominate all edges
- **ความคิดเห็น (เทคนิคความสามารถพิเศษที่จำเป็น):** จำเป็นต้องใช้ความสามารถด้าน Semantic Understanding เช่น Model Deep Learning (CNN, Transformers) ที่ได้มีการฝึกสอน (Train) จากชุดข้อมูล เพื่อให้สามารถเข้าใจบริบทของภาพโดยรวมได้ (Context)
- **รูปภาพอินพุต (Input) และ รูปภาพเอาต์พุต (Output):**
  > **หมายเหตุ:** เอาต์พุตที่แสดงด้านล่างเป็นการรวมภาพ Input ดั้งเดิมไว้ในหน้าต่างเดียวกันทั้งหมดเพื่อเปรียบเทียบผลลัพธ์การทำ Object Segmentation แล้ว
  ![FU9](output/FU9.png)

#### FU10 – Orange Cat on Orange/Yellow Sofa
- **เครดิตที่มาของภาพ (Source):** [คลิกเพื่อดูภาพจากแแหล่งที่มา](https://pin.it/2zp0QnsJ7)
- **วัตถุเป้าหมายที่ต้องการตรวจจับ:** Orange Cat on Orange/Yellow Sofa
- **การทดลองที่ได้ลองทำมาแล้ว:**
  - Attempt 1: Orange HSV range (cat body color)
  - Attempt 2: Sobel gradient magnitude (fur-edge vs. fabric-edge)
  - Attempt 3: Canny (find body silhouette against background)
  - Attempt 4: LAB b-channel (cat fur warm yellow-orange vs. fabric texture)
- **สาเหตุที่ไม่สำเร็จ:** cat fur vs. fabric should have edge cues → boundary detectable => กลายเป็นว่า: cat and sofa share same warm orange hue; Sobel gradients merge fur with fabric weave
- **ความคิดเห็น (เทคนิคความสามารถพิเศษที่จำเป็น):** จำเป็นต้องใช้ความสามารถด้าน Semantic Understanding เช่น Model Deep Learning (CNN, Transformers) ที่ได้มีการฝึกสอน (Train) จากชุดข้อมูล เพื่อให้สามารถเข้าใจบริบทของภาพโดยรวมได้ (Context)
- **รูปภาพอินพุต (Input) และ รูปภาพเอาต์พุต (Output):**
  > **หมายเหตุ:** เอาต์พุตที่แสดงด้านล่างเป็นการรวมภาพ Input ดั้งเดิมไว้ในหน้าต่างเดียวกันทั้งหมดเพื่อเปรียบเทียบผลลัพธ์การทำ Object Segmentation แล้ว
  ![FU10](output/FU10.png)


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
