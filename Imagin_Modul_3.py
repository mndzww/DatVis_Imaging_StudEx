import numpy as np
import cv2
import os

# === STEP 1: Load Gambar ===
image_path = "Unwarped-Image.jpg"
image = cv2.imread(image_path)

if image is None:
    print(f"❌ Gagal memuat gambar: {image_path}")
    exit()

# === STEP 2: Titik referensi penggaris (10 cm di dunia nyata) ===
ruler_pt1 = (100, 200)     # titik awal penggaris
ruler_pt2 = (100, 700)     # titik akhir penggaris
ruler_cm_actual = 10.0     # panjang sebenarnya (cm)

# Hitung panjang penggaris dalam pixel
ruler_px_length = np.linalg.norm(np.array(ruler_pt2) - np.array(ruler_pt1))

# Hitung skala pixel/cm berdasarkan penggaris
pixel_per_cm = ruler_px_length / ruler_cm_actual
print(f"📏 Skala dari penggaris: {pixel_per_cm:.2f} px/cm")

# === STEP 3: Titik objek yang ingin diprediksi panjangnya ===
obj_pt1 = (900, 550)
obj_pt2 = (900, 3600)

# Hitung panjang objek dalam pixel
obj_px_length = np.linalg.norm(np.array(obj_pt2) - np.array(obj_pt1))

# Prediksi panjang objek dalam cm
obj_cm_predicted = obj_px_length / pixel_per_cm

# === STEP 4: Gambar garis dan label panjang ===
cv2.line(image, obj_pt1, obj_pt2, (0, 255, 0), 3)
cv2.circle(image, obj_pt1, 8, (0, 255, 0), -1)
cv2.circle(image, obj_pt2, 8, (0, 255, 0), -1)

label = f"{obj_cm_predicted:.2f} cm"
midpoint = ((obj_pt1[0] + obj_pt2[0]) // 2, obj_pt1[1] - 30)
cv2.putText(image, label, midpoint, cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), 3)

# === STEP 5: Simpan hasil ===
cv2.imwrite("Predicted-Length.jpg", image)
print(f"✅ Gambar akhir disimpan sebagai Predicted-Length.jpg")
print(f"📐 Panjang prediksi objek: {obj_cm_predicted:.2f} cm")
