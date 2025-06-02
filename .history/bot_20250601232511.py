import cv2
import numpy as np
import matplotlib.pyplot as plt
from google.colab import files

# Upload gambar rambu dari komputer
uploaded = files.upload()
img_path = list(uploaded.keys())[0]

# Load dan resize gambar
img = cv2.imread(img_path)
img = cv2.resize(img, (400, 400))
output = img.copy()

# Konversi ke HSV
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# ===================== #
# === DETEKSI WARNA === #
# ===================== #

# Definisikan range warna merah
mask_red = cv2.inRange(hsv, np.array([0,100,100]), np.array([10,255,255])) + \
           cv2.inRange(hsv, np.array([160,100,100]), np.array([179,255,255]))

# Range warna biru
mask_blue = cv2.inRange(hsv, np.array([100,150,0]), np.array([140,255,255]))

# Gabungkan semua mask
masks = {'Merah': mask_red, 'Biru': mask_blue}

# ====================== #
# === PROSES KONTUR === #
# ====================== #

for warna, mask in masks.items():
    # Cari kontur dari mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 800:
            # Deteksi bentuk
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
            shape = "Tidak diketahui"
            
            if len(approx) > 8:
                shape = "Lingkaran"
            elif len(approx) == 3:
                shape = "Segitiga"
            elif len(approx) == 4:
                shape = "Persegi Panjang"
            elif len(approx) == 7:
                shape = "Panah"

            # Prediksi fungsi berdasarkan warna + bentuk
            fungsi = "Tidak diketahui"
            if warna == "Merah" and shape == "Lingkaran":
                fungsi = "STOP"
            elif warna == "Biru" and shape == "Lingkaran":
                fungsi = "Rambu Lain"
            elif warna == "Biru" and "Panah" in shape:
                fungsi = "Belok (Panah)"
            elif warna == "Biru":
                fungsi = "Petunjuk"

            # Gambar bounding box & teks
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(output, (x, y), (x + w, y + h), (0,255,0), 2)
            label = f"{warna}, {shape}, {fungsi}"
            cv2.putText(output, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

# Tampilkan hasil
output_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(8,8))
plt.imshow(output_rgb)
plt.axis('off')
plt.title("Deteksi Rambu: Warna, Bentuk, Fungsi")
plt.show()
