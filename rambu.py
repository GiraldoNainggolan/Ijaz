import cv2
import numpy as np
import pytesseract # Pastikan Anda sudah menginstal pytesseract dan Tesseract-OCR

# Atur lokasi instalasi Tesseract-OCR jika tidak terdeteksi otomatis
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe' # Contoh di Windows

def deteksi_bentuk(cnt):
    """
    Mendeteksi bentuk dari sebuah kontur, dengan penambahan deteksi 6 dan 8 sisi.
    """
    peri = cv2.arcLength(cnt, True)
    # Epsilon yang lebih rendah untuk deteksi sudut yang lebih akurat
    approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

    num_vertices = len(approx)

    if num_vertices == 3:
        return "Segitiga"
    elif num_vertices == 4:
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = float(w) / h
        if 0.9 <= aspect_ratio <= 1.1: # Toleransi lebih longgar untuk persegi
            return "Persegi"
        else:
            return "Persegi Panjang"
    elif num_vertices == 6:
        return "Heksagon" # Bentuk 6 sisi
    elif num_vertices == 8:
        return "Oktagon" # Bentuk 8 sisi (seperti rambu STOP)
    elif num_vertices > 8: # Untuk lingkaran atau bentuk dengan banyak sisi
        return "Lingkaran"
    else:
        return "Tidak diketahui"

def prediksi_fungsi(warna, bentuk, teks_terdeteksi=""):
    """
    Memprediksi fungsi rambu berdasarkan warna, bentuk, dan teks yang terdeteksi.
    """
    teks_terdeteksi = teks_terdeteksi.upper() # Konversi ke huruf besar untuk perbandingan

    # 1. Rambu Peringatan
    if warna == "Kuning" and (bentuk == "Segitiga" or bentuk == "Persegi" or bentuk == "Persegi Panjang"):
        return "RAMBU PERINGATAN"
    
    # 2. Rambu Larangan
    # Rambu STOP (Merah, Oktagon, Teks "STOP")
    elif warna == "Merah" and bentuk == "Oktagon" and "STOP" in teks_terdeteksi:
        return "RAMBU LARANGAN (STOP)"
    # Rambu Larangan umum (Merah, Lingkaran)
    elif warna == "Merah" and bentuk == "Lingkaran":
        return "RAMBU LARANGAN" 

    # 3. Rambu Perintah
    # Rambu Perintah (Biru, Lingkaran)
    elif warna == "Biru" and bentuk == "Lingkaran":
        return "RAMBU PERINTAH"
    # Rambu Perintah Arah (Biru, Panah/Heksagon/Bentuk lain dengan teks arah)
    elif warna == "Biru" and ("Panah" in bentuk or "ARAH" in teks_terdeteksi):
        return "RAMBU PERINTAH (ARAH)"

    # 4. Rambu Petunjuk
    # Rambu Petunjuk Arah/Lokasi (Hijau, Persegi/Persegi Panjang)
    elif warna == "Hijau" and (bentuk == "Persegi" or bentuk == "Persegi Panjang"):
        return "RAMBU PETUNJUK"
    
    # Rambu Informasi/Petunjuk Umum (Biru tanpa bentuk spesifik larangan/perintah)
    elif warna == "Biru":
        return "RAMBU INFORMASI/PETUNJUK UMUM"

    # Jika tidak ada kombinasi yang cocok
    else:
        return "Tidak diketahui"

# Buka kamera (0 untuk kamera default)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Tidak dapat membuka kamera.")
    exit()

# Inisialisasi variabel untuk melacak kontur stabil
# Ini akan membantu menstabilkan bounding box dan label
stable_contours = {}
STABILITY_THRESHOLD = 5 # Jumlah frame kontur harus terdeteksi untuk dianggap stabil
MAX_CONTOUR_AGE = 10 # Kontur akan dihapus jika tidak terdeteksi selama sejumlah frame ini

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Tidak dapat membaca frame dari kamera.")
        break

    frame = cv2.resize(frame, (640, 480))
    output = frame.copy()

    blurred_frame = cv2.GaussianBlur(frame, (5, 5), 0)
    hsv = cv2.cvtColor(blurred_frame, cv2.COLOR_BGR2HSV)

    warna_ranges = {
        "Merah": {
            "lower1": np.array([0, 100, 100]), "upper1": np.array([10, 255, 255]),
            "lower2": np.array([170, 100, 100]), "upper2": np.array([179, 255, 255])
        },
        "Biru": {"lower": np.array([90, 50, 50]), "upper": np.array([130, 255, 255])},
        "Kuning": {"lower": np.array([20, 100, 100]), "upper": np.array([30, 255, 255])},
        "Hijau": {"lower": np.array([40, 50, 50]), "upper": np.array([80, 255, 255])}
    }

    current_frame_detections = []

    for warna, ranges in warna_ranges.items():
        mask = None
        if warna == "Merah":
            mask1 = cv2.inRange(hsv, ranges["lower1"], ranges["upper1"])
            mask2 = cv2.inRange(hsv, ranges["lower2"], ranges["upper2"])
            mask = mask1 + mask2
        else:
            mask = cv2.inRange(hsv, ranges["lower"], ranges["upper"])

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 2000: # Meningkatkan ambang batas area untuk objek yang lebih jelas
                x, y, w, h = cv2.boundingRect(cnt)
                # Filter rasio aspek untuk bentuk yang wajar (hindari garis tipis panjang)
                if not (h == 0 or w / h > 5 or h / w > 5):
                    current_frame_detections.append({
                        "warna": warna,
                        "contour": cnt,
                        "bbox": (x, y, w, h)
                    })
    
    # Perbarui status kontur stabil
    new_stable_contours = {}
    for det in current_frame_detections:
        # Coba cocokkan deteksi saat ini dengan kontur yang sudah stabil
        matched = False
        for stable_id, stable_data in stable_contours.items():
            x_stable, y_stable, w_stable, h_stable = stable_data["bbox"]
            x_curr, y_curr, w_curr, h_curr = det["bbox"]

            # Periksa tumpang tindih (Overlap)
            if (x_curr < x_stable + w_stable and x_curr + w_curr > x_stable and
                y_curr < y_stable + h_stable and y_curr + h_curr > y_stable):
                
                # Update data kontur yang cocok
                new_stable_contours[stable_id] = {
                    "count": stable_data["count"] + 1,
                    "bbox": det["bbox"],
                    "warna": det["warna"],
                    "contour": det["contour"],
                    "last_seen": 0 # Reset hitungan tidak terlihat
                }
                matched = True
                break
        
        if not matched:
            # Jika tidak cocok, tambahkan sebagai kontur baru
            new_stable_contours[len(stable_contours) + 1] = {
                "count": 1,
                "bbox": det["bbox"],
                "warna": det["warna"],
                "contour": det["contour"],
                "last_seen": 0
            }
    
    # Hapus kontur yang tidak lagi terlihat atau tidak stabil
    for stable_id, stable_data in stable_contours.items():
        if stable_id not in new_stable_contours:
            stable_data["last_seen"] += 1
            if stable_data["last_seen"] < MAX_CONTOUR_AGE:
                new_stable_contours[stable_id] = stable_data # Pertahankan sementara
    
    stable_contours = {k: v for k, v in new_stable_contours.items() if v["count"] >= STABILITY_THRESHOLD or v["last_seen"] < MAX_CONTOUR_AGE}

    # Proses kontur yang stabil
    for stable_id, data in stable_contours.items():
        if data["count"] >= STABILITY_THRESHOLD: # Hanya proses yang benar-benar stabil
            x, y, w, h = data["bbox"]
            warna = data["warna"]
            cnt = data["contour"]

            bentuk = deteksi_bentuk(cnt)
            
            # Ekstrak teks dari area bounding box
            teks_terdeteksi = ""
            if w > 30 and h > 30: # Hanya coba OCR jika ukuran rambu cukup besar
                roi = frame[y:y+h, x:x+w]
                # Konversi ROI ke grayscale untuk OCR yang lebih baik
                gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                # Terapkan thresholding atau adaptif thresholding jika diperlukan
                _, thresh_roi = cv2.threshold(gray_roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                
                # Gunakan Tesseract untuk mendeteksi teks
                # config='--psm 8' (Page Segmentation Mode 8: Treat the image as a single word)
                # atau --psm 6 (Assume a single uniform block of text)
                # l='eng' untuk bahasa Inggris
                try:
                    teks_terdeteksi = pytesseract.image_to_string(thresh_roi, config='--psm 8', lang='eng').strip()
                    # Filter teks yang terlalu pendek atau hanya whitespace
                    if len(teks_terdeteksi) < 2 or not any(c.isalnum() for c in teks_terdeteksi):
                        teks_terdeteksi = ""
                except pytesseract.TesseractNotFoundError:
                    teks_terdeteksi = "Tesseract Error"
                    print("Tesseract tidak ditemukan. Pastikan sudah terinstal dan path diatur.")
                except Exception as e:
                    teks_terdeteksi = f"OCR Error: {e}"


            fungsi = prediksi_fungsi(warna, bentuk, teks_terdeteksi)

            # Gambar kotak pembatas
            cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2) # Warna hijau untuk kotak

            # Teks label
            label = f"{warna}, {bentuk}"
            if teks_terdeteksi:
                label += f", Teks: {teks_terdeteksi}"
            label += f", Fungsi: {fungsi}"
            
            # Pastikan label tidak melebihi batas atas frame
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            label_y = y - 10
            if label_y < text_size[1] + 5: # Jika terlalu dekat dengan atas, taruh di bawah
                label_y = y + h + text_size[1] + 10

            cv2.putText(output, label, (x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    cv2.imshow("Deteksi Rambu Lalu Lintas", output)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()