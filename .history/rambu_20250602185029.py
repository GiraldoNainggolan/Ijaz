import cv2
import numpy as np

def deteksi_bentuk(cnt):
    """
    Mendeteksi bentuk dari sebuah kontur.
    """
    # Menghitung keliling kontur
    peri = cv2.arcLength(cnt, True)
    # Mendekati poligon dari kontur dengan akurasi tertentu
    # Parameter 0.03 * peri adalah epsilon, semakin kecil semakin akurat
    approx = cv2.approxPolyDP(cnt, 0.03 * peri, True)

    # Menentukan bentuk berdasarkan jumlah titik pada poligon yang didekati
    if len(approx) > 8:
        return "Lingkaran"
    elif len(approx) == 3:
        return "Segitiga"
    elif len(approx) == 4:
        # Untuk persegi panjang, periksa rasio aspek
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = float(w) / h
        # Jika rasio aspek mendekati 1, itu persegi, jika tidak, persegi panjang
        if 0.95 <= aspect_ratio <= 1.05:
            return "Persegi"
        else:
            return "Persegi Panjang"
    elif len(approx) == 7:
        return "Panah" # Panah sering memiliki 7 titik
    else:
        return "Tidak diketahui"

def prediksi_fungsi(warna, bentuk):
    """
    Memprediksi fungsi rambu berdasarkan warna dan bentuknya,
    sesuai dengan kategori rambu lalu lintas yang disempurnakan.
    """
    # 1. Rambu Peringatan
    # Bentuk: Segitiga, belah ketupat (terdeteksi sebagai Persegi/Persegi Panjang)
    # Warna: Dasar kuning
    if warna == "Kuning" and (bentuk == "Segitiga" or bentuk == "Persegi" or bentuk == "Persegi Panjang"):
        return "RAMBU PERINGATAN"
    
    # 2. Rambu Larangan
    # Bentuk: Lingkaran
    # Warna: Dasar merah
    elif warna == "Merah" and bentuk == "Lingkaran":
        return "RAMBU LARANGAN" # Contoh: Larangan Berhenti, Larangan Parkir

    # 3. Rambu Perintah
    # Bentuk: Lingkaran, atau bentuk khusus lainnya dengan simbol perintah (misal panah)
    # Warna: Dasar biru
    elif warna == "Biru" and bentuk == "Lingkaran":
        return "RAMBU PERINTAH" # Contoh: Perintah Lurus, Perintah Belok
    elif warna == "Biru" and "Panah" in bentuk:
        return "RAMBU PERINTAH (ARAH)" # Perintah mengikuti arah panah

    # 4. Rambu Petunjuk
    # Bentuk: Persegi, atau bentuk khusus lainnya
    # Warna: Dasar hijau
    elif warna == "Hijau" and (bentuk == "Persegi" or bentuk == "Persegi Panjang"):
        return "RAMBU PETUNJUK" # Contoh: Petunjuk Jurusan, Lokasi

    # Kasus lain untuk rambu biru yang tidak spesifik (misal rambu informasi umum)
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

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Tidak dapat membaca frame dari kamera.")
        break

    # Ubah ukuran frame untuk pemrosesan yang lebih cepat
    frame = cv2.resize(frame, (640, 480))
    output = frame.copy()

    # Pra-pemrosesan: Terapkan Gaussian blur untuk mengurangi noise
    blurred_frame = cv2.GaussianBlur(frame, (5, 5), 0)

    # Konversi ke ruang warna HSV
    hsv = cv2.cvtColor(blurred_frame, cv2.COLOR_BGR2HSV)

    # Definisikan rentang HSV untuk berbagai warna
    # Rentang ini disesuaikan untuk deteksi yang lebih robust
    warna_ranges = {
        "Merah": {
            "lower1": np.array([0, 100, 100]), "upper1": np.array([10, 255, 255]),
            "lower2": np.array([170, 100, 100]), "upper2": np.array([179, 255, 255])
        },
        "Biru": {"lower": np.array([90, 50, 50]), "upper": np.array([130, 255, 255])},
        "Kuning": {"lower": np.array([20, 100, 100]), "upper": np.array([30, 255, 255])},
        "Hijau": {"lower": np.array([40, 50, 50]), "upper": np.array([80, 255, 255])}
    }

    # Iterasi melalui setiap warna untuk deteksi
    for warna, ranges in warna_ranges.items():
        mask = None
        if warna == "Merah":
            # Gabungkan dua mask untuk warna merah
            mask1 = cv2.inRange(hsv, ranges["lower1"], ranges["upper1"])
            mask2 = cv2.inRange(hsv, ranges["lower2"], ranges["upper2"])
            mask = mask1 + mask2
        else:
            mask = cv2.inRange(hsv, ranges["lower"], ranges["upper"])

        # Operasi morfologi untuk membersihkan mask
        # Kernel untuk operasi morfologi
        kernel = np.ones((5, 5), np.uint8)
        # Operasi 'opening' (erosi diikuti dilasi) untuk menghilangkan noise kecil
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        # Operasi 'closing' (dilasi diikuti erosi) untuk menutup celah kecil di objek
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


        # Temukan kontur pada mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            # Filter kontur berdasarkan area untuk menghindari deteksi noise kecil
            if area > 1500: # Meningkatkan ambang batas area untuk kontur yang lebih signifikan
                bentuk = deteksi_bentuk(cnt)
                fungsi = prediksi_fungsi(warna, bentuk)

                # Dapatkan kotak pembatas untuk kontur
                x, y, w, h = cv2.boundingRect(cnt)
                # Gambar kotak pembatas di output frame
                cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2) # Warna hijau untuk kotak

                # Teks label
                label = f"{warna}, {bentuk}, {fungsi}"
                # Letakkan teks label di atas kotak pembatas
                cv2.putText(output, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2) # Warna biru untuk teks

    # Tampilkan frame hasil deteksi
    cv2.imshow("Deteksi Rambu Lalu Lintas", output)

    # Tekan 'q' untuk keluar dari loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Lepaskan kamera dan tutup semua jendela OpenCV
cap.release()
cv2.destroyAllWindows()
