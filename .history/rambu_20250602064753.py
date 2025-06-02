import cv2
import numpy as np

# Fungsi label bentuk
def get_shape(approx):
    sides = len(approx)
    if sides == 3:
        return "Segitiga", "Belok"
    elif sides == 4:
        x, y, w, h = cv2.boundingRect(approx)
        aspectRatio = float(w) / h
        if 0.90 <= aspectRatio <= 1.10:
            return "Persegi", "Lurus"
        else:
            return "Persegi Panjang", "Lurus"
    elif sides == 8:
        return "Segi-8", "Putar Balik"
    elif sides > 8:
        return "Lingkaran", "STOP"
    else:
        return "Tidak Dikenal", "Unknown"

# Fungsi deteksi warna dari HSV
def detect_color(hsv_roi):
    avg = np.mean(hsv_roi.reshape(-1, 3), axis=0)
    h, s, v = avg

    if (h < 10 or h > 160) and s > 100:
        return "Merah"
    elif 20 < h < 35:
        return "Kuning"
    elif 35 <= h <= 85:
        return "Hijau"
    elif 85 < h <= 130:
        return "Biru"
    else:
        return "Tidak Jelas"

# Mulai kamera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))
    blur = cv2.GaussianBlur(frame, (5, 5), 0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    # Mask merah (dua range)
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([180, 255, 255])

    # Mask kuning
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([35, 255, 255])

    # Gabungkan semua mask
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask3 = cv2.inRange(hsv, lower_yellow, upper_yellow)
    mask = mask1 | mask2 | mask3

    # Hilangkan noise
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=1)

    # Kontur dari mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for c in contours:
        area = cv2.contourArea(c)
        if area > 1500:
            approx = cv2.approxPolyDP(c, 0.03 * cv2.arcLength(c, True), True)
            x, y, w, h = cv2.boundingRect(approx)

            roi_hsv = hsv[y:y+h, x:x+w]
            warna = detect_color(roi_hsv)

            bentuk, fungsi = get_shape(approx)
            label = f"{warna}, {bentuk}, {fungsi}"

            # Gambar kotak dan label
            cv2.drawContours(frame, [approx], -1, (0, 255, 0), 2)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 0), 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 255), 2)

    cv2.imshow("Deteksi Rambu Stabil", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
