import cv2
import numpy as np

def deteksi_bentuk(cnt):
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
    if len(approx) > 8:
        return "Lingkaran"
    elif len(approx) == 3:
        return "Segitiga"
    elif len(approx) == 4:
        return "Persegi Panjang"
    elif len(approx) == 7:
        return "Panah"
    else:
        return "Tidak diketahui"

def prediksi_fungsi(warna, bentuk):
    if warna == "Merah" and bentuk == "Lingkaran":
        return "STOP"
    elif warna == "Biru" and bentuk == "Lingkaran":
        return "LURUS / UMUM"
    elif warna == "Biru" and "Panah" in bentuk:
        return "ARAH (BEL0K/PUTAR)"
    elif warna == "Biru":
        return "PETUNJUK"
    else:
        return "Tidak diketahui"

# Buka kamera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize & salin frame
    frame = cv2.resize(frame, (640, 480))
    output = frame.copy()

    # Konversi ke HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Deteksi warna merah
    mask_merah = cv2.inRange(hsv, np.array([0,100,100]), np.array([10,255,255])) + \
                 cv2.inRange(hsv, np.array([160,100,100]), np.array([179,255,255]))

    # Deteksi warna biru
    mask_biru = cv2.inRange(hsv, np.array([100,150,0]), np.array([140,255,255]))

    warna_masks = {
        "Merah": mask_merah,
        "Biru": mask_biru
    }

    for warna, mask in warna_masks.items():
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 1000:
                bentuk = deteksi_bentuk(cnt)
                fungsi = prediksi_fungsi(warna, bentuk)

                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)

                label = f"{warna}, {bentuk}, {fungsi}"
                cv2.putText(output, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    cv2.imshow("Deteksi Rambu", output)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()