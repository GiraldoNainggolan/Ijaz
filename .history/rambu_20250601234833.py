import cv2
import numpy as np

# Dictionary bentuk dan fungsinya
shape_function = {
    "circle": "STOP",
    "triangle": "BEL0K KIRI / KANAN",
    "rectangle": "LURUS",
    "octagon": "MUTER BALIK"
}

def get_shape(approx):
    sides = len(approx)
    if sides == 3:
        return "triangle"
    elif sides == 4:
        return "rectangle"
    elif sides == 8:
        return "octagon"
    else:
        return "circle"

def get_color_name(hsv_color):
    h, s, v = hsv_color
    if h < 10 or h > 160:
        return "MERAH"
    elif 10 < h < 30:
        return "KUNING"
    elif 30 < h < 85:
        return "HIJAU"
    elif 85 < h < 130:
        return "BIRU"
    else:
        return "TIDAK DIKENAL"

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize (optional)
    frame = cv2.resize(frame, (640, 480))

    # Blur dan konversi HSV
    blurred = cv2.GaussianBlur(frame, (5, 5), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # Mask warna untuk threshold semua warna dominan
    lower = np.array([0, 100, 100])
    upper = np.array([180, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)

    # Temukan kontur
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 1000:
            approx = cv2.approxPolyDP(contour, 0.04 * cv2.arcLength(contour, True), True)
            x, y, w, h = cv2.boundingRect(approx)

            shape = get_shape(approx)
            # Ambil warna dominan pada area kontur
            roi = hsv[y:y+h, x:x+w]
            avg_hsv = np.mean(roi.reshape(-1, 3), axis=0)
            color_name = get_color_name(avg_hsv)

            fungsi_rambu = shape_function.get(shape, "TIDAK DIKETAHUI")

            label = f"{color_name}, {shape}, {fungsi_rambu}"
            cv2.drawContours(frame, [approx], 0, (0, 255, 0), 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 0), 2)

    cv2.imshow("Deteksi Rambu", frame)
    
    key = cv2.waitKey(1)
    if key == 27:  # tekan ESC untuk keluar
        break

cap.release()
cv2.destroyAllWindows()
