import cv2
import os

# ======== BƯỚC 1: Đọc file input_info.txt =========
INPUT_FILE = "input_info.txt"

try:
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        line = f.readline().strip()
        id_sv, ten_sv, path_video = line.split(";")
except Exception as e:
    print("Không đọc được file input_info.txt hoặc sai định dạng.!!!")
    print("Chi tiết lỗi:", e)
    exit()

print(f"📥 Nhận dữ liệu:\n- ID: {id_sv}\n- Họ tên: {ten_sv}\n- Video: {path_video}")

# ======== BƯỚC 2: Mở video =========
cap = cv2.VideoCapture(path_video)
if not cap.isOpened():
    print("Không mở được video.")
    exit()

# ======== BƯỚC 3: Tạo thư mục lưu ảnh =========
output_dir = os.path.join("data", ten_sv)
os.makedirs(output_dir, exist_ok=True)

# ======== BƯỚC 4: Nạp mô hình phát hiện khuôn mặt Haar Cascade ========
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

frame_count = 0
saved_count = 0
max_images = 30  # Giới hạn số ảnh lưu lại

print("🚀 Đang xử lý video và lưu ảnh khuôn mặt...")

while True:
    ret, frame = cap.read()
    if not ret:
        break  # Hết video

    frame_count += 1

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        face = cv2.resize(face, (200, 200))
        img_path = os.path.join(output_dir, f"{id_sv}_{ten_sv}_{saved_count+1}.jpg")
        cv2.imwrite(img_path, face)
        saved_count += 1
        print(f"Lưu ảnh: {img_path}")

        if saved_count >= max_images:
            break

    if saved_count >= max_images:
        break

cap.release()
print(f"Đã lưu {saved_count} ảnh khuôn mặt vào thư mục: {output_dir}")
