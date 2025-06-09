import os
import cv2
import face_recognition
import pickle
from datetime import datetime
from pathlib import Path

# ========== CONFIGURATION ==========
DATASET_DIR = Path("data")                           # Thư mục chứa dữ liệu ảnh sinh viên
ENCODING_FILE = Path("backend/training/encodings.pickle")  # File lưu encoding
LOG_FILE = Path("backend/training/training.log")     # Ghi nhật ký training

# ========== UTILITY FUNCTIONS ==========

def log(message):
    timestamp = datetime.now().strftime("[%d/%m/%Y %H:%M:%S]")
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"{timestamp} {message}\n")
    print(f"{timestamp} {message}")

def is_image_file(file_name):
    return file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))

def encode_face_from_image(image_path):
    """Đọc ảnh, phát hiện và mã hóa khuôn mặt."""
    image = cv2.imread(str(image_path))
    if image is None:
        log(f"❌ Không đọc được ảnh: {image_path}")
        return None

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    boxes = face_recognition.face_locations(rgb, model="hog")  # model="cnn" nếu bạn có GPU mạnh
    if not boxes:
        log(f"⚠️ Không tìm thấy khuôn mặt trong ảnh: {image_path}")
        return None

    encodings = face_recognition.face_encodings(rgb, boxes)
    return encodings[0] if encodings else None

# ========== MAIN FUNCTION ==========

def train_all_faces():
    known_encodings = []
    known_names = []

    if not DATASET_DIR.exists():
        log("❌ Thư mục dữ liệu không tồn tại.")
        return

    student_dirs = [d for d in DATASET_DIR.iterdir() if d.is_dir()]
    if not student_dirs:
        log("⚠️ Không có dữ liệu sinh viên trong thư mục data/")
        return

    log(f"🚀 Bắt đầu training khuôn mặt từ {len(student_dirs)} sinh viên...")

    for student_folder in student_dirs:
        student_name = student_folder.name
        image_files = [f for f in student_folder.iterdir() if is_image_file(f.name)]

        if not image_files:
            log(f"⚠️ Không có ảnh hợp lệ trong thư mục: {student_name}")
            continue

        log(f"🧑 Sinh viên: {student_name} ({len(image_files)} ảnh)")

        for img_path in image_files:
            encoding = encode_face_from_image(img_path)
            if encoding is not None:
                known_encodings.append(encoding)
                known_names.append(student_name)

    if not known_encodings:
        log("❌ Không có encoding nào được tạo.")
        return

    # Lưu kết quả vào pickle
    encoding_data = {
        "encodings": known_encodings,
        "names": known_names
    }

    ENCODING_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(ENCODING_FILE, "wb") as f:
        pickle.dump(encoding_data, f)

    log(f"✅ Training hoàn tất. Tổng cộng {len(known_encodings)} khuôn mặt đã được mã hóa.")
    log(f"💾 Dữ liệu lưu tại: {ENCODING_FILE}")

# ========== RUN ==========
if __name__ == "__main__":
    log("========== BẮT ĐẦU HUẤN LUYỆN ==========")
    train_all_faces()
    log("========== KẾT THÚC ==========")
