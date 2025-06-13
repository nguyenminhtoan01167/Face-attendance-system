# backend/training/Training.py

from pathlib import Path
import cv2
import face_recognition
import pickle

# --- Thiết lập đường dẫn ---
BASE_DIR         = Path(__file__).resolve().parent.parent  
DATA_DIR         = BASE_DIR / "Images"                    
OUTPUT_ENCODINGS = BASE_DIR / "training" / "encodings.pickle"

def train():
    if not DATA_DIR.is_dir():
        print(f"[ERROR] Không tìm thấy thư mục ảnh: {DATA_DIR}")
        return 0

    known_encodings = []
    known_labels    = []

    # Duyệt mỗi thư mục con (mỗi người) trong Images/
    for person_dir in DATA_DIR.iterdir():
        if not person_dir.is_dir():
            continue

        for img_path in person_dir.iterdir():
            image = cv2.imread(str(img_path))
            if image is None:
                print(f"[WARN] Không đọc được ảnh: {img_path.name}")
                continue

            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            encs = face_recognition.face_encodings(rgb)
            if not encs:
                print(f"[WARN] Không phát hiện khuôn mặt: {img_path.name}")
                continue

            known_encodings.append(encs[0])
            known_labels.append(person_dir.name)
            print(f"[INFO] Mã hoá: {img_path.name} → {person_dir.name}")

    # Tạo thư mục chứa file pickle nếu chưa có
    OUTPUT_ENCODINGS.parent.mkdir(parents=True, exist_ok=True)
    # Lưu encodings + labels
    with open(OUTPUT_ENCODINGS, "wb") as f:
        pickle.dump({
            "encodings": known_encodings,
            "labels":    known_labels
        }, f)

    print(f"Số khuôn mặt được train: {len(known_encodings)}")
    return len(known_encodings)


if __name__ == "__main__":
    train()
