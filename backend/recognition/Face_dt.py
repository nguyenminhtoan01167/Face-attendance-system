# backend/recognition/Face_dt.py

from pathlib import Path
import cv2
import face_recognition
import pickle
import csv
from datetime import datetime

# --- Thiết lập đường dẫn ---
BASE_DIR       = Path(__file__).resolve().parent.parent
ENC_PATH       = BASE_DIR / "training"    / "encodings.pickle"
INFO_FILE      = BASE_DIR / "recognition" / "info.txt"
ATTEND_CSV     = BASE_DIR / "database"    / "dihoc.csv"
DIST_THRESHOLD = 0.5

# --- Khởi tạo file if needed ---
INFO_FILE.parent.mkdir(parents=True, exist_ok=True)
INFO_FILE.touch(exist_ok=True)

ATTEND_CSV.parent.mkdir(parents=True, exist_ok=True)
if not ATTEND_CSV.exists():
    with open(ATTEND_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Name", "Time"])

# --- Load encodings ---
with open(ENC_PATH, "rb") as f:
    data = pickle.load(f)
known_encodings = data.get("encodings", [])
known_labels    = data.get("labels", [])

if not known_encodings:
    print("[ERROR] Chưa có face encodings! Chạy Training.py trước.")
    exit(1)

# --- Mở webcam ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("[ERROR] Không thể mở webcam.")
    exit(1)

print("Nhấn 'q' để thoát.")

recorded = set()
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize và chuyển sang RGB
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small   = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Phát hiện và encode
    face_locs = face_recognition.face_locations(rgb_small)
    face_encs = face_recognition.face_encodings(rgb_small, face_locs)

    for enc, loc in zip(face_encs, face_locs):
        dists = face_recognition.face_distance(known_encodings, enc)
        if len(dists) == 0:
            continue

        best_idx  = dists.argmin()
        best_dist = dists[best_idx]
        name      = "Unknown"

        if best_dist < DIST_THRESHOLD:
            name = known_labels[best_idx]
            if name not in recorded:
                recorded.add(name)
                ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                # Ghi info.txt
                with open(INFO_FILE, "a", encoding="utf-8") as f:
                    f.write(f"{ts} - {name}\n")

                # Ghi CSV điểm danh
                with open(ATTEND_CSV, "a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([name, ts])

                print(f"[INFO] Điểm danh: {name} lúc {ts}")
        else:
            print(f"[INFO] Unknown face (dist={best_dist:.2f})")

        # Vẽ khung và tên (scale lại ×4)
        top, right, bottom, left = [v * 4 for v in loc]
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

    cv2.imshow("Webcam Face Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
