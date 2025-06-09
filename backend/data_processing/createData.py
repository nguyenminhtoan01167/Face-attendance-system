import cv2
import os
from pathlib import Path
from datetime import datetime

# ========== CẤU HÌNH ==========
INPUT_FILE = Path("backend/data_processing/input_info.txt")
VIDEO_DIR = Path("Getting_data_video")
LOG_FILE = Path("backend/data_processing/createData.log")

# ========== HÀM GHI LOG ==========
def log(msg):
    timestamp = datetime.now().strftime("[%d/%m/%Y %H:%M:%S]")
    print(f"{timestamp} {msg}")
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"{timestamp} {msg}\n")

# ========== HÀM ĐỌC INPUT ==========
def read_input_file():
    if not INPUT_FILE.exists():
        log("❌ Không tìm thấy file input_info.txt.")
        return None, None, None

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        line = f.readline().strip()
        try:
            id_sv, ten_sv, video_path = line.split(";")
            return id_sv.strip(), ten_sv.strip(), video_path.strip()
        except ValueError:
            log("❌ Dữ liệu trong input_info.txt phải có dạng: id;ten;path_video")
            return None, None, None

# ========== HÀM KIỂM TRA VIDEO ==========
def validate_video(path):
    if not os.path.exists(path):
        log(f"❌ Video không tồn tại: {path}")
        return False

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        log(f"❌ Không thể mở video: {path}")
        return False
    cap.release()
    return True

# ========== HÀM XEM VIDEO VÀ ĐỌC FRAME ==========
def preview_video(path):
    cap = cv2.VideoCapture(path)
    frame_count = 0
    log("▶️ Đang đọc video... Nhấn ESC để dừng xem sớm.")

    while True:
        ret, frame = cap.read()
        if not ret:
            log("📉 Hết video hoặc không đọc được frame.")
            break

        frame_count += 1
        cv2.imshow("Preview Video", frame)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            log("🛑 Người dùng dừng sớm.")
            break

    cap.release()
    cv2.destroyAllWindows()
    log(f"✅ Tổng số frame đã đọc: {frame_count}")

# ========== MAIN ==========
def main():
    log("======== BẮT ĐẦU createData.py ========")

    id_sv, ten_sv, video_path = read_input_file()
    if not all([id_sv, ten_sv, video_path]):
        return

    log(f"📥 Nhận input: ID={id_sv}, Tên={ten_sv}, Video={video_path}")

    if not validate_video(video_path):
        return

    preview_video(video_path)

    log("======== KẾT THÚC createData.py ========\n")

if __name__ == "__main__":
    main()
