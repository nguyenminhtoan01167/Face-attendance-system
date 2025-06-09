from pathlib import Path
import cv2
from datetime import datetime


class VideoProcessor:
    def __init__(self, input_file="backend/data_processing/input_info.txt", log_dir="backend/data_processing/logs"):
        self.input_file = Path(input_file)
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.student_id = None
        self.student_name = None
        self.video_path = None
        self.cap = None
        self.log_file = None

    def log(self, message):
        timestamp = datetime.now().strftime("[%d/%m/%Y %H:%M:%S]")
        print(f"{timestamp} {message}")
        if self.log_file:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(f"{timestamp} {message}\n")

    def read_input(self):
        if not self.input_file.exists():
            self.log("❌ Không tìm thấy file input_info.txt")
            return False

        with open(self.input_file, "r", encoding="utf-8") as f:
            line = f.readline().strip()
            try:
                self.student_id, self.student_name, self.video_path = map(str.strip, line.split(";"))
            except ValueError:
                self.log("❌ Định dạng file input_info.txt phải là: ID;Tên;Đường_dẫn_video")
                return False

        self.log_file = self.log_dir / f"{self.student_id}.log"
        self.log(f"📥 Nhận thông tin: ID={self.student_id}, Tên={self.student_name}, Video={self.video_path}")
        return True

    def validate_video(self):
        if not Path(self.video_path).exists():
            self.log(f"❌ Video không tồn tại: {self.video_path}")
            return False

        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            self.log("❌ Không thể mở video. Có thể lỗi codec hoặc sai định dạng.")
            return False

        self.log("✅ Đã mở video thành công.")
        return True

    def read_frames(self):
        frame_count = 0
        self.log("▶️ Đang đọc từng frame. Nhấn ESC để dừng sớm.")

        while True:
            ret, frame = self.cap.read()
            if not ret:
                self.log("📉 Kết thúc video hoặc không đọc được frame.")
                break

            frame_count += 1
            cv2.imshow("Xem trước video", frame)

            if cv2.waitKey(1) & 0xFF == 27:  # ESC
                self.log("🛑 Người dùng đã dừng sớm bằng phím ESC.")
                break

        self.cap.release()
        cv2.destroyAllWindows()
        self.log(f"✅ Đã đọc {frame_count} frames.")

    def run(self):
        self.log("========== BẮT ĐẦU XỬ LÝ VIDEO ==========")
        if not self.read_input():
            return
        if not self.validate_video():
            return
        self.read_frames()
        self.log("========== KẾT THÚC ==========\n")


if __name__ == "__main__":
    processor = VideoProcessor()
    processor.run()
