import csv
import os

def init_db():
    filename = 'dihoc.csv'
    
    # Nếu file chưa tồn tại thì tạo mới và ghi header
    if not os.path.exists(filename):
        with open(filename, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(['ID', 'Ten', 'MaMon', 'ThoiGian'])
        print(f"Đã tạo file {filename} với header.")
    else:
        print(f"File {filename} đã tồn tại.")

# Gọi hàm khi chạy script
if __name__ == '__main__':
    init_db()

#Thêm hàm lưu điểm danh
def save_attendance(id, name, mamon, time):
    with open('dihoc.csv', mode='a', newline='', encoding='utf-8') as file:
        file.write(f"{id},{name},{mamon},{time}\n")

#Thêm hàm lấy danh sách điểm danh theo mã môn

def get_attendance(mamon):
    records = []
    try:
        with open('dihoc.csv', 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                if row['MaMon'] == mamon:
                    records.append(row)
    except FileNotFoundError:
        pass  # Có thể log lỗi nếu cần
    return records
