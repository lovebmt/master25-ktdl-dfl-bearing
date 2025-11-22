# flwr-demo

## Hướng dẫn sử dụng

### 1. Cài đặt môi trường và các thư viện cần thiết

- Tạo môi trường ảo (nên dùng):

```bash
python3 -m venv .venv
source .venv/bin/activate
```

- Cài đặt các thư viện cần thiết:

```bash
pip install -r requirements.txt
```

### 2. Chạy script `run_dfl.py`

- Chạy script ở thư mục gốc hoặc thư mục `release/`:

```bash
python run_dfl.py
```

- Nếu chạy trong thư mục `release/`:

```bash
cd release
python run_dfl.py
```


### 3. Xem báo cáo

- Báo cáo PDF:
  - File: `release/report.pdf` (mở trực tiếp để xem báo cáo chính thức)
- Báo cáo kết quả DFL:
  - File: `reports_dfl/dfl_results.json`

### 4. Xem slide trình bày

- Mở file `release/presentation.html` bằng trình duyệt web để xem slide trình bày.

### 5. Xem hình ảnh kết quả

- Các file ảnh kết quả được lưu trong thư mục `reports_dfl/` với định dạng `.png` (ví dụ: `reports_dfl/*.png`).

---