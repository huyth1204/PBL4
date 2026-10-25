# Hướng dẫn chạy nhanh trong PyCharm (nhánh tv2-dm-gnn)

Nhánh này viết tiếp từ `master` của nhóm (commit `c9faa1e add_gnn`). Bản đầy đủ có hình: `tai_lieu/Huong_dan_PyCharm.pdf`.

## 1. Chuẩn bị (một lần)

1. Mở thư mục repo trong PyCharm: **File → Open → chọn thư mục PBL4 → Trust Project**.
2. Chọn interpreter: **Ctrl+Alt+S → Project: PBL4 → Python Interpreter**. Nếu báo lỗi (máy bạn không có `D:\PBL4\.venv` như máy TV1): **Add Interpreter → Add Local Interpreter → Virtualenv → New**, chọn Python 3.10–3.13.
3. Mở Terminal của PyCharm (**Alt+F12**, dòng lệnh phải bắt đầu bằng `(.venv)`) và cài thư viện:

   ```
   pip install -r huong_dan/yeu_cau.txt
   ```

## 2. Chạy từng bước

Ở góc trên bên phải PyCharm có ô chọn cấu hình chạy. Nhóm **Hướng dẫn PBL4** có sẵn 12 cấu hình (lấy từ thư mục `.run/`). Chọn một cấu hình rồi bấm ▶ xanh. Cách khác: chuột phải vào tệp `huong_dan/buoc_*.py` → **Run**.

| Cấu hình | Tệp | Nội dung | Thời gian |
|---|---|---|---|
| 0 · Kiểm tra máy | `huong_dan/buoc_0_kiem_tra_may.py` | Python, thư viện, dữ liệu, mô hình | vài giây |
| 1 · Dữ liệu SAGSIN | `huong_dan/buoc_1_du_lieu.py` | Dữ liệu của TV1: mạng theo tầng, trễ, số chặng | vài giây |
| 2 · Chạy các file test | `huong_dan/buoc_2_chay_test.py` | 49 test của dự án, biểu đồ xanh / đỏ | 10–60 s |
| 3 · DM định tuyến | `huong_dan/buoc_3_dm_dinh_tuyen.py` | Tuyến DM, quá trình khử nhiễu, cả tập test | vài giây |
| 4 · Bốn quy mô mạng | `huong_dan/buoc_4_quy_mo.py` | 73 → 373 nút, độ chính xác, thời gian | 10–30 s |
| 5 · Hỏng liên kết | `huong_dan/buoc_5_hong_lien_ket.py` | Bỏ ngẫu nhiên 0–30% liên kết | vài giây |
| 6 · Số đo có nhiễu | `huong_dan/buoc_6_nhieu_do.py` | DM so với Dijkstra khi số đo sai lệch | 10–30 s |
| 7 · Luật chia đôi | `huong_dan/buoc_7_luat_chia_doi.py` | Dijkstra / DM theo chất lượng đo và mưa thật | 10–30 s |
| 8 · Nhiễu do mưa | `huong_dan/buoc_8_nhieu_mua.py` | Mưa → suy hao → tốc độ → hàng đợi → nhiễu | vài giây |
| 9 · Bản đồ tuyến | `huong_dan/buoc_9_ban_do.py` | Ba bản đồ Việt Nam mở trong trình duyệt | vài giây |
| Chạy tất cả | `huong_dan/chay_tat_ca.py` | Chạy 0 → 9, mở trang kết quả trong trình duyệt | 1–6 phút |
| pytest: mọi test | thư mục `test/` | Cây test xanh / đỏ của PyCharm | 10–60 s |

Mỗi bước hiện hình trong tab **Plots** của PyCharm (hoặc các cửa sổ hình riêng, tùy cài đặt **Settings → Tools → Python Plots**), hình cuối là bảng kiểm tra: hàng xanh ✔ là đạt, hàng đỏ ✘ là lỗi kèm lý do. Terminal chỉ in một dòng tóm tắt.

## 3. Kết quả

`huong_dan/ket_qua/` (không đưa lên git): `bang_dieu_khien.html` là trang tổng hợp mọi bước; `ban_do_*.html` là bản đồ; `buoc_*_hinh*.png` là từng hình.

## 4. Các tệp mới so với master

- `huong_dan/`: 10 bước hướng dẫn, `chay_tat_ca.py`, dữ liệu demo 73 / 173 / 273 / 373 nút.
- `truc_quan_dm.py`: viết tiếp `visualization.py` của nhóm, vẽ tuyến DM, số đo nhiễu, vùng mưa trên bản đồ.
- `src/ai/train_da_quy_mo.py`, `src/ngau_nhien/`: huấn luyện nhiều quy mô, thời tiết thật (ERA5), nhiễu do mưa theo vật lý.
- `test/`: `test_gnn.py`, `test_mo_rong.py`, `test_ngau_nhien.py` và các tệp thí nghiệm (`so_sanh_dijkstra_nhieu.py`, `luat_chia_doi.py`, ...).
- `results/dmx0_4lop.pt`, `results/quy_mo/*.pt`: mô hình đã huấn luyện (mỗi tệp khoảng 0,4 MB).
- `data/thoi_tiet_vn.npz`: mưa, giông theo giờ 2024–2025 tại 23 điểm; `data/topojson/`: bản đồ nền để bản đồ chạy không cần mạng.
- `tai_lieu/`: các báo cáo PDF.
- `.run/`: cấu hình chạy cho PyCharm; `pytest.ini`: cho pytest chỉ chạy `test/test_*.py`.

Tệp của nhóm được sửa (chỉ thêm, không đổi hành vi cũ): `src/ai/data_v2.py` và `src/ai/train_gnn.py` (tham số tùy chọn: nhiễu đo khi huấn luyện, đặc trưng nút theo snapshot), `.gitignore` (cho phép các tệp `.pt` nhỏ ở trên, bỏ qua `huong_dan/ket_qua/`).
