# Mở rộng ngẫu nhiên: mưa thật, giông, sự cố xung đột, tải hàng đợi

Mục đích: đưa độ bất định thật vào mạng SAGSIN 73 nút của TV1, rồi đo xem mô hình học (DM, softmax) có chỗ
để vượt cách làm cổ điển hay không. Tài liệu đầy đủ (công thức, kết quả): `tai_lieu/Mo_rong_ngau_nhien.pdf`.

## Các tệp

| Tệp | Việc |
|---|---|
| `thoi_tiet.py` | tải thời tiết theo giờ từ Open-Meteo (ERA5) cho 23 điểm; hoặc sinh bản giả lập cùng định dạng |
| `kich_ban.py` | tách trễ gốc của TV1, dựng lại trễ theo mưa (ITU-R P.838), giông, xung đột, tải M/M/1 |
| `khoang_loi.py` | tính khoảng lợi trước khi huấn luyện: Dijkstra hiện tại, Dijkstra + lịch quỹ đạo, tối ưu hậu kiểm |
| `tao_du_lieu.py` | tạo `train/val/test.npz` (cùng định dạng cũ + `node_extra`) với nhãn hậu kiểm |
| `danh_gia_mo_hinh.py` | chấm mô hình bằng chi phí thực trong cửa sổ Δ |
| `chay_tat_ca.sh` | chạy toàn bộ từ đầu tới cuối |
| `../../test/test_ngau_nhien.py` | 10 test |

Thay đổi nhỏ ở mã cũ (tương thích ngược, 18 test cũ vẫn đạt): `data_v2.py` và `train_gnn.py` nhận thêm
đặc trưng nút theo snapshot `node_extra` nếu tệp `.npz` có.

## Chạy

```bash
pip install requests                                   # để tải Open-Meteo
python -m src.ngau_nhien.thoi_tiet tai --start 2024-01-01 --end 2025-12-31 --out data/thoi_tiet_vn.npz
python -m src.ngau_nhien.thoi_tiet them-giong --vao data/thoi_tiet_vn.npz     # ERA5 không có giông → lấy CAPE
python -m src.ngau_nhien.thoi_tiet xem data/thoi_tiet_vn.npz                  # in thống kê giông theo năm, tháng
python test/test_ngau_nhien.py dataset/oracle_dataset_v2.npz
bash src/ngau_nhien/chay_tat_ca.sh                     # hoặc: bash src/ngau_nhien/chay_tat_ca.sh gia-lap
```

`dataset/oracle_dataset_v2.npz` và `dataset/split_info.json` là tệp của TV1 (600 snapshot liên tục, chia theo pha).

### Cửa sổ thời tiết của `tao_du_lieu.py`

Mặc định (`--chon-gio roi`) các cửa sổ ~6 giờ được **rải rời nhau** trên trục thời gian và
**loại 04–12/9/2024** (bão Yagi). Bản cũ chọn độc lập từng kịch bản nên các cửa sổ chồng
lên nhau và train/val/test dùng chung thời tiết. `meta.json` ghi `h0` từng tập và script
tự kiểm 3 tập không chung giờ nào (in `Đã kiểm: cửa sổ thời tiết ... rời nhau`).

| Cờ | Việc |
|---|---|
| `--chon-gio roi` | mặc định: rời nhau + loại ngày bão |
| `--chon-gio theo-ngay` | chia train/val theo NGÀY rồi mới rải cửa sổ (luật của `tao_du_lieu_nl`) |
| `--chon-gio doc-lap` | cách cũ, để tái lập số đã công bố |
| `--ty-le-ngay-rong` | chia trục thời gian thành 3 đoạn train → val → test liền nhau |
| `--ngay-loai D0:D1` | đổi khoảng ngày cấm; `--ngay-loai khong` để tắt |

## Đọc kết quả

- `G ngây thơ`: Dijkstra trên W đo được thua tối ưu hậu kiểm bao nhiêu % (có phạt mỗi snapshot tuyến bị đứt).
- `H`: như trên nhưng cho Dijkstra + lịch quỹ đạo + giữ nguyên trạng thái ngẫu nhiên hiện tại (đối chứng cổ điển mạnh
  nhất). **H là phần mà một mô hình học có thể lấy lại**; với dữ liệu tất định của TV1, H = 0 đúng tuyệt đối.
- `tuổi`: thông tin cũ bao nhiêu snapshot (20 s); `Δ`: tuyến được giữ bao nhiêu snapshot.
- Phạt đứt tuyến mặc định 1000 ms mỗi snapshot, lớn hơn trễ lớn nhất của mọi tuyến còn sống, để tối ưu hậu kiểm
  luôn là cận dưới (cột `vi_pham_can_duoi_%` trong JSON phải bằng 0).
- Giông: ERA5 không suy ra được giông, nên `them-giong` lấy CAPE từ kho dự báo lưu trữ của Open-Meteo và dùng luật
  CAPE ≥ 1000 J/kg và mưa ≥ 1 mm/h (mã giông của mô hình dự báo không ổn định theo thời gian nên không dùng).
