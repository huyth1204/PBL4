# Chạy lại 4 thí nghiệm từ đầu

Mỗi tệp tự làm trọn quy trình: sinh dữ liệu → huấn luyện (nếu có) → chấm → in bảng kết quả, rồi so với kết quả gốc trong
`results/`. Chạy từ thư mục gốc dự án (thư mục có `src/`, `data/`, `dataset/`), bằng Terminal hoặc chuột phải → Run trong
PyCharm (Working directory = thư mục gốc).

| Thứ tự | Lệnh | Thí nghiệm | Thời gian (2 lõi CPU) |
|---|---|---|---|
| 1 | `python chay_tu_dau/1_bang_tra_rui_ro.py` | Bảng tra rủi ro bão 3.000 ô | 15–25 phút |
| 2 | `python chay_tu_dau/2_score_toc_do_dam_bao.py` | Chọn trọng số score tốc độ / đảm bảo | 30–40 phút |
| 3 | `python chay_tu_dau/3_tn6_4_uav_cuu_ho.py` | TN6+4: 2 UAV cứu hộ + định tuyến theo score | 1–1,5 giờ (`--chi-v2`: 30–40 phút) |
| 4 | `python chay_tu_dau/4_tn5_dm_vs_softmax.py` | TN5: GNN + DM so với GNN + softmax | 9–10 giờ (`--threads 4` nhanh hơn) |

- Tệp 3 cần bảng tra do tệp 1 tạo ra; muốn chạy tệp 3 trước thì thêm `--bang-co-san` để dùng bản sao nhóm đã dùng.
- Thêm `--thu` vào bất kỳ lệnh nào để chạy thử bản nhỏ (vài phút đến khoảng 20 phút) kiểm tra máy trước khi chạy thật.
  Kết quả chạy thử chỉ để kiểm tra, không có ý nghĩa khoa học.
- Dừng giữa chừng (Ctrl+C) rồi chạy lại cùng lệnh: bước nào xong rồi sẽ được bỏ qua.
- Máy 8 GB RAM: chạy lần lượt từng tệp, không chạy hai tệp cùng lúc.

## Kết quả ghi ở đâu
`ket_qua_tu_chay/<thí nghiệm>/` (có `.gitignore` riêng nên git không đưa lên): dữ liệu sinh ra, mô hình, tệp chấm và
`nhat_ky/` (nhật ký từng bước, xem khi có lỗi). Thư mục `results/` giữ kết quả gốc để so, không bị ghi đè.

## Kỳ vọng khi so với kết quả gốc
- Bảng tra (1) và TN6+4 (3): mô phỏng có seed cố định, cùng phiên bản thư viện thì phải KHỚP tuyệt đối (mã băm bảng tra,
  từng truy vấn của TN6+4). Khác phiên bản numpy/scipy có thể lệch rất nhỏ.
- Score (2): kết quả gốc chạy trước khi nhóm sửa lỗi dự báo quỹ đạo (27/10), mã hiện tại đã có bản sửa → E1 khớp, E2 có thể
  lệch nhẹ.
- TN5 (4): huấn luyện lại 8 mô hình trên dữ liệu đã sửa dự báo; mạng nơ-ron trên máy/luồng CPU khác cho số hơi khác. So
  xem kết luận (K1–K3) có giữ nguyên không, không cần khớp từng số.

## Thư viện
Python 3.10–3.13 với numpy, scipy, matplotlib, torch (bản CPU là đủ; tệp 2–4 đều cần torch vì mã mô phỏng
nhập chung gói `src`): `pip install -r chay_tu_dau/yeu_cau.txt`.
Tiêu chí ghi trước của từng thí nghiệm: `results/bao_v5/TIEU_CHI_5.md`, `results/cuu_ho/TIEU_CHI_6.md`, `TIEU_CHI_6b.md`;
tài liệu bảng tra: `dataset/BANG_TRA_RUI_RO.md`.
