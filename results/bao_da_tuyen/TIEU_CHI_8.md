# Thí nghiệm 8 — Đánh giá chéo 2 × 2: thế giới A và thế giới B — tiêu chí chốt TRƯỚC khi chạy

Ghi ngày 09/11/2026, trước khi sinh dữ liệu thế giới B, huấn luyện hay chấm bất cứ thứ gì của thí nghiệm này. Mọi thay
đổi sau thời điểm này ghi ở cuối tệp kèm lý do. Kế hoạch gốc: `KE_HOACH_8.md`. Đối chứng là TN4 và TN6; TN7b không dùng.

## Câu hỏi

Kết luận "DM / Lai6 an toàn hơn Dijkstra phạt trong thời tiết xấu" có còn đúng khi đổi **giả thiết hỏng hóc của hạ
tầng**, hay chỉ đúng với giả thiết mà chính ta đặt ra? Thời tiết giữ nguyên là dữ liệu đo thật.

## Hai thế giới

Hai thế giới dùng **cùng thời tiết thật, cùng cửa sổ, cùng seed kịch bản** (cùng luồng, cùng mưa dưới giờ). Chỉ mô hình
hỏng hóc khác nhau.

| | Thế giới A (giả thiết cũ, như TN6) | Thế giới B (TN8) |
|---|---|---|
| UAV hạ cánh tạm | giông tại điểm, hoặc gió giật > 15 m/s | giông tại điểm, hoặc gió giật > ngưỡng của loại UAV |
| Loại UAV | một loại, 15 m/s | **15 m/s** cho 30 UAV: toàn bộ 25 UAV ở vùng bão (vĩ độ ≥ 16°N) và 5 UAV ngoài vùng bão nhưng nhiều gió. **12 m/s** cho 5 UAV ngoài vùng bão có ít ngày gió giật ≥ 17,2 m/s nhất trong khí hậu 2017–2021: UAV-3, UAV-5, UAV-BuonMaThuot, UAV-CaMau, UAV-PhuQuoc |
| UAV hỏng hẳn | không có | xác suất mỗi giờ P(d) = 0,30 · exp(−d / 15 km), với d là khoảng cách tới điểm thời tiết có giông gần nhất; d > 50 km → 0 |
| Trạm mặt đất | sét ngắt khi có giông tại điểm: xác suất 0,1 mỗi mốc, tắt trung bình 3 mốc (~1 phút) | bỏ ngắt ngắn; hỏng hẳn với P(d) = 0,05 · exp(−d / 5 km), d > 50 km → 0 |
| Đã hỏng | – | tắt tới hết kịch bản (không sửa kịp trong ~3 giờ 20 phút) |

- Mã: `ThamSo.hong_b` và các tham số `uav_*`, `dat_*` trong `src/ngau_nhien/kich_ban.py`; `TAP_8`, `THE_GIOI_B`,
  `loai_uav()` trong `src/nhieu_luong/tao_du_lieu_dt.py`.
- Bộ định tuyến không được báo thế giới đã đổi: bảng rủi ro và Dijkstra phạt giữ cấu hình hợp đồng. DM chỉ thấy quan
  sát (nút tắt, giông, mưa…) của thế giới đang chạy.
- Ngưỡng 12 / 15 m/s lấy từ thông số drone thương mại và loại gia cố (`KE_HOACH_8.md`, mục 1). Xác suất hỏng do giông
  là **giả thiết** có căn cứ định tính (sét có thể đánh xa cơn giông khoảng 16 km), không phải số đo.
- Kiểm tra trước khi ghi tệp này (không xem kết quả định tuyến):
  - với tham số mặc định, bộ mô phỏng cho kết quả giống hệt bản cũ (hồi quy);
  - hai bộ mô phỏng (`sinh_kich_ban` và `phat_lai`) hỏng giống hệt nhau;
  - mưa ở thế giới A và B trùng nhau;
  - trên 13 cửa sổ C: UAV tắt 11,6% thời gian ở A, 16,3% ở B; cuối cửa sổ có 13,6% UAV hỏng hẳn ở B.

## Bốn tập dữ liệu (2 × 2)

| | Thế giới A | Thế giới B |
|---|---|---|
| Học | train + train6 (dữ liệu của DM6, có sẵn) | `train_b` + `train6_b`, cùng cửa sổ + seed |
| Kiểm định | val + val6 (có sẵn) | `val_b` + `val6_b` |
| Kiểm tra | `c6`, `yagi5` (có sẵn; 4–5 / 10 / 20 luồng) | `c6_b`, `yagi5_b` (cùng cửa sổ + seed) |

## Mô hình

- **Học trong A:** DM6 (seed 0, có sẵn), cùng `dm6_s1`, `dm6_s2` (seed 1, 2).
- **Học trong B:** `dm8b_s0`, `dm8b_s1`, `dm8b_s2`.
- Cả hai họ cùng kiến trúc và cùng 3 đặc trưng giông quanh UAV; H = 64, 20 epoch, B = 32, lr 1e-3. Checkpoint chọn
  theo J trên 300 tình huống kiểm định *của thế giới mình*. Số bước khử nhiễu cố định 5, S = 8 (như DM6).
- **Chính sách Lai** như Lai6: DM nếu ρ̂ < 1,1, ngược lại DM + TKCB. τ = 1,1 cố định, không chọn lại.

## Đánh giá

- Ô tiêu chí như TN6: Lai so với Dijkstra phạt theo C2 (J) và C4 (mất dịch vụ), CI 95% bootstrap ghép cặp theo 13 cửa
  sổ; ở 10 / 20 luồng thêm C5 (≤ TKCB 10 lần và ≤ 1 s).
- **Bảng chéo chính:** mô hình seed 0 (DM6, `dm8b_s0`) × 4 bộ kiểm tra × 3 quy mô × 3 dải tải.
- **Ổn định theo seed:** cả 6 mô hình × 4 bộ kiểm tra ở 4–5 luồng.
- **Phần (a) không tránh được:** tỉ lệ lưu lượng mà không ứng viên nào trong kho sống (sàn); báo phần (a) của từng
  phương pháp vượt trên sàn.

## Giả thuyết (đăng ký trước)

- **H8a:** trên cùng cửa sổ, Dijkstra phạt mất dịch vụ ở thế giới B cao hơn ở A (báo mức tăng và thành phần a / b).
- **H8b (chống "ảo dữ liệu"):** Lai(DM6), không huấn luyện lại, vẫn thắng Dijkstra phạt ở thế giới B. Cụ thể: số ô thắng
  ít nhất một nửa số ô thắng ở thế giới A (C ≥ 3/9, Yagi ≥ 4/9), và không ô nào tệ hơn Dijkstra có ý nghĩa thống kê.
- **H8c:** Lai(DM8b) ở thế giới B thắng ít nhất bằng số ô của Lai(DM6) ở B. Ở ô chéo (DM8b trên A), số ô thắng ít nhất
  một nửa số ô của DM6 trên A.
- **H8d:** ở 4–5 luồng, dấu của Lai − Dijkstra (gộp dải tải) giống nhau ở cả 3 seed của mỗi họ, trên mỗi bộ kiểm tra.

## Dự đoán

- Dijkstra phạt mất dịch vụ tăng ở B, chủ yếu ở phần (a): UAV và trạm hỏng hẳn.
- Sàn (a) tăng rõ ở B, nên phần cải thiện còn có thể đạt được nhỏ đi.
- Lai(DM6) vẫn thắng dải tải thấp và giữa ở B (H8b đạt), nhưng ít ô hơn ở A.
- DM8b hơn DM6 ở B không nhiều, vì đặc trưng không có loại UAV và không có khoảng cách tới giông.

## Thay đổi

(chưa có)
