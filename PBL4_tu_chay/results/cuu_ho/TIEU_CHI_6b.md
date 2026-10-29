# TN6+4 phiên bản 2: đường lên 5 GHz tầm xa + chỉ nhu cầu mặt đất — tiêu chí chốt TRƯỚC khi chạy

Ghi ngày 28/10/2026, sau phiên bản 1 (`TIEU_CHI_6.md`) và trước khi chạy phiên bản 2 trên bất kỳ kịch bản nào.

## Chốt của nhóm (TV2, 28/10): hướng 2 và 3
- **Hướng 2:** UAV cứu hộ nối lên bằng 5 GHz qua HAPS hoặc chuỗi UAV. TV2 đặt cụ thể (`LUAT_5GHZ` trong
  `cuu_ho_hinh_hoc.py`): UAV cứu hộ – HAPS ≤ 300 km và góc ngẩng ≥ 1° (cũ: 150 km, 5°), C = 200 Mbps; UAV cứu hộ – UAV
  (có sẵn hoặc UAV cứu hộ kia) ≤ 100 km (cũ: 20 km), C = 100 Mbps. Giữ nguyên liên kết xuống mặt đất (25 km, 5°) và lên vệ
  tinh 12 GHz (15°). Mưa trên các liên kết 5 GHz tầm xa tính bằng cùng mô hình ITU-R P.838 (đường truyền thấp hơn 5 km nên
  cả quãng nằm trong lớp mưa). Với luật mới, 395/408 ứng viên thấy ít nhất một HAPS (luật cũ: 243/408).
- **Hướng 3:** chỉ tính nhu cầu mặt đất: truy vấn có cả hai đầu là user/gateway (lấy từ bộ v5 ở các ảnh chụp đã dùng);
  Φ̂ chỉ tính nút mặt đất.
- Mọi thứ khác giữ nguyên phiên bản 1: 2 UAV cứu hộ, đổi chỗ mỗi 15 phút, tốc độ 25 m/s, 408 ứng viên, mức khắc nghiệt có
  xung đột, tuổi 9, Δ = 3, bảng tra rủi ro của nhóm (bản sao trong `results/cuu_ho/dau_vao/`), router `score` α 0,8 / 0,1,
  ba cách đặt `khong`, `gan_nut_yeu`, `vet_can`.

## Lưu ý trung thực
Hai thay đổi này được chọn SAU khi xem kết quả và phân tích nguyên nhân của phiên bản 1 trên chính Yagi. Vì vậy kết quả
Yagi của phiên bản 2 là bằng chứng yếu hơn một lần kiểm tra giữ riêng. Chạy val trước (3 cơn val của bảng tra, chỉ để sửa
lỗi), rồi chạy Yagi một lần, và báo cả hai.

## Thước đo và tiêu chí (trên Yagi, truy vấn mặt đất – mặt đất)
Như phiên bản 1: tỉ lệ phục vụ; KTC 95% bootstrap theo 16 kịch bản; "hơn" = KTC nằm hẳn trên 0.
- **C1:** `vet_can` + `score` hơn `khong` + `quy_dao`.
- **C2 (chính của phiên bản 2):** `vet_can` hơn `khong` (cùng `score`) — giá trị của UAV cứu hộ khi có đường lên 5 GHz.
- **C3:** `score` hơn `quy_dao` (cùng `vet_can`).
- **C4:** `vet_can` hơn `gan_nut_yeu` (cùng `score`).
- Báo kèm: trần hậu kiểm từng cách đặt; so với phiên bản 1 trên cùng truy vấn mặt đất (82,78% → 83,13%, trần 84,72%).

Dự đoán trước: phân tích phiên bản 1 cho thấy ~56% số nút mặt đất bị cô lập có UAV cứu hộ bay được nhưng không lên được
vệ tinh; đường lên 5 GHz cứu được một phần, nhưng 2 UAV chỉ phủ 1–2 nút mỗi lúc → C2 dương, cỡ +0,3 đến +2 điểm %.

## Thay đổi
Không có (mã không đổi giữa val và Yagi).

## Kết quả (Yagi, 16 kịch bản, 8.560 truy vấn mặt đất – mặt đất; `results/cuu_ho/v2_yagi/tong_hop.json`)
| | phục vụ | trần hậu kiểm |
|---|---|---|
| khong + quy_dao | 82,78% | 84,72% |
| khong + score | 83,12% | |
| vet_can + score | 83,25% | 84,84% |
| gan_nut_yeu + score | 83,54% | 85,14% |
- C1 ĐẠT: +0,47 điểm [+0,18; +0,79]. C2 TRƯỢT: +0,13 [0,00; +0,29]. C3 ĐẠT: +0,33 [+0,09; +0,62].
  C4 TRƯỢT và ngược chiều: `vet_can` kém `gan_nut_yeu` −0,29 [−0,46; −0,11].
- Val (chạy thử) cùng chiều: C1 +0,95, C2 +0,18 [0,00; +0,45], C3 +0,81, C4 −0,05.

## Phân tích sau khi chấm (không thuộc tiêu chí)
- Cách đặt đơn giản `gan_nut_yeu` (bay tới gần nút mặt đất có tổn thất kỳ vọng lớn nhất) có ích thật và lặp lại ở cả hai
  tập: so với không có UAV cứu hộ (cùng `score`) +0,42 điểm [+0,27; +0,56] trên Yagi, +0,22 [+0,06; +0,45] trên val; cộng
  với `score`: +0,76 điểm [+0,50; +1,05] so với `khong` + `quy_dao` trên Yagi.
- Lý do khả dĩ `vet_can` thua: Φ̂ chỉ đo xác suất nối tới vệ tinh theo đường dự báo tốt nhất, nên nút nào trong dự báo
  vẫn còn một đường tốt thì thêm UAV gần như không làm Φ̂ tăng; cộng với luật hoà "bay ít nhất", UAV gần như đứng yên
  (đổi chỗ 10 lần so với 102 lần của `gan_nut_yeu`). Chưa kiểm trực tiếp độ phẳng của Φ̂ trên Yagi.
