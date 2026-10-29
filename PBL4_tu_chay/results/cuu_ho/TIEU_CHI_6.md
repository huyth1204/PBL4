# Thí nghiệm 6+4 "2 UAV cứu hộ + định tuyến theo score khi bão" — tiêu chí chốt TRƯỚC khi chạy

Ghi ngày 27/10/2026, trước khi viết xong mô phỏng và trước khi chạy bất kỳ kịch bản nào. Thay đổi sau thời điểm này ghi ở
cuối tệp kèm lý do.

## Chốt của nhóm (TV2, 27/10)
- 2 UAV cứu hộ; được di chuyển giữa hai lần cập nhật (mỗi 15 phút).
- Tính cả truy vấn có đầu là UAV.
- Bật xung đột (mức mô phỏng "khắc nghiệt" đầy đủ).

## Giả định TV2 tự đặt (chỉnh được nếu nhóm muốn)
- Mạng v5 (224 nút). Thiết lập như E2 của thí nghiệm 5 và như bảng tra rủi ro: tuổi thông tin 9 ảnh chụp (3 phút),
  giữ tuyến Δ = 3 (1 phút, tính cả bước hiện tại).
- Vị trí ứng viên của UAV cứu hộ: lưới 10 km quanh mỗi nút mặt đất (user, gateway), cách nút tối đa 20 km, độ cao 2,5 km
  như UAV đặt cố định của TV1; gộp các điểm cách nhau dưới 5 km. Liên kết theo đúng luật của TV1 (`src/physics`): tới mặt
  đất ≤ 25 km và góc ngẩng ≥ 5°, tới HAPS ≤ 150 km và ≥ 5°, tới UAV ≤ 20 km, tới vệ tinh LEO góc ngẩng ≥ 15°; trễ =
  lan truyền + 12/C·(1 + 0,3/0,7) + 0,5 ms. Thời tiết của UAV cứu hộ lấy ở điểm thời tiết gần nhất.
- UAV cứu hộ chịu cùng luật hạ cánh (giông hoặc gió giật > 15 m/s tại điểm thời tiết của nó), chỉ làm trạm chuyển tiếp,
  không là đầu truy vấn. Tải: chỉ theo giờ trong ngày (không nhiễu ngẫu nhiên).
- Tốc độ 25 m/s (như UAV tuần tra của TV1) → mỗi 15 phút đi tối đa 22,5 km. Đổi chỗ coi như tức thời ở lần cập nhật;
  lần đặt đầu tiên của mỗi kịch bản được chọn tự do (UAV đã triển khai sẵn).
- Rủi ro đứt từng liên kết: bảng tra 3.000 ô của nhóm (`ma_tran_rui_ro.py`, học trên 7 cơn train, không có xung đột),
  áp dụng nguyên, kể cả cho liên kết của UAV cứu hộ (cùng đặc trưng: băng tần, mưa, xu hướng, gió/giông ở UAV).

## Hai tầng quyết định
- Tầng chậm (mỗi 15 phút, ở các mốc 54 + 45k): chọn vị trí 2 UAV cứu hộ.
  - `khong`: không có UAV cứu hộ.
  - `gan_nut_yeu`: tính xác suất mỗi nút đầu cuối nối được tới vệ tinh (dự báo + bảng tra); mỗi UAV bay tới ứng viên gần
    nhất (trong tầm 22,5 km, không bị hạ cánh) có liên kết với nút có tổn thất kỳ vọng lớn nhất chưa được phủ.
  - `vet_can`: chọn cặp vị trí làm lớn nhất Φ̂ = trung bình (theo tần suất làm đầu truy vấn) của xác suất nút đầu cuối
    nối được tới vệ tinh. Lần đầu: tham lam từng UAV trên mọi ứng viên; các lần sau: vét cạn mọi cặp trong tầm.
  - Φ̂ chỉ dùng thông tin đã quan sát (lúc t − 9) + lịch quỹ đạo + bảng tra.
- Tầng nhanh (mỗi truy vấn): `quy_dao` = Dijkstra trên W dự báo theo lịch quỹ đạo (đối chứng của thí nghiệm 5);
  `score` = chọn trong các tuyến ứng viên (quét λ) tuyến có S = α·(trễ/trễ nhanh nhất − 1) + (1 − α)·P(đứt) nhỏ nhất,
  α = 0,8 ở giai đoạn nhẹ và 0,1 ở giai đoạn bão (từ thí nghiệm nhỏ, không chỉnh thêm); giai đoạn cập nhật mỗi 15 phút
  theo mưa > 11 mm/h hoặc gió > 15 m/s trong 150 km quanh nguồn/đích.

## Dữ liệu
- Chạy thử (được sửa lỗi, không chỉnh α hay luật): 3 cơn val của bảng tra (Prapiroon, Trami, Kalmaegi; 16 cửa sổ của
  danh mục nhóm), ảnh chụp val của v5.
- Kiểm tra (chạy MỘT lần sau khi chốt mã): 16 kịch bản Yagi (bắt đầu mỗi 3 giờ từ 07/9 00h UTC), ảnh chụp test của v5,
  mọi truy vấn của bộ v5 ở các ảnh chụp đó (gồm cả đầu là UAV, gồm cả truy vấn không còn tuyến).

## Thước đo và tiêu chí (trên Yagi)
Tỉ lệ phục vụ = phần truy vấn có tuyến được chọn còn sống ở mọi ảnh chụp t..t+Δ (không tìm được tuyến = không phục vụ).
Khoảng tin cậy 95% của hiệu tỉ lệ phục vụ: bootstrap theo 16 kịch bản (B = 10.000). "Hơn" = KTC nằm hẳn trên 0.
- **C1 (chính):** `vet_can` + `score` phục vụ nhiều hơn `khong` + `quy_dao`.
- **C2:** `vet_can` hơn `khong` (cùng router `score`) — giá trị của UAV cứu hộ.
- **C3:** `score` hơn `quy_dao` (cùng `vet_can`) — giá trị của định tuyến theo score.
- **C4:** `vet_can` hơn `gan_nut_yeu` (cùng `score`).
- Báo kèm: tách truy vấn mặt đất – mặt đất và truy vấn có đầu UAV; trễ trung bình của truy vấn được phục vụ; trần
  "hậu kiểm" (có tồn tại tuyến sống trong mạng thực khi đã có UAV cứu hộ).

Dự đoán trước: phần lớn truy vấn không phục vụ được là do đầu UAV đã hạ cánh hoặc nút mất hết liên kết; 2 UAV chỉ phủ được
vài nút nên C2 dương nhưng nhỏ (dưới 2 điểm %); C3 nhỏ; C4 không chắc.

## Thay đổi (trước khi chạy kịch bản nào)
- 27/10: tải của UAV cứu hộ đổi từ "chỉ theo giờ trong ngày" sang "tải của nút mạng nền gần nhất" (cùng nhiễu AR(1), cùng
  đợt tăng tải do sự cố). Lý do: khi kiểm bộ dựng liên kết bằng cách đặt một ứng viên đúng chỗ UAV-HaiPhong, sống/đứt khớp
  99,98% (24 chỗ lệch là liên kết vệ tinh sát ngưỡng 15°) nhưng trễ chiều đi của ứng viên chỉ bằng 0,68 lần UAV thật, vì
  bỏ nhiễu tải làm hàng đợi ngắn hơn — UAV cứu hộ sẽ được lợi không công bằng.
- 27/10: chạy thử trên 1 cửa sổ val phát hiện scipy bỏ qua cạnh trọng số < ~1e-8 (Φ̂ luôn bằng 0) → đổi 1e-9 thành 1e-6;
  thêm luật hoà Φ̂ thì ưu tiên bay ít nhất. Không đổi α, luật hay thước đo.
- 27/10: lần chạy Yagi đầu tiên bị hết RAM (TV2 chạy song song một phép kiểm khác) khi chưa xong kịch bản nào, không có
  kết quả nào được ghi; chạy lại nguyên trạng.

## Kết quả (Yagi, 16 kịch bản, 38.400 truy vấn; `results/cuu_ho/yagi/tong_hop.json`)
| | tất cả | mặt đất – mặt đất | có đầu UAV |
|---|---|---|---|
| trần hậu kiểm (không UAV cứu hộ) | 59,47% | 84,72% | 52,23% |
| khong + quy_dao | 58,11% | 82,78% | 51,03% |
| vet_can + score | 58,27% | 83,13% | 51,14% |
- C1 ĐẠT nhưng rất nhỏ: +0,16 điểm [+0,07; +0,26]. C2 TRƯỢT: UAV cứu hộ +0,02 điểm [−0,03; +0,06]. C3 ĐẠT: score +0,14
  điểm [+0,06; +0,23]. C4 TRƯỢT: −0,02 [−0,08; +0,04]. Trên val (chạy thử) cùng chiều: C2 +0,04, C3 +0,36.
- `vet_can` chỉ đổi chỗ 5 lần trong 16 kịch bản (Φ̂ gần như không tăng ở chỗ nào), `gan_nut_yeu` 94 lần.

## Phân tích sau khi chấm (không thuộc tiêu chí): vì sao UAV cứu hộ gần như vô dụng
Trên 4 kịch bản Yagi (03, 06, 08, 10; 9.600 truy vấn), 4.598 truy vấn không có tuyến sống nào kể cả khi biết trước:
- 3.685 (80%): đầu truy vấn là UAV đã hạ cánh (hoặc nút bị tắt) — không UAV cứu hộ nào cứu được.
- 514 (11%): nút mặt đất mất hết liên kết. Kể cả có UAV cứu hộ ở MỌI vị trí ứng viên, chỉ cứu được 12; 216 ca mọi ứng
  viên gần đó cũng phải hạ cánh (cùng vùng gió > 15 m/s); 286 ca UAV cứu hộ bay được nhưng không lên được vệ tinh/HAPS
  (liên kết 12 GHz lên vệ tinh cũng đứt vì mưa tại chỗ, HAPS ngoài tầm).
- 399 (9%): mạng bị cắt ở giữa.
