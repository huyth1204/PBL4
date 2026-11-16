# Thí nghiệm 12 — Cơ chế khuếch tán có thật là yếu tố tạo độ chính xác? (ghi trước khi chạy, 16/11/2026)

## Câu hỏi

TN6 và TN11 cho thấy DM hơn softmax cùng mạng 4,2–5,6 điểm %. Nhưng DM khác softmax ở nhiều thứ cùng lúc: học trên trạng
thái nhiễu, sửa dần qua nhiều bước, chạy 32 chuỗi rồi chọn, và ở mỗi bước nhìn lựa chọn tạm của các luồng khác. TN12 tách
từng thứ để biết phần lợi đến từ đâu, và loại hai phản biện: "softmax thua chỉ vì học thuộc" và "lịch cos² không có tác dụng".

## Thiết kế: thang bậc

Mọi bậc chấm trên đúng 11 bộ kiểm tra của TN11, 4–5 luồng, cùng tình huống, cùng 32 thế giới chấm; bậc có mô hình lấy trung
bình 3 seed (0, 1, 2); hạt giống giải mã theo bản ghi như TN11 (`giong_ban_ghi`).

| Bậc | Phương pháp | Thêm gì so với bậc trước | Nguồn |
|---|---|---|---|
| 0 | Dijkstra phạt | quy tắc viết tay, cùng bảng rủi ro | bản ghi TN11 |
| 1 | TKCB trên bảng rủi ro (`online_tkcb`) | tìm kiếm, không học | bản ghi TN11 |
| 2 | SM11, đoán 1 lần | học từ thầy | mô hình TN11 |
| 3 | SM11, 32 mẫu Gumbel, giữ mẫu log-xác suất cao nhất | lấy nhiều mẫu rồi chọn (cùng luật chọn với DM) | giải mã lại |
| 4 | DM11, 1 bước × 32 chuỗi | mạng học trên trạng thái nhiễu | giải mã lại |
| 5 | DM11, 5 bước × 1 chuỗi | sửa dần, không chọn mẫu | giải mã lại |
| 6 | DM11, 5 bước × 32 chuỗi | đầy đủ (= DM11 của TN11) | giải mã lại |
| 7 | DM không phối hợp (`dm12k`) | bỏ việc nhìn lựa chọn tạm của luồng khác | học lại, 3 seed |
| 8 | DM lịch nhiễu tuyến tính ᾱ = 1 − t (`dm12l`) | thay cos² | học lại, 3 seed |
| 9 | Softmax chống quá khớp (`sm12a` / `sm12b`) | dropout + weight decay (AdamW) | học lại, 3 seed |

Thêm: quét số bước {1, 2, 3, 5, 10} × số chuỗi {1, 8, 32} của DM11 seed 0 (mô tả, đường độ chính xác – thời gian).

**Bậc 7 — định nghĩa "không phối hợp".** Mỗi luồng chỉ thấy trạng thái của chính nó. Tải của luồng khác, trọng số tin nhắn
trên liên kết, liên kết nghẽn và mọi thứ đi qua liên kết đều tính từ trạng thái "chưa biết" X_deu (dòng công khai, cùng trọng
số mạng); dòng riêng của ứng viên chỉ thêm phần thuộc chính luồng (X, trùng cạnh / trùng loại với tuyến khác của luồng, ngữ
cảnh luồng). Đã kiểm: đổi lựa chọn của luồng khác thì logit của luồng đang xét không đổi (chênh 0,0).

**Bậc 9 — chọn cấu hình.** Hai cấu hình ghi trước: a = dropout 0,1 + weight decay 1e-4; b = dropout 0,3 + weight decay 1e-3.
Học seed 0 cả hai, giữ cấu hình có J val tốt nhất, rồi học seed 1, 2 của cấu hình đó. Không xem bộ kiểm tra khi chọn.

Mọi mô hình học lại: cùng dữ liệu DM11 (A + 5% cực đoan x11, đã làm sạch), cấu hình H 128, tối đa 120 epoch, dừng sớm 20
epoch theo J val, cùng val 300 tình huống.

## Giả thuyết và luật đọc

Thước đo chính: mất dịch vụ (%). "Đạt" = CI 95% (bootstrap theo cửa sổ, 10.000 lần) đúng chiều ở ít nhất 8/11 bộ.

- **H12.1 — sửa dần có tác dụng:** bậc 6 − bậc 4 < 0.
- **H12.2 — không chỉ nhờ lấy nhiều mẫu:** bậc 6 − bậc 3 < 0.
- **H12.3 — cơ chế là phối hợp giữa các luồng:** bậc 7 − bậc 6 > 0, **và** phần quá tải (b) chiếm ít nhất 60% phần chênh
  mất dịch vụ (11 bộ gộp).
- **H12.4 — softmax thua không chỉ vì học thuộc:** bậc 9 − bậc 6 > 0, **và** cả 3 mô hình bậc 9 qua luật quá khớp của TN11
  (loss val / loss train tại checkpoint ≤ 1,25). Nếu bậc 9 vẫn quá khớp: H12.4 "không kết luận được".
- **H12.5 — lịch cos²:** so bậc 8 với bậc 6. CI nằm trong ±0,3 điểm ở ≥ 8/11 bộ → "lịch nhiễu không quan trọng" (KHÔNG kết
  luận cos² tốt hơn); CI > 0 ở ≥ 8/11 bộ → "cos² tốt hơn"; CI < 0 ở ≥ 8/11 bộ → "tuyến tính tốt hơn"; còn lại "không kết luận".

**Đọc tổng:** H12.1–H12.4 cùng đạt → "cơ chế khuếch tán (sửa dần + phối hợp giữa các luồng) là yếu tố tạo độ chính xác".
Nếu bậc 3 xấp xỉ bậc 6 (H12.2 không đạt), lợi đến từ việc lấy nhiều mẫu; báo đúng như vậy.

**Mô tả thêm (không phải giả thuyết):** mọi bậc so với Dijkstra phạt; thành phần (a) hết tuyến / (b) quá tải theo bậc; bậc 5
so với bậc 4 (sửa dần một chuỗi so với 32 chuỗi một bước); đường quét; đường học các mô hình mới (cờ học chưa tới như TN11).

**Kiểm tra lặp lại:** bậc 6 phải trùng DM11 của TN11 (cùng mô hình, cùng hạt giống) nếu máy còn `tn11/cham_tep/`.

## Chạy

`python -m scripts.tn12 toi-da` trên máy nhóm (cần `results/bao_da_tuyen/mo_hinh11/` và dữ liệu bộ kiểm tra TN11). Kết quả
gửi về: `results/bao_da_tuyen/TN12_ket_qua.zip` (kèm bản ghi chấm từng tình huống và trọng số mô hình mới). Hướng dẫn:
`tai_lieu/HUONG_DAN_TN12_TN13.md`.

## Thay đổi sau khi ghi

(chưa có)
