# Thí nghiệm 13 — Tỷ lệ 7/3 trong J (ghi trước khi chạy, 16/11/2026)

## Câu hỏi

J = 0,7·Ĥ + 0,3·T̂ (Ĥ: mất dịch vụ chuẩn hoá theo luồng, T̂: trễ tuyến chính chuẩn hoá). Tỷ lệ 7/3 do nhóm đề xuất. TN13 trả
lời: trong đa số trường hợp, tỷ lệ "tốt nhất" có lệch khỏi 7/3 không? Nếu đường đánh đổi gần phẳng thì chứng minh kết luận
không phụ thuộc tỷ lệ.

**Không dùng J để chấm chính tỷ lệ của nó** (tỷ lệ nào cũng thắng trên J của mình). Thước đo nằm ngoài J: đường đánh đổi vật lý
(mất dịch vụ %, trễ ms) và **điểm gối** của đường đó. Phân tích hậu nghiệm trên TN6 cho thấy chuẩn hoá theo luồng làm "giá"
của 1 điểm mất dịch vụ đổi ra ms dao động 0,3–9 ms; vì vậy thử thêm J vật lý.

## Các mục tiêu thử

- J_w = w·Ĥ + (1 − w)·T̂, w ∈ {0,3; 0,5; 0,6; 0,7; 0,8; 0,9; 1,0} (chuẩn hoá như cũ).
- J vật lý = mất DV (điểm %) + trễ (ms) / X, X ∈ {1; 3; 10; 30} ms mỗi điểm %.

## Giai đoạn A — tập kiểm định, không học lại

Sinh lại val (3 cơn bão) và val6 (16 ngày giông), 4–5 luồng. Với mỗi tình huống: thầy (TKCB, 6 điểm xuất phát, 32 thế giới
nhãn) giải lại theo **từng** mục tiêu trên **cùng** thế giới nhãn và cùng điểm xuất phát, bộ sinh riêng mỗi mục tiêu; Dijkstra
phạt chạy trên lưới β, γ ∈ {0, 2, 10, 50}² (16 bộ). Tất cả chấm trên 32 thế giới chấm. Kiểm tái lập: tình huống và nhãn thầy
gốc phải trùng bản val của TN11 (cảnh báo nếu dưới 99%, không dừng).

- Dijkstra phạt cho mục tiêu o = bộ β, γ có J_o trung bình nhỏ nhất trên val (chỉnh công bằng theo từng mục tiêu).
- **Điểm gối** của một đường (7 điểm w, theo w tăng): chuẩn hoá trễ và mất DV về [0, 1] trên chính 7 điểm; dây cung nối
  w = 0,3 và w = 1,0; điểm gối = điểm trong có khoảng cách lớn nhất về phía điểm lý tưởng (trễ thấp, mất thấp), phải > 0,05.
  Đường hẹp hơn 0,05 ms hoặc 0,05 điểm %: không có gối. Tính cho từng cửa sổ (32) và cho đường gộp; bootstrap theo cửa sổ
  (2.000 lần) cho phân bố điểm gối gộp.
- **Phẳng:** đổi w 0,5 → 0,9 làm mất DV của thầy đổi < 0,3 điểm % **và** trễ đổi < 1 ms (gộp).

**Luật rẽ nhánh (chọn 2 mục tiêu học lại ở giai đoạn B):**

| Kết quả giai đoạn A | Nhánh | Học lại DM với |
|---|---|---|
| Phẳng | `khong_phu_thuoc` — kết luận không phụ thuộc tỷ lệ | w = 0,5 và 1,0 |
| Gối gộp ∈ {0,6; 0,7; 0,8} và ≥ 2/3 số cửa sổ có gối cũng ở đó | `73_tot_nhat` — 7/3 tốt nhất theo nghĩa điểm gối | w = 0,5 và 0,9 |
| Gối gộp ∈ {0,6; 0,7; 0,8} nhưng dưới 2/3 cửa sổ | `gan_73_khong_on_dinh` | w = 0,5 và 0,9 |
| Không có gối gộp (và không phẳng) | `khong_diem_goi` | w = 0,5 và 1,0 |
| Gối gộp ở chỗ khác (w*) | `khac` — 7/3 chưa tối ưu, báo w* | w* và X* (J vật lý gần điểm gối nhất) |

## Giai đoạn B — học lại và chấm trên bộ kiểm tra

Gán nhãn lại tập học (train10, train6_10, x11 — cùng tình huống, chỉ đổi mục tiêu của thầy) cho 2 mục tiêu đã chọn; học DM13
(cấu hình DM11, 3 seed) — checkpoint chọn theo J của chính mục tiêu đó trên val giai đoạn A. Sinh lại 11 bộ kiểm tra (không thầy)
để có Dijkstra phạt trên lưới β, γ; kiểm tái lập so với bộ kiểm tra TN11. DM theo 7/3 = DM11 (TN11). Chấm DM thuần (không TKCB)
5 bước × 32 chuỗi, 4–5 luồng, trung bình 3 seed.

- **H13.2 — DM hơn Dijkstra phạt ở mọi mục tiêu:** với cả 3 mục tiêu (7/3 + 2 mục tiêu B), mất DV của DM − Dijkstra phạt (bộ
  β, γ chọn cho đúng mục tiêu) có CI < 0 ở ≥ 8/11 bộ.
- **H13.3 — DM nằm dưới đường bao Dijkstra:** mất DV của DM trừ đường bao dưới (Pareto) của 16 bộ β, γ tại cùng trễ có CI < 0
  ở ≥ 8/11 bộ, cho cả 3 mục tiêu (CI bootstrap theo cửa sổ, tính lại đường bao mỗi lần; ngoài hai đầu đường bao giữ giá trị
  đầu mút).
- **H13.4 — theo nhánh:**
  - `khong_phu_thuoc` / `khong_diem_goi`: DM theo 2 mục tiêu so với DM 7/3 (11 bộ gộp): CI mất DV trong ±0,3 điểm và |trễ| < 1 ms
    → "DM không phụ thuộc tỷ lệ".
  - `73_tot_nhat` / `gan_73_khong_on_dinh`: trên 3 điểm DM (w 0,5; 0,7; 0,9) gộp, điểm gối (luật trên) là 0,7 → "7/3 cũng là
    điểm gối của chính sách DM".
  - `khac`: mô tả DM theo w* và X* so với DM 7/3 (mất DV, trễ, CI).

**Đọc tổng:** H13.2 và H13.3 đạt → ưu thế của DM so với Dijkstra phạt không do chọn 7/3. Kết luận về chính tỷ lệ 7/3 lấy theo
nhánh của giai đoạn A, xác nhận bằng H13.4.

## Chạy

`python -m scripts.tn13 toi-da` trên máy nhóm (cần dữ liệu + mô hình TN11). Gửi về `results/bao_da_tuyen/TN13_ket_qua.zip`.
Hướng dẫn: `tai_lieu/HUONG_DAN_TN12_TN13.md`.

## Thay đổi sau khi ghi

1. **17/11/2026 — sửa lỗi code, không đổi thiết kế.** Lần chạy đầu, bước `kiem-a` dừng vì một bản ghi sinh lại có 4 luồng
   còn bản gốc có 5 (so mảng tải luồng khác độ dài). Bước kiểm tái lập giờ đếm bản ghi lệch theo lý do (mốc, mức tải, số
   luồng, cạnh của kho, tải luồng, nhãn) và chạy tiếp, đúng như đã ghi: lệch chỉ là cảnh báo. Lệch ở val nhiều khả năng vì val /
   val6 được sinh từ các thí nghiệm trước (TN4–TN6) bằng phiên bản code cũ, nên tình huống không còn trùng từng bản ghi;
   `kiem_tai_lap_a.json` ghi lý do lệch để xác nhận. Giai đoạn A không phụ thuộc điều này, vì mọi mục tiêu được giải trên cùng tình huống sinh lại. Chưa xem kết quả
   giai đoạn A khi sửa.
