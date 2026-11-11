# Thí nghiệm 11 — DM bền dưới thời tiết cực đoan: tiêu chí chốt TRƯỚC khi chạy

Ghi ngày 11/11/2026, trước khi sinh dữ liệu, huấn luyện hay chấm. Mọi thay đổi sau thời điểm này ghi ở cuối tệp kèm lý
do. Chạy một nơi (máy nhóm, ~20 GB RAM + GPU 8 GB), lệnh `python -m scripts.tn11 toi-da`.

## Câu hỏi

Mục tiêu không phải để DM tái tạo thầy (TN11a: phần lớn khoảng cách tới thầy còn nguyên kể cả trên tình huống đã học).
Mục tiêu là **bền dưới thời tiết xấu và cực đoan**, so với các phương pháp đối chứng.

TN11 hỏi ba câu:
1. Thêm 5% dữ liệu cực đoan vào tập học có giúp DM ở thời tiết cực đoan mà không làm hại thời tiết thường không?
2. DM (sinh tập tuyến bằng khử nhiễu) có hơn mạng softmax cùng dữ liệu, cùng độ rộng không?
3. Lai(DM) có hơn Dijkstra phạt ở thời tiết cực đoan không?

Kèm theo là một quy trình như sản phẩm: làm sạch dữ liệu, kiểm tra rò rỉ, kiểm thử chéo theo nhóm, chống học thuộc /
học chưa tới, các thước đo, biểu đồ.

## Dữ liệu

| Vai trò | Tập | Ghi chú |
|---|---|---|
| Học, thời tiết thường | `train10`, `train6_10` | như TN10 (119 cửa sổ, 13 209 tình huống, thầy 32 thế giới, có thế giới chấm) |
| Học, **cực đoan (mới)** | `x11` | 6 cửa sổ ≈ **5%** (666 tình huống); thầy + thế giới chấm |
| Chọn checkpoint / dừng sớm | `val`, `val6` | 300 tình huống cố định, như TN6–TN10 |
| Kiểm tra, thường | c6, yagi5, c6_b, yagi5_b (4–5 / 10 / 20 luồng); ngay9, nam9, bao9 (4–5) | như TN10 |
| Kiểm tra, **cực đoan** | c7wh, y7wh (TN7a; 4–5 / 10 / 20); **xc11, xy11 (mới, 4–5)** | |

**Dữ liệu cực đoan `x11` (theo đề xuất của nhóm):**
- **Kiểu đỉnh bão.** Chọn 6 cửa sổ dữ dội nhất của train + train6, mỗi cơn bão / ngày tối đa một cửa sổ, để không chỉ
  lấy một cơn bão. Độ dữ dội = đỉnh theo giờ của trung bình các điểm (mưa / 20 + gió giật / 25, mỗi phần chặn ở 1, + có
  giông).
- **Hệ số cực đoan.** Mưa, gió giật và xác suất sét ngắt trạm (`p_set`) nhân với hệ số k ~ U(1,3; 1,5), bốc riêng cho
  từng cửa sổ, seed mới.
- **Bộ định tuyến không biết k.** Như TN7a: bảng rủi ro dùng tham số hợp đồng, nhưng quan sát mưa / gió đã tăng.

**Bộ kiểm tra cực đoan mới `xc11`, `xy11`:** đúng cửa sổ của c6 / yagi5, seed mới, k ~ U(1,3; 1,5) riêng từng cửa sổ.
- c7wh / y7wh của TN7a (mưa × 1,5, gió × 1,25, UAV hạ cánh ở 12 m/s, sét dày hơn) là phép thử cực đoan theo một cách
  khác, đã có từ trước.

## Quy trình (như sản phẩm)

1. **Làm sạch** (`kiem_tra_du_lieu.py`).
   - Kiểm tra từng bản ghi: bài toán rỗng, giá trị không hữu hạn / ngoài miền (rủi ro, tải nền ∈ [0, 1), dung lượng,
     tải > 0), đặc trưng thiếu / sai độ dài, nhãn thầy ngoài miền.
   - Bản ghi lỗi bị loại khỏi học và val. Báo số lượng theo loại.
   - Báo bản ghi trùng (cửa sổ, mốc, mức tải) và thống kê đặc trưng từng tập (độ lệch phân bố thường / cực đoan).
2. **Kiểm tra rò rỉ.** Ngày của mọi cửa sổ kiểm tra không được nằm trong ±1 ngày của cửa sổ học. Có vi phạm thì báo đỏ.
3. **Kiểm thử chéo 4 phần THEO NHÓM.**
   - Nhóm là cơn bão (nửa train: 7 cơn) hoặc ngày (nửa train6). Chia cân bằng bằng tham lam.
   - Cửa sổ cực đoan nằm cùng phần với cửa sổ nguồn.
   - Cách này sửa lỗi rò theo bão của phần chia theo cửa sổ ở TN10.
4. **Chống học chưa tới / học thuộc.**
   - Học tối đa 120 epoch, cosine, B = 32, lr 1e-3 (như DM10).
   - **Giữ checkpoint theo J val, dừng sớm** khi J val không tốt lên sau 20 epoch.
   - Lưu đường học (loss train, loss val, J val) mọi epoch.
5. **Mô hình** (H = 128, 3 đặc trưng giông TN6):

   | Tên | Loại | Dữ liệu |
   |---|---|---|
   | DM11 | DM khử nhiễu, S = 32, 5 bước (cấu hình DM10) | A + x11 |
   | DM10c | DM, như DM11 | chỉ A (đối chứng cho dữ liệu cực đoan) |
   | SM11 | softmax một bước | A + x11 (đối chứng mô hình) |

   - Mỗi họ có 4 mô hình kiểm thử chéo và 3 mô hình cuối (seed 0, 1, 2), học trên toàn bộ.
   - Lai(·) = DM nếu ρ̂ < 1,1, ngược lại DM + TKCB (TN11a giữ 1,1).
6. **Đối chứng không học:** Dijkstra phạt (đối chứng chính, như mọi TN), Dijkstra ngắn nhất (Yen), và thầy (chỉ trên tập
   có thầy).
7. **Thước đo:**
   - mất DV %, chia thành (a) không còn tuyến sống, (b) dự phòng quá tải;
   - J;
   - trễ tuyến chính;
   - thời gian quyết định (ms);
   - G tới thầy (chỉ chẩn đoán);
   - CI 95% bootstrap theo cửa sổ, 10 000 lần.

## Giả thuyết (đăng ký trước)

Trung bình 3 seed theo bản ghi, 4–5 luồng.

- **H11.1 — dữ liệu cực đoan giúp ở cực đoan, không hại thời tiết thường.**
  - Lai(DM11) − Lai(DM10c) có CI cận trên < 0 ở ≥ 2/4 bộ cực đoan (c7wh, y7wh, xc11, xy11).
  - Không bộ thường nào có CI cận dưới > 0.
- **H11.2 — DM hơn softmax.** DM11 thuần − SM11 có CI cận trên < 0 ở ≥ 8/11 bộ.
- **H11.3 — Lai(DM11) bền hơn Dijkstra phạt ở cực đoan.**
  - Lai(DM11) − Dijkstra phạt có CI cận trên < 0 ở cả 4 bộ cực đoan.
  - Ô tiêu chí (C2 + C4, thêm C5 ở 10 / 20 luồng; mô hình seed 0; Holm theo họ mọi ô của mỗi mô hình): không ô nào tệ
    hơn Dijkstra phạt có ý nghĩa, và số ô thắng theo Holm trên bộ cực đoan của Lai(DM11) ≥ của Lai(DM10c).
- **H11.4 — không học chưa tới, không học thuộc** (mọi mô hình kiểm thử chéo).
  - Chưa tới: không mô hình nào chạy hết 120 epoch mà epoch tốt nhất ≥ 108.
  - Học thuộc, cần cả hai điều kiện dưới đây cho mọi mô hình:
    - loss val / loss train tại epoch tốt nhất ≤ 1,25;
    - mất DV kiểm thử chéo − mất DV trên tình huống đã học ≤ 1,0 điểm (trên A và trên x11).
- **Theo dõi, không có luật:**
  - kiểm thử chéo theo nhóm so với theo cửa sổ của TN10 (thường kém hơn, nhưng trung thực hơn);
  - thời gian C5;
  - (a) / (b) ở cực đoan;
  - G tới thầy.

## Một nơi chạy

Kết luận dựa trên một nơi. Bước / giả thuyết nào có hiệu cách ngưỡng < 0,1 điểm thì báo là chưa ổn định.

## Thay đổi

1. **11/11/2026:** ràng buộc cũ "không mô phỏng thêm thời tiết cực đoan" được nhóm thay bằng đề xuất trên: 5% dữ liệu,
   hệ số 1,3–1,5.
2. **12/11/2026, trước khi chạy** (phát hiện khi chạy thử bước kiểm tra rò rỉ):
   - Không có cửa sổ kiểm tra nào trùng đúng ngày với cửa sổ học. Nhưng có cửa sổ **kề ngày** (cách 1 ngày, cùng một
     đợt thời tiết có thể kéo dài):
     - c6 / ngay9: 2 / 13 cửa sổ (03/06/2024, 21/08/2025);
     - val6: 9 / 16 cửa sổ.
   - Sai sót này có từ thiết kế TN6–TN10.
   - Thêm **phân tích độ nhạy**: các bộ dùng cửa sổ của c6 (c6, c6_b, c7wh, xc11) và ngay9 được tính lại khi bỏ 2 cửa
     sổ đó. Giả thuyết nào đổi kết luận khi bỏ thì ghi "phụ thuộc cửa sổ kề ngày".
   - val6 chỉ dùng chọn checkpoint / dừng sớm, nên chỉ ghi nhận.
   - Luật chính không đổi.
