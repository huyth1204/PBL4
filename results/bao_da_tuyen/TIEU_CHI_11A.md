# Thí nghiệm 11a — Chẩn đoán khoảng cách DM10 → thầy: tiêu chí chốt TRƯỚC khi chạy

Ghi ngày 11/11/2026, trước khi tính bất cứ số liệu nào của thí nghiệm này. Mọi thay đổi sau thời điểm này ghi ở cuối tệp
kèm lý do. Chạy một nơi (máy nhóm đã chạy TN10), lệnh `python -m scripts.tn11a tat-ca`.

## Câu hỏi

Sau TN10, DM10 còn cách thầy G ≈ 1,06 điểm mất dịch vụ (kiểm thử chéo).
- Học lâu / rộng hơn thì lợi giảm dần, loss val tăng lại từ epoch ~40.
- Thầy tốt hơn và thêm dữ liệu B đều không giúp.

Có hai cách giải thích:

1. **Thiếu thông tin.**
   - Thầy tối ưu trên các thế giới chép nguyên tương lai thật của 12 mốc tới (`the_gioi_rr.re_nhanh`):
     - mưa nền thật `P`;
     - giông thật `giong_diem`;
     - lịch UAV hạ cánh thật `uav_tat`.
   - DM chỉ biết trạng thái ở o = t − 9.
   - Phần lời giải của thầy dựa vào việc biết trước thì không mô hình nào học được từ đầu vào hợp lệ.
2. **Thiếu năng lực / mục tiêu học chưa đúng.** DM có đủ thông tin nhưng không học ra được lời giải của thầy.

TN11a đo xem phần nào lớn hơn, rồi theo đó chọn hướng cho TN11b. Kèm theo là chọn ngưỡng ρ̂ của chính sách Lai cho DM10.
TN10 thấy trên bộ kiểm tra rằng DM10 thuần tốt hơn Lai(DM10); ở đây chọn lại ngưỡng hợp lệ trên tập học.

## Dữ liệu và mô hình

- Dùng lại tập học có thế giới chấm của TN10 (`train10`, `train6_10`, ~13 200 tình huống), cùng 4 phần kiểm thử chéo.
  Không sinh nhãn mới. Không dùng bộ kiểm tra nào.
- **Hai nửa của tập học, báo riêng. Nửa `train6` là số chính để áp luật.**
  - Nửa `train`: 59 cửa sổ cách nhau 3 giờ của chỉ 7 cơn bão. Kiểm thử chéo chia theo cửa sổ nên cùng một cơn bão rơi
    vào nhiều phần, gây rò theo bão. Mô hình có thể "nhớ" tương lai của cơn bão từ cửa sổ kề bên, làm lệch phép đo A, B.
  - Nửa `train6`: 60 ngày xấu khác nhau.
  - Cả hai nửa gộp lại chỉ để tham khảo.
- Mỗi tình huống được chấm bằng mô hình không học nó, như TN10:
  - hạt giống mẫu DM gắn theo bản ghi;
  - chấm trên thế giới chấm đã lưu;
  - CI 95% bootstrap theo cửa sổ, 10 000 lần.

## A. "DM nhìn trước" (đo phần do thông tin)

- **Đặc trưng nhìn trước** cho mỗi cạnh, tính lại từ `phat_lai` cùng seed. Đây chính là thông tin tương lai mà thế giới
  của thầy dùng:
  - `nt_mua`: **đỉnh** mưa nền thật trên [t, t+3] tại điểm của cạnh (/ 20, như đặc trưng mưa hiện có). Dùng đỉnh vì
    liên kết tính là đứt khi chỉ một mốc vượt ngưỡng;
  - `nt_giong_dat`: tỉ lệ mốc có giông thật trên (o, t+3] ở đầu trạm mặt đất (sét ngắt trạm có thể bắt đầu trước t);
  - `nt_uav_tat`: tỉ lệ mốc UAV ở đầu cạnh hạ cánh thật trên [t, t+3].
- **Kiểm tra trước khi chạy** (16 cửa sổ val trên máy người viết, 45 417 cạnh·bản ghi): `nt_uav_tat` = 0 ở mọi cạnh.
  - UAV hạ cánh trong [t, t+3] hầu như đã hạ cánh từ o, nên đã bị loại khỏi cạnh ứng viên.
  - Vì vậy phần "nhìn trước" của thầy chủ yếu là giông ở trạm mặt đất (`nt_giong_dat` > 0 ở ~5% cạnh) và mưa.
  - Vẫn giữ đặc trưng này. Tỉ lệ khác 0 trên tập học được báo ở `tn11a/bo_sung.json`.
- **Mô hình:** `nt60h128` = đúng cấu hình e60h128 của TN10 (60 epoch, H = 128, cùng seed = phần, cùng dữ liệu, cùng val),
  chỉ thêm 3 đặc trưng trên. Kiểm thử chéo 4 phần, S = 8.
- **Đối chứng:** e60h128 S = 8 của TN10 (G = 1,65), cùng máy, cùng phần.
- **Đại lượng:**
  - ΔG = mất DV(nt60h128) − mất DV(e60h128), ghép theo tình huống;
  - phần do thông tin = −ΔG / G(e60h128).
- **Luật đọc:**

  | Phần do thông tin, và CI cận trên của ΔG < 0 | Kết luận | Hướng TN11b |
  |---|---|---|
  | ≥ 50% | Thông tin là nút thắt chính | Đặc trưng dự báo hợp lệ ở o + tinh chỉnh DM theo mất DV |
  | 20–50% | Thông tin là một phần đáng kể | Cả hai hướng |
  | < 20%, hoặc CI không dưới 0 | Không phải chủ yếu do thông tin | Tinh chỉnh theo mất DV + đổi kiến trúc |

  Đặc trưng nhìn trước chỉ là bản tóm tắt, và e60h128 có thể chưa dùng hết chúng. Vì vậy "phần do thông tin" là
  **cận dưới**.
- **Báo thêm:** theo 3 mức tải, theo phần, epoch tốt nhất.

## B. G trên tình huống đã học (theo dõi, không có luật kết luận cứng)

- Mỗi tình huống ở phần f được chấm bằng mô hình e120h128_p(f+1 mod 4) của TN10, là mô hình **đã học** tình huống đó;
  S = 8.
- So với G kiểm thử chéo e120h128 S = 8 của TN10 (1,58).
- **Cách đọc:**
  - G đã học ≤ 0,5 × G kiểm thử chéo: mô hình khớp được tình huống đã học nhưng không tổng quát hoá. Đây là học thuộc,
    hợp với thiếu thông tin.
  - G đã học ≥ 0,8 × G kiểm thử chéo: không khớp nổi cả tình huống đã học, nghiêng về năng lực / mục tiêu học.

## C. Ngưỡng ρ̂ của Lai cho DM10

- DM10 = e120h128, S = 32, mô hình kiểm thử chéo của TN10.
- DM + TKCB như `cham_lai_dt`: TKCB trên thế giới bốc từ bảng, xuất phát từ lời giải DM.
- Lai(τ) = DM nếu ρ̂ < τ, ngược lại DM + TKCB, với τ ∈ {1,1; 1,3; 1,5; ∞}; ∞ nghĩa là DM thuần.
- **Luật chọn:**
  - chọn τ có mất DV kiểm thử chéo thấp nhất;
  - nếu J của τ đó tệ hơn τ = 1,1 có ý nghĩa (CI cận dưới của ΔJ > 0) thì giữ 1,1.
- τ chọn được dùng cho lần chấm bộ kiểm tra kế tiếp (TN11b), không chấm bộ kiểm tra ở TN11a.

## Thay đổi

(chưa có)
