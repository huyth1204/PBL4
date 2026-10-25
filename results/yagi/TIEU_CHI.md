# Thí nghiệm "bão Yagi + nhiều luồng" — tiêu chí chốt TRƯỚC khi chạy

Ghi ngày 16/10/2026, trước khi tạo dữ liệu, huấn luyện hay chấm bất cứ phương pháp nào. Mọi thay đổi sau thời điểm
này phải ghi rõ ở cuối tệp kèm lý do (xem lịch sử git).

## Giả thuyết
Khi nhiều luồng khẩn cấp phải chia nhau dung lượng (lựa chọn của luồng này làm nghẽn luồng kia), một mô hình khuếch
tán (DM) sinh lựa chọn cho MỌI luồng cùng lúc, tinh chỉnh dần qua các bước khử nhiễu, sẽ tìm được tổ hợp tuyến tốt hơn
các cách chọn tham lam và tốt hơn một mô hình softmax đoán một lần với cùng kiến trúc, đồng thời nhanh hơn nhiều so
với tìm kiếm cục bộ.

## Bài toán
- Mạng 73 nút của TV1, thời tiết thật (ERA5 + CAPE), bộ mô phỏng `kich_ban.py` mức "thường" (không thêm luật hỏng hóc mới).
- Mỗi tình huống: một snapshot t; K ~ U{20..40} luồng khẩn cấp; nguồn = các nút mặt đất có mức bão nặng nhất giờ đó
  (theo mưa và gió giật), đích = các gateway ngoài vùng bão; nhu cầu mỗi luồng ngẫu nhiên rồi nhân cùng một hệ số để
  đạt mức tải.
- Mỗi luồng có 8 tuyến ứng viên (Yen, trên trễ nền lúc t, theo luật chuyển tiếp của TV1). Một lời giải = chọn 1 tuyến
  cho mỗi luồng.
- Chi phí = trễ trung bình theo lưu lượng (ms): Σ_e F_e·d_e(F_e) / Σ λ, với d_e = truyền lan + PKT/C·(1 + h(ρ)) + xử lý,
  ρ = tải nền + tải luồng / dung lượng (đã giảm do mưa), h(ρ) = ρ/(1−ρ) tới ρ = 0,99 rồi kéo dài tuyến tính.
  Báo thêm: % lưu lượng đi qua liên kết quá tải (ρ ≥ 1).
- Ba mức tải, định nghĩa bằng tải lớn nhất mà Dijkstra độc lập gây ra trên một liên kết: nhẹ 0,80; vừa 0,95; nặng 1,10.

## Dữ liệu
- Kiểm tra (giữ riêng): bão Yagi, 16 kịch bản bắt đầu mỗi 3 giờ từ 2024-09-07T00:00 đến 2024-09-08T21:00 UTC, mỗi kịch
  bản 30 snapshot (cách nhau 20 snapshot) × 3 mức tải.
- Huấn luyện / val: các ngày bão, mưa lớn khác trong 2024–2025 và các giờ mưa ngẫu nhiên, LOẠI toàn bộ 2024-09-04 → 2024-09-12.
- Lời giải "tốt nhất tìm được" = tìm kiếm cục bộ (đổi tuyến từng luồng tới khi không cải thiện) chạy R = 10 lần từ các
  điểm xuất phát khác nhau, lấy lời giải tốt nhất. Dùng làm nhãn huấn luyện và làm mốc tính khoảng cách (gap).

## Phương pháp so sánh
1. Dijkstra độc lập (mỗi luồng tuyến ngắn nhất trên trễ nền).
2. Dijkstra tuần tự: lần lượt từng luồng (nhu cầu lớn trước), Dijkstra trên trễ đã cộng tải của các luồng trước.
3. Tìm kiếm cục bộ 1 lần (xuất phát từ 2) và R = 10 lần (mốc tốt nhất).
4. Cận dưới khi cho phép chia luồng (Frank–Wolfe) — chỉ để tham khảo.
5. Softmax: cùng kiến trúc với DM, đoán lựa chọn của mọi luồng trong một lần; lấy S = 16 mẫu, chọn mẫu tốt nhất.
6. DM: khử nhiễu đồng thời lựa chọn của mọi luồng; S = 16 mẫu, chọn mẫu tốt nhất. Số bước khử nhiễu chọn trên val.

## Tiêu chí thắng (xét riêng từng mức tải, trên tập Yagi)
DM "thắng" ở một mức tải nếu đạt CẢ BỐN:
- C1. Chi phí trung bình của DM thấp hơn Dijkstra tuần tự; khoảng tin cậy 95% của hiệu (bootstrap ghép cặp theo 16
  kịch bản Yagi) nằm hoàn toàn dưới 0.
- C2. Như C1 nhưng so với softmax (cùng S = 16 mẫu).
- C3. Khoảng cách trung bình tới lời giải tốt nhất tìm được ≤ 2%.
- C4. Thời gian trung vị mỗi tình huống ≤ 1/10 của tìm kiếm cục bộ R = 10 lần (cùng máy, cùng 1 luồng CPU).
Dự đoán trước: hiệu ứng (nếu có) rõ nhất ở mức vừa và nặng; ở mức nhẹ các phương pháp gần như ngang nhau.
Kết quả sẽ được báo đầy đủ cả khi DM không đạt.

## Thay đổi (ghi trước khi có bất kỳ kết quả nào)
- 18/10/2026 — **Định nghĩa mức tải.** Định nghĩa cũ (ρ lớn nhất do Dijkstra độc lập gây ra = 0,80 / 0,95 / 1,10)
  không dùng được: lúc bão, mưa làm giảm dung lượng nên tải nền một số liên kết trên tuyến chính đã vượt 0,80 (tới
  1,26) trước khi có luồng nào; bỏ các tình huống này sẽ loại đúng các lúc bão nặng nhất. Định nghĩa mới: phần dung
  lượng còn trống C·(1−ρ_nền) bị các luồng dùng hết tại liên kết nghẽn nhất khi chọn Dijkstra độc lập — nhẹ 60%,
  vừa 90%, nặng 120% (với tải nền điển hình 0,5 trùng các mốc cũ). Tiêu chí thắng C1–C4 giữ nguyên.
- 18/10/2026 — Nguồn và đích chỉ chọn trong các nút còn liên lạc được (mưa có thể cắt hết liên kết của một nút, ví
  dụ Hà Nội bị cô lập ~4% thời gian lúc Yagi đổ bộ). Danh sách nút bị cô lập được lưu cùng mỗi tình huống.
- 18/10/2026 — **Liên kết hết dung lượng.** Lúc bão, mưa có thể làm dung lượng một liên kết giảm xuống dưới tải nền
  sẵn có (ρ_nền ≥ 1). Chạy thử cho thấy các liên kết này chi phối toàn bộ chi phí (trễ trung bình hàng trăm ms, mọi
  phương pháp chênh nhau < 0,1%). Liên kết có ρ_nền ≥ 1 được coi là không dùng được cho luồng khẩn cấp (như nhà mạng
  coi liên kết không còn dung lượng trống là hỏng); số liên kết như vậy được lưu cùng mỗi tình huống.
- 18/10/2026 — **Đủ K luồng.** Khi một cặp nguồn – đích bốc ngẫu nhiên không có tuyến, bốc lại (tối đa 20K lần) tới khi
  đủ K luồng; trước đây luồng đó bị bỏ nên K thực tế nhỏ hơn K đã bốc. Tình huống không đủ K luồng bị loại.
- 18/10/2026 — **Khảo sát chỉ với đối chứng (công khai).** Để kiểm tra bài toán có ý nghĩa, đã chạy các phương pháp
  đối chứng (không có DM, chưa huấn luyện gì) trên 40 snapshot của 4 giờ thuộc giai đoạn Yagi (07/9 00h, 06h, 12h;
  08/9 06h), 34 tình huống hợp lệ. Khoảng cách trung bình tới tìm kiếm cục bộ R = 10: Dijkstra độc lập +0,4% / +7,5% /
  +383% (nhẹ / vừa / nặng), Dijkstra tuần tự +0,1% / +2,0% / +6,5%, tìm kiếm cục bộ 1 lần +0,00% / +0,03% / +0,11%.
  Nghĩa là ở quy mô 20–40 luồng, tìm kiếm cục bộ 1 lần đã gần như tối ưu; DM khó hơn được về chất lượng, câu hỏi chính
  trở thành tốc độ – chất lượng khi số luồng tăng. Vì vậy thêm thí nghiệm quy mô bên dưới. Không có tham số nào của
  bài toán hay của DM được chỉnh theo khảo sát này.
- 18/10/2026 — **Cài đặt tìm kiếm cục bộ.** Tính chi phí biên của cả 8 ứng viên của một luồng trong một phép tính mảng
  thay vì vòng lặp từng ứng viên. Kết quả giống hệt bản cũ (48/48 tình huống thử, cùng lời giải, cùng chi phí), nhanh
  hơn khoảng 5 lần (trung vị LS R = 10: 235 ms → 47 ms). Làm trước khi đo thời gian DM để so sánh C4 công bằng (so với
  một cài đặt đối chứng hợp lý, không phải bản chậm).
- 18/10/2026 — **Sửa lỗi chọn nguồn (phát hiện khi kiểm tra số tình huống hợp lệ của tập test, trước khi huấn
  luyện hay chấm mô hình nào).** Lỗi 1: một nút được coi là "còn liên lạc" nếu tới được một gateway bất kỳ — kể cả
  chính nó hoặc một nguồn khác — nên lúc Yagi đổ bộ, Hà Nội và Hải Phòng (chỉ còn nối với nhau) vẫn được chọn làm
  nguồn dù không có đích; các lần bốc luồng từ hai nút này đều hỏng và 3/16 kịch bản chỉ còn 8–19/30 snapshot hợp lệ.
  Lỗi 2: khi vùng bão bị cắt hẳn, nút không mưa cũng có thể thành nguồn. Sửa: nguồn là nút mặt đất có mức bão > 0,
  nặng nhất trước, tới được ít nhất một gateway khác chính nó và khác các nguồn đã chọn; chỉ bốc luồng từ nguồn có
  đích. Nút bão nặng hơn bị bỏ qua vì bị cắt khỏi mạng được lưu cùng tình huống (để báo cáo). Sinh lại toàn bộ dữ
  liệu train / val / test.

## Thí nghiệm phụ: quy mô số luồng (đăng ký 18/10/2026, trước khi tạo dữ liệu và huấn luyện)
- Câu hỏi: khi số luồng tăng, thời gian và chất lượng của DM so với tìm kiếm cục bộ thay đổi thế nào?
- Tập kiểm tra: cùng 16 kịch bản Yagi, cùng 30 snapshot mỗi kịch bản, cùng 3 mức tải, nhưng K cố định = 80 và 160
  (cách sinh luồng, tuyến ứng viên, định nghĩa mức tải giữ nguyên).
- Mô hình: dùng đúng DM và softmax đã huấn luyện cho thí nghiệm chính (K = 20–40), KHÔNG huấn luyện lại — đồng thời
  kiểm tra khả năng tổng quát hoá sang số luồng lớn hơn.
- Báo cho từng K ∈ {20–40, 80, 160} × từng mức tải: thời gian trung vị mỗi tình huống của mọi phương pháp (1 luồng
  CPU); khoảng cách trung bình tới LS R = 10 (khoảng tin cậy 95%, bootstrap theo 16 kịch bản); C1–C4 đạt hay không
  (cùng định nghĩa như trên); thêm "tìm kiếm cục bộ cùng thời gian": xuất phát từ tham lam trên ứng viên, dừng khi
  dùng hết đúng thời gian DM đã dùng cho tình huống đó.
- Dự đoán trước: thời gian tìm kiếm cục bộ tăng ít nhất tỉ lệ thuận với K, thời gian DM tăng chậm hơn (tính song song
  theo mảng), nên tỉ lệ thời gian LS R = 10 / DM tăng theo K; khoảng cách của DM có thể tăng do không huấn luyện lại.
- Báo đủ mọi ô (3 giá trị K × 3 mức tải), không chọn ô đẹp.

## Ghi chú: định nghĩa "mạng 73 nút" vs "mạng mở rộng 101 nút"

Ghi 25/10/2026, sau khi đã có kết quả. Không đổi tiêu chí nào ở trên; mục này chỉ
nói rõ các kết quả nằm trên cấu hình nào, để không trộn hai cấu hình trong một bảng.

- Toàn bộ dữ liệu Yagi của thí nghiệm này dựng từ `dataset/oracle_dataset_v2.npz`
  (**73 nút**) và `data/thoi_tiet_vn.npz` (**23 điểm** thời tiết). "73 nút" không
  phải 51+22: 73 = 23 nút cũ (có thời tiết) + 50 vệ tinh LEO. Mạng mở rộng
  `oracle_dataset_v3.npz` thêm 14 điểm mặt đất, 12 UAV đặt cố định và 2 HAPS mới
  ⇒ 101 nút; trong đó 51 nút có thời tiết, đúng bằng 23 + 14 + 12 + 2.
- Vì vậy **8 UAV mới không có mặt** trong các kết quả Yagi đã báo (chúng chỉ xuất
  hiện từ cấu hình mở rộng). Kết quả vẫn đúng như đã ghi, nhưng khi viết báo cáo
  phải ghi rõ "trên mạng 73 nút", không được ngầm hiểu là mạng 101 nút.
- **Yagi làm tập test, vậy test nằm ở đâu khi dùng cấu hình mở rộng.** Nếu chạy lại
  toàn bộ trên 101 nút, quyết định là: giữ nguyên protocol — loại 2024-09-04 →
  2024-09-12 khỏi train/val, 16 kịch bản Yagi (07–08/9, mỗi 3 giờ, 30 snapshot)
  là test. Không cắt thêm ngày nào, và không lấy một phần Yagi làm val: val là
  ngày khác (12 ngày, 15% số ngày), chọn checkpoint trên val, nên test Yagi vẫn
  chưa từng được nhìn thấy khi chọn mô hình.
- Hệ quả phải chấp nhận: tập test là 16 kịch bản × 30 snapshot × 3 mức tải. Khoảng
  tin cậy 95% bootstrap theo 16 kịch bản, không theo 480 tình huống — đừng báo độ
  phân tán nhỏ hơn thực tế. Tập test này cố định, mọi so sánh phải dùng đúng nó;
  muốn thêm thí nghiệm thì thêm, không đổi tập test.
- Chưa làm: chưa có dữ liệu train/val/test cho mạng 101 nút. `data/yagi/` đã bị
  xoá; muốn có số cho cấu hình mở rộng thì phải chạy lại `tao_du_lieu_nl` trỏ vào
  `oracle_dataset_v3.npz` + `thoi_tiet_vn_mr.npz` rồi huấn luyện lại. Chưa ai làm,
  và không được trộn số của hai cấu hình trong cùng một bảng.
