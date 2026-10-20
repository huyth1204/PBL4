# Thí nghiệm 2 "Yagi + nhiều luồng, DM tự sinh tuyến, chi phí thấp" — tiêu chí chốt TRƯỚC khi chạy

Ghi ngày 20/10/2026, trước khi sinh dữ liệu, huấn luyện hay chấm bất cứ phương pháp nào trên bài toán mới. Mọi thay
đổi sau thời điểm này ghi ở cuối tệp kèm lý do (xem lịch sử git). Nối tiếp thí nghiệm 1 (`results/yagi/TIEU_CHI.md`,
kết quả: DM thắng 0/9 ô; điểm nghẽn là 8 tuyến ứng viên cố định và thời gian DM tăng theo số luồng).

## Giả thuyết
(H1) Bỏ giới hạn 8 tuyến ứng viên, cho DM sinh "bản đồ cạnh nên dùng" cho từng cặp nguồn – đích trên toàn mạng rồi
dựng tuyến bằng Dijkstra có xét tải theo bản đồ đó, sẽ cho lời giải tốt hơn Dijkstra tuần tự, tốt hơn mô hình đoán một
lần cùng kiến trúc, và (ở tải nặng) tốt hơn lời giải tốt nhất trong 8 ứng viên của thí nghiệm 1.
(H3) Vì mạng chạy theo cặp nguồn – đích (5–12 cặp mỗi tình huống, không phụ thuộc số luồng) với ít mẫu và ít bước,
thời gian DM tăng chậm hơn tìm kiếm cục bộ khi số luồng tăng.

## Bài toán
- Giữ nguyên mọi thứ của thí nghiệm 1 (mạng 73 nút, mô phỏng mức "thường", cách chọn nguồn / đích / số luồng / λ,
  ba mức tải, chi phí = trễ trung bình theo lưu lượng với hàng đợi M/M/1), CÙNG các tình huống (cùng hạt giống).
- Khác duy nhất: mỗi luồng được đi BẤT KỲ đường đơn hợp lệ nào trên các liên kết còn sống có tải nền < 1, theo luật
  chuyển tiếp của TV1 (nút người dùng không chuyển tiếp), không chia nhỏ luồng.

## Dữ liệu
- Cùng 16 kịch bản Yagi (giữ riêng), cùng ba quy mô K = 20–40, 80, 160; cùng 600 / 60 kịch bản train / val ngoài
  04–12/9/2024. Huấn luyện chỉ ở K = 20–40.
- Lời giải "tốt nhất tìm được" (mốc, và nhãn huấn luyện) = TKCB-KGH 10 lần: tìm kiếm cục bộ không giới hạn tuyến —
  lần lượt từng luồng, bỏ tải của nó ra, tìm đường Dijkstra theo chi phí biên Σ[f(F+λ) − f(F)] trên toàn mạng, đổi nếu
  rẻ hơn; lặp tới khi một vòng không đổi (tối đa 30 vòng). 10 điểm xuất phát: Dijkstra tuần tự, lời giải TKCB 10 lần
  trên 8 ứng viên (thí nghiệm 1), Dijkstra độc lập, và 7 lần Dijkstra tuần tự với thứ tự luồng ngẫu nhiên.

## Phương pháp so sánh
1. Dijkstra độc lập; 2. Dijkstra tuần tự (luồng lớn trước, trễ đã cộng tải các luồng trước) — như thí nghiệm 1.
3. TKCB-KGH 1 lần (xuất phát từ Dijkstra tuần tự) và 10 lần (mốc).
4. TKCB 10 lần trên 8 ứng viên (mốc của thí nghiệm 1).
5. Cận dưới Frank–Wolfe (chia được luồng) — tham khảo.
6. Một-lần: cùng mạng với DM, đoán bản đồ cạnh từ trạng thái "chưa biết gì" trong một lần; 4 lần dựng tuyến với 4
   thứ tự luồng (lần đầu: luồng lớn trước, 3 lần ngẫu nhiên), giữ lời giải rẻ nhất.
7. DM: khử nhiễu (Gauss, dự đoán x0, DDIM) bản đồ cạnh của MỌI cặp nguồn – đích cùng lúc; S = 4 mẫu; mỗi mẫu dựng tuyến
   một lần (luồng lớn trước), giữ lời giải rẻ nhất.
- Cách dựng tuyến (chung cho 6 và 7): lần lượt từng luồng, Dijkstra trên trọng số = chi phí biên trên mỗi đơn vị lưu
  lượng (ms) + γ·(−ln(ŷ + 0,01)), ŷ = bản đồ cạnh của cặp nguồn – đích đó (cạnh ngoài vùng tính bản đồ: ŷ = 0).
  γ = 0 chính là Dijkstra tuần tự.
- Mạng chỉ tính trên "hành lang" của mỗi cặp: cạnh (u, v) có d0(s→u) + d0(u,v) + d0(v→d) ≤ 3·d0(s→d) + 20 ms
  (d0 = trễ khi chưa có luồng khẩn cấp). Việc dựng tuyến vẫn dùng toàn mạng.
- Chọn trên val (không dùng Yagi): số bước DM ∈ {2, 3, 5} và γ ∈ {0,3; 1; 3} ms — chọn cấu hình có khoảng cách trung bình
  tới mốc nhỏ nhất trong các cấu hình có thời gian trung vị ≤ 1/10 của TKCB-KGH 10 lần trên val; nếu không cấu hình
  nào đạt thì chọn khoảng cách nhỏ nhất. Một-lần chọn γ trên val theo khoảng cách nhỏ nhất.
- Báo thêm (không thuộc tiêu chí): DM + TKCB-KGH (tìm kiếm cục bộ xuất phát từ lời giải DM).

## Tiêu chí thắng (xét riêng từng ô: 3 quy mô × 3 mức tải, trên tập Yagi)
DM "thắng" ở một ô nếu đạt CẢ NĂM:
- C1. Chi phí trung bình thấp hơn Dijkstra tuần tự; khoảng tin cậy 95% của hiệu (ms; bootstrap ghép cặp theo 16
  kịch bản Yagi) nằm hoàn toàn dưới 0.
- C2. Như C1, so với Một-lần.
- C3. Khoảng cách trung bình tới mốc (TKCB-KGH 10 lần) ≤ 2%.
- C4. Thời gian trung vị mỗi tình huống ≤ 1/10 của TKCB-KGH 10 lần (cùng tiến trình, 1 luồng CPU).
- C5. Như C1, so với TKCB 10 lần trên 8 ứng viên (lợi ích của việc tự sinh tuyến).
Dự đoán trước: C5 chỉ có khả năng đạt ở tải nặng (ở tải nhẹ, vừa giới hạn ứng viên gần như không gây thiệt); C4 dễ
đạt ở 80 và 160 luồng hơn ở 20–40 luồng; TKCB-KGH 1 lần có thể vừa nhanh vừa gần mốc — sẽ báo đầy đủ.
Báo đủ cả 9 ô, kể cả khi DM không đạt.

## Thay đổi (ghi trước khi có bất kỳ kết quả nào)
- 20/10/2026 — **Số mẫu S chọn trên val (trước khi huấn luyện).** Đo thời gian các đối chứng trên 6 kịch bản val
  (87 tình huống, không phải Yagi, chưa có mô hình nào): Dijkstra tuần tự 2,5 ms, TKCB-KGH 1 lần 6,1 ms, TKCB-KGH 10 lần
  68,6 ms (trung vị). Mỗi lần dựng tuyến tốn đúng bằng một lần Dijkstra tuần tự (≈ 1/27 của TKCB-KGH 10 lần), nên với
  S = 4 mẫu, riêng phần dựng tuyến đã vượt 1/10 → C4 không thể đạt dù mạng nhanh đến đâu. Sửa: S cũng được chọn trên
  val, cùng quy tắc như số bước và γ, trong {1, 2, 4}. Một-lần dùng đúng S đã chọn cho DM (lần đầu luồng lớn trước,
  các lần sau thứ tự ngẫu nhiên). Tiêu chí C1–C5 giữ nguyên. Cũng trong lần đo này (chỉ đối chứng, val): so với
  TKCB-KGH 10 lần, Dijkstra tuần tự vượt 0,17 / 2,79 / 5,30% (nhẹ / vừa / nặng), TKCB-KGH 1 lần 0,001 / 0,11 / 0,28%,
  TKCB 10 lần trên 8 ứng viên 0,000 / 0,58 / 48,8%; nhãn không có phần nào nằm ngoài hành lang.
