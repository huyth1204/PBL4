# Thí nghiệm 5 "GNN + DM so với GNN + softmax, định tuyến đơn luồng trong bối cảnh bão, mạng v5" — tiêu chí chốt TRƯỚC khi chạy

Ghi ngày 27/10/2026, trước khi sinh dữ liệu chính, huấn luyện hay chấm bất kỳ mô hình nào của thí nghiệm này. Mọi thay
đổi sau thời điểm này ghi ở cuối tệp kèm lý do.

## Câu hỏi
Trên mạng mở rộng v5 của TV1 (224 nút: 150 vệ tinh Starlink thật chọn để phủ Việt Nam, 7 HAPS, 35 UAV, 32 nút mặt đất;
thời tiết ERA5 74 điểm, có giông), khi bão, mô hình khuếch tán (GNN + DM) có định tuyến tốt hơn mô hình đoán một lần cùng
kiến trúc (GNN + softmax) không, và cả hai có tốt hơn Dijkstra + lịch quỹ đạo không?

## Bài toán (giữ nguyên bài toán "chọn tuyến bền trong cửa sổ Δ" của hướng 4, `src/ngau_nhien`)
- Mỗi ảnh chụp t, mỗi cặp nguồn – đích lấy từ `oracle_dataset_v5.npz`: bộ định tuyến chọn một tuyến từ thông tin lúc
  t − tuổi, rồi giữ tuyến đó trong Δ ảnh chụp tiếp theo. Chi phí = trễ trung bình trên cửa sổ, mỗi ảnh chụp tuyến bị đứt
  phạt 1.000 ms. Nhãn huấn luyện = tuyến hậu kiểm (Dijkstra trên trễ trung bình của cửa sổ, chỉ trong các liên kết mô hình
  thấy được).
- Đầu vào của mô hình: **W dự báo từ lịch quỹ đạo** (`--dau-vao quy_dao`) + 8 đặc trưng nút (tải, mưa, giông, gió giật,
  gây nhiễu, nút tắt, giờ địa phương) — cùng thông tin với đối chứng cổ điển mạnh nhất.
- Hai thiết lập:
  - **E1 — bão, thông tin mới:** mức "thường", tuổi 0, Δ = 9 (giữ tuyến 3 phút).
  - **E2 — bão, thông tin cũ 3 phút:** mức "khắc nghiệt", tuổi 9, Δ = 3 (giữ tuyến 1 phút).

## Dữ liệu (`--chon-gio bao`, mới thêm vào `tao_du_lieu.py`)
- Chia NGÀY 2024–2025 như các thí nghiệm trước (15% ngày cho val), loại 04–12/9/2024. Mỗi cửa sổ thời tiết 6 giờ (đúng độ
  dài mà bộ mô phỏng đọc cho 600 ảnh chụp) nằm trọn trong ngày của tập; các cửa sổ cùng tập rời nhau. 70% cửa sổ có mức bão ≥ 10 (mưa mm/h + phần gió giật vượt 15 m/s),
  30% chỉ cần có mưa.
- train: 12 cửa sổ × 1/2 số ảnh chụp train của v5 (147); val: 3 cửa sổ × 1/2 ảnh chụp val (24), có lưu cửa sổ tương lai
  để chọn guidance; **test: 16 kịch bản Yagi** của thí nghiệm 1–3 (bắt đầu mỗi 3 giờ từ 07/9 00h UTC), ảnh chụp test của v5:
  E1 lấy 1/2 (27), E2 lấy hết (54). Seed dữ liệu: E1 = 51, E2 = 52.

## Mô hình
- GNN 4 lớp, H = 64 (như các bản trước). DM: `--head diffusion2 --param x0ce --ddim 2`; softmax: `--head classifier`.
  12 epoch. **Hai seed huấn luyện (0 và 1) cho mỗi mô hình**; chi phí mỗi truy vấn lấy trung bình hai seed.
- Guidance của DM chọn trên val trong {1, 2, 3} theo chi phí thực (G nhỏ nhất, trung bình hai seed), như `chay_tat_ca.sh`.
- Đối chứng: Dijkstra trên W đo được; Dijkstra + lịch quỹ đạo (mạnh nhất); hậu kiểm (cận dưới).

## Thước đo và tiêu chí (xét riêng E1 và E2)
G = chi phí trung bình / chi phí hậu kiểm trung bình − 1 (%). Truy vấn mô hình không tới đích tính như đứt suốt cửa sổ.
Khoảng tin cậy 95% của hiệu G bằng bootstrap theo 16 kịch bản Yagi (B = 10.000, `src/ngau_nhien/cham_bao.py`).
- **K1 (chính):** DM tốt hơn softmax — KTC 95% của G_DM − G_softmax nằm hẳn dưới 0.
- **K2:** DM tốt hơn Dijkstra + lịch quỹ đạo — KTC 95% của G_DM − G_quỹ đạo dưới 0.
- **K3:** softmax tốt hơn Dijkstra + lịch quỹ đạo — như K2.
- Báo kèm: tỉ lệ tới đích, tỉ lệ đứt, G theo từng kịch bản, từng seed.
- Phân tích phụ (không thuộc tiêu chí): các truy vấn có nguồn hoặc đích ở vùng bão Bắc Bộ (vĩ độ ≥ 19,5°, kinh độ ≥ 105,5°).

Dự đoán trước (dựa trên hướng 4): DM ngang softmax (K1 trượt ở cả hai thiết lập); ở E1 thông tin mới, Dijkstra + lịch quỹ
đạo khó bị vượt; ở E2 thông tin cũ, mô hình học có cơ hội vượt.

## Công khai: lần chạy thử trước khi chốt (đã nhìn một cửa sổ Yagi)
Chạy thử bộ sinh dữ liệu trên v5 (1 cửa sổ mỗi tập, đầu vào W đo được, Δ = 9) và huấn luyện DM 2 epoch để đo thời gian
(22 s/epoch cho 4.399 đường). Cửa sổ test của lần thử là kịch bản Yagi 07/9 00h; kết quả đối chứng ở đó: Dijkstra trên W
đo được G = 3.069%, đứt 54%; Dijkstra + lịch quỹ đạo G = 13,8%. Hiện tượng giống hệt trên cửa sổ val (G = 2.601%, đứt
44%): trên v5, liên kết vệ tinh đổi rất nhanh nên chỉ nhìn W đo được thì tuyến đứt gần một nửa thời gian — mô hình không có
cách biết trước. Vì vậy chốt đầu vào là W dự báo từ lịch quỹ đạo cho CẢ HAI thiết lập (hướng 4 cũ dùng W đo được cho E1).
Lần thử không đụng tới mô hình chính, không chỉnh tham số mô hình nào theo nó.

## Thay đổi (ghi trước khi có kết quả)
Không có thay đổi nào về dữ liệu, mô hình, tham số hay tiêu chí. Chỉ có các sự cố vận hành, ghi lại cho minh bạch
(ghi ngày 27/10, khi đã có kết quả E1 nhưng E2 chưa chấm):
- Lệnh nền chạy `chay.sh` bị dừng ở giới hạn 2 giờ khi mô hình E1_sm_s0 đang ở epoch 6. Tệp `.pt` dở dang đã xoá và
  huấn luyện lại từ đầu (cùng seed). `chay.sh` sửa để chỉ coi mô hình là xong khi có `_results.json`, thêm `MOT_BUOC=1`
  để chạy từng mô hình một lệnh.
- E1 được chọn guidance và chấm ngay khi đủ 4 mô hình E1 (cùng lệnh như `chay.sh`), trước khi huấn luyện E2. Kết quả E1
  không dùng để chỉnh gì ở E2.
- E2_dm_s0 huấn luyện xong đủ 12 epoch rồi bị tràn RAM ở bước rollout chẩn đoán trên test của `train_gnn.py` (gom ma trận
  W của 23 nghìn truy vấn). Sửa `eval_rollout` để không gom (kết quả giống hệt, kiểm trên 800 truy vấn), thêm
  `--chi-danh-gia` và chạy lại riêng phần đánh giá sau huấn luyện trên checkpoint đã lưu. Phần chẩn đoán này không thuộc
  tiêu chí; tiêu chí chấm bằng `cham_bao.py`.
- Máy chạy khởi động lại khi E2_sm_s0 mới nạp dữ liệu (chưa xong epoch nào); huấn luyện lại từ đầu.

## Phân tích bổ sung SAU khi chấm (27/10/2026, không thuộc tiêu chí): sửa lỗi dự báo quỹ đạo
Nhóm phát hiện và sửa lỗi trong W dự báo theo lịch quỹ đạo (`src/ngau_nhien/du_bao.py`, commit df8a1df): liên kết tới một
nút đã tắt lúc quan sát (UAV đã hạ cánh, nút mất do sự cố) vẫn hiện là sống nếu ở đầu kia là vệ tinh chưa xuất hiện lúc
quan sát. Toàn bộ thí nghiệm 5 ở trên chạy với dự báo CŨ (có lỗi). Ghi trước khi chạy phần bổ sung:
- Sinh lại E1, E2 với đúng tham số, seed của `chay.sh` nhưng dùng dự báo đã sửa, vào `data/bao_v5f_E{1,2}` (giữ bộ cũ).
- So đối chứng Dijkstra + lịch quỹ đạo cũ và mới trên 16 kịch bản Yagi.
- Chấm lại 4 mô hình đã huấn luyện (KHÔNG huấn luyện lại, giữ guidance đã chọn: E1 w = 1, E2 w = 2) trên tập test mới,
  bằng `cham_bao.py`, ra `results/bao_v5/sua_du_bao/`. Đây là phân tích bổ sung; kết quả chính và tiêu chí ở trên giữ nguyên.
Kết quả phần bổ sung (`results/bao_v5/sua_du_bao/cham_E{1,2}.json`):
- E1: mọi con số giống hệt (tuổi 0 nên lỗi không thể xảy ra).
- E2: W đầu vào khác ở 741/863 ảnh chụp nhưng tuyến Dijkstra + lịch quỹ đạo chỉ đổi ở 3/23.479 truy vấn (bớt 1 truy vấn
  đứt); G 62,0% -> 61,7%. Mô hình chấm lại: softmax 55,5% -> 55,8%, DM 60,0% -> 59,7%. DM − softmax +3,9 điểm
  [+0,2; +7,2]; DM − quỹ đạo −2,1 [−12,5; +12,8]; softmax − quỹ đạo −5,9 [−16,4; +8,7]. Kết luận K1–K3 không đổi.
