# Demo TN6 và TN11: thống kê, học thuộc, học chưa tới, kiểm thử chéo

## Mở và sử dụng

- Mở `index.html`: bộ hình và lời dẫn; ảnh được nhúng nên xem được offline.
- Mở `duong_hoc_tuong_tac.html`: chọn một trong 21 mô hình TN11, rê chuột đọc số, zoom và xuất SVG bằng nút máy ảnh. Trang này cũng chạy offline.
- PNG độ phân giải cao dùng cho slide; SVG dùng khi cần phóng to. Bảng số nằm trong `csv/`.
- `nguon/` lưu kết quả tổng hợp đã dùng. `kiem_chung.json` lưu SHA-256 và các kiểm tra số liệu.
- Tái xuất từ repo: `python -m scripts.xuat_demo_thong_ke` (cần numpy, matplotlib, plotly; không cần GPU). Cần có kết quả TN11 gốc tại `results/bao_da_tuyen/tn11/` (gồm `cv_tep`, `cham_tep`) và 21 nhật ký tại `results/bao_da_tuyen/mo_hinh11/`. Các cache và nhật ký gốc không được kèm trên nhánh demo; bản HTML đã xuất vẫn mở được ngay mà không cần Python. `kiem_chung.json` ghi dấu vết nguồn của lần xuất, không yêu cầu mọi nguồn đều có trong gói xem demo.

## Cách ghép với nhánh demo

Nhánh https://github.com/huyth1204/PBL4/tree/demo đang có bản đồ sinh bởi `ban_do_3d.py`, template trong `assets_ban_do/` và tuyến được xuất bởi `xuat_tuyen_dm.py`. Trình xuất tuyến mặc định đọc `mo_hinh6/dm6`; bản đồ minh họa hành vi định tuyến tại từng thời điểm.

Bản đồ `ban_do_3d.html` và template `assets_ban_do/khung_than.txt` đã có nút **Thống kê TN6 / TN11**. Liên kết tương đương để dùng ở bản demo khác:

```html
<a href="results/bao_da_tuyen/demo_thong_ke/index.html" target="_blank" rel="noopener">
  Thống kê · đường học · kiểm thử chéo
</a>
```

Đường dẫn trên dùng khi bản đồ HTML ở gốc repo. Có thể mở sẵn hai tab thay vì sửa template: một tab bản đồ, một tab báo cáo này. Khi chia sẻ, tải nhánh `demo` bằng Code → Download ZIP, giải nén toàn bộ rồi mở `ban_do_3d.html` hoặc `results/bao_da_tuyen/demo_thong_ke/index.html`. GitHub xem mã nguồn HTML; muốn xem bằng một URL trực tiếp cần hosting riêng.

## Kịch bản thuyết trình 6–8 phút

### 1. Bản đồ TN6: cơ chế, khoảng 45 giây

Chọn trước một tình huống, giữ nguyên bão/mốc/nguồn/đích/tải, chuyển DM và Dijkstra phạt để giải thích tuyến chính và dự phòng. Nếu có thể, chuẩn bị cả tình huống DM có lợi và tình huống không có lợi; không lựa một khung đẹp rồi khẳng định luôn thắng.

Lời dẫn: “Bản đồ cho biết phương pháp chọn tuyến thế nào. Để biết có tốt hơn một cách đáng tin hay không, cần xem nhiều cửa sổ thời tiết và độ bất định.”

### 2. Thống kê TN6: hình 01, khoảng 60 giây

Hiển thị đủ 9 ô: 3 bộ × 3 dải tải ở K=4–5. Trái là hiệu mất dịch vụ, phải là hiệu J; đều lấy Lai6 trừ Dijkstra phạt. Chấm là hiệu trung bình, thanh ngang là khoảng tin cậy 95%. Nằm hoàn toàn bên âm ủng hộ cải thiện; cắt 0 nghĩa là chưa đủ bằng chứng, không phải chứng minh hai phương pháp bằng nhau.

Đây là số liệu TN6 gốc lấy từ `results/bao_da_tuyen/ket_qua_6.json` trên nhánh `demo`, τ=1,2. Có ô chưa thắng và có ô J tệ hơn: trình bày đầy đủ. Đơn vị mất DV là **điểm phần trăm**, không phải phần trăm giảm tương đối. Các CI trên hình chưa hiệu chỉnh Holm.

### 3. TN11: chia dữ liệu và cross-validation, hình 06 → 07, khoảng 90 giây

“Với mô hình p0, fold 0 không được dùng để học trọng số. Làm tương tự p1–p3 để mỗi bản ghi có điểm ngoài fold. Cùng một cơn bão/ngày được giữ trong cùng fold, thay vì chia ngẫu nhiên từng mốc.”

- train: 59 cửa sổ của 7 cơn bão; train6: 60 ngày; x11: 6 cửa sổ cực đoan.
- x11 giữ cùng fold với cửa sổ nguồn; trong lần chạy này cả 6 nguồn x11 đều từ các cơn bão train.
- Đây là cách chia nhóm tham lam được dự án thực hiện, không khẳng định đang gọi trực tiếp thư viện `GroupKFold`.
- Mỗi họ có 4 mô hình CV. Sau đó 3 mô hình cuối theo seed, học toàn bộ tập của họ, chấm bộ test riêng. Tổng 3 họ × (4+3)=21 mô hình.
- Tập validation riêng được dùng chọn checkpoint/dừng sớm; fold giữ lại và test không dùng chọn checkpoint.
- DM10c học A, không học x11; DM11 và SM11 học A+x11.
- Bốn fold khác mức khó; biểu đồ từng fold là mô tả, không lấy độ lệch giữa 4 fold làm CI độc lập.

Bootstrap trong báo cáo vẫn theo **cửa sổ**, còn chia CV theo **bão/ngày**. Đây là hai đơn vị khác nhau. Các cửa sổ cùng bão có thể phụ thuộc; CI theo cửa sổ chưa giải quyết hoàn toàn phụ thuộc theo bão. Nếu làm thêm bootstrap theo bão/ngày, phải đặt tên là phân tích độ nhạy mới, không thay âm thầm số gốc.

### 4. Overfitting và underfitting: hình 03–05, khoảng 90 giây

Mở trang tương tác, chọn `dm11_p2` rồi `sm11_p2`. Với SM11, loss train giảm nhưng loss val tăng về cuối; tại checkpoint tốt nhất, tỷ lệ val/train khoảng 1,49. Đây là một ví dụ thực về overfitting theo loss.

H11.4 có hai kiểm tra:

- Tỷ lệ loss val/train tại checkpoint tốt nhất ≤1,25: `sm11_p0`, `sm11_p2`, `sm11_p3` vượt ngưỡng.
- Mất DV ngoài fold trừ mất DV trên tình huống đã học ≤1 điểm % trên A và x11: SM11/x11 có trung bình khoảng 1,52, vượt ngưỡng. CI vẫn chứa 1,0, nên không nói “cận dưới CI vượt 1”.

Điểm “đã học” dùng mô hình ở fold kế tiếp, không phải cùng một mô hình được chấm hai lần. Phải giữ diễn giải chẩn đoán, không xem đó là phép đo nhân quả duy nhất về học thuộc.

Các mức 1,25 và 1,0 là tiêu chí đăng ký trước của **TN11**, không phải ngưỡng chung cho mọi mạng. Loss train và val có cách lấy nhiễu khác nhau; đây không phải phép so loss hoàn toàn đồng nhất.

Hình 05 cho thấy 0/12 mô hình CV bị cờ thiếu epoch: quy tắc là chạy đủ 120 epoch mà checkpoint tốt nhất từ epoch 108 trở đi. Tất cả đã dừng sớm trước đó. Câu được phép nói: **“Không thấy dấu hiệu bị giới hạn bởi số epoch theo tiêu chí TN11.”** Không đổi thành “đã chứng minh không underfitting”. Underfitting còn có thể do kiến trúc, đặc trưng hoặc mục tiêu; cần thử năng lực hoặc learning curve theo kích thước tập học. Cũng không mặc định các epoch đầu chính là một mô hình underfit cuối cùng.

### 5. Thống kê TN11 và kết luận: hình 08 → 02 → 09, khoảng 90 giây

Hình 08 dựng lại đúng bootstrap của báo cáo cho c7wh: trung bình 3 seed ở từng bản ghi, tính hiệu ghép cặp, lấy mẫu lại 13 cụm cửa sổ có hoàn lại 10.000 lần. Mỗi lần lấy lại toàn bộ bản ghi thuộc cửa sổ, sau đó tính trung bình có trọng số theo số bản ghi. Không coi 1.872 bản ghi hoặc 3 seed × 1.872 là các mẫu thời tiết độc lập.

Hình 02 trình bày đủ 11 bộ, không chỉ bộ thắng:

- H11.1 không đạt: thêm dữ liệu cực đoan chỉ cải thiện có ý nghĩa ở xc11, chưa đạt ≥2/4 bộ. Lợi ích xc11 còn mất ý nghĩa khi bỏ cửa sổ kề ngày.
- H11.2 đạt: DM11 thuần hơn SM11 trên 11/11 bộ. Ghi nhận đối chứng SM11 có dấu hiệu học thuộc, nên không khái quát thành “mọi diffusion luôn hơn mọi softmax”.
- H11.3 đạt: Lai(DM11) hơn Dijkstra phạt trên cả 4 bộ cực đoan; 22 ô cực đoan thắng sau Holm so với 17 của Lai(DM10c), không ô tệ hơn theo tiêu chí trong báo cáo.
- H11.4 không đạt do các cờ softmax; không vì thiếu epoch.

CI từng bộ trên hình 02 không phải CI đồng thời sau Holm. Kiểm định Holm trong báo cáo thực hiện trên họ 69 ô / mô hình seed 0 (bộ × quy mô × dải tải), khác với bảng 11 bộ gộp trung bình 3 seed. Tổng Lai(DM11): 63 ô thắng trước Holm, 58 sau Holm. Không dùng “22 so với 17” làm kiểm định trực tiếp DM11 hơn DM10c; H11.1 mới là so sánh trực tiếp đó.

Hình 09 giữ nguyên cả kết quả đầy đủ và sau loại cửa sổ kề ngày. Không có test trùng đúng ngày học, nhưng một số test kề ngày; val6 có 9/16 cửa sổ kề ngày học. Phân tích độ nhạy test **không sửa lại quá trình chọn checkpoint đã dùng val6**. Kết luận phù hợp phạm vi mô phỏng và cách chia dữ liệu hiện có.

## TN6 có thể demo overfitting và CV đến đâu?

Trong nhánh demo đã kiểm tra, có `ket_qua_6.json` nhưng chưa có thư mục nhật ký `mo_hinh6` và bảng điểm 4-fold TN6. `mo_hinh/dm_nhat_ky.json` thuộc cấu hình khác, không tự động xem là DM6. Máy nhóm có `mo_hinh6/dm6*_nhat_ky.json`, nhưng đó là các lần huấn luyện lại dùng cho TN8, nên không ghép với điểm TN6 gốc như cùng một run.

Vì vậy bộ demo này trình bày thống kê TN6 gốc; phần đường học/4-fold dùng TN11 với nguồn đã xác định. Muốn bổ sung TN6 đầy đủ cần:

1. Lấy đúng nhật ký, checkpoint và cấu hình của run tạo tuyến TN6 trên demo, hoặc ghi rõ “TN6 tái lập / huấn luyện lại”.
2. Huấn luyện lại DM6 ở 4 fold theo cùng nhóm bão/ngày nếu muốn đối chiếu CV công bằng với TN11.
3. Giữ checkpoint theo val, chấm phần ngoài fold và báo đủ số liệu. Nhiều seed của một cách chia không thay thế 4-fold CV.
4. Đối chiếu trên cùng bộ đánh giá, cách chia, seed và tiêu chí. Không trừ trực tiếp con số TN6 gốc khỏi TN11 rồi gọi đó là mức cải thiện do mô hình.

## Kiểm chứng đã thực hiện khi xuất

- Mỗi nhóm bão/ngày chỉ thuộc một fold; x11 cùng fold cửa sổ nguồn.
- Cache CV đủ số tệp theo `phan.json`; số trung bình DM11/DM10c/SM11 khớp báo cáo đã lưu.
- Trung bình và CI bootstrap c7wh tái tạo khớp báo cáo trong sai số 1e-10.
- Các đường học tương tác lấy đủ 21 mô hình; không làm mượt hoặc dựng thêm epoch.
- Bộ xuất không thay đổi nhãn, checkpoint, cache hay tiêu chí thí nghiệm gốc.

## Nguồn

- Demo gốc: https://github.com/huyth1204/PBL4/tree/demo
- TN6: https://github.com/huyth1204/PBL4/blob/demo/results/bao_da_tuyen/ket_qua_6.json (blob SHA trong `nguon/tn6_nguon.json`).
- TN11: `results/bao_da_tuyen/tn11/ket_qua_11.json`, `phan.json`, cache `cv_tep`, `cham_tep`, tiêu chí `TIEU_CHI_11.md`.
- Cross-validation theo nhóm và tách test: https://scikit-learn.org/stable/modules/cross_validation.html
- Diễn giải learning/validation curve: https://scikit-learn.org/stable/modules/learning_curve.html
