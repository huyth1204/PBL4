# Kế hoạch thí nghiệm 8 — Hạ tầng dễ hỏng trên thời tiết thật

**Bản nháp ngày 06/11/2026.** Đây chưa phải tiêu chí chốt: `TIEU_CHI_8.md` sẽ được ghi trước khi sinh dữ liệu.
Thí nghiệm nối tiếp TN4 và TN6; TN7b không dùng làm đối chứng.

## 0. Nguyên tắc

1. **Giữ thời tiết đo thật.** Không nhân mưa hay gió như biến thể W của TN7a. Chỉ thay đổi *mô hình hỏng hóc*, tức cách UAV
   và trạm mặt đất hỏng, vì đây vốn là giả thiết của ta.
2. **Chạy hai nhánh song song** trên cùng cửa sổ, cùng luồng và cùng hạt giống:
   - nhánh A dùng giả thiết hỏng hóc do ta đặt;
   - nhánh B hiệu chỉnh theo số liệu thực tế.

   Khác biệt giữa hai nhánh chỉ đến từ mô hình hỏng hóc. Nếu kết luận "DM an toàn hơn Dijkstra" đúng ở cả hai nhánh, nó
   không phải ảo ảnh của một giả thiết duy nhất (tránh "ảo dữ liệu").
3. **Đánh giá chéo:** mô hình huấn luyện trong thế giới A được thử ở thế giới B, và ngược lại.
4. **Tiêu chí ghi trước,** báo cả khi kết quả âm.

## 1. Căn cứ thực tế (tra cứu nhanh, cần xác minh thêm trước khi chốt)

### UAV

| Yếu tố | Số liệu tìm được | Nguồn |
|---|---|---|
| Gió | DJI Matrice 350 RTK chịu gió tối đa 12 m/s, chuẩn IP55. Nghiên cứu Gao và cộng sự (2021) coi drone phổ thông chịu 10 m/s; tăng lên 15 m/s mở rộng đáng kể số giờ bay được. Thử nghiệm UAV có dây: 15 m/s. | [1], [2], [3] |
| Mưa | Drone phổ thông: 0 mm/h (không bay trong mưa). DJI Dock 2: 2 mm/h. Hai UAV thương mại trong một luận văn: 2 mm/h và < 7,6 mm/h. Loại gia cố: tới 50 mm/h. | [2], [4], [5] |
| Sét | Không tìm thấy quy định chính thức cho drone. Tài liệu ôn thi FAA Part 107 khuyên tránh trong khoảng 20–30 hải lý quanh cơn giông. NWS khuyến cáo người trú ẩn khi sét trong vòng 3 dặm (khoảng 5 km); sét có thể đánh xa cơn giông 10 dặm (khoảng 16 km) trở lên. | [6], [7] |

### Trạm mặt đất

| Yếu tố | Số liệu tìm được | Nguồn |
|---|---|---|
| Bão Yagi (09/2024) | Khoảng 6.285 vị trí trạm BTS ở 15 tỉnh mất liên lạc, **chủ yếu do mất điện**; 27 cột viễn thông gãy đổ. Ngày 8–9/9 khôi phục 3.010 trạm, chủ yếu bằng máy phát. Ngày 11/9 còn khoảng 8% mất liên lạc. Ngày 16/9 đã khôi phục 8.386/9.235 trạm. Bạch Long Vĩ mất sóng hơn 20 giờ, sau đó khôi phục bằng vệ tinh. | [8], [9], [10] |
| Nguyên nhân sự cố trạm trong bão | Báo cáo FCC DIRS (bão Bavi): nguyên nhân gồm mất điện, đường truyền và hư hỏng; mất điện chiếm nhiều nhất. | [11] |
| Sét | Việt Nam thuộc nhóm có mật độ dông sét cao nhất thế giới. Theo một nguồn thứ cấp dẫn QCVN 02:2022/BXD, mật độ khoảng 8–14 lần/km²/năm tùy tỉnh (chưa đối chiếu văn bản gốc). Không tìm thấy xác suất hỏng trạm do sét mỗi năm. | [12], [13] |
| Nguồn dự phòng | Không tìm thấy quy chuẩn số giờ ắc quy cho trạm BTS. Máy phát của hãng quảng cáo chạy tới 14 ngày với bình 1.000 lít. | [14] |

**Hệ quả cho mô hình:**
- Hỏng trạm mặt đất ngoài thực tế chủ yếu do **mất điện kéo dài hàng giờ đến hàng ngày, xảy ra theo vùng**, không phải do sét
  ngắt khoảng 1 phút như mô phỏng hiện tại.
- UAV thực tế bị giới hạn bởi **cả mưa lẫn gió**. Mô phỏng hiện tại bỏ qua mưa khi quyết định UAV có bay được hay không.

## 2. So với mô phỏng hiện tại (đo trên dữ liệu 2024–2025, 35 UAV, 32 trạm mặt đất)

| Quy tắc UAV phải hạ cánh | Tỉ lệ giờ hạ cánh | Riêng 05–08/9/2024 (Yagi) |
|---|---:|---:|
| Mô phỏng hiện tại: giông tại điểm, hoặc giật > 15 m/s | 4,14% | 25,8% |
| Như DJI M350: giật > 12 m/s, hoặc mưa > 2 mm/h | 10,19% | – |
| Như trên, thêm giông trong bán kính 20 km | **11,35%** | **44,3%** |
| UAV gia cố: giật > 15 m/s, hoặc mưa > 50 mm/h, hoặc giông trong 20 km | 5,44% | – |

- Theo ngưỡng thực tế, UAV hạ cánh nhiều gấp khoảng 2,7 lần mô phỏng hiện tại. Riêng mưa trên 2 mm/h chiếm 3,24% số giờ,
  phần mà mô phỏng hiện tại không tính.
- Trạm mặt đất có giông tại điểm 2,17% số giờ. Gió giật trên 25 m/s chỉ 0,03% số giờ, nhưng lên 3,1% trong Yagi, nên đổ cột
  hiếm nhưng tập trung vào bão.
- Mã giông dự báo có sẵn trong dữ liệu (`code_fc`) chỉ báo trước được 6,6% số giờ giông thật, nên chưa dùng được làm đặc
  trưng dự báo.

## 3. Thiết kế hai nhánh song song

**Chung cho hai nhánh:** cùng cửa sổ bộ C (ngày giông) và A (Yagi) với hạt giống kịch bản mới; cùng tải liên tục; 4–5 / 10 /
20 luồng; tiêu chí C2, C4, C5 như TN6.

### TN8-A — giả thiết (quét độ nhạy)

Ba mức giả thiết: hiện tại / vừa / nặng.
- **UAV dễ sập:** tách hai trạng thái.
  - *Hạ cánh tạm:* như hiện tại.
  - *Sập:* mất hẳn trong phần còn lại của cửa sổ. Xác suất mỗi giờ tăng theo gió giật (ngưỡng giữa 15 → 12 → 10 m/s) và
    tăng khi có giông lân cận.
- **Giông phá hủy trạm mặt đất:** mỗi giờ trạm bị phá với xác suất k × P(giông). P(giông) là phiên bản mềm của luật
  CAPE + mưa, giữ tần suất trung bình khoảng 2,2%. Mức k quét theo ba mức. Trạm bị phá tắt trong phần còn lại của cửa sổ.

### TN8-B — hiệu chỉnh theo thực tế

- **Hai loại UAV:**
  - *Phổ thông:* hạ cánh khi giật > 12 m/s, hoặc mưa > 2 mm/h, hoặc giông trong 20 km.
  - *Gia cố hoặc có dây:* hạ cánh khi giật > 15 m/s, hoặc mưa > 50 mm/h, hoặc giông trong 20 km.

  Đây là các trường đã có trong dữ liệu thời tiết theo giờ, không cần giả định thêm.
- **Trạm mặt đất — quá trình mất điện theo vùng:**
  - Xác suất mất điện lưới mỗi giờ phụ thuộc gió giật và mưa tại điểm, và tương quan giữa các trạm cùng tỉnh.
  - Trạm còn chạy thêm B giờ bằng ắc quy, với B ∈ {2; 4; 8}, rồi mới tắt.
  - Sét phá trực tiếp hiếm: xác suất mỗi giờ suy từ mật độ sét.
  - Đổ cột khi gió giật vượt ngưỡng thiết kế (tham số).
- **Hiệu chỉnh:** tỉ lệ trạm mặt đất ngừng trong đỉnh Yagi được khớp với số liệu Yagi ở mục 1, về thứ tự độ lớn, vì chưa có
  tổng số trạm theo tỉnh.

### Đánh giá chéo 2 × 2

| | Thử trong thế giới A | Thử trong thế giới B |
|---|---|---|
| Bộ định tuyến không biết thế giới đã đổi (DM6 / Lai6, bảng rủi ro giữ nguyên) | ✔ | ✔ |
| DM huấn luyện lại trong thế giới A | ✔ (trong miền) | ✔ (chéo) |
| DM huấn luyện lại trong thế giới B | ✔ (chéo) | ✔ (trong miền) |

Lợi thế của DM chỉ được coi là thật nếu còn giữ ở các ô chéo. Mỗi mô hình huấn luyện lại dùng 3 hạt giống huấn luyện để đo
độ dao động.

**Đo thêm:** tách phần "mọi tuyến cùng đứt" thành hai loại:
- *đầu cuối bị hỏng:* không thuật toán nào cứu được;
- *đường đi bị cắt:* chọn tuyến tốt hơn thì cứu được.

## 4. Kiểm soát "ảo dữ liệu"

- [ ] Ghi `TIEU_CHI_8.md` (giả thuyết, tiêu chí, dự đoán) trước khi sinh dữ liệu.
- [ ] Bộ kiểm tra dùng hạt giống kịch bản mới, không dùng khi chọn tham số hay huấn luyện.
- [ ] Hai nhánh A và B cùng cửa sổ, cùng luồng, cùng hạt giống. Chỉ mô hình hỏng hóc khác nhau.
- [ ] Đánh giá chéo như mục 3.
- [ ] Đối chiếu mô phỏng với sự cố thật: phân bố trạm hỏng theo tỉnh trong Yagi so với mô phỏng (định tính).
- [ ] Ba hạt giống huấn luyện cho mỗi mô hình huấn luyện lại; báo khoảng dao động.

## 5. Hướng triển khai thêm (gợi ý)

1. **Giảm phần "mọi tuyến cùng đứt".** Đây là chỗ DM6 vẫn kém Dijkstra phạt (19,8% so với 19,5% trên bộ C). Cách làm: đưa
   ràng buộc rời nhóm rủi ro hoặc đa dạng loại đường lên vào kho ứng viên, hoặc vào điểm của nhãn.
2. **Chọn loại UAV:** UAV có dây và UAV pin có ngưỡng chịu đựng khác nhau. Có thể thêm bài toán bố trí: nên đặt UAV gia cố ở
   đâu.
3. **Kiểm định theo năm:** dữ liệu thời tiết 2017–2021 đã có trong `data/`. Huấn luyện trên 2024–2025, kiểm tra trên năm khác.
4. **Độ nhạy tuổi thông tin** (180 s → 60 / 300 s) và độ dài cửa sổ giữ tuyến.
5. **So sánh tách thành phần:** DM với softmax và với chỉ dùng tìm kiếm cục bộ, cùng ngân sách tính toán. Đây là hạn chế báo
   cáo đang nêu.
6. **Dự báo cho "giờ tới":** `code_fc` hiện quá yếu (6,6%). Cần nguồn khác như radar nowcast; đây là hướng xa.

## 6. Thứ tự và khối lượng dự kiến (máy 8 luồng)

| Bước | Việc | Thời gian ước tính |
|---|---|---|
| 1 | Xác minh số liệu mục 1 (DJI, QCVN 02:2022/BXD, báo cáo Yagi chính thức); chốt tham số A, B và ghi `TIEU_CHI_8.md` | 0,5 ngày |
| 2 | Thêm hai mô hình hỏng hóc vào `src/ngau_nhien/kich_ban.py` (bật bằng tham số, mặc định giữ hành vi cũ) cùng kiểm thử | 0,5 ngày |
| 3 | Sinh dữ liệu kiểm tra A và B (C, A × 3 quy mô), chấm DM6 / Lai6 không huấn luyện lại | khoảng 6 giờ |
| 4 | Sinh dữ liệu huấn luyện A và B, huấn luyện lại DM (3 hạt giống mỗi thế giới) | khoảng 8–10 giờ |
| 5 | Chấm chéo 2 × 2, tách phần "mọi tuyến cùng đứt", phân tích và báo cáo | khoảng 4 giờ |

## Nguồn

1. [DJI Matrice 350 RTK — thông số (bản PDF lưu trên smlouvy.gov.cz)](https://smlouvy.gov.cz/smlouva/soubor/41121093/Matrice%20350%20RTK%20-%20Specs%20-%20DJI.pdf); [Korec — spec sheet M350 RTK](https://www.korecgroup.com/wp-content/uploads/2025/05/DJI-Matrice-350-RTK-Spec-sheet.pdf)
2. [Gao và cộng sự, Scientific Reports (2021) — giới hạn thời tiết của drone](https://www.nature.com/articles/s41598-021-91325-w)
3. [UH — Drone aided efficient wireless network (thử nghiệm gió 15 m/s)](https://hurricane.egr.uh.edu/sites/hurricane.egr.uh.edu/files/files/2017/13-DRONE-AIDED-EFFICIENT-WIRELESS-NETWORK.pdf)
4. [Heliguy — DJI Dock 2: gió và mưa tối đa](https://www.heliguy.com/blogs/knowledge-base/what-is-the-maximum-wind-speed-and-rainfall-in-which-dji-dock-2-can-continue-to-operate-normally/); [Heliguy — IP rating M350 và bay trong mưa](https://heliguy.com/blogs/knowledge-base/what-is-the-ingress-protection-ip-rating-of-m350-rtk-can-i-fly-the-drone-in-the-rain)
5. [SDU UAS Center — weather analysis (luận văn)](https://www.sdu.dk/-/media/files/om_sdu/centre/c_uas/freed/weather-analysis_tobias-lundby_sdu_ed1.pdf)
6. [Open Exam Prep — FAA Part 107: weather hazards](https://open-exam-prep.com/study-guides/faa-part-107/chapter3/weather-hazards)
7. [NWS — Lightning safety](https://www.weather.gov/ffc/lightning_safety)
8. [VnEconomy — Cục Viễn thông: còn khoảng 8% trạm mất liên lạc](https://vneconomy.vn/cuc-vien-thong-con-khoang-8-so-tram-phat-song-bi-mat-lien-lac-vi-bao.htm)
9. [VnEconomy — Dồn lực khắc phục sự cố mạng viễn thông do bão Yagi](https://vneconomy.vn/don-luc-khac-phuc-hau-qua-su-co-mang-vien-thong-do-bao-yagi-gay-ra.htm)
10. [Báo Chính phủ — Khôi phục điện, liên lạc cho trên 90% người dân vùng bão](https://en.baochinhphu.vn/power-supply-communication-resumed-to-over-90-residents-in-yagi-hit-localities-111240916180815622.htm); [Báo Lào Cai — gần 400 trạm BTS hư hỏng](https://baolaocai.vn/toan-tinh-co-gan-400-tram-thu-phat-bts-bi-hu-hong-post389939.html)
11. [FCC — Communications Status Report (DIRS)](https://docs.fcc.gov/public/attachments/DOC-423511A1.txt)
12. [Bảng mật độ sét theo tỉnh (theo QCVN 02:2022/BXD, nguồn thứ cấp)](https://thietbipcccbinhduong.com/ban-do-mat-do-set/)
13. [Tuổi Trẻ — Việt Nam thuộc nhóm quốc gia dông sét dày đặc](https://tuoitre.vn/giong-set.html)
14. [Yanmar — máy phát điện cho viễn thông](https://www.yanmar.com/vn/energy/products/power_generator/telecom_power/)
