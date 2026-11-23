# Thí nghiệm 15b — kiểm chứng lại TN15 trên dữ liệu mới, có học lại (ghi trước khi chạy, 23/11/2026)

> **Lịch sử bản ghi** (trước khi có dữ liệu mới nào):
> - Bản 1 (23/11) không học lại.
> - Bản 2 (23/11, theo yêu cầu người dùng) thêm phần học lại: thêm seed, DM 7/3 học lại cùng đường ống, quét X = 3. Phép so chính
>   chuyển sang DM 7/3 cùng đường ống.
>
> Cả hai bản đều chưa chạy thật. Lần chạy thử duy nhất dùng dữ liệu cũ (mục cuối).

## Vì sao có TN15b

Bản đánh giá `DANH_GIA_TN15.md` (mục 0, 3, 9) chỉ ra rằng phần lớn kết luận của TN15 đúng hướng, nhưng còn có thể là may mắn:
- cùng 11 bộ kiểm tra đã được dùng từ TN11 đến TN15;
- 11 bộ chỉ tương đương khoảng 6–7 bộ độc lập; riêng Yagi là **một** cơn bão;
- chỉ có 3 seed;
- DM11 học ở máy khác, với code cũ hơn DM x10;
- vài kết luận chỉ là phép chạy lại (H15.4) hoặc tính trên chính dữ liệu đã dùng để chọn (H15.0);
- cơ chế được kiểm sau khi có kết quả;
- cấu hình đề xuất chưa được chấm như một khối;
- mới thử một tỷ giá X = 10.

TN15b kiểm lại các điểm này trên một bộ dữ liệu **chưa từng nhìn**, với luật phân tích chốt trước.

## Danh sách điểm cần kiểm (từ DANH_GIA_TN15.md)

| Mã | Điểm còn mở | Kiểm ở |
|---|---|---|
| U1 | x10 nhanh hơn 7/3 mà không kém an toàn — mới thấy trên bộ kiểm tra đã dùng nhiều lần | P1, P2 |
| U2 | Hơn kém có phải do khác máy / khác code (DM11 học ở máy khác) không | P1, P2 (7/3 cùng đường ống), S14 |
| U3 | Chỉ 3 seed — độ dao động giữa các lần học | 5 seed cho x10 và 7/3 cùng đường ống |
| U4 | "x10 an toàn hơn" không qua hiệu chỉnh nhiều phép thử | S2 |
| U5 | Giải mã không tái nhiễu tốt hơn — H15.4 chỉ là phép chạy lại TN12b | P3, P4 |
| U6 | Cơ chế tỷ giá ngầm (đổi dấu tại X_eff = 10) — kiểm sau khi có kết quả | P5, S3 |
| U7 | x10 hơn Dijkstra phạt | P6, S13 |
| U8 | Thầy x10 hơn thầy 7/3 — H15.0 tính trên đúng tập val đã chọn x10 | P7 |
| U9 | Cấu hình đề xuất (x10 + không tái nhiễu + kho nhanh + vá) chưa chấm như một khối | P8, S7 |
| U10 | CI theo cụm có đáng tin không (phép thử A/A) | P0 |
| U11 | Đuôi trễ và quá tải là hai hiệu ứng tách biệt | S4 |
| U12 | x10 nằm ngoài đường biên của họ w | S5 |
| U13 | Khoảng ~1 ms còn lại so với Dijkstra do mục tiêu hay do DM bắt chước kém; X nhỏ hơn có thu hẹp được không | S6, S15 |
| U14 | Chọn seed lúc quyết định (cận trên biết trước −1,0 điểm) | S8 |
| U15 | Cấu hình giải mã 5 × 8 không tái nhiễu không kém 5 × 32 | S9 |
| U16 | "Trễ" chỉ là trễ danh định lúc quyết định — trễ thực khi bão | S10 |
| U17 | x10 ở 10 luồng | S11 |
| U18 | Khác biệt giữa các cơn bão (Yagi là một cơn) | S12 |
| U19 | Kết quả TN15 với đúng mô hình cũ (DM11) có lặp lại trên dữ liệu mới không | S0 |

**Không kiểm trong TN15b:**
- X = 30: thầy TN13 cho thấy gần như trùng x10 (26,62% / 19,4 ms so với 26,65% / 18,6 ms); bỏ để tiết kiệm khoảng 1 giờ.
- Mở rộng kho: để TN sau.

## Dữ liệu — bộ kiểm chứng mới

Thời tiết 2017–2021, **không trùng ngày nào với bất kỳ tập nào đã dùng** (học, val, kiểm tra của TN4–TN15). Luật chọn nằm
trong `tao_du_lieu_dt.cua_so_15b`, tất định:

| Tập | Cửa sổ | Luật |
|---|---:|---|
| `bao15b` | 20 | Đợt bão 2017–21 xếp hạng 14–33 theo cường độ (bao9 đã lấy hạng 1–13); bỏ đợt có giờ mạnh nhất cách đợt đã có ≤ 1 ngày |
| `ngay15b` | 20 | Ngày xấu 2017–21 (luật của ngay_xau), ngoài 13 ngày của nam9 và ngoài mọi đợt bão của bao9 / bao15b (±1 ngày); bốc theo hoán vị default_rng(15026), mỗi ngày cách các ngày đã lấy > 1 ngày |
| `k15b` | 12 | 6 đợt đầu của bao15b + 6 ngày đầu của ngay15b (theo thời gian), 10 luồng |

- Mỗi cửa sổ là **một cụm độc lập** (khác ngày, cách nhau ≥ 2 ngày). Mỗi cửa sổ có 48 mốc × 3 mức tải liên tục = 144 tình
  huống, như bộ kiểm tra TN11. Tổng 40 × 144 = 5 760 tình huống ở 4–5 luồng.
- Ba lần sinh:
  - **A** (4–5 luồng, kho cũ): thầy 7/3 (mặc định) và thầy x10 (`--muc-tieu x10`) trên cùng 32 thế giới nhãn, lưới Dijkstra
    phạt.
  - **B** (4–5 luồng, kho nhanh, không thầy): cho cấu hình đề xuất. Nhóm rủi ro dựng lại như TN14.
  - **K10** (10 luồng, không thầy).

## Học lại (máy GPU của nhóm — cùng máy đã học DM x10 ở TN15)

| Mô hình | Seed | Nhãn | Ghi chú |
|---|---|---|---|
| DM x10 | 3, 4 (mới) + 0–2 (TN15) | `tn13/x10` (TN15 đã gán) | thêm seed → 5 seed |
| **DM 7/3 cùng đường ống** (`dm13_w0.7`) | 0–4 | gán nhãn lại `tn13/w0.7` | đúng đường ống của x10: `--muc-tieu w0.7 --chi-nhan`, THI_NGHIEM 18, checkpoint theo J 7/3 trên val giai đoạn A của TN13 |
| DM x3 (`dm13_x3`) | 0–2 | gán nhãn lại `tn13/x3` | quét X |

- Cấu hình học như DM11 / TN15: H 128, 120 epoch, dừng sớm 20.
- Gán nhãn lại kèm kiểm tái lập so với tập gốc, như TN15 (`kiem_tai_lap.json`).
- Mô hình có sẵn dùng thêm: DM11 s0–2 (TN11, máy khác); DM w0,5 / w0,9 s0–2 (TN13, cùng máy).

## Giải mã

- **đầy đủ:** 5 bước × 32 chuỗi, tái nhiễu (`chon_dm`), hạt giống = `giong_ban_ghi`.
- **ktn:** không tái nhiễu, 5 × 32 (`chon_sua_dan`).
- **nhanh:** ktn 5 × 8.
- **A/A:** DM x10 giải mã đầy đủ lần hai với hạt giống `giong_ban_ghi + 7919`. Cùng mô hình, nên hiệu thật bằng 0.
- **Đề xuất:** DM x10 ktn trên kho nhanh, rồi vá hết tuyến (ρ = 1, bảng 64 thế giới, 2 vòng, như TN14).
- **Chọn seed:** trong 5 lời giải DM x10 ktn, lấy lời giải có J x10 nhỏ nhất ước lượng trên 32 thế giới bốc từ bảng rủi ro. Chỉ
  dùng thông tin lúc quyết định.
- **Thầy:** chấm lời giải của thầy bằng cùng thế giới chấm với DM.
- **Dijkstra phạt:** dp_50_50, bộ β, γ TN13 chọn cho x10 trên val.

## Thước đo

- **mất** = mất dịch vụ (%), kèm (a) và (b).
- **trễ** = trễ tuyến chính danh định (ms).
- **đuôi** = % tình huống có trễ > 100 ms, tính **theo từng seed rồi lấy trung bình** (sửa theo DANH_GIA_TN15 mục 3.4).
  - Đuôi là thước đo trễ **chính**. Lý do: theo phụ lục A4 của DANH_GIA_TN15, con số ms ở đuôi do cách nối dài hàm hàng đợi
    quyết định. Thêm nữa, kể cả với 5 seed, trễ trung bình vẫn chỉ đủ sức kiểm khoảng 55% (xem Cỡ mẫu).
  - Trễ trung bình là thước đo phụ.
- **trễ thực** (thăm dò, S10): trễ của tuyến **đang dùng** trong từng thế giới chấm, tính trên các luồng còn phục vụ.
  - Tuyến đang dùng là tuyến sống đầu tiên của tập, tức có chuyển sang dự phòng.
  - Dung lượng còn trống là mức tệ nhất trong cửa sổ; tải nền lấy như lúc quyết định.
  - Luồng bị nghẽn đã được tính vào (b), nên % > 100 ms gần như bằng 0. S10 đọc ở trễ thực trung bình.

## Phân tích (chốt trước)

- **Đơn vị bốc lại = cửa sổ thời tiết** (40 cụm độc lập; K10: 12).
- **CI hai tầng**, 4 000 lần bốc: bốc lại cửa sổ có hoàn lại, và bốc lại seed có hoàn lại cho mỗi mô hình. Phép so trong cùng
  mô hình (ktn − đầy đủ, A/A, vá − không vá) bốc cặp seed chung.
- **Ước lượng điểm** = trung bình các seed, mọi tình huống nặng như nhau.
- **p một phía** = Φ((ước lượng − ngưỡng) / sai số chuẩn bootstrap). Giả thuyết có hai điều kiện thì p = max của hai p.
- **Họ chính P1–P8: hiệu chỉnh Holm, α = 0,05**. Đạt = p sau Holm < 0,05.
- **"Không kém"** = ngưỡng +0,10 điểm mất dịch vụ.

## Giả thuyết chính (họ Holm)

| Mã | Giả thuyết | Dữ liệu | Phép thử |
|---|---|---|---|
| P1 | x10 không kém an toàn so với 7/3 cùng đường ống (đầy đủ, 5 + 5 seed) | A | mất x10 − w0.7 < +0,10 |
| P2 | x10 bớt đuôi nghẽn so với 7/3 cùng đường ống | A | đuôi x10 − w0.7 < 0 |
| P3 | ktn tốt hơn đầy đủ ở DM 7/3 cùng đường ống (5 seed) | A | mất w0.7 ktn − w0.7 < 0 |
| P4 | ktn tốt hơn đầy đủ ở DM x10 (5 seed) | A | mất x10 ktn − x10 < 0 |
| P5 | Cơ chế: tỷ giá ngầm càng cao, x10 càng kém an toàn hơn 7/3 | A | hệ số log X_eff > 0 trong hồi quy hiệu mất (x10 − w0.7, theo tình huống, trung bình seed) ~ 1 + log X_eff + log trễ Dijkstra |
| P6 | x10 an toàn hơn Dijkstra phạt | A | mất x10 − dp_50_50 < 0 |
| P7 | Thầy x10 so với thầy 7/3: không kém an toàn **và** bớt đuôi | A | mất < +0,10 **và** đuôi < 0 (bốc lại cửa sổ) |
| P8 | Cấu hình đề xuất tốt hơn cấu hình TN11 (DM11) | B | mất (x10 ktn + vá) − DM11 (đầy đủ) < 0 **và** đuôi < 0 |

**P0 — phép thử A/A (kiểm độ tin của CI, không thuộc họ Holm):** x10 (lần 2) − x10 về mất và đuôi. **Phải không có ý nghĩa**:
CI 95% hai phía chứa 0. Nếu có ý nghĩa, CI của cả TN15b được coi là quá hẹp và mọi kết luận phải đọc lại.

## Chỉ số phụ (báo CI, không hiệu chỉnh)

| Mã | Chỉ số |
|---|---|
| S0 | Lặp lại TN15 với mô hình cũ: x10 − DM11 (mất < +0,10; đuôi < 0) |
| S1 | Trễ trung bình và trung vị x10 − w0.7; đuôi ở ngưỡng 50 / 200 ms |
| S2 | mất x10 − w0.7 < 0 ("an toàn hơn") |
| S3 | Hiệu mất (x10 − w0.7) theo nhóm X_eff (< 3, 3–10, 10–30, > 30); hệ số log X_eff của hiệu trễ |
| S4 | Phần đóng góp vào hiệu (b) từ các tình huống w0.7 có đuôi; tương quan hạng giữa hiệu (b) và hiệu trễ |
| S5 | Đường biên: x10 so với DM w0,5, w0.7, w0,9 (mất, đuôi, trễ) |
| S6 | DM x10 − thầy x10 (trễ, đuôi, mất); thầy x10 − Dijkstra phạt (trễ). Luật đọc: CI của (DM x10 − thầy x10) về trễ trung bình nằm trong ±0,5 ms → khoảng còn lại với Dijkstra là do mục tiêu |
| S7 | Vá trên x10 ktn (B): mất (vá) − (không vá) < 0; đề xuất − w0.7 (B) |
| S8 | Chọn seed − trung bình 5 seed (x10 ktn), và khoảng cách tới cận trên biết trước |
| S9 | x10 ktn 5 × 8 − x10 ktn 5 × 32: mất < +0,10 |
| S10 | Trễ thực (trung bình, % > 100 ms) x10 − w0.7 |
| S11 | K10: mất x10 − w0.7 < +0,10 và đuôi < 0; x10 ktn − x10; x10 − DM11; x10 − Dijkstra |
| S12 | Độ lệch chuẩn giữa các cửa sổ của hiệu x10 − w0.7, bão so với ngày; so với ước lượng từ TN15 |
| S13 | x10 − Dijkstra phạt tốt nhất trên lưới chọn **sau** khi xem (đối chứng khó nhất) |
| S14 | Nhiễu do máy / code: w0.7 (cùng đường ống) − DM11 (máy TN11). Đọc: CI trong ±0,10 điểm (mất) và ±0,5 điểm (đuôi) → không có nhiễu đường ống đáng kể |
| S15 | Quét X: x3 − x10 (mất, đuôi, trễ) và x3 − Dijkstra (trễ). Luật đọc: mất x3 − x10 < +0,10 và đuôi < 0 → X nhỏ hơn tốt hơn; x3 − Dijkstra trễ có CI chứa 0 → khoảng trễ với Dijkstra đóng được |

## Cỡ mẫu

- **Nguồn ước lượng:** các cửa sổ 2017–21 của TN15 (nam9, bao9; 144 tình huống mỗi cửa sổ).
  - Độ lệch chuẩn giữa cửa sổ của hiệu: mất 0,22; đuôi 0,82 điểm %; trễ 2,8 ms.
  - Độ lệch chuẩn giữa seed (giả định 7/3 cùng đường ống như DM11): mất 0,088 / 0,043; đuôi 0,19 / 0,06; trễ 0,91 / 0,33 ms
    (7/3 / x10).
- **Với 40 cửa sổ, 5 + 5 seed**, α Holm xấu nhất 0,05 / 8, một phía:
  - P1: z ≈ 5 (độ mạnh gần 100%);
  - P2: z ≈ 3,9 (khoảng 88%, giả định hiệu đuôi −0,6 điểm như nam9 / bao9);
  - P3–P4: độ mạnh cao (sai số do seed gần như triệt tiêu trong phép so cùng mô hình);
  - trễ trung bình: z ≈ 2,6 (khoảng 55%) → để làm chỉ số phụ.

## Đọc tổng (chốt trước)

- **P0 không có ý nghĩa và P1, P2, P6 đạt:** kết luận chính của TN15 đứng trên dữ liệu mới, độc lập, so với 7/3 học cùng máy
  và cùng code.
- **P1 hoặc P2 trượt:** kết luận chính của TN15 **không** lặp lại được. Báo đúng như vậy, không chạy lại với thiết lập khác.
- **P3 và P4 đạt:** ktn làm mặc định có cơ sở độc lập.
- **P5 đạt:** cơ chế tỷ giá ngầm được xác nhận như một dự đoán viết trước.
- **P7 đạt:** lợi thế của mục tiêu x10 ở mức thầy được xác nhận trên dữ liệu không dùng để chọn x10 (thay cho H15.0).
- **P8 đạt:** cấu hình đề xuất được xác nhận như một khối.
- **S14 trong ngưỡng:** hơn kém DM11 – x10 ở TN15 không phải do máy / code.

## Quy tắc khác

- **Đóng băng:** bước đầu của `toi-da` ghi mã băm SHA-256 của tệp này, `scripts/tn15b.py` và `src/nhieu_luong/tao_du_lieu_dt.py`
  vào `tn15b/dong_bang.json`. Bước phân tích kiểm lại mã băm và ghi cảnh báo nếu khác.
- **Mô hình học lỗi:** chạy lại đúng lệnh (mô hình xong được giữ). Không đổi cấu hình học.
- **Cửa sổ sinh lỗi** bị bỏ và ghi vào báo cáo. Bỏ quá 10% số cửa sổ thì ghi CẢNH BÁO.
- **Mọi kết quả được báo**, kể cả khi trượt.

## Chạy thử trước khi đóng băng (23/11/2026, dữ liệu cũ nam9, 117 tình huống; không chạm dữ liệu mới)

- Sửa 2 lỗi code, không đổi thiết kế:
  1. p bị đảo chiều khi sai số chuẩn bootstrap bằng 0;
  2. định dạng hai dòng báo cáo.
- Ghi nhận về trễ thực (S10): luồng bị nghẽn đã được tính vào (b), nên % trễ thực > 100 ms gần như bằng 0.
- Bước gán nhãn và học không chạy thử trên máy người dùng (không học trên máy này). Lệnh học giống hệt TN15 (`lenh_hoc`
  THI_NGHIEM 18 `--muc-tieu`), chỉ khác tên mục tiêu (w0.7, x3; đã kiểm tên khớp khoá của val giai đoạn A).

## Chạy

`python -m scripts.tn15b toi-da` trên máy GPU của nhóm. Tài nguyên tự chọn như TN15; ghi đè được bằng `--tien-trinh`,
`--tien-trinh-sinh`, `--song-song`.
- **Ước tính 6–7 giờ:** gán nhãn khoảng 2 giờ, học 10 mô hình khoảng 1,2 giờ, sinh dữ liệu mới khoảng 0,7 giờ, chấm khoảng
  2,5 giờ. Nên chạy qua đêm.
- Kết quả ở `results/bao_da_tuyen/tn15b/`, gói `TN15B_ket_qua.zip`. Hướng dẫn: `tai_lieu/HUONG_DAN_TN15B.md`.
