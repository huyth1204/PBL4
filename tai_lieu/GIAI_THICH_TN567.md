# Giải thích thí nghiệm 5, 6, 7

Tài liệu này giải thích **bằng tiếng Việt dễ hiểu** ba thí nghiệm, để nắm và trình bày.

---

## 0. Nền tảng: bài toán là gì

Ta có một mạng viễn thông đa tầng (vệ tinh LEO – HAPS – UAV – trạm mặt đất) phủ Việt Nam,
chịu ảnh hưởng của **bão, mưa lớn, giông sét**. Khi thời tiết xấu, một số liên kết mất
hoặc bị nghẽn.

**Nhiệm vụ**: với mỗi luồng dữ liệu (một cặp nguồn → đích), chọn **một tuyến chính** và
**một tuyến dự phòng**, sao cho khi liên kết hỏng thì luồng vẫn còn đường đi.

### Các ký hiệu cần nhớ

| Ký hiệu | Nghĩa | Ví dụ |
|---|---|---|
| **K** | số luồng cần định tuyến cùng lúc | K = 4, 10, 20 |
| **ρ̂** (rho_hat) | **hệ số tải** ước lượng lúc ra quyết định | ρ̂ = 0,8 → mạng dùng 80% dung lượng |
| **J** | điểm tổng hợp, **càng thấp càng tốt** | trộn 0,7·(mất dịch vụ) + 0,3·(trễ) |
| **mất dịch vụ %** | % lưu lượng không truyền được | 12% = mất 12% dữ liệu |
| **ke (a)** | % lưu lượng mà **mọi tuyến đều đứt** | không đường nào sống |
| **ke (b)** | % lưu lượng mà tuyến **đang dùng bị quá tải** | còn tuyến khác nhưng không kịp chuyển |
| **τ** (tau) | **ngưỡng quyết định** của chính sách Lai | τ = 1,1 |
| **TKCB** | **Tìm kiếm cục bộ trên bảng** — cải thiện lời giải bằng thử hoán vị | mất ít thời gian hơn DM |

**Chỉ số mất dịch vụ = (a) + (b)** — hai nguyên nhân khác nhau, cần giảm cả hai.

---

## 1. Ba phương pháp định tuyến

| Phương pháp | Làm gì | Thời gian |
|---|---|---|
| **Dijkstra phạt** (`dphat`) | **Đối thủ chính**. Dijkstra nhưng cộng thêm β·(−ln(1−rủi ro)) vào trọng số | Nhanh |
| **DM** (Diffusion Model) | Mô hình học máy sinh tuyến từ phân phối đã học | Chậm (mạng nơ-ron) |
| **DM + TKCB** | DM rồi **tinh chỉnh** bằng tìm kiếm cục bộ | Chậm nhất |
| **Lai(τ)** | ρ̂ < τ → dùng **DM**; ρ̂ ≥ τ → dùng **DM + TKCB** | Gần DM ở tải thấp |

**Ý tưởng của Lai**: tải thấp thì DM đủ tốt (không cần tinh chỉnh); tải cao thì DM dễ hỏng nên cần TKCB cứu.

---

## 2. Thí nghiệm 5 — "Có nên bật TKCB khi tải cao?"

### Câu hỏi
> "Chính sách lai DM/DM+TKCB theo tải có đạt chất lượng gần DM+TKCB khi tải cao,
> mà thời gian gần DM khi tải thấp không?"

### Cách làm
- **Bộ B — mưa lớn** (`mua5`): 13 cửa sổ, mỗi cửa sổ lấy mốc thời gian và 3 mức tải ngẫu nhiên.
- **Bộ A — Yagi** (`yagi5`): 13 cửa sổ bão Yagi.
- 3 quy mô K: 4–5, 10, 20 luồng.
- Chọn τ trên tập `val5` (16 cửa sổ), thử 9 giá trị τ: 0; 0,6; 0,7; 0,8; 0,9; 1,0; 1,1; 1,2; ∞.

### Tiêu chí đạt (C2, C4, C5)
| Mã | Điều kiện |
|---|---|
| **C2** | CI 95% của ΔJ (Lai − Dijkstra phạt) **< 0** → Lai tốt hơn |
| **C4** | CI 95% của Δ(mất dịch vụ) **< 0** → Lai ít mất hơn |
| **C5** | với K ≥ 10: thời gian Lai ≤ thời gian TKCB×10 **và** ≤ 1000 ms |

Đạt cả 3 → gọi là **"thắng"** (1 ô thắng).

### Hình có sẵn
`results/bao_da_tuyen/hinh_5.png` — 3 panel: (a) mất DV bộ B, (b) mất DV bộ A, (c) thời gian ở tải cao.

---

## 3. Thí nghiệm 6 — "Thêm dữ liệu giông có giúp DM?"

### Câu hỏi
> "Nếu tập train chỉ có **ngày bão**, DM có kém trên **ngày giông thường** không?
> Thêm 60 ngày giông vào train (DM6) có cải thiện không? Cải thiện do **dữ liệu** hay do
> **3 đặc trưng giông mới** quanh UAV?"

### Ba đặc trưng mới (bán kính 100 km quanh UAV)
1. **Có giông lân cận trong 1 giờ qua?**
2. **CAPE hiện tại** tại UAV (chỉ số đối lưu — giông sắp tới)
3. **Giông đang hoạt động quanh UAV?**

Lý do: UAV **hạ cánh** khi gió > 15 m/s hoặc có giông. Nếu mô hình **thấy trước** giông lân cận, nó chọn tuyến tránh UAV đó → không mất liên kết.

**"Điểm mù UAV"** (hình d4): trạng thái UAV chỉ đổi **ở ranh giới giờ**, theo thời tiết của **giờ TỚI**. Nên nếu chỉ nhìn thời tiết hiện tại, mô hình không biết UAV sẽ hạ cánh ngay sau đó.

### Phép ablate (quan trọng nhất)
| Mô hình | Huấn luyện trên | Đặc trưng giông | Ý nghĩa |
|---|---|---|---|
| **DM** | 59 cửa sổ bão | ✗ | DM cũ (TN4) |
| **DM6a** | 59 bão + 60 giông | ✗ | **Chỉ thêm dữ liệu** |
| **DM6** | 59 bão + 60 giông | ✓ | **Thêm dữ liệu + đặc trưng** |

So sánh: `DM6a − DM` = lợi ích do **dữ liệu**; `DM6 − DM6a` = lợi ích do **đặc trưng**.

### Ba bộ test
| Bộ | Nội dung | Vai trò |
|---|---|---|
| **C** `c6` | 13 cửa sổ **ngày giông**, mô hình **chưa từng thấy** | Test chính |
| **B** `mua5` | 13 cửa sổ mưa lớn (của TN5) | Test chính |
| **A** `yagi5` | 13 cửa sổ Yagi | Kiểm tra **thoái lui** (có tệ đi không?) |

### Kết quả đã chốt (từ `hinh_bao_cao_6_7.py` dòng 166)
```
Số ô thắng (Lai6 − Dijkstra phạt):  C = 6/9,  B = 2/9,  A = 8/9
```

Nghĩa là: **Lai6 thắng đối thủ 6/9 ô trên ngày giông, 8/9 ô trên Yagi, chỉ 2/9 trên mưa lớn**.

---

## 4. Thí nghiệm 7 — "Bền vững hay không, và thêm dữ liệu có đáng?"

### 7a — Thế giới cực đoan

Câu hỏi: Lai6 (τ cố định = 1,1) còn thắng khi điều kiện **xấu hơn thực tế** không?

Ba biến thể stress:
| Tên | Mưa | Hạ tầng |
|---|---|---|
| **W** | mưa ×1,5, gió ×1,25 | bình thường |
| **H** | bình thường | `gio_giat_max` 15→12 m/s, `p_set` 0,1→0,2, sét kéo dài 6 giờ |
| **WH** | cả hai | cả hai |

Kết quả chốt:
```
C cực đoan WH: Lai6 thắng 4/9 ô
A cực đoan WH: Lai6 thắng 7/9 ô
```
→ Vẫn thắng (ít hơn nhưng chưa sụp đổ).

### 7b — Đường cong dữ liệu

Câu hỏi: **Thêm dữ liệu có đáng không?** Huấn luyện DM với 25%, 50%, 100%, 200% dữ liệu.

| Mô hình | % dữ liệu | Ghi chú |
|---|---|---|
| `dm7_25` | 25% | lấy ngẫu nhiên |
| `dm7_50` | 50% | |
| `dm6` | 100% | mốc chuẩn |
| `dm7_200` | 200% | gấp đôi (train + train7s + train7x) |

Tiêu chí: `H7b_dat` = CI cận trên của `mất DV(dm7_200) − mất DV(dm6)` **< 0** (200% thực sự tốt hơn 100%).
Và `giam_don_dieu` = 25 > 50 > 100 > 200 (càng nhiều dữ liệu càng ít mất).

Nếu 200% **không** tốt hơn 100% → DM đã **bão hoà** → không cần thu thập thêm dữ liệu.

---

## 5. Dữ liệu đã dùng (số liệu thật từ `so_lieu_du_lieu.json`)

### Số cửa sổ mỗi tập
| Tập | Cửa sổ | Vai trò |
|---|---|---|
| `train` | 59 | Bão (train DM) |
| `train6` | 60 | Ngày giông (train DM6) |
| `train7x` | 60 | Ngày giông tiếp (train 200%) |
| `val` | 16 | Chọn checkpoint |
| `val6` | 16 | Chọn τ6 cho DM6 |
| `c6` | 13 | **Test C** — ngày giông |
| `mua5` | 13 | **Test B** — mưa lớn |
| `yagi5` | 13 | **Test A** — Yagi |

### Mức khắc nghiệt trung vị (%) — [giông, gió >15 m/s, mưa >11 mm/h]
| Tập | Giông | Gió mạnh | Mưa lớn |
|---|---|---|---|
| train (bão) | 4,39 | **18,92** | 2,36 |
| train6 (giông) | 8,11 | 0 | 0 |
| val | 1,69 | 14,86 | 2,53 |
| **c6** | **9,12** | 0 | 0,34 |
| **mua5** | 9,12 | 0 | **1,69** |
| **yagi5** | 2,70 | **37,16** | 3,38 |

**Đọc bảng này**:
- Tập bão train có nhiều **gió mạnh** (18,9%) nhưng ít giông (4,4%)
- Tập `c6` có nhiều **giông** (9,1%) nhưng **không có gió mạnh**
- Tập `yagi5` có **gió rất mạnh** (37,2%) — bão Yagi
- → Đó là lý do DM huấn luyện trên bão **kém** khi gặp ngày giông: nó chưa thấy mẫu giông.

### Ví dụ điểm mù UAV (từ `vi_du_uav`)
```
UAV-DoSon, ngày 2024-07-29
  số lần trạng thái đổi: 6
  giờ hạ cánh          : 3
  giờ giông tại UAV    : 3
  giờ giông lân cận    : 8   ← nhiều hơn giông tại chỗ
```
→ Tín hiệu **giông lân cận** (8 giờ) xuất hiện sớm hơn **giông tại UAV** (3 giờ).
Đó chính là đặc trưng mới mà DM6 dùng.

---

## 6. Hình đã vẽ được (không cần chạy thí nghiệm)

| File | Nội dung |
|---|---|
| `d1_ban_do_mang_va_giong.png` | Bản đồ SAGSIN v5 + tần suất giông 74 điểm |
| `d2_dong_thoi_gian_cac_tap.png` | Timeline 8 tập kịch bản (xanh train, vàng val, cam test) |
| `d3_muc_khac_nghiet_cac_tap.png` | 3 panel (giông / gió / mưa) × 8 tập, strip plot |
| `d4_vi_du_uav_ha_canh.png` | Timeline UAV-DoSon: gust + trạng thái, cho thấy "điểm mù" |
| `k1_ma_tran_o_tieu_chi.png` | Heatmap 9 ô tiêu chí × 5 bộ (C, B, A, C-WH, A-WH) |
| `k2_c4_khoang_tin_cay.png` | Forest plot CI 95% của C4 |

**Chưa vẽ được** (cần chạy thí nghiệm): `hinh_5.png`, `hinh_6.png`, `hinh_7.png`, `k3`, `k4`, `d5`.

---

## 7. Ba câu chốt để trả lời leader

1. **"Thí nghiệm 5 chọn τ = 1,1 vì ở tải thấp DM đã đủ tốt, chỉ khi ρ̂ ≥ 1,1 mới cần TKCB — giảm thời gian mà không mất chất lượng."**

2. **"Thí nghiệm 6 dùng phép ablate: DM6a (chỉ thêm dữ liệu) so DM (không thêm gì), DM6 (thêm dữ liệu + 3 đặc trưng giông quanh UAV) so DM6a. Chênh lệch thứ hai là đóng góp thật của đặc trưng."**

3. **"Thí nghiệm 7 kiểm tra hai thứ: 7a — Lai6 còn thắng khi stress gấp 1,5 lần thực tế; 7b — huấn luyện 25/50/100/200% để biết DM đã bão hoà chưa, có cần thêm dữ liệu không."**
