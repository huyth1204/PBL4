# Schema dữ liệu SAGSIN

Tài liệu này mô tả các tệp `.npz` do `dataset/oracle_labeler_sagsin.py` sinh ra.
Có ba bản: **v2** (73 nút, mốc hồi quy — KHÔNG bao giờ ghi đè) và **v3** (mạng
mở rộng, thêm nút đảo/núi/vùng lũ + trạng thái hạ cánh UAV).

## 1. Bản v2 — mốc hồi quy 73 nút

`dataset/oracle_dataset_v2.npz`, sinh bằng:

```bash
python -m dataset.oracle_labeler_sagsin --out dataset/oracle_dataset_v2.npz
```

`test/test_hoi_quy_73.py` khoá bản này lại: nếu `node_order`, `reach_stats`,
số cạnh mỗi snapshot hoặc W lệch quá 1e-4 thì test đỏ. Bản v2 là hợp đồng với
model đã huấn luyện của Nhật và các hình đã chốt của Trung.

| Khoá | Kích thước | Ý nghĩa |
|---|---|---|
| `W_snapshots` | `[S, N, N]` float32 | Ma trận trễ (ms); `-1` = không có liên kết |
| `x_sources`, `x_targets` | `[M]` | Chỉ số nút nguồn / đích của từng mẫu |
| `y_paths_indices` | `[M, L]` int32 | Tuyến đúng, `-1` = hết (padding) |
| `y_paths_text` | `[M]` str | Tuyến đúng dạng tên nút |
| `snapshot_id` | `[M]` int32 | Mẫu thuộc snapshot nào (tra vào `W_snapshots`) |
| `node_order` | `[N]` str | Tên nút, thứ tự = chỉ số nút |
| `node_layer` | `[N]` str | `space` / `air` / `ground` |
| `node_kind` | `[N]` str | `leo` / `haps` / `uav` / `gateway` / `user` |
| `reach_stats` | `[S, 3]` int32 | `snapshot_id, n_pairs, n_reachable` |
| `infra` | `[k]` str | Tầng bật thêm (ablation): `space`, `haps`, `uav` |
| `seed` | `[1]` | Seed |

## 2. Bản v3 — mạng mở rộng

```bash
python -m dataset.oracle_labeler_sagsin --mo-rong --out dataset/oracle_dataset_v3.npz \
    --tle data/tle/starlink.txt
```

Giữ **nguyên** mọi khoá của v2 và **thêm**:

| Khoá | Kích thước | Ý nghĩa |
|---|---|---|
| `node_latlon_alt` | `[S, N, 3]` float32 | `(lat°, lon°, alt_km)` từng nút từng snapshot. Vệ tinh lấy subpoint skyfield; Air/Ground lấy từ cấu hình (UAV di động nên phải theo snapshot) |
| `link_stats` | `[E, 8]` float64 | Bảng liên kết (xem dưới) |
| `link_stat_cols` | `[8]` str | Tên 8 cột của `link_stats` |
| `mo_rong` | `[1]` bool | True nếu sinh ở chế độ mở rộng |

### Cột `link_stats` (`link_stat_cols`)

| Cột | Ý nghĩa |
|---|---|
| `snapshot_id` | Snapshot chứa liên kết |
| `u`, `v` | Chỉ số hai đầu (tra vào `node_order`) |
| `distance_km` | GSL: khoảng cách nghiêng; ISL/local: khoảng cách thẳng |
| `elevation_deg` | Góc ngẩng (GSL, local); `nan` cho ISL (chưa mô hình hoá) |
| `bandwidth_mbps` | Băng thông theo loại liên kết |
| `packet_loss` | Tỉ lệ mất gói theo loại liên kết |
| `exists` | Luôn `1` — bảng chỉ ghi liên kết đang tồn tại |

### Bộ train/val/test cho v3 — `dataset/v3/`

```bash
python dataset/split_dataset.py --in dataset/oracle_dataset_v3.npz --outdir dataset/v3 -v
```

> ⚠️ **Phải có `--outdir dataset/v3`.** `split_dataset.py` luôn ghi tên cố định
> `train.npz`, `val.npz`, `test.npz`, và mặc định ghi vào thư mục của file vào.
> Bỏ `--outdir` mà trỏ `--in` vào v3 sẽ **ghi đè bộ của v2** ở `dataset/`.

Kết quả (600 snapshot, `--mode phase`, 12 khối × 12 snapshot, chừa 3 biên):

| Tập | Snapshot | Mẫu | Hop TB | Nguồn là trạm |
|---|---|---|---|---|
| train | 357 | 17 850 | 2.89 | 26.8% |
| val | 48 | 2 400 | 2.86 | 27.4% |
| test | 48 | 2 400 | 2.92 | 26.5% |

Bỏ 147 snapshot ở biên. **Giao giữa ba tập = 0 snapshot** (kiểm trong
`dataset/v3/split_info.json`). Test giống train: trung vị Jaccard 0.70 (chia ngẫu
nhiên sẽ là 0.97).

`link_stats` **được chia theo tập** và cột `snapshot_id` đánh lại **0..S_tập−1**
cho khớp `W_snapshots`, `node_latlon_alt` và `snapshot_id` của mẫu:

| Tập | Snapshot | `link_stats` | Cột 0 | Cạnh trong `W_snapshots` |
|---|---|---|---|---|
| train | 321 | 146 974 | 0–320 | 146 974 ✓ |
| val | 48 | 20 848 | 0–47 | 20 848 ✓ |
| test | 54 | 23 496 | 0–53 | 23 496 ✓ |
| tổng | 423 | 191 318 | — | — |

Bảng gốc 267 458 − 191 318 = **76 140 dòng của 177 snapshot bị bỏ ở biên**.
Tra `link_stats[link_stats[:,0] == k]` bằng index cục bộ là đúng; dùng chỉ số gốc
0–599 sẽ lấy sai. Test `test_mang_mo_rong.py::test_v3_split_khong_ro_ri_link_stats`
khoá hợp đồng này (số dòng = số liên kết dương trong `W_snapshots` của chính tập).


Nếu `split_dataset.py` báo `Không biết cách chia các key sau (shape): ...` thì đó
là key metadata mới do labeler thêm — khai báo tên nó vào `SHARED_KEYS` (hoặc
`PER_SNAPSHOT_KEYS` / `PER_LINK_KEYS`) trong `split_dataset.py`. Script cố tình từ
chối đoán để không làm mất hay cắt sai dữ liệu. (`mo_rong` đã được khai báo.)


Chỉ liên kết `exists=True` được ghi; cặp không tồn tại đã thể hiện bằng `W = -1`.
Mỗi liên kết vô hướng được ghi **hai hàng** (`u→v` và `v→u`) để tra cứu nhanh.

## 3. Thứ tự nút — điều quan trọng nhất khi đọc dữ liệu

`node_order = [vệ tinh đã sort] + [air đã sort] + [ground đã sort]`.

Khóa sort là `khoa_sap_xep` (`src/physics/sites.py`):

```
(0, tên)  cho nút thuộc cấu hình CŨ (13 ground + 5 HAPS + 5 UAV tâm)
(1, tên)  cho nút MỚI (đảo / núi / vùng lũ / HAPS mới / UAV đặt cố định)
```

Nhờ khóa này:

* Cấu hình cũ (73 nút) cho kết quả **y hệt** `sorted(tên)` → mạng 73 nút không đổi.
* Bật `--mo-rong`, nút mới **nối đuôi toàn bộ `node_order`** → chỉ số nút cũ **không đổi**.
  Ví dụ tầng ground: 13 ground cũ (đúng thứ tự cũ) rồi mới tới `BachLongVi`,
  `ConDao`, `DaLat`, `HoangSa`, `KonTum`, `LySon`, `PhuQuoc`, `QuangBinh`,
  `QuangTri`, `SaPa`, `TruongSa`.

> **Bẫy đã từng dính (đừng tái phạm).** Nếu chỉ nối nút mới vào cuối *từng tầng*
> (`sat_names + air_names + ground_names`) thì tầng air phình 10 → 14 và **toàn bộ
> tầng ground bị đẩy lệch 4 chỉ số**. Khi đó 13 nút ground cũ vẫn đúng thứ tự
> *tương đối* nên rất dễ tưởng là ổn, nhưng `oracle_dataset_v3` không còn dùng
> chung chỉ số với v2 nữa. `test/test_mang_mo_rong.py` chặn đúng lỗi này: nó so
> chỉ số tuyệt đối của từng nút cũ giữa v2 và v3, và so cả ma trận `W`.
> Cách dựng đúng: xếp nút cũ (space → air → ground) trước, rồi nối toàn bộ nút mới.

Nút đảo xa (Hoàng Sa, Trường Sa…) **chỉ nối được qua vệ tinh**: `LOCAL_RULES`
trong `air_ground.py` không có luật ground↔ground, nên link mặt đất giữa hai
đảo không tồn tại. Đây là hành vi mong muốn, không phải lỗi.

## 4. Thời tiết theo nút (`src/ngau_nhien/thoi_tiet.py`)

```bash
# Tệp chuẩn 23 điểm (dùng cho mạng 73 nút)
python -m src.ngau_nhien.thoi_tiet tai --start 2024-01-01 --end 2025-12-31 \
    --out data/thoi_tiet_vn.npz

# Tệp mở rộng 41 điểm (dùng cho mạng 94 nút) — tên tệp KHÁC, không ghi đè tệp trên
python -m src.ngau_nhien.thoi_tiet tai --mo-rong --start 2024-01-01 --end 2025-12-31 \
    --out data/thoi_tiet_vn_mr.npz
```

> ⚠️ **Khoảng thời gian thật của cả hai tệp là 2024-01-01 → 2025-12-31**
> (17 544 giờ). Đừng chép lệnh cũ `--start 2023-01-01 --end 2024-12-31` ở bản
> tài liệu trước — nó vừa sai thời gian, vừa ghi vào `thoi_tiet_vn.npz` nên sẽ
> **ghi đè tệp 23 điểm**. Luôn dùng `--out data/thoi_tiet_vn_mr.npz` cho bản
> mở rộng.

Tệp thời tiết chứa 23 điểm (mặc định) hoặc 41 điểm (`--mo-rong`, thêm 14 ground
+ 2 HAPS + 5 UAV đặt cố định). `kiem_tra_du_diem()` ném `ValueError` nếu cấu
hình có điểm mà tệp thời tiết thiếu — chặn trường hợp nút mới lặng lẽ không có
mưa/gió. Nguồn cấu hình duy nhất là `src/physics/sites.py`.

## 5. Trạng thái UAV: `tat` và `ha_canh`

> **Cảnh báo về nút cô lập (đo trên v3, 600 snapshot).** Một số nút không có
> liên kết nào trong phần lớn thời gian: `LySon` 63%, `DaLat` 62%, `BachLongVi`
> 61%, `ConDao`/`PhuQuoc` 60%, `SaPa` 29%. **Nhưng đây không phải vấn đề của
> nút mới** — hai nút CŨ của mạng 73 là `BuonMaThuot` 63% và `CaMau` 60% cũng
> y như vậy. Nguyên nhân là lỗ hổng phủ của cấu hình vệ tinh, đã tồn tại từ v2.
> Hệ quả: thêm HAPS/UAV riêng cho Phú Quốc sẽ chữa sai chỗ; muốn giảm thì phải
> sửa cấu hình vệ tinh và chấp nhận đổi cả v2.

`sinh_kich_ban()` (trong `src/ngau_nhien/kich_ban.py`) trả `quan_sat` có hai cờ
liên quan tới UAV:

| Cờ | Ý nghĩa | Dùng làm gì |
|---|---|---|
| `tat` | Nút không dùng được → **cắt mọi cạnh** | Tính W (giữ nguyên như cũ) |
| `ha_canh` | UAV **hạ cánh an toàn**, vẫn tồn tại | Đặc trưng cho model |

Với UAV, cả hai cờ bật khi: có giông tại điểm UAV, **hoặc** gió giật vượt
`gio_giat_max` (mặc định 15 m/s). Tách `ha_canh` khỏi `tat` để model phân biệt
"UAV chờ gió lặng rồi cất cánh lại" với "nút hỏng hẳn".

> **Lưu ý về Yagi — đừng viết sai trong báo cáo.** Trong `data/thoi_tiet_vn.npz`,
> bão Yagi (6–10/9/2024) chỉ làm gió giật vượt 15 m/s tại **Hải Phòng (22/96 giờ)**
> và **Hà Nội (18/96 giờ)**; các điểm khác đều dưới ngưỡng. Nhưng 5 UAV cũ bay
> vòng quanh **Vinh, Nha Trang, Cần Thơ, Lào Cai, Quy Nhơn** — không có UAV nào
> ở Hải Phòng. Chạy luật trên dữ liệu thật, cả 5 UAV đều **không** vào `ha_canh`
> trong suốt bão (gió giật cực đại 10.9 m/s < 15). Nói cách khác: tính năng hạ
> cánh chạy đúng, nhưng **bão Yagi không kích hoạt nó trên mạng 73 nút**. Đừng
> viết "tầng UAV vào `ha_canh` suốt cơn bão" — số liệu không ủng hộ.
>
> Muốn thử `ha_canh` kích hoạt thật thì dùng **tệp thời tiết mở rộng**:
> `python -m src.ngau_nhien.thoi_tiet tai --mo-rong --start 2024-01-01 --end 2025-12-31 --out data/thoi_tiet_vn_mr.npz`
> rồi `them-giong --vao data/thoi_tiet_vn_mr.npz`. Tệp đó có đủ 41 điểm nên
> `node_site` của 18 nút mới khác -1. Không tải được thì dùng `gia-lap --mo-rong`.
>
> **Kết quả trên `thoi_tiet_vn_mr.npz` (đã chạy):** 23 điểm của mạng 73 nút **giống
> hệt từng bit** tệp cũ `thoi_tiet_vn.npz` (cùng 17 544 giờ ⇒ `test_hoi_quy_73.py`
> vẫn dùng được, không đổi kết quả). Điểm mở rộng nối vào cuối, KHÔNG xen vào 23
> cột đầu. Bão Yagi 1 ngày đầu (7/9/2024, T=72 snapshot):
>
> | UAV | Điểm thời tiết | Giật max (Yagi) | `ha_canh` |
> |---|---|---|---|
> | UAV-1..UAV-5 (cũ) | UAV-1..5 | 10.5–14.1 m/s | 0/72 giờ |
> | `UAV-HoangSa` | HoangSa | 17.5 m/s | 48/72 giờ |
> | `UAV-TruongSa` | TruongSa | 11.8 m/s | 0/72 giờ |
> | `UAV-HaiPhong` | UAV-HaiPhong | **41.8 m/s** | **18/72 giờ** |
> | `UAV-QuangYen` | UAV-QuangYen | **45.2 m/s** | **22/72 giờ** |
> | `UAV-VinhBacBo` | UAV-VinhBacBo | **41.4 m/s** | **33/72 giờ** |
>
> Tức là mạng mở rộng **có** kích hoạt `ha_canh`, cả ở hải đảo (Hoàng Sa) lẫn
> **đúng tâm bão** (Hải Phòng / Quảng Ninh), còn mạng 73 nút thì **không**.
>
> **Cụm ven biển Bắc (thêm sau khi có yêu cầu "chủ yếu ven biển Miền Bắc").**
> `EXTRA_COASTAL_SITES` thêm 3 trạm Quảng Ninh / Thái Bình / Nam Định, và
> `EXTRA_UAV_SITES` thêm 3 UAV đặt cố định quanh Hải Phòng. Lý do: trước đó toàn
> bộ nút mở rộng là hải đảo xa, nên **không UAV nào ở gần tâm bão Yagi** — đúng
> chỗ mà `ha_canh` cần để có ý nghĩa.
>
> | UAV | Vị trí | Phủ trong 25 km |
> |---|---|---|
> | `UAV-HaiPhong` | 20.845, 106.688 | HaiPhong (0 km) |
> | `UAV-QuangYen` | 20.870, 106.890 | HaiPhong (21.3 km), QuangNinh (21.8 km) |
> | `UAV-VinhBacBo` | 20.300, 106.800 | ngoài khơi, tiếp cận từ biển |
>
> `BachLongVi` (đảo, 60 km từ bờ) vẫn **cô lập toàn bộ 600/600 snapshot**: UAV chỉ
> với 25 km, HAPS-North cách 213 km > 200 km. Đây là khoảng trống phủ sóng kế thừa
> từ v2, không phải hồi quy.
>
> **Bẫy đã sập khi đặt `UAV-QuangYen` (ghi lại để khỏi lặp):** lần đầu đặt ở
> 20.93/106.85 — cách Hải Phòng 19.4 km và Quảng Ninh **24.1 km**, đều dưới ngưỡng
> 25 km, nhưng Quảng Ninh vẫn cô lập 600/600. Nguyên nhân không phải khoảng cách mà
> là `_UAV_GROUND.min_elev_deg = 5.0`: góc nâng từ Quảng Ninh tới UAV chỉ **4.65°**,
> nên cạnh không được tạo và **cũng không có dòng nào trong `link_stats`** (bảng chỉ
> ghi cặp `exists == 1`). Khi kiểm một vị trí đặt UAV, phải kiểm **cả ba**: khoảng
> cách ≤ `max_range_km`, góc nâng ≥ `min_elev_deg`, và không bị `blocked_by_earth`.
> Test `test_uav_ven_bien_bac_phu_duoc_hai_phong_va_quang_ninh` khoá cả ba điều kiện.
>
> **Hai con số nên nói đúng trong báo cáo:**
> - "Gió giật > 15 m/s ở Hải Phòng trong *mọi* giờ mưa lớn" **chỉ đúng trong
>   cơn bão**: 18/20 giờ mưa lớn ngày 7/9/2024. Cả 2 năm 2024–2025 chỉ có
>   56/945 giờ như vậy ⇒ **phải viết kèm mốc thời gian bão**, không nói chung.
> - `UAV-HoangSa` hạ cánh 48/72 giờ trong bão nhưng đây là **hiện tượng theo
>   mùa**, không phải "24/7 trên biển": gió giật > 15 m/s chiếm 11.3% số giờ cả
>   năm (1978 giờ), tập trung tháng 11–1 (T11/2024: 205 giờ, T12/2024: 417 giờ),
>   còn tháng 4–8 gần như bằng 0. Đừng viết "UAV biển gần như không bao giờ bay".


Tắt tách trạng thái: `ThamSo(uav_ha_canh=False)`.

## 6. Hai tệp dataset: v2 (mốc 73 nút) và v3 (mạng mở rộng 94 nút)

```bash
# Mốc hồi quy — TUYỆT ĐỐI không ghi đè
python dataset/oracle_labeler_sagsin.py --out dataset/oracle_dataset_v2.npz

# Mạng mở rộng (94 nút). Labeler từ chối ghi --mo-rong vào v2.
python dataset/oracle_labeler_sagsin.py --mo-rong --out dataset/oracle_dataset_v3.npz
```

| | v2 | v3 |
|---|---|---|
| Số nút | 73 | 94 |
| Hình dạng `W_snapshots` | (600, 73, 73) | (600, 94, 94) |
| `node_latlon_alt` | ✗ | ✓ (600, 94, 3) |
| `link_stats` + `link_stat_cols` | ✗ | ✓ (267458, 8) |
| `mo_rong` | ✗ | ✓ `[True]` |

94 = 50 vệ tinh + 24 ground + 7 HAPS + 13 UAV (5 bay vòng + 8 đặt cố định).

**Hai tệp dùng chung chỉ số cho 73 nút đầu**: `v3.node_order[:73] == v2.node_order`.
`W_snapshots` của v3 hạn chế về 73 nút đó **trùng khớp tới sai số float32** — KHÔNG
giống tuyệt đối từng bit. Số liệu thật: 14 304 / 3 197 400 ô lệch, chênh lệch lớn
nhất **2.48e-05 ms**, không có ô nào lệch quá 1e-4. Nguyên nhân: ở v3 ma trận
94×94 nên phép chuẩn hoá nội bộ float32 ra chữ số cuối khác so với 73×73. Đây là
làm tròn, không phải lỗi logic — cạnh "không liên kết" (`W = -1`) khớp 100%.
Vì vậy test dùng `np.allclose`, **không** dùng `array_equal`. Đừng viết trong báo
cáo là "W giống hệt từng bit"; viết "trùng khớp tới sai số float32 (≤ 2.5e-5 ms)".
Nhờ vậy model huấn luyện trên v2 vẫn chạy được trên v3 (chỉ cần bỏ 21 nút cuối),
và ngược lại có thể so sánh trực tiếp hai mạng.

`link_stats` — một hàng cho mỗi liên kết mỗi snapshot, ghi **hai chiều** (`u→v`, `v→u`):

`snapshot_id, u, v, distance_km, elevation_deg, bandwidth_mbps, packet_loss, exists`

`u`/`v` là chỉ số trong `node_order`. `elevation_deg` chỉ có nghĩa với liên kết
GSL/local (vệ tinh–mặt đất, air–ground).

