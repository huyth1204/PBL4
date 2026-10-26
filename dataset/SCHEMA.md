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
> Từ nay script **tự tạo** `--outdir` nếu chưa có và **từ chối** ghi đè khi
> `split_info.json` trong đó ghi một tệp vào khác (muốn thay thật thì xoá
> `split_info.json` rồi chạy lại).

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

# Tệp mở rộng 44 điểm (dùng cho mạng 101 nút) — tên tệp KHÁC, không ghi đè tệp trên
python -m src.ngau_nhien.thoi_tiet tai --mo-rong --start 2024-01-01 --end 2025-12-31 \
    --out data/thoi_tiet_vn_mr.npz

# BẮT BUỘC chạy tiếp bước này cho tệp vừa sinh:
python -m src.ngau_nhien.thoi_tiet them-giong --vao data/thoi_tiet_vn_mr.npz
```

> `--start` và (`--end` hoặc `--so-ngay`) là **bắt buộc** — không còn mặc định
> `2023-01-01 → 2024-12-31` (khoảng đó không có bão Yagi và dài 17 520 giờ thay vì
> 17 544). Ghi đè một tệp đã có bằng khoảng ngày khác bị chặn; muốn thay thật thì
> thêm `--cho-ghi-de`.

> ⚠️ **Quên `them-giong` là lỗi câm, không báo lỗi.** Lệnh `tai` chỉ lấy ERA5
> (mưa, gió giật, mã thời tiết) — **không có** `cape`/`lifted_index`/`code_fc`
> và cột `code` không có mã giông 95/97. Hậu quả: UAV không bao giờ hạ cánh vì
> giông, sét không bao giờ ngắt nút mặt đất, và mọi kết quả trên mạng lớn
> **không so sánh được** với kết quả cũ. Tệp đúng có **8 343 ô-giờ giông** trong
> 23 điểm cũ (2.24% số giờ theo luật CAPE).
> Test `test_tep_thoi_tiet_mo_rong_khop_tep_23_diem` chặn đúng lỗi này: nó đòi
> có `cape`/`lifted_index`, đòi 23 cột cũ giống **từng bit** (so theo *tên* điểm,
> vì trạm mới chèn giữa chứ không nối cuối), và đòi số ô-giờ giông > 0.

> ⚠️ **Khoảng thời gian thật của cả hai tệp là 2024-01-01 → 2025-12-31**
> (17 544 giờ). Đừng chép lệnh cũ `--start 2023-01-01 --end 2024-12-31` ở bản
> tài liệu trước — nó vừa sai thời gian, vừa ghi vào `thoi_tiet_vn.npz` nên sẽ
> **ghi đè tệp 23 điểm**. Luôn dùng `--out data/thoi_tiet_vn_mr.npz` cho bản
> mở rộng.

> ⚠️ **Độ cao vệ tinh: hai bẫy cùng lúc.**
> 1. `wgs84.subpoint_of(...).elevation.km` **luôn = 0**: đó là độ cao ellipsoid
>    tại điểm chiếu dưới chân vệ tinh, không phải độ cao quỹ đạo. Bản v3 cũ có
>    `alt = 0.0` cho cả 50 vệ tinh. Phải lấy `‖r‖ − R_earth` từ vector vị trí.
> 2. Tệp TLE công khai chứa cả vệ tinh **đã rơi**: propagate chúng cho ra độ cao
>    âm hàng nghìn km hoặc NaN. `load_tle()` lọc theo ngưỡng 150 km, nhưng phải
>    kiểm trên **toàn bộ khoảng thời gian dataset**, không phải một mốc: lọc ở
>    `now()` chỉ loại 140/10 728, và vệ tinh đang rơi (vd `STARLINK-4390`:
>    +308 km lúc epoch, **−207 km sau 30 ngày**, rồi +81 000 km…) vẫn lọt.
> Bản v3 hiện tại: độ cao vệ tinh **295–584 km**, trung vị 479 km, 0 âm, 0 NaN.
> Test `test_v3_do_cao_ve_tinh_hop_le` khoá cả hai bẫy.

Tệp thời tiết chứa 23 điểm (mặc định) hoặc **70 điểm** (`--mo-rong`: thêm 15
ground + 2 HAPS + 30 UAV đặt cố định vào cuối, cộng 5 tâm UAV tuần tra đã có từ
trước). `kiem_tra_du_diem()` kiểm **hai chiều** — ném `ValueError` nếu tệp *thiếu*
điểm cấu hình cần **hoặc** *thừa* điểm không còn trong cấu hình — và được gọi
ở **ba** nơi: `them_giong()`, `ThoiTiet(...)` (mặc định `kiem_tra=True`, tức mọi
lần đọc tệp) và **`MangGoc.__init__`** — chốt chặn chính, vì đúng chỗ đó mới gán
`node_site = -1`. Kiểm ở `MangGoc` nghĩa là mọi đường chạy (kể cả
`khoang_loi`, `luat_chia_doi`, `phan_tich_nhieu_mua`, `huong_dan/buoc_7`,
`buoc_8`, `tao_du_lieu_nl`) đều bị chặn, không phải nhớ bật cờ ở từng script —
trước đây chỉ `tao_du_lieu.main` bật nên đường nào quên là lỗi âm thầm quay lại.
Nguồn danh sách duy nhất là `src/physics/sites.py`
(`cac_diem_thoi_tiet()`); `thoi_tiet.cac_diem()`, phần kiểm tra và phần vẽ bản
đồ (`huong_dan/tien_ich.py`) đều đọc lại từ đó.

`MangGoc` tự suy `mo_rong` từ số điểm của `ThoiTiet` (`suy_mo_rong()`): tệp 23
điểm ⇒ mạng 73 nút (hẹp), nhiều hơn ⇒ mở rộng. Tham số `mo_rong=` chỉ cần
truyền khi muốn kiểm theo cấu hình khác với số điểm của tệp.

> **Bẫy đã sập một lần — đọc kỹ trước khi sửa danh sách nút.** Tệp
> `thoi_tiet_vn_mr.npz` từng có **44** điểm trong khi cấu hình cần **51**: còn sót
> cột `UAV-VinhBacBo` (đã xoá khỏi `sites.py`) và thiếu 8 UAV mới
> (`UAV-BachLongVi`, `UAV-LySon`, `UAV-BuonMaThuot`, `UAV-DaLat`, `UAV-CaMau`,
> `UAV-ConDao`, `UAV-PhuQuoc`, `UAV-SaPa`).
>
> Hệ quả **không hề báo lỗi**: `MangGoc` gán `node_site = -1`, rồi
> `sinh_kich_ban()` `continue` qua nút đó ⇒ 8 UAV **không bao giờ hạ cánh** vì
> giông hay gió, và `tao_du_lieu` để đặc trưng thời tiết (`R`, `gust`, `J`) bằng
> 0. `UAV-BachLongVi` — đảo giật 58.7 m/s trong Yagi, lý do chính để thêm nó —
> nằm trong số đó. Đo được: ở `h0=2590` (cửa sổ có giông tại Bạch Long Vĩ),
> `ha_canh` của UAV này là **540/600**; gán `node_site=-1` (đúng trạng thái lỗi)
> thì thành **0/600**.
>
> Vì sao không ai phát hiện: test cũ chỉ so **23 cột cũ**, mà 23 cột đó vẫn đúng
> từng bit. Nay có `test_tep_thoi_tiet_mo_rong_phu_dung_moi_nut` (so **đúng tập
> điểm**, cả thiếu lẫn thừa), `test_uav_dat_co_dinh_khop_thoi_tiet_voi_nut_chu`
> (UAV đặt cùng toạ độ nút thì mọi cột thời tiết phải trùng khít) và
> `test_mang_goc_chan_tep_thoi_tiet_lech` (khoá chính chốt chặn ở `MangGoc`: dựng
> bản tệp thiếu 8 điểm rồi đòi `MangGoc` phải ném lỗi kể cả khi nơi gọi không bật
> `kiem_tra`).

## 5. Trạng thái UAV: `tat` và `ha_canh`

> **Nút cô lập đã được xử lý (đo lại trên v3, 600 snapshot): không còn nút nào.**
> Bản v3 trước đây có 8 nút mất hết liên kết ~60% thời gian (`LySon` 63.2%,
> `BuonMaThuot` 62.8%, `DaLat` 62.0%, `BachLongVi` 60.8%, `CaMau` 60.2%,
> `ConDao` 59.8%, `PhuQuoc` 59.7%, `SaPa` 29.2%) vì chúng chỉ có đường vệ tinh.
> Nguyên nhân là lỗ hổng phủ vệ tinh có từ v2, **không phải** lỗi của nút mới.
>
> Cách sửa: đặt UAV ngay trên từng nút (`UAV-LySon`, `UAV-BuonMaThuot`,
> `UAV-DaLat`, `UAV-CaMau`, `UAV-ConDao`, `UAV-PhuQuoc`, `UAV-SaPa`,
> `UAV-BachLongVi`) — khoảng cách ~2.5 km, góc nâng 90°, biên +85° so với ngưỡng
> 5°. Không có ca nào sát ngưỡng nên không thể hỏng khi dịch toạ độ.
> Kiểm lại sau khi sinh: **0 nút** mất hết liên kết quá 10% thời gian.

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
> rồi `them-giong --vao data/thoi_tiet_vn_mr.npz`. Tệp đó có đủ **51 điểm** nên
> `node_site` của toàn bộ 51 nút không phải vệ tinh đều khác -1 (chỉ 50 vệ tinh
> không có cột thời tiết). Không tải được thì dùng `gia-lap --mo-rong`.
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
> | `UAV-BachLongVi` | BachLongVi | **58.7 m/s** | **72/72 giờ** |
>
> `BachLongVi` là nơi giật mạnh **nhất** trong cả mạng (58.7 m/s, mưa 246 mm,
> vượt ngưỡng 15 m/s **toàn bộ 24/24 giờ** ngày 7/9; 4 giờ trong đó có mã giông
> 95, nên `ha_canh` bật cả 24/24 giờ). Trước đây đảo này không có
> hạ tầng nào nên UAV `UAV-VinhBacBo` cũ đặt ngoài khơi 20.30/106.80 không phủ
> được nút nào (Thái Bình cách 51 km) — nó chỉ nối HAPS/vệ tinh, tức hạ cánh mà
> **không ảnh hưởng đường đi của bất kỳ nút nào**. Đã dời về đúng Bạch Long Vĩ.
>
> Tức là mạng mở rộng **có** kích hoạt `ha_canh`, cả ở hải đảo (Hoàng Sa) lẫn
> **đúng tâm bão** (Hải Phòng / Quảng Ninh / Bạch Long Vĩ), còn mạng 73 nút thì
> **không**.
>
> **Cụm VEN BIỂN BẮC BỘ + BẮC TRUNG BỘ — 18 UAV mới (yêu cầu: "chủ yếu ven biển
> miền Bắc để train với bão Yagi").** Trước bản này, mạng mở rộng chỉ có 3 UAV
> gần tâm bão (Hải Phòng, Quảng Yên, Bạch Long Vĩ); bờ biển từ Móng Cái tới Thanh
> Hóa — đúng vệt Yagi quét qua — chỉ có **3 nút** (Quảng Ninh, Thái Bình, Nam
> Định). Nay thêm 18 UAV đặt cố định chia 9 cặp, mỗi cặp gồm **một tầng sát biển**
> (hạ cánh cùng lúc với trạm mặt đất) và **một tầng lùi vào bờ ~17 km** (vẫn bay
> được, giữ vai trò trung kế khi tổ ong mất). Thêm nút `ThanhHoa` để cụm Bắc Trung
> Bộ có nút chủ thật.
>
> | Cặp | UAV sát biển → nút chủ | UAV phía trong → nút chủ |
> |---|---|---|
> | Quảng Ninh | `UAV-HonGai` (12.2 km), `UAV-BaiChay` (11.1), `UAV-VanDon` (8.8), `UAV-HaiHa` (7.7) → `QuangNinh` | |
> | Hải Phòng | `UAV-DoSon` (7.2), `UAV-CatHai` (18.1), `UAV-TienLang` (6.0) → `HaiPhong` | `UAV-UongBi` (10.2), `UAV-ThuongLy` (5.7), `UAV-KienAn` (6.1) → `HaiPhong`; `UAV-SaVi` (17.5), `UAV-CatBa` (13.2) → `QuangNinh` |
> | Thái Bình / Nam Định | `UAV-ThaiThuy` (14.7), `UAV-TienHai` (8.7), `UAV-HaiHau` (12.1) → `ThaiBinh`; `UAV-GiaoThuy` (16.8) → `NamDinh` | |
> | Bắc Trung Bộ | `UAV-SamSon` (8.4) → `ThanhHoa`; `UAV-Vinh` (2.5, 90°) → `Vinh` | |
>
> Mọi khoảng cách ≤ 18.1 km < tầm phủ thật 22.4 km và góc nâng ≥ 6.5° > 5°, nên
> **0 UAV mồ côi**. Góc nâng là điều kiện dễ hỏng nhất (bẫy `UAV-QuangYen` cũ:
> 24.1 km nhưng 4.65° ⇒ không có cạnh, và **không có dòng nào trong `link_stats`**
> vì bảng chỉ ghi cặp `exists == 1`).
> Test `test_uav_ven_bien_bac_trung_bo_phu_khep_vet_bao` khoá cả ba điều kiện.
>
> Số liệu Yagi thật trên các cột mới (7–8/9/2024, `thoi_tiet_vn_mr.npz`):
>
> | Điểm | Mưa max | Giật max | Giờ giông |
> |---|---|---|---|
> | `UAV-DoSon` | 48.2 mm/h | 45.9 m/s | 6 |
> | `UAV-CatBa` | 24.5 mm/h | 51.5 m/s | 9 |
> | `QuangNinh` | 15.8 mm/h | 47.8 m/s | 14 |
> | `HaiPhong` | 44.7 mm/h | 41.8 m/s | 4 |
> | `ThanhHoa` / `UAV-SamSon` | 9.7 / 8.1 mm/h | 16.1 / 17.5 m/s | 0 |
>
> Nghĩa là **cụm Quảng Ninh–Hải Phòng** thật sự vào `ha_canh` trong bão; Thanh
> Hóa chỉ ở rìa (giật 17.5 m/s, vừa qua ngưỡng 15 m/s 1–2 giờ, 0 giờ giông) —
> đừng viết cả 18 UAV đều hạ cánh vì Yagi.
>
> Cập nhật tệp thời tiết sau khi thêm điểm KHÔNG cần tải lại 70 điểm: dùng
> `python dataset/cong_cu_bo_sung_thoi_tiet.py` — nó chỉ tải các cột thiếu, giữ
> nguyên 51 cột cũ **từng bit**, rồi chạy luật CAPE (`them-giong`) cho cột mới.
> Bước giông hay bị `429 Minutely API request limit`; chạy lại bằng
> `--chi-giong` để không phải tải lại precip/gust.
>
> **Cụm ven biển Bắc (bản đầu).**
> `EXTRA_COASTAL_SITES` thêm 3 trạm Quảng Ninh / Thái Bình / Nam Định, và
> `EXTRA_UAV_SITES` thêm 3 UAV đặt cố định quanh Hải Phòng. Lý do: trước đó toàn
> bộ nút mở rộng là hải đảo xa, nên **không UAV nào ở gần tâm bão Yagi** — đúng
> chỗ mà `ha_canh` cần để có ý nghĩa.
>
> Thái Bình (30.7 m/s) và Nam Định (27.6 m/s) nằm ở **rìa** bão, không phải tâm:
> chúng yếu hơn Hải Phòng (41.8), Quảng Ninh (47.8) và Bạch Long Vĩ (58.7) rõ rệt.
> Đừng viết ba trạm này là "tâm bão".
>
> | UAV | Vị trí | Phủ (25 km, góc nâng ≥ 5°) |
> |---|---|---|
> | `UAV-HaiPhong` | 20.845, 106.688 | HaiPhong (0 km) |
> | `UAV-QuangYen` | 20.870, 106.890 | HaiPhong (21.3 km), QuangNinh (21.8 km) |
> | `UAV-BachLongVi` | 20.133, 107.724 | BachLongVi (2.5 km, góc nâng 90°) |
>
> **Độ cao UAV đặt cố định: `UAV_DAT_ALT_M = 2500.0` m** (không phải 2000 m).
> Với ngưỡng `min_elev_deg = 5.0`, góc nâng ở CÙNG một vị trí chỉ phụ thuộc độ
> cao UAV. Đo trên cặp Hải Phòng / Quảng Ninh (cùng cách `UAV-QuangYen` ~21 km):
>
> | Độ cao | Hải Phòng | Quảng Ninh | Biên so với 5° |
> |---|---|---|---|
> | 2000 m | 5.29° | 5.18° | +0.29° / **+0.18°** (quá mỏng) |
> | **2500 m** | 6.63° | 6.49° | **+1.63° / +1.49°** |
>
> Quét lưới toạ độ ±0.12° quanh UAV-QuangYen cho thấy biên đối xứng tối đa chỉ
> **+0.29°** ở 2000 m, nên dịch toạ độ là vô ích — độ cao mới là đòn bẩy đúng.
> Truyền `make_placed_uavs(alt_m=2000.0)` để tái lập dataset sinh trước bản vá.
>
> **`max_range_km` KHÔNG BAO GIỜ chặn (cả UAV lẫn HAPS).** Ở 2000 m, góc nâng 5°
> ứng với tầm **22.4 km** < 25 km; ở 20 km, góc nâng 10° ứng với tầm **108 km**
> < 200 km. Nếu muốn `max_range_km` thật sự có hiệu lực thì phải **hạ
> `min_elev_deg`**, không phải nâng `max_range_km`.
>
> **`HAPS-BienDong` không phủ nút nào** — đã đo, không phải lỗi. Do tầm thật
> 108 km, nó không với tới Hoàng Sa (274 km) hay Trường Sa (>900 km), mà
> Hoàng Sa ↔ Trường Sa cách nhau 906 km nên **không một HAPS đơn lẻ nào** phủ
> được cả hai. Giữ nút làm trung kế Biển Đông; phủ đảo đã do `UAV-HoangSa` /
> `UAV-TruongSa` đảm nhận.
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

## 6. Hai tệp dataset: v2 (mốc 73 nút) và v3 (mạng mở rộng 170 nút)

```bash
# Mốc hồi quy — TUYỆT ĐỐI không ghi đè
python dataset/oracle_labeler_sagsin.py --out dataset/oracle_dataset_v2.npz

# Mạng mở rộng (170 nút, 100 vệ tinh). Labeler từ chối ghi --mo-rong vào v2.
python dataset/oracle_labeler_sagsin.py --mo-rong --sats 100 --infra space,haps,uav \
    --out dataset/oracle_dataset_v3.npz
```

| | v2 | v3 |
|---|---|---|
| Số nút | 73 | **170** |
| Số vệ tinh (`--sats`) | 50 | **100** |
| Hình dạng `W_snapshots` | (600, 73, 73) | (600, 170, 170) |
| `node_latlon_alt` | ✗ | ✓ (600, 170, 3) |
| `link_stats` + `link_stat_cols` | ✗ | ✓ (1 037 010, 8) |
| `mo_rong` | ✗ | ✓ `[True]` |

170 = 100 vệ tinh + 28 ground + 7 HAPS + 35 UAV (5 bay vòng + 30 đặt cố định).
Thời gian sinh trên máy này: **~25 phút** cho 600 snapshot × 170 nút (bản 101 nút
trước đó ~6 phút) — tăng theo bình phương số nút vì mỗi snapshot dựng ma trận
đầy đủ rồi cắt cạnh.

> **Lịch sử kích thước:** 101 nút (50 vệ tinh + 27 ground + 7 HAPS + 17 UAV) là
> bản trước khi thêm 18 UAV ven biển Bắc Bộ/Bắc Trung Bộ và nút `ThanhHoa`.
> Tệp 101 nút cũ KHÔNG còn dùng được với `sites.py` hiện tại: `node_order` của nó
> thiếu 19 UAV mới nên các test số nút (120/170) sẽ báo lệch. Nếu cần bản 101 nút
> để so sánh lịch sử, sinh lại bằng `--sats 50` **sau khi** tạm bỏ 18 UAV — hoặc
> đơn giản là so trên 73 nút đầu (phần chung với v2).

**Hai tệp dùng chung chỉ số cho 73 nút đầu**: khi v3 còn 101 nút
(50 vệ tinh) thì `v3.node_order[:73] == v2.node_order` và `W_snapshots` hạn chế về
73 nút đó **trùng khớp tuyệt đối từng bit** (0 / 3 197 400 ô lệch).

> ⚠️ **Không còn đúng khi v3 dùng 100 vệ tinh.** `sat_names = sorted({...})` sort
> TOÀN CỤC danh sách vệ tinh, nên tên của 50 vệ tinh cũ chen giữa các tên mới:
> chỉ số tuyệt đối đổi (50/50 vệ tinh cũ vẫn còn mặt, `node_order[:73] != v2`).
> Muốn so sánh hai bản phải tra **theo tên** (đó là cách các test hiện làm). Khi
> so theo tên, đo được trên tệp hiện tại:
>
> | Cặp | Ô lệch / tổng |
> |---|---|
> | ground↔ground, ground↔UAV, ground↔HAPS, UAV↔UAV, UAV↔HAPS, HAPS↔HAPS | **0** / 1 584 400 |
> | GSL (vệ tinh ↔ ground/UAV/HAPS) | 23 434 / 1 230 000 |
> | ISL (vệ tinh ↔ vệ tinh) | 305 748 / 1 500 000 |
>
> Nghĩa là **mọi luật mặt đất/không lưu trùng tuyệt đối** (chúng chỉ phụ thuộc toạ
> độ, độ cao, khoảng cách — không phụ thuộc TLE). Chỉ các cặp có vệ tinh đổi, vì
> `sample_sats` lấy mẫu **độc lập** nên "mẫu 100 vệ tinh" không chứa nguyên "mẫu 50"
> theo nghĩa thời điểm nhìn thấy: một số cặp GSL cũ mất, một số cặp mới có.
> Hệ quả & cách xử lý: xem §6b.

### 6b. Vì sao "không mưa nhưng vẫn dứt"

Số đo trên hai tệp (600 snapshot, 2024-01-01 → 2025-12-31):

| | v2 (50 vệ tinh × 23 trạm) | v3 (100 × 70) |
|---|---|---|
| Giờ mỗi trạm thấy **ít nhất 1** vệ tinh | **38.1 %** | 68.7 % |
| Giờ **không** thấy vệ tinh nào | **61.9 %** | 31.3 % |
| Giờ thấy ≥ 2 vệ tinh | 12.0 % | 32.3 % |
| Cặp ISL / snapshot | 314 | 1 294 |

Nguyên nhân gốc **không phải thời tiết**, mà là ngưỡng hình học:

1. `MIN_ELEVATION_DEG = 25.0` (`src/physics/tle_loader.py`) — rất cao so với
   Starlink thật (thường dùng 25° cho chất lượng, nhưng 10–15° vẫn có link). Ở
   550 km, ngưỡng 25° giới hạn bán kính phủ xuống ~**520 km**, nên mỗi trạm chỉ
   "thấy" 0.54 → 0.67 vệ tinh *trung bình*, và **62 % số giờ là 0**.
2. Khi một trạm 0 vệ tinh và không có UAV/HAPS phủ (hoặc UAV đang `ha_canh` vì
   gió), trạm đó cô lập — bất kể mưa. Trong `generate_dataset_snapshots`, mỗi
   cặp endpoint không có `path` bị **`continue` im lặng** (`oracle_labeler_sagsin.py`
   dòng 338–342): nó không vào `reachable_pairs`, nhãn biến mất, và từ góc nhìn
   dữ liệu thì "đường bị đứt" dù trời tạnh.
3. Vì vậy **`label[có nhãn]` phụ thuộc trực tiếp vào số vệ tinh**: đổi `--sats`
   50 → 100 làm 31 % số giờ-thuộc-trạm được cứu, nhưng đồng thời **7 187 cặp GSL
   cũ bị mất** (mẫu 50 ⊄ mẫu 100 theo *thời điểm nhìn thấy*). Hai bản **không so
   sánh trực tiếp** được; muốn so phải cùng một mẫu vệ tinh (xem lưu ý dưới).

**Cách sửa đúng tuỳ mục tiêu:**

* Muốn bớt "dứt" vô cớ → **hạ `min-elev`** khi sinh (có cờ sẵn):
  `--min-elev 15`. Ở 550 km, 15° phủ bán kính ~**880 km**, số vệ tinh thấy được
  tăng ~3× và cặp GSL tăng tương ứng — không đụng gì tới mưa.
* Muốn **giữ mẫu vệ tinh cũ** khi mở rộng → sinh v3 với cùng 50 vệ tinh rồi thêm
  vệ tinh **nối đuôi** (cần sửa `sample_sats` để lấy mẫu lồng nhau thay vì
  `rng_sat.sample` độc lập). Chưa làm: hiện `seed` không làm mẫu lồng nhau, nên
  đổi `--sats` là đổi toàn bộ tập vệ tinh theo thời gian.
* Muốn thấy rõ "dứt do mưa" → tách thống kê: đếm cặp mất đường **khi trời tạnh**
  (`precip < 0.1` và `gust < 15`) — đó là các ca do hình học, không do mưa. Con
  số này chính là bảng trên (31–62 % số giờ-trạm).


> Bản v3 **trước** bản vá này có 14 304 / 3 197 400 ô lệch (tối đa 2.48e-05 ms)
> do ma trận 94×94 chuẩn hoá float32 ra chữ số cuối khác 73×73. Bản hiện tại
> khớp từng bit, nên có thể dùng `array_equal`. Test vẫn dùng `np.allclose` cho
> an toàn nếu sau này mạng đổi kích thước.

`link_stats` — một hàng cho mỗi liên kết mỗi snapshot, ghi **hai chiều** (`u→v`, `v→u`):

`snapshot_id, u, v, distance_km, elevation_deg, bandwidth_mbps, packet_loss, exists`

`u`/`v` là chỉ số trong `node_order`. `elevation_deg` chỉ có nghĩa với liên kết
GSL/local (vệ tinh–mặt đất, air–ground).

