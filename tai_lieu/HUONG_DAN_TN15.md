# Chạy TN15 — hướng dẫn

**TN15:** học lại DM theo J vật lý (1 điểm % mất DV = 10 ms trễ) thay cho J chuẩn hoá 7/3, kiểm xem có nhanh hơn (nhất là bớt
tình huống nghẽn nặng) mà không kém an toàn không; đồng thời xác nhận giải mã không tái nhiễu của TN12b. Tiêu chí chốt trước:
`results/bao_da_tuyen/TIEU_CHI_15.md`.

**Máy chạy:** máy đã chạy TN13 (RTX 4060, 20 GB RAM) — dùng lại `data/bao_da_tuyen/tn13/` (val giai đoạn A có thầy theo x10, 11
bộ kiểm tra sinh lại có lưới Dijkstra) và `results/bao_da_tuyen/tn13/chon_b.json`. Thiếu thì script tự sinh (lâu thêm ~1,5 giờ).

## Trên máy đang làm (một lần)

```bash
git add scripts/tn15.py scripts/tn14.py results/bao_da_tuyen/TIEU_CHI_15.md results/bao_da_tuyen/TIEU_CHI_14.md tai_lieu/HUONG_DAN_TN15.md
```
```bash
git commit -m "TN15: DM học theo J vật lý x10; TN14: lưới giải mã mới + số đo đuôi trễ"
```
```bash
git push
```

## Trên máy chạy

1. `git pull`.
2. Chạy:
   ```
   python -m scripts.tn15 toi-da
   ```
   - Log: `logs/tn15.log`; mỗi mô hình `logs/tn15_dm13_x10_s<seed>.log`.
   - **Thời gian ước tính (chưa đo):** ~3–3,5 giờ — gán nhãn lại tập học theo x10 ~45 phút, học 3 mô hình song song ~45 phút,
     chấm 11 bộ (2 mô hình × 2 cách giải mã × 3 seed) ~1,5–2 giờ.
   - Bước `kiem` in tỷ lệ bản ghi sinh lại khớp bản gốc; dưới 99% có CẢNH BÁO nhưng vẫn chạy tiếp — ghi lại để báo.
   - Bị dừng: chạy lại đúng lệnh; phần đã xong được giữ.
3. **Gửi lại đúng một tệp:** `results/bao_da_tuyen/TN15_ket_qua.zip` (báo cáo `KET_QUA_15.md`, json, hình, bản ghi chấm từng
   tình huống, đường học + trọng số DM x10, log). Bị dừng trước bước cuối: `python -m scripts.tn15 phan-tich`.

## Pipeline làm gì

| Bước | Việc |
|---|---|
| `chuan-bi` | Kiểm / sinh dữ liệu TN13 cần dùng (val giai đoạn A, 11 bộ kiểm tra có lưới Dijkstra) |
| `sinh` | Gán nhãn lại train10, train6_10, x11 theo x10 → `data/bao_da_tuyen/tn13/x10/` |
| `kiem` | Kiểm tái lập so với bản gốc → `tn15/kiem_tai_lap.json` |
| `huan-luyen` | `dm13_x10_s0..2` (checkpoint theo J x10 trên val) → `mo_hinh13/` |
| `cham` | 11 bộ × 3 seed: DM11 và DM x10, mỗi mô hình giải mã đầy đủ + không tái nhiễu |
| `phan-tich` | `tn15/KET_QUA_15.md`, `hinh_15.png`, gói zip |
