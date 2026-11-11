# Chạy TN11a — hướng dẫn

**Chạy trên máy đã chạy TN10** (cần `data/bao_da_tuyen/train10_k4_5`, `train6_10_k4_5`, `val_k4_5`, `val6_k4_5` và
`results/bao_da_tuyen/mo_hinh10/`). Không sinh nhãn mới, không chấm bộ kiểm tra.

Thời gian trên máy 20 luồng: khoảng **1–1,5 giờ**.

| Bước | Thời gian |
|---|---|
| Bổ sung đặc trưng | ~5 phút |
| Huấn luyện 4 mô hình 60 epoch | ~45 phút |
| Chấm | ~30 phút |

Tiêu chí chốt trước: `results/bao_da_tuyen/TIEU_CHI_11A.md`.

## Trên máy đang làm (một lần)

```bash
git add src/nhieu_luong/mo_hinh_dt.py src/nhieu_luong/huan_luyen_dt.py scripts/tn11a.py results/bao_da_tuyen/TIEU_CHI_11A.md tai_lieu/HUONG_DAN_TN11A.md
git commit -m "TN11a: chẩn đoán khoảng cách DM10 → thầy (DM nhìn trước, G đã học, ngưỡng ρ̂)"
git push
```

## Trên máy chạy

1. **Lấy code:** `git pull`.
   - Nếu báo xung đột vì đã sửa tay code ở máy chạy (khoá, HighQoS…): `git stash`, rồi `git pull`.
2. **Chạy** (giữ máy không ngủ):
   ```
   python -m scripts.tn11a tat-ca --tien-trinh 8 --song-song 4 --luong-cpu 5 > logs/tn11a.log 2>&1
   ```
   - Bị dừng giữa chừng thì chạy lại đúng lệnh trên: phần đã xong tự bỏ qua.
   - Bước `bo-sung` chỉ **thêm** 3 khoá đặc trưng vào tệp dữ liệu TN10 (ghi qua tệp tạm). Không đổi gì của TN10.
   - Không bật thêm HighQoS / tiến trình phụ khi đang chấm. TN11a không đo thời gian, nhưng nên để nguyên cho dễ đối
     chiếu.
3. **Gửi lại** (nén cả gói):
   - `results/bao_da_tuyen/tn11a/` (gồm `KET_QUA_11A.md`, `ket_qua_11a.json`, `bo_sung.json` và các tệp chấm);
   - `results/bao_da_tuyen/mo_hinh11a/*_nhat_ky.json`;
   - `logs/tn11a.log`, `logs/tn11a_nt60h128_p*.log`.

## Pipeline làm gì

| Bước | Việc |
|---|---|
| `bo-sung` | Dựng lại kịch bản cùng seed, thêm 3 đặc trưng "nhìn trước" (đỉnh mưa thật, giông thật ở trạm mặt đất, UAV hạ cánh thật trong cửa sổ giữ tuyến) vào train10, train6_10, val, val6 |
| `huan-luyen` | `nt60h128_p0..3`: đúng e60h128 của TN10 (cùng seed, phần, dữ liệu, val), thêm 3 đặc trưng trên → `mo_hinh11a/` |
| `cham` | **A:** kiểm thử chéo nt60h128 (S 8), so với e60h128 của TN10 (đọc kết quả đã lưu ở `tn10/kiem_thu_cheo/`)<br>**B:** e120h128 của TN10 chấm trên phần **đã học** (S 8)<br>**C:** DM10 (e120h128, S 32) + TKCB trên bảng, kèm ρ̂ |
| `phan-tich` | Áp luật của TIEU_CHI_11A → `tn11a/KET_QUA_11A.md` |

Có thể chạy từng bước: `python -m scripts.tn11a bo-sung | huan-luyen | cham | phan-tich` (cùng tham số).
