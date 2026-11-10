# Chạy TN10 — hướng dẫn

**Máy chạy:** CPU ≥ 8 luồng, RAM ≥ 16 GB. Không cần GPU (huấn luyện chạy trên CPU).

**Thời gian** trên máy 8 luồng, máy đã chạy TN8 (có sẵn val, val_b, bộ kiểm tra):
- khoảng 15–18 giờ;
- thêm khoảng 5 giờ nếu bước 3 chọn thầy 64 (phải gắn lại nhãn).
- Máy chưa chạy TN8 / TN9 thì cộng thêm thời gian sinh val_b, val6_b, bộ kiểm tra B 10 / 20 luồng và ngay9, nam9, bao9.

Tiêu chí chốt trước: `results/bao_da_tuyen/TIEU_CHI_10.md`.

## Trên máy đang làm (một lần)

```bash
git add src scripts results/bao_da_tuyen tai_lieu data/thoi_tiet_vn_2017_2021_1.npz .gitignore
git commit -m "TN10: học tới nơi — kiểm thử chéo 4 phần, 4 bước chọn, chạy trên hai máy"
git push
```

`data/thoi_tiet_vn_2017_2021_1.npz` (13 MB) cần cho bộ kiểm tra nam9, bao9.

## Trên máy chạy

1. **Lấy code:** `git pull`.
2. **Cài thư viện** (nếu chưa): `pip install -r requirements.txt`.
3. **Kiểm tra nhanh** (3–5 phút): `python -m scripts.thu_nhanh_tn10`. Phải thấy dòng cuối `OK — chạy: ...`.
4. **Chạy** (giữ máy không ngủ):
   ```
   mkdir logs
   python -m scripts.chay_tn10 --tien-trinh 4 --song-song 2 --luong-cpu 4 > logs/tn10.log 2>&1
   ```
   - `--tien-trinh`: số tiến trình sinh dữ liệu / chấm. Mỗi tiến trình khoảng 2,5 GB RAM.
   - `--song-song` × `--luong-cpu` nên bằng số luồng CPU. Mỗi mô hình huấn luyện tốn khoảng 1,2 GB RAM (A + B: 2,5 GB).
   - Bị dừng giữa chừng thì chạy lại đúng lệnh trên. Dữ liệu, mô hình và kết quả chấm đã xong đều tự bỏ qua.
5. **Gửi lại:**
   - `results/bao_da_tuyen/KET_QUA_10.md`, `ket_qua_10.json`, `hinh_10.png`;
   - cả thư mục `results/bao_da_tuyen/tn10/` (chứa `chon.json` và kết quả kiểm thử chéo);
   - `results/bao_da_tuyen/mo_hinh10/*_nhat_ky.json` (đường học);
   - `logs/tn10.log`.

## Pipeline làm gì

1. Sinh tập học có thế giới chấm cho thế giới A: `train10`, `train6_10`.
2. **Bước 1:** 4 cấu hình × 4 phần = 16 mô hình. Chọn cấu hình bằng kiểm thử chéo.
3. **Bước 2:** chọn số chuỗi S ∈ {8, 16, 32}.
4. **Bước 3:** so thầy 64 với thầy 32 thế giới nhãn trên val10. Chỉ khi qua ngưỡng mới gắn lại nhãn và học lại 4 phần.
5. **Bước 4:** sinh tập học thế giới B, học 4 phần trên A + B, chọn.
6. Huấn luyện DM10 (cấu hình chốt) và DM6r (cấu hình DM6) trên toàn bộ tập học, mỗi họ 3 seed.
7. Chấm Lai(DM6r) và Lai(DM10) trên bộ kiểm tra; phân tích → `KET_QUA_10.md`.

Mỗi lựa chọn và số liệu kiểm thử chéo được ghi ở `results/bao_da_tuyen/tn10/chon.json` ngay khi bước đó xong. Muốn xem
tiến độ: `python -c "import json; print(json.dumps(json.load(open('results/bao_da_tuyen/tn10/chon.json', encoding='utf-8')), ensure_ascii=False, indent=1)[:3000])"`.
