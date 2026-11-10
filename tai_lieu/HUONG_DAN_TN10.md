# Chạy TN10 — hướng dẫn

**Máy chạy:** CPU ≥ 8 luồng, RAM ≥ 16 GB. Không cần GPU (huấn luyện chạy trên CPU).

**Thời gian** trên máy 8 luồng, máy đã chạy TN8 (có sẵn val, val_b, bộ kiểm tra):
- khoảng 15–18 giờ;
- thêm khoảng 5 giờ nếu bước 3 chọn thầy 64 (phải gắn lại nhãn).
- Máy chưa chạy TN8 / TN9 thì cộng thêm thời gian sinh val_b, val6_b, bộ kiểm tra B 10 / 20 luồng và ngay9, nam9, bao9.

Tiêu chí chốt trước: `results/bao_da_tuyen/TIEU_CHI_10.md`.

Hai nơi chạy: **máy bạn trong nhóm** (mục "Trên máy chạy") và **Google Colab** (mục "Chạy trên Colab").

## Trên máy đang làm (một lần)

```bash
git add src/nhieu_luong/huan_luyen_dt.py src/nhieu_luong/mo_hinh_dt.py scripts/kiem_dinh_boi_9.py scripts/thu_nhanh_tn10.py scripts/dong_goi_tn10.py colab/TN10_colab.ipynb results/bao_da_tuyen/TIEU_CHI_10.md tai_lieu/HUONG_DAN_TN10.md .gitignore
git commit -m "TN10: chạy trên Colab (GPU tự chọn, lưu Drive) + tiêu chí chốt trước"
git push
```

- Phải push **đủ** các tệp trên. Thiếu `huan_luyen_dt.py` / `mo_hinh_dt.py` thì huấn luyện báo
  `unrecognized arguments: --thiet-bi auto`. Thiếu `kiem_dinh_boi_9.py` thì bước phân tích cuối lỗi.
- Bước kiểm tra nhanh (`thu_nhanh_tn10`, ô 6 của notebook) có mục `0. Code` để bắt lỗi push thiếu ngay từ đầu.

Rồi tạo gói dữ liệu cho Colab (đã tạo sẵn `TN10_du_lieu.zip` ở gốc repo, ~0,7 GB; muốn tạo lại thì chạy lệnh dưới):

```bash
python -m scripts.dong_goi_tn10
```

## Chạy trên Colab

1. Tải `TN10_du_lieu.zip` lên Google Drive, vào thư mục `PBL4_TN10` (tạo thư mục nếu chưa có). Không có gói cũng chạy
   được, nhưng Colab phải tự sinh thêm val và bộ kiểm tra.
2. Mở `colab/TN10_colab.ipynb` trên Colab: File → Open notebook → GitHub → `huyth1204/PBL4`, nhánh `mang-mo-rong-v3`.
3. Repo private thì thêm `GITHUB_TOKEN` vào Secrets của Colab (biểu tượng chìa khoá).
4. Runtime → Change runtime type → GPU. Colab Pro thì bật thêm High-RAM.
5. Runtime → Run all. Để tab mở.
6. **Bị ngắt** (hết 12 giờ ở bản miễn phí, hoặc mất mạng): mở lại notebook, Run all. Dữ liệu, mô hình, kết quả nằm trên
   Drive nên bước đã xong tự bỏ qua; chỉ mô hình đang học dở phải học lại.
7. Xong: lấy `PBL4_TN10/TN10_ket_qua_colab.zip` trên Drive (ô 9 của notebook tạo ra).

Thời gian ước tính:
- **Colab miễn phí** (2 CPU + T4): khoảng 25–35 giờ, tức 3–4 phiên. Sinh dữ liệu và chấm chạy trên CPU nên chậm; huấn
  luyện chạy trên GPU.
- **Colab Pro, High-RAM** (nhiều CPU): khoảng 15–20 giờ, một phiên 24 giờ.

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
   - Máy có GPU NVIDIA thì thêm `--thiet-bi auto` để huấn luyện trên GPU (lỗi GPU tự quay về CPU).
   - Máy đang chạy commit `2980f7f`: cứ để chạy, kết quả CPU giống hệt bản mới. Riêng bước cuối "phân tích thí nghiệm
     10" sẽ lỗi vì thiếu `kiem_dinh_boi_9.py`. Khi đó `git pull` rồi chạy lại đúng lệnh: mọi bước đã xong tự bỏ qua, chỉ
     chạy phần phân tích.
   - Máy đã `git pull` commit `0f843d8` / `1eb793f` (thiếu `huan_luyen_dt.py` mới) sẽ lỗi `unrecognized arguments:
     --thiet-bi` ngay bước 1. Cũng chỉ cần `git pull` bản đủ rồi chạy lại đúng lệnh; dữ liệu đã sinh không mất.
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
