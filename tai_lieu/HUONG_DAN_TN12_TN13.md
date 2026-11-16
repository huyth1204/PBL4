# Chạy TN12 và TN13 — hướng dẫn

**Máy chạy:** máy nhóm đã chạy TN11 (cần `results/bao_da_tuyen/mo_hinh11/` có các tệp `.pt`, và dữ liệu TN11 trong
`data/bao_da_tuyen/`: `train10`, `train6_10`, `x11`, `val`, `val6`, 11 bộ kiểm tra 4–5 luồng). Thiết kế cho
**~20 GB RAM + GPU 8 GB**, như TN11.

Tiêu chí chốt trước: `results/bao_da_tuyen/TIEU_CHI_12.md`, `results/bao_da_tuyen/TIEU_CHI_13.md`.

- **TN12:** cơ chế khuếch tán có thật là yếu tố tạo độ chính xác (thang bậc 10 phương pháp, 10 mô hình học lại).
- **TN13:** tỷ lệ 7/3 trong J có tốt nhất không, hay kết luận không phụ thuộc tỷ lệ (giai đoạn A trên val, giai đoạn B học lại
  DM theo 2 mục tiêu chọn tự động theo luật ghi trước).

## Trên máy đang làm (một lần)

```bash
git add src/nhieu_luong/mo_hinh_dt.py src/nhieu_luong/da_tuyen.py src/nhieu_luong/huan_luyen_dt.py src/nhieu_luong/tao_du_lieu_dt.py scripts/tn11.py scripts/tn12.py scripts/tn13.py results/bao_da_tuyen/TIEU_CHI_12.md results/bao_da_tuyen/TIEU_CHI_13.md tai_lieu/HUONG_DAN_TN12_TN13.md
```
```bash
git commit -m "TN12: thang bậc tách cơ chế khuếch tán; TN13: kiểm tỷ lệ 7/3 của J"
```
```bash
git push
```

## Trên máy chạy

1. **Lấy code:** `git pull` (nếu xung đột vì sửa tay ở máy chạy: `git stash`, rồi `git pull`).
2. **Chạy TN12 rồi TN13** (giữ máy không ngủ, đóng việc nặng khác). Nếu repo này **không phải** thư mục đã chạy TN11
   (ví dụ repo mới `D:\PBL4_v3`, còn TN11 chạy ở `D:\PBL4`), thêm `--tn11 <thư mục đã chạy TN11>`: script tự chép sang những
   gì còn thiếu — mô hình `mo_hinh11/`, kết quả `tn11/` và dữ liệu (`train10`, `train6_10`, `x11`, `val`, `val6`, 11 bộ kiểm
   tra). Tệp đã có không bị ghi đè. Trong PowerShell:
   ```
   python -m scripts.tn12 toi-da --tn11 D:\PBL4; python -m scripts.tn13 toi-da --tn11 D:\PBL4
   ```
   Hai thí nghiệm độc lập: TN12 lỗi thì TN13 vẫn chạy. Muốn chạy riêng thì gõ từng lệnh.
   - Log tự ghi vào `logs/tn12.log`, `logs/tn13.log`; mỗi mô hình có `logs/tn12_<tên>.log`, `logs/tn13_<tên>.log`.
   - Dòng đầu log in cấu hình tài nguyên đã chọn (như TN11: sinh 4 tiến trình, học 5 mô hình cùng lúc trên GPU, chấm 19
     tiến trình). Nới / siết RAM: thêm `--ram 24` hoặc `--ram 16`. Học bằng CPU: thêm `--vram 0`.
   - **Thời gian ước tính (chưa đo, dựa trên thời gian TN11):** TN12 khoảng 4–6 giờ (học 10 mô hình ~1,5 giờ, chấm ~3 giờ);
     TN13 khoảng 7–10 giờ (giai đoạn A ~2 giờ, gán nhãn lại + sinh lại bộ kiểm tra ~3–4 giờ, học 6 mô hình ~1 giờ, chấm ~1,5
     giờ). Tổng có thể hơn 12 giờ — nên chạy qua đêm.
   - **Bị dừng giữa chừng:** chạy lại đúng lệnh. Phần đã xong (dữ liệu, mô hình, tệp chấm) tự bỏ qua.
   - **Log báo "huấn luyện lỗi":** xem log của mô hình đó rồi chạy lại.
   - Bước `kiem-a` / `kiem-b` của TN13 in tỷ lệ bản ghi sinh lại khớp bản gốc. Dưới 99% thì có dòng CẢNH BÁO nhưng vẫn
     chạy tiếp; ghi lại để báo.
3. **Gửi lại đúng hai tệp:**
   - `results/bao_da_tuyen/TN12_ket_qua.zip`
   - `results/bao_da_tuyen/TN13_ket_qua.zip`

   Mỗi tệp gồm báo cáo `KET_QUA_1x.md`, `ket_qua_1x.json`, biểu đồ, **bản ghi chấm từng tình huống** (`cham_tep/`), đường học
   và trọng số các mô hình mới, mọi log. Nếu bị dừng trước bước cuối: `python -m scripts.tn12 phan-tich` /
   `python -m scripts.tn13 phan-tich` để gói lại phần đã xong.

## Chạy trên máy KHÁC máy đã chạy TN11

Máy mới không có mô hình và dữ liệu TN11 (lỗi "thiếu mô hình TN11 ở …mo_hinh11"). Chép bằng một file zip:

1. **Trên máy đã chạy TN11** (sau `git pull`):
   ```
   python -m scripts.tn12 goi-tn11
   ```
   Tạo `results/bao_da_tuyen/TN11_cho_TN12_TN13.zip` (khoảng 1 GB): mô hình `mo_hinh11/`, kết quả `tn11/`, dữ liệu `train10`,
   `train6_10`, `x11`, `val`, `val6` và 11 bộ kiểm tra.
2. Chép file zip sang máy mới (Google Drive, USB…), để ở đâu cũng được, ví dụ `E:\TN11_cho_TN12_TN13.zip`.
3. **Trên máy mới**, thêm `--tn11 <đường dẫn file zip>` — script tự giải nén những file còn thiếu vào đúng chỗ rồi chạy:
   ```
   python -m scripts.tn12 toi-da --tn11 E:\TN11_cho_TN12_TN13.zip
   ```
   Lần sau không cần `--tn11` nữa.

Máy không GPU, RAM ít hơn 20 GB thì vẫn chạy được nhưng chậm hơn: học bằng CPU, và TN13 chỉ sinh dữ liệu được ~2 tiến trình
với 14 GB RAM. Nếu được, chạy TN13 trên máy có GPU và 20 GB RAM.

## Pipeline làm gì

**TN12** (`python -m scripts.tn12 <bước>`):

| Bước | Việc |
|---|---|
| `huan-luyen` | dm12k (DM không phối hợp), dm12l (DM lịch nhiễu tuyến tính): 3 seed mỗi họ. Softmax chống quá khớp: 2 cấu hình ở seed 0, chọn theo J val, rồi seed 1, 2 → `mo_hinh12/` |
| `cham` | 11 bộ kiểm tra × 3 seed: SM11 1 lần / 32 mẫu; DM11 1 bước × 32, 5 bước × 1, 5 × 32; 3 họ mới; quét số bước × số chuỗi (seed 0) |
| `phan-tich` | `tn12/KET_QUA_12.md`, `hinh_12_thang.png` (thang bậc), `hinh_12_quet.png` (độ chính xác – thời gian), `hinh_12_hoc.png` (đường học), gói zip |

**TN13** (`python -m scripts.tn13 <bước>`):

| Bước | Việc |
|---|---|
| `sinh-a` | Sinh lại val / val6 với thầy theo 11 mục tiêu và Dijkstra phạt 16 bộ β, γ → `data/bao_da_tuyen/tn13/` |
| `kiem-a` | Kiểm tình huống sinh lại có trùng bản gốc |
| `phan-tich-a` | Đường đánh đổi của thầy, điểm gối, luật rẽ nhánh → `tn13/chon_b.json`, `hinh_13_a.png` |
| `sinh-b` | Gán nhãn lại tập học theo 2 mục tiêu đã chọn; sinh lại 11 bộ kiểm tra (không thầy) có lưới Dijkstra phạt |
| `kiem-b` | Kiểm tái lập dữ liệu giai đoạn B |
| `huan-luyen-b` | DM13 theo từng mục tiêu, 3 seed → `mo_hinh13/` |
| `cham-b` | DM 7/3 (DM11) và DM13 trên 11 bộ kiểm tra; Dijkstra phạt mọi bộ β, γ |
| `phan-tich` | `tn13/KET_QUA_13.md`, `hinh_13_b.png`, gói zip |

Dung lượng thêm trên đĩa: TN13 sinh dữ liệu mới trong `data/bao_da_tuyen/tn13/` (ước tính vài GB).
