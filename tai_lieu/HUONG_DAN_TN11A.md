# Chạy TN11a — hướng dẫn

**Chạy trên máy đã chạy TN10.** Cần:
- dữ liệu `data/bao_da_tuyen/train10_k4_5`, `train6_10_k4_5`, `val_k4_5`, `val6_k4_5`;
- mô hình `results/bao_da_tuyen/mo_hinh10/`.

Không sinh nhãn mới, không chấm bộ kiểm tra. Tiêu chí chốt trước: `results/bao_da_tuyen/TIEU_CHI_11A.md`.

## Trên máy đang làm (một lần)

```bash
git add src/nhieu_luong/mo_hinh_dt.py src/nhieu_luong/huan_luyen_dt.py scripts/tn11a.py results/bao_da_tuyen/TIEU_CHI_11A.md tai_lieu/HUONG_DAN_TN11A.md
git commit -m "TN11a: chẩn đoán khoảng cách DM10 → thầy (DM nhìn trước + đối chứng, G đã học, ngưỡng ρ̂), chế độ GPU tối đa"
git push
```

## Trên máy chạy (GPU 8 GB, RAM 32 GB)

1. **Lấy code:** `git pull`.
   - Nếu báo xung đột vì đã sửa tay code ở máy chạy: `git stash`, rồi `git pull`.
2. **Kiểm tra torch có thấy GPU:**
   ```
   python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
   ```
   - Ra `False` tức là torch bản CPU. Cài bản CUDA (~2,5 GB):
     ```
     pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cu121
     ```
   - Không cài cũng chạy được: chế độ tối đa tự chuyển sang chỉ dùng CPU.
3. **Chạy hết công suất** (giữ máy không ngủ, đóng các việc nặng khác):
   ```
   mkdir logs
   python -m scripts.tn11a toi-da > logs/tn11a.log 2>&1
   ```
   - **Tự dò** CPU, RAM, GPU rồi chia việc. Dòng đầu log in ra thông số đã chọn.
   - Với máy 20 luồng, 32 GB, GPU 8 GB, kế hoạch dự kiến:

     | Giai đoạn | Việc |
     |---|---|
     | Bổ sung | 7 tiến trình (mỗi tiến trình ~4 GB RAM) |
     | Học trên GPU | **8 mô hình cùng lúc** (4 nhìn trước + 4 đối chứng), mỗi mô hình 2 luồng CPU nạp dữ liệu |
     | Trong lúc GPU học | 4 tiến trình CPU chấm luôn phần B, C (không cần mô hình mới) |
     | Học xong | 20 tiến trình chấm phần A, rồi phân tích |

   - **Ước tính khoảng 1 giờ.** Chưa đo được trên GPU thật vì máy người viết không có GPU. Không có GPU thì khoảng
     1,5–2 giờ.
   - **Bị dừng giữa chừng:** chạy lại đúng lệnh, phần đã xong tự bỏ qua.
   - **GPU lỗi lúc khởi động một mô hình:** mô hình đó tự học bằng CPU.
   - **GPU lỗi giữa chừng (log báo "huấn luyện lỗi"):** chạy lại đúng lệnh. Vẫn lỗi thì chạy bằng CPU:
     `python -m scripts.tn11a tat-ca --tien-trinh 8 --song-song 4 --luong-cpu 5 --thiet-bi cpu`.
   - Không bật thêm HighQoS / tiến trình phụ khi đang chạy.
4. **Gửi lại** (nén cả gói):
   - `results/bao_da_tuyen/tn11a/`: `KET_QUA_11A.md`, `ket_qua_11a.json`, `bo_sung.json` và các tệp chấm;
   - `results/bao_da_tuyen/mo_hinh11a/*_nhat_ky.json`;
   - `logs/tn11a.log`, `logs/tn11a_*.log`.

## Pipeline làm gì

| Bước | Việc |
|---|---|
| `bo-sung` | Dựng lại kịch bản cùng seed. Thêm 3 đặc trưng "nhìn trước" (đỉnh mưa thật, giông thật ở trạm mặt đất, UAV hạ cánh thật trong cửa sổ giữ tuyến) vào train10, train6_10, val, val6. Chỉ **thêm** khoá, không đổi gì của TN10 |
| `huan-luyen` | `nt60h128_p0..3` (có 3 đặc trưng) và `dc60h128_p0..3` (đối chứng, không có), đều đúng cấu hình e60h128 của TN10 → `mo_hinh11a/` |
| `cham` | **A:** kiểm thử chéo nt và dc (S 8)<br>**B:** e120h128 của TN10 chấm trên phần **đã học** (S 8)<br>**C:** DM10 (e120h128, S 32) + TKCB, kèm ρ̂ |
| `phan-tich` | Áp luật của TIEU_CHI_11A trên nửa train6 → `tn11a/KET_QUA_11A.md` |

Chạy từng bước, chỉ CPU: `python -m scripts.tn11a bo-sung | huan-luyen | cham | phan-tich --tien-trinh 8 --song-song 4
--luong-cpu 5`.
