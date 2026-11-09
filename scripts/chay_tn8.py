"""
chay_tn8.py — Chạy toàn bộ thí nghiệm 8 (đánh giá chéo 2 × 2, thế giới A / thế giới B) bằng một lệnh, từ gốc repo:

    python -m scripts.chay_tn8                         # tự chọn số tiến trình = nửa số luồng CPU (tối đa 6)
    python -m scripts.chay_tn8 --tien-trinh 4          # đặt số tiến trình sinh dữ liệu (mỗi tiến trình ~2,5 GB RAM)
    python -m scripts.chay_tn8 --chi-in                # chỉ in các bước
    python -m scripts.chay_tn8 --tu-buoc 20            # chạy tiếp từ bước 20 (bước đã có sản phẩm tự bỏ qua)

Xem tai_lieu/HUONG_DAN_TN8.md. Kết quả: results/bao_da_tuyen/{KET_QUA_8.md, ket_qua_8.json, hinh_8.png}.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.chay_thi_nghiem import main  # noqa: E402

if __name__ == "__main__":
    sys.argv[1:1] = ["8"]
    main()
