"""
chay_tn10.py — Chạy toàn bộ thí nghiệm 10 ("học tới nơi") bằng một lệnh, từ gốc repo:

    python -m scripts.chay_tn10 --tien-trinh 4 --song-song 2 --luong-cpu 3
    python -m scripts.chay_tn10 --chi-in               # chỉ in các bước

Tiêu chí: results/bao_da_tuyen/TIEU_CHI_10.md. Hướng dẫn: tai_lieu/HUONG_DAN_TN10.md. Kết quả:
results/bao_da_tuyen/{KET_QUA_10.md, ket_qua_10.json, hinh_10.png}, results/bao_da_tuyen/tn10/.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.chay_thi_nghiem import main  # noqa: E402

if __name__ == "__main__":
    sys.argv[1:1] = ["10"]
    main()
