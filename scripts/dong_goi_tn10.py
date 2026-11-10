"""
dong_goi_tn10.py — Đóng gói dữ liệu đã có (git không lưu data/bao_da_tuyen) để chạy TN10 trên Colab nhanh hơn: val dùng
chọn checkpoint và bộ kiểm tra của TN8 / TN9. Thiếu tập nào thì pipeline trên Colab tự sinh.

    python -m scripts.dong_goi_tn10            # → TN10_du_lieu.zip ở thư mục gốc repo (~0,7 GB)

Tải TN10_du_lieu.zip lên Google Drive, vào thư mục PBL4_TN10 (xem colab/TN10_colab.ipynb, tai_lieu/HUONG_DAN_TN10.md).
"""

import sys
import zipfile
from pathlib import Path

GOC = Path(__file__).resolve().parents[1]
DL = "data/bao_da_tuyen"
TAP = ["val_k4_5", "val6_k4_5", "val_b_k4_5", "val6_b_k4_5",
       *[f"{b}_k{k}" for b in ("c6", "yagi5", "c6_b", "yagi5_b") for k in ("4_5", "10", "20")],
       "ngay9_k4_5", "nam9_k4_5", "bao9_k4_5"]


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ra, n, co = GOC / "TN10_du_lieu.zip", 0, []
    with zipfile.ZipFile(ra, "w", compression=zipfile.ZIP_STORED) as z:     # pickle nén kém, không nén thêm cho nhanh
        for t in TAP:
            teps = sorted((GOC / DL / t).glob("kb_*.pkl"))
            if not teps:
                continue
            co.append(f"{t} ({len(teps)})")
            for f in teps:
                z.write(f, f.relative_to(GOC).as_posix())
                n += 1
            for f in (GOC / DL / t).glob(".xong"):
                z.write(f, f.relative_to(GOC).as_posix())
    print("Đã gói (số cửa sổ):", ", ".join(co))
    print("Không có, Colab sẽ tự sinh:", ", ".join(t for t in TAP if not any(c.startswith(t + " ") for c in co)) or "–")
    print(f"{ra}  ({n} tệp, {ra.stat().st_size / 2**20:.0f} MB)")


if __name__ == "__main__":
    main()
