"""
dong_goi_tn8.py — Đóng gói dữ liệu thế giới A của TN6, mô hình DM6 và code + tài liệu TN8 để chép sang máy chạy TN8.

    python -m scripts.dong_goi_tn8                 # → TN8_du_lieu.zip ở thư mục gốc repo (~0,6 GB)

Trên máy chạy: clone / pull repo, giải nén TN8_du_lieu.zip vào thư mục gốc repo (giữ nguyên đường dẫn), rồi chạy
`python -m scripts.thu_nhanh_tn8` và `python -m scripts.chay_tn8` (xem tai_lieu/HUONG_DAN_TN8.md).
"""

import sys
import zipfile
from pathlib import Path

GOC = Path(__file__).resolve().parents[1]
CAN = ["data/bao_da_tuyen/train_k4_5", "data/bao_da_tuyen/train6_k4_5", "data/bao_da_tuyen/val_k4_5",
       "data/bao_da_tuyen/val6_k4_5", *[f"data/bao_da_tuyen/{b}_k{k}" for b in ("c6", "yagi5") for k in ("4_5", "10", "20")],
       "results/bao_da_tuyen/mo_hinh6"]
# Các tệp đầu vào sau đã có trong git, chỉ kiểm tra xem có đủ không
TRONG_GIT = ["data/thoi_tiet_vn_mr.npz", "data/thoi_tiet_vn_2017_2021_1.npz", "dataset/oracle_dataset_v5.npz",
             "dataset/v5/split_info.json", "data/bao_manh/bang_tra_cuoi/bang_tra.npz",
             "data/bao_manh/bang_tra_cuoi/meta.json", "data/bao_manh/danh_muc.json"]
# Code + tài liệu TN8 (đóng kèm để máy chạy không phải chờ push git; giải nén đè vào repo)
CODE = ["src/ngau_nhien/kich_ban.py", "src/ngau_nhien/ma_tran_rui_ro.py", "src/nhieu_luong/the_gioi_rr.py",
        "src/nhieu_luong/tao_du_lieu_dt.py", "src/nhieu_luong/huan_luyen_dt.py", "scripts",
        "results/bao_da_tuyen/TIEU_CHI_8.md", "results/bao_da_tuyen/KE_HOACH_8.md", "tai_lieu/HUONG_DAN_TN8.md",
        "tai_lieu/HUONG_DAN_NHANH_TN8.md"]


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    thieu = [t for t in CAN + TRONG_GIT + CODE if not (GOC / t).exists()]
    if thieu:
        raise SystemExit("Thiếu: " + ", ".join(thieu))
    ra = GOC / "TN8_du_lieu.zip"
    n = 0
    with zipfile.ZipFile(ra, "w", compression=zipfile.ZIP_STORED) as z:     # pickle đã nén kém, không nén thêm cho nhanh
        for t in CAN + CODE:
            p = GOC / t
            teps = [p] if p.is_file() else sorted(p.rglob("*"))
            for f in teps:
                if f.is_file() and not f.name.endswith((".tam", ".pyc")) and "__pycache__" not in f.parts:
                    z.write(f, f.relative_to(GOC).as_posix())
                    n += 1
        z.write(GOC / "tai_lieu" / "HUONG_DAN_NHANH_TN8.md", "DOC_TRUOC_HUONG_DAN_NHANH_TN8.md")
    print(f"{ra}  ({n} tệp, {ra.stat().st_size / 2**20:.0f} MB)")


if __name__ == "__main__":
    main()
