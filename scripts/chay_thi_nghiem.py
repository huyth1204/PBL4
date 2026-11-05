"""
chay_thi_nghiem.py — Chạy lại toàn bộ thí nghiệm 6 hoặc 7 theo đúng thứ tự đã chạy (tai_lieu/BAO_CAO_THI_NGHIEM_6_7.md).

    python -m scripts.chay_thi_nghiem 6 --chi-in          # chỉ in các lệnh
    python -m scripts.chay_thi_nghiem 7                   # chạy hết
    python -m scripts.chay_thi_nghiem 7 --tu-buoc 9       # chạy tiếp từ bước 9 (bước đã xong tự bỏ qua tệp có sẵn)
    python -m scripts.chay_thi_nghiem hinh                # chỉ vẽ lại hình của tài liệu

Điều kiện trước: dữ liệu của thí nghiệm 4–5 (data/bao_da_tuyen/{train,val}_k4_5, val5_k4_5, mua5_k*, yagi5_k*) và mô hình
results/bao_da_tuyen/mo_hinh/dm.pt (thí nghiệm 4). Mỗi bước sinh dữ liệu / chấm đều bỏ qua tệp đã có, nên chạy lại an
toàn. Thời gian trên máy 8 luồng: thí nghiệm 6 ≈ 6–8 giờ, thí nghiệm 7 ≈ 5–6 giờ.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

GOC = Path(__file__).resolve().parents[1]
M4, M6, M7 = "results/bao_da_tuyen/mo_hinh", "results/bao_da_tuyen/mo_hinh6", "results/bao_da_tuyen/mo_hinh7"
K_MOC = {"4-5": [], "10": ["--buoc-moc", "2"], "20": ["--buoc-moc", "4"]}


def sinh(tap, K="4-5", thay=False, tt=2):
    return (f"sinh dữ liệu {tap} {K} luồng", ["-m", "src.nhieu_luong.tao_du_lieu_dt", "--tap", tap, "--K", K, *K_MOC[K],
                                            *([] if thay else ["--khong-thay"]), "--tien-trinh", str(tt)])


def cham(tap, K, mo_hinh, ten, khoa_tkcb10=""):
    return (f"chấm {ten} trên {tap} {K} luồng", ["-m", "src.nhieu_luong.cham_lai_dt", "--tap", tap, "--K", K, "--mo-hinh",
                                                mo_hinh, "--ten", ten, "--khoa", ten, "--khoa-tkcb10", khoa_tkcb10,
                                                "--tien-trinh", "1"])


def buoc_6():
    cu = ["train", "train_k4_5", "val", "val_k4_5", "val5", "val5_k4_5"] + \
         [x for b in ("mua5", "yagi5") for K in ("k4_5", "k10", "k20") for x in (b, f"{b}_{K}")]
    B = [sinh("train6", thay=True), sinh("val6", thay=True),
         ("bổ sung 3 đặc trưng giông quanh UAV cho dữ liệu cũ (169 tệp)", ["-m", "scripts.bo_sung_dac_trung", *cu,
                                                                          "--tien-trinh", "2"]),
         ("huấn luyện DM6 (có đặc trưng mới)", ["-m", "src.nhieu_luong.huan_luyen_dt", "dm", "--thi-nghiem", "6", "--ten",
                                               "dm6", "--them"]),
         ("huấn luyện DM6a (không đặc trưng mới)", ["-m", "src.nhieu_luong.huan_luyen_dt", "dm", "--thi-nghiem", "6",
                                                   "--ten", "dm6a"]),
         ("chọn số bước DM6", ["-m", "src.nhieu_luong.huan_luyen_dt", "chon-buoc", "--thi-nghiem", "6", "--ten", "dm6"]),
         ("chọn số bước DM6a", ["-m", "src.nhieu_luong.huan_luyen_dt", "chon-buoc", "--thi-nghiem", "6", "--ten", "dm6a"]),
         cham("val5", "4-5", M6, "dm6"), cham("val6", "4-5", M6, "dm6"),
         ("chọn τ6 theo mất dịch vụ trên val5 + val6", ["-m", "scripts.phan_tich_6", "chon-tau"])]
    B += [sinh("c6", K) for K in K_MOC]
    for tap in ("c6", "mua5", "yagi5"):
        for K in K_MOC:
            B.append(cham(tap, K, M6, "dm6", "online_tkcb10_6"))
        B.append(cham(tap, "4-5", M6, "dm6a"))
    B += [cham("c6", K, M4, "dm") for K in K_MOC]                        # DM của thí nghiệm 4 trên bộ C mới
    B.append(("phân tích thí nghiệm 6", ["-m", "scripts.phan_tich_6", "phan-tich"]))
    return B


def buoc_7():
    B = []
    for t in ("c7wh", "y7wh", "c7w", "c7h", "y7w", "y7h"):                # 7a: thế giới cực đoan giả lập, seed + 800
        B += [sinh(t, "4-5", tt=1), cham(t, "4-5", M6, "dm6", "online_tkcb10_6")]
    for t in ("c7wh", "y7wh"):
        for K in ("10", "20"):
            B += [sinh(t, K, tt=1), cham(t, K, M6, "dm6", "online_tkcb10_6")]
    B += [sinh("train7s", thay=True), sinh("train7x", thay=True)]       # 7b: dữ liệu cho mốc 200%
    for ten, them in (("dm7_25", ["--ti-le", "0.25"]), ("dm7_50", ["--ti-le", "0.5"]), ("dm7_200", ["--gap-doi"])):
        B.append((f"huấn luyện {ten}", ["-m", "src.nhieu_luong.huan_luyen_dt", "dm", "--thi-nghiem", "7", "--ten", ten,
                                       "--them", *them]))
    B.append(("cố định S = 8, 5 bước cho mọi mô hình 7b", "chon_buoc_7"))
    B += [cham(tap, "4-5", M7, ten) for ten in ("dm7_25", "dm7_50", "dm7_200")
          for tap in ("c6", "mua5", "yagi5", "c7wh", "y7wh")]
    B.append(("phân tích thí nghiệm 7", ["-m", "scripts.phan_tich_7"]))
    return B


def buoc_hinh():
    return [("hình dữ liệu", ["-m", "scripts.truc_quan_du_lieu"]), ("hình kết quả", ["-m", "scripts.hinh_bao_cao_6_7"])]


def chon_buoc_7():
    tep = GOC / M7 / "chon_buoc.json"
    tep.parent.mkdir(parents=True, exist_ok=True)
    tep.write_text(json.dumps({"S": 8, "so_buoc": 5, "quy_tac": "thí nghiệm 7b: cố định 5 bước, S = 8 (không chọn lại)"},
                              ensure_ascii=False, indent=1), encoding="utf-8")


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("thi_nghiem", choices=["6", "7", "hinh"])
    ap.add_argument("--chi-in", action="store_true", help="chỉ in lệnh, không chạy")
    ap.add_argument("--tu-buoc", type=int, default=1)
    a = ap.parse_args()
    B = {"6": buoc_6, "7": buoc_7, "hinh": buoc_hinh}[a.thi_nghiem]()
    for i, (mo_ta, lenh) in enumerate(B, 1):
        if i < a.tu_buoc:
            continue
        hien = lenh if isinstance(lenh, str) else "python " + " ".join(f'"{x}"' if x == "" else x for x in lenh)
        print(f"[{i}/{len(B)}] {mo_ta}\n    {hien}", flush=True)
        if a.chi_in:
            continue
        if lenh == "chon_buoc_7":
            chon_buoc_7()
        else:
            subprocess.run([sys.executable, "-W", "ignore", *lenh], cwd=GOC, check=True)


if __name__ == "__main__":
    main()
