"""
danh_gia_fw.py — Bổ sung (23/10/2026, sau khi chấm thí nghiệm 3): đối chứng Frank–Wolfe + làm tròn tỉ lệ trên đúng các
tình huống Yagi của thí nghiệm 2 (20–160 luồng) và 3 (500–2.000 luồng). Phát hiện trên val; báo như kết quả bổ sung,
không thuộc tiêu chí đã chốt. Ghi results/yagi3/fw_<tập>.npz (cùng thứ tự tình huống với test_<tập>.npz).

    python -m src.nhieu_luong.danh_gia_fw --tap yagi2:k20_40,yagi2:k80
"""
import argparse
import glob
import pickle
import sys
import time
from pathlib import Path

import numpy as np

from src.nhieu_luong import khong_gioi_han as kg

GOC = Path(__file__).resolve().parents[2]
MUC = ("nhe", "vua", "nang")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tap", required=True)
    a = ap.parse_args()
    for muc_tap in a.tap.split(","):
        tn, ten_k = muc_tap.split(":")
        cot = {"kb": [], "t": [], "muc": [], "FWR": [], "tg_FWR": [], "FW_chia": []}
        for f in sorted(glob.glob(str(GOC / "data" / tn / "test" / ten_k / "kb_*.pkl"))):
            for x in pickle.load(open(f, "rb")):
                for m in MUC:
                    md = kg.MangDay.nap({**x["mang"], "lam": x["lam"][m]})
                    s = time.perf_counter()
                    cap, cua = kg.cap_nguon_dich(md)
                    duong, w, c_chia = kg.fw_duong(md, cap, cua)
                    r = kg.lam_tron_ti_le(md, duong, w, cua)
                    tg = time.perf_counter() - s
                    for k, v in (("kb", x["kb"]), ("t", x["t"]), ("muc", m), ("FWR", md.chi_phi(r)), ("tg_FWR", tg),
                                 ("FW_chia", c_chia)):
                        cot[k].append(v)
        np.savez(GOC / "results" / "yagi3" / f"fw_{tn}_{ten_k}.npz", **{k: np.array(v) for k, v in cot.items()})
        print(muc_tap, len(cot["kb"]), "xong", flush=True)


if __name__ == "__main__":
    main()
