"""
bo_sung_dac_trung.py — Thí nghiệm 6 (results/bao_da_tuyen/TIEU_CHI_6.md): thêm 3 đặc trưng giông quanh UAV
(giong_lan_can_uav, giong_1h_uav, cape_uav) cho dữ liệu đã sinh trước thí nghiệm 6, bằng cách dựng lại kịch bản cùng
seed và cùng tập cạnh hợp lệ lúc quyết định. Kiểm tra: băng tần đã lưu phải trùng băng tần tính lại cho mọi cạnh.

    python -m scripts.bo_sung_dac_trung train train_k4_5 val val_k4_5 val5 val5_k4_5 --tien-trinh 2
"""

from __future__ import annotations

import glob
import os
import pickle
import sys
import time
from multiprocessing import Pool

import numpy as np

from src.ngau_nhien.kich_ban import sinh_kich_ban
from src.ngau_nhien.ma_tran_rui_ro import ma_tran_online
from src.nhieu_luong import tao_du_lieu_rr as td
from src.nhieu_luong.bai_toan_rr import DELTA, TUOI
from src.nhieu_luong.tao_du_lieu_dt import RA, cac_cua_so_tap, cape_luc, dac_trung_uav


def _mot_tep(job):
    tap, tep = job
    if td._g is None:
        td._khoi_tao()
    recs = pickle.load(open(tep, "rb"))
    if not recs or "cape_uav" in recs[0]["gon"].dac_trung_canh:
        return tep, 0
    g, tt, p = td._g, td._tt, td._p
    ten, h0, seed = cac_cua_so_tap(tap)[recs[0]["kb"]]
    Wn, q = sinh_kich_ban(g, tt, h0, seed, p)
    canh = {}
    for r in recs:
        t = r["t"]
        o = t - TUOI
        if t not in canh:
            past = {k: q[k][:o + 1] for k in ("R", "gust", "giong", "tat")}
            valid = ma_tran_online(td._bang, g, Wn[o], past, t, TUOI, DELTA, tham_so=p)["valid"] & (q["rho"][o] < 1.0)
            np.fill_diagonal(valid, False)
            canh[t] = np.nonzero(valid)
        gon = r["gon"]
        u, v = canh[t][0][gon.U], canh[t][1][gon.U]
        if not np.array_equal(g.band[u, v], gon.dac_trung_canh["band"]):
            raise ValueError(f"{tep}: ánh xạ cạnh không khớp ở t = {t}")
        lc, g1, cp = dac_trung_uav(g, q, o, cape_luc(h0, o))
        gon.dac_trung_canh.update({"giong_lan_can_uav": lc[u] | lc[v], "giong_1h_uav": g1[u] | g1[v],
                                   "cape_uav": np.maximum(cp[u], cp[v]).astype(np.float32)})
    tam = tep + ".tam"
    with open(tam, "wb") as f:
        pickle.dump(recs, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tam, tep)
    return tep, len(recs)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    a = sys.argv[1:]
    n = 2
    if "--tien-trinh" in a:
        i = a.index("--tien-trinh")
        n = int(a[i + 1])
        a = a[:i] + a[i + 2:]
    jobs = [(tap, f) for tap, thu_muc in zip(a[::2], a[1::2]) for f in sorted(glob.glob(str(RA / thu_muc / "kb_*.pkl")))]
    print(f"{len(jobs)} tệp", flush=True)
    t0 = time.time()
    with Pool(n, initializer=td._khoi_tao) as pool:
        for k, (tep, m) in enumerate(pool.imap_unordered(_mot_tep, jobs)):
            if k % 10 == 0 or k == len(jobs) - 1:
                print(f"  {k + 1}/{len(jobs)}  {os.path.basename(os.path.dirname(tep))}/{os.path.basename(tep)}  {m} bản ghi  "
                      f"({time.time() - t0:.0f} s)", flush=True)
    print("xong", flush=True)


if __name__ == "__main__":
    main()
