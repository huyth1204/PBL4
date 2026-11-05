"""
cham_mo_hinh_dt.py — Chấm DM / softmax trên Yagi đã sinh (thí nghiệm 4, results/bao_da_tuyen/TIEU_CHI_4.md).

    python -m src.nhieu_luong.cham_mo_hinh_dt --K 4-5 --tien-trinh 3

Đọc data/bao_da_tuyen/test_k<K>/kb_*.pkl (đối chứng + thầy đã chấm lúc sinh), thêm DM và softmax: lời giải chọn trong
kho chỉ từ thông tin online, chấm trên cùng 32 thế giới CHẤM và thế giới quan sát đã lưu (thu gọn trên kho, tương
đương chính xác với bài toán đầy đủ cho mọi lời giải trong kho). Thời gian của DM, softmax và TKCB 10 lần trên J bảng
(mốc của C5, đo ở mỗi 4 mốc) đo trong CÙNG tiến trình, 1 luồng CPU, lần lượt trên cùng bài toán.
Ghi đè từng tệp (thêm khoá), chạy lại thì bỏ qua tệp đã có DM.
"""

from __future__ import annotations

import argparse
import glob
import os
import pickle
import sys
import time
from multiprocessing import Pool

import numpy as np
import torch

from src.nhieu_luong import da_tuyen as dt
from src.nhieu_luong import mo_hinh_dt as md
from src.nhieu_luong.tao_du_lieu_dt import RA, SO_DO, phan_tich_K, thu_muc, tkcb10

_chon = None


def _tg(gon, x):
    du = x["du"].astype(float)
    return dt.TheGioi(gon, x["song"], du[:, None], np.zeros_like(du)[:, None])


def _mot_tep(tep):
    global _chon
    torch.set_num_threads(1)
    if _chon is None:
        _chon = md.ham_chon()
    recs = pickle.load(open(tep, "rb"))
    if recs and "dm_tkcb" in recs[0]["ens"]:
        return tep, 0
    chi_them = bool(recs) and "dm" in recs[0]["ens"]          # đã chấm DM: chỉ thêm DM + TKCB (phân tích thêm)
    cac_t = sorted({r["t"] for r in recs})
    for r in recs:
        gon = r["gon"]
        sol = {}
        for kieu, f in _chon.items():
            if chi_them and kieu != "dm":
                continue
            s = time.perf_counter()
            a = f(gon)
            ms = (time.perf_counter() - s) * 1e3
            if not chi_them:
                r["tg_ms"][kieu] = ms
                sol[kieu] = a
            if kieu == "dm":
                # phân tích thêm (ngoài tiêu chí): TKCB trên J bảng xuất phát từ lời giải DM
                s = time.perf_counter()
                B = dt.TheGioi.tu_bang(gon, np.random.default_rng([r["kb"], r["t"], 4]))
                sol["dm_tkcb"], _ = dt.tkcb(B, a)
                r["tg_ms"]["dm_tkcb"] = ms + (time.perf_counter() - s) * 1e3
        if cac_t.index(r["t"]) % 4 == 0 and not chi_them:
            a, ms = tkcb10(gon, np.random.default_rng([r["kb"], r["t"], 3]))
            r["tg_ms"]["online_tkcb10"] = ms
        for ten_tg, khoa in (("ens", "cham"), ("qs", "cham_qs")):
            tgw = _tg(gon, r[khoa])
            for kieu, a in sol.items():
                c = tgw.cham(gon.tu_lua_chon(a))
                r[ten_tg][kieu] = {k: c[k] for k in SO_DO}
    tam = tep + ".tam"
    with open(tam, "wb") as f:
        pickle.dump(recs, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tam, tep)
    return tep, len(recs)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", default="4-5")
    ap.add_argument("--tien-trinh", type=int, default=3)
    a = ap.parse_args()
    teps = sorted(glob.glob(str(thu_muc("test", phan_tich_K(a.K)) / "kb_*.pkl")))
    print(f"test K={a.K}: {len(teps)} tệp", flush=True)
    t0 = time.time()
    with Pool(a.tien_trinh) as pool:
        for k, (tep, n) in enumerate(pool.imap_unordered(_mot_tep, teps)):
            print(f"  {k + 1}/{len(teps)}  {os.path.basename(tep)}  {n} bản ghi  ({time.time() - t0:.0f} s)", flush=True)
    print("xong", flush=True)


if __name__ == "__main__":
    main()
