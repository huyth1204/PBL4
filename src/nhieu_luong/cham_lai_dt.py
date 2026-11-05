"""
cham_lai_dt.py — Thí nghiệm 5 (results/bao_da_tuyen/TIEU_CHI_5.md): chấm DM, DM + TKCB và ρ̂ cho chính sách lai.

    python -m src.nhieu_luong.cham_lai_dt --tap val5 --K 4-5
    python -m src.nhieu_luong.cham_lai_dt --tap mua5 --K 10 --tien-trinh 3

Với mỗi bản ghi (đã sinh bằng tao_du_lieu_dt, có đối chứng và thế giới chấm thu gọn): lấy MỘT mẫu DM (mô hình của thí
nghiệm 4, 5 bước, S = 8), rồi TKCB trên J tính bằng 32 thế giới bốc theo bảng, xuất phát từ CHÍNH mẫu đó; đo thời gian cả
hai trong cùng tiến trình, 1 luồng CPU; ρ̂ tính online từ bài toán gọn; TKCB 10 lần đo thời gian ở mỗi 4 mốc (mốc của
C5). Chấm trên thế giới CHẤM và thế giới quan sát đã lưu. Lai được ghép ở bước phân tích theo τ chọn trên val5.
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
from src.nhieu_luong.cham_mo_hinh_dt import _tg
from src.nhieu_luong.tao_du_lieu_dt import SO_DO, phan_tich_K, thu_muc, tkcb10

_dm = None
_cfg = {"thu_muc": None, "ten": "dm", "khoa": "dm", "khoa_tkcb10": "online_tkcb10"}


def _khoi_tao(cfg):
    _cfg.update(cfg)


def rho_hat(gon):
    """Hệ số tải online: mọi luồng đi tuyến có trễ nền nhỏ nhất trong kho → phần dung lượng còn trống C·(1 − ρ0) bị
    dùng tại liên kết nghẽn nhất. Chỉ dùng λ và dung lượng / tải nền quan sát ở o."""
    d0 = dt.tre(gon.prop, gon.C, gon.rho0, 0.0)
    F = np.zeros(len(gon.U))
    for k in range(gon.K):
        M = gon.inc[k]
        F += gon.lam[k] * M[int(np.argmin((M * d0).sum(1)))]
    m = F > 0
    return float(np.max(F[m] / (gon.C[m] * (1 - gon.rho0[m]))))


def _mot_tep(tep):
    global _dm
    torch.set_num_threads(1)
    if _dm is None:
        _dm = md.ham_chon(_cfg["thu_muc"], _cfg["ten"], softmax=False)["dm"]
    k_dm, k_ls = _cfg["khoa"], _cfg["khoa"] + "_tkcb"
    recs = pickle.load(open(tep, "rb"))
    if recs and k_ls in recs[0]["ens"]:
        return tep, 0
    cac_t = sorted({r["t"] for r in recs})
    for r in recs:
        gon = r["gon"]
        s = time.perf_counter()
        a_dm = _dm(gon)
        ms_dm = (time.perf_counter() - s) * 1e3
        s = time.perf_counter()
        B = dt.TheGioi.tu_bang(gon, np.random.default_rng([r["kb"], r["t"], 4]))
        a_ls, _ = dt.tkcb(B, a_dm)
        r["tg_ms"][k_dm], r["tg_ms"][k_ls] = ms_dm, ms_dm + (time.perf_counter() - s) * 1e3
        if cac_t.index(r["t"]) % 4 == 0 and _cfg["khoa_tkcb10"]:
            r["tg_ms"][_cfg["khoa_tkcb10"]] = tkcb10(gon, np.random.default_rng([r["kb"], r["t"], 3]))[1]
        r["rho_hat"] = rho_hat(gon)
        for ten_tg, khoa in (("ens", "cham"), ("qs", "cham_qs")):
            tgw = _tg(gon, r[khoa])
            for kieu, a in ((k_dm, a_dm), (k_ls, a_ls)):
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
    ap.add_argument("--tap", required=True)
    ap.add_argument("--K", default="4-5")
    ap.add_argument("--tien-trinh", type=int, default=3)
    ap.add_argument("--mo-hinh", default=None, help="thư mục mô hình (mặc định của thí nghiệm 4)")
    ap.add_argument("--ten", default="dm", help="tên checkpoint DM")
    ap.add_argument("--khoa", default="dm", help="khoá kết quả (dm → dm, dm_tkcb)")
    ap.add_argument("--khoa-tkcb10", default="online_tkcb10", help="khoá thời gian TKCB 10 lần; rỗng = không đo")
    a = ap.parse_args()
    from pathlib import Path
    cfg = {"thu_muc": Path(a.mo_hinh) if a.mo_hinh else None, "ten": a.ten, "khoa": a.khoa, "khoa_tkcb10": a.khoa_tkcb10}
    teps = sorted(glob.glob(str(thu_muc(a.tap, phan_tich_K(a.K)) / "kb_*.pkl")))
    print(f"{a.tap} K={a.K}: {len(teps)} tệp, mô hình {cfg['ten']} → khoá {cfg['khoa']}", flush=True)
    t0 = time.time()
    with Pool(a.tien_trinh, initializer=_khoi_tao, initargs=(cfg,)) as pool:
        for k, (tep, n) in enumerate(pool.imap_unordered(_mot_tep, teps)):
            print(f"  {k + 1}/{len(teps)}  {os.path.basename(tep)}  {n} bản ghi  ({time.time() - t0:.0f} s)", flush=True)
    print("xong", flush=True)


if __name__ == "__main__":
    main()
