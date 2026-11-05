"""
chay_thu_da_tuyen.py — Chạy thử thí nghiệm 4 trên VAL trước khi chốt tiêu chí (results/bao_da_tuyen/TIEU_CHI_4.md).
Không đọc dữ liệu Yagi.

    python -m scripts.chay_thu_da_tuyen chay --tien-trinh 3      # 16 cửa sổ val × 24 mốc × 3 mức tải
    python -m scripts.chay_thu_da_tuyen tong-hop                 # → results/bao_da_tuyen/chay_thu_val.json

Đo (1) khoảng trống giữa "thầy" (TKCB trên J ensemble, dùng tương lai — trần cho một bộ chọn trong kho) và Dijkstra
phạt (β, γ tốt nhất trên val) về tỉ lệ mất dịch vụ và J; (2) chỉnh β, γ; (3) TKCB có ra đúng tối ưu duyệt hết ở 3 luồng.
Cùng cơn, cửa sổ, split hình học, seed kịch bản như thí nghiệm 3.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import pickle
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from src.ngau_nhien.kich_ban import sinh_kich_ban
from src.ngau_nhien.ma_tran_rui_ro import ma_tran_online
from src.nhieu_luong import da_tuyen as dt
from src.nhieu_luong import tao_du_lieu_rr as td
from src.nhieu_luong import the_gioi_rr as tg
from src.nhieu_luong.bai_toan import MUC_TAI
from src.nhieu_luong.bai_toan_rr import DELTA, TUOI

GOC = Path(__file__).resolve().parents[1]
RA = GOC / "data" / "bao_da_tuyen" / "chay_thu"
KQ = GOC / "results" / "bao_da_tuyen"
CAU_HINH = [(b, c) for b in dt.BETA for c in dt.GAMMA]
SO_DO = ("J", "mat_dv_%", "khong_tuyen_song_%", "du_phong_qua_tai_%", "tre_chinh_ms")
MAX_DUYET = 3_000                       # số tổ hợp tối đa (2 luồng đầu) khi kiểm TKCB = duyệt hết ở 3 luồng


def _mot_cua_so(job):
    i, ten, h0, seed, moc, tep = job
    if td._g is None:
        td._khoi_tao()
    if Path(tep).exists():
        return tep, 0, 0.0
    g, tt, p = td._g, td._tt, td._p
    t_bd = time.time()
    Wn, q = sinh_kich_ban(g, tt, h0, seed, p)
    pl = tg.phat_lai(g, tt, h0, seed, p)
    rng = np.random.default_rng(seed + 7)
    hang = []
    for so, t in enumerate(moc):
        t = int(t)
        o = t - TUOI
        past = {k: q[k][:o + 1] for k in ("R", "gust", "giong", "tat")}
        r = ma_tran_online(td._bang, g, Wn[o], past, t, TUOI, DELTA, tham_so=p)
        th = dt.tao_tinh_huong(g, q, r, t, rng)
        if th is None:
            continue
        seeds = [int(x.generate_state(1)[0]) for x in np.random.SeedSequence([seed, t]).spawn(dt.SO_TG)]
        s = time.perf_counter()
        Y = dt.yen_m(th)
        tg_yen = time.perf_counter() - s
        S = dt.suurballe_m(th)
        for muc in MUC_TAI:
            lam = th["lam"] * MUC_TAI[muc] / th["u"]
            DP, tg_dp = {}, []
            for b, c in CAU_HINH:
                s = time.perf_counter()
                DP[(b, c)] = dt.dijkstra_phat(th, lam, b, c)
                tg_dp.append(time.perf_counter() - s)
            them = [Y[k] + [r_ for cf in CAU_HINH for r_ in DP[cf][k]] for k in range(len(lam))]
            bt = dt.BaiToan(th, muc, them)
            E = dt.TheGioi.ensemble(bt, g, pl, p, seeds)
            Q = dt.TheGioi.quan_sat(bt, Wn, q)
            s = time.perf_counter()
            B = dt.TheGioi.tu_bang(bt, np.random.default_rng([seed, t, 1]))
            a_on, _ = dt.tkcb(B, dt.tham_lam(B))
            tg_on = time.perf_counter() - s
            sol = {"yen": bt.loi_giai(Y), "suurballe": bt.loi_giai(S), "online_tkcb": bt.tu_lua_chon(a_on)}
            for cf in CAU_HINH:
                sol[f"dphat_{cf[0]:g}_{cf[1]:g}"] = bt.loi_giai(DP[cf])
            # thầy: TKCB trên J ensemble, nhiều điểm xuất phát
            dau = [a_on]
            for ten_dau in ("suurballe", "dphat_10_10"):
                a = [dt.lua_chon_cua(bt, k, [bt.bang[k][j] for j in sol[ten_dau][k]]) for k in range(bt.K)]
                if all(x is not None for x in a):
                    dau.append(np.array(a))
            for _ in range(2):
                dau.append(np.array([rng.integers(len(O)) for O in bt.opts]))
            tot = min((dt.tkcb(E, a0, rng) for a0 in dau), key=lambda x: x[1])
            sol["thay"] = bt.tu_lua_chon(tot[0])
            row = {"kb": i, "bao": ten, "t": t, "muc": muc, "K": bt.K, "m": bt.m.tolist(),
                   "so_loai": [th["kho"][c]["so_loai"] for c in th["cap"]],
                   "min_cut": [th["kho"][c]["min_cut"] for c in th["cap"]],
                   "so_lua_chon": [len(O) for O in bt.opts],
                   "tg_ms": {"yen": tg_yen * 1e3, "dphat": float(np.median(tg_dp)) * 1e3, "online_tkcb": tg_on * 1e3},
                   "ens": {}, "qs": {}}
            for ten_pp, x in sol.items():
                for ten_tg, tgw in (("ens", E), ("qs", Q)):
                    c = tgw.cham(x)
                    row[ten_tg][ten_pp] = {k: c[k] for k in SO_DO}
            # kiểm TKCB = duyệt hết ở 3 luồng, tải nặng, mỗi 4 mốc
            if muc == "nang" and so % 4 == 0 and bt.K >= 3:
                bt3 = dt.BaiToan(th, muc, None, cac_luong=[0, 1, 2])
                if len(bt3.opts[0]) * len(bt3.opts[1]) <= MAX_DUYET:
                    E3 = dt.TheGioi.ensemble(bt3, g, pl, p, seeds)
                    _, Jx = dt.duyet_het(E3)
                    _, J3 = dt.tkcb(E3, dt.tham_lam(E3))
                    row["kiem_duyet_het"] = {"J_duyet_het": Jx, "J_tkcb": J3}
            hang.append(row)
    tam = str(tep) + ".tam"
    with open(tam, "wb") as f:
        pickle.dump(hang, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tam, tep)
    return tep, len(hang), time.time() - t_bd


def chay(tien_trinh):
    td._khoi_tao_nhe()
    moc = td.cac_moc("val")
    jobs = [(i, ten, h0, seed, moc, RA / f"kb_{i:04d}.pkl") for i, (ten, h0, seed) in enumerate(td.cac_cua_so("val"))]
    RA.mkdir(parents=True, exist_ok=True)
    con = [j for j in jobs if not Path(j[-1]).exists()]
    print(f"val: {len(jobs)} cửa sổ × {len(moc)} mốc, còn {len(con)}", flush=True)
    t0 = time.time()
    with Pool(tien_trinh, initializer=td._khoi_tao) as pool:
        for k, (tep, so, tgian) in enumerate(pool.imap_unordered(_mot_cua_so, con)):
            print(f"  {k + 1}/{len(con)}  {Path(tep).name}  {so} dòng  {tgian:.0f} s  ({time.time() - t0:.0f} s)", flush=True)
    (RA / ".xong").write_text(time.strftime("%Y-%m-%d %H:%M:%S"))


def bootstrap(v, kb, rng, B=10_000):
    u = np.unique(kb)
    tong = np.array([v[kb == k].sum() for k in u])
    dem = np.array([(kb == k).sum() for k in u])
    j = rng.integers(0, len(u), (B, len(u)))
    m = tong[j].sum(1) / dem[j].sum(1)
    return [float(v.mean()), [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]]


def tong_hop():
    hang = [r for f in sorted(glob.glob(str(RA / "kb_*.pkl"))) for r in pickle.load(open(f, "rb"))]
    rng = np.random.default_rng(0)
    cf_ten = [f"dphat_{b:g}_{c:g}" for b, c in CAU_HINH]
    J_cf = {c: float(np.mean([r["ens"][c]["J"] for r in hang])) for c in cf_ten}
    tot_cf = min(cf_ten, key=lambda c: J_cf[c])
    ra = {"so_dong": len(hang), "so_cua_so": len({r["kb"] for r in hang}),
          "chinh_beta_gamma": {"J_ens_tb_moi_cau_hinh": J_cf, "chon": tot_cf,
                               "quy_tac": "J ensemble trung bình nhỏ nhất trên mọi tình huống val (cả 3 mức tải)"}}
    PP = ["yen", "suurballe", tot_cf, "online_tkcb", "thay"]
    for muc in MUC_TAI:
        rs = [r for r in hang if r["muc"] == muc]
        kb = np.array([r["kb"] for r in rs])
        o = {"so_tinh_huong": len(rs), "phuong_phap": {}}
        for pp in PP:
            o["phuong_phap"][pp] = {f"{tw}_{k}": float(np.mean([r[tw][pp][k] for r in rs]))
                                    for tw in ("ens", "qs") for k in SO_DO}
        hieu = lambda k, a, b: np.array([r["ens"][a][k] - r["ens"][b][k] for r in rs])
        o["khoang_trong_thay_tru_dphat"] = {"mat_dv_diem_%": bootstrap(hieu("mat_dv_%", "thay", tot_cf), kb, rng),
                                            "du_phong_qua_tai_diem_%": bootstrap(hieu("du_phong_qua_tai_%", "thay", tot_cf), kb, rng),
                                            "J": bootstrap(hieu("J", "thay", tot_cf), kb, rng)}
        o["khoang_trong_thay_tru_online"] = {"mat_dv_diem_%": bootstrap(hieu("mat_dv_%", "thay", "online_tkcb"), kb, rng),
                                             "J": bootstrap(hieu("J", "thay", "online_tkcb"), kb, rng)}
        o["dphat_tru_suurballe_mat_dv_diem_%"] = bootstrap(hieu("mat_dv_%", tot_cf, "suurballe"), kb, rng)
        o["ti_le_thay_tot_hon_dphat_%"] = float(np.mean(hieu("J", "thay", tot_cf) < -1e-9) * 100)
        ra[muc] = o
    kt = [r["kiem_duyet_het"] for r in hang if "kiem_duyet_het" in r]
    ra["kiem_tkcb_duyet_het_3_luong"] = {"so_ca": len(kt), "trung_%": float(np.mean(
        [abs(k["J_tkcb"] - k["J_duyet_het"]) < 1e-9 for k in kt]) * 100) if kt else None,
        "lech_tb": float(np.mean([k["J_tkcb"] - k["J_duyet_het"] for k in kt])) if kt else None}
    ra["quy_mo"] = {"m_tb": float(np.mean([x for r in hang for x in r["m"]])),
                    "so_lua_chon_trung_vi": float(np.median([x for r in hang for x in r["so_lua_chon"]])),
                    "tg_trung_vi_ms": {k: float(np.median([r["tg_ms"][k] for r in hang])) for k in hang[0]["tg_ms"]}}
    KQ.mkdir(parents=True, exist_ok=True)
    json.dump(ra, open(KQ / "chay_thu_val.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    return ra


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("viec", choices=["chay", "tong-hop"])
    ap.add_argument("--tien-trinh", type=int, default=3)
    a = ap.parse_args()
    if a.viec == "chay":
        chay(a.tien_trinh)
    else:
        print(json.dumps(tong_hop(), ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
