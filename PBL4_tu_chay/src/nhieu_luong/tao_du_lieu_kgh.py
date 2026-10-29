"""
tao_du_lieu_kgh.py — Dữ liệu thí nghiệm 2 (không giới hạn tuyến; results/yagi2/TIEU_CHI_2.md).

    python -m src.nhieu_luong.tao_du_lieu_kgh --tap val | train | test

Dùng ĐÚNG các kịch bản, snapshot và hạt giống của thí nghiệm 1 (tao_du_lieu_nl.py) nên các luồng giống hệt.
Mỗi snapshot lưu: trạng thái mạng (danh sách cạnh còn dùng được), các luồng, và với từng mức tải: λ, nhãn = tuyến
của TKCB-KGH 10 lần (mốc). Tập test lưu thêm chi phí của các đối chứng.
Một kịch bản = một tệp data/yagi2/<tập>/kb_XXXX.pkl (chạy lại sẽ bỏ qua tệp đã có).
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from src.ngau_nhien.kich_ban import ThamSo, sinh_kich_ban
from src.ngau_nhien.khoang_loi import MUC
from src.nhieu_luong import doi_chung as dc
from src.nhieu_luong import khong_gioi_han as kg
from src.nhieu_luong import tao_du_lieu_nl as nl
from src.nhieu_luong.bai_toan import tao_tinh_huong

RA = nl.GOC / "data" / "yagi2"
RA3 = nl.GOC / "data" / "yagi3"                               # thí nghiệm 3: 500–2.000 luồng
K_LON = {"k500": 500, "k1000": 1000, "k2000": 2000}


def cac_viec_lon(tap):
    """Thí nghiệm 3 (results/yagi3/TIEU_CHI_3.md). Lưu đủ đối chứng và thời gian lúc sinh cho cả val và test."""
    nl._khoi_tao()
    jobs = []
    if tap == "test_lon":
        for ten_k, K in K_LON.items():
            for i, h0 in enumerate(nl.gio_test()):
                jobs.append(("test", i, h0, 5000 + i, np.arange(0, 600, 60), (K, K), RA3 / "test" / ten_k / f"kb_{i:04d}.pkl"))
    else:                                                       # val_lon: 60 kịch bản val, 2 ảnh chụp, K = 1.000
        for i, (h0, seed) in enumerate(nl.chon_gio(nl._tt, "val")):
            off = int(np.random.default_rng(seed).integers(0, 40))
            jobs.append(("test", i, h0, seed, off + 40 * np.array([0, 7]), (1000, 1000), RA3 / "val_k1000" / f"kb_{i:04d}.pkl"))
    return jobs


def _mot_kich_ban(job):
    tap, i, h0, seed, snaps, so_luong, tep = job
    if nl._g is None:
        nl._khoi_tao()
    if Path(tep).exists():
        return tep, 0, 0.0
    t_bd = time.time()
    _, q = sinh_kich_ban(nl._g, nl._tt, h0, seed, ThamSo(**MUC["thường"]))
    rng = np.random.default_rng(seed + 7)                      # giống thí nghiệm 1 → cùng luồng
    cache, ra = {}, []
    kieu = np.float64 if tap == "test" else np.float32
    for t in snaps:
        bo = tao_tinh_huong(nl._g, q, int(t), rng, so_luong=so_luong, cache=cache)
        if bo is None:
            continue
        x = None
        for ten, th in bo.items():
            a_uv, c_uv = dc.ls_nhieu_lan(th, R=10, seed=int(t))                 # mốc của thí nghiệm 1
            md = kg.MangDay.tu_tinh_huong(th)
            if x is None:
                mg = md.luu()
                for k in ("prop", "C", "rho0"):
                    mg[k] = mg[k].astype(kieu)
                mg.pop("lam")
                x = {"mang": mg, "lam": {}, "nhan": {}, "chi_phi_nhan": {}, "LS10_uv": {}, "h0": int(h0),
                     "gio": nl._T[h0], "seed_kb": int(seed), "kb": int(i), "t": int(t), "meta": dict(th.meta),
                     "doi_chung": {}, "thoi_gian_luc_sinh": {}}
            tuyen_uv = [md.eid[np.array(th.ung_vien[k][a_uv[k]][:-1]), np.array(th.ung_vien[k][a_uv[k]][1:])]
                        for k in range(th.K)]
            s = time.perf_counter()
            tot, c10, cac = kg.tkcb_kgh_10(md, tuyen_uv, seed=int(t))
            tg10 = time.perf_counter() - s
            x["lam"][ten] = md.lam.copy()
            x["nhan"][ten] = [e.astype(np.int16) for e in tot]
            x["chi_phi_nhan"][ten] = float(c10)
            x["LS10_uv"][ten] = float(c_uv)
            if tap == "test":
                kq, tg = {"LSK10": c10, "LSK10_tung_lan": cac, "LS10_uv": c_uv}, {"LSK10": tg10}
                kq["doc_lap"] = md.chi_phi(kg.doc_lap_kgh(md))
                s = time.perf_counter()
                r_tt = kg.tuan_tu_kgh(md)
                tg["tuan_tu"] = time.perf_counter() - s
                kq["tuan_tu"] = md.chi_phi(r_tt)
                s = time.perf_counter()
                r1 = kg.tkcb_kgh(md, kg.tuan_tu_kgh(md))
                tg["LSK1"] = time.perf_counter() - s
                kq["LSK1"] = md.chi_phi(r1)
                kq["qua_tai_LSK10"] = md.qua_tai(md.tai(tot))
                kq["qua_tai_tuan_tu"] = md.qua_tai(md.tai(r_tt))
                kq["FW_can_duoi"], kq["FW_chia_luong"] = (float(v) for v in dc.frank_wolfe(th))
                x["doi_chung"][ten], x["thoi_gian_luc_sinh"][ten] = kq, tg
        ra.append(x)
    tam = str(tep) + ".tam"
    with open(tam, "wb") as f:
        pickle.dump(ra, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tam, tep)
    return tep, len(ra), time.time() - t_bd


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--tap", choices=["train", "val", "test", "val_lon", "test_lon"], required=True)
    ap.add_argument("--tien-trinh", type=int, default=2)
    a = ap.parse_args()
    jobs = cac_viec_lon(a.tap) if a.tap.endswith("_lon") else \
        [(*j[:-1], RA / Path(j[-1]).relative_to(nl.RA)) for j in nl.cac_viec(a.tap)]
    for j in jobs:
        Path(j[-1]).parent.mkdir(parents=True, exist_ok=True)
    con = [j for j in jobs if not Path(j[-1]).exists()]
    print(f"{a.tap}: {len(jobs)} kịch bản, còn {len(con)}", flush=True)
    t0, n = time.time(), 0
    with Pool(a.tien_trinh, initializer=nl._khoi_tao) as p:
        for k, (tep, so, tg) in enumerate(p.imap_unordered(_mot_kich_ban, con)):
            n += so
            if k % 10 == 0 or k == len(con) - 1:
                print(f"  {k + 1}/{len(con)}  {so} snapshot  {tg:.1f} s  (tổng {n}, {time.time() - t0:.0f} s)", flush=True)
    for d in {Path(j[-1]).parent for j in jobs}:
        (d / ".xong").write_text(time.strftime("%Y-%m-%d %H:%M:%S"))
    print("xong", a.tap, flush=True)


if __name__ == "__main__":
    main()
