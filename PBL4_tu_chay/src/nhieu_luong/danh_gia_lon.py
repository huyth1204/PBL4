"""
danh_gia_lon.py — Thí nghiệm 3: 500–2.000 luồng (results/yagi3/TIEU_CHI_3.md).

    python -m src.nhieu_luong.danh_gia_lon chon --phan 0 --so-phan 2      # chọn cấu hình trên val, K = 1.000
    python -m src.nhieu_luong.danh_gia_lon test --phan 0 --so-phan 2      # chấm Yagi, K = 500 / 1.000 / 2.000

Dùng nguyên mô hình DM và Một-lần của thí nghiệm 2. Thời gian mốc TKCB-KGH 10 lần lấy từ lúc sinh dữ liệu; mọi phương
pháp khác đo lại ở đây (1 luồng CPU).
"""

from __future__ import annotations

import argparse
import glob
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch

from src.nhieu_luong import khong_gioi_han as kg
from src.nhieu_luong.danh_gia_kgh import BUOC_LUOI, CACH_LUOI, GIAI_MA, S_LUOI, nap_mo_hinh
from src.nhieu_luong.mo_hinh_kgh import Lo2, TinhHuong2, _day_du, ban_do_dm, ban_do_mot_lan, giai

GOC = Path(__file__).resolve().parents[2]
DL = GOC / "data" / "yagi3"
KQ = GOC / "results" / "yagi3"
MUC = ("nhe", "vua", "nang")


def nap_val():
    ds = []
    for f in sorted(glob.glob(str(DL / "val_k1000" / "kb_*.pkl"))):
        for x in pickle.load(open(f, "rb")):
            ds += [(x, m) for m in MUC]
    return ds


def chon(phan=0, so_phan=1):
    torch.set_num_threads(1)
    va = nap_val()
    han = float(np.median([x["thoi_gian_luc_sinh"][m]["LSK10"] for x, m in va]) * 1e3) / 10
    net_dm, net_ml = nap_mo_hinh("dm"), nap_mo_hinh("mot_lan")
    cau = [(S, n) for S in S_LUOI for n in BUOC_LUOI]
    KQ.mkdir(parents=True, exist_ok=True)
    for j, (S, n) in enumerate(cau):
        f = KQ / f"chon_dm_S{S}_b{n}.json"
        if j % so_phan != phan or f.exists():
            continue
        gen = torch.Generator().manual_seed(0)
        gap = {g: [] for g in CACH_LUOI}
        tgs = {g: [] for g in CACH_LUOI}
        for x, m in va:
            t0 = time.perf_counter()
            th = TinhHuong2(x, m)
            lo = Lo2([th], lap=S)
            ys = [_day_du(th, v.numpy()) for v in lo.tach(ban_do_dm(net_dm, lo, n, gen))]
            t_mang = time.perf_counter() - t0
            for g in CACH_LUOI:
                cs, ts = [], []
                for y in ys:
                    s = time.perf_counter()
                    cs.append(th.md.chi_phi(GIAI_MA[g](th.md, y, th.cap, th.cua, th.mask)))
                    ts.append(time.perf_counter() - s)
                gap[g].append(min(cs) / x["chi_phi_nhan"][m] - 1)
                tgs[g].append(t_mang + sum(ts))
        json.dump({g: {"gap_tb_%": float(np.mean(gap[g]) * 100), "tg_trung_vi_ms": float(np.median(tgs[g]) * 1e3)}
                   for g in CACH_LUOI}, open(f, "w"), indent=1)
        print(S, n, json.load(open(f)), flush=True)
    f = KQ / "chon_mot_lan.json"
    if phan == so_phan - 1 and not f.exists():
        gap = {(S, g): [] for S in S_LUOI for g in CACH_LUOI}
        for x, m in va:
            rng = np.random.default_rng(int(x["t"]))
            th = TinhHuong2(x, m)
            y = _day_du(th, ban_do_mot_lan(net_ml, Lo2([th])).numpy())
            thu_tu = [None] + [rng.permutation(th.md.K) for _ in range(max(S_LUOI) - 1)]
            for g in CACH_LUOI:
                cs = [th.md.chi_phi(GIAI_MA[g](th.md, y, th.cap, th.cua, th.mask, thu_tu=tt)) for tt in thu_tu]
                for S in S_LUOI:
                    gap[(S, g)].append(min(cs[:S]) / x["chi_phi_nhan"][m] - 1)
        json.dump({f"S{S}_g{g}": float(np.mean(v) * 100) for (S, g), v in gap.items()}, open(f, "w"), indent=1)
        print("một-lần", json.load(open(f)), flush=True)
    tep = {(S, n): KQ / f"chon_dm_S{S}_b{n}.json" for S, n in cau}
    if all(f.exists() for f in tep.values()) and (KQ / "chon_mot_lan.json").exists():
        bang = [(S, n, g, v["gap_tb_%"], v["tg_trung_vi_ms"]) for (S, n), f in tep.items() for g, v in json.load(open(f)).items()]
        dat = [b for b in bang if b[4] <= han]
        tot = min(dat or bang, key=lambda b: b[3])
        ml = json.load(open(KQ / "chon_mot_lan.json"))
        g_ml = min(CACH_LUOI, key=lambda g: ml[f"S{tot[0]}_g{g}"])
        ra = {"DM": {"S": tot[0], "so_buoc": tot[1], "cach": tot[2], "gap_val_%": tot[3], "tg_val_ms": tot[4]},
              "mot_lan": {"S": tot[0], "cach": g_ml, "gap_val_%": ml[f"S{tot[0]}_g{g_ml}"]},
              "han_thoi_gian_val_ms": han, "co_cau_hinh_dat_han": bool(dat), "so_tinh_huong_val": len(va), "bang": bang}
        json.dump(ra, open(KQ / "chon_cau_hinh.json", "w"), indent=1)
        print("chọn:", ra["DM"], ra["mot_lan"], "hạn", round(han, 1), "ms", flush=True)


def cham(ten_k, cfg, net_dm, net_ml, phan=0, so_phan=1):
    tep = sorted(glob.glob(str(DL / "test" / ten_k / "kb_*.pkl")))
    thu_muc = KQ / "test_tho" / ten_k
    thu_muc.mkdir(parents=True, exist_ok=True)
    for j, f in enumerate(tep):
        ra = thu_muc / (Path(f).stem + ".npz")
        if j % so_phan != phan or ra.exists():
            continue
        cot = {}
        t_bd = time.time()
        for x in pickle.load(open(f, "rb")):
            for muc in MUC:
                gen = torch.Generator().manual_seed(1000 * x["kb"] + x["t"])
                hang = {"kb": x["kb"], "t": x["t"], "muc": muc, "K": len(x["lam"][muc])}
                s = time.perf_counter()
                th = TinhHuong2(x, muc)
                r_dm, c_dm, _ = giai(net_dm, th, "dm", S=cfg["DM"]["S"], so_buoc=cfg["DM"]["so_buoc"],
                                     cach=cfg["DM"]["cach"], gen=gen)
                hang["tg_DM"], hang["DM"] = time.perf_counter() - s, c_dm
                md = th.md
                hang["qua_tai_DM"], hang["C"] = md.qua_tai(md.tai(r_dm)), len(th.cap)
                s = time.perf_counter()
                hang["DM_LSK"] = md.chi_phi(kg.tkcb_kgh(md, r_dm))
                hang["tg_DM_LSK"] = hang["tg_DM"] + time.perf_counter() - s
                s = time.perf_counter()
                th = TinhHuong2(x, muc)
                _, c_ml, _ = giai(net_ml, th, "mot_lan", S=cfg["mot_lan"]["S"], cach=cfg["mot_lan"]["cach"],
                                  rng=np.random.default_rng(int(x["t"])))
                hang["tg_ML"], hang["ML"] = time.perf_counter() - s, c_ml
                md = kg.MangDay.nap({**x["mang"], "lam": x["lam"][muc]})
                s = time.perf_counter()
                hang["tuan_tu"] = md.chi_phi(kg.tuan_tu_kgh(md))
                hang["tg_tuan_tu"] = time.perf_counter() - s
                s = time.perf_counter()
                hang["LSK1"] = md.chi_phi(kg.tkcb_kgh(md, kg.tuan_tu_kgh(md)))
                hang["tg_LSK1"] = time.perf_counter() - s
                d = x["doi_chung"][muc]
                hang.update({"LSK10": d["LSK10"], "tg_LSK10": x["thoi_gian_luc_sinh"][muc]["LSK10"], "LSK10_luc_sinh": d["LSK10"],
                             "LS10_uv": d["LS10_uv"], "doc_lap": d["doc_lap"], "FW": d["FW_can_duoi"],
                             "qua_tai_LSK10": d["qua_tai_LSK10"], "qua_tai_tuan_tu": d["qua_tai_tuan_tu"]})
                for k, v in hang.items():
                    cot.setdefault(k, []).append(v)
        np.savez(ra, **{k: np.array(v) for k, v in cot.items()})
        print(f"  {ten_k} {Path(f).stem}: {len(cot['kb'])} tình huống, DM tb {np.mean(cot['tg_DM']) * 1e3:.0f} ms, "
              f"tuần tự tb {np.mean(cot['tg_tuan_tu']) * 1e3:.0f} ms ({time.time() - t_bd:.0f} s)", flush=True)


def gop(ten_k):
    n = len(glob.glob(str(DL / "test" / ten_k / "kb_*.pkl")))
    tep = sorted(glob.glob(str(KQ / "test_tho" / ten_k / "kb_*.npz")))
    if len(tep) < n:
        return False
    ds = [dict(np.load(f)) for f in tep]
    np.savez(KQ / f"test_{ten_k}.npz", **{k: np.concatenate([d[k] for d in ds]) for k in ds[0]})
    return True


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("lenh", choices=["chon", "test"])
    ap.add_argument("--k", default="k500,k1000,k2000")
    ap.add_argument("--phan", type=int, default=0)
    ap.add_argument("--so-phan", type=int, default=1)
    a = ap.parse_args()
    if a.lenh == "chon":
        return chon(a.phan, a.so_phan)
    torch.set_num_threads(1)
    cfg = json.load(open(KQ / "chon_cau_hinh.json"))
    net_dm, net_ml = nap_mo_hinh("dm"), nap_mo_hinh("mot_lan")
    for ten_k in a.k.split(","):
        cham(ten_k, cfg, net_dm, net_ml, a.phan, a.so_phan)
        gop(ten_k)


if __name__ == "__main__":
    main()
