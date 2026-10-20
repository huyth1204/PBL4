"""
danh_gia_kgh.py — Chọn cấu hình trên val và chấm tập Yagi cho thí nghiệm 2 (results/yagi2/TIEU_CHI_2.md).

    python -m src.nhieu_luong.danh_gia_kgh chon                    # chọn (S, số bước, γ) cho DM và γ cho Một-lần trên val
    python -m src.nhieu_luong.danh_gia_kgh test --phan 0 --so-phan 2

Thời gian đo trong CÙNG tiến trình, 1 luồng CPU, lần lượt từng phương pháp trên từng tình huống. Thời gian của DM và
Một-lần tính cả bước chuẩn bị (tìm cặp nguồn – đích, tính hành lang).
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
from src.nhieu_luong.huan_luyen_kgh import MUC, nap
from src.nhieu_luong.mo_hinh_kgh import Lo2, MangKGH, TinhHuong2, _day_du, ban_do_dm, ban_do_mot_lan, giai

GOC = Path(__file__).resolve().parents[2]
KQ = GOC / "results" / "yagi2"
S_LUOI, BUOC_LUOI, GAMMA_LUOI = (1, 2, 4), (2, 3, 5), (0.3, 1.0, 3.0)


def nap_mo_hinh(kieu):
    ck = torch.load(KQ / "mo_hinh" / f"{kieu}.pt", weights_only=False)
    net = MangKGH(**ck["cau_hinh"])
    net.load_state_dict(ck["state"])
    return net.eval()


# ---------------------------------------------------------------- chọn cấu hình trên val
def _thu_dm(net, th, S, n, gen):
    t0 = time.perf_counter()
    th2 = TinhHuong2(th.x, th.muc)
    lo = Lo2([th2], lap=S)
    ys = [_day_du(th2, v.numpy()) for v in lo.tach(ban_do_dm(net, lo, n, gen))]
    t_mang = time.perf_counter() - t0
    gia, tg = {}, {}
    for g in GAMMA_LUOI:
        cs, ts = [], []
        for y in ys:
            s = time.perf_counter()
            cs.append(th2.md.chi_phi(kg.dung_tuyen(th2.md, yhat=y, cap=th2.cua, gamma=g)))
            ts.append(time.perf_counter() - s)
        gia[g], tg[g] = min(cs), t_mang + sum(ts)
    return gia, tg


def chon(phan=0, so_phan=1):
    torch.set_num_threads(1)
    va = nap("val")
    # thời gian mốc TKCB-KGH 10 lần trên 300 tình huống val (10 điểm xuất phát, không có lời giải trên ứng viên)
    f_moc = KQ / "chon_moc_val.json"
    if phan == 0 and not f_moc.exists():
        tg = []
        for i in np.random.default_rng(99).choice(len(va), 300, replace=False):
            s = time.perf_counter()
            kg.tkcb_kgh_10(va[i].md, None, seed=int(va[i].x["t"]))
            tg.append(time.perf_counter() - s)
        json.dump({"trung_vi_ms": float(np.median(tg) * 1e3), "so_tinh_huong": 300}, open(f_moc, "w"), indent=1)
        print("mốc val", json.load(open(f_moc)), flush=True)
    net_dm, net_ml = nap_mo_hinh("dm"), nap_mo_hinh("mot_lan")
    cau = [(S, n) for S in S_LUOI for n in BUOC_LUOI]
    for j, (S, n) in enumerate(cau):
        f = KQ / f"chon_dm_S{S}_b{n}.json"
        if j % so_phan != phan or f.exists():
            continue
        gen = torch.Generator().manual_seed(0)
        gap = {g: [] for g in GAMMA_LUOI}
        tgs = {g: [] for g in GAMMA_LUOI}
        for th in va:
            gia, tg = _thu_dm(net_dm, th, S, n, gen)
            for g in GAMMA_LUOI:
                gap[g].append(gia[g] / th.chi_phi_nhan - 1)
                tgs[g].append(tg[g])
        json.dump({str(g): {"gap_tb_%": float(np.mean(gap[g]) * 100), "tg_trung_vi_ms": float(np.median(tgs[g]) * 1e3)}
                   for g in GAMMA_LUOI}, open(f, "w"), indent=1)
        print(S, n, json.load(open(f)), flush=True)
    # Một-lần: S lần dựng tuyến với S thứ tự luồng; tính sẵn cho mọi S, γ
    f = KQ / "chon_mot_lan.json"
    if phan == so_phan - 1 and not f.exists():
        gap = {(S, g): [] for S in S_LUOI for g in GAMMA_LUOI}
        for th in va:
            rng = np.random.default_rng(int(th.x["t"]))
            th2 = TinhHuong2(th.x, th.muc)
            y = _day_du(th2, ban_do_mot_lan(net_ml, Lo2([th2])).numpy())
            thu_tu = [None] + [rng.permutation(th2.md.K) for _ in range(max(S_LUOI) - 1)]
            for g in GAMMA_LUOI:
                cs = [th2.md.chi_phi(kg.dung_tuyen(th2.md, thu_tu=tt, yhat=y, cap=th2.cua, gamma=g)) for tt in thu_tu]
                for S in S_LUOI:
                    gap[(S, g)].append(min(cs[:S]) / th.chi_phi_nhan - 1)
        json.dump({f"S{S}_g{g}": float(np.mean(v) * 100) for (S, g), v in gap.items()}, open(f, "w"), indent=1)
        print("một-lần", json.load(open(f)), flush=True)
    tong_hop()


def tong_hop():
    """Khi đủ kết quả: áp quy tắc chọn đã chốt và ghi results/yagi2/chon_cau_hinh.json."""
    f_moc = KQ / "chon_moc_val.json"
    tep = {(S, n): KQ / f"chon_dm_S{S}_b{n}.json" for S in S_LUOI for n in BUOC_LUOI}
    if not f_moc.exists() or not (KQ / "chon_mot_lan.json").exists() or not all(f.exists() for f in tep.values()):
        return None
    han = json.load(open(f_moc))["trung_vi_ms"] / 10
    bang = [(S, n, float(g), v["gap_tb_%"], v["tg_trung_vi_ms"]) for (S, n), f in tep.items()
            for g, v in json.load(open(f)).items()]
    dat = [b for b in bang if b[4] <= han]
    tot = min(dat or bang, key=lambda b: b[3])
    ml = json.load(open(KQ / "chon_mot_lan.json"))
    g_ml = min(GAMMA_LUOI, key=lambda g: ml[f"S{tot[0]}_g{g}"])
    ra = {"DM": {"S": tot[0], "so_buoc": tot[1], "gamma": tot[2], "gap_val_%": tot[3], "tg_val_ms": tot[4]},
          "mot_lan": {"S": tot[0], "gamma": g_ml, "gap_val_%": ml[f"S{tot[0]}_g{g_ml}"]},
          "han_thoi_gian_val_ms": han, "co_cau_hinh_dat_han": bool(dat), "bang": bang}
    json.dump(ra, open(KQ / "chon_cau_hinh.json", "w"), indent=1)
    print("chọn:", ra["DM"], ra["mot_lan"], "hạn", round(han, 2), "ms", flush=True)
    return ra


# ---------------------------------------------------------------- chấm tập Yagi
def tuyen_ung_vien(x1, md):
    """Lời giải TKCB 10 lần trên 8 ứng viên (dữ liệu thí nghiệm 1) → danh sách chỉ số cạnh của MangDay."""
    out = []
    for k, c in enumerate(x1["nhan"]):
        l = x1["canh"][k, int(c)]
        g = x1["e"][l[l >= 0]].astype(np.int64)
        out.append(md.eid[g // md.N, g % md.N].astype(np.int64))
    return out


def cham(ten_k, cfg, net_dm, net_ml, phan=0, so_phan=1):
    tep = sorted(glob.glob(str(GOC / "data" / "yagi2" / "test" / ten_k / "kb_*.pkl")))
    thu_muc = KQ / "test_tho" / ten_k
    thu_muc.mkdir(parents=True, exist_ok=True)
    for j, f in enumerate(tep):
        ra = thu_muc / (Path(f).stem + ".npz")
        if j % so_phan != phan or ra.exists():
            continue
        cu = {(x["t"], x["muc"]): x for x in pickle.load(open(GOC / "data" / "yagi" / "test" / ten_k / Path(f).name, "rb"))}
        cot = {}
        t_bd = time.time()
        for x in pickle.load(open(f, "rb")):
            for muc in MUC:
                gen = torch.Generator().manual_seed(1000 * x["kb"] + x["t"])
                hang = {"kb": x["kb"], "t": x["t"], "muc": muc, "K": len(x["lam"][muc])}
                # DM
                s = time.perf_counter()
                th = TinhHuong2(x, muc)
                r_dm, c_dm, _ = giai(net_dm, th, "dm", S=cfg["DM"]["S"], so_buoc=cfg["DM"]["so_buoc"],
                                     gamma=cfg["DM"]["gamma"], gen=gen)
                hang["tg_DM"], hang["DM"] = time.perf_counter() - s, c_dm
                md = th.md
                hang["qua_tai_DM"] = md.qua_tai(md.tai(r_dm))
                hang["C"] = len(th.cap)
                # DM + TKCB-KGH (báo thêm)
                s = time.perf_counter()
                hang["DM_LSK"] = md.chi_phi(kg.tkcb_kgh(md, r_dm))
                hang["tg_DM_LSK"] = hang["tg_DM"] + time.perf_counter() - s
                # Một-lần
                s = time.perf_counter()
                th = TinhHuong2(x, muc)
                _, c_ml, _ = giai(net_ml, th, "mot_lan", S=cfg["mot_lan"]["S"], gamma=cfg["mot_lan"]["gamma"],
                                  rng=np.random.default_rng(int(x["t"])))
                hang["tg_ML"], hang["ML"] = time.perf_counter() - s, c_ml
                # Dijkstra tuần tự, TKCB-KGH 1 lần, 10 lần (cùng tiến trình)
                md = kg.MangDay.nap({**x["mang"], "lam": x["lam"][muc]})
                s = time.perf_counter()
                hang["tuan_tu"] = md.chi_phi(kg.tuan_tu_kgh(md))
                hang["tg_tuan_tu"] = time.perf_counter() - s
                s = time.perf_counter()
                hang["LSK1"] = md.chi_phi(kg.tkcb_kgh(md, kg.tuan_tu_kgh(md)))
                hang["tg_LSK1"] = time.perf_counter() - s
                uv = tuyen_ung_vien(cu[(x["t"], muc)], md)
                s = time.perf_counter()
                _, hang["LSK10"], _ = kg.tkcb_kgh_10(md, uv, seed=int(x["t"]))
                hang["tg_LSK10"] = time.perf_counter() - s
                d = x["doi_chung"][muc]
                hang.update({"LSK10_luc_sinh": d["LSK10"], "LS10_uv": d["LS10_uv"], "doc_lap": d["doc_lap"],
                             "FW": d["FW_can_duoi"], "qua_tai_LSK10": d["qua_tai_LSK10"],
                             "qua_tai_tuan_tu": d["qua_tai_tuan_tu"]})
                for k, v in hang.items():
                    cot.setdefault(k, []).append(v)
        np.savez(ra, **{k: np.array(v) for k, v in cot.items()})
        print(f"  {ten_k} {Path(f).stem}: {len(cot['kb'])} tình huống, DM tb {np.mean(cot['tg_DM']) * 1e3:.1f} ms, "
              f"LSK10 tb {np.mean(cot['tg_LSK10']) * 1e3:.0f} ms ({time.time() - t_bd:.0f} s)", flush=True)


def gop(ten_k):
    n = len(glob.glob(str(GOC / "data" / "yagi2" / "test" / ten_k / "kb_*.pkl")))
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
    ap.add_argument("--k", default="k20_40,k80,k160")
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
    if all(gop(k) for k in ("k20_40", "k80", "k160")):
        (KQ / "test.xong").write_text(time.strftime("%Y-%m-%d %H:%M:%S"))


if __name__ == "__main__":
    main()
