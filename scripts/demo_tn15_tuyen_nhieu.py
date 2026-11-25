"""
demo_tn15_tuyen_nhieu.py — Như demo_tn15_tuyen.py nhưng ở NHIỀU thời điểm của mỗi kịch bản, cho thanh thời gian của bản 3D.

Mỗi kịch bản (cùng luật chọn với demo_tn15_tuyen): 12 mốc trải đều trong 48 mốc của cửa sổ + mốc của bản 2D. Ở mỗi mốc dựng
lại tình huống 4–5 luồng như bộ sinh dữ liệu, thay luồng 0 bằng từng cặp (điểm người dùng, trạm mặt đất), cho DM 7/3
(dm13_w0.7_s1) và DM x10 (dm13_x10_s1) chọn tuyến (giải mã không tái nhiễu 5 × 32), chấm trên 32 thế giới chấm.

    python -m scripts.demo_tn15_tuyen_nhieu [--tien-trinh 4] [--so-moc 12]   # ~10 phút với 4 tiến trình
Ra: results/bao_da_tuyen/demo_tn15_15b/tuyen_nhieu.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import zlib
from multiprocessing import Pool
from pathlib import Path

import numpy as np

GOC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(GOC))
from scripts import demo_tn15_tuyen as d1  # noqa: E402

RA = d1.RA
_CACHE = {}


def _khoi_tao():
    import torch

    from src.nhieu_luong import mo_hinh_dt as md
    from src.nhieu_luong import tao_du_lieu_rr as td
    torch.set_num_threads(1)
    td._khoi_tao()
    _CACHE["net"] = {k: md.nap(ten, d) for k, (d, ten) in d1.MO_HINH.items()}


def _kich_ban(tap, kb):
    from src.ngau_nhien.kich_ban import sinh_kich_ban
    from src.nhieu_luong import tao_du_lieu_dt as tdt
    from src.nhieu_luong import tao_du_lieu_rr as td
    from src.nhieu_luong import the_gioi_rr as tg
    if (tap, kb) not in _CACHE:
        ten, h0, seed = tdt.cac_cua_so_tap(tap)[kb]
        tt, p_sim = tdt.moi_truong(tap, kb)
        Wn, q = sinh_kich_ban(td._g, tt, h0, seed, p_sim)
        pl = tg.phat_lai(td._g, tt, h0, seed, p_sim)
        _CACHE[(tap, kb)] = (ten, h0, seed, tt, p_sim, Wn, q, pl)
    return _CACHE[(tap, kb)]


def mot_moc(job):
    import torch

    from src.ngau_nhien.ma_tran_rui_ro import ma_tran_online
    from src.nhieu_luong import da_tuyen as dt
    from src.nhieu_luong import mo_hinh_dt as md
    from src.nhieu_luong import tao_du_lieu_dt as tdt
    from src.nhieu_luong import tao_du_lieu_rr as td
    from src.nhieu_luong.bai_toan_rr import DELTA, TUOI
    i_kb, tap, kb, t = job
    g, p = td._g, td._p
    ten, h0, seed, tt, p_sim, Wn, q, pl = _kich_ban(tap, kb)
    o = t - TUOI
    past = {k: q[k][:o + 1] for k in ("R", "gust", "giong", "tat")}
    r = ma_tran_online(td._bang, g, Wn[o], past, t, TUOI, DELTA, tham_so=p)
    th0 = dt.tao_tinh_huong(g, q, r, t, np.random.default_rng([seed, 5]), so_luong=(4, 5), so_nguon=3)
    ra = {"t": t, "gio_utc": str(tt.time_utc[h0 + int(o * 20 / 3600)])[:16], "cap": {}}
    if th0 is None:
        ra["luong_goc"] = []
        return i_kb, ra
    ra["luong_goc"] = [[g.names[int(s)], g.names[int(d)]] for s, d in zip(th0["src"], th0["dst"])]
    mg = th0["mg"]
    cape = tdt.cape_luc(h0, o, tdt.tep_thoi_tiet(tap))
    sc = tdt._seeds(seed, t, 1, 32)
    nguoi = [n for n in range(g.N) if g.kind[n] == "user"]
    tram = [n for n in range(g.N) if g.kind[n] == "gateway"]
    for s in nguoi:
        for d in tram:
            kt = dt.kho_tuyen(mg, s, d)
            khoa = f"{g.names[s]}|{g.names[d]}"
            if kt is None:
                continue
            th = dict(th0)
            th["kho"] = list(th0["kho"]) + [kt]
            th["src"], th["dst"], th["cap"] = th0["src"].copy(), th0["dst"].copy(), th0["cap"].copy()
            th["src"][0], th["dst"][0], th["cap"][0] = s, d, len(th["kho"]) - 1
            F0 = np.zeros(len(mg.eu))
            for k, c in enumerate(th["cap"]):
                nh = min(th["kho"][c]["kho"], key=lambda rr: float(mg.d0[mg.dt.canh_cua(rr)].sum()))
                F0[mg.dt.canh_cua(nh)] += th["lam"][k]
            th["u"] = float(np.max(F0[F0 > 0] / mg.trong[F0 > 0]))
            bt = dt.BaiToan(th, d1.MUC)
            gon = dt.BaiToanGon(bt)
            gon.dac_trung_canh = tdt.dac_trung_canh(g, q, mg, gon.U, o, cape)
            gon.dac_trung_uv = [tdt.dac_trung_uv(g, th, k) for k in range(bt.K)]
            Ec = dt.TheGioi.ensemble(bt, g, pl, p_sim, sc)
            x = {"kho": [[int(n) for n in rr] for rr in bt.bang[0][:bt.so_kho[0]]],
                 "tre_kho": [round(float(mg.d0[mg.dt.canh_cua(rr)].sum()), 2) for rr in bt.bang[0][:bt.so_kho[0]]]}
            for k_mh, n_mh in _CACHE["net"].items():
                a = md.chon_sua_dan(n_mh, gon, 32, 5, torch.Generator().manual_seed(zlib.crc32(f"{khoa}|{t}".encode())))
                sol = bt.tu_lua_chon(a)
                c = Ec.cham(sol)
                x[k_mh] = {"tuyen": [int(i) for i in sol[0]], "mat_luong": round(float(c["H"][0]) * 100, 2),
                           "tre_luong": round(float(c["d"][0]), 2), "mat_tong": round(float(c["mat_dv_%"]), 2),
                           "tre_tong": round(float(c["tre_chinh_ms"]), 2)}
            ra["cap"][khoa] = x
    return i_kb, ra


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tien-trinh", type=int, default=4)
    ap.add_argument("--so-moc", type=int, default=12)
    a = ap.parse_args()
    from src.nhieu_luong import da_tuyen as dt
    from src.nhieu_luong import tao_du_lieu_dt as tdt
    from src.nhieu_luong import tao_du_lieu_rr as td
    from src.nhieu_luong.bai_toan_rr import TUOI
    td._khoi_tao()
    g = td._g
    moc = td.cac_moc("test")
    ds, _ = d1.chon_cua_so()
    kbs, viec = [], []
    for i, (nhan, tap, kb) in enumerate(ds):
        ten, h0, seed, *_rest = _kich_ban(tap, kb)
        q = _rest[3]
        diem = [float(dt.diem_bao(g, q, int(t) - TUOI).sum()) for t in moc]
        t_goc = int(moc[int(np.argmax(diem))])
        chon = sorted({int(moc[j]) for j in np.linspace(0, len(moc) - 1, a.so_moc).round().astype(int)} | {t_goc})
        kbs.append({"nhan": nhan, "cua_so": ten, "tap": tap, "kb": kb, "moc_goc": t_goc, "thoi_diem": [None] * len(chon)})
        viec += [(i, tap, kb, t) for t in chon]
    with np.load(td.RAW, allow_pickle=False) as z:
        lla = z["node_latlon_alt"]
    s0 = time.time()
    vi_tri_moc = {}
    with Pool(a.tien_trinh, initializer=_khoi_tao) as pool:
        for n, (i_kb, ra) in enumerate(pool.imap_unordered(mot_moc, viec)):
            k = kbs[i_kb]
            ts = sorted(j[3] for j in viec if j[0] == i_kb)
            ra["nut"] = [[round(float(lla[ra["t"], v, 0]), 3), round(float(lla[ra["t"], v, 1]), 3), round(float(lla[ra["t"], v, 2]), 1)]
                         for v in range(g.N)]
            k["thoi_diem"][ts.index(ra["t"])] = ra
            print(f"  {n + 1}/{len(viec)}  {k['nhan']} mốc {ra['t']}: {len(ra['cap'])} cặp tới được ({time.time() - s0:.0f} s)", flush=True)
    LOAI = {"leo": 0, "haps": 1, "uav": 2, "gateway": 3, "user": 4}
    out = {"mo_hinh": {k: v[1] for k, v in d1.MO_HINH.items()}, "muc_tai": d1.MUC, "ten_nut": list(g.names),
           "loai": [LOAI[k] for k in g.kind], "kich_ban": kbs}
    RA.mkdir(parents=True, exist_ok=True)
    (RA / "tuyen_nhieu.json").write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"xong {RA / 'tuyen_nhieu.json'} ({(RA / 'tuyen_nhieu.json').stat().st_size / 1e6:.1f} MB)", flush=True)


if __name__ == "__main__":
    main()
