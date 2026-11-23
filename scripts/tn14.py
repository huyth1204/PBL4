"""
tn14.py — Thí nghiệm 14 (results/bao_da_tuyen/TIEU_CHI_14.md): khắc phục ba nhược điểm của DM11 nêu ở TN11 —
(1) phần "không còn tuyến sống" cao hơn Dijkstra phạt ở thời tiết cực đoan, (2) trễ tuyến chính cao hơn, (3) thời gian
quyết định 0,4–2 s. KHÔNG học lại mô hình: dùng DM11 (3 seed) của TN11; chấm trên 11 bộ kiểm tra 4–5 luồng của TN11.

    python -m scripts.tn14 toi-da                 # chạy hết; máy khác máy TN11: thêm --tn11 <thư mục | zip> như TN12

Từng bước (chạy lại an toàn — phần đã xong tự bỏ qua):
    nhom       dựng lại nhóm rủi ro của từng cạnh (cùng seed, cùng cạnh hợp lệ lúc quyết định) cho val, val6, 11 bộ kiểm tra
               → data/bao_da_tuyen/tn14/nhom/
    chon-val   trên val + val6: chọn ρ của bảng rủi ro tương quan và cấu hình giải mã nhanh theo luật ghi trước
               → tn14/chon_val.json
    thoi-gian  đo riêng từng khâu, 1 tiến trình, 1 luồng CPU, cửa sổ 0 của bộ C, 4–5 / 10 / 20 luồng → tn14/thoi_gian.json
    cham       11 bộ × 3 seed: DM11, Lai(DM11), vá hết tuyến (ρ* / ρ = 0), vá + giảm trễ, cấu hình nhanh, DM14 → tn14/cham_tep/
    kho        sinh lại c6, yagi5, xy11 hai lần (kho cũ / kho nhanh, không thầy) rồi chấm DM11 trên cả hai → tn14/kho_tep/
    phan-tich  → tn14/KET_QUA_14.md, ket_qua_14.json, hinh_14_*.png; gói results/bao_da_tuyen/TN14_ket_qua.zip
Log: logs/tn14.log.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import pickle
import sys
import time
import zlib
from multiprocessing import Pool
from pathlib import Path

import numpy as np

GOC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(GOC))
from scripts import tn11 as t11  # noqa: E402
from scripts import tn12 as t12  # noqa: E402

KQ, DL, LOG, M11 = t11.KQ, t11.DL, t11.LOG, t11.M11
RA, DL14 = KQ / "tn14", DL / "tn14"
NHOM = DL14 / "nhom"
BO_45, CUC_DOAN, TEN_BO, DO, SEEDS = t11.BO_45, t11.CUC_DOAN, t11.TEN_BO, t11.DO, t11.SEEDS
VAL = ("val_k4_5", "val6_k4_5")
W_BANG, RHO_LUOI = 64, (0.0, 0.5, 1.0)                 # số thế giới bảng; lưới ρ của bảng tương quan
# (số bước, số chuỗi, giải mã không tái nhiễu); gốc DM11 = (5, 32, False). Cập nhật 17/11 trước khi chạy (TN12 quét: 10 bước
# × 8 chuỗi gần bằng 5 × 32; TN12b: giải mã không tái nhiễu tốt hơn ~0,19 điểm) — xem TIEU_CHI_14 "Thay đổi trước khi chạy".
LUOI_NHANH = tuple((nb, S, k) for nb, S in ((5, 32), (10, 16), (10, 8), (5, 16), (5, 8)) for k in (False, True)) + ((3, 16, False),)
NGUONG_DUOI = 100.0                                     # ms — "tình huống nghẽn nặng" (TN13: chênh trễ nằm ở đuôi)


def ten_cau_hinh(nb, S, ktn):
    return f"{nb}x{S}" + ("k" if ktn else "")


def giai_ma(dm, gon, nb, S, ktn, gen):
    from src.nhieu_luong import mo_hinh_dt as md
    return md.chon_sua_dan(dm, gon, S, nb, gen) if ktn else md.chon_dm(dm, gon, S, nb, gen)
KHONG_KEM, TRE_NHANH = 0.10, 0.5                        # điểm % (ngưỡng không kém); ms (luật chọn cấu hình nhanh)
TI_LE_THOI_GIAN = 0.5                                   # H14.3: DM14 đầu – cuối ≤ 0,5 × DM11 ở mọi quy mô
KHO_BO = ("c6", "yagi5", "xy11")                        # bộ sinh lại hai lần để so kho cũ / kho nhanh
J_MAT, J_A, J_B, J_TRE = 0, 2, 3, 4                     # chỉ số trong DO
BIEN = [("Dijkstra phạt", "dphat", False), ("Lai(DM11)", "lai", True), ("DM11", "dm", True),
        ("DM11 + vá hết tuyến, bảng độc lập", "p1i", True), ("DM11 + vá hết tuyến", "p1", True),
        ("DM11 + vá + giảm trễ", "p12", True), ("DM11 cấu hình nhanh", "nhanh", True),
        ("DM14 = nhanh + vá + giảm trễ", "dm14", True)]


def py(*a):
    return t11.py(*a)


def _gio(f):
    s = time.perf_counter()
    a = f()
    return a, (time.perf_counter() - s) * 1e3


def _ghi_pickle(tep, x):
    Path(tep).parent.mkdir(parents=True, exist_ok=True)
    tam = str(tep) + ".tam"
    with open(tam, "wb") as f:
        pickle.dump(x, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tam, tep)


# ============================================================================================ nhóm rủi ro
def nhom_cua_canh(g, u, v):
    """Nhóm rủi ro của các cạnh (u, v) như da_tuyen.nhom_rui_ro → (nhom [E, 3] chỉ số nhóm, −1 = trống; nut [số nhóm]: nút
    của nhóm nút, −1 = nhóm mưa)."""
    from src.nhieu_luong import da_tuyen as dt
    M, khoa = dt.nhom_rui_ro(g, u, v)
    M = M.tocsr()
    nhom = -np.ones((len(u), 3), np.int16)
    for e in range(len(u)):
        c = M.indices[M.indptr[e]:M.indptr[e + 1]]
        nhom[e, :len(c)] = c
    nut = -np.ones(len(khoa), np.int32)
    for kk, j in khoa.items():
        if kk[0] == "nut":
            nut[j] = kk[1]
    return nhom, nut


def dau_cuoi(gon, u, v):
    """[K, 2] nút nguồn, đích của từng luồng (từ tuyến đầu trong kho)."""
    ra = []
    for k in range(gon.K):
        e0 = np.flatnonzero(gon.inc[k][0])
        a, b = set(u[e0].tolist()), set(v[e0].tolist())
        ra.append((min(a - b, default=-1), min(b - a, default=-1)))
    return np.array(ra, np.int32)


def _nhom_ban_ghi(g, r, eu, ev):
    gon = r["gon"]
    x = {"t": int(r["t"]), "muc": r["muc"], "ok": False}
    if len(gon.U) == 0 or int(gon.U.max()) >= len(eu):
        return x
    u, v = eu[gon.U], ev[gon.U]
    if not np.array_equal(np.asarray(g.band[u, v], np.int64), np.asarray(gon.dac_trung_canh["band"], np.int64)):
        return x
    nhom, nut = nhom_cua_canh(g, u, v)
    x.update(ok=True, nhom=nhom, nut=nut, dau_cuoi=dau_cuoi(gon, u, v))
    return x


def _nhom_tep(job):
    from src.ngau_nhien.kich_ban import sinh_kich_ban
    from src.ngau_nhien.ma_tran_rui_ro import ma_tran_online
    from src.nhieu_luong import tao_du_lieu_dt as tdt
    from src.nhieu_luong import tao_du_lieu_rr as td
    from src.nhieu_luong.bai_toan_rr import DELTA, TUOI
    d, tep = job
    ra = NHOM / d / Path(tep).name
    if ra.exists():
        return tep, None
    if td._g is None:
        td._khoi_tao()
    recs = pickle.load(open(tep, "rb"))
    out = []
    if recs:
        g, p = td._g, td._p
        tap, kb = d[:-len("_k4_5")], int(recs[0]["kb"])
        _, h0, seed = tdt.cac_cua_so_tap(tap)[kb]
        tt, p_sim = tdt.moi_truong(tap, kb)
        Wn, q = sinh_kich_ban(g, tt, h0, seed, p_sim)
        canh = {}
        for r in recs:
            t = int(r["t"])
            o = t - TUOI
            if t not in canh:
                past = {k: q[k][:o + 1] for k in ("R", "gust", "giong", "tat")}
                valid = ma_tran_online(td._bang, g, Wn[o], past, t, TUOI, DELTA, tham_so=p)["valid"] & (q["rho"][o] < 1.0)
                np.fill_diagonal(valid, False)
                canh[t] = np.nonzero(valid)
            out.append(_nhom_ban_ghi(g, r, *canh[t]))
    _ghi_pickle(ra, out)
    return tep, (sum(x["ok"] for x in out), len(out))


def lenh_nhom(a):
    from src.nhieu_luong import tao_du_lieu_rr as td
    ds = list(VAL) + [t11.thu_muc_bo(b, "4-5") for b in BO_45]
    viec = [(d, t) for d in ds for t in sorted(glob.glob(str(DL / d / "kb_*.pkl")))]
    thieu = [d for d in ds if not (DL / d).exists()]
    if thieu:
        raise SystemExit(f"thiếu dữ liệu TN11: {thieu} (máy khác máy TN11: thêm --tn11 như TN12)")
    con = [v for v in viec if not (NHOM / v[0] / Path(v[1]).name).exists()]
    print(f"  nhóm rủi ro: {len(viec)} tệp, còn {len(con)} ({a.tien_trinh_sinh} tiến trình)", flush=True)
    if con:
        s = time.time()
        with Pool(a.tien_trinh_sinh, initializer=td._khoi_tao) as pool:
            for k, (tep, n) in enumerate(pool.imap_unordered(_nhom_tep, con)):
                if k % 10 == 0 or k == len(con) - 1:
                    print(f"    {k + 1}/{len(con)} tệp ({time.time() - s:.0f} s)", flush=True)
    ok = {}
    for d, t in viec:
        x = pickle.load(open(NHOM / d / Path(t).name, "rb"))
        o = ok.setdefault(d, [0, 0])
        o[0], o[1] = o[0] + sum(z["ok"] for z in x), o[1] + len(x)
    RA.mkdir(parents=True, exist_ok=True)
    (RA / "nhom_khop.json").write_text(json.dumps(ok, ensure_ascii=False), encoding="utf-8")
    for d, (m, n) in ok.items():
        if n and m < 0.99 * n:
            print(f"  CẢNH BÁO {d}: chỉ {m}/{n} bản ghi dựng lại được nhóm rủi ro (cạnh không khớp) — bản ghi đó dùng bảng "
                  "độc lập", flush=True)
    print("  khớp: " + ", ".join(f"{d} {m}/{n}" for d, (m, n) in ok.items()), flush=True)


def _doc_nhom(d, tep):
    return pickle.load(open(NHOM / d / Path(tep).name, "rb"))


def _khop(r, z):
    if int(r["t"]) != z["t"] or r["muc"] != z["muc"]:
        raise SystemExit(f"tệp nhóm lệch bản ghi (t {r['t']} / {z['t']}) — xoá data/bao_da_tuyen/tn14/nhom rồi chạy lại")


# ============================================================================================ hậu xử lý
def bang(gon, z, sd, rho):
    from src.nhieu_luong import da_tuyen as dt
    return dt.TheGioi.tu_bang_nhom(gon, z.get("nhom") if z["ok"] else None, np.random.default_rng([sd, 14]), rho, W_BANG)


def p12(gon, z, sd, rho, a, giam=True):
    """(lời giải sau vá hết tuyến, sau vá + giảm trễ, ms vá — gồm dựng bảng —, ms giảm trễ)."""
    from src.nhieu_luong import hau_xu_ly as hx
    s = time.perf_counter()
    B = bang(gon, z, sd, rho)
    a1 = hx.va_het_tuyen(B, a)
    t1 = time.perf_counter()
    a2 = hx.giam_tre(B, a1) if giam else a1
    return a1, a2, (t1 - s) * 1e3, (time.perf_counter() - t1) * 1e3


def doc_chon_val():
    t = RA / "chon_val.json"
    if not t.exists():
        raise SystemExit("chưa có tn14/chon_val.json — chạy bước chon-val")
    return json.loads(t.read_text(encoding="utf-8"))


# ============================================================================================ chọn trên val
def _val_tep(job):
    import torch

    from src.nhieu_luong import mo_hinh_dt as md
    from src.nhieu_luong.cham_lai_dt import giong_ban_ghi
    from src.nhieu_luong.cham_mo_hinh_dt import _tg
    from src.nhieu_luong import hau_xu_ly as hx
    tep, d = job
    torch.set_num_threads(1)
    G = lambda sd: torch.Generator().manual_seed(sd)
    ra = []
    for r, z in zip(pickle.load(open(tep, "rb")), _doc_nhom(d, tep)):
        _khop(r, z)
        gon = r["gon"]
        tgw, sd = _tg(gon, r["cham"]), giong_ban_ghi(r)
        x = {"kb": int(r["kb"]), "t": int(r["t"]), "ok_nhom": bool(z["ok"])}
        for s in SEEDS:
            dm = t11._net(str(M11), f"dm11_s{s}")
            for nb, S, ktn in LUOI_NHANH:
                c = ten_cau_hinh(nb, S, ktn)
                a, ms = _gio(lambda: giai_ma(dm, gon, nb, S, ktn, G(sd)))
                x[f"dm_{c}_s{s}"], x[f"ms_dm_{c}_s{s}"] = t11._do(gon, tgw, a), ms
                if (nb, S, ktn) == (5, 32, False):
                    a0 = a
            for rh in RHO_LUOI:
                x[f"p1_{rh:g}_s{s}"] = t11._do(gon, tgw, hx.va_het_tuyen(bang(gon, z, sd, rh), a0))
        ra.append(x)
    return tep, ra


def chon(V):
    """Luật ghi trước (TIEU_CHI_14): ρ* = ρ cho mất DV val nhỏ nhất của DM11 + vá hết tuyến (hoà → ρ nhỏ hơn); cấu hình
    nhanh = cấu hình có số bước × số chuỗi nhỏ nhất mà mất DV val ≤ gốc + 0,10 điểm và trễ ≤ gốc + 0,5 ms (hoà chi phí →
    mất DV val thấp hơn). Trung bình 3 seed, gộp val + val6. Lưới gồm cả giải mã không tái nhiễu (cập nhật 17/11)."""
    gop = lambda k, j=J_MAT: float(np.mean([[x[f"{k}_s{s}"][j] for x in V] for s in SEEDS]))
    goc = {"mat_dv": gop("dm_5x32"), "tre": gop("dm_5x32", J_TRE), "a": gop("dm_5x32", J_A)}
    p1 = {f"{rh:g}": {"mat_dv": gop(f"p1_{rh:g}"), "a": gop(f"p1_{rh:g}", J_A)} for rh in RHO_LUOI}
    rho = min(RHO_LUOI, key=lambda rh: (round(p1[f"{rh:g}"]["mat_dv"], 3), rh))
    cau = {ten_cau_hinh(*c): {"mat_dv": gop(f"dm_{ten_cau_hinh(*c)}"), "tre": gop(f"dm_{ten_cau_hinh(*c)}", J_TRE),
                              "chi_phi": c[0] * c[1],
                              "ms": float(np.median([x[f"ms_dm_{ten_cau_hinh(*c)}_s{s}"] for x in V for s in SEEDS]))}
           for c in LUOI_NHANH}
    duoc = [c for c in LUOI_NHANH if cau[ten_cau_hinh(*c)]["mat_dv"] <= goc["mat_dv"] + KHONG_KEM
            and cau[ten_cau_hinh(*c)]["tre"] <= goc["tre"] + TRE_NHANH]
    nb, S, ktn = min(duoc, key=lambda c: (c[0] * c[1], round(cau[ten_cau_hinh(*c)]["mat_dv"], 3), ten_cau_hinh(*c)))
    return {"rho": rho, "nhanh": [nb, S, bool(ktn)], "goc": goc, "p1": p1, "cau_hinh": cau,
            "duoc": [ten_cau_hinh(*c) for c in duoc], "so_ban_ghi": len(V),
            "ti_le_nhom_khop": float(np.mean([x["ok_nhom"] for x in V]))}


def lenh_chon_val(a):
    viec = [(f"{d}__{Path(t).stem}.json", (t, d)) for d in VAL for t in sorted(glob.glob(str(DL / d / "kb_*.pkl")))]
    print(f"  chọn trên val: {len(viec)} tệp", flush=True)
    T = t11._chay_tep(_val_tep, viec, a.tien_trinh, RA / "val_tep")
    cv = chon([x for v in T.values() for x in v])
    (RA / "chon_val.json").write_text(json.dumps(cv, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"  ρ* = {cv['rho']:g} (mất DV val: " + ", ".join(f"ρ {k}: {v['mat_dv']:.3f}" for k, v in cv["p1"].items())
          + f"; DM11 {cv['goc']['mat_dv']:.3f}); cấu hình nhanh {ten_cau_hinh(*cv['nhanh'])} (k = không tái nhiễu)", flush=True)


# ============================================================================================ đo thời gian
def _chi_phi_suurballe(mg, kt):
    return sum(int(np.round(mg.d0[mg.dt.canh_cua(kt["kho"][i])] * 1000).astype(np.int64).sum()) for i in kt["i_suurballe"])


def _ky_tuyen(mg, r):
    return frozenset(int(e) for e in mg.dt.canh_cua(r))


def lenh_thoi_gian(a):
    """Như bao_cao_6_10.do_thoi_gian (TN11 'đo riêng'): cửa sổ 0 của bộ C, MỘT tiến trình, 1 luồng CPU; mỗi mốc dựng tình
    huống rồi đo từng khâu (ms). Kho cũ và kho nhanh dựng trên CÙNG đồ thị, cùng cặp nguồn – đích; so chi phí Suurballe."""
    tep = RA / "thoi_gian.json"
    if tep.exists():
        print(f"  đã có {tep.name}", flush=True)
        return
    import torch

    from src.ngau_nhien.kich_ban import sinh_kich_ban
    from src.ngau_nhien.ma_tran_rui_ro import ma_tran_online
    from src.nhieu_luong import da_tuyen as dt
    from src.nhieu_luong import hau_xu_ly as hx
    from src.nhieu_luong import mo_hinh_dt as md
    from src.nhieu_luong import tao_du_lieu_dt as tdt
    from src.nhieu_luong import tao_du_lieu_rr as td
    from src.nhieu_luong.bai_toan_rr import DELTA, TUOI
    from src.nhieu_luong.cham_lai_dt import rho_hat
    cv = doc_chon_val()
    nb, S, ktn = cv["nhanh"]
    torch.set_num_threads(1)
    pc = time.perf_counter
    td._khoi_tao()
    g, p = td._g, td._p
    tap, cua_so = "c6", 0
    ten, h0, seed = tdt.cac_cua_so_tap(tap)[cua_so]
    tt, p_sim = tdt.moi_truong(tap, cua_so)
    Wn, q = sinh_kich_ban(g, tt, h0, seed, p_sim)
    moc = td.cac_moc(tdt.HINH_HOC.get(tap, tap))
    dm = md.nap("dm11_s0", M11)
    ra, kho = [], {"cap": 0, "chi_phi_bang": 0, "kho_trung": 0, "suurballe_trung": 0}
    for K_rg, buoc in (((4, 5), 1), ((10, 10), 2), ((20, 20), 4)):
        rng = np.random.default_rng([seed, K_rg[1]])
        s0 = time.time()
        for t in moc[::buoc]:
            t = int(t)
            o = t - TUOI
            s = pc()
            past = {k: q[k][:o + 1] for k in ("R", "gust", "giong", "tat")}
            r = ma_tran_online(td._bang, g, Wn[o], past, t, TUOI, DELTA, tham_so=p)
            x = {"rui_ro": pc() - s}
            dt.SUURBALLE_NHANH = False
            th = dt.tao_tinh_huong(g, q, r, t, rng, so_luong=K_rg, so_nguon=tdt.so_nguon(K_rg[1]))
            if th is None:
                continue
            s = pc()
            mg = dt.Mang(g, q, r, t)
            x["mang"] = pc() - s
            cap = sorted({(int(u), int(v)) for u, v in zip(th["src"], th["dst"])})
            kt = {}
            for ten_k, co in (("kho_cu", False), ("kho_nhanh", True)):
                dt.SUURBALLE_NHANH = co
                s = pc()
                kt[ten_k] = [dt.kho_tuyen(mg, u, v) for u, v in cap]
                x[ten_k] = pc() - s
            dt.SUURBALLE_NHANH = False
            for c, n in zip(kt["kho_cu"], kt["kho_nhanh"]):
                kho["cap"] += 1
                kho["chi_phi_bang"] += int(_chi_phi_suurballe(mg, c) == _chi_phi_suurballe(mg, n))
                kho["kho_trung"] += int({_ky_tuyen(mg, r_) for r_ in c["kho"]} == {_ky_tuyen(mg, r_) for r_ in n["kho"]})
                kho["suurballe_trung"] += int({_ky_tuyen(mg, c["kho"][i]) for i in c["i_suurballe"]}
                                              == {_ky_tuyen(mg, n["kho"][i]) for i in n["i_suurballe"]})
            for muc in np.random.default_rng([seed, t, 5]).uniform(*tdt.TAI_LIEN_TUC, 3).tolist():
                y = dict(x)
                lam = th["lam"] * muc / th["u"]
                s = pc()
                dt.dijkstra_phat(th, lam, *tdt.BETA_GAMMA)
                y["dphat"] = pc() - s
                s = pc()
                bt = dt.BaiToan(th, muc)
                gon = dt.BaiToanGon(bt)
                gon.dac_trung_canh = tdt.dac_trung_canh(g, q, th["mg"], gon.U, o, tdt.cape_luc(h0, o, tdt.tep_thoi_tiet(tap)))
                gon.dac_trung_uv = [tdt.dac_trung_uv(g, th, k) for k in range(bt.K)]
                y["gon"] = pc() - s
                Gn = torch.Generator()
                ket = {}
                for ten_m, f in (("dm", lambda: md.chon_dm(dm, gon, 32, 5, Gn.manual_seed(t))),
                                 ("nhanh", lambda: giai_ma(dm, gon, nb, S, ktn, Gn.manual_seed(t)))):
                    s = pc()
                    ket[ten_m] = f()
                    y[ten_m] = pc() - s
                s = pc()
                u_, v_ = th["mg"].eu[gon.U], th["mg"].ev[gon.U]
                nhom, _ = nhom_cua_canh(g, u_, v_)
                B = dt.TheGioi.tu_bang_nhom(gon, nhom, np.random.default_rng([t, 14]), cv["rho"], W_BANG)
                a1 = hx.va_het_tuyen(B, ket["nhanh"])
                y["va"] = pc() - s
                s = pc()
                hx.giam_tre(B, a1)
                y["giam_tre"] = pc() - s
                s = pc()
                if rho_hat(gon) >= t11.TAU:
                    dt.tkcb(dt.TheGioi.tu_bang(gon, np.random.default_rng([seed, t, 4])), ket["dm"])
                y["lai_them"] = pc() - s
                torch.set_num_threads(4)
                for ten_m, f in (("dm_4luong", lambda: md.chon_dm(dm, gon, 32, 5, Gn.manual_seed(t))),
                                 ("nhanh_4luong", lambda: giai_ma(dm, gon, nb, S, ktn, Gn.manual_seed(t)))):
                    s = pc()
                    f()
                    y[ten_m] = pc() - s
                torch.set_num_threads(1)
                ra.append({"K": int(bt.K), "nhom": f"k{K_rg[0]}_{K_rg[1]}" if K_rg[0] != K_rg[1] else f"k{K_rg[0]}", "t": t,
                           **{k: (1e3 * v if isinstance(v, float) else v) for k, v in y.items()}})
        print(f"  {time.strftime('%H:%M:%S')} {K_rg}: {sum(1 for z in ra if z['nhom'].startswith('k' + str(K_rg[0])))} "
              f"tình huống, {time.time() - s0:.0f} s", flush=True)
    RA.mkdir(parents=True, exist_ok=True)
    tep.write_text(json.dumps({"tap": tap, "cua_so": ten, "cau_hinh_nhanh": [nb, S, ktn], "rho": cv["rho"], "kho": kho,
                               "cpu": os.cpu_count(), "ban_ghi": ra}, ensure_ascii=False), encoding="utf-8")
    print(f"  → {tep.name}: {len(ra)} tình huống; Suurballe cùng chi phí {kho['chi_phi_bang']}/{kho['cap']} cặp, "
          f"kho trùng {kho['kho_trung']}/{kho['cap']}", flush=True)


# ============================================================================================ chấm bộ kiểm tra
def _cham_tep(job):
    import torch

    from src.nhieu_luong import hau_xu_ly as hx
    from src.nhieu_luong import mo_hinh_dt as md
    from src.nhieu_luong.cham_lai_dt import giong_ban_ghi, rho_hat
    from src.nhieu_luong.cham_mo_hinh_dt import _tg
    tep, d, rho, nhanh = job
    nb, S, ktn = nhanh
    torch.set_num_threads(1)
    G = lambda sd: torch.Generator().manual_seed(sd)
    ra = []
    for r, z in zip(pickle.load(open(tep, "rb")), _doc_nhom(d, tep)):
        _khop(r, z)
        gon = r["gon"]
        tgw, sd, x = _tg(gon, r["cham"]), giong_ban_ghi(r), t11._co_ban(r)
        x["rho"], x["ok_nhom"], x["san_a_kho"] = rho_hat(gon), bool(z["ok"]), hx.san_a_kho(tgw)
        hc = (lambda a: hx.diem_hong_chung(gon, z["nhom"], z["nut"], z["dau_cuoi"], a)) if z["ok"] else (lambda a: None)
        luu = lambda k, a, ms: x.update({k: t11._do(gon, tgw, a), f"ms_{k}": ms})
        for s in SEEDS:
            dm = t11._net(str(M11), f"dm11_s{s}")
            a, ms = _gio(lambda: md.chon_dm(dm, gon, 32, 5, G(sd)))
            luu(f"dm_s{s}", a, ms)
            if x["rho"] >= t11.TAU:
                al, ms_l = t11._lai(gon, r, a)
                luu(f"lai_s{s}", al, ms + ms_l)
            else:
                luu(f"lai_s{s}", a, ms)
            a1, a12, m1, m2 = p12(gon, z, sd, rho, a)
            luu(f"p1_s{s}", a1, ms + m1)
            luu(f"p12_s{s}", a12, ms + m1 + m2)
            if rho > 0:
                a1i, _, m1i, _ = p12(gon, z, sd, 0.0, a, giam=False)
                luu(f"p1i_s{s}", a1i, ms + m1i)
            else:
                x[f"p1i_s{s}"], x[f"ms_p1i_s{s}"] = x[f"p1_s{s}"], x[f"ms_p1_s{s}"]
            x[f"hc_dm_s{s}"], x[f"hc_p1_s{s}"] = hc(a), hc(a1)
            if (nb, S, ktn) == (5, 32, False):
                af, msf = a, ms
            else:
                af, msf = _gio(lambda: giai_ma(dm, gon, nb, S, ktn, G(sd)))
            luu(f"nhanh_s{s}", af, msf)
            _, a14, m1f, m2f = p12(gon, z, sd, rho, af)
            luu(f"dm14_s{s}", a14, msf + m1f + m2f)
        ra.append(x)
    return tep, ra


def lenh_cham(a):
    cv = doc_chon_val()
    viec = [(f"{t11.thu_muc_bo(b, '4-5')}__{Path(t).stem}.json", (t, t11.thu_muc_bo(b, "4-5"), cv["rho"], cv["nhanh"]))
            for b in BO_45 for t in sorted(glob.glob(str(DL / t11.thu_muc_bo(b, "4-5") / "kb_*.pkl")))]
    if not viec:
        raise SystemExit(f"không thấy bộ kiểm tra TN11 ở {DL}")
    print(f"  chấm bộ kiểm tra: {len(viec)} tệp (ρ* = {cv['rho']:g}, nhanh {ten_cau_hinh(*cv['nhanh'])})", flush=True)
    t11._chay_tep(_cham_tep, viec, a.tien_trinh, RA / "cham_tep")


# ============================================================================================ kho nhanh: chất lượng
def _ky_ban_ghi(gon):
    """Chữ ký kho từng luồng: crc32 của tập tuyến (mỗi tuyến = tập chỉ số cạnh toàn mạng)."""
    ra = []
    for k in range(gon.K):
        tap = sorted(tuple(sorted(int(e) for e in gon.U[np.flatnonzero(row)])) for row in gon.inc[k])
        ra.append(zlib.crc32(repr(tap).encode()))
    return ra


def _kho_tep(job):
    import torch

    from src.nhieu_luong import mo_hinh_dt as md
    from src.nhieu_luong.cham_lai_dt import giong_ban_ghi
    from src.nhieu_luong.cham_mo_hinh_dt import _tg
    (tep,) = job
    torch.set_num_threads(1)
    ra = []
    for r in pickle.load(open(tep, "rb")):
        gon = r["gon"]
        tgw, sd, x = _tg(gon, r["cham"]), giong_ban_ghi(r), t11._co_ban(r)
        x.update(muc=str(r["muc"]), K=int(r["K"]), ky=_ky_ban_ghi(gon), ms_kho=float(r["tg_ms"].get("kho", np.nan)))
        for s in SEEDS:
            a = md.chon_dm(t11._net(str(M11), f"dm11_s{s}"), gon, 32, 5, torch.Generator().manual_seed(sd))
            x[f"dm_s{s}"] = t11._do(gon, tgw, a)
        ra.append(x)
    return tep, ra


def lenh_kho(a):
    for b in KHO_BO:
        for ten, co in (("kho_cu", ()), ("kho_nhanh", ("--kho-nhanh",))):
            d = DL14 / ten / t11.thu_muc_bo(b, "4-5")
            if (d / ".xong").exists():
                continue
            print(f"  sinh {b} ({ten})", flush=True)
            t11.chay(py("-m", "src.nhieu_luong.tao_du_lieu_dt", "--tap", b, "--K", "4-5", "--tien-trinh",
                        str(a.tien_trinh_sinh), "--khong-thay", "--ra", str(DL14 / ten), *co))
    viec = [(f"{ten}__{t11.thu_muc_bo(b, '4-5')}__{Path(t).stem}.json", (t,)) for ten in ("kho_cu", "kho_nhanh")
            for b in KHO_BO for t in sorted(glob.glob(str(DL14 / ten / t11.thu_muc_bo(b, "4-5") / "kb_*.pkl")))]
    print(f"  chấm kho cũ / kho nhanh: {len(viec)} tệp", flush=True)
    t11._chay_tep(_kho_tep, viec, a.tien_trinh, RA / "kho_tep")


# ============================================================================================ phân tích
def _v(rs, k, co_seed, j=J_MAT):
    return t11._ho(rs, k, SEEDS, j) if co_seed else t11._cot(rs, k, j)


def _hi(ci):
    return ci[1][1]


def phan_tich_bo(T):
    bo = {}
    for b, rs in T.items():
        kb = np.array([r["kb"] for r in rs])
        v = {k: {j: _v(rs, k, cs, j) for j in (J_MAT, J_A, J_B, J_TRE)} for _, k, cs in BIEN}
        d = lambda x, y, j: t11._bs(v[x][j] - v[y][j], kb)
        duoi = lambda x, y: t11._bs(((v[x][J_TRE] > NGUONG_DUOI).astype(float) - (v[y][J_TRE] > NGUONG_DUOI)) * 100, kb)
        hc = lambda h: [float(np.mean([r[f"hc_{h}_s{s}"] for r in rs if r[f"hc_{h}_s{s}"] is not None])) if any(
            r[f"hc_{h}_s{s}"] is not None for r in rs) else float("nan") for s in SEEDS]
        bo[b] = {"n": len(rs), "ti_le_nhom_khop": float(np.mean([r["ok_nhom"] for r in rs])),
                 "trung_binh": {ten: {"mat_dv": float(v[k][J_MAT].mean()), "a": float(v[k][J_A].mean()),
                                      "b": float(v[k][J_B].mean()), "tre": float(v[k][J_TRE].mean()),
                                      "tre_trung_vi": float(np.median(v[k][J_TRE])),
                                      "tre_duoi": float(np.mean(v[k][J_TRE] > NGUONG_DUOI) * 100),
                                      "ms": float(np.median([r[f"ms_{k}_s{s}" if cs else f"ms_{k}"] for r in rs for s in
                                                            (SEEDS if cs else (0,))]))}
                                for ten, k, cs in BIEN},
                 "H14.2_duoi": duoi("p12", "p1"), "dm14_duoi_vs_dm": duoi("dm14", "dm"),
                 "dm14_duoi_vs_dphat": duoi("dm14", "dphat"),
                 "san_a_kho": float(np.mean([r["san_a_kho"] for r in rs])),
                 "hong_chung": {"DM11": float(np.nanmean(hc("dm"))), "vá hết tuyến": float(np.nanmean(hc("p1")))},
                 "H14.1_a": d("p1", "dm", J_A), "H14.1_mat": d("p1", "dm", J_MAT), "H14.1_b": d("p1", "dm", J_B),
                 "p1_a_vs_dphat": d("p1", "dphat", J_A), "dm_a_vs_dphat": d("dm", "dphat", J_A),
                 "tuong_quan_a": d("p1", "p1i", J_A), "tuong_quan_mat": d("p1", "p1i", J_MAT),
                 "H14.2_tre": d("p12", "p1", J_TRE), "H14.2_mat": d("p12", "p1", J_MAT),
                 "p12_tre_vs_dphat": d("p12", "dphat", J_TRE), "dm_tre_vs_dphat": d("dm", "dphat", J_TRE),
                 "nhanh_mat": d("nhanh", "dm", J_MAT),
                 "H14.4_mat": d("dm14", "dm", J_MAT), "H14.4_tre": d("dm14", "dm", J_TRE), "H14.4_a": d("dm14", "dm", J_A),
                 "dm14_vs_dphat": d("dm14", "dphat", J_MAT), "dm14_vs_lai": d("dm14", "lai", J_MAT)}
    return bo


def phan_tich_thoi_gian():
    t = RA / "thoi_gian.json"
    if not t.exists():
        return None
    x = json.loads(t.read_text(encoding="utf-8"))
    md_ = lambda rs, f: float(np.median([f(r) for r in rs]))
    ra = {"kho": x["kho"], "cau_hinh_nhanh": x["cau_hinh_nhanh"], "nhom": {}}
    chung = lambda r: r["rui_ro"] + r["mang"] + r["gon"]
    for n in ("k4_5", "k10", "k20"):
        rs = [r for r in x["ban_ghi"] if r["nhom"] == n]
        if not rs:
            continue
        ra["nhom"][n] = {
            "so": len(rs), **{k: md_(rs, lambda r, k=k: r[k]) for k in (
                "rui_ro", "mang", "kho_cu", "kho_nhanh", "gon", "dphat", "dm", "nhanh", "va", "giam_tre", "lai_them",
                "dm_4luong", "nhanh_4luong")},
            "dc_dphat": md_(rs, lambda r: r["rui_ro"] + r["mang"] + r["dphat"]),
            "dc_dm11": md_(rs, lambda r: chung(r) + r["kho_cu"] + r["dm"]),
            "dc_lai11": md_(rs, lambda r: chung(r) + r["kho_cu"] + r["dm"] + r["lai_them"]),
            "dc_dm14": md_(rs, lambda r: chung(r) + r["kho_nhanh"] + r["nhanh"] + r["va"] + r["giam_tre"]),
            "dc_dm14_4luong": md_(rs, lambda r: chung(r) + r["kho_nhanh"] + r["nhanh_4luong"] + r["va"] + r["giam_tre"])}
        v = ra["nhom"][n]
        v["ti_le_dm14_dm11"] = v["dc_dm14"] / v["dc_dm11"]
    return ra


def phan_tich_kho():
    th = RA / "kho_tep"
    if not th.exists():
        return None
    ra = {}
    for b in KHO_BO:
        cu, nh = (t11._gop(th, f"{ten}__{t11.thu_muc_bo(b, '4-5')}") for ten in ("kho_cu", "kho_nhanh"))
        if not cu or len(cu) != len(nh):
            ra[b] = {"loi": f"số bản ghi {len(cu)} / {len(nh)}"}
            continue
        if any((x["kb"], x["t"], x["muc"]) != (y["kb"], y["t"], y["muc"]) for x, y in zip(cu, nh)):
            ra[b] = {"loi": "bản ghi không ghép cặp được"}
            continue
        kb = np.array([x["kb"] for x in cu])
        luong = [(p, q) for x, y in zip(cu, nh) for p, q in zip(x["ky"], y["ky"])]
        dcu, dnh = _v(cu, "dm", True), _v(nh, "dm", True)
        ra[b] = {"n": len(cu), "kho_trung_luong": float(np.mean([p == q for p, q in luong])),
                 "kho_trung_ban_ghi": float(np.mean([x["ky"] == y["ky"] for x, y in zip(cu, nh)])),
                 "mat_dv_cu": float(dcu.mean()), "mat_dv_nhanh": float(dnh.mean()), "hieu": t11._bs(dnh - dcu, kb),
                 "ms_kho_cu": float(np.nanmedian([x["ms_kho"] for x in cu])),
                 "ms_kho_nhanh": float(np.nanmedian([x["ms_kho"] for x in nh])),
                 "dphat_trung": float(np.mean([abs(x["dphat"][0] - y["dphat"][0]) < 1e-9 for x, y in zip(cu, nh)]))}
    return ra


def kiem_lap_lai(T):
    """DM11 của TN14 phải trùng DM11 của TN11 (cùng mô hình, cùng hạt giống) nếu có tn11/cham_tep."""
    c11 = KQ / "tn11" / "cham_tep"
    if not c11.exists():
        return None
    lech, n = 0.0, 0
    for b, rs in T.items():
        r11 = t11._gop(c11, t11.thu_muc_bo(b, "4-5"))
        if len(r11) != len(rs):
            return {"loi": f"{b}: số bản ghi khác ({len(rs)} vs {len(r11)})"}
        for x, y in zip(rs, r11):
            for s in SEEDS:
                lech = max(lech, abs(x[f"dm_s{s}"][0] - y[f"dm11_s{s}"][0]))
                n += 1
    return {"so_so_sanh": n, "lech_toi_da": lech}


def gia_thuyet(bo, tg, kho):
    n = len(bo)
    can = max(1, int(np.ceil(8 / 11 * n)))
    dem = lambda f: [b for b, x in bo.items() if f(x)]
    h1 = dem(lambda x: _hi(x["H14.1_a"]) < 0 and _hi(x["H14.1_mat"]) < KHONG_KEM)
    cd = [b for b in CUC_DOAN if b in bo]
    h1b = [b for b in cd if bo[b]["p1_a_vs_dphat"][0] <= 0]
    h2 = dem(lambda x: _hi(x["H14.2_tre"]) < 0 and _hi(x["H14.2_mat"]) < KHONG_KEM)
    nh = dem(lambda x: _hi(x["nhanh_mat"]) < KHONG_KEM)
    h4 = dem(lambda x: _hi(x["H14.4_mat"]) < KHONG_KEM and _hi(x["H14.4_tre"]) < 0)
    h4dp = dem(lambda x: _hi(x["dm14_vs_dphat"]) < 0)
    tl = {k: v["ti_le_dm14_dm11"] for k, v in (tg or {}).get("nhom", {}).items()}
    k_ = (tg or {}).get("kho", {})
    kho_tg = [b for b, x in (kho or {}).items() if "hieu" in x and -KHONG_KEM < x["hieu"][1][0] and x["hieu"][1][1] < KHONG_KEM]
    c3 = {"thoi_gian": bool(tl) and all(v <= TI_LE_THOI_GIAN for v in tl.values()), "ti_le": tl,
          "suurballe_cung_chi_phi": bool(k_) and k_.get("chi_phi_bang") == k_.get("cap"),
          "kho_tuong_duong": kho_tg, "kho_dat": bool(kho) and len(kho_tg) == len(KHO_BO),
          "nhanh_khong_kem": nh, "nhanh_dat": len(nh) >= can}
    c3["dat"] = c3["thoi_gian"] and c3["suurballe_cung_chi_phi"] and c3["kho_dat"] and c3["nhanh_dat"]
    return {"so_bo_can": can, "so_bo": n,
            "H14.1": {"bo_dat": h1, "dat": len(h1) >= can, "cuc_doan_a_khong_hon_dphat": h1b,
                      "bit_khoang_cach_dphat": len(cd) > 0 and len(h1b) >= int(np.ceil(0.75 * len(cd)))},
            "H14.2": {"bo_dat": h2, "dat": len(h2) >= can},
            "H14.3": c3,
            "H14.4": {"bo_dat": h4, "hon_dphat": h4dp, "dat": len(h4) >= can and len(h4dp) >= can}}


# -------------------------------------------------------------------------------------------- biểu đồ
MAU = {"dm": "#2a78d6", "p1": "#1baf7a", "p12": "#eda100", "dphat": "#eb6834", "dm14": "#6250d6"}


def ve_va(bo):
    plt = t11._plt()
    ten = [b for b in BO_45 if b in bo]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    for i, (khoa, tieu_de, mau) in enumerate((("H14.1_a", "vá hết tuyến − DM11: phần không còn tuyến sống (điểm %)", MAU["p1"]),
                                              ("H14.2_tre", "giảm trễ: (vá + giảm trễ) − vá, trễ tuyến chính (ms)", MAU["p12"]))):
        for y, b in enumerate(ten):
            m, (lo, hi) = bo[b][khoa]
            ax[i].plot([lo, hi], [y, y], color=mau, lw=2.2)
            ax[i].plot([m], [y], "o", color=mau, ms=7, mec="white", mew=1.5)
        ax[i].axvline(0, color="#888888", lw=1)
        ax[i].set_title(tieu_de, loc="left", fontsize=9)
    ax[0].set_yticks(range(len(ten)), [TEN_BO[b] for b in ten])
    ax[0].invert_yaxis()
    fig.suptitle("TN14 — hiệu theo bộ kiểm tra, CI 95% theo cửa sổ (trung bình 3 seed); sang trái là tốt hơn", x=0.01,
                 ha="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(RA / "hinh_14_va.png", bbox_inches="tight")
    plt.close(fig)


def ve_thoi_gian(tg):
    if not tg or not tg["nhom"]:
        return
    plt = t11._plt()
    nhom = list(tg["nhom"])
    khau = [("chuẩn bị (rủi ro, đồ thị, đặc trưng)", lambda v: v["rui_ro"] + v["mang"] + v["gon"], "#bbbbbb"),
            ("kho ứng viên", lambda v, c: v["kho_cu" if c else "kho_nhanh"], "#eb6834"),
            ("DM", lambda v, c: v["dm" if c else "nhanh"], "#2a78d6"),
            ("vá + giảm trễ", lambda v, c: 0.0 if c else v["va"] + v["giam_tre"], "#1baf7a")]
    fig, ax = plt.subplots(figsize=(9, 0.9 + 0.9 * len(nhom)))
    y, nhan = 0, []
    for n in nhom:
        v = tg["nhom"][n]
        for cu in (True, False):
            x0 = 0.0
            for j, (ten, f, mau) in enumerate(khau):
                w = f(v) if j == 0 else f(v, cu)
                ax.barh(y, w, left=x0, color=mau, height=0.7, edgecolor="white", linewidth=1.5,
                        label=ten if y == 1 else None)
                x0 += w
            ax.text(x0, y, f"  {x0:.0f} ms", va="center", fontsize=8, color="#333333")
            nhan.append(f"{n.replace('k', '').replace('_', '–')} luồng · {'DM11 (TN11)' if cu else 'DM14'}")
            y += 1
    ax.set_yticks(range(len(nhan)), nhan)
    ax.invert_yaxis()
    ax.set_xlabel("thời gian một lần quyết định, trung vị (ms), 1 tiến trình, 1 luồng CPU")
    ax.legend(ncol=4, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.32))
    ax.set_title("TN14 — thời gian đầu – cuối theo khâu", loc="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(RA / "hinh_14_thoi_gian.png", bbox_inches="tight")
    plt.close(fig)


# -------------------------------------------------------------------------------------------- báo cáo
def viet_md(ra):
    ci = t11._ci
    bo, gt, cv = ra["bo"], ra["gia_thuyet"], ra["chon_val"]
    L = ["# Thí nghiệm 14 — kết quả (sinh tự động bởi `scripts/tn14.py`)", "",
         "Tiêu chí: TIEU_CHI_14.md. 11 bộ kiểm tra 4–5 luồng của TN11, trung bình 3 seed DM11; hiệu bằng điểm % (trễ: ms), "
         "CI 95% bootstrap theo cửa sổ. Không học lại mô hình.", "",
         f"Chọn trên val ({cv['so_ban_ghi']} bản ghi, nhóm khớp {cv['ti_le_nhom_khop']:.1%}): ρ* = {cv['rho']:g}; cấu hình "
         f"nhanh {ten_cau_hinh(*cv['nhanh'])} — số bước x số chuỗi, k = không tái nhiễu (đạt luật: {', '.join(cv['duoc'])}).", "",
         "## 1. Trung bình theo bộ — mất DV % / (a) không còn tuyến sống % / trễ ms", "",
         "| Phương pháp | " + " | ".join(TEN_BO[b] for b in bo) + " |", "|---|" + "---|" * len(bo)]
    for ten, _, _ in BIEN:
        L.append(f"| {ten} | " + " | ".join(f"{x['trung_binh'][ten]['mat_dv']:.2f} / {x['trung_binh'][ten]['a']:.2f} / "
                                            f"{x['trung_binh'][ten]['tre']:.1f}" for x in bo.values()) + " |")
    L += ["", "## 2. Giả thuyết theo bộ (CI 95%)", "",
          "| Bộ | H14.1 (a): vá − DM11 | H14.1 mất DV | (a) vá − Dijkstra phạt | (a) DM11 − Dijkstra phạt | "
          "tương quan: (a) ρ* − ρ 0 | H14.2 trễ | H14.2 mất DV | nhanh − DM11 mất DV | H14.4 mất DV | H14.4 trễ | "
          "DM14 − Dijkstra phạt |", "|---|" + "---|" * 11]
    for b, x in bo.items():
        L.append(f"| {TEN_BO[b]} | {ci(x['H14.1_a'])} | {ci(x['H14.1_mat'])} | {ci(x['p1_a_vs_dphat'])} | "
                 f"{ci(x['dm_a_vs_dphat'])} | {ci(x['tuong_quan_a'])} | {ci(x['H14.2_tre'], 1)} | {ci(x['H14.2_mat'])} | "
                 f"{ci(x['nhanh_mat'])} | {ci(x['H14.4_mat'])} | {ci(x['H14.4_tre'], 1)} | {ci(x['dm14_vs_dphat'])} |")
    L += ["", f"Đạt = đúng chiều ở ≥ {gt['so_bo_can']}/{gt['so_bo']} bộ; không kém = cận trên CI < +{KHONG_KEM} điểm.", ""]
    for k in ("H14.1", "H14.2", "H14.3", "H14.4"):
        v = gt[k]
        L.append(f"- **{k}: {'ĐẠT' if v['dat'] else 'KHÔNG ĐẠT'}** — " + ", ".join(f"{kk} = {vv}" for kk, vv in v.items()
                                                                                 if kk != "dat"))
    L += ["", f"## 2b. Trễ ở giữa và phần đuôi (trễ trung vị ms / % tình huống trễ > {NGUONG_DUOI:.0f} ms)", "",
          "| Phương pháp | " + " | ".join(TEN_BO[b] for b in bo) + " |", "|---|" + "---|" * len(bo)]
    for ten, _, _ in BIEN:
        L.append(f"| {ten} | " + " | ".join(f"{x['trung_binh'][ten]['tre_trung_vi']:.1f} / {x['trung_binh'][ten]['tre_duoi']:.1f}"
                                            for x in bo.values()) + " |")
    L += ["", "| Bộ | đuôi: (vá + giảm trễ) − vá | đuôi: DM14 − DM11 | đuôi: DM14 − Dijkstra phạt |", "|---|---|---|---|"]
    for b, x in bo.items():
        L.append(f"| {TEN_BO[b]} | {ci(x['H14.2_duoi'], 1)} | {ci(x['dm14_duoi_vs_dm'], 1)} | {ci(x['dm14_duoi_vs_dphat'], 1)} |")
    L += ["", "## 3. Chẩn đoán phần (a)", "", "| Bộ | (a) DM11 | (a) vá | (a) Dijkstra phạt | sàn (a) của kho | "
          "% lưu lượng có điểm hỏng chung: DM11 → vá | nhóm khớp |", "|---|---:|---:|---:|---:|---|---:|"]
    for b, x in bo.items():
        tb = x["trung_binh"]
        L.append(f"| {TEN_BO[b]} | {tb['DM11']['a']:.2f} | {tb['DM11 + vá hết tuyến']['a']:.2f} | "
                 f"{tb['Dijkstra phạt']['a']:.2f} | {x['san_a_kho']:.2f} | {x['hong_chung']['DM11']:.1f} → "
                 f"{x['hong_chung']['vá hết tuyến']:.1f} | {x['ti_le_nhom_khop']:.1%} |")
    L += ["", "Sàn (a) của kho dùng thế giới chấm (chỉ để chẩn đoán): nếu sàn ≈ (a) của DM11 thì kho là giới hạn, cần mở rộng "
          "kho (đề xuất cho TN sau).", ""]
    tg = ra["thoi_gian"]
    if tg:
        L += ["## 4. Thời gian (đo riêng, trung vị ms, 1 tiến trình, 1 luồng CPU)", "",
              "| Khâu | " + " | ".join(tg["nhom"]) + " |", "|---|" + "---:|" * len(tg["nhom"])]
        for k, ten in (("kho_cu", "kho cũ (Suurballe networkx)"), ("kho_nhanh", "kho nhanh"), ("dm", "DM11 5 × 32"),
                       ("nhanh", "DM cấu hình nhanh"), ("dm_4luong", "DM11 5 × 32, 4 luồng CPU"),
                       ("nhanh_4luong", "DM cấu hình nhanh, 4 luồng CPU"), ("va", "vá hết tuyến (gồm dựng nhóm + bảng)"),
                       ("giam_tre", "giảm trễ"), ("dc_dphat", "đầu – cuối Dijkstra phạt"), ("dc_dm11", "đầu – cuối DM11"),
                       ("dc_lai11", "đầu – cuối Lai(DM11)"), ("dc_dm14", "đầu – cuối DM14"),
                       ("dc_dm14_4luong", "đầu – cuối DM14, DM 4 luồng CPU"), ("ti_le_dm14_dm11", "DM14 / DM11")):
            L.append(f"| {ten} | " + " | ".join(f"{v[k]:.2f}" if k == "ti_le_dm14_dm11" else f"{v[k]:.0f}"
                                                for v in tg["nhom"].values()) + " |")
        kk = tg["kho"]
        L += ["", f"Kho nhanh so với kho cũ (cùng cặp): Suurballe cùng chi phí {kk['chi_phi_bang']}/{kk['cap']}, tập Suurballe "
              f"trùng {kk['suurballe_trung']}/{kk['cap']}, cả kho trùng {kk['kho_trung']}/{kk['cap']}. Hình: `hinh_14_thoi_gian.png`.", ""]
    if ra["kho"]:
        L += ["## 5. Kho nhanh — chất lượng (sinh lại hai lần, DM11 3 seed)", "",
              "| Bộ | bản ghi | luồng kho trùng | mất DV kho cũ | kho nhanh | hiệu (CI) | kho ms cũ → nhanh |",
              "|---|---:|---:|---:|---:|---|---|"]
        for b, x in ra["kho"].items():
            if "loi" in x:
                L.append(f"| {TEN_BO[b]} | lỗi: {x['loi']} | | | | | |")
                continue
            L.append(f"| {TEN_BO[b]} | {x['n']} | {x['kho_trung_luong']:.1%} | {x['mat_dv_cu']:.2f} | {x['mat_dv_nhanh']:.2f} | "
                     f"{ci(x['hieu'])} | {x['ms_kho_cu']:.0f} → {x['ms_kho_nhanh']:.0f} |")
        L.append("")
    L += [f"Kiểm lặp lại DM11 so với TN11: {ra['lap_lai']}.", "", "Hình: `hinh_14_va.png`."]
    (RA / "KET_QUA_14.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def lenh_phan_tich(a):
    T = {b: rs for b in BO_45 if (rs := t11._gop(RA / "cham_tep", t11.thu_muc_bo(b, "4-5")))}
    if not T:
        raise SystemExit("chưa có kết quả chấm — chạy bước cham")
    bo, tg, kho = phan_tich_bo(T), phan_tich_thoi_gian(), phan_tich_kho()
    ra = {"bo": bo, "thoi_gian": tg, "kho": kho, "chon_val": doc_chon_val(), "lap_lai": kiem_lap_lai(T)}
    ra["gia_thuyet"] = gia_thuyet(bo, tg, kho)
    RA.mkdir(parents=True, exist_ok=True)
    (RA / "ket_qua_14.json").write_text(json.dumps(ra, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    ve_va(bo)
    ve_thoi_gian(tg)
    viet_md(ra)
    print("  " + "; ".join(f"{k}: {'ĐẠT' if v['dat'] else 'KHÔNG ĐẠT'}" for k, v in ra["gia_thuyet"].items()
                           if k.startswith("H")), flush=True)
    dong_goi()


def dong_goi():
    """Một tệp để gửi lại: kết quả + tệp đệm chấm (cham_tep, val_tep, kho_tep) + log. Không có mô hình mới."""
    import zipfile
    tep = KQ / "TN14_ket_qua.zip"
    gom = [(f, Path(RA.name) / f.relative_to(RA)) for f in sorted(RA.rglob("*")) if f.is_file()]
    gom += [(f, Path("logs") / f.name) for f in sorted(LOG.glob("tn14*.log"))]
    with zipfile.ZipFile(tep, "w", zipfile.ZIP_DEFLATED) as z:
        for f, t in gom:
            z.write(f, str(t))
    print(f"  đã gói {len(gom)} tệp → {tep} ({tep.stat().st_size / 1e6:.1f} MB) — gửi tệp này", flush=True)


# ============================================================================================ chạy
BUOC = (("nhom", lenh_nhom), ("chon-val", lenh_chon_val), ("thoi-gian", lenh_thoi_gian), ("cham", lenh_cham),
        ("kho", lenh_kho), ("phan-tich", lenh_phan_tich))


def lenh_toi_da(a):
    p = t11.tai_nguyen(a.ram, a.vram)
    print(f"máy: {p['cpu']} luồng CPU, ngân sách RAM {p['ram']} GB → sinh / dựng nhóm {p['sinh']} tiến trình, chấm "
          f"{p['cham']} tiến trình (TN14 không học, không cần GPU)", flush=True)
    a.tien_trinh, a.tien_trinh_sinh = p["cham"], p["sinh"]
    loi = []
    for ten, f in BUOC:
        print(f"=== {ten} {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
        if ten == "phan-tich" and loi:
            print(f"  (các bước lỗi: {loi} — phân tích phần đã xong)", flush=True)
        try:
            f(a)
        except (Exception, SystemExit) as e:
            loi.append(ten)
            print(f"  LỖI bước {ten}: {e}", flush=True)
            if ten in ("nhom", "chon-val"):                 # các bước sau cần hai bước này
                break
    if loi:
        raise SystemExit(f"bước lỗi: {loi} — sửa rồi chạy lại đúng lệnh (phần đã xong được giữ)")


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("viec", choices=[b for b, _ in BUOC] + ["toi-da"])
    ap.add_argument("--tien-trinh", type=int, default=4, help="số tiến trình chấm")
    ap.add_argument("--tien-trinh-sinh", type=int, default=2, help="số tiến trình sinh / dựng nhóm (~4 GB RAM mỗi tiến trình)")
    ap.add_argument("--ram", type=float, default=20.0, help="ngân sách RAM (GB) cho toi-da")
    ap.add_argument("--vram", type=float, default=0.0, help="không dùng (TN14 không học)")
    ap.add_argument("--tn11", default=None, help="thư mục repo đã chạy TN11 hoặc tệp TN11_cho_TN12_TN13.zip")
    a = ap.parse_args()
    t12.mo_log("tn14", sys.argv[1:])
    if a.tn11:
        t12.lay_tn11(a.tn11)
    if a.viec == "toi-da":
        lenh_toi_da(a)
    else:
        dict(BUOC)[a.viec](a)
    print("xong", flush=True)


if __name__ == "__main__":
    main()
