"""
tn11a.py — Thí nghiệm 11a (results/bao_da_tuyen/TIEU_CHI_11A.md): chẩn đoán khoảng cách DM10 → thầy. Chạy trên máy đã
chạy TN10 (dùng lại train10 / train6_10, val, val6 và mô hình mo_hinh10), không sinh nhãn mới, không chấm bộ kiểm tra:

    python -m scripts.tn11a tat-ca --tien-trinh 8 --song-song 3 --luong-cpu 5          # ~2 giờ trên máy 20 luồng

Từng bước (chạy lại an toàn: phần đã xong tự bỏ qua):
    bo-sung     thêm 3 đặc trưng nhìn trước (tương lai thật) vào train10, train6_10, val, val6 — chỉ THÊM khoá
    huan-luyen  nt60h128_p0..3 = e60h128 của TN10 (cùng seed / phần / dữ liệu) + --nhin-truoc → mo_hinh11a/
    cham        A: kiểm thử chéo nt60h128 (S 8) · B: e120h128 trên phần ĐÃ học (S 8) · C: DM10 (e120h128, S 32) + TKCB
    phan-tich   → results/bao_da_tuyen/tn11a/{ket_qua_11a.json, KET_QUA_11A.md}
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import pickle
import subprocess
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

GOC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(GOC))
from scripts.tn10 import B_BOOT, SO_BUOC_KHU, TAP_A, bootstrap  # noqa: E402

KQ = GOC / "results" / "bao_da_tuyen"
RA = KQ / "tn11a"
M10, M11 = KQ / "mo_hinh10", KQ / "mo_hinh11a"
CV10 = KQ / "tn10" / "kiem_thu_cheo"
DL = GOC / "data" / "bao_da_tuyen"
LOG = GOC / "logs"
BO_SUNG = {"train10_k4_5": "train10", "train6_10_k4_5": "train6_10", "val_k4_5": "val", "val6_k4_5": "val6"}
NT = "nt60h128"
CAC_TAU = (1.1, 1.3, 1.5, float("inf"))
TAU_GOC = 1.1


# ----------------------------------------------------------------------------------------------- A0. bổ sung đặc trưng
def _bo_sung_tep(job):
    """Dựng lại kịch bản cùng seed (như scripts.bo_sung_dac_trung), tính đặc trưng tương lai thật cho từng cạnh."""
    from src.ngau_nhien.kich_ban import sinh_kich_ban
    from src.ngau_nhien.ma_tran_rui_ro import ma_tran_online
    from src.nhieu_luong import tao_du_lieu_rr as td
    from src.nhieu_luong import the_gioi_rr as tg
    from src.nhieu_luong.bai_toan_rr import DELTA, TUOI
    from src.nhieu_luong.mo_hinh_dt import NHIN_TRUOC_11
    from src.nhieu_luong.tao_du_lieu_dt import cac_cua_so_tap, moi_truong
    tap, tep = job
    if td._g is None:
        td._khoi_tao()
    recs = pickle.load(open(tep, "rb"))

    def dem():                                   # [số cạnh·bản ghi, số > 0 của từng đặc trưng nhìn trước]
        x = [np.concatenate([r["gon"].dac_trung_canh[k] for r in recs]) for k in NHIN_TRUOC_11]
        return [int(len(x[0]))] + [int((v > 0).sum()) for v in x]

    if not recs or NHIN_TRUOC_11[0] in recs[0]["gon"].dac_trung_canh:
        return tep, 0, (dem() if recs else [0] * 4)
    g, p = td._g, td._p
    tt, p_sim = moi_truong(tap)
    _, h0, seed = cac_cua_so_tap(tap)[recs[0]["kb"]]
    Wn, q = sinh_kich_ban(g, tt, h0, seed, p_sim)
    pl = tg.phat_lai(g, tt, h0, seed, p_sim)
    ns, kind = np.asarray(g.node_site), np.asarray(g.kind)
    dat = (ns >= 0) & (kind != "uav") & np.asarray(g.dat, bool)          # trạm mặt đất bị sét ngắt (như re_nhanh)
    canh = {}
    for r in recs:
        t = int(r["t"])
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
        cs, rn = slice(t, t + DELTA + 1), slice(o + 1, t + DELTA + 1)    # cửa sổ giữ tuyến / tầm rẽ nhánh của thầy
        site = g.site[u, v]
        mua = np.where(site >= 0, pl["P"][cs][:, np.maximum(site, 0)].max(0), 0.0)   # đỉnh: đứt khi một mốc vượt ngưỡng
        gd = np.zeros(g.N)
        gd[dat] = pl["giong_diem"][rn][:, ns[dat]].mean(0)
        ut = pl["uav_tat"][cs].mean(0)
        gon.dac_trung_canh.update({"nt_mua": (mua / 20).astype(np.float32),
                                   "nt_giong_dat": np.maximum(gd[u], gd[v]).astype(np.float32),
                                   "nt_uav_tat": np.maximum(ut[u], ut[v]).astype(np.float32)})
    tam = tep + ".tam"
    with open(tam, "wb") as f:
        pickle.dump(recs, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tam, tep)
    return tep, len(recs), dem()


def lenh_bo_sung(a):
    from src.nhieu_luong import tao_du_lieu_rr as td
    jobs = [(tap, f) for d, tap in BO_SUNG.items() for f in sorted(glob.glob(str(DL / d / "kb_*.pkl")))]
    thieu = [d for d in BO_SUNG if not glob.glob(str(DL / d / "kb_*.pkl"))]
    if thieu:
        raise SystemExit(f"thiếu dữ liệu TN10: {thieu} — chạy trên máy đã chạy TN10")
    print(f"bổ sung đặc trưng nhìn trước: {len(jobs)} tệp", flush=True)
    t0, tong = time.time(), {}
    with Pool(a.tien_trinh, initializer=td._khoi_tao) as pool:
        for k, (tep, m, d) in enumerate(pool.imap_unordered(_bo_sung_tep, jobs)):
            tong[Path(tep).parent.name] = (np.array(tong.get(Path(tep).parent.name, [0] * 4)) + d).tolist()
            if k % 20 == 0 or k == len(jobs) - 1:
                print(f"  {k + 1}/{len(jobs)}  {Path(tep).parent.name}/{Path(tep).name}  {m} bản ghi  "
                      f"({time.time() - t0:.0f} s)", flush=True)
    from src.nhieu_luong.mo_hinh_dt import NHIN_TRUOC_11
    ra = {d: {"so_canh_ban_ghi": x[0], **{f"ti_le_{k}_>0": x[1 + j] / max(x[0], 1) for j, k in enumerate(NHIN_TRUOC_11)}}
          for d, x in tong.items()}
    RA.mkdir(parents=True, exist_ok=True)
    (RA / "bo_sung.json").write_text(json.dumps(ra, ensure_ascii=False, indent=1), encoding="utf-8")
    for d, x in ra.items():
        print(f"  {d}: " + ", ".join(f"{k} {v * 100:.1f}%" if k.startswith("ti_le") else f"{k} {v}" for k, v in x.items()))


# ----------------------------------------------------------------------------------------------- A1. huấn luyện
def lenh_huan_luyen(a):
    viec = [f for f in range(4) if not (M11 / f"{NT}_p{f}.xong").exists()]
    LOG.mkdir(exist_ok=True)
    for i in range(0, len(viec), a.song_song):
        chay = []
        for f in viec[i:i + a.song_song]:
            ten = f"{NT}_p{f}"
            lenh = [sys.executable, "-W", "ignore", "-m", "src.nhieu_luong.huan_luyen_dt", "dm", "--thi-nghiem", "14",
                    "--ten", ten, "--them", "--nhin-truoc", "--seed", str(f), "--epochs", "60", "--H", "128",
                    "--luong-cpu", str(a.luong_cpu), "--thiet-bi", a.thiet_bi, "--bo-phan", str(f)]
            print(f"  {time.strftime('%H:%M:%S')} huấn luyện {ten} → logs/tn11a_{ten}.log", flush=True)
            chay.append((ten, subprocess.Popen(lenh, cwd=GOC, stderr=subprocess.STDOUT,
                                               stdout=open(LOG / f"tn11a_{ten}.log", "w", encoding="utf-8"))))
        for ten, p in chay:
            if p.wait() != 0:
                raise SystemExit(f"huấn luyện {ten} lỗi — xem logs/tn11a_{ten}.log")
            print(f"  {time.strftime('%H:%M:%S')} xong {ten}", flush=True)


# ----------------------------------------------------------------------------------------------- chấm
_NET = {}


def _cham_tep(job):
    """Mỗi bản ghi chấm bằng mô hình <ch>_p((phần + lech) mod 4): lech 0 = phần KHÔNG học nó, lech 1 = phần đã học nó.
    tkcb: thêm DM + TKCB (thế giới bốc từ bảng, như cham_lai_dt) và ρ̂. Hàng: [mất DV, J, thầy, Dijkstra phạt
    (, mất DV DM+TKCB, J DM+TKCB, ρ̂)]."""
    import torch
    from src.nhieu_luong import da_tuyen as dt
    from src.nhieu_luong import mo_hinh_dt as md
    from src.nhieu_luong.cham_lai_dt import giong_ban_ghi, rho_hat
    from src.nhieu_luong.cham_mo_hinh_dt import _tg
    from src.nhieu_luong.huan_luyen_dt import phan_cua
    tep, ch, S, mh, lech, tkcb = job
    torch.set_num_threads(1)
    thu_muc, ra = Path(tep).parent.name, []
    for r in pickle.load(open(tep, "rb")):
        ten = f"{ch}_p{(phan_cua(thu_muc, r['kb']) + lech) % 4}"
        if (mh, ten) not in _NET:
            _NET[(mh, ten)] = md.nap(ten, Path(mh))
        gon = r["gon"]
        a = md.chon_dm(_NET[(mh, ten)], gon, S, SO_BUOC_KHU, torch.Generator().manual_seed(giong_ban_ghi(r)))
        tgw = _tg(gon, r["cham"])
        c = tgw.cham(gon.tu_lua_chon(a))
        hang = [c["mat_dv_%"], c["J"], r["ens"]["thay"]["mat_dv_%"], r["ens"]["dphat"]["mat_dv_%"]]
        if tkcb:
            a_ls, _ = dt.tkcb(dt.TheGioi.tu_bang(gon, np.random.default_rng([r["kb"], r["t"], 4])), a)
            c2 = tgw.cham(gon.tu_lua_chon(a_ls))
            hang += [c2["mat_dv_%"], c2["J"], rho_hat(gon)]
        ra.append(hang)
    return tep, ra


def cham(ten_dem, ch, S, mh, lech, tkcb, tt):
    tep_dem = RA / f"{ten_dem}.json"
    if tep_dem.exists():
        return np.array(json.loads(tep_dem.read_text(encoding="utf-8")))
    teps = [t for d in TAP_A for t in sorted(glob.glob(str(DL / d / "kb_*.pkl")))]
    s = time.time()
    with Pool(tt) as pool:
        kq = dict(pool.imap_unordered(_cham_tep, [(t, ch, S, str(mh), lech, tkcb) for t in teps]))
    x = [v for t in teps for v in kq[t]]
    RA.mkdir(parents=True, exist_ok=True)
    tep_dem.write_text(json.dumps(x), encoding="utf-8")
    print(f"  {ten_dem}: mất DV {np.mean([v[0] for v in x]):.3f} ({len(x)} bản ghi, {time.time() - s:.0f} s)", flush=True)
    return np.array(x)


def cv10(ch, S, tt):
    """Kết quả kiểm thử chéo của TN10 (đã lưu); thiếu thì chấm lại bằng mô hình mo_hinh10."""
    tep = CV10 / f"{ch}_S{S}_{'+'.join(TAP_A)}.json"
    if tep.exists():
        return np.array(json.loads(tep.read_text(encoding="utf-8")))
    return cham(f"tn10_{ch}_S{S}", ch, S, M10, 0, False, tt)


def lenh_cham(a):
    for f in range(4):
        if not (M11 / f"{NT}_p{f}.xong").exists():
            raise SystemExit(f"chưa huấn luyện xong {NT}_p{f}")
    cham(f"A_{NT}_S8", NT, 8, M11, 0, False, a.tien_trinh)
    cv10("e60h128", 8, a.tien_trinh)
    cham("B_e120h128_S8_da_hoc", "e120h128", 8, M10, 1, False, a.tien_trinh)
    cv10("e120h128", 8, a.tien_trinh)
    cham("C_e120h128_S32_tkcb", "e120h128", 32, M10, 0, True, a.tien_trinh)


# ----------------------------------------------------------------------------------------------- phân tích
def _bs(v, kb):
    m, ci = bootstrap(v, kb)
    return [round(m, 4), [round(ci[0], 4), round(ci[1], 4)]]


def epoch_tot(thu_muc, ch):
    ra = []
    for f in range(4):
        t = thu_muc / f"{ch}_p{f}_nhat_ky.json"
        if t.exists():
            nk = json.loads(t.read_text(encoding="utf-8"))["nhat_ky"]
            ra.append([min(nk, key=lambda r: r["J_val"])["epoch"], len(nk)])
    return ra


def nhan():
    """Mã cửa sổ, phần, mức tải (0 thấp / 1 vừa / 2 cao) của từng bản ghi — cùng thứ tự với cham()."""
    from src.nhieu_luong.bai_toan import MUC_TAI
    from src.nhieu_luong.huan_luyen_dt import phan_cua
    ma, ph, mu = [], [], []
    for i, d in enumerate(TAP_A):
        for t in sorted(glob.glob(str(DL / d / "kb_*.pkl"))):
            for r in pickle.load(open(t, "rb")):
                ma.append(i * 10_000 + int(r["kb"]))
                ph.append(phan_cua(d, r["kb"]))
                mu.append(list(MUC_TAI).index(r["muc"]))
    return np.array(ma), np.array(ph), np.array(mu)


def lenh_phan_tich(a):
    tt = a.tien_trinh
    kb, ph, muc = nhan()
    nt, e60 = cham(f"A_{NT}_S8", NT, 8, M11, 0, False, tt), cv10("e60h128", 8, tt)
    da, e120 = cham("B_e120h128_S8_da_hoc", "e120h128", 8, M10, 1, False, tt), cv10("e120h128", 8, tt)
    C = cham("C_e120h128_S32_tkcb", "e120h128", 32, M10, 0, True, tt)
    for X in (e60, da, e120, C):                          # cùng thứ tự bản ghi (thầy, Dijkstra phạt trùng)
        assert len(X) == len(nt) and np.allclose(X[:, 2:4], nt[:, 2:4]), "thứ tự bản ghi không khớp"
    G = lambda X, m: float((X[m, 0] - X[m, 2]).mean())
    ra = {"so_ban_ghi": len(nt), "thay": float(nt[:, 2].mean()), "dphat": float(nt[:, 3].mean())}
    # nửa train = cửa sổ cách 3 giờ của 7 cơn bão (rò theo bão giữa các phần); nửa train6 = 60 ngày xấu khác nhau → SỐ CHÍNH
    nua = {"train6 (số chính)": kb >= 10_000, "train (7 cơn bão)": kb < 10_000, "cả hai": np.ones(len(kb), bool)}

    # A
    d = nt[:, 0] - e60[:, 0]

    def phan_A(m):
        dG = _bs(d[m], kb[m])
        return {"mat_dv_nt": float(nt[m, 0].mean()), "mat_dv_e60": float(e60[m, 0].mean()), "G_nt": G(nt, m),
                "G_e60": G(e60, m), "hieu": dG, "hieu_J": _bs(nt[m, 1] - e60[m, 1], kb[m]),
                "phan_do_thong_tin": -dG[0] / G(e60, m), "theo_muc": [float(d[m & (muc == j)].mean()) for j in range(3)]}

    A = {k: phan_A(m) for k, m in nua.items()}
    chinh = A["train6 (số chính)"]
    dG, phan = chinh["hieu"], chinh["phan_do_thong_tin"]
    if dG[1][1] < 0 and phan >= 0.5:
        kl, huong = "thông tin là nút thắt chính", "đặc trưng dự báo hợp lệ ở o + tinh chỉnh DM theo mất DV"
    elif dG[1][1] < 0 and phan >= 0.2:
        kl, huong = "thông tin là một phần đáng kể", "cả hai: đặc trưng dự báo + tinh chỉnh theo mất DV"
    else:
        kl, huong = "không phải chủ yếu do thông tin", "tinh chỉnh theo mất DV + đổi kiến trúc"
    ra["A"] = {"theo_nua": A, "theo_phan": [float(d[ph == f].mean()) for f in range(4)],
               "epoch_tot_nt": epoch_tot(M11, NT), "epoch_tot_e60": epoch_tot(M10, "e60h128"),
               "ket_luan": kl, "huong_11b": huong}

    # B
    def phan_B(m):
        gd, gk = G(da, m), G(e120, m)
        return {"G_da_hoc": gd, "G_kiem_thu_cheo": gk, "ti_so": gd / gk, "hieu": _bs(da[m, 0] - e120[m, 0], kb[m]),
                "doc": ("học thuộc: khớp tình huống đã học, không tổng quát hoá" if gd <= 0.5 * gk else
                        "không khớp nổi cả tình huống đã học" if gd >= 0.8 * gk else "ở giữa")}

    ra["B"] = {k: phan_B(m) for k, m in nua.items()}

    # C
    dm, ls, rho = C[:, 0], C[:, 4], C[:, 6]
    lai = {t: np.where(rho < t, dm, ls) for t in CAC_TAU}
    laiJ = {t: np.where(rho < t, C[:, 1], C[:, 5]) for t in CAC_TAU}
    bang = {str(t): {"mat_dv": float(lai[t].mean()), "J": float(laiJ[t].mean()), "ti_le_tkcb": float((rho >= t).mean()),
                     "hieu_vs_1.1": _bs(lai[t] - lai[TAU_GOC], kb), "hieu_J_vs_1.1": _bs(laiJ[t] - laiJ[TAU_GOC], kb)}
            for t in CAC_TAU}
    tot = min(CAC_TAU, key=lambda t: bang[str(t)]["mat_dv"])
    chon = TAU_GOC if bang[str(tot)]["hieu_J_vs_1.1"][1][0] > 0 else tot
    ra["C"] = {"bang": bang, "tau_tot_mat_dv": tot, "tau_chon": chon, "dm_thuan_vs_lai11": _bs(dm - lai[TAU_GOC], kb),
               "mat_dv_dm32": float(dm.mean()), "mat_dv_dm_tkcb": float(ls.mean()),
               "theo_nua_mat_dv": {k: {str(t): float(lai[t][m].mean()) for t in CAC_TAU} for k, m in nua.items()}}
    tep_bs = RA / "bo_sung.json"
    if tep_bs.exists():
        ra["bo_sung"] = json.loads(tep_bs.read_text(encoding="utf-8"))
    RA.mkdir(parents=True, exist_ok=True)
    (RA / "ket_qua_11a.json").write_text(json.dumps(ra, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    viet_md(ra)
    print(f"  A (train6): {kl} (phần do thông tin {phan * 100:.0f}%); B (train6): {ra['B']['train6 (số chính)']['doc']}; "
          f"C: τ = {chon}", flush=True)


def _ci(x, nd=2):
    return f"{x[0]:+.{nd}f} [{x[1][0]:+.{nd}f}; {x[1][1]:+.{nd}f}]"


def viet_md(ra):
    A, B, C = ra["A"], ra["B"], ra["C"]
    L = ["# Thí nghiệm 11a — kết quả (sinh tự động bởi `scripts/tn11a.py`)", "",
         f"Kiểm thử chéo 4 phần trên {ra['so_ban_ghi']} tình huống của train10 + train6_10 (thầy {ra['thay']:.2f}, "
         f"Dijkstra phạt {ra['dphat']:.2f}). Mất DV bằng %, hiệu bằng điểm %, CI 95% bootstrap theo cửa sổ "
         f"({B_BOOT} lần). Tiêu chí: TIEU_CHI_11A.md.", "",
         "## A. DM nhìn trước (phần khoảng cách do thông tin)", "",
         *([("Tỉ lệ cạnh·bản ghi có đặc trưng nhìn trước > 0: " + "; ".join(
             f"{d}: " + ", ".join(f"{k[6:-3]} {v * 100:.1f}%" for k, v in x.items() if k.startswith("ti_le"))
             for d, x in ra["bo_sung"].items()) + "."), ""] if "bo_sung" in ra else []),
         "Nửa train6 (60 ngày xấu khác nhau) là số chính; nửa train là cửa sổ cách 3 giờ của 7 cơn bão (rò theo bão "
         "giữa các phần).", "",
         "| Nửa | Mất DV e60h128 → nt60h128 | G e60h128 → nt60h128 | Hiệu mất DV | Hiệu J | Thấp / giữa / cao | "
         "Phần do thông tin |", "|---|---|---|---|---|---|---:|"]
    for k, x in A["theo_nua"].items():
        L.append(f"| {k} | {x['mat_dv_e60']:.2f} → {x['mat_dv_nt']:.2f} | {x['G_e60']:.2f} → {x['G_nt']:.2f} | "
                 f"{_ci(x['hieu'])} | {_ci(x['hieu_J'], 3)} | {', '.join(f'{v:+.2f}' for v in x['theo_muc'])} | "
                 f"{x['phan_do_thong_tin'] * 100:.0f}% |")
    L += ["", f"- Theo phần (cả hai nửa): {', '.join(f'{x:+.2f}' for x in A['theo_phan'])}. Epoch tốt nhất: e60h128 "
          f"{A['epoch_tot_e60']}, nt60h128 {A['epoch_tot_nt']}.",
          f"- **Kết luận (nửa train6, cận dưới): {A['ket_luan']}.** Hướng TN11b: {A['huong_11b']}.", "",
          "## B. G trên tình huống đã học (e120h128, S = 8)", "",
          "| Nửa | G đã học | G kiểm thử chéo | Tỉ số | Hiệu | Đọc |", "|---|---:|---:|---:|---|---|"]
    for k, x in B.items():
        L.append(f"| {k} | {x['G_da_hoc']:.2f} | {x['G_kiem_thu_cheo']:.2f} | {x['ti_so']:.2f} | {_ci(x['hieu'])} | "
                 f"{x['doc']} |")
    L += ["",
         "## C. Ngưỡng ρ̂ của Lai cho DM10 (e120h128, S = 32)", "",
         "| τ | Mất DV | J | Tỉ lệ sang TKCB | Hiệu mất DV so với 1,1 | Hiệu J so với 1,1 |", "|---|---:|---:|---:|---|---|"]
    for t, b in C["bang"].items():
        L.append(f"| {t} | {b['mat_dv']:.2f} | {b['J']:.3f} | {b['ti_le_tkcb'] * 100:.0f}% | {_ci(b['hieu_vs_1.1'])} | "
                 f"{_ci(b['hieu_J_vs_1.1'], 3)} |")
    L += ["", "- Mất DV theo nửa: " + "; ".join(f"{k}: " + ", ".join(f"τ {t} {v:.2f}" for t, v in x.items())
                                             for k, x in C["theo_nua_mat_dv"].items()) + ".",
          f"- DM thuần − Lai(1,1): {_ci(C['dm_thuan_vs_lai11'])}.",
          f"- **τ chọn: {C['tau_chon']}** (mất DV thấp nhất: {C['tau_tot_mat_dv']}).", ""]
    (RA / "KET_QUA_11A.md").write_text("\n".join(L), encoding="utf-8")


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("viec", choices=["bo-sung", "huan-luyen", "cham", "phan-tich", "tat-ca"])
    ap.add_argument("--tien-trinh", type=int, default=4, help="số tiến trình bổ sung / chấm")
    ap.add_argument("--song-song", type=int, default=2, help="số mô hình huấn luyện cùng lúc")
    ap.add_argument("--luong-cpu", type=int, default=4, help="luồng PyTorch mỗi mô hình")
    ap.add_argument("--thiet-bi", default="cpu", choices=["cpu", "auto", "cuda"])
    a = ap.parse_args()
    buoc = {"bo-sung": [lenh_bo_sung], "huan-luyen": [lenh_huan_luyen], "cham": [lenh_cham],
            "phan-tich": [lenh_phan_tich],
            "tat-ca": [lenh_bo_sung, lenh_huan_luyen, lenh_cham, lenh_phan_tich]}[a.viec]
    for f in buoc:
        print(f"=== {f.__name__} {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
        f(a)
    print("xong", flush=True)


if __name__ == "__main__":
    main()
