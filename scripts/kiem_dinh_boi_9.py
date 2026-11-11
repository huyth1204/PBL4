"""
kiem_dinh_boi_9.py — Thí nghiệm 9, mục B1 (results/bao_da_tuyen/TIEU_CHI_9.md): hiệu chỉnh so sánh bội cho số ô thắng
của TN4 (9 ô, DM), TN6 (27 ô, Lai6), TN7a (18 ô, Lai6 trên thế giới cực đoan WH).

    python -m scripts.kiem_dinh_boi_9          # → results/bao_da_tuyen/tn9/kiem_dinh_boi.json

p một phía = tỉ lệ mẫu bootstrap (ghép cặp theo cửa sổ, 10 000 lần, như các TN trước) có hiệu ≥ 0; ô thắng khi C2 và C4
cùng đạt nên p_ô = max(p_C2, p_C4) (kiểm định giao – hợp); C5 (thời gian, 10 / 20 luồng) là điều kiện xác định, đọc từ
kết quả đã lưu. α = 0,025 một phía ⇔ luật "CI 95% hai phía < 0". Họ = các ô của một thí nghiệm; Holm (FWER) và
Benjamini–Hochberg (FDR). Kiểm tra: số ô thắng chưa hiệu chỉnh phải bằng số đã báo ở thí nghiệm gốc.
"""

from __future__ import annotations

import glob
import json
import pickle
import sys
from pathlib import Path

import numpy as np

GOC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(GOC))

from scripts.phan_tich_5 import CAC_K, DAI, nap  # noqa: E402
from scripts.phan_tich_dt import B  # noqa: E402
from src.nhieu_luong.bai_toan import MUC_TAI  # noqa: E402

KQ = GOC / "results" / "bao_da_tuyen"
RA = KQ / "tn9"
DL = GOC / "data" / "bao_da_tuyen"
ALPHA, TAU = 0.025, 1.1


def p_mot_phia(v, kb, rng):
    """P(trung bình bootstrap ≥ 0), cùng cách lấy mẫu với phan_tich_dt.bootstrap."""
    u = np.unique(kb)
    tong = np.array([v[kb == k].sum() for k in u])
    dem = np.array([(kb == k).sum() for k in u])
    j = rng.integers(0, len(u), (B, len(u)))
    m = tong[j].sum(1) / dem[j].sum(1)
    return float((m >= 0).mean()), float(v.mean()), float(np.percentile(m, 97.5))


def holm(p, a=ALPHA):
    p = np.asarray(p)
    dat = np.zeros(len(p), bool)
    for i, j in enumerate(np.argsort(p, kind="stable")):
        if p[j] > a / (len(p) - i):
            break
        dat[j] = True
    return dat


def bh(p, a=ALPHA):
    p = np.asarray(p)
    o = np.argsort(p, kind="stable")
    ok = np.flatnonzero(p[o] <= a * np.arange(1, len(p) + 1) / len(p))
    dat = np.zeros(len(p), bool)
    if len(ok):
        dat[o[:ok[-1] + 1]] = True
    return dat


def lai6(r):
    return r["ens"]["dm6" if r["rho_hat"] < TAU else "dm6_tkcb"]


def o_cua(rs, pp, rng):
    kb = np.array([r["kb"] for r in rs])
    v = lambda k: np.array([(lai6(r) if pp == "lai6" else r["ens"][pp])[k] - r["ens"]["dphat"][k] for r in rs])
    pJ, mJ, uJ = p_mot_phia(v("J"), kb, rng)
    pL, mL, uL = p_mot_phia(v("mat_dv_%"), kb, rng)
    return {"n": len(rs), "dJ": mJ, "p_C2": pJ, "dL": mL, "p_C4": pL, "p": max(pJ, pL), "ci_tren_C2": uJ, "ci_tren_C4": uL}


def ho_tn4(rng):
    goc = json.loads((KQ / "ket_qua.json").read_text(encoding="utf-8"))
    ra = {}
    for kd in ("k4_5", "k10", "k20"):
        recs = [r for f in sorted(glob.glob(str(DL / f"test_{kd}" / "kb_*.pkl"))) for r in pickle.load(open(f, "rb"))]
        for muc in MUC_TAI:
            o = o_cua([r for r in recs if r["muc"] == muc], "dm", rng)
            tc = goc[kd][muc]["tieu_chi"]
            o["C5"] = kd == "k4_5" or bool(tc["C5_thoi_gian"])
            o["thang_goc"] = bool(tc["thang"])
            ra[f"Yagi_{kd}_{muc}"] = o
    return ra


def ho_tn6(rng):
    goc = json.loads((KQ / "ket_qua_6.json").read_text(encoding="utf-8"))
    ra = {}
    for bo in ("c6", "mua5", "yagi5"):
        for kd in CAC_K:
            recs = nap(bo, kd)
            for d, (lo, hi) in DAI.items():
                o = o_cua([r for r in recs if lo <= r["rho_hat"] < hi], "lai6", rng)
                tc = goc[f"{bo}_{kd}"][d]["tieu_chi"]
                o["C5"] = kd == "k4_5" or bool(tc["C5"])
                o["thang_goc"] = bool(tc["thang"])
                ra[f"{bo}_{kd}_{d}"] = o
    return ra


def ho_tn7a(rng):
    goc = json.loads((KQ / "ket_qua_7.json").read_text(encoding="utf-8"))["7a"]
    ra = {}
    for bo in ("c7wh", "y7wh"):
        for kd in CAC_K:
            recs = nap(bo, kd)
            for d, (lo, hi) in DAI.items():
                o = o_cua([r for r in recs if lo <= r["rho_hat"] < hi], "lai6", rng)
                tc = goc[f"{bo}_{kd}"][d]
                o["C5"] = kd == "k4_5" or bool(tc["dat_C5"])
                o["thang_goc"] = bool(tc["thang"])
                ra[f"{bo}_{kd}_{d}"] = o
    return ra


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    rng = np.random.default_rng(0)
    RA.mkdir(parents=True, exist_ok=True)
    ra = {}
    for ten, f in (("TN4", ho_tn4), ("TN6", ho_tn6), ("TN7a", ho_tn7a)):
        o = f(rng)
        k = list(o)
        p = np.array([o[x]["p"] for x in k])
        c5 = np.array([o[x]["C5"] for x in k])
        thuong, h, b = (p <= ALPHA) & c5, holm(p) & c5, bh(p) & c5
        for i, x in enumerate(k):
            o[x].update({"thang": bool(thuong[i]), "thang_holm": bool(h[i]), "thang_bh": bool(b[i])})
        ra[ten] = {"so_o": len(k), "thang": int(thuong.sum()), "thang_holm": int(h.sum()), "thang_bh": int(b.sum()),
                   "thang_goc": int(sum(o[x]["thang_goc"] for x in k)), "o": o}
        print(f"{ten}: {len(k)} ô — thắng {thuong.sum()} (gốc {ra[ten]['thang_goc']}), Holm {h.sum()}, BH {b.sum()}",
              flush=True)
    json.dump(ra, open(RA / "kiem_dinh_boi.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
