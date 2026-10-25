"""
Tính Δ (nhiễu đo do mưa/giông) theo loại liên kết và mức mưa, từ hình học liên kết thật của TV1 và mưa thật ERA5.

    python test/phan_tich_nhieu_mua.py --data-dir dataset --thoi-tiet data/thoi_tiet_vn.npz --out results/nhieu_mua.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.ai.data_v2 import SagsinData  # noqa: E402
from src.ngau_nhien.kich_ban import MangGoc, suy_hao_mua  # noqa: E402
from src.ngau_nhien import nhieu_mua  # noqa: E402
from src.ngau_nhien.nhieu_mua import (KL840, L_MAY_GIONG, S_MUA, SNR0_DB, delta, he_so_may,  # noqa: E402
                                      nhieu_nen, thoi_gian_o_lai, ti_le_toc_do)
from src.ngau_nhien.thoi_tiet import GIONG, ThoiTiet  # noqa: E402

MUC = {"mưa nhẹ (0,1–2,5 mm/h)": (0.1, 2.5), "mưa vừa (2,5–10 mm/h)": (2.5, 10.0), "mưa to (> 10 mm/h)": (10.0, 1e9)}
GIA_DINH = {"mưa nhẹ (0,1–2,5 mm/h)": 0.10, "mưa vừa (2,5–10 mm/h)": 0.25, "mưa to (> 10 mm/h)": 0.40, "giông": 0.20}


def ten_loai(ten, band):
    return {("GSL", 20): "Gateway – vệ tinh (20 GHz)", ("GSL", 12): "Người dùng – vệ tinh (12 GHz)",
            ("ASL", 12): "UAV – vệ tinh (12 GHz)", ("HAPS-MĐ", 28): "HAPS – mặt đất (28 GHz)"}.get(
        (ten, band), "Liên kết UAV (5 GHz)" if band == 5 else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--thoi-tiet", required=True)
    ap.add_argument("--mau", type=int, default=4000, help="số cặp (liên kết, giờ) cho mỗi ô")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    te = SagsinData(Path(a.data_dir) / "test.npz")
    tt = ThoiTiet(a.thoi_tiet)
    g = MangGoc(te.W, te.kind, te.node_order, tt)
    s_, u_, v_ = np.nonzero((te.W > 0) & np.triu(np.ones((g.N, g.N), bool), 1)[None])
    loai = np.array([ten_loai(g.ten[u, v], g.band[u, v]) or ("" if g.band[u, v] else g.ten[u, v])
                     for u, v in zip(u_, v_)])
    giong_h = np.isin(tt.code, GIONG)
    kq = {"tham_so": {"S_MUA": S_MUA, "SNR0_DB": SNR0_DB, "L_MAY_GIONG": L_MAY_GIONG, "KL840": KL840},
          "nen": {}, "delta": {}, "do_nhay": {}}

    # nhiễu nền khi trời quang, mọi loại liên kết
    for L in sorted(set(loai)):
        m = loai == L
        x = nhieu_nen(g.C[u_[m], v_[m]], te.W[s_[m], u_[m], v_[m]])
        kq["nen"][L] = {"trung_vi_%": float(np.median(x) * 100), "p10_%": float(np.percentile(x, 10) * 100),
                        "p90_%": float(np.percentile(x, 90) * 100), "so_lien_ket": int(m.sum())}

    def mau(L, muc, n):
        idx = np.flatnonzero(loai == L)
        pick = idx[rng.integers(0, len(idx), n)]
        site = g.site[u_[pick], v_[pick]]
        R, G = np.zeros(n), np.zeros(n)
        for k, st in enumerate(site):
            if muc == "giông":
                hs = np.flatnonzero(giong_h[:, st])
            else:
                lo, hi = MUC[muc]
                hs = np.flatnonzero((tt.precip[:, st] > lo + 1e-4) & (tt.precip[:, st] <= hi + 1e-4) & ~giong_h[:, st])
            h = hs[rng.integers(0, len(hs))]
            R[k], G[k] = tt.precip[h, st], float(muc == "giông")
        s, u, v = s_[pick], u_[pick], v_[pick]
        x = dict(R_gio=R, giong=G, band=g.band[u, v], C=g.C[u, v], Ls=g.Ls[s, u, v], LG=g.LG[s, u, v], W=te.W[s, u, v])
        return x, g.sin_el[s, u, v]

    loai_mua = [L for L in dict.fromkeys(ten_loai(g.ten[u, v], g.band[u, v]) for u, v in zip(u_, v_)) if L]
    bien_the = {"cơ sở": {}, "SNR0 = 10 dB": {"snr0_db": 10.0}, "SNR0 = 20 dB": {"snr0_db": 20.0},
                "ρ0 = 0,5": {"rho": 0.5}, "s = 0,8 (mô-đun cũ)": {"s": 0.8}}
    nghen_bt = {"ngưỡng nghẽn 0,8": 0.8, "ngưỡng nghẽn 0,95": 0.95}
    for L in loai_mua:
        kq["delta"][L] = {}
        for muc in list(MUC) + ["giông"]:
            x, sin_el = mau(L, muc, a.mau)
            d, nghen = delta(**x)
            dd = np.where(np.isfinite(d), d, np.inf)            # nghẽn suốt giờ: xếp cuối khi lấy phân vị
            kq["delta"][L][muc] = {"trung_vi": float(np.median(dd)), "p25": float(np.percentile(dd, 25)),
                                   "p75": float(np.percentile(dd, 75)), "p90": float(np.percentile(dd, 90)),
                                   "nghen_%": float(nghen.mean() * 100),
                                   "R_trung_vi": float(np.median(x["R_gio"])), "goc_ngang_trung_vi_do": float(np.degrees(np.arcsin(np.median(sin_el)))),
                                   "Ls_trung_vi_km": float(np.median(x["Ls"])), "C_Mbps": float(np.median(x["C"])),
                                   "W_trung_vi": float(np.median(x["W"]))}
            if muc == "mưa vừa (2,5–10 mm/h)":
                for bt, kw in bien_the.items():
                    d2, n2 = delta(**x, **kw)
                    kq["do_nhay"].setdefault(L, {})[bt] = [float(np.median(np.where(np.isfinite(d2), d2, np.inf))),
                                                          float(n2.mean() * 100)]
                for bt, ng in nghen_bt.items():
                    cu, nhieu_mua.NGHEN = nhieu_mua.NGHEN, ng
                    d2, n2 = delta(**x)
                    nhieu_mua.NGHEN = cu
                    kq["do_nhay"][L][bt] = [float(np.median(np.where(np.isfinite(d2), d2, np.inf))), float(n2.mean() * 100)]
            print(f"{L:32s} {muc:24s} Δ trung vị {kq['delta'][L][muc]['trung_vi']:.3f} "
                  f"[{kq['delta'][L][muc]['p25']:.3f}; {kq['delta'][L][muc]['p75']:.3f}]  nghẽn {nghen.mean() * 100:.1f}%",
                  flush=True)

    # đường cong Δ theo lượng mưa giờ (không giông), trung vị trên mọi liên kết của từng loại
    luoi = np.round(np.geomspace(0.2, 40, 25), 3)
    kq["duong_cong"] = {"R": luoi.tolist()}
    for L in loai_mua:
        m = np.flatnonzero(loai == L)[:400]
        u, v, s = u_[m], v_[m], s_[m]
        dd, nn = [], []
        for R in luoi:
            d, ng = delta(np.full(len(m), R), np.zeros(len(m)), g.band[u, v], g.C[u, v], g.Ls[s, u, v], g.LG[s, u, v],
                          te.W[s, u, v])
            dd.append(float(np.median(np.where(np.isfinite(d), d, np.inf))))
            nn.append(float(ng.mean() * 100))
        kq["duong_cong"][L] = {"delta_tv": dd, "nghen_%": nn}

    # thống kê mưa thật tại 13 điểm mặt đất
    dat = [i for i, n in enumerate(tt.names) if not n.startswith(("HAPS", "UAV"))]
    P, Gh = tt.precip[:, dat], giong_h[:, dat]
    kq["mua"] = {"gio_mua_%": float((P > 0.1001).mean() * 100),
                 **{f"{m}_%": float(((P > lo + 1e-4) & (P <= hi + 1e-4)).mean() * 100) for m, (lo, hi) in MUC.items()},
                 "gio_giong_%": float(Gh.mean() * 100),
                 "R_trung_vi_gio_giong": float(np.median(P[Gh & (P > 0.1001)])),
                 "R_trung_vi_gio_mua_khong_giong": float(np.median(P[~Gh & (P > 0.1001)])),
                 "gio_giong_co_mua_%": float((P[Gh] > 0.1001).mean() * 100)}

    # ví dụ tính tay: gateway – vệ tinh 20 GHz, góc ngẩng 40°, mưa 5 mm/h, W = 3 ms, C = 100 Mbps
    el, R, C, W, H_R = np.radians(40), 5.0, 100.0, 3.0, 5.0
    Ls = H_R / np.sin(el)
    LG = Ls * np.cos(el)
    A = float(suy_hao_mua(np.array(R), np.array(20), np.array(Ls), np.array(LG)))
    c = float(ti_le_toc_do(A))
    m0, m = float(thoi_gian_o_lai(C, 1.0)), float(thoi_gian_o_lai(C, c))
    kq["vi_du"] = {"Ls_km": Ls, "LG_km": LG, "L0_km": 35 * np.exp(-0.015 * R), "gamma_dB_km": 0.09388 * R ** 1.0199,
                   "A_dB": A, "snr_db": SNR0_DB - A, "c": c, "m0_ms": m0, "m_ms": m,
                   "delta_khong_dao_dong": float(np.sqrt(m ** 2 - m0 ** 2) / W),
                   "delta_co_dao_dong": float(delta(R, 0, 20, C, Ls, LG, W)[0]),
                   "A_may_dB": float(he_so_may(np.array(20)) * L_MAY_GIONG / 5.0 * Ls),
                   "A_nghen_dB": float(SNR0_DB - 10 * np.log10(2 ** (0.3 / 0.9 * np.log2(1 + 10 ** (SNR0_DB / 10))) - 1)),
                   "p_mat_co_dao_dong": float(delta(R, 0, 20, C, Ls, LG, W)[1])}
    kq["gia_dinh_cu"] = GIA_DINH
    json.dump(kq, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps({k: kq[k] for k in ("nen", "mua", "vi_du", "do_nhay")}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
