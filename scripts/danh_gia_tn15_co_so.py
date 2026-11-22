"""
danh_gia_tn15_co_so.py — Đặt cơ sở định lượng cho các chẩn đoán "cảm tính" trong DANH_GIA_TN15.md (mục 9). Không học, không chấm
lại; dùng lại bản ghi chấm TN15 / TN13 và dữ liệu 11 bộ kiểm tra như scripts.danh_gia_tn15.

    python -m scripts.danh_gia_tn15_co_so        # ~6 phút (bước đặc trưng ~3 phút, có đệm)

Ra: results/bao_da_tuyen/danh_gia_tn15/co_so.json (+ dac_trung_b.json đệm).

 C1  độ phủ thật của CI theo bộ (13 cụm): bootstrap phân vị vs t — mô phỏng bốc 13 cụm từ 143 cụm thật;
 C2  số bộ độc lập hiệu dụng: bootstrap chung theo cụm thời tiết dùng chung → ma trận tương quan 11 × 11 → tỉ số tham gia;
 C3  độ bất định "giữa các cơn bão" mà CI của Yagi (một cơn bão) bỏ qua: ước lượng từ 13 đợt bão khác nhau của bao9;
 C4  x10 khác DM11 có hơn mức DM11 khác chính nó không; cỡ hiệu ứng chuẩn hoá theo tình huống;
 C5  DM11 lệch đường nối w0,5 – w0,9 bao nhiêu so với độ dao động seed;
 C6  cơ chế khi đã kiểm soát mức trễ: hồi quy hiệu ~ log X_eff + log trễ DM11, CI bootstrap theo cửa sổ thời tiết; bảng chéo;
 C7  X_eff tính từ thế giới chấm (thay cho bảng) có cho cùng kết luận không;
 C8  trễ tuyến chính > 100 ms có buộc phải có liên kết tải > 0,99 không (cận trên của trễ khi mọi liên kết ở tải 0,99);
 C9  đuôi trễ và quá tải (b): mức đi cùng nhau và phần cải thiện có đến từ cùng tình huống không;
 C10 chọn seed tốt nhất (cận trên kiểu biết trước);
 C11 phần giảm trễ đến từ tình huống "lợi cả hai" hay "đánh đổi";
 C12 nhiều phép thử: p của các hiệu gộp chính, ngưỡng Bonferroni / Holm;
 C13 lệch do chọn ktn sau khi xem bộ kiểm tra; C14 H15.0 khi hiệu chỉnh cho 11 mục tiêu đã so trên cùng val.
"""

from __future__ import annotations

import json
import pickle
import sys
from math import erf, sqrt
from pathlib import Path

import numpy as np

GOC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(GOC))
from scripts import danh_gia_tn15 as dg  # noqa: E402
from src.nhieu_luong import da_tuyen as dt  # noqa: E402
from src.nhieu_luong.bai_toan import h_hang_doi  # noqa: E402

MAT, A, B, TRE, S, BO = dg.MAT, dg.A, dg.B, dg.TRE, dg.S, dg.BO
Phi = lambda z: 0.5 * (1 + erf(z / sqrt(2)))


def dac_trung_b():
    """X_eff theo thế giới chấm; cận trên trễ ở tải 0,99 của mọi tuyến kho; nguồn của mỗi tình huống."""
    tep = dg.RA / "dac_trung_b.json"
    if tep.exists():
        return json.loads(tep.read_text(encoding="utf-8"))
    h099 = float(h_hang_doi(np.array(0.99)))
    ra = {}
    for b in BO:
        dong = []
        for f in sorted((GOC / "data" / "bao_da_tuyen" / f"{b}_k4_5").glob("kb_*.pkl")):
            for r in pickle.load(open(f, "rb")):
                gon = r["gon"]
                du = r["cham"]["du"].astype(float)
                tg = dt.TheGioi(gon, r["cham"]["song"], du[:, None], np.zeros_like(du)[:, None])
                HR = np.maximum(tg.Hmax - tg.Hmin, dt.SAN_H)
                X = (0.7 / 0.3) * gon.drange / (100.0 * HR)
                lien = gon.prop + dt.PKT / gon.C * (1 + h099) + dt.PROC            # trễ mỗi liên kết khi tải đúng 0,99
                d099 = [float((gon.inc[k][:gon.so_kho[k]] * lien).sum(1).max()) for k in range(gon.K)]
                dong.append({"kb": int(r["kb"]), "t": int(r["t"]), "X_cham": [float(x) for x in X], "d099_max": d099,
                             "nguon": list(r.get("nguon", []))})
        ra[b] = dong
        print(f"  đặc trưng b {b}: {len(dong)}", flush=True)
    tep.write_text(json.dumps(ra), encoding="utf-8")
    return ra


def main():
    DT, DB = dg.dac_trung(), dac_trung_b()
    T15, T13 = dg.nap(dg.C15), dg.nap(dg.C13)
    M = {k: {s: [] for s in S} for k in ("dm", "dm_ktn", "x10", "x10_ktn", "w05", "w09")}
    kb, bo_, bao, Xr, Xc, d099, kbi = [], [], [], [], [], [], []
    for b in BO:
        for x, y, d, e in zip(T15[b], T13[b], DT[b], DB[b]):
            assert (x["kb"], x["t"]) == (y["kb"], y["t"]) == (d["kb"], d["t"]) == (e["kb"], e["t"])
            for s in S:
                for k in ("dm", "dm_ktn", "x10", "x10_ktn"):
                    M[k][s].append(x[f"{k}_s{s}"])
                M["w05"][s].append(y[f"dm_w0.5_s{s}"])
                M["w09"][s].append(y[f"dm_w0.9_s{s}"])
            kb.append(f"{b}:{x['kb']}")
            kbi.append(int(x["kb"]))
            bo_.append(b)
            bao.append(d["bao"])
            lam = np.array(d["lam"])
            gm = lambda X: float(np.exp((np.log(np.maximum(np.array(X), 1e-3)) * lam).sum() / lam.sum()))
            Xr.append(gm(d["X"]))
            Xc.append(gm(e["X_cham"]))
            d099.append(max(e["d099_max"]))
    M = {k: {s: np.array(v, float) for s, v in dd.items()} for k, dd in M.items()}
    kb, bo_, bao, Xr, Xc, d099, kbi = map(np.array, (kb, bo_, bao, Xr, Xc, d099, kbi))
    tb = {k: np.mean([M[k][s] for s in S], 0) for k in M}
    dM, dT = tb["x10"][:, MAT] - tb["dm"][:, MAT], tb["x10"][:, TRE] - tb["dm"][:, TRE]
    # cụm thời tiết dùng chung giữa các bộ: ngày (C, nam9, bao9) hoặc (Yagi, chỉ số kịch bản) — kiểm Yagi cùng kb là cùng nguồn
    yagi = np.char.startswith(bao.astype(str), "yagi")
    cum_tt = np.where(yagi, np.char.add(bao.astype(str), np.char.add("#", kbi.astype(str))), bao)
    out = {}
    ng = {}
    for b in ("yagi5", "yagi5_b", "y7wh", "xy11"):
        for e in DB[b]:
            ng.setdefault((e["kb"], e["t"]), set()).add(tuple(e["nguon"]))
    out["yagi_cung_kb_cung_nguon"] = bool(all(len(v) == 1 for v in ng.values()))

    # ---------------------------------------------------------------- C1 độ phủ CI với 13 cụm
    rg = np.random.default_rng(11)
    u, inv = np.unique(kb, return_inverse=True)
    phu = {}
    from scipy import stats
    for ten, v in (("tre", dT), ("mat", dM)):
        cm = np.bincount(inv, v) / np.bincount(inv)            # 143 cụm, cùng cỡ 144 → trung bình cụm = trung bình tình huống
        that = cm.mean()
        ok_bs = ok_t = 0
        rong_bs, rong_t = [], []
        q = stats.t.ppf(0.975, 12)
        for _ in range(1000):
            x = rg.choice(cm, 13, replace=False)
            bm = x[rg.integers(0, 13, (1000, 13))].mean(1)
            lo, hi = np.percentile(bm, [2.5, 97.5])
            ok_bs += lo <= that <= hi
            se = x.std(ddof=1) / np.sqrt(13)
            ok_t += x.mean() - q * se <= that <= x.mean() + q * se
            rong_bs.append(hi - lo)
            rong_t.append(2 * q * se)
        phu[ten] = {"phu_bootstrap_phan_vi": ok_bs / 1000, "phu_t": ok_t / 1000,
                    "rong_tb_bootstrap": float(np.mean(rong_bs)), "rong_tb_t": float(np.mean(rong_t)),
                    "do_lech_cum": float(stats.skew(cm)), "do_nhon_cum": float(stats.kurtosis(cm))}
        # cùng phép thử nhưng quần thể chỉ gồm cụm của MỘT nguồn thời tiết (gần với một bộ thật hơn)
        for nguon, bs in (("C 2024–25", dg.NGUON["C 2024–25"]), ("Yagi 2024", dg.NGUON["Yagi 2024"])):
            cn = np.array([v[kb == k].mean() for k in np.unique(kb[np.isin(bo_, bs)])])
            okb = okt = 0
            for _ in range(1000):
                x = rg.choice(cn, 13, replace=False)
                lo, hi = np.percentile(x[rg.integers(0, 13, (1000, 13))].mean(1), [2.5, 97.5])
                okb += lo <= cn.mean() <= hi
                se = x.std(ddof=1) / np.sqrt(13)
                okt += abs(x.mean() - cn.mean()) <= q * se
            phu[ten][f"phu_trong_nguon_{nguon}"] = {"so_cum": int(len(cn)), "bootstrap": okb / 1000, "t": okt / 1000}
    out["C1_do_phu_13_cum"] = phu

    # ---------------------------------------------------------------- C2 số bộ độc lập hiệu dụng
    ut, it = np.unique(cum_tt, return_inverse=True)
    est = {}
    for ten, v in (("tre", dT), ("mat", dM)):
        tong = np.zeros((len(BO), len(ut)))
        dem = np.zeros((len(BO), len(ut)))
        for i, b in enumerate(BO):
            m = bo_ == b
            tong[i] = np.bincount(it[m], v[m], minlength=len(ut))
            dem[i] = np.bincount(it[m], minlength=len(ut))
        rs = []
        for _ in range(2000):
            j = rg.integers(0, len(ut), len(ut))
            w = np.bincount(j, minlength=len(ut))
            rs.append((tong * w).sum(1) / np.maximum((dem * w).sum(1), 1))
        R = np.corrcoef(np.array(rs).T)
        ev = np.linalg.eigvalsh(R)
        est[ten] = {"so_bo_hieu_dung": float(ev.sum() ** 2 / (ev ** 2).sum()),
                    "tuong_quan_tb_cung_nguon_C": float(np.mean([R[i, j] for i in range(11) for j in range(11) if i < j
                                                                 and BO[i] in dg.NGUON["C 2024–25"] and BO[j] in dg.NGUON["C 2024–25"]])),
                    "tuong_quan_tb_cung_nguon_Yagi": float(np.mean([R[i, j] for i in range(11) for j in range(11) if i < j
                                                                    and BO[i] in dg.NGUON["Yagi 2024"] and BO[j] in dg.NGUON["Yagi 2024"]])),
                    "tuong_quan_tb_khac_nguon": float(np.mean([R[i, j] for i in range(11) for j in range(11) if i < j and not any(
                        BO[i] in bs and BO[j] in bs for bs in dg.NGUON.values())]))}
    out["C2_so_bo_doc_lap"] = est

    # ---------------------------------------------------------------- C3 bất định giữa các cơn bão
    c3 = {}
    m = bo_ == "bao9"
    for ten, v in (("tre", dT), ("mat", dM)):
        cm = np.array([v[m & (bao == x)].mean() for x in np.unique(bao[m])])
        ym = np.array([v[yagi & (kbi == k)].mean() for k in np.unique(kbi[yagi])])
        c3[ten] = {"sd_giua_13_dot_bao9": float(cm.std(ddof=1)), "sd_giua_kich_ban_trong_yagi": float(ym.std(ddof=1)),
                   "yagi_tb": float(v[yagi].mean()),
                   "yagi_khoang_cho_mot_con_bao_moi": [float(v[yagi].mean() - 1.96 * cm.std(ddof=1)),
                                                       float(v[yagi].mean() + 1.96 * cm.std(ddof=1))]}
    out["C3_giua_cac_con_bao"] = c3

    # ---------------------------------------------------------------- C4 khác biệt giữa mô hình vs trong mô hình
    def khac(p, q):
        return (np.abs(p[:, MAT] - q[:, MAT]) > 1e-9) | (np.abs(p[:, TRE] - q[:, TRE]) > 1e-9)
    trong = [khac(M["dm"][i], M["dm"][j]) for i, j in ((0, 1), (0, 2), (1, 2))] + \
            [khac(M["x10"][i], M["x10"][j]) for i, j in ((0, 1), (0, 2), (1, 2))]
    giua = [khac(M["x10"][i], M["dm"][j]) for i in S for j in S]
    dtrong = np.mean(trong, 0)
    dgiua = np.mean(giua, 0)
    c4 = {"ti_le_khac_trong_mo_hinh": float(dtrong.mean()), "ti_le_khac_giua_mo_hinh": float(dgiua.mean()),
          "hieu_giua_tru_trong": dg.bs_cum(dgiua - dtrong, cum_tt)[:3]}
    am = np.mean([np.abs(M["dm"][i][:, MAT] - M["dm"][j][:, MAT]) for i, j in ((0, 1), (0, 2), (1, 2))], 0)
    c4["mat_lech_tuyet_doi_tb_giua_2_seed_dm11"] = float(am.mean())
    c4["hieu_he_thong_x10_dm11"] = float(dM.mean())
    c4["co_hieu_ung_chuan_hoa_theo_tinh_huong_mat"] = float(dM.mean() / dM.std(ddof=1))
    c4["co_hieu_ung_chuan_hoa_theo_tinh_huong_tre"] = float(dT.mean() / dT.std(ddof=1))
    out["C4_khac_biet"] = c4

    # ---------------------------------------------------------------- C5 DM11 so với đường nối w0,5 – w0,9
    u2, i2 = np.unique(kb, return_inverse=True)
    dem2 = np.bincount(i2)
    sm = {k: {s: (np.bincount(i2, M[k][s][:, MAT]), np.bincount(i2, M[k][s][:, TRE])) for s in S} for k in ("w05", "dm", "w09")}

    def du_bao(j, chon):
        n = dem2[j].sum()
        p = {k: (np.mean([sm[k][s][0][j].sum() for s in chon[k]]) / n, np.mean([sm[k][s][1][j].sum() for s in chon[k]]) / n)
             for k in sm}
        (m5, t5), (m7, t7), (m9, t9) = p["w05"], p["dm"], p["w09"]
        return m7 - (m5 + (t7 - t5) / (t9 - t5) * (m9 - m5))
    j0 = np.arange(len(u2))
    diem = du_bao(j0, {k: S for k in sm})
    bs = [du_bao(rg.integers(0, len(u2), len(u2)), {k: rg.integers(0, 3, 3) for k in sm}) for _ in range(2000)]
    out["C5_dm11_lech_duong_noi_w"] = {"diem": float(diem), "ci_2_tang": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}

    # ---------------------------------------------------------------- C6 cơ chế khi kiểm soát mức trễ
    # biến kiểm soát mức nghẽn: trễ của Dijkstra phạt trong cùng tình huống — tất định, không chứa nhiễu seed của DM11
    # (dùng trễ DM11 sẽ dính hồi quy về trung bình vì nhiễu của DM11 nằm ở cả hai vế)
    tre_dp = np.array([x["luoi"]["dp_50_50"][TRE] for b in BO for x in T15[b]])
    lx, lt = np.log(Xr), np.log(np.maximum(tre_dp, 1e-3))
    Xd = np.column_stack([np.ones(len(lx)), lx, lt])
    uw, iw = np.unique(bao, return_inverse=True)

    def hoi_quy(y, Xd_):
        return np.linalg.lstsq(Xd_, y, rcond=None)[0]
    c6 = {}
    for ten, y in (("mat", dM), ("tre", dT)):
        hs = hoi_quy(y, Xd)
        bsl = []
        nhom_idx = [np.flatnonzero(iw == g) for g in range(len(uw))]
        for _ in range(1000):
            j = np.concatenate([nhom_idx[g] for g in rg.integers(0, len(uw), len(uw))])
            bsl.append(hoi_quy(y[j], Xd[j]))
        bsl = np.array(bsl)
        c6[ten] = {"he_so_log_X_eff": [float(hs[1]), float(np.percentile(bsl[:, 1], 2.5)), float(np.percentile(bsl[:, 1], 97.5))],
                   "he_so_log_tre_dijkstra": [float(hs[2]), float(np.percentile(bsl[:, 2], 2.5)), float(np.percentile(bsl[:, 2], 97.5))]}
    c6["spearman_X_eff_vs_tre_dijkstra"] = float(stats.spearmanr(Xr, tre_dp).statistic)
    bang = {}
    tre_dm = tb["dm"][:, TRE]
    for tt, (lo, hi) in {"trễ Dijkstra ≤ 10": (0, 10), "10–30": (10, 30), "> 30": (30, 1e9)}.items():
        for xx, (xl, xh) in {"X_eff ≤ 3": (0, 3), "3–10": (3, 10), "> 10": (10, 1e9)}.items():
            mm = (tre_dp > lo) & (tre_dp <= hi) & (Xr > xl) & (Xr <= xh)
            if mm.sum() >= 50:
                bang[f"{tt} | {xx}"] = {"n": int(mm.sum()), "mat": dg.bs_cum(dM[mm], bao[mm])[:3], "tre": float(dT[mm].mean())}
    c6["bang_cheo"] = bang
    out["C6_co_che_kiem_soat_tre"] = c6

    # ---------------------------------------------------------------- C7 X_eff theo thế giới chấm
    c7 = {"spearman_bang_vs_cham": float(stats.spearmanr(Xr, Xc).statistic)}
    nh = np.digitize(Xc, [3, 10, 30])
    c7["theo_nhom_X_cham"] = {ten: {"ti_le_%": float((nh == g).mean() * 100), "mat": dg.bs_cum(dM[nh == g], bao[nh == g])[:3],
                                    "tre": dg.bs_cum(dT[nh == g], bao[nh == g])[:3]}
                              for g, ten in enumerate(["< 3", "3–10", "10–30", "> 30"]) if (nh == g).sum() >= 30}
    out["C7_X_eff_the_gioi_cham"] = c7

    # ---------------------------------------------------------------- C8 trễ > 100 ms ⇒ có liên kết tải > 0,99?
    out["C8_bao_hoa"] = {"h(0,99)": float(h_hang_doi(np.array(0.99))), "PKT": float(dt.PKT), "PROC": float(dt.PROC),
                         "ti_le_tinh_huong_moi_tuyen_kho_duoi_100ms_o_tai_0.99_%": float((d099 < 100).mean() * 100),
                         "phan_vi_can_tren_tre_o_tai_0.99": {q: float(np.percentile(d099, q)) for q in (50, 90, 99, 100)},
                         "ca_dm11_tren_100ms_ma_can_tren_duoi_100_%": float(
                             ((tre_dm > 100) & (d099 < 100)).sum() / max((tre_dm > 100).sum(), 1) * 100)}

    # ---------------------------------------------------------------- C9 đuôi trễ và quá tải
    duoi_dm = np.mean([M["dm"][s][:, TRE] > 100 for s in S], 0)
    duoi_x = np.mean([M["x10"][s][:, TRE] > 100 for s in S], 0)
    dB = tb["x10"][:, B] - tb["dm"][:, B]
    dD = duoi_x - duoi_dm
    hi_ = tre_dm > 100
    c9 = {"b_dm11_khi_tre_tren_100": dg.bs_cum(tb["dm"][hi_, B], bao[hi_])[:3],
          "b_dm11_khi_tre_duoi_100": dg.bs_cum(tb["dm"][~hi_, B], bao[~hi_])[:3],
          "tuong_quan_diem_hai_chuoi_b_va_duoi": float(stats.pearsonr(tb["dm"][:, B], duoi_dm).statistic),
          "phan_hieu_b_tu_ca_duoi_dm11_%": float(dB[hi_].sum() / dB.sum() * 100),
          "phan_hieu_duoi_tu_ca_duoi_dm11_%": float(dD[hi_].sum() / dD.sum() * 100),
          "ti_le_ca_duoi_dm11_%": float(hi_.mean() * 100),
          "spearman_hieu_b_vs_hieu_tre": float(stats.spearmanr(dB, dT).statistic)}
    out["C9_duoi_va_qua_tai"] = c9

    # ---------------------------------------------------------------- C10 chọn seed tốt nhất (biết trước)
    out["C10_chon_seed_biet_truoc"] = {
        "x10_tb_3_seed": float(tb["x10"][:, MAT].mean()),
        "x10_tot_nhat_moi_tinh_huong": float(np.min([M["x10"][s][:, MAT] for s in S], 0).mean()),
        "x10_ktn_tot_nhat_moi_tinh_huong": float(np.min([M["x10_ktn"][s][:, MAT] for s in S], 0).mean()),
        "x10_ktn_tb_3_seed": float(tb["x10_ktn"][:, MAT].mean())}

    # ---------------------------------------------------------------- C11 lợi cả hai hay đánh đổi
    loi2 = dM <= 0
    out["C11_nguon_giam_tre"] = {"ti_le_tinh_huong_mat_khong_tang_%": float(loi2.mean() * 100),
                                 "phan_giam_tre_tu_ca_mat_khong_tang_%": float(dT[loi2].sum() / dT.sum() * 100),
                                 "phan_giam_tre_tu_ca_mat_tang_%": float(dT[~loi2].sum() / dT.sum() * 100),
                                 "mat_tang_tb_o_ca_danh_doi": float(dM[~loi2].mean()),
                                 "tre_giam_tb_o_ca_danh_doi": float(dT[~loi2].mean())}

    # ---------------------------------------------------------------- C12 nhiều phép thử
    def p2(k1, k0, j, ham=None):
        c1 = {s: np.bincount(inv, ham(M[k1][s]) if ham else M[k1][s][:, j]) for s in S}
        c0 = {s: np.bincount(inv, ham(M[k0][s]) if ham else M[k0][s][:, j]) for s in S}
        dem = np.bincount(inv)
        rr = np.random.default_rng(5)
        bsv = []
        for _ in range(2000):
            jj, s1, s0 = rr.integers(0, len(u), len(u)), rr.integers(0, 3, 3), rr.integers(0, 3, 3)
            n = dem[jj].sum()
            bsv.append((np.mean([c1[s][jj].sum() for s in s1]) - np.mean([c0[s][jj].sum() for s in s0])) / n)
        diem_ = np.mean([c1[s].sum() for s in S]) / dem.sum() - np.mean([c0[s].sum() for s in S]) / dem.sum()
        z = diem_ / np.std(bsv, ddof=1)
        return {"diem": float(diem_), "z": float(z), "p": float(2 * (1 - Phi(abs(z))))}
    duoi = lambda X: (X[:, TRE] > 100).astype(float) * 100
    ps = {"mat x10-dm": p2("x10", "dm", MAT), "tre x10-dm": p2("x10", "dm", TRE), "duoi x10-dm": p2("x10", "dm", None, duoi),
          "mat dm_ktn-dm": p2("dm_ktn", "dm", MAT), "mat x10_ktn-x10": p2("x10_ktn", "x10", MAT),
          "mat x10_ktn-dm": p2("x10_ktn", "dm", MAT)}
    so_phep = 4 * 11 + len(ps) + 2
    # "không kém" (H0: hiệu ≥ +0,10 điểm), một phía, cùng sai số chuẩn hai tầng
    for k in ("mat x10-dm", "mat x10_ktn-dm"):
        se = abs(ps[k]["diem"] / ps[k]["z"])
        z = (ps[k]["diem"] - 0.10) / se
        ps[f"{k} (không kém +0,10)"] = {"diem": ps[k]["diem"], "z": float(z), "p": float(Phi(z))}
    out["C12_nhieu_phep_thu"] = {"so_phep_thu_tinh": so_phep, "nguong_bonferroni": 0.05 / so_phep, "p": ps,
                                 "qua_bonferroni": [k for k, v in ps.items() if v["p"] < 0.05 / so_phep]}

    # ---------------------------------------------------------------- C13 lệch do chọn ktn sau khi xem bộ kiểm tra
    d_ = ps["mat dm_ktn-dm"]["diem"]
    s_ = abs(d_ / ps["mat dm_ktn-dm"]["z"])
    zz = abs(d_) / s_
    lech = s_ * (2 * np.exp(-zz ** 2 / 2) / np.sqrt(2 * np.pi)) - 2 * abs(d_) * (1 - Phi(zz))    # E|D| − |δ| khi D ~ N(δ, s²)
    out["C13_lech_chon_ktn"] = {"hieu": float(d_), "sai_so_chuan": float(s_), "lech_ky_vong_toi_da": float(max(lech, 0.0)),
                                "xac_suat_chon_nham_chieu": float(1 - Phi(zz))}

    # ---------------------------------------------------------------- C14 H15.0 khi hiệu chỉnh 11 mục tiêu
    h150 = {"mat": (-0.08, -0.14, -0.02), "tre": (-6.2, -11.9, -1.8)}       # tn15/KET_QUA_15.md mục 0
    zb = float(stats.norm.ppf(1 - 0.05 / (2 * 10)))                          # 10 so sánh với 7/3 trong 11 mục tiêu
    out["C14_H15_0_hieu_chinh"] = {"z_bonferroni_10_so_sanh": zb,
                                   **{k: [v[0] - zb * (v[2] - v[1]) / 3.92, v[0] + zb * (v[2] - v[1]) / 3.92] for k, v in h150.items()}}

    (dg.RA / "co_so.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    return out


if __name__ == "__main__":
    print(json.dumps(main(), ensure_ascii=False, indent=1))
