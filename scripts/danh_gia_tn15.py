"""
danh_gia_tn15.py — Đánh giá khoa học TN15 (results/bao_da_tuyen/DANH_GIA_TN15.md). Không học, không chấm lại: chỉ đọc
bản ghi chấm của TN15 (tn15_may_nhom/tn15/cham_tep), TN13 (tn13_may_nhom/tn13/cham_tep: DM13 w0,5 / w0,9 cùng tình huống)
và dữ liệu 11 bộ kiểm tra ở data/bao_da_tuyen (TN15 xác nhận sinh lại khớp bản gốc 100%).

    python -m scripts.danh_gia_tn15            # ~4 phút (bước đặc trưng ~3 phút, có đệm)

Ra: results/bao_da_tuyen/danh_gia_tn15/{dac_trung.json, phan_tich.json, hinh_danh_gia_15.png}.

Các phân tích:
 1. theo seed: 9 cặp seed x10 − DM11; CI hai tầng (bốc lại cửa sổ + bốc lại seed của mỗi mô hình);
 2. độc lập giữa 11 bộ: cửa sổ thời tiết dùng chung; gộp 11 bộ với cụm = cửa sổ thời tiết; theo 4 nguồn thời tiết;
 3. ít cụm: CI phân vị bootstrap (như TN15) so với CI t trên trung bình cụm, theo bộ;
 4. phân phối trễ: phân vị, đuôi theo nhiều ngưỡng, phần đóng góp vào hiệu trễ trung bình;
 5. đổi quyết định cùng seed (x10 vs DM11) so với mức khác nhau giữa hai seed của cùng DM11;
 6. cơ chế: "tỷ giá ngầm" của J 7/3 theo luồng X_eff = (0,7/0,3)·drange / (100·(Hmax − Hmin)) ms cho 1 điểm %; hiệu x10 −
    DM11 theo nhóm X_eff;
 7. (b) quá tải và trễ đi cùng nhau đến đâu;
 8. đường biên DM: x10 so với họ w (w0,5 / w0,9 học cùng đường ống TN13, cùng máy với DM x10).
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
from src.nhieu_luong import da_tuyen as dt  # noqa: E402

KQ = GOC / "results" / "bao_da_tuyen"
RA = KQ / "danh_gia_tn15"
C15, C13 = KQ / "tn15_may_nhom" / "tn15" / "cham_tep", KQ / "tn13_may_nhom" / "tn13" / "cham_tep"
BO = ("c6", "yagi5", "c6_b", "yagi5_b", "ngay9", "nam9", "bao9", "c7wh", "y7wh", "xc11", "xy11")
NGUON = {"C 2024–25": ("c6", "c6_b", "ngay9", "c7wh", "xc11"), "Yagi 2024": ("yagi5", "yagi5_b", "y7wh", "xy11"),
         "ngày xấu 2017–21": ("nam9",), "bão 2017–21": ("bao9",)}
MAT, A, B, TRE = 0, 2, 3, 4
S = (0, 1, 2)
NB = 2000


# ============================================================================================ đặc trưng
def dac_trung():
    tep = RA / "dac_trung.json"
    if tep.exists():
        return json.loads(tep.read_text(encoding="utf-8"))
    ra = {}
    for b in BO:
        dong = []
        for f in sorted((GOC / "data" / "bao_da_tuyen" / f"{b}_k4_5").glob("kb_*.pkl")):
            for r in pickle.load(open(f, "rb")):
                gon = r["gon"]
                tg = dt.TheGioi.tu_bang(gon, np.random.default_rng([int(r["kb"]), int(r["t"]), 99]))
                HR = np.maximum(tg.Hmax - tg.Hmin, dt.SAN_H)
                X = (0.7 / 0.3) * gon.drange / (100.0 * HR)
                dong.append({"kb": int(r["kb"]), "t": int(r["t"]), "bao": r.get("bao"), "K": int(gon.K),
                             "X": [float(x) for x in X], "lam": [float(x) for x in gon.lam]})
        ra[b] = dong
        print(f"  đặc trưng {b}: {len(dong)}", flush=True)
    RA.mkdir(parents=True, exist_ok=True)
    tep.write_text(json.dumps(ra), encoding="utf-8")
    return ra


def nap(thu):
    ra = {}
    for b in BO:
        rs = []
        for f in sorted(glob.glob(str(thu / f"{b}_k4_5__*.json"))):
            rs += json.load(open(f, encoding="utf-8"))
        ra[b] = rs
    return ra


# ============================================================================================ thống kê
def bs_cum(v, cum, seed=0):
    """Bootstrap theo cụm, trung bình theo tình huống (như scripts.tn10.bootstrap): [điểm, thấp, cao, số cụm]."""
    u, inv = np.unique(cum, return_inverse=True)
    tong, dem = np.bincount(inv, v), np.bincount(inv)
    j = np.random.default_rng(seed).integers(0, len(u), (NB, len(u)))
    m = tong[j].sum(1) / dem[j].sum(1)
    return [float(v.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5)), int(len(u))]


def t_cum(v, cum):
    """CI t trên trung bình từng cụm (mỗi cụm nặng như nhau; số cụm − 1 bậc tự do)."""
    from scipy import stats
    u, inv = np.unique(cum, return_inverse=True)
    m = np.bincount(inv, v) / np.bincount(inv)
    se, q = m.std(ddof=1) / np.sqrt(len(m)), stats.t.ppf(0.975, len(m) - 1)
    return [float(m.mean()), float(m.mean() - q * se), float(m.mean() + q * se), int(len(u))]


# ============================================================================================ phân tích
def phan_tich():
    DT = dac_trung()
    T15, T13 = nap(C15), nap(C13)
    ten = {"dm": "dm", "dm_ktn": "dm_ktn", "x10": "x10", "x10_ktn": "x10_ktn"}
    M = {k: {s: [] for s in S} for k in (*ten, "w05", "w09")}
    kb, bo_, bao, dp, Xr = [], [], [], [], []
    lech13 = 0.0
    for b in BO:
        for x, y, d in zip(T15[b], T13[b], DT[b]):
            assert (x["kb"], x["t"]) == (y["kb"], y["t"]) == (d["kb"], d["t"])
            for s in S:
                for k, kk in ten.items():
                    M[k][s].append(x[f"{kk}_s{s}"])
                M["w05"][s].append(y[f"dm_w0.5_s{s}"])
                M["w09"][s].append(y[f"dm_w0.9_s{s}"])
                lech13 = max(lech13, abs(y[f"dm_w0.7_s{s}"][0] - x[f"dm_s{s}"][0]))
            kb.append(f"{b}:{x['kb']}")
            bo_.append(b)
            bao.append(d["bao"])
            dp.append(x["luoi"]["dp_50_50"])
            X, lam = np.array(d["X"]), np.array(d["lam"])
            Xr.append(float(np.exp((np.log(np.maximum(X, 1e-3)) * lam).sum() / lam.sum())))
    M = {k: {s: np.array(v, float) for s, v in dd.items()} for k, dd in M.items()}
    kb, bo_, bao, dp, Xr = map(np.array, (kb, bo_, bao, dp, Xr))
    tb = {k: np.mean([M[k][s] for s in S], 0) for k in M}
    out = {"so_tinh_huong": int(len(kb)), "lech_dm11_tn13_tn15": lech13}

    # 1. theo seed
    sg = {k: {s: {"mat": float(M[k][s][:, MAT].mean()), "tre": float(M[k][s][:, TRE].mean()),
                  "duoi": float((M[k][s][:, TRE] > 100).mean() * 100)} for s in S} for k in M}
    out["seed_gop"] = sg
    u, inv = np.unique(kb, return_inverse=True)
    dem = np.bincount(inv)
    cap = {}
    for k1, k0 in (("x10", "dm"), ("dm_ktn", "dm"), ("x10_ktn", "x10"), ("x10_ktn", "dm"), ("x10", "w05"), ("x10", "w09")):
        rg = np.random.default_rng(1)
        c1 = {s: (np.bincount(inv, M[k1][s][:, MAT]), np.bincount(inv, M[k1][s][:, TRE])) for s in S}
        c0 = {s: (np.bincount(inv, M[k0][s][:, MAT]), np.bincount(inv, M[k0][s][:, TRE])) for s in S}
        bm, bt = [], []
        for _ in range(NB):
            j, s1, s0 = rg.integers(0, len(u), len(u)), rg.integers(0, 3, 3), rg.integers(0, 3, 3)
            n = dem[j].sum()
            bm.append((np.mean([c1[s][0][j].sum() for s in s1]) - np.mean([c0[s][0][j].sum() for s in s0])) / n)
            bt.append((np.mean([c1[s][1][j].sum() for s in s1]) - np.mean([c0[s][1][j].sum() for s in s0])) / n)
        dm_ = [sg[k1][i]["mat"] - sg[k0][j]["mat"] for i in S for j in S]
        dtr = [sg[k1][i]["tre"] - sg[k0][j]["tre"] for i in S for j in S]
        cap[f"{k1}-{k0}"] = {
            "mat": float(np.mean(dm_)), "tre": float(np.mean(dtr)),
            "mat_9_cap": [float(min(dm_)), float(max(dm_))], "tre_9_cap": [float(min(dtr)), float(max(dtr))],
            "mat_ci_cua_so": bs_cum(tb[k1][:, MAT] - tb[k0][:, MAT], kb)[1:3],
            "tre_ci_cua_so": bs_cum(tb[k1][:, TRE] - tb[k0][:, TRE], kb)[1:3],
            "mat_ci_2_tang": [float(np.percentile(bm, 2.5)), float(np.percentile(bm, 97.5))],
            "tre_ci_2_tang": [float(np.percentile(bt, 2.5)), float(np.percentile(bt, 97.5))]}
    out["cap_seed"] = cap
    out["sd_seed"] = {k: {"mat": float(np.std([sg[k][s]["mat"] for s in S], ddof=1)),
                          "tre": float(np.std([sg[k][s]["tre"] for s in S], ddof=1))} for k in M}

    # 2. độc lập giữa các bộ
    out["cua_so_theo_bo"] = {b: sorted(set(bao[bo_ == b])) for b in BO}
    out["so_cua_so_thoi_tiet_rieng"] = int(len(set(bao)))
    gop = {}
    for nhan, v in (("mat x10-dm", tb["x10"][:, MAT] - tb["dm"][:, MAT]), ("tre x10-dm", tb["x10"][:, TRE] - tb["dm"][:, TRE]),
                    ("mat x10_ktn-dm", tb["x10_ktn"][:, MAT] - tb["dm"][:, MAT]),
                    ("duoi x10-dm", ((tb["x10"][:, TRE] > 100) * 1.0 - (tb["dm"][:, TRE] > 100)) * 100),
                    ("mat x10-dp", tb["x10"][:, MAT] - dp[:, MAT]), ("tre x10-dp", tb["x10"][:, TRE] - dp[:, TRE])):
        gop[nhan] = {"cum_cua_so_theo_bo (như TN15)": bs_cum(v, kb), "cum_thoi_tiet": bs_cum(v, bao),
                     "moi_cua_so_thoi_tiet_nang_nhu_nhau (t)": t_cum(v, bao)}
    out["gop_11_bo"] = gop
    ng = {}
    for nhan, bs in NGUON.items():
        m = np.isin(bo_, bs)
        cum = np.where(np.char.startswith(bao[m].astype(str), "yagi"), np.char.add(bao[m].astype(str), kb[m]), bao[m])
        ng[nhan] = {"so_tinh_huong": int(m.sum()), "so_cum": int(len(set(cum))),
                    "mat x10-dm": bs_cum(tb["x10"][m, MAT] - tb["dm"][m, MAT], cum)[:3],
                    "tre x10-dm": bs_cum(tb["x10"][m, TRE] - tb["dm"][m, TRE], cum)[:3],
                    "mat x10_ktn-dm": bs_cum(tb["x10_ktn"][m, MAT] - tb["dm"][m, MAT], cum)[:3]}
    out["theo_nguon_thoi_tiet"] = ng

    # 3. ít cụm: H15.1 theo bộ với hai cách lập CI
    ic = {}
    for b in BO:
        m = bo_ == b
        vm, vt = tb["x10"][m, MAT] - tb["dm"][m, MAT], tb["x10"][m, TRE] - tb["dm"][m, TRE]
        ic[b] = {"mat_bs": bs_cum(vm, kb[m]), "mat_t": t_cum(vm, kb[m]), "tre_bs": bs_cum(vt, kb[m]), "tre_t": t_cum(vt, kb[m])}
    out["it_cum"] = ic
    out["H15.1_dat"] = {"bootstrap_phan_vi": [b for b in BO if ic[b]["mat_bs"][2] < 0.10 and ic[b]["tre_bs"][2] < 0],
                        "t_tren_trung_binh_cum": [b for b in BO if ic[b]["mat_t"][2] < 0.10 and ic[b]["tre_t"][2] < 0]}

    # 4. phân phối trễ
    pp = {}
    for k in ("dm", "x10", "x10_ktn", "w05", "w09", "dp"):
        x = dp[:, TRE] if k == "dp" else np.concatenate([M[k][s][:, TRE] for s in S])
        pp[k] = {"phan_vi": {q: float(np.percentile(x, q)) for q in (50, 75, 90, 95, 99, 99.9)},
                 "duoi_%": {g: float((x > g).mean() * 100) for g in (50, 100, 200, 500)},
                 "tb": float(x.mean()), "tb_bo_1%_cao_nhat": float(np.sort(x)[: int(0.99 * len(x))].mean())}
    out["phan_phoi_tre"] = pp
    d = tb["x10"][:, TRE] - tb["dm"][:, TRE]
    o = np.argsort(d)
    out["dong_gop_hieu_tre"] = {"top_%": {p: float(d[o[:max(1, int(p / 100 * len(d)))]].sum() / d.sum() * 100) for p in (0.1, 1, 5)},
                                "tu_ca_dm11_tren_100ms_%": float(d[tb["dm"][:, TRE] > 100].sum() / d.sum() * 100),
                                "ti_le_ca_dm11_tren_100ms_%": float((tb["dm"][:, TRE] > 100).mean() * 100)}
    du = {}
    for g in (50, 100, 200):
        du[g] = sum(bs_cum(((tb["x10"][bo_ == b, TRE] > g) * 1.0 - (tb["dm"][bo_ == b, TRE] > g)) * 100, kb[bo_ == b])[2] < 0
                    for b in BO)
    out["H15.2_so_bo_dat_theo_nguong"] = du

    # 5. đổi quyết định
    giong = lambda p, q: float(((np.abs(p[:, MAT] - q[:, MAT]) < 1e-9) & (np.abs(p[:, TRE] - q[:, TRE]) < 1e-9)).mean() * 100)
    out["trung_ket_qua_%"] = {"x10_vs_dm_cung_seed": [giong(M["x10"][s], M["dm"][s]) for s in S],
                              "dm_seed_vs_seed": [giong(M["dm"][i], M["dm"][j]) for i, j in ((0, 1), (0, 2), (1, 2))]}

    # 6. cơ chế
    out["X_eff_phan_vi"] = {q: float(np.percentile(Xr, q)) for q in (10, 25, 50, 75, 90)}
    nh = np.digitize(Xr, [3, 10, 30])
    cc = {}
    for g, nhan in enumerate(["< 3", "3–10", "10–30", "> 30"]):
        m = nh == g
        cc[nhan] = {"so_tinh_huong": int(m.sum()), "ti_le_%": float(m.mean() * 100),
                    "mat x10-dm": bs_cum(tb["x10"][m, MAT] - tb["dm"][m, MAT], bao[m])[:3],
                    "tre x10-dm": bs_cum(tb["x10"][m, TRE] - tb["dm"][m, TRE], bao[m])[:3],
                    "a x10-dm": float((tb["x10"][m, A] - tb["dm"][m, A]).mean()),
                    "b x10-dm": float((tb["x10"][m, B] - tb["dm"][m, B]).mean()),
                    "tre_dm11": float(tb["dm"][m, TRE].mean()), "duoi_dm11_%": float((tb["dm"][m, TRE] > 100).mean() * 100),
                    "dong_gop_vao_hieu_mat_%": float((tb["x10"][m, MAT] - tb["dm"][m, MAT]).sum() /
                                                      (tb["x10"][:, MAT] - tb["dm"][:, MAT]).sum() * 100),
                    "dong_gop_vao_hieu_tre_%": float(d[m].sum() / d.sum() * 100)}
    out["co_che_theo_X_eff"] = cc
    from scipy import stats
    out["co_che_spearman"] = {"X_eff~hieu_tre": float(stats.spearmanr(Xr, d).statistic),
                              "X_eff~hieu_mat": float(stats.spearmanr(Xr, tb["x10"][:, MAT] - tb["dm"][:, MAT]).statistic)}

    # 7. quá tải và trễ
    hi = tb["dm"][:, TRE] > 100
    out["qua_tai_va_tre"] = {"dm11_b_khi_tre_tren_100": float(tb["dm"][hi, B].mean()),
                             "dm11_b_khi_tre_duoi_100": float(tb["dm"][~hi, B].mean()),
                             "dm11_a_khi_tre_tren_100": float(tb["dm"][hi, A].mean()),
                             "dm11_a_khi_tre_duoi_100": float(tb["dm"][~hi, A].mean())}

    # 8. đường biên DM
    out["bien_dm"] = {k: {"mat": float(tb[k][:, MAT].mean()), "tre": float(tb[k][:, TRE].mean()),
                          "duoi": float((tb[k][:, TRE] > 100).mean() * 100)} for k in ("w05", "dm", "w09", "x10", "dm_ktn", "x10_ktn")}
    out["bien_dm"]["dp"] = {"mat": float(dp[:, MAT].mean()), "tre": float(dp[:, TRE].mean()), "duoi": float((dp[:, TRE] > 100).mean() * 100)}
    (RA / "phan_tich.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    ve(out)
    return out


# ============================================================================================ hình
def ve(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    MAU = {"w05": "#9b8ae0", "dm": "#2a78d6", "w09": "#5b4bb7", "x10": "#eb6834", "x10_ktn": "#c4302b", "dm_ktn": "#6fa8e8", "dp": "#888888"}
    TEN = {"w05": "DM w 0,5", "dm": "DM11 (w 0,7)", "w09": "DM w 0,9", "x10": "DM x10", "x10_ktn": "DM x10, không tái nhiễu",
           "dm_ktn": "DM11, không tái nhiễu", "dp": "Dijkstra phạt"}
    fig, ax = plt.subplots(1, 3, figsize=(15.5, 5.4), gridspec_kw={"width_ratios": [1.25, 1, 1]})
    a = ax[0]
    for k in ("w05", "dm", "w09"):
        for s in S:
            v = out["seed_gop"][k][s]
            a.scatter(v["tre"], v["mat"], s=14, color=MAU[k], alpha=0.35, zorder=2)
    xs = [out["bien_dm"][k]["tre"] for k in ("w05", "dm", "w09")]
    ys = [out["bien_dm"][k]["mat"] for k in ("w05", "dm", "w09")]
    a.plot(xs, ys, color="#5b4bb7", lw=1.2, ls="--", zorder=1, label="họ w (chuẩn hoá theo luồng)")
    for k in ("w05", "dm", "w09", "x10", "x10_ktn", "dm_ktn"):
        v = out["bien_dm"][k]
        a.scatter(v["tre"], v["mat"], s=60, color=MAU[k], edgecolor="white", linewidth=1.5, zorder=3, label=TEN[k])
        if k in ("x10", "x10_ktn"):
            for s in S:
                w = out["seed_gop"][k][s]
                a.scatter(w["tre"], w["mat"], s=14, color=MAU[k], alpha=0.35, zorder=2)
    a.set_xlabel("trễ tuyến chính trung bình (ms)")
    a.set_ylabel("mất dịch vụ (%), gộp 11 bộ")
    a.set_title("Đường biên của DM: x10 nằm ngoài họ w\n(điểm nhỏ = từng seed; Dijkstra phạt: 34,68% / 17,1 ms, ngoài khung)",
                fontsize=10, loc="left")
    a.legend(fontsize=7.5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=3)
    a.grid(alpha=0.3)
    cc = out["co_che_theo_X_eff"]
    nh = list(cc)
    xlab = [f"{n}\n({cc[n]['ti_le_%']:.0f}%)" for n in nh]
    for i, (khoa, tieu_de, don_vi) in enumerate((("mat x10-dm", "Hiệu mất dịch vụ x10 − DM11", "điểm %"),
                                                 ("tre x10-dm", "Hiệu trễ trung bình x10 − DM11", "ms"))):
        a = ax[i + 1]
        v = [cc[n][khoa] for n in nh]
        mau = ["#2e9d6a" if x[0] < 0 else "#c4302b" for x in v] if khoa.startswith("mat") else ["#eb6834"] * len(v)
        a.bar(range(len(nh)), [x[0] for x in v], color=mau, width=0.6, zorder=2)
        a.errorbar(range(len(nh)), [x[0] for x in v], yerr=[[x[0] - x[1] for x in v], [x[2] - x[0] for x in v]],
                   fmt="none", ecolor="#333333", capsize=3, lw=1, zorder=3)
        a.axhline(0, color="#555555", lw=0.8)
        a.set_xticks(range(len(nh)), xlab, fontsize=8)
        a.set_xlabel("tỷ giá ngầm của J 7/3 (ms / điểm %; x10 = 10)\n(trong ngoặc: % tình huống)", fontsize=8)
        a.set_ylabel(don_vi)
        a.set_title(tieu_de + "\ntheo tỷ giá ngầm (CI 95%, cụm = cửa sổ thời tiết)", fontsize=10, loc="left")
        a.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(RA / "hinh_danh_gia_15.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    o = phan_tich()
    print(json.dumps({k: o[k] for k in ("cap_seed", "H15.1_dat", "co_che_theo_X_eff", "bien_dm")}, ensure_ascii=False, indent=1))
