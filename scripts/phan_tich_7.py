"""
phan_tich_7.py — Thí nghiệm 7 (results/bao_da_tuyen/TIEU_CHI_7.md).

    python -m scripts.phan_tich_7          # → ket_qua_7.json, KET_QUA_7.md, hinh_7.png

7a: Lai6 (DM6, τ6 = 1,1, không đổi) so với Dijkstra phạt trên thế giới cực đoan giả lập (W thời tiết, H hạ tầng, WH cả
hai), so với bộ thường cùng cửa sổ (C, Yagi của thí nghiệm 6). 7b: DM thuần huấn luyện trên 25 / 50 / 100 / 200% dữ liệu.
CI 95%: bootstrap ghép cặp theo 13 cửa sổ, 10 000 lần.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from scripts.phan_tich_5 import CAC_K, DAI, TEN_DAI, nap
from scripts.phan_tich_dt import B, bootstrap

GOC = Path(__file__).resolve().parents[1]
KQ = GOC / "results" / "bao_da_tuyen"
TAU = 1.1
BO_7A = {"c7wh": ("c6", "C — ngày giông, cực đoan WH"), "y7wh": ("yagi5", "A — Yagi, cực đoan WH")}
BIEN_THE = {"w": "W thời tiết", "h": "H hạ tầng", "wh": "WH cả hai"}
MO_HINH_7B = {"dm7_25": 25, "dm7_50": 50, "dm6": 100, "dm7_200": 200}
BO_7B = {"c6": "C ngày giông", "mua5": "B mưa lớn", "yagi5": "A Yagi", "c7wh": "C cực đoan", "y7wh": "A cực đoan"}


def lai6(r, tw="ens"):
    return r[tw]["dm6" if r["rho_hat"] < TAU else "dm6_tkcb"]


def gia_tri(r, p, k, tw="ens"):
    return lai6(r, tw)[k] if p == "lai6" else r[tw][p][k]


def bootstrap_cua_so(a, b, rng):
    """Hiệu trung bình theo cửa sổ của hai bộ CÙNG 13 cửa sổ (ghép theo chỉ số cửa sổ): {kb: [giá trị]}."""
    ks = sorted(set(a) & set(b))
    d = np.array([np.mean(a[k]) - np.mean(b[k]) for k in ks])
    j = rng.integers(0, len(ks), (B, len(ks)))
    m = d[j].mean(1)
    return [float(d.mean()), [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]]


def o_tieu_chi(rs, kd, rng):
    """Một ô (bộ × quy mô × dải): Lai6 so với Dijkstra phạt (như thí nghiệm 6)."""
    kb = np.array([r["kb"] for r in rs])
    v = lambda p, k: np.array([gia_tri(r, p, k) for r in rs])
    t10 = [r["tg_ms"]["online_tkcb10_6"] for r in rs if "online_tkcb10_6" in r["tg_ms"]]
    tl = float(np.median([r["tg_ms"]["dm6" if r["rho_hat"] < TAU else "dm6_tkcb"] for r in rs]))
    C2 = bootstrap(v("lai6", "J") - v("dphat", "J"), kb, rng)
    C4 = bootstrap(v("lai6", "mat_dv_%") - v("dphat", "mat_dv_%"), kb, rng)
    C5 = bool(t10) and tl <= float(np.median(t10)) and tl <= 1000
    x = {"so": len(rs), **{f"{p}_{k}": float(v(p, k).mean()) for p in ("dphat", "lai6")
                           for k in ("mat_dv_%", "khong_tuyen_song_%", "du_phong_qua_tai_%", "J")},
         "tg_lai6_ms": tl, "tg_tkcb10_ms": float(np.median(t10)) if t10 else None,
         "C2": C2, "C4": C4, "dat_C2": C2[1][1] < 0, "dat_C4": C4[1][1] < 0, "dat_C5": C5}
    x["thang"] = bool(x["dat_C2"] and x["dat_C4"] and (kd == "k4_5" or C5))
    return x


def phan_tich():
    rng = np.random.default_rng(0)
    ra = {"tau": TAU, "7a": {}, "7b": {}}
    # ---------------- 7a
    for bo, (thuong, ten) in BO_7A.items():
        for kd in CAC_K:
            recs = nap(bo, kd)
            if not recs or any("dm6_tkcb" not in r["ens"] for r in recs):
                continue
            o = {d: o_tieu_chi([r for r in recs if lo <= r["rho_hat"] < hi], kd, rng)
                 for d, (lo, hi) in DAI.items() if any(lo <= r["rho_hat"] < hi for r in recs)}
            rt = nap(thuong, kd)
            if rt and all("dm6_tkcb" in r["ens"] for r in rt):
                o["so_voi_thuong"] = {"thuong": {d: o_tieu_chi([r for r in rt if lo <= r["rho_hat"] < hi], kd, rng)
                                                 for d, (lo, hi) in DAI.items()}}
                gom = lambda rr, p: {k: [gia_tri(r, p, "mat_dv_%") for r in rr if r["kb"] == k] for k in {r["kb"] for r in rr}}
                o["so_voi_thuong"]["dphat_tang_diem_%"] = bootstrap_cua_so(gom(recs, "dphat"), gom(rt, "dphat"), rng)
                o["so_voi_thuong"]["lai6_tang_diem_%"] = bootstrap_cua_so(gom(recs, "lai6"), gom(rt, "lai6"), rng)
            ra["7a"][f"{bo}_{kd}"] = o
    for b in ("c", "y"):                               # biến thể W / H / WH, 4–5 luồng, gộp mọi dải tải
        for s in BIEN_THE:
            recs = nap(f"{b}7{s}", "k4_5")
            if not recs or any("dm6_tkcb" not in r["ens"] for r in recs):
                continue
            kb = np.array([r["kb"] for r in recs])
            v = lambda p, k: np.array([gia_tri(r, p, k) for r in recs])
            ra["7a"][f"bien_the_{b}7{s}"] = {
                **{f"{p}_{k}": float(v(p, k).mean()) for p in ("dphat", "lai6")
                   for k in ("mat_dv_%", "khong_tuyen_song_%", "du_phong_qua_tai_%")},
                "C4_gop": bootstrap(v("lai6", "mat_dv_%") - v("dphat", "mat_dv_%"), kb, rng)}
    # ---------------- 7b
    for bo in BO_7B:
        recs = nap(bo, "k4_5")
        if not recs:
            continue
        co = [m for m in MO_HINH_7B if all(m in r["ens"] for r in recs)]
        if not co:
            continue
        kb = np.array([r["kb"] for r in recs])
        v = lambda p, k: np.array([r["ens"][p][k] for r in recs])
        o = {"so": len(recs), "dphat_mat_dv_%": float(v("dphat", "mat_dv_%").mean()),
             "mo_hinh": {m: {"ti_le_%": MO_HINH_7B[m], "mat_dv_%": float(v(m, "mat_dv_%").mean()),
                             "J": float(v(m, "J").mean()),
                             "khong_tuyen_song_%": float(v(m, "khong_tuyen_song_%").mean()),
                             "du_phong_qua_tai_%": float(v(m, "du_phong_qua_tai_%").mean()),
                             "tru_dphat": bootstrap(v(m, "mat_dv_%") - v("dphat", "mat_dv_%"), kb, rng)} for m in co}}
        for m in co:                                   # theo dải tải
            o["mo_hinh"][m]["theo_dai"] = {d: float(np.mean([r["ens"][m]["mat_dv_%"] for r in recs
                                                             if lo <= r["rho_hat"] < hi])) for d, (lo, hi) in DAI.items()}
        o["dphat_theo_dai"] = {d: float(np.mean([r["ens"]["dphat"]["mat_dv_%"] for r in recs if lo <= r["rho_hat"] < hi]))
                               for d, (lo, hi) in DAI.items()}
        if len(co) == len(MO_HINH_7B):
            dv = [o["mo_hinh"][m]["mat_dv_%"] for m in MO_HINH_7B]
            o["giam_don_dieu"] = bool(all(x > y for x, y in zip(dv, dv[1:])))
        if "dm7_200" in co:
            o["200_tru_100"] = bootstrap(v("dm7_200", "mat_dv_%") - v("dm6", "mat_dv_%"), kb, rng)
            o["H7b_dat"] = o["200_tru_100"][1][1] < 0
            o["200_tru_100_J"] = bootstrap(v("dm7_200", "J") - v("dm6", "J"), kb, rng)
        if "dm7_25" in co:
            o["25_tru_100"] = bootstrap(v("dm7_25", "mat_dv_%") - v("dm6", "mat_dv_%"), kb, rng)
        ra["7b"][bo] = o
    json.dump(ra, open(KQ / "ket_qua_7.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    viet_md(ra)
    ve_hinh(ra)
    return ra


def _ci(v, nd=2):
    return f"{v[0]:+.{nd}f} [{v[1][0]:+.{nd}f}; {v[1][1]:+.{nd}f}]"


def viet_md(ra):
    dd = lambda b: "**đạt**" if b else "không đạt"
    L = ["# Thí nghiệm 7 — kết quả (sinh tự động bởi `phan_tich_7.py`)", "",
         "## 7a — Lai6 (τ6 = 1,1) so với Dijkstra phạt trên thế giới cực đoan WH", "",
         "| Bộ | Quy mô | Dải tải | n | Mất DV Dijkstra phạt (a / b) → Lai6 | C2 ΔJ | C4 Δ mất DV | C5 Lai6 / TKCB10 ms | Thắng |",
         "|---|---|---|---:|---|---|---|---|---|"]
    for bo, (thuong, ten) in BO_7A.items():
        for kd in CAC_K:
            o = ra["7a"].get(f"{bo}_{kd}")
            for d in (DAI if o else ()):
                if d not in o:
                    continue
                x = o[d]
                c5 = (f"{x['tg_lai6_ms']:.0f} / {x['tg_tkcb10_ms']:.0f}" if x["tg_tkcb10_ms"] else "–") + \
                     ("" if kd == "k4_5" else f" → {dd(x['dat_C5'])}")
                L.append(f"| {ten.split(' —')[0]} | {CAC_K[kd]} | {TEN_DAI[d]} | {x['so']} | {x['dphat_mat_dv_%']:.1f} "
                         f"({x['dphat_khong_tuyen_song_%']:.1f} / {x['dphat_du_phong_qua_tai_%']:.1f}) → {x['lai6_mat_dv_%']:.1f} | "
                         f"{_ci(x['C2'], 3)} {dd(x['dat_C2'])} | {_ci(x['C4'])} {dd(x['dat_C4'])} | {c5} | "
                         f"{'**có**' if x['thang'] else 'không'} |")
    L += ["", "**So với bộ thường cùng cửa sổ (thí nghiệm 6)** — mức tăng mất dịch vụ (điểm %, ghép theo cửa sổ):", ""]
    for bo, (thuong, ten) in BO_7A.items():
        for kd in CAC_K:
            o = ra["7a"].get(f"{bo}_{kd}", {}).get("so_voi_thuong")
            if o:
                n_wh = sum(ra["7a"][f"{bo}_{kd}"][d]["thang"] for d in DAI if d in ra["7a"][f"{bo}_{kd}"])
                n_th = sum(o["thuong"][d]["thang"] for d in DAI)
                c4_wh = np.mean([ra["7a"][f"{bo}_{kd}"][d]["C4"][0] for d in DAI if d in ra["7a"][f"{bo}_{kd}"]])
                c4_th = np.mean([o["thuong"][d]["C4"][0] for d in DAI])
                L.append(f"- {ten}, {CAC_K[kd]}: Dijkstra phạt tăng {_ci(o['dphat_tang_diem_%'])}, Lai6 tăng "
                         f"{_ci(o['lai6_tang_diem_%'])}; số ô thắng WH {n_wh}/3 so với thường {n_th}/3; C4 trung bình "
                         f"WH {c4_wh:+.2f} so với thường {c4_th:+.2f}")
    L += ["", "**Biến thể (4–5 luồng, gộp mọi dải tải)** — mất DV Dijkstra phạt (a / b) → Lai6, C4 gộp:", ""]
    for b, ten in (("c", "C ngày giông"), ("y", "A Yagi")):
        for s, ts in BIEN_THE.items():
            x = ra["7a"].get(f"bien_the_{b}7{s}")
            if x:
                L.append(f"- {ten}, {ts}: {x['dphat_mat_dv_%']:.1f} ({x['dphat_khong_tuyen_song_%']:.1f} / "
                         f"{x['dphat_du_phong_qua_tai_%']:.1f}) → {x['lai6_mat_dv_%']:.1f}; C4 {_ci(x['C4_gop'])}")
    L += ["", "## 7b — DM thuần theo khối lượng dữ liệu (4–5 luồng, gộp mọi dải tải)", "",
          "| Bộ test | Dijkstra phạt | DM 25% | DM 50% | DM 100% (DM6) | DM 200% | 200% − 100% (điểm %) |",
          "|---|---|---|---|---|---|---|"]
    for bo, ten in BO_7B.items():
        o = ra["7b"].get(bo)
        if not o:
            continue
        cot = [f"{o['mo_hinh'][m]['mat_dv_%']:.2f}" if m in o["mo_hinh"] else "–" for m in MO_HINH_7B]
        h = f"{_ci(o['200_tru_100'])} {dd(o['H7b_dat'])}" if "200_tru_100" in o else "–"
        L.append(f"| {ten} | {o['dphat_mat_dv_%']:.2f} | " + " | ".join(cot) + f" | {h} |")
    L += ["", "Giảm đơn điệu 25 → 50 → 100 → 200%: " + "; ".join(
        f"{BO_7B[bo]} {'có' if o['giam_don_dieu'] else 'không'}" for bo, o in ra["7b"].items() if "giam_don_dieu" in o),
          "", "25% − 100% (điểm %): " + "; ".join(f"{BO_7B[bo]} {_ci(o['25_tru_100'])}" for bo, o in ra["7b"].items()
                                                 if "25_tru_100" in o),
          "", "200% − 100% về J: " + "; ".join(f"{BO_7B[bo]} {_ci(o['200_tru_100_J'], 3)}" for bo, o in ra["7b"].items()
                                               if "200_tru_100_J" in o),
          "", "**Theo dải tải** — mất DV (%) Dijkstra phạt | DM 25 / 50 / 100 / 200%:", "",
          "| Bộ test | Dải | Dijkstra phạt | " + " | ".join(f"DM {MO_HINH_7B[m]}%" for m in MO_HINH_7B) + " |",
          "|---|---|---|" + "---|" * len(MO_HINH_7B)]
    for bo, ten in BO_7B.items():
        o = ra["7b"].get(bo)
        for d in (DAI if o else ()):
            L.append(f"| {ten} | {TEN_DAI[d]} | {o['dphat_theo_dai'][d]:.2f} | " + " | ".join(
                f"{o['mo_hinh'][m]['theo_dai'][d]:.2f}" if m in o["mo_hinh"] else "–" for m in MO_HINH_7B) + " |")
    (KQ / "KET_QUA_7.md").write_text("\n".join(L) + "\n", encoding="utf-8")


MAU = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]      # 5 ô đầu bảng màu chuẩn, thứ tự cố định


def ve_hinh(ra):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
                         "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
                         "axes.edgecolor": "#8a8984", "axes.labelcolor": "#3b3a36", "xtick.color": "#52514e",
                         "ytick.color": "#52514e"})
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [1.3, 1]})
    luoi = dict(color="#e4e3df", lw=0.8, zorder=0)
    chu = dict(frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0), borderaxespad=0.2, fontsize=9)
    nhan, dp, la = [], [], []
    for b, ten, thuong in (("c", "C", "c6"), ("y", "Yagi", "yagi5")):
        rt = nap(thuong, "k4_5")
        if rt and all("dm6_tkcb" in r["ens"] for r in rt):
            nhan.append(f"{ten}\nthường")
            dp.append(np.mean([r["ens"]["dphat"]["mat_dv_%"] for r in rt]))
            la.append(np.mean([lai6(r)["mat_dv_%"] for r in rt]))
        for s in BIEN_THE:
            x = ra["7a"].get(f"bien_the_{b}7{s}")
            if x:
                nhan.append(f"{ten}\n{s.upper()}")
                dp.append(x["dphat_mat_dv_%"])
                la.append(x["lai6_mat_dv_%"])
    xs = np.arange(len(nhan))
    ax[0].bar(xs - 0.2, dp, 0.38, color=MAU[0], label="Dijkstra phạt", zorder=2)
    ax[0].bar(xs + 0.2, la, 0.38, color=MAU[1], label="Lai6 (DM6)", zorder=2)
    ax[0].set_xticks(xs, nhan, fontsize=8.5)
    ax[0].set_ylabel("Lưu lượng mất dịch vụ (%)")
    ax[0].set_title("(a) 7a — thế giới thường và cực đoan (4–5 luồng)", loc="left", fontsize=11, pad=40)
    ax[0].grid(axis="y", **luoi)
    ax[0].legend(ncol=2, **chu)
    for j, (bo, ten) in enumerate(BO_7B.items()):
        o = ra["7b"].get(bo)
        if not o:
            continue
        ms = [m for m in MO_HINH_7B if m in o["mo_hinh"]]
        ax[1].plot([MO_HINH_7B[m] for m in ms], [o["mo_hinh"][m]["tru_dphat"][0] for m in ms], "-o", color=MAU[j], lw=2,
                   ms=6, mec="white", mew=1.5, label=ten, zorder=3)
    ax[1].axhline(0, color="#8a8984", lw=1)
    ax[1].text(0.98, 0.96, "dưới 0: DM ít mất dịch vụ hơn Dijkstra phạt", transform=ax[1].transAxes, ha="right", va="top",
               fontsize=8.5, color="#52514e")
    ax[1].set_xscale("log", base=2)
    ax[1].set_xticks([25, 50, 100, 200], ["25%", "50%", "100%", "200%"])
    ax[1].set_xlabel("Khối lượng dữ liệu huấn luyện (so với DM6)")
    ax[1].set_ylabel("DM − Dijkstra phạt, mất DV (điểm %)")
    ax[1].set_title("(b) 7b — DM thuần theo khối lượng dữ liệu", loc="left", fontsize=11, pad=40)
    ax[1].grid(**luoi)
    ax[1].legend(ncol=3, **chu)
    fig.tight_layout()
    fig.savefig(KQ / "hinh_7.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    r = phan_tich()
    print(json.dumps({k: {d: v[d]["thang"] for d in DAI if d in v} for k, v in r["7a"].items() if not k.startswith("bien")},
                     ensure_ascii=False))
    print(json.dumps({k: v.get("H7b_dat") for k, v in r["7b"].items()}, ensure_ascii=False))
