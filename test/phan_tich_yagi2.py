"""
phan_tich_yagi2.py — Thống kê và hình cho thí nghiệm 2 theo đúng tiêu chí đã chốt (results/yagi2/TIEU_CHI_2.md).
Khoảng tin cậy 95%: bootstrap ghép cặp theo 16 kịch bản Yagi, 10 000 lần.

    python test/phan_tich_yagi2.py     # → results/yagi2/phan_tich.json, hinh_*.png
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

GOC = Path(__file__).resolve().parents[1]
KQ = GOC / "results" / "yagi2"
B = 10_000
MUC = ("nhe", "vua", "nang")
TEN_MUC = {"nhe": "nhẹ", "vua": "vừa", "nang": "nặng"}
TEN_K = {"k20_40": "20–40 luồng", "k80": "80 luồng", "k160": "160 luồng"}
PP = ("doc_lap", "tuan_tu", "LS10_uv", "ML", "DM", "DM_LSK", "LSK1", "LSK10")
TEN_PP = {"doc_lap": "Dijkstra độc lập", "tuan_tu": "Dijkstra tuần tự", "LS10_uv": "TKCB 10 lần trong 8 ứng viên",
          "ML": "Một-lần", "DM": "DM", "DM_LSK": "DM + TKCB-KGH", "LSK1": "TKCB-KGH 1 lần", "LSK10": "TKCB-KGH 10 lần (mốc)"}
TG = ("DM", "ML", "tuan_tu", "LSK1", "LSK10", "DM_LSK")
MAU = {"doc_lap": "#b8b6af", "tuan_tu": "#8a8984", "LS10_uv": "#c9a227", "ML": "#eb6834", "DM": "#2a78d6",
       "DM_LSK": "#7fb3e6", "LSK1": "#3b3a36", "LSK10": "#000000"}
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "TeX Gyre Termes", "DejaVu Serif"],
                     "axes.spines.top": False, "axes.spines.right": False, "font.size": 11})


def bootstrap(v, kb, rng):
    u = np.unique(kb)
    tong = np.array([v[kb == k].sum() for k in u])
    dem = np.array([(kb == k).sum() for k in u])
    j = rng.integers(0, len(u), (B, len(u)))
    m = tong[j].sum(1) / dem[j].sum(1)
    return float(v.mean()), [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def phan_tich():
    rng = np.random.default_rng(0)
    ra = {}
    for ten_k in TEN_K:
        f = KQ / f"test_{ten_k}.npz"
        if not f.exists():
            continue
        d = dict(np.load(f))
        ra[ten_k] = {"kiem_tra_LSK10_giong_luc_sinh": float(np.max(np.abs(d["LSK10"] / d["LSK10_luc_sinh"] - 1)))}
        for muc in MUC:
            s = d["muc"] == muc
            kb = d["kb"][s]
            moc = d["LSK10"][s]
            r = {"so_tinh_huong": int(s.sum()), "K_tb": float(d["K"][s].mean()), "so_cap_tb": float(d["C"][s].mean())}
            r["chi_phi_tb_ms"] = {p: float(d[p][s].mean()) for p in PP}
            r["gap_%"] = {p: bootstrap((d[p][s] / moc - 1) * 100, kb, rng) for p in PP}
            r["gap_FW_can_duoi_%"] = float(((d["FW"][s] / moc - 1) * 100).mean())
            r["tg_trung_vi_ms"] = {p: float(np.median(d[f"tg_{p}"][s]) * 1e3) for p in TG}
            r["qua_tai_%"] = {p: float(d[f"qua_tai_{p}"][s].mean() * 100) for p in ("tuan_tu", "LSK10", "DM")}
            h = {"DM-tuan_tu": bootstrap(d["DM"][s] - d["tuan_tu"][s], kb, rng),
                 "DM-ML": bootstrap(d["DM"][s] - d["ML"][s], kb, rng),
                 "DM-LS10_uv": bootstrap(d["DM"][s] - d["LS10_uv"][s], kb, rng),
                 "DM-LSK1": bootstrap(d["DM"][s] - d["LSK1"][s], kb, rng)}
            r["hieu_ms"] = h
            tg = r["tg_trung_vi_ms"]
            r["tieu_chi"] = {"C1": h["DM-tuan_tu"][1][1] < 0, "C2": h["DM-ML"][1][1] < 0,
                             "C3": r["gap_%"]["DM"][0] <= 2.0, "C4": tg["DM"] <= tg["LSK10"] / 10,
                             "C5": h["DM-LS10_uv"][1][1] < 0}
            r["DM_thang"] = all(r["tieu_chi"].values())
            r["dem"] = {"DM<tuan_tu": int((d["DM"][s] < d["tuan_tu"][s] - 1e-9).sum()),
                        "DM>tuan_tu": int((d["DM"][s] > d["tuan_tu"][s] + 1e-9).sum()),
                        "DM<LS10_uv": int((d["DM"][s] < d["LS10_uv"][s] - 1e-9).sum()),
                        "DM>LS10_uv": int((d["DM"][s] > d["LS10_uv"][s] + 1e-9).sum())}
            ra[ten_k][muc] = r
    json.dump(ra, open(KQ / "phan_tich.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    return ra


def ve(ra):
    if "k20_40" in ra:
        fig, axs = plt.subplots(1, 2, figsize=(11.5, 5.0), sharey=True)
        lech = {"tuan_tu": (8, 5), "LS10_uv": (8, 5), "ML": (8, 5), "DM": (-8, 9), "DM_LSK": (8, -12),
                "LSK1": (8, -5), "LSK10": (0, 16)}
        for ax, muc in zip(axs, ("vua", "nang")):
            r = ra["k20_40"][muc]
            for p in ("tuan_tu", "ML", "DM", "DM_LSK", "LSK1", "LSK10"):
                g = r["gap_%"][p]
                tg = r["tg_trung_vi_ms"][p]
                ax.errorbar(tg, g[0], yerr=[[g[0] - max(g[1][0], 0)], [max(g[1][1] - g[0], 0)]], fmt="o",
                            color=MAU[p], ms=8, capsize=3, zorder=3)
                ha = "left" if lech[p][0] > 0 else ("right" if lech[p][0] < 0 else "center")
                ax.annotate(f"{TEN_PP[p]}\n{g[0]:.3f}%".replace(".", ","), (tg, g[0]), xytext=lech[p],
                            textcoords="offset points", fontsize=10, ha=ha, va="center")
            g = r["gap_%"]["LS10_uv"][0]
            ax.axhline(g, color=MAU["LS10_uv"], ls="--", lw=1.2)
            ax.text(1.05, g, f" TKCB 10 lần trong 8 ứng viên: +{g:.2f}%".replace(".", ","), color="#7a6312",
                    fontsize=9.5, va="bottom")
            ax.set_xscale("log")
            ax.set_yscale("symlog", linthresh=0.01)
            ax.set_ylim(-0.007, 200)
            ax.set_xlim(1, 3000)
            ax.set_title(f"Tải {TEN_MUC[muc]}", loc="left")
            ax.set_xlabel("thời gian trung vị mỗi tình huống (ms, log)")
            ax.grid(alpha=0.25)
        axs[0].set_ylabel("vượt TKCB-KGH 10 lần (%), thang log gần 0")
        fig.suptitle("Thí nghiệm 2, bão Yagi, 20–40 luồng: chất lượng và thời gian (thanh = 95% KTC theo 16 kịch bản)",
                     x=0.01, ha="left", fontsize=12)
        fig.tight_layout()
        fig.savefig(KQ / "hinh_chat_luong_thoi_gian.png", dpi=200)
        plt.close(fig)
    ks = [k for k in TEN_K if k in ra]
    if len(ks) >= 2:
        from matplotlib.ticker import NullFormatter
        fig, axs = plt.subplots(1, 2, figsize=(12, 4.6))
        xk = [ra[k]["vua"]["K_tb"] for k in ks]
        for p in ("DM", "ML", "LSK10", "LSK1", "tuan_tu"):
            y = [np.mean([ra[k][m]["tg_trung_vi_ms"][p] for m in MUC]) for k in ks]
            axs[0].plot(xk, y, "o-", color=MAU[p], label=TEN_PP[p])
        for p in ("tuan_tu", "LS10_uv", "ML", "DM", "LSK1"):
            y = [ra[k]["nang"]["gap_%"][p][0] for k in ks]
            axs[1].plot(xk, y, "o-", color=MAU[p], label=TEN_PP[p] if p == "LS10_uv" else None)
        for ax in axs:
            ax.set_xscale("log")
            ax.set_xticks(xk)
            ax.set_xticklabels([f"{x:.0f}" for x in xk])
            ax.xaxis.set_minor_formatter(NullFormatter())
            ax.set_xlabel("số luồng K (trung bình)")
            ax.grid(alpha=0.25)
        axs[0].set_yscale("log")
        axs[0].set_ylabel("thời gian trung vị mỗi tình huống (ms)")
        axs[0].set_title("(a) Thời gian, 1 luồng CPU (trung bình 3 mức tải)", loc="left")
        axs[1].set_yscale("symlog", linthresh=0.01)
        axs[1].set_ylim(0, 300)
        axs[1].set_ylabel("vượt TKCB-KGH 10 lần (%), tải nặng")
        axs[1].set_title("(b) Chất lượng ở tải nặng (mô hình không huấn luyện lại)", loc="left")
        h, l = axs[0].get_legend_handles_labels()
        h2, l2 = axs[1].get_legend_handles_labels()
        fig.legend(h + h2, l + l2, loc="lower center", ncol=6, frameon=False, fontsize=9.5)
        fig.tight_layout(rect=(0, 0.08, 1, 1))
        fig.savefig(KQ / "hinh_quy_mo.png", dpi=200)
        plt.close(fig)


def in_bang(ra):
    for k, rk in ra.items():
        print(f"== {TEN_K[k]}   (LSK10 = lúc sinh: sai lệch tối đa {rk['kiem_tra_LSK10_giong_luc_sinh']:.1e})")
        for muc in MUC:
            r = rk[muc]
            g, tg = r["gap_%"], r["tg_trung_vi_ms"]
            print(f"  {TEN_MUC[muc]:5s} n={r['so_tinh_huong']} cặp={r['so_cap_tb']:.1f}  gap%: "
                  + "  ".join(f"{p} {g[p][0]:.3f}" for p in PP) + f"  | FW {r['gap_FW_can_duoi_%']:.3f}")
            print("        ms: " + "  ".join(f"{p} {v:.1f}" for p, v in tg.items()))
            print("        hiệu: " + "  ".join(f"{n} {v[0]:.4f} [{v[1][0]:.4f}, {v[1][1]:.4f}]" for n, v in r["hieu_ms"].items())
                  + f"  | {r['tieu_chi']}  thắng={r['DM_thang']}  {r['dem']}")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ra = phan_tich()
    in_bang(ra)
    ve(ra)
