"""
phan_tich_yagi.py — Thống kê và hình cho thí nghiệm "bão Yagi + nhiều luồng" theo đúng tiêu chí đã chốt
(results/yagi/TIEU_CHI.md). Khoảng tin cậy 95%: bootstrap ghép cặp theo 16 kịch bản Yagi (lấy lại kịch bản,
giữ mọi tình huống của kịch bản đó), 10 000 lần.

    python test/phan_tich_yagi.py      # → results/yagi/phan_tich.json, hinh_yagi_*.png
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

GOC = Path(__file__).resolve().parents[1]
KQ = GOC / "results" / "yagi"
B = 10_000
MUC = ("nhe", "vua", "nang")
TEN_MUC = {"nhe": "nhẹ", "vua": "vừa", "nang": "nặng"}
TEN_K = {"k20_40": "20–40 luồng", "k80": "80 luồng", "k160": "160 luồng"}
PP = ("doc_lap", "tuan_tu", "SM", "DM", "LS_cung_tg", "LS1", "LS10")
TEN_PP = {"doc_lap": "Dijkstra độc lập", "tuan_tu": "Dijkstra tuần tự", "SM": "Softmax (S = 16)",
          "DM": "DM (S = 16)", "LS_cung_tg": "TKCB cùng thời gian DM", "LS1": "TKCB 1 lần", "LS10": "TKCB 10 lần"}
MAU = {"doc_lap": "#b8b6af", "tuan_tu": "#8a8984", "SM": "#eb6834", "DM": "#2a78d6", "LS_cung_tg": "#7fb3e6",
       "LS1": "#3b3a36", "LS10": "#000000"}
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "TeX Gyre Termes", "DejaVu Serif"],
                     "axes.spines.top": False, "axes.spines.right": False, "font.size": 11})


def bootstrap(v, kb, rng):
    """Trung bình của v và khoảng tin cậy 95% khi lấy lại theo kịch bản."""
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
        ra[ten_k] = {"kiem_tra_LS10_giong_luc_sinh": float(np.max(np.abs(d["LS10"] / d["LS10_luc_sinh"] - 1)))}
        for muc in MUC:
            s = d["muc"] == muc
            kb = d["kb"][s]
            r = {"so_tinh_huong": int(s.sum()), "K_tb": float(d["K"][s].mean())}
            ls10 = d["LS10"][s]
            r["chi_phi_tb_ms"] = {p: float(d[p][s].mean()) for p in PP}
            r["gap_%"] = {p: bootstrap((d[p][s] / ls10 - 1) * 100, kb, rng) for p in PP}
            r["gap_FW_can_duoi_%"] = float(((d["FW"][s] / ls10 - 1) * 100).mean())
            r["tg_trung_vi_ms"] = {p: float(np.median(d[f"tg_{p}"][s]) * 1e3) for p in ("DM", "SM", "LS10", "LS1", "tuan_tu")}
            r["qua_tai_%"] = {p: float(d[f"qua_tai_{p}"][s].mean() * 100) for p in ("tuan_tu", "LS10", "DM")}
            h1 = bootstrap(d["DM"][s] - d["tuan_tu"][s], kb, rng)
            h2 = bootstrap(d["DM"][s] - d["SM"][s], kb, rng)
            h_ls1 = bootstrap(d["DM"][s] - d["LS1"][s], kb, rng)
            h_cung = bootstrap(d["DM"][s] - d["LS_cung_tg"][s], kb, rng)
            g = r["gap_%"]["DM"][0]
            tg = r["tg_trung_vi_ms"]
            r["hieu_ms"] = {"DM-tuan_tu": h1, "DM-SM": h2, "DM-LS1": h_ls1, "DM-LS_cung_tg": h_cung}
            r["tieu_chi"] = {"C1": h1[1][1] < 0, "C2": h2[1][1] < 0, "C3": g <= 2.0,
                             "C4": tg["DM"] <= tg["LS10"] / 10}
            r["DM_thang"] = all(r["tieu_chi"].values())
            ra[ten_k][muc] = r
    json.dump(ra, open(KQ / "phan_tich.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    return ra


def ve(ra):
    # Hình 1: chất lượng – thời gian, thí nghiệm chính (mỗi điểm = một phương pháp)
    if "k20_40" in ra:
        fig, axs = plt.subplots(1, 2, figsize=(11, 5.0), sharey=True)
        lech = {"doc_lap": (8, 4), "tuan_tu": (8, 4), "SM": (8, 5), "DM": (-8, 8), "LS_cung_tg": (8, -14),
                "LS1": (8, -4), "LS10": (0, 16)}
        for ax, muc in zip(axs, ("vua", "nang")):
            r = ra["k20_40"][muc]
            for p in PP:
                g = r["gap_%"][p]
                tg = r["tg_trung_vi_ms"].get(p, r["tg_trung_vi_ms"]["DM"] if p == "LS_cung_tg" else None)
                if tg is None:
                    continue
                ax.errorbar(tg, g[0], yerr=[[g[0] - max(g[1][0], 0)], [g[1][1] - g[0]]], fmt="o", color=MAU[p],
                            ms=8, capsize=3, zorder=3)
                ha = "left" if lech[p][0] > 0 else ("right" if lech[p][0] < 0 else "center")
                ax.annotate(f"{TEN_PP[p]}\n{g[0]:.3f}%".replace(".", ","), (tg, g[0]), xytext=lech[p],
                            textcoords="offset points", fontsize=10, ha=ha, va="center")
            ax.set_xscale("log")
            ax.set_yscale("symlog", linthresh=0.01)
            ax.set_ylim(-0.007, 8)
            ax.set_xlim(1, 3000)
            ax.set_title(f"Tải {TEN_MUC[muc]}", loc="left")
            ax.set_xlabel("thời gian trung vị mỗi tình huống (ms, log)")
            ax.grid(alpha=0.25)
        axs[0].set_ylabel("vượt TKCB 10 lần (%), thang log gần 0")
        fig.suptitle("Bão Yagi, 20–40 luồng: chất lượng và thời gian (thanh = 95% KTC theo 16 kịch bản)",
                     x=0.01, ha="left", fontsize=12)
        fig.tight_layout()
        fig.savefig(KQ / "hinh_yagi_chi_phi.png", dpi=200)
        plt.close(fig)
    # Hình 2: quy mô
    ks = [k for k in TEN_K if k in ra]
    if len(ks) >= 2:
        from matplotlib.ticker import NullFormatter
        fig, axs = plt.subplots(1, 2, figsize=(12, 4.6))
        xk = [ra[k]["vua"]["K_tb"] for k in ks]
        for p in ("DM", "SM", "LS10", "LS1", "tuan_tu"):
            y = [np.mean([ra[k][m]["tg_trung_vi_ms"][p] for m in MUC]) for k in ks]
            axs[0].plot(xk, y, "o-", color=MAU[p], label=TEN_PP[p])
        for p in ("tuan_tu", "SM", "DM", "LS1"):
            y = [ra[k]["nang"]["gap_%"][p][0] for k in ks]
            axs[1].plot(xk, y, "o-", color=MAU[p])
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
        axs[1].set_ylim(0, 10)
        axs[1].set_ylabel("vượt TKCB 10 lần (%), tải nặng")
        axs[1].set_title("(b) Chất lượng ở tải nặng (DM, softmax không huấn luyện lại)", loc="left")
        h, l = axs[0].get_legend_handles_labels()
        fig.legend(h, l, loc="lower center", ncol=5, frameon=False, fontsize=10)
        fig.tight_layout(rect=(0, 0.07, 1, 1))
        fig.savefig(KQ / "hinh_yagi_quy_mo.png", dpi=200)
        plt.close(fig)


def boi_canh():
    """Bối cảnh 16 kịch bản Yagi: mức bão lớn nhất, tỉ lệ snapshot mà nút bão nặng nhất bị cắt khỏi mạng."""
    import glob
    import pickle
    tep = sorted(glob.glob(str(GOC / "data" / "yagi" / "test" / "k20_40" / "kb_*.pkl")))
    if not tep:
        return None
    hang = []
    for f in tep:
        ds = [x for x in pickle.load(open(f, "rb")) if x["muc"] == "vua"]
        hang.append({"gio": ds[0]["gio"], "so_snapshot": len(ds),
                     "diem_max_tb": float(np.mean([x["meta"]["diem_max"] for x in ds])),
                     "ti_le_bo_qua_%": float(np.mean([x["meta"]["diem_nguon"][0] < x["meta"]["diem_max"] - 1e-9 for x in ds]) * 100),
                     "so_nut_co_lap_tb": float(np.mean([len(x["meta"]["co_lap"]) for x in ds])),
                     "lk_het_dung_luong_tb": float(np.mean([x["meta"]["lien_ket_het_dung_luong"] for x in ds]))})
    fig, ax = plt.subplots(figsize=(12, 3.6))
    x = np.arange(len(hang))
    ax.bar(x, [h["ti_le_bo_qua_%"] for h in hang], color="#b8b6af", label="% snapshot: nút bão nặng nhất bị cắt khỏi mạng")
    ax.set_ylabel("% snapshot")
    ax.set_ylim(0, 105)
    ax2 = ax.twinx()
    ax2.plot(x, [h["diem_max_tb"] for h in hang], "o-", color="#2a78d6", label="mức bão lớn nhất (mưa mm/h + gió giật > 15 m/s)")
    ax2.set_ylabel("mức bão")
    ax2.spines["top"].set_visible(False)
    ax.set_xticks(x)
    ax.set_xticklabels([h["gio"][5:13].replace("T", "\n") + "h" for h in hang], fontsize=8.5)
    ax.set_xlabel("giờ bắt đầu kịch bản (UTC, 2024)")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, loc="upper right", fontsize=9)
    ax.set_title("16 kịch bản kiểm tra trong bão Yagi", loc="left")
    fig.tight_layout()
    fig.savefig(KQ / "hinh_yagi_boi_canh.png", dpi=200)
    plt.close(fig)
    json.dump(hang, open(KQ / "boi_canh.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    return hang


def in_bang(ra):
    for k, rk in ra.items():
        print(f"== {TEN_K[k]}   (kiểm tra LS10 = lúc sinh: sai lệch tối đa {rk['kiem_tra_LS10_giong_luc_sinh']:.1e})")
        for muc in MUC:
            r = rk[muc]
            g = r["gap_%"]
            tg = r["tg_trung_vi_ms"]
            print(f"  {TEN_MUC[muc]:5s} n={r['so_tinh_huong']}  gap%: " + "  ".join(f"{p} {g[p][0]:.3f}" for p in PP)
                  + f"  | FW {r['gap_FW_can_duoi_%']:.3f}")
            print(f"        ms: " + "  ".join(f"{p} {v:.1f}" for p, v in tg.items())
                  + "  | hiệu DM−TT {:.4f} [{:.4f}, {:.4f}]  DM−SM {:.4f} [{:.4f}, {:.4f}]".format(
                      r["hieu_ms"]["DM-tuan_tu"][0], *r["hieu_ms"]["DM-tuan_tu"][1], r["hieu_ms"]["DM-SM"][0],
                      *r["hieu_ms"]["DM-SM"][1]) + f"  | {r['tieu_chi']}  thắng={r['DM_thang']}")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ra = phan_tich()
    in_bang(ra)
    ve(ra)
