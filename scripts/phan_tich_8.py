"""
phan_tich_8.py — Thí nghiệm 8: đánh giá chéo 2 × 2 giữa thế giới A và thế giới B (results/bao_da_tuyen/TIEU_CHI_8.md).

    python -m scripts.phan_tich_8          # → results/bao_da_tuyen/{ket_qua_8.json, KET_QUA_8.md, hinh_8.png}

- Thế giới A: mô hình hỏng cũ (dữ liệu TN6: c6, yagi5). Thế giới B: UAV 12 / 15 m/s, UAV và trạm hỏng theo khoảng cách
  tới giông (c6_b, yagi5_b; ĐÚNG cửa sổ + seed của c6, yagi5).
- Mô hình học trong A: DM6 (+ dm6_s1, dm6_s2). Mô hình học trong B: dm8b_s0..s2. Chính sách Lai: DM nếu ρ̂ < 1,1, ngược lại
  DM + TKCB (như Lai6). So với Dijkstra phạt theo C2 / C4 / C5, CI 95% bootstrap ghép cặp theo 13 cửa sổ.
- Chạy được khi dữ liệu còn thiếu: phần nào chưa chấm thì bỏ qua và ghi "chưa có".
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from scripts.phan_tich_5 import CAC_K, DAI, TEN_DAI, nap
from scripts.phan_tich_dt import B, bootstrap
from scripts.phan_tich_7 import bootstrap_cua_so

GOC = Path(__file__).resolve().parents[1]
KQ = GOC / "results" / "bao_da_tuyen"
TAU = 1.1
KHOA_T10 = "online_tkcb10_6"
BO = {"A": {"c6": "C ngày giông", "yagi5": "A Yagi"}, "B": {"c6_b": "C ngày giông", "yagi5_b": "A Yagi"}}
CAP = {"c6": "c6_b", "yagi5": "yagi5_b"}                                # bộ A ↔ bộ B cùng cửa sổ / seed
HO = {"A": ("dm6", "dm6_s1", "dm6_s2"), "B": ("dm8b_s0", "dm8b_s1", "dm8b_s2")}
CHINH = {"A": "dm6", "B": "dm8b_s0"}
TEN_HO = {"A": "DM học trong A (DM6)", "B": "DM học trong B (DM8b)"}


def lai(r, m, tw="ens"):
    return r[tw][m if r["rho_hat"] < TAU else m + "_tkcb"]


def co(recs, m):
    return bool(recs) and all(m in r["ens"] and m + "_tkcb" in r["ens"] for r in recs)


def o_tieu_chi(rs, m, kd, rng):
    """Một ô (bộ × quy mô × dải): Lai(m) so với Dijkstra phạt."""
    kb = np.array([r["kb"] for r in rs])
    v = lambda k, pp=None: np.array([lai(r, m)[k] if pp is None else r["ens"][pp][k] for r in rs])
    C2 = bootstrap(v("J") - v("J", "dphat"), kb, rng)
    C4 = bootstrap(v("mat_dv_%") - v("mat_dv_%", "dphat"), kb, rng)
    t10 = [r["tg_ms"][KHOA_T10] for r in rs if KHOA_T10 in r["tg_ms"]]
    tl = float(np.median([r["tg_ms"].get(m if r["rho_hat"] < TAU else m + "_tkcb", np.nan) for r in rs]))
    C5 = bool(t10) and tl <= float(np.median(t10)) and tl <= 1000
    x = {"so": len(rs), "dphat_mat_dv_%": float(v("mat_dv_%", "dphat").mean()), "lai_mat_dv_%": float(v("mat_dv_%").mean()),
         "C2": C2, "C4": C4, "dat_C2": C2[1][1] < 0, "dat_C4": C4[1][1] < 0, "dat_C5": C5, "te_hon": C4[1][0] > 0,
         "tg_lai_ms": tl, "tg_tkcb10_ms": float(np.median(t10)) if t10 else None}
    x["thang"] = bool(x["dat_C2"] and x["dat_C4"] and (kd == "k4_5" or C5))
    return x


def san_a(r):
    """Phần (a) KHÔNG tránh được với kho ứng viên: % lưu lượng (theo λ) mà không ứng viên nào của luồng sống."""
    from src.nhieu_luong.huan_luyen_dt import the_gioi_cham
    tg, lam = the_gioi_cham(r), np.asarray(r["gon"].lam, float)
    het = np.array([(~al.any(1)).mean() for al in tg.alive])                       # [K]
    return float((het * lam).sum() / lam.sum() * 100)


def phan_tich():
    rng = np.random.default_rng(0)
    ra = {"tau": TAU, "o": {}, "tom_tat": {}, "seed": {}, "dich_chuyen": {}, "san_a": {}}
    # ---- 1) ô tiêu chí của mô hình chính (seed 0) trên mọi bộ kiểm tra của cả hai thế giới
    for hoc, m in CHINH.items():
        for kt, bos in BO.items():
            for bo in bos:
                thang, co_du, so_o = 0, True, 0
                for kd in CAC_K:
                    recs = nap(bo, kd)
                    if not co(recs, m):
                        co_du = False
                        continue
                    o = {d: o_tieu_chi([r for r in recs if lo <= r["rho_hat"] < hi], m, kd, rng)
                         for d, (lo, hi) in DAI.items() if any(lo <= r["rho_hat"] < hi for r in recs)}
                    ra["o"][f"{hoc}|{bo}|{kd}"] = o
                    thang += sum(x["thang"] for x in o.values())
                    so_o += len(o)
                ra["tom_tat"][f"{hoc}|{kt}|{bo}"] = {"thang": thang, "du_9_o": co_du, "so_o": so_o}
    # ---- 2) độ ổn định theo seed (4–5 luồng, gộp mọi dải tải)
    for hoc, ho in HO.items():
        for kt, bos in BO.items():
            for bo in bos:
                recs = nap(bo, "k4_5")
                if not recs:
                    continue
                kb = np.array([r["kb"] for r in recs])
                hang = {}
                for m in ho:
                    if not co(recs, m):
                        continue
                    d = np.array([lai(r, m)["mat_dv_%"] - r["ens"]["dphat"]["mat_dv_%"] for r in recs])
                    thang = sum(o_tieu_chi([r for r in recs if lo <= r["rho_hat"] < hi], m, "k4_5", rng)["thang"]
                                for lo, hi in DAI.values())
                    hang[m] = {"C4_gop": bootstrap(d, kb, rng), "thang_3_o": int(thang),
                               "lai_mat_dv_%": float(np.mean([lai(r, m)["mat_dv_%"] for r in recs]))}
                if hang:
                    dau = {np.sign(h["C4_gop"][0]) for h in hang.values()}
                    ra["seed"][f"{hoc}|{kt}|{bo}"] = {"mo_hinh": hang, "cung_dau": len(dau) == 1 and len(hang) == 3}
    # ---- 3) dịch chuyển thế giới A → B (cùng cửa sổ): Dijkstra phạt và các thành phần
    for a, b in CAP.items():
        ra_, rb = nap(a, "k4_5"), nap(b, "k4_5")
        if not ra_ or not rb:
            continue
        gom = lambda rr, k: {w: [r["ens"]["dphat"][k] for r in rr if r["kb"] == w] for w in {r["kb"] for r in rr}}
        ra["dich_chuyen"][a] = {k: {"A": float(np.mean([r["ens"]["dphat"][k] for r in ra_])),
                                    "B": float(np.mean([r["ens"]["dphat"][k] for r in rb])),
                                    "B_tru_A": bootstrap_cua_so(gom(rb, k), gom(ra_, k), rng)}
                                for k in ("mat_dv_%", "khong_tuyen_song_%", "du_phong_qua_tai_%")}
    # ---- 4) sàn của phần (a): không tránh được với kho ứng viên
    for kt, bos in BO.items():
        for bo in bos:
            recs = nap(bo, "k4_5")
            if not recs or "cham" not in recs[0]:
                continue
            s = {"san_a_%": float(np.mean([san_a(r) for r in recs])),
                 "dphat_a_%": float(np.mean([r["ens"]["dphat"]["khong_tuyen_song_%"] for r in recs]))}
            for hoc, m in CHINH.items():
                if co(recs, m):
                    s[f"lai_{hoc}_a_%"] = float(np.mean([lai(r, m)["khong_tuyen_song_%"] for r in recs]))
            ra["san_a"][f"{kt}|{bo}"] = s
    (KQ / "ket_qua_8.json").write_text(json.dumps(ra, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    viet_md(ra)
    ve_hinh(ra)
    return ra


def _ci(v, nd=2):
    return f"{v[0]:+.{nd}f} [{v[1][0]:+.{nd}f}; {v[1][1]:+.{nd}f}]"


def viet_md(ra):
    L = ["# Thí nghiệm 8 — kết quả (sinh tự động bởi `phan_tich_8.py`)", "",
         "Lai(·) = DM nếu ρ̂ < 1,1, ngược lại DM + TKCB. So với Dijkstra phạt; CI 95% bootstrap ghép cặp theo 13 cửa sổ.", "",
         "## 1. Bảng chéo 2 × 2 — số ô thắng (C2 + C4, thêm C5 ở 10 / 20 luồng), mô hình seed 0", "",
         "| Mô hình (thế giới học) | A · C ngày giông | A · Yagi | B · C ngày giông | B · Yagi |", "|---|---|---|---|---|"]
    for hoc in ("A", "B"):
        o = []
        for kt, bos in BO.items():
            for bo in bos:
                x = ra["tom_tat"].get(f"{hoc}|{kt}|{bo}")
                o.append("chưa có" if x is None or not x.get("so_o") else
                         f"{x['thang']}/9" + ("" if x["du_9_o"] else f" (mới {x['so_o']} ô)"))
        L.append(f"| {TEN_HO[hoc]} | " + " | ".join(o) + " |")
    L += ["", "Ô trong miền: DM6 trên A, DM8b trên B. Ô chéo: DM6 trên B (không huấn luyện lại), DM8b trên A.", "",
          "## 2. Chi tiết từng ô", "",
          "| Mô hình | Bộ | Quy mô | Dải | n | Mất DV Dijkstra → Lai | C2 ΔJ | C4 Δ mất DV | C5 ms | Thắng |",
          "|---|---|---|---|---:|---|---|---|---|---|"]
    for k, o in ra["o"].items():
        hoc, bo, kd = k.split("|")
        for d, x in o.items():
            c5 = f"{x['tg_lai_ms']:.0f} / {x['tg_tkcb10_ms']:.0f}" if x["tg_tkcb10_ms"] else "–"
            L.append(f"| {CHINH[hoc]} | {bo} | {CAC_K[kd]} | {TEN_DAI[d]} | {x['so']} | {x['dphat_mat_dv_%']:.1f} → "
                     f"{x['lai_mat_dv_%']:.1f} | {_ci(x['C2'], 3)} | {_ci(x['C4'])} | {c5} | "
                     f"{'**có**' if x['thang'] else ('TỆ HƠN' if x['te_hon'] else 'không')} |")
    L += ["", "## 3. Độ ổn định theo seed huấn luyện (4–5 luồng, gộp mọi dải tải)", "",
          "| Họ mô hình | Bộ | seed: Lai − Dijkstra (điểm %) · số ô thắng /3 | Cùng dấu ở cả 3 seed |", "|---|---|---|---|"]
    for k, x in ra["seed"].items():
        hoc, kt, bo = k.split("|")
        o = "; ".join(f"{m}: {_ci(h['C4_gop'])} · {h['thang_3_o']}/3" for m, h in x["mo_hinh"].items())
        L.append(f"| {TEN_HO[hoc]} | {kt} · {bo} | {o} | {'có' if x['cung_dau'] else 'không / thiếu seed'} |")
    L += ["", "## 4. Dịch chuyển thế giới A → B của Dijkstra phạt (4–5 luồng, ghép theo cửa sổ)", "",
          "| Bộ | Mất DV A → B | (a) mọi tuyến cùng đứt A → B | (b) dự phòng quá tải A → B |", "|---|---|---|---|"]
    for a, x in ra["dich_chuyen"].items():
        L.append(f"| {a} | " + " | ".join(f"{x[k]['A']:.2f} → {x[k]['B']:.2f} ({_ci(x[k]['B_tru_A'])})"
                                          for k in ("mat_dv_%", "khong_tuyen_song_%", "du_phong_qua_tai_%")) + " |")
    L += ["", "## 5. Phần (a) không tránh được (sàn của kho ứng viên, 4–5 luồng)", "",
          "| Thế giới · bộ | Sàn (a) | Dijkstra phạt (a) | Lai A (a) | Lai B (a) |", "|---|---:|---:|---:|---:|"]
    for k, x in ra["san_a"].items():
        L.append(f"| {k.replace('|', ' · ')} | {x['san_a_%']:.2f} | {x['dphat_a_%']:.2f} | "
                 f"{x.get('lai_A_a_%', float('nan')):.2f} | {x.get('lai_B_a_%', float('nan')):.2f} |")
    L += ["", "Phần (a) vượt trên sàn là phần chọn tuyến tốt hơn còn cứu được."]
    (KQ / "KET_QUA_8.md").write_text("\n".join(L) + "\n", encoding="utf-8")


MAU = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#a3a29d"]


def ve_hinh(ra):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
                         "axes.spines.top": False, "axes.spines.right": False, "font.size": 10})
    cot = [(kt, bo) for kt, bos in BO.items() for bo in bos]
    nhan_cot = [f"{kt} · {BO[kt][bo]}" for kt, bo in cot]
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.8), gridspec_kw={"width_ratios": [1.1, 1.3, 1.1]})
    # (a) bảng chéo số ô thắng
    M = np.full((2, 4), np.nan)
    for i, hoc in enumerate(("A", "B")):
        for j, (kt, bo) in enumerate(cot):
            x = ra["tom_tat"].get(f"{hoc}|{kt}|{bo}")
            if x and x.get("so_o"):
                M[i, j] = x["thang"]
    ax[0].imshow(M, cmap="Greens", vmin=0, vmax=9, aspect="auto")
    for i in range(2):
        for j in range(4):
            ax[0].text(j, i, "–" if np.isnan(M[i, j]) else f"{int(M[i, j])}/9", ha="center", va="center", fontsize=12,
                       fontweight="bold" if (i == 0) == (j < 2) else "normal")
    ax[0].set_xticks(range(4), nhan_cot, fontsize=8.5, rotation=15)
    ax[0].set_yticks(range(2), ["DM học trong A", "DM học trong B"])
    ax[0].set_title("(a) Số ô thắng — đậm: trong miền, thường: chéo", loc="left", fontsize=10.5)
    # (b) mất dịch vụ 4–5 luồng: Dijkstra, Lai A, Lai B (vạch: min–max theo 3 seed)
    w = 0.26
    for j, (kt, bo) in enumerate(cot):
        recs = nap(bo, "k4_5")
        if not recs:
            continue
        ax[1].bar(j - w, np.mean([r["ens"]["dphat"]["mat_dv_%"] for r in recs]), w * .9, color=MAU[0],
                  label="Dijkstra phạt" if j == 0 else None)
        for k, hoc in enumerate(("A", "B")):
            gt = [np.mean([lai(r, m)["mat_dv_%"] for r in recs]) for m in HO[hoc] if co(recs, m)]
            if gt:
                ax[1].bar(j + k * w, np.mean(gt), w * .9, color=MAU[1 + k], label=f"Lai — {TEN_HO[hoc]}" if j == 0 else None,
                          yerr=[[np.mean(gt) - min(gt)], [max(gt) - np.mean(gt)]], capsize=3)
    ax[1].set_xticks(range(4), nhan_cot, fontsize=8.5, rotation=15)
    ax[1].set_ylabel("Mất dịch vụ (%), 4–5 luồng")
    ax[1].legend(frameon=False, fontsize=8.5)
    ax[1].set_title("(b) Mất dịch vụ theo thế giới kiểm tra", loc="left", fontsize=10.5)
    ax[1].grid(axis="y", color="#e4e3df", lw=0.8)
    # (c) phần (a): sàn không tránh được + phần tránh được
    keys = [k for k in ra["san_a"]]
    for j, k in enumerate(keys):
        x = ra["san_a"][k]
        for q, (ten, mau) in enumerate((("dphat_a_%", MAU[0]), ("lai_A_a_%", MAU[1]), ("lai_B_a_%", MAU[2]))):
            if ten in x:
                ax[2].bar(j + (q - 1) * w, x["san_a_%"], w * .9, color=MAU[4])
                ax[2].bar(j + (q - 1) * w, x[ten] - x["san_a_%"], w * .9, bottom=x["san_a_%"], color=mau)
    ax[2].set_xticks(range(len(keys)), [k.replace("|", " · ") for k in keys], fontsize=8.5, rotation=15)
    ax[2].set_ylabel("(a) mọi tuyến cùng đứt (%)")
    ax[2].set_title("(c) Phần (a): xám = không tránh được", loc="left", fontsize=10.5)
    ax[2].grid(axis="y", color="#e4e3df", lw=0.8)
    fig.tight_layout()
    fig.savefig(KQ / "hinh_8.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    r = phan_tich()
    print(json.dumps(r["tom_tat"], ensure_ascii=False, indent=1))
