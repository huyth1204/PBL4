"""
phan_tich_5.py — Thí nghiệm 5 (results/bao_da_tuyen/TIEU_CHI_5.md): chọn τ trên val5, rồi chấm chính sách lai.

    python -m scripts.phan_tich_5 chon-tau       # → results/bao_da_tuyen/chon_tau.json
    python -m scripts.phan_tich_5 phan-tich      # → ket_qua_5.json, KET_QUA_5.md, hinh_5.png

Lai(τ): ρ̂ < τ → lời giải DM; ρ̂ ≥ τ → lời giải DM + TKCB (xuất phát từ chính mẫu DM đó). Thời gian Lai = thời gian DM
(+ TKCB khi dùng). Số chính trên 32 thế giới CHẤM; CI 95% bootstrap ghép cặp theo 13 cửa sổ, 10 000 lần.
"""

from __future__ import annotations

import glob
import json
import math
import pickle
import sys
from pathlib import Path

import numpy as np

from scripts.phan_tich_dt import bootstrap

GOC = Path(__file__).resolve().parents[1]
DL = GOC / "data" / "bao_da_tuyen"
KQ = GOC / "results" / "bao_da_tuyen"
TAU = (0.0, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, math.inf)
DAI = {"thap": (0.5, 0.75), "giua": (0.75, 1.0), "cao": (1.0, 1.3001)}
TEN_DAI = {"thap": "thấp [0,5; 0,75)", "giua": "giữa [0,75; 1,0)", "cao": "cao [1,0; 1,3]"}
BO = {"mua5": "B — Mưa lớn mới (bộ chính, độc lập)", "yagi5": "A — Yagi, thực hiện mới (bộ phụ)"}
CAC_K = {"k4_5": "4–5 luồng", "k10": "10 luồng", "k20": "20 luồng"}
PP = ("yen", "suurballe", "dphat", "online_tkcb", "dm", "dm_tkcb", "lai")
TEN_PP = {"yen": "Yen", "suurballe": "Suurballe", "dphat": "Dijkstra phạt (đối thủ chính)",
          "online_tkcb": "TKCB trên J bảng", "dm": "DM", "dm_tkcb": "DM + TKCB", "lai": "Lai (DM / DM + TKCB theo ρ̂)"}
SO_DO = ("J", "mat_dv_%", "khong_tuyen_song_%", "du_phong_qua_tai_%", "tre_chinh_ms")


def nap(tap, kd):
    return [r for f in sorted(glob.glob(str(DL / f"{tap}_{kd}" / "kb_*.pkl"))) for r in pickle.load(open(f, "rb"))]


def ket_qua(r, p, tau, tw="ens"):
    if p == "lai":
        p = "dm" if r["rho_hat"] < tau else "dm_tkcb"
    return r[tw][p]


def tg(r, p, tau):
    if p == "lai":
        p = "dm" if r["rho_hat"] < tau else "dm_tkcb"
    return r["tg_ms"].get(p)


def chon_tau():
    recs = nap("val5", "k4_5")
    bang = {}
    for tau in TAU:
        bang[str(tau)] = {"J": float(np.mean([ket_qua(r, "lai", tau)["J"] for r in recs])),
                          "mat_dv_%": float(np.mean([ket_qua(r, "lai", tau)["mat_dv_%"] for r in recs]))}
    tot = min(TAU, key=lambda tau: (round(bang[str(tau)]["J"], 9), -tau))
    theo_dai = {}
    for d, (lo, hi) in DAI.items():
        rs = [r for r in recs if lo <= r["rho_hat"] < hi]
        theo_dai[d] = {"so": len(rs), **{p: {k: float(np.mean([r["ens"][p][k] for r in rs])) for k in ("J", "mat_dv_%")}
                                         for p in ("dphat", "dm", "dm_tkcb")}}
    ra = {"tau": tot, "bang": bang, "theo_dai_val5": theo_dai, "so_ban_ghi": len(recs),
          "quy_tac": "J trung bình (thế giới chấm) nhỏ nhất trên val5; hoà → τ lớn hơn"}
    json.dump(ra, open(KQ / "chon_tau.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1,
              default=lambda x: str(x))
    return ra


def phan_tich():
    tau = float(json.loads((KQ / "chon_tau.json").read_text(encoding="utf-8"))["tau"])
    rng = np.random.default_rng(0)
    ra = {"tau": tau}
    for bo in BO:
        for kd in CAC_K:
            recs = nap(bo, kd)
            if not recs or "dm_tkcb" not in recs[0]["ens"]:
                continue
            o = {"so_ban_ghi": len(recs), "rho_hat_lech_he_so_tai_max": float(max(abs(r["rho_hat"] - r["muc"]) for r in recs))}
            for d, (lo, hi) in DAI.items():
                rs = [r for r in recs if lo <= r["rho_hat"] < hi]
                if not rs:
                    continue
                kb = np.array([r["kb"] for r in rs])
                v = lambda p, k, tw="ens": np.array([ket_qua(r, p, tau, tw)[k] for r in rs])
                x = {"so_tinh_huong": len(rs), "so_cua_so": int(len(np.unique(kb))),
                     "ti_le_dung_tkcb_%": float(np.mean([r["rho_hat"] >= tau for r in rs]) * 100), "phuong_phap": {}}
                for p in PP:
                    x["phuong_phap"][p] = {k: float(v(p, k).mean()) for k in SO_DO}
                    x["phuong_phap"][p]["qs_mat_dv_%"] = float(v(p, "mat_dv_%", "qs").mean())
                    t = [tg(r, p, tau) for r in rs if tg(r, p, tau) is not None]
                    if t:
                        x["phuong_phap"][p]["tg_trung_vi_ms"] = float(np.median(t))
                t10 = [r["tg_ms"]["online_tkcb10"] for r in rs if "online_tkcb10" in r["tg_ms"]]
                x["tg_trung_vi_tkcb10_ms"] = float(np.median(t10)) if t10 else None
                C2 = bootstrap(v("lai", "J") - v("dphat", "J"), kb, rng)
                C4 = bootstrap(v("lai", "mat_dv_%") - v("dphat", "mat_dv_%"), kb, rng)
                tl = x["phuong_phap"]["lai"]["tg_trung_vi_ms"]
                C5 = bool(t10) and tl <= x["tg_trung_vi_tkcb10_ms"] and tl <= 1000
                h = lambda a, b, k: bootstrap(v(a, k) - v(b, k), kb, rng)
                x["tieu_chi"] = {"C2_J_lai_tru_dphat": C2, "C2": C2[1][1] < 0,
                                 "C4_mat_dv_lai_tru_dphat": C4, "C4": C4[1][1] < 0, "C5": C5,
                                 "C4a": h("lai", "dphat", "khong_tuyen_song_%"), "C4b": h("lai", "dphat", "du_phong_qua_tai_%"),
                                 "lai_tru_dm_J": h("lai", "dm", "J"), "lai_tru_dm_mat_dv": h("lai", "dm", "mat_dv_%"),
                                 "lai_tru_dmtkcb_J": h("lai", "dm_tkcb", "J"),
                                 "lai_tru_dmtkcb_mat_dv": h("lai", "dm_tkcb", "mat_dv_%"),
                                 "lai_tru_online_J": h("lai", "online_tkcb", "J"),
                                 "lai_tru_online_mat_dv": h("lai", "online_tkcb", "mat_dv_%"),
                                 "qs_lai_tru_dphat_mat_dv": bootstrap(v("lai", "mat_dv_%", "qs") - v("dphat", "mat_dv_%", "qs"),
                                                                      kb, rng)}
                x["tieu_chi"]["thang"] = bool(x["tieu_chi"]["C2"] and x["tieu_chi"]["C4"] and (kd == "k4_5" or C5))
                o[d] = x
            ra[f"{bo}_{kd}"] = o
    json.dump(ra, open(KQ / "ket_qua_5.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    viet_md(ra)
    ve_hinh(ra)
    return ra


def _ci(v, nd=2):
    return f"{v[0]:+.{nd}f} [{v[1][0]:+.{nd}f}; {v[1][1]:+.{nd}f}]"


def viet_md(ra):
    dd = lambda b: "**đạt**" if b else "không đạt"
    L = [f"# Thí nghiệm 5 — kết quả (sinh tự động bởi `phan_tich_5.py`)", "",
         f"τ chọn trên val5 = **{ra['tau']:g}** (ρ̂ ≥ τ → DM + TKCB). Số chính trên 32 thế giới CHẤM; CI 95% bootstrap "
         "ghép cặp theo 13 cửa sổ. Dải tải theo ρ̂.", "",
         "## Bảng tóm tắt tiêu chí", "", "| Bộ | Quy mô | Dải tải | n | C2 ΔJ Lai − Dijkstra phạt | C4 Δ mất DV (điểm %) | "
         "C5 Lai / TKCB 10 lần (ms) | Thắng |", "|---|---|---|---:|---|---|---|---|"]
    for bo in BO:
        for kd in CAC_K:
            o = ra.get(f"{bo}_{kd}")
            if not o:
                continue
            for d in DAI:
                if d not in o:
                    continue
                x, tc = o[d], o[d]["tieu_chi"]
                c5 = (f"{x['phuong_phap']['lai']['tg_trung_vi_ms']:.0f} / "
                      f"{x['tg_trung_vi_tkcb10_ms']:.0f}" if x["tg_trung_vi_tkcb10_ms"] else "–")
                c5 += "" if kd == "k4_5" else f" → {dd(tc['C5'])}"
                L.append(f"| {BO[bo][0]} | {CAC_K[kd]} | {TEN_DAI[d]} | {x['so_tinh_huong']} | {_ci(tc['C2_J_lai_tru_dphat'], 3)} "
                         f"{dd(tc['C2'])} | {_ci(tc['C4_mat_dv_lai_tru_dphat'])} {dd(tc['C4'])} | {c5} | "
                         f"{'**có**' if tc['thang'] else 'không'} |")
    for bo in BO:
        for kd in CAC_K:
            o = ra.get(f"{bo}_{kd}")
            if not o:
                continue
            for d in DAI:
                if d not in o:
                    continue
                x, tc = o[d], o[d]["tieu_chi"]
                L += ["", f"## {BO[bo]}, {CAC_K[kd]}, tải {TEN_DAI[d]} ({x['so_tinh_huong']} tình huống, {x['so_cua_so']} cửa sổ, "
                      f"Lai dùng TKCB ở {x['ti_le_dung_tkcb_%']:.0f}%)", "",
                      "| Phương pháp | J | Mất DV % | (a) hết tuyến % | (b) dự phòng quá tải % | Trễ chính ms | "
                      "Mất DV % (quan sát) | Thời gian ms |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
                for p, y in x["phuong_phap"].items():
                    t_ = f"{y['tg_trung_vi_ms']:.0f}" if "tg_trung_vi_ms" in y else "–"
                    L.append(f"| {TEN_PP[p]} | {y['J']:.3f} | {y['mat_dv_%']:.2f} | {y['khong_tuyen_song_%']:.2f} | "
                             f"{y['du_phong_qua_tai_%']:.2f} | {y['tre_chinh_ms']:.1f} | {y['qs_mat_dv_%']:.2f} | {t_} |")
                L += ["", f"- Lai − Dijkstra phạt: phần (a) {_ci(tc['C4a'])}, phần (b) {_ci(tc['C4b'])}; thế giới quan sát "
                      f"{_ci(tc['qs_lai_tru_dphat_mat_dv'])}",
                      f"- Lai − DM: J {_ci(tc['lai_tru_dm_J'], 3)}, mất DV {_ci(tc['lai_tru_dm_mat_dv'])}; "
                      f"Lai − DM + TKCB: J {_ci(tc['lai_tru_dmtkcb_J'], 3)}, mất DV {_ci(tc['lai_tru_dmtkcb_mat_dv'])}; "
                      f"Lai − TKCB trên J bảng: J {_ci(tc['lai_tru_online_J'], 3)}, mất DV {_ci(tc['lai_tru_online_mat_dv'])}"]
    (KQ / "KET_QUA_5.md").write_text("\n".join(L) + "\n", encoding="utf-8")


MAU = {"dphat": "#2a78d6", "dm": "#eb6834", "dm_tkcb": "#1baf7a", "lai": "#eda100"}      # 4 ô đầu bảng màu chuẩn


def ve_hinh(ra):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
                         "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
                         "axes.edgecolor": "#8a8984", "axes.labelcolor": "#3b3a36", "xtick.color": "#52514e",
                         "ytick.color": "#52514e"})
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.8), gridspec_kw={"width_ratios": [1.4, 1.4, 1]})
    luoi = dict(color="#e4e3df", lw=0.8, zorder=0)
    chu = dict(frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0), borderaxespad=0.2, fontsize=9)
    for a, bo in zip(ax[:2], BO):
        nhom = [(kd, d) for kd in CAC_K if f"{bo}_{kd}" in ra for d in DAI if d in ra[f"{bo}_{kd}"]]
        x = np.arange(len(nhom))
        bw = 0.2
        for j, p in enumerate(("dphat", "dm", "dm_tkcb", "lai")):
            a.bar(x + (j - 1.5) * (bw + 0.01), [ra[f"{bo}_{kd}"][d]["phuong_phap"][p]["mat_dv_%"] for kd, d in nhom], bw,
                  color=MAU[p], label=TEN_PP[p], zorder=2)
        a.set_xticks(x, [f"{CAC_K[kd].split()[0]} luồng\n{TEN_DAI[d].split()[0]}" for kd, d in nhom], fontsize=8.5)
        a.set_ylabel("Lưu lượng mất dịch vụ (%)")
        a.set_title(f"({'a' if bo == 'mua5' else 'b'}) {BO[bo].split(' (')[0]}", loc="left", fontsize=11, pad=36)
        a.grid(axis="y", **luoi)
        a.legend(ncol=2, **chu)
    ks = [kd for kd in CAC_K if f"mua5_{kd}" in ra and "cao" in ra[f"mua5_{kd}"]]
    so_k = [int(CAC_K[k].split()[0].split("–")[-1]) for k in ks]
    for p, ten, mau in (("lai", "Lai (dải tải cao)", MAU["lai"]), ("dm", "DM", MAU["dm"]),
                        ("dphat", "Dijkstra phạt", MAU["dphat"])):
        ax[2].plot(so_k, [ra[f"mua5_{k}"]["cao"]["phuong_phap"][p]["tg_trung_vi_ms"] for k in ks], "-o", color=mau, lw=2,
                   ms=6, mec="white", mew=1.5, label=ten, zorder=3)
    ax[2].plot(so_k, [ra[f"mua5_{k}"]["cao"]["tg_trung_vi_tkcb10_ms"] for k in ks], "-o", color="#3b3a36", lw=2, ms=6,
               mec="white", mew=1.5, label="TKCB 10 lần", zorder=3)
    ax[2].axhline(1000, color="#8a8984", lw=1, ls="--")
    ax[2].set_yscale("log")
    ax[2].set_xticks(so_k)
    ax[2].set_xlabel("Số luồng mỗi tình huống")
    ax[2].set_ylabel("Thời gian trung vị (ms, thang log)")
    ax[2].set_title("(c) Thời gian, bộ B, dải tải cao", loc="left", fontsize=11, pad=36)
    ax[2].grid(**luoi)
    ax[2].legend(ncol=2, **chu)
    fig.tight_layout()
    fig.savefig(KQ / "hinh_5.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if sys.argv[1] == "chon-tau":
        r = chon_tau()
        print(json.dumps(r, ensure_ascii=False, indent=1, default=str))
    else:
        r = phan_tich()
        print(json.dumps({k: {d: v[d]["tieu_chi"]["thang"] for d in DAI if d in v} for k, v in r.items() if k != "tau"},
                         ensure_ascii=False))
