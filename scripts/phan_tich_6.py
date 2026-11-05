"""
phan_tich_6.py — Thí nghiệm 6 (results/bao_da_tuyen/TIEU_CHI_6.md): chọn τ6 theo mất dịch vụ, chấm Lai6 trên C, B, A.

    python -m scripts.phan_tich_6 chon-tau       # → results/bao_da_tuyen/chon_tau_6.json
    python -m scripts.phan_tich_6 phan-tich      # → ket_qua_6.json, KET_QUA_6.md, hinh_6.png

Lai6(τ6): ρ̂ < τ6 → DM6; ρ̂ ≥ τ6 → DM6 + TKCB. Thời gian Lai6 = DM6 (+ TKCB khi dùng); mốc C5 = TKCB 10 lần đo trong
cùng lần chấm DM6 (khoá online_tkcb10_6). Số chính trên 32 thế giới CHẤM; CI 95% bootstrap ghép cặp theo 13 cửa sổ.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from scripts.phan_tich_5 import DAI, TAU, TEN_DAI, CAC_K, SO_DO, nap
from scripts.phan_tich_dt import bootstrap

GOC = Path(__file__).resolve().parents[1]
KQ = GOC / "results" / "bao_da_tuyen"
BO = {"c6": "C — Ngày giông mới (chính, chưa từng nhìn)", "mua5": "B — Mưa lớn của thí nghiệm 5 (chính)",
      "yagi5": "A — Yagi thực hiện mới (kiểm tra thoái lui)"}
PP = ("dphat", "online_tkcb", "dm", "dm_tkcb", "dm6", "dm6_tkcb", "lai6", "dm6a")
TEN_PP = {"dphat": "Dijkstra phạt (đối thủ chính)", "online_tkcb": "TKCB trên J bảng", "dm": "DM (thí nghiệm 4)",
          "dm_tkcb": "DM + TKCB (thí nghiệm 5)", "dm6": "DM6", "dm6_tkcb": "DM6 + TKCB",
          "lai6": "Lai6 (DM6 / DM6 + TKCB theo ρ̂)", "dm6a": "DM6a (không đặc trưng mới)"}
KHOA_TG10 = "online_tkcb10_6"


def ket_qua(r, p, tau, tw="ens"):
    if p == "lai6":
        p = "dm6" if r["rho_hat"] < tau else "dm6_tkcb"
    return r[tw][p]


def tg(r, p, tau):
    if p == "lai6":
        p = "dm6" if r["rho_hat"] < tau else "dm6_tkcb"
    return r["tg_ms"].get(p)


def chon_tau():
    recs = nap("val5", "k4_5") + nap("val6", "k4_5")
    bang = {str(t): {"mat_dv_%": float(np.mean([ket_qua(r, "lai6", t)["mat_dv_%"] for r in recs])),
                     "J": float(np.mean([ket_qua(r, "lai6", t)["J"] for r in recs]))} for t in TAU}
    tot = min(TAU, key=lambda t: (round(bang[str(t)]["mat_dv_%"], 9), -t))
    ra = {"tau": tot, "bang": bang, "so_ban_ghi": len(recs),
          "quy_tac": "mất dịch vụ trung bình (thế giới chấm) nhỏ nhất trên val5 + val6; hoà → τ lớn hơn"}
    json.dump(ra, open(KQ / "chon_tau_6.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
    return ra


def phan_tich():
    tau = float(json.loads((KQ / "chon_tau_6.json").read_text(encoding="utf-8"))["tau"])
    rng = np.random.default_rng(0)
    ra = {"tau": tau, "co_che": {}}
    for bo in BO:
        for kd in CAC_K:
            recs = nap(bo, kd)
            if not recs or "dm6_tkcb" not in recs[0]["ens"]:
                continue
            o = {"so_ban_ghi": len(recs)}
            pp = [p for p in PP if p == "lai6" or p in recs[0]["ens"]]
            for d, (lo, hi) in DAI.items():
                rs = [r for r in recs if lo <= r["rho_hat"] < hi]
                if not rs:
                    continue
                kb = np.array([r["kb"] for r in rs])
                v = lambda p, k, tw="ens": np.array([ket_qua(r, p, tau, tw)[k] for r in rs])
                x = {"so_tinh_huong": len(rs), "ti_le_dung_tkcb_%": float(np.mean([r["rho_hat"] >= tau for r in rs]) * 100),
                     "phuong_phap": {}}
                for p in pp:
                    x["phuong_phap"][p] = {k: float(v(p, k).mean()) for k in SO_DO}
                    x["phuong_phap"][p]["qs_mat_dv_%"] = float(v(p, "mat_dv_%", "qs").mean())
                    t = [tg(r, p, tau) for r in rs if tg(r, p, tau) is not None]
                    if t:
                        x["phuong_phap"][p]["tg_trung_vi_ms"] = float(np.median(t))
                t10 = [r["tg_ms"][KHOA_TG10] for r in rs if KHOA_TG10 in r["tg_ms"]]
                x["tg_trung_vi_tkcb10_ms"] = float(np.median(t10)) if t10 else None
                C2 = bootstrap(v("lai6", "J") - v("dphat", "J"), kb, rng)
                C4 = bootstrap(v("lai6", "mat_dv_%") - v("dphat", "mat_dv_%"), kb, rng)
                tl = x["phuong_phap"]["lai6"]["tg_trung_vi_ms"]
                C5 = bool(t10) and tl <= x["tg_trung_vi_tkcb10_ms"] and tl <= 1000
                h = lambda a, b, k: bootstrap(v(a, k) - v(b, k), kb, rng)
                x["tieu_chi"] = {"C2_J": C2, "C2": C2[1][1] < 0, "C4_mat_dv": C4, "C4": C4[1][1] < 0, "C5": C5,
                                 "C4a": h("lai6", "dphat", "khong_tuyen_song_%"),
                                 "C4b": h("lai6", "dphat", "du_phong_qua_tai_%"),
                                 "dm6_tru_dm_mat_dv": h("dm6", "dm", "mat_dv_%"),
                                 "qs_lai6_tru_dphat": bootstrap(v("lai6", "mat_dv_%", "qs") - v("dphat", "mat_dv_%", "qs"),
                                                                kb, rng)}
                x["tieu_chi"]["thang"] = bool(x["tieu_chi"]["C2"] and x["tieu_chi"]["C4"] and (kd == "k4_5" or C5))
                o[d] = x
            ra[f"{bo}_{kd}"] = o
            if kd == "k4_5" and bo in ("c6", "mua5"):
                kb = np.array([r["kb"] for r in recs])
                a = lambda p: np.array([r["ens"][p]["khong_tuyen_song_%"] for r in recs])
                ra["co_che"][bo] = {"a_dm6_tru_dm": bootstrap(a("dm6") - a("dm"), kb, rng),
                                    **({"a_dm6a_tru_dm": bootstrap(a("dm6a") - a("dm"), kb, rng)} if "dm6a" in pp else {}),
                                    "a_tb": {p: float(a(p).mean()) for p in ("dphat", "dm", "dm6", "dm6a") if p in pp}}
                ra["co_che"][bo]["dat"] = ra["co_che"][bo]["a_dm6_tru_dm"][1][1] < 0
    json.dump(ra, open(KQ / "ket_qua_6.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    viet_md(ra)
    ve_hinh(ra)
    return ra


def _ci(v, nd=2):
    return f"{v[0]:+.{nd}f} [{v[1][0]:+.{nd}f}; {v[1][1]:+.{nd}f}]"


def viet_md(ra):
    dd = lambda b: "**đạt**" if b else "không đạt"
    L = ["# Thí nghiệm 6 — kết quả (sinh tự động bởi `phan_tich_6.py`)", "",
         f"τ6 chọn trên val5 + val6 theo mất dịch vụ = **{ra['tau']:g}**. Số chính trên 32 thế giới CHẤM; CI 95% bootstrap "
         "ghép cặp theo 13 cửa sổ.", "", "## Tóm tắt tiêu chí (Lai6 so với Dijkstra phạt)", "",
         "| Bộ | Quy mô | Dải tải | n | C2 ΔJ | C4 Δ mất DV (điểm %) | C5 Lai6 / TKCB 10 lần (ms) | Thắng |",
         "|---|---|---|---:|---|---|---|---|"]
    for bo in BO:
        for kd in CAC_K:
            o = ra.get(f"{bo}_{kd}")
            for d in (DAI if o else ()):
                if d not in o:
                    continue
                x, tc = o[d], o[d]["tieu_chi"]
                c5 = (f"{x['phuong_phap']['lai6']['tg_trung_vi_ms']:.0f} / {x['tg_trung_vi_tkcb10_ms']:.0f}"
                      if x["tg_trung_vi_tkcb10_ms"] else "–") + ("" if kd == "k4_5" else f" → {dd(tc['C5'])}")
                L.append(f"| {BO[bo][0]} | {CAC_K[kd]} | {TEN_DAI[d]} | {x['so_tinh_huong']} | {_ci(tc['C2_J'], 3)} "
                         f"{dd(tc['C2'])} | {_ci(tc['C4_mat_dv'])} {dd(tc['C4'])} | {c5} | {'**có**' if tc['thang'] else 'không'} |")
    L += ["", "## Kiểm tra cơ chế (4–5 luồng, mọi dải tải): phần (a) mọi tuyến cùng đứt, DM6 − DM thí nghiệm 4", ""]
    for bo, c in ra["co_che"].items():
        L.append(f"- {BO[bo]}: DM6 − DM {_ci(c['a_dm6_tru_dm'])} → {dd(c['dat'])}"
                 + (f"; DM6a − DM {_ci(c['a_dm6a_tru_dm'])}" if "a_dm6a_tru_dm" in c else "")
                 + "; trung bình " + ", ".join(f"{TEN_PP[p]} {v:.2f}%" for p, v in c["a_tb"].items()))
    for bo in BO:
        for kd in CAC_K:
            o = ra.get(f"{bo}_{kd}")
            for d in (DAI if o else ()):
                if d not in o:
                    continue
                x, tc = o[d], o[d]["tieu_chi"]
                L += ["", f"## {BO[bo]}, {CAC_K[kd]}, tải {TEN_DAI[d]} ({x['so_tinh_huong']} tình huống; Lai6 dùng TKCB ở "
                      f"{x['ti_le_dung_tkcb_%']:.0f}%)", "",
                      "| Phương pháp | J | Mất DV % | (a) % | (b) % | Trễ chính ms | Mất DV % (quan sát) | Thời gian ms |",
                      "|---|---:|---:|---:|---:|---:|---:|---:|"]
                for p, y in x["phuong_phap"].items():
                    t_ = f"{y['tg_trung_vi_ms']:.0f}" if "tg_trung_vi_ms" in y else "–"
                    L.append(f"| {TEN_PP[p]} | {y['J']:.3f} | {y['mat_dv_%']:.2f} | {y['khong_tuyen_song_%']:.2f} | "
                             f"{y['du_phong_qua_tai_%']:.2f} | {y['tre_chinh_ms']:.1f} | {y['qs_mat_dv_%']:.2f} | {t_} |")
                L += ["", f"- Lai6 − Dijkstra phạt: (a) {_ci(tc['C4a'])}, (b) {_ci(tc['C4b'])}; thế giới quan sát "
                      f"{_ci(tc['qs_lai6_tru_dphat'])}; DM6 − DM về mất DV {_ci(tc['dm6_tru_dm_mat_dv'])}"]
    (KQ / "KET_QUA_6.md").write_text("\n".join(L) + "\n", encoding="utf-8")


MAU = {"dphat": "#2a78d6", "dm": "#eb6834", "lai6": "#1baf7a", "dm6": "#eda100"}      # 4 ô đầu bảng màu chuẩn


def ve_hinh(ra):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
                         "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
                         "axes.edgecolor": "#8a8984", "axes.labelcolor": "#3b3a36", "xtick.color": "#52514e",
                         "ytick.color": "#52514e"})
    bos = [b for b in BO if any(f"{b}_{kd}" in ra for kd in CAC_K)]
    fig, ax = plt.subplots(1, len(bos), figsize=(6.2 * len(bos), 4.8), squeeze=False)
    for a, bo in zip(ax[0], bos):
        nhom = [(kd, d) for kd in CAC_K if f"{bo}_{kd}" in ra for d in DAI if d in ra[f"{bo}_{kd}"]]
        x = np.arange(len(nhom))
        bw = 0.2
        for j, p in enumerate(("dphat", "dm", "dm6", "lai6")):
            a.bar(x + (j - 1.5) * (bw + 0.01), [ra[f"{bo}_{kd}"][d]["phuong_phap"][p]["mat_dv_%"] for kd, d in nhom], bw,
                  color=MAU[p], label=TEN_PP[p], zorder=2)
        a.set_xticks(x, [f"{CAC_K[kd].split()[0]} luồng\n{TEN_DAI[d].split()[0]}" for kd, d in nhom], fontsize=8)
        a.set_ylabel("Lưu lượng mất dịch vụ (%)")
        a.set_title(BO[bo].split(" (")[0], loc="left", fontsize=11, pad=36)
        a.grid(axis="y", color="#e4e3df", lw=0.8, zorder=0)
        a.legend(frameon=False, ncol=2, loc="lower left", bbox_to_anchor=(0, 1.0), borderaxespad=0.2, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(KQ / "hinh_6.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if sys.argv[1] == "chon-tau":
        print(json.dumps(chon_tau(), ensure_ascii=False, indent=1, default=str))
    else:
        r = phan_tich()
        print(json.dumps({k: {d: v[d]["tieu_chi"]["thang"] for d in DAI if d in v}
                          for k, v in r.items() if k not in ("tau", "co_che")}, ensure_ascii=False))
        print(json.dumps({b: c["dat"] for b, c in r["co_che"].items()}, ensure_ascii=False))
