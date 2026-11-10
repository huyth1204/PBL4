"""
phan_tich_10.py — Thí nghiệm 10 (results/bao_da_tuyen/TIEU_CHI_10.md): tóm tắt 4 bước chọn trên val và kiểm định
H10.1–H10.3 trên bộ kiểm tra.

    python -m scripts.phan_tich_10        # → results/bao_da_tuyen/{ket_qua_10.json, KET_QUA_10.md, hinh_10.png}

Họ DM6r = khoá r6, r6_s1, r6_s2 (cấu hình DM6 học lại trên cùng tập học); họ DM10 = d10, d10_s1, d10_s2. Lai(·) = DM nếu ρ̂ < 1,1, ngược lại
DM + TKCB. So sánh hai họ: trung bình 3 seed theo bản ghi, CI 95% bootstrap theo cửa sổ (10 000 lần). Ô tiêu chí như
TN8 (mô hình seed 0), p một phía bootstrap, Holm theo họ 36 ô mỗi mô hình.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from scripts.kiem_dinh_boi_9 import ALPHA, bh, holm, p_mot_phia
from scripts.phan_tich_5 import CAC_K, DAI, TEN_DAI, nap
from scripts.phan_tich_dt import bootstrap

GOC = Path(__file__).resolve().parents[1]
KQ = GOC / "results" / "bao_da_tuyen"
TAU, KHOA_T10 = 1.1, "online_tkcb10_6"
HO = {"DM6r": ("r6", "r6_s1", "r6_s2"), "DM10": ("d10", "d10_s1", "d10_s2")}
TEST_45 = ("c6", "yagi5", "c6_b", "yagi5_b", "ngay9", "nam9", "bao9")
CO_THAY = ("ngay9", "nam9", "bao9")
TEST_LON = ("c6", "yagi5", "c6_b", "yagi5_b")
TEN_BO = {"c6": "TG A · C ngày giông", "yagi5": "TG A · Yagi", "c6_b": "TG B · C ngày giông", "yagi5_b": "TG B · Yagi",
          "ngay9": "ngày giông mới", "nam9": "ngày xấu 2017–21", "bao9": "đợt mạnh 2017–21"}


def lai(r, m, k="mat_dv_%"):
    return r["ens"][m if r["rho_hat"] < TAU else m + "_tkcb"][k]


def co(recs, m):
    return bool(recs) and all(m in r["ens"] and m + "_tkcb" in r["ens"] for r in recs)


def _ci(v, nd=2):
    return f"{v[0]:+.{nd}f} [{v[1][0]:+.{nd}f}; {v[1][1]:+.{nd}f}]"


def tb_ho(recs, h, f):
    return np.array([np.mean([f(r, m) for m in HO[h]]) for r in recs])


def o_cua(rs, m, kd, rng):
    kb = np.array([r["kb"] for r in rs])
    v = lambda k, pp=None: np.array([lai(r, m, k) if pp is None else r["ens"][pp][k] for r in rs])
    dJ, dL = v("J") - v("J", "dphat"), v("mat_dv_%") - v("mat_dv_%", "dphat")
    t10 = [r["tg_ms"][KHOA_T10] for r in rs if KHOA_T10 in r["tg_ms"]]
    tl = float(np.median([r["tg_ms"].get(m if r["rho_hat"] < TAU else m + "_tkcb", np.nan) for r in rs]))
    c5 = kd == "k4_5" or (bool(t10) and tl <= float(np.median(t10)) and tl <= 1000)
    pJ, pL = p_mot_phia(dJ, kb, rng)[0], p_mot_phia(dL, kb, rng)[0]
    return {"so": len(rs), "dphat_%": float(v("mat_dv_%", "dphat").mean()), "lai_%": float(v("mat_dv_%").mean()),
            "C2": bootstrap(dJ, kb, rng), "C4": bootstrap(dL, kb, rng), "p": max(pJ, pL), "dat_C5": bool(c5),
            "tg_lai_ms": tl, "tg_tkcb10_ms": float(np.median(t10)) if t10 else None}


def phan_tich():
    rng = np.random.default_rng(0)
    chon_tep = KQ / "tn10" / "chon.json"
    ra = {"chon": json.loads(chon_tep.read_text(encoding="utf-8")) if chon_tep.exists() else {}, "so_sanh_ho": {},
          "o": {}, "tom_tat_o": {}, "gia_thuyet": {}}
    du = {(b, "k4_5"): nap(b, "k4_5") for b in TEST_45}
    du.update({(b, k): nap(b, k) for b in TEST_LON for k in ("k10", "k20")})
    # ---- so sánh hai họ (4–5 luồng): DM, Lai; và G trên bộ có thầy
    for b in TEST_45:
        recs = du[(b, "k4_5")]
        if not all(co(recs, m) for h in HO for m in HO[h]):
            continue
        kb = np.array([r["kb"] for r in recs])
        dm = {h: tb_ho(recs, h, lambda r, m: r["ens"][m]["mat_dv_%"]) for h in HO}
        la = {h: tb_ho(recs, h, lai) for h in HO}
        x = {"dm": {h: float(v.mean()) for h, v in dm.items()}, "lai": {h: float(v.mean()) for h, v in la.items()},
             "dphat": float(np.mean([r["ens"]["dphat"]["mat_dv_%"] for r in recs])),
             "dm10_tru_dm6": bootstrap(dm["DM10"] - dm["DM6r"], kb, rng),
             "lai10_tru_lai6": bootstrap(la["DM10"] - la["DM6r"], kb, rng),
             "lai10_tru_dphat": bootstrap(la["DM10"] - np.array([r["ens"]["dphat"]["mat_dv_%"] for r in recs]), kb, rng),
             "lai6_tru_dphat": bootstrap(la["DM6r"] - np.array([r["ens"]["dphat"]["mat_dv_%"] for r in recs]), kb, rng)}
        if b in CO_THAY and all("thay" in r["ens"] for r in recs):
            th = np.array([r["ens"]["thay"]["mat_dv_%"] for r in recs])
            x["G"] = {h: float((dm[h] - th).mean()) for h in HO}
            x["phan_lap_%"] = float((1 - x["G"]["DM10"] / x["G"]["DM6r"]) * 100) if x["G"]["DM6r"] > 0 else None
        ra["so_sanh_ho"][b] = x
    # ---- ô tiêu chí, mô hình seed 0 của mỗi họ
    for h, m in (("DM6r", "r6"), ("DM10", "d10")):
        o = {}
        for b in TEST_LON:
            for kd in CAC_K:
                recs = du.get((b, kd))
                if not recs or not co(recs, m):
                    continue
                for d, (lo, hi) in DAI.items():
                    rs = [r for r in recs if lo <= r["rho_hat"] < hi]
                    if rs:
                        o[f"{b}|{kd}|{d}"] = o_cua(rs, m, kd, rng)
        if not o:
            continue
        k = list(o)
        p, c5 = np.array([o[x]["p"] for x in k]), np.array([o[x]["dat_C5"] for x in k])
        for i, (th, hm, b_) in enumerate(zip((p <= ALPHA) & c5, holm(p) & c5, bh(p) & c5)):
            o[k[i]].update({"thang": bool(th), "thang_holm": bool(hm), "thang_bh": bool(b_), "te_hon": o[k[i]]["C4"][1][0] > 0})
        ra["o"][h] = o
        ra["tom_tat_o"][h] = {"so_o": len(k), **{t: int(sum(o[x][t] for x in k)) for t in ("thang", "thang_holm", "thang_bh", "te_hon")},
                              "theo_bo": {b: {t: int(sum(o[x][t] for x in k if x.startswith(b + "|"))) for t in ("thang", "thang_holm")}
                                          for b in TEST_LON},
                              "tai_cao": int(sum(o[x]["thang"] for x in k if x.endswith("|cao")))}
    # ---- giả thuyết
    s = ra["so_sanh_ho"]
    if all(b in s and "G" in s[b] for b in CO_THAY):
        n = sum(s[b]["dm10_tru_dm6"][1][1] < 0 for b in CO_THAY)
        ra["gia_thuyet"]["H10.1"] = {"so_bo_giam": int(n), "dat": bool(n >= 2)}
    if all(b in s for b in TEST_45):
        n = sum(s[b]["lai10_tru_lai6"][1][1] < 0 for b in TEST_45)
        te = sum(s[b]["lai10_tru_lai6"][1][0] > 0 for b in TEST_45)
        ra["gia_thuyet"]["H10.2"] = {"so_bo_giam": int(n), "so_bo_te_hon": int(te), "dat": bool(n >= 4 and te == 0)}
    t = ra["tom_tat_o"]
    if "DM6r" in t and "DM10" in t and t["DM10"]["so_o"] == 36 and all(b in s for b in ("nam9", "bao9")):
        nam_bao = all(s[b]["lai10_tru_dphat"][1][1] < 0 for b in ("nam9", "bao9"))
        ra["gia_thuyet"]["H10.3"] = {"holm_DM10": t["DM10"]["thang_holm"], "holm_DM6r": t["DM6r"]["thang_holm"],
                                     "te_hon_DM10": t["DM10"]["te_hon"], "nam9_bao9": bool(nam_bao),
                                     "dat": bool(t["DM10"]["thang_holm"] >= t["DM6r"]["thang_holm"] and t["DM10"]["te_hon"] == 0
                                                 and nam_bao)}
    (KQ / "ket_qua_10.json").write_text(json.dumps(ra, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    viet_md(ra)
    ve_hinh(ra)
    return ra


def viet_md(ra):
    c = ra["chon"]
    L = ["# Thí nghiệm 10 — kết quả (sinh tự động bởi `phan_tich_10.py`)", "",
         "Mất dịch vụ tính bằng %, hiệu bằng điểm %. Họ = trung bình 3 seed huấn luyện; CI 95% bootstrap theo cửa sổ.", "",
         "## 1. Bốn bước chọn bằng kiểm thử chéo 4 phần (tập học, chia theo cửa sổ)", ""]
    if "buoc1" in c:
        g = c["buoc1"]["goc"]
        L += ["**Bước 1 — năng lực** (kiểm thử chéo 4 phần trên tập học A, DM, S = 8):", "",
              "| Cấu hình | Mất DV | Hiệu so với e20h64 | Theo phần | Epoch tốt nhất / số epoch |", "|---|---:|---|---|---|",
              f"| e20h64 (DM6) | {g['mat_dv']:.2f} | – | {', '.join(f'{v:.2f}' for v in g['theo_phan'])} | "
              f"{', '.join(f'{e}/{n}' for e, n in g['epoch_tot'])} |"]
        for ten, x in c["buoc1"]["so_sanh"].items():
            m = x["moi"]
            L.append(f"| {ten} | {m['mat_dv']:.2f} | {_ci(x['hieu'])} | {', '.join(f'{v:.2f}' for v in m['theo_phan'])} | "
                     f"{', '.join(f'{e}/{n}' for e, n in m['epoch_tot'])} |")
        L += ["", f"→ chọn **{c['buoc1']['ho']}**.", ""]
    if "buoc2" in c:
        L += ["**Bước 2 — số chuỗi S:** " + ", ".join(f"S = {s}: {v:.2f}" for s, v in c["buoc2"]["bang"].items())
              + f" → chọn **S = {c['buoc2']['S']}**.", ""]
    if "buoc3a" in c:
        x = c["buoc3a"]
        L += [f"**Bước 3 — thầy:** mất DV của thầy trên val10: 32 thế giới nhãn {x['thay32']:.2f}, 64 thế giới {x['thay64']:.2f}, "
              f"hiệu {_ci(x['hieu'])}; cùng tình huống: {'có' if x['cung_tinh_huong'] else 'KHÔNG'} → "
              f"{'gắn lại nhãn bằng thầy 64' if x['gan_nhan_lai'] else 'giữ thầy 32'}.", ""]
    if "buoc3b" in c and "so_sanh" in c["buoc3b"]:
        x = c["buoc3b"]["so_sanh"]
        L += [f"  Học trên nhãn thầy 64: {x['moi']['mat_dv']:.2f} so với {x['cu']['mat_dv']:.2f}, hiệu {_ci(x['hieu'])} → "
              f"**{c['buoc3b']['ho']}**.", ""]
    if "buoc4" in c:
        x, y = c["buoc4"]["so_sanh_AB"], c["buoc4"]["so_sanh_A"]
        L += [f"**Bước 4 — dữ liệu A + B** (kiểm thử chéo): A + B {_ci(x['hieu'])}, riêng A {_ci(y['hieu'])} → "
              f"**{c['buoc4']['ho']}**.", ""]
    if "chot" in c:
        L += [f"**DM10 = {c['chot']['ho']}, S = {c['chot']['S']}**" +
              (" (trùng DM6r — không bước nào qua)" if c["chot"]["trung_DM6r"] else "") + ".", ""]
    L += ["## 2. DM10 so với DM6r trên bộ kiểm tra (4–5 luồng)", "",
          "| Bộ | Dijkstra phạt | Lai6r | Lai10 | Lai10 − Lai6r | DM10 − DM6r | G6r → G10 (lấp) |", "|---|---:|---:|---:|---|---|---|"]
    for b, x in ra["so_sanh_ho"].items():
        g = f"{x['G']['DM6r']:.2f} → {x['G']['DM10']:.2f} ({x['phan_lap_%']:.0f}%)" if "G" in x and x.get("phan_lap_%") is not None else "–"
        L.append(f"| {TEN_BO[b]} | {x['dphat']:.2f} | {x['lai']['DM6r']:.2f} | {x['lai']['DM10']:.2f} | {_ci(x['lai10_tru_lai6'])} | "
                 f"{_ci(x['dm10_tru_dm6'])} | {g} |")
    L += ["", "## 3. Ô tiêu chí so với Dijkstra phạt (mô hình seed 0; 4 bộ × 3 quy mô × 3 dải)", "",
          "| Mô hình | Thắng | Holm | BH | Tệ hơn | Tải cao thắng | " + " | ".join(TEN_BO[b] for b in TEST_LON) + " |",
          "|---|---:|---:|---:|---:|---:|" + "---|" * len(TEST_LON)]
    for h, t in ra["tom_tat_o"].items():
        L.append(f"| {h} | {t['thang']}/{t['so_o']} | {t['thang_holm']} | {t['thang_bh']} | {t['te_hon']} | {t['tai_cao']} | "
                 + " | ".join(f"{t['theo_bo'][b]['thang']} / {t['theo_bo'][b]['thang_holm']}" for b in TEST_LON) + " |")
    L += ["", "Cột từng bộ: thắng chưa hiệu chỉnh / Holm, trên 9.", "",
          "| Mô hình | Bộ | Quy mô | Dải | Dijkstra → Lai | C2 ΔJ | C4 Δ mất DV | p | C5 ms (Lai / TKCB10) | Thắng / Holm |",
          "|---|---|---|---|---|---|---|---:|---|---|"]
    for h, o in ra["o"].items():
        for k, x in o.items():
            b, kd, d = k.split("|")
            c5 = f"{x['tg_lai_ms']:.0f} / {x['tg_tkcb10_ms']:.0f}" if x["tg_tkcb10_ms"] else "–"
            L.append(f"| {h} | {b} | {CAC_K[kd]} | {TEN_DAI[d]} | {x['dphat_%']:.1f} → {x['lai_%']:.1f} | {_ci(x['C2'], 3)} | "
                     f"{_ci(x['C4'])} | {x['p']:.4f} | {c5} | {'có' if x['thang'] else ('TỆ HƠN' if x['te_hon'] else 'không')} / "
                     f"{'có' if x['thang_holm'] else 'không'} |")
    L += ["", "## 4. Giả thuyết", ""]
    for k, x in ra["gia_thuyet"].items():
        L.append(f"- **{k}: {'ĐẠT' if x['dat'] else 'không đạt'}** — " + ", ".join(f"{a} = {b}" for a, b in x.items() if a != "dat"))
    (KQ / "KET_QUA_10.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def ve_hinh(ra):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
                         "axes.spines.top": False, "axes.spines.right": False, "font.size": 10})
    mau = {"DM6r": "#2a78d6", "DM10": "#eb6834", "dm": "#1baf7a"}
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))
    # (a) bước 1–2 (kiểm thử chéo): mất DV của DM theo cấu hình / số chuỗi
    c, nh, v = ra["chon"], [], []
    if "buoc1" in c:
        nh.append("e20h64\n(DM6)"), v.append(c["buoc1"]["goc"]["mat_dv"])
        for ten, y in c["buoc1"]["so_sanh"].items():
            nh.append(ten), v.append(y["moi"]["mat_dv"])
    if "buoc2" in c:
        for s_, y in c["buoc2"]["bang"].items():
            nh.append(f"{c['buoc2']['ho']}\nS = {s_}"), v.append(y)
    if v:
        ax[0].bar(range(len(v)), v, 0.7, color=["#a3a29d"] + [mau["DM10"]] * (len(v) - 1))
        ax[0].set_xticks(range(len(v)), nh, fontsize=7.5)
        ax[0].set_ylim(min(v) - 1.5, max(v) + 0.8)
    ax[0].set_ylabel("Mất DV của DM, kiểm thử chéo (%)")
    ax[0].set_title("(a) Bước 1–2: kiểm thử chéo 4 phần", loc="left", fontsize=10.5)
    ax[0].grid(axis="y", color="#e4e3df", lw=0.8)
    ax[0].set_axisbelow(True)
    # (b) DM10 − DM6 trên từng bộ kiểm tra
    bs = list(ra["so_sanh_ho"])
    for i, (k, ten, m) in enumerate((("lai10_tru_lai6", "Lai10 − Lai6r", mau["DM10"]), ("dm10_tru_dm6", "DM10 − DM6r", mau["dm"]))):
        for j, b in enumerate(bs):
            y = ra["so_sanh_ho"][b][k]
            ax[1].errorbar(j + (i - 0.5) * 0.3, y[0], yerr=[[y[0] - y[1][0]], [y[1][1] - y[0]]], fmt="o", ms=5, color=m,
                           capsize=3, label=ten if j == 0 else None)
    ax[1].axhline(0, color="#1a1a19", lw=0.8)
    ax[1].set_xticks(range(len(bs)), [TEN_BO[b] for b in bs], fontsize=7.5, rotation=20)
    ax[1].set_ylabel("Hiệu mất DV (điểm %), 4–5 luồng")
    ax[1].legend(frameon=False, fontsize=8.5)
    ax[1].set_title("(b) DM10 so với DM6r trên bộ kiểm tra (CI 95%)", loc="left", fontsize=10.5)
    ax[1].grid(axis="y", color="#e4e3df", lw=0.8)
    ax[1].set_axisbelow(True)
    # (c) số ô thắng theo Holm
    t = ra["tom_tat_o"]
    for i, h in enumerate(("DM6r", "DM10")):
        if h in t:
            ax[2].bar(np.arange(len(TEST_LON)) + (i - 0.5) * 0.36, [t[h]["theo_bo"][b]["thang_holm"] for b in TEST_LON], 0.34,
                      color=mau[h], label=f"Lai({h})")
    ax[2].set_xticks(range(len(TEST_LON)), [TEN_BO[b] for b in TEST_LON], fontsize=8, rotation=15)
    ax[2].set_ylim(0, 9.5)
    ax[2].set_ylabel("Số ô thắng Dijkstra phạt theo Holm (/9)")
    ax[2].legend(frameon=False, fontsize=8.5)
    ax[2].set_title("(c) Ô tiêu chí sau hiệu chỉnh Holm", loc="left", fontsize=10.5)
    ax[2].grid(axis="y", color="#e4e3df", lw=0.8)
    ax[2].set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(KQ / "hinh_10.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    r = phan_tich()
    print(json.dumps(r["gia_thuyet"], ensure_ascii=False, indent=1))
