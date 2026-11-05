"""
hinh_bao_cao_6_7.py — Hình KẾT QUẢ tổng hợp cho tai_lieu/BAO_CAO_THI_NGHIEM_6_7.md (mục 5–6).

    python -m scripts.hinh_bao_cao_6_7       # cần ket_qua_6.json, ket_qua_7.json (scripts.phan_tich_6 / phan_tich_7)

k1  ma trận ô tiêu chí: Lai6 so với Dijkstra phạt (TN 6: C, B, A; TN 7a: C-WH, A-WH) — màu = Δ mất dịch vụ
k2  Δ mất dịch vụ (C4) kèm CI 95% của từng ô
k3  TN 6 — đóng góp của dữ liệu và của đặc trưng: DM (TN 4) → DM6a (thêm dữ liệu) → DM6 (thêm đặc trưng)
k4  TN 7b — hai thành phần của mất dịch vụ theo khối lượng dữ liệu
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from scripts.phan_tich_5 import CAC_K, DAI, nap
from scripts.truc_quan_du_lieu import MAU, RA, kieu

GOC = Path(__file__).resolve().parents[1]
KQ = GOC / "results" / "bao_da_tuyen"
NHAN_K = {"k4_5": "4–5", "k10": "10", "k20": "20"}
NHAN_DAI = {"thap": "thấp", "giua": "giữa", "cao": "cao"}
O = [(kd, d) for kd in CAC_K for d in DAI]                      # 9 ô theo thứ tự cột


def o_tn6():
    r = json.loads((KQ / "ket_qua_6.json").read_text(encoding="utf-8"))
    ra = {}
    for bo, nhan in (("c6", "TN 6 — C ngày giông"), ("mua5", "TN 6 — B mưa lớn"), ("yagi5", "TN 6 — A Yagi")):
        hang = []
        for kd, d in O:
            tc = r[f"{bo}_{kd}"][d]["tieu_chi"]
            thang = bool(tc["C2"] and tc["C4"] and (kd == "k4_5" or tc["C5"]))
            hang.append({"C4": tc["C4_mat_dv"], "C2": tc["C2_J"], "thang": thang, "dat_C2": tc["C2"], "dat_C4": tc["C4"]})
        ra[nhan] = hang
    return ra


def o_tn7():
    r = json.loads((KQ / "ket_qua_7.json").read_text(encoding="utf-8"))["7a"]
    ra = {}
    for bo, nhan in (("c7wh", "TN 7a — C cực đoan WH"), ("y7wh", "TN 7a — A cực đoan WH")):
        ra[nhan] = [{"C4": r[f"{bo}_{kd}"][d]["C4"], "C2": r[f"{bo}_{kd}"][d]["C2"], "thang": r[f"{bo}_{kd}"][d]["thang"],
                     "dat_C2": r[f"{bo}_{kd}"][d]["dat_C2"], "dat_C4": r[f"{bo}_{kd}"][d]["dat_C4"]} for kd, d in O]
    return ra


def k1_ma_tran(plt, bang):
    from matplotlib.colors import TwoSlopeNorm
    hang = list(bang)
    M = np.array([[x["C4"][0] for x in bang[h]] for h in hang])
    fig, ax = plt.subplots(figsize=(11, 4.6))
    im = ax.imshow(M, cmap="RdBu", norm=TwoSlopeNorm(vcenter=0, vmin=-3.2, vmax=1.0), aspect="auto")
    for i, h in enumerate(hang):
        for j, x in enumerate(bang[h]):
            ky = "THẮNG" if x["thang"] else ("trượt C2" if x["dat_C4"] else ("trượt C4" if x["dat_C2"] else "trượt"))
            ax.text(j, i, f"{x['C4'][0]:+.1f}\n{ky}", ha="center", va="center", fontsize=8.5,
                    color="white" if x["C4"][0] < -2.0 else "#1f1e1b", fontweight="bold" if x["thang"] else "normal")
        ax.text(len(O) - 0.35, i, f"  {sum(x['thang'] for x in bang[h])}/9", va="center", fontsize=10, color="#1f1e1b")
    ax.set_yticks(range(len(hang)), hang, fontsize=9)
    ax.set_xticks(range(len(O)), [f"{NHAN_K[kd]} luồng\n{NHAN_DAI[d]}" for kd, d in O], fontsize=8.5)
    for x in (2.5, 5.5):
        ax.axvline(x, color="white", lw=3)
    ax.axhline(2.5, color="white", lw=3)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.08)
    cb.set_label("Lai6 − Dijkstra phạt, mất dịch vụ (điểm %)")
    cb.outline.set_visible(False)
    ax.set_title("Ô tiêu chí: số trên = Δ mất dịch vụ (Lai6 − Dijkstra phạt); THẮNG = đạt cả C2 (J) lẫn C4, CI 95%", loc="left",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(RA / "k1_ma_tran_o_tieu_chi.png", dpi=160)
    plt.close(fig)


def k2_rung(plt, bang):
    hang = list(bang)
    fig, ax = plt.subplots(1, len(hang), figsize=(13, 4.6), sharey=True)
    for i, h in enumerate(hang):
        for j, x in enumerate(bang[h]):
            m, (lo, hi) = x["C4"]
            mau = MAU[2] if x["thang"] else (MAU[0] if hi < 0 else MAU[7])
            ax[i].plot([lo, hi], [j, j], color=mau, lw=2)
            ax[i].plot(m, j, "o", color=mau, ms=6, mec="white", mew=1.2)
        ax[i].axvline(0, color="#8a8984", lw=1)
        ax[i].set_title(h.replace(" — ", "\n"), loc="left", fontsize=9.5)
        ax[i].grid(axis="x", color="#e4e3df", lw=0.8)
        ax[i].set_xlabel("Δ mất DV (điểm %)", fontsize=9)
        for y in (2.5, 5.5):
            ax[i].axhline(y, color="#e4e3df", lw=0.8)
    ax[0].set_yticks(range(len(O)), [f"{NHAN_K[kd]} luồng · {NHAN_DAI[d]}" for kd, d in O], fontsize=8.5)
    ax[0].invert_yaxis()
    fig.suptitle("C4 — Lai6 − Dijkstra phạt và CI 95% theo 13 cửa sổ (xanh lá: ô thắng; xanh dương: chỉ đạt C4; xám: "
                 "không đạt C4)", x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(RA / "k2_c4_khoang_tin_cay.png", dpi=160)
    plt.close(fig)


def k3_dong_gop_tn6(plt):
    r = json.loads((KQ / "ket_qua_6.json").read_text(encoding="utf-8"))
    pp = [("dphat", "Dijkstra phạt"), ("dm", "DM (TN 4, dữ liệu bão)"), ("dm6a", "DM6a (+ dữ liệu ngày giông)"),
          ("dm6", "DM6 (+ dữ liệu + 3 đặc trưng giông quanh UAV)")]
    bo = [("c6", "C — ngày giông"), ("mua5", "B — mưa lớn"), ("yagi5", "A — Yagi")]
    fig, ax = plt.subplots(1, 3, figsize=(13, 4.4))
    w = 0.2
    for i, (b, ten) in enumerate(bo):
        for k, (m, nhan) in enumerate(pp):
            v = [r[f"{b}_k4_5"][d]["phuong_phap"][m]["mat_dv_%"] for d in DAI]
            ax[i].bar(np.arange(3) + (k - 1.5) * w, v, w * 0.92, color=MAU[k], label=nhan, zorder=2)
        ax[i].set_xticks(range(3), [f"dải {NHAN_DAI[d]}" for d in DAI])
        ax[i].set_title(ten, loc="left", fontsize=10.5)
        ax[i].grid(axis="y", color="#e4e3df", lw=0.8, zorder=0)
        lo = min(r[f"{b}_k4_5"][d]["phuong_phap"][m]["mat_dv_%"] for d in DAI for m, _ in pp)
        ax[i].set_ylim(lo * 0.9, None)
    ax[0].set_ylabel("Lưu lượng mất dịch vụ (%), 4–5 luồng")
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, ncol=4, frameon=False, loc="upper left", bbox_to_anchor=(0.01, 0.99), fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(RA / "k3_tn6_dong_gop_du_lieu_dac_trung.png", dpi=160)
    plt.close(fig)


def k4_thanh_phan_7b(plt):
    mh = [("dm7_25", 25), ("dm7_50", 50), ("dm6", 100), ("dm7_200", 200)]
    bo = [("c6", "C — ngày giông"), ("yagi5", "A — Yagi"), ("c7wh", "C cực đoan WH"), ("y7wh", "A cực đoan WH")]
    fig, ax = plt.subplots(1, 4, figsize=(14, 4.2))
    so = {}
    for i, (b, ten) in enumerate(bo):
        rs = nap(b, "k4_5")
        tb = lambda m, k: float(np.mean([r["ens"][m][k] for r in rs]))
        so[b] = {m: (tb(m, "khong_tuyen_song_%"), tb(m, "du_phong_qua_tai_%")) for m, _ in mh + [("dphat", 0)]}
        x = [t for _, t in mh]
        for k, (khoa, nhan, mau) in enumerate((("khong_tuyen_song_%", "(a) mọi tuyến cùng đứt", MAU[1]),
                                                ("du_phong_qua_tai_%", "(b) dự phòng quá tải", MAU[0]))):
            ax[i].plot(x, [so[b][m][k] for m, _ in mh], "-o", color=mau, lw=2, ms=6, mec="white", mew=1.4,
                       label=f"DM — {nhan}", zorder=3)
            ax[i].axhline(so[b]["dphat"][k], color=mau, lw=1.2, ls="--", label=f"Dijkstra phạt — {nhan}", zorder=2)
        ax[i].set_xscale("log", base=2)
        ax[i].set_xticks(x, ["25%", "50%", "100%", "200%"])
        ax[i].set_title(ten, loc="left", fontsize=10.5)
        ax[i].grid(color="#e4e3df", lw=0.8, zorder=0)
        ax[i].set_xlabel("Dữ liệu huấn luyện (so với DM6)", fontsize=9)
    ax[0].set_ylabel("Tỉ lệ lưu lượng (%)")
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, ncol=4, frameon=False, loc="upper left", bbox_to_anchor=(0.01, 0.99), fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(RA / "k4_tn7b_thanh_phan_mat_dich_vu.png", dpi=160)
    plt.close(fig)
    return so


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    RA.mkdir(parents=True, exist_ok=True)
    plt = kieu()
    bang = {**o_tn6(), **o_tn7()}
    so_thang = {h: sum(x["thang"] for x in v) for h, v in bang.items()}
    assert list(so_thang.values()) == [6, 2, 8, 4, 7], so_thang       # khớp BAO_CAO_6 / BAO_CAO_7
    k1_ma_tran(plt, bang)
    k2_rung(plt, bang)
    k3_dong_gop_tn6(plt)
    so = k4_thanh_phan_7b(plt)
    print(json.dumps({"so_o_thang": so_thang, "thanh_phan_7b": so}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
