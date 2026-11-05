"""
phan_tich_dt.py — Thống kê, tiêu chí C2 / C4 / C5 và hình cho thí nghiệm 4 (results/bao_da_tuyen/TIEU_CHI_4.md).

    python -m scripts.phan_tich_dt          # → results/bao_da_tuyen/ket_qua.json, KET_QUA.md, hinh_ket_qua.png

Số chính: 32 thế giới CHẤM (rẽ nhánh tại o, seed độc lập với nhãn). CI 95%: bootstrap ghép cặp theo 13 cửa sổ Yagi,
10 000 lần.
"""

from __future__ import annotations

import glob
import json
import pickle
import sys
from pathlib import Path

import numpy as np

from src.nhieu_luong.bai_toan import MUC_TAI
from src.nhieu_luong.tao_du_lieu_rr import THOI_TIET, cua_so_yagi
from src.ngau_nhien.thoi_tiet import ThoiTiet

GOC = Path(__file__).resolve().parents[1]
DL = GOC / "data" / "bao_da_tuyen"
KQ = GOC / "results" / "bao_da_tuyen"
CAC_K = {"k4_5": "4–5 luồng", "k10": "10 luồng", "k20": "20 luồng", "k40": "40 luồng"}
NGOAI_TIEU_CHI = {"k40"}                 # quy mô thêm sau khi chốt: chỉ báo, không tính vào kết luận chính
PP = ("yen", "suurballe", "dphat", "online_tkcb", "softmax", "dm", "dm_tkcb", "thay")
TEN_PP = {"yen": "Yen (Dijkstra yếu)", "suurballe": "Suurballe", "dphat": "Dijkstra phạt (đối thủ chính)",
          "online_tkcb": "TKCB trên J bảng", "softmax": "Softmax", "dm": "DM",
          "dm_tkcb": "DM + TKCB trên J bảng (thêm, ngoài tiêu chí)", "thay": "Thầy (dùng tương lai, mốc)"}
TEN_MUC = {"nhe": "nhẹ", "vua": "vừa", "nang": "nặng"}
B = 10_000


def bootstrap(v, kb, rng):
    u = np.unique(kb)
    tong = np.array([v[kb == k].sum() for k in u])
    dem = np.array([(kb == k).sum() for k in u])
    j = rng.integers(0, len(u), (B, len(u)))
    m = tong[j].sum(1) / dem[j].sum(1)
    return [float(v.mean()), [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]]


def cuong_do_gio():
    """Mức bão của từng giờ Yagi: % điểm có mưa > 11 mm/h hoặc gió giật > 15 m/s (luật chọn cửa sổ của bao_manh)."""
    tt = ThoiTiet(str(THOI_TIET), kiem_tra=False)
    return tt, ((tt.precip > 11) | (tt.gust > 15)).mean(1) * 100


def phan_tich():
    rng = np.random.default_rng(0)
    tt, cd = cuong_do_gio()
    h0s = cua_so_yagi(tt)
    ra = {}
    for kd, ten_k in CAC_K.items():
        recs = [r for f in sorted(glob.glob(str(DL / f"test_{kd}" / "kb_*.pkl"))) for r in pickle.load(open(f, "rb"))]
        if not recs:
            continue
        pp = [p for p in PP if p in recs[0]["ens"]]
        ra[kd] = {"ten": ten_k, "so_ban_ghi": len(recs)}
        for muc in MUC_TAI:
            rs = [r for r in recs if r["muc"] == muc]
            kb = np.array([r["kb"] for r in rs])
            v = lambda p, k, tw="ens": np.array([r[tw][p][k] for r in rs])
            o = {"so_tinh_huong": len(rs), "phuong_phap": {}}
            for p in pp:
                o["phuong_phap"][p] = {k: float(v(p, k).mean()) for k in rs[0]["ens"][p]}
                o["phuong_phap"][p]["qs_mat_dv_%"] = float(v(p, "mat_dv_%", "qs").mean())
                t = [r["tg_ms"][p] for r in rs if p in r["tg_ms"]]
                if t:
                    o["phuong_phap"][p]["tg_trung_vi_ms"] = float(np.median(t))
            t10 = [r["tg_ms"]["online_tkcb10"] for r in rs if "online_tkcb10" in r["tg_ms"]]
            o["tg_trung_vi_tkcb10_ms"] = float(np.median(t10)) if t10 else None
            if "dm" in pp:
                C2 = bootstrap(v("dm", "J") - v("dphat", "J"), kb, rng)
                C4 = bootstrap(v("dm", "mat_dv_%") - v("dphat", "mat_dv_%"), kb, rng)
                tdm = o["phuong_phap"]["dm"]["tg_trung_vi_ms"]
                C5_tg = bool(t10) and tdm <= o["tg_trung_vi_tkcb10_ms"] and tdm <= 1000
                o["tieu_chi"] = {
                    "C2_J_dm_tru_dphat": C2, "C2": C2[1][1] < 0,
                    "C4_mat_dv_dm_tru_dphat_diem_%": C4, "C4": C4[1][1] < 0,
                    "C4a_khong_tuyen_song": bootstrap(v("dm", "khong_tuyen_song_%") - v("dphat", "khong_tuyen_song_%"), kb, rng),
                    "C4b_du_phong_qua_tai": bootstrap(v("dm", "du_phong_qua_tai_%") - v("dphat", "du_phong_qua_tai_%"), kb, rng),
                    "C5_thoi_gian": C5_tg,
                    "phu_dm_tru_softmax_J": bootstrap(v("dm", "J") - v("softmax", "J"), kb, rng),
                    "phu_dm_tru_suurballe_khong_tuyen_song": bootstrap(
                        v("dm", "khong_tuyen_song_%") - v("suurballe", "khong_tuyen_song_%"), kb, rng),
                    "phu_qs_mat_dv_dm_tru_dphat": bootstrap(v("dm", "mat_dv_%", "qs") - v("dphat", "mat_dv_%", "qs"), kb, rng)}
                if "thay" in pp:
                    o["tieu_chi"]["phu_dm_tru_thay_mat_dv"] = bootstrap(v("dm", "mat_dv_%") - v("thay", "mat_dv_%"), kb, rng)
                if "dm_tkcb" in pp:
                    o["tieu_chi"]["them_dm_tkcb_tru_dphat_J"] = bootstrap(v("dm_tkcb", "J") - v("dphat", "J"), kb, rng)
                    o["tieu_chi"]["them_dm_tkcb_tru_dphat_mat_dv"] = bootstrap(
                        v("dm_tkcb", "mat_dv_%") - v("dphat", "mat_dv_%"), kb, rng)
                o["tieu_chi"]["thang"] = bool(o["tieu_chi"]["C2"] and o["tieu_chi"]["C4"]
                                              and (kd == "k4_5" or o["tieu_chi"]["C5_thoi_gian"]))
                # phân tầng theo cường độ bão của giờ quyết định
                gio = np.array([h0s[r["kb"]] + int(r["t"] * 20 / 3600) for r in rs])
                c = cd[gio]
                bien = np.percentile(c, [100 / 3, 200 / 3])
                tang = np.digitize(c, bien)
                o["theo_cuong_do"] = {f"tang_{j}": {"cuong_do_%": [float(c[tang == j].min()), float(c[tang == j].max())],
                                                    "so": int((tang == j).sum()),
                                                    "mat_dv_dm": float(v("dm", "mat_dv_%")[tang == j].mean()),
                                                    "mat_dv_dphat": float(v("dphat", "mat_dv_%")[tang == j].mean())}
                                      for j in range(3) if (tang == j).any()}
            ra[kd][muc] = o
    KQ.mkdir(parents=True, exist_ok=True)
    json.dump(ra, open(KQ / "ket_qua.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    viet_md(ra)
    ve_hinh(ra)
    return ra


def _ci(v, nd=2):
    return f"{v[0]:+.{nd}f} [{v[1][0]:+.{nd}f}; {v[1][1]:+.{nd}f}]"


def viet_md(ra):
    dd = lambda b: "**đạt**" if b else "không đạt"
    L = ["# Thí nghiệm 4 — kết quả trên Yagi (sinh tự động bởi `phan_tich_dt.py`)", "",
         "Số chính trên 32 thế giới CHẤM (rẽ nhánh tại o, seed độc lập với bộ nhãn). CI 95%: bootstrap ghép cặp theo "
         "13 cửa sổ. Mất dịch vụ = % lưu lượng không còn tuyến sống (a) hoặc tuyến đang dùng quá tải sau chuyển mạch (b).", ""]
    for kd, rk in ra.items():
        for muc in MUC_TAI:
            if muc not in rk:
                continue
            o = rk[muc]
            L += [f"## {rk['ten']}, tải {TEN_MUC[muc]} ({o['so_tinh_huong']} tình huống)", "",
                  "| Phương pháp | J | Mất DV % | (a) hết tuyến % | (b) dự phòng quá tải % | Trễ tuyến chính ms | "
                  "Mất DV % (thế giới quan sát) | Thời gian ms |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
            for p, x in o["phuong_phap"].items():
                tg_ = f"{x['tg_trung_vi_ms']:.0f}" if "tg_trung_vi_ms" in x else "–"
                L.append(f"| {TEN_PP[p]} | {x['J']:.3f} | {x['mat_dv_%']:.2f} | {x['khong_tuyen_song_%']:.2f} | "
                         f"{x['du_phong_qua_tai_%']:.2f} | {x['tre_chinh_ms']:.1f} | {x['qs_mat_dv_%']:.2f} | {tg_} |")
            if o.get("tg_trung_vi_tkcb10_ms"):
                L.append(f"| TKCB 10 lần trên J bảng (chỉ đo thời gian) | | | | | | | {o['tg_trung_vi_tkcb10_ms']:.0f} |")
            tc = o.get("tieu_chi")
            if tc:
                L += ["", f"- C2 (J: DM − Dijkstra phạt): {_ci(tc['C2_J_dm_tru_dphat'], 3)} → {dd(tc['C2'])}",
                      f"- C4 (mất DV: DM − Dijkstra phạt, điểm %): {_ci(tc['C4_mat_dv_dm_tru_dphat_diem_%'])} → {dd(tc['C4'])}"
                      f" — phần (a) {_ci(tc['C4a_khong_tuyen_song'])}, phần (b) {_ci(tc['C4b_du_phong_qua_tai'])}"]
                if kd != "k4_5":
                    L.append(f"- C5 (thời gian DM ≤ TKCB 10 lần và ≤ 1 s): {dd(tc['C5_thoi_gian'])}")
                ngoai = " (quy mô thêm sau khi chốt — chỉ báo, không thuộc kết luận chính)" if kd in NGOAI_TIEU_CHI else ""
                L += [f"- **DM thắng ô này: {'có' if tc['thang'] else 'không'}**{ngoai}",
                      f"- Phụ: J DM − softmax {_ci(tc['phu_dm_tru_softmax_J'], 3)}; hết tuyến DM − Suurballe "
                      f"{_ci(tc['phu_dm_tru_suurballe_khong_tuyen_song'])}; "
                      + (f"mất DV DM − thầy {_ci(tc['phu_dm_tru_thay_mat_dv'])}; " if "phu_dm_tru_thay_mat_dv" in tc else "")
                      + f"thế giới quan sát DM − Dijkstra phạt {_ci(tc['phu_qs_mat_dv_dm_tru_dphat'])}"]
                if "them_dm_tkcb_tru_dphat_J" in tc:
                    L.append(f"- Thêm (ngoài tiêu chí): DM + TKCB − Dijkstra phạt: J {_ci(tc['them_dm_tkcb_tru_dphat_J'], 3)}, "
                             f"mất DV {_ci(tc['them_dm_tkcb_tru_dphat_mat_dv'])}")
                L += [
                      "- Theo cường độ bão của giờ quyết định: " + "; ".join(
                          f"{x['cuong_do_%'][0]:.0f}–{x['cuong_do_%'][1]:.0f}% điểm bão ({x['so']}): DM {x['mat_dv_dm']:.1f}% / "
                          f"Dijkstra phạt {x['mat_dv_dphat']:.1f}%" for x in o["theo_cuong_do"].values())]
            L.append("")
    (KQ / "KET_QUA.md").write_text("\n".join(L), encoding="utf-8")


MAU = {"dphat": "#2a78d6", "dm": "#eb6834", "thay": "#1baf7a"}        # 3 ô đầu của bảng màu chuẩn


def ve_hinh(ra):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
                         "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
                         "axes.edgecolor": "#8a8984", "axes.labelcolor": "#3b3a36", "xtick.color": "#52514e",
                         "ytick.color": "#52514e"})
    ks = [k for k in CAC_K if k in ra and "dm" in ra[k]["nhe"]["phuong_phap"]]
    if not ks:
        return
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw={"width_ratios": [1.6, 1]})
    luoi = dict(color="#e4e3df", lw=0.8, zorder=0)
    nhom = [(k, m) for k in ks for m in MUC_TAI]
    x = np.arange(len(nhom))
    bw = 0.26
    for j, p in enumerate(("dphat", "dm", "thay")):
        v = [ra[k][m]["phuong_phap"].get(p, {}).get("mat_dv_%", np.nan) for k, m in nhom]      # 40 luồng: không có thầy
        ax[0].bar(x + (j - 1) * (bw + 0.02), v, bw, color=MAU[p], label=TEN_PP[p], zorder=2)
    ax[0].set_xticks(x, [f"{CAC_K[k].split()[0]} luồng{' (thêm)' if k in NGOAI_TIEU_CHI else ''}\ntải {TEN_MUC[m]}"
                         for k, m in nhom], fontsize=8.5)
    ax[0].set_ylabel("Lưu lượng mất dịch vụ (%)")
    ax[0].set_title("(a) Mất dịch vụ trên Yagi (32 thế giới chấm)", loc="left", fontsize=11, pad=24)
    ax[0].grid(axis="y", **luoi)
    ax[0].legend(frameon=False, ncol=3, loc="lower left", bbox_to_anchor=(0, 1.0), borderaxespad=0.2, fontsize=9)
    so_k = [int(CAC_K[k].split()[0].split("–")[-1]) for k in ks]
    for p, ten, mau in (("dm", "DM", MAU["dm"]), ("dphat", "Dijkstra phạt", MAU["dphat"])):
        ax[1].plot(so_k, [ra[k]["vua"]["phuong_phap"][p]["tg_trung_vi_ms"] for k in ks], "-o", color=mau, lw=2, ms=6,
                   mec="white", mew=1.5, label=ten, zorder=3)
    t10 = [ra[k]["vua"]["tg_trung_vi_tkcb10_ms"] for k in ks]
    ax[1].plot(so_k, t10, "-o", color="#3b3a36", lw=2, ms=6, mec="white", mew=1.5, label="TKCB 10 lần", zorder=3)
    ax[1].axhline(1000, color="#8a8984", lw=1, ls="--")
    ax[1].annotate("1 s", (so_k[0], 1000), xytext=(2, 3), textcoords="offset points", fontsize=8.5, color="#52514e")
    ax[1].set_yscale("log")
    ax[1].set_xticks(so_k)
    ax[1].set_xlabel("Số luồng mỗi tình huống")
    ax[1].set_ylabel("Thời gian trung vị (ms, thang log)")
    ax[1].set_title("(b) Thời gian theo quy mô (tải vừa)", loc="left", fontsize=11, pad=24)
    ax[1].grid(**luoi)
    ax[1].legend(frameon=False, ncol=3, loc="lower left", bbox_to_anchor=(0, 1.0), borderaxespad=0.2, fontsize=9)
    fig.tight_layout()
    fig.savefig(KQ / "hinh_ket_qua.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    r = phan_tich()
    print(json.dumps({k: {m: v[m].get("tieu_chi", {}).get("thang") for m in MUC_TAI if m in v} for k, v in r.items()},
                     ensure_ascii=False))
