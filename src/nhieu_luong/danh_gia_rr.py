"""
danh_gia_rr.py — Chấm thí nghiệm 3 trên tập Yagi (xem results/bao_rr/TIEU_CHI_3.md).

    python -m src.nhieu_luong.danh_gia_rr chon-buoc               # số bước DM ∈ {3, 5, 10} trên val
    python -m src.nhieu_luong.danh_gia_rr test --phan 0/3          # chấm 1/3 số cửa sổ Yagi (chạy song song được)
    python -m src.nhieu_luong.danh_gia_rr phan-tich                # gộp, bootstrap, hình, KET_QUA.md

Mọi phương pháp chạy trong CÙNG một tiến trình, 1 luồng CPU, lần lượt trên từng tình huống. Kết quả từng tình huống:
results/bao_rr/test_<phần>.npz (mảng [tình huống, phương pháp] cho từng số đo). Khoảng tin cậy 95%: bootstrap ghép cặp
theo 13 cửa sổ Yagi, 10 000 lần (như test/phan_tich_yagi2.py).
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from src.nhieu_luong import bai_toan_rr as br
from src.nhieu_luong.huan_luyen_rr import nap, val_co_dinh
from src.nhieu_luong.mo_hinh_rr import MangRR, lay_mau_dm, lay_mau_softmax

GOC = Path(__file__).resolve().parents[2]
KQ = GOC / "results" / "bao_rr"
BUOC = (3, 5, 10)
S = 16
PP = ("nhanh_nhat", "tuan_tu_tre", "an_toan_nhat", "diem_doc_lap", "tham_lam", "tkcb", "softmax", "dm", "toi_uu")
TEN_PP = {"nhanh_nhat": "Nhanh nhất (chỉ trễ)", "tuan_tu_tre": "Tuần tự theo trễ", "an_toan_nhat":
          "An toàn nhất (chỉ rủi ro)", "diem_doc_lap": "Điểm, từng luồng độc lập", "tham_lam": "Tham lam theo điểm",
          "tkcb": "TKCB theo điểm", "softmax": "Softmax (S = 16)", "dm": "DM (S = 16)", "toi_uu": "Tối ưu (duyệt hết)"}
SO_DO = ("J", "H", "T", "rui_ro_%", "tre_ms", "qua_tai_%", "dut_luu_luong_%", "dut_luong_%", "tre_thuc_ms")
W_QUET = (0.0, 0.3, 0.5, 0.7, 0.9, 1.0)
MUC = ("nhe", "vua", "nang")
TEN_MUC = {"nhe": "nhẹ", "vua": "vừa", "nang": "nặng"}
B_BOOT = 10_000


def nap_mo_hinh(kieu):
    ck = torch.load(KQ / "mo_hinh" / f"{kieu}.pt", weights_only=False)
    net = MangRR(**ck["cau_hinh"])
    net.load_state_dict(ck["state"])
    return net.eval()


def chon_buoc():
    torch.set_num_threads(1)
    va = val_co_dinh(nap("val"))
    net = nap_mo_hinh("dm")
    ra = {}
    for n in BUOC:
        gen = torch.Generator().manual_seed(0)
        dJ, tg = [], []
        for x in va:
            s = time.perf_counter()
            _, J, _ = lay_mau_dm(net, x, S, n, gen)
            tg.append(time.perf_counter() - s)
            dJ.append(J - x["J_nhan"])
        ra[n] = {"dJ_tb": float(np.mean(dJ)), "ti_le_toi_uu_%": float(np.mean(np.array(dJ) <= 1e-9) * 100),
                 "tg_trung_vi_ms": float(np.median(tg) * 1e3)}
        print(n, ra[n], flush=True)
    tot = min(BUOC, key=lambda n: (round(ra[n]["dJ_tb"], 9), n))
    json.dump({"bang": ra, "so_buoc": tot, "quy_tac": "ΔJ trung bình nhỏ nhất trên 300 tình huống val; hoà → ít bước"},
              open(KQ / "chon_buoc.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("chọn", tot)


def _chay(ten, x, dm, sm, so_buoc, gen):
    if ten == "dm":
        return lay_mau_dm(dm, x, S, so_buoc, gen)[0]
    if ten == "softmax":
        return lay_mau_softmax(sm, x, S, gen)[0]
    if ten == "tkcb":
        return br.tkcb(x, br.tham_lam(x))
    if ten == "toi_uu":
        return br.toi_uu(x)[0]
    return getattr(br, ten)(x)


def test(phan):
    torch.set_num_threads(1)
    i, n = map(int, phan.split("/"))
    so_buoc = json.loads((KQ / "chon_buoc.json").read_text(encoding="utf-8"))["so_buoc"]
    dm, sm = nap_mo_hinh("dm"), nap_mo_hinh("softmax")
    gen = torch.Generator().manual_seed(2026 + i)
    ds = [x for x in nap("test") if x["kb"] % n == i]
    print(f"phần {phan}: {len(ds)} tình huống, DM {so_buoc} bước", flush=True)
    M = {k: np.full((len(ds), len(PP)), np.nan) for k in SO_DO + ("tg_ms", "dJ")}
    Q = {k: np.full((len(ds), len(W_QUET)), np.nan) for k in ("dut_luu_luong_%", "tre_thuc_ms", "rui_ro_%", "tre_ms")}
    t0 = time.time()
    for r, x in enumerate(ds):
        for c, ten in enumerate(PP):
            s = time.perf_counter()
            a = _chay(ten, x, dm, sm, so_buoc, gen)
            M["tg_ms"][r, c] = (time.perf_counter() - s) * 1e3
            kq = br.ket_qua(x, a)
            for k in SO_DO:
                M[k][r, c] = kq[k]
            M["dJ"][r, c] = kq["J"] - x["J_nhan"]
        for c, w in enumerate(W_QUET):
            kq = br.ket_qua(x, br.toi_uu(x, w)[0])
            for k in Q:
                Q[k][r, c] = kq[k]
        if r % 100 == 0:
            print(f"  {r + 1}/{len(ds)}  ({time.time() - t0:.0f} s)", flush=True)
    np.savez_compressed(KQ / f"test_{i}_{n}.npz", pp=np.array(PP), w=np.array(W_QUET),
                        kb=np.array([x["kb"] for x in ds]), t=np.array([x["t"] for x in ds]),
                        muc=np.array([x["muc"] for x in ds]), K=np.array([len(x["lam"]) for x in ds]),
                        **{f"m_{k}": v for k, v in M.items()}, **{f"q_{k}": v for k, v in Q.items()})
    print("xong", phan, flush=True)


# ---------------------------------------------------------------- phân tích
def bootstrap(v, kb, rng):
    """Trung bình và CI 95% khi lấy lại các cửa sổ (ghép cặp: v đã là hiệu từng tình huống)."""
    ok = np.isfinite(v)
    v, kb = v[ok], kb[ok]
    u = np.unique(kb)
    tong = np.array([v[kb == k].sum() for k in u])
    dem = np.array([(kb == k).sum() for k in u])
    j = rng.integers(0, len(u), (B_BOOT, len(u)))
    m = tong[j].sum(1) / dem[j].sum(1)
    return [float(v.mean()), [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]]


def _gop():
    tep = sorted(glob.glob(str(KQ / "test_*_*.npz")))
    z = [np.load(f, allow_pickle=False) for f in tep]
    return {k: np.concatenate([d[k] for d in z]) for k in z[0].files if k not in ("pp", "w")}


def phan_tich():
    d = _gop()
    rng = np.random.default_rng(0)
    ci = PP.index
    ra = {"so_tinh_huong": int(len(d["kb"])), "so_cua_so": int(len(np.unique(d["kb"])))}
    for muc in MUC:
        s = d["muc"] == muc
        kb = d["kb"][s]
        m = {k[2:]: d[k][s] for k in d if k.startswith("m_")}
        q = {k[2:]: d[k][s] for k in d if k.startswith("q_")}
        r = {"so_tinh_huong": int(s.sum()), "K_tb": float(d["K"][s].mean()), "phuong_phap": {}}
        for c, p in enumerate(PP):
            r["phuong_phap"][p] = {
                **{k: float(np.nanmean(m[k][:, c])) for k in SO_DO},
                "tre_thuc_trung_vi_ms": float(np.nanmedian(m["tre_thuc_ms"][:, c])),
                "dJ": float(m["dJ"][:, c].mean()),
                "ti_le_toi_uu_%": float((m["dJ"][:, c] <= 1e-9).mean() * 100),
                "tg_trung_vi_ms": float(np.median(m["tg_ms"][:, c]))}
        hieu = lambda k, a, b: m[k][:, ci(a)] - m[k][:, ci(b)]
        R1 = bootstrap(hieu("dut_luu_luong_%", "toi_uu", "nhanh_nhat"), kb, rng)
        R2 = bootstrap(hieu("dut_luu_luong_%", "toi_uu", "tuan_tu_tre"), kb, rng)
        C1 = bootstrap(hieu("J", "dm", "tham_lam"), kb, rng)
        C2 = bootstrap(hieu("J", "dm", "softmax"), kb, rng)
        C3 = r["phuong_phap"]["dm"]["ti_le_toi_uu_%"]
        r["Q1"] = {"R1_dut_toi_uu_tru_nhanh_nhat_diem_%": R1, "R2_dut_toi_uu_tru_tuan_tu_tre_diem_%": R2,
                   "tre_thuc_toi_uu_tru_nhanh_nhat_ms": bootstrap(hieu("tre_thuc_ms", "toi_uu", "nhanh_nhat"), kb, rng),
                   "tre_thuc_toi_uu_tru_tuan_tu_tre_ms": bootstrap(hieu("tre_thuc_ms", "toi_uu", "tuan_tu_tre"), kb, rng),
                   "tre_du_bao_toi_uu_tru_nhanh_nhat_ms": bootstrap(hieu("tre_ms", "toi_uu", "nhanh_nhat"), kb, rng),
                   "R1": R1[1][1] < 0, "R2": R2[1][1] < 0}
        r["Q1"]["dat"] = bool(r["Q1"]["R1"] and r["Q1"]["R2"])
        r["Q2"] = {"C1_J_dm_tru_tham_lam": C1, "C2_J_dm_tru_softmax": C2, "C3_ti_le_toi_uu_%": C3,
                   "J_dm_tru_tkcb": bootstrap(hieu("J", "dm", "tkcb"), kb, rng),
                   "C1": C1[1][1] < 0, "C2": C2[1][1] < 0, "C3": C3 >= 90}
        r["Q2"]["dat"] = bool(r["Q2"]["C1"] and r["Q2"]["C2"] and r["Q2"]["C3"])
        r["quet_w"] = {f"{w:g}": {"dut_luu_luong_%": float(np.nanmean(q["dut_luu_luong_%"][:, c])),
                                  "tre_thuc_trung_vi_ms": float(np.nanmedian(q["tre_thuc_ms"][:, c])),
                                  "tre_du_bao_ms": float(np.nanmean(q["tre_ms"][:, c])),
                                  "rui_ro_du_bao_%": float(np.nanmean(q["rui_ro_%"][:, c]))}
                       for c, w in enumerate(W_QUET)}
        ra[muc] = r
    ra["hau_kiem"] = hau_kiem()
    json.dump(ra, open(KQ / "ket_qua.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    ve_hinh(ra)
    viet_md(ra)
    print(json.dumps({m: {"Q1": ra[m]["Q1"]["dat"], "Q2": ra[m]["Q2"]["dat"]} for m in MUC}, ensure_ascii=False))


def hau_kiem():
    """Phân tích SAU khi có kết quả (không thuộc tiêu chí): phần đứt nào chọn tuyến còn tránh được, và bảng rủi ro hiệu
    chuẩn thế nào theo loại liên kết. Tuyến ứng viên, rủi ro và đứt thực tế không phụ thuộc mức tải → dùng mức vừa."""
    from src.ngau_nhien.kich_ban import loai_lien_ket
    with np.load(GOC / "dataset" / "oracle_dataset_v5.npz", allow_pickle=False) as z:
        kind = [str(k) for k in z["node_kind"]]
    N = len(kind)
    ra = {}
    for tap in ("val", "test"):
        ds = [x for x in nap(tap) if x["muc"] == "vua"]
        P, Y = {}, {}
        for x in ds:
            for e, r, s in zip(x["e"], x["risk"], x["song_thuc"]):
                u, v = divmod(int(e), N)
                t = loai_lien_ket(kind[u], kind[v])[0]
                nhom = t + (" có UAV" if "uav" in (kind[u], kind[v]) and "UAV" not in t else "")
                P.setdefault(nhom, []).append(float(r))
                Y.setdefault(nhom, []).append(not s)
        ra[f"hieu_chuan_{tap}"] = {k: {"so_canh": len(P[k]), "rui_ro_du_bao_%": float(np.mean(P[k]) * 100),
                                       "dut_thuc_te_%": float(np.mean(Y[k]) * 100)} for k in sorted(P) if len(P[k]) >= 100}
        if tap != "test":
            continue
        lam, loai, dut = [], [], {"nhanh_nhat": [], "an_toan_nhat": [], "toi_uu": []}
        for x in ds:
            m = x["canh"] >= 0
            d_uv = np.where(m, ~x["song_thuc"][np.maximum(x["canh"], 0)], False).any(-1)
            for k in range(len(x["lam"])):
                d = d_uv[k][x["hop_le"][k]]
                lam.append(x["lam"][k])
                loai.append("tat_ca_dut" if d.all() else ("khong_dut" if not d.any() else "tron"))
                for p, c in (("nhanh_nhat", 0), ("an_toan_nhat", x["i_an_toan"][k]), ("toi_uu", x["nhan"][k])):
                    dut[p].append(bool(d_uv[k, c]))
        lam, loai = np.array(lam), np.array(loai)
        ra["phan_luong_test_vua"] = {l: {"luong_%": float((loai == l).mean() * 100),
                                         "luu_luong_%": float(lam[loai == l].sum() / lam.sum() * 100)}
                                     for l in ("khong_dut", "tat_ca_dut", "tron")}
        ra["can_duoi_dut_%"] = ra["phan_luong_test_vua"]["tat_ca_dut"]["luu_luong_%"]
        s = loai == "tron"
        ra["dut_trong_nhom_tron_%"] = {p: float((np.array(v)[s] * lam[s]).sum() / lam[s].sum() * 100)
                                       for p, v in dut.items()}
    return ra


MAU_MUC = {"nhe": "#2a78d6", "vua": "#eb6834", "nang": "#1baf7a"}     # 3 ô đầu của bảng màu chuẩn (đủ mọi cặp)


def ve_hinh(ra):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
                         "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
                         "axes.edgecolor": "#8a8984", "axes.labelcolor": "#3b3a36", "xtick.color": "#52514e",
                         "ytick.color": "#52514e"})
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6), gridspec_kw={"width_ratios": [1.7, 1, 1]})
    luoi = dict(color="#e4e3df", lw=0.8, zorder=0)
    chu_thich = dict(frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0), borderaxespad=0.2, fontsize=9)
    x = np.arange(len(PP))
    bw = 0.26
    for j, muc in enumerate(MUC):
        v = [ra[muc]["phuong_phap"][p]["dut_luu_luong_%"] for p in PP]
        ax[0].bar(x + (j - 1) * (bw + 0.02), v, bw, color=MAU_MUC[muc], label=f"tải {TEN_MUC[muc]}", zorder=2)
    cd = ra["hau_kiem"]["can_duoi_dut_%"]
    ax[0].axhline(cd, color="#3b3a36", lw=1.2, ls="--", zorder=3)
    ax[0].annotate(f"cận dưới khi biết trước tương lai: {cd:.1f}%", (len(PP) - 0.5, cd), xytext=(0, -16),
                   textcoords="offset points", ha="right", fontsize=8.5, color="#3b3a36", zorder=4,
                   bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none"))
    ax[0].set_xticks(x, [TEN_PP[p] for p in PP], rotation=30, ha="right")
    ax[0].set_ylabel("Lưu lượng bị đứt thực tế trong [t, t+60 s] (%)")
    ax[0].set_title("(a) Đứt thực tế trên Yagi, theo phương pháp", loc="left", fontsize=11, pad=24)
    ax[0].grid(axis="y", **luoi)
    ax[0].legend(ncol=3, **chu_thich)
    for muc in MUC:
        qw = ra[muc]["quet_w"]
        ax[1].plot(W_QUET, [qw[f"{w:g}"]["rui_ro_du_bao_%"] for w in W_QUET], "-o", color=MAU_MUC[muc], lw=2, ms=6,
                   mec="white", mew=1.5, zorder=3, label=f"tải {TEN_MUC[muc]}")
        ax[1].plot(W_QUET, [qw[f"{w:g}"]["dut_luu_luong_%"] for w in W_QUET], "--o", color=MAU_MUC[muc], lw=1.5,
                   ms=5, mec="white", mew=1.2, zorder=3)
        ax[2].plot(W_QUET, [qw[f"{w:g}"]["tre_du_bao_ms"] for w in W_QUET], "-o", color=MAU_MUC[muc], lw=2, ms=6,
                   mec="white", mew=1.5, zorder=3, label=f"tải {TEN_MUC[muc]}")
    ax[1].set_ylim(0, None)
    ax[1].text(0.03, 0.05, "nét đứt (trên): đứt thực tế\nnét liền (dưới): rủi ro dự báo (bảng tra)",
               transform=ax[1].transAxes, va="bottom", fontsize=8.5, color="#52514e")
    ax[1].set_xlabel("Trọng số rủi ro w (đã chọn 0,7)")
    ax[1].set_ylabel("% lưu lượng")
    ax[1].set_title("(b) Rủi ro dự báo giảm, đứt thật thì không", loc="left", fontsize=11, pad=24)
    ax[2].set_yscale("log")
    ax[2].set_xlabel("Trọng số rủi ro w (đã chọn 0,7)")
    ax[2].set_ylabel("Trễ dự báo trung bình (ms, thang log)")
    ax[2].set_title("(c) Trễ phải trả theo w", loc="left", fontsize=11, pad=24)
    for a in ax[1:]:
        a.axvline(0.7, color="#8a8984", lw=0.8, ls=":", zorder=1)
        a.grid(**luoi)
        a.legend(ncol=3, **chu_thich)
    fig.tight_layout()
    fig.savefig(KQ / "hinh_ket_qua.png", dpi=160)
    plt.close(fig)


def _ci(v, nd=2):
    return f"{v[0]:+.{nd}f} [{v[1][0]:+.{nd}f}; {v[1][1]:+.{nd}f}]"


def viet_md(ra):
    L = ["# Thí nghiệm 3 — kết quả trên Yagi (sinh tự động bởi `danh_gia_rr.py phan-tich`)", "",
         f"{ra['so_tinh_huong']} tình huống = {ra['so_cua_so']} cửa sổ Yagi × mốc test hình học v5 × 3 mức tải; "
         "4–5 luồng mỗi tình huống. CI 95%: bootstrap ghép cặp theo cửa sổ.", ""]
    for muc in MUC:
        r = ra[muc]
        L += [f"## Tải {TEN_MUC[muc]} ({r['so_tinh_huong']} tình huống, K trung bình {r['K_tb']:.2f})", "",
              "| Phương pháp | J | ΔJ tới tối ưu | % đạt tối ưu | Rủi ro dự báo % | Trễ dự báo ms | Quá tải % | "
              "**Đứt thực tế %** (lưu lượng) | Đứt % (số luồng) | Trễ thật TB / trung vị ms | Thời gian ms |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for p in PP:
            v = r["phuong_phap"][p]
            L.append(f"| {TEN_PP[p]} | {v['J']:.4f} | {v['dJ']:.4f} | {v['ti_le_toi_uu_%']:.1f} | {v['rui_ro_%']:.2f} | "
                     f"{v['tre_ms']:.2f} | {v['qua_tai_%']:.2f} | **{v['dut_luu_luong_%']:.2f}** | "
                     f"{v['dut_luong_%']:.2f} | {v['tre_thuc_ms']:.1f} / {v['tre_thuc_trung_vi_ms']:.2f} | "
                     f"{v['tg_trung_vi_ms']:.1f} |")
        q1, q2 = r["Q1"], r["Q2"]
        dd = lambda b: "đạt" if b else "không đạt"
        L += ["", f"- R1 (đứt: tối ưu − nhanh nhất, điểm %): {_ci(q1['R1_dut_toi_uu_tru_nhanh_nhat_diem_%'])} → "
              f"{dd(q1['R1'])}",
              f"- R2 (đứt: tối ưu − tuần tự theo trễ, điểm %): {_ci(q1['R2_dut_toi_uu_tru_tuan_tu_tre_diem_%'])} → "
              f"{dd(q1['R2'])}",
              f"- Trễ phải trả (tối ưu − nhanh nhất): dự báo {_ci(q1['tre_du_bao_toi_uu_tru_nhanh_nhat_ms'])} ms, "
              f"thật {_ci(q1['tre_thuc_toi_uu_tru_nhanh_nhat_ms'], 1)} ms",
              f"- **Q1: {dd(q1['dat'])}**",
              f"- C1 (J: DM − tham lam): {_ci(q2['C1_J_dm_tru_tham_lam'], 4)} → {dd(q2['C1'])}",
              f"- C2 (J: DM − softmax): {_ci(q2['C2_J_dm_tru_softmax'], 4)} → {dd(q2['C2'])}",
              f"- C3 (DM đạt tối ưu): {q2['C3_ti_le_toi_uu_%']:.1f}% (cần ≥ 90%) → {dd(q2['C3'])}",
              f"- Tham khảo, J: DM − TKCB: {_ci(q2['J_dm_tru_tkcb'], 4)}",
              f"- **Q2 (DM thắng): {dd(q2['dat'])}**", "",
              "Quét trọng số rủi ro w (lời giải tối ưu theo từng w):", "",
              "| w | Đứt thực tế % | Rủi ro dự báo % | Trễ dự báo ms | Trễ thật trung vị ms |", "|---:|---:|---:|---:|---:|"]
        for w, v in r["quet_w"].items():
            L.append(f"| {w} | {v['dut_luu_luong_%']:.2f} | {v['rui_ro_du_bao_%']:.2f} | {v['tre_du_bao_ms']:.2f} | "
                     f"{v['tre_thuc_trung_vi_ms']:.2f} |")
        L.append("")
    hk = ra["hau_kiem"]
    L += ["## Hậu kiểm (phân tích sau khi có kết quả — không thuộc tiêu chí)", "",
          "Phân loại luồng trên Yagi (mức tải vừa; tuyến ứng viên và đứt thực tế không đổi theo mức tải):", "",
          "| Nhóm luồng | % số luồng | % lưu lượng |", "|---|---:|---:|"]
    ten_nhom = {"khong_dut": "Không ứng viên nào đứt", "tat_ca_dut": "Mọi ứng viên đều đứt (không tránh được)",
                "tron": "Có ứng viên đứt, có ứng viên sống (chọn tuyến quyết định)"}
    for l, v in hk["phan_luong_test_vua"].items():
        L.append(f"| {ten_nhom[l]} | {v['luong_%']:.1f} | {v['luu_luong_%']:.1f} |")
    L += ["", "Đứt trong nhóm \"chọn tuyến quyết định\": " +
          ", ".join(f"{TEN_PP[p]} {v:.1f}%" for p, v in hk["dut_trong_nhom_tron_%"].items()), "",
          "Hiệu chuẩn bảng rủi ro theo loại liên kết (các cạnh nằm trên ứng viên):", "",
          "| Loại liên kết | Val: n | Val: dự báo % | Val: thực tế % | Yagi: n | Yagi: dự báo % | Yagi: thực tế % |",
          "|---|---:|---:|---:|---:|---:|---:|"]
    hv, ht = hk["hieu_chuan_val"], hk["hieu_chuan_test"]
    for k in sorted(set(hv) | set(ht)):
        a, b = hv.get(k), ht.get(k)
        f = lambda d: (f"{d['so_canh']} | {d['rui_ro_du_bao_%']:.2f} | {d['dut_thuc_te_%']:.2f}" if d else "– | – | –")
        L.append(f"| {k} | {f(a)} | {f(b)} |")
    L.append("")
    (KQ / "KET_QUA.md").write_text("\n".join(L), encoding="utf-8")


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("viec", choices=["chon-buoc", "test", "phan-tich"])
    ap.add_argument("--phan", default="0/1")
    a = ap.parse_args()
    if a.viec == "chon-buoc":
        chon_buoc()
    elif a.viec == "test":
        test(a.phan)
    else:
        phan_tich()


if __name__ == "__main__":
    main()
