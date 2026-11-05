"""
truc_quan_du_lieu.py — Hình trực quan hoá DỮ LIỆU của thí nghiệm 6–7 (tai_lieu/BAO_CAO_THI_NGHIEM_6_7.md, mục 3).

    python -m scripts.truc_quan_du_lieu          # → tai_lieu/hinh_tn6_tn7/d1_…d5_*.png

d1  bản đồ mạng SAGSIN (nút mặt đất, UAV, HAPS) + tần suất giờ giông ở 74 điểm thời tiết
d2  dòng thời gian: cửa sổ của từng tập train / val / test trên 2 năm thời tiết 2024–2025
d3  mức khắc nghiệt thời tiết của từng tập (giông, gió giật, mưa lớn trong cửa sổ)
d4  ví dụ điểm mù UAV (lý do của thí nghiệm 6): UAV hạ cánh khi sang giờ mới
d5  phân bố hệ số tải ρ̂ và ba dải tải dùng để chia ô tiêu chí

Chỉ đọc dữ liệu thời tiết + các tệp kịch bản đã sinh (data/bao_da_tuyen/*); không sinh lại gì.
"""

from __future__ import annotations

import glob
import json
import pickle
import sys
from pathlib import Path

import numpy as np

from src.ngau_nhien.thoi_tiet import GIONG
from src.nhieu_luong import tao_du_lieu_dt as tdt
from src.nhieu_luong import tao_du_lieu_rr as td

GOC = Path(__file__).resolve().parents[1]
RA = GOC / "tai_lieu" / "hinh_tn6_tn7"
MAU = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#8a5cd1", "#52514e", "#a3a29d"]
VUNG = (101.5, 113.5, 7.8, 23.8)                       # lon_min, lon_max, lat_min, lat_max
GIO_CUA_SO = 4                                          # mỗi kịch bản 600 mốc × 20 s ≈ 3 giờ 20 phút → xét 4 giờ thời tiết


def kieu():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
                         "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
                         "axes.edgecolor": "#8a8984", "axes.labelcolor": "#3b3a36", "xtick.color": "#52514e",
                         "ytick.color": "#52514e"})
    return plt


def duong_bo():
    """Giải mã topojson world_50m (đường bờ biển + biên giới) → danh sách polyline (lon, lat) trong VUNG."""
    t = json.loads((GOC / "data" / "topojson" / "world_50m.json").read_text(encoding="utf-8"))
    sx, sy = t["transform"]["scale"]
    tx, ty = t["transform"]["translate"]
    ra = []
    for arc in t["arcs"]:
        q = np.cumsum(np.array(arc, float), axis=0)
        x, y = q[:, 0] * sx + tx, q[:, 1] * sy + ty
        if (x.max() >= VUNG[0] - 2) and (x.min() <= VUNG[1] + 2) and (y.max() >= VUNG[2] - 2) and (y.min() <= VUNG[3] + 2):
            ra.append((x, y))
    return ra


def nut_mang():
    """Tên, loại, toạ độ của các nút có vị trí cố định (bỏ 150 vệ tinh LEO)."""
    z = np.load(GOC / "dataset" / "oracle_dataset_v5.npz", allow_pickle=False)
    kind, ten = [str(k) for k in z["node_kind"]], [str(n) for n in z["node_order"]]
    tt = td._tt
    ra = {}
    for n, k in zip(ten, kind):
        if k != "leo" and n in tt.names:
            i = tt.idx(n)
            ra.setdefault(k, []).append((n, float(tt.lat[i]), float(tt.lon[i])))
    return ra, kind.count("leo")


def d1_ban_do(plt):
    tt = td._tt
    g = np.isin(tt.code, GIONG)                                                   # [giờ, điểm]
    tan_suat = 100 * g.mean(0)
    nut, so_leo = nut_mang()
    fig, ax = plt.subplots(figsize=(7.2, 8.4))
    for x, y in duong_bo():
        ax.plot(x, y, color="#a3a29d", lw=0.6, zorder=1)
    sc = ax.scatter(tt.lon, tt.lat, c=tan_suat, s=170, cmap="YlOrRd", vmin=0, vmax=max(4.0, float(tan_suat.max())),
                    edgecolors="none", alpha=0.55, zorder=2)
    kieu_nut = {"gateway": ("s", MAU[0], 70, "Gateway (5)"), "user": ("o", MAU[6], 26, "Người dùng mặt đất (27)"),
                "uav": ("^", MAU[2], 42, "UAV (35)"), "haps": ("D", MAU[5], 60, "HAPS (7)")}
    for k, (mk, mau, s, nhan) in kieu_nut.items():
        xs = nut.get(k, [])
        ax.scatter([p[2] for p in xs], [p[1] for p in xs], marker=mk, s=s, color=mau, edgecolors="white", linewidths=0.8,
                   label=nhan, zorder=4)
    lech = {"HaiPhong": (8, -12), "Hanoi": (-38, 6)}
    for n, la, lo in nut.get("gateway", []):
        ax.annotate(n, (lo, la), xytext=lech.get(n, (6, 4)), textcoords="offset points", fontsize=8, color=MAU[0],
                    zorder=5)
    ax.set_xlim(VUNG[0], VUNG[1])
    ax.set_ylim(VUNG[2], VUNG[3])
    ax.set_aspect(1 / np.cos(np.radians(16)))
    ax.set_xlabel("Kinh độ (°E)")
    ax.set_ylabel("Vĩ độ (°N)")
    cb = fig.colorbar(sc, ax=ax, shrink=0.55, pad=0.02)
    cb.set_label("Tỉ lệ giờ có giông tại điểm thời tiết (%)")
    cb.outline.set_visible(False)
    ax.legend(loc="lower left", frameon=False, fontsize=8.5, title=f"+ {so_leo} vệ tinh LEO (không vẽ)",
              title_fontsize=8.5)
    ax.set_title("Mạng SAGSIN v5 và tần suất giông (2024–2025)", loc="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(RA / "d1_ban_do_mang_va_giong.png", dpi=160)
    plt.close(fig)


TAP = [  # (tên tập trong mã, nhãn hiển thị, vai trò)
    ("train", "Train bão (TN 4)", "train"), ("train6", "Train6 — ngày giông", "train"),
    ("train7x", "Train7x — ngày xấu mới (7b)", "train"), ("val", "Val bão", "val"), ("val6", "Val6 — ngày giông", "val"),
    ("c6", "Test C — ngày giông", "test"), ("mua5", "Test B — mưa lớn", "test"), ("yagi5", "Test A — Yagi", "test")]
MAU_VAI = {"train": MAU[0], "val": MAU[3], "test": MAU[1]}


def cua_so():
    return {tap: tdt.cac_cua_so_tap(tap) for tap, _, _ in TAP}


def d2_dong_thoi_gian(plt, cs):
    import matplotlib.dates as mdates
    from datetime import datetime
    T = td._T
    fig, ax = plt.subplots(figsize=(11, 4.2))
    for j, (tap, nhan, vai) in enumerate(TAP):
        ngay = [datetime.fromisoformat(T[h0]) for _, h0, _ in cs[tap]]
        ax.scatter(ngay, [j] * len(ngay), marker="|", s=260, color=MAU_VAI[vai], linewidths=1.6, zorder=3)
    ax.set_yticks(range(len(TAP)), [f"{n} ({len(cs[tap])})" for tap, n, _ in TAP], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(datetime(2024, 1, 1), datetime(2026, 1, 1))
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 4, 7, 10)))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%Y"))
    ax.grid(axis="x", color="#e4e3df", lw=0.8, zorder=0)
    ax.set_title("Cửa sổ kịch bản của từng tập (số cửa sổ trong ngoặc; xanh: train, vàng: val, cam: test)", loc="left",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(RA / "d2_dong_thoi_gian_cac_tap.png", dpi=160)
    plt.close(fig)


def muc_khac_nghiet(h0):
    tt = td._tt
    h = np.arange(h0, min(h0 + GIO_CUA_SO, tt.H))
    return (100 * np.isin(tt.code[h], GIONG).mean(), 100 * (tt.gust[h] > 15).mean(), 100 * (tt.precip[h] > 11).mean())


def d3_khac_nghiet(plt, cs):
    ten_do = ["Giờ-điểm có giông (%)", "Giờ-điểm gió giật > 15 m/s (%)", "Giờ-điểm mưa > 11 mm/h (%)"]
    gia_tri = {tap: np.array([muc_khac_nghiet(h0) for _, h0, _ in cs[tap]]) for tap, _, _ in TAP}
    fig, ax = plt.subplots(1, 3, figsize=(13, 4.4), sharey=True)
    rng = np.random.default_rng(0)
    for i, ten in enumerate(ten_do):
        for j, (tap, nhan, vai) in enumerate(TAP):
            v = gia_tri[tap][:, i]
            ax[i].scatter(v, j + rng.uniform(-0.18, 0.18, len(v)), s=12, color=MAU_VAI[vai], alpha=0.65, edgecolors="none",
                          zorder=3)
            ax[i].plot([np.median(v)] * 2, [j - 0.32, j + 0.32], color="#3b3a36", lw=1.6, zorder=4)
        ax[i].set_xlabel(ten)
        ax[i].grid(axis="x", color="#e4e3df", lw=0.8, zorder=0)
        ax[i].set_xscale("symlog", linthresh=1)
    ax[0].set_yticks(range(len(TAP)), [n for _, n, _ in TAP], fontsize=9)
    ax[0].invert_yaxis()
    fig.suptitle("Mức khắc nghiệt thời tiết trong 4 giờ của mỗi cửa sổ (mỗi chấm một cửa sổ, vạch đen: trung vị)",
                 x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(RA / "d3_muc_khac_nghiet_cac_tap.png", dpi=160)
    plt.close(fig)
    return {tap: [float(x) for x in np.median(v, 0)] for tap, v in gia_tri.items()}


def d4_uav_ha_canh(plt, cs):
    """Ngày C và UAV có nhiều giờ hạ cánh DO GIÔNG nhất quanh cửa sổ (ưu tiên ca có giông lân cận báo trước)."""
    tt = td._tt
    nut, _ = nut_mang()
    uav = [(n, tt.idx(n)) for n, _, _ in nut["uav"]]
    la, lo = np.radians(tt.lat), np.radians(tt.lon)

    def lan_can(s):
        return 6371 * np.arccos(np.clip(np.sin(la[s]) * np.sin(la) + np.cos(la[s]) * np.cos(la) * np.cos(lo - lo[s]),
                                        -1, 1)) <= 100

    tot = None
    for ten, h0, _ in cs["c6"]:
        h = np.arange(h0 - 6, h0 + 10)
        for n, s in uav:
            gi = np.isin(tt.code[h, s], GIONG)
            ga = np.isin(tt.code[h][:, lan_can(s)], GIONG).any(1)
            ha = gi | (tt.gust[h, s] > 15)
            bao_truoc = int((ga[:-1] & ~gi[:-1] & gi[1:]).sum())     # giông lân cận xuất hiện TRƯỚC khi UAV hạ cánh
            khoa = (bao_truoc, int(np.abs(np.diff(ha.astype(int))).sum()), int(gi.sum()))
            if tot is None or khoa > tot[0]:
                tot = (khoa, ten, h0, n, s)
    _, ten, h0, n, s = tot
    h = np.arange(h0 - 6, h0 + 10)
    gio = h - h0
    giong = np.isin(tt.code[h, s], GIONG)
    gan = np.isin(tt.code[h][:, lan_can(s)], GIONG).any(1)
    ha = giong | (tt.gust[h, s] > 15)
    fig, ax = plt.subplots(2, 1, figsize=(10, 5.2), sharex=True, gridspec_kw={"height_ratios": [1.4, 1]})
    ax[0].step(gio, tt.gust[h, s], where="post", color=MAU[0], lw=2, label="Gió giật tại điểm UAV")
    ax[0].axhline(15, color=MAU[1], lw=1.2, ls="--")
    ax[0].text(gio[-1] + 0.9, 15, "ngưỡng hạ cánh 15 m/s", color=MAU[1], fontsize=8.5, va="bottom", ha="right")
    ax[0].set_ylabel("Gió giật (m/s)")
    ax[0].grid(axis="y", color="#e4e3df", lw=0.8)
    for y, (v, mau, nhan) in enumerate([(gan, MAU[3], "Giông trong 100 km (đặc trưng mới, thấy lúc quyết định)"),
                                        (giong, MAU[4], "Giông tại điểm UAV"),
                                        (ha, MAU[1], "UAV hạ cánh (mô phỏng)")]):
        for g, b in zip(gio, v):
            if b:
                ax[1].broken_barh([(g, 1)], (y - 0.35, 0.7), color=mau)
    ax[1].set_yticks([0, 1, 2], ["Giông ≤ 100 km", "Giông tại UAV", "UAV hạ cánh"], fontsize=9)
    for a in ax:
        for g in gio:
            a.axvline(g, color="#e4e3df", lw=0.6, zorder=0)
        a.axvspan(0, 3 + 1 / 3, color="#2a78d6", alpha=0.06, zorder=0)
    ax[1].set_xlabel(f"Giờ so với đầu cửa sổ (cửa sổ tô xanh nhạt) — {n}, ngày {td._T[h0][:10]}")
    ax[1].set_xlim(gio[0], gio[-1] + 1)
    ax[0].set_title("Điểm mù UAV: trạng thái chỉ đổi ở ranh giới giờ, theo thời tiết của giờ TỚI", loc="left",
                    fontsize=11)
    fig.tight_layout()
    fig.savefig(RA / "d4_vi_du_uav_ha_canh.png", dpi=160)
    plt.close(fig)
    return {"uav": n, "ngay": td._T[h0][:10], "so_lan_doi": int(np.abs(np.diff(ha.astype(int))).sum()),
            "gio_ha_canh": int(ha.sum()), "gio_giong_tai_uav": int(giong.sum()), "gio_giong_lan_can": int(gan.sum())}


def d5_tai(plt):
    from scripts.phan_tich_5 import DAI
    bo = [("c6", "C ngày giông"), ("mua5", "B mưa lớn"), ("yagi5", "A Yagi")]
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    so = {}
    bins = np.linspace(0.5, 1.3, 33)
    for j, (tap, nhan) in enumerate(bo):
        rho = [r["rho_hat"] for f in sorted(glob.glob(str(tdt.RA / f"{tap}_k4_5" / "kb_*.pkl")))
               for r in pickle.load(open(f, "rb"))]
        so[tap] = {d: int(sum(lo <= x < hi for x in rho)) for d, (lo, hi) in DAI.items()}
        ax.hist(rho, bins=bins, histtype="step", lw=2, color=MAU[j], label=f"{nhan} (n = {len(rho)})")
    for (d, (lo, hi)), ten in zip(DAI.items(), ("thấp", "giữa", "cao")):
        ax.axvspan(lo, min(hi, 1.3), color="#3b3a36", alpha=[0.03, 0.07, 0.11][list(DAI).index(d)], zorder=0)
        ax.text((lo + min(hi, 1.3)) / 2, 0.98, f"dải {ten}", transform=ax.get_xaxis_transform(), ha="center", va="top",
                fontsize=9, color="#3b3a36")
    ax.set_xlabel("Hệ số tải ρ̂ ước lượng lúc quyết định (tải sinh liên tục U[0,5; 1,3])")
    ax.set_ylabel("Số tình huống")
    ax.legend(frameon=False, fontsize=8.5, loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=3)
    fig.tight_layout()
    fig.savefig(RA / "d5_phan_bo_tai.png", dpi=160)
    plt.close(fig)
    return so


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    RA.mkdir(parents=True, exist_ok=True)
    td._khoi_tao_nhe()
    plt = kieu()
    cs = cua_so()
    d1_ban_do(plt)
    d2_dong_thoi_gian(plt, cs)
    tom = {"so_cua_so": {tap: len(v) for tap, v in cs.items()}, "trung_vi_khac_nghiet_%": d3_khac_nghiet(plt, cs),
           "vi_du_uav": d4_uav_ha_canh(plt, cs), "so_tinh_huong_theo_dai_k4_5": d5_tai(plt)}
    (RA / "so_lieu_du_lieu.json").write_text(json.dumps(tom, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(tom, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
