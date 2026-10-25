"""
Bước 8 — Nhiễu do mưa và giông, suy từ vật lý
File dự án: src/ngau_nhien/nhieu_mua.py. Tài liệu: tai_lieu/Co_so_nhieu_mua_giong.pdf.
Chuỗi: mưa làm tín hiệu yếu đi (suy hao) → modem hạ tốc độ (Shannon) → gói xếp hàng lâu hơn (hàng đợi M/M/1) →
thời gian xếp hàng ngẫu nhiên nên số đo trễ dao động mạnh hơn: đó là nhiễu đo Δ.
Sẽ thấy: sơ đồ chuỗi; mưa thật ở Hà Nội tháng 7/2024; ví dụ tính tay; mô phỏng kiểm chứng hai công thức; đường Δ theo
lượng mưa cho từng loại liên kết; bảng kiểm tra.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from huong_dan.tien_ich import *  # noqa: E402,F403


def chay():
    from src.ngau_nhien.kich_ban import MangGoc, suy_hao_mua
    from src.ngau_nhien.nhieu_mua import S_MUA, delta, thoi_gian_o_lai, ti_le_toc_do
    from src.ngau_nhien.thoi_tiet import ThoiTiet
    tt = ThoiTiet(str(THOI_TIET))

    fig, ax = plt.subplots(figsize=(15, 1.8))
    ve_chuoi(["Mưa giờ $\\bar R$\n(ERA5 thật)", "Mưa từng lúc\n$R=\\bar R e^{sz-s^2/2}$", "Suy hao\n$A=kR^\\alpha L_s r$",
              "SNR $=15-A$ dB", "Tốc độ còn lại\n$c$ (Shannon)", "Hàng đợi M/M/1\n$m=\\frac{12/(cC)}{1-\\rho_0/c}$",
              "Nhiễu đo $\\Delta$"], ax=ax, mau={6: "#e8f0fb"}, w=1.75)

    gio = [k for k, t in enumerate(tt.time_utc) if str(t).startswith("2024-07")]
    P = tt.precip[gio, tt.names.index("Hanoi")]
    mau = np.where(P > 10, "#b2182b", np.where(P > 2.5, MAU["cam"], np.where(P > 0.1, MAU["xanh"], "#cccccc")))
    fig, ax = plt.subplots(figsize=(14, 2.8))
    ax.bar(np.arange(len(P)) / 24, P, width=1 / 24, color=mau)
    ax.set_xlabel("Ngày trong tháng 7/2024")
    ax.set_ylabel("mm/h")
    ax.set_title("Mưa thật tại Hà Nội (ERA5): xanh = nhẹ, cam = vừa, đỏ = to", loc="left", fontsize=11)
    fig.tight_layout()

    th, R, C, Wv = np.radians(40), 5.0, 100.0, 3.0
    Ls, LG = 5 / np.sin(th), 5 / np.sin(th) * np.cos(th)
    A = float(suy_hao_mua(np.array(R), np.array(20), np.array(Ls), np.array(LG)))
    c = float(ti_le_toc_do(A))
    m0, m = float(thoi_gian_o_lai(C, 1.0)), float(thoi_gian_o_lai(C, c))
    dl, pm = delta(R, 0.0, 20, C, Ls, LG, Wv)
    buoc = [("Quãng đường trong mưa", "Ls = 5 / sin 40°", f"{Ls:.2f} km"),
            ("Suy hao do mưa", "A = k·R^α·Ls·r (20 GHz)", f"{A:.2f} dB"),
            ("Tín hiệu / nhiễu", "SNR = 15 − A", f"{15 - A:.1f} dB"),
            ("Tốc độ còn lại", "c = log2(1+SNR) / log2(1+SNR0)", f"{c:.3f}"),
            ("Thời gian trong hàng đợi", "m0 (trời quang) → m (mưa)", f"{m0:.3f} → {m:.3f} ms"),
            ("Nhiễu đo do mưa (tính cả mưa dao động)", "Δ = √(2E[m²] − E[m]² − m0²) / W", f"{float(dl):.3f}"),
            ("Thời gian liên kết nghẽn", "P(tải > 90% tốc độ còn lại)", f"{float(pm) * 100:.1f}%")]
    fig, ax = plt.subplots(figsize=(13, 3.0))
    ax.axis("off")
    bang = ax.table(cellText=[list(b) for b in buoc], colLabels=["Bước", "Công thức", "Giá trị"], loc="center",
                    cellLoc="left", colWidths=[0.35, 0.45, 0.2])
    bang.auto_set_font_size(False)
    bang.set_fontsize(10.5)
    bang.scale(1, 1.45)
    for (r, _), o in bang.get_celld().items():
        o.set_edgecolor("#d9d8d3")
        if r == 0:
            o.set_facecolor("#f3f2ee")
    ax.set_title("Ví dụ tính tay: gateway – vệ tinh 20 GHz, góc ngẩng 40°, mưa 5 mm/h, W = 3 ms", loc="left", fontsize=11)

    rng = np.random.default_rng(0)
    Rm = 5.0 * np.exp(S_MUA * rng.standard_normal(400_000) - S_MUA ** 2 / 2)
    a = 1 / m
    T = rng.exponential(1 / a, 400_000)
    fig, axs = plt.subplots(1, 2, figsize=(13, 3.4))
    axs[0].hist(Rm, bins=np.geomspace(0.05, 300, 80), color=MAU["xanh"], alpha=0.8)
    axs[0].axvline(Rm.mean(), color=MAU["cam"], lw=2, label=f"trung bình mô phỏng {Rm.mean():.2f} mm/h (lý thuyết 5)".replace(".", ","))
    axs[0].set_xscale("log")
    axs[0].set_xlabel("Mưa từng lúc R (mm/h)")
    axs[0].legend(frameon=False, fontsize=9)
    axs[0].set_title("Mưa dao động quanh trung bình giờ 5 mm/h (s = 1,4)", loc="left", fontsize=11)
    axs[1].hist(T, bins=80, density=True, color=MAU["luc"], alpha=0.7, label="mô phỏng")
    t = np.linspace(0, T.max(), 300)
    axs[1].plot(t, a * np.exp(-a * t), color=MAU["den"], lw=1.5, label="lý thuyết $a e^{-at}$")
    axs[1].set_xlim(0, 6 * m)
    axs[1].set_xlabel("Thời gian trong hàng đợi T (ms)")
    axs[1].set_title(f"Phân phối mũ: trung bình {T.mean():.3f}, độ lệch chuẩn {T.std():.3f} ms".replace(".", ","), loc="left", fontsize=11)
    axs[1].legend(frameon=False)
    fig.tight_layout()

    d7 = doc_demo(73)
    g7 = MangGoc(d7.W, d7.kind, d7.node_order, tt)
    s_, u_, v_ = np.nonzero((d7.W > 0) & np.triu(np.ones((73, 73), bool), 1)[None])
    LOAI = {("GSL", 20): "Gateway – vệ tinh 20 GHz", ("GSL", 12): "Người dùng – vệ tinh 12 GHz",
            ("ASL", 12): "UAV – vệ tinh 12 GHz", ("HAPS-MĐ", 28): "HAPS – mặt đất 28 GHz"}
    ten = np.array([LOAI.get((g7.ten[u, v], g7.band[u, v]), "Liên kết UAV 5 GHz" if g7.band[u, v] == 5 else "")
                    for u, v in zip(u_, v_)])
    Rg = np.geomspace(0.2, 40, 18)
    fig, axs = plt.subplots(1, 2, figsize=(13, 3.8))
    cong = {}
    for L, mau in zip(["HAPS – mặt đất 28 GHz", "Gateway – vệ tinh 20 GHz", "Người dùng – vệ tinh 12 GHz",
                       "UAV – vệ tinh 12 GHz", "Liên kết UAV 5 GHz"], (MAU["cam"], MAU["vang"], MAU["xanh"], MAU["luc"], MAU["xam"])):
        mk = np.flatnonzero(ten == L)[:150]
        s, u, v = s_[mk], u_[mk], v_[mk]
        dd, nn = [], []
        for Rv in Rg:
            dx, px = delta(np.full(len(mk), Rv), np.zeros(len(mk)), g7.band[u, v], g7.C[u, v], g7.Ls[s, u, v],
                           g7.LG[s, u, v], d7.W[s, u, v])
            dd.append(np.median(np.where(np.isfinite(dx), dx, np.inf)))
            nn.append(px.mean() * 100)
        cong[L] = dd
        axs[0].plot(Rg, dd, "o-", ms=3, color=mau, lw=1.8, label=L)
        axs[1].plot(Rg, nn, "o-", ms=3, color=mau, lw=1.8)
    axs[0].plot([0.2, 2.5, 2.5, 10, 10, 40], [0.1, 0.1, 0.25, 0.25, 0.4, 0.4], "k--", lw=1.2, label="Bảng giả định cũ")
    for ax, t_ in zip(axs, ("Nhiễu đo do mưa Δ", "Thời gian liên kết nghẽn (%)")):
        ax.set_xscale("log")
        ax.set_xlabel("Mưa trung bình giờ (mm/h)")
        ax.set_title(t_, loc="left", fontsize=11)
    axs[0].set_ylim(0, 0.5)
    axs[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return [kiem_tra("Mưa mô phỏng giữ đúng trung bình", lambda: (abs(Rm.mean() - 5) < 0.1, f"{Rm.mean():.3f} mm/h")),
            kiem_tra("Phân phối mũ: độ lệch chuẩn = trung bình", lambda: (abs(T.std() / T.mean() - 1) < 0.01, f"tỉ số {T.std() / T.mean():.4f}")),
            kiem_tra("Không mưa thì không thêm nhiễu", lambda: (float(delta(0.0, 0.0, 20, 100.0, Ls, LG, 3.0)[0]) < 1e-6, "")),
            kiem_tra("Mưa càng to nhiễu càng lớn", lambda: (all(np.diff(cong["Gateway – vệ tinh 20 GHz"]) >= -5e-3), "20 GHz")),
            kiem_tra("Tần số càng cao càng chịu mưa", lambda: (cong["HAPS – mặt đất 28 GHz"][9] > cong["Gateway – vệ tinh 20 GHz"][9]
                     > cong["Người dùng – vệ tinh 12 GHz"][9] > cong["Liên kết UAV 5 GHz"][9], f"ở mưa {Rg[9]:.1f} mm/h"))]


if __name__ == "__main__":
    hien_thi(chay, "Bước 8 — Nhiễu do mưa")
