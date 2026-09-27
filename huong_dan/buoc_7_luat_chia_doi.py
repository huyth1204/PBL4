"""
Bước 7 — Luật chia đôi Dijkstra / DM với thời tiết thật
File dự án: test/luat_chia_doi.py, src/ngau_nhien/nhieu_mua.py. Dữ liệu thời tiết: data/thoi_tiet_vn.npz (ERA5 2024–2025).
Luật: đo trễ mỗi liên kết 2 lần để ước lượng chất lượng đo σ̂0; nếu σ̂0 ≤ 7,5% và mọi liên kết trên tuyến Dijkstra có
nhiễu dự báo √(σ̂0² + Δ²) ≤ 7,5% thì dùng Dijkstra (nhanh), ngược lại gọi DM học với nhiễu (bền với nhiễu).
Sẽ thấy: sơ đồ khối của luật; một giờ mưa thật trên mạng 373 nút với các liên kết đang có mưa; so sánh trễ và tỉ lệ ai
định tuyến; bảng kiểm tra.
"""
import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from huong_dan.tien_ich import *  # noqa: E402,F403

TH = 0.075


def ve_luu_do():
    fig, ax = plt.subplots(figsize=(13, 3.2))
    ve_hop(ax, 0, 1, "Đo trễ mỗi liên kết 2 lần\nước lượng $\\hat\\sigma_0$", w=2.9, h=1.0)
    ve_hop(ax, 3.7, 1, "$\\hat\\sigma_0 \\leq 7{,}5\\%$ ?", mau="#fff4d6", w=2.2, h=1.0)
    ve_hop(ax, 7.2, 1, "Dijkstra tìm tuyến P", w=2.6, h=1.0)
    ve_hop(ax, 11.0, 1, "Mọi liên kết trên P:\n$\\sqrt{\\hat\\sigma_0^2+\\Delta_{ij}^2} \\leq 7{,}5\\%$ ?", mau="#fff4d6", w=3.2, h=1.0)
    ve_hop(ax, 14.4, 1, "Dùng P\n(nhanh)", mau="#eaf7f1", w=1.8, h=1.0)
    ve_hop(ax, 7.2, -0.7, "Gọi DM học với nhiễu", mau="#e8f0fb", w=3.0, h=0.9)
    ve_mui_ten(ax, (1.5, 1), (2.6, 1))
    ve_mui_ten(ax, (4.8, 1), (5.9, 1), "có")
    ve_mui_ten(ax, (8.5, 1), (9.4, 1))
    ve_mui_ten(ax, (12.6, 1), (13.5, 1), "có")
    ve_mui_ten(ax, (3.7, 0.5), (5.7, -0.6), "không", lech=(-0.6, -0.05))
    ve_mui_ten(ax, (11.0, 0.5), (8.7, -0.6), "không", lech=(0.6, -0.05))
    ax.text(7.2, -1.35, "Δ: nhiễu do mưa, tính từ lượng mưa thật (bước 8)", ha="center", fontsize=9, color="#52514e")
    ax.set_xlim(-1.6, 15.5)
    ax.set_ylim(-1.6, 1.7)
    ax.axis("off")


def chay():
    from src.ngau_nhien.kich_ban import MangGoc
    from src.ngau_nhien.thoi_tiet import ThoiTiet
    from test.luat_chia_doi import delta_matran, nhieu_lk, uoc_luong_sigma0
    from test.so_sanh_dijkstra_nhieu import dijkstra
    from huong_dan.buoc_4_quy_mo import do_ms
    ve_luu_do()
    tt = ThoiTiet(str(THOI_TIET))
    d = doc_demo(373)
    m = tai_mo_hinh("results/quy_mo/dm_nhieu.pt")
    t_d = do_ms(lambda k: dijkstra(d.W[d.sid[k]], d.paths[k][0], d.paths[k][-1], d.can_relay), range(0, 300, 15))
    t_dm = do_ms(lambda k: dinh_tuyen_dm(m, d, [k]), range(0, 300, 30))
    g = MangGoc(d.W, d.kind, d.node_order, tt)
    h = int(np.argmax((tt.precip[:, :13] > 2.5).sum(1)))           # giờ có nhiều điểm mưa vừa nhất
    ket, vi_du = {}, None
    for s0 in (0.02, 0.20):
        rng = np.random.default_rng(1)
        A, sh, D, mua = d.W.copy(), {}, {}, {}
        for s in range(len(d.W)):
            D[s], mua[s] = delta_matran(g, d, tt, s, h)
            S = np.minimum(np.sqrt(s0 ** 2 + D[s] ** 2), 0.8)
            A[s] = nhieu_lk(d.W[s], S, rng)
            sh[s] = uoc_luong_sigma0(A[s], nhieu_lk(d.W[s], S, rng), np.triu(d.W[s] > 0, 1) & ~mua[s])
        dn = copy.copy(d)
        dn.W = A
        q = np.arange(len(d))
        dm, okm = dinh_tuyen_dm(m, dn, q)
        dj = [dijkstra(A[d.sid[k]], d.paths[k][0], d.paths[k][-1], d.can_relay) for k in q]
        sang = np.array([not (sh[d.sid[k]] <= TH and np.sqrt(sh[d.sid[k]] ** 2 + D[d.sid[k]][p[:-1], p[1:]] ** 2).max() <= TH)
                         for k, p in zip(q, dj)])
        luat = [dm[j] if sang[j] else dj[j] for j in range(len(q))]

        def vuot(T):
            return np.mean([tre(d.W[d.sid[k]], t) / tre(d.W[d.sid[k]], d.paths[k]) - 1 for k, t in zip(q, T) if t]) * 100
        ket[s0] = {"Dijkstra": vuot(dj), "Luôn DM": vuot(dm), "Luật chia đôi": vuot(luat), "sang": sang.mean() * 100,
                   "ms": np.mean([(sh[d.sid[k]] <= TH) * t_d + sang[j] * t_dm for j, k in enumerate(q)]),
                   "sai_so": np.mean([abs(sh[s] - s0) for s in sh])}
        if s0 == 0.02:
            qua_mua = [j for j in range(len(q)) if sang[j] and mua[d.sid[q[j]]][dj[j][:-1], dj[j][1:]].any() and dm[j] != dj[j]]
            j = qua_mua[0] if qua_mua else int(np.argmax(sang))
            vi_du = (q[j], dj[j], luat[j], mua[d.sid[q[j]]])

    k, pd, pl, M = vi_du
    fig = plt.figure(figsize=(15, 4.6))
    ax = fig.add_axes([0.05, 0.2, 0.5, 0.7])
    iu, iv = np.nonzero(np.triu(M, 1))
    ve_mang(d, d.sid[k], tuyen=[(pd, MAU["cam"], "Dijkstra (đi qua mưa → bị loại)"), (pl, MAU["xanh"], "Luật chọn DM")],
            canh_to=[(list(zip(iu, iv)), "#7fb3e6", "Liên kết đang có mưa", "-")], ax=ax, ve_canh=False,
            tieu_de=f"Giờ có nhiều điểm mưa vừa nhất ({str(tt.time_utc[h])[:10]}, {str(tt.time_utc[h])[11:13]}h UTC), σ0 = 2%")
    ax2 = fig.add_axes([0.62, 0.2, 0.2, 0.68])
    x = np.arange(3)
    for dx, s0, mau in ((-0.18, 0.02, MAU["luc"]), (0.18, 0.20, "#7b4fa6")):
        ax2.bar(x + dx, [ket[s0][t] for t in ("Dijkstra", "Luôn DM", "Luật chia đôi")], 0.36, color=mau, label=f"σ0 = {s0:.0%}")
    ax2.set_xticks(x)
    ax2.set_xticklabels(["Dijkstra", "Luôn DM", "Luật"])
    ax2.set_ylabel("Trễ vượt tối ưu (%)")
    ax2.legend(frameon=False, fontsize=8)
    ax3 = fig.add_axes([0.84, 0.22, 0.15, 0.6])
    ax3.pie([100 - ket[0.02]["sang"], ket[0.02]["sang"]], labels=["Dijkstra", "DM"], colors=[MAU["cam"], MAU["xanh"]],
            autopct="%.0f%%", startangle=90, textprops={"fontsize": 9})
    ax3.set_title(f"σ0 = 2%: ai định tuyến\n({ket[0.02]['ms']:.1f} ms / truy vấn,\nluôn DM: {t_dm:.1f} ms)", fontsize=9)
    return [kiem_tra("Ước lượng σ0 bằng 2 lần đo", lambda: (max(ket[s]["sai_so"] for s in ket) < 0.01,
                     ", ".join(f"σ0 = {s:.0%}: sai {ket[s]['sai_so']:.4f}" for s in ket))),
            kiem_tra("Số đo tốt: luật không kém luôn DM", lambda: (ket[0.02]["Luật chia đôi"] <= ket[0.02]["Luôn DM"] + 0.05,
                     f"{ket[0.02]['Luật chia đôi']:.2f}% so với {ket[0.02]['Luôn DM']:.2f}%, gọi DM {ket[0.02]['sang']:.0f}%")),
            kiem_tra("Số đo kém: luật giao hết cho DM", lambda: (ket[0.20]["sang"] == 100, f"gọi DM {ket[0.20]['sang']:.0f}%")),
            kiem_tra("Số đo kém: DM tốt hơn Dijkstra", lambda: (ket[0.20]["Luôn DM"] < ket[0.20]["Dijkstra"],
                     f"{ket[0.20]['Luôn DM']:.2f}% so với {ket[0.20]['Dijkstra']:.2f}%"))]


if __name__ == "__main__":
    hien_thi(chay, "Bước 7 — Luật chia đôi")
