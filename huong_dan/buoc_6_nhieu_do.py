"""
Bước 6 — So với Dijkstra khi số đo có nhiễu
File dự án: test/so_sanh_dijkstra_nhieu.py. Mô hình: results/quy_mo/dm_nhieu.pt (học thêm với số đo nhiễu).
Trên dữ liệu sạch Dijkstra luôn đúng và nhanh hơn. Nhưng nếu bộ định tuyến chỉ thấy số đo W·(1 + σz) thì Dijkstra hay
chọn nhầm những liên kết bị đo thấp hơn thật. Tuyến luôn được chấm bằng trễ thật.
Sẽ thấy: một ví dụ thật ở σ = 40% (mạng 373 nút), đường trễ vượt tối ưu theo σ, bảng kiểm tra.
"""
import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from huong_dan.tien_ich import *  # noqa: E402,F403


def chay():
    from test.so_sanh_dijkstra_nhieu import co_ve_trung_binh, dijkstra, lam_nhieu, thong_ke_loai
    m = tai_mo_hinh("results/quy_mo/dm_nhieu.pt")
    d = doc_demo(373)
    vd = None
    for sd in range(1, 20):
        A = lam_nhieu(d.W, 0.4, sd)
        dn = copy.copy(d)
        dn.W = A
        q = np.arange(sd % 5, len(d), 5)
        rt, ok = dinh_tuyen_dm(m, dn, q)
        for k, r, o in zip(q, rt, ok):
            W = d.W[d.sid[k]]
            dj = dijkstra(A[d.sid[k]], d.paths[k][0], d.paths[k][-1], d.can_relay)
            if o and r == d.paths[k] and tre(W, dj) > 1.15 * tre(W, d.paths[k]):
                vd = (k, dj, r)
                break
        if vd:
            break
    k, dj, r = vd
    W = d.W[d.sid[k]]
    fig, ax = plt.subplots(figsize=(12, 4.8))
    ve_mang(d, d.sid[k], tuyen=[(dj, MAU["cam"], f"Dijkstra trên số đo: {tre(W, dj):.2f} ms thật"),
                                (r, MAU["xanh"], f"DM học với nhiễu = tối ưu: {tre(W, r):.2f} ms thật")],
            ax=ax, ve_canh=False, tieu_de="σ = 40%, mạng 373 nút: Dijkstra bị số đo đánh lừa")
    fig.subplots_adjust(bottom=0.22)

    loai, tk = thong_ke_loai(d)
    q = np.arange(0, len(d), 2)
    SG = [0.0, 0.1, 0.2, 0.4, 0.6]
    kq = {"Dijkstra trên số đo": [], "Dijkstra khử nhiễu (biết σ)": [], "DM học với nhiễu": []}
    for sg in SG:
        A = lam_nhieu(d.W, sg, 7)
        dn = copy.copy(d)
        dn.W = A
        K = co_ve_trung_binh(A, sg, loai, tk)
        rt, ok = dinh_tuyen_dm(m, dn, q)
        for ten, tuyen in (("Dijkstra trên số đo", [dijkstra(A[d.sid[j]], d.paths[j][0], d.paths[j][-1], d.can_relay) for j in q]),
                           ("Dijkstra khử nhiễu (biết σ)", [dijkstra(K[d.sid[j]], d.paths[j][0], d.paths[j][-1], d.can_relay) for j in q]),
                           ("DM học với nhiễu", rt)):
            kq[ten].append(np.mean([tre(d.W[d.sid[j]], t) / tre(d.W[d.sid[j]], d.paths[j]) - 1 for j, t in zip(q, tuyen) if t]) * 100)
    fig, ax = plt.subplots(figsize=(8, 3.8))
    for (ten, y), mau in zip(kq.items(), (MAU["cam"], MAU["luc"], MAU["xanh"])):
        y = np.array(y)
        y[y < 1e-6] = np.nan
        ax.plot([s * 100 for s in SG], y, "o-", color=mau, lw=2, label=ten)
    ax.set_yscale("log")
    ax.set_xlabel("σ nhiễu đo (%)")
    ax.set_ylabel("Trễ vượt tối ưu (%)")
    ax.set_title("Mạng 373 nút (demo, 150 truy vấn). Ở σ = 0 Dijkstra đúng tối ưu (0%) nên không có điểm", loc="left", fontsize=10)
    ax.legend(frameon=False)
    fig.tight_layout()
    return [kiem_tra("σ = 0: Dijkstra đúng tối ưu", lambda: (kq["Dijkstra trên số đo"][0] < 1e-6, "")),
            kiem_tra("σ = 40%: DM tốt hơn Dijkstra trên số đo", lambda: (kq["DM học với nhiễu"][3] < kq["Dijkstra trên số đo"][3],
                     f"{kq['DM học với nhiễu'][3]:.2f}% so với {kq['Dijkstra trên số đo'][3]:.2f}%")),
            kiem_tra("σ = 40%: DM tốt hơn cả Dijkstra khử nhiễu", lambda: (kq["DM học với nhiễu"][3] < kq["Dijkstra khử nhiễu (biết σ)"][3],
                     f"{kq['DM học với nhiễu'][3]:.2f}% so với {kq['Dijkstra khử nhiễu (biết σ)'][3]:.2f}%"))]


if __name__ == "__main__":
    hien_thi(chay, "Bước 6 — Số đo có nhiễu")
