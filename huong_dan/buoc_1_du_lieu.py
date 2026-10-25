"""
Bước 1 — Dữ liệu SAGSIN của TV1
File dự án: dataset/oracle_labeler_sagsin.py (sinh dữ liệu), dataset/split_dataset.py (chia), src/ai/data_v2.py (đọc).
Dữ liệu: dataset/test.npz — 54 snapshot, 2.700 truy vấn, 73 nút (50 vệ tinh, 5 HAPS, 5 UAV, 5 gateway, 8 người dùng).
Sẽ thấy: mạng vẽ theo tầng với một đường tối ưu, độ trễ theo loại liên kết, số chặng của đường tối ưu, và bảng kiểm tra.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from huong_dan.tien_ich import *  # noqa: E402,F403


def chay():
    from src.ai.data_v2 import SagsinData
    from src.ngau_nhien.kich_ban import loai_lien_ket
    from test.so_sanh_dijkstra_nhieu import dijkstra
    d = SagsinData(GOC / "dataset" / "test.npz")
    i = next(k for k in range(len(d)) if len(d.paths[k]) >= 4)
    fig = plt.figure(figsize=(15, 4.8))
    ax = fig.add_axes([0.07, 0.2, 0.52, 0.72])
    ve_mang(d, d.sid[i], tuyen=[(d.paths[i], MAU["cam"], "Đường tối ưu (nhãn Dijkstra)")], ax=ax,
            tieu_de=f"Mạng 73 nút, snapshot {d.sid[i]}: {int((d.W[d.sid[i]] > 0).sum() // 2)} liên kết")
    W = d.W[d.sid[i]]
    theo_loai = {}
    for u, v in zip(*np.nonzero(np.triu(W > 0, 1))):
        theo_loai.setdefault(loai_lien_ket(str(d.kind[u]), str(d.kind[v]))[0], []).append(W[u, v])
    ten = sorted(theo_loai, key=lambda t: np.median(theo_loai[t]))
    ax2 = fig.add_axes([0.7, 0.2, 0.28, 0.72])
    ax2.boxplot([theo_loai[t] for t in ten], vert=False, widths=0.6)
    ax2.set_yticks(range(1, len(ten) + 1))
    ax2.set_yticklabels([f"{t} ({len(theo_loai[t])})" for t in ten])
    ax2.set_xlabel("Độ trễ một liên kết (ms)")
    ax2.set_title("Trễ theo loại liên kết (số liên kết)", loc="left", fontsize=11)

    fig2, axs = plt.subplots(1, 2, figsize=(12, 3.2))
    so_chang = np.array([len(p) - 1 for p in d.paths])
    axs[0].bar(*np.unique(so_chang, return_counts=True), color=MAU["xanh"])
    axs[0].set_xlabel("Số chặng của đường tối ưu")
    axs[0].set_ylabel("Số truy vấn")
    axs[0].set_title(f"{len(d.paths):,} truy vấn trong tập test".replace(",", "."), loc="left", fontsize=11)
    qua_vt = np.mean([any(str(d.layer[n]) == "space" for n in p) for p in d.paths]) * 100
    axs[1].pie([qua_vt, 100 - qua_vt], labels=["đi qua vệ tinh", "chỉ HAPS/UAV/mặt đất"],
               colors=[MAU["xam"], MAU["vang"]], autopct="%.0f%%", startangle=90)
    axs[1].set_title("Đường tối ưu đi qua tầng nào", fontsize=11)
    fig2.tight_layout()

    def t_nhan_la_toi_uu():
        q = range(0, len(d), 10)
        lech = sum(abs(tre(d.W[d.sid[k]], dijkstra(d.W[d.sid[k]], d.paths[k][0], d.paths[k][-1], d.can_relay))
                       - tre(d.W[d.sid[k]], d.paths[k])) > 1e-4 for k in q)
        return lech == 0, f"{len(q)} truy vấn, {lech} truy vấn lệch"

    return [
        kiem_tra("Ma trận trễ đối xứng", lambda: (bool(np.allclose(d.W, d.W.transpose(0, 2, 1))), "W[i,j] = W[j,i]")),
        kiem_tra("Nhãn đúng là đường Dijkstra", t_nhan_la_toi_uu),
        kiem_tra("Người dùng không làm trạm trung gian",
                 lambda: (sum(not d.can_relay[n] for p in d.paths for n in p[1:-1]) == 0, f"{len(d.paths)} đường")),
        kiem_tra("Mọi nhãn đi trên liên kết có thật",
                 lambda: (all(d.W[d.sid[k]][u, v] > 0 for k, p in enumerate(d.paths) for u, v in zip(p[:-1], p[1:])), "")),
    ]


if __name__ == "__main__":
    hien_thi(chay, "Bước 1 — Dữ liệu TV1")
