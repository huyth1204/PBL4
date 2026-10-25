"""
Bước 3 — Mô hình GNN + DM định tuyến
File dự án: src/ai/gnn_diffusion_v2.py (mô hình), src/ai/train_gnn.py (huấn luyện; hàm rollout đi từng bước).
Mô hình: results/dmx0_4lop.pt (4 lớp GNN, 107.330 tham số, học trên mạng 73 nút). Dữ liệu: dataset/test.npz.
Cách DM chọn đường: tại nút hiện tại, GNN mã hóa cả mạng; DM bắt đầu từ điểm số ngẫu nhiên cho các ứng viên next-hop
rồi khử nhiễu dần thành điểm số rõ ràng; chọn ứng viên điểm cao nhất, sang nút đó, lặp lại tới đích.
Sẽ thấy: tuyến DM cạnh tuyến tối ưu, quá trình khử nhiễu, phân bố trễ của DM trên cả tập test, bảng kiểm tra.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from huong_dan.tien_ich import *  # noqa: E402,F403


def chay():
    from src.ai.data_v2 import SagsinData, valid_mask
    d = SagsinData(GOC / "dataset" / "test.npz")
    m = tai_mo_hinh("results/dmx0_4lop.pt")

    def so_ung_vien(k):
        p, W = d.paths[k], d.W[d.sid[k]]
        v = np.zeros((1, d.N), bool)
        v[0, p[0]] = True
        return int(valid_mask(W[p[0]][None], np.array([p[-1]]), d.can_relay, v).sum())

    i = max((k for k in range(0, len(d), 7) if len(d.paths[k]) >= 4), key=so_ung_vien)
    rt, ok = dinh_tuyen_dm(m, d, [i])
    W = d.W[d.sid[i]]
    fig, ax = plt.subplots(figsize=(12, 4.8))
    ve_mang(d, d.sid[i], tuyen=[(d.paths[i], MAU["cam"], f"Tối ưu (Dijkstra): {tre(W, d.paths[i]):.2f} ms"),
                                (rt[0], MAU["xanh"], f"DM: {tre(W, rt[0]):.2f} ms")], ax=ax,
            tieu_de="Tuyến DM chọn so với tuyến tối ưu (trùng nhau thì chỉ thấy màu xanh)")
    fig.subplots_adjust(bottom=0.22)

    ung, lich, p = diem_ung_vien(m, d, i, n_steps=6)
    Y = lich[:, ung]
    fig, axs = plt.subplots(1, 2, figsize=(13, 3.8), gridspec_kw={"width_ratios": [1.6, 1]})
    v = np.abs(Y).max()
    im = axs[0].imshow(Y, aspect="auto", cmap="RdBu_r", vmin=-v, vmax=v)
    nhan = [ten_ngan(d.node_order[u]) for u in ung]
    axs[0].set_xticks(range(len(ung)))
    axs[0].set_xticklabels(nhan, rotation=45, fontsize=8, ha="right")
    axs[0].set_yticks(range(len(Y)))
    axs[0].set_yticklabels(["nhiễu ban đầu"] + [f"sau bước {k}" for k in range(1, len(Y))])
    axs[0].set_title(f"Khử nhiễu dần: điểm từng ứng viên next-hop từ {d.node_order[p[0]]} (đỏ = cao)", loc="left", fontsize=11)
    plt.colorbar(im, ax=axs[0], fraction=0.03)
    k = list(ung).index(p[1])
    axs[0].add_patch(plt.Rectangle((k - 0.5, -0.5), 1, len(Y), fill=False, ec=MAU["cam"], lw=2.5))
    xs = np.exp(Y[-1] - Y[-1].max())
    xs /= xs.sum()
    axs[1].bar(range(len(ung)), xs, color=[MAU["cam"] if u == p[1] else MAU["xanh"] for u in ung])
    axs[1].set_xticks(range(len(ung)))
    axs[1].set_xticklabels(nhan, rotation=45, fontsize=8, ha="right")
    axs[1].set_title("Xác suất chọn ở bước cuối (cam = next-hop tối ưu)", loc="left", fontsize=11)
    fig.tight_layout()

    idx = np.arange(len(d))
    rt, ok = dinh_tuyen_dm(m, d, idx)
    tl = np.array([tre(d.W[d.sid[k]], r) / tre(d.W[d.sid[k]], d.paths[k]) for k, r, o in zip(idx, rt, ok) if o])
    trung = np.mean([r == d.paths[k] for k, r in zip(idx, rt)]) * 100
    fig, ax = plt.subplots(figsize=(10, 3.2))
    vuot = (tl - 1) * 100
    ax.hist(vuot[vuot > 1e-6], bins=40, color=MAU["xanh"])
    ax.set_xlabel("Trễ vượt tối ưu (%) — chỉ vẽ các truy vấn DM không trùng tối ưu")
    ax.set_ylabel("Số truy vấn")
    ax.set_title(f"Cả tập test: {len(idx)} truy vấn, DM trùng tối ưu {trung:.1f}%, tới đích {ok.mean() * 100:.1f}%",
                 loc="left", fontsize=11)
    fig.tight_layout()
    return [kiem_tra("DM tới đích", lambda: (ok.mean() > 0.99, f"{ok.mean() * 100:.1f}% của {len(idx)} truy vấn")),
            kiem_tra("DM trùng đường tối ưu", lambda: (trung > 90, f"{trung:.1f}%")),
            kiem_tra("Trễ DM gần tối ưu", lambda: ((tl.mean() - 1) * 100 < 1, f"vượt tối ưu trung bình {(tl.mean() - 1) * 100:.2f}%")),
            kiem_tra("Khử nhiễu dồn về next-hop tối ưu", lambda: (int(np.argmax(Y[-1])) == k, f"{d.node_order[p[1]]}"))]


if __name__ == "__main__":
    hien_thi(chay, "Bước 3 — DM định tuyến")
