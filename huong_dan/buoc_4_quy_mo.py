"""
Bước 4 — Bốn quy mô mạng: độ chính xác và thời gian
File dự án: test/danh_gia_quy_mo.py (đánh giá), src/ai/train_da_quy_mo.py (huấn luyện một mô hình cho mọi quy mô).
Mô hình: results/quy_mo/dm_da_quy_mo.pt. Dữ liệu: huong_dan/du_lieu_demo/n73…n373.npz (6 snapshot mỗi quy mô).
Mạng lớn dần bằng cách thêm vệ tinh: 73 → 173 → 273 → 373 nút; cùng một bộ trọng số chạy được trên mọi quy mô.
Sẽ thấy: bốn mạng cạnh nhau, độ chính xác theo số nút, thời gian DM và Dijkstra đo ngay trên máy bạn.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from huong_dan.tien_ich import *  # noqa: E402,F403

NS = [73, 173, 273, 373]


def do_ms(ham, cac_truy_van):
    """Thời gian một lần gọi (ms): chạy nháp một lần, rồi lấy trung vị."""
    cac_truy_van = list(cac_truy_van)
    ham(cac_truy_van[0])
    t = []
    for k in cac_truy_van:
        t0 = time.perf_counter()
        ham(k)
        t.append(time.perf_counter() - t0)
    return float(np.median(t) * 1000)


def do_quy_mo(m, n):
    from test.so_sanh_dijkstra_nhieu import dijkstra
    d = doc_demo(n)
    idx = np.arange(len(d))
    rt, ok = dinh_tuyen_dm(m, d, idx)
    tl = [tre(d.W[d.sid[k]], r) / tre(d.W[d.sid[k]], d.paths[k]) for k, r, o in zip(idx, rt, ok) if o]
    return d, rt, dict(toi=ok.mean() * 100, trung=np.mean([r == d.paths[k] for k, r in zip(idx, rt)]) * 100,
                       vuot=(np.mean(tl) - 1) * 100, canh=int((d.W[0] > 0).sum() // 2),
                       t_d=do_ms(lambda k: dijkstra(d.W[d.sid[k]], d.paths[k][0], d.paths[k][-1], d.can_relay),
                                 range(0, 300, 10)),
                       t_dm=do_ms(lambda k: dinh_tuyen_dm(m, d, [k]), range(0, 300, 20)))


def chay():
    m = tai_mo_hinh("results/quy_mo/dm_da_quy_mo.pt")
    bang = {}
    fig, axs = plt.subplots(1, 4, figsize=(16, 3.4), sharey=True)
    for ax, n in zip(axs, NS):
        d, rt, bang[n] = do_quy_mo(m, n)
        ve_mang(d, d.sid[5], tuyen=[(rt[5], MAU["xanh"], "DM")], ax=ax, ve_canh=False, ghi_ten=False,
                tieu_de=f"N = {n}: {bang[n]['canh']:,} liên kết".replace(",", "."))
        ax.get_legend().remove()
    fig.tight_layout()
    fig, axs = plt.subplots(1, 2, figsize=(12, 3.6))
    axs[0].plot(NS, [bang[n]["trung"] for n in NS], "o-", color=MAU["xanh"], lw=2, label="Trùng đường tối ưu (%)")
    axs[0].plot(NS, [bang[n]["toi"] for n in NS], "s-", color=MAU["luc"], lw=2, label="Tới đích (%)")
    axs[0].set_ylim(80, 101)
    axs[0].set_xticks(NS)
    axs[0].set_xlabel("Số nút")
    axs[0].legend(frameon=False)
    axs[0].set_title("Một mô hình cho mọi quy mô", loc="left", fontsize=11)
    axs[1].plot(NS, [bang[n]["t_dm"] for n in NS], "o-", color=MAU["xanh"], lw=2, label="DM, một đường")
    axs[1].plot(NS, [bang[n]["t_d"] for n in NS], "o-", color=MAU["cam"], lw=2, label="Dijkstra (Python)")
    axs[1].set_yscale("log")
    axs[1].set_xticks(NS)
    axs[1].set_xlabel("Số nút")
    axs[1].set_ylabel("ms mỗi đường")
    axs[1].legend(frameon=False)
    axs[1].set_title("Thời gian đo trên máy bạn", loc="left", fontsize=11)
    fig.tight_layout()
    return [kiem_tra(f"N = {n}", lambda n=n: (bang[n]["toi"] >= 99 and bang[n]["vuot"] < 1,
                     f"tới đích {bang[n]['toi']:.1f}%, trùng {bang[n]['trung']:.1f}%, vượt {bang[n]['vuot']:.2f}%, "
                     f"DM {bang[n]['t_dm']:.1f} ms / Dijkstra {bang[n]['t_d']:.2f} ms")) for n in NS]


if __name__ == "__main__":
    hien_thi(chay, "Bước 4 — Bốn quy mô mạng")
