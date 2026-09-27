"""
Bước 5 — Liên kết bị hỏng
File dự án: test/danh_gia_quy_mo.py (phần hỏng liên kết ngẫu nhiên). Mô hình: results/quy_mo/dm_da_quy_mo.pt.
Bỏ ngẫu nhiên 20% liên kết của một snapshot 173 nút rồi cho DM tìm đường trên mạng đã hỏng. DM chưa từng thấy mạng hỏng
khi học. Sau đó lặp lại với các mức hỏng 0–30% để xem độ bền.
Sẽ thấy: đường cũ bị cắt, tuyến DM tìm lại cạnh tối ưu mới; đồ thị trễ theo mức hỏng; bảng kiểm tra.
"""
import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from huong_dan.tien_ich import *  # noqa: E402,F403


def lam_hong(W, ti_le, rng):
    Wh = W.copy()
    iu, iv = np.nonzero(np.triu(Wh > 0, 1))
    bo = rng.random(len(iu)) < ti_le
    Wh[iu[bo], iv[bo]] = Wh[iv[bo], iu[bo]] = -1
    return Wh, set(zip(iu[bo], iv[bo])) | set(zip(iv[bo], iu[bo])), bo


def chay():
    from test.so_sanh_dijkstra_nhieu import dijkstra
    d = doc_demo(173)
    m = tai_mo_hinh("results/quy_mo/dm_da_quy_mo.pt")
    s = 0
    Wh, hong, bo = lam_hong(d.W[s], 0.2, np.random.default_rng(3))
    cand = [k for k in np.flatnonzero(d.sid == s)
            if any((u, v) in hong for u, v in zip(d.paths[k][:-1], d.paths[k][1:]))
            and dijkstra(Wh, d.paths[k][0], d.paths[k][-1], d.can_relay) is not None]
    k = cand[0]
    dh = copy.copy(d)
    dh.W = d.W.copy()
    dh.W[s] = Wh
    rt, ok = dinh_tuyen_dm(m, dh, [k])
    moi = dijkstra(Wh, d.paths[k][0], d.paths[k][-1], d.can_relay)
    cu_hong = [(u, v) for u, v in zip(d.paths[k][:-1], d.paths[k][1:]) if (u, v) in hong]
    fig, ax = plt.subplots(figsize=(12, 4.8))
    ve_mang(d, s, tuyen=[(moi, MAU["luc"], f"Tối ưu mới: {tre(Wh, moi):.2f} ms"),
                         (rt[0], MAU["xanh"], f"DM trên mạng hỏng: {tre(Wh, rt[0]):.2f} ms")],
            canh_to=[(list(zip(d.paths[k][:-1], d.paths[k][1:])), "#b9b8b3", "Đường cũ", "--"),
                     (cu_hong, "#d62728", "Liên kết cũ bị hỏng", "-")],
            ax=ax, ve_canh=False, tieu_de=f"Mạng 173 nút mất {bo.sum()} / {len(bo)} liên kết (trùng nhau thì chỉ thấy màu xanh)")
    fig.subplots_adjust(bottom=0.22)

    muc, vuot, trung = [0.0, 0.05, 0.1, 0.2, 0.3], [], []
    rng = np.random.default_rng(0)
    for tl in muc:
        dh = copy.copy(d)
        dh.W = d.W.copy()
        for s_ in range(len(d.W)):
            dh.W[s_] = lam_hong(d.W[s_], tl, rng)[0]
        q = [j for j in range(0, len(d), 2)
             if dijkstra(dh.W[d.sid[j]], d.paths[j][0], d.paths[j][-1], d.can_relay) is not None]
        rt2, ok2 = dinh_tuyen_dm(m, dh, q)
        tu = [dijkstra(dh.W[d.sid[j]], d.paths[j][0], d.paths[j][-1], d.can_relay) for j in q]
        vuot.append(np.mean([tre(dh.W[d.sid[j]], r) / tre(dh.W[d.sid[j]], t) - 1 for j, r, t, o in zip(q, rt2, tu, ok2) if o]) * 100)
        trung.append(np.mean([r == t for r, t in zip(rt2, tu)]) * 100)
    fig, axs = plt.subplots(1, 2, figsize=(12, 3.4))
    axs[0].plot([x * 100 for x in muc], trung, "o-", color=MAU["xanh"], lw=2)
    axs[0].set_title("DM trùng tối ưu mới (%)", loc="left", fontsize=11)
    axs[1].plot([x * 100 for x in muc], vuot, "o-", color=MAU["cam"], lw=2)
    axs[1].set_title("Trễ DM vượt tối ưu mới (%)", loc="left", fontsize=11)
    for a in axs:
        a.set_xlabel("Tỉ lệ liên kết bị hỏng (%)")
    fig.tight_layout()
    return [kiem_tra("DM tới đích trên mạng hỏng", lambda: (bool(ok[0]), "")),
            kiem_tra("Tuyến DM chỉ dùng liên kết còn sống", lambda: (all(Wh[u, v] > 0 for u, v in zip(rt[0][:-1], rt[0][1:])), "")),
            kiem_tra("Trễ DM so với tối ưu mới", lambda: (tre(Wh, rt[0]) <= 1.1 * tre(Wh, moi), f"{tre(Wh, rt[0]) / tre(Wh, moi):.3f} lần")),
            kiem_tra("Hỏng 30%: DM vẫn gần tối ưu mới", lambda: (vuot[-1] < 5, f"vượt {vuot[-1]:.2f}%, trùng {trung[-1]:.1f}%"))]


if __name__ == "__main__":
    hien_thi(chay, "Bước 5 — Hỏng liên kết")
