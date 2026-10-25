"""
Đánh giá theo quy mô mạng (73, 173, 273, 373 nút): độ chính xác, thời gian, biến động liên kết, độ bền của tuyến
khi mạng thay đổi theo thời gian và khi một phần liên kết bị hỏng.

    python test/danh_gia_quy_mo.py --raw data/quy_mo/n173.npz --data-dir data/quy_mo/n173 \
        --mo-hinh "DM (huấn luyện 173)=results/quy_mo/dm_173.pt:2:1" "DM (chuyển từ 73)=results/dmx0_4lop.pt:2:1" \
        --out results/quy_mo/danh_gia_173.json
Mỗi mô hình: "tên=đường_dẫn.pt:số_bước_khử_nhiễu:guidance".
"""
import argparse
import copy
import heapq
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra as sp_dijkstra

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.ai.data_v2 import SagsinData, valid_mask  # noqa: E402
from src.ai.train_gnn import build_model, encode_pair, eval_nexthop, model_chooser, rollout, route_delay  # noqa: E402


def relay_dijkstra(W, src, dst, can_relay):
    """Dijkstra có quy tắc chuyển tiếp (như oracle_labeler). Trả đường đi hoặc None."""
    N = len(W)
    dist, prev, pq = np.full(N, np.inf), -np.ones(N, int), [(0.0, src)]
    dist[src] = 0
    while pq:
        d, u = heapq.heappop(pq)
        if u == dst:
            break
        if d > dist[u] or (u != src and not can_relay[u]):
            continue
        for v in np.flatnonzero(W[u] > 0):
            if d + W[u, v] < dist[v]:
                dist[v], prev[v] = d + W[u, v], u
                heapq.heappush(pq, (dist[v], v))
    if not np.isfinite(dist[dst]):
        return None
    p, v = [], dst
    while v != -1:
        p.append(int(v))
        v = prev[v]
    return p[::-1]


def scipy_dijkstra(W, src, dst, can_relay):
    A = np.where(W > 0, W, 0.0)
    block = ~can_relay.copy()
    block[src] = False
    A[block, :] = 0.0
    return sp_dijkstra(csr_matrix(A), indices=src)[dst]


def load(path):
    ck = torch.load(path, weights_only=False)
    m = build_model(ck["args"], ck["f_in"])
    m.load_state_dict(ck["state"])
    return m.eval()


def chay_mo_hinh(m, data, idx, w, n_steps, chunk):
    routes, oks = [], []
    torch.manual_seed(0)
    with torch.no_grad():
        for s in range(0, len(idx), chunk):
            part = idx[s:s + chunk]
            W = data.W[data.sid[part]]
            dst = np.array([data.paths[i][-1] for i in part])
            hc, hu = encode_pair(m, W, data.static_feat, dst)
            rt, ok, _ = rollout(data, part, model_chooser(m, hc, hu, w, n_steps))
            routes += rt
            oks += list(ok)
    return routes, np.array(oks)


def chat_luong(data, idx, routes, oks, ref=None):
    """So với đường tham chiếu (mặc định nhãn Dijkstra): tới đích, tỉ lệ trễ, trùng đường."""
    ref = ref or [data.paths[i] for i in idx]
    rat, trung = [], []
    for k, i in enumerate(idx):
        W = data.W[data.sid[i]]
        if oks[k]:
            rat.append(route_delay(W, routes[k]) / max(route_delay(W, ref[k]), 1e-9))
        trung.append(bool(oks[k]) and list(routes[k]) == list(ref[k]))
    rat = np.array(rat)
    return {"toi_dich_%": float(np.mean(oks) * 100), "tre_so_voi_toi_uu": float(rat.mean()) if len(rat) else None,
            "trong_10pct_%": float(np.mean(rat <= 1.1) * 100) if len(rat) else None,
            "trung_duong_%": float(np.mean(trung) * 100)}


def song_sau(Wraw, t, r, D):
    """Tuyến r (lập tại snapshot t) còn nguyên vẹn ở t+D không."""
    if t + D >= len(Wraw):
        return None
    a, b = np.array(r[:-1]), np.array(r[1:])
    return bool((Wraw[t + D][a, b] > 0).all())


def thong_ke_mang(Wraw, te):
    A = Wraw > 0
    T = len(A)
    so_canh = A.reshape(T, -1).sum(1)
    doi = [(A[t] ^ A[t + 1]).sum() / max(A[t].sum(), 1) for t in range(T - 1)]
    # thời gian sống của liên kết: độ dài các đoạn có mặt liên tiếp
    dai = []
    X = A.reshape(T, -1)[:, A.reshape(T, -1).any(0)]
    for j in range(X.shape[1]):
        x = np.concatenate([[0], X[:, j].astype(int), [0]])
        d = np.diff(x)
        dai += list(np.flatnonzero(d == -1) - np.flatnonzero(d == 1))
    dai = np.array(dai)
    kind = np.array([str(k) for k in te.kind])
    buoc = [(p[k], p[-1]) for p in te.paths for k in range(len(p) - 1)]
    rows = np.array([te.W[te.sid[i]][p[k]] for i, p in enumerate(te.paths) for k in range(len(p) - 1)])
    mask = valid_mask(rows, np.array([d for _, d in buoc]), te.can_relay)
    return {"N": int(te.N), "so_ve_tinh": int((kind == "leo").sum()),
            "canh_co_huong_TB": float(so_canh.mean()), "bac_ra_TB": float(so_canh.mean() / te.N),
            "ung_vien_moi_buoc_TB": float(mask.sum(1).mean()), "ung_vien_moi_buoc_max": int(mask.sum(1).max()),
            "so_hop_TB": float(np.mean([len(p) - 1 for p in te.paths])),
            "duong_qua_ve_tinh_%": float(np.mean([any(kind[n] == "leo" for n in p) for p in te.paths]) * 100),
            "lien_ket_doi_moi_buoc_%": float(np.mean(doi) * 100),
            "tuoi_tho_lien_ket_TB_s": float(dai.mean() * 20), "tuoi_tho_lien_ket_trung_vi_s": float(np.median(dai) * 20)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, help="tệp 600 snapshot liên tục của TV1")
    ap.add_argument("--data-dir", required=True, help="thư mục có train/val/test.npz và split_info.json")
    ap.add_argument("--mo-hinh", nargs="+", default=[])
    ap.add_argument("--n-single", type=int, default=150)
    ap.add_argument("--n-hong", type=int, default=800, help="số đường test dùng cho thử nghiệm hỏng liên kết")
    ap.add_argument("--muc-hong", default="0.05,0.1,0.2,0.3")
    ap.add_argument("--chunk", type=int, default=200)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--khong-do-thoi-gian", action="store_true",
                    help="bỏ phần đo thời gian (khi máy đang bận); đo riêng bằng test/do_thoi_gian_quy_mo.py")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    dd = Path(a.data_dir)
    te = SagsinData(dd / "test.npz")
    Wraw = np.load(a.raw)["W_snapshots"]
    ids = sorted(json.load(open(dd / "split_info.json", encoding="utf-8"))["snapshot_ids"]["test"])
    t_goc = np.array([ids[s] for s in te.sid])                      # snapshot gốc của từng đường test
    idx = np.arange(len(te))
    rng = np.random.default_rng(0)
    single = rng.choice(len(te), min(a.n_single, len(te)), replace=False)
    kq = {"mang": thong_ke_mang(Wraw, te), "so_duong_test": int(len(te)), "phuong_phap": {}}
    print(json.dumps(kq["mang"], ensure_ascii=False), flush=True)

    # Dijkstra: thời gian và độ bền theo thời gian của đường tối ưu
    if a.khong_do_thoi_gian:
        single = single[:0]
    t0 = time.perf_counter()
    for i in single:
        relay_dijkstra(te.W[te.sid[i]], te.paths[i][0], te.paths[i][-1], te.can_relay)
    t_py = (time.perf_counter() - t0) / max(len(single), 1) * 1000
    t0 = time.perf_counter()
    for i in single:
        scipy_dijkstra(te.W[te.sid[i]], te.paths[i][0], te.paths[i][-1], te.can_relay)
    t_c = (time.perf_counter() - t0) / max(len(single), 1) * 1000
    ben = {D: np.mean([x for x in (song_sau(Wraw, t_goc[i], te.paths[i], D) for i in idx) if x is not None]) * 100
           for D in (1, 3, 9, 45)}
    kq["phuong_phap"]["Dijkstra"] = {"ms_tung_duong_python": t_py, "ms_tung_duong_scipy_C": t_c,
                                     "con_song_sau_%": {str(D): float(v) for D, v in ben.items()}}
    print("Dijkstra", kq["phuong_phap"]["Dijkstra"], flush=True)

    # tập con cho thử nghiệm hỏng liên kết: cùng các snapshot, bỏ ngẫu nhiên một phần liên kết (cả hai chiều)
    con = np.sort(rng.choice(len(te), min(a.n_hong, len(te)), replace=False))
    muc = [float(x) for x in a.muc_hong.split(",")]
    hong = {}
    for f in muc:
        d = copy.copy(te)
        W2 = te.W.copy()
        r2 = np.random.default_rng(int(f * 1000))
        for s in range(len(W2)):
            iu = np.argwhere(np.triu(W2[s] > 0, 1))
            bo = iu[r2.random(len(iu)) < f]
            W2[s][bo[:, 0], bo[:, 1]] = -1
            W2[s][bo[:, 1], bo[:, 0]] = -1
        d.W = W2
        toi_uu = [relay_dijkstra(W2[te.sid[i]], te.paths[i][0], te.paths[i][-1], te.can_relay) for i in con]
        con_duong = np.array([p is not None for p in toi_uu])
        cu_con_dung = np.mean([all(W2[te.sid[i]][u, v] > 0 for u, v in zip(te.paths[i][:-1], te.paths[i][1:]))
                               for i in con[con_duong]]) * 100
        hong[f] = (d, con[con_duong], [p for p in toi_uu if p is not None], float(cu_con_dung))
        print(f"hỏng {f:.0%}: còn đường {con_duong.mean() * 100:.1f}%, đường tối ưu cũ còn dùng được {cu_con_dung:.1f}%",
              flush=True)
    kq["hong_lien_ket"] = {str(f): {"duong_cu_con_dung_%": h[3], "so_duong": int(len(h[1]))} for f, h in hong.items()}

    for spec in a.mo_hinh:
        ten, rest = spec.split("=", 1)
        path, n_steps, w = rest.rsplit(":", 2)
        n_steps, w = int(n_steps), float(w)
        m = load(path)
        r = {"so_tham_so": int(sum(p.numel() for p in m.parameters()))}
        acc, rand, greedy = eval_nexthop(m, te, (w,), n_steps=n_steps, bs=64)
        r["next_hop_%"] = acc[w] * 100
        r["next_hop_ngau_nhien_%"] = rand * 100
        r["next_hop_tham_lam_%"] = greedy * 100
        t0 = time.perf_counter()
        routes, oks = chay_mo_hinh(m, te, idx, w, n_steps, a.chunk)
        r["ms_moi_duong_theo_lo"] = (time.perf_counter() - t0) / len(idx) * 1000
        r.update(chat_luong(te, idx, routes, oks))
        t0 = time.perf_counter()
        for i in single:
            chay_mo_hinh(m, te, np.array([i]), w, n_steps, 1)
        r["ms_tung_duong"] = (time.perf_counter() - t0) / max(len(single), 1) * 1000
        ben = {D: [song_sau(Wraw, t_goc[i], routes[i], D) for i in idx if oks[i]] for D in (1, 3, 9, 45)}
        r["con_song_sau_%"] = {str(D): float(np.mean([x for x in v if x is not None]) * 100) for D, v in ben.items()}
        r["hong_lien_ket"] = {}
        for f, (d, sub, toi_uu, _) in hong.items():
            rt, ok = chay_mo_hinh(m, d, sub, w, n_steps, a.chunk)
            r["hong_lien_ket"][str(f)] = chat_luong(d, sub, rt, ok, toi_uu)
        kq["phuong_phap"][ten] = r
        print(ten, json.dumps(r, ensure_ascii=False), flush=True)
    json.dump(kq, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("Lưu", a.out)


if __name__ == "__main__":
    main()
