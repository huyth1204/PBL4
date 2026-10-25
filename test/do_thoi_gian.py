"""
Đo thời gian tìm đường: Dijkstra, tham lam, softmax, DM (nhiều số bước), GNN 0/4 lớp.

    python test/do_thoi_gian.py
    python test/do_thoi_gian.py --n-single 300

Hai chế độ:
  từng đường  mỗi yêu cầu (nguồn, đích) xử lý riêng — giống bộ định tuyến trả lời từng gói/luồng
  theo lô     xử lý cả tập test một lần — thông lượng khi có nhiều yêu cầu cùng lúc
"""
import argparse
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
from src.ai.data_v2 import SagsinData  # noqa: E402
from src.ai.train_gnn import build_model, encode_pair, greedy_chooser, model_chooser, rollout  # noqa: E402


def relay_dijkstra(W, src, dst, can_relay):
    """Giống single_source_relay_dijkstra: không đi ra từ nút không chuyển tiếp (trừ nguồn)."""
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
    return dist[dst]


def scipy_dijkstra(W, src, dst, can_relay):
    """Bản C (scipy): chặn cạnh đi ra từ nút không chuyển tiếp rồi chạy Dijkstra một nguồn."""
    A = np.where(W > 0, W, 0.0)
    block = ~can_relay.copy()
    block[src] = False
    A[block, :] = 0.0
    return sp_dijkstra(csr_matrix(A), indices=src)[dst]


def load(path):
    ck = torch.load(path)
    m = build_model(ck["args"], ck["f_in"])
    m.load_state_dict(ck["state"])
    return m.eval()


def route_fn(model, n_steps=None, guidance=1.0):
    def run(data, part):
        W = data.W[data.sid[part]]
        dst = np.array([data.paths[i][-1] for i in part])
        hc, hu = encode_pair(model, W, data.static_feat, dst)
        return rollout(data, part, model_chooser(model, hc, hu, guidance, n_steps))
    return run


def greedy_fn(data, part):
    return rollout(data, part, greedy_chooser)


def timed(fn, reps=1):
    t0 = time.perf_counter()
    for _ in range(reps):
        out = fn()
    return (time.perf_counter() - t0) / reps, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(ROOT / "dataset"))
    ap.add_argument("--out-dir", default=str(ROOT / "results"))
    ap.add_argument("--n-single", type=int, default=200)
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    out = Path(a.out_dir)
    te = SagsinData(Path(a.data_dir) / "test.npz")
    n_all = len(te)
    single = np.random.default_rng(0).choice(n_all, min(a.n_single, n_all), replace=False)

    methods = [("Dijkstra (Python, như oracle_labeler)", "dij_py", None),
               ("Dijkstra (scipy, viết bằng C)", "dij_c", None),
               ("Tham lam trễ thấp", "greedy", greedy_fn)]
    for tag, L in (("clsA_73", 0), ("clsB_73", 4)):
        if (out / f"{tag}.pt").exists():
            methods.append((f"Softmax, GNN {L} lớp", tag, route_fn(load(out / f"{tag}.pt"))))
    for tag, L in (("dmx0_0lop", 0), ("dmx0_4lop", 4)):
        if (out / f"{tag}.pt").exists():
            m = load(out / f"{tag}.pt")
            for n in (1, 2, 20):
                methods.append((f"DM {n} bước, GNN {L} lớp", f"{tag}_{n}b", route_fn(m, n)))

    res = {}
    print(f"TEST {n_all} yêu cầu, {te.N} nút | từng đường: {len(single)} yêu cầu mẫu\n")
    print(f"{'Phương pháp':40s} {'từng đường (ms)':>16s} {'theo lô (ms/đường)':>20s}")
    for name, key, fn in methods:
        if key.startswith("dij"):
            f = relay_dijkstra if key == "dij_py" else scipy_dijkstra
            t_single, _ = timed(lambda: [f(te.W[te.sid[i]], te.paths[i][0], te.paths[i][-1], te.can_relay)
                                         for i in single])
            t_single /= len(single)
            t_batch = t_single          # Dijkstra không gộp lô được: mỗi yêu cầu chạy riêng
        else:
            with torch.no_grad():
                fn(te, single[:5])      # khởi động
                t_single, _ = timed(lambda: [fn(te, np.array([i])) for i in single])
                t_single /= len(single)
                t_batch, _ = timed(lambda: [fn(te, np.arange(s, min(s + 400, n_all)))
                                            for s in range(0, n_all, 400)])
                t_batch /= n_all
        res[key] = {"ten": name, "ms_tung_duong": t_single * 1000, "ms_moi_duong_theo_lo": t_batch * 1000}
        print(f"{name:40s} {t_single * 1000:16.2f} {t_batch * 1000:20.3f}", flush=True)
    json.dump(res, open(out / "do_thoi_gian.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\nĐã lưu {out / 'do_thoi_gian.json'}")


if __name__ == "__main__":
    main()
