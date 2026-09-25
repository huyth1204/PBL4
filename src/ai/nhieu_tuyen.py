"""
nhieu_tuyen.py — Đánh giá khả năng SINH NHIỀU TUYẾN của mô hình (đường dự phòng).

Với mỗi cặp nguồn-đích, sinh K tuyến (DM: K hạt nhiễu khác nhau; softmax: lấy mẫu có nhiệt độ T) và đo:
  so_tuyen_khac_nhau   số tuyến khác nhau tới được đích trong K mẫu
  best_of_K            trễ của tuyến tốt nhất trong K mẫu, so với Dijkstra
  du_phong             giả sử MỘT liên kết của tuyến chính bị hỏng: trong K mẫu có sẵn tuyến tránh được
                       liên kết đó không (chỉ xét khi đồ thị còn đường khác), và trễ của nó so với
                       Dijkstra chạy lại trên đồ thị đã bỏ liên kết hỏng.
"""

from __future__ import annotations

import heapq

import numpy as np
import torch

from src.ai.train_gnn import encode_pair, rollout


def relay_dijkstra(W, src, dst, can_relay, banned=None, banned_nodes=None):
    """Dijkstra theo trễ, giữ quy tắc chuyển tiếp; banned = cạnh có hướng (a, b) cấm đi; banned_nodes = nút cấm."""
    N = len(W)
    dist, prev, pq = np.full(N, np.inf), -np.ones(N, int), [(0.0, src)]
    dist[src] = 0.0
    while pq:
        d, u = heapq.heappop(pq)
        if u == dst:
            break
        if d > dist[u] or (u != src and not can_relay[u]):
            continue
        for v in np.flatnonzero(W[u] > 0):
            if banned is not None and (u, v) in banned:
                continue
            if banned_nodes is not None and v in banned_nodes:
                continue
            nd = d + W[u, v]
            if nd < dist[v]:
                dist[v], prev[v] = nd, u
                heapq.heappush(pq, (nd, v))
    if not np.isfinite(dist[dst]):
        return None
    p, v = [], dst
    while v != -1:
        p.append(int(v))
        v = prev[v]
    return p[::-1]


def route_delay(W, r):
    return float(sum(W[a, b] for a, b in zip(r[:-1], r[1:])))


@torch.no_grad()
def sample_routes(model, data, idx, K, decide_kw, chunk=300, seed0=1000):
    """K lần sinh cả tuyến cho mỗi cặp; lần k dùng hạt nhiễu seed0 + k. None = không tới đích."""
    out = [[None] * K for _ in idx]
    for s in range(0, len(idx), chunk):
        part = idx[s:s + chunk]
        W = data.W[data.sid[part]]
        dst = np.array([data.paths[i][-1] for i in part])
        hc, hu = encode_pair(model, W, data.static_feat, dst)
        for k in range(K):
            torch.manual_seed(seed0 + k)

            def choose(act, cur, dst_, rows, mask):
                a = torch.from_numpy(act)
                return model.decide(hc[a], hu[a], torch.from_numpy(cur), torch.from_numpy(dst_),
                                    torch.from_numpy(rows), torch.from_numpy(mask), **decide_kw).numpy()
            routes, ok, _ = rollout(data, part, choose)
            for j in range(len(part)):
                out[s + j][k] = tuple(routes[j]) if ok[j] else None
    return out


def evaluate_samples(data, idx, primary, samples):
    """primary: tuyến chính mỗi cặp (tuple|None); samples: K tuyến mỗi cặp."""
    n_dist, best, cover, bratio, n_fail_case = [], [], [], [], 0
    for i, prim, rs in zip(idx, primary, samples):
        W = data.W[data.sid[i]]
        opt = route_delay(W, data.paths[i])
        good = [r for r in rs if r is not None]
        if prim is not None:
            good = good + [prim]
        uniq = set(good)
        n_dist.append(len(uniq))
        if uniq:
            best.append(min(route_delay(W, r) for r in uniq) / opt)
        if prim is None:
            continue
        for a, b in zip(prim[:-1], prim[1:]):
            banned = {(a, b), (b, a)}
            alt = relay_dijkstra(W, prim[0], prim[-1], data.can_relay, banned)
            if alt is None:
                continue                                   # đồ thị không còn đường nào khác
            n_fail_case += 1
            ok = [r for r in uniq if not any((x, y) in banned for x, y in zip(r[:-1], r[1:]))]
            cover.append(bool(ok))
            if ok:
                bratio.append(min(route_delay(W, r) for r in ok) / route_delay(W, alt))
    n_dist = np.array(n_dist)
    return {
        "so_tuyen_khac_nhau_TB": float(n_dist.mean()),
        "ty_le_cap_co_tu_2_tuyen": float((n_dist >= 2).mean()),
        "tre_best_of_K_so_Dijkstra": float(np.mean(best)) if best else float("nan"),
        "so_tinh_huong_hong_lien_ket": n_fail_case,
        "co_san_du_phong": float(np.mean(cover)) if cover else float("nan"),
        "tre_du_phong_so_Dijkstra_chay_lai": float(np.mean(bratio)) if bratio else float("nan"),
    }


def yen_near_optimal(W, src, dst, can_relay, k=4, eps=0.10):
    """Tối đa k tuyến đơn ngắn nhất (thuật toán Yen) có trễ <= (1 + eps) x trễ tối ưu, giữ quy tắc chuyển tiếp."""
    first = relay_dijkstra(W, src, dst, can_relay)
    if first is None:
        return []
    A, B, seen = [first], [], {tuple(first)}
    limit = route_delay(W, first) * (1 + eps) + 1e-9
    while len(A) < k:
        last = A[-1]
        for i in range(len(last) - 1):
            spur, root = last[i], last[:i + 1]
            if i > 0 and not can_relay[spur]:
                continue
            banned = {(p[i], p[i + 1]) for p in A if len(p) > i + 1 and p[:i + 1] == root}
            sp = relay_dijkstra(W, spur, dst, can_relay, banned, set(root[:-1]))
            if sp is None:
                continue
            cand = root[:-1] + sp
            if tuple(cand) not in seen:
                seen.add(tuple(cand))
                heapq.heappush(B, (route_delay(W, cand), cand))
        if not B:
            break
        d, p = heapq.heappop(B)
        if d > limit:
            break
        A.append(p)
    return A
