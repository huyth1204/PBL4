

from __future__ import annotations

import numpy as np

DELAY_SCALE_MS = 25.0


def normalize_state(x, n):
    x = np.asarray(x, dtype=np.float32)
    return np.clip(np.where(x < 0, 0.0, x / DELAY_SCALE_MS), 0.0, 1.0)


def _one_hot(i, n):
    v = np.zeros(n, dtype=np.float32)
    v[i] = 1.0
    return v


def load_items(npz_path, strict=True):
    """Đọc .npz -> (x_norm theo snapshot, danh sách item next-hop, N)."""
    d = np.load(npz_path, allow_pickle=True)
    N = len(d["node_order"])
    Xw, Y, sid = d["x_weights"], d["y_paths_indices"], d["snapshot_id"]

    # Trong một snapshot, trạng thái mạng x giống nhau -> lưu 1 lần cho đỡ tốn bộ nhớ.
    snap_rep = {}
    for row, s in enumerate(sid):
        snap_rep.setdefault(int(s), row)
    snap_ids = sorted(snap_rep)
    snap_pos = {s: k for k, s in enumerate(snap_ids)}
    x_norm = np.stack([normalize_state(Xw[snap_rep[s]], N) for s in snap_ids])  # (n_snap, N*N)
    W_by_snap = {s: Xw[snap_rep[s]].reshape(N, N) for s in snap_ids}

    items, bad = [], 0
    for row in range(len(Xw)):
        s = int(sid[row])
        W = W_by_snap[s]
        path = [int(p) for p in Y[row] if p >= 0]
        if len(path) < 2:
            continue
        dst = path[-1]
        for k in range(len(path) - 1):
            cur, nxt = path[k], path[k + 1]
            if W[cur, nxt] <= 0:  # nhãn không nằm trong cạnh có thật
                bad += 1
                if strict:
                    raise ValueError(f"Nhãn mâu thuẫn mặt nạ tại row {row}: {cur}->{nxt}")
                continue
            mask = (W[cur, :] > 0).astype(np.int8)
            items.append((snap_pos[s], cur, dst, nxt, mask, s))
    return x_norm, items, N, snap_ids, bad


try:
    from torch.utils.data import Dataset as _Dataset
except ImportError:
    _Dataset = object


class RoutingNpzDataset(_Dataset):
    def __init__(self, x_norm, items, n_nodes):
        self.x_norm = x_norm
        self.items = items
        self.N = n_nodes

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        import torch
        snap, cur, dst, nxt, mask, _ = self.items[i]
        cond = np.concatenate([self.x_norm[snap], _one_hot(cur, self.N), _one_hot(dst, self.N)])
        return {
            "condition": torch.from_numpy(cond.astype(np.float32)),
            "target": torch.from_numpy(_one_hot(nxt, self.N)),
            "mask": torch.from_numpy(mask.astype(np.float32)),
        }


def split_by_snapshot(x_norm, items, n_nodes, ratios=(0.7, 0.15, 0.15), seed=42):
    """Tách theo snapshot_id: cùng một ảnh chụp không lọt cả train lẫn test."""
    snaps = sorted({it[5] for it in items})
    rng = np.random.default_rng(seed)
    rng.shuffle(snaps)
    n_tr = int(len(snaps) * ratios[0])
    n_va = int(len(snaps) * ratios[1])
    tr, va = set(snaps[:n_tr]), set(snaps[n_tr:n_tr + n_va])
    buckets = {"train": [], "val": [], "test": []}
    for it in items:
        key = "train" if it[5] in tr else "val" if it[5] in va else "test"
        buckets[key].append(it)
    return {k: RoutingNpzDataset(x_norm, v, n_nodes) for k, v in buckets.items()}