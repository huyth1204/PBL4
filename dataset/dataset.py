"""
dataset.py — RoutingDataset cho mô hình khuếch tán.

Đọc nhãn Dijkstra (dataset/sample_labels.json), cắt mỗi đường đi thành các
mẫu NEXT-HOP kèm mặt nạ hợp lệ, trả về tensor sẵn cho huấn luyện.

Mỗi mẫu ra: condition = [x_chuẩn_hoá || one-hot(nguồn) || one-hot(đích)],
            target    = one-hot(next-hop),
            mask      = các nút kề hợp lệ tại nút hiện tại.
"""

from __future__ import annotations

import json
from hashlib import md5
from pathlib import Path

import numpy as np

DELAY_SCALE_MS = 25.0  # trần co độ trễ về [0,1]

# torch chỉ cần cho RoutingDataset; nạp muộn để phần lõi chạy được không cần torch.


# ---- lõi thuần numpy (test được không cần torch) ---------------------------

def normalize_state(x, n_nodes):
    # -1 (không nối) -> 0; độ trễ -> chia scale, kẹp [0,1].
    # Cân bằng thang đo để không chiều nào lấn gradient.
    x = np.asarray(x, dtype=np.float32)
    if x.size != n_nodes * n_nodes:
        raise ValueError(
            f"len(x)={x.size} nhưng N*N={n_nodes*n_nodes}. Sai N: phải dùng "
            f"node_order đầy đủ của đồ thị, không dùng số nút trong đường đi."
        )
    return np.clip(np.where(x < 0, 0.0, x / DELAY_SCALE_MS), 0.0, 1.0)


def mask_row(W, cur_idx):
    # Mặt nạ hợp lệ = out-neighbors (trọng số > 0). Đồ thị có hướng nên dùng HÀNG.
    return (W[cur_idx, :] > 0).astype(np.int64)


def cut_nexthop(sample):
    """
    QUAN TRỌNG: mô hình học từng next-hop, không phải cả đường đi.
    Cắt một mẫu {x, source, target, y_star, node_order} thành list mẫu next-hop.
    node_order là BẮT BUỘC — nếu thiếu thì không dựng được mặt nạ (sửa oracle_labeler).
    """
    if "node_order" not in sample:
        raise KeyError(
            "Mẫu thiếu 'node_order'. Sửa oracle_labeler lưu node_order=list(G.nodes())."
        )
    order = sample["node_order"]
    n = len(order)
    idx = {name: i for i, name in enumerate(order)}
    W = np.asarray(sample["x"], dtype=np.float32).reshape(n, n)
    x_norm = normalize_state(sample["x"], n).ravel()
    dst = idx[sample["target"]]

    out = []
    path = sample["y_star"]
    for k in range(len(path) - 1):
        cur, nxt = idx[path[k]], idx[path[k + 1]]
        m = mask_row(W, cur)
        if m[nxt] != 1:  # nhãn hợp lệ: Dijkstra chỉ đi cạnh có thật
            raise ValueError(
                f"Nhãn mâu thuẫn tại {path[k]}->{path[k+1]}: x nói không có cạnh này."
            )
        out.append({"x_norm": x_norm, "cur": cur, "dst": dst, "next": nxt, "mask": m, "n": n})
    return out


def one_hot(i, n):
    v = np.zeros(n, dtype=np.float32)
    v[i] = 1.0
    return v


def snapshot_id(sample):
    # Định danh một ảnh chụp mạng = hash của x. Dùng để tách train/val/test
    # THEO SNAPSHOT, tránh rò rỉ (cùng trạng thái mạng lọt cả train lẫn test).
    return md5(np.asarray(sample["x"], dtype=np.float32).tobytes()).hexdigest()


def split_by_snapshot(samples, ratios=(0.7, 0.15, 0.15), seed=42):
    ids = sorted({snapshot_id(s) for s in samples})
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    n_tr = int(len(ids) * ratios[0])
    n_va = int(len(ids) * ratios[1])
    tr, va = set(ids[:n_tr]), set(ids[n_tr:n_tr + n_va])
    buckets = {"train": [], "val": [], "test": []}
    for s in samples:
        sid = snapshot_id(s)
        key = "train" if sid in tr else "val" if sid in va else "test"
        buckets[key].append(s)
    return buckets


# ---- torch Dataset (bọc mỏng quanh lõi trên) -------------------------------

def _torch():
    import torch
    return torch


try:
    from torch.utils.data import Dataset as _Dataset
except ImportError:
    _Dataset = object  # cho phép import module khi chưa có torch


class RoutingDataset(_Dataset):
    def __init__(self, samples_or_path, n_nodes=None):
        if isinstance(samples_or_path, (str, Path)):
            with open(samples_or_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
        else:
            raw = samples_or_path
        if not raw:
            raise ValueError("Danh sách mẫu rỗng.")

        # Cắt tất cả đường đi thành mẫu next-hop
        self.items = []
        for j, s in enumerate(raw):
            try:
                self.items.extend(cut_nexthop(s))
            except (KeyError, ValueError) as e:
                raise type(e)(f"Mẫu thô #{j}: {e}") from e

        # N phải cố định để gom batch được — buộc oracle_labeler dùng roster nút cố định
        Ns = {it["n"] for it in self.items}
        if n_nodes is not None:
            Ns.add(n_nodes)
        if len(Ns) != 1:
            raise ValueError(
                f"Số nút N không đồng nhất giữa các mẫu: {sorted(Ns)}. Mọi ảnh chụp "
                f"phải dùng CÙNG một roster nút cố định (sửa ở oracle_labeler)."
            )
        self.n_nodes = Ns.pop()

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        torch = _torch()
        it = self.items[i]
        n = it["n"]
        cond = np.concatenate([it["x_norm"], one_hot(it["cur"], n), one_hot(it["dst"], n)])
        return {
            "condition": torch.from_numpy(cond.astype(np.float32)),
            "target": torch.from_numpy(one_hot(it["next"], n)),
            "mask": torch.from_numpy(it["mask"].astype(np.float32)),
        }


def apply_validity_mask(logits, mask, neg_inf=-1e9):
    # Áp trước softmax/argmax khi SINH: nút không kề bị kéo về ~0 xác suất.
    torch = _torch()
    return logits + torch.where(mask > 0, torch.zeros_like(logits),
                                torch.full_like(logits, neg_inf))


if __name__ == "__main__":
    p = Path("dataset/sample_labels.json")
    if not p.exists():
        raise FileNotFoundError(p)
    ds = RoutingDataset(p)
    print(f"{len(ds)} mẫu next-hop, N={ds.n_nodes}")
    b = ds[0]
    for k, v in b.items():
        print(f"  {k:<10}: {tuple(v.shape)}")