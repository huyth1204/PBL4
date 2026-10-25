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

DELAY_SCALE_MS = 20.0  # trần co độ trễ về [0,1] — khớp DELAY_SCALE bên src/ai/data_v2.py

# Quy ước trọng số: < 0 nghĩa là KHÔNG có liên kết; >= 0 là độ trễ thật (ms).
# Dùng đúng một quy ước này cho cả normalize_state lẫn mask_row.
KHONG_CO_LIEN_KET = 0.0

# torch chỉ cần cho RoutingDataset; nạp muộn để phần lõi chạy được không cần torch.


# ---- lõi thuần numpy (test được không cần torch) ---------------------------

def normalize_state(x, n_nodes):
    # -1 (không nối) -> 0; độ trễ -> chia scale, kẹp [0,1].
    # Độ trễ 0 ms là liên kết CÓ THẬT nên không được gộp chung với "không nối";
    # vì thang đã kẹp về [0,1] nên 0 ms và "không nối" đều ra 0.0 — chỗ phân biệt
    # hai trường hợp này là `mask_row` (xem mask_row).
    x = np.asarray(x, dtype=np.float32)
    if x.size != n_nodes * n_nodes:
        raise ValueError(
            f"len(x)={x.size} nhưng N*N={n_nodes*n_nodes}. Sai N: phải dùng "
            f"node_order đầy đủ của đồ thị, không dùng số nút trong đường đi."
        )
    return np.clip(np.where(x < 0, 0.0, x / DELAY_SCALE_MS), 0.0, 1.0)


def mask_row(W, cur_idx):
    # Mặt nạ hợp lệ = out-neighbors (trọng số >= 0). Đồ thị có hướng nên dùng HÀNG.
    # Dùng >= 0 (không phải > 0) để liên kết có độ trễ đúng 0 ms vẫn hợp lệ;
    # "không có liên kết" được mã hoá bằng số âm.
    return (W[cur_idx, :] >= KHONG_CO_LIEN_KET).astype(np.int64)


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
    if not ids:
        raise ValueError("Không có snapshot nào để chia.")
    if len(ids) < 3:
        raise ValueError(
            f"Chỉ có {len(ids)} snapshot: không đủ để chia 3 tập khác snapshot."
        )
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    n = len(ids)
    n_tr = int(n * ratios[0])
    n_va = int(n * ratios[1])
    # int() cắt xuống 0 khi n nhỏ (n<7 => val rỗng) -> ép mỗi tập ít nhất 1 snapshot.
    n_tr = max(1, n_tr)
    n_va = max(1, n_va)
    n_tr = min(n_tr, n - 2)  # chừa tối thiểu 1 cho val và 1 cho test
    n_va = min(n_va, n - n_tr - 1)
    buckets = {"train": [], "val": [], "test": []}
    if n_va < 1 or n - n_tr - n_va < 1:  # pragma: no cover - chặn bởi kiểm tra trên
        raise ValueError(f"Không chia nổi {n} snapshot theo tỉ lệ {ratios}.")
    tr, va = set(ids[:n_tr]), set(ids[n_tr:n_tr + n_va])
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


def samples_from_npz(path):
    """Đọc npz do `oracle_labeler*.py` sinh ra -> đúng định dạng `samples` mà
    cut_nexthop() cần ({x, source, target, y_star, node_order}).

    Đây là cầu nối để pipeline này dùng được DỮ LIỆU THẬT. `sample_labels.json`
    trong repo là file cũ, không có `node_order` và không có script sinh -> không dùng.
    """
    with np.load(path, allow_pickle=False) as z:
        if "W_snapshots" not in z.files or "y_paths_text" not in z.files:
            raise KeyError(
                f"{path} không phải npz của oracle_labeler "
                f"(cần W_snapshots và y_paths_text; có: {sorted(z.files)})."
            )
        node_order = [str(s) for s in z["node_order"]]
        idx = {name: i for i, name in enumerate(node_order)}
        sid = z["snapshot_id"].astype(np.int64)
        W = z["W_snapshots"]
        src = z["x_sources"].argmax(1)
        dst = z["x_targets"].argmax(1)
        paths = z["y_paths_text"]
        out = []
        for k in range(len(sid)):
            # y_paths_text là tên nút ngăn bởi "->" (vd 'UAV-2->STARLINK-2606->UAV-4')
            path = [t for t in str(paths[k]).split("->") if t]
            if len(path) < 2 or any(t not in idx for t in path):
                continue  # đường đi không dùng được (đích không tới được)
            out.append({
                "x": W[int(sid[k])].ravel(),
                "source": node_order[src[k]],
                "target": node_order[dst[k]],
                "y_star": path,
                "node_order": node_order,
            })
    return out


class RoutingDataset(_Dataset):
    def __init__(self, samples_or_path, n_nodes=None):
        if isinstance(samples_or_path, (str, Path)):
            p = Path(samples_or_path)
            if p.suffix == ".npz":
                raw = samples_from_npz(p)
            else:
                with open(p, "r", encoding="utf-8") as f:
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
    import argparse

    ap = argparse.ArgumentParser(description="Kiểm tra nhanh RoutingDataset trên npz thật.")
    ap.add_argument("npz", nargs="?", default="dataset/oracle_dataset_v2.npz")
    args = ap.parse_args()

    p = Path(args.npz)
    if not p.exists():
        raise FileNotFoundError(
            f"{p} không có. Sinh bằng: python dataset/oracle_labeler_sagsin.py"
        )
    ds = RoutingDataset(p)
    print(f"{len(ds)} mẫu next-hop, N={ds.n_nodes}")
    b = ds[0]
    for k, v in b.items():
        print(f"  {k:<10}: {tuple(v.shape)}")