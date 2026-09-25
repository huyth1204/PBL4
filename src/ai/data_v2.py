"""
data_v2.py — Loader cho dữ liệu SAGSIN 3 tầng (oracle_labeler_sagsin + split_dataset).

Mỗi file train/val/test.npz chứa:
    W_snapshots (S, N, N)   độ trễ ms, -1 = không có liên kết
    y_paths_indices (M, L)  đường đi oracle (Dijkstra theo delay_ms), -1 = đệm
    snapshot_id (M,)        chỉ số tra vào W_snapshots
    node_order / node_layer / node_kind (N,)

Đơn vị huấn luyện là ĐƯỜNG ĐI (không phải từng bước): GNN chỉ phụ thuộc
(snapshot, đích) nên chạy một lần cho cả đường, rồi tách ra các bước next-hop.
"""

from __future__ import annotations

import numpy as np

LAYERS = ["space", "air", "ground"]
KINDS = ["leo", "haps", "uav", "gateway", "user"]
DELAY_SCALE = 20.0   # ms; độ trễ lớn nhất trong dữ liệu ~17.2 ms
DEG_SCALE = 50.0     # bậc lớn nhất ~50


class SagsinData:
    def __init__(self, npz_path):
        d = np.load(npz_path, allow_pickle=True)
        if "W_snapshots" not in d.files:
            raise ValueError(f"{npz_path} là định dạng cũ (không có W_snapshots). Tạo lại bằng: "
                             "python dataset/split_dataset.py --in dataset/oracle_dataset_v2.npz --outdir dataset")
        self.W = d["W_snapshots"].astype(np.float32)
        self.node_order = d["node_order"]
        self.layer = d["node_layer"]
        self.kind = d["node_kind"]
        self.N = len(self.node_order)
        # user chỉ làm điểm đầu/cuối, không được chuyển tiếp (khớp single_source_relay_dijkstra)
        self.can_relay = np.array([k != "user" for k in self.kind])

        lay = np.array([[l == x for x in LAYERS] for l in self.layer], dtype=np.float32)
        kin = np.array([[k == x for x in KINDS] for k in self.kind], dtype=np.float32)
        self.static_feat = np.concatenate([lay, kin, self.can_relay[:, None].astype(np.float32)], 1)

        self.sid = d["snapshot_id"].astype(np.int64)
        self.paths = [[int(i) for i in row if i >= 0] for row in d["y_paths_indices"]]
        keep = [i for i, p in enumerate(self.paths) if len(p) >= 2]
        self.sid = self.sid[keep]
        self.paths = [self.paths[i] for i in keep]
        self.n_steps = sum(len(p) - 1 for p in self.paths)
        self.lookahead = False     # bật: thêm "nút này có nối thẳng tới đích không, trễ bao nhiêu"

    def __len__(self):
        return len(self.paths)

    @property
    def f_in(self):
        return self.static_feat.shape[1] + 3 + 2 * self.lookahead   # + bậc, trễ TB cạnh kề, cờ đích


def node_features(W, static_feat, dst, drop=None, lookahead=False):
    """Đặc trưng nút cho một lô đồ thị. W (P,N,N), dst (P,). drop (P,) bool: bỏ cờ đích (CFG).
    lookahead: thêm cột W[:, dst] (nút j nối thẳng tới đích? trễ?) — tri thức định tuyến, không phải GNN."""
    P, N, _ = W.shape
    A = W > 0
    deg = A.sum(-1) / DEG_SCALE
    mean_delay = np.where(A, W, 0).sum(-1) / np.maximum(A.sum(-1), 1) / DELAY_SCALE
    is_dst = np.zeros((P, N), np.float32)
    is_dst[np.arange(P), dst] = 1.0
    if drop is not None:
        is_dst[drop] = 0.0
    cols = [np.broadcast_to(static_feat, (P, N, static_feat.shape[1])),
            deg[..., None], mean_delay[..., None], is_dst[..., None]]
    if lookahead:
        to_dst = W[np.arange(P), :, dst]                 # (P,N) = W[p, j, dst[p]]
        adj = (to_dst > 0).astype(np.float32)
        dly = np.where(to_dst > 0, to_dst, 0) / DELAY_SCALE
        if drop is not None:
            adj[drop] = 0.0
            dly[drop] = 0.0
        cols += [adj[..., None], dly[..., None]]
    return np.concatenate(cols, -1).astype(np.float32)


def valid_mask(W_rows, dst, can_relay, visited=None):
    """Ứng viên next-hop hợp lệ. W_rows (S,N) = hàng W tại nút hiện tại.
    Hợp lệ khi: có cạnh, và (nút được chuyển tiếp hoặc chính là đích), chưa đi qua."""
    S, N = W_rows.shape
    m = (W_rows > 0) & can_relay[None, :]
    m[np.arange(S), dst] |= W_rows[np.arange(S), dst] > 0
    if visited is not None:
        m &= ~visited
    return m


def make_batch(data, path_idx, drop_prob=0.0, rng=None):
    """Lô gồm P đường đi -> đồ thị cấp đường + các bước next-hop."""
    rng = rng or np.random
    sids = data.sid[path_idx]
    W = data.W[sids]                                   # (P,N,N)
    dst = np.array([data.paths[i][-1] for i in path_idx])
    drop = rng.random(len(path_idx)) < drop_prob if drop_prob > 0 else None
    X = node_features(W, data.static_feat, dst, drop, data.lookahead)

    step_p, cur, nxt = [], [], []
    for p, i in enumerate(path_idx):
        path = data.paths[i]
        for k in range(len(path) - 1):
            step_p.append(p); cur.append(path[k]); nxt.append(path[k + 1])
    step_p, cur, nxt = map(np.array, (step_p, cur, nxt))
    rows = W[step_p, cur]                              # (S,N)
    mask = valid_mask(rows, dst[step_p], data.can_relay)
    return {
        "W": W, "X": X, "dst": dst, "drop": drop,
        "step_p": step_p, "cur": cur, "nxt": nxt, "rows": rows, "mask": mask,
    }
