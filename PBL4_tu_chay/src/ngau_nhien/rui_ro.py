"""Nhãn hậu kiểm Monte Carlo: xác suất đứt trong (t, t + Δ], có điều kiện còn sống tại t.

Không nhân xác suất cạnh để tính tuyến: cùng bão/nút hỏng làm các cạnh phụ thuộc nhau.
NaN nghĩa là không có mẫu hợp lệ, không phải rủi ro bằng 0.
"""
from __future__ import annotations

import numpy as np


def cua_so_dut(song, moc, delta):
    """Trả mask ít nhất một lần đứt ở Δ bước TƯƠNG LAI, không bao gồm bước hiện tại."""
    song = np.asarray(song, dtype=bool)
    moc = np.asarray(moc)
    if song.ndim != 3 or song.shape[1] != song.shape[2]:
        raise ValueError("song phải có shape (T,N,N)")
    if delta < 1 or moc.ndim != 1 or not np.issubdtype(moc.dtype, np.integer):
        raise ValueError("delta phải >= 1 và moc là vector chỉ số nguyên")
    if len(moc) == 0 or np.any(moc < 0) or np.any(moc + delta >= len(song)):
        raise ValueError("mỗi mốc phải có đủ cửa sổ tương lai, không cắt ngắn ở cuối")
    return np.stack([~song[t + 1:t + delta + 1].all(axis=0) for t in moc])


def xac_suat(so_dut, so_mau):
    p = np.full(so_mau.shape, np.nan, dtype=np.float32)
    np.divide(so_dut, so_mau, out=p, where=so_mau > 0)
    return p


class TichLuyRuiRo:
    """Đếm trên các thế giới mô phỏng, dùng cùng thế giới cho mọi cạnh của một tuyến."""

    def __init__(self, moc, n_nodes, delta, query_snapshot, paths):
        self.moc = np.asarray(moc, dtype=np.int64)
        self.delta = delta
        self.query_snapshot = np.asarray(query_snapshot, dtype=np.int64)
        self.paths = [np.asarray(p, dtype=np.int64) for p in paths]
        if len(self.paths) != len(self.query_snapshot):
            raise ValueError("mỗi truy vấn phải có một tuyến (có thể rỗng nếu không tới được)")
        if np.any(self.query_snapshot < 0) or np.any(self.query_snapshot >= len(self.moc)):
            raise ValueError("query_snapshot nằm ngoài moc")
        if any(p.ndim != 1 or np.any((p < 0) | (p >= n_nodes)) for p in self.paths):
            raise ValueError("chỉ số nút trong tuyến không hợp lệ")
        shape = (len(self.moc), n_nodes, n_nodes)
        self.edge_count = np.zeros(shape, dtype=np.uint32)
        self.edge_failed = np.zeros(shape, dtype=np.uint32)
        self.route_count = np.zeros(len(paths), dtype=np.uint32)
        self.route_failed = np.zeros(len(paths), dtype=np.uint32)
        self.worlds = 0

    def them(self, song):
        song = np.asarray(song, dtype=bool)
        failure = cua_so_dut(song, self.moc, self.delta)
        current = song[self.moc].copy()
        diag = np.arange(song.shape[1])
        current[:, diag, diag] = False
        if current.shape != self.edge_count.shape:
            raise ValueError("kích thước mạng không khớp")
        self.edge_count += current
        self.edge_failed += current & failure
        for i, (k, p) in enumerate(zip(self.query_snapshot, self.paths)):
            if len(p) < 2:
                continue
            u, v = p[:-1], p[1:]
            if current[k, u, v].all():
                self.route_count[i] += 1
                self.route_failed[i] += int(failure[k, u, v].any())
        self.worlds += 1

    def ket_qua(self):
        return {
            "risk_edge_oracle": xac_suat(self.edge_failed, self.edge_count),
            "risk_edge_count": self.edge_count.copy(),
            "risk_route_oracle": xac_suat(self.route_failed, self.route_count),
            "risk_route_count": self.route_count.copy(),
        }
