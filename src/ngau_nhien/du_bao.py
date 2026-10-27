"""Dự báo liên kết bằng lịch quỹ đạo và trạng thái đã quan sát, không đọc tương lai thời tiết."""
import numpy as np


def du_bao_quy_dao(W_base, W_observed, observed_at, decision_at, delta, node_off):
    if delta < 1 or not 0 <= observed_at <= decision_at or decision_at + delta >= len(W_base):
        raise ValueError("Mốc quan sát/quyết định/cửa sổ không hợp lệ")
    G0 = W_base[observed_at]
    if W_observed.shape != G0.shape or np.asarray(node_off).shape != (len(G0),):
        raise ValueError("Kích thước quan sát/nút tắt không khớp mạng")
    factor = np.ones_like(G0, dtype=float)
    np.divide(np.maximum(W_observed, 0), G0, out=factor, where=G0 > 0)
    future = W_base[decision_at + 1:decision_at + delta + 1]
    valid = (W_base[decision_at] > 0) & (future > 0).all(0)
    # Cạnh mới xuất hiện theo quỹ đạo vẫn phải tôn trọng nút đã biết là không hoạt động.
    valid &= ~np.asarray(node_off, bool)[:, None] & ~np.asarray(node_off, bool)[None, :]
    return np.where(valid, future.mean(0) * factor, 0.0)
