"""
Giai đoạn 5: Xuất mặt nạ hợp lệ (Validity Mask)
=================================================

Module này trích xuất mặt nạ hợp lệ nhị phân m tại mỗi nút trung gian,
để triệt tiêu hoàn toàn các lỗi sinh ra bước đi không hợp lệ (tới nút
không kết nối) của mô hình AI sinh (diffusion model).

Thuật toán:
    Tại nút hiện tại v_i, dựa trên ma trận kề A_τ hiện tại, xuất ra
    vector nhị phân kích thước N với:
        m_j = 1 nếu có liên kết trực tiếp v_i -> v_j
        m_j = 0 nếu không có liên kết

Cách dùng phía model (không phải phần việc của bạn, nhưng để tham khảo):
    logit_j <- logit_j + log(m_j)   (log(0) = -inf, log(1) = 0)
    rồi đưa vào Softmax -> xác suất chọn nút không hợp lệ luôn bằng 0.

Chạy thử:
    python src/mask/validity_mask.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import networkx as nx
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.graph.graph_builder import build_graph, extract_adjacency_matrix  # noqa: E402
from src.physics.tle_loader import (  # noqa: E402
    GroundStation,
    compute_gsl_links,
    compute_isl_links,
    load_tle,
    satellite_positions_km,
)
from skyfield.api import load  # noqa: E402

# Giá trị dùng khi cộng logit cho nút KHÔNG hợp lệ (mô phỏng log(0) = -inf,
# nhưng dùng số hữu hạn rất âm để tránh NaN khi tính toán số thực trong PyTorch).
NEG_INF_LOGIT = -1e9


# ----------------------------------------------------------------------------
# Bước 1: xuất mặt nạ cho MỘT nút cụ thể
# ----------------------------------------------------------------------------
def get_validity_mask(A: np.ndarray, node_order: list[str], current_node: str) -> np.ndarray:
    """
    Tại nút current_node, trả về vector nhị phân kích thước N:
        m[j] = 1 nếu A[i, j] == 1 (có liên kết trực tiếp current_node -> node_order[j])
        m[j] = 0 nếu không.

    A: ma trận kề nhị phân (N x N), lấy từ extract_adjacency_matrix() ở Giai đoạn 2.
    node_order: danh sách tên nút theo đúng thứ tự dùng khi dựng A.
    """
    if current_node not in node_order:
        raise ValueError(f"Nút '{current_node}' không có trong node_order.")

    i = node_order.index(current_node)
    mask = A[i, :].copy()
    return mask.astype(int)


# ----------------------------------------------------------------------------
# Bước 2: xuất mặt nạ cho TẤT CẢ các nút cùng lúc (dùng khi cần batch)
# ----------------------------------------------------------------------------
def get_all_validity_masks(A: np.ndarray) -> np.ndarray:
    """
    Trả về chính ma trận A — vì với đồ thị đã có A, mặt nạ hợp lệ tại
    nút i CHÍNH LÀ hàng thứ i của A. Hàm này tồn tại để code gọi rõ ràng
    về mặt ý nghĩa (semantic), tách biệt với việc dùng A cho mục đích khác
    (ví dụ tính bậc, hay làm input x).
    """
    return A.copy()


# ----------------------------------------------------------------------------
# Bước 3: chuyển mặt nạ nhị phân sang logit additive (log-mask)
# ----------------------------------------------------------------------------
def mask_to_additive_logit(mask: np.ndarray, neg_inf: float = NEG_INF_LOGIT) -> np.ndarray:
    """
    Chuyển mask {0, 1} sang giá trị cộng logit:
        m_j = 1 -> 0        (không ảnh hưởng logit)
        m_j = 0 -> neg_inf  (kéo logit xuống rất thấp, Softmax ~ 0)

    Đây là bước mà nhóm AI sẽ dùng để cộng vào logit trước Softmax:
        y_j <- y_j + additive_logit_j
    """
    return np.where(mask == 1, 0.0, neg_inf)


# ----------------------------------------------------------------------------
# Bước 4: hàm kiểm tra tính hợp lệ của MỘT đường đi hoàn chỉnh
# ----------------------------------------------------------------------------
def validate_path(A: np.ndarray, node_order: list[str], path: list[str]) -> tuple[bool, str | None]:
    """
    Kiểm tra một đường đi y (danh sách tên nút) có hợp lệ trên đồ thị vật
    lý thực tế hay không — mọi cạnh liên tiếp trong path phải tồn tại
    trong A. Dùng để lọc/validate output của mô hình sinh trước khi áp
    dụng, như đặc tả yêu cầu ở mục 2.1.5.

    Trả về (True, None) nếu hợp lệ, hoặc (False, "mô tả lỗi") nếu không.
    """
    for u, v in zip(path[:-1], path[1:]):
        if u not in node_order or v not in node_order:
            return False, f"Nút '{u}' hoặc '{v}' không tồn tại trong đồ thị."
        mask = get_validity_mask(A, node_order, u)
        j = node_order.index(v)
        if mask[j] == 0:
            return False, f"Không có liên kết trực tiếp {u} -> {v} trên topo vật lý."
    return True, None


# ----------------------------------------------------------------------------
# Demo / smoke test
# ----------------------------------------------------------------------------
def main():
    ts = load.timescale()
    t = ts.now()

    tle_path = Path("data/tle/starlink.txt")
    if not tle_path.exists():
        raise FileNotFoundError(f"Không tìm thấy {tle_path}.")

    print("[1/4] Nạp TLE, dựng đồ thị G_τ (dùng lại Giai đoạn 1-2) ...")
    satellites = load_tle(tle_path)
    sample = satellites[:15]
    ground_stations = [
        GroundStation("Hanoi", 21.0285, 105.8542),
        GroundStation("DaNang", 16.0544, 108.2022),
        GroundStation("HoChiMinh", 10.7626, 106.6602),
    ]
    gsl_links = compute_gsl_links(sample, ground_stations, t)
    positions = satellite_positions_km(sample, t)
    isl_links = compute_isl_links(positions)
    G = build_graph(gsl_links, isl_links)
    node_order = list(G.nodes())
    A = extract_adjacency_matrix(G, node_order)
    print(f"      -> Đồ thị: {len(node_order)} nút, ma trận A shape {A.shape}")

    print("[2/4] Xuất mặt nạ hợp lệ cho một nút cụ thể ...")
    current_node = node_order[0]
    mask = get_validity_mask(A, node_order, current_node)
    valid_neighbors = [node_order[j] for j in range(len(node_order)) if mask[j] == 1]
    print(f"      -> Tại nút {current_node}: {len(valid_neighbors)} nút kế tiếp hợp lệ")
    print(f"         {valid_neighbors[:5]}")

    print("[3/4] Chuyển mask sang additive logit (log-mask) ...")
    additive_logit = mask_to_additive_logit(mask)
    print(f"      -> additive_logit shape {additive_logit.shape}, "
          f"ví dụ: {additive_logit[:5]} (0 = giữ nguyên, số âm lớn = loại bỏ)")

    print("[4/4] Kiểm tra tính hợp lệ của một đường đi mẫu ...")
    if valid_neighbors:
        # Đường đi hợp lệ: current_node -> 1 hàng xóm hợp lệ
        good_path = [current_node, valid_neighbors[0]]
        is_valid, err = validate_path(A, node_order, good_path)
        print(f"      -> Path {good_path}: hợp lệ={is_valid}")

        # Đường đi KHÔNG hợp lệ (cố tình nối 2 nút không kề nhau, nếu có)
        invalid_candidates = [node_order[j] for j in range(len(node_order)) if mask[j] == 0]
        if invalid_candidates:
            bad_path = [current_node, invalid_candidates[0]]
            is_valid_bad, err_bad = validate_path(A, node_order, bad_path)
            print(f"      -> Path {bad_path}: hợp lệ={is_valid_bad}  (lỗi: {err_bad})")

    print("\nHoàn tất Giai đoạn 5 (smoke test).")


if __name__ == "__main__":
    main()
