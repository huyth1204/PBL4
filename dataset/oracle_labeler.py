"""
Giai đoạn 3: Bộ tạo nhãn hoạt động ngoại tuyến (Dijkstra Oracle Labeler)
=========================================================================

Module này tạo dữ liệu huấn luyện D = {(x_i, y*_i)} cho nhóm AI:

1. Với mỗi snapshot G_τk, chạy Dijkstra của NetworkX tìm đường đi tối ưu y*
   giữa các cặp nguồn-đích ngẫu nhiên.
2. Vì Dijkstra chỉ cộng tuyến tính được trọng số, tỷ lệ mất gói p_ij được
   chuyển sang -log(1 - p_ij) để tương thích phép cộng, kết hợp với độ trễ:

       Weight_ij = w1 * d_ij - w2 * log(1 - p_ij)

3. Gom tất cả (x_i, y*_i) và lưu thành .json để huấn luyện offline.

Chạy thử:
    python src/labeler/oracle_labeler.py
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import networkx as nx
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.graph.graph_builder import (  # noqa: E402
    build_graph,
    extract_adjacency_matrix,
    extract_weight_matrix,
    flatten_weight_matrix,
    max_degree,
)
from src.physics.tle_loader import (  # noqa: E402
    GroundStation,
    compute_gsl_links,
    compute_isl_links,
    load_tle,
    satellite_positions_km,
)
from skyfield.api import load  # noqa: E402

# ----------------------------------------------------------------------------
# Trọng số tổ hợp cho Dijkstra đa chỉ số
# ----------------------------------------------------------------------------
W1_DELAY = 1.0       # trọng số ưu tiên độ trễ
W2_RELIABILITY = 5.0  # trọng số ưu tiên độ tin cậy (mất gói)


# ----------------------------------------------------------------------------
# Bước 1: gán trọng số tổ hợp lên đồ thị (không sửa d_ij/p_ij gốc)
# ----------------------------------------------------------------------------
def add_dijkstra_weight(G: nx.DiGraph, w1: float = W1_DELAY, w2: float = W2_RELIABILITY) -> None:
    """
    Với mỗi cạnh, tính:
        weight_ij = w1 * d_ij - w2 * log(1 - p_ij)

    và lưu vào thuộc tính 'dijkstra_weight' (giữ nguyên delay_ms, packet_loss
    gốc để dùng lại ở Giai đoạn 4 khi tính 5 chỉ số).
    """
    for u, v, data in G.edges(data=True):
        d_ij = data["delay_ms"]
        p_ij = data["packet_loss"]
        # Giới hạn p_ij < 1 để tránh log(0) khi mất gói = 100%
        p_ij_safe = min(p_ij, 0.999999)
        reliability_cost = -np.log(1.0 - p_ij_safe)
        data["dijkstra_weight"] = w1 * d_ij + w2 * reliability_cost


# ----------------------------------------------------------------------------
# Bước 2: chạy Dijkstra cho một cặp nguồn-đích
# ----------------------------------------------------------------------------
def find_optimal_path(G: nx.DiGraph, source: str, target: str) -> list[str] | None:
    """
    Chạy Dijkstra (networkx) trên trọng số tổ hợp 'dijkstra_weight'.
    Trả về danh sách nút của đường đi tối ưu y*, hoặc None nếu không có đường đi.
    """
    try:
        path = nx.dijkstra_path(G, source, target, weight="dijkstra_weight")
        return path
    except nx.NetworkXNoPath:
        return None


# ----------------------------------------------------------------------------
# Bước 3: sinh nhiều cặp nguồn-đích ngẫu nhiên và gom dataset
# ----------------------------------------------------------------------------
def generate_dataset_sample(
    G: nx.DiGraph,
    node_order: list[str],
    n_pairs: int = 10,
    seed: int | None = None,
) -> list[dict]:
    """
    Với MỘT snapshot G_τk cố định (đã có node_order, ma trận A/W cố định),
    sinh n_pairs cặp (nguồn, đích) ngẫu nhiên, chạy Dijkstra cho từng cặp,
    và trả về danh sách các mẫu dataset:

        {
            "x": vec(W_τk) — vector trạng thái mạng (giống nhau cho mọi cặp
                 trong cùng 1 snapshot, vì trạng thái mạng không đổi),
            "source": tên nút nguồn,
            "target": tên nút đích,
            "y_star": danh sách nút trên đường đi tối ưu (nhãn),
        }

    Trong thực tế, "x" đầy đủ theo đặc tả (mục 2.1.5) còn cần mã hóa
    one-hot cặp nguồn-đích; ở đây lưu source/target riêng để nhóm AI tự
    ghép one-hot theo định dạng họ cần.
    """
    add_dijkstra_weight(G)

    W = extract_weight_matrix(G, node_order)
    x = flatten_weight_matrix(W).tolist()

    rng = random.Random(seed)
    samples = []
    attempts = 0
    max_attempts = n_pairs * 20  # tránh vòng lặp vô hạn nếu đồ thị rời rạc

    while len(samples) < n_pairs and attempts < max_attempts:
        attempts += 1
        source, target = rng.sample(node_order, 2)
        y_star = find_optimal_path(G, source, target)
        if y_star is None:
            continue  # không có đường đi giữa cặp này, thử cặp khác

        samples.append(
            {
                "x": x,
                "source": source,
                "target": target,
                "y_star": y_star,
                "n_hops": len(y_star) - 1,
            }
        )

    return samples


# ----------------------------------------------------------------------------
# Bước 4: lưu dataset ra file .json
# ----------------------------------------------------------------------------
def save_dataset(samples: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)


# ----------------------------------------------------------------------------
# Demo / smoke test
# ----------------------------------------------------------------------------
def main():
    ts = load.timescale()
    t = ts.now()

    tle_path = Path("data/tle/starlink.txt")
    if not tle_path.exists():
        raise FileNotFoundError(f"Không tìm thấy {tle_path}.")

    print("[1/5] Nạp TLE và lấy mẫu vệ tinh ...")
    satellites = load_tle(tle_path)
    # Dùng mẫu lớn hơn Giai đoạn 2 để đồ thị có nhiều đường đi hơn cho Dijkstra chọn.
    sample = satellites[:40]
    print(f"      -> Dùng {len(sample)} vệ tinh mẫu.")

    print("[2/5] Tính liên kết GSL/ISL và dựng đồ thị G_τ ...")
    ground_stations = [
        GroundStation("Hanoi", 21.0285, 105.8542),
        GroundStation("DaNang", 16.0544, 108.2022),
        GroundStation("HoChiMinh", 10.7626, 106.6602),
    ]
    gsl_links = compute_gsl_links(sample, ground_stations, t)
    positions = satellite_positions_km(sample, t)
    isl_links = compute_isl_links(positions)
    G = build_graph(gsl_links, isl_links)
    print(f"      -> Đồ thị có {G.number_of_nodes()} nút, {G.number_of_edges()} cạnh.")

    node_order = list(G.nodes())
    delta = max_degree(G)
    print(f"      -> Δ (bậc lớn nhất) = {delta}")

    print("[3/5] Gán trọng số tổ hợp Dijkstra (w1*delay - w2*log(1-p)) ...")
    add_dijkstra_weight(G, W1_DELAY, W2_RELIABILITY)
    sample_edge = list(G.edges(data=True))[0]
    print(
        f"      -> Ví dụ cạnh {sample_edge[0]}->{sample_edge[1]}: "
        f"delay={sample_edge[2]['delay_ms']:.2f}ms, "
        f"dijkstra_weight={sample_edge[2]['dijkstra_weight']:.2f}"
    )

    print("[4/5] Sinh dataset: chạy Dijkstra cho nhiều cặp nguồn-đích ngẫu nhiên ...")
    samples = generate_dataset_sample(G, node_order, n_pairs=10, seed=42)
    print(f"      -> Tạo được {len(samples)} mẫu (x, y*).")
    for s in samples[:3]:
        print(
            f"         {s['source']:<15} -> {s['target']:<15} "
            f"y*: {' -> '.join(s['y_star'][:4])}{' ...' if len(s['y_star']) > 4 else ''} "
            f"({s['n_hops']} hops)"
        )

    print("[5/5] Lưu dataset ra dataset/sample_labels.json ...")
    out_path = Path("dataset/sample_labels.json")
    save_dataset(samples, out_path)
    print(f"      -> Đã lưu {len(samples)} mẫu vào {out_path}")

    print("\nHoàn tất Giai đoạn 3 (smoke test).")


if __name__ == "__main__":
    main()
