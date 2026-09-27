
from __future__ import annotations

import sys
from pathlib import Path

import networkx as nx
import numpy as np

# Cho phép chạy trực tiếp file này (python src/graph/graph_builder.py)
# mà vẫn import được package "src.physics" khi chạy từ thư mục gốc D:\PBL4.
sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.physics.tle_loader import (  # noqa: E402
    GroundStation,
    compute_gsl_links,
    compute_isl_links,
    load_tle,
    satellite_positions_km,
)
from skyfield.api import load  # noqa: E402

# ----------------------------------------------------------------------------
# Hằng số hệ thống (băng thông, trễ giả định/thực nghiệm)
# ----------------------------------------------------------------------------
SPEED_OF_LIGHT_KM_S = 3.0e5      # c ≈ 3 x 10^5 km/s (300,000 km/s)

BANDWIDTH_ISL_MBPS = 1000.0      # băng thông ISL (liên vệ tinh)
BANDWIDTH_GSL_MBPS = 100.0       # băng thông GSL (vệ tinh - mặt đất)

PACKET_SIZE_BITS = 1500 * 8      # kích thước gói tin giả định (1500 byte MTU)
PROCESSING_DELAY_MS = 0.5        # d_proc: trễ xử lý hằng số tại mỗi nút

# Tham số hàng đợi M/M/1 giả định: λ (tốc độ đến) / μ (tốc độ phục vụ)
# Dùng chung một hệ số tải rho cho toàn mạng ở mức demo; thực tế mỗi liên
# kết có thể có rho riêng dựa trên lưu lượng mô phỏng.
QUEUE_LOAD_FACTOR = 0.3          # rho = λ/μ, phải < 1 để hàng đợi ổn định

PACKET_LOSS_GSL = 0.02           # xác suất mất gói do nhiễu thời tiết (GSL)
PACKET_LOSS_ISL = 0.001          # xác suất mất gói ISL (ổn định hơn, ít nhiễu)

INF = float("inf")


# ----------------------------------------------------------------------------
# Tính từng thành phần độ trễ d_ij
# ----------------------------------------------------------------------------
def propagation_delay_ms(distance_km: float) -> float:
    """d_prop = ℓ_ij / c."""
    return (distance_km / SPEED_OF_LIGHT_KM_S) * 1000.0  # đổi sang ms


def transmission_delay_ms(bandwidth_mbps: float, packet_size_bits: int = PACKET_SIZE_BITS) -> float:
    """d_trans = kích thước gói / băng thông."""
    bandwidth_bps = bandwidth_mbps * 1e6
    return (packet_size_bits / bandwidth_bps) * 1000.0  # đổi sang ms


def queueing_delay_ms(bandwidth_mbps: float, rho: float = QUEUE_LOAD_FACTOR) -> float:
    """
    d_queue: mô phỏng theo hàng đợi M/M/1.
    Thời gian chờ trung bình trong hệ thống M/M/1: W = 1 / (μ - λ) = (1/μ) * rho / (1 - rho)
    Ở đây dùng thời gian phục vụ 1 gói (1/μ) xấp xỉ bằng transmission_delay.
    """
    service_time_ms = transmission_delay_ms(bandwidth_mbps)
    if rho >= 1.0:
        return INF  # hàng đợi không ổn định
    return service_time_ms * (rho / (1.0 - rho))


def total_link_delay_ms(distance_km: float, bandwidth_mbps: float) -> float:
    """d_ij = d_prop + d_trans + d_queue + d_proc."""
    d_prop = propagation_delay_ms(distance_km)
    d_trans = transmission_delay_ms(bandwidth_mbps)
    d_queue = queueing_delay_ms(bandwidth_mbps)
    d_proc = PROCESSING_DELAY_MS
    return d_prop + d_trans + d_queue + d_proc


# ----------------------------------------------------------------------------
# Dựng đồ thị G_τ từ kết quả Giai đoạn 1
# ----------------------------------------------------------------------------
def build_graph(
    gsl_links: list[dict],
    isl_links: list[dict],
) -> nx.DiGraph:
    """
    Dựng đồ thị có hướng G_τ = (V, E_τ) từ danh sách liên kết GSL và ISL.

    - GSL: thêm CẢ 2 chiều (uplink: gs->sat, downlink: sat->gs), vì băng
      thông/độ trễ có thể khác nhau giữa 2 chiều trong thực tế (ở đây demo
      dùng cùng công thức, nhưng cấu trúc đã sẵn sàng để phân biệt sau).
    - ISL: thêm CẢ 2 chiều (đối xứng), vì laser liên vệ tinh thường
      song công (full-duplex).

    Mỗi cạnh có thuộc tính: delay_ms, bandwidth_mbps, packet_loss.
    """
    G = nx.DiGraph()

    # --- Cạnh GSL ---
    for link in gsl_links:
        if not link["exists"]:
            continue
        gs = link["ground_station"]
        sat = link["satellite"]
        distance_km = link["slant_range_km"]
        delay = total_link_delay_ms(distance_km, BANDWIDTH_GSL_MBPS)

        # Uplink: ground -> satellite
        G.add_edge(
            gs, sat,
            delay_ms=delay,
            bandwidth_mbps=BANDWIDTH_GSL_MBPS,
            packet_loss=PACKET_LOSS_GSL,
            link_type="GSL_uplink",
        )
        # Downlink: satellite -> ground
        G.add_edge(
            sat, gs,
            delay_ms=delay,
            bandwidth_mbps=BANDWIDTH_GSL_MBPS,
            packet_loss=PACKET_LOSS_GSL,
            link_type="GSL_downlink",
        )

    # --- Cạnh ISL ---
    for link in isl_links:
        if not link["exists"]:
            continue
        sat_a = link["sat_a"]
        sat_b = link["sat_b"]
        distance_km = link["distance_km"]
        delay = total_link_delay_ms(distance_km, BANDWIDTH_ISL_MBPS)

        G.add_edge(
            sat_a, sat_b,
            delay_ms=delay,
            bandwidth_mbps=BANDWIDTH_ISL_MBPS,
            packet_loss=PACKET_LOSS_ISL,
            link_type="ISL",
        )
        G.add_edge(
            sat_b, sat_a,
            delay_ms=delay,
            bandwidth_mbps=BANDWIDTH_ISL_MBPS,
            packet_loss=PACKET_LOSS_ISL,
            link_type="ISL",
        )

    return G


# ----------------------------------------------------------------------------
# Trích xuất ma trận toán học từ đồ thị
# ----------------------------------------------------------------------------
def extract_adjacency_matrix(G: nx.DiGraph, node_order: list[str] | None = None) -> np.ndarray:
    """Ma trận kề nhị phân A ∈ {0,1}^(N x N)."""
    if node_order is None:
        node_order = list(G.nodes())
    A = nx.to_numpy_array(G, nodelist=node_order, weight=None)
    return (A > 0).astype(int)


def extract_weight_matrix(G: nx.DiGraph, node_order: list[str] | None = None) -> np.ndarray:
    """
    Ma trận trọng số độ trễ W ∈ R^(N x N).
    Các cặp nút không kết nối được gán trọng số +∞.
    """
    if node_order is None:
        node_order = list(G.nodes())
    N = len(node_order)
    W = np.full((N, N), INF, dtype=float)

    index = {name: i for i, name in enumerate(node_order)}
    for u, v, data in G.edges(data=True):
        i, j = index[u], index[v]
        W[i, j] = data["delay_ms"]

    np.fill_diagonal(W, 0.0)  # trễ từ 1 nút tới chính nó = 0
    return W


def max_degree(G: nx.DiGraph) -> int:
    """Bậc lớn nhất Δ (dùng out-degree, vì đó là số lựa chọn tối đa tại mỗi bước định tuyến)."""
    if G.number_of_nodes() == 0:
        return 0
    return max(dict(G.out_degree()).values())


def flatten_weight_matrix(W: np.ndarray, inf_replacement: float = -1.0) -> np.ndarray:
    """
    vec(W_τ): làm phẳng ma trận trọng số thành vector 1 chiều để làm điều
    kiện đầu vào x cho mô hình. Giá trị +inf được thay bằng inf_replacement
    (mặc định -1) vì mạng nơ-ron không xử lý được inf trực tiếp.
    """
    W_clean = np.where(np.isinf(W), inf_replacement, W)
    return W_clean.flatten()


# ----------------------------------------------------------------------------
# Demo / smoke test
# ----------------------------------------------------------------------------
def main():
    ts = load.timescale()
    t = ts.now()

    tle_path = Path("data/tle/starlink.txt")
    if not tle_path.exists():
        raise FileNotFoundError(f"Không tìm thấy {tle_path}.")

    print(f"[1/5] Nạp TLE và lấy mẫu vệ tinh ...")
    satellites = load_tle(tle_path)
    sample = satellites[:15]  # mẫu nhỏ để đồ thị demo gọn, dễ đọc
    print(f"      -> Dùng {len(sample)} vệ tinh mẫu.")

    print("[2/5] Tính liên kết GSL/ISL (Giai đoạn 1) ...")
    ground_stations = [
        GroundStation("Hanoi", 21.0285, 105.8542),
        GroundStation("DaNang", 16.0544, 108.2022),
        GroundStation("HoChiMinh", 10.7626, 106.6602),
    ]
    gsl_links = compute_gsl_links(sample, ground_stations, t)
    positions = satellite_positions_km(sample, t)
    isl_links = compute_isl_links(positions)

    n_gsl = sum(1 for l in gsl_links if l["exists"])
    n_isl = sum(1 for l in isl_links if l["exists"])
    print(f"      -> {n_gsl} liên kết GSL, {n_isl} liên kết ISL khả dụng.")

    print("[3/5] Dựng đồ thị G_τ (nx.DiGraph) ...")
    G = build_graph(gsl_links, isl_links)
    print(f"      -> Đồ thị có {G.number_of_nodes()} nút, {G.number_of_edges()} cạnh có hướng.")

    if G.number_of_edges() == 0:
        print("      !! Không có cạnh nào trong đồ thị (do mẫu 15 vệ tinh quá nhỏ,")
        print("         không bay ngang trạm mặt đất tại thời điểm này). Vẫn tiếp tục demo ma trận rỗng.")

    print("[4/5] Trích xuất ma trận A, W ...")
    node_order = list(G.nodes())
    A = extract_adjacency_matrix(G, node_order)
    W = extract_weight_matrix(G, node_order)
    delta = max_degree(G)
    print(f"      -> A shape: {A.shape}, W shape: {W.shape}, Δ (bậc lớn nhất) = {delta}")

    # In một vài cạnh mẫu kèm delay để kiểm tra thủ công
    for u, v, data in list(G.edges(data=True))[:5]:
        print(
            f"         {u:<15} -> {v:<15} "
            f"[{data['link_type']:<13}] delay={data['delay_ms']:.3f}ms "
            f"bw={data['bandwidth_mbps']:.0f}Mbps loss={data['packet_loss']:.3f}"
        )

    print("[5/5] Làm phẳng ma trận W thành vector trạng thái x ...")
    x = flatten_weight_matrix(W)
    print(f"      -> vec(W_τ) shape: {x.shape}  (đây là input x cho model AI)")

    print("\nHoàn tất Giai đoạn 2 (smoke test).")


if __name__ == "__main__":
    main()
