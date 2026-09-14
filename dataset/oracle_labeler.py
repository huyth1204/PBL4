from __future__ import annotations

import argparse
import random
import sys
from datetime import timedelta
from pathlib import Path

import networkx as nx
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from skyfield.api import load

from src.graph.graph_builder import (
    build_graph,
    extract_weight_matrix,
    flatten_weight_matrix,
)

from src.physics.tle_loader import (
    GroundStation,
    compute_gsl_links,
    compute_isl_links,
    load_tle,
    satellite_positions_km,
)


def find_optimal_path(
    G: nx.DiGraph,
    source: str,
    target: str,
    weight_key: str = "delay_ms",
) -> list[str] | None:
    if not G.has_node(source) or not G.has_node(target):
        return None

    try:
        return nx.dijkstra_path(
            G,
            source,
            target,
            weight=weight_key,
        )
    except nx.NetworkXNoPath:
        return None


def encode_one_hot(
    node_name: str,
    node_order: list[str],
) -> np.ndarray:
    vec = np.zeros(len(node_order), dtype=np.float32)

    try:
        idx = node_order.index(node_name)
        vec[idx] = 1.0
    except ValueError:
        pass

    return vec


def encode_path_to_indices(
    path: list[str],
    node_order: list[str],
) -> np.ndarray:
    node_to_index = {
        node: idx
        for idx, node in enumerate(node_order)
    }

    indices = [
        node_to_index[node]
        for node in path
        if node in node_to_index
    ]

    return np.array(indices, dtype=np.int32)


def generate_dataset_snapshots(
    tle_path: Path,
    ground_stations: list[GroundStation],
    num_snapshots: int = 50,
    time_step_seconds: int = 10,
    num_pairs_per_snapshot: int = 5,
    sample_sat_count: int = 50,
    min_elevation_deg: float = 15.0,
    seed: int = 42,
) -> dict[str, np.ndarray]:

    if not tle_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy file TLE: {tle_path}"
        )

    if num_snapshots <= 0:
        raise ValueError("num_snapshots phải > 0")

    if time_step_seconds <= 0:
        raise ValueError("time_step_seconds phải > 0")

    if num_pairs_per_snapshot <= 0:
        raise ValueError("num_pairs_per_snapshot phải > 0")

    if sample_sat_count <= 0:
        raise ValueError("sample_sat_count phải > 0")

    random.seed(seed)
    np.random.seed(seed)

    ts = load.timescale()
    t_start = ts.now()

    satellites = load_tle(tle_path)

    if len(satellites) == 0:
        raise ValueError("File TLE không chứa vệ tinh nào.")

    sample_sats = satellites[:sample_sat_count]

    all_sat_names = [
        sat.name
        for sat in sample_sats
    ]

    all_gs_names = [
        gs.name
        for gs in ground_stations
    ]

    global_node_order = sorted(
        set(all_sat_names + all_gs_names)
    )

    N = len(global_node_order)

    if N == 0:
        raise ValueError("Không có node nào trong mạng.")

    node_to_index = {
        node: idx
        for idx, node in enumerate(global_node_order)
    }

    x_weights_list = []
    x_sources_list = []
    x_targets_list = []

    y_paths_indices = []
    y_paths_text = []

    print(
        f"[Oracle Labeler] Bắt đầu sinh "
        f"{num_snapshots} snapshots..."
    )

    print(
        f" -> Không gian mạng cố định: "
        f"N = {N} nodes "
        f"({len(sample_sats)} satellites + "
        f"{len(ground_stations)} ground stations)"
    )

    for k in range(num_snapshots):

        current_datetime = (
            t_start.utc_datetime()
            + timedelta(
                seconds=k * time_step_seconds
            )
        )

        t_k = ts.utc(current_datetime)

        gsl_links = compute_gsl_links(
            sample_sats,
            ground_stations,
            t_k,
            min_elevation_deg=min_elevation_deg,
        )

        positions = satellite_positions_km(
            sample_sats,
            t_k,
        )

        isl_links = compute_isl_links(
            positions
        )

        G = build_graph(
            gsl_links,
            isl_links,
        )

        G.add_nodes_from(
            global_node_order
        )

        if G.number_of_edges() == 0:
            print(
                f" [Snapshot {k + 1}/{num_snapshots}] "
                f"Cảnh báo: graph không có cạnh."
            )
            continue

        W = extract_weight_matrix(
            G,
            global_node_order,
        )

        vec_W = flatten_weight_matrix(
            W,
            inf_replacement=-1.0,
        )

        if len(vec_W) != N * N:
            raise ValueError(
                f"Kích thước vec(W) không đúng. "
                f"Expected={N * N}, "
                f"Got={len(vec_W)}"
            )

        connected_nodes = [
            node
            for node in global_node_order
            if G.degree(node) > 0
        ]

        if len(connected_nodes) < 2:
            print(
                f" [Snapshot {k + 1}/{num_snapshots}] "
                f"Cảnh báo: số node có kết nối < 2."
            )
            continue

        active_gs_sources = [
            gs.name
            for gs in ground_stations
            if G.has_node(gs.name)
            and G.degree(gs.name) > 0
        ]

        if active_gs_sources:
            possible_sources = active_gs_sources
        else:
            possible_sources = connected_nodes

        possible_targets = connected_nodes

        reachable_pairs = []

        for src in possible_sources:

            try:
                distances, paths = nx.single_source_dijkstra(
                    G,
                    src,
                    weight="delay_ms",
                )
            except (
                nx.NetworkXError,
                nx.NodeNotFound,
            ):
                continue

            for tgt, path in paths.items():

                if src == tgt:
                    continue

                if tgt not in possible_targets:
                    continue

                if len(path) < 2:
                    continue

                if not np.isfinite(
                    distances.get(tgt, np.inf)
                ):
                    continue

                reachable_pairs.append(
                    (src, tgt, path)
                )

        if not reachable_pairs:
            print(
                f" [Snapshot {k + 1}/{num_snapshots}] "
                f"Cảnh báo: không có cặp source-target "
                f"nào có đường đi."
            )
            continue

        sample_count = min(
            num_pairs_per_snapshot,
            len(reachable_pairs),
        )

        selected_pairs = random.sample(
            reachable_pairs,
            sample_count,
        )

        for src, tgt, path in selected_pairs:

            onehot_src = np.zeros(
                N,
                dtype=np.float32,
            )

            onehot_tgt = np.zeros(
                N,
                dtype=np.float32,
            )

            onehot_src[node_to_index[src]] = 1.0
            onehot_tgt[node_to_index[tgt]] = 1.0

            path_idx = np.array(
                [
                    node_to_index[node]
                    for node in path
                ],
                dtype=np.int32,
            )

            x_weights_list.append(
                np.asarray(
                    vec_W,
                    dtype=np.float32,
                )
            )

            x_sources_list.append(
                onehot_src
            )

            x_targets_list.append(
                onehot_tgt
            )

            y_paths_indices.append(
                path_idx
            )

            y_paths_text.append(
                "->".join(path)
            )

        print(
            f" [Snapshot {k + 1}/{num_snapshots}] "
            f"Nodes={N} | "
            f"Edges={G.number_of_edges()} | "
            f"Active GS={len(active_gs_sources)} | "
            f"Reachable pairs={len(reachable_pairs)} | "
            f"Samples={sample_count} | "
            f"Total={len(x_weights_list)}"
        )

    if len(x_weights_list) == 0:
        raise ValueError(
            "Không tạo được mẫu dữ liệu nào. "
            "Hãy kiểm tra TLE, Ground Station "
            "và graph_builder."
        )

    max_path_len = max(
        len(path)
        for path in y_paths_indices
    )

    padded_y_paths = np.full(
        (
            len(y_paths_indices),
            max_path_len,
        ),
        -1,
        dtype=np.int32,
    )

    for i, path in enumerate(
        y_paths_indices
    ):
        padded_y_paths[
            i,
            :len(path)
        ] = path

    dataset = {
        "x_weights": np.asarray(
            x_weights_list,
            dtype=np.float32,
        ),
        "x_sources": np.asarray(
            x_sources_list,
            dtype=np.float32,
        ),
        "x_targets": np.asarray(
            x_targets_list,
            dtype=np.float32,
        ),
        "y_paths_indices": padded_y_paths,
        "y_paths_text": np.asarray(
            y_paths_text,
            dtype=str,
        ),
        "node_order": np.asarray(
            global_node_order,
            dtype=str,
        ),
    }

    return dataset


def save_dataset_npz(
    dataset: dict[str, np.ndarray],
    output_path: Path,
) -> None:

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez_compressed(
        output_path,
        x_weights=dataset["x_weights"],
        x_sources=dataset["x_sources"],
        x_targets=dataset["x_targets"],
        y_paths_indices=dataset["y_paths_indices"],
        y_paths_text=dataset["y_paths_text"],
        node_order=dataset["node_order"],
    )

    print(
        "\n[Oracle Labeler] "
        "Đã xuất dataset thành công:"
    )

    print(
        f" - File: {output_path}"
    )

    print(
        f" - Tổng số samples: "
        f"{len(dataset['x_weights'])}"
    )

    print(
        f" - x_weights: "
        f"{dataset['x_weights'].shape}"
    )

    print(
        f" - x_sources: "
        f"{dataset['x_sources'].shape}"
    )

    print(
        f" - x_targets: "
        f"{dataset['x_targets'].shape}"
    )

    print(
        f" - y_paths_indices: "
        f"{dataset['y_paths_indices'].shape}"
    )

    print(
        f" - node_order: "
        f"{dataset['node_order'].shape}"
    )


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Bộ sinh dataset Oracle "
            "gán nhãn đường đi bằng Dijkstra"
        )
    )

    parser.add_argument(
        "--tle",
        type=str,
        default="data/tle/starlink.txt",
        help="Đường dẫn file TLE",
    )

    parser.add_argument(
        "--out",
        type=str,
        default="dataset/oracle_dataset.npz",
        help="Đường dẫn dataset đầu ra",
    )

    parser.add_argument(
        "--snapshots",
        type=int,
        default=50,
        help="Số lượng snapshot",
    )

    parser.add_argument(
        "--step",
        type=int,
        default=10,
        help="Khoảng thời gian giữa các snapshot (giây)",
    )

    parser.add_argument(
        "--pairs",
        type=int,
        default=5,
        help="Số cặp source-target mỗi snapshot",
    )

    parser.add_argument(
        "--sats",
        type=int,
        default=50,
        help="Số lượng vệ tinh lấy từ TLE",
    )

    parser.add_argument(
        "--min-elev",
        type=float,
        default=15.0,
        help="Góc ngẩng tối thiểu",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    args = parser.parse_args()

    tle_path = Path(args.tle)
    out_path = Path(args.out)

    if not tle_path.exists():
        print(
            f"!! Không tìm thấy file TLE: "
            f"'{tle_path}'"
        )
        print(
            "Hãy đặt file TLE vào đúng đường dẫn "
            "hoặc sử dụng --tle để chỉ định file."
        )
        sys.exit(1)

    ground_stations = [
        GroundStation(
            "Hanoi",
            21.0285,
            105.8542,
        ),
        GroundStation(
            "DaNang",
            16.0544,
            108.2022,
        ),
        GroundStation(
            "HoChiMinh",
            10.7626,
            106.6602,
        ),
    ]

    dataset = generate_dataset_snapshots(
        tle_path=tle_path,
        ground_stations=ground_stations,
        num_snapshots=args.snapshots,
        time_step_seconds=args.step,
        num_pairs_per_snapshot=args.pairs,
        sample_sat_count=args.sats,
        min_elevation_deg=args.min_elev,
        seed=args.seed,
    )

    save_dataset_npz(
        dataset,
        out_path,
    )


if __name__ == "__main__":
    main()