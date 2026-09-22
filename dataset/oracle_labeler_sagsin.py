"""
Oracle Labeler v2: thêm tầng Air (HAPS, UAV) + Ground (gateway, user) vào
bên cạnh Space đã có. So với oracle_labeler.py gốc:

  1. Thứ tự node đổi từ sorted(tên) sang THEO KHỐI: space -> air -> ground,
     mỗi khối sort theo tên. Lý do: nếu chỉ sort theo tên toàn cục, thêm
     HAPS/UAV sẽ xáo trộn chỉ số của TẤT CẢ các node cũ -> checkpoint model
     cũ (nếu có) hỏng ngay. Sort theo khối thì thêm node mới chỉ nối đuôi.
  2. build_graph_multilayer thay cho build_graph: ráp cả GSL, ISL, và
     local links (HAPS-Ground, UAV-Ground, Air-Air).
  3. single_source_relay_dijkstra thay cho single_source_dijkstra: chặn
     path đi XUYÊN QUA node có can_relay=False (user terminal).
  4. Nguồn/đích khi lấy mẫu: gateway + user + uav (endpoint=True), KHÔNG
     còn giới hạn chỉ ground station như bản gốc (vì giờ có UAV cũng là
     điểm cuối hợp lệ).
  5. Thêm --infra để chạy ablation: "space" | "space,haps" | "space,haps,uav"
     (ground luôn bật vì là endpoint).
  6. Lưu thêm node_layer, node_kind vào npz, và thống kê tỷ lệ reachable
     mỗi snapshot (để chứng minh Air/Ground cải thiện kết nối).
  7. SỬA LỖI BỘ NHỚ: bản gốc lưu một bản vec_W (dài N^2) cho MỖI mẫu, dù
     các mẫu cùng snapshot dùng chung W. Bản v2 lưu W theo từng snapshot
     (mảng [S, N, N]) và mỗi mẫu chỉ giữ snapshot_id để tra ngược.
     => ĐÂY LÀ THAY ĐỔI SCHEMA, cần báo cho Nhật/Trung trước khi merge,
     vì dataloader phía model phải tra W qua snapshot_id thay vì đọc
     thẳng x_weights[i].

Chưa động vào: format Dijkstra theo delay_ms, cách mã hoá path thành
index, cách lưu y_paths_text - giữ y hệt bản gốc để không phá code hai
bạn kia đang dùng cho phần đó.
"""
from __future__ import annotations

import argparse
import random
import sys
from datetime import datetime, timedelta
from pathlib import Path

import networkx as nx
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from skyfield.api import load  # noqa: E402

from src.graph.multilayer_builder import (  # noqa: E402
    ALL_INFRA,
    build_graph_multilayer,
    single_source_relay_dijkstra,
)
from src.physics.air_ground import (  # noqa: E402
    compute_local_links,
    make_uav_patrols,
    nodes_at,
    static_nodes,
)
from src.physics.tle_loader import (  # noqa: E402
    GroundStation,
    compute_gsl_links,
    compute_isl_links,
    load_tle,
    satellite_positions_km,
)

# ---------------------------------------------------------------------------
# Tiện ích mã hoá (giữ nguyên bản gốc)
# ---------------------------------------------------------------------------


def encode_path_to_indices(path: list[str], node_order: list[str]) -> np.ndarray:
    node_to_index = {node: idx for idx, node in enumerate(node_order)}
    return np.array([node_to_index[n] for n in path if n in node_to_index], dtype=np.int32)


def resolve_start_time(ts, start_time: str, sats):
    if start_time == "now":
        return ts.now()
    if start_time == "tle":
        epochs = [float(sat.epoch.tt) for sat in sats if hasattr(sat, "epoch")]
        if epochs:
            return ts.tt_jd(float(np.median(epochs)))
        print("!! Cảnh báo: vệ tinh không có thuộc tính epoch, dùng thời điểm hiện tại.")
        return ts.now()
    start_dt = datetime.fromisoformat(start_time)
    return ts.utc(start_dt.year, start_dt.month, start_dt.day, start_dt.hour, start_dt.minute, start_dt.second)


# ---------------------------------------------------------------------------
# Sinh dataset đa tầng
# ---------------------------------------------------------------------------


def generate_dataset_snapshots(
    tle_path: Path,
    num_snapshots: int = 600,
    time_step_seconds: int = 20,
    num_pairs_per_snapshot: int = 50,
    sample_sat_count: int = 50,
    min_elevation_deg: float = 15.0,
    seed: int = 42,
    start_time: str = "tle",
    sat_select: str = "random",
    infra: tuple[str, ...] = ALL_INFRA,
) -> dict[str, np.ndarray]:
    if not tle_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file TLE: {tle_path}")
    for name, val in [("num_snapshots", num_snapshots), ("time_step_seconds", time_step_seconds),
                       ("num_pairs_per_snapshot", num_pairs_per_snapshot), ("sample_sat_count", sample_sat_count)]:
        if val <= 0:
            raise ValueError(f"{name} phải > 0")
    if sat_select not in ("random", "first"):
        raise ValueError("sat_select phải là 'random' hoặc 'first'")

    rng_sat = random.Random(seed)
    rng_pairs = random.Random(seed + 1)

    ts = load.timescale()
    satellites = load_tle(tle_path)
    if len(satellites) == 0:
        raise ValueError("File TLE không chứa vệ tinh nào.")

    n_take = min(sample_sat_count, len(satellites))
    sample_sats = rng_sat.sample(list(satellites), n_take) if sat_select == "random" else satellites[:n_take]
    t_start = resolve_start_time(ts, start_time, sample_sats)

    # --- Node Air/Ground cố định + UAV chuyển động theo seed riêng ---
    statics = static_nodes()
    patrols = make_uav_patrols(seed=seed)
    # (đã xoá dòng "ground_stations = [...]" ở đây — build lại mỗi snapshot bên dưới, vì UAV di động)

    # --- Thứ tự node THEO KHỐI: space -> air -> ground (không sort toàn cục) ---
    sat_names = sorted({sat.name for sat in sample_sats})
    if len(sat_names) != len(sample_sats):
        print("!! Cảnh báo: có vệ tinh trùng tên trong mẫu, N sẽ nhỏ hơn kỳ vọng.")
    node0 = nodes_at(0.0, statics, patrols)  # chỉ để lấy tên/layer cố định, không dùng vị trí này
    air_names = sorted(n.name for n in node0 if n.layer == "air")
    ground_names = sorted(n.name for n in node0 if n.layer == "ground")
    global_node_order = sat_names + air_names + ground_names
    N = len(global_node_order)
    if N == 0:
        raise ValueError("Không có node nào trong mạng.")
    node_to_index = {node: idx for idx, node in enumerate(global_node_order)}
    node_layer = {n.name: n.layer for n in node0}
    node_layer.update({s: "space" for s in sat_names})
    node_kind = {n.name: n.kind for n in node0}
    node_kind.update({s: "leo" for s in sat_names})

    W_per_snapshot: list[np.ndarray] = []
    x_sources_list, x_targets_list = [], []
    snapshot_id_per_sample = []
    y_paths_indices, y_paths_text = [], []
    stats_rows = []
    kept_snapshots = 0

    print(f"[Oracle Labeler v2] infra={infra} | {num_snapshots} snapshots, bước {time_step_seconds}s")
    print(f" -> N = {N} nodes ({len(sat_names)} sat + {len(air_names)} air + {len(ground_names)} ground)")

    for k in range(num_snapshots):
        current_dt = t_start.utc_datetime() + timedelta(seconds=k * time_step_seconds)
        t_k = ts.utc(current_dt)
        elapsed_s = k * time_step_seconds

        # Vị trí Air/Ground tại thời điểm t_k (chỉ UAV thay đổi theo thời gian)
        nodes_k = nodes_at(elapsed_s, statics, patrols)
        ground_stations_k = [n.to_ground_station() for n in nodes_k]
        gsl_links = compute_gsl_links(sample_sats, ground_stations_k, t_k, min_elevation_deg=min_elevation_deg)
        # Bổ sung GSL cho Air (HAPS/UAV): compute_gsl_links coi mọi ground_stations
        # như nhau, nên ground_stations ở trên đã gồm cả Air (elevation_m = độ cao thật).
        positions = satellite_positions_km(sample_sats, t_k)
        isl_links = compute_isl_links(positions)
        local_links = compute_local_links(nodes_k)

        G = build_graph_multilayer(sat_names, nodes_k, gsl_links, isl_links, local_links, infra=infra)
        G.add_nodes_from(global_node_order)  # đảm bảo đủ N node dù bị cô lập

        if G.number_of_edges() == 0:
            print(f"  [Snapshot {k+1}/{num_snapshots}] Cảnh báo: graph không có cạnh.")
            continue

        W = np.full((N, N), -1.0, dtype=np.float32)  # -1 = không kết nối (thay cho +inf)
        for u, v, data in G.edges(data=True):
            W[node_to_index[u], node_to_index[v]] = data["delay_ms"]
        np.fill_diagonal(W, 0.0)

        endpoints = [n for n in global_node_order if G.nodes[n].get("endpoint")]
        if len(endpoints) < 2:
            print(f"  [Snapshot {k+1}/{num_snapshots}] Cảnh báo: số endpoint < 2.")
            continue

        reachable_pairs = []
        n_pairs_total = 0
        for src in endpoints:
            try:
                distances, paths = single_source_relay_dijkstra(G, src, weight="delay_ms")
            except (nx.NetworkXError, nx.NodeNotFound):
                continue
            for tgt in endpoints:
                if src == tgt:
                    continue
                n_pairs_total += 1
                path = paths.get(tgt)
                if path is None or len(path) < 2:
                    continue
                if not np.isfinite(distances.get(tgt, np.inf)):
                    continue
                reachable_pairs.append((src, tgt, path))

        stats_rows.append((k, n_pairs_total, len(reachable_pairs)))
        if not reachable_pairs:
            print(f"  [Snapshot {k+1}/{num_snapshots}] Cảnh báo: không có cặp endpoint nào có đường đi.")
            continue

        sample_count = min(num_pairs_per_snapshot, len(reachable_pairs))
        selected_pairs = rng_pairs.sample(reachable_pairs, sample_count)

        snap_idx = len(W_per_snapshot)
        W_per_snapshot.append(W)
        for src, tgt, path in selected_pairs:
            onehot_src = np.zeros(N, dtype=np.float32)
            onehot_tgt = np.zeros(N, dtype=np.float32)
            onehot_src[node_to_index[src]] = 1.0
            onehot_tgt[node_to_index[tgt]] = 1.0
            x_sources_list.append(onehot_src)
            x_targets_list.append(onehot_tgt)
            snapshot_id_per_sample.append(snap_idx)
            y_paths_indices.append(encode_path_to_indices(path, global_node_order))
            y_paths_text.append("->".join(path))

        kept_snapshots += 1
        if (k + 1) % 50 == 0 or k == num_snapshots - 1:
            print(f"  [Snapshot {k+1}/{num_snapshots}] Edges={G.number_of_edges()} | "
                  f"Endpoints={len(endpoints)} | Reachable={len(reachable_pairs)}/{n_pairs_total} | "
                  f"Samples={sample_count} | Total={len(x_sources_list)}")

    if len(x_sources_list) == 0:
        raise ValueError("Không tạo được mẫu dữ liệu nào. Kiểm tra TLE, node Air/Ground và luật link.")

    print(f"\n -> Giữ lại {kept_snapshots}/{num_snapshots} snapshots.")
    reach_ratio = np.mean([r / t for _, t, r in stats_rows if t > 0])
    print(f" -> Tỷ lệ endpoint-pair reachable trung bình: {reach_ratio:.2%} (infra={infra})")

    max_path_len = max(len(p) for p in y_paths_indices)
    padded_y_paths = np.full((len(y_paths_indices), max_path_len), -1, dtype=np.int32)
    for i, path in enumerate(y_paths_indices):
        padded_y_paths[i, :len(path)] = path

    return {
        "W_snapshots": np.stack(W_per_snapshot).astype(np.float32),  # [S, N, N]
        "x_sources": np.asarray(x_sources_list, dtype=np.float32),
        "x_targets": np.asarray(x_targets_list, dtype=np.float32),
        "y_paths_indices": padded_y_paths,
        "y_paths_text": np.asarray(y_paths_text, dtype=str),
        "snapshot_id": np.asarray(snapshot_id_per_sample, dtype=np.int32),  # tra vào W_snapshots
        "node_order": np.asarray(global_node_order, dtype=str),
        "node_layer": np.asarray([node_layer[n] for n in global_node_order], dtype=str),
        "node_kind": np.asarray([node_kind[n] for n in global_node_order], dtype=str),
        "reach_stats": np.asarray(stats_rows, dtype=np.int32),  # cols: snapshot_id, n_pairs, n_reachable
        "infra": np.asarray(list(infra), dtype=str),
        "seed": np.asarray([seed]),
    }


def save_dataset_npz(dataset: dict[str, np.ndarray], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **dataset)
    print(f"\n[Oracle Labeler v2] Đã xuất: {output_path}")
    print(f" - Samples: {len(dataset['x_sources'])} | Snapshots giữ lại: {dataset['W_snapshots'].shape[0]}")
    print(f" - N nodes: {dataset['node_order'].shape[0]} | W_snapshots: {dataset['W_snapshots'].shape}")


def main():
    parser = argparse.ArgumentParser(description="Oracle Labeler v2 (Space + Air + Ground)")
    parser.add_argument("--tle", type=str, default="data/tle/starlink.txt")
    parser.add_argument("--out", type=str, default="dataset/oracle_dataset_v2.npz")
    parser.add_argument("--snapshots", type=int, default=600)
    parser.add_argument("--step", type=int, default=20)
    parser.add_argument("--pairs", type=int, default=50)
    parser.add_argument("--sats", type=int, default=150)
    parser.add_argument("--sat-select", type=str, default="random", choices=["random", "first"])
    parser.add_argument("--min-elev", type=float, default=15.0)
    parser.add_argument("--start", type=str, default="tle")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--infra", type=str, default="space,haps,uav",
                         help="Tầng bật thêm ground: vd 'space' | 'space,haps' | 'space,haps,uav' (dùng cho ablation)")
    args = parser.parse_args()

    tle_path = Path(args.tle)
    if not tle_path.exists():
        print(f"!! Không tìm thấy file TLE: '{tle_path}'")
        sys.exit(1)

    infra = tuple(s.strip() for s in args.infra.split(",") if s.strip())
    dataset = generate_dataset_snapshots(
        tle_path=tle_path, num_snapshots=args.snapshots, time_step_seconds=args.step,
        num_pairs_per_snapshot=args.pairs, sample_sat_count=args.sats, min_elevation_deg=args.min_elev,
        seed=args.seed, start_time=args.start, sat_select=args.sat_select, infra=infra,
    )
    save_dataset_npz(dataset, Path(args.out))


if __name__ == "__main__":
    main()
