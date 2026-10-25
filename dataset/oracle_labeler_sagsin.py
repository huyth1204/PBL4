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

from skyfield.api import load, wgs84  # noqa: E402

from src.graph.multilayer_builder import (  # noqa: E402
    ALL_INFRA,
    build_graph_multilayer,
    single_source_relay_dijkstra,
)
from src.physics.air_ground import (  # noqa: E402
    compute_local_links,
    make_placed_uavs,
    make_uav_patrols,
    node_ecef_km,
    nodes_at,
    static_nodes,
)
from src.physics.sites import khoa_sap_xep  # noqa: E402
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


# Cột của bảng liên kết (bước 7 — "bảng liên kết" theo yêu cầu của Huy).
# Lưu thành mảng 2 chiều thay vì object để np.savez/git diff đọc được.
LINK_STAT_COLS = ["snapshot_id", "u", "v", "distance_km", "elevation_deg",
                  "bandwidth_mbps", "packet_loss", "exists"]

# Băng thông / mất gói theo loại liên kết (đồng bộ với multilayer_builder.py
# và graph_builder.py) — dùng để điền bảng liên kết khi graph edge không lưu.
from src.graph.graph_builder import (  # noqa: E402
    BANDWIDTH_GSL_MBPS,
    BANDWIDTH_ISL_MBPS,
    PACKET_LOSS_GSL,
    PACKET_LOSS_ISL,
)
from src.graph.multilayer_builder import BANDWIDTH_ASL_MBPS, PACKET_LOSS_ASL  # noqa: E402

_BW_GSL_OR_ASL = {False: BANDWIDTH_GSL_MBPS, True: BANDWIDTH_ASL_MBPS}
_LOSS_GSL_OR_ASL = {False: PACKET_LOSS_GSL, True: PACKET_LOSS_ASL}
_BW_ISL, _LOSS_ISL = BANDWIDTH_ISL_MBPS, PACKET_LOSS_ISL
# Tập tên nút Air (điền sau khi biết node_order), để tra GSL vs ASL.
_ASL_NAMES: frozenset = frozenset()


def _rows_from_links(snapshot_id: int, node_to_index: dict, gsl_links, isl_links, local_links) -> list[tuple]:
    """Gom mọi liên kết (GSL, ISL, local) của một snapshot thành các hàng số.

    GSL: khoảng cách = slant_range_km, góc ngẩng = elevation_deg.
    ISL : khoảng cách = distance_km, góc ngẩng = nan (chưa mô hình hoá).
    Local (HAPS/UAV-Ground, Air-Air): có cả distance_km lẫn elevation_deg.
    Chỉ ghi liên kết `exists=True` (cột exists luôn = 1) để bảng khỏi phình;
    cặp không tồn tại đã thể hiện bằng W = -1.
    """
    rows: list[tuple] = []

    def add(u, v, dist, elev, bw, loss):
        iu, iv = node_to_index.get(u), node_to_index.get(v)
        if iu is None or iv is None:
            return
        rows.append((snapshot_id, iu, iv, float(dist), float(elev), float(bw), float(loss), 1))

    for link in gsl_links:
        if not link["exists"]:
            continue
        # GSL: ground/air <-> vệ tinh, hai chiều cùng thông số
        bw = _BW_GSL_OR_ASL[link["ground_station"] in _ASL_NAMES]
        loss = _LOSS_GSL_OR_ASL[link["ground_station"] in _ASL_NAMES]
        add(link["ground_station"], link["satellite"], link["slant_range_km"], link["elevation_deg"], bw, loss)
        add(link["satellite"], link["ground_station"], link["slant_range_km"], link["elevation_deg"], bw, loss)
    for link in isl_links:
        if not link["exists"]:
            continue
        add(link["sat_a"], link["sat_b"], link["distance_km"], np.nan, _BW_ISL, _LOSS_ISL)
        add(link["sat_b"], link["sat_a"], link["distance_km"], np.nan, _BW_ISL, _LOSS_ISL)
    for link in local_links:
        if not link["exists"]:
            continue
        add(link["node_a"], link["node_b"], link["distance_km"], link["elevation_deg"],
            link["bandwidth_mbps"], link["packet_loss"])
        add(link["node_b"], link["node_a"], link["distance_km"], link["elevation_deg"],
            link["bandwidth_mbps"], link["packet_loss"])
    return rows


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
    mo_rong: bool = False,
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
    statics = static_nodes(mo_rong=mo_rong)
    patrols = make_uav_patrols(seed=seed) + (make_placed_uavs() if mo_rong else [])
    # (đã xoá dòng "ground_stations = [...]" ở đây — build lại mỗi snapshot bên dưới, vì UAV di động)

    # --- Thứ tự node THEO KHỐI: space -> air -> ground (không sort toàn cục) ---
    # Sort bằng `khoa_sap_xep`: nút cấu hình cũ luôn đứng trước, nút mở rộng nối
    # đuôi. Với cấu hình cũ, khóa này tương đương sorted(tên) => mạng 73 nút không đổi.
    sat_names = sorted({sat.name for sat in sample_sats})
    if len(sat_names) != len(sample_sats):
        print("!! Cảnh báo: có vệ tinh trùng tên trong mẫu, N sẽ nhỏ hơn kỳ vọng.")
    node0 = nodes_at(0.0, statics, patrols)  # chỉ để lấy tên/layer cố định, không dùng vị trí này
    air_names = sorted((n.name for n in node0 if n.layer == "air"), key=khoa_sap_xep)
    ground_names = sorted((n.name for n in node0 if n.layer == "ground"), key=khoa_sap_xep)
    # --- Thứ tự node: nút CŨ giữ nguyên tuyệt đối, nút MỚI nối vào CUỐI ---
    # Sort theo khối (space -> air -> ground) chỉ đúng khi các khối có kích
    # thước KHÔNG ĐỔI. Nếu chỉ append nút mới vào cuối từng khối thì khối air
    # phình từ 10 -> 14, đẩy TOÀN BỘ khối ground lệch đi 4 bậc => phá hợp đồng
    # "chạy lại cấu hình cũ ra đúng mạng 73 nút" (chỉ số nút không còn khớp v2).
    # Vì vậy: dựng order CHỈ từ nút cũ (đúng như v2) rồi nối nút mới vào cuối.
    # Nút cũ luôn ở đúng chỉ số của nó trong v2 => model cũ dùng được.
    # Lưu ý: vệ tinh không nằm trong CAC_TEN_CU (tên lấy từ TLE), nên nhóm
    # "nút cũ" xét theo layer: space luôn cũ; air/ground dùng khóa khoa_sap_xep.
    cu = sorted((n.name for n in node0 if khoa_sap_xep(n.name)[0] == 0), key=khoa_sap_xep)
    moi = sorted((n.name for n in node0 if khoa_sap_xep(n.name)[0] == 1), key=khoa_sap_xep)
    air_set = {n.name for n in node0 if n.layer == "air"}
    ground_set = {n.name for n in node0 if n.layer == "ground"}
    order_cu = (list(sat_names)                                        # space: luôn cũ
                + [n for n in cu if n in air_set]
                + [n for n in cu if n in ground_set])
    global_node_order = order_cu + moi

    global _ASL_NAMES
    _ASL_NAMES = frozenset(air_names)  # GSL cho nút air dùng băng thông ASL
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
    latlon_alt_per_snapshot: list[np.ndarray] = []   # [S, N, 3] lat/lon/alt_km — khôi phục như v1
    link_stats_rows: list[tuple] = []                 # bảng liên kết tổng hợp mỗi snapshot
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

        # --- Tọa độ (lat, lon, alt_km) theo snapshot: khôi phục node_latlon_alt
        # như v1 (600, 53, 3). Air/Ground lấy từ nodes_k; vệ tinh lấy subpoint
        # của skyfield (wgs84) tại thời điểm t_k.
        latlon_alt = np.full((N, 3), np.nan, dtype=np.float32)
        for n in nodes_k:
            latlon_alt[node_to_index[n.name]] = (n.lat_deg, n.lon_deg, n.alt_m / 1000.0)
        for sat in sample_sats:
            i = node_to_index.get(sat.name)
            if i is None:
                continue
            sp = wgs84.subpoint_of(sat.at(t_k))
            latlon_alt[i] = (sp.latitude.degrees, sp.longitude.degrees, sp.elevation.km)
        latlon_alt_per_snapshot.append(latlon_alt)

        # --- Bảng liên kết: (snapshot, u, v, khoảng_cách_km, góc_ngẩng, băng thông,
        # mất gói, tồn tại). Lấy từ gsl_links/isl_links/local_links vì graph edge
        # không lưu khoảng cách/góc ngẩng. GSL có slant_range_km; ISL/local có
        # distance_km; góc ngẩng chỉ có ở GSL và local.
        link_stats_rows.extend(_rows_from_links(k, node_to_index, gsl_links, isl_links, local_links))

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
        # --- Schema v3: tọa độ theo snapshot + bảng liên kết (thêm mới, không bỏ key cũ)
        "node_latlon_alt": np.stack(latlon_alt_per_snapshot).astype(np.float32),  # [S, N, 3] lat/lon/alt_km
        "link_stats": np.asarray(link_stats_rows, dtype=np.float64) if link_stats_rows
                      else np.zeros((0, len(LINK_STAT_COLS)), dtype=np.float64),
        "link_stat_cols": np.asarray(LINK_STAT_COLS, dtype=str),
        "mo_rong": np.asarray([bool(mo_rong)]),
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
    parser.add_argument("--sats", type=int, default=50)
    parser.add_argument("--sat-select", type=str, default="random", choices=["random", "first"])
    parser.add_argument("--min-elev", type=float, default=15.0)
    parser.add_argument("--start", type=str, default="tle")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--infra", type=str, default="space,haps,uav",
                         help="Tầng bật thêm ground: vd 'space' | 'space,haps' | 'space,haps,uav' (dùng cho ablation)")
    parser.add_argument("--mo-rong", action="store_true",
                         help="Bật mạng mở rộng: thêm nút đảo/núi/vùng lũ + HAPS/UAV mới (nối đuôi, nút cũ không đổi)")
    # Console Windows mặc định cp1252 => in tiếng Việt là UnicodeEncodeError.
    # Ép stdout/stderr sang UTF-8 trước khi in bất cứ thứ gì.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    args = parser.parse_args()

    tle_path = Path(args.tle)
    if not tle_path.exists():
        print(f"!! Không tìm thấy file TLE: '{tle_path}'")
        sys.exit(1)

    # Mạng mở rộng KHÔNG BAO GIỜ ghi đè v2 — bản v2 là mốc hồi quy 73 nút.
    out_path = Path(args.out)
    if args.mo_rong and out_path.resolve() == (Path(__file__).resolve().parent / "oracle_dataset_v2.npz"):
        print("!! --mo-rong không được ghi vào oracle_dataset_v2.npz (mốc hồi quy). "
              "Dùng --out dataset/oracle_dataset_v3.npz.")
        sys.exit(1)

    infra = tuple(s.strip() for s in args.infra.split(",") if s.strip())
    dataset = generate_dataset_snapshots(
        tle_path=tle_path, num_snapshots=args.snapshots, time_step_seconds=args.step,
        num_pairs_per_snapshot=args.pairs, sample_sat_count=args.sats, min_elevation_deg=args.min_elev,
        seed=args.seed, start_time=args.start, sat_select=args.sat_select, infra=infra,
        mo_rong=args.mo_rong,
    )
    save_dataset_npz(dataset, out_path)


if __name__ == "__main__":
    main()
