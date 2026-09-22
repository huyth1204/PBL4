from __future__ import annotations

import argparse
import json
import sys
from datetime import timedelta
from pathlib import Path

import numpy as np
import plotly.graph_objects as go


# ============================================================
# Paths / dataset
# ============================================================


def project_root() -> Path:
    return Path(__file__).resolve().parent


def find_default_dataset() -> Path:
    """Tự tìm dataset mới, không phụ thuộc thư mục chạy lệnh."""
    root = project_root()
    dataset_dir = root / "dataset"

    # Ưu tiên Oracle v2, sau đó tới các file đã split.
    candidates = [
        dataset_dir / "oracle_dataset_v2.npz",
        dataset_dir / "val.npz",
        dataset_dir / "test.npz",
        dataset_dir / "train.npz",
    ]

    for path in candidates:
        if path.exists():
            return path

    npz_files = sorted(dataset_dir.glob("*.npz"))
    if npz_files:
        return npz_files[0]

    raise FileNotFoundError(
        f"Không tìm thấy dataset .npz trong: {dataset_dir}\n"
        "Hãy kiểm tra thư mục dataset/ hoặc dùng --data <file.npz>."
    )


def load_npz(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as raw:
        data = {k: raw[k] for k in raw.files}
    data["_name"] = path.stem.lower()
    return data


def load_split_info(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def is_split_dataset(data: dict) -> bool:
    return data.get("_name", "") in {"train", "val", "test"}


def original_snapshot_id(data: dict, local_sid: int, split_info: dict) -> int:
    """Đổi snapshot_id cục bộ của train/val/test về ID trong oracle gốc."""
    if is_split_dataset(data):
        ids = split_info.get("snapshot_ids", {}).get(data["_name"], [])
        if 0 <= int(local_sid) < len(ids):
            return int(ids[int(local_sid)])
    return int(local_sid)


# ============================================================
# Node metadata
# ============================================================


def infer_layers(data: dict) -> np.ndarray:
    if "node_layer" in data:
        return data["node_layer"].astype(str)

    names = data["node_order"].astype(str)
    known_ground = {
        "Hanoi", "DaNang", "HoChiMinh", "HaiPhong", "CanTho",
        "Vinh", "Hue", "NhaTrang", "QuyNhon", "BuonMaThuot",
        "CaMau", "LaoCai", "PleiKu",
    }
    return np.array(["ground" if n in known_ground else "space" for n in names])


def infer_kinds(data: dict) -> np.ndarray:
    if "node_kind" in data:
        return data["node_kind"].astype(str)

    layers = infer_layers(data)
    names = data["node_order"].astype(str)
    return np.where(layers == "space", "leo", np.where(
        np.isin(names, ["Hanoi", "DaNang", "HoChiMinh", "HaiPhong", "CanTho"]),
        "gateway", "user"
    ))


def classify_link(layer_a: str, layer_b: str) -> str:
    if layer_a == "space" and layer_b == "space":
        return "ISL"
    if layer_a == "space" or layer_b == "space":
        return "GSL"
    return "LOCAL"


# ============================================================
# Reconstruct coordinates for Oracle v2
# ============================================================


def reconstruct_positions(
    node_order: np.ndarray,
    node_layer: np.ndarray,
    original_sid: int,
    tle_path: Path,
    seed: int = 42,
    step_seconds: int = 20,
) -> tuple[np.ndarray, str]:
    """
    Oracle v2 không lưu lat/lon của node.
    Hàm này tái tạo lại đúng trạng thái theo:
      - TLE hiện tại
      - seed của oracle
      - snapshot index * 20s
    """
    root = project_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from skyfield.api import load
    from src.physics.air_ground import make_uav_patrols, nodes_at, static_nodes
    from src.physics.tle_loader import load_tle

    satellites = load_tle(tle_path)
    sat_by_name = {sat.name: sat for sat in satellites}

    node_order = np.asarray(node_order).astype(str)
    node_layer = np.asarray(node_layer).astype(str)
    sat_names = [
        name for name, layer in zip(node_order, node_layer)
        if layer == "space"
    ]

    missing = [name for name in sat_names if name not in sat_by_name]
    if missing:
        raise ValueError(
            "TLE hiện tại không chứa các satellite trong dataset. "
            f"Ví dụ thiếu: {', '.join(missing[:5])}"
        )

    # Chỉ cần đúng tập satellite, thứ tự không ảnh hưởng median epoch.
    sample_sats = [sat_by_name[name] for name in sat_names]
    epochs = [float(sat.epoch.tt) for sat in sample_sats if hasattr(sat, "epoch")]

    ts = load.timescale()
    if epochs:
        t_start = ts.tt_jd(float(np.median(epochs)))
    else:
        t_start = ts.now()

    start_dt = t_start.utc_datetime()
    current_dt = start_dt + timedelta(seconds=int(original_sid) * step_seconds)

    t = ts.utc(
        current_dt.year,
        current_dt.month,
        current_dt.day,
        current_dt.hour,
        current_dt.minute,
        current_dt.second + current_dt.microsecond / 1e6,
    )

    positions: dict[str, tuple[float, float, float]] = {}

    # Dùng subpoint của satellite để vẽ trên bản đồ 2D.
    for sat in sample_sats:
        subpoint = sat.at(t).subpoint()
        positions[sat.name] = (
            float(subpoint.latitude.degrees),
            float(subpoint.longitude.degrees),
            float(subpoint.elevation.km),
        )

    # Air + Ground: phải dùng cùng seed + elapsed_s như oracle.
    statics = static_nodes()
    patrols = make_uav_patrols(seed=seed)
    elapsed_s = int(original_sid) * step_seconds
    nodes = nodes_at(elapsed_s, statics, patrols)

    for node in nodes:
        positions[node.name] = (
            float(node.lat_deg),
            float(node.lon_deg),
            float(node.alt_m / 1000.0),
        )

    coords = []
    for name in node_order:
        if name not in positions:
            raise ValueError(f"Không tái tạo được tọa độ cho node: {name}")
        coords.append(positions[name])

    timestamp = current_dt.isoformat(timespec="seconds") + "Z"
    return np.asarray(coords, dtype=np.float64), timestamp


def get_coordinates(
    data: dict,
    local_sid: int,
    original_sid: int,
    tle_path: Path,
    seed: int,
    step_seconds: int,
) -> tuple[np.ndarray, str]:
    """Dùng tọa độ lưu sẵn nếu dataset có; Oracle v2 thì tái tạo."""
    if "node_latlon_alt" in data:
        arr = np.asarray(data["node_latlon_alt"])
        sid = int(local_sid)
        if arr.ndim == 3 and 0 <= sid < arr.shape[0]:
            stamp = ""
            if "snapshot_times" in data and sid < len(data["snapshot_times"]):
                stamp = str(data["snapshot_times"][sid])
            return arr[sid].astype(np.float64), stamp

    if "node_layer" not in data:
        raise ValueError(
            "Dataset không có node_latlon_alt và cũng không có node_layer; "
            "không thể xác định vị trí node."
        )

    return reconstruct_positions(
        data["node_order"],
        data["node_layer"],
        original_sid,
        tle_path,
        seed=seed,
        step_seconds=step_seconds,
    )


# ============================================================
# Geographic helpers
# ============================================================


def split_geo_line(lons, lats, jump=180.0):
    """Tách line khi đi qua kinh tuyến +/-180 để không kéo ngang cả bản đồ."""
    out_lon, out_lat = [], []
    for i, (lon, lat) in enumerate(zip(lons, lats)):
        if i > 0 and abs(float(lon) - float(lons[i - 1])) > jump:
            out_lon.append(None)
            out_lat.append(None)
        out_lon.append(float(lon))
        out_lat.append(float(lat))
    return out_lon, out_lat


def add_link_traces(fig: go.Figure, W: np.ndarray, coords: np.ndarray, layers: np.ndarray):
    """Vẽ mỗi physical link đúng 1 lần dù W có hai chiều."""
    traces = {}
    for kind in ("ISL", "GSL", "LOCAL"):
        traces[kind] = len(fig.data)
        fig.add_trace(go.Scattergeo(
            lon=[], lat=[], mode="lines", name=kind,
            hoverinfo="skip", visible=True,
        ))

    lon_buf = {k: [] for k in traces}
    lat_buf = {k: [] for k in traces}

    n = W.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            # W dùng -1 = không có link, 0 = đường chéo.
            if W[i, j] <= 0 and W[j, i] <= 0:
                continue

            typ = classify_link(str(layers[i]), str(layers[j]))
            lons, lats = split_geo_line(
                [coords[i, 1], coords[j, 1]],
                [coords[i, 0], coords[j, 0]],
            )
            lon_buf[typ].extend(lons + [None])
            lat_buf[typ].extend(lats + [None])

    styles = {
        "ISL": dict(width=1.2),
        "GSL": dict(width=2.2),
        "LOCAL": dict(width=2.6),
    }
    for typ, idx in traces.items():
        fig.data[idx].lon = lon_buf[typ]
        fig.data[idx].lat = lat_buf[typ]
        fig.data[idx].line = styles[typ]

    return traces


# ============================================================
# Figure
# ============================================================


def build_figure(
    data: dict,
    sample_idx: int,
    snapshot_arg: int | None,
    tle_path: Path,
    split_info: dict,
    seed: int,
    step_seconds: int,
) -> tuple[go.Figure, int, int, list[int], str]:
    node_order = data["node_order"].astype(str)
    layers = infer_layers(data)
    kinds = infer_kinds(data)
    n = len(node_order)

    if "W_snapshots" not in data:
        raise ValueError("Dataset mới phải có W_snapshots.")
    if "snapshot_id" not in data:
        raise ValueError("Dataset thiếu snapshot_id.")
    if "y_paths_indices" not in data:
        raise ValueError("Dataset thiếu y_paths_indices.")

    W_all = np.asarray(data["W_snapshots"], dtype=np.float32)
    if W_all.ndim != 3 or W_all.shape[1:] != (n, n):
        raise ValueError(
            f"W_snapshots có shape {W_all.shape}, nhưng node_order có N={n}."
        )

    sample_count = len(data["y_paths_indices"])
    if not 0 <= sample_idx < sample_count:
        raise IndexError(f"sample={sample_idx}; dataset có {sample_count} samples.")

    # Nếu chỉ định snapshot, lấy sample đầu tiên thuộc snapshot đó.
    if snapshot_arg is not None:
        matches = np.where(data["snapshot_id"] == int(snapshot_arg))[0]
        if len(matches) == 0:
            raise ValueError(
                f"Không có sample thuộc snapshot_id={snapshot_arg} trong dataset này."
            )
        sample_idx = int(matches[0])

    local_sid = int(data["snapshot_id"][sample_idx])
    if not 0 <= local_sid < len(W_all):
        raise IndexError(
            f"snapshot_id={local_sid} ngoài W_snapshots ({len(W_all)} snapshots)."
        )

    original_sid = original_snapshot_id(data, local_sid, split_info)
    W = W_all[local_sid]

    coords, timestamp = get_coordinates(
        data,
        local_sid,
        original_sid,
        tle_path,
        seed,
        step_seconds,
    )

    # Route.
    path = [
        int(x) for x in data["y_paths_indices"][sample_idx]
        if int(x) >= 0 and int(x) < n
    ]
    if len(path) < 2:
        raise ValueError(f"Sample {sample_idx} không có route hợp lệ.")

    fig = go.Figure()
    link_traces = add_link_traces(fig, W, coords, layers)

    # ---------- Satellite ----------
    space_idx = np.where(layers == "space")[0].tolist()
    route_set = set(path)
    route_sat = [i for i in space_idx if i in route_set]
    other_sat = [i for i in space_idx if i not in route_set]

    sat_trace = len(fig.data)
    fig.add_trace(go.Scattergeo(
        lon=coords[other_sat, 1],
        lat=coords[other_sat, 0],
        mode="markers",
        marker=dict(size=4, opacity=0.72),
        name="Satellite",
        text=node_order[other_sat],
        customdata=np.c_[node_order[other_sat], np.round(coords[other_sat, 2], 1)],
        hovertemplate="%{customdata[0]}<br>Altitude: %{customdata[1]} km<extra></extra>",
    ))

    route_sat_trace = len(fig.data)
    fig.add_trace(go.Scattergeo(
        lon=coords[route_sat, 1] if route_sat else [],
        lat=coords[route_sat, 0] if route_sat else [],
        mode="markers",
        marker=dict(size=9, symbol="circle"),
        name="Satellite trên route",
        text=node_order[route_sat] if route_sat else [],
        hovertemplate="%{text}<extra></extra>",
    ))

    # ---------- Air ----------
    air_idx = np.where(layers == "air")[0].tolist()
    air_route = [i for i in air_idx if i in route_set]
    air_other = [i for i in air_idx if i not in route_set]

    air_trace = len(fig.data)
    fig.add_trace(go.Scattergeo(
        lon=coords[air_other, 1] if air_other else [],
        lat=coords[air_other, 0] if air_other else [],
        mode="markers",
        marker=dict(size=8, symbol="triangle-up"),
        name="Air (HAPS/UAV)",
        text=node_order[air_other] if air_other else [],
        customdata=kinds[air_other] if air_other else [],
        hovertemplate="%{text}<br>Kind: %{customdata}<extra></extra>",
    ))

    air_route_trace = len(fig.data)
    fig.add_trace(go.Scattergeo(
        lon=coords[air_route, 1] if air_route else [],
        lat=coords[air_route, 0] if air_route else [],
        mode="markers",
        marker=dict(size=12, symbol="triangle-up"),
        name="Air trên route",
        text=node_order[air_route] if air_route else [],
        hovertemplate="%{text}<extra></extra>",
    ))

    # ---------- Ground ----------
    ground_idx = np.where(layers == "ground")[0].tolist()
    ground_trace = len(fig.data)
    fig.add_trace(go.Scattergeo(
        lon=coords[ground_idx, 1] if ground_idx else [],
        lat=coords[ground_idx, 0] if ground_idx else [],
        mode="markers+text",
        marker=dict(size=9, symbol="square"),
        text=node_order[ground_idx] if ground_idx else [],
        textposition="top center",
        name="Ground",
        customdata=kinds[ground_idx] if ground_idx else [],
        hovertemplate="%{text}<br>Kind: %{customdata}<extra></extra>",
    ))

    # ---------- Selected route ----------
    route_lon, route_lat = split_geo_line(
        coords[path, 1].tolist(),
        coords[path, 0].tolist(),
    )
    route_trace = len(fig.data)
    fig.add_trace(go.Scattergeo(
        lon=route_lon,
        lat=route_lat,
        mode="lines",
        line=dict(width=6),
        name="Selected route",
        hoverinfo="skip",
    ))

    route_num_trace = len(fig.data)
    fig.add_trace(go.Scattergeo(
        lon=coords[path, 1],
        lat=coords[path, 0],
        mode="markers+text",
        marker=dict(size=11, symbol="circle"),
        text=[str(i + 1) for i in range(len(path))],
        textposition="middle center",
        customdata=node_order[path],
        name="Route nodes",
        hovertemplate="%{customdata}<br>Hop %{text}<extra></extra>",
    ))

    # ---------- View ----------
    # Chỉ lấy Air + Ground + node route không phải satellite để tính khung nhìn,
    # tránh vệ tinh trải khắp toàn cầu làm bản đồ zoom quá rộng.
    route_non_space = [i for i in path if i not in space_idx]
    focus_idx = sorted(set(route_non_space) | set(air_idx) | set(ground_idx))
    if not focus_idx:
        focus_idx = sorted(set(path)) or list(range(n))

    focus_lons = coords[focus_idx, 1]
    focus_lats = coords[focus_idx, 0]
    lon_min, lon_max = float(np.min(focus_lons)), float(np.max(focus_lons))
    lat_min, lat_max = float(np.min(focus_lats)), float(np.max(focus_lats))

    lon_span = lon_max - lon_min
    lat_span = lat_max - lat_min

    # Ép khung theo landscape (rộng > cao), luôn giới hạn trong [-180,180]/[-90,90].
    lon_pad = max(12.0, lon_span * 0.18)
    lat_pad = max(4.0, lat_span * 0.10)
    lon_lo = max(-180.0, lon_min - lon_pad)
    lon_hi = min(180.0, lon_max + lon_pad)
    lat_lo = max(-90.0, lat_min - lat_pad)
    lat_hi = min(90.0, lat_max + lat_pad)

    # Tối thiểu khoảng 70° kinh độ và 32° vĩ độ -> bản đồ nằm ngang.
    cur_lon_span = lon_hi - lon_lo
    if cur_lon_span < 70.0:
        c = (lon_min + lon_max) / 2.0
        lon_lo = max(-180.0, c - 35.0)
        lon_hi = min(180.0, c + 35.0)

    cur_lat_span = lat_hi - lat_lo
    if cur_lat_span < 32.0:
        c = (lat_min + lat_max) / 2.0
        lat_lo = max(-90.0, c - 16.0)
        lat_hi = min(90.0, c + 16.0)

    # Nếu vẫn quá rộng, thu hẹp về vùng trung tâm thay vì zoom toàn cầu.
    if lon_span > 140:
        center_lon = float(np.mean(focus_lons))
        lon_lo = max(-180.0, center_lon - 70.0)
        lon_hi = min(180.0, center_lon + 70.0)

    if lat_span > 110:
        center_lat = float(np.mean(focus_lats))
        lat_lo = max(-90.0, center_lat - 55.0)
        lat_hi = min(90.0, center_lat + 55.0)

    route_text = " → ".join(node_order[path].tolist())
    layer_counts = {
        layer: int(np.sum(layers == layer)) for layer in np.unique(layers)
    }
    kind_counts = {
        kind: int(np.sum(kinds == kind)) for kind in np.unique(kinds)
    }

    fig.update_geos(
        domain=dict(x=[0.01, 0.82], y=[0.10, 0.88]),
        projection_type="equirectangular",
        projection_scale=1.0,
        showland=True,
        showocean=True,
        bgcolor="rgba(245,248,252,1)",
        showcountries=True,
        showcoastlines=True,
        coastlinecolor="black",
        fitbounds=False,
        lonaxis_range=[lon_lo, lon_hi],
        lataxis_range=[lat_lo, lat_hi],
        center=dict(
            lat=(lat_lo + lat_hi) / 2,
            lon=(lon_lo + lon_hi) / 2,
        ),
    )

    # ---------- Buttons ----------
    all_links = list(link_traces.values())
    all_nodes = [
        sat_trace, route_sat_trace,
        air_trace, air_route_trace,
        ground_trace,
        route_trace, route_num_trace,
    ]
    full = all_links + all_nodes
    route_only = [route_sat_trace, air_route_trace, ground_trace, route_trace, route_num_trace]
    network_only = all_links + [sat_trace, air_trace, ground_trace]
    clean = [sat_trace, route_sat_trace, air_trace, air_route_trace, ground_trace]

    def visibility(indices):
        visible = [False] * len(fig.data)
        for idx in indices:
            visible[idx] = True
        return visible

    fig.update_layout(
        # Mặc định Plotly là dragmode="zoom" (khung chọn vùng zoom kiểu Cartesian),
        # áp lên bản đồ geo bị lệch tọa độ tạo khung đỏ ngoằn ngoèo. Đổi sang "pan"
        # để kéo chuột di chuyển bản đồ đúng như bản đồ thường.
        dragmode="pan",
        title=dict(
            text=(
                f"SAGSIN Network Visualization — sample {sample_idx} "
                f"| snapshot {local_sid}"
            ),
            x=0.02, xanchor="left", y=0.985, yanchor="top",
            font=dict(size=18),
        ),
        height=690,
        margin=dict(l=12, r=12, t=68, b=12),
        paper_bgcolor="white",
        plot_bgcolor="white",
        geo=dict(domain=dict(x=[0.01, 0.82], y=[0.10, 0.88])),
        updatemenus=[
            dict(
                type="buttons", direction="right",
                x=0.02, y=1.035,
                xanchor="left", yanchor="top",
                pad=dict(r=6, t=2),
                buttons=[
                    dict(label="Full topology", method="restyle", args=[{"visible": visibility(full)}]),
                    dict(label="Route only", method="restyle", args=[{"visible": visibility(route_only)}]),
                    dict(label="Network only", method="restyle", args=[{"visible": visibility(network_only)}]),
                    dict(label="Clean", method="restyle", args=[{"visible": visibility(clean)}]),
                ],
            ),
        ],
        legend=dict(
            orientation="v",
            x=0.835, y=0.94,
            xanchor="left", yanchor="top",
            bgcolor="rgba(255,255,255,0.94)",
            bordercolor="rgba(120,130,140,0.35)",
            borderwidth=1,
            font=dict(size=12),
            itemclick="toggle",
            itemdoubleclick="toggleothers",
            tracegroupgap=7,
        ),
        annotations=[
            dict(
                x=0.835, y=0.55,
                xref="paper", yref="paper",
                xanchor="left", yanchor="top",
                showarrow=False, align="left",
                width=250,
                bgcolor="rgba(248,250,252,0.96)",
                bordercolor="rgba(120,130,140,0.25)",
                borderwidth=1, borderpad=10,
                text=(
                    f"<b>NETWORK</b><br>"
                    f"Nodes: {n}<br>"
                    f"Space: {layer_counts.get('space', 0)}<br>"
                    f"Air: {layer_counts.get('air', 0)}<br>"
                    f"Ground: {layer_counts.get('ground', 0)}<br><br>"
                    f"<b>NODE TYPES</b><br>"
                    + "<br>".join(f"{k}: {v}" for k, v in kind_counts.items())
                    + "<br><br>"
                    f"<b>ROUTE</b><br>"
                    f"Sample: {sample_idx}<br>"
                    f"Snapshot: {local_sid}<br>"
                    + (f"Original snapshot: {original_sid}<br>" if original_sid != local_sid else "")
                    + f"Hops: {len(path) - 1}<br><br>"
                    f"<b>PATH</b><br>{route_text.replace(' → ', ' →<br>')}<br><br>"
                    f"<b>TIME</b><br>{timestamp or 'reconstructed from TLE'}"
                ),
                font=dict(size=10),
            )
        ],
    )

    return fig, local_sid, original_sid, path, timestamp


# ============================================================
# CLI
# ============================================================


def main():
    parser = argparse.ArgumentParser(
        description="Visualization cho Oracle SAGSIN v2 (173 node / W_snapshots)."
    )
    parser.add_argument("--data", type=str, default=None)
    parser.add_argument("--sample", type=int, default=0)
    parser.add_argument("--snapshot", type=int, default=None,
                        help="snapshot_id trong dataset đang mở; lấy sample đầu tiên của snapshot đó")
    parser.add_argument("--output", type=str, default="map.html")
    parser.add_argument("--tle", type=str, default=None)
    parser.add_argument("--split-info", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42,
                        help="seed phải trùng seed khi Oracle tạo UAV")
    parser.add_argument("--step", type=int, default=20,
                        help="khoảng thời gian giữa hai snapshot, mặc định 20 giây")
    args = parser.parse_args()

    root = project_root()
    data_path = Path(args.data) if args.data else find_default_dataset()
    if not data_path.is_absolute():
        data_path = root / data_path
    data_path = data_path.resolve()

    if not data_path.exists():
        raise FileNotFoundError(f"Không tìm thấy dataset: {data_path}")

    data = load_npz(data_path)
    required = {"W_snapshots", "x_sources", "x_targets", "y_paths_indices", "snapshot_id", "node_order"}
    missing = required - set(data.keys())
    if missing:
        raise ValueError(f"Dataset thiếu trường: {sorted(missing)}")

    tle_path = Path(args.tle) if args.tle else root / "data" / "tle" / "starlink.txt"
    if not tle_path.is_absolute():
        tle_path = root / tle_path
    tle_path = tle_path.resolve()

    # Chỉ cần TLE khi dataset không lưu lat/lon.
    if "node_latlon_alt" not in data and not tle_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy TLE: {tle_path}\n"
            "Oracle v2 không lưu tọa độ nên visualization cần TLE để tái tạo vị trí."
        )

    split_info_path = (
        Path(args.split_info) if args.split_info else root / "dataset" / "split_info.json"
    )
    if not split_info_path.is_absolute():
        split_info_path = root / split_info_path
    split_info = load_split_info(split_info_path.resolve())

    fig, local_sid, original_sid, path, timestamp = build_figure(
        data=data,
        sample_idx=args.sample,
        snapshot_arg=args.snapshot,
        tle_path=tle_path,
        split_info=split_info,
        seed=args.seed,
        step_seconds=args.step,
    )

    # Lấy lại giới hạn bản đồ đã tính trong build_figure (biến cục bộ, không truy cập trực tiếp được).
    lon_lo, lon_hi = [float(x) for x in fig.layout.geo.lonaxis.range]
    lat_lo, lat_hi = [float(x) for x in fig.layout.geo.lataxis.range]

    output = Path(args.output)
    if not output.is_absolute():
        output = root / output
    output.parent.mkdir(parents=True, exist_ok=True)

    # Biên độ pan/zoom cho phép tính từ vị trí gốc (đo lại đúng giá trị gốc ở
    # JS, không đoán trước, vì scale nội bộ của Plotly có thể khác 1.0 khi đã
    # set sẵn lonaxis_range/lataxis_range).
    pan_lon_span = 60.0
    pan_lat_span = 30.0
    safe_lon_margin = 15.0  # không bao giờ pan tới sát kinh tuyến +/-180 (equirectangular không nối vòng)
    safe_lat_margin = 10.0  # không bao giờ pan tới sát cực +/-90
    scale_min_ratio = 0.5   # zoom ra tối đa gấp đôi so với ban đầu
    scale_max_ratio = 6.0   # zoom vào tối đa gấp 6 lần so với ban đầu

    post_script = f"""
const gd = document.getElementById('{{plot_id}}');
let home = null; // trạng thái gốc, đo đúng 1 lần sau khi render xong

const PAN_LON_SPAN = {pan_lon_span};
const PAN_LAT_SPAN = {pan_lat_span};
const SAFE_LON_MARGIN = {safe_lon_margin};
const SAFE_LAT_MARGIN = {safe_lat_margin};
const SCALE_MIN_RATIO = {scale_min_ratio};
const SCALE_MAX_RATIO = {scale_max_ratio};

function clampNum(v, lo, hi) {{
    return Math.max(lo, Math.min(hi, v));
}}

gd.on('plotly_afterplot', function() {{
    if (home) return;
    const geo = gd.layout.geo;
    if (!geo || !geo.center || !geo.projection) return;
    home = {{
        lon: Number(geo.center.lon),
        lat: Number(geo.center.lat),
        scale: Number(geo.projection.scale) || 1,
        lonRange: geo.lonaxis.range.slice(),
        latRange: geo.lataxis.range.slice(),
    }};
}});

let debounceTimer = null;

// Kéo/zoom dùng transform nội bộ riêng, chỉnh giữa chừng sẽ bị thao tác đang
// diễn ra ghi đè ngay. Nên chỉ chỉnh SAU KHI thao tác đã dừng hẳn (debounce).
function applyClamp() {{
    if (!home) return;
    const geo = gd.layout.geo;
    if (!geo || !geo.center || !geo.projection) return;

    const lonLo = clampNum(home.lon - PAN_LON_SPAN, -180 + SAFE_LON_MARGIN, 180 - SAFE_LON_MARGIN);
    const lonHi = clampNum(home.lon + PAN_LON_SPAN, -180 + SAFE_LON_MARGIN, 180 - SAFE_LON_MARGIN);
    const latLo = clampNum(home.lat - PAN_LAT_SPAN, -90 + SAFE_LAT_MARGIN, 90 - SAFE_LAT_MARGIN);
    const latHi = clampNum(home.lat + PAN_LAT_SPAN, -90 + SAFE_LAT_MARGIN, 90 - SAFE_LAT_MARGIN);

    const lon = clampNum(Number(geo.center.lon), lonLo, lonHi);
    const lat = clampNum(Number(geo.center.lat), latLo, latHi);
    const scale = clampNum(
        Number(geo.projection.scale),
        home.scale * SCALE_MIN_RATIO,
        home.scale * SCALE_MAX_RATIO
    );

    const lonBad = Math.abs(lon - Number(geo.center.lon)) > 1e-6;
    const latBad = Math.abs(lat - Number(geo.center.lat)) > 1e-6;
    const scaleBad = Math.abs(scale - Number(geo.projection.scale)) > 1e-6;

    if (lonBad || latBad || scaleBad) {{
        // Đặt lại range gốc mỗi lần chỉnh để tránh bản đồ bị vỡ/mất góc do kéo quá đà.
        Plotly.relayout(gd, {{
            'geo.center.lon': lon,
            'geo.center.lat': lat,
            'geo.projection.scale': scale,
            'geo.lonaxis.range': home.lonRange,
            'geo.lataxis.range': home.latRange,
        }});
    }}
}}

gd.on('plotly_relayout', function() {{
    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(applyClamp, 150);
}});
"""
    fig.write_html(
        str(output),
        include_plotlyjs=True,
        post_script=post_script,
        config={"scrollZoom": True, "doubleClick": "reset"},
    )

    names = data["node_order"].astype(str)
    print(f"Dataset: {data_path}")
    print(f"Nodes: {len(names)}")
    print(f"Sample: {args.sample if args.snapshot is None else 'first sample of snapshot ' + str(args.snapshot)}")
    print(f"Snapshot local: {local_sid}")
    print(f"Snapshot original: {original_sid}")
    print("Route: " + " -> ".join(names[path]))
    print(f"Đã tạo: {output}")


if __name__ == "__main__":
    main()