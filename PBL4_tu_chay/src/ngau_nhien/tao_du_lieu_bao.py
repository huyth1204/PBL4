"""Sinh bộ đánh giá bão mạnh và ma trận rủi ro; không ghi đè bộ train/val/test cũ.

python -m src.ngau_nhien.tao_du_lieu_bao --out-dir /workspace/pbl4-data/yagi-risk
Rủi ro có hậu tố _oracle là NHÃN dùng tương lai thời tiết, không phải feature online.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import gc
import hashlib
import json
from pathlib import Path

import numpy as np

from src.ngau_nhien.khoang_loi import dijkstra
from src.ngau_nhien.kich_ban import MangGoc, ThamSo, sinh_kich_ban, tham_so_dict
from src.ngau_nhien.rui_ro import TichLuyRuiRo
from src.ngau_nhien.thoi_tiet import ThoiTiet

ROOT = Path(__file__).resolve().parents[2]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def tao_bo(raw_path, weather_path, out_dir, start="2024-09-07T06:00", snapshots=600,
           stride=20, delta=9, ensemble=16, pairs=32, seed=42, rain_scale=1.0, gust_scale=1.0):
    """Mạng nền giữ nguyên; scale != 1 là stress test tổng hợp, không phải Yagi quan trắc."""
    if min(snapshots, stride, delta, ensemble, pairs) < 1 or snapshots <= delta or ensemble < 2:
        raise ValueError("cần snapshots > delta, ensemble >= 2 và các số đếm dương")
    if not np.isfinite([rain_scale, gust_scale]).all() or min(rain_scale, gust_scale) <= 0:
        raise ValueError("hệ số thời tiết phải hữu hạn và dương")
    raw_path, weather_path, out = Path(raw_path), Path(weather_path), Path(out_dir)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Từ chối ghi đè thư mục có dữ liệu: {out}")
    tt = ThoiTiet(str(weather_path))
    t0 = np.datetime64(start, "m")
    times = tt.time_utc.astype("datetime64[m]")
    pos = np.flatnonzero(times == t0)
    if len(pos) != 1:
        raise ValueError("start phải khớp một giờ UTC trong tệp thời tiết")
    h0 = int(pos[0])
    # Nội suy đọc thêm giờ kế tiếp: không cho phép lặp lại giờ cuối một cách âm thầm.
    last_hour = h0 + int(np.floor((snapshots - 1) * 20 / 3600)) + 1
    if last_hour >= tt.H or not np.all(np.diff(times[h0:last_hour + 1]) == np.timedelta64(60, "m")):
        raise ValueError("không đủ thời tiết liên tục cho cả kịch bản và giờ nội suy kế tiếp")
    tt.precip = tt.precip * rain_scale
    tt.gust = tt.gust * gust_scale
    with np.load(raw_path, allow_pickle=False) as z:
        if snapshots > len(z["W_snapshots"]):
            raise ValueError("snapshots vượt độ dài mạng nền; không lặp quỹ đạo giả")
        W = z["W_snapshots"][:snapshots]
        names, kinds, layers = (z[k] for k in ("node_order", "node_kind", "node_layer"))
    g = MangGoc(W, kinds, names, tt)
    params = ThamSo(xung_dot=False, tai=False)
    moc = np.arange(0, snapshots - delta, stride, dtype=np.int32)
    endpoints = np.flatnonzero(np.isin(kinds, ["user", "gateway"]))
    all_pairs = np.array([(s, d) for s in endpoints for d in endpoints if s != d], dtype=np.int32)
    if not len(all_pairs) or pairs > len(all_pairs):
        raise ValueError("pairs phải không vượt số cặp gateway/user khác nhau")
    chosen = all_pairs[np.random.default_rng(seed).choice(len(all_pairs), pairs, replace=False)]
    query_snapshot = np.repeat(np.arange(len(moc), dtype=np.int32), pairs)
    src, dst = (np.tile(chosen[:, i], len(moc)) for i in (0, 1))
    # Thế giới quan sát và ensemble nhãn dùng các luồng seed độc lập.
    children = np.random.SeedSequence(seed).spawn(ensemble + 1)
    seeds = [int(c.generate_state(1)[0]) for c in children]
    observed, q = sinh_kich_ban(g, tt, h0, seeds[0], params)
    X = np.zeros((len(moc), len(names), 8), np.float32)
    from src.ngau_nhien.tao_du_lieu import dac_trung_nut
    for k, t in enumerate(moc):
        X[k] = dac_trung_nut(g, q, int(t))
    del q
    paths = [dijkstra(observed[moc[k]], int(s), int(d), kinds != "user") or []
             for k, s, d in zip(query_snapshot, src, dst)]
    path_len = max(2, max(map(len, paths)))
    padded = np.full((len(paths), path_len), -1, np.int32)
    reachable = np.array([bool(p) for p in paths])
    actual_failure = np.full(len(paths), -1, np.int8)
    for i, (k, p) in enumerate(zip(query_snapshot, paths)):
        padded[i, :len(p)] = p
        if p:
            future = observed[moc[k] + 1:moc[k] + delta + 1]
            actual_failure[i] = int(np.any(future[:, p[:-1], p[1:]] <= 0))
    current = observed[moc].copy()
    del observed
    acc = TichLuyRuiRo(moc, len(names), delta, query_snapshot, paths)
    for i, draw_seed in enumerate(seeds[1:]):
        wn, q = sinh_kich_ban(g, tt, h0, draw_seed, params)
        acc.them(wn > 0)
        del wn, q
        gc.collect()
        print(f"Monte Carlo {i + 1}/{ensemble}", flush=True)
    arrays = {
        "W_snapshots": current, "node_extra": X, "node_order": names, "node_kind": kinds, "node_layer": layers,
        "base_snapshot_id": moc, "query_snapshot_id": query_snapshot, "query_source": src, "query_target": dst,
        "query_reachable": reachable, "current_path_indices": padded,
        "route_failed_observed": actual_failure, **acc.ket_qua(),
    }
    iso = datetime.fromisoformat(str(t0)).replace(tzinfo=timezone.utc)
    arrays["weather_time_utc"] = np.array([(iso + timedelta(seconds=int(t) * 20)).isoformat() for t in moc])
    metadata = {
        "schema": "bao-rui-ro-v1", "purpose": "evaluation_only_yagi_holdout", "seed": seed,
        "raw": {"path": str(raw_path.resolve()), "sha256": sha256(raw_path)},
        "weather": {"path": str(weather_path.resolve()), "sha256": sha256(weather_path), "source": tt.nguon,
                    "start_utc": str(t0), "rain_scale": rain_scale, "gust_scale": gust_scale},
        "synthetic_stress": rain_scale != 1.0 or gust_scale != 1.0,
        "dt_seconds": 20, "snapshots": snapshots, "stride": stride, "horizon_steps": delta,
        "horizon_seconds": delta * 20, "ensemble": ensemble, "observation_seed": seeds[0], "ensemble_seeds": seeds[1:],
        "parameters": tham_so_dict(params), "risk_definition": "P(any edge/route down in (t,t+delta] | same edge/route up at t)",
        "risk_role": "oracle_target_uses_future_weather_not_model_input",
        "missing_risk": "NaN with count=0; diagonal/nonexistent/unobserved candidates are not zero-risk links",
        "route_risk": "joint Monte Carlo; no independent-edge product; fixed route chosen in observation world",
        "query_policy": "fixed ordered gateway/user pairs; unreachable queries retained; no rerouting during horizon",
        "summary": {"nodes": len(names), "risk_snapshots": len(moc), "queries": len(paths),
                    "unreachable_observed": int((~reachable).sum()),
                    "reachable_route_failures_observed": int((actual_failure == 1).sum())},
        "limitations": ["Finite ensemble: probabilities have sampling uncertainty; inspect count arrays.",
                        "Hourly gridded weather plus assumed AR(1), thunder and UAV rules; not measured network failures.",
                        "Orbital dates belong to the raw dataset; weather dates do not reconstruct Yagi-2024 satellite positions.",
                        "Underlying raw topology is retained, including any pre-existing ISL line-of-sight limitations.",
                        "Risk conditions on present availability and fixed weather trajectory, not full observed stochastic state.",
                        "Do not train/tune on this Yagi holdout or inject oracle risk into online features."],
    }
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "test.npz", **arrays)
    metadata["output_sha256"] = sha256(out / "test.npz")
    (out / "meta.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return metadata


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", default=str(ROOT / "dataset/oracle_dataset_v5.npz"))
    ap.add_argument("--thoi-tiet", default=str(ROOT / "data/thoi_tiet_vn_mr.npz"))
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--start", default="2024-09-07T06:00", help="giờ UTC trong tệp weather")
    for name, default in [("snapshots", 600), ("stride", 20), ("delta", 9), ("ensemble", 16), ("pairs", 32), ("seed", 42)]:
        ap.add_argument(f"--{name}", type=int, default=default)
    ap.add_argument("--rain-scale", type=float, default=1.0)
    ap.add_argument("--gust-scale", type=float, default=1.0)
    a = ap.parse_args()
    m = tao_bo(a.raw, a.thoi_tiet, a.out_dir, a.start, a.snapshots, a.stride, a.delta, a.ensemble,
               a.pairs, a.seed, a.rain_scale, a.gust_scale)
    print(json.dumps(m["summary"], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
