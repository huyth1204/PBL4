"""
them_dijkstra.py — BƯỚC B (bổ sung): thêm tuyến Dijkstra THUẦN theo tổng trễ cho K luồng.

Khác `dphat`: `dphat` phạt rủi ro (`prop + 10·(−ln(1−risk))`), còn `dijkstra` chỉ
tối thiểu tổng trễ `W` của đúng snapshot. Dùng `networkx.shortest_path`.

Đọc `duong_di_nut.json`, thêm trường `dijkstra` cho mỗi bản ghi (giữ nguyên dm/thay/dphat).

    python them_dijkstra.py --out duong_di_nut.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

GOC = Path(r"E:\PBL4")
if str(GOC) not in sys.path:
    sys.path.insert(0, str(GOC))


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="duong_di_nut.json")
    a = ap.parse_args()

    import networkx as nx

    raw = np.load(GOC / "dataset" / "oracle_dataset_v5.npz", allow_pickle=False)
    W = raw["W_snapshots"]
    names = [str(x) for x in raw["node_order"]]
    so_snap = W.shape[0]

    p = GOC / a.out
    d = json.loads(p.read_text(encoding="utf-8"))
    print(f"bản ghi: {len(d)} | snapshot tối đa: {so_snap - 1}")

    # cache đồ thị theo t
    G_cache: dict[int, "nx.Graph"] = {}

    def do_thi(t: int):
        if t in G_cache:
            return G_cache[t]
        Wt = W[t]
        G = nx.Graph()
        iu, iv = np.nonzero(np.triu(Wt > 0, 1))
        for u, v in zip(iu, iv):
            G.add_edge(int(u), int(v), weight=float(Wt[u, v]))
        G_cache[t] = G
        return G

    ok, that_bai = 0, 0
    bo_qua = 0
    for khoa, v in d.items():
        t = int(v["t"])
        if t >= so_snap:
            bo_qua += 1
            continue
        G = do_thi(t)
        ds = []
        for l in v.get("dm", []):
            s, e = names.index(l["nguon"]), names.index(l["dich"])
            try:
                nut = nx.shortest_path(G, s, e, weight="weight")
                ds.append({"luong": l["luong"], "nguon": l["nguon"], "dich": l["dich"],
                           "nut": [int(x) for x in nut], "day_du": True,
                           "tre_ms": round(float(nx.shortest_path_length(G, s, e, weight="weight")), 2)})
                ok += 1
            except Exception:
                ds.append({"luong": l["luong"], "nguon": l["nguon"], "dich": l["dich"],
                           "nut": [], "day_du": False, "tre_ms": None})
                that_bai += 1
        v["dijkstra"] = ds

    p.write_text(json.dumps(d, separators=(",", ":")), encoding="utf-8")
    print(f"\ndijkstra: {ok} thành công | {that_bai} không có đường | "
          f"{bo_qua} bản ghi bỏ qua (t >= {so_snap})")
    k0 = next(iter(d))
    print(f"\nví dụ {k0}:")
    print(f"  DM      : {d[k0]['dm'][0]['nut']}")
    print(f"  dijkstra: {d[k0]['dijkstra'][0]['nut']} ({d[k0]['dijkstra'][0]['tre_ms']} ms)")


if __name__ == "__main__":
    main()
