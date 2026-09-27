"""Đo thời gian tìm đường theo quy mô trên máy rảnh: Dijkstra (Python, C) và mô hình (từng đường, theo lô).

    python test/do_thoi_gian_quy_mo.py --cau-hinh "73=data/n73:results/dmx0_4lop.pt:1" ... --out results/quy_mo/thoi_gian.json
Mỗi cấu hình: "N=thư_mục_dữ_liệu:mô_hình.pt:guidance[:mô_hình_2.pt]".
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.ai.data_v2 import SagsinData  # noqa: E402
from test.danh_gia_quy_mo import chay_mo_hinh, load, relay_dijkstra, scipy_dijkstra  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--cau-hinh", nargs="+", required=True)
ap.add_argument("--n-single", type=int, default=150)
ap.add_argument("--n-lo", type=int, default=1200)
ap.add_argument("--threads", type=int, default=2)
ap.add_argument("--out", required=True)
a = ap.parse_args()
torch.set_num_threads(a.threads)
kq = {}
for c in a.cau_hinh:
    n, rest = c.split("=", 1)
    parts = rest.split(":")
    te = SagsinData(Path(parts[0]) / "test.npz")
    rng = np.random.default_rng(0)
    single = rng.choice(len(te), min(a.n_single, len(te)), replace=False)
    lo = np.sort(rng.choice(len(te), min(a.n_lo, len(te)), replace=False))
    r = {}
    for ten, f in (("dijkstra_python", relay_dijkstra), ("dijkstra_C", scipy_dijkstra)):
        t0 = time.perf_counter()
        for i in single:
            f(te.W[te.sid[i]], te.paths[i][0], te.paths[i][-1], te.can_relay)
        r[f"ms_{ten}"] = (time.perf_counter() - t0) / len(single) * 1000
    mo_hinh = [("dm", parts[1], float(parts[2]), 2)] + ([("softmax", parts[3], 1.0, 1)] if len(parts) > 3 else [])
    for ten, path, w, ns in mo_hinh:
        m = load(path)
        chay_mo_hinh(m, te, single[:3], w, ns, 1)                     # khởi động
        t0 = time.perf_counter()
        for i in single:
            chay_mo_hinh(m, te, np.array([i]), w, ns, 1)
        r[f"ms_{ten}_tung_duong"] = (time.perf_counter() - t0) / len(single) * 1000
        t0 = time.perf_counter()
        chay_mo_hinh(m, te, lo, w, ns, 200)
        r[f"ms_{ten}_theo_lo"] = (time.perf_counter() - t0) / len(lo) * 1000
    kq[n] = r
    print(n, r, flush=True)
json.dump(kq, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
