"""Đo thời gian huấn luyện DM theo quy mô trong cùng điều kiện (1 luồng, không chạy việc khác).

    python test/do_huan_luyen_quy_mo.py --dirs data/quy_mo/n73 data/quy_mo/n173 ... --out results/quy_mo/huan_luyen.json
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
from src.ai.data_v2 import SagsinData, make_batch  # noqa: E402
from src.ai.train_gnn import build_model, to_t  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--dirs", nargs="+", required=True)
ap.add_argument("--so-lo", type=int, default=15)
ap.add_argument("--epoch", nargs="+", type=int, default=None, help="số epoch đã dùng cho từng thư mục")
ap.add_argument("--out", required=True)
a = ap.parse_args()
torch.set_num_threads(1)
kq = {}
for k, d in enumerate(a.dirs):
    tr = SagsinData(Path(d) / "train.npz")
    args = {"hidden": 64, "layers": 4, "head": "diffusion2", "param": "x0ce"}
    m = build_model(args, tr.f_in)
    opt = torch.optim.AdamW(m.parameters(), lr=2e-3)
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(tr))
    ts = []
    for j in range(a.so_lo + 2):
        t0 = time.perf_counter()
        b = to_t(make_batch(tr, perm[j * 64:(j + 1) * 64], 0.15, rng))
        loss = m.training_loss(b)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if j >= 2:
            ts.append(time.perf_counter() - t0)
    so_lo = int(np.ceil(len(tr) / 64))
    kq[str(tr.N)] = {"s_moi_lo": float(np.mean(ts)), "so_lo_moi_epoch": so_lo, "s_moi_epoch": float(np.mean(ts) * so_lo),
                     "epoch": a.epoch[k] if a.epoch else None, "so_duong_train": len(tr)}
    print(tr.N, kq[str(tr.N)], flush=True)
json.dump(kq, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
