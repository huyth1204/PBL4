"""
train_da_quy_mo.py — Huấn luyện MỘT mô hình (GNN + DM hoặc softmax) trên nhiều quy mô mạng cùng lúc.

Mỗi lô lấy từ một quy mô (các đồ thị trong lô phải cùng số nút); các quy mô xen kẽ nhau, mỗi epoch lấy cùng số lô từ
mỗi quy mô để không quy mô nào lấn át. Có thể khởi tạo từ một mô hình đã có (ví dụ mô hình 73 nút).

    python -m src.ai.train_da_quy_mo --data-dirs dataset data/quy_mo/n173 data/quy_mo/n273 data/quy_mo/n373 \
        --init results/dmx0_4lop.pt --epochs 12 --lo-moi-quy-mo 100 --tag dm_da_quy_mo --out-dir results/quy_mo

Học với số đo trễ có nhiễu (nhãn vẫn là đường tối ưu trên trễ thật; chọn checkpoint trên val có nhiễu σ ở giữa khoảng):
    python -m src.ai.train_da_quy_mo ... --init results/quy_mo/dm_da_quy_mo.pt --nhieu 0.6 --lr 3e-4 --epochs 6 \
        --lo-moi-quy-mo 60 --tag dm_nhieu
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

import copy

from src.ai.data_v2 import SagsinData, make_batch, nhieu_nhan
from src.ai.train_gnn import build_model, eval_nexthop, to_t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dirs", nargs="+", required=True)
    ap.add_argument("--init", default=None, help="checkpoint khởi tạo (cùng kiến trúc)")
    ap.add_argument("--head", choices=["diffusion2", "classifier"], default="diffusion2")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--lo-moi-quy-mo", type=int, default=100, help="số lô mỗi quy mô trong một epoch")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--drop", type=float, default=0.15)
    ap.add_argument("--ddim", type=int, default=2)
    ap.add_argument("--val-paths", type=int, default=400)
    ap.add_argument("--eval-every", type=int, default=2)
    ap.add_argument("--tag", default="dm_da_quy_mo")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--nhieu", type=float, default=0.0, help="σ lớn nhất của nhiễu đo khi huấn luyện (0 = sạch)")
    ap.add_argument("--nhieu-min", type=float, default=0.0, help="σ nhỏ nhất (chỉ học vùng nhiễu cao)")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    trs = [SagsinData(Path(d) / "train.npz") for d in a.data_dirs]
    vas = [SagsinData(Path(d) / "val.npz") for d in a.data_dirs]
    if a.nhieu > 0:
        for i, va in enumerate(vas):
            vas[i] = copy.copy(va)
            vas[i].W = nhieu_nhan(va.W, np.full(len(va.W), (a.nhieu_min + a.nhieu) / 2), np.random.default_rng(123))
    f_in = trs[0].f_in
    assert all(t.f_in == f_in for t in trs), "các quy mô phải cùng số đặc trưng nút"
    if a.init:
        ck = torch.load(a.init, weights_only=False)
        args = dict(ck["args"])
        model = build_model(args, ck["f_in"])
        model.load_state_dict(ck["state"])
        print(f"[{a.tag}] khởi tạo từ {a.init}", flush=True)
    else:
        args = {"hidden": 64, "layers": 4, "head": a.head, "param": "x0ce", "self_cond": False, "attn": False,
                "t_power": 1.0, "lookahead": False}
        model = build_model(args, f_in)
    args.update(tag=a.tag, data_dirs=a.data_dirs, da_quy_mo=True, ddim=a.ddim, nhieu=a.nhieu, nhieu_min=a.nhieu_min)
    if args.get("head") == "classifier":
        a.drop = 0.0
    Ns = [t.N for t in trs]
    print(f"[{a.tag}] quy mô {Ns} | train {[len(t) for t in trs]} đường | {a.lo_moi_quy_mo} lô/quy mô/epoch | "
          f"tham số {sum(p.numel() for p in model.parameters()):,}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    so_buoc = a.epochs * a.lo_moi_quy_mo * len(trs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=so_buoc)
    best, ckpt, lich_su = -1.0, out / f"{a.tag}.pt", []
    w = 1.0
    for ep in range(a.epochs):
        model.train()
        t0, tong = time.time(), {n: 0.0 for n in Ns}
        perms = [rng.permutation(len(t)) for t in trs]
        for j in range(a.lo_moi_quy_mo):
            for k, tr in enumerate(trs):
                p = perms[k]
                s = (j * a.batch) % max(len(p) - a.batch, 1)
                b = to_t(make_batch(tr, p[s:s + a.batch], a.drop, rng, a.nhieu, a.nhieu_min))
                loss = model.training_loss(b)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                tong[tr.N] += loss.item() / a.lo_moi_quy_mo
        msg = f"[{a.tag}] epoch {ep:2d} | loss " + " ".join(f"{n}:{v:.3f}" for n, v in tong.items()) + \
              f" | {time.time() - t0:.0f}s"
        if ep % a.eval_every == a.eval_every - 1 or ep == a.epochs - 1:
            model.eval()
            accs = {va.N: eval_nexthop(model, va, (w,), max_paths=a.val_paths, n_steps=a.ddim, bs=64)[0][w]
                    for va in vas}
            tb = float(np.mean(list(accs.values())))
            msg += " | val " + " ".join(f"{n}:{v * 100:.1f}%" for n, v in accs.items()) + f" | TB {tb * 100:.2f}%"
            lich_su.append({"epoch": ep, "val": {str(n): v for n, v in accs.items()}, "tb": tb})
            if tb > best:
                best = tb
                torch.save({"state": model.state_dict(), "args": args, "f_in": f_in}, ckpt)
                msg += " *"
        print(msg, flush=True)
    json.dump({"tag": a.tag, "quy_mo": Ns, "lich_su": lich_su, "best_val_tb": best},
              open(out / f"{a.tag}_train.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"[{a.tag}] xong, lưu {ckpt} (val TB tốt nhất {best * 100:.2f}%)", flush=True)


if __name__ == "__main__":
    main()
