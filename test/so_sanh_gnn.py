"""
So sánh DM (dự đoán thẳng y0 + cross-entropy) KHÔNG có GNN (0 lớp) và CÓ GNN (4 lớp).

    python test/so_sanh_gnn.py                  # huấn luyện nếu chưa có checkpoint, rồi so sánh
    python test/so_sanh_gnn.py --epochs 5       # chạy nhanh để thử
    python test/so_sanh_gnn.py --retrain        # huấn luyện lại dù đã có checkpoint

Cả hai dùng cùng dữ liệu, cùng số epoch, cùng cách chọn (số bước DDIM, guidance) trên VAL,
rồi mới đo trên TEST. Kết quả in thành bảng và lưu results/so_sanh_gnn.json.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.ai.data_v2 import SagsinData  # noqa: E402
from src.ai.train_gnn import (build_model, encode_pair, eval_nexthop, eval_rollout,  # noqa: E402
                              greedy_chooser, make_random_chooser, model_chooser)

VARIANTS = [("Không GNN (0 lớp)", 0, "dmx0_0lop"), ("Có GNN (4 lớp)", 4, "dmx0_4lop")]
STEPS, WS = (1, 2, 5, 20), (1.0, 2.0, 4.0)


def train(tag, layers, a):
    cmd = [sys.executable, "-u", str(ROOT / "src/ai/train_gnn.py"), "--data-dir", a.data_dir, "--out-dir", a.out_dir,
           "--epochs", str(a.epochs), "--layers", str(layers), "--head", "diffusion2", "--param", "x0ce",
           "--ddim", "2", "--tag", tag, "--threads", str(a.threads)]
    print(">>", " ".join(cmd[2:]), flush=True)
    subprocess.run(cmd, check=True)


def load(path):
    ck = torch.load(path)
    m = build_model(ck["args"], ck["f_in"])
    m.load_state_dict(ck["state"])
    return m.eval()


def evaluate(model, va, te):
    # chọn cấu hình suy luận trên VAL
    best, grid = (None, -1.0), {}
    for n in STEPS:
        acc, _, _ = eval_nexthop(model, va, WS, n_steps=n)
        for w in WS:
            grid[f"{n}b_w{w:g}"] = acc[w]
            if acc[w] > best[1]:
                best = ((n, w), acc[w])
    (n, w), _ = best
    acc_te, rand, greedy = eval_nexthop(model, te, (w,), n_steps=n)
    idx = np.arange(len(te))

    def model_fn(part):
        W = te.W[te.sid[part]]
        dst = np.array([te.paths[i][-1] for i in part])
        hc, hu = encode_pair(model, W, te.static_feat, dst)
        return model_chooser(model, hc, hu, w, n)
    t0 = time.time()
    roll, _ = eval_rollout(te, idx, model_fn)
    ms = (time.time() - t0) / len(idx) * 1000
    return {"so_buoc_ddim": n, "guidance": w, "val_luoi": grid, "nexthop": acc_te[w],
            "rollout": roll, "ms_moi_duong": ms}, rand, greedy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(ROOT / "dataset"))
    ap.add_argument("--out-dir", default=str(ROOT / "results"))
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--retrain", action="store_true")
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    for _, L, tag in VARIANTS:
        if a.retrain or not (out / f"{tag}.pt").exists():
            train(tag, L, a)

    va, te = SagsinData(Path(a.data_dir) / "val.npz"), SagsinData(Path(a.data_dir) / "test.npz")
    res, base = {}, {}
    for name, L, tag in VARIANTS:
        print(f"Đánh giá {name} ...", flush=True)
        r, rand, greedy = evaluate(load(out / f"{tag}.pt"), va, te)
        r["tham_so"] = sum(p.numel() for p in load(out / f"{tag}.pt").parameters())
        res[name] = r
        base = {"ngau_nhien": rand, "tham_lam": greedy}
    g = eval_rollout(te, np.arange(len(te)), lambda part: greedy_chooser)[0]
    rd = eval_rollout(te, np.arange(len(te)), lambda part: make_random_chooser(0))[0]

    A, B = (res[n] for n, _, _ in VARIANTS)
    rows = [
        ("Số tham số", f"{A['tham_so']:,}", f"{B['tham_so']:,}", "", "", ""),
        ("Cấu hình (bước DDIM, w) chọn trên val", f"{A['so_buoc_ddim']}, {A['guidance']:g}",
         f"{B['so_buoc_ddim']}, {B['guidance']:g}", "", "", ""),
        ("Chọn đúng next-hop", A["nexthop"], B["nexthop"], base["tham_lam"], base["ngau_nhien"], "%"),
        ("Tới được đích", A["rollout"]["ty_le_toi_dich"], B["rollout"]["ty_le_toi_dich"],
         g["ty_le_toi_dich"], rd["ty_le_toi_dich"], "%"),
        ("Trễ / Dijkstra (TB, đường tới đích)", A["rollout"]["tre_so_voi_dijkstra_TB"],
         B["rollout"]["tre_so_voi_dijkstra_TB"], g["tre_so_voi_dijkstra_TB"], rd["tre_so_voi_dijkstra_TB"], "x"),
        ("Trong 10% tối ưu", A["rollout"]["trong_10pct_toi_uu"], B["rollout"]["trong_10pct_toi_uu"],
         g["trong_10pct_toi_uu"], rd["trong_10pct_toi_uu"], "%"),
        ("Trùng đường oracle", A["rollout"]["trung_khop_duong_oracle"], B["rollout"]["trung_khop_duong_oracle"],
         g["trung_khop_duong_oracle"], rd["trung_khop_duong_oracle"], "%"),
        ("Thời gian sinh 1 đường (ms)", A["ms_moi_duong"], B["ms_moi_duong"], None, None, "ms"),
    ]

    def f(v, u):
        if v is None or v == "":
            return "" if v == "" else "-"
        if isinstance(v, str):
            return v
        return f"{v * 100:.1f}%" if u == "%" else (f"{v:.3f}×" if u == "x" else f"{v:.1f}")

    head = ["Chỉ số", "Không GNN", "Có GNN", "Chênh lệch", "Tham lam", "Ngẫu nhiên"]
    table = []
    for lab, a0, b4, gr, rn, u in rows:
        diff = ""
        if u == "%":
            diff = f"{(b4 - a0) * 100:+.1f} điểm"
        elif u == "x":
            diff = f"{b4 - a0:+.3f}×"
        table.append([lab, f(a0, u), f(b4, u), diff, f(gr, u), f(rn, u)])
    wid = [max(len(r[i]) for r in table + [head]) for i in range(6)]
    line = lambda r: "  ".join(c.ljust(wid[i]) if i == 0 else c.rjust(wid[i]) for i, c in enumerate(r))  # noqa: E731
    print(f"\nDM dự đoán y0 + cross-entropy | TEST {len(te):,} đường, {te.N} nút\n")
    print(line(head))
    print("-" * (sum(wid) + 10))
    for r in table:
        print(line(r))

    print("\nTheo kiểu tuyến (tới đích | trễ TB):")
    for name, _, _ in VARIANTS:
        kt = res[name]["rollout"]["theo_kieu_tuyen"]
        print(f"  {name:18s} " + " | ".join(
            f"{k}: {v['so_duong']} đường, {v['toi_dich'] * 100:.1f}%, {v['tre_TB']:.3f}×" for k, v in kt.items()))

    checks = [
        ("Cả hai chọn next-hop tốt hơn tham lam", A["nexthop"] > base["tham_lam"] and B["nexthop"] > base["tham_lam"]),
        ("Cả hai tới đích nhiều hơn tham lam",
         A["rollout"]["ty_le_toi_dich"] > g["ty_le_toi_dich"] and B["rollout"]["ty_le_toi_dich"] > g["ty_le_toi_dich"]),
        ("Có GNN chọn next-hop tốt hơn không GNN", B["nexthop"] > A["nexthop"]),
    ]
    print()
    for lab, ok in checks:
        print(f"  [{'ĐẠT' if ok else 'KHÔNG ĐẠT'}] {lab}")
    json.dump({"bien_the": res, "tham_lam": g, "ngau_nhien": rd, "baseline_nexthop": base,
               "N": te.N, "so_duong_test": len(te)},
              open(out / "so_sanh_gnn.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\nĐã lưu {out / 'so_sanh_gnn.json'}")


if __name__ == "__main__":
    main()
