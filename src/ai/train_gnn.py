"""
train_gnn.py — Huấn luyện nấc B (GNN + diffusion) trên dữ liệu SAGSIN 3 tầng.

    python src/ai/train_gnn.py --data-dir dataset --epochs 15
    python src/ai/train_gnn.py --layers 0 --tag nacA     # ablation: không truyền tin trên đồ thị
    python src/ai/train_gnn.py --head classifier --tag clsB   # ablation: bỏ diffusion, giữ GNN
    python src/ai/train_gnn.py --head diffusion2 --layers 0 --self-cond --attn --tag dmC   # DM cải tiến

Đánh giá: next-hop accuracy (guidance chọn trên VAL), rollout cả đường trên TEST
(tỷ lệ tới đích, độ trễ so với Dijkstra, chọn đúng tầng), so với 2 baseline.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.ai.data_v2 import SagsinData, make_batch, node_features, valid_mask  # noqa: E402
from src.ai.gnn_classifier import GraphClassifier  # noqa: E402
from src.ai.gnn_diffusion import GraphDiffusion  # noqa: E402
from src.ai.gnn_diffusion_v2 import GraphDiffusionV2  # noqa: E402


def build_model(a, f_in):
    """a: dict tham số dòng lệnh (lưu trong checkpoint) -> mô hình tương ứng."""
    head = a.get("head", "diffusion")
    if head == "classifier":
        m = GraphClassifier(f_in, a["hidden"], a["layers"], a.get("attn", False))
    elif head == "diffusion2":
        m = GraphDiffusionV2(f_in, a["hidden"], a["layers"], param=a.get("param", "x0ce"),
                             self_cond=a.get("self_cond", False), attn=a.get("attn", False),
                             t_power=a.get("t_power", 1.0))
    else:
        m = GraphDiffusion(f_in, a["hidden"], a["layers"])
    m.lookahead = a.get("lookahead", False)
    return m


def to_t(b):
    out = {}
    for k, v in b.items():
        if v is None:
            out[k] = None
        elif v.dtype == bool:
            out[k] = torch.from_numpy(v)
        elif np.issubdtype(v.dtype, np.integer):
            out[k] = torch.from_numpy(v.astype(np.int64))
        else:
            out[k] = torch.from_numpy(np.ascontiguousarray(v))
    return out


@torch.no_grad()
def encode_pair(model, W, static, dst):
    """Mã hóa có đích (điều kiện) và không đích (cho CFG)."""
    Wt = torch.from_numpy(W)
    la = getattr(model, "lookahead", False)
    hc = model.enc(torch.from_numpy(node_features(W, static, dst, None, la)), Wt)
    hu = model.enc(torch.from_numpy(node_features(W, static, dst, np.ones(len(dst), bool), la)), Wt)
    return hc, hu


@torch.no_grad()
def eval_nexthop(model, data, guidances=(2.0,), max_paths=None, n_steps=20, bs=128):
    idx = np.arange(len(data))
    if max_paths and len(idx) > max_paths:
        idx = np.sort(np.random.default_rng(0).choice(idx, max_paths, replace=False))
    corr = {w: 0 for w in guidances}
    total, rand, greedy = 0, 0.0, 0
    for s in range(0, len(idx), bs):
        b = make_batch(data, idx[s:s + bs])
        hc, hu = encode_pair(model, b["W"], data.static_feat, b["dst"])
        sp = torch.from_numpy(b["step_p"])
        args = (torch.from_numpy(b["cur"]), torch.from_numpy(b["dst"][b["step_p"]]),
                torch.from_numpy(b["rows"]), torch.from_numpy(b["mask"]))
        for w in guidances:
            torch.manual_seed(0)
            pred = model.decide(hc[sp], hu[sp], *args, n_steps=n_steps, guidance=w)
            corr[w] += int((pred.numpy() == b["nxt"]).sum())
        total += len(b["nxt"])
        rand += float((1.0 / b["mask"].sum(1)).sum())
        greedy += int((np.where(b["mask"], b["rows"], np.inf).argmin(1) == b["nxt"]).sum())
    return {w: corr[w] / total for w in guidances}, rand / total, greedy / total


def rollout(data, path_idx, chooser, max_hops=12):
    """Sinh cả đường từ nguồn tới đích; chặn quay lại nút đã đi qua."""
    P = len(path_idx)
    W = data.W[data.sid[path_idx]]
    src = np.array([data.paths[i][0] for i in path_idx])
    dst = np.array([data.paths[i][-1] for i in path_idx])
    cur = src.copy()
    visited = np.zeros((P, data.N), bool)
    visited[np.arange(P), src] = True
    routes = [[int(s)] for s in src]
    done = np.zeros(P, bool)
    ok = np.zeros(P, bool)
    for _ in range(max_hops):
        act = np.where(~done)[0]
        if len(act) == 0:
            break
        rows = W[act, cur[act]]
        mask = valid_mask(rows, dst[act], data.can_relay, visited[act])
        empty = mask.sum(1) == 0
        done[act[empty]] = True            # kẹt: không còn ứng viên -> thất bại
        act, rows, mask = act[~empty], rows[~empty], mask[~empty]
        if len(act) == 0:
            break
        nxt = chooser(act, cur[act], dst[act], rows, mask)
        for p, n in zip(act, nxt):
            routes[p].append(int(n))
            visited[p, n] = True
            cur[p] = n
            if n == dst[p]:
                done[p] = ok[p] = True
    return routes, ok, W


def route_delay(W, route):
    return float(sum(W[u, v] for u, v in zip(route[:-1], route[1:])))


def rollout_metrics(data, path_idx, routes, ok, W):
    lay = data.layer
    uses_space = lambda r: any(lay[n] == "space" for n in r)  # noqa: E731
    ratios, exact, tier_ok, hop_diff = [], 0, 0, []
    by_type = {"qua_ve_tinh": [0, 0, []], "chi_tang_khong_mat_dat": [0, 0, []]}
    for k, i in enumerate(path_idx):
        orc = data.paths[i]
        key = "qua_ve_tinh" if uses_space(orc) else "chi_tang_khong_mat_dat"
        by_type[key][0] += 1
        if not ok[k]:
            continue
        by_type[key][1] += 1
        r = route_delay(W[k], routes[k]) / max(route_delay(W[k], orc), 1e-9)
        ratios.append(r)
        by_type[key][2].append(r)
        exact += routes[k] == orc
        tier_ok += uses_space(routes[k]) == uses_space(orc)
        hop_diff.append(len(routes[k]) - len(orc))
    n, nr = len(path_idx), max(int(ok.sum()), 1)
    ratios = np.array(ratios)
    has = len(ratios) > 0
    return {
        "ty_le_toi_dich": float(ok.mean()),
        "tre_so_voi_dijkstra_TB": float(ratios.mean()) if has else float("nan"),
        "tre_so_voi_dijkstra_trung_vi": float(np.median(ratios)) if has else float("nan"),
        "trong_10pct_toi_uu": float(np.mean(ratios <= 1.10)) if has else 0.0,
        "trung_khop_duong_oracle": exact / n,
        "chon_dung_tang": tier_ok / nr,
        "chenh_so_hop_TB": float(np.mean(hop_diff)) if hop_diff else float("nan"),
        "theo_kieu_tuyen": {k: {"so_duong": v[0], "toi_dich": v[1] / max(v[0], 1),
                                "tre_TB": float(np.mean(v[2])) if v[2] else float("nan")}
                            for k, v in by_type.items()},
    }


def model_chooser(model, hc_all, hu_all, guidance, n_steps):
    def choose(act, cur, dst, rows, mask):
        a = torch.from_numpy(act)
        return model.decide(hc_all[a], hu_all[a], torch.from_numpy(cur), torch.from_numpy(dst),
                            torch.from_numpy(rows), torch.from_numpy(mask),
                            n_steps=n_steps, guidance=guidance).numpy()
    return choose


def greedy_chooser(act, cur, dst, rows, mask):
    return np.where(mask, rows, np.inf).argmin(1)


def make_random_chooser(seed=0):
    rng = np.random.default_rng(seed)

    def choose(act, cur, dst, rows, mask):
        return np.array([rng.choice(np.flatnonzero(m)) for m in mask])
    return choose


@torch.no_grad()
def eval_rollout(data, path_idx, chooser_fn, chunk=400, max_hops=12):
    routes_all, ok_all, W_all = [], [], []
    for s in range(0, len(path_idx), chunk):
        part = path_idx[s:s + chunk]
        chooser = chooser_fn(part)
        r, ok, W = rollout(data, part, chooser, max_hops)
        routes_all += r
        ok_all.append(ok)
        W_all.append(W)
    return rollout_metrics(data, path_idx, routes_all, np.concatenate(ok_all), np.concatenate(W_all)), routes_all


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(ROOT / "dataset"))
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--drop", type=float, default=0.15, help="tỷ lệ bỏ điều kiện đích (CFG)")
    ap.add_argument("--ddim", type=int, default=20)
    ap.add_argument("--eval-every", type=int, default=3)
    ap.add_argument("--head", choices=["diffusion", "diffusion2", "classifier"], default="diffusion")
    ap.add_argument("--param", choices=["eps", "x0ce"], default="x0ce", help="chỉ cho diffusion2")
    ap.add_argument("--self-cond", action="store_true", help="chỉ cho diffusion2")
    ap.add_argument("--attn", action="store_true", help="attention giữa các ứng viên (diffusion2/classifier)")
    ap.add_argument("--lookahead", action="store_true", help="thêm đặc trưng nối thẳng tới đích")
    ap.add_argument("--t-power", type=float, default=1.0, help=">1: huấn luyện nghiêng về nhiễu cao (diffusion2)")
    ap.add_argument("--nhieu-tuyen", type=float, default=0.0,
                    help=">0: nhãn nhiều tuyến — mỗi epoch chọn ngẫu nhiên 1 trong các tuyến có trễ <= (1+x) tối ưu")
    ap.add_argument("--k-tuyen", type=int, default=4, help="số tuyến gần tối ưu tối đa mỗi cặp (thuật toán Yen)")
    ap.add_argument("--tag", default="nacB")
    ap.add_argument("--out-dir", default=str(ROOT / "results"))
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    dd = Path(args.data_dir)
    tr, va, te = (SagsinData(dd / f"{s}.npz") for s in ("train", "val", "test"))
    for d in (tr, va, te):
        d.lookahead = args.lookahead
    print(f"[{args.tag}] N={tr.N} | train {len(tr)} đường/{tr.n_steps} bước | "
          f"val {len(va)}/{va.n_steps} | test {len(te)}/{te.n_steps}", flush=True)

    if args.head == "classifier":
        args.drop = 0.0                                 # không dùng CFG
    model = build_model(vars(args), tr.f_in)
    print(f"[{args.tag}] tham số: {sum(p.numel() for p in model.parameters()):,} | "
          f"GNN {args.layers} lớp, H={args.hidden}, đầu ra {args.head}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    n_batches = int(np.ceil(len(tr) / args.batch))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=args.epochs * n_batches)

    best, ckpt = -1.0, out / f"{args.tag}.pt"
    alts = None
    if args.nhieu_tuyen > 0:
        from src.ai.nhieu_tuyen import yen_near_optimal
        t0 = time.time()
        orig = tr.paths
        alts = [yen_near_optimal(tr.W[tr.sid[i]], p[0], p[-1], tr.can_relay, args.k_tuyen, args.nhieu_tuyen) or [p]
                for i, p in enumerate(orig)]
        print(f"[{args.tag}] nhãn nhiều tuyến (trễ <= {1 + args.nhieu_tuyen:g}x tối ưu, tối đa {args.k_tuyen}): "
              f"TB {np.mean([len(a) for a in alts]):.2f} tuyến/cặp, {np.mean([len(a) > 1 for a in alts]) * 100:.1f}% "
              f"cặp có >= 2 tuyến ({time.time() - t0:.0f}s)", flush=True)
    for ep in range(args.epochs):
        model.train()
        t0, tot = time.time(), 0.0
        if alts is not None:                            # mỗi epoch một bộ nhãn khác nhau
            tr.paths = [a[rng.integers(len(a))] for a in alts]
        perm = rng.permutation(len(tr))
        for s in range(0, len(perm), args.batch):
            b = to_t(make_batch(tr, perm[s:s + args.batch], args.drop, rng))
            loss = model.training_loss(b)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += loss.item()
        msg = f"[{args.tag}] epoch {ep:2d} | loss {tot / n_batches:.4f} | {time.time() - t0:.0f}s"
        if ep % args.eval_every == args.eval_every - 1 or ep == args.epochs - 1:
            model.eval()
            acc, _, _ = eval_nexthop(model, va, (2.0,), max_paths=800, n_steps=args.ddim)
            msg += f" | val next-hop {acc[2.0] * 100:.1f}%"
            if acc[2.0] > best:
                best = acc[2.0]
                torch.save({"state": model.state_dict(), "args": vars(args), "f_in": tr.f_in}, ckpt)
                msg += " *"
        print(msg, flush=True)

    model.load_state_dict(torch.load(ckpt)["state"])
    model.eval()
    ws = (1.0,) if args.head == "classifier" else (1.0, 2.0, 3.0, 4.0)
    acc_val, _, _ = eval_nexthop(model, va, ws, n_steps=args.ddim)
    w_best = max(ws, key=lambda w: acc_val[w])
    acc_te, rand_te, greedy_te = eval_nexthop(model, te, ws, n_steps=args.ddim)
    print(f"[{args.tag}] guidance chọn trên VAL: w={w_best} "
          f"(val {', '.join(f'w{w}={acc_val[w] * 100:.1f}%' for w in ws)})", flush=True)
    print(f"[{args.tag}] TEST next-hop: mô hình {acc_te[w_best] * 100:.1f}% | "
          f"ngẫu nhiên {rand_te * 100:.1f}% | tham lam trễ thấp {greedy_te * 100:.1f}%", flush=True)

    idx = np.arange(len(te))
    t0 = time.time()

    def model_fn(part):
        W = te.W[te.sid[part]]
        dst = np.array([te.paths[i][-1] for i in part])
        hc, hu = encode_pair(model, W, te.static_feat, dst)
        return model_chooser(model, hc, hu, w_best, args.ddim)

    roll_model, routes = eval_rollout(te, idx, model_fn)
    roll_greedy, _ = eval_rollout(te, idx, lambda part: greedy_chooser)
    roll_rand, _ = eval_rollout(te, idx, lambda part: make_random_chooser(0))
    print(f"[{args.tag}] rollout xong sau {time.time() - t0:.0f}s", flush=True)

    res = {
        "tag": args.tag, "args": vars(args), "guidance": w_best,
        "val_nexthop": {str(w): acc_val[w] for w in ws},
        "test_nexthop": {"mo_hinh": acc_te[w_best], "ngau_nhien": rand_te, "tham_lam": greedy_te,
                         "theo_guidance": {str(w): acc_te[w] for w in ws}},
        "test_rollout": {"mo_hinh": roll_model, "tham_lam": roll_greedy, "ngau_nhien": roll_rand},
    }
    (out / f"{args.tag}_results.json").write_text(json.dumps(res, ensure_ascii=False, indent=2))
    np.save(out / f"{args.tag}_test_routes.npy", np.array(routes, dtype=object), allow_pickle=True)
    for name, r in res["test_rollout"].items():
        print(f"[{args.tag}] ROLLOUT {name:10s}: tới đích {r['ty_le_toi_dich'] * 100:5.1f}% | "
              f"trễ/Dijkstra TB {r['tre_so_voi_dijkstra_TB']:.3f} | trong 10% tối ưu "
              f"{r['trong_10pct_toi_uu'] * 100:5.1f}% | trùng oracle {r['trung_khop_duong_oracle'] * 100:5.1f}% | "
              f"đúng tầng {r['chon_dung_tang'] * 100:5.1f}%", flush=True)
    print(f"[{args.tag}] Đã lưu {ckpt} và {out / (args.tag + '_results.json')}", flush=True)


if __name__ == "__main__":
    main()
