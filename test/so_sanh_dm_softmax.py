"""
So sánh GNN 4 lớp + diffusion (DM) với GNN 4 lớp + softmax (đối chứng) trên cùng dữ liệu.

    python test/so_sanh_dm_softmax.py                 # huấn luyện mô hình nào còn thiếu, rồi so sánh
    python test/so_sanh_dm_softmax.py --epochs 5      # chạy nhanh để thử
    python test/so_sanh_dm_softmax.py --n-pairs 300   # phần sinh nhiều tuyến trên ít cặp hơn

Bốn mô hình: DM / softmax, mỗi loại học bằng
  nhãn đơn          đường Dijkstra duy nhất
  nhãn nhiều tuyến  mỗi epoch chọn ngẫu nhiên 1 trong các tuyến có trễ <= 1,1 x tối ưu (thuật toán Yen)
Ba phần: (1) độ chính xác, (2) thời gian, (3) sinh nhiều tuyến và đường dự phòng khi hỏng một liên kết.
Kết quả in thành bảng và lưu results/so_sanh_dm_softmax.json.
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
from src.ai.nhieu_tuyen import evaluate_samples, sample_routes  # noqa: E402
from src.ai.train_gnn import build_model, encode_pair, eval_nexthop, eval_rollout, model_chooser  # noqa: E402

MODELS = [  # (tên, tag, head, nhãn nhiều tuyến)
    ("DM, nhãn đơn", "dmx0_4lop", "diffusion2", 0.0),
    ("Softmax, nhãn đơn", "clsB_73", "classifier", 0.0),
    ("DM, nhãn nhiều tuyến", "dm_nt", "diffusion2", 0.1),
    ("Softmax, nhãn nhiều tuyến", "sm_nt", "classifier", 0.1),
]


def train(tag, head, nt, a):
    cmd = [sys.executable, "-u", str(ROOT / "src/ai/train_gnn.py"), "--data-dir", a.data_dir, "--out-dir", a.out_dir,
           "--epochs", str(a.epochs), "--layers", "4", "--head", head, "--tag", tag, "--threads", str(a.threads)]
    if head == "diffusion2":
        cmd += ["--param", "x0ce", "--ddim", "2"]
    if nt > 0:
        cmd += ["--nhieu-tuyen", str(nt)]
    print(">>", " ".join(cmd[2:]), flush=True)
    subprocess.run(cmd, check=True)


def load(path):
    ck = torch.load(path)
    m = build_model(ck["args"], ck["f_in"])
    m.load_state_dict(ck["state"])
    return m.eval()


def pick_config(model, va, is_dm):
    """DM: chọn số bước DDIM và w trên VAL. Softmax: không có gì để chọn."""
    if not is_dm:
        return {}
    best, cfg = -1.0, None
    for n in (1, 2, 5, 20):
        acc, _, _ = eval_nexthop(model, va, (1.0, 2.0, 4.0), n_steps=n)
        for w, v in acc.items():
            if v > best:
                best, cfg = v, {"n_steps": n, "guidance": w}
    return cfg


def accuracy(model, te, kw):
    acc, _, _ = eval_nexthop(model, te, (kw.get("guidance", 1.0),), n_steps=kw.get("n_steps", 1))
    idx = np.arange(len(te))

    def fn(part):
        W = te.W[te.sid[part]]
        dst = np.array([te.paths[i][-1] for i in part])
        hc, hu = encode_pair(model, W, te.static_feat, dst)
        return model_chooser(model, hc, hu, kw.get("guidance", 1.0), kw.get("n_steps", 1))
    t0 = time.perf_counter()
    roll, _ = eval_rollout(te, idx, fn)
    ms_lo = (time.perf_counter() - t0) / len(idx) * 1000
    single = idx[:200]
    t0 = time.perf_counter()
    for i in single:
        eval_rollout(te, np.array([i]), fn)
    ms_1 = (time.perf_counter() - t0) / len(single) * 1000
    return {"nexthop": list(acc.values())[0], "rollout": roll, "ms_tung_duong": ms_1, "ms_theo_lo": ms_lo}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(ROOT / "dataset"))
    ap.add_argument("--out-dir", default=str(ROOT / "results"))
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--n-pairs", type=int, default=600)
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for _, tag, head, nt in MODELS:
        if not (out / f"{tag}.pt").exists():
            train(tag, head, nt, a)

    va, te = SagsinData(Path(a.data_dir) / "val.npz"), SagsinData(Path(a.data_dir) / "test.npz")
    res = {"do_chinh_xac": {}, "nhieu_tuyen": {}}
    models = {}
    print(f"\n(1)+(2) Độ chính xác và thời gian | TEST {len(te):,} đường, {te.N} nút")
    print(f"{'Mô hình':28s} {'cấu hình':>14s} {'next-hop':>9s} {'tới đích':>9s} {'trễ/Dijkstra':>13s} "
          f"{'trùng oracle':>13s} {'ms/đường':>9s} {'ms theo lô':>11s}")
    for name, tag, head, _ in MODELS:
        m = load(out / f"{tag}.pt")
        kw = pick_config(m, va, head == "diffusion2")
        r = accuracy(m, te, kw)
        models[name] = (m, kw)
        res["do_chinh_xac"][name] = {"cau_hinh": kw, **r}
        ro = r["rollout"]
        cfg = f"{kw['n_steps']} bước, w={kw['guidance']:g}" if kw else "-"
        print(f"{name:28s} {cfg:>14s} {r['nexthop'] * 100:8.1f}% {ro['ty_le_toi_dich'] * 100:8.1f}% "
              f"{ro['tre_so_voi_dijkstra_TB']:12.3f}× {ro['trung_khop_duong_oracle'] * 100:12.1f}% "
              f"{r['ms_tung_duong']:9.2f} {r['ms_theo_lo']:11.3f}", flush=True)

    idx = np.sort(np.random.default_rng(0).choice(len(te), min(a.n_pairs, len(te)), replace=False))
    print(f"\n(3) Sinh {a.K} tuyến mỗi cặp | {len(idx)} cặp test | dự phòng: giả sử hỏng từng liên kết của tuyến chính")
    print(f"{'Mô hình':28s} {'cách lấy mẫu':>20s} {'tuyến khác nhau':>16s} {'>=2 tuyến':>10s} "
          f"{'best-of-K':>10s} {'có dự phòng':>12s} {'trễ dự phòng':>13s}")
    for name, (m, kw) in models.items():
        prim = [r[0] for r in sample_routes(m, te, idx, 1, kw, seed0=0)]
        if "DM" in name:
            settings = [("DDIM tất định", {**kw, "eta": 0.0}),
                        ("η=1 (DDPM)", {**kw, "n_steps": 20, "eta": 1.0}),
                        ("η=1, w=0.5", {**kw, "n_steps": 20, "eta": 1.0, "guidance": 0.5})]
        else:
            settings = [("T=1", {"temperature": 1.0}), ("T=2", {"temperature": 2.0}), ("T=3", {"temperature": 3.0})]
        res["nhieu_tuyen"][name] = {}
        for sname, skw in settings:
            t0 = time.perf_counter()
            r = evaluate_samples(te, idx, prim, sample_routes(m, te, idx, a.K, skw))
            r["ms_K_tuyen_moi_cap"] = (time.perf_counter() - t0) / len(idx) * 1000
            res["nhieu_tuyen"][name][sname] = r
            print(f"{name:28s} {sname:>20s} {r['so_tuyen_khac_nhau_TB']:16.2f} {r['ty_le_cap_co_tu_2_tuyen'] * 100:9.1f}% "
                  f"{r['tre_best_of_K_so_Dijkstra']:9.3f}× {r['co_san_du_phong'] * 100:11.1f}% "
                  f"{r['tre_du_phong_so_Dijkstra_chay_lai']:12.3f}×", flush=True)
    res["so_cap_nhieu_tuyen"], res["K"] = len(idx), a.K
    json.dump(res, open(out / "so_sanh_dm_softmax.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2, default=float)
    print(f"\nĐã lưu {out / 'so_sanh_dm_softmax.json'}")


if __name__ == "__main__":
    main()
