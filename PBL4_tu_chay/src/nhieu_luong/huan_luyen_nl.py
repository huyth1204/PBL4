"""
huan_luyen_nl.py — Huấn luyện DM hoặc softmax cho bài toán nhiều luồng (cùng mạng MangNL, cùng dữ liệu, cùng số epoch).

    python -m src.nhieu_luong.huan_luyen_nl --kieu dm
    python -m src.nhieu_luong.huan_luyen_nl --kieu softmax

Nhãn = lời giải tốt nhất của tìm kiếm cục bộ R = 10 lần. Chọn checkpoint theo khoảng cách chi phí trung bình tới nhãn
trên 300 tình huống val cố định (DM: 10 bước, S = 16; softmax: S = 16). Không dùng dữ liệu Yagi.
Ghi results/yagi/mo_hinh/<kiểu>.pt, <kiểu>_nhat_ky.json và tệp đánh dấu <kiểu>.xong.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch

from src.nhieu_luong.mo_hinh import Lo, MangNL, lay_mau_dm, lay_mau_softmax, mat_mat_dm, mat_mat_softmax, nhan_lo

GOC = Path(__file__).resolve().parents[2]
RA = GOC / "results" / "yagi" / "mo_hinh"
KHOA = ("e", "prop", "C", "rho0", "lam", "canh", "hop_le", "nhan", "chi_phi_nhan", "kb", "t", "muc")


def nap(tap):
    ds = []
    for f in sorted(glob.glob(str(GOC / "data" / "yagi" / tap / "kb_*.pkl"))):
        ds += [{k: x[k] for k in KHOA} for x in pickle.load(open(f, "rb"))]
    return ds


def chia_lo(ds, B, rng):
    """Xáo trộn, rồi trong từng nhóm 20 lô xếp theo số liên kết để ít phải đệm."""
    idx = rng.permutation(len(ds))
    lo = []
    for a in range(0, len(idx), 20 * B):
        g = sorted(idx[a:a + 20 * B], key=lambda i: (len(ds[i]["e"]), len(ds[i]["lam"])))
        lo += [g[j:j + B] for j in range(0, len(g), B)]
    rng.shuffle(lo)
    return lo


def danh_gia(net, kieu, ds, so_buoc=10, seed=0):
    net.eval()
    gen = torch.Generator().manual_seed(seed)
    gap = []
    for x in ds:
        if kieu == "dm":
            _, c, _ = lay_mau_dm(net, x, 16, so_buoc, gen)
        else:
            _, c, _ = lay_mau_softmax(net, x, 16, gen)
        gap.append(c / x["chi_phi_nhan"] - 1)
    net.train()
    return float(np.mean(gap) * 100)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--kieu", choices=["dm", "softmax"], required=True)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--B", type=int, default=32)
    ap.add_argument("--H", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--luong-cpu", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.set_num_threads(a.luong_cpu)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    RA.mkdir(parents=True, exist_ok=True)
    tr, va = nap("train"), nap("val")
    va_nho = [va[i] for i in np.random.default_rng(99).choice(len(va), 300, replace=False)]
    print(f"{a.kieu}: train {len(tr)}, val {len(va)} tình huống", flush=True)
    net = MangNL(H=a.H)
    opt = torch.optim.Adam(net.parameters(), lr=a.lr)
    so_lo = math.ceil(len(tr) / a.B) * a.epochs
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * min(s / so_lo, 1))))
    mat = mat_mat_dm if a.kieu == "dm" else mat_mat_softmax
    nk, tot, t0 = [], np.inf, time.time()
    for ep in range(a.epochs):
        L = []
        for b in chia_lo(tr, a.B, rng):
            d = [tr[i] for i in b]
            lo = Lo(d)
            l = mat(net, lo, nhan_lo(d, lo.K))
            opt.zero_grad()
            l.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sch.step()
            L.append(l.item())
        with torch.no_grad():
            g = torch.Generator().manual_seed(1)
            Lv = []
            for j in range(0, len(va), 64):
                d = va[j:j + 64]
                lo = Lo(d)
                Lv.append(mat(net.eval(), lo, nhan_lo(d, lo.K), g).item())
            net.train()
        gap = danh_gia(net, a.kieu, va_nho)
        nk.append({"epoch": ep + 1, "loss_train": float(np.mean(L)), "loss_val": float(np.mean(Lv)),
                   "gap_val_%": gap, "giay": time.time() - t0})
        print(f"  ep {ep + 1:2d}  loss {np.mean(L):.4f}  val {np.mean(Lv):.4f}  gap val {gap:.3f}%  "
              f"({time.time() - t0:.0f} s)", flush=True)
        if gap < tot:
            tot = gap
            torch.save({"state": net.state_dict(), "cau_hinh": net.cau_hinh, "epoch": ep + 1, "gap_val_%": gap},
                       RA / f"{a.kieu}.pt")
        json.dump({"tham_so": vars(a), "nhat_ky": nk, "tot_nhat_gap_val_%": tot}, open(RA / f"{a.kieu}_nhat_ky.json", "w",
                  encoding="utf-8"), ensure_ascii=False, indent=1)
    (RA / f"{a.kieu}.xong").write_text(time.strftime("%Y-%m-%d %H:%M:%S"))
    print("xong", a.kieu, flush=True)


if __name__ == "__main__":
    main()
