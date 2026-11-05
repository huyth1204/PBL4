"""
huan_luyen_rr.py — Huấn luyện DM hoặc softmax cho thí nghiệm 3 (cùng mạng MangRR, cùng dữ liệu, cùng số epoch).

    python -m src.nhieu_luong.huan_luyen_rr --kieu dm
    python -m src.nhieu_luong.huan_luyen_rr --kieu softmax

Nhãn = lời giải tối ưu chính xác theo J (0,7·rủi ro + 0,3·tốc độ). Chọn checkpoint theo ΔJ trung bình tới nhãn trên
300 tình huống val cố định (DM: 5 bước, S = 16; softmax: S = 16). Không dùng dữ liệu Yagi.
Ghi results/bao_rr/mo_hinh/<kiểu>.pt, <kiểu>_nhat_ky.json và tệp đánh dấu <kiểu>.xong.
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

from src.nhieu_luong.huan_luyen_nl import chia_lo
from src.nhieu_luong.mo_hinh import mat_mat_dm, mat_mat_softmax, nhan_lo
from src.nhieu_luong.mo_hinh_rr import LoRR, MangRR, lay_mau_dm, lay_mau_softmax

GOC = Path(__file__).resolve().parents[2]
DL = GOC / "data" / "bao_rr"
RA = GOC / "results" / "bao_rr" / "mo_hinh"


def nap(tap):
    ds = []
    for f in sorted(glob.glob(str(DL / tap / "kb_*.pkl"))):
        ds += pickle.load(open(f, "rb"))
    return ds


def val_co_dinh(va, n=300):
    return [va[i] for i in np.random.default_rng(99).choice(len(va), min(n, len(va)), replace=False)]


def danh_gia(net, kieu, ds, so_buoc=5, seed=0):
    """ΔJ trung bình tới lời giải tối ưu (điểm, không phải %: J* có thể bằng 0)."""
    net.eval()
    gen = torch.Generator().manual_seed(seed)
    gap = []
    for x in ds:
        if kieu == "dm":
            _, J, _ = lay_mau_dm(net, x, 16, so_buoc, gen)
        else:
            _, J, _ = lay_mau_softmax(net, x, 16, gen)
        gap.append(J - x["J_nhan"])
    net.train()
    return float(np.mean(gap))


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
    va_nho = val_co_dinh(va)
    print(f"{a.kieu}: train {len(tr)}, val {len(va)} tình huống", flush=True)
    net = MangRR(H=a.H)
    opt = torch.optim.Adam(net.parameters(), lr=a.lr)
    so_lo = math.ceil(len(tr) / a.B) * a.epochs
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * min(s / so_lo, 1))))
    mat = mat_mat_dm if a.kieu == "dm" else mat_mat_softmax
    nk, tot, t0 = [], np.inf, time.time()
    for ep in range(a.epochs):
        L = []
        for b in chia_lo(tr, a.B, rng):
            d = [tr[i] for i in b]
            lo = LoRR(d)
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
                lo = LoRR(d)
                Lv.append(mat(net.eval(), lo, nhan_lo(d, lo.K), g).item())
            net.train()
        gap = danh_gia(net, a.kieu, va_nho)
        nk.append({"epoch": ep + 1, "loss_train": float(np.mean(L)), "loss_val": float(np.mean(Lv)),
                   "dJ_val": gap, "giay": time.time() - t0})
        print(f"  ep {ep + 1:2d}  loss {np.mean(L):.4f}  val {np.mean(Lv):.4f}  ΔJ val {gap:.5f}  "
              f"({time.time() - t0:.0f} s)", flush=True)
        if gap < tot:
            tot = gap
            torch.save({"state": net.state_dict(), "cau_hinh": net.cau_hinh, "epoch": ep + 1, "dJ_val": gap},
                       RA / f"{a.kieu}.pt")
        json.dump({"tham_so": vars(a), "nhat_ky": nk, "tot_nhat_dJ_val": tot}, open(RA / f"{a.kieu}_nhat_ky.json", "w",
                  encoding="utf-8"), ensure_ascii=False, indent=1)
    (RA / f"{a.kieu}.xong").write_text(time.strftime("%Y-%m-%d %H:%M:%S"))
    print("xong", a.kieu, flush=True)


if __name__ == "__main__":
    main()
