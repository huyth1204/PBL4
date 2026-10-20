"""
huan_luyen_kgh.py — Huấn luyện DM hoặc Một-lần cho thí nghiệm 2 (cùng mạng MangKGH, cùng dữ liệu, cùng số epoch).

    python -m src.nhieu_luong.huan_luyen_kgh --kieu dm
    python -m src.nhieu_luong.huan_luyen_kgh --kieu mot_lan

Nhãn = bản đồ cạnh của lời giải TKCB-KGH 10 lần. Chọn checkpoint theo khoảng cách chi phí trung bình tới nhãn trên 300
tình huống val cố định (cấu hình theo dõi: S = 1, 3 bước, γ = 1 ms). Không dùng dữ liệu Yagi.
Ghi results/yagi2/mo_hinh/<kiểu>.pt, <kiểu>_nhat_ky.json, <kiểu>.xong.
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

from src.nhieu_luong import khong_gioi_han as kg
from src.nhieu_luong.mo_hinh_kgh import Lo2, MangKGH, TinhHuong2, giai, mat_mat_dm, mat_mat_mot_lan

GOC = Path(__file__).resolve().parents[2]
RA = GOC / "results" / "yagi2" / "mo_hinh"
MUC = ("nhe", "vua", "nang")


def nap(tap):
    """Danh sách TinhHuong2 (mỗi snapshot × mỗi mức tải); hành lang tính một lần mỗi snapshot."""
    ds = []
    for f in sorted(glob.glob(str(GOC / "data" / "yagi2" / tap / "kb_*.pkl"))):
        for x in pickle.load(open(f, "rb")):
            th0 = TinhHuong2(x, MUC[0])
            ds.append(th0)
            for m in MUC[1:]:
                ds.append(TinhHuong2(x, m, hl=(th0.mask, th0.rc), cap_cua=(th0.cap, th0.cua)))
    return ds


def chia_lo(ds, B, rng):
    idx = rng.permutation(len(ds))
    return [idx[a:a + B] for a in range(0, len(idx), B)]


def danh_gia(net, kieu, ds, seed=0):
    net.eval()
    gen = torch.Generator().manual_seed(seed)
    g = [giai(net, th, kieu, S=1, so_buoc=3, gamma=1.0, gen=gen)[1] / th.chi_phi_nhan - 1 for th in ds]
    net.train()
    return float(np.mean(g) * 100)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--kieu", choices=["dm", "mot_lan"], required=True)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--B", type=int, default=48)
    ap.add_argument("--H", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--luong-cpu", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.set_num_threads(a.luong_cpu)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    RA.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    tr, va = nap("train"), nap("val")
    va_nho = [va[i] for i in np.random.default_rng(99).choice(len(va), 300, replace=False)]
    print(f"{a.kieu}: train {len(tr)}, val {len(va)} tình huống (nạp {time.time() - t0:.0f} s)", flush=True)
    net = MangKGH(H=a.H)
    print("tham số:", sum(p.numel() for p in net.parameters()), flush=True)
    opt = torch.optim.Adam(net.parameters(), lr=a.lr)
    so_lo = math.ceil(len(tr) / a.B) * a.epochs
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * min(s / so_lo, 1))))
    mat = mat_mat_dm if a.kieu == "dm" else mat_mat_mot_lan
    gen = torch.Generator().manual_seed(a.seed)
    nk, tot, t0 = [], np.inf, time.time()
    for ep in range(a.epochs):
        L = []
        for b in chia_lo(tr, a.B, rng):
            lo = Lo2([tr[i] for i in b], voi_nhan=True)
            l = mat(net, lo, gen)
            opt.zero_grad()
            l.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sch.step()
            L.append(l.item())
        with torch.no_grad():
            g = torch.Generator().manual_seed(1)
            net.eval()
            Lv = [mat(net, Lo2(va[j:j + 96], voi_nhan=True), g).item() for j in range(0, len(va), 96)]
            net.train()
        gap = danh_gia(net, a.kieu, va_nho)
        nk.append({"epoch": ep + 1, "loss_train": float(np.mean(L)), "loss_val": float(np.mean(Lv)), "gap_val_%": gap,
                   "giay": time.time() - t0})
        print(f"  ep {ep + 1:2d}  loss {np.mean(L):.4f}  val {np.mean(Lv):.4f}  gap val {gap:.3f}%  ({time.time() - t0:.0f} s)",
              flush=True)
        if gap < tot:
            tot = gap
            torch.save({"state": net.state_dict(), "cau_hinh": net.cau_hinh, "epoch": ep + 1, "gap_val_%": gap},
                       RA / f"{a.kieu}.pt")
        json.dump({"tham_so": vars(a), "nhat_ky": nk, "tot_nhat_gap_val_%": tot},
                  open(RA / f"{a.kieu}_nhat_ky.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    (RA / f"{a.kieu}.xong").write_text(time.strftime("%Y-%m-%d %H:%M:%S"))
    print("xong", a.kieu, flush=True)


if __name__ == "__main__":
    main()
