"""
huan_luyen_dt.py — Huấn luyện DM / softmax chọn tập tuyến (thí nghiệm 4 và 6, results/bao_da_tuyen/TIEU_CHI_4.md, _6.md).

    python -m src.nhieu_luong.huan_luyen_dt dm                           # thí nghiệm 4
    python -m src.nhieu_luong.huan_luyen_dt softmax
    python -m src.nhieu_luong.huan_luyen_dt chon-buoc                    # số bước DM ∈ {3, 5, 10} trên val, S = 8
    python -m src.nhieu_luong.huan_luyen_dt dm --thi-nghiem 6 --ten dm6 --them      # thí nghiệm 6, có đặc trưng mới
    python -m src.nhieu_luong.huan_luyen_dt dm --thi-nghiem 6 --ten dm6a            # thí nghiệm 6, không đặc trưng mới
    python -m src.nhieu_luong.huan_luyen_dt chon-buoc --thi-nghiem 6 --ten dm6

Nhãn = lời giải của thầy (TKCB trên J của 32 thế giới NHÃN), chỉ ở K = 4–5. Chọn checkpoint theo J trung bình trên 300
tình huống val cố định (thí nghiệm 6: 150 val bão + 150 val6), chấm bằng 32 thế giới CHẤM (seed độc lập với bộ nhãn).
Không dùng dữ liệu test.
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

from src.nhieu_luong import da_tuyen as dt
from src.nhieu_luong import mo_hinh_dt as md

GOC = Path(__file__).resolve().parents[2]
DL = GOC / "data" / "bao_da_tuyen"
BUOC, S = (3, 5, 10), 8
THI_NGHIEM = {4: {"train": ("train_k4_5",), "val": ("val_k4_5",), "thu_muc": md.KQ},
              6: {"train": ("train_k4_5", "train6_k4_5"), "val": ("val_k4_5", "val6_k4_5"),
                  "thu_muc": GOC / "results" / "bao_da_tuyen" / "mo_hinh6"},
              7: {"train": ("train_k4_5", "train6_k4_5"), "val": ("val_k4_5", "val6_k4_5"),
                  "thu_muc": GOC / "results" / "bao_da_tuyen" / "mo_hinh7"}}
GAP_DOI = ("train7s_k4_5", "train7x_k4_5")      # thí nghiệm 7b: thêm vào để có mốc 200%


def nap(tap, K="k4_5"):
    return nap_thu_muc(f"{tap}_{K}")


def nap_thu_muc(ten, ti_le=1.0):
    """Mọi bản ghi của thư mục; ti_le < 1 (thí nghiệm 7b): chỉ lấy ngẫu nhiên round(ti_le · số cửa sổ) cửa sổ (tệp),
    default_rng(7)."""
    teps = sorted(glob.glob(str(DL / ten / "kb_*.pkl")))
    if ti_le < 1:
        teps = [teps[i] for i in sorted(np.random.default_rng(7).choice(len(teps), round(ti_le * len(teps)), replace=False))]
    return [r for f in teps for r in pickle.load(open(f, "rb"))]


def the_gioi_cham(r):
    """Thế giới chấm đã lưu (sống, dung lượng còn trống nhỏ nhất) → TheGioi trên kho."""
    if "_tg" not in r:
        du = r["cham"]["du"].astype(float)
        r["_tg"] = dt.TheGioi(r["gon"], r["cham"]["song"], du[:, None], np.zeros_like(du)[:, None])
    return r["_tg"]


def val_co_dinh(va, n=300):
    return [va[i] for i in np.random.default_rng(99).choice(len(va), min(n, len(va)), replace=False)]


def val_tron(cac_val, n=300):
    """n tình huống val cố định, chia đều cho các thư mục val."""
    return [r for va in cac_val for r in val_co_dinh(va, n // len(cac_val))]


def danh_gia(net, kieu, ds, so_buoc=5, seed=0):
    """J trung bình (thế giới chấm) của lời giải mô hình, và của thầy trên cùng thế giới."""
    net.eval()
    gen = torch.Generator().manual_seed(seed)
    J, Jt = [], []
    for r in ds:
        a = md.chon_dm(net, r["gon"], S, so_buoc, gen) if kieu == "dm" else md.chon_softmax(net, r["gon"])
        J.append(the_gioi_cham(r).cham(r["gon"].tu_lua_chon(a))["J"])
        Jt.append(r["ens"]["thay"]["J"])
    net.train()
    return float(np.mean(J)), float(np.mean(Jt))


def chia_lo(ds, B, rng):
    idx = rng.permutation(len(ds))
    lo = []
    for a in range(0, len(idx), 20 * B):
        g = sorted(idx[a:a + 20 * B], key=lambda i: (ds[i][0].K, len(ds[i][0].U)))
        lo += [g[j:j + B] for j in range(0, len(g), B)]
    rng.shuffle(lo)
    return lo


def huan_luyen(kieu, epochs=20, B=32, H=64, lr=1e-3, seed=0, tn=4, ten=None, them=False, ti_le=1.0, gap_doi=False):
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    cfg = THI_NGHIEM[tn]
    ra, ten = cfg["thu_muc"], ten or kieu
    ra.mkdir(parents=True, exist_ok=True)
    cac_train = cfg["train"] + (GAP_DOI if gap_doi else ())
    tr = [(r["gon"], r["nhan"]) for d in cac_train for r in nap_thu_muc(d, ti_le)]
    cac_val = [nap_thu_muc(d) for d in cfg["val"]]
    va_nho = val_tron(cac_val)
    vl = [(r["gon"], r["nhan"]) for va in cac_val for r in va]
    print(f"{ten} ({kieu}, đặc trưng mới: {them}, tỉ lệ {ti_le}, gấp đôi {gap_doi}): train {len(tr)}, val {len(vl)}",
          flush=True)
    net = md.MangDT(H=H, them=them)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    so_lo = math.ceil(len(tr) / B) * epochs
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * min(s / so_lo, 1))))
    mat = md.mat_mat_dm if kieu == "dm" else md.mat_mat_softmax
    nk, tot, t0 = [], np.inf, time.time()
    for ep in range(epochs):
        L = []
        for b in chia_lo(tr, B, rng):
            d = [tr[i] for i in b]
            lo = md.LoDT([x for x, _ in d], them=them)
            l = mat(net, lo, md.trang_thai_nhan(d, lo.K))
            opt.zero_grad()
            l.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sch.step()
            L.append(l.item())
        with torch.no_grad():
            g = torch.Generator().manual_seed(1)
            Lv = []
            for j in range(0, len(vl), 64):
                d = vl[j:j + 64]
                lo = md.LoDT([x for x, _ in d], them=them)
                Lv.append(mat(net.eval(), lo, md.trang_thai_nhan(d, lo.K), g).item())
            net.train()
        J, Jt = danh_gia(net, kieu, va_nho)
        nk.append({"epoch": ep + 1, "loss_train": float(np.mean(L)), "loss_val": float(np.mean(Lv)), "J_val": J,
                   "J_thay_val": Jt, "giay": time.time() - t0})
        print(f"  ep {ep + 1:2d}  loss {np.mean(L):.4f}  val {np.mean(Lv):.4f}  J val {J:.4f} (thầy {Jt:.4f})  "
              f"({time.time() - t0:.0f} s)", flush=True)
        if J < tot:
            tot = J
            torch.save({"state": net.state_dict(), "cau_hinh": net.cau_hinh, "epoch": ep + 1, "J_val": J}, ra / f"{ten}.pt")
        json.dump({"nhat_ky": nk, "tot_nhat_J_val": tot}, open(ra / f"{ten}_nhat_ky.json", "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)
    (ra / f"{ten}.xong").write_text(time.strftime("%Y-%m-%d %H:%M:%S"))


def chon_buoc(tn=4, ten="dm"):
    torch.set_num_threads(1)
    cfg = THI_NGHIEM[tn]
    va = val_tron([nap_thu_muc(d) for d in cfg["val"]])
    net = md.nap(ten, cfg["thu_muc"])
    bang = {}
    for n in BUOC:
        s = time.perf_counter()
        J, Jt = danh_gia(net, "dm", va, n)
        bang[n] = {"J_val": J, "J_thay_val": Jt, "giay": time.perf_counter() - s}
        print(n, bang[n], flush=True)
    tot = min(BUOC, key=lambda n: (round(bang[n]["J_val"], 9), n))
    tep = cfg["thu_muc"] / ("chon_buoc.json" if ten == "dm" else f"chon_buoc_{ten}.json")
    json.dump({"S": S, "so_buoc": tot, "bang": bang, "quy_tac": "J val nhỏ nhất (thế giới chấm); hoà → ít bước"},
              open(tep, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("chọn", tot)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("viec", choices=["dm", "softmax", "chon-buoc"])
    ap.add_argument("--thi-nghiem", type=int, default=4, choices=sorted(THI_NGHIEM))
    ap.add_argument("--ten", default=None)
    ap.add_argument("--them", action="store_true", help="thêm 3 đặc trưng giông quanh UAV (thí nghiệm 6)")
    ap.add_argument("--ti-le", type=float, default=1.0, help="thí nghiệm 7b: phần cửa sổ train dùng")
    ap.add_argument("--gap-doi", action="store_true", help="thí nghiệm 7b: thêm dữ liệu train7s + train7x")
    a = ap.parse_args()
    if a.viec == "chon-buoc":
        chon_buoc(a.thi_nghiem, a.ten or "dm")
    else:
        huan_luyen(a.viec, tn=a.thi_nghiem, ten=a.ten, them=a.them, ti_le=a.ti_le, gap_doi=a.gap_doi)
