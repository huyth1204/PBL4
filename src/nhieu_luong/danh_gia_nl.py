"""
danh_gia_nl.py — Chấm DM, softmax và các đối chứng trên tập Yagi (xem results/yagi/TIEU_CHI.md).

    python -m src.nhieu_luong.danh_gia_nl chon-buoc      # chọn số bước khử nhiễu của DM trên val: {5, 10, 20}
    python -m src.nhieu_luong.danh_gia_nl test           # chấm tập Yagi K = 20–40, 80, 160 (1 luồng CPU, tuần tự)

Thời gian đo trong CÙNG một tiến trình, 1 luồng CPU, lần lượt từng phương pháp trên từng tình huống:
DM, softmax, tìm kiếm cục bộ R = 10 và 1 lần (cài đặt mảng, chạy trên tình huống dạng gọn — chỉ các liên kết trên
tuyến ứng viên, nên còn hơi nhanh hơn bản trên mạng đầy đủ), "tìm kiếm cục bộ cùng thời gian" (dừng khi hết đúng thời
gian DM đã dùng). Dijkstra độc lập / tuần tự và Frank–Wolfe lấy từ lúc sinh dữ liệu (cần mạng đầy đủ).
Kết quả từng tình huống: results/yagi/test_<K>.npz; số bước đã chọn: results/yagi/chon_buoc.json.
"""

from __future__ import annotations

import argparse
import glob
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch

from src.nhieu_luong import doi_chung as dc
from src.nhieu_luong.bai_toan import TinhHuong
from src.nhieu_luong.huan_luyen_nl import nap
from src.nhieu_luong.mo_hinh import MangNL, lay_mau_dm, lay_mau_softmax

GOC = Path(__file__).resolve().parents[2]
KQ = GOC / "results" / "yagi"
BUOC = (5, 10, 20)


def nap_mo_hinh(kieu):
    ck = torch.load(KQ / "mo_hinh" / f"{kieu}.pt", weights_only=False)
    net = MangNL(**ck["cau_hinh"])
    net.load_state_dict(ck["state"])
    return net.eval()


def th_gon(x):
    """TinhHuong chỉ gồm các liên kết trên tuyến ứng viên (đánh số lại 0..E−1) — đủ cho tìm kiếm cục bộ."""
    E = len(x["e"])
    N = int(np.ceil(np.sqrt(E)))
    pad = lambda a, v: np.concatenate([np.asarray(a, float), np.full(N * N - E, v)])
    K = len(x["lam"])
    return TinhHuong(N, np.ones(N, bool), np.zeros(N * N, bool), pad(x["prop"], 0.0), pad(x["C"], 1.0),
                     pad(x["rho0"], 0.0), np.zeros(K, int), np.zeros(K, int), np.asarray(x["lam"], float), [],
                     x["canh"].astype(np.int64), x["hop_le"].copy())


def ls_cung_thoi_gian(th, han):
    """Tìm kiếm cục bộ từ tham lam trên ứng viên, dừng khi hết `han` giây (tính cả bước tham lam)."""
    t0 = time.perf_counter()
    a = dc.tham_lam_ung_vien(th)
    Ei, m = dc._vec(th)[:2]
    F = th.tai_tu_chon(a)
    while time.perf_counter() - t0 < han:
        doi = False
        for k in range(th.K):
            if time.perf_counter() - t0 >= han:
                break
            F[Ei[k, a[k]][m[k, a[k]]]] -= th.lam[k]
            gia = dc._gia_ung_vien(th, F, k)
            c = int(np.argmin(gia))
            if c != a[k] and gia[c] < gia[a[k]] - 1e-9:
                a[k], doi = c, True
            F[Ei[k, a[k]][m[k, a[k]]]] += th.lam[k]
        if not doi:
            break
    return a


def chon_buoc(cac_buoc=BUOC):
    """Mỗi số bước ghi một tệp chon_buoc_<n>.json (chạy song song được); đủ cả ba thì gộp và chọn."""
    torch.set_num_threads(1)
    va = nap("val")
    net = nap_mo_hinh("dm")
    for n in cac_buoc:
        f = KQ / f"chon_buoc_{n}.json"
        if f.exists():
            continue
        gen = torch.Generator().manual_seed(0)
        t0 = time.time()
        g = [lay_mau_dm(net, x, 16, n, gen)[1] / x["chi_phi_nhan"] - 1 for x in va]
        json.dump({"gap_tb_%": float(np.mean(g) * 100), "giay": time.time() - t0}, open(f, "w"))
        print(n, json.load(open(f)), flush=True)
    if all((KQ / f"chon_buoc_{n}.json").exists() for n in BUOC):
        kq = {n: json.load(open(KQ / f"chon_buoc_{n}.json")) for n in BUOC}
        tot = min(BUOC, key=lambda n: kq[n]["gap_tb_%"])
        json.dump({"so_buoc": tot, "theo_buoc": kq, "so_tinh_huong_val": len(va)}, open(KQ / "chon_buoc.json", "w"),
                  indent=1)
        print("chọn", tot)


def cham_test(ten_k, so_buoc, net_dm, net_sm, phan=0, so_phan=1):
    """Chấm từng kịch bản, ghi results/yagi/test_tho/<K>/kb_XXXX.npz (bỏ qua tệp đã có → tiếp tục được)."""
    tep = sorted(glob.glob(str(GOC / "data" / "yagi" / "test" / ten_k / "kb_*.pkl")))
    thu_muc = KQ / "test_tho" / ten_k
    thu_muc.mkdir(parents=True, exist_ok=True)
    for j, f in enumerate(tep):
        ra = thu_muc / (Path(f).stem + ".npz")
        if j % so_phan != phan or ra.exists():
            continue
        cot = {k: [] for k in ("kb", "t", "muc", "K", "E", "DM", "SM", "LS10", "LS1", "LS_cung_tg", "doc_lap",
                               "tuan_tu", "FW", "qua_tai_tuan_tu", "qua_tai_LS10", "qua_tai_DM", "tg_DM", "tg_SM",
                               "tg_LS10", "tg_LS1", "tg_tuan_tu", "LS10_luc_sinh")}
        t_bd = time.time()
        for x in pickle.load(open(f, "rb")):
            gen = torch.Generator().manual_seed(1000 * x["kb"] + x["t"])
            s = time.perf_counter()
            a_dm, c_dm, _ = lay_mau_dm(net_dm, x, 16, so_buoc, gen)
            tg_dm = time.perf_counter() - s
            s = time.perf_counter()
            _, c_sm, _ = lay_mau_softmax(net_sm, x, 16, gen)
            tg_sm = time.perf_counter() - s
            th = th_gon(x)
            s = time.perf_counter()
            _, c10 = dc.ls_nhieu_lan(th, R=10, seed=int(x["t"]))
            tg10 = time.perf_counter() - s
            th = th_gon(x)
            s = time.perf_counter()
            a1 = dc.tim_cuc_bo(th, dc.tham_lam_ung_vien(th))
            tg1 = time.perf_counter() - s
            c1 = th.chi_phi(a1)
            th = th_gon(x)
            c_cung = th.chi_phi(ls_cung_thoi_gian(th, tg_dm))
            d = x["doi_chung"]
            for k, v in (("kb", x["kb"]), ("t", x["t"]), ("muc", x["muc"]), ("K", len(x["lam"])), ("E", len(x["e"])),
                         ("DM", c_dm), ("SM", c_sm), ("LS10", c10), ("LS1", c1), ("LS_cung_tg", c_cung),
                         ("doc_lap", d["doc_lap"]), ("tuan_tu", d["tuan_tu"]), ("FW", d["FW_can_duoi"]),
                         ("qua_tai_tuan_tu", d["qua_tai_tuan_tu"]), ("qua_tai_LS10", d["qua_tai_LS10"]),
                         ("qua_tai_DM", th.qua_tai_F(th.tai_tu_chon(a_dm))), ("tg_DM", tg_dm), ("tg_SM", tg_sm),
                         ("tg_LS10", tg10), ("tg_LS1", tg1), ("tg_tuan_tu", x["thoi_gian_luc_sinh"]["tuan_tu"]),
                         ("LS10_luc_sinh", d["LS10"])):
                cot[k].append(v)
        np.savez(ra, **{k: np.array(v) for k, v in cot.items()})
        print(f"  {ten_k} {Path(f).stem}: {len(cot['kb'])} tình huống, DM tb {np.mean(cot['tg_DM']) * 1e3:.0f} ms, "
              f"LS10 tb {np.mean(cot['tg_LS10']) * 1e3:.0f} ms ({time.time() - t_bd:.0f} s)", flush=True)


def gop(ten_k):
    """Gộp các tệp từng kịch bản thành results/yagi/test_<K>.npz khi đủ."""
    n = len(glob.glob(str(GOC / "data" / "yagi" / "test" / ten_k / "kb_*.pkl")))
    tep = sorted(glob.glob(str(KQ / "test_tho" / ten_k / "kb_*.npz")))
    if len(tep) < n:
        return False
    ds = [dict(np.load(f)) for f in tep]
    np.savez(KQ / f"test_{ten_k}.npz", **{k: np.concatenate([d[k] for d in ds]) for k in ds[0]})
    return True


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("lenh", choices=["chon-buoc", "test"])
    ap.add_argument("--k", default="k20_40,k80,k160")
    ap.add_argument("--buoc", default="5,10,20")
    ap.add_argument("--phan", type=int, default=0)
    ap.add_argument("--so-phan", type=int, default=1)
    a = ap.parse_args()
    if a.lenh == "chon-buoc":
        return chon_buoc(tuple(int(b) for b in a.buoc.split(",")))
    torch.set_num_threads(1)
    so_buoc = json.load(open(KQ / "chon_buoc.json"))["so_buoc"]
    net_dm, net_sm = nap_mo_hinh("dm"), nap_mo_hinh("softmax")
    for ten_k in a.k.split(","):
        cham_test(ten_k, so_buoc, net_dm, net_sm, a.phan, a.so_phan)
        gop(ten_k)
    if all(gop(k) for k in ("k20_40", "k80", "k160")):
        (KQ / "test.xong").write_text(time.strftime("%Y-%m-%d %H:%M:%S"))


if __name__ == "__main__":
    main()
