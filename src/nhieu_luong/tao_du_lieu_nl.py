"""
tao_du_lieu_nl.py — Sinh dữ liệu cho thí nghiệm "bão Yagi + nhiều luồng" (xem results/yagi/TIEU_CHI.md).

    python -m src.nhieu_luong.tao_du_lieu_nl --tap train     # 600 kịch bản bão / mưa lớn khác, 15 snapshot mỗi kịch bản
    python -m src.nhieu_luong.tao_du_lieu_nl --tap val       # 60 kịch bản, ngày khác với train
    python -m src.nhieu_luong.tao_du_lieu_nl --tap test      # Yagi: 16 kịch bản × 30 snapshot, K = 20–40, 80, 160

Mỗi kịch bản ghi một tệp data/yagi/<tập>/kb_XXXX.pkl (chạy lại sẽ bỏ qua tệp đã có → tiếp tục được khi bị ngắt);
xong cả tập ghi tệp đánh dấu data/yagi/<tập>/.xong.
Một tình huống được lưu ở dạng gọn: chỉ các liên kết nằm trên tuyến ứng viên (đủ để tính chi phí chính xác), kèm
nhãn = lời giải tốt nhất của tìm kiếm cục bộ R = 10 lần. Tập test lưu thêm kết quả của mọi đối chứng (tính trên mạng đầy đủ).
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from src.ngau_nhien.kich_ban import MangGoc, ThamSo, sinh_kich_ban
from src.ngau_nhien.khoang_loi import MUC
from src.ngau_nhien.thoi_tiet import ThoiTiet
from src.nhieu_luong import doi_chung as dc
from src.nhieu_luong.bai_toan import tao_tinh_huong

GOC = Path(__file__).resolve().parents[2]
RA = GOC / "data" / "yagi"
NGAY_YAGI = ("2024-09-04", "2024-09-12")            # loại khỏi train / val (cả hai đầu)
SO_GIO_KB = 600 * 20 / 3600                          # một kịch bản dài 600 snapshot × 20 s
CAU_HINH = {"train": dict(so_kb=600, so_snap=15, buoc=40, seed=11),
            "val": dict(so_kb=60, so_snap=15, buoc=40, seed=12)}
TEST_K = {"k20_40": (20, 40), "k80": (80, 80), "k160": (160, 160)}

_g = _tt = _T = None


def _khoi_tao():
    global _g, _tt, _T
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    raw = np.load(GOC / "dataset" / "oracle_dataset_v2.npz")
    _tt = ThoiTiet(str(GOC / "data" / "thoi_tiet_vn.npz"))
    _g = MangGoc(raw["W_snapshots"], raw["node_kind"], raw["node_order"], _tt)
    _T = [str(x) for x in _tt.time_utc]


# ---------------------------------------------------------------- chọn giờ bắt đầu cho train / val
def diem_gio(tt):
    """Mức bão theo giờ = max trên 13 điểm mặt đất của mưa (mm/h) + phần gió giật vượt 15 m/s."""
    return (tt.precip[:, :13] + np.maximum(0.0, tt.gust[:, :13] - 15.0)).max(1)


def chon_gio(tt, tap):
    """Danh sách (h0, seed) cho train / val. 70% kịch bản có mức bão ≥ 10 trong cửa sổ, 30% chỉ cần có mưa (≥ 1).
    Chia theo ngày: 15% số ngày cho val; cửa sổ phải nằm trọn trong ngày thuộc đúng tập; loại 04–12/9/2024."""
    s = diem_gio(tt)
    T = [str(x) for x in tt.time_utc]
    ngay = np.array([x[:10] for x in T])
    so = int(np.ceil(SO_GIO_KB)) + 1
    h = np.arange(len(T) - so)
    s_cs = np.array([s[i:i + so].max() for i in h])
    d0, d1 = ngay[h], ngay[h + so - 1]
    ngoai_yagi = ~(((d0 >= NGAY_YAGI[0]) & (d0 <= NGAY_YAGI[1])) | ((d1 >= NGAY_YAGI[0]) & (d1 <= NGAY_YAGI[1])))
    cac_ngay = np.unique(ngay)
    r = np.random.default_rng(2026)
    ngay_val = set(r.choice(cac_ngay, size=int(0.15 * len(cac_ngay)), replace=False))
    la_val = np.array([d in ngay_val for d in d0]) & np.array([d in ngay_val for d in d1])
    la_train = np.array([d not in ngay_val for d in d0]) & np.array([d not in ngay_val for d in d1])
    hop = ngoai_yagi & (la_val if tap == "val" else la_train)
    cfg = CAU_HINH[tap]
    rng = np.random.default_rng(cfg["seed"])
    nang = h[hop & (s_cs >= 10)]
    mua = h[hop & (s_cs >= 1)]
    n_nang = int(round(0.7 * cfg["so_kb"]))
    chon = list(rng.choice(nang, size=n_nang, replace=len(nang) < n_nang)) + \
        list(rng.choice(mua, size=cfg["so_kb"] - n_nang, replace=False))
    return [(int(x), int(cfg["seed"] * 100000 + i)) for i, x in enumerate(chon)]


def gio_test():
    """16 kịch bản Yagi: bắt đầu mỗi 3 giờ từ 2024-09-07T00:00 UTC (đã chốt trong TIEU_CHI.md)."""
    return [_T.index(f"2024-09-0{7 + i * 3 // 24}T{i * 3 % 24:02d}:00") for i in range(16)]


# ---------------------------------------------------------------- dạng gọn
def nen(th):
    """Chỉ giữ các liên kết nằm trên tuyến ứng viên; chỉ số cạnh đổi sang chỉ số cục bộ 0..E−1."""
    dung = np.unique(th.canh[th.canh >= 0])
    loc = -np.ones(th.N * th.N, int)
    loc[dung] = np.arange(len(dung))
    canh = np.where(th.canh >= 0, loc[np.maximum(th.canh, 0)], -1).astype(np.int16)
    return {"e": dung.astype(np.int32), "prop": th.prop[dung], "C": th.C[dung], "rho0": th.rho0[dung],
            "src": th.src.astype(np.int16), "dst": th.dst.astype(np.int16), "lam": th.lam.astype(float),
            "canh": canh, "hop_le": th.hop_le.copy(), "meta": dict(th.meta)}


def _mot_kich_ban(job):
    """job = (tập, chỉ số, h0, seed, cách lấy snapshot, khoảng K)."""
    tap, i, h0, seed, snaps, so_luong, tep = job
    if _g is None:
        _khoi_tao()
    if Path(tep).exists():
        return tep, 0, 0.0
    t_bd = time.time()
    _, q = sinh_kich_ban(_g, _tt, h0, seed, ThamSo(**MUC["thường"]))
    rng = np.random.default_rng(seed + 7)
    cache, ra = {}, []
    for t in snaps:
        bo = tao_tinh_huong(_g, q, int(t), rng, so_luong=so_luong, cache=cache)
        if bo is None:
            continue
        for ten, th in bo.items():
            s = time.perf_counter()
            a, c = dc.ls_nhieu_lan(th, R=10, seed=int(t))
            t_ls10 = time.perf_counter() - s
            x = nen(th)
            x.update({"nhan": a.astype(np.int8), "chi_phi_nhan": float(c), "h0": int(h0), "gio": _T[h0],
                      "seed_kb": int(seed), "kb": int(i), "t": int(t), "muc": ten})
            if tap == "test":
                kq = {"LS10": c}
                tg = {"LS10": t_ls10}
                s = time.perf_counter()
                c0 = th.chi_phi(dc.doc_lap(th))
                tg["doc_lap"] = time.perf_counter() - s
                kq["doc_lap"] = c0
                s = time.perf_counter()
                _, F = dc.tuan_tu(th)
                tg["tuan_tu"] = time.perf_counter() - s
                kq["tuan_tu"] = th.chi_phi_F(F)
                kq["qua_tai_tuan_tu"] = th.qua_tai_F(F)
                s = time.perf_counter()
                a1 = dc.tim_cuc_bo(th, dc.tham_lam_ung_vien(th))
                tg["LS1"] = time.perf_counter() - s
                kq["LS1"] = th.chi_phi(a1)
                kq["qua_tai_LS10"] = th.qua_tai_F(th.tai_tu_chon(a))
                s = time.perf_counter()
                kq["FW_can_duoi"], kq["FW_chia_luong"] = dc.frank_wolfe(th)
                tg["FW"] = time.perf_counter() - s
                x["doi_chung"], x["thoi_gian_luc_sinh"] = kq, tg
            ra.append(x)
    tam = str(tep) + ".tam"
    with open(tam, "wb") as f:
        pickle.dump(ra, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tam, tep)
    return tep, len(ra), time.time() - t_bd


def cac_viec(tap):
    _khoi_tao()
    jobs = []
    if tap in CAU_HINH:
        cfg = CAU_HINH[tap]
        for i, (h0, seed) in enumerate(chon_gio(_tt, tap)):
            off = int(np.random.default_rng(seed).integers(0, cfg["buoc"]))
            snaps = off + cfg["buoc"] * np.arange(cfg["so_snap"])
            jobs.append((tap, i, h0, seed, snaps[snaps < 600], (20, 40), RA / tap / f"kb_{i:04d}.pkl"))
    else:
        for ten_k, sl in TEST_K.items():
            for i, h0 in enumerate(gio_test()):
                jobs.append(("test", i, h0, 5000 + i, np.arange(0, 600, 20), sl, RA / "test" / ten_k / f"kb_{i:04d}.pkl"))
    return jobs


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--tap", choices=["train", "val", "test"], required=True)
    ap.add_argument("--tien-trinh", type=int, default=2)
    a = ap.parse_args()
    jobs = cac_viec(a.tap)
    for j in jobs:
        Path(j[-1]).parent.mkdir(parents=True, exist_ok=True)
    con = [j for j in jobs if not Path(j[-1]).exists()]
    print(f"{a.tap}: {len(jobs)} kịch bản, còn {len(con)}", flush=True)
    t0, n = time.time(), 0
    with Pool(a.tien_trinh, initializer=_khoi_tao) as p:
        for k, (tep, so, tg) in enumerate(p.imap_unordered(_mot_kich_ban, con)):
            n += so
            if k % 10 == 0 or k == len(con) - 1:
                print(f"  {k + 1}/{len(con)}  {so} tình huống  {tg:.1f} s  (tổng {n}, {time.time() - t0:.0f} s)", flush=True)
    for d in {Path(j[-1]).parent for j in jobs}:
        (d / ".xong").write_text(time.strftime("%Y-%m-%d %H:%M:%S"))
    print("xong", a.tap, flush=True)


if __name__ == "__main__":
    main()
