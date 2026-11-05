"""
tao_du_lieu_rr.py — Sinh dữ liệu cho thí nghiệm 3 "bão mạnh + nhiều luồng, điểm 70/30" (results/bao_rr/TIEU_CHI_3.md).

    python -m src.nhieu_luong.tao_du_lieu_rr --tap train     # 59 cửa sổ của 7 cơn train của bảng rủi ro
    python -m src.nhieu_luong.tao_du_lieu_rr --tap val       # 16 cửa sổ Prapiroon / Trami / Kalmaegi
    python -m src.nhieu_luong.tao_du_lieu_rr --tap test      # Yagi: 13 cửa sổ × 48 mốc test hình học v5

Mạng v5 (224 nút) + thời tiết 74 điểm, tham số mô phỏng lấy ĐÚNG từ data/bao_manh/bang_tra_cuoi/meta.json (bắt buộc
theo hợp đồng của bảng). Mỗi cửa sổ ghi data/bao_rr/<tập>/kb_XXXX.pkl (chạy lại bỏ qua tệp đã có → tiếp tục được khi
bị ngắt); xong cả tập ghi data/bao_rr/<tập>/.xong. Mỗi mốc sinh đủ 3 mức tải; nhãn = lời giải tối ưu chính xác.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from src.ngau_nhien.kich_ban import MangGoc, ThamSo, sinh_kich_ban
from src.ngau_nhien.ma_tran_rui_ro import DEFAULT_VAL, ma_tran_online, nap_bang
from src.ngau_nhien.thoi_tiet import ThoiTiet
from src.nhieu_luong.bai_toan_rr import DELTA, TUOI, tao_tinh_huong, toi_uu

GOC = Path(__file__).resolve().parents[2]
RA = GOC / "data" / "bao_rr"
RAW = GOC / "dataset" / "oracle_dataset_v5.npz"
THOI_TIET = GOC / "data" / "thoi_tiet_vn_mr.npz"
BANG = GOC / "data" / "bao_manh" / "bang_tra_cuoi"
SPLIT = GOC / "dataset" / "v5" / "split_info.json"
DANH_MUC = GOC / "data" / "bao_manh" / "danh_muc.json"
YAGI = ("2024-09-05T00:00", "2024-09-08T23:00")
BUOC_MOC = {"train": 7, "val": 2, "test": 1}           # lấy mỗi n mốc của split hình học tương ứng
SEED = {"train": 300000, "val": 400000, "test": 9000}
LICH_SU = 45                                            # ma_tran_online cần 15 phút lịch sử trước quan sát

_g = _tt = _T = _p = _bang = None


def _khoi_tao_nhe():
    """Chỉ thời tiết — đủ để liệt kê cửa sổ / mốc ở tiến trình chính (MangGoc 224 nút tốn ~1,6 GB mỗi tiến trình)."""
    global _tt, _T
    _tt = ThoiTiet(str(THOI_TIET))
    _T = [str(x) for x in _tt.time_utc]


def _khoi_tao():
    global _g, _tt, _T, _p, _bang
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    meta = json.loads((BANG / "meta.json").read_text(encoding="utf-8"))
    _p = ThamSo(**meta["parameters"])
    _bang = nap_bang(BANG / "bang_tra.npz", tham_so=_p)
    raw = np.load(RAW, allow_pickle=False)
    _tt = ThoiTiet(str(THOI_TIET))
    _g = MangGoc(raw["W_snapshots"], raw["node_kind"], raw["node_order"], _tt)
    _T = [str(x) for x in _tt.time_utc]


def cua_so_yagi(tt):
    """Đúng luật chọn cửa sổ của bao_manh.py: giờ có ≥ 5% điểm mưa > 11 mm/h hoặc ≥ 20% điểm gió giật > 15 m/s;
    bắt đầu mỗi 3 giờ, cửa sổ [h, h+4] phải chứa giờ mạnh."""
    T = [str(x) for x in tt.time_utc]
    idx = np.arange(T.index(YAGI[0]), T.index(YAGI[1]) + 1)
    manh = ((tt.precip[idx] > 11).mean(1) >= .05) | ((tt.gust[idx] > 15).mean(1) >= .20)
    return [int(idx[h]) for h in range(0, len(idx) - 4, 3) if manh[h:h + 5].any()]


def cac_cua_so(tap):
    """[(tên bão, h0, seed)]."""
    if tap == "test":
        return [("yagi_2024", h0, SEED["test"] + i) for i, h0 in enumerate(cua_so_yagi(_tt))]
    dm = json.loads(DANH_MUC.read_text(encoding="utf-8"))
    ra = []
    for j, b in enumerate(dm["bao"]):
        if (b["ten"] in DEFAULT_VAL) != (tap == "val"):
            continue
        goc = _T.index(b["start_utc"])
        ra += [(b["ten"], goc + h, SEED[tap] + 1000 * j + k) for k, h in enumerate(b["gio_bat_dau_kich_ban"])]
    return ra


def cac_moc(tap):
    ids = np.array(json.loads(SPLIT.read_text(encoding="utf-8"))["snapshot_ids"][tap], int)
    with np.load(RAW, allow_pickle=False) as z:
        so_snap = z["W_snapshots"].shape[0]
    ids = ids[(ids >= TUOI + LICH_SU) & (ids + DELTA < so_snap)]
    return ids[::BUOC_MOC[tap]]


def _mot_cua_so(job):
    tap, i, ten, h0, seed, moc, tep = job
    if _g is None:
        _khoi_tao()
    if Path(tep).exists():
        return tep, 0, 0.0
    t_bd = time.time()
    Wn, q = sinh_kich_ban(_g, _tt, h0, seed, _p)
    rng = np.random.default_rng(seed + 7)
    ra = []
    for t in moc:
        t = int(t)
        o = t - TUOI
        past = {k: q[k][:o + 1] for k in ("R", "gust", "giong", "tat")}
        r = ma_tran_online(_bang, _g, Wn[o], past, t, TUOI, DELTA, tham_so=_p)
        bo = tao_tinh_huong(_g, Wn, q, r, t, rng)
        if bo is None:
            continue
        for x in bo.values():
            s = time.perf_counter()
            a, J = toi_uu(x)
            x.update({"nhan": a.astype(np.int8), "J_nhan": float(J), "tg_toi_uu": time.perf_counter() - s,
                      "kb": int(i), "bao": ten, "h0": int(h0), "gio": _T[h0], "seed_kb": int(seed)})
            ra.append(x)
    tam = str(tep) + ".tam"
    with open(tam, "wb") as f:
        pickle.dump(ra, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tam, tep)
    return tep, len(ra), time.time() - t_bd


def cac_viec(tap):
    _khoi_tao_nhe()
    moc = cac_moc(tap)
    return [(tap, i, ten, h0, seed, moc, RA / tap / f"kb_{i:04d}.pkl")
            for i, (ten, h0, seed) in enumerate(cac_cua_so(tap))]


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--tap", choices=["train", "val", "test"], required=True)
    ap.add_argument("--tien-trinh", type=int, default=2)
    a = ap.parse_args()
    jobs = cac_viec(a.tap)
    (RA / a.tap).mkdir(parents=True, exist_ok=True)
    con = [j for j in jobs if not Path(j[-1]).exists()]
    print(f"{a.tap}: {len(jobs)} cửa sổ × {len(jobs[0][5])} mốc, còn {len(con)} cửa sổ", flush=True)
    t0, n = time.time(), 0
    with Pool(a.tien_trinh, initializer=_khoi_tao) as p:
        for k, (tep, so, tg) in enumerate(p.imap_unordered(_mot_cua_so, con)):
            n += so
            print(f"  {k + 1}/{len(con)}  {Path(tep).name}  {so} tình huống  {tg:.1f} s  "
                  f"(tổng {n}, {time.time() - t0:.0f} s)", flush=True)
    (RA / a.tap / ".xong").write_text(time.strftime("%Y-%m-%d %H:%M:%S"))
    print("xong", a.tap, flush=True)


if __name__ == "__main__":
    main()
