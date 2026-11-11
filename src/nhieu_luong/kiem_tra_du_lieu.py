"""
kiem_tra_du_lieu.py — Thí nghiệm 11 (results/bao_da_tuyen/TIEU_CHI_11.md): làm sạch / kiểm tra dữ liệu như một sản phẩm.

- loi_ban_ghi(r): danh sách lỗi của MỘT bản ghi (rỗng = hợp lệ). Huấn luyện TN11 loại bản ghi có lỗi (huan_luyen_dt, "sach").
- bao_cao_thu_muc(d): số bản ghi, lỗi theo loại, bản ghi trùng (kb, t, mức tải), thống kê đặc trưng — để so phân bố
  giữa tập học / val / kiểm tra và giữa thời tiết thường / cực đoan.
"""

from __future__ import annotations

import glob
import pickle
from collections import Counter
from pathlib import Path

import numpy as np

KHOANG = {"risk": (0.0, 1.0), "rho0": (0.0, 1.0)}           # nửa mở [a, b)
DAC_TRUNG_SO = ("R", "dR", "gust", "cape_uav")


def loi_ban_ghi(r, can_nhan=False):
    """Lỗi: bài toán rỗng, giá trị không hữu hạn / ngoài miền, đặc trưng thiếu / sai độ dài, nhãn thầy sai."""
    g, loi = r["gon"], []
    if g.K < 1 or len(g.U) == 0:
        return ["bai_toan_rong"]
    for ten in ("prop", "C", "rho0", "risk", "lam"):
        x = np.asarray(getattr(g, ten), float)
        if not np.all(np.isfinite(x)):
            loi.append(f"{ten}_khong_huu_han")
        elif ten in KHOANG and np.any((x < KHOANG[ten][0]) | (x >= KHOANG[ten][1])):
            loi.append(f"{ten}_ngoai_mien")
        elif ten in ("C", "lam") and np.any(x <= 0):
            loi.append(f"{ten}_khong_duong")
    d = g.dac_trung_canh or {}
    for k, v in d.items():
        v = np.asarray(v)
        if len(v) != len(g.U):
            loi.append(f"dac_trung_{k}_sai_do_dai")
        elif v.dtype.kind == "f" and not np.all(np.isfinite(v)):
            loi.append(f"dac_trung_{k}_khong_huu_han")
    if any(len(o) == 0 for o in g.opts):
        loi.append("luong_khong_co_lua_chon")
    if can_nhan:
        a = r.get("nhan")
        if a is None:
            loi.append("thieu_nhan")
        elif len(a) != g.K or any(not 0 <= int(a[k]) < len(g.opts[k]) for k in range(g.K)):
            loi.append("nhan_sai")
    return loi


def he_so_tai(muc):
    """Mức tải → hệ số: 'nhe' / 'vua' / 'nang' (0,6 / 0,9 / 1,2) hoặc hệ số liên tục."""
    from src.nhieu_luong.bai_toan import MUC_TAI
    return MUC_TAI[muc] if isinstance(muc, str) else float(muc)


def dai_tai(muc):
    """0 thấp [0,5; 0,75) · 1 giữa [0,75; 1) · 2 cao [1; 1,3] — như ô tiêu chí."""
    return int(np.searchsorted([0.75, 1.0], he_so_tai(muc), side="right"))


def bao_cao_thu_muc(thu_muc, can_nhan=False):
    teps = sorted(glob.glob(str(Path(thu_muc) / "kb_*.pkl")))
    loi, khoa, n, so_loi = Counter(), Counter(), 0, 0
    tk = {k: [] for k in ("risk", "rho0", "lam", "K", "so_canh", *DAC_TRUNG_SO)}
    dai = Counter()
    for t in teps:
        for r in pickle.load(open(t, "rb")):
            n += 1
            e = loi_ban_ghi(r, can_nhan)
            so_loi += bool(e)
            loi.update(e)
            khoa[(int(r["kb"]), int(r["t"]), str(r["muc"]))] += 1
            g = r["gon"]
            dai[dai_tai(r["muc"])] += 1
            tk["risk"].append(float(np.mean(g.risk)))
            tk["rho0"].append(float(np.mean(g.rho0)))
            tk["lam"].append(float(np.sum(g.lam)))
            tk["K"].append(g.K)
            tk["so_canh"].append(len(g.U))
            for k in DAC_TRUNG_SO:
                if g.dac_trung_canh and k in g.dac_trung_canh:
                    tk[k].append(float(np.max(g.dac_trung_canh[k])))
    trung = sum(c - 1 for c in khoa.values() if c > 1)
    tom = {k: {"tb": float(np.mean(v)), "p99": float(np.percentile(v, 99)), "max": float(np.max(v))}
           for k, v in tk.items() if v}
    return {"so_tep": len(teps), "so_ban_ghi": n, "so_ban_ghi_loi": so_loi, "loi": dict(loi), "so_trung": trung,
            "dai_tai": {str(k): dai[k] for k in range(3)}, "thong_ke": tom}
