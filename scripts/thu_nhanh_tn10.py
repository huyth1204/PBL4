"""
thu_nhanh_tn10.py — Kiểm tra nhanh máy chạy TN10 trước khi chạy dài (~3–5 phút, KHÔNG ghi vào data/ hay results/):

    python -m scripts.thu_nhanh_tn10

0. Code đủ và đúng bản (bắt lỗi push thiếu tệp), có GPU hay không.
1. Đủ tệp đầu vào; dữ liệu nào đã có (val, val_b, bộ kiểm tra) thì chay_tn10 bỏ qua.
2. Sinh thử 1 cửa sổ × 2 mốc của train10 (nhãn thầy + thế giới chấm) và train_b10 (thế giới B) vào thư mục tạm.
3. Kiểm thử chéo: 4 phần đủ cửa sổ, cùng cửa sổ ở A và B cùng phần.
4. Thầy dùng bộ sinh ngẫu nhiên riêng: val10_t32 và val10_t64 cùng tình huống.
"""

import glob
import os
import pickle
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
GOC = Path(__file__).resolve().parents[1]
DL = GOC / "data" / "bao_da_tuyen"


def kiem_tra_code():
    """Bắt sớm lỗi push thiếu: mọi module TN10 import được và code huấn luyện là bản có --thiet-bi."""
    import importlib
    loi = []
    for m in ("scripts.tn10", "scripts.phan_tich_10", "src.nhieu_luong.huan_luyen_dt", "src.nhieu_luong.cham_lai_dt"):
        try:
            importlib.import_module(m)
        except ImportError as e:
            loi.append(f"{m}: {e}")
    hl, md = sys.modules.get("src.nhieu_luong.huan_luyen_dt"), sys.modules.get("src.nhieu_luong.mo_hinh_dt")
    if (hl is not None and not hasattr(hl, "chon_thiet_bi")) or (md is not None and not hasattr(md, "_rand")):
        loi.append("src/nhieu_luong/huan_luyen_dt.py, mo_hinh_dt.py là bản cũ (chưa có --thiet-bi)")
    if loi:
        raise SystemExit("0. Code THIẾU / CŨ — push đủ tệp rồi `git pull` (Colab: Run all lại):\n   " + "\n   ".join(loi))
    import torch
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "không có (huấn luyện trên CPU)"
    print(f"0. Code: đủ module TN10, huấn luyện có --thiet-bi. GPU: {gpu}")


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    kiem_tra_code()
    can =["data/thoi_tiet_vn_mr.npz", "data/thoi_tiet_vn_2017_2021_1.npz"]
    thieu = [t for t in can if not (GOC / t).exists()]
    if thieu:
        raise SystemExit("1. THIẾU tệp đầu vào: " + ", ".join(thieu) + " — chạy `git pull` lại.")
    co = {d: len(glob.glob(str(DL / d / "kb_*.pkl"))) for d in
          ("val_k4_5", "val6_k4_5", "val_b_k4_5", "val6_b_k4_5", "c6_k4_5", "yagi5_k4_5", "c6_b_k4_5", "yagi5_b_k4_5",
           "c6_b_k10", "c6_b_k20", "ngay9_k4_5", "nam9_k4_5", "bao9_k4_5")}
    print("1. Tệp đầu vào: đủ. Dữ liệu có sẵn (số cửa sổ):", ", ".join(f"{k} {v}" for k, v in co.items()))
    from src.nhieu_luong import tao_du_lieu_dt as tdt
    from src.nhieu_luong import tao_du_lieu_rr as td
    from src.nhieu_luong.huan_luyen_dt import phan_cua
    t0 = time.time()
    td._khoi_tao()
    print(f"   khởi tạo mạng + bảng rủi ro: {time.time() - t0:.0f} s, {os.cpu_count()} luồng CPU")
    with tempfile.TemporaryDirectory() as tam:
        tg_moc = {}
        for tap in ("train10", "train_b10", "val10_t32", "val10_t64"):
            goc = tdt.TAP_10[tap][0]
            goc = tdt.TAP_8.get(goc, goc)
            moc = td.cac_moc(tdt.HINH_HOC.get(goc, goc))
            ten, h0, seed = tdt.cac_cua_so_tap(tap)[2]
            s = time.time()
            tdt._mot_cua_so((tap, (4, 5), 2, ten, h0, seed, moc[10:12], str(Path(tam) / f"{tap}.pkl"), False, True, False))
            tg_moc[tap] = (time.time() - s) / 2
            r = pickle.load(open(Path(tam) / f"{tap}.pkl", "rb"))
            print(f"2. {tap}: ~{tg_moc[tap]:.0f} s / mốc; nhãn thầy: {r[0]['nhan'] is not None}; thế giới chấm: {'cham' in r[0]}")
        a, b = (pickle.load(open(Path(tam) / f"{t}.pkl", "rb")) for t in ("val10_t32", "val10_t64"))
        cung = all(x["t"] == y["t"] and np.allclose(x["gon"].lam, y["gon"].lam) for x, y in zip(a, b))
        print("4. Thầy 32 / 64 cùng tình huống:", "có" if cung else "KHÔNG — dừng lại")
    dem = [sum(phan_cua("train10_k4_5", k) == f for k in range(59)) for f in range(4)]
    cung_phan = all(phan_cua("train10_k4_5", k) == phan_cua("train_b10_k4_5", k) == phan_cua("train10_t64_k4_5", k)
                    for k in range(59))
    print(f"3. Kiểm thử chéo: số cửa sổ train mỗi phần {dem}; A / B / thầy 64 cùng phần: {'có' if cung_phan else 'KHÔNG'}")
    tt = max(1, min(6, (os.cpu_count() or 2) // 2))
    so_moc = 119 * len(td.cac_moc("train"))
    gio = (tg_moc["train10"] + tg_moc["train_b10"]) * so_moc / tt / 3600
    print(f"Ước tính sinh tập học A + B với {tt} tiến trình: ~{gio:.1f} giờ (cộng huấn luyện ~8–10 giờ, chấm ~3 giờ)")
    print("OK — chạy: python -m scripts.chay_tn10 --tien-trinh 4 --song-song 2 --luong-cpu 4 > logs/tn10.log 2>&1")


if __name__ == "__main__":
    main()
