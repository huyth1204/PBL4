"""
thu_nhanh_tn8.py — Kiểm tra nhanh máy chạy TN8 trước khi chạy dài (~3–5 phút, KHÔNG ghi vào data/ hay results/):

    python -m scripts.thu_nhanh_tn8

1. Đủ tệp đầu vào và dữ liệu thế giới A / mô hình DM6.
2. Thế giới A với tham số mặc định không bị đổi (so 1 kịch bản: hai bộ mô phỏng khớp nhau, không có nút hỏng).
3. Thế giới B: loại UAV, tỉ lệ UAV / trạm tắt và hỏng hẳn trên 4 cửa sổ C; hai bộ mô phỏng hỏng giống hệt nhau.
4. Sinh thử 1 cửa sổ × 2 mốc của train_b (có thầy) và c6_b (có chấm) vào thư mục tạm, đo thời gian → ước tính tổng.
"""

import os
import pickle
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.dong_goi_tn8 import CAN, TRONG_GIT  # noqa: E402

GOC = Path(__file__).resolve().parents[1]


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    thieu = [t for t in CAN + TRONG_GIT if not (GOC / t).exists()]
    print("1. Tệp đầu vào:", "đủ" if not thieu else "THIẾU " + ", ".join(thieu))
    if thieu:
        raise SystemExit("Giải nén TN8_du_lieu.zip vào thư mục gốc repo rồi chạy lại.")
    from src.ngau_nhien import kich_ban as KB
    from src.nhieu_luong import tao_du_lieu_dt as tdt
    from src.nhieu_luong import tao_du_lieu_rr as td
    from src.nhieu_luong import the_gioi_rr as tg
    t0 = time.time()
    td._khoi_tao()
    g, tt, p = td._g, td._tt, td._p
    print(f"   khởi tạo mạng + bảng rủi ro: {time.time() - t0:.0f} s, {os.cpu_count()} luồng CPU")
    ten, h0, seed = tdt.cac_cua_so_tap("c6")[0]
    _, q = KB.sinh_kich_ban(g, tt, h0, seed, p)
    pl = tg.phat_lai(g, tt, h0, seed, p)
    print("2. Thế giới A:", "ổn" if np.array_equal(q["tat"], pl["tat"]) and not q["hong"].any() else "LỆCH — dừng lại")
    _, pB = tdt.moi_truong("c6_b")
    print("3. Thế giới B: UAV 12 m/s =", sorted(k for k, v in dict(pB.uav_gio_nguong).items() if v == 12.0))
    uav = [n for n in range(g.N) if g.kind[n] == "uav"]
    dat = [n for n in range(g.N) if g.dat[n]]
    tA, tB, hB, khop = [], [], [], True
    for ten, h0, seed in tdt.cac_cua_so_tap("c6")[:4]:
        _, qa = KB.sinh_kich_ban(g, tt, h0, seed, p)
        _, qb = KB.sinh_kich_ban(g, tt, h0, seed, pB)
        plb = tg.phat_lai(g, tt, h0, seed, pB)
        khop &= np.array_equal(plb["hong"], qb["hong"]) and np.array_equal(qa["R"], qb["R"])
        tA.append([qa["tat"][:, uav].mean(), qa["tat"][:, dat].mean()])
        tB.append([qb["tat"][:, uav].mean(), qb["tat"][:, dat].mean()])
        hB.append([qb["hong"][-1, uav].mean(), qb["hong"][-1, dat].mean()])
    tA, tB, hB = (100 * np.mean(x, 0) for x in (tA, tB, hB))
    print(f"   UAV tắt A {tA[0]:.1f}% → B {tB[0]:.1f}%, hỏng hẳn cuối cửa sổ {hB[0]:.1f}% | trạm tắt A {tA[1]:.1f}% → "
          f"B {tB[1]:.1f}%, hỏng hẳn {hB[1]:.1f}% | hai bộ mô phỏng khớp: {'có' if khop else 'KHÔNG — dừng lại'}")
    with tempfile.TemporaryDirectory() as tam:
        tg_mot = {}
        for tap, thay in (("train_b", True), ("c6_b", False)):
            goc = tdt.TAP_8[tap]
            moc = td.cac_moc(tdt.HINH_HOC.get(goc, goc))
            ten, h0, seed = tdt.cac_cua_so_tap(tap)[0]
            do = []
            for so_moc in (1, 3):                                  # tách chi phí cố định / cửa sổ và chi phí / mốc
                s = time.time()
                _, n, _ = tdt._mot_cua_so((tap, (4, 5), 0, ten, h0, seed, moc[:so_moc],
                                           str(Path(tam) / f"{tap}_{so_moc}.pkl"), False, thay, goc in tdt.HINH_HOC))
                do.append(time.time() - s)
            moi_moc = max((do[1] - do[0]) / 2, 0.1)
            tg_mot[tap] = (max(do[0] - moi_moc, 0.0), moi_moc)
            r = pickle.load(open(Path(tam) / f"{tap}_3.pkl", "rb"))[0]
            print(f"4. {tap}: cố định ~{tg_mot[tap][0]:.0f} s / cửa sổ, ~{moi_moc:.1f} s / mốc; "
                  f"nhãn thầy: {r['nhan'] is not None}; chấm: {'ens' in r}")
    tt_ = max(1, min(6, (os.cpu_count() or 2) // 2))
    n_tr, n_va, n_te = (len(td.cac_moc(t)) for t in ("train", "val", "test"))
    (c_th, m_th), (c_te, m_te) = tg_mot["train_b"], tg_mot["c6_b"]
    giay = (c_th * 151 + m_th * (119 * n_tr + 32 * n_va)                    # train_b, train6_b, val_b, val6_b (có thầy)
            + c_te * 78 + m_te * 26 * n_te * (1 + 0.5 * 1.5 + 0.25 * 2.5))  # c6_b, yagi5_b ở 4–5 / 10 / 20 luồng
    gio = giay / tt_ / 3600
    print(f"Ước tính sinh dữ liệu thế giới B với {tt_} tiến trình: ~{gio:.1f} giờ (cộng huấn luyện + chấm: tổng ~{gio + 6:.0f} giờ)")
    print("OK — chạy: python -m scripts.chay_tn8")


if __name__ == "__main__":
    main()
