"""phan_tich_trong_so.py — Tổng hợp kết quả thu_trong_so.py: với mỗi α và mỗi giai đoạn, tuyến được chọn theo score S
có trễ thực bao nhiêu, đứt bao nhiêu.

    python -m src.ngau_nhien.phan_tich_trong_so --vao results/thu_trong_so/E2
"""

import argparse
import json
from pathlib import Path

import numpy as np

ALPHA = [round(x, 1) for x in np.arange(0, 1.01, 0.1)]


def chon(D, P, a, kieu):
    """Chỉ số ứng viên có S nhỏ nhất (hoà thì lấy ứng viên đứng trước = nhanh hơn)."""
    if kieu == "minmax":
        def chuan(x):
            lo, hi = np.nanmin(x, 1, keepdims=True), np.nanmax(x, 1, keepdims=True)
            r = hi - lo
            return np.where(r > 0, (x - lo) / np.where(r > 0, r, 1), 0.0)
        sp, rk = chuan(D), chuan(P)
    else:                                                   # tỉ lệ: chậm hơn tuyến nhanh nhất bao nhiêu, xác suất đứt
        sp, rk = D / np.nanmin(D, 1, keepdims=True) - 1, P
    S = np.where(np.isnan(D), np.inf, a * sp + (1 - a) * rk)
    return S.argmin(1)


def doc(vao):
    z = [np.load(f) for f in sorted(Path(vao, "tam").glob("val_*.npz"))]
    gop = {k: np.concatenate([x[k] for x in z]) for k in ("gd5", "gd15", "D_du", "P_du", "chi_phi", "dut_tl", "tre_song", "D_hk")}
    gop["cua_so"] = np.concatenate([np.full(len(x["t"]), i) for i, x in enumerate(z)])
    gop["loai_cs"] = np.array([str(x["loai"]) for x in z])
    return gop


def chi_so(g, m, idx):
    r = np.arange(len(idx))
    c, dl, ts = g["chi_phi"][m][r, idx], g["dut_tl"][m][r, idx], g["tre_song"][m][r, idx]
    return {"G_%": float((c.mean() / g["D_hk"][m].mean() - 1) * 100), "dut_snapshot_%": float(dl.mean() * 100),
            "truy_van_dut_%": float((dl > 0).mean() * 100), "so_truy_van_dut": int((dl > 0).sum()),
            "tre_song_ms": float(np.nanmean(ts))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vao", required=True)
    a = ap.parse_args()
    g = doc(a.vao)
    tranh = np.isfinite(g["D_hk"])                          # hậu kiểm còn tuyến sống suốt cửa sổ (như TN5)
    kq = {"so_truy_van": int(len(tranh)), "so_cua_so": int(len(g["loai_cs"])),
          "khong_tranh_duoc_%": {}, "theo_giai_doan": {}}
    for cn in ("gd15", "gd5"):
        for gd, ten in ((1, "bão"), (0, "nhẹ")):
            mm = g[cn] == gd
            kq["khong_tranh_duoc_%"][f"{cn}_{ten}"] = float((~tranh[mm]).mean() * 100) if mm.any() else None
    h = {k: v[tranh] for k, v in g.items() if k != "loai_cs"}
    so_uv = np.sum(~np.isnan(h["D_du"]), 1)
    kq["co_tu_2_ung_vien_%"] = float((so_uv >= 2).mean() * 100)
    kq["oracle_ung_vien"] = {"dut_nhanh_nhat_%": float(h["dut_tl"][:, 0].mean() * 100),
                             "dut_it_nhat_trong_ung_vien_%": float(np.nanmin(h["dut_tl"], 1).mean() * 100)}
    for cn in ("gd15", "gd5"):
        for gd, ten in ((1, "bão"), (0, "nhẹ")):
            m = h[cn] == gd
            if not m.any():
                continue
            o = {"so_truy_van": int(m.sum())}
            for kieu in ("minmax", "ti_le"):
                o[kieu] = {str(al): chi_so(h, m, chon(h["D_du"][m], h["P_du"][m], al, kieu)) for al in ALPHA}
            o["it_dut_nhat_trong_ung_vien"] = chi_so(h, m, np.nanargmin(np.where(np.isnan(h["dut_tl"][m]), np.inf,
                                                                              h["dut_tl"][m] + 1e-9 * np.arange(12)), 1))
            kq["theo_giai_doan"][f"{cn}_{ten}"] = o
    json.dump(kq, open(Path(a.vao) / "tong_hop.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"{kq['so_truy_van']} truy vấn, {kq['so_cua_so']} cửa sổ val; không tránh được đứt: "
          + ", ".join(f"{k} {v:.1f}%" for k, v in kq["khong_tranh_duoc_%"].items() if v is not None))
    print(f"có >= 2 ứng viên: {kq['co_tu_2_ung_vien_%']:.0f}%; oracle trong ứng viên: {kq['oracle_ung_vien']}")
    for k, o in kq["theo_giai_doan"].items():
        if not k.startswith("gd15"):
            continue
        print(f"\n== {k} ({o['so_truy_van']} truy vấn)   [chuẩn hoá min-max | tỉ lệ]")
        for al in ALPHA:
            x, y = o["minmax"][str(al)], o["ti_le"][str(al)]
            print(f"α={al:.1f}: G {x['G_%']:6.2f}%  đứt {x['dut_snapshot_%']:.3f}% ({x['so_truy_van_dut']:3d} truy vấn)  "
                  f"trễ sống {x['tre_song_ms']:6.1f} ms  |  G {y['G_%']:6.2f}%  đứt {y['dut_snapshot_%']:.3f}% "
                  f"({y['so_truy_van_dut']:3d})  trễ {y['tre_song_ms']:6.1f}")
        x = o["it_dut_nhat_trong_ung_vien"]
        print(f"ít đứt nhất trong ứng viên (biết trước): đứt {x['dut_snapshot_%']:.3f}% ({x['so_truy_van_dut']}), "
              f"trễ {x['tre_song_ms']:.1f}")


if __name__ == "__main__":
    main()
