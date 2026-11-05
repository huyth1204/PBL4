"""
tai_them_nam.py — tải thời tiết 74 điểm cho THÊM NĂM (vd 2022–2023) cùng định dạng với data/thoi_tiet_vn_mr.npz,
để cắt thêm cơn bão (chia 10 / 5 / 5 cơn).

Chạy trên máy có mạng (khoảng 1 giờ cho 2 năm; phiên Claude không tải được):

    python dataset/tai_them_nam.py --start 2022-01-01 --end 2023-12-31
    python dataset/tai_them_nam.py --start 2022-01-01 --end 2023-12-31 --sua   # chỉ sửa toạ độ tệp đã tải (vài phút)

Ra data/thoi_tiet_vn_2022_2023.npz (khoảng 7 MB). Gửi lại bằng cách đẩy lên một nhánh GitHub riêng hoặc đính kèm vào chat.

Ba bước, đúng như tệp 2024–2025 đã làm:
    1) ERA5 qua Open-Meteo archive: mưa, mã thời tiết, gió giật (`thoi_tiet.tai_open_meteo`, --mo-rong);
    2) giông: CAPE, lifted index, mã dự báo qua Historical Forecast API, luật CAPE (`thoi_tiet.them_giong`);
    3) sắp lại cột điểm ĐÚNG thứ tự của tệp mẫu (tệp 2024–2025 có 4 điểm nối thêm ở cuối nên thứ tự khác cấu hình);
    4) điểm nào có toạ độ trong cấu hình (sites.py) khác toạ độ đã dùng khi tải tệp mẫu (17 tâm tuần tra UAV đã dời sau khi
       tải 2024–2025, lệch tới 0,6°) thì tải lại riêng các cột đó TẠI TOẠ ĐỘ CỦA TỆP MẪU, để mọi năm lấy thời tiết ở cùng chỗ;
    rồi kiểm: cùng tên, cùng toạ độ, đủ giờ, giờ thiếu CAPE.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

GOC = Path(__file__).resolve().parents[1]
if str(GOC) not in sys.path:
    sys.path.insert(0, str(GOC))
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError, OSError):
        pass


def sap_theo_mau(d: dict, mau: dict) -> dict:
    """Sắp các cột điểm của d theo thứ tự site_names của tệp mẫu (mọi trường [giờ, điểm] và lat / lon)."""
    ten = [str(x) for x in d["site_names"]]
    ten_mau = [str(x) for x in mau["site_names"]]
    if sorted(ten) != sorted(ten_mau):
        raise SystemExit(f"Tập điểm khác tệp mẫu: thiếu {set(ten_mau) - set(ten)}, thừa {set(ten) - set(ten_mau)}")
    idx = np.array([ten.index(n) for n in ten_mau])
    S, H = len(ten), len(d["time_utc"])
    ra = {}
    for k, v in d.items():
        if k == "site_names":
            ra[k] = np.array(ten_mau)
        elif k in ("lat", "lon"):
            ra[k] = v[idx]
        elif getattr(v, "ndim", 0) == 2 and v.shape == (H, S):
            ra[k] = v[:, idx]
        else:
            ra[k] = v
    return ra


def sua_toa_do(d: dict, mau: dict, goi=None) -> list:
    """Tải lại (ERA5 + giông) các cột có toạ độ khác tệp mẫu, tại toạ độ tệp mẫu; đặt lại giông theo luật CAPE cho cả tệp.
    goi: hàm gọi Open-Meteo (mặc định thoi_tiet._goi; tham số để kiểm ngoại tuyến). Trả danh sách điểm đã sửa."""
    from src.ngau_nhien import thoi_tiet as T
    goi = goi or T._goi
    lech = np.flatnonzero(np.maximum(np.abs(d["lat"] - mau["lat"]), np.abs(d["lon"] - mau["lon"])) > 1e-4)
    if not len(lech):
        return []
    la, lo = mau["lat"][lech], mau["lon"][lech]
    t = [str(x) for x in d["time_utc"]]
    a, b = t[0][:10], t[-1][:10]
    print(f"Tải lại {len(lech)} điểm tại toạ độ tệp mẫu: " + ", ".join(str(mau["site_names"][i]) for i in lech))
    tg, c = goi("https://archive-api.open-meteo.com/v1/archive", la, lo, a, b,
                ["precipitation", "weather_code", "wind_gusts_10m"], 180, {"wind_speed_unit": "ms"})
    if [str(x) for x in tg] != t:
        raise SystemExit("Giờ ERA5 tải lại không khớp tệp")
    d["precip"][:, lech] = np.nan_to_num(c["precipitation"], nan=0.0).astype(np.float32)
    d["gust"][:, lech] = np.nan_to_num(c["wind_gusts_10m"], nan=0.0).astype(np.float32)
    d["code_era5"][:, lech] = np.nan_to_num(c["weather_code"], nan=0.0).astype(np.int16)
    tf, c = goi("https://historical-forecast-api.open-meteo.com/v1/forecast", la, lo, a, b,
                ["weather_code", "cape", "lifted_index"], 180)
    vt = {x: i for i, x in enumerate(tf)}
    idx = np.array([vt.get(x, -1) for x in t])
    ok = idx >= 0
    for k, v, kieu in (("code_fc", "weather_code", np.int16), ("cape", "cape", np.float32), ("lifted_index", "lifted_index", np.float32)):
        m = np.zeros((len(t), len(lech)))
        m[ok] = np.nan_to_num(c[v][idx[ok]])
        d[k][:, lech] = m.astype(kieu)
    d["lat"], d["lon"] = d["lat"].copy(), d["lon"].copy()
    d["lat"][lech], d["lon"][lech] = la, lo
    T.ap_luat_giong(d, "cape")                         # giông = CAPE ≥ ngưỡng và mưa ≥ ngưỡng, tính lại từ code_era5
    return [str(mau["site_names"][i]) for i in lech]


def kiem(d: dict, mau: dict, start: str, end: str):
    loi = []
    if [str(x) for x in d["site_names"]] != [str(x) for x in mau["site_names"]]:
        loi.append("thứ tự điểm khác tệp mẫu")
    for k in ("lat", "lon"):
        if np.max(np.abs(d[k] - mau[k])) > 1e-4:
            loi.append(f"{k} lệch tệp mẫu")
    for k in mau:
        if getattr(mau[k], "ndim", 0) == 2 and k not in d:
            loi.append(f"thiếu trường {k}")
    gio = int((datetime.fromisoformat(end) - datetime.fromisoformat(start)).days + 1) * 24
    if len(d["time_utc"]) != gio:
        loi.append(f"số giờ {len(d['time_utc'])} khác {gio}")
    t = np.array(d["time_utc"], dtype="datetime64[h]")
    if np.any(np.diff(t) != np.timedelta64(1, "h")):
        loi.append("giờ không liên tục")
    thieu_cape = float(np.mean((d["cape"] == 0).all(1)) * 100)          # giờ mà mọi điểm CAPE = 0 ≈ API không có dữ liệu
    if thieu_cape > 2:
        loi.append(f"{thieu_cape:.1f}% số giờ không có CAPE ở mọi điểm (Historical Forecast API thiếu dữ liệu?)")
    return loi, thieu_cape


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--mau", default=str(GOC / "data/thoi_tiet_vn_mr.npz"), help="tệp mẫu để lấy thứ tự điểm")
    ap.add_argument("--sua", action="store_true", help="tệp ra đã có (tải xong, kiểm lệch toạ độ): chỉ tải lại cột lệch")
    a = ap.parse_args()
    out = Path(a.out or GOC / f"data/thoi_tiet_vn_{a.start[:4]}_{a.end[:4]}.npz")
    mau = dict(np.load(a.mau, allow_pickle=False))
    tam = out.with_suffix(".tam.npz")
    if a.sua:
        if not out.exists():
            raise SystemExit(f"--sua cần tệp {out} đã tải")
        d = sap_theo_mau(dict(np.load(out, allow_pickle=False)), mau)
    else:
        if out.exists():
            raise SystemExit(f"{out} đã có; xoá, dùng --out khác, hoặc thêm --sua nếu chỉ lệch toạ độ")
        from src.ngau_nhien.thoi_tiet import tai_open_meteo, them_giong
        if not tam.exists():
            tai_open_meteo(a.start, a.end, str(tam), mo_rong=True)
        them_giong(str(tam), str(tam), luat="cape")
        d = sap_theo_mau(dict(np.load(tam, allow_pickle=False)), mau)
    da_sua = sua_toa_do(d, mau)
    if da_sua:
        d["nguon"] = np.array(f"{str(d['nguon'])}; {len(da_sua)} điểm tải lại tại toạ độ tệp mẫu")
    loi, thieu = kiem(d, mau, a.start, a.end)
    np.savez_compressed(out, **d)
    h = hashlib.sha256(out.read_bytes()).hexdigest()
    print(f"\nĐã lưu {out}: {len(d['time_utc'])} giờ × {len(d['site_names'])} điểm, sha256 {h[:16]}…, "
          f"giờ thiếu CAPE {thieu:.2f}%")
    if loi:
        print("CẢNH BÁO — kiểm không đạt:\n  - " + "\n  - ".join(loi))
        sys.exit(1)
    if tam.exists():
        tam.unlink()
    print("Kiểm đạt. Gửi tệp này cho TV2 (nhánh GitHub riêng hoặc đính kèm).")


if __name__ == "__main__":
    main()