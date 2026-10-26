"""
cong_cu_bo_sung_thoi_tiet.py — nối cột thời tiết cho các điểm MỚI của cấu hình vào tệp .npz có sẵn.

Vì sao cần: `thoi_tiet_vn_mr.npz` (2024-01-01 → 2025-12-31, 17 544 giờ) đã có sẵn 51
điểm; khi thêm nút/UAV mới vào `src/physics/sites.py` mà chạy lại `thoi_tiet tai --mo-rong`
thì tải lại TOÀN BỘ 70 điểm qua mạng (~1 giờ, dễ đứt giữa đường) và có nguy cơ ghi
đè bằng dữ liệu khác. Công cụ này chỉ tải các cột CÒN THIẾU, rồi ghép vào cuối —
51 cột cũ giữ nguyên từng bit nên mọi kết quả hồi quy vẫn so sánh được.

    python dataset/cong_cu_bo_sung_thoi_tiet.py
    python dataset/cong_cu_bo_sung_thoi_tiet.py --vao data/thoi_tiet_vn_mr.npz --ra data/thoi_tiet_vn_mr.npz

Ba bước:
    1) `tai_open_meteo` cho ĐÚNG các điểm thiếu (precip, code, gust) — nguồn ERA5;
    2) `them_giong` cho các cột mới (cape, lifted_index, code_fc + luật CAPE) — nguồn
       Historical Forecast API, cùng luật với 51 cột cũ để không lệch định nghĩa giông;
    3) kiểm `kiem_tra_du_diem` hai chiều trước khi ghi.
Nếu mạng lỗi ở bước 2, tệp vẫn được ghi nhưng KHÔNG có `cape`: đường chạy nào cần
giông sẽ báo lỗi rõ ràng (`MangGoc` đòi `cape`) chứ không lặng lẽ cho 0 giờ giông.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

GOC = Path(__file__).resolve().parents[1]
if str(GOC) not in sys.path:
    sys.path.insert(0, str(GOC))

# Console Windows mặc định cp1252 -> in chữ có dấu là UnicodeEncodeError.
for _s in (sys.stdout, sys.stderr):
    _r = getattr(_s, "reconfigure", None)
    if _r is not None:
        try:
            _r(encoding="utf-8")
        except (ValueError, OSError):
            pass

from src.ngau_nhien.thoi_tiet import (cac_diem, kiem_tra_du_diem,  # noqa: E402
                                       tai_open_meteo, them_giong)


def _ghep(a: dict, b: dict, ten_moi: list[str]) -> dict:
    """Nối các cột của `b` (chỉ chứa điểm mới) vào `a`, theo đúng thứ tự `ten_moi`."""
    idx = [list(b["site_names"]).index(n) for n in ten_moi]
    ra = {k: a[k] for k in a}
    ra["site_names"] = np.concatenate([a["site_names"], np.array(ten_moi)])
    for k in ("lat", "lon"):
        ra[k] = np.concatenate([a[k], b[k][idx]])
    for k in ("precip", "code", "gust"):
        ra[k] = np.concatenate([a[k], b[k][:, idx]], axis=1)
    return ra


def bo_sung(vao: str, ra: str | None = None, giu_moc_giong: bool = True):
    ra = ra or vao
    d = dict(np.load(vao, allow_pickle=False))
    co = [str(x) for x in d["site_names"]]
    can = cac_diem(mo_rong=True)
    thieu = [(n, la, lo) for n, la, lo in can if n not in set(co)]
    if not thieu:
        print(f"{vao} đã đủ {len(co)} điểm, không cần bổ sung.")
        return d
    # Điểm thừa (đã xoá khỏi cấu hình) thì bị loại luôn — cột chết che lỗi thiếu điểm.
    thua = [n for n in co if n not in {x[0] for x in can}]
    if thua:
        print(f"Loại {len(thua)} cột không còn trong cấu hình: {thua}")
        giu = [i for i, n in enumerate(co) if n not in set(thua)]
        d = {**d, "site_names": d["site_names"][giu], "lat": d["lat"][giu], "lon": d["lon"][giu],
             **{k: d[k][:, giu] for k in ("precip", "code", "gust", "cape", "lifted_index", "code_fc")
                if k in d and d[k].ndim == 2}}
        co = [co[i] for i in giu]

    t = [str(x) for x in d["time_utc"]]
    start, end = t[0][:10], t[-1][:10]
    tmp = GOC / "data" / "_moi.npz"
    ten_moi = [n for n, _, _ in thieu]
    print(f"Tải {len(ten_moi)} điểm mới ({start} → {end}): {ten_moi}")
    # Tai theo cùng khoảng ngày và đúng danh sách điểm mới: hàm dùng `cac_diem`, nên
    # truyền tạm danh sách qua biến môi trường là không cần — gọi thẳng `_goi`.
    from src.ngau_nhien.thoi_tiet import _goi
    lat = [la for _, la, _ in thieu]
    lon = [lo for _, _, lo in thieu]
    times, c = _goi("https://archive-api.open-meteo.com/v1/archive", lat, lon, start, end,
                    ["precipitation", "weather_code", "wind_gusts_10m"], 180, {"wind_speed_unit": "ms"})
    P, C, G = (np.nan_to_num(c[v], nan=0.0) for v in ("precipitation", "weather_code", "wind_gusts_10m"))
    np.savez_compressed(tmp, site_names=np.array(ten_moi), lat=np.array(lat), lon=np.array(lon),
                        time_utc=np.array(times), precip=P.astype(np.float32),
                        code=C.astype(np.int16), gust=G.astype(np.float32), nguon=np.array("open-meteo-era5"))

    if giu_moc_giong:
        try:
            them_giong(str(tmp), str(tmp), kiem_tra=False)
        except Exception as e:                                          # noqa: BLE001
            print(f"CẢNH BÁO: không thêm được giông cho cột mới ({e}). "
                  f"Nối tiếp tục, nhưng cần chạy lại bước này trước khi mô phỏng.")

    moi = dict(np.load(tmp, allow_pickle=False))
    tmp.unlink(missing_ok=True)
    ra_d = _ghep(d, moi, ten_moi)
    kiem_tra_du_diem_dict(ra_d)
    np.savez_compressed(ra, **ra_d)
    print(f"Đã lưu {ra}: {len(co)} + {len(ten_moi)} = {len(ra_d['site_names'])} điểm, "
          f"{ra_d['precip'].shape[0]} giờ. Có cape: {'cape' in ra_d}")
    return ra_d


def bo_sung_giong(vao: str, ra: str | None = None):
    """Chỉ chạy bước `them_giong` cho các cột MỚI (đã nối nhưng thiếu cape/code_fc).

    Tách riêng để không phải tải lại precip/gust qua mạng khi bước giông đứt.
    Cột cũ giữ nguyên; cột mới được gán code theo CÙNG luật CAPE như cột cũ.
    """
    ra = ra or vao
    d = dict(np.load(vao, allow_pickle=False))
    ten = [str(x) for x in d["site_names"]]
    if "cape" not in d:
        raise SystemExit(f"{vao} chưa có cột 'cape' — chạy lại toàn bộ bổ sung.")
    can = {n for n, _, _ in cac_diem(mo_rong=True)}
    # Chỉ bổ sung cho cột chưa có CAPE (cape = 0 suốt kỳ hoặc cột chưa tồn tại):
    # cột cũ đã có giông thì bỏ qua, nếu không sẽ gọi 70 điểm × 2 năm và ghi đè
    # định nghĩa giông của cột cũ.
    def _co_cape(i):
        return "cape" in d and i < d["cape"].shape[1] and bool(np.any(d["cape"][:, i] > 0))
    moi = [i for i, n in enumerate(ten) if n in can and not _co_cape(i)]
    if not moi:
        print("Không có cột mới nào.")
        return d
    tmp = GOC / "data" / "_moi.npz"
    t = [str(x) for x in d["time_utc"]]
    np.savez_compressed(tmp, site_names=d["site_names"], lat=d["lat"], lon=d["lon"],
                        time_utc=d["time_utc"], precip=d["precip"], code=d["code"],
                        gust=d["gust"], nguon=d.get("nguon", np.array("open-meteo-era5")))
    print(f"Chạy them-giong cho {len(moi)} cột thiếu ...")
    them_giong(str(tmp), str(tmp), kiem_tra=False)
    m = dict(np.load(tmp, allow_pickle=False))
    tmp.unlink(missing_ok=True)
    ra_d = {k: d[k] for k in d}
    S = len(ten)
    for k in ("cape", "lifted_index", "code", "code_fc", "code_era5"):
        if k not in m:
            continue
        cu = ra_d.get(k)
        if cu is None or cu.ndim != 2:
            ra_d[k] = np.zeros((len(t), S), dtype=m[k].dtype)   # cột chưa từng có
        elif cu.shape[1] < S:                                   # cột cũ 51 -> mở rộng đủ
            ra_d[k] = np.concatenate([cu, np.zeros((len(t), S - cu.shape[1]), dtype=cu.dtype)], axis=1)
        else:
            ra_d[k] = cu.copy()
        ra_d[k][:, moi] = m[k][:, moi]
    if isinstance(m["nguon"], np.ndarray) and m["nguon"].ndim == 0:
        ra_d["nguon"] = m["nguon"]
    np.savez_compressed(ra, **ra_d)
    print(f"Đã lưu {ra}: cape/code_fc đã có cho {len(moi)} cột mới; tổng {S} điểm.")
    return ra_d


def kiem_tra_du_diem_dict(d: dict):
    """`kiem_tra_du_diem` chỉ đọc tệp; ở đây kiểm trực tiếp trên dict trước khi ghi."""
    from src.ngau_nhien.thoi_tiet import cac_diem as _can
    can = [n for n, _, _ in _can(mo_rong=True)]
    co = [str(x) for x in d["site_names"]]
    thieu = [n for n in can if n not in set(co)]
    thua = [n for n in co if n not in set(can)]
    if thieu or thua:
        raise SystemExit(f"Tập điểm vẫn lệch: thiếu {thieu}, thừa {thua}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Bổ sung cột thời tiết cho điểm mới của cấu hình")
    p.add_argument("--vao", default="data/thoi_tiet_vn_mr.npz")
    p.add_argument("--ra", default=None, help="mặc định ghi đè chính tệp --vao")
    p.add_argument("--chi-giong", action="store_true",
                   help="chỉ chạy bước giông cho các cột mới (cột đã nối nhưng cape = 0)")
    p.add_argument("--khong-giong", action="store_true", help="bỏ bước them-giong (chỉ để thử mạng)")
    a = p.parse_args()
    if a.chi_giong:
        bo_sung_giong(a.vao, a.ra)
    else:
        bo_sung(a.vao, a.ra, giu_moc_giong=not a.khong_giong)
