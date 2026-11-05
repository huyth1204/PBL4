"""
tai_cape_era5.py — tải CAPE ERA5 (NCAR RDA / GDEX, bộ d633000, 0,25°) cho 74 điểm, ĐÚNG các ô lưới đã dùng ở tệp 2017–2021,
để hiệu chỉnh ngưỡng giông (TN9, results/tn9/TIEU_CHI_9.md).

    pip install xarray netCDF4
    python dataset/tai_cape_era5.py --start 2022-01 --end 2025-12 --o-luoi data/thoi_tiet_vn_2017_2021_1.npz

  --o-luoi : tệp 2017–2021 nhóm đã làm (lấy cape_grid_lat / cape_grid_lon để chắc chắn cùng ô); nếu không có thì lấy ô 0,25°
             gần nhất của toạ độ tệp mẫu data/thoi_tiet_vn_mr.npz (cùng cách).
Ra data/cape_era5_2022_2025.npz (time_utc, cape [giờ, 74 điểm], cape_grid_lat, cape_grid_lon, site_names, nguon), khoảng 10 MB.
Đọc qua OPeNDAP từng tệp tháng `e5.oper.an.sfc.128_059_cape.ll025sc.<YYYYMM>0100_<YYYYMM><ngày cuối>23.nc`, chỉ lấy hộp lưới
quanh Việt Nam (khoảng 6 MB / tháng). Mỗi tháng lưu riêng (*.tam_YYYYMM.npz): đứt mạng chạy lại đúng lệnh là tải tiếp.
Nếu nhóm đã có cách tải khác cho 2017–2021 thì dùng lại cách đó cho 2022–2025 cũng được, miễn ra cùng các trường trên.
"""
from __future__ import annotations

import argparse
import calendar
import sys
import time
from pathlib import Path

import numpy as np

GOC = Path(__file__).resolve().parents[1]
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError, OSError):
        pass

GOC_URL = ("https://tds.gdex.ucar.edu/thredds/dodsC/files/g/d633000/e5.oper.an.sfc",
           "https://thredds.rda.ucar.edu/thredds/dodsC/files/g/d633000/e5.oper.an.sfc")


def ten_tep(y, m):
    n = calendar.monthrange(y, m)[1]
    return f"{y}{m:02d}/e5.oper.an.sfc.128_059_cape.ll025sc.{y}{m:02d}0100_{y}{m:02d}{n:02d}23.nc"


def mo_thang(y, m, mo=None):
    """Mở tệp tháng qua OPeNDAP (thử lần lượt các máy chủ); mo: hàm mở thay thế để kiểm ngoại tuyến."""
    import xarray as xr
    mo = mo or (lambda u: xr.open_dataset(u))
    loi = None
    for g in GOC_URL:
        for lan in range(4):
            try:
                return mo(f"{g}/{ten_tep(y, m)}")
            except Exception as e:                       # mạng / máy chủ bận
                loi = e
                time.sleep(15 * (lan + 1))
    raise SystemExit(f"Không mở được tháng {y}-{m:02d}: {loi}")


def tai_thang(y, m, la_o, lo_o, mo=None, ngu=time.sleep, so_lan=6):
    """Mở + đọc một tháng, thử lại khi mạng chập chờn (lỗi 'curl error: Timeout' thường xảy ra lúc ĐỌC, không phải lúc mở)."""
    import xarray as xr
    mo = mo or (lambda u: xr.open_dataset(u))
    loi = None
    for lan in range(so_lan):
        for g in GOC_URL:
            ds = None
            try:
                ds = mo(f"{g}/{ten_tep(y, m)}")
                return lay_thang(ds, la_o, lo_o)
            except SystemExit:
                raise
            except Exception as e:
                loi = e
                print(f"  {y}-{m:02d}: lỗi {type(e).__name__}: {str(e)[:120]} — thử lại", flush=True)
            finally:
                if ds is not None:
                    try:
                        ds.close()
                    except Exception:
                        pass
        ngu(30 * (lan + 1))
    raise SystemExit(f"Tháng {y}-{m:02d} lỗi {so_lan} lượt ({loi}). Chạy lại đúng lệnh: các tháng đã xong được giữ.")


def lay_thang(ds, la_o, lo_o):
    """CAPE [giờ, điểm] tại các ô (la_o, lo_o); chỉ đọc hộp nhỏ bao các ô."""
    ten = "CAPE" if "CAPE" in ds else [v for v in ds.data_vars][0]
    la, lo = ds["latitude"].values, ds["longitude"].values
    lo_o = np.mod(lo_o, 360)
    il = np.array([int(np.argmin(np.abs(la - x))) for x in la_o])
    jl = np.array([int(np.argmin(np.abs(lo - x))) for x in lo_o])
    if np.max(np.abs(la[il] - la_o)) > 1e-6 or np.max(np.abs(lo[jl] - lo_o)) > 1e-6:
        raise SystemExit("Ô lưới không khớp lưới 0,25° của tệp ERA5")
    i0, i1, j0, j1 = il.min(), il.max() + 1, jl.min(), jl.max() + 1
    hop = ds[ten].isel(latitude=slice(i0, i1), longitude=slice(j0, j1)).values      # [giờ, vĩ, kinh] — chỉ tải hộp
    t = [str(x)[:16] for x in ds["time"].values.astype("datetime64[m]")]
    return t, hop[:, il - i0, jl - j0].astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", required=True, help="YYYY-MM")
    ap.add_argument("--end", required=True, help="YYYY-MM")
    ap.add_argument("--o-luoi", default=None)
    ap.add_argument("--mau", default=str(GOC / "data/thoi_tiet_vn_mr.npz"))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    mau = np.load(a.mau, allow_pickle=False)
    ten = [str(x) for x in mau["site_names"]]
    if a.o_luoi:
        z = np.load(a.o_luoi, allow_pickle=False)
        if [str(x) for x in z["site_names"]] != ten:
            raise SystemExit("Thứ tự điểm của --o-luoi khác tệp mẫu")
        la_o, lo_o = z["cape_grid_lat"].astype(float), z["cape_grid_lon"].astype(float)
    else:
        la_o, lo_o = np.round(mau["lat"] * 4) / 4, np.round(mau["lon"] * 4) / 4
    y0, m0 = map(int, a.start.split("-"))
    y1, m1 = map(int, a.end.split("-"))
    out = Path(a.out or GOC / f"data/cape_era5_{y0}_{y1}.npz")
    T, C = [], []
    y, m = y0, m0
    while (y, m) <= (y1, m1):
        tam = out.with_name(f"{out.stem}.tam_{y}{m:02d}.npz")
        if tam.exists():
            z = np.load(tam, allow_pickle=False)
            t, c = [str(x) for x in z["time"]], z["cape"]
        else:
            t, c = tai_thang(y, m, la_o, lo_o)
            np.savez_compressed(tam, time=np.array(t), cape=c)
        n = calendar.monthrange(y, m)[1]
        if len(t) != 24 * n or np.isnan(c).any():
            raise SystemExit(f"Tháng {y}-{m:02d}: {len(t)} giờ (cần {24 * n}), NaN {int(np.isnan(c).sum())}")
        print(f"  {y}-{m:02d}: {len(t)} giờ, CAPE trung vị {np.median(c):.0f}", flush=True)
        T += t
        C.append(c)
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    np.savez_compressed(out, time_utc=np.array(T), cape=np.concatenate(C), cape_grid_lat=la_o, cape_grid_lon=lo_o,
                        site_names=np.array(ten),
                        nguon=np.array("NCAR d633000 ERA5 CAPE (128_059), ô 0,25° như tệp 2017–2021, qua OPeNDAP"))
    for f in out.parent.glob(f"{out.stem}.tam_*.npz"):
        f.unlink()
    print(f"Đã lưu {out}: {len(T)} giờ × {len(ten)} điểm. Gửi tệp này cho TV2.")


if __name__ == "__main__":
    main()