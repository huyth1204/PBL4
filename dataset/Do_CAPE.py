"""
do_cape.py — dò xem nguồn Open-Meteo nào có CAPE cho các năm trước 2022 (mỗi nguồn 1 yêu cầu nhỏ: 1 điểm, 1 ngày).

    python dataset/do_cape.py

In bảng: nguồn, số giờ có CAPE / 24 ở 4 ngày mẫu (2017, 2019, 2020, 2021). Gửi bảng này cho TV2.
"""
import sys
import time

import requests

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError, OSError):
        pass

LAT, LON = 16.05, 108.20                      # Đà Nẵng
NGAY = ("2017-09-15", "2019-09-02", "2020-10-28", "2021-09-23")
DU_BAO = "https://historical-forecast-api.open-meteo.com/v1/forecast"
LUU_TRU = "https://archive-api.open-meteo.com/v1/archive"
NGUON = [(DU_BAO, m) for m in (None, "ecmwf_ifs", "ecmwf_ifs025", "gfs_seamless", "gfs_global", "jma_seamless", "jma_gsm",
                               "cma_grapes_global", "bom_access_global", "icon_seamless", "meteofrance_seamless",
                               "ukmo_seamless", "kma_seamless", "gem_seamless")] + \
        [(LUU_TRU, m) for m in (None, "era5", "era5_seamless", "ecmwf_ifs", "ecmwf_ifs_analysis_long_window", "era5_land")]

print(f"{'nguồn':60s} " + "  ".join(NGAY))
for url, m in NGUON:
    kq = []
    for d in NGAY:
        p = {"latitude": LAT, "longitude": LON, "start_date": d, "end_date": d, "hourly": "cape", "timezone": "GMT"}
        if m:
            p["models"] = m
        for _ in range(5):
            r = requests.get(url, params=p, timeout=60)
            if r.status_code != 429:
                break
            time.sleep(65)
        if r.status_code != 200:
            kq.append(f"lỗi {r.status_code}")
        else:
            v = r.json().get("hourly", {}).get("cape", [])
            kq.append(f"{sum(x is not None for x in v):2d}/24")
        time.sleep(1)
    ten = ("dự báo lưu trữ" if url == DU_BAO else "lưu trữ ERA5") + f" / {m or 'mặc định'}"
    print(f"{ten:60s} " + "  ".join(f"{k:>10s}" for k in kq), flush=True)