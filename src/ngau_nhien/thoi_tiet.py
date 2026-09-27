"""
thoi_tiet.py — Dữ liệu thời tiết theo giờ cho các điểm mặt đất / trên không của SAGSIN.

Nguồn thật: Open-Meteo Historical Weather API (ERA5, miễn phí, không cần khoá).
    python -m src.ngau_nhien.thoi_tiet tai --start 2023-01-01 --end 2024-12-31 --out data/thoi_tiet_vn.npz
ERA5 không suy ra được giông (mã thời tiết chỉ tính từ mây và mưa), nên thêm bước lấy giông từ
Historical Forecast API (mô hình dự báo, có CAPE và mã giông 95/97, dữ liệu từ khoảng 2022):
    python -m src.ngau_nhien.thoi_tiet them-giong --vao data/thoi_tiet_vn.npz
Bản giả lập cùng định dạng (để chạy thử khi chưa tải được):
    python -m src.ngau_nhien.thoi_tiet gia-lap --out data/thoi_tiet_gia_lap.npz

Định dạng .npz:
    site_names [S], lat [S], lon [S]
    time_utc   [H]  chuỗi ISO, cách nhau 1 giờ
    precip     [H,S] mm trong giờ trước đó (xem như cường độ mưa trung bình, mm/h)
    code       [H,S] mã thời tiết WMO (95, 96, 99 = giông)
    gust       [H,S] gió giật 10 m, m/s
    nguon      "open-meteo-era5" hoặc "gia-lap"
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

GIONG = (95, 96, 97, 99)          # mã WMO giông (Open-Meteo dùng 95, 97; 96, 99 khi mô hình có dự báo mưa đá)
CAPE_GIONG, MUA_GIONG = 1000.0, 1.0   # luật dự phòng: CAPE ≥ 1000 J/kg và mưa ≥ 1 mm/h

# Bản sao cấu hình của TV1 (src/physics/air_ground.py) để script chạy được cả khi không nằm trong repo
_DIEM_TV1 = [
    ("Hanoi", 21.0285, 105.8542), ("DaNang", 16.0544, 108.2022), ("HoChiMinh", 10.7626, 106.6602),
    ("HaiPhong", 20.8449, 106.6881), ("CanTho", 10.0452, 105.7469),
    ("Vinh", 18.6796, 105.6813), ("Hue", 16.4637, 107.5909), ("NhaTrang", 12.2388, 109.1967),
    ("QuyNhon", 13.7830, 109.2199), ("BuonMaThuot", 12.6667, 108.0500), ("CaMau", 9.1769, 105.1500),
    ("LaoCai", 22.4833, 103.9750), ("PleiKu", 13.9833, 108.0000),
    ("HAPS-North", 21.0, 105.9), ("HAPS-Central", 16.3, 108.0), ("HAPS-South", 10.9, 106.7),
    ("HAPS-Highland", 13.9, 108.0), ("HAPS-Mekong", 10.0, 105.7),
    ("UAV-1", 18.6796, 105.6813), ("UAV-2", 12.2388, 109.1967), ("UAV-3", 10.0452, 105.7469),
    ("UAV-4", 22.4833, 103.9750), ("UAV-5", 13.7830, 109.2199),
]


def cac_diem():
    """Tên và toạ độ các nút không phải vệ tinh theo cấu hình của TV1 (UAV lấy tâm vòng tuần tra)."""
    try:
        from src.physics.air_ground import DEFAULT_GATEWAYS, DEFAULT_HAPS, DEFAULT_UAV_CENTERS, DEFAULT_USERS
        return [(n, la, lo) for n, la, lo in DEFAULT_GATEWAYS + DEFAULT_USERS + DEFAULT_HAPS + DEFAULT_UAV_CENTERS]
    except ImportError:
        return list(_DIEM_TV1)


def _goi(url, lat, lon, start, end, hourly, chunk_days=180, extra=None):
    """Gọi Open-Meteo theo từng khúc ngày cho nhiều điểm; trả (times, {biến: mảng [H,S]})."""
    import requests
    t0, t1 = datetime.fromisoformat(start), datetime.fromisoformat(end)
    times, cols = [], {v: [] for v in hourly}
    a = t0
    while a <= t1:
        b = min(a + timedelta(days=chunk_days - 1), t1)
        params = {"latitude": ",".join(f"{x:.4f}" for x in lat), "longitude": ",".join(f"{x:.4f}" for x in lon),
                  "start_date": a.date().isoformat(), "end_date": b.date().isoformat(),
                  "hourly": ",".join(hourly), "timezone": "GMT", **(extra or {})}
        for lan in range(6):
            r = requests.get(url, params=params, timeout=180)
            if r.status_code == 200:
                break
            print(f"  lỗi {r.status_code}: {r.text[:200]} — thử lại sau {20 * (lan + 1)} s")
            time.sleep(20 * (lan + 1))
        r.raise_for_status()
        js = r.json()
        js = js if isinstance(js, list) else [js]
        times += js[0]["hourly"]["time"]
        for v in hourly:
            cols[v].append(np.array([j["hourly"].get(v, [None] * len(j["hourly"]["time"])) for j in js], float).T)
        print(f"  {a.date()} → {b.date()}: {len(js[0]['hourly']['time'])} giờ × {len(js)} điểm")
        a = b + timedelta(days=1)
        time.sleep(1.0)
    return times, {v: np.concatenate(x) for v, x in cols.items()}


def tai_open_meteo(start: str, end: str, out: str, chunk_days: int = 180):
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    diem = cac_diem()
    names = [d[0] for d in diem]
    lat = np.array([d[1] for d in diem])
    lon = np.array([d[2] for d in diem])
    times, c = _goi("https://archive-api.open-meteo.com/v1/archive", lat, lon, start, end,
                    ["precipitation", "weather_code", "wind_gusts_10m"], chunk_days, {"wind_speed_unit": "ms"})
    P, C, G = (np.nan_to_num(c[v], nan=0.0) for v in ("precipitation", "weather_code", "wind_gusts_10m"))
    np.savez_compressed(out, site_names=np.array(names), lat=lat, lon=lon, time_utc=np.array(times),
                        precip=P.astype(np.float32), code=C.astype(np.int16), gust=G.astype(np.float32),
                        nguon=np.array("open-meteo-era5"))
    print(f"Đã lưu {out}: {P.shape[0]} giờ × {P.shape[1]} điểm; giờ có mưa {np.mean(P > 0.1) * 100:.1f}%, "
          f"giờ có giông {np.mean(np.isin(C, GIONG)) * 100:.2f}% (ERA5 không có giông; chạy tiếp them-giong)")


def ap_luat_giong(d: dict, luat: str = "cape"):
    """Đặt lại cột code theo luật giông. cape: CAPE ≥ 1000 J/kg và mưa ≥ 1 mm/h (mặc định);
    ma: mã 95/97 của mô hình dự báo (không ổn định theo thời gian vì mô hình 'best_match' thay đổi)."""
    code0 = d.get("code_era5", d["code"])
    g_cape = (d["cape"] >= CAPE_GIONG) & (d["precip"] >= MUA_GIONG)
    g_ma = np.isin(d.get("code_fc", np.zeros_like(code0)), GIONG)
    g = {"cape": g_cape, "ma": g_ma, "ca_hai": g_cape | g_ma}[luat]
    d.update(code_era5=code0, code=np.where(g, 95, code0).astype(np.int16),
             nguon=np.array(f"open-meteo-era5 + giông ({luat})"))
    return g_cape, g_ma


def them_giong(vao: str, ra: str | None = None, chunk_days: int = 180, luat: str = "cape"):
    """Bổ sung CAPE, chỉ số nâng và mã thời tiết của mô hình dự báo lưu trữ vào tệp ERA5, rồi đặt giông theo luật.
    Giữ mã ERA5 gốc ở code_era5, mã dự báo ở code_fc."""
    ra = ra or vao
    d = dict(np.load(vao, allow_pickle=False))
    t = [str(x) for x in d["time_utc"]]
    times, c = _goi("https://historical-forecast-api.open-meteo.com/v1/forecast", d["lat"], d["lon"],
                    t[0][:10], t[-1][:10], ["weather_code", "cape", "lifted_index"], chunk_days)
    vi_tri = {x: i for i, x in enumerate(times)}
    idx = np.array([vi_tri.get(x, -1) for x in t])
    thieu = float(np.mean(idx < 0) * 100)
    H, S = d["precip"].shape
    fc, cape, li = (np.full((H, S), np.nan) for _ in range(3))
    ok = idx >= 0
    fc[ok], cape[ok], li[ok] = c["weather_code"][idx[ok]], c["cape"][idx[ok]], c["lifted_index"][idx[ok]]
    d.update(code_fc=np.nan_to_num(fc).astype(np.int16), cape=np.nan_to_num(cape).astype(np.float32),
             lifted_index=np.nan_to_num(li).astype(np.float32))
    g_cape, g_ma = ap_luat_giong(d, luat)
    np.savez_compressed(ra, **d)
    print(f"Đã lưu {ra} (luật giông: {luat}). Thiếu dữ liệu dự báo ở {thieu:.1f}% số giờ.")
    thong_ke_giong(d)


def thong_ke_giong(d: dict):
    t = [str(x) for x in d["time_utc"]]
    nam, thang = np.array([int(x[:4]) for x in t]), np.array([int(x[5:7]) for x in t])
    g_cape = (d["cape"] >= CAPE_GIONG) & (d["precip"] >= MUA_GIONG)
    g_ma = np.isin(d["code_fc"], GIONG) if "code_fc" in d else None
    for ten, g in (("luật CAPE", g_cape), ("mã dự báo", g_ma)):
        if g is None:
            continue
        print(f"  {ten}: {g.mean() * 100:.2f}% số giờ")
        for y in np.unique(nam):
            print(f"    {y}: " + "  ".join(f"T{m}:{g[(nam == y) & (thang == m)].mean() * 100:.1f}"
                                        for m in range(1, 13) if ((nam == y) & (thang == m)).any()))


def gia_lap(out: str, days: int = 184, seed: int = 7, start: str = "2024-05-01"):
    """Thời tiết giả lập kiểu mùa mưa Việt Nam: mưa theo vùng (Markov), đối lưu chiều, giông khi mưa to.

    Chỉ để chạy thử quy trình. Các tham số là giả định, không phải số đo.
    """
    rng = np.random.default_rng(seed)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    diem = cac_diem()
    names = [d[0] for d in diem]
    lat = np.array([d[1] for d in diem])
    lon = np.array([d[2] for d in diem])
    vung = np.where(lat > 19, 0, np.where(lat > 12, 1, 2))          # Bắc, Trung, Nam
    H, S = days * 24, len(diem)
    t0 = datetime.fromisoformat(start)
    times = [(t0 + timedelta(hours=h)).strftime("%Y-%m-%dT%H:%M") for h in range(H)]
    gio_vn = (np.arange(H) + 7) % 24
    uot = np.zeros((H, 3), bool)
    for h in range(1, H):
        chieu = 13 <= gio_vn[h] <= 19
        p_bat = 0.04 * (2.5 if chieu else 1.0)
        uot[h] = np.where(uot[h - 1], rng.random(3) > 0.25, rng.random(3) < p_bat)
    mua_diem = uot[:, vung] & (rng.random((H, S)) < 0.7)
    P = np.where(mua_diem, rng.gamma(0.8, 3.0, (H, S)), 0.0)
    chieu = ((gio_vn >= 13) & (gio_vn <= 21))[:, None]
    p_giong = np.where(P > 5, np.where(chieu, 0.5, 0.15), np.where(P > 0.5, 0.03, 0.0))
    giong = rng.random((H, S)) < p_giong
    code = np.where(giong, 95, np.where(P > 7.6, 65, np.where(P > 2.5, 63, np.where(P > 0.1, 61, 2))))
    gust = rng.weibull(2.0, (H, S)) * 6.0 + giong * rng.uniform(5, 12, (H, S))
    np.savez_compressed(out, site_names=np.array(names), lat=lat, lon=lon, time_utc=np.array(times),
                        precip=P.astype(np.float32), code=code.astype(np.int16), gust=gust.astype(np.float32),
                        nguon=np.array("gia-lap"))
    print(f"Đã lưu {out} (GIẢ LẬP): {H} giờ × {S} điểm; giờ có mưa {np.mean(P > 0.1) * 100:.1f}%, "
          f"giờ có giông {np.mean(giong) * 100:.2f}%")


def doc_thang(s: str | None):
    """'5-10' → {5,…,10}; '11-2' → {11,12,1,2}; None/'' → None (cả năm)."""
    if not s:
        return None
    a, b = (int(x) for x in s.split("-"))
    return set(range(a, b + 1)) if a <= b else set(range(a, 13)) | set(range(1, b + 1))


class ThoiTiet:
    def __init__(self, path: str):
        d = np.load(path, allow_pickle=False)
        self.names = [str(x) for x in d["site_names"]]
        self.lat, self.lon = d["lat"].astype(float), d["lon"].astype(float)
        self.time_utc = d["time_utc"]
        self.precip, self.code, self.gust = d["precip"].astype(float), d["code"], d["gust"].astype(float)
        self.nguon = str(d["nguon"])
        self.H = len(self.time_utc)
        self.gio_utc = np.array([int(str(t)[11:13]) for t in self.time_utc])
        self.thang = np.array([int(str(t)[5:7]) for t in self.time_utc])

    def gio_bat_dau(self, so_gio: int, thang=None):
        """Các giờ có thể bắt đầu một kịch bản dài so_gio giờ; thang: tập tháng được phép (ví dụ mùa mưa)."""
        h = np.arange(self.H - so_gio)
        return h if not thang else h[np.isin(self.thang[h], list(thang))]

    def idx(self, name: str) -> int:
        return self.names.index(name)

    def tom_tat(self):
        return {"nguon": self.nguon, "so_gio": int(self.H), "so_diem": len(self.names),
                "tu": str(self.time_utc[0]), "den": str(self.time_utc[-1]),
                "gio_co_mua_%": float(np.mean(self.precip > 0.1 + 1e-4) * 100),    # ERA5 làm tròn 0,1 mm
                "gio_co_giong_%": float(np.mean(np.isin(self.code, GIONG)) * 100)}


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")    # PowerShell / cmd trên Windows
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="lenh", required=True)
    a = sub.add_parser("tai")
    a.add_argument("--start", default="2023-01-01")
    a.add_argument("--end", default="2024-12-31")
    a.add_argument("--out", default="data/thoi_tiet_vn.npz")
    b = sub.add_parser("gia-lap")
    b.add_argument("--out", default="data/thoi_tiet_gia_lap.npz")
    b.add_argument("--days", type=int, default=184)
    b.add_argument("--seed", type=int, default=7)
    g = sub.add_parser("them-giong")
    g.add_argument("--vao", default="data/thoi_tiet_vn.npz")
    g.add_argument("--ra", default=None)
    g.add_argument("--luat", choices=["cape", "ma", "ca_hai"], default="cape")
    e = sub.add_parser("dat-giong", help="đặt lại giông theo luật khác, không tải lại")
    e.add_argument("--vao", default="data/thoi_tiet_vn.npz")
    e.add_argument("--luat", choices=["cape", "ma", "ca_hai"], default="cape")
    c = sub.add_parser("xem")
    c.add_argument("path")
    args = ap.parse_args()
    if args.lenh == "tai":
        tai_open_meteo(args.start, args.end, args.out)
    elif args.lenh == "them-giong":
        them_giong(args.vao, args.ra, luat=args.luat)
    elif args.lenh == "dat-giong":
        z = dict(np.load(args.vao, allow_pickle=False))
        ap_luat_giong(z, args.luat)
        np.savez_compressed(args.vao, **z)
        print(f"Đã đặt lại giông theo luật {args.luat}: {np.isin(z['code'], GIONG).mean() * 100:.2f}% số giờ")
        thong_ke_giong(z)
    elif args.lenh == "gia-lap":
        gia_lap(args.out, args.days, args.seed)
    else:
        print(json.dumps(ThoiTiet(args.path).tom_tat(), ensure_ascii=False, indent=1))
        z = dict(np.load(args.path, allow_pickle=False))
        if "cape" in z:
            thong_ke_giong(z)
