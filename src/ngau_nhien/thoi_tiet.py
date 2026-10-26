"""
thoi_tiet.py — Dữ liệu thời tiết theo giờ cho các điểm mặt đất / trên không của SAGSIN.

Nguồn thật: Open-Meteo Historical Weather API (ERA5, miễn phí, không cần khoá).
Khoảng ngày của hai tệp đang dùng là 2024-01-01 → 2025-12-31 (17 544 giờ). `--start`
và `--end`/`--so-ngay` là BẮT BUỘC — KHÔNG có mặc định. Trước đây mặc định là
2023-01-01 → 2024-12-31, nên chạy `tai` mà quên cờ sẽ ra tệp 17 520 giờ lệch hẳn
một năm và GHI ĐÈ data/thoi_tiet_vn.npz (mất Yagi 7–8/9/2024, sai `--thang`).
    python -m src.ngau_nhien.thoi_tiet tai --start 2024-01-01 --end 2025-12-31 --out data/thoi_tiet_vn.npz
Thêm cờ --mo-rong để lấy cả các điểm mới (đảo/núi/lũ, HAPS mới, UAV đặt cố định):
    python -m src.ngau_nhien.thoi_tiet tai --mo-rong --start 2024-01-01 --end 2025-12-31 --out data/thoi_tiet_vn_mr.npz
(23 điểm cũ giữ nguyên thứ tự và chỉ số cột, điểm mới nối vào cuối.)
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

# Nguồn cấu hình DUY NHẤT: src/physics/sites.py. Trước đây file này giữ bản sao
# `_DIEM_TV1`; bản sao đó là bẫy — thêm nút mới ở air_ground.py mà quên chỗ này
# thì nút mới lặng lẽ KHÔNG có mưa/gió. Từ nay chỉ còn một nguồn.
if str(Path(__file__).resolve().parents[2]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.physics.sites import cac_diem_thoi_tiet  # noqa: E402


def cac_diem(mo_rong: bool = False):
    """Tên và toạ độ các điểm có cột thời tiết (UAV lấy tâm vòng tuần tra).

    mo_rong=False: đúng 23 điểm cũ, GIỮ NGUYÊN thứ tự => tệp thời tiết cũ còn dùng được.
    mo_rong=True : nối thêm điểm mới vào cuối (đảo/núi/lũ, HAPS mới, UAV đặt cố định).

    Thân hàm nằm ở `sites.cac_diem_thoi_tiet` để cả ba nơi (sinh tệp, kiểm tập
    điểm, vẽ bản đồ) dùng đúng MỘT danh sách — lệch danh sách chính là lỗi đã
    xảy ra: tệp có `UAV-VinhBacBo` (đã xoá khỏi cấu hình) mà thiếu 8 UAV mới.
    """
    return cac_diem_thoi_tiet(mo_rong=mo_rong)


def kiem_tra_du_diem(npz_path, mo_rong: bool = False):
    """BẮT BUỘC: tập điểm trong tệp thời tiết phải KHỚP ĐÚNG tập điểm của cấu hình.

    Kiểm hai chiều:
      * thiếu -> nút lặng lẽ không có mưa/gió: `kich_ban.sinh_kich_ban()` bỏ qua
        nút có `node_site < 0`, nên UAV đó KHÔNG BAO GIỜ hạ cánh vì giông/gió và
        `tao_du_lieu` để đặc trưng thời tiết bằng 0.
      * thừa -> cột chết. Nguy hiểm hơn vẻ ngoài: một tên đã bị xoá khỏi
        `sites.py` (vd `UAV-VinhBacBo`) vẫn còn cột thì tệp vẫn "đủ điểm" nếu chỉ
        kiểm một chiều, và người đọc tưởng UAV mới đã có thời tiết.

    Ném ValueError kèm cả hai danh sách. Không có bước này, lệch cấu hình chỉ lộ
    ra khi soi kết quả mô phỏng.
    """
    can = [n for n, _, _ in cac_diem(mo_rong=mo_rong)]
    with np.load(npz_path, allow_pickle=False) as d:
        co = [str(x) for x in d["site_names"]]
    thieu = [n for n in can if n not in set(co)]
    thua = [n for n in co if n not in set(can)]
    if thieu or thua:
        raise ValueError(
            f"Tệp thời tiết '{npz_path}' lệch với cấu hình (mo_rong={mo_rong}): "
            f"{len(co)} điểm trong tệp, cấu hình cần {len(can)}. "
            f"Thiếu {len(thieu)}: {thieu}. Thừa {len(thua)}: {thua}. "
            f"Chạy lại 'thoi_tiet tai --mo-rong' rồi 'them-giong' để đủ điểm.")
    return co


def suy_mo_rong(so_diem: int) -> bool:
    """Tệp 23 điểm là mạng 73 nút (hẹp); tệp nhiều hơn là mạng mở rộng."""
    return so_diem > len(cac_diem(mo_rong=False))



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


def kiem_tra_ghi_de(out: str, start: str, end: str, cho_phep: bool = False):
    """Chặn ghi đè một tệp thời tiết đã có bằng khoảng ngày KHÁC (hoặc khác tập điểm).

    Bẫy đã sập: mặc định cũ 2023-01-01 → 2024-12-31 của lệnh `tai` ghi thẳng vào
    data/thoi_tiet_vn.npz, xoá mất khoảng 2024-01-01 → 2025-12-31 (17 544 giờ, có bão
    Yagi 7–8/9/2024) mà không báo gì. Mọi kết quả phụ thuộc ngày sau đó đều sai âm thầm.
    Cùng khoảng ngày thì ghi đè bình thường (đó là cách tải lại dữ liệu).
    """
    p = Path(out)
    if not p.exists() or cho_phep:
        return
    try:
        with np.load(p, allow_pickle=False) as d:
            t = [str(x) for x in d["time_utc"]]
            so_diem = int(len(d["site_names"])) if "site_names" in d.files else -1
    except Exception as e:                                      # tệp hỏng / không phải tệp thời tiết
        print(f"CẢNH BÁO: {out} đã có nhưng không đọc được ({e}); sẽ ghi đè.")
        return
    cu = (t[0][:10], t[-1][:10]) if t else (None, None)
    if cu != (start, end):
        raise SystemExit(
            f"TỪ CHỐI GHI ĐÈ {out}\n"
            f"  tệp đang có : {cu[0]} → {cu[1]} ({len(t)} giờ, {so_diem} điểm)\n"
            f"  sắp ghi     : {start} → {end}\n"
            "Ghi đè bằng khoảng khác sẽ làm sai mọi kết quả phụ thuộc ngày (Yagi, --thang).\n"
            "  - Muốn giữ tệp cũ: dùng --out khác (vd data/thoi_tiet_vn_mr.npz).\n"
            "  - Chắc chắn muốn thay: thêm --cho-ghi-de.")


def tai_open_meteo(start: str, end: str, out: str, chunk_days: int = 180, mo_rong: bool = False):
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    diem = cac_diem(mo_rong=mo_rong)
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
    # Chặn ngay tại đây: tệp vừa tải --mo-rong mà lệch cấu hình thì mọi bước sau
    # (giông, mô phỏng) đều sai âm thầm.
    kiem_tra_du_diem(vao, mo_rong=len(d["site_names"]) > 23)
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


def gia_lap(out: str, days: int = 184, seed: int = 7, start: str = "2024-05-01", mo_rong: bool = False):
    """Thời tiết giả lập kiểu mùa mưa Việt Nam: mưa theo vùng (Markov), đối lưu chiều, giông khi mưa to.

    Chỉ để chạy thử quy trình. Các tham số là giả định, không phải số đo.
    """
    rng = np.random.default_rng(seed)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    diem = cac_diem(mo_rong=mo_rong)
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
    def __init__(self, path: str, mo_rong: bool | None = None, kiem_tra: bool = True):
        """mo_rong=None: suy từ số điểm (23 -> hẹp, nhiều hơn -> mở rộng).

        kiem_tra=True (mặc định): bắt buộc tập điểm khớp cấu hình NGAY khi đọc.
        Lệch tập điểm làm `MangGoc` gán `node_site = -1`, rồi `sinh_kich_ban()`
        bỏ qua nút đó — UAV lặng lẽ không bao giờ hạ cánh, đặc trưng thời tiết
        bằng 0, không một dòng cảnh báo. `MangGoc` cũng tự kiểm (chốt chặn
        chính, xem `src/ngau_nhien/kich_ban.py`), nên đường chạy nào dựng mạng
        cũng bị chặn kể cả khi quên bật ở đây.
        """
        d = np.load(path, allow_pickle=False)
        self.path = str(path)
        self.names = [str(x) for x in d["site_names"]]
        if kiem_tra:
            kiem_tra_du_diem(self.path, mo_rong=suy_mo_rong(len(self.names)) if mo_rong is None else mo_rong)
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
    # KHÔNG đặt mặc định cho --start / --end: mặc định cũ (2023-01-01 → 2024-12-31)
    # âm thầm sinh tệp lệch một năm và ghi đè data/thoi_tiet_vn.npz. Bắt buộc truyền.
    ngay = a.add_mutually_exclusive_group(required=True)
    ngay.add_argument("--end", help="ngày cuối, YYYY-MM-DD (dùng cùng --start)")
    ngay.add_argument("--so-ngay", type=int, help="lấy --so-ngay ngày tính từ --start")
    a.add_argument("--start", required=True, help="ngày đầu, YYYY-MM-DD (bắt buộc)")
    a.add_argument("--out", default="data/thoi_tiet_vn.npz")
    a.add_argument("--cho-ghi-de", action="store_true",
                   help="cho phép ghi đè tệp đã có bằng một khoảng thời gian khác khoảng của tệp")
    a.add_argument("--mo-rong", action="store_true",
                   help="tải cả điểm mới (đảo/núi/lũ, HAPS mới, UAV đặt cố định); 23 điểm cũ giữ nguyên chỉ số")
    b = sub.add_parser("gia-lap")
    b.add_argument("--out", default="data/thoi_tiet_gia_lap.npz")
    b.add_argument("--days", type=int, default=184)
    b.add_argument("--seed", type=int, default=7)
    b.add_argument("--mo-rong", action="store_true")
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
        if args.so_ngay is not None:
            if args.so_ngay < 1:
                ap.error("--so-ngay phải >= 1")
            end = (datetime.fromisoformat(args.start) + timedelta(days=args.so_ngay - 1)).date().isoformat()
        else:
            end = args.end
        if datetime.fromisoformat(end) < datetime.fromisoformat(args.start):
            ap.error(f"--end ({end}) phải >= --start ({args.start})")
        kiem_tra_ghi_de(args.out, args.start, end, args.cho_ghi_de)
        tai_open_meteo(args.start, end, args.out, mo_rong=args.mo_rong)
    elif args.lenh == "them-giong":
        them_giong(args.vao, args.ra, luat=args.luat)
    elif args.lenh == "dat-giong":
        z = dict(np.load(args.vao, allow_pickle=False))
        ap_luat_giong(z, args.luat)
        np.savez_compressed(args.vao, **z)
        print(f"Đã đặt lại giông theo luật {args.luat}: {np.isin(z['code'], GIONG).mean() * 100:.2f}% số giờ")
        thong_ke_giong(z)
    elif args.lenh == "gia-lap":
        gia_lap(args.out, args.days, args.seed, mo_rong=args.mo_rong)
    else:
        print(json.dumps(ThoiTiet(args.path).tom_tat(), ensure_ascii=False, indent=1))
        z = dict(np.load(args.path, allow_pickle=False))
        if "cape" in z:
            thong_ke_giong(z)
