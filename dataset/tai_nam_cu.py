"""
tai_nam_cu.py — tải thời tiết 74 điểm cho các năm TRƯỚC 2022 (vd 2017–2021), cùng định dạng và cùng toạ độ với
data/thoi_tiet_vn_mr.npz, để thêm cơn bão.

Chạy trên máy có mạng, ở nhánh mang-mo-rong-v3, đặt cạnh tai_them_nam.py (dùng lại hàm của nó):

    python dataset/tai_nam_cu.py --start 2017-01-01 --end 2021-12-31

Ra data/thoi_tiet_vn_2017_2021.npz. Chạy lại cùng lệnh nếu đứt mạng: phần đã tải nằm trong các tệp *.tam_*.npz.

Khác tai_them_nam.py:
  - tải thẳng tại TOẠ ĐỘ TỆP MẪU (không cần bước --sua cho 17 cột UAV);
  - giông: thử Historical Forecast API mặc định (best_match, như 2022–2025). Trước 2022 kho này thường thiếu; nếu thiếu
    CAPE ở hơn 2% số giờ thì lấy CAPE của ECMWF IFS HRES 9 km (models=ecmwf_ifs, có từ 2017-01-01) cho CẢ khoảng, và hiệu chỉnh
    ngưỡng CAPE để tỉ lệ giờ giông khớp tệp mẫu: tải thêm CAPE của IFS cho 2024–2025, chọn ngưỡng c* sao cho
    (CAPE_IFS ≥ c*) và (mưa ≥ 1 mm/h) cho đúng tỉ lệ giông của tệp mẫu (luật CAPE ≥ 1000 với best_match). Ghi c* vào trường
    `cape_nguong` và `nguon`, in độ trùng giờ giông giữa hai cách trên 2024–2025.
Bộ mô phỏng chỉ dùng mưa, gió giật và mã giông (không dùng giá trị CAPE), nên khớp tỉ lệ giông là đủ để các năm nhất quán.

Giới hạn Open-Meteo miễn phí: 600 lượt / phút, 5.000 / giờ, 10.000 / ngày; theo công bố, yêu cầu dài hơn 14 ngày hoặc
nhiều hơn 10 biến tính nhiều lượt (theo số điểm × số ngày / 14), nên một khúc 74 điểm × 180 ngày có thể tính tới khoảng
950 lượt (lần tải 2022–2023 thực tế chỉ gặp giới hạn phút). Vì vậy công cụ:
  - tải từng khúc 180 ngày, lưu từng khúc (đứt giữa chừng chạy lại không mất phần đã tải), nghỉ giữa các khúc để dưới
    600 lượt / phút; gặp 429 thì chờ theo loại giới hạn (phút: 65 s; giờ: 60 phút; ngày: dừng, chạy lại sau 7 giờ sáng);
  - thử best_match bằng MỘT yêu cầu nhỏ trước; thiếu thì bỏ qua, không tải cả khoảng;
  - với IFS chỉ tải CAPE (code_fc, lifted_index để 0 — bộ mô phỏng không dùng).
Nếu chạm giới hạn giờ, công cụ tự chờ 1 giờ; chạm giới hạn ngày thì dừng, sáng hôm sau chạy lại đúng lệnh là tải tiếp.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

GOC = Path(__file__).resolve().parents[1]
for p in (str(GOC), str(GOC / "dataset")):
    if p not in sys.path:
        sys.path.insert(0, p)
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError, OSError):
        pass

from tai_them_nam import kiem  # noqa: E402

ERA5 = "https://archive-api.open-meteo.com/v1/archive"
DU_BAO = "https://historical-forecast-api.open-meteo.com/v1/forecast"
MAU_KHOANG = ("2024-01-01", "2025-12-31")      # khoảng của tệp mẫu, dùng để hiệu chỉnh ngưỡng CAPE của IFS


def goi_tung_khuc(url, lat, lon, a, b, bien, extra=None, tam=None, get=None, ngu=time.sleep, khuc=180):
    """Gọi Open-Meteo theo khúc `khuc` ngày; lưu từng khúc vào <tam>_<ngày đầu>.npz; tôn trọng giới hạn lượt."""
    import requests
    get = get or requests.get
    t0, t1 = datetime.fromisoformat(a), datetime.fromisoformat(b)
    times, cols = [], {v: [] for v in bien}
    x = t0
    while x <= t1:
        y = min(x + timedelta(days=khuc - 1), t1)
        f = Path(f"{tam}_{x.date()}.npz") if tam else None
        if f is not None and f.exists():
            z = np.load(f, allow_pickle=False)
            times += [str(u) for u in z["time"]]
            for v in bien:
                cols[v].append(z[v])
            x = y + timedelta(days=1)
            continue
        params = {"latitude": ",".join(f"{u:.4f}" for u in lat), "longitude": ",".join(f"{u:.4f}" for u in lon),
                  "start_date": x.date().isoformat(), "end_date": y.date().isoformat(), "hourly": ",".join(bien),
                  "timezone": "GMT", **(extra or {})}
        for lan in range(30):
            r = get(url, params=params, timeout=300)
            if r.status_code == 200:
                break
            ly_do = r.text[:200]
            if r.status_code == 429 and "Daily" in ly_do:
                raise SystemExit("Hết hạn mức NGÀY của Open-Meteo. Phần đã tải được giữ; chạy lại đúng lệnh sau 7 giờ sáng (0h UTC).")
            cho = 3700 if (r.status_code == 429 and "Hourly" in ly_do) else 65 if r.status_code == 429 else 20 * min(lan + 1, 6)
            print(f"  lỗi {r.status_code}: {ly_do} — chờ {cho} s", flush=True)
            ngu(cho)
        else:
            raise SystemExit("Thử lại quá nhiều lần; chạy lại đúng lệnh (phần đã tải được giữ).")
        js = r.json()
        js = js if isinstance(js, list) else [js]
        tt = js[0]["hourly"]["time"]
        khoi = {v: np.array([j["hourly"].get(v, [None] * len(tt)) for j in js], float).T for v in bien}
        if f is not None:
            np.savez_compressed(f, time=np.array(tt), **khoi)
        times += tt
        for v in bien:
            cols[v].append(khoi[v])
        so_ngay = (y - x).days + 1
        luot = len(lat) * max(so_ngay / 14, 1) * max(len(bien) / 10, 1)
        print(f"  {x.date()} → {y.date()}: {len(tt)} giờ × {len(js)} điểm (≈ {luot:.0f} lượt)", flush=True)
        ngu(max(1.0, luot / 600 * 60 * 1.1))                 # giữ dưới 600 lượt / phút
        x = y + timedelta(days=1)
    return times, {v: np.concatenate(cols[v]) for v in bien}


def _tai(tep: Path, goi, url, lat, lon, a, b, bien, extra=None):
    """Tải (hoặc đọc lại khối đã có, kể cả khối của bản công cụ trước) một khối Open-Meteo."""
    if tep.exists():
        z = np.load(tep, allow_pickle=False)
        return [str(x) for x in z["time"]], {v: z[v] for v in bien}
    if goi is not None:                                        # kiểm ngoại tuyến
        t, c = goi(url, lat, lon, a, b, bien, 180, extra)
    else:
        t, c = goi_tung_khuc(url, lat, lon, a, b, bien, extra, tam=str(tep)[:-4])
    np.savez_compressed(tep, time=np.array(t), **{v: np.asarray(c[v], float) for v in bien})
    for f in tep.parent.glob(tep.stem + "_*.npz"):
        f.unlink()
    return [str(x) for x in t], {v: np.asarray(c[v], float) for v in bien}


def _khop_gio(t_dich, t_nguon, c):
    vt = {x: i for i, x in enumerate(t_nguon)}
    idx = np.array([vt.get(x, -1) for x in t_dich])
    ra = {}
    for k, v in c.items():
        m = np.full((len(t_dich), v.shape[1]), np.nan)
        m[idx >= 0] = v[idx[idx >= 0]]
        ra[k] = m
    return ra


def thieu_pt(cape):
    """% số giờ mà MỌI điểm thiếu CAPE (NaN)."""
    return float(np.mean(np.isnan(cape).all(1)) * 100)


def nguong_hieu_chinh(cape_ifs, mua, ti_le):
    """c* sao cho mean((cape_ifs ≥ c*) & (mua ≥ 1)) ≈ ti_le."""
    from src.ngau_nhien.thoi_tiet import MUA_GIONG
    co_mua = mua >= MUA_GIONG
    x = np.nan_to_num(cape_ifs[co_mua])
    q = 1 - ti_le / co_mua.mean()
    return float(np.quantile(x, np.clip(q, 0, 1)))


def tao(start, end, mau, thu_muc: Path, goi=None, ep_ifs=False):
    from src.ngau_nhien import thoi_tiet as T
    la, lo = mau["lat"], mau["lon"]
    tam = lambda ten: thu_muc / f"thoi_tiet_vn_{start[:4]}_{end[:4]}.tam_{ten}.npz"
    t, c = _tai(tam("era5"), goi, ERA5, la, lo, start, end, ["precipitation", "weather_code", "wind_gusts_10m"],
                {"wind_speed_unit": "ms"})
    P = np.nan_to_num(c["precipitation"], nan=0.0).astype(np.float32)
    G = np.nan_to_num(c["wind_gusts_10m"], nan=0.0).astype(np.float32)
    C0 = np.nan_to_num(c["weather_code"], nan=0.0).astype(np.int16)
    print(f"ERA5: {len(t)} giờ × {P.shape[1]} điểm")
    bien = ["weather_code", "cape", "lifted_index"]
    nguong, ghi = T.CAPE_GIONG, "best_match"
    fc = None
    if not ep_ifs:                                             # thăm dò: 1 điểm, 1 ngày ở đầu khoảng
        _, cp = (goi or (lambda *x: goi_tung_khuc(*x[:6], x[7])))(DU_BAO, la[:1], lo[:1], start, start, ["cape"], 1, None)
        if np.isnan(cp["cape"]).all():
            print("best_match không có CAPE ở đầu khoảng → dùng ECMWF IFS HRES")
            ep_ifs = True
    if not ep_ifs:
        tf, cf = _tai(tam("du_bao"), goi, DU_BAO, la, lo, start, end, bien)
        fc = _khop_gio(t, tf, cf)
        th = thieu_pt(fc["cape"])
        print(f"Dự báo lưu trữ mặc định (best_match): thiếu CAPE {th:.1f}% số giờ")
        if th > 2:
            fc = None
    if fc is None:
        tf, cf = _tai(tam("ifs"), goi, DU_BAO, la, lo, start, end, ["cape"], {"models": "ecmwf_ifs"})
        fc = _khop_gio(t, tf, cf)
        fc["weather_code"] = np.zeros_like(fc["cape"])           # không tải (bộ mô phỏng không dùng)
        fc["lifted_index"] = np.zeros_like(fc["cape"])
        th = thieu_pt(fc["cape"])
        print(f"ECMWF IFS HRES: thiếu CAPE {th:.1f}% số giờ")
        if th > 2:
            raise SystemExit("IFS cũng thiếu CAPE > 2% — dừng, gửi log cho TV2 (không tự đổi luật giông).")
        tm, cm = _tai(tam("ifs_hieu_chinh"), goi, DU_BAO, la, lo, *MAU_KHOANG, ["cape"], {"models": "ecmwf_ifs"})
        cm = _khop_gio([str(x) for x in mau["time_utc"]], tm, cm)["cape"]
        g_mau = mau["code"] == 95
        nguong = nguong_hieu_chinh(cm, mau["precip"], g_mau.mean())
        g_ifs = (np.nan_to_num(cm) >= nguong) & (mau["precip"] >= T.MUA_GIONG)
        jac = (g_ifs & g_mau).sum() / max((g_ifs | g_mau).sum(), 1)
        print(f"Hiệu chỉnh trên 2024–2025: ngưỡng CAPE IFS c* = {nguong:.0f} J/kg; giông {g_ifs.mean() * 100:.2f}% "
              f"(mẫu {g_mau.mean() * 100:.2f}%); độ trùng giờ–điểm (Jaccard) {jac:.2f}")
        ghi = (f"ecmwf_ifs, ngưỡng CAPE hiệu chỉnh {nguong:.0f} J/kg trên 2024–2025 (Jaccard {jac:.2f}); "
               "code_fc, lifted_index không tải (= 0)")
    cape = np.nan_to_num(fc["cape"]).astype(np.float32)
    giong = (cape >= nguong) & (P >= T.MUA_GIONG)
    d = {"site_names": np.array([str(x) for x in mau["site_names"]]), "lat": la.copy(), "lon": lo.copy(),
         "time_utc": np.array(t), "precip": P, "code": np.where(giong, 95, C0).astype(np.int16), "gust": G,
         "nguon": np.array(f"open-meteo-era5 + giông (cape; {ghi}); tải tại toạ độ tệp mẫu"),
         "code_fc": np.nan_to_num(fc["weather_code"]).astype(np.int16), "cape": cape,
         "lifted_index": np.nan_to_num(fc["lifted_index"]).astype(np.float32), "code_era5": C0,
         "cape_nguong": np.float32(nguong)}
    return d


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--mau", default=str(GOC / "data/thoi_tiet_vn_mr.npz"))
    ap.add_argument("--ep-ifs", action="store_true", help="bỏ qua best_match, dùng thẳng IFS + hiệu chỉnh")
    a = ap.parse_args()
    out = Path(a.out or GOC / f"data/thoi_tiet_vn_{a.start[:4]}_{a.end[:4]}.npz")
    if out.exists():
        raise SystemExit(f"{out} đã có; xoá hoặc dùng --out khác")
    mau = dict(np.load(a.mau, allow_pickle=False))
    d = tao(a.start, a.end, mau, out.parent, ep_ifs=a.ep_ifs)
    loi, thieu = kiem(d, mau, a.start, a.end)
    np.savez_compressed(out, **d)
    h = hashlib.sha256(out.read_bytes()).hexdigest()
    nam = np.array([int(str(x)[:4]) for x in d["time_utc"]])
    print(f"\nĐã lưu {out}: {len(d['time_utc'])} giờ × {len(d['site_names'])} điểm, sha256 {h[:16]}…")
    print("  giông theo năm: " + "  ".join(f"{y}: {(d['code'][nam == y] == 95).mean() * 100:.2f}%" for y in np.unique(nam)))
    if loi:
        print("CẢNH BÁO — kiểm không đạt:\n  - " + "\n  - ".join(loi))
        sys.exit(1)
    for f in out.parent.glob(f"{out.stem}.tam_*.npz"):
        f.unlink()
    print("Kiểm đạt. Gửi tệp này cho TV2 (đính kèm vào chat).")


if __name__ == "__main__":
    main()