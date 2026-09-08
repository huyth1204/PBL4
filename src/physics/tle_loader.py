
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import numpy as np
from skyfield.api import EarthSatellite, load, wgs84

# ----------------------------------------------------------------------------
# Hằng số vật lý / ngưỡng hệ thống
# ----------------------------------------------------------------------------
MIN_ELEVATION_DEG = 25.0       # ngưỡng góc ngẩng tối thiểu cho GSL (tránh suy hao khí quyển)
MAX_ISL_RANGE_KM = 5000.0      # giới hạn vật lý của laser bám hướng cho ISL
EARTH_RADIUS_KM = 6371.0


@dataclass
class GroundStation:
    """Trạm mặt đất cố định."""
    name: str
    lat_deg: float
    lon_deg: float
    elevation_m: float = 0.0


# ----------------------------------------------------------------------------
# 1. Nạp dữ liệu quỹ đạo (TLE) -> danh sách EarthSatellite
# ----------------------------------------------------------------------------
def load_tle(source: str | Path) -> list[EarthSatellite]:
    """
    Nạp file TLE (đường dẫn local hoặc URL) và trả về danh sách vệ tinh
    dưới dạng đối tượng skyfield.EarthSatellite (đã tích hợp SGP4).

    source: đường dẫn local (vd: 'data/tle/starlink.txt') hoặc URL trực tiếp
            tới file TLE (Skyfield sẽ tự tải và cache).
    """
    ts = load.timescale()
    satellites = load.tle_file(str(source), ts=ts)
    return satellites


# ----------------------------------------------------------------------------
# 2. Tính vị trí địa tâm (x, y, z) tại thời điểm snapshot τ_k
# ----------------------------------------------------------------------------
def satellite_positions_km(
    satellites: list[EarthSatellite], t
) -> dict[str, np.ndarray]:
    """
    Với mỗi vệ tinh, tính vị trí địa tâm (ECI, đơn vị km) tại thời điểm t.

    Trả về: {tên_vệ_tinh: array([x, y, z])}
    """
    positions = {}
    for sat in satellites:
        geocentric = sat.at(t)
        x, y, z = geocentric.position.km
        positions[sat.name] = np.array([x, y, z], dtype=float)
    return positions


# ----------------------------------------------------------------------------
# 3a. Liên kết Vệ tinh - Mặt đất (GSL): góc ngẩng + khoảng cách nghiêng
# ----------------------------------------------------------------------------
def compute_gsl_links(
    satellites: list[EarthSatellite],
    ground_stations: list[GroundStation],
    t,
    min_elevation_deg: float = MIN_ELEVATION_DEG,
) -> list[dict]:
    """
    Với mỗi cặp (trạm mặt đất, vệ tinh), tính góc ngẩng θ.
    Nếu θ >= min_elevation_deg, liên kết GSL tồn tại.

    Trả về danh sách dict:
        {
            "ground_station": tên trạm,
            "satellite": tên vệ tinh,
            "elevation_deg": góc ngẩng θ,
            "slant_range_km": khoảng cách nghiêng ℓ(θ),
            "exists": True/False,
        }
    """
    ts = load.timescale()
    links = []

    for gs in ground_stations:
        observer = wgs84.latlon(gs.lat_deg, gs.lon_deg, elevation_m=gs.elevation_m)
        for sat in satellites:
            difference = sat - observer
            topocentric = difference.at(t)
            alt, az, distance = topocentric.altaz()

            elevation_deg = alt.degrees
            slant_range_km = distance.km
            exists = elevation_deg >= min_elevation_deg

            links.append(
                {
                    "ground_station": gs.name,
                    "satellite": sat.name,
                    "elevation_deg": elevation_deg,
                    "slant_range_km": slant_range_km,
                    "exists": exists,
                }
            )
    return links


# ----------------------------------------------------------------------------
# 3b. Liên kết liên vệ tinh (ISL): khoảng cách Euclidean giữa các vệ tinh lân cận
# ----------------------------------------------------------------------------
def compute_isl_links(
    positions: dict[str, np.ndarray],
    max_range_km: float = MAX_ISL_RANGE_KM,
) -> list[dict]:
    """
    Với mọi cặp vệ tinh, tính khoảng cách Euclidean trong không gian ECI.
    Nếu khoảng cách < max_range_km, liên kết ISL được thiết lập.

    Trả về danh sách dict:
        {
            "sat_a": tên vệ tinh A,
            "sat_b": tên vệ tinh B,
            "distance_km": khoảng cách,
            "exists": True/False,
        }
    """
    links = []
    names = list(positions.keys())

    for name_a, name_b in combinations(names, 2):
        pos_a = positions[name_a]
        pos_b = positions[name_b]
        distance_km = float(np.linalg.norm(pos_a - pos_b))
        exists = distance_km < max_range_km

        links.append(
            {
                "sat_a": name_a,
                "sat_b": name_b,
                "distance_km": distance_km,
                "exists": exists,
            }
        )
    return links


# ----------------------------------------------------------------------------
# Demo / smoke test
# ----------------------------------------------------------------------------
def main():
    ts = load.timescale()
    t = ts.now()  # snapshot τ_k = thời điểm hiện tại

    tle_path = Path("data/tle/starlink.txt")
    if not tle_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy {tle_path}. Hãy tải TLE về data/tle/starlink.txt trước."
        )

    print(f"[1/4] Đang nạp TLE từ {tle_path} ...")
    satellites = load_tle(tle_path)
    print(f"      -> Nạp thành công {len(satellites)} vệ tinh.")

    # Tập nhỏ (20 vệ tinh) dùng để demo ISL nhanh, không tốn thời gian tính O(N^2).
    sample = satellites[:20]

    print(f"[2/4] Tính vị trí địa tâm tại snapshot τ = {t.utc_iso()} ...")
    positions = satellite_positions_km(sample, t)
    first_name = sample[0].name
    print(f"      -> {first_name}: {positions[first_name].round(1)} km (x, y, z)")

    print(f"[3/4] Tính liên kết GSL (trạm mặt đất - vệ tinh) trên TOÀN BỘ {len(satellites)} vệ tinh ...")
    ground_stations = [
        GroundStation("Hanoi", 21.0285, 105.8542),
        GroundStation("DaNang", 16.0544, 108.2022),
        GroundStation("HoChiMinh", 10.7626, 106.6602),
    ]
    # Dùng toàn bộ vệ tinh (không chỉ sample 20 con) để có xác suất bắt được
    # cửa sổ khả kiến (~4.5 phút/lượt) cao hơn, và để xác nhận công thức đúng.
    gsl_links = compute_gsl_links(satellites, ground_stations, t)
    active_gsl = [link for link in gsl_links if link["exists"]]
    print(f"      -> {len(active_gsl)}/{len(gsl_links)} liên kết GSL khả dụng (θ >= {MIN_ELEVATION_DEG}°)")
    for link in active_gsl[:5]:
        print(
            f"         {link['ground_station']:>10} <-> {link['satellite']:<15} "
            f"θ={link['elevation_deg']:.1f}°  ℓ={link['slant_range_km']:.0f} km"
        )

    # Dù có liên kết hay không, in ra góc ngẩng CAO NHẤT tìm được cho mỗi trạm
    # để xác nhận công thức tính đúng (không phải lỗi luôn ra âm/0).
    print("      Góc ngẩng cao nhất tìm được (để xác nhận công thức đúng):")
    for gs in ground_stations:
        best = max(
            (link for link in gsl_links if link["ground_station"] == gs.name),
            key=lambda link: link["elevation_deg"],
        )
        print(
            f"         {gs.name:>10}: θ_max={best['elevation_deg']:.1f}° "
            f"(với {best['satellite']}, ℓ={best['slant_range_km']:.0f} km)"
        )

    print("[4/4] Tính liên kết ISL (liên vệ tinh) trên tập mẫu 20 vệ tinh ...")
    isl_links = compute_isl_links(positions)
    active_isl = [link for link in isl_links if link["exists"]]
    print(f"      -> {len(active_isl)}/{len(isl_links)} liên kết ISL khả dụng (< {MAX_ISL_RANGE_KM} km)")
    for link in active_isl[:5]:
        print(
            f"         {link['sat_a']:<15} <-> {link['sat_b']:<15} "
            f"d={link['distance_km']:.0f} km"
        )

    print("\nHoàn tất Giai đoạn 1 (smoke test).")


if __name__ == "__main__":
    main()