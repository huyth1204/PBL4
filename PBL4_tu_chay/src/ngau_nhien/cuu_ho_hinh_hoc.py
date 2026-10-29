"""
cuu_ho_hinh_hoc.py — Vị trí ứng viên của UAV cứu hộ và hình học liên kết của chúng trên mạng v5 (TN6+4).

Luật liên kết lấy đúng của TV1 (src/physics/air_ground.py, tle_loader.py; kiểm lại bằng link_stats của v5):
    UAV – mặt đất (user, gateway)  ≤ 25 km, góc ngẩng nhìn từ mặt đất ≥ 5°,  C = 100 Mbps
    UAV – HAPS                     ≤ 150 km, góc ngẩng nhìn từ UAV ≥ 5°,     C = 200 Mbps
    UAV – UAV                      ≤ 20 km,                                   C = 100 Mbps
    UAV – vệ tinh LEO              góc ngẩng nhìn từ UAV ≥ 15°,               C = 100 Mbps
    trễ = khoảng cách / 300 + 12/C · (1 + 0,3/0,7) + 0,5   (ms)
Ứng viên: lưới 10 km quanh mỗi nút mặt đất, cách nút ≤ 20 km, độ cao 2,5 km (như UAV đặt cố định của TV1),
gộp các điểm cách nhau < 5 km. Thời tiết của ứng viên: điểm thời tiết gần nhất trong 74 điểm.

    python -m src.ngau_nhien.cuu_ho_hinh_hoc --kiem      # kiểm luật bằng một UAV có sẵn của v5
"""

from __future__ import annotations

import argparse

import numpy as np

from src.ngau_nhien.kich_ban import C_KM_MS, PKT, PROC, RHO0, _haversine

ALT_CUU_KM = 2.5
WGS84_A, WGS84_E2 = 6378.137, 6.69437999014e-3
LUAT = {"mat_dat": (25.0, 5.0, 100.0), "haps": (150.0, 5.0, 200.0), "uav": (20.0, None, 100.0), "leo": (None, 15.0, 100.0)}
# TN6+4 phiên bản 2 (hướng 2): UAV cứu hộ có thêm đường lên 5 GHz tầm xa — tới HAPS ≤ 300 km, góc ngẩng ≥ 1°; tới UAV khác
# (có sẵn hoặc UAV cứu hộ kia) ≤ 100 km. Liên kết xuống mặt đất và lên vệ tinh giữ như TV1.
LUAT_5GHZ = {"mat_dat": (25.0, 5.0, 100.0), "haps": (300.0, 1.0, 200.0), "uav": (100.0, None, 100.0), "leo": (None, 15.0, 100.0)}


def ecef(lat, lon, alt_km):
    la, lo = np.radians(lat), np.radians(lon)
    n = WGS84_A / np.sqrt(1.0 - WGS84_E2 * np.sin(la) ** 2)
    return np.stack([(n + alt_km) * np.cos(la) * np.cos(lo), (n + alt_km) * np.cos(la) * np.sin(lo),
                     (n * (1.0 - WGS84_E2) + alt_km) * np.sin(la)], -1)


def goc_ngang(lat_o, lon_o, P_o, P_t):
    """Góc ngẩng (độ) của đích P_t nhìn từ người quan sát (lat_o, lon_o, P_o); phát sóng theo mọi cặp."""
    la, lo = np.radians(lat_o), np.radians(lon_o)
    up = np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], -1)
    v = P_t - P_o
    d = np.linalg.norm(v, axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        s = np.clip((up * v).sum(-1) / d, -1, 1)
    return np.degrees(np.arcsin(s)), d


def tre(d_km, C):
    return d_km / C_KM_MS + PKT / C * (1 + RHO0 / (1 - RHO0)) + PROC


def ung_vien(raw, buoc_km=10.0, ban_kinh_km=20.0, gop_km=5.0):
    kind = np.array([str(k) for k in raw["node_kind"]])
    ten = [str(n) for n in raw["node_order"]]
    X0 = raw["node_latlon_alt"][0]
    lat, lon, nhan = [], [], []
    lech = [(dx, dy) for dx in np.arange(-ban_kinh_km, ban_kinh_km + 1e-9, buoc_km)
            for dy in np.arange(-ban_kinh_km, ban_kinh_km + 1e-9, buoc_km) if np.hypot(dx, dy) <= ban_kinh_km + 1e-9]
    for i in np.flatnonzero(np.isin(kind, ["user", "gateway"])):
        for k, (dx, dy) in enumerate(lech):
            la = X0[i, 0] + dy / 111.32
            lo = X0[i, 1] + dx / (111.32 * np.cos(np.radians(X0[i, 0])))
            if lat and _haversine(la, lo, np.array(lat), np.array(lon)).min() < gop_km:
                continue
            lat.append(float(la)); lon.append(float(lo)); nhan.append(f"CH-{ten[i]}-{k}")
    return np.array(lat), np.array(lon), nhan


def hinh_hoc(raw, lat, lon, luat=None):
    """G[t, c, v]: trễ cơ sở (ms) của liên kết ứng viên c – nút v ở ảnh chụp t, −1 nếu không có liên kết.
    Liên kết hai chiều cùng trễ cơ sở (như TV1). Trả kèm C[c, v] (Mbps) của từng cặp."""
    LUAT_ = LUAT if luat is None else luat
    X = raw["node_latlon_alt"].astype(np.float64)
    kind = np.array([str(k) for k in raw["node_kind"]])
    T, N = X.shape[0], X.shape[1]
    K = len(lat)
    Pc = ecef(lat, lon, np.full(K, ALT_CUU_KM))                                     # [K,3]
    G = np.full((T, K, N), -1.0, np.float32)
    C = np.zeros((K, N), np.float32)
    # tĩnh: mặt đất (góc nhìn từ mặt đất), HAPS (góc nhìn từ UAV)
    for loai, nhom in (("mat_dat", np.isin(kind, ["user", "gateway"])), ("haps", kind == "haps")):
        j = np.flatnonzero(nhom)
        Pv = ecef(X[0, j, 0], X[0, j, 1], X[0, j, 2])                                 # [J,3]
        if loai == "mat_dat":
            el, d = goc_ngang(X[0, j, 0][None], X[0, j, 1][None], Pv[None], Pc[:, None])
        else:
            el, d = goc_ngang(lat[:, None], lon[:, None], Pc[:, None], Pv[None])
        rmax, emin, bw = LUAT_[loai]
        co = (d <= rmax) & (el >= emin)
        G[:, :, j] = np.where(co, tre(d, bw), -1.0)[None]
        C[:, j] = bw
    # động: UAV có sẵn (UAV tuần tra di chuyển), vệ tinh
    ju, jl = np.flatnonzero(kind == "uav"), np.flatnonzero(kind == "leo")
    C[:, ju], C[:, jl] = LUAT_["uav"][2], LUAT_["leo"][2]
    for t in range(T):
        Pu = ecef(X[t, ju, 0], X[t, ju, 1], X[t, ju, 2])
        d = np.linalg.norm(Pc[:, None] - Pu[None], axis=-1)
        G[t][:, ju] = np.where(d <= LUAT_["uav"][0], tre(d, LUAT_["uav"][2]), -1.0)
        Pl = ecef(X[t, jl, 0], X[t, jl, 1], X[t, jl, 2])
        el, d = goc_ngang(lat[:, None], lon[:, None], Pc[:, None], Pl[None])
        G[t][:, jl] = np.where(el >= LUAT_["leo"][1], tre(d, LUAT_["leo"][2]), -1.0)
    return G, C


def kiem(raw):
    """Đặt một ứng viên đúng chỗ một UAV cố định có sẵn của v5 rồi so với ma trận W của TV1."""
    ten = [str(n) for n in raw["node_order"]]
    W = raw["W_snapshots"]
    for ten_uav in ("UAV-HaiPhong", "UAV-QuangYen", "UAV-CatBa"):
        i = ten.index(ten_uav)
        x = raw["node_latlon_alt"][0, i]
        G, _ = hinh_hoc(raw, np.array([x[0]]), np.array([x[1]]))
        g, w = G[:, 0, :].copy(), W[:, i, :].copy()
        g[:, i] = w[:, i] = -1
        co_g, co_w = g > 0, w > 0
        sai = np.abs(g[co_g & co_w] - w[co_g & co_w]).max()
        print(f"{ten_uav}: alt {x[2]:.1f} km | liên kết khớp tồn tại {np.mean(co_g == co_w) * 100:.3f}% "
              f"(chỉ ở ứng viên {int((co_g & ~co_w).sum())}, chỉ ở v5 {int((~co_g & co_w).sum())}) | sai trễ max {sai:.2e} ms")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="dataset/oracle_dataset_v5.npz")
    ap.add_argument("--kiem", action="store_true")
    a = ap.parse_args()
    raw = np.load(a.raw)
    if a.kiem:
        kiem(raw)
    la, lo, nh = ung_vien(raw)
    print(f"{len(la)} vị trí ứng viên")
