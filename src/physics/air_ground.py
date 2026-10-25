"""
Tầng AIR (HAPS, UAV) và GROUND (gateway, user terminal) cho SAGSIN.

Nguyên tắc:
  * Mỗi nút không phải vệ tinh là một `Node` (tên, layer, kind, lat/lon/alt).
  * Liên kết Air/Ground <-> Vệ tinh: TÁI SỬ DỤNG `compute_gsl_links` của
    tle_loader.py (Node.to_ground_station() có elevation_m = độ cao thật),
    nên góc ngẩng/khoảng cách nghiêng tính đúng bằng skyfield như cũ.
  * Liên kết cục bộ (HAPS-Ground, UAV-Ground, Air-Air): tính bằng ECEF + luật
    trong LOCAL_RULES bên dưới. Ground<->Ground KHÔNG có link (không có
    backbone cáp quang) - xem ghi chú ở LOCAL_RULES.
  * Tất cả tham số vật lý ở đây là GIẢ ĐỊNH để mô phỏng; nên trích nguồn
    (bài báo/chuẩn) khi đưa vào báo cáo.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations

import numpy as np

from src.physics.sites import (
    DEFAULT_GATEWAYS,
    DEFAULT_HAPS,
    DEFAULT_UAV_CENTERS,
    DEFAULT_USERS,
    EXTRA_HAPS,
    EXTRA_UAV_SITES,
    HAPS_ALT_M,
    ground_sites,
)
from src.physics.tle_loader import GroundStation

EARTH_RADIUS_KM = 6371.0
WGS84_A_KM = 6378.137
WGS84_E2 = 6.69437999014e-3

# ----------------------------------------------------------------------------
# Node
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class Node:
    name: str
    layer: str  # "air" | "ground"   (tầng space do TLE quản lý)
    kind: str  # "haps" | "uav" | "gateway" | "user"
    lat_deg: float
    lon_deg: float
    alt_m: float = 0.0
    can_relay: bool = True  # False: chỉ là điểm đầu/cuối, không được làm trung gian
    endpoint: bool = False  # True: được chọn làm src/dst khi gán nhãn

    def to_ground_station(self) -> GroundStation:
        """Dùng lại compute_gsl_links(); elevation_m = độ cao của nút."""
        return GroundStation(self.name, self.lat_deg, self.lon_deg, self.alt_m)


# ----------------------------------------------------------------------------
# Hình học (ECEF, km)
# ----------------------------------------------------------------------------


def geodetic_to_ecef_km(lat_deg: float, lon_deg: float, alt_m: float) -> np.ndarray:
    lat, lon = math.radians(lat_deg), math.radians(lon_deg)
    h = alt_m / 1000.0
    n = WGS84_A_KM / math.sqrt(1.0 - WGS84_E2 * math.sin(lat) ** 2)
    x = (n + h) * math.cos(lat) * math.cos(lon)
    y = (n + h) * math.cos(lat) * math.sin(lon)
    z = (n * (1.0 - WGS84_E2) + h) * math.sin(lat)
    return np.array([x, y, z], dtype=float)


def node_ecef_km(node: Node) -> np.ndarray:
    return geodetic_to_ecef_km(node.lat_deg, node.lon_deg, node.alt_m)


def elevation_deg(observer: Node, target_ecef_km: np.ndarray) -> float:
    """Góc ngẩng của `target` nhìn từ `observer` (so với mặt phẳng chân trời cục bộ)."""
    lat, lon = math.radians(observer.lat_deg), math.radians(observer.lon_deg)
    up = np.array(
        [math.cos(lat) * math.cos(lon), math.cos(lat) * math.sin(lon), math.sin(lat)]
    )
    v = target_ecef_km - node_ecef_km(observer)
    d = float(np.linalg.norm(v))
    if d == 0.0:
        return 90.0
    return math.degrees(math.asin(max(-1.0, min(1.0, float(up @ v) / d))))


def blocked_by_earth(a: np.ndarray, b: np.ndarray, radius_km: float = EARTH_RADIUS_KM) -> bool:
    """True nếu đoạn thẳng a-b xuyên qua hình cầu Trái Đất.
    Chỉ dùng cho cặp Air-Air (cả hai đều cao hơn mặt đất vài km)."""
    ab = b - a
    denom = float(ab @ ab)
    if denom == 0.0:
        return False
    s = min(1.0, max(0.0, float(-a @ ab) / denom))
    return float(np.linalg.norm(a + s * ab)) < radius_km


# ----------------------------------------------------------------------------
# Luật liên kết cục bộ
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class LocalLinkRule:
    link_type: str
    max_range_km: float
    min_elev_deg: float | None  # None: không xét góc ngẩng, chỉ xét bị Trái Đất che
    bandwidth_mbps: float
    packet_loss: float


_HAPS_GROUND = LocalLinkRule("HAPS_GROUND", 200.0, 10.0, 300.0, 0.005)
_UAV_GROUND = LocalLinkRule("UAV_GROUND", 25.0, 5.0, 100.0, 0.01)

# Độ cao UAV ĐẶT CỐ ĐỊNH (relay đứng yên ở một điểm) — khác `UAVPatrol.alt_m`
# của UAV bay vòng (2000 m, giữ nguyên để không đổi mạng cũ).
#
# Vì sao 2500 m: khi `min_elev_deg` là tiêu chí chặn, góc nâng ở CÙNG một vị
# trí đặt chỉ phụ thuộc độ cao UAV. Đo trên cặp Hải Phòng / Quảng Ninh (cùng
# cách `UAV-QuangYen` ~21 km):
#
#   alt 2000 m -> 5.29° / 5.18°  biên +0.29° / +0.18°  (mỏng: dịch toạ độ vài
#                                                      trăm mét là mất link)
#   alt 2500 m -> 6.63° / 6.49°  biên +1.63° / +1.49°
#
# Ngoài ra `max_range_km = 25` của UAV_GROUND KHÔNG BAO GIỜ chặn: ở alt 2000 m,
# góc nâng 5° ứng với tầm 22.4 km, tức tiêu chí chặt hơn đã bao trùm trong 25 km.
# Nếu muốn `max_range_km` thật sự có hiệu lực thì phải hạ `min_elev_deg`, không
# phải nâng `max_range_km`.
UAV_DAT_ALT_M: float = 2500.0

# Khoá là frozenset({kind_a, kind_b}). Cặp KHÔNG có trong bảng => không có link.
# Ground<->Ground cố ý KHÔNG có: nếu thêm backbone cáp quang giữa các gateway,
# Dijkstra sẽ luôn chọn cáp cho cặp gateway-gateway và nhãn trở nên tầm thường.
LOCAL_RULES: dict[frozenset, LocalLinkRule] = {
    frozenset(("haps", "gateway")): _HAPS_GROUND,
    frozenset(("haps", "user")): _HAPS_GROUND,
    frozenset(("uav", "gateway")): _UAV_GROUND,
    frozenset(("uav", "user")): _UAV_GROUND,
    frozenset(("haps", "haps")): LocalLinkRule("AIR_AIR", 700.0, None, 500.0, 0.005),
    frozenset(("haps", "uav")): LocalLinkRule("AIR_AIR", 150.0, 5.0, 200.0, 0.005),
    frozenset(("uav", "uav")): LocalLinkRule("AIR_AIR", 20.0, None, 100.0, 0.01),
}


def compute_local_links(
    nodes: list[Node],
    rules: dict[frozenset, LocalLinkRule] | None = None,
) -> list[dict]:
    """Liên kết giữa các nút Air/Ground với nhau. Cùng giao thức với compute_gsl_links:
    trả về MỌI cặp có luật, kèm cờ `exists`."""
    rules = LOCAL_RULES if rules is None else rules
    ecef = {n.name: node_ecef_km(n) for n in nodes}
    links: list[dict] = []
    for a, b in combinations(nodes, 2):
        rule = rules.get(frozenset((a.kind, b.kind)))
        if rule is None:
            continue
        low, high = (a, b) if a.alt_m <= b.alt_m else (b, a)
        dist = float(np.linalg.norm(ecef[a.name] - ecef[b.name]))
        elev = elevation_deg(low, ecef[high.name])
        if dist > rule.max_range_km:
            exists = False
        elif rule.min_elev_deg is not None:
            exists = elev >= rule.min_elev_deg
        else:
            exists = not blocked_by_earth(ecef[a.name], ecef[b.name])
        links.append(
            {
                "node_a": a.name,
                "node_b": b.name,
                "distance_km": dist,
                "elevation_deg": elev,
                "link_type": rule.link_type,
                "bandwidth_mbps": rule.bandwidth_mbps,
                "packet_loss": rule.packet_loss,
                "exists": exists,
            }
        )
    return links


# ----------------------------------------------------------------------------
# Cấu hình (Việt Nam) — nguồn sự thật ở src/physics/sites.py
#
# `mo_rong=False` (mặc định) tái lập ĐÚNG cấu hình cũ: 13 ground + 5 HAPS
# + 5 UAV = 23 nút không-vệ-tinh, giữ nguyên thứ tự để mạng 73 nút không đổi.
# `mo_rong=True` nối thêm nút mới vào CUỐI, không xáo trộn nút cũ.
# ----------------------------------------------------------------------------
# Tái xuất các hằng số để code cũ (thoi_tiet.py, ...) import được như trước.
__all__ = [
    "Node", "UAVPatrol", "static_nodes", "make_uav_patrols", "make_placed_uavs",
    "UAV_DAT_ALT_M",
    "nodes_at", "compute_local_links", "LOCAL_RULES", "LocalLinkRule",
    "DEFAULT_GATEWAYS", "DEFAULT_USERS", "DEFAULT_HAPS", "DEFAULT_UAV_CENTERS", "HAPS_ALT_M",
]


# Tên nút cũ (13 ground) — để phân biệt với nút mở rộng khi xếp thứ tự.
_TEN_GROUND_CU = frozenset(n for n, *_ in DEFAULT_GATEWAYS + DEFAULT_USERS)


def _la_them(name: str) -> bool:
    """True nếu `name` không thuộc cấu hình ground cũ (tức là nút mở rộng)."""
    return name not in _TEN_GROUND_CU


def static_nodes(
    gateways=None,
    users=None,
    haps=None,
    mo_rong: bool = False,
) -> list[Node]:
    """Nút đứng yên (ground + HAPS).

    mo_rong=False: đúng cấu hình cũ (13 ground, 5 HAPS), thứ tự không đổi.
    mo_rong=True : nối thêm ground mới (đảo/núi/lũ) rồi HAPS mới vào cuối.
    Có thể truyền `gateways`/`users`/`haps` riêng để ghi đè hoàn toàn.
    """
    if gateways is None:
        # Thứ tự hợp đồng: gateway cũ -> user cũ -> nút ground mới (nối đuôi).
        # KHÔNG lọc gateway-rồi-user trên toàn bộ, vì sẽ kéo gateway mới lên
        # trước user cũ, làm lệch vị trí nút cũ => phá mạng 73 nút.
        gs = ground_sites(mo_rong=mo_rong)
        cu_gw = [(n, la, lo) for n, la, lo, k in gs if k == "gateway" and not _la_them(n)]
        cu_user = [(n, la, lo) for n, la, lo, k in gs if k == "user" and not _la_them(n)]
        them = [(n, la, lo, k) for n, la, lo, k in gs if _la_them(n)]
        gateways = cu_gw + [(n, la, lo) for n, la, lo, k in them if k == "gateway"]
        users = cu_user + [(n, la, lo) for n, la, lo, k in them if k == "user"]
    else:
        gateways = list(gateways)
        users = DEFAULT_USERS if users is None else users
    if haps is None:
        haps = list(DEFAULT_HAPS) + (list(EXTRA_HAPS) if mo_rong else [])

    nodes: list[Node] = []
    for name, lat, lon in gateways:
        nodes.append(Node(name, "ground", "gateway", lat, lon, 0.0, can_relay=True, endpoint=True))
    for name, lat, lon in users:
        nodes.append(Node(name, "ground", "user", lat, lon, 0.0, can_relay=False, endpoint=True))
    for name, lat, lon in haps:
        nodes.append(Node(name, "air", "haps", lat, lon, HAPS_ALT_M, can_relay=True, endpoint=False))
    return nodes


# ----------------------------------------------------------------------------
# UAV: tuần tra tròn, xác định hoàn toàn bởi (seed, thời gian)
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class UAVPatrol:
    name: str
    center_lat_deg: float
    center_lon_deg: float
    radius_km: float = 10.0
    alt_m: float = 2000.0
    speed_mps: float = 25.0
    phase0_rad: float = 0.0

    def node_at(self, elapsed_s: float) -> Node:
        # radius_km = 0 => UAV ĐẶT CỐ ĐỊNH: luôn ở tâm, không chia cho 0,
        # giữ nguyên phase0/speed để hồ sơ nút vẫn đầy đủ (không thêm lớp mới).
        if self.radius_km <= 0.0:
            dlat = dlon = 0.0
        else:
            omega = (self.speed_mps / 1000.0) / self.radius_km  # rad/s
            theta = self.phase0_rad + omega * elapsed_s
            dlat = self.radius_km * math.cos(theta) / 111.32
            dlon = self.radius_km * math.sin(theta) / (
                111.32 * math.cos(math.radians(self.center_lat_deg))
            )
        return Node(
            self.name, "air", "uav",
            self.center_lat_deg + dlat, self.center_lon_deg + dlon, self.alt_m,
            can_relay=True, endpoint=True,
        )


def make_uav_patrols(seed: int, centers=DEFAULT_UAV_CENTERS) -> list[UAVPatrol]:
    """Pha đầu ngẫu nhiên nhưng cố định theo seed (rng RIÊNG, không đụng random toàn cục)."""
    rng = np.random.default_rng(seed)
    return [
        UAVPatrol(name, lat, lon, phase0_rad=float(rng.uniform(0.0, 2.0 * math.pi)))
        for name, lat, lon in centers
    ]


def make_placed_uavs(sites=EXTRA_UAV_SITES, alt_m: float = UAV_DAT_ALT_M) -> list[UAVPatrol]:
    """UAV ĐẶT CỐ ĐỊNH tại một vị trí (không bay vòng).

    Tách "vị trí đặt" khỏi "vòng tuần tra": dùng lại UAVPatrol với radius_km=0
    thì `node_at` trả về đúng tâm ở mọi thời điểm, nên không cần lớp mới và
    hành vi cũ (bay vòng) hoàn toàn không bị đụng.

    `alt_m` mặc định là `UAV_DAT_ALT_M` (2500 m), KHÔNG phải 2000 m: xem chú
    thích của hằng số đó. Truyền `alt_m=2000.0` nếu cần tái lập các bản dataset
    sinh trước commit này (chúng dùng 2000 m).
    """
    return [UAVPatrol(name, lat, lon, radius_km=0.0, alt_m=alt_m)
            for name, lat, lon in sites]


def nodes_at(elapsed_s: float, statics: list[Node], patrols: list[UAVPatrol]) -> list[Node]:
    """Toàn bộ nút Air+Ground tại thời điểm `elapsed_s` (giây kể từ snapshot đầu)."""
    return list(statics) + [p.node_at(elapsed_s) for p in patrols]
