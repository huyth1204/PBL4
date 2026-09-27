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
# Cấu hình mặc định (Việt Nam) - chỉnh tại đây hoặc truyền danh sách riêng
# ----------------------------------------------------------------------------
DEFAULT_GATEWAYS = [
    ("Hanoi", 21.0285, 105.8542),
    ("DaNang", 16.0544, 108.2022),
    ("HoChiMinh", 10.7626, 106.6602),
    ("HaiPhong", 20.8449, 106.6881),
    ("CanTho", 10.0452, 105.7469),
]
DEFAULT_USERS = [
    ("Vinh", 18.6796, 105.6813),
    ("Hue", 16.4637, 107.5909),
    ("NhaTrang", 12.2388, 109.1967),
    ("QuyNhon", 13.7830, 109.2199),
    ("BuonMaThuot", 12.6667, 108.0500),
    ("CaMau", 9.1769, 105.1500),
    ("LaoCai", 22.4833, 103.9750),
    ("PleiKu", 13.9833, 108.0000),
]
HAPS_ALT_M = 20_000.0
DEFAULT_HAPS = [
    ("HAPS-North", 21.0, 105.9),
    ("HAPS-Central", 16.3, 108.0),
    ("HAPS-South", 10.9, 106.7),
    ("HAPS-Highland", 13.9, 108.0),   # phủ Tây Nguyên
    ("HAPS-Mekong", 10.0, 105.7),      # phủ ĐB sông Cửu Long
]
DEFAULT_UAV_CENTERS = [
    ("UAV-1", 18.6796, 105.6813),   # quanh Vinh
    ("UAV-2", 12.2388, 109.1967),   # quanh Nha Trang
    ("UAV-3", 10.0452, 105.7469),   # quanh Cần Thơ
    ("UAV-4", 22.4833, 103.9750),   # quanh Lào Cai
    ("UAV-5", 13.7830, 109.2199),   # quanh Quy Nhơn
]

def static_nodes(
    gateways=DEFAULT_GATEWAYS,
    users=DEFAULT_USERS,
    haps=DEFAULT_HAPS,
) -> list[Node]:
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


def nodes_at(elapsed_s: float, statics: list[Node], patrols: list[UAVPatrol]) -> list[Node]:
    """Toàn bộ nút Air+Ground tại thời điểm `elapsed_s` (giây kể từ snapshot đầu)."""
    return list(statics) + [p.node_at(elapsed_s) for p in patrols]
