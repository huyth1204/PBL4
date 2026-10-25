"""
sites.py — Nguồn cấu hình DUY NHẤT cho các nút mặt đất / trên không của SAGSIN.

Lý do tách riêng: trước đây danh sách điểm bị chép làm hai nơi
(`src/physics/air_ground.py` và bản sao `_DIEM_TV1` trong
`src/ngau_nhien/thoi_tiet.py`). Thêm nút mới mà quên cập nhật bản sao thì nút
đó mặc nhiên không có mưa/gió, KHÔNG báo lỗi. Từ nay cả hai bên cùng đọc một
nguồn, nên không thể lệch.

Nguyên tắc:
  * Cấu hình cũ (`DEFAULT_*`) GIỮ NGUYÊN giá trị và THỨ TỰ — đây là hợp đồng
    với mạng 73 nút. Mọi thay đổi ở đây sẽ làm lệch `node_order`.
  * Nút mới (`EXTRA_*`) là danh sách RIÊNG, chỉ dùng khi bật `mo_rong=True`.
    Nút mới luôn được nối vào CUỐI `node_order`, không xáo trộn nút cũ.
  * `kind` quyết định luật liên kết (`LOCAL_RULES` trong air_ground.py), nên
    nút đảo xa chỉ nối được qua vệ tinh là hệ quả tự nhiên (không có luật
    ground<->ground).
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Tầng GROUND cũ (13 nút) — THỨ TỰ LÀ HỢP ĐỒNG, không đổi
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Tầng AIR cũ (5 HAPS) — THỨ TỰ LÀ HỢP ĐỒNG, không đổi
# ---------------------------------------------------------------------------
HAPS_ALT_M = 20_000.0
DEFAULT_HAPS = [
    ("HAPS-North", 21.0, 105.9),
    ("HAPS-Central", 16.3, 108.0),
    ("HAPS-South", 10.9, 106.7),
    ("HAPS-Highland", 13.9, 108.0),   # phủ Tây Nguyên
    ("HAPS-Mekong", 10.0, 105.7),      # phủ ĐB sông Cửu Long
]

# Tâm vòng tuần tra UAV cũ (5 UAV) — THỨ TỰ LÀ HỢP ĐỒNG, không đổi
DEFAULT_UAV_CENTERS = [
    ("UAV-1", 18.6796, 105.6813),   # quanh Vinh
    ("UAV-2", 12.2388, 109.1967),   # quanh Nha Trang
    ("UAV-3", 10.0452, 105.7469),   # quanh Cần Thơ
    ("UAV-4", 22.4833, 103.9750),   # quanh Lào Cai
    ("UAV-5", 13.7830, 109.2199),   # quanh Quy Nhơn
]

# ---------------------------------------------------------------------------
# Nút MỞ RỘNG (chỉ dùng khi mo_rong=True) — nối vào CUỐI node_order
#
# Đảo / quần đảo: chỉ nối được qua vệ tinh (không có luật ground<->ground).
#   - Hoàng Sa, Trường Sa: là quần đảo, cần hiện diện để bản đồ của Trung
#     vẽ đúng; đặt tại đảo lớn nhất của quần đảo.
#   - Côn Đảo, Phú Quốc, Lý Sơn, Bạch Long Vĩ: đảo có dân, xa bờ.
# Vùng núi: địa hình cao, che khuất tầm nhìn -> ứng viên cần HAPS/UAV phủ.
# Vùng hay lũ: miền Trung, hay ngập và đứt liên lạc mặt đất.
# ---------------------------------------------------------------------------
# (tên, lat, lon, kind) — kind "gateway" (relay được) hoặc "user" (chỉ đầu/cuối)
EXTRA_ISLAND_SITES = [
    ("HoangSa", 16.8300, 112.3300, "gateway"),     # đảo Phú Lâm, quần đảo Hoàng Sa
    ("TruongSa", 8.6400, 111.9200, "gateway"),     # quần đảo Trường Sa
    ("ConDao", 8.6833, 106.6167, "user"),          # Côn Đảo, Bà Rịa - Vũng Tàu
    ("PhuQuoc", 10.2289, 103.9640, "gateway"),     # Phú Quốc, Kiên Giang
    ("LySon", 15.3800, 109.1170, "user"),          # Lý Sơn, Quảng Ngãi
    ("BachLongVi", 20.1333, 107.7242, "user"),     # Bạch Long Vĩ, Hải Phòng
]
# Ven biển + đồng bằng Bắc Bộ: vùng bão Yagi đổ bộ và chịu ảnh hưởng nặng nhất.
# Trước đây cấu hình mở rộng chỉ có nút hải đảo xa, nên không UAV nào ở gần tâm
# bão => luật ha_canh không bao giờ kích hoạt trong Yagi (xem SCHEMA.md §5).
EXTRA_COASTAL_SITES = [
    ("QuangNinh", 20.9500, 107.0800, "user"),      # Hạ Long, Quảng Ninh (bờ vịnh Bắc Bộ)
    ("ThaiBinh", 20.4500, 106.3400, "user"),       # Thái Bình (bờ biển, tâm bão Yagi)
    ("NamDinh", 20.4300, 106.1700, "user"),        # Nam Định (bờ biển)
]
EXTRA_MOUNTAIN_SITES = [
    ("SaPa", 22.3364, 103.8438, "user"),           # Sa Pa, Lào Cai
    ("DaLat", 11.9404, 108.4583, "gateway"),       # Đà Lạt, Lâm Đồng
    ("KonTum", 14.3497, 108.0000, "user"),         # Kon Tum, Tây Nguyên
]
EXTRA_FLOOD_SITES = [
    ("QuangBinh", 17.4800, 106.6000, "user"),      # Đồng Hới, Quảng Bình (hay lũ)
    ("QuangTri", 16.7500, 107.1900, "user"),       # Đông Hà, Quảng Trị (hay lũ)
]

EXTRA_GROUND_SITES = (EXTRA_ISLAND_SITES + EXTRA_COASTAL_SITES
                      + EXTRA_MOUNTAIN_SITES + EXTRA_FLOOD_SITES)

# HAPS phủ thêm cho các vùng mới (chỉ bật khi mo_rong=True). HAPS đứng yên,
# dùng làm phương án THAY THẾ UAV khi vùng có gió giật vượt ngưỡng.
EXTRA_HAPS = [
    ("HAPS-BienDong", 14.5, 111.5),   # phủ Biển Đông (Hoàng Sa / Trường Sa)
    ("HAPS-BacTrungBo", 17.4, 106.9),  # phủ vùng lũ Quảng Bình - Quảng Trị
]

# Vị trí đặt UAV TÙY CHỌN (đứng yên thay vì bay vòng). (tên, lat, lon)
# Dùng khi cần UAV làm relay cố định tại một điểm, ví dụ phủ đảo.
# Cụm ven biển Bắc đặt quanh Hải Phòng — nơi bão Yagi đổ bộ — để luật ha_canh
# thật sự có dịp kích hoạt trên dữ liệu bão thật (xem SCHEMA.md §5).
EXTRA_UAV_SITES = [
    ("UAV-HoangSa", 16.8300, 112.3300),   # đảo Phú Lâm (khớp toạ độ nút HoangSa)
    ("UAV-TruongSa", 8.6400, 111.9200),
    ("UAV-HaiPhong", 20.8449, 106.6881),   # trên TP Hải Phòng (cách HaiPhong 0 km)
    ("UAV-VinhBacBo", 20.3000, 106.8000),  # giữa vịnh Bắc Bộ, tiếp cận từ biển
    ("UAV-QuangYen", 20.8700, 106.8900),   # cân bằng: Hải Phòng 21.3 km / Quảng Ninh
                                           # 21.8 km, góc nâng đều >=5° (ngưỡng UAV_GROUND)
]


def validate_sites(sites, ten="sites"):
    """Kiểm tra tọa độ trước khi đưa vào mạng: lat ∈ [-90, 90], lon ∈ [-180, 180].

    Trả về danh sách đã kiểm; ném ValueError kèm tên điểm sai nếu có.
    """
    for row in sites:
        name, lat, lon = row[0], row[1], row[2]
        if not (-90.0 <= lat <= 90.0):
            raise ValueError(f"{ten}: điểm '{name}' có vĩ độ {lat} ngoài [-90, 90].")
        if not (-180.0 <= lon <= 180.0):
            raise ValueError(f"{ten}: điểm '{name}' có kinh độ {lon} ngoài [-180, 180].")
        if name in {"", None}:
            raise ValueError(f"{ten}: có điểm không tên.")
    return list(sites)


def ground_sites(mo_rong: bool = False):
    """Toàn bộ điểm mặt đất theo thứ tự cũ-trước-mới-sau. kind suy từ danh sách."""
    out = [(n, la, lo, "gateway") for n, la, lo in DEFAULT_GATEWAYS]
    out += [(n, la, lo, "user") for n, la, lo in DEFAULT_USERS]
    if mo_rong:
        out += validate_sites(EXTRA_GROUND_SITES, "EXTRA_GROUND_SITES")
    return out


# ---------------------------------------------------------------------------
# Khóa sắp xếp GIỮ NGUYÊN THỨ TỰ CŨ khi thêm nút mới
#
# oracle_labeler_sagsin.py sắp mỗi tầng theo tên. Nếu không sửa, nút mới như
# "HoangSa" sẽ chèn vào giữa danh sách ground và đẩy lệch chỉ số của TẤT CẢ
# nút cũ => hỏng model của Nhật/Trung. Khóa dưới đây trả (0, tên) cho nút cũ
# và (1, tên) cho nút mới, nên nút cũ luôn đứng trước, nút mới nối đuôi.
# ---------------------------------------------------------------------------
CAC_TEN_CU = (
    frozenset(n for n, *_ in DEFAULT_GATEWAYS)
    | frozenset(n for n, *_ in DEFAULT_USERS)
    | frozenset(n for n, *_ in DEFAULT_HAPS)
    | frozenset(n for n, *_ in DEFAULT_UAV_CENTERS)
)


def khoa_sap_xep(name: str):
    """Khóa sort: nút cũ (0) luôn trước nút mới (1); trong nhóm sort theo tên.

    Với cấu hình cũ (toàn nút cũ) khóa này tương đương `sorted(tên)`, nên
    mạng 73 nút không đổi.
    """
    return (0, name) if name in CAC_TEN_CU else (1, name)

