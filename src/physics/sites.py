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
    # Hoàng Sa / Trường Sa là "user", KHÔNG phải gateway: đảo không có hạ tầng
    # nối ra ngoài, nên không được là đích hợp lệ của luồng khẩn cấp. Trước đây
    # để "gateway" nên luồng khẩn cấp có thể chọn đích ra đảo — sai hiện thực.
    ("HoangSa", 16.8300, 112.3300, "user"),         # đảo Phú Lâm, quần đảo Hoàng Sa
    ("TruongSa", 8.6400, 111.9200, "user"),         # quần đảo Trường Sa
    ("ConDao", 8.6833, 106.6167, "user"),           # Côn Đảo, Bà Rịa - Vũng Tàu
    ("PhuQuoc", 10.2289, 103.9640, "user"),         # Phú Quốc, Kiên Giang
    ("LySon", 15.3800, 109.1170, "user"),           # Lý Sơn, Quảng Ngãi
    ("BachLongVi", 20.1333, 107.7242, "user"),      # Bạch Long Vĩ, Hải Phòng
]
# Ven biển + đồng bằng Bắc Bộ: vùng bão Yagi đổ bộ và chịu ảnh hưởng nặng nhất.
# Trước đây cấu hình mở rộng chỉ có nút hải đảo xa, nên không UAV nào ở gần tâm
# bão => luật ha_canh không bao giờ kích hoạt trong Yagi (xem SCHEMA.md §5).
EXTRA_COASTAL_SITES = [
    ("QuangNinh", 20.9500, 107.0800, "user"),      # Hạ Long, Quảng Ninh (bờ vịnh Bắc Bộ)
    # Thái Bình / Nam Định KHÔNG sát biển: trung tâm thành phố cách bờ ~20-30 km.
    # Trong bão Yagi (7-9/9/2024) hai nút này chỉ ở RÌA bão: giật 30.7 và 27.6 m/s,
    # mưa 217 và 175 mm — thấp hơn Bạch Long Vĩ (58.7 m/s) và Hải Phòng (41.8 m/s).
    # Đừng gọi chúng là "tâm bão" khi viết báo cáo.
    ("ThaiBinh", 20.4500, 106.3400, "user"),       # TP Thái Bình (đồng bằng, rìa bão)
    ("NamDinh", 20.4300, 106.1700, "user"),        # TP Nam Định (đồng bằng, rìa bão)
]
EXTRA_MOUNTAIN_SITES = [
    ("SaPa", 22.3364, 103.8438, "user"),           # Sa Pa, Lào Cai
    # Đà Lạt là "user": đây là TP trên cao nguyên, không có hạ tầng nối ra ngoài
    # mà mạng đang mô hình hoá; để gateway thì luồng khẩn cấp chọn đích vào đây.
    ("DaLat", 11.9404, 108.4583, "user"),          # Đà Lạt, Lâm Đồng
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
    # HAPS-BienDong KHÔNG phủ nút nào — đây là sự thật đã đo, không phải lỗi.
    # Ngưỡng góc nâng 10° của luật HAPS_GROUND giới hạn tầm phủ thật ở ~108 km
    # (KHÔNG phải 200 km của max_range_km). Hoàng Sa cách 274 km, Trường Sa còn
    # xa hơn 900 km, nên không một vị trí HAPS đơn lẻ nào phủ được cả hai. Giữ
    # nút này làm trung kế Biển Đông (nối HAPS/HAPS và HAPS/UAV với các HAPS ven
    # bờ). Nếu cần phủ đảo thì đặt HAPS riêng cách đảo < 108 km, hoặc dùng UAV
    # (đã làm: UAV-HoangSa, UAV-TruongSa).
    ("HAPS-BienDong", 14.5, 111.5),    # trung kế Biển Đông (không phủ nút mặt đất)
    ("HAPS-BacTrungBo", 17.4, 106.9),  # phủ Quảng Bình / Quảng Trị
]

# Vị trí đặt UAV TÙY CHỌN (đứng yên thay vì bay vòng). (tên, lat, lon)
# Dùng khi cần UAV làm relay cố định tại một điểm, ví dụ phủ đảo.
#
# MỌI UAV đặt cố định luôn bay ở UAV_DAT_ALT_M. Đừng hạ xuống 2000 m: xem chú
# thích của hằng số đó — biên góc ngẩng của cặp Hải Phòng/Quảng Ninh rất mỏng.
EXTRA_UAV_SITES = [
    ("UAV-HoangSa", 16.8300, 112.3300),   # đảo Phú Lâm (khớp toạ độ nút HoangSa)
    ("UAV-TruongSa", 8.6400, 111.9200),
    ("UAV-HaiPhong", 20.8449, 106.6881),   # trên TP Hải Phòng (cách HaiPhong 0 km)
    # Bạch Long Vĩ: trong bão Yagi đây là nơi giật mạnh NHẤT (58.7 m/s, mưa 246 mm)
    # mà trước đó không có hạ tầng nào ngoài vệ tinh. Đặt UAV đúng trên đảo thì
    # khoảng cách 2 km / góc nâng 90° — dư biên rất lớn, không phải ca sát ngưỡng.
    ("UAV-BachLongVi", 20.1333, 107.7242),
    # Cân bằng giữa Hải Phòng và Quảng Ninh (tâm bão thứ hai). Toạ độ 20.87/106.89
    # là gần tối ưu cho biên góc ngẩng: ở UAV_DAT_ALT_M cho 6.63° và 6.49°
    # (biên +1.63°/+1.49° so với ngưỡng 5°). Không dịch toạ độ thêm được nữa —
    # quét lưới ±0.12° cho thấy biên đối xứng tối đa chỉ +0.29° ở alt 2000 m.
    ("UAV-QuangYen", 20.8700, 106.8900),
    # --- UAV cứu các nút từng mất hết liên kết 60% thời gian ---
    # Trước đây 7 nút này chỉ có đường vệ tinh; khi vệ tinh khuất thì cô lập hoàn
    # toàn (LySon 63.2%, BuonMaThuot 62.8%, DaLat 62.0%, CaMau 60.2%, ConDao
    # 59.8%, PhuQuoc 59.7%, SaPa 29.2% số snapshot). Đặt UAV ngay trên nút:
    # khoảng cách ~2.5 km, góc nâng 90°, biên +85° so với ngưỡng 5° — không phải
    # ca sát ngưỡng, không thể hỏng khi dịch toạ độ.
    ("UAV-LySon", 15.3800, 109.1170),          # Lý Sơn, Quảng Ngãi
    ("UAV-BuonMaThuot", 12.6667, 108.0500),    # Buôn Ma Thuột, Đắk Lắk
    ("UAV-DaLat", 11.9404, 108.4583),          # Đà Lạt, Lâm Đồng
    ("UAV-CaMau", 9.1769, 105.1500),           # Cà Mau
    ("UAV-ConDao", 8.6833, 106.6167),          # Côn Đảo
    ("UAV-PhuQuoc", 10.2289, 103.9640),        # Phú Quốc
    ("UAV-SaPa", 22.3364, 103.8438),           # Sa Pa, Lào Cai
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


def cac_diem_thoi_tiet(mo_rong: bool = False, chi_diem_cau_hinh: bool = False):
    """Danh sách điểm (tên, lat, lon) có cột thời tiết, ĐÚNG THỨ TỰ cột trong tệp.

    mo_rong=False: 23 điểm cũ (gateway, user, HAPS cũ, tâm UAV tuần tra).
    mo_rong=True : nối thêm ground/HAPS/UAV đặt cố định vào CUỐI.

    chi_diem_cau_hinh=True: bỏ 5 tâm UAV tuần tra — mạng chỉ chứa UAV đặt cố
    định (`UAV-1`..`UAV-5` là tâm vòng bay, không phải nút). Dùng để đối chiếu
    tập NÚT THẬT với tệp thời tiết. Đừng dùng cờ này để sinh tệp: 5 tâm vẫn cần
    cột để UAV tuần tra có mưa/gió.
    """
    cap = len(DEFAULT_UAV_CENTERS) if chi_diem_cau_hinh else 0
    diem = [(n, la, lo) for n, la, lo in DEFAULT_GATEWAYS + DEFAULT_USERS]
    diem += [(n, la, lo) for n, la, lo in DEFAULT_HAPS]
    diem += [(n, la, lo) for n, la, lo in DEFAULT_UAV_CENTERS[:len(DEFAULT_UAV_CENTERS) - cap]]
    if mo_rong:
        diem += [(n, la, lo) for n, la, lo, _ in EXTRA_GROUND_SITES]
        diem += [(n, la, lo) for n, la, lo in EXTRA_HAPS]
        diem += [(n, la, lo) for n, la, lo in EXTRA_UAV_SITES]
    return diem


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

