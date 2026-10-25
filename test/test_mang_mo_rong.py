"""
test_mang_mo_rong.py — Hợp đồng "chạy lại cấu hình cũ phải ra đúng mạng 73 nút".

Khi bật --mo-rong, nút mới PHẢI nối vào CUỐI node_order. Nếu chỉ nối vào cuối
từng khối (space/air/ground) thì khối air phình 10 -> 14 bậc, đẩy toàn bộ khối
ground lệch 4 chỉ số => model huấn luyện trên mạng 73 nút không còn dùng được.

Chạy: python test/test_mang_mo_rong.py
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.physics.air_ground import make_placed_uavs, make_uav_patrols, nodes_at, static_nodes  # noqa: E402
from src.physics.sites import (  # noqa: E402
    CAC_TEN_CU,
    DEFAULT_GATEWAYS,
    DEFAULT_HAPS,
    DEFAULT_USERS,
    DEFAULT_UAV_CENTERS,
    EXTRA_GROUND_SITES,
    EXTRA_UAV_SITES,
    khoa_sap_xep,
)
from test.tien_ich_test import BoQua, chay_tests, dung_utf8  # noqa: E402

dung_utf8()

V2 = ROOT / "dataset" / "oracle_dataset_v2.npz"
V3 = ROOT / "dataset" / "oracle_dataset_v3.npz"


def _mang(mo_rong):
    """(air_names, ground_names) đúng như oracle_labeler_sagsin dựng."""
    st = static_nodes(mo_rong=mo_rong)
    pt = make_uav_patrols(seed=42) + (make_placed_uavs() if mo_rong else [])
    n0 = nodes_at(0.0, st, pt)
    air = sorted((n.name for n in n0 if n.layer == "air"), key=khoa_sap_xep)
    gnd = sorted((n.name for n in n0 if n.layer == "ground"), key=khoa_sap_xep)
    return air, gnd


def test_cau_hinh_cu_van_ra_73_nut():
    """Không bật mở rộng => đúng 73 nút, phân bố 50/5/5/13."""
    air, gnd = _mang(False)
    assert len(air) == len(DEFAULT_HAPS) + len(DEFAULT_UAV_CENTERS) == 10
    assert len(gnd) == len(DEFAULT_GATEWAYS) + len(DEFAULT_USERS) == 13
    assert 50 + len(air) + len(gnd) == 73


def test_mo_rong_them_nut_va_giu_nguyen_khoi_cu():
    """Bật mở rộng: +18 nút, và 73 nút cũ giữ nguyên thứ tự tuyệt đối."""
    a0, g0 = _mang(False)
    a1, g1 = _mang(True)
    assert len(g1) - len(g0) == len(EXTRA_GROUND_SITES) == 14   # 11 cũ + 3 ven biển Bắc
    assert len(a1) - len(a0) == 2 + 5                 # 2 HAPS + 5 UAV đặt cố định
    old_air = [n for n in a1 if n in set(a0)]
    old_gnd = [n for n in g1 if n in set(g0)]
    assert old_air == a0 and old_gnd == g0, "thứ tự nút cũ bị xáo trộn"
    assert (a1[:len(a0)], g1[:len(g0)]) == (a0, g0), "nút mới phải nối vào CUỐI khối"


def test_moi_nut_moi_deu_duoc_khai_bao_ngoai_CAC_TEN_CU():
    """Nút mới phải được khai báo, không lọt vào nhóm 'nút cũ'."""
    air, gnd = _mang(True)
    for n in air + gnd:
        assert (n in CAC_TEN_CU) == (khoa_sap_xep(n)[0] == 0), n
    moi = [n for n in air + gnd if khoa_sap_xep(n)[0] == 1]
    assert len(moi) == len(EXTRA_GROUND_SITES) + 2 + len(EXTRA_UAV_SITES) == 21


def test_v3_giu_nguyen_chi_so_cua_73_nut_trong_v2():
    """Tệp v3 thật: mọi nút cũ phải ở ĐÚNG chỉ số như v2."""
    if not (V2.exists() and V3.exists()):
        raise BoQua("chưa có oracle_dataset_v2/v3.npz")
    a = np.load(V2, allow_pickle=True)
    b = np.load(V3, allow_pickle=True)
    o2 = [str(x) for x in a["node_order"]]
    o3 = [str(x) for x in b["node_order"]]
    assert len(o2) == 73
    # 73 nút cũ + 14 ground + 2 HAPS + 5 UAV đặt cố định; 50 vệ tinh thêm ở tầng space
    assert len(o3) == 73 + len(EXTRA_GROUND_SITES) + 2 + len(EXTRA_UAV_SITES)
    lech = [(i, n, o3.index(n)) for i, n in enumerate(o2) if o3.index(n) != i]
    assert not lech, f"nút cũ bị lệch chỉ số: {lech[:5]}"


def test_v3_giu_nguyen_W_cua_cac_nut_cu():
    """Ma trận W giữa các nút cũ phải giống hệt v2."""
    if not (V2.exists() and V3.exists()):
        raise BoQua("chưa có oracle_dataset_v2/v3.npz")
    a = np.load(V2, allow_pickle=True)
    b = np.load(V3, allow_pickle=True)
    o2 = [str(x) for x in a["node_order"]]
    o3 = [str(x) for x in b["node_order"]]
    idx = np.array([o3.index(n) for n in o2])
    Wb = b["W_snapshots"][:, idx][:, :, idx]
    assert np.allclose(a["W_snapshots"], Wb)


def test_v3_co_toa_do_va_thong_tin_lien_ket():
    """v3 phải lưu tọa độ từng nút và thông tin liên kết (yêu cầu đề bài)."""
    if not V3.exists():
        raise BoQua("chưa có oracle_dataset_v3.npz")
    b = np.load(V3, allow_pickle=True)
    for k in ("node_latlon_alt", "link_stats", "link_stat_cols", "mo_rong"):
        assert k in b.files, f"thiếu khóa {k}"
    assert bool(b["mo_rong"][0])
    lat = b["node_latlon_alt"]
    n_node = len(b["node_order"])
    assert lat.shape == (600, n_node, 3)
    assert n_node == 94, f"v3 phải có 94 nút (73 cũ + 16 ground/HAPS + 5 UAV), đang {n_node}"
    assert np.isfinite(lat[:, :, :2]).all(), "tọa độ lat/lon không được NaN"
    cot = [str(x) for x in b["link_stat_cols"]]
    assert cot == ["snapshot_id", "u", "v", "distance_km", "elevation_deg",
                   "bandwidth_mbps", "packet_loss", "exists"]
    assert b["link_stats"].shape[1] == 8 and len(b["link_stats"]) > 0


def test_v3_co_hoang_sa_truong_sa_va_uav_dat_co_dinh():
    """Hai quần đảo phải có mặt, kèm UAV đặt cố định phủ chúng."""
    if not V3.exists():
        raise BoQua("chưa có oracle_dataset_v3.npz")
    b = np.load(V3, allow_pickle=True)
    o = [str(x) for x in b["node_order"]]
    for n in ("HoangSa", "TruongSa", "UAV-HoangSa", "UAV-TruongSa"):
        assert n in o, f"thiếu nút {n}"


def test_v3_split_khong_ro_ri_link_stats():
    """link_stats của từng tập phải lọc theo snapshot của tập và đánh lại cột 0.

    Trước đây bảng bị chép nguyên 253.856 dòng vào cả 3 tập, nên tra bằng chỉ số
    cục bộ sẽ lấy sai snapshot (và lộ thông tin liên kết của tập khác).
    """
    import json
    d = ROOT / "dataset" / "v3"
    if not (d / "split_info.json").exists():
        raise BoQua("chưa có dataset/v3/split_info.json")
    info = json.loads((d / "split_info.json").read_text(encoding="utf-8"))
    for ten, snap_ids in info["snapshot_ids"].items():
        f = np.load(d / f"{ten}.npz", allow_pickle=True)
        ls, W = f["link_stats"], f["W_snapshots"]
        assert len(ls) < 253856, f"{ten}: link_stats vẫn là cả bảng gốc ({len(ls)} dòng)"
        # cột 0 phải là chỉ số cục bộ của tập này
        assert ls[:, 0].min() >= 0 and ls[:, 0].max() < W.shape[0], \
            f"{ten}: snapshot_id trong link_stats vượt {W.shape[0]} snapshot của tập"
        # số dòng phải bằng số liên kết dương của W_snapshots (bỏ đường chéo)
        so_canh = int((W >= 0).sum()) - W.shape[0] * W.shape[1]
        assert len(ls) == so_canh, f"{ten}: {len(ls)} dòng link_stats != {so_canh} cạnh trong W"
        # mỗi snapshot đều có mặt đúng số liên kết của nó
        dem = np.bincount(ls[:, 0].astype(np.int64), minlength=W.shape[0])
        assert (dem > 0).all(), f"{ten}: có snapshot không có dòng link_stats nào"
        assert abs(int(dem.sum()) - len(ls)) == 0


def test_uav_ven_bien_bac_phu_duoc_hai_phong_va_quang_ninh():
    """UAV đặt cố định phải thật sự nối được, không chỉ 'gần'.

    Bẫy đã sập một lần: đặt UAV-QuangYen cách Quảng Ninh 24.1 km (< 25 km) nhưng
    góc nâng chỉ 4.65° < ngưỡng 5° của luật UAV_GROUND, nên cạnh KHÔNG được tạo
    và nút Quảng Ninh cô lập suốt 600/600 snapshot mà không báo lỗi.
    Test này khoá cả hai điều kiện: khoảng cách và góc nâng.
    """
    from src.physics.air_ground import (
        _UAV_GROUND,
        Node,
        blocked_by_earth,
        elevation_deg,
        node_ecef_km,
    )
    uav = make_placed_uavs()
    for nut_ten, la, lo, kind in [("HaiPhong", 20.8449, 106.6881, "gateway"),
                                  ("QuangNinh", 20.9500, 107.0800, "user")]:
        nut = Node(nut_ten, "ground", kind, la, lo, 0.0,
                   can_relay=(kind == "gateway"), endpoint=True)
        # chọn UAV gần nhất
        gan = min(uav, key=lambda u: (node_ecef_km(u.node_at(0.0))
                                      - node_ecef_km(nut)).dot(
                                          node_ecef_km(u.node_at(0.0)) - node_ecef_km(nut)))
        un = gan.node_at(0.0)
        e_n, e_u = node_ecef_km(nut), node_ecef_km(un)
        dist = float(np.linalg.norm(e_u - e_n))
        elev = elevation_deg(nut, e_u)
        assert dist <= _UAV_GROUND.max_range_km, \
            f"{nut_ten}: UAV gần nhất ({gan.name}) cách {dist:.1f} km > {_UAV_GROUND.max_range_km} km"
        assert elev >= _UAV_GROUND.min_elev_deg, \
            f"{nut_ten}: góc nâng {elev:.2f}° < {_UAV_GROUND.min_elev_deg}° (cạnh sẽ không được tạo)"
        assert not blocked_by_earth(e_u, e_n), f"{nut_ten}: bị Trái Đất chắn"


def _run():
    return chay_tests(globals(), "test_mang_mo_rong")


if __name__ == "__main__":
    sys.exit(0 if _run() else 1)
