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
    """Bật mở rộng: thêm nút, và 73 nút cũ giữ nguyên thứ tự tuyệt đối."""
    a0, g0 = _mang(False)
    a1, g1 = _mang(True)
    assert len(g1) - len(g0) == len(EXTRA_GROUND_SITES)
    assert len(a1) - len(a0) == 2 + len(EXTRA_UAV_SITES)   # 2 HAPS + UAV đặt cố định
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
    assert len(moi) == len(EXTRA_GROUND_SITES) + 2 + len(EXTRA_UAV_SITES)


def test_v3_giu_nguyen_chi_so_cua_73_nut_trong_v2():
    """Tệp v3 thật: mọi nút cũ phải ở ĐÚNG chỉ số như v2."""
    if not (V2.exists() and V3.exists()):
        raise BoQua("chưa có oracle_dataset_v2/v3.npz")
    a = np.load(V2, allow_pickle=True)
    b = np.load(V3, allow_pickle=True)
    o2 = [str(x) for x in a["node_order"]]
    o3 = [str(x) for x in b["node_order"]]
    assert len(o2) == 73
    # 73 nút cũ + ground mở rộng + 2 HAPS + UAV đặt cố định; vệ tinh thêm ở tầng space
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
    # 73 nút cũ + ground mở rộng + 2 HAPS + UAV đặt cố định
    kyen_vong = 73 + len(EXTRA_GROUND_SITES) + 2 + len(EXTRA_UAV_SITES)
    assert n_node == kyen_vong, \
        f"v3 phải có {kyen_vong} nút (73 cũ + ground + HAPS + UAV), đang {n_node}"
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


def test_tep_thoi_tiet_mo_rong_khop_tep_23_diem():
    """Tệp thời tiết mở rộng phải chứa ĐÚNG dữ liệu của tệp 23 điểm.

    Bẫy đã sập một lần: sinh lại `thoi_tiet_vn_mr.npz` bằng lệnh `tai --mo-rong`
    mà QUÊN chạy `them-giong`. Hậu quả không báo lỗi:
      - mất hẳn `cape` / `lifted_index` / `code_fc`  -> không khí quyển nào có giông;
      - cột `code` chỉ còn mã ERA5, số giờ giông về 0 (tệp đúng có 8 343 ô-giờ);
      - UAV không bao giờ hạ cánh vì giông, sét không bao giờ ngắt nút mặt đất;
      - mọi số liệu trên mạng mở rộng thành KHÔNG SO SÁNH ĐƯỢC với kết quả cũ.

    Test so theo TÊN điểm (không theo chỉ số cột), vì 3 trạm ven biển mới được
    chèn vào giữa danh sách chứ không nối vào cuối.
    """
    import numpy as np
    from src.ngau_nhien.thoi_tiet import GIONG

    mr = ROOT / "data" / "thoi_tiet_vn_mr.npz"
    goc = ROOT / "data" / "thoi_tiet_vn.npz"
    if not mr.exists() or not goc.exists():
        raise BoQua("thiếu tệp thời tiết, bỏ qua")
    a = np.load(goc, allow_pickle=False)
    b = np.load(mr, allow_pickle=False)
    ten_a = [str(x) for x in a["site_names"]]
    ten_b = [str(x) for x in b["site_names"]]
    thieu = [n for n in ten_a if n not in ten_b]
    assert not thieu, f"tệp mở rộng thiếu điểm: {thieu}"
    idx = [ten_b.index(n) for n in ten_a]

    assert np.array_equal(a["time_utc"], b["time_utc"]), "trục thời gian phải khớp tuyệt đối"

    # 1) Bắt buộc có đủ bộ biến khí quyển mà nhánh ERA5 không có (do them-giong sinh).
    #    `code_fc` không có trong tệp 23 điểm cũ nên chỉ kiểm khi tệp gốc có.
    for k in ("cape", "lifted_index"):
        assert k in b.keys(), \
            f"tệp mở rộng THIẾU '{k}': quên chạy `them-giong --vao data/thoi_tiet_vn_mr.npz`"

    # 2) 23 cột cũ phải giống TỪNG BIT (kể cả mã thời tiết) => hồi quy cũ vẫn đúng.
    for k in ("cape", "code", "code_fc", "lifted_index", "precip", "gust"):
        if k not in a.keys():
            continue
        assert np.array_equal(a[k], b[k][:, idx], equal_nan=True), \
            f"cột '{k}' của 23 điểm cũ đã bị thay đổi so với tệp gốc"

    # 3) Phải THẬT SỰ có giông, không chỉ có cột rỗng.
    so_giong = int(np.isin(b["code"][:, idx], GIONG).sum())
    assert so_giong > 0, \
        "số ô-giờ giông trong 23 điểm = 0: 'code' chưa được đặt lại theo luật CAPE"


def test_tep_thoi_tiet_mo_rong_phu_dung_moi_nut():
    """Tệp thời tiết phải có ĐÚNG các điểm mà mạng cần — không thiếu, không thừa.

    Bẫy đã sập một lần: `thoi_tiet_vn_mr.npz` có 44 điểm (còn `UAV-VinhBacBo` đã
    xoá, thiếu 8 UAV mới) trong khi cấu hình cần 51. Hệ quả KHÔNG báo lỗi:
    `MangGoc` gán `node_site = -1`, rồi `kich_ban.sinh_kich_ban()` `continue` qua
    nút đó, nên 8 UAV — trong đó có `UAV-BachLongVi` (đảo giật 58.7 m/s trong
    Yagi, lý do chính để thêm nó) — KHÔNG BAO GIỜ hạ cánh vì giông hay gió, và
    `tao_du_lieu` để đặc trưng thời tiết của chúng bằng 0.

    Test cũ chỉ so 23 cột cũ nên không bắt được. Test này đòi khớp ĐÚNG tập điểm.
    """
    from src.ngau_nhien.thoi_tiet import cac_diem

    mr = ROOT / "data" / "thoi_tiet_vn_mr.npz"
    if not mr.exists():
        raise BoQua("thiếu data/thoi_tiet_vn_mr.npz, bỏ qua")
    with np.load(mr, allow_pickle=False) as d:
        co = [str(x) for x in d["site_names"]]
    can = [n for n, _, _ in cac_diem(mo_rong=True)]
    thieu = [n for n in can if n not in set(co)]
    thua = [n for n in co if n not in set(can)]
    assert not thieu, (
        f"tệp thời tiết thiếu {len(thieu)} điểm mà mạng cần: {thieu}. "
        f"Các nút này sẽ có node_site = -1 (không mưa, không gió, UAV không hạ cánh). "
        f"Chạy lại 'thoi_tiet tai --mo-rong' rồi 'them-giong'.")
    assert not thua, (
        f"tệp thời tiết có {len(thua)} điểm không còn trong cấu hình: {thua}. "
        f"Cột chết này che lỗi thiếu điểm nếu chỉ kiểm một chiều.")
    assert len(can) == 51, f"cấu hình mở rộng phải có 51 điểm, đang {len(can)}"


def test_uav_dat_co_dinh_khop_thoi_tiet_voi_nut_chu():
    """UAV đặt đúng toạ độ một nút thì cột thời tiết phải TRÙNG KHÍT nút đó.

    Đây là bất biến vật lý: cùng một chỗ thì cùng mưa, cùng gió, cùng giông.
    Nếu lệch, tức là toạ độ hai bên đã trôi khỏi nhau (sửa một chỗ quên chỗ kia)
    — và bản đồ/báo cáo sẽ nói UAV ở nơi nó không ở.
    """
    from src.ngau_nhien.thoi_tiet import cac_diem

    mr = ROOT / "data" / "thoi_tiet_vn_mr.npz"
    if not mr.exists():
        raise BoQua("thiếu data/thoi_tiet_vn_mr.npz, bỏ qua")
    toa_do = {n: (la, lo) for n, la, lo in cac_diem(mo_rong=True)}
    cap_uav = [(u, h) for u, h in (
        ("UAV-BachLongVi", "BachLongVi"), ("UAV-LySon", "LySon"),
        ("UAV-BuonMaThuot", "BuonMaThuot"), ("UAV-DaLat", "DaLat"),
        ("UAV-CaMau", "CaMau"), ("UAV-ConDao", "ConDao"),
        ("UAV-PhuQuoc", "PhuQuoc"), ("UAV-SaPa", "SaPa"),
        ("UAV-HoangSa", "HoangSa"), ("UAV-TruongSa", "TruongSa"),
        ("UAV-HaiPhong", "HaiPhong")) if u in toa_do and h in toa_do]

    with np.load(mr, allow_pickle=False) as d:
        ten = [str(x) for x in d["site_names"]]
        bien = [k for k in ("precip", "code", "gust", "cape", "code_fc") if k in d.keys()]
        for u, h in cap_uav:
            (la_u, lo_u), (la_h, lo_h) = toa_do[u], toa_do[h]
            assert abs(la_u - la_h) < 1e-4 and abs(lo_u - lo_h) < 1e-4, \
                f"{u} ({la_u}/{lo_u}) không còn trùng toạ độ {h} ({la_h}/{lo_h})"
            iu, ih = ten.index(u), ten.index(h)
            for k in bien:
                assert np.array_equal(d[k][:, iu], d[k][:, ih], equal_nan=True), \
                    f"cột '{k}' của {u} khác {h} dù cùng toạ độ"


def test_uav_dat_co_dinh_phu_duoc_it_nhat_mot_nut_mat_dat():
    """Mỗi UAV đặt cố định phải phục vụ được ít nhất một nút mặt đất.

    Bẫy đã sập một lần: `UAV-VinhBacBo` đặt ở 20.30/106.80 "giữa vịnh Bắc Bộ".
    Nút gần nhất (Thái Bình) cách 51 km, xa hơn tầm với thật của luật UAV_GROUND
    (~22.4 km như phân tích trong air_ground.UAV_DAT_ALT_M), nên UAV này chỉ nối
    được HAPS và vệ tinh — nó hạ cánh khi có gió nhưng KHÔNG ảnh hưởng đường đi
    của bất kỳ nút nào. Đã dời về đúng Bạch Long Vĩ.
    """
    from src.physics.air_ground import _UAV_GROUND, elevation_deg, node_ecef_km

    st = static_nodes(mo_rong=True)
    mat_dat = [n for n in st if n.layer == "ground"]
    for u in make_placed_uavs():
        un = u.node_at(0.0)
        e_u = node_ecef_km(un)
        phu = [g.name for g in mat_dat
               if float(np.linalg.norm(node_ecef_km(g) - e_u)) <= _UAV_GROUND.max_range_km
               and elevation_deg(g, e_u) >= (_UAV_GROUND.min_elev_deg or 0.0)]
        assert phu, (f"{u.name} ({u.center_lat_deg}, {u.center_lon_deg}) không phục vụ "
                     f"nút mặt đất nào — chỉ nối được HAPS/vệ tinh")


def test_bach_long_vi_co_ha_tang_cuu_nut():
    """Bạch Long Vĩ là nơi giật mạnh nhất trong bão Yagi và từng cô lập 100%.

    Nút này phải có ít nhất một liên kết cục bộ (UAV/HAPS), không chỉ vệ tinh.
    """
    from src.physics.air_ground import compute_local_links

    st = static_nodes(mo_rong=True)
    links = compute_local_links(st + [p.node_at(0.0) for p in make_placed_uavs()])
    co = [l for l in links if l["exists"] and "BachLongVi" in (l["node_a"], l["node_b"])]
    assert co, "BachLongVi không có liên kết cục bộ nào (chỉ còn đường vệ tinh)"
    print("    BachLongVi nối cục bộ với: "
          + ", ".join(sorted({l["node_b"] if l["node_a"] == "BachLongVi" else l["node_a"]
                              for l in co})))


def test_v3_do_cao_ve_tinh_hop_le():
    """`node_latlon_alt` của vệ tinh phải là ĐỘ CAO QUỸ ĐẠO, không phải 0.

    Bẫy đã sập một lần: code ghi `wgs84.subpoint_of(...).elevation.km`, luôn = 0.
    Đó là độ cao ellipsoid tại điểm chiếu dưới chân vệ tinh, KHÔNG phải độ cao
    quỹ đạo, nên cả 50 vệ tinh đều có alt = 0.0 mà không báo lỗi — trong khi
    Starlink thật bay ở ~550 km. Bẫy thứ hai: TLE chứa cả vệ tinh đã rơi; chúng
    cho độ cao ÂM (vd -2855 km) hoặc NaN, làm hỏng mọi tính hình học.
    """
    if not V3.exists():
        raise BoQua("chưa có oracle_dataset_v3.npz")
    b = np.load(V3, allow_pickle=True)
    lop = [str(x) for x in b["node_layer"]]
    alt = b["node_latlon_alt"]
    i_sat = [i for i, l in enumerate(lop) if l == "space"]
    assert i_sat, "v3 không có vệ tinh nào"
    a = alt[:, i_sat, 2].ravel()
    assert np.isfinite(a).all(), f"có {int((~np.isfinite(a)).sum())} độ cao vệ tinh NaN"
    assert (a > 0).all(), f"có {int((a <= 0).sum())} độ cao vệ tinh <= 0 (min {a.min():.1f} km)"
    # Starlink: quỹ đạo thấp ~300-600 km. Cho biên rộng nhưng phải loại được cả 0
    # và các giá trị vô nghĩa, đồng thời bắt được nếu ai đó lại ghi subpoint.
    assert 100.0 < float(np.median(a)) < 2000.0, \
        f"trung vị độ cao vệ tinh {np.median(a):.1f} km không giống quỹ đạo thật"

    # Nút mặt đất: độ cao nhỏ (km, tức vài trăm mét)
    i_gnd = [i for i, l in enumerate(lop) if l == "ground"]
    if i_gnd:
        g = alt[:, i_gnd, 2]
        assert np.isfinite(g).all()
        assert (np.abs(g) < 10.0).all(), "độ cao nút mặt đất bất thường"


def _run():
    return chay_tests(globals(), "test_mang_mo_rong")




def test_mang_goc_chan_tep_thoi_tiet_lech():
    """`MangGoc` phải tự chặn tệp thời tiết lệch — kể cả khi nơi gọi QUÊN bật kiểm.

    Đây là chốt chặn chính vì `MangGoc` mới là nơi gán `node_site = -1`. Trước
    đây việc kiểm nằm rải ở từng script (`tao_du_lieu.main` bật, còn
    `khoang_loi`, `luat_chia_doi`, `phan_tich_nhieu_mua`, `huong_dan/buoc_7`,
    `buoc_8` thì không) ⇒ đường chạy nào quên bật là lỗi âm thầm quay lại.

    Test dựng bản tệp lỗi riêng (bỏ 8 UAV mới) thay vì dùng tệp thật, để không
    ghi đè `data/thoi_tiet_vn_mr.npz`.
    """
    import tempfile
    from src.ngau_nhien.kich_ban import MangGoc
    from src.ngau_nhien.thoi_tiet import ThoiTiet

    mr = ROOT / "data" / "thoi_tiet_vn_mr.npz"
    if not mr.exists():
        raise BoQua("thiếu data/thoi_tiet_vn_mr.npz, bỏ qua")
    d = dict(np.load(mr, allow_pickle=False))
    ten = [str(x) for x in d["site_names"]]
    bo = {"UAV-BachLongVi", "UAV-LySon", "UAV-BuonMaThuot", "UAV-DaLat",
          "UAV-CaMau", "UAV-ConDao", "UAV-PhuQuoc", "UAV-SaPa"}
    giu = [k for k, n in enumerate(ten) if n not in bo]
    assert len(giu) == len(ten) - 8, f"tệp đã khác giả định: {len(ten)} điểm"
    loi = {k: (v[..., giu] if v.ndim == 2 else v[giu])
           for k, v in d.items() if k in ("lat", "lon", "precip", "code", "gust",
                                          "code_fc", "cape", "lifted_index", "code_era5")}
    loi.update(time_utc=d["time_utc"], nguon=d["nguon"],
               site_names=np.array([ten[k] for k in giu]))

    voi = tempfile.TemporaryDirectory()
    p = Path(voi.name) / "lech.npz"
    np.savez(p, **loi)
    try:
        # kiem_tra=False ở ThoiTiet: mô phỏng đúng cảnh nơi gọi KHÔNG bật kiểm.
        tt = ThoiTiet(str(p), kiem_tra=False)
        try:
            MangGoc(np.ones((1, 101, 101)) * 2.0, ["user"] * 101,
                    [f"n{k}" for k in range(101)], tt)
        except ValueError as e:
            assert "UAV-BachLongVi" in str(e), f"lỗi không nêu tên điểm thiếu: {e}"
        else:
            raise AssertionError(
                "MangGoc nhận tệp thiếu 8 điểm mà không báo lỗi: node_site sẽ = -1 "
                "và 8 UAV lặng lẽ không bao giờ hạ cánh.")
    finally:
        voi.cleanup()

    # Tệp đúng thì phải dựng được mạng (chốt chặn không được chặn nhầm).
    tt_ok = ThoiTiet(str(mr))
    MangGoc(np.ones((1, 101, 101)) * 2.0, ["user"] * 101,
            [f"n{k}" for k in range(101)], tt_ok)

if __name__ == "__main__":
    sys.exit(0 if _run() else 1)
