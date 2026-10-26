"""
test_chom_walker.py — khoá các bất biến của tệp TLE chòm Walker sinh bởi
`dataset/sinh_tle_walker.py`.

Vì sao cần test riêng: tệp TLE sinh bằng tay rất dễ sai ở CỘT (TLE là định
dạng cột cố định) và sai ở CHECKSUM. Cả hai lỗi đều KHÔNG báo: `skyfield`/
`sgp4` đọc sai cột ra một quỹ đạo khác nhưng vẫn propagate trơn tru, và bỏ
checksum thì `skyfield` vẫn chạy. Hậu quả chỉ lộ ra khi đem kết quả đi so với
"Walker chuẩn" — lúc đó đã muộn.

Test đọc LẠI tệp đã ghi (không dùng lại `make_tle`) rồi dựng công thức Walker
viết ĐỘC LẬP, để lỗi trong `make_tle` không tự đồng ý với chính nó.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.physics.tle_loader import timescale  # noqa: E402

TLE = ROOT / "data" / "tle" / "walker256.txt"
T, P, F = 256, 16, 1
INC_DEG, CAO_KM = 53.0, 550.0
MU, RE = 398600.4418, 6378.137

# Sai lệch cho phép giữa SGP4 và công thức Walker trung bình, ĐO TRONG HỆ TEME
# (chính là hệ của phần tử TLE). Đo được: 0.066° trung bình / 0.099° tối đa
# (~12 km) — do SGP4 coi phần tử TLE là giá trị TRUNG BÌNH rồi cộng dao động
# short-period của J2. Ngưỡng rộng hơn ~3 lần, đủ bắt lỗi nhầm cột (sai hàng
# chục độ) mà không đỏ vì khác phiên bản thư viện.
#
# TẠI SAO PHẢI LÀ TEME: nếu đo trong GCRS (skyfield `.position`) sẽ ra
# 0.30° TB / 0.45° max, gấp ~4.5 lần — phần dôi ra là phép quay
# precession/nutation giữa TEME và GCRS, KHÔNG phải sai số của tệp. Đã xác
# nhận |r| của hai bên bằng nhau tới 1e-11 km nên không phải lỗi quỹ đạo.
SAI_GOC_TOI_DA_DEG = 0.30



def _doc_tle():
    """Đọc tệp TLE thành list (tên, dòng 1, dòng 2) — tách khỏi `make_tle`."""
    if not TLE.exists():
        return None
    dong = [x.rstrip("\n") for x in TLE.read_text(encoding="utf-8").splitlines() if x.strip()]
    assert len(dong) % 3 == 0, f"tệp TLE phải là bội của 3 dòng, đang {len(dong)}"
    return [(dong[k], dong[k + 1], dong[k + 2]) for k in range(0, len(dong), 3)]


def _checksum(dong: str) -> str:
    """Checksum chuẩn TLE, viết lại độc lập với `sinh_tle_walker._checksum`."""
    tong = sum(int(c) if c.isdigit() else 1 for c in dong[:68] if c.isdigit() or c == "-")
    return str(tong % 10)


def _huong_walker(p: int, s: int):
    """Hướng đơn vị của vệ tinh (mặt phẳng p, slot s) theo công thức Walker.

    Suy từ hai phép quay r = R_z(Ω)·R_x(i)·[a cos u, a sin u, 0], quỹ đạo tròn.
    """
    i = np.radians(INC_DEG)
    O = np.radians(p * 360.0 / P)
    u = np.radians((s * 360.0 / (T // P) + p * F * 360.0 / T) % 360.0)
    return np.array([
        np.cos(O) * np.cos(u) - np.sin(O) * np.sin(u) * np.cos(i),
        np.sin(O) * np.cos(u) + np.cos(O) * np.sin(u) * np.cos(i),
        np.sin(u) * np.sin(i)])


def test_tep_walker_ton_tai_va_du_so_ve_tinh():
    bo = _doc_tle()
    if bo is None:
        raise RuntimeError(f"thiếu {TLE} — chạy: python dataset/sinh_tle_walker.py")
    assert len(bo) == T, f"phải có {T} vệ tinh, đang {len(bo)}"
    ten = [x[0] for x in bo]
    assert len(set(ten)) == len(ten), "tên vệ tinh bị trùng"
    # Tên phải mã hoá được (mặt phẳng, slot) vì test dưới giải mã từ tên.
    for t in ten:
        phan = t.split("-")
        assert len(phan) == 3, f"tên '{t}' sai dạng TEN-PP-SS"
        assert 0 <= int(phan[1]) < P, f"'{t}': số mặt phẳng ngoài [0,{P})"
        assert 0 <= int(phan[2]) < T // P, f"'{t}': slot ngoài [0,{T//P})"


def test_moi_dong_tle_dung_69_ky_tu_va_checksum_dung():
    """TLE là định dạng CỘT CỐ ĐỊNH: lệch một ký tự là đọc sai phần tử."""
    bo = _doc_tle()
    if bo is None:
        raise RuntimeError(f"thiếu {TLE}")
    for ten, d1, d2 in bo:
        for nhan, d in (("1", d1), ("2", d2)):
            assert len(d) == 69, f"{ten} dòng {nhan}: dài {len(d)} ký tự, phải đúng 69"
            assert d.startswith(f"{nhan} "), f"{ten} dòng {nhan}: phải bắt đầu '{nhan} '"
            assert _checksum(d) == d[68], \
                f"{ten} dòng {nhan}: checksum phải là {_checksum(d)}, đang {d[68]!r}"
        # Số hiệu vệ tinh ở hai dòng phải giống nhau.
        assert d1[2:7] == d2[2:7], f"{ten}: số hiệu vệ tinh hai dòng khác nhau"
    print(f"    {len(bo)} vệ tinh: 69 ký tự/dòng, checksum đúng toàn bộ")


def test_phan_tu_giong_nhau_tren_toan_chom():
    """Một vỏ quỹ đạo: mọi vệ tinh phải cùng độ nghiêng, e = 0, cùng n."""
    bo = _doc_tle()
    if bo is None:
        raise RuntimeError(f"thiếu {TLE}")
    nghieng = {float(d2[8:16]) for _, _, d2 in bo}
    lech_tam = {d2[26:33] for _, _, d2 in bo}
    n_vong = {float(d2[52:63]) for _, _, d2 in bo}
    assert nghieng == {INC_DEG}, f"độ nghiêng phải đồng nhất {INC_DEG}°, đang {sorted(nghieng)}"
    assert lech_tam == {"0000000"}, f"quỹ đạo tròn phải có e=0, đang {sorted(lech_tam)}"
    assert len(n_vong) == 1, f"mean motion phải đồng nhất, đang {sorted(n_vong)}"
    # mean motion phải đúng theo định luật Kepler cho độ cao danh nghĩa.
    a = RE + CAO_KM
    n_ky_vong = 86400.0 / (2 * np.pi * (a ** 3 / MU) ** 0.5)
    n_trong_tep = n_vong.pop()
    assert abs(n_trong_tep - n_ky_vong) < 1e-6, \
        f"mean motion lệch Kepler: {n_trong_tep} vs {n_ky_vong:.8f}"


def test_raan_va_mean_anomaly_dung_cong_thuc_walker():
    """RAAN phải chia đều P mặt phẳng; mean anomaly theo đúng T/P/F.

    Đây là phần dễ sai nhất khi viết tay dòng 2: nhầm cột 18-25 (RAAN) với
    27-33 (độ lệch tâm) hoặc quên số hạng dịch pha F.
    """
    bo = _doc_tle()
    if bo is None:
        raise RuntimeError(f"thiếu {TLE}")
    mp = T // P
    for ten, _, d2 in bo:
        p, s = int(ten.split("-")[1]), int(ten.split("-")[2])
        raan = float(d2[17:25])
        ma = float(d2[43:51])
        raan_lt = (p * 360.0 / P) % 360.0
        ma_lt = (s * 360.0 / mp + p * F * 360.0 / T) % 360.0
        assert abs(raan - raan_lt) < 1e-3, f"{ten}: RAAN {raan} != {raan_lt:.4f}"
        assert abs(ma - ma_lt) < 1e-3, f"{ten}: mean anomaly {ma} != {ma_lt:.4f}"
    print(f"    RAAN chia đúng {P} mặt phẳng, mean anomaly đúng dịch pha F={F}")


def test_sgp4_cho_vi_tri_khop_cong_thuc_walker():
    """Propagate bằng SGP4 rồi so HƯỚNG với công thức Walker độc lập, TRONG HỆ TEME.

    Phải đo trong TEME vì đó là hệ quy chiếu của chính phần tử TLE. Đo trong
    GCRS (skyfield `.position`) cộng thêm phép quay precession/nutation
    (~0.23°) — đã kiểm: đó KHÔNG phải sai số tệp.

    Không đòi khớp tuyệt đối: SGP4 coi phần tử TLE là giá trị TRUNG BÌNH nên
    vị trí tức thời lệch ~0.07° (đo được). Ngưỡng 0.30° = 3 lần giá trị đo —
    đủ chặt để bắt lỗi nhầm cột (sai hàng chục độ) mà không đỏ oan.

    Bẫy đã sập khi viết test này: lần đầu tôi so `skyfield .position` (GCRS)
    với công thức và thấy lệch 54 km rồi KẾT LUẬN SAI là "tệp hỏng". Thực chất
    54 km = 0.45°, trong đó 0.23° chỉ là chuyển hệ quy chiếu. Bài học: so
    vector phải cùng hệ, và đừng suy ra "hỏng" từ một phép đo chưa hiệu chuẩn.
    """
    from sgp4.api import Satrec, WGS72
    from skyfield.api import load

    if not TLE.exists():
        raise RuntimeError(f"thiếu {TLE}")
    ts = timescale()
    sats = load.tle_file(str(TLE), ts=ts)
    assert len(sats) == T, f"skyfield đọc được {len(sats)}/{T} vệ tinh"
    # Tên phải đọc đúng — nếu None thì mọi thứ gộp về một khóa dict duy nhất.
    assert all(s.name for s in sats), "skyfield không đọc được tên vệ tinh (None)"
    assert len({s.name for s in sats}) == T, "tên vệ tinh không duy nhất"

    bo = _doc_tle()
    a_do = None
    goc = []
    for k, sat in enumerate(sats):
        p, s = int(sat.name.split("-")[1]), int(sat.name.split("-")[2])
        t = sat.epoch
        # LƯU Ý: trên sgp4 >= 2.0, `jdsatepoch` ĐÃ chứa phần lẻ; cộng thêm
        # `jdsatepochF` nữa là trễ 2×fraction (đo được 43 km). Cách skyfield
        # dùng — tách jd nguyên và fraction tính từ tai_fraction — là đúng.
        rec = Satrec.twoline2rv(bo[k][1], bo[k][2], WGS72)
        frac = t.tai_fraction - t._leap_seconds() / 86400.0
        e, r, _ = rec.sgp4(t.whole, frac)
        assert e == 0, f"{sat.name}: SGP4 báo lỗi {e}"
        r = np.array(r)
        a_do = np.linalg.norm(r) if a_do is None else a_do
        c = float(np.dot(r / np.linalg.norm(r), _huong_walker(p, s)))
        goc.append(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))

    goc = np.array(goc)
    print(f"    góc lệch SGP4(TEME) vs công thức: TB {goc.mean():.4f}°, "
          f"max {goc.max():.4f}° ({np.radians(goc.max())*a_do:.1f} km)")
    assert goc.max() < SAI_GOC_TOI_DA_DEG, (
        f"vệ tinh lệch tới {goc.max():.3f}° so với công thức Walker "
        f"(ngưỡng {SAI_GOC_TOI_DA_DEG}°) — nhiều khả năng nhầm CỘT trong dòng 2")

    # Bán kính quỹ đạo phải đúng theo độ cao danh nghĩa (J2 làm phồng ±13 km).
    assert abs(a_do - (RE + CAO_KM)) < 20.0, \
        f"bán kính {a_do:.1f} km lệch quá 20 km so với độ cao danh nghĩa {CAO_KM}"



def test_dataset_v4_dung_dung_chom_walker():
    """`dataset/oracle_dataset_v4.npz` phải là bản dùng chòm Walker, KHÔNG phải Starlink.

    Vì sao cần: v3 và v4 giống nhau ở mọi thứ trừ tên vệ tinh. Chỉ cần quên
    `--tle data/tle/walker256.txt` là v4 lặng lẽ trở thành "v3 với 256 vệ tinh
    Starlink thật" — tên tệp vẫn nói Walker, kết quả so sánh sau đó vô nghĩa.

    Khoá: đủ 256 nút tên `SYN-PP-SS`, đúng (mặt phẳng, slot) có trong tệp TLE,
    không lẫn một tên Starlink nào, và số nút tổng đúng 330 (= 256 vệ tinh +
    42 air + 32 ground); ma trận W phải là (S, 330, 330) khớp `node_order`.
    """
    import numpy as np

    V4 = ROOT / "dataset" / "oracle_dataset_v4.npz"
    if not V4.exists():
        raise RuntimeError(
            f"thiếu {V4} — chạy:\n"
            "  python dataset/sinh_tle_walker.py\n"
            "  python dataset/oracle_labeler_sagsin.py --mo-rong --sats 256 "
            "--tle data/tle/walker256.txt --out dataset/oracle_dataset_v4.npz")
    bo = _doc_tle()
    if bo is None:
        raise RuntimeError(f"thiếu {TLE} — chạy: python dataset/sinh_tle_walker.py")

    d = np.load(V4, allow_pickle=True)
    order = [str(x) for x in d["node_order"]]
    layer = [str(x) for x in d["node_layer"]]

    # 1) Đúng số nút: 256 vệ tinh + phần air/ground của cấu hình mở rộng.
    assert len(order) == 330, f"v4 phải có 330 nút (256 vệ tinh + 42 + 32), đang {len(order)}"
    assert layer.count("space") == T, f"phải có {T} nút space, đang {layer.count('space')}"
    assert len(set(order)) == len(order), "node_order có tên trùng"

    # 2) Mọi tên vệ tinh phải là tên sinh bởi `sinh_tle_walker` và KHÔNG có tên lạ
    #    (một `STARLINK-xxx` lọt vào là dấu hiệu quên `--tle`).
    space = [n for n, l in zip(order, layer) if l == "space"]
    la = [n for n in space if not n.startswith("SYN-")]
    assert not la, (f"{len(la)} nút space KHÔNG phải vệ tinh Walker (vd {la[:3]}) — "
                    f"nhiều khả năng quên `--tle data/tle/walker256.txt`")
    assert set(space) == {t[0] for t in bo}, \
        "tập nút space không khớp danh sách vệ tinh trong tệp TLE"

    # 3) Mọi nút không phải vệ tinh đều có cột thời tiết (74 trạm).
    tt = ROOT / "data" / "thoi_tiet_vn_mr.npz"
    if tt.exists():
        ten_tt = {str(x) for x in np.load(tt, allow_pickle=False)["site_names"]}
        ngoai = [n for n, l in zip(order, layer) if l != "space" and n not in ten_tt]
        assert not ngoai, f"{len(ngoai)} nút không có cột thời tiết: {ngoai[:5]}"

    # 4) Hình dạng phải tự khớp, không chỉ "đúng số".
    W = d["W_snapshots"]
    assert W.shape[1:] == (len(order), len(order)), \
        f"W_snapshots {W.shape} không khớp {len(order)} nút"
    assert d["node_latlon_alt"].shape[1] == len(order), "node_latlon_alt sai số nút"
    print(f"    v4: {len(order)} nút, {layer.count('space')} vệ tinh Walker, "
          f"W={W.shape}")


if __name__ == "__main__":
    for _fn in (test_tep_walker_ton_tai_va_du_so_ve_tinh,
                test_moi_dong_tle_dung_69_ky_tu_va_checksum_dung,
                test_phan_tu_giong_nhau_tren_toan_chom,
                test_raan_va_mean_anomaly_dung_cong_thuc_walker,
                test_sgp4_cho_vi_tri_khop_cong_thuc_walker,
                test_dataset_v4_dung_dung_chom_walker):
        print(_fn.__name__)
        _fn()
    print("OK")


