"""Test cho bộ sinh kịch bản ngẫu nhiên. Chạy: python test/test_ngau_nhien.py [đường_dẫn oracle_dataset_v2.npz]"""
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.ngau_nhien import thoi_tiet  # noqa: E402
from src.ngau_nhien.khoang_loi import chi_phi, danh_gia, dijkstra  # noqa: E402
from src.ngau_nhien.kich_ban import ITU838, MangGoc, ThamSo, sinh_kich_ban, suy_hao_mua  # noqa: E402
from src.ngau_nhien.thoi_tiet import ThoiTiet  # noqa: E402

RAW = os.environ.get("PBL4_RAW") or str(ROOT / "dataset" / "oracle_dataset_v2.npz")   # chạy tay: PBL4_RAW=<tệp> python ...
TMP = Path(tempfile.mkdtemp())
TAT = dict(mua=False, giong=False, xung_dot=False, tai=False)


def _thoi_tiet(precip=None, code=None, gust=None, H=48):
    diem = thoi_tiet.cac_diem()
    S = len(diem)
    f = TMP / f"tt_{np.random.randint(1e9)}.npz"
    np.savez(f, site_names=np.array([d[0] for d in diem]), lat=np.array([d[1] for d in diem]),
             lon=np.array([d[2] for d in diem]),
             time_utc=np.array([f"2024-07-01T{h % 24:02d}:00" for h in range(H)]),
             precip=np.full((H, S), 0.0 if precip is None else precip, np.float32),
             code=np.full((H, S), 0 if code is None else code, np.int16),
             gust=np.full((H, S), 3.0 if gust is None else gust, np.float32), nguon=np.array("test"))
    return ThoiTiet(str(f))


_RAW = np.load(RAW)
W0 = _RAW["W_snapshots"][:40]


def mang(tt):
    return MangGoc(W0, _RAW["node_kind"], _RAW["node_order"], tt)


def test_tat_het_thi_trung_du_lieu_goc():
    tt = _thoi_tiet(precip=30.0, code=95, gust=30.0)
    g = mang(tt)
    Wn, _ = sinh_kich_ban(g, tt, 0, 1, ThamSo(**TAT))
    assert np.abs(Wn - W0).max() < 1e-4


def test_suy_hao_mua_don_dieu_va_theo_tan_so():
    Ls, LG = np.full(3, 10.0), np.full(3, 5.0)
    for f in ITU838:
        a = suy_hao_mua(np.array([0.0, 10.0, 50.0]), np.full(3, f), Ls, LG)
        assert a[0] == 0 and a[0] < a[1] < a[2]
    a12 = suy_hao_mua(np.array([20.0]), np.array([12]), Ls[:1], LG[:1])
    a28 = suy_hao_mua(np.array([20.0]), np.array([28]), Ls[:1], LG[:1])
    assert a28 > 3 * a12


def test_mua_to_cat_lien_ket_28GHz_nhung_khong_dung_ISL():
    tt = _thoi_tiet(precip=80.0)
    g = mang(tt)
    Wn, _ = sinh_kich_ban(g, tt, 0, 1, ThamSo(**{**TAT, "mua": True}, sigma_mua=1e-6))
    haps_md = (g.ten == "HAPS-MĐ")[None] & (W0 > 0)
    isl = (g.ten == "ISL")[None] & (W0 > 0)
    assert (Wn[haps_md] < 0).mean() > 0.95
    assert np.abs(Wn[isl] - W0[isl]).max() < 1e-4


def test_mua_nhe_lam_tang_tre_khong_cat():
    tt = _thoi_tiet(precip=1.0)
    g = mang(tt)
    p = ThamSo(**{**TAT, "mua": True}, sigma_mua=1e-6)
    Wn, _ = sinh_kich_ban(g, tt, 0, 1, p)
    m = (g.ten == "HAPS-MĐ")[None] & (W0 > 0)
    assert (Wn[m] > 0).all() and (Wn[m] >= W0[m] - 1e-5).all() and (Wn[m] > W0[m] + 1e-4).any()


def test_giong_ha_canh_UAV():
    tt = _thoi_tiet(code=95)
    g = mang(tt)
    Wn, q = sinh_kich_ban(g, tt, 0, 1, ThamSo(**{**TAT, "giong": True}))
    uav = [i for i, k in enumerate(g.kind) if k == "uav"]
    assert q["tat"][:, uav].all()
    assert (Wn[:, uav][W0[:, uav] > 0] < 0).all()


def test_tai_lam_tang_tre_va_bi_chan_boi_bo_dem():
    tt = _thoi_tiet()
    g = mang(tt)
    p = ThamSo(**{**TAT, "tai": True}, rho_tb=3.0)
    Wn, _ = sinh_kich_ban(g, tt, 0, 1, p)
    co = W0 > 0
    tren = g.prop + 12.0 / g.C[None] * (1 + p.k_dem) + 0.5
    assert (Wn[co] <= tren[co] + 1e-3).all()
    assert (Wn[co] > W0[co]).mean() > 0.99


def test_hau_kiem_khong_te_hon_ngay_tho_va_tat_dinh_thi_quy_dao_toi_uu():
    tt = _thoi_tiet(precip=5.0)
    g = mang(tt)
    can = np.array([k != "user" for k in g.kind])
    tv = []
    for sid, row in zip(_RAW["snapshot_id"], _RAW["y_paths_indices"]):
        if sid in (5, 20):
            p = [int(x) for x in row if x >= 0]
            tv.append((int(sid), p[0], p[-1], p))
    Wn, _ = sinh_kich_ban(g, tt, 0, 3, ThamSo())
    kq = danh_gia(Wn, g.W, tv, [3, 9], [0], can, 100.0)
    for D in (3, 9):
        r = kq[(0, D)]
        assert (r["D_hau_kiem"] <= r["D_ngay_tho"] + 1e-6).all()
        assert (r["D_hau_kiem"] <= r["D_quy_dao"] + 1e-6).all()
    Wd, _ = sinh_kich_ban(g, tt, 0, 3, ThamSo(**TAT))
    kd = danh_gia(Wd, g.W, tv, [9], [0], can, 100.0)[(0, 9)]
    assert np.allclose(kd["D_quy_dao"], kd["D_hau_kiem"])


def test_chi_phi_phat_khi_dut():
    win = np.ones((4, 3, 3), np.float32)
    win[2, 1, 2] = -1
    c, dut = chi_phi(win, [0, 1, 2], 100.0)
    assert dut and abs(c - (2 + 2 + 100 + 2) / 4) < 1e-6
    c, dut = chi_phi(win, [0, 2], 100.0)
    assert not dut and abs(c - 1) < 1e-6


def test_dijkstra_ton_trong_quy_tac_chuyen_tiep():
    W = -np.ones((3, 3))
    W[0, 1] = W[1, 2] = 1.0
    W[0, 2] = 5.0
    assert dijkstra(W, 0, 2, np.array([True, False, True])) == [0, 2]
    assert dijkstra(W, 0, 2, np.array([True, True, True])) == [0, 1, 2]


def test_tai_open_meteo_va_them_giong():
    import requests
    diem = thoi_tiet.cac_diem()

    class R:
        status_code = 200

        def __init__(self, url, params):
            n = (np.datetime64(params["end_date"]) - np.datetime64(params["start_date"])).astype(int) + 1
            h = 24 * n
            gio = [str(np.datetime64(params["start_date"]) + np.timedelta64(i, "h"))[:16] for i in range(h)]
            if "archive" in url:
                hv = {"precipitation": [2.0] * h, "weather_code": [63] * h, "wind_gusts_10m": [None] * h}
            else:
                hv = {"weather_code": [95 if i % 10 == 0 else 3 for i in range(h)], "cape": [1500.0] * h,
                      "lifted_index": [-3.0] * h}
            self.js = [{"hourly": {"time": gio, **hv}} for _ in diem]

        def json(self):
            return self.js

        def raise_for_status(self):
            pass

    goc, ngu = requests.get, thoi_tiet.time.sleep
    requests.get = lambda url, params, timeout: R(url, params)
    thoi_tiet.time.sleep = lambda s: None
    try:
        f = TMP / "om.npz"
        thoi_tiet.tai_open_meteo("2024-01-01", "2024-01-10", str(f), chunk_days=4)
        tt = ThoiTiet(str(f))
        assert tt.precip.shape == (240, len(diem)) and (tt.code == 63).all() and (tt.gust == 0).all()
        thoi_tiet.them_giong(str(f), luat="ma")
        tt = ThoiTiet(str(f))
        assert abs(np.isin(tt.code, thoi_tiet.GIONG).mean() - 0.1) < 1e-9
        z = dict(np.load(f))
        assert (z["code_era5"] == 63).all()
        thoi_tiet.ap_luat_giong(z, "cape")                  # CAPE 1500, mưa 2 mm/h → mọi giờ đều giông
        assert np.isin(z["code"], thoi_tiet.GIONG).all()
    finally:
        requests.get, thoi_tiet.time.sleep = goc, ngu


if __name__ == "__main__":
    tests = [(k, v) for k, v in dict(globals()).items() if k.startswith("test_")]
    for k, f in tests:
        f()
        print("OK ", k)
    print(f"{len(tests)} test đạt")
