"""Test cho phần mở rộng: nhiễu đo, so sánh Dijkstra, luật chia đôi, nhiễu mưa vật lý. Chạy: python test/test_mo_rong.py
Dùng dữ liệu demo huong_dan/du_lieu_demo (bỏ qua các test cần nó nếu chưa có)."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.ai.data_v2 import SagsinData, make_batch, nhieu_nhan  # noqa: E402
from src.ngau_nhien.nhieu_mua import (NGHEN, S_MUA, Z, delta, he_so_mua, thoi_gian_o_lai,  # noqa: E402
                                      ti_le_toc_do)
from test.tien_ich_test import BoQua, chay_tests, dung_utf8  # noqa: E402
from test.so_sanh_dijkstra_nhieu import co_ve_trung_binh, dijkstra, lam_nhieu, thong_ke_loai, tre  # noqa: E402

dung_utf8()

DEMO = ROOT / "huong_dan" / "du_lieu_demo"
THOI_TIET = ROOT / "data" / "thoi_tiet_vn.npz"


def _demo(n):
    f = DEMO / f"n{n}.npz"
    if not f.exists():
        return None
    return SagsinData(f)


def _can_demo(n, thu_muc=True):
    """Lấy SagsinData cỡ n; thiếu dữ liệu -> ném BoQua (SKIP, không phải PASS)."""
    if not thu_muc:
        raise BoQua("chưa có thư mục dữ liệu demo huong_dan/du_lieu_demo")
    d = _demo(n)
    if d is None:
        raise BoQua(f"chưa có dữ liệu demo n{n}.npz")
    return d


def test_nhieu_do_doi_xung_duong_va_giu_lien_ket():
    d = _can_demo(73)
    W = d.W[:3]
    A = nhieu_nhan(W, np.full(3, 0.4), np.random.default_rng(0))
    assert np.allclose(A, A.transpose(0, 2, 1))
    assert ((A > 0) == (W > 0)).all()
    assert np.allclose(nhieu_nhan(W, np.zeros(3), np.random.default_rng(0)), W)


def test_lo_huan_luyen_co_nhieu_giu_nguyen_mat_na():
    d = _can_demo(73)
    a = make_batch(d, np.arange(16), 0.0, np.random.default_rng(0))
    b = make_batch(d, np.arange(16), 0.0, np.random.default_rng(0), nhieu=0.6)
    assert (a["mask"] == b["mask"]).all() and (a["nxt"] == b["nxt"]).all()


def test_dijkstra_moi_khop_nhan_cua_TV1():
    for n in (73, 373):
        d = _can_demo(n)
        for i in range(0, len(d), 37):
            p = d.paths[i]
            W = d.W[d.sid[i]]
            q = dijkstra(W, p[0], p[-1], d.can_relay)
            assert abs(tre(W, q) - tre(W, p)) < 1e-4, (n, i)


def test_dijkstra_theo_so_chang_khong_nhieu_chang_hon():
    d = _can_demo(173)
    for i in range(0, len(d), 29):
        p = d.paths[i]
        q = dijkstra(d.W[d.sid[i]], p[0], p[-1], d.can_relay, theo_so_chang=True)
        assert len(q) <= len(p)


def test_khu_nhieu_bayes_hai_dau_mut():
    d = _can_demo(73)
    loai, tk = thong_ke_loai(d)
    W = d.W[:2]
    assert np.allclose(co_ve_trung_binh(W, 0.0, loai, tk), W)
    K = co_ve_trung_binh(lam_nhieu(W, 0.4, 1), np.inf, loai, tk)
    L = next(iter(tk))
    m = (W[0] > 0) & (loai == L)
    if m.any():
        assert np.allclose(K[0][m], tk[L][0], rtol=1e-4)


def test_uoc_luong_sigma_bang_hai_lan_do():
    from test.luat_chia_doi import nhieu_lk, uoc_luong_sigma0
    d = _can_demo(373)
    rng = np.random.default_rng(0)
    W = d.W[0]
    S = np.full(W.shape, 0.1, np.float32)
    s_hat = uoc_luong_sigma0(nhieu_lk(W, S, rng), nhieu_lk(W, S, rng), np.triu(W > 0, 1))
    assert abs(s_hat - 0.1) < 0.01, s_hat


def test_mua_trong_gio_giu_nguyen_trung_binh():
    for s in (0.8, S_MUA):
        assert abs(np.exp(s * Z - s * s / 2).mean() - 1) < 0.03          # tích phân Gauss: E[e^(sz)] = e^(s²/2)
        assert abs(he_so_mua(s).mean() - 1) < 1e-12
    z = np.random.default_rng(0).standard_normal(2_000_000)
    assert abs(np.exp(S_MUA * z - S_MUA ** 2 / 2).mean() - 1) < 0.02


def test_hang_doi_khop_tre_cua_TV1_va_nghen():
    for C in (100.0, 300.0, 1000.0):
        assert abs(float(thoi_gian_o_lai(C, 1.0)) - (12 / C + 12 / C * 0.3 / 0.7)) < 1e-9
    m = [float(thoi_gian_o_lai(100.0, c)) for c in (1.0, 0.8, 0.5)]
    assert m[0] < m[1] < m[2]
    assert np.isnan(thoi_gian_o_lai(100.0, 0.3 / NGHEN - 1e-6))
    assert abs(float(ti_le_toc_do(0.0)) - 1.0) < 1e-12


def test_khong_mua_thi_khong_them_nhieu():
    d, p = delta(0.0, 0.0, 20, 100.0, 7.8, 6.0, 3.0)
    assert float(d) < 1e-6 and float(p) == 0.0


def test_nhieu_mua_tang_theo_luong_mua_va_tan_so():
    ds = [float(delta(R, 0.0, 20, 100.0, 7.8, 6.0, 3.0)[0]) for R in (0.5, 2.0, 5.0)]
    assert ds[0] < ds[1] < ds[2], ds
    theo_bang = [float(delta(5.0, 0.0, f, 100.0, 7.8, 6.0, 3.0)[0]) for f in (5, 12, 20, 28)]
    assert theo_bang == sorted(theo_bang), theo_bang


def test_ma_tran_nhieu_mua_doi_xung_va_ve_tinh_khong_bi_mua():
    from src.ngau_nhien.kich_ban import MangGoc
    from src.ngau_nhien.thoi_tiet import ThoiTiet
    from test.luat_chia_doi import delta_matran
    d = _can_demo(73)
    if not THOI_TIET.exists():
        raise BoQua(f"chưa có dữ liệu thời tiết {THOI_TIET.name}")
    tt = ThoiTiet(str(THOI_TIET))
    g = MangGoc(d.W, d.kind, d.node_order, tt)
    h = int(np.argmax((tt.precip > 2).sum(1)))                     # giờ mưa ở nhiều điểm nhất
    D, mua = delta_matran(g, d, tt, 0, h)
    assert mua.any()
    assert np.allclose(D, D.T) and (mua == mua.T).all()
    ve_tinh = np.array([str(k) == "leo" for k in d.kind])
    assert not mua[np.ix_(ve_tinh, ve_tinh)].any()
    haps = np.array([str(k) == "haps" for k in d.kind])
    assert not mua[np.ix_(haps, ve_tinh)].any()


def test_mo_hinh_nhieu_chay_moi_quy_mo_va_tuyen_hop_le():
    from test.danh_gia_quy_mo import chay_mo_hinh, load
    f = ROOT / "results" / "quy_mo" / "dm_nhieu.pt"
    if not f.exists():
        raise BoQua(f"chưa có mô hình {f.relative_to(ROOT)}")
    _can_demo(73)
    m = load(f)
    for n in (73, 373):
        d = _can_demo(n)
        idx = np.arange(0, len(d), 25)
        rt, ok = chay_mo_hinh(m, d, idx, 1.0, 2, 200)
        assert ok.mean() > 0.95
        for r, o, i in zip(rt, ok, idx):
            if not o:
                continue
            W = d.W[d.sid[i]]
            # Độ trễ >= 0 là liên kết có thật; "không có liên kết" mã hoá bằng số âm.
            assert all(W[u, v] >= 0 for u, v in zip(r[:-1], r[1:]))
            assert all(d.can_relay[u] for u in r[1:-1])


def _run():
    return chay_tests(globals(), "test_mo_rong")


if __name__ == "__main__":
    sys.exit(0 if _run() else 1)
