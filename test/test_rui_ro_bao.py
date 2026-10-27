"""Kiểm chứng xác suất có điều kiện, đứt tương quan và giữ ca mất kết nối."""
import json
import numpy as np
import pytest

from src.ngau_nhien.rui_ro import TichLuyRuiRo, cua_so_dut
from src.ngau_nhien.tao_du_lieu_bao import tao_bo
from src.ngau_nhien.thoi_tiet import cac_diem


def world():
    a = np.zeros((3, 3, 3), bool)
    a[:, 0, 1] = a[:, 1, 2] = True
    return a


def test_rui_ro_tuyen_giu_phu_thuoc_cung_bao():
    a, b = world(), world()
    a[1:, 0, 1] = a[1:, 1, 2] = False
    acc = TichLuyRuiRo([0], 3, 2, [0], [[0, 1, 2]])
    acc.them(a)
    acc.them(b)
    r = acc.ket_qua()
    assert r['risk_edge_oracle'][0, 0, 1] == 0.5
    assert r['risk_edge_oracle'][0, 1, 2] == 0.5
    assert r['risk_route_oracle'][0] == 0.5  # không phải 1 - (1 - .5)^2 = .75
    assert r['risk_route_count'][0] == 2


def test_dieu_kien_con_song_va_nan_khong_phai_rui_ro_0():
    a, b = world(), world()
    a[:, 0, 1] = False
    acc = TichLuyRuiRo([0], 3, 2, [0], [[0, 1, 2]])
    acc.them(a)
    acc.them(b)
    r = acc.ket_qua()
    assert r['risk_edge_count'][0, 0, 1] == 1
    assert r['risk_edge_oracle'][0, 0, 1] == 0
    assert r['risk_route_count'][0] == 1
    assert np.isnan(r['risk_edge_oracle'][0, 0, 2])
    assert np.isnan(np.diagonal(r['risk_edge_oracle'][0])).all()


def test_dut_do_quy_dao_van_la_rui_ro_va_khong_cat_cua_so():
    a = world()
    a[2, 0, 1] = False
    acc = TichLuyRuiRo([0], 3, 2, [0], [[0, 1, 2]])
    acc.them(a)
    assert acc.ket_qua()['risk_route_oracle'][0] == 1
    with pytest.raises(ValueError, match='đủ cửa sổ'):
        cua_so_dut(a, np.array([1]), 2)
    with pytest.raises(ValueError):
        cua_so_dut(a, np.array([0]), 0)


def test_cua_so_chi_dung_tuong_lai():
    a = world()
    a[0, 0, 1] = False
    assert not cua_so_dut(a, np.array([0]), 2)[0, 0, 1]


def test_tuyen_rong_khong_duoc_dem_la_an_toan():
    acc = TichLuyRuiRo([0], 3, 2, [0], [[]])
    acc.them(world())
    r = acc.ket_qua()
    assert r['risk_route_count'][0] == 0
    assert np.isnan(r['risk_route_oracle'][0])


def test_sinh_du_lieu_bao_giu_truy_van_mat_ket_noi_va_tai_lap(tmp_path):
    # Hai gateway chỉ có thể liên lạc qua UAV; gió mạnh hạ cánh UAV.
    W = np.full((12, 3, 3), -1, np.float32)
    W[:, 0, 2] = W[:, 2, 0] = W[:, 1, 2] = W[:, 2, 1] = 1
    W[:, np.arange(3), np.arange(3)] = 0
    raw = tmp_path / 'raw.npz'
    np.savez(raw, W_snapshots=W, node_order=['Hanoi', 'DaNang', 'UAV-1'],
             node_kind=['gateway', 'gateway', 'uav'], node_layer=['ground', 'ground', 'air'])
    sites = cac_diem()
    weather = tmp_path / 'weather.npz'
    np.savez(weather, site_names=[s[0] for s in sites], lat=[s[1] for s in sites], lon=[s[2] for s in sites],
             time_utc=['2024-09-07T06:00', '2024-09-07T07:00'], precip=np.zeros((2, len(sites))),
             code=np.zeros((2, len(sites)), np.int16), gust=np.full((2, len(sites)), 32.), nguon='test-synthetic')
    kwargs = dict(snapshots=12, stride=3, delta=2, ensemble=3, pairs=2, seed=42)
    a, b = tmp_path / 'a', tmp_path / 'b'
    meta = tao_bo(raw, weather, a, **kwargs)
    tao_bo(raw, weather, b, **kwargs)
    with np.load(a / 'test.npz') as z, np.load(b / 'test.npz') as y:
        assert z['query_reachable'].size == 8
        assert not z['query_reachable'].any()
        assert (z['route_failed_observed'] == -1).all()
        assert (z['current_path_indices'] == -1).all()
        assert np.isnan(z['risk_route_oracle']).all()
        assert np.all(z['risk_edge_count'] == 0)
        for k in z.files:
            np.testing.assert_array_equal(z[k], y[k], err_msg=k)
    assert meta['summary']['unreachable_observed'] == 8
    assert meta['purpose'] == 'evaluation_only_yagi_holdout'
    assert not meta['synthetic_stress']
    assert meta['observation_seed'] not in meta['ensemble_seeds']
    assert json.loads((a / 'meta.json').read_text())['output_sha256']
    with pytest.raises(FileExistsError):
        tao_bo(raw, weather, a, **kwargs)
    with pytest.raises(ValueError, match='thời tiết liên tục'):
        tao_bo(raw, weather, tmp_path / 'tail', start='2024-09-07T07:00', **kwargs)
