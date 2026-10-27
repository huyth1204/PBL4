import json
from dataclasses import replace
from types import SimpleNamespace
import numpy as np
import pytest
from src.ngau_nhien.du_bao import du_bao_quy_dao
from src.ngau_nhien.ma_tran_rui_ro import (NCELLS, SHAPE, hoc_bang, o_dac_trung,
                                        ma_tran_online, metrics, nap_bang, hop_dong_bang, pham_vi_ap_dung, kiem_tra_cau_hinh)
from src.ngau_nhien.bao_manh import STORMS, overlaps_yagi
from src.ngau_nhien.kich_ban import ThamSo, tham_so_dict
from src.ngau_nhien.tao_du_lieu_bao import sha256


PARAMS = ThamSo(xung_dot=False, tai=False)


def metadata():
    return {'schema': 'storm-lookup-v1', 'parameters': tham_so_dict(PARAMS),
            'applicability': pham_vi_ap_dung(), 'tuoi': 9, 'delta': 3, 'dt_seconds': 20}


def bind(b):
    b['contract_json'] = json.dumps(hop_dong_bang(metadata()))
    return b


def save_table(path, b):
    np.savez(path, **b)
    meta = metadata(); meta['table_sha256'] = sha256(path)
    path.with_name('meta.json').write_text(json.dumps(meta))


def fixture():
    W = np.ones((80, 3, 3)); W[:,np.arange(3),np.arange(3)] = 0
    g = SimpleNamespace(W=W, kind=['leo','uav','user'], node_site=np.array([-1,0,1]),
                        site=np.array([[-1,0,1],[0,-1,1],[1,1,-1]]),
                        band=np.array([[0,12,12],[12,0,5],[12,5,0]]))
    q = {'R': np.ones((80,2)), 'gust':np.zeros((80,2)), 'giong':np.zeros((80,2),bool),
         'tat':np.zeros((80,3),bool)}
    s,n=np.zeros((2,NCELLS)),np.zeros((2,NCELLS));n[0]=10;s[0]=8
    b=hoc_bang(s,n,['train_storm','val_storm'],['val_storm']);b.update(tuoi=9,delta=3)
    return g,q,bind(b)


def test_new_satellite_cannot_connect_to_known_landed_uav():
    g,q,_=fixture();g.W[:51,0,1]=g.W[:51,1,0]=0
    Wo=g.W[50].copy();off=np.array([False,True,False])
    p=du_bao_quy_dao(g.W,Wo,50,59,3,off)
    assert p[0,1]==p[1,0]==0 and p[1].sum()==p[:,1].sum()==0
    assert p[0,2]>0
    active=du_bao_quy_dao(g.W,Wo,50,59,3,np.zeros(3,bool))
    assert active[0,1]>0  # thực sự tái hiện nhánh hệ số=1 của lỗi cũ


def test_online_does_not_read_future_weather_or_node_state():
    g,q,b=fixture()
    expected=ma_tran_online(b,g,g.W[50],q,59,tham_so=PARAMS)
    q['R'][51:]=1000;q['gust'][51:]=100;q['giong'][51:]=True;q['tat'][51:]=True
    actual=ma_tran_online(b,g,g.W[50],q,59,tham_so=PARAMS)
    for k in expected:np.testing.assert_array_equal(expected[k],actual[k])
    truncated={k:v[:51] for k,v in q.items()}
    np.testing.assert_array_equal(expected['risk_edge'],ma_tran_online(b,g,g.W[50],truncated,59,tham_so=PARAMS)['risk_edge'])


def test_validation_counts_do_not_change_table():
    n=np.ones((3,NCELLS))*10;s=n*.8
    a=hoc_bang(s,n,['a','b','c'],['c'])
    n[2]=1e9;s[2]=0
    b=hoc_bang(s,n,['a','b','c'],['c'])
    np.testing.assert_array_equal(a['p_song'],b['p_song'])
    with pytest.raises(ValueError):hoc_bang(s,n,['a','b','yagi'],['b'])
    with pytest.raises(ValueError):hoc_bang(s,n,['a','b','c'],['a','b','c'])


def test_bins_rain_trend_wind_and_thunder():
    g,q,_=fixture();q['R'][50,0]=20;q['gust'][50,0]=15;q['giong'][50,0]=True
    c=o_dac_trung(g,q,50)
    assert np.unravel_index(c[0,1],SHAPE)==(2,7,2,3,1,0)
    q['gust'][50,0]=15.01
    assert np.unravel_index(o_dac_trung(g,q,50)[0,1],SHAPE)[3]==4
    with pytest.raises(ValueError):o_dac_trung(g,q,44)


def test_unseen_cells_fallback_and_invalid_edges(tmp_path):
    g,q,_=fixture();n=np.zeros((2,NCELLS));s=n.copy()
    c=o_dac_trung(g,q,50)[0,1];n[0,c]=100;s[0,c]=80
    b=hoc_bang(s,n,['train','val'],['val']);b.update(tuoi=9,delta=3)
    p=tmp_path/'table.npz';save_table(p,bind(b))
    b=nap_bang(p,tham_so=PARAMS);r=ma_tran_online(b,g,g.W[50],q,59,tham_so=PARAMS)
    assert r['support_level'][0,1]==3 and r['support'][0,1]==100
    assert r['support_level'][1,2]==0 and r['risk_edge'][1,2]==.5
    assert np.isnan(np.diag(r['risk_edge'])).all()
    q['tat'][50,1]=True
    assert not ma_tran_online(b,g,g.W[50],q,59,tham_so=PARAMS)['valid'][1].any()
    with pytest.raises(ValueError):ma_tran_online(b,g,g.W[50],q,59,delta=9,tham_so=PARAMS)


def test_catalog_and_metric_known_values():
    assert len(STORMS)==10
    for _,start,end in STORMS:
        assert not overlaps_yagi(np.arange(np.datetime64(start),np.datetime64(end),np.timedelta64(1,'h')))
    assert overlaps_yagi(['2024-09-12T23:00'])
    assert metrics(np.array([.8]),np.array([8]),np.array([10]))['brier']==pytest.approx(.16)


def test_metric_is_finite_for_float32_certainties():
    result = metrics(np.array([0, 1], dtype=np.float32), [0, 1], [1, 1])
    assert np.isfinite(result['log_loss']) and result['log_loss'] < 1e-6
    wrong = metrics(np.array([0, 1], dtype=np.float32), [1, 0], [1, 1])
    assert np.isfinite(wrong['log_loss']) and wrong['log_loss'] > 18


@pytest.mark.parametrize('key,value', [('xung_dot',True),('tai',True),('snr0_db',12),
                                     ('p_set',.2),('gio_giat_max',16),('dt_s',30)])
def test_load_and_online_reject_changed_scenario(tmp_path, key, value):
    g,q,b=fixture();path=tmp_path/'table.npz';save_table(path,b)
    changed=replace(PARAMS,**{key:value})
    with pytest.raises(ValueError,match=key):nap_bang(path,tham_so=changed)
    loaded=nap_bang(path,tham_so=PARAMS)
    with pytest.raises(ValueError,match=key):
        ma_tran_online(loaded,g,g.W[50],q,59,tham_so=changed)


def test_runtime_configuration_is_required_and_rechecked(tmp_path):
    g,q,b=fixture();path=tmp_path/'table.npz';save_table(path,b)
    with pytest.raises(TypeError):nap_bang(path)
    with pytest.raises(TypeError):ma_tran_online(b,g,g.W[50],q,59)
    scenario=replace(PARAMS);loaded=nap_bang(path,tham_so=scenario)
    scenario.tai=True
    with pytest.raises(ValueError,match='tai'):
        ma_tran_online(loaded,g,g.W[50],q,59,tham_so=scenario)
    # Không có hợp đồng thì không được lách loader bằng cách tự np.load.
    b.pop('contract_json')
    with pytest.raises(ValueError,match='contract_json'):
        ma_tran_online(b,g,g.W[50],q,59,tham_so=PARAMS)


def test_load_requires_matching_sidecar_and_embedded_contract(tmp_path):
    _,_,b=fixture();path=tmp_path/'table.npz';save_table(path,b)
    sidecar=path.with_name('meta.json');meta=json.loads(sidecar.read_text())
    meta['table_sha256']='wrong';sidecar.write_text(json.dumps(meta))
    with pytest.raises(ValueError,match='Hash'):nap_bang(path,tham_so=PARAMS)
    save_table(path,b);meta=json.loads(sidecar.read_text())
    meta['applicability']['target']='a different event';sidecar.write_text(json.dumps(meta))
    with pytest.raises(ValueError,match='nhúng'):nap_bang(path,tham_so=PARAMS)
    sidecar.unlink()
    with pytest.raises(FileNotFoundError):nap_bang(path,tham_so=PARAMS)


def test_applicability_and_complete_parameters_are_required():
    meta=metadata();meta['applicability'].pop('required_flags')
    with pytest.raises(ValueError,match='applicability'):kiem_tra_cau_hinh(PARAMS,meta)
    meta=metadata();meta['parameters'].pop('p_set')
    with pytest.raises(ValueError,match='p_set'):kiem_tra_cau_hinh(PARAMS,meta)
    # Chỉ sửa cả parameters để hợp thức hóa bật tải cũng không qua required_flags.
    meta=metadata();meta['parameters']['tai']=True
    with pytest.raises(ValueError,match='tai'):kiem_tra_cau_hinh(replace(PARAMS,tai=True),meta)


def test_published_table_contract_and_empty_uav_states():
    from pathlib import Path
    folder=Path(__file__).resolve().parents[1]/'data/bao_manh/bang_tra'
    meta=json.loads((folder/'meta.json').read_text())
    b=nap_bang(folder/'bang_tra.npz',tham_so=ThamSo(**meta['parameters']))
    with np.load(folder/'dem.npz',allow_pickle=False) as d:
        all_counts=d['n'].sum(0).reshape(SHAPE)
        assert all_counts[:,:,:,4,:,:].sum()==0
        assert all_counts[:,:,:,:,1,:].sum()==0
        rebuilt=hoc_bang(d['song'],d['n'],d['ten_bao'],b['val_storms'],float(b['prior']))
        np.testing.assert_array_equal(b['p_song'],rebuilt['p_song'])
