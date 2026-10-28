import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from src.ngau_nhien.chon_prior import chon_va_hoc_lai, run
from src.ngau_nhien.ma_tran_rui_ro import NCELLS, hoc_bang, hoc_bang_tat_ca, nap_bang, ma_tran_online
from src.ngau_nhien.bao_cao_rui_ro import bao_cao_validation
from src.ngau_nhien.kich_ban import ThamSo
from src.ngau_nhien.tao_du_lieu_bao import sha256


def example():
    n=np.zeros((3,NCELLS),int);s=n.copy()
    n[0,:2]=10;s[0,:2]=[9,1]
    # Cơn nhỏ muốn p=.5; cơn lớn muốn p=.9/.1. Trung bình theo cơn muốn .7/.3.
    n[1,:2]=200;s[1,:2]=[100,100]
    n[2,:2]=20000;s[2,:2]=[18000,2000]
    names=['train','small_val','large_val']
    return {'n':n,'song':s,'ten_bao':names},hoc_bang(s,n,names,names[1:])


def test_selection_is_macro_not_pooled_and_final_refit_includes_validation():
    dem,b=example();before=b['p_song'].copy()
    r,chosen,final=chon_va_hoc_lai(dem,b,[.1,10,200])
    assert r['selected_prior']==10
    assert min(r['candidates'],key=lambda c:c['pooled']['brier'])['prior']==.1
    for row in r['candidates']:
        assert row['mean_storm_brier']==pytest.approx(np.mean([v['brier'] for v in row['by_storm'].values()]))
    np.testing.assert_array_equal(chosen['n_train'],dem['n'][0])
    np.testing.assert_array_equal(chosen['song_train'],dem['song'][0])
    np.testing.assert_array_equal(final['n_train'],dem['n'].sum(0))
    np.testing.assert_array_equal(final['song_train'],dem['song'].sum(0))
    np.testing.assert_array_equal(b['p_song'],before)
    assert chosen['val_storms']==['small_val','large_val']
    assert final['val_storms']==[] and final['train_storms']==dem['ten_bao']
    assert final['prior']==chosen['prior']==10
    with pytest.raises(ValueError,match='Split'):
        bao_cao_validation(dem,final)


def test_selection_predictions_do_not_fit_validation_counts():
    dem,b=example()
    _,chosen,_=chon_va_hoc_lai(dem,b,[20])
    dem['song'][1:]=0
    _,changed,_=chon_va_hoc_lai(dem,b,[20])
    np.testing.assert_array_equal(chosen['p_song'],changed['p_song'])


def test_ties_prefer_larger_prior_and_order_is_irrelevant():
    dem,b=example();dem['song'][0,:2]=5
    b=hoc_bang(dem['song'],dem['n'],dem['ten_bao'],dem['ten_bao'][1:])
    r,_,_=chon_va_hoc_lai(dem,b,[200,1,20,20])
    assert r['selected_prior']==200 and r['boundary_winner']
    assert r['prior_grid']==[1,20,200]


@pytest.mark.parametrize('grid',[[],[0],[-1],[float('nan')],[float('inf')]])
def test_invalid_grid_is_rejected(grid):
    dem,b=example()
    with pytest.raises(ValueError,match='Lưới'):chon_va_hoc_lai(dem,b,grid)


def test_refit_rejects_yagi_and_original_fit_still_requires_holdout():
    dem,b=example()
    with pytest.raises(ValueError,match='Yagi'):
        hoc_bang_tat_ca(dem['song'],dem['n'],['train','yagi_2024','val'],prior=20)
    with pytest.raises(ValueError,match='val'):
        hoc_bang(dem['song'],dem['n'],dem['ten_bao'],[],20)


def test_exported_tables_load_online_and_keep_original_artifacts(tmp_path):
    source=Path(__file__).resolve().parents[1]/'data/bao_manh/bang_tra'
    originals={p:sha256(source/p) for p in ('dem.npz','bang_tra.npz','meta.json','validation.json')}
    r=run(source,tmp_path/'final',[20,200])
    assert r['selected_prior']==200
    for sub,stage,count in [('', 'refit_all_storms',10),('chon_prior','selection_train_only',7)]:
        folder=tmp_path/'final'/sub
        meta=json.loads((folder/'meta.json').read_text());params=ThamSo(**meta['parameters'])
        table=nap_bang(folder/'bang_tra.npz',tham_so=params)
        assert meta['fit_stage']==stage and len(table['train_storms'])==count
        assert float(table['prior'])==200 and meta['selection_sha256']==sha256(tmp_path/'final/selection.json')
        W=np.ones((80,2,2));W[:,np.arange(2),np.arange(2)]=0
        g=SimpleNamespace(W=W,kind=['leo','user'],site=np.array([[-1,0],[0,-1]]),
                          node_site=np.array([-1,0]),band=np.array([[0,12],[12,0]]))
        q={'R':np.ones((51,1)),'gust':np.zeros((51,1)),
           'giong':np.zeros((51,1),bool),'tat':np.zeros((51,2),bool)}
        online=ma_tran_online(table,g,W[50],q,59,tham_so=params)
        assert online['valid'][0,1] and np.isfinite(online['risk_edge'][0,1])
    assert not (tmp_path/'final/validation.json').exists()
    assert (tmp_path/'final/chon_prior/validation.json').exists()
    assert originals=={p:sha256(source/p) for p in originals}
    with pytest.raises(FileExistsError):run(source,tmp_path/'final',[20,200])


def test_published_final_table_has_all_counts_and_separate_selection_scores():
    root=Path(__file__).resolve().parents[1]/'data/bao_manh'
    folder=root/'bang_tra_cuoi'
    selection=json.loads((folder/'selection.json').read_text())
    assert selection['selected_prior']==200 and not selection['boundary_winner']
    assert len(selection['prior_grid'])==15
    meta=json.loads((folder/'meta.json').read_text())
    final=nap_bang(folder/'bang_tra.npz',tham_so=ThamSo(**meta['parameters']))
    with np.load(root/'bang_tra/dem.npz',allow_pickle=False) as d:
        expected=hoc_bang_tat_ca(d['song'],d['n'],d['ten_bao'],prior=200)
        np.testing.assert_array_equal(final['n_train'],d['n'].sum(0))
    np.testing.assert_allclose(final['p_song'],expected['p_song'],rtol=1e-12,atol=1e-15)
    assert int(final['n_train'].sum())==26088602 and np.count_nonzero(final['n_train'])==364
    assert meta['evaluation_status']=='not_evaluated_after_refit'
    assert not len(final['val_storms']) and len(final['train_storms'])==10
    assert not (folder/'validation.json').exists()
    assert (folder/'chon_prior/validation.json').exists()
