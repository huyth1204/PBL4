import json
from pathlib import Path
import numpy as np
import pytest
from src.ngau_nhien.ma_tran_rui_ro import SHAPE, NCELLS, hoc_bang
from src.ngau_nhien.ablation_giong import hoc_bo_giong_mat_dat, bao_cao_ablation


def test_ablation_pools_train_counts_and_ignores_thunder_value():
    n=np.zeros((2,NCELLS),int);s=n.copy()
    # Hai ô chỉ khác giông mặt đất, số mẫu cân bằng: gộp phải cho p=.5.
    n[0,:2]=100;s[0,:2]=[90,10]
    n[1,:2]=2000;s[1,:2]=[1800,200]
    b=hoc_bang(s,n,['train','val'],['val'])
    p=hoc_bo_giong_mat_dat(b).reshape(SHAPE)
    assert p.flat[0]==pytest.approx(.5) and p.flat[1]==pytest.approx(.5)
    np.testing.assert_array_equal(p[...,0],p[...,1])
    before=b['p_song'].copy()
    report=bao_cao_ablation({'song':s,'n':n,'ten_bao':['train','val']},b)
    assert report['without_ground_thunder']['pooled']['lookup']['brier']==pytest.approx(.25)
    assert report['full_model']['pooled']['lookup']['brier'] < .1
    np.testing.assert_array_equal(b['p_song'],before)
    # Đảo hoàn toàn nhãn val không được thay đổi xác suất đã học của ablation.
    s[1]=n[1]-s[1]
    b2=hoc_bang(s,n,['train','val'],['val'])
    np.testing.assert_array_equal(hoc_bo_giong_mat_dat(b),hoc_bo_giong_mat_dat(b2))


def test_committed_ablation_compares_same_validation_samples():
    folder=Path(__file__).resolve().parents[1]/'data/bao_manh/bang_tra'
    r=bao_cao_ablation(folder/'dem.npz',folder/'bang_tra.npz')
    for key in ['pooled','prapiroon_2024','trami_2024','kalmaegi_2025']:
        a,b=r['full_model'][key],r['without_ground_thunder'][key]
        assert a['samples']==b['samples'] and a['failures']==b['failures']
        assert a['constant_train_baseline']==b['constant_train_baseline']
    json.dumps(r,allow_nan=False)
