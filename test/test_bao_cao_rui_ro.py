import json
from pathlib import Path
import numpy as np
import pytest
from src.ngau_nhien.ma_tran_rui_ro import NCELLS, hoc_bang, metrics, metrics_rui_ro
from src.ngau_nhien.bao_cao_rui_ro import bao_cao_validation


def assert_report_close(actual, expected):
    """Float có dung sai qua phiên bản NumPy; cấu trúc và số đếm vẫn chính xác."""
    if isinstance(expected, dict):
        assert isinstance(actual, dict) and actual.keys() == expected.keys()
        for key in expected:assert_report_close(actual[key], expected[key])
    elif isinstance(expected, list):
        assert isinstance(actual, list) and len(actual) == len(expected)
        for a,e in zip(actual,expected):assert_report_close(a,e)
    elif isinstance(expected, float):
        assert actual == pytest.approx(expected, rel=1e-9, abs=1e-15)
    else:
        assert type(actual) is type(expected)
        assert actual == expected


def test_report_comparison_tolerates_roundoff_but_not_count_changes():
    expected={'nested':[{'log_loss':.006317121328864498,'samples':5673610,'mau_it':False,'bss':None}]}
    actual=json.loads(json.dumps(expected));actual['nested'][0]['log_loss']+=1e-17
    assert_report_close(actual,expected)
    actual['nested'][0]['samples']+=1
    with pytest.raises(AssertionError):assert_report_close(actual,expected)


def test_hand_calculated_bss_and_tied_average_precision():
    # Ở ngưỡng .9: 3/4 đứt, recall=3/4. Ở .2: 4/10 đứt, recall=1.
    # AP=(3/4)*(3/4)+(4/10)*(1/4)=53/80; Brier=.168, baseline=.24.
    risk=np.array([.9,.9,.2]); failed=np.array([2,1,1]); n=np.array([2,2,6])
    m=metrics_rui_ro(risk,failed,n,p_baseline=.6)
    assert m['brier']==pytest.approx(.168)
    assert m['brier_baseline']==pytest.approx(.24)
    assert m['bss']==pytest.approx(.3)
    assert m['auprc']==pytest.approx(53/80)
    assert m['failures']==4 and m['failure_rate']==m['auprc_baseline']==.4
    # Hoán đổi hai nhóm đồng điểm, khác số mẫu/đứt, phải cho cùng AP.
    for order in ([1,0,2],[2,0,1]):
        other=metrics_rui_ro(risk[order],failed[order],n[order],p_baseline=.6)
        assert other['auprc']==pytest.approx(m['auprc'])
    expanded_risk=np.repeat(risk,n)
    expanded_failures=np.concatenate([np.r_[np.ones(f),np.zeros(nn-f)] for f,nn in zip(failed,n)])
    edge=metrics_rui_ro(expanded_risk,expanded_failures,np.ones(n.sum()),p_baseline=.6)
    old_api=metrics(1-risk,n-failed,n,p_baseline=.6)
    for key in m:
        assert edge[key]==pytest.approx(m[key])
        assert old_api[key]==pytest.approx(m[key])


def test_constant_scores_have_ap_equal_to_prevalence():
    r=metrics_rui_ro([.2,.2,.2],[3,0,1],[3,7,2])
    assert r['auprc']==pytest.approx(4/12)
    assert r['bss'] is None and r['brier_baseline'] is None


@pytest.mark.parametrize('risk,failed,n,p0', [
    ([0],[0],[10],1), ([1],[10],[10],0), ([],[],[],1), ([.5],[0],[0],.5)])
def test_boundaries_are_json_safe(risk,failed,n,p0):
    r=metrics_rui_ro(risk,failed,n,p_baseline=p0)
    json.dumps(r,allow_nan=False)
    assert r['bss'] is None
    if not sum(failed):assert r['auprc'] is None
    else:assert r['auprc']==1
    if sum(n):assert r['brier_baseline']==0
    else:
        assert r['samples']==r['failures']==0
        for key in ('survival_rate','failure_rate','brier','log_loss','brier_baseline','auprc_baseline'):
            assert r[key] is None


@pytest.mark.parametrize('risk,failed,n', [([float('nan')],[0],[1]),([1.1],[0],[1]),
                                         ([.2],[2],[1]),([.2],[.5],[1])])
def test_invalid_grouped_data_is_rejected(risk,failed,n):
    with pytest.raises(ValueError):metrics_rui_ro(risk,failed,n)


def make_report(val_n=None, val_failed=None):
    support=np.array([0,1,9,10,999,1000])
    n=np.zeros((2,NCELLS),dtype=int);s=n.copy()
    n[0,:6]=support;s[0,:6]=support-support//5
    n[1,:6]=[130,60,100,400,800,2000] if val_n is None else val_n
    failed=np.array([2,2,10,35,100,300] if val_failed is None else val_failed)
    s[1,:6]=n[1,:6]-failed
    dem={'n':n,'song':s,'ten_bao':['train','val']}
    bang=hoc_bang(s,n,['train','val'],['val'])
    return bao_cao_validation(dem,bang,['val']),bang


def test_support_partition_uses_train_counts_and_conserves_samples_failures():
    r,b=make_report();pooled=r['pooled'];groups=r['by_support']
    assert groups['n=0']['samples']==130
    assert groups['0<n<10']['samples']==160
    assert groups['10<=n<1000']['samples']==1200
    assert groups['n>=1000']['samples']==2000
    assert [groups[k]['cells_with_validation'] for k in groups]==[1,2,2,1]
    for field in ('by_support','by_band','by_ground_thunder'):
        for key in ('samples','failures','cells','cells_with_validation'):
            assert sum(g[key] for g in r[field].values())==pooled[key]
        assert sum(g['failure_share_percent'] for g in r[field].values())==pytest.approx(100)
    assert groups['n=0']['mau_it']
    assert groups['n=0']['lookup']['bss'] is None
    assert groups['n=0']['lookup']['auprc'] is None
    assert groups['n=0']['lookup']['auprc_baseline']==pytest.approx(2/130)
    assert not groups['10<=n<1000']['mau_it']
    assert r['baseline']['p_survival']==pytest.approx((b['song_train'].sum()+1)/(b['n_train'].sum()+2))
    assert groups['n=0']['failure_share_denominator']==pooled['failures']
    json.dumps(r,allow_nan=False)


@pytest.mark.parametrize('samples,failed,low',[(999,30,True),(1000,29,True),(1000,30,False)])
def test_small_sample_thresholds(samples,failed,low):
    r,_=make_report([0,0,0,0,0,samples],[0,0,0,0,0,failed])
    group=r['by_support']['n>=1000']
    assert group['mau_it']==low
    assert (group['lookup']['bss'] is None)==low
    assert (group['lookup']['auprc'] is None)==low


def test_empty_validation_and_failure_share_without_failures():
    for n in ([0]*6,[0,0,0,0,0,1000]):
        r,_=make_report(n,[0]*6)
        assert r['pooled']['failure_share_percent'] is None
        assert r['pooled']['lookup']['auprc'] is None
        json.dumps(r,allow_nan=False)


def test_committed_validation_regression():
    folder=Path(__file__).resolve().parents[1]/'data/bao_manh/bang_tra'
    r=bao_cao_validation(folder/'dem.npz',folder/'bang_tra.npz')
    m=r['pooled']['lookup']
    assert m['bss']==pytest.approx(.203,abs=.001)
    assert m['auprc']==pytest.approx(.287,abs=.001)
    assert m['samples']==5673610 and m['failures']==11063
    assert r['by_support']['n=0']['samples']==130
    assert r['by_band']['5']['lookup']['bss']==pytest.approx(.06,abs=.005)
    assert r['by_band']['28']['lookup']['bss']==pytest.approx(.25,abs=.005)
    assert_report_close(r,json.loads((folder/'validation.json').read_text()))


def test_yagi_uses_same_small_sample_rule_for_band_reports():
    from src.ngau_nhien.danh_gia_bang_rui_ro import cham_ma_tran
    from src.ngau_nhien.ma_tran_rui_ro import SHAPE
    sizes=[169148,2058,13874];failures=[0,0,94]
    arrays={'valid':np.ones(sum(sizes),bool), 'risk_edge':np.full(sum(sizes),.01),
            'target_survives_oracle':np.concatenate([np.arange(n)>=f for n,f in zip(sizes,failures)]),
            'cell_id':np.concatenate([np.full(n,np.ravel_multi_index((b,0,0,0,0,0),SHAPE)) for b,n in enumerate(sizes)])}
    score=cham_ma_tran(arrays,{'song_train':np.array([9900]),'n_train':np.array([10000])})
    for name in ['no_rain_band','5GHz','20GHz','28GHz']:
        g=score['by_band'][name]
        assert g['mau_it'] and g['bss'] is None and g['auprc'] is None
    g=score['by_band']['12GHz']
    assert not g['mau_it'] and g['failures']==94 and g['bss'] is not None
    assert score['small_sample_rule']==bao_cao_validation(
        Path(__file__).resolve().parents[1]/'data/bao_manh/bang_tra/dem.npz',
        Path(__file__).resolve().parents[1]/'data/bao_manh/bang_tra/bang_tra.npz')['small_sample_rule']
    json.dumps(score,allow_nan=False)
