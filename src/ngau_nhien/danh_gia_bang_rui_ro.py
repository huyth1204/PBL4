"""Áp bảng đã đóng băng lên Yagi, xuất ma trận online và chấm bằng nhãn tương lai riêng."""
import argparse
import json
from pathlib import Path
import numpy as np
from src.ngau_nhien.kich_ban import MangGoc, ThamSo, sinh_kich_ban
from src.ngau_nhien.thoi_tiet import ThoiTiet
from src.ngau_nhien.ma_tran_rui_ro import ROOT, SHAPE, nap_bang, ma_tran_online, metrics
from src.ngau_nhien.tao_du_lieu_bao import sha256


def danh_gia(table_dir, out, raw=ROOT/'dataset/oracle_dataset_v5.npz',
             weather=ROOT/'data/thoi_tiet_vn_mr.npz', split=ROOT/'dataset/v5/split_info.json',
             start='2024-09-07T06:00', seed=2026):
    table_dir, out = Path(table_dir), Path(out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'Từ chối ghi đè {out}')
    meta=json.loads((table_dir/'meta.json').read_text(encoding='utf-8'))
    if sha256(table_dir/'bang_tra.npz') != meta['table_sha256'] or sha256(raw) != meta['raw_sha256'] or sha256(split) != meta['split_sha256']:
        raise ValueError('Bảng/raw/split không khớp manifest')
    params=ThamSo(**meta['parameters'])
    b=nap_bang(table_dir/'bang_tra.npz', tham_so=params);age,delta=int(b['tuoi']),int(b['delta'])
    with np.load(raw,allow_pickle=False) as z:
        W,kinds,names=z['W_snapshots'],z['node_kind'],z['node_order']
    tt=ThoiTiet(weather);starts=np.flatnonzero(tt.time_utc.astype(str)==start)
    if len(starts)!=1:raise ValueError('start phải là giờ có trong tệp weather')
    h0=int(starts[0]);end=h0+int(np.floor((len(W)-1)*20/3600))+1
    if end>=tt.H:raise ValueError('Không đủ weather cho kịch bản')
    g=MangGoc(W,kinds,names,tt)
    wn,q=sinh_kich_ban(g,tt,h0,seed,params)
    ids=np.array(json.loads(Path(split).read_text(encoding='utf-8'))['snapshot_ids']['test'])
    ids=ids[(ids>=age+45)&(ids+delta<len(W))]
    if not len(ids):raise ValueError('Không có mốc test đủ lịch sử/cửa sổ')
    rows=[];ys=[]
    for t in ids:
        # Cắt q ngay tại quan sát để chốt việc không nhìn trước tương lai trong runtime.
        o=int(t-age)
        past={k:q[k][:o+1] for k in ('R','gust','giong','tat')}
        rows.append(ma_tran_online(b,g,wn[o],past,int(t),age,delta,tham_so=params))
        ys.append((wn[t:t+delta+1]>0).all(0))
    arrays={k:np.stack([r[k] for r in rows]) for k in rows[0]}
    arrays.update(target_survives_oracle=np.stack(ys),base_snapshot_id=ids,node_order=names)
    valid=arrays['valid'];p=1-arrays['risk_edge'][valid];y=arrays['target_survives_oracle'][valid]
    gp=float((b['song_train'].sum()+1)/(b['n_train'].sum()+2))
    score={'lookup':metrics(p,y.astype(int),np.ones(len(y)),p_baseline=gp),
           'constant_train_baseline':metrics(np.full(len(y),gp),y.astype(int),np.ones(len(y)),p_baseline=gp),
           'assume_forecast_edges_survive':metrics(np.ones(len(y)),y.astype(int),np.ones(len(y)),p_baseline=gp),
           'by_band':{}}
    bands=np.unravel_index(arrays['cell_id'][valid],SHAPE)[0]
    for i,name in enumerate(['no_rain_band','5GHz','12GHz','20GHz','28GHz']):
        m=bands==i;score['by_band'][name]=metrics(p[m],y[m].astype(int),np.ones(m.sum()),p_baseline=gp)
    score.update(start_utc=start,seed=seed,matrix_shape=list(arrays['risk_edge'].shape),
                 table_sha256=sha256(table_dir/'bang_tra.npz'),weather_sha256=sha256(weather),
                 scope='One Yagi window, v5 test geometry snapshots. Evaluation only; no refit.',
                 unknown_band_edges=int(((arrays['support_level']==0)&valid).sum()))
    out.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out/'matrices.npz',**arrays)
    (out/'evaluation.json').write_text(json.dumps(score,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    return score


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--table-dir',required=True);ap.add_argument('--out',required=True)
    ap.add_argument('--start',default='2024-09-07T06:00');ap.add_argument('--seed',type=int,default=2026)
    a=ap.parse_args();r=danh_gia(a.table_dir,a.out,start=a.start,seed=a.seed)
    print(json.dumps(r,ensure_ascii=False,indent=2))
