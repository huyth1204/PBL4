"""Bỏ đặc trưng giông mặt đất khỏi bảng tra; giữ nguyên nhãn và split mô phỏng."""
import argparse
import json
from pathlib import Path
import numpy as np
from src.ngau_nhien.ma_tran_rui_ro import SHAPE
from src.ngau_nhien.bao_cao_rui_ro import _read, bao_cao_validation


def hoc_bo_giong_mat_dat(bang):
    """Học 1500 nhóm từ train: gộp hai giá trị của trục giông mặt đất.

    Giữ nguyên prior và tầng băng/băng×mưa. Broadcast về 3000 chỉ để chấm cùng
    các ô val; không nhân đôi số đếm hoặc sửa bảng đang sử dụng.
    """
    s=np.asarray(bang['song_train']).reshape(SHAPE).sum(-1,keepdims=True)
    n=np.asarray(bang['n_train']).reshape(SHAPE).sum(-1,keepdims=True)
    prior=float(bang['prior'])
    if not np.isfinite(prior) or prior <= 0:
        raise ValueError('prior phải dương và hữu hạn')
    # Cùng công thức làm trơn của hoc_bang, sau khi gộp nhóm trên train.
    sb,nb=s.sum((1,2,3,4,5)),n.sum((1,2,3,4,5))
    pb=(sb+1)/(nb+2)
    sbr,nbr=s.sum((2,3,4,5)),n.sum((2,3,4,5))
    pbr=(sbr+prior*pb[:,None])/(nbr+prior)
    p=(s+prior*pbr[:,:,None,None,None,None])/(n+prior)
    return np.broadcast_to(p,SHAPE).copy().reshape(-1)


def bao_cao_ablation(dem, bang):
    dem,bang=_read(dem),_read(bang)
    full=bao_cao_validation(dem,bang)
    ablated={**bang,'p_song':hoc_bo_giong_mat_dat(bang)}
    without=bao_cao_validation(dem,ablated)
    deltas={}
    for key in ('pooled',*map(str,bang['val_storms'])):
        a,b=full[key]['lookup'],without[key]['lookup']
        deltas[key]={m:None if a[m] is None or b[m] is None else b[m]-a[m]
                     for m in ('brier','log_loss','bss','auprc')}
    return {
        'schema':'ground-thunder-feature-ablation-v1',
        'full_model':full, 'without_ground_thunder':without,
        'delta_without_minus_full':deltas,
        'ghi_chu':[
            'Ablation đặc trưng bảng tra, không tắt giông trong mô phỏng và không phải ablation GNN.',
            'Gộp số đếm TRAIN trên trục giông mặt đất, giữ năm trục còn lại và prior cố định. Không dùng val để học hay chọn prior.',
            'Cả hai mô hình được chấm trên cùng nhãn/cạnh/cơn val và cùng baseline train.',
            'Phân nhóm hỗ trợ trong hai báo cáo dùng cùng n_train của 3000 ô gốc để so sánh cùng tập mẫu; không phải hỗ trợ của 1500 nhóm sau gộp.',
            'Giữ nguyên bảng triển khai. Không thay bằng mô hình ablation dựa trên kết quả val này.',
            'Chỉ ba cơn val, không khoảng tin cậy theo cạnh. Không suy ra tác động nhân quả của giông hoặc khả năng GNN học tắt.',
        ],
    }


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dem',required=True);ap.add_argument('--bang',required=True)
    ap.add_argument('--out',required=True)
    a=ap.parse_args();r=bao_cao_ablation(a.dem,a.bang)
    Path(a.out).write_text(json.dumps(r,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(r['delta_without_minus_full'],indent=2,allow_nan=False))
