"""Báo cáo validation từ số đếm có sẵn, không chạy lại mô phỏng."""
import argparse
import json
from pathlib import Path
import numpy as np
from src.ngau_nhien.ma_tran_rui_ro import SHAPE, NCELLS, metrics


def _read(value):
    if isinstance(value, (str, Path)):
        with np.load(value, allow_pickle=False) as z:
            return {k: z[k] for k in z.files}
    return value


def bao_cao_validation(dem, bang, val_names=None):
    """Giữ các đường dẫn cũ <cơn>/pooled.lookup và constant_train_baseline.

    Các phân nhóm dùng cùng baseline cố định học trên TOÀN BỘ train, không fit
    lại baseline trên val hay từng nhóm. Hỗ trợ chỉ lấy từ n_train.
    """
    dem, bang = _read(dem), _read(bang)
    names = list(map(str, dem['ten_bao']))
    val_names = list(map(str, bang['val_storms'] if val_names is None else val_names))
    train_names = list(map(str, bang['train_storms']))
    if (len(names) != len(set(names)) or not val_names or len(val_names) != len(set(val_names))
            or not set(val_names) <= set(names) or not set(train_names) <= set(names)
            or not set(val_names) <= set(map(str, bang['val_storms']))
            or set(val_names) & set(train_names)):
        raise ValueError('Split theo cơn không hợp lệ hoặc val trùng train')
    s, n = np.asarray(dem['song']), np.asarray(dem['n'])
    if s.shape != n.shape or n.shape != (len(names), NCELLS):
        raise ValueError('Số đếm phải có kích thước [cơn, 3000]')
    p, support = np.asarray(bang['p_song']), np.asarray(bang['n_train'])
    if p.shape != (NCELLS,) or support.shape != p.shape:
        raise ValueError('Bảng phải có 3000 ô')
    tr = np.array([name in train_names for name in names])
    if (not np.array_equal(n[tr].sum(0), support)
            or not np.array_equal(s[tr].sum(0), bang['song_train'])):
        raise ValueError('Bảng và số đếm train không khớp')
    # Kiểm tra cả số đếm train, không chỉ các nhóm val được báo cáo.
    metrics(np.broadcast_to(p, n.shape), s, n)
    gp = float((np.asarray(bang['song_train']).sum()+1)/(support.sum()+2))
    vi = [names.index(name) for name in val_names]
    sv, nv = s[vi].sum(0), n[vi].sum(0)
    total_failures = int((nv-sv).sum())
    all_cells = np.ones(NCELLS, dtype=bool)

    def group(ss, nn, mask=all_cells):
        a = metrics(p[mask], ss[mask], nn[mask], p_baseline=gp)
        baseline = metrics(np.full(mask.sum(), gp), ss[mask], nn[mask], p_baseline=gp)
        low = a['samples'] < 1000 or a['failures'] < 30
        if low:
            for row in (a, baseline):
                row['bss'] = row['auprc'] = None
        return {'cells': int(mask.sum()), 'cells_with_validation': int(np.count_nonzero(nn[mask])),
                'samples': a['samples'], 'failures': a['failures'],
                'failure_share_percent': None if total_failures == 0 else 100*a['failures']/total_failures,
                'failure_share_denominator': total_failures,
                'mau_it': low, 'lookup': a, 'constant_train_baseline': baseline}

    report = {name: group(s[names.index(name)], n[names.index(name)]) for name in val_names}
    report['pooled'] = group(sv, nv)
    band, _, _, _, _, thunder = np.unravel_index(np.arange(NCELLS), SHAPE)
    report['by_band'] = {str(ghz): group(sv, nv, band == i) for i, ghz in enumerate([0, 5, 12, 20, 28])}
    masks = {'n=0': support == 0, '0<n<10': (support > 0) & (support < 10),
             '10<=n<1000': (support >= 10) & (support < 1000), 'n>=1000': support >= 1000}
    report['by_support'] = {label: group(sv, nv, mask) for label, mask in masks.items()}
    report['by_ground_thunder'] = {'no': group(sv, nv, thunder == 0), 'yes': group(sv, nv, thunder == 1)}
    report['schema'] = 'storm-validation-v2'
    report['baseline'] = {'p_survival': gp, 'source': 'all train storms; (sum(song_train)+1)/(sum(n_train)+2)',
                          'train_storms': train_names, 'validation_storms': val_names}
    report['denominators'] = {
        'survival_rate': 'survivals / samples in the reported group',
        'failure_rate': 'failures / samples in the reported group',
        'auprc_baseline': 'failure_rate in the reported group (descriptive prevalence, not a fitted model)',
        'failure_share_percent': '100 * group failures / failures pooled over the listed validation storms',
        'bss': '1 - group Brier / group Brier of the same global train constant baseline',
        'cells': 'number of schema cells in group (including cells with zero validation samples)',
        'cells_with_validation': 'number of cells in group with validation n > 0',
        'support': 'n_train per cell, never validation n',
    }
    report['small_sample_rule'] = {'min_samples': 1000, 'min_failures': 30,
                                    'rule': 'mau_it when either threshold is not met; bss and auprc are null'}
    report['ghi_chu'] = [
        'Dữ liệu mô phỏng; bảng học bằng n_train, không phải số quan sát sự cố ngoài thực địa.',
        'AUPRC là average precision của lớp đứt: gộp đồng điểm, cộng precision × phần tăng recall; không nội suy hình thang.',
        'Chỉ có 3 cơn val trong bộ đã công bố: báo riêng từng cơn, không bootstrap theo cạnh và không báo khoảng tin cậy.',
        'Không có mẫu: các tỷ lệ/chỉ số là null. Không có ca đứt: AUPRC null. Baseline Brier=0: BSS null.',
    ]
    return report


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dem', required=True); ap.add_argument('--bang', required=True)
    ap.add_argument('--out', required=True); ap.add_argument('--val', nargs='+')
    a = ap.parse_args()
    result = bao_cao_validation(a.dem, a.bang, a.val)
    Path(a.out).write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps(result['pooled']['lookup'], ensure_ascii=False, indent=2, allow_nan=False))
