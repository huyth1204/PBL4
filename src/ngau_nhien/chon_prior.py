"""Chọn prior trên val theo cơn, lưu bảng 7 cơn và refit bảng cuối riêng."""
import argparse
import copy
import json
from pathlib import Path
import numpy as np
from src.ngau_nhien.kich_ban import ThamSo
from src.ngau_nhien.ma_tran_rui_ro import hoc_bang, hoc_bang_tat_ca, hop_dong_bang, nap_bang
from src.ngau_nhien.bao_cao_rui_ro import _read, bao_cao_validation
from src.ngau_nhien.tao_du_lieu_bao import sha256

# Chốt trước khi chấm val; không mở rộng lưới theo kết quả/Yagi trong cùng lần chạy.
PRIOR_GRID = (0.1, 0.5, 1., 2., 5., 10., 20., 50., 100., 200., 500., 1000., 2000., 5000., 10000.)


def chon_va_hoc_lai(dem, bang, priors=PRIOR_GRID):
    """Không nhận dữ liệu Yagi. Val chỉ được đưa vào số đếm fit sau khi chọn."""
    dem, bang = _read(dem), _read(bang)
    # Kiểm tra bảng gốc khớp số đếm train và split trước khi chọn tham số.
    bao_cao_validation(dem, bang)
    grid = sorted(set(map(float, priors)))
    if not grid or not all(np.isfinite(p) and p > 0 for p in grid):
        raise ValueError('Lưới prior phải có các giá trị dương hữu hạn')
    names = list(map(str, dem['ten_bao']))
    val = list(map(str, bang['val_storms']))
    train = list(map(str, bang['train_storms']))
    if set(names) != set(train) | set(val):
        raise ValueError('Mỗi cơn trong số đếm phải thuộc train hoặc val đã định')
    rows = []
    for prior in grid:
        candidate = hoc_bang(dem['song'], dem['n'], names, val, prior)
        report = bao_cao_validation(dem, candidate, val)
        values = [report[name]['lookup']['brier'] for name in val]
        if any(v is None for v in values):
            raise ValueError('Mỗi cơn val phải có mẫu để chọn prior theo cơn')
        rows.append({'prior': prior, 'mean_storm_brier': float(np.mean(values)),
                     'pooled': report['pooled']['lookup'],
                     'by_storm': {name: report[name]['lookup'] for name in val}})
    # Chỉ coi là hòa nếu sai số rất nhỏ; chọn prior lớn hơn (làm trơn mạnh hơn).
    minimum = min(row['mean_storm_brier'] for row in rows)
    tied = [row for row in rows if np.isclose(row['mean_storm_brier'], minimum, rtol=1e-12, atol=1e-15)]
    selected = max(tied, key=lambda row: row['prior'])
    prior = selected['prior']
    selection_table = hoc_bang(dem['song'], dem['n'], names, val, prior)
    final_table = hoc_bang_tat_ca(dem['song'], dem['n'], names, prior=prior)
    selection = {
        'schema': 'storm-prior-selection-v1', 'prior_grid': grid,
        'criterion': 'minimize arithmetic mean of per-storm validation Brier; equal weight per storm',
        'tie_break': 'largest prior within rtol=1e-12, atol=1e-15 of minimum',
        'selection_split': {'train': train, 'val': val},
        'selected_prior': prior, 'selected_mean_storm_brier': selected['mean_storm_brier'],
        'boundary_winner': prior in (grid[0], grid[-1]), 'candidates': rows,
        'final_fit_storms': names, 'final_fit_samples': int(np.asarray(final_table['n_train']).sum()),
        'final_occupied_cells': int(np.count_nonzero(final_table['n_train'])),
        'ghi_chu': [
            'Lưới và tiêu chí được chốt trước khi chấm; không dùng Yagi để chọn prior hoặc fit.',
            'Điểm val dùng để chọn siêu tham số, không phải ước lượng test độc lập sau chọn.',
            'Điểm trong candidates/chon_prior/validation.json chỉ thuộc bảng fit trên train của bước chọn.',
            'Bảng cuối đã fit cả các cơn val; không gán điểm val của bảng chọn cho bảng cuối.',
            'Nếu thắng ở biên, chỉ kết luận tốt nhất trên lưới này; không tuyên bố tối ưu toàn cục.',
        ],
    }
    return selection, selection_table, final_table


def _write_table(folder, table, source_meta, selection, source_hashes, stage):
    folder.mkdir(parents=True, exist_ok=True)
    meta = copy.deepcopy(source_meta)
    meta.update(source_hashes)
    meta.update(fit_stage=stage, selected_prior=selection['selected_prior'],
                selection_split=selection['selection_split'],
                storm_split={'train': list(table['train_storms']), 'val': list(table['val_storms'])},
                fit_samples=int(np.asarray(table['n_train']).sum()),
                evaluation_status=('validation_used_for_selection' if stage == 'selection_train_only'
                                   else 'not_evaluated_after_refit'))
    meta['applicability']['prior_selection'] = (
        f"prior={selection['selected_prior']} selected by equal-storm mean validation Brier; "
        f"stage={stage}; fit storms are storm_split.train; selection scores do not evaluate the refit table.")
    table = {**table, 'tuoi': meta['tuoi'], 'delta': meta['delta'],
             'contract_json': json.dumps(hop_dong_bang(meta), ensure_ascii=False, sort_keys=True)}
    np.savez_compressed(folder/'bang_tra.npz', **{k: np.asarray(v) for k,v in table.items()})
    meta['table_sha256'] = sha256(folder/'bang_tra.npz')
    (folder/'meta.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def run(table_dir, out, priors=PRIOR_GRID):
    table_dir, out = Path(table_dir), Path(out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'Từ chối ghi đè {out}')
    meta = json.loads((table_dir/'meta.json').read_text(encoding='utf-8'))
    bang = nap_bang(table_dir/'bang_tra.npz', tham_so=ThamSo(**meta['parameters']))
    dem = _read(table_dir/'dem.npz')
    selection, chosen, final = chon_va_hoc_lai(dem, bang, priors)
    source_hashes = {'source_counts_sha256': sha256(table_dir/'dem.npz'),
                     'source_meta_sha256': sha256(table_dir/'meta.json'),
                     'source_table_sha256': sha256(table_dir/'bang_tra.npz')}
    selection['sources'] = source_hashes
    out.mkdir(parents=True, exist_ok=True)
    (out/'selection.json').write_text(json.dumps(selection,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    source_hashes = {**source_hashes, 'selection_sha256': sha256(out/'selection.json')}
    _write_table(out/'chon_prior', chosen, meta, selection, source_hashes, 'selection_train_only')
    report = bao_cao_validation(dem, chosen)
    report['usage'] = 'Validation used for prior selection; not an independent test and not a score of the 10-storm refit.'
    (out/'chon_prior/validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    _write_table(out, final, meta, selection, source_hashes, 'refit_all_storms')
    return selection


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--table-dir', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--priors', nargs='+', type=float, default=PRIOR_GRID)
    a = ap.parse_args(); result = run(a.table_dir, a.out, a.priors)
    print(json.dumps({k:result[k] for k in ('selected_prior','selected_mean_storm_brier','boundary_winner',
                                          'final_fit_samples','final_occupied_cells')},indent=2))
