"""Export reproducible TN6/TN11 demo figures from saved results; no training.

    python -m scripts.xuat_demo_thong_ke

PNG (220 dpi), SVG, CSV, an offline HTML gallery and interactive learning curves.
TN6 demo results and locally retrained DM6 histories are explicitly separate sources.
"""
from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import html
import json
from collections import Counter, defaultdict
from pathlib import Path
import shutil
import zipfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
KQ = ROOT / 'results' / 'bao_da_tuyen'
BLUE, ORANGE, GREEN, RED, GRAY = '#2563a6', '#d07828', '#12836a', '#bd414b', '#637087'
NAMES = {'c6': 'C · ngày giông', 'yagi5': 'Yagi', 'c6_b': 'C · thế giới B',
         'yagi5_b': 'Yagi · thế giới B', 'ngay9': 'Ngày giông mới', 'nam9': 'Ngày xấu 2017–21',
         'bao9': 'Bão 2017–21', 'c7wh': 'C · cực đoan WH', 'y7wh': 'Yagi · cực đoan WH',
         'xc11': 'C · cực đoan mới', 'xy11': 'Yagi · cực đoan mới'}
GROUPS = {'train10_k4_5': 'train · bão', 'train6_10_k4_5': 'train6 · ngày', 'x11_k4_5': 'x11 · cực đoan'}
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.titlesize': 12,
                     'axes.titleweight': 'bold', 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.labelcolor': '#263449', 'text.color': '#263449', 'axes.edgecolor': '#ccd3dc',
                     'grid.color': '#e5eaf0', 'svg.fonttype': 'none', 'figure.facecolor': 'white',
                     'savefig.facecolor': 'white'})


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def csv_out(out, name, rows):
    with (out / 'csv' / f'{name}.csv').open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def figure(title, subtitle, size=(13, 7), nrows=1, ncols=1, **kw):
    fig, ax = plt.subplots(nrows, ncols, figsize=size, **kw)
    fig.suptitle(title, x=.03, ha='left', y=.98, fontsize=18, weight='bold')
    fig.text(.03, .918, subtitle, ha='left', va='top', fontsize=10, color=GRAY)
    return fig, ax


def save(fig, out, stem, note, rect=(.02, .10, .98, .85)):
    fig.text(.03, .025, note, fontsize=9, color=GRAY, va='bottom')
    fig.tight_layout(rect=rect, w_pad=2.8, h_pad=2.4)
    for suffix in ('png', 'svg'):
        fig.savefig(out / f'{stem}.{suffix}', dpi=220)
    plt.close(fig)


def forest(ax, labels, values, title, xlabel='Hiệu mất dịch vụ (điểm %)'):
    for i, (mean, (lo, hi)) in enumerate(values):
        color = GREEN if hi < 0 else RED if lo > 0 else GRAY
        ax.hlines(i, lo, hi, color=color, lw=2.5)
        ax.plot(mean, i, 'o', color=color, ms=6)
    ax.axvline(0, color='#263449', lw=1, ls='--')
    ax.set_yticks(range(len(labels)), labels)
    ax.invert_yaxis()
    ax.set_title(title, loc='left', pad=12)
    ax.set_xlabel(xlabel)
    ax.grid(axis='x', alpha=.8)
    ax.margins(y=.08, x=.12)


def tn6_plot(out, r6):
    rows, labels, dv, jj = [], [], [], []
    for bo, label in [('c6', 'C · ngày giông'), ('mua5', 'B · mưa lớn'), ('yagi5', 'A · Yagi')]:
        for dai, name in [('thap', 'thấp'), ('giua', 'giữa'), ('cao', 'cao')]:
            x = r6[f'{bo}_k4_5'][dai]
            a, b = x['tieu_chi']['C4_mat_dv'], x['tieu_chi']['C2_J']
            labels.append(f'{label} / {name}')
            dv.append(a)
            jj.append(b)
            rows.append({'bo': bo, 'dai': dai, 'n': x['so_tinh_huong'], 'delta_mat_dv': a[0],
                         'ci_low': a[1][0], 'ci_high': a[1][1], 'delta_J': b[0], 'J_ci_low': b[1][0],
                         'J_ci_high': b[1][1], 'thang_C2_C4': x['tieu_chi']['thang']})
    fig, axes = figure('TN6 · Demo thống kê từ kết quả gốc',
                       f'Lai6 − Dijkstra phạt · 4–5 luồng · τ = {r6["tau"]} · 9 ô được trình bày đầy đủ',
                       ncols=2, size=(14, 7.5))
    forest(axes[0], labels, dv, 'Mất dịch vụ: âm là tốt hơn')
    forest(axes[1], labels, jj, 'Hàm mục tiêu J: âm là tốt hơn', 'Hiệu J')
    save(fig, out, '01_tn6_thong_ke',
         'Chấm = hiệu trung bình; đoạn = CI 95% bootstrap ghép cặp theo cửa sổ. CI chạm/cắt 0: chưa đủ bằng chứng.\n'
         'Nguồn: ket_qua_6.json trên nhánh demo. Không ghép với DM6 huấn luyện lại tại máy nhóm; chưa hiệu chỉnh Holm ở hình này.')
    csv_out(out, '01_tn6_thong_ke', rows)


def tn11_forest(out, r):
    bo = list(NAMES)
    labels = [NAMES[b] for b in bo]
    tests = r['kiem_tra']
    comparisons = [('H11.1 · Lai(DM11) − Lai(DM10c)', 'dm11_tru_dm10c'),
                   ('H11.2 · DM11 − SM11', 'dm11_tru_sm11'), ('H11.3 · Lai(DM11) − Dijkstra', None)]
    fig, axes = figure('TN11 · Hiệu quả đi kèm độ bất định',
                       '11 bộ kiểm tra · 4–5 luồng · trung bình 3 seed trên cùng tình huống · càng âm càng tốt',
                       ncols=3, size=(18, 8))
    rows = []
    for ax, (title, key) in zip(axes, comparisons):
        values = [tests[b][key] if key else tests[b]['hieu_vs_dphat']['Lai(DM11)'] for b in bo]
        forest(ax, labels, values, title)
        for i in range(7, 11):
            ax.axhspan(i-.45, i+.45, color=BLUE, alpha=.05)
        for b, (m, ci) in zip(bo, values):
            rows.append({'bo': b, 'so_sanh': title, 'mean': m, 'ci_low': ci[0], 'ci_high': ci[1]})
    save(fig, out, '02_tn11_hieu_va_ci',
         'Vùng nền xanh = 4 bộ cực đoan. CI 95% bootstrap theo cửa sổ (10.000 lần), chưa hiệu chỉnh bội cho các CI ở đây.\n'
         'H11.1 không đạt; H11.2 đạt; H11.3 đạt cả điều kiện Holm riêng trong báo cáo. Không đồng nhất CI từng bộ với kết quả Holm.')
    csv_out(out, '02_tn11_hieu_va_ci', rows)


def curves(out, r):
    fig, axes = figure('TN11 · Đường học và checkpoint',
                       'Hai ví dụ cố định: DM11_p2 và SM11_p2 · toàn bộ 21 mô hình xem ở duong_hoc_tuong_tac.html',
                       nrows=2, ncols=2, size=(14, 9))
    for row, name in enumerate(('dm11_p2', 'sm11_p2')):
        h = r['hoc'][name]
        nk = np.array(h['nk'])
        a, b = axes[row]
        a.plot(nk[:, 0], nk[:, 1], color=BLUE, label='Loss train')
        a.plot(nk[:, 0], nk[:, 2], color=ORANGE, label='Loss val')
        b.plot(nk[:, 0], nk[:, 3], color=GREEN, label='J val')
        b.axhline(h['J_thay'], color=GRAY, ls=':', label='J thầy (tham khảo)')
        for ax in (a, b):
            ax.axvline(h['epoch_tot'], ls='--', color=RED, lw=1,
                       label=f'Checkpoint: epoch {h["epoch_tot"]}')
            ax.set_xlabel('Epoch')
            ax.grid(alpha=.7)
            ax.legend(fontsize=8, loc='best')
        a.set_title(f'{name} · val/train tại checkpoint = {h["ti_le_val_train"]:.2f}', loc='left')
        a.set_ylabel('Loss nhãn thầy')
        b.set_title(f'Dừng ở epoch {h["so_epoch"]}; chọn checkpoint theo J val', loc='left')
        b.set_ylabel('J val (thấp hơn là tốt hơn)')
    save(fig, out, '03_tn11_duong_hoc',
         'SM11_p2: loss train giảm nhưng loss val tăng về cuối → dấu hiệu overfitting theo loss.\n'
         'Loss val không dùng cùng mẫu ngẫu nhiên với loss train; không diễn giải tỷ lệ 1,25 như định luật phổ quát.')


def fit_diagnostics(out, r):
    cv = [(k, v) for k, v in r['hoc'].items() if v['phan'] is not None]
    fig, axes = figure('TN11 · Hai phép kiểm tra học thuộc',
                       'Ngưỡng được đăng ký trước cho TN11; đây là tín hiệu chẩn đoán, không phải chứng minh tuyệt đối',
                       ncols=2, size=(15, 7.5))
    a, b = axes
    ratios = [v['ti_le_val_train'] for _, v in cv]
    a.barh([k for k, _ in cv], ratios, color=[RED if x > 1.25 else BLUE for x in ratios])
    a.axvline(1.25, color=RED, ls='--', label='Ngưỡng 1,25')
    a.invert_yaxis()
    a.set_title('Loss val / train tại checkpoint tốt nhất', loc='left')
    a.set_xlabel('Tỷ lệ (3/12 mô hình vượt ngưỡng)')
    a.grid(axis='x', alpha=.7)
    a.legend()
    gap_rows = []
    for i, part in enumerate(('A (thường)', 'x11 (cực đoan)')):
        for j, (h, color) in enumerate((('dm11', BLUE), ('sm11', ORANGE))):
            m, (lo, hi) = r['kiem_thu_cheo'][part][f'{h}_khoang_hoc_thuoc']
            y = i*3 + j
            b.hlines(y, lo, hi, color=color, lw=3)
            b.plot(m, y, 'o', color=color)
            b.annotate(f'{m:.2f}', (m, y), xytext=(0, 9), textcoords='offset points', ha='center', color=color)
            gap_rows.append({'phan': part, 'mo_hinh': h, 'gap': m, 'ci_low': lo, 'ci_high': hi})
    b.set_yticks([0, 1, 3, 4], ['DM11 · A', 'SM11 · A', 'DM11 · x11', 'SM11 · x11'])
    b.invert_yaxis()
    b.axvline(0, color=GRAY, ls=':')
    b.axvline(1, color=RED, ls='--', label='Ngưỡng gap trung bình: 1 điểm %')
    b.set_title('Mất DV: ngoài fold − tình huống đã học', loc='left')
    b.set_xlabel('Chênh lệch (điểm %) · CI 95% theo cửa sổ')
    b.grid(axis='x', alpha=.7)
    b.margins(y=.22, x=.15)
    b.legend(fontsize=8, loc='lower right')
    save(fig, out, '04_tn11_overfitting',
         'SM11_p0, p2, p3 vượt ngưỡng tỷ lệ loss. SM11 trên x11 có gap trung bình 1,52 > 1,00 điểm %.\n'
         'Luật H11.4 xét trung bình gap; CI của SM11/x11 vẫn chứa 1,00. Điểm đã học do mô hình fold kế tiếp chấm.')
    csv_out(out, '04_tn11_gap', gap_rows)
    csv_out(out, '04_tn11_loss_ratio', [{'model': k, 'ratio': v['ti_le_val_train'],
                                      'epoch_best': v['epoch_tot']} for k, v in cv])


def underfit(out, r):
    rows = [(k, v) for k, v in r['hoc'].items() if v['phan'] is not None]
    fig, ax = figure('TN11 · Có bị giới hạn bởi số epoch?',
                     'Dấu tròn = checkpoint có J val thấp nhất · dấu vuông = epoch kết thúc · patience = 20',
                     size=(13, 7.5))
    for i, (name, v) in enumerate(rows):
        ax.hlines(i, v['epoch_tot'], v['so_epoch'], color=BLUE, lw=3, alpha=.5)
        ax.plot(v['epoch_tot'], i, 'o', color=BLUE, ms=7)
        ax.plot(v['so_epoch'], i, 's', color=ORANGE, ms=6)
        ax.text(v['so_epoch']+2, i, f'{v["epoch_tot"]} → {v["so_epoch"]}', va='center', fontsize=9)
    ax.axvspan(108, 120, color=RED, alpha=.07)
    ax.axvline(120, color=RED, ls='--')
    ax.set_xlim(0, 124)
    ax.set_yticks(range(len(rows)), [k for k, _ in rows])
    ax.invert_yaxis()
    ax.set_xlabel('Epoch · vùng 108–120 là vùng cảnh báo nếu chạy hết ngân sách')
    ax.grid(axis='x', alpha=.8)
    save(fig, out, '05_tn11_hoc_chua_toi',
         '0/12 mô hình bị cờ “học chưa tới”: không mô hình nào chạy đủ 120 epoch và tốt nhất từ epoch 108 trở đi.\n'
         'Điều này không loại trừ underfitting do kiến trúc/mục tiêu học. Cần thí nghiệm năng lực hoặc learning curve theo cỡ dữ liệu để kết luận thêm.')


def folds_plot(out, r, folds):
    source = {'train': 'train10_k4_5', 'train6': 'train6_10_k4_5'}
    for d, groups in folds['_nhom'].items():
        actual = defaultdict(set)
        for kb, name in groups.items():
            actual[name].add(folds[d][kb])
        assert all(len(s) == 1 for s in actual.values()), f'Group crosses folds: {d}'
    for x in r['kiem_tra_du_lieu']['x11']:
        d, kb = x['nguon']
        assert folds['x11_k4_5'][str(x['kb'])] == folds[source[d]][str(kb)]
    fig, (a, b) = figure('TN11 · Kiểm thử chéo theo nhóm, 4 fold',
                         'Cùng cơn bão / ngày ở cùng một fold; x11 đi cùng cửa sổ nguồn. Mỗi bản ghi được chấm ngoài fold một lần.',
                         ncols=2, size=(14, 7.5), gridspec_kw={'width_ratios': [1, 1.25]})
    a.imshow(np.eye(4), cmap=ListedColormap(['#dce9f7', ORANGE]), vmin=0, vmax=1, aspect='auto')
    for row in range(4):
        for col in range(4):
            a.text(col, row, 'GIỮ LẠI\nđể chấm' if row == col else 'HỌC', ha='center', va='center', fontsize=11)
    a.set_xticks(range(4), [f'Nhóm fold {f}' for f in range(4)])
    a.set_yticks(range(4), [f'Mô hình p{f}' for f in range(4)])
    a.set_title('Quy trình cho từng họ: DM11, DM10c, SM11', loc='left')
    a.tick_params(length=0)
    rows = []
    x = np.arange(4)
    for j, (d, label) in enumerate(GROUPS.items()):
        counts = Counter(folds[d].values())
        vals = [counts[f] for f in range(4)]
        bars = b.bar(x+(j-1)*.24, vals, width=.23, label=label, color=[BLUE, GREEN, ORANGE][j])
        b.bar_label(bars, padding=3, fontsize=9)
        rows.extend({'tap': d, 'fold': f, 'so_cua_so': counts[f]} for f in range(4))
    b.set_xticks(x, [f'Fold {f}' for f in range(4)])
    b.set_ylabel('Số cửa sổ thực tế (không phải số bản ghi độc lập)')
    b.set_ylim(0, 20)
    b.set_title('Phân bố cửa sổ từ phan.json', loc='left')
    b.legend(loc='upper right', fontsize=9)
    b.grid(axis='y', alpha=.6)
    save(fig, out, '06_tn11_cross_validation',
         'Validation riêng chọn checkpoint; fold giữ lại dùng đánh giá. Bộ test cuối riêng, 3 seed mỗi họ.\n'
         'Vẫn có 9/16 cửa sổ val6 kề ngày học và một số test kề ngày: trình bày giới hạn + phân tích độ nhạy, không gọi là hoàn toàn hết rò rỉ.')
    csv_out(out, '06_tn11_phan_fold', rows)


def fold_scores(out, tn11, folds, r):
    records = defaultdict(list)
    files = sorted((tn11 / 'cv_tep').glob('*.json'))
    for p in files:
        d = p.name.split('__')[0]
        for row in read(p):
            records[(d, folds[d][str(row['kb'])])].append(row)
    assert len(files) == sum(len(folds[d]) for d in GROUPS)
    fig, axes = figure('TN11 · Kết quả ngoài fold trên từng phần giữ lại',
                       'Đây là trung bình mô tả theo fold; không coi 4 fold là 4 mẫu độc lập để suy ra CI 95%',
                       ncols=3, size=(16, 7))
    rows = []
    for ax, (d, label) in zip(axes, GROUPS.items()):
        for h, color in [('dm11', BLUE), ('dm10c', GREEN), ('sm11', ORANGE)]:
            vals = []
            for f in range(4):
                rs = records[(d, f)]
                value = float(np.mean([x[h][0] for x in rs])) if rs else None
                vals.append(value if value is not None else np.nan)
                rows.append({'tap': d, 'fold': f, 'model': h, 'n': len(rs), 'mat_dv': value})
            ax.plot(range(4), vals, 'o-', color=color, label=h.upper())
        ax.set_title(label, loc='left')
        ax.set_xticks(range(4), [f'p{i}' for i in range(4)])
        ax.set_xlabel('Mô hình giữ lại fold tương ứng')
        ax.set_ylabel('Mất dịch vụ (%) · thấp hơn tốt hơn')
        ax.grid(alpha=.7)
        ax.legend()
    # Independent check that aggregation matches the saved primary summary.
    for (d, title) in zip(GROUPS, ('train (7 cơn bão)', 'train6 (60 ngày)', 'x11 (cực đoan)')):
        rs = [v for f in range(4) for v in records[(d, f)]]
        for h in ('dm11', 'dm10c', 'sm11'):
            assert np.isclose(np.mean([x[h][0] for x in rs]), r['kiem_thu_cheo'][title][h], atol=1e-10, rtol=0)
    save(fig, out, '07_tn11_ket_qua_tung_fold',
         'Các fold chứa bão/ngày và mức khó khác nhau nên không kỳ vọng cùng một mức mất DV.\n'
         'DM10c không học x11 ở bất kỳ fold nào; điểm trên x11 là đánh giá dữ liệu ngoài tập học của họ đối chứng này.')
    csv_out(out, '07_tn11_diem_tung_fold', rows)
    return files


def bootstrap_demo(out, tn11, r):
    fig, axes = figure('TN11 · Vì sao cần khoảng tin cậy?',
                       'Ví dụ c7wh: Lai(DM11) − Dijkstra phạt · trung bình 3 seed trước khi bootstrap',
                       ncols=2, size=(14, 7))
    rows = [x for p in sorted((tn11 / 'cham_tep').glob('c7wh_k4_5__*.json')) for x in read(p)]
    v = np.array([np.mean([x[f'dm11_lai_s{s}'][0] for s in range(3)])-x['dphat'][0] for x in rows])
    kb = np.array([x['kb'] for x in rows])
    unique = np.unique(kb)
    sums = np.array([v[kb == k].sum() for k in unique])
    counts = np.array([(kb == k).sum() for k in unique])
    idx = np.random.default_rng(0).integers(0, len(unique), (10000, len(unique)))
    boot = sums[idx].sum(1)/counts[idx].sum(1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    expected = r['kiem_tra']['c7wh']['hieu_vs_dphat']['Lai(DM11)']
    assert np.allclose([v.mean(), lo, hi], [expected[0], *expected[1]], atol=1e-10, rtol=0)
    a, b = axes
    means = sums/counts
    a.bar(unique, means, color=[GREEN if x < 0 else RED for x in means])
    a.axhline(0, color=GRAY, lw=1)
    a.set_xticks(unique)
    a.set_xlabel('Cửa sổ thời tiết (kb)')
    a.set_ylabel('Hiệu mất DV trung bình (điểm %)')
    a.set_title(f'{len(rows):,} tình huống, nhưng {len(unique)} cụm cửa sổ', loc='left')
    a.grid(axis='y', alpha=.7)
    b.hist(boot, bins=45, color=BLUE, alpha=.85)
    b.axvspan(lo, hi, color=GREEN, alpha=.10)
    for x in (lo, hi):
        b.axvline(x, color=GREEN, ls='--')
    b.axvline(0, color=RED, ls='--', label='Không chênh lệch')
    b.set_title(f'10.000 lần lấy lại cụm\nΔ = {v.mean():.2f}; CI [{lo:.2f}; {hi:.2f}]', loc='left')
    b.set_xlabel('Hiệu mất DV bootstrap (điểm %)')
    b.set_ylabel('Số lần')
    b.legend(fontsize=9)
    save(fig, out, '08_tn11_bootstrap',
         'Lấy mẫu lại cả cửa sổ có hoàn lại, giữ ghép cặp phương pháp; không lấy từng bản ghi như mẫu độc lập.\n'
         'CI này tái tạo đúng phương pháp báo cáo: theo cửa sổ, không theo cơn bão. Phụ thuộc giữa cửa sổ cùng bão vẫn là giới hạn.')
    csv_out(out, '08_tn11_hieu_tung_cua_so', [{'kb': int(k), 'n': int(n), 'delta': float(m)}
                                          for k, n, m in zip(unique, counts, means)])
    return {'mean': float(v.mean()), 'ci': [float(lo), float(hi)], 'n': len(rows), 'clusters': len(unique),
            'resamples': 10000, 'matched_report': True}


def sensitivity(out, r):
    tests = r['kiem_tra']
    bo = [b for b in NAMES if 'bo_ro_ri' in tests[b]]
    fig, axes = figure('TN11 · Kết luận có phụ thuộc cửa sổ kề ngày học?',
                       'So kết quả đầy đủ với kết quả loại 2 cửa sổ kề ngày học ở từng bộ liên quan',
                       ncols=2, size=(14, 7.5))
    rows = []
    for ax, key, clean, title in [(axes[0], None, 'Lai(DM11) − Dijkstra phạt', 'H11.3 · So với Dijkstra phạt'),
                                 (axes[1], 'dm11_tru_dm10c', 'Lai(DM11) − Lai(DM10c)', 'H11.1 · Lợi ích thêm dữ liệu cực đoan')]:
        for i, b in enumerate(bo):
            original = tests[b][key] if key else tests[b]['hieu_vs_dphat']['Lai(DM11)']
            cleaned = tests[b]['bo_ro_ri'][clean]
            for offset, val, color, label in [(-.13, original, BLUE, 'Đầy đủ'), (.13, cleaned, ORANGE, 'Bỏ kề ngày')]:
                m, (lo, hi) = val
                ax.hlines(i+offset, lo, hi, color=color, lw=2)
                ax.plot(m, i+offset, 'o', color=color, label=label if i == 0 else None)
                rows.append({'so_sanh': clean, 'bo': b, 'du_lieu': label, 'mean': m, 'low': lo, 'high': hi})
        ax.axvline(0, color=GRAY, ls='--')
        ax.set_yticks(range(len(bo)), [NAMES[b] for b in bo])
        ax.invert_yaxis()
        ax.margins(y=.15, x=.12)
        ax.set_title(title, loc='left')
        ax.set_xlabel('Hiệu mất DV (điểm %) · CI 95%')
        ax.grid(axis='x', alpha=.7)
        ax.legend(fontsize=9)
    save(fig, out, '09_tn11_do_nhay_ro_ri',
         'H11.3 giữ kết luận trên các bộ cực đoan liên quan. Lợi ích H11.1 ở xc11 mất ý nghĩa khi bỏ cửa sổ kề ngày.\n'
         'Đây là phân tích lại điểm test đã có; không phải huấn luyện lại sau khi làm sạch val6.')
    csv_out(out, '09_tn11_do_nhay', rows)


def interactive(out, r):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    fig = make_subplots(rows=1, cols=2, subplot_titles=('Loss train / val', 'J val · chọn checkpoint'))
    options = list(r['hoc'].items())
    for i, (name, h) in enumerate(options):
        nk = np.array(h['nk'])
        for col, idx, label, color in [(1, 1, 'Loss train', BLUE), (1, 2, 'Loss val', ORANGE), (2, 3, 'J val', GREEN)]:
            fig.add_trace(go.Scatter(x=nk[:, 0], y=nk[:, idx], mode='lines', name=label,
                                    visible=i == 0, line={'color': color}), row=1, col=col)
    def title(name, h):
        return f'TN11 · {name} | checkpoint {h["epoch_tot"]} / kết thúc {h["so_epoch"]} | val/train {h["ti_le_val_train"]:.2f}'
    def shapes(h):
        return [{'type': 'line', 'xref': ref, 'yref': domain, 'x0': h['epoch_tot'], 'x1': h['epoch_tot'],
                 'y0': 0, 'y1': 1, 'line': {'color': RED, 'dash': 'dash'}}
                for ref, domain in [('x', 'y domain'), ('x2', 'y2 domain')]]
    buttons = []
    for i, (name, h) in enumerate(options):
        buttons.append({'label': name, 'method': 'update',
                        'args': [{'visible': [j//3 == i for j in range(3*len(options))]},
                                 {'title.text': title(name, h), 'shapes': shapes(h),
                                  'xaxis.autorange': True, 'xaxis2.autorange': True,
                                  'yaxis.autorange': True, 'yaxis2.autorange': True}]})
    fig.update_layout(template='plotly_white', height=600, title=title(*options[0]), shapes=shapes(options[0][1]),
                      margin={'t': 150, 'b': 70}, font={'family': 'Arial', 'size': 13},
                      updatemenus=[{'buttons': buttons, 'x': 0, 'y': 1.25, 'xanchor': 'left', 'yanchor': 'top'}],
                      legend={'orientation': 'h', 'y': -.15})
    fig.update_xaxes(title_text='Epoch')
    fig.write_html(out / 'duong_hoc_tuong_tac.html', include_plotlyjs=True, full_html=True,
                   config={'displaylogo': False, 'toImageButtonOptions': {'format': 'svg', 'filename': 'tn11_duong_hoc'}})


CAPTIONS = [
    ('01_tn6_thong_ke', 'TN6: từ một tuyến đẹp đến bằng chứng trên nhiều tình huống',
     'Chỉ rõ CI cắt 0 ở một số ô tải cao. Không suy từ bản đồ một thời điểm rằng DM luôn thắng. τ của kết quả TN6 gốc là 1,2.'),
    ('02_tn11_hieu_va_ci', 'TN11: xem hiệu và khoảng tin cậy',
     'H11.1 chưa đạt; H11.2 và H11.3 đạt. CI trên hình là từng phép so sánh, chưa hiệu chỉnh Holm; điều kiện Holm xem báo cáo gốc.'),
    ('03_tn11_duong_hoc', 'Overfitting: xem diễn biến qua epoch',
     'Mở trang tương tác để đổi giữa 21 mô hình. Đánh dấu checkpoint chọn bằng J val; không chọn bằng điểm test.'),
    ('04_tn11_overfitting', 'Overfitting: hai dấu hiệu độc lập về cách đo',
     '3 mô hình SM11 vượt tỷ lệ loss 1,25; gap trung bình SM11/x11 vượt 1 điểm %. Ngưỡng là quy ước của TN11.'),
    ('05_tn11_hoc_chua_toi', 'Underfitting: nói đúng mức bằng chứng',
     'Không bị cờ thiếu epoch không đồng nghĩa không underfitting. Không vẽ một đường underfitting giả rồi gọi đó là kết quả TN11.'),
    ('06_tn11_cross_validation', 'Cross-validation: ai học nhóm nào?',
     'Mỗi fold là nhóm bão/ngày. Các cửa sổ nguồn và bản cực đoan x11 ở cùng fold. Giữ val riêng và test cuối riêng.'),
    ('07_tn11_ket_qua_tung_fold', 'Cross-validation: kết quả thật của từng fold',
     'Các fold có mức khó khác nhau. Điểm được lấy từ mô hình không học fold đó; không dùng sai số giữa 4 fold làm CI thống kê.'),
    ('08_tn11_bootstrap', 'Thống kê: lấy mẫu lại theo cửa sổ',
     'Giữ ghép cặp phương pháp, trung bình 3 seed trước, lấy lại 13 cửa sổ 10.000 lần. CI tái tạo được từ cache, không lấy mẫu lại từng bản ghi.'),
    ('09_tn11_do_nhay_ro_ri', 'Độ nhạy: bỏ cửa sổ kề ngày học',
     'Cho thấy kết quả nào vững và kết quả nào phụ thuộc dữ liệu kề ngày. val6 kề ngày học vẫn là giới hạn của lần chạy này.'),
]


def gallery(out):
    cards = []
    for stem, title, text in CAPTIONS:
        picture = base64.b64encode((out / f'{stem}.png').read_bytes()).decode('ascii')
        cards.append(f'<section id="{stem}"><h2>{html.escape(title)}</h2><p>{html.escape(text)}</p>'
                     f'<a href="{stem}.png">PNG</a> · <a href="{stem}.svg">SVG</a>'
                     f'<img src="data:image/png;base64,{picture}" alt="{html.escape(title)}"></section>')
    toc = ''.join(f'<a href="#{stem}">{i+1:02d} · {html.escape(title)}</a>' for i, (stem, title, _) in enumerate(CAPTIONS))
    content = '''<!doctype html><html lang="vi"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>TN6 & TN11 · Bằng chứng thống kê</title><style>
body{margin:0;background:#eef2f6;color:#203047;font:16px/1.65 system-ui,sans-serif}main{max-width:1250px;margin:auto;padding:36px 24px}
header,section,nav{background:white;border-radius:16px;padding:28px;margin-bottom:22px;box-shadow:0 3px 20px #20304708}
h1{font-size:36px;line-height:1.2;margin:12px 0}h2{font-size:24px;line-height:1.35}img{display:block;width:100%;height:auto;margin-top:18px}
a{color:#2563a6}nav a{display:block;padding:5px}.tag{color:#12836a;font-weight:700;letter-spacing:2px;font-size:13px}
.button{display:inline-block;background:#2563a6;color:white;padding:10px 16px;border-radius:8px;text-decoration:none;margin-right:10px}
.note{border-left:4px solid #d07828;padding:12px 18px;background:#fff8ef}section{scroll-margin-top:20px}
@media print{nav,.button{display:none}section{break-before:page;box-shadow:none}body{background:white}main{padding:0}}
</style><main><header><span class="tag">PBL4 · DEMO KẾT QUẢ ĐÃ CHẠY</span>
<h1>Từ tuyến minh họa<br>đến bằng chứng thống kê</h1>
<p>TN6 trên nhánh demo và TN11 trên máy nhóm. Hình từ số liệu thật; không huấn luyện lại.</p>
<a class="button" href="duong_hoc_tuong_tac.html">Chọn mô hình & xem đường học</a>
<a href="HUONG_DAN_DEMO.md">Kịch bản thuyết trình & giới hạn</a>
<p class="note">TN6 gốc: chưa có nhật ký DM6 và kết quả 4-fold trong gói demo đã kiểm tra.
Không gắn đường học DM của TN4 hoặc nhật ký DM6 huấn luyện lại ở TN8 thành bằng chứng của TN6 gốc.</p>
</header><nav>''' + toc + '</nav>' + ''.join(cards) + '</main></html>'
    (out / 'index.html').write_text(content, encoding='utf-8')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, default=KQ / 'demo_thong_ke')
    ap.add_argument('--tn11', type=Path, default=KQ / 'tn11')
    ap.add_argument('--tn6', type=Path, default=KQ / 'demo_thong_ke' / 'nguon' / 'tn6_demo_ket_qua.json')
    a = ap.parse_args()
    out = a.out
    out.mkdir(parents=True, exist_ok=True)
    (out / 'csv').mkdir(exist_ok=True)
    (out / 'nguon').mkdir(exist_ok=True)
    r, r6, folds = read(a.tn11 / 'ket_qua_11.json'), read(a.tn6), read(a.tn11 / 'phan.json')
    histories = []
    for name, h in r['hoc'].items():
        path = a.tn11.parent / 'mo_hinh11' / f'{name}_nhat_ky.json'
        history = read(path)['nhat_ky']
        best = min(history, key=lambda row: row['J_val'])
        assert len(history) == h['so_epoch'] and best['epoch'] == h['epoch_tot']
        assert np.isclose(best['J_val'], h['J_val_tot'], atol=1e-10, rtol=0)
        h['nk'] = [[row[k] for k in ('epoch', 'loss_train', 'loss_val', 'J_val')] for row in history]
        histories.append(path)
    tn6_plot(out, r6)
    tn11_forest(out, r)
    curves(out, r)
    fit_diagnostics(out, r)
    underfit(out, r)
    folds_plot(out, r, folds)
    cvfiles = fold_scores(out, a.tn11, folds, r)
    bs = bootstrap_demo(out, a.tn11, r)
    sensitivity(out, r)
    interactive(out, r)
    gallery(out)
    for name in ('ket_qua_11.json', 'phan.json', 'kiem_tra_du_lieu.json'):
        shutil.copy2(a.tn11 / name, out / 'nguon' / name)
    source_files = [Path(__file__), a.tn6, a.tn11 / 'ket_qua_11.json', a.tn11 / 'phan.json',
                    *histories, *cvfiles, *sorted((a.tn11 / 'cham_tep').glob('c7wh_k4_5__*.json'))]
    manifest = {'checks': {'group_folds_disjoint': True, 'x11_matches_source_fold': True,
                           'cv_aggregates_match_report': True, 'bootstrap': bs},
                'sources_sha256': {str(p.relative_to(ROOT)).replace('\\', '/'): hashlib.sha256(p.read_bytes()).hexdigest()
                                   for p in source_files},
                'figures': [stem for stem, _, _ in CAPTIONS],
                'notes': ['Original TN6 demo result is separate from local DM6 retraining logs.',
                          'CIs are window-cluster bootstrap, not storm-cluster bootstrap.',
                          'Figures preserve the saved hypotheses and do not retrain or reselect checkpoints.']}
    write_json(out / 'kiem_chung.json', manifest)
    csv_out(out, 'duong_hoc_tat_ca', [{'model': name, 'epoch': row[0], 'loss_train': row[1],
                                     'loss_val': row[2], 'J_val': row[3], 'epoch_best': h['epoch_tot']}
                                    for name, h in r['hoc'].items() for row in h['nk']])
    # Only package this report, never the original results or training data.
    archive = out.parent / 'TN6_TN11_demo_thong_ke.zip'
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.rglob('*')):
            if p.is_file():
                z.write(p, Path(out.name) / p.relative_to(out))
        z.write(Path(__file__), 'xuat_demo_thong_ke.py')
    print(f'OK: {len(CAPTIONS)} PNG + SVG; interactive histories; CSV; checks passed.\n{out}\n{archive}')


if __name__ == '__main__':
    main()
