"""Bảng tra 3000 ô học từ các cơn train, áp dụng online bằng quan sát quá khứ.

CLI đếm lại sau khi chặn nút đã tắt trong mạng dự báo, học và đánh giá theo cơn.
"""
from __future__ import annotations
import argparse
import gc
import json
from pathlib import Path
import numpy as np
from src.ngau_nhien.bao_manh import overlaps_yagi
from src.ngau_nhien.du_bao import du_bao_quy_dao
from src.ngau_nhien.kich_ban import MangGoc, ThamSo, sinh_kich_ban, tham_so_dict
from src.ngau_nhien.khoang_loi import MUC
from src.ngau_nhien.thoi_tiet import ThoiTiet
from src.ngau_nhien.tao_du_lieu_bao import sha256

SHAPE = (5, 10, 3, 5, 2, 2)
NCELLS = int(np.prod(SHAPE))
RAIN_BINS = np.array([.5, 2, 5, 8, 11, 15, 20, 30, 50])
DEFAULT_VAL = ('prapiroon_2024', 'trami_2024', 'kalmaegi_2025')
ROOT = Path(__file__).resolve().parents[2]


def pham_vi_ap_dung():
    """Hợp đồng sử dụng bảng; parameters trong manifest vẫn là cấu hình đầy đủ."""
    return {
        'scope': 'weather_only_with_fixed_background_load',
        'required_flags': {'xung_dot': False, 'tai': False},
        'parameter_matching': 'Match parameters, geometry, observation age and horizon in meta.json.',
        'distribution_shift': 'Recount and refit before claiming calibrated risk with conflicts or dynamic load enabled.',
        'target': 'Link availability at decision t and throughout t+1..t+delta; not an SLA or congestion-failure probability.',
        'node_shutdown': 'Known-off nodes at observation are blocked; shutdowns after observation may cause a failure label.',
        'prior_selection': 'Fixed prior=20; not selected on validation; table fits storm_split.train only.',
    }


def hop_dong_bang(meta):
    """Phần metadata nhúng trong NPZ, không chứa hash của chính NPZ."""
    keys = ('schema', 'parameters', 'applicability', 'tuoi', 'delta', 'dt_seconds')
    if not all(k in meta for k in keys):
        raise ValueError('Thiếu metadata cấu hình/phạm vi của bảng; cần bản xuất có hợp đồng')
    return {k: meta[k] for k in keys}


def kiem_tra_cau_hinh(tham_so, meta):
    """Báo lỗi nếu ThamSo thực tế khác cấu hình đếm; không tự sửa cấu hình."""
    if not isinstance(tham_so, ThamSo):
        raise TypeError('tham_so phải là ThamSo thực tế truyền cho sinh_kich_ban')
    contract = hop_dong_bang(meta)
    expected = contract['parameters']
    actual = tham_so_dict(tham_so)
    if contract['schema'] != 'storm-lookup-v1' or not isinstance(expected, dict):
        raise ValueError('Schema/cấu hình bảng không được hỗ trợ')
    app = contract['applicability']
    if (not isinstance(app, dict) or app.get('scope') != 'weather_only_with_fixed_background_load'
            or app.get('required_flags') != {'xung_dot': False, 'tai': False}):
        raise ValueError('applicability thiếu hoặc không thuộc phạm vi được hỗ trợ')
    bad = [k for k in sorted(set(expected) | set(actual)) if expected.get(k) != actual.get(k)]
    bad += [k for k, v in app['required_flags'].items() if actual[k] != v or expected.get(k) != v]
    if contract['dt_seconds'] != actual['dt_s'] or actual['dt_s'] != 20:
        bad.append('dt_seconds/dt_s')
    if bad:
        raise ValueError('Cấu hình khác bảng ở: ' + ', '.join(sorted(set(bad))) + '; cần đếm/học lại')


def _hop_dong_nhung(bang):
    try:
        contract = json.loads(str(np.asarray(bang['contract_json']).item()))
    except (KeyError, ValueError, TypeError) as exc:
        raise ValueError('Bảng thiếu/hỏng contract_json; cần dùng bản xuất mới có cấu hình') from exc
    if not isinstance(contract, dict):
        raise ValueError('contract_json phải là object')
    contract = hop_dong_bang(contract)
    if any(k not in bang or int(bang[k]) != contract[k] for k in ('tuoi', 'delta')):
        raise ValueError('Tuổi thông tin/horizon không khớp hợp đồng nhúng')
    return contract


def o_dac_trung(g, q, observed_at, history=45):
    """Đọc duy nhất t_obs và t_obs−15phút; có đủ lịch sử mới tạo đặc trưng."""
    if observed_at < history:
        raise ValueError('Cần đủ 15 phút lịch sử trước quan sát')
    site = np.maximum(g.site, 0)
    rain = np.where(g.site >= 0, q['R'][observed_at, site], 0)
    previous = np.where(g.site >= 0, q['R'][observed_at - history, site], 0)
    trend = np.where(rain - previous < -1, 0, np.where(rain - previous > 1, 2, 1))
    kinds = np.array(g.kind)
    node_site = np.maximum(g.node_site, 0)
    is_uav = kinds == 'uav'
    is_ground = np.isin(kinds, ['gateway', 'user'])
    gust = np.where(is_uav & (g.node_site >= 0), q['gust'][observed_at, node_site], 0)
    wind = np.maximum(gust[:, None], gust[None, :])
    has_uav = is_uav[:, None] | is_uav[None, :]
    wind_bin = np.where(has_uav, np.where(wind < 10, 1, np.where(wind < 13, 2, np.where(wind <= 15, 3, 4))), 0)
    thunder = q['giong'][observed_at, node_site] & (g.node_site >= 0)
    tu, tg = thunder & is_uav, thunder & is_ground
    band = np.searchsorted([0, 5, 12, 20, 28], g.band)
    if not np.isin(g.band, [0, 5, 12, 20, 28]).all():
        raise ValueError('Băng tần không thuộc schema')
    return np.ravel_multi_index((band, np.digitize(rain, RAIN_BINS), trend, wind_bin,
                                 (tu[:, None] | tu[None, :]).astype(int),
                                 (tg[:, None] | tg[None, :]).astype(int)), SHAPE)


def hoc_bang(song, n, names, val_names=DEFAULT_VAL, prior=20.0):
    """Làm trơn ô -> băng×mưa -> băng; tuyệt đối không lấy số đếm validation."""
    song, n = np.asarray(song), np.asarray(n)
    names = list(map(str, names)); val_names = list(val_names)
    if len(set(names)) != len(names) or any('yagi' in s.lower() for s in names):
        raise ValueError('Danh sách cơn phải duy nhất và không có Yagi')
    if not val_names or not set(val_names) <= set(names) or len(set(val_names)) != len(val_names):
        raise ValueError('Cần danh sách val hợp lệ, không trùng')
    tr = np.array([s not in val_names for s in names])
    if not tr.any() or song.shape != n.shape or n.shape != (len(names), NCELLS):
        raise ValueError('Số đếm/split không hợp lệ')
    if not np.isfinite(song).all() or not np.isfinite(n).all() or np.any(song < 0) or np.any(song > n) or not np.isfinite(prior) or prior <= 0:
        raise ValueError('Số đếm hoặc prior không hợp lệ')
    s, count = song[tr].sum(0).reshape(SHAPE), n[tr].sum(0).reshape(SHAPE)
    sb, nb = s.sum((1, 2, 3, 4, 5)), count.sum((1, 2, 3, 4, 5))
    pband = (sb + 1) / (nb + 2)
    sbr, nbr = s.sum((2, 3, 4, 5)), count.sum((2, 3, 4, 5))
    pbr = (sbr + prior * pband[:, None]) / (nbr + prior)
    p = (s + prior * pbr[:, :, None, None, None, None]) / (count + prior)
    return {'p_song': p.reshape(-1), 'n_train': count.reshape(-1), 'song_train': s.reshape(-1),
            'n_band': nb, 'n_band_rain': nbr,
            'train_storms': [v for v, keep in zip(names, tr) if keep], 'val_storms': val_names, 'prior': prior}


def ma_tran_online(bang, g, W_observed, q_observed, decision_at, tuoi=9, delta=3, *, tham_so):
    """W_observed và q_observed chỉ cần lịch sử đến t−tuoi; không nhận W thực tương lai.

    Trả risk[N,N], mask cạnh hợp lệ, support số mẫu ô, cấp fallback và W dự báo.
    Cạnh không dùng được: NaN. Cấp 0: băng chưa từng thấy, prior 0.5, không coi là đã học.
    """
    kiem_tra_cau_hinh(tham_so, _hop_dong_nhung(bang))
    if tuoi < 0:
        raise ValueError('tuoi phải >= 0')
    if 'tuoi' in bang and (int(bang['tuoi']) != tuoi or int(bang['delta']) != delta):
        raise ValueError('Tuổi thông tin/horizon không khớp bảng; cần đếm và học lại')
    o = decision_at - tuoi
    cells = o_dac_trung(g, q_observed, o)
    Wq = du_bao_quy_dao(g.W, W_observed, o, decision_at, delta, q_observed['tat'][o])
    mask = Wq > 0
    risk = np.full(Wq.shape, np.nan, np.float32)
    risk[mask] = 1 - bang['p_song'][cells[mask]]
    support = np.where(mask, bang['n_train'][cells], 0)
    band, rain = np.unravel_index(cells, SHAPE)[:2]
    level = np.where(bang['n_band'][band] > 0, 1, 0)
    level = np.where(bang['n_band_rain'][band, rain] > 0, 2, level)
    level = np.where(support > 0, 3, level)
    level = np.where(mask, level, -1).astype(np.int8)
    return {'risk_edge': risk, 'valid': mask, 'cell_id': cells, 'support': support, 'support_level': level, 'W_forecast': Wq}


def nap_bang(path, *, tham_so):
    """Bắt buộc sidecar meta.json, hash đúng, hợp đồng nhúng khớp và cấu hình thật."""
    path = Path(path)
    meta = json.loads(path.with_name('meta.json').read_text(encoding='utf-8'))
    kiem_tra_cau_hinh(tham_so, meta)
    if sha256(path) != meta.get('table_sha256'):
        raise ValueError('Hash bảng không khớp meta.json')
    with np.load(path, allow_pickle=False) as z:
        bang = {k: z[k] for k in z.files}
    if _hop_dong_nhung(bang) != hop_dong_bang(meta):
        raise ValueError('Cấu hình nhúng trong bảng không khớp meta.json')
    return bang


def metrics_rui_ro(risk, failures, n, *, p_baseline=None):
    """Chỉ số theo nhóm (điểm rủi ro, số đứt, số mẫu), không bung từng mẫu.

    AUPRC là average precision cho lớp đứt, gộp đồng điểm trước khi tính recall.
    p_baseline là xác suất SỐNG cố định học từ train; không suy ra từ nhóm chấm.
    """
    risk, failures, n = (np.asarray(v, dtype=np.float64) for v in (risk, failures, n))
    if risk.shape != failures.shape or risk.shape != n.shape:
        raise ValueError('Điểm/số đứt/số mẫu phải cùng kích thước')
    if (not all(np.isfinite(v).all() for v in (risk, failures, n))
            or np.any((risk < 0) | (risk > 1))
            or np.any((failures < 0) | (failures > n))
            or np.any(n != np.floor(n)) or np.any(failures != np.floor(failures))):
        raise ValueError('Điểm phải trong [0,1], số đếm nguyên không âm và hữu hạn')
    if p_baseline is not None:
        if np.ndim(p_baseline) != 0 or not np.isfinite(p_baseline) or not 0 <= p_baseline <= 1:
            raise ValueError('p_baseline phải là xác suất sống cố định trong [0,1]')
        p_baseline = float(p_baseline)
    risk, failures, n = (v.ravel() for v in (risk, failures, n))
    total = float(n.sum())
    if not np.isfinite(total):
        raise ValueError('Tổng số mẫu không hữu hạn')
    if not total:
        return dict(samples=0, failures=0, survival_rate=None, failure_rate=None,
                    brier=None, log_loss=None, brier_baseline=None, bss=None,
                    auprc=None, auprc_baseline=None)
    failed = float(failures.sum()); survived = n - failures
    w_fail, w_survive = failures/total, survived/total
    brier = float((w_fail*(1-risk)**2 + w_survive*risk**2).sum())
    # Chỉ clip log-loss; Brier phải bằng 0 thật khi dự đoán hoàn hảo.
    log_risk = np.clip(risk, 1e-8, 1-1e-8)
    log_loss = float(-(w_fail*np.log(log_risk) + w_survive*np.log1p(-log_risk)).sum())
    bb = None if p_baseline is None else float(
        (failed/total)*p_baseline**2 + ((total-failed)/total)*(1-p_baseline)**2)
    bss = None if bb is None or bb == 0 else 1-brier/bb
    if bss is not None and not np.isfinite(bss):
        bss = None
    ap = None
    if failed:
        present = n > 0
        order = np.argsort(-risk[present], kind='stable')
        scores = risk[present][order]
        # Một ngưỡng cho toàn bộ nhóm có điểm bằng nhau, không phụ thuộc thứ tự.
        ends = np.r_[np.flatnonzero(scores[:-1] != scores[1:]), len(scores)-1]
        cum_failed = np.cumsum(failures[present][order])[ends]
        cum_n = np.cumsum(n[present][order])[ends]
        ap = float(np.sum((cum_failed/cum_n)*np.diff(np.r_[0, cum_failed]))/failed)
    return dict(samples=int(total), failures=int(failed), survival_rate=(total-failed)/total,
                failure_rate=failed/total, brier=brier, log_loss=log_loss,
                brier_baseline=bb, bss=bss,
                auprc=ap, auprc_baseline=failed/total)


def metrics(p, s, n, *, p_baseline=None):
    """API cũ theo xác suất sống/số sống; các khóa cũ được giữ nguyên."""
    p, s, n = (np.asarray(v, dtype=np.float64) for v in (p, s, n))
    if p.shape != s.shape or p.shape != n.shape:
        raise ValueError('Xác suất/số sống/số mẫu phải cùng kích thước')
    return metrics_rui_ro(1-p, n-s, n, p_baseline=p_baseline)


def run(catalog, raw, split, out, tuoi=9, delta=3, stride=3, muc='khắc nghiệt', val_names=DEFAULT_VAL):
    if tuoi < 0 or delta < 1 or stride < 1:
        raise ValueError('tuoi>=0, delta>=1, stride>=1')
    catalog, raw, split, out = map(Path, (catalog, raw, split, out))
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'Từ chối ghi đè {out}')
    manifest = json.loads(catalog.read_text(encoding='utf-8'))
    storms = manifest['bao']; names = [b['ten'] for b in storms]
    # Validate split trước mô phỏng tốn thời gian.
    hoc_bang(np.zeros((len(names), NCELLS)), np.zeros((len(names), NCELLS)), names, val_names)
    with np.load(raw, allow_pickle=False) as z:
        W, kinds, node_names = z['W_snapshots'], z['node_kind'], z['node_order']
    meta_split = json.loads(split.read_text(encoding='utf-8'))
    if Path(meta_split['input'].replace('\\', '/')).name != raw.name:
        raise ValueError('split không thuộc raw đang dùng')
    ids = np.array(meta_split['snapshot_ids']['train'], int)
    ids = ids[(ids >= tuoi + 45) & (ids + delta < len(W))][::stride]
    if not len(ids):
        raise ValueError('Không có mốc train đủ lịch sử và tương lai')
    params = ThamSo(**{**MUC[muc], 'xung_dot': False, 'tai': False})
    successes = np.zeros((len(storms), NCELLS), np.int64); counts = np.zeros_like(successes)
    config = {'schema': 'storm-lookup-v1', 'tuoi': tuoi, 'delta': delta, 'dt_seconds': 20, 'history_steps': 45,
              'applicability': pham_vi_ap_dung(),
              'muc': muc, 'parameters': tham_so_dict(params), 'stride': stride, 'snapshots_used': ids.tolist(),
              'raw_sha256': sha256(raw), 'split_sha256': sha256(split), 'catalog_sha256': sha256(catalog),
              'storm_split': {'train': [s for s in names if s not in val_names], 'val': list(val_names)},
              'label': 'survive at decision t AND all t+1..t+delta; candidates visible in corrected forecast',
              'features': 'observation at t-age and 15-minute historical rain trend only',
              'cases': []}
    out.mkdir(parents=True, exist_ok=True)
    seen_times = set()
    for j, b in enumerate(storms):
        path = catalog.parent / b['tep']; tt = ThoiTiet(path)
        if overlaps_yagi(tt.time_utc) or 'yagi' in b['ten'].lower():
            raise ValueError('Không dùng Yagi để xây bảng tra')
        if b.get('sha256') != sha256(path):
            raise ValueError(f'Weather thay đổi sau khi lập danh mục: {path}')
        all_times = set(map(str, tt.time_utc))
        if seen_times & all_times:
            raise ValueError('Các cơn dùng chung giờ thời tiết; cần sửa catalog trước khi chia')
        seen_times |= all_times
        g = MangGoc(W, kinds, node_names, tt)
        for k, h0 in enumerate(b['gio_bat_dau_kich_ban']):
            last = h0 + int(np.floor((len(W)-1)*20/3600)) + 1
            if h0 < 0 or last >= tt.H:
                raise ValueError('Kịch bản không đủ thời tiết cho nội suy')
            seed = 100000 + 1000*j + k
            wn, q = sinh_kich_ban(g, tt, h0, seed, params)
            for t in ids:
                o = int(t-tuoi)
                pred = du_bao_quy_dao(W, wn[o], o, int(t), delta, q['tat'][o])
                cells = o_dac_trung(g, q, o)
                valid = pred > 0
                survived = (wn[t:t+delta+1] > 0).all(0)
                counts[j] += np.bincount(cells[valid], minlength=NCELLS)
                successes[j] += np.bincount(cells[valid & survived], minlength=NCELLS)
            config['cases'].append({'storm': b['ten'], 'h0': h0, 'seed': seed})
            del wn,q
            gc.collect()
            print(f"{b['ten']} {k+1}/{len(b['gio_bat_dau_kich_ban'])}: {counts[j].sum():,} mẫu", flush=True)
        del g
        gc.collect()
        # Checkpoint số đếm, chưa tuyên bố hoàn tất cho tới khi có meta.json.
        np.savez_compressed(out/'dem.npz', song=successes, n=counts, ten_bao=np.array(names))
    bang = hoc_bang(successes, counts, names, val_names)
    bang.update(tuoi=tuoi, delta=delta,
                contract_json=json.dumps(hop_dong_bang(config), ensure_ascii=False, sort_keys=True))
    np.savez_compressed(out/'bang_tra.npz', **{k: np.asarray(v) for k,v in bang.items()})
    from src.ngau_nhien.bao_cao_rui_ro import bao_cao_validation
    validation = bao_cao_validation({'song': successes, 'n': counts, 'ten_bao': names}, bang, val_names)
    (out/'validation.json').write_text(json.dumps(validation, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    config['samples'] = int(counts.sum()); config['table_sha256'] = sha256(out/'bang_tra.npz')
    config['limitations'] = ['Correlated directed edges and overlapping scenarios are not independent samples.',
                             'Fixed v5 geometry retains pre-existing ISL limitations; weather is gridded model data.',
                             '3000 cells do not model all geometry/link-margin differences; probabilities require validation.']
    (out/'meta.json').write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8')
    return config


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--catalog', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--raw', default=str(ROOT/'dataset/oracle_dataset_v5.npz'))
    ap.add_argument('--split', default=str(ROOT/'dataset/v5/split_info.json'))
    ap.add_argument('--tuoi', type=int, default=9); ap.add_argument('--delta', type=int, default=3)
    ap.add_argument('--stride', type=int, default=3); ap.add_argument('--muc', choices=list(MUC), default='khắc nghiệt')
    ap.add_argument('--val', nargs='+', default=list(DEFAULT_VAL))
    a = ap.parse_args()
    result = run(a.catalog,a.raw,a.split,a.out,a.tuoi,a.delta,a.stride,a.muc,a.val)
    print(f"Hoàn tất: {result['samples']:,} mẫu", flush=True)
