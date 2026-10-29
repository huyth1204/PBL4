"""Cắt 10 cơn ngoài Yagi từ weather 74 điểm theo Du_lieu_bao_manh.docx.

python -m src.ngau_nhien.bao_manh --out /workspace/pbl4-data/bao-manh
Không tải hoặc tự nhận đã kiểm chứng nguồn khí tượng; giữ provenance tệp đầu vào.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from src.ngau_nhien.tao_du_lieu_bao import sha256

ROOT = Path(__file__).resolve().parents[2]
STORMS = (
    ('prapiroon_2024', '2024-07-21T16:00', '2024-07-24T00:00'),
    ('soulik_2024', '2024-09-17T18:00', '2024-09-21T22:00'),
    ('trami_2024', '2024-10-26T07:00', '2024-10-27T19:00'),
    ('wutip_2025', '2025-06-10T20:00', '2025-06-13T19:00'),
    ('wipha_2025', '2025-07-19T17:00', '2025-07-23T00:00'),
    ('kajiki_2025', '2025-08-24T12:00', '2025-08-26T06:00'),
    ('nongfa_2025', '2025-08-29T15:00', '2025-08-30T20:00'),
    ('bualoi_2025', '2025-09-26T14:00', '2025-09-30T04:00'),
    ('matmo_2025', '2025-10-06T08:00', '2025-10-07T08:00'),
    ('kalmaegi_2025', '2025-11-06T00:00', '2025-11-07T06:00'),
)


def overlaps_yagi(times):
    t = np.asarray(times, dtype='datetime64[m]')
    return bool(np.any((t >= np.datetime64('2024-09-04')) & (t < np.datetime64('2024-09-13'))))


def tach(weather, out):
    weather, out = Path(weather), Path(out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Thư mục đã có dữ liệu: {out}")
    with np.load(weather, allow_pickle=False) as z:
        data = {k: z[k] for k in z.files}
    times = data['time_utc'].astype('datetime64[m]')
    manifest = {'schema': 'bao-manh-cuts-v1', 'source_sha256': sha256(weather),
                'source_path': str(weather.resolve()), 'catalog_source': 'Du_lieu_bao_manh.docx supplied by user',
                'bao': []}
    out.mkdir(parents=True, exist_ok=True)
    for name, start, end in STORMS:
        idx = np.flatnonzero((times >= np.datetime64(start)) & (times <= np.datetime64(end)))
        expected = int((np.datetime64(end) - np.datetime64(start)) / np.timedelta64(1, 'h')) + 1
        if len(idx) != expected or overlaps_yagi(times[idx]) or not np.all(np.diff(times[idx]) == np.timedelta64(1, 'h')):
            raise ValueError(f"Khoảng thời tiết thiếu/không hợp lệ hoặc chạm Yagi: {name}")
        sliced = {k: (v[idx] if v.ndim >= 1 and len(v) == len(times) else v) for k, v in data.items()}
        sliced['nguon'] = np.array(f"{str(data['nguon'])}; {name}; {start}..{end} UTC")
        path = out / f'{name}.npz'
        np.savez_compressed(path, **sliced)
        strong = (sliced['precip'] > 11).mean(1) >= .05
        strong |= (sliced['gust'] > 15).mean(1) >= .20
        # 600*20s, cần thêm giờ thứ 4 để nội suy. Cửa sổ cách nhau 3h và có giờ mạnh.
        starts = [h for h in range(0, len(idx) - 4, 3) if strong[h:h + 5].any()]
        manifest['bao'].append({'ten': name, 'tep': path.name, 'start_utc': start, 'end_utc': end,
                                'sha256': sha256(path), 'gio_bao': int(strong.sum()),
                                'gio_bat_dau_kich_ban': starts})
    (out / 'danh_muc.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    return manifest


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--weather', default=str(ROOT / 'data/thoi_tiet_vn_mr.npz'))
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    m = tach(a.weather, a.out)
    print(json.dumps({b['ten']: len(b['gio_bat_dau_kich_ban']) for b in m['bao']}, indent=2))
