/* Pure statistics, shared by the offline browser worker and Node verification. */
(function (root) {
  'use strict';
  const metrics = ['mat_dv_%', 'J', 'khong_tuyen_song_%', 'du_phong_qua_tai_%', 'tre_chinh_ms'];
  function rng32(seed) {
    let a = seed >>> 0;
    return function () {
      a = (a + 0x6D2B79F5) >>> 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t ^= t + Math.imul(t ^ (t >>> 7), 61 | t);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  function quantile(sorted, p) {
    const x = (sorted.length - 1) * p, lo = Math.floor(x);
    return sorted[lo] + (sorted[Math.ceil(x)] - sorted[lo]) * (x - lo);
  }
  function select(dataset, config) {
    const i = metrics.indexOf(config.metric);
    if (i < 0) throw Error('Thước đo không hợp lệ.');
    if (config.a === config.b) throw Error('Chọn hai phương pháp khác nhau.');
    const groups = new Map();
    function value(cell, family) {
      const keys = dataset.kind === 'test' && !['dphat', 'yen'].includes(family)
        ? (config.seed === 'all' ? dataset.seeds : [Number(config.seed)]).map(s => `${family}_s${s}`)
        : [family];
      if (keys.some(k => !cell.values[k])) throw Error('Không có kết quả cho phương pháp / seed đã chọn.');
      return keys.reduce((sum, k) => sum + cell.values[k][i], 0) / keys.length;
    }
    for (const cell of dataset.cells) {
      if (config.band !== 'all' && cell.band !== Number(config.band)) continue;
      if (config.fold !== 'all' && Number(dataset.folds[String(cell.kb)]) !== Number(config.fold)) continue;
      if (config.excludeAdjacent && dataset.adjacent.includes(cell.kb)) continue;
      let g = groups.get(cell.kb);
      if (!g) { g = {kb: cell.kb, n: 0, sumA: 0, sumB: 0}; groups.set(cell.kb, g); }
      g.n += cell.n; g.sumA += value(cell, config.a); g.sumB += value(cell, config.b);
    }
    const rows = [...groups.values()].sort((a, b) => a.kb - b.kb);
    if (!rows.length) throw Error('Không còn dữ liệu sau bộ lọc. Hãy chọn phạm vi rộng hơn.');
    return rows.map(g => ({...g, meanA: g.sumA / g.n, meanB: g.sumB / g.n,
                           delta: (g.sumA - g.sumB) / g.n}));
  }
  function bootstrap(rows, repetitions, seed, progress = () => {}) {
    if (![1000, 10000, 50000].includes(repetitions)) throw Error('Số lần bootstrap không hợp lệ.');
    if (!Number.isInteger(seed) || seed < 0 || seed > 4294967295) throw Error('Seed phải là số nguyên từ 0 đến 4294967295.');
    const n = rows.reduce((s, x) => s + x.n, 0);
    const meanA = rows.reduce((s, x) => s + x.sumA, 0) / n;
    const meanB = rows.reduce((s, x) => s + x.sumB, 0) / n;
    const base = {n, clusters: rows.length, meanA, meanB, delta: meanA - meanB, ci: null, samples: [], rows};
    if (rows.length < 2) return base;
    const rand = rng32(seed), samples = [];
    for (let b = 0; b < repetitions; b++) {
      let sum = 0, count = 0;
      for (let j = 0; j < rows.length; j++) {
        const row = rows[Math.floor(rand() * rows.length)];
        sum += row.sumA - row.sumB; count += row.n;
      }
      samples.push(sum / count);
      if ((b + 1) % 500 === 0) progress((b + 1) / repetitions);
    }
    samples.sort((a, b) => a - b);
    return {...base, ci: [quantile(samples, .025), quantile(samples, .975)], samples};
  }
  function fromTN6(source, config) {
    if (config.a === config.b) throw Error('Chọn hai phương pháp khác nhau.');
    const bands = config.band === 'all' ? ['thap', 'giua', 'cao'] : [config.band];
    const rows = bands.filter(k => source[k]).map(k => {
      const row = source[k], a = row.phuong_phap[config.a], b = row.phuong_phap[config.b];
      if (!a || !b || !Number.isFinite(a[config.metric]) || !Number.isFinite(b[config.metric]))
        throw Error('Bảng TN6 gốc không có giá trị đã chọn.');
      return {band: k, n: row.so_tinh_huong, meanA: a[config.metric], meanB: b[config.metric],
              delta: a[config.metric] - b[config.metric]};
    });
    if (!rows.length) throw Error('Không có ô TN6 tương ứng.');
    const n = rows.reduce((s, x) => s + x.n, 0);
    const meanA = rows.reduce((s, x) => s + x.n * x.meanA, 0) / n;
    const meanB = rows.reduce((s, x) => s + x.n * x.meanB, 0) / n;
    const metricKey = {'mat_dv_%': 'C4_mat_dv', J: 'C2_J',
                       'khong_tuyen_song_%': 'C4a', 'du_phong_qua_tai_%': 'C4b'}[config.metric];
    const forward = config.a === 'lai6' && config.b === 'dphat';
    const reverse = config.b === 'lai6' && config.a === 'dphat';
    let ci = null;
    if (rows.length === 1 && metricKey && (forward || reverse)) {
      const saved = source[rows[0].band].tieu_chi[metricKey];
      if (saved) ci = forward ? saved[1].slice() : [-saved[1][1], -saved[1][0]];
    }
    return {n, clusters: null, meanA, meanB, delta: meanA - meanB, ci, rows, samples: [],
            ciSource: ci ? 'CI 95% của báo cáo TN6 gốc; không bootstrap lại' : 'Không đủ dữ liệu cửa sổ TN6 để tính CI cho lựa chọn này'};
  }
  function analyze(data, config, progress) {
    if (config.kind === 'tn6') return fromTN6(data.tn6[config.dataset], config);
    const dataset = data.datasets.find(d => d.id === config.dataset && d.kind === config.kind);
    if (!dataset) throw Error('Không tìm thấy bộ dữ liệu.');
    return {...bootstrap(select(dataset, config), config.repetitions, config.rngSeed, progress),
            ciSource: 'CI 95% percentile, bootstrap ghép cặp theo cửa sổ; chưa hiệu chỉnh đa kiểm định'};
  }
  const api = {rng32, quantile, select, bootstrap, fromTN6, analyze};
  root.CompareStats = api;
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
})(globalThis);
