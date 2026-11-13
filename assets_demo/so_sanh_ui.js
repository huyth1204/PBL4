(function () {
  'use strict';
  const data = JSON.parse(document.getElementById('comparison-data').textContent);
  const $ = id => document.getElementById(id);
  const names = {dphat: 'Dijkstra phạt', yen: 'Dijkstra (Yen)', dm11: 'DM11 thuần', dm11_lai: 'Lai(DM11)',
    dm10c: 'DM10c thuần', dm10c_lai: 'Lai(DM10c)', sm11: 'SM11', dm11_da_hoc: 'DM11 · đã học',
    sm11_da_hoc: 'SM11 · đã học', dm6: 'DM6', lai6: 'Lai6', dm6_tkcb: 'DM6 + TKCB',
    dm: 'DM (TN4)', dm_tkcb: 'DM + TKCB (TN5)', dm6a: 'DM6a', online_tkcb: 'TKCB trên J bảng'};
  const fmt = (x, digits = 3) => x === null ? '—' : Number(x).toLocaleString('vi-VN', {maximumFractionDigits: digits});
  const esc = text => String(text).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const unit = metric => metric.endsWith('_%') ? '%' : metric === 'J' ? 'J' : 'ms';
  const deltaUnit = metric => metric.endsWith('_%') ? 'điểm %' : unit(metric);
  let worker = null, workerURL = null, latest = null, chartSVG = '', runID = 0;
  function options(id, values, preferred) {
    const select = $(id); select.replaceChildren();
    for (const [value, text] of values) select.add(new Option(text, value));
    if (values.some(x => x[0] === preferred)) select.value = preferred;
  }
  function stop() {
    if (worker) worker.terminate(); worker = null;
    if (workerURL) URL.revokeObjectURL(workerURL); workerURL = null;
    $('run').disabled = false; $('cancel').hidden = true; $('progress').hidden = true;
  }
  function invalidate() {
    runID++; stop(); latest = null; chartSVG = '';
    $('result').hidden = true; $('empty').hidden = false;
    $('status').textContent = 'Đã đổi lựa chọn. Bấm Chạy so sánh để tính lại.'; $('status').className = '';
  }
  function refreshKind() {
    const kind = $('kind').value;
    const sets = kind === 'tn6' ? Object.keys(data.tn6).filter(k => /_k/.test(k)).map(id => {
      const [bo, scale] = id.split('_k'); return [id, `${data.labels[bo] || bo} · K=${scale.replace('_', '–')}`];
    }) : data.datasets.filter(d => d.kind === kind).map(d => [d.id, `${d.label} · K=${d.scale}`]);
    options('dataset', sets, kind === 'test' ? 'c7wh_k4_5' : kind === 'cv' ? 'x11_k4_5' : 'c6_k4_5');
    const families = kind === 'tn6' ? ['lai6','dm6','dm6_tkcb','dphat','dm','dm_tkcb','dm6a','online_tkcb']
      : ['dm11_lai','dm11','dm10c_lai','dm10c','sm11','dphat','yen', ...(kind === 'cv' ? ['dm11_da_hoc','sm11_da_hoc'] : [])];
    const opts = families.map(x => [x, names[x] + (kind === 'cv' && !x.endsWith('_da_hoc') && !['dphat','yen'].includes(x) ? ' · ngoài fold' : '')]);
    options('methodA', opts, kind === 'tn6' ? 'lai6' : kind === 'cv' ? 'sm11' : 'dm11_lai');
    options('methodB', opts, kind === 'cv' ? 'sm11_da_hoc' : 'dphat');
    options('band', [['all','Tất cả'], ...(kind === 'tn6' ? [['thap','Thấp'],['giua','Giữa'],['cao','Cao']] : [['0','Thấp'],['1','Giữa'],['2','Cao']])], kind === 'tn6' ? 'thap' : 'all');
    $('fold').value = 'all'; $('excludeAdjacent').checked = false;
    $('seedWrap').hidden = kind !== 'test'; $('foldWrap').hidden = kind !== 'cv';
    $('adjWrap').hidden = kind !== 'test'; $('bootWrap').hidden = kind === 'tn6';
    $('rngSeed').disabled = kind === 'tn6';
    refreshDataset();
    $('modeNote').textContent = kind === 'tn6'
      ? 'TN6 có bảng tổng hợp gốc. Chỉ giữ CI gốc cho Lai6 ↔ Dijkstra phạt trong một dải tải; không thể bootstrap mới khi thiếu dữ liệu từng cửa sổ.'
      : kind === 'cv' ? 'Điểm ngoài fold từ 4 mô hình giữ lại nhóm bão/ngày. Bootstrap theo cửa sổ, không dùng 4 fold như 4 mẫu độc lập.'
      : 'Trung bình các seed mô hình trước, rồi bootstrap ghép cặp theo cửa sổ. Seed mô hình không được tính như mẫu thời tiết độc lập.';
  }
  function refreshDataset() {
    const d = data.datasets.find(d => d.id === $('dataset').value && d.kind === $('kind').value);
    if ($('kind').value === 'tn6') {
      const source = data.tn6[$('dataset').value];
      const bands = ['thap','giua','cao'].filter(b => source[b]);
      const available = ['lai6','dm6','dm6_tkcb','dphat','dm','dm_tkcb','dm6a','online_tkcb']
        .filter(m => bands.every(b => source[b].phuong_phap[m])).map(m => [m,names[m]]);
      options('methodA', available, $('methodA').value);
      options('methodB', available, $('methodB').value);
    }
    if (d) options('seed', [['all', d.seeds.length > 1 ? 'Trung bình 3 seed' : 'Seed 0'], ...(d.seeds.length > 1 ? d.seeds.map(s => [String(s), `Seed ${s}`]) : [])], 'all');
    $('excludeAdjacent').disabled = !d || !d.adjacent.length;
    if ($('excludeAdjacent').disabled) $('excludeAdjacent').checked = false;
  }
  function config() {
    return {kind: $('kind').value, dataset: $('dataset').value, metric: $('metric').value,
      a: $('methodA').value, b: $('methodB').value, band: $('band').value,
      seed: $('seed').value || 'all', fold: $('kind').value === 'cv' ? $('fold').value : 'all',
      excludeAdjacent: $('kind').value === 'test' && $('excludeAdjacent').checked,
      repetitions: Number($('repetitions').value), rngSeed: Number($('rngSeed').value)};
  }
  function plot(result, c) {
    const A = names[c.a], B = names[c.b], u = unit(c.metric), du = deltaUnit(c.metric);
    const maximum = Math.max(result.meanA, result.meanB, 1e-9) * 1.15;
    const bar = v => Math.max(0, v / maximum * 650);
    const deltaMax = Math.max(Math.abs(result.delta), ...(result.ci || []).map(Math.abs), .001) * 1.25;
    const x = v => 510 + v / deltaMax * 320;
    const text = (xx, yy, s, extra = '') => `<text x="${xx}" y="${yy}" ${extra}>${esc(s)}</text>`;
    let svg = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 940 480" role="img" aria-label="So sánh trung bình và khoảng tin cậy" style="font:14px system-ui,sans-serif;fill:#203047"><rect width="940" height="480" fill="white"/>`;
    svg += text(26, 32, `${$('dataset').selectedOptions[0].textContent} · ${$('metric').selectedOptions[0].textContent}`, 'font-size="16" font-weight="600"');
    const filters = `Tải: ${$('band').selectedOptions[0].textContent}`
      + (c.kind === 'test' ? ` · ${$('seed').selectedOptions[0].textContent}` : c.kind === 'cv' ? ` · ${$('fold').selectedOptions[0].textContent}` : ' · TN6 gốc')
      + (c.excludeAdjacent ? ' · bỏ cửa sổ kề ngày học' : '');
    svg += text(26, 58, filters + ' · thấp hơn là tốt hơn', 'fill="#607087"');
    [result.meanA, result.meanB].forEach((v, i) => {
      const yy = 88 + i * 66;
      svg += text(26, yy + 23, i === 0 ? A : B, 'font-size="13"');
      svg += `<rect x="190" y="${yy}" width="${bar(v)}" height="34" rx="4" fill="${i === 0 ? '#365977' : '#8c959d'}"/>`;
      svg += text(198 + bar(v), yy + 23, `${fmt(v)} ${u}`, 'font-size="13"');
    });
    svg += text(26, 250, `Hiệu A − B (${du})`, 'font-size="16" font-weight="650"');
    svg += `<line x1="190" y1="300" x2="830" y2="300" stroke="#b7c4d2"/><line x1="510" y1="270" x2="510" y2="330" stroke="#98a7b8" stroke-dasharray="5 4"/>`;
    svg += text(190, 347, fmt(-deltaMax), 'text-anchor="middle"');
    svg += text(510, 347, '0', 'text-anchor="middle"');
    svg += text(830, 347, fmt(deltaMax), 'text-anchor="middle"');
    if (result.ci) {
      const color = result.ci[1] < 0 ? '#466b58' : result.ci[0] > 0 ? '#9d433a' : '#637087';
      svg += `<line x1="${x(result.ci[0])}" y1="300" x2="${x(result.ci[1])}" y2="300" stroke="${color}" stroke-width="5"/>`;
      for (const v of result.ci) svg += `<line x1="${x(v)}" y1="289" x2="${x(v)}" y2="311" stroke="${color}" stroke-width="2"/>`;
    }
    svg += `<circle cx="${x(result.delta)}" cy="300" r="7" fill="#365977"/>`;
    svg += text(26, 389, `Hiệu: ${fmt(result.delta)} ${du} · ${result.ci ? 'CI 95% [' + result.ci.map(v => fmt(v)).join('; ') + ']' : 'Chưa có CI phù hợp'}`);
    svg += text(26, 420, `n=${result.n} · ${result.clusters === null ? 'TN6: tổng hợp từ báo cáo gốc' : result.clusters + ' cửa sổ · bootstrap ' + c.repetitions + ' lần · seed ' + c.rngSeed}`, 'font-size="12" fill="#607087"');
    svg += text(26, 446, 'Phân tích lại kết quả đã lưu · CI chưa hiệu chỉnh đa kiểm định · không phải huấn luyện lại', 'font-size="12" fill="#607087"');
    return svg + '</svg>';
  }
  function render(result, c, seconds) {
    const title = `${names[c.a]} − ${names[c.b]}`;
    const ds = $('dataset').selectedOptions[0].textContent;
    const scope = `${$('kind').selectedOptions[0].textContent} · ${ds} · tải ${$('band').selectedOptions[0].textContent.toLowerCase()}`
      + (c.kind === 'test' ? ` · ${$('seed').selectedOptions[0].textContent}` : c.kind === 'cv' ? ` · ${$('fold').selectedOptions[0].textContent}` : '')
      + (c.excludeAdjacent ? ' · đã bỏ cửa sổ kề ngày học' : '');
    const warnings = c.kind === 'tn6'
      ? 'TN6 chỉ có số liệu tổng hợp. Gộp các dải tải bằng số tình huống; không gộp CI bằng trung bình. Không dùng số TN6 và TN11 khác lần chạy làm so sánh trực tiếp.'
      : `CI theo cửa sổ, chưa loại hết phụ thuộc giữa các cửa sổ cùng bão; chưa hiệu chỉnh đa kiểm định. ${result.clusters < 5 ? 'Có ít hơn 5 cửa sổ: CI rất thiếu ổn định. ' : ''}`
        + (c.kind === 'cv' ? 'Chia fold theo bão/ngày; “đã học” dùng mô hình fold kế tiếp. DM10c không học x11. ' : '')
        + (c.excludeAdjacent ? 'Bỏ cửa sổ test kề ngày không sửa quá trình chọn checkpoint bằng validation. ' : '')
        + 'Bootstrap dùng bộ sinh số ngẫu nhiên của trang này, nên cận CI có thể lệch nhẹ so với báo cáo Python.';
    latest = {version: 1, createdAt: new Date().toISOString(), sourceDigest: data.digest, config: c,
      scope, methodA: names[c.a], methodB: names[c.b], metricUnit: unit(c.metric), deltaUnit: deltaUnit(c.metric),
      seconds, warnings, result: {...result, samples: undefined}};
    $('empty').hidden = true; $('result').hidden = false;
    $('resultTitle').textContent = title; $('scope').textContent = scope;
    $('count').textContent = fmt(result.n, 0); $('clusterCount').textContent = result.clusters === null ? 'Chưa có dữ liệu cụm' : fmt(result.clusters, 0);
    $('clusterCount').classList.toggle('small', result.clusters === null);
    $('delta').textContent = `${fmt(result.delta)} ${deltaUnit(c.metric)}`; $('delta').className = 'small';
    const verdict = !result.ci ? 'Chỉ mô tả chênh lệch; chưa có CI để kết luận độ chắc chắn.'
      : result.ci[1] < 0 ? 'CI nằm dưới 0: A thấp hơn B trong phạm vi lựa chọn.'
      : result.ci[0] > 0 ? 'CI nằm trên 0: A cao hơn B trong phạm vi lựa chọn.'
      : 'CI chứa 0: chưa đủ bằng chứng về chênh lệch.';
    $('verdict').textContent = verdict;
    $('ciNote').textContent = (result.ci ? `CI 95%: [${result.ci.map(x => fmt(x)).join('; ')}] ${deltaUnit(c.metric)}. ` : '') + result.ciSource;
    $('means').innerHTML = [c.a,c.b].map((m, i) => `<tr><td>${esc(names[m])}</td><td>${fmt(i ? result.meanB : result.meanA, 6)}</td><td>${unit(c.metric)}</td></tr>`).join('');
    chartSVG = plot(result, c); $('chart').innerHTML = chartSVG; $('chart').firstElementChild.classList.add('chart');
    $('limitations').textContent = warnings;
    $('detailTitle').textContent = c.kind === 'tn6' ? 'Xem kết quả từng dải tải' : 'Xem kết quả từng cửa sổ';
    $('windowRows').innerHTML = result.rows.map(r => `<tr><td>${esc(r.kb === undefined ? r.band : 'kb_' + String(r.kb).padStart(4,'0'))}</td><td>${r.n}</td><td>${fmt(r.meanA,6)}</td><td>${fmt(r.meanB,6)}</td><td>${fmt(r.delta,6)}</td></tr>`).join('');
    $('provenance').textContent = `Nguồn SHA-256: ${data.digest} · Tính xong ${latest.createdAt} · Không gửi dữ liệu ra mạng.`;
    $('status').textContent = `Hoàn tất trong ${fmt(seconds, 2)} giây. Có thể tải kết quả.`;
  }
  $('form').addEventListener('submit', e => {
    e.preventDefault(); invalidate(); const id = runID, c = config(), start = performance.now();
    $('run').disabled = true; $('cancel').hidden = false; $('progress').hidden = false;
    $('progress').value = 0; $('status').textContent = 'Đang tính lại so sánh…';
    const fail = error => { if (id !== runID) return; stop(); $('status').textContent = error; $('status').className = 'error'; };
    try {
      const source = $('comparison-core').textContent + '\nself.onmessage = function(e) { try { const r=CompareStats.analyze(e.data.data,e.data.config,p=>postMessage({progress:p})); postMessage({result:r}); } catch(e) { postMessage({error:e.message}); } };';
      workerURL = URL.createObjectURL(new Blob([source], {type:'text/javascript'})); worker = new Worker(workerURL);
      worker.onmessage = event => {
        if (id !== runID) return;
        if (event.data.error) return fail(event.data.error);
        if (event.data.progress !== undefined) $('progress').value = event.data.progress;
        if (event.data.result) { stop(); render(event.data.result, c, (performance.now() - start)/1000); }
      };
      worker.onerror = () => fail('Không chạy được worker offline. Hãy mở file bằng Chrome hoặc Edge.');
      const compact = c.kind === 'tn6' ? {tn6: {[c.dataset]: data.tn6[c.dataset]}}
        : {datasets: data.datasets.filter(d => d.id === c.dataset && d.kind === c.kind)};
      worker.postMessage({data: compact, config:c});
    } catch (error) { fail(error.message); }
  });
  $('cancel').onclick = () => { invalidate(); $('status').textContent = 'Đã hủy. Chưa có kết quả cho lựa chọn này.'; };
  $('kind').onchange = () => { refreshKind(); invalidate(); };
  $('dataset').onchange = () => { refreshDataset(); invalidate(); };
  for (const id of ['metric','methodA','methodB','band','seed','fold','excludeAdjacent','repetitions','rngSeed']) $(id).addEventListener('input', invalidate);
  function download(text, type, ext, suffix) {
    if (!latest) return;
    const url = URL.createObjectURL(new Blob([text],{type})), a = document.createElement('a');
    a.href = url; a.download = `${latest.config.kind}_${latest.config.dataset}_${suffix}.${ext}`;
    document.body.append(a); a.click(); a.remove(); setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
  function csv(rows) {
    return '\ufeff' + rows.map(row => row.map(x => '"' + String(x ?? '').replace(/"/g,'""') + '"').join(',')).join('\r\n');
  }
  $('csv').onclick = () => {
    if (!latest) return;
    const c=latest.config,r=latest.result;
    download(csv([['experiment','dataset','metric','method_A','method_B','band','model_seed','fold','exclude_adjacent','n','windows','mean_A','mean_B','delta_A_minus_B','delta_unit','ci95_low','ci95_high','ci_source','bootstrap_repetitions','bootstrap_seed','source_digest'],
      [c.kind,c.dataset,c.metric,c.a,c.b,c.band,c.kind==='test'?c.seed:'',c.kind==='cv'?c.fold:'',c.excludeAdjacent,r.n,r.clusters,r.meanA,r.meanB,r.delta,latest.deltaUnit,...(r.ci||['','']),r.ciSource,c.kind==='tn6'?'':c.repetitions,c.kind==='tn6'?'':c.rngSeed,data.digest]]), 'text/csv;charset=utf-8','csv','ket_qua');
  };
  $('windowsCsv').onclick = () => {
    if (!latest) return; const c=latest.config;
    download(csv([['experiment','dataset','metric','method_A','method_B','band_filter','model_seed','fold','exclude_adjacent','window_or_band','n','mean_A','mean_B','delta_A_minus_B','source_digest'],
      ...latest.result.rows.map(r=>[c.kind,c.dataset,c.metric,c.a,c.b,c.band,c.kind==='test'?c.seed:'',c.kind==='cv'?c.fold:'',c.excludeAdjacent,r.kb??r.band,r.n,r.meanA,r.meanB,r.delta,data.digest])]),'text/csv;charset=utf-8','csv','chi_tiet');
  };
  $('json').onclick = () => { if (latest) download(JSON.stringify(latest,null,2),'application/json','json','lan_chay'); };
  $('svg').onclick = () => { if (latest) download(chartSVG,'image/svg+xml','svg','bieu_do'); };
  refreshKind();
})();
