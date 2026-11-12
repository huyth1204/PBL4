/* Run: node --test test/test_demo_so_sanh.cjs. No browser or network required. */
const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const core = require('../assets_demo/so_sanh_core.js');
const root = path.resolve(__dirname, '..');
const html = fs.readFileSync(path.join(root, 'results/bao_da_tuyen/demo_thong_ke/chay_so_sanh.html'), 'utf8');
const data = JSON.parse(html.match(/<script id="comparison-data" type="application\/json">([\s\S]*?)<\/script>/)[1]);
const report = JSON.parse(fs.readFileSync(path.join(root, 'results/bao_da_tuyen/demo_thong_ke/nguon/ket_qua_11.json'), 'utf8'));
const defaults = {kind:'test', dataset:'c7wh_k4_5', metric:'mat_dv_%', a:'dm11_lai', b:'dphat',
  band:'all', seed:'all', fold:'all', excludeAdjacent:false, repetitions:10000, rngSeed:0};
const close = (a,b) => assert.ok(Math.abs(a-b)<1e-9, `${a} != ${b}`);

test('weighted cluster bootstrap preserves observation weights and pairing', () => {
  // Exactly four equally likely draws: (0,0)=1, (0,1)=(1,0)=2.5, (1,1)=3.
  const rows = [{kb:0,n:1,sumA:11,sumB:10}, {kb:1,n:3,sumA:39,sumB:30}];
  const r = core.bootstrap(rows,10000,12);
  close(r.meanA,12.5); close(r.meanB,10); close(r.delta,2.5);
  assert.deepEqual(r.ci,[1,3]); assert.deepEqual([...new Set(r.samples)],[1,2.5,3]);
  assert.deepEqual(r.samples,core.bootstrap(rows,10000,12).samples);
  const reverse = core.bootstrap(rows.map(r=>({...r,sumA:r.sumB,sumB:r.sumA})),10000,12);
  assert.deepEqual(reverse.ci,[-3,-1]); close(reverse.delta,-2.5);
});
test('zero difference is exact and one cluster does not create a fake CI', () => {
  const rows=[{kb:0,n:2,sumA:6,sumB:6},{kb:1,n:3,sumA:6,sumB:6}];
  assert.deepEqual(core.bootstrap(rows,1000,0).ci,[0,0]);
  assert.equal(core.bootstrap(rows.slice(0,1),1000,0).ci,null);
  assert.throws(()=>core.bootstrap(rows,1000,-1),/Seed/);
});
test('all 11 TN11 test means and three principal contrasts match saved report', () => {
  for (const d of data.datasets.filter(d=>d.kind==='test'&&d.scale==='4–5')) {
    const old=report.kiem_tra[d.bo];
    for (const [a,b,key] of [['dm11_lai','dphat','hieu_vs_dphat'],['dm11_lai','dm10c_lai','dm11_tru_dm10c'],['dm11','sm11','dm11_tru_sm11']]) {
      const r=core.analyze(data,{...defaults,dataset:d.id,a,b,repetitions:1000});
      const expected=key==='hieu_vs_dphat'?old[key]['Lai(DM11)'][0]:old[key][0];
      close(r.delta,expected); assert.equal(r.n,old.n);
    }
  }
});
test('all datasets and metrics are finite; seed averaging is before bootstrap', () => {
  for (const d of data.datasets) for (const metric of data.metrics) {
    const r=core.analyze(data,{...defaults,dataset:d.id,kind:d.kind,metric,repetitions:1000});
    assert.ok([r.meanA,r.meanB,r.delta,...(r.ci||[])].every(Number.isFinite));
  }
  const d=data.datasets.find(d=>d.id==='c7wh_k4_5');
  const values=['0','1','2','all'].map(seed=>core.analyze(data,{...defaults,seed,repetitions:1000}));
  close(values[3].meanA,values.slice(0,3).reduce((s,r)=>s+r.meanA,0)/3);
  assert.equal(values[3].n,values[0].n);
  assert.equal(values[3].clusters,values[0].clusters);
  assert.throws(()=>core.select(d,{...defaults,seed:'9'}),/seed/);
  assert.throws(()=>core.select(d,{...defaults,a:'dphat'}),/khác nhau/);
});
test('load, held-out fold and adjacent-day filters retain the intended windows', () => {
  const d=data.datasets.find(d=>d.id==='x11_k4_5');
  for (let fold=0;fold<4;fold++) {
    const rows=core.select(d,{...defaults,kind:'cv',a:'sm11',b:'sm11_da_hoc',fold:String(fold)});
    assert.ok(rows.every(r=>d.folds[String(r.kb)]===fold));
  }
  const c=data.datasets.find(d=>d.id==='c6_k4_5');
  const rows=core.select(c,{...defaults,band:'2',excludeAdjacent:true});
  assert.ok(rows.every(r=>!c.adjacent.includes(r.kb)));
  assert.equal(rows.reduce((s,r)=>s+r.n,0),c.cells.filter(r=>r.band===2&&!c.adjacent.includes(r.kb)).reduce((s,r)=>s+r.n,0));
  assert.throws(()=>core.select(c,{...defaults,band:'9'}),/Không còn/);
});
test('TN6 preserves exact saved CI, reverses signs, never averages CIs', () => {
  const c={...defaults,kind:'tn6',dataset:'c6_k4_5',a:'lai6',b:'dphat',band:'thap'};
  const r=core.analyze(data,c);
  assert.deepEqual(r.ci,data.tn6[c.dataset].thap.tieu_chi.C4_mat_dv[1]);
  const reverse=core.analyze(data,{...c,a:c.b,b:c.a});
  assert.deepEqual(reverse.ci,[-r.ci[1],-r.ci[0]]);
  assert.equal(core.analyze(data,{...c,band:'all'}).ci,null);
  assert.equal(core.analyze(data,{...c,a:'dm6'}).ci,null);
  for(const dataset of Object.keys(data.tn6).filter(x=>x.includes('_k'))) {
    const all=core.analyze(data,{...c,dataset,band:'all'});
    close(all.meanA,all.rows.reduce((s,r)=>s+r.n*r.meanA,0)/all.n);
  }
});
test('offline page has embedded data/code and no runtime remote dependencies', () => {
  assert.equal(/<script[^>]+src=/i.test(html),false);
  assert.equal(/<link[^>]+href=/i.test(html),false);
  assert.equal(/\bfetch\s*\(/.test(html),false);
  for (const id of ['run','csv','windowsCsv','json','svg']) assert.ok(html.includes(`id="${id}"`));
});
