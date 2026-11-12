/* Exercise page events, worker messages, downloads and stale results without browser automation. */
const {test}=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const path=require('node:path');
const vm=require('node:vm');
const root=path.resolve(__dirname,'..');
const html=fs.readFileSync(path.join(root,'results/bao_da_tuyen/demo_thong_ke/chay_so_sanh.html'),'utf8');

function page() {
  const elements=new Map(), blobs=new Map(), downloads=[], jobs=[];
  class Element {
    constructor(id) {this.id=id;this.options=[];this.events={};this.hidden=false;this.checked=false;this.disabled=false;this.textContent='';this.innerHTML='';this.classList={toggle(){},add(){}};this.firstElementChild={classList:{add(){}}};this._value='';}
    set value(v){this._value=String(v);} get value(){return this._value;}
    get selectedOptions(){return this.options.filter(o=>o.value===this.value);}
    replaceChildren(){this.options=[];this.value='';}
    add(o){this.options.push(o);if(this.options.length===1)this.value=o.value;}
    addEventListener(name,fn){this.events[name]=fn;}
    append(){} remove(){}
    click(){if(this.href)downloads.push({filename:this.download,blob:blobs.get(this.href)});else if(this.onclick)this.onclick();}
  }
  for(const match of html.matchAll(/\bid="([^"]+)"/g)) elements.set(match[1],new Element(match[1]));
  for(const [id,values,value] of [
    ['kind',[['test','TN11 · tập kiểm tra độc lập'],['cv','TN11 · cross-validation'],['tn6','TN6 · báo cáo gốc']],'test'],
    ['metric',[['mat_dv_%','Mất dịch vụ (%)'],['J','Hàm mục tiêu J']],'mat_dv_%'],
    ['fold',[['all','Cả 4 fold'],['0','Fold 0'],['1','Fold 1']],'all'],
    ['repetitions',[['1000','1.000'],['10000','10.000'],['50000','50.000']],'10000']]) {
    const el=elements.get(id); for(const [v,t]of values)el.add({value:v,textContent:t});el.value=value;
  }
  elements.get('rngSeed').value='0';
  elements.get('comparison-data').textContent=html.match(/<script id="comparison-data" type="application\/json">([\s\S]*?)<\/script>/)[1];
  elements.get('comparison-core').textContent=fs.readFileSync(path.join(root,'assets_demo/so_sanh_core.js'),'utf8');
  const context={console,performance,Blob,setTimeout:fn=>{jobs.push(fn);},Option:class{constructor(t,v){this.textContent=t;this.value=v;}},
    document:{getElementById:id=>elements.get(id),body:new Element('body'),createElement:tag=>new Element(tag)},
    URL:{createObjectURL:b=>{const id='blob:'+blobs.size;blobs.set(id,b);return id;},revokeObjectURL(){}},
    Worker:class {
      constructor(url){this.blob=blobs.get(url);this.stopped=false;}
      terminate(){this.stopped=true;}
      postMessage(input){jobs.push(async()=>{if(this.stopped)return;const workerContext={postMessage:message=>{if(!this.stopped)this.onmessage({data:message});}};workerContext.self=workerContext;vm.runInNewContext(await this.blob.text(),workerContext);workerContext.onmessage({data:input});});}
    }};
  vm.runInNewContext(fs.readFileSync(path.join(root,'assets_demo/so_sanh_ui.js'),'utf8'),context);
  return {elements,downloads,jobs,
    async run(){elements.get('form').events.submit({preventDefault(){}});while(jobs.length)await jobs.shift()();},
    change(id,value){const e=elements.get(id);e.value=value;(e.onchange||e.events.input)();}};
}

test('Run creates fresh TN11 outputs; CSV/JSON/SVG agree; changing inputs invalidates exports',async()=>{
  const p=page();await p.run();
  assert.equal(p.elements.get('result').hidden,false);
  assert.match(p.elements.get('status').textContent,/Hoàn tất/);
  for(const id of ['csv','windowsCsv','json','svg'])p.elements.get(id).click();
  assert.equal(p.downloads.length,4);
  const csv=await p.downloads[0].blob.text(),json=JSON.parse(await p.downloads[2].blob.text()),svg=await p.downloads[3].blob.text();
  assert.equal(json.result.n,1872);assert.equal(json.result.clusters,13);
  assert.ok(Math.abs(json.result.delta-(-1.5164252992928566))<1e-9);
  assert.ok(csv.includes(String(json.result.delta)));
  assert.ok(svg.includes('n=1872'));assert.ok(svg.includes('Trung bình 3 seed'));
  assert.ok(p.downloads[0].filename.includes('c7wh_k4_5'));
  p.change('metric','J');assert.equal(p.elements.get('result').hidden,true);
  p.elements.get('csv').click();assert.equal(p.downloads.length,4);
  await p.run();p.elements.get('json').click();
  assert.equal(JSON.parse(await p.downloads.at(-1).blob.text()).config.metric,'J');
});

test('TN6 reports saved CI and removes missing models when dataset changes',async()=>{
  const p=page();p.change('kind','tn6');await p.run();
  p.elements.get('json').click();const first=JSON.parse(await p.downloads.at(-1).blob.text());
  assert.equal(first.config.a,'lai6');assert.equal(first.config.b,'dphat');
  assert.match(first.result.ciSource,/gốc/);assert.ok(first.result.ci);
  p.change('dataset','c6_k10');
  assert.equal(p.elements.get('methodA').options.some(o=>o.value==='dm6a'),false);
  p.change('band','all');await p.run();p.elements.get('json').click();
  assert.equal(JSON.parse(await p.downloads.at(-1).blob.text()).result.ci,null);
});

test('CV single-window selection withholds CI; identical methods show an actionable error',async()=>{
  const p=page();p.change('kind','cv');p.change('fold','0');await p.run();
  p.elements.get('json').click();const r=JSON.parse(await p.downloads.at(-1).blob.text());
  assert.equal(r.result.clusters,1);assert.equal(r.result.ci,null);
  p.change('methodB',p.elements.get('methodA').value);await p.run();
  assert.match(p.elements.get('status').textContent,/khác nhau/);
  assert.equal(p.elements.get('result').hidden,true);
});

test('cancel stops pending calculation without leaving downloadable stale output',async()=>{
  const p=page();p.elements.get('form').events.submit({preventDefault(){}});
  p.elements.get('cancel').click();while(p.jobs.length)await p.jobs.shift()();
  assert.match(p.elements.get('status').textContent,/Đã hủy/);
  assert.equal(p.elements.get('result').hidden,true);assert.equal(p.elements.get('run').disabled,false);
  p.elements.get('csv').click();assert.equal(p.downloads.length,0);
});
