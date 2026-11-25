"""
demo_tn15_3d.py — Ghép bản đồ 3D (ban_do_3d.html của nhánh demo) với demo TN15 → TN15b thành MỘT tệp HTML:
  • giữ nguyên bản đồ bão 3D gốc (TN6) với mọi lớp và điều khiển;
  • thêm chế độ "So sánh DM 7/3 ↔ DM x10" trên quả địa cầu:
      - thanh thời gian nhiều mốc (chạy / dừng / bước / tua / tốc độ; phím Space), vệ tinh di chuyển mượt giữa các mốc;
      - hai cách xem: CHIA ĐÔI màn hình (hai nửa xoay đồng bộ) hoặc CHỒNG hai tuyến lên MỘT bản đồ;
      - chọn điểm đầu / điểm cuối bằng ô chọn hoặc bấm lên quả địa cầu (điểm người dùng = điểm đầu, trạm mặt đất = điểm cuối);
      - biểu đồ nhỏ theo thời gian của cặp đang chọn (bấm để nhảy tới mốc);
  • thêm bảng demo 5 tab nổi trên nền 3D; từ tab chọn tuyến có nút đẩy cặp đang xem sang quả địa cầu.

    python -m scripts.demo_tn15_tuyen_nhieu                    # (một lần, ~10 phút) tuyến của hai mô hình ở nhiều mốc
    python -m scripts.demo_tn15_15b                            # demo 2D (được nhúng vào bản 3D)
    python -m scripts.demo_tn15_3d [--ban-do <ban_do_3d.html>]  # → results/bao_da_tuyen/demo_tn15_15b/ban_do_3d_tn15b.html

Chỉ vá ve_3d.js ở hai chỗ (lộ đối tượng three.js ra window.__g3d; cho phép vẽ thay khi bật chế độ so sánh).
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

GOC = Path(__file__).resolve().parents[1]
RA = GOC / "results" / "bao_da_tuyen" / "demo_tn15_15b"
BAN_DO_MAC_DINH = Path(r"D:\PBL4-demo\PBL4\ban_do_3d.html")

CSS = r"""
<style>
#bar15{position:fixed;top:14px;left:50%;transform:translateX(-50%);z-index:40;display:flex;gap:8px}
#bar15 button,.p15 button,.p15 select{font:inherit;font-size:12.5px;padding:6px 10px;border-radius:9px;border:1px solid rgba(255,255,255,.18);
  background:rgba(8,16,28,.82);color:#e8eef5;cursor:pointer}
#bar15 button:hover,.p15 button:hover{border-color:#4fc3f7}
#bar15 .chinh{border-color:#ff8a3d;color:#ffd2b3}
.p15 button.dang{background:#e8eef5;color:#0b1320;border-color:#e8eef5}
#ov15{position:fixed;inset:18px;z-index:60;display:none;border-radius:14px;overflow:hidden;border:1px solid rgba(255,255,255,.15);
  box-shadow:0 20px 60px rgba(0,0,0,.6);background:#14171c}
#ov15 iframe{width:100%;height:100%;border:0;display:block}
#ov15 .dong{position:absolute;top:10px;right:14px;z-index:2;font-size:13px;padding:7px 12px;border-radius:9px;border:1px solid rgba(255,255,255,.2);
  background:rgba(8,16,28,.9);color:#e8eef5;cursor:pointer}
.p15{position:fixed;z-index:45;background:rgba(8,16,28,.80);border:1px solid rgba(255,255,255,.10);border-radius:14px;color:#e8eef5;
  backdrop-filter:blur(10px);-webkit-backdrop-filter:blur(10px);font-size:12.5px;line-height:1.45;display:none}
#k15{top:10px;left:50%;transform:translateX(-50%);padding:9px 14px;display:none;flex-direction:column;gap:6px;align-items:center;max-width:97vw}
#k15 .hang{display:flex;gap:7px;align-items:center;flex-wrap:wrap;justify-content:center}
#k15 b{color:#4fc3f7;letter-spacing:.05em}
#k15 input[type=range]{width:min(360px,40vw);accent-color:#ff8a3d}
#k15 .moc{font-variant-numeric:tabular-nums;color:#cfd8dc;min-width:250px;text-align:left}
#k15 .ngan{width:1px;height:20px;background:rgba(255,255,255,.18)}
#bd15{cursor:pointer}
.m15 h4{margin:0 0 6px;font-size:14px} #m15l h4{color:#7fb6ff} #m15r h4{color:#ffb37a}
#m15l{border-color:rgba(79,157,255,.45)} #m15r{border-color:rgba(255,138,61,.45)}
.m15{bottom:16px;width:calc(50% - 32px);max-height:24vh;overflow:auto;padding:8px 12px}
#bd15p{left:16px;bottom:calc(24vh + 28px);width:min(480px,calc(50% - 32px));padding:8px 12px}
#bd15p .dau{display:flex;gap:6px;align-items:center;flex-wrap:wrap} #bd15p .gon{margin-left:auto}
#bd15p.thu .than{display:none} #bd15p .chong-only{display:none;margin-top:4px;color:#cfd8dc;font-size:11.5px}
body.chong15 #bd15p .chong-only{display:block}
#m15l{left:16px} #m15r{left:calc(50% + 16px)}
.m15 table{border-collapse:collapse;width:100%} .m15 td{padding:3px 6px;border-top:1px solid rgba(255,255,255,.07);vertical-align:top}
.m15 td:first-child{color:#8fa3b8;white-space:nowrap} .m15 .so{font-variant-numeric:tabular-nums;font-weight:600}
#vach15{position:fixed;top:0;bottom:0;left:50%;width:2px;margin-left:-1px;background:rgba(255,255,255,.25);z-index:41;display:none;pointer-events:none}
#tong15{color:#cfd8dc;font-size:12px;margin-top:4px}
#lb15{position:fixed;inset:0;pointer-events:none;z-index:42}
#lb15 div{position:absolute;transform:translate(8px,-50%);font-size:11px;font-weight:600;color:#e8eef5;text-shadow:0 0 4px #000,0 0 2px #000;white-space:nowrap}
body.cd15 .panel,body.cd15 #labels,body.cd15 #tip,body.cd15 #bar15{display:none!important}
body.cd15 .m15,body.cd15 #bd15p{display:block}
body.cd15.chia15 #vach15{display:block}
body.cd15 #k15{display:flex}
.hon15{color:#66e3a0} .kem15{color:#ff7a8a}
</style>
"""

HTML_PHAN = r"""
<div id="bar15">
  <button class="chinh" id="b15so">🌐 So sánh DM 7/3 ↔ DM x10 (TN15b)</button>
  <button id="b15demo">📊 Demo TN15 → TN15b</button>
</div>
<div id="k15" class="p15">
  <div class="hang"><b>TN15b · <span style="color:#7fb6ff">DM 7/3</span> ↔ <span style="color:#ffb37a">DM x10</span></b>
    <label>Kịch bản <select id="k15kb"></select></label>
    <label>Điểm đầu <select id="k15dau"></select></label>
    <label>Điểm cuối <select id="k15cuoi"></select></label>
    <span class="ngan"></span>
    <button id="k15chia" class="dang" title="Mỗi mô hình một nửa màn hình">◧ Chia đôi</button>
    <button id="k15chong" title="Hai tuyến trên cùng một bản đồ">⧉ Chồng</button>
    <span class="ngan"></span>
    <button id="k15demo">📊 Bảng demo</button>
    <button id="k15thoat">← Về bản đồ bão</button></div>
  <div class="hang">
    <button id="k15lui" title="Mốc trước">⏮</button>
    <button id="k15choi" title="Chạy / dừng (phím Space)">▶ Chạy</button>
    <button id="k15toi" title="Mốc sau">⏭</button>
    <input id="k15tua" type="range" min="0" max="1" step="0.001" value="0" title="Kéo để tua">
    <label>Tốc độ <select id="k15toc"><option value="0.5">0,5×</option><option value="1" selected>1×</option><option value="2">2×</option><option value="4">4×</option></select></label>
    <span class="moc" id="k15moc"></span></div>
</div>
<div id="bd15p" class="p15">
  <div class="dau"><b style="color:#4fc3f7">Cặp đang chọn theo thời gian</b>
    <button id="bd15tre" class="dang">trễ tuyến chính</button><button id="bd15mat">mất dịch vụ</button>
    <button class="gon" id="bd15gon" title="Thu gọn / mở">▾</button></div>
  <div class="than"><svg id="bd15" width="430" height="64" viewBox="0 0 430 64" style="width:100%;height:auto"></svg>
    <div id="tong15"></div>
    <div class="chong-only"><span style="color:#7fb6ff">━ DM 7/3</span> ống rộng, mờ · <span style="color:#ffb37a">━ DM x10</span> ống hẹp, đậm ·
      nét đứt = dự phòng · đi trùng thì thấy ống cam nằm trong ống xanh</div></div>
</div>
<div id="m15l" class="p15 m15"></div>
<div id="m15r" class="p15 m15"></div>
<div id="vach15"></div>
<div id="lb15"></div>
<div id="ov15"><button class="dong" id="ov15dong">✕ Đóng</button><iframe id="ov15f" title="Demo TN15 → TN15b"></iframe></div>
"""

JS = r"""
<script>
(function(){
"use strict";
const T = __T15__;
const DEMO = __DEMO__;
const $ = id => document.getElementById(id);
const G = () => window.__g3d;
const N = T.ten_nut.length, IDX = {}; T.ten_nut.forEach((n, i) => IDX[n] = i);
const MAU = {"7/3": 0x4f9dff, "x10": 0xff8a3d}, LOP = {"7/3": 1, "x10": 2}, MH = ["7/3", "x10"];
const KSTYLE = [{c:0x4fc3f7,s:5},{c:0xffb300,s:12},{c:0xab47bc,s:10},{c:0xef5350,s:9},{c:0x66bb6a,s:8}];
const f2 = (x,n=2) => (x==null||Number.isNaN(x)) ? "–" : Number(x).toLocaleString("vi-VN",{minimumFractionDigits:n,maximumFractionDigits:n});
const fs = (x,n=2) => (x>0?"+":"") + f2(x,n);
const vn = s => (G() && G().vn) ? G().vn(s) : s;
const tenNut = i => vn(T.ten_nut[i]);

let bat = false, kb = 0, tf = 0, choi = false, toc = 1, che = "chia", bdKieu = "tre";
let dau = null, cuoi = null, S15 = null, diemKind = [], ongs = [], khoLines = [], dsNhan = [[], []], dangJ = -1, canVe = true, last = 0;
const NP = new Float32Array(N*3);

// ---------------------------------------------------------------- bảng demo nổi (iframe)
function moDemo(){ const f = $("ov15f"); if (!f.srcdoc) f.srcdoc = DEMO; $("ov15").style.display = "block"; }
$("b15demo").onclick = moDemo; $("k15demo").onclick = moDemo;
$("ov15dong").onclick = () => { $("ov15").style.display = "none"; };
addEventListener("message", e => { const d = e.data; if (!d || d.loai !== "d15_3d") return;
  $("ov15").style.display = "none"; batSoSanh(d.kb, d.dau, d.cuoi, true); });

// ---------------------------------------------------------------- thời gian và vị trí nút
const KB = () => T.kich_ban[kb];
const soMoc = () => KB().thoi_diem.length;
function viTriTai(i, j, out){ const g = G(), p = KB().thoi_diem[j].nut[i], k = T.loai[i];
  return g.ll(p[0], p[1], 1 + p[2]/g.RE + (k >= 3 ? 0.0012 : 0), out); }
const _a = new THREE.Vector3(), _b = new THREE.Vector3();
function tinhViTri(){
  const n = soMoc(), j0 = Math.min(Math.floor(tf), n-1), j1 = Math.min(j0+1, n-1), tds = KB().thoi_diem;
  let a = tf - j0; if (tds[j1].t - tds[j0].t > 4) a = a < .5 ? 0 : 1;    // hai mốc cách xa (> 80 giây): không nội suy, chuyển thẳng
  for (let i = 0; i < N; i++){ viTriTai(i, j0, _a); viTriTai(i, j1, _b); _a.lerp(_b, a); NP[3*i] = _a.x; NP[3*i+1] = _a.y; NP[3*i+2] = _a.z; }
}
const P3 = (i, out) => out.set(NP[3*i], NP[3*i+1], NP[3*i+2]);

// ---------------------------------------------------------------- cảnh 3D
function dungCanh(){
  const g = G();
  S15 = new THREE.Scene(); S15.background = new THREE.Color(0x01030a);
  S15.add(new THREE.Mesh(g.earth.geometry, g.earth.material));
  S15.add(new THREE.Mesh(new THREE.SphereGeometry(1.14, 64, 48), new THREE.ShaderMaterial({
    vertexShader:"varying vec3 vN; void main(){ vN=normalize(normalMatrix*normal); gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.0); }",
    fragmentShader:"varying vec3 vN; void main(){ float i=pow(max(0.0,0.74-dot(vN,vec3(0.0,0.0,1.0))),3.0); gl_FragColor=vec4(0.25,0.6,1.0,1.0)*i*1.4; }",
    blending:THREE.AdditiveBlending, side:THREE.BackSide, transparent:true, depthWrite:false })));
  diemKind = [0,1,2,3,4].map(q => {
    const ds = []; T.loai.forEach((k, i) => { if (k === q) ds.push(i); });
    const geo = new THREE.BufferGeometry(); geo.setAttribute("position", new THREE.BufferAttribute(new Float32Array(ds.length*3), 3));
    const tex = q===1||q===2 ? g.TEX_DIAMOND : (q===3 ? g.TEX_SQUARE : g.TEX_SOLID);
    const p = new THREE.Points(geo, new THREE.PointsMaterial({size:KSTYLE[q].s*g.dpr, sizeAttenuation:false, map:tex, color:KSTYLE[q].c,
      transparent:true, alphaTest:.05, depthWrite:false, opacity: q===0 ? .55 : 1}));
    p.frustumCulled = false; S15.add(p); return {ds, geo};
  });
}
const GEO_ONG = new THREE.CylinderGeometry(1, 1, 1, 10, 1, true), _y = new THREE.Vector3(0,1,0), _d = new THREE.Vector3();
let nhomTuyen = null;
function themOng(a, b, f0, f1, mh, chinh){
  const m = new THREE.Mesh(GEO_ONG, new THREE.MeshBasicMaterial({color:MAU[mh], transparent:true, opacity:1, depthWrite:true}));
  m.layers.set(LOP[mh]); m.renderOrder = mh === "x10" ? 3 : 2; nhomTuyen.add(m);
  ongs.push({m, a, b, f0, f1, mh, chinh});
}
function dungTuyen(){
  if (nhomTuyen) S15.remove(nhomTuyen);
  nhomTuyen = new THREE.Group(); S15.add(nhomTuyen); ongs = []; khoLines = []; dsNhan = [[], []];
  const td = KB().thoi_diem[dangJ], cap = td.cap[dau + "|" + cuoi];
  capNhatBang(td, cap); veBieuDo();
  if (!cap) return;
  const vachKho = new THREE.LineBasicMaterial({color:0x90a4ae, transparent:true, opacity:.33});
  for (const tu of cap.kho){ const geo = new THREE.BufferGeometry(); geo.setAttribute("position", new THREE.BufferAttribute(new Float32Array(tu.length*3), 3));
    const l = new THREE.Line(geo, vachKho); l.frustumCulled = false; nhomTuyen.add(l); khoLines.push({l, tu}); }
  MH.forEach((mh, i) => {
    const chon = cap[mh].tuyen;
    chon.forEach((j, q) => { const tu = cap.kho[j];
      for (let s = 0; s < tu.length-1; s++){
        if (q === 0) themOng(tu[s], tu[s+1], 0, 1, mh, true);
        else { const n = 9; for (let k = 0; k < n; k += 2) themOng(tu[s], tu[s+1], k/n, Math.min(1, (k+1)/n), mh, false); } } });
    dsNhan[i] = [...new Set(chon.flatMap(j => cap.kho[j]))];
  });
  apMauChe();
}
function apMauChe(){
  for (const o of ongs){ const m = o.m.material;
    if (che === "chong" && o.mh === "7/3"){ m.opacity = o.chinh ? .5 : .45; m.depthWrite = false; }
    else { m.opacity = o.chinh ? 1 : .9; m.depthWrite = o.chinh; } m.needsUpdate = true; }
}
function capNhatHinh(){
  diemKind.forEach(k => { const a = k.geo.getAttribute("position").array; k.ds.forEach((i, q) => { a[3*q] = NP[3*i]; a[3*q+1] = NP[3*i+1]; a[3*q+2] = NP[3*i+2]; });
    k.geo.getAttribute("position").needsUpdate = true; });
  for (const {l, tu} of khoLines){ const a = l.geometry.getAttribute("position").array; tu.forEach((i, q) => { a[3*q] = NP[3*i]; a[3*q+1] = NP[3*i+1]; a[3*q+2] = NP[3*i+2]; });
    l.geometry.getAttribute("position").needsUpdate = true; }
  const cam = G().camera, r = .0032 * Math.max(cam.position.length() - 1, .12);
  for (const o of ongs){
    P3(o.a, _a); P3(o.b, _b); const A = _a.clone().lerp(_b, o.f0), B = _a.clone().lerp(_b, o.f1);
    _d.subVectors(B, A); const L = _d.length(); if (L < 1e-9){ o.m.visible = false; continue; } o.m.visible = true;
    let he = o.chinh ? 1.5 : .75; if (che === "chong") he *= (o.mh === "7/3" ? 1.9 : .85);
    o.m.position.copy(A).addScaledVector(_d, .5); o.m.quaternion.setFromUnitVectors(_y, _d.normalize()); o.m.scale.set(r*he, L, r*he);
  }
}

// ---------------------------------------------------------------- bảng số, biểu đồ, nhãn
function bangMh(cap, mh){
  const x = cap[mh], k = cap.kho, ten = j => k[j].map(tenNut).join(" → ");
  return `<table><tr><td>Tuyến chính</td><td>${ten(x.tuyen[0])} <span style="color:#8fa3b8">(trễ cơ sở ${f2(cap.tre_kho[x.tuyen[0]],1)} ms)</span></td></tr>
    <tr><td>Dự phòng (${x.tuyen.length-1})</td><td>${x.tuyen.slice(1).map(j=>ten(j)).join("<br>") || "–"}</td></tr>
    <tr><td>Luồng này</td><td>mất dịch vụ <span class="so">${f2(x.mat_luong)}%</span> · trễ tuyến chính <span class="so">${f2(x.tre_luong,1)} ms</span></td></tr>
    <tr><td>Cả kịch bản</td><td>mất <span class="so">${f2(x.mat_tong)}%</span> · trễ <span class="so">${f2(x.tre_tong,1)} ms</span></td></tr></table>`;
}
function capNhatBang(td, cap){
  const tieu = mh => mh === "7/3" ? (che === "chia" ? "◀ " : "") + "DM 7/3 (học lại cùng đường ống)" : "DM x10 (J vật lý)" + (che === "chia" ? " ▶" : "");
  if (!cap){ ["m15l","m15r"].forEach((id, i) => $(id).innerHTML = `<h4>${tieu(MH[i])}</h4><i>Cặp này không có đường nào sống ở mốc này.</i>`);
    $("tong15").innerHTML = `<span style="color:#ff7a8a">${vn(dau)} → ${vn(cuoi)}: không tới được ở mốc này</span>`; return; }
  $("m15l").innerHTML = `<h4>${tieu("7/3")}</h4>` + bangMh(cap, "7/3"); $("m15r").innerHTML = `<h4>${tieu("x10")}</h4>` + bangMh(cap, "x10");
  const a = cap["7/3"], b = cap["x10"], d1 = b.mat_luong - a.mat_luong, d2 = b.tre_luong - a.tre_luong;
  const cl = (d, n) => Math.abs(d) < .5*Math.pow(10,-n) ? "" : (d < 0 ? "hon15" : "kem15");
  const cungTap = [...a.tuyen].sort().join() === [...b.tuyen].sort().join(), cungChinh = a.tuyen[0] === b.tuyen[0];
  $("tong15").innerHTML = `${cungTap ? "Cùng một tập tuyến" : (cungChinh ? "Cùng tuyến chính, khác dự phòng" : "Khác tuyến chính")} · x10 − 7/3: mất luồng
    <b class="${cl(d1,2)}">${fs(d1)}%</b>, trễ <b class="${cl(d2,1)}">${fs(d2,1)} ms</b>
    <br><span style="color:#8fa3b8">luồng nền của kịch bản ở mốc này: ${td.luong_goc.map(x => vn(x[0]) + "→" + vn(x[1])).join(", ")} · mỗi mốc chỉ là một tình huống minh hoạ</span>`;
}
function veBieuDo(){
  const svg = $("bd15"), W = 430, H = 64, l = 30, r = 8, t = 6, b = 14; svg.innerHTML = "";
  const tds = KB().thoi_diem, n = tds.length, khoa = dau + "|" + cuoi, mk = bdKieu === "tre" ? "tre_luong" : "mat_luong";
  const v = MH.map(mh => tds.map(td => td.cap[khoa] ? td.cap[khoa][mh][mk] : null));
  const vs = v.flat().filter(x => x != null); if (!vs.length){ svg.innerHTML = `<text x="${W/2}" y="36" fill="#8fa3b8" font-size="11" text-anchor="middle">cặp này không tới được ở mốc nào</text>`; return; }
  let lo = Math.min(...vs), hi = Math.max(...vs); if (hi - lo < 1e-6){ lo -= 1; hi += 1; }
  const X = j => l + (W-l-r) * (n > 1 ? j/(n-1) : .5), Y = x => t + (H-t-b) * (1 - (x-lo)/(hi-lo));
  const NS = "http://www.w3.org/2000/svg", el = (tag, at) => { const e = document.createElementNS(NS, tag); for (const k in at) e.setAttribute(k, at[k]); svg.appendChild(e); return e; };
  el("text", {x: 2, y: t+8, fill:"#8fa3b8", "font-size":10}).textContent = f2(hi, 1);
  el("text", {x: 2, y: H-b, fill:"#8fa3b8", "font-size":10}).textContent = f2(lo, 1);
  el("text", {x: W-r, y: H-2, fill:"#8fa3b8", "font-size":10, "text-anchor":"end"}).textContent = bdKieu === "tre" ? "ms · từng mốc" : "% · từng mốc";
  tds.forEach((td, j) => { const c = td.cap[khoa]; if (c && c["7/3"].tuyen[0] !== c["x10"].tuyen[0])
    el("rect", {x: X(j)-3, y: t, width: 6, height: H-t-b, fill:"rgba(255,255,255,.08)"}); });
  [["7/3","#7fb6ff",2.6],["x10","#ffb37a",1.6]].forEach(([mh, c, w], i) => {
    let d = ""; v[i].forEach((x, j) => { if (x == null) return; d += (d ? "L" : "M") + X(j).toFixed(1) + " " + Y(x).toFixed(1); });
    el("path", {d, fill:"none", stroke:c, "stroke-width":w, "stroke-opacity": mh === "7/3" ? .75 : 1});
    v[i].forEach((x, j) => { if (x != null) el("circle", {cx: X(j), cy: Y(x), r: 2.4, fill: c}); }); });
  el("line", {id:"bd15vach", x1: X(tf), x2: X(tf), y1: t-2, y2: H-b+2, stroke:"#fff", "stroke-width":1.2});
  svg.dataset.n = n;
}
function capNhatVachBd(){ const v = $("bd15vach"); if (!v) return; const n = soMoc(), x = 30 + (430-30-8) * (n > 1 ? tf/(n-1) : .5); v.setAttribute("x1", x); v.setAttribute("x2", x); }
const _v = new THREE.Vector3();
function nhan(w0, ww, i, ds, cp, cam, h){
  let html = "";
  for (const ni of ds){ P3(ni, _a); if (_a.clone().normalize().dot(cp) < .12) continue;
    _v.copy(_a).project(cam); if (_v.z > 1) continue;
    const x = (_v.x + 1) / 2 * ww + w0, y = (1 - _v.y) / 2 * h; if (x < w0 || x > w0 + ww - 20) continue;
    const ten = T.ten_nut[ni], dam = (ten === dau || ten === cuoi);
    if (T.loai[ni] === 0 && !dam) continue;
    html += `<div style="left:${x}px;top:${y}px;${dam ? "font-size:12.5px;color:#fff" : ""}">${tenNut(ni).replace("HAPS ","H·").replace("UAV ","U·")}</div>`; }
  return html;
}

// ---------------------------------------------------------------- vòng vẽ (gọi từ vòng lặp của bản đồ gốc)
function ve(){
  const g = G(), R = g.renderer, cam = g.camera, w = innerWidth, h = innerHeight, now = performance.now();
  if (!S15) return;
  const dt = Math.min((now - last) / 1000, .1); last = now; const n = soMoc();
  if (choi && n > 1){ tf += dt * toc * .35; if (tf > n - 1) tf = 0; }
  const j = Math.min(Math.round(tf), n - 1);
  if (j !== dangJ || canVe){ dangJ = j; canVe = false; dungTuyen(); }
  $("k15tua").value = tf / Math.max(n - 1, 1); capNhatVachBd();
  const td = KB().thoi_diem[j], phut = Math.round((td.t - KB().thoi_diem[0].t) * 20 / 60);
  $("k15moc").textContent = `mốc ${j+1}/${n} · phút thứ ${phut} của cửa sổ · ${td.gio_utc.slice(0,13)}h UTC${td.t === KB().moc_goc ? " · mốc của bản 2D" : ""}`;
  tinhViTri(); capNhatHinh();
  cam.clearViewOffset(); const cp = cam.position.clone().normalize(); let html = "";
  if (che === "chia"){
    cam.aspect = (w/2) / h; cam.updateProjectionMatrix(); R.setScissorTest(true);
    [[0,1],[1,2]].forEach(([i, lop]) => { R.setViewport(i*w/2, 0, w/2, h); R.setScissor(i*w/2, 0, w/2, h);
      cam.layers.set(0); cam.layers.enable(lop); R.render(S15, cam); html += nhan(i*w/2, w/2, i, dsNhan[i], cp, cam, h); });
    R.setScissorTest(false);
  } else {
    cam.aspect = w / h; cam.updateProjectionMatrix(); R.setViewport(0, 0, w, h);
    cam.layers.set(0); cam.layers.enable(1); cam.layers.enable(2); R.render(S15, cam);
    html = nhan(0, w, 0, [...new Set([...dsNhan[0], ...dsNhan[1]])], cp, cam, h);
  }
  R.setViewport(0, 0, w, h); cam.layers.set(0); $("lb15").innerHTML = html;
}

// ---------------------------------------------------------------- điều khiển
function napChon(){
  const k = KB(), sd = $("k15dau"), sc = $("k15cuoi");
  if (!sd.options.length){
    T.ten_nut.map((n, i) => [n, T.loai[i]]).filter(x => x[1] === 4).map(x => x[0]).sort().forEach(n => sd.appendChild(new Option(vn(n), n)));
    T.ten_nut.map((n, i) => [n, T.loai[i]]).filter(x => x[1] === 3).map(x => x[0]).sort().forEach(n => sc.appendChild(new Option(vn(n), n))); }
  if (!dau){ const jg = Math.max(0, k.thoi_diem.findIndex(td => td.t === k.moc_goc)); const cap = k.thoi_diem[jg].cap;
    const ok = Object.entries(cap); const khac = ok.find(([key, v]) => v["7/3"].tuyen[0] !== v["x10"].tuyen[0]) || ok[0];
    if (khac) [dau, cuoi] = khac[0].split("|"); }
  sd.value = dau; sc.value = cuoi;
}
function denMocGoc(){ const k = KB(); tf = Math.max(0, k.thoi_diem.findIndex(td => td.t === k.moc_goc)); }
function datChoi(p){ choi = p; $("k15choi").textContent = p ? "⏸ Dừng" : "▶ Chạy"; $("k15choi").classList.toggle("dang", p); }
function datChe(c){ che = c; $("k15chia").classList.toggle("dang", c === "chia"); $("k15chong").classList.toggle("dang", c === "chong");
  document.body.classList.toggle("chia15", c === "chia"); document.body.classList.toggle("chong15", c === "chong"); apMauChe(); canVe = true; }
function batSoSanh(k_, d_, c_, veMocGoc){
  if (!G()) return;
  const doiKb = k_ !== undefined && k_ !== kb;
  if (k_ !== undefined){ kb = k_; dau = d_; cuoi = c_; }
  if (!S15) dungCanh();
  bat = true; document.body.classList.add("cd15"); $("k15kb").value = kb; datChe(che);
  napChon(); if (veMocGoc || doiKb || !last) denMocGoc(); canVe = true; last = performance.now(); G().goToVN();
  window.__g15 = {bat: true, ve};
}
function tatSoSanh(){ bat = false; datChoi(false); document.body.classList.remove("cd15", "chia15", "chong15"); $("lb15").innerHTML = "";
  window.__g15 = {bat: false, ve}; const g = G(); g.camera.layers.set(0); g.resize(); }
T.kich_ban.forEach((k, i) => $("k15kb").appendChild(new Option(`${k.nhan} — ${k.cua_so}`, i)));
$("k15kb").onchange = () => { kb = +$("k15kb").value; dau = null; napChon(); denMocGoc(); canVe = true; };
$("k15dau").onchange = () => { dau = $("k15dau").value; canVe = true; };
$("k15cuoi").onchange = () => { cuoi = $("k15cuoi").value; canVe = true; };
$("k15chia").onclick = () => datChe("chia"); $("k15chong").onclick = () => datChe("chong");
$("k15choi").onclick = () => datChoi(!choi);
$("k15lui").onclick = () => { datChoi(false); tf = Math.max(0, Math.ceil(tf - 1e-6) - 1); };
$("k15toi").onclick = () => { datChoi(false); tf = Math.min(soMoc() - 1, Math.floor(tf + 1e-6) + 1); };
$("k15tua").oninput = () => { datChoi(false); tf = +$("k15tua").value * (soMoc() - 1); };
$("k15toc").onchange = () => { toc = +$("k15toc").value; };
$("bd15tre").onclick = () => { bdKieu = "tre"; $("bd15tre").classList.add("dang"); $("bd15mat").classList.remove("dang"); veBieuDo(); };
$("bd15mat").onclick = () => { bdKieu = "mat"; $("bd15mat").classList.add("dang"); $("bd15tre").classList.remove("dang"); veBieuDo(); };
$("bd15").addEventListener("click", e => { const rc = $("bd15").getBoundingClientRect(), n = soMoc();
  const u = Math.min(Math.max((e.clientX - rc.left) / rc.width * 430 - 30, 0) / (430 - 38), 1); datChoi(false); tf = Math.round(u * (n - 1)); });
$("b15so").onclick = () => batSoSanh(); $("k15thoat").onclick = tatSoSanh;
$("bd15gon").onclick = () => { const p = $("bd15p"); p.classList.toggle("thu"); $("bd15gon").textContent = p.classList.contains("thu") ? "▸" : "▾"; };
addEventListener("keydown", e => { if (!bat || e.code !== "Space" || /SELECT|INPUT/.test((e.target || {}).tagName || "")) return;
  e.preventDefault(); e.stopImmediatePropagation(); datChoi(!choi); }, true);

// bấm lên quả địa cầu: điểm người dùng gần nhất = điểm đầu, trạm mặt đất gần nhất = điểm cuối
let _xuong = null;
$("gl").addEventListener("pointerdown", e => { _xuong = [e.clientX, e.clientY]; });
$("gl").addEventListener("click", e => {
  if (!bat || !_xuong || Math.hypot(e.clientX - _xuong[0], e.clientY - _xuong[1]) > 5) return;
  const cam = G().camera, w = innerWidth, h = innerHeight, cp = cam.position.clone().normalize();
  const w0 = che === "chia" ? (e.clientX < w/2 ? 0 : w/2) : 0, ww = che === "chia" ? w/2 : w;
  cam.aspect = ww / h; cam.updateProjectionMatrix();
  let tot = -1, dmin = 16;
  for (let i = 0; i < N; i++){ if (T.loai[i] < 3) continue; P3(i, _a); if (_a.clone().normalize().dot(cp) < .12) continue;
    _v.copy(_a).project(cam); const x = (_v.x + 1)/2*ww + w0, y = (1 - _v.y)/2*h, dd = Math.hypot(x - e.clientX, y - e.clientY);
    if (dd < dmin){ dmin = dd; tot = i; } }
  if (tot < 0) return;
  if (T.loai[tot] === 4) dau = T.ten_nut[tot]; else cuoi = T.ten_nut[tot];
  $("k15dau").value = dau; $("k15cuoi").value = cuoi; canVe = true;
});
window.__g15 = {bat: false, ve};
})();
</script>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ban-do", default=str(BAN_DO_MAC_DINH), help="ban_do_3d.html của nhánh demo")
    a = ap.parse_args()
    goc = Path(a.ban_do).read_text(encoding="utf-8")
    demo = (RA / "demo_tn15_15b.html").read_text(encoding="utf-8").replace('<html lang="vi">', '<html lang="vi" data-theme="dark">', 1)
    tep_nhieu = RA / "tuyen_nhieu.json"
    if not tep_nhieu.exists():
        raise SystemExit("thiếu tuyen_nhieu.json — chạy python -m scripts.demo_tn15_tuyen_nhieu")
    du_lieu = json.loads(tep_nhieu.read_text(encoding="utf-8"))
    # vá 1: cho phép vẽ thay bằng chế độ so sánh
    cu = "renderer.render(scene, camera); requestAnimationFrame(frame);"
    if goc.count(cu) != 1:
        raise SystemExit("không tìm thấy đúng một chỗ vẽ trong ve_3d.js — bản đồ đã đổi cấu trúc?")
    goc = goc.replace(cu, "if (window.__g15 && window.__g15.bat) { window.__g15.ve(); } else { renderer.render(scene, camera); } "
                          "requestAnimationFrame(frame);")
    # vá 2: lộ đối tượng three.js ra window.__g3d (ngay trước lệnh khởi động vòng lặp cuối cùng)
    m = list(re.finditer(r"requestAnimationFrame\(frame\);(\r?\n)\}\)\(\);", goc))
    if len(m) != 1:
        raise SystemExit("không tìm thấy đúng một chỗ khởi động vòng lặp")
    lo = ("window.__g3d = {scene, camera, renderer, controls, ll, RE, earth, TEX_SOLID, TEX_DIAMOND, TEX_SQUARE, TEX_DOT, dpr, vn, "
          "goToVN: () => goTo('vn'), resize};" + m[0].group(1))
    goc = goc[:m[0].start()] + lo + goc[m[0].start():]
    an_toan = lambda x: json.dumps(x, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")   # không đóng thẻ script sớm
    chen = CSS + HTML_PHAN + JS.replace("__T15__", an_toan(du_lieu)).replace("__DEMO__", an_toan(demo))
    i = goc.rfind("</body>")
    ra = goc[:i] + chen + goc[i:]
    RA.mkdir(parents=True, exist_ok=True)
    tep = RA / "ban_do_3d_tn15b.html"
    tep.write_text(ra, encoding="utf-8", newline="")
    print(f"đã ghi {tep} ({tep.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
