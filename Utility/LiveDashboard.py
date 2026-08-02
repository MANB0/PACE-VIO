"""Non-blocking local dashboard for the real MACVO + VIO pipeline.

The dashboard is deliberately a read-only side channel. It never enters the
optimizer and it keeps only the newest JSON-safe snapshot plus a bounded
trajectory history. The displayed pose is composed at the IMU origin from
the runtime camera-to-IMU transform and is rebased to the first IMU sample.
"""

from __future__ import annotations

import copy
import base64
import csv
import io
import json
import math
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import numpy as np
import pypose as pp

try:
    from PIL import Image
except Exception:  # pragma: no cover - dashboard image support is optional
    Image = None

from Utility.PoseFrame import convert_pose_world_frame_only
from Utility.TrajectoryReference import compose_camera_to_imu_poses


_HTML = r'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>MACVO + VIO live dashboard</title>
<style>
:root { color-scheme: light; --ink:#172033; --muted:#64748b; --line:#d7dee8; --blue:#2563eb; --green:#059669; --orange:#ea580c; --black:#111827; --panel:#ffffff; --bg:#f4f7fb; }
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--ink); font-family:Inter,Segoe UI,Arial,sans-serif; }
header { padding:18px 24px 12px; background:#fff; border-bottom:1px solid var(--line); }
h1 { margin:0 0 5px; font-size:22px; }
.subtitle { color:var(--muted); font-size:13px; }
.layout { display:grid; grid-template-columns:minmax(0,1.6fr) minmax(320px,.8fr); gap:14px; padding:14px; }
.panel { background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:12px; box-shadow:0 2px 10px #1720330a; }
.plot-panel { min-height:560px; min-width:0; }
.toolbar { display:flex; align-items:center; gap:12px; margin-bottom:8px; flex-wrap:wrap; }
.playback-bar { display:grid; grid-template-columns:auto minmax(120px,1fr) minmax(120px,auto) auto; align-items:center; gap:10px; margin:8px 0 10px; padding:8px 10px; border:1px solid #e5eaf1; border-radius:6px; background:#f8fafc; }
.playback-bar input[type="range"] { width:100%; min-width:0; accent-color:var(--blue); }
.playback-time { min-width:0; overflow:hidden; text-overflow:ellipsis; color:var(--muted); font:12px ui-monospace,SFMono-Regular,Consolas,monospace; text-align:right; white-space:nowrap; }
.playback-time.replay { color:#b45309; font-weight:700; }
label { color:var(--muted); font-size:13px; }
select { padding:6px 9px; border:1px solid #b9c5d5; border-radius:6px; background:#fff; color:var(--ink); }
button { padding:6px 10px; border:1px solid #b9c5d5; border-radius:6px; background:#fff; color:var(--ink); cursor:pointer; }
button:hover { background:#f1f5f9; }
.live { margin-left:auto; font-size:12px; font-weight:700; color:#047857; }
.live.off { color:#b91c1c; }
canvas { display:block; width:100%; max-width:100%; height:auto; border:1px solid #e5eaf1; border-radius:6px; background:#fbfcfe; }
.cards { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:8px; }
.metric { border:1px solid #e5eaf1; border-radius:6px; padding:8px; min-height:58px; }
.metric .k { color:var(--muted); font-size:11px; }
.metric .v { font-size:14px; font-weight:650; margin-top:4px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
h2 { font-size:15px; margin:0 0 9px; }
.section { margin-top:12px; }
.section-head { display:flex; align-items:center; justify-content:space-between; gap:10px; margin-bottom:8px; }
.section-head h2 { margin:0; }
.pipeline-lights { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:8px; margin-bottom:9px; }
.signal { border:1px solid #e5eaf1; border-radius:6px; padding:8px; min-width:0; }
.signal-title { color:var(--muted); font-size:11px; margin-bottom:6px; }
.signal-row { display:flex; align-items:center; gap:5px; }
.lamp { width:12px; height:12px; border-radius:50%; background:#d9dee7; box-shadow:inset 0 0 0 1px #c6ced9; }
.lamp.red.on { background:#dc2626; box-shadow:0 0 8px #dc262680; }
.lamp.amber.on { background:#d97706; box-shadow:0 0 8px #d9770680; }
.lamp.green.on { background:#059669; box-shadow:0 0 8px #05966980; }
.signal-text { margin-left:4px; min-width:0; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; font-size:12px; font-weight:650; }
pre { margin:0; white-space:pre-wrap; font:12px ui-monospace,SFMono-Regular,Consolas,monospace; color:#334155; }
.legend { display:flex; gap:16px; flex-wrap:wrap; margin:8px 0 0; color:var(--muted); font-size:12px; }
.dot { display:inline-block; width:10px; height:10px; border-radius:50%; margin-right:5px; }
.image-grid { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:8px; }
.image-grid figure { margin:0; border:1px solid #e5eaf1; border-radius:6px; overflow:hidden; background:#f8fafc; }
.image-grid img { width:100%; aspect-ratio:16/9; object-fit:contain; display:block; background:#eef2f7; }
.image-grid figcaption { padding:4px 7px; color:var(--muted); font-size:11px; }
.image-status { margin-top:5px; color:var(--muted); font-size:11px; }
.warning { color:#b45309; }
table { width:100%; border-collapse:collapse; font-size:11px; }
td { border-bottom:1px solid #edf0f4; padding:4px 3px; text-align:right; }
td:first-child { text-align:left; color:var(--muted); }
@media(max-width:900px){ .layout{grid-template-columns:1fr;} .plot-panel{min-height:0;} .live{margin-left:0;} .playback-bar{grid-template-columns:auto minmax(100px,1fr) auto;} .playback-time{grid-column:1/-1;grid-row:2;text-align:left;} }
</style>
</head>
<body>
<header><h1>MACVO + VIO live dashboard</h1><div class="subtitle" id="meta">Waiting for the real data pipeline...</div></header>
<main class="layout">
<section class="panel plot-panel">
  <div class="toolbar"><label>Projection <select id="projection"><option>XY</option><option>XZ</option><option>YZ</option></select></label><button id="reset-view" type="button">Reset view</button><span class="live off" id="live">OFFLINE</span></div>
  <div class="playback-bar"><button id="playback-toggle" type="button" title="Play or pause retained history">Play</button><input id="timeline" type="range" min="0" max="0" value="0" step="1" aria-label="Replay progress"><span class="playback-time" id="playback-time">Waiting...</span><button id="go-live" type="button" title="Return to the newest available frame">Latest</button></div>
  <canvas id="trajectory" width="1200" height="650"></canvas>
  <div class="legend"><span><i class="dot" style="background:#111827"></i>GT</span><span><i class="dot" style="background:#2563eb"></i>MACVO raw</span><span><i class="dot" style="background:#059669"></i>VIO committed</span></div>
  <div class="section"><h2>Current stereo pair</h2><div class="image-grid"><figure><img id="image-left" alt="Left camera"><figcaption>Left</figcaption></figure><figure><img id="image-right" alt="Right camera"><figcaption>Right</figcaption></figure></div><div class="image-status" id="image-status">Waiting for stereo images...</div></div>
  <div class="section"><div class="section-head"><h2>Recent IMU samples</h2><button id="reset-imu-view" type="button">Latest</button></div><canvas id="imu" width="1200" height="250"></canvas></div>
</section>
<aside>
  <section class="panel"><h2>Pipeline status</h2><div class="pipeline-lights" id="stage-lights"></div><div class="cards" id="metrics"></div></section>
  <section class="panel section"><h2>Current IMU-center state</h2><pre id="state">Waiting...</pre></section>
  <section class="panel section"><h2>Optimizer diagnostics</h2><pre id="diag">Waiting...</pre></section>
  <section class="panel section"><h2>Coordinate contract</h2><pre id="contract">Waiting...</pre></section>
  <section class="panel section"><h2>Recent frames</h2><table><tbody id="recent"></tbody></table></section>
</aside>
</main>
<script>
const $ = id => document.getElementById(id);
let latest = null;
let playback={live:true,playing:false,frameIdx:null,index:0,requestToken:0,stereo:null};
const colors = {gt:'#111827', raw:'#2563eb', committed:'#059669'};
function fmt(v,n=4){ return Number.isFinite(Number(v)) ? Number(v).toFixed(n) : '—'; }
function vec(v,n=4){ return v && v.length ? '['+v.map(x=>fmt(x,n)).join(', ')+']' : '—'; }
function signal(name,level,text){return '<div class="signal"><div class="signal-title">'+name+'</div><div class="signal-row"><i class="lamp red '+(level==='red'?'on':'')+'"></i><i class="lamp amber '+(level==='amber'?'on':'')+'"></i><i class="lamp green '+(level==='green'?'on':'')+'"></i><span class="signal-text">'+text+'</span></div></div>';}
function playbackIndex(s){
  const h=s?.history||[];if(!h.length)return -1;if(playback.live)return h.length-1;
  const matched=h.findIndex(x=>Number(x.frame_idx)===Number(playback.frameIdx));
  return matched>=0?matched:Math.max(0,Math.min(h.length-1,Number(playback.index)||0));
}
function playbackEntry(s){const h=s?.history||[],i=playbackIndex(s);return i>=0?h[i]:null;}
function playbackDisplayState(s){const h=s?.history||[],i=playbackIndex(s),entry=i>=0?h[i]:null;return {...s,history:i>=0?h.slice(0,i+1):[],frame_idx:entry?.frame_idx??s.frame_idx,timestamp_ns:entry?.timestamp_ns??s.timestamp_ns};}
function playbackTimestampNs(s){return playbackEntry(s)?.timestamp_ns??s?.timestamp_ns??null;}
function setStageLights(s){
  const raw=Number.isFinite(Number(s.raw_latest_frame_idx))?Number(s.raw_latest_frame_idx):null;
  const committed=Number.isFinite(Number(s.committed_latest_frame_idx))?Number(s.committed_latest_frame_idx):null;
  const frontendLevel=raw===null?'red':'green';
  const frontendText=raw===null?'waiting':'frame '+raw;
  let vioLevel='red',vioText='waiting';
  if(!s.static_initialized){vioLevel='amber';vioText='IMU initializing';}
  else if(committed!==null){const lag=raw===null?0:Math.max(0,raw-committed);vioLevel=lag<=1?'green':'amber';vioText='frame '+committed+(lag?' · lag '+lag:'');}
  const backend=s.optimizer?.backend||'two_state';
  $('stage-lights').innerHTML=signal('MACVO frontend',frontendLevel,frontendText)+signal('VIO · '+backend,vioLevel,vioText);
}
function setMetrics(s){
  const c=s.current||{}; const p=c.position||[];
  setStageLights(s);
  const d=s.optimizer||{};
  const rows=[['Frame',s.frame_idx],['Time',fmt((s.timestamp_ns||0)*1e-9,3)+' s'],['Static init',s.static_initialized?'done':'active'],['IMU samples',s.imu_recent?.length||s.static_sample_count||0],['Frontend',fmt(s.frontend_ms,1)+' ms'],['Backend',String(d.backend||'two_state')+' · '+fmt(s.backend_ms,1)+' ms'],['History revision',d.history_revision?('yes · '+(d.state_count??'')+' states'):('no · '+(d.state_count??'')+' states')],['Position',vec(p,3)]];
  $('metrics').innerHTML=rows.map(r=>'<div class="metric"><div class="k">'+r[0]+'</div><div class="v">'+r[1]+'</div></div>').join('');
  $('state').textContent='position  '+vec(c.position,5)+'\nvelocity  '+vec(c.velocity,5)+'\nacc bias  '+vec(c.acc_bias,6)+'\ngyro bias '+vec(c.gyro_bias,6)+'\nquaternion '+vec(c.orientation,6);
  const rs=s.raw_solver||{}; $('diag').textContent='cost total '+fmt(d.cost_total,4)+'\npose cost  '+fmt(d.pose_cost,4)+'\nIMU cost   '+fmt(d.imu_cost,4)+'\np/v cost   '+fmt(d.bias_cost,4)+'\niterations '+(d.iterations??'—')+'\nbackend '+(d.backend||'two_state')+'\nupdate '+fmt(d.update_ms,3)+' ms\nvisual path '+(d.visual_action||'—')+'\ncompression '+(rs.pace_compression_source||rs.t2_compression_source||'—')+'\nstatic ZUPT '+(s.static_zupt_active?'active':'inactive');
  const ct=s.contract||{}; $('contract').textContent='world frame: '+(ct.world_frame||'NWU')+'\nreference: '+(ct.reference_point||'IMU origin')+'\norigin: '+(ct.origin||'first valid IMU pose')+'\npose source: '+(ct.pose_source||'T_WI = T_WC * T_CI');
  $('meta').textContent=(s.meta?.project||'PACE-VIO')+' · '+(s.meta?.scene||'scene')+' · '+(s.meta?.mode||'real pipeline');
}
function project(p,mode){ if(mode==='XZ') return [p[0],p[2]]; if(mode==='YZ') return [p[1],p[2]]; return [p[0],p[1]]; }
function drawArrow(ctx,a,b,color){ if(!a||!b) return; const dx=b[0]-a[0],dy=b[1]-a[1],l=Math.hypot(dx,dy)||1; const ux=dx/l,uy=dy/l; const h=10; ctx.strokeStyle=color;ctx.fillStyle=color;ctx.lineWidth=2.4;ctx.beginPath();ctx.moveTo(a[0],a[1]);ctx.lineTo(b[0],b[1]);ctx.stroke();ctx.beginPath();ctx.moveTo(b[0],b[1]);ctx.lineTo(b[0]-h*ux+h*.55*uy,b[1]-h*uy-h*.55*ux);ctx.lineTo(b[0]-h*ux-h*.55*uy,b[1]-h*uy+h*.55*ux);ctx.closePath();ctx.fill(); }
let imuView={manual:false,center:null,span:8,drag:null,lockToCursor:false};
function imuCanvasPoint(e){const c=$('imu'),r=c.getBoundingClientRect();return{x:(e.clientX-r.left)*c.width/r.width,y:(e.clientY-r.top)*c.height/r.height};}
function resetImuView(){imuView={manual:false,center:null,span:8,drag:null,lockToCursor:false};if(latest)drawImu(latest,null);}
function drawImu(s,cursorTimestampNs=null){
 const c=$('imu'),ctx=c.getContext('2d'),all=(s.imu_recent||[]).filter(q=>Number.isFinite(Number(q.t))).slice().sort((a,b)=>a.t-b.t);ctx.clearRect(0,0,c.width,c.height);
 if(!all.length){ctx.fillStyle='#64748b';ctx.font='13px sans-serif';ctx.fillText('Waiting for IMU samples...',20,28);return;}
 const first=all[0].t*1e-9,last=all[all.length-1].t*1e-9,cursor=cursorTimestampNs===null?NaN:Number(cursorTimestampNs)*1e-9;
 if(Number.isFinite(cursor)&&imuView.lockToCursor){imuView.center=cursor;}else if(!imuView.manual||imuView.center===null){imuView.span=Math.min(8,Math.max(1,last-first||1));imuView.center=last-imuView.span/2;}
 imuView.span=Math.max(.2,Math.min(Math.max(8,last-first+1),imuView.span));const t0=imuView.center-imuView.span/2,t1=imuView.center+imuView.span/2;let a=all.filter(q=>q.t*1e-9>=t0&&q.t*1e-9<=t1);if(!a.length)a=[all.reduce((best,q)=>Math.abs(q.t*1e-9-imuView.center)<Math.abs(best.t*1e-9-imuView.center)?q:best,all[0])];
 const vals=[];for(const q of a)for(const k of ['gx','gy','gz','ax','ay','az'])if(Number.isFinite(Number(q[k])))vals.push(Number(q[k]));let lo=Math.min(...vals),hi=Math.max(...vals);const pad=(hi-lo||1)*.1;lo-=pad;hi+=pad;const L=52,R=12,T=12,B=30,sy=(c.height-T-B)/(hi-lo||1),x=t=>L+(t-t0)*(c.width-L-R)/(t1-t0),y=v=>c.height-B-(v-lo)*sy;
 ctx.strokeStyle='#e5eaf1';ctx.lineWidth=1;ctx.fillStyle='#64748b';ctx.font='10px sans-serif';for(let i=0;i<5;i++){const yy=T+(c.height-T-B)*i/4;ctx.beginPath();ctx.moveTo(L,yy);ctx.lineTo(c.width-R,yy);ctx.stroke();ctx.fillText(fmt(hi-(hi-lo)*i/4,2),4,yy+3);}for(let i=0;i<6;i++){const xx=L+(c.width-L-R)*i/5;ctx.beginPath();ctx.moveTo(xx,T);ctx.lineTo(xx,c.height-B);ctx.stroke();ctx.fillText(fmt(t0+(t1-t0)*i/5,2)+'s',xx-15,c.height-11);}
 [['gx','#2563eb'],['gy','#7c3aed'],['gz','#dc2626'],['ax','#059669'],['ay','#d97706'],['az','#0f766e']].forEach(([key,col])=>{ctx.strokeStyle=col;ctx.lineWidth=1.5;ctx.beginPath();let started=false;for(const q of a){const value=Number(q[key]);if(!Number.isFinite(value)){started=false;continue;}const xx=x(q.t*1e-9),yy=y(value);if(started)ctx.lineTo(xx,yy);else{ctx.moveTo(xx,yy);started=true;}}ctx.stroke();});
 if(Number.isFinite(cursor)&&cursor>=t0&&cursor<=t1){const xx=x(cursor);ctx.save();ctx.strokeStyle='#111827';ctx.lineWidth=2;ctx.setLineDash([5,4]);ctx.beginPath();ctx.moveTo(xx,T);ctx.lineTo(xx,c.height-B);ctx.stroke();ctx.setLineDash([]);ctx.fillStyle='#111827';ctx.font='10px sans-serif';ctx.fillText('play '+fmt(cursor,3)+'s',Math.min(c.width-92,xx+5),T+12);ctx.restore();}
 ctx.fillStyle='#64748b';ctx.font='11px sans-serif';ctx.fillText('gyro: blue/purple/red · acc: green/orange/teal',L, c.height-3);
}
let view = {manual:false, center:null, scale:null, drag:null};
function trajectoryPoints(hist,mode){const out=[];for(const h of hist)for(const k of ['gt','raw','committed'])if(h[k])out.push(project(h[k],mode));return out;}
function defaultView(points,c){const L=56,R=20,T=20,B=42;if(!points.length)return {center:[0,0],scale:1};let minx=Math.min(...points.map(p=>p[0])),maxx=Math.max(...points.map(p=>p[0])),miny=Math.min(...points.map(p=>p[1])),maxy=Math.max(...points.map(p=>p[1]));const pad=Math.max(0.2,Math.max(maxx-minx,maxy-miny)*.08);minx-=pad;maxx+=pad;miny-=pad;maxy+=pad;const sx=(c.width-L-R)/(maxx-minx||1),sy=(c.height-T-B)/(maxy-miny||1);return {center:[(minx+maxx)/2,(miny+maxy)/2],scale:Math.min(sx,sy)*.92};}
function drawTrajectory(s){const c=$('trajectory'),ctx=c.getContext('2d'),hist=(s.history||[]),mode=$('projection').value;ctx.clearRect(0,0,c.width,c.height);const points=trajectoryPoints(hist,mode);if(!points.length){ctx.fillStyle='#64748b';ctx.font='16px sans-serif';ctx.fillText('Waiting for trajectory samples...',30,40);return;}const recent=hist.slice(-300),base=defaultView(trajectoryPoints(recent,mode),c);if(!view.manual||!view.center||!view.scale){let tail=null;for(let i=hist.length-1;i>=0&&!tail;i--){const h=hist[i];tail=h.committed||h.raw||h.gt||null;}view.center=tail?project(tail,mode):base.center;view.scale=base.scale;}const midX=c.width/2,midY=c.height/2,xy=p=>[midX+(p[0]-view.center[0])*view.scale,midY-(p[1]-view.center[1])*view.scale];ctx.strokeStyle='#e4eaf2';ctx.lineWidth=1;for(let i=0;i<8;i++){const x=56+(c.width-76)*i/7;ctx.beginPath();ctx.moveTo(x,20);ctx.lineTo(x,c.height-42);ctx.stroke();const y=20+(c.height-62)*i/7;ctx.beginPath();ctx.moveTo(56,y);ctx.lineTo(c.width-20,y);ctx.stroke();}for(const k of ['gt','raw','committed']){ctx.strokeStyle=colors[k];ctx.lineWidth=k==='gt'?3:2;ctx.beginPath();let started=false;for(const h of hist){if(!h[k]){started=false;continue;}const q=xy(project(h[k],mode));if(!started){ctx.moveTo(q[0],q[1]);started=true;}else ctx.lineTo(q[0],q[1]);}ctx.stroke();}for(const k of ['raw','committed']){const vals=hist.filter(h=>h[k]);if(vals.length){const h=vals[vals.length-1],q=xy(project(h[k],mode));const prev=vals[Math.max(0,vals.length-4)];const a=xy(project(prev[k],mode));drawArrow(ctx,a,q,colors[k]);}}ctx.fillStyle='#64748b';ctx.font='12px sans-serif';ctx.fillText(mode+' · x / m',c.width/2-25,c.height-12);ctx.save();ctx.translate(14,c.height/2+30);ctx.rotate(-Math.PI/2);ctx.fillText(mode+' · y / m',0,0);ctx.restore();}
function currentTrajectoryState(){return latest?playbackDisplayState(latest):null;}
function currentImuCursor(){return latest&&!playback.live?playbackTimestampNs(latest):null;}
function redrawTrajectory(){const s=currentTrajectoryState();if(s)drawTrajectory(s);}
function redrawImu(){if(latest)drawImu(latest,currentImuCursor());}
function resetView(){view={manual:false,center:null,scale:null,drag:null};redrawTrajectory();}
function canvasPoint(e){const r=$('trajectory').getBoundingClientRect(),c=$('trajectory');return{x:(e.clientX-r.left)*c.width/r.width,y:(e.clientY-r.top)*c.height/r.height};}
const trajectoryCanvas=$('trajectory');trajectoryCanvas.addEventListener('pointerdown',e=>{view.manual=true;view.drag=canvasPoint(e);trajectoryCanvas.setPointerCapture(e.pointerId);});trajectoryCanvas.addEventListener('pointermove',e=>{if(!view.drag||!view.scale)return;const p=canvasPoint(e),dx=p.x-view.drag.x,dy=p.y-view.drag.y;view.center[0]-=dx/view.scale;view.center[1]+=dy/view.scale;view.drag=p;redrawTrajectory();});trajectoryCanvas.addEventListener('pointerup',e=>{view.drag=null;trajectoryCanvas.releasePointerCapture(e.pointerId);});trajectoryCanvas.addEventListener('pointercancel',()=>{view.drag=null;});trajectoryCanvas.addEventListener('wheel',e=>{e.preventDefault();if(!view.scale)return;const p=canvasPoint(e),midX=trajectoryCanvas.width/2,midY=trajectoryCanvas.height/2,before=[view.center[0]+(p.x-midX)/view.scale,view.center[1]-(p.y-midY)/view.scale];view.manual=true;view.scale*=e.deltaY<0?1.15:.87;view.center=[before[0]-(p.x-midX)/view.scale,before[1]+(p.y-midY)/view.scale];redrawTrajectory();},{passive:false});
const imuCanvas=$('imu');imuCanvas.addEventListener('pointerdown',e=>{imuView.manual=true;imuView.lockToCursor=false;imuView.drag=imuCanvasPoint(e);imuCanvas.setPointerCapture(e.pointerId);});imuCanvas.addEventListener('pointermove',e=>{if(!imuView.drag)return;const p=imuCanvasPoint(e),usable=Math.max(1,imuCanvas.width-64);imuView.center-=(p.x-imuView.drag.x)*imuView.span/usable;imuView.drag=p;redrawImu();});imuCanvas.addEventListener('pointerup',e=>{imuView.drag=null;imuCanvas.releasePointerCapture(e.pointerId);});imuCanvas.addEventListener('pointercancel',()=>{imuView.drag=null;});imuCanvas.addEventListener('wheel',e=>{e.preventDefault();const p=imuCanvasPoint(e),L=52,R=12,ratio=Math.max(0,Math.min(1,(p.x-L)/Math.max(1,imuCanvas.width-L-R))),anchor=imuView.center-imuView.span/2+ratio*imuView.span,newSpan=imuView.span*(e.deltaY<0?.82:1.22);imuView.manual=true;imuView.span=Math.max(.2,newSpan);imuView.center=anchor+(0.5-ratio)*imuView.span;redrawImu();},{passive:false});
function drawRecent(s){const a=(s.history||[]).slice(-8).reverse();$('recent').innerHTML=a.map(h=>'<tr><td>'+h.frame_idx+'</td><td>'+fmt((h.timestamp_ns||0)*1e-9,2)+' s</td><td>'+(h.raw?'raw':'')+' '+(h.committed?'commit':'')+'</td></tr>').join('');}
function updateStereo(s){const im=s.stereo_images||{};if(im.left)$('image-left').src=im.left;if(im.right)$('image-right').src=im.right;$('image-status').textContent=im.timestamp_ns?'frame '+(s.frame_idx??'')+' · '+fmt(Number(im.timestamp_ns)*1e-9,3)+' s':'Waiting for stereo images...';}
let imuArchive=[],imuArchiveKey=null,imuArchiveLast=null;
function mergeImuArchive(s){const key=s.meta?.scene||'scene',incoming=s.imu_recent||[],incomingMax=incoming.length?Math.max(...incoming.map(q=>Number(q.t)).filter(Number.isFinite)):null;if(key!==imuArchiveKey||(Number.isFinite(incomingMax)&&imuArchiveLast!==null&&incomingMax<imuArchiveLast)){imuArchiveKey=key;imuArchive=[];imuArchiveLast=null;}for(const q of incoming){const t=Number(q.t);if(Number.isFinite(t)&&(imuArchiveLast===null||t>imuArchiveLast)){imuArchive.push(q);imuArchiveLast=t;}}s.imu_recent=imuArchive;}
function syncTimeline(s){const h=s.history||[],slider=$('timeline');slider.max=String(Math.max(0,h.length-1));if(!h.length){slider.value='0';$('playback-time').textContent='Waiting...';return;}let i=playbackIndex(s);if(playback.live){i=h.length-1;playback.frameIdx=h[i].frame_idx;}else if(i<0){i=0;playback.frameIdx=h[0].frame_idx;}playback.index=i;slider.value=String(i);const e=h[i],tail=h[h.length-1],mode=playback.live?'LIVE':'REPLAY';$('playback-time').textContent=mode+' · f'+e.frame_idx+' · '+fmt(Number(e.timestamp_ns)*1e-9,3)+' / '+fmt(Number(tail.timestamp_ns)*1e-9,3)+' s';$('playback-time').classList.toggle('replay',!playback.live);$('playback-toggle').textContent=playback.playing?'Pause':'Play';}
async function loadReplayStereo(frameIdx){const token=++playback.requestToken;$('image-status').textContent='Loading replay frame '+frameIdx+'...';try{const r=await fetch('/api/replay?frame_idx='+encodeURIComponent(frameIdx)+'&ts='+Date.now(),{cache:'no-store'});if(!r.ok)throw Error('unavailable');const item=await r.json();if(token!==playback.requestToken||playback.live||Number(playback.frameIdx)!==Number(frameIdx))return;playback.stereo=item;updateStereo(item);}catch(e){if(token===playback.requestToken&&!playback.live)$('image-status').textContent='Stereo image unavailable near frame '+frameIdx;}}
function renderPlayback(){if(!latest)return;syncTimeline(latest);redrawTrajectory();redrawImu();drawRecent(latest);if(playback.live)updateStereo(latest);else if(playback.stereo)updateStereo(playback.stereo);}
function selectPlaybackIndex(index,loadStereo=true){const h=latest?.history||[];if(!h.length)return;const i=Math.max(0,Math.min(h.length-1,Number(index)||0)),e=h[i];playback.live=false;playback.index=i;playback.frameIdx=e.frame_idx;playback.stereo=null;imuView.manual=true;imuView.lockToCursor=true;imuView.center=Number(e.timestamp_ns)*1e-9;renderPlayback();if(loadStereo)loadReplayStereo(e.frame_idx);}
function goLive(){playback.live=true;playback.playing=false;playback.stereo=null;playback.requestToken++;imuView.manual=false;imuView.center=null;imuView.lockToCursor=false;renderPlayback();}
let playbackTimer=null;
function setPlaybackPlaying(value){playback.playing=Boolean(value);if(playbackTimer){clearInterval(playbackTimer);playbackTimer=null;}if(playback.playing){if(playback.live)selectPlaybackIndex(0,true);playbackTimer=setInterval(()=>{const h=latest?.history||[],i=playbackIndex(latest);if(!h.length||i>=h.length-1){setPlaybackPlaying(false);return;}selectPlaybackIndex(i+1,true);},80);}syncTimeline(latest||{});}
async function poll(){try{const r=await fetch('/api/state?ts='+Date.now(),{cache:'no-store'});if(!r.ok)throw Error('offline');latest=await r.json();mergeImuArchive(latest);$('live').textContent='LIVE';$('live').classList.remove('off');setMetrics(latest);renderPlayback();}catch(e){$('live').textContent='OFFLINE';$('live').classList.add('off');}setTimeout(poll,150);}
$('timeline').addEventListener('input',e=>{const index=Number(e.target.value);setPlaybackPlaying(false);selectPlaybackIndex(index,true);});$('playback-toggle').addEventListener('click',()=>setPlaybackPlaying(!playback.playing));$('go-live').addEventListener('click',goLive);$('projection').addEventListener('change',()=>{view={manual:false,center:null,scale:null,drag:null};redrawTrajectory();});$('reset-view').addEventListener('click',resetView);$('reset-imu-view').addEventListener('click',resetImuView);poll();
</script>
</body></html>'''


def _finite(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _finite(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite(v) for v in value]
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    return str(value)


def _scalar(value: Any) -> Any:
    """Convert torch/numpy scalar-like diagnostics without importing torch."""
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "item"):
        try:
            return value.item()
        except ValueError:
            return value
    return value


class LiveStateStore:
    """Latest-only state store; publishing never waits on the browser."""

    def __init__(self, max_history: int = 3000) -> None:
        self._lock = threading.Lock()
        self._state: dict[str, Any] = {"history": [], "updated_at": None}
        self._history: dict[int, dict[str, Any]] = {}
        self._replay: dict[int, dict[str, Any]] = {}
        self._max_history = int(max_history)

    def publish(self, payload: dict[str, Any]) -> None:
        frame_idx = payload.get("frame_idx")
        if frame_idx is not None:
            with self._lock:
                snapshot = dict(payload)
                history_revision = snapshot.pop("committed_history_revision", None) or []
                for revised in history_revision:
                    revised_index = int(revised["frame_idx"])
                    revised_entry = self._history.get(revised_index)
                    if revised_entry is None:
                        revised_entry = {
                            "frame_idx": revised_index,
                            "timestamp_ns": revised.get("timestamp_ns"),
                        }
                        self._history[revised_index] = revised_entry
                    elif revised.get("timestamp_ns") is not None:
                        revised_entry["timestamp_ns"] = revised["timestamp_ns"]
                    if revised.get("committed") is not None:
                        revised_entry["committed"] = revised["committed"]
                entry = self._history.get(int(frame_idx))
                if entry is None:
                    entry = {
                        "frame_idx": int(frame_idx),
                        "timestamp_ns": payload.get("timestamp_ns"),
                    }
                    self._history[int(frame_idx)] = entry
                for key in ("gt", "raw", "committed"):
                    if payload.get(key) is not None:
                        entry[key] = payload[key]
                stereo = payload.get("stereo_images") or {}
                if (
                    stereo.get("timestamp_ns") is not None
                    and int(stereo["timestamp_ns"]) == int(payload.get("timestamp_ns", -1))
                    and (stereo.get("left") is not None or stereo.get("right") is not None)
                ):
                    self._replay[int(frame_idx)] = {
                        "frame_idx": int(frame_idx),
                        "timestamp_ns": payload.get("timestamp_ns"),
                        "stereo_images": stereo,
                    }
                history_items = [self._history[key] for key in sorted(self._history)][-self._max_history:]
                self._history = {int(item["frame_idx"]): item for item in history_items}
                retained = set(self._history)
                self._replay = {
                    index: replay for index, replay in self._replay.items()
                    if index in retained
                }
                snapshot["history"] = history_items
                snapshot["updated_at"] = time.time()
                self._state = _finite(snapshot)
        else:
            with self._lock:
                snapshot = dict(payload)
                snapshot["history"] = list(self._history.values())[-self._max_history:]
                snapshot["updated_at"] = time.time()
                self._state = _finite(snapshot)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return copy.deepcopy(self._state)

    def replay_snapshot(self, frame_idx: int) -> dict[str, Any] | None:
        """Return the closest retained stereo frame at or before frame_idx."""
        with self._lock:
            candidates = [index for index in self._replay if index <= int(frame_idx)]
            if not candidates:
                candidates = [index for index in self._replay if index >= int(frame_idx)]
            if not candidates:
                return None
            index = max(candidates) if max(candidates) <= int(frame_idx) else min(candidates)
            return copy.deepcopy(self._replay[index])


class _Handler(BaseHTTPRequestHandler):
    server_version = "MACVO-Live/1.0"

    def log_message(self, *_args) -> None:
        return

    def _send(self, content: bytes, content_type: str, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(content)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/api/state":
            payload = json.dumps(self.server.live_store.snapshot(), allow_nan=False).encode("utf-8")
            self._send(payload, "application/json; charset=utf-8")
            return
        if parsed.path == "/api/replay":
            try:
                frame_idx = int(parse_qs(parsed.query).get("frame_idx", [""])[0])
            except (TypeError, ValueError):
                self._send(b'{"error":"invalid frame_idx"}', "application/json", 400)
                return
            replay = self.server.live_store.replay_snapshot(frame_idx)
            if replay is None:
                self._send(b'{"error":"frame unavailable"}', "application/json", 404)
                return
            payload = json.dumps(_finite(replay), allow_nan=False).encode("utf-8")
            self._send(payload, "application/json; charset=utf-8")
            return
        if parsed.path == "/api/health":
            self._send(b'{"ok":true}', "application/json")
            return
        if parsed.path in {"/", "/index.html"}:
            self._send(_HTML.encode("utf-8"), "text/html; charset=utf-8")
            return
        self._send(b"not found", "text/plain", 404)


class LiveDashboard:
    def __init__(self, host: str, port: int, *, dataset_root: str | Path | None = None) -> None:
        self.host = str(host)
        self.port = int(port)
        self.store = LiveStateStore()
        self.server = ThreadingHTTPServer((self.host, self.port), _Handler)
        self.server.live_store = self.store
        self.thread = threading.Thread(target=self.server.serve_forever, name="macvo-live-dashboard", daemon=True)
        self.dataset_root = Path(dataset_root) if dataset_root is not None else None
        self._gt: dict[int, np.ndarray] = {}
        self._gt_anchor: np.ndarray | None = None
        self._load_gt()
        self._meta: dict[str, Any] = {}
        self._stereo_cache: dict[str, Any] = {}
        self._last_stereo_timestamp_ns: int | None = None
        self._last_stereo_encode_time = 0.0
        self._imu_recent: deque[dict[str, float | int]] = deque(maxlen=6000)
        self._imu_last_timestamp_ns: int | None = None

    @classmethod
    def start(cls, host: str = "127.0.0.1", port: int = 8765, *, dataset_root=None) -> "LiveDashboard":
        dashboard = cls(host, port, dataset_root=dataset_root)
        dashboard.thread.start()
        return dashboard

    def set_meta(self, **values: Any) -> None:
        self._meta.update(values)
        self.store.publish({"meta": self._meta, "contract": self._contract()})

    def _contract(self) -> dict[str, str]:
        return {
            "world_frame": "NWU",
            "reference_point": "IMU center",
            "origin": "first valid IMU pose",
            "pose_source": "T_WI = T_WC * T_CI",
            "raw_source": "independent visual-only UVD solve; no IMU/prior/PACE-VIO pose",
            "committed_source": "PACE-VIO result after optimizer write-back",
        }

    def _load_gt(self) -> None:
        if self.dataset_root is None:
            return
        path = self.dataset_root / "ref_pose.csv"
        metadata_path = self.dataset_root / "metadata.json"
        if not path.exists():
            return
        try:
            body_to_imu = np.zeros(3, dtype=np.float64)
            gt_is_imu_center = False
            if metadata_path.exists():
                meta = json.loads(metadata_path.read_text(encoding="utf-8"))
                ground_truth = meta.get("ground_truth", {})
                trajectory = meta.get("trajectory", {})
                reference_point = str(
                    ground_truth.get(
                        "reference_point",
                        trajectory.get("reference_point", ""),
                    )
                ).strip().lower().replace("_", "").replace(" ", "")
                if reference_point in {
                    "imu",
                    "imucenter",
                    "imuorigin",
                    "imusocket",
                }:
                    gt_is_imu_center = True
                elif reference_point not in {
                    "",
                    "body",
                    "bodycenter",
                    "bodyorigin",
                    "cameraleft",
                    "cameraleftsocket",
                    "leftcamera",
                    "leftcameracenter",
                }:
                    raise ValueError(
                        "unsupported ground-truth reference point: "
                        f"{reference_point!r}"
                    )
                matrix_CI = np.asarray(
                    meta.get("extrinsics", {}).get("T_CI"),
                    dtype=np.float64,
                )
                if matrix_CI.shape != (4, 4):
                    raise ValueError("metadata.extrinsics.T_CI must be 4x4")
                if not gt_is_imu_center:
                    # Legacy HoloOcean ref_pose files use the camera/body
                    # reference point. T_CI translation is expressed in MACVO
                    # camera FRD/NED axes, hence D*t_CI before the lever shift.
                    ned_to_nwu = np.diag([1.0, -1.0, -1.0])
                    body_to_imu = ned_to_nwu @ matrix_CI[:3, 3]
            rows = []
            with path.open("r", newline="", encoding="utf-8") as stream:
                for row in csv.DictReader(stream):
                    ts = int(row["timestamp"])
                    p = np.array([
                        float(row.get("x_m", row.get("x", 0.0))),
                        float(row.get("y_m", row.get("y", 0.0))),
                        float(row.get("z_m", row.get("z", 0.0))),
                    ], dtype=np.float64)
                    q = np.array([
                        float(row.get("qx", 0.0)), float(row.get("qy", 0.0)),
                        float(row.get("qz", 0.0)), float(row.get("qw", 1.0)),
                    ], dtype=np.float64)
                    rows.append((ts, p, q))
            if rows:
                q = np.asarray([row[2] for row in rows])
                R = pp.SO3(__import__("torch").from_numpy(q)).matrix().cpu().numpy()
                p = np.asarray([row[1] for row in rows])
                p_imu = p + np.einsum(
                    "nij,nj->ni",
                    R,
                    np.broadcast_to(body_to_imu, p.shape),
                )
                p_imu -= p_imu[0]
                for row, position in zip(rows, p_imu):
                    self._gt[row[0]] = position.tolist()
        except Exception:
            self._gt = {}

    def _pose_imu_nwu(self, system: Any, index: int) -> tuple[list[float], list[float]] | None:
        frames = system.graph.frames.data
        if index < 0 or index >= len(system.graph.frames):
            return None
        pose = frames["pose"][index].detach().cpu().double().reshape(1, 7).numpy()
        ext = frames["imu_vio_sensor_T_imu"][index].detach().cpu().double().reshape(1, 7).numpy()
        imu_internal = compose_camera_to_imu_poses(pose, ext)
        imu_nwu = convert_pose_world_frame_only(imu_internal, "NED", "NWU")[0]
        if self._gt_anchor is None:
            self._gt_anchor = imu_nwu[:3].copy()
        position = (imu_nwu[:3] - self._gt_anchor).tolist()
        return position, imu_nwu[3:].tolist()

    def _raw_pose_imu_nwu(self, system: Any, index: int) -> tuple[list[float], list[float]] | None:
        """Convert the independent visual-only MACVO pose to IMU center."""
        raw_poses = getattr(system, "_live_macvo_raw_poses", {}) or {}
        raw_pose = raw_poses.get(int(index))
        if raw_pose is None or index < 0 or index >= len(system.graph.frames):
            return None
        try:
            pose = raw_pose.detach().cpu().double().reshape(1, 7).numpy()
            ext = system.graph.frames.data["imu_vio_sensor_T_imu"][index].detach().cpu().double().reshape(1, 7).numpy()
            imu_internal = compose_camera_to_imu_poses(pose, ext)
            imu_nwu = convert_pose_world_frame_only(imu_internal, "NED", "NWU")[0]
            if self._gt_anchor is None:
                self._gt_anchor = imu_nwu[:3].copy()
            return (imu_nwu[:3] - self._gt_anchor).tolist(), imu_nwu[3:].tolist()
        except Exception:
            return None

    @staticmethod
    def _encode_image(image: Any, max_side: int = 640) -> str | None:
        """Make a small JPEG data URL without sending image tensors to the UI."""
        if Image is None or image is None or not hasattr(image, "detach"):
            return None
        try:
            array = image.detach()
            while array.ndim > 3:
                array = array[0]
            array = array.float().cpu().numpy()
            if array.ndim == 2:
                array = np.repeat(array[..., None], 3, axis=-1)
            elif array.ndim == 3 and array.shape[0] in (1, 3, 4):
                array = np.moveaxis(array, 0, -1)
            if array.ndim != 3:
                return None
            if array.shape[-1] == 1:
                array = np.repeat(array, 3, axis=-1)
            if array.shape[-1] > 3:
                array = array[..., :3]
            array = np.nan_to_num(array, nan=0.0, posinf=1.0, neginf=0.0)
            if float(array.max(initial=0.0)) <= 1.5:
                array = array * 255.0
            array = np.clip(array, 0.0, 255.0).astype(np.uint8)
            pil = Image.fromarray(array, mode="RGB")
            pil.thumbnail((max_side, max_side), Image.Resampling.BILINEAR)
            stream = io.BytesIO()
            pil.save(stream, format="JPEG", quality=78, optimize=True)
            encoded = base64.b64encode(stream.getvalue()).decode("ascii")
            return "data:image/jpeg;base64," + encoded
        except Exception:
            return None

    def _encode_stereo(self, frame: Any | None, timestamp_ns: int) -> dict[str, Any]:
        if frame is None or not hasattr(frame, "stereo"):
            return self._stereo_cache
        now = time.monotonic()
        if (
            self._stereo_cache
            and self._last_stereo_timestamp_ns == timestamp_ns
        ) or (
            self._stereo_cache
            and now - self._last_stereo_encode_time < 0.25
        ):
            return self._stereo_cache
        stereo = frame.stereo
        left = self._encode_image(getattr(stereo, "imageL", None))
        right = self._encode_image(getattr(stereo, "imageR", None))
        if left is None and right is None:
            return self._stereo_cache
        self._stereo_cache = {
            "left": left,
            "right": right,
            "timestamp_ns": int(timestamp_ns),
        }
        self._last_stereo_timestamp_ns = int(timestamp_ns)
        self._last_stereo_encode_time = now
        return self._stereo_cache

    def _update_imu_recent(self, frame: Any | None) -> None:
        """Append unseen raw IMU samples independently of dashboard stage."""
        if not hasattr(self, "_imu_recent"):
            self._imu_recent = deque(maxlen=6000)
            self._imu_last_timestamp_ns = None
        if frame is None or not hasattr(frame, "imu") or frame.imu is None:
            return
        if not hasattr(frame.imu, "time_ns") or not frame.imu.time_ns.numel():
            return
        timestamps = frame.imu.time_ns.reshape(-1).detach().cpu().numpy().astype(np.int64)
        acceleration = frame.imu.acc.reshape(-1, 3).detach().cpu().numpy()
        angular_rate = frame.imu.gyro.reshape(-1, 3).detach().cpu().numpy()
        for timestamp, acc, gyro in zip(timestamps, acceleration, angular_rate):
            timestamp = int(timestamp)
            if self._imu_last_timestamp_ns is not None and timestamp <= self._imu_last_timestamp_ns:
                continue
            self._imu_recent.append({
                "t": timestamp,
                "ax": float(acc[0]), "ay": float(acc[1]), "az": float(acc[2]),
                "gx": float(gyro[0]), "gy": float(gyro[1]), "gz": float(gyro[2]),
            })
            self._imu_last_timestamp_ns = timestamp

    def _payload(self, system: Any, frame: Any | None, stage: str) -> dict[str, Any] | None:
        if not hasattr(system, "graph") or len(system.graph.frames) == 0:
            return None
        self._update_imu_recent(frame)
        opt = getattr(system.Optimizer, "last_pair_diagnostics", {}) or {}
        raw_poses = getattr(system, "_live_macvo_raw_poses", {}) or {}
        raw_latest_index = int(max(raw_poses)) if raw_poses else None
        committed_latest_index = opt.get("frame_idx")
        committed_latest_index = (
            int(committed_latest_index) if committed_latest_index is not None else None
        )
        if stage == "vio_committed":
            committed_index = opt.get("frame_idx")
            if committed_index is None:
                return None
            index = int(committed_index)
        elif raw_poses:
            index = int(max(raw_poses))
        else:
            return None
        timestamp_ns = int(system.graph.frames.data["time_ns"][index].item())
        pose = self._pose_imu_nwu(system, index)
        if pose is None:
            return None
        raw_pose = self._raw_pose_imu_nwu(system, index)
        position, orientation = pose
        current = {"position": position, "orientation": orientation}
        frames = system.graph.frames.data
        for key, output in (("imu_vio_velocity_world", "velocity"), ("imu_vio_acc_bias", "acc_bias"), ("imu_vio_gyro_bias", "gyro_bias")):
            if key in frames:
                current[output] = frames[key][index].detach().cpu().float().reshape(-1).tolist()
        imu_losses = [
            _scalar(opt.get("imu_rot_loss")),
            _scalar(opt.get("imu_trans_loss")),
            _scalar(opt.get("imu_vel_loss")),
        ]
        imu_losses = [float(value) for value in imu_losses if value is not None]
        diag = {
            "cost_total": _scalar(opt.get("final_loss")),
            "pose_cost": _scalar(opt.get("visual_loss")),
            "imu_cost": sum(imu_losses) if imu_losses else None,
            "bias_cost": _scalar(opt.get("energy_pv_weighted")),
            "iterations": _scalar(opt.get("two_state_solver_iterations")),
            "visual_action": opt.get("visual_pose_gate_action"),
            "backend": opt.get("vio_backend", "two_state"),
            "update_ms": _scalar(opt.get("isam2_update_ms")),
            "state_count": _scalar(opt.get("isam2_state_count")),
            "history_revision": bool(opt.get("isam2_history_revision", False)),
        }
        committed_history_revision = None
        if (
            stage == "vio_committed"
            and diag["backend"] == "isam2"
            and diag["history_revision"]
            and diag["state_count"] is not None
        ):
            state_count = max(0, int(diag["state_count"]))
            first_index = max(0, index - state_count + 1)
            committed_history_revision = []
            for revised_index in range(first_index, index + 1):
                revised_pose = self._pose_imu_nwu(system, revised_index)
                if revised_pose is None:
                    continue
                revised_timestamp = int(frames["time_ns"][revised_index].item())
                committed_history_revision.append({
                    "frame_idx": revised_index,
                    "timestamp_ns": revised_timestamp,
                    "committed": revised_pose[0],
                })
        pending = getattr(system, "_pipeline_pending", None) or {}
        static_diag = getattr(system, "_imu_static_init_diag", {}) or {}
        stereo_images = self._encode_stereo(frame, timestamp_ns)
        payload: dict[str, Any] = {
            "frame_idx": index,
            "timestamp_ns": timestamp_ns,
            "stage": stage,
            "raw_latest_frame_idx": raw_latest_index,
            "committed_latest_frame_idx": committed_latest_index,
            # A committed publication includes both trajectories at the exact
            # same frame index.  This prevents the one-frame-delayed backend
            # result from being compared with the next frontend frame.
            "raw": raw_pose[0] if raw_pose is not None else None,
            "committed": position if stage == "vio_committed" else None,
            "gt": self._gt.get(timestamp_ns),
            "stereo_images": stereo_images,
            "current": current,
            "optimizer": diag,
            "frontend_ms": pending.get("frontend_ms"),
            "backend_ms": _scalar(opt.get("local_ba_optimize_total_s")) * 1000.0 if opt.get("local_ba_optimize_total_s") is not None else None,
            "static_initialized": bool(getattr(system, "_imu_static_initialized", True)),
            "static_zupt_active": bool(getattr(system, "_imu_static_zupt_active", False)),
            "static_sample_count": static_diag.get("sample_count", static_diag.get("num_samples", 0)),
            "meta": self._meta,
            "contract": self._contract(),
            "raw_solver": getattr(system, "_live_macvo_raw_last_diagnostics", {}),
            "imu_recent": list(getattr(self, "_imu_recent", ())),
            "committed_history_revision": committed_history_revision,
        }
        return payload

    def publish_system(self, system: Any, frame: Any | None, stage: str) -> None:
        try:
            payload = self._payload(system, frame, stage)
            if payload is not None:
                self.store.publish(payload)
        except Exception:
            # A dashboard must never interrupt the estimator.
            return
