#!/usr/bin/env python3
"""Plot every stored VIO Bias state against the saved HoloOcean Bias truth."""

from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path

import numpy as np


WORKDIR = Path("/home/admin1/macvo-dev")
RESULT = (
    WORKDIR
    / "Results/rectangle_isolated_imu_after_fixes_20260713/trial_1"
    / "vio_preintegrated_full_imuatt_staticinit_calibrated_biasfix_floor_1e-8"
    / "clear_stop_turn_rectangle_truth_bias_no_noise"
)
DATASET = Path(
    "/mnt/e/\u6587\u6863/holoocean/code/recordings/"
    "batch_clear_truth_paths_20260713_static63_variants/"
    "clear_stop_turn_rectangle_truth_bias_no_noise"
)
OUTDIR = WORKDIR / "analysis_rectangle_bias_only_framewise_truth_20260714"
AXES = ("x", "y", "z")
FLU_TO_NED_SIGN = np.asarray([1.0, -1.0, -1.0], dtype=np.float64)


def truth_columns(data: np.ndarray, prefix: str) -> np.ndarray:
    source_flu = np.stack(
        [data[f"{prefix}_{axis}"] for axis in AXES], axis=1
    ).astype(np.float64)
    return source_flu * FLU_TO_NED_SIGN.reshape(1, 3)


def interpolate(time_src: np.ndarray, values: np.ndarray, time_dst: np.ndarray) -> np.ndarray:
    output = np.empty((time_dst.size, 3), dtype=np.float64)
    for axis in range(3):
        output[:, axis] = np.interp(time_dst, time_src, values[:, axis])
    return output


def vector_rmse(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum(values * values, axis=1))))


def axis_rmse(values: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean(values * values, axis=0))


def compact(values: np.ndarray, digits: int = 12) -> list:
    if values.ndim == 1:
        return [round(float(value), digits) for value in values]
    return [[round(float(value), digits) for value in row] for row in values]


def write_frame_csv(
    frame_time: np.ndarray,
    init_index: int,
    estimates: dict[str, np.ndarray],
    truths: dict[str, np.ndarray],
) -> None:
    path = OUTDIR / "framewise_bias_estimate_vs_truth.csv"
    columns = ["frame", "timestamp_ns", "time_s", "phase"]
    for sensor in ("gyro", "acc"):
        for kind in ("estimate", "truth", "error"):
            columns.extend(f"{sensor}_{kind}_{axis}" for axis in AXES)
        columns.extend(
            [
                f"{sensor}_estimate_norm",
                f"{sensor}_truth_norm",
                f"{sensor}_error_norm",
            ]
        )

    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for frame in range(frame_time.size):
            row: dict[str, object] = {
                "frame": frame,
                "timestamp_ns": int(frame_time[frame]),
                "time_s": float(frame_time[frame] * 1e-9),
                "phase": "startup_uninitialized" if frame < init_index else "post_static_init",
            }
            for sensor in ("gyro", "acc"):
                estimate = estimates[sensor][frame]
                truth = truths[sensor][frame]
                error = estimate - truth
                for axis, name in enumerate(AXES):
                    row[f"{sensor}_estimate_{name}"] = float(estimate[axis])
                    row[f"{sensor}_truth_{name}"] = float(truth[axis])
                    row[f"{sensor}_error_{name}"] = float(error[axis])
                row[f"{sensor}_estimate_norm"] = float(np.linalg.norm(estimate))
                row[f"{sensor}_truth_norm"] = float(np.linalg.norm(truth))
                row[f"{sensor}_error_norm"] = float(np.linalg.norm(error))
            writer.writerow(row)


def summarize(
    frame_time: np.ndarray,
    init_index: int,
    estimates: dict[str, np.ndarray],
    truths: dict[str, np.ndarray],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    active = slice(init_index, None)
    for sensor in ("gyro", "acc"):
        estimate = estimates[sensor][active]
        truth = truths[sensor][active]
        error = estimate - truth
        row: dict[str, object] = {
            "sensor": sensor,
            "all_frames": int(frame_time.size),
            "startup_frames": int(init_index),
            "post_init_frames": int(frame_time.size - init_index),
            "init_frame": int(init_index),
            "init_timestamp_ns": int(frame_time[init_index]),
            "init_time_s": float(frame_time[init_index] * 1e-9),
            "post_init_error_vector_rmse": vector_rmse(error),
            "post_init_estimate_drift_norm": float(np.linalg.norm(estimate[-1] - estimate[0])),
            "post_init_estimate_max_change_norm": float(
                np.linalg.norm(estimate - estimate[0], axis=1).max()
            ),
            "post_init_truth_drift_norm": float(np.linalg.norm(truth[-1] - truth[0])),
            "initial_error_norm": float(np.linalg.norm(error[0])),
            "final_error_norm": float(np.linalg.norm(error[-1])),
        }
        rms = axis_rmse(error)
        for axis, name in enumerate(AXES):
            row[f"rmse_{name}"] = float(rms[axis])
            row[f"initial_estimate_{name}"] = float(estimate[0, axis])
            row[f"initial_truth_{name}"] = float(truth[0, axis])
            row[f"final_estimate_{name}"] = float(estimate[-1, axis])
            row[f"final_truth_{name}"] = float(truth[-1, axis])
        rows.append(row)
    return rows


def write_summary_csv(rows: list[dict[str, object]]) -> None:
    path = OUTDIR / "bias_comparison_summary.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_payload(
    frame_time: np.ndarray,
    init_index: int,
    estimates: dict[str, np.ndarray],
    truths: dict[str, np.ndarray],
    summaries: list[dict[str, object]],
) -> dict[str, object]:
    time_s = frame_time.astype(np.float64) * 1e-9
    payload: dict[str, object] = {
        "frame": list(range(frame_time.size)),
        "timestampNs": [int(value) for value in frame_time],
        "time": compact(time_s, 9),
        "initIndex": int(init_index),
        "initTime": float(time_s[init_index]),
        "summaries": summaries,
        "sensors": {},
    }
    for sensor in ("gyro", "acc"):
        estimate = estimates[sensor]
        truth = truths[sensor]
        error = estimate - truth
        payload["sensors"][sensor] = {
            "estimate": compact(estimate),
            "truth": compact(truth),
            "error": compact(error),
            "estimateNorm": compact(np.linalg.norm(estimate, axis=1)),
            "truthNorm": compact(np.linalg.norm(truth, axis=1)),
            "errorNorm": compact(np.linalg.norm(error, axis=1)),
        }
    return payload


def write_html(
    payload: dict[str, object],
    summaries: list[dict[str, object]],
    page_title: str,
    scene_label: str,
    method_label: str,
) -> None:
    by_sensor = {str(row["sensor"]): row for row in summaries}
    cards = []
    for sensor, label, unit in (
        ("gyro", "陀螺仪 Bias", "rad/s"),
        ("acc", "加速度计 Bias", "m/s^2"),
    ):
        row = by_sensor[sensor]
        cards.append(
            "<article class='card'>"
            f"<h2>{label}</h2>"
            f"<span>在线帧数 <strong>{int(row['post_init_frames'])}</strong></span>"
            f"<span>误差向量 RMSE <strong>{float(row['post_init_error_vector_rmse']):.6g} {unit}</strong></span>"
            f"<span>估计值全程变化 <strong>{float(row['post_init_estimate_drift_norm']):.3g} {unit}</strong></span>"
            f"<span>真值首尾漂移 <strong>{float(row['post_init_truth_drift_norm']):.6g} {unit}</strong></span>"
            "</article>"
        )

    template = r"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>__PAGE_TITLE__</title>
<style>
:root{color-scheme:light;--ink:#26323f;--muted:#667483;--line:#d7dee6;--panel:#fff;--bg:#eef1f4;--blue:#1769aa}
*{box-sizing:border-box}body{margin:0;padding:16px;font-family:Arial,"Microsoft YaHei",sans-serif;background:var(--bg);color:var(--ink)}
.shell{max-width:1480px;margin:auto;background:var(--panel);border:1px solid var(--line)}
header{padding:16px 18px 12px;border-bottom:1px solid var(--line)}h1{font-size:22px;margin:0 0 8px;letter-spacing:0}p{margin:5px 0;line-height:1.55;color:var(--muted)}
.cards{display:grid;grid-template-columns:repeat(2,minmax(280px,1fr));gap:8px;margin-top:12px}.card{border:1px solid var(--line);padding:10px;display:grid;gap:5px}.card h2{font-size:14px;margin:0 0 3px}.card span{font-size:12px;color:var(--muted)}.card strong{color:var(--ink)}
.toolbar{display:flex;flex-wrap:wrap;gap:7px;padding:10px 14px;border-bottom:1px solid var(--line);align-items:center}button,select{min-height:34px;border:1px solid #aeb9c5;background:#f8fafc;color:var(--ink);padding:6px 10px;font:inherit;cursor:pointer}button.active{background:var(--blue);border-color:var(--blue);color:#fff}.spacer{flex:1}
#legend{display:flex;flex-wrap:wrap;gap:10px 14px;padding:9px 14px;border-bottom:1px solid var(--line);font-size:12px}#legend label{display:flex;gap:5px;align-items:center;cursor:pointer}
#plotWrap{height:620px;position:relative;background:#fff}canvas{width:100%;height:100%;display:block}#tooltip{position:absolute;display:none;pointer-events:none;background:#fff;border:1px solid #8593a0;padding:7px 9px;font-size:12px;line-height:1.45;box-shadow:0 2px 8px #0002;white-space:nowrap}
.player{display:grid;grid-template-columns:auto auto minmax(220px,1fr) auto;gap:8px;align-items:center;padding:10px 14px;border-top:1px solid var(--line)}input[type=range]{width:100%}#frameLabel{font-variant-numeric:tabular-nums;font-size:12px;min-width:190px;text-align:right}
.readout{overflow:auto;border-top:1px solid var(--line)}table{border-collapse:collapse;width:100%;font-size:12px}th,td{padding:7px 10px;border-right:1px solid #edf0f3;text-align:right;font-variant-numeric:tabular-nums}th:first-child,td:first-child{text-align:left}thead{background:#f7f9fb}
.foot{padding:10px 14px;color:var(--muted);font-size:12px;border-top:1px solid var(--line)}
@media(max-width:760px){body{padding:6px}.cards{grid-template-columns:1fr}#plotWrap{height:520px}.player{grid-template-columns:auto auto 1fr}.player #frameLabel{grid-column:1/-1;text-align:left}}
</style>
</head>
<body><main class="shell">
<header><h1>__PAGE_TITLE__</h1>
<p>场景：<code>__SCENE_LABEL__</code>，3 秒静止后运动。方法：<code>__METHOD_LABEL__</code>。估计值来自 <code>tensor_map.npz</code> 的全部 1890 个相机帧；真值来自同时间戳的 <code>imu_truth_decomposition.csv</code>。</p>
<p>真值由传感器 FLU 转为优化器内部 NED 后比较。浅灰区域是初始化前状态，此时系统实际存储的 Bias 为 0；竖线处完成静止初始化，后续才是在线融合阶段。</p>
<section class="cards">__CARDS__</section></header>
<section class="toolbar">
<button class="active" data-mode="gyro">陀螺仪 Bias</button><button data-mode="acc">加速度计 Bias</button>
<button data-mode="gyroError">陀螺仪误差</button><button data-mode="accError">加速度计误差</button>
<button data-mode="gyroNorm">陀螺仪范数</button><button data-mode="accNorm">加速度计范数</button>
<span class="spacer"></span><button id="full">全时段</button><button id="postInit">初始化后</button><button id="reset">重置视图</button>
</section>
<section id="legend"></section><section id="plotWrap"><canvas id="plot"></canvas><div id="tooltip"></div></section>
<section class="player"><button id="play">播放</button><select id="speed"><option value="1">1x</option><option value="4">4x</option><option value="10">10x</option></select><input id="frame" type="range" min="0" step="1"><span id="frameLabel"></span></section>
<section class="readout"><table><thead><tr><th>量</th><th>x</th><th>y</th><th>z</th><th>向量范数</th></tr></thead><tbody id="readout"></tbody></table></section>
<footer class="foot">鼠标滚轮缩放时间轴，拖动平移，点击图例切换曲线。滑块和播放按钮用于逐帧查看 Bias 状态与真值。</footer>
</main>
<script>
const DATA=__PAYLOAD__;
const AXIS=['x','y','z'], COLORS=['#c44536','#16866f','#276fbf'];
const canvas=document.getElementById('plot'), wrap=document.getElementById('plotWrap'), ctx=canvas.getContext('2d'), legend=document.getElementById('legend'), tip=document.getElementById('tooltip');
const slider=document.getElementById('frame'), frameLabel=document.getElementById('frameLabel'), readout=document.getElementById('readout');
const M={l:88,r:24,t:28,b:58}; let state={mode:'gyro',xDomain:null,visible:{},selected:0,drag:null,playing:false,timer:null};
slider.max=DATA.frame.length-1;
function sensorForMode(){return state.mode.startsWith('gyro')?'gyro':'acc'}
function unit(){return sensorForMode()==='gyro'?'rad/s':'m/s^2'}
function traceSpec(){
  const sensor=sensorForMode(), d=DATA.sensors[sensor], traces=[];
  if(state.mode.endsWith('Error')){
    AXIS.forEach((a,i)=>traces.push({name:`误差 ${a}`,color:COLORS[i],dash:false,y:d.error.map(v=>v[i])}));
    traces.push({name:'误差向量范数',color:'#202832',dash:true,y:d.errorNorm});
  }else if(state.mode.endsWith('Norm')){
    traces.push({name:'估计范数',color:'#202832',dash:true,y:d.estimateNorm});
    traces.push({name:'真值范数',color:'#7a5195',dash:false,y:d.truthNorm});
    traces.push({name:'误差范数',color:'#e07a18',dash:false,y:d.errorNorm});
  }else{
    AXIS.forEach((a,i)=>{
      traces.push({name:`真值 ${a}`,color:COLORS[i],dash:false,y:d.truth.map(v=>v[i])});
      traces.push({name:`估计 ${a}`,color:COLORS[i],dash:true,y:d.estimate.map(v=>v[i])});
    });
  }
  return traces.map(t=>({...t,x:DATA.time}));
}
function resetDomain(post=false){state.xDomain=post?[DATA.initTime,DATA.time.at(-1)]:[DATA.time[0],DATA.time.at(-1)]}
function visibleTraces(){return traceSpec().filter(t=>state.visible[t.name]!==false)}
function domains(){
  if(!state.xDomain)resetDomain(false); let ys=[];
  for(const t of visibleTraces())for(let i=0;i<t.y.length;i++)if(t.x[i]>=state.xDomain[0]&&t.x[i]<=state.xDomain[1]&&Number.isFinite(t.y[i]))ys.push(t.y[i]);
  let lo=Math.min(...ys), hi=Math.max(...ys); if(!Number.isFinite(lo)){lo=-1;hi=1} if(Math.abs(hi-lo)<1e-15){const p=Math.max(Math.abs(lo)*.1,1e-9);lo-=p;hi+=p}
  const pad=(hi-lo)*.09; return{x:state.xDomain,y:[lo-pad,hi+pad]};
}
function mapX(x,d,w){return M.l+(x-d.x[0])/(d.x[1]-d.x[0])*(w-M.l-M.r)}
function mapY(y,d,h){return h-M.b-(y-d.y[0])/(d.y[1]-d.y[0])*(h-M.t-M.b)}
function fmt(v){if(!Number.isFinite(v))return 'n/a';const a=Math.abs(v);return (a!==0&&(a<1e-3||a>=1e4))?v.toExponential(5):v.toFixed(7)}
function draw(){
  const w=wrap.clientWidth,h=wrap.clientHeight,d=domains();ctx.clearRect(0,0,w,h);ctx.font='12px Arial';
  const initX=mapX(DATA.initTime,d,w); if(DATA.initTime>d.x[0]){ctx.fillStyle='#edf0f3';ctx.fillRect(M.l,M.t,Math.max(0,Math.min(initX,w-M.r)-M.l),h-M.t-M.b)}
  ctx.strokeStyle='#d9e0e7';ctx.fillStyle='#34404c';ctx.lineWidth=1;
  for(let i=0;i<=6;i++){const x=d.x[0]+i*(d.x[1]-d.x[0])/6,px=mapX(x,d,w);ctx.beginPath();ctx.moveTo(px,M.t);ctx.lineTo(px,h-M.b);ctx.stroke();ctx.fillText(x.toFixed(1),px-13,h-M.b+21)}
  for(let i=0;i<=5;i++){const y=d.y[0]+i*(d.y[1]-d.y[0])/5,py=mapY(y,d,h);ctx.beginPath();ctx.moveTo(M.l,py);ctx.lineTo(w-M.r,py);ctx.stroke();ctx.fillText(fmt(y),6,py+4)}
  if(DATA.initTime>=d.x[0]&&DATA.initTime<=d.x[1]){ctx.strokeStyle='#5b6773';ctx.setLineDash([4,4]);ctx.beginPath();ctx.moveTo(initX,M.t);ctx.lineTo(initX,h-M.b);ctx.stroke();ctx.setLineDash([]);ctx.fillStyle='#5b6773';ctx.fillText('静止初始化完成',Math.min(initX+5,w-112),M.t+14)}
  ctx.strokeStyle='#202832';ctx.strokeRect(M.l,M.t,w-M.l-M.r,h-M.t-M.b);
  for(const t of visibleTraces()){ctx.strokeStyle=t.color;ctx.lineWidth=t.dash?1.9:2.25;ctx.setLineDash(t.dash?[7,5]:[]);ctx.beginPath();let started=false;for(let i=0;i<t.x.length;i++){const x=t.x[i],y=t.y[i];if(x<d.x[0]||x>d.x[1]||!Number.isFinite(y))continue;const px=mapX(x,d,w),py=mapY(y,d,h);if(!started){ctx.moveTo(px,py);started=true}else ctx.lineTo(px,py)}ctx.stroke()}
  ctx.setLineDash([]);const cursorX=mapX(DATA.time[state.selected],d,w);if(cursorX>=M.l&&cursorX<=w-M.r){ctx.strokeStyle='#111827';ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(cursorX,M.t);ctx.lineTo(cursorX,h-M.b);ctx.stroke()}
  ctx.fillStyle='#26323f';ctx.font='bold 13px Arial';ctx.fillText('time / s',w/2-27,h-14);ctx.save();ctx.translate(18,h/2);ctx.rotate(-Math.PI/2);ctx.fillText(unit(),-25,0);ctx.restore();
}
function buildLegend(){legend.innerHTML='';for(const t of traceSpec()){if(!(t.name in state.visible))state.visible[t.name]=true;const l=document.createElement('label'),c=document.createElement('input'),s=document.createElement('span');c.type='checkbox';c.checked=state.visible[t.name];c.onchange=()=>{state.visible[t.name]=c.checked;draw()};s.textContent=t.name;s.style.color=t.color;l.append(c,s);legend.append(l)}}
function updateReadout(){
  const i=state.selected,s=sensorForMode(),d=DATA.sensors[s],phase=i<DATA.initIndex?'初始化前，状态尚未写入':'静止初始化后 / 在线阶段';slider.value=i;frameLabel.textContent=`frame ${i} · ${DATA.time[i].toFixed(6)} s · ${phase}`;
  const rows=[['估计',d.estimate[i],d.estimateNorm[i]],['真值',d.truth[i],d.truthNorm[i]],['估计 - 真值',d.error[i],d.errorNorm[i]]];
  readout.innerHTML=rows.map(r=>`<tr><td>${r[0]}</td><td>${fmt(r[1][0])}</td><td>${fmt(r[1][1])}</td><td>${fmt(r[1][2])}</td><td>${fmt(r[2])}</td></tr>`).join('');draw();
}
function setSelected(i){state.selected=Math.max(0,Math.min(DATA.frame.length-1,Math.round(i)));updateReadout()}
function resize(){const dpr=devicePixelRatio||1;canvas.width=Math.max(1,Math.round(wrap.clientWidth*dpr));canvas.height=Math.max(1,Math.round(wrap.clientHeight*dpr));ctx.setTransform(dpr,0,0,dpr,0,0);draw()}
document.querySelectorAll('[data-mode]').forEach(b=>b.onclick=()=>{document.querySelectorAll('[data-mode]').forEach(x=>x.classList.remove('active'));b.classList.add('active');state.mode=b.dataset.mode;buildLegend();updateReadout()});
document.getElementById('full').onclick=()=>{resetDomain(false);draw()};document.getElementById('postInit').onclick=()=>{resetDomain(true);setSelected(Math.max(state.selected,DATA.initIndex))};document.getElementById('reset').onclick=()=>{resetDomain(false);setSelected(0)};
slider.oninput=()=>setSelected(Number(slider.value));
document.getElementById('play').onclick=()=>{state.playing=!state.playing;document.getElementById('play').textContent=state.playing?'暂停':'播放';if(state.playing){state.timer=setInterval(()=>{const step=Number(document.getElementById('speed').value);if(state.selected>=DATA.frame.length-1){state.playing=false;clearInterval(state.timer);document.getElementById('play').textContent='播放'}else setSelected(state.selected+step)},33)}else clearInterval(state.timer)};
canvas.onwheel=e=>{e.preventDefault();const r=canvas.getBoundingClientRect(),d=domains(),ratio=Math.max(0,Math.min(1,(e.clientX-r.left-M.l)/(r.width-M.l-M.r))),cx=d.x[0]+ratio*(d.x[1]-d.x[0]),f=e.deltaY<0?.82:1.22;state.xDomain=[cx+(d.x[0]-cx)*f,cx+(d.x[1]-cx)*f];draw()};
canvas.onpointerdown=e=>{state.drag={x:e.clientX,domain:[...domains().x]};canvas.setPointerCapture(e.pointerId)};
canvas.onpointermove=e=>{const r=canvas.getBoundingClientRect(),d=domains();if(state.drag){const span=state.drag.domain[1]-state.drag.domain[0],dx=(e.clientX-state.drag.x)/(wrap.clientWidth-M.l-M.r)*span;state.xDomain=[state.drag.domain[0]-dx,state.drag.domain[1]-dx];draw();return}const ratio=(e.clientX-r.left-M.l)/(r.width-M.l-M.r);if(ratio<0||ratio>1){tip.style.display='none';return}const t=d.x[0]+ratio*(d.x[1]-d.x[0]);let lo=0,hi=DATA.time.length-1;while(lo<hi){const mid=(lo+hi)>>1;if(DATA.time[mid]<t)lo=mid+1;else hi=mid}const i=Math.max(0,Math.min(DATA.time.length-1,lo));const s=sensorForMode(),v=DATA.sensors[s];tip.innerHTML=`frame ${i}<br>${DATA.time[i].toFixed(6)} s<br>估计范数 ${fmt(v.estimateNorm[i])}<br>真值范数 ${fmt(v.truthNorm[i])}<br>误差范数 ${fmt(v.errorNorm[i])}`;tip.style.display='block';tip.style.left=Math.min(e.clientX-r.left+12,r.width-180)+'px';tip.style.top=Math.max(8,e.clientY-r.top-70)+'px'};
canvas.onpointerup=e=>{state.drag=null;try{canvas.releasePointerCapture(e.pointerId)}catch(_){}};canvas.onpointerleave=()=>{tip.style.display='none';state.drag=null};canvas.ondblclick=e=>{const r=canvas.getBoundingClientRect(),d=domains(),ratio=(e.clientX-r.left-M.l)/(r.width-M.l-M.r),t=d.x[0]+ratio*(d.x[1]-d.x[0]);let i=0,best=Infinity;DATA.time.forEach((x,j)=>{const q=Math.abs(x-t);if(q<best){best=q;i=j}});setSelected(i)};
window.onresize=resize;buildLegend();resetDomain(false);updateReadout();resize();
</script></body></html>"""
    page = (
        template.replace("__CARDS__", "".join(cards))
        .replace("__PAYLOAD__", json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
        .replace("__PAGE_TITLE__", html.escape(page_title))
        .replace("__SCENE_LABEL__", html.escape(scene_label))
        .replace("__METHOD_LABEL__", html.escape(method_label))
    )
    (OUTDIR / "interactive_bias_estimate_vs_truth.html").write_text(page, encoding="utf-8")


def write_report(
    summaries: list[dict[str, object]], scene_label: str, method_label: str
) -> None:
    by_sensor = {str(row["sensor"]): row for row in summaries}
    gyro = by_sensor["gyro"]
    acc = by_sensor["acc"]
    lines = [
        "# Bias-only 每帧 Bias 估计与真值对比",
        "",
        "## 数据边界",
        "",
        f"- 数据：`{scene_label}`。",
        f"- 方法：`{method_label}`。",
        "- 估计值：`tensor_map.npz` 中全部 1890 个相机帧的 `imu_vio_acc_bias` / `imu_vio_gyro_bias`。",
        "- 真值：`imu_truth_decomposition.csv` 的 Bias，由 FLU 转到优化器内部 NED，并插值到相机帧时间戳。",
        f"- 初始化写入发生在 frame {int(gyro['init_frame'])}，时间 {float(gyro['init_time_s']):.6f} s。此前存储值为 0，不代表在线优化结果。",
        "",
        "## 结果",
        "",
        f"- 陀螺仪 Bias：初始化后估计值首尾变化 `{float(gyro['post_init_estimate_drift_norm']):.3e} rad/s`，真值首尾漂移 `{float(gyro['post_init_truth_drift_norm']):.6e} rad/s`，误差向量 RMSE `{float(gyro['post_init_error_vector_rmse']):.6e} rad/s`。",
        f"- 加速度计 Bias：初始化后估计值首尾变化 `{float(acc['post_init_estimate_drift_norm']):.3e} m/s^2`，真值首尾漂移 `{float(acc['post_init_truth_drift_norm']):.6e} m/s^2`，误差向量 RMSE `{float(acc['post_init_error_vector_rmse']):.6e} m/s^2`。",
        "- 两类 Bias 在初始化后是否合理，需要结合真值 Bias 是否发生漂移判断：真值漂移而估计不变表示没有在线追踪；真值恒为零且估计接近零则是正确结果。",
        "- 陀螺仪静止初始化通常能够估计启动期均值；加速度计静止初始化无法仅凭单一静止姿态完整分离三轴 Bias 与重力。",
        "",
        "## 产物",
        "",
        "- `interactive_bias_estimate_vs_truth.html`：交互图、逐帧滑块和播放。",
        "- `framewise_bias_estimate_vs_truth.csv`：1890 帧估计、真值和误差。",
        "- `bias_comparison_summary.csv`：初始化后统计。",
    ]
    (OUTDIR / "analysis_summary_cn.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, default=RESULT)
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--outdir", type=Path, default=OUTDIR)
    parser.add_argument(
        "--title", default="Bias-only / no-noise：每帧 Bias 估计与真值"
    )
    parser.add_argument(
        "--scene-label", default="clear_stop_turn_rectangle_truth_bias_no_noise"
    )
    parser.add_argument(
        "--method-label",
        default="vio_preintegrated_full_imuatt_staticinit_calibrated_biasfix_floor_1e-8",
    )
    return parser.parse_args()


def main() -> None:
    global RESULT, DATASET, OUTDIR
    args = parse_args()
    RESULT = args.result
    DATASET = args.dataset
    OUTDIR = args.outdir
    OUTDIR.mkdir(parents=True, exist_ok=True)
    tensor = np.load(RESULT / "tensor_map.npz", allow_pickle=False)
    frame_time = tensor["frames//time_ns"].astype(np.int64)
    estimates = {
        "gyro": tensor["frames//imu_vio_gyro_bias"].astype(np.float64),
        "acc": tensor["frames//imu_vio_acc_bias"].astype(np.float64),
    }
    initialized = np.linalg.norm(estimates["gyro"], axis=1) + np.linalg.norm(
        estimates["acc"], axis=1
    )
    nonzero = np.flatnonzero(initialized > 0.0)
    if nonzero.size == 0:
        raise RuntimeError("No initialized Bias state found in tensor_map.npz")
    init_index = int(nonzero[0])

    truth = np.genfromtxt(DATASET / "imu_truth_decomposition.csv", delimiter=",", names=True)
    truth_time = truth["timestamp"].astype(np.int64)
    truths = {
        "gyro": interpolate(truth_time, truth_columns(truth, "gyro_bias"), frame_time),
        "acc": interpolate(truth_time, truth_columns(truth, "acc_bias"), frame_time),
    }

    summaries = summarize(frame_time, init_index, estimates, truths)
    write_frame_csv(frame_time, init_index, estimates, truths)
    write_summary_csv(summaries)
    payload = build_payload(frame_time, init_index, estimates, truths, summaries)
    write_html(payload, summaries, args.title, args.scene_label, args.method_label)
    write_report(summaries, args.scene_label, args.method_label)
    (OUTDIR / "summary.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(OUTDIR / "interactive_bias_estimate_vs_truth.html")


if __name__ == "__main__":
    main()
