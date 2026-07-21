#!/usr/bin/env python3
"""Plot the three accelerometer axes for the rectangular Bias-only sequence."""

from __future__ import annotations

import csv
import json
from pathlib import Path


WORKDIR = Path("/home/admin1/macvo-dev")
INPUT = (
    WORKDIR
    / "analysis_rectangle_bias_only_imu_corrected_vs_truth_20260714"
    / "imu_samples_corrected_vs_truth.csv"
)
OUTDIR = WORKDIR / "analysis_rectangle_bias_only_accelerometer_axes_20260714"
AXES = ("x", "y", "z")
KINDS = ("truth", "measured", "corrected")


def load_payload() -> dict[str, object]:
    time: list[float] = []
    phase: list[str] = []
    values = {
        kind: {axis: [] for axis in AXES}
        for kind in KINDS
    }
    bias = {axis: [] for axis in AXES}

    with INPUT.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            time.append(float(row["time_s"]))
            phase.append(row["phase"])
            for kind in KINDS:
                for axis in AXES:
                    values[kind][axis].append(float(row[f"acc_{kind}_{axis}"]))
            for axis in AXES:
                bias[axis].append(float(row[f"acc_estimated_bias_{axis}"]))

    init_indices = [
        index for index, value in enumerate(phase) if value == "post_static_init"
    ]
    init_index = init_indices[0] if init_indices else 0
    return {
        "time": time,
        "initIndex": init_index,
        "initTime": time[init_index],
        "values": values,
        "bias": bias,
    }


def write_html(payload: dict[str, object]) -> Path:
    template = r'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>矩形 Bias-only 加速度计三轴数据</title>
<style>
:root{--ink:#202a34;--muted:#66727e;--line:#d8dee5;--bg:#eef1f4;--panel:#fff}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font-family:Arial,"Microsoft YaHei",sans-serif}
main{max-width:1480px;margin:12px auto;background:var(--panel);border:1px solid var(--line)}
header{padding:16px 18px;border-bottom:1px solid var(--line)}h1{font-size:21px;margin:0 0 7px}p{margin:4px 0;color:var(--muted);line-height:1.5;font-size:13px}
.toolbar{display:flex;flex-wrap:wrap;align-items:center;gap:14px;padding:10px 18px;border-bottom:1px solid var(--line);font-size:13px}
.toolbar label{display:flex;align-items:center;gap:6px;cursor:pointer}.toolbar button{border:1px solid #adb8c3;background:#f8fafc;padding:7px 11px;cursor:pointer}.spacer{flex:1}
.plot{height:245px;position:relative;border-bottom:1px solid var(--line)}canvas{width:100%;height:100%;display:block}
#cursor{padding:9px 18px;font-size:12px;overflow:auto}.readout{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}.readout th,.readout td{border:1px solid #e1e6eb;padding:6px 8px;text-align:right}.readout th:first-child,.readout td:first-child{text-align:left}.readout thead{background:#f6f8fa}
.legend{display:inline-flex;align-items:center;gap:5px}.swatch{width:18px;height:3px;display:inline-block}.foot{padding:9px 18px;color:var(--muted);font-size:12px;border-top:1px solid var(--line)}
@media(max-width:720px){main{margin:4px}.plot{height:220px}.toolbar{gap:9px}.spacer{display:none}}
</style></head><body><main>
<header><h1>矩形 Bias-only：加速度计三轴数据</h1>
<p>数据已按融合程序的约定从传感器 FLU 转换到内部 NED。纵轴单位均为 m/s²，静止时 z 轴约为 -9.8 m/s²。</p>
<p>实线为保存的无 Bias 真值，细线为原始测量，虚线为减去估计 Bias 后的观测。该序列没有白噪声，测量与真值的差主要是真实 Bias。</p></header>
<section class="toolbar">
<label><input data-kind="truth" type="checkbox" checked><span class="legend"><i class="swatch" style="background:#202a34"></i>无 Bias 真值</span></label>
<label><input data-kind="measured" type="checkbox" checked><span class="legend"><i class="swatch" style="background:#d24b40"></i>原始测量</span></label>
<label><input data-kind="corrected" type="checkbox" checked><span class="legend"><i class="swatch" style="background:#167c68"></i>估计 Bias 校正后</span></label>
<span class="spacer"></span><button id="full">完整时段</button><button id="post">初始化后</button>
</section>
<section class="plot"><canvas data-axis="x"></canvas></section>
<section class="plot"><canvas data-axis="y"></canvas></section>
<section class="plot"><canvas data-axis="z"></canvas></section>
<section id="cursor"></section>
<footer class="foot">鼠标滚轮缩放时间轴，拖动平移，移动鼠标查看最近采样；三个面板共享相同时间窗口和采样指针。</footer>
</main><script>
const DATA=__PAYLOAD__;
const AXES=['x','y','z'];
const STYLE={truth:{name:'无 Bias 真值',color:'#202a34',dash:[],width:2},measured:{name:'原始测量',color:'#d24b40',dash:[],width:1},corrected:{name:'估计 Bias 校正后',color:'#167c68',dash:[7,4],width:2}};
const M={l:82,r:22,t:24,b:38};
let domain=[DATA.time[0],DATA.time.at(-1)],selected=0,drag=null;
const canvases=[...document.querySelectorAll('canvas[data-axis]')];
function enabled(){return Object.keys(STYLE).filter(k=>document.querySelector(`[data-kind="${k}"]`).checked)}
function nearest(t){let lo=0,hi=DATA.time.length-1;while(lo<hi){const m=(lo+hi)>>1;if(DATA.time[m]<t)lo=m+1;else hi=m}return lo}
function sx(x,w){return M.l+(x-domain[0])/(domain[1]-domain[0])*(w-M.l-M.r)}
function range(axis){let lo=Infinity,hi=-Infinity;for(const kind of enabled()){const y=DATA.values[kind][axis];for(let i=0;i<y.length;i++){if(DATA.time[i]<domain[0]||DATA.time[i]>domain[1])continue;lo=Math.min(lo,y[i]);hi=Math.max(hi,y[i])}}if(!Number.isFinite(lo)){lo=-1;hi=1}if(hi-lo<1e-12){const q=Math.max(Math.abs(lo)*.05,1e-5);lo-=q;hi+=q}const pad=(hi-lo)*.1;return[lo-pad,hi+pad]}
function sy(y,h,r){return h-M.b-(y-r[0])/(r[1]-r[0])*(h-M.t-M.b)}
function fmt(v){const a=Math.abs(v);return a!==0&&(a<1e-4||a>1e4)?v.toExponential(5):v.toFixed(7)}
function drawOne(canvas){const q=devicePixelRatio||1,w=canvas.clientWidth,h=canvas.clientHeight;if(canvas.width!==Math.round(w*q)||canvas.height!==Math.round(h*q)){canvas.width=Math.round(w*q);canvas.height=Math.round(h*q)}const c=canvas.getContext('2d');c.setTransform(q,0,0,q,0,0);c.clearRect(0,0,w,h);const axis=canvas.dataset.axis,r=range(axis);c.font='12px Arial';c.strokeStyle='#dce2e8';c.fillStyle='#45515d';for(let i=0;i<=6;i++){const x=domain[0]+i*(domain[1]-domain[0])/6,px=sx(x,w);c.beginPath();c.moveTo(px,M.t);c.lineTo(px,h-M.b);c.stroke();c.fillText(x.toFixed(1),px-12,h-M.b+20)}for(let i=0;i<=4;i++){const y=r[0]+i*(r[1]-r[0])/4,py=sy(y,h,r);c.beginPath();c.moveTo(M.l,py);c.lineTo(w-M.r,py);c.stroke();c.fillText(fmt(y),5,py+4)}const initX=sx(DATA.initTime,w);if(initX>=M.l&&initX<=w-M.r){c.strokeStyle='#7c8792';c.setLineDash([4,4]);c.beginPath();c.moveTo(initX,M.t);c.lineTo(initX,h-M.b);c.stroke();c.setLineDash([]);c.fillText('静止初始化结束',Math.min(initX+5,w-110),M.t+12)}for(const kind of enabled()){const st=STYLE[kind],y=DATA.values[kind][axis];c.strokeStyle=st.color;c.lineWidth=st.width;c.setLineDash(st.dash);c.beginPath();let started=false;for(let i=0;i<y.length;i++){const x=DATA.time[i];if(x<domain[0]||x>domain[1])continue;const px=sx(x,w),py=sy(y[i],h,r);if(!started){c.moveTo(px,py);started=true}else c.lineTo(px,py)}c.stroke()}c.setLineDash([]);c.strokeStyle='#1c2530';c.strokeRect(M.l,M.t,w-M.l-M.r,h-M.t-M.b);const cx=sx(DATA.time[selected],w);if(cx>=M.l&&cx<=w-M.r){c.strokeStyle='#2b6ea6';c.beginPath();c.moveTo(cx,M.t);c.lineTo(cx,h-M.b);c.stroke()}c.font='bold 13px Arial';c.fillStyle='#202a34';c.fillText(`a_${axis} / m/s²`,M.l+7,M.t+15);c.fillText('time / s',w/2-25,h-8)}
function draw(){canvases.forEach(drawOne);readout()}
function readout(){const i=selected;document.getElementById('cursor').innerHTML=`<table class="readout"><thead><tr><th>sample ${i} · ${DATA.time[i].toFixed(6)} s</th><th>a_x</th><th>a_y</th><th>a_z</th><th>向量范数</th></tr></thead><tbody>${Object.keys(STYLE).map(k=>{const v=AXES.map(a=>DATA.values[k][a][i]),n=Math.hypot(...v);return `<tr><td>${STYLE[k].name}</td><td>${fmt(v[0])}</td><td>${fmt(v[1])}</td><td>${fmt(v[2])}</td><td>${fmt(n)}</td></tr>`}).join('')}<tr><td>优化器使用的 Bias</td>${AXES.map(a=>`<td>${fmt(DATA.bias[a][i])}</td>`).join('')}<td>${fmt(Math.hypot(...AXES.map(a=>DATA.bias[a][i])))}</td></tr></tbody></table>`}
function setFromPointer(e,canvas){const r=canvas.getBoundingClientRect(),u=Math.max(0,Math.min(1,(e.clientX-r.left-M.l)/(r.width-M.l-M.r))),t=domain[0]+u*(domain[1]-domain[0]);selected=nearest(t);draw()}
canvases.forEach(canvas=>{canvas.onpointermove=e=>{if(drag){const span=drag.domain[1]-drag.domain[0],dx=(e.clientX-drag.x)/(canvas.clientWidth-M.l-M.r)*span;domain=[drag.domain[0]-dx,drag.domain[1]-dx]}else setFromPointer(e,canvas);draw()};canvas.onpointerdown=e=>{drag={x:e.clientX,domain:[...domain]};canvas.setPointerCapture(e.pointerId)};canvas.onpointerup=e=>{drag=null;try{canvas.releasePointerCapture(e.pointerId)}catch(_){}};canvas.onpointerleave=()=>{drag=null};canvas.onwheel=e=>{e.preventDefault();const r=canvas.getBoundingClientRect(),u=Math.max(0,Math.min(1,(e.clientX-r.left-M.l)/(r.width-M.l-M.r))),cx=domain[0]+u*(domain[1]-domain[0]),f=e.deltaY<0?.82:1.22;domain=[cx+(domain[0]-cx)*f,cx+(domain[1]-cx)*f];draw()}});
document.querySelectorAll('[data-kind]').forEach(x=>x.onchange=draw);document.getElementById('full').onclick=()=>{domain=[DATA.time[0],DATA.time.at(-1)];draw()};document.getElementById('post').onclick=()=>{domain=[DATA.initTime,DATA.time.at(-1)];draw()};window.onresize=draw;draw();
</script></body></html>'''
    OUTDIR.mkdir(parents=True, exist_ok=True)
    path = OUTDIR / "interactive_accelerometer_axes.html"
    path.write_text(
        template.replace(
            "__PAYLOAD__",
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        ),
        encoding="utf-8",
    )
    return path


def main() -> None:
    if not INPUT.exists():
        raise FileNotFoundError(INPUT)
    print(write_html(load_payload()))


if __name__ == "__main__":
    main()
