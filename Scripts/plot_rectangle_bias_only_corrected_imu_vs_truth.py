#!/usr/bin/env python3
"""Compare Bias-corrected three-axis IMU samples with saved clean truth."""

from __future__ import annotations

import csv
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
OUTDIR = WORKDIR / "analysis_rectangle_bias_only_imu_corrected_vs_truth_20260714"
AXES = ("x", "y", "z")
FLU_TO_NED_SIGN = np.asarray([1.0, -1.0, -1.0], dtype=np.float64)


def columns(data: np.ndarray, prefix: str) -> np.ndarray:
    flu = np.stack([data[f"{prefix}_{axis}"] for axis in AXES], axis=1).astype(
        np.float64
    )
    return flu * FLU_TO_NED_SIGN.reshape(1, 3)


def hold_previous(
    source_time: np.ndarray, source_value: np.ndarray, target_time: np.ndarray
) -> np.ndarray:
    index = np.searchsorted(source_time, target_time, side="right") - 1
    index = np.clip(index, 0, source_time.size - 1)
    return source_value[index]


def vector_rmse(value: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum(value * value, axis=1))))


def axis_rmse(value: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean(value * value, axis=0))


def compact(value: np.ndarray, digits: int = 10) -> list:
    if value.ndim == 1:
        return [round(float(item), digits) for item in value]
    return [[round(float(item), digits) for item in row] for row in value]


def summarize(
    init_index: int,
    imu_time: np.ndarray,
    frame_time: np.ndarray,
    signals: dict[str, dict[str, np.ndarray]],
) -> list[dict[str, object]]:
    init_time = int(frame_time[init_index])
    post = imu_time >= init_time
    output: list[dict[str, object]] = []
    for sensor in ("gyro", "acc"):
        group = signals[sensor]
        raw_error = group["measured"][post] - group["truth"][post]
        corrected_error = group["corrected"][post] - group["truth"][post]
        raw_rms = vector_rmse(raw_error)
        corrected_rms = vector_rmse(corrected_error)
        raw_axis = axis_rmse(raw_error)
        corrected_axis = axis_rmse(corrected_error)
        row: dict[str, object] = {
            "sensor": sensor,
            "all_imu_samples": int(imu_time.size),
            "post_init_imu_samples": int(post.sum()),
            "init_frame": int(init_index),
            "init_time_s": float(init_time * 1e-9),
            "raw_error_vector_rmse": raw_rms,
            "corrected_error_vector_rmse": corrected_rms,
            "corrected_over_raw_rmse_ratio": corrected_rms / raw_rms if raw_rms else 0.0,
            "raw_final_error_norm": float(np.linalg.norm(raw_error[-1])),
            "corrected_final_error_norm": float(np.linalg.norm(corrected_error[-1])),
        }
        for axis, name in enumerate(AXES):
            row[f"raw_rmse_{name}"] = float(raw_axis[axis])
            row[f"corrected_rmse_{name}"] = float(corrected_axis[axis])
        output.append(row)
    return output


def write_csv(
    imu_time: np.ndarray,
    init_time: int,
    signals: dict[str, dict[str, np.ndarray]],
) -> None:
    fields = ["sample", "timestamp_ns", "time_s", "phase"]
    for sensor in ("gyro", "acc"):
        for kind in ("truth", "measured", "estimated_bias", "corrected"):
            fields.extend(f"{sensor}_{kind}_{axis}" for axis in AXES)
        fields.extend(
            f"{sensor}_{kind}_error_{axis}" for kind in ("raw", "corrected") for axis in AXES
        )
        fields.extend((f"{sensor}_raw_error_norm", f"{sensor}_corrected_error_norm"))

    with (OUTDIR / "imu_samples_corrected_vs_truth.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for index, timestamp in enumerate(imu_time):
            row: dict[str, object] = {
                "sample": index,
                "timestamp_ns": int(timestamp),
                "time_s": float(timestamp * 1e-9),
                "phase": "startup_uninitialized" if timestamp < init_time else "post_static_init",
            }
            for sensor in ("gyro", "acc"):
                group = signals[sensor]
                raw_error = group["measured"][index] - group["truth"][index]
                corrected_error = group["corrected"][index] - group["truth"][index]
                for axis, name in enumerate(AXES):
                    for kind in ("truth", "measured", "estimated_bias", "corrected"):
                        row[f"{sensor}_{kind}_{name}"] = float(group[kind][index, axis])
                    row[f"{sensor}_raw_error_{name}"] = float(raw_error[axis])
                    row[f"{sensor}_corrected_error_{name}"] = float(corrected_error[axis])
                row[f"{sensor}_raw_error_norm"] = float(np.linalg.norm(raw_error))
                row[f"{sensor}_corrected_error_norm"] = float(
                    np.linalg.norm(corrected_error)
                )
            writer.writerow(row)


def write_summary(rows: list[dict[str, object]]) -> None:
    with (OUTDIR / "imu_correction_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_payload(
    imu_time: np.ndarray,
    init_time: int,
    signals: dict[str, dict[str, np.ndarray]],
    summaries: list[dict[str, object]],
) -> dict[str, object]:
    payload: dict[str, object] = {
        "sample": list(range(imu_time.size)),
        "time": compact(imu_time.astype(np.float64) * 1e-9, 9),
        "initTime": float(init_time * 1e-9),
        "summaries": summaries,
        "sensors": {},
    }
    for sensor in ("gyro", "acc"):
        group = signals[sensor]
        raw_error = group["measured"] - group["truth"]
        corrected_error = group["corrected"] - group["truth"]
        payload["sensors"][sensor] = {
            "truth": compact(group["truth"]),
            "measured": compact(group["measured"]),
            "bias": compact(group["estimated_bias"]),
            "corrected": compact(group["corrected"]),
            "rawError": compact(raw_error),
            "correctedError": compact(corrected_error),
            "rawErrorNorm": compact(np.linalg.norm(raw_error, axis=1)),
            "correctedErrorNorm": compact(np.linalg.norm(corrected_error, axis=1)),
        }
    return payload


def write_html(payload: dict[str, object], summaries: list[dict[str, object]]) -> None:
    by_sensor = {str(row["sensor"]): row for row in summaries}
    cards = []
    for sensor, label, unit in (
        ("gyro", "陀螺仪", "rad/s"),
        ("acc", "加速度计", "m/s^2"),
    ):
        row = by_sensor[sensor]
        cards.append(
            "<article class='card'>"
            f"<h2>{label}</h2>"
            f"<span>校正前 RMSE <strong>{float(row['raw_error_vector_rmse']):.6g} {unit}</strong></span>"
            f"<span>校正后 RMSE <strong>{float(row['corrected_error_vector_rmse']):.6g} {unit}</strong></span>"
            f"<span>校正后 / 校正前 <strong>{float(row['corrected_over_raw_rmse_ratio']):.6f}</strong></span>"
            "</article>"
        )

    template = r"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Bias-only 三轴 IMU 校正值与真值</title><style>
:root{--ink:#26323f;--muted:#667483;--line:#d7dee6;--panel:#fff;--bg:#eef1f4;--blue:#1769aa}*{box-sizing:border-box}body{margin:0;padding:16px;font-family:Arial,"Microsoft YaHei",sans-serif;background:var(--bg);color:var(--ink)}
.shell{max-width:1480px;margin:auto;background:#fff;border:1px solid var(--line)}header{padding:16px 18px 12px;border-bottom:1px solid var(--line)}h1{font-size:22px;margin:0 0 8px}p{margin:5px 0;line-height:1.55;color:var(--muted)}
.cards{display:grid;grid-template-columns:repeat(2,minmax(280px,1fr));gap:8px;margin-top:12px}.card{border:1px solid var(--line);padding:10px;display:grid;gap:5px}.card h2{font-size:14px;margin:0}.card span{font-size:12px;color:var(--muted)}.card strong{color:var(--ink)}
.toolbar{display:flex;flex-wrap:wrap;gap:7px;padding:10px 14px;border-bottom:1px solid var(--line);align-items:center}button,select{min-height:34px;border:1px solid #aeb9c5;background:#f8fafc;color:var(--ink);padding:6px 10px;font:inherit;cursor:pointer}button.active{background:var(--blue);border-color:var(--blue);color:#fff}.spacer{flex:1}
#legend{display:flex;flex-wrap:wrap;gap:10px 14px;padding:9px 14px;border-bottom:1px solid var(--line);font-size:12px}#legend label{display:flex;gap:5px;align-items:center;cursor:pointer}#plotWrap{height:620px;position:relative}canvas{width:100%;height:100%;display:block}#tooltip{position:absolute;display:none;pointer-events:none;background:#fff;border:1px solid #8593a0;padding:7px 9px;font-size:12px;line-height:1.45;box-shadow:0 2px 8px #0002;white-space:nowrap}
.player{display:grid;grid-template-columns:auto auto minmax(220px,1fr) auto;gap:8px;align-items:center;padding:10px 14px;border-top:1px solid var(--line)}input[type=range]{width:100%}#sampleLabel{font-size:12px;min-width:190px;text-align:right;font-variant-numeric:tabular-nums}.readout{overflow:auto;border-top:1px solid var(--line)}table{border-collapse:collapse;width:100%;font-size:12px}th,td{padding:7px 10px;border-right:1px solid #edf0f3;text-align:right;font-variant-numeric:tabular-nums}th:first-child,td:first-child{text-align:left}thead{background:#f7f9fb}.foot{padding:10px 14px;color:var(--muted);font-size:12px;border-top:1px solid var(--line)}
@media(max-width:760px){body{padding:6px}.cards{grid-template-columns:1fr}#plotWrap{height:520px}.player{grid-template-columns:auto auto 1fr}.player #sampleLabel{grid-column:1/-1;text-align:left}}
</style></head><body><main class="shell"><header><h1>Bias-only：三轴 IMU 校正值与真值</h1>
<p><code>corrected = measured - estimated Bias</code>。数据没有白噪声，因此校正误差完全反映真实 Bias 与估计 Bias 的差异。</p>
<p>全部量已从传感器 FLU 转换到优化器内部 NED。浅灰区域为 3 秒静止初始化前；校正 Bias 按相机帧状态零阶保持到每个 100 Hz IMU 采样点。</p><section class="cards">__CARDS__</section></header>
<section class="toolbar"><button class="active" data-mode="gyroSignal">陀螺仪三轴</button><button data-mode="gyroError">陀螺仪误差</button><button data-mode="accSignal">加速度计三轴</button><button data-mode="accError">加速度计误差</button><span class="spacer"></span><button id="full">全时段</button><button id="postInit">初始化后</button><button id="reset">重置视图</button></section>
<section id="legend"></section><section id="plotWrap"><canvas id="plot"></canvas><div id="tooltip"></div></section>
<section class="player"><button id="play">播放</button><select id="speed"><option value="1">1x</option><option value="5">5x</option><option value="20">20x</option></select><input id="sample" type="range" min="0" step="1"><span id="sampleLabel"></span></section>
<section class="readout"><table><thead><tr><th>量</th><th>x</th><th>y</th><th>z</th><th>误差范数</th></tr></thead><tbody id="readout"></tbody></table></section>
<footer class="foot">鼠标滚轮缩放、拖动平移、点击图例切换曲线。双击曲线可把逐采样指针移动到对应时刻。</footer></main>
<script>
const DATA=__PAYLOAD__,AXIS=['x','y','z'],COLORS=['#c44536','#16866f','#276fbf'];const canvas=document.getElementById('plot'),wrap=document.getElementById('plotWrap'),ctx=canvas.getContext('2d'),legend=document.getElementById('legend'),tip=document.getElementById('tooltip'),slider=document.getElementById('sample'),sampleLabel=document.getElementById('sampleLabel'),readout=document.getElementById('readout');const M={l:88,r:24,t:28,b:58};let state={mode:'gyroSignal',xDomain:null,visible:{},selected:0,drag:null,playing:false,timer:null};slider.max=DATA.sample.length-1;
function sensor(){return state.mode.startsWith('gyro')?'gyro':'acc'}function unit(){return sensor()==='gyro'?'rad/s':'m/s^2'}
function traces(){const d=DATA.sensors[sensor()],out=[];if(state.mode.endsWith('Error')){AXIS.forEach((a,i)=>{out.push({name:`校正前误差 ${a}`,color:COLORS[i],dash:true,y:d.rawError.map(v=>v[i])});out.push({name:`校正后误差 ${a}`,color:COLORS[i],dash:false,y:d.correctedError.map(v=>v[i])})});out.push({name:'校正后误差范数',color:'#202832',dash:false,y:d.correctedErrorNorm})}else{AXIS.forEach((a,i)=>{out.push({name:`真值 ${a}`,color:COLORS[i],dash:false,y:d.truth.map(v=>v[i])});out.push({name:`校正值 ${a}`,color:COLORS[i],dash:true,y:d.corrected.map(v=>v[i])});out.push({name:`原始测量 ${a}`,color:COLORS[i],dash:true,thin:true,y:d.measured.map(v=>v[i])})})}return out.map(t=>({...t,x:DATA.time}))}
function resetDomain(post=false){state.xDomain=post?[DATA.initTime,DATA.time.at(-1)]:[DATA.time[0],DATA.time.at(-1)]}function visible(){return traces().filter(t=>state.visible[t.name]!==false)}
function domains(){if(!state.xDomain)resetDomain();let ys=[];for(const t of visible())for(let i=0;i<t.y.length;i++)if(t.x[i]>=state.xDomain[0]&&t.x[i]<=state.xDomain[1]&&Number.isFinite(t.y[i]))ys.push(t.y[i]);let lo=Math.min(...ys),hi=Math.max(...ys);if(!Number.isFinite(lo)){lo=-1;hi=1}if(Math.abs(hi-lo)<1e-15){const p=Math.max(Math.abs(lo)*.1,1e-9);lo-=p;hi+=p}const pad=(hi-lo)*.09;return{x:state.xDomain,y:[lo-pad,hi+pad]}}
function mx(x,d,w){return M.l+(x-d.x[0])/(d.x[1]-d.x[0])*(w-M.l-M.r)}function my(y,d,h){return h-M.b-(y-d.y[0])/(d.y[1]-d.y[0])*(h-M.t-M.b)}function fmt(v){const a=Math.abs(v);return(a!==0&&(a<1e-3||a>=1e4))?v.toExponential(5):v.toFixed(7)}
function draw(){const w=wrap.clientWidth,h=wrap.clientHeight,d=domains();ctx.clearRect(0,0,w,h);ctx.font='12px Arial';const ix=mx(DATA.initTime,d,w);if(DATA.initTime>d.x[0]){ctx.fillStyle='#edf0f3';ctx.fillRect(M.l,M.t,Math.max(0,Math.min(ix,w-M.r)-M.l),h-M.t-M.b)}ctx.strokeStyle='#d9e0e7';ctx.fillStyle='#34404c';for(let i=0;i<=6;i++){const x=d.x[0]+i*(d.x[1]-d.x[0])/6,px=mx(x,d,w);ctx.beginPath();ctx.moveTo(px,M.t);ctx.lineTo(px,h-M.b);ctx.stroke();ctx.fillText(x.toFixed(1),px-13,h-M.b+21)}for(let i=0;i<=5;i++){const y=d.y[0]+i*(d.y[1]-d.y[0])/5,py=my(y,d,h);ctx.beginPath();ctx.moveTo(M.l,py);ctx.lineTo(w-M.r,py);ctx.stroke();ctx.fillText(fmt(y),6,py+4)}if(DATA.initTime>=d.x[0]&&DATA.initTime<=d.x[1]){ctx.strokeStyle='#5b6773';ctx.setLineDash([4,4]);ctx.beginPath();ctx.moveTo(ix,M.t);ctx.lineTo(ix,h-M.b);ctx.stroke();ctx.setLineDash([]);ctx.fillText('静止初始化完成',Math.min(ix+5,w-112),M.t+14)}ctx.strokeStyle='#202832';ctx.strokeRect(M.l,M.t,w-M.l-M.r,h-M.t-M.b);for(const t of visible()){ctx.strokeStyle=t.color;ctx.globalAlpha=t.thin?.38:1;ctx.lineWidth=t.thin?1:2;ctx.setLineDash(t.dash?[7,5]:[]);ctx.beginPath();let started=false;for(let i=0;i<t.x.length;i++){const x=t.x[i],y=t.y[i];if(x<d.x[0]||x>d.x[1]||!Number.isFinite(y))continue;const px=mx(x,d,w),py=my(y,d,h);if(!started){ctx.moveTo(px,py);started=true}else ctx.lineTo(px,py)}ctx.stroke()}ctx.globalAlpha=1;ctx.setLineDash([]);const cx=mx(DATA.time[state.selected],d,w);if(cx>=M.l&&cx<=w-M.r){ctx.strokeStyle='#111827';ctx.beginPath();ctx.moveTo(cx,M.t);ctx.lineTo(cx,h-M.b);ctx.stroke()}ctx.font='bold 13px Arial';ctx.fillStyle='#26323f';ctx.fillText('time / s',w/2-27,h-14);ctx.save();ctx.translate(18,h/2);ctx.rotate(-Math.PI/2);ctx.fillText(unit(),-25,0);ctx.restore()}
function buildLegend(){legend.innerHTML='';for(const t of traces()){if(!(t.name in state.visible))state.visible[t.name]=true;const l=document.createElement('label'),c=document.createElement('input'),s=document.createElement('span');c.type='checkbox';c.checked=state.visible[t.name];c.onchange=()=>{state.visible[t.name]=c.checked;draw()};s.textContent=t.name;s.style.color=t.color;l.append(c,s);legend.append(l)}}
function update(){const i=state.selected,d=DATA.sensors[sensor()],phase=DATA.time[i]<DATA.initTime?'初始化前':'初始化后';slider.value=i;sampleLabel.textContent=`sample ${i} · ${DATA.time[i].toFixed(6)} s · ${phase}`;const raw=d.rawError[i],corr=d.correctedError[i],rows=[['真值',d.truth[i],0],['原始测量',d.measured[i],d.rawErrorNorm[i]],['Bias 校正值',d.corrected[i],d.correctedErrorNorm[i]],['使用的 Bias',d.bias[i],Math.sqrt(d.bias[i].reduce((s,v)=>s+v*v,0))]];readout.innerHTML=rows.map(r=>`<tr><td>${r[0]}</td><td>${fmt(r[1][0])}</td><td>${fmt(r[1][1])}</td><td>${fmt(r[1][2])}</td><td>${fmt(r[2])}</td></tr>`).join('');draw()}function setSelected(i){state.selected=Math.max(0,Math.min(DATA.sample.length-1,Math.round(i)));update()}function resize(){const q=devicePixelRatio||1;canvas.width=Math.round(wrap.clientWidth*q);canvas.height=Math.round(wrap.clientHeight*q);ctx.setTransform(q,0,0,q,0,0);draw()}
document.querySelectorAll('[data-mode]').forEach(b=>b.onclick=()=>{document.querySelectorAll('[data-mode]').forEach(x=>x.classList.remove('active'));b.classList.add('active');state.mode=b.dataset.mode;buildLegend();update()});document.getElementById('full').onclick=()=>{resetDomain();draw()};document.getElementById('postInit').onclick=()=>{resetDomain(true);draw()};document.getElementById('reset').onclick=()=>{resetDomain();setSelected(0)};slider.oninput=()=>setSelected(Number(slider.value));document.getElementById('play').onclick=()=>{state.playing=!state.playing;document.getElementById('play').textContent=state.playing?'暂停':'播放';if(state.playing)state.timer=setInterval(()=>{const step=Number(document.getElementById('speed').value);if(state.selected>=DATA.sample.length-1){state.playing=false;clearInterval(state.timer);document.getElementById('play').textContent='播放'}else setSelected(state.selected+step)},20);else clearInterval(state.timer)};
canvas.onwheel=e=>{e.preventDefault();const r=canvas.getBoundingClientRect(),d=domains(),q=Math.max(0,Math.min(1,(e.clientX-r.left-M.l)/(r.width-M.l-M.r))),cx=d.x[0]+q*(d.x[1]-d.x[0]),f=e.deltaY<0?.82:1.22;state.xDomain=[cx+(d.x[0]-cx)*f,cx+(d.x[1]-cx)*f];draw()};canvas.onpointerdown=e=>{state.drag={x:e.clientX,domain:[...domains().x]};canvas.setPointerCapture(e.pointerId)};canvas.onpointermove=e=>{const r=canvas.getBoundingClientRect(),d=domains();if(state.drag){const span=state.drag.domain[1]-state.drag.domain[0],dx=(e.clientX-state.drag.x)/(wrap.clientWidth-M.l-M.r)*span;state.xDomain=[state.drag.domain[0]-dx,state.drag.domain[1]-dx];draw();return}const q=(e.clientX-r.left-M.l)/(r.width-M.l-M.r);if(q<0||q>1){tip.style.display='none';return}const t=d.x[0]+q*(d.x[1]-d.x[0]);let lo=0,hi=DATA.time.length-1;while(lo<hi){const m=(lo+hi)>>1;if(DATA.time[m]<t)lo=m+1;else hi=m}const v=DATA.sensors[sensor()];tip.innerHTML=`sample ${lo}<br>${DATA.time[lo].toFixed(6)} s<br>校正前误差 ${fmt(v.rawErrorNorm[lo])}<br>校正后误差 ${fmt(v.correctedErrorNorm[lo])}`;tip.style.display='block';tip.style.left=Math.min(e.clientX-r.left+12,r.width-180)+'px';tip.style.top=Math.max(8,e.clientY-r.top-60)+'px'};canvas.onpointerup=e=>{state.drag=null;try{canvas.releasePointerCapture(e.pointerId)}catch(_){}};canvas.onpointerleave=()=>{tip.style.display='none';state.drag=null};canvas.ondblclick=e=>{const r=canvas.getBoundingClientRect(),d=domains(),q=(e.clientX-r.left-M.l)/(r.width-M.l-M.r),t=d.x[0]+q*(d.x[1]-d.x[0]);let i=0,b=Infinity;DATA.time.forEach((x,j)=>{const z=Math.abs(x-t);if(z<b){b=z;i=j}});setSelected(i)};window.onresize=resize;buildLegend();resetDomain();update();resize();
</script></body></html>"""
    page = template.replace("__CARDS__", "".join(cards)).replace(
        "__PAYLOAD__", json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    )
    (OUTDIR / "interactive_imu_corrected_vs_truth.html").write_text(page, encoding="utf-8")


def write_report(summaries: list[dict[str, object]]) -> None:
    by_sensor = {str(row["sensor"]): row for row in summaries}
    gyro, acc = by_sensor["gyro"], by_sensor["acc"]
    lines = [
        "# Bias-only 三轴 IMU 校正值与真值",
        "",
        "- 定义：`corrected = measured - estimated Bias`。",
        "- 数据没有白噪声，因此 `measured - truth` 就是真实 Bias，`corrected - truth` 就是 Bias 估计误差。",
        "- Bias 估计由相机帧状态零阶保持到每个 100 Hz IMU 采样点。",
        "",
        "## 初始化后结果",
        "",
        f"- gyro：校正前 RMSE `{float(gyro['raw_error_vector_rmse']):.6e} rad/s`，校正后 `{float(gyro['corrected_error_vector_rmse']):.6e} rad/s`，比例 `{float(gyro['corrected_over_raw_rmse_ratio']):.6f}`。",
        f"- acc：校正前 RMSE `{float(acc['raw_error_vector_rmse']):.6e} m/s^2`，校正后 `{float(acc['corrected_error_vector_rmse']):.6e} m/s^2`，比例 `{float(acc['corrected_over_raw_rmse_ratio']):.6f}`。",
        "- 比例小于 1 表示 Bias 校正改善了 IMU 数据；大于 1 表示固定初始化 Bias 使全时段误差更大。",
    ]
    (OUTDIR / "analysis_summary_cn.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    tensor = np.load(RESULT / "tensor_map.npz", allow_pickle=False)
    frame_time = tensor["frames//time_ns"].astype(np.int64)
    frame_bias = {
        "gyro": tensor["frames//imu_vio_gyro_bias"].astype(np.float64),
        "acc": tensor["frames//imu_vio_acc_bias"].astype(np.float64),
    }
    activity = np.linalg.norm(frame_bias["gyro"], axis=1) + np.linalg.norm(
        frame_bias["acc"], axis=1
    )
    initialized = np.flatnonzero(activity > 0.0)
    if initialized.size == 0:
        raise RuntimeError("No initialized Bias state found")
    init_index = int(initialized[0])
    init_time = int(frame_time[init_index])

    truth_file = np.genfromtxt(
        DATASET / "imu_truth_decomposition.csv", delimiter=",", names=True
    )
    imu_time = truth_file["timestamp"].astype(np.int64)
    signals: dict[str, dict[str, np.ndarray]] = {}
    for sensor, truth_prefix, measured_prefix in (
        ("gyro", "true_ang_vel", "measured_ang_vel"),
        ("acc", "true_lin_acc", "measured_lin_acc"),
    ):
        estimate = hold_previous(frame_time, frame_bias[sensor], imu_time)
        measured = columns(truth_file, measured_prefix)
        clean = columns(truth_file, truth_prefix)
        signals[sensor] = {
            "truth": clean,
            "measured": measured,
            "estimated_bias": estimate,
            "corrected": measured - estimate,
        }

    summaries = summarize(init_index, imu_time, frame_time, signals)
    write_csv(imu_time, init_time, signals)
    write_summary(summaries)
    write_html(build_payload(imu_time, init_time, signals, summaries), summaries)
    write_report(summaries)
    (OUTDIR / "summary.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(OUTDIR / "interactive_imu_corrected_vs_truth.html")


if __name__ == "__main__":
    main()
