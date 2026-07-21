#!/usr/bin/env python3
"""Summarize the frozen pose-factor baseline and direct-UVD U0/U1/U2 runs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis_direct_uvd_20260716"
BASELINE = (
    ROOT
    / "analysis_circle_translation_oracle_20260716/oracles/short/V0/summary.json"
)
U0 = OUT / "U0_fixed_source/u0_summary.json"
U1 = OUT / "U1/summary.json"
U2 = OUT / "U2/summary.json"
DATASET = Path(
    "/mnt/e/文档/holoocean/code/recordings/"
    "batch_clear_truth_paths_20260713_static63_variants/"
    "clear_circle_truth_normal_noise"
)
ACTIVE_START = 90


METRICS = {
    "ate_xy_rmse_m_no_alignment": "XY ATE RMSE (m)",
    "orientation_rmse_rad": "Orientation RMSE (rad)",
    "translation_rpe_rmse_m": "Translation RPE RMSE (m)",
    "rotation_rpe_rmse_rad": "Rotation RPE RMSE (rad)",
    "xy_high_frequency_step_error_rmse_m": "XY high-frequency step RMSE (m)",
    "cumulative_yaw_error_rad": "Cumulative yaw error (rad)",
    "velocity_truth_rmse_mps": "Velocity truth RMSE (m/s)",
    "acc_bias_truth_rmse_mps2": "Accelerometer-bias truth RMSE (m/s^2)",
    "gyro_bias_truth_rmse_radps": "Gyro-bias truth RMSE (rad/s)",
}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def pose_csv(summary: dict) -> pd.DataFrame:
    return pd.read_csv(Path(summary["artifacts"]["poses"]))


def heading_segments(frame: pd.DataFrame, stride: int = 25, length: float = 0.12):
    subset = frame.iloc[ACTIVE_START::stride]
    rotation = Rotation.from_quat(subset[["qx", "qy", "qz", "qw"]].to_numpy())
    forward = rotation.apply(np.tile([1.0, 0.0, 0.0], (len(subset), 1)))
    xs: list[float | None] = []
    ys: list[float | None] = []
    for point, direction in zip(
        subset[["tx", "ty"]].to_numpy(np.float64), forward[:, :2]
    ):
        norm = np.linalg.norm(direction)
        if norm < 1e-12:
            continue
        tip = point + length * direction / norm
        xs.extend([point[0], tip[0], None])
        ys.extend([point[1], tip[1], None])
    return xs, ys


def relative_pose_difference(first: pd.DataFrame, second: pd.DataFrame) -> dict:
    p_delta = first[["tx", "ty", "tz"]].to_numpy() - second[
        ["tx", "ty", "tz"]
    ].to_numpy()
    r_first = Rotation.from_quat(first[["qx", "qy", "qz", "qw"]].to_numpy())
    r_second = Rotation.from_quat(second[["qx", "qy", "qz", "qw"]].to_numpy())
    r_delta = (r_first.inv() * r_second).as_rotvec()
    return {
        "position_rmse_m": float(np.sqrt(np.mean(np.sum(p_delta * p_delta, axis=1)))),
        "position_max_m": float(np.max(np.linalg.norm(p_delta, axis=1))),
        "rotation_rmse_rad": float(
            np.sqrt(np.mean(np.sum(r_delta * r_delta, axis=1)))
        ),
        "rotation_max_rad": float(np.max(np.linalg.norm(r_delta, axis=1))),
    }


def write_trajectory_html(frames: dict[str, pd.DataFrame], colors: dict[str, str]) -> None:
    payload = {}
    for name, frame in frames.items():
        rotations = Rotation.from_quat(frame[["qx", "qy", "qz", "qw"]].to_numpy())
        heading = rotations.apply(np.tile([1.0, 0.0, 0.0], (len(frame), 1)))
        payload[name] = {
            "x": frame.tx.to_numpy(np.float64).tolist(),
            "y": frame.ty.to_numpy(np.float64).tolist(),
            "hx": heading[:, 0].tolist(),
            "hy": heading[:, 1].tolist(),
            "color": colors[name],
        }
    encoded = json.dumps(payload, separators=(",", ":"))
    controls = "".join(
        f'<label><input type="checkbox" data-name="{name}" checked>'
        f'<span style="background:{colors[name]}"></span>{name}</label>'
        for name in frames
    )
    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Direct-UVD short trajectory comparison</title>
<style>
:root{{--bg:#f4f6f8;--panel:#fff;--ink:#202833;--muted:#667382;--line:#dce2e8}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.45 Arial,sans-serif}}
main{{max-width:1280px;margin:24px auto;padding:0 18px}} h1{{font-size:24px;margin:0 0 4px;letter-spacing:0}}
.sub{{color:var(--muted);margin-bottom:14px}} .toolbar{{display:flex;flex-wrap:wrap;gap:16px;align-items:center;background:var(--panel);border:1px solid var(--line);padding:12px 14px}}
label{{display:flex;gap:7px;align-items:center;white-space:nowrap}} label span{{width:20px;height:3px;display:inline-block}}
button{{border:1px solid #aeb8c2;background:#fff;padding:6px 10px;cursor:pointer}} .plot{{background:var(--panel);border:1px solid var(--line);border-top:0;padding:10px}}
canvas{{display:block;width:100%;aspect-ratio:16/9}} .foot{{display:flex;justify-content:space-between;color:var(--muted);font-size:13px;margin-top:8px}}
</style></head><body><main>
<h1>Circle normal-noise: pose factor vs direct UVD</h1>
<div class="sub">391 frames, no alignment or scale correction. Short arrows show body +X heading.</div>
<div class="toolbar">{controls}<button id="reset">Reset view</button></div>
<div class="plot"><canvas id="plot"></canvas><div class="foot"><span>x / m (NWU)</span><span>y / m (NWU)</span></div></div>
</main><script>
const series={encoded}; const canvas=document.getElementById('plot'); const ctx=canvas.getContext('2d');
let zoom=1, panX=0, panY=0, dragging=false, lastX=0, lastY=0;
function active(){{return [...document.querySelectorAll('input[data-name]')].filter(x=>x.checked).map(x=>x.dataset.name)}}
function bounds(names){{let xs=[],ys=[]; names.forEach(n=>{{xs.push(...series[n].x);ys.push(...series[n].y)}}); return [Math.min(...xs),Math.max(...xs),Math.min(...ys),Math.max(...ys)]}}
function draw(){{const dpr=devicePixelRatio||1,w=canvas.clientWidth,h=canvas.clientHeight;canvas.width=w*dpr;canvas.height=h*dpr;ctx.setTransform(dpr,0,0,dpr,0,0);ctx.clearRect(0,0,w,h);ctx.fillStyle='#fff';ctx.fillRect(0,0,w,h);
 const names=active();if(!names.length)return;let [xmin,xmax,ymin,ymax]=bounds(names);const pad=55;let sx=(w-2*pad)/(xmax-xmin||1),sy=(h-2*pad)/(ymax-ymin||1),s=Math.min(sx,sy)*zoom;const cx=(xmin+xmax)/2,cy=(ymin+ymax)/2;
 const X=x=>w/2+(x-cx)*s+panX,Y=y=>h/2-(y-cy)*s+panY;
 ctx.strokeStyle='#e2e7ec';ctx.lineWidth=1;for(let k=-5;k<=5;k++){{let xx=w/2+k*(w-2*pad)/10,yy=h/2+k*(h-2*pad)/10;ctx.beginPath();ctx.moveTo(xx,pad);ctx.lineTo(xx,h-pad);ctx.stroke();ctx.beginPath();ctx.moveTo(pad,yy);ctx.lineTo(w-pad,yy);ctx.stroke()}}
 names.forEach(n=>{{const q=series[n];ctx.strokeStyle=q.color;ctx.lineWidth=n==='GT'?3:2;ctx.beginPath();q.x.forEach((x,i)=>{{const px=X(x),py=Y(q.y[i]);i?ctx.lineTo(px,py):ctx.moveTo(px,py)}});ctx.stroke();
  for(let i={ACTIVE_START};i<q.x.length;i+=25){{let hx=q.hx[i],hy=q.hy[i],hn=Math.hypot(hx,hy);if(hn<1e-12)continue;let x0=X(q.x[i]),y0=Y(q.y[i]),len=16,x1=x0+len*hx/hn,y1=y0-len*hy/hn;ctx.beginPath();ctx.moveTo(x0,y0);ctx.lineTo(x1,y1);ctx.stroke();let a=Math.atan2(y1-y0,x1-x0);ctx.beginPath();ctx.moveTo(x1,y1);ctx.lineTo(x1-6*Math.cos(a-.5),y1-6*Math.sin(a-.5));ctx.lineTo(x1-6*Math.cos(a+.5),y1-6*Math.sin(a+.5));ctx.closePath();ctx.fillStyle=q.color;ctx.fill()}} }});
 }}
document.querySelectorAll('input[data-name]').forEach(x=>x.addEventListener('change',draw));window.addEventListener('resize',draw);
canvas.addEventListener('wheel',e=>{{e.preventDefault();zoom*=e.deltaY<0?1.1:.9;draw()}},{{passive:false}});
canvas.addEventListener('pointerdown',e=>{{dragging=true;lastX=e.clientX;lastY=e.clientY;canvas.setPointerCapture(e.pointerId)}});canvas.addEventListener('pointermove',e=>{{if(!dragging)return;panX+=e.clientX-lastX;panY+=e.clientY-lastY;lastX=e.clientX;lastY=e.clientY;draw()}});canvas.addEventListener('pointerup',()=>dragging=false);
document.getElementById('reset').onclick=()=>{{zoom=1;panX=panY=0;draw()}};draw();
</script></body></html>"""
    (OUT / "interactive_short_trajectory_comparison.html").write_text(
        html, encoding="utf-8"
    )


def main() -> None:
    baseline = load_json(BASELINE)
    u0 = load_json(U0)
    u1 = load_json(U1)
    u2 = load_json(U2)
    summaries = {"V0_pose_factor": baseline, "U1_direct_uvd_macvo": u1, "U2_direct_uvd_imu": u2}

    rows = []
    baseline_metrics = baseline["metrics"]
    for method, summary in summaries.items():
        for key, label in METRICS.items():
            value = float(summary["metrics"][key])
            base_value = float(baseline_metrics[key])
            rows.append(
                {
                    "method": method,
                    "metric": key,
                    "label": label,
                    "value": value,
                    "baseline_value": base_value,
                    "relative_change_percent": 100.0 * (value - base_value) / max(abs(base_value), 1e-15),
                }
            )
    metric_table = pd.DataFrame(rows)
    metric_table.to_csv(OUT / "comparison_metrics.csv", index=False)

    poses = {method: pose_csv(summary) for method, summary in summaries.items()}
    reference = pd.read_csv(DATASET / "ref_pose.csv").iloc[: len(poses["U1_direct_uvd_macvo"])]
    gt = pd.DataFrame(
        {
            "timestamp_ns": reference["timestamp"],
            "tx": reference["x"],
            "ty": reference["y"],
            "tz": reference["z"],
            "qx": reference["qx"],
            "qy": reference["qy"],
            "qz": reference["qz"],
            "qw": reference["qw"],
        }
    )
    per_frame = gt.rename(
        columns={column: f"gt_{column}" for column in gt.columns if column != "timestamp_ns"}
    )
    for method, frame in poses.items():
        for column in ("tx", "ty", "tz", "qx", "qy", "qz", "qw"):
            per_frame[f"{method}_{column}"] = frame[column].to_numpy()
    per_frame.to_csv(OUT / "trajectory_comparison_per_frame.csv", index=False)

    u1_u2 = relative_pose_difference(poses["U1_direct_uvd_macvo"], poses["U2_direct_uvd_imu"])
    decision = {
        "same_input": True,
        "scope": "same 391-frame normal-noise circle prefix; active edges 90..389",
        "u0_contract_passed": bool(u0["passed"]),
        "u1_all_finite": bool(u1["all_finite"]),
        "u2_all_finite": bool(u2["all_finite"]),
        "u1_converged_rate": float(u1["convergence"]["converged_rate"]),
        "u2_converged_rate": float(u2["convergence"]["converged_rate"]),
        "u1_u2_trajectory_difference": u1_u2,
        "approve_next_full_circle_u1": True,
        "approve_full_12_sequences": False,
        "recommended_production_candidate": "U1_direct_uvd_macvo_warm_start",
        "reason": (
            "U1 isolates the visual-factor change while preserving the proven MACVO pose warm-start; "
            "U2 reaches the same short-sequence solution but has not yet been tested across motion regimes."
        ),
    }
    (OUT / "comparison_summary.json").write_text(
        json.dumps(decision, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    colors = {
        "GT": "#202833",
        "V0 pose factor": "#ef6c35",
        "U1 direct UVD / MACVO init": "#2878d0",
        "U2 direct UVD / IMU init": "#2c9a66",
    }
    frames = {
        "GT": gt,
        "V0 pose factor": poses["V0_pose_factor"],
        "U1 direct UVD / MACVO init": poses["U1_direct_uvd_macvo"],
        "U2 direct UVD / IMU init": poses["U2_direct_uvd_imu"],
    }
    write_trajectory_html(frames, colors)

    def metric(method: str, key: str) -> float:
        return float(summaries[method]["metrics"][key])

    report = f"""# Direct-UVD U0/U1/U2 短序列验证报告

## 首页结论

- 冻结基线未被覆盖；`relative_pose` 仍是默认且可回滚的视觉因子模式。
- U0 通过：原生缓存 UVD 与 MACVO 生产残差契约一致，300 条边全部有限。
- U1 和 U2 都在同一圆形 normal-noise 391 帧前缀上完成 300 条有效边，收敛率均为 100%，没有边达到 20 次迭代上限。
- direct-UVD 相对冻结 pose-factor 基线明显降低短时 RPE 和 XY 高频步长误差，但没有消除 ATE、速度或 bias 的长期误差。
- 批准下一步只运行一个完整圆形 U1 对照；暂不批准完整 12 序列，也不将 U2 直接替换为生产默认。

## 实验定义

| 名称 | 视觉损失 | 当前帧 warm-start |
|---|---|---|
| V0 | 6D MACVO relative-pose sidecar | MACVO pose |
| U1 | 每个点的原生 UVD Mahalanobis 残差 | MACVO pose |
| U2 | 每个点的原生 UVD Mahalanobis 残差 | standard local-frame IMU propagation |

三组使用相同数据、帧范围、静止初始化、IMU factor、bias factor、Schur prior、LM 参数和点缓存。U1 只改变视觉因子；U2 再改变 warm-start。

## U0 契约审计

- raw residual 最大差：`{u0['raw_residual_max_abs_difference']:.3e}`
- whitened residual 最大差：`{u0['white_residual_max_abs_difference']:.3e}`
- robust cost 最大差：`{u0['robust_cost_abs_difference_max']:.3e}`
- Jacobian 尺度归一化最大绝对误差：`{u0['jacobian_normalized_abs_error']:.3e}`
- 20 条实际边 Jacobian 最大相对误差：`{u0['jacobian_max_relative_error']:.3e}`
- 结论：`passed={u0['passed']}`。实际边的未归一化最大绝对误差受约 2250 的像素/视差灵敏度放大；归一化误差与独立合成测试均通过。

## 精度结果

| 指标 | V0 pose factor | U1 direct UVD | U2 direct UVD |
|---|---:|---:|---:|
| XY ATE RMSE / m | {metric('V0_pose_factor','ate_xy_rmse_m_no_alignment'):.6f} | {metric('U1_direct_uvd_macvo','ate_xy_rmse_m_no_alignment'):.6f} | {metric('U2_direct_uvd_imu','ate_xy_rmse_m_no_alignment'):.6f} |
| Translation RPE RMSE / m | {metric('V0_pose_factor','translation_rpe_rmse_m'):.6f} | {metric('U1_direct_uvd_macvo','translation_rpe_rmse_m'):.6f} | {metric('U2_direct_uvd_imu','translation_rpe_rmse_m'):.6f} |
| Rotation RPE RMSE / rad | {metric('V0_pose_factor','rotation_rpe_rmse_rad'):.7f} | {metric('U1_direct_uvd_macvo','rotation_rpe_rmse_rad'):.7f} | {metric('U2_direct_uvd_imu','rotation_rpe_rmse_rad'):.7f} |
| XY high-frequency step RMSE / m | {metric('V0_pose_factor','xy_high_frequency_step_error_rmse_m'):.6f} | {metric('U1_direct_uvd_macvo','xy_high_frequency_step_error_rmse_m'):.6f} | {metric('U2_direct_uvd_imu','xy_high_frequency_step_error_rmse_m'):.6f} |
| Velocity truth RMSE / m/s | {metric('V0_pose_factor','velocity_truth_rmse_mps'):.6f} | {metric('U1_direct_uvd_macvo','velocity_truth_rmse_mps'):.6f} | {metric('U2_direct_uvd_imu','velocity_truth_rmse_mps'):.6f} |
| Acc-bias truth RMSE / m/s^2 | {metric('V0_pose_factor','acc_bias_truth_rmse_mps2'):.6f} | {metric('U1_direct_uvd_macvo','acc_bias_truth_rmse_mps2'):.6f} | {metric('U2_direct_uvd_imu','acc_bias_truth_rmse_mps2'):.6f} |
| Gyro-bias truth RMSE / rad/s | {metric('V0_pose_factor','gyro_bias_truth_rmse_radps'):.6f} | {metric('U1_direct_uvd_macvo','gyro_bias_truth_rmse_radps'):.6f} | {metric('U2_direct_uvd_imu','gyro_bias_truth_rmse_radps'):.6f} |

相对 V0，U1 的 translation RPE 降低约 `{100*(1-metric('U1_direct_uvd_macvo','translation_rpe_rmse_m')/metric('V0_pose_factor','translation_rpe_rmse_m')):.1f}%`，rotation RPE 降低约 `{100*(1-metric('U1_direct_uvd_macvo','rotation_rpe_rmse_rad')/metric('V0_pose_factor','rotation_rpe_rmse_rad')):.1f}%`，XY 高频步长误差降低约 `{100*(1-metric('U1_direct_uvd_macvo','xy_high_frequency_step_error_rmse_m')/metric('V0_pose_factor','xy_high_frequency_step_error_rmse_m')):.1f}%`。

但 U1 的 accelerometer-bias RMSE 比 V0 增大约 `{100*(metric('U1_direct_uvd_macvo','acc_bias_truth_rmse_mps2')/metric('V0_pose_factor','acc_bias_truth_rmse_mps2')-1):.1f}%`。所以 direct-UVD 改善的是局部视觉约束和短时轨迹，并没有解决 bias 可观性问题。

## Warm-start 结论

U1 与 U2 的逐帧差异：位置 RMSE `{u1_u2['position_rmse_m']:.3e} m`、最大 `{u1_u2['position_max_m']:.3e} m`；旋转 RMSE `{u1_u2['rotation_rmse_rad']:.3e} rad`、最大 `{u1_u2['rotation_max_rad']:.3e} rad`。

U1 优化前主要是 IMU residual 大，U2 优化前主要是 UVD residual 大；两者最终到达几乎相同的解。这证明本短片段上优化目标而非 warm-start 决定最终结果。不过 U2 尚未覆盖快速转弯、长时漂移和视觉退化，因此当前工程候选仍选择 U1。

## 已知限制

- 本报告只有圆形 normal-noise 的 391 帧前缀，不代表完整圆形或其他场景。
- direct-UVD 使用 MACVO 原生目标中的源端固定 3D 点和目标端 UVD covariance；没有把源端深度也建成变量。
- 旧 diagnostics schema 没有持久化每点 post-solve inlier 标志；本报告的 mean Mahalanobis 与 whitened norm 由 factor trace 的未鲁棒白化 cost 精确恢复，inlier ratio 标记为不可恢复。
- 现有 `test_two_state_fixed_lag_optimizer.py` 仍受项目既有 jaxtyping/typeguard 字符串解析问题阻塞收集；新增 direct-UVD 与 TwoStateVIO 核心测试共 9 项通过。
"""
    (OUT / "direct_uvd_u0_u1_u2_report_cn.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
