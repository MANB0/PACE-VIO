#!/usr/bin/env python3
"""Evaluate online T2 and the read-only full-history smoother on one frame range."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Utility.PoseFrame import convert_pose_world_frame_only
from Utility.T2HistorySmoother import (
    factor_cost_breakdown,
    load_t2_history_archive,
)
from Utility.TwoStateVIO import NavigationState


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", required=True, type=Path)
    parser.add_argument("--tensor-map", required=True, type=Path)
    parser.add_argument("--gt-imu-csv", required=True, type=Path)
    parser.add_argument("--ref-pose", required=True, type=Path)
    parser.add_argument("--bias-truth", required=True, type=Path)
    parser.add_argument("--baseline-online", type=Path)
    return parser.parse_args()


def rmse(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values))))


def load_estimate(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {
        "frame", "timestamp_ns", "tx_nwu", "ty_nwu", "tz_nwu",
        "qx_nwu", "qy_nwu", "qz_nwu", "qw_nwu",
        "vx_nwu", "vy_nwu", "vz_nwu",
        "ba_x_internal", "ba_y_internal", "ba_z_internal",
        "bg_x_internal", "bg_y_internal", "bg_z_internal",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path} lacks columns {missing}")
    if not np.isfinite(frame.select_dtypes(include=[np.number]).to_numpy()).all():
        raise FloatingPointError(f"{path} contains NaN/Inf")
    return frame


def select_truth(path: Path, timestamps: np.ndarray) -> pd.DataFrame:
    truth = pd.read_csv(path)
    selected = truth.set_index("timestamp_ns").reindex(timestamps)
    if selected.isna().any().any():
        raise ValueError("GT IMU-center CSV does not cover every smoother timestamp")
    return selected.reset_index()


def interpolate_rows(
    path: Path,
    timestamps: np.ndarray,
    time_column: str,
    columns: list[str],
) -> np.ndarray:
    source = pd.read_csv(path)
    source_time = source[time_column].to_numpy(np.int64)
    values = source[columns].to_numpy(np.float64)
    return np.column_stack(
        [np.interp(timestamps, source_time, values[:, axis]) for axis in range(3)]
    )


def relative_errors(
    gt_position: np.ndarray,
    gt_rotation: Rotation,
    est_position: np.ndarray,
    est_rotation: Rotation,
    starts: np.ndarray,
    ends: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    gt_i = gt_rotation[starts]
    est_i = est_rotation[starts]
    gt_delta_r = gt_i.inv() * gt_rotation[ends]
    est_delta_r = est_i.inv() * est_rotation[ends]
    gt_delta_t = gt_i.inv().apply(gt_position[ends] - gt_position[starts])
    est_delta_t = est_i.inv().apply(est_position[ends] - est_position[starts])
    translation = np.linalg.norm(
        gt_delta_r.inv().apply(est_delta_t - gt_delta_t), axis=1
    )
    rotation_deg = np.rad2deg((gt_delta_r.inv() * est_delta_r).magnitude())
    return translation, rotation_deg


def evaluate_method(
    name: str,
    estimate: pd.DataFrame,
    truth: pd.DataFrame,
    velocity_truth: np.ndarray,
    acc_bias_truth: np.ndarray,
    gyro_bias_truth: np.ndarray,
) -> dict[str, float | int | str]:
    p = estimate[["tx_nwu", "ty_nwu", "tz_nwu"]].to_numpy(np.float64)
    p_gt = truth[["tx", "ty", "tz"]].to_numpy(np.float64)
    p = p - p[0]
    p_gt = p_gt - p_gt[0]
    r = Rotation.from_quat(
        estimate[["qx_nwu", "qy_nwu", "qz_nwu", "qw_nwu"]].to_numpy(np.float64)
    )
    r_gt = Rotation.from_quat(
        truth[["qx", "qy", "qz", "qw"]].to_numpy(np.float64)
    )
    position_error = p - p_gt
    orientation_error_deg = np.rad2deg((r_gt.inv() * r).magnitude())
    starts = np.arange(len(estimate) - 1, dtype=np.int64)
    ends = starts + 1
    frame_t, frame_r = relative_errors(p_gt, r_gt, p, r, starts, ends)
    time_ns = estimate["timestamp_ns"].to_numpy(np.int64)
    one_second_end = np.searchsorted(time_ns, time_ns + 1_000_000_000, side="left")
    valid = one_second_end < len(time_ns)
    one_second_t, one_second_r = relative_errors(
        p_gt,
        r_gt,
        p,
        r,
        np.arange(len(time_ns), dtype=np.int64)[valid],
        one_second_end[valid],
    )
    velocity = estimate[["vx_nwu", "vy_nwu", "vz_nwu"]].to_numpy(np.float64)
    acc_bias = estimate[
        ["ba_x_internal", "ba_y_internal", "ba_z_internal"]
    ].to_numpy(np.float64)
    gyro_bias = estimate[
        ["bg_x_internal", "bg_y_internal", "bg_z_internal"]
    ].to_numpy(np.float64)
    return {
        "method": name,
        "frame_count": int(len(estimate)),
        "ate_xyz_rmse_m": rmse(np.linalg.norm(position_error, axis=1)),
        "ate_xy_rmse_m": rmse(np.linalg.norm(position_error[:, :2], axis=1)),
        "ape_rotation_rmse_deg": rmse(orientation_error_deg),
        "rpe_frame_translation_rmse_m": rmse(frame_t),
        "rpe_frame_rotation_rmse_deg": rmse(frame_r),
        "rpe_1s_translation_rmse_m": rmse(one_second_t),
        "rpe_1s_rotation_rmse_deg": rmse(one_second_r),
        "endpoint_translation_error_m": float(np.linalg.norm(position_error[-1])),
        "velocity_rmse_mps": rmse(np.linalg.norm(velocity - velocity_truth, axis=1)),
        "acc_bias_rmse_mps2": rmse(np.linalg.norm(acc_bias - acc_bias_truth, axis=1)),
        "gyro_bias_rmse_radps": rmse(np.linalg.norm(gyro_bias - gyro_bias_truth, axis=1)),
    }


def navigation_states(frame: pd.DataFrame) -> tuple[NavigationState, ...]:
    pose_nwu = frame[
        ["tx_nwu", "ty_nwu", "tz_nwu", "qx_nwu", "qy_nwu", "qz_nwu", "qw_nwu"]
    ].to_numpy(np.float64)
    pose_internal = convert_pose_world_frame_only(pose_nwu, "NWU", "NED")
    velocity_nwu = frame[["vx_nwu", "vy_nwu", "vz_nwu"]].to_numpy(np.float64)
    velocity_internal = velocity_nwu.copy()
    velocity_internal[:, 1:3] *= -1.0
    ba = frame[["ba_x_internal", "ba_y_internal", "ba_z_internal"]].to_numpy(np.float64)
    bg = frame[["bg_x_internal", "bg_y_internal", "bg_z_internal"]].to_numpy(np.float64)
    import torch

    return tuple(
        NavigationState(
            pose_WB=torch.as_tensor(pose_internal[row], dtype=torch.float64).reshape(1, 7),
            velocity_W=torch.as_tensor(velocity_internal[row], dtype=torch.float64),
            acc_bias=torch.as_tensor(ba[row], dtype=torch.float64),
            gyro_bias=torch.as_tensor(bg[row], dtype=torch.float64),
        )
        for row in range(len(frame))
    )


def write_factor_costs(
    path: Path,
    frame_indices: np.ndarray,
    online: dict[str, np.ndarray],
    smoothed: dict[str, np.ndarray],
) -> dict[str, dict[str, float]]:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "frame_i", "frame_j",
                "online_imu_cost", "online_bias_cost", "online_visual_cost",
                "smoothed_imu_cost", "smoothed_bias_cost", "smoothed_visual_cost",
            ]
        )
        for edge in range(len(frame_indices) - 1):
            writer.writerow(
                [
                    int(frame_indices[edge]), int(frame_indices[edge + 1]),
                    online["imu"][edge], online["bias"][edge], online["visual"][edge],
                    smoothed["imu"][edge], smoothed["bias"][edge], smoothed["visual"][edge],
                ]
            )
    return {
        name: {
            "prior": float(values["prior"].sum()),
            "imu": float(values["imu"].sum()),
            "bias": float(values["bias"].sum()),
            "visual": float(values["visual"].sum()),
            "total": float(sum(component.sum() for component in values.values())),
        }
        for name, values in (("online", online), ("smoothed", smoothed))
    }


def write_plot(
    output: Path,
    truth: pd.DataFrame,
    online: pd.DataFrame,
    smoothed: pd.DataFrame,
    metrics: list[dict[str, float | int | str]],
) -> None:
    gt = truth[["tx", "ty", "tz"]].to_numpy(np.float64)
    curves = {
        "GT": (gt - gt[0], "#111827"),
        "Online T2": (
            online[["tx_nwu", "ty_nwu", "tz_nwu"]].to_numpy(np.float64)
            - online[["tx_nwu", "ty_nwu", "tz_nwu"]].to_numpy(np.float64)[0],
            "#dc2626",
        ),
        "Full-history 15D": (
            smoothed[["tx_nwu", "ty_nwu", "tz_nwu"]].to_numpy(np.float64)
            - smoothed[["tx_nwu", "ty_nwu", "tz_nwu"]].to_numpy(np.float64)[0],
            "#2563eb",
        ),
    }
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    for axis, indices, labels in zip(
        axes, ((0, 1), (0, 2), (1, 2)), (("x", "y"), ("x", "z"), ("y", "z"))
    ):
        for label, (values, color) in curves.items():
            axis.plot(values[:, indices[0]], values[:, indices[1]], label=label, color=color)
        axis.set_xlabel(f"{labels[0]} / m (NWU)")
        axis.set_ylabel(f"{labels[1]} / m (NWU)")
        axis.axis("equal")
        axis.grid(alpha=0.25)
    axes[0].legend()
    fig.suptitle("T2 online vs read-only full-history smoothing, frames 90-299")
    fig.savefig(output / "t2_online_vs_history_smoother.png", dpi=180)
    plt.close(fig)

    payload = {
        "metrics": metrics,
        "traces": [
            {"name": name, "color": color, "xyz": values.tolist()}
            for name, (values, color) in curves.items()
        ],
    }
    html = """<!doctype html><html><head><meta charset='utf-8'><title>T2 history smoother</title>
<style>body{margin:0;background:#f4f7fb;color:#172033;font:15px Arial,sans-serif}header{padding:22px 28px;background:white;border-bottom:1px solid #dbe3ef}h1{font-size:24px;margin:0 0 6px}main{padding:18px 28px}.toolbar{display:flex;gap:8px;align-items:center;margin-bottom:12px}button{border:1px solid #b9c7da;background:white;padding:8px 13px;border-radius:6px;cursor:pointer}button.active{background:#2563eb;color:white}.layout{display:grid;grid-template-columns:minmax(0,1fr) 320px;gap:16px}.panel{background:white;border:1px solid #dbe3ef;border-radius:8px;padding:14px}canvas{width:100%;height:620px;display:block;cursor:grab}.metric{padding:10px 0;border-bottom:1px solid #e7edf5}.metric b{display:block;margin-bottom:5px}.swatch{display:inline-block;width:18px;height:3px;margin-right:8px;vertical-align:middle}@media(max-width:900px){.layout{grid-template-columns:1fr}canvas{height:480px}}</style></head>
<body><header><h1>T2 Online vs Full-history 15D Smoother</h1><div>Frames 90-299, IMU center, NWU, translation rebased at frame 90. No SE(3) fitting or scale correction.</div></header><main><div class='toolbar'><button data-view='xy' class='active'>XY</button><button data-view='xz'>XZ</button><button data-view='yz'>YZ</button><button id='reset'>Reset</button></div><div class='layout'><div class='panel'><canvas id='plot'></canvas></div><aside class='panel' id='metrics'></aside></div></main>
<script>const D=__DATA__;const c=document.getElementById('plot'),x=c.getContext('2d');let view='xy',manual=false,scale=1,ox=0,oy=0,drag=null;const idx={xy:[0,1],xz:[0,2],yz:[1,2]};
function resize(){const r=c.getBoundingClientRect(),d=devicePixelRatio||1;c.width=r.width*d;c.height=r.height*d;x.setTransform(d,0,0,d,0,0);draw()}
function bounds(){const k=idx[view],v=D.traces.flatMap(t=>t.xyz.map(p=>[p[k[0]],p[k[1]]]));let xs=v.map(p=>p[0]),ys=v.map(p=>p[1]);return [Math.min(...xs),Math.max(...xs),Math.min(...ys),Math.max(...ys)]}
function draw(){const w=c.clientWidth,h=c.clientHeight;x.clearRect(0,0,w,h);const b=bounds(),pad=45,bw=Math.max(b[1]-b[0],1e-6),bh=Math.max(b[3]-b[2],1e-6),s=Math.min((w-2*pad)/bw,(h-2*pad)/bh)*scale,cx=(b[0]+b[1])/2,cy=(b[2]+b[3])/2;const P=p=>[w/2+(p[0]-cx)*s+ox,h/2-(p[1]-cy)*s+oy];x.strokeStyle='#d9e2ef';x.lineWidth=1;for(let i=0;i<=10;i++){x.beginPath();x.moveTo(i*w/10,0);x.lineTo(i*w/10,h);x.stroke();x.beginPath();x.moveTo(0,i*h/10);x.lineTo(w,i*h/10);x.stroke()}const k=idx[view];for(const t of D.traces){x.beginPath();t.xyz.forEach((p,i)=>{const q=P([p[k[0]],p[k[1]]]);i?x.lineTo(...q):x.moveTo(...q)});x.strokeStyle=t.color;x.lineWidth=t.name==='GT'?3:2;x.stroke()}x.fillStyle='#41516a';x.fillText(view[0].toUpperCase()+' / m',w-42,h-12);x.fillText(view[1].toUpperCase()+' / m',10,18)}
c.onwheel=e=>{e.preventDefault();manual=true;scale*=e.deltaY<0?1.12:.89;draw()};c.onmousedown=e=>{drag=[e.clientX,e.clientY,ox,oy];c.style.cursor='grabbing'};window.onmouseup=()=>{drag=null;c.style.cursor='grab'};window.onmousemove=e=>{if(!drag)return;manual=true;ox=drag[2]+e.clientX-drag[0];oy=drag[3]+e.clientY-drag[1];draw()};document.querySelectorAll('[data-view]').forEach(b=>b.onclick=()=>{view=b.dataset.view;document.querySelectorAll('[data-view]').forEach(v=>v.classList.toggle('active',v===b));scale=1;ox=oy=0;draw()});document.getElementById('reset').onclick=()=>{scale=1;ox=oy=0;manual=false;draw()};
document.getElementById('metrics').innerHTML=D.traces.map(t=>`<div><span class='swatch' style='background:${t.color}'></span>${t.name}</div>`).join('')+D.metrics.map(m=>`<div class='metric'><b>${m.method}</b>ATE 3D: ${m.ate_xyz_rmse_m.toFixed(4)} m<br>ATE XY: ${m.ate_xy_rmse_m.toFixed(4)} m<br>Frame RPE-t: ${m.rpe_frame_translation_rmse_m.toFixed(4)} m<br>Velocity RMSE: ${m.velocity_rmse_mps.toFixed(4)} m/s</div>`).join('');window.onresize=resize;resize();</script></body></html>"""
    (output / "interactive_t2_online_vs_history_smoother.html").write_text(
        html.replace("__DATA__", json.dumps(payload, ensure_ascii=False)),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    output = args.result_dir.expanduser().resolve()
    online = load_estimate(output / "t2_online_states.csv")
    smoothed = load_estimate(output / "t2_smoothed_states.csv")
    if not np.array_equal(online["timestamp_ns"], smoothed["timestamp_ns"]):
        raise AssertionError("online and smoothed timestamps differ")
    timestamps = online["timestamp_ns"].to_numpy(np.int64)
    truth = select_truth(args.gt_imu_csv, timestamps)
    velocity_truth = interpolate_rows(
        args.ref_pose, timestamps, "timestamp", ["vx", "vy", "vz"]
    )
    acc_bias_truth = interpolate_rows(
        args.bias_truth,
        timestamps,
        "timestamp",
        ["acc_bias_x", "acc_bias_y", "acc_bias_z"],
    )
    gyro_bias_truth = interpolate_rows(
        args.bias_truth,
        timestamps,
        "timestamp",
        ["gyro_bias_x", "gyro_bias_y", "gyro_bias_z"],
    )
    metrics = [
        evaluate_method(
            "online_t2", online, truth, velocity_truth, acc_bias_truth, gyro_bias_truth
        ),
        evaluate_method(
            "full_history_15d",
            smoothed,
            truth,
            velocity_truth,
            acc_bias_truth,
            gyro_bias_truth,
        ),
    ]
    pd.DataFrame(metrics).to_csv(
        output / "t2_history_smoother_accuracy_metrics.csv", index=False
    )

    archive = load_t2_history_archive(
        args.tensor_map,
        start_frame=int(online.frame.iloc[0]),
        end_frame=int(online.frame.iloc[-1]),
    )
    online_cost = factor_cost_breakdown(navigation_states(online), archive)
    smoothed_cost = factor_cost_breakdown(navigation_states(smoothed), archive)
    cost_summary = write_factor_costs(
        output / "t2_history_smoother_factor_costs_per_edge.csv",
        online.frame.to_numpy(np.int64),
        online_cost,
        smoothed_cost,
    )

    baseline_contract: dict[str, float | bool] | None = None
    if args.baseline_online is not None:
        baseline = pd.read_csv(args.baseline_online)
        baseline = baseline.set_index("timestamp_ns").reindex(timestamps).reset_index()
        current = online[["tx_nwu", "ty_nwu", "tz_nwu"]].to_numpy(np.float64)
        reference = baseline[["tx", "ty", "tz"]].to_numpy(np.float64)
        current -= current[0]
        reference -= reference[0]
        baseline_contract = {
            "timestamps_match": bool(not baseline.isna().any().any()),
            "max_rebased_position_difference_m": float(np.max(np.abs(current - reference))),
        }

    online_metric, smooth_metric = metrics
    summary = {
        "schema_version": 1,
        "mode": "read_only_full_history_no_feedback",
        "metrics": metrics,
        "factor_costs": cost_summary,
        "baseline_online_contract": baseline_contract,
        "improvement_percent": {
            key: 100.0 * (float(online_metric[key]) - float(smooth_metric[key])) / float(online_metric[key])
            for key in (
                "ate_xyz_rmse_m",
                "ate_xy_rmse_m",
                "ape_rotation_rmse_deg",
                "rpe_frame_translation_rmse_m",
                "rpe_frame_rotation_rmse_deg",
                "velocity_rmse_mps",
                "acc_bias_rmse_mps2",
                "gyro_bias_rmse_radps",
            )
        },
    }
    (output / "t2_history_smoother_evaluation_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    report = f"""# T2 全历史 15 维平滑器 300 帧离线验证

## 结论

本实验只读取在线 T2 已生成的 IMU、bias random-walk 和压缩 UVD 视觉因子，不重新运行 MACVO，也不把结果反馈给实时 T2。在线轨迹输入与既有 T2 结果在重基准后的最大位置差为 `{baseline_contract['max_rebased_position_difference_m'] if baseline_contract else float('nan'):.3e} m`。

| 指标 | 在线 T2 | 全历史 15D | 改善 |
|---|---:|---:|---:|
| 3D ATE RMSE | {online_metric['ate_xyz_rmse_m']:.6f} m | {smooth_metric['ate_xyz_rmse_m']:.6f} m | {summary['improvement_percent']['ate_xyz_rmse_m']:.2f}% |
| XY ATE RMSE | {online_metric['ate_xy_rmse_m']:.6f} m | {smooth_metric['ate_xy_rmse_m']:.6f} m | {summary['improvement_percent']['ate_xy_rmse_m']:.2f}% |
| 姿态 APE RMSE | {online_metric['ape_rotation_rmse_deg']:.6f} deg | {smooth_metric['ape_rotation_rmse_deg']:.6f} deg | {summary['improvement_percent']['ape_rotation_rmse_deg']:.2f}% |
| 逐帧平移 RPE | {online_metric['rpe_frame_translation_rmse_m']:.6f} m | {smooth_metric['rpe_frame_translation_rmse_m']:.6f} m | {summary['improvement_percent']['rpe_frame_translation_rmse_m']:.2f}% |
| 逐帧旋转 RPE | {online_metric['rpe_frame_rotation_rmse_deg']:.6f} deg | {smooth_metric['rpe_frame_rotation_rmse_deg']:.6f} deg | {summary['improvement_percent']['rpe_frame_rotation_rmse_deg']:.2f}% |
| 速度 RMSE | {online_metric['velocity_rmse_mps']:.6f} m/s | {smooth_metric['velocity_rmse_mps']:.6f} m/s | {summary['improvement_percent']['velocity_rmse_mps']:.2f}% |
| 加速度 bias RMSE | {online_metric['acc_bias_rmse_mps2']:.6f} m/s^2 | {smooth_metric['acc_bias_rmse_mps2']:.6f} m/s^2 | {summary['improvement_percent']['acc_bias_rmse_mps2']:.2f}% |
| 陀螺 bias RMSE | {online_metric['gyro_bias_rmse_radps']:.6f} rad/s | {smooth_metric['gyro_bias_rmse_radps']:.6f} rad/s | {summary['improvement_percent']['gyro_bias_rmse_radps']:.2f}% |

## 因子代价

在线两状态估计在下一轮会重新调整当前状态，而已经移出活动窗口的旧 IMU 和 bias random-walk 因子不会再次闭合。因此，把最终在线状态重新放回完整历史图时，IMU 与 bias 代价分别为 `{cost_summary['online']['imu']:.6g}` 和 `{cost_summary['online']['bias']:.6g}`，这不是归档字段错位。全历史联合优化后，prior、IMU、bias 和压缩视觉代价分别为 `{cost_summary['smoothed']['prior']:.6g}`、`{cost_summary['smoothed']['imu']:.6g}`、`{cost_summary['smoothed']['bias']:.6g}`、`{cost_summary['smoothed']['visual']:.6g}`。该结果说明历史平滑确实在同一套 T2 因子上重新分配了误差，而不是对轨迹坐标做低通滤波。

## 适用范围

本轮仅验证 frame 90--299（源序列前 300 帧中的 209 条有效边）。结果证明离线全历史平滑路径工作且能降低高频 RPE；它尚不能证明全序列收益、异步实时调度和反馈重基准已经完成。
"""
    (output / "t2_history_smoother_report_cn.md").write_text(report, encoding="utf-8")
    write_plot(output, truth, online, smoothed, metrics)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
