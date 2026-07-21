#!/usr/bin/env python3
"""Summarize the controlled direct-UVD U1 covariance A/B replay."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import Scripts.run_circle_translation_oracles as base  # noqa: E402


OUT = ROOT / "analysis_direct_uvd_20260716"
CURRENT = OUT / "U1"
SAMPLING = OUT / "U1_sampling_aware"
COMPARISON = OUT / "U1_sampling_aware_ab"
ACTIVE_START = base.ACTIVE_START_FRAME


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def final_trace(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    return (
        frame.sort_values(["frame_i", "solver_call"])
        .groupby(["frame_i", "frame_j"], as_index=False)
        .tail(1)
        .query("frame_i >= @ACTIVE_START")
        .reset_index(drop=True)
    )


def nested_differences(left, right, prefix="") -> list[str]:
    if isinstance(left, dict) and isinstance(right, dict):
        result: list[str] = []
        for key in sorted(set(left) | set(right)):
            path = f"{prefix}.{key}" if prefix else key
            if key not in left or key not in right:
                result.append(path)
            else:
                result.extend(nested_differences(left[key], right[key], path))
        return result
    return [] if left == right else [prefix]


def read_run(root: Path) -> dict:
    summary = load_json(root / "summary.json")
    pose = pd.read_csv(Path(summary["artifacts"]["poses"]))
    diagnostics = pd.read_csv(Path(summary["artifacts"]["diagnostics"]))
    diagnostics = diagnostics.query("frame_i >= @ACTIVE_START").reset_index(drop=True)
    trace = final_trace(root / "factor_trace.csv")
    tensor_path = Path(summary["artifacts"]["tensor_map"])
    with np.load(tensor_path, allow_pickle=False) as tensor:
        covariance = tensor["frames//imu_vio_cov"].astype(np.float64)
    return {
        "summary": summary,
        "pose": pose,
        "diagnostics": diagnostics,
        "trace": trace,
        "covariance": covariance,
        "tensor_path": tensor_path,
    }


def pose_matrices(frame: pd.DataFrame) -> np.ndarray:
    result = np.tile(np.eye(4), (len(frame), 1, 1))
    result[:, :3, :3] = Rotation.from_quat(
        frame[["qx", "qy", "qz", "qw"]].to_numpy(np.float64)
    ).as_matrix()
    result[:, :3, 3] = frame[["tx", "ty", "tz"]].to_numpy(np.float64)
    return result


def edge_rows(name: str, run: dict, measurement: np.ndarray, gt: np.ndarray) -> pd.DataFrame:
    trace = run["trace"]
    diagnostics = run["diagnostics"].set_index(["frame_i", "frame_j"])
    pose = pose_matrices(run["pose"])
    covariance = run["covariance"]
    rows = []
    for item in trace.itertuples(index=False):
        frame_i = int(item.frame_i)
        frame_j = int(item.frame_j)
        cov = covariance[frame_j]
        z_est = np.linalg.inv(pose[frame_i]) @ pose[frame_j]
        z_gt = np.linalg.inv(gt[frame_i]) @ gt[frame_j]
        error = np.linalg.inv(z_gt) @ z_est
        correction = np.linalg.inv(measurement[frame_i]) @ z_est
        diagnostic = diagnostics.loc[(frame_i, frame_j)]
        rows.append(
            {
                "frame_i": frame_i,
                "frame_j": frame_j,
                f"{name}_imu_cov_trace": float(np.trace(cov)),
                f"{name}_imu_weight_trace": float(np.trace(np.linalg.inv(cov))),
                f"{name}_imu_whitened_norm": float(diagnostic.imu_vio_whitened_norm),
                f"{name}_translation_step_error_m": float(np.linalg.norm(error[:3, 3])),
                f"{name}_rotation_step_error_rad": float(
                    Rotation.from_matrix(error[:3, :3]).magnitude()
                ),
                f"{name}_macvo_correction_translation_m": float(
                    np.linalg.norm(correction[:3, 3])
                ),
                f"{name}_macvo_correction_rotation_rad": float(
                    Rotation.from_matrix(correction[:3, :3]).magnitude()
                ),
                f"{name}_prior_cost_after": float(item.prior_cost_after),
                f"{name}_imu_cost_after": float(item.imu_cost_after),
                f"{name}_bias_cost_after": float(item.bias_cost_after),
                f"{name}_pose_cost_after": float(item.pose_cost_after),
                f"{name}_iterations": int(item.iterations),
                f"{name}_converged": bool(item.converged),
            }
        )
    return pd.DataFrame(rows)


def metric_rows(current: dict, sampling: dict) -> pd.DataFrame:
    rows = []
    for key in current["metrics"]:
        current_value = float(current["metrics"][key])
        sampling_value = float(sampling["metrics"][key])
        rows.append(
            {
                "metric": key,
                "current": current_value,
                "sampling_aware": sampling_value,
                "absolute_change": sampling_value - current_value,
                "relative_change_percent": 100.0
                * (sampling_value - current_value)
                / max(abs(current_value), 1.0e-15),
            }
        )
    for key in ("imu_vio_cov_trace", "imu_vio_whitened_norm", "imu_vio_weight_trace"):
        current_value = float(current["covariance_diagnostics"][key]["mean"])
        sampling_value = float(sampling["covariance_diagnostics"][key]["mean"])
        rows.append(
            {
                "metric": f"mean_{key}",
                "current": current_value,
                "sampling_aware": sampling_value,
                "absolute_change": sampling_value - current_value,
                "relative_change_percent": 100.0
                * (sampling_value - current_value)
                / max(abs(current_value), 1.0e-15),
            }
        )
    return pd.DataFrame(rows)


def write_html(gt: pd.DataFrame, current: pd.DataFrame, sampling: pd.DataFrame) -> None:
    def payload(frame: pd.DataFrame, color: str) -> dict:
        rotation = Rotation.from_quat(
            frame[["qx", "qy", "qz", "qw"]].to_numpy(np.float64)
        )
        heading = rotation.apply(np.tile([1.0, 0.0, 0.0], (len(frame), 1)))
        return {
            "x": frame.tx.to_list(),
            "y": frame.ty.to_list(),
            "hx": heading[:, 0].tolist(),
            "hy": heading[:, 1].tolist(),
            "color": color,
        }

    data = {
        "GT": payload(gt, "#202833"),
        "U1 Current": payload(current, "#e2692d"),
        "U1 Sampling-aware": payload(sampling, "#2677d5"),
    }
    encoded = json.dumps(data, separators=(",", ":"))
    controls = "".join(
        f'<label><input type="checkbox" data-name="{name}" checked>'
        f'<span style="background:{series["color"]}"></span>{name}</label>'
        for name, series in data.items()
    )
    html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>U1 Sampling-aware covariance A/B</title><style>
*{{box-sizing:border-box}}body{{margin:0;background:#f4f6f8;color:#202833;font:15px/1.45 Arial,sans-serif}}
main{{max-width:1280px;margin:22px auto;padding:0 18px}}h1{{font-size:24px;margin:0}}p{{color:#667382;margin:5px 0 14px}}
.toolbar{{display:flex;gap:18px;flex-wrap:wrap;align-items:center;background:#fff;border:1px solid #dce2e8;padding:12px 14px}}
label{{display:flex;align-items:center;gap:7px}}label span{{width:20px;height:3px}}button{{background:#fff;border:1px solid #aeb8c2;padding:6px 10px}}
.plot{{background:#fff;border:1px solid #dce2e8;border-top:0;padding:10px}}canvas{{display:block;width:100%;aspect-ratio:16/9}}
</style></head><body><main><h1>Direct-UVD U1: Current vs Sampling-aware</h1>
<p>Circle normal-noise, 391 total frames, 300 active edges. No alignment or scale fitting. Arrows show body +X heading.</p>
<div class="toolbar">{controls}<button id="reset">Reset view</button></div><div class="plot"><canvas id="plot"></canvas></div>
<script>const S={encoded},C=document.getElementById('plot'),X=C.getContext('2d');let z=1,px=0,py=0,drag=false,lx=0,ly=0;
function active(){{return [...document.querySelectorAll('input[data-name]')].filter(x=>x.checked).map(x=>x.dataset.name)}}
function draw(){{let d=devicePixelRatio||1,w=C.clientWidth,h=C.clientHeight;C.width=w*d;C.height=h*d;X.setTransform(d,0,0,d,0,0);X.clearRect(0,0,w,h);X.fillStyle='#fff';X.fillRect(0,0,w,h);let n=active();if(!n.length)return;let xs=[],ys=[];n.forEach(k=>{{xs.push(...S[k].x);ys.push(...S[k].y)}});let a=Math.min(...xs),b=Math.max(...xs),c=Math.min(...ys),e=Math.max(...ys),pad=55,s=Math.min((w-2*pad)/(b-a||1),(h-2*pad)/(e-c||1))*z,cx=(a+b)/2,cy=(c+e)/2;let fx=q=>w/2+(q-cx)*s+px,fy=q=>h/2-(q-cy)*s+py;X.strokeStyle='#e4e8ec';for(let k=-5;k<=5;k++){{let xx=w/2+k*(w-2*pad)/10,yy=h/2+k*(h-2*pad)/10;X.beginPath();X.moveTo(xx,pad);X.lineTo(xx,h-pad);X.stroke();X.beginPath();X.moveTo(pad,yy);X.lineTo(w-pad,yy);X.stroke()}}n.forEach(k=>{{let q=S[k];X.strokeStyle=q.color;X.lineWidth=k==='GT'?3:2;X.beginPath();q.x.forEach((v,i)=>i?X.lineTo(fx(v),fy(q.y[i])):X.moveTo(fx(v),fy(q.y[i])));X.stroke();for(let i={ACTIVE_START};i<q.x.length;i+=25){{let hn=Math.hypot(q.hx[i],q.hy[i]);if(hn<1e-12)continue;let x0=fx(q.x[i]),y0=fy(q.y[i]),x1=x0+16*q.hx[i]/hn,y1=y0-16*q.hy[i]/hn,A=Math.atan2(y1-y0,x1-x0);X.beginPath();X.moveTo(x0,y0);X.lineTo(x1,y1);X.stroke();X.beginPath();X.moveTo(x1,y1);X.lineTo(x1-6*Math.cos(A-.5),y1-6*Math.sin(A-.5));X.lineTo(x1-6*Math.cos(A+.5),y1-6*Math.sin(A+.5));X.closePath();X.fillStyle=q.color;X.fill()}}}})}}
document.querySelectorAll('input[data-name]').forEach(x=>x.onchange=draw);window.onresize=draw;C.onwheel=e=>{{e.preventDefault();z*=e.deltaY<0?1.1:.9;draw()}};C.onpointerdown=e=>{{drag=true;lx=e.clientX;ly=e.clientY;C.setPointerCapture(e.pointerId)}};C.onpointermove=e=>{{if(!drag)return;px+=e.clientX-lx;py+=e.clientY-ly;lx=e.clientX;ly=e.clientY;draw()}};C.onpointerup=()=>drag=false;document.getElementById('reset').onclick=()=>{{z=1;px=py=0;draw()}};draw();
</script></main></body></html>"""
    (COMPARISON / "interactive_u1_current_vs_sampling_aware.html").write_text(
        html, encoding="utf-8"
    )


def main() -> None:
    COMPARISON.mkdir(parents=True, exist_ok=True)
    current = read_run(CURRENT)
    sampling = read_run(SAMPLING)
    current_summary = current["summary"]
    sampling_summary = sampling["summary"]

    current_config = yaml.safe_load(
        (CURRENT / "effective_odometry.yaml").read_text(encoding="utf-8")
    )
    sampling_config = yaml.safe_load(
        (SAMPLING / "effective_odometry.yaml").read_text(encoding="utf-8")
    )
    current_config["Odometry"]["args"].setdefault(
        "two_state_covariance_mode", "current_independent_step"
    )
    differences = nested_differences(current_config, sampling_config)
    expected_difference = ["Odometry.args.two_state_covariance_mode"]
    if differences != expected_difference:
        raise AssertionError(f"unexpected config differences: {differences}")

    for left, right, label in (
        (current["pose"].timestamp_ns, sampling["pose"].timestamp_ns, "pose timestamps"),
        (current["trace"].frame_i, sampling["trace"].frame_i, "trace frame_i"),
        (current["trace"].frame_j, sampling["trace"].frame_j, "trace frame_j"),
    ):
        if not np.array_equal(np.asarray(left), np.asarray(right)):
            raise AssertionError(f"A/B mismatch: {label}")

    ref = pd.read_csv(base.DEFAULT_DATASET / "ref_pose.csv").iloc[: len(current["pose"])]
    gt = np.tile(np.eye(4), (len(ref), 1, 1))
    gt[:, :3, :3] = Rotation.from_quat(
        ref[["qx", "qy", "qz", "qw"]].to_numpy(np.float64)
    ).as_matrix()
    gt[:, :3, 3] = ref[["x", "y", "z"]].to_numpy(np.float64)
    gt_frame = pd.DataFrame(
        {
            "tx": ref.x,
            "ty": ref.y,
            "tz": ref.z,
            "qx": ref.qx,
            "qy": ref.qy,
            "qz": ref.qz,
            "qw": ref.qw,
        }
    )
    cache = base.cache_root("short", "V0")
    with np.load(cache / "relative_pose_factors.npz", allow_pickle=False) as factors:
        measurement = np.stack(
            [base.se3_from_xyzw(row.reshape(7)) for row in factors["measurement_CiCj"]]
        )

    per_edge = edge_rows("current", current, measurement, gt).merge(
        edge_rows("sampling_aware", sampling, measurement, gt),
        on=["frame_i", "frame_j"],
        validate="one_to_one",
    )
    per_edge["cov_trace_ratio_sampling_over_current"] = (
        per_edge.sampling_aware_imu_cov_trace / per_edge.current_imu_cov_trace
    )
    per_edge["translation_step_error_change_m"] = (
        per_edge.sampling_aware_translation_step_error_m
        - per_edge.current_translation_step_error_m
    )
    per_edge["rotation_step_error_change_rad"] = (
        per_edge.sampling_aware_rotation_step_error_rad
        - per_edge.current_rotation_step_error_rad
    )
    per_edge_all_finite = bool(
        np.isfinite(per_edge.select_dtypes(include=[np.number]).to_numpy()).all()
    )
    if not per_edge_all_finite:
        raise AssertionError("per-edge A/B comparison contains NaN or Inf")
    per_edge.to_csv(COMPARISON / "per_edge_comparison.csv", index=False)

    metrics = metric_rows(current_summary, sampling_summary)
    metrics.to_csv(COMPARISON / "metric_comparison.csv", index=False)

    cache_contract = {
        "manifest": str(cache / "manifest.json"),
        "manifest_sha256": sha256(cache / "manifest.json"),
        "relative_pose_sidecar": str(cache / "relative_pose_factors.npz"),
        "relative_pose_sidecar_sha256": sha256(cache / "relative_pose_factors.npz"),
    }
    contract = {
        "same_input_timestamps": True,
        "same_edge_ids": True,
        "frame_count": int(len(current["pose"])),
        "active_edge_count": int(len(per_edge)),
        "active_start_frame": int(ACTIVE_START),
        "normalized_config_differences": differences,
        "current_mode": current_summary["covariance_mode"],
        "sampling_aware_mode": sampling_summary["covariance_mode"],
        "per_edge_all_finite": per_edge_all_finite,
        "shared_visual_cache": cache_contract,
    }
    (COMPARISON / "ab_contract.json").write_text(
        json.dumps(contract, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    def value(key: str, variant: str) -> float:
        row = metrics.loc[metrics.metric == key].iloc[0]
        return float(row[variant])

    def change(key: str) -> float:
        return float(metrics.loc[metrics.metric == key, "relative_change_percent"].iloc[0])

    decision = {
        "sampling_aware_math_regression_passed": True,
        "controlled_ab_contract_passed": True,
        "all_finite": bool(current_summary["all_finite"] and sampling_summary["all_finite"]),
        "sampling_aware_improves_high_frequency_xy": change(
            "xy_high_frequency_step_error_rmse_m"
        ) < 0.0,
        "approve_full_sequence_sampling_aware_u1": False,
        "reason": (
            "Sampling-aware covariance is mathematically validated, but in direct-UVD U1 "
            "it increases translation/rotation RPE and XY high-frequency step error on the "
            "controlled 300-active-edge replay."
        ),
        "recommended_next_experiment": (
            "Keep the sampling-aware model fixed and isolate online accelerometer/gyro bias "
            "optimization on the same replay before changing window length or weights."
        ),
    }
    (COMPARISON / "decision.json").write_text(
        json.dumps(decision, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    report = f"""# Direct-UVD U1 Sampling-aware covariance A/B 报告

## 结论

本轮是严格单变量实验：两边使用相同 391 帧、相同 300 条有效边、相同视觉缓存、相同 direct-UVD 因子、相同 MACVO warm-start、相同 bias 优化、相同 Schur prior 和相同 LM 参数。归一化后配置只差 `{differences[0]}`。

Sampling-aware 的数学实现是正确的，而且本轮所有状态与代价均为有限值；但是它没有降低 U1 的轨迹抖动。本短片段中，Sampling-aware 的平移 RPE 增加 `{change('translation_rpe_rmse_m'):.2f}%`，旋转 RPE 增加 `{change('rotation_rpe_rmse_rad'):.2f}%`，XY 高频步长误差增加 `{change('xy_high_frequency_step_error_rmse_m'):.2f}%`。因此目前不批准直接运行 Sampling-aware U1 完整序列。

## 主要数值

| 指标 | Current | Sampling-aware | 相对变化 |
|---|---:|---:|---:|
| XY ATE RMSE / m | {value('ate_xy_rmse_m_no_alignment','current'):.6f} | {value('ate_xy_rmse_m_no_alignment','sampling_aware'):.6f} | {change('ate_xy_rmse_m_no_alignment'):+.2f}% |
| Translation RPE RMSE / m | {value('translation_rpe_rmse_m','current'):.6f} | {value('translation_rpe_rmse_m','sampling_aware'):.6f} | {change('translation_rpe_rmse_m'):+.2f}% |
| Rotation RPE RMSE / rad | {value('rotation_rpe_rmse_rad','current'):.7f} | {value('rotation_rpe_rmse_rad','sampling_aware'):.7f} | {change('rotation_rpe_rmse_rad'):+.2f}% |
| XY 高频步长 RMSE / m | {value('xy_high_frequency_step_error_rmse_m','current'):.6f} | {value('xy_high_frequency_step_error_rmse_m','sampling_aware'):.6f} | {change('xy_high_frequency_step_error_rmse_m'):+.2f}% |
| Velocity truth RMSE / m/s | {value('velocity_truth_rmse_mps','current'):.6f} | {value('velocity_truth_rmse_mps','sampling_aware'):.6f} | {change('velocity_truth_rmse_mps'):+.2f}% |
| Acc-bias truth RMSE / m/s^2 | {value('acc_bias_truth_rmse_mps2','current'):.6f} | {value('acc_bias_truth_rmse_mps2','sampling_aware'):.6f} | {change('acc_bias_truth_rmse_mps2'):+.2f}% |
| Gyro-bias truth RMSE / rad/s | {value('gyro_bias_truth_rmse_radps','current'):.6f} | {value('gyro_bias_truth_rmse_radps','sampling_aware'):.6f} | {change('gyro_bias_truth_rmse_radps'):+.2f}% |
| Mean covariance trace | {value('mean_imu_vio_cov_trace','current'):.9e} | {value('mean_imu_vio_cov_trace','sampling_aware'):.9e} | {change('mean_imu_vio_cov_trace'):+.2f}% |
| Mean inverse-covariance trace | {value('mean_imu_vio_weight_trace','current'):.9e} | {value('mean_imu_vio_weight_trace','sampling_aware'):.9e} | {change('mean_imu_vio_weight_trace'):+.2f}% |
| Mean whitened IMU norm | {value('mean_imu_vio_whitened_norm','current'):.6f} | {value('mean_imu_vio_whitened_norm','sampling_aware'):.6f} | {change('mean_imu_vio_whitened_norm'):+.2f}% |
| Converged edge rate | {current_summary['convergence']['converged_rate']:.2%} | {sampling_summary['convergence']['converged_rate']:.2%} | - |
| Mean solver iterations | {current_summary['convergence']['iterations']['mean']:.2f} | {sampling_summary['convergence']['iterations']['mean']:.2f} | - |
| Runtime / s | {current_summary['runtime_seconds']:.1f} | {sampling_summary['runtime_seconds']:.1f} | {100.0 * (sampling_summary['runtime_seconds'] / current_summary['runtime_seconds'] - 1.0):+.2f}% |

## 如何解释

Sampling-aware 不是“给 IMU 降权”的固定规则。对这组 100 Hz IMU 到 30 Hz 相机边界插值，它识别出相邻积分 knot 共享原始样本，因此本片段得到的 covariance trace 比旧独立步模型平均小约 `{abs(change('mean_imu_vio_cov_trace')):.2f}%`，逆 covariance 的总量增大约 `{change('mean_imu_vio_weight_trace'):.2f}%`。优化器因此更强地追随带噪 IMU，whitened IMU residual 也增大，局部 RPE 与高频 XY 步长随之变差。

这不说明 Sampling-aware 公式错误；它说明旧模型在这个采样契约下更保守，而正确的 Sampling-aware covariance 暴露了两状态系统中的下一层问题。结合既有证据，下一步应保持 Sampling-aware 不变，在同一短片段上做 bias 状态消融，判断在线 `b_a/b_g` 是否在追逐白噪声。此时不应先调四个 sigma、LM 或视觉权重。

## 产物

- `metric_comparison.csv`：总体指标与相对变化。
- `per_edge_comparison.csv`：300 条边的 covariance、残差、RPE、cost、迭代与收敛对比。
- `ab_contract.json`：输入、缓存哈希和配置单变量契约。
- `decision.json`：是否批准完整序列及下一步建议。
- `interactive_u1_current_vs_sampling_aware.html`：带机头箭头的交互轨迹。
"""
    (COMPARISON / "report_cn.md").write_text(report, encoding="utf-8")
    write_html(gt_frame, current["pose"], sampling["pose"])
    print(json.dumps({"comparison": str(COMPARISON), **decision}, indent=2))


if __name__ == "__main__":
    main()
