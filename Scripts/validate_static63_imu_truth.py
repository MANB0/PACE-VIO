#!/usr/bin/env python3
"""Validate Static63 IMU calibration, decomposition, and oracle corrections."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pypose as pp
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Utility.IMUKinematics import (
    covariance_information_matrix,
    estimate_static_imu_initialization,
    imu_bias_sigma_to_continuous_random_walk_density,
    imu_sigma_to_continuous_density,
    vio_preintegrated_imu_residual,
)
from Utility.VIOConventionDiagnostics import camera_velocity_to_imu_origin


DEFAULT_DATA_ROOT = Path(
    "/mnt/e/文档/holoocean/code/recordings/"
    "batch_clear_truth_paths_20260713_static63_variants"
)
DEFAULT_OUTPUT = Path(
    "/home/admin1/macvo-dev/analysis_static63_imu_truth_validation_20260713"
)
GEOMETRIES = ("circle", "stop_turn_rectangle", "straight")
VARIANTS = ("normal_noise", "bias_no_noise", "noise_no_bias", "no_noise_no_bias")
NED_BASIS = torch.diag(torch.tensor([1.0, -1.0, -1.0], dtype=torch.float64))


_PREINT_SPEC = importlib.util.spec_from_file_location(
    "imu_preintegration_truth_validation",
    REPO_ROOT / "Module" / "IMUPreintegration.py",
)
_PREINT_MODULE = importlib.util.module_from_spec(_PREINT_SPEC)
assert _PREINT_SPEC is not None and _PREINT_SPEC.loader is not None
_PREINT_SPEC.loader.exec_module(_PREINT_MODULE)
preintegrate_imu = _PREINT_MODULE.preintegrate_imu


def _dataset(root: Path, geometry: str, variant: str) -> Path:
    return root / f"clear_{geometry}_truth_{variant}"


def _load_csv(path: Path) -> np.ndarray:
    return np.genfromtxt(path, delimiter=",", names=True)


def _columns(data: np.ndarray, names: tuple[str, str, str]) -> np.ndarray:
    return np.stack([data[name] for name in names], axis=1).astype(np.float64)


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _metadata(path: Path) -> dict:
    return json.loads((path / "metadata.json").read_text(encoding="utf-8"))


def _internal_vectors(values_flu: np.ndarray) -> torch.Tensor:
    values = torch.from_numpy(values_flu).double()
    return (NED_BASIS @ values.T).T.float()


def _pose_internal(ref: np.void) -> pp.LieTensor:
    quaternion = torch.tensor(
        [ref["qx"], ref["qy"], ref["qz"], ref["qw"]], dtype=torch.float64
    ).reshape(1, 4)
    rotation_nwu = pp.SO3(quaternion).matrix().reshape(3, 3)
    rotation_ned = pp.from_matrix(
        (NED_BASIS @ rotation_nwu @ NED_BASIS).reshape(1, 3, 3),
        pp.SO3_type,
    )
    position_nwu = torch.tensor([ref["x"], ref["y"], ref["z"]], dtype=torch.float64)
    position_ned = NED_BASIS @ position_nwu
    return pp.SE3(torch.cat([position_ned.reshape(1, 3), rotation_ned.tensor()], dim=1)).float()


def _camera_to_imu_internal(meta: dict) -> pp.LieTensor:
    extrinsics = meta["extrinsics"]
    body_imu = np.asarray(extrinsics["T_body_imu"]["translation_body_nwu_m"], dtype=np.float64)
    body_camera = np.asarray(extrinsics["T_body_camera"]["translation_body_nwu_m"], dtype=np.float64)
    camera_to_imu_nwu = body_imu - body_camera
    camera_to_imu_ned = NED_BASIS @ torch.from_numpy(camera_to_imu_nwu).double()
    return pp.SE3(
        torch.cat(
            [camera_to_imu_ned.reshape(1, 3), torch.tensor([[0.0, 0.0, 0.0, 1.0]])],
            dim=1,
        )
    ).float()


def _imu_velocity_internal(ref: np.void, pose: pp.LieTensor, lever: pp.LieTensor) -> torch.Tensor:
    return camera_velocity_to_imu_origin(
        camera_velocity_world_nwu=torch.tensor(
            [ref["vx"], ref["vy"], ref["vz"]], dtype=torch.float32
        ),
        angular_velocity_body_nwu=torch.tensor(
            [ref["wx"], ref["wy"], ref["wz"]], dtype=torch.float32
        ),
        camera_to_imu_body_internal=lever.translation().reshape(3),
        camera_rotation_body_to_world_internal=pose.rotation().matrix().reshape(3, 3),
        internal_world_frame="NED",
    ).float()


def _truth_arrays(truth: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "true_gyro": _columns(truth, ("true_ang_vel_x", "true_ang_vel_y", "true_ang_vel_z")),
        "gyro_bias": _columns(truth, ("gyro_bias_x", "gyro_bias_y", "gyro_bias_z")),
        "gyro_noise": _columns(truth, ("gyro_noise_x", "gyro_noise_y", "gyro_noise_z")),
        "measured_gyro": _columns(
            truth, ("measured_ang_vel_x", "measured_ang_vel_y", "measured_ang_vel_z")
        ),
        "true_acc": _columns(truth, ("true_lin_acc_x", "true_lin_acc_y", "true_lin_acc_z")),
        "acc_bias": _columns(truth, ("acc_bias_x", "acc_bias_y", "acc_bias_z")),
        "acc_noise": _columns(truth, ("acc_noise_x", "acc_noise_y", "acc_noise_z")),
        "measured_acc": _columns(
            truth, ("measured_lin_acc_x", "measured_lin_acc_y", "measured_lin_acc_z")
        ),
    }


def _query_interpolated_interval(
    timestamps: np.ndarray,
    values: np.ndarray,
    start_ns: int,
    end_ns: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    interior = (timestamps > int(start_ns)) & (timestamps < int(end_ns))
    output_time = np.concatenate(
        [np.asarray([start_ns], dtype=np.int64), timestamps[interior], np.asarray([end_ns], dtype=np.int64)]
    )
    output_values = np.empty((output_time.size, 3), dtype=np.float64)
    for axis in range(3):
        output_values[:, axis] = np.interp(output_time, timestamps, values[:, axis])
    return torch.from_numpy(output_time).long(), _internal_vectors(output_values)


def _variant_expected(arrays: dict[str, np.ndarray], variant: str) -> tuple[np.ndarray, np.ndarray]:
    keep_bias = variant in {"normal_noise", "bias_no_noise"}
    keep_noise = variant in {"normal_noise", "noise_no_bias"}
    gyro = arrays["true_gyro"].copy()
    acc = arrays["true_acc"].copy()
    if keep_bias:
        gyro += arrays["gyro_bias"]
        acc += arrays["acc_bias"]
    if keep_noise:
        gyro += arrays["gyro_noise"]
        acc += arrays["acc_noise"]
    return gyro, acc


def validate_variant_identity(root: Path) -> list[dict]:
    rows: list[dict] = []
    for geometry in GEOMETRIES:
        normal_path = _dataset(root, geometry, "normal_noise")
        truth = _load_csv(normal_path / "imu_truth_decomposition.csv")
        arrays = _truth_arrays(truth)
        truth_timestamps = truth["timestamp"].astype(np.int64)
        for variant in VARIANTS:
            imu = _load_csv(_dataset(root, geometry, variant) / "imu_data.csv")
            expected_gyro, expected_acc = _variant_expected(arrays, variant)
            actual_gyro = _columns(imu, ("ang_vel_x", "ang_vel_y", "ang_vel_z"))
            actual_acc = _columns(imu, ("lin_acc_x", "lin_acc_y", "lin_acc_z"))
            timestamp_match = np.array_equal(imu["timestamp"].astype(np.int64), truth_timestamps)
            gyro_error = actual_gyro - expected_gyro
            acc_error = actual_acc - expected_acc
            rows.append(
                {
                    "geometry": geometry,
                    "variant": variant,
                    "timestamp_match": timestamp_match,
                    "gyro_identity_rmse": float(np.sqrt(np.mean(gyro_error**2))),
                    "gyro_identity_max_abs": float(np.max(np.abs(gyro_error))),
                    "acc_identity_rmse": float(np.sqrt(np.mean(acc_error**2))),
                    "acc_identity_max_abs": float(np.max(np.abs(acc_error))),
                }
            )
    return rows


def validate_noise_and_bias_statistics(root: Path) -> list[dict]:
    rows: list[dict] = []
    for geometry in GEOMETRIES:
        path = _dataset(root, geometry, "normal_noise")
        meta = _metadata(path)
        imu_meta = meta["imu"]
        truth = _load_csv(path / "imu_truth_decomposition.csv")
        arrays = _truth_arrays(truth)
        expected_noise = {
            "gyro": float(imu_meta["AngVelSigma"]),
            "acc": float(imu_meta["AccelSigma"]),
        }
        expected_increment = imu_meta["expected_exported_bias_increment_std"]
        for sensor in ("gyro", "acc"):
            noise = arrays[f"{sensor}_noise"]
            bias_increment = np.diff(arrays[f"{sensor}_bias"], axis=0)
            for axis, axis_name in enumerate("xyz"):
                rows.append(
                    {
                        "geometry": geometry,
                        "sensor": sensor,
                        "axis": axis_name,
                        "noise_mean": float(noise[:, axis].mean()),
                        "noise_std": float(noise[:, axis].std()),
                        "expected_noise_std": expected_noise[sensor],
                        "noise_std_ratio": float(noise[:, axis].std() / expected_noise[sensor]),
                        "bias_increment_mean": float(bias_increment[:, axis].mean()),
                        "bias_increment_std": float(bias_increment[:, axis].std()),
                        "expected_bias_increment_std": float(expected_increment[sensor]),
                        "bias_increment_std_ratio": float(
                            bias_increment[:, axis].std() / float(expected_increment[sensor])
                        ),
                    }
                )
    return rows


def _static_initialization_for_variant(path: Path):
    meta = _metadata(path)
    imu_meta = meta["imu"]
    imu = _load_csv(path / "imu_data.csv")
    timestamps = imu["timestamp"].astype(np.int64)
    end_time = int(timestamps[0]) + 3_000_000_000
    mask = timestamps <= end_time
    time_tensor = torch.from_numpy(timestamps[mask]).long()
    acc = _internal_vectors(_columns(imu, ("lin_acc_x", "lin_acc_y", "lin_acc_z"))[mask])
    gyro = _internal_vectors(_columns(imu, ("ang_vel_x", "ang_vel_y", "ang_vel_z"))[mask])
    rate_hz = float(imu_meta["rate_hz"])
    acc_density = imu_sigma_to_continuous_density(
        imu_meta["AccelSigma"], rate_hz, imu_meta["sigma_unit"]
    )
    gyro_density = imu_sigma_to_continuous_density(
        imu_meta["AngVelSigma"], rate_hz, imu_meta["sigma_unit"]
    )
    acc_limit = max(float(acc_density) * rate_hz**0.5 * 5.0, 0.05)
    gyro_limit = max(float(gyro_density) * rate_hz**0.5 * 5.0, 0.005)
    result = estimate_static_imu_initialization(
        time_ns=time_tensor,
        acc_body=acc,
        gyro_body=gyro,
        initial_body_to_world=pp.identity_SO3(dtype=torch.float32),
        gravity=9.8,
        min_duration_s=2.99,
        gyro_mean_norm_max=0.03,
        gyro_std_max=gyro_limit,
        acc_norm_error_max=0.6,
        acc_std_max=acc_limit,
    )
    return result, mask


def validate_static_initialization(root: Path) -> list[dict]:
    rows: list[dict] = []
    for geometry in GEOMETRIES:
        normal_truth = _load_csv(
            _dataset(root, geometry, "normal_noise") / "imu_truth_decomposition.csv"
        )
        truth_arrays = _truth_arrays(normal_truth)
        for variant in VARIANTS:
            path = _dataset(root, geometry, variant)
            result, mask = _static_initialization_for_variant(path)
            keep_bias = variant in {"normal_noise", "bias_no_noise"}
            expected_gyro_bias_flu = (
                truth_arrays["gyro_bias"][mask].mean(axis=0) if keep_bias else np.zeros(3)
            )
            expected_gyro_bias = _internal_vectors(expected_gyro_bias_flu.reshape(1, 3)).reshape(3)
            gravity_world = torch.tensor([0.0, 0.0, 9.8])
            expected_specific_force = -result.body_to_world.Inv().Act(gravity_world).reshape(3)
            reconstruction_error = expected_specific_force + result.acc_bias - result.acc_mean
            rows.append(
                {
                    "geometry": geometry,
                    "variant": variant,
                    "stationary": bool(result.stationary),
                    "duration_s": float(result.duration_s),
                    "sample_count": int(result.sample_count),
                    "gyro_bias_error_norm": float((result.gyro_bias - expected_gyro_bias).norm().item()),
                    "acc_static_reconstruction_error_norm": float(reconstruction_error.norm().item()),
                    "estimated_acc_bias_x": float(result.acc_bias[0].item()),
                    "estimated_acc_bias_y": float(result.acc_bias[1].item()),
                    "estimated_acc_bias_z": float(result.acc_bias[2].item()),
                    "estimated_gyro_bias_x": float(result.gyro_bias[0].item()),
                    "estimated_gyro_bias_y": float(result.gyro_bias[1].item()),
                    "estimated_gyro_bias_z": float(result.gyro_bias[2].item()),
                    "failure_reason": result.failure_reason,
                }
            )
    return rows


def validate_preintegration_oracles(root: Path, pair_stride: int) -> tuple[list[dict], list[dict]]:
    pair_rows: list[dict] = []
    for geometry in GEOMETRIES:
        path = _dataset(root, geometry, "normal_noise")
        meta = _metadata(path)
        imu_meta = meta["imu"]
        truth = _load_csv(path / "imu_truth_decomposition.csv")
        arrays = _truth_arrays(truth)
        ref = _load_csv(path / "ref_pose.csv")
        timestamps = truth["timestamp"].astype(np.int64)
        static_result, _ = _static_initialization_for_variant(path)
        static_acc_bias = static_result.acc_bias.reshape(1, 3)
        static_gyro_bias = static_result.gyro_bias.reshape(1, 3)

        source = {
            "measured": (arrays["measured_gyro"], arrays["measured_acc"]),
            "minus_true_bias": (
                arrays["measured_gyro"] - arrays["gyro_bias"],
                arrays["measured_acc"] - arrays["acc_bias"],
            ),
            "minus_true_noise": (
                arrays["measured_gyro"] - arrays["gyro_noise"],
                arrays["measured_acc"] - arrays["acc_noise"],
            ),
            "truth": (arrays["true_gyro"], arrays["true_acc"]),
            "minus_static_estimated_bias": (
                arrays["measured_gyro"] - (NED_BASIS @ static_gyro_bias.double().T).T.numpy(),
                arrays["measured_acc"] - (NED_BASIS @ static_acc_bias.double().T).T.numpy(),
            ),
        }
        rate_hz = float(imu_meta["rate_hz"])
        sigma_acc = imu_sigma_to_continuous_density(
            imu_meta["AccelSigma"], rate_hz, imu_meta["sigma_unit"]
        )
        sigma_gyro = imu_sigma_to_continuous_density(
            imu_meta["AngVelSigma"], rate_hz, imu_meta["sigma_unit"]
        )
        bias_rate_hz = float(imu_meta.get("bias_random_walk_update_hz", rate_hz))
        sigma_acc_w = imu_bias_sigma_to_continuous_random_walk_density(
            imu_meta["AccelBiasSigma"], bias_rate_hz, imu_meta["bias_sigma_unit"]
        )
        sigma_gyro_w = imu_bias_sigma_to_continuous_random_walk_density(
            imu_meta["AngVelBiasSigma"], bias_rate_hz, imu_meta["bias_sigma_unit"]
        )
        lever = _camera_to_imu_internal(meta)

        for frame_index in range(1, len(ref), max(int(pair_stride), 1)):
            start = int(ref["timestamp"][frame_index - 1])
            end = int(ref["timestamp"][frame_index])
            if start < int(timestamps[0]) or end > int(timestamps[-1]):
                continue
            pose_i = _pose_internal(ref[frame_index - 1])
            pose_j = _pose_internal(ref[frame_index])
            velocity_i = _imu_velocity_internal(ref[frame_index - 1], pose_i, lever)
            velocity_j = _imu_velocity_internal(ref[frame_index], pose_j, lever)
            for oracle, (gyro_flu, acc_flu) in source.items():
                interval_time, interval_acc = _query_interpolated_interval(
                    timestamps, acc_flu, start, end
                )
                _, interval_gyro = _query_interpolated_interval(
                    timestamps, gyro_flu, start, end
                )
                preint = preintegrate_imu(
                    time_ns=interval_time,
                    acc=interval_acc,
                    gyro=interval_gyro,
                    R0_world=pose_i.rotation(),
                    gravity=9.8,
                    sigma_acc=sigma_acc,
                    sigma_gyro=sigma_gyro,
                    sigma_acc_w=sigma_acc_w,
                    sigma_gyro_w=sigma_gyro_w,
                )
                residual = vio_preintegrated_imu_residual(
                    from_pose=pose_i,
                    to_pose=pose_j,
                    prev_velocity_world=velocity_i,
                    curr_velocity_world=velocity_j,
                    delta_R=preint.delta_R,
                    delta_v=preint.delta_v,
                    delta_p=preint.delta_p,
                    dt_total=preint.dt_total,
                    sensor_T_imu=lever,
                ).reshape(9)
                information = covariance_information_matrix(preint.cov.double())
                nis = float((residual.double() @ information @ residual.double()).item())
                pair_rows.append(
                    {
                        "geometry": geometry,
                        "oracle": oracle,
                        "frame_index": frame_index,
                        "timestamp": end,
                        "position_residual_norm": float(residual[0:3].norm().item()),
                        "velocity_residual_norm": float(residual[3:6].norm().item()),
                        "rotation_residual_norm": float(residual[6:9].norm().item()),
                        "nis9": nis,
                    }
                )

    summary_rows: list[dict] = []
    for geometry in GEOMETRIES:
        for oracle in (
            "measured",
            "minus_true_bias",
            "minus_true_noise",
            "truth",
            "minus_static_estimated_bias",
        ):
            selected = [row for row in pair_rows if row["geometry"] == geometry and row["oracle"] == oracle]
            if not selected:
                continue
            summary_rows.append(
                {
                    "geometry": geometry,
                    "oracle": oracle,
                    "pair_count": len(selected),
                    "position_residual_rmse": float(
                        np.sqrt(np.mean([row["position_residual_norm"] ** 2 for row in selected]))
                    ),
                    "velocity_residual_rmse": float(
                        np.sqrt(np.mean([row["velocity_residual_norm"] ** 2 for row in selected]))
                    ),
                    "rotation_residual_rmse": float(
                        np.sqrt(np.mean([row["rotation_residual_norm"] ** 2 for row in selected]))
                    ),
                    "nis9_median": float(np.median([row["nis9"] for row in selected])),
                    "nis9_mean": float(np.mean([row["nis9"] for row in selected])),
                }
            )
    return pair_rows, summary_rows


def write_interactive_html(output: Path, summary: list[dict], static_rows: list[dict]) -> None:
    payload = json.dumps({"summary": summary, "static": static_rows}, ensure_ascii=True)
    html = r'''<!doctype html>
<html><head><meta charset="utf-8"><title>Static63 IMU truth diagnostics</title>
<style>
body{font-family:Arial,sans-serif;margin:0;background:#f3f4f6;color:#1f2937}header{padding:20px 28px;background:#fff;border-bottom:1px solid #d1d5db}h1{font-size:22px;margin:0 0 12px}.controls{display:flex;gap:12px;align-items:center;flex-wrap:wrap}select{padding:7px 10px;border:1px solid #9ca3af;background:#fff}main{padding:20px 28px}.panel{background:#fff;border:1px solid #d1d5db;border-radius:6px;padding:16px;margin-bottom:18px}canvas{width:100%;height:430px;display:block}table{width:100%;border-collapse:collapse;font-size:13px}th,td{text-align:right;padding:8px;border-bottom:1px solid #e5e7eb}th:first-child,td:first-child{text-align:left}.legend{display:flex;gap:14px;flex-wrap:wrap;font-size:12px;margin-top:10px}.swatch{width:12px;height:12px;display:inline-block;margin-right:5px;vertical-align:-2px}.note{color:#4b5563;font-size:13px}
</style></head><body><header><h1>Static63 IMU truth decomposition and oracle diagnostics</h1><div class="controls"><label>Geometry <select id="geometry"></select></label><label>Metric <select id="metric"><option value="position_residual_rmse">Position residual RMSE</option><option value="velocity_residual_rmse">Velocity residual RMSE</option><option value="rotation_residual_rmse">Rotation residual RMSE</option><option value="nis9_median">Median 9D NIS</option></select></label></div></header><main><section class="panel"><canvas id="chart"></canvas><div id="legend" class="legend"></div><p class="note">Log-scale bars. Hover a bar for the exact value.</p></section><section class="panel"><table><thead><tr><th>Oracle</th><th>p RMSE</th><th>v RMSE</th><th>R RMSE</th><th>NIS median</th><th>NIS mean</th></tr></thead><tbody id="rows"></tbody></table></section><section class="panel"><h2>Static initialization</h2><table><thead><tr><th>Variant</th><th>Stationary</th><th>Gyro bias error</th><th>Acc reconstruction error</th><th>Samples</th></tr></thead><tbody id="staticRows"></tbody></table></section></main><script>
const data=__PAYLOAD__;const colors=['#2563eb','#16a34a','#ea580c','#7c3aed','#db2777'];const geometry=document.getElementById('geometry');const metric=document.getElementById('metric');const canvas=document.getElementById('chart');const ctx=canvas.getContext('2d');let hits=[];
[...new Set(data.summary.map(x=>x.geometry))].forEach(x=>{const o=document.createElement('option');o.value=x;o.textContent=x;geometry.appendChild(o)});
function fmt(x){return Number(x).toExponential(4)}
function draw(){const rows=data.summary.filter(x=>x.geometry===geometry.value);const values=rows.map(x=>Math.max(Number(x[metric.value]),1e-30));const dpr=window.devicePixelRatio||1;const rect=canvas.getBoundingClientRect();canvas.width=rect.width*dpr;canvas.height=430*dpr;ctx.setTransform(dpr,0,0,dpr,0,0);ctx.clearRect(0,0,rect.width,430);const left=70,right=22,top=25,bottom=70,w=rect.width-left-right,h=430-top-bottom;const logs=values.map(Math.log10);let lo=Math.floor(Math.min(...logs)-.5),hi=Math.ceil(Math.max(...logs)+.5);if(lo===hi)hi=lo+1;ctx.strokeStyle='#9ca3af';ctx.fillStyle='#4b5563';ctx.font='12px Arial';for(let k=lo;k<=hi;k++){const y=top+(hi-k)/(hi-lo)*h;ctx.beginPath();ctx.moveTo(left,y);ctx.lineTo(left+w,y);ctx.strokeStyle='#e5e7eb';ctx.stroke();ctx.fillText('1e'+k,8,y+4)}hits=[];const gap=16,bw=Math.max(30,(w-gap*(rows.length+1))/rows.length);rows.forEach((row,i)=>{const x=left+gap+i*(bw+gap);const y=top+(hi-logs[i])/(hi-lo)*h;ctx.fillStyle=colors[i%colors.length];ctx.fillRect(x,y,bw,top+h-y);ctx.save();ctx.translate(x+bw/2,top+h+12);ctx.rotate(-.45);ctx.fillStyle='#374151';ctx.textAlign='right';ctx.fillText(row.oracle,0,0);ctx.restore();hits.push({x,y,w:bw,h:top+h-y,row,value:values[i]})});document.getElementById('legend').innerHTML=rows.map((r,i)=>`<span><i class="swatch" style="background:${colors[i%colors.length]}"></i>${r.oracle}</span>`).join('');document.getElementById('rows').innerHTML=rows.map(r=>`<tr><td>${r.oracle}</td><td>${fmt(r.position_residual_rmse)}</td><td>${fmt(r.velocity_residual_rmse)}</td><td>${fmt(r.rotation_residual_rmse)}</td><td>${fmt(r.nis9_median)}</td><td>${fmt(r.nis9_mean)}</td></tr>`).join('');const s=data.static.filter(x=>x.geometry===geometry.value);document.getElementById('staticRows').innerHTML=s.map(r=>`<tr><td>${r.variant}</td><td>${r.stationary}</td><td>${fmt(r.gyro_bias_error_norm)}</td><td>${fmt(r.acc_static_reconstruction_error_norm)}</td><td>${r.sample_count}</td></tr>`).join('')}
canvas.addEventListener('mousemove',e=>{const r=canvas.getBoundingClientRect(),x=e.clientX-r.left,y=e.clientY-r.top;const hit=hits.find(b=>x>=b.x&&x<=b.x+b.w&&y>=b.y&&y<=b.y+b.h);canvas.title=hit?`${hit.row.oracle}: ${hit.value}`:''});geometry.onchange=draw;metric.onchange=draw;window.onresize=draw;draw();
</script></body></html>'''.replace('__PAYLOAD__', payload)
    (output / "diagnostics_interactive.html").write_text(html, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--pair-stride", type=int, default=10)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    identity_rows = validate_variant_identity(args.data_root)
    noise_rows = validate_noise_and_bias_statistics(args.data_root)
    static_rows = validate_static_initialization(args.data_root)
    pair_rows, oracle_summary = validate_preintegration_oracles(args.data_root, args.pair_stride)

    _write_csv(args.output / "variant_identity.csv", identity_rows)
    _write_csv(args.output / "noise_bias_statistics.csv", noise_rows)
    _write_csv(args.output / "static_initialization.csv", static_rows)
    _write_csv(args.output / "preintegration_oracle_pairs.csv", pair_rows)
    _write_csv(args.output / "preintegration_oracle_summary.csv", oracle_summary)
    write_interactive_html(args.output, oracle_summary, static_rows)
    print(args.output / "diagnostics_interactive.html")


if __name__ == "__main__":
    main()
