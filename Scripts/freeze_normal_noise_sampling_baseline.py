#!/usr/bin/env python3
"""Freeze the 300-frame Standard-preintegration normal-noise baseline."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pypose as pp
import torch
from scipy import signal


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis_two_state_math_audit_20260715 import (  # noqa: E402
    run_short_sequence_audit as replay,
)
from Utility.TwoStateVIO import (  # noqa: E402
    TwoStateVIOSolver,
    _factor_residuals,
    _true_cost,
    state_boxminus,
)


FRAME_LIMIT = 300
FIRST_VALID_FRAME = 90
EXPECTED_EDGES = 209
ITERATION_LIMIT = 20
SCENE = "clear_stop_turn_rectangle_truth_normal_noise"
SOURCE_CACHE = ROOT / "VisualCache/static63_unique_visual_20260713" / SCENE
SOURCE_RESULT_ROOT = ROOT / "Results/rectangle_normal_noise_two_state_standard_full_20260715"
SOURCE_RESULT = (
    SOURCE_RESULT_ROOT
    / "trial_1/vio_two_state_fixed_lag_standard_full"
    / SCENE
)
ODOM_SOURCE = SOURCE_RESULT_ROOT / "configs/odometry.yaml"
DATASET_DIR = (
    Path("/mnt/e")
    / "\u6587\u6863/holoocean/code/recordings"
    / "batch_clear_truth_paths_20260713_static63_variants"
    / SCENE
)
PURE_MACVO_POSES = (
    ROOT
    / "Results/visual_factor_cache_static63_unique_source_20260713"
    / "trial_1/pure_macvo"
    / SCENE
    / "poses.csv"
)
DEFAULT_OUTPUT = ROOT / "analysis_normal_noise_sampling_aware_20260716"
FLU_TO_NED = np.diag([1.0, -1.0, -1.0])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scalar_stats(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"count": 0, "min": math.nan, "median": math.nan,
                "mean": math.nan, "p95": math.nan, "max": math.nan}
    return {
        "count": int(values.size),
        "min": float(np.min(values)),
        "median": float(np.median(values)),
        "mean": float(np.mean(values)),
        "p95": float(np.quantile(values, 0.95)),
        "max": float(np.max(values)),
    }


def append_vector(row: dict[str, Any], prefix: str, values: torch.Tensor | np.ndarray) -> None:
    array = np.asarray(
        values.detach().cpu().reshape(-1).tolist()
        if isinstance(values, torch.Tensor)
        else values,
        dtype=np.float64,
    ).reshape(-1)
    labels = ("x", "y", "z") if array.size == 3 else tuple(str(i) for i in range(array.size))
    for label, value in zip(labels, array):
        row[f"{prefix}_{label}"] = float(value)
    row[f"{prefix}_norm"] = float(np.linalg.norm(array))


def append_pose(row: dict[str, Any], prefix: str, pose_tensor: torch.Tensor) -> None:
    values = pose_tensor.detach().cpu().reshape(7).double().numpy()
    for label, value in zip(("tx", "ty", "tz", "qx", "qy", "qz", "qw"), values):
        row[f"{prefix}_{label}"] = float(value)


TRACE_FIELDS: list[str] = [
    "frame_i", "frame_j",
    "total_cost_before", "total_cost_after",
    "prior_cost_before", "prior_cost_after",
    "imu_cost_before", "imu_cost_after",
    "bias_cost_before", "bias_cost_after",
    "pose_cost_before", "pose_cost_after",
    "imu_whitened_residual_norm_before", "imu_whitened_residual_norm_after",
    "pose_whitened_residual_norm_before", "pose_whitened_residual_norm_after",
]
for state_name in ("i_before", "i_after", "j_before", "j_after"):
    TRACE_FIELDS.extend(
        [f"pose_{state_name}_{label}" for label in ("tx", "ty", "tz", "qx", "qy", "qz", "qw")]
    )
    for field in ("velocity", "ba", "bg"):
        TRACE_FIELDS.extend([f"{field}_{state_name}_{axis}" for axis in ("x", "y", "z")])
        TRACE_FIELDS.append(f"{field}_{state_name}_norm")
for delta_name in ("state_i_update", "state_j_update"):
    TRACE_FIELDS.extend([f"{delta_name}_{index}" for index in range(15)])
    TRACE_FIELDS.append(f"{delta_name}_norm")
for name in ("visual_measurement", "relative_pose_before", "relative_pose_after", "pose_correction_before", "pose_correction_after"):
    TRACE_FIELDS.extend([f"{name}_{index}" for index in range(6)])
    TRACE_FIELDS.append(f"{name}_norm")
TRACE_FIELDS.extend(
    [
        "marginal_prior_min_eigenvalue", "marginal_prior_max_eigenvalue",
        "marginal_prior_rank", "marginal_prior_condition_number",
        "solver_iteration_count", "solver_converged", "solver_convergence_reason",
        "final_step_norm", "final_gradient_inf_norm", "accepted_steps", "rejected_steps",
        "reintegrated", "pose_covariance_min_eigenvalue",
        "pose_covariance_max_eigenvalue", "pose_covariance_condition_number",
        "visual_pose_inlier_ratio", "visual_pose_mean_mahalanobis_sq",
        "visual_pose_whitened_residual_norm", "visual_pose_covariance_inflation",
        "visual_pose_gate_action",
    ]
)


def block_costs(blocks: dict[str, torch.Tensor], huber_delta: float) -> dict[str, float]:
    visual_norm = float(torch.linalg.vector_norm(blocks["visual_pose_unweighted"]).item())
    if visual_norm <= huber_delta:
        visual_cost = 0.5 * visual_norm * visual_norm
    else:
        visual_cost = huber_delta * visual_norm - 0.5 * huber_delta * huber_delta
    return {
        "prior": 0.5 * float(blocks["prior"].square().sum().item()),
        "imu": 0.5 * float(blocks["imu"].square().sum().item()),
        "bias": 0.5 * float(blocks["bias"].square().sum().item()),
        "pose": visual_cost,
        "imu_norm": float(torch.linalg.vector_norm(blocks["imu"]).item()),
        "pose_norm": visual_norm,
    }


def audited_solve(self: TwoStateVIOSolver, problem):
    dtype = torch.float64
    device = problem.state_i.pose_WB.device
    state_i = problem.state_i.to(device=device, dtype=dtype)
    state_j = problem.state_j.to(device=device, dtype=dtype)
    prior = problem.prior_i.to(device=device, dtype=dtype)
    imu = problem.imu.to(device=device, dtype=dtype)
    visual = problem.visual_pose.to(device=device, dtype=dtype)

    _, before_blocks = _factor_residuals(
        state_i, state_j, prior, imu, visual,
        covariance_eigenvalue_floor=self.covariance_eigenvalue_floor,
        robust_visual=False,
    )
    before = block_costs(before_blocks, visual.huber_delta)
    before_total = float(_true_cost(before_blocks, visual.huber_delta).item())
    result = replay.ORIGINAL_SOLVE(self, problem)
    _, after_blocks = _factor_residuals(
        result.state_i, result.state_j, prior, imu, visual,
        covariance_eigenvalue_floor=self.covariance_eigenvalue_floor,
        robust_visual=False,
    )
    after = block_costs(after_blocks, visual.huber_delta)
    after_total = float(_true_cost(after_blocks, visual.huber_delta).item())

    prior_hessian = result.prior_j.sqrt_information.mT @ result.prior_j.sqrt_information
    prior_values = torch.linalg.eigvalsh(0.5 * (prior_hessian + prior_hessian.mT))
    prior_scale = max(float(prior_values.abs().max().item()), 1.0)
    rank_tolerance = 15 * torch.finfo(dtype).eps * prior_scale
    active = prior_values[prior_values > rank_tolerance]
    prior_rank = int(active.numel())
    prior_condition = float((active.max() / active.min()).item()) if prior_rank else math.inf
    pose_values = torch.linalg.eigvalsh(0.5 * (visual.covariance + visual.covariance.mT))

    measurement = pp.SE3(visual.measurement_BiBj.reshape(1, 7))
    predicted_before = pp.SE3(state_i.pose_WB.reshape(1, 7)).Inv() @ pp.SE3(state_j.pose_WB.reshape(1, 7))
    predicted_after = pp.SE3(result.state_i.pose_WB.reshape(1, 7)).Inv() @ pp.SE3(result.state_j.pose_WB.reshape(1, 7))
    measurement_tangent = measurement.Log().tensor().reshape(6)
    relative_before = predicted_before.Log().tensor().reshape(6)
    relative_after = predicted_after.Log().tensor().reshape(6)
    correction_before = (measurement.Inv() @ predicted_before).Log().tensor().reshape(6)
    correction_after = (measurement.Inv() @ predicted_after).Log().tensor().reshape(6)

    row: dict[str, Any] = {
        "frame_i": replay.CURRENT_EDGE["frame_i"],
        "frame_j": replay.CURRENT_EDGE["frame_j"],
        "total_cost_before": before_total,
        "total_cost_after": after_total,
        "prior_cost_before": before["prior"], "prior_cost_after": after["prior"],
        "imu_cost_before": before["imu"], "imu_cost_after": after["imu"],
        "bias_cost_before": before["bias"], "bias_cost_after": after["bias"],
        "pose_cost_before": before["pose"], "pose_cost_after": after["pose"],
        "imu_whitened_residual_norm_before": before["imu_norm"],
        "imu_whitened_residual_norm_after": after["imu_norm"],
        "pose_whitened_residual_norm_before": before["pose_norm"],
        "pose_whitened_residual_norm_after": after["pose_norm"],
        "marginal_prior_min_eigenvalue": float(prior_values.min()),
        "marginal_prior_max_eigenvalue": float(prior_values.max()),
        "marginal_prior_rank": prior_rank,
        "marginal_prior_condition_number": prior_condition,
        "solver_iteration_count": result.iterations,
        "solver_converged": result.converged,
        "solver_convergence_reason": result.convergence_reason,
        "final_step_norm": result.final_step_norm,
        "final_gradient_inf_norm": result.final_gradient_inf_norm,
        "accepted_steps": result.accepted_steps,
        "rejected_steps": result.rejected_steps,
        "reintegrated": False,
        "pose_covariance_min_eigenvalue": float(pose_values.min()),
        "pose_covariance_max_eigenvalue": float(pose_values.max()),
        "pose_covariance_condition_number": float(pose_values.max() / pose_values.min()),
    }
    for name, state in (
        ("i_before", state_i), ("i_after", result.state_i),
        ("j_before", state_j), ("j_after", result.state_j),
    ):
        append_pose(row, f"pose_{name}", state.pose_WB)
        append_vector(row, f"velocity_{name}", state.velocity_W)
        append_vector(row, f"ba_{name}", state.acc_bias)
        append_vector(row, f"bg_{name}", state.gyro_bias)
    append_vector(row, "state_i_update", state_boxminus(result.state_i, state_i))
    append_vector(row, "state_j_update", state_boxminus(result.state_j, state_j))
    append_vector(row, "visual_measurement", measurement_tangent)
    append_vector(row, "relative_pose_before", relative_before)
    append_vector(row, "relative_pose_after", relative_after)
    append_vector(row, "pose_correction_before", correction_before)
    append_vector(row, "pose_correction_after", correction_after)
    replay.append_trace(row)
    return result


def read_csv_pose(path: Path, count: int) -> tuple[np.ndarray, pp.LieTensor]:
    frame = pd.read_csv(path).iloc[:count]
    time = frame["timestamp_ns"].to_numpy(np.int64)
    names = ["tx", "ty", "tz", "qx", "qy", "qz", "qw"]
    pose = pp.SE3(torch.from_numpy(frame[names].to_numpy(np.float64)))
    return time, pose


def gt_pose_and_velocity(count: int) -> tuple[np.ndarray, pp.LieTensor, np.ndarray]:
    frame = pd.read_csv(DATASET_DIR / "ref_pose.csv").iloc[:count]
    time = frame["timestamp"].to_numpy(np.int64)
    values = frame[["x", "y", "z", "qx", "qy", "qz", "qw"]].to_numpy(np.float64)
    nwu = pp.SE3(torch.from_numpy(values))
    nwu_to_ned = pp.SE3(torch.tensor([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]], dtype=torch.float64))
    pose = nwu_to_ned @ nwu @ nwu_to_ned.Inv()
    velocity = frame[["vx", "vy", "vz"]].to_numpy(np.float64) @ FLU_TO_NED.T
    return time, pose, velocity


def interpolate_rows(times: np.ndarray, values: np.ndarray, target: np.ndarray) -> np.ndarray:
    return np.column_stack(
        [np.interp(target.astype(np.float64), times.astype(np.float64), values[:, axis])
         for axis in range(values.shape[1])]
    )


def highpass_rms(values: np.ndarray, sample_rate_hz: float, cutoff_hz: float = 1.0) -> float:
    values = np.asarray(values, dtype=np.float64)
    if values.shape[0] < 16:
        return math.nan
    sos = signal.butter(4, cutoff_hz, btype="highpass", fs=sample_rate_hz, output="sos")
    filtered = signal.sosfiltfilt(sos, values, axis=0)
    return float(np.sqrt(np.mean(np.sum(filtered * filtered, axis=1))))


def rmse_norm(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    return float(np.sqrt(np.mean(np.sum(values * values, axis=1))))


def summarize_baseline(output_dir: Path) -> None:
    result_pose_path = replay.find_pose_file(replay.RUN_RESULT_ROOT)
    result_map_path = next(replay.RUN_RESULT_ROOT.rglob("tensor_map.npz"))
    shutil.copy2(result_pose_path, output_dir / "baseline_trajectory.csv")
    with np.load(result_map_path, allow_pickle=False) as tensor_map:
        frame_time = tensor_map["frames//time_ns"][:FRAME_LIMIT].astype(np.int64)
        estimate_pose = pp.SE3(torch.from_numpy(tensor_map["frames//pose"][:FRAME_LIMIT].astype(np.float64)))
        estimate_velocity = tensor_map["frames//imu_vio_velocity_world"][:FRAME_LIMIT].astype(np.float64)
        estimate_ba = tensor_map["frames//imu_vio_acc_bias"][:FRAME_LIMIT].astype(np.float64)
        estimate_bg = tensor_map["frames//imu_vio_gyro_bias"][:FRAME_LIMIT].astype(np.float64)

    gt_time, gt_pose, gt_velocity = gt_pose_and_velocity(FRAME_LIMIT)
    if not np.array_equal(frame_time, gt_time):
        raise AssertionError("Baseline tensor-map and GT timestamps differ")
    decomposition = pd.read_csv(DATASET_DIR / "imu_truth_decomposition.csv")
    imu_time = decomposition["timestamp"].to_numpy(np.int64)
    gt_ba_flu = decomposition[["acc_bias_x", "acc_bias_y", "acc_bias_z"]].to_numpy(np.float64)
    gt_bg_flu = decomposition[["gyro_bias_x", "gyro_bias_y", "gyro_bias_z"]].to_numpy(np.float64)
    gt_ba = interpolate_rows(imu_time, gt_ba_flu, frame_time) @ FLU_TO_NED.T
    gt_bg = interpolate_rows(imu_time, gt_bg_flu, frame_time) @ FLU_TO_NED.T

    pose_error = (gt_pose.Inv() @ estimate_pose).Log().tensor().numpy()
    position_error = estimate_pose.translation().numpy() - gt_pose.translation().numpy()
    velocity_error = estimate_velocity - gt_velocity
    ba_error = estimate_ba - gt_ba
    bg_error = estimate_bg - gt_bg
    estimate_relative = estimate_pose[:-1].Inv() @ estimate_pose[1:]
    gt_relative = gt_pose[:-1].Inv() @ gt_pose[1:]
    relative_error = (gt_relative.Inv() @ estimate_relative).Log().tensor().numpy()
    state_rows: list[dict[str, Any]] = []
    for index in range(FRAME_LIMIT):
        row: dict[str, Any] = {
            "frame": index,
            "timestamp_ns": int(frame_time[index]),
            "vio_active": index >= FIRST_VALID_FRAME,
        }
        for prefix, values in (
            ("p_error", position_error[index]),
            ("R_error", pose_error[index, 3:6]),
            ("v_est", estimate_velocity[index]), ("v_gt", gt_velocity[index]),
            ("v_error", velocity_error[index]),
            ("ba_est", estimate_ba[index]), ("ba_gt", gt_ba[index]),
            ("ba_error", ba_error[index]),
            ("bg_est", estimate_bg[index]), ("bg_gt", gt_bg[index]),
            ("bg_error", bg_error[index]),
        ):
            for axis, value in zip(("x", "y", "z"), values):
                row[f"{prefix}_{axis}"] = float(value)
            row[f"{prefix}_norm"] = float(np.linalg.norm(values))
        state_rows.append(row)
    pd.DataFrame(state_rows).to_csv(output_dir / "baseline_state_per_frame.csv", index=False)

    factor = pd.read_csv(replay.TRACE_PATH)
    if len(factor) != EXPECTED_EDGES:
        raise AssertionError(f"Expected {EXPECTED_EDGES} factor rows, found {len(factor)}")
    factor["visual_pose_gate_action"] = factor["visual_pose_gate_action"].astype("object")
    for row_index, row in factor.iterrows():
        gate = replay.GATE_BY_EDGE.get((int(row["frame_i"]), int(row["frame_j"])), {})
        for name, value in gate.items():
            factor.loc[row_index, name] = value
    factor.to_csv(replay.TRACE_PATH, index=False)

    valid = slice(FIRST_VALID_FRAME, FRAME_LIMIT)
    valid_time = frame_time[valid]
    sample_rate = 1.0 / float(np.median(np.diff(valid_time)) * 1e-9)
    pure_time, pure_pose = read_csv_pose(PURE_MACVO_POSES, FRAME_LIMIT)
    if not np.array_equal(frame_time, pure_time):
        raise AssertionError("Pure MACVO and baseline timestamps differ")
    xy_error = position_error[valid, :2]
    xy_correction = (
        estimate_pose.translation().numpy()[valid, :2]
        - pure_pose.translation().numpy()[valid, :2]
    )
    velocity_error_valid = velocity_error[valid]
    correction_columns = [f"pose_correction_after_{index}" for index in range(6)]
    relative_correction = factor[correction_columns].to_numpy(np.float64)
    relative_correction_change = np.diff(relative_correction, axis=0)

    iterations = factor["solver_iteration_count"].to_numpy(int)
    converged = factor["solver_converged"].astype(str).str.lower().eq("true").to_numpy()
    summary = {
        "gate": 0,
        "scene": SCENE,
        "frame_range": [0, FRAME_LIMIT - 1],
        "valid_edge_range": [FIRST_VALID_FRAME, FRAME_LIMIT - 1],
        "edge_count": int(len(factor)),
        "method": "D_two_state_fixed_lag",
        "preintegration": "standard_local_frame_preintegration",
        "continuous_sigmas": {
            "sigma_a": 0.0141258, "sigma_g": 0.00182898,
            "sigma_aw": 0.000386071, "sigma_gw": 3.57864e-05,
        },
        "truth_metrics_valid_frames": {
            "ate_position_rmse_m": rmse_norm(position_error[valid]),
            "xy_position_rmse_m": rmse_norm(position_error[valid, :2]),
            "orientation_rmse_rad": rmse_norm(pose_error[valid, 3:6]),
            "velocity_rmse_mps": rmse_norm(velocity_error[valid]),
            "acc_bias_rmse_mps2": rmse_norm(ba_error[valid]),
            "gyro_bias_rmse_radps": rmse_norm(bg_error[valid]),
            "translation_rpe_rmse_m": rmse_norm(
                relative_error[FIRST_VALID_FRAME:FRAME_LIMIT - 1, :3]
            ),
            "rotation_rpe_rmse_rad": rmse_norm(
                relative_error[FIRST_VALID_FRAME:FRAME_LIMIT - 1, 3:6]
            ),
        },
        "high_frequency_metrics_valid_frames": {
            "sample_rate_hz": sample_rate,
            "trend_cutoff_hz": 1.0,
            "xy_position_error_highpass_rms_m": highpass_rms(xy_error, sample_rate),
            "xy_position_correction_second_difference_rms_m": rmse_norm(np.diff(xy_correction, n=2, axis=0)),
            "velocity_truth_error_highpass_rms_mps": highpass_rms(velocity_error_valid, sample_rate),
            "relative_pose_correction_change_translation_rms_m": rmse_norm(relative_correction_change[:, :3]),
            "relative_pose_correction_change_rotation_rms_rad": rmse_norm(relative_correction_change[:, 3:]),
            "turn_and_stop_local_max": None,
            "turn_and_stop_note": "The frozen 0:300-frame slice ends before the first rectangle turn/stop event.",
        },
        "factor_cost_statistics": {
            column: scalar_stats(factor[column].to_numpy(np.float64))
            for column in (
                "prior_cost_before", "prior_cost_after", "imu_cost_before", "imu_cost_after",
                "bias_cost_before", "bias_cost_after", "pose_cost_before", "pose_cost_after",
            )
        },
        "solver": {
            "iteration_limit": ITERATION_LIMIT,
            "converged_count": int(converged.sum()),
            "converged_ratio": float(converged.mean()),
            "iteration_distribution": {
                str(value): int(np.sum(iterations == value)) for value in np.unique(iterations)
            },
        },
        "input_hashes": {
            "metadata_json": sha256_file(DATASET_DIR / "metadata.json"),
            "imu_data_csv": sha256_file(DATASET_DIR / "imu_data.csv"),
            "imu_truth_decomposition_csv": sha256_file(DATASET_DIR / "imu_truth_decomposition.csv"),
            "visual_sidecar": sha256_file(SOURCE_CACHE / "relative_pose_factors.npz"),
            "effective_odometry_yaml": sha256_file(replay.ODOM_CONFIG),
        },
        "production_code_modified": False,
    }
    (output_dir / "baseline_normal_noise_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    target_outputs = [
        output_dir / "baseline_normal_noise_summary.json",
        output_dir / "baseline_state_per_frame.csv",
        output_dir / "baseline_factor_per_edge.csv",
    ]
    if not args.force and any(path.exists() for path in target_outputs):
        raise FileExistsError("Frozen baseline already exists; pass --force to replace it")

    replay.OUT = output_dir
    replay.FRAME_LIMIT = FRAME_LIMIT
    replay.ITERATION_LIMIT = ITERATION_LIMIT
    replay.SOURCE_CACHE = SOURCE_CACHE
    replay.PREFIX_CACHE = output_dir / "baseline_cache_rectangle_300"
    replay.SOURCE_RESULT = SOURCE_RESULT
    replay.ODOM_SOURCE = ODOM_SOURCE
    replay.GT_PATH = DATASET_DIR / "ref_pose.csv"
    replay.RUN_RESULT_ROOT = output_dir / "baseline_run_result"
    replay.TRACE_PATH = output_dir / "baseline_factor_per_edge.csv"
    replay.ODOM_CONFIG = output_dir / "baseline_odometry.yaml"
    replay.DATA_CONFIG = output_dir / "baseline_data.yaml"
    replay.TRACE_FIELDS = TRACE_FIELDS
    replay.audited_solve = audited_solve
    replay.summarize_run = lambda: summarize_baseline(output_dir)
    replay.GATE_BY_EDGE.clear()
    return replay.main()


if __name__ == "__main__":
    raise SystemExit(main())
