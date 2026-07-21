#!/usr/bin/env python3
"""Prepare, run, and summarize circle visual-translation oracle replays.

Full-sequence execution is intentionally a separate explicit command.  Preparation never
launches MACVO, so the generated full batch can be handed to the user for manual start.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import runpy
import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pypose as pp
import torch
import yaml
from scipy.spatial.transform import Rotation


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Module.Optimization.TwoFramePGO.Optimizer import TwoFrame_PGO  # noqa: E402
from Utility.TwoStateVIO import (  # noqa: E402
    TwoStateVIOProblem,
    TwoStateVIOSolver,
    _factor_residuals,
    state_boxminus,
)
from Scripts.audit_circle_translation_oracle import (  # noqa: E402
    ACTIVE_START_FRAME,
    DEFAULT_CACHE,
    DEFAULT_DATASET,
    DEFAULT_OUTPUT,
    invert_transform,
    load_truth,
    make_transform,
    pose_nwu_to_internal,
    rotation_log,
    se3_from_xyzw,
    se3_to_xyzw,
)


OUT = DEFAULT_OUTPUT
SOURCE_CONFIG = ROOT / "analysis_initialization_boundary_audit_20260716/counterfactual_odometry_sampling_aware.yaml"
DATA_CONFIG_SOURCE = ROOT / "analysis_initialization_boundary_audit_20260716/counterfactual_data_circle.yaml"
SHORT_FRAME_LIMIT = 391
FULL_FRAME_LIMIT = 1890
FLU_TO_NED = np.diag([1.0, -1.0, -1.0])

MODES: dict[str, dict[str, Any]] = {
    "V0": {"rotation": "mac", "translation": "mac", "opt_ba": True, "opt_bg": True},
    "V1": {"rotation": "mac", "translation": "gt", "opt_ba": True, "opt_bg": True},
    "V2": {"rotation": "gt", "translation": "mac", "opt_ba": True, "opt_bg": True},
    "V3": {"rotation": "gt", "translation": "gt", "opt_ba": True, "opt_bg": True},
    "O3": {"rotation": "mac", "translation": "mac", "opt_ba": False, "opt_bg": True},
    "O4": {"rotation": "mac", "translation": "gt", "opt_ba": False, "opt_bg": True},
}
ALIASES = {"O1": "V0", "O2": "V1"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--prepare", action="store_true")
    action.add_argument("--run-short", action="store_true")
    action.add_argument("--run-one", action="store_true")
    action.add_argument("--summarize", action="store_true")
    parser.add_argument("--scope", choices=["short", "full", "both"], default="short")
    parser.add_argument("--mode", choices=sorted(MODES))
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def jsonify(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): jsonify(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonify(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(jsonify(payload), indent=2, ensure_ascii=False), encoding="utf-8")


def frame_limit(scope: str) -> int:
    return SHORT_FRAME_LIMIT if scope == "short" else FULL_FRAME_LIMIT


def mode_root(scope: str, mode: str) -> Path:
    return OUT / "oracles" / scope / mode


def cache_root(scope: str, mode: str) -> Path:
    return OUT / "oracle_caches" / scope / mode


def config_path(scope: str, mode: str) -> Path:
    return mode_root(scope, mode) / "odometry.yaml"


def safe_remove(path: Path, parent: Path) -> None:
    if not path.exists() and not path.is_symlink():
        return
    if path.is_symlink():
        if path.parent.resolve() != parent.resolve():
            raise RuntimeError(f"refusing to remove symlink outside {parent}: {path}")
        path.unlink()
        return
    if parent.resolve() not in path.resolve().parents:
        raise RuntimeError(f"refusing to remove path outside {parent}: {path}")
    if path.is_file():
        path.unlink()
    else:
        shutil.rmtree(path)


def prepare_cache(scope: str, mode: str) -> dict[str, Any]:
    spec = MODES[mode]
    limit = frame_limit(scope)
    destination = cache_root(scope, mode)
    destination.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((DEFAULT_CACHE / "manifest.json").read_text(encoding="utf-8"))
    manifest["frame_count"] = limit
    if isinstance(manifest.get("source"), dict):
        manifest["source"]["frame_count"] = limit
    manifest["timestamps_ns"] = manifest["timestamps_ns"][:limit]
    manifest["pairs"] = manifest["pairs"][: limit - 1]
    (destination / "manifest.json").write_text(
        json.dumps(manifest, separators=(",", ":"), ensure_ascii=False), encoding="utf-8"
    )
    pair_link = destination / "pairs"
    safe_remove(pair_link, destination)
    os.symlink(DEFAULT_CACHE / "pairs", pair_link, target_is_directory=True)

    _, truth_pose = load_truth(DEFAULT_DATASET)
    with np.load(DEFAULT_CACHE / "relative_pose_factors.npz", allow_pickle=False) as source:
        arrays = {
            key: (source[key].copy() if key == "schema_version" else source[key][: limit - 1].copy())
            for key in source.files
        }
    source_arrays = {key: value.copy() for key, value in arrays.items()}
    original_measurement = arrays["measurement_CiCj"].copy()
    replacement_count = 0
    max_rotation_change = 0.0
    max_translation_change = 0.0
    for frame_i in range(ACTIVE_START_FRAME, limit - 1):
        frame_j = frame_i + 1
        z_mac = se3_from_xyzw(original_measurement[frame_i].reshape(7))
        z_gt = invert_transform(truth_pose[frame_i]) @ truth_pose[frame_j]
        rotation = z_mac[:3, :3] if spec["rotation"] == "mac" else z_gt[:3, :3]
        translation = z_mac[:3, 3] if spec["translation"] == "mac" else z_gt[:3, 3]
        replacement = make_transform(rotation, translation)
        arrays["measurement_CiCj"][frame_i, 0] = se3_to_xyzw(replacement)
        max_rotation_change = max(
            max_rotation_change,
            float(np.linalg.norm(rotation_log(z_mac[:3, :3].T @ replacement[:3, :3]))),
        )
        max_translation_change = max(
            max_translation_change, float(np.linalg.norm(z_mac[:3, 3] - replacement[:3, 3]))
        )
        replacement_count += 1
    with (destination / "relative_pose_factors.npz").open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    covariance_difference = 0.0
    quality_unchanged = all(
        np.array_equal(arrays[key], source_arrays[key])
        for key in ("covariance", "num_points", "num_inliers", "mean_mahalanobis_sq", "visual_sha256")
    )
    return {
        "scope": scope,
        "mode": mode,
        "frame_limit": limit,
        "spec": spec,
        "active_replacement_count": replacement_count,
        "max_rotation_measurement_change_rad": max_rotation_change,
        "max_translation_measurement_change_m": max_translation_change,
        "covariance_max_abs_change": covariance_difference,
        "quality_fields_bitwise_unchanged": quality_unchanged,
        "pre_static_measurements_bitwise_unchanged": bool(
            np.array_equal(
                arrays["measurement_CiCj"][:ACTIVE_START_FRAME],
                original_measurement[:ACTIVE_START_FRAME],
            )
        ),
    }


def prepare_config(scope: str, mode: str) -> dict[str, Any]:
    destination = mode_root(scope, mode)
    destination.mkdir(parents=True, exist_ok=True)
    source = yaml.safe_load(SOURCE_CONFIG.read_text(encoding="utf-8"))
    config = copy.deepcopy(source)
    config["Odometry"]["args"]["visual_cache_path"] = str(cache_root(scope, mode))
    optimizer = config["Odometry"]["optimizer"]["args"]
    optimizer["two_state_optimize_acc_bias"] = bool(MODES[mode]["opt_ba"])
    optimizer["two_state_optimize_gyro_bias"] = bool(MODES[mode]["opt_bg"])
    config_path(scope, mode).write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return {
        "changed_keys_only": {
            "Odometry.args.visual_cache_path": str(cache_root(scope, mode)),
            "Odometry.optimizer.args.two_state_optimize_acc_bias": bool(MODES[mode]["opt_ba"]),
            "Odometry.optimizer.args.two_state_optimize_gyro_bias": bool(MODES[mode]["opt_bg"]),
        },
        "locked": {
            "covariance_mode": config["Odometry"]["args"]["two_state_covariance_mode"],
            "gravity_handling": config["Odometry"]["args"]["imu_vio_gravity_handling"],
            "max_iterations": optimizer["two_state_max_iterations"],
            "visual_huber_delta": optimizer["two_state_visual_huber_delta"],
            "factor_mode": optimizer["imu_factor_mode"],
        },
    }


def prepare(scope: str) -> None:
    contracts = {}
    for mode in MODES:
        cache_contract = prepare_cache(scope, mode)
        config_contract = prepare_config(scope, mode)
        contracts[mode] = {"cache": cache_contract, "config": config_contract}
    write_json(OUT / f"circle_visual_oracle_{scope}_preparation_contract.json", contracts)


TRACE_ROWS: list[dict[str, Any]] = []
CURRENT_EDGE = {"i": -1, "j": -1, "call": 0}
ORIGINAL_SOLVE = TwoStateVIOSolver.solve
ORIGINAL_OPTIMIZE = TwoFrame_PGO._optimize_two_state_fixed_lag


def factor_cost(residual: torch.Tensor) -> float:
    return 0.5 * float(residual.square().sum().item())


def traced_solve(self: TwoStateVIOSolver, problem: TwoStateVIOProblem):
    dtype = torch.float64
    device = problem.state_i.pose_WB.device
    normalized = replace(
        problem,
        state_i=problem.state_i.to(device=device, dtype=dtype),
        state_j=problem.state_j.to(device=device, dtype=dtype),
        prior_i=problem.prior_i.to(device=device, dtype=dtype),
        imu=problem.imu.to(device=device, dtype=dtype),
        visual_pose=problem.visual_pose.to(device=device, dtype=dtype),
    )
    _, before = _factor_residuals(
        normalized.state_i,
        normalized.state_j,
        normalized.prior_i,
        normalized.imu,
        normalized.visual_pose,
        covariance_eigenvalue_floor=self.covariance_eigenvalue_floor,
        robust_visual=False,
    )
    result = ORIGINAL_SOLVE(self, problem)
    _, after = _factor_residuals(
        result.state_i,
        result.state_j,
        normalized.prior_i,
        normalized.imu,
        normalized.visual_pose,
        covariance_eigenvalue_floor=self.covariance_eigenvalue_floor,
        robust_visual=False,
    )
    update_i = state_boxminus(result.state_i, normalized.state_i).detach().cpu().numpy()
    update_j = state_boxminus(result.state_j, normalized.state_j).detach().cpu().numpy()
    CURRENT_EDGE["call"] += 1
    row: dict[str, Any] = {
        "frame_i": CURRENT_EDGE["i"], "frame_j": CURRENT_EDGE["j"], "solver_call": CURRENT_EDGE["call"],
        "iterations": int(result.iterations), "converged": bool(result.converged),
        "final_step_norm": float(result.final_step_norm),
        "final_gradient_inf_norm": float(result.final_gradient_inf_norm),
        "convergence_reason": str(result.convergence_reason),
    }
    for name in ("prior", "imu", "bias"):
        row[f"{name}_cost_before"] = factor_cost(before[name])
        row[f"{name}_cost_after"] = factor_cost(after[name])
    row["pose_cost_before"] = factor_cost(before["visual_pose_unweighted"])
    row["pose_cost_after"] = factor_cost(after["visual_pose_unweighted"])
    for state_name, state_before, state_after, update in (
        ("i", normalized.state_i, result.state_i, update_i),
        ("j", normalized.state_j, result.state_j, update_j),
    ):
        for sensor_name, before_values, after_values in (
            ("v", state_before.velocity_W, state_after.velocity_W),
            ("ba", state_before.acc_bias, state_after.acc_bias),
            ("bg", state_before.gyro_bias, state_after.gyro_bias),
        ):
            for axis, value_before, value_after in zip("xyz", before_values, after_values):
                row[f"{sensor_name}_{state_name}_before_{axis}"] = float(value_before)
                row[f"{sensor_name}_{state_name}_after_{axis}"] = float(value_after)
        for index, value in enumerate(update):
            row[f"state_{state_name}_update_{index}"] = float(value)
    TRACE_ROWS.append(row)
    return result


def traced_optimize(context, graph_data):
    CURRENT_EDGE["i"] = int(graph_data.from_idx.reshape(-1)[0].item())
    CURRENT_EDGE["j"] = int(graph_data.frame_idx.reshape(-1)[0].item())
    CURRENT_EDGE["call"] = 0
    return ORIGINAL_OPTIMIZE(context, graph_data)


def latest(root: Path, name: str) -> Path:
    candidates = sorted(root.rglob(name))
    if not candidates:
        raise FileNotFoundError(f"{root} has no {name}")
    return candidates[-1]


def run_one(scope: str, mode: str, *, force: bool) -> None:
    root = mode_root(scope, mode)
    run_root = root / "run_result"
    trace_path = root / "factor_trace.csv"
    if run_root.exists():
        if not force:
            raise FileExistsError(f"{run_root} exists; pass --force")
        safe_remove(run_root, root)
    TRACE_ROWS.clear()
    TwoStateVIOSolver.solve = traced_solve
    TwoFrame_PGO._optimize_two_state_fixed_lag = staticmethod(traced_optimize)
    try:
        sys.argv = [
            str(ROOT / "MACVO.py"),
            "--odom", str(config_path(scope, mode)),
            "--data", str(DATA_CONFIG_SOURCE),
            "--resultRoot", str(run_root),
            "--visual-cache-mode", "replay",
            "--visual-cache-path", str(cache_root(scope, mode)),
            "--seq_to", str(frame_limit(scope)),
        ]
        runpy.run_path(str(ROOT / "MACVO.py"), run_name="__main__")
    finally:
        TwoStateVIOSolver.solve = ORIGINAL_SOLVE
        TwoFrame_PGO._optimize_two_state_fixed_lag = staticmethod(ORIGINAL_OPTIMIZE)
    if not TRACE_ROWS:
        raise RuntimeError("oracle replay produced no two-state trace rows")
    pd.DataFrame(TRACE_ROWS).to_csv(trace_path, index=False)
    summarize_one(scope, mode)


def interpolate_rows(time_source: np.ndarray, values: np.ndarray, time_target: np.ndarray) -> np.ndarray:
    return np.stack([np.interp(time_target, time_source, values[:, axis]) for axis in range(values.shape[1])], axis=1)


def rmse_norm(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum(np.asarray(values, dtype=np.float64) ** 2, axis=1))))


def distribution(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    return {
        "mean": float(np.mean(values)), "median": float(np.median(values)),
        "p95": float(np.quantile(values, 0.95)), "max": float(np.max(values)),
    }


def read_pose_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    frame = pd.read_csv(path)
    time = frame["timestamp_ns"].to_numpy(np.int64)
    pose = np.stack(
        [make_transform(Rotation.from_quat(row[3:7]).as_matrix(), row[:3]) for row in frame[["tx", "ty", "tz", "qx", "qy", "qz", "qw"]].to_numpy(np.float64)]
    )
    return time, pose


def trajectory_metrics(estimate: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    position_error = estimate[:, :3, 3] - truth[:, :3, 3]
    rotation_error = np.stack(
        [rotation_log(truth[k, :3, :3].T @ estimate[k, :3, :3]) for k in range(len(estimate))]
    )
    trans_rpe, rot_rpe = [], []
    for k in range(len(estimate) - 1):
        z_est = invert_transform(estimate[k]) @ estimate[k + 1]
        z_gt = invert_transform(truth[k]) @ truth[k + 1]
        error = invert_transform(z_gt) @ z_est
        trans_rpe.append(np.linalg.norm(error[:3, 3]))
        rot_rpe.append(np.linalg.norm(rotation_log(error[:3, :3])))
    xy_error = position_error[:, :2]
    xy_step = np.diff(xy_error, axis=0)
    yaw_est = np.unwrap([Rotation.from_matrix(p[:3, :3]).as_euler("xyz")[2] for p in estimate])
    yaw_gt = np.unwrap([Rotation.from_matrix(p[:3, :3]).as_euler("xyz")[2] for p in truth])
    return {
        "ate_xyz_rmse_m_no_alignment": rmse_norm(position_error),
        "ate_xy_rmse_m_no_alignment": rmse_norm(xy_error),
        "orientation_rmse_rad": rmse_norm(rotation_error),
        "translation_rpe_rmse_m": float(np.sqrt(np.mean(np.square(trans_rpe)))),
        "rotation_rpe_rmse_rad": float(np.sqrt(np.mean(np.square(rot_rpe)))),
        "xy_high_frequency_step_error_rmse_m": rmse_norm(xy_step),
        "cumulative_yaw_error_rad": float((yaw_est[-1] - yaw_est[0]) - (yaw_gt[-1] - yaw_gt[0])),
    }


def summarize_one(scope: str, mode: str) -> dict[str, Any]:
    root = mode_root(scope, mode)
    run_root = root / "run_result"
    pose_path = latest(run_root, "poses.csv")
    tensor_path = latest(run_root, "tensor_map.npz")
    diagnostics_path = latest(run_root, "frame_pair_diagnostics.csv")
    time, pose_est = read_pose_csv(pose_path)
    ref = pd.read_csv(DEFAULT_DATASET / "ref_pose.csv").iloc[: len(time)]
    if not np.array_equal(time, ref["timestamp"].to_numpy(np.int64)):
        raise AssertionError("oracle result timestamps differ from reference")
    pose_gt = np.stack(
        [make_transform(Rotation.from_quat(row[3:7]).as_matrix(), row[:3]) for row in ref[["x", "y", "z", "qx", "qy", "qz", "qw"]].to_numpy(np.float64)]
    )
    active = slice(ACTIVE_START_FRAME, len(time))
    metrics = trajectory_metrics(pose_est[active], pose_gt[active])
    with np.load(tensor_path, allow_pickle=False) as tensor:
        velocity_est = tensor["frames//imu_vio_velocity_world"][: len(time)].astype(np.float64)
        ba_est = tensor["frames//imu_vio_acc_bias"][: len(time)].astype(np.float64)
        bg_est = tensor["frames//imu_vio_gyro_bias"][: len(time)].astype(np.float64)
        pose_internal = np.stack([se3_from_xyzw(row) for row in tensor["frames//pose"][: len(time)].astype(np.float64)])
    diagnostics = pd.read_csv(diagnostics_path)
    valid_diag = diagnostics[(diagnostics["frame_i"] >= ACTIVE_START_FRAME) & (diagnostics["frame_j"] < len(time))]
    velocity_truth_by_frame: dict[int, np.ndarray] = {}
    for row in valid_diag.itertuples(index=False):
        velocity_truth_by_frame[int(row.frame_i)] = np.array([row.gt_velocity_i_x, row.gt_velocity_i_y, row.gt_velocity_i_z])
        velocity_truth_by_frame[int(row.frame_j)] = np.array([row.gt_velocity_j_x, row.gt_velocity_j_y, row.gt_velocity_j_z])
    fallback_velocity = ref[["vx", "vy", "vz"]].to_numpy(np.float64) @ FLU_TO_NED.T
    velocity_gt = np.stack([velocity_truth_by_frame.get(index, fallback_velocity[index]) for index in range(len(time))])
    decomposition = pd.read_csv(DEFAULT_DATASET / "imu_truth_decomposition.csv")
    imu_time = decomposition["timestamp"].to_numpy(np.int64)
    ba_gt = interpolate_rows(
        imu_time, decomposition[["acc_bias_x", "acc_bias_y", "acc_bias_z"]].to_numpy(np.float64), time
    ) @ FLU_TO_NED.T
    bg_gt = interpolate_rows(
        imu_time, decomposition[["gyro_bias_x", "gyro_bias_y", "gyro_bias_z"]].to_numpy(np.float64), time
    ) @ FLU_TO_NED.T
    metrics.update(
        {
            "velocity_truth_rmse_mps": rmse_norm(velocity_est[active] - velocity_gt[active]),
            "acc_bias_truth_rmse_mps2": rmse_norm(ba_est[active] - ba_gt[active]),
            "gyro_bias_truth_rmse_radps": rmse_norm(bg_est[active] - bg_gt[active]),
        }
    )
    metadata = json.loads((DEFAULT_DATASET / "metadata.json").read_text(encoding="utf-8"))
    imu_meta = metadata["imu"]
    sigma_aw = float(imu_meta["AccelBiasSigma"]) * math.sqrt(float(imu_meta["bias_random_walk_update_hz"]))
    sigma_gw = float(imu_meta["AngVelBiasSigma"]) * math.sqrt(float(imu_meta["bias_random_walk_update_hz"]))
    dt = np.diff(time[ACTIVE_START_FRAME:]).astype(np.float64) * 1.0e-9
    ba_standard = np.diff(ba_est[active], axis=0) / (sigma_aw * np.sqrt(dt)[:, None])
    bg_standard = np.diff(bg_est[active], axis=0) / (sigma_gw * np.sqrt(dt)[:, None])
    standardized = {
        "ba_increment_norm": distribution(np.linalg.norm(ba_standard, axis=1)),
        "bg_increment_norm": distribution(np.linalg.norm(bg_standard, axis=1)),
        "ba_axis_rms": np.sqrt(np.mean(ba_standard**2, axis=0)),
        "bg_axis_rms": np.sqrt(np.mean(bg_standard**2, axis=0)),
    }
    with np.load(cache_root(scope, mode) / "relative_pose_factors.npz", allow_pickle=False) as factor_cache:
        measurement = np.stack([se3_from_xyzw(row.reshape(7)) for row in factor_cache["measurement_CiCj"]])
    correction_rows = []
    for i in range(ACTIVE_START_FRAME, len(time) - 1):
        z_est = invert_transform(pose_internal[i]) @ pose_internal[i + 1]
        correction = invert_transform(measurement[i]) @ z_est
        correction_rows.append(np.r_[correction[:3, 3], rotation_log(correction[:3, :3])])
    correction_array = np.asarray(correction_rows)
    trace = pd.read_csv(root / "factor_trace.csv")
    trace = trace.sort_values(["frame_i", "solver_call"]).groupby(["frame_i", "frame_j"], as_index=False).tail(1)
    trace = trace[(trace["frame_i"] >= ACTIVE_START_FRAME) & (trace["frame_j"] < len(time))].reset_index(drop=True)
    if len(trace) != len(time) - ACTIVE_START_FRAME - 1:
        raise AssertionError(f"{mode}/{scope}: final trace does not contain one row per active edge")
    if not MODES[mode]["opt_ba"]:
        ba_update_columns = [f"state_{state}_update_{index}" for state in ("i", "j") for index in range(9, 12)]
        max_ba_update = float(np.max(np.abs(trace[ba_update_columns].to_numpy(np.float64))))
        if max_ba_update > 1.0e-12:
            raise AssertionError(f"{mode}: fixed accelerometer bias changed by {max_ba_update}")
    factor_costs = {
        name: {
            "before": distribution(trace[f"{name}_cost_before"].to_numpy()),
            "after": distribution(trace[f"{name}_cost_after"].to_numpy()),
            "sum_before": float(trace[f"{name}_cost_before"].sum()),
            "sum_after": float(trace[f"{name}_cost_after"].sum()),
        }
        for name in ("prior", "imu", "bias", "pose")
    }
    first_trace = trace.iloc[0]
    first_update = {
        "ba_i_before": [first_trace[f"ba_i_before_{axis}"] for axis in "xyz"],
        "ba_i_after": [first_trace[f"ba_i_after_{axis}"] for axis in "xyz"],
        "bg_i_before": [first_trace[f"bg_i_before_{axis}"] for axis in "xyz"],
        "bg_i_after": [first_trace[f"bg_i_after_{axis}"] for axis in "xyz"],
    }
    convergence = {
        "edge_count": int(len(trace)),
        "converged_rate": float(trace["converged"].astype(bool).mean()),
        "iterations": distribution(trace["iterations"].to_numpy(np.float64)),
        "reached_iteration_limit_count": int((trace["iterations"] >= 20).sum()),
    }
    per_frame = []
    pose_position_error = pose_est[:, :3, 3] - pose_gt[:, :3, 3]
    pose_rotation_error = np.stack(
        [rotation_log(pose_gt[k, :3, :3].T @ pose_est[k, :3, :3]) for k in range(len(time))]
    )
    for index in range(ACTIVE_START_FRAME, len(time)):
        row = {
            "scope": scope, "mode": mode, "frame": index, "timestamp_ns": int(time[index]),
        }
        for prefix, values in (
            ("position_error", pose_position_error[index]),
            ("rotation_error", pose_rotation_error[index]),
            ("velocity_est", velocity_est[index]), ("velocity_gt", velocity_gt[index]),
            ("ba_est", ba_est[index]), ("ba_gt", ba_gt[index]),
            ("bg_est", bg_est[index]), ("bg_gt", bg_gt[index]),
        ):
            for axis, value in zip("xyz", values):
                row[f"{prefix}_{axis}"] = float(value)
        if index < len(time) - 1:
            correction = correction_array[index - ACTIVE_START_FRAME]
            for name, value in zip(("rho_x", "rho_y", "rho_z", "phi_x", "phi_y", "phi_z"), correction):
                row[f"visual_correction_{name}"] = float(value)
        per_frame.append(row)
    pd.DataFrame(per_frame).to_csv(root / "oracle_state_per_frame.csv", index=False)
    summary = {
        "scope": scope, "mode": mode, "spec": MODES[mode], "metrics": metrics,
        "bias_standardized_increment": standardized,
        "visual_relative_pose_correction": {
            "translation_norm_m": distribution(np.linalg.norm(correction_array[:, :3], axis=1)),
            "rotation_norm_rad": distribution(np.linalg.norm(correction_array[:, 3:], axis=1)),
            "mean_6d": np.mean(correction_array, axis=0),
        },
        "factor_costs": factor_costs,
        "first_edge_bias_update": first_update,
        "convergence": convergence,
        "artifacts": {
            "poses": pose_path, "tensor_map": tensor_path, "diagnostics": diagnostics_path,
            "factor_trace": root / "factor_trace.csv", "state_per_frame": root / "oracle_state_per_frame.csv",
        },
    }
    write_json(root / "summary.json", summary)
    return summary


def percent_improvement(before: float, after: float) -> float | None:
    if abs(before) < 1.0e-15:
        return None
    return float((before - after) / before * 100.0)


def aggregate() -> None:
    all_summaries: dict[str, dict[str, Any]] = {}
    frames = []
    metric_rows = []
    for scope in ("short", "full"):
        scope_values: dict[str, Any] = {}
        for mode in MODES:
            path = mode_root(scope, mode) / "summary.json"
            if not path.exists():
                continue
            scope_values[mode] = json.loads(path.read_text(encoding="utf-8"))
            frames.append(pd.read_csv(mode_root(scope, mode) / "oracle_state_per_frame.csv"))
            summary = scope_values[mode]
            metric_row = {"scope": scope, "mode": mode, **summary["spec"], **summary["metrics"]}
            metric_row.update(
                {
                    "converged_rate": summary["convergence"]["converged_rate"],
                    "iteration_mean": summary["convergence"]["iterations"]["mean"],
                    "visual_translation_correction_mean_m": summary["visual_relative_pose_correction"][
                        "translation_norm_m"
                    ]["mean"],
                    "visual_rotation_correction_mean_rad": summary["visual_relative_pose_correction"][
                        "rotation_norm_rad"
                    ]["mean"],
                    "ba_standardized_increment_norm_mean": summary["bias_standardized_increment"][
                        "ba_increment_norm"
                    ]["mean"],
                    "bg_standardized_increment_norm_mean": summary["bias_standardized_increment"][
                        "bg_increment_norm"
                    ]["mean"],
                }
            )
            metric_rows.append(metric_row)
        if scope_values:
            all_summaries[scope] = scope_values
    visual = {
        scope: {mode: values[mode] for mode in ("V0", "V1", "V2", "V3") if mode in values}
        for scope, values in all_summaries.items()
    }
    write_json(OUT / "circle_visual_oracle_summary.json", visual)
    interaction: dict[str, Any] = {}
    for scope, values in all_summaries.items():
        required = {"V0", "V1", "O3", "O4"}
        if not required.issubset(values):
            continue
        interaction[scope] = {
            "O1": values["V0"], "O2": values["V1"], "O3": values["O3"], "O4": values["O4"],
            "non_additive_comparisons_percent": {},
        }
        for metric in (
            "ate_xy_rmse_m_no_alignment", "translation_rpe_rmse_m", "velocity_truth_rmse_mps",
            "acc_bias_truth_rmse_mps2", "gyro_bias_truth_rmse_radps",
        ):
            o1 = values["V0"]["metrics"][metric]
            o2 = values["V1"]["metrics"][metric]
            o3 = values["O3"]["metrics"][metric]
            o4 = values["O4"]["metrics"][metric]
            interaction[scope]["non_additive_comparisons_percent"][metric] = {
                "O1_to_O2_translation_oracle": percent_improvement(o1, o2),
                "O1_to_O3_fixed_ba": percent_improvement(o1, o3),
                "O2_to_O4_fixed_ba_given_gt_translation": percent_improvement(o2, o4),
                "warning": "Do not add these percentages; translation and ba effects are coupled.",
            }
    write_json(OUT / "circle_translation_bias_interaction_summary.json", interaction)
    if frames:
        pd.concat(frames, ignore_index=True).to_csv(OUT / "circle_visual_oracle_per_frame.csv", index=False)
    if metric_rows:
        pd.DataFrame(metric_rows).to_csv(OUT / "circle_visual_oracle_metric_table.csv", index=False)


def main() -> int:
    args = parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    scopes = ["short", "full"] if args.scope == "both" else [args.scope]
    if args.prepare:
        for scope in scopes:
            prepare(scope)
        print(json.dumps({"prepared": scopes, "output": str(OUT)}, ensure_ascii=False))
        return 0
    if args.run_short:
        prepare("short")
        for mode in MODES:
            command = [
                sys.executable, str(Path(__file__).resolve()), "--run-one", "--scope", "short", "--mode", mode,
            ]
            if args.force:
                command.append("--force")
            subprocess.run(command, cwd=ROOT, check=True)
        aggregate()
        return 0
    if args.run_one:
        if args.scope == "both" or args.mode is None:
            raise ValueError("--run-one requires one --scope and --mode")
        run_one(args.scope, args.mode, force=args.force)
        return 0
    if args.summarize:
        for scope in scopes:
            for mode in MODES:
                if (mode_root(scope, mode) / "run_result").exists():
                    summarize_one(scope, mode)
        aggregate()
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
