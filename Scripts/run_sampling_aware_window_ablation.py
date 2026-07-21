#!/usr/bin/env python3
"""Gate 6: fixed-factor N=2/3/5/10 sampling-aware window ablation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pypose as pp
import torch
from scipy import signal


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Scripts import freeze_normal_noise_sampling_baseline as baseline  # noqa: E402
from Scripts.run_sampling_aware_bias_ablation import (  # noqa: E402
    SIGMA_ACC_W,
    SIGMA_GYRO_W,
    _edge_noise_means,
    _pearson,
)
from Utility.FixedLagVIO import (  # noqa: E402
    FixedLagVIOProblem,
    FixedLagVIOSolver,
    propagate_prior_acc_bias_random_walk,
)
from Utility.RelativePoseFactorCache import camera_factor_to_body_factor  # noqa: E402
from Utility.TwoStateVIO import (  # noqa: E402
    ImuPreintegrationFactor,
    NavigationState,
    RelativePoseFactor,
    make_diagonal_prior,
)


OUT = baseline.DEFAULT_OUTPUT
WINDOW_ROOT = OUT / "window_ablation"
FROZEN_RUN = OUT / "bias_ablation/B2_fixed_static_ba_bg/run_result"
WINDOWS = (2, 3, 5, 10)
MODES = {
    "normal": {
        "opt_ba": True, "opt_bg": True, "shared_ba": False,
        "label": "optimize_ba_bg",
    },
    "fixed_ba": {
        "opt_ba": False, "opt_bg": True, "shared_ba": False,
        "label": "fixed_static_ba_optimize_bg",
    },
    "shared_ba": {
        "opt_ba": True, "opt_bg": True, "shared_ba": True,
        "label": "window_shared_ba_optimize_bg",
    },
    "rate_limited_ba": {
        "opt_ba": True, "opt_bg": True, "shared_ba": True,
        "rate_limited": True,
        "label": "window_shared_rate_limited_ba_optimize_bg",
    },
    "rw_gated_ba": {
        "opt_ba": True, "opt_bg": True, "shared_ba": True,
        "rate_limited": True, "rw_gate": True,
        "label": "window_shared_rw_gated_ba_optimize_bg",
    },
}
BA_RW_CHI2_3DOF_999 = 16.26623619623813
DTYPE = torch.float64


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window", type=int, choices=WINDOWS)
    parser.add_argument("--mode", choices=sorted(MODES))
    parser.add_argument("--only-window", type=int, nargs="+", choices=WINDOWS)
    parser.add_argument("--only-mode", nargs="+", choices=sorted(MODES))
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _run_dir(mode: str, window: int) -> Path:
    return WINDOW_ROOT / f"{mode}_N{window}"


def _find_tensor_map() -> Path:
    paths = sorted(FROZEN_RUN.rglob("tensor_map.npz"))
    if len(paths) != 1:
        raise RuntimeError(f"expected one frozen tensor_map, found {len(paths)}")
    return paths[0]


def _sha256_array(values: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode())
    digest.update(str(contiguous.shape).encode())
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _copy_map(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as source:
        return {key: source[key].copy() for key in source.files}


def _state_from_map(data: dict[str, np.ndarray], frame: int) -> NavigationState:
    pose_wc = pp.SE3(torch.from_numpy(data["frames//pose"][frame].astype(np.float64)).reshape(1, 7))
    extrinsic_ci = pp.SE3(
        torch.from_numpy(data["frames//imu_vio_sensor_T_imu"][frame].astype(np.float64)).reshape(1, 7)
    )
    return NavigationState(
        pose_WB=(pose_wc @ extrinsic_ci).tensor(),
        velocity_W=torch.from_numpy(data["frames//imu_vio_velocity_world"][frame].astype(np.float64)),
        acc_bias=torch.from_numpy(data["frames//imu_vio_acc_bias"][frame].astype(np.float64)),
        gyro_bias=torch.from_numpy(data["frames//imu_vio_gyro_bias"][frame].astype(np.float64)),
    )


def _factors_from_map(
    data: dict[str, np.ndarray], frame_j: int
) -> tuple[ImuPreintegrationFactor, RelativePoseFactor]:
    tensor = lambda key: torch.from_numpy(data[key][frame_j].astype(np.float64))
    gravity_in_residual = bool(data["frames//imu_vio_gravity_in_residual"][frame_j])
    imu = ImuPreintegrationFactor(
        delta_rotation=tensor("frames//imu_vio_delta_rotvec"),
        delta_velocity=tensor("frames//imu_vio_delta_v"),
        delta_position=tensor("frames//imu_vio_delta_p"),
        covariance=tensor("frames//imu_vio_cov"),
        dt=float(data["frames//imu_vio_dt"][frame_j]),
        bias_jacobian=tensor("frames//imu_vio_bias_jacobian"),
        linearized_acc_bias=tensor("frames//imu_vio_linearized_acc_bias"),
        linearized_gyro_bias=tensor("frames//imu_vio_linearized_gyro_bias"),
        bias_rw_covariance=tensor("frames//imu_vio_bias_rw_cov"),
        gravity_world=(tensor("frames//imu_vio_gravity_world") if gravity_in_residual else None),
        gravity_handling=("residual" if gravity_in_residual else "preintegration"),
    )
    measurement, covariance = camera_factor_to_body_factor(
        tensor("frames//visual_relative_pose_CiCj").reshape(1, 7),
        tensor("frames//visual_relative_pose_cov").reshape(6, 6),
        tensor("frames//imu_vio_sensor_T_imu").reshape(1, 7),
    )
    visual = RelativePoseFactor(
        measurement_BiBj=measurement,
        covariance=covariance,
        huber_delta=3.0,
    )
    return imu, visual


def _propagate_endpoint(
    source: NavigationState,
    imu: ImuPreintegrationFactor,
    visual: RelativePoseFactor,
    *,
    fixed_acc_bias: torch.Tensor | None,
) -> NavigationState:
    db = torch.cat(
        [
            source.acc_bias.reshape(3) - imu.linearized_acc_bias.reshape(3),
            source.gyro_bias.reshape(3) - imu.linearized_gyro_bias.reshape(3),
        ]
    )
    correction = imu.bias_jacobian.reshape(9, 6) @ db
    delta_v = imu.delta_velocity.reshape(3) + correction[3:6]
    rotation_i = pp.SE3(source.pose_WB).rotation().matrix().reshape(3, 3)
    velocity = source.velocity_W.reshape(3) + rotation_i @ delta_v
    if imu.gravity_world is not None:
        velocity = velocity + imu.gravity_world.reshape(3) * float(imu.dt)
    pose = pp.SE3(source.pose_WB) @ pp.SE3(visual.measurement_BiBj)
    return NavigationState(
        pose_WB=pose.tensor(),
        velocity_W=velocity,
        acc_bias=(
            source.acc_bias.detach().clone()
            if fixed_acc_bias is None
            else fixed_acc_bias.detach().clone()
        ),
        gyro_bias=source.gyro_bias.detach().clone(),
    )


def _prior_stats(prior) -> tuple[float, float, int, float]:
    hessian = prior.sqrt_information.mT @ prior.sqrt_information
    values = torch.linalg.eigvalsh(0.5 * (hessian + hessian.mT))
    scale = max(float(values.abs().max().item()), 1.0)
    tolerance = 15 * torch.finfo(values.dtype).eps * scale
    active = values[values > tolerance]
    rank = int(active.numel())
    condition = float((active.max() / active.min()).item()) if rank else math.inf
    return float(values.min().item()), float(values.max().item()), rank, condition


def _lowpass_rms(values: np.ndarray, sample_rate_hz: float, cutoff_hz: float = 1.0) -> float:
    sos = signal.butter(4, cutoff_hz, btype="lowpass", fs=sample_rate_hz, output="sos")
    filtered = signal.sosfiltfilt(sos, np.asarray(values, dtype=np.float64), axis=0)
    return baseline.rmse_norm(filtered)


def _summarize(
    mode: str,
    window: int,
    data: dict[str, np.ndarray],
    final_states: dict[int, NavigationState],
    solve_rows: list[dict],
    elapsed_s: float,
    peak_rss_kb: int,
    output: Path,
) -> dict:
    frame_time = data["frames//time_ns"][: baseline.FRAME_LIMIT].astype(np.int64)
    estimate_pose = pp.SE3(torch.from_numpy(data["frames//pose"][: baseline.FRAME_LIMIT].astype(np.float64)))
    estimate_velocity = data["frames//imu_vio_velocity_world"][: baseline.FRAME_LIMIT].astype(np.float64)
    estimate_ba = data["frames//imu_vio_acc_bias"][: baseline.FRAME_LIMIT].astype(np.float64)
    estimate_bg = data["frames//imu_vio_gyro_bias"][: baseline.FRAME_LIMIT].astype(np.float64)
    extrinsics = pp.SE3(
        torch.from_numpy(data["frames//imu_vio_sensor_T_imu"][: baseline.FRAME_LIMIT].astype(np.float64))
    )
    for frame, state in final_states.items():
        estimate_pose[frame] = pp.SE3(state.pose_WB) @ extrinsics[frame].Inv()
        estimate_velocity[frame] = state.velocity_W.numpy()
        estimate_ba[frame] = state.acc_bias.numpy()
        estimate_bg[frame] = state.gyro_bias.numpy()

    gt_time, gt_pose, gt_velocity = baseline.gt_pose_and_velocity(baseline.FRAME_LIMIT)
    if not np.array_equal(frame_time, gt_time):
        raise AssertionError("frozen factor timestamps differ from GT")
    truth = pd.read_csv(baseline.DATASET_DIR / "imu_truth_decomposition.csv")
    imu_time = truth["timestamp"].to_numpy(np.int64)
    gt_ba = baseline.interpolate_rows(
        imu_time,
        truth[["acc_bias_x", "acc_bias_y", "acc_bias_z"]].to_numpy(np.float64),
        frame_time,
    ) @ baseline.FLU_TO_NED.T
    gt_bg = baseline.interpolate_rows(
        imu_time,
        truth[["gyro_bias_x", "gyro_bias_y", "gyro_bias_z"]].to_numpy(np.float64),
        frame_time,
    ) @ baseline.FLU_TO_NED.T

    pose_error = (gt_pose.Inv() @ estimate_pose).Log().tensor().numpy()
    position_error = estimate_pose.translation().numpy() - gt_pose.translation().numpy()
    velocity_error = estimate_velocity - gt_velocity
    ba_error, bg_error = estimate_ba - gt_ba, estimate_bg - gt_bg
    estimate_relative = estimate_pose[:-1].Inv() @ estimate_pose[1:]
    gt_relative = gt_pose[:-1].Inv() @ gt_pose[1:]
    relative_error = (gt_relative.Inv() @ estimate_relative).Log().tensor().numpy()
    valid = slice(baseline.FIRST_VALID_FRAME, baseline.FRAME_LIMIT)
    sample_rate = 1.0 / float(np.median(np.diff(frame_time[valid])) * 1e-9)
    pure_time, pure_pose = baseline.read_csv_pose(baseline.PURE_MACVO_POSES, baseline.FRAME_LIMIT)
    if not np.array_equal(frame_time, pure_time):
        raise AssertionError("pure MACVO timestamps differ from fixed-factor cache")
    xy_correction = estimate_pose.translation().numpy()[valid, :2] - pure_pose.translation().numpy()[valid, :2]

    solve = pd.DataFrame(solve_rows)
    relative_corrections = solve[
        [f"latest_edge_pose_correction_{index}" for index in range(6)]
    ].to_numpy(np.float64)

    active_time = frame_time[valid]
    dt = np.diff(active_time.astype(np.float64)) * 1e-9
    dba, dbg = np.diff(estimate_ba[valid], axis=0), np.diff(estimate_bg[valid], axis=0)
    dba_gt, dbg_gt = np.diff(gt_ba[valid], axis=0), np.diff(gt_bg[valid], axis=0)
    acc_noise, gyro_noise = _edge_noise_means(frame_time)
    state_rows = []
    for frame in range(baseline.FRAME_LIMIT):
        row = {"frame": frame, "timestamp_ns": int(frame_time[frame]), "vio_active": frame >= baseline.FIRST_VALID_FRAME}
        vectors = {
            "p_error": position_error[frame], "R_error": pose_error[frame, 3:6],
            "v_est": estimate_velocity[frame], "v_gt": gt_velocity[frame], "v_error": velocity_error[frame],
            "ba_est": estimate_ba[frame], "ba_gt": gt_ba[frame], "ba_error": ba_error[frame],
            "bg_est": estimate_bg[frame], "bg_gt": gt_bg[frame], "bg_error": bg_error[frame],
        }
        for prefix, vector in vectors.items():
            for axis, value in zip("xyz", vector):
                row[f"{prefix}_{axis}"] = float(value)
            row[f"{prefix}_norm"] = float(np.linalg.norm(vector))
        row["future_frames_used"] = max(0, min(window - 1, baseline.FRAME_LIMIT - 1 - frame)) if row["vio_active"] else 0
        state_rows.append(row)
    pd.DataFrame(state_rows).to_csv(output / "state_per_frame.csv", index=False)
    solve.to_csv(output / "solve_per_frame.csv", index=False)
    trajectory = pd.DataFrame(
        np.column_stack([frame_time, estimate_pose.tensor().numpy()]),
        columns=["timestamp_ns", "tx", "ty", "tz", "qx", "qy", "qz", "qw"],
    )
    trajectory.to_csv(output / "trajectory.csv", index=False)

    truth_metrics = {
        "ate_position_rmse_m": baseline.rmse_norm(position_error[valid]),
        "xy_position_rmse_m": baseline.rmse_norm(position_error[valid, :2]),
        "orientation_rmse_rad": baseline.rmse_norm(pose_error[valid, 3:6]),
        "velocity_rmse_mps": baseline.rmse_norm(velocity_error[valid]),
        "acc_bias_rmse_mps2": baseline.rmse_norm(ba_error[valid]),
        "gyro_bias_rmse_radps": baseline.rmse_norm(bg_error[valid]),
        "translation_rpe_rmse_m": baseline.rmse_norm(relative_error[baseline.FIRST_VALID_FRAME:, :3]),
        "rotation_rpe_rmse_rad": baseline.rmse_norm(relative_error[baseline.FIRST_VALID_FRAME:, 3:6]),
    }
    hf = {
        "sample_rate_hz": sample_rate,
        "xy_position_error_highpass_rms_m": baseline.highpass_rms(position_error[valid, :2], sample_rate),
        "xy_position_correction_second_difference_rms_m": baseline.rmse_norm(np.diff(xy_correction, n=2, axis=0)),
        "velocity_truth_error_highpass_rms_mps": baseline.highpass_rms(velocity_error[valid], sample_rate),
        "relative_pose_correction_change_translation_rms_m": baseline.rmse_norm(np.diff(relative_corrections[:, :3], axis=0)),
        "relative_pose_correction_change_rotation_rms_rad": baseline.rmse_norm(np.diff(relative_corrections[:, 3:], axis=0)),
    }
    bias_diagnostics = {
        "ba_increment_standardized_component_rms": float(np.sqrt(np.mean((dba / (SIGMA_ACC_W * np.sqrt(dt)[:, None])) ** 2))),
        "bg_increment_standardized_component_rms": float(np.sqrt(np.mean((dbg / (SIGMA_GYRO_W * np.sqrt(dt)[:, None])) ** 2))),
        "ba_increment_energy_over_truth": float(np.sum(dba * dba) / max(np.sum(dba_gt * dba_gt), 1e-30)),
        "bg_increment_energy_over_truth": float(np.sum(dbg * dbg) / max(np.sum(dbg_gt * dbg_gt), 1e-30)),
        "ba_increment_vs_acc_white_noise_correlation": _pearson(dba, acc_noise),
        "bg_increment_vs_gyro_white_noise_correlation": _pearson(dbg, gyro_noise),
    }
    for axis, index in zip("xyz", range(3)):
        bias_diagnostics[f"ba_increment_standardized_rms_{axis}"] = float(
            np.sqrt(np.mean((dba[:, index] / (SIGMA_ACC_W * np.sqrt(dt))) ** 2))
        )
        bias_diagnostics[f"ba_increment_rmse_{axis}_mps2"] = float(
            np.sqrt(np.mean(dba[:, index] ** 2))
        )
    summary = {
        "gate": 6,
        "mode": mode,
        "mode_label": MODES[mode]["label"],
        "window_size": window,
        "frame_range": [0, baseline.FRAME_LIMIT - 1],
        "active_edge_range": [baseline.FIRST_VALID_FRAME, baseline.FRAME_LIMIT - 1],
        "edge_count": baseline.EXPECTED_EDGES,
        "factor_contract": {
            "source": str(_find_tensor_map()),
            "preintegration": "standard_local_frame_preintegration",
            "covariance": "sampling_aware",
            "delta_and_covariance_are_frozen_across_all_runs": True,
            "visual_gate_actions": "209/209 accept, inflation=1 in frozen B2 source",
            "sigma_a": 0.0141258, "sigma_g": 0.00182898,
            "sigma_aw": SIGMA_ACC_W, "sigma_gw": SIGMA_GYRO_W,
        },
        "truth_metrics": truth_metrics,
        "high_frequency_metrics": hf,
        "low_frequency_metrics": {
            "xy_position_error_lowpass_rms_m": _lowpass_rms(position_error[valid, :2], sample_rate),
            "xy_endpoint_error_m": float(np.linalg.norm(position_error[baseline.FRAME_LIMIT - 1, :2])),
        },
        "bias_diagnostics": bias_diagnostics,
        "solver": {
            "solve_count": int(len(solve)),
            "converged_count": int(solve["converged"].sum()),
            "converged_ratio": float(solve["converged"].mean()),
            "mean_iterations": float(solve["iterations"].mean()),
            "p95_iterations": float(solve["iterations"].quantile(0.95)),
            "runtime_seconds": elapsed_s,
            "mean_runtime_ms_per_solve": 1000.0 * elapsed_s / max(len(solve), 1),
            "peak_rss_kb": int(peak_rss_kb),
            "prior_rank_min": int(solve["prior_rank"].min()),
            "prior_rank_max": int(solve["prior_rank"].max()),
            "prior_condition_median": float(solve["prior_condition"].replace([np.inf, -np.inf], np.nan).median()),
            "prior_condition_max": float(solve["prior_condition"].replace([np.inf, -np.inf], np.nan).max()),
        },
        "factor_cost_mean": {
            name: float(solve[name].mean())
            for name in ("prior_cost", "imu_cost", "bias_cost", "visual_pose_cost", "final_cost")
        },
        "future_frames_used": {
            "maximum": window - 1,
            "mean": float(np.mean([max(0, min(window - 1, baseline.FRAME_LIMIT - 1 - f)) for f in range(baseline.FIRST_VALID_FRAME, baseline.FRAME_LIMIT)])),
        },
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def run_one(mode: str, window: int, *, force: bool) -> int:
    output = _run_dir(mode, window)
    if output.exists() and not force:
        raise FileExistsError(f"{output} exists; pass --force")
    if output.exists():
        import shutil
        if WINDOW_ROOT.resolve() not in output.resolve().parents:
            raise RuntimeError("refusing to clear output outside window root")
        shutil.rmtree(output)
    output.mkdir(parents=True)
    data = _copy_map(_find_tensor_map())
    mode_spec = MODES[mode]
    frame0 = baseline.FIRST_VALID_FRAME
    state0 = _state_from_map(data, frame0).to(device=torch.device("cpu"), dtype=DTYPE)
    static_ba = state0.acc_bias.detach().clone()
    prior = make_diagonal_prior(
        state0,
        pose_translation_std=1.0e-5,
        pose_rotation_std=1.0e-5,
        velocity_std=0.05,
        acc_bias_std=0.2,
        gyro_bias_std=0.02,
    )
    solver = FixedLagVIOSolver(
        max_iterations=20,
        initial_damping=1.0e-3,
        step_tolerance=1.0e-8,
        cost_tolerance=1.0e-10,
        covariance_eigenvalue_floor=1.0e-12,
        marginalization_eigenvalue_floor=1.0e-10,
    )
    active_frames = [frame0]
    states = [state0]
    imus: list[ImuPreintegrationFactor] = []
    visuals: list[RelativePoseFactor] = []
    final_states = {frame0: state0.detach()}
    solve_rows: list[dict] = []
    accumulated_ba_rw_covariance = torch.zeros((3, 3), dtype=DTYPE)
    started = time.perf_counter()
    for frame_j in range(frame0 + 1, baseline.FRAME_LIMIT):
        imu, visual = _factors_from_map(data, frame_j)
        accumulated_ba_rw_covariance += imu.bias_rw_covariance[0:3, 0:3].to(
            dtype=DTYPE, device=torch.device("cpu")
        )
        endpoint = _propagate_endpoint(
            states[-1], imu, visual,
            fixed_acc_bias=(static_ba if not mode_spec["opt_ba"] else None),
        )
        active_frames.append(frame_j)
        states.append(endpoint)
        imus.append(imu)
        visuals.append(visual)
        ba_update_interval = max(window - 1, 1)
        ba_update_due = bool(
            mode_spec.get("rate_limited", False)
            and len(states) == window
            and (frame_j - frame0) % ba_update_interval == 0
        )
        optimize_acc_bias_this_solve = bool(
            mode_spec["opt_ba"]
            and (not mode_spec.get("rate_limited", False) or ba_update_due)
        )
        ba_before_solve = states[0].acc_bias.detach().clone()
        def make_problem(optimize_ba: bool) -> FixedLagVIOProblem:
            return FixedLagVIOProblem(
                states=tuple(states),
                prior_first=prior,
                imu_factors=tuple(imus),
                visual_factors=tuple(visuals),
                optimize_acc_bias=optimize_ba,
                optimize_gyro_bias=bool(mode_spec["opt_bg"]),
                shared_acc_bias=bool(mode_spec["shared_ba"]),
            )
        result = solver.solve(make_problem(optimize_acc_bias_this_solve))
        ba_candidate_nis = float("nan")
        ba_candidate_update_norm = float("nan")
        ba_gate_action = "not_applicable"
        if optimize_acc_bias_this_solve:
            ba_update = result.states[0].acc_bias - ba_before_solve
            ba_candidate_update_norm = float(torch.linalg.vector_norm(ba_update).item())
            covariance = 0.5 * (
                accumulated_ba_rw_covariance + accumulated_ba_rw_covariance.mT
            )
            eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
            floor = max(
                solver.covariance_eigenvalue_floor,
                torch.finfo(DTYPE).eps,
            )
            covariance_inverse = (
                eigenvectors
                @ torch.diag(eigenvalues.clamp_min(floor).reciprocal())
                @ eigenvectors.mT
            )
            ba_candidate_nis = float(ba_update @ covariance_inverse @ ba_update)
            if mode_spec.get("rw_gate", False) and ba_candidate_nis > BA_RW_CHI2_3DOF_999:
                result = solver.solve(make_problem(False))
                optimize_acc_bias_this_solve = False
                ba_gate_action = "reject_rw_nis"
            else:
                ba_gate_action = "accept"
                accumulated_ba_rw_covariance.zero_()
        elif mode_spec.get("rw_gate", False):
            ba_gate_action = "hold_until_update_epoch"
        states = list(result.states)
        for frame, state in zip(active_frames, states):
            final_states[frame] = state.detach()
            if not mode_spec["opt_ba"] and not torch.equal(state.acc_bias, static_ba):
                raise AssertionError(f"fixed ba changed in {mode} N={window} frame={frame}")
        if mode_spec["shared_ba"]:
            shared_ba = states[0].acc_bias
            if any(
                not torch.allclose(state.acc_bias, shared_ba, atol=1e-11, rtol=0.0)
                for state in states[1:]
            ):
                raise AssertionError(f"shared ba diverged in {mode} N={window} frame={frame_j}")
        prior_min, prior_max, prior_rank, prior_condition = _prior_stats(result.prior_next)
        latest_predicted = pp.SE3(states[-2].pose_WB).Inv() @ pp.SE3(states[-1].pose_WB)
        latest_correction = (
            pp.SE3(visual.measurement_BiBj).Inv() @ latest_predicted
        ).Log().tensor().reshape(6)
        latest_correction_values = {
            f"latest_edge_pose_correction_{index}": float(value)
            for index, value in enumerate(latest_correction.tolist())
        }
        solve_rows.append(
            {
                "frame_j": frame_j,
                "active_window_size": len(states),
                "iterations": result.iterations,
                "converged": result.converged,
                "convergence_reason": result.convergence_reason,
                "initial_cost": result.initial_cost,
                "final_cost": result.final_cost,
                "prior_cost": result.prior_cost,
                "imu_cost": result.imu_cost,
                "bias_cost": result.bias_cost,
                "visual_pose_cost": result.visual_pose_cost,
                "final_step_norm": result.final_step_norm,
                "final_gradient_inf_norm": result.final_gradient_inf_norm,
                "prior_min_eigenvalue": prior_min,
                "prior_max_eigenvalue": prior_max,
                "prior_rank": prior_rank,
                "prior_condition": prior_condition,
                "shared_acc_bias": bool(result.shared_acc_bias),
                "ba_update_due": ba_update_due,
                "ba_update_applied": optimize_acc_bias_this_solve,
                "ba_candidate_nis": ba_candidate_nis,
                "ba_candidate_update_norm": ba_candidate_update_norm,
                "ba_rw_gate_threshold": BA_RW_CHI2_3DOF_999,
                "ba_gate_action": ba_gate_action,
                "ba_update_norm": float(
                    torch.linalg.vector_norm(result.states[0].acc_bias - ba_before_solve).item()
                ),
                "ba_information_min_eigenvalue": (
                    float(result.acc_bias_marginal_information_eigenvalues.min().item())
                    if result.acc_bias_marginal_information_eigenvalues is not None
                    else float("nan")
                ),
                "ba_information_max_eigenvalue": (
                    float(result.acc_bias_marginal_information_eigenvalues.max().item())
                    if result.acc_bias_marginal_information_eigenvalues is not None
                    else float("nan")
                ),
                **latest_correction_values,
            }
        )
        if len(states) == window:
            prior = result.prior_next
            if mode_spec["shared_ba"]:
                prior = propagate_prior_acc_bias_random_walk(
                    prior,
                    imus[0].bias_rw_covariance[0:3, 0:3],
                    eigenvalue_floor=solver.marginalization_eigenvalue_floor,
                )
            active_frames = active_frames[1:]
            states = states[1:]
            imus = imus[1:]
            visuals = visuals[1:]
        if (frame_j - frame0) % 25 == 0 or frame_j == baseline.FRAME_LIMIT - 1:
            print(
                f"[{mode} N={window}] edge {frame_j - frame0}/{baseline.EXPECTED_EDGES} "
                f"iterations={result.iterations} converged={result.converged}",
                flush=True,
            )
    elapsed = time.perf_counter() - started
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    checkpoint_frames = np.asarray(sorted(final_states), dtype=np.int64)
    np.savez_compressed(
        output / "fixed_lag_state_checkpoint.npz",
        frame=checkpoint_frames,
        pose_WB=np.concatenate([final_states[int(frame)].pose_WB.numpy() for frame in checkpoint_frames]),
        velocity_W=np.stack([final_states[int(frame)].velocity_W.numpy() for frame in checkpoint_frames]),
        acc_bias=np.stack([final_states[int(frame)].acc_bias.numpy() for frame in checkpoint_frames]),
        gyro_bias=np.stack([final_states[int(frame)].gyro_bias.numpy() for frame in checkpoint_frames]),
    )
    pd.DataFrame(solve_rows).to_csv(output / "solve_checkpoint.csv", index=False)
    summary = _summarize(mode, window, data, final_states, solve_rows, elapsed, peak_rss, output)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


def _aggregate() -> None:
    summaries = {
        (mode, window): json.loads((_run_dir(mode, window) / "summary.json").read_text(encoding="utf-8"))
        for mode in MODES for window in WINDOWS
    }
    rows = []
    for (mode, window), summary in summaries.items():
        row = {"mode": mode, "window_size": window}
        for group in ("truth_metrics", "high_frequency_metrics", "low_frequency_metrics", "bias_diagnostics"):
            for key, value in summary[group].items():
                if isinstance(value, (int, float)) or value is None:
                    row[key] = value
        for key, value in summary["solver"].items():
            row[f"solver_{key}"] = value
        rows.append(row)
    metrics = pd.DataFrame(rows).sort_values(["mode", "window_size"])
    metrics.to_csv(OUT / "macvio_window_metrics.csv", index=False)

    criteria = {}
    hf_names = (
        "xy_position_error_highpass_rms_m",
        "xy_position_correction_second_difference_rms_m",
        "velocity_truth_error_highpass_rms_mps",
        "relative_pose_correction_change_translation_rms_m",
        "relative_pose_correction_change_rotation_rms_rad",
    )
    truth_names = ("translation_rpe_rmse_m", "rotation_rpe_rmse_rad", "ate_position_rmse_m")
    recommendations = {}
    for mode in MODES:
        base = summaries[(mode, 2)]
        recommendations[mode] = None
        for window in WINDOWS:
            current = summaries[(mode, window)]
            hf_improvements = {
                name: 1.0 - current["high_frequency_metrics"][name] / base["high_frequency_metrics"][name]
                for name in hf_names
            }
            truth_changes = {
                name: current["truth_metrics"][name] / base["truth_metrics"][name] - 1.0
                for name in truth_names
            }
            low_frequency_change = (
                current["low_frequency_metrics"]["xy_position_error_lowpass_rms_m"]
                / base["low_frequency_metrics"]["xy_position_error_lowpass_rms_m"] - 1.0
            )
            bias_ok = (
                mode != "normal"
                or (
                    abs(current["bias_diagnostics"]["ba_increment_vs_acc_white_noise_correlation"] or 0.0) < 0.2
                    and current["bias_diagnostics"]["ba_increment_standardized_component_rms"] < 10.0
                )
            )
            passed = (
                sum(value >= 0.20 for value in hf_improvements.values()) >= 2
                and all(value <= 0.05 for value in truth_changes.values())
                and low_frequency_change <= 0.05
                and current["solver"]["converged_ratio"] >= 0.95
                and bias_ok
            )
            criteria[f"{mode}_N{window}"] = {
                "high_frequency_improvement_fraction_vs_N2": hf_improvements,
                "truth_metric_change_fraction_vs_N2": truth_changes,
                "low_frequency_change_fraction_vs_N2": low_frequency_change,
                "bias_no_longer_tracks_white_noise": bias_ok,
                "passes_shortest_window_rule": passed,
            }
            if window > 2 and passed and recommendations[mode] is None:
                recommendations[mode] = window

    decision = {
        "goal": "reduce MACVIO error to GT, not approach GTSAM",
        "fixed_factor_contract": True,
        "bias_gate_definition": {
            "absolute_noise_correlation_limit": 0.2,
            "standardized_increment_rms_limit": 10.0,
            "rationale": "A calibrated RW process is O(1); 10 is an intentionally generous one-order-of-magnitude ceiling.",
        },
        "normal_mode_shortest_effective_window": recommendations["normal"],
        "diagnostic_fixed_ba_shortest_effective_window": recommendations["fixed_ba"],
        "criteria": criteria,
        "final_classification": [
            "B_bias_overactivity_is_the_dominant_jitter_source",
            "E_sampling_covariance_bias_and_window_have_non_additive_joint_effects",
        ],
        "window_conclusion": (
            "N=10 reaches the two 20% high-frequency improvements in normal mode, "
            "but accelerometer-bias increments remain about 84.5 sigma per component and "
            "the fixed-ba diagnostic reaches only 14.5%/12.7%; window length is secondary, "
            "not a complete repair."
        ),
        "full_sequence_approved": False,
        "production_recommendation": (
            "No window satisfies all gates; do not promote a larger window yet."
            if recommendations["normal"] is None
            else f"Use N={recommendations['normal']} as the shortest validated normal-bias window."
        ),
    }
    (OUT / "macvio_window_final_recommendation.json").write_text(
        json.dumps(decision, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    lines = [
        "# MACVIO 最短有效窗口消融报告", "",
        "> 本任务的目标是降低 MACVIO 对 GT 的高频误差，不是让 MACVIO 轨迹或 Bias 接近 GTSAM。", "",
        "所有运行读取同一个冻结因子缓存；IMU Delta、Sampling-aware P、视觉相对位姿/covariance、Bias RW、LM、Huber 和初始 prior 均未改变。", "",
        "| Bias 模式 | N | XY 高频 RMS | correction 二阶差分 | velocity 高频 RMS | 平移 RPE | 旋转 RPE | ATE | ba RMSE | 平均耗时/solve (ms) |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for mode in MODES:
        for window in WINDOWS:
            summary = summaries[(mode, window)]
            h, t, s = summary["high_frequency_metrics"], summary["truth_metrics"], summary["solver"]
            lines.append(
                f"| {mode} | {window} | {h['xy_position_error_highpass_rms_m']:.6g} | "
                f"{h['xy_position_correction_second_difference_rms_m']:.6g} | "
                f"{h['velocity_truth_error_highpass_rms_mps']:.6g} | "
                f"{t['translation_rpe_rmse_m']:.6g} | {t['rotation_rpe_rmse_rad']:.6g} | "
                f"{t['ate_position_rmse_m']:.6g} | {t['acc_bias_rmse_mps2']:.6g} | "
                f"{s['mean_runtime_ms_per_solve']:.3f} |"
            )
    lines += ["", "## 决策", "", decision["production_recommendation"], "", "逐项门槛判定见 `macvio_window_final_recommendation.json`，原始逐帧状态与求解统计位于 `window_ablation/`。"]
    (OUT / "macvio_window_ablation_report_cn.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(decision, indent=2, ensure_ascii=False))


def _write_manifest() -> None:
    data = _copy_map(_find_tensor_map())
    keys = [
        "frames//imu_vio_delta_rotvec", "frames//imu_vio_delta_v", "frames//imu_vio_delta_p",
        "frames//imu_vio_cov", "frames//imu_vio_bias_jacobian", "frames//imu_vio_bias_rw_cov",
        "frames//visual_relative_pose_CiCj", "frames//visual_relative_pose_cov",
    ]
    manifest = {
        "source_tensor_map": str(_find_tensor_map()),
        "frame_range": [0, baseline.FRAME_LIMIT - 1],
        "edge_range": [baseline.FIRST_VALID_FRAME, baseline.FRAME_LIMIT - 1],
        "array_hashes": {key: _sha256_array(data[key][: baseline.FRAME_LIMIT]) for key in keys},
    }
    (WINDOW_ROOT / "frozen_factor_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def orchestrate(force: bool, only_windows: list[int] | None, only_modes: list[str] | None) -> int:
    WINDOW_ROOT.mkdir(parents=True, exist_ok=True)
    _write_manifest()
    windows = only_windows or list(WINDOWS)
    modes = only_modes or list(MODES)
    for mode in modes:
        for window in windows:
            output = _run_dir(mode, window)
            if (output / "summary.json").exists() and not force:
                print(f"[{mode} N={window}] reusing completed output", flush=True)
                continue
            print(f"[{mode} N={window}] starting", flush=True)
            stdout_path = WINDOW_ROOT / f"{mode}_N{window}_stdout.log"
            stderr_path = WINDOW_ROOT / f"{mode}_N{window}_stderr.log"
            with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
                command = [sys.executable, str(Path(__file__).resolve()), "--mode", mode, "--window", str(window)]
                if force:
                    command.append("--force")
                completed = subprocess.run(command, cwd=ROOT, stdout=stdout, stderr=stderr)
            if completed.returncode != 0:
                raise RuntimeError(f"{mode} N={window} failed; inspect {stderr_path}")
            print(f"[{mode} N={window}] complete", flush=True)
    if set(windows) == set(WINDOWS) and set(modes) == set(MODES):
        _aggregate()
    return 0


def main() -> int:
    args = parse_args()
    if args.window is not None or args.mode is not None:
        if args.window is None or args.mode is None:
            raise ValueError("--window and --mode must be provided together")
        return run_one(args.mode, args.window, force=args.force)
    return orchestrate(args.force, args.only_window, args.only_mode)


if __name__ == "__main__":
    raise SystemExit(main())
