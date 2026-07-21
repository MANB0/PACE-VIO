#!/usr/bin/env python3
"""Gate 5: N=2/3/5 fixed-lag comparison using frozen short-circle V3 factors."""

from __future__ import annotations

import argparse
import json
import math
import resource
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pypose as pp
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import Scripts.run_circle_translation_oracles as oracle  # noqa: E402
import Scripts.run_v3_backend_information_oracles as information  # noqa: E402
from Utility.FixedLagVIO import FixedLagVIOProblem, FixedLagVIOSolver  # noqa: E402
from Utility.RelativePoseFactorCache import camera_factor_to_body_factor  # noqa: E402
from Utility.TwoStateVIO import (  # noqa: E402
    ImuPreintegrationFactor,
    NavigationState,
    RelativePoseFactor,
    make_diagonal_prior,
)


OUT = ROOT / "analysis_circle_v3_backend_pointlevel_20260716/window_ablation"
SOURCE = information.OUT / "oracles/short/C0"
FRAME_LIMIT = 391
FIRST_FRAME = oracle.ACTIVE_START_FRAME
WINDOWS = (2, 3, 5)
DTYPE = torch.float64


def latest(root: Path, name: str) -> Path:
    matches = sorted(root.rglob(name))
    if len(matches) != 1:
        raise RuntimeError(f"expected one {name} below {root}, found {len(matches)}")
    return matches[0]


def load_map() -> dict[str, np.ndarray]:
    path = latest(SOURCE / "run_result", "tensor_map.npz")
    with np.load(path, allow_pickle=False) as stream:
        return {name: stream[name].copy() for name in stream.files}


def state_from_map(data: dict[str, np.ndarray], frame: int) -> NavigationState:
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


def factor_from_map(
    data: dict[str, np.ndarray], frame_j: int, gate_inflation: dict[int, float]
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
        gravity_world=tensor("frames//imu_vio_gravity_world") if gravity_in_residual else None,
        gravity_handling="residual" if gravity_in_residual else "preintegration",
    )
    measurement, covariance = camera_factor_to_body_factor(
        tensor("frames//visual_relative_pose_CiCj").reshape(1, 7),
        tensor("frames//visual_relative_pose_cov").reshape(6, 6),
        tensor("frames//imu_vio_sensor_T_imu").reshape(1, 7),
    )
    covariance = covariance * float(gate_inflation.get(frame_j - 1, 1.0))
    return imu, RelativePoseFactor(
        measurement_BiBj=measurement,
        covariance=covariance,
        huber_delta=3.0,
    )


def propagate_endpoint(
    source: NavigationState, imu: ImuPreintegrationFactor, visual: RelativePoseFactor
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
        acc_bias=source.acc_bias.detach().clone(),
        gyro_bias=source.gyro_bias.detach().clone(),
    )


def rmse(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    return float(np.sqrt(np.mean(np.sum(values * values, axis=1))))


def prior_stats(prior) -> dict[str, float | int]:
    hessian = prior.sqrt_information.mT @ prior.sqrt_information
    values = torch.linalg.eigvalsh(0.5 * (hessian + hessian.mT))
    scale = max(float(values.abs().max()), 1.0)
    threshold = max(1.0e-10, torch.finfo(values.dtype).eps * scale)
    positive = values[values > threshold]
    return {
        "prior_rank": int(positive.numel()),
        "prior_min_positive_eigenvalue": float(positive.min()) if positive.numel() else 0.0,
        "prior_max_eigenvalue": float(values.max()),
        "prior_condition": float(positive.max() / positive.min()) if positive.numel() else math.inf,
    }


def truth_states(data: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ref, truth_pose = oracle.load_truth(oracle.DEFAULT_DATASET)
    time_ns = ref["timestamp"].to_numpy(np.int64)[:FRAME_LIMIT]
    velocity = ref[["vx", "vy", "vz"]].to_numpy(np.float64)[:FRAME_LIMIT] @ oracle.FLU_TO_NED.T
    decomposition = pd.read_csv(oracle.DEFAULT_DATASET / "imu_truth_decomposition.csv")
    imu_time = decomposition["timestamp"].to_numpy(np.int64)
    ba = oracle.interpolate_rows(
        imu_time,
        decomposition[["acc_bias_x", "acc_bias_y", "acc_bias_z"]].to_numpy(np.float64),
        time_ns,
    ) @ oracle.FLU_TO_NED.T
    bg = oracle.interpolate_rows(
        imu_time,
        decomposition[["gyro_bias_x", "gyro_bias_y", "gyro_bias_z"]].to_numpy(np.float64),
        time_ns,
    ) @ oracle.FLU_TO_NED.T
    return truth_pose[:FRAME_LIMIT], velocity, ba, bg


def summarize(
    *,
    window: int,
    data: dict[str, np.ndarray],
    final_states: dict[int, NavigationState],
    solve_rows: list[dict],
    elapsed: float,
    output: Path,
) -> dict:
    extrinsics = pp.SE3(
        torch.from_numpy(data["frames//imu_vio_sensor_T_imu"][:FRAME_LIMIT].astype(np.float64))
    )
    pose = pp.SE3(torch.from_numpy(data["frames//pose"][:FRAME_LIMIT].astype(np.float64)))
    velocity = data["frames//imu_vio_velocity_world"][:FRAME_LIMIT].astype(np.float64).copy()
    ba = data["frames//imu_vio_acc_bias"][:FRAME_LIMIT].astype(np.float64).copy()
    bg = data["frames//imu_vio_gyro_bias"][:FRAME_LIMIT].astype(np.float64).copy()
    for frame, state in final_states.items():
        pose[frame] = pp.SE3(state.pose_WB) @ extrinsics[frame].Inv()
        velocity[frame] = state.velocity_W.numpy()
        ba[frame] = state.acc_bias.numpy()
        bg[frame] = state.gyro_bias.numpy()
    gt_pose_np, gt_velocity, gt_ba, gt_bg = truth_states(data)
    gt_pose = pp.SE3(torch.from_numpy(np.stack([oracle.se3_to_xyzw(item) for item in gt_pose_np])))
    position_error = pose.translation().numpy() - gt_pose.translation().numpy()
    rotation_error = (gt_pose.Inv() @ pose).Log().tensor().numpy()[:, 3:6]
    relative_error = (
        (gt_pose[:-1].Inv() @ gt_pose[1:]).Inv() @ (pose[:-1].Inv() @ pose[1:])
    ).Log().tensor().numpy()
    valid = slice(FIRST_FRAME, FRAME_LIMIT)
    solve = pd.DataFrame(solve_rows)
    metrics = {
        "ate_xy_rmse_m_no_alignment": rmse(position_error[valid, :2]),
        "ate_xyz_rmse_m_no_alignment": rmse(position_error[valid]),
        "orientation_rmse_rad": rmse(rotation_error[valid]),
        "translation_rpe_rmse_m": rmse(relative_error[FIRST_FRAME:, :3]),
        "rotation_rpe_rmse_rad": rmse(relative_error[FIRST_FRAME:, 3:6]),
        "velocity_truth_rmse_mps": rmse(velocity[valid] - gt_velocity[valid]),
        "acc_bias_truth_rmse_mps2": rmse(ba[valid] - gt_ba[valid]),
        "gyro_bias_truth_rmse_radps": rmse(bg[valid] - gt_bg[valid]),
    }
    state_rows = []
    for frame in range(FIRST_FRAME, FRAME_LIMIT):
        row = {"window": window, "frame": frame}
        vectors = {
            "position_error": position_error[frame],
            "rotation_error": rotation_error[frame],
            "velocity_error": velocity[frame] - gt_velocity[frame],
            "ba_error": ba[frame] - gt_ba[frame],
            "bg_error": bg[frame] - gt_bg[frame],
        }
        for prefix, vector in vectors.items():
            for axis, value in zip("xyz", vector):
                row[f"{prefix}_{axis}"] = float(value)
            row[f"{prefix}_norm"] = float(np.linalg.norm(vector))
        state_rows.append(row)
    pd.DataFrame(state_rows).to_csv(output / "state_per_frame.csv", index=False)
    solve.to_csv(output / "solve_per_edge.csv", index=False)
    summary = {
        "window": window,
        "edge_count": len(solve_rows),
        "factor_contract": {
            "source_tensor_map": str(latest(SOURCE / "run_result", "tensor_map.npz")),
            "visual_mean": "GT T_CiCj",
            "visual_covariance": "original MACVO covariance with C0 runtime gate inflation",
            "imu_delta_covariance_bias_jacobian": "frozen from C0 standard-local sampling-aware replay",
            "huber_delta": 3.0,
            "bias_variables": "online ba/bg",
            "lm_max_iterations": 20,
        },
        "metrics": metrics,
        "factor_cost_mean": {
            name: float(solve[name].mean())
            for name in ("prior_cost", "imu_cost", "bias_cost", "visual_pose_cost", "final_cost")
        },
        "solver": {
            "converged_rate": float(solve["converged"].mean()),
            "iterations_mean": float(solve["iterations"].mean()),
            "iterations_p95": float(solve["iterations"].quantile(0.95)),
            "elapsed_seconds": elapsed,
            "peak_rss_kb": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        },
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def run(window: int, *, force: bool) -> None:
    output = OUT / f"N{window}"
    if output.exists():
        if not force:
            raise FileExistsError(f"{output} exists; pass --force")
        if OUT.resolve() not in output.resolve().parents:
            raise RuntimeError("refusing to remove output outside audit root")
        shutil.rmtree(output)
    output.mkdir(parents=True)
    data = load_map()
    gate = pd.read_csv(SOURCE / "visual_gate_runtime.csv")
    inflation = {
        int(row.frame_i): float(row.covariance_inflation) for row in gate.itertuples(index=False)
    }
    state0 = state_from_map(data, FIRST_FRAME).to(device=torch.device("cpu"), dtype=DTYPE)
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
    active_frames = [FIRST_FRAME]
    states = [state0]
    imus: list[ImuPreintegrationFactor] = []
    visuals: list[RelativePoseFactor] = []
    final_states = {FIRST_FRAME: state0.detach()}
    solve_rows: list[dict] = []
    started = time.perf_counter()
    for frame_j in range(FIRST_FRAME + 1, FRAME_LIMIT):
        imu, visual = factor_from_map(data, frame_j, inflation)
        active_frames.append(frame_j)
        states.append(propagate_endpoint(states[-1], imu, visual))
        imus.append(imu)
        visuals.append(visual)
        result = solver.solve(
            FixedLagVIOProblem(
                states=tuple(states),
                prior_first=prior,
                imu_factors=tuple(imus),
                visual_factors=tuple(visuals),
                optimize_acc_bias=True,
                optimize_gyro_bias=True,
            )
        )
        states = list(result.states)
        for frame, state in zip(active_frames, states):
            final_states[frame] = state.detach()
        latest_predicted = pp.SE3(states[-2].pose_WB).Inv() @ pp.SE3(states[-1].pose_WB)
        correction = (pp.SE3(visual.measurement_BiBj).Inv() @ latest_predicted).Log().tensor().reshape(6)
        solve_rows.append(
            {
                "frame_j": frame_j,
                "active_window_size": len(states),
                "iterations": int(result.iterations),
                "converged": bool(result.converged),
                "convergence_reason": result.convergence_reason,
                "initial_cost": result.initial_cost,
                "final_cost": result.final_cost,
                "prior_cost": result.prior_cost,
                "imu_cost": result.imu_cost,
                "bias_cost": result.bias_cost,
                "visual_pose_cost": result.visual_pose_cost,
                "final_step_norm": result.final_step_norm,
                "final_gradient_inf_norm": result.final_gradient_inf_norm,
                "gate_inflation": inflation.get(frame_j - 1, 1.0),
                **prior_stats(result.prior_next),
                **{f"latest_pose_correction_{axis}": float(value) for axis, value in enumerate(correction)},
            }
        )
        if len(states) == window:
            prior = result.prior_next
            active_frames = active_frames[1:]
            states = states[1:]
            imus = imus[1:]
            visuals = visuals[1:]
        if (frame_j - FIRST_FRAME) % 50 == 0 or frame_j == FRAME_LIMIT - 1:
            print(f"[N={window}] {frame_j - FIRST_FRAME}/300", flush=True)
    summary = summarize(
        window=window,
        data=data,
        final_states=final_states,
        solve_rows=solve_rows,
        elapsed=time.perf_counter() - started,
        output=output,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def aggregate() -> None:
    rows = []
    for window in WINDOWS:
        path = OUT / f"N{window}/summary.json"
        if not path.exists():
            continue
        summary = json.loads(path.read_text(encoding="utf-8"))
        row = {"window": window, **summary["metrics"]}
        row.update({f"cost_mean_{key}": value for key, value in summary["factor_cost_mean"].items()})
        row.update({f"solver_{key}": value for key, value in summary["solver"].items()})
        rows.append(row)
    pd.DataFrame(rows).sort_values("window").to_csv(OUT / "v3_window_summary.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--window", type=int, choices=WINDOWS)
    parser.add_argument("--aggregate", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.aggregate:
        aggregate()
    elif args.window is not None:
        run(args.window, force=args.force)
        aggregate()
    else:
        raise ValueError("provide --window or --aggregate")


if __name__ == "__main__":
    main()
