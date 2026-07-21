#!/usr/bin/env python3
"""Run the approved 391-frame direct-UVD U1/U2 covariance experiments."""

from __future__ import annotations

import argparse
import json
import math
import runpy
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from scipy.spatial.transform import Rotation


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import Scripts.run_circle_translation_oracles as base  # noqa: E402
from Module.Optimization.TwoFramePGO.Optimizer import TwoFrame_PGO  # noqa: E402
from Utility.TwoStateVIO import TwoStateVIOSolver  # noqa: E402
from Utility.TwoStateVIO import state_boxminus  # noqa: E402
from Utility.TwoStateSamplingAwareVIO import (  # noqa: E402
    CrossEdgeTwoStateProblem,
    CrossEdgeTwoStateSolver,
    cross_edge_factor_residuals,
    symmetric_matrix_diagnostics,
)


OUT = ROOT / "analysis_direct_uvd_20260716"
BASELINE_CONFIG = (
    ROOT
    / "Baselines/two_state_relative_pose_standard_full_20260716/configs/odometry.yaml"
)
CASES = {
    "U1": {
        "warm_start": "macvo_pose",
        "covariance_mode": "current_independent_step",
    },
    "U2": {
        "warm_start": "imu_propagation",
        "covariance_mode": "current_independent_step",
    },
    "U1_sampling_aware": {
        "warm_start": "macvo_pose",
        "covariance_mode": "sampling_aware",
    },
    "U1_sampling_aware_v2": {
        "warm_start": "macvo_pose",
        "covariance_mode": "sampling_aware_cross_edge",
    },
    "U1_sampling_aware_v2_rank_aware": {
        "warm_start": "macvo_pose",
        "covariance_mode": "sampling_aware_cross_edge",
        "rank_aware_imu_whitening": True,
    },
}


ORIGINAL_CROSS_EDGE_SOLVE = CrossEdgeTwoStateSolver.solve


def traced_cross_edge_solve(
    self: CrossEdgeTwoStateSolver, problem: CrossEdgeTwoStateProblem
):
    dtype = torch.float64
    device = problem.state_i.pose_WB.device
    normalized = replace(
        problem,
        state_i=problem.state_i.to(device=device, dtype=dtype),
        state_j=problem.state_j.to(device=device, dtype=dtype),
        noise_i=problem.noise_i.reshape(-1).to(device=device, dtype=dtype),
        noise_j=problem.noise_j.reshape(-1).to(device=device, dtype=dtype),
        prior_i=problem.prior_i.to(device=device, dtype=dtype),
        imu=problem.imu.to(device=device, dtype=dtype),
        visual=problem.visual.to(device=device, dtype=dtype),
    )
    trace_imu = normalized.imu
    unique_diagnostics = symmetric_matrix_diagnostics(
        trace_imu.unique_covariance,
        eigenvalue_floor=self.covariance_eigenvalue_floor,
    )
    if (
        self.rank_aware_imu_whitening
        and unique_diagnostics.effective_rank < unique_diagnostics.dimension
    ):
        trace_imu = replace(
            trace_imu,
            unique_covariance=trace_imu.base.covariance,
            incoming_sensitivity=torch.zeros_like(trace_imu.incoming_sensitivity),
            outgoing_sensitivity=torch.zeros_like(trace_imu.outgoing_sensitivity),
        )
    _, before = cross_edge_factor_residuals(
        normalized.state_i,
        normalized.state_j,
        normalized.noise_i,
        normalized.noise_j,
        normalized.prior_i,
        trace_imu,
        normalized.visual,
        covariance_eigenvalue_floor=self.covariance_eigenvalue_floor,
        robust_visual=False,
    )
    result = ORIGINAL_CROSS_EDGE_SOLVE(self, problem)
    _, after = cross_edge_factor_residuals(
        result.state_i,
        result.state_j,
        result.noise_i,
        result.noise_j,
        normalized.prior_i,
        trace_imu,
        normalized.visual,
        covariance_eigenvalue_floor=self.covariance_eigenvalue_floor,
        robust_visual=False,
    )
    update_i = state_boxminus(result.state_i, normalized.state_i).detach().cpu().numpy()
    update_j = state_boxminus(result.state_j, normalized.state_j).detach().cpu().numpy()
    base.CURRENT_EDGE["call"] += 1
    row = {
        "frame_i": base.CURRENT_EDGE["i"],
        "frame_j": base.CURRENT_EDGE["j"],
        "solver_call": base.CURRENT_EDGE["call"],
        "iterations": int(result.iterations),
        "converged": bool(result.converged),
        "final_step_norm": float(result.final_step_norm),
        "final_gradient_inf_norm": float(result.final_gradient_inf_norm),
        "convergence_reason": str(result.convergence_reason),
        "sampling_noise_cost_before": base.factor_cost(before["sampling_noise"]),
        "sampling_noise_cost_after": base.factor_cost(after["sampling_noise"]),
        "cross_covariance_frobenius_norm": float(
            result.cross_covariance_frobenius_norm
        ),
        "rank_aware_imu_whitening": bool(result.rank_aware_imu_whitening),
        "rank_aware_fallback_active": bool(result.rank_aware_fallback_active),
        "rank_aware_imu_residual_dimension": int(
            result.rank_aware_imu_residual_dimension
        ),
        "unique_cov_min_eigenvalue": float(
            result.unique_covariance_diagnostics.min_eigenvalue
        ),
        "unique_cov_max_eigenvalue": float(
            result.unique_covariance_diagnostics.max_eigenvalue
        ),
        "unique_cov_effective_rank": int(
            result.unique_covariance_diagnostics.effective_rank
        ),
        "unique_cov_condition_number": float(
            result.unique_covariance_diagnostics.condition_number
        ),
        "prior_i_min_eigenvalue": float(
            result.incoming_prior_diagnostics.min_eigenvalue
        ),
        "prior_i_max_eigenvalue": float(
            result.incoming_prior_diagnostics.max_eigenvalue
        ),
        "prior_i_effective_rank": int(
            result.incoming_prior_diagnostics.effective_rank
        ),
        "prior_i_condition_number": float(
            result.incoming_prior_diagnostics.condition_number
        ),
        "h_mm_min_eigenvalue": float(
            result.marginalization_diagnostics.h_mm.min_eigenvalue
        ),
        "h_mm_max_eigenvalue": float(
            result.marginalization_diagnostics.h_mm.max_eigenvalue
        ),
        "h_mm_effective_rank": int(
            result.marginalization_diagnostics.h_mm.effective_rank
        ),
        "h_mm_condition_number": float(
            result.marginalization_diagnostics.h_mm.condition_number
        ),
        "prior_j_min_eigenvalue": float(
            result.marginalization_diagnostics.schur_prior.min_eigenvalue
        ),
        "prior_j_max_eigenvalue": float(
            result.marginalization_diagnostics.schur_prior.max_eigenvalue
        ),
        "prior_j_effective_rank": int(
            result.marginalization_diagnostics.schur_prior.effective_rank
        ),
        "prior_j_condition_number": float(
            result.marginalization_diagnostics.schur_prior.condition_number
        ),
        "schur_quadratic_relative_error": float(
            result.marginalization_diagnostics.quadratic_relative_error
        ),
        "common_translation_update_world_norm": float(
            torch.linalg.vector_norm(result.common_translation_update_world)
        ),
        "differential_translation_update_world_norm": float(
            torch.linalg.vector_norm(result.differential_translation_update_world)
        ),
    }
    for name in ("prior", "imu", "bias"):
        row[f"{name}_cost_before"] = base.factor_cost(before[name])
        row[f"{name}_cost_after"] = base.factor_cost(after[name])
    row["pose_cost_before"] = base.factor_cost(before["visual_pose_unweighted"])
    row["pose_cost_after"] = base.factor_cost(after["visual_pose_unweighted"])
    for state_name, state_before, state_after, update in (
        ("i", normalized.state_i, result.state_i, update_i),
        ("j", normalized.state_j, result.state_j, update_j),
    ):
        for sensor_name, before_values, after_values in (
            ("v", state_before.velocity_W, state_after.velocity_W),
            ("ba", state_before.acc_bias, state_after.acc_bias),
            ("bg", state_before.gyro_bias, state_after.gyro_bias),
        ):
            for axis, value_before, value_after in zip(
                "xyz", before_values, after_values
            ):
                row[f"{sensor_name}_{state_name}_before_{axis}"] = float(value_before)
                row[f"{sensor_name}_{state_name}_after_{axis}"] = float(value_after)
        for index, value in enumerate(update):
            row[f"state_{state_name}_update_{index}"] = float(value)
    base.TRACE_ROWS.append(row)
    return result


def write_json(path: Path, payload) -> None:
    path.write_text(
        json.dumps(base.jsonify(payload), indent=2, ensure_ascii=False), encoding="utf-8"
    )


def prepare_cache() -> Path:
    cache = base.cache_root("short", "V0")
    manifest = cache / "manifest.json"
    sidecar = cache / "relative_pose_factors.npz"
    if not manifest.exists() or not sidecar.exists():
        base.prepare_cache("short", "V0")
    return cache


def prepare_config(case: str) -> Path:
    root = OUT / case
    root.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load(BASELINE_CONFIG.read_text(encoding="utf-8"))
    optimizer = config["Odometry"]["optimizer"]["args"]
    optimizer["two_state_visual_factor_mode"] = "direct_uvd"
    optimizer["two_state_warm_start"] = CASES[case]["warm_start"]
    optimizer["two_state_uvd_huber_delta"] = 0.1
    if CASES[case].get("rank_aware_imu_whitening", False):
        optimizer["two_state_cross_edge_rank_aware_imu_whitening"] = True
    odometry = config["Odometry"]["args"]
    odometry["mapping"] = False
    odometry["two_state_covariance_mode"] = CASES[case]["covariance_mode"]
    path = root / "effective_odometry.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def run(case: str, *, force: bool) -> None:
    root = OUT / case
    run_root = root / "run_result"
    trace_path = root / "factor_trace.csv"
    if run_root.exists():
        if not force:
            raise FileExistsError(f"{run_root} exists; pass --force")
        base.safe_remove(run_root, root)
    cache = prepare_cache()
    config = prepare_config(case)

    base.TRACE_ROWS.clear()
    TwoStateVIOSolver.solve = base.traced_solve
    CrossEdgeTwoStateSolver.solve = traced_cross_edge_solve
    TwoFrame_PGO._optimize_two_state_fixed_lag = staticmethod(base.traced_optimize)
    started = time.perf_counter()
    try:
        sys.argv = [
            str(ROOT / "MACVO.py"),
            "--odom",
            str(config),
            "--data",
            str(base.DATA_CONFIG_SOURCE),
            "--resultRoot",
            str(run_root),
            "--visual-cache-mode",
            "replay",
            "--visual-cache-path",
            str(cache),
            "--seq_to",
            str(base.SHORT_FRAME_LIMIT),
        ]
        runpy.run_path(str(ROOT / "MACVO.py"), run_name="__main__")
    finally:
        TwoStateVIOSolver.solve = base.ORIGINAL_SOLVE
        CrossEdgeTwoStateSolver.solve = ORIGINAL_CROSS_EDGE_SOLVE
        TwoFrame_PGO._optimize_two_state_fixed_lag = staticmethod(base.ORIGINAL_OPTIMIZE)
    elapsed = time.perf_counter() - started
    if not base.TRACE_ROWS:
        raise RuntimeError("direct-UVD replay produced no two-state trace rows")
    pd.DataFrame(base.TRACE_ROWS).to_csv(trace_path, index=False)
    (root / "runtime_seconds.txt").write_text(f"{elapsed:.9f}\n", encoding="utf-8")
    summarize(case)


def summarize(case: str) -> dict:
    root = OUT / case
    run_root = root / "run_result"
    pose_path = base.latest(run_root, "poses.csv")
    tensor_path = base.latest(run_root, "tensor_map.npz")
    diagnostics_path = base.latest(run_root, "frame_pair_diagnostics.csv")
    time_ns, pose_est = base.read_pose_csv(pose_path)
    ref = pd.read_csv(base.DEFAULT_DATASET / "ref_pose.csv").iloc[: len(time_ns)]
    if not np.array_equal(time_ns, ref["timestamp"].to_numpy(np.int64)):
        raise AssertionError("direct-UVD timestamps differ from reference")
    pose_gt = np.stack(
        [
            base.make_transform(Rotation.from_quat(row[3:7]).as_matrix(), row[:3])
            for row in ref[["x", "y", "z", "qx", "qy", "qz", "qw"]].to_numpy(
                np.float64
            )
        ]
    )
    active = slice(base.ACTIVE_START_FRAME, len(time_ns))
    metrics = base.trajectory_metrics(pose_est[active], pose_gt[active])

    with np.load(tensor_path, allow_pickle=False) as tensor:
        velocity_est = tensor["frames//imu_vio_velocity_world"][: len(time_ns)].astype(
            np.float64
        )
        ba_est = tensor["frames//imu_vio_acc_bias"][: len(time_ns)].astype(np.float64)
        bg_est = tensor["frames//imu_vio_gyro_bias"][: len(time_ns)].astype(np.float64)
        imu_covariance = tensor["frames//imu_vio_cov"][: len(time_ns)].astype(
            np.float64
        )
        pose_internal = np.stack(
            [
                base.se3_from_xyzw(row)
                for row in tensor["frames//pose"][: len(time_ns)].astype(np.float64)
            ]
        )
    diagnostics = pd.read_csv(diagnostics_path)
    valid_diag = diagnostics[
        (diagnostics["frame_i"] >= base.ACTIVE_START_FRAME)
        & (diagnostics["frame_j"] < len(time_ns))
    ]
    edge_covariance = imu_covariance[base.ACTIVE_START_FRAME + 1 : len(time_ns)]
    covariance_diagnostics = {
        "imu_vio_cov_trace": base.distribution(
            np.trace(edge_covariance, axis1=1, axis2=2)
        ),
        "imu_vio_whitened_norm": base.distribution(
            valid_diag["imu_vio_whitened_norm"].to_numpy(np.float64)
        ),
        "imu_vio_weight_trace": base.distribution(
            np.trace(np.linalg.inv(edge_covariance), axis1=1, axis2=2)
        ),
        "source": "tensor_map_full_9x9_covariance",
    }
    velocity_by_frame: dict[int, np.ndarray] = {}
    for row in valid_diag.itertuples(index=False):
        velocity_by_frame[int(row.frame_i)] = np.array(
            [row.gt_velocity_i_x, row.gt_velocity_i_y, row.gt_velocity_i_z]
        )
        velocity_by_frame[int(row.frame_j)] = np.array(
            [row.gt_velocity_j_x, row.gt_velocity_j_y, row.gt_velocity_j_z]
        )
    fallback_velocity = (
        ref[["vx", "vy", "vz"]].to_numpy(np.float64) @ base.FLU_TO_NED.T
    )
    velocity_gt = np.stack(
        [velocity_by_frame.get(index, fallback_velocity[index]) for index in range(len(time_ns))]
    )
    decomposition = pd.read_csv(base.DEFAULT_DATASET / "imu_truth_decomposition.csv")
    imu_time = decomposition["timestamp"].to_numpy(np.int64)
    ba_gt = base.interpolate_rows(
        imu_time,
        decomposition[["acc_bias_x", "acc_bias_y", "acc_bias_z"]].to_numpy(np.float64),
        time_ns,
    ) @ base.FLU_TO_NED.T
    bg_gt = base.interpolate_rows(
        imu_time,
        decomposition[["gyro_bias_x", "gyro_bias_y", "gyro_bias_z"]].to_numpy(np.float64),
        time_ns,
    ) @ base.FLU_TO_NED.T
    metrics.update(
        {
            "velocity_truth_rmse_mps": base.rmse_norm(
                velocity_est[active] - velocity_gt[active]
            ),
            "acc_bias_truth_rmse_mps2": base.rmse_norm(ba_est[active] - ba_gt[active]),
            "gyro_bias_truth_rmse_radps": base.rmse_norm(bg_est[active] - bg_gt[active]),
        }
    )

    with np.load(prepare_cache() / "relative_pose_factors.npz", allow_pickle=False) as factors:
        measurement = np.stack(
            [base.se3_from_xyzw(row.reshape(7)) for row in factors["measurement_CiCj"]]
        )
    corrections = []
    for index in range(base.ACTIVE_START_FRAME, len(time_ns) - 1):
        z_est = base.invert_transform(pose_internal[index]) @ pose_internal[index + 1]
        correction = base.invert_transform(measurement[index]) @ z_est
        corrections.append(
            np.r_[correction[:3, 3], base.rotation_log(correction[:3, :3])]
        )
    corrections = np.asarray(corrections)

    trace = pd.read_csv(root / "factor_trace.csv")
    trace = (
        trace.sort_values(["frame_i", "solver_call"])
        .groupby(["frame_i", "frame_j"], as_index=False)
        .tail(1)
    )
    trace = trace[
        (trace["frame_i"] >= base.ACTIVE_START_FRAME)
        & (trace["frame_j"] < len(time_ns))
    ].reset_index(drop=True)
    convergence = {
        "edge_count": int(len(trace)),
        "converged_rate": float(trace.converged.astype(bool).mean()),
        "iterations": base.distribution(trace.iterations.to_numpy(np.float64)),
        "reached_iteration_limit_count": int((trace.iterations >= 20).sum()),
        "final_step_norm": base.distribution(trace.final_step_norm.to_numpy(np.float64)),
        "final_gradient_inf_norm": base.distribution(
            trace.final_gradient_inf_norm.to_numpy(np.float64)
        ),
    }
    factor_costs = {
        name: {
            "before": base.distribution(trace[f"{name}_cost_before"].to_numpy(np.float64)),
            "after": base.distribution(trace[f"{name}_cost_after"].to_numpy(np.float64)),
            "sum_before": float(trace[f"{name}_cost_before"].sum()),
            "sum_after": float(trace[f"{name}_cost_after"].sum()),
        }
        for name in ("prior", "imu", "bias", "pose")
    }
    visual_columns = {
        "inlier_ratio": "visual_pose_inlier_ratio",
        "mean_mahalanobis_sq": "visual_pose_mean_mahalanobis_sq",
        "whitened_residual_norm": "visual_pose_whitened_residual_norm",
    }
    if all(column in valid_diag.columns for column in visual_columns.values()):
        visual_diagnostics = {
            name: base.distribution(valid_diag[column].to_numpy(np.float64))
            for name, column in visual_columns.items()
        }
        visual_diagnostics["source"] = "frame_pair_diagnostics"
    else:
        # The legacy diagnostics schema does not persist the direct-UVD gate fields.
        # Recover the two aggregate quantities that are exactly available from the
        # unrobust whitened point cost recorded by the factor trace. Per-point
        # inlier ratio cannot be reconstructed after the run and is marked as such.
        point_counts = valid_diag.set_index(["frame_i", "frame_j"])[
            "local_ba_num_visual_residual_blocks"
        ]
        trace_point_counts = np.array(
            [point_counts.loc[(int(row.frame_i), int(row.frame_j))] for row in trace.itertuples()],
            dtype=np.float64,
        )
        post_square_sum = 2.0 * trace.pose_cost_after.to_numpy(np.float64)
        visual_diagnostics = {
            "inlier_ratio": {
                "available": False,
                "reason": "direct-UVD per-point norms were not persisted by the legacy diagnostics schema",
            },
            "mean_mahalanobis_sq": base.distribution(
                post_square_sum / np.maximum(trace_point_counts, 1.0)
            ),
            "whitened_residual_norm": base.distribution(
                np.sqrt(np.maximum(post_square_sum, 0.0))
            ),
            "source": "factor_trace_unrobust_whitened_cost",
        }
    runtime_seconds = float((root / "runtime_seconds.txt").read_text().strip())
    summary = {
        "case": case,
        "visual_factor_mode": "direct_uvd",
        "warm_start": CASES[case]["warm_start"],
        "covariance_mode": CASES[case]["covariance_mode"],
        "frame_count": int(len(time_ns)),
        "active_start_frame": int(base.ACTIVE_START_FRAME),
        "active_edge_count": int(len(trace)),
        "metrics": metrics,
        "convergence": convergence,
        "factor_costs": factor_costs,
        "covariance_diagnostics": covariance_diagnostics,
        "visual_diagnostics": visual_diagnostics,
        "relative_to_macvo_measurement": {
            "translation_norm_m": base.distribution(np.linalg.norm(corrections[:, :3], axis=1)),
            "rotation_norm_rad": base.distribution(np.linalg.norm(corrections[:, 3:], axis=1)),
        },
        "runtime_seconds": runtime_seconds,
        "runtime_seconds_per_active_edge": runtime_seconds / max(len(trace), 1),
        "all_finite": bool(
            np.isfinite(trace.select_dtypes(include=[np.number])).all().all()
            and all(math.isfinite(float(value)) for value in metrics.values())
        ),
        "artifacts": {
            "poses": pose_path,
            "tensor_map": tensor_path,
            "diagnostics": diagnostics_path,
            "factor_trace": root / "factor_trace.csv",
        },
    }
    write_json(root / "summary.json", summary)
    print(json.dumps(base.jsonify(summary), indent=2, ensure_ascii=False))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=tuple(CASES), required=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    arguments = parser.parse_args()
    if arguments.summarize_only:
        summarize(arguments.case)
    else:
        run(arguments.case, force=bool(arguments.force))


if __name__ == "__main__":
    main()
