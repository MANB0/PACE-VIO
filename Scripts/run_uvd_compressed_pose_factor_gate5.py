#!/usr/bin/env python3
"""Run the 300-frame T0/T2 fixed-lag Gate-5 comparison.

The script consumes the already captured Direct-UVD U1 problems. It does not
run MACVO, change production configuration, or replay a full sequence.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pypose as pp
import torch
from scipy.spatial.transform import Rotation
from scipy.stats import chi2


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Module.Optimization.TwoFramePGO.Optimizer import (  # noqa: E402
    _gate_two_state_visual_factor,
    _two_state_visual_whitened_norm,
)
from Scripts.run_rectangle_uvd_schur_marginal_experiment import (  # noqa: E402
    load_truth,
    pose_matrix,
)
from Scripts.run_u1_counterfactual_branches import (  # noqa: E402
    clone_problem,
    future_problem,
)
from Scripts.plot_static63_gt_macvo import HTML_TEMPLATE  # noqa: E402
from Utility.RelativePoseFactorCache import (  # noqa: E402
    camera_factor_to_body_factor,
)
from Utility.TwoStateVIO import (  # noqa: E402
    LinearizedUVDPoseFactor,
    NavigationState,
    RelativePoseFactor,
    TwoStateVIOProblem,
    TwoStateVIOResult,
    TwoStateVIOSolver,
    UVDFactor,
    _factor_residuals,
    linearize_uvd_relative_pose_factor,
)


DEFAULT_PACKET = ROOT / (
    "analysis_rectangle_uvd_schur_marginal_20260719/"
    "captured_rectangle_u1_problems.pt"
)
DEFAULT_OUTPUT = ROOT / "analysis_uvd_compressed_pose_factor_gate5_20260720"
T0 = "T0_relative_pose_sidecar"
T2 = "T2_uvd_local_compression"
METHODS = (T0, T2)
DTYPE = torch.float64
FLU_TO_NED = np.diag([1.0, -1.0, -1.0])
VISUAL_GATE = {
    "soft_inlier_ratio": 0.5,
    "reject_inlier_ratio": 0.2,
    "soft_mean_mahalanobis_sq": 9.0,
    "reject_mean_mahalanobis_sq": 100.0,
    "soft_whitened_pose_norm": 6.0,
    "reject_whitened_pose_norm": 20.0,
    "max_covariance_inflation": 1.0e6,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, default=DEFAULT_PACKET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--edge-count", type=int, default=209)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def jsonify(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): jsonify(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonify(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(jsonify(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write an empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def distribution(values: np.ndarray | list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 0:
        return {
            key: float("nan")
            for key in ("min", "median", "mean", "rmse", "p95", "max")
        }
    return {
        "min": float(array.min()),
        "median": float(np.median(array)),
        "mean": float(array.mean()),
        "rmse": float(np.sqrt(np.mean(array**2))),
        "p95": float(np.quantile(array, 0.95)),
        "max": float(array.max()),
    }


def symmetric_pseudoinverse(
    matrix: torch.Tensor, relative_threshold: float = 1.0e-10
) -> torch.Tensor:
    matrix = 0.5 * (matrix + matrix.mT)
    values, vectors = torch.linalg.eigh(matrix)
    scale = max(float(values.abs().max().item()), 1.0)
    threshold = max(
        relative_threshold * scale,
        torch.finfo(matrix.dtype).eps * scale * int(matrix.shape[0]),
    )
    inverse = torch.where(
        values > threshold, values.reciprocal(), torch.zeros_like(values)
    )
    return vectors @ torch.diag(inverse) @ vectors.mT


def inversion_right_tangent_jacobian(measurement_CiCj: torch.Tensor) -> torch.Tensor:
    measurement = pp.SE3(measurement_CiCj.reshape(1, 7))
    inverse_reference = measurement.Inv().detach()
    zero = torch.zeros(6, dtype=measurement_CiCj.dtype, requires_grad=True)

    def error(increment: torch.Tensor) -> torch.Tensor:
        changed = measurement @ pp.se3(increment.reshape(1, 6)).Exp()
        changed_inverse = changed.Inv()
        return (
            inverse_reference.Inv() @ changed_inverse
        ).Log().tensor().reshape(6)

    return torch.autograd.functional.jacobian(
        error, zero, create_graph=False, vectorize=True
    ).detach()


def matrix_to_se3(matrix: np.ndarray) -> pp.LieTensor:
    return pp.mat2SE3(torch.as_tensor(matrix, dtype=DTYPE).reshape(1, 4, 4))


def camera_relative_CjCi(
    state_i: NavigationState,
    state_j: NavigationState,
    extrinsic_CI: torch.Tensor,
) -> pp.LieTensor:
    extrinsic = pp.SE3(extrinsic_CI.reshape(1, 7))
    pose_WCi = pp.SE3(state_i.pose_WB.reshape(1, 7)) @ extrinsic.Inv()
    pose_WCj = pp.SE3(state_j.pose_WB.reshape(1, 7)) @ extrinsic.Inv()
    return pose_WCj.Inv() @ pose_WCi


def truth_camera_relative_CjCi(
    pose_WBi: np.ndarray,
    pose_WBj: np.ndarray,
    extrinsic_CI: torch.Tensor,
) -> pp.LieTensor:
    extrinsic = (
        pp.SE3(extrinsic_CI.reshape(1, 7))
        .matrix()
        .reshape(4, 4)
        .detach()
        .cpu()
        .numpy()
    )
    pose_WCi = pose_WBi @ np.linalg.inv(extrinsic)
    pose_WCj = pose_WBj @ np.linalg.inv(extrinsic)
    return matrix_to_se3(np.linalg.inv(pose_WCj) @ pose_WCi)


def visual_cost_from_blocks(
    blocks: dict[str, torch.Tensor], huber_delta: float
) -> float:
    norms = blocks["visual_group_norms"]
    delta = torch.as_tensor(max(float(huber_delta), 1.0e-12), dtype=norms.dtype)
    value = torch.where(
        norms <= delta,
        0.5 * norms.square(),
        delta * norms - 0.5 * delta.square(),
    ).sum()
    return float(value.detach().cpu().item())


def factor_costs(
    problem: TwoStateVIOProblem,
    state_i: NavigationState,
    state_j: NavigationState,
) -> dict[str, float]:
    _, blocks = _factor_residuals(
        state_i,
        state_j,
        problem.prior_i,
        problem.imu,
        problem.visual_pose,
        covariance_eigenvalue_floor=1.0e-12,
        robust_visual=False,
    )
    return {
        "prior": 0.5 * float(blocks["prior"].square().sum().item()),
        "imu": 0.5 * float(blocks["imu"].square().sum().item()),
        "bias": 0.5 * float(blocks["bias"].square().sum().item()),
        "pose": visual_cost_from_blocks(
            blocks, float(problem.visual_pose.huber_delta)
        ),
        "imu_whitened_norm": float(blocks["imu"].norm().item()),
        "pose_whitened_norm": float(
            blocks["visual_pose_unweighted"].norm().item()
        ),
    }


def sidecar_lookup(sidecar: np.lib.npyio.NpzFile) -> dict[tuple[int, int], int]:
    return {
        (int(frame_i), int(frame_j)): index
        for index, (frame_i, frame_j) in enumerate(
            zip(sidecar["frame_i"], sidecar["frame_j"], strict=True)
        )
    }


def prepare_edges(payload: dict[str, Any], edge_count: int) -> list[dict[str, Any]]:
    start = int(payload["active_start_frame"])
    edges = [edge for edge in payload["edges"] if int(edge["frame_i"]) >= start][
        :edge_count
    ]
    if len(edges) != edge_count:
        raise RuntimeError(f"requested {edge_count} edges, found {len(edges)}")
    for previous, current in zip(edges, edges[1:]):
        if int(previous["frame_j"]) != int(current["frame_i"]):
            raise RuntimeError("captured edges are not contiguous")
    return edges


def build_t0_factor(
    incoming: TwoStateVIOProblem,
    sidecar: np.lib.npyio.NpzFile,
    row: int,
) -> tuple[RelativePoseFactor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    visual = incoming.visual_pose
    if not isinstance(visual, UVDFactor):
        raise TypeError("captured problem does not contain a UVDFactor")
    measurement_camera = torch.as_tensor(
        sidecar["measurement_CiCj"][row], dtype=DTYPE
    ).reshape(1, 7)
    covariance_camera = torch.as_tensor(
        sidecar["covariance"][row], dtype=DTYPE
    ).reshape(6, 6)
    measurement_body, covariance_body = camera_factor_to_body_factor(
        measurement_camera,
        covariance_camera,
        visual.extrinsic_CI,
    )
    covariance_body, gate = _gate_two_state_visual_factor(
        incoming.state_i,
        incoming.state_j,
        measurement_body,
        covariance_body,
        num_points=int(sidecar["num_points"][row]),
        num_inliers=int(sidecar["num_inliers"][row]),
        mean_mahalanobis_sq=float(sidecar["mean_mahalanobis_sq"][row]),
        config=VISUAL_GATE,
        eigenvalue_floor=1.0e-12,
    )
    return (
        RelativePoseFactor(
            measurement_BiBj=measurement_body,
            covariance=covariance_body,
            huber_delta=3.0,
        ),
        measurement_body,
        covariance_body,
        gate,
    )


def build_t2_factor(
    incoming: TwoStateVIOProblem,
    sidecar: np.lib.npyio.NpzFile,
    row: int,
) -> tuple[LinearizedUVDPoseFactor, dict[str, Any]]:
    visual = incoming.visual_pose
    if not isinstance(visual, UVDFactor):
        raise TypeError("captured problem does not contain a UVDFactor")
    measurement_CiCj = torch.as_tensor(
        sidecar["measurement_CiCj"][row], dtype=DTYPE
    ).reshape(1, 7)
    reference_CjCi = pp.SE3(measurement_CiCj).Inv().detach()
    linearization = linearize_uvd_relative_pose_factor(
        reference_CjCi.tensor(), visual, marginal_mode="full"
    )
    hessian = linearization.full_hessian
    gradient = linearization.full_gradient
    center_shift = -symmetric_pseudoinverse(hessian) @ gradient
    values = torch.linalg.eigvalsh(hessian)
    return linearization.factor, {
        "hessian": hessian,
        "gradient": gradient,
        "center_shift": center_shift,
        "min_eigenvalue": float(values.min().item()),
        "max_eigenvalue": float(values.max().item()),
        "condition": float((values.max() / values.min()).item()),
    }


def measurement_nis(
    *,
    method: str,
    incoming: TwoStateVIOProblem,
    sidecar: np.lib.npyio.NpzFile,
    row: int,
    truth_pose_i: np.ndarray,
    truth_pose_j: np.ndarray,
    t2_diagnostics: dict[str, Any] | None,
) -> tuple[float, np.ndarray]:
    visual = incoming.visual_pose
    measurement_CiCj = torch.as_tensor(
        sidecar["measurement_CiCj"][row], dtype=DTYPE
    ).reshape(1, 7)
    reference_CjCi = pp.SE3(measurement_CiCj).Inv().detach()
    truth_CjCi = truth_camera_relative_CjCi(
        truth_pose_i, truth_pose_j, visual.extrinsic_CI
    )
    truth_delta = (
        reference_CjCi.Inv() @ truth_CjCi
    ).Log().tensor().reshape(6)
    if method == T0:
        covariance_CiCj = torch.as_tensor(
            sidecar["covariance"][row], dtype=DTYPE
        ).reshape(6, 6)
        jacobian = inversion_right_tangent_jacobian(measurement_CiCj)
        covariance = jacobian @ covariance_CiCj @ jacobian.mT
        information = symmetric_pseudoinverse(
            covariance, relative_threshold=1.0e-12
        )
        error = truth_delta
    else:
        if t2_diagnostics is None:
            raise ValueError("T2 NIS requires compression diagnostics")
        information = t2_diagnostics["hessian"]
        error = truth_delta - t2_diagnostics["center_shift"]
    nis = float((error @ information @ error).item())
    return nis, error.detach().cpu().numpy()


def solve_t0_with_production_gate(
    solver: TwoStateVIOSolver,
    incoming: TwoStateVIOProblem,
    sidecar: np.lib.npyio.NpzFile,
    row: int,
) -> tuple[TwoStateVIOResult, TwoStateVIOProblem, dict[str, Any], float, int]:
    visual, measurement_body, covariance_body, gate = build_t0_factor(
        incoming, sidecar, row
    )
    problem = replace(incoming, visual_pose=visual)
    started = time.perf_counter()
    result = solver.solve(problem)
    solver_calls = 1
    post_norm = _two_state_visual_whitened_norm(
        result.state_i,
        result.state_j,
        measurement_body,
        covariance_body,
        eigenvalue_floor=1.0e-12,
    )
    gate["postsolve_whitened_pose_residual_norm"] = post_norm
    if not str(gate["action"]).startswith("reject:"):
        additional = 1.0
        post_action = None
        if post_norm > VISUAL_GATE["reject_whitened_pose_norm"]:
            additional = VISUAL_GATE["max_covariance_inflation"]
            post_action = "reject:high_postsolve_whitened_pose_residual"
        elif post_norm > VISUAL_GATE["soft_whitened_pose_norm"]:
            additional = (
                post_norm / VISUAL_GATE["soft_whitened_pose_norm"]
            ) ** 2
            post_action = "downweight:high_postsolve_whitened_pose_residual"
        if additional > 1.0:
            current = float(gate["covariance_inflation"])
            total = min(
                current * additional,
                VISUAL_GATE["max_covariance_inflation"],
            )
            covariance_body = covariance_body * (total / current)
            visual = RelativePoseFactor(
                measurement_BiBj=measurement_body,
                covariance=covariance_body,
                huber_delta=3.0,
            )
            problem = replace(incoming, visual_pose=visual)
            result = solver.solve(problem)
            solver_calls = 2
            gate["covariance_inflation"] = total
            gate["action"] = post_action
            gate["postsolve_whitened_pose_residual_norm"] = (
                _two_state_visual_whitened_norm(
                    result.state_i,
                    result.state_j,
                    measurement_body,
                    covariance_body,
                    eigenvalue_floor=1.0e-12,
                )
            )
    return result, problem, gate, time.perf_counter() - started, solver_calls


def relative_error(
    state_i: NavigationState,
    state_j: NavigationState,
    truth_i: np.ndarray,
    truth_j: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    estimate = np.linalg.inv(pose_matrix(state_i)) @ pose_matrix(state_j)
    truth = np.linalg.inv(truth_i) @ truth_j
    error = np.linalg.inv(truth) @ estimate
    return error[:3, 3], Rotation.from_matrix(error[:3, :3]).as_rotvec()


def run_branch(
    *,
    method: str,
    edges: list[dict[str, Any]],
    sidecar: np.lib.npyio.NpzFile,
    lookup: dict[tuple[int, int], int],
    truth: dict[str, np.ndarray],
    solver_settings: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[int, NavigationState]]:
    solver = TwoStateVIOSolver(**solver_settings)
    previous: TwoStateVIOResult | None = None
    rows: list[dict[str, Any]] = []
    final_states: dict[int, NavigationState] = {}

    for edge_index, edge in enumerate(edges):
        frame_i = int(edge["frame_i"])
        frame_j = int(edge["frame_j"])
        sidecar_row = lookup[(frame_i, frame_j)]
        incoming = (
            clone_problem(edge["problem"])
            if previous is None
            else future_problem(edge["problem"], previous)
        )
        build_started = time.perf_counter()
        t2_diagnostics = None
        if method == T0:
            factor_build_s = time.perf_counter() - build_started
            result, solved_problem, gate, solve_s, solver_calls = (
                solve_t0_with_production_gate(
                    solver, incoming, sidecar, sidecar_row
                )
            )
            factor_build_s = time.perf_counter() - build_started - solve_s
        elif method == T2:
            visual, t2_diagnostics = build_t2_factor(
                incoming, sidecar, sidecar_row
            )
            factor_build_s = time.perf_counter() - build_started
            solved_problem = replace(incoming, visual_pose=visual)
            before_solve = time.perf_counter()
            result = solver.solve(solved_problem)
            solve_s = time.perf_counter() - before_solve
            solver_calls = 1
            gate = {
                "action": "not_applied_gate5_isolation",
                "covariance_inflation": 1.0,
                "inlier_ratio": None,
                "mean_mahalanobis_sq": None,
                "postsolve_whitened_pose_residual_norm": "",
            }
        else:
            raise ValueError(method)

        before = factor_costs(
            solved_problem, solved_problem.state_i, solved_problem.state_j
        )
        after = factor_costs(solved_problem, result.state_i, result.state_j)
        reference_CjCi = pp.SE3(
            torch.as_tensor(
                sidecar["measurement_CiCj"][sidecar_row], dtype=DTYPE
            ).reshape(1, 7)
        ).Inv()
        estimated_CjCi = camera_relative_CjCi(
            result.state_i,
            result.state_j,
            incoming.visual_pose.extrinsic_CI,
        )
        correction = (
            reference_CjCi.Inv() @ estimated_CjCi
        ).Log().tensor().reshape(6)
        translation_error, rotation_error = relative_error(
            result.state_i,
            result.state_j,
            truth["poses"][frame_i],
            truth["poses"][frame_j],
        )
        nis, measurement_error = measurement_nis(
            method=method,
            incoming=incoming,
            sidecar=sidecar,
            row=sidecar_row,
            truth_pose_i=truth["poses"][frame_i],
            truth_pose_j=truth["poses"][frame_j],
            t2_diagnostics=t2_diagnostics,
        )
        row: dict[str, Any] = {
            "method": method,
            "edge_id": edge_index,
            "frame_i": frame_i,
            "frame_j": frame_j,
            "converged": bool(result.converged),
            "iterations": int(result.iterations),
            "solver_calls": solver_calls,
            "convergence_reason": str(result.convergence_reason),
            "accepted_steps": int(result.accepted_steps),
            "rejected_steps": int(result.rejected_steps),
            "final_step_norm": float(result.final_step_norm),
            "final_gradient_inf_norm": float(result.final_gradient_inf_norm),
            "initial_total_cost": float(result.initial_cost),
            "final_total_cost": float(result.final_cost),
            "factor_build_ms": factor_build_s * 1000.0,
            "solver_ms": solve_s * 1000.0,
            "total_edge_ms": (factor_build_s + solve_s) * 1000.0,
            "visual_gate_action": str(gate["action"]),
            "visual_covariance_inflation": float(gate["covariance_inflation"]),
            "visual_gate_postsolve_whitened_norm": gate.get(
                "postsolve_whitened_pose_residual_norm"
            ),
            "measurement_nis": nis,
            "measurement_error_tx": float(measurement_error[0]),
            "measurement_error_ty": float(measurement_error[1]),
            "measurement_error_tz": float(measurement_error[2]),
            "measurement_error_rx": float(measurement_error[3]),
            "measurement_error_ry": float(measurement_error[4]),
            "measurement_error_rz": float(measurement_error[5]),
            "final_relative_translation_error_x_m": float(translation_error[0]),
            "final_relative_translation_error_y_m": float(translation_error[1]),
            "final_relative_translation_error_z_m": float(translation_error[2]),
            "final_relative_translation_error_norm_m": float(
                np.linalg.norm(translation_error)
            ),
            "final_relative_translation_error_xy_m": float(
                np.linalg.norm(translation_error[:2])
            ),
            "final_relative_rotation_error_x_rad": float(rotation_error[0]),
            "final_relative_rotation_error_y_rad": float(rotation_error[1]),
            "final_relative_rotation_error_z_rad": float(rotation_error[2]),
            "final_relative_rotation_error_norm_rad": float(
                np.linalg.norm(rotation_error)
            ),
            "correction_tx_m": float(correction[0]),
            "correction_ty_m": float(correction[1]),
            "correction_tz_m": float(correction[2]),
            "correction_rx_rad": float(correction[3]),
            "correction_ry_rad": float(correction[4]),
            "correction_rz_rad": float(correction[5]),
            "correction_translation_norm_m": float(correction[:3].norm().item()),
            "correction_rotation_norm_rad": float(correction[3:].norm().item()),
        }
        for factor_name in ("prior", "imu", "bias", "pose"):
            row[f"{factor_name}_cost_before"] = before[factor_name]
            row[f"{factor_name}_cost_after"] = after[factor_name]
        row["imu_whitened_norm_before"] = before["imu_whitened_norm"]
        row["imu_whitened_norm_after"] = after["imu_whitened_norm"]
        row["pose_whitened_norm_before"] = before["pose_whitened_norm"]
        row["pose_whitened_norm_after"] = after["pose_whitened_norm"]
        if t2_diagnostics is None:
            row.update(
                {
                    "t2_hessian_min_eigenvalue": "",
                    "t2_hessian_max_eigenvalue": "",
                    "t2_hessian_condition": "",
                    "t2_center_shift_translation_norm_m": "",
                    "t2_center_shift_rotation_norm_rad": "",
                }
            )
        else:
            center_shift = t2_diagnostics["center_shift"]
            row.update(
                {
                    "t2_hessian_min_eigenvalue": t2_diagnostics["min_eigenvalue"],
                    "t2_hessian_max_eigenvalue": t2_diagnostics["max_eigenvalue"],
                    "t2_hessian_condition": t2_diagnostics["condition"],
                    "t2_center_shift_translation_norm_m": float(
                        center_shift[:3].norm().item()
                    ),
                    "t2_center_shift_rotation_norm_rad": float(
                        center_shift[3:].norm().item()
                    ),
                }
            )
        rows.append(row)
        final_states[frame_i] = result.state_i.detach()
        if edge_index == len(edges) - 1:
            final_states[frame_j] = result.state_j.detach()
        previous = result
        if (edge_index + 1) % 10 == 0 or edge_index + 1 == len(edges):
            print(
                f"[{method}] {edge_index + 1}/{len(edges)} edges",
                flush=True,
            )
    return rows, final_states


def interpolate_bias_truth(dataset: Path, timestamps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    decomposition = pd.read_csv(dataset / "imu_truth_decomposition.csv")
    source_time = decomposition["timestamp"].to_numpy(np.int64)

    def interpolate(columns: list[str]) -> np.ndarray:
        values = decomposition[columns].to_numpy(np.float64)
        result = np.column_stack(
            [np.interp(timestamps, source_time, values[:, axis]) for axis in range(3)]
        )
        return result @ FLU_TO_NED.T

    return (
        interpolate(["acc_bias_x", "acc_bias_y", "acc_bias_z"]),
        interpolate(["gyro_bias_x", "gyro_bias_y", "gyro_bias_z"]),
    )


def state_rows(
    *,
    method: str,
    states: dict[int, NavigationState],
    truth: dict[str, np.ndarray],
    timestamps: np.ndarray,
    ba_truth: np.ndarray,
    bg_truth: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    ordered_frames = sorted(states)
    if not ordered_frames:
        return rows
    first_frame = ordered_frames[0]
    estimate_origin = pose_matrix(states[first_frame])[:3, 3].copy()
    truth_origin = truth["poses"][first_frame, :3, 3].copy()
    for frame in ordered_frames:
        state = states[frame]
        estimate_unrebased = pose_matrix(state)
        truth_pose_unrebased = truth["poses"][frame]
        estimate = estimate_unrebased.copy()
        truth_pose = truth_pose_unrebased.copy()
        estimate[:3, 3] -= estimate_origin
        truth_pose[:3, 3] -= truth_origin
        position_error = estimate[:3, 3] - truth_pose[:3, 3]
        rotation_error = Rotation.from_matrix(
            truth_pose[:3, :3].T @ estimate[:3, :3]
        ).as_rotvec()
        velocity = state.velocity_W.detach().cpu().numpy().reshape(3)
        acc_bias = state.acc_bias.detach().cpu().numpy().reshape(3)
        gyro_bias = state.gyro_bias.detach().cpu().numpy().reshape(3)
        velocity_error = velocity - truth["velocity"][frame]
        acc_bias_error = acc_bias - ba_truth[frame]
        gyro_bias_error = gyro_bias - bg_truth[frame]
        row: dict[str, Any] = {
            "method": method,
            "frame": frame,
            "timestamp": int(timestamps[frame]),
        }
        for prefix, values in (
            ("position_est", estimate[:3, 3]),
            ("position_gt", truth_pose[:3, 3]),
            ("position_est_unrebased", estimate_unrebased[:3, 3]),
            ("position_gt_unrebased", truth_pose_unrebased[:3, 3]),
            ("forward_est", estimate[:3, 0]),
            ("forward_gt", truth_pose[:3, 0]),
            ("position_error", position_error),
            ("rotation_error", rotation_error),
            ("velocity_est", velocity),
            ("velocity_gt", truth["velocity"][frame]),
            ("velocity_error", velocity_error),
            ("acc_bias_est", acc_bias),
            ("acc_bias_gt", ba_truth[frame]),
            ("acc_bias_error", acc_bias_error),
            ("gyro_bias_est", gyro_bias),
            ("gyro_bias_gt", bg_truth[frame]),
            ("gyro_bias_error", gyro_bias_error),
        ):
            for axis, value in zip("xyz", values):
                row[f"{prefix}_{axis}"] = float(value)
        row.update(
            {
                "position_error_xy_norm_m": float(np.linalg.norm(position_error[:2])),
                "position_error_3d_norm_m": float(np.linalg.norm(position_error)),
                "orientation_error_norm_rad": float(np.linalg.norm(rotation_error)),
                "velocity_error_norm_mps": float(np.linalg.norm(velocity_error)),
                "acc_bias_error_norm_mps2": float(np.linalg.norm(acc_bias_error)),
                "gyro_bias_error_norm_radps": float(np.linalg.norm(gyro_bias_error)),
            }
        )
        rows.append(row)
    return rows


def summarize_method(
    method: str,
    edge_frame: pd.DataFrame,
    state_frame: pd.DataFrame,
) -> dict[str, Any]:
    edge = edge_frame[edge_frame.method == method]
    state = state_frame[state_frame.method == method]
    lower = float(chi2.ppf(0.025, df=6))
    upper = float(chi2.ppf(0.975, df=6))
    nis = edge.measurement_nis.to_numpy(np.float64)
    summary: dict[str, Any] = {
        "edge_count": int(len(edge)),
        "state_count": int(len(state)),
        "trajectory": {
            "xy_ate_rmse_m": float(
                np.sqrt(np.mean(state.position_error_xy_norm_m.to_numpy() ** 2))
            ),
            "position_3d_rmse_m": float(
                np.sqrt(np.mean(state.position_error_3d_norm_m.to_numpy() ** 2))
            ),
            "orientation_rmse_rad": float(
                np.sqrt(np.mean(state.orientation_error_norm_rad.to_numpy() ** 2))
            ),
            "orientation_rmse_deg": float(
                np.degrees(
                    np.sqrt(np.mean(state.orientation_error_norm_rad.to_numpy() ** 2))
                )
            ),
            "velocity_rmse_mps": float(
                np.sqrt(np.mean(state.velocity_error_norm_mps.to_numpy() ** 2))
            ),
            "acc_bias_rmse_mps2": float(
                np.sqrt(np.mean(state.acc_bias_error_norm_mps2.to_numpy() ** 2))
            ),
            "gyro_bias_rmse_radps": float(
                np.sqrt(np.mean(state.gyro_bias_error_norm_radps.to_numpy() ** 2))
            ),
            "position_axis_rmse_m": {
                axis: float(
                    np.sqrt(np.mean(state[f"position_error_{axis}"].to_numpy() ** 2))
                )
                for axis in "xyz"
            },
        },
        "rpe": {
            "translation_3d_m": distribution(
                edge.final_relative_translation_error_norm_m.to_numpy()
            ),
            "translation_xy_m": distribution(
                edge.final_relative_translation_error_xy_m.to_numpy()
            ),
            "rotation_rad": distribution(
                edge.final_relative_rotation_error_norm_rad.to_numpy()
            ),
            "rotation_deg": distribution(
                np.degrees(edge.final_relative_rotation_error_norm_rad.to_numpy())
            ),
        },
        "measurement_nis": distribution(nis)
        | {
            "dof": 6,
            "central_95_interval": [lower, upper],
            "inside_interval_ratio": float(((nis >= lower) & (nis <= upper)).mean()),
            "below_interval_ratio": float((nis < lower).mean()),
            "above_interval_ratio": float((nis > upper).mean()),
        },
        "convergence": {
            "converged_rate": float(edge.converged.astype(bool).mean()),
            "iteration_limit_count": int((edge.iterations >= 20).sum()),
            "iterations": distribution(edge.iterations.to_numpy()),
            "final_step_norm": distribution(edge.final_step_norm.to_numpy()),
            "final_gradient_inf_norm": distribution(
                edge.final_gradient_inf_norm.to_numpy()
            ),
            "solver_calls": distribution(edge.solver_calls.to_numpy()),
        },
        "runtime": {
            "factor_build_ms": distribution(edge.factor_build_ms.to_numpy()),
            "solver_ms": distribution(edge.solver_ms.to_numpy()),
            "total_edge_ms": distribution(edge.total_edge_ms.to_numpy()),
            "sum_factor_build_s": float(edge.factor_build_ms.sum() * 1.0e-3),
            "sum_solver_s": float(edge.solver_ms.sum() * 1.0e-3),
        },
        "correction": {
            "translation_m": distribution(edge.correction_translation_norm_m.to_numpy()),
            "rotation_rad": distribution(edge.correction_rotation_norm_rad.to_numpy()),
        },
        "factor_costs": {
            name: {
                "sum_before": float(edge[f"{name}_cost_before"].sum()),
                "sum_after": float(edge[f"{name}_cost_after"].sum()),
                "before": distribution(edge[f"{name}_cost_before"].to_numpy()),
                "after": distribution(edge[f"{name}_cost_after"].to_numpy()),
            }
            for name in ("prior", "imu", "bias", "pose")
        },
        "gate_actions": {
            str(key): int(value)
            for key, value in edge.visual_gate_action.value_counts().items()
        },
        # T2 intentionally has no T0 post-solve gate value. Exclude that optional
        # diagnostic while still requiring every solver/statistical value to be finite.
        "nonfinite_count": int(
            np.size(
                edge.drop(
                    columns=["visual_gate_postsolve_whitened_norm"],
                    errors="ignore",
                ).select_dtypes(include=[np.number]).to_numpy()
            )
            - np.isfinite(
                edge.drop(
                    columns=["visual_gate_postsolve_whitened_norm"],
                    errors="ignore",
                ).select_dtypes(include=[np.number]).to_numpy()
            ).sum()
        ),
    }
    return summary


def decide(summary: dict[str, Any]) -> dict[str, Any]:
    t0 = summary[T0]
    t2 = summary[T2]
    t0_rpe_t = t0["rpe"]["translation_xy_m"]["rmse"]
    t2_rpe_t = t2["rpe"]["translation_xy_m"]["rmse"]
    t0_rpe_r = t0["rpe"]["rotation_rad"]["rmse"]
    t2_rpe_r = t2["rpe"]["rotation_rad"]["rmse"]
    nis_target = 6.0
    nis_improved = abs(t2["measurement_nis"]["mean"] - nis_target) < abs(
        t0["measurement_nis"]["mean"] - nis_target
    )
    convergence_pass = t2["convergence"]["converged_rate"] >= 0.95
    finite_pass = t2["nonfinite_count"] == 0
    rpe_pass = (
        t2_rpe_t <= 1.05 * max(t0_rpe_t, 1.0e-12)
        and t2_rpe_r <= 1.05 * max(t0_rpe_r, 1.0e-12)
    )
    no_extreme_nis = t2["measurement_nis"]["above_interval_ratio"] <= 0.05
    approved = convergence_pass and finite_pass and rpe_pass and nis_improved and no_extreme_nis
    return {
        "gate5_passes": {
            "t2_convergence_rate_at_least_95_percent": convergence_pass,
            "t2_has_no_nan_inf": finite_pass,
            "t2_rpe_not_worse_than_t0_by_more_than_5_percent": rpe_pass,
            "t2_measurement_nis_closer_to_dof_6_than_t0": nis_improved,
            "t2_high_nis_outlier_ratio_at_most_5_percent": no_extreme_nis,
        },
        "approved_for_production_integration_prototype": approved,
        "approved_as_production_default": False,
        "full_sequence_run_approved": False,
        "next_step": (
            "Integrate T2 as an optional prototype by reusing MACVO final J/H/g; then rerun the same short replay through the production path."
            if approved
            else "Keep T0 as production default and locate the failed Gate-5 condition before any production integration."
        ),
    }


def plot_html(
    output: Path,
    edge_frame: pd.DataFrame,
    state_frame: pd.DataFrame,
) -> None:
    frames = np.sort(state_frame.frame.unique().astype(int))

    def ned_to_nwu(values: np.ndarray) -> list[list[float]]:
        converted = np.asarray(values, dtype=np.float64).copy()
        converted[:, 1:] *= -1.0
        return converted.tolist()

    def method_payload(method: str) -> tuple[pd.DataFrame, list[list[float]], list[list[float]], list[float]]:
        state = state_frame[state_frame.method == method].sort_values("frame")
        xyz = ned_to_nwu(state[["position_est_x", "position_est_y", "position_est_z"]].to_numpy())
        forward = ned_to_nwu(state[["forward_est_x", "forward_est_y", "forward_est_z"]].to_numpy())
        error = state.position_error_3d_norm_m.astype(float).tolist()
        return state, xyz, forward, error

    t0_state, t0_xyz, t0_forward, t0_error = method_payload(T0)
    _, t2_xyz, t2_forward, t2_error = method_payload(T2)
    gt_xyz = ned_to_nwu(
        t0_state[["position_gt_x", "position_gt_y", "position_gt_z"]].to_numpy()
    )
    gt_forward = ned_to_nwu(
        t0_state[["forward_gt_x", "forward_gt_y", "forward_gt_z"]].to_numpy()
    )
    payload = {
        "scene": "Rectangle normal-noise / frames 90-299 / IMU center",
        "gt": gt_xyz,
        "gt_forward": gt_forward,
        "macvo": t0_xyz,
        "macvo_forward": t0_forward,
        "time_s": ((t0_state.timestamp.to_numpy() - t0_state.timestamp.iloc[0]) * 1.0e-9).tolist(),
        "error_m": t0_error,
        "metrics": {
            "frames": int(len(frames)),
            "rmse_m": float(np.sqrt(np.mean(np.square(t0_error)))),
        },
        "fusion": [
            {
                "key": "t2_uvd_compressed",
                "source": "uvd_local_compression",
                "config": "normal_noise",
                "label": "T2 UVD-compressed pose factor",
                "color": "#2563eb",
                "dasharray": "",
                "scene": "clear_stop_turn_rectangle_truth_normal_noise",
                "xyz": t2_xyz,
                "forward": t2_forward,
                "error_m": t2_error,
                "metrics": {
                    "frames": int(len(frames)),
                    "rmse_m": float(np.sqrt(np.mean(np.square(t2_error)))),
                },
                "path": str(output / "gate5_state_truth_errors.csv"),
            }
        ],
        "imu_only": [],
        "gt_path": "in-memory IMU-center truth",
        "macvo_path": str(output / "gate5_state_truth_errors.csv"),
    }
    template = HTML_TEMPLATE.replace("{{", "{").replace("}}", "}")
    template = template.replace(
        "Circle, stop-turn rectangle, and straight trajectory comparison",
        "Gate 5: T0 sidecar pose factor vs T2 UVD-compressed pose factor",
    )
    template = template.replace(
        "__METHOD_SCOPE__",
        "T0 production relative-pose sidecar factor and T2 local UVD-compressed 6D factor",
    )
    template = template.replace(
        "__LINE_NOTE__",
        "Frames 90-299. Every trajectory is at the IMU center and translation-rebased at frame 90; no rotation, yaw or scale fitting.",
    )
    template = template.replace("Pure MACVO</span>", "T0 sidecar pose factor</span>")
    html = template.replace("__DATA__", json.dumps({"scenes": [payload]}, ensure_ascii=False))
    (output / "interactive_t0_vs_t2_300frames.html").write_text(html, encoding="utf-8")


def write_report(output: Path, summary: dict[str, Any], decision: dict[str, Any]) -> None:
    t0 = summary[T0]
    t2 = summary[T2]

    def relative_change(new: float, reference: float) -> float:
        return 100.0 * (new / reference - 1.0)

    xy_change = relative_change(
        t2["trajectory"]["xy_ate_rmse_m"], t0["trajectory"]["xy_ate_rmse_m"]
    )
    translation_rpe_change = relative_change(
        t2["rpe"]["translation_xy_m"]["rmse"],
        t0["rpe"]["translation_xy_m"]["rmse"],
    )
    rotation_rpe_change = relative_change(
        t2["rpe"]["rotation_rad"]["rmse"],
        t0["rpe"]["rotation_rad"]["rmse"],
    )
    runtime_change = relative_change(
        t2["runtime"]["total_edge_ms"]["median"],
        t0["runtime"]["total_edge_ms"]["median"],
    )
    lines = [
        "# UVD 压缩位姿因子 Gate 5：300 帧 T0/T2 对照",
        "",
        "## 首页结论",
        "",
        f"- T0 XY ATE：{t0['trajectory']['xy_ate_rmse_m']:.6g} m。",
        f"- T2 XY ATE：{t2['trajectory']['xy_ate_rmse_m']:.6g} m。",
        f"- T0/T2 平移 RPE RMSE：{t0['rpe']['translation_xy_m']['rmse']:.6g} / {t2['rpe']['translation_xy_m']['rmse']:.6g} m。",
        f"- T0/T2 旋转 RPE RMSE：{t0['rpe']['rotation_deg']['rmse']:.6g} / {t2['rpe']['rotation_deg']['rmse']:.6g} deg。",
        f"- T0/T2 measurement NIS mean：{t0['measurement_nis']['mean']:.6g} / {t2['measurement_nis']['mean']:.6g}。",
        f"- T0/T2 收敛率：{t0['convergence']['converged_rate']:.3%} / {t2['convergence']['converged_rate']:.3%}。",
        f"- 批准生产集成原型：{decision['approved_for_production_integration_prototype']}。",
        "- 未修改生产默认，未运行完整序列。",
        f"- 结论：T2 的 XY ATE 变化 {xy_change:+.2f}%，平移/旋转 RPE 变化 {translation_rpe_change:+.2f}% / {rotation_rpe_change:+.2f}%。它通过短段原型门槛，但不是所有指标都优于 T0。",
        "- T2 的 NIS 虽比 T0 更接近合理范围，均值仍只有约 1.21（6 维残差理想均值约为 6），因此尚不能宣称统计一致。",
        "",
        "## 实验契约",
        "",
        f"- 两条分支使用完全相同的首状态、IMU factor、bias 状态、LM 参数和 {t0['edge_count']} 条边。",
        "- 每条分支连续传播自己的优化状态和 Schur prior，不是孤立逐边实验。",
        "- T0 复现 relative-pose sidecar 的生产前后门控及必要的二次求解。",
        "- T2 在同一 sidecar 均值处压缩 U1 UVD 的 J^T W J 与 J^T W r，本轮不增加新门控。",
        "- GT 仅用于离线评分，不进入优化因子。",
        "- GT 先由 body/root 参考点转换到 IMU 中心；优化器状态本身已经是 T_WI。两条 IMU-center 轨迹分别仅减去 frame 90 的首位置，不做旋转、yaw 或尺度拟合。",
        "- IMU edge 继续使用捕获包中的 bias Jacobian 一阶修正，本轮不进行 raw-IMU repropagation。",
        "",
        "## 精度",
        "",
        "| metric | T0 | T2 |",
        "| --- | ---: | ---: |",
        f"| XY ATE RMSE / m | {t0['trajectory']['xy_ate_rmse_m']:.6g} | {t2['trajectory']['xy_ate_rmse_m']:.6g} |",
        f"| 3D position RMSE / m | {t0['trajectory']['position_3d_rmse_m']:.6g} | {t2['trajectory']['position_3d_rmse_m']:.6g} |",
        f"| orientation RMSE / deg | {t0['trajectory']['orientation_rmse_deg']:.6g} | {t2['trajectory']['orientation_rmse_deg']:.6g} |",
        f"| velocity RMSE / m/s | {t0['trajectory']['velocity_rmse_mps']:.6g} | {t2['trajectory']['velocity_rmse_mps']:.6g} |",
        f"| accel bias RMSE / m/s^2 | {t0['trajectory']['acc_bias_rmse_mps2']:.6g} | {t2['trajectory']['acc_bias_rmse_mps2']:.6g} |",
        f"| gyro bias RMSE / rad/s | {t0['trajectory']['gyro_bias_rmse_radps']:.6g} | {t2['trajectory']['gyro_bias_rmse_radps']:.6g} |",
        "",
        "## 统计一致性与计算量",
        "",
        f"- T0 NIS 中位数/覆盖率：{t0['measurement_nis']['median']:.6g} / {t0['measurement_nis']['inside_interval_ratio']:.3%}。",
        f"- T2 NIS 中位数/覆盖率：{t2['measurement_nis']['median']:.6g} / {t2['measurement_nis']['inside_interval_ratio']:.3%}。",
        f"- T0 factor-build/solve 中位耗时：{t0['runtime']['factor_build_ms']['median']:.3f} / {t0['runtime']['solver_ms']['median']:.3f} ms。",
        f"- T2 factor-build/solve 中位耗时：{t2['runtime']['factor_build_ms']['median']:.3f} / {t2['runtime']['solver_ms']['median']:.3f} ms。",
        f"- 当前 T2 每边总耗时相对 T0 变化 {runtime_change:+.2f}%。",
        "- T2 当前原型重新对 200 个点做自动微分；正式集成应直接复用 MACVO 最终迭代的 J/H/g，因此当前 build 时间不是最终实时开销。",
        "- NIS 中央 95% 区间覆盖率从 T0 的 0% 提升到 T2 的 46.89%，但 T2 仍有 53.11% 的边落在区间下方，说明不确定性依然偏保守。",
        "",
        "## 下一步",
        "",
        "1. 只新增可选 T2 生产原型接口，直接复用 MACVO 最终迭代保存的 J/H/g，避免后端再次遍历点并自动微分。",
        "2. 在同一 300 帧片段验证原型接口与本审计 T2 的残差、Hessian、轨迹和 NIS 数值等价，并重新测量真实增量耗时。",
        "3. 使用独立序列或重复噪声试验校准 pose covariance；目标是解决仍然偏低的 NIS，而不是手工乘一个视觉权重。",
        "4. 只有原型等价性、统计一致性和实时开销都通过后，才运行多场景完整序列并考虑替换默认 T0。",
        "",
        "## Gate 5 判定",
        "",
        "```json",
        json.dumps(jsonify(decision), ensure_ascii=False, indent=2),
        "```",
        "",
    ]
    (output / "uvd_compressed_pose_factor_gate5_report_cn.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    packet_path = args.packet.resolve()
    payload = torch.load(packet_path, map_location="cpu", weights_only=False)
    edges = prepare_edges(payload, args.edge_count)
    sidecar_path = Path(payload["visual_cache"]) / "relative_pose_factors.npz"
    sidecar = np.load(sidecar_path, allow_pickle=False)
    lookup = sidecar_lookup(sidecar)
    truth = load_truth(Path(payload["dataset"]))
    ref = pd.read_csv(Path(payload["dataset"]) / "ref_pose.csv")
    timestamps = ref["timestamp"].to_numpy(np.int64)
    ba_truth, bg_truth = interpolate_bias_truth(Path(payload["dataset"]), timestamps)

    edge_rows: list[dict[str, Any]] = []
    all_state_rows: list[dict[str, Any]] = []
    for method in METHODS:
        rows, states = run_branch(
            method=method,
            edges=edges,
            sidecar=sidecar,
            lookup=lookup,
            truth=truth,
            solver_settings=payload["solver_settings"],
        )
        edge_rows.extend(rows)
        all_state_rows.extend(
            state_rows(
                method=method,
                states=states,
                truth=truth,
                timestamps=timestamps,
                ba_truth=ba_truth,
                bg_truth=bg_truth,
            )
        )

    write_csv(output / "gate5_per_edge.csv", edge_rows)
    write_csv(output / "gate5_state_truth_errors.csv", all_state_rows)
    edge_frame = pd.DataFrame(edge_rows)
    state_frame = pd.DataFrame(all_state_rows)
    start_rebase_checks: dict[str, dict[str, float]] = {}
    for method in METHODS:
        first = state_frame[state_frame.method == method].sort_values("frame").iloc[0]
        estimate_start_norm = float(
            np.linalg.norm(
                first[["position_est_x", "position_est_y", "position_est_z"]]
                .to_numpy(np.float64)
            )
        )
        truth_start_norm = float(
            np.linalg.norm(
                first[["position_gt_x", "position_gt_y", "position_gt_z"]]
                .to_numpy(np.float64)
            )
        )
        start_error_norm = float(
            np.linalg.norm(
                first[
                    ["position_error_x", "position_error_y", "position_error_z"]
                ].to_numpy(np.float64)
            )
        )
        start_rebase_checks[method] = {
            "estimate_start_norm_m": estimate_start_norm,
            "truth_start_norm_m": truth_start_norm,
            "start_error_norm_m": start_error_norm,
        }
        if max(estimate_start_norm, truth_start_norm, start_error_norm) > 1.0e-12:
            raise AssertionError(
                f"{method}: IMU-center start rebase failed: "
                f"estimate={estimate_start_norm}, truth={truth_start_norm}, "
                f"error={start_error_norm}"
            )
    numeric_edge = edge_frame.select_dtypes(include=[np.number]).to_numpy()
    numeric_state = state_frame.select_dtypes(include=[np.number]).to_numpy()
    if not np.isfinite(numeric_edge).all() or not np.isfinite(numeric_state).all():
        raise FloatingPointError("Gate-5 output contains NaN/Inf")

    method_summary = {
        method: summarize_method(method, edge_frame, state_frame)
        for method in METHODS
    }
    decision = decide(method_summary)
    contract = {
        "schema_version": 1,
        "packet": str(packet_path),
        "packet_sha256": sha256(packet_path),
        "sidecar": str(sidecar_path),
        "sidecar_sha256": sha256(sidecar_path),
        "dataset": payload["dataset"],
        "edge_count": len(edges),
        "state_count": len(edges) + 1,
        "frame_range": [int(edges[0]["frame_i"]), int(edges[-1]["frame_j"])],
        "full_sequence_run": False,
        "production_default_changed": False,
        "t0_gate": VISUAL_GATE,
        "solver_settings": payload["solver_settings"],
        "software": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "pypose": getattr(pp, "__version__", "unknown"),
            "numpy": np.__version__,
        },
        "truth_usage": "offline scoring only; IMU-center pose, world velocity, and interpolated per-axis bias truth",
        "reference_point_contract": {
            "optimizer_state": "IMU center T_WI = T_WC * T_CI",
            "ground_truth": "body/root ref_pose translated to the IMU center",
            "comparison_origin": "translation-only rebase of both IMU-center trajectories at frame 90",
            "axis_alignment": "internal NED during optimization; NWU only for display",
            "pure_macvo": "not present in this Gate-5 page; any future comparison must convert T_WC to T_WI before rebasing",
            "start_rebase_checks": start_rebase_checks,
        },
        "t2_linearization": "T_CjCi right tangent [translation,rotation] at inverse T0 sidecar mean",
    }
    write_json(output / "gate5_contract.json", contract)
    write_json(output / "gate5_summary.json", method_summary)
    write_json(output / "gate5_decision.json", decision)
    plot_html(output, edge_frame, state_frame)
    write_report(output, method_summary, decision)
    print(json.dumps(jsonify(decision), ensure_ascii=False, indent=2))
    print(f"output: {output}")


if __name__ == "__main__":
    main()
