#!/usr/bin/env python3
"""Gate 4: per-factor information and bias-lifecycle audit for short-circle V3 C0."""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import Scripts.run_circle_translation_oracles as base  # noqa: E402
import Scripts.run_v3_backend_information_oracles as experiment  # noqa: E402
from Utility.TwoStateVIO import (  # noqa: E402
    NavigationState,
    TwoStateVIOProblem,
    TwoStateVIOSolver,
    _linearize,
)


OUT = experiment.OUT
ROWS: list[dict[str, Any]] = []
CHECKS: list[dict[str, Any]] = []
PRODUCTION_SOLVE = TwoStateVIOSolver.solve

BLOCKS = {
    "pose_i_t": slice(0, 3),
    "pose_i_R": slice(3, 6),
    "v_i": slice(6, 9),
    "ba_i": slice(9, 12),
    "bg_i": slice(12, 15),
    "pose_j_t": slice(15, 18),
    "pose_j_R": slice(18, 21),
    "v_j": slice(21, 24),
    "ba_j": slice(24, 27),
    "bg_j": slice(27, 30),
}


def normalized_problem(problem: TwoStateVIOProblem) -> TwoStateVIOProblem:
    device = problem.state_i.pose_WB.device
    dtype = torch.float64
    return replace(
        problem,
        state_i=problem.state_i.to(device=device, dtype=dtype),
        state_j=problem.state_j.to(device=device, dtype=dtype),
        prior_i=problem.prior_i.to(device=device, dtype=dtype),
        imu=problem.imu.to(device=device, dtype=dtype),
        visual_pose=problem.visual_pose.to(device=device, dtype=dtype),
    )


def information_row(
    *,
    stage: str,
    factor: str,
    residual: torch.Tensor,
    jacobian: torch.Tensor,
) -> dict[str, Any]:
    residual = residual.detach()
    jacobian = jacobian.detach()
    hessian = 0.5 * (jacobian.mT @ jacobian + (jacobian.mT @ jacobian).mT)
    gradient = jacobian.mT @ residual
    eigenvalues = torch.linalg.eigvalsh(hessian).cpu().numpy()
    maximum = max(float(np.max(np.abs(eigenvalues))), 1.0)
    threshold = max(1.0e-10, np.finfo(np.float64).eps * maximum)
    positive = eigenvalues[eigenvalues > threshold]
    row: dict[str, Any] = {
        "frame_i": int(base.CURRENT_EDGE["i"]),
        "frame_j": int(base.CURRENT_EDGE["j"]),
        "stage": stage,
        "factor": factor,
        "residual_dim": int(residual.numel()),
        "cost": 0.5 * float(residual.square().sum().cpu().item()),
        "residual_norm": float(torch.linalg.vector_norm(residual).cpu().item()),
        "gradient_norm": float(torch.linalg.vector_norm(gradient).cpu().item()),
        "gradient_inf_norm": float(gradient.abs().max().cpu().item()),
        "hessian_trace": float(torch.trace(hessian).cpu().item()),
        "hessian_frobenius": float(torch.linalg.matrix_norm(hessian).cpu().item()),
        "hessian_rank": int(positive.size),
        "hessian_min_positive_eigenvalue": float(positive.min()) if positive.size else 0.0,
        "hessian_max_eigenvalue": float(eigenvalues.max()),
        "hessian_positive_condition": (
            float(positive.max() / positive.min()) if positive.size else float("inf")
        ),
    }
    hessian_np = hessian.cpu().numpy()
    gradient_np = gradient.cpu().numpy()
    names = list(BLOCKS)
    for name, block in BLOCKS.items():
        diagonal = hessian_np[block, block]
        row[f"Hdiag_trace__{name}"] = float(np.trace(diagonal))
        row[f"Hdiag_fro__{name}"] = float(np.linalg.norm(diagonal))
        row[f"gradient_norm__{name}"] = float(np.linalg.norm(gradient_np[block]))
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            row[f"Hcross_fro__{left}__{right}"] = float(
                np.linalg.norm(hessian_np[BLOCKS[left], BLOCKS[right]])
            )
    row["H_pose_velocity_cross_fro"] = float(
        sum(
            np.linalg.norm(hessian_np[BLOCKS[p], BLOCKS[v]])
            for p in ("pose_i_t", "pose_i_R", "pose_j_t", "pose_j_R")
            for v in ("v_i", "v_j")
        )
    )
    row["H_pose_acc_bias_cross_fro"] = float(
        sum(
            np.linalg.norm(hessian_np[BLOCKS[p], BLOCKS[b]])
            for p in ("pose_i_t", "pose_i_R", "pose_j_t", "pose_j_R")
            for b in ("ba_i", "ba_j")
        )
    )
    row["H_pose_gyro_bias_cross_fro"] = float(
        sum(
            np.linalg.norm(hessian_np[BLOCKS[p], BLOCKS[b]])
            for p in ("pose_i_t", "pose_i_R", "pose_j_t", "pose_j_R")
            for b in ("bg_i", "bg_j")
        )
    )
    row["H_velocity_acc_bias_cross_fro"] = float(
        sum(
            np.linalg.norm(hessian_np[BLOCKS[v], BLOCKS[b]])
            for v in ("v_i", "v_j")
            for b in ("ba_i", "ba_j")
        )
    )
    row["H_rotation_gyro_bias_cross_fro"] = float(
        sum(
            np.linalg.norm(hessian_np[BLOCKS[r], BLOCKS[b]])
            for r in ("pose_i_R", "pose_j_R")
            for b in ("bg_i", "bg_j")
        )
    )
    return row


def split_information(
    *, stage: str, state_i: NavigationState, state_j: NavigationState, problem: TwoStateVIOProblem
) -> None:
    residual, jacobian, hessian, gradient, _ = _linearize(
        state_i,
        state_j,
        problem.prior_i,
        problem.imu,
        problem.visual_pose,
        1.0e-12,
    )
    prior_dim = int(problem.prior_i.sqrt_information.shape[0])
    spans = {
        "prior": slice(0, prior_dim),
        "imu": slice(prior_dim, prior_dim + 9),
        "bias": slice(prior_dim + 9, prior_dim + 15),
        "pose": slice(prior_dim + 15, prior_dim + 21),
    }
    factor_hessians = []
    factor_gradients = []
    for factor, span in spans.items():
        factor_residual = residual[span]
        factor_jacobian = jacobian[span]
        ROWS.append(
            information_row(
                stage=stage,
                factor=factor,
                residual=factor_residual,
                jacobian=factor_jacobian,
            )
        )
        factor_hessians.append(factor_jacobian.mT @ factor_jacobian)
        factor_gradients.append(factor_jacobian.mT @ factor_residual)
    CHECKS.append(
        {
            "frame_i": int(base.CURRENT_EDGE["i"]),
            "frame_j": int(base.CURRENT_EDGE["j"]),
            "stage": stage,
            "hessian_sum_max_abs_error": float(
                (torch.stack(factor_hessians).sum(0) - hessian).abs().max().cpu().item()
            ),
            "gradient_sum_max_abs_error": float(
                (torch.stack(factor_gradients).sum(0) - gradient).abs().max().cpu().item()
            ),
        }
    )


def instrumented_solve(self: TwoStateVIOSolver, problem: TwoStateVIOProblem):
    normalized = normalized_problem(problem)
    split_information(
        stage="initial",
        state_i=normalized.state_i,
        state_j=normalized.state_j,
        problem=normalized,
    )
    result = PRODUCTION_SOLVE(self, problem)
    split_information(
        stage="final",
        state_i=result.state_i,
        state_j=result.state_j,
        problem=normalized,
    )
    return result


def build_bias_lifecycle() -> None:
    root = base.mode_root("short", "C0")
    trace = pd.read_csv(root / "factor_trace.csv")
    trace = (
        trace.sort_values(["frame_i", "solver_call"])
        .groupby(["frame_i", "frame_j"], as_index=False)
        .tail(1)
        .sort_values("frame_i")
        .reset_index(drop=True)
    )
    rows = []
    for index in range(len(trace) - 1):
        current = trace.iloc[index]
        following = trace.iloc[index + 1]
        row: dict[str, Any] = {
            "frame_state": int(current.frame_j),
            "produced_by_edge_i": int(current.frame_i),
            "consumed_by_edge_j": int(following.frame_j),
        }
        for sensor in ("ba", "bg"):
            produced = np.array([current[f"{sensor}_j_after_{axis}"] for axis in "xyz"], dtype=np.float64)
            consumed_before = np.array(
                [following[f"{sensor}_i_before_{axis}"] for axis in "xyz"], dtype=np.float64
            )
            consumed_after = np.array(
                [following[f"{sensor}_i_after_{axis}"] for axis in "xyz"], dtype=np.float64
            )
            for axis, value in zip("xyz", produced):
                row[f"{sensor}_produced_{axis}"] = float(value)
            for axis, value in zip("xyz", consumed_before):
                row[f"{sensor}_next_start_before_{axis}"] = float(value)
            for axis, value in zip("xyz", consumed_after):
                row[f"{sensor}_next_start_after_{axis}"] = float(value)
            row[f"{sensor}_carry_error_norm"] = float(np.linalg.norm(consumed_before - produced))
            row[f"{sensor}_next_reoptimization_norm"] = float(np.linalg.norm(consumed_after - consumed_before))
        rows.append(row)
    lifecycle = pd.DataFrame(rows)
    lifecycle.to_csv(OUT / "bias_lifecycle_per_frame.csv", index=False)
    summary = {
        "frame_count": int(len(lifecycle)),
        "ba_carry_error_max": float(lifecycle["ba_carry_error_norm"].max()),
        "bg_carry_error_max": float(lifecycle["bg_carry_error_norm"].max()),
        "ba_next_reoptimization_median": float(lifecycle["ba_next_reoptimization_norm"].median()),
        "ba_next_reoptimization_p95": float(lifecycle["ba_next_reoptimization_norm"].quantile(0.95)),
        "bg_next_reoptimization_median": float(lifecycle["bg_next_reoptimization_norm"].median()),
        "bg_next_reoptimization_p95": float(lifecycle["bg_next_reoptimization_norm"].quantile(0.95)),
    }
    (OUT / "bias_lifecycle_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def main() -> None:
    experiment.configure_base()
    original = base.ORIGINAL_SOLVE
    base.ORIGINAL_SOLVE = instrumented_solve
    try:
        experiment.run_case("C0", force=True)
    finally:
        base.ORIGINAL_SOLVE = original
        TwoStateVIOSolver.solve = PRODUCTION_SOLVE
    pd.DataFrame(ROWS).to_csv(OUT / "factor_hessian_gradient_per_edge.csv", index=False)
    pd.DataFrame(CHECKS).to_csv(OUT / "factor_information_reconstruction_checks.csv", index=False)
    build_bias_lifecycle()


if __name__ == "__main__":
    main()
