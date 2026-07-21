#!/usr/bin/env python3
"""Reproduce the low-rank SA-v2 Schur failure and rank-aware fallback."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pypose as pp
import torch


ROOT = Path("/home/admin1/macvo-dev")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Utility.TwoStateSamplingAwareVIO import (  # noqa: E402
    CrossEdgeImuFactor,
    CrossEdgeTwoStateProblem,
    CrossEdgeTwoStateSolver,
    make_cross_edge_diagonal_prior,
)
from Utility.TwoStateVIO import (  # noqa: E402
    ImuPreintegrationFactor,
    NavigationState,
    RelativePoseFactor,
)


DTYPE = torch.float64


def state() -> NavigationState:
    pose = pp.identity_SE3(1, dtype=DTYPE).tensor()
    return NavigationState(
        pose_WB=pose,
        velocity_W=torch.zeros(3, dtype=DTYPE),
        acc_bias=torch.zeros(3, dtype=DTYPE),
        gyro_bias=torch.zeros(3, dtype=DTYPE),
    )


def make_problem() -> CrossEdgeTwoStateProblem:
    generator = torch.Generator().manual_seed(83)
    state_i = state()
    tangent = torch.tensor(
        [[2.0e-5, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=DTYPE
    )
    state_j = NavigationState(
        pose_WB=(pp.SE3(state_i.pose_WB) @ pp.se3(tangent).Exp()).tensor(),
        velocity_W=state_i.velocity_W.clone(),
        acc_bias=state_i.acc_bias.clone(),
        gyro_bias=state_i.gyro_bias.clone(),
    )
    incoming = torch.randn((9, 6), generator=generator, dtype=DTYPE) * 1.0e-4
    outgoing = torch.randn((9, 6), generator=generator, dtype=DTYPE) * 1.0e-4
    unique = torch.diag(torch.tensor([1.0e-5] * 6 + [0.0] * 3, dtype=DTYPE))
    total = unique + incoming @ incoming.mT + outgoing @ outgoing.mT
    base = ImuPreintegrationFactor(
        delta_rotation=torch.zeros(3, dtype=DTYPE),
        delta_velocity=torch.zeros(3, dtype=DTYPE),
        delta_position=torch.zeros(3, dtype=DTYPE),
        covariance=total,
        dt=0.01,
        bias_jacobian=torch.zeros((9, 6), dtype=DTYPE),
        linearized_acc_bias=torch.zeros(3, dtype=DTYPE),
        linearized_gyro_bias=torch.zeros(3, dtype=DTYPE),
        bias_rw_covariance=torch.eye(6, dtype=DTYPE) * 1.0e-6,
        gravity_handling="preintegration",
    )
    imu = CrossEdgeImuFactor(
        base=base,
        unique_covariance=unique,
        incoming_raw_time_ns=torch.tensor([0], dtype=torch.long),
        outgoing_raw_time_ns=torch.tensor([1], dtype=torch.long),
        incoming_sensitivity=incoming,
        outgoing_sensitivity=outgoing,
    )
    prior = make_cross_edge_diagonal_prior(
        state_i,
        imu.incoming_raw_time_ns,
        pose_translation_std=0.1,
        pose_rotation_std=0.1,
        velocity_std=0.1,
        acc_bias_std=0.1,
        gyro_bias_std=0.1,
    )
    visual = RelativePoseFactor(
        measurement_BiBj=pp.identity_SE3(1, dtype=DTYPE).tensor(),
        covariance=torch.eye(6, dtype=DTYPE),
        huber_delta=3.0,
    )
    return CrossEdgeTwoStateProblem(
        state_i=state_i,
        state_j=state_j,
        noise_i=torch.zeros(6, dtype=DTYPE),
        noise_j=torch.zeros(6, dtype=DTYPE),
        prior_i=prior,
        imu=imu,
        visual=visual,
    )


def summarize(result) -> dict[str, object]:
    marginal = result.marginalization_diagnostics
    return {
        "rank_aware_requested": bool(result.rank_aware_imu_whitening),
        "rank_aware_fallback_active": bool(result.rank_aware_fallback_active),
        "imu_residual_dimension": int(result.rank_aware_imu_residual_dimension),
        "unique_covariance_rank": int(
            result.unique_covariance_diagnostics.effective_rank
        ),
        "unique_covariance_dimension": int(
            result.unique_covariance_diagnostics.dimension
        ),
        "h_mm_min_eigenvalue": marginal.h_mm.min_eigenvalue,
        "h_mm_max_eigenvalue": marginal.h_mm.max_eigenvalue,
        "h_mm_condition_number": marginal.h_mm.condition_number,
        "schur_prior_min_eigenvalue": marginal.schur_prior.min_eigenvalue,
        "schur_prior_max_eigenvalue": marginal.schur_prior.max_eigenvalue,
        "schur_prior_condition_number": marginal.schur_prior.condition_number,
        "schur_quadratic_relative_error": marginal.quadratic_relative_error,
        "cross_covariance_frobenius_norm": (
            result.cross_covariance_frobenius_norm
        ),
        "common_translation_update_world_norm": float(
            torch.linalg.vector_norm(result.common_translation_update_world)
        ),
        "final_cost": result.final_cost,
        "iterations": result.iterations,
        "converged": result.converged,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "analysis_circle_sa_v2_prior_rank_aware_20260717"
        / "synthetic_rank_aware_audit.json",
    )
    args = parser.parse_args()
    problem = make_problem()
    legacy = CrossEdgeTwoStateSolver(max_iterations=6).solve(problem)
    rank_aware = CrossEdgeTwoStateSolver(
        max_iterations=6,
        rank_aware_imu_whitening=True,
    ).solve(problem)
    payload = {
        "contract": {
            "unique_covariance_rank": "6/9",
            "legacy": "hard eigenvalue floor on singular P_unique",
            "rank_aware": (
                "use full per-edge P_total and zero endpoint sensitivities when "
                "P_unique is rank deficient"
            ),
        },
        "legacy": summarize(legacy),
        "rank_aware": summarize(rank_aware),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(args.output)


if __name__ == "__main__":
    main()
