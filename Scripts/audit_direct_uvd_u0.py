#!/usr/bin/env python3
"""U0: verify the new direct-UVD factor against MACVO's production disp objective."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pypose as pp
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import Scripts.audit_macvo_translation_uvd_point_level as production  # noqa: E402
from Utility.TwoStateVIO import (  # noqa: E402
    NavigationState,
    UVDFactor,
    retract_state,
    visual_whitened_residuals,
)
from Utility.VisualFactorCache import VisualFactorCacheReader  # noqa: E402


CACHE = ROOT / "VisualCache/static63_unique_visual_20260713/clear_circle_truth_normal_noise"
OUT = ROOT / "analysis_direct_uvd_20260716/U0_fixed_source"
START = 90
STOP = 391
DTYPE = torch.float64


def state(pose: torch.Tensor) -> NavigationState:
    return NavigationState(
        pose_WB=pose.reshape(1, 7).to(dtype=DTYPE),
        velocity_W=torch.zeros(3, dtype=DTYPE),
        acc_bias=torch.zeros(3, dtype=DTYPE),
        gyro_bias=torch.zeros(3, dtype=DTYPE),
    )


def make_factor(packet) -> UVDFactor:
    covariance = torch.from_numpy(production.observation_covariance(packet)).to(DTYPE)
    return UVDFactor(
        points_Ci=packet.points_local.to(DTYPE),
        target_uv=packet.match_fields["pixel2_uv"].to(DTYPE),
        target_disparity=packet.match_fields["pixel2_disp"].to(DTYPE),
        covariance_uvd=covariance,
        intrinsic=packet.K.to(DTYPE),
        baseline=float(packet.baseline_m),
        extrinsic_CI=pp.identity_SE3(1, dtype=DTYPE).tensor(),
        huber_delta=production.HUBER_DELTA,
    )


def factor_raw(state_i: NavigationState, state_j: NavigationState, factor: UVDFactor) -> torch.Tensor:
    relative = pp.SE3(state_j.pose_WB).Inv() @ pp.SE3(state_i.pose_WB)
    points_j = relative.Act(factor.points_Ci)
    K = factor.intrinsic
    predicted = torch.stack(
        [
            K[0, 0] * points_j[:, 1] / points_j[:, 0] + K[0, 2],
            K[1, 1] * points_j[:, 2] / points_j[:, 0] + K[1, 2],
            K[0, 0] * float(factor.baseline) / points_j[:, 0],
        ],
        dim=-1,
    )
    observation = torch.cat([factor.target_uv, factor.target_disparity], dim=-1)
    return predicted - observation


def jacobian_check(
    state_i: NavigationState,
    state_j: NavigationState,
    factor: UVDFactor,
) -> tuple[float, float, float]:
    zero = torch.zeros(6, dtype=DTYPE, requires_grad=True)

    def evaluate(increment: torch.Tensor) -> torch.Tensor:
        candidate = retract_state(
            state_j,
            torch.cat([increment, torch.zeros(9, dtype=DTYPE)]),
        )
        return visual_whitened_residuals(
            state_i, candidate, factor, 1.0e-12
        ).reshape(-1)

    autodiff = torch.autograd.functional.jacobian(evaluate, zero, vectorize=True)
    epsilon = 1.0e-4
    numerical = torch.empty_like(autodiff)
    for column in range(6):
        step = torch.zeros(6, dtype=DTYPE)
        step[column] = epsilon
        numerical[:, column] = (evaluate(step) - evaluate(-step)) / (2.0 * epsilon)
    difference = (autodiff - numerical).abs()
    mask = numerical.abs() > 1.0e-4
    relative = difference[mask] / numerical[mask].abs() if bool(mask.any()) else difference[:0]
    normalized_absolute = difference.max() / autodiff.abs().max().clamp_min(1.0)
    return (
        float(difference.max()),
        float(relative.max()) if relative.numel() else 0.0,
        float(normalized_absolute),
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    reader = VisualFactorCacheReader(CACHE)
    source_result = Path(reader.manifest.source["result"])
    with np.load(source_result / "tensor_map.npz", allow_pickle=False) as stream:
        poses = torch.from_numpy(stream["frames//pose"].copy()).to(DTYPE)
    with np.load(CACHE / "relative_pose_factors.npz", allow_pickle=False) as stream:
        sidecar = stream["measurement_CiCj"].copy()

    check_edges = set(np.linspace(START, STOP - 2, 20, dtype=int).tolist())
    rows: list[dict[str, float | int | bool]] = []
    for frame_i in range(START, STOP - 1):
        frame_j = frame_i + 1
        packet = reader.load_pair(
            frame_i,
            frame_j,
            int(reader.manifest.timestamps_ns[frame_i]),
            int(reader.manifest.timestamps_ns[frame_j]),
        )
        factor = make_factor(packet)
        state_i = state(poses[frame_i])
        state_j = state(poses[frame_j])
        z_from_poses = pp.SE3(poses[frame_i : frame_i + 1]).Inv() @ pp.SE3(
            poses[frame_j : frame_j + 1]
        )
        z_sidecar = pp.SE3(torch.from_numpy(sidecar[frame_i].reshape(1, 7)).to(DTYPE))
        sidecar_error = float(
            torch.linalg.vector_norm((z_sidecar.Inv() @ z_from_poses).Log().tensor())
        )

        points = packet.points_local.numpy().astype(np.float64)
        observation = np.column_stack(
            [
                packet.match_fields["pixel2_uv"].numpy(),
                packet.match_fields["pixel2_disp"].numpy().reshape(-1),
            ]
        )
        covariance = production.observation_covariance(packet)
        z_matrix = production.se3_from_xyzw(sidecar[frame_i].reshape(7))
        reference_terms = production.uvd_terms(
            z_matrix,
            points,
            observation,
            covariance,
            packet.K.numpy().astype(np.float64),
            float(packet.baseline_m),
            use_covariance=True,
            robust=True,
        )
        direct_raw = factor_raw(state_i, state_j, factor).detach().numpy()
        direct_white = visual_whitened_residuals(
            state_i, state_j, factor, 1.0e-12
        ).detach().numpy()
        direct_norm = np.linalg.norm(direct_white, axis=1)
        direct_cost = np.where(
            direct_norm < production.HUBER_DELTA,
            0.5 * direct_norm**2,
            production.HUBER_DELTA * direct_norm - 0.5 * production.HUBER_DELTA**2,
        ).sum()

        selected = np.ones(len(points), dtype=bool)
        refined, diagnostics = production.optimize(
            z_matrix,
            points,
            observation,
            covariance,
            packet.K.numpy().astype(np.float64),
            float(packet.baseline_m),
            selected,
            fixed_rotation=None,
            use_covariance=True,
            robust=True,
        )
        refined_lie = pp.from_matrix(torch.from_numpy(refined).to(DTYPE), pp.SE3_type)
        refit_delta = (z_sidecar.Inv() @ refined_lie).Log().tensor().reshape(6)
        jac_abs = jac_rel = jac_normalized = np.nan
        if frame_i in check_edges:
            jac_abs, jac_rel, jac_normalized = jacobian_check(state_i, state_j, factor)
        rows.append(
            {
                "frame_i": frame_i,
                "frame_j": frame_j,
                "num_points": len(points),
                "sidecar_vs_pose_se3_norm": sidecar_error,
                "raw_residual_max_abs_difference": float(
                    np.max(np.abs(direct_raw - reference_terms["residual"]))
                ),
                "white_residual_max_abs_difference": float(
                    np.max(np.abs(direct_white - reference_terms["white_residual"]))
                ),
                "robust_cost_abs_difference": float(
                    abs(float(direct_cost) - float(reference_terms["cost"].sum()))
                ),
                "jacobian_max_abs_error": jac_abs,
                "jacobian_max_relative_error": jac_rel,
                "jacobian_normalized_abs_error": jac_normalized,
                "refit_translation_change_m": float(torch.linalg.vector_norm(refit_delta[:3])),
                "refit_rotation_change_rad": float(torch.linalg.vector_norm(refit_delta[3:])),
                "refit_cost_before": float(diagnostics["initial_cost"]),
                "refit_cost_after": float(diagnostics["final_cost"]),
                "refit_iterations": int(diagnostics["iterations"]),
                "refit_converged": bool(diagnostics["converged"]),
            }
        )

    table = pd.DataFrame(rows)
    table.to_csv(OUT / "u0_per_edge.csv", index=False)
    jacobian = table.dropna(subset=["jacobian_max_abs_error"])
    summary = {
        "edge_count": int(len(table)),
        "point_count_min": int(table.num_points.min()),
        "point_count_max": int(table.num_points.max()),
        "sidecar_vs_pose_se3_norm_max": float(table.sidecar_vs_pose_se3_norm.max()),
        "raw_residual_max_abs_difference": float(table.raw_residual_max_abs_difference.max()),
        "white_residual_max_abs_difference": float(table.white_residual_max_abs_difference.max()),
        "robust_cost_abs_difference_max": float(table.robust_cost_abs_difference.max()),
        "jacobian_checked_edges": int(len(jacobian)),
        "jacobian_max_abs_error": float(jacobian.jacobian_max_abs_error.max()),
        "jacobian_max_relative_error": float(jacobian.jacobian_max_relative_error.max()),
        "jacobian_normalized_abs_error": float(
            jacobian.jacobian_normalized_abs_error.max()
        ),
        "refit_translation_change_median": float(table.refit_translation_change_m.median()),
        "refit_translation_change_p95": float(table.refit_translation_change_m.quantile(0.95)),
        "refit_rotation_change_median": float(table.refit_rotation_change_rad.median()),
        "refit_rotation_change_p95": float(table.refit_rotation_change_rad.quantile(0.95)),
        "refit_converged_rate": float(table.refit_converged.mean()),
        "acceptance": {
            "same_raw_residual": bool(table.raw_residual_max_abs_difference.max() <= 1.0e-6),
            "same_whitened_residual": bool(table.white_residual_max_abs_difference.max() <= 1.0e-6),
            "same_robust_cost": bool(table.robust_cost_abs_difference.max() <= 1.0e-6),
            "jacobian_scale_normalized": bool(
                jacobian.jacobian_normalized_abs_error.max() <= 1.0e-5
                and jacobian.jacobian_max_relative_error.max() <= 1.0e-3
            ),
            "finite": bool(
                np.isfinite(
                    table.drop(
                        columns=[
                            "jacobian_max_abs_error",
                            "jacobian_max_relative_error",
                            "jacobian_normalized_abs_error",
                        ]
                    ).select_dtypes(include=[np.number])
                ).all().all()
                and np.isfinite(jacobian.select_dtypes(include=[np.number])).all().all()
            ),
        },
        "scope_note": (
            "The original pre-optimization motion-model pose was not persisted. Refit starts at the "
            "stored pure-MACVO optimum; any movement measures original stopping/solver differences, "
            "not a different direct-UVD residual contract."
        ),
    }
    summary["passed"] = bool(all(summary["acceptance"].values()))
    (OUT / "u0_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
