#!/usr/bin/env python3
"""Point-level P0-P7 counterfactuals using MACVO's production disp residual."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import Scripts.audit_macvo_translation_point_level as shared  # noqa: E402
from Scripts.audit_circle_translation_oracle import (  # noqa: E402
    DEFAULT_CACHE,
    DEFAULT_DATASET,
    invert_transform,
    load_truth,
    rotation_log,
    se3_from_xyzw,
)
from Utility.VisualFactorCache import VisualFactorCacheReader  # noqa: E402


OUT = ROOT / "analysis_circle_v3_backend_pointlevel_20260716/point_level_uvd"
SELECTION = ROOT / "analysis_circle_v3_backend_pointlevel_20260716/point_level/representative_edge_selection.csv"
HUBER_DELTA = 0.1
MODES = shared.MODES


def observation_covariance(packet) -> np.ndarray:
    uv = packet.match_fields["pixel2_uv_cov"].numpy().astype(np.float64)
    disparity = packet.match_fields["pixel2_disp_cov"].numpy().astype(np.float64).reshape(-1)
    covariance = np.zeros((len(uv), 3, 3), dtype=np.float64)
    covariance[:, 0, 0] = uv[:, 0]
    covariance[:, 1, 1] = uv[:, 1]
    covariance[:, 0, 1] = uv[:, 2]
    covariance[:, 1, 0] = uv[:, 2]
    covariance[:, 2, 2] = disparity
    values, vectors = np.linalg.eigh(0.5 * (covariance + np.swapaxes(covariance, 1, 2)))
    return np.einsum(
        "nij,nj,nkj->nik", vectors, np.maximum(values, 1.0e-12), vectors
    )


def uvd_terms(
    transform_cicj: np.ndarray,
    points_i: np.ndarray,
    observation: np.ndarray,
    covariance: np.ndarray,
    K: np.ndarray,
    baseline: float,
    *,
    use_covariance: bool,
    robust: bool,
) -> dict[str, np.ndarray]:
    inverse = invert_transform(transform_cicj)
    points_j = (inverse[:3, :3] @ points_i.T).T + inverse[:3, 3]
    x, y, z = points_j.T
    safe_x = np.where(np.abs(x) < 1.0e-9, np.sign(x) * 1.0e-9 + (x == 0) * 1.0e-9, x)
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    predicted = np.stack(
        [fx * y / safe_x + cx, fy * z / safe_x + cy, fx * baseline / safe_x], axis=1
    )
    residual = predicted - observation
    projection_jacobian = np.zeros((len(points_i), 3, 3), dtype=np.float64)
    projection_jacobian[:, 0, 0] = -fx * y / safe_x**2
    projection_jacobian[:, 0, 1] = fx / safe_x
    projection_jacobian[:, 1, 0] = -fy * z / safe_x**2
    projection_jacobian[:, 1, 2] = fy / safe_x
    projection_jacobian[:, 2, 0] = -fx * baseline / safe_x**2
    point_jacobian = np.concatenate(
        [-np.broadcast_to(np.eye(3), (len(points_i), 3, 3)), shared.skew(points_j)], axis=2
    )
    jacobian = np.einsum("nij,njk->nik", projection_jacobian, point_jacobian)
    if use_covariance:
        lower = np.linalg.cholesky(covariance)
        white_residual = np.linalg.solve(lower, residual[..., None])[..., 0]
        white_jacobian = np.linalg.solve(lower, jacobian)
    else:
        white_residual = residual.copy()
        white_jacobian = jacobian.copy()
    norms = np.linalg.norm(white_residual, axis=1)
    if robust:
        weights = np.where(norms < HUBER_DELTA, 1.0, HUBER_DELTA / np.maximum(norms, 1.0e-12))
        costs = np.where(
            norms < HUBER_DELTA,
            0.5 * norms**2,
            HUBER_DELTA * norms - 0.5 * HUBER_DELTA**2,
        )
    else:
        weights = np.ones_like(norms)
        costs = 0.5 * norms**2
    return {
        "points_j": points_j,
        "predicted": predicted,
        "residual": residual,
        "jacobian": jacobian,
        "white_residual": white_residual,
        "white_jacobian": white_jacobian,
        "norm": norms,
        "mahalanobis_sq": norms**2,
        "weight": weights,
        "cost": costs,
    }


def objective(terms: dict[str, np.ndarray], selected: np.ndarray) -> float:
    return float(np.sum(terms["cost"][selected]))


def optimize(
    initial: np.ndarray,
    points_i: np.ndarray,
    observation: np.ndarray,
    covariance: np.ndarray,
    K: np.ndarray,
    baseline: float,
    selected: np.ndarray,
    *,
    fixed_rotation: np.ndarray | None,
    use_covariance: bool,
    robust: bool,
) -> tuple[np.ndarray, dict[str, Any]]:
    transform = initial.copy()
    if fixed_rotation is not None:
        transform[:3, :3] = fixed_rotation
    damping = 1.0e-3
    terms = uvd_terms(
        transform, points_i, observation, covariance, K, baseline,
        use_covariance=use_covariance, robust=robust,
    )
    initial_cost = current_cost = objective(terms, selected)
    accepted = rejected = 0
    converged = False
    final_step = math.inf
    iterations = 0
    for iteration in range(50):
        iterations = iteration + 1
        terms = uvd_terms(
            transform, points_i, observation, covariance, K, baseline,
            use_covariance=use_covariance, robust=robust,
        )
        sqrt_weight = np.sqrt(terms["weight"][selected])[:, None, None]
        jacobian = sqrt_weight * terms["white_jacobian"][selected]
        residual = sqrt_weight[..., 0] * terms["white_residual"][selected]
        active_jacobian = jacobian[:, :, :3] if fixed_rotation is not None else jacobian
        hessian = np.einsum("nki,nkj->ij", active_jacobian, active_jacobian)
        gradient = np.einsum("nki,nk->i", active_jacobian, residual)
        system = hessian + damping * np.diag(np.maximum(np.abs(np.diag(hessian)), 1.0))
        try:
            step = np.linalg.solve(system, -gradient)
        except np.linalg.LinAlgError:
            step = np.linalg.pinv(system) @ (-gradient)
        final_step = float(np.linalg.norm(step))
        if final_step <= 1.0e-10:
            converged = True
            break
        if fixed_rotation is None:
            candidate = shared.right_update(transform, step)
        else:
            candidate = transform.copy()
            candidate[:3, 3] += fixed_rotation @ step
            candidate[:3, :3] = fixed_rotation
        candidate_terms = uvd_terms(
            candidate, points_i, observation, covariance, K, baseline,
            use_covariance=use_covariance, robust=robust,
        )
        candidate_cost = objective(candidate_terms, selected)
        if candidate_cost < current_cost:
            accepted += 1
            previous = current_cost
            transform = candidate
            current_cost = candidate_cost
            damping = max(damping * 0.25, 1.0e-12)
            if abs(previous - current_cost) <= 1.0e-12:
                converged = True
                break
        else:
            rejected += 1
            damping = min(damping * 10.0, 1.0e12)
    final_terms = uvd_terms(
        transform, points_i, observation, covariance, K, baseline,
        use_covariance=use_covariance, robust=robust,
    )
    sqrt_weight = np.sqrt(final_terms["weight"][selected])[:, None, None]
    jacobian = sqrt_weight * final_terms["white_jacobian"][selected]
    residual = sqrt_weight[..., 0] * final_terms["white_residual"][selected]
    gradient = np.einsum("nki,nk->i", jacobian, residual)
    return transform, {
        "iterations": iterations,
        "converged": converged,
        "accepted_steps": accepted,
        "rejected_steps": rejected,
        "initial_cost": initial_cost,
        "final_cost": objective(final_terms, selected),
        "final_step_norm": final_step,
        "final_gradient_norm": float(np.linalg.norm(gradient)),
    }


def jacobian_check(
    transform: np.ndarray, points_i: np.ndarray, observation: np.ndarray,
    covariance: np.ndarray, K: np.ndarray, baseline: float,
) -> dict[str, float]:
    analytical = uvd_terms(
        transform, points_i, observation, covariance, K, baseline,
        use_covariance=True, robust=True,
    )["jacobian"]
    epsilon = 1.0e-7
    max_abs = max_rel = 0.0
    for point in np.linspace(0, len(points_i) - 1, min(20, len(points_i)), dtype=int):
        numerical = np.zeros((3, 6))
        for column in range(6):
            delta = np.zeros(6)
            delta[column] = epsilon
            plus = uvd_terms(
                shared.right_update(transform, delta), points_i, observation, covariance, K, baseline,
                use_covariance=True, robust=True,
            )["residual"][point]
            minus = uvd_terms(
                shared.right_update(transform, -delta), points_i, observation, covariance, K, baseline,
                use_covariance=True, robust=True,
            )["residual"][point]
            numerical[:, column] = (plus - minus) / (2.0 * epsilon)
        difference = np.abs(analytical[point] - numerical)
        max_abs = max(max_abs, float(difference.max()))
        mask = np.abs(numerical) > 1.0e-5
        if np.any(mask):
            max_rel = max(max_rel, float(np.max(difference[mask] / np.abs(numerical[mask]))))
    return {"jacobian_epsilon": epsilon, "jacobian_max_abs_error": max_abs, "jacobian_max_relative_error": max_rel}


def point_rows(
    *, label: str, mode: str, packet, transform: np.ndarray, selected: np.ndarray,
    original_inlier: np.ndarray, use_covariance: bool, robust: bool,
    z_gt: np.ndarray, pose_wci: np.ndarray,
) -> list[dict[str, Any]]:
    points_i = packet.points_local.numpy().astype(np.float64)
    K = packet.K.numpy().astype(np.float64)
    observation = np.column_stack(
        [
            packet.match_fields["pixel2_uv"].numpy(),
            packet.match_fields["pixel2_disp"].numpy().reshape(-1),
        ]
    )
    covariance = observation_covariance(packet)
    terms = uvd_terms(
        transform, points_i, observation, covariance, K, packet.baseline_m,
        use_covariance=use_covariance, robust=robust,
    )
    gt_terms = uvd_terms(
        z_gt, points_i, observation, covariance, K, packet.baseline_m,
        use_covariance=True, robust=True,
    )
    sqrt_weight = np.sqrt(terms["weight"])[:, None, None]
    weighted_j = sqrt_weight * terms["white_jacobian"]
    weighted_r = sqrt_weight[..., 0] * terms["white_residual"]
    hessian = np.einsum("nki,nkj->nij", weighted_j, weighted_j)
    gradient = np.einsum("nki,nk->ni", weighted_j, weighted_r)
    gt_sqrt_weight = np.sqrt(gt_terms["weight"])[:, None, None]
    gt_weighted_j = gt_sqrt_weight * gt_terms["white_jacobian"]
    gt_weighted_r = gt_sqrt_weight[..., 0] * gt_terms["white_residual"]
    gt_gradient = np.einsum("nki,nk->ni", gt_weighted_j, gt_weighted_r)
    basis = shared.motion_basis(z_gt, pose_wci)
    lateral_tangent = z_gt[:3, :3].T @ basis[1] if basis is not None else None
    pixel1 = packet.match_fields["pixel1_uv"].numpy()
    pixel2 = packet.match_fields["pixel2_uv"].numpy()
    flow_cov = packet.match_fields["pixel2_uv_cov"].numpy()
    depth_cov = packet.match_fields["pixel1_d_cov"].numpy().reshape(-1) + packet.match_fields["pixel2_d_cov"].numpy().reshape(-1)
    upper = [(i, j) for i in range(6) for j in range(i, 6)]
    rows = []
    for point in range(len(points_i)):
        row: dict[str, Any] = {
            "edge_label": label,
            "frame_i": packet.frame_i,
            "frame_j": packet.frame_j,
            "mode": mode,
            "point": point,
            "selected": bool(selected[point]),
            "sidecar_original_inlier": bool(original_inlier[point]),
            "pixel1_u": float(pixel1[point, 0]), "pixel1_v": float(pixel1[point, 1]),
            "pixel2_u": float(pixel2[point, 0]), "pixel2_v": float(pixel2[point, 1]),
            "flow_u": float(pixel2[point, 0] - pixel1[point, 0]),
            "flow_v": float(pixel2[point, 1] - pixel1[point, 1]),
            "depth1": float(packet.match_fields["pixel1_d"][point]),
            "depth2": float(packet.match_fields["pixel2_d"][point]),
            "disparity2": float(packet.match_fields["pixel2_disp"][point]),
            "depth_cov_sum": float(depth_cov[point]),
            "flow_u_cov": float(flow_cov[point, 0]),
            "flow_v_cov": float(flow_cov[point, 1]),
            "flow_uv_cov": float(flow_cov[point, 2]),
            "disp_cov": float(packet.match_fields["pixel2_disp_cov"][point]),
            "uvd_residual_u": float(terms["residual"][point, 0]),
            "uvd_residual_v": float(terms["residual"][point, 1]),
            "uvd_residual_disp": float(terms["residual"][point, 2]),
            "uvd_whitened_norm": float(terms["norm"][point]),
            "robust_weight": float(terms["weight"][point]),
            "gt_uvd_residual_u": float(gt_terms["residual"][point, 0]),
            "gt_uvd_residual_v": float(gt_terms["residual"][point, 1]),
            "gt_uvd_residual_disp": float(gt_terms["residual"][point, 2]),
            "gt_uvd_whitened_norm": float(gt_terms["norm"][point]),
            "gt_robust_weight": float(gt_terms["weight"][point]),
            "gt_translation_gradient_lateral": (
                float(gt_gradient[point, :3] @ lateral_tangent)
                if lateral_tangent is not None else math.nan
            ),
            "cost_contribution": float(terms["cost"][point]) if selected[point] else 0.0,
            "hessian_trace_contribution": float(np.trace(hessian[point])) if selected[point] else 0.0,
        }
        for axis, value in zip("xyz", points_i[point]):
            row[f"point_i_{axis}"] = float(value)
        for r in range(3):
            for c in range(6):
                row[f"J_{r}{c}"] = float(terms["jacobian"][point, r, c])
        for index, value in enumerate(gradient[point]):
            row[f"g_{index}"] = float(value) if selected[point] else 0.0
        for r, c in upper:
            row[f"H_{r}{c}"] = float(hessian[point, r, c]) if selected[point] else 0.0
        rows.append(row)
    return rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    representatives = pd.read_csv(SELECTION)
    reader = VisualFactorCacheReader(DEFAULT_CACHE)
    _, truth_pose = load_truth(DEFAULT_DATASET)
    with np.load(DEFAULT_CACHE / "relative_pose_factors.npz", allow_pickle=False) as stream:
        measurements = stream["measurement_CiCj"].copy()
    summary_rows: list[dict[str, Any]] = []
    all_points: list[dict[str, Any]] = []
    self_checks = []
    refit_rows: list[dict[str, Any]] = []
    for edge in representatives.itertuples(index=False):
        frame_i, frame_j = int(edge.frame_i), int(edge.frame_j)
        packet = reader.load_pair(
            frame_i, frame_j,
            int(reader.manifest.timestamps_ns[frame_i]), int(reader.manifest.timestamps_ns[frame_j]),
        )
        points_i = packet.points_local.numpy().astype(np.float64)
        K = packet.K.numpy().astype(np.float64)
        observation = np.column_stack(
            [packet.match_fields["pixel2_uv"].numpy(), packet.match_fields["pixel2_disp"].numpy().reshape(-1)]
        )
        covariance = observation_covariance(packet)
        z_mac = se3_from_xyzw(measurements[frame_i].reshape(7))
        z_gt = invert_transform(truth_pose[frame_i]) @ truth_pose[frame_j]
        sidecar_terms = shared.point_terms(
            z_mac,
            points_i,
            shared.pixel2point_NED(
                packet.match_fields["pixel2_uv"].double().unsqueeze(0),
                packet.match_fields["pixel2_d"].double().reshape(1, -1),
                packet.K.double(),
            ).squeeze(0).numpy(),
            packet.match_fields["obs1_covTc"].numpy().astype(np.float64),
            packet.match_fields["obs2_covTc"].numpy().astype(np.float64),
            use_covariance=True,
            robust=True,
        )
        original_inlier = sidecar_terms["mahalanobis_norm"] <= 3.0
        flow_covariance = packet.match_fields["pixel2_uv_cov"].numpy()
        flow_cov = flow_covariance[:, 0] + flow_covariance[:, 1]
        depth_cov = packet.match_fields["pixel1_d_cov"].numpy().reshape(-1) + packet.match_fields["pixel2_d_cov"].numpy().reshape(-1)
        masks = {
            "P0": np.ones(len(points_i), dtype=bool),
            "P1": np.ones(len(points_i), dtype=bool),
            "P2": np.ones(len(points_i), dtype=bool),
            "P3": original_inlier.copy(),
            "P4": shared.lower_half_mask(flow_cov, original_inlier),
            "P5": shared.lower_half_mask(depth_cov, original_inlier),
            "P6": sidecar_terms["mahalanobis_norm"] <= 2.0,
            "P7": shared.stratified_mask(packet, original_inlier),
        }
        p0_terms = uvd_terms(
            z_mac, points_i, observation, covariance, K, packet.baseline_m,
            use_covariance=True, robust=True,
        )
        p0_weight = np.sqrt(p0_terms["weight"])[:, None, None]
        p0_gradient = np.einsum(
            "nki,nk->i",
            p0_weight * p0_terms["white_jacobian"],
            p0_weight[..., 0] * p0_terms["white_residual"],
        )
        results: dict[str, tuple[np.ndarray, dict[str, Any], bool, bool]] = {
            "P0": (
                z_mac,
                {
                    "iterations": 0, "converged": True,
                    "initial_cost": objective(p0_terms, masks["P0"]),
                    "final_cost": objective(p0_terms, masks["P0"]),
                    "final_step_norm": 0.0,
                    "final_gradient_norm": float(np.linalg.norm(p0_gradient)),
                    "final_translation_gradient_norm": float(np.linalg.norm(p0_gradient[:3])),
                    "final_rotation_gradient_norm": float(np.linalg.norm(p0_gradient[3:])),
                },
                True,
                True,
            )
        }
        results["P1"] = (*optimize(
            z_mac, points_i, observation, covariance, K, packet.baseline_m, masks["P1"],
            fixed_rotation=z_gt[:3, :3], use_covariance=True, robust=True,
        ), True, True)
        results["P2"] = (*optimize(
            z_mac, points_i, observation, covariance, K, packet.baseline_m, masks["P2"],
            fixed_rotation=z_mac[:3, :3], use_covariance=True, robust=True,
        ), True, True)
        results["P3"] = (*optimize(
            z_mac, points_i, observation, covariance, K, packet.baseline_m, masks["P3"],
            fixed_rotation=None, use_covariance=False, robust=False,
        ), False, False)
        for mode in ("P4", "P5", "P6", "P7"):
            results[mode] = (*optimize(
                z_mac, points_i, observation, covariance, K, packet.baseline_m, masks[mode],
                fixed_rotation=None, use_covariance=True, robust=True,
            ), True, True)
        refit_pose, refit_solve = optimize(
            z_mac, points_i, observation, covariance, K, packet.baseline_m, masks["P0"],
            fixed_rotation=None, use_covariance=True, robust=True,
        )
        refit_error = refit_pose[:3, 3] - z_gt[:3, 3]
        refit_rows.append(
            {
                "edge_label": str(edge.selection_purpose),
                "frame_i": frame_i,
                "translation_error_before_m": float(np.linalg.norm(z_mac[:3, 3] - z_gt[:3, 3])),
                "translation_error_after_refit_m": float(np.linalg.norm(refit_error)),
                "rotation_error_before_rad": float(np.linalg.norm(rotation_log(z_gt[:3, :3].T @ z_mac[:3, :3]))),
                "rotation_error_after_refit_rad": float(np.linalg.norm(rotation_log(z_gt[:3, :3].T @ refit_pose[:3, :3]))),
                **shared.motion_components(refit_error, z_gt, truth_pose[frame_i]),
                **refit_solve,
            }
        )
        self_checks.append(
            {
                "frame_i": frame_i,
                "sidecar_inlier_count_cached": int(edge.inlier_count),
                "sidecar_inlier_count_recomputed": int(original_inlier.sum()),
                **jacobian_check(z_mac, points_i, observation, covariance, K, packet.baseline_m),
            }
        )
        label = str(edge.selection_purpose)
        for mode in MODES:
            estimate, solve, use_covariance, robust = results[mode]
            translation_error = estimate[:3, 3] - z_gt[:3, 3]
            terms = uvd_terms(
                estimate, points_i, observation, covariance, K, packet.baseline_m,
                use_covariance=use_covariance, robust=robust,
            )
            summary_rows.append(
                {
                    "edge_label": label,
                    "frame_i": frame_i,
                    "frame_j": frame_j,
                    "mode": mode,
                    "selected_point_count": int(masks[mode].sum()),
                    "translation_error_norm_m": float(np.linalg.norm(translation_error)),
                    "rotation_error_norm_rad": float(np.linalg.norm(rotation_log(z_gt[:3, :3].T @ estimate[:3, :3]))),
                    **shared.motion_components(translation_error, z_gt, truth_pose[frame_i]),
                    "uvd_cost": objective(terms, masks[mode]),
                    "uvd_whitened_norm_median": float(np.median(terms["norm"][masks[mode]])),
                    "uvd_whitened_norm_p95": float(np.quantile(terms["norm"][masks[mode]], 0.95)),
                    **solve,
                }
            )
            all_points.extend(
                point_rows(
                    label=label, mode=mode, packet=packet, transform=estimate,
                    selected=masks[mode], original_inlier=original_inlier,
                    use_covariance=use_covariance, robust=robust, z_gt=z_gt,
                    pose_wci=truth_pose[frame_i],
                )
            )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT / "macvo_uvd_translation_counterfactual_per_edge.csv", index=False)
    pd.DataFrame(all_points).to_csv(OUT / "macvo_uvd_point_contributions.csv", index=False)
    pd.DataFrame(refit_rows).to_csv(OUT / "macvo_uvd_original_objective_refit.csv", index=False)
    checks = pd.DataFrame(self_checks)
    checks.to_csv(OUT / "macvo_uvd_self_checks.csv", index=False)
    aggregate = []
    for mode, group in summary.groupby("mode"):
        lateral = group.e_lateral.to_numpy(np.float64)
        lateral = lateral[np.isfinite(lateral)]
        aggregate.append(
            {
                "mode": mode,
                "translation_error_norm_mean_m": float(group.translation_error_norm_m.mean()),
                "translation_error_norm_median_m": float(group.translation_error_norm_m.median()),
                "translation_error_norm_max_m": float(group.translation_error_norm_m.max()),
                "lateral_error_mean_m": float(np.mean(lateral)) if lateral.size else None,
                "rotation_error_mean_rad": float(group.rotation_error_norm_rad.mean()),
                "selected_points_mean": float(group.selected_point_count.mean()),
                "initial_cost_sum": float(group.initial_cost.sum()),
                "final_cost_sum": float(group.final_cost.sum()),
                "converged_fraction": float(group.converged.astype(bool).mean()),
            }
        )
    payload = {
        "production_contract": {
            "graph_type": "disp",
            "residual": "predicted current-frame [u,v,disparity] minus observed [u,v,disparity]",
            "fixed_measurement": "previous-frame local 3D points",
            "covariance": "pixel2_uv_cov 2x2 plus pixel2_disp_cov",
            "robust_kernel": "PyPose FastTriggs Huber delta=0.1 equivalent first-order weight",
        },
        "aggregate": aggregate,
        "self_checks": {
            "sidecar_inlier_count_max_abs_difference": int(
                np.max(np.abs(checks.sidecar_inlier_count_cached - checks.sidecar_inlier_count_recomputed))
            ),
            "uvd_jacobian_max_abs_error": float(checks.jacobian_max_abs_error.max()),
            "uvd_jacobian_max_relative_error": float(checks.jacobian_max_relative_error.max()),
            "all_costs_nonincreasing": bool(
                np.all(summary.final_cost.to_numpy(np.float64) <= summary.initial_cost.to_numpy(np.float64) + 1.0e-10)
            ),
        },
        "artifacts": {
            "per_edge": str(OUT / "macvo_uvd_translation_counterfactual_per_edge.csv"),
            "per_point": str(OUT / "macvo_uvd_point_contributions.csv"),
            "checks": str(OUT / "macvo_uvd_self_checks.csv"),
            "original_objective_refit": str(OUT / "macvo_uvd_original_objective_refit.csv"),
        },
    }
    (OUT / "macvo_uvd_translation_counterfactual_summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
