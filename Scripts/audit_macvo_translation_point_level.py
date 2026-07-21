#!/usr/bin/env python3
"""Representative-edge point-level audit of the MACVO relative-translation bias."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pypose as pp
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Scripts.audit_circle_translation_oracle import (  # noqa: E402
    DEFAULT_CACHE,
    DEFAULT_DATASET,
    invert_transform,
    load_truth,
    make_transform,
    rotation_log,
    se3_from_xyzw,
    se3_to_xyzw,
)
from Utility.Point import pixel2point_NED, point2pixel_NED  # noqa: E402
from Utility.VisualFactorCache import VisualFactorCacheReader  # noqa: E402


OUT = ROOT / "analysis_circle_v3_backend_pointlevel_20260716/point_level"
EDGE_TABLE = ROOT / "analysis_circle_translation_oracle_20260716/circle_translation_error_per_edge.csv"
HUBER_DELTA = 3.0
MODES = ("P0", "P1", "P2", "P3", "P4", "P5", "P6", "P7")


def skew(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    result = np.zeros(points.shape[:-1] + (3, 3), dtype=np.float64)
    result[..., 0, 1] = -points[..., 2]
    result[..., 0, 2] = points[..., 1]
    result[..., 1, 0] = points[..., 2]
    result[..., 1, 2] = -points[..., 0]
    result[..., 2, 0] = -points[..., 1]
    result[..., 2, 1] = points[..., 0]
    return result


def transform_from_pp(value: torch.Tensor) -> np.ndarray:
    return se3_from_xyzw(value.detach().cpu().numpy().reshape(7))


def right_update(transform: np.ndarray, increment: np.ndarray) -> np.ndarray:
    delta = pp.se3(torch.from_numpy(np.asarray(increment, dtype=np.float64)).reshape(1, 6)).Exp()
    return transform @ transform_from_pp(delta.tensor())


def point_terms(
    transform: np.ndarray,
    points_i: np.ndarray,
    points_j: np.ndarray,
    covariance_i: np.ndarray,
    covariance_j: np.ndarray,
    *,
    use_covariance: bool,
    robust: bool,
) -> dict[str, np.ndarray]:
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    predicted_i = (rotation @ points_j.T).T + translation
    residual = points_i - predicted_i
    jacobian = np.concatenate(
        [
            -np.broadcast_to(rotation, (len(points_j), 3, 3)),
            np.einsum("ab,nbc->nac", rotation, skew(points_j)),
        ],
        axis=2,
    )
    covariance = covariance_i + np.einsum(
        "ab,nbc,dc->nad", rotation, covariance_j, rotation
    )
    covariance = 0.5 * (covariance + np.swapaxes(covariance, 1, 2))
    if use_covariance:
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        stabilized = np.einsum(
            "nij,nj,nkj->nik", eigenvectors, np.maximum(eigenvalues, 1.0e-12), eigenvectors
        )
        lower = np.linalg.cholesky(stabilized)
        whitened_residual = np.linalg.solve(lower, residual[..., None])[..., 0]
        whitened_jacobian = np.linalg.solve(lower, jacobian)
    else:
        stabilized = np.broadcast_to(np.eye(3), covariance.shape).copy()
        whitened_residual = residual.copy()
        whitened_jacobian = jacobian.copy()
    norms = np.linalg.norm(whitened_residual, axis=1)
    if robust:
        weights = np.where(norms <= HUBER_DELTA, 1.0, HUBER_DELTA / np.maximum(norms, 1.0e-12))
        costs = np.where(
            norms <= HUBER_DELTA,
            0.5 * norms**2,
            HUBER_DELTA * norms - 0.5 * HUBER_DELTA**2,
        )
    else:
        weights = np.ones_like(norms)
        costs = 0.5 * norms**2
    return {
        "predicted_i": predicted_i,
        "residual": residual,
        "covariance": stabilized,
        "jacobian": jacobian,
        "whitened_residual": whitened_residual,
        "whitened_jacobian": whitened_jacobian,
        "mahalanobis_norm": norms,
        "mahalanobis_sq": norms**2,
        "weight": weights,
        "cost": costs,
    }


def objective(terms: dict[str, np.ndarray], selected: np.ndarray) -> float:
    return float(np.sum(terms["cost"][selected]))


def optimize_pose(
    initial: np.ndarray,
    points_i: np.ndarray,
    points_j: np.ndarray,
    covariance_i: np.ndarray,
    covariance_j: np.ndarray,
    selected: np.ndarray,
    *,
    fixed_rotation: np.ndarray | None,
    use_covariance: bool,
    robust: bool,
    max_iterations: int = 50,
) -> tuple[np.ndarray, dict[str, Any]]:
    transform = initial.copy()
    if fixed_rotation is not None:
        transform[:3, :3] = fixed_rotation
    damping = 1.0e-3
    converged = False
    accepted = 0
    rejected = 0
    current_terms = point_terms(
        transform, points_i, points_j, covariance_i, covariance_j,
        use_covariance=use_covariance, robust=robust,
    )
    initial_cost = objective(current_terms, selected)
    current_cost = initial_cost
    final_step = math.inf
    iterations = 0
    for iteration in range(max_iterations):
        iterations = iteration + 1
        terms = point_terms(
            transform, points_i, points_j, covariance_i, covariance_j,
            use_covariance=use_covariance, robust=robust,
        )
        sqrt_weight = np.sqrt(terms["weight"][selected])[:, None, None]
        jacobian = sqrt_weight * terms["whitened_jacobian"][selected]
        residual = sqrt_weight[..., 0] * terms["whitened_residual"][selected]
        if fixed_rotation is not None:
            jacobian_active = jacobian[:, :, :3]
        else:
            jacobian_active = jacobian
        hessian = np.einsum("nki,nkj->ij", jacobian_active, jacobian_active)
        gradient = np.einsum("nki,nk->i", jacobian_active, residual)
        diagonal = np.maximum(np.abs(np.diag(hessian)), 1.0)
        system = hessian + damping * np.diag(diagonal)
        try:
            step_active = np.linalg.solve(system, -gradient)
        except np.linalg.LinAlgError:
            step_active = np.linalg.pinv(system) @ (-gradient)
        final_step = float(np.linalg.norm(step_active))
        if final_step <= 1.0e-10:
            converged = True
            break
        if fixed_rotation is not None:
            candidate = transform.copy()
            candidate[:3, 3] += fixed_rotation @ step_active
            candidate[:3, :3] = fixed_rotation
        else:
            candidate = right_update(transform, step_active)
        candidate_terms = point_terms(
            candidate, points_i, points_j, covariance_i, covariance_j,
            use_covariance=use_covariance, robust=robust,
        )
        candidate_cost = objective(candidate_terms, selected)
        if candidate_cost < current_cost:
            accepted += 1
            if abs(current_cost - candidate_cost) <= 1.0e-12:
                transform = candidate
                current_terms = candidate_terms
                current_cost = candidate_cost
                converged = True
                break
            transform = candidate
            current_terms = candidate_terms
            current_cost = candidate_cost
            damping = max(damping * 0.25, 1.0e-12)
        else:
            rejected += 1
            damping = min(damping * 10.0, 1.0e12)
    final_terms = point_terms(
        transform, points_i, points_j, covariance_i, covariance_j,
        use_covariance=use_covariance, robust=robust,
    )
    sqrt_weight = np.sqrt(final_terms["weight"][selected])[:, None, None]
    final_jacobian = sqrt_weight * final_terms["whitened_jacobian"][selected]
    final_residual = sqrt_weight[..., 0] * final_terms["whitened_residual"][selected]
    final_gradient = np.einsum("nki,nk->i", final_jacobian, final_residual)
    return transform, {
        "iterations": iterations,
        "converged": converged,
        "accepted_steps": accepted,
        "rejected_steps": rejected,
        "initial_cost": initial_cost,
        "final_cost": objective(final_terms, selected),
        "final_step_norm": final_step,
        "final_gradient_norm": float(np.linalg.norm(final_gradient)),
    }


def closest_row(frame: pd.DataFrame, column: str, target: float) -> pd.Series:
    return frame.iloc[int(np.argmin(np.abs(frame[column].to_numpy(np.float64) - target)))]


def select_edges(frame: pd.DataFrame) -> pd.DataFrame:
    selections: list[tuple[str, pd.Series]] = []
    phases = {
        "static_median": frame[(frame.frame_i >= 90) & (frame.frame_i < 190)],
        "startup_median": frame[(frame.frame_i >= 190) & (frame.frame_i < 280)],
        "steady_turn_median": frame[(frame.frame_i >= 500) & (frame.frame_i < 1600)],
        "stopping_median": frame[(frame.frame_i >= 1701) & (frame.frame_i < 1790)],
    }
    for name, subset in phases.items():
        selections.append((name, closest_row(subset, "translation_error_norm_m", float(subset.translation_error_norm_m.astype(float).median()))))
    errors = frame.translation_error_norm_m.to_numpy(np.float64)
    selections.extend(
        [
            ("global_median", closest_row(frame, "translation_error_norm_m", float(np.median(errors)))),
            ("global_p95", closest_row(frame, "translation_error_norm_m", float(np.quantile(errors, 0.95)))),
            ("global_max", frame.iloc[int(np.argmax(errors))]),
        ]
    )
    moving = frame[frame.motion_basis_valid.astype(str).str.lower().isin(("true", "1"))]
    selections.append(("max_abs_lateral", moving.iloc[int(np.argmax(np.abs(moving.e_lateral.to_numpy(np.float64))))]))
    rows = []
    seen: set[int] = set()
    for purpose, row in selections:
        edge = int(row.frame_i)
        if edge in seen:
            continue
        seen.add(edge)
        payload = row.to_dict()
        payload["selection_purpose"] = purpose
        rows.append(payload)
    return pd.DataFrame(rows)


def stratified_mask(packet, original_inlier: np.ndarray) -> np.ndarray:
    uv = packet.match_fields["pixel2_uv"].numpy().astype(np.float64)
    depth = packet.match_fields["pixel2_d"].numpy().astype(np.float64).reshape(-1)
    candidates = np.flatnonzero(original_inlier)
    if candidates.size < 12:
        return original_inlier.copy()
    u_edges = np.quantile(uv[candidates, 0], [0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0])
    v_edges = np.quantile(uv[candidates, 1], [0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0])
    d_edges = np.quantile(depth[candidates], [0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0])
    groups: list[np.ndarray] = []
    for u_bin in range(3):
        for v_bin in range(3):
            for d_bin in range(3):
                mask = original_inlier.copy()
                mask &= uv[:, 0] >= u_edges[u_bin]
                mask &= uv[:, 0] <= u_edges[u_bin + 1] if u_bin == 2 else uv[:, 0] < u_edges[u_bin + 1]
                mask &= uv[:, 1] >= v_edges[v_bin]
                mask &= uv[:, 1] <= v_edges[v_bin + 1] if v_bin == 2 else uv[:, 1] < v_edges[v_bin + 1]
                mask &= depth >= d_edges[d_bin]
                mask &= depth <= d_edges[d_bin + 1] if d_bin == 2 else depth < d_edges[d_bin + 1]
                index = np.flatnonzero(mask)
                if index.size:
                    groups.append(index)
    if not groups:
        return original_inlier.copy()
    target = max(1, min(len(group) for group in groups))
    selected = np.zeros_like(original_inlier)
    for group in groups:
        selected[group[:target]] = True
    return selected


def lower_half_mask(metric: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    """Select exactly the lower half with stable index tie-breaking."""

    index = np.flatnonzero(candidates)
    order = index[np.argsort(np.asarray(metric)[index], kind="stable")]
    selected = np.zeros_like(candidates, dtype=bool)
    selected[order[: max(1, int(math.ceil(len(order) * 0.5)))]] = True
    return selected


def motion_components(error: np.ndarray, z_gt: np.ndarray, pose_wci: np.ndarray) -> dict[str, float]:
    translation = z_gt[:3, 3]
    norm = float(np.linalg.norm(translation))
    if norm < 1.0e-4:
        return {"e_forward": math.nan, "e_lateral": math.nan, "e_vertical": math.nan}
    forward = translation / norm
    vertical_local = pose_wci[:3, :3].T @ np.array([0.0, 0.0, 1.0])
    vertical_local /= max(np.linalg.norm(vertical_local), 1.0e-12)
    lateral = np.cross(vertical_local, forward)
    if np.linalg.norm(lateral) < 1.0e-9:
        lateral = np.array([0.0, 1.0, 0.0])
    lateral /= np.linalg.norm(lateral)
    vertical = np.cross(forward, lateral)
    vertical /= np.linalg.norm(vertical)
    return {
        "e_forward": float(error @ forward),
        "e_lateral": float(error @ lateral),
        "e_vertical": float(error @ vertical),
    }


def motion_basis(z_gt: np.ndarray, pose_wci: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    translation = z_gt[:3, 3]
    norm = float(np.linalg.norm(translation))
    if norm < 1.0e-4:
        return None
    forward = translation / norm
    vertical_local = pose_wci[:3, :3].T @ np.array([0.0, 0.0, 1.0])
    vertical_local /= max(np.linalg.norm(vertical_local), 1.0e-12)
    lateral = np.cross(vertical_local, forward)
    if np.linalg.norm(lateral) < 1.0e-9:
        lateral = np.array([0.0, 1.0, 0.0])
    lateral /= np.linalg.norm(lateral)
    vertical = np.cross(forward, lateral)
    vertical /= np.linalg.norm(vertical)
    return forward, lateral, vertical


def per_point_rows(
    *, edge_label: str, mode: str, packet, transform: np.ndarray, selected: np.ndarray,
    original_inlier: np.ndarray, use_covariance: bool, robust: bool,
    z_gt: np.ndarray, pose_wci: np.ndarray,
) -> list[dict[str, Any]]:
    points_i = packet.points_local.numpy().astype(np.float64)
    points_j = pixel2point_NED(
        packet.match_fields["pixel2_uv"].double().unsqueeze(0),
        packet.match_fields["pixel2_d"].double().reshape(1, -1),
        packet.K.double(),
    ).squeeze(0).numpy()
    covariance_i = packet.match_fields["obs1_covTc"].numpy().astype(np.float64)
    covariance_j = packet.match_fields["obs2_covTc"].numpy().astype(np.float64)
    terms = point_terms(
        transform, points_i, points_j, covariance_i, covariance_j,
        use_covariance=use_covariance, robust=robust,
    )
    sqrt_weight = np.sqrt(terms["weight"])[:, None, None]
    weighted_j = sqrt_weight * terms["whitened_jacobian"]
    weighted_r = sqrt_weight[..., 0] * terms["whitened_residual"]
    hessian = np.einsum("nki,nkj->nij", weighted_j, weighted_j)
    gradient = np.einsum("nki,nk->ni", weighted_j, weighted_r)
    pixel1 = packet.match_fields["pixel1_uv"].numpy()
    pixel2 = packet.match_fields["pixel2_uv"].numpy()
    expected_j = (z_gt[:3, :3].T @ (points_i - z_gt[:3, 3]).T).T
    expected_uv = point2pixel_NED(
        torch.from_numpy(expected_j).double(), packet.K.double()
    ).numpy()
    gt_residual = points_i - ((z_gt[:3, :3] @ points_j.T).T + z_gt[:3, 3])
    basis = motion_basis(z_gt, pose_wci)
    viewing_direction = points_i / np.maximum(np.linalg.norm(points_i, axis=1, keepdims=True), 1.0e-12)
    gt_radial = np.sum(gt_residual * viewing_direction, axis=1)
    gt_transverse = gt_residual - gt_radial[:, None] * viewing_direction
    depth_cov = (
        packet.match_fields["pixel1_d_cov"].numpy().reshape(-1)
        + packet.match_fields["pixel2_d_cov"].numpy().reshape(-1)
    )
    flow_covariance = packet.match_fields["pixel2_uv_cov"].numpy()
    flow_cov = flow_covariance[:, 0] + flow_covariance[:, 1]
    rows = []
    upper = [(i, j) for i in range(6) for j in range(i, 6)]
    for point in range(len(points_i)):
        row: dict[str, Any] = {
            "edge_label": edge_label,
            "frame_i": int(packet.frame_i),
            "frame_j": int(packet.frame_j),
            "mode": mode,
            "point": point,
            "selected": bool(selected[point]),
            "original_inlier": bool(original_inlier[point]),
            "pixel1_u": float(pixel1[point, 0]), "pixel1_v": float(pixel1[point, 1]),
            "pixel2_u": float(pixel2[point, 0]), "pixel2_v": float(pixel2[point, 1]),
            "flow_u": float(pixel2[point, 0] - pixel1[point, 0]),
            "flow_v": float(pixel2[point, 1] - pixel1[point, 1]),
            "gt_expected_pixel2_u": float(expected_uv[point, 0]),
            "gt_expected_pixel2_v": float(expected_uv[point, 1]),
            "pixel2_error_to_gt_anchor_u": float(pixel2[point, 0] - expected_uv[point, 0]),
            "pixel2_error_to_gt_anchor_v": float(pixel2[point, 1] - expected_uv[point, 1]),
            "depth1": float(packet.match_fields["pixel1_d"][point]),
            "depth2": float(packet.match_fields["pixel2_d"][point]),
            "gt_expected_depth2_from_anchor": float(expected_j[point, 0]),
            "depth2_error_to_gt_anchor": float(packet.match_fields["pixel2_d"][point] - expected_j[point, 0]),
            "disparity1": float(packet.match_fields["pixel1_disp"][point]),
            "disparity2": float(packet.match_fields["pixel2_disp"][point]),
            "depth_cov_sum": float(depth_cov[point]),
            "flow_uv_cov_trace_sum": float(flow_cov[point]),
            "flow_uv_cov_cross": float(flow_covariance[point, 2]),
            "cov3d_i_trace": float(np.trace(covariance_i[point])),
            "cov3d_j_trace": float(np.trace(covariance_j[point])),
            "raw_residual_norm": float(np.linalg.norm(terms["residual"][point])),
            "gt_raw_residual_norm": float(np.linalg.norm(gt_residual[point])),
            "gt_residual_viewing_radial": float(gt_radial[point]),
            "gt_residual_viewing_transverse_norm": float(np.linalg.norm(gt_transverse[point])),
            "gt_residual_lateral": (
                float(gt_residual[point] @ basis[1]) if basis is not None else math.nan
            ),
            "mahalanobis_sq": float(terms["mahalanobis_sq"][point]),
            "robust_weight": float(terms["weight"][point]),
            "cost_contribution": float(terms["cost"][point]) if selected[point] else 0.0,
            "hessian_trace_contribution": float(np.trace(hessian[point])) if selected[point] else 0.0,
            "hessian_fro_contribution": float(np.linalg.norm(hessian[point])) if selected[point] else 0.0,
        }
        for prefix, vector in (("p_i", points_i[point]), ("p_j", points_j[point]), ("residual", terms["residual"][point])):
            for axis, value in zip("xyz", vector):
                row[f"{prefix}_{axis}"] = float(value)
        for r in range(3):
            for c in range(6):
                row[f"J_{r}{c}"] = float(terms["jacobian"][point, r, c])
        for index, value in enumerate(gradient[point]):
            row[f"g_{index}"] = float(value) if selected[point] else 0.0
        for r, c in upper:
            row[f"H_{r}{c}"] = float(hessian[point, r, c]) if selected[point] else 0.0
        rows.append(row)
    return rows


def raw_jacobian_check(
    transform: np.ndarray, points_i: np.ndarray, points_j: np.ndarray,
    covariance_i: np.ndarray, covariance_j: np.ndarray,
) -> dict[str, float]:
    terms = point_terms(
        transform, points_i, points_j, covariance_i, covariance_j,
        use_covariance=True, robust=True,
    )
    epsilon = 1.0e-7
    indices = np.linspace(0, len(points_i) - 1, min(20, len(points_i)), dtype=int)
    maximum_absolute = 0.0
    maximum_relative = 0.0
    for point in indices:
        numerical = np.zeros((3, 6), dtype=np.float64)
        for column in range(6):
            increment = np.zeros(6, dtype=np.float64)
            increment[column] = epsilon
            plus = point_terms(
                right_update(transform, increment), points_i, points_j, covariance_i, covariance_j,
                use_covariance=True, robust=True,
            )["residual"][point]
            minus = point_terms(
                right_update(transform, -increment), points_i, points_j, covariance_i, covariance_j,
                use_covariance=True, robust=True,
            )["residual"][point]
            numerical[:, column] = (plus - minus) / (2.0 * epsilon)
        analytical = terms["jacobian"][point]
        difference = np.abs(analytical - numerical)
        maximum_absolute = max(maximum_absolute, float(difference.max()))
        mask = np.abs(numerical) > 1.0e-7
        if np.any(mask):
            maximum_relative = max(
                maximum_relative,
                float(np.max(difference[mask] / np.abs(numerical[mask]))),
            )
    return {
        "epsilon": epsilon,
        "maximum_absolute_error": maximum_absolute,
        "maximum_relative_error_for_reference_above_1e7": maximum_relative,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    edge_frame = pd.read_csv(EDGE_TABLE)
    representatives = select_edges(edge_frame)
    representatives.to_csv(OUT / "representative_edge_selection.csv", index=False)
    reader = VisualFactorCacheReader(DEFAULT_CACHE)
    _, truth_pose = load_truth(DEFAULT_DATASET)
    with np.load(DEFAULT_CACHE / "relative_pose_factors.npz", allow_pickle=False) as stream:
        measurements = stream["measurement_CiCj"].copy()
    summary_rows: list[dict[str, Any]] = []
    point_rows: list[dict[str, Any]] = []
    refit_rows: list[dict[str, Any]] = []
    self_checks: list[dict[str, Any]] = []
    for selected_edge in representatives.itertuples(index=False):
        frame_i = int(selected_edge.frame_i)
        frame_j = int(selected_edge.frame_j)
        packet = reader.load_pair(
            frame_i,
            frame_j,
            int(reader.manifest.timestamps_ns[frame_i]),
            int(reader.manifest.timestamps_ns[frame_j]),
        )
        points_i = packet.points_local.numpy().astype(np.float64)
        points_j = pixel2point_NED(
            packet.match_fields["pixel2_uv"].double().unsqueeze(0),
            packet.match_fields["pixel2_d"].double().reshape(1, -1),
            packet.K.double(),
        ).squeeze(0).numpy()
        covariance_i = packet.match_fields["obs1_covTc"].numpy().astype(np.float64)
        covariance_j = packet.match_fields["obs2_covTc"].numpy().astype(np.float64)
        z_mac = se3_from_xyzw(measurements[frame_i].reshape(7))
        z_gt = invert_transform(truth_pose[frame_i]) @ truth_pose[frame_j]
        p0_terms = point_terms(
            z_mac, points_i, points_j, covariance_i, covariance_j,
            use_covariance=True, robust=True,
        )
        original_inlier = p0_terms["mahalanobis_norm"] <= HUBER_DELTA
        self_checks.append(
            {
                "frame_i": frame_i,
                "cached_inlier_count": int(selected_edge.inlier_count),
                "recomputed_inlier_count": int(original_inlier.sum()),
                **raw_jacobian_check(z_mac, points_i, points_j, covariance_i, covariance_j),
            }
        )
        flow_covariance = packet.match_fields["pixel2_uv_cov"].numpy()
        flow_cov = flow_covariance[:, 0] + flow_covariance[:, 1]
        depth_cov = (
            packet.match_fields["pixel1_d_cov"].numpy().reshape(-1)
            + packet.match_fields["pixel2_d_cov"].numpy().reshape(-1)
        )
        masks = {
            "P0": np.ones(len(points_i), dtype=bool),
            "P1": np.ones(len(points_i), dtype=bool),
            "P2": np.ones(len(points_i), dtype=bool),
            "P3": original_inlier.copy(),
            "P4": lower_half_mask(flow_cov, original_inlier),
            "P5": lower_half_mask(depth_cov, original_inlier),
            "P6": p0_terms["mahalanobis_norm"] <= 2.0,
            "P7": stratified_mask(packet, original_inlier),
        }
        p0_sqrt_weight = np.sqrt(p0_terms["weight"])[:, None, None]
        p0_jacobian = p0_sqrt_weight * p0_terms["whitened_jacobian"]
        p0_residual = p0_sqrt_weight[..., 0] * p0_terms["whitened_residual"]
        p0_gradient = np.einsum("nki,nk->i", p0_jacobian, p0_residual)
        mode_results: dict[str, tuple[np.ndarray, dict[str, Any], bool, bool]] = {
            "P0": (z_mac, {"iterations": 0, "converged": True, "initial_cost": objective(p0_terms, masks["P0"]), "final_cost": objective(p0_terms, masks["P0"]), "final_step_norm": 0.0, "final_gradient_norm": float(np.linalg.norm(p0_gradient))}, True, True)
        }
        mode_results["P1"] = (*optimize_pose(
            z_mac, points_i, points_j, covariance_i, covariance_j, masks["P1"],
            fixed_rotation=z_gt[:3, :3], use_covariance=True, robust=True,
        ), True, True)
        mode_results["P2"] = (*optimize_pose(
            z_mac, points_i, points_j, covariance_i, covariance_j, masks["P2"],
            fixed_rotation=z_mac[:3, :3], use_covariance=True, robust=True,
        ), True, True)
        mode_results["P3"] = (*optimize_pose(
            z_mac, points_i, points_j, covariance_i, covariance_j, masks["P3"],
            fixed_rotation=None, use_covariance=False, robust=False,
        ), False, False)
        for mode in ("P4", "P5", "P6", "P7"):
            mode_results[mode] = (*optimize_pose(
                z_mac, points_i, points_j, covariance_i, covariance_j, masks[mode],
                fixed_rotation=None, use_covariance=True, robust=True,
            ), True, True)
        refit_pose, refit_solve = optimize_pose(
            z_mac, points_i, points_j, covariance_i, covariance_j, masks["P0"],
            fixed_rotation=None, use_covariance=True, robust=True,
        )
        refit_error = refit_pose[:3, 3] - z_gt[:3, 3]
        refit_rows.append(
            {
                "edge_label": str(selected_edge.selection_purpose),
                "frame_i": frame_i,
                "translation_error_norm_before_m": float(np.linalg.norm(z_mac[:3, 3] - z_gt[:3, 3])),
                "translation_error_norm_after_refit_m": float(np.linalg.norm(refit_error)),
                "rotation_error_before_rad": float(np.linalg.norm(rotation_log(z_gt[:3, :3].T @ z_mac[:3, :3]))),
                "rotation_error_after_refit_rad": float(np.linalg.norm(rotation_log(z_gt[:3, :3].T @ refit_pose[:3, :3]))),
                **motion_components(refit_error, z_gt, truth_pose[frame_i]),
                **refit_solve,
            }
        )
        edge_label = str(selected_edge.selection_purpose)
        for mode in MODES:
            estimate, solve, use_covariance, robust = mode_results[mode]
            error = estimate[:3, 3] - z_gt[:3, 3]
            rotation_error = rotation_log(z_gt[:3, :3].T @ estimate[:3, :3])
            terms = point_terms(
                estimate, points_i, points_j, covariance_i, covariance_j,
                use_covariance=use_covariance, robust=robust,
            )
            summary_rows.append(
                {
                    "edge_label": edge_label,
                    "frame_i": frame_i,
                    "frame_j": frame_j,
                    "mode": mode,
                    "selected_point_count": int(masks[mode].sum()),
                    "translation_error_x": float(error[0]),
                    "translation_error_y": float(error[1]),
                    "translation_error_z": float(error[2]),
                    "translation_error_norm_m": float(np.linalg.norm(error)),
                    "rotation_error_norm_rad": float(np.linalg.norm(rotation_error)),
                    **motion_components(error, z_gt, truth_pose[frame_i]),
                    "point_cost": objective(terms, masks[mode]),
                    "point_mahalanobis_median": float(np.median(terms["mahalanobis_sq"][masks[mode]])),
                    "point_mahalanobis_p95": float(np.quantile(terms["mahalanobis_sq"][masks[mode]], 0.95)),
                    **solve,
                }
            )
            point_rows.extend(
                per_point_rows(
                    edge_label=edge_label,
                    mode=mode,
                    packet=packet,
                    transform=estimate,
                    selected=masks[mode],
                    original_inlier=original_inlier,
                    use_covariance=use_covariance,
                    robust=robust,
                    z_gt=z_gt,
                    pose_wci=truth_pose[frame_i],
                )
            )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT / "macvo_translation_counterfactual_per_edge.csv", index=False)
    pd.DataFrame(point_rows).to_csv(OUT / "macvo_translation_point_contributions.csv", index=False)
    pd.DataFrame(refit_rows).to_csv(OUT / "macvo_original_objective_refit_sanity.csv", index=False)
    self_check_frame = pd.DataFrame(self_checks)
    self_check_frame.to_csv(OUT / "point_level_self_checks.csv", index=False)
    aggregate = []
    for mode, group in summary.groupby("mode"):
        valid_lateral = group.e_lateral.to_numpy(np.float64)
        valid_lateral = valid_lateral[np.isfinite(valid_lateral)]
        aggregate.append(
            {
                "mode": mode,
                "representative_edge_count": int(len(group)),
                "translation_error_norm_mean_m": float(group.translation_error_norm_m.mean()),
                "translation_error_norm_median_m": float(group.translation_error_norm_m.median()),
                "translation_error_norm_max_m": float(group.translation_error_norm_m.max()),
                "lateral_error_mean_m": float(np.mean(valid_lateral)) if valid_lateral.size else None,
                "rotation_error_mean_rad": float(group.rotation_error_norm_rad.mean()),
                "selected_points_mean": float(group.selected_point_count.mean()),
                "converged_fraction": float(group.converged.astype(bool).mean()),
            }
        )
    payload = {
        "scope": "representative edges only; not a full-sequence replacement",
        "operational_definitions": {
            "P0": "existing pure-MACVO relative pose; no refit",
            "P1": "GT rotation fixed, translation optimized with original two-sided covariance and Huber",
            "P2": "MACVO rotation fixed, translation optimized with original two-sided covariance and Huber",
            "P3": "P0 inliers only, equal raw 3D residual weight, full SE3 optimized",
            "P4": "lower half of pixel2 flow covariance sigma_uu+sigma_vv among P0 inliers",
            "P5": "lower half of summed two-view depth covariance among P0 inliers",
            "P6": "P0 point Mahalanobis norm <= 2 instead of Huber threshold 3",
            "P7": "equal-count sampling over 3x3 image quantiles and three depth quantiles",
        },
        "aggregate": aggregate,
        "self_checks": {
            "inlier_count_max_abs_difference": int(
                np.max(np.abs(self_check_frame.cached_inlier_count - self_check_frame.recomputed_inlier_count))
            ),
            "raw_residual_jacobian_max_abs_error": float(self_check_frame.maximum_absolute_error.max()),
            "raw_residual_jacobian_max_relative_error": float(
                self_check_frame.maximum_relative_error_for_reference_above_1e7.max()
            ),
            "all_counterfactual_costs_nonincreasing": bool(
                np.all(summary.final_cost.to_numpy(np.float64) <= summary.initial_cost.to_numpy(np.float64) + 1.0e-10)
            ),
        },
        "artifacts": {
            "edge_selection": str(OUT / "representative_edge_selection.csv"),
            "per_edge": str(OUT / "macvo_translation_counterfactual_per_edge.csv"),
            "per_point": str(OUT / "macvo_translation_point_contributions.csv"),
            "original_objective_refit": str(OUT / "macvo_original_objective_refit_sanity.csv"),
            "self_checks": str(OUT / "point_level_self_checks.csv"),
        },
    }
    (OUT / "macvo_translation_counterfactual_summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(payload["aggregate"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
