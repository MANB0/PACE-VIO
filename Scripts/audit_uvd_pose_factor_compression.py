from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pypose as pp
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Utility.TwoStateVIO import (  # noqa: E402
    UVDFactor,
    linearize_uvd_relative_pose_factor,
    uvd_whitened_rows_from_relative,
)


DEFAULT_PACKET = ROOT / (
    "analysis_rectangle_uvd_schur_marginal_20260719/"
    "captured_rectangle_u1_problems.pt"
)
DEFAULT_OUTPUT = ROOT / "analysis_uvd_pose_factor_compression_20260720"
DTYPE = torch.float64


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit local UVD-to-pose-factor compression on captured U1 edges."
    )
    parser.add_argument("--packet", type=Path, default=DEFAULT_PACKET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--start-frame", type=int, default=90)
    parser.add_argument("--edge-count", type=int, default=209)
    parser.add_argument("--finite-difference-epsilon", type=float, default=1.0e-6)
    parser.add_argument("--samples-per-scale", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260720)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _float(value: torch.Tensor | np.ndarray | float | int) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().item())
    if isinstance(value, np.ndarray):
        return float(value.item())
    return float(value)


def _summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {key: float("nan") for key in ("min", "median", "mean", "p95", "max")}
    return {
        "min": float(np.min(array)),
        "median": float(np.median(array)),
        "mean": float(np.mean(array)),
        "p95": float(np.percentile(array, 95.0)),
        "max": float(np.max(array)),
    }


def _matrix_rank_and_condition(
    matrix: torch.Tensor, relative_threshold: float = 1.0e-10
) -> tuple[int, float, torch.Tensor]:
    eigenvalues = torch.linalg.eigvalsh(0.5 * (matrix + matrix.mT))
    maximum = max(_float(eigenvalues.abs().max()), 1.0)
    threshold = max(relative_threshold * maximum, torch.finfo(matrix.dtype).eps * maximum * 6)
    positive = eigenvalues > threshold
    rank = int(positive.sum().item())
    condition = (
        _float(eigenvalues[positive].max() / eigenvalues[positive].min())
        if rank > 0
        else float("inf")
    )
    return rank, condition, eigenvalues


def _symmetric_pseudoinverse(
    matrix: torch.Tensor, relative_threshold: float = 1.0e-10
) -> torch.Tensor:
    matrix = 0.5 * (matrix + matrix.mT)
    eigenvalues, eigenvectors = torch.linalg.eigh(matrix)
    maximum = max(_float(eigenvalues.abs().max()), 1.0)
    threshold = max(relative_threshold * maximum, torch.finfo(matrix.dtype).eps * maximum * 6)
    inverse = torch.where(
        eigenvalues > threshold,
        eigenvalues.reciprocal(),
        torch.zeros_like(eigenvalues),
    )
    return eigenvectors @ torch.diag(inverse) @ eigenvectors.mT


def _huber_cost(rows: torch.Tensor, delta: float) -> torch.Tensor:
    norms = torch.linalg.vector_norm(rows.reshape(-1, 3), dim=-1)
    threshold = torch.as_tensor(max(float(delta), 1.0e-12), dtype=rows.dtype)
    return torch.where(
        norms <= threshold,
        0.5 * norms.square(),
        threshold * norms - 0.5 * threshold.square(),
    ).sum()


def _base_irls_sqrt_weight(rows: torch.Tensor, delta: float) -> torch.Tensor:
    norms = torch.linalg.vector_norm(rows.reshape(-1, 3), dim=-1)
    threshold = torch.as_tensor(max(float(delta), 1.0e-12), dtype=rows.dtype)
    weights = torch.where(
        norms <= threshold,
        torch.ones_like(norms),
        threshold / norms.clamp_min(1.0e-12),
    )
    return weights.sqrt().unsqueeze(-1)


def _right_candidate(reference: pp.LieTensor, increment: torch.Tensor) -> torch.Tensor:
    return (reference @ pp.se3(increment.reshape(1, 6)).Exp()).tensor()


def _inversion_right_tangent_jacobian(
    measurement_CiCj: torch.Tensor,
) -> tuple[torch.Tensor, float]:
    measurement = pp.SE3(measurement_CiCj.reshape(1, 7))
    inverse_reference = measurement.Inv().detach()
    zero = torch.zeros(6, dtype=measurement_CiCj.dtype, requires_grad=True)

    def inverse_error(increment: torch.Tensor) -> torch.Tensor:
        changed = measurement @ pp.se3(increment.reshape(1, 6)).Exp()
        changed_inverse = changed.Inv()
        return (
            inverse_reference.Inv() @ changed_inverse
        ).Log().tensor().reshape(6)

    jacobian = torch.autograd.functional.jacobian(
        inverse_error, zero, create_graph=False, vectorize=True
    ).detach()
    epsilon = 1.0e-6
    finite_difference = torch.empty_like(jacobian)
    for column in range(6):
        step = torch.zeros(6, dtype=zero.dtype)
        step[column] = epsilon
        finite_difference[:, column] = (
            inverse_error(step) - inverse_error(-step)
        ) / (2.0 * epsilon)
    return jacobian, _float((jacobian - finite_difference).abs().max())


def _finite_difference_jacobian(
    residual: Callable[[torch.Tensor], torch.Tensor],
    *,
    epsilon: float,
    dtype: torch.dtype,
) -> torch.Tensor:
    baseline = residual(torch.zeros(6, dtype=dtype))
    result = torch.empty((baseline.numel(), 6), dtype=dtype)
    for column in range(6):
        step = torch.zeros(6, dtype=dtype)
        step[column] = epsilon
        result[:, column] = (
            residual(step) - residual(-step)
        ) / (2.0 * epsilon)
    return result


def _relative_entry_error(
    analytic: torch.Tensor, numerical: torch.Tensor, threshold: float = 1.0e-7
) -> float:
    mask = numerical.abs() > threshold
    if not bool(mask.any()):
        return 0.0
    return _float(
        ((analytic[mask] - numerical[mask]).abs() / numerical[mask].abs()).max()
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(_json_ready(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _sidecar_lookup(sidecar: np.lib.npyio.NpzFile) -> dict[tuple[int, int], int]:
    return {
        (int(frame_i), int(frame_j)): index
        for index, (frame_i, frame_j) in enumerate(
            zip(sidecar["frame_i"], sidecar["frame_j"], strict=True)
        )
    }


def _select_edges(payload: dict[str, Any], start_frame: int, edge_count: int) -> list[dict[str, Any]]:
    selected = [
        edge
        for edge in payload["edges"]
        if int(edge["frame_i"]) >= start_frame
    ][:edge_count]
    if len(selected) != edge_count:
        raise RuntimeError(f"requested {edge_count} edges but found {len(selected)}")
    for previous, current in zip(selected, selected[1:]):
        if int(previous["frame_j"]) != int(current["frame_i"]):
            raise RuntimeError("captured edge sequence is not contiguous")
    return selected


def _perturbation(
    generator: torch.Generator, translation_scale: float, rotation_scale: float
) -> torch.Tensor:
    translation = torch.randn(3, generator=generator, dtype=DTYPE)
    rotation = torch.randn(3, generator=generator, dtype=DTYPE)
    translation = translation / translation.norm().clamp_min(1.0e-12) * translation_scale
    rotation = rotation / rotation.norm().clamp_min(1.0e-12) * rotation_scale
    return torch.cat([translation, rotation])


def _audit(
    args: argparse.Namespace,
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, np.ndarray],
]:
    packet_path = args.packet.resolve()
    payload = torch.load(packet_path, map_location="cpu", weights_only=False)
    edges = _select_edges(payload, args.start_frame, args.edge_count)
    sidecar_path = Path(payload["visual_cache"]) / "relative_pose_factors.npz"
    sidecar = np.load(sidecar_path, allow_pickle=False)
    sidecar_lookup = _sidecar_lookup(sidecar)
    generator = torch.Generator().manual_seed(args.seed)
    per_edge: list[dict[str, Any]] = []
    perturbation_rows: list[dict[str, Any]] = []
    raw_lists: dict[str, list[np.ndarray | int]] = {
        "frame_i": [],
        "frame_j": [],
        "reference_CjCi": [],
        "hessian_t2": [],
        "gradient_t2": [],
        "sqrt_information_t2": [],
        "residual_offset_t2": [],
        "covariance_t2": [],
        "covariance_t0_common": [],
        "information_t0_common": [],
        "inversion_right_tangent_jacobian": [],
    }
    scale_contract = {
        "small": (1.0e-3, 5.0e-4),
        "medium": (5.0e-3, 2.0e-3),
        "large": (1.0e-2, 1.0e-2),
    }

    for edge_index, edge in enumerate(edges):
        frame_i = int(edge["frame_i"])
        frame_j = int(edge["frame_j"])
        key = (frame_i, frame_j)
        if key not in sidecar_lookup:
            raise RuntimeError(f"sidecar is missing edge {frame_i}->{frame_j}")
        sidecar_index = sidecar_lookup[key]
        problem = edge["problem"]
        visual = problem.visual_pose.to(device=torch.device("cpu"), dtype=DTYPE)
        if not isinstance(visual, UVDFactor):
            raise TypeError(f"edge {key} does not contain a UVDFactor")

        measurement_CiCj = torch.as_tensor(
            sidecar["measurement_CiCj"][sidecar_index], dtype=DTYPE
        ).reshape(1, 7)
        covariance_CiCj = torch.as_tensor(
            sidecar["covariance"][sidecar_index], dtype=DTYPE
        ).reshape(6, 6)
        reference_CjCi = pp.SE3(measurement_CiCj).Inv().detach()
        linearization = linearize_uvd_relative_pose_factor(
            reference_CjCi.tensor(), visual, marginal_mode="full"
        )
        factor = linearization.factor
        sqrt_information = factor.sqrt_information
        residual_offset = factor.residual_offset
        reconstructed_hessian = sqrt_information.mT @ sqrt_information
        reconstructed_gradient = sqrt_information.mT @ residual_offset
        hessian = linearization.full_hessian
        gradient = linearization.full_gradient
        hessian_scale = max(_float(torch.linalg.matrix_norm(hessian)), 1.0e-12)
        gradient_scale = max(_float(torch.linalg.vector_norm(gradient)), 1.0e-12)
        hessian_abs_error = _float((reconstructed_hessian - hessian).abs().max())
        hessian_relative_error = _float(
            torch.linalg.matrix_norm(reconstructed_hessian - hessian) / hessian_scale
        )
        gradient_abs_error = _float((reconstructed_gradient - gradient).abs().max())
        gradient_relative_error = _float(
            torch.linalg.vector_norm(reconstructed_gradient - gradient) / gradient_scale
        )

        base_rows = uvd_whitened_rows_from_relative(
            reference_CjCi.tensor(), visual
        )
        sqrt_weight = _base_irls_sqrt_weight(base_rows, visual.huber_delta)

        def fixed_residual(increment: torch.Tensor) -> torch.Tensor:
            candidate = _right_candidate(reference_CjCi, increment)
            rows = uvd_whitened_rows_from_relative(candidate, visual)
            return (sqrt_weight * rows).reshape(-1)

        finite_difference = _finite_difference_jacobian(
            fixed_residual,
            epsilon=args.finite_difference_epsilon,
            dtype=DTYPE,
        )
        jacobian_absolute_error = _float(
            (linearization.relative_jacobian - finite_difference).abs().max()
        )
        jacobian_frobenius_relative_error = _float(
            torch.linalg.matrix_norm(
                linearization.relative_jacobian - finite_difference
            )
            / torch.linalg.matrix_norm(finite_difference).clamp_min(1.0e-12)
        )
        jacobian_max_abs_normalized = jacobian_absolute_error / max(
            _float(finite_difference.abs().max()), 1.0e-12
        )
        jacobian_relative_error = _relative_entry_error(
            linearization.relative_jacobian, finite_difference
        )
        no_nonfinite = bool(
            torch.isfinite(hessian).all()
            and torch.isfinite(gradient).all()
            and torch.isfinite(finite_difference).all()
            and torch.isfinite(sqrt_information).all()
            and torch.isfinite(residual_offset).all()
        )

        rank_t2, condition_t2, eigenvalues_t2 = _matrix_rank_and_condition(hessian)
        covariance_t2 = _symmetric_pseudoinverse(hessian)
        covariance_t2_rank, covariance_t2_condition, covariance_t2_eigenvalues = (
            _matrix_rank_and_condition(covariance_t2)
        )
        center_shift = -_symmetric_pseudoinverse(hessian) @ gradient

        inversion_jacobian, inversion_jacobian_fd_error = (
            _inversion_right_tangent_jacobian(measurement_CiCj)
        )
        covariance_t0_common = (
            inversion_jacobian @ covariance_CiCj @ inversion_jacobian.mT
        )
        information_t0_common = _symmetric_pseudoinverse(
            covariance_t0_common, relative_threshold=1.0e-12
        )
        rank_t0, condition_t0, eigenvalues_t0 = _matrix_rank_and_condition(
            information_t0_common, relative_threshold=1.0e-12
        )
        information_relative_error = _float(
            torch.linalg.matrix_norm(hessian - information_t0_common)
            / torch.linalg.matrix_norm(information_t0_common).clamp_min(1.0e-12)
        )
        covariance_relative_error = _float(
            torch.linalg.matrix_norm(covariance_t2 - covariance_t0_common)
            / torch.linalg.matrix_norm(covariance_t0_common).clamp_min(1.0e-12)
        )
        trace_information_ratio = _float(
            torch.trace(hessian) / torch.trace(information_t0_common).clamp_min(1.0e-12)
        )
        trace_covariance_ratio = _float(
            torch.trace(covariance_t2) / torch.trace(covariance_t0_common).clamp_min(1.0e-12)
        )
        raw_lists["frame_i"].append(frame_i)
        raw_lists["frame_j"].append(frame_j)
        raw_lists["reference_CjCi"].append(
            reference_CjCi.tensor().detach().cpu().numpy().reshape(7)
        )
        raw_lists["hessian_t2"].append(hessian.detach().cpu().numpy())
        raw_lists["gradient_t2"].append(gradient.detach().cpu().numpy())
        raw_lists["sqrt_information_t2"].append(
            sqrt_information.detach().cpu().numpy()
        )
        raw_lists["residual_offset_t2"].append(
            residual_offset.detach().cpu().numpy()
        )
        raw_lists["covariance_t2"].append(covariance_t2.detach().cpu().numpy())
        raw_lists["covariance_t0_common"].append(
            covariance_t0_common.detach().cpu().numpy()
        )
        raw_lists["information_t0_common"].append(
            information_t0_common.detach().cpu().numpy()
        )
        raw_lists["inversion_right_tangent_jacobian"].append(
            inversion_jacobian.detach().cpu().numpy()
        )

        base_fixed_cost = 0.5 * linearization.robust_residual.square().sum()
        base_huber_cost = _huber_cost(base_rows, visual.huber_delta)
        for scale_name, (translation_scale, rotation_scale) in scale_contract.items():
            for sample in range(args.samples_per_scale):
                increment = _perturbation(generator, translation_scale, rotation_scale)
                candidate_rows = uvd_whitened_rows_from_relative(
                    _right_candidate(reference_CjCi, increment), visual
                )
                direct_fixed_change = (
                    0.5 * fixed_residual(increment).square().sum() - base_fixed_cost
                )
                compressed_change = (
                    0.5
                    * (sqrt_information @ increment + residual_offset).square().sum()
                    - 0.5 * residual_offset.square().sum()
                )
                true_huber_change = (
                    _huber_cost(candidate_rows, visual.huber_delta) - base_huber_cost
                )
                local_denominator = max(
                    abs(_float(direct_fixed_change)),
                    abs(_float(compressed_change)),
                    1.0e-9,
                )
                robust_denominator = max(
                    abs(_float(true_huber_change)),
                    abs(_float(compressed_change)),
                    1.0e-9,
                )
                perturbation_rows.append(
                    {
                        "edge_id": edge_index,
                        "frame_i": frame_i,
                        "frame_j": frame_j,
                        "scale": scale_name,
                        "sample": sample,
                        "translation_norm_m": _float(increment[:3].norm()),
                        "rotation_norm_rad": _float(increment[3:].norm()),
                        "direct_fixed_irls_cost_change": _float(direct_fixed_change),
                        "compressed_quadratic_cost_change": _float(compressed_change),
                        "true_huber_cost_change": _float(true_huber_change),
                        "fixed_vs_compressed_abs_error": abs(
                            _float(direct_fixed_change - compressed_change)
                        ),
                        "fixed_vs_compressed_relative_error": abs(
                            _float(direct_fixed_change - compressed_change)
                        )
                        / local_denominator,
                        "huber_vs_compressed_abs_error": abs(
                            _float(true_huber_change - compressed_change)
                        ),
                        "huber_vs_compressed_relative_error": abs(
                            _float(true_huber_change - compressed_change)
                        )
                        / robust_denominator,
                    }
                )

        quality = edge.get("quality", {})
        per_edge.append(
            {
                "edge_id": edge_index,
                "frame_i": frame_i,
                "frame_j": frame_j,
                "point_count": int(visual.points_Ci.shape[0]),
                "huber_delta": float(visual.huber_delta),
                "num_observations": int(quality.get("num_observations", visual.points_Ci.shape[0])),
                "coverage": float(quality.get("coverage", float("nan"))),
                "depth_spread": float(quality.get("depth_spread", float("nan"))),
                "visual_obs_cov_mean": float(quality.get("visual_obs_cov_mean", float("nan"))),
                "base_true_huber_cost": _float(base_huber_cost),
                "base_fixed_irls_cost": _float(base_fixed_cost),
                "gradient_norm": _float(gradient.norm()),
                "gradient_translation_norm": _float(gradient[:3].norm()),
                "gradient_rotation_norm": _float(gradient[3:].norm()),
                "center_shift_translation_norm_m": _float(center_shift[:3].norm()),
                "center_shift_rotation_norm_rad": _float(center_shift[3:].norm()),
                "t2_rank": rank_t2,
                "t2_condition": condition_t2,
                "t2_information_trace": _float(torch.trace(hessian)),
                "t2_information_eigenvalues": json.dumps(
                    eigenvalues_t2.detach().cpu().tolist()
                ),
                "t2_covariance_rank": covariance_t2_rank,
                "t2_covariance_condition": covariance_t2_condition,
                "t2_covariance_trace": _float(torch.trace(covariance_t2)),
                "t2_covariance_eigenvalues": json.dumps(
                    covariance_t2_eigenvalues.detach().cpu().tolist()
                ),
                "t0_rank_common": rank_t0,
                "t0_condition_common": condition_t0,
                "t0_information_trace_common": _float(torch.trace(information_t0_common)),
                "t0_information_eigenvalues_common": json.dumps(
                    eigenvalues_t0.detach().cpu().tolist()
                ),
                "information_frobenius_relative_difference": information_relative_error,
                "covariance_frobenius_relative_difference": covariance_relative_error,
                "trace_information_t2_over_t0": trace_information_ratio,
                "trace_covariance_t2_over_t0": trace_covariance_ratio,
                "hessian_reconstruction_max_abs_error": hessian_abs_error,
                "hessian_reconstruction_relative_error": hessian_relative_error,
                "gradient_reconstruction_max_abs_error": gradient_abs_error,
                "gradient_reconstruction_relative_error": gradient_relative_error,
                "jacobian_fd_max_abs_error": jacobian_absolute_error,
                "jacobian_fd_frobenius_relative_error": jacobian_frobenius_relative_error,
                "jacobian_fd_max_abs_normalized": jacobian_max_abs_normalized,
                "jacobian_fd_max_relative_error": jacobian_relative_error,
                "inversion_jacobian_fd_max_abs_error": inversion_jacobian_fd_error,
                "no_nan_inf": no_nonfinite,
                "sidecar_num_points": int(sidecar["num_points"][sidecar_index]),
                "sidecar_num_inliers": int(sidecar["num_inliers"][sidecar_index]),
                "sidecar_mean_mahalanobis_sq": float(
                    sidecar["mean_mahalanobis_sq"][sidecar_index]
                ),
            }
        )
        if (edge_index + 1) % 10 == 0 or edge_index + 1 == len(edges):
            print(f"audited {edge_index + 1}/{len(edges)} edges", flush=True)

    prior_truth_summary_path = ROOT / (
        "analysis_visual_covariance_ablation_20260715/"
        "visual_sidecar_truth_covariance_summary.json"
    )
    prior_truth_summary = None
    if prior_truth_summary_path.exists():
        prior_truth_summary = json.loads(prior_truth_summary_path.read_text(encoding="utf-8"))

    summary_fields = {
        field: _summary([float(row[field]) for row in per_edge])
        for field in (
            "base_true_huber_cost",
            "gradient_norm",
            "center_shift_translation_norm_m",
            "center_shift_rotation_norm_rad",
            "t2_condition",
            "t0_condition_common",
            "information_frobenius_relative_difference",
            "covariance_frobenius_relative_difference",
            "trace_information_t2_over_t0",
            "trace_covariance_t2_over_t0",
            "hessian_reconstruction_max_abs_error",
            "hessian_reconstruction_relative_error",
            "gradient_reconstruction_max_abs_error",
            "gradient_reconstruction_relative_error",
            "jacobian_fd_max_abs_error",
            "jacobian_fd_frobenius_relative_error",
            "jacobian_fd_max_abs_normalized",
            "jacobian_fd_max_relative_error",
            "inversion_jacobian_fd_max_abs_error",
        )
    }
    perturbation_summary: dict[str, Any] = {}
    for scale_name in scale_contract:
        subset = [row for row in perturbation_rows if row["scale"] == scale_name]
        perturbation_summary[scale_name] = {
            field: _summary([float(row[field]) for row in subset])
            for field in (
                "fixed_vs_compressed_abs_error",
                "fixed_vs_compressed_relative_error",
                "huber_vs_compressed_abs_error",
                "huber_vs_compressed_relative_error",
            )
        }

    exact_algebra_pass = (
        max(row["hessian_reconstruction_relative_error"] for row in per_edge) <= 1.0e-10
        and max(row["gradient_reconstruction_relative_error"] for row in per_edge) <= 1.0e-10
    )
    jacobian_pass = (
        max(row["jacobian_fd_frobenius_relative_error"] for row in per_edge) <= 1.0e-6
        and max(row["jacobian_fd_max_abs_normalized"] for row in per_edge) <= 1.0e-6
        and max(row["inversion_jacobian_fd_max_abs_error"] for row in per_edge) <= 1.0e-5
        and all(bool(row["no_nan_inf"]) for row in per_edge)
    )
    small_local_p95 = perturbation_summary["small"][
        "fixed_vs_compressed_relative_error"
    ]["p95"]
    local_model_pass = bool(small_local_p95 <= 0.05)
    approved_for_short_replay = exact_algebra_pass and jacobian_pass and local_model_pass

    summary = {
        "schema_version": 1,
        "edge_count": len(per_edge),
        "frame_range": [per_edge[0]["frame_i"], per_edge[-1]["frame_j"]],
        "finite_difference_epsilon": args.finite_difference_epsilon,
        "samples_per_scale_per_edge": args.samples_per_scale,
        "rank_histogram_t2": dict(Counter(str(row["t2_rank"]) for row in per_edge)),
        "nonfinite_edge_count": sum(not bool(row["no_nan_inf"]) for row in per_edge),
        "per_edge_statistics": summary_fields,
        "perturbation_statistics": perturbation_summary,
        "acceptance": {
            "exact_normal_equation_reconstruction": exact_algebra_pass,
            "central_finite_difference_jacobians": jacobian_pass,
            "small_perturbation_p95_relative_error_le_5_percent": local_model_pass,
            "approved_for_gate5_short_replay": approved_for_short_replay,
            "production_default_changed": False,
        },
        "existing_t0_truth_covariance_audit": prior_truth_summary,
    }
    contract = {
        "schema_version": 1,
        "packet": str(packet_path),
        "packet_sha256": _sha256(packet_path),
        "sidecar": str(sidecar_path),
        "sidecar_sha256": _sha256(sidecar_path),
        "dataset": payload.get("dataset"),
        "frozen_config": payload.get("frozen_config"),
        "active_start_frame": payload.get("active_start_frame"),
        "audited_frame_range": summary["frame_range"],
        "software": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "pypose": getattr(pp, "__version__", "unknown"),
            "numpy": np.__version__,
        },
        "common_pose_tangent": {
            "measurement_direction": "T_CjCi: maps a point in camera i into camera j",
            "update": "T_new = T_reference * Exp(delta)",
            "increment_order": ["tx", "ty", "tz", "rx", "ry", "rz"],
            "frame": "right/local tangent at T_CjCi",
        },
        "u1_point_objective": {
            "source": "points_Ci are fixed source-frame 3D observations",
            "target": "target_uv and target_disparity are fixed frame-j UVD observations",
            "covariance": "per-point target-frame [u,v,disparity] covariance",
            "whitening": "per-point Cholesky solve",
            "robust_loss": "Huber on each whitened 3-vector",
        },
        "t2_compressed_factor": {
            "linearization_mean": "inverse of current T0 sidecar T_CiCj mean",
            "information": "J^T W J of the same U1 UVD objective",
            "gradient": "J^T W r of the same U1 UVD objective",
            "factor_residual": "sqrt_information * delta + residual_offset",
            "rank_handling": "retain only positive Hessian eigenmodes; no nullspace fill",
            "robust_weight_policy": "IRLS weights frozen at compression linearization",
        },
        "t0_sidecar_factor": {
            "native_direction": "T_CiCj: maps a point in camera j into camera i",
            "native_tangent": "right/local [translation,rotation]",
            "covariance_origin": "two-sided 3D-3D residual covariance and fixed Huber IRLS",
            "comparison_transform": "full autodiff inversion Jacobian, not blind block permutation",
        },
        "scope": {
            "production_code_path_changed": False,
            "full_dataset_run": False,
            "gate5_short_replay_run": False,
        },
    }
    raw_matrices = {
        key: np.asarray(values)
        for key, values in raw_lists.items()
    }
    return (
        {"contract": contract, "summary": summary},
        per_edge,
        perturbation_rows,
        raw_matrices,
    )


def _write_report(path: Path, result: dict[str, Any]) -> None:
    summary = result["summary"]
    stats = summary["per_edge_statistics"]
    perturb = summary["perturbation_statistics"]
    acceptance = summary["acceptance"]
    truth = summary.get("existing_t0_truth_covariance_audit")

    def metric(name: str) -> str:
        item = stats[name]
        return (
            f"median={item['median']:.6g}, p95={item['p95']:.6g}, "
            f"max={item['max']:.6g}"
        )

    truth_note = "未找到既有 T0 真值 NIS 审计。"
    if isinstance(truth, dict):
        nis = truth.get("nis", truth.get("pose_nis", truth))
        if isinstance(nis, dict):
            truth_note = (
                "既有 T0 真值 NIS："
                f"median={float(nis['median']):.6g}, mean={float(nis['mean']):.6g}, "
                f"p95={float(nis['p95']):.6g}, max={float(nis['max']):.6g}；"
                f"6 维卡方中央 95% 区间覆盖率={float(nis['inside_interval_ratio']):.3%}。"
            )

    decision = (
        "批准进入 Gate 5 的短序列 T0/T2 对照，但不批准替换生产默认。"
        if acceptance["approved_for_gate5_short_replay"]
        else "暂不进入 Gate 5；应先修复局部等价性或 Jacobian 问题。"
    )
    lines = [
        "# MACVO UVD 点级不确定性压缩审计（Gate 0-4）",
        "",
        "## 首页结论",
        "",
        f"- 审计范围：{summary['edge_count']} 条边，frame {summary['frame_range'][0]}->{summary['frame_range'][1]}。",
        f"- 正规方程精确重建：{acceptance['exact_normal_equation_reconstruction']}。",
        f"- 中心有限差分 Jacobian：{acceptance['central_finite_difference_jacobians']}。",
        f"- 小扰动局部模型 P95 相对误差 <= 5%：{acceptance['small_perturbation_p95_relative_error_le_5_percent']}。",
        f"- 决策：{decision}",
        "- 本轮没有修改生产默认，没有运行完整序列。",
        "",
        "## 方法定义",
        "",
        "T0 是现有 relative-pose sidecar：均值来自 MACVO 相对位姿，协方差由两端 3D 点及其协方差重建。",
        "T2 是同目标压缩因子：在 T0 均值处，直接对 U1 的 UVD 残差计算 J、固定当次 Huber IRLS 权重，形成 H=J^T W J、g=J^T W r，再保存秩感知平方根因子。",
        "两者比较前均统一到 T_CjCi、右扰动、[translation, rotation] 切空间；T0 使用完整求逆 Jacobian 变换。",
        "",
        "## 数值正确性",
        "",
        f"- Hessian 重建相对误差：{metric('hessian_reconstruction_relative_error')}。",
        f"- gradient 重建相对误差：{metric('gradient_reconstruction_relative_error')}。",
        f"- UVD Jacobian 中心差分绝对误差：{metric('jacobian_fd_max_abs_error')}。",
        f"- UVD Jacobian 中心差分 Frobenius 相对误差：{metric('jacobian_fd_frobenius_relative_error')}。",
        f"- UVD Jacobian 最大绝对差 / 最大 Jacobian：{metric('jacobian_fd_max_abs_normalized')}。",
        f"- 求逆坐标变换 Jacobian 中心差分绝对误差：{metric('inversion_jacobian_fd_max_abs_error')}。",
        f"- NaN/Inf 边数：{summary['nonfinite_edge_count']}。",
        f"- T2 有效秩分布：{summary['rank_histogram_t2']}。",
        "",
        "## 局部压缩误差",
        "",
    ]
    for scale in ("small", "medium", "large"):
        fixed = perturb[scale]["fixed_vs_compressed_relative_error"]
        robust = perturb[scale]["huber_vs_compressed_relative_error"]
        lines.append(
            f"- {scale}: 固定 IRLS 与二次因子误差 median={fixed['median']:.4g}, "
            f"p95={fixed['p95']:.4g}; 真实 Huber 与二次因子误差 "
            f"median={robust['median']:.4g}, p95={robust['p95']:.4g}。"
        )
    lines.extend(
        [
            "",
            "## T0 与 T2 的信息差异",
            "",
            f"- information Frobenius 相对差：{metric('information_frobenius_relative_difference')}。",
            f"- covariance Frobenius 相对差：{metric('covariance_frobenius_relative_difference')}。",
            f"- trace(H_T2)/trace(H_T0)：{metric('trace_information_t2_over_t0')}。",
            f"- trace(P_T2)/trace(P_T0)：{metric('trace_covariance_t2_over_t0')}。",
            f"- T0 common-space condition：{metric('t0_condition_common')}。",
        f"- T2 condition：{metric('t2_condition')}。",
            f"- T0 均值到 UVD 局部牛顿中心的平移：{metric('center_shift_translation_norm_m')} m。",
            f"- T0 均值到 UVD 局部牛顿中心的旋转：{metric('center_shift_rotation_norm_rad')} rad。",
            "",
            "这部分不要求两者相等。差异反映的是 T0 的 3D-3D 协方差模型与 U1 的目标帧 UVD 模型并非同一个统计目标。",
            "固定 IRLS 二次因子保留的是当前一次 Gauss-Newton/IRLS 的 H 和 g；它没有保留后续位姿变化时 Huber 权重的重新计算，因此不是完整非线性 Huber 目标的全局等价替代。",
            "",
            "## T0 既有统计证据",
            "",
            truth_note,
            "此前 209 边审计显示 T0 pose NIS 明显偏低，这意味着协方差偏保守；本报告不会把 T2 的局部 Hessian 自动宣称为统计校准后的 covariance。",
            "",
            "## 下一步边界",
            "",
            "若 Gate 5 获批，只允许在同一短序列上比较 T0 与 T2：轨迹、RPE、迭代数、耗时、pose NIS 与门控表现。统计一致性必须用独立数据或留出边验证，不能仅凭本次局部等价性宣布完成。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = _parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    result, per_edge, perturbations, raw_matrices = _audit(args)
    _write_csv(args.output / "uvd_pose_compression_per_edge.csv", per_edge)
    _write_csv(
        args.output / "uvd_pose_compression_perturbations.csv", perturbations
    )
    _write_json(args.output / "uvd_pose_compression_contract.json", result["contract"])
    _write_json(args.output / "uvd_pose_compression_summary.json", result["summary"])
    np.savez_compressed(
        args.output / "uvd_pose_compression_raw_matrices.npz", **raw_matrices
    )
    _write_report(args.output / "uvd_pose_compression_report_cn.md", result)
    print(json.dumps(_json_ready(result["summary"]["acceptance"]), indent=2))
    print(f"output: {args.output.resolve()}")


if __name__ == "__main__":
    main()
