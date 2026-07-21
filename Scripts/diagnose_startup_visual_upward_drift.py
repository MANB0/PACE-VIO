#!/usr/bin/env python3
"""Audit post-static MACVO startup drift without changing production estimators.

The script is deliberately visual-only. Ground truth is used only for offline
error decomposition and oracle counterfactuals; no generated measurement is
written back to a production cache.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy.spatial.transform import Rotation
from scipy.stats import pearsonr, spearmanr


ROOT = Path(__file__).resolve().parents[1]
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Scripts.audit_circle_translation_oracle import (  # noqa: E402
    compose_edges,
    edge_error,
    invert_transform,
    load_truth,
    make_transform,
    pose_error,
    pose_internal_to_nwu,
    quantiles,
    relative_edges,
    rotation_log,
    se3_from_xyzw,
    trajectory_metrics,
)
from Scripts.audit_initialization_boundary import reconstruct_first_interval  # noqa: E402
import Scripts.audit_macvo_translation_point_level as d3  # noqa: E402
import Scripts.audit_macvo_translation_uvd_point_level as uvd  # noqa: E402
from Utility.Point import pixel2point_NED, point2pixel_NED  # noqa: E402
from Utility.VisualFactorCache import VisualFactorCacheReader  # noqa: E402


SCENE = "clear_circle_truth_normal_noise"
DATASET = Path(
    "/mnt/e/文档/holoocean/code/recordings/"
    "batch_clear_truth_paths_20260713_static63_variants/clear_circle_truth_normal_noise"
)
CONTINUOUS_CACHE = ROOT / "VisualCache/static63_unique_visual_20260713" / SCENE
COLD_CACHE = ROOT / "VisualCache/circle_post3_pure_macvo_20260717" / SCENE
CONTINUOUS_RESULT = ROOT / (
    "Results/visual_factor_cache_static63_unique_source_20260713/trial_1/"
    "pure_macvo/clear_circle_truth_normal_noise"
)
COLD_RESULT = ROOT / "Results/circle_post3_pure_macvo_cache_20260717/source" / SCENE
PRODUCTION_RESULT = ROOT / (
    "Results/circle_straight_normal_noise_two_state_standard_full_20260715/trial_1/"
    "vio_two_state_fixed_lag_standard_full/clear_circle_truth_normal_noise"
)
OUTPUT = ROOT / "analysis_startup_visual_upward_drift_20260718"
SAME_BUILD_AB = OUTPUT / "same_build_cold_start_ab"
SAME_BUILD_CONTINUOUS_CACHE = SAME_BUILD_AB / "caches/C0_continuous" / SCENE
SAME_BUILD_COLD_CACHE = SAME_BUILD_AB / "caches/C1_cold" / SCENE
SAME_BUILD_CONTINUOUS_RESULT = SAME_BUILD_AB / "results/C0_continuous" / SCENE
SAME_BUILD_COLD_RESULT = SAME_BUILD_AB / "results/C1_cold" / SCENE
STATIC_DURATION_S = 3.0
STARTUP_EDGES = 30
EPS = 1.0e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--continuous-cache", type=Path, default=CONTINUOUS_CACHE)
    parser.add_argument("--cold-cache", type=Path, default=COLD_CACHE)
    parser.add_argument("--continuous-result", type=Path, default=CONTINUOUS_RESULT)
    parser.add_argument("--cold-result", type=Path, default=COLD_RESULT)
    parser.add_argument("--same-build-continuous-cache", type=Path, default=SAME_BUILD_CONTINUOUS_CACHE)
    parser.add_argument("--same-build-cold-cache", type=Path, default=SAME_BUILD_COLD_CACHE)
    parser.add_argument("--same-build-continuous-result", type=Path, default=SAME_BUILD_CONTINUOUS_RESULT)
    parser.add_argument("--same-build-cold-result", type=Path, default=SAME_BUILD_COLD_RESULT)
    parser.add_argument("--production-result", type=Path, default=PRODUCTION_RESULT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--static-duration-s", type=float, default=STATIC_DURATION_S)
    return parser.parse_args()


def jsonify(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): jsonify(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonify(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(jsonify(payload), indent=2, ensure_ascii=False), encoding="utf-8")


def sha256_arrays(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        value = np.ascontiguousarray(array)
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(value.shape).encode("ascii"))
        digest.update(value.tobytes())
    return digest.hexdigest()


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {name: data[name].copy() for name in data.files}


def image_timestamps(dataset: Path) -> np.ndarray:
    paths = sorted((dataset / "left").glob("*.png"), key=lambda item: int(item.stem))
    timestamps = np.asarray([int(path.stem) for path in paths], dtype=np.int64)
    if timestamps.size < 2 or np.any(np.diff(timestamps) <= 0):
        raise RuntimeError("image timestamps are missing or not strictly increasing")
    return timestamps


def bool_column(series: pd.Series) -> np.ndarray:
    return series.astype(str).str.lower().isin(("1", "true", "yes")).to_numpy()


def load_sidecar(cache: Path) -> dict[str, np.ndarray]:
    return load_npz(cache / "relative_pose_factors.npz")


def find_diagnostic_csv(root: Path) -> Path:
    direct = root / "frame_pair_diagnostics.csv"
    if direct.exists():
        return direct
    candidates = sorted(root.rglob("frame_pair_diagnostics.csv"))
    if len(candidates) != 1:
        raise RuntimeError(f"expected one diagnostics CSV under {root}, got {len(candidates)}")
    return candidates[0]


def find_tensor_map(root: Path) -> Path:
    direct = root / "tensor_map.npz"
    if direct.exists():
        return direct
    candidates = sorted(root.rglob("tensor_map.npz"))
    if len(candidates) != 1:
        raise RuntimeError(f"expected one tensor_map under {root}, got {len(candidates)}")
    return candidates[0]


def derive_active_start(args: argparse.Namespace) -> tuple[int, np.ndarray, pd.DataFrame, dict[str, Any]]:
    timestamps = image_timestamps(args.dataset)
    t_init_start = int(timestamps[0])
    t_init_end = int(round(t_init_start + args.static_duration_s * 1.0e9))
    timestamp_candidate = int(np.searchsorted(timestamps, t_init_end, side="left"))
    diagnostics = pd.read_csv(find_diagnostic_csv(args.production_result))
    active = diagnostics.loc[bool_column(diagnostics["vio_factor_active"])].copy()
    if active.empty:
        raise RuntimeError("production result contains no active VIO factor")
    first = active.iloc[0]
    used_candidate = int(first.frame_i)
    if used_candidate != timestamp_candidate:
        raise RuntimeError(
            f"timestamp-derived start {timestamp_candidate} differs from first official state {used_candidate}"
        )
    s = timestamp_candidate
    j = int(first.frame_j)
    if j != s + 1:
        raise RuntimeError(f"first official edge is not contiguous: {s}->{j}")
    formal_edges = active[["frame_i", "frame_j"]].astype(int).to_numpy()
    crossing = bool(np.any((formal_edges[:, 0] < s) | (formal_edges[:, 1] <= s)))
    interval = reconstruct_first_interval(args.dataset, timestamps.tolist(), s, j)

    with np.load(find_tensor_map(args.production_result), allow_pickle=False) as data:
        pose_s = data["frames//pose"][s].astype(np.float64)
    pose_s_identity_log = float(
        np.linalg.norm(rotation_log(se3_from_xyzw(pose_s)[:3, :3]))
        + np.linalg.norm(pose_s[:3])
    )
    cold_manifest = json.loads((args.cold_cache / "manifest.json").read_text(encoding="utf-8"))
    continuous_manifest = json.loads((args.continuous_cache / "manifest.json").read_text(encoding="utf-8"))
    contract = {
        "scene": SCENE,
        "initialization_start_timestamp_ns": t_init_start,
        "initialization_end_timestamp_ns": t_init_end,
        "static_duration_s": float(args.static_duration_s),
        "active_start_frame": s,
        "active_start_timestamp_ns": int(timestamps[s]),
        "next_active_frame": j,
        "next_active_timestamp_ns": int(timestamps[j]),
        "first_visual_edge": [s, j],
        "first_imu_edge": [s, j],
        "first_imu_interval": interval,
        "formal_factor_count": int(len(formal_edges)),
        "formal_edge_crosses_boundary": crossing,
        "visual_endpoints_post_boundary": bool(timestamps[s] >= t_init_end and timestamps[j] >= t_init_end),
        "imu_interval_exact": bool(
            interval["first_knot_ns"] == int(timestamps[s])
            and interval["last_knot_ns"] == int(timestamps[j])
            and abs(interval["sum_dt_s"] - interval["image_dt_s"]) < 1.0e-12
        ),
        "pre_start_imu_support_used": bool(
            interval["contains_pre_start_knot"] or interval["contains_pre_start_raw_support"]
        ),
        "production_first_state_identity_log_norm": pose_s_identity_log,
        "production_first_state_is_identity": pose_s_identity_log < 1.0e-9,
        "production_visual_measurement_provenance": "continuous MACVO cache built from frame 0",
        "production_structurally_uses_pre_boundary_visual_run": True,
        "production_numeric_dependency_requires_gate1": True,
        "diagnostic_primary_cache": str(args.cold_cache),
        "cold_cache_first_timestamp_ns": int(cold_manifest["timestamps_ns"][0]),
        "cold_cache_frame_0_maps_to_source_frame": s,
        "continuous_cache_first_timestamp_ns": int(continuous_manifest["timestamps_ns"][0]),
        "gate0_pass": bool(not crossing and interval["first_knot_ns"] == int(timestamps[s])),
        "strict_history_contract_pass_for_existing_production": False,
        "strict_history_contract_explanation": (
            "Formal factors and the first navigation state obey the boundary, but the existing production sidecar "
            "was generated by a continuous frame-0 MACVO pass. Gates 1-7 therefore use the true cold-start cache."
        ),
    }
    return s, timestamps, active, contract


def gate0(args: argparse.Namespace, output: Path) -> tuple[int, np.ndarray, dict[str, Any]]:
    s, timestamps, active, contract = derive_active_start(args)
    formal = {(int(row.frame_i), int(row.frame_j)) for row in active.itertuples(index=False)}
    rows = []
    for frame in range(max(0, s - 6), min(len(timestamps), s + 8)):
        incoming = (frame - 1, frame)
        rows.append(
            {
                "source_frame": frame,
                "timestamp_ns": int(timestamps[frame]),
                "time_from_init_start_s": float((timestamps[frame] - timestamps[0]) * 1.0e-9),
                "timestamp_is_post_init": bool(timestamps[frame] >= contract["initialization_end_timestamp_ns"]),
                "is_active_start": frame == s,
                "official_vio_state": frame >= s,
                "formal_visual_factor_incoming": incoming in formal,
                "formal_imu_factor_incoming": incoming in formal,
                "cold_cache_local_frame": frame - s if frame >= s else -1,
                "allowed_in_primary_visual_diagnostic": frame >= s,
            }
        )
    pd.DataFrame(rows).to_csv(output / "active_start_frame_manifest.csv", index=False)
    write_json(output / "active_start_boundary_contract.json", contract)
    if contract["formal_edge_crosses_boundary"]:
        raise SystemExit("Gate 0 stop: a formal factor crosses the initialization boundary")
    return s, timestamps, contract


def packet_array(packet, name: str) -> np.ndarray:
    if name == "relative_pose_init":
        return packet.relative_pose_init.detach().cpu().numpy()
    if name == "points_local":
        return packet.points_local.detach().cpu().numpy()
    if name == "points_cov_local":
        return packet.points_cov_local.detach().cpu().numpy()
    return packet.match_fields[name].detach().cpu().numpy()


def cached_pose_matrix(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.size == 7:
        return se3_from_xyzw(array.reshape(7))
    if array.size == 16:
        return array.reshape(4, 4)
    raise ValueError(f"unsupported cached pose shape: {array.shape}")


def max_abs_or_nan(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a - b))) if a.shape == b.shape and a.size else math.nan


def diagnostics_by_edge(path: Path) -> dict[tuple[int, int], pd.Series]:
    frame = pd.read_csv(find_diagnostic_csv(path))
    return {(int(row.frame_i), int(row.frame_j)): row for _, row in frame.iterrows()}


def gate1(
    args: argparse.Namespace,
    output: Path,
    s: int,
    timestamps: np.ndarray,
) -> tuple[VisualFactorCacheReader, dict[str, np.ndarray], np.ndarray]:
    same_build_paths = (
        args.same_build_continuous_cache,
        args.same_build_cold_cache,
        args.same_build_continuous_result,
        args.same_build_cold_result,
    )
    if not all(path.exists() for path in same_build_paths):
        raise RuntimeError(
            "Gate 1 requires the same-build C0/C1 products. Run "
            "Scripts/run_startup_macvo_cold_start_ab.py first."
        )

    # Gate 1 uses a strictly same-build A/B. The full cold-start cache below is
    # retained only for Gates 2-7 because it covers the complete active slice.
    continuous_reader = VisualFactorCacheReader(args.same_build_continuous_cache)
    comparison_cold_reader = VisualFactorCacheReader(args.same_build_cold_cache)
    continuous_sidecar = load_sidecar(args.same_build_continuous_cache)
    comparison_cold_sidecar = load_sidecar(args.same_build_cold_cache)
    continuous_diag = diagnostics_by_edge(args.same_build_continuous_result)
    comparison_cold_diag = diagnostics_by_edge(args.same_build_cold_result)
    full_cold_reader = VisualFactorCacheReader(args.cold_cache)
    full_cold_sidecar = load_sidecar(args.cold_cache)

    c0_timestamps = np.asarray(continuous_reader.manifest.timestamps_ns, dtype=np.int64)
    c1_timestamps = np.asarray(comparison_cold_reader.manifest.timestamps_ns, dtype=np.int64)
    c0_start = int(np.searchsorted(c0_timestamps, timestamps[s]))
    if c0_start >= len(c0_timestamps) or c0_timestamps[c0_start] != timestamps[s]:
        raise RuntimeError("same-build C0 does not contain active-start timestamp")
    if c1_timestamps[0] != timestamps[s]:
        raise RuntimeError("same-build C1 does not start at the derived active timestamp")

    c0_source = continuous_reader.manifest.source
    c1_source = comparison_cold_reader.manifest.source
    same_git = c0_source.get("git") == c1_source.get("git")
    same_dataset = c0_source.get("dataset") == c1_source.get("dataset")
    same_calibration = bool(
        np.allclose(continuous_reader.manifest.K, comparison_cold_reader.manifest.K)
        and math.isclose(
            continuous_reader.manifest.baseline_m,
            comparison_cold_reader.manifest.baseline_m,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
    )
    if not (same_git and same_dataset and same_calibration):
        raise RuntimeError("same-build C0/C1 provenance or calibration mismatch")

    repeat_root = args.same_build_continuous_cache.parents[1]
    c0_repeat_path = repeat_root / "C0_continuous_repeat" / SCENE
    c1_repeat_path = repeat_root / "C1_cold_repeat" / SCENE
    if not (c0_repeat_path.exists() and c1_repeat_path.exists()):
        raise RuntimeError(
            "Gate 1 requires C0/C1 repeatability controls. Run "
            "Scripts/run_startup_macvo_cold_start_ab.py --repeat-controls first."
        )
    c0_repeat_reader = VisualFactorCacheReader(c0_repeat_path)
    c1_repeat_reader = VisualFactorCacheReader(c1_repeat_path)
    c0_repeat_sidecar = load_sidecar(c0_repeat_path)
    c1_repeat_sidecar = load_sidecar(c1_repeat_path)
    fields = (
        "points_local", "points_cov_local", "pixel1_uv", "pixel2_uv", "pixel1_d", "pixel2_d",
        "pixel1_uv_cov", "pixel2_uv_cov", "obs1_covTc", "obs2_covTc",
    )
    rows: list[dict[str, Any]] = []
    for local_i in range(STARTUP_EDGES):
        source_i, source_j = s + local_i, s + local_i + 1
        c0_i, c0_j = c0_start + local_i, c0_start + local_i + 1
        continuous = continuous_reader.load_pair(
            c0_i, c0_j, int(timestamps[source_i]), int(timestamps[source_j])
        )
        cold = comparison_cold_reader.load_pair(
            local_i, local_i + 1,
            int(c1_timestamps[local_i]),
            int(c1_timestamps[local_i + 1]),
        )
        z0 = se3_from_xyzw(continuous_sidecar["measurement_CiCj"][c0_i].reshape(7))
        z1 = se3_from_xyzw(comparison_cold_sidecar["measurement_CiCj"][local_i].reshape(7))
        delta = invert_transform(z0) @ z1
        c0_repeat = c0_repeat_reader.load_pair(
            c0_i, c0_j, int(timestamps[source_i]), int(timestamps[source_j])
        )
        c1_repeat = c1_repeat_reader.load_pair(
            local_i, local_i + 1, int(timestamps[source_i]), int(timestamps[source_j])
        )
        z0_repeat = se3_from_xyzw(c0_repeat_sidecar["measurement_CiCj"][c0_i].reshape(7))
        z1_repeat = se3_from_xyzw(c1_repeat_sidecar["measurement_CiCj"][local_i].reshape(7))
        delta_c0_repeat = invert_transform(z0) @ z0_repeat
        delta_c1_repeat = invert_transform(z1) @ z1_repeat
        cov0 = continuous_sidecar["covariance"][c0_i].astype(np.float64)
        cov1 = comparison_cold_sidecar["covariance"][local_i].astype(np.float64)
        cov0_repeat = c0_repeat_sidecar["covariance"][c0_i].astype(np.float64)
        cov1_repeat = c1_repeat_sidecar["covariance"][local_i].astype(np.float64)
        row: dict[str, Any] = {
            "edge_id": local_i,
            "source_frame_i": source_i,
            "source_frame_j": source_j,
            "timestamp_i_ns": int(timestamps[source_i]),
            "timestamp_j_ns": int(timestamps[source_j]),
            "continuous_point_count": int(len(continuous.points_local)),
            "cold_point_count": int(len(cold.points_local)),
            "point_count_equal": int(len(continuous.points_local)) == int(len(cold.points_local)),
            "measurement_translation_difference_norm_m": float(np.linalg.norm(delta[:3, 3])),
            "measurement_rotation_difference_norm_rad": float(np.linalg.norm(rotation_log(delta[:3, :3]))),
            "measurement_se3_difference_norm": float(
                np.linalg.norm(np.r_[delta[:3, 3], rotation_log(delta[:3, :3])])
            ),
            "continuous_repeat_measurement_se3_difference_norm": float(
                np.linalg.norm(
                    np.r_[delta_c0_repeat[:3, 3], rotation_log(delta_c0_repeat[:3, :3])]
                )
            ),
            "cold_repeat_measurement_se3_difference_norm": float(
                np.linalg.norm(
                    np.r_[delta_c1_repeat[:3, 3], rotation_log(delta_c1_repeat[:3, :3])]
                )
            ),
            "covariance_frobenius_relative_error": float(
                np.linalg.norm(cov1 - cov0) / max(np.linalg.norm(cov0), EPS)
            ),
            "continuous_repeat_covariance_frobenius_relative_error": float(
                np.linalg.norm(cov0_repeat - cov0) / max(np.linalg.norm(cov0), EPS)
            ),
            "cold_repeat_covariance_frobenius_relative_error": float(
                np.linalg.norm(cov1_repeat - cov1) / max(np.linalg.norm(cov1), EPS)
            ),
            "covariance_trace_ratio": float(np.trace(cov1) / max(np.trace(cov0), EPS)),
            "continuous_inlier_count": int(continuous_sidecar["num_inliers"][c0_i]),
            "cold_inlier_count": int(comparison_cold_sidecar["num_inliers"][local_i]),
            "continuous_final_cost": float(
                pd.to_numeric(continuous_diag[(c0_i, c0_j)].get("visual_loss_raw_sum"), errors="coerce")
            ),
            "cold_final_cost": float(
                pd.to_numeric(comparison_cold_diag[(local_i, local_i + 1)].get("visual_loss_raw_sum"), errors="coerce")
            ),
            "continuous_initial_motion_log_norm": float(
                np.linalg.norm(cached_pose_matrix(packet_array(continuous, "relative_pose_init"))[:3, 3])
                + np.linalg.norm(rotation_log(cached_pose_matrix(packet_array(continuous, "relative_pose_init"))[:3, :3]))
            ),
            "cold_initial_motion_log_norm": float(
                np.linalg.norm(cached_pose_matrix(packet_array(cold, "relative_pose_init"))[:3, 3])
                + np.linalg.norm(rotation_log(cached_pose_matrix(packet_array(cold, "relative_pose_init"))[:3, :3]))
            ),
            "continuous_repeat_points_local_same_shape": (
                packet_array(continuous, "points_local").shape
                == packet_array(c0_repeat, "points_local").shape
            ),
            "continuous_repeat_points_local_max_abs_difference": max_abs_or_nan(
                packet_array(continuous, "points_local"), packet_array(c0_repeat, "points_local")
            ),
            "cold_repeat_points_local_same_shape": (
                packet_array(cold, "points_local").shape
                == packet_array(c1_repeat, "points_local").shape
            ),
            "cold_repeat_points_local_max_abs_difference": max_abs_or_nan(
                packet_array(cold, "points_local"), packet_array(c1_repeat, "points_local")
            ),
        }
        for field in fields:
            a, b = packet_array(continuous, field), packet_array(cold, field)
            row[f"{field}_same_shape"] = a.shape == b.shape
            row[f"{field}_max_abs_difference"] = max_abs_or_nan(a, b)
            row[f"{field}_continuous_sha256"] = sha256_arrays(a)
            row[f"{field}_cold_sha256"] = sha256_arrays(b)
        rows.append(row)
    comparison = pd.DataFrame(rows)
    comparison.to_csv(output / "continuous_vs_cold_start_per_edge.csv", index=False)
    exact_fields = {
        field: float((comparison[f"{field}_max_abs_difference"].fillna(np.inf) == 0.0).mean())
        for field in fields
    }
    target_repeatability = quantiles(comparison.measurement_se3_difference_norm)
    continuous_repeatability = quantiles(
        comparison.continuous_repeat_measurement_se3_difference_norm
    )
    cold_repeatability = quantiles(comparison.cold_repeat_measurement_se3_difference_norm)
    repeatability_upper = max(
        continuous_repeatability["p95"], cold_repeatability["p95"], EPS
    )
    exceeds_repeatability = bool(
        target_repeatability["p95"] > 1.25 * repeatability_upper
    )
    reset_audit = {
        "comparison_contract": "same code/build, odometry config, dataset, timestamps, calibration and edge range",
        "continuous_process": str(args.same_build_continuous_result),
        "continuous_cache": str(args.same_build_continuous_cache),
        "cold_start_process": str(args.same_build_cold_result),
        "cold_start_cache": str(args.same_build_cold_cache),
        "full_cold_cache_for_gates_2_to_7": str(args.cold_cache),
        "same_git": same_git,
        "same_git_revision": c0_source.get("git"),
        "same_dataset": same_dataset,
        "same_calibration": same_calibration,
        "same_edge_timestamps": bool(
            np.array_equal(c0_timestamps[c0_start:c0_start + STARTUP_EDGES + 1], c1_timestamps)
        ),
        "source_sequence_starts_at_derived_frame": s,
        "frame_s_is_local_frame_zero": True,
        "T_WC_s_is_identity": True,
        "new_MACVO_instance": True,
        "previous_frame_pointer_initially_empty": True,
        "motion_model": "new StaticMotionModel instance",
        "local_optimization_graph_recreated_per_pair": True,
        "pre_s_keypoints_or_matches_loaded": False,
        "continuous_vs_cold": {
            "edge_count": int(len(comparison)),
            "measurement_se3_difference": quantiles(comparison.measurement_se3_difference_norm),
            "point_count_equal_fraction": float(comparison.point_count_equal.mean()),
            "points_local_exact_fraction": float(
                (comparison.points_local_max_abs_difference.fillna(np.inf) == 0.0).mean()
            ),
            "exact_fraction_by_cached_field": exact_fields,
            "covariance_relative_error": quantiles(comparison.covariance_frobenius_relative_error),
        },
        "run_to_run_repeatability": {
            "continuous_repeat_measurement_se3_difference": continuous_repeatability,
            "cold_repeat_measurement_se3_difference": cold_repeatability,
            "continuous_repeat_covariance_relative_error": quantiles(
                comparison.continuous_repeat_covariance_frobenius_relative_error
            ),
            "cold_repeat_covariance_relative_error": quantiles(
                comparison.cold_repeat_covariance_frobenius_relative_error
            ),
            "continuous_repeat_points_exact_fraction": float(
                (
                    comparison.continuous_repeat_points_local_max_abs_difference.fillna(np.inf)
                    == 0.0
                ).mean()
            ),
            "cold_repeat_points_exact_fraction": float(
                (
                    comparison.cold_repeat_points_local_max_abs_difference.fillna(np.inf)
                    == 0.0
                ).mean()
            ),
        },
        "startup_difference_exceeds_run_to_run_repeatability": exceeds_repeatability,
        "history_is_primary_cause": exceeds_repeatability,
        "classification_rule": (
            "history-specific effect is flagged only if C0-vs-C1 p95 exceeds 1.25 times the larger "
            "of C0-vs-C0-repeat and C1-vs-C1-repeat p95; otherwise it is not distinguishable from "
            "the unseeded random keypoint-selection baseline"
        ),
        "known_nondeterminism": (
            "CovAwareSelector_NoDepth calls torch.randperm once per pair without a pair-local seed; "
            "separate runs and different prefix lengths therefore select different subsets"
        ),
        "code_evidence": [
            "Scripts/run_startup_macvo_cold_start_ab.py launches C0 and C1 from the same checkout and odometry YAML",
            "cold cache local frame 0 is source frame s and its first pose is identity",
            "Odometry uses StaticMotionModel, so no learned constant-velocity state is inherited",
            "Module/KeypointSelector.py CovAwareSelector_NoDepth uses unseeded torch.randperm",
        ],
    }
    write_json(output / "cold_start_reset_audit.json", reset_audit)
    return full_cold_reader, full_cold_sidecar, comparison.measurement_se3_difference_norm.to_numpy()


def cold_measurements(cold_sidecar: dict[str, np.ndarray]) -> np.ndarray:
    return np.stack([se3_from_xyzw(row.reshape(7)) for row in cold_sidecar["measurement_CiCj"]])


def gate2(
    output: Path,
    s: int,
    timestamps: np.ndarray,
    truth_pose: np.ndarray,
    z_mac: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    visual = compose_edges(z_mac)
    truth_local = np.stack(
        [invert_transform(truth_pose[s]) @ truth_pose[k] for k in range(s, len(truth_pose))]
    )
    if len(visual) != len(truth_local):
        raise RuntimeError("cold visual edge count does not match active truth trajectory")
    rows_visual = []
    rows_truth = []
    for local, source in enumerate(range(s, len(timestamps))):
        common = {
            "local_frame": local,
            "source_frame": source,
            "timestamp_ns": int(timestamps[source]),
            "elapsed_s": float((timestamps[source] - timestamps[s]) * 1.0e-9),
        }
        for destination, pose in ((rows_visual, visual[local]), (rows_truth, truth_local[local])):
            nwu = pose_internal_to_nwu(pose)
            destination.append(
                {
                    **common,
                    "x_nwu": float(nwu[0, 3]), "y_nwu": float(nwu[1, 3]), "z_nwu": float(nwu[2, 3]),
                    "tx_internal": float(pose[0, 3]), "ty_internal": float(pose[1, 3]), "tz_internal": float(pose[2, 3]),
                    "qx": float(Rotation.from_matrix(pose[:3, :3]).as_quat()[0]),
                    "qy": float(Rotation.from_matrix(pose[:3, :3]).as_quat()[1]),
                    "qz": float(Rotation.from_matrix(pose[:3, :3]).as_quat()[2]),
                    "qw": float(Rotation.from_matrix(pose[:3, :3]).as_quat()[3]),
                }
            )
    pd.DataFrame(rows_visual).to_csv(output / "active_visual_trajectory.csv", index=False)
    pd.DataFrame(rows_truth).to_csv(output / "active_gt_local_trajectory.csv", index=False)
    audit = {
        "active_start_frame": s,
        "visual_T_s_identity_max_abs": float(np.max(np.abs(visual[0] - np.eye(4)))),
        "truth_T_s_identity_max_abs": float(np.max(np.abs(truth_local[0] - np.eye(4)))),
        "composition": "T(k+1) = T(k) * Z_CkCk+1",
        "source": "true cold-start relative-pose sidecar",
        "segments_exported_in_same_csv": [30, 60, 120, int(len(visual))],
        "pass": bool(
            np.max(np.abs(visual[0] - np.eye(4))) < 1.0e-12
            and np.max(np.abs(truth_local[0] - np.eye(4))) < 1.0e-12
        ),
    }
    write_json(output / "active_trajectory_rebase_audit.json", audit)
    return visual, truth_local


def safe_basis(z_gt: np.ndarray, pose_wci: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    physical_up_world_internal = np.array([0.0, 0.0, -1.0])
    physical_up_local = pose_wci[:3, :3].T @ physical_up_world_internal
    physical_up_local /= max(np.linalg.norm(physical_up_local), EPS)
    translation = z_gt[:3, 3]
    if np.linalg.norm(translation) >= 1.0e-6:
        forward = translation / np.linalg.norm(translation)
    else:
        forward = np.array([1.0, 0.0, 0.0])
    lateral = np.cross(physical_up_local, forward)
    if np.linalg.norm(lateral) < 1.0e-9:
        lateral = np.array([0.0, 1.0, 0.0])
    lateral /= np.linalg.norm(lateral)
    vertical = np.cross(forward, lateral)
    vertical /= max(np.linalg.norm(vertical), EPS)
    return forward, lateral, vertical


def gate3(
    output: Path,
    s: int,
    timestamps: np.ndarray,
    truth_pose: np.ndarray,
    z_mac: np.ndarray,
    z_gt: np.ndarray,
) -> pd.DataFrame:
    plot_up_world_internal = np.array([0.0, -1.0, 0.0])  # NWU plot +Y -> internal NED -Y.
    rows = []
    cumulative_plot_up = 0.0
    for local, (mac, gt) in enumerate(zip(z_mac, z_gt)):
        source_i, source_j = s + local, s + local + 1
        error = mac[:3, 3] - gt[:3, 3]
        plot_up_ci = truth_pose[source_i, :3, :3].T @ plot_up_world_internal
        plot_up_ci /= max(np.linalg.norm(plot_up_ci), EPS)
        forward, lateral, vertical = safe_basis(gt, truth_pose[source_i])
        cumulative_plot_up += float(plot_up_ci @ error)
        rows.append(
            {
                "edge_id": local,
                "frame_i": source_i,
                "frame_j": source_j,
                "timestamp_i_ns": int(timestamps[source_i]),
                "timestamp_j_ns": int(timestamps[source_j]),
                "elapsed_s": float((timestamps[source_i] - timestamps[s]) * 1.0e-9),
                "t_mac_x": mac[0, 3], "t_mac_y": mac[1, 3], "t_mac_z": mac[2, 3],
                "t_gt_x": gt[0, 3], "t_gt_y": gt[1, 3], "t_gt_z": gt[2, 3],
                "e_tx": error[0], "e_ty": error[1], "e_tz": error[2],
                "e_t_norm": float(np.linalg.norm(error)),
                "e_forward": float(forward @ error),
                "e_lateral": float(lateral @ error),
                "e_vertical": float(vertical @ error),
                "e_plot_up": float(plot_up_ci @ error),
                "cumulative_plot_up_local_sum_m": cumulative_plot_up,
                "plot_up_camera_x": plot_up_ci[0],
                "plot_up_camera_y": plot_up_ci[1],
                "plot_up_camera_z": plot_up_ci[2],
                "physical_vertical_camera_x": vertical[0],
                "physical_vertical_camera_y": vertical[1],
                "physical_vertical_camera_z": vertical[2],
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "startup_translation_error_per_edge.csv", index=False)
    cumulative = {
        str(count): float(frame.iloc[:count].e_plot_up.sum())
        for count in (1, 3, 5, 10, 20, 30)
    }
    write_json(
        output / "startup_direction_contract.json",
        {
            "plot_horizontal_axis": "world NWU +X",
            "plot_vertical_axis": "world NWU +Y",
            "plot_up_world_nwu": [0.0, 1.0, 0.0],
            "plot_up_world_internal_ned": plot_up_world_internal,
            "nwu_to_internal_ned": np.diag([1.0, -1.0, -1.0]),
            "physical_vertical_up_world_nwu": [0.0, 0.0, 1.0],
            "physical_vertical_up_world_internal_ned": [0.0, 0.0, -1.0],
            "camera_frame": "MACVO internal NED-like camera coordinates with +x depth",
            "translation_measurement": "T_CiCj translation expressed in C_i",
            "cumulative_plot_up_error_m": cumulative,
        },
    )
    return frame


def segment_metrics(predicted: np.ndarray, truth: np.ndarray, count: int) -> dict[str, Any]:
    endpoint = min(count + 1, len(predicted), len(truth))
    p_error, r_error = pose_error(predicted[:endpoint], truth[:endpoint])
    rpe_t, rpe_r = edge_error(predicted[:endpoint], truth[:endpoint])
    return {
        "frames": endpoint,
        "position_rmse_m": float(np.sqrt(np.mean(np.sum(p_error * p_error, axis=1)))),
        "position_final_error_m": float(np.linalg.norm(p_error[-1])),
        "orientation_rmse_rad": float(np.sqrt(np.mean(np.sum(r_error * r_error, axis=1)))),
        "translation_rpe_rmse_m": float(np.sqrt(np.mean(np.sum(rpe_t * rpe_t, axis=1)))),
        "rotation_rpe_rmse_rad": float(np.sqrt(np.mean(np.sum(rpe_r * rpe_r, axis=1)))),
        "final_error_internal": p_error[-1],
        "final_plot_up_error_nwu_y_m": float(pose_internal_to_nwu(predicted[endpoint - 1])[1, 3] - pose_internal_to_nwu(truth[endpoint - 1])[1, 3]),
    }


def gate4(output: Path, z_mac: np.ndarray, z_gt: np.ndarray, truth_local: np.ndarray) -> dict[str, np.ndarray]:
    variants: dict[str, np.ndarray] = {}
    definitions = {
        "A0": (0, False), "A1": (1, False), "A3": (3, False), "A5": (5, False),
        "A10": (10, False), "A20": (20, False), "AR5": (5, True), "ART5": (5, True),
    }
    trajectories = []
    summary: dict[str, Any] = {"modes": {}}
    for mode, (count, replace_rotation) in definitions.items():
        edges = z_mac.copy()
        if mode.startswith("A") and mode not in ("AR5", "ART5") and count:
            edges[:count, :3, 3] = z_gt[:count, :3, 3]
        if mode == "AR5":
            edges[:5, :3, :3] = z_gt[:5, :3, :3]
        if mode == "ART5":
            edges[:5] = z_gt[:5]
        trajectory = compose_edges(edges)
        variants[mode] = trajectory
        summary["modes"][mode] = {
            "replacement": {
                "translation_edges": count if mode not in ("AR5",) else 0,
                "rotation_edges": 5 if replace_rotation else 0,
            },
            "first_30": segment_metrics(trajectory, truth_local, 30),
            "first_60": segment_metrics(trajectory, truth_local, 60),
            "first_120": segment_metrics(trajectory, truth_local, 120),
        }
        for local in range(min(121, len(trajectory))):
            nwu = pose_internal_to_nwu(trajectory[local])
            trajectories.append(
                {"mode": mode, "local_frame": local, "x_nwu": nwu[0, 3], "y_nwu": nwu[1, 3], "z_nwu": nwu[2, 3]}
            )
    a0 = summary["modes"]["A0"]["first_120"]["position_rmse_m"]
    improvements = {
        mode: float((a0 - summary["modes"][mode]["first_120"]["position_rmse_m"]) / max(a0, EPS))
        for mode in summary["modes"]
    }
    if improvements["A1"] >= 0.7 * max(improvements["A20"], EPS):
        classification = "first_edge_dominant"
    elif improvements["A5"] >= 0.8 * max(improvements["A20"], EPS):
        classification = "first_five_edges_dominant"
    else:
        classification = "persistent_visual_translation_bias"
    summary["first_120_improvement_fraction_vs_A0"] = improvements
    summary["classification"] = classification
    summary["gt_usage"] = "offline oracle only; no production measurement is modified"
    pd.DataFrame(trajectories).to_csv(output / "startup_edge_oracle_trajectories.csv", index=False)
    write_json(output / "startup_edge_oracle_summary.json", summary)
    return variants


def projection_rotation_only_flow(packet, transform: np.ndarray) -> np.ndarray:
    points_i = packet.points_local.double()
    rotation_ji = torch.from_numpy(transform[:3, :3].T).double()
    points_j_rotation_only = (rotation_ji @ points_i.mT).mT
    uv_rotation_only = point2pixel_NED(points_j_rotation_only, packet.K.double()).numpy()
    uv1 = packet.match_fields["pixel1_uv"].double().numpy()
    uv2 = packet.match_fields["pixel2_uv"].double().numpy()
    return (uv2 - uv1) - (uv_rotation_only - uv1)


def inv_sqrt_psd(matrix: np.ndarray) -> np.ndarray:
    values, vectors = np.linalg.eigh(0.5 * (matrix + matrix.T))
    floor = max(float(np.max(np.abs(values))) * 1.0e-12, 1.0e-12)
    return vectors @ np.diag(np.maximum(values, floor) ** -0.5) @ vectors.T


def solve_psd(matrix: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    try:
        return np.linalg.solve(matrix, rhs)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(matrix, rcond=1.0e-12) @ rhs


def uvd_linearization(packet, transform: np.ndarray) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    points = packet.points_local.numpy().astype(np.float64)
    observation = np.column_stack(
        [packet.match_fields["pixel2_uv"].numpy(), packet.match_fields["pixel2_disp"].numpy().reshape(-1)]
    )
    covariance = uvd.observation_covariance(packet)
    terms = uvd.uvd_terms(
        transform, points, observation, covariance, packet.K.numpy(), packet.baseline_m,
        use_covariance=True, robust=True,
    )
    sqrt_weight = np.sqrt(terms["weight"])[:, None, None]
    jacobian = sqrt_weight * terms["white_jacobian"]
    residual = sqrt_weight[..., 0] * terms["white_residual"]
    hessian = np.einsum("nki,nkj->ij", jacobian, jacobian)
    gradient = np.einsum("nki,nk->i", jacobian, residual)
    return terms, 0.5 * (hessian + hessian.T), gradient


def scalar_corr(frame: pd.DataFrame, target: str, signal: str) -> dict[str, Any]:
    data = frame[[target, signal]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(data) < 3 or data[target].nunique() < 2 or data[signal].nunique() < 2:
        return {"count": int(len(data)), "pearson": None, "spearman": None}
    p = pearsonr(data[target], data[signal])
    s = spearmanr(data[target], data[signal])
    return {"count": int(len(data)), "pearson": float(p.statistic), "spearman": float(s.statistic)}


def gate5(
    output: Path,
    cold_reader: VisualFactorCacheReader,
    s: int,
    truth_pose: np.ndarray,
    z_mac: np.ndarray,
    z_gt: np.ndarray,
    translation_errors: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for local in range(STARTUP_EDGES):
        packet = cold_reader.load_pair(
            local, local + 1,
            int(cold_reader.manifest.timestamps_ns[local]), int(cold_reader.manifest.timestamps_ns[local + 1]),
        )
        terms, hessian, gradient = uvd_linearization(packet, z_mac[local])
        h_tt, h_tR, h_Rt, h_RR = hessian[:3, :3], hessian[:3, 3:], hessian[3:, :3], hessian[3:, 3:]
        values_t, vectors_t = np.linalg.eigh(h_tt)
        values_r = np.linalg.eigvalsh(h_RR)
        weak = vectors_t[:, 0]
        plot_up_ci = truth_pose[s + local, :3, :3].T @ np.array([0.0, -1.0, 0.0])
        plot_up_tangent = z_mac[local, :3, :3].T @ plot_up_ci
        plot_up_tangent /= max(np.linalg.norm(plot_up_tangent), EPS)
        weak_angle = float(math.acos(np.clip(abs(weak @ plot_up_tangent), -1.0, 1.0)))
        coupling = inv_sqrt_psd(h_RR) @ h_Rt @ inv_sqrt_psd(h_tt)
        schur_t = h_tt - h_tR @ solve_psd(h_RR, h_Rt)
        schur_r = h_RR - h_Rt @ solve_psd(h_tt, h_tR)
        flow = packet.match_fields["pixel2_uv"].numpy() - packet.match_fields["pixel1_uv"].numpy()
        derotated = projection_rotation_only_flow(packet, z_mac[local])
        uv = packet.match_fields["pixel2_uv"].numpy()
        depth = np.r_[packet.match_fields["pixel1_d"].numpy().reshape(-1), packet.match_fields["pixel2_d"].numpy().reshape(-1)]
        depth_cov = np.r_[packet.match_fields["pixel1_d_cov"].numpy().reshape(-1), packet.match_fields["pixel2_d_cov"].numpy().reshape(-1)]
        flow_cov = packet.match_fields["pixel2_uv_cov"].numpy()
        row: dict[str, Any] = {
            "edge_id": local,
            "frame_i": s + local,
            "frame_j": s + local + 1,
            "point_count": int(len(terms["weight"])),
            "inlier_count_huber": int(np.sum(terms["norm"] <= uvd.HUBER_DELTA)),
            "inlier_ratio_huber": float(np.mean(terms["norm"] <= uvd.HUBER_DELTA)),
            "huber_trigger_ratio": float(np.mean(terms["weight"] < 1.0)),
            "uvd_final_cost": float(np.sum(terms["cost"])),
            "uvd_whitened_norm_median": float(np.median(terms["norm"])),
            "median_flow_u": float(np.median(flow[:, 0])),
            "median_flow_v": float(np.median(flow[:, 1])),
            "median_derotated_flow_u": float(np.median(derotated[:, 0])),
            "median_derotated_flow_v": float(np.median(derotated[:, 1])),
            "median_derotated_flow_norm": float(np.median(np.linalg.norm(derotated, axis=1))),
            "median_depth_m": float(np.median(depth)),
            "median_depth_cov": float(np.median(depth_cov)),
            "median_flow_cov": float(np.median(flow_cov[:, 0] + flow_cov[:, 1])),
            "image_coverage_fraction": float(
                (np.ptp(uv[:, 0]) * np.ptp(uv[:, 1])) / max(2.0 * packet.K[0, 2].item() * 2.0 * packet.K[1, 2].item(), EPS)
            ),
            "rotation_error_norm_rad": float(np.linalg.norm(rotation_log(z_mac[local, :3, :3].T @ z_gt[local, :3, :3]))),
            "translation_plot_up_error_m": float(translation_errors.iloc[local].e_plot_up),
            "translation_error_norm_m": float(translation_errors.iloc[local].e_t_norm),
            "H_tt_eig_min": float(values_t[0]), "H_tt_eig_mid": float(values_t[1]), "H_tt_eig_max": float(values_t[2]),
            "H_tt_condition": float(values_t[-1] / max(values_t[0], EPS)),
            "H_RR_eig_min": float(values_r[0]), "H_RR_eig_max": float(values_r[-1]),
            "H_RR_condition": float(values_r[-1] / max(values_r[0], EPS)),
            "normalized_Rt_coupling_sigma_max": float(np.linalg.svd(coupling, compute_uv=False)[0]),
            "schur_translation_eig_min": float(np.linalg.eigvalsh(0.5 * (schur_t + schur_t.T))[0]),
            "schur_rotation_eig_min": float(np.linalg.eigvalsh(0.5 * (schur_r + schur_r.T))[0]),
            "weak_translation_angle_to_plot_up_rad": weak_angle,
            "weak_translation_angle_to_plot_up_deg": math.degrees(weak_angle),
            "weak_tangent_x": weak[0], "weak_tangent_y": weak[1], "weak_tangent_z": weak[2],
            "plot_up_tangent_x": plot_up_tangent[0], "plot_up_tangent_y": plot_up_tangent[1], "plot_up_tangent_z": plot_up_tangent[2],
        }
        for r in range(6):
            row[f"g_{r}"] = float(gradient[r])
            for c in range(6):
                row[f"H_{r}{c}"] = float(hessian[r, c])
        for axis in range(3):
            for eig in range(3):
                row[f"Htt_eigvec_{axis}{eig}"] = float(vectors_t[axis, eig])
        rows.append(row)
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "startup_uvd_observability_per_edge.csv", index=False)
    signals = [
        "H_tt_eig_min", "H_tt_condition", "normalized_Rt_coupling_sigma_max",
        "weak_translation_angle_to_plot_up_deg", "median_derotated_flow_norm", "median_depth_cov",
        "median_flow_cov", "point_count", "inlier_ratio_huber", "image_coverage_fraction",
    ]
    correlations = {signal: scalar_corr(frame, "translation_plot_up_error_m", signal) for signal in signals}
    worst = frame.iloc[np.argsort(-np.abs(frame.translation_plot_up_error_m.to_numpy()))[:5]]
    summary = {
        "tangent_order": "[translation, rotation] right perturbation of T_CiCj",
        "H_blocks": {"H_tt": "rows/cols 0:3", "H_RR": "rows/cols 3:6"},
        "production_UVD_contract": {
            "residual": "predicted target [u,v,disparity] - observed target [u,v,disparity]",
            "fixed_source": "frame-i local 3D points",
            "covariance": "target pixel2_uv_cov plus pixel2_disp_cov",
            "huber_delta": uvd.HUBER_DELTA,
        },
        "H_tt_condition": quantiles(frame.H_tt_condition),
        "coupling_sigma_max": quantiles(frame.normalized_Rt_coupling_sigma_max),
        "weak_direction_angle_to_plot_up_deg": quantiles(frame.weak_translation_angle_to_plot_up_deg),
        "correlations_with_plot_up_error": correlations,
        "worst_plot_up_edges": worst[[
            "frame_i", "translation_plot_up_error_m", "H_tt_condition", "normalized_Rt_coupling_sigma_max",
            "weak_translation_angle_to_plot_up_deg", "median_depth_cov", "inlier_ratio_huber",
        ]].to_dict("records"),
    }
    write_json(output / "startup_uvd_observability_summary.json", summary)
    return frame


def optimize_uvd(
    packet,
    initial: np.ndarray,
    *,
    fixed_rotation: np.ndarray | None = None,
    max_iterations: int = 50,
    small_joint: bool = False,
) -> tuple[np.ndarray, dict[str, Any]]:
    points = packet.points_local.numpy().astype(np.float64)
    observation = np.column_stack(
        [packet.match_fields["pixel2_uv"].numpy(), packet.match_fields["pixel2_disp"].numpy().reshape(-1)]
    )
    covariance = uvd.observation_covariance(packet)
    transform = initial.copy()
    if fixed_rotation is not None:
        transform[:3, :3] = fixed_rotation
    damping = 1.0e-3
    accepted = rejected = 0
    converged = False
    current_terms, current_h, current_g = uvd_linearization(packet, transform)
    current_cost = initial_cost = float(np.sum(current_terms["cost"]))
    last_step = math.inf
    for iteration in range(max_iterations):
        terms, hessian, gradient = uvd_linearization(packet, transform)
        active_h = hessian[:3, :3] if fixed_rotation is not None else hessian
        active_g = gradient[:3] if fixed_rotation is not None else gradient
        system = active_h + damping * np.diag(np.maximum(np.abs(np.diag(active_h)), 1.0))
        step = solve_psd(system, -active_g)
        if small_joint and fixed_rotation is None:
            t_norm, r_norm = np.linalg.norm(step[:3]), np.linalg.norm(step[3:])
            if t_norm > 0.002:
                step[:3] *= 0.002 / t_norm
            if r_norm > math.radians(0.1):
                step[3:] *= math.radians(0.1) / r_norm
        last_step = float(np.linalg.norm(step))
        if last_step < 1.0e-10:
            converged = True
            break
        if fixed_rotation is None:
            candidate = d3.right_update(transform, step)
        else:
            candidate = transform.copy()
            candidate[:3, 3] += fixed_rotation @ step
            candidate[:3, :3] = fixed_rotation
        candidate_terms, _, _ = uvd_linearization(packet, candidate)
        candidate_cost = float(np.sum(candidate_terms["cost"]))
        if candidate_cost < current_cost:
            transform, current_cost = candidate, candidate_cost
            accepted += 1
            damping = max(damping * 0.25, 1.0e-12)
            if abs(float(np.sum(terms["cost"])) - candidate_cost) < 1.0e-12:
                converged = True
                break
        else:
            rejected += 1
            damping = min(damping * 10.0, 1.0e12)
    final_terms, final_h, final_g = uvd_linearization(packet, transform)
    return transform, {
        "iterations": iteration + 1,
        "accepted_steps": accepted,
        "rejected_steps": rejected,
        "converged": converged,
        "initial_cost": initial_cost,
        "final_cost": float(np.sum(final_terms["cost"])),
        "final_step_norm": last_step,
        "final_gradient_norm": float(np.linalg.norm(final_g)),
        "H_tt_condition": float(
            np.linalg.eigvalsh(final_h[:3, :3])[-1] / max(np.linalg.eigvalsh(final_h[:3, :3])[0], EPS)
        ),
    }


def weighted_kabsch_rotation(packet, reference_transform: np.ndarray) -> np.ndarray:
    points_i = packet.points_local.numpy().astype(np.float64)
    points_j = pixel2point_NED(
        packet.match_fields["pixel2_uv"].double().unsqueeze(0),
        packet.match_fields["pixel2_d"].double().reshape(1, -1),
        packet.K.double(),
    ).squeeze(0).numpy()
    terms, _, _ = uvd_linearization(packet, reference_transform)
    weights = terms["weight"].astype(np.float64)
    weights /= max(np.sum(weights), EPS)
    center_i = np.sum(weights[:, None] * points_i, axis=0)
    center_j = np.sum(weights[:, None] * points_j, axis=0)
    cross = (weights[:, None] * (points_i - center_i)).T @ (points_j - center_j)
    u, _, vt = np.linalg.svd(cross)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1.0
        rotation = u @ vt
    return rotation


def rt_error(estimate: np.ndarray, truth: np.ndarray, pose_wci: np.ndarray) -> dict[str, float]:
    error = estimate[:3, 3] - truth[:3, 3]
    forward, lateral, vertical = safe_basis(truth, pose_wci)
    plot_up = pose_wci[:3, :3].T @ np.array([0.0, -1.0, 0.0])
    return {
        "translation_error_norm_m": float(np.linalg.norm(error)),
        "translation_error_forward_m": float(forward @ error),
        "translation_error_lateral_m": float(lateral @ error),
        "translation_error_vertical_m": float(vertical @ error),
        "translation_error_plot_up_m": float(plot_up @ error),
        "rotation_error_norm_rad": float(np.linalg.norm(rotation_log(estimate[:3, :3].T @ truth[:3, :3]))),
    }


def gate6(
    output: Path,
    cold_reader: VisualFactorCacheReader,
    s: int,
    truth_pose: np.ndarray,
    z_mac: np.ndarray,
    z_gt: np.ndarray,
    truth_local: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, list[np.ndarray]]]:
    modes = ("U0", "U1", "U2", "U3", "U4", "U5")
    estimates: dict[str, list[np.ndarray]] = {mode: [] for mode in modes}
    rows = []
    for local in range(STARTUP_EDGES):
        packet = cold_reader.load_pair(
            local, local + 1,
            int(cold_reader.manifest.timestamps_ns[local]), int(cold_reader.manifest.timestamps_ns[local + 1]),
        )
        rotation_u2 = weighted_kabsch_rotation(packet, z_mac[local])
        u0_terms, u0_h, u0_g = uvd_linearization(packet, z_mac[local])
        results: dict[str, tuple[np.ndarray, dict[str, Any]]] = {
            "U0": (z_mac[local], {
                "iterations": 0, "converged": True, "initial_cost": float(np.sum(u0_terms["cost"])),
                "final_cost": float(np.sum(u0_terms["cost"])), "final_step_norm": 0.0,
                "final_gradient_norm": float(np.linalg.norm(u0_g)),
                "H_tt_condition": float(np.linalg.cond(u0_h[:3, :3])),
            })
        }
        results["U1"] = optimize_uvd(packet, z_mac[local], fixed_rotation=z_mac[local, :3, :3])
        results["U2"] = optimize_uvd(packet, z_mac[local], fixed_rotation=rotation_u2)
        results["U3"] = optimize_uvd(packet, z_mac[local], fixed_rotation=z_gt[local, :3, :3])
        results["U4"] = optimize_uvd(packet, results["U2"][0], max_iterations=1, small_joint=True)
        results["U5"] = optimize_uvd(packet, results["U2"][0], max_iterations=50)
        for mode, (estimate, solver) in results.items():
            estimates[mode].append(estimate)
            rows.append(
                {
                    "edge_id": local, "frame_i": s + local, "frame_j": s + local + 1, "mode": mode,
                    **rt_error(estimate, z_gt[local], truth_pose[s + local]),
                    **solver,
                    "rotation_candidate": (
                        "production_joint" if mode == "U0" else
                        "production_R" if mode == "U1" else
                        "weighted_centered_3D_visual_R" if mode in ("U2", "U4", "U5") else "GT_R_offline_oracle"
                    ),
                }
            )
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "startup_rt_decoupling_per_edge.csv", index=False)
    summary: dict[str, Any] = {"modes": {}}
    for mode in modes:
        group = frame[frame["mode"] == mode]
        edges120 = z_mac[:120].copy()
        edges120[:STARTUP_EDGES] = np.stack(estimates[mode])
        trajectory = compose_edges(edges120)
        summary["modes"][mode] = {
            "translation_error_norm_m": quantiles(group.translation_error_norm_m),
            "translation_plot_up_error_m": quantiles(group.translation_error_plot_up_m),
            "rotation_error_norm_rad": quantiles(group.rotation_error_norm_rad),
            "UVD_final_cost": quantiles(group.final_cost),
            "H_tt_condition": quantiles(group.H_tt_condition),
            "first_120_trajectory": segment_metrics(trajectory, truth_local, 120),
        }
    u0 = summary["modes"]["U0"]["translation_plot_up_error_m"]["mean"]
    u3 = summary["modes"]["U3"]["translation_plot_up_error_m"]["mean"]
    summary["gt_rotation_plot_up_mean_change_m"] = float(u3 - u0)
    summary["visual_rotation_candidate"] = (
        "weighted centered 3D Procrustes on the same matches; uses neither GT nor IMU"
    )
    summary["U4_small_joint_caps"] = {"translation_m": 0.002, "rotation_deg": 0.1, "iterations": 1}
    write_json(output / "startup_rt_decoupling_summary.json", summary)
    return frame, estimates


def three_d_inputs(packet) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    points_i = packet.points_local.numpy().astype(np.float64)
    points_j = pixel2point_NED(
        packet.match_fields["pixel2_uv"].double().unsqueeze(0),
        packet.match_fields["pixel2_d"].double().reshape(1, -1),
        packet.K.double(),
    ).squeeze(0).numpy()
    covariance_i = packet.match_fields["obs1_covTc"].numpy().astype(np.float64)
    covariance_j = packet.match_fields["obs2_covTc"].numpy().astype(np.float64)
    return points_i, points_j, covariance_i, covariance_j


def gate7(
    output: Path,
    cold_reader: VisualFactorCacheReader,
    s: int,
    truth_pose: np.ndarray,
    z_mac: np.ndarray,
    z_gt: np.ndarray,
    truth_local: np.ndarray,
    rt_estimates: dict[str, list[np.ndarray]],
) -> pd.DataFrame:
    modes = ("F0", "F1", "F2", "F3", "F4", "F5")
    estimates: dict[str, list[np.ndarray]] = {mode: [] for mode in modes}
    rows = []
    for local in range(STARTUP_EDGES):
        packet = cold_reader.load_pair(
            local, local + 1,
            int(cold_reader.manifest.timestamps_ns[local]), int(cold_reader.manifest.timestamps_ns[local + 1]),
        )
        points_i, points_j, cov_i, cov_j = three_d_inputs(packet)
        rotation_u2 = rt_estimates["U2"][local][:3, :3]
        depth_cov = (
            packet.match_fields["pixel1_d_cov"].numpy().reshape(-1)
            + packet.match_fields["pixel2_d_cov"].numpy().reshape(-1)
        )
        all_points = np.ones(len(points_i), dtype=bool)
        low_depth = depth_cov <= np.median(depth_cov)
        results: dict[str, tuple[np.ndarray, dict[str, Any], str, np.ndarray]] = {}
        start = time.perf_counter()
        u0_terms, u0_h, u0_g = uvd_linearization(packet, z_mac[local])
        results["F0"] = (z_mac[local], {
            "iterations": 0, "converged": True, "initial_cost": float(np.sum(u0_terms["cost"])),
            "final_cost": float(np.sum(u0_terms["cost"])), "final_step_norm": 0.0,
            "final_gradient_norm": float(np.linalg.norm(u0_g)),
        }, "UVD", all_points)
        runtime_f0 = time.perf_counter() - start
        settings = {
            "F1": (None, all_points),
            "F2": (z_mac[local, :3, :3], all_points),
            "F3": (rotation_u2, all_points),
            "F5": (rotation_u2, low_depth),
        }
        runtimes: dict[str, float] = {"F0": runtime_f0}
        for mode, (fixed_rotation, selected) in settings.items():
            start = time.perf_counter()
            estimate, solver = d3.optimize_pose(
                z_mac[local], points_i, points_j, cov_i, cov_j, selected,
                fixed_rotation=fixed_rotation, use_covariance=True, robust=True,
            )
            runtimes[mode] = time.perf_counter() - start
            results[mode] = (estimate, solver, "3D-3D", selected)
        results["F4"] = (rt_estimates["U2"][local], {
            "iterations": 0, "converged": True,
            "initial_cost": float(frame_value := np.sum(uvd_linearization(packet, rt_estimates["U2"][local])[0]["cost"])),
            "final_cost": float(frame_value), "final_step_norm": 0.0, "final_gradient_norm": math.nan,
        }, "UVD", all_points)
        runtimes["F4"] = 0.0
        for mode in modes:
            estimate, solver, residual_type, selected = results[mode]
            estimates[mode].append(estimate)
            if residual_type == "3D-3D":
                terms = d3.point_terms(
                    estimate, points_i, points_j, cov_i, cov_j, use_covariance=True, robust=True
                )
                sqrt_w = np.sqrt(terms["weight"][selected])[:, None, None]
                j = sqrt_w * terms["whitened_jacobian"][selected]
                h_tt = np.einsum("nki,nkj->ij", j[:, :, :3], j[:, :, :3])
                mahal = terms["mahalanobis_norm"][selected]
                robust_weight = terms["weight"][selected]
                residual_cov_trace = np.trace(terms["covariance"][selected], axis1=1, axis2=2)
            else:
                terms_u, h_u, _ = uvd_linearization(packet, estimate)
                h_tt = h_u[:3, :3]
                mahal = terms_u["norm"][selected]
                robust_weight = terms_u["weight"][selected]
                residual_cov_trace = np.trace(uvd.observation_covariance(packet)[selected], axis1=1, axis2=2)
            eig = np.linalg.eigvalsh(0.5 * (h_tt + h_tt.T))
            rows.append(
                {
                    "edge_id": local, "frame_i": s + local, "frame_j": s + local + 1,
                    "mode": mode, "residual_type": residual_type,
                    "selected_point_count": int(np.sum(selected)),
                    **rt_error(estimate, z_gt[local], truth_pose[s + local]),
                    "H_tt_eig_min": float(eig[0]), "H_tt_eig_max": float(eig[-1]),
                    "H_tt_condition": float(eig[-1] / max(eig[0], EPS)),
                    "mahalanobis_norm_median": float(np.median(mahal)),
                    "robust_weight_mean": float(np.mean(robust_weight)),
                    "residual_covariance_trace_median": float(np.median(residual_cov_trace)),
                    "runtime_s": float(runtimes[mode]),
                    **solver,
                }
            )
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "startup_uvd_vs_3d3d_per_edge.csv", index=False)
    summary: dict[str, Any] = {"modes": {}}
    for mode in modes:
        group = frame[frame["mode"] == mode]
        edges = z_mac[:120].copy()
        edges[:STARTUP_EDGES] = np.stack(estimates[mode])
        summary["modes"][mode] = {
            "residual_type": str(group.residual_type.iloc[0]),
            "translation_error_norm_m": quantiles(group.translation_error_norm_m),
            "translation_plot_up_error_m": quantiles(group.translation_error_plot_up_m),
            "rotation_error_norm_rad": quantiles(group.rotation_error_norm_rad),
            "H_tt_condition": quantiles(group.H_tt_condition),
            "runtime_s": quantiles(group.runtime_s),
            "first_120_trajectory": segment_metrics(compose_edges(edges), truth_local, 120),
        }
    summary["contract"] = {
        "same_matches": True,
        "3D_3D_covariance": "Sigma_i + R Sigma_j R^T, full 3x3 per endpoint",
        "3D_3D_huber_delta": d3.HUBER_DELTA,
        "UVD_huber_delta": uvd.HUBER_DELTA,
        "F5_selection": "lower half by sum of endpoint depth covariance; no GT used",
    }
    write_json(output / "startup_uvd_vs_3d3d_summary.json", summary)
    return frame


def make_policy_summary(output: Path, oracle: dict[str, np.ndarray], obs: pd.DataFrame, rt: pd.DataFrame, f: pd.DataFrame) -> None:
    # Gate 8 is intentionally not silently approximated by a visual-only chain.
    # It needs a controlled VIO replay after the visual evidence selects a policy.
    u0 = rt[rt["mode"] == "U0"]
    u2 = rt[rt["mode"] == "U2"]
    f3 = f[f["mode"] == "F3"]
    payload = {
        "gate8_executed": False,
        "reason": (
            "Gates 1-7 are visual diagnostics. V1-V5 require new diagnostic factor policies inside the "
            "two-state VIO. Substituting a visual-only trajectory would not satisfy the requested same-IMU, "
            "same-prior N=2 contract, so no fabricated Gate-8 result is reported."
        ),
        "approval_status": "pending evidence review",
        "candidate_non_gt_observability_rule": {
            "signals": ["H_tt minimum eigenvalue", "H_tt condition", "derotated flow", "low-depth-cov point count"],
            "thresholds": "must be selected without GT on a separate calibration split",
        },
        "evidence_for_next_replay": {
            "oracle_classification": json.loads((output / "startup_edge_oracle_summary.json").read_text())["classification"],
            "U0_plot_up_mean_m": float(u0.translation_error_plot_up_m.mean()),
            "U2_plot_up_mean_m": float(u2.translation_error_plot_up_m.mean()),
            "F3_plot_up_mean_m": float(f3.translation_error_plot_up_m.mean()),
            "median_Htt_condition": float(obs.H_tt_condition.median()),
        },
        "planned_modes": ["V0", "V1", "V2", "V3", "V4", "V5"],
        "production_code_modified": False,
    }
    write_json(output / "startup_visual_policy_summary.json", payload)


def build_report(output: Path, contract: dict[str, Any]) -> None:
    cold = json.loads((output / "cold_start_reset_audit.json").read_text(encoding="utf-8"))
    oracle = json.loads((output / "startup_edge_oracle_summary.json").read_text(encoding="utf-8"))
    obs = json.loads((output / "startup_uvd_observability_summary.json").read_text(encoding="utf-8"))
    rt = json.loads((output / "startup_rt_decoupling_summary.json").read_text(encoding="utf-8"))
    f = json.loads((output / "startup_uvd_vs_3d3d_summary.json").read_text(encoding="utf-8"))
    policy = json.loads((output / "startup_visual_policy_summary.json").read_text(encoding="utf-8"))
    history = "数值上不是主因" if not cold["history_is_primary_cause"] else "仍是主因候选"
    u0 = rt["modes"]["U0"]["translation_plot_up_error_m"]["mean"]
    u2 = rt["modes"]["U2"]["translation_plot_up_error_m"]["mean"]
    u3 = rt["modes"]["U3"]["translation_plot_up_error_m"]["mean"]
    f3 = f["modes"]["F3"]["translation_plot_up_error_m"]["mean"]
    f4 = f["modes"]["F4"]["translation_plot_up_error_m"]["mean"]
    report = f"""# 三秒 IMU 静止初始化后的 MACVO 启动向上漂移诊断

## 首页结论

1. **前三秒 MACVO 是否完全没有进入现有正式轨迹和优化器：** 形式上的因子边界正确，但严格来源契约不完全满足。第一正式视觉/IMU边均为 `{contract['first_visual_edge'][0]}->{contract['first_visual_edge'][1]}`，不存在跨界边；第一 VIO 状态近似单位阵。不过现有生产 relative-pose sidecar 来自 frame 0 连续 MACVO。主诊断因此改用真正从活动帧冷启动的缓存。
2. **MACVO 是否在活动帧真正建立新起点：** 冷启动对照是。新进程以 frame s 为 local frame 0，`T_WC_s=I`，不加载 s 以前的点或匹配。连续与冷启动前 30 边差异表明视觉历史{history}。
3. **向上漂移来自前几条边还是持续偏差：** Gate 4 分类为 `{oracle['classification']}`。A1/A3/A5/A10/A20 的 30/60/120 帧误差见 `startup_edge_oracle_summary.json`。
4. **图中向上是否为 H_tt 弱方向：** 最弱方向与 plot-up 的夹角中位数为 `{obs['weak_direction_angle_to_plot_up_deg']['median']:.2f}` 度；H_tt 条件数中位数 `{obs['H_tt_condition']['median']:.3e}`。只有夹角小且误差/条件数相关显著时，才能判为弱可观方向。
5. **R/t 耦合解释多少：** U0 的前 30 边 plot-up 均值 `{u0:.6e}` m/edge；固定非 GT 视觉旋转的 U2 为 `{u2:.6e}`，GT 旋转 oracle U3 为 `{u3:.6e}`。这三者的差是本轮对耦合贡献的直接量化，不能用最终轨迹外观替代。
6. **3D-3D 是否有独立收益：** 在同一个 U2 旋转下，3D-3D F3 的 plot-up 均值 `{f3:.6e}`，UVD F4 为 `{f4:.6e}` m/edge。二者差异才是残差形式的独立收益。
7. **是否立刻采用 rotation-only 或方向性降权：** 尚未批准。Gate 8 没有用视觉轨迹冒充 N=2 VIO；需要根据本报告选定一个不使用 GT 的可观性规则后，单独实现并做相同 IMU/prior 的受控 replay。
8. **生产修复应落在哪层：** 先依据 Gate 6/7 判定是 R/t 解耦、depth-aware 3D-3D，还是输入观测均值；不是在输出端加 EKF，也不是改 IMU sigma。

## Gate 0：唯一活动起点

- `t_init_start = {contract['initialization_start_timestamp_ns']}` ns。
- `t_init_end = {contract['initialization_end_timestamp_ns']}` ns。
- 时间戳推导的 `s = {contract['active_start_frame']}`，`t_s = {contract['active_start_timestamp_ns']}` ns。
- 第一视觉边和第一 IMU 边均为 `{contract['first_visual_edge']}`。
- IMU knot 区间为 `[{contract['first_imu_interval']['first_knot_ns']}, {contract['first_imu_interval']['last_knot_ns']}]`，无 pre-s support。

现有生产路径的“边界”与“来源”必须分开看：边界没有跨越 3 秒，但 sidecar 的生成来源是连续 MACVO。冷启动 Gate 1 用于判断这种来源是否造成实际数值污染。

## Gate 1：连续与冷启动

- measurement SE(3) 差：median `{cold['continuous_vs_cold']['measurement_se3_difference']['median']:.3e}`，p95 `{cold['continuous_vs_cold']['measurement_se3_difference']['p95']:.3e}`，max `{cold['continuous_vs_cold']['measurement_se3_difference']['max']:.3e}`。
- 点数相同比例 `{cold['continuous_vs_cold']['point_count_equal_fraction']:.1%}`。
- pose covariance Frobenius 相对差 median `{cold['continuous_vs_cold']['covariance_relative_error']['median']:.3e}`。

## Gate 2–4：局部轨迹与启动边 Oracle

纯视觉与 GT 都在 frame s 重基准到单位阵，轨迹由冷启动边直接连乘。GT 替换只存在于离线 oracle CSV/JSON，不写入任何缓存。分类：`{oracle['classification']}`。

## Gate 5：UVD 可观性

- UVD 右扰动切空间排列为 `[translation, rotation]`。
- H_tt condition median/p95/max：`{obs['H_tt_condition']['median']:.3e}` / `{obs['H_tt_condition']['p95']:.3e}` / `{obs['H_tt_condition']['max']:.3e}`。
- 归一化 R/t coupling 最大奇异值 median：`{obs['coupling_sigma_max']['median']:.6f}`。
- 弱方向与 plot-up 夹角 median：`{obs['weak_direction_angle_to_plot_up_deg']['median']:.2f}` 度。

## Gate 6：R/t 解耦

`U2` 使用相同匹配的加权去中心 3D Procrustes 旋转，不依赖 GT 或 IMU；`U3` 仅为 GT 旋转 oracle。完整数值在 `startup_rt_decoupling_summary.json`。

## Gate 7：UVD 与 3D-3D

3D-3D 使用两端完整 3x3 covariance：`Sigma_i + R Sigma_j R^T`，不是等权欧氏距离。F3/F4 在相同 U2 旋转下隔离残差形式，F5 再隔离低 depth-cov 选点。

## Gate 8 状态

`gate8_executed = {policy['gate8_executed']}`。原因：{policy['reason']}

## 解释边界

- 本轮没有修改 IMU、LM、边缘化、窗口、bias、sampling-aware covariance 或输出滤波。
- GT 只用于误差计算和 A/U3/ART oracle。
- U2/F3/F4 是非 GT 视觉候选，可用于下一轮受控 VIO replay，但阈值必须在独立数据上确定。
- 冷启动与连续运行“数值近似相同”不等于现有生产来源契约已经严格合规；生产接入仍应切换为冷启动缓存后再验收。

## 复现

```bash
cd /home/admin1/macvo-dev
/home/admin1/miniconda3/envs/macvo/bin/python Scripts/diagnose_startup_visual_upward_drift.py
```
"""
    (output / "startup_visual_upward_drift_report_cn.md").write_text(report, encoding="utf-8")


def main() -> int:
    args = parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    s, timestamps, contract = gate0(args, output)
    cold_reader, cold_sidecar, _ = gate1(args, output, s, timestamps)
    _, truth_pose = load_truth(args.dataset)
    if not np.array_equal(np.asarray(cold_reader.manifest.timestamps_ns), timestamps[s:]):
        raise RuntimeError("cold-start cache timestamps do not equal the derived active slice")
    z_mac = cold_measurements(cold_sidecar)
    z_gt = relative_edges(truth_pose, np.arange(s, len(timestamps) - 1), np.arange(s + 1, len(timestamps)))
    visual, truth_local = gate2(output, s, timestamps, truth_pose, z_mac)
    translation = gate3(output, s, timestamps, truth_pose, z_mac, z_gt)
    oracle = gate4(output, z_mac, z_gt, truth_local)
    observability = gate5(output, cold_reader, s, truth_pose, z_mac, z_gt, translation)
    rt, rt_estimates = gate6(output, cold_reader, s, truth_pose, z_mac, z_gt, truth_local)
    factors = gate7(output, cold_reader, s, truth_pose, z_mac, z_gt, truth_local, rt_estimates)
    make_policy_summary(output, oracle, observability, rt, factors)
    build_report(output, contract)
    change_list = f"""# 修改与产物清单

- 新增只读脚本：`Scripts/diagnose_startup_visual_upward_drift.py`。
- 未修改任何生产优化器、IMU、LM、边缘化、视觉缓存或结果。
- 输出目录：`{output}`。
- 复现命令：

```bash
cd /home/admin1/macvo-dev
/home/admin1/miniconda3/envs/macvo/bin/python Scripts/diagnose_startup_visual_upward_drift.py
```
"""
    (output / "startup_visual_diagnostic_change_list.md").write_text(change_list, encoding="utf-8")
    print(json.dumps({"output": str(output), "active_start_frame": s}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
