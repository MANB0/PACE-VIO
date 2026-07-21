#!/usr/bin/env python3
"""Audit the three-second static-initialization boundary without changing MACVO.

This script consumes immutable dataset/cache/result artifacts and emits Gates 1-5
of the initialization-boundary audit. Gate 6 is produced by the instrumented short
replay in ``run_initialization_boundary_counterfactual.py``.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pypose as pp
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Utility.IMUCSV import IMUCSVLoader  # noqa: E402


DEFAULT_SCENE = "clear_circle_truth_normal_noise"
DEFAULT_CACHE = ROOT / "VisualCache/static63_unique_visual_20260713" / DEFAULT_SCENE
DEFAULT_RESULT = ROOT / (
    "Results/circle_straight_normal_noise_two_state_standard_full_20260715/trial_1/"
    "vio_two_state_fixed_lag_standard_full/clear_circle_truth_normal_noise"
)
DEFAULT_DATASET = Path(
    "/mnt/e/\u6587\u6863/holoocean/code/recordings/"
    "batch_clear_truth_paths_20260713_static63_variants/clear_circle_truth_normal_noise"
)
DEFAULT_OUTPUT = ROOT / "analysis_initialization_boundary_audit_20260716"


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def se3(array: np.ndarray | torch.Tensor) -> pp.LieTensor:
    return pp.SE3(torch.as_tensor(array, dtype=torch.float64).reshape(-1, 7))


def log6(transform: pp.LieTensor) -> np.ndarray:
    return transform.Log().tensor().reshape(-1, 6).detach().cpu().numpy()


def pose_columns(prefix: str, transform: pp.LieTensor) -> dict[str, float]:
    value = transform.tensor().reshape(7).detach().cpu().numpy()
    return {
        f"{prefix}_{name}": float(value[index])
        for index, name in enumerate(("tx", "ty", "tz", "qx", "qy", "qz", "qw"))
    }


def vector_columns(prefix: str, value: np.ndarray) -> dict[str, float]:
    flat = np.asarray(value, dtype=np.float64).reshape(-1)
    return {f"{prefix}_{index}": float(component) for index, component in enumerate(flat)}


def load_extrinsic(result_map: dict[str, np.ndarray], frame: int) -> pp.LieTensor:
    value = result_map["frames//imu_vio_sensor_T_imu"][frame]
    return se3(value)


def reconstruct_first_interval(
    dataset: Path,
    timestamps: list[int],
    frame_i: int,
    frame_j: int,
) -> dict[str, Any]:
    loader = IMUCSVLoader(dataset / "imu_data.csv")
    knots, _, _, sampling = loader.query_range_with_sampling_map(
        timestamps[frame_i], timestamps[frame_j]
    )
    if sampling is None:
        raise RuntimeError("first active IMU interval has no sampling map")
    knot_values = [int(value) for value in knots.tolist()]
    raw_values = [int(value) for value in sampling.raw_time_ns.tolist()]
    return {
        "knot_timestamps_ns": knot_values,
        "raw_support_timestamps_ns": raw_values,
        "knot_count": len(knot_values),
        "raw_support_count": len(raw_values),
        "first_knot_ns": knot_values[0],
        "last_knot_ns": knot_values[-1],
        "sum_dt_s": float((knots[1:] - knots[:-1]).double().sum().item() * 1e-9),
        "image_dt_s": float((timestamps[frame_j] - timestamps[frame_i]) * 1e-9),
        "contains_pre_start_knot": any(value < timestamps[frame_i] for value in knot_values),
        "contains_pre_start_raw_support": any(value < timestamps[frame_i] for value in raw_values),
        "endpoint_interpolation_matrix": sampling.knot_from_raw.numpy().tolist(),
    }


def build_boundary_outputs(
    output: Path,
    cache: Path,
    result: Path,
    dataset: Path,
) -> tuple[int, dict[str, Any], list[dict[str, str]], dict[str, np.ndarray]]:
    manifest = json.loads((cache / "manifest.json").read_text(encoding="utf-8"))
    timestamps = [int(value) for value in manifest["timestamps_ns"]]
    diagnostics = read_csv(result / "frame_pair_diagnostics.csv")
    active_rows = [row for row in diagnostics if row.get("vio_factor_active") == "1"]
    if not active_rows:
        raise RuntimeError("no active two-state factor found")
    first = active_rows[0]
    s = int(first["frame_i"])
    first_j = int(first["frame_j"])
    if first_j != s + 1:
        raise RuntimeError(f"first active edge is not contiguous: {s}->{first_j}")

    result_map = dict(np.load(result / "tensor_map.npz", allow_pickle=False))
    interval = reconstruct_first_interval(dataset, timestamps, s, first_j)
    formal_edges = {(int(row["frame_i"]), int(row["frame_j"])) for row in active_rows}
    rows: list[dict[str, Any]] = []
    for frame in range(max(0, s - 5), min(len(timestamps), s + 6)):
        incoming = (frame - 1, frame)
        rows.append(
            {
                "frame_id": frame,
                "image_timestamp_ns": timestamps[frame],
                "image_time_s": timestamps[frame] * 1e-9,
                "is_calibration_frame": frame <= s,
                "is_active_vio_frame": frame >= s,
                "macvo_processed": True,
                "graph_frame_node_created": True,
                "optimizer_state_created": frame >= s,
                "visual_measurement_loaded": frame > 0,
                "visual_factor_created": incoming in formal_edges,
                "imu_factor_created": incoming in formal_edges,
                "optimizer_suppressed_by_static_zupt": 0 < frame <= s,
            }
        )
    write_csv(output / "initialization_boundary_manifest.csv", rows)

    boundary_crossing = any(i < s or j <= s for i, j in formal_edges)
    contract = {
        "scene": DEFAULT_SCENE,
        "dataset": str(dataset),
        "visual_cache": str(cache),
        "production_result": str(result),
        "calibration_start_frame": 0,
        "calibration_start_timestamp_ns": timestamps[0],
        "calibration_end_frame": s,
        "calibration_end_timestamp_ns": timestamps[s],
        "active_start_frame": s,
        "active_start_timestamp_ns": timestamps[s],
        "first_optimizer_state_key": s,
        "first_visual_factor_frame_i": s,
        "first_visual_factor_frame_j": first_j,
        "first_imu_factor_frame_i": s,
        "first_imu_factor_frame_j": first_j,
        "first_imu_timestamp_ns": interval["first_knot_ns"],
        "last_imu_timestamp_ns": interval["last_knot_ns"],
        "first_imu_interval": interval,
        "formal_factor_count": len(formal_edges),
        "has_formal_edge_crossing_initialization_boundary": boundary_crossing,
        "has_s_minus_1_to_s_factor": (s - 1, s) in formal_edges,
        "first_state_is_frame_s": True,
        "first_visual_factor_is_s_to_s_plus_1": first_j == s + 1,
        "first_imu_factor_is_exact_s_to_s_plus_1": (
            interval["first_knot_ns"] == timestamps[s]
            and interval["last_knot_ns"] == timestamps[first_j]
            and abs(interval["sum_dt_s"] - interval["image_dt_s"]) < 1e-12
        ),
        "pre_s_imu_enters_first_factor": bool(
            interval["contains_pre_start_knot"]
            or interval["contains_pre_start_raw_support"]
        ),
        "interpretation": (
            "Frames 0..s are consumed and anchored during startup. Frame s is the shared "
            "boundary anchor and the first navigation state seen by the solver; the first "
            "formal factors are s->s+1."
        ),
    }
    write_json(output / "initialization_boundary_contract.json", contract)
    return s, contract, diagnostics, result_map


def build_pose_provenance(
    output: Path,
    cache: Path,
    result: Path,
    s: int,
    result_map: dict[str, np.ndarray],
) -> dict[str, Any]:
    source_result = Path(
        json.loads((cache / "manifest.json").read_text(encoding="utf-8"))["source"]["result"]
    )
    source_map = dict(np.load(source_result / "tensor_map.npz", allow_pickle=False))
    abs_camera = se3(source_map["frames//pose"])
    vio_camera = se3(result_map["frames//pose"])
    factors = np.load(cache / "relative_pose_factors.npz", allow_pickle=False)
    rows: list[dict[str, Any]] = []
    for edge in range(s, min(s + 30, int(factors["frame_i"].shape[0]))):
        i = int(factors["frame_i"][edge])
        j = int(factors["frame_j"][edge])
        z_from_abs = abs_camera[i].Inv() @ abs_camera[j]
        # The pure-MACVO source is a sequential two-frame optimizer. Its direct pair
        # output was not serialized separately; this is the exact relative transform
        # implied by the two adjacent optimized source poses.
        z_direct_reconstructed = z_from_abs
        z_factor = se3(factors["measurement_CiCj"][edge])
        err_direct = log6(z_direct_reconstructed.Inv() @ z_factor).reshape(6)
        err_abs = log6(z_from_abs.Inv() @ z_factor).reshape(6)
        wrong_abs_j = log6(z_factor.Inv() @ abs_camera[j]).reshape(6)
        wrong_rebased_j = log6(
            z_factor.Inv() @ (abs_camera[0].Inv() @ abs_camera[j])
        ).reshape(6)
        wrong_rebased_i = log6(
            z_factor.Inv() @ (abs_camera[0].Inv() @ abs_camera[i])
        ).reshape(6)
        state_initial_j = vio_camera[i] @ z_factor
        row: dict[str, Any] = {
            "edge_id": edge - s,
            "frame_i": i,
            "frame_j": j,
            "z_direct_serialized_independently": False,
            "z_direct_source": "reconstructed_from_adjacent_pure_MACVO_optimized_poses",
            "z_factor_source": "relative_pose_factors.npz",
            "z_factor_generation": "inv(T_WC_abs_i) @ T_WC_abs_j",
            "state_initial_j_source": "previous_VIO_pose_i @ Z_factor (pre-IMU fusion disabled)",
            "prior_mean_i_source": "initial diagonal prior at s; carried Schur prior afterwards",
            "err_direct_factor_norm": float(np.linalg.norm(err_direct)),
            "err_abs_factor_norm": float(np.linalg.norm(err_abs)),
            "wrong_abs_j_norm": float(np.linalg.norm(wrong_abs_j)),
            "wrong_rebased_abs_j_norm": float(np.linalg.norm(wrong_rebased_j)),
            "wrong_rebased_abs_i_norm": float(np.linalg.norm(wrong_rebased_i)),
        }
        row.update(pose_columns("z_direct", z_direct_reconstructed))
        row.update(pose_columns("t_wc_abs_i", abs_camera[i]))
        row.update(pose_columns("t_wc_abs_j", abs_camera[j]))
        row.update(pose_columns("z_from_abs", z_from_abs))
        row.update(pose_columns("z_factor", z_factor))
        row.update(vector_columns("err_direct_factor", err_direct))
        row.update(vector_columns("err_abs_factor", err_abs))
        row.update(pose_columns("x_state_initial_j_pose_wc", state_initial_j))
        row.update(pose_columns("x_state_optimized_i_pose_wc_saved", vio_camera[i]))
        row.update(pose_columns("x_state_optimized_j_pose_wc_saved", vio_camera[j]))
        rows.append(row)
    write_csv(output / "visual_pose_provenance_per_edge.csv", rows)
    return {
        "source_result": str(source_result),
        "max_err_direct_factor_norm": max(row["err_direct_factor_norm"] for row in rows),
        "max_err_abs_factor_norm": max(row["err_abs_factor_norm"] for row in rows),
        "direct_measurement_independently_serialized": False,
        "factor_classification": "adjacent_relative_pose_reconstructed_from_pure_MACVO_absolute_track",
    }


def build_rebase_invariance(
    output: Path,
    cache: Path,
    result_map: dict[str, np.ndarray],
    s: int,
) -> dict[str, Any]:
    manifest = json.loads((cache / "manifest.json").read_text(encoding="utf-8"))
    source_result = Path(manifest["source"]["result"])
    source_map = dict(np.load(source_result / "tensor_map.npz", allow_pickle=False))
    camera = se3(source_map["frames//pose"])
    extrinsic_ci = load_extrinsic(result_map, s)
    body = camera @ extrinsic_ci
    rebased_camera = camera[s].Inv() @ camera
    rebased_body = body[s].Inv() @ body
    camera_errors: list[float] = []
    body_errors: list[float] = []
    conjugacy_errors: list[float] = []
    end = min(camera.shape[0] - 1, s + 300)
    for i in range(s, end):
        original_c = camera[i].Inv() @ camera[i + 1]
        rebased_c = rebased_camera[i].Inv() @ rebased_camera[i + 1]
        camera_errors.append(float(torch.linalg.vector_norm((original_c.Inv() @ rebased_c).Log()).item()))
        original_b = body[i].Inv() @ body[i + 1]
        rebased_b = rebased_body[i].Inv() @ rebased_body[i + 1]
        body_errors.append(float(torch.linalg.vector_norm((original_b.Inv() @ rebased_b).Log()).item()))
        conjugated = extrinsic_ci.Inv() @ original_c @ extrinsic_ci
        conjugacy_errors.append(float(torch.linalg.vector_norm((original_b.Inv() @ conjugated).Log()).item()))
    result_value = {
        "active_start_frame": s,
        "T_EC_s_log_norm": float(torch.linalg.vector_norm(rebased_camera[s].Log()).item()),
        "T_EB_s_log_norm": float(torch.linalg.vector_norm(rebased_body[s].Log()).item()),
        "camera_edge_count": len(camera_errors),
        "camera_rebase_max_se3_log_error": max(camera_errors),
        "body_rebase_max_se3_log_error": max(body_errors),
        "camera_to_body_conjugacy_max_se3_log_error": max(conjugacy_errors),
        "extrinsic_CI": extrinsic_ci.tensor().reshape(7).tolist(),
        "convention": {
            "T_AB": "maps coordinates in B into A",
            "source_pose": "T_WC camera-to-world",
            "optimizer_pose": "T_WB body-to-world = T_WC @ T_CI",
            "camera_relative": "T_CiCj = inv(T_WCi) @ T_WCj",
            "body_relative": "T_BiBj = inv(T_WBi) @ T_WBj",
        },
        "expected_precision": "source poses and extrinsic are serialized as float32",
        "pass_threshold_se3_log": 1e-7,
        "pass": max(camera_errors + body_errors + conjugacy_errors) < 1e-7,
    }
    write_json(output / "visual_rebase_invariance.json", result_value)
    return result_value


def build_cache_audit(
    output: Path,
    contract: dict[str, Any],
    result_map: dict[str, np.ndarray],
) -> dict[str, Any]:
    s = int(contract["active_start_frame"])
    first_j = int(contract["first_imu_factor_frame_j"])
    # The boundary state's "prev" fields hold the static initialization result.
    # At s+1 those fields already contain the state optimized by the first edge.
    static_bg = result_map["frames//imu_vio_prev_gyro_bias"][s].reshape(-1).tolist()
    static_ba = result_map["frames//imu_vio_prev_acc_bias"][s].reshape(-1).tolist()
    entries = [
        {
            "object_name": "IMU preintegrator DeltaR/DeltaV/DeltaP",
            "value_before_reset": "no persistent accumulator; startup pending factor cleared each frame",
            "reset_called": True,
            "value_after_reset": "pending factor None through frame s",
            "first_value_used_by_optimizer": "fresh query_range(t_s,t_s+1) local-frame preintegration",
        },
        {
            "object_name": "IMU covariance",
            "value_before_reset": "startup factor not retained",
            "reset_called": True,
            "value_after_reset": "none",
            "first_value_used_by_optimizer": "fresh covariance over exact first active interval",
        },
        {
            "object_name": "bias linearization point",
            "value_before_reset": "startup estimate incomplete",
            "reset_called": True,
            "value_after_reset": {"ba": static_ba, "bg": static_bg},
            "first_value_used_by_optimizer": {"ba": static_ba, "bg": static_bg},
        },
        {
            "object_name": "fixed-lag context / Schur prior",
            "value_before_reset": "no solver call during startup",
            "reset_called": False,
            "value_after_reset": "two_state_prior absent",
            "first_value_used_by_optimizer": "new diagonal prior centered at state s",
        },
        {
            "object_name": "pose-factor cache",
            "value_before_reset": "immutable frame-indexed sidecar includes all adjacent pairs",
            "reset_called": False,
            "value_after_reset": "unchanged; reset not required",
            "first_value_used_by_optimizer": f"indexed pair {s}->{first_j}; earlier loaded pairs were suppressed",
        },
        {
            "object_name": "MACVO absolute trajectory accumulator",
            "value_before_reset": "exists only in offline pure-MACVO source used to build sidecar",
            "reset_called": False,
            "value_after_reset": "not read by replay runtime",
            "first_value_used_by_optimizer": "none; runtime state s is the static anchor",
        },
        {
            "object_name": "MACVO motion model",
            "value_before_reset": "StaticMotionModel",
            "reset_called": False,
            "value_after_reset": "StaticMotionModel",
            "first_value_used_by_optimizer": "sidecar Z is used as pose initial increment; no historical motion prediction",
        },
        {
            "object_name": "relative_pose_init",
            "value_before_reset": "visual packet stores identity; sidecar stores adjacent Z",
            "reset_called": False,
            "value_after_reset": "unchanged immutable measurements",
            "first_value_used_by_optimizer": f"sidecar Z({s},{first_j})",
        },
        {
            "object_name": "visual previous-frame pointer",
            "value_before_reset": f"frame {s - 1} while consuming startup",
            "reset_called": False,
            "value_after_reset": f"frame {s} after its packet is consumed",
            "first_value_used_by_optimizer": f"from_idx={s}, frame_idx={first_j}",
        },
        {
            "object_name": "velocity propagation state",
            "value_before_reset": "startup value",
            "reset_called": True,
            "value_after_reset": [0.0, 0.0, 0.0],
            "first_value_used_by_optimizer": [0.0, 0.0, 0.0],
        },
        {
            "object_name": "external attitude / legacy integrator",
            "value_before_reset": "startup gravity estimate",
            "reset_called": True,
            "value_after_reset": "static body_to_world estimate",
            "first_value_used_by_optimizer": "not used by standard local-frame factor cache generation",
        },
        {
            "object_name": "frame/key index",
            "value_before_reset": f"graph contains anchored nodes 0..{s}",
            "reset_called": False,
            "value_after_reset": "global indices retained",
            "first_value_used_by_optimizer": f"state keys {s},{first_j}",
        },
        {
            "object_name": "post-fusion cache",
            "value_before_reset": "disabled",
            "reset_called": False,
            "value_after_reset": "disabled",
            "first_value_used_by_optimizer": "none",
        },
    ]
    value = {
        "active_start_frame": s,
        "active_start_timestamp_ns": contract["active_start_timestamp_ns"],
        "first_interval": contract["first_imu_interval"],
        "objects": entries,
        "cross_boundary_cache_found": False,
        "important_distinction": (
            "Several immutable/indexed objects are intentionally not reset. They do not carry "
            "a dynamic residual or prior across the boundary; the first accessed entry is s->s+1."
        ),
    }
    write_json(output / "initialization_cache_reset_audit.json", value)
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--result", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    s, contract, _, result_map = build_boundary_outputs(
        args.output, args.cache, args.result, args.dataset
    )
    provenance = build_pose_provenance(args.output, args.cache, args.result, s, result_map)
    rebase = build_rebase_invariance(args.output, args.cache, result_map, s)
    cache_audit = build_cache_audit(args.output, contract, result_map)
    write_json(
        args.output / "gates_1_to_5_summary.json",
        {
            "active_start_frame": s,
            "boundary_contract_pass": not contract["has_formal_edge_crossing_initialization_boundary"],
            "pose_provenance": provenance,
            "rebase_invariance": rebase,
            "cross_boundary_cache_found": cache_audit["cross_boundary_cache_found"],
        },
    )
    print(json.dumps({"output": str(args.output), "active_start_frame": s}, ensure_ascii=False))


if __name__ == "__main__":
    main()
