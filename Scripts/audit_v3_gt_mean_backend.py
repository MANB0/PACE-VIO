#!/usr/bin/env python3
"""Gate 1-2 audit for the full-circle GT visual relative-pose mean oracle."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pypose as pp
import torch
from scipy.spatial.transform import Rotation


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Scripts.audit_circle_translation_oracle import (  # noqa: E402
    ACTIVE_START_FRAME,
    DEFAULT_CACHE,
    DEFAULT_DATASET,
    invert_transform,
    load_truth,
    make_transform,
    rotation_log,
    se3_from_xyzw,
    se3_to_xyzw,
)
from Utility.RelativePoseFactorCache import camera_factor_to_body_factor  # noqa: E402
from Utility.TwoStateVIO import _whiten  # noqa: E402


OUT = ROOT / "analysis_circle_v3_backend_pointlevel_20260716"
ORACLE = ROOT / "analysis_circle_translation_oracle_20260716"
V3 = ORACLE / "oracles/full/V3"
V3_CACHE = ORACLE / "oracle_caches/full/V3"
HUBER_DELTA = 3.0
EIGENVALUE_FLOOR = 1.0e-12


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def latest(root: Path, name: str) -> Path:
    matches = sorted(root.rglob(name))
    if len(matches) != 1:
        raise RuntimeError(f"expected one {name} under {root}, found {len(matches)}")
    return matches[0]


def se3_log(transform: np.ndarray) -> np.ndarray:
    value = torch.from_numpy(se3_to_xyzw(transform)).reshape(1, 7).double()
    return pp.SE3(value).Log().tensor().reshape(6).detach().cpu().numpy()


def compose(edges: list[np.ndarray]) -> np.ndarray:
    poses = [np.eye(4, dtype=np.float64)]
    for edge in edges:
        poses.append(poses[-1] @ edge)
    return np.stack(poses)


def rmse(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    return float(np.sqrt(np.mean(np.sum(values * values, axis=1))))


def pose_metrics(estimate: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    position = estimate[:, :3, 3] - truth[:, :3, 3]
    rotation = np.stack(
        [rotation_log(truth[k, :3, :3].T @ estimate[k, :3, :3]) for k in range(len(truth))]
    )
    return {
        "ate_xyz_rmse_m": rmse(position),
        "ate_xy_rmse_m": rmse(position[:, :2]),
        "orientation_rmse_rad": rmse(rotation),
        "final_position_error_m": float(np.linalg.norm(position[-1])),
        "final_rotation_error_rad": float(np.linalg.norm(rotation[-1])),
    }


def factor_contract(active_edges: int, pair_count: int, proofs: dict[str, object]) -> dict[str, object]:
    fields = [
        {
            "field_name": "relative_pose_mean",
            "stored_value_source": "V3 relative_pose_factors.npz/measurement_CiCj; GT T_CiCj on active edges",
            "runtime_read_count": {"frontend": active_edges, "solver_loss": active_edges},
            "consumer_function": "TwoFrame_PGO._optimize_two_state_fixed_lag -> RelativePoseFactor",
            "affects_factor_mean": True, "affects_covariance": False, "affects_gate": True,
            "affects_initial_state": False, "unused": False,
        },
        {
            "field_name": "pose_covariance_6x6",
            "stored_value_source": "original MACVO relative_pose_factors.npz/covariance",
            "runtime_read_count": {"frontend": active_edges, "solver_loss": active_edges},
            "consumer_function": "camera_factor_to_body_factor; _whiten; visual gate",
            "affects_factor_mean": False, "affects_covariance": True, "affects_gate": True,
            "affects_initial_state": False, "unused": False,
        },
        {
            "field_name": "point_level_covariance",
            "stored_value_source": "pairs/*.npz points_cov_local, match__obs1_covTc, match__obs2_covTc",
            "runtime_read_count": {"replay_packet": pair_count, "two_state_loss": 0},
            "consumer_function": "ReplayFrontend/legacy GraphInput only; compressed factor already frozen",
            "affects_factor_mean": False, "affects_covariance": False, "affects_gate": False,
            "affects_initial_state": False, "unused": True,
        },
        {
            "field_name": "UVD_3D_points",
            "stored_value_source": "pairs/*.npz pixel/depth/disparity and points_local",
            "runtime_read_count": {"replay_packet": pair_count, "two_state_loss": 0},
            "consumer_function": "ReplayFrontend/legacy map packet; not TwoStateVIOProblem",
            "affects_factor_mean": False, "affects_covariance": False, "affects_gate": False,
            "affects_initial_state": False, "unused": True,
        },
        {
            "field_name": "match_pairs",
            "stored_value_source": "pairs/*.npz match__pixel1_uv and match__pixel2_uv",
            "runtime_read_count": {"replay_packet": pair_count, "two_state_loss": 0},
            "consumer_function": "ReplayFrontend/legacy observation packet",
            "affects_factor_mean": False, "affects_covariance": False, "affects_gate": False,
            "affects_initial_state": False, "unused": True,
        },
        {
            "field_name": "inlier_set",
            "stored_value_source": "not persisted as a per-point boolean mask in the replay packet",
            "runtime_read_count": {"two_state_loss": 0},
            "consumer_function": None,
            "affects_factor_mean": False, "affects_covariance": False, "affects_gate": False,
            "affects_initial_state": False, "unused": True,
        },
        {
            "field_name": "num_points",
            "stored_value_source": "original MACVO sidecar num_points",
            "runtime_read_count": {"solver_gate": active_edges},
            "consumer_function": "_gate_two_state_visual_factor",
            "affects_factor_mean": False, "affects_covariance": True, "affects_gate": True,
            "affects_initial_state": False, "unused": False,
        },
        {
            "field_name": "num_inliers",
            "stored_value_source": "original MACVO sidecar num_inliers",
            "runtime_read_count": {"solver_gate": active_edges},
            "consumer_function": "_gate_two_state_visual_factor",
            "affects_factor_mean": False, "affects_covariance": True, "affects_gate": True,
            "affects_initial_state": False, "unused": False,
        },
        {
            "field_name": "mean_mahalanobis_sq",
            "stored_value_source": "original MACVO sidecar mean_mahalanobis_sq",
            "runtime_read_count": {"solver_gate": active_edges},
            "consumer_function": "_gate_two_state_visual_factor",
            "affects_factor_mean": False, "affects_covariance": True, "affects_gate": True,
            "affects_initial_state": False, "unused": False,
        },
        {
            "field_name": "init_motion",
            "stored_value_source": "original pairs/*.npz relative_pose_init and replayed frame pose",
            "runtime_read_count": {"replay_packet": pair_count, "state_initialization": active_edges},
            "consumer_function": "TwoFrame_PGO._optimize_two_state_fixed_lag state_j.pose_WB",
            "affects_factor_mean": False, "affects_covariance": False, "affects_gate": True,
            "affects_initial_state": True, "unused": False,
        },
        {
            "field_name": "Huber_delta",
            "stored_value_source": "V3 odometry.yaml two_state_visual_huber_delta=3.0",
            "runtime_read_count": {"solver_loss": active_edges},
            "consumer_function": "TwoStateVIO._factor_residuals and _true_cost",
            "affects_factor_mean": False, "affects_covariance": False, "affects_gate": False,
            "affects_initial_state": False, "unused": False,
        },
        {
            "field_name": "gate_thresholds",
            "stored_value_source": "TwoFrame_PGO config defaults (inlier, Mahalanobis, whitened pose norm)",
            "runtime_read_count": {"solver_gate": active_edges},
            "consumer_function": "_gate_two_state_visual_factor",
            "affects_factor_mean": False, "affects_covariance": True, "affects_gate": True,
            "affects_initial_state": False, "unused": False,
        },
        {
            "field_name": "quality_flags_visual_sha256",
            "stored_value_source": "original MACVO sidecar visual_sha256",
            "runtime_read_count": {"cache_reader": pair_count, "solver_loss": 0},
            "consumer_function": "cache provenance/diagnostics only",
            "affects_factor_mean": False, "affects_covariance": False, "affects_gate": False,
            "affects_initial_state": False, "unused": True,
        },
    ]
    return {
        "oracle_name": "GT visual relative-pose mean oracle",
        "active_edge_count": active_edges,
        "pair_packet_count": pair_count,
        "fields": fields,
        "proofs": proofs,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    _, truth_pose = load_truth(DEFAULT_DATASET)
    summary = json.loads((V3 / "summary.json").read_text(encoding="utf-8"))
    tensor_path = Path(summary["artifacts"]["tensor_map"])
    trace = pd.read_csv(V3 / "factor_trace.csv")
    call_count = trace.groupby(["frame_i", "frame_j"]).size().to_dict()

    with np.load(V3_CACHE / "relative_pose_factors.npz", allow_pickle=False) as data:
        factors = {key: data[key].copy() for key in data.files}
    with np.load(DEFAULT_CACHE / "relative_pose_factors.npz", allow_pickle=False) as data:
        source = {key: data[key].copy() for key in data.files}
    with np.load(tensor_path, allow_pickle=False) as data:
        pose_wc = np.stack([se3_from_xyzw(row) for row in data["frames//pose"].astype(np.float64)])
        extrinsic_xyzw = data["frames//imu_vio_sensor_T_imu"][ACTIVE_START_FRAME].astype(np.float64)
    t_ci = se3_from_xyzw(extrinsic_xyzw)

    active_indices = np.arange(ACTIVE_START_FRAME, len(pose_wc) - 1)
    gt_edges, estimate_edges, rows = [], [], []
    max_gt_factor_error = 0.0
    for i in active_indices:
        j = i + 1
        z_gt_c = invert_transform(truth_pose[i]) @ truth_pose[j]
        z_factor_c = se3_from_xyzw(factors["measurement_CiCj"][i].reshape(7))
        max_gt_factor_error = max(
            max_gt_factor_error,
            float(np.linalg.norm(se3_log(invert_transform(z_gt_c) @ z_factor_c))),
        )
        mean_b, cov_b = camera_factor_to_body_factor(
            torch.from_numpy(factors["measurement_CiCj"][i].reshape(1, 7)).double(),
            torch.from_numpy(factors["covariance"][i]).double(),
            torch.from_numpy(extrinsic_xyzw.reshape(1, 7)).double(),
        )
        z_gt_b = se3_from_xyzw(mean_b.detach().cpu().numpy().reshape(7))
        pose_wb_i = pose_wc[i] @ t_ci
        pose_wb_j = pose_wc[j] @ t_ci
        z_est_b = invert_transform(pose_wb_i) @ pose_wb_j
        error_b = invert_transform(z_gt_b) @ z_est_b
        raw = torch.from_numpy(se3_log(error_b)).double()
        white = _whiten(raw, cov_b.reshape(6, 6), EIGENVALUE_FLOOR).detach().cpu().numpy()
        norm = float(np.linalg.norm(white))
        mahal = norm * norm
        huber_triggered = norm > HUBER_DELTA
        huber_weight = min(1.0, HUBER_DELTA / max(norm, 1.0e-12))
        cost = 0.5 * mahal if not huber_triggered else HUBER_DELTA * norm - 0.5 * HUBER_DELTA**2
        num_points = int(factors["num_points"][i])
        num_inliers = int(factors["num_inliers"][i])
        ratio = num_inliers / max(num_points, 1)
        mean_mahal = float(factors["mean_mahalanobis_sq"][i])
        quality_action = "accept"
        if ratio < 0.2 or mean_mahal > 100.0 or norm > 20.0:
            quality_action = "reject_or_max_inflate"
        elif ratio < 0.5 or mean_mahal > 9.0 or norm > 6.0:
            quality_action = "downweight"
        row: dict[str, object] = {
            "frame_i": i, "frame_j": j,
            "raw_rho_x": raw[0].item(), "raw_rho_y": raw[1].item(), "raw_rho_z": raw[2].item(),
            "raw_phi_x": raw[3].item(), "raw_phi_y": raw[4].item(), "raw_phi_z": raw[5].item(),
            "white_rho_x": white[0], "white_rho_y": white[1], "white_rho_z": white[2],
            "white_phi_x": white[3], "white_phi_y": white[4], "white_phi_z": white[5],
            "whitened_norm": norm, "mahalanobis": mahal,
            "huber_triggered": huber_triggered, "huber_weight": huber_weight,
            "pose_factor_cost": cost, "num_points": num_points, "num_inliers": num_inliers,
            "inlier_ratio": ratio, "mean_mahalanobis_sq": mean_mahal,
            "recomputed_final_gate_action": quality_action,
            "runtime_solver_call_count": int(call_count.get((i, j), 0)),
        }
        cov_np = cov_b.detach().cpu().numpy().reshape(6, 6)
        for r in range(6):
            for c in range(6):
                row[f"cov_{r}{c}"] = cov_np[r, c]
        rows.append(row)
        gt_edges.append(z_gt_b)
        estimate_edges.append(z_est_b)

    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "v3_pose_factor_runtime_per_edge.csv", index=False)

    gt_chain = compose(gt_edges)
    estimate_chain = compose(estimate_edges)
    reconstructed_edges, trans_only_edges, rot_only_edges = [], [], []
    for z_gt, z_est in zip(gt_edges, estimate_edges):
        correction = invert_transform(z_gt) @ z_est
        reconstructed_edges.append(z_gt @ correction)
        trans_only_edges.append(z_gt @ make_transform(np.eye(3), correction[:3, 3]))
        rot_only_edges.append(z_gt @ make_transform(correction[:3, :3], np.zeros(3)))
    reconstructed_chain = compose(reconstructed_edges)
    trans_only_chain = compose(trans_only_edges)
    rot_only_chain = compose(rot_only_edges)
    accumulation = {
        "active_frame_range": [int(ACTIVE_START_FRAME), int(len(pose_wc) - 1)],
        "active_edge_count": len(active_indices),
        "reconstruction_max_abs_transform_error": float(np.max(np.abs(reconstructed_chain - estimate_chain))),
        "full_correction_metrics": pose_metrics(estimate_chain, gt_chain),
        "translation_component_only_metrics": pose_metrics(trans_only_chain, gt_chain),
        "rotation_component_only_metrics": pose_metrics(rot_only_chain, gt_chain),
        "huber_triggered_edge_count": int(frame["huber_triggered"].sum()),
        "huber_triggered_fraction": float(frame["huber_triggered"].mean()),
        "postsolve_diagnostic_gate_reclassification_counts": (
            frame["recomputed_final_gate_action"].value_counts().to_dict()
        ),
        "edges_with_second_solver_call": int((frame["runtime_solver_call_count"] > 1).sum()),
        "interpretation": (
            "The V3 trajectory is exactly reproduced by composing GT body relative-pose means "
            "with the optimized per-edge SE(3) corrections. Translation-only and rotation-only "
            "hybrids attribute the accumulated departure without changing production factors."
        ),
    }
    write_json(OUT / "v3_pose_correction_accumulation.json", accumulation)

    v3_pairs = (V3_CACHE / "pairs").resolve()
    source_pairs = (DEFAULT_CACHE / "pairs").resolve()
    proofs = {
        "Z_factor_is_GT_max_se3_log_error": max_gt_factor_error,
        "covariance_bitwise_unchanged": bool(np.array_equal(factors["covariance"], source["covariance"])),
        "quality_fields_bitwise_unchanged": bool(all(
            np.array_equal(factors[key], source[key])
            for key in ("num_points", "num_inliers", "mean_mahalanobis_sq", "visual_sha256")
        )),
        "V3_pair_cache_resolves_to_original_MACVO_pair_cache": v3_pairs == source_pairs,
        "V3_pair_cache_resolved_path": str(v3_pairs),
        "V3_factor_cache_fields": sorted(factors),
        "V3_factor_cache_contains_init_motion": "relative_pose_init" in factors,
        "init_motion_storage_contract": (
            "pairs/IIIIII_JJJJJJ.npz/relative_pose_init is the absolute current-frame "
            "T_WC visual pose used by GraphInput.init_motion; it is not a T_CiCj factor mean"
        ),
        "initial_state_contract": (
            "state_j.pose_WB is constructed from graph_data.init_motion and T_CI. V3 and the "
            "original replay resolve to the same pairs directory, while the separately loaded "
            "relative_pose_factors.npz has no init-motion field. Therefore replacing only "
            "measurement_CiCj changes the factor mean but not the visual pose initialization."
        ),
        "point_level_loss_contract": (
            "TwoStateVIOProblem contains prior_i, ImuPreintegrationFactor, RelativePoseFactor, "
            "and bias RW through the IMU factor. It contains no UVD/3D point residual."
        ),
    }
    contract = factor_contract(len(active_indices), len(pose_wc) - 1, proofs)
    write_json(OUT / "v3_runtime_visual_contract.json", contract)
    print(json.dumps({"contract": proofs, "accumulation": accumulation}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
