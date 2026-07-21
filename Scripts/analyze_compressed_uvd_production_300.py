#!/usr/bin/env python3
"""Evaluate the production compressed-UVD replay at the IMU reference point."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.spatial.transform import Rotation


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


NWU_TO_NED = np.diag([1.0, -1.0, -1.0])


def make_transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    transform[:3, 3] = np.asarray(translation, dtype=np.float64).reshape(3)
    return transform


def invert(transform: np.ndarray) -> np.ndarray:
    rotation = transform[:3, :3]
    return make_transform(rotation.T, -rotation.T @ transform[:3, 3])


def stats(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "min": float(np.min(values)),
        "median": float(np.median(values)),
        "mean": float(np.mean(values)),
        "rmse": float(np.sqrt(np.mean(values**2))),
        "p95": float(np.quantile(values, 0.95)),
        "max": float(np.max(values)),
    }


def load_camera_poses(path: Path) -> tuple[np.ndarray, np.ndarray]:
    frame = pd.read_csv(path)
    poses = []
    for row in frame.itertuples(index=False):
        poses.append(
            make_transform(
                Rotation.from_quat([row.qx, row.qy, row.qz, row.qw]).as_matrix(),
                np.array([row.tx, row.ty, row.tz], dtype=np.float64),
            )
        )
    return frame["timestamp_ns"].to_numpy(np.int64), np.stack(poses)


def load_truth_imu(dataset: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ref = pd.read_csv(dataset / "ref_pose.csv")
    metadata = json.loads((dataset / "metadata.json").read_text(encoding="utf-8"))
    lever = NWU_TO_NED @ np.asarray(
        metadata["extrinsics"]["T_body_imu"]["translation_body_nwu_m"],
        dtype=np.float64,
    )
    poses = []
    for row in ref.itertuples(index=False):
        rotation_nwu = Rotation.from_quat([row.qx, row.qy, row.qz, row.qw]).as_matrix()
        body = make_transform(
            NWU_TO_NED @ rotation_nwu @ NWU_TO_NED,
            NWU_TO_NED @ np.array([row.x, row.y, row.z], dtype=np.float64),
        )
        poses.append(body @ make_transform(np.eye(3), lever))
    velocity = ref[["vx", "vy", "vz"]].to_numpy(np.float64) @ NWU_TO_NED.T
    return ref["timestamp"].to_numpy(np.int64), np.stack(poses), velocity


def interpolate_columns(
    source_time: np.ndarray,
    values: np.ndarray,
    query_time: np.ndarray,
) -> np.ndarray:
    return np.column_stack(
        [np.interp(query_time, source_time, values[:, axis]) for axis in range(values.shape[1])]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--start-frame", type=int, default=90)
    parser.add_argument("--end-frame", type=int, default=299)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pose_time, camera_poses = load_camera_poses(args.result_dir / "poses.csv")
    payload = torch.load(args.capture, map_location="cpu", weights_only=False)
    extrinsic_ci_tensor = (
        payload["edges"][0]["problem"].visual_pose.extrinsic_CI
        .detach()
        .cpu()
        .numpy()
        .reshape(7)
        .astype(np.float64)
    )
    extrinsic_ci = make_transform(
        Rotation.from_quat(extrinsic_ci_tensor[3:]).as_matrix(),
        extrinsic_ci_tensor[:3],
    )
    estimated_imu_poses = camera_poses @ extrinsic_ci

    truth_time, truth_poses_all, truth_velocity_all = load_truth_imu(args.dataset)
    truth_indices = {int(value): index for index, value in enumerate(truth_time)}
    selected_frames = np.arange(args.start_frame, args.end_frame + 1, dtype=np.int64)
    selected_time = pose_time[selected_frames]
    truth_rows = np.array([truth_indices[int(value)] for value in selected_time], dtype=np.int64)
    estimate = estimated_imu_poses[selected_frames].copy()
    truth = truth_poses_all[truth_rows].copy()

    estimate[:, :3, 3] -= estimate[0, :3, 3]
    truth[:, :3, 3] -= truth[0, :3, 3]
    position_error = estimate[:, :3, 3] - truth[:, :3, 3]
    rotation_error = np.stack(
        [
            Rotation.from_matrix(truth[index, :3, :3].T @ estimate[index, :3, :3]).as_rotvec()
            for index in range(len(selected_frames))
        ]
    )

    translation_rpe = []
    translation_xy_rpe = []
    rotation_rpe = []
    for index in range(len(selected_frames) - 1):
        relative_estimate = invert(estimate[index]) @ estimate[index + 1]
        relative_truth = invert(truth[index]) @ truth[index + 1]
        error = invert(relative_truth) @ relative_estimate
        translation_rpe.append(np.linalg.norm(error[:3, 3]))
        translation_xy_rpe.append(np.linalg.norm(error[:2, 3]))
        rotation_rpe.append(np.linalg.norm(Rotation.from_matrix(error[:3, :3]).as_rotvec()))

    diagnostics = pd.read_csv(args.result_dir / "frame_pair_diagnostics.csv")
    diagnostics = diagnostics[
        (diagnostics["frame_j"] >= args.start_frame + 1)
        & (diagnostics["frame_j"] <= args.end_frame)
    ].copy()
    endpoint_frames = diagnostics["frame_j"].to_numpy(np.int64)
    endpoint_time = diagnostics["timestamp_j"].to_numpy(np.int64)
    endpoint_truth_rows = np.array(
        [truth_indices[int(value)] for value in endpoint_time], dtype=np.int64
    )
    velocity_estimate = diagnostics[
        ["est_velocity_j_x", "est_velocity_j_y", "est_velocity_j_z"]
    ].to_numpy(np.float64)
    velocity_truth = truth_velocity_all[endpoint_truth_rows]

    decomposition = pd.read_csv(args.dataset / "imu_truth_decomposition.csv")
    decomposition_time = decomposition["timestamp"].to_numpy(np.int64)
    acc_bias_truth_flu = interpolate_columns(
        decomposition_time,
        decomposition[["acc_bias_x", "acc_bias_y", "acc_bias_z"]].to_numpy(np.float64),
        endpoint_time,
    )
    gyro_bias_truth_flu = interpolate_columns(
        decomposition_time,
        decomposition[["gyro_bias_x", "gyro_bias_y", "gyro_bias_z"]].to_numpy(np.float64),
        endpoint_time,
    )
    acc_bias_truth = acc_bias_truth_flu @ NWU_TO_NED.T
    gyro_bias_truth = gyro_bias_truth_flu @ NWU_TO_NED.T
    acc_bias_estimate = diagnostics[
        ["imu_vio_acc_bias_x", "imu_vio_acc_bias_y", "imu_vio_acc_bias_z"]
    ].to_numpy(np.float64)
    gyro_bias_estimate = diagnostics[
        ["imu_vio_gyro_bias_x", "imu_vio_gyro_bias_y", "imu_vio_gyro_bias_z"]
    ].to_numpy(np.float64)

    state_rows = []
    endpoint_by_frame = {int(frame): index for index, frame in enumerate(endpoint_frames)}
    for local_index, frame_index in enumerate(selected_frames):
        row = {
            "frame": int(frame_index),
            "timestamp_ns": int(selected_time[local_index]),
            "position_error_x": float(position_error[local_index, 0]),
            "position_error_y": float(position_error[local_index, 1]),
            "position_error_z": float(position_error[local_index, 2]),
            "rotation_error_x": float(rotation_error[local_index, 0]),
            "rotation_error_y": float(rotation_error[local_index, 1]),
            "rotation_error_z": float(rotation_error[local_index, 2]),
        }
        endpoint_index = endpoint_by_frame.get(int(frame_index))
        if endpoint_index is not None:
            row.update(
                velocity_error_x=float(velocity_estimate[endpoint_index, 0] - velocity_truth[endpoint_index, 0]),
                velocity_error_y=float(velocity_estimate[endpoint_index, 1] - velocity_truth[endpoint_index, 1]),
                velocity_error_z=float(velocity_estimate[endpoint_index, 2] - velocity_truth[endpoint_index, 2]),
                acc_bias_error_x=float(acc_bias_estimate[endpoint_index, 0] - acc_bias_truth[endpoint_index, 0]),
                acc_bias_error_y=float(acc_bias_estimate[endpoint_index, 1] - acc_bias_truth[endpoint_index, 1]),
                acc_bias_error_z=float(acc_bias_estimate[endpoint_index, 2] - acc_bias_truth[endpoint_index, 2]),
                gyro_bias_error_x=float(gyro_bias_estimate[endpoint_index, 0] - gyro_bias_truth[endpoint_index, 0]),
                gyro_bias_error_y=float(gyro_bias_estimate[endpoint_index, 1] - gyro_bias_truth[endpoint_index, 1]),
                gyro_bias_error_z=float(gyro_bias_estimate[endpoint_index, 2] - gyro_bias_truth[endpoint_index, 2]),
            )
        state_rows.append(row)
    pd.DataFrame(state_rows).to_csv(
        args.output_dir / "production_300_state_truth_errors.csv", index=False
    )

    velocity_error = velocity_estimate - velocity_truth
    acc_bias_error = acc_bias_estimate - acc_bias_truth
    gyro_bias_error = gyro_bias_estimate - gyro_bias_truth
    convergence = diagnostics["two_state_solver_converged"].dropna()
    iterations = diagnostics["two_state_solver_iterations"].dropna().to_numpy(np.float64)
    final_step_norm = diagnostics["two_state_final_step_norm"].dropna().to_numpy(np.float64)
    final_gradient_inf_norm = (
        diagnostics["two_state_final_gradient_inf_norm"].dropna().to_numpy(np.float64)
    )
    summary = {
        "contract": {
            "estimate_reference_point": "IMU center: T_WI = T_WC @ T_CI",
            "truth_reference_point": "IMU center: T_WI = T_WB @ T_BI from metadata",
            "world_frame": "internal NED",
            "evaluation_alignment": "translation-only rebase at frame 90; no rotation, scale, or SE(3) fit",
            "frames": [int(args.start_frame), int(args.end_frame)],
            "state_count": int(len(selected_frames)),
            "edge_count": int(len(diagnostics)),
        },
        "trajectory": {
            "xy_ate_rmse_m": float(np.sqrt(np.mean(np.sum(position_error[:, :2] ** 2, axis=1)))),
            "position_3d_rmse_m": float(np.sqrt(np.mean(np.sum(position_error**2, axis=1)))),
            "position_axis_rmse_m": {
                axis: float(np.sqrt(np.mean(position_error[:, index] ** 2)))
                for index, axis in enumerate("xyz")
            },
            "orientation_rmse_rad": float(np.sqrt(np.mean(np.sum(rotation_error**2, axis=1)))),
            "orientation_rmse_deg": float(np.degrees(np.sqrt(np.mean(np.sum(rotation_error**2, axis=1))))),
            "velocity_rmse_mps": float(np.sqrt(np.mean(np.sum(velocity_error**2, axis=1)))),
            "acc_bias_rmse_mps2": float(np.sqrt(np.mean(np.sum(acc_bias_error**2, axis=1)))),
            "gyro_bias_rmse_radps": float(np.sqrt(np.mean(np.sum(gyro_bias_error**2, axis=1)))),
        },
        "rpe": {
            "translation_3d_m": stats(np.asarray(translation_rpe)),
            "translation_xy_m": stats(np.asarray(translation_xy_rpe)),
            "rotation_rad": stats(np.asarray(rotation_rpe)),
            "rotation_deg": stats(np.degrees(np.asarray(rotation_rpe))),
        },
        "runtime": {
            "factor_reconstruction_ms": stats(
                diagnostics["local_ba_graph_build_s"].to_numpy(np.float64) * 1000.0
            ),
            "solver_ms": stats(diagnostics["local_ba_lm_s"].to_numpy(np.float64) * 1000.0),
            "total_edge_ms": stats(
                diagnostics["local_ba_optimize_total_s"].to_numpy(np.float64) * 1000.0
            ),
        },
        "convergence": {
            "reported_edge_count": int(len(convergence)),
            "converged_rate": float(convergence.astype(bool).mean()) if len(convergence) else None,
            "iterations": stats(iterations) if len(iterations) else None,
            "reasons": {
                str(key): int(value)
                for key, value in diagnostics[
                    "two_state_solver_convergence_reason"
                ].value_counts(dropna=False).items()
            },
            "final_step_norm": stats(final_step_norm) if len(final_step_norm) else None,
            "final_gradient_inf_norm": (
                stats(final_gradient_inf_norm) if len(final_gradient_inf_norm) else None
            ),
            "accepted_steps": stats(
                diagnostics["two_state_solver_accepted_steps"].dropna().to_numpy(np.float64)
            ),
            "rejected_steps": stats(
                diagnostics["two_state_solver_rejected_steps"].dropna().to_numpy(np.float64)
            ),
        },
        "costs": {
            "visual_sum": float(diagnostics["visual_loss_raw_sum"].sum()),
            "imu_rotation_sum": float(diagnostics["imu_rot_loss"].sum()),
            "imu_position_sum": float(diagnostics["imu_trans_loss"].sum()),
            "imu_velocity_sum": float(diagnostics["imu_vel_loss"].sum()),
            "total_sum": float(diagnostics["total_loss"].sum()),
            "note": "The current frame-pair CSV does not expose prior and bias costs separately.",
        },
    }
    (args.output_dir / "production_300_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
