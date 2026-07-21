#!/usr/bin/env python3
"""Convert historical normal-noise trajectories to the IMU origin and compare them."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation


ROOT = Path("/home/admin1/macvo-dev")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Scripts.plot_circle_direct_uvd_u1_vs_pose_factor import (  # noqa: E402
    BATCH_ROOT,
    HTML_TEMPLATE,
    metrics,
    position_errors,
    read_forward_axes,
    read_xyz,
    xyz,
)
from Scripts.plot_output_eskf3d_offline_ablation import (  # noqa: E402
    ABLATIONS,
    METHODS,
    add_two_dimensional_filter_logic,
    replace_filter_controls,
)
from Utility.TrajectoryReference import (  # noqa: E402
    rebase_pose_positions,
    translate_pose_reference_point,
)


OUTPUT = ROOT / "analysis_imu_center_all_methods_20260719"
PURE_ROOT = ROOT / (
    "Results/visual_factor_cache_static63_unique_source_20260713/"
    "trial_1/pure_macvo"
)

SCENES: dict[str, dict[str, object]] = {
    "circle": {
        "label": "Circle / Normal noise / IMU-center evaluation",
        "dataset": "clear_circle_truth_normal_noise",
        "frame_count": 1890,
        "raw": {
            "t_factor": ROOT / (
                "Results/circle_straight_normal_noise_two_state_standard_full_20260715/"
                "trial_1/vio_two_state_fixed_lag_standard_full/"
                "clear_circle_truth_normal_noise/poses.csv"
            ),
            "u1": ROOT / (
                "Results/circle_normal_noise_direct_uvd_u1_full_20260716/"
                "trial_1/vio_two_state_direct_uvd_u1_standard_full/"
                "clear_circle_truth_normal_noise/poses.csv"
            ),
            "sa_v1": ROOT / (
                "Results/normal_noise_sa_v1_full_three_scenes_20260717/"
                "trial_1/vio_two_state_direct_uvd_sampling_aware_v1_full/"
                "clear_circle_truth_normal_noise/poses.csv"
            ),
            "sa_v2": ROOT / (
                "Results/normal_noise_sa_v2_full_three_scenes_20260717/"
                "trial_1/vio_two_state_direct_uvd_sampling_aware_v2_full/"
                "clear_circle_truth_normal_noise/poses.csv"
            ),
        },
    },
    "rectangle": {
        "label": "Stop-turn rectangle / Normal noise / IMU-center evaluation",
        "dataset": "clear_stop_turn_rectangle_truth_normal_noise",
        "frame_count": 1890,
        "raw": {
            "t_factor": ROOT / (
                "Results/rectangle_normal_noise_two_state_standard_full_20260715/"
                "trial_1/vio_two_state_fixed_lag_standard_full/"
                "clear_stop_turn_rectangle_truth_normal_noise/poses.csv"
            ),
            "u1": ROOT / (
                "Results/normal_noise_direct_uvd_u1_missing_full_scenes_20260717/"
                "trial_1/vio_two_state_direct_uvd_u1_standard_full/"
                "clear_stop_turn_rectangle_truth_normal_noise/poses.csv"
            ),
            "sa_v1": ROOT / (
                "Results/normal_noise_sa_v1_full_three_scenes_20260717/"
                "trial_1/vio_two_state_direct_uvd_sampling_aware_v1_full/"
                "clear_stop_turn_rectangle_truth_normal_noise/poses.csv"
            ),
            "sa_v2": ROOT / (
                "Results/normal_noise_sa_v2_full_three_scenes_20260717/"
                "trial_1/vio_two_state_direct_uvd_sampling_aware_v2_full/"
                "clear_stop_turn_rectangle_truth_normal_noise/poses.csv"
            ),
        },
    },
    "straight": {
        "label": "Straight / Normal noise / IMU-center evaluation",
        "dataset": "clear_straight_truth_normal_noise",
        "frame_count": 630,
        "raw": {
            "t_factor": ROOT / (
                "Results/circle_straight_normal_noise_two_state_standard_full_20260715/"
                "trial_1/vio_two_state_fixed_lag_standard_full/"
                "clear_straight_truth_normal_noise/poses.csv"
            ),
            "u1": ROOT / (
                "Results/normal_noise_direct_uvd_u1_missing_full_scenes_20260717/"
                "trial_1/vio_two_state_direct_uvd_u1_standard_full/"
                "clear_straight_truth_normal_noise/poses.csv"
            ),
            "sa_v1": ROOT / (
                "Results/normal_noise_sa_v1_full_three_scenes_20260717/"
                "trial_1/vio_two_state_direct_uvd_sampling_aware_v1_full/"
                "clear_straight_truth_normal_noise/poses.csv"
            ),
            "sa_v2": ROOT / (
                "Results/normal_noise_sa_v2_full_three_scenes_20260717/"
                "trial_1/vio_two_state_direct_uvd_sampling_aware_v2_full/"
                "clear_straight_truth_normal_noise/poses.csv"
            ),
        },
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_poses(path: Path) -> tuple[np.ndarray, np.ndarray]:
    frame = pd.read_csv(path)
    timestamp_column = "timestamp_ns" if "timestamp_ns" in frame else "timestamp"
    position_columns = ["tx", "ty", "tz"] if "tx" in frame else ["x", "y", "z"]
    values = np.column_stack(
        [
            frame[position_columns].to_numpy(np.float64),
            frame[["qx", "qy", "qz", "qw"]].to_numpy(np.float64),
        ]
    )
    return frame[timestamp_column].to_numpy(np.int64), values


def write_poses(path: Path, timestamps: np.ndarray, poses: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(
        {
            "timestamp_ns": timestamps,
            "tx": poses[:, 0],
            "ty": poses[:, 1],
            "tz": poses[:, 2],
            "qx": poses[:, 3],
            "qy": poses[:, 4],
            "qz": poses[:, 5],
            "qw": poses[:, 6],
        }
    )
    frame.to_csv(path, index=False, float_format="%.17g")


def metadata_levers(metadata: dict) -> dict[str, np.ndarray]:
    extrinsics = metadata["extrinsics"]
    body_to_imu = np.asarray(
        extrinsics["T_body_imu"]["translation_body_nwu_m"], dtype=np.float64
    )
    body_to_camera = np.asarray(
        extrinsics["T_body_camera"]["translation_body_nwu_m"], dtype=np.float64
    )
    if body_to_imu.shape != (3,) or body_to_camera.shape != (3,):
        raise ValueError("metadata extrinsic translations must be 3D")
    return {
        "body_to_imu_nwu": body_to_imu,
        "body_to_camera_nwu": body_to_camera,
        "camera_to_imu_nwu": body_to_imu - body_to_camera,
    }


def source_paths(scene: str, spec: dict[str, object]) -> dict[str, Path]:
    raw = spec["raw"]
    assert isinstance(raw, dict)
    result: dict[str, Path] = {}
    for ablation in ABLATIONS:
        for method in METHODS:
            key = f"{ablation}_{method}"
            if ablation == "A_raw":
                result[key] = Path(raw[method])
            elif ablation == "B_ekf2d":
                if scene == "circle":
                    result[key] = (
                        ROOT
                        / "Results/circle_output_ekf_four_methods_20260718/strong"
                        / method
                        / "poses.csv"
                    )
                else:
                    result[key] = (
                        ROOT
                        / "Results/rectangle_straight_output_ekf2d_ablation_20260718"
                        / scene
                        / method
                        / "poses.csv"
                    )
            else:
                mode = {
                    "C_eskf3d_no_gate": "no_gate",
                    "D_eskf3d_gate": "gate",
                    "E_eskf3d_gate_adaptive": "gate_adaptive",
                }[ablation]
                if scene == "circle":
                    result[key] = (
                        ROOT
                        / "Results/circle_output_eskf3d_ablation_20260718"
                        / mode
                        / method
                        / "poses.csv"
                    )
                else:
                    result[key] = (
                        ROOT
                        / "Results/rectangle_straight_output_eskf3d_ablation_20260718"
                        / scene
                        / mode
                        / method
                        / "poses.csv"
                    )
    return result


def relative_pose_metrics(gt: np.ndarray, estimate: np.ndarray) -> dict[str, float]:
    gt_rotation = Rotation.from_quat(gt[:, 3:7])
    estimate_rotation = Rotation.from_quat(estimate[:, 3:7])
    gt_relative_rotation = gt_rotation[:-1].inv() * gt_rotation[1:]
    estimate_relative_rotation = estimate_rotation[:-1].inv() * estimate_rotation[1:]
    rotation_error = (gt_relative_rotation.inv() * estimate_relative_rotation).magnitude()
    gt_relative_translation = gt_rotation[:-1].inv().apply(
        gt[1:, :3] - gt[:-1, :3]
    )
    estimate_relative_translation = estimate_rotation[:-1].inv().apply(
        estimate[1:, :3] - estimate[:-1, :3]
    )
    translation_error = np.linalg.norm(
        estimate_relative_translation - gt_relative_translation, axis=1
    )
    return {
        "translation_rpe_rmse_m": float(np.sqrt(np.mean(translation_error**2))),
        "rotation_rpe_rmse_rad": float(np.sqrt(np.mean(rotation_error**2))),
    }


def accuracy_metrics(
    gt: np.ndarray, estimate: np.ndarray, *, active_from: int
) -> dict[str, float | int]:
    if gt.shape != estimate.shape:
        raise ValueError(f"trajectory shape mismatch: {gt.shape} vs {estimate.shape}")
    error = estimate[:, :3] - gt[:, :3]
    xy = np.linalg.norm(error[:, :2], axis=1)
    xyz_error = np.linalg.norm(error, axis=1)
    orientation_error = (
        Rotation.from_quat(gt[:, 3:7]).inv()
        * Rotation.from_quat(estimate[:, 3:7])
    ).magnitude()
    active = slice(min(active_from, len(gt) - 1), None)
    return {
        "frame_count": int(len(gt)),
        "xy_ate_rmse_m": float(np.sqrt(np.mean(xy**2))),
        "xyz_ate_rmse_m": float(np.sqrt(np.mean(xyz_error**2))),
        "active_xy_ate_rmse_m": float(np.sqrt(np.mean(xy[active] ** 2))),
        "active_xyz_ate_rmse_m": float(np.sqrt(np.mean(xyz_error[active] ** 2))),
        "x_rmse_m": float(np.sqrt(np.mean(error[:, 0] ** 2))),
        "y_rmse_m": float(np.sqrt(np.mean(error[:, 1] ** 2))),
        "z_rmse_m": float(np.sqrt(np.mean(error[:, 2] ** 2))),
        "orientation_rmse_rad": float(np.sqrt(np.mean(orientation_error**2))),
        "xy_max_m": float(np.max(xy)),
        "xyz_max_m": float(np.max(xyz_error)),
        **relative_pose_metrics(gt, estimate),
    }


def turn_groups(poses: np.ndarray) -> list[tuple[int, int]]:
    rotation = Rotation.from_quat(poses[:, 3:7])
    angle = (rotation[:-1].inv() * rotation[1:]).magnitude()
    moving = np.flatnonzero(angle > 0.005)
    if moving.size == 0:
        return []
    groups: list[tuple[int, int]] = []
    start = previous = int(moving[0])
    for value in moving[1:]:
        current = int(value)
        if current > previous + 1:
            if previous - start >= 10:
                groups.append((start, previous + 1))
            start = current
        previous = current
    if previous - start >= 10:
        groups.append((start, previous + 1))
    return groups


def audit_gt_reference(raw_gt: np.ndarray, levers: dict[str, np.ndarray]) -> dict:
    groups = turn_groups(raw_gt)
    candidates = {
        "raw": raw_gt,
        "body_to_imu": translate_pose_reference_point(
            raw_gt, levers["body_to_imu_nwu"]
        ),
        "camera_to_imu": translate_pose_reference_point(
            raw_gt, levers["camera_to_imu_nwu"]
        ),
    }
    rows = []
    for start, end in groups:
        row: dict[str, float | int] = {"start_frame": start, "end_frame": end}
        for name, poses in candidates.items():
            displacement = poses[start : end + 1, :3] - poses[start, :3]
            row[f"{name}_turn_chord_m"] = float(
                np.linalg.norm(poses[end, :3] - poses[start, :3])
            )
            row[f"{name}_max_displacement_m"] = float(
                np.max(np.linalg.norm(displacement, axis=1))
            )
        rows.append(row)
    raw_chords = [float(row["raw_turn_chord_m"]) for row in rows]
    return {
        "metadata_explicit_position_sensor": False,
        "metadata_position_fields": (
            "world frame and relative-to-start origin only; no physical sensor point named"
        ),
        "empirical_classification": (
            "body/root rotation origin strongly supported"
            if raw_chords and float(np.median(raw_chords)) < 1.0e-3
            else "inconclusive"
        ),
        "important_physical_note": (
            "An IMU offset from the body rotation origin must trace a small arc during yaw; "
            "point rotation is evidence for the body/root origin, not the IMU origin."
        ),
        "turn_groups": rows,
    }


def converted_rows(path: Path) -> list[tuple[int, float, float, float]]:
    return read_xyz(path)


def build() -> tuple[Path, Path, Path]:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    payloads: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []
    reference_delta_rows: list[dict[str, object]] = []
    manifest_rows: list[dict[str, object]] = []
    turn_arc_rows: list[dict[str, object]] = []
    runtime_extrinsic_rows: list[dict[str, object]] = []
    contracts: dict[str, object] = {
        "schema_version": 1,
        "estimate_input_reference": "CameraLeftSocket / T_WC",
        "estimate_output_reference": "IMU origin / T_WI",
        "estimate_conversion": "p_WI = p_WC + R_WC * (t_BI - t_BC)",
        "gt_input_reference": "body/root rotation origin (empirically inferred)",
        "gt_output_reference": "IMU origin / T_WI",
        "gt_conversion": "p_WI = p_WB + R_WB * t_BI",
        "rebase": "translation-only subtraction of each converted trajectory's first IMU position",
        "alignment": "no SE(3) fitting, no yaw fitting, no scale fitting",
        "warning": (
            "metadata does not explicitly identify the physical point used for ref_pose x/y/z; "
            "body/root is inferred from near-zero position motion during 90-degree turns"
        ),
        "scenes": {},
    }

    for scene, spec in SCENES.items():
        dataset = str(spec["dataset"])
        expected_frames = int(spec["frame_count"])
        dataset_dir = BATCH_ROOT / dataset
        metadata_path = dataset_dir / "metadata.json"
        gt_path = dataset_dir / "ref_pose.csv"
        pure_path = PURE_ROOT / dataset / "poses.csv"
        paths = source_paths(scene, spec)
        for path in (metadata_path, gt_path, pure_path, *paths.values()):
            if not path.is_file():
                raise FileNotFoundError(path)

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        levers = metadata_levers(metadata)
        gt_time, raw_gt = load_poses(gt_path)
        if len(gt_time) != expected_frames:
            raise AssertionError(f"{scene}: expected {expected_frames} GT rows")
        gt_imu = translate_pose_reference_point(raw_gt, levers["body_to_imu_nwu"])
        gt_rebased = rebase_pose_positions(gt_imu)
        scene_unrebased = OUTPUT / "converted" / scene / "unrebased"
        scene_rebased = OUTPUT / "converted" / scene / "rebased"
        gt_unrebased_path = scene_unrebased / "gt_imu.csv"
        gt_rebased_path = scene_rebased / "gt_imu.csv"
        write_poses(gt_unrebased_path, gt_time, gt_imu)
        write_poses(gt_rebased_path, gt_time, gt_rebased)

        gt_audit = audit_gt_reference(raw_gt, levers)
        contracts["scenes"][scene] = {
            "dataset": dataset,
            "metadata_sha256": sha256(metadata_path),
            "ref_pose_sha256": sha256(gt_path),
            "body_to_imu_nwu_m": levers["body_to_imu_nwu"].tolist(),
            "body_to_camera_nwu_m": levers["body_to_camera_nwu"].tolist(),
            "camera_to_imu_nwu_m": levers["camera_to_imu_nwu"].tolist(),
            "gt_reference_audit": gt_audit,
        }

        all_paths = {"pure_macvo": pure_path, **paths}
        converted_paths: dict[str, Path] = {}
        trajectories: dict[str, np.ndarray] = {"GT": gt_rebased}
        for key, source in all_paths.items():
            timestamp, camera_pose = load_poses(source)
            if not np.array_equal(timestamp, gt_time):
                raise AssertionError(f"{scene}/{key}: timestamp mismatch")
            imu_pose = translate_pose_reference_point(
                camera_pose, levers["camera_to_imu_nwu"]
            )
            imu_rebased = rebase_pose_positions(imu_pose)
            unrebased_path = scene_unrebased / f"{key}_imu.csv"
            rebased_path = scene_rebased / f"{key}_imu.csv"
            write_poses(unrebased_path, timestamp, imu_pose)
            write_poses(rebased_path, timestamp, imu_rebased)
            converted_paths[key] = rebased_path
            trajectories[key] = imu_rebased
            if key == "pure_macvo" or key.startswith("A_raw_"):
                tensor_path = source.with_name("tensor_map.npz")
                if not tensor_path.is_file():
                    raise FileNotFoundError(tensor_path)
                with np.load(tensor_path, allow_pickle=False) as tensor_map:
                    runtime_extrinsic = np.asarray(
                        tensor_map["frames//imu_vio_sensor_T_imu"], dtype=np.float64
                    ).reshape(-1, 7)
                expected_internal = np.concatenate(
                    [
                        levers["camera_to_imu_nwu"]
                        * np.asarray([1.0, -1.0, -1.0]),
                        np.asarray([0.0, 0.0, 0.0, 1.0]),
                    ]
                )
                runtime_extrinsic_rows.append(
                    {
                        "scene": scene,
                        "key": key,
                        "tensor_map": str(tensor_path),
                        "row_count": int(len(runtime_extrinsic)),
                        "max_frame_variation": float(
                            np.max(np.abs(runtime_extrinsic - runtime_extrinsic[:1]))
                        ),
                        "max_metadata_error": float(
                            np.max(np.abs(runtime_extrinsic - expected_internal))
                        ),
                        "matches_metadata_atol_1e-7": bool(
                            np.allclose(
                                runtime_extrinsic,
                                expected_internal.reshape(1, 7),
                                atol=1.0e-7,
                                rtol=0.0,
                            )
                        ),
                    }
                )
            legacy_metrics = accuracy_metrics(raw_gt, camera_pose, active_from=90)
            unified_metrics = accuracy_metrics(gt_rebased, imu_rebased, active_from=90)
            reference_delta_rows.append(
                {
                    "scene": scene,
                    "key": key,
                    "legacy_xy_ate_rmse_m": legacy_metrics["xy_ate_rmse_m"],
                    "imu_center_xy_ate_rmse_m": unified_metrics["xy_ate_rmse_m"],
                    "xy_ate_change_m": float(unified_metrics["xy_ate_rmse_m"])
                    - float(legacy_metrics["xy_ate_rmse_m"]),
                    "legacy_translation_rpe_rmse_m": legacy_metrics[
                        "translation_rpe_rmse_m"
                    ],
                    "imu_center_translation_rpe_rmse_m": unified_metrics[
                        "translation_rpe_rmse_m"
                    ],
                }
            )
            if scene == "rectangle":
                for group_index, (start, end) in enumerate(turn_groups(raw_gt)):
                    turn_arc_rows.append(
                        {
                            "key": key,
                            "turn_index": group_index,
                            "start_frame": start,
                            "end_frame": end,
                            "gt_body_chord_m": float(
                                np.linalg.norm(raw_gt[end, :3] - raw_gt[start, :3])
                            ),
                            "gt_imu_chord_m": float(
                                np.linalg.norm(gt_imu[end, :3] - gt_imu[start, :3])
                            ),
                            "estimate_camera_chord_m": float(
                                np.linalg.norm(
                                    camera_pose[end, :3] - camera_pose[start, :3]
                                )
                            ),
                            "estimate_imu_chord_m": float(
                                np.linalg.norm(imu_pose[end, :3] - imu_pose[start, :3])
                            ),
                        }
                    )
            manifest_rows.append(
                {
                    "scene": scene,
                    "key": key,
                    "source_path": str(source),
                    "source_sha256": sha256(source),
                    "unrebased_imu_path": str(unrebased_path),
                    "rebased_imu_path": str(rebased_path),
                    "frame_count": int(len(timestamp)),
                    "timestamp_start_ns": int(timestamp[0]),
                    "timestamp_end_ns": int(timestamp[-1]),
                }
            )

        pure_metrics = accuracy_metrics(gt_rebased, trajectories["pure_macvo"], active_from=90)
        metric_rows.append(
            {
                "scene": scene,
                "optimizer": "pure_macvo",
                "output_method": "A_raw",
                **pure_metrics,
                "source_path": str(pure_path),
                "converted_path": str(converted_paths["pure_macvo"]),
            }
        )

        gt_rows = converted_rows(gt_rebased_path)
        pure_rows = converted_rows(converted_paths["pure_macvo"])
        fusion: list[dict[str, object]] = []
        for ablation, ablation_spec in ABLATIONS.items():
            for method, method_spec in METHODS.items():
                key = f"{ablation}_{method}"
                rows = converted_rows(converted_paths[key])
                method_metrics = accuracy_metrics(
                    gt_rebased, trajectories[key], active_from=90
                )
                metric_rows.append(
                    {
                        "scene": scene,
                        "optimizer": method,
                        "output_method": ablation,
                        **method_metrics,
                        "source_path": str(paths[key]),
                        "converted_path": str(converted_paths[key]),
                    }
                )
                fusion.append(
                    {
                        "key": f"{scene}_{key}",
                        "source": key,
                        "config": ablation,
                        "optimizer": method,
                        "label": f"{method_spec['label']} / {ablation_spec['label']}",
                        "color": method_spec["color"],
                        "dasharray": ablation_spec["dasharray"],
                        "scene": dataset,
                        "xyz": xyz(rows),
                        "forward": read_forward_axes(converted_paths[key]),
                        "error_m": position_errors(gt_rows, rows, xy_only=False),
                        "metrics": metrics(gt_rows, rows),
                        "path": str(converted_paths[key]),
                    }
                )

        payloads.append(
            {
                "scene": str(spec["label"]),
                "gt": xyz(gt_rows),
                "gt_forward": read_forward_axes(gt_rebased_path),
                "macvo": xyz(pure_rows),
                "macvo_forward": read_forward_axes(converted_paths["pure_macvo"]),
                "time_s": [(value - gt_time[0]) * 1.0e-9 for value in gt_time],
                "error_m": position_errors(gt_rows, pure_rows, xy_only=False),
                "metrics": metrics(gt_rows, pure_rows),
                "fusion": fusion,
                "imu_only": [],
                "gt_path": str(gt_rebased_path),
                "macvo_path": str(converted_paths["pure_macvo"]),
            }
        )

    metrics_path = OUTPUT / "imu_center_accuracy_metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(metric_rows[0]))
        writer.writeheader()
        writer.writerows(metric_rows)
    manifest_path = OUTPUT / "imu_center_trajectory_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)
    reference_delta_path = OUTPUT / "reference_point_metric_delta.csv"
    with reference_delta_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(reference_delta_rows[0]))
        writer.writeheader()
        writer.writerows(reference_delta_rows)
    turn_arc_path = OUTPUT / "rectangle_turn_reference_point_audit.csv"
    with turn_arc_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(turn_arc_rows[0]))
        writer.writeheader()
        writer.writerows(turn_arc_rows)
    runtime_extrinsic_path = OUTPUT / "runtime_extrinsic_validation.csv"
    with runtime_extrinsic_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(runtime_extrinsic_rows[0]))
        writer.writeheader()
        writer.writerows(runtime_extrinsic_rows)
    contract_path = OUTPUT / "imu_center_reference_contract.json"
    contract_path.write_text(
        json.dumps(contracts, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    report_lines = [
        "# 历史轨迹 IMU 中心统一转换与精度复评估",
        "",
        "## 首页结论",
        "",
        "1. 所有估计轨迹原始 `poses.csv` 均按相机中心解释，并使用 metadata 杆臂转换为 IMU 中心。",
        "2. `ref_pose.csv` 的位置不是 IMU 中心：矩形原地转向时其位移为微米量级，而 metadata 中 IMU 相对 body 的水平杆臂约 0.12 m。",
        "3. GT 位置按 body/root 旋转中心解释后，通过 `p_WI = p_WB + R_WB t_BI` 转为 IMU 中心。",
        "4. 精度计算使用转换后轨迹，并分别减去各自首帧 IMU 位置；未做 SE(3)、yaw 或尺度拟合。",
        "5. 共转换 3 个场景、每场景 21 条估计轨迹，共 63 条。",
        "",
        "## 原始优化器结果（未加输出滤波）的 XY 指标",
        "",
        "| 场景 | 方法 | XY ATE RMSE (m) | 平移 RPE RMSE (m) | 旋转 RPE RMSE (rad) |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in metric_rows:
        if row["output_method"] != "A_raw":
            continue
        report_lines.append(
            "| {scene} | {optimizer} | {xy:.6f} | {tr:.6f} | {rr:.6f} |".format(
                scene=row["scene"],
                optimizer=row["optimizer"],
                xy=float(row["xy_ate_rmse_m"]),
                tr=float(row["translation_rpe_rmse_m"]),
                rr=float(row["rotation_rpe_rmse_rad"]),
            )
        )
    rectangle_audit = contracts["scenes"]["rectangle"]["gt_reference_audit"]
    report_lines.extend(
        [
            "",
            "## GT 参考点证据",
            "",
            "矩形轨迹检测到 4 段持续转向。原始 GT 在每段的起止位移约为 `6-9e-6 m`；",
            "按 body 到 IMU 杆臂转换后约为 `0.165 m`，符合偏置 IMU 绕 body 原点转过约 90° 的量级。",
            "因此，原始 GT 的点式转弯恰好说明它是 body/root 原点，而不是 IMU 原点。",
            "",
            f"经验分类：`{rectangle_audit['empirical_classification']}`。",
            "",
            "metadata 只写明 ref_pose 位置处于 world NWU 且相对起点，没有明确命名位置传感器。",
            "所以该结论在当前数值证据下很强，但仍建议让 HoloOcean 数据生成侧确认具体 position source。",
            "",
            "## 产物",
            "",
            "- `interactive_imu_center_all_methods.html`：三场景交互对比。",
            "- `imu_center_accuracy_metrics.csv`：统一参考点后的完整指标。",
            "- `reference_point_metric_delta.csv`：旧参考点混用与新 IMU 中心评估的差值。",
            "- `rectangle_turn_reference_point_audit.csv`：转角杆臂闭合。",
            "- `imu_center_trajectory_manifest.csv`：63 条轨迹源文件、哈希与转换路径。",
            "- `imu_center_reference_contract.json`：逐场景 metadata 外参与 GT 证据。",
            "",
        ]
    )
    (OUTPUT / "imu_center_reference_audit_report_cn.md").write_text(
        "\n".join(report_lines), encoding="utf-8"
    )

    (OUTPUT / "holocean_gt_position_reference_query_cn.md").write_text(
        """# 请交给 HoloOcean 数据生成侧确认

当前 `ref_pose.csv` 的 x/y/z 在矩形四次约 90 度原地转向中仅移动约 6-9 微米，
但 metadata 给出 body->IMU 平移 [-0.097,-0.07,+0.06] m、body->CameraLeft
平移 [0.32,+0.11,+0.155] m。请明确回答：

1. `ref_pose.csv` 的 x/y/z 具体来自 agent root/body origin、CameraLeftSocket
   LocationSensor、IMUSocket LocationSensor，还是其他接口？
2. 请给出生成代码的文件、函数和字段来源。
3. position、orientation、velocity 是否可能分别来自不同物理参考点？
4. 原地 yaw 时，body、CameraLeftSocket、IMUSocket 各自理论位置轨迹是什么？
5. 若 x/y/z 是 body/root origin，请在 metadata 中增加
   `ground_truth.position_source` 与 `position_reference_point`。
""",
        encoding="utf-8",
    )

    template = replace_filter_controls(HTML_TEMPLATE)
    template = template.replace("{{", "{").replace("}}", "}")
    template = add_two_dimensional_filter_logic(template)
    template = template.replace(
        "Circle, stop-turn rectangle, and straight trajectory comparison",
        "All historical trajectories evaluated at the IMU origin",
    )
    template = template.replace(
        "__METHOD_SCOPE__",
        "Pure MACVO, T_factor, U1, SA-v1 and SA-v2; raw and all output-filter variants",
    )
    template = template.replace(
        "__LINE_NOTE__",
        (
            "Every estimate is converted from CameraLeftSocket to the IMU origin using metadata. "
            "GT is converted from the empirically inferred body/root origin to IMU. All curves "
            "are translation-rebased at frame 0; no SE(3), yaw, or scale fitting is applied."
        ),
    )
    html_path = OUTPUT / "interactive_imu_center_all_methods.html"
    html_path.write_text(
        template.replace(
            "__DATA__", json.dumps({"scenes": payloads}, ensure_ascii=False)
        ),
        encoding="utf-8",
    )

    figure, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)
    selected = "E_eskf3d_gate_adaptive"
    for axis, (scene, spec), payload in zip(axes, SCENES.items(), payloads):
        gt_points = np.asarray(payload["gt"], dtype=np.float64)
        pure_points = np.asarray(payload["macvo"], dtype=np.float64)
        axis.plot(gt_points[:, 0], gt_points[:, 1], color="#111827", lw=2.5, label="GT")
        axis.plot(
            pure_points[:, 0], pure_points[:, 1], color="#e8590c", lw=1.2, label="Pure MACVO"
        )
        for method, method_spec in METHODS.items():
            item = next(
                trace
                for trace in payload["fusion"]
                if trace["config"] == selected and trace["optimizer"] == method
            )
            values = np.asarray(item["xyz"], dtype=np.float64)
            axis.plot(
                values[:, 0], values[:, 1], color=method_spec["color"], lw=1.2,
                label=f"{method_spec['label']} / 3D ESKF adaptive",
            )
        axis.set_title(str(spec["label"]))
        axis.set_xlabel("x / m (NWU)")
        axis.set_ylabel("y / m (NWU)")
        axis.set_aspect("equal", adjustable="box")
        axis.grid(True, alpha=0.25)
    axes[0].legend(fontsize=8)
    figure.savefig(OUTPUT / "imu_center_selected_xy.png", dpi=180)
    plt.close(figure)

    raw_figure, raw_axes = plt.subplots(
        1, 3, figsize=(18, 6), constrained_layout=True
    )
    for axis, (scene, spec), payload in zip(raw_axes, SCENES.items(), payloads):
        gt_points = np.asarray(payload["gt"], dtype=np.float64)
        pure_points = np.asarray(payload["macvo"], dtype=np.float64)
        axis.plot(
            gt_points[:, 0], gt_points[:, 1], color="#111827", lw=2.7, label="GT"
        )
        axis.plot(
            pure_points[:, 0],
            pure_points[:, 1],
            color="#e8590c",
            lw=1.35,
            label="Pure MACVO",
        )
        for method, method_spec in METHODS.items():
            item = next(
                trace
                for trace in payload["fusion"]
                if trace["config"] == "A_raw" and trace["optimizer"] == method
            )
            values = np.asarray(item["xyz"], dtype=np.float64)
            axis.plot(
                values[:, 0],
                values[:, 1],
                color=method_spec["color"],
                lw=1.25,
                label=method_spec["label"],
            )
        axis.set_title(str(spec["label"]).replace(" / IMU-center evaluation", ""))
        axis.set_xlabel("x / m (NWU)")
        axis.set_ylabel("y / m (NWU)")
        axis.set_aspect("equal", adjustable="box")
        axis.grid(True, alpha=0.25)
    raw_axes[0].legend(fontsize=8)
    raw_figure.savefig(OUTPUT / "imu_center_raw_three_scenes_xy.png", dpi=180)
    plt.close(raw_figure)

    raw_metrics = pd.DataFrame(metric_rows)
    raw_metrics = raw_metrics[raw_metrics["output_method"] == "A_raw"].copy()
    method_order = ["pure_macvo", *METHODS.keys()]
    scene_order = list(SCENES)
    metric_specs = [
        ("xy_ate_rmse_m", "XY ATE RMSE / m"),
        ("translation_rpe_rmse_m", "Translation RPE RMSE / m"),
        ("rotation_rpe_rmse_rad", "Rotation RPE RMSE / rad"),
    ]
    metric_figure, metric_axes = plt.subplots(
        1, 3, figsize=(18, 5.8), constrained_layout=True
    )
    x = np.arange(len(scene_order), dtype=np.float64)
    width = 0.15
    colors_by_method = {
        "pure_macvo": "#e8590c",
        **{key: str(value["color"]) for key, value in METHODS.items()},
    }
    labels_by_method = {
        "pure_macvo": "Pure MACVO",
        **{key: str(value["label"]) for key, value in METHODS.items()},
    }
    for axis, (column, title) in zip(metric_axes, metric_specs):
        for method_index, method in enumerate(method_order):
            values = []
            for scene in scene_order:
                row = raw_metrics[
                    (raw_metrics["scene"] == scene)
                    & (raw_metrics["optimizer"] == method)
                ]
                if len(row) != 1:
                    raise AssertionError(f"missing raw metric: {scene}/{method}")
                values.append(float(row.iloc[0][column]))
            offset = (method_index - (len(method_order) - 1) / 2.0) * width
            axis.bar(
                x + offset,
                values,
                width=width,
                color=colors_by_method[method],
                label=labels_by_method[method],
            )
        axis.set_title(title)
        axis.set_xticks(x, [SCENES[name]["label"].split(" /")[0] for name in scene_order])
        axis.grid(True, axis="y", alpha=0.25)
    metric_axes[0].legend(fontsize=8)
    metric_figure.savefig(
        OUTPUT / "imu_center_raw_cross_scene_metrics.png", dpi=180
    )
    plt.close(metric_figure)
    return html_path, metrics_path, contract_path


def main() -> None:
    for path in build():
        print(path)


if __name__ == "__main__":
    main()
