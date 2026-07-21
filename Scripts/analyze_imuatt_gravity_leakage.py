#!/usr/bin/env python3
"""Diagnose whether imuatt attitude drift causes gravity-leakage trajectory drift.

This is an offline attribution tool. It reproduces the deployed
``imu_integrated_estinit`` attitude source from existing IMU recordings, then
compares its gravity projection with GT. It does not run MACVO and does not
modify production code or existing results.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation, Slerp


WORKDIR = Path("/home/admin1/macvo-dev")
DEFAULT_DATA_ROOT = Path(
    "/mnt/e/文档/holoocean/code/recordings/"
    "batch_zed100_closed_paths_smooth_20260705"
)
DEFAULT_RESULT_ROOT = (
    WORKDIR
    / "Results"
    / "gravity_handling_ablation_20260711"
    / "trial_1"
    / "vio_preintegrated_full_imuatt_estinit"
)
DEFAULT_PURE_MACVO_ROOT = (
    WORKDIR
    / "Results"
    / "closed_paths_latest_20260706_timeinterp"
    / "trial_1"
    / "pure_macvo"
)
DEFAULT_OUT_ROOT = WORKDIR / "analysis_imuatt_gravity_leakage_20260711"
SCENES = {
    "clear_rectangle_zero_noise": "zero_noise/clear_rectangle_path",
    "clear_rectangle_normal_noise": "normal_noise/clear_rectangle_path",
}
NWU_TO_NED = np.diag([1.0, -1.0, -1.0])
COLORS = {
    "zero": "#168aad",
    "normal": "#d95f02",
    "actual": "#7b2cbf",
    "predicted": "#2a9d8f",
}


def query_imu_interval(
    timestamps: np.ndarray,
    values: np.ndarray,
    start_ns: int,
    end_ns: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Match ``IMUCSVLoader.query_range`` endpoint interpolation semantics."""
    timestamps = np.asarray(timestamps, dtype=np.int64).reshape(-1)
    values = np.asarray(values, dtype=np.float64)
    if end_ns < start_ns:
        start_ns, end_ns = end_ns, start_ns
    if timestamps.size == 0 or end_ns < timestamps[0] or start_ns > timestamps[-1]:
        return timestamps[:0], values[:0]
    start_ns = max(int(start_ns), int(timestamps[0]))
    end_ns = min(int(end_ns), int(timestamps[-1]))

    def interpolate(target_ns: int) -> np.ndarray:
        idx = int(np.searchsorted(timestamps, target_ns, side="left"))
        if idx < timestamps.size and int(timestamps[idx]) == target_ns:
            return values[idx].copy()
        if idx <= 0:
            return values[0].copy()
        if idx >= timestamps.size:
            return values[-1].copy()
        left = idx - 1
        alpha = (target_ns - int(timestamps[left])) / float(
            int(timestamps[idx]) - int(timestamps[left])
        )
        return values[left] + (values[idx] - values[left]) * alpha

    if start_ns == end_ns:
        return np.array([start_ns], dtype=np.int64), interpolate(start_ns)[None]
    i0 = int(np.searchsorted(timestamps, start_ns, side="right"))
    i1 = int(np.searchsorted(timestamps, end_ns, side="left"))
    out_t = np.concatenate(
        [np.array([start_ns], dtype=np.int64), timestamps[i0:i1], np.array([end_ns], dtype=np.int64)]
    )
    out_v = np.concatenate(
        [interpolate(start_ns)[None], values[i0:i1], interpolate(end_ns)[None]], axis=0
    )
    return out_t, out_v


def integrate_gyro_attitude(
    rotation_body_to_world: np.ndarray,
    time_ns: np.ndarray,
    gyro_body: np.ndarray,
) -> np.ndarray:
    """Reproduce ``integrate_gyro_attitude_world`` with midpoint samples."""
    rotation = np.asarray(rotation_body_to_world, dtype=np.float64).reshape(3, 3).copy()
    time_ns = np.asarray(time_ns, dtype=np.int64).reshape(-1)
    gyro_body = np.asarray(gyro_body, dtype=np.float64).reshape(-1, 3)
    if gyro_body.size == 0:
        return rotation
    if len(gyro_body) <= 1 or len(time_ns) <= 1:
        dt_values = np.array([0.01], dtype=np.float64)
    else:
        dt_values = np.maximum(np.diff(time_ns).astype(np.float64), 1.0) * 1e-9
    for k in range(max(len(gyro_body) - 1, 1)):
        kn = min(k + 1, len(gyro_body) - 1)
        dt = float(dt_values[min(k, len(dt_values) - 1)])
        gyro_mid = 0.5 * (gyro_body[k] + gyro_body[kn])
        rotation = rotation @ Rotation.from_rotvec(gyro_mid * dt).as_matrix()
    return rotation


def gravity_leakage_world(
    *,
    rotation_est: np.ndarray,
    rotation_gt: np.ndarray,
    gravity_world: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Gravity correction error caused only by using the wrong attitude."""
    rotation_est = np.asarray(rotation_est, dtype=np.float64).reshape(3, 3)
    rotation_gt = np.asarray(rotation_gt, dtype=np.float64).reshape(3, 3)
    gravity_world = np.asarray(gravity_world, dtype=np.float64).reshape(3)
    leakage_body = rotation_est.T @ gravity_world - rotation_gt.T @ gravity_world
    return leakage_body, rotation_gt @ leakage_body


def accumulate_constant_interval_acceleration(
    *,
    velocity: np.ndarray,
    position: np.ndarray,
    acceleration: np.ndarray,
    dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    velocity = np.asarray(velocity, dtype=np.float64).reshape(3)
    position = np.asarray(position, dtype=np.float64).reshape(3)
    acceleration = np.asarray(acceleration, dtype=np.float64).reshape(3)
    next_position = position + velocity * dt + 0.5 * acceleration * dt * dt
    next_velocity = velocity + acceleration * dt
    return next_velocity, next_position


def internal_rotation_from_xyzw(quaternion: np.ndarray) -> np.ndarray:
    rotation_nwu = Rotation.from_quat(np.asarray(quaternion, dtype=np.float64)).as_matrix()
    return NWU_TO_NED @ rotation_nwu @ NWU_TO_NED


def angle_deg(rotation_error: np.ndarray) -> float:
    return math.degrees(float(np.linalg.norm(Rotation.from_matrix(rotation_error).as_rotvec())))


def vector_cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom < 1e-12:
        return math.nan
    return float(np.dot(a, b) / denom)


def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    valid = np.isfinite(a) & np.isfinite(b)
    if int(valid.sum()) < 3 or np.std(a[valid]) < 1e-12 or np.std(b[valid]) < 1e-12:
        return math.nan
    return float(np.corrcoef(a[valid], b[valid])[0, 1])


def last_finite(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = values[np.isfinite(values)]
    return float(finite[-1]) if finite.size else math.nan


def vector_fit(actual: np.ndarray, predicted: np.ndarray) -> tuple[float, float]:
    """Fit actual ~= scale * predicted and return scale and uncentered R2."""
    actual = np.asarray(actual, dtype=np.float64).reshape(-1)
    predicted = np.asarray(predicted, dtype=np.float64).reshape(-1)
    valid = np.isfinite(actual) & np.isfinite(predicted)
    actual = actual[valid]
    predicted = predicted[valid]
    denom = float(np.dot(predicted, predicted))
    if actual.size < 3 or denom < 1e-12:
        return math.nan, math.nan
    scale = float(np.dot(predicted, actual) / denom)
    residual = actual - scale * predicted
    total = float(np.dot(actual, actual))
    r2 = 1.0 - float(np.dot(residual, residual)) / total if total > 1e-12 else math.nan
    return scale, r2


def load_metadata(scene_root: Path) -> dict:
    with (scene_root / "metadata.json").open("r", encoding="utf-8") as handle:
        return json.load(handle)


def interpolate_imu_gt_rotations(imu: pd.DataFrame, frame_times: np.ndarray) -> np.ndarray:
    imu_times = imu["timestamp"].to_numpy(np.int64)
    base = float(imu_times[0])
    rotations = Rotation.from_quat(imu[["qx", "qy", "qz", "qw"]].to_numpy(np.float64))
    slerp = Slerp((imu_times.astype(np.float64) - base) * 1e-9, rotations)
    sampled = slerp((frame_times.astype(np.float64) - base) * 1e-9).as_matrix()
    return np.einsum("ab,nbc,cd->nad", NWU_TO_NED, sampled, NWU_TO_NED)


def analyze_scene(
    *,
    scene: str,
    scene_root: Path,
    result_dir: Path,
    pure_macvo_dir: Path,
    out_root: Path,
) -> tuple[pd.DataFrame, dict[str, object]]:
    metadata = load_metadata(scene_root)
    gravity = float(metadata["imu"]["gravity_m_s2"])
    imu = pd.read_csv(scene_root / "imu_data.csv")
    ref = pd.read_csv(scene_root / "ref_pose.csv").rename(columns={"timestamp": "timestamp_ns"})
    poses = pd.read_csv(result_dir / "poses.csv")
    pure_poses = pd.read_csv(pure_macvo_dir / "poses.csv").set_index("timestamp_ns")
    diagnostics = pd.read_csv(result_dir / "frame_pair_diagnostics.csv")

    frame_times = poses["timestamp_ns"].to_numpy(np.int64)
    ref = ref.set_index("timestamp_ns").loc[frame_times].reset_index()
    if len(frame_times) != len(ref):
        raise ValueError(f"{scene}: pose/ref length mismatch")

    imu_times = imu["timestamp"].to_numpy(np.int64)
    gyro_flu = imu[["ang_vel_x", "ang_vel_y", "ang_vel_z"]].to_numpy(np.float64)
    gyro_ned = gyro_flu @ NWU_TO_NED.T
    gt_rot_imu = interpolate_imu_gt_rotations(imu, frame_times)
    gt_rot_ref = np.stack(
        [
            internal_rotation_from_xyzw(q)
            for q in ref[["qx", "qy", "qz", "qw"]].to_numpy(np.float64)
        ],
        axis=0,
    )
    gt_rotation_reference_delta_deg = np.array(
        [angle_deg(gt_rot_imu[i].T @ gt_rot_ref[i]) for i in range(len(frame_times))]
    )

    pose_rotations = np.stack(
        [
            Rotation.from_quat(q).as_matrix()
            for q in poses[["qx", "qy", "qz", "qw"]].to_numpy(np.float64)
        ],
        axis=0,
    )
    estimated_position = poses[["tx", "ty", "tz"]].to_numpy(np.float64)
    pure_position = pure_poses.loc[frame_times][["tx", "ty", "tz"]].to_numpy(np.float64)
    gt_position = ref[["x", "y", "z"]].to_numpy(np.float64) @ NWU_TO_NED.T
    actual_position_error = estimated_position - gt_position
    imu_effect_position = estimated_position - pure_position

    n = len(frame_times)
    integrated_rotations = np.zeros((n, 3, 3), dtype=np.float64)
    integrated_rotations[0] = pose_rotations[0]
    attitude_error_deg = np.zeros(n, dtype=np.float64)
    tilt_error_deg = np.zeros(n, dtype=np.float64)
    optimizer_attitude_error_deg = np.zeros(n, dtype=np.float64)
    leakage_body = np.zeros((n, 3), dtype=np.float64)
    leakage_world = np.zeros((n, 3), dtype=np.float64)
    leakage_world_gt_mapping = np.zeros((n, 3), dtype=np.float64)
    predicted_velocity_error = np.zeros((n, 3), dtype=np.float64)
    predicted_position_error = np.zeros((n, 3), dtype=np.float64)
    imu_sample_counts = np.zeros(n, dtype=np.int64)

    gravity_world = np.array([0.0, 0.0, gravity], dtype=np.float64)
    for i in range(n):
        full_error = integrated_rotations[i].T @ gt_rot_ref[i]
        attitude_error_deg[i] = angle_deg(full_error)
        optimizer_attitude_error_deg[i] = angle_deg(pose_rotations[i].T @ gt_rot_ref[i])
        leakage_body[i], leakage_world_gt_mapping[i] = gravity_leakage_world(
            rotation_est=integrated_rotations[i],
            rotation_gt=gt_rot_ref[i],
            gravity_world=gravity_world,
        )
        # This is the mapping actually used by the coupled residual: the IMU
        # delta is expressed in body_i and mapped by the optimized pose_i.
        leakage_world[i] = pose_rotations[i] @ leakage_body[i]
        g_est = integrated_rotations[i].T @ gravity_world
        g_gt = gt_rot_ref[i].T @ gravity_world
        tilt_error_deg[i] = math.degrees(
            math.acos(float(np.clip(np.dot(g_est, g_gt) / (gravity * gravity), -1.0, 1.0)))
        )
        if i == n - 1:
            continue
        start_ns = int(frame_times[i])
        end_ns = int(frame_times[i + 1])
        interval_t, interval_gyro = query_imu_interval(
            imu_times, gyro_ned, start_ns, end_ns
        )
        imu_sample_counts[i + 1] = len(interval_t)
        dt = (end_ns - start_ns) * 1e-9
        predicted_velocity_error[i + 1], predicted_position_error[i + 1] = (
            accumulate_constant_interval_acceleration(
                velocity=predicted_velocity_error[i],
                position=predicted_position_error[i],
                acceleration=leakage_world[i],
                dt=dt,
            )
        )
        integrated_rotations[i + 1] = integrate_gyro_attitude(
            integrated_rotations[i], interval_t, interval_gyro
        )

    actual_velocity_error = np.full((n, 3), np.nan, dtype=np.float64)
    diag_by_frame = diagnostics.set_index("frame_j")
    for frame_j, row in diag_by_frame.iterrows():
        idx = int(frame_j)
        if 0 <= idx < n:
            actual_velocity_error[idx] = np.array(
                [row.est_velocity_j_x - row.gt_velocity_j_x,
                 row.est_velocity_j_y - row.gt_velocity_j_y,
                 row.est_velocity_j_z - row.gt_velocity_j_z],
                dtype=np.float64,
            )

    time_s = (frame_times - frame_times[0]).astype(np.float64) * 1e-9
    output = pd.DataFrame(
        {
            "scene": scene,
            "frame": np.arange(n),
            "timestamp_ns": frame_times,
            "time_s": time_s,
            "imu_samples": imu_sample_counts,
            "attitude_error_deg": attitude_error_deg,
            "tilt_error_deg": tilt_error_deg,
            "optimizer_attitude_error_deg": optimizer_attitude_error_deg,
            "gt_imu_vs_ref_attitude_deg": gt_rotation_reference_delta_deg,
            "gt_imu_vs_ref_tilt_deg": [
                math.degrees(
                    math.acos(
                        float(
                            np.clip(
                                np.dot(gt_rot_imu[i].T @ gravity_world, gt_rot_ref[i].T @ gravity_world)
                                / (gravity * gravity),
                                -1.0,
                                1.0,
                            )
                        )
                    )
                )
                for i in range(n)
            ],
            "gravity_leakage_norm_m_s2": np.linalg.norm(leakage_world, axis=1),
            "predicted_velocity_error_norm_m_s": np.linalg.norm(predicted_velocity_error, axis=1),
            "actual_velocity_error_norm_m_s": np.linalg.norm(actual_velocity_error, axis=1),
            "predicted_position_error_norm_m": np.linalg.norm(predicted_position_error, axis=1),
            "actual_position_error_norm_m": np.linalg.norm(actual_position_error, axis=1),
            "imu_effect_vs_pure_macvo_norm_m": np.linalg.norm(imu_effect_position, axis=1),
        }
    )
    vectors = {
        "gravity_leakage_body": leakage_body,
        "gravity_leakage_world": leakage_world,
        "gravity_leakage_world_gt_mapping": leakage_world_gt_mapping,
        "predicted_velocity_error": predicted_velocity_error,
        "actual_velocity_error": actual_velocity_error,
        "predicted_position_error": predicted_position_error,
        "actual_position_error": actual_position_error,
        "imu_effect_vs_pure_macvo": imu_effect_position,
    }
    for name, values in vectors.items():
        for axis, suffix in enumerate(("x", "y", "z")):
            output[f"{name}_{suffix}"] = values[:, axis]
    output["velocity_error_cosine"] = [
        vector_cosine(actual_velocity_error[i], predicted_velocity_error[i]) for i in range(n)
    ]
    output["position_error_cosine"] = [
        vector_cosine(actual_position_error[i], predicted_position_error[i]) for i in range(n)
    ]
    output["imu_effect_vs_predicted_cosine"] = [
        vector_cosine(imu_effect_position[i], predicted_position_error[i]) for i in range(n)
    ]

    velocity_scale, velocity_r2 = vector_fit(actual_velocity_error[1:], predicted_velocity_error[1:])
    position_scale, position_r2 = vector_fit(actual_position_error[1:], predicted_position_error[1:])
    imu_effect_scale, imu_effect_r2 = vector_fit(imu_effect_position[1:], predicted_position_error[1:])
    actual_velocity_norm = np.linalg.norm(actual_velocity_error, axis=1)
    velocity_cosines = output["velocity_error_cosine"].to_numpy(np.float64)
    summary: dict[str, object] = {
        "scene": scene,
        "frames": n,
        "duration_s": float(time_s[-1]),
        "attitude_error_median_deg": float(np.median(attitude_error_deg)),
        "attitude_error_final_deg": float(attitude_error_deg[-1]),
        "attitude_error_max_deg": float(np.max(attitude_error_deg)),
        "tilt_error_median_deg": float(np.median(tilt_error_deg)),
        "tilt_error_final_deg": float(tilt_error_deg[-1]),
        "tilt_error_max_deg": float(np.max(tilt_error_deg)),
        "gravity_leakage_median_m_s2": float(np.median(np.linalg.norm(leakage_world, axis=1))),
        "gravity_leakage_final_m_s2": float(np.linalg.norm(leakage_world[-1])),
        "gravity_leakage_max_m_s2": float(np.max(np.linalg.norm(leakage_world, axis=1))),
        "predicted_velocity_error_final_m_s": float(np.linalg.norm(predicted_velocity_error[-1])),
        "actual_velocity_error_final_m_s": last_finite(actual_velocity_norm),
        "velocity_norm_correlation": safe_corr(
            np.linalg.norm(predicted_velocity_error[1:], axis=1),
            np.linalg.norm(actual_velocity_error[1:], axis=1),
        ),
        "velocity_vector_scale": velocity_scale,
        "velocity_vector_r2": velocity_r2,
        "velocity_cosine_median": float(np.nanmedian(output["velocity_error_cosine"])),
        "velocity_cosine_final": last_finite(velocity_cosines),
        "predicted_position_error_final_m": float(np.linalg.norm(predicted_position_error[-1])),
        "actual_position_error_final_m": float(np.linalg.norm(actual_position_error[-1])),
        "position_norm_correlation": safe_corr(
            np.linalg.norm(predicted_position_error[1:], axis=1),
            np.linalg.norm(actual_position_error[1:], axis=1),
        ),
        "position_vector_scale": position_scale,
        "position_vector_r2": position_r2,
        "position_cosine_median": float(np.nanmedian(output["position_error_cosine"])),
        "position_cosine_final": float(output["position_error_cosine"].iloc[-1]),
        "imu_effect_norm_correlation": safe_corr(
            np.linalg.norm(predicted_position_error[1:], axis=1),
            np.linalg.norm(imu_effect_position[1:], axis=1),
        ),
        "imu_effect_vector_scale": imu_effect_scale,
        "imu_effect_vector_r2": imu_effect_r2,
        "imu_effect_cosine_median": float(np.nanmedian(output["imu_effect_vs_predicted_cosine"])),
        "imu_effect_cosine_final": float(output["imu_effect_vs_predicted_cosine"].iloc[-1]),
        "gt_imu_vs_ref_attitude_max_deg": float(np.max(gt_rotation_reference_delta_deg)),
        "gt_imu_vs_ref_tilt_max_deg": float(np.max(output["gt_imu_vs_ref_tilt_deg"])),
        "interval_imu_sample_count_median": float(np.median(imu_sample_counts[1:])),
        "source_scene_root": str(scene_root),
        "source_result_dir": str(result_dir),
        "source_pure_macvo_dir": str(pure_macvo_dir),
    }
    scene_dir = out_root / scene
    scene_dir.mkdir(parents=True, exist_ok=True)
    output.to_csv(scene_dir / "frame_gravity_leakage_diagnostics.csv", index=False)
    return output, summary


def write_summary_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_static_plot(out_root: Path, frames: dict[str, pd.DataFrame]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.5), dpi=170, sharex=True)
    specs = [
        ("tilt_error_deg", "attitude tilt error / deg"),
        ("gravity_leakage_norm_m_s2", "gravity leakage / m/s^2"),
        ("actual_velocity_error_norm_m_s", "velocity error / m/s"),
        ("actual_position_error_norm_m", "position error / m"),
    ]
    for ax, (column, label) in zip(axes.flat, specs):
        for scene, frame in frames.items():
            color = COLORS["normal" if "normal" in scene else "zero"]
            ax.plot(frame["time_s"], frame[column], color=color, lw=1.5, label=scene)
            if column.startswith("actual_velocity"):
                ax.plot(frame["time_s"], frame["predicted_velocity_error_norm_m_s"], color=color, lw=1.0, ls="--")
            if column.startswith("actual_position"):
                ax.plot(frame["time_s"], frame["predicted_position_error_norm_m"], color=color, lw=1.0, ls="--")
        ax.set_ylabel(label)
        ax.grid(True, linewidth=0.4, alpha=0.45)
    axes[1, 0].set_xlabel("time / s")
    axes[1, 1].set_xlabel("time / s")
    axes[0, 0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_root / "gravity_leakage_attribution_overview.png")
    plt.close(fig)


def rounded(values: pd.Series) -> list[float]:
    return [round(float(v), 7) if np.isfinite(v) else None for v in values]


def write_interactive_html(
    out_root: Path,
    frames: dict[str, pd.DataFrame],
    summaries: list[dict[str, object]],
) -> None:
    view_specs = {
        "attitude": ("姿态误差", [("attitude_error_deg", "完整姿态误差"), ("tilt_error_deg", "倾斜误差")], "deg"),
        "gravity": ("重力泄漏", [("gravity_leakage_norm_m_s2", "重力泄漏")], "m/s²"),
        "velocity": ("速度漂移", [("actual_velocity_error_norm_m_s", "实际 VIO"), ("predicted_velocity_error_norm_m_s", "仅重力泄漏预测")], "m/s"),
        "position": ("位置漂移", [("actual_position_error_norm_m", "实际 VIO 对 GT"), ("predicted_position_error_norm_m", "仅重力泄漏预测"), ("imu_effect_vs_pure_macvo_norm_m", "VIO 相对 pure MACVO 改变量")], "m"),
        "direction": ("方向一致性", [("velocity_error_cosine", "速度误差余弦"), ("position_error_cosine", "位置误差余弦")], "cosine"),
    }
    payload: dict[str, object] = {"views": {}, "summaries": summaries}
    for view, (title, metrics, unit) in view_specs.items():
        traces = []
        for scene, frame in frames.items():
            base = COLORS["normal" if "normal" in scene else "zero"]
            for metric_index, (column, metric_name) in enumerate(metrics):
                trace_color = (
                    base
                    if metric_index == 0
                    else COLORS["predicted"]
                    if metric_index == 1
                    else COLORS["actual"]
                )
                traces.append(
                    {
                        "name": f"{scene} / {metric_name}",
                        "color": trace_color,
                        "dash": metric_index > 0,
                        "x": rounded(frame["time_s"]),
                        "y": rounded(frame[column]),
                    }
                )
        payload["views"][view] = {"title": title, "unit": unit, "traces": traces}

    data_json = json.dumps(payload, ensure_ascii=False)
    cards = "".join(
        f"<div class='card'><strong>{html.escape(str(row['scene']))}</strong>"
        f"<span>tilt final: {float(row['tilt_error_final_deg']):.3f} deg</span>"
        f"<span>gravity leak final: {float(row['gravity_leakage_final_m_s2']):.3f} m/s²</span>"
        f"<span>velocity corr: {float(row['velocity_norm_correlation']):.3f}</span>"
        f"<span>position corr: {float(row['position_norm_correlation']):.3f}</span></div>"
        for row in summaries
    )
    page = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>imuatt 姿态漂移与重力泄漏诊断</title>
<style>
body{{font-family:Arial,Helvetica,sans-serif;margin:0;background:#f4f6f8;color:#26323f;padding:18px}}
.panel{{max-width:1240px;margin:auto;background:#fff;border:1px solid #d6dde5}}
.head{{padding:14px 16px;border-bottom:1px solid #d6dde5}}h1{{font-size:20px;margin:0 0 8px}}p{{margin:5px 0;line-height:1.5}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(270px,1fr));gap:8px;margin-top:10px}}.card{{border:1px solid #d6dde5;padding:9px;display:grid;gap:4px;font-size:12px}}
.toolbar{{display:flex;gap:7px;flex-wrap:wrap;padding:10px 14px;border-bottom:1px solid #d6dde5}}button{{padding:6px 10px;border:1px solid #9da9b5;background:#f8fafc;cursor:pointer}}button.active{{background:#1769aa;color:#fff;border-color:#1769aa}}
#legend{{display:flex;gap:10px;flex-wrap:wrap;padding:8px 14px;font-size:12px;border-bottom:1px solid #e3e7eb}}label{{display:flex;gap:4px;align-items:center}}
#wrap{{height:650px;position:relative}}canvas{{width:100%;height:100%;display:block}}#tip{{position:absolute;display:none;background:#fff;border:1px solid #8e99a4;padding:6px;font-size:12px;pointer-events:none}}
.foot{{padding:9px 14px;border-top:1px solid #d6dde5;color:#5b6773;font-size:12px}}
</style></head><body><div class="panel"><div class="head"><h1>imuatt 独立姿态积分与重力泄漏归因</h1>
<p>实线为实际量，虚线为仅由姿态误差造成的重力泄漏预测。预测未包含加速度噪声、视觉修正或优化器反馈，因此它是单一机制解释量，不是轨迹重放。</p><div class="cards">{cards}</div></div>
<div class="toolbar"><button data-view="attitude" class="active">姿态</button><button data-view="gravity">重力泄漏</button><button data-view="velocity">速度</button><button data-view="position">位置</button><button data-view="direction">方向一致性</button><button id="reset">重置视图</button></div>
<div id="legend"></div><div id="wrap"><canvas id="plot"></canvas><div id="tip"></div></div><div class="foot">鼠标滚轮缩放时间轴，拖动平移，点击图例切换曲线。</div></div>
<script>
const DATA={data_json}; const canvas=document.getElementById('plot'),ctx=canvas.getContext('2d'),wrap=document.getElementById('wrap'),legend=document.getElementById('legend'),tip=document.getElementById('tip');
let state={{view:'attitude',xDomain:null,visible:{{}},drag:null}}; const M={{l:72,r:24,t:28,b:56}};
function traces(){{return DATA.views[state.view].traces}}
function finite(a){{return a.filter(v=>v!==null&&Number.isFinite(v))}}
function resetDomain(){{const xs=traces().flatMap(t=>finite(t.x));state.xDomain=[Math.min(...xs),Math.max(...xs)]}}
function resize(){{const dpr=devicePixelRatio||1;canvas.width=wrap.clientWidth*dpr;canvas.height=wrap.clientHeight*dpr;ctx.setTransform(dpr,0,0,dpr,0,0);draw()}}
function domains(){{if(!state.xDomain)resetDomain();let ys=[];traces().forEach(t=>{{if(state.visible[t.name]!==false)ys.push(...finite(t.y))}});let lo=Math.min(...ys),hi=Math.max(...ys);if(!Number.isFinite(lo)){{lo=0;hi=1}}if(Math.abs(hi-lo)<1e-9){{lo-=1;hi+=1}}const pad=(hi-lo)*.08;return{{x:state.xDomain,y:[lo-pad,hi+pad]}}}}
function mapX(x,d,w){{return M.l+(x-d.x[0])/(d.x[1]-d.x[0])*(w-M.l-M.r)}}function mapY(y,d,h){{return h-M.b-(y-d.y[0])/(d.y[1]-d.y[0])*(h-M.t-M.b)}}
function draw(){{const w=wrap.clientWidth,h=wrap.clientHeight;ctx.clearRect(0,0,w,h);const d=domains();ctx.font='12px Arial';ctx.strokeStyle='#d8dee6';ctx.fillStyle='#26323f';for(let i=0;i<=5;i++){{let x=d.x[0]+i*(d.x[1]-d.x[0])/5,px=mapX(x,d,w);ctx.beginPath();ctx.moveTo(px,M.t);ctx.lineTo(px,h-M.b);ctx.stroke();ctx.fillText(x.toFixed(1),px-12,h-M.b+20);let y=d.y[0]+i*(d.y[1]-d.y[0])/5,py=mapY(y,d,h);ctx.beginPath();ctx.moveTo(M.l,py);ctx.lineTo(w-M.r,py);ctx.stroke();ctx.fillText(y.toFixed(3),6,py+4)}}ctx.strokeStyle='#202832';ctx.strokeRect(M.l,M.t,w-M.l-M.r,h-M.t-M.b);ctx.font='bold 13px Arial';ctx.fillText('time / s',w/2-25,h-14);ctx.save();ctx.translate(18,h/2);ctx.rotate(-Math.PI/2);ctx.fillText(DATA.views[state.view].unit,0,0);ctx.restore();traces().forEach(t=>{{if(state.visible[t.name]===false)return;ctx.strokeStyle=t.color;ctx.lineWidth=2;ctx.setLineDash(t.dash?[7,5]:[]);ctx.beginPath();let started=false;t.x.forEach((x,i)=>{{let y=t.y[i];if(y===null)return;let px=mapX(x,d,w),py=mapY(y,d,h);if(!started){{ctx.moveTo(px,py);started=true}}else ctx.lineTo(px,py)}});ctx.stroke()}});ctx.setLineDash([])}}
function buildLegend(){{legend.innerHTML='';traces().forEach(t=>{{if(!(t.name in state.visible))state.visible[t.name]=true;const l=document.createElement('label'),c=document.createElement('input'),s=document.createElement('span');c.type='checkbox';c.checked=state.visible[t.name];c.onchange=()=>{{state.visible[t.name]=c.checked;draw()}};s.textContent=t.name;s.style.color=t.color;l.append(c,s);legend.append(l)}})}}
document.querySelectorAll('[data-view]').forEach(b=>b.onclick=()=>{{document.querySelectorAll('[data-view]').forEach(x=>x.classList.remove('active'));b.classList.add('active');state.view=b.dataset.view;state.xDomain=null;buildLegend();draw()}});document.getElementById('reset').onclick=()=>{{state.xDomain=null;draw()}};
canvas.onwheel=e=>{{e.preventDefault();const rect=canvas.getBoundingClientRect(),d=domains(),cx=d.x[0]+(e.clientX-rect.left-M.l)/(rect.width-M.l-M.r)*(d.x[1]-d.x[0]),f=e.deltaY<0?.82:1.22;state.xDomain=[cx+(d.x[0]-cx)*f,cx+(d.x[1]-cx)*f];draw()}};
canvas.onpointerdown=e=>{{state.drag={{x:e.clientX,domain:[...domains().x]}};canvas.setPointerCapture(e.pointerId)}};canvas.onpointermove=e=>{{if(!state.drag)return;const span=state.drag.domain[1]-state.drag.domain[0],dx=(e.clientX-state.drag.x)/(wrap.clientWidth-M.l-M.r)*span;state.xDomain=[state.drag.domain[0]-dx,state.drag.domain[1]-dx];draw()}};canvas.onpointerup=e=>{{state.drag=null;try{{canvas.releasePointerCapture(e.pointerId)}}catch(_){{}}}};
window.onresize=resize;buildLegend();resize();
</script></body></html>"""
    (out_root / "diagnostics_interactive.html").write_text(page, encoding="utf-8")


def write_report(out_root: Path, summaries: list[dict[str, object]]) -> None:
    by_scene = {str(row["scene"]): row for row in summaries}
    zero = by_scene["clear_rectangle_zero_noise"]
    normal = by_scene["clear_rectangle_normal_noise"]
    tilt_ratio = float(normal["tilt_error_median_deg"]) / max(float(zero["tilt_error_median_deg"]), 1e-12)
    leakage_ratio = float(normal["gravity_leakage_median_m_s2"]) / max(float(zero["gravity_leakage_median_m_s2"]), 1e-12)
    lines = [
        "# imuatt 姿态漂移与重力泄漏诊断",
        "",
        "## 诊断边界",
        "",
        "- 复现生产链路的相机帧边界插值、FLU/NWU 到内部 NED 转换以及 gyro 梯形积分。",
        "- 使用 `imu_data.csv` 中的姿态四元数作为 IMU 姿态真值。",
        "- 只计算姿态错误造成的重力投影误差；不包含加速度白噪声、隐藏 Bias、视觉项和优化器反馈。",
        "- 未修改生产代码，未运行 MACVO。",
        "",
        "## 核心统计",
        "",
        "| scene | tilt median deg | tilt final deg | gravity leak median m/s2 | velocity norm corr | velocity vector R2 | position norm corr | position vector R2 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            f"| {row['scene']} | {float(row['tilt_error_median_deg']):.6f} | "
            f"{float(row['tilt_error_final_deg']):.6f} | {float(row['gravity_leakage_median_m_s2']):.6f} | "
            f"{float(row['velocity_norm_correlation']):.6f} | {float(row['velocity_vector_r2']):.6f} | "
            f"{float(row['position_norm_correlation']):.6f} | {float(row['position_vector_r2']):.6f} |"
        )
    lines.extend(
        [
            "",
            "## 可直接确认的代码事实",
            "",
            "1. `imu_integrated_estinit` 的独立姿态只由原始 gyro 累积。",
            "2. 该姿态用于 `R0_world^{-1} g_world`，因此 roll/pitch 误差必然转化为水平加速度泄漏。",
            "3. 独立姿态积分没有减去优化器估计的 gyro Bias，也不会被优化后的视觉姿态校正。",
            "",
            "## 场景差异",
            "",
            f"- normal-noise 的中位倾斜误差是 zero-noise 的 `{tilt_ratio:.2f}x`。",
            f"- normal-noise 的中位重力泄漏是 zero-noise 的 `{leakage_ratio:.2f}x`。",
            f"- normal-noise 速度误差模长相关系数为 `{float(normal['velocity_norm_correlation']):.4f}`，位置误差模长相关系数为 `{float(normal['position_norm_correlation']):.4f}`。",
            f"- normal-noise 中，VIO 相对 pure MACVO 的位置改变量与重力泄漏预测的模长相关系数为 `{float(normal['imu_effect_norm_correlation']):.4f}`，向量拟合 R2 为 `{float(normal['imu_effect_vector_r2']):.4f}`。",
            "",
            "## 因果判断",
            "",
            "结论是 **部分支持，而不是完全支持**：",
            "",
            "1. normal-noise 下，独立 gyro 姿态确实产生了明显 roll/pitch 漂移，并形成足以影响估计的错误重力分量；zero-noise 下该机制基本不存在。",
            f"2. 速度层证据较强：预测与实际误差方向余弦中位数为 `{float(normal['velocity_cosine_median']):.4f}`，向量拟合 R2 为 `{float(normal['velocity_vector_r2']):.4f}`。因此错误重力投影是速度漂移的重要来源。",
            f"3. 但仅靠重力泄漏预测的末端速度误差为 `{float(normal['predicted_velocity_error_final_m_s']):.4f} m/s`，实际为 `{float(normal['actual_velocity_error_final_m_s']):.4f} m/s`，它不能解释全部误差。",
            f"4. 位置误差虽然模长增长高度相关，但方向余弦中位数为 `{float(normal['position_cosine_median']):.4f}`，说明最终轨迹方向还受到视觉残差、自由速度状态和逐帧优化反馈的显著影响。",
            "",
            "因此可以确认：当前 `imuatt` 的独立姿态重力消除链路在 normal-noise 下有实质性缺陷，并会拖坏融合；但不能把整条坏轨迹全部归因于它。",
        ]
    )
    (out_root / "analysis_report_cn.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--pure-macvo-root", type=Path, default=DEFAULT_PURE_MACVO_ROOT)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out_root.mkdir(parents=True, exist_ok=True)
    frames: dict[str, pd.DataFrame] = {}
    summaries: list[dict[str, object]] = []
    for scene, relative_root in SCENES.items():
        frame, summary = analyze_scene(
            scene=scene,
            scene_root=args.data_root / relative_root,
            result_dir=args.result_root / scene,
            pure_macvo_dir=args.pure_macvo_root / scene,
            out_root=args.out_root,
        )
        frames[scene] = frame
        summaries.append(summary)
    write_summary_csv(args.out_root / "summary.csv", summaries)
    write_static_plot(args.out_root, frames)
    write_interactive_html(args.out_root, frames, summaries)
    write_report(args.out_root, summaries)
    print(pd.DataFrame(summaries).to_string(index=False))
    print(f"Interactive: {args.out_root / 'diagnostics_interactive.html'}")
    print(f"Summary: {args.out_root / 'summary.csv'}")
    print(f"Report: {args.out_root / 'analysis_report_cn.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
