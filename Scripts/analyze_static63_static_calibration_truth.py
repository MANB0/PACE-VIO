#!/usr/bin/env python3
"""Compare Static63 startup calibration and corrected IMU data with saved truth."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np


WORKDIR = Path("/home/admin1/macvo-dev")
DATA_ROOT = Path(
    "/mnt/e/\u6587\u6863/holoocean/code/recordings/"
    "batch_clear_truth_paths_20260713_static63_variants"
)
IMU_ONLY_SUMMARY = (
    WORKDIR
    / "Results/static63_calibrated_imu_only_four_configs_20260713/imu_only_summary.csv"
)
FUSION_ROOT = (
    WORKDIR
    / "Results/static63_cached_imu_fusion_staticinit_calibrated_20260713/trial_1"
    / "vio_preintegrated_full_imuatt_staticinit_calibrated"
)
OUTDIR = WORKDIR / "analysis_static63_static_calibration_truth_20260713"

SCENES = {
    "circle": "clear_circle_truth_normal_noise",
    "rectangle": "clear_stop_turn_rectangle_truth_normal_noise",
    "straight": "clear_straight_truth_normal_noise",
}
AXES = ("x", "y", "z")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _load_structured(path: Path) -> np.ndarray:
    return np.genfromtxt(path, delimiter=",", names=True)


def _columns(data: np.ndarray, prefix: str) -> np.ndarray:
    return np.stack([data[f"{prefix}_{axis}"] for axis in AXES], axis=1).astype(np.float64)


def _vector_from_row(row: dict[str, str], prefix: str) -> np.ndarray:
    return np.asarray([float(row[f"{prefix}_{axis}"]) for axis in AXES], dtype=np.float64)


def _static_sample_times(camera_time: np.ndarray, imu_time: np.ndarray, end_ns: int) -> np.ndarray:
    """Reproduce the production frame-interval interpolation time sequence."""
    chunks: list[np.ndarray] = []
    last_time: int | None = None
    for index in range(1, camera_time.size):
        start = max(int(camera_time[index - 1]), int(imu_time[0]))
        end = min(int(camera_time[index]), int(imu_time[-1]))
        if end < start:
            continue
        interior = imu_time[(imu_time > start) & (imu_time < end)]
        interval = np.concatenate(
            [np.asarray([start], dtype=np.int64), interior, np.asarray([end], dtype=np.int64)]
        )
        if last_time is not None:
            interval = interval[interval > last_time]
        if interval.size == 0:
            continue
        chunks.append(interval)
        last_time = int(interval[-1])
        if last_time >= int(end_ns):
            break
    if not chunks:
        raise RuntimeError("No startup-static IMU timestamps were collected")
    times = np.concatenate(chunks)
    end_index = int(np.searchsorted(times, int(end_ns), side="left"))
    return times[: min(end_index + 1, times.size)]


def _interpolate(time_src: np.ndarray, values: np.ndarray, time_dst: np.ndarray) -> np.ndarray:
    output = np.empty((time_dst.size, 3), dtype=np.float64)
    for axis in range(3):
        output[:, axis] = np.interp(time_dst, time_src, values[:, axis])
    return output


def _angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator <= 0.0:
        return math.nan
    cosine = float(np.clip(np.dot(a, b) / denominator, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def _residual_row(
    geometry: str,
    sensor: str,
    correction: str,
    residual: np.ndarray,
    reference_noise_std: np.ndarray,
) -> dict[str, object]:
    mean = residual.mean(axis=0)
    std = residual.std(axis=0)
    axis_rmse = np.sqrt(np.mean(residual**2, axis=0))
    vector_norm = np.linalg.norm(residual, axis=1)
    row: dict[str, object] = {
        "geometry": geometry,
        "sensor": sensor,
        "correction": correction,
        "sample_count": int(residual.shape[0]),
        "vector_rmse": float(np.sqrt(np.mean(vector_norm**2))),
        "mean_vector_norm": float(vector_norm.mean()),
        "final_vector_norm": float(vector_norm[-1]),
    }
    for axis, axis_name in enumerate(AXES):
        row[f"mean_{axis_name}"] = float(mean[axis])
        row[f"std_{axis_name}"] = float(std[axis])
        row[f"rmse_{axis_name}"] = float(axis_rmse[axis])
        row[f"std_over_truth_noise_{axis_name}"] = (
            float(std[axis] / reference_noise_std[axis])
            if reference_noise_std[axis] > 0.0
            else math.nan
        )
    return row


def _fusion_bias_activity(scene: str) -> dict[str, object]:
    rows = _read_csv(FUSION_ROOT / scene / "frame_pair_diagnostics.csv")
    output: dict[str, object] = {"diagnostic_frames": len(rows)}
    for sensor in ("acc", "gyro"):
        states = np.asarray(
            [
                [float(row[f"imu_vio_{sensor}_bias_{axis}"] or 0.0) for axis in AXES]
                for row in rows
            ],
            dtype=np.float64,
        )
        updates = np.asarray(
            [float(row[f"update_{sensor}_bias_norm"] or 0.0) for row in rows],
            dtype=np.float64,
        )
        output[f"{sensor}_bias_state_max_norm"] = float(np.linalg.norm(states, axis=1).max())
        output[f"{sensor}_bias_update_max_norm"] = float(updates.max())
        output[f"{sensor}_bias_final_norm"] = float(np.linalg.norm(states[-1]))
    return output


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    summary_rows = _read_csv(IMU_ONLY_SUMMARY)
    summary_by_scene = {
        row["scene"]: row
        for row in summary_rows
        if row["scene"].endswith("normal_noise")
    }

    static_rows: list[dict[str, object]] = []
    residual_rows: list[dict[str, object]] = []
    fusion_rows: list[dict[str, object]] = []

    for geometry, scene in SCENES.items():
        dataset = DATA_ROOT / scene
        truth = _load_structured(dataset / "imu_truth_decomposition.csv")
        ref = _load_structured(dataset / "ref_pose.csv")
        time = truth["timestamp"].astype(np.int64)
        camera_time = ref["timestamp"].astype(np.int64)
        result = summary_by_scene[scene]
        static_end_ns = int(float(result["static_end_ns"]))
        static_time = _static_sample_times(camera_time, time, static_end_ns)

        arrays = {
            "gyro_true": _columns(truth, "true_ang_vel"),
            "gyro_bias": _columns(truth, "gyro_bias"),
            "gyro_noise": _columns(truth, "gyro_noise"),
            "gyro_measured": _columns(truth, "measured_ang_vel"),
            "acc_true": _columns(truth, "true_lin_acc"),
            "acc_bias": _columns(truth, "acc_bias"),
            "acc_noise": _columns(truth, "acc_noise"),
            "acc_measured": _columns(truth, "measured_lin_acc"),
        }
        estimates = {
            "gyro": _vector_from_row(result, "static_gyro_bias"),
            "acc": _vector_from_row(result, "static_acc_bias"),
        }

        for sensor in ("gyro", "acc"):
            static = {
                name: _interpolate(time, arrays[f"{sensor}_{name}"], static_time)
                for name in ("true", "bias", "noise", "measured")
            }
            true_mean = static["true"].mean(axis=0)
            bias_mean = static["bias"].mean(axis=0)
            noise_mean = static["noise"].mean(axis=0)
            measured_offset_mean = (static["measured"] - static["true"]).mean(axis=0)
            estimate = estimates[sensor]
            estimation_error = estimate - bias_mean
            static_standard_error = static["noise"].std(axis=0) / math.sqrt(static_time.size)
            row: dict[str, object] = {
                "geometry": geometry,
                "sensor": sensor,
                "sample_count": int(static_time.size),
                "duration_s": float((static_time[-1] - static_time[0]) * 1e-9),
                "estimate_error_norm": float(np.linalg.norm(estimation_error)),
                "noise_mean_norm": float(np.linalg.norm(noise_mean)),
                "remaining_offset_mean_norm": float(np.linalg.norm(measured_offset_mean - estimate)),
                "measurement_identity_error_norm": float(
                    np.linalg.norm(measured_offset_mean - bias_mean - noise_mean)
                ),
                "specific_force_direction_error_deg": (
                    _angle_deg(static["measured"].mean(axis=0), true_mean)
                    if sensor == "acc"
                    else math.nan
                ),
            }
            for axis, axis_name in enumerate(AXES):
                row[f"estimate_{axis_name}"] = float(estimate[axis])
                row[f"true_bias_mean_{axis_name}"] = float(bias_mean[axis])
                row[f"noise_mean_{axis_name}"] = float(noise_mean[axis])
                row[f"estimate_error_{axis_name}"] = float(estimation_error[axis])
                row[f"noise_standard_error_{axis_name}"] = float(static_standard_error[axis])
                row[f"error_in_standard_errors_{axis_name}"] = (
                    float(estimation_error[axis] / static_standard_error[axis])
                    if static_standard_error[axis] > 0.0
                    else math.nan
                )
                row[f"remaining_offset_mean_{axis_name}"] = float(
                    measured_offset_mean[axis] - estimate[axis]
                )
            static_rows.append(row)

            post = time > static_end_ns
            true = arrays[f"{sensor}_true"][post]
            bias = arrays[f"{sensor}_bias"][post]
            noise = arrays[f"{sensor}_noise"][post]
            measured = arrays[f"{sensor}_measured"][post]
            reference_noise_std = noise.std(axis=0)
            corrections = {
                "raw_measured": measured - true,
                "minus_static_estimate": measured - estimate.reshape(1, 3) - true,
                "minus_true_bias": measured - bias - true,
                "minus_true_noise": measured - noise - true,
                "truth": np.zeros_like(true),
                "true_bias_minus_static_estimate": bias - estimate.reshape(1, 3),
            }
            for correction, residual in corrections.items():
                residual_rows.append(
                    _residual_row(
                        geometry,
                        sensor,
                        correction,
                        residual,
                        reference_noise_std,
                    )
                )

        fusion_rows.append({"geometry": geometry, "scene": scene, **_fusion_bias_activity(scene)})

    _write_csv(OUTDIR / "static_calibration_vs_truth.csv", static_rows)
    _write_csv(OUTDIR / "post_static_corrected_measurement_residuals.csv", residual_rows)
    _write_csv(OUTDIR / "fusion_bias_activity.csv", fusion_rows)

    compact = {
        "static_calibration": static_rows,
        "post_static_residuals": residual_rows,
        "fusion_bias_activity": fusion_rows,
        "interpretation": {
            "noise_parameters_denoise_samples": False,
            "static_calibration_removes_white_noise": False,
            "fusion_online_bias_state_active_in_this_batch": any(
                float(row["acc_bias_update_max_norm"]) > 0.0
                or float(row["gyro_bias_update_max_norm"]) > 0.0
                for row in fusion_rows
            ),
        },
    }
    (OUTDIR / "summary.json").write_text(
        json.dumps(compact, indent=2, ensure_ascii=False, allow_nan=True),
        encoding="utf-8",
    )
    print(OUTDIR)


if __name__ == "__main__":
    main()
