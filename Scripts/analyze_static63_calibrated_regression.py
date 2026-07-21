#!/usr/bin/env python3
"""Diagnose the Static63 calibrated/static-init regression against saved IMU truth."""

from __future__ import annotations

import csv
import json
import math
import statistics
from pathlib import Path


WORKDIR = Path("/home/admin1/macvo-dev")
DATASET = Path(
    "/mnt/e/文档/holoocean/code/recordings/"
    "batch_clear_truth_paths_20260713_static63_variants/"
    "clear_stop_turn_rectangle_truth_no_noise_no_bias"
)
CURRENT = (
    WORKDIR
    / "Results/static63_cached_imu_fusion_staticinit_calibrated_20260713/trial_1"
    / "vio_preintegrated_full_imuatt_staticinit_calibrated"
    / "clear_stop_turn_rectangle_truth_no_noise_no_bias"
)
PREVIOUS = (
    WORKDIR
    / "Results/static63_cached_imu_fusion_four_configs_20260713/trial_1"
    / "vio_preintegrated_full_imuatt_estinit"
    / "clear_stop_turn_rectangle_truth_no_noise_no_bias"
)
PURE = (
    WORKDIR
    / "Results/visual_factor_cache_static63_unique_source_20260713/trial_1/pure_macvo"
    / "clear_stop_turn_rectangle_truth_normal_noise"
)
VALIDATION = WORKDIR / "analysis_static63_imu_truth_validation_20260713"
OUTDIR = WORKDIR / "analysis_static63_calibrated_regression_20260713"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def number(row: dict[str, str], name: str, default: float = math.nan) -> float:
    try:
        return float(row.get(name, ""))
    except (TypeError, ValueError):
        return default


def vector(row: dict[str, str], names: tuple[str, str, str]) -> tuple[float, float, float]:
    return tuple(number(row, name) for name in names)  # type: ignore[return-value]


def norm3(value: tuple[float, float, float]) -> float:
    return math.sqrt(sum(component * component for component in value))


def distance(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return norm3(tuple(x - y for x, y in zip(a, b)))  # type: ignore[arg-type]


def quat_angle_error(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    an = math.sqrt(sum(x * x for x in a))
    bn = math.sqrt(sum(x * x for x in b))
    if an <= 0.0 or bn <= 0.0:
        return math.nan
    dot = abs(sum(x * y for x, y in zip(a, b)) / (an * bn))
    return 2.0 * math.acos(min(1.0, max(-1.0, dot)))


def finite(values: list[float]) -> list[float]:
    return [value for value in values if math.isfinite(value)]


def median(values: list[float]) -> float:
    selected = finite(values)
    return statistics.median(selected) if selected else math.nan


def rmse(values: list[float]) -> float:
    selected = finite(values)
    return math.sqrt(sum(value * value for value in selected) / len(selected)) if selected else math.nan


def phase_for(gt: dict[str, str], time_s: float) -> str:
    speed = norm3(vector(gt, ("vx", "vy", "vz")))
    angular = norm3(vector(gt, ("wx", "wy", "wz")))
    if time_s <= 3.0 + 1e-6:
        return "startup_static"
    if angular >= 0.05:
        return "turn"
    if speed >= 0.02:
        return "straight"
    return "stop_or_transition"


def first_frame(rows: list[dict[str, object]], predicate) -> dict[str, object] | None:
    return next((row for row in rows if predicate(row)), None)


def diag_summary(diag: dict[int, dict[str, str]], frames: list[int], field: str) -> dict[str, float]:
    values = [number(diag[frame], field) for frame in frames if frame in diag]
    selected = finite(values)
    return {
        "median": statistics.median(selected) if selected else math.nan,
        "max": max(selected) if selected else math.nan,
    }


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    gt = read_csv(DATASET / "ref_pose.csv")
    current = read_csv(CURRENT / "poses.csv")
    previous = read_csv(PREVIOUS / "poses.csv")
    pure = read_csv(PURE / "poses.csv")
    current_diag_rows = read_csv(CURRENT / "frame_pair_diagnostics.csv")
    previous_diag_rows = read_csv(PREVIOUS / "frame_pair_diagnostics.csv")
    current_diag = {int(row["frame_j"]): row for row in current_diag_rows}
    previous_diag = {int(row["frame_j"]): row for row in previous_diag_rows}

    count = min(len(gt), len(current), len(previous), len(pure))
    t0 = int(float(gt[0]["timestamp"]))
    frame_rows: list[dict[str, object]] = []
    for index in range(count):
        gt_row = gt[index]
        current_row = current[index]
        previous_row = previous[index]
        pure_row = pure[index]
        timestamp = int(float(gt_row["timestamp"]))
        time_s = (timestamp - t0) * 1e-9
        gt_position = vector(gt_row, ("x", "y", "z"))
        current_position = vector(current_row, ("tx", "ty", "tz"))
        previous_position = vector(previous_row, ("tx", "ty", "tz"))
        pure_position = vector(pure_row, ("tx", "ty", "tz"))
        gt_quat = tuple(number(gt_row, name) for name in ("qx", "qy", "qz", "qw"))
        current_quat = tuple(number(current_row, name) for name in ("qx", "qy", "qz", "qw"))
        previous_quat = tuple(number(previous_row, name) for name in ("qx", "qy", "qz", "qw"))
        pure_quat = tuple(number(pure_row, name) for name in ("qx", "qy", "qz", "qw"))
        frame_rows.append(
            {
                "frame": index,
                "timestamp": timestamp,
                "time_s": time_s,
                "phase": phase_for(gt_row, time_s),
                "gt_speed": norm3(vector(gt_row, ("vx", "vy", "vz"))),
                "gt_angular_speed": norm3(vector(gt_row, ("wx", "wy", "wz"))),
                "current_position_error": distance(current_position, gt_position),
                "previous_position_error": distance(previous_position, gt_position),
                "pure_position_error": distance(pure_position, gt_position),
                "current_previous_position_distance": distance(current_position, previous_position),
                "current_rotation_error_rad": quat_angle_error(current_quat, gt_quat),
                "previous_rotation_error_rad": quat_angle_error(previous_quat, gt_quat),
                "pure_rotation_error_rad": quat_angle_error(pure_quat, gt_quat),
            }
        )

    frame_fields = list(frame_rows[0])
    with (OUTDIR / "frame_comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=frame_fields)
        writer.writeheader()
        writer.writerows(frame_rows)

    thresholds = {}
    for threshold in (0.05, 0.1, 0.5, 1.0):
        row = first_frame(
            frame_rows,
            lambda item, threshold=threshold: item["time_s"] > 3.0
            and item["current_position_error"] >= threshold,
        )
        thresholds[f"current_error_ge_{threshold:g}m"] = row
    thresholds["current_worse_than_previous_by_0.1m"] = first_frame(
        frame_rows,
        lambda item: item["time_s"] > 3.0
        and item["current_position_error"] - item["previous_position_error"] >= 0.1,
    )
    thresholds["current_previous_distance_ge_0.1m"] = first_frame(
        frame_rows,
        lambda item: item["time_s"] > 3.0
        and item["current_previous_position_distance"] >= 0.1,
    )
    thresholds["first_turn"] = first_frame(
        frame_rows,
        lambda item: item["time_s"] > 3.0 and item["phase"] == "turn",
    )

    event_frames = sorted(
        {
            int(event["frame"])
            for event in thresholds.values()
            if event is not None
        }
    )
    event_fields = [
        "frame",
        "time_s",
        "phase",
        "current_position_error",
        "previous_position_error",
        "current_previous_position_distance",
        "method",
        "r_R_whitened_norm",
        "r_p_whitened_norm",
        "r_v_whitened_norm",
        "imu_vio_weight_trace",
        "imu_vio_weight_diag_min",
        "imu_vio_weight_diag_max",
        "imu_vio_cov_trace",
        "imu_vio_acc_bias_norm",
        "imu_vio_gyro_bias_norm",
        "imu_attitude_source_angle_to_est_rad",
        "est_velocity_error_norm",
        "rotation_error_angle",
        "visual_input_sha256",
    ]
    event_rows: list[dict[str, object]] = []
    for frame in event_frames:
        base = frame_rows[frame]
        for method, diag in (("current", current_diag), ("previous", previous_diag)):
            source = diag.get(frame, {})
            event_rows.append(
                {
                    "frame": frame,
                    "time_s": base["time_s"],
                    "phase": base["phase"],
                    "current_position_error": base["current_position_error"],
                    "previous_position_error": base["previous_position_error"],
                    "current_previous_position_distance": base["current_previous_position_distance"],
                    "method": method,
                    **{field: source.get(field, "") for field in event_fields[7:]},
                }
            )
    with (OUTDIR / "event_diagnostics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=event_fields)
        writer.writeheader()
        writer.writerows(event_rows)

    truth = read_csv(DATASET / "imu_truth_decomposition.csv")
    imu_t0 = int(float(truth[0]["timestamp"]))
    static_truth = [
        row for row in truth
        if (int(float(row["timestamp"])) - imu_t0) * 1e-9 <= 3.0 + 1e-9
    ]
    static_stats = {}
    for prefix in ("true_ang_vel", "gyro_bias", "gyro_noise", "measured_ang_vel", "true_lin_acc", "acc_bias", "acc_noise", "measured_lin_acc"):
        axes = []
        stds = []
        for axis in ("x", "y", "z"):
            values = [number(row, f"{prefix}_{axis}") for row in static_truth]
            axes.append(sum(values) / len(values))
            stds.append(statistics.pstdev(values))
        static_stats[prefix] = {"mean": axes, "std": stds}

    calibration_rows = read_csv(VALIDATION / "static_initialization.csv")
    calibration = next(
        row for row in calibration_rows
        if row["geometry"] == "stop_turn_rectangle" and row["variant"] == "no_noise_no_bias"
    )
    oracle_rows = read_csv(VALIDATION / "preintegration_oracle_summary.csv")
    truth_oracle = next(
        row for row in oracle_rows
        if row["geometry"] == "stop_turn_rectangle" and row["oracle"] == "truth"
    )

    phase_summary = {}
    for phase in sorted({str(row["phase"]) for row in frame_rows}):
        selected = [row for row in frame_rows if row["phase"] == phase]
        frames = [int(row["frame"]) for row in selected]
        phase_summary[phase] = {
            "frames": len(selected),
            "current_position_rmse": rmse([float(row["current_position_error"]) for row in selected]),
            "previous_position_rmse": rmse([float(row["previous_position_error"]) for row in selected]),
            "pure_position_rmse": rmse([float(row["pure_position_error"]) for row in selected]),
            "current_rotation_rmse_rad": rmse([float(row["current_rotation_error_rad"]) for row in selected]),
            "previous_rotation_rmse_rad": rmse([float(row["previous_rotation_error_rad"]) for row in selected]),
            "current_weight_trace": diag_summary(current_diag, frames, "imu_vio_weight_trace"),
            "previous_weight_trace": diag_summary(previous_diag, frames, "imu_vio_weight_trace"),
            "current_r_p_whitened": diag_summary(current_diag, frames, "r_p_whitened_norm"),
            "previous_r_p_whitened": diag_summary(previous_diag, frames, "r_p_whitened_norm"),
            "current_r_v_whitened": diag_summary(current_diag, frames, "r_v_whitened_norm"),
            "previous_r_v_whitened": diag_summary(previous_diag, frames, "r_v_whitened_norm"),
            "current_r_R_whitened": diag_summary(current_diag, frames, "r_R_whitened_norm"),
            "previous_r_R_whitened": diag_summary(previous_diag, frames, "r_R_whitened_norm"),
        }

    shared_frames = sorted(set(current_diag) & set(previous_diag))
    hash_mismatch = sum(
        current_diag[frame].get("visual_input_sha256")
        != previous_diag[frame].get("visual_input_sha256")
        for frame in shared_frames
    )
    summary = {
        "frame_count": count,
        "static_calibration": calibration,
        "static_truth_statistics": static_stats,
        "truth_preintegration_oracle": truth_oracle,
        "events": thresholds,
        "phase_summary": phase_summary,
        "visual_hash_overlap_frames": len(shared_frames),
        "visual_hash_mismatch_frames": hash_mismatch,
    }
    (OUTDIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=True),
        encoding="utf-8",
    )

    print(json.dumps({
        "static_calibration": calibration,
        "truth_preintegration_oracle": truth_oracle,
        "events": thresholds,
        "phase_summary": phase_summary,
        "visual_hash_overlap_frames": len(shared_frames),
        "visual_hash_mismatch_frames": hash_mismatch,
    }, ensure_ascii=False, indent=2, allow_nan=True))
    print(OUTDIR / "frame_comparison.csv")
    print(OUTDIR / "event_diagnostics.csv")
    print(OUTDIR / "summary.json")


if __name__ == "__main__":
    main()
