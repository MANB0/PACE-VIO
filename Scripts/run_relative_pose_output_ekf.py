#!/usr/bin/env python3
"""Apply a causal, output-only XY+yaw EKF to a completed VIO pose CSV."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation


ROOT = Path("/home/admin1/macvo-dev")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Utility.OutputTrajectoryEKF import (  # noqa: E402
    CausalPoseOutputEKF,
    OutputEKFNoise,
    wrap_angle,
)


INPUT = ROOT / (
    "Results/circle_straight_normal_noise_two_state_standard_full_20260715/"
    "trial_1/vio_two_state_fixed_lag_standard_full/"
    "clear_circle_truth_normal_noise/poses.csv"
)
OUTPUT = ROOT / "Results/circle_relative_pose_output_ekf_short_20260718"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _robust_measurement_std(values: np.ndarray, floor: float) -> np.ndarray:
    second_difference = np.diff(values, n=2, axis=0)
    center = np.median(second_difference, axis=0)
    mad = np.median(np.abs(second_difference - center), axis=0)
    estimate = 1.4826 * mad / np.sqrt(6.0)
    return np.maximum(np.asarray(estimate, dtype=np.float64), float(floor))


def _odd_window(sample_count: int, target: int) -> int:
    window = min(int(target), int(sample_count))
    if window % 2 == 0:
        window -= 1
    return max(window, 5)


def estimate_noise_without_truth(
    frame: pd.DataFrame,
    yaw_unwrapped: np.ndarray,
    *,
    active_from: int,
    calibration_frames: int,
    low_frequency_window_s: float,
    process_quantile: float,
) -> tuple[OutputEKFNoise, dict[str, object]]:
    stop = min(len(frame), int(calibration_frames))
    if stop - active_from < 9:
        raise ValueError("not enough post-initialization samples to calibrate output EKF")
    timestamps = frame["timestamp_ns"].to_numpy(np.int64) * 1.0e-9
    dt = float(np.median(np.diff(timestamps[active_from:stop])))
    xy = frame[["tx", "ty"]].to_numpy(np.float64)[active_from:stop]
    yaw = yaw_unwrapped[active_from:stop]
    position_measurement_std = _robust_measurement_std(xy, 1.0e-6)
    yaw_measurement_std = float(_robust_measurement_std(yaw[:, None], 1.0e-7)[0])

    target_window = int(round(float(low_frequency_window_s) / dt))
    window = _odd_window(len(xy), target_window)
    smoothed_xy = np.column_stack(
        [savgol_filter(xy[:, axis], window, 2) for axis in range(2)]
    )
    smoothed_yaw = savgol_filter(yaw, window, 2)
    apparent_acceleration = np.diff(smoothed_xy, n=2, axis=0) / (dt * dt)
    apparent_yaw_acceleration = np.diff(smoothed_yaw, n=2) / (dt * dt)
    acceleration_process_std = np.maximum(
        np.quantile(np.abs(apparent_acceleration), process_quantile, axis=0),
        1.0e-8,
    )
    yaw_acceleration_process_std = max(
        float(np.quantile(np.abs(apparent_yaw_acceleration), process_quantile)),
        1.0e-8,
    )
    noise = OutputEKFNoise(
        position_measurement_std=position_measurement_std,
        yaw_measurement_std=yaw_measurement_std,
        linear_acceleration_process_std=acceleration_process_std,
        yaw_acceleration_process_std=yaw_acceleration_process_std,
    )
    contract = {
        "truth_used_for_calibration": False,
        "active_from_frame": int(active_from),
        "calibration_stop_frame_exclusive": int(stop),
        "median_dt_s": dt,
        "measurement_noise_estimator": "1.4826*MAD(second_difference)/sqrt(6)",
        "low_frequency_model_estimator": "Savitzky-Golay order 2",
        "low_frequency_window_samples": int(window),
        "low_frequency_window_s_requested": float(low_frequency_window_s),
        "process_quantile": float(process_quantile),
        "position_measurement_std_m": position_measurement_std.tolist(),
        "yaw_measurement_std_rad": yaw_measurement_std,
        "linear_acceleration_process_std_mps2": acceleration_process_std.tolist(),
        "yaw_acceleration_process_std_radps2": yaw_acceleration_process_std,
    }
    return noise, contract


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=INPUT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--seq-to", type=int, default=300)
    parser.add_argument("--active-from", type=int, default=90)
    parser.add_argument("--calibration-frames", type=int, default=300)
    parser.add_argument("--low-frequency-window-s", type=float, default=1.0)
    parser.add_argument("--process-quantile", type=float, default=0.75)
    parser.add_argument("--measurement-std-scale", type=float, default=1.0)
    parser.add_argument("--process-std-scale", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(args.input)
    source = pd.read_csv(args.input).iloc[: int(args.seq_to)].copy()
    if len(source) < 3:
        raise ValueError("input must contain at least three poses")
    timestamps = source["timestamp_ns"].to_numpy(np.int64)
    if (np.diff(timestamps) <= 0).any():
        raise ValueError("pose timestamps must be strictly increasing")
    rotations = Rotation.from_quat(
        source[["qx", "qy", "qz", "qw"]].to_numpy(np.float64)
    )
    yaw_unwrapped = np.unwrap(rotations.as_euler("zyx")[:, 0])
    noise, calibration_contract = estimate_noise_without_truth(
        source,
        yaw_unwrapped,
        active_from=int(args.active_from),
        calibration_frames=int(args.calibration_frames),
        low_frequency_window_s=float(args.low_frequency_window_s),
        process_quantile=float(args.process_quantile),
    )
    if args.measurement_std_scale <= 0.0 or args.process_std_scale <= 0.0:
        raise ValueError("output EKF noise scales must be positive")
    noise = OutputEKFNoise(
        position_measurement_std=(
            noise.position_measurement_std * float(args.measurement_std_scale)
        ),
        yaw_measurement_std=(
            noise.yaw_measurement_std * float(args.measurement_std_scale)
        ),
        linear_acceleration_process_std=(
            noise.linear_acceleration_process_std * float(args.process_std_scale)
        ),
        yaw_acceleration_process_std=(
            noise.yaw_acceleration_process_std * float(args.process_std_scale)
        ),
    )
    calibration_contract["measurement_std_scale"] = float(
        args.measurement_std_scale
    )
    calibration_contract["process_std_scale"] = float(args.process_std_scale)
    calibration_contract["effective_position_measurement_std_m"] = (
        noise.position_measurement_std.tolist()
    )
    calibration_contract["effective_yaw_measurement_std_rad"] = float(
        noise.yaw_measurement_std
    )
    calibration_contract["effective_linear_acceleration_process_std_mps2"] = (
        noise.linear_acceleration_process_std.tolist()
    )
    calibration_contract["effective_yaw_acceleration_process_std_radps2"] = float(
        noise.yaw_acceleration_process_std
    )
    xy = source[["tx", "ty"]].to_numpy(np.float64)
    output_filter = CausalPoseOutputEKF(xy[0], yaw_unwrapped[0], noise)
    filtered_xy = np.zeros_like(xy)
    filtered_yaw = np.zeros(len(source), dtype=np.float64)
    diagnostics: list[dict[str, object]] = []
    for index in range(len(source)):
        dt = None
        if index > 0:
            dt = float(timestamps[index] - timestamps[index - 1]) * 1.0e-9
        step = output_filter.step(dt, xy[index], yaw_unwrapped[index])
        filtered_xy[index] = output_filter.state[0:2]
        filtered_yaw[index] = output_filter.state[4]
        diagnostics.append(
            {
                "frame_idx": index,
                "timestamp_ns": int(timestamps[index]),
                "innovation_x_m": float(step.innovation[0]),
                "innovation_y_m": float(step.innovation[1]),
                "innovation_yaw_rad": float(step.innovation[2]),
                "nis": float(step.nis),
                "correction_x_m": float(step.correction[0]),
                "correction_y_m": float(step.correction[1]),
                "correction_vx_mps": float(step.correction[2]),
                "correction_vy_mps": float(step.correction[3]),
                "correction_yaw_rad": float(step.correction[4]),
                "correction_yaw_rate_radps": float(step.correction[5]),
                "filtered_vx_mps": float(output_filter.state[2]),
                "filtered_vy_mps": float(output_filter.state[3]),
                "filtered_yaw_rate_radps": float(output_filter.state[5]),
                "covariance_min_eigenvalue": float(
                    np.linalg.eigvalsh(output_filter.covariance).min()
                ),
                "covariance_max_eigenvalue": float(
                    np.linalg.eigvalsh(output_filter.covariance).max()
                ),
            }
        )

    result = source.copy()
    result[["tx", "ty"]] = filtered_xy
    yaw_correction = np.asarray(wrap_angle(filtered_yaw - yaw_unwrapped))
    yaw_rotation = Rotation.from_rotvec(
        np.column_stack([np.zeros(len(source)), np.zeros(len(source)), yaw_correction])
    )
    filtered_rotation = yaw_rotation * rotations
    result[["qx", "qy", "qz", "qw"]] = filtered_rotation.as_quat()

    args.output.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output / "poses.csv", index=False)
    with (args.output / "output_ekf_diagnostics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(diagnostics[0]))
        writer.writeheader()
        writer.writerows(diagnostics)
    manifest = {
        "method": "output_only_xy_yaw_causal_ekf",
        "frame_count": int(len(source)),
        "input_pose_csv": str(args.input),
        "input_pose_csv_sha256": _sha256(args.input),
        "truth_used_by_filter": False,
        "feedback_to_vio": False,
        "contract": (
            "Consumes completed relative-pose factor-graph poses only; output must not "
            "be used as VIO state, prior, bias, covariance, or warm start"
        ),
        "state_order": ["x", "y", "vx", "vy", "yaw", "yaw_rate"],
        "calibration": calibration_contract,
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(args.output)


if __name__ == "__main__":
    main()
