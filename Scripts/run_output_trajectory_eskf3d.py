#!/usr/bin/env python3
"""Replay a completed pose CSV through the output-only 3D ESKF."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation


ROOT = Path("/home/admin1/macvo-dev")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Utility.OutputTrajectoryESKF3D import (  # noqa: E402
    CausalPoseOutputESKF3D,
    OutputESKF3DNoise,
    OutputESKFGate,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def robust_std(samples: np.ndarray, divisor: float, floor: float) -> np.ndarray:
    values = np.asarray(samples, dtype=np.float64)
    center = np.median(values, axis=0)
    mad = np.median(np.abs(values - center), axis=0)
    return np.maximum(1.4826 * mad / float(divisor), float(floor))


def odd_window(sample_count: int, target: int) -> int:
    window = min(int(target), int(sample_count))
    if window % 2 == 0:
        window -= 1
    return max(window, 5)


def estimate_noise_without_truth(
    frame: pd.DataFrame,
    rotations: Rotation,
    *,
    active_from: int,
    calibration_frames: int,
    low_frequency_window_s: float,
    process_quantile: float,
) -> tuple[OutputESKF3DNoise, dict[str, object]]:
    stop = min(len(frame), int(calibration_frames))
    if stop - active_from < 9:
        raise ValueError("not enough post-initialization poses for calibration")
    timestamps = frame["timestamp_ns"].to_numpy(np.int64) * 1.0e-9
    calibration_time = timestamps[active_from:stop]
    dt = float(np.median(np.diff(calibration_time)))
    position = frame[["tx", "ty", "tz"]].to_numpy(np.float64)[active_from:stop]
    rotation = rotations[active_from:stop]

    position_measurement_std = robust_std(
        np.diff(position, n=2, axis=0), np.sqrt(6.0), 1.0e-6
    )
    rotation_increment = (rotation[:-1].inv() * rotation[1:]).as_rotvec()
    rotation_measurement_std = robust_std(
        np.diff(rotation_increment, axis=0), np.sqrt(6.0), 1.0e-7
    )

    target_window = int(round(float(low_frequency_window_s) / dt))
    position_window = odd_window(len(position), target_window)
    smooth_position = np.column_stack(
        [savgol_filter(position[:, axis], position_window, 2) for axis in range(3)]
    )
    apparent_acceleration = np.diff(smooth_position, n=2, axis=0) / (dt * dt)
    linear_acceleration_process_std = np.maximum(
        np.quantile(np.abs(apparent_acceleration), process_quantile, axis=0),
        1.0e-8,
    )

    angular_velocity = rotation_increment / dt
    angular_window = odd_window(len(angular_velocity), target_window)
    smooth_angular_velocity = np.column_stack(
        [
            savgol_filter(angular_velocity[:, axis], angular_window, 2)
            for axis in range(3)
        ]
    )
    apparent_angular_acceleration = np.diff(smooth_angular_velocity, axis=0) / dt
    angular_acceleration_process_std = np.maximum(
        np.quantile(
            np.abs(apparent_angular_acceleration), process_quantile, axis=0
        ),
        1.0e-8,
    )
    noise = OutputESKF3DNoise(
        position_measurement_std=position_measurement_std,
        rotation_measurement_std=rotation_measurement_std,
        linear_acceleration_process_std=linear_acceleration_process_std,
        angular_acceleration_process_std=angular_acceleration_process_std,
    )
    contract = {
        "truth_used_for_calibration": False,
        "deployable_as_immediate_online_calibration": False,
        "active_from_frame": int(active_from),
        "calibration_stop_frame_exclusive": int(stop),
        "median_dt_s": dt,
        "measurement_noise_estimator": (
            "axis-wise 1.4826*MAD(second difference)/sqrt(6)"
        ),
        "process_noise_estimator": (
            "axis-wise quantile of Savitzky-Golay low-frequency acceleration"
        ),
        "low_frequency_window_s_requested": float(low_frequency_window_s),
        "position_window_samples": int(position_window),
        "angular_velocity_window_samples": int(angular_window),
        "process_quantile": float(process_quantile),
        "base_position_measurement_std_m": position_measurement_std.tolist(),
        "base_rotation_measurement_std_rad": rotation_measurement_std.tolist(),
        "base_linear_acceleration_process_std_mps2": (
            linear_acceleration_process_std.tolist()
        ),
        "base_angular_acceleration_process_std_radps2": (
            angular_acceleration_process_std.tolist()
        ),
    }
    return noise, contract


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seq-to", type=int, default=1890)
    parser.add_argument("--active-from", type=int, default=90)
    parser.add_argument("--calibration-frames", type=int, default=300)
    parser.add_argument("--low-frequency-window-s", type=float, default=1.0)
    parser.add_argument("--process-quantile", type=float, default=0.75)
    parser.add_argument("--measurement-std-scale", type=float, default=4.0)
    parser.add_argument("--process-std-scale", type=float, default=0.25)
    parser.add_argument(
        "--mode", choices=("no_gate", "gate", "gate_adaptive"), required=True
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(args.input)
    if args.measurement_std_scale <= 0.0 or args.process_std_scale <= 0.0:
        raise ValueError("noise scales must be positive")
    source = pd.read_csv(args.input).iloc[: int(args.seq_to)].copy()
    required = {"timestamp_ns", "tx", "ty", "tz", "qx", "qy", "qz", "qw"}
    missing = required.difference(source.columns)
    if missing:
        raise ValueError(f"pose CSV is missing columns: {sorted(missing)}")
    timestamps = source["timestamp_ns"].to_numpy(np.int64)
    if (np.diff(timestamps) <= 0).any():
        raise ValueError("pose timestamps must be strictly increasing")
    measured_position = source[["tx", "ty", "tz"]].to_numpy(np.float64)
    measured_rotation = Rotation.from_quat(
        source[["qx", "qy", "qz", "qw"]].to_numpy(np.float64)
    )
    noise, calibration = estimate_noise_without_truth(
        source,
        measured_rotation,
        active_from=int(args.active_from),
        calibration_frames=int(args.calibration_frames),
        low_frequency_window_s=float(args.low_frequency_window_s),
        process_quantile=float(args.process_quantile),
    )
    noise = OutputESKF3DNoise(
        position_measurement_std=(
            noise.position_measurement_std * float(args.measurement_std_scale)
        ),
        rotation_measurement_std=(
            noise.rotation_measurement_std * float(args.measurement_std_scale)
        ),
        linear_acceleration_process_std=(
            noise.linear_acceleration_process_std * float(args.process_std_scale)
        ),
        angular_acceleration_process_std=(
            noise.angular_acceleration_process_std * float(args.process_std_scale)
        ),
    )
    calibration.update(
        {
            "measurement_std_scale": float(args.measurement_std_scale),
            "process_std_scale": float(args.process_std_scale),
            "effective_position_measurement_std_m": (
                noise.position_measurement_std.tolist()
            ),
            "effective_rotation_measurement_std_rad": (
                noise.rotation_measurement_std.tolist()
            ),
            "effective_linear_acceleration_process_std_mps2": (
                noise.linear_acceleration_process_std.tolist()
            ),
            "effective_angular_acceleration_process_std_radps2": (
                noise.angular_acceleration_process_std.tolist()
            ),
        }
    )

    gate = None if args.mode == "no_gate" else OutputESKFGate()
    output_filter = CausalPoseOutputESKF3D(
        measured_position[0],
        measured_rotation[0].as_matrix(),
        noise,
        gate=gate,
        adaptive_process=args.mode == "gate_adaptive",
    )
    filtered_position = np.zeros_like(measured_position)
    filtered_rotation = np.zeros((len(source), 3, 3), dtype=np.float64)
    diagnostics: list[dict[str, object]] = []
    start = time.perf_counter()
    for index in range(len(source)):
        dt = None
        if index > 0:
            dt = float(timestamps[index] - timestamps[index - 1]) * 1.0e-9
        step = output_filter.step(
            dt,
            measured_position[index],
            measured_rotation[index].as_matrix(),
        )
        filtered_position[index] = output_filter.state.position_W
        filtered_rotation[index] = output_filter.state.rotation_WB
        diagnostics.append(
            {
                "frame_idx": index,
                "timestamp_ns": int(timestamps[index]),
                **{
                    f"position_innovation_{axis}_m": float(
                        step.position_innovation[axis_index]
                    )
                    for axis_index, axis in enumerate("xyz")
                },
                **{
                    f"rotation_innovation_{axis}_rad": float(
                        step.rotation_innovation[axis_index]
                    )
                    for axis_index, axis in enumerate("xyz")
                },
                "position_nis": float(step.position_nis),
                "rotation_nis": float(step.rotation_nis),
                "position_action": step.position_action,
                "rotation_action": step.rotation_action,
                "position_correction_norm_m": float(
                    np.linalg.norm(step.correction[0:3])
                ),
                "rotation_correction_norm_rad": float(
                    np.linalg.norm(step.correction[6:9])
                ),
                **{
                    f"filtered_velocity_{axis}_mps": float(
                        output_filter.state.velocity_W[axis_index]
                    )
                    for axis_index, axis in enumerate("xyz")
                },
                **{
                    f"filtered_angular_velocity_{axis}_radps": float(
                        output_filter.state.angular_velocity_B[axis_index]
                    )
                    for axis_index, axis in enumerate("xyz")
                },
                "process_scale_before": float(step.process_scale_before),
                "process_scale_after": float(step.process_scale_after),
                "covariance_min_eigenvalue": float(
                    step.covariance_min_eigenvalue
                ),
                "covariance_max_eigenvalue": float(
                    step.covariance_max_eigenvalue
                ),
            }
        )
    elapsed = time.perf_counter() - start

    result = source.copy()
    result[["tx", "ty", "tz"]] = filtered_position
    result[["qx", "qy", "qz", "qw"]] = Rotation.from_matrix(
        filtered_rotation
    ).as_quat()
    args.output.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output / "poses.csv", index=False)
    with (args.output / "output_eskf3d_diagnostics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(diagnostics[0]))
        writer.writeheader()
        writer.writerows(diagnostics)

    action_counts = {
        axis: {
            action: sum(row[f"{axis}_action"] == action for row in diagnostics)
            for action in ("accept", "inflate", "reject")
        }
        for axis in ("position", "rotation")
    }
    manifest = {
        "method": "output_only_3d_right_error_eskf",
        "mode": args.mode,
        "frame_count": int(len(source)),
        "input_pose_csv": str(args.input),
        "input_pose_csv_sha256": sha256(args.input),
        "truth_used_by_filter": False,
        "feedback_to_vio": False,
        "source_pose_contract": "T_WC (camera-to-world) from poses.csv",
        "extrinsic_applied_by_filter": False,
        "measurement_fields": ["position_WC", "rotation_WC"],
        "unavailable_offline_measurements": ["velocity_W", "state_covariance"],
        "nominal_state": [
            "position_WC",
            "velocity_WC",
            "rotation_WC",
            "omega_C",
        ],
        "error_state_order": ["dp_W", "dv_W", "dtheta_C", "domega_C"],
        "rotation_update": "R_new = R * Exp(dtheta)",
        "calibration": calibration,
        "gate": (
            None
            if gate is None
            else {
                "inflate_nis": gate.inflate_nis,
                "reject_nis": gate.reject_nis,
                "maximum_inflation": gate.maximum_inflation,
                "position_reject_norm_m": gate.position_reject_norm_m,
                "rotation_reject_norm_rad": gate.rotation_reject_norm_rad,
            }
        ),
        "action_counts": action_counts,
        "runtime": {
            "total_s": elapsed,
            "mean_us_per_frame": elapsed / len(source) * 1.0e6,
        },
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(args.output)
    print(json.dumps(manifest["runtime"]))


if __name__ == "__main__":
    main()
