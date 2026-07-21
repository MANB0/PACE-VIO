#!/usr/bin/env python3
"""Compare current and legacy covariance weighting at GT on Static63 rectangle."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Scripts.validate_static63_imu_truth import (
    DEFAULT_DATA_ROOT,
    _camera_to_imu_internal,
    _columns,
    _imu_velocity_internal,
    _internal_vectors,
    _load_csv,
    _metadata,
    _pose_internal,
    _query_interpolated_interval,
    _truth_arrays,
    preintegrate_imu,
)
from Utility.IMUKinematics import (
    covariance_information_matrix,
    imu_bias_sigma_to_continuous_random_walk_density,
    imu_sigma_to_continuous_density,
    vio_preintegrated_imu_residual,
)


OUTDIR = Path("/home/admin1/macvo-dev/analysis_static63_calibrated_regression_20260713")
DATASET = DEFAULT_DATA_ROOT / "clear_stop_turn_rectangle_truth_no_noise_no_bias"


def phase(ref: np.void, time_s: float) -> str:
    speed = math.sqrt(sum(float(ref[name]) ** 2 for name in ("vx", "vy", "vz")))
    angular = math.sqrt(sum(float(ref[name]) ** 2 for name in ("wx", "wy", "wz")))
    if time_s <= 3.0 + 1e-6:
        return "startup_static"
    if angular >= 0.05:
        return "turn"
    if speed >= 0.02:
        return "straight"
    return "stop_or_transition"


def quantiles(values: list[float]) -> dict[str, float]:
    data = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(data.mean()),
        "median": float(np.median(data)),
        "q95": float(np.quantile(data, 0.95)),
        "max": float(data.max()),
    }


def main() -> None:
    meta = _metadata(DATASET)
    imu_meta = meta["imu"]
    truth = _load_csv(DATASET / "imu_truth_decomposition.csv")
    arrays = _truth_arrays(truth)
    ref = _load_csv(DATASET / "ref_pose.csv")
    timestamps = truth["timestamp"].astype(np.int64)
    rate_hz = float(imu_meta["rate_hz"])
    bias_rate_hz = float(imu_meta.get("bias_random_walk_update_hz", rate_hz))
    sigma_acc = imu_sigma_to_continuous_density(
        imu_meta["AccelSigma"], rate_hz, imu_meta["sigma_unit"]
    )
    sigma_gyro = imu_sigma_to_continuous_density(
        imu_meta["AngVelSigma"], rate_hz, imu_meta["sigma_unit"]
    )
    sigma_acc_w = imu_bias_sigma_to_continuous_random_walk_density(
        imu_meta["AccelBiasSigma"], bias_rate_hz, imu_meta["bias_sigma_unit"]
    )
    sigma_gyro_w = imu_bias_sigma_to_continuous_random_walk_density(
        imu_meta["AngVelBiasSigma"], bias_rate_hz, imu_meta["bias_sigma_unit"]
    )
    lever = _camera_to_imu_internal(meta)
    rows: list[dict[str, float | int | str]] = []

    for frame_index in range(1, len(ref)):
        start = int(ref["timestamp"][frame_index - 1])
        end = int(ref["timestamp"][frame_index])
        interval_time, interval_acc = _query_interpolated_interval(
            timestamps, arrays["true_acc"], start, end
        )
        _, interval_gyro = _query_interpolated_interval(
            timestamps, arrays["true_gyro"], start, end
        )
        pose_i = _pose_internal(ref[frame_index - 1])
        pose_j = _pose_internal(ref[frame_index])
        velocity_i = _imu_velocity_internal(ref[frame_index - 1], pose_i, lever)
        velocity_j = _imu_velocity_internal(ref[frame_index], pose_j, lever)
        preint = preintegrate_imu(
            time_ns=interval_time,
            acc=interval_acc,
            gyro=interval_gyro,
            R0_world=pose_i.rotation(),
            gravity=9.8,
            sigma_acc=sigma_acc,
            sigma_gyro=sigma_gyro,
            sigma_acc_w=sigma_acc_w,
            sigma_gyro_w=sigma_gyro_w,
        )
        residual = vio_preintegrated_imu_residual(
            from_pose=pose_i,
            to_pose=pose_j,
            prev_velocity_world=velocity_i,
            curr_velocity_world=velocity_j,
            delta_R=preint.delta_R,
            delta_v=preint.delta_v,
            delta_p=preint.delta_p,
            dt_total=preint.dt_total,
            sensor_T_imu=lever,
        ).reshape(9).double()
        covariance = preint.cov.double()
        current_information = covariance_information_matrix(covariance)
        legacy_information = covariance_information_matrix(
            covariance,
            absolute_diagonal_floor=1e-8,
        )
        time_s = end * 1e-9
        rows.append(
            {
                "frame": frame_index,
                "timestamp": end,
                "time_s": time_s,
                "phase": phase(ref[frame_index], time_s),
                "position_residual_norm": float(residual[0:3].norm().item()),
                "velocity_residual_norm": float(residual[3:6].norm().item()),
                "rotation_residual_norm": float(residual[6:9].norm().item()),
                "current_nis9": float(residual @ current_information @ residual),
                "legacy_floor_nis9": float(residual @ legacy_information @ residual),
                "current_information_trace": float(current_information.trace().item()),
                "legacy_information_trace": float(legacy_information.trace().item()),
                "covariance_min_eigenvalue": float(torch.linalg.eigvalsh(covariance).min().item()),
                "covariance_max_eigenvalue": float(torch.linalg.eigvalsh(covariance).max().item()),
            }
        )

    OUTDIR.mkdir(parents=True, exist_ok=True)
    output_csv = OUTDIR / "gt_factor_consistency_per_frame.csv"
    with output_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary: dict[str, object] = {
        "pair_count": len(rows),
        "noise_density": {
            "acc": sigma_acc,
            "gyro": sigma_gyro,
            "acc_bias_rw": sigma_acc_w,
            "gyro_bias_rw": sigma_gyro_w,
        },
        "phase": {},
        "largest_current_nis": sorted(rows, key=lambda row: float(row["current_nis9"]), reverse=True)[:12],
    }
    phase_summary = summary["phase"]
    assert isinstance(phase_summary, dict)
    for phase_name in ("startup_static", "straight", "turn", "stop_or_transition"):
        selected = [row for row in rows if row["phase"] == phase_name]
        if not selected:
            continue
        phase_summary[phase_name] = {
            "count": len(selected),
            "position_residual_norm": quantiles([float(row["position_residual_norm"]) for row in selected]),
            "velocity_residual_norm": quantiles([float(row["velocity_residual_norm"]) for row in selected]),
            "rotation_residual_norm": quantiles([float(row["rotation_residual_norm"]) for row in selected]),
            "current_nis9": quantiles([float(row["current_nis9"]) for row in selected]),
            "legacy_floor_nis9": quantiles([float(row["legacy_floor_nis9"]) for row in selected]),
            "information_trace_ratio_median": float(
                np.median(
                    [
                        float(row["current_information_trace"]) / float(row["legacy_information_trace"])
                        for row in selected
                    ]
                )
            ),
        }
    output_json = OUTDIR / "gt_factor_consistency_summary.json"
    output_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(output_csv)
    print(output_json)


if __name__ == "__main__":
    main()
