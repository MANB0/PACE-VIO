#!/usr/bin/env python3
"""Audit the late SA-v2 trajectory outliers without rerunning optimization.

The two-state backend writes both states. Frame k is first written as state_j
for edge (k-1, k), then written again as state_i for edge (k, k+1). The final
pose of frame k-1 and the edge-k relative translation therefore reconstruct
the first write of frame k. Comparing it with the final pose of frame k
measures the next-edge rewrite directly.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation


ROOT = Path("/home/admin1/macvo-dev")
DEFAULT_SA2 = ROOT / (
    "Results/circle_direct_uvd_sampling_aware_v2_full_20260717/"
    "sampling_aware_v2/trial_1/"
    "vio_two_state_direct_uvd_sampling_aware_v2_full/"
    "clear_circle_truth_normal_noise"
)
DEFAULT_U1 = ROOT / (
    "Results/circle_normal_noise_direct_uvd_u1_full_20260716/trial_1/"
    "vio_two_state_direct_uvd_u1_standard_full/"
    "clear_circle_truth_normal_noise"
)
DEFAULT_OUTPUT = ROOT / "analysis_circle_sa_v2_outlier_6167_20260717"


def load_rows(run_dir: Path) -> tuple[np.ndarray, dict[int, dict[str, str]]]:
    poses = np.genfromtxt(run_dir / "poses.csv", delimiter=",", names=True)
    with (run_dir / "frame_pair_diagnostics.csv").open(newline="") as handle:
        diagnostics = {int(row["frame_j"]): row for row in csv.DictReader(handle)}
    return poses, diagnostics


def finite_float(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def audit_run(run_dir: Path, method: str) -> list[dict[str, float | int | str]]:
    poses, diagnostics = load_rows(run_dir)
    positions = np.column_stack([poses["tx"], poses["ty"], poses["tz"]])
    quaternions = np.column_stack([poses["qx"], poses["qy"], poses["qz"], poses["qw"]])
    timestamps = poses["timestamp_ns"].astype(np.int64)
    tensor_path = run_dir / "tensor_map.npz"
    tensor_map = np.load(tensor_path, allow_pickle=False) if tensor_path.exists() else None

    records: list[dict[str, float | int | str]] = []
    for frame_j in sorted(diagnostics):
        row = diagnostics[frame_j]
        frame_i = int(row["frame_i"])
        if frame_j != frame_i + 1 or frame_j >= len(positions) - 1:
            continue

        relative_t = np.array(
            [finite_float(row, "est_delta_x"), finite_float(row, "est_delta_y"), finite_float(row, "est_delta_z")]
        )
        first_as_j = positions[frame_i] + Rotation.from_quat(quaternions[frame_i]).apply(relative_t)
        rewrite = positions[frame_j] - first_as_j
        saved_step = positions[frame_j] - positions[frame_i]

        incoming_count = outgoing_count = 0
        unique_min = unique_max = total_min = total_max = float("nan")
        if tensor_map is not None and "frames//imu_vio_sa_v2_unique_cov" in tensor_map:
            incoming_count = int(tensor_map["frames//imu_vio_sa_v2_incoming_count"][frame_j])
            outgoing_count = int(tensor_map["frames//imu_vio_sa_v2_outgoing_count"][frame_j])
            unique_cov = tensor_map["frames//imu_vio_sa_v2_unique_cov"][frame_j].astype(np.float64)
            total_cov = tensor_map["frames//imu_vio_cov"][frame_j].astype(np.float64)
            unique_eigenvalues = np.linalg.eigvalsh(0.5 * (unique_cov + unique_cov.T))
            total_eigenvalues = np.linalg.eigvalsh(0.5 * (total_cov + total_cov.T))
            unique_min, unique_max = float(unique_eigenvalues[0]), float(unique_eigenvalues[-1])
            total_min, total_max = float(total_eigenvalues[0]), float(total_eigenvalues[-1])

        records.append(
            {
                "method": method,
                "frame_i": frame_i,
                "frame_j": frame_j,
                "timestamp_ns": int(timestamps[frame_j]),
                "time_from_start_s": float((timestamps[frame_j] - timestamps[0]) * 1e-9),
                "saved_step_x_m": float(saved_step[0]),
                "saved_step_y_m": float(saved_step[1]),
                "saved_step_z_m": float(saved_step[2]),
                "saved_step_xy_norm_m": float(np.linalg.norm(saved_step[:2])),
                "reported_relative_t_norm_m": finite_float(row, "est_delta_t_norm"),
                "reported_relative_R_angle_rad": finite_float(row, "est_delta_R_angle"),
                "first_as_j_x_m": float(first_as_j[0]),
                "first_as_j_y_m": float(first_as_j[1]),
                "first_as_j_z_m": float(first_as_j[2]),
                "final_after_next_edge_x_m": float(positions[frame_j, 0]),
                "final_after_next_edge_y_m": float(positions[frame_j, 1]),
                "final_after_next_edge_z_m": float(positions[frame_j, 2]),
                "rewrite_x_m": float(rewrite[0]),
                "rewrite_y_m": float(rewrite[1]),
                "rewrite_z_m": float(rewrite[2]),
                "rewrite_xy_norm_m": float(np.linalg.norm(rewrite[:2])),
                "rewrite_3d_norm_m": float(np.linalg.norm(rewrite)),
                "sampling_noise_cost": finite_float(row, "imu_vio_sa_v2_sampling_noise_cost"),
                "imu_whitened_norm": finite_float(row, "imu_vio_whitened_norm"),
                "visual_loss_per_residual": finite_float(row, "visual_loss_per_residual"),
                "num_valid_points": int(float(row.get("num_valid_points", 0) or 0)),
                "incoming_endpoint_samples": incoming_count,
                "outgoing_endpoint_samples": outgoing_count,
                "unique_cov_min_eigenvalue": unique_min,
                "unique_cov_max_eigenvalue": unique_max,
                "total_cov_min_eigenvalue": total_min,
                "total_cov_max_eigenvalue": total_max,
            }
        )
    if tensor_map is not None:
        tensor_map.close()
    return records


def write_csv(path: Path, records: list[dict[str, float | int | str]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def percentile_summary(records: list[dict[str, float | int | str]], start: float, end: float) -> dict[str, float | int]:
    values = np.asarray(
        [float(row["rewrite_xy_norm_m"]) for row in records if start <= float(row["time_from_start_s"]) < end]
    )
    return {
        "count": int(values.size),
        "median_m": float(np.median(values)),
        "p95_m": float(np.percentile(values, 95)),
        "max_m": float(values.max()),
        "count_over_0p1_m": int(np.count_nonzero(values > 0.1)),
    }


def plot(sa2: list[dict[str, float | int | str]], u1: list[dict[str, float | int | str]], output: Path) -> None:
    def col(records: list[dict[str, float | int | str]], key: str) -> np.ndarray:
        return np.asarray([float(row[key]) for row in records])

    tsa, tu1 = col(sa2, "time_from_start_s"), col(u1, "time_from_start_s")
    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True)
    axes[0].plot(tsa, col(sa2, "saved_step_xy_norm_m"), label="SA-v2 saved XY frame step", color="#0b9b70")
    axes[0].plot(tsa, col(sa2, "reported_relative_t_norm_m"), label="SA-v2 edge relative translation", color="#34495e")
    axes[0].set_ylabel("m / frame")
    axes[0].set_yscale("log")
    axes[0].legend(loc="upper left")

    axes[1].plot(tsa, col(sa2, "rewrite_xy_norm_m"), label="SA-v2 next-edge rewrite", color="#0b9b70")
    axes[1].plot(tu1, col(u1, "rewrite_xy_norm_m"), label="Current U1 next-edge rewrite", color="#2878d0", alpha=0.85)
    axes[1].set_ylabel("rewrite XY (m)")
    axes[1].set_yscale("log")
    axes[1].legend(loc="upper left")

    axes[2].plot(tsa, col(sa2, "sampling_noise_cost"), label="SA-v2 sampling latent cost", color="#d35400")
    axes[2].set_ylabel("cost")
    axes[2].set_xlabel("time from sequence start (s)")
    axes[2].legend(loc="upper left")

    for axis in axes:
        axis.axvspan(59.803333333, 63.0, color="#7f8c8d", alpha=0.1, label="terminal static")
        axis.axvline(61.666666667, color="#c0392b", linestyle="--", linewidth=1.5)
        axis.grid(True, alpha=0.25)
        axis.set_xlim(59.5, 63.0)
    fig.suptitle("SA-v2 late outlier: saved trajectory vs per-edge solution and next-edge rewrite")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sa2", type=Path, default=DEFAULT_SA2)
    parser.add_argument("--u1", type=Path, default=DEFAULT_U1)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    sa2 = audit_run(args.sa2, "SA-v2")
    u1 = audit_run(args.u1, "Current U1")
    write_csv(args.output / "sa_v2_frame_rewrite_audit.csv", sa2)
    write_csv(args.output / "u1_frame_rewrite_audit.csv", u1)
    neighborhood = [row for row in sa2 if 61.3 <= float(row["time_from_start_s"]) <= 62.7]
    write_csv(args.output / "sa_v2_outlier_neighborhood.csv", neighborhood)
    plot(sa2, u1, args.output / "sa_v2_late_outlier_diagnostic.png")

    top = sorted(sa2, key=lambda row: float(row["rewrite_xy_norm_m"]), reverse=True)[:20]
    summary = {
        "method": "SA-v2",
        "definition": "rewrite = final pose after frame is state_i - reconstructed first pose when it was state_j",
        "terminal_static_start_s": 59.803333333,
        "segments": {
            "moving_and_decelerating_3_to_59p8_s": percentile_summary(sa2, 3.0, 59.8),
            "early_terminal_static_59p8_to_61p3_s": percentile_summary(sa2, 59.8, 61.3),
            "late_terminal_static_61p3_to_63_s": percentile_summary(sa2, 61.3, 63.0),
            "u1_late_terminal_static_61p3_to_63_s": percentile_summary(u1, 61.3, 63.0),
        },
        "top_rewrites": top,
        "code_contract": {
            "writeback": "all_two_state",
            "unique_covariance_whitening": "eigenvalue clamp followed by Cholesky",
            "covariance_floor": 1e-12,
            "float64_relative_floor_when_scale_is_one": float(np.finfo(np.float64).eps),
        },
    }
    with (args.output / "sa_v2_outlier_summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)

    print(args.output)
    print(json.dumps(summary["segments"], indent=2))


if __name__ == "__main__":
    main()
