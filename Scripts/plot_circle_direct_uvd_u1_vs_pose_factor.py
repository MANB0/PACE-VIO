#!/usr/bin/env python3
"""Compare the full-circle direct-UVD U1 run with the frozen T_ij pose factor."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

WORKDIR = Path("/home/admin1/macvo-dev")
if str(WORKDIR) not in sys.path:
    sys.path.insert(0, str(WORKDIR))

from Scripts.plot_static63_gt_macvo import (
    HTML_TEMPLATE,
    metrics,
    read_forward_axes,
    read_xyz,
)
SCENE = "clear_circle_truth_normal_noise"
BATCH_ROOT = (
    Path("/mnt/e")
    / "\u6587\u6863"
    / "holoocean/code/recordings/batch_clear_truth_paths_20260713_static63_variants"
)
MACVO_POSE = (
    WORKDIR
    / "Results/visual_factor_cache_static63_unique_source_20260713"
    / "trial_1/pure_macvo"
    / SCENE
    / "poses.csv"
)
POSE_FACTOR_POSE = (
    WORKDIR
    / "Results/circle_straight_normal_noise_two_state_standard_full_20260715"
    / "trial_1/vio_two_state_fixed_lag_standard_full"
    / SCENE
    / "poses.csv"
)
DEFAULT_U1_ROOT = WORKDIR / "Results/circle_normal_noise_direct_uvd_u1_full_20260716"
DEFAULT_OUTPUT = WORKDIR / "analysis_circle_direct_uvd_u1_vs_pose_factor_full_20260716"
U1_VARIANT = "vio_two_state_direct_uvd_u1_standard_full"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--u1-result-root", type=Path, default=DEFAULT_U1_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def xyz(rows):
    return [[x, y, z] for _, x, y, z in rows]


def position_errors(gt, estimate, *, xy_only: bool) -> list[float]:
    result = []
    for expected, actual in zip(gt, estimate):
        delta = np.asarray(actual[1:4], dtype=np.float64) - np.asarray(
            expected[1:4], dtype=np.float64
        )
        if xy_only:
            delta[2] = 0.0
        result.append(float(np.linalg.norm(delta)))
    return result


def xy_metrics(gt, estimate) -> dict[str, float | int]:
    error = np.asarray(position_errors(gt, estimate, xy_only=True))
    return {
        "xy_frames": int(error.size),
        "xy_rmse_m": float(np.sqrt(np.mean(error * error))),
        "xy_mean_m": float(np.mean(error)),
        "xy_max_m": float(np.max(error)),
        "xy_final_m": float(error[-1]),
    }


def pose_matrices(path: Path) -> tuple[np.ndarray, np.ndarray]:
    frame = pd.read_csv(path)
    rotation = Rotation.from_quat(frame[["qx", "qy", "qz", "qw"]].to_numpy())
    matrices = np.tile(np.eye(4), (len(frame), 1, 1))
    matrices[:, :3, :3] = rotation.as_matrix()
    position_columns = (
        ["tx", "ty", "tz"] if "tx" in frame.columns else ["x", "y", "z"]
    )
    matrices[:, :3, 3] = frame[position_columns].to_numpy(np.float64)
    timestamp_column = "timestamp_ns" if "timestamp_ns" in frame.columns else "timestamp"
    return frame[timestamp_column].to_numpy(np.int64), matrices


def relative_metrics(gt_path: Path, estimate_path: Path) -> dict[str, float]:
    gt_time, gt = pose_matrices(gt_path)
    est_time, estimate = pose_matrices(estimate_path)
    if not np.array_equal(gt_time, est_time):
        raise AssertionError(f"timestamp mismatch: {estimate_path}")
    gt_rel = np.linalg.inv(gt[:-1]) @ gt[1:]
    est_rel = np.linalg.inv(estimate[:-1]) @ estimate[1:]
    error = np.linalg.inv(gt_rel) @ est_rel
    translation = np.linalg.norm(error[:, :3, 3], axis=1)
    rotation = Rotation.from_matrix(error[:, :3, :3]).magnitude()
    return {
        "translation_rpe_rmse_m": float(np.sqrt(np.mean(translation * translation))),
        "rotation_rpe_rmse_rad": float(np.sqrt(np.mean(rotation * rotation))),
    }


def main() -> None:
    args = parse_args()
    gt_path = BATCH_ROOT / SCENE / "ref_pose.csv"
    u1_path = (
        args.u1_result_root / "trial_1" / U1_VARIANT / SCENE / "poses.csv"
    )
    paths = {
        "pure_macvo": MACVO_POSE,
        "pose_factor_Tij": POSE_FACTOR_POSE,
        "direct_uvd_U1": u1_path,
    }
    for path in (gt_path, *paths.values()):
        if not path.exists():
            raise FileNotFoundError(path)

    trajectories = {"GT": read_xyz(gt_path)}
    trajectories.update({name: read_xyz(path) for name, path in paths.items()})
    lengths = {name: len(rows) for name, rows in trajectories.items()}
    if len(set(lengths.values())) != 1:
        raise AssertionError(f"trajectory lengths differ: {lengths}")
    timestamps = {
        name: [row[0] for row in rows] for name, rows in trajectories.items()
    }
    reference_time = timestamps["GT"]
    for name, values in timestamps.items():
        if values != reference_time:
            raise AssertionError(f"timestamp mismatch: GT vs {name}")

    forwards = {"GT": read_forward_axes(gt_path)}
    forwards.update({name: read_forward_axes(path) for name, path in paths.items()})
    gt = trajectories["GT"]
    time_zero = gt[0][0]
    fusion_specs = (
        (
            "pose_factor_Tij",
            "Fusion · Relative pose T_ij factor",
            "#dc2626",
            "6D_Tij",
        ),
        (
            "direct_uvd_U1",
            "Fusion · Direct UVD factor · U1",
            "#2563eb",
            "direct_UVD",
        ),
    )
    payload = {
        "scene": "Circle / Normal noise / Full 63 s",
        "gt": xyz(gt),
        "gt_forward": forwards["GT"],
        "macvo": xyz(trajectories["pure_macvo"]),
        "macvo_forward": forwards["pure_macvo"],
        "time_s": [(timestamp - time_zero) / 1e9 for timestamp in reference_time],
        "error_m": position_errors(gt, trajectories["pure_macvo"], xy_only=False),
        "metrics": metrics(gt, trajectories["pure_macvo"]),
        "fusion": [
            {
                "key": key,
                "source": source,
                "config": "normal_noise",
                "label": label,
                "color": color,
                "dasharray": "8 5" if key == "pose_factor_Tij" else "",
                "scene": SCENE,
                "xyz": xyz(trajectories[key]),
                "forward": forwards[key],
                "error_m": position_errors(gt, trajectories[key], xy_only=False),
                "metrics": metrics(gt, trajectories[key]),
                "path": str(paths[key]),
            }
            for key, label, color, source in fusion_specs
        ],
        "imu_only": [],
        "gt_path": str(gt_path),
        "macvo_path": str(MACVO_POSE),
    }

    summary_rows = []
    for method, path in paths.items():
        rows = trajectories[method]
        summary_rows.append(
            {
                "method": method,
                **metrics(gt, rows),
                **xy_metrics(gt, rows),
                **relative_metrics(gt_path, path),
                "estimate_path": str(path),
            }
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "trajectory_metrics.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)

    html_template = HTML_TEMPLATE.replace("{{", "{").replace("}}", "}")
    html_template = html_template.replace(
        "Circle, stop-turn rectangle, and straight trajectory comparison",
        "Full-circle direct-UVD U1 vs relative-pose T_ij factor",
    )
    html_template = html_template.replace(
        "__METHOD_SCOPE__",
        "GT, Pure MACVO, original 6D T_ij pose factor and direct-UVD U1",
    )
    html_template = html_template.replace(
        "__LINE_NOTE__",
        "Timestamp-matched NWU trajectories; no alignment, fitting or scale correction. XY is the primary evaluation plane.",
    )
    html = html_template.replace(
        "__DATA__", json.dumps({"scenes": [payload]}, ensure_ascii=False)
    )
    html_path = args.output_dir / "interactive_u1_vs_pose_factor.html"
    html_path.write_text(html, encoding="utf-8")

    figure, axis = plt.subplots(figsize=(12, 7.5), constrained_layout=True)
    plot_specs = (
        ("GT", "GT", "#202833", "-", 2.8),
        ("pure_macvo", "Pure MACVO", "#f97316", "-", 1.7),
        ("pose_factor_Tij", "Relative pose T_ij factor", "#dc2626", "--", 1.8),
        ("direct_uvd_U1", "Direct UVD U1", "#2563eb", "-", 2.2),
    )
    for key, label, color, linestyle, width in plot_specs:
        rows = trajectories[key]
        values = np.asarray([[row[1], row[2]] for row in rows], dtype=np.float64)
        axis.plot(
            values[:, 0], values[:, 1], label=label, color=color,
            linestyle=linestyle, linewidth=width,
        )
        forward = np.asarray(forwards[key], dtype=np.float64)
        arrow_index = np.arange(90, len(values), 180)
        axis.quiver(
            values[arrow_index, 0], values[arrow_index, 1],
            forward[arrow_index, 0], forward[arrow_index, 1],
            color=color, angles="xy", scale_units="xy", scale=5.0,
            width=0.003, headwidth=4.5, headlength=5.5,
        )
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("x / m (NWU)")
    axis.set_ylabel("y / m (NWU)")
    axis.set_title("Full circle normal-noise: direct UVD U1 vs relative-pose factor")
    axis.grid(True, color="#dbe2ea", linewidth=0.8)
    axis.legend(loc="best")
    static_path = args.output_dir / "trajectory_xy_comparison.png"
    figure.savefig(static_path, dpi=180)
    plt.close(figure)
    print(html_path)
    print(summary_path)
    print(static_path)


if __name__ == "__main__":
    main()
