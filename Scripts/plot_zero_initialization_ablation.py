#!/usr/bin/env python3
"""Plot the circle static-initialization ablation at the IMU center."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation


DATASET_DEFAULT = Path(
    "/mnt/e/文档/holoocean/code/recordings/"
    "batch_clear_truth_paths_20260713_static63_variants/"
    "clear_circle_truth_normal_noise"
)
BASELINE_DEFAULT = Path(
    "/home/admin1/macvo-dev/Results/"
    "normal_noise_compressed_uvd_t2_full_three_scenes_20260720/"
    "trial_1/vio_two_state_compressed_uvd_t2_full/"
    "clear_circle_truth_normal_noise/poses_imu.csv"
)
ZERO_ROOT_DEFAULT = Path(
    "/home/admin1/macvo-realtime-t2-minimal/Results/"
    "zero_init_circle_full_final_20260721/clear_circle_truth_normal_noise"
)
OUTPUT_DEFAULT = Path(
    "/home/admin1/macvo-realtime-t2-minimal/"
    "analysis_circle_zero_initialization_ablation_20260721"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DATASET_DEFAULT)
    parser.add_argument("--baseline", type=Path, default=BASELINE_DEFAULT)
    parser.add_argument("--zero-root", type=Path, default=ZERO_ROOT_DEFAULT)
    parser.add_argument("--output", type=Path, default=OUTPUT_DEFAULT)
    parser.add_argument("--active-from", type=int, default=90)
    return parser.parse_args()


def unique_file(root: Path, name: str) -> Path:
    matches = list(root.rglob(name))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one {name} under {root}, found {matches}")
    return matches[0]


def load_poses(path: Path) -> tuple[np.ndarray, np.ndarray]:
    frame = pd.read_csv(path)
    timestamp = "timestamp_ns" if "timestamp_ns" in frame else "timestamp"
    position = ["tx", "ty", "tz"] if "tx" in frame else ["x", "y", "z"]
    poses = np.column_stack(
        [
            frame[position].to_numpy(np.float64),
            frame[["qx", "qy", "qz", "qw"]].to_numpy(np.float64),
        ]
    )
    if not np.isfinite(poses).all():
        raise FloatingPointError(f"NaN/Inf in {path}")
    return frame[timestamp].to_numpy(np.int64), poses


def translate_reference(poses: np.ndarray, offset_body: np.ndarray) -> np.ndarray:
    result = poses.copy()
    result[:, :3] += Rotation.from_quat(poses[:, 3:7]).apply(offset_body)
    return result


def rebase_position(poses: np.ndarray) -> np.ndarray:
    result = poses.copy()
    result[:, :3] -= result[0, :3]
    return result


def relative_pose_errors(gt: np.ndarray, estimate: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gt_rot = Rotation.from_quat(gt[:, 3:7])
    est_rot = Rotation.from_quat(estimate[:, 3:7])
    gt_dt = gt_rot[:-1].inv().apply(gt[1:, :3] - gt[:-1, :3])
    est_dt = est_rot[:-1].inv().apply(estimate[1:, :3] - estimate[:-1, :3])
    translation = np.linalg.norm(est_dt - gt_dt, axis=1)
    gt_dr = gt_rot[:-1].inv() * gt_rot[1:]
    est_dr = est_rot[:-1].inv() * est_rot[1:]
    rotation = (gt_dr.inv() * est_dr).magnitude()
    return translation, rotation


def metrics(gt: np.ndarray, estimate: np.ndarray, active_from: int) -> dict[str, float | int]:
    error = estimate[:, :3] - gt[:, :3]
    xy = np.linalg.norm(error[:, :2], axis=1)
    xyz = np.linalg.norm(error, axis=1)
    orientation = (
        Rotation.from_quat(gt[:, 3:7]).inv()
        * Rotation.from_quat(estimate[:, 3:7])
    ).magnitude()
    rpe_t, rpe_r = relative_pose_errors(gt, estimate)
    active = slice(active_from, None)
    active_edge = slice(max(active_from - 1, 0), None)
    return {
        "frame_count": len(gt),
        "active_from_frame": active_from,
        "xy_ate_rmse_m": float(np.sqrt(np.mean(xy[active] ** 2))),
        "xyz_ate_rmse_m": float(np.sqrt(np.mean(xyz[active] ** 2))),
        "x_rmse_m": float(np.sqrt(np.mean(error[active, 0] ** 2))),
        "y_rmse_m": float(np.sqrt(np.mean(error[active, 1] ** 2))),
        "z_rmse_m": float(np.sqrt(np.mean(error[active, 2] ** 2))),
        "orientation_rmse_deg": float(
            np.degrees(np.sqrt(np.mean(orientation[active] ** 2)))
        ),
        "translation_rpe_rmse_m": float(
            np.sqrt(np.mean(rpe_t[active_edge] ** 2))
        ),
        "rotation_rpe_rmse_deg": float(
            np.degrees(np.sqrt(np.mean(rpe_r[active_edge] ** 2)))
        ),
        "xy_max_m": float(np.max(xy[active])),
        "xyz_max_m": float(np.max(xyz[active])),
    }


def bias_summary(path: Path) -> dict[str, object]:
    frame = pd.read_csv(path)
    columns = [
        "imu_vio_acc_bias_x",
        "imu_vio_acc_bias_y",
        "imu_vio_acc_bias_z",
        "imu_vio_gyro_bias_x",
        "imu_vio_gyro_bias_y",
        "imu_vio_gyro_bias_z",
    ]
    values = frame[columns].apply(pd.to_numeric, errors="coerce").to_numpy(np.float64)
    valid = values[np.isfinite(values).all(axis=1)]
    if valid.size == 0:
        return {"sample_count": 0}
    return {
        "sample_count": int(len(valid)),
        "first": dict(zip(columns, valid[0].tolist())),
        "last": dict(zip(columns, valid[-1].tolist())),
        "max_abs": dict(zip(columns, np.max(np.abs(valid), axis=0).tolist())),
        "rmse": dict(zip(columns, np.sqrt(np.mean(valid**2, axis=0)).tolist())),
    }


def main() -> int:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    zero_pose_path = unique_file(args.zero_root, "poses_imu.csv")
    zero_diag_path = unique_file(args.zero_root, "frame_pair_diagnostics.csv")
    static_path = args.zero_root / "static_initialization.json"

    metadata = json.loads((args.dataset / "metadata.json").read_text(encoding="utf-8"))
    matrix_CI = np.asarray(metadata["extrinsics"]["T_CI"], dtype=np.float64)
    if matrix_CI.shape != (4, 4):
        raise ValueError("metadata.extrinsics.T_CI must be 4x4")
    body_to_imu = np.diag([1.0, -1.0, -1.0]) @ matrix_CI[:3, 3]
    gt_time, gt_body = load_poses(args.dataset / "ref_pose.csv")
    baseline_time, baseline = load_poses(args.baseline)
    zero_time, zero = load_poses(zero_pose_path)
    if not (
        np.array_equal(gt_time, baseline_time)
        and np.array_equal(gt_time, zero_time)
    ):
        raise AssertionError("GT, baseline and zero-initialization timestamps differ")

    gt = rebase_position(translate_reference(gt_body, body_to_imu))
    baseline = rebase_position(baseline)
    zero = rebase_position(zero)
    trajectories = {
        "GT / IMU center": gt,
        "T2 / fixed 3 s / estimated state": baseline,
        "T2 / adaptive 2.1 s / zero state": zero,
    }
    colors = ["#111827", "#2563eb", "#dc2626"]
    metric_rows = []
    for name, trajectory in list(trajectories.items())[1:]:
        metric_rows.append({"method": name, **metrics(gt, trajectory, args.active_from)})
    metric_frame = pd.DataFrame(metric_rows)
    metric_frame.to_csv(args.output / "metrics.csv", index=False, float_format="%.12g")

    time_s = (gt_time - gt_time[0]) * 1.0e-9
    figure, axes = plt.subplots(2, 2, figsize=(14, 11), constrained_layout=True)
    views = ((0, 1, "XY"), (0, 2, "XZ"), (1, 2, "YZ"))
    for (name, trajectory), color in zip(trajectories.items(), colors):
        width = 2.6 if name.startswith("GT") else 1.5
        for axis, (x_index, y_index, title) in zip(axes.flat[:3], views):
            axis.plot(
                trajectory[:, x_index],
                trajectory[:, y_index],
                color=color,
                linewidth=width,
                label=name,
            )
            axis.set_title(title)
            axis.set_aspect("equal", adjustable="datalim")
            axis.grid(alpha=0.25)
        if not name.startswith("GT"):
            position_error = np.linalg.norm(trajectory[:, :3] - gt[:, :3], axis=1)
            axes[1, 1].plot(
                time_s,
                position_error,
                color=color,
                linewidth=width,
                label=f"{name} error",
            )
    axes[0, 0].set_xlabel("x / m (NWU)")
    axes[0, 0].set_ylabel("y / m (NWU)")
    axes[0, 1].set_xlabel("x / m (NWU)")
    axes[0, 1].set_ylabel("z / m (NWU)")
    axes[1, 0].set_xlabel("y / m (NWU)")
    axes[1, 0].set_ylabel("z / m (NWU)")
    axes[1, 1].set_title("3D position error")
    axes[1, 1].set_xlabel("time / s")
    axes[1, 1].set_ylabel("error / m")
    axes[1, 1].grid(alpha=0.25)
    axes[1, 1].legend(fontsize=8)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=3)
    figure.suptitle(
        "Circle normal-noise: static estimate applied vs forced-zero VIO state"
    )
    png_name = "zero_initialization_ablation.png"
    figure.savefig(args.output / png_name, dpi=180, bbox_inches="tight")
    plt.close(figure)
    table_html = metric_frame.to_html(index=False, float_format=lambda value: f"{value:.6g}")
    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Circle zero initialization ablation</title>
<style>body{{font-family:Arial,sans-serif;margin:24px;color:#18212f}} table{{border-collapse:collapse;width:100%;margin-top:20px}} th,td{{border:1px solid #d8dee8;padding:8px;text-align:right}} th:first-child,td:first-child{{text-align:left}} .note{{color:#526071;line-height:1.5}}</style></head>
<body><h1>Circle normal-noise initialization ablation</h1>
<p class="note">All trajectories are expressed at the IMU center in world NWU and independently translation-rebased at frame 0. No rotation, scale, SE(3), or trajectory fitting is applied. Metrics use the common segment from frame {args.active_from}. The blue baseline uses the trusted cached T2 full run with fixed 3 s estimated initialization; the red run uses live MACVO with adaptive completion at 2.1 s and deliberately applies zero attitude, velocity, accelerometer bias, and gyro bias.</p>
<img src="{png_name}" alt="trajectory comparison" style="width:100%;max-width:1500px"><h2>Metrics</h2>{table_html}</body></html>"""
    (args.output / "interactive_zero_initialization_ablation.html").write_text(
        html, encoding="utf-8"
    )

    static_diag = json.loads(static_path.read_text(encoding="utf-8"))
    bias = {
        "baseline_fixed_estimated": bias_summary(
            args.baseline.parent / "frame_pair_diagnostics.csv"
        ),
        "adaptive_zero": bias_summary(zero_diag_path),
    }
    (args.output / "zero_initialization_bias_summary.json").write_text(
        json.dumps(bias, indent=2), encoding="utf-8"
    )
    contract = {
        "dataset": str(args.dataset),
        "world_frame": "NWU",
        "trajectory_reference": "IMU center",
        "rebase": "translation-only at frame 0",
        "alignment": "none",
        "common_metric_start_frame": args.active_from,
        "baseline": {
            "path": str(args.baseline),
            "static_mode": "fixed",
            "duration_s": 3.0,
            "state_policy": "estimated",
            "visual_source": "trusted cached full T2 run",
        },
        "zero_initialization": {
            "path": str(zero_pose_path),
            "static_diagnostics": static_diag,
            "visual_source": "live MACVO stereo frontend",
        },
        "comparison_limit": (
            "The runs share dataset, model family and T2 backend but do not reuse "
            "an identical point-level frontend realization or initialization boundary."
        ),
    }
    (args.output / "experiment_contract.json").write_text(
        json.dumps(contract, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(metric_frame.to_string(index=False))
    print(json.dumps(bias, indent=2))
    print(args.output / "interactive_zero_initialization_ablation.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
