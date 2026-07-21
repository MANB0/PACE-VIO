#!/usr/bin/env python3
"""Compare full-history pure MACVO with a pure-MACVO restart at 3 seconds."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Scripts.plot_static63_gt_macvo import HTML_TEMPLATE, metrics  # noqa: E402


SCENE = "clear_circle_truth_normal_noise"
SOURCE_START = 90
DATASET = Path(
    "/mnt/e/文档/holoocean/code/recordings/"
    "batch_clear_truth_paths_20260713_static63_variants/clear_circle_truth_normal_noise"
)
FULL_POSES = ROOT / (
    "Results/visual_factor_cache_static63_unique_source_20260713/trial_1/"
    "pure_macvo/clear_circle_truth_normal_noise/poses.csv"
)
POST3_POSES = ROOT / (
    "Results/circle_post3_pure_macvo_cache_20260717/source/"
    "clear_circle_truth_normal_noise/poses.csv"
)
GT_POSES = DATASET / "ref_pose.csv"
OUTPUT = ROOT / "analysis_circle_post3_pure_macvo_comparison_20260717"


def read_matrices(path: Path) -> tuple[np.ndarray, np.ndarray]:
    frame = pd.read_csv(path)
    timestamp_name = "timestamp_ns" if "timestamp_ns" in frame else "timestamp"
    position_names = ["tx", "ty", "tz"] if "tx" in frame else ["x", "y", "z"]
    timestamps = frame[timestamp_name].to_numpy(np.int64)
    matrices = np.repeat(np.eye(4, dtype=np.float64)[None], len(frame), axis=0)
    matrices[:, :3, :3] = Rotation.from_quat(
        frame[["qx", "qy", "qz", "qw"]].to_numpy(np.float64)
    ).as_matrix()
    matrices[:, :3, 3] = frame[position_names].to_numpy(np.float64)
    return timestamps, matrices


def rows(timestamps: np.ndarray, poses: np.ndarray) -> list[tuple[int, float, float, float]]:
    return [
        (int(timestamp), float(pose[0, 3]), float(pose[1, 3]), float(pose[2, 3]))
        for timestamp, pose in zip(timestamps, poses)
    ]


def forward_axes(poses: np.ndarray) -> list[list[float]]:
    return poses[:, :3, 0].tolist()


def xyz(poses: np.ndarray) -> list[list[float]]:
    return poses[:, :3, 3].tolist()


def errors(reference: np.ndarray, estimate: np.ndarray, *, xy_only: bool) -> np.ndarray:
    delta = estimate[:, :3, 3] - reference[:, :3, 3]
    if xy_only:
        delta = delta.copy()
        delta[:, 2] = 0.0
    return np.linalg.norm(delta, axis=1)


def error_summary(reference: np.ndarray, estimate: np.ndarray, *, xy_only: bool) -> dict[str, float]:
    value = errors(reference, estimate, xy_only=xy_only)
    return {
        "rmse_m": float(np.sqrt(np.mean(value * value))),
        "mean_m": float(np.mean(value)),
        "p95_m": float(np.quantile(value, 0.95)),
        "max_m": float(np.max(value)),
        "final_m": float(value[-1]),
    }


def relative_summary(reference: np.ndarray, estimate: np.ndarray) -> dict[str, float]:
    reference_delta = np.linalg.inv(reference[:-1]) @ reference[1:]
    estimate_delta = np.linalg.inv(estimate[:-1]) @ estimate[1:]
    error = np.linalg.inv(reference_delta) @ estimate_delta
    translation = np.linalg.norm(error[:, :3, 3], axis=1)
    rotation = Rotation.from_matrix(error[:, :3, :3]).magnitude()
    return {
        "translation_rpe_rmse_m": float(np.sqrt(np.mean(translation * translation))),
        "translation_rpe_p95_m": float(np.quantile(translation, 0.95)),
        "rotation_rpe_rmse_rad": float(np.sqrt(np.mean(rotation * rotation))),
        "rotation_rpe_p95_rad": float(np.quantile(rotation, 0.95)),
    }


def write_metrics(
    timestamps: np.ndarray,
    gt_common: np.ndarray,
    old_common: np.ndarray,
    restart_common: np.ndarray,
) -> Path:
    output = OUTPUT / "trajectory_metrics.csv"
    records = []
    for method, estimate in (
        ("MACVO 0-63s, suffix after 3s", old_common),
        ("MACVO restarted at 3s", restart_common),
    ):
        record: dict[str, object] = {
            "method": method,
            "frames": len(timestamps),
            "timestamp_start_ns": int(timestamps[0]),
            "timestamp_end_ns": int(timestamps[-1]),
        }
        record.update({f"xyz_{key}": value for key, value in error_summary(gt_common, estimate, xy_only=False).items()})
        record.update({f"xy_{key}": value for key, value in error_summary(gt_common, estimate, xy_only=True).items()})
        record.update(relative_summary(gt_common, estimate))
        records.append(record)
    pairwise = {
        "method": "restart minus full-history MACVO",
        "frames": len(timestamps),
        "timestamp_start_ns": int(timestamps[0]),
        "timestamp_end_ns": int(timestamps[-1]),
    }
    pairwise.update(
        {f"xyz_{key}": value for key, value in error_summary(old_common, restart_common, xy_only=False).items()}
    )
    pairwise.update(
        {f"xy_{key}": value for key, value in error_summary(old_common, restart_common, xy_only=True).items()}
    )
    pairwise.update(relative_summary(old_common, restart_common))
    records.append(pairwise)
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    return output


def write_html(
    timestamps: np.ndarray,
    gt_common: np.ndarray,
    old_common: np.ndarray,
    restart_common: np.ndarray,
) -> Path:
    gt_rows = rows(timestamps, gt_common)
    old_rows = rows(timestamps, old_common)
    restart_rows = rows(timestamps, restart_common)
    time_s = ((timestamps - timestamps[0]) * 1e-9).tolist()
    payload = {
        "scene": "Circle / Normal noise / Shared interval 3.003-62.97 s",
        "gt": xyz(gt_common),
        "gt_forward": forward_axes(gt_common),
        "macvo": xyz(old_common),
        "macvo_forward": forward_axes(old_common),
        "time_s": time_s,
        "error_m": errors(gt_common, old_common, xy_only=False).tolist(),
        "metrics": metrics(gt_rows, old_rows),
        "fusion": [
            {
                "key": "post3_restart",
                "source": "pure_macvo_restart",
                "config": "normal_noise",
                "label": "Pure MACVO restarted at 3 s (no earlier visual history)",
                "color": "#2563eb",
                "dasharray": "",
                "scene": SCENE,
                "xyz": xyz(restart_common),
                "forward": forward_axes(restart_common),
                "error_m": errors(gt_common, restart_common, xy_only=False).tolist(),
                "metrics": metrics(gt_rows, restart_rows),
                "path": str(POST3_POSES),
            }
        ],
        "imu_only": [],
        "gt_path": str(GT_POSES),
        "macvo_path": str(FULL_POSES),
    }
    template = HTML_TEMPLATE.replace("{{", "{").replace("}}", "}")
    template = template.replace(
        "Circle, stop-turn rectangle, and straight trajectory comparison",
        "Pure MACVO: full-history run vs restart at 3 seconds",
    )
    template = template.replace(
        "__METHOD_SCOPE__",
        "Pure MACVO 0-63 s and a fresh pure-MACVO run started at source frame 90",
    )
    template = template.replace(
        "__LINE_NOTE__",
        (
            "Both curves share the old run's pose at 3.003 s as their display anchor. "
            "No GT fitting, scale correction or post-hoc trajectory alignment is used."
        ),
    )
    template = template.replace("Fusion · ${item.label}", "${item.label}")
    template = template.replace(
        "Pure MACVO</span>",
        "Pure MACVO 0-63 s (shown from 3 s; retains earlier history)</span>",
    )
    template = template.replace(
        '<label>IMU data\n        <select data-config-filter aria-label="Filter trajectories by IMU data configuration">\n'
        '          <option value="all">All configurations</option>\n'
        '          <option value="normal_noise">Normal noise + bias</option>\n'
        '          <option value="bias_no_noise">Bias only</option>\n'
        '          <option value="noise_no_bias">White noise only</option>\n'
        '          <option value="no_noise_no_bias">No bias / no noise</option>\n'
        "        </select>\n      </label>",
        "",
    )
    html = template.replace("__DATA__", json.dumps({"scenes": [payload]}, ensure_ascii=False))
    output = OUTPUT / "interactive_post3_restart_vs_full_history.html"
    output.write_text(html, encoding="utf-8")
    return output


def write_png(gt_common: np.ndarray, old_common: np.ndarray, restart_common: np.ndarray) -> Path:
    figure, axis = plt.subplots(figsize=(12.5, 8), constrained_layout=True)
    specifications = (
        (gt_common, "GT motion after 3 s", "#111827", "-", 3.0),
        (old_common, "Pure MACVO 0-63 s (retains pre-3 s history)", "#f97316", "-", 2.0),
        (restart_common, "Pure MACVO restarted at 3 s", "#2563eb", "-", 2.2),
    )
    arrow_indices = np.arange(0, len(gt_common), 180)
    for poses, label, color, linestyle, width in specifications:
        position = poses[:, :3, 3]
        forward = poses[:, :3, 0]
        axis.plot(position[:, 0], position[:, 1], label=label, color=color, linestyle=linestyle, linewidth=width)
        axis.quiver(
            position[arrow_indices, 0], position[arrow_indices, 1],
            forward[arrow_indices, 0], forward[arrow_indices, 1],
            color=color, angles="xy", scale_units="xy", scale=5.0,
            width=0.003, headwidth=4.5, headlength=5.5,
        )
    anchor = old_common[0, :3, 3]
    axis.scatter(anchor[0], anchor[1], color="#111827", s=55, marker="o", zorder=8)
    axis.annotate("shared anchor at 3.003 s", (anchor[0], anchor[1]), xytext=(8, 10), textcoords="offset points")
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("x / m (NWU)")
    axis.set_ylabel("y / m (NWU)")
    axis.set_title("Pure MACVO after 3 s: retained history vs fresh restart")
    axis.grid(True, color="#dbe2ea", linewidth=0.8)
    axis.legend(loc="best")
    output = OUTPUT / "trajectory_xy_post3_restart_vs_full_history.png"
    figure.savefig(output, dpi=180)
    plt.close(figure)
    return output


def main() -> None:
    for path in (FULL_POSES, POST3_POSES, GT_POSES):
        if not path.exists():
            raise FileNotFoundError(path)
    full_time, full = read_matrices(FULL_POSES)
    post3_time, post3 = read_matrices(POST3_POSES)
    gt_time, gt = read_matrices(GT_POSES)
    if len(full) != 1890 or len(post3) != 1800 or len(gt) != 1890:
        raise AssertionError(f"unexpected lengths: full={len(full)}, post3={len(post3)}, gt={len(gt)}")
    shared_time = full_time[SOURCE_START:]
    if not np.array_equal(shared_time, post3_time) or not np.array_equal(shared_time, gt_time[SOURCE_START:]):
        raise AssertionError("the two MACVO results and GT are not timestamp matched after frame 90")
    if not np.allclose(post3[0], np.eye(4), atol=1e-10):
        raise AssertionError("post-3s MACVO result does not start at identity")

    # Display all post-3s trajectories in the old run's world frame at frame 90.
    anchor = full[SOURCE_START]
    old_common = full[SOURCE_START:]
    restart_common = anchor[None] @ post3
    gt_relative = np.linalg.inv(gt[SOURCE_START])[None] @ gt[SOURCE_START:]
    gt_common = anchor[None] @ gt_relative

    OUTPUT.mkdir(parents=True, exist_ok=True)
    metrics_path = write_metrics(shared_time, gt_common, old_common, restart_common)
    html_path = write_html(shared_time, gt_common, old_common, restart_common)
    png_path = write_png(gt_common, old_common, restart_common)
    print(html_path)
    print(metrics_path)
    print(png_path)


if __name__ == "__main__":
    main()
