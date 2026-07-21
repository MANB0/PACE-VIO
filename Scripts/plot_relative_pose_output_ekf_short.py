#!/usr/bin/env python3
"""Compare the relative-pose factor graph with its output-only EKF trajectory."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path("/home/admin1/macvo-dev")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Scripts.plot_circle_direct_uvd_u1_vs_pose_factor import (  # noqa: E402
    BATCH_ROOT,
    HTML_TEMPLATE,
    MACVO_POSE,
    POSE_FACTOR_POSE,
    SCENE,
    metrics,
    position_errors,
    read_forward_axes,
    read_xyz,
    xy_metrics,
    xyz,
)
from Scripts.plot_relative_pose_clone_eskf_short import (  # noqa: E402
    _relative_metrics,
    _xy_error_smoothness,
)


FILTERED = ROOT / "Results/circle_relative_pose_output_ekf_short_20260718/poses.csv"
OUTPUT = ROOT / "analysis_circle_relative_pose_output_ekf_short_20260718"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--filtered", type=Path, default=FILTERED)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument(
        "--filtered-label",
        default="Factor graph + output-only XY/yaw EKF",
    )
    parser.add_argument("--comparison-filtered", type=Path)
    parser.add_argument(
        "--comparison-label",
        default="Factor graph + output-only XY/yaw EKF (reference)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    gt_path = BATCH_ROOT / SCENE / "ref_pose.csv"
    paths = {
        "pure_macvo": MACVO_POSE,
        "pose_factor": POSE_FACTOR_POSE,
        "output_ekf": args.filtered,
    }
    if args.comparison_filtered is not None:
        paths["output_ekf_reference"] = args.comparison_filtered
    for path in (gt_path, *paths.values()):
        if not path.exists():
            raise FileNotFoundError(path)
    count = len(pd.read_csv(args.filtered))
    trajectories = {"GT": read_xyz(gt_path)[:count]}
    trajectories.update({key: read_xyz(path)[:count] for key, path in paths.items()})
    forwards = {"GT": read_forward_axes(gt_path)[:count]}
    forwards.update({key: read_forward_axes(path)[:count] for key, path in paths.items()})
    timestamps = [row[0] for row in trajectories["GT"]]
    for key, rows in trajectories.items():
        if [row[0] for row in rows] != timestamps:
            raise AssertionError(f"timestamp mismatch for {key}")

    labels = {
        "pose_factor": "Relative pose T_ij factor graph",
        "output_ekf": args.filtered_label,
        "output_ekf_reference": args.comparison_label,
    }
    colors = {
        "pose_factor": "#f97316",
        "output_ekf": "#16a34a",
        "output_ekf_reference": "#2563eb",
    }
    metric_payload = {}
    for key in paths:
        metric_payload[key] = {
            **metrics(trajectories["GT"], trajectories[key]),
            **xy_metrics(trajectories["GT"], trajectories[key]),
            **_relative_metrics(gt_path, paths[key], count),
            **_xy_error_smoothness(gt_path, paths[key], count, active_from=90),
        }

    payload = {
        "scene": f"Circle / Normal noise / First {count} frames",
        "gt": xyz(trajectories["GT"]),
        "gt_forward": forwards["GT"],
        "macvo": xyz(trajectories["pure_macvo"]),
        "macvo_forward": forwards["pure_macvo"],
        "time_s": [(value - timestamps[0]) * 1.0e-9 for value in timestamps],
        "error_m": position_errors(
            trajectories["GT"], trajectories["pure_macvo"], xy_only=False
        ),
        "metrics": metrics(trajectories["GT"], trajectories["pure_macvo"]),
        "fusion": [
            {
                "key": key,
                "source": key,
                "config": "normal_noise",
                "label": labels[key],
                "color": colors[key],
                "dasharray": "",
                "scene": SCENE,
                "xyz": xyz(trajectories[key]),
                "forward": forwards[key],
                "error_m": position_errors(
                    trajectories["GT"], trajectories[key], xy_only=False
                ),
                "metrics": metrics(trajectories["GT"], trajectories[key]),
                "path": str(paths[key]),
            }
            for key in paths
            if key != "pure_macvo"
        ],
        "imu_only": [],
        "gt_path": str(gt_path),
        "macvo_path": str(MACVO_POSE),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    range_name = "full" if count > 300 else "short"
    template = HTML_TEMPLATE.replace("{{", "{").replace("}}", "}")
    template = template.replace(
        "Circle, stop-turn rectangle, and straight trajectory comparison",
        "Relative-pose factor graph with output-only EKF",
    )
    template = template.replace(
        "__METHOD_SCOPE__",
        "GT, Pure MACVO, Relative-pose factor graph, and its output-only EKF",
    )
    template = template.replace(
        "__LINE_NOTE__",
        "Timestamp-matched NWU; output EKF never feeds back into VIO. XY is primary.",
    )
    (args.output / f"interactive_relative_pose_output_ekf_{range_name}.html").write_text(
        template.replace("__DATA__", json.dumps({"scenes": [payload]}, ensure_ascii=False)),
        encoding="utf-8",
    )
    (args.output / "metrics.json").write_text(
        json.dumps(metric_payload, indent=2), encoding="utf-8"
    )

    figure, axis = plt.subplots(figsize=(10, 7))
    axis.plot(
        [row[1] for row in trajectories["GT"]],
        [row[2] for row in trajectories["GT"]],
        color="#111827",
        linewidth=2.5,
        label="GT",
    )
    for key in paths:
        if key == "pure_macvo":
            continue
        axis.plot(
            [row[1] for row in trajectories[key]],
            [row[2] for row in trajectories[key]],
            color=colors[key],
            linewidth=1.8,
            label=labels[key],
        )
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("x / m (NWU)")
    axis.set_ylabel("y / m (NWU)")
    axis.grid(True, alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(
        args.output / f"relative_pose_output_ekf_{range_name}_xy.png", dpi=180
    )
    plt.close(figure)
    print(args.output)


if __name__ == "__main__":
    main()
