#!/usr/bin/env python3
"""Plot circle and straight normal-noise two-state VIO trajectories."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path


WORKDIR = Path("/home/admin1/macvo-dev")
if str(WORKDIR) not in sys.path:
    sys.path.insert(0, str(WORKDIR))

from Scripts.plot_static63_gt_macvo import (
    HTML_TEMPLATE,
    metrics,
    read_forward_axes,
    read_xyz,
)


BATCH_ROOT = (
    Path("/mnt/e")
    / "\u6587\u6863"
    / "holoocean/code/recordings/batch_clear_truth_paths_20260713_static63_variants"
)
MACVO_ROOT = (
    WORKDIR
    / "Results/visual_factor_cache_static63_unique_source_20260713"
    / "trial_1/pure_macvo"
)
DEFAULT_RESULT_ROOT = (
    WORKDIR / "Results/circle_straight_normal_noise_two_state_standard_full_20260715"
)
DEFAULT_OUTPUT_DIR = (
    WORKDIR / "analysis_circle_straight_two_state_standard_normal_noise_full_20260715"
)
VARIANT = "vio_two_state_fixed_lag_standard_full"
SCENES = (
    (
        "clear_circle_truth_normal_noise",
        "Circle / Normal noise / Current two-state Standard",
    ),
    (
        "clear_straight_truth_normal_noise",
        "Straight / Normal noise / Current two-state Standard",
    ),
)


Trajectory = list[tuple[int, float, float, float]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def assert_aligned(scene: str, trajectories: dict[str, Trajectory]) -> int:
    lengths = {name: len(rows) for name, rows in trajectories.items()}
    if min(lengths.values(), default=0) == 0:
        raise ValueError(f"{scene}: an input trajectory is empty: {lengths}")
    if len(set(lengths.values())) != 1:
        raise ValueError(f"{scene}: trajectory lengths differ: {lengths}")

    timestamps = {
        name: [row[0] for row in rows] for name, rows in trajectories.items()
    }
    reference_name = next(iter(timestamps))
    reference = timestamps[reference_name]
    for name, values in timestamps.items():
        if values != reference:
            mismatch = next(
                index
                for index, (lhs, rhs) in enumerate(zip(reference, values))
                if lhs != rhs
            )
            raise ValueError(
                f"{scene}: timestamp mismatch at row {mismatch}: "
                f"{reference_name}={reference[mismatch]}, {name}={values[mismatch]}"
            )
    return lengths[reference_name]


def xyz(rows: Trajectory) -> list[list[float]]:
    return [[x, y, z] for _, x, y, z in rows]


def position_errors(gt: Trajectory, estimate: Trajectory, *, xy_only: bool) -> list[float]:
    errors: list[float] = []
    for gt_row, estimate_row in zip(gt, estimate):
        dx = estimate_row[1] - gt_row[1]
        dy = estimate_row[2] - gt_row[2]
        dz = 0.0 if xy_only else estimate_row[3] - gt_row[3]
        errors.append(math.sqrt(dx * dx + dy * dy + dz * dz))
    return errors


def xy_metrics(gt: Trajectory, estimate: Trajectory) -> dict[str, float | int]:
    errors = position_errors(gt, estimate, xy_only=True)
    count = len(errors)
    return {
        "xy_frames": count,
        "xy_rmse_m": (
            math.sqrt(sum(error * error for error in errors) / count)
            if count
            else float("nan")
        ),
        "xy_mean_m": sum(errors) / count if count else float("nan"),
        "xy_max_m": max(errors) if errors else float("nan"),
        "xy_final_m": errors[-1] if errors else float("nan"),
    }


def build_scene(
    scene: str,
    title: str,
    result_root: Path,
) -> tuple[dict, list[dict[str, object]]]:
    gt_path = BATCH_ROOT / scene / "ref_pose.csv"
    macvo_path = MACVO_ROOT / scene / "poses.csv"
    estimate_path = result_root / "trial_1" / VARIANT / scene / "poses.csv"

    trajectories = {
        "GT": read_xyz(gt_path),
        "MACVO": read_xyz(macvo_path),
        "two-state": read_xyz(estimate_path),
    }
    count = assert_aligned(scene, trajectories)
    gt_rows = trajectories["GT"]
    macvo_rows = trajectories["MACVO"]
    estimate_rows = trajectories["two-state"]
    gt_forward = read_forward_axes(gt_path)
    macvo_forward = read_forward_axes(macvo_path)
    estimate_forward = read_forward_axes(estimate_path)
    if not all(len(values) == count for values in (gt_forward, macvo_forward, estimate_forward)):
        raise ValueError(f"{scene}: pose and orientation row counts differ")
    time_zero = gt_rows[0][0]

    payload = {
        "scene": title,
        "gt": xyz(gt_rows),
        "gt_forward": gt_forward,
        "macvo": xyz(macvo_rows),
        "macvo_forward": macvo_forward,
        "time_s": [
            (gt_rows[index][0] - time_zero) / 1e9 for index in range(count)
        ],
        "error_m": position_errors(gt_rows, macvo_rows, xy_only=False),
        "metrics": metrics(gt_rows, macvo_rows),
        "fusion": [
            {
                "key": f"{scene}_two_state_standard",
                "source": "current_two_state_standard",
                "config": "normal_noise",
                "label": "Fusion / Current two-state / Standard",
                "color": "#2563eb",
                "dasharray": "",
                "scene": scene,
                "xyz": xyz(estimate_rows),
                "forward": estimate_forward,
                "error_m": position_errors(gt_rows, estimate_rows, xy_only=False),
                "metrics": metrics(gt_rows, estimate_rows),
                "path": str(estimate_path),
            }
        ],
        "imu_only": [],
        "gt_path": str(gt_path),
        "macvo_path": str(macvo_path),
    }

    summary_rows: list[dict[str, object]] = []
    for method, rows, path in (
        ("pure_macvo", macvo_rows, macvo_path),
        ("current_two_state_standard_normal_noise", estimate_rows, estimate_path),
    ):
        summary_rows.append(
            {
                "scene": scene,
                "method": method,
                **metrics(gt_rows, rows),
                **xy_metrics(gt_rows, rows),
                "estimate_path": str(path),
            }
        )
    return payload, summary_rows


def main() -> None:
    args = parse_args()
    scene_payloads: list[dict] = []
    summary_rows: list[dict[str, object]] = []
    for scene, title in SCENES:
        payload, rows = build_scene(scene, title, args.result_root)
        scene_payloads.append(payload)
        summary_rows.extend(rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "trajectory_summary.csv"
    fieldnames = [
        "scene",
        "method",
        "frames",
        "rmse_m",
        "mean_m",
        "max_m",
        "final_m",
        "xy_frames",
        "xy_rmse_m",
        "xy_mean_m",
        "xy_max_m",
        "xy_final_m",
        "estimate_path",
    ]
    with summary_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    html_template = HTML_TEMPLATE.replace("{{", "{").replace("}}", "}")
    html_template = html_template.replace(
        "Circle, stop-turn rectangle, and straight trajectory comparison",
        "Circle and straight normal-noise trajectory comparison",
    )
    html_template = html_template.replace(
        "__METHOD_SCOPE__",
        "GT, Pure MACVO and current two-state fixed-lag VIO with Standard preintegration",
    )
    html_template = html_template.replace(
        "__LINE_NOTE__",
        "Timestamp-matched NWU trajectories; no alignment, fitting or scale correction. XY is the primary evaluation plane.",
    )
    html = html_template.replace(
        "__DATA__", json.dumps({"scenes": scene_payloads}, ensure_ascii=False)
    )
    html_path = args.output_dir / "interactive_trajectory_gt_vs_est.html"
    html_path.write_text(html, encoding="utf-8")

    print(html_path)
    print(summary_path)


if __name__ == "__main__":
    main()
