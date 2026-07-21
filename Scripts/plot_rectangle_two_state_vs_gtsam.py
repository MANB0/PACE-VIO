#!/usr/bin/env python3
"""Compare current two-state VIO and GTSAM VIO on rectangle IMU variants."""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

WORKDIR = Path("/home/admin1/macvo-dev")
GTSAM_ROOT = Path("/home/admin1/gtsam/macvo_vio/results")
DATA_ROOT = (
    Path("/mnt/e/文档/holoocean/code/recordings")
    / "batch_clear_truth_paths_20260713_static63_variants"
)
if str(WORKDIR) not in sys.path:
    sys.path.insert(0, str(WORKDIR))

from Scripts.plot_static63_gt_macvo import HTML_TEMPLATE, metrics, read_xyz


MACVO_PATH = (
    WORKDIR
    / "Results/visual_factor_cache_static63_unique_source_20260713/trial_1/pure_macvo"
    / "clear_stop_turn_rectangle_truth_normal_noise/poses.csv"
)
OUTDIR = WORKDIR / "analysis_rectangle_two_state_vs_gtsam_20260715"

CASES = [
    {
        "key": "normal_noise",
        "title": "Stop-turn rectangle / Normal noise / Full 63 s",
        "scene": "clear_stop_turn_rectangle_truth_normal_noise",
        "current": (
            WORKDIR
            / "Results/rectangle_normal_noise_two_state_standard_full_20260715"
            / "trial_1/vio_two_state_fixed_lag_standard_full"
            / "clear_stop_turn_rectangle_truth_normal_noise/poses.csv"
        ),
        "gtsam": GTSAM_ROOT / "rectangle_normal_noise/gtsam_vio_estimate.csv",
    },
    {
        "key": "bias_only_no_noise",
        "title": "Stop-turn rectangle / Bias only, no noise / Full 63 s",
        "scene": "clear_stop_turn_rectangle_truth_bias_no_noise",
        "current": (
            WORKDIR
            / "Results/rectangle_bias_no_noise_two_state_standard_full_20260715"
            / "trial_1/vio_two_state_fixed_lag_standard_full"
            / "clear_stop_turn_rectangle_truth_bias_no_noise/poses.csv"
        ),
        "gtsam": GTSAM_ROOT / "rectangle_bias_only/gtsam_vio_estimate.csv",
    },
]


def read_gtsam_camera_xyz(path: Path) -> list[tuple[int, float, float, float]]:
    rows: list[tuple[int, float, float, float]] = []
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            rows.append(
                (
                    int(row["timestamp_ns"]),
                    float(row["est_c_x"]),
                    float(row["est_c_y"]),
                    float(row["est_c_z"]),
                )
            )
    return rows


def position_errors(
    gt: list[tuple[int, float, float, float]],
    est: list[tuple[int, float, float, float]],
) -> list[float]:
    return [
        math.sqrt(
            (estimate[1] - truth[1]) ** 2
            + (estimate[2] - truth[2]) ** 2
            + (estimate[3] - truth[3]) ** 2
        )
        for truth, estimate in zip(gt, est, strict=True)
    ]


def validate_rows(
    name: str,
    gt: list[tuple[int, float, float, float]],
    rows: list[tuple[int, float, float, float]],
) -> None:
    if len(rows) != len(gt):
        raise ValueError(f"{name} frame count differs: GT={len(gt)}, estimate={len(rows)}")
    mismatches = [index for index, (truth, estimate) in enumerate(zip(gt, rows)) if truth[0] != estimate[0]]
    if mismatches:
        first = mismatches[0]
        raise ValueError(
            f"{name} timestamp mismatch at frame {first}: GT={gt[first][0]}, estimate={rows[first][0]}"
        )


def main() -> None:
    macvo_rows = read_xyz(MACVO_PATH)
    scenes: list[dict] = []
    summary_rows: list[dict[str, object]] = []

    for case in CASES:
        gt_path = DATA_ROOT / str(case["scene"]) / "ref_pose.csv"
        gt_rows = read_xyz(gt_path)
        current_rows = read_xyz(Path(case["current"]))
        gtsam_rows = read_gtsam_camera_xyz(Path(case["gtsam"]))
        validate_rows("Pure MACVO", gt_rows, macvo_rows)
        validate_rows("Current two-state", gt_rows, current_rows)
        validate_rows("GTSAM VIO", gt_rows, gtsam_rows)

        time_zero = gt_rows[0][0]
        scenes.append(
            {
                "scene": case["title"],
                "gt": [[x, y, z] for _, x, y, z in gt_rows],
                "macvo": [[x, y, z] for _, x, y, z in macvo_rows],
                "time_s": [(row[0] - time_zero) / 1e9 for row in gt_rows],
                "error_m": position_errors(gt_rows, macvo_rows),
                "metrics": metrics(gt_rows, macvo_rows),
                "fusion": [
                    {
                        "key": f"current_{case['key']}",
                        "source": "current_two_state_fixed_lag",
                        "config": case["key"],
                        "label": "Current two-state / Standard",
                        "color": "#2563eb",
                        "dasharray": "",
                        "scene": case["scene"],
                        "xyz": [[x, y, z] for _, x, y, z in current_rows],
                        "error_m": position_errors(gt_rows, current_rows),
                        "metrics": metrics(gt_rows, current_rows),
                        "path": str(case["current"]),
                    },
                    {
                        "key": f"gtsam_{case['key']}",
                        "source": "gtsam_isam2",
                        "config": case["key"],
                        "label": "GTSAM VIO / iSAM2",
                        "color": "#dc2626",
                        "dasharray": "7 4",
                        "scene": case["scene"],
                        "xyz": [[x, y, z] for _, x, y, z in gtsam_rows],
                        "error_m": position_errors(gt_rows, gtsam_rows),
                        "metrics": metrics(gt_rows, gtsam_rows),
                        "path": str(case["gtsam"]),
                    },
                ],
                "imu_only": [],
                "gt_path": str(gt_path),
                "macvo_path": str(MACVO_PATH),
            }
        )

        for method, rows, path in (
            ("pure_macvo", macvo_rows, MACVO_PATH),
            ("current_two_state_standard", current_rows, case["current"]),
            ("gtsam_isam2", gtsam_rows, case["gtsam"]),
        ):
            summary_rows.append(
                {
                    "imu_config": case["key"],
                    "method": method,
                    **metrics(gt_rows, rows),
                    "estimate_path": path,
                }
            )

    OUTDIR.mkdir(parents=True, exist_ok=True)
    summary_path = OUTDIR / "trajectory_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "imu_config",
                "method",
                "frames",
                "rmse_m",
                "mean_m",
                "max_m",
                "final_m",
                "estimate_path",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    html_template = HTML_TEMPLATE.replace("{{", "{").replace("}}", "}")
    html_template = html_template.replace(
        "Circle, stop-turn rectangle, and straight trajectory comparison",
        "Current two-state VIO vs GTSAM VIO",
    )
    html_template = html_template.replace(
        "__METHOD_SCOPE__",
        "GT, Pure MACVO, current two-state fixed-lag and GTSAM iSAM2",
    )
    html_template = html_template.replace(
        "__LINE_NOTE__",
        "Camera trajectories are timestamp-matched in NWU with equal XYZ scale; no alignment or fitting is applied.",
    )
    html_template = html_template.replace("Fusion 璺?", "Fusion ")
    html = html_template.replace("__DATA__", json.dumps({"scenes": scenes}, ensure_ascii=False))
    html_path = OUTDIR / "interactive_current_vs_gtsam.html"
    html_path.write_text(html, encoding="utf-8")
    print(html_path)
    print(summary_path)


if __name__ == "__main__":
    main()
