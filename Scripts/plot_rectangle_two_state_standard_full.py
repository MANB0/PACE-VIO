#!/usr/bin/env python3
"""Plot the full stop-turn rectangle Standard two-state result."""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

WORKDIR = Path("/home/admin1/macvo-dev")
if str(WORKDIR) not in sys.path:
    sys.path.insert(0, str(WORKDIR))

from Scripts.plot_static63_gt_macvo import HTML_TEMPLATE, metrics, read_xyz


SCENE = "clear_stop_turn_rectangle_truth_bias_no_noise"
GT_PATH = (
    Path("/mnt/e/文档/holoocean/code/recordings")
    / "batch_clear_truth_paths_20260713_static63_variants"
    / SCENE
    / "ref_pose.csv"
)
MACVO_PATH = (
    WORKDIR
    / "Results/visual_factor_cache_static63_unique_source_20260713/trial_1/pure_macvo"
    / "clear_stop_turn_rectangle_truth_normal_noise/poses.csv"
)
EST_PATH = (
    WORKDIR
    / "Results/rectangle_bias_no_noise_two_state_standard_full_20260715"
    / "trial_1/vio_two_state_fixed_lag_standard_full"
    / SCENE
    / "poses.csv"
)
OUTDIR = WORKDIR / "analysis_rectangle_two_state_standard_full_20260715"


def position_errors(
    gt: list[tuple[int, float, float, float]],
    est: list[tuple[int, float, float, float]],
    n: int,
) -> list[float]:
    return [
        math.sqrt(
            (est[index][1] - gt[index][1]) ** 2
            + (est[index][2] - gt[index][2]) ** 2
            + (est[index][3] - gt[index][3]) ** 2
        )
        for index in range(n)
    ]


def main() -> None:
    gt_rows = read_xyz(GT_PATH)
    macvo_rows = read_xyz(MACVO_PATH)
    est_rows = read_xyz(EST_PATH)
    n = min(len(gt_rows), len(macvo_rows), len(est_rows))
    if n == 0:
        raise ValueError("GT, Pure MACVO, or two-state trajectory is empty")
    if len({len(gt_rows), len(macvo_rows), len(est_rows)}) != 1:
        raise ValueError(
            "Trajectory lengths differ: "
            f"GT={len(gt_rows)}, MACVO={len(macvo_rows)}, two-state={len(est_rows)}"
        )

    gt_xyz = [[x, y, z] for _, x, y, z in gt_rows[:n]]
    macvo_xyz = [[x, y, z] for _, x, y, z in macvo_rows[:n]]
    est_xyz = [[x, y, z] for _, x, y, z in est_rows[:n]]
    time_zero = gt_rows[0][0]
    scene_payload = {
        "scene": "Stop-turn rectangle / Bias only / Full 63 s",
        "gt": gt_xyz,
        "macvo": macvo_xyz,
        "time_s": [(gt_rows[index][0] - time_zero) / 1e9 for index in range(n)],
        "error_m": position_errors(gt_rows, macvo_rows, n),
        "metrics": metrics(gt_rows, macvo_rows),
        "fusion": [
            {
                "key": "standard_bias_no_noise",
                "source": "two_state_D_standard_full",
                "config": "bias_no_noise",
                "label": "Two-state D / Standard / Bias only",
                "color": "#2563eb",
                "dasharray": "",
                "scene": SCENE,
                "xyz": est_xyz,
                "error_m": position_errors(gt_rows, est_rows, n),
                "metrics": metrics(gt_rows, est_rows),
                "path": str(EST_PATH),
            }
        ],
        "imu_only": [],
        "gt_path": str(GT_PATH),
        "macvo_path": str(MACVO_PATH),
    }

    OUTDIR.mkdir(parents=True, exist_ok=True)
    summary_path = OUTDIR / "trajectory_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
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
        for method, rows, path in (
            ("pure_macvo", macvo_rows, MACVO_PATH),
            ("two_state_D_standard_full", est_rows, EST_PATH),
        ):
            writer.writerow(
                {
                    "method": method,
                    **metrics(gt_rows, rows),
                    "estimate_path": path,
                }
            )

    html_template = HTML_TEMPLATE.replace("{{", "{").replace("}}", "}")
    html_template = html_template.replace(
        "Circle, stop-turn rectangle, and straight trajectory comparison",
        "Stop-turn rectangle: GT vs Pure MACVO vs two-state D",
    )
    html_template = html_template.replace(
        "__METHOD_SCOPE__",
        "GT, Pure MACVO and two-state D with Standard local-frame preintegration",
    )
    html_template = html_template.replace(
        "__LINE_NOTE__",
        "All trajectories use the same NWU coordinates and timestamps.",
    )
    html_template = html_template.replace("Fusion 路 ", "Fusion · ")
    html = html_template.replace(
        "__DATA__", json.dumps({"scenes": [scene_payload]}, ensure_ascii=False)
    )
    html_path = OUTDIR / "interactive_trajectory_gt_vs_est.html"
    html_path.write_text(html, encoding="utf-8")
    print(html_path)
    print(summary_path)


if __name__ == "__main__":
    main()
