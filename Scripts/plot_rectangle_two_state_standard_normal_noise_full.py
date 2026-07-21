#!/usr/bin/env python3
"""Plot the full normal-noise stop-turn rectangle two-state result."""

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


SCENE = "clear_stop_turn_rectangle_truth_normal_noise"
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
    / "Results/rectangle_normal_noise_two_state_standard_full_20260715"
    / "trial_1/vio_two_state_fixed_lag_standard_full"
    / SCENE
    / "poses.csv"
)
OUTDIR = WORKDIR / "analysis_rectangle_two_state_standard_normal_noise_full_20260715"


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
    lengths = {"GT": len(gt_rows), "MACVO": len(macvo_rows), "two-state": len(est_rows)}
    if min(lengths.values()) == 0:
        raise ValueError(f"A trajectory is empty: {lengths}")
    if len(set(lengths.values())) != 1:
        raise ValueError(f"Trajectory lengths differ: {lengths}")

    n = len(gt_rows)
    gt_xyz = [[x, y, z] for _, x, y, z in gt_rows]
    macvo_xyz = [[x, y, z] for _, x, y, z in macvo_rows]
    est_xyz = [[x, y, z] for _, x, y, z in est_rows]
    time_zero = gt_rows[0][0]
    scene_payload = {
        "scene": "Stop-turn rectangle / Normal noise / Full 63 s",
        "gt": gt_xyz,
        "macvo": macvo_xyz,
        "time_s": [(gt_rows[index][0] - time_zero) / 1e9 for index in range(n)],
        "error_m": position_errors(gt_rows, macvo_rows, n),
        "metrics": metrics(gt_rows, macvo_rows),
        "fusion": [
            {
                "key": "standard_normal_noise",
                "source": "two_state_D_standard_full",
                "config": "normal_noise",
                "label": "Two-state D / Standard / Normal noise",
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
            ("two_state_D_standard_full_normal_noise", est_rows, EST_PATH),
        ):
            writer.writerow({"method": method, **metrics(gt_rows, rows), "estimate_path": path})

    html_template = HTML_TEMPLATE.replace("{{", "{").replace("}}", "}")
    html_template = html_template.replace(
        "Circle, stop-turn rectangle, and straight trajectory comparison",
        "Stop-turn rectangle normal-noise result",
    )
    html_template = html_template.replace(
        "__METHOD_SCOPE__",
        "GT, Pure MACVO and two-state D with Standard local-frame preintegration",
    )
    html_template = html_template.replace(
        "__LINE_NOTE__",
        "All trajectories use synchronized timestamps and the same NWU coordinates with equal XYZ scale.",
    )
    html_template = html_template.replace("Fusion 璺?", "Fusion ")
    html = html_template.replace(
        "__DATA__", json.dumps({"scenes": [scene_payload]}, ensure_ascii=False)
    )
    html_path = OUTDIR / "interactive_trajectory_gt_vs_est.html"
    html_path.write_text(html, encoding="utf-8")
    print(html_path)
    print(summary_path)


if __name__ == "__main__":
    main()
