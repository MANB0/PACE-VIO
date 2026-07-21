#!/usr/bin/env python3
"""Plot the no-noise rectangle calibration/fusion causal comparison."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

WORKDIR = Path("/home/admin1/macvo-dev")
if str(WORKDIR) not in sys.path:
    sys.path.insert(0, str(WORKDIR))

from Scripts.plot_static63_gt_macvo import HTML_TEMPLATE, metrics, read_xyz  # noqa: E402


BATCH_ROOT = Path(
    "/mnt/e/\u6587\u6863/holoocean/code/recordings/"
    "batch_clear_truth_paths_20260713_static63_variants"
)
SCENE = "clear_stop_turn_rectangle_truth_no_noise_no_bias"
VISUAL_SCENE = "clear_stop_turn_rectangle_truth_normal_noise"
CONFIG = "no_noise_no_bias"
OUTDIR = WORKDIR / "analysis_rectangle_calibrated_imu_only_causal_20260713"

GT_PATH = BATCH_ROOT / SCENE / "ref_pose.csv"
MACVO_PATH = (
    WORKDIR
    / "Results/visual_factor_cache_static63_unique_source_20260713/trial_1/pure_macvo"
    / VISUAL_SCENE
    / "poses.csv"
)
STATICINIT_FUSION_PATH = (
    WORKDIR
    / "Results/static63_cached_imu_fusion_staticinit_calibrated_20260713/trial_1"
    / "vio_preintegrated_full_imuatt_staticinit_calibrated"
    / SCENE
    / "poses.csv"
)
PREVIOUS_FUSION_PATH = (
    WORKDIR
    / "Results/static63_cached_imu_fusion_four_configs_20260713/trial_1"
    / "vio_preintegrated_full_imuatt_estinit"
    / SCENE
    / "poses.csv"
)
RAW_IMU_ONLY_PATH = (
    WORKDIR
    / "Results/static63_imu_only_four_configs_20260713/trajectories"
    / f"{SCENE}_imu_only_poses.csv"
)
CALIBRATED_IMU_ONLY_PATH = (
    WORKDIR
    / "Results/static63_calibrated_imu_only_four_configs_20260713/trajectories"
    / f"{SCENE}_imu_only_staticinit_calibrated_poses.csv"
)


def _xyz(rows):
    return [[x, y, z] for _, x, y, z in rows]


def _error(gt, est):
    count = min(len(gt), len(est))
    return [
        ((est[i][1] - gt[i][1]) ** 2 + (est[i][2] - gt[i][2]) ** 2 + (est[i][3] - gt[i][3]) ** 2) ** 0.5
        for i in range(count)
    ]


def main() -> None:
    paths = {
        "gt": GT_PATH,
        "pure_macvo": MACVO_PATH,
        "imuatt_estinit": PREVIOUS_FUSION_PATH,
        "staticinit_calibrated_fusion": STATICINIT_FUSION_PATH,
        "raw_imu_only": RAW_IMU_ONLY_PATH,
        "calibrated_imu_only": CALIBRATED_IMU_ONLY_PATH,
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing comparison inputs:\n" + "\n".join(missing))

    rows = {name: read_xyz(path) for name, path in paths.items()}
    frame_count = min(len(value) for value in rows.values())
    gt_rows = rows["gt"][:frame_count]
    macvo_rows = rows["pure_macvo"][:frame_count]
    t0 = gt_rows[0][0]

    fusion_specs = (
        ("imuatt_estinit", "Previous imuatt_estinit", "#7c3aed", "8 5"),
        ("staticinit_calibrated_fusion", "Fusion staticinit_calibrated", "#dc2626", ""),
    )
    imu_only_specs = (
        ("raw_imu_only", "Original mechanization", "#d97706"),
        ("calibrated_imu_only", "Staticinit calibrated", "#059669"),
    )
    fusion = []
    for key, label, color, dasharray in fusion_specs:
        estimate = rows[key][:frame_count]
        fusion.append(
            {
                "key": key,
                "source": label,
                "config": CONFIG,
                "label": label,
                "color": color,
                "dasharray": dasharray,
                "scene": SCENE,
                "xyz": _xyz(estimate),
                "error_m": _error(gt_rows, estimate),
                "metrics": metrics(gt_rows, estimate),
                "path": str(paths[key]),
            }
        )
    imu_only = []
    for key, label, color in imu_only_specs:
        estimate = rows[key][:frame_count]
        imu_only.append(
            {
                "key": key,
                "config": CONFIG,
                "label": label,
                "color": color,
                "scene": SCENE,
                "xyz": _xyz(estimate),
                "error_m": _error(gt_rows, estimate),
                "metrics": metrics(gt_rows, estimate),
                "path": str(paths[key]),
            }
        )

    scene = {
        "scene": "Stop-turn rectangle / No bias / no noise",
        "gt": _xyz(gt_rows),
        "macvo": _xyz(macvo_rows),
        "time_s": [(row[0] - t0) / 1e9 for row in gt_rows],
        "error_m": _error(gt_rows, macvo_rows),
        "metrics": metrics(gt_rows, macvo_rows),
        "fusion": fusion,
        "imu_only": imu_only,
        "gt_path": str(GT_PATH),
        "macvo_path": str(MACVO_PATH),
    }

    OUTDIR.mkdir(parents=True, exist_ok=True)
    summary_path = OUTDIR / "trajectory_metrics.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=("method", "frames", "rmse_m", "mean_m", "max_m", "final_m", "path"),
        )
        writer.writeheader()
        for key in ("pure_macvo", "imuatt_estinit", "staticinit_calibrated_fusion", "raw_imu_only", "calibrated_imu_only"):
            result = metrics(gt_rows, rows[key][:frame_count])
            writer.writerow({"method": key, **result, "path": paths[key]})

    html = HTML_TEMPLATE.replace("{{", "{").replace("}}", "}")
    html = html.replace(
        "Circle, stop-turn rectangle, and straight trajectory comparison",
        "Rectangle calibrated IMU-only causal comparison",
    )
    html = html.replace(
        "__METHOD_SCOPE__",
        "GT, Pure MACVO, two Fusion versions and two IMU-only chains",
    )
    html = html.replace(
        "__LINE_NOTE__",
        "Select traces independently. ",
    )
    html = html.replace("__DATA__", json.dumps({"scenes": [scene]}, ensure_ascii=False))
    html_path = OUTDIR / "interactive_trajectory_gt_vs_est.html"
    html_path.write_text(html, encoding="utf-8")
    print(html_path)
    print(summary_path)


if __name__ == "__main__":
    main()
