#!/usr/bin/env python3
"""Analyze zero-noise rectangle runs: pure MACVO, full VIO, and IMU-only."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

WORKDIR = Path("/home/admin1/macvo-dev")
if str(WORKDIR) not in sys.path:
    sys.path.insert(0, str(WORKDIR))

import Scripts.analyse_clear_circle_pair_vio as pair_analysis
import Scripts.run_clear_circle_imu_only_mechanization as imu_mech


SCENE = "clear_rectangle_zero_noise"
SCENE_ROOT = Path(
    "/mnt/e/文档/holoocean/code/recordings/"
    "batch_clear_rectangle_zero_noise_20260704/clear_rectangle_path"
)
DEFAULT_RESULT_ROOT = WORKDIR / "Results" / "clear_rectangle_zero_noise_methods_20260704"
DEFAULT_OUTPUT_ROOT = WORKDIR / "analysis_clear_rectangle_zero_noise_methods_20260704"
EXTRA_VARIANTS = [
    {
        "scene": SCENE,
        "variant": "cpb_fd_only",
        "scene_root": str(SCENE_ROOT),
        "result_dir": str(
            DEFAULT_RESULT_ROOT / "trial_1" / "cpb_fd_only" / SCENE
        ),
        "imu_factor_mode": "legacy_pose_prior",
    }
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def write_report(output_root: Path, summaries: list[dict[str, object]], figure_paths: list[Path]) -> None:
    lines = [
        "# Clear-rectangle zero-noise methods",
        "",
        f"Scene root: `{SCENE_ROOT}`",
        "",
        "| method | RMSE m | median m | final m | max m |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in sorted(summaries, key=lambda r: str(r["method"])):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["method"]),
                    f"{float(row['ate_rmse_m']):.6f}",
                    f"{float(row['ate_median_m']):.6f}",
                    f"{float(row['ate_final_m']):.6f}",
                    f"{float(row['ate_max_m']):.6f}",
                ]
            )
            + " |"
        )
    lines.extend(["", "Figures:"])
    for path in figure_paths:
        lines.append(f"- `{path.relative_to(output_root)}`")
    lines.extend(["", "Trajectory CSV files:", "- `trajectories/clear_rectangle_zero_noise_imu_only_poses.csv`"])
    (output_root / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    summaries, trajectories = imu_mech.load_existing_runs([args.result_root])
    summaries = [row for row in summaries if row["scene"] == SCENE]
    trajectories = {label: df for label, df in trajectories.items() if label.startswith(f"{SCENE} / ")}

    for row in EXTRA_VARIANTS:
        result_dir = Path(str(row["result_dir"]))
        if not result_dir.exists():
            continue
        summary, joined = pair_analysis.evaluate_run(row)
        summaries.append(
            imu_mech.evaluate_joined(
                label=str(summary["label"]),
                scene=str(summary["scene"]),
                method=str(summary["variant"]),
                joined=joined,
                source=str(summary["poses_path"]),
            )
        )
        trajectories[str(summary["label"])] = joined

    if not trajectories:
        raise RuntimeError(f"No MACVO/VIO trajectories found for {SCENE} in {args.result_root}")

    imu_summary, imu_joined = imu_mech.run_imu_only_for_scene(SCENE, SCENE_ROOT, args.output_root)
    summaries.append(imu_summary)
    trajectories[str(imu_summary["label"])] = imu_joined

    imu_mech.write_summary(args.output_root / "trajectory_summary.csv", summaries)
    figure_paths = [
        imu_mech.plot_xy(trajectories, args.output_root, gt_region=False),
        imu_mech.plot_xy(trajectories, args.output_root, gt_region=True),
        imu_mech.plot_xz(trajectories, args.output_root),
        imu_mech.plot_error(trajectories, args.output_root),
        imu_mech.plot_imu_only_focus(trajectories, args.output_root),
    ]
    pair_analysis.write_interactive_html(trajectories, args.output_root)
    figure_paths.append(args.output_root / "interactive_trajectory_gt_vs_est.html")
    write_report(args.output_root, summaries, figure_paths)

    print(f"Wrote {args.output_root}")
    print(f"Summary rows: {len(summaries)}")
    print(f"Interactive: {args.output_root / 'interactive_trajectory_gt_vs_est.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
