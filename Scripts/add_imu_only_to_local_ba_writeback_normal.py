#!/usr/bin/env python3
"""Add an IMU-only mechanization trajectory to the Local BA writeback report."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

WORKDIR = Path("/home/admin1/macvo-dev")
if str(WORKDIR) not in sys.path:
    sys.path.insert(0, str(WORKDIR))

from Scripts import run_clear_circle_imu_only_mechanization as imu_mech
from Scripts import run_latest_closed_paths_methods as closed_paths


DEFAULT_SCENE = "clear_rectangle_normal_noise"
DEFAULT_OUTPUT_ROOT = WORKDIR / "analysis_local_ba_writeback_validation_20260708"
DEFAULT_SCENE_ROOT = (
    Path("/mnt/e/文档/holoocean/code/recordings/batch_zed100_closed_paths_smooth_20260705")
    / "normal_noise"
    / "clear_rectangle_path"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", default=DEFAULT_SCENE)
    parser.add_argument("--scene-root", type=Path, default=DEFAULT_SCENE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--seq-to", type=int, default=150)
    return parser.parse_args()


def load_existing_scene_trajectories(scene_dir: Path, scene: str) -> tuple[list[dict[str, object]], dict[str, pd.DataFrame]]:
    summary_path = scene_dir / "trajectory_summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)
    rows = pd.read_csv(summary_path).to_dict("records")

    summaries: list[dict[str, object]] = []
    trajectories: dict[str, pd.DataFrame] = {}
    for row in rows:
        method = str(row["method"])
        if method == "imu_only_mechanization":
            continue
        label = str(row.get("label") or f"{scene} / {method}")
        joined_path = scene_dir / "trajectories" / f"{scene}_{method}_joined.csv"
        if not joined_path.exists():
            raise FileNotFoundError(joined_path)
        summaries.append(row)
        trajectories[label] = pd.read_csv(joined_path)
    return summaries, trajectories


def make_imu_only_joined(
    *,
    scene: str,
    scene_root: Path,
    scene_dir: Path,
    seq_to: int,
) -> tuple[dict[str, object], pd.DataFrame]:
    _, joined = imu_mech.run_imu_only_for_scene(scene, scene_root, scene_dir)
    if seq_to > 0:
        joined = joined.iloc[:seq_to].copy()
    method = "imu_only_mechanization"
    label = f"{scene} / {method}"
    joined["err_m"] = (
        (
            joined[["tx_est", "ty_est", "tz_est"]].to_numpy(float)
            - joined[["tx_gt", "ty_gt", "tz_gt"]].to_numpy(float)
        )
        ** 2
    ).sum(axis=1) ** 0.5
    source = scene_dir / "trajectories" / f"{scene}_{method}_joined.csv"
    summary = imu_mech.evaluate_joined(
        label=label,
        scene=scene,
        method=method,
        joined=joined,
        source=str(source),
    )
    return summary, joined


def update_global_summary(output_root: Path, scene: str, added_summary: dict[str, object]) -> list[dict[str, object]]:
    summary_path = output_root / "trajectory_summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)
    rows = pd.read_csv(summary_path).to_dict("records")
    rows = [
        row
        for row in rows
        if not (str(row.get("scene")) == scene and str(row.get("method")) == "imu_only_mechanization")
    ]
    rows.append(added_summary)
    imu_mech.write_summary(summary_path, rows)
    return rows


def main() -> int:
    args = parse_args()
    scene = str(args.scene)
    output_root = args.output_root
    scene_dir = output_root / scene
    scene_dir.mkdir(parents=True, exist_ok=True)

    summaries, trajectories = load_existing_scene_trajectories(scene_dir, scene)
    imu_summary, imu_joined = make_imu_only_joined(
        scene=scene,
        scene_root=args.scene_root,
        scene_dir=scene_dir,
        seq_to=int(args.seq_to),
    )
    summaries.append(imu_summary)
    trajectories[str(imu_summary["label"])] = imu_joined

    page = closed_paths.write_scene_artifacts(
        output_root=output_root,
        scene=scene,
        summaries=summaries,
        trajectories=trajectories,
    )

    all_summaries = update_global_summary(output_root, scene, imu_summary)
    scene_pages = {
        "clear_rectangle_normal_noise": page,
        "clear_rectangle_zero_noise": output_root
        / "clear_rectangle_zero_noise"
        / "interactive_trajectory_gt_vs_est.html",
    }
    closed_paths.write_index(
        output_root=output_root,
        summaries=all_summaries,
        scene_pages={k: v for k, v in scene_pages.items() if v.exists()},
        skipped=[],
    )

    print(f"Added {imu_summary['label']}")
    print(
        "IMU-only metrics: "
        f"RMSE={float(imu_summary['ate_rmse_m']):.6f} m, "
        f"median={float(imu_summary['ate_median_m']):.6f} m, "
        f"final={float(imu_summary['ate_final_m']):.6f} m, "
        f"max={float(imu_summary['ate_max_m']):.6f} m"
    )
    print(f"Wrote page: {page}")
    print(f"Wrote summary: {output_root / 'trajectory_summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
