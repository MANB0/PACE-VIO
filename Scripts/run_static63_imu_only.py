#!/usr/bin/env python3
"""Run pure IMU mechanization for three trajectories and four sensor variants."""

from __future__ import annotations

import argparse
import csv
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

from run_clear_circle_imu_only_mechanization import run_imu_only_for_scene
from run_visual_factor_cache_batch import switch_dashboard


WORKDIR = Path("/home/admin1/macvo-dev")
DEFAULT_BATCH_ROOT = Path(
    "/mnt/e/文档/holoocean/code/recordings/"
    "batch_clear_truth_paths_20260713_static63_variants"
)
DEFAULT_OUTPUT_ROOT = WORKDIR / "Results" / "static63_imu_only_four_configs_20260713"
DEFAULT_LOG = WORKDIR / "logs" / "static63_imu_only_three_geometry_20260713.log"
SCENES = (
    "clear_circle_truth_normal_noise",
    "clear_circle_truth_bias_no_noise",
    "clear_circle_truth_noise_no_bias",
    "clear_circle_truth_no_noise_no_bias",
    "clear_stop_turn_rectangle_truth_normal_noise",
    "clear_stop_turn_rectangle_truth_bias_no_noise",
    "clear_stop_turn_rectangle_truth_noise_no_bias",
    "clear_stop_turn_rectangle_truth_no_noise_no_bias",
    "clear_straight_truth_normal_noise",
    "clear_straight_truth_bias_no_noise",
    "clear_straight_truth_noise_no_bias",
    "clear_straight_truth_no_noise_no_bias",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-root", type=Path, default=DEFAULT_BATCH_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--dashboard-port", type=int, default=8765)
    parser.add_argument("--no-dashboard", action="store_true")
    parser.add_argument("--force", action="store_true", help="Recompute complete existing results.")
    return parser.parse_args()


def write_dashboard_manifest(output_root: Path, batch_root: Path) -> None:
    fields = ["trial", "scene", "variant", "scene_root", "result_dir", "seq_to", "created_at"]
    with (output_root / "run_manifest.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for scene in SCENES:
            writer.writerow(
                {
                    "trial": 1,
                    "scene": scene,
                    "variant": "imu_only_mechanization",
                    "scene_root": batch_root / scene,
                    "result_dir": output_root / "trajectories",
                    "seq_to": "",
                    "created_at": datetime.now().isoformat(timespec="seconds"),
                }
            )
    (output_root / "progress.csv").write_text(
        "trial,scene,variant,status,return_code,runtime_s,result_dir\n",
        encoding="utf-8",
    )


def append_progress(output_root: Path, scene: str, *, status: str, runtime_s: str = "") -> None:
    with (output_root / "progress.csv").open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["trial", "scene", "variant", "status", "return_code", "runtime_s", "result_dir"],
        )
        writer.writerow(
            {
                "trial": 1,
                "scene": scene,
                "variant": "imu_only_mechanization",
                "status": status,
                "return_code": 0 if status == "ok" else "",
                "runtime_s": runtime_s,
                "result_dir": output_root / "trajectories",
            }
        )


def csv_data_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return max(sum(1 for _ in stream) - 1, 0)


def reusable_result(
    output_root: Path,
    scene_root: Path,
    scene: str,
    existing_summaries: dict[str, dict[str, object]],
) -> bool:
    trajectory = output_root / "trajectories" / f"{scene}_imu_only_poses.csv"
    frame_metrics = output_root / "frame_metrics" / f"{scene}_imu_only_frame_metrics.csv"
    ref_pose = scene_root / "ref_pose.csv"
    if scene not in existing_summaries or not trajectory.exists() or not frame_metrics.exists():
        return False
    expected_rows = csv_data_rows(ref_pose)
    return (
        expected_rows > 0
        and csv_data_rows(trajectory) == expected_rows
        and csv_data_rows(frame_metrics) == expected_rows
    )


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "trajectories").mkdir(parents=True, exist_ok=True)
    (args.output_root / "frame_metrics").mkdir(parents=True, exist_ok=True)
    summary_path = args.output_root / "imu_only_summary.csv"
    if summary_path.exists():
        existing_df = pd.read_csv(summary_path)
        existing_summaries = {
            str(row["scene"]): row.to_dict() for _, row in existing_df.iterrows()
        }
    else:
        existing_summaries = {}

    write_dashboard_manifest(args.output_root, args.batch_root)
    if not args.no_dashboard:
        switch_dashboard(args.output_root, DEFAULT_LOG, port=int(args.dashboard_port))

    summaries: list[dict[str, object]] = []
    for index, scene in enumerate(SCENES, start=1):
        scene_root = args.batch_root / scene
        required = (scene_root / "imu_data.csv", scene_root / "ref_pose.csv")
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError(f"{scene}: missing required inputs: {missing}")

        if not args.force and reusable_result(
            args.output_root,
            scene_root,
            scene,
            existing_summaries,
        ):
            print(f"[{index}/{len(SCENES)}] IMU-only reused: {scene}", flush=True)
            summaries.append(existing_summaries[scene])
            append_progress(args.output_root, scene, status="ok", runtime_s="reused")
            continue

        print(f"[{index}/{len(SCENES)}] IMU-only: {scene}", flush=True)
        append_progress(args.output_root, scene, status="running")
        started = time.monotonic()
        summary, joined = run_imu_only_for_scene(scene, scene_root, args.output_root)
        summaries.append(summary)
        joined.to_csv(
            args.output_root / "frame_metrics" / f"{scene}_imu_only_frame_metrics.csv",
            index=False,
        )
        append_progress(
            args.output_root,
            scene,
            status="ok",
            runtime_s=f"{time.monotonic() - started:.1f}",
        )

    pd.DataFrame(summaries).to_csv(summary_path, index=False)
    (args.output_root / "trajectories" / "pose_coordinate_frame.txt").write_text(
        "world NWU; trajectory point is CameraLeftSocket; absolute origin, no alignment\n",
        encoding="utf-8",
    )
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()
