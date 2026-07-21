#!/usr/bin/env python3
"""Run the corrected noiseless rectangle fusion with two covariance floors."""

from __future__ import annotations

import argparse
import csv
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import yaml

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from Scripts.run_static63_cached_imu_fusion import (  # noqa: E402
    CALIBRATED_STATICINIT_METHOD,
    TASKS,
    result_dir,
    run_task,
)
from Scripts.run_vio_imu_prior_mode_grid import make_odom_cfg  # noqa: E402
from Scripts.run_visual_factor_cache_batch import (  # noqa: E402
    LATEST_IMUATT_METHOD,
    RETAINED_VARIANTS,
    switch_dashboard,
)


WORKDIR = Path("/home/admin1/macvo-dev")
DEFAULT_RESULT_ROOT = WORKDIR / "Results" / "rectangle_covariance_floor_ablation_20260713"
DEFAULT_LOG = WORKDIR / "logs" / "rectangle_covariance_floor_ablation_20260713.log"
SCENE = "clear_stop_turn_rectangle_truth_no_noise_no_bias"


@dataclass(frozen=True)
class FloorCase:
    label: str
    value: float

    @property
    def method_name(self) -> str:
        return f"{CALIBRATED_STATICINIT_METHOD}_{self.label}"


CASES = (
    FloorCase("floor_0", 0.0),
    FloorCase("floor_1e-8", 1e-8),
)


def rectangle_task():
    matches = [task for task in TASKS if task.dataset_scene == SCENE]
    if len(matches) != 1:
        raise RuntimeError(f"expected exactly one Static63 task for {SCENE}, found {len(matches)}")
    return matches[0]


def make_floor_config(case: FloorCase, root: Path) -> Path:
    variant = RETAINED_VARIANTS[LATEST_IMUATT_METHOD]._replace(name=case.method_name)
    odom_cfg = make_odom_cfg(variant, root)
    with odom_cfg.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    odom_args = config["Odometry"]["args"]
    optimizer_args = config["Odometry"]["optimizer"]["args"]
    odom_args["imu_static_initialization_enable"] = True
    odom_args["imu_static_initialization_duration_s"] = 3.0
    odom_args["imu_static_sigma_multiplier"] = 5.0
    odom_args["imu_static_gyro_mean_norm_max"] = 0.03
    odom_args["imu_static_acc_norm_error_max"] = 0.6
    optimizer_args["imu_vio_cov_diagonal_floor"] = float(case.value)
    output = root / f"odom_{case.label}.yaml"
    with output.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(config, stream, sort_keys=False)
    return output


def write_manifest(result_root: Path, *, seq_to: int | None) -> None:
    task = rectangle_task()
    result_root.mkdir(parents=True, exist_ok=True)
    fields = [
        "trial",
        "scene",
        "variant",
        "trajectory",
        "imu_config",
        "scene_root",
        "cache_scene",
        "cache_dir",
        "result_dir",
        "seq_to",
        "args",
        "created_at",
    ]
    with (result_root / "run_manifest.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for case in CASES:
            writer.writerow(
                {
                    "trial": 1,
                    "scene": task.dataset_scene,
                    "variant": case.method_name,
                    "trajectory": task.trajectory,
                    "imu_config": task.imu_config,
                    "scene_root": task.scene_root,
                    "cache_scene": task.cache_scene,
                    "cache_dir": task.cache_dir,
                    "result_dir": result_dir(result_root, task, case.method_name),
                    "seq_to": "" if seq_to is None else int(seq_to),
                    "args": (
                        "visual-cache replay; corrected static Bias persistence; "
                        f"imu_vio_cov_diagonal_floor={case.value:.12g}"
                    ),
                    "created_at": datetime.now().isoformat(timespec="seconds"),
                }
            )


def validate_inputs() -> None:
    task = rectangle_task()
    required = (
        task.scene_root / "imu_data.csv",
        task.scene_root / "ref_pose.csv",
        task.scene_root / "metadata.json",
        task.cache_dir / "manifest.json",
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("missing floor-ablation input:\n" + "\n".join(missing))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--seq-to", type=int, default=None)
    parser.add_argument("--timeout", type=int, default=7200)
    parser.add_argument("--dashboard-port", type=int, default=8765)
    parser.add_argument("--no-dashboard", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result_root = args.result_root.expanduser().resolve()
    validate_inputs()
    write_manifest(result_root, seq_to=args.seq_to)
    progress_path = result_root / "progress.csv"
    if progress_path.exists() and args.force:
        progress_path.unlink()

    print("Rectangle covariance-floor ablation")
    print(f"Scene: {SCENE}")
    print(f"Visual cache: {rectangle_task().cache_dir}")
    print(f"Result root: {result_root}")
    for case in CASES:
        print(f"  {case.label}: imu_vio_cov_diagonal_floor={case.value:.12g}")
    if args.dry_run:
        return 0

    if not args.no_dashboard:
        switch_dashboard(result_root, DEFAULT_LOG, port=int(args.dashboard_port))

    task = rectangle_task()
    failures = 0
    with tempfile.TemporaryDirectory(prefix="rectangle_cov_floor_") as temporary:
        config_root = Path(temporary)
        for case in CASES:
            case_root = config_root / case.label
            case_root.mkdir(parents=True, exist_ok=True)
            odom_cfg = make_floor_config(case, case_root)
            config_snapshot = result_root / "configs" / f"odom_{case.label}.yaml"
            config_snapshot.parent.mkdir(parents=True, exist_ok=True)
            config_snapshot.write_text(odom_cfg.read_text(encoding="utf-8"), encoding="utf-8")
            failures += run_task(
                task,
                result_root=result_root,
                odom_cfg=odom_cfg,
                config_root=case_root,
                timeout_s=int(args.timeout),
                seq_to=args.seq_to,
                force=bool(args.force),
                method_name=case.method_name,
            ) != 0
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
