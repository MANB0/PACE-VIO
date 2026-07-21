#!/usr/bin/env python3
"""
V3b++ latest CP-B-FD-only early-7 rerun.

This script creates a fresh, uniformly named result batch for the original
seven early scenes using the current paper-facing CP-B-FD-only protocol:
Rule B two-level VC + FD cooldown=30 + no FD-E grace.

It does not run any fixed-mode baseline or validation/locked scenes.

Usage:
    cd /home/admin1/macvo-dev
    conda activate macvo
    python Scripts/run_v3bpp_latest_cpb_fdonly_early7_3x.py --dry-run
    python Scripts/run_v3bpp_latest_cpb_fdonly_early7_3x.py
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import NamedTuple

import yaml

WORKDIR = Path("/home/admin1/macvo-dev")
sys.path.insert(0, str(WORKDIR))
from Utility.Config import IncludeLoader

SOURCE_NAME = "latest_cpb_early7"
METHOD_NAME = "CP-B-FD-only"

BATCH_ROOT = Path("/mnt/e/文档/holoocean/code/recordings/batch_20260515_155653")
SCENES = [
    "turbid_harbor",
    "clear_shallow",
    "deep_dark",
    "caustic_shallow",
    "dam_inspection",
    "murky_coast",
    "open_water",
]
TRIALS = 3

RESULT_ROOT = WORKDIR / "Results" / "v3bpp_latest_cpb_fdonly_early7_3x"
BASE_ODOM_CFG = WORKDIR / "Config/Experiment/MACVO/MACVO_HoloOcean_IMU.yaml"
SEQ_TEMPLATE = WORKDIR / "Config/Sequence/holoocean_imu.yaml"
RUN_TIMEOUT_S = 7200


class RunSpec(NamedTuple):
    trial: int
    scene: str
    batch_root: Path
    result_dir: Path


def cpb_fd_only_args() -> list[str]:
    return [
        "--adaptive-v3b",
        "--v3b-vc-mode",
        "two_level",
        "--v3b-vc-severe-thr",
        "30",
        "--v3b-vc-severe-sustain",
        "1",
        "--v3b-vc-mild-thr",
        "50",
        "--v3b-vc-mild-sustain",
        "5",
        "--v3b-fd-cooldown",
        "30",
    ]


def build_run_specs(
    *,
    scenes: list[str] | None = None,
    trials: int = TRIALS,
    batch_root: Path = BATCH_ROOT,
    result_root: Path = RESULT_ROOT,
) -> list[RunSpec]:
    scene_list = scenes or SCENES
    return [
        RunSpec(
            trial=trial,
            scene=scene,
            batch_root=batch_root,
            result_dir=result_root / f"trial_{trial}" / scene,
        )
        for trial in range(1, trials + 1)
        for scene in scene_list
    ]


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.load(f, IncludeLoader)


def write_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def make_odom_cfg(tmpdir: Path) -> Path:
    cfg = load_yaml(BASE_ODOM_CFG)
    odometry = cfg["Odometry"]
    optimizer_args = odometry["optimizer"]["args"]
    optimizer_args["post_imu_fusion_enable"] = False
    optimizer_args["post_imu_fusion_mode"] = "none"
    optimizer_args["autodiff"] = False
    odometry["args"]["imu_rot_prior_enable"] = True
    odometry["args"]["imu_trans_prior_enable"] = True
    odometry["args"]["mapping"] = False
    optimizer_args["imu_rot_prior"] = True
    out = tmpdir / "odom_latest_cpb_fdonly.yaml"
    write_yaml(out, cfg)
    return out


def make_seq_cfg(spec: RunSpec, tmpdir: Path) -> Path:
    cfg = load_yaml(SEQ_TEMPLATE)
    cfg["args"]["root"] = str(spec.batch_root / spec.scene)
    cfg["args"]["batch_root"] = str(spec.batch_root)
    cfg["args"]["scene"] = spec.scene
    out = tmpdir / f"seq_trial{spec.trial}_{spec.scene}.yaml"
    write_yaml(out, cfg)
    return out


def sanity_check_batch(specs: list[RunSpec]) -> bool:
    ok = True
    checked = set()
    for spec in specs:
        key = (spec.batch_root, spec.scene)
        if key in checked:
            continue
        checked.add(key)
        root = spec.batch_root / spec.scene
        if not root.exists():
            print(f"ERROR: missing scene directory: {root}")
            ok = False
            continue
        for subdir in ("left", "right"):
            if not (root / subdir).is_dir():
                print(f"ERROR: missing {subdir}/ for {spec.scene}: {root / subdir}")
                ok = False
        for filename in ("imu_data.csv", "ref_pose.csv", "metadata.json"):
            if not (root / filename).exists():
                print(f"ERROR: missing {filename} for {spec.scene}: {root / filename}")
                ok = False
    return ok


def has_completed_pose(result_dir: Path) -> bool:
    return (result_dir / "poses.csv").exists() or any(result_dir.rglob("poses.csv"))


def flatten_nested(result_dir: Path) -> None:
    direct_pose = result_dir / "poses.csv"
    if direct_pose.exists():
        return
    nested_poses = [p for p in sorted(result_dir.rglob("poses.csv")) if p.parent != result_dir]
    if not nested_poses:
        return
    nested_dir = nested_poses[0].parent
    for src in nested_dir.iterdir():
        if not src.is_file():
            continue
        dest = result_dir / src.name
        if not dest.exists():
            os.rename(str(src), str(dest))


def write_manifest(result_root: Path, specs: list[RunSpec], args: list[str]) -> None:
    result_root.mkdir(parents=True, exist_ok=True)
    manifest = result_root / "run_manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "source",
                "method",
                "trial",
                "scene",
                "batch_root",
                "result_dir",
                "args",
                "created_at",
            ]
        )
        for spec in specs:
            writer.writerow(
                [
                    SOURCE_NAME,
                    METHOD_NAME,
                    spec.trial,
                    spec.scene,
                    str(spec.batch_root),
                    str(spec.result_dir),
                    " ".join(args),
                    datetime.now().isoformat(timespec="seconds"),
                ]
            )


def append_progress(result_root: Path, row: dict) -> None:
    progress = result_root / "progress.csv"
    exists = progress.exists()
    with progress.open("a", newline="", encoding="utf-8") as f:
        fieldnames = ["trial", "scene", "status", "return_code", "runtime_s", "result_dir"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in fieldnames})


def run_one(spec: RunSpec, odom_cfg: Path, seq_cfg: Path, result_root: Path, timeout_s: int) -> int:
    spec.result_dir.mkdir(parents=True, exist_ok=True)
    if has_completed_pose(spec.result_dir):
        print(f"  -> SKIP existing poses.csv: {spec.result_dir}")
        return 0

    cmd = [
        sys.executable,
        str(WORKDIR / "MACVO.py"),
        "--odom",
        str(odom_cfg),
        "--data",
        str(seq_cfg),
        "--resultRoot",
        str(spec.result_dir),
    ] + cpb_fd_only_args()

    started = time.time()
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(WORKDIR),
            stdout=None,
            stderr=None,
            timeout=timeout_s,
        )
        flatten_nested(spec.result_dir)
        elapsed = time.time() - started
        status = "OK" if proc.returncode == 0 else f"FAIL(rc={proc.returncode})"
        print(f"  -> {status} ({elapsed:.1f}s)")
        append_progress(
            result_root,
            {
                "trial": spec.trial,
                "scene": spec.scene,
                "status": "ok" if proc.returncode == 0 else "failed",
                "return_code": proc.returncode,
                "runtime_s": f"{elapsed:.1f}",
                "result_dir": str(spec.result_dir),
            },
        )
        return proc.returncode
    except subprocess.TimeoutExpired:
        elapsed = time.time() - started
        print(f"  -> TIMEOUT ({timeout_s}s)")
        append_progress(
            result_root,
            {
                "trial": spec.trial,
                "scene": spec.scene,
                "status": "timeout",
                "return_code": "",
                "runtime_s": f"{elapsed:.1f}",
                "result_dir": str(spec.result_dir),
            },
        )
        return 124


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run latest CP-B-FD-only on early 7 scenes x 3 trials.")
    parser.add_argument("--batch-root", type=Path, default=BATCH_ROOT)
    parser.add_argument("--result-root", type=Path, default=RESULT_ROOT)
    parser.add_argument("--scenes", nargs="*", default=None)
    parser.add_argument("--trials", type=int, default=TRIALS)
    parser.add_argument("--timeout", type=int, default=RUN_TIMEOUT_S)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result_root = args.result_root
    specs = build_run_specs(
        scenes=args.scenes,
        trials=args.trials,
        batch_root=args.batch_root,
        result_root=result_root,
    )
    args_base = cpb_fd_only_args()

    print("=" * 72)
    print("  V3b++ latest CP-B-FD-only early-7 rerun")
    print(f"  Source: {SOURCE_NAME}")
    print(f"  Method: {METHOD_NAME}")
    print(f"  Batch:  {args.batch_root}")
    print(f"  Results:{result_root}")
    print(f"  Runs:   {len(specs)}")
    print(f"  Args:   {' '.join(args_base)}")
    print("=" * 72)

    if not sanity_check_batch(specs):
        return 1

    if args.dry_run:
        print("\nDRY RUN - no MACVO process will be started.")
        for idx, spec in enumerate(specs, start=1):
            marker = "SKIP" if has_completed_pose(spec.result_dir) else "RUN"
            print(f"  [{idx:02d}/{len(specs):02d}] {marker} trial={spec.trial} scene={spec.scene} -> {spec.result_dir}")
        return 0

    result_root.mkdir(parents=True, exist_ok=True)
    write_manifest(result_root, specs, args_base)

    tmpdir = Path(tempfile.mkdtemp(prefix="v3bpp_latest_cpb_early7_"))
    print(f"Temp config dir: {tmpdir}")
    odom_cfg = make_odom_cfg(tmpdir)

    failures = []
    started_all = time.time()
    for idx, spec in enumerate(specs, start=1):
        seq_cfg = make_seq_cfg(spec, tmpdir)
        print(f"\n-- Run {idx}/{len(specs)}: trial={spec.trial} scene={spec.scene} --")
        rc = run_one(spec, odom_cfg, seq_cfg, result_root, args.timeout)
        if rc != 0:
            failures.append((spec, rc))

    elapsed_all = time.time() - started_all
    print("\n" + "=" * 72)
    print(f"  Completed attempted schedule in {elapsed_all / 60:.1f} min")
    print(f"  Results: {result_root}")
    print(f"  Manifest: {result_root / 'run_manifest.csv'}")
    print(f"  Progress: {result_root / 'progress.csv'}")
    if failures:
        print(f"  Failures: {len(failures)}")
        for spec, rc in failures:
            print(f"    - trial={spec.trial} scene={spec.scene} rc={rc}")
    else:
        print("  No failed return codes in this run.")
    print("=" * 72)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
