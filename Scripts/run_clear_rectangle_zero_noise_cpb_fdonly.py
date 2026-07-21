#!/usr/bin/env python3
"""Run the legacy CP-B-FD-only controller on the zero-noise rectangle scene."""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import yaml

WORKDIR = Path("/home/admin1/macvo-dev")
if str(WORKDIR) not in sys.path:
    sys.path.insert(0, str(WORKDIR))

from Utility.Config import IncludeLoader


SCENE = "clear_rectangle_zero_noise"
SCENE_ROOT = Path(
    "/mnt/e/文档/holoocean/code/recordings/"
    "batch_clear_rectangle_zero_noise_20260704/clear_rectangle_path"
)
BASE_ODOM_CFG = WORKDIR / "Config/Experiment/MACVO/MACVO_HoloOcean_IMU.yaml"
SEQ_TEMPLATE = WORKDIR / "Config/Sequence/holoocean_imu.yaml"
DEFAULT_RESULT_ROOT = WORKDIR / "Results" / "clear_rectangle_zero_noise_methods_20260704"
RUN_TIMEOUT_S = 7200


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--timeout", type=int, default=RUN_TIMEOUT_S)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


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
    odom_args = odometry["args"]
    optimizer_args = odometry["optimizer"]["args"]

    # Match the paper-facing legacy CP-B-FD-only scripts exactly: enable the
    # controller's rotation and translation prior hooks, but do not enable the
    # newer post-fusion or preintegrated-VIO code paths.
    optimizer_args["post_imu_fusion_enable"] = False
    optimizer_args["post_imu_fusion_mode"] = "none"
    optimizer_args["autodiff"] = False
    odom_args["imu_rot_prior_enable"] = True
    odom_args["imu_trans_prior_enable"] = True
    odom_args["mapping"] = False
    optimizer_args["imu_rot_prior"] = True

    out = tmpdir / "odom_cpb_fdonly.yaml"
    write_yaml(out, cfg)
    return out


def make_seq_cfg(tmpdir: Path) -> Path:
    cfg = load_yaml(SEQ_TEMPLATE)
    cfg["args"]["root"] = str(SCENE_ROOT)
    cfg["args"]["batch_root"] = str(SCENE_ROOT.parent)
    cfg["args"]["scene"] = SCENE
    out = tmpdir / f"seq_{SCENE}.yaml"
    write_yaml(out, cfg)
    return out


def sanity_check() -> bool:
    ok = True
    if not SCENE_ROOT.exists():
        print(f"ERROR: missing scene directory: {SCENE_ROOT}")
        return False
    for subdir in ("left", "right"):
        if not (SCENE_ROOT / subdir).is_dir():
            print(f"ERROR: missing {subdir}/: {SCENE_ROOT / subdir}")
            ok = False
    for filename in ("imu_data.csv", "ref_pose.csv", "metadata.json"):
        if not (SCENE_ROOT / filename).exists():
            print(f"ERROR: missing {filename}: {SCENE_ROOT / filename}")
            ok = False
    return ok


def has_completed_pose(result_dir: Path) -> bool:
    return (result_dir / "poses.csv").exists() or any(result_dir.rglob("poses.csv"))


def flatten_nested(result_dir: Path) -> None:
    if (result_dir / "poses.csv").exists():
        return
    nested_poses = [p for p in sorted(result_dir.rglob("poses.csv")) if p.parent != result_dir]
    if not nested_poses:
        return
    nested_dir = nested_poses[0].parent
    for src in nested_dir.iterdir():
        if not src.is_file():
            continue
        os.replace(str(src), str(result_dir / src.name))


def append_manifest(result_root: Path, result_dir: Path) -> None:
    manifest = result_root / "run_manifest_extra_cpb_fdonly.csv"
    exists = manifest.exists()
    with manifest.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["scene", "variant", "scene_root", "result_dir", "args"],
        )
        if not exists:
            writer.writeheader()
        writer.writerow(
            {
                "scene": SCENE,
                "variant": "cpb_fd_only",
                "scene_root": str(SCENE_ROOT),
                "result_dir": str(result_dir),
                "args": " ".join(cpb_fd_only_args()),
            }
        )


def main() -> int:
    args = parse_args()
    result_dir = args.result_root / "trial_1" / "cpb_fd_only" / SCENE

    print("=" * 78)
    print("  Clear-rectangle zero-noise CP-B-FD-only run")
    print(f"  Scene:   {SCENE}")
    print(f"  Data:    {SCENE_ROOT}")
    print(f"  Result:  {result_dir}")
    print(f"  Args:    {' '.join(cpb_fd_only_args())}")
    print("=" * 78)

    if not sanity_check():
        return 1
    if args.dry_run:
        marker = "SKIP" if has_completed_pose(result_dir) and not args.overwrite else "RUN"
        print(f"{marker}: {result_dir}")
        return 0

    if result_dir.exists() and args.overwrite:
        for path in sorted(result_dir.rglob("*"), reverse=True):
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                path.rmdir()
    result_dir.mkdir(parents=True, exist_ok=True)

    if has_completed_pose(result_dir):
        print(f"SKIP existing poses.csv: {result_dir}")
        append_manifest(args.result_root, result_dir)
        return 0

    tmpdir = Path(tempfile.mkdtemp(prefix="clear_rectangle_cpb_fdonly_"))
    odom_cfg = make_odom_cfg(tmpdir)
    seq_cfg = make_seq_cfg(tmpdir)

    cmd = [
        sys.executable,
        str(WORKDIR / "MACVO.py"),
        "--odom",
        str(odom_cfg),
        "--data",
        str(seq_cfg),
        "--resultRoot",
        str(result_dir),
    ] + cpb_fd_only_args()

    started = time.time()
    try:
        proc = subprocess.run(cmd, cwd=str(WORKDIR), stdout=None, stderr=None, timeout=int(args.timeout))
    except subprocess.TimeoutExpired:
        print(f"TIMEOUT after {args.timeout}s")
        return 124

    flatten_nested(result_dir)
    elapsed = time.time() - started
    print(f"Return code: {proc.returncode} ({elapsed:.1f}s)")
    print(f"Has poses.csv: {has_completed_pose(result_dir)}")
    append_manifest(args.result_root, result_dir)
    return proc.returncode if proc.returncode != 0 else int(not has_completed_pose(result_dir))


if __name__ == "__main__":
    raise SystemExit(main())
