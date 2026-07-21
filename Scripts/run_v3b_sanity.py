#!/usr/bin/env python3
"""
V3b Sanity Check: Run 7 scenes with adaptive V3b gate, 1 trial each.

Usage:
  python Scripts/run_v3b_sanity.py --batch-root /path/to/batch

Output: Results/v3b_sanity_YYYYMMDD_HHMMSS/
"""

import sys, subprocess, time, os, yaml, copy
from datetime import datetime
from pathlib import Path

import numpy as np

WORKDIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKDIR))
from Utility.Config import IncludeLoader

BATCH_ROOT_DEFAULT = Path("/mnt/e/文档/holoocean/code/recordings/batch_20260515_155653")
ALL_SCENES = ["turbid_harbor","clear_shallow","deep_dark","caustic_shallow",
              "dam_inspection","murky_coast","open_water"]


def load_yaml(p):
    with open(p) as f:
        return yaml.load(f, IncludeLoader)


def make_config(scene_root, result_dir, force_mode=""):
    """Build V3b adaptive config with rotation_only as base IMU setting."""
    with open(WORKDIR / "Config/Experiment/MACVO/MACVO_HoloOcean_IMU.yaml") as f:
        cfg = yaml.load(f, IncludeLoader)
    with open(WORKDIR / "Config/Sequence/holoocean_imu.yaml") as f:
        seq = yaml.load(f, IncludeLoader)

    cfg = copy.deepcopy(cfg)
    od = cfg["Odometry"]
    op = od["optimizer"]["args"]
    op["post_imu_fusion_enable"] = False
    op["post_imu_fusion_mode"] = "none"
    op["autodiff"] = False

    # Base: enable both IMU priors (V3b gate will override per decision)
    od["args"]["imu_rot_prior_enable"] = True
    od["args"]["imu_trans_prior_enable"] = True
    op["imu_rot_prior"] = True

    seq["args"]["root"] = str(scene_root)

    tmp_dir = result_dir / ".tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    (tmp_dir / "odom.yaml").write_text(yaml.safe_dump(cfg))
    (tmp_dir / "seq.yaml").write_text(yaml.safe_dump(seq))
    return tmp_dir


def flatten_nested(result_dir):
    """Move files from MACVO nested dir up to result_dir."""
    for nested in sorted(result_dir.rglob("poses.csv")):
        nd = nested.parent
        if nd == result_dir:
            continue
        for f in nd.iterdir():
            if f.is_file():
                dest = result_dir / f.name
                if not dest.exists():
                    os.rename(str(f), str(dest))
        try:
            nd.rmdir()
        except:
            pass


def evaluate_poses(poses_path, ref_path):
    """Quick evaluation: direct_ATE (no alignment)."""
    try:
        est = np.genfromtxt(poses_path, delimiter=',', dtype=float, skip_header=1)
        gt = np.genfromtxt(ref_path, delimiter=',', dtype=float, skip_header=1)
        if est.ndim == 1:
            est = est.reshape(1, -1)
        if gt.ndim == 1:
            gt = gt.reshape(1, -1)
        n = min(len(est), len(gt))
        e = est[:n, 1:4]
        g = gt[:n, 1:4]
        return float(np.sqrt(np.mean(np.sum((e - g)**2, axis=1))))
    except:
        return float('nan')


def run_one(scene, result_root, batch_root, timeout_s, force_mode=""):
    """Run one MACVO experiment with V3b gate. Returns True on success."""
    result_dir = result_root / scene
    result_dir.mkdir(parents=True, exist_ok=True)
    scene_root = batch_root / scene
    tmp_dir = make_config(scene_root, result_dir, force_mode)

    # Run MACVO.py with adaptive_v3b flag
    cmd = [
        sys.executable, str(WORKDIR / "MACVO.py"),
        "--odom", str(tmp_dir / "odom.yaml"),
        "--data", str(tmp_dir / "seq.yaml"),
        "--resultRoot", str(result_dir),
        "--adaptive-v3b",
    ]
    if force_mode:
        cmd.extend(["--v3b-force", force_mode])

    print(f"  {'='*60}")
    print(f"  Scene: {scene}")
    print(f"  Result: {result_dir}")
    print(f"  {'='*60}")

    start_t = time.time()
    try:
        proc = subprocess.run(cmd, cwd=str(WORKDIR), timeout=timeout_s)
        elapsed = time.time() - start_t

        if proc.returncode != 0:
            print(f"  ❌ FAILED (exit={proc.returncode}, {elapsed:.0f}s)")
            return False, float('nan')

        flatten_nested(result_dir)

        # Quick evaluation
        poses_path = result_dir / "poses.csv"
        ref_path = batch_root / scene / "ref_pose.csv"
        ate = evaluate_poses(poses_path, ref_path) if poses_path.exists() else float('nan')

        print(f"  ✅ DONE (ATE={ate:.1f}m, {elapsed:.0f}s)")
        return True, ate

    except subprocess.TimeoutExpired:
        print(f"  ⏰ TIMEOUT ({timeout_s}s)")
        return False, float('nan')
    except KeyboardInterrupt:
        print(f"\n  ⚠ INTERRUPTED")
        raise


def main():
    import argparse
    parser = argparse.ArgumentParser(description="V3b 7-scene sanity check")
    parser.add_argument("--batch-root", type=Path, default=BATCH_ROOT_DEFAULT)
    parser.add_argument("--result-root", type=Path, default=None)
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--scenes", nargs="*", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", type=str, default="",
                        help="Force mode for all scenes: rotation_only, pure_macvo, full_imu")
    args = parser.parse_args()

    batch_root = Path(args.batch_root)
    scenes = args.scenes or ALL_SCENES

    if args.result_root:
        result_root = Path(args.result_root)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_root = WORKDIR / "Results" / f"v3b_sanity_{ts}"

    result_root.mkdir(parents=True, exist_ok=True)
    print(f"V3b Sanity Check: {len(scenes)} scenes → {result_root}")
    print(f"Batch root: {batch_root}")
    print()

    if args.dry_run:
        for scene in scenes:
            print(f"  [DRY-RUN] {scene}")
        return

    results = {}
    for i, scene in enumerate(scenes):
        print(f"\n[{i+1}/{len(scenes)}] {scene}")
        ok, ate = run_one(scene, result_root, batch_root, args.timeout, args.force)
        results[scene] = {"ok": ok, "ate": ate}

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for scene, r in results.items():
        status = "✅" if r["ok"] else "❌"
        print(f"  {status} {scene}: ATE={r['ate']:.1f}m")

    # Write summary CSV
    import csv
    summary_path = result_root / "v3b_sanity_summary.csv"
    with open(summary_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scene", "ok", "direct_ATE"])
        for scene, r in results.items():
            w.writerow([scene, "1" if r["ok"] else "0", f"{r['ate']:.3f}"])
    print(f"\nSummary: {summary_path}")


if __name__ == "__main__":
    main()
