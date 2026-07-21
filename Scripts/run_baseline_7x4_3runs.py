#!/usr/bin/env python3
"""
Run 7 scenes × 4 methods × 3 trials baseline experiments.
Clean fresh runs, no reuse of old results.

Usage:
  python Scripts/run_baseline_7x4_3runs.py --batch-root /path/to/batch
  python Scripts/run_baseline_7x4_3runs.py --dry-run
  python Scripts/run_baseline_7x4_3runs.py --scenes turbid_harbor deep_dark

Output: Results/baseline_7x4_3runs/trial_{n}/{scene}/{method}/
"""
import sys, subprocess, time, shutil, os, yaml, copy, csv, math
from datetime import datetime
from pathlib import Path

import numpy as np

WORKDIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKDIR))
from Utility.Config import IncludeLoader

# ── Config ──────────────────────────────────────────────────────────
BATCH_ROOT_DEFAULT = Path("/mnt/e/文档/holoocean/code/recordings/batch_20260515_155653")
ALL_SCENES = ["turbid_harbor","clear_shallow","deep_dark","caustic_shallow",
              "dam_inspection","murky_coast","open_water"]
ALL_METHODS = ["pure_macvo","rotation_only","translation_only","full_imu"]
N_TRIALS = 3

def load_yaml(p):
    with open(p) as f: return yaml.load(f, IncludeLoader)

def make_config(method, scene_root, result_dir):
    """Build odom + seq config for one run."""
    with open(WORKDIR / "Config/Experiment/MACVO/MACVO_HoloOcean_IMU.yaml") as f:
        cfg = yaml.load(f, IncludeLoader)
    with open(WORKDIR / "Config/Sequence/holoocean_imu.yaml") as f:
        seq = yaml.load(f, IncludeLoader)

    cfg = copy.deepcopy(cfg)
    od = cfg["Odometry"]; op = od["optimizer"]["args"]
    op["post_imu_fusion_enable"] = False
    op["post_imu_fusion_mode"] = "none"
    op["autodiff"] = False

    if method == "pure_macvo":
        od["args"]["imu_rot_prior_enable"] = False
        od["args"]["imu_trans_prior_enable"] = False
        op["imu_rot_prior"] = False
    elif method == "rotation_only":
        od["args"]["imu_rot_prior_enable"] = True
        od["args"]["imu_trans_prior_enable"] = False
        op["imu_rot_prior"] = True
    elif method == "translation_only":
        od["args"]["imu_rot_prior_enable"] = False
        od["args"]["imu_trans_prior_enable"] = True
        op["imu_rot_prior"] = True
    elif method == "full_imu":
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
        if nd == result_dir: continue
        for f in nd.iterdir():
            if f.is_file():
                dest = result_dir / f.name
                if not dest.exists():
                    os.rename(str(f), str(dest))
        try: nd.rmdir()
        except: pass

def evaluate_poses(poses_path, ref_path):
    """Quick evaluation: direct_ATE."""
    try:
        est = np.genfromtxt(poses_path, delimiter=',', dtype=float, skip_header=1)
        gt = np.genfromtxt(ref_path, delimiter=',', dtype=float, skip_header=1)
        if est.ndim == 1: est = est.reshape(1,-1)
        if gt.ndim == 1: gt = gt.reshape(1,-1)
        n = min(len(est), len(gt))
        e = est[:n, 1:4]; g = gt[:n, 1:4]
        return float(np.sqrt(np.mean(np.sum((e-g)**2, axis=1))))
    except: return float('nan')

def run_one(scene, method, trial_num, result_root, batch_root, timeout_s):
    """Run one MACVO experiment. Returns True on success."""
    result_dir = result_root / f"trial_{trial_num}" / scene / method
    if (result_dir / "poses.csv").exists():
        return True  # already done

    result_dir.mkdir(parents=True, exist_ok=True)
    scene_root = batch_root / scene
    tmp_dir = make_config(method, scene_root, result_dir)

    # Run MACVO.py — stdout/stderr go to terminal so user sees tqdm bar
    cmd = [
        sys.executable, str(WORKDIR / "MACVO.py"),
        "--odom", str(tmp_dir / "odom.yaml"),
        "--data", str(tmp_dir / "seq.yaml"),
        "--resultRoot", str(result_dir),
    ]

    start_t = time.time()
    try:
        proc = subprocess.run(cmd, cwd=str(WORKDIR),
                              timeout=timeout_s)
        elapsed = time.time() - start_t

        if proc.returncode != 0:
            print(f"    ❌ FAILED (exit={proc.returncode}, {elapsed:.0f}s)")
            return False

        flatten_nested(result_dir)

        # Quick evaluation
        poses_path = result_dir / "poses.csv"
        ref_path = batch_root / scene / "ref_pose.csv"
        ate = evaluate_poses(poses_path, ref_path) if poses_path.exists() else float('nan')

        print(f"    ✅ DONE (ATE={ate:.1f}m, {elapsed:.0f}s)")
        return True

    except subprocess.TimeoutExpired:
        print(f"    ⏰ TIMEOUT ({timeout_s}s)")
        return False
    except KeyboardInterrupt:
        print(f"\n    ⚠ INTERRUPTED by user")
        raise

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run 7×4×3 baseline stability experiments")
    parser.add_argument("--batch-root", type=Path, default=BATCH_ROOT_DEFAULT)
    parser.add_argument("--result-root", type=Path, default=None)
    parser.add_argument("--timeout", type=int, default=7200)
    parser.add_argument("--scenes", nargs="*", default=None)
    parser.add_argument("--methods", nargs="*", default=None)
    parser.add_argument("--trials", type=int, default=N_TRIALS)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    batch_root = Path(args.batch_root)
    scenes = args.scenes or ALL_SCENES
    methods = args.methods or ALL_METHODS
    n_trials = args.trials

    if args.result_root:
        result_root = Path(args.result_root)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_root = WORKDIR / "Results" / f"baseline_7x4_3runs_{ts}"
    result_root.mkdir(parents=True, exist_ok=True)

    total = len(scenes) * len(methods) * n_trials
    if args.dry_run:
        print(f"DRY RUN: {total} experiments")
        print(f"Output: {result_root}")
        for trial in range(1, n_trials+1):
            print(f"\n--- Trial {trial} ---")
            for scene in scenes:
                for method in methods:
                    print(f"  {scene}/{method} → {result_root}/trial_{trial}/{scene}/{method}/")
        return

    # ── Progress file ──────────────────────────────────────────────
    progress_csv = result_root / "progress.csv"
    progress_log = result_root / "progress.log"

    def plog(msg):
        ts = datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        with open(progress_log, "a") as f: f.write(line + "\n")
        print(line)

    plog(f"=== BASELINE 7×4×{n_trials} STARTED ===")
    plog(f"Scenes: {scenes}")
    plog(f"Methods: {methods}")
    plog(f"Total runs: {total}")
    plog(f"Output: {result_root}")

    # Track progress
    progress = {}
    started_at = time.time()
    done = 0; failed = 0; skipped = 0

    with open(progress_csv, "w") as pf:
        pf.write("trial,scene,method,status,ate,runtime_s\n")

    for trial in range(1, n_trials+1):
        plog(f"\n{'='*50}")
        plog(f"TRIAL {trial}/{n_trials} ({done+failed}/{total} done so far)")
        plog(f"{'='*50}")

        for scene_idx, scene in enumerate(scenes):
            for method_idx, method in enumerate(methods):
                overall_idx = (trial-1)*len(scenes)*len(methods) + scene_idx*len(methods) + method_idx + 1

                # Check if already exists
                result_dir = result_root / f"trial_{trial}" / scene / method
                if (result_dir / "poses.csv").exists():
                    ate = evaluate_poses(result_dir / "poses.csv", batch_root / scene / "ref_pose.csv")
                    plog(f"  ⏭ [{overall_idx}/{total}] {scene}/{method} trial {trial} (already done, ATE={ate:.1f})")
                    skipped += 1
                    with open(progress_csv, "a") as pf:
                        pf.write(f"{trial},{scene},{method},skipped,{ate:.1f},0\n")
                    continue

                plog(f"  ▶ [{overall_idx}/{total}] {scene}/{method} trial {trial}")

                ok = run_one(scene, method, trial, result_root, batch_root, args.timeout)

                if ok:
                    done += 1
                    poses_path = result_dir / "poses.csv"
                    ate = evaluate_poses(poses_path, batch_root / scene / "ref_pose.csv")
                    with open(progress_csv, "a") as pf:
                        pf.write(f"{trial},{scene},{method},ok,{ate:.1f},0\n")
                else:
                    failed += 1
                    with open(progress_csv, "a") as pf:
                        pf.write(f"{trial},{scene},{method},failed,nan,0\n")

    # ── Done ────────────────────────────────────────────────────────
    elapsed = time.time() - started_at
    plog(f"\n=== DONE ===")
    plog(f"Total: {total}, Done: {done}, Failed: {failed}, Skipped: {skipped}")
    plog(f"Elapsed: {elapsed/3600:.1f}h")
    plog(f"Output: {result_root}")

    # Quick summary
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Output: {result_root}")

    # Count results
    for trial in range(1, n_trials+1):
        print(f"\n--- Trial {trial} ---")
        for scene in scenes:
            for method in methods:
                d = result_root / f"trial_{trial}" / scene / method
                has_pose = (d / "poses.csv").exists()
                ate = evaluate_poses(d/"poses.csv", batch_root/scene/"ref_pose.csv") if has_pose else float('nan')
                mark = "✅" if has_pose else "❌"
                print(f"  {mark} {scene:20s} {method:20s} ATE={ate:8.1f}" if has_pose else f"  {mark} {scene:20s} {method:20s} MISSING")

if __name__ == "__main__":
    main()
