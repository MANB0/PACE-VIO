#!/usr/bin/env python3
"""Run adaptive_v3a experiments on HoloOcean benchmark.

v3a = binary pure/full gate via running median flow_cov.
Only ONE method: adaptive_v3a. Uses --adaptive-v3a flag in MACVO.py.

Usage:
  # Short test (limited frames)
  python Scripts/run_v3a_experiments.py --batch-root /path/to/batch --max-frames 200 --short-test

  # Full 7-scene experiment
  python Scripts/run_v3a_experiments.py --batch-root /path/to/batch

  # Specific scenes
  python Scripts/run_v3a_experiments.py --batch-root /path/to/batch --scenes turbid_harbor clear_shallow
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

WORKDIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKDIR))

from Utility.Config import IncludeLoader  # noqa: E402
from Scripts.eval_qa_vif import evaluate_trajectory, evaluate_trajectory_direct  # noqa: E402

# ── Constants ──────────────────────────────────────────────────────────
BATCH_ROOT_DEFAULT = Path("/mnt/e/文档/holoocean/code/recordings/batch_20260515_155653")
ALL_SCENES = [
    "turbid_harbor", "clear_shallow", "deep_dark", "caustic_shallow",
    "dam_inspection", "murky_coast", "open_water",
]
SHORT_TEST_SCENES = ["clear_shallow", "caustic_shallow", "dam_inspection", "open_water", "murky_coast"]
SHORT_TEST_FRAMES = {"clear_shallow": 200, "caustic_shallow": 200, "dam_inspection": 300,
                     "open_water": 300, "murky_coast": 194}  # murky has 194 frames max

BASE_ODOM_CFG = WORKDIR / "Config/Experiment/MACVO/MACVO_HoloOcean_IMU.yaml"
SEQ_TEMPLATE = WORKDIR / "Config/Sequence/holoocean_imu.yaml"


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.load(f, IncludeLoader)


def write_yaml(path: Path, data: dict) -> None:
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")


def make_v3a_config(odom_cfg: dict) -> dict:
    """Modify odometry config for adaptive_v3a.
    v3a starts with pure_macvo defaults; the gate overrides at runtime.
    Set both IMU switches TRUE so full_imu can be activated by the gate.
    """
    import copy
    cfg = copy.deepcopy(odom_cfg)
    odom = cfg["Odometry"]
    opt = odom["optimizer"]["args"]

    opt["post_imu_fusion_enable"] = False
    opt["post_imu_fusion_mode"] = "none"
    opt["autodiff"] = False
    opt["imu_rot_prior"] = True  # needed for IMU prior path

    # Start with both enabled — the gate will disable as needed
    odom["args"]["imu_rot_prior_enable"] = True
    odom["args"]["imu_trans_prior_enable"] = True

    return cfg


def make_sequence_config(scene_root: Path) -> dict:
    cfg = load_yaml(SEQ_TEMPLATE)
    cfg["args"]["root"] = str(scene_root)
    return cfg


def run_macvo_v3a(
    odom_cfg: dict,
    seq_cfg: dict,
    result_dir: Path,
    seq_to: int | None,
    timeout_s: int = 7200,
) -> subprocess.CompletedProcess:
    result_dir.mkdir(parents=True, exist_ok=True)
    tmpdir = result_dir / ".tmp"
    tmpdir.mkdir(exist_ok=True)

    odom_path = tmpdir / "odom.yaml"
    seq_path = tmpdir / "seq.yaml"
    write_yaml(odom_path, odom_cfg)
    write_yaml(seq_path, seq_cfg)

    cmd = [
        sys.executable, str(WORKDIR / "MACVO.py"),
        "--odom", str(odom_path),
        "--data", str(seq_path),
        "--resultRoot", str(result_dir),
        "--adaptive-v3a",
    ]
    if seq_to is not None:
        cmd.extend(["--seq_to", str(seq_to)])

    run_log = result_dir / "run.log"
    stderr_log = result_dir / "stderr.log"

    with open(run_log, "w", encoding="utf-8") as out_f, open(stderr_log, "w", encoding="utf-8") as err_f:
        proc = subprocess.run(
            cmd, cwd=str(WORKDIR), text=True,
            stdout=out_f, stderr=err_f,
            timeout=timeout_s,
        )
    return proc


def find_poses(result_dir: Path) -> Path | None:
    candidates = sorted(result_dir.rglob("poses.csv"), key=lambda p: p.stat().st_mtime)
    return candidates[-1] if candidates else None


def flatten_results(result_dir: Path) -> None:
    import shutil
    for poses_path in sorted(result_dir.rglob("poses.csv")):
        nested_dir = poses_path.parent
        if nested_dir == result_dir:
            continue
        for f in nested_dir.iterdir():
            if f.is_file():
                dest = result_dir / f.name
                if not dest.exists():
                    shutil.move(str(f), str(dest))
        try:
            nested_dir.rmdir()
        except OSError:
            pass


def check_complete(result_dir: Path, expected_frames: int = 1800) -> tuple[bool, str]:
    poses = find_poses(result_dir)
    if poses is None:
        return False, "no poses.csv"
    diag = result_dir / "frame_pair_diagnostics.csv"
    if not diag.exists():
        return False, "no frame_pair_diagnostics.csv"
    dec = result_dir / "adaptive_decisions.csv"
    if not dec.exists():
        return False, "no adaptive_decisions.csv"
    try:
        n_pose = sum(1 for _ in open(poses)) - 1
        n_diag = sum(1 for _ in open(diag)) - 1
        n_dec = sum(1 for _ in open(dec)) - 1
        if expected_frames > 0:
            if n_pose < expected_frames * 0.95:
                return False, f"poses only {n_pose}/{expected_frames}"
        return True, f"OK ({n_pose} poses, {n_diag} pairs, {n_dec} decisions)"
    except Exception:
        return False, "error reading files"


def evaluate_run(scene_root: Path, result_dir: Path) -> dict:
    poses = find_poses(result_dir)
    ref_pose = scene_root / "ref_pose.csv"
    if poses is None or not ref_pose.exists():
        return {"status": "no_data"}

    try:
        metrics = evaluate_trajectory(poses, ref_pose, f"{result_dir.parent.name}/{result_dir.name}")
        direct = evaluate_trajectory_direct(poses, ref_pose)
        return {
            "status": "evaluated",
            "ate_rmse": metrics["ate"].get("ate_rmse", float("nan")),
            "rpe_trans_rmse": metrics["rpe"].get("rpe_trans_rmse", float("nan")),
            "direct_ate_rmse": direct.get("ate_rmse", float("nan")),
            "ate_mean": metrics["ate"].get("ate_mean", float("nan")),
            "rpe_trans_mean": metrics["rpe"].get("rpe_trans_mean", float("nan")),
        }
    except Exception as e:
        return {"status": "eval_failed", "error": str(e)}


def analyze_decision_csv(dec_csv: Path) -> dict:
    """Analyze adaptive_decisions.csv for mode percentages."""
    if not dec_csv.exists():
        return {}
    mode_counts = {}
    total = 0
    with open(dec_csv) as f:
        reader = csv.DictReader(f)
        for r in reader:
            m = r.get("mode", "")
            mode_counts[m] = mode_counts.get(m, 0) + 1
            total += 1
    if total == 0:
        return {}
    return {
        "full_imu_percent": round(100 * mode_counts.get("full_imu", 0) / total, 1),
        "pure_macvo_percent": round(100 * mode_counts.get("pure_macvo", 0) / total, 1),
        "rotation_only_percent": round(100 * mode_counts.get("rotation_only", 0) / total, 1),
        "translation_only_percent": round(100 * mode_counts.get("translation_only", 0) / total, 1),
        "total_decisions": total,
    }


def run_short_tests(batch_root: Path, result_root: Path, timeout_s: int = 1800) -> list[dict]:
    """Run short tests on 5 key scenes with limited frames."""
    results = []
    print(f"\n{'='*60}")
    print("SHORT TEST MODE")
    print(f"{'='*60}")

    for scene in SHORT_TEST_SCENES:
        max_frames = SHORT_TEST_FRAMES.get(scene, 200)
        scene_root = batch_root / scene
        result_dir = result_root / scene
        result_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n--- {scene} (max {max_frames} frames) ---")

        odom_cfg = load_yaml(BASE_ODOM_CFG)
        odom_cfg = make_v3a_config(odom_cfg)
        seq_cfg = make_sequence_config(scene_root)

        started = time.time()
        status = {"scene": scene, "method": "adaptive_v3a", "max_frames": max_frames}

        try:
            proc = run_macvo_v3a(odom_cfg, seq_cfg, result_dir, max_frames, timeout_s)
            elapsed = time.time() - started
            status["runtime_sec"] = round(elapsed, 1)
            status["return_code"] = proc.returncode

            if proc.returncode != 0:
                status["status"] = "failed"
                status["note"] = f"return code {proc.returncode}"
                results.append(status)
                continue

            flatten_results(result_dir)
            complete, msg = check_complete(result_dir, expected_frames=max_frames)
            status["complete"] = complete
            status["check_msg"] = msg

            if complete:
                eval_result = evaluate_run(scene_root, result_dir)
                status.update(eval_result)

            # Analyze decisions
            dec_csv = result_dir / "adaptive_decisions.csv"
            dec_analysis = analyze_decision_csv(dec_csv)
            status.update(dec_analysis)

            status["status"] = "complete" if complete else "incomplete"

        except subprocess.TimeoutExpired:
            status["status"] = "timeout"
            status["runtime_sec"] = timeout_s
        except Exception as e:
            status["status"] = "error"
            status["note"] = str(e)

        results.append(status)

        # Print summary
        fpct = status.get("full_imu_percent", "N/A")
        print(f"  status={status['status']}, full_imu={fpct}%")

    return results


def run_full_experiments(batch_root: Path, result_root: Path, scenes: list[str],
                         timeout_s: int = 7200) -> list[dict]:
    """Run full 7-scene v3a experiments."""
    results = []
    print(f"\n{'='*60}")
    print("FULL 7-SCENE EXPERIMENT")
    print(f"{'='*60}")

    for scene in scenes:
        scene_root = batch_root / scene
        result_dir = result_root / scene
        result_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n--- {scene} (full 1800 frames) ---")

        odom_cfg = load_yaml(BASE_ODOM_CFG)
        odom_cfg = make_v3a_config(odom_cfg)
        seq_cfg = make_sequence_config(scene_root)

        started = time.time()
        status = {"scene": scene, "method": "adaptive_v3a"}

        try:
            proc = run_macvo_v3a(odom_cfg, seq_cfg, result_dir, None, timeout_s)
            elapsed = time.time() - started
            status["runtime_sec"] = round(elapsed, 1)
            status["return_code"] = proc.returncode

            if proc.returncode != 0:
                status["status"] = "failed"
                status["note"] = f"return code {proc.returncode}"
                results.append(status)
                continue

            flatten_results(result_dir)
            complete, msg = check_complete(result_dir, expected_frames=1800)
            status["complete"] = complete
            status["check_msg"] = msg

            if complete:
                eval_result = evaluate_run(scene_root, result_dir)
                status.update(eval_result)

            dec_csv = result_dir / "adaptive_decisions.csv"
            dec_analysis = analyze_decision_csv(dec_csv)
            status.update(dec_analysis)

            status["status"] = "complete" if complete else "incomplete"

        except subprocess.TimeoutExpired:
            status["status"] = "timeout"
            status["runtime_sec"] = timeout_s
        except Exception as e:
            status["status"] = "error"
            status["note"] = str(e)

        results.append(status)
        fpct = status.get("full_imu_percent", "N/A")
        print(f"  status={status['status']}, full_imu={fpct}%")

    return results


def main():
    parser = argparse.ArgumentParser(description="Run adaptive_v3a experiments")
    parser.add_argument("--batch-root", type=Path, default=BATCH_ROOT_DEFAULT)
    parser.add_argument("--result-root", type=Path, default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--timeout", type=int, default=7200)
    parser.add_argument("--short-test", action="store_true",
                        help="Run short test (200-300 frames per scene) before full experiment")
    parser.add_argument("--scenes", nargs="*", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    batch_root = Path(args.batch_root)
    if not batch_root.exists():
        print(f"FAIL: batch root does not exist: {batch_root}")
        sys.exit(1)

    if args.result_root:
        result_root = Path(args.result_root)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_root = WORKDIR / "Results" / f"holoocean_adaptive_v3a_{ts}"
    result_root.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        scenes = args.scenes or (SHORT_TEST_SCENES if args.short_test else ALL_SCENES)
        print(f"DRY RUN — {len(scenes)} scenes")
        print(f"Batch root : {batch_root}")
        print(f"Result root: {result_root}")
        print(f"Short test : {args.short_test}")
        for s in scenes:
            print(f"  {s}")
        return

    if args.short_test:
        results = run_short_tests(batch_root, result_root, timeout_s=args.timeout)
        # Write short test report
        csv_path = result_root / "adaptive_v3a_short_test_summary.csv"
        if results:
            with open(csv_path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
                w.writeheader()
                w.writerows(results)
            print(f"\n✅ Short test summary → {csv_path}")

        # Print summary
        print(f"\n{'='*60}")
        print("SHORT TEST RESULTS")
        print(f"{'='*60}")
        for r in results:
            print(f"  {r['scene']:20s}: status={r['status']:12s} full_imu={r.get('full_imu_percent','N/A')}%")
    else:
        scenes = args.scenes or ALL_SCENES
        results = run_full_experiments(batch_root, result_root, scenes, timeout_s=args.timeout)

        # Write summary
        csv_path = result_root / "experiment_status.csv"
        if results:
            with open(csv_path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
                w.writeheader()
                w.writerows(results)
            print(f"\n✅ Experiment status → {csv_path}")

        print(f"\n{'='*60}")
        print("FULL EXPERIMENT RESULTS")
        print(f"{'='*60}")
        for r in results:
            print(f"  {r['scene']:20s}: status={r['status']:12s} full_imu={r.get('full_imu_percent','N/A')}% "
                  f"direct_ate={r.get('direct_ate_rmse','N/A')}")


if __name__ == "__main__":
    main()
