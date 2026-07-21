#!/usr/bin/env python3
"""Run 7 scenes × 4 methods = 28 experiments for MACVO HoloOcean benchmark.

Methods (clean ablation, no post-fusion):
  pure_macvo       – no IMU rotation, no IMU translation
  rotation_only    – IMU rotation only
  translation_only – IMU translation only
  full_imu         – both IMU rotation and translation

Usage:
  python Scripts/run_7x4_experiments.py --batch-root /path/to/batch --dry-run
  python Scripts/run_7x4_experiments.py --batch-root /path/to/batch --max-frames 30
  python Scripts/run_7x4_experiments.py --batch-root /path/to/batch --resume
  python Scripts/run_7x4_experiments.py --batch-root /path/to/batch --force
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
from Scripts.eval_qa_vif import evaluate_trajectory  # noqa: E402

# ── Constants ──────────────────────────────────────────────────────────
BATCH_ROOT_DEFAULT = Path("/mnt/e/文档/holoocean/code/recordings/batch_20260515_155653")
SCENES = [
    "turbid_harbor", "clear_shallow", "deep_dark", "caustic_shallow",
    "dam_inspection", "murky_coast", "open_water",
]
METHODS = ["pure_macvo", "rotation_only", "translation_only", "full_imu"]

BASE_ODOM_CFG = WORKDIR / "Config/Experiment/MACVO/MACVO_HoloOcean_IMU.yaml"
SEQ_TEMPLATE = WORKDIR / "Config/Sequence/holoocean_imu.yaml"


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.load(f, IncludeLoader)


def write_yaml(path: Path, data: dict) -> None:
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")


def make_method_config(method: str, odom_cfg: dict) -> dict:
    """Modify odometry config to set the correct IMU ablation switches.
    Returns a deep copy with method-specific changes.
    """
    import copy
    cfg = copy.deepcopy(odom_cfg)
    odom = cfg["Odometry"]
    opt = odom["optimizer"]["args"]

    # All methods: disable post-fusion for clean ablation
    opt["post_imu_fusion_enable"] = False
    opt["post_imu_fusion_mode"] = "none"
    opt["autodiff"] = False

    if method == "pure_macvo":
        odom["args"]["imu_rot_prior_enable"] = False
        odom["args"]["imu_trans_prior_enable"] = False
        opt["imu_rot_prior"] = False
    elif method == "rotation_only":
        odom["args"]["imu_rot_prior_enable"] = True
        odom["args"]["imu_trans_prior_enable"] = False
        opt["imu_rot_prior"] = True
    elif method == "translation_only":
        odom["args"]["imu_rot_prior_enable"] = False
        odom["args"]["imu_trans_prior_enable"] = True
        opt["imu_rot_prior"] = True  # needed for get_graph_data to read translation prior
    elif method == "full_imu":
        odom["args"]["imu_rot_prior_enable"] = True
        odom["args"]["imu_trans_prior_enable"] = True
        opt["imu_rot_prior"] = True
    else:
        raise ValueError(f"Unknown method: {method}")

    return cfg


def make_sequence_config(scene_root: Path) -> dict:
    """Create sequence config pointing to a specific scene."""
    cfg = load_yaml(SEQ_TEMPLATE)
    cfg["args"]["root"] = str(scene_root)
    return cfg


def run_macvo(
    odom_cfg: dict,
    seq_cfg: dict,
    result_dir: Path,
    seq_to: int | None,
    timeout_s: int = 7200,
) -> subprocess.CompletedProcess:
    """Run MACVO.py with given configs. Write temps, capture output.
    MACVO creates nested Results/<project>/<timestamp>/ dir under resultRoot.
    """
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
        "--resultRoot", str(result_dir),  # direct output to our result dir
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


def find_latest_poses(result_dir: Path) -> Path | None:
    """Find poses.csv in nested result directory structure (MACVO creates
    Results/<project>/<timestamp>/ inside resultRoot)."""
    candidates = sorted(result_dir.rglob("poses.csv"), key=lambda p: p.stat().st_mtime)
    return candidates[-1] if candidates else None


def flatten_results(result_dir: Path) -> None:
    """Move files from MACVO's nested output dir up to result_dir."""
    import shutil
    for poses_path in sorted(result_dir.rglob("poses.csv")):
        nested_dir = poses_path.parent
        if nested_dir == result_dir:
            continue  # already flat
        for f in nested_dir.iterdir():
            if f.is_file():
                dest = result_dir / f.name
                if not dest.exists():
                    shutil.move(str(f), str(dest))
        # Remove empty dirs
        try:
            for f in sorted(nested_dir.iterdir()):
                pass  # check if dir empty
            nested_dir.rmdir()
        except OSError:
            pass


def check_complete(result_dir: Path, expected_frames: int = 1800) -> tuple[bool, str]:
    """Check if a run is complete."""
    poses = find_latest_poses(result_dir)
    if poses is None:
        return False, "no poses.csv found"

    diag = result_dir / "frame_pair_diagnostics.csv"
    if not diag.exists():
        return False, "no frame_pair_diagnostics.csv"

    trace = result_dir / "metadata_usage_trace.json"
    if not trace.exists():
        return False, "no metadata_usage_trace.json"

    try:
        n_pose = len(list(open(poses))) - 1  # minus header
        n_diag = len(list(open(diag))) - 1
        if expected_frames > 0:
            if n_pose < expected_frames * 0.95:
                return False, f"poses only {n_pose}/{expected_frames}"
            if n_diag < (expected_frames - 1) * 0.95:
                return False, f"diagnostics only {n_diag}/{expected_frames - 1}"
    except Exception:
        return False, "error reading files"

    return True, f"OK ({n_pose} poses, {n_diag} pairs)"


def evaluate_run(scene_root: Path, result_dir: Path) -> dict:
    """Evaluate a completed run."""
    poses = find_latest_poses(result_dir)
    ref_pose = scene_root / "ref_pose.csv"
    if poses is None or not ref_pose.exists():
        return {"status": "no_data"}

    try:
        metrics = evaluate_trajectory(poses, ref_pose, f"{result_dir.parent.name}_{result_dir.name}")
        return {
            "status": "evaluated",
            "ate_rmse": metrics["ate"].get("ate_rmse", float("nan")),
            "rpe_trans_rmse": metrics["rpe"].get("rpe_trans_rmse", float("nan")),
            "scale": metrics["ate"].get("scale", float("nan")),
            "ate_mean": metrics["ate"].get("ate_mean", float("nan")),
            "rpe_trans_mean": metrics["rpe"].get("rpe_trans_mean", float("nan")),
        }
    except Exception as e:
        return {"status": "eval_failed", "error": str(e)}


def collect_diagnostics_stats(diag_csv: Path) -> dict:
    """Compute per-column statistics from frame_pair_diagnostics.csv."""
    try:
        data = {}
        with open(diag_csv, "r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            if not rows:
                return {}

        for col in reader.fieldnames or []:
            vals = []
            for row in rows:
                try:
                    v = float(row[col])
                    if not math.isnan(v):
                        vals.append(v)
                except (ValueError, TypeError):
                    pass
            if vals:
                data[f"mean_{col}"] = float(np.mean(vals))
                data[f"median_{col}"] = float(np.median(vals))
                data[f"std_{col}"] = float(np.std(vals))
        return data
    except Exception:
        return {}


def run_experiment(
    scene: str,
    method: str,
    batch_root: Path,
    result_root: Path,
    seq_to: int | None,
    timeout_s: int,
    force: bool,
) -> dict:
    """Run one scene-method experiment. Returns status dict."""
    scene_root = batch_root / scene
    result_dir = result_root / scene / method
    result_dir.mkdir(parents=True, exist_ok=True)

    status = {
        "scene": scene, "method": method, "status": "unknown",
        "return_code": None, "result_dir": str(result_dir),
    }

    # Check completeness
    if not force:
        complete, msg = check_complete(result_dir, expected_frames=seq_to or 1800)
        if complete:
            status["status"] = "complete_cached"
            status["note"] = msg
            return status

    # Build configs
    odom_cfg = load_yaml(BASE_ODOM_CFG)
    odom_cfg = make_method_config(method, odom_cfg)
    seq_cfg = make_sequence_config(scene_root)

    started = time.time()
    try:
        proc = run_macvo(odom_cfg, seq_cfg, result_dir, seq_to, timeout_s)
        elapsed = time.time() - started
        status["return_code"] = proc.returncode
        status["runtime_sec"] = round(elapsed, 1)

        if proc.returncode != 0:
            status["status"] = "failed"
            status["note"] = f"return code {proc.returncode}"
            return status

        # Move results from nested dir to our result_dir
        nested_dir = None
        for d in sorted(result_dir.rglob("poses.csv"), key=lambda p: p.stat().st_mtime):
            nested_dir = d.parent
            break
        if nested_dir and nested_dir != result_dir:
            import shutil
            for f in nested_dir.iterdir():
                if not (result_dir / f.name).exists():
                    shutil.move(str(f), str(result_dir / f.name))

        # Evaluate
        complete, msg = check_complete(result_dir, expected_frames=seq_to or 1800)
        if complete:
            eval_result = evaluate_run(scene_root, result_dir)
            status.update(eval_result)

            # Collect diagnostics stats
            diag_csv = result_dir / "frame_pair_diagnostics.csv"
            if diag_csv.exists():
                diag_stats = collect_diagnostics_stats(diag_csv)
                status.update(diag_stats)

            status["status"] = "complete"
            status["num_frames"] = len(list(open(result_dir / "poses.csv"))) - 1
            diag = result_dir / "frame_pair_diagnostics.csv"
            status["num_pairs"] = len(list(open(diag))) - 1 if diag.exists() else 0
        else:
            status["status"] = "incomplete"
            status["note"] = msg

    except subprocess.TimeoutExpired:
        status["status"] = "timeout"
        status["runtime_sec"] = timeout_s
    except Exception as e:
        status["status"] = "error"
        status["note"] = str(e)
        status["traceback"] = traceback.format_exc()

    return status


def main():
    parser = argparse.ArgumentParser(description="Run 7×4 HoloOcean experiments")
    parser.add_argument("--batch-root", type=Path, default=BATCH_ROOT_DEFAULT)
    parser.add_argument("--result-root", type=Path, default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--timeout", type=int, default=7200, help="Timeout per run in seconds")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true", help="Re-run even if complete")
    parser.add_argument("--resume", action="store_true", help="Skip already-complete runs")
    parser.add_argument("--scenes", nargs="*", default=None)
    parser.add_argument("--methods", nargs="*", default=None)
    args = parser.parse_args()

    batch_root = Path(args.batch_root)
    if not batch_root.exists():
        print(f"FAIL: batch root does not exist: {batch_root}")
        sys.exit(1)

    scenes = args.scenes or SCENES
    methods = args.methods or METHODS

    # Verify scenes exist
    missing = [s for s in scenes if not (batch_root / s / "metadata.json").exists()]
    if missing:
        print(f"FAIL: missing scenes: {missing}")
        sys.exit(1)

    # Setup result root
    if args.result_root:
        result_root = Path(args.result_root)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_root = WORKDIR / "Results" / f"holoocean_7x4_{ts}"
    result_root.mkdir(parents=True, exist_ok=True)

    total = len(scenes) * len(methods)

    if args.dry_run:
        print(f"DRY RUN — would execute {total} experiments")
        print(f"Batch root : {batch_root}")
        print(f"Result root: {result_root}")
        print(f"Max frames : {args.max_frames or 'full 1800'}")
        print(f"Resume     : {args.resume}")
        print(f"Force      : {args.force}")
        print()
        for scene in scenes:
            for method in methods:
                result_dir = result_root / scene / method
                print(f"  [{scene:20s}][{method:16s}] → {result_dir}")
        print(f"\nTotal: {total} experiments")
        return

    # Write experiment manifest
    manifest = {
        "batch_root": str(batch_root),
        "result_root": str(result_root),
        "scenes": scenes,
        "methods": methods,
        "max_frames": args.max_frames,
        "started_at": datetime.now().isoformat(),
    }
    (result_root / "experiment_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    status_csv = result_root / "experiment_status.csv"
    status_fields = [
        "scene", "method", "status", "return_code", "runtime_sec",
        "num_frames", "num_pairs", "ate_rmse", "rpe_trans_rmse", "scale"
    ]
    status_fp = open(status_csv, "w", newline="", encoding="utf-8")
    status_writer = csv.DictWriter(status_fp, fieldnames=status_fields, extrasaction="ignore")
    status_writer.writeheader()

    results = []
    completed = 0
    failed = 0

    for si, scene in enumerate(scenes):
        for mi, method in enumerate(methods):
            idx = si * len(methods) + mi + 1
            print(f"\n[{idx}/{total}] {scene} / {method} ", end="", flush=True)

            status = run_experiment(
                scene=scene, method=method,
                batch_root=batch_root, result_root=result_root,
                seq_to=args.max_frames, timeout_s=args.timeout,
                force=args.force,
            )
            results.append(status)

            # Write status row
            row = {k: status.get(k, "") for k in status_fields}
            status_writer.writerow(row)
            status_fp.flush()

            st = status.get("status", "?")
            if st == "complete" or st == "complete_cached":
                completed += 1
                ate = status.get("ate_rmse", float("nan"))
                print(f"✅ ATE={ate:.4f}" if not math.isnan(float(ate)) else "✅")
            else:
                failed += 1
                print(f"❌ {st} ({status.get('note', '')})")

    status_fp.close()
    print(f"\n{'='*60}")
    print(f"Done: {completed} completed, {failed} failed, {total} total")
    print(f"Results: {result_root}")
    print(f"Status : {status_csv}")


if __name__ == "__main__":
    main()
