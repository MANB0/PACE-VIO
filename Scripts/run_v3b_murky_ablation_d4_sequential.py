#!/usr/bin/env python3
"""
V3b murky ablation D4 — sequential script generator

D4 = visual_collapse_sustain 5 → 1

Run 3 trials of murky_coast only, strictly sequential.
MACVO's own tqdm progress bar is preserved (streams directly to terminal).

Usage:
  conda activate macvo
  python Scripts/run_v3b_murky_ablation_d4_sequential.py

  # Dry run (print what would run, do not execute)
  python Scripts/run_v3b_murky_ablation_d4_sequential.py --dry-run

Output: Results/v3b_murky_ablation_d4_YYYYMMDD_HHMMSS/
"""
from __future__ import annotations

import sys
import subprocess
import time
import os
import yaml
import copy
from datetime import datetime
from pathlib import Path

import numpy as np

# ============================================================================
# CONFIGURATION
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PYTHON_EXECUTABLE = sys.executable

BATCH_ROOT = Path("/mnt/e/文档/holoocean/code/recordings/batch_20260515_155653")
SCENE = "murky_coast"
TRIALS = [1, 2, 3]

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_ROOT = PROJECT_ROOT / "Results" / f"v3b_murky_ablation_d4_{TIMESTAMP}"

# D4: visual_collapse_sustain = 1 (default is 5)
# All other thresholds remain at default V3b values.
V3B_FLAGS = [
    "--adaptive-v3b",
    "--v3b-visual-collapse-sustain", "1",
]

RUN_TIMEOUT_S = 7200  # 2 hours per run

# ============================================================================

sys.path.insert(0, str(PROJECT_ROOT))
from Utility.Config import IncludeLoader


def build_config(scene_root: Path, result_dir: Path) -> Path:
    """Create temporary odom.yaml + seq.yaml for one V3b run."""
    with open(PROJECT_ROOT / "Config/Experiment/MACVO/MACVO_HoloOcean_IMU.yaml") as f:
        cfg = yaml.load(f, IncludeLoader)
    with open(PROJECT_ROOT / "Config/Sequence/holoocean_imu.yaml") as f:
        seq = yaml.load(f, IncludeLoader)

    cfg = copy.deepcopy(cfg)
    od = cfg["Odometry"]
    op = od["optimizer"]["args"]

    op["post_imu_fusion_enable"] = False
    op["post_imu_fusion_mode"] = "none"
    op["autodiff"] = False
    od["args"]["imu_rot_prior_enable"] = True
    od["args"]["imu_trans_prior_enable"] = True
    op["imu_rot_prior"] = True
    od["args"]["mapping"] = False

    seq["args"]["root"] = str(scene_root)

    tmp_dir = result_dir / ".tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    (tmp_dir / "odom.yaml").write_text(yaml.safe_dump(cfg))
    (tmp_dir / "seq.yaml").write_text(yaml.safe_dump(seq))
    return tmp_dir


def flatten_nested(result_dir: Path):
    """Move output files from nested subdirectories up to result_dir."""
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
        except Exception:
            pass


def write_completed_marker(result_dir: Path, scene: str, trial: int,
                           return_code: int):
    """Write completed.ok with detailed ablation logging fields."""
    # Read adaptive_decisions.csv for D4 verification fields
    vc_sustain_config = ""
    vc_first_trigger = ""
    first_full_enter = ""
    ad_path = result_dir / "adaptive_decisions.csv"
    if ad_path.exists():
        try:
            import csv
            with open(ad_path) as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            if rows:
                # visual_collapse_sustain_config — read from first row
                vc_sustain_config = rows[0].get("visual_collapse_sustain_config", "")
                # First pair where visual_collapse_triggered == 1
                for r in rows:
                    if r.get("visual_collapse_triggered", "0") == "1":
                        vc_first_trigger = r.get("pair_id", "")
                        break
                # First pair where adaptive_mode contains "full_imu"
                for r in rows:
                    mode = r.get("adaptive_mode", "")
                    if "full_imu" in mode:
                        first_full_enter = r.get("pair_id", "")
                        break
        except Exception:
            pass

    marker = result_dir / "completed.ok"
    marker.write_text(
        f"scene={scene}\n"
        f"trial={trial}\n"
        f"timestamp={datetime.now().isoformat()}\n"
        f"return_code={return_code}\n"
        f"poses_exists={(result_dir / 'poses.csv').exists()}\n"
        f"frame_pair_diagnostics_exists={(result_dir / 'frame_pair_diagnostics.csv').exists()}\n"
        f"adaptive_decisions_exists={(result_dir / 'adaptive_decisions.csv').exists()}\n"
        f"ablation_name=D4\n"
        f"ablation_D4_enabled=True\n"
        f"visual_collapse_sustain_config={vc_sustain_config}\n"
        f"visual_collapse_first_trigger_pair={vc_first_trigger}\n"
        f"first_full_enter_pair={first_full_enter}\n"
    )


def run_one(scene: str, trial: int, result_root: Path,
            batch_root: Path, timeout_s: int) -> dict:
    """Run one MACVO experiment. MACVO's tqdm goes directly to terminal."""
    scene_result = result_root / f"trial_{trial}" / scene
    scene_result.mkdir(parents=True, exist_ok=True)
    scene_root = batch_root / scene

    # ── Check completed marker ────────────────────────────────────
    marker = scene_result / "completed.ok"
    if marker.exists():
        return {
            "scene": scene, "trial": trial, "ok": True,
            "ate": float("nan"), "elapsed": 0.0,
            "note": "SKIPPED (already done)",
        }

    # ── Build config ───────────────────────────────────────────────
    tmp_dir = build_config(scene_root, scene_result)

    # ── Build command ──────────────────────────────────────────────
    cmd = [
        str(PYTHON_EXECUTABLE), str(PROJECT_ROOT / "MACVO.py"),
        "--odom", str(tmp_dir / "odom.yaml"),
        "--data", str(tmp_dir / "seq.yaml"),
        "--resultRoot", str(scene_result),
    ] + V3B_FLAGS

    log_path = scene_result / "run.log"

    # ── Print status (simple print, no outer tqdm) ─────────────────
    label = f"[{scene} trial_{trial}]"
    print()
    print("─" * 60)
    print(f"  {label}")
    print(f"  Output: {scene_result}")
    print(f"  Ablation: D4 (visual_collapse_sustain = 1)")
    print("─" * 60)

    start_t = time.time()

    # ── Run MACVO — stdout/stderr inherit parent terminal ─────────
    # CRITICAL: Do NOT use subprocess.PIPE — it makes stdout a non-TTY,
    # which causes tqdm to output a new line per frame instead of
    # updating in-place with \r. Inheriting the parent terminal (None)
    # preserves MACVO's tqdm \r behavior.
    #
    # stdout/stderr are written directly to the terminal so the user
    # can see MACVO's tqdm progress bar and status messages live.

    try:
        proc = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            stdout=None,   # inherit parent terminal (preserves TTY → tqdm \r works)
            stderr=None,   # inherit parent terminal
            timeout=timeout_s,
        )
        elapsed = time.time() - start_t

        if proc.returncode != 0:
            with open(log_path, "w") as lf:
                lf.write(f"FAILED exit={proc.returncode}\n")
            print(f"\n  ❌ FAILED {label}")
            print(f"     Scene:   {scene}")
            print(f"     Trial:   {trial}")
            print(f"     Run dir: {scene_result}")
            print(f"     Log:     {log_path}")
            print(f"     Exit:    {proc.returncode}")
            print(f"     Aborting — fix the issue and re-run.")
            sys.exit(1)

        # ── Flatten ────────────────────────────────────────────────
        flatten_nested(scene_result)

        # ── Quick ATE ──────────────────────────────────────────────
        poses_path = scene_result / "poses.csv"
        ref_path = batch_root / scene / "ref_pose.csv"
        ate = float("nan")
        if poses_path.exists() and ref_path.exists():
            try:
                est = np.genfromtxt(poses_path, delimiter=',', dtype=float, skip_header=1)
                gt = np.genfromtxt(ref_path, delimiter=',', dtype=float, skip_header=1)
                if est.ndim == 1:
                    est = est.reshape(1, -1)
                if gt.ndim == 1:
                    gt = gt.reshape(1, -1)
                n = min(len(est), len(gt))
                ate = float(np.sqrt(np.mean(np.sum((est[:n, 1:4] - gt[:n, 1:4]) ** 2, axis=1))))
            except Exception:
                pass

        write_completed_marker(scene_result, scene, trial, proc.returncode)
        print(f"  ✅ {label} ATE={ate:.1f}m  ({elapsed:.0f}s)")
        return {"scene": scene, "trial": trial, "ok": True,
                "ate": ate, "elapsed": elapsed, "note": ""}

    except subprocess.TimeoutExpired:
        with open(log_path, "w") as lf:
            lf.write("TIMEOUT\n")
        print(f"\n  ⏰ TIMEOUT {label}")
        print(f"     Scene:   {scene}")
        print(f"     Trial:   {trial}")
        print(f"     Run dir: {scene_result}")
        print(f"     Log:     {log_path}")
        sys.exit(1)

    except KeyboardInterrupt:
        print(f"\n  ⚠ INTERRUPTED {label}")
        print(f"     Run dir: {scene_result}")
        print(f"     Log:     {log_path}")
        raise

    except Exception as e:
        with open(log_path, "w") as lf:
            lf.write(f"ERROR {e}\n")
        print(f"\n  💥 ERROR {label}: {e}")
        print(f"     Scene:   {scene}")
        print(f"     Trial:   {trial}")
        print(f"     Run dir: {scene_result}")
        print(f"     Log:     {log_path}")
        sys.exit(1)


# ============================================================================
# MAIN
# ============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="V3b murky ablation D4 — sequential evaluation"
    )
    parser.add_argument("--batch-root", type=Path, default=BATCH_ROOT)
    parser.add_argument("--result-root", type=Path, default=None)
    parser.add_argument("--timeout", type=int, default=RUN_TIMEOUT_S)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    batch_root = args.batch_root
    result_root = args.result_root or RUN_ROOT
    timeout_s = args.timeout

    if args.dry_run:
        result_root = PROJECT_ROOT / "Results" / "v3b_murky_ablation_d4_DRYRUN"

    result_root.mkdir(parents=True, exist_ok=True)

    # ── Build run list ─────────────────────────────────────────────
    runs = [(SCENE, t) for t in TRIALS]   # murky_coast × 3
    total = len(runs)  # 3

    print("=" * 70)
    print("V3b murky ablation D4 — Sequential Evaluation")
    print(f"  Ablation:  D4 (visual_collapse_sustain 5 → 1)")
    print(f"  Scene:     {SCENE}")
    print(f"  Trials:    {len(TRIALS)} ({', '.join(str(t) for t in TRIALS)})")
    print(f"  Total:     {total} runs")
    print(f"  Batch:     {batch_root}")
    print(f"  Output:    {result_root}")
    print(f"  Timeout:   {timeout_s}s per run")
    print(f"  Flags:     {' '.join(V3B_FLAGS)}")
    print(f"  MACVO tqdm: preserved (output streams directly to terminal)")
    print(f"  Parallel:  NO — strictly sequential")
    print(f"  Skip completed: yes")
    print("=" * 70)
    print()

    if args.dry_run:
        print("DRY RUN — would execute:")
        for i, (scene, trial) in enumerate(runs, 1):
            out_dir = result_root / f"trial_{trial}" / scene
            marker = out_dir / "completed.ok"
            skip = "  [SKIP: exists]" if marker.exists() else ""
            print(f"  [{i:2d}/{total}] trial_{trial}/{scene}{skip}")
        print()
        print(f"Total: {total} runs, strictly sequential, no parallelism.")
        print("Run without --dry-run to execute.")
        return

    print("Starting... (MACVO's tqdm progress bars will appear below)")
    print()

    results = []
    try:
        for i, (scene, trial) in enumerate(runs):
            print(f"\n─── [{i+1}/{total}] trial_{trial}/{scene} ───")
            r = run_one(scene, trial, result_root, batch_root, timeout_s)
            results.append(r)

    except KeyboardInterrupt:
        print()
        print("╔════════════════════════════════════════════════════════╗")
        print("║  INTERRUPTED                                         ║")
        print("║  Completed runs are saved. Re-run to continue.       ║")
        print(f"║  {result_root}        ║")
        print("╚════════════════════════════════════════════════════════╝")
        sys.exit(130)

    # ── Summary ────────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("V3b murky ablation D4 — COMPLETE")
    print("=" * 70)
    print(f"Output: {result_root}")
    print()

    print(f"{'Trial':>8}  {'ATE':>8}  {'Elapsed':>8}  {'Status':>10}")
    print("-" * 42)
    for r in results:
        status = "✅" if r["ok"] else "❌"
        ate_str = f"{r['ate']:.1f}m" if not np.isnan(r["ate"]) else "nan"
        print(f"  {r['trial']:>4}  {ate_str:>8}  {r['elapsed']:>7.0f}s  {status:>10}")

    print()
    print("To analyze: tell the agent 'murky ablation D4 run 已完成，开始 analyze.'")
    print(f"Results directory: {result_root}")


if __name__ == "__main__":
    main()
