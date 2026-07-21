#!/usr/bin/env python3
"""
V3b+ Rule B 7×3 Sequential Evaluation — 7 scenes × 3 trials, strictly sequential.
MACVO's own tqdm progress bars are preserved (output streams to terminal).
No outer tqdm wrapper — only simple print statements for status.

Rule B: two-level Visual Collapse Guard
  severe VC: num_visual_residuals < 30 × 1
  mild VC:   num_visual_residuals < 50 × 5
  trigger:   severe_vc_triggered OR mild_vc_triggered

No D2, no velocity reset, no translation_only.

Usage:
  conda activate macvo
  python Scripts/run_v3bplus_ruleB_7x3_sequential.py

  # Dry run (print what would run, do not execute)
  python Scripts/run_v3bplus_ruleB_7x3_sequential.py --dry-run

Output: Results/v3bplus_ruleB_7x3_YYYYMMDD_HHMMSS/
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
SCENES = ["turbid_harbor", "clear_shallow", "deep_dark", "caustic_shallow",
          "dam_inspection", "murky_coast", "open_water"]
TRIALS = [1, 2, 3]

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_ROOT = PROJECT_ROOT / "Results" / f"v3bplus_ruleB_7x3_{TIMESTAMP}"

# Rule B two-level VC guard flags.
# severe: <30×1, mild: <50×5
# No D2, no velocity reset.
V3B_FLAGS = [
    "--adaptive-v3b",
    "--v3b-vc-mode", "two_level",
    "--v3b-vc-severe-thr", "30",
    "--v3b-vc-severe-sustain", "1",
    "--v3b-vc-mild-thr", "50",
    "--v3b-vc-mild-sustain", "5",
]

RUN_TIMEOUT_S = 7200  # 2 hours per run

# ============================================================================

sys.path.insert(0, str(PROJECT_ROOT))
from Utility.Config import IncludeLoader


def build_config(scene_root: Path, result_dir: Path) -> Path:
    """Create temporary odom.yaml + seq.yaml for one V3b+ Rule B run."""
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
    """Write completed.ok with Rule B verification fields."""
    vc_mode = ""
    sev_thr = ""
    sev_sus = ""
    mild_thr = ""
    mild_sus = ""
    first_full = ""
    vc_source = ""
    vc_first = ""
    ad_path = result_dir / "adaptive_decisions.csv"
    if ad_path.exists():
        try:
            import csv
            with open(ad_path) as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            if rows:
                vc_mode = rows[0].get("vc_mode", "")
                sev_thr = rows[0].get("severe_vc_threshold", "")
                sev_sus = rows[0].get("severe_vc_sustain_config", "")
                mild_thr = rows[0].get("mild_vc_threshold", "")
                mild_sus = rows[0].get("mild_vc_sustain_config", "")
                for r in rows:
                    mode = r.get("adaptive_mode", "")
                    if "full_imu" in mode:
                        if not first_full:
                            first_full = r.get("pair_id", "")
                    if r.get("visual_collapse_triggered", "0") == "1":
                        if not vc_first:
                            vc_first = r.get("pair_id", "")
                            vc_source = r.get("visual_collapse_trigger_source", "")
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
        f"vc_mode={vc_mode}\n"
        f"severe_vc_threshold={sev_thr}\n"
        f"severe_vc_sustain_config={sev_sus}\n"
        f"mild_vc_threshold={mild_thr}\n"
        f"mild_vc_sustain_config={mild_sus}\n"
        f"D2_rerun_enabled=false\n"
        f"velocity_reset_enabled=false\n"
        f"first_full_enter_pair={first_full}\n"
        f"visual_collapse_trigger_source={vc_source}\n"
        f"visual_collapse_first_trigger_pair={vc_first}\n"
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
    print(f"  Config: Rule B two-level VC (severe <30×1 | mild <50×5)")
    print("─" * 60)

    start_t = time.time()

    # ── Run MACVO — stdout/stderr inherit parent terminal ─────────
    # Do NOT use subprocess.PIPE — it breaks tqdm \r behavior.
    # Inheriting the parent terminal (None) preserves MACVO's tqdm.

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
        description="V3b+ Rule B 7×3 Sequential Evaluation — two-level VC guard"
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
        result_root = PROJECT_ROOT / "Results" / "v3bplus_ruleB_7x3_DRYRUN"

    result_root.mkdir(parents=True, exist_ok=True)

    # ── Build run list ─────────────────────────────────────────────
    runs = []
    for trial in TRIALS:
        for scene in SCENES:
            runs.append((scene, trial))

    total = len(runs)  # 21

    print("=" * 70)
    print("V3b+ Rule B 7×3 Sequential Evaluation")
    print(f"  Config:    Rule B two-level VC (severe <30×1 | mild <50×5)")
    print(f"  No D2, no velocity reset, no translation_only")
    print(f"  {total} runs total ({len(SCENES)} scenes × {len(TRIALS)} trials)")
    print(f"  Batch root: {batch_root}")
    print(f"  Output:     {result_root}")
    print(f"  Timeout:    {timeout_s}s per run")
    print(f"  Flags:      {' '.join(V3B_FLAGS)}")
    print(f"  MACVO tqdm: preserved (output streams directly to terminal)")
    print(f"  Outer tqdm: NOT used — only simple status prints")
    print(f"  Parallel:   NO — strictly sequential")
    print(f"  Skip completed: yes")
    print("=" * 70)

    if args.dry_run:
        print()
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

    print()
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
    print("V3b+ Rule B 7×3 COMPLETE")
    print("=" * 70)
    print(f"Output: {result_root}")
    print()
    print(f"{'Scene':<20}  {'T1':>8}  {'T2':>8}  {'T3':>8}  {'Median':>8}")
    print("-" * 60)

    from collections import defaultdict
    by_scene = defaultdict(dict)
    for r in results:
        by_scene[r["scene"]][r["trial"]] = r["ate"]

    for scene in SCENES:
        vals = [by_scene[scene].get(t, float("nan")) for t in [1, 2, 3]]
        cells = []
        for v in vals:
            cells.append(f"{v:.1f}" if not np.isnan(v) else "nan")
        med = np.nanmedian(vals)
        cells.append(f"{med:.1f}" if not np.isnan(med) else "nan")
        print(f"  {scene:<18} {cells[0]:>8} {cells[1]:>8} {cells[2]:>8} {cells[3]:>8}")

    print()
    print("To analyze: tell the agent 'V3b+ Rule B 7×3 run 已完成，开始 analyze.'")
    print(f"Results directory: {result_root}")

    print()
    print("Next step:")
    print("  python analysis_v3bplus_ruleB_7x3_report/analyze_ruleB_7x3.py")
    print(f"  (will generate report from {result_root})")
    print()


if __name__ == "__main__":
    main()
