#!/usr/bin/env python3
"""
Analyse all holdout validation results.
========================================
Usage:
    cd /home/admin1/macvo-dev
    conda activate macvo
    python Scripts/analyse_holdout_results.py
"""

from __future__ import annotations

import sys
import csv
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

WORKDIR = Path("/home/admin1/macvo-dev")
sys.path.insert(0, str(WORKDIR))

from Utility.Config import IncludeLoader
from Scripts.eval_qa_vif import evaluate_trajectory_direct  # direct ATE (no rotation alignment)

RESULT_ROOT = WORKDIR / "Results/holdout_validation"
SCENES = ["moderate_turbidity", "open_water_overcast", "twilight_coast"]


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.load(f, IncludeLoader)


def identify_run(config: dict, dirname: str) -> tuple[str, str, str, dict]:
    """Return (scene, mode_label, trial_label, metrics_dict) for a single run dir."""
    od = config["Odometry"]
    args = od["args"]
    opt_args = od["optimizer"]["args"]

    # Identify scene from dirname
    scene = "unknown"
    for s in SCENES:
        if s in dirname:
            scene = s
            break

    # Identify mode from IMU flags
    imu_rot = args.get("imu_rot_prior_enable", False)
    imu_trans = args.get("imu_trans_prior_enable", False)
    imu_rot_prior = opt_args.get("imu_rot_prior", False)
    mapping = args.get("mapping", True)
    post_fusion = opt_args.get("post_imu_fusion_enable", True)

    # Rule B: mapping=False, post_fusion disabled, requires adaptive-v3b
    if not mapping and not post_fusion:
        mode = "ruleB"
        # Check if it has adaptive gate info
        v3b = config.get("VisualHealthGateV3b", None)
    else:
        if not imu_rot and not imu_trans:
            mode = "pure_macvo"
        elif imu_rot and not imu_trans:
            mode = "rotation_only"
        elif not imu_rot and imu_trans:
            mode = "translation_only"
        elif imu_rot and imu_trans:
            mode = "full_imu"
        else:
            mode = "unknown"

    # Timestamp as trial identifier
    trial_id = dirname.split("_")[0:3]  # e.g. 05_28_210111

    return scene, mode, "_".join(trial_id), {}


def compute_ate(poses_path: Path, gt_path: Path, tag: str) -> Optional[dict]:
    """Compute direct ATE (no rotation alignment). Returns None on failure."""
    if not poses_path.exists():
        return None
    try:
        result = evaluate_trajectory_direct(poses_path, gt_path, tag)
        scale = result["ate"].get("scale", 1.0)
        return {
            "ate_rmse": float(result["ate"]["ate_rmse"]),
            "ate_mean": float(result["ate"]["ate_mean"]),
            "ate_std": float(result["ate"]["ate_std"]),
            "rpe_trans_rmse": float(result["rpe"]["rpe_trans_rmse"]),
            "rpe_rot_rmse_deg": float(result["rpe"]["rpe_rot_rmse_deg"]),
            "scale": float(scale) if scale is not None else 1.0,
        }
    except Exception as e:
        print(f"  [WARN] ATE failed for {tag}: {e}")
        return None


def fmt_ate(val: float) -> str:
    """Format ATE value with color indicators."""
    if val < 0.5:
        return f"{val:.3f}"
    elif val < 1.0:
        return f"{val:.3f}"
    else:
        return f"{val:.3f}"


def main():
    print("=" * 80)
    print("  Holdout Validation — Full Results Analysis")
    print("=" * 80)

    # ── Collect all runs ──────────────────────────────────────────
    runs = []  # list of (scene, mode, trial_str, ate_result)

    for phase in ["fixed_baseline", "ruleB"]:
        phase_root = RESULT_ROOT / phase
        project_dirs = list(phase_root.glob("*/"))
        for proj_dir in project_dirs:
            for run_dir in sorted(proj_dir.glob("*/")):
                config_path = run_dir / "config.yaml"
                poses_path = run_dir / "poses.csv"
                if not config_path.exists():
                    continue

                config = load_yaml(config_path)
                scene, mode, trial_id, _ = identify_run(config, run_dir.name)

                gt_path = Path(f"/mnt/e/文档/holoocean/code/recordings/batch_20260528_203401/{scene}/ref_pose.csv")
                tag = f"{scene}_{mode}_t{trial_id}"
                ate = compute_ate(poses_path, gt_path, tag)

                runs.append({
                    "scene": scene,
                    "mode": mode,
                    "trial_id": trial_id,
                    "run_dir": str(run_dir),
                    "ate": ate,
                })

    print(f"\nCollected {len(runs)} runs.\n")

    # ── Group by (scene, mode) ──────────────────────────────────
    grouped = defaultdict(list)
    for r in runs:
        key = (r["scene"], r["mode"])
        grouped[key].append(r)

    # ── Build per-scene per-mode table ──────────────────────────
    MODES_ORDER = ["pure_macvo", "rotation_only", "translation_only", "full_imu", "ruleB"]
    SCENES_ORDER = SCENES

    print("=" * 120)
    print("  TABLE 1: ATE RMSE (direct, no rotation alignment) — Median of 3 trials")
    print("=" * 120)
    header = f"{'Scene':<28}"
    for m in MODES_ORDER:
        header += f"{m:>16}"
    print(header)
    print("-" * 120)

    # Store for later analysis
    all_results = {}  # (scene, mode) -> list of ate_rmse values

    for scene in SCENES_ORDER:
        row = f"{scene:<28}"
        for mode in MODES_ORDER:
            key = (scene, mode)
            entries = grouped.get(key, [])
            ates = [e["ate"]["ate_rmse"] for e in entries if e["ate"] is not None]
            all_results[key] = ates
            if len(ates) >= 3:
                med = np.median(ates)
                row += f"{med:>15.3f} "
            elif len(ates) > 0:
                med = np.median(ates)
                row += f"{med:>12.3f}* "
            else:
                row += f"{'N/A':>16}"
        print(row)

    # ── Cross-scene summary ────────────────────────────────────
    print("\n" + "=" * 120)
    print("  TABLE 2: Cross-Scene Summary (median of per-scene medians)")
    print("=" * 120)
    header = f"{'Method':<28}{'Median':>10}{'Min':>10}{'Max':>10}{'Std':>10}{'Wins':>8}"
    print(header)
    print("-" * 120)

    # Find per-scene winner for each mode
    for mode in MODES_ORDER:
        scene_meds = []
        wins = 0
        for scene in SCENES_ORDER:
            key = (scene, mode)
            ates = all_results.get(key, [])
            if len(ates) >= 3:
                med = np.median(ates)
                scene_meds.append(med)
                # Check if this mode is best for this scene
                all_meds = {}
                for m2 in MODES_ORDER:
                    a2 = all_results.get((scene, m2), [])
                    if len(a2) >= 3:
                        all_meds[m2] = np.median(a2)
                if all_meds:
                    best_mode = min(all_meds, key=all_meds.get)
                    if best_mode == mode:
                        wins += 1

        if scene_meds:
            print(f"{mode:<28}{np.median(scene_meds):>10.3f}{np.min(scene_meds):>10.3f}"
                  f"{np.max(scene_meds):>10.3f}{np.std(scene_meds):>10.3f}{wins:>8}")

    # ── Rule B vs Best Fixed comparison ─────────────────────────
    print("\n" + "=" * 120)
    print("  TABLE 3: Rule B vs Oracle (best fixed per scene, median of 3 trials)")
    print("=" * 120)
    print(f"{'Scene':<28}{'Rule B':>12}{'Best Fixed':>12}{'Best Method':>18}{'Δ ATE':>12}{'Δ %':>10}")
    print("-" * 120)

    for scene in SCENES_ORDER:
        ruleB_ates = all_results.get((scene, "ruleB"), [])
        ruleB_med = np.median(ruleB_ates) if len(ruleB_ates) >= 3 else float('nan')

        best_fixed = float('inf')
        best_method = ""
        for mode in ["pure_macvo", "rotation_only", "translation_only", "full_imu"]:
            ates = all_results.get((scene, mode), [])
            if len(ates) >= 3:
                med = np.median(ates)
                if med < best_fixed:
                    best_fixed = med
                    best_method = mode

        if np.isfinite(ruleB_med) and np.isfinite(best_fixed):
            delta = ruleB_med - best_fixed
            pct = (delta / best_fixed) * 100
            sign = "+" if delta > 0 else ""
            print(f"{scene:<28}{ruleB_med:>12.3f}{best_fixed:>12.3f}"
                  f"{best_method:>18}{sign}{delta:>11.3f}{sign}{pct:>9.1f}%")

    # ── Per-trial breakdown (raw data) ─────────────────────────
    print("\n" + "=" * 120)
    print("  TABLE 4: Per-Trial Raw ATE RMSE")
    print("=" * 120)
    print(f"{'Scene':<22}{'Mode':<20}{'Trial':<24}{'ATE':>10}{'RPE_T':>10}{'RPE_R':>10}{'Scale':>10}")
    print("-" * 120)

    for r in sorted(runs, key=lambda x: (x["scene"], MODES_ORDER.index(x["mode"]) if x["mode"] in MODES_ORDER else 99, x["trial_id"])):
        ate = r["ate"]
        if ate:
            print(f"{r['scene']:<22}{r['mode']:<20}{r['trial_id']:<24}"
                  f"{ate['ate_rmse']:>10.3f}{ate['rpe_trans_rmse']:>10.3f}"
                  f"{ate['rpe_rot_rmse_deg']:>10.3f}{ate['scale']:>10.3f}")
        else:
            print(f"{r['scene']:<22}{r['mode']:<20}{r['trial_id']:<24}{'FAIL':>10}")

    # ── Stability analysis ──────────────────────────────────────
    print("\n" + "=" * 120)
    print("  TABLE 5: Stability — Std Dev across 3 trials (per scene, per mode)")
    print("=" * 120)
    header = f"{'Scene':<28}"
    for m in MODES_ORDER:
        header += f"{m:>16}"
    print(header)
    print("-" * 120)

    for scene in SCENES_ORDER:
        row = f"{scene:<28}"
        for mode in MODES_ORDER:
            ates = all_results.get((scene, mode), [])
            if len(ates) >= 3:
                std_val = np.std(ates)
                row += f"{std_val:>15.4f} "
            else:
                row += f"{'N/A':>16}"
        print(row)

    # ── Performance tiers ───────────────────────────────────────
    print("\n" + "=" * 80)
    print("  SUMMARY: Performance Tiers (cross-scene median ATE)")
    print("=" * 80)

    tier_summary = []
    for mode in MODES_ORDER:
        all_vals = []
        for scene in SCENES_ORDER:
            ates = all_results.get((scene, mode), [])
            if len(ates) >= 3:
                all_vals.append(np.median(ates))
        if all_vals:
            tier_summary.append((mode, np.median(all_vals), np.mean(all_vals),
                                np.min(all_vals), np.max(all_vals), len(all_vals)))

    tier_summary.sort(key=lambda x: x[1])
    print(f"{'Rank':<6}{'Method':<22}{'Median':>10}{'Mean':>10}{'Min':>10}{'Max':>10}{'#Scenes':>10}")
    print("-" * 80)
    for i, (mode, med, mean, mn, mx, n) in enumerate(tier_summary, 1):
        print(f"{i:<6}{mode:<22}{med:>10.3f}{mean:>10.3f}{mn:>10.3f}{mx:>10.3f}{n:>10}")

    print("\n" + "=" * 80)
    print("  Analysis complete.")
    print("=" * 80)


if __name__ == "__main__":
    main()
