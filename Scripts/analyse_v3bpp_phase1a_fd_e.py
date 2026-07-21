#!/usr/bin/env python3
"""
V3b++ Phase 1a Analysis Script — FD-E grace=30, 5-scene × 3-trial.

Usage (after runs complete):
    cd /home/admin1/macvo-dev
    conda activate macvo
    python Scripts/analyse_v3bpp_phase1a_fd_e.py
"""

from __future__ import annotations

import csv, sys, yaml
from pathlib import Path
from collections import defaultdict

import numpy as np

WORKDIR = Path("/home/admin1/macvo-dev")
sys.path.insert(0, str(WORKDIR))

from Utility.Config import IncludeLoader
from Scripts.eval_qa_vif import evaluate_trajectory_direct

RESULT_ROOT = WORKDIR / "Results" / "v3bpp_phase1a_fd_e_grace30_5scene_3x"
OUTDIR = WORKDIR / "analysis_v3bpp_phase1a_fd_e_grace30_5scene_3x_report"
SCENES = ["moderate_turbidity", "murky_coast", "open_water", "open_water_overcast", "twilight_coast"]
SCENES_BY_MATCH = sorted(SCENES, key=len, reverse=True)
OLD7_BATCH = Path("/mnt/e/文档/holoocean/code/recordings/batch_20260515_155653")
NEW3_BATCH = Path("/mnt/e/文档/holoocean/code/recordings/batch_20260528_203401")


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)

    # ── Find all result dirs ──────────────────────────────────────
    runs = []
    proj_dirs = list(RESULT_ROOT.glob("*/"))
    for proj_dir in proj_dirs:
        for run_dir in sorted(proj_dir.glob("*/")):
            if not (run_dir / "poses.csv").exists():
                continue
            scene = "unknown"
            for s in SCENES_BY_MATCH:
                if s in run_dir.name:
                    scene = s
                    break
            if scene == "unknown":
                print(f"[WARN] Could not infer scene from run dir: {run_dir.name}")
            runs.append({"scene": scene, "run_dir": run_dir})

    print(f"Found {len(runs)} result directories.")

    # ── Collect all data ──────────────────────────────────────────
    all_data = []
    for r in runs:
        scene = r["scene"]
        run_dir = r["run_dir"]
        trial_id = run_dir.name[:17]

        # ATE
        batch = OLD7_BATCH if scene in ("murky_coast", "open_water") else NEW3_BATCH
        gt_path = batch / scene / "ref_pose.csv"
        try:
            result = evaluate_trajectory_direct(run_dir / "poses.csv", gt_path, f"{scene}_{trial_id}")
            ate = float(result["ate"]["ate_rmse"])
        except Exception:
            ate = float("nan")

        # Adaptive decisions
        ad_path = run_dir / "adaptive_decisions.csv"
        mode_timeline = defaultdict(int)
        full_imu_frames = 0
        cooldown_frames = 0
        fd_raw_count = 0
        fd_triggered_count = 0
        fd_suppressed_count = 0
        full_imu_episodes = []
        fd_trigger_pairs = []
        severe_vc_count = 0
        mild_vc_count = 0
        total_frames = 0
        first_fd_pair = -1
        nvis_vals = []

        if ad_path.exists():
            with open(ad_path) as f:
                reader = csv.DictReader(f)
                rows = [r for r in reader if int(r.get("pair_id", -1)) > 0]

            total_frames = len(rows)
            prev_state = ""
            cur_ep_len = 0
            for row in rows:
                state = row.get("state_name", "")
                mode_timeline[state] += 1

                if "full_imu" in state:
                    full_imu_frames += 1
                    cur_ep_len += 1
                else:
                    if cur_ep_len > 0:
                        full_imu_episodes.append(cur_ep_len)
                        cur_ep_len = 0

                if "cooldown" in state:
                    cooldown_frames += 1

                if row.get("full_divergence_raw", "0") == "1":
                    fd_raw_count += 1
                if row.get("full_divergence_triggered", "0") == "1":
                    fd_triggered_count += 1
                    fd_trigger_pairs.append(row.get("pair_id", "?"))
                if row.get("fd_check_suppressed_by_grace", "0") == "1":
                    fd_suppressed_count += 1
                if row.get("severe_vc_triggered", "0") == "1":
                    severe_vc_count += 1
                if row.get("mild_vc_triggered", "0") == "1":
                    mild_vc_count += 1

                nv = row.get("num_visual_residuals", "0")
                try:
                    nvis_vals.append(int(nv))
                except ValueError:
                    pass

                prev_state = state

            if cur_ep_len > 0:
                full_imu_episodes.append(cur_ep_len)

            if fd_trigger_pairs and first_fd_pair < 0:
                first_fd_pair = int(fd_trigger_pairs[0])

        all_data.append({
            "scene": scene,
            "trial_id": trial_id,
            "ate": ate,
            "total_frames": total_frames,
            "full_imu_frames": full_imu_frames,
            "cooldown_frames": cooldown_frames,
            "full_imu_episodes": full_imu_episodes,
            "num_episodes": len(full_imu_episodes),
            "fd_raw_count": fd_raw_count,
            "fd_triggered_count": fd_triggered_count,
            "fd_suppressed_count": fd_suppressed_count,
            "fd_trigger_pairs": fd_trigger_pairs,
            "first_fd_pair": first_fd_pair,
            "severe_vc_count": severe_vc_count,
            "mild_vc_count": mild_vc_count,
            "mode_timeline": dict(mode_timeline),
            "nvis_median": float(np.median(nvis_vals)) if nvis_vals else float("nan"),
        })

    # ═══════════════════════════════════════════════════════════════
    # CSV 1: phase1a_fd_e_summary.csv — per-trial raw data
    # ═══════════════════════════════════════════════════════════════
    with open(OUTDIR / "phase1a_fd_e_summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scene", "trial", "ATE", "total_frames", "full_imu_frames",
                    "cooldown_frames", "num_episodes", "episode_lengths",
                    "fd_raw_count", "fd_triggered_count", "fd_suppressed_count",
                    "fd_trigger_pairs", "severe_vc", "mild_vc", "nvis_median"])
        for d in sorted(all_data, key=lambda x: (SCENES.index(x["scene"]) if x["scene"] in SCENES else 99, x["trial_id"])):
            w.writerow([d["scene"], d["trial_id"], f"{d['ate']:.3f}" if not np.isnan(d["ate"]) else "N/A",
                       d["total_frames"], d["full_imu_frames"], d["cooldown_frames"],
                       d["num_episodes"], ",".join(str(x) for x in d["full_imu_episodes"]),
                       d["fd_raw_count"], d["fd_triggered_count"], d["fd_suppressed_count"],
                       ",".join(d["fd_trigger_pairs"][:10]),
                       d["severe_vc_count"], d["mild_vc_count"], f"{d['nvis_median']:.1f}"])

    # ═══════════════════════════════════════════════════════════════
    # CSV 2: phase1a_fd_e_scene_summary.csv — per-scene median stats
    # ═══════════════════════════════════════════════════════════════
    with open(OUTDIR / "phase1a_fd_e_scene_summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scene", "median_ATE", "mean_ATE", "std_ATE",
                    "median_full_imu_frames", "median_cooldown_frames",
                    "median_episodes", "median_fd_raw", "median_fd_triggered",
                    "median_fd_suppressed", "median_severe_vc", "median_mild_vc"])

        for scene in SCENES:
            scene_data = [d for d in all_data if d["scene"] == scene and not np.isnan(d["ate"])]
            if len(scene_data) < 3:
                continue
            ates = [d["ate"] for d in scene_data]
            w.writerow([scene,
                       f"{np.median(ates):.3f}", f"{np.mean(ates):.3f}", f"{np.std(ates):.3f}",
                       f"{np.median([d['full_imu_frames'] for d in scene_data]):.0f}",
                       f"{np.median([d['cooldown_frames'] for d in scene_data]):.0f}",
                       f"{np.median([d['num_episodes'] for d in scene_data]):.0f}",
                       f"{np.median([d['fd_raw_count'] for d in scene_data]):.0f}",
                       f"{np.median([d['fd_triggered_count'] for d in scene_data]):.0f}",
                       f"{np.median([d['fd_suppressed_count'] for d in scene_data]):.0f}",
                       f"{np.median([d['severe_vc_count'] for d in scene_data]):.0f}",
                       f"{np.median([d['mild_vc_count'] for d in scene_data]):.0f}"])

    # ═══════════════════════════════════════════════════════════════
    # CSV 3: phase1a_fd_e_fd_timeline.csv — FD trigger details
    # ═══════════════════════════════════════════════════════════════
    with open(OUTDIR / "phase1a_fd_e_fd_timeline.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scene", "trial", "first_fd_pair", "fd_triggered_count",
                    "fd_suppressed_count", "fd_raw_count", "fd_trigger_pairs"])
        for d in sorted(all_data, key=lambda x: (SCENES.index(x["scene"]) if x["scene"] in SCENES else 99, x["trial_id"])):
            w.writerow([d["scene"], d["trial_id"], d["first_fd_pair"],
                       d["fd_triggered_count"], d["fd_suppressed_count"],
                       d["fd_raw_count"], ",".join(d["fd_trigger_pairs"][:20])])

    # ═══════════════════════════════════════════════════════════════
    # CSV 4: phase1a_fd_e_mode_usage.csv — mode distribution
    # ═══════════════════════════════════════════════════════════════
    with open(OUTDIR / "phase1a_fd_e_mode_usage.csv", "w", newline="") as f:
        # Collect all unique state names
        all_states = set()
        for d in all_data:
            all_states.update(d["mode_timeline"].keys())
        all_states = sorted(all_states)

        w = csv.writer(f)
        w.writerow(["scene", "trial"] + all_states)
        for d in sorted(all_data, key=lambda x: (SCENES.index(x["scene"]) if x["scene"] in SCENES else 99, x["trial_id"])):
            row = [d["scene"], d["trial_id"]]
            for s in all_states:
                row.append(str(d["mode_timeline"].get(s, 0)))
            w.writerow(row)

    # ═══════════════════════════════════════════════════════════════
    # REPORT 5: phase1a_fd_e_report.md
    # ═══════════════════════════════════════════════════════════════
    # Known baselines for comparison (from previous analyses)
    BASELINES = {
        "moderate_turbidity":  {"ruleB_ate": 120.19, "oracle_ate": 46.13,  "oracle_method": "full_imu",       "goal": "reduce ATE towards 46m"},
        "murky_coast":         {"ruleB_ate": 19.25,  "oracle_ate": 8.90,   "oracle_method": "full_imu",       "goal": "no regression (>19.25m is FAIL)"},
        "open_water":          {"ruleB_ate": 225.41, "oracle_ate": 113.46, "oracle_method": "full_imu",       "goal": "no full_imu false positive"},
        "open_water_overcast": {"ruleB_ate": 19.08,  "oracle_ate": 19.04,  "oracle_method": "pure_macvo",     "goal": "zero VC, zero full_imu"},
        "twilight_coast":      {"ruleB_ate": 265.38, "oracle_ate": 167.91, "oracle_method": "translation_only","goal": "OBSERVATION ONLY"},
    }

    with open(OUTDIR / "phase1a_fd_e_report.md", "w") as f:
        f.write("# V3b++ Phase 1a: FD-E grace=30 — Analysis Report\n\n")
        f.write(f"**Date**: 2026-05-29\n")
        f.write(f"**Experiment**: FD-E grace period only (no cooldown changes, no translation_only)\n\n")

        f.write("## 1. Per-Scene ATE Summary\n\n")
        f.write("| Scene | T1 | T2 | T3 | Median | Rule B Baseline | Oracle | Δ vs Rule B | Judgment |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|---|\n")
        for scene in SCENES:
            scene_data = [d for d in all_data if d["scene"] == scene]
            ates = [d["ate"] for d in scene_data if not np.isnan(d["ate"])]
            if len(ates) >= 3:
                med = np.median(ates)
                bl = BASELINES.get(scene, {})
                rb = bl.get("ruleB_ate", float("nan"))
                oracle = bl.get("oracle_ate", float("nan"))
                delta = med - rb
                sign = "+" if delta > 0 else ""
                goal = bl.get("goal", "")

                if scene == "twilight_coast":
                    judgment = "OBSERVE ONLY"
                elif scene == "open_water_overcast":
                    judgment = "PASS" if med < 20 else "CHECK"
                elif scene == "open_water":
                    judgment = "SAFE" if med < 250 else "REGRESSION"
                elif scene == "murky_coast":
                    judgment = "SAFE" if med < 22 else "REGRESSION"
                elif scene == "moderate_turbidity":
                    if med < 80:
                        judgment = "IMPROVED"
                    elif med < 120:
                        judgment = "MARGINAL"
                    else:
                        judgment = "NO_CHANGE"
                else:
                    judgment = "CHECK"

                f.write(f"| {scene} | {ates[0]:.2f} | {ates[1]:.2f} | {ates[2]:.2f} | {med:.2f} | {rb:.2f} | {oracle:.2f} | {sign}{delta:.2f} | **{judgment}** |\n")

        f.write("\n## 2. Full-IMU Usage\n\n")
        f.write("| Scene | Median Full-IMU Frames | Median Episodes | Median Cooldown | FD Suppressed | FD Triggered |\n")
        f.write("|---|---:|---:|---:|---:|---:|\n")
        for scene in SCENES:
            scene_data = [d for d in all_data if d["scene"] == scene]
            f.write(f"| {scene} | {np.median([d['full_imu_frames'] for d in scene_data]):.0f} | {np.median([d['num_episodes'] for d in scene_data]):.0f} | {np.median([d['cooldown_frames'] for d in scene_data]):.0f} | {np.median([d['fd_suppressed_count'] for d in scene_data]):.0f} | {np.median([d['fd_triggered_count'] for d in scene_data]):.0f} |\n")

        f.write("\n## 3. Phase 1a Judgment Criteria\n\n")
        f.write("| Criterion | Scene | Threshold | Status |\n")
        f.write("|---|---:|---|\n")

        for scene in SCENES:
            scene_data = [d for d in all_data if d["scene"] == scene]
            if len(scene_data) < 3:
                continue
            ates = [d["ate"] for d in scene_data if not np.isnan(d["ate"])]
            med = np.median(ates) if ates else float("nan")

            if scene == "moderate_turbidity":
                status = "PASS" if med < 120 else "INCONCLUSIVE"
                f.write(f"| ATE < 120m (Rule B baseline) | {scene} | <120 | {status} ({med:.1f}m) |\n")
                status2 = "PASS" if med < 80 else "NOT_YET"
                f.write(f"| ATE approaching oracle (46m) | {scene} | <80 | {status2} ({med:.1f}m) |\n")
            elif scene == "murky_coast":
                status = "PASS" if med < 22 else "REGRESSION"
                f.write(f"| No regression (>22m) | {scene} | <22 | {status} ({med:.1f}m) |\n")
            elif scene == "open_water":
                status = "PASS" if med < 250 else "REGRESSION"
                f.write(f"| No full_imu false positive (<250m) | {scene} | <250 | {status} ({med:.1f}m) |\n")
            elif scene == "open_water_overcast":
                status = "PASS" if med < 20 else "CHECK"
                f.write(f"| Rotation_only maintained | {scene} | <20 | {status} ({med:.1f}m) |\n")
            elif scene == "twilight_coast":
                f.write(f"| Observation only | {scene} | N/A | {med:.1f}m |\n")

        f.write("\n---\n*End of Phase 1a report.*\n")

    print(f"\nAll outputs written to {OUTDIR}/")
    for fname in sorted(OUTDIR.glob("*")):
        print(f"  {fname.name}")


if __name__ == "__main__":
    main()
