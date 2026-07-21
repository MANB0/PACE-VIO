#!/usr/bin/env python3
"""
V3b++ Phase 1b Unified Analysis — CP-B experiments.

Usage:
    cd /home/admin1/macvo-dev
    conda activate macvo
    python Scripts/analyse_v3bpp_phase1b_cpb.py
"""

import csv, sys, yaml, numpy as np
from pathlib import Path
from collections import defaultdict

WORKDIR = Path("/home/admin1/macvo-dev")
sys.path.insert(0, str(WORKDIR))
from Utility.Config import IncludeLoader
from Scripts.eval_qa_vif import evaluate_trajectory_direct

OUTDIR = WORKDIR / "analysis_v3bpp_phase1b_cpb_report"

# Experiment result roots
EXP_ROOTS = {
    "CP-B-FD-only": WORKDIR / "Results/v3bpp_phase1b_cpb_fdonly_5scene_3x",
    "FD-E+CP-B":    WORKDIR / "Results/v3bpp_phase1b_fd_e_plus_cpb_fdonly_5scene_3x",
}

SCENES = ["moderate_turbidity", "murky_coast", "open_water", "open_water_overcast", "twilight_coast"]
BATCH_OLD = Path("/mnt/e/文档/holoocean/code/recordings/batch_20260515_155653")
BATCH_NEW = Path("/mnt/e/文档/holoocean/code/recordings/batch_20260528_203401")

# Baselines from previous analyses
RULE_B = {
    "moderate_turbidity": 120.19, "murky_coast": 19.25, "open_water": 225.41,
    "open_water_overcast": 19.08, "twilight_coast": 265.38,
}
FD_E_ONLY = {
    "moderate_turbidity": 127.48, "murky_coast": 19.25, "open_water": 208.74,
    "open_water_overcast": 19.00, "twilight_coast": 295.41,
}
ORACLE = {
    "moderate_turbidity": 46.13, "murky_coast": 8.90, "open_water": 113.46,
    "open_water_overcast": 19.04, "twilight_coast": 167.91,
}


def _is_true(value):
    return str(value).strip().lower() in ("1", "true", "yes")


def _count_adaptive_mode(row, stats):
    """Count mutually exclusive adaptive modes from the decision CSV."""
    mode = row.get("adaptive_mode", "") or row.get("mode", "")
    state = row.get("state_name", "")

    if mode == "rotation_only":
        stats["rotation_only"] += 1
    elif mode == "pure_macvo":
        stats["pure_macvo"] += 1
    elif mode == "translation_only":
        stats["translation_only_mode"] += 1
    elif not mode:
        if "rotation" in state:
            stats["rotation_only"] += 1
        elif "pure" in state:
            stats["pure_macvo"] += 1


def collect_runs(exp_name, result_root):
    """Collect all runs from an experiment directory."""
    runs = []
    if not result_root.exists():
        print(f"  [WARN] {exp_name}: result dir not found at {result_root}")
        return runs

    for proj_dir in result_root.glob("*/"):
        for run_dir in sorted(proj_dir.glob("*/")):
            if not (run_dir / "poses.csv").exists():
                continue
            name = run_dir.name
            scene = "unknown"
            for s in sorted(SCENES, key=len, reverse=True):
                if s in name:
                    scene = s
                    break
            trial_id = name[:17]

            batch = BATCH_OLD if scene in ("murky_coast", "open_water") else BATCH_NEW
            gt_path = batch / scene / "ref_pose.csv"
            try:
                result = evaluate_trajectory_direct(run_dir / "poses.csv", gt_path, f"{scene}_{trial_id}")
                ate = float(result["ate"]["ate_rmse"])
            except:
                ate = float("nan")

            ad_path = run_dir / "adaptive_decisions.csv"
            stats = {"full_imu":0, "cooldown_fd":0, "cooldown_rh":0, "cooldown_total":0,
                     "rotation_only":0, "pure_macvo":0, "translation_only_mode":0,
                     "fd_raw":0, "fd_trig":0, "fd_supp":0,
                     "severe_vc":0, "mild_vc":0, "episodes":[], "total_frames":0,
                     "fd_trigger_pairs":[], "nvis_median":float("nan")}
            nvis_list = []

            if ad_path.exists():
                with open(ad_path) as f:
                    rows = [r for r in csv.DictReader(f) if int(r.get("pair_id",-1))>0]
                stats["total_frames"] = len(rows)
                cur_ep = 0
                for row in rows:
                    state = row.get("state_name","")
                    _count_adaptive_mode(row, stats)
                    if "full_imu" in state:
                        stats["full_imu"] += 1
                        cur_ep += 1
                    else:
                        if cur_ep > 0:
                            stats["episodes"].append(cur_ep)
                            cur_ep = 0
                    if "cooldown" in state:
                        stats["cooldown_total"] += 1
                        reason = row.get("cooldown_reason","")
                        if "full_div" in reason: stats["cooldown_fd"] += 1
                        elif "rot_harm" in reason: stats["cooldown_rh"] += 1
                    if _is_true(row.get("full_divergence_raw","0")): stats["fd_raw"]+=1
                    if _is_true(row.get("full_divergence_triggered","0")):
                        stats["fd_trig"]+=1
                        stats["fd_trigger_pairs"].append(int(row.get("pair_id",0)))
                    if _is_true(row.get("fd_check_suppressed_by_grace","0")): stats["fd_supp"]+=1
                    if _is_true(row.get("severe_vc_triggered","0")): stats["severe_vc"]+=1
                    if _is_true(row.get("mild_vc_triggered","0")): stats["mild_vc"]+=1
                    try: nvis_list.append(int(row.get("num_visual_residuals","0")))
                    except: pass
                if cur_ep > 0: stats["episodes"].append(cur_ep)
                if nvis_list: stats["nvis_median"] = float(np.median(nvis_list))

            runs.append({"exp": exp_name, "scene": scene, "trial": trial_id, "ate": ate, **stats})
    return runs


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    all_runs = []
    for exp_name, root in EXP_ROOTS.items():
        runs = collect_runs(exp_name, root)
        all_runs.extend(runs)
        print(f"  {exp_name}: {len(runs)} runs")

    if not all_runs:
        print("No results found. Run experiments first.")
        return

    # ═══════════════════════════════════════════════════════════
    # CSV 1: trial_summary
    # ═══════════════════════════════════════════════════════════
    with open(OUTDIR/"phase1b_trial_summary.csv","w",newline="") as f:
        w=csv.writer(f)
        w.writerow(["exp","scene","trial","ATE","full_imu_frames","full_imu_pct",
                    "cooldown_total","cooldown_fd","cooldown_rh",
                    "rotation_only_frames","pure_macvo_frames",
                    "episodes","ep_lengths","fd_raw","fd_trig","fd_supp",
                    "severe_vc","mild_vc","nvis_median"])
        for d in sorted(all_runs, key=lambda x:(x["exp"],SCENES.index(x["scene"]) if x["scene"] in SCENES else 99,x["trial"])):
            w.writerow([d["exp"],d["scene"],d["trial"],f"{d['ate']:.3f}" if not np.isnan(d["ate"]) else "N/A",
                       d["full_imu"],f"{d['full_imu']/max(d['total_frames'],1)*100:.1f}",
                       d["cooldown_total"],d["cooldown_fd"],d["cooldown_rh"],
                       d["rotation_only"],d["pure_macvo"],
                       len(d["episodes"]),",".join(str(x) for x in d["episodes"][:10]),
                       d["fd_raw"],d["fd_trig"],d["fd_supp"],
                       d["severe_vc"],d["mild_vc"],f"{d['nvis_median']:.1f}"])

    # ═══════════════════════════════════════════════════════════
    # CSV 2: scene_summary — with comparison columns
    # ═══════════════════════════════════════════════════════════
    with open(OUTDIR/"phase1b_scene_summary.csv","w",newline="") as f:
        w=csv.writer(f)
        w.writerow(["exp","scene","median_ATE","mean_ATE","std_ATE",
                    "median_full_imu","median_cooldown","median_episodes",
                    "median_fd_trig","median_fd_supp",
                    "ruleB_ATE","fd_e_only_ATE","oracle_ATE",
                    "delta_vs_ruleB","delta_vs_fd_e"])

        for exp_name in EXP_ROOTS:
            for scene in SCENES:
                sd=[d for d in all_runs if d["exp"]==exp_name and d["scene"]==scene and not np.isnan(d["ate"])]
                if len(sd)<3: continue
                ates=[d["ate"] for d in sd]
                med=np.median(ates)
                rb=RULE_B.get(scene,float("nan"))
                fe=FD_E_ONLY.get(scene,float("nan"))
                ora=ORACLE.get(scene,float("nan"))
                w.writerow([exp_name,scene,
                           f"{med:.3f}",f"{np.mean(ates):.3f}",f"{np.std(ates):.3f}",
                           f"{np.median([d['full_imu'] for d in sd]):.0f}",
                           f"{np.median([d['cooldown_total'] for d in sd]):.0f}",
                           f"{np.median([len(d['episodes']) for d in sd]):.0f}",
                           f"{np.median([d['fd_trig'] for d in sd]):.0f}",
                           f"{np.median([d['fd_supp'] for d in sd]):.0f}",
                           f"{rb:.2f}",f"{fe:.2f}",f"{ora:.2f}",
                           f"{med-rb:+.2f}",f"{med-fe:+.2f}"])

    # ═══════════════════════════════════════════════════════════
    # CSV 3: mode_usage
    # ═══════════════════════════════════════════════════════════
    with open(OUTDIR/"phase1b_mode_usage.csv","w",newline="") as f:
        w=csv.writer(f)
        w.writerow(["exp","scene","trial","full_imu_pct","cooldown_pct","rotation_only_pct",
                    "pure_macvo_pct","translation_only_pct",
                    "full_imu_frames","cooldown_frames","rotation_only_frames",
                    "pure_macvo_frames","translation_only_frames","episodes","ep_lengths"])
        for d in sorted(all_runs, key=lambda x:(x["exp"],SCENES.index(x["scene"]) if x["scene"] in SCENES else 99)):
            tf=max(d["total_frames"],1)
            w.writerow([d["exp"],d["scene"],d["trial"],
                       f"{d['full_imu']/tf*100:.1f}",f"{d['cooldown_total']/tf*100:.1f}",
                       f"{d['rotation_only']/tf*100:.1f}",
                       f"{d['pure_macvo']/tf*100:.1f}",
                       f"{d['translation_only_mode']/tf*100:.1f}",
                       d["full_imu"],d["cooldown_total"],d["rotation_only"],
                       d["pure_macvo"],d["translation_only_mode"],
                       len(d["episodes"]),",".join(str(x) for x in d["episodes"][:10])])

    # ═══════════════════════════════════════════════════════════
    # CSV 4: cooldown_timeline
    # ═══════════════════════════════════════════════════════════
    with open(OUTDIR/"phase1b_cooldown_timeline.csv","w",newline="") as f:
        w=csv.writer(f)
        w.writerow(["exp","scene","trial","cooldown_total","cooldown_fd_frames",
                    "cooldown_rh_frames","fd_trigger_pairs"])
        for d in sorted(all_runs, key=lambda x:(x["exp"],SCENES.index(x["scene"]) if x["scene"] in SCENES else 99,x["trial"])):
            w.writerow([d["exp"],d["scene"],d["trial"],d["cooldown_total"],
                       d["cooldown_fd"],d["cooldown_rh"],
                       ",".join(str(x) for x in d["fd_trigger_pairs"][:15])])

    # ═══════════════════════════════════════════════════════════
    # CSV 5: episode_summary
    # ═══════════════════════════════════════════════════════════
    with open(OUTDIR/"phase1b_episode_summary.csv","w",newline="") as f:
        w=csv.writer(f)
        w.writerow(["exp","scene","trial","num_episodes","ep_lengths","median_ep_len",
                    "min_ep_len","max_ep_len","fd_trig_total"])
        for d in sorted(all_runs, key=lambda x:(x["exp"],SCENES.index(x["scene"]) if x["scene"] in SCENES else 99)):
            eps=d["episodes"]
            if not eps: eps=[0]
            w.writerow([d["exp"],d["scene"],d["trial"],len(eps),
                       ",".join(str(x) for x in eps),
                       f"{np.median(eps):.0f}",f"{min(eps)}",f"{max(eps)}",
                       d["fd_trig"]])

    # ═══════════════════════════════════════════════════════════
    # REPORT: phase1b_report.md
    # ═══════════════════════════════════════════════════════════
    with open(OUTDIR/"phase1b_report.md","w") as f:
        f.write("# V3b++ Phase 1b: CP-B Cooldown Experiment — Report\n\n")
        f.write("## 1. ATE Comparison\n\n")
        f.write("| Scene | Rule B | FD-E only | CP-B-FD-only | FD-E+CP-B | Oracle | Best vs Rule B |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|\n")

        for scene in SCENES:
            row=f"| {scene} | {RULE_B.get(scene,0):.2f} | {FD_E_ONLY.get(scene,0):.2f} | "
            for exp_name in EXP_ROOTS:
                sd=[d for d in all_runs if d["exp"]==exp_name and d["scene"]==scene and not np.isnan(d["ate"])]
                if len(sd)>=3:
                    row+=f"{np.median([d['ate'] for d in sd]):.2f} | "
                else:
                    row+="N/A | "
            row+=f"{ORACLE.get(scene,0):.2f} | "
            # Best among CP-B and FD-E+CP-B
            all_meds=[]
            for exp_name in EXP_ROOTS:
                sd=[d for d in all_runs if d["exp"]==exp_name and d["scene"]==scene and not np.isnan(d["ate"])]
                if len(sd)>=3: all_meds.append(np.median([d['ate'] for d in sd]))
            if all_meds:
                best=min(all_meds)
                rb=RULE_B.get(scene,0)
                delta=best-rb
                sign="+" if delta>0 else ""
                row+=f"{sign}{delta:.2f} |"
            else:
                row+="N/A |"
            f.write(row+"\n")

        f.write("\n## 2. Key Question: Is cooldown the main bottleneck?\n\n")
        # moderate_turbidity focus
        for scene in ["moderate_turbidity"]:
            sd_ruleB = RULE_B.get(scene, 0)
            for exp_name in EXP_ROOTS:
                sd=[d for d in all_runs if d["exp"]==exp_name and d["scene"]==scene and not np.isnan(d["ate"])]
                if len(sd)>=3:
                    med=np.median([d['ate'] for d in sd])
                    f.write(f"- **{exp_name}**: {med:.2f}m (Rule B: {sd_ruleB:.2f}m, Δ={med-sd_ruleB:+.2f}m)\n")
                    f.write(
                        f"  - Median Full-IMU: "
                        f"{np.median([d['full_imu'] for d in sd]):.0f} frames "
                        f"({np.median([d['full_imu']/max(d['total_frames'],1)*100 for d in sd]):.1f}%)\n"
                    )
                    f.write(f"  - Median cooldown: {np.median([d['cooldown_total'] for d in sd]):.0f} frames\n")
                    f.write(f"  - Median episodes: {np.median([len(d['episodes']) for d in sd]):.0f}\n")

        f.write("\n## 3. Regression Guards\n\n")
        for scene in ["murky_coast","open_water","open_water_overcast"]:
            sd_ruleB=RULE_B.get(scene,0)
            for exp_name in EXP_ROOTS:
                sd=[d for d in all_runs if d["exp"]==exp_name and d["scene"]==scene and not np.isnan(d["ate"])]
                if len(sd)>=3:
                    med=np.median([d['ate'] for d in sd])
                    status="SAFE" if abs(med-sd_ruleB)<3 else "CHECK"
                    f.write(f"- {exp_name} {scene}: {med:.2f}m (vs {sd_ruleB:.2f}) → {status}\n")

        f.write("\n---\n*End of Phase 1b report.*\n")

    print(f"\nAll outputs → {OUTDIR}/")
    for fn in sorted(OUTDIR.glob("*")): print(f"  {fn.name}")


if __name__ == "__main__":
    main()
