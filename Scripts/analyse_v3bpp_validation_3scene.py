#!/usr/bin/env python3
"""
V3b++ Validation Analysis: 3 scenes, fixed baseline + Rule B + CP-B-FD-only.

Usage (after 54 runs complete):
    cd /home/admin1/macvo-dev
    conda activate macvo
    python Scripts/analyse_v3bpp_validation_3scene.py
"""

import csv, sys, yaml, numpy as np
from pathlib import Path
from collections import defaultdict

WORKDIR = Path("/home/admin1/macvo-dev")
sys.path.insert(0, str(WORKDIR))
from Utility.Config import IncludeLoader
from Scripts.eval_qa_vif import evaluate_trajectory_direct

RESULT_ROOT = WORKDIR / "Results/v3bpp_validation_3scene_54runs"
OUTDIR = WORKDIR / "analysis_v3bpp_validation_3scene_report"
BATCH = Path("/mnt/e/文档/holoocean/code/recordings/batch_20260601_162707")

SCENES = ["validation_moderate_harbor", "validation_transient_dropout", "validation_twilight_structure"]
METHODS_ORDER = ["pure_macvo", "rotation_only", "translation_only", "full_imu", "ruleB", "cpb_fd_only"]

SCENE_LABELS = {
    "validation_moderate_harbor": "V1: moderate harbor",
    "validation_transient_dropout": "V2: transient dropout",
    "validation_twilight_structure": "V3: twilight structure",
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
        # Backward-compatible fallback for older decision CSVs.
        if "rotation" in state:
            stats["rotation_only"] += 1
        elif "pure" in state:
            stats["pure_macvo"] += 1


def collect_runs():
    runs = []
    if not RESULT_ROOT.exists():
        print(f"Result dir not found: {RESULT_ROOT}")
        return runs

    for exp_name in ["fixed_baseline", "ruleB_baseline", "cpb_fd_only"]:
        exp_root = RESULT_ROOT / exp_name
        if not exp_root.exists(): continue
        for proj_dir in exp_root.glob("*/"):
            for run_dir in sorted(proj_dir.glob("*/")):
                if not (run_dir / "poses.csv").exists(): continue
                name = run_dir.name
                scene = "unknown"
                for s in sorted(SCENES, key=len, reverse=True):
                    if s in name: scene = s; break
                trial_id = name[:17]

                # Identify method
                config_path = run_dir / "config.yaml"
                method = exp_name  # default
                if exp_name == "fixed_baseline" and config_path.exists():
                    with open(config_path) as f:
                        cfg = yaml.load(f, IncludeLoader)
                    args = cfg["Odometry"]["args"]
                    imu_rot = args.get("imu_rot_prior_enable", False)
                    imu_trans = args.get("imu_trans_prior_enable", False)
                    if not imu_rot and not imu_trans: method = "pure_macvo"
                    elif imu_rot and not imu_trans: method = "rotation_only"
                    elif not imu_rot and imu_trans: method = "translation_only"
                    elif imu_rot and imu_trans: method = "full_imu"
                elif exp_name == "ruleB_baseline": method = "ruleB"
                elif exp_name == "cpb_fd_only": method = "cpb_fd_only"

                # ATE
                gt_path = BATCH / scene / "ref_pose.csv"
                try:
                    result = evaluate_trajectory_direct(run_dir / "poses.csv", gt_path, f"{scene}_{method}")
                    ate = float(result["ate"]["ate_rmse"])
                except:
                    ate = float("nan")

                # Adaptive decisions (for ruleB/cpb)
                ad_path = run_dir / "adaptive_decisions.csv"
                stats = {"full_imu":0, "cooldown_total":0, "cooldown_fd":0, "cooldown_rh":0,
                         "rotation_only":0, "pure_macvo":0, "translation_only_mode":0,
                         "fd_raw":0, "fd_trig":0, "fd_supp":0,
                         "severe_vc":0, "mild_vc":0, "vc_trig":0, "rh_trig":0,
                         "episodes":[], "total_frames":0,
                         "expected_missing_nan":0, "illegal":0, "pid_neg1":0, "trans_only":0}
                nvis_list = []
                if ad_path.exists():
                    with open(ad_path) as f:
                        rows = [r for r in csv.DictReader(f) if int(r.get("pair_id",-1))>0]
                    stats["total_frames"] = len(rows)
                    cur_ep = 0
                    for row in rows:
                        state = row.get("state_name","")
                        _count_adaptive_mode(row, stats)
                        if "full_imu" in state: stats["full_imu"]+=1; cur_ep+=1
                        else:
                            if cur_ep>0: stats["episodes"].append(cur_ep); cur_ep=0
                        if "cooldown" in state: stats["cooldown_total"]+=1
                        reason = row.get("cooldown_reason","")
                        if "full_div" in reason: stats["cooldown_fd"]+=1
                        elif "rot_harm" in reason: stats["cooldown_rh"]+=1
                        if _is_true(row.get("full_divergence_raw","0")): stats["fd_raw"]+=1
                        if _is_true(row.get("full_divergence_triggered","0")): stats["fd_trig"]+=1
                        if _is_true(row.get("fd_check_suppressed_by_grace","0")): stats["fd_supp"]+=1
                        if _is_true(row.get("severe_vc_triggered","0")): stats["severe_vc"]+=1
                        if _is_true(row.get("mild_vc_triggered","0")): stats["mild_vc"]+=1
                        if _is_true(row.get("visual_collapse_triggered","0")): stats["vc_trig"]+=1
                        if _is_true(row.get("rot_harm_triggered","0")): stats["rh_trig"]+=1
                        if _is_true(row.get("use_imu_translation","0")): stats["trans_only"]+=1
                        for k in ["r_p_whitened_norm","imu_trans_loss","median_flow_cov"]:
                            if row.get(k,"")=="nan": stats["expected_missing_nan"]+=1
                        try: nvis_list.append(int(row.get("num_visual_residuals","0")))
                        except: pass
                    if cur_ep>0: stats["episodes"].append(cur_ep)

                runs.append({"scene":scene, "method":method, "trial":trial_id, "ate":ate,
                            "nvis_median": float(np.median(nvis_list)) if nvis_list else float("nan"),
                            **stats})
    return runs


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    all_runs = collect_runs()
    print(f"Found {len(all_runs)} runs.")

    if not all_runs:
        print("No results. Run the experiment first."); return

    # CSV 1: trial_summary
    with open(OUTDIR/"validation_trial_summary.csv","w",newline="") as f:
        w=csv.writer(f)
        w.writerow(["scene","method","trial","ATE","full_imu","full_imu_pct","cooldown_total",
                    "cooldown_fd","cooldown_rh","rotation_only","pure_macvo",
                    "episodes","ep_lengths",
                    "fd_raw","fd_trig","severe_vc","mild_vc","vc_trig","rh_trig",
                    "expected_missing_nan","illegal","pid_neg1","trans_only","nvis_median"])
        for d in sorted(all_runs, key=lambda x:(SCENES.index(x["scene"]) if x["scene"] in SCENES else 99,
                                               METHODS_ORDER.index(x["method"]) if x["method"] in METHODS_ORDER else 99,
                                               x["trial"])):
            eps = d["episodes"]
            w.writerow([d["scene"],d["method"],d["trial"],f"{d['ate']:.4f}" if not np.isnan(d["ate"]) else "N/A",
                       d["full_imu"],f"{d['full_imu']/max(d['total_frames'],1)*100:.1f}",
                       d["cooldown_total"],d["cooldown_fd"],d["cooldown_rh"],
                       d["rotation_only"],d["pure_macvo"],
                       len(eps),",".join(str(x) for x in eps[:12]),
                       d["fd_raw"],d["fd_trig"],d["severe_vc"],d["mild_vc"],d["vc_trig"],d["rh_trig"],
                       d["expected_missing_nan"],d["illegal"],d["pid_neg1"],d["trans_only"],
                       f"{d['nvis_median']:.1f}" if not np.isnan(d["nvis_median"]) else "nan"])

    # CSV 2: scene_summary
    with open(OUTDIR/"validation_scene_summary.csv","w",newline="") as f:
        w=csv.writer(f)
        w.writerow(["scene","method","median_ATE","mean_ATE","std_ATE","cv",
                    "median_full_imu","median_cooldown","median_episodes","median_fd_trig"])
        for scene in SCENES:
            for method in METHODS_ORDER:
                sd=[d for d in all_runs if d["scene"]==scene and d["method"]==method and not np.isnan(d["ate"])]
                if len(sd)<3: continue
                ates=[d["ate"] for d in sd]; med=np.median(ates); mn=np.mean(ates); st=np.std(ates)
                cv=st/mn if mn>0 else 0
                w.writerow([scene,method,f"{med:.4f}",f"{mn:.4f}",f"{st:.4f}",f"{cv:.3f}",
                           f"{np.median([d['full_imu'] for d in sd]):.0f}",
                           f"{np.median([d['cooldown_total'] for d in sd]):.0f}",
                           f"{np.median([len(d['episodes']) for d in sd]):.0f}",
                           f"{np.median([d['fd_trig'] for d in sd]):.0f}"])

    # CSV 3: fixed_oracle
    with open(OUTDIR/"validation_fixed_oracle_by_scene.csv","w",newline="") as f:
        w=csv.writer(f)
        w.writerow(["scene","oracle_method","oracle_median_ATE",
                    "pure_macvo","rotation_only","translation_only","full_imu"])
        for scene in SCENES:
            row=[scene]; best=("",float("inf"))
            for m in ["pure_macvo","rotation_only","translation_only","full_imu"]:
                sd=[d for d in all_runs if d["scene"]==scene and d["method"]==m and not np.isnan(d["ate"])]
                med=np.median([d["ate"] for d in sd]) if len(sd)>=3 else float("nan")
                row.append(f"{med:.4f}" if not np.isnan(med) else "N/A")
                if len(sd)>=3 and med<best[1]: best=(m,med)
            w.writerow([scene,best[0],f"{best[1]:.4f}"]+row[1:])

    # CSV 4: adaptive_comparison
    with open(OUTDIR/"validation_adaptive_comparison.csv","w",newline="") as f:
        w=csv.writer(f)
        w.writerow(["scene","ruleB_median_ATE","cpb_median_ATE","delta_cpb_vs_ruleB",
                    "oracle_method","oracle_ATE","cpb_vs_oracle_ratio",
                    "cpb_full_imu_pct","cpb_cooldown","cpb_fd_trig",
                    "ruleB_full_imu_pct","ruleB_cooldown","ruleB_fd_trig","judgment"])
        for scene in SCENES:
            rb=[d for d in all_runs if d["scene"]==scene and d["method"]=="ruleB" and not np.isnan(d["ate"])]
            cb=[d for d in all_runs if d["scene"]==scene and d["method"]=="cpb_fd_only" and not np.isnan(d["ate"])]
            if len(rb)<3 or len(cb)<3: continue
            rb_med=np.median([d["ate"] for d in rb]); cb_med=np.median([d["ate"] for d in cb])
            delta=cb_med-rb_med
            # Oracle
            oracle_m=""; oracle_v=float("inf")
            for m in ["pure_macvo","rotation_only","translation_only","full_imu"]:
                sd=[d for d in all_runs if d["scene"]==scene and d["method"]==m and not np.isnan(d["ate"])]
                if len(sd)>=3:
                    med=np.median([d["ate"] for d in sd])
                    if med<oracle_v: oracle_v=med; oracle_m=m
            ratio=cb_med/oracle_v if oracle_v>0 else float("nan")

            if "moderate_harbor" in scene:
                judgment = "IMPROVED" if delta<-1 else ("MARGINAL" if delta<0 else "NO_IMPROVEMENT")
            elif "transient" in scene:
                judgment = "SAFE" if abs(delta)<1 else "CHECK"
            else:
                judgment = "OBSERVE"

            w.writerow([scene,f"{rb_med:.4f}",f"{cb_med:.4f}",f"{delta:+.4f}",
                       oracle_m,f"{oracle_v:.4f}",f"{ratio:.3f}",
                       f"{np.median([d['full_imu']/max(d['total_frames'],1)*100 for d in cb]):.1f}",
                       f"{np.median([d['cooldown_total'] for d in cb]):.0f}",
                       f"{np.median([d['fd_trig'] for d in cb]):.0f}",
                       f"{np.median([d['full_imu']/max(d['total_frames'],1)*100 for d in rb]):.1f}",
                       f"{np.median([d['cooldown_total'] for d in rb]):.0f}",
                       f"{np.median([d['fd_trig'] for d in rb]):.0f}",
                       judgment])

    # CSV 5/6/7: mode_usage, cooldown, episode
    for name, cols, extract in [
        ("validation_mode_usage.csv",
         ["scene","method","trial","full_imu_pct","cooldown_pct","pure_macvo_pct",
          "rotation_only_pct","translation_only_pct",
          "full_imu_frames","cooldown_frames","pure_macvo_frames","rotation_only_frames",
          "translation_only_frames","episodes","ep_lengths"],
         lambda d: [d["scene"],d["method"],d["trial"],
                   f"{d['full_imu']/max(d['total_frames'],1)*100:.1f}",
                   f"{d['cooldown_total']/max(d['total_frames'],1)*100:.1f}",
                   f"{d['pure_macvo']/max(d['total_frames'],1)*100:.1f}",
                   f"{d['rotation_only']/max(d['total_frames'],1)*100:.1f}",
                   f"{d['translation_only_mode']/max(d['total_frames'],1)*100:.1f}",
                   d["full_imu"],d["cooldown_total"],d["pure_macvo"],d["rotation_only"],
                   d["translation_only_mode"],
                   len(d["episodes"]),",".join(str(x) for x in d["episodes"][:12])]),
        ("validation_cooldown_timeline.csv",
         ["scene","method","trial","cooldown_total","cooldown_fd","cooldown_rh","fd_trig","rh_trig"],
         lambda d: [d["scene"],d["method"],d["trial"],d["cooldown_total"],d["cooldown_fd"],d["cooldown_rh"],d["fd_trig"],d["rh_trig"]]),
        ("validation_episode_summary.csv",
         ["scene","method","trial","num_episodes","ep_lengths","median_ep_len","min_ep_len","max_ep_len"],
         lambda d: [d["scene"],d["method"],d["trial"],len(d["episodes"]),
                   ",".join(str(x) for x in d["episodes"]),
                   f"{np.median(d['episodes']):.0f}" if d["episodes"] else "0",
                   f"{min(d['episodes'])}" if d["episodes"] else "0",
                   f"{max(d['episodes'])}" if d["episodes"] else "0"]),
    ]:
        with open(OUTDIR/name,"w",newline="") as f:
            w=csv.writer(f); w.writerow(cols)
            for d in sorted(all_runs, key=lambda x:(SCENES.index(x["scene"]) if x["scene"] in SCENES else 99,
                                                   METHODS_ORDER.index(x["method"]) if x["method"] in METHODS_ORDER else 99)):
                if d["total_frames"] <= 0:
                    continue
                w.writerow(extract(d))

    # REPORT
    with open(OUTDIR/"validation_report.md","w") as f:
        f.write("# V3b++ Validation Report — 3 Scenes\n\n")
        f.write("## 1. Fixed Oracle\n\n")
        f.write("| Scene | Oracle | Oracle ATE | pure_macvo | rotation_only | translation_only | full_imu |\n")
        f.write("|---|---:|---:|---:|---:|---:|\n")
        for scene in SCENES:
            row=f"| {scene} | "; best=("",float("inf"))
            vals={}
            for m in ["pure_macvo","rotation_only","translation_only","full_imu"]:
                sd=[d for d in all_runs if d["scene"]==scene and d["method"]==m and not np.isnan(d["ate"])]
                if len(sd)>=3:
                    med=np.median([d["ate"] for d in sd]); vals[m]=med
                    if med<best[1]: best=(m,med)
            row+=f"{best[0]} | {best[1]:.2f} | "
            for m in ["pure_macvo","rotation_only","translation_only","full_imu"]:
                row+=f"{vals.get(m,0):.2f} | "
            f.write(row+"\n")

        f.write("\n## 2. Adaptive Comparison\n\n")
        f.write("| Scene | Rule B | CP-B-FD-only | Δ | Oracle | CP-B vs Oracle | Judgment |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|\n")
        for scene in SCENES:
            rb=[d for d in all_runs if d["scene"]==scene and d["method"]=="ruleB" and not np.isnan(d["ate"])]
            cb=[d for d in all_runs if d["scene"]==scene and d["method"]=="cpb_fd_only" and not np.isnan(d["ate"])]
            if len(rb)<3 or len(cb)<3: continue
            rb_med=np.median([d["ate"] for d in rb]); cb_med=np.median([d["ate"] for d in cb])
            oracle_v=float("inf"); oracle_m=""
            for m in ["pure_macvo","rotation_only","translation_only","full_imu"]:
                sd=[d for d in all_runs if d["scene"]==scene and d["method"]==m and not np.isnan(d["ate"])]
                if len(sd)>=3:
                    med=np.median([d["ate"] for d in sd])
                    if med<oracle_v: oracle_v=med; oracle_m=m

            if "moderate_harbor" in scene: j="IMPROVED" if cb_med<rb_med else "CHECK"
            elif "transient" in scene: j="SAFE" if abs(cb_med-rb_med)<1 else "CHECK"
            else: j="OBSERVE"

            f.write(f"| {scene} | {rb_med:.2f} | {cb_med:.2f} | {cb_med-rb_med:+.2f} | {oracle_m} ({oracle_v:.2f}) | {cb_med/oracle_v:.2f}× | **{j}** |\n")

        f.write("\n## 3. Gate Behavior\n\n")
        f.write("| Scene | Method | Full-IMU% | Cooldown | Episodes | FD Trig | Sev VC |\n")
        f.write("|---|---|---:|---:|---:|---:|---:|\n")
        for scene in SCENES:
            for method in ["ruleB","cpb_fd_only"]:
                sd=[d for d in all_runs if d["scene"]==scene and d["method"]==method]
                if not sd: continue
                f.write(
                    f"| {scene} | {method} | "
                    f"{np.median([d['full_imu']/max(d['total_frames'],1)*100 for d in sd]):.1f}% | "
                    f"{np.median([d['cooldown_total'] for d in sd]):.0f} | "
                    f"{np.median([len(d['episodes']) for d in sd]):.0f} | "
                    f"{np.median([d['fd_trig'] for d in sd]):.0f} | "
                    f"{np.median([d['severe_vc'] for d in sd]):.0f} |\n"
                )

        f.write("\n## 4. Conclusion\n\n")
        f.write("*CP-B-FD-only is NOT final V3b++. Validation scenes are used as held-out diagnostics: safe cases are preserved, while failures expose entry-coverage limitations.*\n")
        f.write("*translation_only is observed only in fixed baseline; NOT added to adaptive candidate set.*\n")

    print(f"\nAll outputs → {OUTDIR}/")
    for fn in sorted(OUTDIR.glob("*")): print(f"  {fn.name}")


if __name__ == "__main__":
    main()
