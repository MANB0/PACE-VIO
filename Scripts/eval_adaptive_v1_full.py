#!/usr/bin/env python3
"""Evaluate adaptive_v1 full results and generate all reports."""
import csv, json, math, numpy as np, sys
from pathlib import Path
from datetime import datetime

ADAPTIVE_DIR = sys.argv[1] if len(sys.argv) > 1 else None
if ADAPTIVE_DIR is None:
    # auto-find latest
    candidates = sorted(Path("/home/admin1/macvo-dev/Results").glob("holoocean_adaptive_v1_*"))
    ADAPTIVE_DIR = str(candidates[-1]) if candidates else None
if ADAPTIVE_DIR is None:
    print("No adaptive results found"); sys.exit(1)

RDIR = Path(ADAPTIVE_DIR)
FIXED = Path("Results/holoocean_7x4_20260515_211733")
BATCH = Path("/mnt/e/文档/holoocean/code/recordings/batch_20260515_155653")
SCENES = ["turbid_harbor","clear_shallow","deep_dark","caustic_shallow","dam_inspection","murky_coast","open_water"]
METHODS = ["pure_macvo","rotation_only","translation_only","full_imu"]

# Load fixed ATEs
fixed_ate = {}
with open(FIXED/"evaluation_results.csv") as f:
    for row in csv.DictReader(f):
        try: fixed_ate[(row["scene"],row["method"])] = float(row["ate_rmse"])
        except: fixed_ate[(row["scene"],row["method"])] = float("nan")

# Load oracle
oracle = {}
with open(FIXED/"oracle_analysis/oracle_best_table.csv") as f:
    for row in csv.DictReader(f):
        oracle[row["scene"]] = {"best_m": row["oracle_best_method"], "best_ate": float(row["oracle_best_ATE"])}

# Evaluate adaptive_v1 ATE
from Scripts.eval_qa_vif import evaluate_trajectory
adaptive_ate = {}
for s in SCENES:
    d = RDIR / s
    poses = sorted(d.rglob("poses.csv"))
    ref = BATCH / s / "ref_pose.csv"
    if poses and ref.exists():
        try:
            m = evaluate_trajectory(poses[-1], ref, f"adapt_{s}")
            adaptive_ate[s] = float(m["ate"]["ate_rmse"])
        except:
            adaptive_ate[s] = float("nan")
    else:
        adaptive_ate[s] = float("nan")

# Mode usage stats
mode_stats = {}
window_stats = []
for s in SCENES:
    ad_csv = RDIR / s / "adaptive_decisions.csv"
    if not ad_csv.exists(): continue
    modes = []
    for row in csv.DictReader(open(ad_csv)):
        modes.append(row["mode"])
    total = len(modes)
    counts = {m: modes.count(m) for m in ["pure_macvo","rotation_only","translation_only","full_imu"]}
    mode_stats[s] = {"total": total, **counts}

    # Window analysis
    win_size = 100
    for w_start in range(0, total, win_size):
        w_end = min(w_start + win_size, total)
        window_modes = modes[w_start:w_end]
        window_counts = {m: window_modes.count(m) for m in ["pure_macvo","rotation_only","translation_only","full_imu"]}

        # Read window diagnostics
        diag = RDIR / s / "frame_pair_diagnostics.csv"
        diag_rows = list(csv.DictReader(open(diag))) if diag.exists() else []
        w_diag = diag_rows[w_start:w_end]

        vis_scores = []
        deg_scores = []
        v_losses = []
        n_resids = []
        for dr in w_diag:
            try: vis_scores.append(float(dr.get("visual_health_score", float("nan"))))
            except: pass
            try: deg_scores.append(float(dr.get("degeneracy_score", float("nan"))))
            except: pass
            try: v_losses.append(float(dr.get("visual_loss", float("nan"))))
            except: pass
            try: n_resids.append(float(dr.get("num_visual_residuals", float("nan"))))
            except: pass

        # First translation frame in window
        first_trans = -1
        for j, m in enumerate(window_modes):
            if m == "translation_only":
                first_trans = w_start + j
                break

        window_stats.append({
            "scene": s, "window_start": w_start, "window_end": w_end,
            "pure_count": window_counts.get("pure_macvo", 0),
            "rotation_count": window_counts.get("rotation_only", 0),
            "translation_count": window_counts.get("translation_only", 0),
            "full_count": window_counts.get("full_imu", 0),
            "mean_health": np.mean(vis_scores) if vis_scores else float("nan"),
            "mean_degeneracy": np.mean(deg_scores) if deg_scores else float("nan"),
            "median_v_loss": np.median(v_losses) if v_losses else float("nan"),
            "median_n_resid": np.median(n_resids) if n_resids else float("nan"),
            "first_trans_frame": first_trans,
        })

# Write window analysis
with open(RDIR/"adaptive_mode_usage_by_window.csv","w",newline="") as f:
    w = csv.DictWriter(f, fieldnames=window_stats[0].keys() if window_stats else [])
    w.writeheader()
    w.writerows(window_stats)

# Write mode usage table
with open(RDIR/"adaptive_mode_usage_table.csv","w",newline="") as f:
    w = csv.writer(f)
    w.writerow(["scene","total","pure_pct","rotation_pct","translation_pct","full_pct","most_used","first_trans_frame","trans_segments","mean_seg_len"])
    for s in SCENES:
        ms = mode_stats.get(s, {})
        tot = ms.get("total", 0)
        if tot == 0:
            w.writerow([s,0,0,0,0,0,"none","never",0,0])
            continue
        pcts = {m: ms.get(m,0)/tot*100 for m in ["pure_macvo","rotation_only","translation_only","full_imu"]}
        most = max(pcts, key=pcts.get)
        # Count translation segments
        ad_csv = RDIR / s / "adaptive_decisions.csv"
        modes = [r["mode"] for r in csv.DictReader(open(ad_csv))]
        trans_segs = 0; in_trans = False; seg_lens = []; cur_len = 0; first_trans = -1
        for i,m in enumerate(modes):
            if m == "translation_only":
                if not in_trans:
                    trans_segs += 1
                    if first_trans < 0: first_trans = i
                    in_trans = True
                    cur_len = 1
                else:
                    cur_len += 1
            else:
                if in_trans:
                    seg_lens.append(cur_len)
                    in_trans = False
        if in_trans: seg_lens.append(cur_len)
        w.writerow([s, tot, f"{pcts['pure_macvo']:.1f}", f"{pcts['rotation_only']:.1f}", f"{pcts['translation_only']:.1f}", f"{pcts['full_imu']:.1f}", most, first_trans, trans_segs, f"{np.mean(seg_lens):.1f}" if seg_lens else 0])

# adaptive_vs_baselines
with open(RDIR/"adaptive_vs_baselines.csv","w",newline="") as f:
    w = csv.writer(f)
    w.writerow(["scene","pure_ATE","rotation_ATE","translation_ATE","full_ATE","oracle_best","oracle_ATE","adaptive_ATE","adapt_vs_full_pct","adapt_oracle_gap","oracle_recovery","trans_pct","most_used","verdict"])
    for s in SCENES:
        pure = fixed_ate.get((s,"pure_macvo"), float("nan"))
        full = fixed_ate.get((s,"full_imu"), float("nan"))
        ad_ate = adaptive_ate.get(s, float("nan"))
        o = oracle.get(s, {"best_ate": float("nan")})
        gap = ad_ate - o["best_ate"] if not math.isnan(ad_ate) else float("nan")
        rec = (full - ad_ate)/(full - o["best_ate"]) if (full - o["best_ate"]) and abs(full - o["best_ate"]) > 1e-12 else float("nan")
        vs_full = (full - ad_ate)/full*100 if full else 0
        ms = mode_stats.get(s, {})
        tot = ms.get("total", 0)
        trans_pct = ms.get("translation_only", 0)/tot*100 if tot else 0
        most = max(ms, key=ms.get) if ms else "?"
        # Verdict
        if s == "clear_shallow" and trans_pct < 5 and not math.isnan(ad_ate) and ad_ate < full:
            verdict = "PASS_MINIMAL"
        elif math.isnan(ad_ate) or ad_ate > full:
            verdict = "FAIL"
        elif s == "open_water":
            # Check if translation triggered mid/late
            ad_csv = RDIR / s / "adaptive_decisions.csv"
            if ad_csv.exists():
                modes = [r["mode"] for r in csv.DictReader(open(ad_csv))]
                has_trans = "translation_only" in modes
                if not has_trans:
                    verdict = "FAIL_TRIGGER_OPEN_WATER"
                else:
                    first = modes.index("translation_only") if has_trans else -1
                    if first < 100:
                        verdict = "OK_EARLY_TRIGGER"
                    elif first < len(modes)*0.5:
                        verdict = "OK_DYNAMIC_TRIGGER"
                    else:
                        verdict = "OK_LATE_TRIGGER"
            else:
                verdict = "NO_DATA"
        else:
            if not math.isnan(ad_ate) and ad_ate < full:
                verdict = "PASS_MINIMAL"
            else:
                verdict = "FAIL"
        w.writerow([s, f"{pure:.4f}", f"{fixed_ate.get((s,'rotation_only'),0):.4f}", f"{fixed_ate.get((s,'translation_only'),0):.4f}", f"{full:.4f}", o["best_m"], f"{o['best_ate']:.4f}", f"{ad_ate:.4f}", f"{vs_full:.1f}", f"{gap:.4f}", f"{rec:.4f}", f"{trans_pct:.1f}", most, verdict])

# Print summary
print(f"Adaptive results in: {RDIR}")
for s in SCENES:
    ad = adaptive_ate.get(s, float("nan"))
    tr = mode_stats.get(s, {}).get("translation_only", 0)
    tot = mode_stats.get(s, {}).get("total", 1)
    print(f"  {s:20s}: ATE={ad:8.4f}  trans={tr}/{tot} ({tr/tot*100 if tot else 0:.1f}%)")
print(f"\nFiles: {RDIR}/adaptive_vs_baselines.csv, adaptive_mode_usage_table.csv, adaptive_mode_usage_by_window.csv")
