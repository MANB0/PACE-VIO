#!/usr/bin/env python3
"""
V3b++ Offline Fitting Dataset Builder.

Extracts pair-level, episode-level, and trigger-level features from existing
experiment logs for future FD Controller v1 / adaptive formula development.

Does NOT train models, select formulas, or modify online MACVO logic.

Usage:
    cd /home/admin1/macvo-dev
    conda activate macvo
    python Scripts/build_v3bpp_fitting_dataset.py
"""

import csv, sys, yaml, numpy as np
from pathlib import Path
from collections import defaultdict

WORKDIR = Path("/home/admin1/macvo-dev")
sys.path.insert(0, str(WORKDIR))
from Utility.Config import IncludeLoader
from Scripts.eval_qa_vif import evaluate_trajectory_direct

OUTDIR = WORKDIR / "analysis_v3bpp_fitting_dataset"

# ── Input experiments (auto-detect, skip missing) ──────────────
EXP_SOURCES = [
    ("Rule_B_7x3", WORKDIR / "Results/v3bplus_ruleB_7x3_20260521_235803", "trial_{trial}/{scene}"),
    ("Rule_B_holdout", WORKDIR / "Results/holdout_validation/ruleB/MACVO-HoloOcean-IMU@holoocean_imu", ""),
    ("Phase1a_FD_E", WORKDIR / "Results/v3bpp_phase1a_fd_e_grace30_5scene_3x/MACVO-HoloOcean-IMU@holoocean_imu", ""),
    ("Phase1b_CPB_FD_only", WORKDIR / "Results/v3bpp_phase1b_cpb_fdonly_5scene_3x/MACVO-HoloOcean-IMU@holoocean_imu", ""),
    ("Phase1b_FD_E_CPB", WORKDIR / "Results/v3bpp_phase1b_fd_e_plus_cpb_fdonly_5scene_3x/MACVO-HoloOcean-IMU@holoocean_imu", ""),
    ("Candidate_CPB_7scene_regression", WORKDIR / "Results/v3bpp_candidate_cpb_fdonly_7scene_3x/MACVO-HoloOcean-IMU@holoocean_imu", ""),
]

SCENES_OLD = ["turbid_harbor","clear_shallow","deep_dark","caustic_shallow","dam_inspection","murky_coast","open_water"]
SCENES_NEW = ["moderate_turbidity","open_water_overcast","twilight_coast"]
ALL_SCENES = SCENES_OLD + SCENES_NEW
BATCH_OLD = Path("/mnt/e/文档/holoocean/code/recordings/batch_20260515_155653")
BATCH_NEW = Path("/mnt/e/文档/holoocean/code/recordings/batch_20260528_203401")


def identify_scene(name):
    for s in sorted(ALL_SCENES, key=len, reverse=True):
        if s in name: return s
    return "unknown"


def compute_ate(run_dir, scene):
    batch = BATCH_OLD if scene in SCENES_OLD else BATCH_NEW
    gt = batch / scene / "ref_pose.csv"
    if not gt.exists(): return float("nan")
    poses = run_dir / "poses.csv"
    if not poses.exists(): return float("nan")
    try:
        r = evaluate_trajectory_direct(poses, gt, f"{scene}")
        return float(r["ate"]["ate_rmse"])
    except:
        return float("nan")


def collect_all_runs():
    """Scan all experiment directories and return list of (exp_name, run_dir, scene, trial)."""
    found = []
    for exp_name, root, pattern in EXP_SOURCES:
        if not root.exists():
            print(f"  [SKIP] {exp_name}: dir not found ({root})")
            continue

        if "{trial}" in pattern:
            # 7x3 style: trial_1/scene/, trial_2/scene/, etc
            for trial_d in sorted(root.glob("trial_*")):
                trial_name = trial_d.name
                for scene_d in sorted(trial_d.glob("*/")):
                    scene = identify_scene(scene_d.name)
                    if scene == "unknown": continue
                    # Look for MACVO-HoloOcean-IMU subdir or adaptive_decisions directly
                    ad_csv = scene_d / "adaptive_decisions.csv"
                    if not ad_csv.exists():
                        sub = list(scene_d.glob("MACVO-HoloOcean-IMU@holoocean_imu/*/"))
                        if sub: scene_d = sub[0]
                    found.append((exp_name, scene_d, scene, trial_name))
        else:
            # Flat style: all result dirs in one folder
            for run_dir in sorted(root.glob("*/")):
                scene = identify_scene(run_dir.name)
                if scene == "unknown": continue
                trial_id = run_dir.name[:17]
                found.append((exp_name, run_dir, scene, trial_id))

    print(f"Collected {len(found)} run directories.")
    return found


def sliding_window(vals, w):
    """Return list of (median, slope) for sliding window of size w."""
    if len(vals) < w: return []
    results = []
    for i in range(len(vals) - w + 1):
        win = vals[i:i+w]
        med = np.nanmedian(win)
        slope = (win[-1] - win[0]) / max(w-1, 1) if w > 1 else 0
        results.append((med, slope))
    return results


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    all_runs = collect_all_runs()
    if not all_runs:
        print("No runs found."); return

    pair_rows = []
    episode_rows = []
    fd_trigger_rows = []
    cooldown_rows = []
    outcome_rows = []

    for exp_name, run_dir, scene, trial_id in all_runs:
        ad_csv = run_dir / "adaptive_decisions.csv"
        if not ad_csv.exists():
            ad_csv2 = run_dir / ".." / "adaptive_decisions.csv"
            if ad_csv2.exists(): ad_csv = ad_csv2
        if not ad_csv.exists():
            print(f"  [WARN] No adaptive_decisions.csv for {exp_name}/{scene}/{trial_id}")
            continue

        ate = compute_ate(run_dir, scene)

        with open(ad_csv) as f:
            rows_raw = list(csv.DictReader(f))

        # Filter valid rows
        valid = [r for r in rows_raw if int(r.get("pair_id", -1)) > 0]

        # ── A. Pair-level features ──────────────────────────────
        for row in valid:
            pid = int(row.get("pair_id", 0))
            pair_rows.append({
                "experiment_name": exp_name, "scene": scene, "trial": trial_id,
                "pair_id": pid,
                "mode": row.get("state_name", ""),
                "num_visual_residuals": int(row.get("num_visual_residuals", 0)),
                "severe_vc_raw": int(row.get("severe_vc_raw", "0")),
                "mild_vc_raw": int(row.get("mild_vc_raw", "0")),
                "visual_collapse_triggered": int(row.get("visual_collapse_triggered", "0")),
                "r_p_whitened_norm": float(row.get("r_p_whitened_norm", "nan")) if row.get("r_p_whitened_norm","nan")!="nan" else float("nan"),
                "imu_trans_loss": float(row.get("imu_trans_loss", "nan")) if row.get("imu_trans_loss","nan")!="nan" else float("nan"),
                "full_divergence_raw": int(row.get("full_divergence_raw", "0")),
                "full_divergence_triggered": int(row.get("full_divergence_triggered", "0")),
                "r_R_whitened_norm": float(row.get("r_R_whitened_norm", "nan")) if row.get("r_R_whitened_norm","nan")!="nan" else float("nan"),
                "median_flow_cov": float(row.get("median_flow_cov", "nan")) if row.get("median_flow_cov","nan")!="nan" else float("nan"),
                "rot_harm_triggered": int(row.get("rot_harm_triggered", "0")),
                "cooldown_active": int(row.get("full_divergence_cooldown_remaining", "0")) > 0 or int(row.get("rot_harm_cooldown_remaining", "0")) > 0,
                "cooldown_remaining": int(row.get("full_divergence_cooldown_remaining", "0")),
                "cooldown_reason": row.get("cooldown_reason", ""),
                "fd_cooldown_config": int(row.get("fd_cooldown_config", "100")),
                "rot_harm_cooldown_config": int(row.get("rot_harm_cooldown_config", "100")),
                "fd_grace_enabled": int(row.get("fd_grace_enabled", "0")),
                "fd_check_suppressed_by_grace": int(row.get("fd_check_suppressed_by_grace", "0")),
                "full_imu_episode_frame_idx": int(row.get("full_imu_episode_frame_idx", "0")),
                "est_delta_t_norm": float(row.get("est_delta_t_norm", "nan")) if row.get("est_delta_t_norm","nan")!="nan" else float("nan"),
                "run_ATE": ate if not np.isnan(ate) else "",
            })

        # ── B. Full-IMU episode features ────────────────────────
        episodes = []
        cur = []
        for row in valid:
            if "full_imu" in row.get("state_name", ""):
                cur.append(row)
            else:
                if cur: episodes.append(cur); cur = []
        if cur: episodes.append(cur)

        for ep_idx, ep in enumerate(episodes):
            start_p = int(ep[0]["pair_id"])
            end_p = int(ep[-1]["pair_id"])
            ep_len = len(ep)
            rp_vals = [float(r["r_p_whitened_norm"]) for r in ep if r.get("r_p_whitened_norm","nan")!="nan"]
            imu_vals = [float(r["imu_trans_loss"]) for r in ep if r.get("imu_trans_loss","nan")!="nan"]
            dt_vals = [float(r.get("est_delta_t_norm","nan")) for r in ep if r.get("est_delta_t_norm","nan")!="nan"]

            fd_raw_c = sum(1 for r in ep if r.get("full_divergence_raw","0")=="1")
            fd_trig_c = sum(1 for r in ep if r.get("full_divergence_triggered","0")=="1")
            fd_trig_rel = next((int(r["pair_id"])-start_p for r in ep if r.get("full_divergence_triggered","0")=="1"), -1)

            exited_by_fd = any(r.get("full_divergence_triggered","0")=="1" for r in ep)
            exited_by_rh = any(r.get("rot_harm_triggered","0")=="1" for r in ep)

            # Next cooldown length: check rows after this episode
            next_cd_len = 0
            ep_end_idx = valid.index(ep[-1])
            for r in valid[ep_end_idx+1:]:
                if "cooldown" in r.get("state_name",""):
                    next_cd_len += 1
                else:
                    break

            episode_rows.append({
                "experiment_name": exp_name, "scene": scene, "trial": trial_id,
                "episode_id": ep_idx, "start_pair": start_p, "end_pair": end_p,
                "episode_len": ep_len,
                "fd_grace_enabled": int(ep[0].get("fd_grace_enabled","0")),
                "fd_cooldown_config": int(ep[0].get("fd_cooldown_config","100")),
                "r_p_baseline_median_first_10": np.nanmedian(rp_vals[:10]) if len(rp_vals)>=10 else (np.nanmedian(rp_vals) if rp_vals else float("nan")),
                "r_p_baseline_median_first_30": np.nanmedian(rp_vals[:30]) if len(rp_vals)>=30 else (np.nanmedian(rp_vals) if rp_vals else float("nan")),
                "imu_loss_baseline_median_first_10": np.nanmedian(imu_vals[:10]) if len(imu_vals)>=10 else (np.nanmedian(imu_vals) if imu_vals else float("nan")),
                "imu_loss_baseline_median_first_30": np.nanmedian(imu_vals[:30]) if len(imu_vals)>=30 else (np.nanmedian(imu_vals) if imu_vals else float("nan")),
                "est_delta_t_baseline_median_first_10": np.nanmedian(dt_vals[:10]) if len(dt_vals)>=10 else (np.nanmedian(dt_vals) if dt_vals else float("nan")),
                "est_delta_t_baseline_median_first_30": np.nanmedian(dt_vals[:30]) if len(dt_vals)>=30 else (np.nanmedian(dt_vals) if dt_vals else float("nan")),
                "r_p_end_median_last_10": np.nanmedian(rp_vals[-10:]) if len(rp_vals)>=10 else (np.nanmedian(rp_vals) if rp_vals else float("nan")),
                "imu_loss_end_median_last_10": np.nanmedian(imu_vals[-10:]) if len(imu_vals)>=10 else (np.nanmedian(imu_vals) if imu_vals else float("nan")),
                "est_delta_t_end_median_last_10": np.nanmedian(dt_vals[-10:]) if len(dt_vals)>=10 else (np.nanmedian(dt_vals) if dt_vals else float("nan")),
                "fd_raw_count": fd_raw_c, "fd_triggered_count": fd_trig_c,
                "fd_trigger_pair_relative": fd_trig_rel,
                "exited_by_fd": int(exited_by_fd),
                "exited_by_rot_harm": int(exited_by_rh),
                "exited_by_other": int(not exited_by_fd and not exited_by_rh),
                "next_cooldown_len": next_cd_len,
                "run_ATE": ate if not np.isnan(ate) else "",
            })

        # ── C. FD trigger window features ───────────────────────
        fd_triggers = [(i, r) for i, r in enumerate(valid) if r.get("full_divergence_triggered","0")=="1"]
        for trig_idx, (vi, row) in enumerate(fd_triggers):
            trigger_pair = int(row["pair_id"])
            mode_at_trig = row.get("state_name","")
            # Find which episode this trigger belongs to
            ep_id = -1
            for ep_i, ep in enumerate(episodes):
                if ep[0] is row or (id(ep[0]) < id(row) and id(ep[-1]) >= id(row)):
                    ep_id = ep_i
                    break

            # Window stats
            rp_all = [float(r["r_p_whitened_norm"]) for r in valid if r.get("r_p_whitened_norm","nan")!="nan"]
            imu_all = [float(r["imu_trans_loss"]) for r in valid if r.get("imu_trans_loss","nan")!="nan"]
            dt_all = [float(r.get("est_delta_t_norm","nan")) for r in valid if r.get("est_delta_t_norm","nan")!="nan"]

            w10_rp = rp_all[max(0,vi-9):vi+1] if vi < len(rp_all) else []
            w10_imu = imu_all[max(0,vi-9):vi+1] if vi < len(imu_all) else []
            w10_dt = dt_all[max(0,vi-9):vi+1] if vi < len(dt_all) else []
            w20_rp = rp_all[max(0,vi-19):vi+1] if vi < len(rp_all) else []
            w20_imu = imu_all[max(0,vi-19):vi+1] if vi < len(imu_all) else []

            cd_after = 0
            for r in valid[vi:]:
                if int(r.get("full_divergence_cooldown_remaining","0"))>0: cd_after+=1
                else: break

            fd_trigger_rows.append({
                "experiment_name": exp_name, "scene": scene, "trial": trial_id,
                "trigger_pair": trigger_pair,
                "mode_at_trigger": mode_at_trig, "episode_id": ep_id,
                "window_10_r_p_median": np.nanmedian(w10_rp) if w10_rp else float("nan"),
                "window_10_r_p_slope": (w10_rp[-1]-w10_rp[0])/max(len(w10_rp)-1,1) if len(w10_rp)>=2 else 0,
                "window_10_imu_loss_median": np.nanmedian(w10_imu) if w10_imu else float("nan"),
                "window_10_imu_loss_slope": (w10_imu[-1]-w10_imu[0])/max(len(w10_imu)-1,1) if len(w10_imu)>=2 else 0,
                "window_10_est_delta_t_median": np.nanmedian(w10_dt) if w10_dt else float("nan"),
                "window_20_r_p_median": np.nanmedian(w20_rp) if w20_rp else float("nan"),
                "window_20_r_p_slope": (w20_rp[-1]-w20_rp[0])/max(len(w20_rp)-1,1) if len(w20_rp)>=2 else 0,
                "window_20_imu_loss_median": np.nanmedian(w20_imu) if w20_imu else float("nan"),
                "window_20_imu_loss_slope": (w20_imu[-1]-w20_imu[0])/max(len(w20_imu)-1,1) if len(w20_imu)>=2 else 0,
                "cooldown_after_trigger": cd_after,
                "cooldown_reason": row.get("cooldown_reason",""),
                "run_ATE": ate if not np.isnan(ate) else "",
            })

        # ── D. Cooldown effect features ─────────────────────────
        cd_episodes = []
        cur_cd = []
        in_cd = False
        for row in valid:
            if "cooldown" in row.get("state_name",""):
                cur_cd.append(row); in_cd = True
            else:
                if in_cd and cur_cd:
                    cd_episodes.append(cur_cd); cur_cd = []
                in_cd = False
        if cur_cd: cd_episodes.append(cur_cd)

        for cd_idx, cd_ep in enumerate(cd_episodes):
            cd_reason = cd_ep[0].get("cooldown_reason","")
            cd_type = "fd" if "full_div" in cd_reason else ("rh" if "rot_harm" in cd_reason else "unknown")

            vc_raw_c = sum(1 for r in cd_ep if r.get("visual_collapse_raw","0")=="1")
            vc_trig_c = sum(1 for r in cd_ep if r.get("visual_collapse_triggered","0")=="1")
            fd_raw_c = sum(1 for r in cd_ep if r.get("full_divergence_raw","0")=="1")

            mode_after = ""
            next_start = -1
            next_fd = -1
            cd_end_idx = valid.index(cd_ep[-1])
            for r in valid[cd_end_idx+1:]:
                if mode_after == "": mode_after = r.get("state_name","")
                if "full_imu" in r.get("state_name","") and next_start < 0:
                    next_start = int(r.get("pair_id",-1))
                if r.get("full_divergence_triggered","0")=="1" and next_fd < 0:
                    next_fd = int(r.get("pair_id",-1))
                if mode_after and next_start >= 0 and next_fd >= 0: break

            cooldown_rows.append({
                "experiment_name": exp_name, "scene": scene, "trial": trial_id,
                "cooldown_id": cd_idx, "cooldown_reason": cd_reason,
                "start_pair": int(cd_ep[0]["pair_id"]), "end_pair": int(cd_ep[-1]["pair_id"]),
                "cooldown_len": len(cd_ep),
                "fd_or_rot_harm": cd_type,
                "vc_raw_during_cooldown_count": vc_raw_c,
                "vc_triggered_during_cooldown_count": vc_trig_c,
                "fd_raw_during_cooldown_count": fd_raw_c,
                "mode_after_cooldown": mode_after,
                "next_full_imu_start_pair": next_start,
                "next_fd_trigger_pair": next_fd,
                "run_ATE": ate if not np.isnan(ate) else "",
            })

        # ── E. Scene/trial outcomes ─────────────────────────────
        full_imu_f = sum(1 for r in valid if "full_imu" in r.get("state_name",""))
        pure_f = sum(1 for r in valid if "pure" in r.get("state_name",""))
        rot_f = sum(1 for r in valid if "rotation_only" in r.get("state_name",""))
        cd_f = sum(1 for r in valid if "cooldown" in r.get("state_name",""))
        fd_trig_c = sum(1 for r in valid if r.get("full_divergence_triggered","0")=="1")
        rh_trig_c = sum(1 for r in valid if r.get("rot_harm_triggered","0")=="1")
        vc_trig_c = sum(1 for r in valid if r.get("visual_collapse_triggered","0")=="1")
        sev_c = sum(1 for r in valid if r.get("severe_vc_triggered","0")=="1")
        mild_c = sum(1 for r in valid if r.get("mild_vc_triggered","0")=="1")
        fd_supp_c = sum(1 for r in valid if r.get("fd_check_suppressed_by_grace","0")=="1")

        ep_lens = []
        cur_ep = 0
        for r in valid:
            if "full_imu" in r.get("state_name",""): cur_ep+=1
            else:
                if cur_ep>0: ep_lens.append(cur_ep); cur_ep=0
        if cur_ep>0: ep_lens.append(cur_ep)

        tf = max(len(valid), 1)
        outcome_rows.append({
            "experiment_name": exp_name, "scene": scene, "trial": trial_id,
            "direct_ATE": ate if not np.isnan(ate) else "",
            "full_imu_frames": full_imu_f,
            "full_imu_ratio": f"{full_imu_f/tf*100:.1f}",
            "pure_frames": pure_f, "rotation_only_frames": rot_f,
            "cooldown_frames": cd_f,
            "fd_trigger_count": fd_trig_c,
            "rot_harm_trigger_count": rh_trig_c,
            "vc_trigger_count": vc_trig_c,
            "severe_vc_count": sev_c, "mild_vc_count": mild_c,
            "fd_grace_suppressed_count": fd_supp_c,
            "median_episode_len": np.median(ep_lens) if ep_lens else 0,
            "max_episode_len": max(ep_lens) if ep_lens else 0,
            "num_full_imu_episodes": len(ep_lens),
            "illegal_mode_count": sum(1 for r in valid if r.get("use_imu_translation","0")=="1"),
            "nan_count": sum(1 for r in valid for k in ["r_p_whitened_norm","imu_trans_loss"] if r.get(k,"")=="nan"),
            "pair_id_minus_one_count": sum(1 for r in rows_raw if int(r.get("pair_id",-1))<0),
        })

    # ── Write all CSVs ──────────────────────────────────────────
    def write_csv(name, rows, cols):
        p = OUTDIR / name
        with open(p, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader(); w.writerows(rows)
        print(f"  {name}: {len(rows)} rows")

    write_csv("pair_level_detector_features.csv", pair_rows,
              ["experiment_name","scene","trial","pair_id","mode",
               "num_visual_residuals","severe_vc_raw","mild_vc_raw","visual_collapse_triggered",
               "r_p_whitened_norm","imu_trans_loss","full_divergence_raw","full_divergence_triggered",
               "r_R_whitened_norm","median_flow_cov","rot_harm_triggered",
               "cooldown_active","cooldown_remaining","cooldown_reason",
               "fd_cooldown_config","rot_harm_cooldown_config",
               "fd_grace_enabled","fd_check_suppressed_by_grace","full_imu_episode_frame_idx",
               "est_delta_t_norm","run_ATE"])

    write_csv("full_imu_episode_features.csv", episode_rows,
              ["experiment_name","scene","trial","episode_id","start_pair","end_pair","episode_len",
               "fd_grace_enabled","fd_cooldown_config",
               "r_p_baseline_median_first_10","r_p_baseline_median_first_30",
               "imu_loss_baseline_median_first_10","imu_loss_baseline_median_first_30",
               "est_delta_t_baseline_median_first_10","est_delta_t_baseline_median_first_30",
               "r_p_end_median_last_10","imu_loss_end_median_last_10","est_delta_t_end_median_last_10",
               "fd_raw_count","fd_triggered_count","fd_trigger_pair_relative",
               "exited_by_fd","exited_by_rot_harm","exited_by_other",
               "next_cooldown_len","run_ATE"])

    write_csv("fd_trigger_window_features.csv", fd_trigger_rows,
              ["experiment_name","scene","trial","trigger_pair","mode_at_trigger","episode_id",
               "window_10_r_p_median","window_10_r_p_slope",
               "window_10_imu_loss_median","window_10_imu_loss_slope",
               "window_10_est_delta_t_median",
               "window_20_r_p_median","window_20_r_p_slope",
               "window_20_imu_loss_median","window_20_imu_loss_slope",
               "cooldown_after_trigger","cooldown_reason","run_ATE"])

    write_csv("cooldown_effect_features.csv", cooldown_rows,
              ["experiment_name","scene","trial","cooldown_id","cooldown_reason",
               "start_pair","end_pair","cooldown_len","fd_or_rot_harm",
               "vc_raw_during_cooldown_count","vc_triggered_during_cooldown_count",
               "fd_raw_during_cooldown_count","mode_after_cooldown",
               "next_full_imu_start_pair","next_fd_trigger_pair","run_ATE"])

    write_csv("scene_trial_outcomes.csv", outcome_rows,
              ["experiment_name","scene","trial","direct_ATE",
               "full_imu_frames","full_imu_ratio","pure_frames","rotation_only_frames","cooldown_frames",
               "fd_trigger_count","rot_harm_trigger_count","vc_trigger_count",
               "severe_vc_count","mild_vc_count","fd_grace_suppressed_count",
               "median_episode_len","max_episode_len","num_full_imu_episodes",
               "illegal_mode_count","nan_count","pair_id_minus_one_count"])

    # Report
    with open(OUTDIR/"fitting_dataset_build_report.md","w") as f:
        f.write("# V3b++ Offline Fitting Dataset — Build Report\n\n")
        f.write(f"**Runs collected**: {len(all_runs)}\n\n")
        f.write("## Output Tables\n\n")
        f.write(f"| Table | Rows | Description |\n")
        f.write(f"|---|---:|---|\n")
        f.write(f"| pair_level_detector_features.csv | {len(pair_rows)} | Per-pair detector signals |\n")
        f.write(f"| full_imu_episode_features.csv | {len(episode_rows)} | Per-episode statistics |\n")
        f.write(f"| fd_trigger_window_features.csv | {len(fd_trigger_rows)} | FD trigger context windows |\n")
        f.write(f"| cooldown_effect_features.csv | {len(cooldown_rows)} | Cooldown episode effects |\n")
        f.write(f"| scene_trial_outcomes.csv | {len(outcome_rows)} | Per-run summary |\n")
        f.write("\n## Usage Notes\n\n")
        f.write("- **Feature vs Label**: `run_ATE` is a run-level label, NOT an online signal\n")
        f.write("- **No model trained**: This dataset is for offline analysis only\n")
        f.write("- **No online integration**: These features are NOT fed into VisualHealthGateV3b\n")
        f.write("- **Development stress scenes** (moderate_turbidity, open_water_overcast, twilight_coast)\n")
        f.write("  are marked in the dataset; they have been used for diagnosis and should NOT\n")
        f.write("  be used as final validation\n")
        f.write("\n## Candidate Hypotheses (for future FD Controller v1)\n\n")
        f.write("1. **Episode-relative FD threshold**: r_p > max(10, k × episode_median_r_p)\n")
        f.write("2. **Evidence accumulation**: require FD evidence to accumulate across episodes\n")
        f.write("3. **Adaptive cooldown**: shorten cooldown when VC remains severe\n")
        f.write("4. **r_p slope detection**: FD should trigger when r_p is rising, not stable\n")
        f.write("\n*All hypotheses are UNVALIDATED. Do NOT implement without offline analysis.*\n")

    print(f"\nAll outputs → {OUTDIR}/")


if __name__ == "__main__":
    main()
