#!/usr/bin/env python3
"""
V3b++ Design Plan — Data Extraction
Extract r_p, imu_trans_loss, est_delta_t_norm distributions across scenes.
"""

import csv, sys, numpy as np
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, '/home/admin1/macvo-dev')

WORKDIR = Path('/home/admin1/macvo-dev')
OUTDIR = WORKDIR / 'analysis_v3bpp_design_plan'

# ── Data sources ──────────────────────────────────────────────────
# Original 7 scenes: Rule B adaptive_decisions
OLD7_BASE = WORKDIR / 'Results/v3bplus_ruleB_7x3_20260521_235803'
OLD7_SCENES = ['turbid_harbor', 'clear_shallow', 'deep_dark', 'caustic_shallow',
               'dam_inspection', 'murky_coast', 'open_water']

# New 3 scenes: Rule B adaptive_decisions + fixed full_imu frame_pair_diagnostics
NEW3_DECISIONS = WORKDIR / 'Results/holdout_validation/ruleB/MACVO-HoloOcean-IMU@holoocean_imu'
NEW3_FIXED = WORKDIR / 'Results/holdout_validation/fixed_baseline/MACVO-HoloOcean-IMU@holoocean_imu'
NEW3_SCENES = ['moderate_turbidity', 'open_water_overcast', 'twilight_coast']

ALL_SCENES = OLD7_SCENES + NEW3_SCENES


def collect_ruleB_rp_imu_stats(scene: str, trial_dirs: list[Path]) -> list[dict]:
    """Collect r_p, imu_trans_loss from Rule B adaptive_decisions."""
    rows_all = []
    for d in trial_dirs:
        csv_path = d / 'adaptive_decisions.csv'
        if not csv_path.exists():
            continue
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for r in reader:
                pid = int(r.get('pair_id', -1))
                if pid <= 0:
                    continue
                state = r.get('state_name', '')
                mode_label = 'pure' if 'pure' in state else ('full_imu' if 'full_imu' in state else 'rotation_only')

                rp_str = r.get('r_p_whitened_norm', 'nan')
                imu_str = r.get('imu_trans_loss', 'nan')
                dt_str = r.get('d2_pre_rerun_est_delta_t_norm', 'nan')
                nvis_str = r.get('num_visual_residuals', 'nan')
                fc_str = r.get('median_flow_cov', 'nan')
                fd_trig = r.get('full_divergence_triggered', '0')

                try:
                    rp = float(rp_str) if rp_str not in ('nan', '') else np.nan
                except ValueError:
                    rp = np.nan
                try:
                    imu_l = float(imu_str) if imu_str not in ('nan', '') else np.nan
                except ValueError:
                    imu_l = np.nan
                try:
                    dt = float(dt_str) if dt_str not in ('nan', '') else np.nan
                except ValueError:
                    dt = np.nan

                rows_all.append({
                    'scene': scene,
                    'pair_id': pid,
                    'state': state,
                    'mode_label': mode_label,
                    'r_p': rp,
                    'imu_trans_loss': imu_l,
                    'est_delta_t_norm': dt,
                    'fd_triggered': fd_trig == '1',
                })
    return rows_all


def collect_fixed_full_imu_stats(scene: str, run_dirs: list[Path]) -> list[dict]:
    """Collect r_p, imu_trans_loss from fixed full_imu frame_pair_diagnostics."""
    rows_all = []
    for d in run_dirs:
        csv_path = d / 'frame_pair_diagnostics.csv'
        if not csv_path.exists():
            continue
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for r in reader:
                pid = int(r.get('pair_id', -1))
                if pid <= 0:
                    continue
                rp_str = r.get('r_p_whitened_norm', 'nan')
                imu_str = r.get('imu_trans_loss', 'nan')
                dt_str = r.get('est_delta_t_norm', 'nan')

                try:
                    rp = float(rp_str) if rp_str not in ('nan', '') else np.nan
                except ValueError:
                    rp = np.nan
                try:
                    imu_l = float(imu_str) if imu_str not in ('nan', '') else np.nan
                except ValueError:
                    imu_l = np.nan
                try:
                    dt = float(dt_str) if dt_str not in ('nan', '') else np.nan
                except ValueError:
                    dt = np.nan

                rows_all.append({
                    'scene': scene,
                    'pair_id': pid,
                    'r_p': rp,
                    'imu_trans_loss': imu_l,
                    'est_delta_t_norm': dt,
                })
    return rows_all


def percentile_or_nan(vals, p):
    clean = [v for v in vals if not np.isnan(v)]
    if not clean:
        return float('nan')
    return np.percentile(clean, p)


def main():
    # ═══════════════════════════════════════════════════════════════
    # Collect data from all scenes
    # ═══════════════════════════════════════════════════════════════

    # Old 7 scenes: Rule B
    old7_rb_data = {}
    for scene in OLD7_SCENES:
        trial_dirs = []
        for trial in ['trial_1', 'trial_2', 'trial_3']:
            d = OLD7_BASE / trial / scene
            if (d / 'adaptive_decisions.csv').exists():
                trial_dirs.append(d)
        old7_rb_data[scene] = collect_ruleB_rp_imu_stats(scene, trial_dirs)
        print(f"  Old7 {scene}: {len(old7_rb_data[scene])} Rule B frames")

    # New 3 scenes: Rule B
    new3_rb_data = {}
    for scene in NEW3_SCENES:
        trial_dirs = list(NEW3_DECISIONS.glob(f'*{scene}*'))
        new3_rb_data[scene] = collect_ruleB_rp_imu_stats(scene, trial_dirs)
        print(f"  New3 {scene}: {len(new3_rb_data[scene])} Rule B frames")

    # New 3 scenes: fixed full_imu
    new3_fixed_data = {}
    for scene in NEW3_SCENES:
        run_dirs = []
        for d in sorted(NEW3_FIXED.glob(f'*{scene}*')):
            # Check if it's full_imu
            import yaml
            sys.path.insert(0, str(WORKDIR))
            from Utility.Config import IncludeLoader
            config_path = d / 'config.yaml'
            with open(config_path) as f:
                cfg = yaml.load(f, IncludeLoader)
            args = cfg['Odometry']['args']
            if args.get('imu_rot_prior_enable') and args.get('imu_trans_prior_enable'):
                run_dirs.append(d)
        new3_fixed_data[scene] = collect_fixed_full_imu_stats(scene, run_dirs)
        print(f"  New3 fixed {scene}: {len(new3_fixed_data[scene])} fixed full_imu frames")

    # ═══════════════════════════════════════════════════════════════
    # CSV 1: fd_distribution_comparison.csv
    # ═══════════════════════════════════════════════════════════════
    with open(OUTDIR / 'fd_distribution_comparison.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['scene', 'source', 'mode', 'n_frames',
                    'median_r_p', 'p75_r_p', 'p90_r_p', 'p95_r_p',
                    'median_imu_trans_loss', 'p75_imu_trans_loss', 'p90_imu_trans_loss', 'p95_imu_trans_loss',
                    'median_est_delta_t_norm',
                    'ate_median', 'is_full_imu_beneficial', 'notes'])

        # Known ATE values (from previous analyses)
        # Old 7 scenes ATE (Rule B)
        old7_ate = {
            'turbid_harbor': {'rb': 0.87, 'ff': 7.28, 'beneficial': False},
            'clear_shallow': {'rb': 2.79, 'ff': 23.19, 'beneficial': False},
            'deep_dark': {'rb': 6.64, 'ff': 20.20, 'beneficial': False},
            'caustic_shallow': {'rb': 4.58, 'ff': 560.38, 'beneficial': False},
            'dam_inspection': {'rb': 26.81, 'ff': 34.86, 'beneficial': False},
            'murky_coast': {'rb': 19.25, 'ff': 8.90, 'beneficial': True},
            'open_water': {'rb': 225.41, 'ff': 113.46, 'beneficial': True},
        }
        new3_ate = {
            'moderate_turbidity': {'rb': 120.19, 'ff': 46.13, 'beneficial': True,
                                    'fixed_pure': 175.69, 'fixed_rot': 192.40, 'fixed_trans': 122.85},
            'open_water_overcast': {'rb': 19.08, 'ff': 29.08, 'beneficial': False},
            'twilight_coast': {'rb': 265.38, 'ff': 224.57, 'beneficial': False,
                                'fixed_rot': 237.77, 'fixed_trans': 167.91},
        }

        for scene in ALL_SCENES:
            # Rule B data
            if scene in old7_rb_data:
                rb_rows = old7_rb_data[scene]
            else:
                rb_rows = new3_rb_data.get(scene, [])

            # Rule B full_imu frames only
            rb_full = [r for r in rb_rows if r['mode_label'] == 'full_imu']
            rb_all = rb_rows

            rp_rb = [r['r_p'] for r in rb_full]
            imu_rb = [r['imu_trans_loss'] for r in rb_full]
            dt_rb = [r['est_delta_t_norm'] for r in rb_full]

            if rp_rb:
                # Rule B: full_imu source
                ate_info = old7_ate.get(scene, new3_ate.get(scene, {}))
                w.writerow([scene, 'RuleB_full_imu', 'full_imu (episodes)', len(rb_full),
                           f"{np.nanmedian(rp_rb):.3f}", f"{percentile_or_nan(rp_rb,75):.3f}",
                           f"{percentile_or_nan(rp_rb,90):.3f}", f"{percentile_or_nan(rp_rb,95):.3f}",
                           f"{np.nanmedian(imu_rb):.4f}", f"{percentile_or_nan(imu_rb,75):.4f}",
                           f"{percentile_or_nan(imu_rb,90):.4f}", f"{percentile_or_nan(imu_rb,95):.4f}",
                           f"{np.nanmedian(dt_rb):.4f}" if dt_rb else 'N/A',
                           f"{ate_info.get('rb', 'N/A')}",
                           str(ate_info.get('beneficial', 'N/A')),
                           ''])

            # Rule B all frames
            rp_all = [r['r_p'] for r in rb_all]
            imu_all = [r['imu_trans_loss'] for r in rb_all]
            if rp_all:
                w.writerow([scene, 'RuleB_all', 'all (rotation_only dominant)', len(rb_all),
                           f"{np.nanmedian(rp_all):.3f}", f"{percentile_or_nan(rp_all,75):.3f}",
                           f"{percentile_or_nan(rp_all,90):.3f}", f"{percentile_or_nan(rp_all,95):.3f}",
                           f"{np.nanmedian(imu_all):.4f}", f"{percentile_or_nan(imu_all,75):.4f}",
                           f"{percentile_or_nan(imu_all,90):.4f}", f"{percentile_or_nan(imu_all,95):.4f}",
                           'N/A', 'N/A', 'N/A', ''])

            # Fixed full_imu data (only available for new 3 scenes)
            if scene in new3_fixed_data:
                fix_rows = new3_fixed_data[scene]
                rp_fix = [r['r_p'] for r in fix_rows]
                imu_fix = [r['imu_trans_loss'] for r in fix_rows]
                dt_fix = [r['est_delta_t_norm'] for r in fix_rows]
                ate_info = new3_ate.get(scene, {})
                w.writerow([scene, 'fixed_full_imu', 'full_imu (whole trajectory)', len(fix_rows),
                           f"{np.nanmedian(rp_fix):.3f}", f"{percentile_or_nan(rp_fix,75):.3f}",
                           f"{percentile_or_nan(rp_fix,90):.3f}", f"{percentile_or_nan(rp_fix,95):.3f}",
                           f"{np.nanmedian(imu_fix):.4f}", f"{percentile_or_nan(imu_fix,75):.4f}",
                           f"{percentile_or_nan(imu_fix,90):.4f}", f"{percentile_or_nan(imu_fix,95):.4f}",
                           f"{np.nanmedian(dt_fix):.4f}" if dt_fix else 'N/A',
                           f"{ate_info.get('ff', 'N/A')}",
                           str(ate_info.get('beneficial', 'N/A')),
                           f"FD thresholds r_p>10, imu>0.02: above_threshold={np.nanmedian(rp_fix)>10 if rp_fix else 'N/A'}"])

    print("CSV 1/6: fd_distribution_comparison.csv ✓")

    # ═══════════════════════════════════════════════════════════════
    # CSV 2: cooldown_behavior_comparison.csv
    # ═══════════════════════════════════════════════════════════════
    with open(OUTDIR / 'cooldown_behavior_comparison.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['scene', 'trial', 'full_imu_episode_count', 'full_imu_episode_lengths',
                    'cooldown_total_frames', 'cooldown_entry_count', 'first_cooldown_pair',
                    'cooldown_entry_reasons', 'ate', 'notes'])

        for scene in ALL_SCENES:
            if scene in old7_rb_data:
                rb_rows = old7_rb_data[scene]
            else:
                rb_rows = new3_rb_data.get(scene, [])

            # Group by trial (use unique pair ranges)
            # For simplicity, use all frames
            # Count full_imu episodes (contiguous blocks)
            episodes = []
            cur = []
            for r in rb_rows:
                if r['mode_label'] == 'full_imu':
                    cur.append(r)
                else:
                    if cur:
                        episodes.append(cur)
                        cur = []
            if cur:
                episodes.append(cur)

            ep_lengths = [len(ep) for ep in episodes]
            cooldown_frames = sum(1 for r in rb_rows if 'cooldown' in r['state'])
            cooldown_entries = 0
            prev_state = ''
            cooldown_reasons = set()
            first_cd_pair = -1
            for r in rb_rows:
                if 'cooldown' in r['state'] and 'cooldown' not in prev_state:
                    cooldown_entries += 1
                    reason = r.get('cooldown_reason', 'unknown')
                    cooldown_reasons.add(reason)
                    if first_cd_pair < 0:
                        first_cd_pair = r['pair_id']
                prev_state = r['state']

            ate_info = old7_ate.get(scene, new3_ate.get(scene, {}))
            rb_ate = ate_info.get('rb', 'N/A')

            notes = ''
            if scene == 'moderate_turbidity':
                notes = '10 episodes, all killed by FD within 11-35 frames; cooldown dominates (990 frames)'
            elif scene == 'murky_coast':
                notes = 'full_imu from pair 1 (VC triggers immediately); sustained full_imu; no FD false positive'
            elif scene == 'twilight_coast':
                notes = 'VC rarely triggers; rot_harm cooldown(99) + FD cooldown(198); IMU rarely used'
            elif scene == 'open_water_overcast':
                notes = '0 episodes; gate never triggers; perfect pass'

            w.writerow([scene, 'all_trials', len(episodes),
                       ','.join(str(x) for x in ep_lengths[:20]),
                       cooldown_frames, cooldown_entries, first_cd_pair,
                       ','.join(sorted(cooldown_reasons)),
                       str(rb_ate), notes])

    print("CSV 2/6: cooldown_behavior_comparison.csv ✓")


if __name__ == '__main__':
    main()
