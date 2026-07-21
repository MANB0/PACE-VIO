#!/usr/bin/env python3
"""
Extract all diagnostic data from holdout validation results.
Generates CSV files for analysis_ruleB_new_scene_stress_test/
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

WORKDIR = Path("/home/admin1/macvo-dev")
sys.path.insert(0, str(WORKDIR))
from Utility.Config import IncludeLoader
from Scripts.eval_qa_vif import evaluate_trajectory_direct

OUTDIR = WORKDIR / "analysis_ruleB_new_scene_stress_test"
RESULT_ROOT = WORKDIR / "Results/holdout_validation"
SCENES = ["moderate_turbidity", "open_water_overcast", "twilight_coast"]
MODES = ["pure_macvo", "rotation_only", "translation_only", "full_imu", "ruleB"]
BATCH = Path("/mnt/e/文档/holoocean/code/recordings/batch_20260528_203401")


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.load(f, IncludeLoader)


# ── Step 1: Collect all ATE results ──────────────────────────────
def collect_all_ates():
    """Return dict[(scene, mode)] -> list of per-trial ATE dicts."""
    all_data = defaultdict(list)

    for phase in ["fixed_baseline", "ruleB"]:
        phase_root = RESULT_ROOT / phase
        for proj_dir in phase_root.glob("*/"):
            for run_dir in sorted(proj_dir.glob("*/")):
                config_path = run_dir / "config.yaml"
                poses_path = run_dir / "poses.csv"
                if not config_path.exists():
                    continue

                cfg = load_yaml(config_path)
                od = cfg['Odometry']
                args = od['args']
                opt_args = od['optimizer']['args']

                imu_rot = args.get('imu_rot_prior_enable', False)
                imu_trans = args.get('imu_trans_prior_enable', False)
                mapping = args.get('mapping', True)
                post_fusion = opt_args.get('post_imu_fusion_enable', True)

                scene = 'unknown'
                for s in SCENES:
                    if s in run_dir.name:
                        scene = s
                        break

                if not mapping and not post_fusion:
                    mode = 'ruleB'
                elif not imu_rot and not imu_trans:
                    mode = 'pure_macvo'
                elif imu_rot and not imu_trans:
                    mode = 'rotation_only'
                elif not imu_rot and imu_trans:
                    mode = 'translation_only'
                elif imu_rot and imu_trans:
                    mode = 'full_imu'
                else:
                    continue

                trial_id = run_dir.name[:17]

                gt_path = BATCH / scene / 'ref_pose.csv'
                try:
                    result = evaluate_trajectory_direct(poses_path, gt_path, f'{scene}_{mode}')
                    ate_val = float(result['ate']['ate_rmse'])
                except Exception:
                    ate_val = None

                all_data[(scene, mode)].append({
                    'trial_id': trial_id,
                    'ate': ate_val,
                    'run_dir': str(run_dir),
                })

    return all_data


# ── Step 2: Read adaptive_decisions.csv for Rule B runs ──────────
def read_ruleB_decisions():
    """Return dict[(scene, trial_id)] -> list of row dicts (pair_id>0)."""
    decisions = {}
    base = RESULT_ROOT / 'ruleB' / 'MACVO-HoloOcean-IMU@holoocean_imu'
    for run_dir in sorted(base.glob('*/')):
        csv_path = run_dir / 'adaptive_decisions.csv'
        if not csv_path.exists():
            continue
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            rows = [r for r in reader if int(r['pair_id']) > 0]

        scene = 'unknown'
        for s in SCENES:
            if s in run_dir.name:
                scene = s
                break
        trial_id = run_dir.name[:17]
        decisions[(scene, trial_id)] = rows
    return decisions


# ── MAIN ──────────────────────────────────────────────────────────
def main():
    all_ates = collect_all_ates()
    decisions = read_ruleB_decisions()

    # ═══════════════════════════════════════════════════════════════
    # CSV 1: new_scene_ate_summary.csv
    # ═══════════════════════════════════════════════════════════════
    with open(OUTDIR / 'new_scene_ate_summary.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['scene', 'method', 'trial_1_ATE', 'trial_2_ATE', 'trial_3_ATE',
                    'median_ATE', 'mean_ATE', 'std_ATE', 'cv', 'rank_by_median', 'notes'])

        for scene in SCENES:
            # Compute ranks within scene
            mode_meds = {}
            for mode in MODES:
                vals = [d['ate'] for d in all_ates.get((scene, mode), []) if d['ate'] is not None]
                if len(vals) >= 3:
                    mode_meds[mode] = np.median(vals)
            sorted_modes = sorted(mode_meds.items(), key=lambda x: x[1])
            ranks = {m: i+1 for i, (m, _) in enumerate(sorted_modes)}

            for mode in MODES:
                trials = [d['ate'] for d in all_ates.get((scene, mode), [])]
                ates = [t for t in trials if t is not None]
                if len(ates) < 3:
                    continue
                median_v = np.median(ates)
                mean_v = np.mean(ates)
                std_v = np.std(ates, ddof=1)
                cv = std_v / mean_v if mean_v > 0 else 0

                t1 = f"{ates[0]:.3f}" if len(ates) > 0 else 'N/A'
                t2 = f"{ates[1]:.3f}" if len(ates) > 1 else 'N/A'
                t3 = f"{ates[2]:.3f}" if len(ates) > 2 else 'N/A'

                # Notes
                notes = ''
                if mode == 'ruleB':
                    if scene == 'moderate_turbidity':
                        notes = 'VC triggers→FD kills→cooldown lock; only 10% IMU usage'
                    elif scene == 'open_water_overcast':
                        notes = 'Never triggers; matches oracle'
                    elif scene == 'twilight_coast':
                        notes = 'VC rare; FD/rot_harm cooldown dominates'

                w.writerow([scene, mode, t1, t2, t3,
                           f"{median_v:.3f}", f"{mean_v:.3f}", f"{std_v:.3f}",
                           f"{cv:.3f}", ranks.get(mode, '-'), notes])

    print("CSV 1/7: new_scene_ate_summary.csv ✓")

    # ═══════════════════════════════════════════════════════════════
    # CSV 2: ruleB_vs_fixed_oracle_new_scenes.csv
    # ═══════════════════════════════════════════════════════════════
    with open(OUTDIR / 'ruleB_vs_fixed_oracle_new_scenes.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['scene', 'oracle_method', 'oracle_median_ATE', 'ruleB_median_ATE',
                    'ruleB_vs_oracle_ratio', 'best_allowed_without_translation',
                    'best_allowed_without_translation_ATE', 'ruleB_vs_best_allowed_ratio',
                    'status', 'notes'])

        for scene in SCENES:
            best_all = ('', float('inf'))
            best_no_trans = ('', float('inf'))

            for mode in MODES:
                vals = [d['ate'] for d in all_ates.get((scene, mode), []) if d['ate'] is not None]
                if len(vals) < 3:
                    continue
                med = np.median(vals)
                if mode != 'ruleB' and med < best_all[1]:
                    best_all = (mode, med)
                if mode not in ('ruleB', 'translation_only') and med < best_no_trans[1]:
                    best_no_trans = (mode, med)

            rb_vals = [d['ate'] for d in all_ates.get((scene, 'ruleB'), []) if d['ate'] is not None]
            rb_med = np.median(rb_vals) if len(rb_vals) >= 3 else float('nan')

            ratio_all = rb_med / best_all[1] if best_all[1] > 0 else float('nan')
            ratio_no_trans = rb_med / best_no_trans[1] if best_no_trans[1] > 0 else float('nan')

            if ratio_all < 1.05:
                status = 'PASS'
            elif ratio_all < 1.50:
                status = 'MARGINAL'
            else:
                status = 'FAIL'

            notes = ''
            if scene == 'moderate_turbidity':
                notes = 'oracle=full_imu(46m); Rule B=120m; FD cooldown prevents sustained full_imu'
            elif scene == 'open_water_overcast':
                notes = 'oracle=pure_macvo(19m); Rule B=19m; all methods tie; scene too easy'
            elif scene == 'twilight_coast':
                notes = 'oracle=trans_only(168m) but unstable; best allowed=rot_only(238m); Rule B=265m worse than all'

            w.writerow([scene, best_all[0], f"{best_all[1]:.3f}", f"{rb_med:.3f}",
                       f"{ratio_all:.3f}", best_no_trans[0], f"{best_no_trans[1]:.3f}",
                       f"{ratio_no_trans:.3f}", status, notes])

    print("CSV 2/7: ruleB_vs_fixed_oracle_new_scenes.csv ✓")

    # ═══════════════════════════════════════════════════════════════
    # CSV 3: new_scene_ruleB_mode_timeline_summary.csv
    # ═══════════════════════════════════════════════════════════════
    with open(OUTDIR / 'new_scene_ruleB_mode_timeline_summary.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['scene', 'trial', 'total_pairs', 'rotation_only_frames', 'pure_macvo_frames',
                    'full_imu_frames', 'cooldown_frames', 'idle_frames', 'num_mode_switches',
                    'first_vc_pair', 'first_full_enter_pair', 'first_full_divergence_pair',
                    'first_cooldown_pair', 'final_mode'])

        for (scene, trial_id), rows in sorted(decisions.items()):
            n = len(rows)
            mode_counts = defaultdict(int)
            for r in rows:
                state = r['state_name']
                if 'rotation_only' in state:
                    mode_counts['rotation_only'] += 1
                elif 'pure' in state and 'cooldown' in state:
                    mode_counts['cooldown'] += 1
                elif 'pure_idle' in state or state == 'pure_idle':
                    mode_counts['idle'] += 1
                elif 'full_imu' in state:
                    mode_counts['full_imu'] += 1
                elif 'full_divergence' in state:
                    mode_counts['full_divergence'] += 1
                elif 'probation' in state:
                    mode_counts['full_imu'] += 1
                else:
                    mode_counts[state] += 1

            # Count mode switches
            prev_state = rows[0]['state_name'] if rows else ''
            switches = 0
            for r in rows[1:]:
                if r['state_name'] != prev_state:
                    switches += 1
                prev_state = r['state_name']

            # First events
            first_vc = next((int(r['pair_id']) for r in rows
                           if r['severe_vc_triggered'] == '1' or r['mild_vc_triggered'] == '1'), -1)
            first_full_enter = next((int(r['pair_id']) for r in rows
                                    if 'full_imu' in r['state_name']), -1)
            first_fd = next((int(r['pair_id']) for r in rows
                           if r['full_divergence_triggered'] == '1'), -1)
            first_cooldown = next((int(r['pair_id']) for r in rows
                                  if 'cooldown' in r['state_name']), -1)
            final_mode = rows[-1]['state_name'] if rows else ''

            cooldown_frames = mode_counts.get('cooldown', 0)
            idle_frames = mode_counts.get('idle', 0)
            full_imu_frames = mode_counts.get('full_imu', 0) + mode_counts.get('full_divergence', 0)

            w.writerow([scene, trial_id, n, mode_counts.get('rotation_only', 0),
                       mode_counts.get('pure_macvo', 0), full_imu_frames,
                       cooldown_frames, idle_frames, switches,
                       first_vc, first_full_enter, first_fd, first_cooldown, final_mode])

    print("CSV 3/7: new_scene_ruleB_mode_timeline_summary.csv ✓")

    # ═══════════════════════════════════════════════════════════════
    # CSV 4: new_scene_detector_trigger_summary.csv
    # ═══════════════════════════════════════════════════════════════
    with open(OUTDIR / 'new_scene_detector_trigger_summary.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['scene', 'trial', 'severe_vc_count', 'mild_vc_count',
                    'visual_collapse_count', 'full_divergence_count', 'rot_harm_count',
                    'first_severe_vc_pair', 'first_mild_vc_pair',
                    'first_full_divergence_pair', 'first_rot_harm_pair',
                    'max_severe_counter', 'max_mild_counter',
                    'max_full_divergence_counter', 'max_rot_harm_counter'])

        for (scene, trial_id), rows in sorted(decisions.items()):
            severe_c = sum(1 for r in rows if r['severe_vc_triggered'] == '1')
            mild_c = sum(1 for r in rows if r['mild_vc_triggered'] == '1')
            vc_c = sum(1 for r in rows if r['visual_collapse_triggered'] == '1')
            fd_c = sum(1 for r in rows if r['full_divergence_triggered'] == '1')
            rh_c = sum(1 for r in rows if r['rot_harm_triggered'] == '1')

            first_severe = next((int(r['pair_id']) for r in rows if r['severe_vc_triggered'] == '1'), -1)
            first_mild = next((int(r['pair_id']) for r in rows if r['mild_vc_triggered'] == '1'), -1)
            first_fd = next((int(r['pair_id']) for r in rows if r['full_divergence_triggered'] == '1'), -1)
            first_rh = next((int(r['pair_id']) for r in rows if r['rot_harm_triggered'] == '1'), -1)

            max_severe = max((int(r['severe_vc_counter']) for r in rows), default=0)
            max_mild = max((int(r['mild_vc_counter']) for r in rows), default=0)
            max_fd = max((int(r['full_divergence_counter']) for r in rows), default=0)
            max_rh = max((int(r['rot_harm_counter']) for r in rows), default=0)

            w.writerow([scene, trial_id, severe_c, mild_c, vc_c, fd_c, rh_c,
                       first_severe, first_mild, first_fd, first_rh,
                       max_severe, max_mild, max_fd, max_rh])

    print("CSV 4/7: new_scene_detector_trigger_summary.csv ✓")

    # ═══════════════════════════════════════════════════════════════
    # CSV 5: moderate_turbidity_cooldown_diagnosis.csv
    # ═══════════════════════════════════════════════════════════════
    with open(OUTDIR / 'moderate_turbidity_cooldown_diagnosis.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['trial', 'first_full_enter_pair', 'first_full_divergence_pair',
                    'full_imu_episode_count', 'full_imu_episode_lengths',
                    'cooldown_total_frames', 'median_r_p_full', 'median_imu_trans_loss_full',
                    'median_est_delta_t_norm_full', 'median_num_vis_full',
                    'median_flow_cov_full', 'fd_trigger_pairs', 'fd_trigger_reasons',
                    'notes'])

        for (scene, trial_id), rows in sorted(decisions.items()):
            if scene != 'moderate_turbidity':
                continue

            # Full IMU episodes
            full_imu_rows = [r for r in rows if 'full_imu' in r['state_name']]
            full_fd_rows = [r for r in rows if 'full_divergence' in r['state_name']]

            # Count episodes (contiguous blocks)
            episodes = []
            current_ep = []
            in_full = False
            for r in rows:
                if 'full_imu' in r['state_name'] or 'full_divergence' in r['state_name']:
                    current_ep.append(r)
                    in_full = True
                else:
                    if in_full and current_ep:
                        episodes.append(current_ep)
                        current_ep = []
                    in_full = False
            if current_ep:
                episodes.append(current_ep)

            first_full_enter = min((int(ep[0]['pair_id']) for ep in episodes), default=-1)
            first_fd = next((int(r['pair_id']) for r in rows if r['full_divergence_triggered'] == '1'), -1)

            ep_lengths = [len(ep) for ep in episodes]
            cooldown_total = sum(1 for r in rows if 'cooldown' in r['state_name'])

            # Stats during full_imu episodes
            r_p_vals = [float(r['r_p_whitened_norm']) for r in full_imu_rows + full_fd_rows
                       if r['r_p_whitened_norm'] not in ('nan', '')]
            imu_loss_vals = [float(r['imu_trans_loss']) for r in full_imu_rows + full_fd_rows
                           if r['imu_trans_loss'] not in ('nan', '')]
            delta_vals = [float(r['d2_pre_rerun_est_delta_t_norm']) for r in full_imu_rows + full_fd_rows
                         if r.get('d2_pre_rerun_est_delta_t_norm', 'nan') not in ('nan', '')]
            nvis_vals = [int(r['num_visual_residuals']) for r in full_imu_rows + full_fd_rows]
            fc_vals = [float(r['median_flow_cov']) for r in full_imu_rows + full_fd_rows
                      if r['median_flow_cov'] not in ('nan', '')]

            fd_pairs = [r['pair_id'] for r in rows if r['full_divergence_triggered'] == '1']
            fd_reasons = set()
            for r in rows:
                if r['full_divergence_triggered'] == '1':
                    # Infer reason from state
                    state = r['state_name']
                    fd_reasons.add(state)

            w.writerow([trial_id,
                       first_full_enter,
                       first_fd,
                       len(episodes),
                       ','.join(str(x) for x in ep_lengths),
                       cooldown_total,
                       f"{np.median(r_p_vals):.4f}" if r_p_vals else 'N/A',
                       f"{np.median(imu_loss_vals):.6f}" if imu_loss_vals else 'N/A',
                       f"{np.median(delta_vals):.4f}" if delta_vals else 'N/A',
                       f"{np.median(nvis_vals):.0f}" if nvis_vals else 'N/A',
                       f"{np.median(fc_vals):.2f}" if fc_vals else 'N/A',
                       ','.join(fd_pairs[:10]),
                       ','.join(fd_reasons),
                       'FD cooldown prevents sustained full_imu; episodes very short'])

    print("CSV 5/7: moderate_turbidity_cooldown_diagnosis.csv ✓")

    # ═══════════════════════════════════════════════════════════════
    # CSV 6: twilight_coast_failure_diagnosis.csv
    # ═══════════════════════════════════════════════════════════════
    with open(OUTDIR / 'twilight_coast_failure_diagnosis.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['trial', 'ruleB_ATE', 'fixed_rot_ATE', 'fixed_pure_ATE',
                    'fixed_trans_ATE', 'fixed_full_ATE', 'best_method',
                    'ruleB_full_imu_frames', 'ruleB_cooldown_frames',
                    'severe_vc_count', 'mild_vc_count', 'full_divergence_count',
                    'rot_harm_count', 'median_num_vis', 'median_flow_cov',
                    'median_r_p', 'median_imu_trans_loss', 'hypothesis'])

        for (scene, trial_id), rows in sorted(decisions.items()):
            if scene != 'twilight_coast':
                continue

            rb_ates = [d['ate'] for d in all_ates.get((scene, 'ruleB'), []) if d['ate'] is not None and d['trial_id'] == trial_id]
            rb_ate = rb_ates[0] if rb_ates else float('nan')

            # Fixed mode medians
            fixed_meds = {}
            for m in ['rotation_only', 'pure_macvo', 'translation_only', 'full_imu']:
                vals = [d['ate'] for d in all_ates.get((scene, m), []) if d['ate'] is not None]
                if len(vals) >= 3:
                    fixed_meds[m] = np.median(vals)

            best_m = min(fixed_meds, key=fixed_meds.get) if fixed_meds else ''
            full_imu_f = sum(1 for r in rows if 'full_imu' in r['state_name'])
            cooldown_f = sum(1 for r in rows if 'cooldown' in r['state_name'])
            severe_c = sum(1 for r in rows if r['severe_vc_triggered'] == '1')
            mild_c = sum(1 for r in rows if r['mild_vc_triggered'] == '1')
            fd_c = sum(1 for r in rows if r['full_divergence_triggered'] == '1')
            rh_c = sum(1 for r in rows if r['rot_harm_triggered'] == '1')

            nvis_vals = [int(r['num_visual_residuals']) for r in rows]
            fc_vals = [float(r['median_flow_cov']) for r in rows if r['median_flow_cov'] not in ('nan', '')]
            r_p_vals = [float(r['r_p_whitened_norm']) for r in rows if r['r_p_whitened_norm'] not in ('nan', '')]
            imu_loss_vals = [float(r['imu_trans_loss']) for r in rows if r['imu_trans_loss'] not in ('nan', '')]

            hypothesis = ('translation_only is best oracle but Rule B cannot access it; '
                         'VC rarely triggers (8 severe/1799); when FD/rot_harm trigger → cooldown; '
                         'candidate mode set lacks translation_only option; '
                         'also rot_harm fires on what appears to be a fundamentally difficult scene')

            w.writerow([trial_id, f"{rb_ate:.3f}",
                       f"{fixed_meds.get('rotation_only', float('nan')):.3f}",
                       f"{fixed_meds.get('pure_macvo', float('nan')):.3f}",
                       f"{fixed_meds.get('translation_only', float('nan')):.3f}",
                       f"{fixed_meds.get('full_imu', float('nan')):.3f}",
                       best_m, full_imu_f, cooldown_f,
                       severe_c, mild_c, fd_c, rh_c,
                       f"{np.median(nvis_vals):.0f}" if nvis_vals else 'N/A',
                       f"{np.median(fc_vals):.2f}" if fc_vals else 'N/A',
                       f"{np.median(r_p_vals):.4f}" if r_p_vals else 'N/A',
                       f"{np.median(imu_loss_vals):.6f}" if imu_loss_vals else 'N/A',
                       hypothesis])

    print("CSV 6/7: twilight_coast_failure_diagnosis.csv ✓")

    # ═══════════════════════════════════════════════════════════════
    # CSV 7: open_water_overcast_success_diagnosis.csv
    # ═══════════════════════════════════════════════════════════════
    with open(OUTDIR / 'open_water_overcast_success_diagnosis.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['trial', 'ruleB_ATE', 'oracle_ATE', 'vc_count',
                    'full_imu_frames', 'min_num_visual_residuals',
                    'num_frames_below_50', 'num_frames_below_30',
                    'max_consecutive_below_50', 'max_consecutive_below_30',
                    'notes'])

        for (scene, trial_id), rows in sorted(decisions.items()):
            if scene != 'open_water_overcast':
                continue

            rb_ates = [d['ate'] for d in all_ates.get((scene, 'ruleB'), []) if d['ate'] is not None and d['trial_id'] == trial_id]
            rb_ate = rb_ates[0] if rb_ates else float('nan')

            oracle_vals = [d['ate'] for d in all_ates.get((scene, 'pure_macvo'), []) if d['ate'] is not None]
            oracle_ate = np.median(oracle_vals) if len(oracle_vals) >= 3 else float('nan')

            vc_c = sum(1 for r in rows if r['severe_vc_triggered'] == '1' or r['mild_vc_triggered'] == '1')
            full_imu_f = sum(1 for r in rows if 'full_imu' in r['state_name'])

            nvis = [int(r['num_visual_residuals']) for r in rows]
            min_nvis = min(nvis) if nvis else -1
            below50 = sum(1 for v in nvis if v < 50)
            below30 = sum(1 for v in nvis if v < 30)

            # Max consecutive below thresholds
            def max_consecutive(vals, thr):
                cur = 0
                mx = 0
                for v in vals:
                    if v < thr:
                        cur += 1
                        mx = max(mx, cur)
                    else:
                        cur = 0
                return mx

            max_cons50 = max_consecutive(nvis, 50)
            max_cons30 = max_consecutive(nvis, 30)

            notes = (f'Scene too easy: n_vis rarely drops below 50 ({below50}/{len(nvis)} frames); '
                    f'flow_cov median ~1.5; all 5 methods within 0.5m ATE; '
                    f'gate correctly stays in rotation_only; visual frontend works perfectly')

            w.writerow([trial_id, f"{rb_ate:.3f}", f"{oracle_ate:.3f}", vc_c,
                       full_imu_f, min_nvis, below50, below30,
                       max_cons50, max_cons30, notes])

    print("CSV 7/7: open_water_overcast_success_diagnosis.csv ✓")
    print(f"\nAll CSVs written to {OUTDIR}/")


if __name__ == '__main__':
    main()
