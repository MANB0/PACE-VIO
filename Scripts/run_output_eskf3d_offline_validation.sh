#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/admin1/macvo-dev
PYTHON=/home/admin1/miniconda3/envs/macvo/bin/python
OUTPUT="$ROOT/Results/circle_output_eskf3d_ablation_20260718"

declare -A INPUTS=(
  [t_factor]="$ROOT/Results/circle_straight_normal_noise_two_state_standard_full_20260715/trial_1/vio_two_state_fixed_lag_standard_full/clear_circle_truth_normal_noise/poses.csv"
  [u1]="$ROOT/Results/circle_normal_noise_direct_uvd_u1_full_20260716/trial_1/vio_two_state_direct_uvd_u1_standard_full/clear_circle_truth_normal_noise/poses.csv"
  [sa_v1]="$ROOT/Results/normal_noise_sa_v1_full_three_scenes_20260717/trial_1/vio_two_state_direct_uvd_sampling_aware_v1_full/clear_circle_truth_normal_noise/poses.csv"
  [sa_v2]="$ROOT/Results/normal_noise_sa_v2_full_three_scenes_20260717/trial_1/vio_two_state_direct_uvd_sampling_aware_v2_full/clear_circle_truth_normal_noise/poses.csv"
)

cd "$ROOT"
for mode in no_gate gate gate_adaptive; do
  for method in t_factor u1 sa_v1 sa_v2; do
    "$PYTHON" Scripts/run_output_trajectory_eskf3d.py \
      --input "${INPUTS[$method]}" \
      --output "$OUTPUT/$mode/$method" \
      --mode "$mode"
  done
done

"$PYTHON" Scripts/plot_output_eskf3d_offline_ablation.py
"$PYTHON" -m pytest -q \
  Scripts/UnitTest/test_output_trajectory_eskf3d.py \
  Scripts/UnitTest/test_output_trajectory_ekf.py \
  | tee "$ROOT/analysis_circle_output_eskf3d_ablation_20260718/test_output.txt"
