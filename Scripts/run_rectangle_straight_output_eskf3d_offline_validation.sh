#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/admin1/macvo-dev
PYTHON=/home/admin1/miniconda3/envs/macvo/bin/python
EKF2D_OUTPUT="$ROOT/Results/rectangle_straight_output_ekf2d_ablation_20260718"
ESKF3D_OUTPUT="$ROOT/Results/rectangle_straight_output_eskf3d_ablation_20260718"

declare -A FRAME_COUNTS=(
  [rectangle]=1890
  [straight]=630
)

declare -A INPUTS=(
  [rectangle_t_factor]="$ROOT/Results/rectangle_normal_noise_two_state_standard_full_20260715/trial_1/vio_two_state_fixed_lag_standard_full/clear_stop_turn_rectangle_truth_normal_noise/poses.csv"
  [rectangle_u1]="$ROOT/Results/normal_noise_direct_uvd_u1_missing_full_scenes_20260717/trial_1/vio_two_state_direct_uvd_u1_standard_full/clear_stop_turn_rectangle_truth_normal_noise/poses.csv"
  [rectangle_sa_v1]="$ROOT/Results/normal_noise_sa_v1_full_three_scenes_20260717/trial_1/vio_two_state_direct_uvd_sampling_aware_v1_full/clear_stop_turn_rectangle_truth_normal_noise/poses.csv"
  [rectangle_sa_v2]="$ROOT/Results/normal_noise_sa_v2_full_three_scenes_20260717/trial_1/vio_two_state_direct_uvd_sampling_aware_v2_full/clear_stop_turn_rectangle_truth_normal_noise/poses.csv"
  [straight_t_factor]="$ROOT/Results/circle_straight_normal_noise_two_state_standard_full_20260715/trial_1/vio_two_state_fixed_lag_standard_full/clear_straight_truth_normal_noise/poses.csv"
  [straight_u1]="$ROOT/Results/normal_noise_direct_uvd_u1_missing_full_scenes_20260717/trial_1/vio_two_state_direct_uvd_u1_standard_full/clear_straight_truth_normal_noise/poses.csv"
  [straight_sa_v1]="$ROOT/Results/normal_noise_sa_v1_full_three_scenes_20260717/trial_1/vio_two_state_direct_uvd_sampling_aware_v1_full/clear_straight_truth_normal_noise/poses.csv"
  [straight_sa_v2]="$ROOT/Results/normal_noise_sa_v2_full_three_scenes_20260717/trial_1/vio_two_state_direct_uvd_sampling_aware_v2_full/clear_straight_truth_normal_noise/poses.csv"
)

cd "$ROOT"
for scene in rectangle straight; do
  seq_to="${FRAME_COUNTS[$scene]}"
  for method in t_factor u1 sa_v1 sa_v2; do
    input="${INPUTS[${scene}_${method}]}"
    "$PYTHON" Scripts/run_relative_pose_output_ekf.py \
      --input "$input" \
      --output "$EKF2D_OUTPUT/$scene/$method" \
      --seq-to "$seq_to" \
      --active-from 90 \
      --calibration-frames 300 \
      --measurement-std-scale 4.0 \
      --process-std-scale 0.25
    for mode in no_gate gate gate_adaptive; do
      "$PYTHON" Scripts/run_output_trajectory_eskf3d.py \
        --input "$input" \
        --output "$ESKF3D_OUTPUT/$scene/$mode/$method" \
        --seq-to "$seq_to" \
        --active-from 90 \
        --calibration-frames 300 \
        --measurement-std-scale 4.0 \
        --process-std-scale 0.25 \
        --mode "$mode"
    done
  done
done

"$PYTHON" Scripts/plot_rectangle_straight_output_eskf3d_offline_ablation.py
"$PYTHON" -m pytest -q \
  Scripts/UnitTest/test_plot_output_eskf3d_filters.py \
  Scripts/UnitTest/test_output_trajectory_eskf3d.py \
  Scripts/UnitTest/test_output_trajectory_ekf.py \
  | tee "$ROOT/analysis_rectangle_straight_output_eskf3d_ablation_20260718/test_output.txt"
