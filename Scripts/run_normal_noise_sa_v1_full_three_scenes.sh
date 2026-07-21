#!/usr/bin/env bash
set -euo pipefail

WORKDIR="/home/admin1/macvo-dev"
PYTHON="/home/admin1/miniconda3/envs/macvo/bin/python"
RESULT_ROOT="${WORKDIR}/Results/normal_noise_sa_v1_full_three_scenes_20260717"

cd "${WORKDIR}"

exec "${PYTHON}" Scripts/run_static63_cached_imu_fusion.py \
    --method two-state-fixed-lag \
    --result-root "${RESULT_ROOT}" \
    --variant-name vio_two_state_direct_uvd_sampling_aware_v1_full \
    --scenes \
        clear_circle_truth_normal_noise \
        clear_stop_turn_rectangle_truth_normal_noise \
        clear_straight_truth_normal_noise \
    --imu-vio-gravity-handling standard_local_frame_preintegration \
    --two-state-visual-factor-mode direct_uvd \
    --two-state-warm-start macvo_pose \
    --two-state-uvd-huber-delta 0.1 \
    --two-state-covariance-mode sampling_aware \
    --dashboard-port 8765 \
    --jobs 3 \
    --timeout 43200 \
    "$@"
