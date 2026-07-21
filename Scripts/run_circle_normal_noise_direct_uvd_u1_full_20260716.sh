#!/usr/bin/env bash
set -euo pipefail

WORKDIR="/home/admin1/macvo-dev"
PYTHON="/home/admin1/miniconda3/envs/macvo/bin/python"
RESULT_ROOT="${WORKDIR}/Results/circle_normal_noise_direct_uvd_u1_full_20260716"

cd "${WORKDIR}"

"${PYTHON}" Scripts/run_static63_cached_imu_fusion.py \
    --method two-state-fixed-lag \
    --result-root "${RESULT_ROOT}" \
    --variant-name vio_two_state_direct_uvd_u1_standard_full \
    --scenes clear_circle_truth_normal_noise \
    --imu-vio-gravity-handling standard_local_frame_preintegration \
    --two-state-visual-factor-mode direct_uvd \
    --two-state-warm-start macvo_pose \
    --two-state-uvd-huber-delta 0.1 \
    --dashboard-port 8765 \
    --jobs 1 \
    --timeout 21600 \
    "$@"

if [[ " ${*} " != *" --dry-run "* ]]; then
    "${PYTHON}" Scripts/plot_circle_direct_uvd_u1_vs_pose_factor.py \
        --u1-result-root "${RESULT_ROOT}"
fi
