#!/usr/bin/env bash
set -euo pipefail

WORKDIR="/home/admin1/macvo-dev"
PYTHON="/home/admin1/miniconda3/envs/macvo/bin/python"
CACHE_DIR="${WORKDIR}/VisualCache/static63_unique_visual_20260713/clear_stop_turn_rectangle_truth_normal_noise"
RESULT_ROOT="${WORKDIR}/Results/rectangle_normal_noise_two_state_standard_full_20260715"

cd "${WORKDIR}"

# The four IMU variants share identical images. Reuse the existing rectangle
# visual sidecar and only rebuild it if the cache is incomplete.
"${PYTHON}" Scripts/build_relative_pose_factor_cache.py \
    --cache-dir "${CACHE_DIR}"

exec "${PYTHON}" Scripts/run_static63_cached_imu_fusion.py \
    --method two-state-fixed-lag \
    --result-root "${RESULT_ROOT}" \
    --variant-name vio_two_state_fixed_lag_standard_full \
    --scenes clear_stop_turn_rectangle_truth_normal_noise \
    --imu-vio-gravity-handling standard_local_frame_preintegration \
    --dashboard-port 8765 \
    --jobs 1 \
    --timeout 21600 \
    "$@"
