#!/usr/bin/env bash
set -euo pipefail

WORKDIR="/home/admin1/macvo-dev"
PYTHON="/home/admin1/miniconda3/envs/macvo/bin/python"
CACHE_ROOT="${WORKDIR}/VisualCache/static63_unique_visual_20260713"
RESULT_ROOT="${WORKDIR}/Results/circle_straight_normal_noise_two_state_standard_full_20260715"

CIRCLE_SCENE="clear_circle_truth_normal_noise"
STRAIGHT_SCENE="clear_straight_truth_normal_noise"

cd "${WORKDIR}"

# Validate or refresh the pure-visual relative-pose sidecars before replay.
"${PYTHON}" Scripts/build_relative_pose_factor_cache.py \
    --cache-dir "${CACHE_ROOT}/${CIRCLE_SCENE}" \
    --cache-dir "${CACHE_ROOT}/${STRAIGHT_SCENE}"

"${PYTHON}" Scripts/run_static63_cached_imu_fusion.py \
    --method two-state-fixed-lag \
    --result-root "${RESULT_ROOT}" \
    --variant-name vio_two_state_fixed_lag_standard_full \
    --scenes "${CIRCLE_SCENE}" "${STRAIGHT_SCENE}" \
    --imu-vio-gravity-handling standard_local_frame_preintegration \
    --dashboard-port 8765 \
    --jobs 1 \
    --timeout 21600 \
    "$@"

dry_run=false
for argument in "$@"; do
    if [[ "${argument}" == "--dry-run" ]]; then
        dry_run=true
        break
    fi
done

if [[ "${dry_run}" == "false" ]]; then
    "${PYTHON}" Scripts/plot_circle_straight_normal_noise_two_state_standard_full.py \
        --result-root "${RESULT_ROOT}"
fi

