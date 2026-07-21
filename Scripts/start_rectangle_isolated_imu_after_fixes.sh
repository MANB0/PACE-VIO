#!/usr/bin/env bash
set -euo pipefail

cd /home/admin1/macvo-dev
export PYTHONUNBUFFERED=1

exec /home/admin1/miniconda3/envs/macvo/bin/python \
  Scripts/run_static63_cached_imu_fusion.py \
  --method calibrated-staticinit \
  --variant-name vio_preintegrated_full_imuatt_staticinit_calibrated_biasfix_floor_1e-8 \
  --imu-vio-cov-diagonal-floor 1e-8 \
  --result-root /home/admin1/macvo-dev/Results/rectangle_isolated_imu_after_fixes_20260713 \
  --scenes \
    clear_stop_turn_rectangle_truth_bias_no_noise \
    clear_stop_turn_rectangle_truth_noise_no_bias \
  --jobs 1 \
  --dashboard-port 8765 \
  "$@"
