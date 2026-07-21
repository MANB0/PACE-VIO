#!/usr/bin/env bash
set -euo pipefail

WORKDIR=/home/admin1/macvo-dev
PYTHON=/home/admin1/miniconda3/envs/macvo/bin/python
CACHE_ROOT="$WORKDIR/VisualCache/static63_unique_visual_20260713"

cd "$WORKDIR"

"$PYTHON" Scripts/build_relative_pose_factor_cache.py \
  --cache-dir "$CACHE_ROOT/clear_circle_truth_normal_noise" \
  --cache-dir "$CACHE_ROOT/clear_stop_turn_rectangle_truth_normal_noise" \
  --cache-dir "$CACHE_ROOT/clear_straight_truth_normal_noise"

exec "$PYTHON" Scripts/run_static63_cached_imu_fusion.py \
  --method two-state-fixed-lag \
  --dashboard-port 8765 \
  --jobs 1 \
  "$@"
