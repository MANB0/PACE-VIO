#!/usr/bin/env bash
set -euo pipefail

cd /home/admin1/macvo-dev

export PYTHONUNBUFFERED=1
exec /home/admin1/miniconda3/envs/macvo/bin/python \
  Scripts/run_static63_cached_imu_fusion.py \
  --method calibrated-staticinit \
  --jobs 1 \
  --dashboard-port 8765 \
  "$@"
