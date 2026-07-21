#!/usr/bin/env bash
set -euo pipefail

cd /home/admin1/macvo-dev
export PYTHONUNBUFFERED=1

exec /home/admin1/miniconda3/envs/macvo/bin/python \
  Scripts/run_rectangle_covariance_floor_ablation.py \
  --dashboard-port 8765 \
  "$@"
