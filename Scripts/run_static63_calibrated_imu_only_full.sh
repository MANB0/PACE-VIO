#!/usr/bin/env bash
set -euo pipefail

cd /home/admin1/macvo-dev
mkdir -p logs

PYTHONUNBUFFERED=1 /home/admin1/miniconda3/envs/macvo/bin/python \
  Scripts/run_static63_calibrated_imu_only.py \
  --force \
  2>&1 | tee logs/static63_calibrated_imu_only_four_configs_20260713.log
