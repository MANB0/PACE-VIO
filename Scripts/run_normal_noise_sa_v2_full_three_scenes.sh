#!/usr/bin/env bash
set -euo pipefail

cd /home/admin1/macvo-dev
exec /home/admin1/miniconda3/envs/macvo/bin/python \
  Scripts/run_normal_noise_sa_v2_full_three_scenes.py "$@"
