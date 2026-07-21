#!/usr/bin/env bash
set -euo pipefail

WORKDIR="/home/admin1/macvo-dev"
PYTHON="/home/admin1/miniconda3/envs/macvo/bin/python"

cd "${WORKDIR}"
exec "${PYTHON}" Scripts/run_normal_noise_u1_missing_full_scenes.py "$@"
