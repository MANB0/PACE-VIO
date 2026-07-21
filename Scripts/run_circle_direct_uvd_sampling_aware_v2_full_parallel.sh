#!/usr/bin/env bash
set -euo pipefail

WORKDIR="/home/admin1/macvo-dev"
PYTHON="/home/admin1/miniconda3/envs/macvo/bin/python"

cd "${WORKDIR}"
exec "${PYTHON}" Scripts/run_circle_direct_uvd_sampling_aware_v2_full_parallel.py "$@"
