#!/usr/bin/env bash
set -euo pipefail

cd /home/admin1/macvo-dev
PYTHON=/home/admin1/miniconda3/envs/macvo/bin/python
RUNNER=Scripts/run_circle_translation_oracles.py
OUTPUT=/home/admin1/macvo-dev/analysis_circle_translation_oracle_20260716

echo "Preparing full-circle oracle caches and locked configs..."
"${PYTHON}" "${RUNNER}" --prepare --scope full

for mode in V0 V1 V2 V3 O3 O4; do
  echo "Running full-circle ${mode}..."
  "${PYTHON}" "${RUNNER}" --run-one --scope full --mode "${mode}"
done

"${PYTHON}" "${RUNNER}" --summarize --scope full
echo "Full-circle oracle results: ${OUTPUT}"
