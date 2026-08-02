#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:?Set DATA_ROOT to the directory containing the three datasets}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/Results/pace_vio_isam2_three_scenes}"
MODEL="${MODEL:-$ROOT/Model/MACVO_FrontendCov.pth}"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"

scenes=(
  clear_circle_truth_normal_noise
  clear_stop_turn_rectangle_truth_normal_noise
  clear_straight_truth_normal_noise
)

mkdir -p "$OUTPUT_ROOT/logs"
printf 'scene,status,start_utc,end_utc\n' > "$OUTPUT_ROOT/three_scene_status.csv"

for scene in "${scenes[@]}"; do
  start_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf '%s,running,%s,\n' "$scene" "$start_utc" >> "$OUTPUT_ROOT/three_scene_status.csv"
  echo "[$(date '+%F %T')] START $scene"
  "$PYTHON_BIN" "$ROOT/Scripts/run_pace_vio.py" \
    --dataset "$DATA_ROOT/$scene" \
    --output "$OUTPUT_ROOT" \
    --model "$MODEL" \
    --vio-backend isam2 \
    --static-init-mode fixed \
    --static-init-duration-s 3.0 \
    --no-live-display \
    --timeout 7200 \
    2>&1 | tee "$OUTPUT_ROOT/logs/$scene.log"
  end_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf '%s,complete,%s,%s\n' "$scene" "$start_utc" "$end_utc" >> "$OUTPUT_ROOT/three_scene_status.csv"
  echo "[$(date '+%F %T')] DONE  $scene"
done

echo "All three real-frontend iSAM2 scenes completed: $OUTPUT_ROOT"
