#!/bin/bash
# =============================================================================
# V3b Sanity Check — 7-scene sequential run
# =============================================================================
# Runs all 7 HoloOcean scenes with the V3b adaptive gate, one by one.
# tqdm progress bars are visible because we use direct conda activate.
#
# Usage:
#   bash Scripts/run_v3b_sanity.sh
#
# Output: Results/v3b_sanity_YYYYMMDD_HHMMSS/
# =============================================================================

set -e

# ── Paths ────────────────────────────────────────────────────────────
BATCH_ROOT="/mnt/e/文档/holoocean/code/recordings/batch_20260515_155653"
WORKDIR="/home/admin1/macvo-dev"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULT_DIR="${WORKDIR}/Results/v3b_sanity_${TIMESTAMP}"

SCENES=(
    "turbid_harbor"
    "clear_shallow"
    "deep_dark"
    "caustic_shallow"
    "dam_inspection"
    "murky_coast"
    "open_water"
)

# ── Activate conda ───────────────────────────────────────────────────
echo "Activating conda environment: macvo"
eval "$(conda shell.bash hook)"
conda activate macvo
echo "Python: $(which python)"
echo ""

mkdir -p "${RESULT_DIR}"
echo "scene,direct_ATE" > "${RESULT_DIR}/ate_summary.csv"

echo "============================================================================"
echo "V3b Sanity Check — $(date)"
echo "============================================================================"
echo "Batch root: ${BATCH_ROOT}"
echo "Results:    ${RESULT_DIR}"
echo "Scenes:     ${SCENES[*]}"
echo "============================================================================"

# ── Run all scenes sequentially ──────────────────────────────────────
total=${#SCENES[@]}
for i in "${!SCENES[@]}"; do
    scene="${SCENES[$i]}"
    idx=$((i + 1))

    echo ""
    echo "================================================================"
    echo "[${idx}/${total}] ${scene}"
    echo "================================================================"

    scene_result="${RESULT_DIR}/${scene}"
    mkdir -p "${scene_result}"

    start_t=$(date +%s)

    # Build temporary config
    tmp_dir="${scene_result}/.tmp"
    mkdir -p "${tmp_dir}"

    python -c "
import yaml, copy, sys
sys.path.insert(0, '${WORKDIR}')
from Utility.Config import IncludeLoader

with open('${WORKDIR}/Config/Experiment/MACVO/MACVO_HoloOcean_IMU.yaml') as f:
    cfg = yaml.load(f, IncludeLoader)
with open('${WORKDIR}/Config/Sequence/holoocean_imu.yaml') as f:
    seq = yaml.load(f, IncludeLoader)

cfg = copy.deepcopy(cfg)
od = cfg['Odometry']
op = od['optimizer']['args']
op['post_imu_fusion_enable'] = False
op['post_imu_fusion_mode'] = 'none'
op['autodiff'] = False
od['args']['imu_rot_prior_enable'] = True
od['args']['imu_trans_prior_enable'] = True
op['imu_rot_prior'] = True
seq['args']['root'] = '${BATCH_ROOT}/${scene}'

import pathlib
pathlib.Path('${tmp_dir}/odom.yaml').write_text(yaml.safe_dump(cfg))
pathlib.Path('${tmp_dir}/seq.yaml').write_text(yaml.safe_dump(seq))
print('Config built: ${scene}')
"

    # Run MACVO with V3b adaptive gate (tqdm output preserved)
    python "${WORKDIR}/MACVO.py" \
        --odom "${tmp_dir}/odom.yaml" \
        --data "${tmp_dir}/seq.yaml" \
        --resultRoot "${scene_result}" \
        --adaptive-v3b

    # Flatten nested output directory
    for nested_dir in $(find "${scene_result}" -name "poses.csv" -exec dirname {} \; 2>/dev/null); do
        if [ "${nested_dir}" != "${scene_result}" ]; then
            for f in "${nested_dir}"/*; do
                if [ -f "$f" ]; then
                    dest="${scene_result}/$(basename $f)"
                    [ ! -e "$dest" ] && mv "$f" "$dest" 2>/dev/null || true
                fi
            done
            rmdir "${nested_dir}" 2>/dev/null || true
        fi
    done

    elapsed=$(($(date +%s) - start_t))

    # Evaluate direct ATE
    poses_csv="${scene_result}/poses.csv"
    ref_csv="${BATCH_ROOT}/${scene}/ref_pose.csv"
    if [ -f "${poses_csv}" ] && [ -f "${ref_csv}" ]; then
        ate=$(python -c "
import numpy as np
e = np.genfromtxt('${poses_csv}', delimiter=',', dtype=float, skip_header=1)
g = np.genfromtxt('${ref_csv}', delimiter=',', dtype=float, skip_header=1)
if e.ndim==1: e=e.reshape(1,-1)
if g.ndim==1: g=g.reshape(1,-1)
n=min(len(e),len(g))
print(f'{np.sqrt(np.mean(np.sum((e[:n,1:4]-g[:n,1:4])**2,axis=1))):.1f}')
" 2>/dev/null || echo "nan")
        echo "  ✅ ${scene}: direct_ATE=${ate}m  (${elapsed}s)"
    else
        ate="nan"
        echo "  ⚠ ${scene}: ATE unavailable (${elapsed}s)"
    fi

    echo "${scene},${ate}" >> "${RESULT_DIR}/ate_summary.csv"

done

# ── Final Summary ────────────────────────────────────────────────────
echo ""
echo "============================================================================"
echo "V3B SANITY CHECK COMPLETE — $(date)"
echo "============================================================================"
echo "Results: ${RESULT_DIR}"
echo ""
echo "Per-scene direct ATE:"
cat "${RESULT_DIR}/ate_summary.csv"
echo ""
echo "Output files per scene:"
echo "  poses.csv, frame_pair_diagnostics.csv, adaptive_decisions.csv, config.yaml"
echo ""
echo "============================================================================"
