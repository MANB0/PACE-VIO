#!/bin/bash
# V3b Debug Sanity — 5-scene re-run (dam, murky, open, deep, caustic)
set -e

BATCH_ROOT="/mnt/e/文档/holoocean/code/recordings/batch_20260515_155653"
WORKDIR="/home/admin1/macvo-dev"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULT_DIR="${WORKDIR}/Results/v3b_debug_sanity_${TIMESTAMP}"

SCENES=("dam_inspection" "murky_coast" "open_water" "deep_dark" "caustic_shallow")

eval "$(conda shell.bash hook)"
conda activate macvo
echo "Python: $(which python)"
mkdir -p "${RESULT_DIR}"
echo "scene,direct_ATE" > "${RESULT_DIR}/ate_summary.csv"

echo "============================================================================"
echo "V3b Debug Sanity — ${TIMESTAMP}"
echo "Scenes: ${SCENES[*]}"
echo "Results: ${RESULT_DIR}"
echo "============================================================================"

total=${#SCENES[@]}
for i in "${!SCENES[@]}"; do
    scene="${SCENES[$i]}"
    idx=$((i + 1))
    echo ""
    echo "===== [${idx}/${total}] ${scene} ====="

    scene_result="${RESULT_DIR}/${scene}"
    mkdir -p "${scene_result}"
    tmp_dir="${scene_result}/.tmp"
    mkdir -p "${tmp_dir}"

    start_t=$(date +%s)

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
print('Config OK: ${scene}')
"

    python "${WORKDIR}/MACVO.py" \
        --odom "${tmp_dir}/odom.yaml" \
        --data "${tmp_dir}/seq.yaml" \
        --resultRoot "${scene_result}" \
        --adaptive-v3b

    # Flatten
    for nd in $(find "${scene_result}" -name "poses.csv" -exec dirname {} \; 2>/dev/null); do
        [ "$nd" != "${scene_result}" ] && for f in "$nd"/*; do
            [ -f "$f" ] && [ ! -e "${scene_result}/$(basename $f)" ] && mv "$f" "${scene_result}/" 2>/dev/null
        done && rmdir "$nd" 2>/dev/null
    done

    elapsed=$(($(date +%s) - start_t))

    poses_csv="${scene_result}/poses.csv"
    ref_csv="${BATCH_ROOT}/${scene}/ref_pose.csv"
    if [ -f "${poses_csv}" ] && [ -f "${ref_csv}" ]; then
        ate=$(python -c "
import numpy as np
e=np.genfromtxt('${poses_csv}',delimiter=',',dtype=float,skip_header=1)
g=np.genfromtxt('${ref_csv}',delimiter=',',dtype=float,skip_header=1)
if e.ndim==1: e=e.reshape(1,-1)
if g.ndim==1: g=g.reshape(1,-1)
n=min(len(e),len(g))
print(f'{np.sqrt(np.mean(np.sum((e[:n,1:4]-g[:n,1:4])**2,axis=1))):.1f}')
" 2>/dev/null || echo "nan")
        echo "  ✅ ${scene}: ATE=${ate}m (${elapsed}s)"
    else
        ate="nan"
        echo "  ⚠ ${scene}: ATE unavailable (${elapsed}s)"
    fi
    echo "${scene},${ate}" >> "${RESULT_DIR}/ate_summary.csv"
done

echo ""
echo "============================================================================"
echo "DEBUG SANITY COMPLETE"
echo "============================================================================"
cat "${RESULT_DIR}/ate_summary.csv"
