#!/usr/bin/env bash
#SBATCH --partition=gpu
#SBATCH --job-name=pace-paper-2x
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=pace-paper-2x-%j.out

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
CONDA_ENV="${CONDA_ENV:-pace-vio}"
MANIFEST="${MANIFEST:-$PROJECT_ROOT/Config/paper_experiments.json}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
cd "$PROJECT_ROOT"

export LD_LIBRARY_PATH="$PROJECT_ROOT/.deps/gtsam-install/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

echo "Job: ${SLURM_JOB_ID:-interactive}"
echo "Manifest: $MANIFEST"
nvidia-smi -L

python Scripts/run_paper_experiments.py --manifest "$MANIFEST"
