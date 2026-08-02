#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${1:-macvo-t2}"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda was not found. Install Miniconda/Anaconda first." >&2
  exit 1
fi

conda create -y -n "$ENV_NAME" python=3.10 pip cmake ninja
conda run -n "$ENV_NAME" python -m pip install --upgrade pip
conda run -n "$ENV_NAME" python -m pip install \
  torch==2.4.0 torchvision==0.19.0 \
  --index-url https://download.pytorch.org/whl/cu121
conda run -n "$ENV_NAME" python -m pip install -r "$ROOT/requirements.txt"
conda run -n "$ENV_NAME" python "$ROOT/Scripts/download_models.py"
conda run -n "$ENV_NAME" bash "$ROOT/Scripts/bootstrap_gtsam_local.sh"
GTSAM_DIR="$ROOT/.deps/gtsam-install/lib/cmake/GTSAM"
if [[ ! -f "$GTSAM_DIR/GTSAMConfig.cmake" && -f "$HOME/.local/lib/cmake/GTSAM/GTSAMConfig.cmake" ]]; then
  GTSAM_DIR="$HOME/.local/lib/cmake/GTSAM"
fi
PYTHON_BIN="$(conda run -n "$ENV_NAME" which python)" \
  GTSAM_DIR="$GTSAM_DIR" \
  bash "$ROOT/Scripts/build_pace_vio_isam2.sh"
conda run -n "$ENV_NAME" python "$ROOT/Scripts/check_runtime.py" --require-isam2

echo "Environment ready. Activate it with: conda activate $ENV_NAME"
