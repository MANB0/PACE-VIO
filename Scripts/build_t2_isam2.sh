#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
BUILD_DIR="${T2_ISAM2_BUILD_DIR:-$ROOT/build/t2_isam2}"

if [[ -z "${PYTHON_BIN}" ]]; then
  echo "Python was not found in the active environment." >&2
  exit 1
fi

cmake_args=(
  -S "$ROOT/cpp/t2_isam2"
  -B "$BUILD_DIR"
  -DPython_EXECUTABLE="$PYTHON_BIN"
)

if [[ -n "${GTSAM_DIR:-}" ]]; then
  cmake_args+=( -DGTSAM_DIR="$GTSAM_DIR" )
elif [[ -d "$HOME/.local/lib/cmake/GTSAM" ]]; then
  cmake_args+=( -DGTSAM_DIR="$HOME/.local/lib/cmake/GTSAM" )
fi

if pybind11_dir="$($PYTHON_BIN -m pybind11 --cmakedir 2>/dev/null)"; then
  cmake_args+=( -Dpybind11_DIR="$pybind11_dir" )
elif [[ -n "${T2_PYBIND11_SOURCE_DIR:-}" ]]; then
  cmake_args+=( -DT2_PYBIND11_SOURCE_DIR="$T2_PYBIND11_SOURCE_DIR" )
fi

cmake "${cmake_args[@]}"
cmake --build "$BUILD_DIR" -j "${BUILD_JOBS:-$(nproc)}"
ctest --test-dir "$BUILD_DIR" --output-on-failure

echo "T2 iSAM2 backend ready: $BUILD_DIR/python"
