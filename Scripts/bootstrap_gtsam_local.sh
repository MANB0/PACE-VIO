#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEPS_ROOT="${PACE_VIO_DEPS_ROOT:-$ROOT/.deps}"
SOURCE_DIR="$DEPS_ROOT/gtsam-src"
BUILD_DIR="$DEPS_ROOT/gtsam-build"
INSTALL_DIR="$DEPS_ROOT/gtsam-install"
GTSAM_REF="${PACE_VIO_GTSAM_REF:-4.3a1}"

for tool in git cmake c++; do
  if ! command -v "$tool" >/dev/null 2>&1; then
    echo "Required build tool is missing: $tool" >&2
    exit 1
  fi
done

if [[ -n "${GTSAM_DIR:-}" && -f "$GTSAM_DIR/GTSAMConfig.cmake" ]]; then
  echo "Using GTSAM_DIR=$GTSAM_DIR"
  exit 0
fi
if [[ -f "$HOME/.local/lib/cmake/GTSAM/GTSAMConfig.cmake" ]]; then
  echo "Using existing GTSAM at $HOME/.local"
  exit 0
fi
if [[ -f "$INSTALL_DIR/lib/cmake/GTSAM/GTSAMConfig.cmake" ]]; then
  echo "Using project-local GTSAM at $INSTALL_DIR"
  exit 0
fi

mkdir -p "$DEPS_ROOT"
if [[ ! -d "$SOURCE_DIR/.git" ]]; then
  git clone --depth 1 --branch "$GTSAM_REF" \
    https://github.com/borglab/gtsam.git "$SOURCE_DIR"
fi

cmake_args=(
  -S "$SOURCE_DIR"
  -B "$BUILD_DIR"
  -DCMAKE_BUILD_TYPE=Release
  -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR"
  -DBUILD_SHARED_LIBS=ON
  -DGTSAM_BUILD_TESTS=OFF
  -DGTSAM_BUILD_EXAMPLES_ALWAYS=OFF
  -DGTSAM_BUILD_UNSTABLE=OFF
  -DGTSAM_BUILD_PYTHON=OFF
  -DGTSAM_BUILD_WITH_MARCH_NATIVE=OFF
  -DGTSAM_WITH_TBB=OFF
  -DGTSAM_USE_SYSTEM_METIS=OFF
  -DCMAKE_BUILD_RPATH_USE_ORIGIN=ON
  '-DCMAKE_INSTALL_RPATH=$ORIGIN'
  -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=ON
)
if [[ -n "${CONDA_PREFIX:-}" ]]; then
  cmake_args+=( -DCMAKE_PREFIX_PATH="$CONDA_PREFIX" )
fi
cmake "${cmake_args[@]}"
cmake --build "$BUILD_DIR" -j "${BUILD_JOBS:-$(nproc)}"
cmake --install "$BUILD_DIR"

echo "Project-local GTSAM ready: $INSTALL_DIR"
