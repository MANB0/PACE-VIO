#!/usr/bin/env bash
set -euo pipefail

# Legacy launcher retained for archived deployment instructions.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PACE_VIO_ISAM2_BUILD_DIR="${T2_ISAM2_BUILD_DIR:-${PACE_VIO_ISAM2_BUILD_DIR:-$ROOT/build/pace_vio_isam2}}"
exec "$ROOT/Scripts/build_pace_vio_isam2.sh" "$@"
