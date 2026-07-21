#!/usr/bin/env python3
"""Validate current-only vs all-optimized Local BA writeback on rectangle scenes.

This is a focused batch, not a publication batch. It uses short 150-frame
windows by default so W=2/W=3 behavior can be compared quickly.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

WORKDIR = Path("/home/admin1/macvo-dev")
if str(WORKDIR) not in sys.path:
    sys.path.insert(0, str(WORKDIR))

from Scripts import run_latest_closed_paths_methods as latest
from Scripts import run_vio_imu_prior_mode_grid as grid


VALIDATION_SCENES = [
    "clear_rectangle_zero_noise",
    "clear_rectangle_normal_noise",
]

BASELINE_VARIANTS = [
    "pure_macvo",
    "vio_preintegrated_full_imuatt_estinit",
]

LOCAL_BA_VARIANTS = [
    "vio_local_ba_w2_imuatt",
    "vio_local_ba_w2_imuatt_all",
    "vio_local_ba_w3_imuatt",
    "vio_local_ba_w3_imuatt_all",
]

VALIDATION_VARIANTS = BASELINE_VARIANTS + LOCAL_BA_VARIANTS

DEFAULT_RESULT_ROOT = WORKDIR / "Results" / "local_ba_writeback_validation_20260708"
DEFAULT_OUTPUT_ROOT = WORKDIR / "analysis_local_ba_writeback_validation_20260708"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--variants", nargs="*", default=VALIDATION_VARIANTS)
    parser.add_argument("--scenes", nargs="*", default=VALIDATION_SCENES)
    parser.add_argument("--seq-to", type=int, default=150)
    parser.add_argument("--timeout", type=int, default=grid.RUN_TIMEOUT_S)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--run-only", action="store_true")
    parser.add_argument("--analyze-only", action="store_true")
    parser.add_argument("--overwrite-manifest", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.analyze_only and args.run_only:
        print("ERROR: --analyze-only and --run-only cannot be combined.")
        return 1

    rc = 0
    if not args.analyze_only:
        rc = latest.run_batch(args)
        if rc != 0 or args.run_only or args.dry_run:
            return rc
    return latest.analyze_batch(args)


if __name__ == "__main__":
    raise SystemExit(main())
