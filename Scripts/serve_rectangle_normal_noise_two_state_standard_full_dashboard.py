#!/usr/bin/env python3
"""Serve the manual-launch panel for the rectangle normal-noise full run."""

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from Scripts.run_visual_factor_cache_batch import switch_dashboard


WORKDIR = Path("/home/admin1/macvo-dev")
RESULT_ROOT = WORKDIR / "Results" / "rectangle_normal_noise_two_state_standard_full_20260715"
LAUNCH_SCRIPT = WORKDIR / "Scripts" / "run_rectangle_normal_noise_two_state_standard_full.sh"


def main() -> int:
    launch_log = RESULT_ROOT / "manual_launch.log"
    switch_dashboard(
        RESULT_ROOT,
        launch_log,
        port=8765,
        launch_script=LAUNCH_SCRIPT,
        launch_log=launch_log,
        host="127.0.0.1",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
