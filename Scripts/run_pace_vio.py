#!/usr/bin/env python3
"""Canonical PACE-VIO realtime entry point.

The legacy ``run_realtime_t2.py`` module remains available so archived launch
commands and tests keep working.
"""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Scripts.run_realtime_t2 import *  # noqa: F403
from Scripts.run_realtime_t2 import main


if __name__ == "__main__":
    raise SystemExit(main())
