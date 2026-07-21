#!/usr/bin/env python3
"""Run the post-fix W=3 local inertial BA validation batch.

This script intentionally only starts the run and the progress dashboard.  The
combined old/new trajectory comparison is handled after the user confirms the
batch has completed.
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


WORKDIR = Path("/home/admin1/macvo-dev")
PYTHON = Path("/home/admin1/miniconda3/envs/macvo/bin/python")
if not PYTHON.exists():
    PYTHON = Path(sys.executable)

SCENE = "clear_rectangle_normal_noise"
W3_VARIANTS = ["vio_local_ba_w3_imuatt", "vio_local_ba_w3_imuatt_all"]

DEFAULT_RESULT_ROOT = WORKDIR / "Results" / "w3_after_fix_validation_20260709"
DEFAULT_OUTPUT_ROOT = WORKDIR / "analysis_w3_after_fix_validation_20260709"
DEFAULT_LOG_PATH = WORKDIR / "logs" / "w3_after_fix_validation_20260709.log"
DEFAULT_DASHBOARD_LOG = WORKDIR / "logs" / "progress_dashboard_8765.log"
DEFAULT_W2_ANALYSIS_ROOT = WORKDIR / "analysis_w2_equivalence_biaslinfix_20260709"
DEFAULT_PURE_ANALYSIS_ROOT = WORKDIR / "analysis_closed_paths_latest_20260706_timeinterp"


@dataclass(frozen=True)
class ComparisonSpec:
    method: str
    root: Path
    source_variant: str


def build_run_command(args: argparse.Namespace) -> list[str]:
    cmd = [
        str(PYTHON),
        "Scripts/run_latest_closed_paths_methods.py",
        "--result-root",
        str(args.result_root),
        "--output-root",
        str(args.output_root),
        "--scenes",
        SCENE,
        "--variants",
        *W3_VARIANTS,
        "--jobs",
        str(int(args.jobs)),
        "--timeout",
        str(int(args.timeout)),
    ]
    if args.seq_to is not None:
        cmd.extend(["--seq-to", str(int(args.seq_to))])
    if bool(args.overwrite_manifest):
        cmd.append("--overwrite-manifest")
    return cmd


def build_dashboard_command(result_root: Path, log_path: Path, *, port: int) -> list[str]:
    return [
        str(PYTHON),
        "Scripts/run_progress_dashboard.py",
        "--result-root",
        str(result_root),
        "--log",
        str(log_path),
        "--port",
        str(int(port)),
    ]


def comparison_specs(output_root: Path) -> list[ComparisonSpec]:
    return [
        ComparisonSpec("pure_macvo_reused", DEFAULT_PURE_ANALYSIS_ROOT, "pure_macvo"),
        ComparisonSpec("two_frame_imuatt", DEFAULT_W2_ANALYSIS_ROOT, "vio_preintegrated_full_imuatt_estinit"),
        ComparisonSpec("w2_current", DEFAULT_W2_ANALYSIS_ROOT, "vio_local_ba_w2_imuatt"),
        ComparisonSpec("w3_current", output_root, "vio_local_ba_w3_imuatt"),
        ComparisonSpec("w3_all_optimized", output_root, "vio_local_ba_w3_imuatt_all"),
    ]


def _quote_cmd(cmd: list[str]) -> str:
    return " ".join(subprocess.list2cmdline([part]) for part in cmd)


def _existing_dashboard_pids() -> list[int]:
    proc = subprocess.run(
        ["pgrep", "-f", "Scripts/run_progress_dashboard.py"],
        cwd=str(WORKDIR),
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    pids: list[int] = []
    for raw in proc.stdout.split():
        try:
            pid = int(raw)
        except ValueError:
            continue
        if pid != os.getpid():
            pids.append(pid)
    return pids


def switch_dashboard(result_root: Path, log_path: Path, *, port: int) -> None:
    for pid in _existing_dashboard_pids():
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            continue
    time.sleep(0.5)

    DEFAULT_DASHBOARD_LOG.parent.mkdir(parents=True, exist_ok=True)
    with DEFAULT_DASHBOARD_LOG.open("ab") as f:
        subprocess.Popen(
            build_dashboard_command(result_root, log_path, port=port),
            cwd=str(WORKDIR),
            stdout=f,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    time.sleep(1.0)


def run_batch(args: argparse.Namespace) -> int:
    args.result_root.mkdir(parents=True, exist_ok=True)
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.log_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = build_run_command(args)
    print("Progress dashboard:")
    print(f"  http://127.0.0.1:{int(args.dashboard_port)}/")
    print("Result root:")
    print(f"  {args.result_root}")
    print("Log path:")
    print(f"  {args.log_path}")
    print("Run command:")
    print(f"  {_quote_cmd(cmd)}")

    if args.dry_run:
        return 0

    if not args.no_dashboard:
        switch_dashboard(args.result_root, args.log_path, port=int(args.dashboard_port))

    with args.log_path.open("wb") as log_file:
        proc = subprocess.run(
            cmd,
            cwd=str(WORKDIR),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            check=False,
        )
    print(f"Batch exited with return code {proc.returncode}.")
    print("After completion, notify Codex to generate the combined comparison page.")
    return int(proc.returncode)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--log-path", type=Path, default=DEFAULT_LOG_PATH)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=7200)
    parser.add_argument("--seq-to", type=int, default=None)
    parser.add_argument("--dashboard-port", type=int, default=8765)
    parser.add_argument("--no-dashboard", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite-manifest", action="store_true", default=True)
    return parser.parse_args()


def main() -> int:
    return run_batch(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
