#!/usr/bin/env python3
"""Run the manually started gravity-handling A/B isolation batch.

The launcher schedules only the current preintegration-compensated VIO and the
standard residual-gravity VIO on the existing rectangle zero/normal-noise
scenes. It does not generate HoloOcean data. Final comparison artifacts are
built only after the user confirms that the batch has completed.
"""

from __future__ import annotations

import argparse
import csv
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


WORKDIR = Path("/home/admin1/macvo-dev")
PYTHON = Path("/home/admin1/miniconda3/envs/macvo/bin/python")
if not PYTHON.exists():
    PYTHON = Path(sys.executable)

SCENES = [
    "clear_rectangle_zero_noise",
    "clear_rectangle_normal_noise",
]
VARIANTS = [
    "vio_preintegrated_full_imuatt_estinit",
    "vio_preintegrated_full_residual_gravity",
]

DEFAULT_RESULT_ROOT = WORKDIR / "Results" / "gravity_handling_ablation_20260711"
DEFAULT_OUTPUT_ROOT = WORKDIR / "analysis_gravity_handling_ablation_20260711"
DEFAULT_LOG_PATH = WORKDIR / "logs" / "gravity_handling_ablation_20260711.log"
DEFAULT_DASHBOARD_LOG = WORKDIR / "logs" / "progress_dashboard_8765.log"


def build_run_command(args: argparse.Namespace) -> list[str]:
    cmd = [
        str(PYTHON),
        "Scripts/run_latest_closed_paths_methods.py",
        "--result-root",
        str(args.result_root),
        "--output-root",
        str(args.output_root),
        "--scenes",
        *SCENES,
        "--variants",
        *VARIANTS,
        "--jobs",
        str(int(args.jobs)),
        "--timeout",
        str(int(args.timeout)),
        "--run-only",
        "--overwrite-manifest",
    ]
    if args.seq_to is not None:
        cmd.extend(["--seq-to", str(int(args.seq_to))])
    if args.dry_run:
        cmd.append("--dry-run")
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
    with DEFAULT_DASHBOARD_LOG.open("ab") as dashboard_log:
        subprocess.Popen(
            build_dashboard_command(result_root, log_path, port=port),
            cwd=str(WORKDIR),
            stdout=dashboard_log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    time.sleep(1.0)


def _quote_command(cmd: list[str]) -> str:
    return " ".join(subprocess.list2cmdline([part]) for part in cmd)


def _read_visual_fingerprints(result_dir: Path) -> dict[tuple[str, str], str]:
    diagnostics = result_dir / "frame_pair_diagnostics.csv"
    if not diagnostics.exists():
        return {}
    fingerprints: dict[tuple[str, str], str] = {}
    with diagnostics.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            value = str(row.get("visual_input_sha256", "")).strip()
            if value:
                fingerprints[(str(row.get("frame_i", "")), str(row.get("frame_j", "")))] = value
    return fingerprints


def validate_visual_input_identity(result_root: Path) -> tuple[Path, bool]:
    manifest = result_root / "run_manifest.csv"
    if not manifest.exists():
        raise FileNotFoundError(f"Missing run manifest: {manifest}")

    by_scene: dict[str, dict[str, Path]] = {}
    with manifest.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            by_scene.setdefault(str(row["scene"]), {})[str(row["variant"])] = Path(row["result_dir"])

    report_path = result_root / "visual_input_identity.csv"
    all_exact = True
    report_rows: list[dict[str, object]] = []
    for scene in sorted(by_scene):
        variants = by_scene[scene]
        legacy = _read_visual_fingerprints(variants.get(VARIANTS[0], Path("/__missing_legacy__")))
        residual = _read_visual_fingerprints(variants.get(VARIANTS[1], Path("/__missing_residual__")))
        keys = set(legacy) | set(residual)
        mismatched = sum(legacy.get(key) != residual.get(key) for key in keys)
        exact = bool(keys) and mismatched == 0 and set(legacy) == set(residual)
        all_exact = all_exact and exact
        report_rows.append(
            {
                "scene": scene,
                "compared_pairs": len(keys),
                "legacy_pairs": len(legacy),
                "residual_pairs": len(residual),
                "mismatched_pairs": mismatched,
                "exact_match": int(exact),
            }
        )

    with report_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "scene",
                "compared_pairs",
                "legacy_pairs",
                "residual_pairs",
                "mismatched_pairs",
                "exact_match",
            ],
        )
        writer.writeheader()
        writer.writerows(report_rows)
    return report_path, all_exact


def run_batch(args: argparse.Namespace) -> int:
    cmd = build_run_command(args)
    print("Gravity-handling A/B isolation")
    print(f"Progress dashboard: http://127.0.0.1:{int(args.dashboard_port)}/")
    print(f"Result root: {args.result_root}")
    print(f"Log path: {args.log_path}")
    print(f"Command: {_quote_command(cmd)}")

    if args.dry_run:
        return int(subprocess.run(cmd, cwd=str(WORKDIR), check=False).returncode)

    args.result_root.mkdir(parents=True, exist_ok=True)
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.log_path.parent.mkdir(parents=True, exist_ok=True)

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
    if proc.returncode == 0:
        report_path, exact_match = validate_visual_input_identity(args.result_root)
        print(f"Visual input identity report: {report_path}")
        if not exact_match:
            print("ERROR: visual observation fingerprints differ; do not interpret this batch as a causal A/B test.")
            return 3
    print("After completion, notify Codex to analyze and build the combined interactive comparison.")
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
    return parser.parse_args()


def main() -> int:
    return run_batch(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
