#!/usr/bin/env python3
"""Run one manually started full-sequence VIO causal diagnostic.

The formal schedule is deliberately fixed to clear_rectangle_normal_noise and
vio_preintegrated_full_imuatt_estinit. Analysis is performed only after the user
confirms the run has completed.
"""

from __future__ import annotations

import argparse
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
if str(WORKDIR) not in sys.path:
    sys.path.insert(0, str(WORKDIR))

from Scripts import run_vio_imu_prior_mode_grid as grid


SCENE = "clear_rectangle_normal_noise"
VARIANT = "vio_preintegrated_full_imuatt_estinit"
SCENE_ROOT = Path(
    "/mnt/e/文档/holoocean/code/recordings/"
    "batch_zed100_closed_paths_smooth_20260705/normal_noise/clear_rectangle_path"
)
DEFAULT_RESULT_ROOT = WORKDIR / "Results" / "vio_causal_diagnostics_rectangle_normal_20260710"
DEFAULT_DASHBOARD_LOG = WORKDIR / "logs" / "progress_dashboard_8765.log"


def build_specs(result_root: Path) -> list[grid.RunSpec]:
    grid.SCENE_ROOTS[SCENE] = SCENE_ROOT
    return grid.build_specs(
        scenes=[SCENE],
        variants=[VARIANT],
        trials=1,
        result_root=result_root,
    )


def make_causal_odom_cfg(config_dir: Path, *, influence_interval: int) -> Path:
    config_dir.mkdir(parents=True, exist_ok=True)
    path = grid.make_odom_cfg(grid.VARIANTS[VARIANT], config_dir, autodiff=False)
    cfg = grid.load_yaml(path)
    optimizer = cfg["Odometry"]["optimizer"]["args"]
    optimizer["vio_causal_diagnostics_enable"] = True
    optimizer["vio_causal_diagnostics_interval"] = max(1, int(influence_interval))
    # Make the no-tuning contract explicit in the archived run configuration.
    optimizer["imu_vio_alpha_p"] = 1.0
    optimizer["imu_vio_alpha_v"] = 1.0
    optimizer["imu_vio_alpha_R"] = 1.0
    grid.write_yaml(path, cfg)
    return path


def build_dashboard_command(result_root: Path, *, port: int) -> list[str]:
    return [
        str(PYTHON),
        "Scripts/run_progress_dashboard.py",
        "--result-root",
        str(result_root),
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


def switch_dashboard(result_root: Path, *, port: int) -> None:
    for pid in _existing_dashboard_pids():
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            continue
    time.sleep(0.5)

    DEFAULT_DASHBOARD_LOG.parent.mkdir(parents=True, exist_ok=True)
    with DEFAULT_DASHBOARD_LOG.open("ab") as log_file:
        subprocess.Popen(
            build_dashboard_command(result_root, port=port),
            cwd=str(WORKDIR),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    time.sleep(1.0)


def run_batch(args: argparse.Namespace) -> int:
    specs = build_specs(args.result_root)
    print("=" * 78)
    print("  VIO causal diagnostic: factor-victory audit")
    print(f"  Scene:      {SCENE}")
    print(f"  Method:     {VARIANT}")
    print(f"  Sequence:   {args.seq_to if args.seq_to is not None else 'full'}")
    print("  Jobs:       1")
    print(f"  Influence:  every {max(1, int(args.influence_interval))} frame pairs")
    print(f"  Results:    {args.result_root}")
    print("=" * 78)

    if not grid.sanity_check(specs):
        return 1

    if args.dry_run:
        spec = specs[0]
        print("DRY RUN - no dashboard or MACVO process will be started.")
        print(f"Would run: scene={spec.scene} variant={spec.variant.name}")
        print(f"Result dir: {spec.result_dir}")
        return 0

    args.result_root.mkdir(parents=True, exist_ok=True)
    manifest = args.result_root / "run_manifest.csv"
    if manifest.exists() and not args.overwrite_manifest:
        if not grid.manifest_matches_specs(args.result_root, specs, args.seq_to, autodiff=False):
            print(f"ERROR: existing manifest does not match the fixed schedule: {manifest}")
            print("Choose a fresh --result-root or pass --overwrite-manifest.")
            return 1
    else:
        grid.write_manifest_guarded(
            args.result_root,
            specs,
            args.seq_to,
            autodiff=False,
            overwrite=bool(args.overwrite_manifest),
        )

    config_dir = args.result_root / "run_configs"
    odom_cfg = make_causal_odom_cfg(
        config_dir,
        influence_interval=int(args.influence_interval),
    )
    if not args.no_dashboard:
        switch_dashboard(args.result_root, port=int(args.dashboard_port))
        print(f"Progress dashboard: http://127.0.0.1:{int(args.dashboard_port)}/")

    failures = grid.execute_run_schedule(
        specs,
        {VARIANT: odom_cfg},
        config_dir,
        args.result_root,
        timeout_s=int(args.timeout),
        seq_to=args.seq_to,
        jobs=1,
    )
    print(f"Manifest: {manifest}")
    print(f"Progress: {args.result_root / 'progress.csv'}")
    if failures:
        for spec, return_code in failures:
            print(f"FAILED: {spec.scene} / {spec.variant.name}, rc={return_code}")
        return 1
    print("Sequence finished. Notify Codex before running the causal analysis.")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--seq-to", type=int, default=None, help="Optional debugging crop; omit for the formal full sequence.")
    parser.add_argument("--timeout", type=int, default=grid.RUN_TIMEOUT_S)
    parser.add_argument("--influence-interval", type=int, default=5)
    parser.add_argument("--dashboard-port", type=int, default=8765)
    parser.add_argument("--no-dashboard", action="store_true")
    parser.add_argument("--overwrite-manifest", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    return run_batch(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
