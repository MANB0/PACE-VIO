#!/usr/bin/env python3
"""Run full normal-noise circle, rectangle, and straight SA-v2 replays."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from Scripts.run_visual_factor_cache_batch import switch_dashboard  # noqa: E402


WORKDIR = Path("/home/admin1/macvo-dev")
PYTHON = Path("/home/admin1/miniconda3/envs/macvo/bin/python")
RUNNER = WORKDIR / "Scripts/run_static63_cached_imu_fusion.py"
BATCH_ROOT = Path(
    "/mnt/e/\u6587\u6863/holoocean/code/recordings/"
    "batch_clear_truth_paths_20260713_static63_variants"
)
DEFAULT_RESULT_ROOT = (
    WORKDIR / "Results/normal_noise_sa_v2_full_three_scenes_20260717"
)
VARIANT = "vio_two_state_direct_uvd_sampling_aware_v2_full"

SCENES = {
    "circle": "clear_circle_truth_normal_noise",
    "rectangle": "clear_stop_turn_rectangle_truth_normal_noise",
    "straight": "clear_straight_truth_normal_noise",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--jobs", type=int, default=len(SCENES))
    parser.add_argument(
        "--threads-per-run",
        type=int,
        default=max(1, (os.cpu_count() or len(SCENES)) // len(SCENES)),
        help="CPU thread limit inherited by each MACVO process.",
    )
    parser.add_argument("--timeout", type=int, default=43200)
    parser.add_argument("--monitor-interval", type=float, default=15.0)
    parser.add_argument("--dashboard-port", type=int, default=8765)
    parser.add_argument("--no-dashboard", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def count_csv_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8", errors="replace") as stream:
        return max(0, sum(1 for _ in stream) - 1)


def expected_frames(scene: str) -> int:
    return count_csv_rows(BATCH_ROOT / scene / "ref_pose.csv")


def result_dir(result_root: Path, scene: str) -> Path:
    return result_root / "trial_1" / VARIANT / scene


def task_log(result_root: Path, scene: str) -> Path:
    return result_root / "logs" / f"{VARIANT}__{scene}.log"


def latest_diagnostics_frame(root: Path) -> int | None:
    candidates = sorted(root.rglob("frame_pair_diagnostics.csv")) if root.exists() else []
    if not candidates:
        return None
    path = max(candidates, key=lambda item: item.stat().st_mtime)
    try:
        with path.open("r", newline="", encoding="utf-8", errors="replace") as stream:
            rows = list(csv.DictReader(stream))
    except Exception:
        return None
    for row in reversed(rows):
        value = row.get("frame_j", "") or row.get("frame_idx", "")
        try:
            return int(float(value))
        except (TypeError, ValueError):
            continue
    return None


def latest_logged_frame(path: Path) -> int | None:
    if not path.exists():
        return None
    with path.open("rb") as stream:
        stream.seek(0, os.SEEK_END)
        size = stream.tell()
        stream.seek(max(0, size - 512 * 1024), os.SEEK_SET)
        tail = stream.read()
    matches = re.findall(rb"VisualMap\(#frame=(\d+)", tail)
    return max((int(value) for value in matches), default=None)


def scene_progress(result_root: Path, scene: str, elapsed: float) -> dict[str, object]:
    total = expected_frames(scene)
    frame = latest_diagnostics_frame(result_dir(result_root, scene))
    if frame is None:
        frame = latest_logged_frame(task_log(result_root, scene))
    completed = count_csv_rows(result_dir(result_root, scene) / "poses.csv")
    if completed >= total > 0:
        current = total
        status = "complete"
    else:
        current = 0 if frame is None else min(frame + 1, total)
        status = "running" if current > 0 else "pending"
    rate = current / elapsed if elapsed > 0.0 and current > 0 else 0.0
    eta = (total - current) / rate if rate > 0.0 else None
    return {
        "scene": scene,
        "trajectory": next(name for name, value in SCENES.items() if value == scene),
        "status": status,
        "frame": int(current),
        "total_frames": int(total),
        "percent": 100.0 * current / total if total else 0.0,
        "elapsed_s": float(elapsed),
        "eta_s": None if eta is None else float(eta),
        "result_dir": str(result_dir(result_root, scene)),
        "task_log": str(task_log(result_root, scene)),
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }


def duration(value: object) -> str:
    if value is None:
        return "--"
    seconds = max(0, int(float(value)))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def write_monitor_files(
    result_root: Path,
    records: list[dict[str, object]],
    *,
    command: list[str],
    return_code: int | None,
) -> None:
    fields = [
        "updated_at",
        "trajectory",
        "scene",
        "status",
        "frame",
        "total_frames",
        "percent",
        "elapsed_s",
        "eta_s",
        "result_dir",
        "task_log",
    ]
    with (result_root / "sa_v2_three_scene_live_progress.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    payload = {
        "method": "SA-v2",
        "full_sequence": True,
        "imu_config": "normal_noise",
        "visual_factor": "direct_uvd",
        "preintegration": "standard_local_frame_preintegration",
        "covariance_mode": "sampling_aware_cross_edge",
        "return_code": return_code,
        "command": command,
        "scenes": records,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }
    (result_root / "sa_v2_three_scene_run_status.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def build_command(args: argparse.Namespace, result_root: Path) -> list[str]:
    command = [
        str(PYTHON),
        str(RUNNER),
        "--method",
        "two-state-fixed-lag",
        "--result-root",
        str(result_root),
        "--variant-name",
        VARIANT,
        "--scenes",
        *SCENES.values(),
        "--imu-vio-gravity-handling",
        "standard_local_frame_preintegration",
        "--two-state-visual-factor-mode",
        "direct_uvd",
        "--two-state-warm-start",
        "macvo_pose",
        "--two-state-uvd-huber-delta",
        "0.1",
        "--two-state-covariance-mode",
        "sampling_aware_cross_edge",
        "--jobs",
        str(int(args.jobs)),
        "--timeout",
        str(int(args.timeout)),
        "--no-dashboard",
    ]
    if args.force:
        command.append("--force")
    if args.dry_run:
        command.append("--dry-run")
    return command


def main() -> int:
    args = parse_args()
    if args.jobs <= 0 or args.jobs > len(SCENES):
        raise ValueError(f"--jobs must be between 1 and {len(SCENES)}")
    if args.threads_per_run <= 0:
        raise ValueError("--threads-per-run must be positive")
    if args.monitor_interval <= 0.0:
        raise ValueError("--monitor-interval must be positive")
    if args.timeout <= 0:
        raise ValueError("--timeout must be positive")

    for scene in SCENES.values():
        for required in (
            BATCH_ROOT / scene / "imu_data.csv",
            BATCH_ROOT / scene / "ref_pose.csv",
            BATCH_ROOT / scene / "metadata.json",
        ):
            if not required.exists():
                raise FileNotFoundError(required)

    result_root = args.result_root.expanduser().resolve()
    result_root.mkdir(parents=True, exist_ok=True)
    command = build_command(args, result_root)
    print("Command:")
    print(" ".join(command))
    if args.dry_run:
        return subprocess.run(command, cwd=WORKDIR, check=False).returncode

    monitor_log = result_root / "sa_v2_three_scene_monitor.log"
    if not args.no_dashboard:
        switch_dashboard(result_root, monitor_log, port=int(args.dashboard_port))
        print(f"Progress dashboard: http://127.0.0.1:{int(args.dashboard_port)}/")

    environment = os.environ.copy()
    thread_count = str(int(args.threads_per_run))
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        environment[name] = thread_count

    runner_log = result_root / "sa_v2_three_scene_runner.log"
    started = time.monotonic()
    with runner_log.open("w", encoding="utf-8") as stream:
        process = subprocess.Popen(
            command,
            cwd=str(WORKDIR),
            env=environment,
            stdout=stream,
            stderr=subprocess.STDOUT,
        )
        return_code: int | None = None
        while return_code is None:
            elapsed = time.monotonic() - started
            records = [
                scene_progress(result_root, scene, elapsed)
                for scene in SCENES.values()
            ]
            write_monitor_files(
                result_root,
                records,
                command=command,
                return_code=None,
            )
            summary = " | ".join(
                f"{record['trajectory']}: {record['frame']}/{record['total_frames']} "
                f"({float(record['percent']):.1f}%), ETA {duration(record['eta_s'])}"
                for record in records
            )
            line = f"[{datetime.now().strftime('%H:%M:%S')}] {summary}"
            print(line, flush=True)
            with monitor_log.open("a", encoding="utf-8") as monitor:
                monitor.write(line + "\n")
            try:
                return_code = process.wait(timeout=float(args.monitor_interval))
            except subprocess.TimeoutExpired:
                return_code = None

    elapsed = time.monotonic() - started
    records = [
        scene_progress(result_root, scene, elapsed) for scene in SCENES.values()
    ]
    for record in records:
        if return_code != 0 and record["status"] != "complete":
            record["status"] = "failed"
    write_monitor_files(
        result_root,
        records,
        command=command,
        return_code=int(return_code),
    )
    print(f"Runner finished: rc={return_code}, elapsed={duration(elapsed)}")
    print(f"Runner log: {runner_log}")
    return int(return_code)


if __name__ == "__main__":
    raise SystemExit(main())
