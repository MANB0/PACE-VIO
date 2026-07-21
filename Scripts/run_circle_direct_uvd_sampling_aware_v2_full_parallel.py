#!/usr/bin/env python3
"""Run full-circle sampling-aware v1 and v2 replays concurrently."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from Scripts.run_visual_factor_cache_batch import switch_dashboard  # noqa: E402


WORKDIR = Path("/home/admin1/macvo-dev")
PYTHON = Path("/home/admin1/miniconda3/envs/macvo/bin/python")
RUNNER = WORKDIR / "Scripts/run_static63_cached_imu_fusion.py"
PLOTTER = WORKDIR / "Scripts/plot_circle_direct_uvd_sampling_aware_v2_full.py"
SCENE = "clear_circle_truth_normal_noise"
SCENE_ROOT = Path(
    "/mnt/e/\u6587\u6863/holoocean/code/recordings/"
    "batch_clear_truth_paths_20260713_static63_variants"
) / SCENE
DEFAULT_RESULT_ROOT = (
    WORKDIR / "Results/circle_direct_uvd_sampling_aware_v2_full_20260717"
)
DEFAULT_ANALYSIS_DIR = (
    WORKDIR / "analysis_circle_direct_uvd_sampling_aware_v2_full_20260717"
)

CASES = {
    "sampling_aware_v1": {
        "variant": "vio_two_state_direct_uvd_sampling_aware_v1_full",
        "covariance_mode": "sampling_aware",
    },
    "sampling_aware_v2": {
        "variant": "vio_two_state_direct_uvd_sampling_aware_v2_full",
        "covariance_mode": "sampling_aware_cross_edge",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--analysis-dir", type=Path, default=DEFAULT_ANALYSIS_DIR)
    parser.add_argument("--timeout", type=int, default=43200)
    parser.add_argument(
        "--threads-per-run",
        type=int,
        default=max(1, (os.cpu_count() or 2) // len(CASES)),
        help="CPU thread budget assigned to each of the two concurrent processes.",
    )
    parser.add_argument(
        "--monitor-interval",
        type=float,
        default=15.0,
        help="Seconds between dashboard/progress-file updates.",
    )
    parser.add_argument("--dashboard-port", type=int, default=8765)
    parser.add_argument("--no-dashboard", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-plot", action="store_true")
    return parser.parse_args()


def case_root(result_root: Path, case: str) -> Path:
    return result_root / case


def case_result_dir(result_root: Path, case: str) -> Path:
    return (
        case_root(result_root, case)
        / "trial_1"
        / str(CASES[case]["variant"])
        / SCENE
    )


def task_log_path(result_root: Path, case: str) -> Path:
    variant = str(CASES[case]["variant"])
    return case_root(result_root, case) / "logs" / f"{variant}__{SCENE}.log"


def latest_logged_frame(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("rb") as stream:
        stream.seek(0, os.SEEK_END)
        size = stream.tell()
        stream.seek(max(0, size - 512 * 1024), os.SEEK_SET)
        tail = stream.read()
    matches = re.findall(rb"VisualMap\(#frame=(\d+)", tail)
    return max((int(value) for value in matches), default=0)


def progress_record(
    case: str,
    *,
    result_root: Path,
    started_monotonic: float,
    status: str,
    expected_frames: int,
) -> dict[str, object]:
    frame = min(latest_logged_frame(task_log_path(result_root, case)), expected_frames)
    elapsed = max(0.0, time.monotonic() - started_monotonic)
    rate = frame / elapsed if frame > 0 and elapsed > 0.0 else 0.0
    eta = (expected_frames - frame) / rate if rate > 0.0 else None
    return {
        "case": case,
        "status": status,
        "frame": int(frame),
        "expected_frames": int(expected_frames),
        "percent": 100.0 * frame / expected_frames,
        "elapsed_s": float(elapsed),
        "eta_s": None if eta is None else float(eta),
        "task_log": str(task_log_path(result_root, case)),
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }


def append_progress_csv(path: Path, records: list[dict[str, object]]) -> None:
    fields = [
        "updated_at",
        "case",
        "status",
        "frame",
        "expected_frames",
        "percent",
        "elapsed_s",
        "eta_s",
        "task_log",
    ]
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerows(records)


def write_dashboard_manifest(path: Path, result_root: Path) -> None:
    fields = [
        "trial",
        "scene",
        "variant",
        "trajectory",
        "imu_config",
        "scene_root",
        "result_dir",
        "seq_to",
        "args",
        "created_at",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for case, spec in CASES.items():
            writer.writerow(
                {
                    "trial": 1,
                    "scene": SCENE,
                    "variant": spec["variant"],
                    "trajectory": "circle",
                    "imu_config": "normal_noise",
                    "scene_root": SCENE_ROOT,
                    "result_dir": case_result_dir(result_root, case),
                    "seq_to": 1890,
                    "args": (
                        "parallel direct-UVD full replay; "
                        f"two_state_covariance_mode={spec['covariance_mode']}"
                    ),
                    "created_at": datetime.now().isoformat(timespec="seconds"),
                }
            )


def append_dashboard_progress(
    path: Path,
    records: list[dict[str, object]],
    *,
    result_root: Path,
) -> None:
    fields = [
        "trial",
        "scene",
        "variant",
        "status",
        "return_code",
        "runtime_s",
        "result_dir",
    ]
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        if not exists:
            writer.writeheader()
        for record in records:
            case = str(record["case"])
            writer.writerow(
                {
                    "trial": 1,
                    "scene": SCENE,
                    "variant": CASES[case]["variant"],
                    "status": record["status"],
                    "return_code": record.get("return_code", ""),
                    "runtime_s": record.get("runtime_s", record.get("elapsed_s", "")),
                    "result_dir": case_result_dir(result_root, case),
                }
            )


def append_dashboard_log(path: Path, message: str) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(f"[{datetime.now().isoformat(timespec='seconds')}] {message}\n")


def format_duration(value: object) -> str:
    if value is None:
        return "--"
    seconds = max(0, int(float(value)))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def build_command(
    case: str,
    *,
    result_root: Path,
    timeout: int,
    force: bool,
    dry_run: bool,
) -> list[str]:
    spec = CASES[case]
    command = [
        str(PYTHON),
        str(RUNNER),
        "--method",
        "two-state-fixed-lag",
        "--result-root",
        str(case_root(result_root, case)),
        "--variant-name",
        str(spec["variant"]),
        "--scenes",
        SCENE,
        "--imu-vio-gravity-handling",
        "standard_local_frame_preintegration",
        "--two-state-visual-factor-mode",
        "direct_uvd",
        "--two-state-warm-start",
        "macvo_pose",
        "--two-state-uvd-huber-delta",
        "0.1",
        "--two-state-covariance-mode",
        str(spec["covariance_mode"]),
        "--jobs",
        "1",
        "--timeout",
        str(int(timeout)),
        "--no-dashboard",
    ]
    if force:
        command.append("--force")
    if dry_run:
        command.append("--dry-run")
    return command


def run_case(
    case: str,
    *,
    command: list[str],
    log_dir: Path,
    threads_per_run: int,
) -> dict[str, object]:
    log_path = log_dir / f"{case}.log"
    environment = os.environ.copy()
    thread_count = str(max(1, int(threads_per_run)))
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        environment[name] = thread_count

    started_at = datetime.now().isoformat(timespec="seconds")
    started = time.monotonic()
    with log_path.open("w", encoding="utf-8") as stream:
        process = subprocess.run(
            command,
            cwd=str(WORKDIR),
            env=environment,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
        )
    elapsed = time.monotonic() - started
    return {
        "case": case,
        "return_code": int(process.returncode),
        "runtime_s": float(elapsed),
        "started_at": started_at,
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "threads_per_run": int(thread_count),
        "command": command,
        "log": str(log_path),
    }


def write_status(path: Path, payload: dict[str, object]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    if args.timeout <= 0:
        raise ValueError("--timeout must be positive")
    if args.threads_per_run <= 0:
        raise ValueError("--threads-per-run must be positive")
    if args.monitor_interval <= 0.0:
        raise ValueError("--monitor-interval must be positive")

    result_root = args.result_root.expanduser().resolve()
    analysis_dir = args.analysis_dir.expanduser().resolve()
    log_dir = result_root / "parallel_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir.mkdir(parents=True, exist_ok=True)

    commands = {
        case: build_command(
            case,
            result_root=result_root,
            timeout=int(args.timeout),
            force=bool(args.force),
            dry_run=bool(args.dry_run),
        )
        for case in CASES
    }
    manifest_path = result_root / "parallel_run_status.json"
    progress_path = result_root / "parallel_progress.csv"
    dashboard_manifest_path = result_root / "run_manifest.csv"
    dashboard_progress_path = result_root / "progress.csv"
    dashboard_log_path = log_dir / "parallel_dashboard.log"
    if progress_path.exists():
        progress_path.write_text("", encoding="utf-8")
    if dashboard_progress_path.exists():
        dashboard_progress_path.write_text("", encoding="utf-8")
    write_dashboard_manifest(dashboard_manifest_path, result_root)
    queued_records = [
        {
            "case": case,
            "status": "pending",
            "return_code": "",
            "runtime_s": "",
        }
        for case in CASES
    ]
    append_dashboard_progress(
        dashboard_progress_path,
        queued_records,
        result_root=result_root,
    )
    status: dict[str, object] = {
        "scene": SCENE,
        "full_sequence": True,
        "expected_frame_count": 1890,
        "parallel_processes": len(CASES),
        "threads_per_run": int(args.threads_per_run),
        "dry_run": bool(args.dry_run),
        "monitor_interval_s": float(args.monitor_interval),
        "cases": {
            case: {
                "status": "queued",
                "command": command,
                "task_log": str(task_log_path(result_root, case)),
            }
            for case, command in commands.items()
        },
    }
    write_status(manifest_path, status)

    if not args.no_dashboard and not args.dry_run:
        switch_dashboard(
            result_root,
            dashboard_log_path,
            port=int(args.dashboard_port),
        )
        print(f"Progress dashboard: http://127.0.0.1:{int(args.dashboard_port)}/")

    print(
        f"Launching {len(CASES)} concurrent full-sequence processes, "
        f"{args.threads_per_run} CPU threads per process"
    )
    failures = 0
    expected_frames = 1890
    launch_times = {case: time.monotonic() for case in CASES}
    with ThreadPoolExecutor(max_workers=len(CASES)) as executor:
        futures = {
            executor.submit(
                run_case,
                case,
                command=command,
                log_dir=log_dir,
                threads_per_run=int(args.threads_per_run),
            ): case
            for case, command in commands.items()
        }
        pending = set(futures)
        while pending:
            completed, pending = wait(
                pending,
                timeout=float(args.monitor_interval),
                return_when=FIRST_COMPLETED,
            )
            for future in completed:
                case = futures[future]
                try:
                    result = future.result()
                except Exception as error:
                    result = {
                        "case": case,
                        "return_code": 2,
                        "error": repr(error),
                    }
                result["status"] = (
                    "complete" if int(result["return_code"]) == 0 else "failed"
                )
                result["progress"] = progress_record(
                    case,
                    result_root=result_root,
                    started_monotonic=launch_times[case],
                    status=str(result["status"]),
                    expected_frames=expected_frames,
                )
                status["cases"][case] = result  # type: ignore[index]
                failures += int(result["return_code"]) != 0
                append_dashboard_progress(
                    dashboard_progress_path,
                    [
                        {
                            "case": case,
                            "status": "ok" if int(result["return_code"]) == 0 else "failed",
                            "return_code": int(result["return_code"]),
                            "runtime_s": float(result.get("runtime_s", 0.0)),
                        }
                    ],
                    result_root=result_root,
                )
                print(
                    f"{case}: rc={result['return_code']}, "
                    f"runtime={float(result.get('runtime_s', 0.0)):.1f}s",
                    flush=True,
                )

            active_records = []
            for future in pending:
                case = futures[future]
                record = progress_record(
                    case,
                    result_root=result_root,
                    started_monotonic=launch_times[case],
                    status="running",
                    expected_frames=expected_frames,
                )
                active_records.append(record)
                case_status = status["cases"][case]  # type: ignore[index]
                case_status["status"] = "running"
                case_status["progress"] = record
            if active_records:
                append_progress_csv(progress_path, active_records)
                append_dashboard_progress(
                    dashboard_progress_path,
                    active_records,
                    result_root=result_root,
                )
                summary = " | ".join(
                    (
                        f"{record['case']}: {record['frame']}/{expected_frames} "
                        f"({float(record['percent']):.1f}%), "
                        f"elapsed {format_duration(record['elapsed_s'])}, "
                        f"ETA {format_duration(record['eta_s'])}"
                    )
                    for record in sorted(active_records, key=lambda item: str(item["case"]))
                )
                print(f"[{datetime.now().strftime('%H:%M:%S')}] {summary}", flush=True)
                append_dashboard_log(dashboard_log_path, summary)
            write_status(manifest_path, status)

    if failures:
        print(f"{failures} replay process(es) failed; plotting was not started")
        return 1
    if args.dry_run or args.skip_plot:
        return 0

    plot_command = [
        str(PYTHON),
        str(PLOTTER),
        "--current-result-root",
        str(
            WORKDIR
            / "Results/circle_normal_noise_direct_uvd_u1_full_20260716"
        ),
        "--sa-v1-result-root",
        str(case_root(result_root, "sampling_aware_v1")),
        "--sa-v2-result-root",
        str(case_root(result_root, "sampling_aware_v2")),
        "--output-dir",
        str(analysis_dir),
    ]
    plot_process = subprocess.run(
        plot_command,
        cwd=str(WORKDIR),
        check=False,
    )
    status["plot"] = {
        "return_code": int(plot_process.returncode),
        "command": plot_command,
        "output_dir": str(analysis_dir),
    }
    write_status(manifest_path, status)
    return int(plot_process.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
