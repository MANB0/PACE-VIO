#!/usr/bin/env python3
"""Run stereo + raw IMU through the validated online MACVO T2 pipeline."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
BASE_ODOM = ROOT / "Config/Experiment/MACVO/MACVO_Realtime_T2.yaml"
BASE_SEQUENCE = ROOT / "Config/Sequence/realtime_stereo_imu.yaml"
DEFAULT_MODEL = ROOT / "Model/MACVO_FrontendCov.pth"


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def write_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(data, stream, sort_keys=False, allow_unicode=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def image_format(dataset: Path) -> str:
    left = dataset / "left"
    for suffix in ("png", "jpg", "jpeg"):
        if next(left.glob(f"*.{suffix}"), None) is not None:
            return suffix
    raise FileNotFoundError(f"No PNG/JPEG images found under {left}")


def validate_dataset(dataset: Path) -> dict[str, Path]:
    if not dataset.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {dataset}")
    for folder in ("left", "right"):
        if not (dataset / folder).is_dir():
            raise FileNotFoundError(f"Missing stereo folder: {dataset / folder}")
    metadata = dataset / "metadata.json"
    if not metadata.is_file():
        raise FileNotFoundError(
            f"Missing {metadata}. T2 requires metadata for camera, IMU noise, "
            "time and camera/IMU extrinsic contracts."
        )
    imu = dataset / "imu_data.csv"
    if not imu.is_file():
        imu = dataset / "imu.csv"
    if not imu.is_file():
        raise FileNotFoundError(f"Missing imu_data.csv (or imu.csv) under {dataset}")
    return {"metadata.json": metadata, imu.name: imu}


def configure_odom(
    target: Path,
    *,
    model: Path,
    trace: Path,
    parallel: bool,
    static_mode: str,
    static_state_policy: str,
    static_duration_s: float | None,
    static_min_duration_s: float,
    static_max_duration_s: float,
    static_window_s: float,
    static_stable_hold_s: float,
    cpu_threads: int,
    vio_backend: str,
) -> None:
    cfg = load_yaml(BASE_ODOM)
    args = cfg["Odometry"]["args"]
    frontend = cfg["Odometry"]["frontend"]["args"]
    optimizer = cfg["Odometry"]["optimizer"]["args"]
    args["pipeline_trace_path"] = str(trace)
    args.pop("imu_static_initialization_enable", None)
    args["imu_static_initialization_mode"] = str(static_mode)
    args["imu_static_initialization_state_policy"] = str(static_state_policy)
    if static_duration_s is None:
        args.pop("imu_static_initialization_duration_s", None)
    else:
        args["imu_static_initialization_duration_s"] = float(static_duration_s)
    args["imu_static_adaptive_min_duration_s"] = float(static_min_duration_s)
    args["imu_static_adaptive_max_duration_s"] = float(static_max_duration_s)
    args["imu_static_adaptive_window_s"] = float(static_window_s)
    args["imu_static_adaptive_stable_hold_s"] = float(static_stable_hold_s)
    frontend["weight"] = str(model)
    optimizer["parallel"] = bool(parallel)
    optimizer["two_state_cpu_threads"] = int(cpu_threads)
    optimizer["two_state_backend"] = str(vio_backend)
    write_yaml(target, cfg)


def configure_sequence(target: Path, dataset: Path, fmt: str) -> None:
    cfg = load_yaml(BASE_SEQUENCE)
    args = cfg["args"]
    args.update(
        {
            "root": str(dataset),
            "batch_root": str(dataset.parent),
            "scene": dataset.name,
            "format": fmt,
        }
    )
    write_yaml(target, cfg)


def append_progress(path: Path, row: dict[str, object]) -> None:
    fields = ["dataset", "mode", "status", "return_code", "runtime_s", "result_root"]
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in fields})


def validate_static_initialization_options(args: argparse.Namespace) -> None:
    if args.static_init_state_policy == "zero" and args.static_init_mode == "off":
        raise ValueError(
            "--static-init-state-policy zero requires fixed or adaptive static initialization"
        )
    if args.static_init_mode == "fixed":
        if args.static_init_duration_s is None:
            raise ValueError(
                "--static-init-mode fixed requires --static-init-duration-s"
            )
        if args.static_init_duration_s <= 0.0:
            raise ValueError("--static-init-duration-s must be > 0 in fixed mode")
    elif args.static_init_duration_s is not None:
        raise ValueError(
            "--static-init-duration-s is valid only with --static-init-mode fixed"
        )
    if args.static_init_min_duration_s <= 0.0:
        raise ValueError("--static-init-min-duration-s must be > 0")
    if args.static_init_max_duration_s < args.static_init_min_duration_s:
        raise ValueError(
            "--static-init-max-duration-s must be >= --static-init-min-duration-s"
        )
    if args.static_init_window_s <= 0.0:
        raise ValueError("--static-init-window-s must be > 0")
    if args.static_init_stable_hold_s < 2.0 * args.static_init_window_s:
        raise ValueError(
            "--static-init-stable-hold-s must be at least 2 * --static-init-window-s"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "Results/realtime_t2")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--mode", choices=("pipeline", "serial"), default="pipeline")
    parser.add_argument("--seq-from", type=int, default=0)
    parser.add_argument("--seq-to", type=int, default=-1, help="Exclusive stop; -1 runs all frames")
    parser.add_argument(
        "--static-init-mode",
        choices=("fixed", "adaptive", "off"),
        default="adaptive",
        help=(
            "adaptive detects a sufficient stationary prefix; fixed requires "
            "--static-init-duration-s; off assumes no stationary prefix"
        ),
    )
    parser.add_argument(
        "--static-init-duration-s",
        type=float,
        default=None,
        help="Required positive duration for --static-init-mode fixed only.",
    )
    parser.add_argument(
        "--static-init-state-policy",
        choices=("estimated", "zero"),
        default="estimated",
        help=(
            "estimated applies the detected attitude and biases; zero keeps the same "
            "static boundary but starts VIO from identity attitude and zero biases"
        ),
    )
    parser.add_argument("--static-init-min-duration-s", type=float, default=1.0)
    parser.add_argument("--static-init-max-duration-s", type=float, default=8.0)
    parser.add_argument("--static-init-window-s", type=float, default=0.25)
    parser.add_argument("--static-init-stable-hold-s", type=float, default=0.75)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument(
        "--vio-backend",
        choices=("two_state", "isam2"),
        default="two_state",
        help=(
            "two_state preserves the validated online T2 solver; isam2 consumes "
            "the same compressed-UVD/IMU/bias factor packets incrementally"
        ),
    )
    parser.add_argument("--timeout", type=int, default=21600)
    parser.add_argument(
        "--live-display",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--dashboard-host", default="127.0.0.1")
    parser.add_argument("--dashboard-port", type=int, default=8765)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and write runtime configs without launching MACVO.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset = args.dataset.expanduser().resolve()
    output = args.output.expanduser().resolve()
    model = args.model.expanduser().resolve()
    inputs = validate_dataset(dataset)
    fmt = image_format(dataset)

    validate_static_initialization_options(args)
    if args.cpu_threads < 1:
        raise ValueError("--cpu-threads must be >= 1")
    if not args.dry_run and not model.is_file():
        raise FileNotFoundError(
            f"Missing frontend model: {model}\n"
            "Run: python Scripts/download_models.py"
        )
    if args.vio_backend == "isam2":
        from Utility.T2ISAM2Backend import IncrementalT2ISAM2Backend

        IncrementalT2ISAM2Backend(
            initial_prior_std={
                "pose_translation_std": 1.0e-5,
                "pose_rotation_std": 1.0e-5,
                "velocity_std": 0.05,
                "acc_bias_std": 0.2,
                "gyro_bias_std": 0.02,
            }
        )

    result_root = output / dataset.name
    config_root = result_root / "runtime_configs"
    result_root.mkdir(parents=True, exist_ok=True)
    odom_path = config_root / "odometry.yaml"
    sequence_path = config_root / "sequence.yaml"
    trace_path = result_root / "pipeline_trace.csv"
    configure_odom(
        odom_path,
        model=model,
        trace=trace_path,
        parallel=args.mode == "pipeline",
        static_mode=args.static_init_mode,
        static_state_policy=args.static_init_state_policy,
        static_duration_s=args.static_init_duration_s,
        static_min_duration_s=args.static_init_min_duration_s,
        static_max_duration_s=args.static_init_max_duration_s,
        static_window_s=args.static_init_window_s,
        static_stable_hold_s=args.static_init_stable_hold_s,
        cpu_threads=args.cpu_threads,
        vio_backend=args.vio_backend,
    )
    configure_sequence(sequence_path, dataset, fmt)

    optional_ref = dataset / "ref_pose.csv"
    contract = {
        "project_root": str(ROOT),
        "dataset": str(dataset),
        "mode": args.mode,
        "frontend": "live MACVO stereo frontend (no visual cache)",
        "backend": f"{args.vio_backend} + compressed_uvd T2 factor packets",
        "preintegration": "standard_local_frame_preintegration",
        "trajectory_reference": "IMU center for VIO output",
        "static_initialization": {
            "mode": args.static_init_mode,
            "state_policy": args.static_init_state_policy,
            "fixed_duration_s": args.static_init_duration_s,
            "adaptive_min_duration_s": args.static_init_min_duration_s,
            "adaptive_max_duration_s": args.static_init_max_duration_s,
            "adaptive_window_s": args.static_init_window_s,
            "adaptive_stable_hold_s": args.static_init_stable_hold_s,
        },
        "ground_truth_available": optional_ref.is_file(),
        "input_sha256": {name: sha256(path) for name, path in inputs.items()},
    }
    if optional_ref.is_file():
        contract["input_sha256"]["ref_pose.csv"] = sha256(optional_ref)
    (result_root / "runtime_contract.json").write_text(
        json.dumps(contract, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    command = [
        sys.executable,
        str(ROOT / "MACVO.py"),
        "--odom",
        str(odom_path),
        "--data",
        str(sequence_path),
        "--resultRoot",
        str(result_root),
        "--timing",
        "--noeval",
        "--seq_from",
        str(args.seq_from),
    ]
    if args.seq_to >= 0:
        command += ["--seq_to", str(args.seq_to)]
    if args.live_display:
        command += [
            "--live-display",
            "--dashboard-host",
            args.dashboard_host,
            "--dashboard-port",
            str(args.dashboard_port),
        ]

    print(f"Dataset: {dataset}")
    print(f"Output:  {result_root}")
    print(f"Mode:    {args.mode}")
    print(f"Backend: {args.vio_backend}")
    print(f"Command: {shlex.join(command)}", flush=True)
    if args.live_display:
        print(f"Dashboard: http://{args.dashboard_host}:{args.dashboard_port}/", flush=True)
    if args.dry_run:
        print("Dry run complete: inputs and generated configs are valid.")
        return 0

    append_progress(
        output / "progress.csv",
        {"dataset": dataset.name, "mode": args.mode, "status": "running", "result_root": result_root},
    )
    started = time.monotonic()
    process = subprocess.Popen(command, cwd=ROOT, start_new_session=True)
    try:
        return_code = process.wait(timeout=args.timeout)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        raise
    elapsed = time.monotonic() - started
    status = "ok" if return_code == 0 else "failed"
    append_progress(
        output / "progress.csv",
        {
            "dataset": dataset.name,
            "mode": args.mode,
            "status": status,
            "return_code": return_code,
            "runtime_s": f"{elapsed:.3f}",
            "result_root": result_root,
        },
    )
    print(f"Finished: {status}, runtime={elapsed:.1f}s", flush=True)
    return int(return_code)


if __name__ == "__main__":
    raise SystemExit(main())
