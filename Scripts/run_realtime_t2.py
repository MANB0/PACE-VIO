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
    static_duration_s: float,
    cpu_threads: int,
) -> None:
    cfg = load_yaml(BASE_ODOM)
    args = cfg["Odometry"]["args"]
    frontend = cfg["Odometry"]["frontend"]["args"]
    optimizer = cfg["Odometry"]["optimizer"]["args"]
    args["pipeline_trace_path"] = str(trace)
    args["imu_static_initialization_enable"] = static_duration_s > 0.0
    args["imu_static_initialization_duration_s"] = float(static_duration_s)
    frontend["weight"] = str(model)
    optimizer["parallel"] = bool(parallel)
    optimizer["two_state_cpu_threads"] = int(cpu_threads)
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "Results/realtime_t2")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--mode", choices=("pipeline", "serial"), default="pipeline")
    parser.add_argument("--seq-from", type=int, default=0)
    parser.add_argument("--seq-to", type=int, default=-1, help="Exclusive stop; -1 runs all frames")
    parser.add_argument(
        "--static-init-duration-s",
        type=float,
        default=3.0,
        help="Verified default is 3 s. Use 0 only when the dataset has no static prefix.",
    )
    parser.add_argument("--cpu-threads", type=int, default=4)
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

    if args.static_init_duration_s < 0.0:
        raise ValueError("--static-init-duration-s must be >= 0")
    if args.cpu_threads < 1:
        raise ValueError("--cpu-threads must be >= 1")
    if not args.dry_run and not model.is_file():
        raise FileNotFoundError(
            f"Missing frontend model: {model}\n"
            "Run: python Scripts/download_models.py"
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
        static_duration_s=args.static_init_duration_s,
        cpu_threads=args.cpu_threads,
    )
    configure_sequence(sequence_path, dataset, fmt)

    optional_ref = dataset / "ref_pose.csv"
    contract = {
        "project_root": str(ROOT),
        "dataset": str(dataset),
        "mode": args.mode,
        "frontend": "live MACVO stereo frontend (no visual cache)",
        "backend": "two_state_fixed_lag + compressed_uvd",
        "preintegration": "standard_local_frame_preintegration",
        "trajectory_reference": "IMU center for VIO output",
        "static_initialization_duration_s": args.static_init_duration_s,
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
