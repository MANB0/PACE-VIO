#!/usr/bin/env python3
"""Run stereo + raw IMU through the validated online PACE-VIO pipeline."""

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
from Utility.NearZeroVelocityDetector import FROZEN_NEAR_ZERO_VELOCITY_V2

BASE_ODOM = ROOT / "Config/Experiment/MACVO/PACE_VIO_Realtime.yaml"
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
            f"Missing {metadata}. PACE-VIO requires metadata for camera, IMU noise, "
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
    visual_factor: str,
    near_zero_velocity_detector: str,
    near_zero_velocity_prior_std_m_s: float,
    visual_cache_mode: str = "live",
    visual_cache_path: Path | None = None,
) -> None:
    cfg = load_yaml(BASE_ODOM)
    args = cfg["Odometry"]["args"]
    frontend = cfg["Odometry"]["frontend"]["args"]
    optimizer = cfg["Odometry"]["optimizer"]["args"]
    args["pipeline_trace_path"] = str(trace)
    args["visual_cache_mode"] = "replay" if visual_cache_mode == "replay" else "off"
    args["visual_cache_path"] = (
        str(visual_cache_path) if visual_cache_mode == "replay" else None
    )
    args.pop("imu_static_initialization_enable", None)
    if visual_cache_mode == "record":
        # Cache recording is a pure visual MACVO pass. The visual TwoFrame_PGO
        # is retained only to chain a camera trajectory and express cached
        # local observations consistently; no IMU or VIO backend is active.
        cfg["Odometry"]["name"] = "MACVO-Visual-Cache-Recorder"
        args.update(
            {
                "mapping": False,
                "imu_static_initialization_mode": "off",
                "imu_static_initialization_state_policy": "estimated",
                "imu_rot_prior_enable": False,
                "imu_trans_prior_enable": False,
                "imu_pose_fusion_enable": False,
                "imu_vio_velocity_feedback_enable": False,
                "imu_vio_bias_feedback_enable": False,
            }
        )
        cfg["Odometry"]["optimizer"] = {
            "type": "TwoFrame_PGO",
            "args": {
                "device": "cpu",
                "parallel": bool(parallel),
                "autodiff": False,
                "graph_type": "disp",
                "vectorize": True,
                "imu_rot_prior": False,
            },
        }
        frontend["weight"] = str(model)
        write_yaml(target, cfg)
        return
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
    optimizer["two_state_visual_factor_mode"] = {
        "pose": "relative_pose",
        "uvd": "direct_uvd",
        "pace": "compressed_uvd",
    }[str(visual_factor)]
    if visual_cache_mode == "replay":
        cfg["Odometry"]["frontend"] = {"type": "ReplayFrontend", "args": {}}
    optimizer["two_state_near_zero_velocity_enable"] = (
        near_zero_velocity_detector != "off"
    )
    if near_zero_velocity_detector != "off":
        optimizer["two_state_near_zero_velocity_detector_version"] = str(
            near_zero_velocity_detector
        )
        optimizer["two_state_near_zero_velocity_prior_std_m_s"] = float(
            near_zero_velocity_prior_std_m_s
        )
    if near_zero_velocity_detector == "v2":
        optimizer.update(
            {
                "two_state_near_zero_velocity_v2_minimum_imu_angular_rate_rad_s": FROZEN_NEAR_ZERO_VELOCITY_V2.minimum_imu_angular_rate_rad_s,
                "two_state_near_zero_velocity_v2_minimum_visual_angular_rate_rad_s": FROZEN_NEAR_ZERO_VELOCITY_V2.minimum_visual_angular_rate_rad_s,
                "two_state_near_zero_velocity_v2_maximum_rotation_vector_rate_difference_rad_s": FROZEN_NEAR_ZERO_VELOCITY_V2.maximum_rotation_vector_rate_difference_rad_s,
                "two_state_near_zero_velocity_v2_minimum_rotation_axis_cosine": FROZEN_NEAR_ZERO_VELOCITY_V2.minimum_rotation_axis_cosine,
                "two_state_near_zero_velocity_v2_maximum_zero_translation_nis_per_dof": FROZEN_NEAR_ZERO_VELOCITY_V2.maximum_zero_translation_nis_per_dof,
                "two_state_near_zero_velocity_enter_hold_s": FROZEN_NEAR_ZERO_VELOCITY_V2.enter_hold_s,
                "two_state_near_zero_velocity_release_hold_s": FROZEN_NEAR_ZERO_VELOCITY_V2.release_hold_s,
            }
        )
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


def validate_visual_cache_options(args: argparse.Namespace) -> Path | None:
    mode = str(args.visual_cache_mode)
    cache = (
        None
        if args.visual_cache_path is None
        else args.visual_cache_path.expanduser().resolve()
    )
    if mode == "live":
        if cache is not None:
            raise ValueError("--visual-cache-path is valid only in record or replay mode")
        return None
    if cache is None:
        raise ValueError(f"--visual-cache-mode {mode} requires --visual-cache-path")
    if args.seq_from != 0 or args.seq_to != -1:
        raise ValueError("visual cache record/replay requires the complete sequence")
    if mode == "record":
        if cache.exists():
            raise FileExistsError(
                f"visual cache target already exists: {cache}; choose a new path"
            )
        return cache
    if mode == "replay":
        from Scripts.record_visual_factor_cache import validate_visual_cache_bundle

        return validate_visual_cache_bundle(cache, args.visual_factor)
    raise ValueError(f"unsupported visual cache mode: {mode!r}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "Results/pace_vio")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--mode", choices=("pipeline", "serial"), default="pipeline")
    parser.add_argument("--seq-from", type=int, default=0)
    parser.add_argument("--seq-to", type=int, default=-1, help="Exclusive stop; -1 runs all frames")
    parser.add_argument(
        "--visual-cache-mode",
        choices=("live", "record", "replay"),
        default="live",
        help=(
            "live runs MACVO normally; record runs MACVO once and writes a reusable "
            "visual cache; replay skips the neural frontend and loads that cache"
        ),
    )
    parser.add_argument(
        "--visual-cache-path",
        type=Path,
        help="Cache output for record mode or existing cache directory for replay mode.",
    )
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
            "static boundary and camera/IMU frame alignment, but discards the detected "
            "roll/pitch and starts with zero biases"
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
            "Select the two-state or incremental iSAM2 backend. Both accept Pose, "
            "point-level UVD and PACE visual factors."
        ),
    )
    parser.add_argument(
        "--visual-factor",
        choices=("pose", "uvd", "pace"),
        default="pace",
        help=(
            "Visual representation: robust relative Pose, native point-level UVD, "
            "or the compressed PACE factor (default)."
        ),
    )
    parser.add_argument(
        "--near-zero-velocity-detector",
        choices=("off", "v1", "v2"),
        default="off",
        help=(
            "Opt-in causal turn-stop velocity factor. The production default "
            "remains off; v2 uses IMU/visual rotation agreement and local "
            "zero-translation NIS."
        ),
    )
    parser.add_argument(
        "--near-zero-velocity-prior-std-m-s",
        type=float,
        default=0.01,
        help="Per-axis velocity prior standard deviation when the detector is active.",
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
        "--paper-evaluation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "After a successful run, export full-sequence APE/RPE, timing, "
            "solver and detector outputs when ref_pose.csv is available."
        ),
    )
    parser.add_argument(
        "--evaluation-alignment-json",
        type=Path,
        help=(
            "Optional explicit fixed time/SE(3) evaluation contract. Scale must "
            "remain one; omitted for synchronized HoloOcean sequences."
        ),
    )
    parser.add_argument(
        "--motion-reference-csv",
        type=Path,
        help="Optional per-edge reference labels for detector confusion statistics.",
    )
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
    visual_cache_path = validate_visual_cache_options(args)
    if args.cpu_threads < 1:
        raise ValueError("--cpu-threads must be >= 1")
    if args.near_zero_velocity_prior_std_m_s <= 0.0:
        raise ValueError("--near-zero-velocity-prior-std-m-s must be > 0")
    if args.visual_cache_mode == "record" and args.near_zero_velocity_detector != "off":
        raise ValueError("record mode does not run the conditional velocity detector")
    if (
        args.visual_cache_mode != "record"
        and args.near_zero_velocity_detector != "off"
        and args.vio_backend != "isam2"
    ):
        raise ValueError(
            "--near-zero-velocity-detector currently requires --vio-backend isam2"
        )
    if (
        not args.dry_run
        and args.visual_cache_mode in {"live", "record"}
        and not model.is_file()
    ):
        raise FileNotFoundError(
            f"Missing frontend model: {model}\n"
            "Run: python Scripts/download_models.py"
        )
    if args.visual_cache_mode != "record" and args.vio_backend == "isam2":
        from Utility.PACEISAM2Backend import IncrementalPACEISAM2Backend

        IncrementalPACEISAM2Backend(
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
        visual_factor=args.visual_factor,
        near_zero_velocity_detector=args.near_zero_velocity_detector,
        near_zero_velocity_prior_std_m_s=(
            args.near_zero_velocity_prior_std_m_s
        ),
        visual_cache_mode=args.visual_cache_mode,
        visual_cache_path=visual_cache_path,
    )
    configure_sequence(sequence_path, dataset, fmt)

    optional_ref = dataset / "ref_pose.csv"
    contract = {
        "project_root": str(ROOT),
        "dataset": str(dataset),
        "mode": args.mode,
        "frontend": {
            "live": "live MACVO stereo frontend",
            "record": "live MACVO stereo frontend with post-run cache recording",
            "replay": "validated MACVO visual cache replay",
        }[args.visual_cache_mode],
        "visual_cache": {
            "mode": args.visual_cache_mode,
            "path": None if visual_cache_path is None else str(visual_cache_path),
            "complete_sequence_required": args.visual_cache_mode != "live",
        },
        "project": "PACE-VIO",
        "backend": (
            "none (pure visual cache recording)"
            if args.visual_cache_mode == "record"
            else args.vio_backend
        ),
        "visual_factor": (
            "raw MACVO observations"
            if args.visual_cache_mode == "record"
            else args.visual_factor
        ),
        "near_zero_velocity": {
            "detector": args.near_zero_velocity_detector,
            "prior_std_m_s": args.near_zero_velocity_prior_std_m_s,
            "production_default": "off",
            "thresholds": (
                None
                if args.near_zero_velocity_detector != "v2"
                else {
                    "minimum_imu_angular_rate_rad_s": FROZEN_NEAR_ZERO_VELOCITY_V2.minimum_imu_angular_rate_rad_s,
                    "minimum_visual_angular_rate_rad_s": FROZEN_NEAR_ZERO_VELOCITY_V2.minimum_visual_angular_rate_rad_s,
                    "maximum_rotation_vector_rate_difference_rad_s": FROZEN_NEAR_ZERO_VELOCITY_V2.maximum_rotation_vector_rate_difference_rad_s,
                    "minimum_rotation_axis_cosine": FROZEN_NEAR_ZERO_VELOCITY_V2.minimum_rotation_axis_cosine,
                    "maximum_zero_translation_nis_per_dof": FROZEN_NEAR_ZERO_VELOCITY_V2.maximum_zero_translation_nis_per_dof,
                    "enter_hold_s": FROZEN_NEAR_ZERO_VELOCITY_V2.enter_hold_s,
                    "release_hold_s": FROZEN_NEAR_ZERO_VELOCITY_V2.release_hold_s,
                }
            ),
        },
        "preintegration": "standard_local_frame_preintegration",
        "trajectory_reference": "IMU center for VIO output",
        "frame_contract": {
            "world_internal": "NED",
            "world_export": "NWU",
            "imu_measurement_and_preintegration_frame": "raw FLU",
            "camera_internal_frame": "FRD/NED",
        },
        "extrinsic_contract": {
            "field": "metadata.extrinsics.T_CI",
            "equation": "p_C = T_CI p_I",
            "state_composition": "T_WI = T_WC * T_CI",
            "raw_imu_samples_transformed": False,
        },
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
        "paper_evaluation": {
            "enabled": bool(args.paper_evaluation and args.visual_cache_mode != "record"),
            "sequence_scope": "complete active sequence",
            "alignment_json": (
                None
                if args.evaluation_alignment_json is None
                else str(args.evaluation_alignment_json.expanduser().resolve())
            ),
            "motion_reference_csv": (
                None
                if args.motion_reference_csv is None
                else str(args.motion_reference_csv.expanduser().resolve())
            ),
        },
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
    if args.visual_cache_mode == "replay":
        command += [
            "--visual-cache-mode",
            "replay",
            "--visual-cache-path",
            str(visual_cache_path),
        ]
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
    if args.visual_cache_mode == "record":
        print("Backend: none (pure visual MACVO)")
        print("Factor:  raw MACVO observations")
    else:
        print(f"Backend: {args.vio_backend}")
        print(f"Factor:  {args.visual_factor}")
    print(f"Visual:  {args.visual_cache_mode}")
    if visual_cache_path is not None:
        print(f"Cache:   {visual_cache_path}")
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
    cache_record_runtime_s = None
    if return_code == 0 and args.visual_cache_mode == "record":
        from Scripts.record_visual_factor_cache import record_visual_factor_cache

        cache_started = time.monotonic()
        try:
            cache = record_visual_factor_cache(
                result_root,
                visual_cache_path,
                dataset.name,
                dataset,
            )
            print(f"Visual cache ready: {cache}", flush=True)
        except Exception as error:
            status = "cache_record_failed"
            return_code = 3
            print(f"Visual cache recording failed: {error}", file=sys.stderr, flush=True)
        cache_record_runtime_s = time.monotonic() - cache_started
    if (
        return_code == 0
        and args.paper_evaluation
        and args.visual_cache_mode != "record"
    ):
        if optional_ref.is_file():
            from Utility.PaperEvaluation import export_paper_evaluation

            try:
                evaluation_dir = export_paper_evaluation(
                    project_root=ROOT,
                    dataset_root=dataset,
                    result_root=result_root,
                    alignment_path=args.evaluation_alignment_json,
                    motion_reference_path=args.motion_reference_csv,
                )
                print(f"Paper evaluation: {evaluation_dir}", flush=True)
            except Exception as error:
                status = "evaluation_failed"
                return_code = 2
                print(f"Paper evaluation failed: {error}", file=sys.stderr, flush=True)
        else:
            print(
                "Paper evaluation skipped: dataset has no ref_pose.csv.",
                flush=True,
            )
    (result_root / "run_execution.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "process_wall_runtime_s": elapsed,
                "cache_record_runtime_s": cache_record_runtime_s,
                "total_wall_runtime_s": time.monotonic() - started,
                "return_code": int(return_code),
                "status": status,
                "mode": args.mode,
                "visual_cache_mode": args.visual_cache_mode,
                "visual_cache_path": (
                    None if visual_cache_path is None else str(visual_cache_path)
                ),
                "seq_from": args.seq_from,
                "seq_to": args.seq_to,
                "startup_and_initialization_included": True,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
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
