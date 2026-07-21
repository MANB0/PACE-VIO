#!/usr/bin/env python3
"""Replay real HoloOcean image/IMU files through MACVO + online T2.

This runner deliberately does not create or read a visual cache.  It writes a
temporary odometry configuration that lets MACVO build the UVD factor from the
current frontend pair and lets the existing N=2 T2 backend run either
sequentially or in the MACVO optimizer worker process.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import yaml


ROOT = Path("/home/admin1/macvo-dev")
BASE_ODOM = ROOT / "Config/Experiment/MACVO/MACVO_HoloOcean_IMU.yaml"
SEQ_TEMPLATE = ROOT / "Config/Sequence/holoocean_imu.yaml"
DATA_ROOT = Path(
    "/mnt/e/文档/holoocean/code/recordings/"
    "batch_clear_truth_paths_20260713_static63_variants"
)
SCENES = {
    "circle": "clear_circle_truth_normal_noise",
    "rectangle": "clear_stop_turn_rectangle_truth_normal_noise",
    "straight": "clear_straight_truth_normal_noise",
}
DEFAULT_OUTPUT = ROOT / "Results/real_t2_pipeline_validation_20260720"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def write_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(data, stream, sort_keys=False, allow_unicode=True)


def configure_odom(path: Path, *, parallel: bool, trace_path: Path) -> None:
    cfg = load_yaml(BASE_ODOM)
    odom = cfg["Odometry"]
    args = odom["args"]
    optimizer = odom["optimizer"]["args"]

    args.update(
        {
            "device": "cuda",
            "mapping": False,
            "profile": False,
            "visual_cache_mode": "off",
            "visual_cache_path": None,
            "pipeline_trace_path": str(trace_path),
            "imu_vio_gravity_handling": "standard_local_frame_preintegration",
            "imu_vio_gravity_pose_source": "imu_integrated_estinit",
            "imu_static_initialization_enable": True,
            "imu_static_initialization_duration_s": 3.0,
            "imu_static_sigma_multiplier": 5.0,
            "imu_static_gyro_mean_norm_max": 0.03,
            "imu_static_acc_norm_error_max": 0.6,
            "imu_trans_prior_enable": True,
            "imu_trans_prior_mode": "imu_velocity_composed",
        }
    )
    optimizer.update(
        {
            "device": "cpu",
            "parallel": bool(parallel),
            "autodiff": True,
            "imu_factor_mode": "two_state_fixed_lag",
            "two_state_visual_factor_mode": "compressed_uvd",
            "two_state_warm_start": "macvo_pose",
            "two_state_max_iterations": 20,
            "two_state_cpu_threads": 4,
            "two_state_visual_huber_delta": 3.0,
            "two_state_uvd_huber_delta": 0.1,
            "two_state_initial_pose_translation_std": 1.0e-5,
            "two_state_initial_pose_rotation_std": 1.0e-5,
            "two_state_initial_velocity_std": 0.05,
            "two_state_initial_acc_bias_std": 0.2,
            "two_state_initial_gyro_bias_std": 0.02,
            "two_state_cross_edge_rank_aware_imu_whitening": False,
        }
    )
    write_yaml(path, cfg)


def configure_sequence(path: Path, scene_root: Path, scene_name: str) -> None:
    cfg = load_yaml(SEQ_TEMPLATE)
    args = cfg["args"]
    args["root"] = str(scene_root)
    args["batch_root"] = str(scene_root.parent)
    args["scene"] = scene_name
    write_yaml(path, cfg)


def append_progress(path: Path, row: dict[str, object]) -> None:
    fields = ["scene", "mode", "status", "return_code", "runtime_s", "result_root"]
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in fields})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenes", nargs="+", choices=sorted(SCENES), default=["rectangle"])
    parser.add_argument("--mode", choices=("pipeline", "serial"), default="pipeline")
    parser.add_argument(
        "--seq-to",
        type=int,
        default=300,
        help="Frame stop (exclusive). 300 covers the first 209 post-static edges; use -1 for full sequence.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--timeout", type=int, default=21600)
    parser.add_argument(
        "--live-display",
        action="store_true",
        help="Pass the read-only local browser dashboard to MACVO.py.",
    )
    parser.add_argument("--dashboard-host", default="127.0.0.1")
    parser.add_argument("--dashboard-port", type=int, default=8765)
    return parser.parse_args()


def run_scene(args: argparse.Namespace, scene_key: str) -> int:
    scene_name = SCENES[scene_key]
    scene_root = DATA_ROOT / scene_name
    if not scene_root.is_dir():
        raise FileNotFoundError(scene_root)
    for filename in ("metadata.json", "imu_data.csv", "ref_pose.csv"):
        if not (scene_root / filename).is_file():
            raise FileNotFoundError(scene_root / filename)

    mode = args.mode
    result_root = args.output / mode / scene_name
    result_root.mkdir(parents=True, exist_ok=True)
    trace_path = result_root / "pipeline_trace.csv"
    odom_path = args.output / "configs" / f"odometry_{mode}.yaml"
    seq_path = args.output / "configs" / f"sequence_{scene_key}.yaml"
    configure_odom(odom_path, parallel=(mode == "pipeline"), trace_path=trace_path)
    configure_sequence(seq_path, scene_root, scene_name)

    manifest = {
        "scene_key": scene_key,
        "scene": scene_name,
        "scene_root": str(scene_root),
        "mode": mode,
        "uses_visual_cache": False,
        "visual_factor_mode": "compressed_uvd_online_from_current_macvo_pair",
        "optimizer_mode": "two_state_fixed_lag",
        "parallel_optimizer": mode == "pipeline",
        "static_initialization_duration_s": 3.0,
        "expected_first_active_edge": "90->91",
        "seq_to": args.seq_to,
        "input_sha256": {
            name: sha256(scene_root / name)
            for name in ("metadata.json", "imu_data.csv", "ref_pose.csv")
        },
    }
    (result_root / "pipeline_contract.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    cmd = [
        sys.executable,
        str(ROOT / "MACVO.py"),
        "--odom", str(odom_path),
        "--data", str(seq_path),
        "--resultRoot", str(result_root),
        "--timing",
        "--noeval",
    ]
    if args.seq_to >= 0:
        cmd += ["--seq_to", str(args.seq_to)]
    if args.live_display:
        cmd += [
            "--live-display",
            "--dashboard-host", str(args.dashboard_host),
            "--dashboard-port", str(args.dashboard_port),
        ]

    print("=" * 78, flush=True)
    print(f"REAL DATA {scene_key}: mode={mode}", flush=True)
    print(f"  input:  {scene_root}", flush=True)
    print(f"  output: {result_root}", flush=True)
    print(f"  cmd:    {' '.join(cmd)}", flush=True)
    started = time.monotonic()
    append_progress(args.output / "progress.csv", {
        "scene": scene_name, "mode": mode, "status": "running", "result_root": result_root,
    })
    completed = subprocess.run(cmd, cwd=str(ROOT), timeout=args.timeout)
    elapsed = time.monotonic() - started
    status = "ok" if completed.returncode == 0 else "failed"
    append_progress(args.output / "progress.csv", {
        "scene": scene_name, "mode": mode, "status": status,
        "return_code": completed.returncode, "runtime_s": f"{elapsed:.3f}",
        "result_root": result_root,
    })
    print(f"RESULT {scene_key}/{mode}: {status}, runtime={elapsed:.1f}s", flush=True)
    return int(completed.returncode)


def main() -> int:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    failures = 0
    for scene in args.scenes:
        failures += int(run_scene(args, scene) != 0)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
