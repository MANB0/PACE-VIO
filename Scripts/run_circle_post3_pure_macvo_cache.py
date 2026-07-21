#!/usr/bin/env python3
"""Regenerate the circle pure-MACVO cache from the first frame after 3 s.

This driver deliberately stops after the pure visual source pass and cache
export.  It does not start any VIO, IMU factor, fixed-lag, or SA-v2 run.

The exported cache is locally indexed: cache frame 0 is source frame 90.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Scripts.build_relative_pose_factor_cache import build_cache as build_pose_sidecar  # noqa: E402
from Scripts.export_visual_factor_cache import export_result_to_visual_cache  # noqa: E402
from Scripts.run_vio_imu_prior_mode_grid import cpb_fd_only_args, flatten_nested  # noqa: E402


SCENE = "clear_circle_truth_normal_noise"
SOURCE_FRAME_START = 90
STATIC_DURATION_S = 3.0
DATASET = Path(
    "/mnt/e/文档/holoocean/code/recordings/"
    "batch_clear_truth_paths_20260713_static63_variants/clear_circle_truth_normal_noise"
)
REFERENCE_RESULT = ROOT / (
    "Results/visual_factor_cache_static63_unique_source_20260713/trial_1/"
    "pure_macvo/clear_circle_truth_normal_noise"
)
OUTPUT_ROOT = ROOT / "Results/circle_post3_pure_macvo_cache_20260717"
SOURCE_RESULT = OUTPUT_ROOT / "source" / SCENE
CONFIG_ROOT = OUTPUT_ROOT / "configs"
ODOM_CONFIG = CONFIG_ROOT / "pure_macvo_post3.yaml"
DATA_CONFIG = CONFIG_ROOT / "circle_normal_noise.yaml"
CACHE_DIR = ROOT / "VisualCache/circle_post3_pure_macvo_20260717" / SCENE
CONTRACT_PATH = OUTPUT_ROOT / "post3_cache_contract.json"


def _image_timestamps() -> list[int]:
    paths = sorted((DATASET / "left").glob("*.png"), key=lambda path: int(path.stem))
    timestamps = [int(path.stem) for path in paths]
    if len(timestamps) <= SOURCE_FRAME_START:
        raise RuntimeError("dataset does not contain source frame 90")
    if len(timestamps) != len(set(timestamps)) or any(b <= a for a, b in zip(timestamps, timestamps[1:])):
        raise RuntimeError("left-camera timestamps are not unique and strictly increasing")
    return timestamps


def _write_configs() -> None:
    source = yaml.safe_load((REFERENCE_RESULT / "config.yaml").read_text(encoding="utf-8"))
    odometry = source["Odometry"]

    # Preserve the previously validated pure-MACVO source path exactly.
    odometry["args"]["mapping"] = False
    odometry["args"]["imu_rot_prior_enable"] = False
    odometry["args"]["imu_trans_prior_enable"] = False
    odometry["args"]["imu_pose_fusion_enable"] = False
    optimizer = odometry["optimizer"]["args"]
    optimizer["imu_factor_mode"] = "legacy_pose_prior"
    optimizer["imu_rot_prior"] = False
    optimizer["post_imu_fusion_enable"] = False
    optimizer["post_imu_fusion_prepose_enable"] = False

    CONFIG_ROOT.mkdir(parents=True, exist_ok=True)
    ODOM_CONFIG.write_text(
        yaml.safe_dump(
            {"Common": {"device": "cuda"}, "Odometry": odometry, "Preprocess": None},
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )

    data = {
        "type": "GeneralStereoIMU",
        "name": "holoocean_imu",
        "args": {
            "root": str(DATASET),
            "batch_root": str(DATASET.parent),
            "scene": SCENE,
            "format": "png",
            "bl": 0.225,
            "camera": {"fx": 320, "fy": 320, "cx": 320, "cy": 240},
            "gravity": 9.8,
            "imu_T_BS": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            "imu_time_offset_ns": 0,
            "auto_estimate_time_offset": True,
            "imu_window_ns": 100000000,
            "imu_fallback_max_dt_ns": -1,
            "pose_output_frame": "NWU",
        },
    }
    DATA_CONFIG.write_text(
        yaml.safe_dump(data, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def _source_is_complete(expected_frames: int) -> bool:
    tensor_map = SOURCE_RESULT / "tensor_map.npz"
    if not tensor_map.exists() or not (SOURCE_RESULT / "config.yaml").exists():
        return False
    try:
        with np.load(tensor_map, allow_pickle=False) as data:
            return int(data["frames//time_ns"].shape[0]) == expected_frames
    except (OSError, ValueError, KeyError):
        return False


def _run_source(expected_frames: int, *, force: bool) -> None:
    if force and SOURCE_RESULT.exists():
        shutil.rmtree(SOURCE_RESULT)
    if _source_is_complete(expected_frames):
        print(f"SKIP complete pure-MACVO source: {SOURCE_RESULT}", flush=True)
        return

    SOURCE_RESULT.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(ROOT / "MACVO.py"),
        "--odom",
        str(ODOM_CONFIG),
        "--data",
        str(DATA_CONFIG),
        "--resultRoot",
        str(SOURCE_RESULT),
        "--seq_from",
        str(SOURCE_FRAME_START),
    ] + cpb_fd_only_args("pure_macvo")
    print("PURE MACVO COMMAND:", flush=True)
    print(" ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)
    flatten_nested(SOURCE_RESULT)
    if not _source_is_complete(expected_frames):
        raise RuntimeError("pure-MACVO process completed without the expected tensor-map frame count")


def _export_cache(*, force: bool) -> None:
    if force and CACHE_DIR.exists():
        shutil.rmtree(CACHE_DIR)
    manifest = CACHE_DIR / "manifest.json"
    if manifest.exists() and (CACHE_DIR / "relative_pose_factors.npz").exists():
        print(f"SKIP complete post-3s cache: {CACHE_DIR}", flush=True)
        return
    export_result_to_visual_cache(SOURCE_RESULT, CACHE_DIR, SCENE, DATASET)
    build_pose_sidecar(CACHE_DIR, force=True, huber_delta=3.0)


def _read_pose_timestamps() -> list[int]:
    with (SOURCE_RESULT / "poses.csv").open("r", encoding="utf-8", newline="") as stream:
        return [int(row["timestamp_ns"]) for row in csv.DictReader(stream)]


def _verify(all_timestamps: list[int]) -> dict[str, object]:
    expected = all_timestamps[SOURCE_FRAME_START:]
    with np.load(SOURCE_RESULT / "tensor_map.npz", allow_pickle=False) as data:
        source_timestamps = data["frames//time_ns"].astype(np.int64).tolist()
        source_poses = data["frames//pose"].astype(np.float64)
    manifest = json.loads((CACHE_DIR / "manifest.json").read_text(encoding="utf-8"))
    cache_timestamps = [int(value) for value in manifest["timestamps_ns"]]
    pose_timestamps = _read_pose_timestamps()

    checks = {
        "source_timestamp_sequence_matches_dataset_slice": source_timestamps == expected,
        "cache_timestamp_sequence_matches_dataset_slice": cache_timestamps == expected,
        "poses_timestamp_sequence_matches_dataset_slice": pose_timestamps == expected,
        "cache_frame_count_matches": int(manifest["frame_count"]) == len(expected),
        "cache_pair_count_matches": len(manifest["pairs"]) == len(expected) - 1,
        "first_local_pose_is_identity": bool(
            np.allclose(source_poses[0, :3], 0.0, atol=1e-10)
            and np.allclose(source_poses[0, 3:], [0.0, 0.0, 0.0, 1.0], atol=1e-10)
        ),
        "relative_pose_sidecar_exists": (CACHE_DIR / "relative_pose_factors.npz").exists(),
    }
    contract = {
        "scope": "pure MACVO source pass and visual cache only; no VIO run",
        "scene": SCENE,
        "dataset": str(DATASET),
        "source_frame_start": SOURCE_FRAME_START,
        "source_frame_count_total": len(all_timestamps),
        "local_cache_frame_count": len(expected),
        "local_cache_frame_0_corresponds_to_source_frame": SOURCE_FRAME_START,
        "source_timestamp_ns": expected[0],
        "source_time_s": expected[0] * 1e-9,
        "static_duration_s_requested": STATIC_DURATION_S,
        "first_cached_edge_source_indices": [SOURCE_FRAME_START, SOURCE_FRAME_START + 1],
        "first_cached_edge_local_indices": [0, 1],
        "pre_3s_visual_history_used": False,
        "vio_or_imu_fusion_run": False,
        "source_result": str(SOURCE_RESULT),
        "cache_dir": str(CACHE_DIR),
        "checks": checks,
        "pass": all(checks.values()),
        "generated_unix_time_s": time.time(),
    }
    CONTRACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONTRACT_PATH.write_text(json.dumps(contract, indent=2, ensure_ascii=False), encoding="utf-8")
    if not contract["pass"]:
        raise RuntimeError(f"post-3s cache verification failed; inspect {CONTRACT_PATH}")
    return contract


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Delete and regenerate only this experiment's outputs.")
    parser.add_argument("--verify-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    arguments = parse_args()
    timestamps = _image_timestamps()
    expected_frames = len(timestamps) - SOURCE_FRAME_START
    _write_configs()
    if not arguments.verify_only:
        _run_source(expected_frames, force=bool(arguments.force))
        _export_cache(force=bool(arguments.force))
    contract = _verify(timestamps)
    print(json.dumps(contract, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
