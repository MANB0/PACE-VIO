#!/usr/bin/env python3
"""Run a same-build short pure-MACVO continuous/cold-start A/B."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
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
DATASET = Path(
    "/mnt/e/文档/holoocean/code/recordings/"
    "batch_clear_truth_paths_20260713_static63_variants/clear_circle_truth_normal_noise"
)
REFERENCE_CONFIG = ROOT / (
    "Results/circle_post3_pure_macvo_cache_20260717/source/"
    "clear_circle_truth_normal_noise/config.yaml"
)
OUTPUT = ROOT / "analysis_startup_visual_upward_drift_20260718/same_build_cold_start_ab"
EDGE_COUNT = 30


def timestamps() -> np.ndarray:
    values = sorted((DATASET / "left").glob("*.png"), key=lambda path: int(path.stem))
    result = np.asarray([int(path.stem) for path in values], dtype=np.int64)
    if result.size < 2 or np.any(np.diff(result) <= 0):
        raise RuntimeError("invalid camera timestamp sequence")
    return result


def active_start(values: np.ndarray) -> int:
    end = int(values[0] + 3_000_000_000)
    return int(np.searchsorted(values, end, side="left"))


def write_configs(output: Path) -> tuple[Path, Path]:
    source = yaml.safe_load(REFERENCE_CONFIG.read_text(encoding="utf-8"))
    odometry = source["Odometry"]
    odometry["args"]["mapping"] = False
    odometry["args"]["imu_rot_prior_enable"] = False
    odometry["args"]["imu_trans_prior_enable"] = False
    odometry["args"]["imu_pose_fusion_enable"] = False
    optimizer = odometry["optimizer"]["args"]
    optimizer["imu_factor_mode"] = "legacy_pose_prior"
    optimizer["imu_rot_prior"] = False
    optimizer["post_imu_fusion_enable"] = False
    optimizer["post_imu_fusion_prepose_enable"] = False
    odom_path = output / "pure_macvo_same_build.yaml"
    odom_path.write_text(
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
            "root": str(DATASET), "batch_root": str(DATASET.parent), "scene": SCENE,
            "format": "png", "bl": 0.225,
            "camera": {"fx": 320, "fy": 320, "cx": 320, "cy": 240},
            "gravity": 9.8, "imu_T_BS": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            "imu_time_offset_ns": 0, "auto_estimate_time_offset": True,
            "imu_window_ns": 100000000, "imu_fallback_max_dt_ns": -1,
            "pose_output_frame": "NWU",
        },
    }
    data_path = output / "circle_normal_noise.yaml"
    data_path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return odom_path, data_path


def result_complete(result: Path, expected: int) -> bool:
    path = result / "tensor_map.npz"
    if not path.exists():
        return False
    try:
        with np.load(path, allow_pickle=False) as data:
            return int(data["frames//time_ns"].shape[0]) == expected
    except (OSError, KeyError, ValueError):
        return False


def run_case(
    label: str,
    start: int,
    stop: int,
    odom: Path,
    data: Path,
    output: Path,
    *,
    force: bool,
) -> tuple[Path, Path]:
    result = output / "results" / label / SCENE
    cache = output / "caches" / label / SCENE
    expected = stop - start
    if force and result.exists():
        shutil.rmtree(result)
    if not result_complete(result, expected):
        result.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable, str(ROOT / "MACVO.py"), "--odom", str(odom), "--data", str(data),
            "--resultRoot", str(result), "--seq_from", str(start), "--seq_to", str(stop),
        ] + cpb_fd_only_args("pure_macvo")
        print(f"[{label}] {' '.join(command)}", flush=True)
        subprocess.run(command, cwd=ROOT, check=True)
        flatten_nested(result)
    if force and cache.exists():
        shutil.rmtree(cache)
    if not (cache / "relative_pose_factors.npz").exists():
        export_result_to_visual_cache(result, cache, SCENE, DATASET)
        build_pose_sidecar(cache, force=True, huber_delta=3.0)
    return result, cache


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--edge-count",
        type=int,
        default=EDGE_COUNT,
        help="Number of post-initialization edges to export (default: 30).",
    )
    parser.add_argument(
        "--cold-only",
        action="store_true",
        help="Only run C1 from the active-start frame; keep existing C0 artifacts untouched.",
    )
    parser.add_argument(
        "--repeat-controls",
        action="store_true",
        help="Also run independent C0/C1 repeats to quantify random keypoint-selection repeatability.",
    )
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    values = timestamps()
    s = active_start(values)
    edge_count = int(args.edge_count)
    if edge_count < 1 or s + edge_count + 1 > len(values):
        raise ValueError("--edge-count is outside the available post-initialization frames")
    stop = s + edge_count + 1
    odom, data = write_configs(OUTPUT)
    c0_result = OUTPUT / "results" / "C0_continuous" / SCENE
    c0_cache = OUTPUT / "caches" / "C0_continuous" / SCENE
    if not args.cold_only:
        c0_result, c0_cache = run_case(
            "C0_continuous", 0, stop, odom, data, OUTPUT, force=args.force
        )
    c1_result, c1_cache = run_case("C1_cold", s, stop, odom, data, OUTPUT, force=args.force)
    contract = {
        "dataset": str(DATASET), "active_start_frame": s, "stop_frame_exclusive": stop,
        "edge_count": edge_count, "cold_only": bool(args.cold_only),
        "same_python": sys.executable, "same_odometry_config": str(odom),
        "same_data_config": str(data), "C0_result": str(c0_result), "C0_cache": str(c0_cache),
        "C1_result": str(c1_result), "C1_cache": str(c1_cache),
    }
    if args.repeat_controls:
        c0_repeat_result, c0_repeat_cache = run_case(
            "C0_continuous_repeat", 0, stop, odom, data, OUTPUT, force=args.force
        )
        c1_repeat_result, c1_repeat_cache = run_case(
            "C1_cold_repeat", s, stop, odom, data, OUTPUT, force=args.force
        )
        contract.update(
            {
                "C0_repeat_result": str(c0_repeat_result),
                "C0_repeat_cache": str(c0_repeat_cache),
                "C1_repeat_result": str(c1_repeat_result),
                "C1_repeat_cache": str(c1_repeat_cache),
                "repeat_control_purpose": (
                    "quantify run-to-run differences caused by unseeded torch.randperm keypoint selection"
                ),
            }
        )
    (OUTPUT / "same_build_ab_contract.json").write_text(
        json.dumps(contract, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(contract, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
