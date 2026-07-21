#!/usr/bin/env python3
"""Restart pure visual MACVO at the post-static boundary and compare factors."""

from __future__ import annotations

import csv
import dataclasses
import json
import runpy
import shutil
import sys
from pathlib import Path

import numpy as np
import pypose as pp
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Utility.RelativePoseFactorCache import (  # noqa: E402
    RelativePoseFactorCacheReader,
    relative_pose_information_from_packet,
)
from Utility.VisualFactorCache import (  # noqa: E402
    VisualFactorCacheReader,
    write_visual_factor_cache,
)
from Odometry.MACVO import MACVO  # noqa: E402


OUT = ROOT / "analysis_initialization_boundary_audit_20260716"
SCENE = "clear_circle_truth_normal_noise"
ACTIVE_START = 90
FRAME_LIMIT = 391
SOURCE_RESULT = ROOT / (
    "Results/visual_factor_cache_static63_unique_source_20260713/trial_1/"
    "pure_macvo/clear_circle_truth_normal_noise"
)
SOURCE_CACHE = ROOT / "VisualCache/static63_unique_visual_20260713" / SCENE
SLICED_CACHE = OUT / "counterfactual_r2_visual_cache_301"
DATA_CONFIG = OUT / "counterfactual_data_circle.yaml"
ODOM_CONFIG = OUT / "counterfactual_r2_visual_restart.yaml"
RUN_ROOT = OUT / "counterfactual_r2_visual_restart_result"
COMPARISON_CSV = OUT / "r2_visual_restart_comparison_per_edge.csv"
COMPARISON_JSON = OUT / "r2_visual_restart_comparison.json"


def validate_r2_cache_slice(self: MACVO, sequence) -> None:
    """Permit only the exact diagnostic slice while retaining cache index checks."""
    reader = self._visual_cache_reader
    if reader is None:
        return
    indices = getattr(sequence, "indices", None)
    actual = [int(value) for value in indices.tolist()]
    expected = list(range(ACTIVE_START, FRAME_LIMIT))
    if actual != expected:
        raise RuntimeError("R2 replay sequence is not the exact requested source-index slice")
    if reader.manifest.frame_count != FRAME_LIMIT - ACTIVE_START:
        raise RuntimeError("R2 replay cache does not contain exactly the requested slice")
    self._visual_cache_sequence_frame_count = len(actual)


def write_sliced_cache() -> None:
    if SLICED_CACHE.exists():
        shutil.rmtree(SLICED_CACHE)
    reader = VisualFactorCacheReader(SOURCE_CACHE)
    packets = []
    for local_i, source_i in enumerate(range(ACTIVE_START, FRAME_LIMIT - 1)):
        source_j = source_i + 1
        packet = reader.load_pair(
            source_i,
            source_j,
            int(reader.manifest.timestamps_ns[source_i]),
            int(reader.manifest.timestamps_ns[source_j]),
        )
        packets.append(dataclasses.replace(packet, frame_i=local_i, frame_j=local_i + 1))
    source = dict(reader.manifest.source)
    source["frame_count"] = FRAME_LIMIT - ACTIVE_START
    write_visual_factor_cache(SLICED_CACHE, SCENE, packets, source=source)


def write_config() -> None:
    source = yaml.safe_load((SOURCE_RESULT / "config.yaml").read_text(encoding="utf-8"))
    odometry = source["Odometry"]
    args = odometry["args"]
    args["visual_cache_mode"] = "replay"
    args["visual_cache_path"] = str(SLICED_CACHE)
    args["imu_static_initialization_enable"] = False
    args["imu_rot_prior_enable"] = False
    args["imu_trans_prior_enable"] = False
    odometry["frontend"] = {"type": "ReplayFrontend", "args": {}}
    optimizer = odometry["optimizer"]["args"]
    optimizer["imu_factor_mode"] = "legacy_pose_prior"
    optimizer["imu_rot_prior"] = False
    optimizer["post_imu_fusion_enable"] = False
    optimizer["post_imu_fusion_prepose_enable"] = False
    ODOM_CONFIG.write_text(
        yaml.safe_dump(
            {"Common": {"device": "cuda"}, "Odometry": odometry, "Preprocess": None},
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def read_poses(path: Path) -> pp.LieTensor:
    rows = list(csv.DictReader(path.open("r", encoding="utf-8")))
    return pp.SE3(
        torch.tensor(
            [[float(row[name]) for name in ("tx", "ty", "tz", "qx", "qy", "qz", "qw")] for row in rows],
            dtype=torch.float64,
        )
    )


def summarize(tensor_map_path: Path) -> None:
    with np.load(tensor_map_path, allow_pickle=False) as data:
        poses = pp.SE3(torch.from_numpy(data["frames//pose"].copy()).to(dtype=torch.float64))
    if poses.shape[0] != FRAME_LIMIT - ACTIVE_START:
        raise ValueError(f"expected {FRAME_LIMIT - ACTIVE_START} poses, got {poses.shape[0]}")
    regenerated = poses[:-1].Inv() @ poses[1:]
    visual_reader = VisualFactorCacheReader(SOURCE_CACHE)
    factor_reader = RelativePoseFactorCacheReader(SOURCE_CACHE)
    rows: list[dict[str, object]] = []
    mean_norms: list[float] = []
    whitened_mean_norms: list[float] = []
    covariance_rel_errors: list[float] = []
    for local_edge, frame_i in enumerate(range(ACTIVE_START, FRAME_LIMIT - 1)):
        frame_j = frame_i + 1
        packet = visual_reader.load_pair(
            frame_i,
            frame_j,
            int(visual_reader.manifest.timestamps_ns[frame_i]),
            int(visual_reader.manifest.timestamps_ns[frame_j]),
        )
        original = factor_reader.load_pair(frame_i, frame_j, packet.visual_sha256)
        z_r2 = regenerated[local_edge : local_edge + 1]
        mean_error = (pp.SE3(original.measurement_CiCj).to(torch.float64).Inv() @ z_r2).Log().tensor().reshape(6)
        covariance_r2, _ = relative_pose_information_from_packet(
            packet,
            z_r2.tensor(),
            huber_delta=3.0,
        )
        covariance_original = original.covariance.to(torch.float64)
        covariance_r2 = covariance_r2.to(torch.float64)
        covariance_rel = torch.linalg.matrix_norm(covariance_r2 - covariance_original) / torch.linalg.matrix_norm(
            covariance_original
        ).clamp_min(1e-30)
        mean_norm = torch.linalg.vector_norm(mean_error)
        covariance_sym = 0.5 * (covariance_original + covariance_original.mT)
        values, vectors = torch.linalg.eigh(covariance_sym)
        floor = max(1e-12, float(values.abs().max()) * torch.finfo(torch.float64).eps)
        covariance_stable = vectors @ torch.diag(values.clamp_min(floor)) @ vectors.mT
        whitened_mean = torch.linalg.solve_triangular(
            torch.linalg.cholesky(covariance_stable),
            mean_error.reshape(6, 1),
            upper=False,
        ).reshape(6)
        whitened_mean_norm = torch.linalg.vector_norm(whitened_mean)
        mean_norms.append(float(mean_norm))
        whitened_mean_norms.append(float(whitened_mean_norm))
        covariance_rel_errors.append(float(covariance_rel))
        row: dict[str, object] = {
            "edge_id": local_edge,
            "frame_i": frame_i,
            "frame_j": frame_j,
            "mean_se3_log_norm": float(mean_norm),
            "mean_whitened_norm_using_original_covariance": float(whitened_mean_norm),
            "covariance_frobenius_relative_error": float(covariance_rel),
        }
        for axis, value in enumerate(mean_error.tolist()):
            row[f"mean_error_{axis}"] = value
        rows.append(row)
    with COMPARISON_CSV.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "mode": "R2 actual pure-visual restart at frame s",
        "active_start_frame": ACTIVE_START,
        "frame_limit_exclusive": FRAME_LIMIT,
        "edge_count": len(rows),
        "uses_pre_s_absolute_pose": False,
        "uses_pre_s_motion_model_state": False,
        "visual_packets_are_pair_local_replay_measurements": True,
        "mean_se3_log_norm": {
            "median": float(np.median(mean_norms)),
            "p95": float(np.quantile(mean_norms, 0.95)),
            "max": float(np.max(mean_norms)),
        },
        "mean_whitened_norm_using_original_covariance": {
            "median": float(np.median(whitened_mean_norms)),
            "p95": float(np.quantile(whitened_mean_norms, 0.95)),
            "max": float(np.max(whitened_mean_norms)),
        },
        "covariance_frobenius_relative_error": {
            "median": float(np.median(covariance_rel_errors)),
            "p95": float(np.quantile(covariance_rel_errors, 0.95)),
            "max": float(np.max(covariance_rel_errors)),
        },
        "pass_threshold": {
            "mean_se3_log_norm": 1e-5,
            "mean_whitened_norm": 0.01,
            "covariance_frobenius_relative_error": 1e-5,
        },
        "pass": bool(
            max(mean_norms) <= 1e-5
            and max(whitened_mean_norms) <= 0.01
            and max(covariance_rel_errors) <= 1e-5
        ),
        "tensor_map_output": str(tensor_map_path),
    }
    COMPARISON_JSON.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    if "--summarize-only" in sys.argv:
        candidates = sorted(RUN_ROOT.rglob("tensor_map.npz"))
        if not candidates:
            raise FileNotFoundError("saved R2 restart has no tensor_map.npz")
        summarize(candidates[-1])
        print(COMPARISON_JSON.read_text(encoding="utf-8"))
        return 0
    write_sliced_cache()
    write_config()
    if RUN_ROOT.exists():
        resolved = RUN_ROOT.resolve()
        if OUT.resolve() not in resolved.parents:
            raise RuntimeError("refusing to remove output outside audit directory")
        shutil.rmtree(RUN_ROOT)
    sys.argv = [
        str(ROOT / "MACVO.py"),
        "--odom", str(ODOM_CONFIG),
        "--data", str(DATA_CONFIG),
        "--resultRoot", str(RUN_ROOT),
        "--visual-cache-mode", "replay",
        "--visual-cache-path", str(SLICED_CACHE),
        "--seq_from", str(ACTIVE_START),
        "--seq_to", str(FRAME_LIMIT),
    ]
    MACVO._validate_visual_cache_sequence_metadata = validate_r2_cache_slice
    runpy.run_path(str(ROOT / "MACVO.py"), run_name="__main__")
    tensor_map_candidates = sorted(RUN_ROOT.rglob("tensor_map.npz"))
    if not tensor_map_candidates:
        raise FileNotFoundError("R2 visual restart did not produce tensor_map.npz")
    summarize(tensor_map_candidates[-1])
    print(COMPARISON_JSON.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
