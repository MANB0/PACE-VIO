#!/usr/bin/env python3
"""Build compact MACVO relative-pose factors beside visual cache v1 packets."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pypose as pp
import torch


WORKDIR = Path(__file__).resolve().parents[1]
if str(WORKDIR) not in sys.path:
    sys.path.insert(0, str(WORKDIR))

from Utility.RelativePoseFactorCache import (  # noqa: E402
    RELATIVE_POSE_FACTOR_FILENAME,
    RelativePoseFactorPacket,
    relative_pose_information_from_packet,
    write_relative_pose_factor_cache,
)
from Utility.VisualFactorCache import VisualFactorCacheReader  # noqa: E402


def _cache_directories(arguments: argparse.Namespace) -> list[Path]:
    directories = [Path(value).expanduser().resolve() for value in arguments.cache_dir]
    for root_value in arguments.cache_root:
        root = Path(root_value).expanduser().resolve()
        directories.extend(path.parent for path in sorted(root.rglob("manifest.json")))
    unique = sorted(set(directories))
    if not unique:
        raise ValueError("no visual cache directories were selected")
    return unique


def _source_poses(reader: VisualFactorCacheReader) -> torch.Tensor:
    source_result = reader.manifest.source.get("result")
    if not isinstance(source_result, str) or not source_result:
        raise ValueError("visual cache manifest does not identify its pure-MACVO result")
    tensor_map_path = Path(source_result) / "tensor_map.npz"
    try:
        with np.load(tensor_map_path, allow_pickle=False) as data:
            poses = torch.from_numpy(data["frames//pose"].copy()).to(dtype=torch.float64)
    except (OSError, ValueError, KeyError) as error:
        raise ValueError(f"unable to load pure-MACVO poses from {tensor_map_path}") from error
    if poses.shape != (reader.manifest.frame_count, 7) or not torch.isfinite(poses).all():
        raise ValueError("pure-MACVO pose tensor is malformed")
    return poses


def build_cache(cache_dir: Path, *, force: bool, huber_delta: float) -> Path:
    destination = cache_dir / RELATIVE_POSE_FACTOR_FILENAME
    if destination.exists() and not force:
        print(f"SKIP {cache_dir}: {destination.name} already exists", flush=True)
        return destination
    reader = VisualFactorCacheReader(cache_dir)
    poses = _source_poses(reader)
    packets: list[RelativePoseFactorPacket] = []
    pair_count = len(reader.manifest.pairs)
    for row, pair in enumerate(reader.manifest.pairs):
        frame_i = int(pair["frame_i"])
        frame_j = int(pair["frame_j"])
        visual = reader.load_pair(
            frame_i,
            frame_j,
            int(reader.manifest.timestamps_ns[frame_i]),
            int(reader.manifest.timestamps_ns[frame_j]),
        )
        relative = pp.SE3(poses[frame_i : frame_i + 1]).Inv() @ pp.SE3(poses[frame_j : frame_j + 1])
        covariance, diagnostics = relative_pose_information_from_packet(
            visual,
            relative.tensor(),
            huber_delta=huber_delta,
        )
        packets.append(
            RelativePoseFactorPacket(
                frame_i=frame_i,
                frame_j=frame_j,
                measurement_CiCj=relative.tensor().detach().cpu(),
                covariance=covariance.detach().cpu(),
                visual_sha256=visual.visual_sha256,
                num_points=int(diagnostics["num_points"]),
                num_inliers=int(diagnostics["num_inliers"]),
                mean_mahalanobis_sq=float(diagnostics["mean_mahalanobis_sq"]),
            )
        )
        if row == 0 or (row + 1) % 100 == 0 or row + 1 == pair_count:
            print(f"BUILD {cache_dir.name}: {row + 1}/{pair_count}", flush=True)
    path = write_relative_pose_factor_cache(cache_dir, packets)
    print(f"OK {cache_dir}: {path}", flush=True)
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", action="append", default=[], help="One visual cache directory.")
    parser.add_argument("--cache-root", action="append", default=[], help="Recursively find visual caches below this root.")
    parser.add_argument("--huber-delta", type=float, default=3.0)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    arguments = parse_args()
    if arguments.huber_delta <= 0.0:
        raise ValueError("--huber-delta must be positive")
    for cache_dir in _cache_directories(arguments):
        build_cache(cache_dir, force=bool(arguments.force), huber_delta=float(arguments.huber_delta))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
