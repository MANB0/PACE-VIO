#!/usr/bin/env python3
"""Build cached UVD pose normal equations at each final pure-MACVO pose."""

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

from Utility.CompressedUVDFactorCache import (  # noqa: E402
    COMPRESSED_UVD_FACTOR_FILENAME,
    CompressedUVDFactorPacket,
    write_compressed_uvd_factor_cache,
)
from Utility.TwoStateVIO import (  # noqa: E402
    UVDFactor,
    linearize_uvd_relative_pose_factor,
    uvd_whitened_rows_from_relative,
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
            timestamps = np.asarray(data["frames//time_ns"], dtype=np.int64)
    except (OSError, ValueError, KeyError) as error:
        raise ValueError(f"unable to load pure-MACVO poses from {tensor_map_path}") from error
    frame_count = int(reader.manifest.frame_count)
    if (
        poses.ndim != 2
        or poses.shape[1] != 7
        or poses.shape[0] < frame_count
        or timestamps.shape[0] < frame_count
        or not bool(torch.isfinite(poses).all())
    ):
        raise ValueError("pure-MACVO pose tensor is malformed")
    expected_timestamps = np.asarray(reader.manifest.timestamps_ns, dtype=np.int64)
    if not np.array_equal(timestamps[:frame_count], expected_timestamps):
        raise ValueError("pure-MACVO timestamps do not match the visual-cache prefix")
    return poses[:frame_count]


def _uvd_factor(packet, *, huber_delta: float) -> UVDFactor:
    count = int(packet.points_local.reshape(-1, 3).shape[0])
    uv_cov = packet.match_fields["pixel2_uv_cov"].reshape(count, 3).to(dtype=torch.float64)
    disparity_cov = packet.match_fields["pixel2_disp_cov"].reshape(count).to(dtype=torch.float64)
    covariance = torch.zeros((count, 3, 3), dtype=torch.float64)
    covariance[:, 0, 0] = uv_cov[:, 0]
    covariance[:, 1, 1] = uv_cov[:, 1]
    covariance[:, 0, 1] = uv_cov[:, 2]
    covariance[:, 1, 0] = uv_cov[:, 2]
    covariance[:, 2, 2] = disparity_cov
    return UVDFactor(
        points_Ci=packet.points_local,
        target_uv=packet.match_fields["pixel2_uv"],
        target_disparity=packet.match_fields["pixel2_disp"],
        covariance_uvd=covariance,
        intrinsic=packet.K,
        baseline=float(packet.baseline_m),
        extrinsic_CI=pp.identity_SE3(1, dtype=torch.float64).tensor(),
        huber_delta=float(huber_delta),
    ).to(device=torch.device("cpu"), dtype=torch.float64)


def build_cache(cache_dir: Path, *, force: bool, huber_delta: float) -> Path:
    destination = cache_dir / COMPRESSED_UVD_FACTOR_FILENAME
    if destination.exists() and not force:
        print(f"SKIP {cache_dir}: {destination.name} already exists", flush=True)
        return destination
    reader = VisualFactorCacheReader(cache_dir)
    poses = _source_poses(reader)
    packets: list[CompressedUVDFactorPacket] = []
    pair_count = len(reader.manifest.pairs)
    for row, pair in enumerate(reader.manifest.pairs):
        frame_i = int(pair["frame_i"])
        frame_j = int(pair["frame_j"])
        visual_packet = reader.load_pair(
            frame_i,
            frame_j,
            int(reader.manifest.timestamps_ns[frame_i]),
            int(reader.manifest.timestamps_ns[frame_j]),
        )
        relative_CiCj = (
            pp.SE3(poses[frame_i : frame_i + 1]).Inv()
            @ pp.SE3(poses[frame_j : frame_j + 1])
        )
        reference_CjCi = relative_CiCj.Inv().detach()
        visual = _uvd_factor(visual_packet, huber_delta=huber_delta)
        linearization = linearize_uvd_relative_pose_factor(
            reference_CjCi.tensor(),
            visual,
            marginal_mode="full",
        )
        whitened_rows = uvd_whitened_rows_from_relative(
            reference_CjCi.tensor(), visual
        )
        norms = torch.linalg.vector_norm(whitened_rows, dim=-1)
        packets.append(
            CompressedUVDFactorPacket(
                frame_i=frame_i,
                frame_j=frame_j,
                reference_CjCi=reference_CjCi.tensor().detach().cpu(),
                hessian=linearization.full_hessian.detach().cpu(),
                gradient=linearization.full_gradient.detach().cpu(),
                robust_cost=0.5 * float(linearization.robust_residual.square().sum().item()),
                visual_sha256=visual_packet.visual_sha256,
                num_points=int(norms.numel()),
                num_inliers=int((norms <= float(huber_delta)).sum().item()),
                mean_mahalanobis_sq=float(norms.square().mean().item()),
                huber_delta=float(huber_delta),
            )
        )
        if row == 0 or (row + 1) % 100 == 0 or row + 1 == pair_count:
            print(f"BUILD {cache_dir.name}: {row + 1}/{pair_count}", flush=True)
    path = write_compressed_uvd_factor_cache(cache_dir, packets)
    print(f"OK {cache_dir}: {path}", flush=True)
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", action="append", default=[])
    parser.add_argument("--cache-root", action="append", default=[])
    parser.add_argument("--huber-delta", type=float, default=0.1)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    arguments = parse_args()
    if arguments.huber_delta <= 0.0:
        raise ValueError("--huber-delta must be positive")
    for cache_dir in _cache_directories(arguments):
        build_cache(
            cache_dir,
            force=bool(arguments.force),
            huber_delta=float(arguments.huber_delta),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
