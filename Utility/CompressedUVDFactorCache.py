"""Portable cache for MACVO UVD normal equations at the final visual pose."""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from Utility.VisualFactorCache import VisualFactorCacheError


COMPRESSED_UVD_FACTOR_SCHEMA_VERSION = 1
COMPRESSED_UVD_FACTOR_FILENAME = "compressed_uvd_pose_factors.npz"
TANGENT_ORDER = "translation_rotation"
PERTURBATION_SIDE = "right"
POSE_DIRECTION = "Cj_from_Ci"


@dataclass(frozen=True)
class CompressedUVDFactorPacket:
    frame_i: int
    frame_j: int
    reference_CjCi: torch.Tensor
    hessian: torch.Tensor
    gradient: torch.Tensor
    robust_cost: float
    visual_sha256: str
    num_points: int
    num_inliers: int
    mean_mahalanobis_sq: float
    huber_delta: float


def write_compressed_uvd_factor_cache(
    cache_dir: str | Path,
    packets: Sequence[CompressedUVDFactorPacket],
) -> Path:
    cache_path = Path(cache_dir)
    ordered = list(packets)
    if not ordered:
        raise VisualFactorCacheError("compressed UVD factor cache cannot be empty")
    for expected_i, packet in enumerate(ordered):
        if (packet.frame_i, packet.frame_j) != (expected_i, expected_i + 1):
            raise VisualFactorCacheError("compressed UVD factor pairs must be contiguous")
        if packet.reference_CjCi.shape != (1, 7):
            raise VisualFactorCacheError("compressed UVD reference pose must have shape (1, 7)")
        if packet.hessian.shape != (6, 6) or packet.gradient.shape != (6,):
            raise VisualFactorCacheError("compressed UVD normal equations have invalid shape")
        if not bool(
            torch.isfinite(packet.reference_CjCi).all()
            and torch.isfinite(packet.hessian).all()
            and torch.isfinite(packet.gradient).all()
        ):
            raise VisualFactorCacheError("compressed UVD factor contains NaN/Inf")
        if float((packet.hessian - packet.hessian.mT).abs().max().item()) > 1.0e-8:
            raise VisualFactorCacheError("compressed UVD Hessian is not symmetric")
        if not str(packet.visual_sha256):
            raise VisualFactorCacheError("compressed UVD visual hash is empty")
        if packet.num_points < 3 or not 0 <= packet.num_inliers <= packet.num_points:
            raise VisualFactorCacheError("compressed UVD point diagnostics are invalid")
        if not np.isfinite(packet.robust_cost) or not np.isfinite(packet.mean_mahalanobis_sq):
            raise VisualFactorCacheError("compressed UVD scalar diagnostics must be finite")
        if not np.isfinite(packet.huber_delta) or packet.huber_delta <= 0.0:
            raise VisualFactorCacheError("compressed UVD Huber delta must be positive")
        eigenvalues = torch.linalg.eigvalsh(0.5 * (packet.hessian + packet.hessian.mT))
        tolerance = max(torch.finfo(eigenvalues.dtype).eps * max(float(eigenvalues.abs().max()), 1.0) * 6.0, 1.0e-10)
        if float(eigenvalues.min().item()) < -tolerance or not bool((eigenvalues > tolerance).any()):
            raise VisualFactorCacheError("compressed UVD Hessian is not positive semidefinite with nonzero rank")

    arrays = {
        "schema_version": np.asarray(COMPRESSED_UVD_FACTOR_SCHEMA_VERSION, dtype=np.int64),
        "tangent_order": np.asarray(TANGENT_ORDER),
        "perturbation_side": np.asarray(PERTURBATION_SIDE),
        "pose_direction": np.asarray(POSE_DIRECTION),
        "frame_i": np.asarray([packet.frame_i for packet in ordered], dtype=np.int64),
        "frame_j": np.asarray([packet.frame_j for packet in ordered], dtype=np.int64),
        "reference_CjCi": np.stack([packet.reference_CjCi.detach().cpu().numpy() for packet in ordered]),
        "hessian": np.stack([packet.hessian.detach().cpu().numpy() for packet in ordered]),
        "gradient": np.stack([packet.gradient.detach().cpu().numpy() for packet in ordered]),
        "robust_cost": np.asarray([packet.robust_cost for packet in ordered], dtype=np.float64),
        "visual_sha256": np.asarray([packet.visual_sha256 for packet in ordered]),
        "num_points": np.asarray([packet.num_points for packet in ordered], dtype=np.int64),
        "num_inliers": np.asarray([packet.num_inliers for packet in ordered], dtype=np.int64),
        "mean_mahalanobis_sq": np.asarray([packet.mean_mahalanobis_sq for packet in ordered], dtype=np.float64),
        "huber_delta": np.asarray([packet.huber_delta for packet in ordered], dtype=np.float64),
    }
    cache_path.mkdir(parents=True, exist_ok=True)
    destination = cache_path / COMPRESSED_UVD_FACTOR_FILENAME
    fd, temporary_name = tempfile.mkstemp(
        prefix=destination.name + ".", suffix=".tmp", dir=cache_path
    )
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


class CompressedUVDFactorCacheReader:
    def __init__(self, cache_dir: str | Path):
        path = Path(cache_dir) / COMPRESSED_UVD_FACTOR_FILENAME
        required = {
            "schema_version", "tangent_order", "perturbation_side", "pose_direction",
            "frame_i", "frame_j", "reference_CjCi", "hessian", "gradient",
            "robust_cost", "visual_sha256", "num_points", "num_inliers",
            "mean_mahalanobis_sq", "huber_delta",
        }
        try:
            with np.load(path, allow_pickle=False) as data:
                if set(data.files) != required:
                    raise VisualFactorCacheError("compressed UVD cache fields do not match schema")
                if int(data["schema_version"].item()) != COMPRESSED_UVD_FACTOR_SCHEMA_VERSION:
                    raise VisualFactorCacheError("unsupported compressed UVD cache schema")
                if str(data["tangent_order"].item()) != TANGENT_ORDER:
                    raise VisualFactorCacheError("compressed UVD tangent order differs")
                if str(data["perturbation_side"].item()) != PERTURBATION_SIDE:
                    raise VisualFactorCacheError("compressed UVD perturbation side differs")
                if str(data["pose_direction"].item()) != POSE_DIRECTION:
                    raise VisualFactorCacheError("compressed UVD pose direction differs")
                self.frame_i = data["frame_i"].copy()
                self.frame_j = data["frame_j"].copy()
                self.references = torch.from_numpy(data["reference_CjCi"].copy())
                self.hessians = torch.from_numpy(data["hessian"].copy())
                self.gradients = torch.from_numpy(data["gradient"].copy())
                self.robust_costs = data["robust_cost"].copy()
                self.visual_hashes = data["visual_sha256"].astype(str).copy()
                self.num_points = data["num_points"].copy()
                self.num_inliers = data["num_inliers"].copy()
                self.mean_mahalanobis_sq = data["mean_mahalanobis_sq"].copy()
                self.huber_delta = data["huber_delta"].copy()
        except VisualFactorCacheError:
            raise
        except (OSError, ValueError, KeyError, TypeError) as error:
            raise VisualFactorCacheError("unable to read compressed UVD factor cache") from error

        count = len(self.frame_i)
        values = (
            self.frame_j, self.references, self.hessians, self.gradients,
            self.robust_costs, self.visual_hashes, self.num_points,
            self.num_inliers, self.mean_mahalanobis_sq, self.huber_delta,
        )
        if count == 0 or any(len(value) != count for value in values):
            raise VisualFactorCacheError("compressed UVD cache row counts differ")
        expected_i = np.arange(count, dtype=np.int64)
        if not np.array_equal(self.frame_i, expected_i) or not np.array_equal(self.frame_j, expected_i + 1):
            raise VisualFactorCacheError("compressed UVD factor indices are not contiguous")
        if self.references.shape != (count, 1, 7) or self.hessians.shape != (count, 6, 6) or self.gradients.shape != (count, 6):
            raise VisualFactorCacheError("compressed UVD cache tensor shape is invalid")
        if not bool(torch.isfinite(self.references).all() and torch.isfinite(self.hessians).all() and torch.isfinite(self.gradients).all()):
            raise VisualFactorCacheError("compressed UVD cache contains NaN/Inf")
        if float((self.hessians - self.hessians.transpose(-1, -2)).abs().max().item()) > 1.0e-8:
            raise VisualFactorCacheError("compressed UVD cache Hessian is not symmetric")
        if any(not str(value) for value in self.visual_hashes):
            raise VisualFactorCacheError("compressed UVD cache visual hash is empty")

    def load_pair(
        self, frame_i: int, frame_j: int, visual_sha256: str
    ) -> CompressedUVDFactorPacket:
        if frame_j != frame_i + 1 or frame_i < 0 or frame_i >= len(self.frame_i):
            raise VisualFactorCacheError("compressed UVD factor pair is out of range")
        row = int(frame_i)
        if int(self.frame_i[row]) != frame_i or int(self.frame_j[row]) != frame_j:
            raise VisualFactorCacheError("compressed UVD factor pair indices differ")
        if str(self.visual_hashes[row]) != str(visual_sha256):
            raise VisualFactorCacheError("compressed UVD factor visual hash differs")
        return CompressedUVDFactorPacket(
            frame_i=frame_i,
            frame_j=frame_j,
            reference_CjCi=self.references[row].clone(),
            hessian=self.hessians[row].clone(),
            gradient=self.gradients[row].clone(),
            robust_cost=float(self.robust_costs[row]),
            visual_sha256=str(self.visual_hashes[row]),
            num_points=int(self.num_points[row]),
            num_inliers=int(self.num_inliers[row]),
            mean_mahalanobis_sq=float(self.mean_mahalanobis_sq[row]),
            huber_delta=float(self.huber_delta[row]),
        )
