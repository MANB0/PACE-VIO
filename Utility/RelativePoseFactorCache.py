from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pypose as pp
import torch

from Utility.Point import pixel2point_NED
from Utility.VisualFactorCache import VisualFactorCacheError, VisualFactorPacket


RELATIVE_POSE_FACTOR_SCHEMA_VERSION = 1
RELATIVE_POSE_FACTOR_FILENAME = "relative_pose_factors.npz"


@dataclass(frozen=True)
class RelativePoseFactorPacket:
    frame_i: int
    frame_j: int
    measurement_CiCj: torch.Tensor
    covariance: torch.Tensor
    visual_sha256: str
    num_points: int
    num_inliers: int
    mean_mahalanobis_sq: float


def _skew(points: torch.Tensor) -> torch.Tensor:
    zeros = torch.zeros_like(points[..., 0])
    x, y, z = points.unbind(dim=-1)
    return torch.stack(
        [
            zeros, -z, y,
            z, zeros, -x,
            -y, x, zeros,
        ],
        dim=-1,
    ).reshape(points.shape[:-1] + (3, 3))


def relative_pose_information_from_correspondences(
    points_Ci: torch.Tensor,
    points_Cj: torch.Tensor,
    covariance_Ci: torch.Tensor,
    covariance_Cj: torch.Tensor,
    measurement_CiCj: torch.Tensor,
    *,
    huber_delta: float = 3.0,
    covariance_eigenvalue_floor: float = 1.0e-12,
    information_eigenvalue_floor: float = 1.0e-12,
) -> tuple[torch.Tensor, dict[str, float | int]]:
    """Build the Pose-factor covariance from paired camera-frame 3D points."""

    dtype = torch.float64
    pose = pp.SE3(measurement_CiCj.reshape(1, 7).to(dtype=dtype, device="cpu"))
    points_i = points_Ci.detach().to(device="cpu", dtype=dtype).reshape(-1, 3)
    points_j = points_Cj.detach().to(device="cpu", dtype=dtype).reshape(-1, 3)
    covariance_i = covariance_Ci.detach().to(
        device="cpu", dtype=dtype
    ).reshape(-1, 3, 3)
    covariance_j = covariance_Cj.detach().to(
        device="cpu", dtype=dtype
    ).reshape(-1, 3, 3)
    count = int(points_i.shape[0])
    if count < 3 or points_j.shape != points_i.shape:
        raise VisualFactorCacheError(
            "relative pose covariance requires at least three paired 3D points"
        )
    if int(covariance_i.shape[0]) != count or int(covariance_j.shape[0]) != count:
        raise VisualFactorCacheError(
            "relative pose covariance rows must match paired 3D points"
        )
    finite = torch.cat(
        [
            points_i.reshape(-1),
            points_j.reshape(-1),
            covariance_i.reshape(-1),
            covariance_j.reshape(-1),
        ]
    )
    if not bool(torch.isfinite(finite).all()):
        raise VisualFactorCacheError(
            "relative pose covariance inputs contain NaN/Inf"
        )

    rotation = pose.rotation().matrix().reshape(3, 3)
    translation = pose.translation().reshape(3)
    predicted_i = (rotation @ points_j.mT).mT + translation
    residual = points_i - predicted_i
    covariance = covariance_i + rotation.unsqueeze(0) @ covariance_j @ rotation.mT.unsqueeze(0)
    covariance = 0.5 * (covariance + covariance.transpose(-1, -2))
    values, vectors = torch.linalg.eigh(covariance)
    covariance_floor = max(float(covariance_eigenvalue_floor), torch.finfo(dtype).eps)
    covariance = vectors @ torch.diag_embed(values.clamp_min(covariance_floor)) @ vectors.transpose(-1, -2)
    lower = torch.linalg.cholesky(covariance)

    rotation_batch = rotation.unsqueeze(0).expand(points_j.shape[0], -1, -1)
    jacobian = torch.cat(
        [
            -rotation_batch,
            rotation_batch @ _skew(points_j),
        ],
        dim=-1,
    )
    whitened_residual = torch.linalg.solve_triangular(
        lower,
        residual.unsqueeze(-1),
        upper=False,
    ).squeeze(-1)
    whitened_jacobian = torch.linalg.solve_triangular(lower, jacobian, upper=False)
    norms = torch.linalg.vector_norm(whitened_residual, dim=-1)
    delta = max(float(huber_delta), 1.0e-12)
    weights = torch.where(
        norms <= delta,
        torch.ones_like(norms),
        torch.as_tensor(delta, dtype=dtype) / norms.clamp_min(1.0e-12),
    )
    weighted_jacobian = weights.sqrt().reshape(-1, 1, 1) * whitened_jacobian
    information = torch.einsum("nki,nkj->ij", weighted_jacobian, weighted_jacobian)
    information = 0.5 * (information + information.mT)

    info_values, info_vectors = torch.linalg.eigh(information)
    info_floor = max(float(information_eigenvalue_floor), torch.finfo(dtype).eps)
    inlier_mask = norms <= delta
    inlier_mean_mahalanobis_sq = (
        whitened_residual[inlier_mask].square().sum(dim=-1).mean()
        if bool(inlier_mask.any())
        else whitened_residual.square().sum(dim=-1).mean()
    )
    covariance_inflation = max(float(inlier_mean_mahalanobis_sq.item()) / 3.0, 1.0)
    covariance_pose = (
        info_vectors
        @ torch.diag(info_values.clamp_min(info_floor).reciprocal())
        @ info_vectors.mT
    ) * covariance_inflation
    covariance_pose = 0.5 * (covariance_pose + covariance_pose.mT)
    diagnostics: dict[str, float | int] = {
        "num_points": count,
        "num_inliers": int(inlier_mask.sum().item()),
        "mean_mahalanobis_sq": float(
            whitened_residual.square().sum(dim=-1).mean().item()
        ),
        "inlier_mean_mahalanobis_sq": float(inlier_mean_mahalanobis_sq.item()),
        "covariance_inflation": float(covariance_inflation),
        "information_min_eigenvalue": float(info_values.min().item()),
        "information_max_eigenvalue": float(info_values.max().item()),
    }
    return covariance_pose, diagnostics


def relative_pose_information_from_packet(
    packet: VisualFactorPacket,
    measurement_CiCj: torch.Tensor,
    *,
    huber_delta: float = 3.0,
    covariance_eigenvalue_floor: float = 1.0e-12,
    information_eigenvalue_floor: float = 1.0e-12,
) -> tuple[torch.Tensor, dict[str, float | int]]:
    """Compress cached two-sided 3D measurements into a 6x6 pose covariance."""

    dtype = torch.float64
    points_i = packet.points_local.detach().to(device="cpu", dtype=dtype).reshape(-1, 3)
    points_j = pixel2point_NED(
        packet.match_fields["pixel2_uv"].detach().to(device="cpu", dtype=dtype).unsqueeze(0),
        packet.match_fields["pixel2_d"].detach().to(device="cpu", dtype=dtype).reshape(1, -1),
        packet.K.detach().to(device="cpu", dtype=dtype),
    ).squeeze(0).reshape(-1, 3)
    covariance_i = packet.match_fields["obs1_covTc"].detach().to(device="cpu", dtype=dtype)
    covariance_j = packet.match_fields["obs2_covTc"].detach().to(device="cpu", dtype=dtype)
    return relative_pose_information_from_correspondences(
        points_i,
        points_j,
        covariance_i,
        covariance_j,
        measurement_CiCj,
        huber_delta=huber_delta,
        covariance_eigenvalue_floor=covariance_eigenvalue_floor,
        information_eigenvalue_floor=information_eigenvalue_floor,
    )


def camera_factor_to_body_factor(
    measurement_CiCj: torch.Tensor,
    covariance_camera: torch.Tensor,
    sensor_T_imu: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert T_CiCj and its right-perturbation covariance to IMU/body coordinates."""

    dtype = covariance_camera.dtype
    device = covariance_camera.device
    extrinsic_CI = pp.SE3(sensor_T_imu.reshape(1, 7).to(device=device, dtype=dtype))
    measurement_camera = pp.SE3(measurement_CiCj.reshape(1, 7).to(device=device, dtype=dtype))
    measurement_body = extrinsic_CI.Inv() @ measurement_camera @ extrinsic_CI
    basis = pp.se3(torch.eye(6, dtype=dtype, device=device))
    adjoint = extrinsic_CI.Inv().Adj(basis).tensor().mT
    covariance_body = adjoint @ covariance_camera.reshape(6, 6).to(adjoint) @ adjoint.mT
    return measurement_body.tensor(), 0.5 * (covariance_body + covariance_body.mT)


def write_relative_pose_factor_cache(
    cache_dir: str | Path,
    packets: Sequence[RelativePoseFactorPacket],
) -> Path:
    cache_path = Path(cache_dir)
    ordered = list(packets)
    if not ordered:
        raise VisualFactorCacheError("relative pose factor cache cannot be empty")
    for expected_i, packet in enumerate(ordered):
        if (packet.frame_i, packet.frame_j) != (expected_i, expected_i + 1):
            raise VisualFactorCacheError("relative pose factor pairs must be contiguous")
        if packet.measurement_CiCj.shape != (1, 7):
            raise VisualFactorCacheError("relative pose measurement must have shape (1, 7)")
        if packet.covariance.shape != (6, 6):
            raise VisualFactorCacheError("relative pose covariance must have shape (6, 6)")
        if not torch.isfinite(packet.measurement_CiCj).all() or not torch.isfinite(packet.covariance).all():
            raise VisualFactorCacheError("relative pose factors must be finite")

    arrays = {
        "schema_version": np.asarray(RELATIVE_POSE_FACTOR_SCHEMA_VERSION, dtype=np.int64),
        "frame_i": np.asarray([packet.frame_i for packet in ordered], dtype=np.int64),
        "frame_j": np.asarray([packet.frame_j for packet in ordered], dtype=np.int64),
        "measurement_CiCj": np.stack([packet.measurement_CiCj.detach().cpu().numpy() for packet in ordered]),
        "covariance": np.stack([packet.covariance.detach().cpu().numpy() for packet in ordered]),
        "visual_sha256": np.asarray([packet.visual_sha256 for packet in ordered]),
        "num_points": np.asarray([packet.num_points for packet in ordered], dtype=np.int64),
        "num_inliers": np.asarray([packet.num_inliers for packet in ordered], dtype=np.int64),
        "mean_mahalanobis_sq": np.asarray([packet.mean_mahalanobis_sq for packet in ordered], dtype=np.float64),
    }
    cache_path.mkdir(parents=True, exist_ok=True)
    destination = cache_path / RELATIVE_POSE_FACTOR_FILENAME
    fd, temporary_name = tempfile.mkstemp(prefix=destination.name + ".", suffix=".tmp", dir=cache_path)
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


class RelativePoseFactorCacheReader:
    def __init__(self, cache_dir: str | Path):
        path = Path(cache_dir) / RELATIVE_POSE_FACTOR_FILENAME
        try:
            with np.load(path, allow_pickle=False) as data:
                required = {
                    "schema_version", "frame_i", "frame_j", "measurement_CiCj", "covariance",
                    "visual_sha256", "num_points", "num_inliers", "mean_mahalanobis_sq",
                }
                if set(data.files) != required:
                    raise VisualFactorCacheError("relative pose factor cache fields do not match schema")
                if int(data["schema_version"].item()) != RELATIVE_POSE_FACTOR_SCHEMA_VERSION:
                    raise VisualFactorCacheError("unsupported relative pose factor cache schema")
                self.frame_i = data["frame_i"].copy()
                self.frame_j = data["frame_j"].copy()
                self.measurements = torch.from_numpy(data["measurement_CiCj"].copy())
                self.covariances = torch.from_numpy(data["covariance"].copy())
                self.visual_hashes = data["visual_sha256"].astype(str).copy()
                self.num_points = data["num_points"].copy()
                self.num_inliers = data["num_inliers"].copy()
                self.mean_mahalanobis_sq = data["mean_mahalanobis_sq"].copy()
        except VisualFactorCacheError:
            raise
        except (OSError, ValueError, KeyError, TypeError) as error:
            raise VisualFactorCacheError("unable to read relative pose factor cache") from error
        count = len(self.frame_i)
        if any(len(values) != count for values in (
            self.frame_j,
            self.measurements,
            self.covariances,
            self.visual_hashes,
            self.num_points,
            self.num_inliers,
            self.mean_mahalanobis_sq,
        )):
            raise VisualFactorCacheError("relative pose factor cache row counts differ")
        if self.measurements.shape != (count, 1, 7) or self.covariances.shape != (count, 6, 6):
            raise VisualFactorCacheError("relative pose factor cache tensor shape is invalid")
        expected_i = np.arange(count, dtype=np.int64)
        if count == 0 or not np.array_equal(self.frame_i, expected_i):
            raise VisualFactorCacheError("relative pose factor source indices are not contiguous")
        if not np.array_equal(self.frame_j, expected_i + 1):
            raise VisualFactorCacheError("relative pose factor destination indices are not contiguous")
        if not torch.isfinite(self.measurements).all() or not torch.isfinite(self.covariances).all():
            raise VisualFactorCacheError("relative pose factor cache contains non-finite values")
        covariance_symmetry_error = (
            self.covariances - self.covariances.transpose(-1, -2)
        ).abs().max()
        if float(covariance_symmetry_error.item()) > 1.0e-8:
            raise VisualFactorCacheError("relative pose factor covariance is not symmetric")
        if bool((torch.linalg.eigvalsh(self.covariances) <= 0.0).any()):
            raise VisualFactorCacheError("relative pose factor covariance is not positive definite")
        if any(not str(value) for value in self.visual_hashes):
            raise VisualFactorCacheError("relative pose factor visual hash is empty")

    def load_pair(self, frame_i: int, frame_j: int, visual_sha256: str) -> RelativePoseFactorPacket:
        if frame_j != frame_i + 1 or frame_i < 0 or frame_i >= len(self.frame_i):
            raise VisualFactorCacheError("relative pose factor pair is out of range")
        row = int(frame_i)
        if int(self.frame_i[row]) != frame_i or int(self.frame_j[row]) != frame_j:
            raise VisualFactorCacheError("relative pose factor pair indices differ")
        if str(self.visual_hashes[row]) != str(visual_sha256):
            raise VisualFactorCacheError("relative pose factor visual hash differs")
        return RelativePoseFactorPacket(
            frame_i=frame_i,
            frame_j=frame_j,
            measurement_CiCj=self.measurements[row].clone(),
            covariance=self.covariances[row].clone(),
            visual_sha256=str(self.visual_hashes[row]),
            num_points=int(self.num_points[row]),
            num_inliers=int(self.num_inliers[row]),
            mean_mahalanobis_sq=float(self.mean_mahalanobis_sq[row]),
        )
