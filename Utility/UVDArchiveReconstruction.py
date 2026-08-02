"""Reconstruct archived compressed-UVD normal equations from raw MACVO data.

Older realtime recordings retained the point-level UVD observations but did
not persist the online six-dimensional compression.  This module recreates
the same local quadratic factor for offline audits.  It does not run a neural
network and it does not alter the production optimizer.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pypose as pp
import torch

from Utility.TwoStateVIO import UVDFactor, linearize_uvd_relative_pose_factor


def _read_camera_poses(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open("r", newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        required = ("timestamp_ns", "tx", "ty", "tz", "qx", "qy", "qz", "qw")
        if reader.fieldnames is None or any(name not in reader.fieldnames for name in required):
            raise ValueError(f"raw MACVO pose CSV lacks required columns: {path}")
        timestamps: list[int] = []
        poses: list[list[float]] = []
        for row in reader:
            timestamps.append(int(row["timestamp_ns"]))
            poses.append([float(row[name]) for name in required[1:]])
    return np.asarray(timestamps, dtype=np.int64), np.asarray(poses, dtype=np.float64)


def _cached_result_is_compatible(
    cache_path: Path,
    timestamps_ns: np.ndarray,
    end_frame: int,
) -> bool:
    if not cache_path.is_file():
        return False
    try:
        with np.load(cache_path, allow_pickle=False) as cache:
            return bool(
                int(cache["schema_version"]) == 1
                and int(cache["end_frame"]) >= end_frame
                and cache["reference_CjCi"].shape[0] > end_frame
                and np.array_equal(cache["timestamps_ns"][: end_frame + 1], timestamps_ns)
                and np.isfinite(cache["hessian"][: end_frame + 1]).all()
                and np.isfinite(cache["gradient"][: end_frame + 1]).all()
            )
    except (KeyError, OSError, ValueError):
        return False


def reconstruct_uvd_normal_equations(
    tensor_map_path: str | Path,
    raw_camera_pose_csv: str | Path,
    cache_path: str | Path,
    *,
    end_frame: int,
    huber_delta: float = 0.1,
    normal_eigenvalue_floor: float = 1.0e-10,
    progress_every: int = 250,
) -> Path:
    """Recreate full 6D UVD ``H`` and ``g`` through ``end_frame`` inclusive."""

    tensor_path = Path(tensor_map_path).expanduser().resolve()
    raw_path = Path(raw_camera_pose_csv).expanduser().resolve()
    output = Path(cache_path).expanduser().resolve()
    raw_timestamps, raw_poses = _read_camera_poses(raw_path)

    with np.load(tensor_path, allow_pickle=False) as data:
        frame_timestamps = np.asarray(data["frames//time_ns"], dtype=np.int64)
        frame_count = int(frame_timestamps.shape[0])
        if not (1 <= end_frame < frame_count):
            raise ValueError(f"invalid UVD reconstruction end frame {end_frame}/{frame_count - 1}")
        if raw_poses.shape[0] < end_frame + 1:
            raise ValueError("raw MACVO trajectory is shorter than the requested factor range")
        if not np.array_equal(raw_timestamps[: end_frame + 1], frame_timestamps[: end_frame + 1]):
            raise ValueError("raw MACVO and tensor-map timestamps are not exactly aligned")
        timestamps = frame_timestamps[: end_frame + 1].copy()
        if _cached_result_is_compatible(output, timestamps, end_frame):
            print(f"using cached UVD normal equations: {output}", flush=True)
            return output

        ranges = np.asarray(data["edge/frame2match/ranges"])
        match_frame_i = np.asarray(data["edge/match2frame1/mapping"])
        match_frame_j = np.asarray(data["edge/match2frame2/mapping"])
        match_to_point = np.asarray(data["edge/match2point/mapping"])
        points = np.asarray(data["points//pos_Tc"])
        target_uv = np.asarray(data["match//pixel2_uv"])
        target_disp = np.asarray(data["match//pixel2_disp"])
        target_uv_cov = np.asarray(data["match//pixel2_uv_cov"])
        target_disp_cov = np.asarray(data["match//pixel2_disp_cov"])
        intrinsics = np.asarray(data["frames//K"])
        baselines = np.asarray(data["frames//baseline"])

        references = np.zeros((end_frame + 1, 7), dtype=np.float64)
        references[:, 6] = 1.0
        hessians = np.zeros((end_frame + 1, 6, 6), dtype=np.float64)
        gradients = np.zeros((end_frame + 1, 6), dtype=np.float64)
        point_counts = np.zeros(end_frame + 1, dtype=np.int64)
        identity_extrinsic = pp.identity_SE3(1, dtype=torch.float64).tensor()
        camera_poses = pp.SE3(torch.as_tensor(raw_poses[: end_frame + 1], dtype=torch.float64))

        for frame_j in range(1, end_frame + 1):
            frame_i = frame_j - 1
            start = int(ranges[frame_j, 0, 0])
            count = int(ranges[frame_j, 0, 1])
            if start < 0 or count < 3:
                raise ValueError(f"edge {frame_i}->{frame_j} has invalid match range")
            match_indices = np.arange(start, start + count, dtype=np.int64)
            if not (
                np.all(match_frame_i[match_indices] == frame_i)
                and np.all(match_frame_j[match_indices] == frame_j)
            ):
                raise ValueError(f"edge {frame_i}->{frame_j} match direction is inconsistent")
            point_indices = match_to_point[match_indices]
            if np.any(point_indices < 0):
                raise ValueError(f"edge {frame_i}->{frame_j} contains an unmapped point")

            covariance = torch.zeros((count, 3, 3), dtype=torch.float64)
            uv_cov = torch.as_tensor(target_uv_cov[match_indices], dtype=torch.float64)
            covariance[:, 0, 0] = uv_cov[:, 0]
            covariance[:, 1, 1] = uv_cov[:, 1]
            covariance[:, 0, 1] = uv_cov[:, 2]
            covariance[:, 1, 0] = uv_cov[:, 2]
            covariance[:, 2, 2] = torch.as_tensor(
                target_disp_cov[match_indices].reshape(count), dtype=torch.float64
            )
            visual = UVDFactor(
                points_Ci=torch.as_tensor(points[point_indices], dtype=torch.float64),
                target_uv=torch.as_tensor(target_uv[match_indices], dtype=torch.float64),
                target_disparity=torch.as_tensor(target_disp[match_indices], dtype=torch.float64),
                covariance_uvd=covariance,
                intrinsic=torch.as_tensor(intrinsics[frame_j], dtype=torch.float64),
                baseline=float(baselines[frame_j]),
                extrinsic_CI=identity_extrinsic,
                huber_delta=float(huber_delta),
            )
            reference = (camera_poses[frame_i].Inv() @ camera_poses[frame_j]).Inv()
            linearization = linearize_uvd_relative_pose_factor(
                reference.tensor().reshape(1, 7),
                visual,
                marginal_mode="full",
                normal_eigenvalue_floor=float(normal_eigenvalue_floor),
            )
            references[frame_j] = reference.tensor().reshape(7).detach().cpu().numpy()
            hessians[frame_j] = linearization.full_hessian.detach().cpu().numpy()
            gradients[frame_j] = linearization.full_gradient.detach().cpu().numpy()
            point_counts[frame_j] = count
            if progress_every > 0 and (
                frame_j == 1 or frame_j % progress_every == 0 or frame_j == end_frame
            ):
                print(f"reconstructed UVD factor {frame_j}/{end_frame}", flush=True)

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        schema_version=np.asarray(1, dtype=np.int64),
        end_frame=np.asarray(end_frame, dtype=np.int64),
        timestamps_ns=timestamps,
        reference_CjCi=references,
        hessian=hessians,
        gradient=gradients,
        point_count=point_counts,
    )
    return output
