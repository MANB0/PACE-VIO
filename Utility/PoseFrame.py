from __future__ import annotations

import csv

import numpy as np
import pypose as pp
import torch


POSE_CSV_HEADER = ("timestamp_ns", "tx", "ty", "tz", "qx", "qy", "qz", "qw")


def se3_ned_to_nwu(poses: np.ndarray) -> np.ndarray:
    """Convert SE3 coordinates between MACVO internal NED and HoloOcean NWU."""
    converted = np.asarray(poses).copy()
    if converted.ndim != 2 or converted.shape[1] != 7:
        raise ValueError(f"Expected Nx7 SE3 poses, got shape={converted.shape}")
    converted[:, 1] *= -1.0
    converted[:, 2] *= -1.0
    converted[:, 4] *= -1.0
    converted[:, 5] *= -1.0
    return converted


def convert_pose_frame(poses: np.ndarray, source_frame: str, target_frame: str) -> np.ndarray:
    pose_array = np.asarray(poses)
    if pose_array.ndim != 2 or pose_array.shape[1] != 7:
        raise ValueError(f"Expected Nx7 SE3 poses, got shape={pose_array.shape}")
    source = source_frame.strip().upper()
    target = target_frame.strip().upper()
    if source == target:
        return pose_array.copy()
    if {source, target} == {"NED", "NWU"}:
        return se3_ned_to_nwu(pose_array)
    raise ValueError(f"Unsupported pose coordinate conversion: {source_frame} -> {target_frame}")


def convert_pose_world_frame_only(
    poses: np.ndarray,
    source_world_frame: str,
    target_world_frame: str,
) -> np.ndarray:
    """Change only the world basis while preserving the pose's local frame.

    This is required for the VIO state ``T_WI``: its local frame is the raw IMU
    FLU frame even though its world coordinates use internal NED. Conjugating
    both sides, as :func:`convert_pose_frame` does, would incorrectly rotate the
    already-defined raw IMU axes a second time.
    """
    values = np.asarray(poses, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 7:
        raise ValueError(f"Expected Nx7 SE3 poses, got shape={values.shape}")
    source = source_world_frame.strip().upper()
    target = target_world_frame.strip().upper()
    if source == target:
        return values.copy()
    if {source, target} != {"NED", "NWU"}:
        raise ValueError(
            f"Unsupported world-only pose conversion: {source_world_frame} -> {target_world_frame}"
        )

    # NED <-> NWU is its own inverse: R_x(pi), represented as qx=1.
    world_change = pp.SE3(torch.tensor(
        [[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]],
        dtype=torch.float64,
    ))
    return (world_change @ pp.SE3(torch.from_numpy(values))).tensor().cpu().numpy()


def write_timed_se3_csv(path, time_ns: np.ndarray, poses: np.ndarray) -> None:
    """Write Nx(timestamp_ns + SE3) without converting timestamps to float."""
    timestamps = np.asarray(time_ns).reshape(-1)
    poses = np.asarray(poses)
    assert poses.ndim == 2 and poses.shape[1] == 7, f"Expected Nx7 poses, got {poses.shape}"
    assert timestamps.shape[0] == poses.shape[0], (
        f"Timestamp/pose length mismatch: {timestamps.shape[0]} vs {poses.shape[0]}"
    )

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(POSE_CSV_HEADER)
        for timestamp, pose in zip(timestamps, poses):
            writer.writerow([str(int(timestamp)), *[f"{float(value):.17g}" for value in pose]])
