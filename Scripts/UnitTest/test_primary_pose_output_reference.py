from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pypose as pp
import torch

from Odometry.Interface import IOdometry


class _Frames:
    def __init__(self) -> None:
        camera_poses = pp.SE3(torch.tensor(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                [2.0, 0.0, 0.0, 0.0, 0.0, 2.0**-0.5, 2.0**-0.5],
            ],
            dtype=torch.float64,
        )).tensor()
        extrinsic = pp.SE3(torch.tensor(
            [[1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]],
            dtype=torch.float64,
        )).tensor().repeat(2, 1)
        identity = pp.identity_SE3(2, dtype=torch.float64).tensor()
        self.data = {
            "pose": SimpleNamespace(tensor=camera_poses),
            "T_BS": SimpleNamespace(tensor=identity),
            "time_ns": SimpleNamespace(tensor=torch.tensor([0, 1], dtype=torch.int64)),
            "imu_vio_sensor_T_imu": SimpleNamespace(tensor=extrinsic),
        }


class _Map:
    def __init__(self) -> None:
        self.frames = _Frames()

    @staticmethod
    def serialize() -> dict:
        return {}


class _Odometry(IOdometry):
    def __init__(self) -> None:
        super().__init__()
        self._map = _Map()

    def _run_sequence(self, sequence, on_frame_finished):
        return [], []

    def get_map(self):
        return self._map

    def run(self, frame) -> None:
        return None


class _Sandbox:
    def __init__(self, root: Path) -> None:
        self.folder = root

    def path(self, name: str) -> Path:
        return self.folder / name


def _read_positions(path: Path) -> np.ndarray:
    with path.open("r", newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    return np.asarray(
        [[float(row["tx"]), float(row["ty"]), float(row["tz"])] for row in rows]
    )


def test_primary_pose_output_is_the_imu_origin(tmp_path: Path) -> None:
    sequence = SimpleNamespace(pose_output_frame="NED")
    odometry = _Odometry()

    odometry.receive_frames(sequence, _Sandbox(tmp_path))

    primary = _read_positions(tmp_path / "poses.csv")
    imu_alias = _read_positions(tmp_path / "poses_imu.csv")
    camera = _read_positions(tmp_path / "poses_camera.csv")
    contract = json.loads(
        (tmp_path / "pose_reference_points.json").read_text(encoding="utf-8")
    )

    np.testing.assert_allclose(primary, imu_alias, atol=1.0e-12)
    np.testing.assert_allclose(
        camera,
        np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        primary,
        np.array([[1.0, 0.0, 0.0], [2.0, 1.0, 0.0]]),
        atol=1.0e-12,
    )
    assert contract["canonical_trajectory"] == "poses.csv"
    assert contract["poses.csv"].startswith("IMU origin")
