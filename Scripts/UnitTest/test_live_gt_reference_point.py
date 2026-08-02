from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from Utility.LiveDashboard import LiveDashboard


def _write_dataset(root: Path, reference_point: str) -> None:
    root.mkdir(parents=True, exist_ok=True)
    metadata = {
        "extrinsics": {
            "T_CI": [
                [1.0, 0.0, 0.0, 1.0],
                [0.0, -1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        },
        "ground_truth": {"reference_point": reference_point},
    }
    (root / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    half_sqrt = np.sqrt(0.5)
    rows = [
        [0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        [1, 2.0, 0.0, 0.0, 0.0, 0.0, half_sqrt, half_sqrt],
    ]
    with (root / "ref_pose.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["timestamp", "x", "y", "z", "qx", "qy", "qz", "qw"])
        writer.writerows(rows)


def _load_positions(root: Path) -> np.ndarray:
    dashboard = LiveDashboard.__new__(LiveDashboard)
    dashboard.dataset_root = root
    dashboard._gt = {}
    dashboard._load_gt()
    return np.asarray([dashboard._gt[key] for key in sorted(dashboard._gt)])


def test_imu_center_ref_pose_is_not_shifted_twice(tmp_path: Path) -> None:
    root = tmp_path / "imu_center"
    _write_dataset(root, "IMUSocket")

    positions = _load_positions(root)

    np.testing.assert_allclose(
        positions,
        np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
        atol=1.0e-12,
    )


def test_legacy_camera_ref_pose_receives_lever_shift(tmp_path: Path) -> None:
    root = tmp_path / "camera_center"
    _write_dataset(root, "CameraLeftSocket")

    positions = _load_positions(root)

    np.testing.assert_allclose(
        positions,
        np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 0.0]]),
        atol=1.0e-12,
    )
