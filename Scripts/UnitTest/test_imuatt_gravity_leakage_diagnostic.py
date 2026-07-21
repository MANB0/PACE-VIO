from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np


WORKDIR = Path("/home/admin1/macvo-dev")
if str(WORKDIR) not in sys.path:
    sys.path.insert(0, str(WORKDIR))

from Scripts.analyze_imuatt_gravity_leakage import (
    accumulate_constant_interval_acceleration,
    gravity_leakage_world,
    query_imu_interval,
)


def rot_x(angle_rad: float) -> np.ndarray:
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def test_gravity_leakage_is_zero_for_correct_attitude() -> None:
    rotation = rot_x(math.radians(17.0))
    leakage_body, leakage_world = gravity_leakage_world(
        rotation_est=rotation,
        rotation_gt=rotation,
        gravity_world=np.array([0.0, 0.0, 9.8]),
    )
    np.testing.assert_allclose(leakage_body, 0.0, atol=1e-12)
    np.testing.assert_allclose(leakage_world, 0.0, atol=1e-12)


def test_one_degree_tilt_has_expected_gravity_leakage_norm() -> None:
    angle = math.radians(1.0)
    _, leakage_world = gravity_leakage_world(
        rotation_est=rot_x(angle),
        rotation_gt=np.eye(3),
        gravity_world=np.array([0.0, 0.0, 9.8]),
    )
    expected = 2.0 * 9.8 * math.sin(angle / 2.0)
    assert math.isclose(np.linalg.norm(leakage_world), expected, rel_tol=1e-12)


def test_constant_leakage_accumulates_velocity_and_position() -> None:
    velocity, position = accumulate_constant_interval_acceleration(
        velocity=np.array([1.0, 0.0, 0.0]),
        position=np.array([0.0, 0.0, 0.0]),
        acceleration=np.array([0.0, 2.0, 0.0]),
        dt=0.5,
    )
    np.testing.assert_allclose(velocity, [1.0, 1.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(position, [0.5, 0.25, 0.0], atol=1e-12)


def test_imu_interval_includes_interpolated_endpoints() -> None:
    timestamps = np.array([0, 10, 20], dtype=np.int64)
    values = np.array([[0.0], [10.0], [20.0]])
    out_t, out_values = query_imu_interval(timestamps, values, 5, 15)
    np.testing.assert_array_equal(out_t, [5, 10, 15])
    np.testing.assert_allclose(out_values[:, 0], [5.0, 10.0, 15.0], atol=1e-12)
