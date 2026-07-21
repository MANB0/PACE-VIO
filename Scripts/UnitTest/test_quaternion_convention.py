import math

import pytest
import torch

from Utility.QuaternionConvention import nwu_xyzw_quaternion_to_internal_so3


def test_holoocean_xyzw_quaternion_identity_remains_identity_in_internal_ned():
    rotation = nwu_xyzw_quaternion_to_internal_so3(
        [0.0, 0.0, 0.0, 1.0],
        internal_world_frame="NED",
    )

    assert torch.allclose(
        rotation.matrix().reshape(3, 3).float(),
        torch.eye(3),
        atol=1e-6,
    )


def test_holoocean_xyzw_quaternion_yaw_rotation_keeps_angle_in_internal_ned():
    half_yaw = math.pi / 4.0
    rotation = nwu_xyzw_quaternion_to_internal_so3(
        [0.0, 0.0, math.sin(half_yaw), math.cos(half_yaw)],
        internal_world_frame="NED",
    )

    angle = float(rotation.Log().tensor().norm().item())

    assert angle == pytest.approx(math.pi / 2.0, abs=1e-6)
