import numpy as np
import pypose as pp
import pytest
import torch

from Utility.TrajectoryReference import (
    camera_nwu_poses_to_imu_nwu,
    camera_velocity_to_imu_velocity_nwu,
    compose_camera_to_imu_poses,
    constant_camera_T_imu,
    rebase_pose_positions,
    translate_pose_reference_point,
)


def test_compose_camera_to_imu_pose_uses_rotated_lever_arm():
    dtype = torch.float64
    identity = pp.identity_SE3(1, dtype=dtype)
    quarter_turn = pp.SE3(
        torch.cat(
            [
                torch.zeros((1, 3), dtype=dtype),
                pp.so3(torch.tensor([[0.0, 0.0, np.pi / 2]], dtype=dtype)).Exp().tensor(),
            ],
            dim=-1,
        )
    )
    camera = torch.cat([identity.tensor(), quarter_turn.tensor()], dim=0).numpy()
    camera_T_imu = np.asarray([-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])

    imu = compose_camera_to_imu_poses(camera, camera_T_imu)

    np.testing.assert_allclose(imu[0, :3], [-1.0, 0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(imu[1, :3], [0.0, -1.0, 0.0], atol=1e-12)


def test_nwu_pose_conversion_uses_internal_ned_extrinsic_once():
    camera_nwu = np.asarray([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
    camera_T_imu_ned = np.asarray([-0.4, 0.2, 0.1, 0.0, 0.0, 0.0, 1.0])

    imu_nwu = camera_nwu_poses_to_imu_nwu(camera_nwu, camera_T_imu_ned)

    np.testing.assert_allclose(imu_nwu[0, :3], [-0.4, -0.2, -0.1], atol=1e-12)


def test_velocity_conversion_applies_omega_cross_lever_arm():
    velocity_camera = np.asarray([[2.0, 3.0, 0.0]])
    omega_body = np.asarray([[0.0, 0.0, 1.0]])
    quaternion = np.asarray([[0.0, 0.0, 0.0, 1.0]])
    # Internal NED [1, 0, 0] is the same body-NWU lever arm.
    camera_T_imu_ned = np.asarray([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])

    velocity_imu = camera_velocity_to_imu_velocity_nwu(
        velocity_camera,
        omega_body,
        quaternion,
        camera_T_imu_ned,
    )

    np.testing.assert_allclose(velocity_imu, [[2.0, 4.0, 0.0]], atol=1e-12)


def test_body_origin_pose_moves_to_imu_origin_with_body_lever_arm():
    poses = np.asarray(
        [
            [1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0],
            [1.0, 2.0, 3.0, 0.0, 0.0, np.sqrt(0.5), np.sqrt(0.5)],
        ]
    )
    body_to_imu = np.asarray([-0.1, -0.2, 0.05])

    converted = translate_pose_reference_point(poses, body_to_imu)

    np.testing.assert_allclose(converted[0, :3], [0.9, 1.8, 3.05], atol=1e-12)
    np.testing.assert_allclose(converted[1, :3], [1.2, 1.9, 3.05], atol=1e-12)
    np.testing.assert_allclose(converted[:, 3:7], poses[:, 3:7], atol=1e-12)


def test_rebase_pose_positions_preserves_axes_and_orientation():
    poses = np.asarray(
        [
            [3.0, -2.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            [4.5, 0.0, 0.5, 0.0, 0.0, np.sqrt(0.5), np.sqrt(0.5)],
        ]
    )

    rebased = rebase_pose_positions(poses)

    np.testing.assert_allclose(rebased[0, :3], [0.0, 0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(rebased[1, :3], [1.5, 2.0, -0.5], atol=1e-12)
    np.testing.assert_allclose(rebased[:, 3:7], poses[:, 3:7], atol=1e-12)


def test_constant_extrinsic_rejects_frame_varying_calibration():
    extrinsics = np.asarray(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        ]
    )
    with pytest.raises(ValueError, match="changes across frames"):
        constant_camera_T_imu(extrinsics)
