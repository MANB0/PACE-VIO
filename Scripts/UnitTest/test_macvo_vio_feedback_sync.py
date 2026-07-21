from types import SimpleNamespace

import torch

from Odometry.MACVO import MACVO


class _Frames:
    def __init__(self):
        self.data = {
            "imu_vio_velocity_world": torch.tensor(
                [[0.0, 0.0, 0.0], [1.25, -0.50, 0.75]],
                dtype=torch.float32,
            ),
            "imu_vio_acc_bias": torch.tensor(
                [[0.0, 0.0, 0.0], [0.01, -0.02, 0.03]],
                dtype=torch.float32,
            ),
            "imu_vio_gyro_bias": torch.tensor(
                [[0.0, 0.0, 0.0], [-0.001, 0.002, -0.003]],
                dtype=torch.float32,
            ),
        }

    def __len__(self):
        return 2


def _macvo_shell(mode: str):
    odom = object.__new__(MACVO)
    odom.prev_keyframe = (None, 1, None)
    odom.Optimizer = SimpleNamespace(config=SimpleNamespace(imu_factor_mode=mode))
    odom.graph = SimpleNamespace(frames=_Frames())
    odom.imu_vio_velocity_feedback_enable = True
    odom.imu_vio_bias_feedback_enable = True
    odom._imu_vel_w = torch.zeros(3, dtype=torch.float32)
    odom._imu_acc_bias = torch.zeros(3, dtype=torch.float32)
    odom._imu_gyro_bias = torch.zeros(3, dtype=torch.float32)
    return odom


def test_preintegrated_vio_syncs_optimized_velocity_and_bias_state():
    odom = _macvo_shell("preintegrated_vio")

    odom._sync_optimized_vio_velocity_from_map()

    assert torch.allclose(odom._imu_vel_w, torch.tensor([1.25, -0.50, 0.75]))
    assert torch.allclose(odom._imu_acc_bias, torch.tensor([0.01, -0.02, 0.03]))
    assert torch.allclose(odom._imu_gyro_bias, torch.tensor([-0.001, 0.002, -0.003]))


def test_local_inertial_ba_syncs_optimized_velocity_and_bias_state():
    odom = _macvo_shell("local_inertial_ba")

    odom._sync_optimized_vio_velocity_from_map()

    assert torch.allclose(odom._imu_vel_w, torch.tensor([1.25, -0.50, 0.75]))
    assert torch.allclose(odom._imu_acc_bias, torch.tensor([0.01, -0.02, 0.03]))
    assert torch.allclose(odom._imu_gyro_bias, torch.tensor([-0.001, 0.002, -0.003]))


def test_legacy_pose_prior_does_not_sync_vio_state():
    odom = _macvo_shell("legacy_pose_prior")

    odom._sync_optimized_vio_velocity_from_map()

    assert torch.allclose(odom._imu_vel_w, torch.zeros(3))
    assert torch.allclose(odom._imu_acc_bias, torch.zeros(3))
    assert torch.allclose(odom._imu_gyro_bias, torch.zeros(3))

