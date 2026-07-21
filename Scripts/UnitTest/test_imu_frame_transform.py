import importlib.util
from pathlib import Path

import pypose as pp
import torch

from DataLoader.Dataset.GeneralStereoIMU import _flu_to_ned_se3
from Utility.IMUKinematics import gravity_for_world_frame

_PREINT_SPEC = importlib.util.spec_from_file_location(
    "imu_preintegration_under_test",
    Path("Module/IMUPreintegration.py"),
)
_PREINT_MODULE = importlib.util.module_from_spec(_PREINT_SPEC)
assert _PREINT_SPEC is not None and _PREINT_SPEC.loader is not None
_PREINT_SPEC.loader.exec_module(_PREINT_MODULE)
preintegrate_imu = _PREINT_MODULE.preintegrate_imu


def test_flu_to_ned_transform_preserves_forward_and_flips_left_up():
    transform = _flu_to_ned_se3()
    rotation = transform.rotation().matrix().reshape(3, 3).float()

    flu_vector = torch.tensor([1.0, 2.0, 3.0])
    ned_vector = rotation @ flu_vector

    assert torch.allclose(
        ned_vector,
        torch.tensor([1.0, -2.0, -3.0]),
        atol=1e-6,
    )


def test_transform_imu_samples_to_internal_frame_applies_tbs_to_acc_and_gyro_without_mutation():
    from Utility.IMUKinematics import transform_imu_samples_to_internal_frame

    transform = _flu_to_ned_se3()
    acc_flu = torch.tensor([[1.0, 2.0, 3.0], [4.0, -5.0, 6.0]])
    gyro_flu = torch.tensor([[0.1, 0.2, 0.3], [-0.4, 0.5, -0.6]])
    acc_original = acc_flu.clone()
    gyro_original = gyro_flu.clone()

    acc_ned, gyro_ned = transform_imu_samples_to_internal_frame(
        acc_flu,
        gyro_flu,
        transform,
    )

    assert torch.allclose(acc_flu, acc_original)
    assert torch.allclose(gyro_flu, gyro_original)
    assert torch.allclose(acc_ned, torch.tensor([[1.0, -2.0, -3.0], [4.0, 5.0, -6.0]]), atol=1e-6)
    assert torch.allclose(gyro_ned, torch.tensor([[0.1, -0.2, -0.3], [-0.4, -0.5, 0.6]]), atol=1e-6)


def test_transform_imu_samples_to_internal_frame_uses_inverse_tbs_rotation_for_sensor_measurements():
    from Utility.IMUKinematics import transform_imu_samples_to_internal_frame

    T_BS_matrix = torch.eye(4)
    T_BS_matrix[:3, :3] = torch.tensor([
        [0.0, -1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    T_BS = pp.from_matrix(T_BS_matrix.reshape(1, 4, 4), pp.SE3_type)

    acc_sensor = torch.tensor([[0.0, 1.0, 0.0]])
    gyro_sensor = torch.tensor([[1.0, 0.0, 0.0]])
    acc_body, gyro_body = transform_imu_samples_to_internal_frame(acc_sensor, gyro_sensor, T_BS)

    assert torch.allclose(acc_body, torch.tensor([[1.0, 0.0, 0.0]]), atol=1e-6)
    assert torch.allclose(gyro_body, torch.tensor([[0.0, -1.0, 0.0]]), atol=1e-6)


def test_holoocean_stationary_flu_gravity_cancels_after_internal_transform():
    from Utility.IMUKinematics import transform_imu_samples_to_internal_frame

    time_ns = torch.tensor([0, 10_000_000, 20_000_000], dtype=torch.long)
    acc_flu = torch.tensor([[0.0, 0.0, 9.8]] * 3)
    gyro_flu = torch.zeros((3, 3))
    acc_ned, gyro_ned = transform_imu_samples_to_internal_frame(
        acc_flu,
        gyro_flu,
        _flu_to_ned_se3(),
    )

    result = preintegrate_imu(
        time_ns=time_ns,
        acc=acc_ned,
        gyro=gyro_ned,
        R0_world=None,
        gravity=gravity_for_world_frame(9.8, "NED"),
    )

    assert torch.allclose(acc_ned, torch.tensor([[0.0, 0.0, -9.8]] * 3), atol=1e-6)
    assert torch.allclose(result.delta_v, torch.zeros(3), atol=1e-6)
    assert torch.allclose(result.delta_p, torch.zeros(3), atol=1e-8)
    assert result.dt_total == 0.02
