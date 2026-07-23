import numpy as np
import pypose as pp
import pytest
import torch

from DataLoader.Dataset.GeneralStereoIMU import _load_t_ci
from Module.IMUPreintegration import preintegrate_imu_local_frame
from Utility.IMUKinematics import (
    estimate_static_imu_initialization,
    vio_preintegrated_imu_residual,
)
from Utility.PoseFrame import convert_pose_frame, convert_pose_world_frame_only
from Utility.TwoStateVIO import NavigationState, _camera_relative_CjCi


T_CI_MATRIX = [
    [1.0, 0.0, 0.0, -0.417],
    [0.0, -1.0, 0.0, 0.180],
    [0.0, 0.0, -1.0, 0.095],
    [0.0, 0.0, 0.0, 1.0],
]


def _state(pose: pp.LieTensor) -> NavigationState:
    return NavigationState(
        pose_WB=pose.tensor().double(),
        velocity_W=torch.zeros(3, dtype=torch.float64),
        acc_bias=torch.zeros(3, dtype=torch.float64),
        gyro_bias=torch.zeros(3, dtype=torch.float64),
    )


def test_t_ci_loader_accepts_full_transform_and_rejects_legacy_or_invalid_inputs():
    extrinsic = _load_t_ci({"T_CI": T_CI_MATRIX})
    assert torch.allclose(
        extrinsic.matrix().reshape(4, 4),
        torch.tensor(T_CI_MATRIX, dtype=torch.float64),
        atol=1e-10,
        rtol=0.0,
    )

    with pytest.raises(ValueError, match="T_CI is required"):
        _load_t_ci({"T_body_imu": {}})
    with pytest.raises(ValueError, match="must contain only T_CI"):
        _load_t_ci({"T_CI": T_CI_MATRIX, "T_body_imu": {}})
    with pytest.raises(ValueError, match="shape 4x4"):
        _load_t_ci({"T_CI": torch.eye(3).tolist()})
    improper = torch.eye(4, dtype=torch.float64)
    improper[2, 2] = -1.0
    with pytest.raises(ValueError, match="proper"):
        _load_t_ci({"T_CI": improper.tolist()})


def test_full_t_ci_round_trip_recovers_camera_relative_pose():
    extrinsic = _load_t_ci({"T_CI": T_CI_MATRIX}).double()
    pose_WCi = pp.se3(torch.tensor(
        [[0.4, -0.2, 0.1, 0.07, -0.03, 0.12]], dtype=torch.float64
    )).Exp()
    pose_WCj = pose_WCi @ pp.se3(torch.tensor(
        [[0.15, -0.04, 0.02, -0.02, 0.05, 0.08]], dtype=torch.float64
    )).Exp()

    state_i = _state(pose_WCi @ extrinsic)
    state_j = _state(pose_WCj @ extrinsic)
    recovered = _camera_relative_CjCi(state_i, state_j, extrinsic.tensor())
    expected = pose_WCj.Inv() @ pose_WCi
    error = (expected.Inv() @ recovered).Log().tensor().reshape(6)
    assert float(error.abs().max().item()) < 1e-10


def test_raw_flu_static_preintegration_closes_with_ned_gravity():
    rate_hz = 100
    count = 101
    time_ns = torch.arange(count, dtype=torch.long) * int(1e9 / rate_hz)
    acc_flu = torch.tensor([0.0, 0.0, 9.8], dtype=torch.float64).repeat(count, 1)
    gyro_flu = torch.zeros(count, 3, dtype=torch.float64)
    preint = preintegrate_imu_local_frame(
        time_ns,
        acc_flu,
        gyro_flu,
        sigma_acc=0.01,
        sigma_gyro=0.001,
    )

    pose_WI = _load_t_ci({"T_CI": T_CI_MATRIX}).double()
    residual = vio_preintegrated_imu_residual(
        pose_WI,
        pose_WI,
        torch.zeros(3, dtype=torch.float64),
        torch.zeros(3, dtype=torch.float64),
        preint.delta_R.double(),
        preint.delta_v.double(),
        preint.delta_p.double(),
        preint.dt_total,
        gravity_world=torch.tensor([0.0, 0.0, 9.8], dtype=torch.float64),
        gravity_handling="residual",
    )
    # Production preintegration stores deltas as float32; this is its expected
    # numerical floor, not a model/sign mismatch.
    assert float(residual.abs().max().item()) < 5e-7


def test_static_initialization_uses_raw_flu_attitude_and_zero_bias():
    count = 301
    time_ns = torch.arange(count, dtype=torch.long) * 10_000_000
    acc_flu = torch.tensor([0.0, 0.0, 9.8]).repeat(count, 1)
    gyro_flu = torch.zeros(count, 3)
    initial_R_WI = _load_t_ci({"T_CI": T_CI_MATRIX}).rotation().float()
    result = estimate_static_imu_initialization(
        time_ns,
        acc_flu,
        gyro_flu,
        initial_R_WI,
        9.8,
        min_duration_s=3.0,
        gyro_mean_norm_max=0.01,
        gyro_std_max=0.01,
        acc_norm_error_max=0.1,
        acc_std_max=0.1,
    )
    assert result.stationary
    assert float(result.acc_bias.abs().max().item()) < 1e-6
    assert float(result.gyro_bias.abs().max().item()) < 1e-12
    attitude_error = (initial_R_WI.Inv() @ result.body_to_world).Log().tensor()
    assert float(attitude_error.abs().max().item()) < 1e-6


def test_world_only_conversion_does_not_rotate_raw_imu_axes_twice():
    pose_WI_ned = _load_t_ci({"T_CI": T_CI_MATRIX}).tensor().numpy()
    imu_nwu = convert_pose_world_frame_only(pose_WI_ned, "NED", "NWU")
    camera_nwu = convert_pose_frame(
        pp.identity_SE3(1, dtype=torch.float64).tensor().numpy(), "NED", "NWU"
    )

    # At the initial aligned pose, both reported body/IMU orientations are identity.
    assert np.max(np.abs(pp.SE3(imu_nwu).rotation().Log().tensor().numpy())) < 1e-10
    assert np.max(np.abs(pp.SE3(camera_nwu).rotation().Log().tensor().numpy())) < 1e-10
    assert np.allclose(imu_nwu[0, :3], [-0.417, -0.180, -0.095], atol=1e-10)
