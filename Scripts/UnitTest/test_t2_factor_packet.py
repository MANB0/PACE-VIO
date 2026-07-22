import numpy as np
import pypose as pp
import pytest
import torch

from Utility.T2FactorPacket import T2FactorPacket
from Utility.TwoStateVIO import (
    ImuPreintegrationFactor,
    LinearizedUVDPoseFactor,
    NavigationState,
    make_diagonal_prior,
)


def _state(translation, velocity) -> NavigationState:
    tangent = torch.tensor(
        [[*translation, 0.02, -0.01, 0.03]], dtype=torch.float64
    )
    return NavigationState(
        pose_WB=pp.se3(tangent).Exp().tensor(),
        velocity_W=torch.tensor(velocity, dtype=torch.float64),
        acc_bias=torch.tensor([0.01, -0.02, 0.03], dtype=torch.float64),
        gyro_bias=torch.tensor([0.001, -0.002, 0.003], dtype=torch.float64),
    )


def _packet() -> T2FactorPacket:
    identity = pp.identity_SE3(1, dtype=torch.float64).tensor()
    imu = ImuPreintegrationFactor(
        delta_rotation=torch.tensor([0.01, -0.02, 0.03], dtype=torch.float64),
        delta_velocity=torch.tensor([0.1, -0.2, 0.3], dtype=torch.float64),
        delta_position=torch.tensor([0.01, -0.02, 0.03], dtype=torch.float64),
        covariance=torch.diag(torch.linspace(1.0e-5, 9.0e-5, 9, dtype=torch.float64)),
        dt=0.1,
        bias_jacobian=torch.arange(54, dtype=torch.float64).reshape(9, 6) * 1.0e-5,
        linearized_acc_bias=torch.zeros(3, dtype=torch.float64),
        linearized_gyro_bias=torch.zeros(3, dtype=torch.float64),
        bias_rw_covariance=torch.diag(
            torch.linspace(1.0e-8, 6.0e-8, 6, dtype=torch.float64)
        ),
        gravity_world=torch.tensor([0.0, 0.0, 9.81], dtype=torch.float64),
        gravity_handling="residual",
    )
    visual = LinearizedUVDPoseFactor(
        reference_relative_CjCi=pp.se3(
            torch.tensor([[0.1, 0.0, 0.0, 0.0, 0.01, 0.0]], dtype=torch.float64)
        ).Exp().tensor(),
        sqrt_information=torch.eye(6, dtype=torch.float64) * 4.0,
        residual_offset=torch.linspace(-0.2, 0.3, 6, dtype=torch.float64),
        extrinsic_CI=identity,
        marginal_mode="full",
    )
    return T2FactorPacket.create(
        frame_i=90,
        frame_j=91,
        state_i_initial=_state([0.0, 0.0, 0.0], [0.1, 0.0, 0.0]),
        state_j_initial=_state([0.01, 0.0, 0.0], [0.1, 0.01, 0.0]),
        imu=imu,
        visual=visual,
        extrinsic_CI=identity,
    )


def test_packet_materializes_backend_neutral_float64_payload():
    packet = _packet()
    payload = packet.incremental_payload()

    assert packet.frame_i == 90 and packet.frame_j == 91
    assert packet.state_i_initial.pose_WB.device.type == "cpu"
    assert packet.state_i_initial.pose_WB.dtype == torch.float64
    assert payload["imu_covariance_pvr"].shape == (9, 9)
    assert payload["visual_sqrt_information"].shape == (6, 6)
    assert np.isfinite(payload["state_j_pose_WB"]).all()
    assert payload["gravity_handling"] == "residual"


def test_packet_builds_existing_two_state_problem_without_rebuilding_factors():
    packet = _packet()
    prior = make_diagonal_prior(
        packet.state_i_initial,
        pose_translation_std=1.0e-5,
        pose_rotation_std=1.0e-5,
        velocity_std=0.05,
        acc_bias_std=0.2,
        gyro_bias_std=0.02,
    )
    problem = packet.to_two_state_problem(
        prior_i=prior,
        device=torch.device("cpu"),
        optimize_acc_bias=True,
        optimize_gyro_bias=True,
    )

    assert torch.equal(problem.imu.covariance, packet.imu.covariance)
    assert torch.equal(
        problem.visual_pose.sqrt_information, packet.visual.sqrt_information
    )
    assert torch.equal(problem.visual_pose.residual_offset, packet.visual.residual_offset)


def test_packet_rejects_mismatched_visual_extrinsic():
    packet = _packet()
    wrong_extrinsic = pp.se3(
        torch.tensor([[0.01, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float64)
    ).Exp().tensor()
    with pytest.raises(ValueError, match="different T_CI"):
        T2FactorPacket.create(
            frame_i=packet.frame_i,
            frame_j=packet.frame_j,
            state_i_initial=packet.state_i_initial,
            state_j_initial=packet.state_j_initial,
            imu=packet.imu,
            visual=packet.visual,
            extrinsic_CI=wrong_extrinsic,
        )


def test_packet_rejects_nonfinite_measurement():
    packet = _packet()
    bad_imu = ImuPreintegrationFactor(
        delta_rotation=packet.imu.delta_rotation,
        delta_velocity=torch.tensor([float("nan"), 0.0, 0.0], dtype=torch.float64),
        delta_position=packet.imu.delta_position,
        covariance=packet.imu.covariance,
        dt=packet.imu.dt,
        bias_jacobian=packet.imu.bias_jacobian,
        linearized_acc_bias=packet.imu.linearized_acc_bias,
        linearized_gyro_bias=packet.imu.linearized_gyro_bias,
        bias_rw_covariance=packet.imu.bias_rw_covariance,
        gravity_world=packet.imu.gravity_world,
        gravity_handling=packet.imu.gravity_handling,
    )
    with pytest.raises(ValueError, match="NaN/Inf"):
        T2FactorPacket.create(
            frame_i=packet.frame_i,
            frame_j=packet.frame_j,
            state_i_initial=packet.state_i_initial,
            state_j_initial=packet.state_j_initial,
            imu=bad_imu,
            visual=packet.visual,
            extrinsic_CI=packet.extrinsic_CI,
        )
