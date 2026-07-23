import pypose as pp
import pytest
import torch

from Utility.T2FactorPacket import T2FactorPacket
from Utility.T2ISAM2Backend import IncrementalT2ISAM2Backend
from Module.Optimization.TwoFramePGO.Optimizer import TwoFrame_PGO
from Utility.TwoStateVIO import (
    ImuPreintegrationFactor,
    LinearizedUVDPoseFactor,
    NavigationState,
)


EXTRINSIC_CI = pp.from_matrix(torch.tensor([[
    [1.0, 0.0, 0.0, -0.417],
    [0.0, -1.0, 0.0, 0.180],
    [0.0, 0.0, -1.0, 0.095],
    [0.0, 0.0, 0.0, 1.0],
]], dtype=torch.float64), pp.SE3_type)


def _state(x: float) -> NavigationState:
    pose_WC = pp.identity_SE3(1, dtype=torch.float64)
    pose_WC.tensor()[0, 0] = x
    pose = (pose_WC @ EXTRINSIC_CI).tensor()
    return NavigationState(
        pose_WB=pose,
        velocity_W=torch.tensor([0.1, 0.0, 0.0], dtype=torch.float64),
        acc_bias=torch.zeros(3, dtype=torch.float64),
        gyro_bias=torch.zeros(3, dtype=torch.float64),
    )


def _packet(
    frame_i: int,
    frame_j: int,
    x_i: float,
    x_j: float,
    *,
    state_i: NavigationState | None = None,
) -> T2FactorPacket:
    extrinsic = EXTRINSIC_CI.tensor()
    covariance = torch.eye(9, dtype=torch.float64) * 1.0e-3
    bias_covariance = torch.eye(6, dtype=torch.float64) * 1.0e-6
    return T2FactorPacket.create(
        frame_i=frame_i,
        frame_j=frame_j,
        state_i_initial=_state(x_i) if state_i is None else state_i,
        state_j_initial=_state(x_j),
        imu=ImuPreintegrationFactor(
            delta_rotation=torch.zeros(3, dtype=torch.float64),
            delta_velocity=torch.zeros(3, dtype=torch.float64),
            delta_position=torch.tensor([0.01, 0.0, 0.0], dtype=torch.float64),
            covariance=covariance,
            dt=0.1,
            bias_jacobian=torch.zeros(9, 6, dtype=torch.float64),
            linearized_acc_bias=torch.zeros(3, dtype=torch.float64),
            linearized_gyro_bias=torch.zeros(3, dtype=torch.float64),
            bias_rw_covariance=bias_covariance,
            gravity_world=torch.zeros(3, dtype=torch.float64),
            gravity_handling="residual",
        ),
        visual=LinearizedUVDPoseFactor(
            reference_relative_CjCi=pp.se3(
                torch.tensor([[-0.01, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float64)
            ).Exp().tensor(),
            sqrt_information=torch.eye(6, dtype=torch.float64) * 10.0,
            residual_offset=torch.zeros(6, dtype=torch.float64),
            extrinsic_CI=extrinsic,
            marginal_mode="full",
        ),
        extrinsic_CI=extrinsic,
    )


def _backend() -> IncrementalT2ISAM2Backend:
    return IncrementalT2ISAM2Backend(
        initial_prior_std={
            "pose_translation_std": 1.0e-5,
            "pose_rotation_std": 1.0e-5,
            "velocity_std": 0.05,
            "acc_bias_std": 0.2,
            "gyro_bias_std": 0.02,
        }
    )


def test_incremental_backend_consumes_packets_and_revises_history():
    backend = _backend()
    first = backend.consume(_packet(90, 91, 0.0, 0.01))
    first_camera = pp.SE3(first.state.pose_WB) @ EXTRINSIC_CI.Inv()
    second = backend.consume(_packet(
        91,
        92,
        first_camera.translation()[0, 0].item(),
        0.02,
        state_i=first.state,
    ))
    history = backend.history()

    assert first.frame_idx == 91
    assert second.frame_idx == 92
    assert backend.state_count == 3
    assert [frame for frame, _ in history] == [90, 91, 92]
    assert first.update_ms >= 0.0 and second.update_ms >= 0.0
    assert torch.isfinite(second.state.pose_WB).all()
    assert second.initial_pose_mismatch_norm < 1.0e-9


def test_incremental_backend_rejects_discontinuous_packet():
    backend = _backend()
    backend.consume(_packet(90, 91, 0.0, 0.01))
    with pytest.raises(ValueError, match="discontinuity"):
        backend.consume(_packet(92, 93, 0.02, 0.03))


def test_optimizer_finalize_returns_one_complete_isam2_history_snapshot():
    backend = _backend()
    first_packet = _packet(90, 91, 0.0, 0.01)
    first = backend.consume(first_packet)
    first_camera = pp.SE3(first.state.pose_WB) @ EXTRINSIC_CI.Inv()
    second_packet = _packet(
        91,
        92,
        first_camera.translation()[0, 0].item(),
        0.02,
        state_i=first.state,
    )
    backend.consume(second_packet)
    context = {
        "two_state_backend_name": "isam2",
        "two_state_isam2_backend": backend,
        "two_state_last_factor_packet": second_packet,
    }

    updated_context, output = TwoFrame_PGO._finalize_context(context)

    assert updated_context is context
    assert output is not None
    assert output.two_state_solver_convergence_reason == "isam2_final_history_snapshot"
    assert output.local_ba_writeback == "all_isam2_history"
    assert output.isam2_history_revision is True
    assert output.isam2_state_count == 3
    assert output.window_frame_indices.tolist() == [90, 91, 92]
    assert output.window_motions.shape == (3, 7)
    assert output.window_velocity_world.shape == (3, 3)
    assert output.window_acc_bias.shape == (3, 3)
    assert output.window_gyro_bias.shape == (3, 3)
    assert output.from_idx.tolist() == [91]
    assert output.frame_idx.tolist() == [92]
    assert backend.state_count == 3
    assert torch.allclose(output.motion, output.window_motions[-1:])
