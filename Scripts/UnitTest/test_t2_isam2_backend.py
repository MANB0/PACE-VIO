import pypose as pp
import pytest
import torch

from Utility.PACEFactorPacket import PACEFactorPacket
from Utility.PACEISAM2Backend import IncrementalPACEISAM2Backend
from Utility.T2ISAM2Backend import IncrementalT2ISAM2Backend
from Module.Optimization.TwoFramePGO.Optimizer import TwoFrame_PGO
from Utility.Point import point2pixel_NED
from Utility.TwoStateVIO import (
    ImuPreintegrationFactor,
    LinearizedUVDPoseFactor,
    NavigationState,
    RelativePoseFactor,
    UVDFactor,
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
    visual_mode: str = "compressed_uvd",
) -> PACEFactorPacket:
    extrinsic = EXTRINSIC_CI.tensor()
    covariance = torch.eye(9, dtype=torch.float64) * 1.0e-3
    bias_covariance = torch.eye(6, dtype=torch.float64) * 1.0e-6
    initial_i = _state(x_i) if state_i is None else state_i
    initial_j = _state(x_j)
    if visual_mode == "compressed_uvd":
        visual = LinearizedUVDPoseFactor(
            reference_relative_CjCi=pp.se3(
                torch.tensor([[-0.01, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float64)
            ).Exp().tensor(),
            sqrt_information=torch.eye(6, dtype=torch.float64) * 10.0,
            residual_offset=torch.zeros(6, dtype=torch.float64),
            extrinsic_CI=extrinsic,
            marginal_mode="full",
        )
    elif visual_mode == "relative_pose":
        measurement = pp.SE3(initial_i.pose_WB).Inv() @ pp.SE3(initial_j.pose_WB)
        visual = RelativePoseFactor(
            measurement_BiBj=measurement.tensor(),
            covariance=torch.eye(6, dtype=torch.float64) * 1.0e-3,
            huber_delta=3.0,
        )
    elif visual_mode == "direct_uvd":
        points = torch.tensor(
            [[4.0, -0.4, -0.2], [5.0, 0.2, 0.1], [6.0, 0.5, -0.3]],
            dtype=torch.float64,
        )
        intrinsic = torch.tensor(
            [[320.0, 0.0, 320.0], [0.0, 320.0, 240.0], [0.0, 0.0, 1.0]],
            dtype=torch.float64,
        )
        pose_WCi = pp.SE3(initial_i.pose_WB) @ EXTRINSIC_CI.Inv()
        pose_WCj = pp.SE3(initial_j.pose_WB) @ EXTRINSIC_CI.Inv()
        relative_CjCi = pose_WCj.Inv() @ pose_WCi
        points_j = relative_CjCi.Act(points)
        baseline = 0.225
        visual = UVDFactor(
            points_Ci=points,
            target_uv=point2pixel_NED(points_j, intrinsic),
            target_disparity=(
                intrinsic[0, 0] * baseline / points_j[:, 0:1]
            ),
            covariance_uvd=torch.eye(3, dtype=torch.float64).repeat(3, 1, 1),
            intrinsic=intrinsic,
            baseline=baseline,
            extrinsic_CI=extrinsic,
            huber_delta=0.1,
        )
    else:
        raise ValueError(visual_mode)
    return PACEFactorPacket.create(
        frame_i=frame_i,
        frame_j=frame_j,
        state_i_initial=initial_i,
        state_j_initial=initial_j,
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
        visual=visual,
        extrinsic_CI=extrinsic,
    )


def _backend() -> IncrementalPACEISAM2Backend:
    return IncrementalPACEISAM2Backend(
        initial_prior_std={
            "pose_translation_std": 1.0e-5,
            "pose_rotation_std": 1.0e-5,
            "velocity_std": 0.05,
            "acc_bias_std": 0.2,
            "gyro_bias_std": 0.02,
        }
    )


def test_legacy_t2_backend_name_is_a_compatibility_alias():
    assert IncrementalT2ISAM2Backend is IncrementalPACEISAM2Backend


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


@pytest.mark.parametrize(
    "visual_mode",
    ("relative_pose", "direct_uvd", "compressed_uvd"),
)
def test_incremental_backend_accepts_all_visual_factor_modes(visual_mode: str):
    backend = _backend()
    update = backend.consume(
        _packet(90, 91, 0.0, 0.01, visual_mode=visual_mode)
    )

    assert update.frame_idx == 91
    assert update.update_ms >= 0.0
    assert update.visual_cost >= 0.0
    assert torch.isfinite(update.state.pose_WB).all()


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
