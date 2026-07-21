import pypose as pp
import torch

from Module.IMUPreintegration import (
    build_sampling_aware_covariance_components,
    preintegrate_imu_local_frame,
)
from Utility.TwoStateSamplingAwareVIO import (
    CrossEdgeImuFactor,
    CrossEdgeTwoStateProblem,
    CrossEdgeTwoStateSolver,
    make_cross_edge_diagonal_prior,
    symmetric_matrix_diagnostics,
)
from Utility.TwoStateVIO import (
    ImuPreintegrationFactor,
    NavigationState,
    RelativePoseFactor,
)


DTYPE = torch.float64


def _pose() -> torch.Tensor:
    return pp.identity_SE3(1, dtype=DTYPE).tensor()


def _state() -> NavigationState:
    return NavigationState(
        pose_WB=_pose(),
        velocity_W=torch.zeros(3, dtype=DTYPE),
        acc_bias=torch.zeros(3, dtype=DTYPE),
        gyro_bias=torch.zeros(3, dtype=DTYPE),
    )


def _base_imu() -> ImuPreintegrationFactor:
    return ImuPreintegrationFactor(
        delta_rotation=torch.zeros(3, dtype=DTYPE),
        delta_velocity=torch.zeros(3, dtype=DTYPE),
        delta_position=torch.zeros(3, dtype=DTYPE),
        covariance=torch.eye(9, dtype=DTYPE),
        dt=0.01,
        bias_jacobian=torch.zeros((9, 6), dtype=DTYPE),
        linearized_acc_bias=torch.zeros(3, dtype=DTYPE),
        linearized_gyro_bias=torch.zeros(3, dtype=DTYPE),
        bias_rw_covariance=torch.eye(6, dtype=DTYPE),
        gravity_handling="preintegration",
    )


def _visual() -> RelativePoseFactor:
    return RelativePoseFactor(
        measurement_BiBj=_pose(),
        covariance=torch.eye(6, dtype=DTYPE),
        huber_delta=3.0,
    )


def test_sampling_components_reconstruct_sa_v1_total_covariance():
    raw_time_ns = torch.tensor([0, 10, 20, 30, 40], dtype=torch.long) * 1_000_000
    knot_time_ns = torch.tensor([5, 15, 25, 35], dtype=torch.long) * 1_000_000
    knot_from_raw = torch.zeros((4, 5), dtype=DTYPE)
    for row in range(4):
        knot_from_raw[row, row] = 0.5
        knot_from_raw[row, row + 1] = 0.5
    acc = torch.tensor(
        [[0.1, -0.2, -9.7], [0.2, -0.1, -9.8], [0.0, 0.1, -9.9], [0.1, 0.0, -9.8]],
        dtype=torch.float32,
    )
    gyro = torch.tensor(
        [[0.01, 0.02, -0.01], [0.02, 0.01, 0.0], [0.0, 0.01, 0.02], [0.01, 0.0, 0.01]],
        dtype=torch.float32,
    )
    preintegration = preintegrate_imu_local_frame(
        time_ns=knot_time_ns,
        acc=acc,
        gyro=gyro,
        sigma_acc=[0.01, 0.02, 0.03],
        sigma_gyro=[0.001, 0.002, 0.003],
        sigma_acc_w=1.0e-4,
        sigma_gyro_w=1.0e-5,
    )
    components = build_sampling_aware_covariance_components(
        preintegration,
        time_ns=knot_time_ns,
        acc_internal=acc,
        gyro_internal=gyro,
        knot_from_raw=knot_from_raw,
        sensor_to_internal_rotation=torch.eye(3, dtype=DTYPE),
        measurement_rate_hz=100.0,
        sigma_acc=[0.01, 0.02, 0.03],
        sigma_gyro=[0.001, 0.002, 0.003],
        acc_bias=torch.zeros(3),
        gyro_bias=torch.zeros(3),
        raw_time_ns=raw_time_ns,
    )

    reconstructed = (
        components.unique_covariance
        + components.incoming_sensitivity @ components.incoming_sensitivity.mT
        + components.outgoing_sensitivity @ components.outgoing_sensitivity.mT
    )
    torch.testing.assert_close(
        reconstructed, components.total_covariance, rtol=1.0e-12, atol=1.0e-15
    )
    assert components.incoming_raw_time_ns.tolist() == raw_time_ns[:2].tolist()
    assert components.outgoing_raw_time_ns.tolist() == raw_time_ns[-2:].tolist()
    assert components.incoming_sensitivity.shape == (9, 12)
    assert components.outgoing_sensitivity.shape == (9, 12)
    diagnostics = symmetric_matrix_diagnostics(
        components.unique_covariance,
        eigenvalue_floor=1.0e-12,
    )
    assert diagnostics.effective_rank < diagnostics.dimension
    assert diagnostics.min_eigenvalue < 1.0e-12


def test_shared_latent_reproduces_adjacent_edge_cross_covariance():
    generator = torch.Generator().manual_seed(19)
    outgoing_previous = torch.randn((9, 12), generator=generator, dtype=DTYPE) * 0.01
    incoming_current = torch.randn((9, 12), generator=generator, dtype=DTYPE) * 0.01
    expected = outgoing_previous @ incoming_current.mT

    joint_sensitivity = torch.cat(
        [outgoing_previous, incoming_current], dim=0
    )
    joint_covariance = joint_sensitivity @ joint_sensitivity.mT
    torch.testing.assert_close(
        joint_covariance[:9, 9:], expected, rtol=1.0e-13, atol=1.0e-15
    )
    assert torch.linalg.matrix_norm(expected) > 0.0


def test_cross_edge_prior_carries_raw_sample_identity_to_next_edge():
    state = _state()
    sensitivity_a = torch.zeros((9, 6), dtype=DTYPE)
    sensitivity_b = torch.zeros((9, 6), dtype=DTYPE)
    sensitivity_a[0, 0] = 0.02
    sensitivity_b[0, 0] = 0.03
    first_imu = CrossEdgeImuFactor(
        base=_base_imu(),
        unique_covariance=torch.eye(9, dtype=DTYPE) * 0.1,
        incoming_raw_time_ns=torch.tensor([10], dtype=torch.long),
        outgoing_raw_time_ns=torch.tensor([20], dtype=torch.long),
        incoming_sensitivity=sensitivity_a,
        outgoing_sensitivity=sensitivity_b,
    )
    first_prior = make_cross_edge_diagonal_prior(
        state,
        first_imu.incoming_raw_time_ns,
        pose_translation_std=0.1,
        pose_rotation_std=0.1,
        velocity_std=0.1,
        acc_bias_std=0.1,
        gyro_bias_std=0.1,
    )
    solver = CrossEdgeTwoStateSolver(
        max_iterations=3,
        rank_aware_imu_whitening=True,
    )
    first = solver.solve(
        CrossEdgeTwoStateProblem(
            state_i=state,
            state_j=state,
            noise_i=torch.zeros(6, dtype=DTYPE),
            noise_j=torch.zeros(6, dtype=DTYPE),
            prior_i=first_prior,
            imu=first_imu,
            visual=_visual(),
        )
    )
    assert first.final_cost < 1.0e-18
    assert first.prior_j.sqrt_information.shape[1] == 21
    assert first.prior_j.raw_time_ns.tolist() == [20]
    assert first.unique_covariance_diagnostics.effective_rank == 9
    assert not first.rank_aware_fallback_active
    assert first.marginalization_diagnostics.quadratic_relative_error < 1.0e-12

    second_imu = CrossEdgeImuFactor(
        base=_base_imu(),
        unique_covariance=torch.eye(9, dtype=DTYPE) * 0.1,
        incoming_raw_time_ns=torch.tensor([20], dtype=torch.long),
        outgoing_raw_time_ns=torch.tensor([30], dtype=torch.long),
        incoming_sensitivity=sensitivity_a,
        outgoing_sensitivity=sensitivity_b,
    )
    second = solver.solve(
        CrossEdgeTwoStateProblem(
            state_i=first.state_j,
            state_j=state,
            noise_i=first.noise_j,
            noise_j=torch.zeros(6, dtype=DTYPE),
            prior_i=first.prior_j,
            imu=second_imu,
            visual=_visual(),
        )
    )
    assert second.final_cost < 1.0e-18
    assert second.cross_covariance_frobenius_norm > 0.0
    assert not second.rank_aware_fallback_active
    assert second.prior_j.raw_time_ns.tolist() == [30]
    assert second.marginalization_diagnostics.quadratic_relative_error < 1.0e-12


def test_rank_deficient_endpoint_support_remains_finite_over_long_static_chain():
    generator = torch.Generator().manual_seed(83)
    state = _state()
    incoming_time = torch.tensor([0], dtype=torch.long)
    incoming_sensitivity = (
        torch.randn((9, 6), generator=generator, dtype=DTYPE) * 1.0e-4
    )
    prior = make_cross_edge_diagonal_prior(
        state,
        incoming_time,
        pose_translation_std=0.1,
        pose_rotation_std=0.1,
        velocity_std=0.1,
        acc_bias_std=0.1,
        gyro_bias_std=0.1,
    )
    solver = CrossEdgeTwoStateSolver(
        max_iterations=6,
        rank_aware_imu_whitening=True,
    )
    maximum_common_translation_update = 0.0

    for edge_index in range(120):
        outgoing_time = torch.tensor([edge_index + 1], dtype=torch.long)
        outgoing_sensitivity = (
            torch.randn((9, 6), generator=generator, dtype=DTYPE) * 1.0e-4
        )
        unique_covariance = torch.diag(
            torch.tensor(
                [1.0e-5] * 6 + [0.0] * 3,
                dtype=DTYPE,
            )
        )
        total_covariance = (
            unique_covariance
            + incoming_sensitivity @ incoming_sensitivity.mT
            + outgoing_sensitivity @ outgoing_sensitivity.mT
        )
        imu = CrossEdgeImuFactor(
            base=ImuPreintegrationFactor(
                delta_rotation=torch.zeros(3, dtype=DTYPE),
                delta_velocity=torch.zeros(3, dtype=DTYPE),
                delta_position=torch.zeros(3, dtype=DTYPE),
                covariance=total_covariance,
                dt=0.01,
                bias_jacobian=torch.zeros((9, 6), dtype=DTYPE),
                linearized_acc_bias=torch.zeros(3, dtype=DTYPE),
                linearized_gyro_bias=torch.zeros(3, dtype=DTYPE),
                bias_rw_covariance=torch.eye(6, dtype=DTYPE) * 1.0e-6,
                gravity_handling="preintegration",
            ),
            unique_covariance=unique_covariance,
            incoming_raw_time_ns=incoming_time,
            outgoing_raw_time_ns=outgoing_time,
            incoming_sensitivity=incoming_sensitivity,
            outgoing_sensitivity=outgoing_sensitivity,
        )
        tangent = torch.zeros((1, 6), dtype=DTYPE)
        tangent[0, 0] = 2.0e-5 * torch.sin(
            torch.tensor(float(edge_index), dtype=DTYPE)
        )
        initial_j = NavigationState(
            pose_WB=(pp.SE3(state.pose_WB) @ pp.se3(tangent).Exp()).tensor(),
            velocity_W=state.velocity_W.clone(),
            acc_bias=state.acc_bias.clone(),
            gyro_bias=state.gyro_bias.clone(),
        )
        result = solver.solve(
            CrossEdgeTwoStateProblem(
                state_i=state,
                state_j=initial_j,
                noise_i=prior.reference_noise.clone(),
                noise_j=torch.zeros(6, dtype=DTYPE),
                prior_i=prior,
                imu=imu,
                visual=_visual(),
            )
        )
        assert torch.isfinite(result.hessian).all()
        assert torch.isfinite(result.gradient).all()
        assert result.unique_covariance_diagnostics.effective_rank == 6
        assert result.rank_aware_imu_whitening
        assert result.rank_aware_fallback_active
        assert result.rank_aware_imu_residual_dimension == 9
        assert result.cross_covariance_frobenius_norm == 0.0
        assert result.marginalization_diagnostics.quadratic_relative_error < 1.0e-9
        maximum_common_translation_update = max(
            maximum_common_translation_update,
            float(torch.linalg.vector_norm(result.common_translation_update_world)),
        )
        state = result.state_j
        prior = result.prior_j
        incoming_time = outgoing_time
        incoming_sensitivity = outgoing_sensitivity

    assert maximum_common_translation_update < 0.05
