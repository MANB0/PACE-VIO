import pypose as pp
import torch

from Utility.FixedLagVIO import (
    FixedLagVIOProblem,
    FixedLagVIOSolver,
    _shared_acc_bias_projection,
    propagate_prior_acc_bias_random_walk,
)
from Utility.TwoStateVIO import (
    ImuPreintegrationFactor,
    NavigationState,
    RelativePoseFactor,
    TwoStateVIOProblem,
    TwoStateVIOSolver,
    make_diagonal_prior,
)


DTYPE = torch.float64


def _pose(x=0.0, y=0.0, z=0.0) -> torch.Tensor:
    return pp.SE3(
        torch.tensor([[x, y, z, 0.0, 0.0, 0.0, 1.0]], dtype=DTYPE)
    ).tensor()


def _state(
    x=0.0,
    *,
    velocity=(0.0, 0.0, 0.0),
    acc_bias=(0.0, 0.0, 0.0),
    gyro_bias=(0.0, 0.0, 0.0),
) -> NavigationState:
    return NavigationState(
        pose_WB=_pose(x),
        velocity_W=torch.tensor(velocity, dtype=DTYPE),
        acc_bias=torch.tensor(acc_bias, dtype=DTYPE),
        gyro_bias=torch.tensor(gyro_bias, dtype=DTYPE),
    )


def _imu(
    *,
    dt=1.0,
    delta_p=(0.0, 0.0, 0.0),
    delta_v=(0.0, 0.0, 0.0),
    bias_jacobian=None,
    covariance_scale=1.0e-3,
    bias_covariance_scale=1.0e-3,
) -> ImuPreintegrationFactor:
    if bias_jacobian is None:
        bias_jacobian = torch.zeros((9, 6), dtype=DTYPE)
    return ImuPreintegrationFactor(
        delta_rotation=torch.zeros(3, dtype=DTYPE),
        delta_velocity=torch.tensor(delta_v, dtype=DTYPE),
        delta_position=torch.tensor(delta_p, dtype=DTYPE),
        covariance=torch.eye(9, dtype=DTYPE) * covariance_scale,
        dt=dt,
        bias_jacobian=bias_jacobian,
        linearized_acc_bias=torch.zeros(3, dtype=DTYPE),
        linearized_gyro_bias=torch.zeros(3, dtype=DTYPE),
        bias_rw_covariance=torch.eye(6, dtype=DTYPE) * bias_covariance_scale,
        gravity_handling="preintegration",
    )


def _visual(relative_x: float, covariance_scale=1.0e-4) -> RelativePoseFactor:
    return RelativePoseFactor(
        measurement_BiBj=_pose(relative_x),
        covariance=torch.eye(6, dtype=DTYPE) * covariance_scale,
        huber_delta=5.0,
    )


def _prior(state: NavigationState, *, bias_std=0.1):
    return make_diagonal_prior(
        state,
        pose_translation_std=1.0e-3,
        pose_rotation_std=1.0e-3,
        velocity_std=1.0e-3,
        acc_bias_std=bias_std,
        gyro_bias_std=bias_std,
    )


def test_exact_stationary_pair_has_zero_cost_and_produces_prior():
    state = _state(acc_bias=(0.02, -0.01, 0.03), gyro_bias=(0.001, 0.002, -0.001))
    imu = _imu()
    imu = ImuPreintegrationFactor(
        **{
            **imu.__dict__,
            "linearized_acc_bias": state.acc_bias.clone(),
            "linearized_gyro_bias": state.gyro_bias.clone(),
        }
    )
    result = TwoStateVIOSolver(max_iterations=5).solve(
        TwoStateVIOProblem(
            state_i=state,
            state_j=state,
            prior_i=_prior(state),
            imu=imu,
            visual_pose=_visual(0.0),
        )
    )

    assert result.final_cost < 1.0e-18
    assert result.prior_j.sqrt_information.shape[1] == 15
    assert result.prior_j.sqrt_information.shape[0] > 0
    assert torch.allclose(result.state_i.pose_WB, state.pose_WB, atol=1.0e-12)
    assert torch.allclose(result.state_j.pose_WB, state.pose_WB, atol=1.0e-12)


def test_current_edge_updates_source_bias_and_propagates_it_to_terminal_bias():
    jacobian = torch.zeros((9, 6), dtype=DTYPE)
    jacobian[0:3, 0:3] = torch.eye(3, dtype=DTYPE)
    initial_i = _state()
    initial_j = _state()
    result = TwoStateVIOSolver(max_iterations=20, initial_damping=1.0e-4).solve(
        TwoStateVIOProblem(
            state_i=initial_i,
            state_j=initial_j,
            prior_i=make_diagonal_prior(
                initial_i,
                pose_translation_std=1.0e-4,
                pose_rotation_std=1.0e-4,
                velocity_std=1.0e-4,
                acc_bias_std=10.0,
                gyro_bias_std=0.01,
            ),
            imu=_imu(
                delta_p=(0.2, 0.0, 0.0),
                bias_jacobian=jacobian,
                covariance_scale=1.0e-6,
                bias_covariance_scale=1.0e-5,
            ),
            visual_pose=_visual(0.0, covariance_scale=1.0e-8),
        )
    )

    assert abs(float(result.state_i.acc_bias[0]) + 0.2) < 2.0e-3
    assert abs(float(result.state_j.acc_bias[0]) + 0.2) < 2.0e-3
    assert abs(float(result.state_i.acc_bias[0])) > 0.15
    assert torch.linalg.vector_norm(result.gradient) < 2.0e-3


def test_marginalized_prior_can_seed_the_next_pair():
    state0 = _state(0.0, velocity=(1.0, 0.0, 0.0))
    state1 = _state(1.1, velocity=(1.0, 0.0, 0.0))
    solver = TwoStateVIOSolver(max_iterations=15)
    first = solver.solve(
        TwoStateVIOProblem(
            state_i=state0,
            state_j=state1,
            prior_i=_prior(state0),
            imu=_imu(dt=1.0),
            visual_pose=_visual(1.0),
        )
    )

    state2 = _state(2.3, velocity=(1.0, 0.0, 0.0))
    second = solver.solve(
        TwoStateVIOProblem(
            state_i=first.state_j,
            state_j=state2,
            prior_i=first.prior_j,
            imu=_imu(dt=1.0),
            visual_pose=_visual(1.0),
        )
    )

    assert abs(float(second.state_j.pose_WB[0, 0]) - 2.0) < 2.0e-3
    assert second.final_cost < second.initial_cost
    assert second.prior_j.reference.pose_WB.shape == (1, 7)


def test_fixed_bias_dofs_are_conditioned_out_of_solve_and_prior():
    jacobian = torch.zeros((9, 6), dtype=DTYPE)
    jacobian[0:3, 0:3] = torch.eye(3, dtype=DTYPE)
    state_i = _state(acc_bias=(0.03, -0.02, 0.01), gyro_bias=(0.004, 0.002, -0.003))
    state_j = _state(acc_bias=(0.03, -0.02, 0.01), gyro_bias=(0.004, 0.002, -0.003))
    result = TwoStateVIOSolver(max_iterations=20).solve(
        TwoStateVIOProblem(
            state_i=state_i,
            state_j=state_j,
            prior_i=_prior(state_i, bias_std=10.0),
            imu=_imu(
                delta_p=(0.2, 0.0, 0.0),
                bias_jacobian=jacobian,
                covariance_scale=1.0e-6,
            ),
            visual_pose=_visual(0.0, covariance_scale=1.0e-8),
            optimize_acc_bias=False,
            optimize_gyro_bias=False,
        )
    )

    assert torch.equal(result.state_i.acc_bias, state_i.acc_bias)
    assert torch.equal(result.state_j.acc_bias, state_j.acc_bias)
    assert torch.equal(result.state_i.gyro_bias, state_i.gyro_bias)
    assert torch.equal(result.state_j.gyro_bias, state_j.gyro_bias)
    assert torch.count_nonzero(result.prior_j.sqrt_information[:, 9:15]) == 0


def test_fixed_lag_n2_is_numerically_equivalent_to_two_state_solver():
    jacobian = torch.zeros((9, 6), dtype=DTYPE)
    jacobian[0:3, 0:3] = torch.eye(3, dtype=DTYPE) * 0.15
    state_i = _state(
        0.1,
        velocity=(0.8, -0.1, 0.05),
        acc_bias=(0.02, -0.01, 0.03),
        gyro_bias=(0.001, 0.002, -0.001),
    )
    state_j = _state(
        1.05,
        velocity=(0.9, -0.08, 0.03),
        acc_bias=(0.021, -0.011, 0.029),
        gyro_bias=(0.0012, 0.0018, -0.0009),
    )
    prior = _prior(state_i, bias_std=0.3)
    imu = _imu(
        dt=1.0,
        delta_p=(0.82, -0.04, 0.01),
        delta_v=(0.08, 0.01, -0.02),
        bias_jacobian=jacobian,
        covariance_scale=2.0e-3,
        bias_covariance_scale=5.0e-4,
    )
    visual = _visual(0.9, covariance_scale=2.0e-3)
    kwargs = dict(
        max_iterations=20,
        initial_damping=1.0e-3,
        step_tolerance=1.0e-8,
        cost_tolerance=1.0e-10,
    )
    two_state = TwoStateVIOSolver(**kwargs).solve(
        TwoStateVIOProblem(
            state_i=state_i,
            state_j=state_j,
            prior_i=prior,
            imu=imu,
            visual_pose=visual,
        )
    )
    fixed_lag = FixedLagVIOSolver(**kwargs).solve(
        FixedLagVIOProblem(
            states=(state_i, state_j),
            prior_first=prior,
            imu_factors=(imu,),
            visual_factors=(visual,),
        )
    )

    for expected, actual in zip(
        (two_state.state_i, two_state.state_j), fixed_lag.states
    ):
        assert torch.allclose(actual.pose_WB, expected.pose_WB, atol=1.0e-11, rtol=1.0e-11)
        assert torch.allclose(actual.velocity_W, expected.velocity_W, atol=1.0e-11, rtol=1.0e-11)
        assert torch.allclose(actual.acc_bias, expected.acc_bias, atol=1.0e-11, rtol=1.0e-11)
        assert torch.allclose(actual.gyro_bias, expected.gyro_bias, atol=1.0e-11, rtol=1.0e-11)
    assert abs(fixed_lag.final_cost - two_state.final_cost) < 1.0e-11
    assert torch.allclose(fixed_lag.hessian, two_state.hessian, atol=1.0e-10, rtol=1.0e-10)
    assert torch.allclose(fixed_lag.gradient, two_state.gradient, atol=1.0e-10, rtol=1.0e-10)
    assert torch.allclose(
        fixed_lag.prior_next.sqrt_information,
        two_state.prior_j.sqrt_information,
        atol=1.0e-10,
        rtol=1.0e-10,
    )
    assert torch.allclose(
        fixed_lag.prior_next.residual_offset,
        two_state.prior_j.residual_offset,
        atol=1.0e-10,
        rtol=1.0e-10,
    )


def test_fixed_lag_three_state_chain_uses_future_edge_and_can_fix_acc_bias():
    jacobian = torch.zeros((9, 6), dtype=DTYPE)
    jacobian[0:3, 0:3] = torch.eye(3, dtype=DTYPE)
    states = (_state(0.0), _state(0.0), _state(0.0))
    factors = (
        _imu(
            delta_p=(0.2, 0.0, 0.0),
            bias_jacobian=jacobian,
            covariance_scale=1.0e-6,
            bias_covariance_scale=1.0e-5,
        ),
        _imu(
            delta_p=(0.2, 0.0, 0.0),
            bias_jacobian=jacobian,
            covariance_scale=1.0e-6,
            bias_covariance_scale=1.0e-5,
        ),
    )
    result = FixedLagVIOSolver(max_iterations=20, initial_damping=1.0e-4).solve(
        FixedLagVIOProblem(
            states=states,
            prior_first=make_diagonal_prior(
                states[0],
                pose_translation_std=1.0e-4,
                pose_rotation_std=1.0e-4,
                velocity_std=1.0e-4,
                acc_bias_std=10.0,
                gyro_bias_std=0.01,
            ),
            imu_factors=factors,
            visual_factors=(
                _visual(0.0, covariance_scale=1.0e-8),
                _visual(0.0, covariance_scale=1.0e-8),
            ),
            optimize_acc_bias=False,
            optimize_gyro_bias=True,
        )
    )

    assert result.final_cost < result.initial_cost
    for initial, optimized in zip(states, result.states):
        assert torch.equal(optimized.acc_bias, initial.acc_bias)
    assert torch.count_nonzero(result.prior_next.sqrt_information[:, 9:12]) == 0


def test_shared_acc_bias_projection_ties_all_ba_increments_only():
    projection = _shared_acc_bias_projection(
        5,
        device=torch.device("cpu"),
        dtype=DTYPE,
        optimize_acc_bias=True,
        optimize_gyro_bias=True,
    )
    reduced = torch.arange(projection.shape[1], dtype=DTYPE)
    full = projection @ reduced
    shared = full[9:12]
    for index in range(5):
        block = full[index * 15 : (index + 1) * 15]
        assert torch.equal(block[9:12], shared)
    assert not torch.equal(full[0:9], full[15:24])
    assert not torch.equal(full[12:15], full[27:30])


def test_fixed_lag_shared_acc_bias_is_identical_and_uses_multiple_edges():
    jacobian = torch.zeros((9, 6), dtype=DTYPE)
    jacobian[0:3, 0:3] = torch.eye(3, dtype=DTYPE)
    states = tuple(_state(0.0) for _ in range(5))
    factors = tuple(
        _imu(
            delta_p=(0.2, -0.1, 0.05),
            bias_jacobian=jacobian,
            covariance_scale=1.0e-6,
            bias_covariance_scale=1.0e-5,
        )
        for _ in range(4)
    )
    result = FixedLagVIOSolver(max_iterations=30, initial_damping=1.0e-4).solve(
        FixedLagVIOProblem(
            states=states,
            prior_first=make_diagonal_prior(
                states[0],
                pose_translation_std=1.0e-4,
                pose_rotation_std=1.0e-4,
                velocity_std=1.0e-4,
                acc_bias_std=10.0,
                gyro_bias_std=0.01,
            ),
            imu_factors=factors,
            visual_factors=tuple(
                _visual(0.0, covariance_scale=1.0e-8) for _ in factors
            ),
            optimize_acc_bias=True,
            optimize_gyro_bias=True,
            shared_acc_bias=True,
        )
    )

    assert result.final_cost < result.initial_cost
    assert result.shared_acc_bias
    for state in result.states[1:]:
        assert torch.allclose(state.acc_bias, result.states[0].acc_bias, atol=1e-12, rtol=0)
    assert torch.linalg.vector_norm(result.states[0].acc_bias) > 1.0e-3
    assert result.acc_bias_marginal_information_eigenvalues is not None
    assert torch.all(result.acc_bias_marginal_information_eigenvalues > 0)
    assert torch.count_nonzero(result.prior_next.sqrt_information[:, 9:12]) > 0


def test_shared_acc_bias_rejects_inconsistent_initial_values():
    states = (_state(acc_bias=(0.0, 0.0, 0.0)), _state(acc_bias=(0.1, 0.0, 0.0)))
    try:
        FixedLagVIOSolver(max_iterations=1).solve(
            FixedLagVIOProblem(
                states=states,
                prior_first=_prior(states[0]),
                imu_factors=(_imu(),),
                visual_factors=(_visual(0.0),),
                shared_acc_bias=True,
            )
        )
    except ValueError as error:
        assert "identical initial ba" in str(error)
    else:
        raise AssertionError("shared ba accepted inconsistent initial values")


def test_acc_bias_random_walk_prediction_inflates_only_ba_covariance():
    state = _state(acc_bias=(0.02, -0.01, 0.03))
    prior = make_diagonal_prior(
        state,
        pose_translation_std=0.1,
        pose_rotation_std=0.2,
        velocity_std=0.3,
        acc_bias_std=0.4,
        gyro_bias_std=0.5,
    )
    q_ba = torch.diag(torch.tensor([1.0e-4, 2.0e-4, 3.0e-4], dtype=DTYPE))
    predicted = propagate_prior_acc_bias_random_walk(prior, q_ba)
    before_h = prior.sqrt_information.mT @ prior.sqrt_information
    after_h = predicted.sqrt_information.mT @ predicted.sqrt_information
    before_p = torch.linalg.inv(before_h)
    after_p = torch.linalg.inv(after_h)
    expected = before_p.clone()
    expected[9:12, 9:12] += q_ba

    assert torch.allclose(after_p, expected, atol=1e-12, rtol=1e-11)
    assert torch.allclose(predicted.residual_offset, torch.zeros_like(predicted.residual_offset))
    assert torch.equal(predicted.reference.acc_bias, prior.reference.acc_bias)


def test_shared_acc_bias_hold_keeps_ba_fixed_but_carries_information_forward():
    jacobian = torch.zeros((9, 6), dtype=DTYPE)
    jacobian[0:3, 0:3] = torch.eye(3, dtype=DTYPE)
    states = tuple(_state(acc_bias=(0.02, -0.01, 0.03)) for _ in range(3))
    result = FixedLagVIOSolver(max_iterations=10, initial_damping=1.0e-4).solve(
        FixedLagVIOProblem(
            states=states,
            prior_first=make_diagonal_prior(
                states[0],
                pose_translation_std=1.0e-4,
                pose_rotation_std=1.0e-4,
                velocity_std=1.0e-4,
                acc_bias_std=0.2,
                gyro_bias_std=0.01,
            ),
            imu_factors=tuple(
                _imu(delta_p=(0.1, 0.0, 0.0), bias_jacobian=jacobian) for _ in range(2)
            ),
            visual_factors=tuple(_visual(0.0) for _ in range(2)),
            optimize_acc_bias=False,
            optimize_gyro_bias=True,
            shared_acc_bias=True,
        )
    )

    for state in result.states:
        assert torch.equal(state.acc_bias, states[0].acc_bias)
    assert result.acc_bias_marginal_information_eigenvalues is not None
    assert torch.all(result.acc_bias_marginal_information_eigenvalues > 0)
    assert torch.count_nonzero(result.prior_next.sqrt_information[:, 9:12]) > 0
