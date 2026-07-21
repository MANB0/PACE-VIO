from __future__ import annotations

import math
from dataclasses import replace

import pypose as pp
import torch

from Utility.Point import point2pixel_NED
from Utility.TwoStateVIO import (
    ImuPreintegrationFactor,
    NavigationState,
    TwoStateVIOProblem,
    TwoStateVIOSolver,
    UVDFactor,
    make_diagonal_prior,
    retract_state,
    state_boxminus,
    visual_whitened_residuals,
)


DTYPE = torch.float64


def _state(pose: pp.LieTensor) -> NavigationState:
    return NavigationState(
        pose_WB=pose.tensor(),
        velocity_W=torch.zeros(3, dtype=DTYPE),
        acc_bias=torch.zeros(3, dtype=DTYPE),
        gyro_bias=torch.zeros(3, dtype=DTYPE),
    )


def _fixture() -> tuple[NavigationState, NavigationState, UVDFactor]:
    pose_i = pp.SE3(
        torch.tensor([[0.2, -0.1, 0.05, 0.0, 0.0, 0.0, 1.0]], dtype=DTYPE)
    )
    pose_j = pose_i @ pp.se3(
        torch.tensor([[0.08, -0.03, 0.02, 0.012, -0.009, 0.018]], dtype=DTYPE)
    ).Exp()
    points_i = torch.tensor(
        [
            [2.0, -0.4, -0.2],
            [2.8, 0.3, 0.15],
            [3.4, -0.1, 0.4],
            [4.1, 0.6, -0.3],
            [5.0, -0.8, 0.25],
        ],
        dtype=DTYPE,
    )
    intrinsic = torch.tensor(
        [[320.0, 0.0, 320.0], [0.0, 318.0, 240.0], [0.0, 0.0, 1.0]],
        dtype=DTYPE,
    )
    baseline = 0.12
    predicted_j = (pose_j.Inv() @ pose_i).Act(points_i)
    target_uv = point2pixel_NED(predicted_j, intrinsic)
    target_disparity = intrinsic[0, 0] * baseline / predicted_j[:, 0:1]
    covariance = torch.diag(torch.tensor([0.3, 0.4, 0.2], dtype=DTYPE)).repeat(
        points_i.shape[0], 1, 1
    )
    factor = UVDFactor(
        points_Ci=points_i,
        target_uv=target_uv,
        target_disparity=target_disparity,
        covariance_uvd=covariance,
        intrinsic=intrinsic,
        baseline=baseline,
        extrinsic_CI=pp.identity_SE3(1, dtype=DTYPE).tensor(),
        huber_delta=0.1,
    )
    return _state(pose_i), _state(pose_j), factor


def _residual(
    state_i: NavigationState,
    state_j: NavigationState,
    factor: UVDFactor,
) -> torch.Tensor:
    return visual_whitened_residuals(
        state_i, state_j, factor, 1.0e-12
    ).reshape(-1)


def test_direct_uvd_exact_geometry_closes() -> None:
    state_i, state_j, factor = _fixture()
    residual = _residual(state_i, state_j, factor)
    assert float(residual.abs().max()) < 1.0e-11


def test_direct_uvd_depends_on_both_poses_and_matches_central_difference() -> None:
    state_i, state_j, factor = _fixture()
    state_j = retract_state(
        state_j,
        torch.tensor(
            [0.006, -0.003, 0.002, 0.001, -0.0015, 0.002, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            dtype=DTYPE,
        ),
    )
    zero = torch.zeros(12, dtype=DTYPE, requires_grad=True)

    def evaluate(increment: torch.Tensor) -> torch.Tensor:
        candidate_i = retract_state(
            state_i, torch.cat([increment[:6], torch.zeros(9, dtype=DTYPE)])
        )
        candidate_j = retract_state(
            state_j, torch.cat([increment[6:], torch.zeros(9, dtype=DTYPE)])
        )
        return _residual(candidate_i, candidate_j, factor)

    autodiff = torch.autograd.functional.jacobian(evaluate, zero, vectorize=True)
    epsilon = 1.0e-6
    finite_difference = torch.empty_like(autodiff)
    for column in range(12):
        step = torch.zeros(12, dtype=DTYPE)
        step[column] = epsilon
        finite_difference[:, column] = (evaluate(step) - evaluate(-step)) / (2.0 * epsilon)

    error = (autodiff - finite_difference).abs()
    assert float(error.max()) < 1.0e-5
    assert float(torch.linalg.vector_norm(autodiff[:, :6])) > 1.0
    assert float(torch.linalg.vector_norm(autodiff[:, 6:])) > 1.0
    assert bool(torch.isfinite(autodiff).all())


def test_direct_uvd_is_invariant_to_common_world_transform() -> None:
    state_i, state_j, factor = _fixture()
    common = pp.se3(
        torch.tensor([[0.4, -0.2, 0.1, 0.08, -0.04, 0.03]], dtype=DTYPE)
    ).Exp()
    transformed_i = _state(common @ pp.SE3(state_i.pose_WB))
    transformed_j = _state(common @ pp.SE3(state_j.pose_WB))
    assert torch.allclose(
        _residual(state_i, state_j, factor),
        _residual(transformed_i, transformed_j, factor),
        atol=1.0e-10,
        rtol=1.0e-10,
    )


def test_direct_uvd_full_mode_is_the_default() -> None:
    state_i, state_j, factor = _fixture()
    explicit = replace(factor, optimization_mode="full")
    assert torch.equal(
        _residual(state_i, state_j, factor),
        _residual(state_i, state_j, explicit),
    )


def test_rotation_only_uses_current_rotation_and_anchor_translation() -> None:
    state_i, state_j, factor = _fixture()
    anchor = pp.SE3(state_j.pose_WB).Inv() @ pp.SE3(state_i.pose_WB)
    constrained = replace(
        factor,
        optimization_mode="rotation_only",
        anchor_relative_CjCi=anchor.tensor(),
    )
    translated_j = retract_state(
        state_j,
        torch.tensor(
            [0.03, -0.02, 0.01, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            dtype=DTYPE,
        ),
    )
    assert torch.allclose(
        _residual(state_i, state_j, constrained),
        _residual(state_i, translated_j, constrained),
        atol=1.0e-11,
        rtol=1.0e-11,
    )


def test_translation_only_uses_anchor_rotation_and_current_translation() -> None:
    state_i, state_j, factor = _fixture()
    anchor = pp.SE3(state_j.pose_WB).Inv() @ pp.SE3(state_i.pose_WB)
    constrained = replace(
        factor,
        optimization_mode="translation_only",
        anchor_relative_CjCi=anchor.tensor(),
    )
    zero = torch.zeros(12, dtype=DTYPE, requires_grad=True)

    def evaluate(increment: torch.Tensor) -> torch.Tensor:
        candidate_i = retract_state(
            state_i, torch.cat([increment[:6], torch.zeros(9, dtype=DTYPE)])
        )
        candidate_j = retract_state(
            state_j, torch.cat([increment[6:], torch.zeros(9, dtype=DTYPE)])
        )
        return _residual(candidate_i, candidate_j, constrained)

    jacobian = torch.autograd.functional.jacobian(evaluate, zero, vectorize=True)
    assert float(torch.linalg.vector_norm(jacobian[:, [0, 1, 2, 6, 7, 8]])) > 1.0
    assert bool(torch.isfinite(jacobian).all())


def test_no_visual_mode_returns_an_empty_residual() -> None:
    state_i, state_j, factor = _fixture()
    disabled = replace(factor, optimization_mode="no_visual")
    residual = visual_whitened_residuals(state_i, state_j, disabled, 1.0e-12)
    assert residual.shape == (0, 3)


def _solver_problem() -> TwoStateVIOProblem:
    state_i, state_j, visual = _fixture()
    pose_i = pp.SE3(state_i.pose_WB)
    pose_j = pp.SE3(state_j.pose_WB)
    relative = pose_i.Inv() @ pose_j
    delta_rotation = relative.rotation().Log().tensor().reshape(3)
    delta_position = pose_i.rotation().Inv().Act(
        pose_j.Act(torch.zeros(3, dtype=DTYPE))
        - pose_i.Act(torch.zeros(3, dtype=DTYPE))
    ).reshape(3)
    imu = ImuPreintegrationFactor(
        delta_rotation=delta_rotation,
        delta_velocity=torch.zeros(3, dtype=DTYPE),
        delta_position=delta_position,
        covariance=torch.eye(9, dtype=DTYPE) * 1.0e-3,
        dt=0.1,
        bias_jacobian=torch.zeros((9, 6), dtype=DTYPE),
        linearized_acc_bias=torch.zeros(3, dtype=DTYPE),
        linearized_gyro_bias=torch.zeros(3, dtype=DTYPE),
        bias_rw_covariance=torch.eye(6, dtype=DTYPE) * 1.0e-4,
        gravity_world=torch.zeros(3, dtype=DTYPE),
        gravity_handling="residual",
    )
    return TwoStateVIOProblem(
        state_i=state_i,
        state_j=state_j,
        prior_i=make_diagonal_prior(
            state_i,
            pose_translation_std=0.1,
            pose_rotation_std=0.1,
            velocity_std=0.1,
            acc_bias_std=0.1,
            gyro_bias_std=0.1,
        ),
        imu=imu,
        visual_pose=visual,
    )


def test_full_solver_replay_is_deterministic_and_does_not_mutate_prior() -> None:
    problem = _solver_problem()
    prior_information = problem.prior_i.sqrt_information.clone()
    prior_offset = problem.prior_i.residual_offset.clone()
    solver = TwoStateVIOSolver(max_iterations=5)
    first = solver.solve(problem)
    second = solver.solve(problem)
    assert float(torch.linalg.vector_norm(state_boxminus(first.state_i, second.state_i))) < 1e-12
    assert float(torch.linalg.vector_norm(state_boxminus(first.state_j, second.state_j))) < 1e-12
    assert torch.equal(first.prior_j.sqrt_information, second.prior_j.sqrt_information)
    assert torch.equal(first.prior_j.residual_offset, second.prior_j.residual_offset)
    assert torch.equal(problem.prior_i.sqrt_information, prior_information)
    assert torch.equal(problem.prior_i.residual_offset, prior_offset)


def test_all_uvd_modes_solve_without_nan_or_inf() -> None:
    problem = _solver_problem()
    anchor = (
        pp.SE3(problem.state_j.pose_WB).Inv() @ pp.SE3(problem.state_i.pose_WB)
    ).tensor()
    solver = TwoStateVIOSolver(max_iterations=5)
    for mode in ("full", "rotation_only", "translation_only", "no_visual"):
        visual = replace(
            problem.visual_pose,
            optimization_mode=mode,
            anchor_relative_CjCi=(
                anchor if mode in {"rotation_only", "translation_only"} else None
            ),
        )
        result = solver.solve(replace(problem, visual_pose=visual))
        assert bool(torch.isfinite(result.state_i.pose_WB).all())
        assert bool(torch.isfinite(result.state_j.pose_WB).all())
        assert math.isfinite(result.final_cost)
