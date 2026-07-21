from __future__ import annotations

import pypose as pp
import torch

from Utility.Point import point2pixel_NED
from Utility.TwoStateVIO import (
    LinearizedUVDPoseFactor,
    NavigationState,
    UVDFactor,
    linearize_uvd_pose_factor,
    retract_state,
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
        torch.tensor([[0.3, -0.2, 0.1, 0.0, 0.0, 0.0, 1.0]], dtype=DTYPE)
    )
    truth_j = pose_i @ pp.se3(
        torch.tensor([[0.09, -0.025, 0.018, 0.014, -0.011, 0.02]], dtype=DTYPE)
    ).Exp()
    points_i = torch.tensor(
        [
            [1.8, -0.55, -0.3],
            [2.2, 0.4, 0.25],
            [2.7, -0.2, 0.5],
            [3.1, 0.7, -0.35],
            [3.8, -0.9, 0.15],
            [4.5, 0.2, -0.6],
            [5.2, 1.0, 0.45],
            [5.8, -1.2, -0.1],
        ],
        dtype=DTYPE,
    )
    intrinsic = torch.tensor(
        [[330.0, 0.0, 320.0], [0.0, 328.0, 240.0], [0.0, 0.0, 1.0]],
        dtype=DTYPE,
    )
    baseline = 0.12
    points_j = (truth_j.Inv() @ pose_i).Act(points_i)
    target_uv = point2pixel_NED(points_j, intrinsic)
    target_disparity = intrinsic[0, 0] * baseline / points_j[:, 0:1]
    covariance = torch.diag(
        torch.tensor([0.35, 0.45, 0.25], dtype=DTYPE)
    ).repeat(points_i.shape[0], 1, 1)
    visual = UVDFactor(
        points_Ci=points_i,
        target_uv=target_uv,
        target_disparity=target_disparity,
        covariance_uvd=covariance,
        intrinsic=intrinsic,
        baseline=baseline,
        extrinsic_CI=pp.identity_SE3(1, dtype=DTYPE).tensor(),
        huber_delta=0.8,
    )
    initial_j = truth_j @ pp.se3(
        torch.tensor(
            [[0.004, -0.002, 0.0015, 0.0012, -0.0008, 0.0015]],
            dtype=DTYPE,
        )
    ).Exp()
    return _state(pose_i), _state(initial_j), visual


def _residual(
    state_i: NavigationState,
    state_j: NavigationState,
    factor: UVDFactor | LinearizedUVDPoseFactor,
) -> torch.Tensor:
    return visual_whitened_residuals(
        state_i, state_j, factor, 1.0e-12
    ).reshape(-1)


def _pair_jacobian(
    state_i: NavigationState,
    state_j: NavigationState,
    evaluate,
) -> torch.Tensor:
    zero = torch.zeros(12, dtype=DTYPE, requires_grad=True)

    def residual(increment: torch.Tensor) -> torch.Tensor:
        candidate_i = retract_state(
            state_i, torch.cat([increment[:6], torch.zeros(9, dtype=DTYPE)])
        )
        candidate_j = retract_state(
            state_j, torch.cat([increment[6:], torch.zeros(9, dtype=DTYPE)])
        )
        return evaluate(candidate_i, candidate_j)

    return torch.autograd.functional.jacobian(
        residual, zero, create_graph=False, vectorize=True
    )


def test_full_linear_factor_reproduces_uvd_normal_equations() -> None:
    state_i, state_j, visual = _fixture()
    linearization = linearize_uvd_pose_factor(
        state_i, state_j, visual, marginal_mode="full"
    )
    factor = linearization.factor
    reconstructed_hessian = factor.sqrt_information.mT @ factor.sqrt_information
    reconstructed_gradient = (
        factor.sqrt_information.mT @ factor.residual_offset
    )
    assert torch.allclose(
        reconstructed_hessian,
        linearization.full_hessian,
        atol=2.0e-8,
        rtol=2.0e-10,
    )
    assert torch.allclose(
        reconstructed_gradient,
        linearization.full_gradient,
        atol=2.0e-8,
        rtol=2.0e-10,
    )
    assert float(torch.linalg.eigvalsh(reconstructed_hessian).min()) > -1.0e-8


def test_schur_marginals_match_profiled_full_quadratic() -> None:
    state_i, state_j, visual = _fixture()
    full = linearize_uvd_pose_factor(
        state_i, state_j, visual, marginal_mode="full"
    )
    for mode in ("translation", "rotation"):
        reduced = linearize_uvd_pose_factor(
            state_i, state_j, visual, marginal_mode=mode
        )
        retained = reduced.retained_indices
        nuisance = reduced.nuisance_indices
        h_nn = full.full_hessian[nuisance][:, nuisance]
        h_nr = full.full_hessian[nuisance][:, retained]
        g_n = full.full_gradient[nuisance]
        h_nn_inverse = torch.linalg.pinv(0.5 * (h_nn + h_nn.mT))

        x_zero = torch.zeros(3, dtype=DTYPE)
        nuisance_zero = -h_nn_inverse @ g_n
        test_values = (
            torch.tensor([0.002, -0.001, 0.0015], dtype=DTYPE),
            torch.tensor([-0.001, 0.0025, -0.0007], dtype=DTYPE),
        )

        def full_profiled_cost(x: torch.Tensor) -> torch.Tensor:
            n = -h_nn_inverse @ (h_nr @ x + g_n)
            z = torch.zeros(6, dtype=DTYPE)
            z[retained] = x
            z[nuisance] = n
            return 0.5 * z @ full.full_hessian @ z + full.full_gradient @ z

        base_full = full_profiled_cost(x_zero)
        base_reduced = (
            0.5 * x_zero @ reduced.reduced_hessian @ x_zero
            + reduced.reduced_gradient @ x_zero
        )
        assert bool(torch.isfinite(nuisance_zero).all())
        for x in test_values:
            full_change = full_profiled_cost(x) - base_full
            reduced_cost = (
                0.5 * x @ reduced.reduced_hessian @ x
                + reduced.reduced_gradient @ x
            )
            reduced_change = reduced_cost - base_reduced
            assert torch.allclose(
                full_change, reduced_change, atol=2.0e-8, rtol=2.0e-8
            )


def test_marginal_factors_have_zero_information_in_eliminated_block() -> None:
    state_i, state_j, visual = _fixture()
    translation = linearize_uvd_pose_factor(
        state_i, state_j, visual, marginal_mode="translation"
    ).factor
    rotation = linearize_uvd_pose_factor(
        state_i, state_j, visual, marginal_mode="rotation"
    ).factor
    assert torch.equal(
        translation.sqrt_information[:, 3:],
        torch.zeros_like(translation.sqrt_information[:, 3:]),
    )
    assert torch.equal(
        rotation.sqrt_information[:, :3],
        torch.zeros_like(rotation.sqrt_information[:, :3]),
    )
    assert translation.marginal_mode == "translation"
    assert rotation.marginal_mode == "rotation"


def test_linearized_factor_pair_jacobian_matches_central_difference() -> None:
    state_i, state_j, visual = _fixture()
    factor = linearize_uvd_pose_factor(
        state_i, state_j, visual, marginal_mode="translation"
    ).factor
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
        finite_difference[:, column] = (
            evaluate(step) - evaluate(-step)
        ) / (2.0 * epsilon)
    assert float((autodiff - finite_difference).abs().max()) < 1.0e-5
    assert bool(torch.isfinite(autodiff).all())


def test_full_linear_factor_matches_direct_pair_hessian_and_gradient() -> None:
    state_i, state_j, visual = _fixture()
    linearization = linearize_uvd_pose_factor(
        state_i, state_j, visual, marginal_mode="full"
    )
    base_rows = visual_whitened_residuals(
        state_i, state_j, visual, 1.0e-12
    )
    base_norms = torch.linalg.vector_norm(base_rows, dim=-1)
    weight = torch.where(
        base_norms <= visual.huber_delta,
        torch.ones_like(base_norms),
        torch.as_tensor(visual.huber_delta, dtype=DTYPE)
        / base_norms.clamp_min(1.0e-12),
    ).detach()

    def direct(candidate_i: NavigationState, candidate_j: NavigationState) -> torch.Tensor:
        rows = visual_whitened_residuals(
            candidate_i, candidate_j, visual, 1.0e-12
        )
        return (weight.sqrt().unsqueeze(-1) * rows).reshape(-1)

    direct_jacobian = _pair_jacobian(state_i, state_j, direct)
    direct_residual = direct(state_i, state_j)
    linear_jacobian = _pair_jacobian(
        state_i,
        state_j,
        lambda candidate_i, candidate_j: _residual(
            candidate_i, candidate_j, linearization.factor
        ),
    )
    linear_residual = _residual(state_i, state_j, linearization.factor)
    assert torch.allclose(
        direct_jacobian.mT @ direct_jacobian,
        linear_jacobian.mT @ linear_jacobian,
        atol=5.0e-7,
        rtol=2.0e-9,
    )
    assert torch.allclose(
        direct_jacobian.mT @ direct_residual,
        linear_jacobian.mT @ linear_residual,
        atol=5.0e-7,
        rtol=2.0e-9,
    )
