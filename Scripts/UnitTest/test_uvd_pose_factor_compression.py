from __future__ import annotations

import pypose as pp
import torch

from Utility.Point import point2pixel_NED
from Utility.TwoStateVIO import (
    NavigationState,
    UVDFactor,
    linearize_uvd_pose_factor,
    linearize_uvd_relative_pose_factor,
    linearized_uvd_pose_factor_from_normal_equations,
)


DTYPE = torch.float64


def _state(pose: pp.LieTensor) -> NavigationState:
    return NavigationState(
        pose_WB=pose.tensor(),
        velocity_W=torch.zeros(3, dtype=DTYPE),
        acc_bias=torch.zeros(3, dtype=DTYPE),
        gyro_bias=torch.zeros(3, dtype=DTYPE),
    )


def _visual(relative_CjCi: pp.LieTensor, count: int = 12) -> UVDFactor:
    generator = torch.Generator().manual_seed(17 + count)
    points = torch.empty((count, 3), dtype=DTYPE)
    points[:, 0] = 2.0 + 3.0 * torch.rand(count, generator=generator, dtype=DTYPE)
    points[:, 1:] = 1.2 * (torch.rand((count, 2), generator=generator, dtype=DTYPE) - 0.5)
    intrinsic = torch.tensor(
        [[330.0, 0.0, 320.0], [0.0, 328.0, 240.0], [0.0, 0.0, 1.0]],
        dtype=DTYPE,
    )
    baseline = 0.12
    target = relative_CjCi.Act(points)
    covariance = torch.diag(torch.tensor([0.35, 0.45, 0.25], dtype=DTYPE)).repeat(
        count, 1, 1
    )
    return UVDFactor(
        points_Ci=points,
        target_uv=point2pixel_NED(target, intrinsic),
        target_disparity=intrinsic[0, 0] * baseline / target[:, 0:1],
        covariance_uvd=covariance,
        intrinsic=intrinsic,
        baseline=baseline,
        extrinsic_CI=pp.identity_SE3(1, dtype=DTYPE).tensor(),
        huber_delta=0.8,
    )


def _fixed_weight_residual(
    reference: pp.LieTensor,
    visual: UVDFactor,
    increment: torch.Tensor,
) -> torch.Tensor:
    candidate = reference @ pp.se3(increment.reshape(1, 6)).Exp()
    predicted = candidate.Act(visual.points_Ci)
    raw = torch.cat(
        [
            point2pixel_NED(predicted, visual.intrinsic) - visual.target_uv,
            visual.intrinsic[0, 0] * visual.baseline / predicted[:, 0:1]
            - visual.target_disparity,
        ],
        dim=-1,
    )
    lower = torch.linalg.cholesky(visual.covariance_uvd)
    rows = torch.linalg.solve_triangular(
        lower, raw.unsqueeze(-1), upper=False
    ).squeeze(-1)
    base_predicted = reference.Act(visual.points_Ci)
    base_raw = torch.cat(
        [
            point2pixel_NED(base_predicted, visual.intrinsic) - visual.target_uv,
            visual.intrinsic[0, 0] * visual.baseline / base_predicted[:, 0:1]
            - visual.target_disparity,
        ],
        dim=-1,
    )
    base_rows = torch.linalg.solve_triangular(
        lower, base_raw.unsqueeze(-1), upper=False
    ).squeeze(-1)
    norms = torch.linalg.vector_norm(base_rows, dim=-1)
    weight = torch.where(
        norms <= visual.huber_delta,
        torch.ones_like(norms),
        torch.as_tensor(visual.huber_delta, dtype=DTYPE) / norms.clamp_min(1.0e-12),
    )
    return (weight.sqrt().unsqueeze(-1) * rows).reshape(-1)


def test_supplied_relative_pose_matches_two_state_wrapper() -> None:
    relative = pp.se3(
        torch.tensor([[0.08, -0.03, 0.02, 0.01, -0.015, 0.02]], dtype=DTYPE)
    ).Exp()
    visual = _visual(relative)
    state_i = _state(pp.identity_SE3(1, dtype=DTYPE))
    state_j = _state(relative.Inv())
    direct = linearize_uvd_relative_pose_factor(
        relative.tensor(), visual, marginal_mode="full"
    )
    wrapped = linearize_uvd_pose_factor(
        state_i, state_j, visual, marginal_mode="full"
    )
    assert torch.allclose(direct.full_hessian, wrapped.full_hessian, atol=1.0e-10)
    assert torch.allclose(direct.full_gradient, wrapped.full_gradient, atol=1.0e-10)
    assert torch.allclose(
        direct.factor.reference_relative_CjCi,
        wrapped.factor.reference_relative_CjCi,
        atol=1.0e-12,
    )


def test_twenty_random_nonzero_pose_jacobians_match_central_difference() -> None:
    generator = torch.Generator().manual_seed(20260720)
    epsilon = 1.0e-6
    maximum_error = 0.0
    for _ in range(20):
        truth_tangent = torch.cat(
            [
                0.08 * (torch.rand(3, generator=generator, dtype=DTYPE) - 0.5),
                0.04 * (torch.rand(3, generator=generator, dtype=DTYPE) - 0.5),
            ]
        ).reshape(1, 6)
        truth = pp.se3(truth_tangent).Exp()
        visual = _visual(truth)
        offset = torch.cat(
            [
                0.01 * (torch.rand(3, generator=generator, dtype=DTYPE) - 0.5),
                0.006 * (torch.rand(3, generator=generator, dtype=DTYPE) - 0.5),
            ]
        )
        reference = truth @ pp.se3(offset.reshape(1, 6)).Exp()
        linearization = linearize_uvd_relative_pose_factor(
            reference.tensor(), visual, marginal_mode="full"
        )
        finite_difference = torch.empty_like(linearization.relative_jacobian)
        for column in range(6):
            step = torch.zeros(6, dtype=DTYPE)
            step[column] = epsilon
            finite_difference[:, column] = (
                _fixed_weight_residual(reference, visual, step)
                - _fixed_weight_residual(reference, visual, -step)
            ) / (2.0 * epsilon)
        error = float(
            (linearization.relative_jacobian - finite_difference).abs().max().item()
        )
        maximum_error = max(maximum_error, error)
        assert bool(torch.isfinite(finite_difference).all())
    assert maximum_error < 1.0e-5


def test_rank_aware_factor_does_not_fill_unobservable_modes() -> None:
    relative = pp.se3(
        torch.tensor([[0.04, -0.01, 0.02, 0.01, 0.005, -0.008]], dtype=DTYPE)
    ).Exp()
    visual = _visual(relative, count=1)
    linearization = linearize_uvd_relative_pose_factor(
        relative.tensor(), visual, marginal_mode="full"
    )
    factor = linearization.factor
    reconstructed = factor.sqrt_information.mT @ factor.sqrt_information
    eigenvalues = torch.linalg.eigvalsh(reconstructed)
    assert factor.sqrt_information.shape[0] <= 3
    assert int((eigenvalues <= 1.0e-9).sum().item()) >= 3
    assert torch.allclose(
        reconstructed, linearization.full_hessian, atol=1.0e-8, rtol=1.0e-10
    )


def test_cached_normal_equations_reconstruct_identical_local_factor() -> None:
    truth = pp.se3(
        torch.tensor([[0.06, -0.025, 0.015, 0.012, -0.007, 0.018]], dtype=DTYPE)
    ).Exp()
    visual = _visual(truth, count=18)
    reference = truth @ pp.se3(
        torch.tensor([[0.008, -0.004, 0.003, 0.002, -0.001, 0.0015]], dtype=DTYPE)
    ).Exp()
    direct = linearize_uvd_relative_pose_factor(
        reference.tensor(), visual, marginal_mode="full"
    )
    cached = linearized_uvd_pose_factor_from_normal_equations(
        reference.tensor(),
        direct.full_hessian,
        direct.full_gradient,
        visual.extrinsic_CI,
    )
    assert torch.allclose(
        cached.sqrt_information.mT @ cached.sqrt_information,
        direct.full_hessian,
        atol=1.0e-8,
        rtol=1.0e-10,
    )
    assert torch.allclose(
        cached.sqrt_information.mT @ cached.residual_offset,
        direct.full_gradient,
        atol=1.0e-8,
        rtol=1.0e-10,
    )
    for _ in range(20):
        delta = 2.0e-3 * torch.randn(6, dtype=DTYPE)
        direct_residual = (
            direct.factor.sqrt_information @ delta
            + direct.factor.residual_offset
        )
        cached_residual = cached.sqrt_information @ delta + cached.residual_offset
        assert torch.allclose(
            direct_residual.square().sum(),
            cached_residual.square().sum(),
            atol=1.0e-9,
            rtol=1.0e-10,
        )
