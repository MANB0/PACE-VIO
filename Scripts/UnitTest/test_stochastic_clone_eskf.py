from __future__ import annotations

import pypose as pp
import torch

from Utility.StochasticCloneESKF import (
    AUGMENTED_DOF,
    ESKFNoiseDensities,
    ESKFNominalState,
    augment_with_current_pose_clone,
    initial_navigation_covariance,
    propagate_imu_knots,
    reclone_current_pose,
    relative_pose_jacobian_central,
    relative_pose_residual,
    update_relative_pose,
)


DTYPE = torch.float64


def nominal() -> ESKFNominalState:
    pose = pp.se3(torch.tensor([[0.3, -0.1, 0.2, 0.1, -0.04, 0.08]], dtype=DTYPE)).Exp()
    return ESKFNominalState(
        pose_WB=pose.tensor(),
        velocity_W=torch.tensor([0.4, -0.2, 0.1], dtype=DTYPE),
        acc_bias=torch.tensor([0.01, -0.02, 0.03], dtype=DTYPE),
        gyro_bias=torch.tensor([0.002, -0.003, 0.004], dtype=DTYPE),
    )


def initial_state():
    covariance = initial_navigation_covariance(
        pose_translation_std=0.1,
        pose_rotation_std=0.05,
        velocity_std=0.2,
        acc_bias_std=0.05,
        gyro_bias_std=0.01,
        dtype=DTYPE,
    )
    return augment_with_current_pose_clone(nominal(), covariance, 1_000_000_000)


def test_clone_augmentation_preserves_exact_current_pose_cross_covariance():
    state = initial_state()
    selector = torch.zeros((6, 15), dtype=DTYPE)
    selector[0:3, 0:3] = torch.eye(3, dtype=DTYPE)
    selector[3:6, 6:9] = torch.eye(3, dtype=DTYPE)
    assert torch.allclose(
        state.covariance[:15, 15:21],
        state.covariance[:15, :15] @ selector.mT,
        atol=1e-14,
    )
    assert torch.linalg.eigvalsh(state.covariance).min() >= -1e-12


def test_stationary_specific_force_propagates_without_nominal_motion():
    state = initial_state()
    zero_pose = pp.identity_SE3(1, dtype=DTYPE)
    state = augment_with_current_pose_clone(
        ESKFNominalState(
            pose_WB=zero_pose.tensor(),
            velocity_W=torch.zeros(3, dtype=DTYPE),
            acc_bias=torch.zeros(3, dtype=DTYPE),
            gyro_bias=torch.zeros(3, dtype=DTYPE),
        ),
        state.covariance[:15, :15],
        0,
    )
    time = torch.tensor([0, 10_000_000, 20_000_000], dtype=torch.long)
    acc = torch.tensor([[0.0, 0.0, -9.8]] * 3, dtype=DTYPE)
    gyro = torch.zeros((3, 3), dtype=DTYPE)
    noise = ESKFNoiseDensities(*[torch.full((3,), 1e-4, dtype=DTYPE) for _ in range(4)])
    propagated = propagate_imu_knots(
        state,
        time,
        acc,
        gyro,
        torch.tensor([0.0, 0.0, 9.8], dtype=DTYPE),
        noise,
    )
    assert torch.linalg.vector_norm(pp.SE3(propagated.nominal.pose_WB).translation()) < 1e-12
    assert torch.linalg.vector_norm(propagated.nominal.velocity_W) < 1e-12
    assert torch.linalg.vector_norm(pp.SE3(propagated.nominal.pose_WB).rotation().Log()) < 1e-12
    assert torch.isfinite(propagated.covariance).all()
    assert torch.linalg.eigvalsh(propagated.covariance).min() >= -1e-12


def test_relative_pose_jacobian_has_exact_velocity_and_bias_zero_blocks():
    state = initial_state()
    measurement = pp.identity_SE3(1, dtype=DTYPE).tensor()
    jacobian = relative_pose_jacobian_central(state, measurement, epsilon=1e-6)
    assert jacobian.shape == (6, AUGMENTED_DOF)
    assert jacobian[:, 3:6].abs().max() < 1e-10
    assert jacobian[:, 9:15].abs().max() < 1e-10
    assert jacobian[:, 0:3].abs().max() > 1e-3
    assert jacobian[:, 6:9].abs().max() > 1e-3
    assert jacobian[:, 15:21].abs().max() > 1e-3


def test_relative_pose_is_invariant_to_common_world_translation():
    state = initial_state()
    measurement = pp.se3(
        torch.tensor([[0.1, -0.02, 0.03, 0.01, -0.02, 0.04]], dtype=DTYPE)
    ).Exp().tensor()
    common = torch.tensor([0.7, -0.4, 0.2], dtype=DTYPE)
    current = pp.SE3(state.nominal.pose_WB)
    clone = pp.SE3(state.clone.pose_WB)
    translated = type(state)(
        nominal=ESKFNominalState(
            pose_WB=pp.SE3(
                torch.cat(
                    [
                        current.translation().reshape(3) + common,
                        current.rotation().tensor().reshape(4),
                    ]
                ).reshape(1, 7)
            ).tensor(),
            velocity_W=state.nominal.velocity_W,
            acc_bias=state.nominal.acc_bias,
            gyro_bias=state.nominal.gyro_bias,
        ),
        clone=type(state.clone)(
            pose_WB=pp.SE3(
                torch.cat(
                    [
                        clone.translation().reshape(3) + common,
                        clone.rotation().tensor().reshape(4),
                    ]
                ).reshape(1, 7)
            ).tensor(),
            timestamp_ns=state.clone.timestamp_ns,
        ),
        covariance=state.covariance,
        timestamp_ns=state.timestamp_ns,
    )
    assert torch.allclose(
        relative_pose_residual(state, measurement),
        relative_pose_residual(translated, measurement),
        atol=1e-12,
    )


def test_relative_pose_update_reduces_residual_and_keeps_covariance_psd():
    state = initial_state()
    propagated_nominal = ESKFNominalState(
        pose_WB=(
            pp.SE3(state.nominal.pose_WB)
            @ pp.se3(torch.tensor([[0.2, -0.05, 0.03, 0.04, -0.01, 0.02]], dtype=DTYPE)).Exp()
        ).tensor(),
        velocity_W=state.nominal.velocity_W,
        acc_bias=state.nominal.acc_bias,
        gyro_bias=state.nominal.gyro_bias,
    )
    propagated = type(state)(
        nominal=propagated_nominal,
        clone=state.clone,
        covariance=state.covariance + torch.eye(21, dtype=DTYPE) * 1e-3,
        timestamp_ns=state.timestamp_ns + 33_333_333,
    )
    measurement = pp.se3(
        torch.tensor([[0.17, -0.04, 0.02, 0.03, -0.008, 0.018]], dtype=DTYPE)
    ).Exp().tensor()
    before = torch.linalg.vector_norm(relative_pose_residual(propagated, measurement))
    updated, diagnostics = update_relative_pose(
        propagated,
        measurement,
        torch.diag(torch.tensor([0.02] * 3 + [0.01] * 3, dtype=DTYPE).square()),
    )
    after = torch.linalg.vector_norm(relative_pose_residual(updated, measurement))
    assert after < before
    assert diagnostics.residual_norm_after < diagnostics.residual_norm_before
    assert diagnostics.nis >= 0.0
    assert diagnostics.finite
    assert torch.linalg.eigvalsh(updated.covariance).min() >= -1e-12
    recloned = reclone_current_pose(updated)
    assert recloned.clone.timestamp_ns == updated.timestamp_ns
    assert torch.allclose(recloned.clone.pose_WB, updated.nominal.pose_WB)
