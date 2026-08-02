from pathlib import Path

import numpy as np
import pypose as pp
import torch

from Utility.MotionWindowInitializer import (
    _rotation_mapping_vector,
    estimate_motion_window_initialization,
)
from Utility.T2HistorySmoother import ArchivedT2Edge, T2HistoryArchive
from Utility.TwoStateVIO import (
    ImuPreintegrationFactor,
    LinearizedUVDPoseFactor,
    NavigationState,
    make_diagonal_prior,
)


def test_rotation_mapping_vector_aligns_gravity_without_scaling():
    source = torch.tensor([0.8, -0.4, 9.7], dtype=torch.float64)
    target = torch.tensor([0.0, 0.0, 9.81], dtype=torch.float64)

    rotation = _rotation_mapping_vector(source, target)
    mapped = rotation @ (source / source.norm())

    assert torch.allclose(mapped, target / target.norm(), atol=1.0e-10, rtol=0.0)
    assert torch.allclose(rotation.T @ rotation, torch.eye(3, dtype=torch.float64), atol=1.0e-10)
    assert torch.det(rotation) > 0.0


def test_rotation_mapping_vector_handles_already_aligned_input():
    gravity = torch.tensor([0.0, 0.0, 9.81], dtype=torch.float64)
    rotation = _rotation_mapping_vector(gravity, gravity)
    assert torch.equal(rotation, torch.eye(3, dtype=torch.float64))


def test_moving_window_recovers_consistent_gravity_and_velocity_without_bias():
    state_count = 8
    dt = 0.1
    gravity = torch.tensor([0.0, 0.0, 9.81], dtype=torch.float64)
    velocity = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
    identity = pp.identity_SE3(1, dtype=torch.float64).tensor()
    states = []
    edges = []
    for index in range(state_count):
        pose = identity.clone()
        pose[0, 0] = index * dt
        states.append(
            NavigationState(
                pose_WB=pose,
                velocity_W=torch.zeros(3, dtype=torch.float64),
                acc_bias=torch.zeros(3, dtype=torch.float64),
                gyro_bias=torch.zeros(3, dtype=torch.float64),
            )
        )
        if index == 0:
            continue
        reference = identity.clone()
        reference[0, 0] = -dt
        visual = LinearizedUVDPoseFactor(
            reference_relative_CjCi=reference,
            sqrt_information=torch.eye(6, dtype=torch.float64),
            residual_offset=torch.zeros(6, dtype=torch.float64),
            extrinsic_CI=identity,
            marginal_mode="full",
        )
        imu = ImuPreintegrationFactor(
            delta_rotation=torch.zeros(3, dtype=torch.float64),
            delta_velocity=-gravity * dt,
            delta_position=-0.5 * gravity * dt * dt,
            covariance=torch.eye(9, dtype=torch.float64) * 1.0e-4,
            dt=dt,
            bias_jacobian=torch.zeros((9, 6), dtype=torch.float64),
            linearized_acc_bias=torch.zeros(3, dtype=torch.float64),
            linearized_gyro_bias=torch.zeros(3, dtype=torch.float64),
            bias_rw_covariance=torch.eye(6, dtype=torch.float64) * 1.0e-6,
            gravity_world=gravity,
            gravity_handling="residual",
        )
        edges.append(
            ArchivedT2Edge(
                frame_i=index - 1,
                frame_j=index,
                imu=imu,
                visual=visual,
                cached_hessian=torch.eye(6, dtype=torch.float64),
                cached_gradient=torch.zeros(6, dtype=torch.float64),
            )
        )
    archive = T2HistoryArchive(
        source_path=Path("synthetic.npz"),
        source_sha256="synthetic",
        frame_indices=np.arange(state_count, dtype=np.int64),
        timestamps_ns=np.arange(state_count, dtype=np.int64) * int(dt * 1.0e9),
        online_states=tuple(states),
        initial_prior=make_diagonal_prior(
            states[0],
            pose_translation_std=1.0,
            pose_rotation_std=1.0,
            velocity_std=1.0,
            acc_bias_std=1.0,
            gyro_bias_std=1.0,
        ),
        edges=tuple(edges),
        extrinsic_CI=identity,
    )

    result = estimate_motion_window_initialization(
        archive,
        window_duration_s=0.7,
        estimate_acc_bias=False,
        estimate_gyro_bias=False,
    )

    assert torch.allclose(
        result.gravity_world_before_alignment, gravity, atol=1.0e-8, rtol=0.0
    )
    assert torch.allclose(
        result.initial_state.velocity_W, velocity, atol=1.0e-8, rtol=0.0
    )
    assert result.diagnostics["linear_system_rank"] == result.diagnostics[
        "linear_system_dimension"
    ]
