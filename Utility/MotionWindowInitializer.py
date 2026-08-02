"""Motion-window initialization for sequences without a stationary prefix.

The initializer consumes only archived PACE visual factors and IMU
preintegrations. Ground truth is deliberately absent from this module. The
estimated startup state can be used to reset an incremental backend before all
buffered factors are replayed from the first frame.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np
import pypose as pp
import torch

from Utility.T2HistorySmoother import T2HistoryArchive
from Utility.TwoStateVIO import NavigationState


@dataclass(frozen=True)
class MotionWindowInitialization:
    initial_state: NavigationState
    visual_body_poses: torch.Tensor
    velocity_world: torch.Tensor
    gravity_world_before_alignment: torch.Tensor
    world_alignment_rotation: torch.Tensor
    window_state_count: int
    window_duration_s: float
    diagnostics: dict[str, object]


def _factor_mean_relative(factor) -> pp.LieTensor:
    """Return the minimum of the stored local quadratic visual factor."""

    a = factor.sqrt_information.reshape(-1, 6).double()
    c = factor.residual_offset.reshape(-1).double()
    local = torch.linalg.lstsq(a, -c).solution.reshape(1, 6)
    return pp.SE3(factor.reference_relative_CjCi.double()) @ pp.se3(local).Exp()


def visual_body_pose_chain(archive: T2HistoryArchive) -> torch.Tensor:
    """Chain visual factor means while retaining the archived first IMU pose."""

    if len(archive.online_states) != len(archive.edges) + 1:
        raise ValueError("PACE archive state/edge count is not contiguous")
    extrinsic_ci = pp.SE3(archive.extrinsic_CI.double())
    body = pp.SE3(archive.online_states[0].pose_WB.double())
    poses = [body.tensor().reshape(7).detach().clone()]
    for edge in archive.edges:
        camera_i = body @ extrinsic_ci.Inv()
        relative_cj_ci = _factor_mean_relative(edge.visual)
        camera_j = camera_i @ relative_cj_ci.Inv()
        body = camera_j @ extrinsic_ci
        poses.append(body.tensor().reshape(7).detach().clone())
    return torch.stack(poses, dim=0)


def _rotation_mapping_vector(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Return a 3x3 rotation mapping ``source`` onto ``target``."""

    a = source.double().reshape(3)
    b = target.double().reshape(3)
    a = a / a.norm().clamp_min(1.0e-15)
    b = b / b.norm().clamp_min(1.0e-15)
    dot = torch.dot(a, b).clamp(-1.0, 1.0)
    cross = torch.linalg.cross(a, b)
    if float(cross.norm().item()) < 1.0e-12:
        if float(dot.item()) > 0.0:
            return torch.eye(3, dtype=torch.float64)
        basis = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
        if float(torch.abs(torch.dot(a, basis)).item()) > 0.9:
            basis = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64)
        axis = torch.linalg.cross(a, basis)
        axis = axis / axis.norm()
        return pp.so3((axis * math.pi).reshape(1, 3)).Exp().matrix().reshape(3, 3)
    quaternion = torch.cat([cross, (1.0 + dot).reshape(1)])
    quaternion = quaternion / quaternion.norm()
    return pp.SO3(quaternion.reshape(1, 4)).matrix().reshape(3, 3)


def _sqrt_information(covariance: np.ndarray, floor: float) -> np.ndarray:
    covariance = 0.5 * (covariance + covariance.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    scale = max(float(np.max(np.abs(eigenvalues))), 1.0)
    active = eigenvalues > max(float(floor), scale * 1.0e-12)
    if not np.any(active):
        raise ValueError("IMU covariance has no active information direction")
    return (
        np.diag(1.0 / np.sqrt(eigenvalues[active]))
        @ eigenvectors[:, active].T
    )


def _window_linear_system(
    archive: T2HistoryArchive,
    poses: torch.Tensor,
    state_count: int,
    *,
    covariance_floor: float,
    estimate_acc_bias: bool,
    estimate_gyro_bias: bool,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    bias_columns = tuple(
        ([0, 1, 2] if estimate_acc_bias else [])
        + ([3, 4, 5] if estimate_gyro_bias else [])
    )
    variable_count = 3 * state_count + 3 + len(bias_columns)
    gravity_slice = slice(3 * state_count, 3 * state_count + 3)
    bias_slice = slice(3 * state_count + 3, variable_count)
    pose_lie = pp.SE3(poses[:state_count].double())
    rotations = pose_lie.rotation().matrix().detach().cpu().numpy()
    edge_a: list[np.ndarray] = []
    edge_c: list[np.ndarray] = []

    for index, edge in enumerate(archive.edges[: state_count - 1]):
        factor = edge.imu
        dt = float(factor.dt)
        rotation_i = rotations[index]
        relative = pose_lie[index].Inv() @ pose_lie[index + 1]
        relative_translation = relative.Act(
            torch.zeros(3, dtype=torch.float64)
        ).reshape(3).detach().cpu().numpy()
        delta_rotation = factor.delta_rotation.reshape(-1)
        if int(delta_rotation.numel()) == 3:
            delta_rotation_lie = pp.so3(delta_rotation.reshape(1, 3)).Exp()
        else:
            delta_rotation_lie = pp.SO3(delta_rotation.reshape(1, 4))
        rotation_error = (
            delta_rotation_lie.Inv() @ relative.rotation()
        ).Log().tensor().reshape(3).detach().cpu().numpy()

        jacobian = factor.bias_jacobian.reshape(9, 6).detach().cpu().numpy()
        linearized_bias = np.concatenate(
            [
                factor.linearized_acc_bias.reshape(3).detach().cpu().numpy(),
                factor.linearized_gyro_bias.reshape(3).detach().cpu().numpy(),
            ]
        )
        delta_p = factor.delta_position.reshape(3).detach().cpu().numpy()
        delta_v = factor.delta_velocity.reshape(3).detach().cpu().numpy()

        matrix = np.zeros((9, variable_count), dtype=np.float64)
        constant = np.zeros(9, dtype=np.float64)
        velocity_i = slice(3 * index, 3 * index + 3)
        velocity_j = slice(3 * (index + 1), 3 * (index + 1) + 3)
        rotation_world_to_body = rotation_i.T

        matrix[0:3, velocity_i] = -rotation_world_to_body * dt
        matrix[0:3, gravity_slice] = -0.5 * rotation_world_to_body * dt * dt
        matrix[0:3, bias_slice] = -jacobian[0:3, bias_columns]
        constant[0:3] = relative_translation - delta_p + jacobian[0:3] @ linearized_bias

        matrix[3:6, velocity_i] = -rotation_world_to_body
        matrix[3:6, velocity_j] = rotation_world_to_body
        matrix[3:6, gravity_slice] = -rotation_world_to_body * dt
        matrix[3:6, bias_slice] = -jacobian[3:6, bias_columns]
        constant[3:6] = -delta_v + jacobian[3:6] @ linearized_bias

        matrix[6:9, bias_slice] = -jacobian[6:9, bias_columns]
        constant[6:9] = rotation_error + jacobian[6:9] @ linearized_bias

        whitener = _sqrt_information(
            factor.covariance.reshape(9, 9).detach().cpu().numpy(),
            covariance_floor,
        )
        edge_a.append(whitener @ matrix)
        edge_c.append(whitener @ constant)
    return edge_a, edge_c


def _solve_irls(
    edge_a: Sequence[np.ndarray],
    edge_c: Sequence[np.ndarray],
    initial: np.ndarray,
    *,
    state_count: int,
    gravity_m_s2: float,
    gravity_norm_std: float,
    acc_bias_prior_std: float,
    gyro_bias_prior_std: float,
    estimate_acc_bias: bool,
    estimate_gyro_bias: bool,
    huber_delta: float,
    iterations: int,
) -> tuple[np.ndarray, dict[str, object]]:
    value = initial.copy()
    gravity_start = 3 * state_count
    bias_start = gravity_start + 3
    edge_weights = np.ones(len(edge_a), dtype=np.float64)
    singular_values = np.empty(0, dtype=np.float64)
    rank = 0

    for _ in range(max(1, int(iterations))):
        matrices = [math.sqrt(weight) * matrix for weight, matrix in zip(edge_weights, edge_a)]
        targets = [-math.sqrt(weight) * constant for weight, constant in zip(edge_weights, edge_c)]

        regularization_rows: list[np.ndarray] = []
        active_bias_offset = bias_start
        if estimate_acc_bias:
            rows = np.zeros((3, value.size), dtype=np.float64)
            rows[:, active_bias_offset : active_bias_offset + 3] = (
                np.eye(3) / float(acc_bias_prior_std)
            )
            regularization_rows.append(rows)
            active_bias_offset += 3
        if estimate_gyro_bias:
            rows = np.zeros((3, value.size), dtype=np.float64)
            rows[:, active_bias_offset : active_bias_offset + 3] = (
                np.eye(3) / float(gyro_bias_prior_std)
            )
            regularization_rows.append(rows)
            active_bias_offset += 3
        gravity = value[gravity_start : gravity_start + 3]
        gravity_norm = max(float(np.linalg.norm(gravity)), 1.0e-12)
        gravity_jacobian = gravity / gravity_norm / float(gravity_norm_std)
        gravity_row = np.zeros((1, value.size), dtype=np.float64)
        gravity_row[0, gravity_start : gravity_start + 3] = gravity_jacobian
        gravity_residual = (gravity_norm - float(gravity_m_s2)) / float(gravity_norm_std)
        gravity_offset = gravity_residual - float(gravity_jacobian @ gravity)
        regularization_rows.append(gravity_row)
        regularization = np.vstack(regularization_rows)
        regularization_target = np.zeros(regularization.shape[0], dtype=np.float64)
        regularization_target[-1] = -gravity_offset

        design = np.vstack([*matrices, regularization])
        target = np.concatenate([*targets, regularization_target])
        value, _, rank, singular_values = np.linalg.lstsq(design, target, rcond=1.0e-10)

        norms = np.asarray(
            [np.linalg.norm(matrix @ value + constant) / math.sqrt(matrix.shape[0])
             for matrix, constant in zip(edge_a, edge_c)],
            dtype=np.float64,
        )
        edge_weights = np.where(
            norms <= float(huber_delta),
            1.0,
            float(huber_delta) / np.maximum(norms, 1.0e-12),
        )

    final_norms = np.asarray(
        [np.linalg.norm(matrix @ value + constant) for matrix, constant in zip(edge_a, edge_c)],
        dtype=np.float64,
    )
    condition = (
        float(singular_values[0] / singular_values[-1])
        if singular_values.size and singular_values[-1] > 0.0
        else float("inf")
    )
    return value, {
        "linear_system_rank": int(rank),
        "linear_system_dimension": int(value.size),
        "linear_system_condition_number": condition,
        "edge_whitened_residual_norm_median": float(np.median(final_norms)),
        "edge_whitened_residual_norm_p95": float(np.percentile(final_norms, 95.0)),
        "edge_whitened_residual_norm_max": float(np.max(final_norms)),
        "robust_edge_weight_min": float(np.min(edge_weights)),
        "robust_downweighted_edge_count": int(np.count_nonzero(edge_weights < 1.0)),
    }


def _transform_pose_world(
    poses: torch.Tensor,
    rotation: torch.Tensor,
) -> torch.Tensor:
    pose = pp.SE3(poses.double())
    world_rotation = pp.from_matrix(
        rotation.double().reshape(3, 3), pp.SO3_type, check=False
    )
    position = pose.translation()
    transformed_position = (rotation.double() @ (position - position[0]).mT).mT
    transformed_rotation = world_rotation @ pose.rotation()
    return torch.cat(
        [transformed_position, transformed_rotation.tensor()], dim=-1
    ).detach()


def estimate_motion_window_initialization(
    archive: T2HistoryArchive,
    *,
    window_duration_s: float = 5.0,
    gravity_m_s2: float = 9.81,
    covariance_floor: float = 1.0e-12,
    gravity_norm_std: float = 0.02,
    acc_bias_prior_std: float = 0.2,
    gyro_bias_prior_std: float = 0.02,
    estimate_acc_bias: bool = False,
    estimate_gyro_bias: bool = True,
    huber_delta: float = 3.0,
    irls_iterations: int = 8,
) -> MotionWindowInitialization:
    """Estimate startup state from a moving visual/IMU window without GT."""

    if window_duration_s <= 0.0:
        raise ValueError("motion initialization window must be positive")
    timestamps = np.asarray(archive.timestamps_ns, dtype=np.int64)
    cutoff = int(timestamps[0] + round(float(window_duration_s) * 1.0e9))
    state_count = int(np.searchsorted(timestamps, cutoff, side="right"))
    state_count = min(max(state_count, 3), len(archive.online_states))
    if state_count < 3:
        raise ValueError("motion initialization requires at least three states")

    poses = visual_body_pose_chain(archive)
    pose_lie = pp.SE3(poses[:state_count])
    positions = pose_lie.translation().detach().cpu().numpy()
    velocity_guess = np.zeros((state_count, 3), dtype=np.float64)
    time_s = (timestamps[:state_count] - timestamps[0]).astype(np.float64) * 1.0e-9
    velocity_guess[:-1] = np.diff(positions, axis=0) / np.diff(time_s)[:, None]
    velocity_guess[-1] = velocity_guess[-2]

    bias_columns = tuple(
        ([0, 1, 2] if estimate_acc_bias else [])
        + ([3, 4, 5] if estimate_gyro_bias else [])
    )
    variable_count = 3 * state_count + 3 + len(bias_columns)
    initial = np.zeros(variable_count, dtype=np.float64)
    initial[: 3 * state_count] = velocity_guess.reshape(-1)
    initial[3 * state_count : 3 * state_count + 3] = [0.0, 0.0, gravity_m_s2]
    edge_a, edge_c = _window_linear_system(
        archive,
        poses,
        state_count,
        covariance_floor=covariance_floor,
        estimate_acc_bias=estimate_acc_bias,
        estimate_gyro_bias=estimate_gyro_bias,
    )
    solution, solve_diagnostics = _solve_irls(
        edge_a,
        edge_c,
        initial,
        state_count=state_count,
        gravity_m_s2=gravity_m_s2,
        gravity_norm_std=gravity_norm_std,
        acc_bias_prior_std=acc_bias_prior_std,
        gyro_bias_prior_std=gyro_bias_prior_std,
        estimate_acc_bias=estimate_acc_bias,
        estimate_gyro_bias=estimate_gyro_bias,
        huber_delta=huber_delta,
        iterations=irls_iterations,
    )

    velocity = torch.as_tensor(
        solution[: 3 * state_count].reshape(state_count, 3), dtype=torch.float64
    )
    gravity_before = torch.as_tensor(
        solution[3 * state_count : 3 * state_count + 3], dtype=torch.float64
    )
    active_bias = solution[3 * state_count + 3 :]
    bias_cursor = 0
    acc_bias = torch.zeros(3, dtype=torch.float64)
    gyro_bias = torch.zeros(3, dtype=torch.float64)
    if estimate_acc_bias:
        acc_bias = torch.as_tensor(active_bias[bias_cursor : bias_cursor + 3]).double()
        bias_cursor += 3
    if estimate_gyro_bias:
        gyro_bias = torch.as_tensor(active_bias[bias_cursor : bias_cursor + 3]).double()
    gravity_target = torch.tensor([0.0, 0.0, gravity_m_s2], dtype=torch.float64)
    world_rotation = _rotation_mapping_vector(gravity_before, gravity_target)
    aligned_poses = _transform_pose_world(poses, world_rotation)
    aligned_velocity = (world_rotation @ velocity.T).T
    initial_state = NavigationState(
        pose_WB=aligned_poses[0].reshape(1, 7),
        velocity_W=aligned_velocity[0],
        acc_bias=acc_bias,
        gyro_bias=gyro_bias,
    )

    cosine = torch.dot(
        gravity_before / gravity_before.norm(), gravity_target / gravity_target.norm()
    ).clamp(-1.0, 1.0)
    diagnostics = {
        "uses_ground_truth": False,
        "frame_start": int(archive.frame_indices[0]),
        "frame_end": int(archive.frame_indices[state_count - 1]),
        "window_state_count": int(state_count),
        "window_edge_count": int(state_count - 1),
        "window_duration_s": float(time_s[-1]),
        "gravity_world_before_alignment": gravity_before.tolist(),
        "gravity_norm_m_s2": float(gravity_before.norm().item()),
        "gravity_alignment_tilt_deg": float(torch.rad2deg(torch.acos(cosine)).item()),
        "initial_velocity_world_after_alignment": aligned_velocity[0].tolist(),
        "estimated_acc_bias": acc_bias.tolist(),
        "estimated_gyro_bias": gyro_bias.tolist(),
        "acc_bias_estimation_active": bool(estimate_acc_bias),
        "gyro_bias_estimation_active": bool(estimate_gyro_bias),
        **solve_diagnostics,
    }
    if not all(
        bool(torch.isfinite(value).all())
        for value in (
            aligned_poses,
            aligned_velocity,
            gravity_before,
            world_rotation,
            acc_bias,
            gyro_bias,
        )
    ):
        raise FloatingPointError("motion-window initialization produced NaN/Inf")
    return MotionWindowInitialization(
        initial_state=initial_state,
        visual_body_poses=aligned_poses,
        velocity_world=aligned_velocity,
        gravity_world_before_alignment=gravity_before,
        world_alignment_rotation=world_rotation,
        window_state_count=state_count,
        window_duration_s=float(time_s[-1]),
        diagnostics=diagnostics,
    )


def rotate_archived_states(
    states: Sequence[NavigationState],
    rotation: torch.Tensor,
    *,
    initial_state: NavigationState,
) -> tuple[NavigationState, ...]:
    """Rotate archived guesses into the initialized world and replace state 0."""

    poses = torch.cat([state.pose_WB.reshape(1, 7).double() for state in states], dim=0)
    transformed_poses = _transform_pose_world(poses, rotation.double())
    transformed: list[NavigationState] = []
    for index, state in enumerate(states):
        if index == 0:
            transformed.append(initial_state)
            continue
        transformed.append(
            NavigationState(
                pose_WB=transformed_poses[index].reshape(1, 7),
                velocity_W=rotation.double() @ state.velocity_W.double(),
                acc_bias=initial_state.acc_bias.detach().clone(),
                gyro_bias=initial_state.gyro_bias.detach().clone(),
            )
        )
    return tuple(transformed)
