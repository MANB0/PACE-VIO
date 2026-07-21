from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class SlidingWindowEdge:
    i: int
    j: int
    dt: float
    visual_delta_p_body: np.ndarray
    visual_delta_R: np.ndarray
    imu_delta_p_body: np.ndarray
    imu_delta_v_body: np.ndarray
    imu_delta_R: np.ndarray


@dataclass(frozen=True)
class SlidingWindowSequence:
    time_ns: np.ndarray
    position_w: np.ndarray
    rotation_bw: np.ndarray
    velocity_w: np.ndarray
    edges: list[SlidingWindowEdge]


@dataclass(frozen=True)
class SlidingWindowConfig:
    window_size: int = 10
    stride: int = 5
    max_nfev: int = 12
    visual_position_weight: float = 1.0
    visual_rotation_weight: float = 1.0
    imu_position_weight: float = 1.0
    imu_velocity_weight: float = 1.0
    imu_rotation_weight: float = 1.0
    anchor_position_weight: float = 100.0
    anchor_rotation_weight: float = 100.0
    anchor_velocity_weight: float = 1.0
    velocity_damping_weight: float = 1e-3


@dataclass(frozen=True)
class SlidingWindowResult:
    time_ns: np.ndarray
    position_w: np.ndarray
    rotation_bw: np.ndarray
    velocity_w: np.ndarray
    num_windows: int
    total_nfev: int
    total_cost: float


def _rotation_log(R: np.ndarray) -> np.ndarray:
    return Rotation.from_matrix(np.asarray(R, dtype=np.float64).reshape(3, 3)).as_rotvec()


def _pack_window(
    position: np.ndarray,
    rotation: np.ndarray,
    velocity: np.ndarray,
    start: int,
    end: int,
) -> np.ndarray:
    values: list[np.ndarray] = []
    for idx in range(start + 1, end + 1):
        values.extend(
            [
                position[idx].reshape(3),
                _rotation_log(rotation[idx]).reshape(3),
                velocity[idx].reshape(3),
            ]
        )
    if not values:
        return np.zeros(0, dtype=np.float64)
    return np.concatenate(values).astype(np.float64)


def _unpack_window(
    x: np.ndarray,
    position: np.ndarray,
    rotation: np.ndarray,
    velocity: np.ndarray,
    start: int,
    end: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    p = position[start : end + 1].copy()
    R = rotation[start : end + 1].copy()
    v = velocity[start : end + 1].copy()
    cursor = 0
    for local_idx in range(1, end - start + 1):
        p[local_idx] = x[cursor : cursor + 3]
        cursor += 3
        R[local_idx] = Rotation.from_rotvec(x[cursor : cursor + 3]).as_matrix()
        cursor += 3
        v[local_idx] = x[cursor : cursor + 3]
        cursor += 3
    return p, R, v


def _edge_residuals(
    edge: SlidingWindowEdge,
    p: np.ndarray,
    R: np.ndarray,
    v: np.ndarray,
    local_i: int,
    config: SlidingWindowConfig,
) -> list[np.ndarray]:
    local_j = local_i + 1
    R_i = R[local_i]
    R_j = R[local_j]
    rel_R = R_i.T @ R_j
    rel_p_body = R_i.T @ (p[local_j] - p[local_i])
    dv_body = R_i.T @ (v[local_j] - v[local_i])
    pred_p_minus_vel = R_i.T @ (p[local_j] - p[local_i] - v[local_i] * float(edge.dt))

    residuals: list[np.ndarray] = []
    if config.visual_position_weight > 0.0:
        residuals.append(
            float(config.visual_position_weight)
            * (rel_p_body - np.asarray(edge.visual_delta_p_body, dtype=np.float64).reshape(3))
        )
    if config.visual_rotation_weight > 0.0:
        residuals.append(
            float(config.visual_rotation_weight)
            * _rotation_log(np.asarray(edge.visual_delta_R, dtype=np.float64).reshape(3, 3).T @ rel_R)
        )
    if config.imu_position_weight > 0.0:
        residuals.append(
            float(config.imu_position_weight)
            * (pred_p_minus_vel - np.asarray(edge.imu_delta_p_body, dtype=np.float64).reshape(3))
        )
    if config.imu_velocity_weight > 0.0:
        residuals.append(
            float(config.imu_velocity_weight)
            * (dv_body - np.asarray(edge.imu_delta_v_body, dtype=np.float64).reshape(3))
        )
    if config.imu_rotation_weight > 0.0:
        residuals.append(
            float(config.imu_rotation_weight)
            * _rotation_log(np.asarray(edge.imu_delta_R, dtype=np.float64).reshape(3, 3).T @ rel_R)
        )
    if config.velocity_damping_weight > 0.0:
        residuals.append(float(config.velocity_damping_weight) * (v[local_j] - v[local_i]))
    return residuals


def _window_residual(
    x: np.ndarray,
    sequence: SlidingWindowSequence,
    position: np.ndarray,
    rotation: np.ndarray,
    velocity: np.ndarray,
    start: int,
    end: int,
    config: SlidingWindowConfig,
) -> np.ndarray:
    p, R, v = _unpack_window(x, position, rotation, velocity, start, end)
    residuals: list[np.ndarray] = []

    if config.anchor_position_weight > 0.0:
        residuals.append(float(config.anchor_position_weight) * (p[0] - position[start]))
    if config.anchor_rotation_weight > 0.0:
        residuals.append(float(config.anchor_rotation_weight) * _rotation_log(rotation[start].T @ R[0]))
    if config.anchor_velocity_weight > 0.0:
        residuals.append(float(config.anchor_velocity_weight) * (v[0] - velocity[start]))

    for edge in sequence.edges[start:end]:
        residuals.extend(_edge_residuals(edge, p, R, v, edge.i - start, config))

    if not residuals:
        return np.zeros(0, dtype=np.float64)
    return np.concatenate([r.reshape(-1) for r in residuals]).astype(np.float64)


def optimize_sliding_window_sequence(
    sequence: SlidingWindowSequence,
    config: SlidingWindowConfig | None = None,
) -> SlidingWindowResult:
    cfg = config or SlidingWindowConfig()
    n = int(len(sequence.time_ns))
    if n == 0:
        raise ValueError("sequence must contain at least one state")
    if n != len(sequence.position_w) or n != len(sequence.rotation_bw) or n != len(sequence.velocity_w):
        raise ValueError("time, position, rotation, and velocity arrays must have the same length")
    if cfg.window_size < 2:
        raise ValueError("window_size must be at least 2")
    if cfg.stride < 1:
        raise ValueError("stride must be at least 1")

    position = np.asarray(sequence.position_w, dtype=np.float64).reshape(n, 3).copy()
    rotation = np.asarray(sequence.rotation_bw, dtype=np.float64).reshape(n, 3, 3).copy()
    velocity = np.asarray(sequence.velocity_w, dtype=np.float64).reshape(n, 3).copy()

    num_windows = 0
    total_nfev = 0
    total_cost = 0.0
    start = 0
    while start < n - 1:
        end = min(n - 1, start + int(cfg.window_size) - 1)
        if end <= start:
            break
        x0 = _pack_window(position, rotation, velocity, start, end)
        if x0.size > 0:
            result = least_squares(
                _window_residual,
                x0,
                args=(sequence, position, rotation, velocity, start, end, cfg),
                max_nfev=int(cfg.max_nfev),
                loss="linear",
                x_scale="jac",
            )
            p_win, R_win, v_win = _unpack_window(result.x, position, rotation, velocity, start, end)
            position[start : end + 1] = p_win
            rotation[start : end + 1] = R_win
            velocity[start : end + 1] = v_win
            total_nfev += int(result.nfev)
            total_cost += float(result.cost)
            num_windows += 1
        if end == n - 1:
            break
        start += int(cfg.stride)

    return SlidingWindowResult(
        time_ns=np.asarray(sequence.time_ns, dtype=np.int64).copy(),
        position_w=position,
        rotation_bw=rotation,
        velocity_w=velocity,
        num_windows=num_windows,
        total_nfev=total_nfev,
        total_cost=total_cost,
    )
