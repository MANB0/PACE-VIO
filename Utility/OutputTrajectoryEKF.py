"""Causal output-only EKF for smoothing XY position and world yaw.

The filter state is ``[x, y, vx, vy, yaw, yaw_rate]``. It consumes completed
VIO poses as measurements and must never feed its state back into the VIO
optimizer, marginal prior, bias state, or next-frame initialization.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


STATE_DOF = 6
MEASUREMENT_DOF = 3


def wrap_angle(angle: float | np.ndarray) -> float | np.ndarray:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


@dataclass(frozen=True)
class OutputEKFNoise:
    position_measurement_std: np.ndarray
    yaw_measurement_std: float
    linear_acceleration_process_std: np.ndarray
    yaw_acceleration_process_std: float

    def __post_init__(self) -> None:
        position = np.asarray(self.position_measurement_std, dtype=np.float64).reshape(2)
        acceleration = np.asarray(
            self.linear_acceleration_process_std, dtype=np.float64
        ).reshape(2)
        scalars = np.asarray(
            [self.yaw_measurement_std, self.yaw_acceleration_process_std],
            dtype=np.float64,
        )
        if not (
            np.isfinite(position).all()
            and np.isfinite(acceleration).all()
            and np.isfinite(scalars).all()
        ):
            raise ValueError("output EKF noise contains NaN/Inf")
        if (position <= 0.0).any() or (acceleration <= 0.0).any() or (scalars <= 0.0).any():
            raise ValueError("output EKF standard deviations must be positive")
        object.__setattr__(self, "position_measurement_std", position)
        object.__setattr__(self, "linear_acceleration_process_std", acceleration)


@dataclass(frozen=True)
class OutputEKFStepDiagnostics:
    innovation: np.ndarray
    innovation_covariance: np.ndarray
    correction: np.ndarray
    nis: float


class CausalPoseOutputEKF:
    """Linear constant-velocity filter with a wrapped yaw innovation."""

    def __init__(
        self,
        initial_xy: np.ndarray,
        initial_yaw: float,
        noise: OutputEKFNoise,
        *,
        initial_linear_velocity_std: float = 1.0,
        initial_yaw_rate_std: float = 1.0,
    ) -> None:
        self.noise = noise
        self.state = np.zeros(STATE_DOF, dtype=np.float64)
        self.state[0:2] = np.asarray(initial_xy, dtype=np.float64).reshape(2)
        self.state[4] = float(initial_yaw)
        standard_deviation = np.asarray(
            [
                noise.position_measurement_std[0],
                noise.position_measurement_std[1],
                initial_linear_velocity_std,
                initial_linear_velocity_std,
                noise.yaw_measurement_std,
                initial_yaw_rate_std,
            ],
            dtype=np.float64,
        )
        self.covariance = np.diag(np.square(standard_deviation))

    def _transition_and_process_covariance(
        self, dt: float
    ) -> tuple[np.ndarray, np.ndarray]:
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError(f"invalid output EKF dt={dt}")
        transition = np.eye(STATE_DOF, dtype=np.float64)
        transition[0, 2] = dt
        transition[1, 3] = dt
        transition[4, 5] = dt
        process = np.zeros((STATE_DOF, STATE_DOF), dtype=np.float64)
        white_acceleration = np.asarray(
            [[0.25 * dt**4, 0.5 * dt**3], [0.5 * dt**3, dt**2]],
            dtype=np.float64,
        )
        for position, velocity, sigma in (
            (0, 2, self.noise.linear_acceleration_process_std[0]),
            (1, 3, self.noise.linear_acceleration_process_std[1]),
            (4, 5, self.noise.yaw_acceleration_process_std),
        ):
            process[np.ix_([position, velocity], [position, velocity])] = (
                float(sigma) ** 2 * white_acceleration
            )
        return transition, process

    def predict(self, dt: float) -> None:
        transition, process = self._transition_and_process_covariance(float(dt))
        self.state = transition @ self.state
        self.covariance = transition @ self.covariance @ transition.T + process
        self.covariance = 0.5 * (self.covariance + self.covariance.T)

    def update(
        self, measured_xy: np.ndarray, measured_yaw: float
    ) -> OutputEKFStepDiagnostics:
        measurement = np.asarray(
            [*np.asarray(measured_xy, dtype=np.float64).reshape(2), float(measured_yaw)],
            dtype=np.float64,
        )
        observation = np.zeros((MEASUREMENT_DOF, STATE_DOF), dtype=np.float64)
        observation[0, 0] = 1.0
        observation[1, 1] = 1.0
        observation[2, 4] = 1.0
        measurement_covariance = np.diag(
            np.square(
                np.asarray(
                    [
                        self.noise.position_measurement_std[0],
                        self.noise.position_measurement_std[1],
                        self.noise.yaw_measurement_std,
                    ],
                    dtype=np.float64,
                )
            )
        )
        innovation = measurement - observation @ self.state
        innovation[2] = float(wrap_angle(innovation[2]))
        innovation_covariance = (
            observation @ self.covariance @ observation.T + measurement_covariance
        )
        gain = np.linalg.solve(
            innovation_covariance, observation @ self.covariance
        ).T
        correction = gain @ innovation
        self.state = self.state + correction
        self.state[4] = float(wrap_angle(self.state[4]))
        identity = np.eye(STATE_DOF, dtype=np.float64)
        joseph = identity - gain @ observation
        self.covariance = (
            joseph @ self.covariance @ joseph.T
            + gain @ measurement_covariance @ gain.T
        )
        self.covariance = 0.5 * (self.covariance + self.covariance.T)
        nis = float(innovation @ np.linalg.solve(innovation_covariance, innovation))
        if not (
            np.isfinite(self.state).all()
            and np.isfinite(self.covariance).all()
            and np.isfinite(nis)
        ):
            raise FloatingPointError("output EKF contains NaN/Inf")
        return OutputEKFStepDiagnostics(
            innovation=innovation.copy(),
            innovation_covariance=innovation_covariance.copy(),
            correction=correction.copy(),
            nis=nis,
        )

    def step(
        self, dt: float | None, measured_xy: np.ndarray, measured_yaw: float
    ) -> OutputEKFStepDiagnostics:
        if dt is not None:
            self.predict(float(dt))
        return self.update(measured_xy, measured_yaw)
