"""Causal output-only 3D ESKF for smoothing completed VIO poses.

The generic nominal state is ``{p_WL, v_WL, R_WL, omega_L}``, where ``L`` is
the local frame carried by the input pose.  The current replay input is
``T_WC``, so ``L=C``.  The historical ``*_WB`` field names below are internal
names only and must not be interpreted as an IMU/body-frame conversion.  This
module is deliberately independent of the VIO state, prior, preintegration,
and warm start.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


STATE_DOF = 12
POSITION = slice(0, 3)
VELOCITY = slice(3, 6)
ROTATION = slice(6, 9)
ANGULAR_VELOCITY = slice(9, 12)


def skew(vector: np.ndarray) -> np.ndarray:
    x, y, z = np.asarray(vector, dtype=np.float64).reshape(3)
    return np.asarray([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def so3_exp(rotvec: np.ndarray) -> np.ndarray:
    vector = np.asarray(rotvec, dtype=np.float64).reshape(3)
    theta = float(np.linalg.norm(vector))
    matrix = skew(vector)
    if theta < 1.0e-8:
        return np.eye(3) + matrix + 0.5 * matrix @ matrix
    a = np.sin(theta) / theta
    b = (1.0 - np.cos(theta)) / (theta * theta)
    return np.eye(3) + a * matrix + b * matrix @ matrix


def so3_log(rotation: np.ndarray) -> np.ndarray:
    matrix = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    cosine = float(np.clip((np.trace(matrix) - 1.0) * 0.5, -1.0, 1.0))
    theta = float(np.arccos(cosine))
    vee = np.asarray(
        [
            matrix[2, 1] - matrix[1, 2],
            matrix[0, 2] - matrix[2, 0],
            matrix[1, 0] - matrix[0, 1],
        ]
    )
    if theta < 1.0e-8:
        return 0.5 * vee
    if np.pi - theta < 1.0e-5:
        symmetric = 0.5 * (matrix + np.eye(3))
        axis = np.sqrt(np.maximum(np.diag(symmetric), 0.0))
        dominant = int(np.argmax(axis))
        if axis[dominant] < 1.0e-8:
            axis = np.asarray([1.0, 0.0, 0.0])
        else:
            for index in range(3):
                if index != dominant:
                    axis[index] = symmetric[dominant, index] / axis[dominant]
            axis /= np.linalg.norm(axis)
        return theta * axis
    return theta / (2.0 * np.sin(theta)) * vee


def right_jacobian_so3(rotvec: np.ndarray) -> np.ndarray:
    vector = np.asarray(rotvec, dtype=np.float64).reshape(3)
    theta = float(np.linalg.norm(vector))
    matrix = skew(vector)
    if theta < 1.0e-7:
        return np.eye(3) - 0.5 * matrix + matrix @ matrix / 6.0
    return (
        np.eye(3)
        - (1.0 - np.cos(theta)) / (theta * theta) * matrix
        + (theta - np.sin(theta)) / (theta**3) * matrix @ matrix
    )


def left_jacobian_inverse_so3(rotvec: np.ndarray) -> np.ndarray:
    vector = np.asarray(rotvec, dtype=np.float64).reshape(3)
    theta = float(np.linalg.norm(vector))
    matrix = skew(vector)
    if theta < 1.0e-7:
        return np.eye(3) - 0.5 * matrix + matrix @ matrix / 12.0
    coefficient = 1.0 / (theta * theta) - (
        (1.0 + np.cos(theta)) / (2.0 * theta * np.sin(theta))
    )
    return np.eye(3) - 0.5 * matrix + coefficient * matrix @ matrix


def project_rotation(rotation: np.ndarray) -> np.ndarray:
    left, _, right = np.linalg.svd(np.asarray(rotation, dtype=np.float64))
    result = left @ right
    if np.linalg.det(result) < 0.0:
        left[:, -1] *= -1.0
        result = left @ right
    return result


@dataclass
class OutputESKF3DState:
    position_W: np.ndarray
    velocity_W: np.ndarray
    rotation_WB: np.ndarray
    angular_velocity_B: np.ndarray

    def copy(self) -> "OutputESKF3DState":
        return OutputESKF3DState(
            self.position_W.copy(),
            self.velocity_W.copy(),
            self.rotation_WB.copy(),
            self.angular_velocity_B.copy(),
        )


def boxplus(state: OutputESKF3DState, delta: np.ndarray) -> OutputESKF3DState:
    increment = np.asarray(delta, dtype=np.float64).reshape(STATE_DOF)
    return OutputESKF3DState(
        position_W=state.position_W + increment[POSITION],
        velocity_W=state.velocity_W + increment[VELOCITY],
        rotation_WB=project_rotation(
            state.rotation_WB @ so3_exp(increment[ROTATION])
        ),
        angular_velocity_B=(
            state.angular_velocity_B + increment[ANGULAR_VELOCITY]
        ),
    )


def boxminus(
    state: OutputESKF3DState, reference: OutputESKF3DState
) -> np.ndarray:
    return np.concatenate(
        [
            state.position_W - reference.position_W,
            state.velocity_W - reference.velocity_W,
            so3_log(reference.rotation_WB.T @ state.rotation_WB),
            state.angular_velocity_B - reference.angular_velocity_B,
        ]
    )


@dataclass(frozen=True)
class OutputESKF3DNoise:
    position_measurement_std: np.ndarray
    rotation_measurement_std: np.ndarray
    linear_acceleration_process_std: np.ndarray
    angular_acceleration_process_std: np.ndarray

    def __post_init__(self) -> None:
        for name in (
            "position_measurement_std",
            "rotation_measurement_std",
            "linear_acceleration_process_std",
            "angular_acceleration_process_std",
        ):
            value = np.asarray(getattr(self, name), dtype=np.float64).reshape(3)
            if not np.isfinite(value).all() or (value <= 0.0).any():
                raise ValueError(f"{name} must contain three finite positive values")
            object.__setattr__(self, name, value)


@dataclass(frozen=True)
class OutputESKFGate:
    inflate_nis: float = 11.344866730144373  # chi-square(3), 99%
    reject_nis: float = 16.26623619623813  # chi-square(3), 99.9%
    maximum_inflation: float = 100.0
    position_reject_norm_m: float = 1.0
    rotation_reject_norm_rad: float = 0.5

    def __post_init__(self) -> None:
        if not (0.0 < self.inflate_nis < self.reject_nis):
            raise ValueError("gate thresholds must satisfy 0 < inflate < reject")
        if self.maximum_inflation < 1.0:
            raise ValueError("maximum_inflation must be at least one")
        if self.position_reject_norm_m <= 0.0 or self.rotation_reject_norm_rad <= 0.0:
            raise ValueError("absolute gate limits must be positive")


@dataclass(frozen=True)
class OutputESKF3DStepDiagnostics:
    position_innovation: np.ndarray
    rotation_innovation: np.ndarray
    position_nis: float
    rotation_nis: float
    position_action: str
    rotation_action: str
    correction: np.ndarray
    process_scale_before: float
    process_scale_after: float
    covariance_min_eigenvalue: float
    covariance_max_eigenvalue: float


def predict_nominal(
    state: OutputESKF3DState, dt: float
) -> OutputESKF3DState:
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError(f"invalid output ESKF dt={dt}")
    return OutputESKF3DState(
        position_W=state.position_W + state.velocity_W * dt,
        velocity_W=state.velocity_W.copy(),
        rotation_WB=project_rotation(
            state.rotation_WB @ so3_exp(state.angular_velocity_B * dt)
        ),
        angular_velocity_B=state.angular_velocity_B.copy(),
    )


def transition_jacobian(state: OutputESKF3DState, dt: float) -> np.ndarray:
    phi = state.angular_velocity_B * float(dt)
    delta_rotation = so3_exp(phi)
    jacobian = np.eye(STATE_DOF)
    jacobian[POSITION, VELOCITY] = np.eye(3) * dt
    jacobian[ROTATION, ROTATION] = delta_rotation.T
    jacobian[ROTATION, ANGULAR_VELOCITY] = right_jacobian_so3(phi) * dt
    return jacobian


def measurement_residual_and_jacobian(
    state: OutputESKF3DState,
    measured_position_W: np.ndarray,
    measured_rotation_WB: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    position_residual = (
        np.asarray(measured_position_W, dtype=np.float64).reshape(3)
        - state.position_W
    )
    rotation_residual = so3_log(
        state.rotation_WB.T
        @ np.asarray(measured_rotation_WB, dtype=np.float64).reshape(3, 3)
    )
    residual = np.concatenate([position_residual, rotation_residual])
    jacobian = np.zeros((6, STATE_DOF), dtype=np.float64)
    jacobian[0:3, POSITION] = np.eye(3)
    jacobian[3:6, ROTATION] = left_jacobian_inverse_so3(rotation_residual)
    return residual, jacobian


class CausalPoseOutputESKF3D:
    """3D kinematic output filter with optional block gating and adaptive Q."""

    def __init__(
        self,
        initial_position_W: np.ndarray,
        initial_rotation_WB: np.ndarray,
        noise: OutputESKF3DNoise,
        *,
        initial_velocity_W: np.ndarray | None = None,
        initial_angular_velocity_B: np.ndarray | None = None,
        initial_velocity_std: float = 1.0,
        initial_angular_velocity_std: float = 1.0,
        gate: OutputESKFGate | None = None,
        adaptive_process: bool = False,
    ) -> None:
        self.noise = noise
        self.gate = gate
        self.adaptive_process = bool(adaptive_process)
        self.process_scale = 1.0
        self.state = OutputESKF3DState(
            position_W=np.asarray(initial_position_W, dtype=np.float64).reshape(3),
            velocity_W=(
                np.zeros(3)
                if initial_velocity_W is None
                else np.asarray(initial_velocity_W, dtype=np.float64).reshape(3)
            ),
            rotation_WB=project_rotation(initial_rotation_WB),
            angular_velocity_B=(
                np.zeros(3)
                if initial_angular_velocity_B is None
                else np.asarray(initial_angular_velocity_B, dtype=np.float64).reshape(3)
            ),
        )
        standard_deviation = np.concatenate(
            [
                noise.position_measurement_std,
                np.full(3, float(initial_velocity_std)),
                noise.rotation_measurement_std,
                np.full(3, float(initial_angular_velocity_std)),
            ]
        )
        self.covariance = np.diag(np.square(standard_deviation))

    def _process_covariance(self, dt: float) -> np.ndarray:
        gain = np.zeros((STATE_DOF, 6), dtype=np.float64)
        gain[POSITION, 0:3] = np.eye(3) * (0.5 * dt * dt)
        gain[VELOCITY, 0:3] = np.eye(3) * dt
        phi = self.state.angular_velocity_B * dt
        gain[ROTATION, 3:6] = (
            right_jacobian_so3(phi) * (0.5 * dt * dt)
        )
        gain[ANGULAR_VELOCITY, 3:6] = np.eye(3) * dt
        standard_deviation = np.concatenate(
            [
                self.noise.linear_acceleration_process_std,
                self.noise.angular_acceleration_process_std,
            ]
        ) * self.process_scale
        return gain @ np.diag(np.square(standard_deviation)) @ gain.T

    def predict(self, dt: float) -> None:
        dt = float(dt)
        jacobian = transition_jacobian(self.state, dt)
        process = self._process_covariance(dt)
        self.state = predict_nominal(self.state, dt)
        self.covariance = jacobian @ self.covariance @ jacobian.T + process
        self.covariance = 0.5 * (self.covariance + self.covariance.T)

    @staticmethod
    def _block_nis(
        residual: np.ndarray, covariance: np.ndarray
    ) -> float:
        return float(residual @ np.linalg.solve(covariance, residual))

    def _gate_block(
        self, nis: float, residual_norm: float, absolute_reject_norm: float
    ) -> tuple[str, float]:
        if self.gate is None:
            return "accept", 1.0
        if nis > self.gate.reject_nis and residual_norm > absolute_reject_norm:
            return "reject", 1.0
        if nis > self.gate.inflate_nis:
            inflation = min(
                max(nis / self.gate.inflate_nis, 1.0),
                self.gate.maximum_inflation,
            )
            return "inflate", inflation
        return "accept", 1.0

    def update(
        self,
        measured_position_W: np.ndarray,
        measured_rotation_WB: np.ndarray,
    ) -> OutputESKF3DStepDiagnostics:
        residual, measurement_jacobian = measurement_residual_and_jacobian(
            self.state, measured_position_W, measured_rotation_WB
        )
        measurement_covariance = np.diag(
            np.square(
                np.concatenate(
                    [
                        self.noise.position_measurement_std,
                        self.noise.rotation_measurement_std,
                    ]
                )
            )
        )
        innovation_covariance = (
            measurement_jacobian @ self.covariance @ measurement_jacobian.T
            + measurement_covariance
        )
        position_nis = self._block_nis(
            residual[0:3], innovation_covariance[0:3, 0:3]
        )
        rotation_nis = self._block_nis(
            residual[3:6], innovation_covariance[3:6, 3:6]
        )
        if self.gate is None:
            position_reject_norm = np.inf
            rotation_reject_norm = np.inf
        else:
            position_reject_norm = self.gate.position_reject_norm_m
            rotation_reject_norm = self.gate.rotation_reject_norm_rad
        position_action, position_inflation = self._gate_block(
            position_nis,
            float(np.linalg.norm(residual[0:3])),
            position_reject_norm,
        )
        rotation_action, rotation_inflation = self._gate_block(
            rotation_nis,
            float(np.linalg.norm(residual[3:6])),
            rotation_reject_norm,
        )
        selected: list[int] = []
        if position_action != "reject":
            selected.extend(range(3))
            measurement_covariance[0:3, 0:3] *= position_inflation
        if rotation_action != "reject":
            selected.extend(range(3, 6))
            measurement_covariance[3:6, 3:6] *= rotation_inflation

        process_scale_before = self.process_scale
        correction = np.zeros(STATE_DOF)
        if selected:
            index = np.asarray(selected, dtype=np.int64)
            active_residual = residual[index]
            active_jacobian = measurement_jacobian[index]
            active_measurement_covariance = measurement_covariance[np.ix_(index, index)]
            active_innovation_covariance = (
                active_jacobian @ self.covariance @ active_jacobian.T
                + active_measurement_covariance
            )
            gain = np.linalg.solve(
                active_innovation_covariance,
                active_jacobian @ self.covariance,
            ).T
            correction = gain @ active_residual
            self.state = boxplus(self.state, correction)
            identity = np.eye(STATE_DOF)
            joseph = identity - gain @ active_jacobian
            self.covariance = (
                joseph @ self.covariance @ joseph.T
                + gain @ active_measurement_covariance @ gain.T
            )
            self.covariance = 0.5 * (self.covariance + self.covariance.T)

        if self.adaptive_process:
            ratios = []
            if position_action != "reject":
                ratios.append(position_nis / 3.0)
            if rotation_action != "reject":
                ratios.append(rotation_nis / 3.0)
            target = np.clip(np.sqrt(max(ratios, default=1.0)), 0.35, 4.0)
            self.process_scale = float(
                np.clip(0.9 * self.process_scale + 0.1 * target, 0.25, 8.0)
            )

        state_values = np.concatenate(
            [
                self.state.position_W,
                self.state.velocity_W,
                self.state.rotation_WB.reshape(-1),
                self.state.angular_velocity_B,
            ]
        )
        eigenvalues = np.linalg.eigvalsh(self.covariance)
        if not (
            np.isfinite(state_values).all()
            and np.isfinite(self.covariance).all()
            and np.isfinite(eigenvalues).all()
        ):
            raise FloatingPointError("output ESKF contains NaN/Inf")
        if eigenvalues.min() < -1.0e-10:
            raise FloatingPointError("output ESKF covariance is not PSD")
        return OutputESKF3DStepDiagnostics(
            position_innovation=residual[0:3].copy(),
            rotation_innovation=residual[3:6].copy(),
            position_nis=position_nis,
            rotation_nis=rotation_nis,
            position_action=position_action,
            rotation_action=rotation_action,
            correction=correction.copy(),
            process_scale_before=process_scale_before,
            process_scale_after=self.process_scale,
            covariance_min_eigenvalue=float(eigenvalues.min()),
            covariance_max_eigenvalue=float(eigenvalues.max()),
        )

    def step(
        self,
        dt: float | None,
        measured_position_W: np.ndarray,
        measured_rotation_WB: np.ndarray,
    ) -> OutputESKF3DStepDiagnostics:
        if dt is not None:
            self.predict(float(dt))
        return self.update(measured_position_W, measured_rotation_WB)
