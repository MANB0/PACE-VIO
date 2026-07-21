"""Right-error stochastic-clone ESKF for relative body-pose measurements.

The navigation error order is ``[dp_W, dv_W, dtheta_B, dba, dbg]`` and the
pose clone adds ``[dp_clone_W, dtheta_clone_B]``. Rotations use right
perturbations, while positions and velocities use additive world-frame errors.
The visual innovation is ``Log(Z_BiBj^-1 * (T_WBi^-1 * T_WBj))`` in
``[translation, rotation]`` order.
"""

from __future__ import annotations

from dataclasses import dataclass

import pypose as pp
import torch


NAV_DOF = 15
CLONE_DOF = 6
AUGMENTED_DOF = NAV_DOF + CLONE_DOF


def _skew(vector: torch.Tensor) -> torch.Tensor:
    x, y, z = vector.reshape(3)
    zero = torch.zeros((), dtype=vector.dtype, device=vector.device)
    return torch.stack(
        [zero, -z, y, z, zero, -x, -y, x, zero]
    ).reshape(3, 3)


def _symmetrize(matrix: torch.Tensor) -> torch.Tensor:
    return 0.5 * (matrix + matrix.mT)


def _solve_spd(matrix: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    matrix = _symmetrize(matrix)
    values, vectors = torch.linalg.eigh(matrix)
    scale = max(float(values.abs().max().item()), 1.0)
    floor = torch.finfo(matrix.dtype).eps * scale
    stabilized = vectors @ torch.diag(values.clamp_min(floor)) @ vectors.mT
    lower = torch.linalg.cholesky(_symmetrize(stabilized))
    return torch.cholesky_solve(right, lower)


def _project_psd(matrix: torch.Tensor, *, tolerance: float = 1.0e-9) -> torch.Tensor:
    matrix = _symmetrize(matrix)
    values, vectors = torch.linalg.eigh(matrix)
    if float(values.min().item()) < -float(tolerance):
        raise FloatingPointError(
            f"ESKF covariance lost positive semidefiniteness: min_eig={values.min().item():.3e}"
        )
    return _symmetrize(vectors @ torch.diag(values.clamp_min(0.0)) @ vectors.mT)


@dataclass(frozen=True)
class ESKFNoiseDensities:
    accel: torch.Tensor
    gyro: torch.Tensor
    accel_bias_rw: torch.Tensor
    gyro_bias_rw: torch.Tensor

    def to(self, *, dtype: torch.dtype, device: torch.device) -> "ESKFNoiseDensities":
        return ESKFNoiseDensities(
            accel=self.accel.reshape(3).to(dtype=dtype, device=device),
            gyro=self.gyro.reshape(3).to(dtype=dtype, device=device),
            accel_bias_rw=self.accel_bias_rw.reshape(3).to(dtype=dtype, device=device),
            gyro_bias_rw=self.gyro_bias_rw.reshape(3).to(dtype=dtype, device=device),
        )


@dataclass(frozen=True)
class ESKFNominalState:
    pose_WB: torch.Tensor
    velocity_W: torch.Tensor
    acc_bias: torch.Tensor
    gyro_bias: torch.Tensor

    def to(self, *, dtype: torch.dtype, device: torch.device) -> "ESKFNominalState":
        return ESKFNominalState(
            pose_WB=self.pose_WB.reshape(1, 7).to(dtype=dtype, device=device),
            velocity_W=self.velocity_W.reshape(3).to(dtype=dtype, device=device),
            acc_bias=self.acc_bias.reshape(3).to(dtype=dtype, device=device),
            gyro_bias=self.gyro_bias.reshape(3).to(dtype=dtype, device=device),
        )


@dataclass(frozen=True)
class PoseClone:
    pose_WB: torch.Tensor
    timestamp_ns: int


@dataclass(frozen=True)
class StochasticCloneState:
    nominal: ESKFNominalState
    clone: PoseClone
    covariance: torch.Tensor
    timestamp_ns: int


@dataclass(frozen=True)
class RelativePoseUpdateDiagnostics:
    residual: torch.Tensor
    innovation_covariance: torch.Tensor
    kalman_gain: torch.Tensor
    increment: torch.Tensor
    nis: float
    residual_norm_before: float
    residual_norm_after: float
    covariance_min_eigenvalue: float
    covariance_max_eigenvalue: float
    finite: bool


def initial_navigation_covariance(
    *,
    pose_translation_std: float,
    pose_rotation_std: float,
    velocity_std: float,
    acc_bias_std: float,
    gyro_bias_std: float,
    dtype: torch.dtype = torch.float64,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    std = torch.tensor(
        [pose_translation_std] * 3
        + [velocity_std] * 3
        + [pose_rotation_std] * 3
        + [acc_bias_std] * 3
        + [gyro_bias_std] * 3,
        dtype=dtype,
        device=device,
    )
    if bool((std <= 0.0).any()):
        raise ValueError("all initial ESKF standard deviations must be positive")
    return torch.diag(std.square())


def augment_with_current_pose_clone(
    nominal: ESKFNominalState,
    covariance_nav: torch.Tensor,
    timestamp_ns: int,
) -> StochasticCloneState:
    covariance_nav = covariance_nav.reshape(NAV_DOF, NAV_DOF)
    selector = torch.zeros(
        (CLONE_DOF, NAV_DOF),
        dtype=covariance_nav.dtype,
        device=covariance_nav.device,
    )
    selector[0:3, 0:3] = torch.eye(3, dtype=selector.dtype, device=selector.device)
    selector[3:6, 6:9] = torch.eye(3, dtype=selector.dtype, device=selector.device)
    covariance = torch.cat(
        [
            torch.cat([covariance_nav, covariance_nav @ selector.mT], dim=1),
            torch.cat([selector @ covariance_nav, selector @ covariance_nav @ selector.mT], dim=1),
        ],
        dim=0,
    )
    return StochasticCloneState(
        nominal=nominal,
        clone=PoseClone(nominal.pose_WB.detach().clone(), int(timestamp_ns)),
        covariance=_project_psd(covariance),
        timestamp_ns=int(timestamp_ns),
    )


def reclone_current_pose(state: StochasticCloneState) -> StochasticCloneState:
    return augment_with_current_pose_clone(
        state.nominal,
        state.covariance[0:NAV_DOF, 0:NAV_DOF],
        state.timestamp_ns,
    )


def _inject_augmented_error(
    state: StochasticCloneState,
    increment: torch.Tensor,
) -> StochasticCloneState:
    increment = increment.reshape(AUGMENTED_DOF)
    current_pose = pp.SE3(state.nominal.pose_WB)
    current_rotation = current_pose.rotation() @ pp.so3(
        increment[6:9].reshape(1, 3)
    ).Exp()
    pose = pp.SE3(
        torch.cat(
            [
                current_pose.translation().reshape(3) + increment[0:3],
                current_rotation.tensor().reshape(4),
            ]
        ).reshape(1, 7)
    )
    old_clone_pose = pp.SE3(state.clone.pose_WB)
    clone_rotation = old_clone_pose.rotation() @ pp.so3(
        increment[18:21].reshape(1, 3)
    ).Exp()
    clone_pose = pp.SE3(
        torch.cat(
            [
                old_clone_pose.translation().reshape(3) + increment[15:18],
                clone_rotation.tensor().reshape(4),
            ]
        ).reshape(1, 7)
    )
    return StochasticCloneState(
        nominal=ESKFNominalState(
            pose_WB=pose.tensor(),
            velocity_W=state.nominal.velocity_W + increment[3:6],
            acc_bias=state.nominal.acc_bias + increment[9:12],
            gyro_bias=state.nominal.gyro_bias + increment[12:15],
        ),
        clone=PoseClone(clone_pose.tensor(), state.clone.timestamp_ns),
        covariance=state.covariance,
        timestamp_ns=state.timestamp_ns,
    )


def relative_pose_residual(
    state: StochasticCloneState,
    measurement_BiBj: torch.Tensor,
) -> torch.Tensor:
    predicted = pp.SE3(state.clone.pose_WB).Inv() @ pp.SE3(state.nominal.pose_WB)
    return (
        pp.SE3(measurement_BiBj.reshape(1, 7)).Inv() @ predicted
    ).Log().tensor().reshape(6)


def relative_pose_jacobian_central(
    state: StochasticCloneState,
    measurement_BiBj: torch.Tensor,
    *,
    epsilon: float = 1.0e-6,
) -> torch.Tensor:
    columns: list[torch.Tensor] = []
    for column in range(AUGMENTED_DOF):
        increment = torch.zeros(
            AUGMENTED_DOF,
            dtype=state.covariance.dtype,
            device=state.covariance.device,
        )
        increment[column] = float(epsilon)
        plus = relative_pose_residual(
            _inject_augmented_error(state, increment), measurement_BiBj
        )
        minus = relative_pose_residual(
            _inject_augmented_error(state, -increment), measurement_BiBj
        )
        columns.append((plus - minus) / (2.0 * float(epsilon)))
    return torch.stack(columns, dim=1)


def propagate_imu_knots(
    state: StochasticCloneState,
    time_ns: torch.Tensor,
    acc_body: torch.Tensor,
    gyro_body: torch.Tensor,
    gravity_world: torch.Tensor,
    noise: ESKFNoiseDensities,
    *,
    maximum_dt_s: float = 0.05,
) -> StochasticCloneState:
    timestamps = time_ns.reshape(-1).long()
    acc = acc_body.reshape(-1, 3).to(state.covariance)
    gyro = gyro_body.reshape(-1, 3).to(state.covariance)
    if timestamps.numel() != acc.shape[0] or acc.shape != gyro.shape:
        raise ValueError("IMU knot time/acc/gyro shapes differ")
    if timestamps.numel() < 2:
        return state

    nominal = state.nominal.to(dtype=state.covariance.dtype, device=state.covariance.device)
    pose = pp.SE3(nominal.pose_WB)
    position = pose.translation().reshape(3)
    rotation = pose.rotation()
    velocity = nominal.velocity_W.reshape(3)
    covariance = state.covariance
    gravity = gravity_world.reshape(3).to(covariance)
    noise = noise.to(dtype=covariance.dtype, device=covariance.device)
    process_density = torch.diag(
        torch.cat(
            [
                noise.accel.square(),
                noise.gyro.square(),
                noise.accel_bias_rw.square(),
                noise.gyro_bias_rw.square(),
            ]
        )
    )

    for index in range(timestamps.numel() - 1):
        dt = float((timestamps[index + 1] - timestamps[index]).item()) * 1.0e-9
        if not (0.0 < dt <= float(maximum_dt_s)):
            raise ValueError(f"invalid ESKF propagation dt={dt:.9g}s")
        omega = 0.5 * (gyro[index] + gyro[index + 1]) - nominal.gyro_bias
        specific_force = 0.5 * (acc[index] + acc[index + 1]) - nominal.acc_bias
        half_rotation = rotation @ pp.so3((0.5 * omega * dt).reshape(1, 3)).Exp()
        rotation_matrix = half_rotation.matrix().reshape(3, 3)
        acceleration_world = rotation_matrix @ specific_force + gravity
        position = position + velocity * dt + 0.5 * acceleration_world * (dt * dt)
        velocity = velocity + acceleration_world * dt
        rotation = rotation @ pp.so3((omega * dt).reshape(1, 3)).Exp()

        continuous = torch.zeros(
            (AUGMENTED_DOF, AUGMENTED_DOF),
            dtype=covariance.dtype,
            device=covariance.device,
        )
        continuous[0:3, 3:6] = torch.eye(3, dtype=covariance.dtype, device=covariance.device)
        continuous[3:6, 6:9] = -rotation_matrix @ _skew(specific_force)
        continuous[3:6, 9:12] = -rotation_matrix
        continuous[6:9, 6:9] = -_skew(omega)
        continuous[6:9, 12:15] = -torch.eye(3, dtype=covariance.dtype, device=covariance.device)
        transition = torch.eye(
            AUGMENTED_DOF, dtype=covariance.dtype, device=covariance.device
        ) + continuous * dt

        noise_map = torch.zeros(
            (AUGMENTED_DOF, 12), dtype=covariance.dtype, device=covariance.device
        )
        noise_map[3:6, 0:3] = -rotation_matrix
        noise_map[6:9, 3:6] = -torch.eye(3, dtype=covariance.dtype, device=covariance.device)
        noise_map[9:12, 6:9] = torch.eye(3, dtype=covariance.dtype, device=covariance.device)
        noise_map[12:15, 9:12] = torch.eye(3, dtype=covariance.dtype, device=covariance.device)
        covariance = transition @ covariance @ transition.mT
        covariance = covariance + noise_map @ process_density @ noise_map.mT * dt
        covariance = _project_psd(covariance)

    propagated_pose = pp.SE3(
        torch.cat([position, rotation.tensor().reshape(4)]).reshape(1, 7)
    )
    return StochasticCloneState(
        nominal=ESKFNominalState(
            pose_WB=propagated_pose.tensor(),
            velocity_W=velocity,
            acc_bias=nominal.acc_bias,
            gyro_bias=nominal.gyro_bias,
        ),
        clone=state.clone,
        covariance=covariance,
        timestamp_ns=int(timestamps[-1].item()),
    )


def update_relative_pose(
    state: StochasticCloneState,
    measurement_BiBj: torch.Tensor,
    covariance_measurement: torch.Tensor,
    *,
    finite_difference_epsilon: float = 1.0e-6,
) -> tuple[StochasticCloneState, RelativePoseUpdateDiagnostics]:
    measurement = measurement_BiBj.reshape(1, 7).to(state.covariance)
    measurement_covariance = _symmetrize(
        covariance_measurement.reshape(6, 6).to(state.covariance)
    )
    residual = relative_pose_residual(state, measurement)
    jacobian = relative_pose_jacobian_central(
        state, measurement, epsilon=finite_difference_epsilon
    )
    innovation = _symmetrize(
        jacobian @ state.covariance @ jacobian.mT + measurement_covariance
    )
    gain = _solve_spd(
        innovation,
        jacobian @ state.covariance,
    ).mT
    increment = -gain @ residual

    identity = torch.eye(
        AUGMENTED_DOF,
        dtype=state.covariance.dtype,
        device=state.covariance.device,
    )
    correction = identity - gain @ jacobian
    covariance = correction @ state.covariance @ correction.mT
    covariance = covariance + gain @ measurement_covariance @ gain.mT
    covariance = _project_psd(covariance)

    updated = _inject_augmented_error(state, increment)
    reset = torch.eye(
        AUGMENTED_DOF,
        dtype=state.covariance.dtype,
        device=state.covariance.device,
    )
    reset[6:9, 6:9] = (
        torch.eye(3, dtype=reset.dtype, device=reset.device)
        - 0.5 * _skew(increment[6:9])
    )
    reset[18:21, 18:21] = (
        torch.eye(3, dtype=reset.dtype, device=reset.device)
        - 0.5 * _skew(increment[18:21])
    )
    covariance = _project_psd(reset @ covariance @ reset.mT)
    updated = StochasticCloneState(
        nominal=updated.nominal,
        clone=updated.clone,
        covariance=covariance,
        timestamp_ns=updated.timestamp_ns,
    )
    residual_after = relative_pose_residual(updated, measurement)
    nis = float((residual @ _solve_spd(innovation, residual.reshape(6, 1)).reshape(6)).item())
    eigenvalues = torch.linalg.eigvalsh(covariance)
    finite = bool(
        torch.isfinite(residual).all()
        and torch.isfinite(increment).all()
        and torch.isfinite(covariance).all()
    )
    if not finite:
        raise FloatingPointError("relative-pose clone ESKF update contains NaN/Inf")
    return updated, RelativePoseUpdateDiagnostics(
        residual=residual.detach().clone(),
        innovation_covariance=innovation.detach().clone(),
        kalman_gain=gain.detach().clone(),
        increment=increment.detach().clone(),
        nis=nis,
        residual_norm_before=float(torch.linalg.vector_norm(residual).item()),
        residual_norm_after=float(torch.linalg.vector_norm(residual_after).item()),
        covariance_min_eigenvalue=float(eigenvalues.min().item()),
        covariance_max_eigenvalue=float(eigenvalues.max().item()),
        finite=finite,
    )
