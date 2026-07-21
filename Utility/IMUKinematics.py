from __future__ import annotations

from dataclasses import dataclass

import torch
import pypose as pp


TRANSLATION_PRIOR_MODES = {
    "off",
    "damping_delta_p",
    "visual_velocity_composed",
    "imu_velocity_composed",
}

LEGACY_EXTERNAL_ATTITUDE_GRAVITY_COMPENSATION = (
    "legacy_external_attitude_gravity_compensation"
)
STANDARD_LOCAL_FRAME_PREINTEGRATION = "standard_local_frame_preintegration"

GRAVITY_HANDLING_ALIASES = {
    "preintegration": LEGACY_EXTERNAL_ATTITUDE_GRAVITY_COMPENSATION,
    LEGACY_EXTERNAL_ATTITUDE_GRAVITY_COMPENSATION: (
        LEGACY_EXTERNAL_ATTITUDE_GRAVITY_COMPENSATION
    ),
    "residual": STANDARD_LOCAL_FRAME_PREINTEGRATION,
    STANDARD_LOCAL_FRAME_PREINTEGRATION: STANDARD_LOCAL_FRAME_PREINTEGRATION,
}

GRAVITY_HANDLING_MODES = set(GRAVITY_HANDLING_ALIASES)


def normalize_gravity_handling(mode: str | None) -> str:
    normalized = str(mode or "preintegration").strip().lower()
    if normalized not in GRAVITY_HANDLING_ALIASES:
        raise ValueError(
            f"Unsupported gravity_handling={mode!r}; expected one of "
            f"{sorted(GRAVITY_HANDLING_MODES)}"
        )
    return GRAVITY_HANDLING_ALIASES[normalized]


def gravity_is_legacy_external_attitude(mode: str | None) -> bool:
    return normalize_gravity_handling(mode) == LEGACY_EXTERNAL_ATTITUDE_GRAVITY_COMPENSATION


def gravity_is_standard_local_frame(mode: str | None) -> bool:
    return normalize_gravity_handling(mode) == STANDARD_LOCAL_FRAME_PREINTEGRATION


def transform_imu_samples_to_internal_frame(
    acc: torch.Tensor,
    gyro: torch.Tensor,
    imu_T_BS: pp.LieTensor | torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotate IMU samples from sensor measurement frame into MACVO's body frame.

    ``imu_T_BS`` follows the project-wide body-to-sensor convention, so sensor
    measurements are mapped back to body/internal coordinates with the inverse
    rotation.
    """
    acc_samples = acc.reshape(-1, 3)
    gyro_samples = gyro.reshape(-1, 3)
    transform = pp.SE3(
        torch.as_tensor(
            imu_T_BS.tensor() if isinstance(imu_T_BS, pp.LieTensor) else imu_T_BS,
            device=acc_samples.device,
            dtype=torch.float64,
        ).reshape(7)
    )
    body_to_sensor_rotation = transform.rotation().matrix().reshape(3, 3)
    identity = torch.eye(3, device=body_to_sensor_rotation.device, dtype=body_to_sensor_rotation.dtype)
    if torch.allclose(body_to_sensor_rotation, identity, atol=1e-6):
        return acc_samples.clone(), gyro_samples.clone()

    sensor_to_body_rotation = body_to_sensor_rotation.transpose(0, 1)
    acc_rotation = sensor_to_body_rotation.to(device=acc_samples.device, dtype=acc_samples.dtype)
    gyro_rotation = sensor_to_body_rotation.to(device=gyro_samples.device, dtype=gyro_samples.dtype)
    acc_internal = (acc_rotation @ acc_samples.unsqueeze(-1)).squeeze(-1)
    gyro_internal = (gyro_rotation @ gyro_samples.unsqueeze(-1)).squeeze(-1)
    return acc_internal, gyro_internal


def _normalized_imu_sigma_unit(sigma_unit: str | None) -> str:
    unit = "legacy_sqrt_rate_scaled" if sigma_unit is None else str(sigma_unit).strip().lower()
    return unit.replace("_", " ").replace("-", " ")


def _convert_imu_sigma_value(
    sigma_value: float | list[float] | tuple[float, ...] | torch.Tensor,
    scale: float,
) -> float | tuple[float, float, float]:
    values = torch.as_tensor(sigma_value, dtype=torch.float64).reshape(-1)
    if values.numel() not in {1, 3}:
        raise ValueError(
            f"IMU sigma must contain one isotropic value or three axis values, got {values.numel()}"
        )
    converted = values * float(scale)
    if converted.numel() == 1:
        return float(converted.item())
    return tuple(float(value) for value in converted.tolist())


def imu_sigma_rms(
    sigma_value: float | list[float] | tuple[float, ...] | torch.Tensor,
) -> float:
    """Return one representative standard deviation without discarding axes."""
    values = torch.as_tensor(sigma_value, dtype=torch.float64).reshape(-1)
    if values.numel() not in {1, 3}:
        raise ValueError(
            f"IMU sigma must contain one isotropic value or three axis values, got {values.numel()}"
        )
    return float(torch.sqrt(torch.mean(values.square())).item())


def is_valid_imu_sigma(
    sigma_value: float | list[float] | tuple[float, ...] | torch.Tensor,
) -> bool:
    try:
        values = torch.as_tensor(sigma_value, dtype=torch.float64).reshape(-1)
    except (TypeError, ValueError):
        return False
    return bool(
        values.numel() in {1, 3}
        and torch.isfinite(values).all().item()
        and (values >= 0.0).all().item()
    )


def format_imu_sigma(
    sigma_value: float | list[float] | tuple[float, ...] | torch.Tensor | None,
) -> str:
    if sigma_value is None:
        return "unset"
    values = torch.as_tensor(sigma_value, dtype=torch.float64).reshape(-1)
    if values.numel() == 1:
        return f"{float(values.item()):.7g}"
    return "[" + ", ".join(f"{float(value):.7g}" for value in values.tolist()) + "]"


def imu_sigma_to_continuous_density(
    sigma_value: float | list[float] | tuple[float, ...] | torch.Tensor,
    rate_hz: float,
    sigma_unit: str | None = None,
) -> float | tuple[float, float, float]:
    """Convert IMU metadata sigma values to continuous noise density.

    Existing datasets without an explicit unit keep the historical sqrt(rate)
    scaling. HoloOcean metadata marks its values as per-sample standard
    deviations, which need the inverse conversion.
    """
    rate = max(float(rate_hz), 1e-9)
    unit = _normalized_imu_sigma_unit(sigma_unit)

    if "per" in unit and "sample" in unit:
        return _convert_imu_sigma_value(sigma_value, 1.0 / (rate ** 0.5))
    if "continuous" in unit or "noise density" in unit:
        return _convert_imu_sigma_value(sigma_value, 1.0)
    if "legacy" in unit and "sqrt" in unit:
        return _convert_imu_sigma_value(sigma_value, rate ** 0.5)
    raise ValueError(
        f"Unsupported IMU sigma_unit={sigma_unit!r}; expected per-sample, "
        "continuous noise density, or legacy sqrt-rate scaling"
    )


def imu_bias_sigma_to_continuous_random_walk_density(
    sigma_value: float | list[float] | tuple[float, ...] | torch.Tensor,
    rate_hz: float,
    sigma_unit: str | None = None,
) -> float | tuple[float, float, float]:
    """Convert bias random-walk metadata sigma to continuous density.

    HoloOcean applies ``AccelBiasSigma`` and ``AngVelBiasSigma`` as a per-tick
    random-walk increment on the hidden sensor bias. If the metadata reports
    that per-sample increment standard deviation, the equivalent continuous
    random-walk density is ``sigma_step / sqrt(dt)``.
    """
    rate = max(float(rate_hz), 1e-9)
    unit = _normalized_imu_sigma_unit(sigma_unit)

    if "per" in unit and ("sample" in unit or "tick" in unit):
        return _convert_imu_sigma_value(sigma_value, rate ** 0.5)
    if "continuous" in unit or "noise density" in unit or "random walk density" in unit:
        return _convert_imu_sigma_value(sigma_value, 1.0)
    if "legacy" in unit and "sqrt" in unit:
        return _convert_imu_sigma_value(sigma_value, rate ** 0.5)
    raise ValueError(
        f"Unsupported IMU bias sigma_unit={sigma_unit!r}; expected per-sample bias "
        "increment, continuous random-walk density, or legacy sqrt-rate scaling"
    )


def gravity_for_world_frame(gravity: float, world_frame: str | None) -> float:
    """Return the signed z-axis gravity value for the preintegrator.

    The preintegrator stores gravity as ``[0, 0, gravity]`` in the world
    frame. NED is z-down, so gravity is positive. NWU/ENU are z-up, so gravity
    is negative.
    """
    g = abs(float(gravity))
    frame = str(world_frame or "NED").strip().upper()
    if "NWU" in frame or "ENU" in frame or "Z-UP" in frame:
        return -g
    if "NED" in frame or "Z-DOWN" in frame:
        return g
    raise ValueError(f"Unsupported IMU world_frame={world_frame!r}; expected NWU/ENU or NED")


@dataclass(frozen=True)
class GravityRollPitchAlignment:
    rotation: pp.LieTensor
    active: bool
    correction_angle_rad: float
    acc_norm: float


@dataclass(frozen=True)
class StaticImuInitialization:
    body_to_world: pp.LieTensor
    acc_bias: torch.Tensor
    gyro_bias: torch.Tensor
    duration_s: float
    sample_count: int
    acc_mean: torch.Tensor
    gyro_mean: torch.Tensor
    acc_std: torch.Tensor
    gyro_std: torch.Tensor
    stationary: bool
    failure_reason: str = ""


def _minimal_rotation_between_unit_vectors(
    source: torch.Tensor,
    target: torch.Tensor,
) -> tuple[pp.LieTensor, float]:
    source = source.reshape(3)
    target = target.reshape(3).to(source)
    dot = torch.dot(source, target).clamp(-1.0, 1.0)
    axis = torch.cross(source, target, dim=0)
    axis_norm = axis.norm()

    if float(axis_norm.item()) < 1e-12:
        if float(dot.item()) > 0.0:
            return pp.identity_SO3(device=source.device, dtype=source.dtype), 0.0
        basis = torch.tensor([1.0, 0.0, 0.0], device=source.device, dtype=source.dtype)
        if float(torch.cross(source, basis, dim=0).norm().item()) < 1e-6:
            basis = torch.tensor([0.0, 1.0, 0.0], device=source.device, dtype=source.dtype)
        axis = torch.cross(source, basis, dim=0)
        axis = axis / axis.norm().clamp(min=1e-12)
        angle = torch.tensor(torch.pi, device=source.device, dtype=source.dtype)
    else:
        axis = axis / axis_norm
        angle = torch.atan2(axis_norm, dot)

    return pp.so3((axis * angle).reshape(1, 3)).Exp(), float(angle.item())


def gravity_roll_pitch_aligned_rotation(
    estimated_body_to_world: pp.LieTensor | torch.Tensor,
    acc_body: torch.Tensor,
    gravity: float,
    *,
    correction_gain: float = 1.0,
    acc_norm_tol: float = 0.15,
    window: int | None = None,
) -> GravityRollPitchAlignment:
    """Correct roll/pitch by aligning measured specific force with world up.

    The accelerometer is used only as a gravity-direction cue. The correction is
    gated by acceleration norm so translational acceleration does not become an
    attitude measurement.
    """
    R_est = pp.SO3(estimated_body_to_world)
    dtype = R_est.tensor().dtype
    device = R_est.tensor().device
    acc = acc_body.reshape(-1, 3).to(device=device, dtype=dtype)
    if acc.numel() == 0:
        return GravityRollPitchAlignment(R_est, False, 0.0, 0.0)
    if window is not None and int(window) > 0:
        acc = acc[-int(window):]

    acc_mean = acc.mean(dim=0)
    acc_norm = acc_mean.norm()
    acc_norm_value = float(acc_norm.item())
    g_abs = abs(float(gravity))
    if g_abs < 1e-9 or acc_norm_value < 1e-9 or float(correction_gain) <= 0.0:
        return GravityRollPitchAlignment(R_est, False, 0.0, acc_norm_value)
    if abs(acc_norm_value - g_abs) / g_abs > float(acc_norm_tol):
        return GravityRollPitchAlignment(R_est, False, 0.0, acc_norm_value)

    measured_up_body = acc_mean / acc_norm.clamp(min=1e-12)
    estimated_up_world = R_est.Act(measured_up_body)
    estimated_up_world = estimated_up_world.reshape(3)
    estimated_up_world = estimated_up_world / estimated_up_world.norm().clamp(min=1e-12)

    target_up_world = torch.tensor(
        [0.0, 0.0, -float(gravity) / g_abs],
        device=device,
        dtype=dtype,
    )
    correction, angle = _minimal_rotation_between_unit_vectors(
        estimated_up_world,
        target_up_world,
    )
    gain = min(max(float(correction_gain), 0.0), 1.0)
    if angle > 0.0 and gain < 1.0:
        corr_vec = correction.Log().tensor().reshape(3) * gain
        correction = pp.so3(corr_vec.reshape(1, 3)).Exp()
        angle *= gain

    return GravityRollPitchAlignment(correction @ R_est, True, angle, acc_norm_value)


def estimate_static_imu_initialization(
    time_ns: torch.Tensor,
    acc_body: torch.Tensor,
    gyro_body: torch.Tensor,
    initial_body_to_world: pp.LieTensor | torch.Tensor,
    gravity: float,
    *,
    min_duration_s: float = 3.0,
    gyro_mean_norm_max: float = 0.03,
    gyro_std_max: float | list[float] | tuple[float, ...] | torch.Tensor = 0.1,
    acc_norm_error_max: float = 0.5,
    acc_std_max: float | list[float] | tuple[float, ...] | torch.Tensor = 0.8,
) -> StaticImuInitialization:
    """Estimate initial attitude and per-axis IMU biases from a static interval.

    Yaw remains anchored to ``initial_body_to_world`` because gravity only
    observes roll and pitch. The accelerometer bias is the temporal per-axis
    mean minus the specific force predicted by the gravity-aligned attitude.
    """
    stamps = time_ns.reshape(-1).long()
    acc = acc_body.reshape(-1, 3).double()
    gyro = gyro_body.reshape(-1, 3).double().to(acc)
    initial_rotation = pp.SO3(initial_body_to_world).double().to(acc.device)
    if stamps.numel() == 0 or acc.size(0) == 0 or gyro.size(0) == 0:
        zeros = torch.zeros(3, device=acc.device, dtype=torch.float64)
        return StaticImuInitialization(
            initial_rotation.float(), zeros.float(), zeros.float(), 0.0, 0,
            zeros.float(), zeros.float(), zeros.float(), zeros.float(), False,
            "empty IMU interval",
        )
    if acc.size(0) != gyro.size(0) or stamps.numel() != acc.size(0):
        raise ValueError("Static IMU initialization requires aligned time, acc, and gyro arrays")

    duration_s = float((stamps[-1] - stamps[0]).item()) * 1e-9
    acc_mean = acc.mean(dim=0)
    gyro_mean = gyro.mean(dim=0)
    acc_std = acc.std(dim=0, unbiased=False)
    gyro_std = gyro.std(dim=0, unbiased=False)

    gyro_limit = torch.as_tensor(gyro_std_max, device=acc.device, dtype=acc.dtype).reshape(-1)
    acc_limit = torch.as_tensor(acc_std_max, device=acc.device, dtype=acc.dtype).reshape(-1)
    if gyro_limit.numel() == 1:
        gyro_limit = gyro_limit.repeat(3)
    if acc_limit.numel() == 1:
        acc_limit = acc_limit.repeat(3)
    if gyro_limit.numel() != 3 or acc_limit.numel() != 3:
        raise ValueError("Static IMU standard-deviation thresholds must be scalar or three-axis")

    reasons: list[str] = []
    if duration_s + 1e-9 < float(min_duration_s):
        reasons.append(f"duration {duration_s:.6f}s < {float(min_duration_s):.6f}s")
    if float(gyro_mean.norm().item()) > float(gyro_mean_norm_max):
        reasons.append("gyro mean indicates motion")
    if bool((gyro_std > gyro_limit).any().item()):
        reasons.append("gyro standard deviation indicates motion")
    if abs(float(acc_mean.norm().item()) - abs(float(gravity))) > float(acc_norm_error_max):
        reasons.append("accelerometer norm is inconsistent with gravity")
    if bool((acc_std > acc_limit).any().item()):
        reasons.append("accelerometer standard deviation indicates motion")

    alignment = gravity_roll_pitch_aligned_rotation(
        estimated_body_to_world=initial_rotation,
        acc_body=acc_mean.reshape(1, 3),
        gravity=float(gravity),
        correction_gain=1.0,
        acc_norm_tol=max(float(acc_norm_error_max) / max(abs(float(gravity)), 1e-9), 1e-6),
        window=None,
    )
    aligned_rotation = alignment.rotation.double()
    gravity_world = torch.tensor([0.0, 0.0, float(gravity)], device=acc.device, dtype=acc.dtype)
    expected_static_specific_force = -aligned_rotation.Inv().Act(gravity_world).reshape(3)
    acc_bias = acc_mean - expected_static_specific_force
    gyro_bias = gyro_mean

    return StaticImuInitialization(
        body_to_world=aligned_rotation.float(),
        acc_bias=acc_bias.float(),
        gyro_bias=gyro_bias.float(),
        duration_s=duration_s,
        sample_count=int(acc.size(0)),
        acc_mean=acc_mean.float(),
        gyro_mean=gyro_mean.float(),
        acc_std=acc_std.float(),
        gyro_std=gyro_std.float(),
        stationary=len(reasons) == 0,
        failure_reason="; ".join(reasons),
    )


def integrate_gyro_attitude_world(
    body_to_world: pp.LieTensor | torch.Tensor,
    time_ns: torch.Tensor,
    gyro_body: torch.Tensor,
) -> pp.LieTensor:
    """Propagate body-to-world attitude with body-frame gyro samples.

    This is an attitude-only source for VIO gravity compensation. It deliberately
    ignores accelerometer samples so dynamic linear acceleration cannot bias the
    rotation source.
    """
    R = pp.SO3(body_to_world)
    dtype = R.tensor().dtype
    device = R.tensor().device
    gyro = gyro_body.reshape(-1, 3).to(device=device, dtype=dtype)
    stamps = time_ns.reshape(-1).to(device=device)

    if gyro.numel() == 0:
        return R
    if gyro.size(0) <= 1 or stamps.numel() <= 1:
        dt_arr = torch.tensor([0.01], device=device, dtype=dtype)
    else:
        dt_arr = (stamps[1:] - stamps[:-1]).to(dtype).clamp(min=1.0) * 1e-9

    steps = max(int(gyro.size(0)) - 1, 1)
    for k in range(steps):
        kn = min(k + 1, int(gyro.size(0)) - 1)
        dt = dt_arr[min(k, int(dt_arr.numel()) - 1)]
        gyro_mid = 0.5 * (gyro[k] + gyro[kn])
        R = R @ pp.so3((gyro_mid * dt).reshape(1, 3)).Exp()
    return R


def resolve_translation_prior_mode(mode: str | None, use_velocity: bool) -> str:
    if mode is None or str(mode).strip() == "":
        return "imu_velocity_composed" if use_velocity else "damping_delta_p"
    normalized = str(mode).strip().lower()
    if normalized not in TRANSLATION_PRIOR_MODES:
        allowed = ", ".join(sorted(TRANSLATION_PRIOR_MODES))
        raise ValueError(f"Unsupported imu_trans_prior_mode={mode!r}; expected one of {allowed}")
    return normalized


def translation_prior_semantics(
    mode_or_use_velocity: str | bool | None = None,
    *,
    use_velocity: bool | None = None,
) -> str:
    """Describe the deployable meaning of the IMU translation prior."""
    if use_velocity is not None:
        mode = resolve_translation_prior_mode(None, bool(use_velocity))
    elif isinstance(mode_or_use_velocity, bool):
        mode = resolve_translation_prior_mode(None, mode_or_use_velocity)
    else:
        mode = resolve_translation_prior_mode(mode_or_use_velocity, False)

    if mode == "off":
        return "translation_prior_disabled"
    if mode == "imu_velocity_composed":
        return "velocity_composed_vio_like"
    if mode == "visual_velocity_composed":
        return "visual_velocity_composed_pose_prior"
    return "delta_p_motion_damping"


def select_rotation_prior_std(
    preintegrated_std: float,
    sensor_noise_std: float,
    configured_floor: float,
) -> float:
    """Choose a conservative pose-level rotation-prior standard deviation.

    The optimizer uses the IMU rotation as a pose prior, not as a full VIO
    factor with velocity, bias, timing, and extrinsic states. Therefore its
    effective weight should not become tighter than the configured engineering
    floor even when simulated IMU noise is very small.
    """
    return max(float(preintegrated_std), float(sensor_noise_std), float(configured_floor))


def select_translation_active_rotation_prior_std(
    base_std: float,
    translation_active: bool,
    translation_active_floor: float | None,
) -> float:
    """Optionally relax the rotation prior only when translation prior is active."""
    if not translation_active or translation_active_floor is None:
        return float(base_std)
    return max(float(base_std), float(translation_active_floor))


def should_enable_preintegrated_vio_factor(
    *,
    use_imu_rotation: bool,
    use_imu_translation: bool,
    dt_total: float,
) -> bool:
    """A coupled VIO factor is valid only for full inertial-assist mode."""
    return bool(use_imu_rotation and use_imu_translation and float(dt_total) > 0.0)


def compose_imu_translation_prior(
    delta_p_body: torch.Tensor,
    velocity_world: torch.Tensor,
    R_body_to_world: torch.Tensor,
    dt_total: float,
) -> torch.Tensor:
    """Compose full relative translation in body_i frame.

    IMU preintegration returns delta_p = R_i^T(p_j - p_i - v_i dt).
    The optimizer translation factor needs R_i^T(p_j - p_i), so the
    velocity term must be added back before using it as a pose prior.
    """
    delta_p_body = delta_p_body.reshape(3)
    velocity_world = velocity_world.reshape(3).to(delta_p_body)
    R_body_to_world = R_body_to_world.reshape(3, 3).to(delta_p_body)
    velocity_body_dt = R_body_to_world.T @ (velocity_world * float(dt_total))
    return delta_p_body + velocity_body_dt


def compose_translation_prior_by_mode(
    mode: str,
    delta_p_body: torch.Tensor,
    imu_velocity_world: torch.Tensor | None,
    visual_velocity_world: torch.Tensor | None,
    R_body_to_world: torch.Tensor,
    dt_total: float,
) -> torch.Tensor:
    mode = resolve_translation_prior_mode(mode, use_velocity=False)
    delta_p_body = delta_p_body.reshape(3).float()
    if mode in {"off", "damping_delta_p"}:
        return delta_p_body
    if mode == "visual_velocity_composed":
        if visual_velocity_world is None:
            return delta_p_body
        return compose_imu_translation_prior(
            delta_p_body=delta_p_body,
            velocity_world=visual_velocity_world,
            R_body_to_world=R_body_to_world,
            dt_total=dt_total,
        )
    if imu_velocity_world is None:
        return delta_p_body
    return compose_imu_translation_prior(
        delta_p_body=delta_p_body,
        velocity_world=imu_velocity_world,
        R_body_to_world=R_body_to_world,
        dt_total=dt_total,
    )


def propagate_imu_velocity_world(
    velocity_world: torch.Tensor,
    delta_v_body: torch.Tensor,
    R_body_to_world: torch.Tensor,
    gravity_world: torch.Tensor | None = None,
    dt_total: float = 0.0,
    gravity_handling: str = "preintegration",
) -> torch.Tensor:
    """Propagate world-frame velocity using preintegrated delta_v.

    In ``preintegration`` mode the delta already contains gravity compensation.
    In standard ``residual`` mode the delta contains specific-force integration
    only, so world gravity is added during state propagation.
    """
    velocity_world = velocity_world.reshape(3)
    delta_v_body = delta_v_body.reshape(3).to(velocity_world)
    R_body_to_world = R_body_to_world.reshape(3, 3).to(velocity_world)
    mode = normalize_gravity_handling(gravity_handling)
    propagated = velocity_world + (R_body_to_world @ delta_v_body)
    if mode == STANDARD_LOCAL_FRAME_PREINTEGRATION:
        if gravity_world is None:
            raise ValueError("gravity_world is required when gravity_handling='residual'")
        propagated = propagated + gravity_world.reshape(3).to(propagated) * float(dt_total)
    return propagated


def _as_so3(delta_R: pp.LieTensor | torch.Tensor) -> pp.LieTensor:
    if isinstance(delta_R, pp.LieTensor):
        return pp.SO3(delta_R)

    delta_R_tensor = torch.as_tensor(delta_R)
    if delta_R_tensor.numel() == 4:
        return pp.SO3(delta_R_tensor.reshape(1, 4))
    if delta_R_tensor.numel() == 3:
        return pp.so3(delta_R_tensor.reshape(1, 3)).Exp()
    raise ValueError("delta_R must be an SO3 quaternion LieTensor/tensor or a 3-vector rotation vector")


def vio_preintegrated_imu_residual(
    from_pose: pp.LieTensor | torch.Tensor,
    to_pose: pp.LieTensor | torch.Tensor,
    prev_velocity_world: torch.Tensor,
    curr_velocity_world: torch.Tensor,
    delta_R: pp.LieTensor | torch.Tensor,
    delta_v: torch.Tensor,
    delta_p: torch.Tensor,
    dt_total: float,
    prev_acc_bias: torch.Tensor | None = None,
    prev_gyro_bias: torch.Tensor | None = None,
    curr_acc_bias: torch.Tensor | None = None,
    curr_gyro_bias: torch.Tensor | None = None,
    linearized_acc_bias: torch.Tensor | None = None,
    linearized_gyro_bias: torch.Tensor | None = None,
    bias_jacobian: torch.Tensor | None = None,
    sensor_T_imu: pp.LieTensor | torch.Tensor | None = None,
    gravity_world: torch.Tensor | None = None,
    gravity_handling: str = "preintegration",
) -> torch.Tensor:
    """Compute a Forster/GTSAM-style two-frame IMU residual.

    Poses are body-to-world SE(3). Velocities must use the same world frame as
    the poses. Preintegrated delta quantities are expressed in the previous
    body frame. ``preintegration`` preserves the historical convention where
    delta-p/delta-v already include gravity compensation. ``residual`` uses the
    standard VIO convention and applies world gravity in the residual.

    Returns rows ordered as [position, velocity, rotation], matching the
    existing preintegration covariance order [delta_p, delta_v, delta_phi].
    """
    pose_i = pp.SE3(from_pose)
    pose_j = pp.SE3(to_pose)
    if sensor_T_imu is not None:
        extrinsic = pp.SE3(sensor_T_imu).to(
            device=pose_i.tensor().device,
            dtype=pose_i.tensor().dtype,
        )
        pose_i = pose_i @ extrinsic
        pose_j = pose_j @ extrinsic
    rel_ij = pose_i.Inv() @ pose_j

    dtype = rel_ij.tensor().dtype
    device = rel_ij.tensor().device
    v_i_w = prev_velocity_world.reshape(3).to(device=device, dtype=dtype)
    v_j_w = curr_velocity_world.reshape(3).to(device=device, dtype=dtype)
    d_v = delta_v.reshape(3).to(device=device, dtype=dtype)
    d_p = delta_p.reshape(3).to(device=device, dtype=dtype)
    d_R = _as_so3(delta_R).to(device=device, dtype=dtype)

    if (
        bias_jacobian is not None
        and prev_acc_bias is not None
        and prev_gyro_bias is not None
    ):
        lin_acc = (
            linearized_acc_bias.reshape(3).to(device=device, dtype=dtype)
            if linearized_acc_bias is not None
            else prev_acc_bias.reshape(3).to(device=device, dtype=dtype)
        )
        lin_gyro = (
            linearized_gyro_bias.reshape(3).to(device=device, dtype=dtype)
            if linearized_gyro_bias is not None
            else prev_gyro_bias.reshape(3).to(device=device, dtype=dtype)
        )
        db = torch.cat(
            [
                prev_acc_bias.reshape(3).to(device=device, dtype=dtype) - lin_acc,
                prev_gyro_bias.reshape(3).to(device=device, dtype=dtype) - lin_gyro,
            ],
            dim=0,
        )
        correction = bias_jacobian.reshape(9, 6).to(device=device, dtype=dtype) @ db
        d_p = d_p + correction[0:3]
        d_v = d_v + correction[3:6]
        d_R = d_R @ pp.so3(correction[6:9].reshape(1, 3)).Exp()

    R_i_w = pose_i.rotation().matrix().reshape(3, 3).to(device=device, dtype=dtype)
    rel_translation_body = rel_ij.Act(
        torch.zeros(3, dtype=dtype, device=device)
    ).reshape(3)
    rel_rotation = rel_ij.rotation()

    v_i_body = R_i_w.T @ v_i_w
    delta_v_body = R_i_w.T @ (v_j_w - v_i_w)

    mode = normalize_gravity_handling(gravity_handling)
    gravity_body = torch.zeros(3, device=device, dtype=dtype)
    if mode == STANDARD_LOCAL_FRAME_PREINTEGRATION:
        if gravity_world is None:
            raise ValueError("gravity_world is required when gravity_handling='residual'")
        gravity_body = R_i_w.T @ gravity_world.reshape(3).to(device=device, dtype=dtype)

    dt = float(dt_total)
    r_p = rel_translation_body - v_i_body * dt - 0.5 * gravity_body * (dt ** 2) - d_p
    r_v = delta_v_body - gravity_body * dt - d_v
    r_R = (d_R.Inv() @ rel_rotation).Log().tensor().reshape(3)

    return torch.stack([r_p, r_v, r_R], dim=0)


def vio_bias_random_walk_residual(
    prev_acc_bias: torch.Tensor,
    prev_gyro_bias: torch.Tensor,
    curr_acc_bias: torch.Tensor,
    curr_gyro_bias: torch.Tensor,
) -> torch.Tensor:
    """Residual rows for IMU bias random walk: [b_a_j - b_a_i, b_g_j - b_g_i]."""
    dtype = curr_acc_bias.dtype
    device = curr_acc_bias.device
    acc_step = curr_acc_bias.reshape(3) - prev_acc_bias.reshape(3).to(device=device, dtype=dtype)
    gyro_step = curr_gyro_bias.reshape(3).to(device=device, dtype=dtype) - prev_gyro_bias.reshape(3).to(device=device, dtype=dtype)
    return torch.stack([acc_step, gyro_step], dim=0)


def vio_preintegrated_covariance_blocks(
    cov9: torch.Tensor,
    *,
    diagonal_floor: float = 0.0,
) -> torch.Tensor:
    """Split a 9x9 preintegration covariance into optimizer block rows.

    The current two-frame optimizer consumes block-diagonal covariance rows.
    This helper preserves the preintegration order [delta_p, delta_v,
    delta_phi] while dropping cross-block covariance for the first deployable
    two-frame VIO factor.
    """
    cov = cov9.reshape(9, 9)
    blocks = torch.stack([cov[0:3, 0:3], cov[3:6, 3:6], cov[6:9, 6:9]], dim=0)
    if diagonal_floor > 0.0:
        eye = torch.eye(3, device=blocks.device, dtype=blocks.dtype).unsqueeze(0)
        blocks = blocks + eye * float(diagonal_floor)
    return blocks


def vio_preintegrated_covariance_matrix(
    cov9: torch.Tensor,
    *,
    diagonal_floor: float = 0.0,
) -> torch.Tensor:
    """Return a regularized 9x9 covariance without dropping cross terms."""
    cov = cov9.reshape(9, 9)
    cov = 0.5 * (cov + cov.T)
    if diagonal_floor > 0.0:
        cov = cov + torch.eye(9, device=cov.device, dtype=cov.dtype) * float(diagonal_floor)
    return cov


def covariance_information_matrix(
    covariance: torch.Tensor,
    *,
    relative_correlation_jitter: float = 1e-9,
    absolute_diagonal_floor: float = 0.0,
) -> torch.Tensor:
    """Invert covariance after unit-scale equilibration.

    The old fixed variance floor was applied directly to mixed position,
    velocity, and rotation units. Here each active dimension is normalized by
    its own standard deviation, a tiny jitter is applied to the dimensionless
    correlation matrix, and the inverse is mapped back. Exact zero-variance
    dimensions remain outside the information subspace instead of receiving an
    arbitrary physical variance.
    """
    cov = covariance.reshape(covariance.shape[-2], covariance.shape[-1])
    dtype = cov.dtype
    device = cov.device
    cov64 = 0.5 * (cov.double() + cov.double().T)
    if absolute_diagonal_floor > 0.0:
        cov64 = cov64 + torch.eye(cov64.shape[0], device=device, dtype=torch.float64) * float(
            absolute_diagonal_floor
        )

    diagonal = cov64.diagonal().clamp(min=0.0)
    scale_reference = max(float(diagonal.max().item()), 1.0)
    active = diagonal > torch.finfo(torch.float64).eps * scale_reference
    information = torch.zeros_like(cov64)
    if not bool(active.any().item()):
        return information.to(dtype=dtype)

    indices = torch.nonzero(active, as_tuple=False).reshape(-1)
    active_cov = cov64.index_select(0, indices).index_select(1, indices)
    std = torch.sqrt(diagonal.index_select(0, indices)).clamp(min=torch.finfo(torch.float64).tiny)
    inv_std = std.reciprocal()
    correlation = inv_std[:, None] * active_cov * inv_std[None, :]
    correlation = 0.5 * (correlation + correlation.T)
    if relative_correlation_jitter > 0.0:
        correlation = correlation + torch.eye(
            correlation.shape[0], device=device, dtype=torch.float64
        ) * float(relative_correlation_jitter)
    correlation_inverse = torch.linalg.pinv(correlation, hermitian=True)
    active_information = inv_std[:, None] * correlation_inverse * inv_std[None, :]
    information[indices[:, None], indices[None, :]] = active_information
    return information.to(dtype=dtype)


def build_weight_matrix_from_covariances(
    block_covariances: torch.Tensor,
    *,
    full_covariances: list[torch.Tensor] | tuple[torch.Tensor, ...] = (),
    diagonal_floor: float = 0.0,
) -> torch.Tensor:
    """Build a flattened residual weight matrix from mixed covariance blocks.

    ``block_covariances`` covers ordinary row-wise residual blocks, such as the
    MACVO visual residuals. ``full_covariances`` covers larger coupled residual
    blocks, such as the 9D preintegrated IMU residual or 6D bias random walk.
    """
    blocks: list[torch.Tensor] = []
    cov_blocks = block_covariances
    if cov_blocks.numel() > 0:
        cov_blocks = cov_blocks.reshape(-1, cov_blocks.shape[-2], cov_blocks.shape[-1])
        for cov in cov_blocks:
            blocks.append(
                covariance_information_matrix(
                    cov,
                    absolute_diagonal_floor=diagonal_floor,
                )
            )

    for cov in full_covariances:
        cov_full = cov.reshape(cov.shape[-2], cov.shape[-1])
        blocks.append(
            covariance_information_matrix(
                cov_full,
                absolute_diagonal_floor=diagonal_floor,
            )
        )

    if len(blocks) == 0:
        return block_covariances.new_zeros((0, 0))
    return torch.block_diag(*blocks)


def compose_adaptive_fallback_pose(
    prev_pose: pp.LieTensor | torch.Tensor,
    visual_pose: pp.LieTensor | torch.Tensor,
    imu_rel_pose: pp.LieTensor | torch.Tensor | None,
    *,
    use_imu_rotation: bool,
    use_imu_translation: bool,
    sensor_T_imu: pp.LieTensor | torch.Tensor | None = None,
) -> pp.LieTensor:
    """Compose a lost-track fallback pose without bypassing adaptive IMU gates."""
    visual_pose_se3 = pp.SE3(visual_pose)
    if imu_rel_pose is None or (not use_imu_rotation and not use_imu_translation):
        return visual_pose_se3

    prev_pose_se3 = pp.SE3(prev_pose)
    rel_visual = prev_pose_se3.Inv() @ visual_pose_se3
    rel_imu = pp.SE3(imu_rel_pose)
    if sensor_T_imu is not None:
        extrinsic = pp.SE3(sensor_T_imu).to(
            device=rel_imu.tensor().device,
            dtype=rel_imu.tensor().dtype,
        )
        rel_imu = extrinsic @ rel_imu @ extrinsic.Inv()

    trans = rel_imu.translation() if use_imu_translation else rel_visual.translation()
    rot = rel_imu.rotation() if use_imu_rotation else rel_visual.rotation()
    rel_selected = pp.SE3(torch.cat([trans.reshape(1, 3), rot.tensor().reshape(1, 4)], dim=-1))
    return (prev_pose_se3 @ rel_selected).float()
