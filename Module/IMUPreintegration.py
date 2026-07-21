"""
IMU Preintegration for MAC-VO tight IMU-visual coupling.

Computes relative motion (delta_R, delta_v, delta_p) and their uncertainty
covariance from raw IMU measurements between consecutive keyframes.

Convention:
  The integration frame is the body/world convention supplied by the dataset
  loader. NED uses gravity world = [0, 0, +g]. Z-up frames such as NWU/ENU use
  gravity world = [0, 0, -g] by passing a negative ``gravity`` value.

IMU measurement model (standard specific force convention):
  a_meas = a_kinematic_body - R_w2b @ g_world + bias_a + noise
  So: a_kinematic_body = a_meas + g_body  where g_body = R_w2b @ g_world

  Note: metadata-declared HoloOcean NWU/FLU measurements are rotated into the
  internal NED body frame by the caller before integration. After that
  conversion, a stationary accelerometer has acc_z_NED = -g while NED gravity
  is passed as +g, so a_corr = acc + g_body -> 0.

References:
  Forster et al., "On-Manifold Preintegration for Real-Time Visual-Inertial
  Odometry", TRO 2017.
"""

import math
from dataclasses import dataclass, replace

import torch
import pypose as pp

from Utility.IMUKinematics import (
    LEGACY_EXTERNAL_ATTITUDE_GRAVITY_COMPENSATION,
    STANDARD_LOCAL_FRAME_PREINTEGRATION,
    normalize_gravity_handling,
)


CURRENT_INDEPENDENT_STEP = "current_independent_step"
SAMPLING_AWARE = "sampling_aware"
SAMPLING_AWARE_CROSS_EDGE = "sampling_aware_cross_edge"


def normalize_preintegration_covariance_mode(value: str | None) -> str:
    mode = CURRENT_INDEPENDENT_STEP if value is None else str(value).strip().lower()
    if mode == "sampling_aware_v2":
        mode = SAMPLING_AWARE_CROSS_EDGE
    if mode not in {
        CURRENT_INDEPENDENT_STEP,
        SAMPLING_AWARE,
        SAMPLING_AWARE_CROSS_EDGE,
    }:
        raise ValueError(
            "preintegration covariance mode must be "
            f"'{CURRENT_INDEPENDENT_STEP}', '{SAMPLING_AWARE}', or "
            f"'{SAMPLING_AWARE_CROSS_EDGE}', got {value!r}"
        )
    return mode


@dataclass
class PreintResult:
    """
    Output of IMU preintegration over an interval [t_i, t_j].

    All delta quantities are expressed in the body frame at time t_i.

    Fields:
        delta_R  - SO3 LieTensor: relative rotation  R_i^T R_j
        delta_v  - (3,) float32:  relative velocity in body_i frame (v_j - v_i)
        delta_p  - (3,) float32:  relative position in body_i frame (p_j - p_i - v_i * T)
        cov      - (9,9) float32: covariance of [delta_p(3), delta_v(3), delta_phi(3)]
        dt_total - float:         total integration time (seconds)
        bias_jacobian - (9,6) float32: first-order sensitivity of
                        [delta_p, delta_v, delta_phi] to [b_a, b_g]
        bias_rw_cov   - (6,6) float32: propagated random-walk covariance for
                        [b_a, b_g]
        linearized_acc_bias  - (3,) float32: accelerometer bias used when
                               this preintegrated edge was built
        linearized_gyro_bias - (3,) float32: gyroscope bias used when this
                               preintegrated edge was built
    """
    delta_R : pp.LieTensor   # SO3
    delta_v : torch.Tensor   # (3,)
    delta_p : torch.Tensor   # (3,)
    cov     : torch.Tensor   # (9, 9)
    dt_total: float
    bias_jacobian: torch.Tensor | None = None  # (9, 6)
    bias_rw_cov: torch.Tensor | None = None    # (6, 6)
    linearized_acc_bias: torch.Tensor | None = None   # (3,)
    linearized_gyro_bias: torch.Tensor | None = None  # (3,)
    measurement_cov: torch.Tensor | None = None       # (9, 9)
    bias_process_cov: torch.Tensor | None = None      # (9, 9)


@dataclass(frozen=True)
class SamplingAwareCovarianceComponents:
    """Raw-sample decomposition used by cross-edge sampling-aware factors.

    Sensitivities multiply standardized six-channel raw-sample noise ordered as
    ``[acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z]`` for each timestamp.
    The incoming and outgoing groups are the raw samples used to interpolate
    the camera-time endpoints. ``unique_covariance`` contains every other raw
    measurement contribution plus the unchanged within-edge bias-process
    contribution.
    """

    total_covariance: torch.Tensor
    measurement_covariance: torch.Tensor
    unique_covariance: torch.Tensor
    incoming_raw_time_ns: torch.Tensor
    outgoing_raw_time_ns: torch.Tensor
    incoming_sensitivity: torch.Tensor
    outgoing_sensitivity: torch.Tensor
    full_sensitivity: torch.Tensor


def _axis_noise_density(
    value: float | list[float] | tuple[float, ...] | torch.Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    density = torch.as_tensor(value, device=device, dtype=dtype).reshape(-1)
    if density.numel() == 1:
        return density.repeat(3)
    if density.numel() != 3:
        raise ValueError(
            f"IMU noise density must contain one isotropic value or three axis values, got {density.numel()}"
        )
    return density


def preintegrate_imu(
    time_ns   : torch.Tensor,
    acc       : torch.Tensor,
    gyro      : torch.Tensor,
    R0_world  : pp.LieTensor | None,
    gravity   : float = 9.81,
    sigma_acc : float | list[float] | tuple[float, ...] | torch.Tensor = 0.3,
    sigma_gyro: float | list[float] | tuple[float, ...] | torch.Tensor = 0.01,
    sigma_acc_w: float | list[float] | tuple[float, ...] | torch.Tensor = 0.0,
    sigma_gyro_w: float | list[float] | tuple[float, ...] | torch.Tensor = 0.0,
    gyro_bias : torch.Tensor | None = None,
    acc_bias  : torch.Tensor | None = None,
    gravity_handling: str = "preintegration",
) -> PreintResult:
    """
    Preintegrate IMU measurements between two keyframes.

    Args:
        time_ns   : (N,) int64 - timestamps in nanoseconds
        acc       : (N, 3) float32 - accelerometer measurements (body frame, with gravity)
        gyro      : (N, 3) float32 - gyroscope measurements (body frame)
        R0_world  : SO3 LieTensor describing the rotation of body_i in world frame.
                    Used to compute gravity in body_i frame.
                    If None, identity is assumed (Z-down NED).
        gravity   : float - signed world z gravity (m/s²): +g for NED,
                    -g for z-up frames such as NWU/ENU
        sigma_acc : float - accelerometer noise density (m/s²/√Hz)
        sigma_gyro: float - gyroscope noise density (rad/s/√Hz)
        sigma_acc_w : float - accelerometer bias random walk density (m/s³/√Hz)
        sigma_gyro_w: float - gyroscope bias random walk density (rad/s²/√Hz)
        gyro_bias : (3,) float32 - gyroscope bias to subtract
        acc_bias  : (3,) float32 - accelerometer bias to subtract
        gravity_handling: ``preintegration`` keeps the historical gravity-
                          compensated deltas; ``residual`` integrates specific
                          force only and leaves gravity to the VIO residual.

    Returns:
        PreintResult with delta_R, delta_v, delta_p, cov, dt_total,
        bias_jacobian, and bias_rw_cov
    """
    device = acc.device
    dtype  = torch.float64
    N = acc.size(0)
    gravity_mode = normalize_gravity_handling(gravity_handling)

    linearized_gyro_bias = (
        gyro_bias.reshape(3).to(device=device, dtype=torch.float32).clone()
        if gyro_bias is not None
        else torch.zeros(3, device=device, dtype=torch.float32)
    )
    linearized_acc_bias = (
        acc_bias.reshape(3).to(device=device, dtype=torch.float32).clone()
        if acc_bias is not None
        else torch.zeros(3, device=device, dtype=torch.float32)
    )

    # Apply bias correction
    if gyro_bias is not None:
        gyro = gyro - gyro_bias.to(gyro)
    if acc_bias is not None:
        acc = acc - acc_bias.to(acc)

    acc_d  = acc.to(dtype)
    gyro_d = gyro.to(dtype)
    sigma_acc_axis = _axis_noise_density(sigma_acc, device=device, dtype=dtype)
    sigma_gyro_axis = _axis_noise_density(sigma_gyro, device=device, dtype=dtype)
    sigma_acc_w_axis = _axis_noise_density(sigma_acc_w, device=device, dtype=dtype)
    sigma_gyro_w_axis = _axis_noise_density(sigma_gyro_w, device=device, dtype=dtype)

    # Only the legacy branch is allowed to see an external/global attitude.
    # Standard local-frame preintegration never constructs world gravity and is
    # therefore invariant to the caller's pose estimate by construction.
    g_body_i = None
    if gravity_mode == LEGACY_EXTERNAL_ATTITUDE_GRAVITY_COMPENSATION:
        g_w = torch.tensor([0.0, 0.0, gravity], dtype=dtype, device=device)
        if R0_world is None:
            g_body_i = g_w.clone()
        else:
            g_body_i = R0_world.double().to(device).Inv().Act(g_w)

    # Time steps
    if N <= 1:
        dt_arr   = torch.tensor([0.01], dtype=dtype, device=device)
        dt_total = 0.01
    else:
        dt_ns    = (time_ns[1:] - time_ns[:-1]).to(dtype).clamp(min=1.0) * 1e-9
        dt_total = float(dt_ns.sum().item())
        # Repeat last dt so length matches N (for indexing in loop below)
        dt_arr   = torch.cat([dt_ns, dt_ns[-1:]], dim=0)

    # State: delta_R (SO3), delta_v (3,), delta_p (3,)
    delta_R : pp.LieTensor = pp.identity_SO3(device=device, dtype=dtype)
    delta_v = torch.zeros(3, device=device, dtype=dtype)
    delta_p = torch.zeros(3, device=device, dtype=dtype)

    # Error-state covariance over [delta_p(3), delta_v(3), delta_phi(3), b_a(3), b_g(3)]
    Cov15 = torch.zeros(15, 15, device=device, dtype=dtype)
    BiasProcessCov15 = torch.zeros(15, 15, device=device, dtype=dtype)
    Phi15 = torch.eye(15, device=device, dtype=dtype)

    steps = max(N - 1, 1)
    for k in range(steps):
        dt_k = float(dt_arr[k].item())

        # Mid-point rule
        kn = min(k + 1, N - 1)
        acc_mid  = 0.5 * (acc_d[k]  + acc_d[kn])
        gyro_mid = 0.5 * (gyro_d[k] + gyro_d[kn])

        # Gravity-corrected acceleration in body_i frame.
        # Standard specific force model: a_meas = a_kinematic - g_body
        # Therefore: a_kinematic = a_meas + g_body
        # (After FLU→NED rotation, stationary acc_z_NED = -g; +g_body_i = +g → cancels to 0.)
        a_corr = delta_R.Act(acc_mid)
        if gravity_mode == LEGACY_EXTERNAL_ATTITUDE_GRAVITY_COMPENSATION:
            assert g_body_i is not None
            a_corr = a_corr + g_body_i

        # Position and velocity update (use CURRENT delta_R before rotating)
        delta_p = delta_p + delta_v * dt_k + 0.5 * a_corr * (dt_k ** 2)
        delta_v = delta_v + a_corr * dt_k

        # Rotation update
        delta_R = delta_R @ pp.so3(gyro_mid * dt_k).Exp()

        # -- Covariance propagation (15-state with bias random walk) --
        R_mat    = delta_R.matrix()                                 # (3,3)
        acc_skew = pp.vec2skew(delta_R.Act(acc_mid)).squeeze(0)    # (3,3)

        # Discrete state transition F (15x15)
        F = torch.eye(15, device=device, dtype=dtype)
        F[0:3, 3:6] = torch.eye(3, device=device, dtype=dtype) * dt_k
        F[0:3, 6:9] = -0.5 * acc_skew * (dt_k ** 2)
        F[0:3, 9:12] = -0.5 * R_mat * (dt_k ** 2)
        F[3:6, 6:9] = -acc_skew * dt_k
        F[3:6, 9:12] = -R_mat * dt_k
        F[6:9, 12:15] = -torch.eye(3, device=device, dtype=dtype) * dt_k

        # Noise input matrix G (15x12): [n_a(3), n_g(3), n_ba(3), n_bg(3)]
        G = torch.zeros(15, 12, device=device, dtype=dtype)
        G[0:3, 0:3] = 0.5 * R_mat * (dt_k ** 2)
        G[3:6, 0:3] = R_mat * dt_k
        G[6:9, 3:6] = torch.eye(3, device=device, dtype=dtype) * dt_k
        G[9:12, 6:9] = torch.eye(3, device=device, dtype=dtype) * dt_k
        G[12:15, 9:12] = torch.eye(3, device=device, dtype=dtype) * dt_k

        # Process noise covariance. The G matrix above is already the
        # discrete input Jacobian with dt and dt^2 terms, so sampled white
        # noise uses variance density / dt for this interval.
        Q_sample = torch.zeros(12, 12, device=device, dtype=dtype)
        inv_dt = 1.0 / max(dt_k, 1e-12)
        Q_sample[0:3, 0:3] = torch.diag(sigma_acc_axis.square() * inv_dt)
        Q_sample[3:6, 3:6] = torch.diag(sigma_gyro_axis.square() * inv_dt)
        Q_sample[6:9, 6:9] = torch.diag(sigma_acc_w_axis.square() * inv_dt)
        Q_sample[9:12, 9:12] = torch.diag(sigma_gyro_w_axis.square() * inv_dt)

        Cov15 = F @ Cov15 @ F.T + G @ Q_sample @ G.T
        Q_bias = torch.zeros_like(Q_sample)
        Q_bias[6:12, 6:12] = Q_sample[6:12, 6:12]
        BiasProcessCov15 = F @ BiasProcessCov15 @ F.T + G @ Q_bias @ G.T
        Phi15 = F @ Phi15

    # Keep the propagated covariance physically raw. Numerical regularization
    # belongs at inversion/whitening time, where it can be scale-aware.
    Cov15 = 0.5 * (Cov15 + Cov15.T)
    Cov = Cov15[0:9, 0:9]
    bias_process_cov = 0.5 * (
        BiasProcessCov15[0:9, 0:9] + BiasProcessCov15[0:9, 0:9].T
    )
    measurement_cov = 0.5 * (
        (Cov - bias_process_cov) + (Cov - bias_process_cov).T
    )
    bias_jacobian = Phi15[0:9, 9:15]
    bias_rw_cov = Cov15[9:15, 9:15]

    return PreintResult(
        delta_R  = delta_R.float(),
        delta_v  = delta_v.float(),
        delta_p  = delta_p.float(),
        cov      = Cov.float(),
        dt_total = dt_total,
        bias_jacobian = bias_jacobian.float(),
        bias_rw_cov = bias_rw_cov.float(),
        linearized_acc_bias=linearized_acc_bias.float(),
        linearized_gyro_bias=linearized_gyro_bias.float(),
        measurement_cov=measurement_cov.float(),
        bias_process_cov=bias_process_cov.float(),
    )


def _skew(vector: torch.Tensor) -> torch.Tensor:
    x, y, z = vector.unbind(dim=-1)
    zero = torch.zeros_like(x)
    return torch.stack(
        [zero, -z, y, z, zero, -x, -y, x, zero], dim=-1
    ).reshape(*vector.shape[:-1], 3, 3)


def _so3_exp(rotvec: torch.Tensor) -> torch.Tensor:
    theta = torch.linalg.vector_norm(rotvec, dim=-1, keepdim=True)
    matrix = _skew(rotvec)
    a = torch.sinc(theta / math.pi)[..., None]
    b = (0.5 * torch.sinc(theta / (2.0 * math.pi)).square())[..., None]
    identity = torch.eye(3, dtype=rotvec.dtype, device=rotvec.device)
    return identity + a * matrix + b * (matrix @ matrix)


def _so3_log(rotation: torch.Tensor) -> torch.Tensor:
    vee = 0.5 * torch.stack(
        [
            rotation[..., 2, 1] - rotation[..., 1, 2],
            rotation[..., 0, 2] - rotation[..., 2, 0],
            rotation[..., 1, 0] - rotation[..., 0, 1],
        ],
        dim=-1,
    )
    sine = torch.linalg.vector_norm(vee, dim=-1, keepdim=True)
    cosine = (
        (torch.diagonal(rotation, dim1=-2, dim2=-1).sum(dim=-1, keepdim=True) - 1.0)
        * 0.5
    ).clamp(-1.0, 1.0)
    theta = torch.atan2(sine, cosine)
    raw_factor = theta / sine.clamp_min(1e-12)
    series_factor = 1.0 + sine.square() / 6.0
    return torch.where(sine > 1e-7, raw_factor, series_factor) * vee


def _integrate_midpoint_measurements(
    acc_mid: torch.Tensor,
    gyro_mid: torch.Tensor,
    dt_s: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rotation = torch.eye(3, dtype=acc_mid.dtype, device=acc_mid.device)
    velocity = torch.zeros(3, dtype=acc_mid.dtype, device=acc_mid.device)
    position = torch.zeros(3, dtype=acc_mid.dtype, device=acc_mid.device)
    for step in range(acc_mid.shape[0]):
        dt = dt_s[step]
        acceleration = rotation @ acc_mid[step]
        position = position + velocity * dt + 0.5 * acceleration * dt.square()
        velocity = velocity + acceleration * dt
        rotation = rotation @ _so3_exp(gyro_mid[step] * dt)
    return position, velocity, rotation


def build_sampling_aware_covariance_components(
    preintegration: PreintResult,
    *,
    time_ns: torch.Tensor,
    acc_internal: torch.Tensor,
    gyro_internal: torch.Tensor,
    knot_from_raw: torch.Tensor,
    sensor_to_internal_rotation: torch.Tensor,
    measurement_rate_hz: float,
    sigma_acc: float | list[float] | tuple[float, ...] | torch.Tensor,
    sigma_gyro: float | list[float] | tuple[float, ...] | torch.Tensor,
    acc_bias: torch.Tensor,
    gyro_bias: torch.Tensor,
    raw_time_ns: torch.Tensor | None = None,
) -> SamplingAwareCovarianceComponents:
    """Build the within-edge covariance and its endpoint-sample decomposition.

    The mean deltas, bias Jacobian, bias random-walk covariance, and bias
    process contribution are not changed. ``knot_from_raw`` includes
    camera-time endpoint interpolation. Midpoint averaging is added here, so
    shared raw samples retain their correlations.
    """
    dtype = torch.float64
    device = acc_internal.device
    knot_from_raw = knot_from_raw.to(device=device, dtype=dtype)
    time_ns = time_ns.reshape(-1).to(device=device)
    acc_internal = acc_internal.reshape(-1, 3).to(device=device, dtype=dtype)
    gyro_internal = gyro_internal.reshape(-1, 3).to(device=device, dtype=dtype)
    if time_ns.numel() < 2 or knot_from_raw.shape[0] != time_ns.numel():
        raise ValueError(
            "sampling-aware covariance requires one interpolation-map row per IMU knot"
        )
    if measurement_rate_hz <= 0.0:
        raise ValueError("measurement_rate_hz must be positive")

    dt_s = (time_ns[1:] - time_ns[:-1]).to(dtype).clamp(min=1.0) * 1e-9
    nominal_acc_mid = 0.5 * (acc_internal[:-1] + acc_internal[1:])
    nominal_gyro_mid = 0.5 * (gyro_internal[:-1] + gyro_internal[1:])
    nominal_acc_mid = nominal_acc_mid - acc_bias.reshape(1, 3).to(device=device, dtype=dtype)
    nominal_gyro_mid = nominal_gyro_mid - gyro_bias.reshape(1, 3).to(device=device, dtype=dtype)

    reference_p, reference_v, reference_R = _integrate_midpoint_measurements(
        nominal_acc_mid, nominal_gyro_mid, dt_s
    )

    def error_from_processed_noise(noise_flat: torch.Tensor) -> torch.Tensor:
        noise = noise_flat.reshape(dt_s.numel(), 6)
        noisy_p, noisy_v, noisy_R = _integrate_midpoint_measurements(
            nominal_acc_mid + noise[:, 0:3],
            nominal_gyro_mid + noise[:, 3:6],
            dt_s,
        )
        return torch.cat(
            [
                noisy_p - reference_p,
                noisy_v - reference_v,
                _so3_log(reference_R.mT @ noisy_R),
            ]
        )

    zero = torch.zeros(dt_s.numel() * 6, device=device, dtype=dtype, requires_grad=True)
    jacobian_processed = torch.autograd.functional.jacobian(
        error_from_processed_noise,
        zero,
        create_graph=False,
        strict=True,
        vectorize=False,
    )

    midpoint_from_knot = torch.zeros(
        (dt_s.numel(), time_ns.numel()), device=device, dtype=dtype
    )
    step_index = torch.arange(dt_s.numel(), device=device)
    midpoint_from_knot[step_index, step_index] = 0.5
    midpoint_from_knot[step_index, step_index + 1] = 0.5
    temporal_map = midpoint_from_knot @ knot_from_raw
    rotation = sensor_to_internal_rotation.reshape(3, 3).to(device=device, dtype=dtype)
    channel_rotation = torch.zeros((6, 6), device=device, dtype=dtype)
    channel_rotation[0:3, 0:3] = rotation
    channel_rotation[3:6, 3:6] = rotation
    processed_from_raw = torch.kron(temporal_map.contiguous(), channel_rotation.contiguous())
    jacobian_raw = jacobian_processed @ processed_from_raw

    sigma_acc_axis = _axis_noise_density(sigma_acc, device=device, dtype=dtype)
    sigma_gyro_axis = _axis_noise_density(sigma_gyro, device=device, dtype=dtype)
    raw_variance = torch.cat(
        [sigma_acc_axis.square(), sigma_gyro_axis.square()]
    ) * float(measurement_rate_hz)
    raw_variance = raw_variance.repeat(knot_from_raw.shape[1])
    standardized_sensitivity = jacobian_raw * raw_variance.sqrt().reshape(1, -1)
    sampling_measurement_cov = standardized_sensitivity @ standardized_sensitivity.T
    sampling_measurement_cov = 0.5 * (
        sampling_measurement_cov + sampling_measurement_cov.T
    )
    bias_process_cov = preintegration.bias_process_cov
    if bias_process_cov is None:
        raise ValueError("preintegration does not expose its bias-process covariance")
    sampling_total_cov = sampling_measurement_cov + bias_process_cov.to(
        device=device, dtype=dtype
    )
    sampling_total_cov = 0.5 * (sampling_total_cov + sampling_total_cov.T)

    support_threshold = 32.0 * torch.finfo(dtype).eps
    incoming_samples = torch.nonzero(
        knot_from_raw[0].abs() > support_threshold, as_tuple=False
    ).reshape(-1)
    outgoing_samples = torch.nonzero(
        knot_from_raw[-1].abs() > support_threshold, as_tuple=False
    ).reshape(-1)
    if incoming_samples.numel() == 0 or outgoing_samples.numel() == 0:
        raise ValueError("camera-time endpoints must be supported by raw IMU samples")

    all_samples = torch.arange(knot_from_raw.shape[1], device=device)
    boundary_mask = torch.zeros(knot_from_raw.shape[1], dtype=torch.bool, device=device)
    boundary_mask[incoming_samples] = True
    boundary_mask[outgoing_samples] = True
    unique_samples = all_samples[~boundary_mask]
    channels = torch.arange(6, device=device)

    def sample_columns(samples: torch.Tensor) -> torch.Tensor:
        if samples.numel() == 0:
            return torch.empty(0, dtype=torch.long, device=device)
        return (samples.reshape(-1, 1) * 6 + channels.reshape(1, 6)).reshape(-1)

    incoming_columns = sample_columns(incoming_samples)
    outgoing_columns = sample_columns(outgoing_samples)
    unique_columns = sample_columns(unique_samples)
    incoming_sensitivity = standardized_sensitivity[:, incoming_columns]
    outgoing_sensitivity = standardized_sensitivity[:, outgoing_columns]
    unique_sensitivity = standardized_sensitivity[:, unique_columns]
    unique_covariance = unique_sensitivity @ unique_sensitivity.T
    unique_covariance = unique_covariance + bias_process_cov.to(device=device, dtype=dtype)
    unique_covariance = 0.5 * (unique_covariance + unique_covariance.T)

    if raw_time_ns is None:
        raw_time_ns = torch.arange(knot_from_raw.shape[1], device=device, dtype=torch.long)
    else:
        raw_time_ns = raw_time_ns.reshape(-1).to(device=device, dtype=torch.long)
        if raw_time_ns.numel() != knot_from_raw.shape[1]:
            raise ValueError(
                "raw_time_ns must contain one timestamp per raw interpolation column"
            )
    return SamplingAwareCovarianceComponents(
        total_covariance=sampling_total_cov,
        measurement_covariance=sampling_measurement_cov,
        unique_covariance=unique_covariance,
        incoming_raw_time_ns=raw_time_ns[incoming_samples],
        outgoing_raw_time_ns=raw_time_ns[outgoing_samples],
        incoming_sensitivity=incoming_sensitivity,
        outgoing_sensitivity=outgoing_sensitivity,
        full_sensitivity=standardized_sensitivity,
    )


def replace_with_sampling_aware_covariance(
    preintegration: PreintResult,
    *,
    time_ns: torch.Tensor,
    acc_internal: torch.Tensor,
    gyro_internal: torch.Tensor,
    knot_from_raw: torch.Tensor,
    sensor_to_internal_rotation: torch.Tensor,
    measurement_rate_hz: float,
    sigma_acc: float | list[float] | tuple[float, ...] | torch.Tensor,
    sigma_gyro: float | list[float] | tuple[float, ...] | torch.Tensor,
    acc_bias: torch.Tensor,
    gyro_bias: torch.Tensor,
) -> PreintResult:
    """Replace only the 9D covariance with the SA-v1 within-edge result."""
    components = build_sampling_aware_covariance_components(
        preintegration,
        time_ns=time_ns,
        acc_internal=acc_internal,
        gyro_internal=gyro_internal,
        knot_from_raw=knot_from_raw,
        sensor_to_internal_rotation=sensor_to_internal_rotation,
        measurement_rate_hz=measurement_rate_hz,
        sigma_acc=sigma_acc,
        sigma_gyro=sigma_gyro,
        acc_bias=acc_bias,
        gyro_bias=gyro_bias,
    )
    return replace(
        preintegration,
        cov=components.total_covariance.float(),
        measurement_cov=components.measurement_covariance.float(),
    )


def preintegrate_imu_local_frame(
    time_ns: torch.Tensor,
    acc: torch.Tensor,
    gyro: torch.Tensor,
    sigma_acc: float | list[float] | tuple[float, ...] | torch.Tensor = 0.3,
    sigma_gyro: float | list[float] | tuple[float, ...] | torch.Tensor = 0.01,
    sigma_acc_w: float | list[float] | tuple[float, ...] | torch.Tensor = 0.0,
    sigma_gyro_w: float | list[float] | tuple[float, ...] | torch.Tensor = 0.0,
    gyro_bias: torch.Tensor | None = None,
    acc_bias: torch.Tensor | None = None,
) -> PreintResult:
    """Standard local-frame IMU preintegration with no world-pose input.

    The inputs are raw body-frame specific force, body-frame angular velocity,
    sample times, bias linearization point, and noise densities. Gravity belongs
    exclusively to the downstream IMU residual.
    """
    return preintegrate_imu(
        time_ns=time_ns,
        acc=acc,
        gyro=gyro,
        R0_world=None,
        gravity=0.0,
        sigma_acc=sigma_acc,
        sigma_gyro=sigma_gyro,
        sigma_acc_w=sigma_acc_w,
        sigma_gyro_w=sigma_gyro_w,
        gyro_bias=gyro_bias,
        acc_bias=acc_bias,
        gravity_handling=STANDARD_LOCAL_FRAME_PREINTEGRATION,
    )
