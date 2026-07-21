from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pypose as pp
import torch

from Module.Map.Template import MatchObs, PointNode
from Module.Optimization.TwoFramePGO.Graphs import GraphInput
from Utility.Point import pixel2point_NED, point2pixel_NED


SIGMA_ACC = 0.014125836301591877
SIGMA_GYRO = 0.0018289805264456304
SIGMA_ACC_W = 0.00038607055555878352
SIGMA_GYRO_W = 0.000035786418170002551
BASE_ACC_BIAS = torch.tensor([0.004, -0.003, 0.002], dtype=torch.float64)
BASE_GYRO_BIAS = torch.tensor([0.0004, -0.0003, 0.002], dtype=torch.float64)
VISUAL_INTRINSIC = torch.tensor(
    [[320.0, 0.0, 320.0], [0.0, 320.0, 240.0], [0.0, 0.0, 1.0]],
    dtype=torch.float32,
)
VISUAL_BASELINE_M = 0.225
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480
MAX_VISUAL_POINTS_PER_EDGE = 96
MOTION_SAMPLING_CONTRACT = "analytic_yaw_rate_at_imu_timestamp_v1"


@dataclass(frozen=True)
class SyntheticSequenceConfig:
    duration_s: float
    trajectory_period_s: float = field(default=10.0, init=False)
    camera_rate_hz: float = 30.0
    imu_rate_hz: float = 100.0
    radius_m: float = 0.3
    depth_amplitude_m: float = 0.05
    gravity_m_s2: float = 9.8
    seed: int = 20260711


@dataclass(frozen=True)
class EvaluationTruth:
    camera_time_ns: torch.Tensor
    imu_time_ns: torch.Tensor
    pose_body_to_world: pp.LieTensor
    position_world: torch.Tensor
    velocity_world: torch.Tensor
    acceleration_world: torch.Tensor
    angular_velocity_body: torch.Tensor
    true_acc_bias: torch.Tensor
    true_gyro_bias: torch.Tensor
    bias_mode: str
    noise_mode: str


@dataclass(frozen=True)
class EstimatorIMUInput:
    time_ns: torch.Tensor
    measured_acc_body: torch.Tensor
    measured_gyro_body: torch.Tensor


@dataclass(frozen=True)
class EstimatorVisualInput:
    intrinsic: torch.Tensor
    baseline_m: float
    pose_initial: pp.LieTensor
    relative_motion_initial: pp.LieTensor
    edges: tuple[GraphInput, ...]


@dataclass(frozen=True)
class _VisualCondition:
    pixel_sigma: float
    disparity_sigma: float
    relative_scale_bias: float
    relative_scale_sigma: float
    relative_translation_sigma_m: float
    relative_rotation_sigma_rad: float


_VISUAL_CONDITIONS = {
    "clean_visual": _VisualCondition(0.15, 0.08, 0.0, 0.0002, 0.00005, 0.00005),
    "drifted_visual": _VisualCondition(0.8, 0.4, 0.003, 0.002, 0.002, 0.0015),
}


def _sample_times(duration_s: float, rate_hz: float) -> tuple[torch.Tensor, torch.Tensor]:
    sample_count = int(round(duration_s * rate_hz)) + 1
    expected = (sample_count - 1) / rate_hz
    if not math.isclose(expected, duration_s, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(f"duration_s={duration_s} is not aligned with rate_hz={rate_hz}")
    indices = torch.arange(sample_count, dtype=torch.float64)
    time_s = indices / rate_hz
    time_ns = torch.round(time_s * 1e9).to(torch.int64)
    return time_s, time_ns


def _trajectory(config: SyntheticSequenceConfig, time_s: torch.Tensor):
    omega = 2.0 * math.pi / config.trajectory_period_s
    theta = omega * time_s
    x = config.radius_m * torch.sin(theta)
    y = config.radius_m * (1.0 - torch.cos(theta))
    z = config.depth_amplitude_m * (1.0 - torch.cos(2.0 * theta))
    vx = config.radius_m * omega * torch.cos(theta)
    vy = config.radius_m * omega * torch.sin(theta)
    vz = 2.0 * config.depth_amplitude_m * omega * torch.sin(2.0 * theta)
    ax = -config.radius_m * omega**2 * torch.sin(theta)
    ay = config.radius_m * omega**2 * torch.cos(theta)
    az = 4.0 * config.depth_amplitude_m * omega**2 * torch.cos(2.0 * theta)
    roll = torch.zeros_like(theta)
    pitch = torch.zeros_like(theta)
    yaw = theta
    angular_velocity = torch.stack(
        (
            torch.zeros_like(time_s),
            torch.zeros_like(time_s),
            torch.full_like(time_s, omega),
        ),
        dim=-1,
    )
    position = torch.stack((x, y, z), dim=-1)
    velocity = torch.stack((vx, vy, vz), dim=-1)
    acceleration = torch.stack((ax, ay, az), dim=-1)
    return position, velocity, acceleration, roll, pitch, yaw, angular_velocity


def _continuous_quaternions(roll: torch.Tensor, pitch: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    half_roll = 0.5 * roll
    half_pitch = 0.5 * pitch
    half_yaw = 0.5 * yaw
    cr, sr = torch.cos(half_roll), torch.sin(half_roll)
    cp, sp = torch.cos(half_pitch), torch.sin(half_pitch)
    cy, sy = torch.cos(half_yaw), torch.sin(half_yaw)
    quat = torch.stack(
        (
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy,
        ),
        dim=-1,
    )
    for idx in range(1, quat.shape[0]):
        if torch.dot(quat[idx - 1], quat[idx]) < 0:
            quat[idx] = -quat[idx]
    return quat


def _motion_samples(
    config: SyntheticSequenceConfig,
    time_s: torch.Tensor,
) -> dict[str, torch.Tensor | pp.LieTensor]:
    position, velocity, acceleration, roll, pitch, yaw, angular_velocity = _trajectory(config, time_s)
    quaternion = _continuous_quaternions(roll, pitch, yaw)
    rotation = pp.SO3(quaternion)
    pose = pp.SE3(torch.cat((position, quaternion), dim=-1))
    return {
        "pose": pose,
        "position": position,
        "velocity": velocity,
        "acceleration": acceleration,
        "angular_velocity": angular_velocity,
    }


def _realize_bias(config: SyntheticSequenceConfig, sample_count: int, bias_mode: str) -> tuple[torch.Tensor, torch.Tensor]:
    zeros = torch.zeros((sample_count, 3), dtype=torch.float64)
    if bias_mode == "zero_bias":
        return zeros.clone(), zeros.clone()
    if bias_mode == "constant_bias":
        return BASE_ACC_BIAS.repeat(sample_count, 1), BASE_GYRO_BIAS.repeat(sample_count, 1)
    if bias_mode == "drifting_bias":
        acc_bias = BASE_ACC_BIAS.repeat(sample_count, 1)
        gyro_bias = BASE_GYRO_BIAS.repeat(sample_count, 1)
        if sample_count > 1:
            acc_gen = torch.Generator().manual_seed(config.seed)
            gyro_gen = torch.Generator().manual_seed(config.seed + 1)
            acc_steps = torch.randn((sample_count - 1, 3), generator=acc_gen, dtype=torch.float64)
            gyro_steps = torch.randn((sample_count - 1, 3), generator=gyro_gen, dtype=torch.float64)
            acc_bias[1:] += acc_steps.cumsum(dim=0) * (SIGMA_ACC_W / math.sqrt(config.imu_rate_hz))
            gyro_bias[1:] += gyro_steps.cumsum(dim=0) * (SIGMA_GYRO_W / math.sqrt(config.imu_rate_hz))
        return acc_bias, gyro_bias
    raise ValueError(f"Unsupported bias_mode: {bias_mode}")


def _white_noise(config: SyntheticSequenceConfig, sample_count: int, noise_mode: str) -> tuple[torch.Tensor, torch.Tensor]:
    zeros = torch.zeros((sample_count, 3), dtype=torch.float64)
    if noise_mode == "mean_measurement":
        return zeros.clone(), zeros.clone()
    if noise_mode == "fixed_seed_normal":
        acc_gen = torch.Generator().manual_seed(config.seed + 2)
        gyro_gen = torch.Generator().manual_seed(config.seed + 3)
        acc_noise = torch.randn((sample_count, 3), generator=acc_gen, dtype=torch.float64)
        gyro_noise = torch.randn((sample_count, 3), generator=gyro_gen, dtype=torch.float64)
        return acc_noise * (SIGMA_ACC * math.sqrt(config.imu_rate_hz)), gyro_noise * (
            SIGMA_GYRO * math.sqrt(config.imu_rate_hz)
        )
    raise ValueError(f"Unsupported noise_mode: {noise_mode}")


def _validate_truth_modes(truth: EvaluationTruth, bias_mode: str, noise_mode: str) -> None:
    if truth.bias_mode != bias_mode or truth.noise_mode != noise_mode:
        raise ValueError(
            "Requested mode does not match truth mode: "
            f"truth=({truth.bias_mode}, {truth.noise_mode}) requested=({bias_mode}, {noise_mode})"
        )


def generate_truth(
    config: SyntheticSequenceConfig,
    bias_mode: str = "zero_bias",
    noise_mode: str = "mean_measurement",
) -> EvaluationTruth:
    camera_time_s, camera_time_ns = _sample_times(config.duration_s, config.camera_rate_hz)
    _, imu_time_ns = _sample_times(config.duration_s, config.imu_rate_hz)
    camera_motion = _motion_samples(config, camera_time_s)
    acc_bias, gyro_bias = _realize_bias(config, int(imu_time_ns.numel()), bias_mode)
    truth = EvaluationTruth(
        camera_time_ns=camera_time_ns,
        imu_time_ns=imu_time_ns,
        pose_body_to_world=camera_motion["pose"],
        position_world=camera_motion["position"],
        velocity_world=camera_motion["velocity"],
        acceleration_world=camera_motion["acceleration"],
        angular_velocity_body=camera_motion["angular_velocity"],
        true_acc_bias=acc_bias,
        true_gyro_bias=gyro_bias,
        bias_mode=bias_mode,
        noise_mode=noise_mode,
    )
    return truth


def generate_imu_input(
    config: SyntheticSequenceConfig,
    truth: EvaluationTruth,
    bias_mode: str,
    noise_mode: str,
) -> EstimatorIMUInput:
    _validate_truth_modes(truth, bias_mode, noise_mode)
    imu_time_s, imu_time_ns = _sample_times(config.duration_s, config.imu_rate_hz)
    imu_motion = _motion_samples(config, imu_time_s)
    gravity_world = torch.tensor([0.0, 0.0, config.gravity_m_s2], dtype=torch.float64)
    specific_force = imu_motion["pose"].rotation().Inv().Act(imu_motion["acceleration"] - gravity_world)
    acc_noise, gyro_noise = _white_noise(config, imu_time_ns.numel(), noise_mode)
    return EstimatorIMUInput(
        time_ns=imu_time_ns,
        measured_acc_body=specific_force + truth.true_acc_bias + acc_noise,
        measured_gyro_body=imu_motion["angular_velocity"] + truth.true_gyro_bias + gyro_noise,
    )


def _visual_initial_poses(
    config: SyntheticSequenceConfig,
    truth: EvaluationTruth,
    condition: _VisualCondition,
) -> tuple[pp.LieTensor, pp.LieTensor]:
    truth_pose = truth.pose_body_to_world
    pose_generator = torch.Generator().manual_seed(config.seed + 10)
    accumulated = [truth_pose[0].tensor().clone()]
    relative_motions: list[torch.Tensor] = []

    for frame_j in range(1, truth_pose.shape[0]):
        relative = truth_pose[frame_j - 1].Inv() @ truth_pose[frame_j]
        scale_noise = torch.randn((), generator=pose_generator, dtype=torch.float64)
        scale = 1.0 + condition.relative_scale_bias + condition.relative_scale_sigma * scale_noise
        translation_noise = torch.randn((3,), generator=pose_generator, dtype=torch.float64)
        rotation_noise = torch.randn((3,), generator=pose_generator, dtype=torch.float64)
        relative_translation = (
            relative.translation().reshape(3) * scale
            + translation_noise * condition.relative_translation_sigma_m
        )
        rotation_delta = pp.so3(rotation_noise * condition.relative_rotation_sigma_rad).Exp()
        relative_rotation = relative.rotation() @ rotation_delta
        perturbed_relative = pp.SE3(
            torch.cat((relative_translation, relative_rotation.tensor().reshape(4)), dim=0)
        )
        relative_motions.append(perturbed_relative.tensor())
        accumulated.append((pp.SE3(accumulated[-1]) @ perturbed_relative).tensor())

    return pp.SE3(torch.stack(accumulated, dim=0)), pp.SE3(torch.stack(relative_motions, dim=0))


def _candidate_points_world(truth: EvaluationTruth) -> torch.Tensor:
    depth = torch.tensor([3.5, 5.0, 7.0], dtype=torch.float64)
    horizontal = torch.tensor([-0.60, -0.30, 0.0, 0.30, 0.60], dtype=torch.float64)
    vertical = torch.tensor([-0.40, -0.13, 0.13, 0.40], dtype=torch.float64)
    local_template = torch.stack(
        [
            torch.tensor([x, x * y_ratio, x * z_ratio], dtype=torch.float64)
            for x in depth.tolist()
            for y_ratio in horizontal.tolist()
            for z_ratio in vertical.tolist()
        ],
        dim=0,
    )

    frame_count = truth.pose_body_to_world.shape[0]
    anchor_count = min(frame_count, 31)
    anchor_indices = torch.linspace(0, frame_count - 1, anchor_count, dtype=torch.float64).round().long().unique()
    return torch.cat(
        [(truth.pose_body_to_world[idx] * local_template).reshape(-1, 3) for idx in anchor_indices.tolist()],
        dim=0,
    )


def _common_visible_points(
    points_world: torch.Tensor,
    pose_i: pp.LieTensor,
    pose_j: pp.LieTensor,
    intrinsic: torch.Tensor,
    image_margin: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    points_i = (pose_i.Inv() * points_world).reshape(-1, 3)
    points_j = (pose_j.Inv() * points_world).reshape(-1, 3)
    pixels_i = point2pixel_NED(points_i, intrinsic)
    pixels_j = point2pixel_NED(points_j, intrinsic)

    def visible(points: torch.Tensor, pixels: torch.Tensor) -> torch.Tensor:
        return (
            (points[:, 0] > 0.0)
            & (pixels[:, 0] >= image_margin)
            & (pixels[:, 0] < IMAGE_WIDTH - image_margin)
            & (pixels[:, 1] >= image_margin)
            & (pixels[:, 1] < IMAGE_HEIGHT - image_margin)
        )

    visible_indices = torch.nonzero(visible(points_i, pixels_i) & visible(points_j, pixels_j), as_tuple=False).flatten()
    if visible_indices.numel() > MAX_VISUAL_POINTS_PER_EDGE:
        selected_offsets = torch.linspace(
            0,
            visible_indices.numel() - 1,
            MAX_VISUAL_POINTS_PER_EDGE,
            dtype=torch.float64,
        ).round().long()
        visible_indices = visible_indices[selected_offsets]
    return (
        points_world[visible_indices],
        points_i[visible_indices],
        points_j[visible_indices],
        pixels_i[visible_indices],
        pixels_j[visible_indices],
    )


def _stereo_covariance_ned(
    pixels: torch.Tensor,
    disparity: torch.Tensor,
    intrinsic: torch.Tensor,
    baseline_m: float,
    pixel_sigma: float,
    disparity_sigma: float,
) -> torch.Tensor:
    fx = intrinsic[0, 0]
    fy = intrinsic[1, 1]
    cx = intrinsic[0, 2]
    cy = intrinsic[1, 2]
    u = pixels[:, 0]
    v = pixels[:, 1]
    disp = disparity.reshape(-1)
    baseline = torch.as_tensor(baseline_m, dtype=torch.float64)

    jacobian = torch.zeros((pixels.shape[0], 3, 3), dtype=torch.float64)
    jacobian[:, 0, 2] = -fx * baseline / disp.square()
    jacobian[:, 1, 0] = baseline / disp
    jacobian[:, 1, 2] = -(u - cx) * baseline / disp.square()
    jacobian[:, 2, 1] = fx * baseline / (fy * disp)
    jacobian[:, 2, 2] = -(v - cy) * fx * baseline / (fy * disp.square())

    measurement_covariance = torch.diag(
        torch.tensor(
            [pixel_sigma**2, pixel_sigma**2, disparity_sigma**2],
            dtype=torch.float64,
        )
    ).expand(pixels.shape[0], -1, -1)
    return jacobian @ measurement_covariance @ jacobian.transpose(-1, -2)


def _visual_covariance_mean(observations: MatchObs) -> float:
    point_variance = observations.data["obs2_covTc"].diagonal(dim1=-2, dim2=-1).mean(dim=-1)
    weights = 1.0 / (point_variance + 1e-6)
    weights = weights / weights.sum()
    return float((weights * point_variance).sum().item())


def _build_visual_edge(
    frame_i: int,
    frame_j: int,
    points_i: torch.Tensor,
    points_j: torch.Tensor,
    pixels_i: torch.Tensor,
    pixels_j: torch.Tensor,
    pose_initial: pp.LieTensor,
    intrinsic: torch.Tensor,
    condition: _VisualCondition,
    measurement_generator: torch.Generator,
) -> GraphInput:
    point_count = points_i.shape[0]
    pixel_noise_i = torch.randn((point_count, 2), generator=measurement_generator, dtype=torch.float64)
    pixel_noise_j = torch.randn((point_count, 2), generator=measurement_generator, dtype=torch.float64)
    disparity_noise_i = torch.randn((point_count,), generator=measurement_generator, dtype=torch.float64)
    disparity_noise_j = torch.randn((point_count,), generator=measurement_generator, dtype=torch.float64)

    noisy_pixels_i = pixels_i + pixel_noise_i * condition.pixel_sigma
    noisy_pixels_j = pixels_j + pixel_noise_j * condition.pixel_sigma
    true_disparity_i = intrinsic[0, 0] * VISUAL_BASELINE_M / points_i[:, 0]
    true_disparity_j = intrinsic[0, 0] * VISUAL_BASELINE_M / points_j[:, 0]
    noisy_disparity_i = true_disparity_i + disparity_noise_i * condition.disparity_sigma
    noisy_disparity_j = true_disparity_j + disparity_noise_j * condition.disparity_sigma
    noisy_valid = (
        (noisy_pixels_i[:, 0] >= 0.0)
        & (noisy_pixels_i[:, 0] < IMAGE_WIDTH)
        & (noisy_pixels_i[:, 1] >= 0.0)
        & (noisy_pixels_i[:, 1] < IMAGE_HEIGHT)
        & (noisy_pixels_j[:, 0] >= 0.0)
        & (noisy_pixels_j[:, 0] < IMAGE_WIDTH)
        & (noisy_pixels_j[:, 1] >= 0.0)
        & (noisy_pixels_j[:, 1] < IMAGE_HEIGHT)
        & (noisy_disparity_i > 0.0)
        & (noisy_disparity_j > 0.0)
    )
    noisy_pixels_i = noisy_pixels_i[noisy_valid].float()
    noisy_pixels_j = noisy_pixels_j[noisy_valid].float()
    noisy_disparity_i = noisy_disparity_i[noisy_valid].float()
    noisy_disparity_j = noisy_disparity_j[noisy_valid].float()
    if noisy_pixels_i.shape[0] < 12:
        raise ValueError(f"Visual edge {frame_i}->{frame_j} has fewer than 12 common observations")

    noisy_pixels_i_double = noisy_pixels_i.double()
    noisy_pixels_j_double = noisy_pixels_j.double()
    noisy_disparity_i_double = noisy_disparity_i.double()
    noisy_disparity_j_double = noisy_disparity_j.double()
    source_depth = (intrinsic[0, 0] * VISUAL_BASELINE_M / noisy_disparity_i_double).float()
    target_depth = (intrinsic[0, 0] * VISUAL_BASELINE_M / noisy_disparity_j_double).float()
    source_points = pixel2point_NED(noisy_pixels_i_double, source_depth.double(), intrinsic)
    anchored_points_world = (pose_initial[frame_i] * source_points).reshape(-1, 3)

    obs1_covariance = _stereo_covariance_ned(
        noisy_pixels_i_double,
        noisy_disparity_i_double,
        intrinsic,
        VISUAL_BASELINE_M,
        condition.pixel_sigma,
        condition.disparity_sigma,
    )
    obs2_covariance = _stereo_covariance_ned(
        noisy_pixels_j_double,
        noisy_disparity_j_double,
        intrinsic,
        VISUAL_BASELINE_M,
        condition.pixel_sigma,
        condition.disparity_sigma,
    )
    pixel_covariance = torch.tensor(
        [condition.pixel_sigma**2, condition.pixel_sigma**2, 0.0],
        dtype=torch.float32,
    ).repeat(noisy_pixels_i.shape[0], 1)
    disparity_covariance = torch.full(
        (noisy_pixels_i.shape[0], 1),
        condition.disparity_sigma**2,
        dtype=torch.float32,
    )
    observations = MatchObs.init(
        {
            "pixel1_uv": noisy_pixels_i,
            "pixel1_d": source_depth.unsqueeze(-1),
            "pixel2_uv": noisy_pixels_j,
            "pixel2_d": target_depth.unsqueeze(-1),
            "pixel1_disp": noisy_disparity_i.unsqueeze(-1),
            "pixel2_disp": noisy_disparity_j.unsqueeze(-1),
            "pixel1_uv_cov": pixel_covariance.clone(),
            "pixel2_uv_cov": pixel_covariance.clone(),
            "pixel1_d_cov": obs1_covariance[:, 0, 0].float().unsqueeze(-1),
            "pixel2_d_cov": obs2_covariance[:, 0, 0].float().unsqueeze(-1),
            "pixel1_disp_cov": disparity_covariance.clone(),
            "pixel2_disp_cov": disparity_covariance.clone(),
            "obs1_covTc": obs1_covariance,
            "obs2_covTc": obs2_covariance,
        }
    )

    rotation_world_from_source = pose_initial[frame_i].rotation().matrix().reshape(3, 3).double()
    point_covariance_world = (
        rotation_world_from_source @ obs1_covariance @ rotation_world_from_source.transpose(-1, -2)
    )
    points = PointNode.init(
        {
            "pos_Tw": anchored_points_world.float(),
            "cov_Tw": point_covariance_world,
            "color": torch.zeros((noisy_pixels_i.shape[0], 3), dtype=torch.uint8),
        }
    )
    from_pose = pose_initial[frame_i : frame_i + 1]
    init_motion = pose_initial[frame_j : frame_j + 1]
    observation_count = len(observations)
    return GraphInput(
        frame_idx=torch.tensor([frame_j], dtype=torch.long),
        from_idx=torch.tensor([frame_i], dtype=torch.long),
        init_motion=init_motion,
        from_pose=from_pose,
        baseline=torch.tensor([VISUAL_BASELINE_M], dtype=torch.float32),
        observations=observations,
        points=points,
        images_intrinsic=intrinsic.float(),
        edges_index=torch.zeros((observation_count,), dtype=torch.long),
        device="cpu",
        visual_obs_cov_mean=_visual_covariance_mean(observations),
        num_observations=observation_count,
    )


def generate_visual_input(
    config: SyntheticSequenceConfig,
    truth: EvaluationTruth,
    condition: str,
) -> EstimatorVisualInput:
    try:
        visual_condition = _VISUAL_CONDITIONS[condition]
    except KeyError as exc:
        raise ValueError(f"Unsupported visual condition: {condition}") from exc
    if truth.camera_time_ns.numel() < 2:
        raise ValueError("Visual input requires at least two camera frames")

    intrinsic = VISUAL_INTRINSIC.clone()
    intrinsic_double = intrinsic.double()
    pose_initial, relative_motion_initial = _visual_initial_poses(config, truth, visual_condition)
    candidates_world = _candidate_points_world(truth)
    measurement_generator = torch.Generator().manual_seed(config.seed + 11)
    image_margin = 1.0 + 5.0 * visual_condition.pixel_sigma
    edges: list[GraphInput] = []
    for frame_i in range(truth.camera_time_ns.numel() - 1):
        frame_j = frame_i + 1
        _, points_i, points_j, pixels_i, pixels_j = _common_visible_points(
            candidates_world,
            truth.pose_body_to_world[frame_i],
            truth.pose_body_to_world[frame_j],
            intrinsic_double,
            image_margin,
        )
        edges.append(
            _build_visual_edge(
                frame_i,
                frame_j,
                points_i,
                points_j,
                pixels_i,
                pixels_j,
                pose_initial,
                intrinsic_double,
                visual_condition,
                measurement_generator,
            )
        )

    return EstimatorVisualInput(
        intrinsic=intrinsic,
        baseline_m=VISUAL_BASELINE_M,
        pose_initial=pose_initial,
        relative_motion_initial=relative_motion_initial,
        edges=tuple(edges),
    )


def write_visual_artifact(output_dir: str | Path, visual_input: EstimatorVisualInput) -> Path:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = out_dir / "synthetic_visual_observations.npz"

    point_counts = np.asarray([len(edge.observations) for edge in visual_input.edges], dtype=np.int64)
    point_offsets = np.concatenate((np.zeros(1, dtype=np.int64), np.cumsum(point_counts[:-1], dtype=np.int64)))
    arrays: dict[str, np.ndarray] = {
        "intrinsic": visual_input.intrinsic.cpu().numpy(),
        "baseline_m": np.asarray(visual_input.baseline_m, dtype=np.float64),
        "pose_initial": visual_input.pose_initial.tensor().cpu().numpy(),
        "relative_motion_initial": visual_input.relative_motion_initial.tensor().cpu().numpy(),
        "edge_from_idx": torch.cat([edge.from_idx.reshape(-1) for edge in visual_input.edges]).cpu().numpy(),
        "edge_frame_idx": torch.cat([edge.frame_idx.reshape(-1) for edge in visual_input.edges]).cpu().numpy(),
        "edge_from_pose": torch.cat(
            [edge.from_pose.tensor().reshape(1, 7) for edge in visual_input.edges], dim=0
        ).cpu().numpy(),
        "edge_init_motion": torch.cat(
            [edge.init_motion.tensor().reshape(1, 7) for edge in visual_input.edges], dim=0
        ).cpu().numpy(),
        "edge_baseline": torch.stack(
            [edge.baseline.reshape(-1) for edge in visual_input.edges], dim=0
        ).cpu().numpy(),
        "edge_images_intrinsic": torch.stack(
            [edge.images_intrinsic for edge in visual_input.edges], dim=0
        ).cpu().numpy(),
        "edge_edges_index": torch.cat(
            [edge.edges_index.reshape(-1) for edge in visual_input.edges], dim=0
        ).cpu().numpy(),
        "edge_device_code": np.zeros(len(visual_input.edges), dtype=np.int8),
        "edge_num_observations": np.asarray(
            [edge.num_observations for edge in visual_input.edges], dtype=np.int64
        ),
        "edge_visual_obs_cov_mean": np.asarray(
            [edge.visual_obs_cov_mean for edge in visual_input.edges], dtype=np.float64
        ),
        "edge_point_offset": point_offsets,
        "edge_point_count": point_counts,
    }
    for key in visual_input.edges[0].observations.data:
        arrays[f"edge_{key}"] = torch.cat(
            [edge.observations.data[key] for edge in visual_input.edges], dim=0
        ).cpu().numpy()
    for key in visual_input.edges[0].points.data:
        arrays[f"edge_{key}"] = torch.cat(
            [edge.points.data[key] for edge in visual_input.edges], dim=0
        ).cpu().numpy()

    np.savez(artifact_path, **arrays)
    return artifact_path


def _write_csv(path: Path, header: list[str], rows: list[list[object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def write_sensor_artifacts(
    output_dir: str | Path,
    config: SyntheticSequenceConfig,
    truth: EvaluationTruth,
    imu_input: EstimatorIMUInput,
) -> dict[str, Path]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    imu_time_s, _ = _sample_times(config.duration_s, config.imu_rate_hz)
    imu_motion = _motion_samples(config, imu_time_s)
    imu_pose = imu_motion["pose"]
    imu_position = imu_motion["position"]
    imu_velocity = imu_motion["velocity"]
    imu_acceleration = imu_motion["acceleration"]
    imu_angular_velocity = imu_motion["angular_velocity"]

    ground_truth_path = out_dir / "synthetic_ground_truth.csv"
    imu_path = out_dir / "synthetic_imu.csv"
    manifest_path = out_dir / "generation_manifest.json"

    gt_rows: list[list[object]] = []
    quaternions = imu_pose.rotation().tensor().reshape(-1, 4)
    for idx, timestamp_ns in enumerate(truth.imu_time_ns.tolist()):
        gt_rows.append(
            [
                timestamp_ns,
                *imu_position[idx].tolist(),
                *quaternions[idx].tolist(),
                *imu_velocity[idx].tolist(),
                *imu_acceleration[idx].tolist(),
                *imu_angular_velocity[idx].tolist(),
                *truth.true_acc_bias[idx].tolist(),
                *truth.true_gyro_bias[idx].tolist(),
            ]
        )
    _write_csv(
        ground_truth_path,
        [
            "timestamp_ns",
            "position_world_x",
            "position_world_y",
            "position_world_z",
            "orientation_body_to_world_qx",
            "orientation_body_to_world_qy",
            "orientation_body_to_world_qz",
            "orientation_body_to_world_qw",
            "velocity_world_x",
            "velocity_world_y",
            "velocity_world_z",
            "acceleration_world_x",
            "acceleration_world_y",
            "acceleration_world_z",
            "angular_velocity_body_x",
            "angular_velocity_body_y",
            "angular_velocity_body_z",
            "true_acc_bias_x",
            "true_acc_bias_y",
            "true_acc_bias_z",
            "true_gyro_bias_x",
            "true_gyro_bias_y",
            "true_gyro_bias_z",
        ],
        gt_rows,
    )

    imu_rows = [
        [
            timestamp_ns,
            *imu_input.measured_acc_body[idx].tolist(),
            *imu_input.measured_gyro_body[idx].tolist(),
        ]
        for idx, timestamp_ns in enumerate(imu_input.time_ns.tolist())
    ]
    _write_csv(
        imu_path,
        [
            "timestamp_ns",
            "measured_acc_body_x",
            "measured_acc_body_y",
            "measured_acc_body_z",
            "measured_gyro_body_x",
            "measured_gyro_body_y",
            "measured_gyro_body_z",
        ],
        imu_rows,
    )

    manifest = {
        "duration_s": config.duration_s,
        "trajectory_period_s": config.trajectory_period_s,
        "camera_rate_hz": config.camera_rate_hz,
        "imu_rate_hz": config.imu_rate_hz,
        "radius_m": config.radius_m,
        "depth_amplitude_m": config.depth_amplitude_m,
        "gravity_m_s2": config.gravity_m_s2,
        "seed": config.seed,
        "frame_convention": "world/body NED",
        "orientation_profile": "roll=0, pitch=0, yaw=2*pi*t/trajectory_period_s",
        "motion_sampling_contract": MOTION_SAMPLING_CONTRACT,
        "noise_parameter_semantics": "continuous-time density",
        "bias_mode": truth.bias_mode,
        "noise_mode": truth.noise_mode,
        "noise_parameters": {
            "sigma_acc": SIGMA_ACC,
            "sigma_gyro": SIGMA_GYRO,
            "sigma_acc_w": SIGMA_ACC_W,
            "sigma_gyro_w": SIGMA_GYRO_W,
        },
        "base_bias": {
            "acc_m_s2": BASE_ACC_BIAS.tolist(),
            "gyro_rad_s": BASE_GYRO_BIAS.tolist(),
        },
        "formulas": [
            "theta = 2*pi*t/trajectory_period_s",
            "specific_force_body = R_body_to_world^T * (acceleration_world - gravity_world)",
            "gyro_body = [0, 0, 2*pi/trajectory_period_s] at each IMU timestamp",
            "white_noise_sample_std = sigma_density*sqrt(imu_rate_hz)",
            "bias_random_walk_step_std = sigma_walk_density/sqrt(imu_rate_hz)",
            "measurement = truth_signal + bias + optional white noise",
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    return {
        "ground_truth_csv": ground_truth_path,
        "imu_csv": imu_path,
        "manifest_json": manifest_path,
    }
