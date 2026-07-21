from __future__ import annotations

import math
import time
from dataclasses import dataclass, replace
from types import SimpleNamespace
from typing import Callable

import pandas as pd
import pypose as pp
import torch

from Module.IMUPreintegration import preintegrate_imu
from Module.Optimization.TwoFramePGO.Graphs import (
    GraphInput,
    LocalWindowGraphInput,
    LocalWindowInertialGraph,
)
from Module.Optimization.TwoFramePGO.Optimizer import TwoFrame_PGO
from Scripts.synthetic_w3_validation_data import (
    EstimatorIMUInput,
    EstimatorVisualInput,
    SIGMA_ACC,
    SIGMA_ACC_W,
    SIGMA_GYRO,
    SIGMA_GYRO_W,
    SyntheticSequenceConfig,
)


ProgressEvent = dict[str, int | float | str]
ProgressCallback = Callable[[ProgressEvent], None]

_INTEGER_DTYPES = {
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
}


@dataclass(frozen=True)
class ValidationCase:
    name: str
    visual_condition: str
    imu_noise_mode: str
    bias_mode: str
    bias_enabled: bool
    writeback: str


@dataclass(frozen=True)
class CaseRunResult:
    case: ValidationCase
    frame_time_ns: torch.Tensor
    pose_est: pp.LieTensor
    velocity_est: torch.Tensor
    acc_bias_est: torch.Tensor
    gyro_bias_est: torch.Tensor
    frame_diagnostics: pd.DataFrame
    num_windows: int
    local_ba_window_sizes: list[int]
    max_shift_source_bias_error: float
    max_shift_source_pose_translation_error_m: float
    max_shift_source_pose_rotation_error_rad: float
    max_shift_source_velocity_error_m_s: float
    bias_state_active: bool


@dataclass(frozen=True)
class _StoredEdge:
    graph_input: GraphInput
    interval_time_ns: torch.Tensor
    creation_source_rotation: torch.Tensor
    creation_source_acc_bias: torch.Tensor
    creation_source_gyro_bias: torch.Tensor


_BIAS_EDGE_FIELDS = (
    "imu_vio_prev_acc_bias",
    "imu_vio_prev_gyro_bias",
    "imu_vio_curr_acc_bias_init",
    "imu_vio_curr_gyro_bias_init",
    "imu_vio_linearized_acc_bias",
    "imu_vio_linearized_gyro_bias",
    "imu_vio_bias_jacobian",
    "imu_vio_bias_rw_cov",
)


def build_cases(duration_s: float) -> list[ValidationCase]:
    cases = [
        ValidationCase(
            name="visual_initial",
            visual_condition="drifted_visual",
            imu_noise_mode="fixed_seed_normal",
            bias_mode="constant_bias",
            bias_enabled=False,
            writeback="none",
        ),
        ValidationCase(
            name="w3_bias_locked_zero",
            visual_condition="drifted_visual",
            imu_noise_mode="fixed_seed_normal",
            bias_mode="constant_bias",
            bias_enabled=False,
            writeback="all_optimized",
        ),
        ValidationCase(
            name="w3_full_current",
            visual_condition="drifted_visual",
            imu_noise_mode="fixed_seed_normal",
            bias_mode="constant_bias",
            bias_enabled=True,
            writeback="current",
        ),
        ValidationCase(
            name="w3_full_all",
            visual_condition="drifted_visual",
            imu_noise_mode="fixed_seed_normal",
            bias_mode="constant_bias",
            bias_enabled=True,
            writeback="all_optimized",
        ),
        ValidationCase(
            name="w3_full_all_zero_mean",
            visual_condition="clean_visual",
            imu_noise_mode="mean_measurement",
            bias_mode="zero_bias",
            bias_enabled=True,
            writeback="all_optimized",
        ),
        ValidationCase(
            name="w3_full_all_zero_normal",
            visual_condition="drifted_visual",
            imu_noise_mode="fixed_seed_normal",
            bias_mode="zero_bias",
            bias_enabled=True,
            writeback="all_optimized",
        ),
        ValidationCase(
            name="w3_full_all_drifting_bias",
            visual_condition="drifted_visual",
            imu_noise_mode="fixed_seed_normal",
            bias_mode="drifting_bias",
            bias_enabled=True,
            writeback="all_optimized",
        ),
    ]
    if math.isclose(float(duration_s), 10.0, rel_tol=0.0, abs_tol=1e-9):
        cases = [case for case in cases if case.name != "w3_full_all_zero_mean"]
    return cases


def _frame_time_ns(config: SyntheticSequenceConfig, frame_count: int) -> torch.Tensor:
    if frame_count < 0:
        raise ValueError("frame_count must be non-negative")
    rate_hz = float(config.camera_rate_hz)
    if rate_hz <= 0.0:
        raise ValueError("camera_rate_hz must be positive")
    return torch.round(
        torch.arange(frame_count, dtype=torch.float64) * (1e9 / rate_hz)
    ).to(torch.int64)


def _validate_runner_timeline(
    config: SyntheticSequenceConfig,
    visual_input: EstimatorVisualInput,
    case: ValidationCase,
) -> int:
    duration_s = float(config.duration_s)
    camera_rate_hz = float(config.camera_rate_hz)
    if camera_rate_hz <= 0.0:
        raise ValueError("camera_rate_hz must be positive")
    interval_count = int(round(duration_s * camera_rate_hz))
    aligned_duration_s = interval_count / camera_rate_hz
    if not math.isclose(duration_s, aligned_duration_s, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(
            f"duration_s={duration_s} is not aligned with camera_rate_hz={camera_rate_hz}"
        )

    expected_frame_count = interval_count + 1
    pose_count = int(visual_input.pose_initial.shape[0])
    if pose_count != expected_frame_count:
        raise ValueError(
            f"expected_frame_count={expected_frame_count} from config but pose_initial={pose_count}"
        )
    expected_edge_count = pose_count - 1
    if len(visual_input.edges) != expected_edge_count:
        raise ValueError(
            f"len(edges)={len(visual_input.edges)} but expected {expected_edge_count}"
        )

    def scalar_integer_index(edge_index: int, name: str, value: torch.Tensor) -> int:
        if not isinstance(value, torch.Tensor) or value.dtype not in _INTEGER_DTYPES:
            raise ValueError(f"Visual edge {edge_index} {name} must use integer dtype")
        flattened = value.reshape(-1)
        if flattened.numel() != 1:
            raise ValueError(f"Visual edge {edge_index} {name} must contain exactly one value")
        return int(flattened.item())

    for edge_index, edge in enumerate(visual_input.edges):
        from_index = scalar_integer_index(edge_index, "from_idx", edge.from_idx)
        frame_index = scalar_integer_index(edge_index, "frame_idx", edge.frame_idx)
        if from_index != edge_index or frame_index != edge_index + 1:
            raise ValueError(
                f"Visual edge {edge_index} must have from_idx={edge_index} and "
                f"frame_idx={edge_index + 1}; got from_idx={from_index}, frame_idx={frame_index}"
            )
        edges_index = edge.edges_index
        if not isinstance(edges_index, torch.Tensor) or edges_index.dtype not in _INTEGER_DTYPES:
            raise ValueError(f"Visual edge {edge_index} edges_index must use integer dtype")
        if edges_index.ndim != 1:
            raise ValueError(f"Visual edge {edge_index} edges_index must be 1D")
        expected_observations = int(edge.num_observations)
        if edges_index.numel() != expected_observations:
            raise ValueError(
                f"Visual edge {edge_index} edges_index length={edges_index.numel()} "
                f"but num_observations={expected_observations}"
            )
        if bool(torch.count_nonzero(edges_index)):
            raise ValueError(
                f"Visual edge {edge_index} single-pose edge edges_index must be all zero"
            )

    if case.name != "visual_initial" and pose_count < 3:
        raise ValueError("W3 validation requires at least 3 frames")
    return pose_count


def _validate_imu_input(imu_input: EstimatorIMUInput) -> int:
    time_ns = imu_input.time_ns
    if not isinstance(time_ns, torch.Tensor) or time_ns.ndim != 1 or time_ns.dtype != torch.int64:
        raise ValueError("time_ns must be 1D int64")
    sample_count = int(time_ns.numel())
    if sample_count < 2:
        raise ValueError("IMU input requires at least 2 samples")

    for name, values in (
        ("measured_acc_body", imu_input.measured_acc_body),
        ("measured_gyro_body", imu_input.measured_gyro_body),
    ):
        if not isinstance(values, torch.Tensor) or tuple(values.shape) != (sample_count, 3):
            raise ValueError(
                f"{name} must have shape ({sample_count}, 3); got {getattr(values, 'shape', None)}"
            )
        if not torch.is_floating_point(values):
            raise ValueError(f"{name} must be floating point")
        if not bool(torch.isfinite(values).all()):
            raise ValueError(f"{name} must be finite")

    if not bool(torch.all(time_ns[1:] > time_ns[:-1])):
        raise ValueError("time_ns must be strictly increasing")
    return sample_count


def _interpolate_imu_sample(
    imu_input: EstimatorIMUInput,
    target_ns: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    time_ns = imu_input.time_ns
    target = time_ns.new_tensor(int(target_ns))
    idx = int(torch.searchsorted(time_ns, target, right=False).item())
    if idx < time_ns.numel() and int(time_ns[idx].item()) == int(target_ns):
        return (
            time_ns[idx : idx + 1],
            imu_input.measured_acc_body[idx : idx + 1],
            imu_input.measured_gyro_body[idx : idx + 1],
        )
    if idx <= 0 or idx >= time_ns.numel():
        raise ValueError(f"IMU interpolation target {target_ns} is outside the sample range")

    left = idx - 1
    right = idx
    left_ns = int(time_ns[left].item())
    right_ns = int(time_ns[right].item())
    alpha = (int(target_ns) - left_ns) / float(right_ns - left_ns)
    acc = imu_input.measured_acc_body[left : left + 1] + (
        imu_input.measured_acc_body[right : right + 1]
        - imu_input.measured_acc_body[left : left + 1]
    ) * alpha
    gyro = imu_input.measured_gyro_body[left : left + 1] + (
        imu_input.measured_gyro_body[right : right + 1]
        - imu_input.measured_gyro_body[left : left + 1]
    ) * alpha
    return time_ns.new_tensor([int(target_ns)]), acc, gyro


def query_imu_interval(
    imu_input: EstimatorIMUInput,
    start_ns: int,
    end_ns: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    _validate_imu_input(imu_input)
    if end_ns < start_ns:
        start_ns, end_ns = end_ns, start_ns
    if imu_input.time_ns.numel() == 0:
        raise ValueError("Cannot query an empty IMU input")
    first_ns = int(imu_input.time_ns[0].item())
    last_ns = int(imu_input.time_ns[-1].item())
    if start_ns < first_ns or end_ns > last_ns:
        raise ValueError(
            f"Camera interval [{start_ns}, {end_ns}] exceeds IMU range [{first_ns}, {last_ns}]"
        )

    start_time, start_acc, start_gyro = _interpolate_imu_sample(imu_input, start_ns)
    if start_ns == end_ns:
        return start_time, start_acc, start_gyro

    start_target = imu_input.time_ns.new_tensor(int(start_ns))
    end_target = imu_input.time_ns.new_tensor(int(end_ns))
    i0 = int(torch.searchsorted(imu_input.time_ns, start_target, right=True).item())
    i1 = int(torch.searchsorted(imu_input.time_ns, end_target, right=False).item())
    end_time, end_acc, end_gyro = _interpolate_imu_sample(imu_input, end_ns)
    result = (
        torch.cat([start_time, imu_input.time_ns[i0:i1], end_time], dim=0),
        torch.cat([start_acc, imu_input.measured_acc_body[i0:i1], end_acc], dim=0),
        torch.cat([start_gyro, imu_input.measured_gyro_body[i0:i1], end_gyro], dim=0),
    )
    if not bool(torch.all(result[0][1:] > result[0][:-1])):
        raise ValueError("query_imu_interval result must be strictly increasing")
    return result


def _visual_velocity_world(
    config: SyntheticSequenceConfig,
    visual_input: EstimatorVisualInput,
) -> torch.Tensor:
    position = visual_input.pose_initial.translation().reshape(-1, 3).float()
    if position.shape[0] <= 1:
        return torch.zeros_like(position)
    time_ns = _frame_time_ns(config, position.shape[0])
    dt = (time_ns[1:] - time_ns[:-1]).to(torch.float32).unsqueeze(-1) * 1e-9
    interval_velocity = (position[1:] - position[:-1]) / dt
    return torch.cat([interval_velocity[:1], interval_velocity], dim=0)


def _fake_map(
    config: SyntheticSequenceConfig,
    visual_input: EstimatorVisualInput,
) -> SimpleNamespace:
    frame_count = int(visual_input.pose_initial.shape[0])
    velocity = _visual_velocity_world(config, visual_input)
    previous_velocity = velocity.clone()
    if frame_count > 1:
        previous_velocity[1:] = velocity[:-1]
    zeros = torch.zeros((frame_count, 3), dtype=torch.float32)
    return SimpleNamespace(
        frames=SimpleNamespace(
            data={
                "pose": visual_input.pose_initial.tensor().float().clone(),
                "imu_vio_prev_velocity_world": previous_velocity,
                "imu_vio_curr_velocity_init_world": velocity.clone(),
                "imu_vio_velocity_world": velocity.clone(),
                "imu_vio_prev_acc_bias": zeros.clone(),
                "imu_vio_prev_gyro_bias": zeros.clone(),
                "imu_vio_acc_bias": zeros.clone(),
                "imu_vio_gyro_bias": zeros.clone(),
            }
        )
    )


def _build_stored_edge(
    config: SyntheticSequenceConfig,
    imu_input: EstimatorIMUInput,
    visual_edge: GraphInput,
    frame_time_ns: torch.Tensor,
    frame_data: dict[str, torch.Tensor],
) -> _StoredEdge:
    source_index = int(visual_edge.from_idx.reshape(-1)[0].item())
    target_index = int(visual_edge.frame_idx.reshape(-1)[0].item())
    if target_index != source_index + 1:
        raise ValueError("Synthetic W=3 runner requires adjacent visual edges")
    if source_index < 0 or target_index >= frame_time_ns.numel():
        raise IndexError("Visual edge index is outside the generated camera timeline")

    interval_time_ns, interval_acc, interval_gyro = query_imu_interval(
        imu_input,
        int(frame_time_ns[source_index].item()),
        int(frame_time_ns[target_index].item()),
    )
    source_pose = pp.SE3(frame_data["pose"][source_index : source_index + 1].clone())
    target_pose = pp.SE3(frame_data["pose"][target_index : target_index + 1].clone())
    source_acc_bias = frame_data["imu_vio_acc_bias"][source_index].clone().float()
    source_gyro_bias = frame_data["imu_vio_gyro_bias"][source_index].clone().float()
    preintegrated = preintegrate_imu(
        time_ns=interval_time_ns,
        acc=interval_acc.float(),
        gyro=interval_gyro.float(),
        R0_world=source_pose.rotation(),
        gravity=float(config.gravity_m_s2),
        sigma_acc=SIGMA_ACC,
        sigma_gyro=SIGMA_GYRO,
        sigma_acc_w=SIGMA_ACC_W,
        sigma_gyro_w=SIGMA_GYRO_W,
        acc_bias=source_acc_bias,
        gyro_bias=source_gyro_bias,
    )
    if (
        preintegrated.bias_jacobian is None
        or preintegrated.bias_rw_cov is None
        or preintegrated.linearized_acc_bias is None
        or preintegrated.linearized_gyro_bias is None
    ):
        raise RuntimeError("Production preintegration did not return Bias linearization fields")
    interval_dt_s = float((interval_time_ns[-1] - interval_time_ns[0]).item()) * 1e-9
    if not math.isclose(preintegrated.dt_total, interval_dt_s, rel_tol=0.0, abs_tol=1e-12):
        raise RuntimeError("Production preintegration duration does not match camera endpoints")

    graph_input = replace(
        visual_edge,
        from_pose=source_pose,
        init_motion=target_pose,
        device="cpu",
        imu_vio_factor_enable=True,
        imu_vio_prev_velocity_world=frame_data["imu_vio_velocity_world"][source_index].clone(),
        imu_vio_curr_velocity_init_world=frame_data["imu_vio_curr_velocity_init_world"][
            target_index
        ].clone(),
        imu_vio_prev_acc_bias=source_acc_bias.clone(),
        imu_vio_prev_gyro_bias=source_gyro_bias.clone(),
        imu_vio_curr_acc_bias_init=frame_data["imu_vio_acc_bias"][target_index].clone(),
        imu_vio_curr_gyro_bias_init=frame_data["imu_vio_gyro_bias"][target_index].clone(),
        imu_vio_linearized_acc_bias=preintegrated.linearized_acc_bias.clone().float(),
        imu_vio_linearized_gyro_bias=preintegrated.linearized_gyro_bias.clone().float(),
        imu_vio_bias_jacobian=preintegrated.bias_jacobian.clone().float(),
        imu_vio_bias_rw_cov=preintegrated.bias_rw_cov.clone().float(),
        imu_vio_delta_rotvec=preintegrated.delta_R.Log().tensor().reshape(3).clone().float(),
        imu_vio_delta_v=preintegrated.delta_v.clone().float(),
        imu_vio_delta_p=preintegrated.delta_p.clone().float(),
        imu_vio_cov=preintegrated.cov.clone().float(),
        imu_vio_dt=torch.tensor([interval_dt_s], dtype=torch.float64),
        imu_vio_sensor_T_imu=pp.identity_SE3(1, dtype=torch.float32).tensor(),
        imu_vio_alpha_p=1.0,
        imu_vio_alpha_v=1.0,
        imu_vio_alpha_R=1.0,
    )
    return _StoredEdge(
        graph_input=graph_input,
        interval_time_ns=interval_time_ns.clone(),
        creation_source_rotation=source_pose.rotation().tensor().detach().clone(),
        creation_source_acc_bias=source_acc_bias.detach().clone(),
        creation_source_gyro_bias=source_gyro_bias.detach().clone(),
    )


def _rebase_window_edge(
    stored_edge: _StoredEdge,
    frame_data: dict[str, torch.Tensor],
    bias_enabled: bool,
) -> GraphInput:
    edge = stored_edge.graph_input
    source_index = int(edge.from_idx.reshape(-1)[0].item())
    target_index = int(edge.frame_idx.reshape(-1)[0].item())
    updates = {
        "from_pose": pp.SE3(frame_data["pose"][source_index : source_index + 1].clone()),
        "init_motion": pp.SE3(frame_data["pose"][target_index : target_index + 1].clone()),
        "imu_vio_prev_velocity_world": frame_data["imu_vio_prev_velocity_world"][
            target_index
        ].clone(),
        "imu_vio_curr_velocity_init_world": frame_data["imu_vio_curr_velocity_init_world"][
            target_index
        ].clone(),
        "imu_vio_prev_acc_bias": frame_data["imu_vio_prev_acc_bias"][target_index].clone(),
        "imu_vio_prev_gyro_bias": frame_data["imu_vio_prev_gyro_bias"][target_index].clone(),
        "imu_vio_curr_acc_bias_init": frame_data["imu_vio_acc_bias"][target_index].clone(),
        "imu_vio_curr_gyro_bias_init": frame_data["imu_vio_gyro_bias"][target_index].clone(),
    }
    if not bias_enabled:
        updates.update({name: None for name in _BIAS_EDGE_FIELDS})
    return replace(edge, **updates)


def _optimizer_config(writeback: str) -> SimpleNamespace:
    return SimpleNamespace(
        graph_type="disp",
        device="cpu",
        vectorize=True,
        parallel=False,
        autodiff=True,
        imu_rot_prior=True,
        imu_factor_mode="local_inertial_ba",
        local_ba_window_size=3,
        local_ba_fix_first_frame=True,
        local_ba_writeback=writeback,
        vio_causal_diagnostics_enable=True,
        vio_causal_diagnostics_interval=1_000_000_000,
    )


def _tensor_equal(left: torch.Tensor | pp.LieTensor | None, right: torch.Tensor | pp.LieTensor | None) -> bool:
    if left is None or right is None:
        return left is None and right is None
    left_tensor = left.tensor() if isinstance(left, pp.LieTensor) else left
    right_tensor = right.tensor() if isinstance(right, pp.LieTensor) else right
    return bool(torch.equal(left_tensor.detach().cpu(), right_tensor.detach().cpu()))


def _edge_runtime_checks(
    edge: GraphInput,
    frame_data: dict[str, torch.Tensor],
    bias_enabled: bool,
) -> dict[str, bool]:
    source_index = int(edge.from_idx.reshape(-1)[0].item())
    target_index = int(edge.frame_idx.reshape(-1)[0].item())
    checks = {
        "from_pose": _tensor_equal(
            edge.from_pose,
            frame_data["pose"][source_index : source_index + 1],
        ),
        "init_motion": _tensor_equal(
            edge.init_motion,
            frame_data["pose"][target_index : target_index + 1],
        ),
        "prev_velocity": _tensor_equal(
            edge.imu_vio_prev_velocity_world,
            frame_data["imu_vio_prev_velocity_world"][target_index],
        )
        and _tensor_equal(
            edge.imu_vio_prev_velocity_world,
            frame_data["imu_vio_velocity_world"][source_index],
        ),
        "curr_velocity": _tensor_equal(
            edge.imu_vio_curr_velocity_init_world,
            frame_data["imu_vio_curr_velocity_init_world"][target_index],
        )
        and _tensor_equal(
            edge.imu_vio_curr_velocity_init_world,
            frame_data["imu_vio_velocity_world"][target_index],
        ),
    }
    if bias_enabled:
        checks.update(
            {
                "prev_acc_bias": _tensor_equal(
                    edge.imu_vio_prev_acc_bias,
                    frame_data["imu_vio_prev_acc_bias"][target_index],
                )
                and _tensor_equal(
                    edge.imu_vio_prev_acc_bias,
                    frame_data["imu_vio_acc_bias"][source_index],
                ),
                "prev_gyro_bias": _tensor_equal(
                    edge.imu_vio_prev_gyro_bias,
                    frame_data["imu_vio_prev_gyro_bias"][target_index],
                )
                and _tensor_equal(
                    edge.imu_vio_prev_gyro_bias,
                    frame_data["imu_vio_gyro_bias"][source_index],
                ),
                "curr_acc_bias": _tensor_equal(
                    edge.imu_vio_curr_acc_bias_init,
                    frame_data["imu_vio_acc_bias"][target_index],
                ),
                "curr_gyro_bias": _tensor_equal(
                    edge.imu_vio_curr_gyro_bias_init,
                    frame_data["imu_vio_gyro_bias"][target_index],
                ),
            }
        )
    else:
        checks.update(
            {
                "prev_acc_bias": edge.imu_vio_prev_acc_bias is None,
                "prev_gyro_bias": edge.imu_vio_prev_gyro_bias is None,
                "curr_acc_bias": edge.imu_vio_curr_acc_bias_init is None,
                "curr_gyro_bias": edge.imu_vio_curr_gyro_bias_init is None,
            }
        )
    return checks


def _preintegration_fields_immutable(
    stored_edge: _StoredEdge,
    rebased_edge: GraphInput,
    bias_enabled: bool,
) -> bool:
    main_fields = (
        "imu_vio_delta_rotvec",
        "imu_vio_delta_v",
        "imu_vio_delta_p",
        "imu_vio_cov",
        "imu_vio_dt",
        "imu_vio_sensor_T_imu",
    )
    if not all(
        _tensor_equal(getattr(stored_edge.graph_input, name), getattr(rebased_edge, name))
        for name in main_fields
    ):
        return False
    if bias_enabled:
        return all(
            _tensor_equal(getattr(stored_edge.graph_input, name), getattr(rebased_edge, name))
            for name in _BIAS_EDGE_FIELDS[4:]
        )
    return all(getattr(stored_edge.graph_input, name) is not None for name in _BIAS_EDGE_FIELDS[4:])


def _bias_residual_jacobians(
    graph: LocalWindowInertialGraph,
    edge_position: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    edge = graph.edges[edge_position]
    local_i, local_j = graph._edge_local_indices(edge)
    poses = graph._all_poses()
    residual = graph._imu_edge_residual(
        edge,
        poses[local_i],
        poses[local_j],
        local_i,
        local_j,
    ).reshape(9)
    parameter_rows = int(graph.acc_bias_window.shape[0])
    jac_acc_rows: list[torch.Tensor] = []
    jac_gyro_rows: list[torch.Tensor] = []
    for residual_index in range(residual.numel()):
        if residual[residual_index].requires_grad:
            grad_acc, grad_gyro = torch.autograd.grad(
                residual[residual_index],
                (graph.acc_bias_window, graph.gyro_bias_window),
                retain_graph=True,
                allow_unused=True,
            )
        else:
            grad_acc = None
            grad_gyro = None
        jac_acc_rows.append(torch.zeros_like(graph.acc_bias_window) if grad_acc is None else grad_acc)
        jac_gyro_rows.append(torch.zeros_like(graph.gyro_bias_window) if grad_gyro is None else grad_gyro)
    jac_acc = torch.stack(jac_acc_rows, dim=0).reshape(9, parameter_rows, 3)
    jac_gyro = torch.stack(jac_gyro_rows, dim=0).reshape(9, parameter_rows, 3)

    def for_local_frame(local_index: int) -> torch.Tensor:
        parameter_index = local_index - 1 if graph.fixed_first_frame else local_index
        if parameter_index < 0 or parameter_index >= parameter_rows:
            return residual.new_zeros((9, 6))
        return torch.cat(
            [jac_acc[:, parameter_index, :], jac_gyro[:, parameter_index, :]],
            dim=1,
        )

    return for_local_frame(local_i), for_local_frame(local_j)


def _bias_rw_energy(
    frame_indices: torch.Tensor,
    edges: list[GraphInput],
    acc_bias: torch.Tensor,
    gyro_bias: torch.Tensor,
) -> float:
    index_to_local = {int(value.item()): index for index, value in enumerate(frame_indices)}
    energy = torch.zeros((), dtype=torch.float64)
    for edge in edges:
        if not LocalWindowInertialGraph._edge_has_bias(edge):
            continue
        source = index_to_local[int(edge.from_idx.reshape(-1)[0].item())]
        target = index_to_local[int(edge.frame_idx.reshape(-1)[0].item())]
        residual = torch.cat(
            [
                acc_bias[target].double() - acc_bias[source].double(),
                gyro_bias[target].double() - gyro_bias[source].double(),
            ]
        )
        covariance = edge.imu_vio_bias_rw_cov.reshape(6, 6).double()
        covariance = 0.5 * (covariance + covariance.mT)
        information = torch.linalg.pinv(covariance, hermitian=True)
        energy = energy + residual @ information @ residual
    return float(torch.clamp(energy, min=0.0).item())


def _window_bias_rw_energies(
    window_input: LocalWindowGraphInput,
    diagnostic_graph: LocalWindowInertialGraph,
    output,
) -> tuple[float, float]:
    if output.window_acc_bias is None or output.window_gyro_bias is None:
        raise RuntimeError("Production local-window output omitted final Bias states")
    initial_energy = _bias_rw_energy(
        window_input.frame_indices,
        window_input.edges,
        diagnostic_graph._all_acc_bias(),
        diagnostic_graph._all_gyro_bias(),
    )
    final_energy = _bias_rw_energy(
        window_input.frame_indices,
        window_input.edges,
        output.window_acc_bias.reshape(-1, 3),
        output.window_gyro_bias.reshape(-1, 3),
    )
    return initial_energy, final_energy


def _float_or_nan(value: float | int | None) -> float:
    return float("nan") if value is None else float(value)


def _window_diagnostic_row(
    case: ValidationCase,
    target_index: int,
    context: dict,
    frame_data: dict[str, torch.Tensor],
    window_input: LocalWindowGraphInput,
    stored_edges: tuple[_StoredEdge, _StoredEdge],
    rebased_edges: tuple[GraphInput, GraphInput],
    diagnostic_graph: LocalWindowInertialGraph,
    output,
    shift_source_bias_error: float,
    shift_source_pose_translation_error_m: float,
    shift_source_pose_rotation_error_rad: float,
    shift_source_velocity_error_m_s: float,
) -> dict[str, object]:
    if (
        output.window_motions is None
        or output.window_velocity_world is None
        or output.window_acc_bias is None
        or output.window_gyro_bias is None
    ):
        raise RuntimeError("Production local-window output omitted optimized state arrays")

    initial_pose = diagnostic_graph._all_poses().tensor().detach().cpu().double()
    initial_velocity = diagnostic_graph._all_velocity().detach().cpu().double()
    initial_acc_bias = diagnostic_graph._all_acc_bias().detach().cpu().double()
    initial_gyro_bias = diagnostic_graph._all_gyro_bias().detach().cpu().double()
    final_pose = output.window_motions.reshape(-1, 7).detach().cpu().double()
    final_velocity = output.window_velocity_world.reshape(-1, 3).detach().cpu().double()
    final_acc_bias = output.window_acc_bias.reshape(-1, 3).detach().cpu().double()
    final_gyro_bias = output.window_gyro_bias.reshape(-1, 3).detach().cpu().double()

    relative_update = pp.SE3(initial_pose).Inv() @ pp.SE3(final_pose)
    pose_translation_update = float(relative_update.translation().norm().item())
    pose_rotation_update = float(relative_update.rotation().Log().tensor().norm().item())
    velocity_update = float((final_velocity - initial_velocity).norm().item())
    acc_bias_update = float((final_acc_bias - initial_acc_bias).norm().item())
    gyro_bias_update = float((final_gyro_bias - initial_gyro_bias).norm().item())

    initial_rw, final_rw = _window_bias_rw_energies(
        window_input,
        diagnostic_graph,
        output,
    )

    source_jacobian, terminal_jacobian = _bias_residual_jacobians(
        diagnostic_graph,
        edge_position=1,
    )
    source_rank = int(torch.linalg.matrix_rank(source_jacobian.detach()).item())
    terminal_rank = int(torch.linalg.matrix_rank(terminal_jacobian.detach()).item())

    leading_checks = _edge_runtime_checks(rebased_edges[0], frame_data, case.bias_enabled)
    trailing_checks = _edge_runtime_checks(rebased_edges[1], frame_data, case.bias_enabled)
    leading_immutable = _preintegration_fields_immutable(
        stored_edges[0], rebased_edges[0], case.bias_enabled
    )
    trailing_immutable = _preintegration_fields_immutable(
        stored_edges[1], rebased_edges[1], case.bias_enabled
    )

    leading_original = stored_edges[0].graph_input
    leading_linearized_bias = torch.cat(
        [
            leading_original.imu_vio_linearized_acc_bias.reshape(3),
            leading_original.imu_vio_linearized_gyro_bias.reshape(3),
        ]
    ).float()
    leading_source_index = int(leading_original.from_idx.reshape(-1)[0].item())
    leading_runtime_source_bias = torch.cat(
        [
            frame_data["imu_vio_acc_bias"][leading_source_index],
            frame_data["imu_vio_gyro_bias"][leading_source_index],
        ]
    ).float()
    linearization_runtime_difference = float(
        (leading_linearized_bias - leading_runtime_source_bias).norm().item()
    )

    trailing_original = stored_edges[1].graph_input
    trailing_linearized_matches_creation = _tensor_equal(
        trailing_original.imu_vio_linearized_acc_bias,
        stored_edges[1].creation_source_acc_bias,
    ) and _tensor_equal(
        trailing_original.imu_vio_linearized_gyro_bias,
        stored_edges[1].creation_source_gyro_bias,
    )
    trailing_rotation_matches_creation = _tensor_equal(
        pp.SE3(rebased_edges[1].from_pose).rotation(),
        stored_edges[1].creation_source_rotation,
    )

    bias_state_active = all(
        LocalWindowInertialGraph._edge_has_bias(edge) for edge in rebased_edges
    )
    bias_parameter_structurally_present = all(
        isinstance(getattr(diagnostic_graph, name, None), torch.nn.Parameter)
        for name in ("acc_bias_window", "gyro_bias_window")
    )
    vio_factor_active = all(
        LocalWindowInertialGraph._edge_has_vio(edge) for edge in rebased_edges
    )
    imu_residual_rows = sum(
        3 + (2 if LocalWindowInertialGraph._edge_has_bias(edge) else 0)
        for edge in rebased_edges
        if LocalWindowInertialGraph._edge_has_vio(edge)
    )
    cleared_counts = [
        sum(getattr(edge, name) is None for name in _BIAS_EDGE_FIELDS)
        for edge in rebased_edges
    ]
    bias_fields_cleared_per_edge = (
        cleared_counts[0] if len(set(cleared_counts)) == 1 else min(cleared_counts)
    )

    energy_and_profile = {
        "initial_visual_energy": _float_or_nan(output.initial_energy_visual_weighted),
        "final_visual_energy": _float_or_nan(output.energy_visual_weighted),
        "initial_p_energy": _float_or_nan(output.initial_energy_p_weighted),
        "final_p_energy": _float_or_nan(output.energy_p_weighted),
        "initial_v_energy": _float_or_nan(output.initial_energy_v_weighted),
        "final_v_energy": _float_or_nan(output.energy_v_weighted),
        "initial_R_energy": _float_or_nan(output.initial_energy_R_weighted),
        "final_R_energy": _float_or_nan(output.energy_R_weighted),
        "initial_main_imu_energy": _float_or_nan(output.initial_energy_imu_weighted),
        "final_main_imu_energy": _float_or_nan(output.energy_imu_weighted),
        "initial_bias_rw_energy": initial_rw,
        "final_bias_rw_energy": final_rw,
        "pose_translation_update_norm": pose_translation_update,
        "pose_rotation_update_norm": pose_rotation_update,
        "velocity_update_norm": velocity_update,
        "acc_bias_update_norm": acc_bias_update,
        "gyro_bias_update_norm": gyro_bias_update,
        "local_ba_graph_build_s": _float_or_nan(output.local_ba_graph_build_s),
        "local_ba_lm_s": _float_or_nan(output.local_ba_lm_s),
        "local_ba_refine_s": _float_or_nan(output.local_ba_refine_s),
        "local_ba_optimize_total_s": _float_or_nan(output.local_ba_optimize_total_s),
    }
    state_finite = all(
        bool(torch.isfinite(value).all())
        for value in (final_pose, final_velocity, final_acc_bias, final_gyro_bias)
    )
    scalar_finite = all(math.isfinite(value) for value in energy_and_profile.values())
    optimizer_final_loss_finite = output.final_loss is not None and math.isfinite(
        float(output.final_loss)
    )

    row: dict[str, object] = {
        "case": case.name,
        "window_start": target_index - 2,
        "window_target": target_index,
        "graph_class": type(diagnostic_graph).__name__,
        "context_device": str(context["device"]),
        "bias_state_active": bias_state_active,
        "bias_parameter_structurally_present": bias_parameter_structurally_present,
        "bias_parameter_limitation": (
            "parameters_exist_even_when_bias_factor_is_disabled"
        ),
        "vio_factor_active": vio_factor_active,
        "imu_residual_rows": imu_residual_rows,
        **energy_and_profile,
        "source_main_imu_bias_jacobian_rank": source_rank,
        "terminal_main_imu_bias_jacobian_rank": terminal_rank,
        "finite": state_finite and scalar_finite,
        "optimizer_completed": True,
        "optimizer_final_loss_finite": optimizer_final_loss_finite,
        "optimizer_convergence_status": "unknown_not_exposed_by_production",
        "influence_sampled": int(output.influence_sampled),
        "leading_interval_start_ns": int(stored_edges[0].interval_time_ns[0].item()),
        "leading_interval_end_ns": int(stored_edges[0].interval_time_ns[-1].item()),
        "leading_imu_vio_dt_s": float(rebased_edges[0].imu_vio_dt.reshape(-1)[0].item()),
        "trailing_interval_start_ns": int(stored_edges[1].interval_time_ns[0].item()),
        "trailing_interval_end_ns": int(stored_edges[1].interval_time_ns[-1].item()),
        "trailing_imu_vio_dt_s": float(rebased_edges[1].imu_vio_dt.reshape(-1)[0].item()),
        "all_edges_rebase_fields_match_runtime": all(leading_checks.values())
        and all(trailing_checks.values()),
        "leading_preintegration_fields_immutable": leading_immutable,
        "trailing_preintegration_fields_immutable": trailing_immutable,
        "trailing_linearization_bias_matches_creation_source": trailing_linearized_matches_creation,
        "trailing_preintegration_rotation_matches_creation_source": trailing_rotation_matches_creation,
        "leading_linearized_bias": tuple(float(v) for v in leading_linearized_bias.tolist()),
        "leading_runtime_source_bias": tuple(
            float(v) for v in leading_runtime_source_bias.tolist()
        ),
        "leading_linearization_runtime_bias_difference_norm": linearization_runtime_difference,
        "shift_source_bias_error": shift_source_bias_error,
        "shift_source_pose_translation_error_m": shift_source_pose_translation_error_m,
        "shift_source_pose_rotation_error_rad": shift_source_pose_rotation_error_rad,
        "shift_source_velocity_error_m_s": shift_source_velocity_error_m_s,
        "bias_fields_cleared_per_edge": bias_fields_cleared_per_edge,
        "edge_12_source_state_matches_runtime": (
            int(rebased_edges[0].from_idx.reshape(-1)[0].item()) == 1
            and int(rebased_edges[0].frame_idx.reshape(-1)[0].item()) == 2
            and all(leading_checks.values())
        ),
    }
    for prefix, checks in (("leading", leading_checks), ("trailing", trailing_checks)):
        for name, matches in checks.items():
            row[f"{prefix}_{name}_matches_runtime"] = matches
    return row


def run_validation_case(
    config: SyntheticSequenceConfig,
    imu_input: EstimatorIMUInput,
    visual_input: EstimatorVisualInput,
    case: ValidationCase,
    progress: ProgressCallback | None = None,
) -> CaseRunResult:
    _validate_imu_input(imu_input)
    frame_count = _validate_runner_timeline(config, visual_input, case)
    frame_time_ns = _frame_time_ns(config, frame_count)
    if case.name == "visual_initial":
        fake_map = _fake_map(config, visual_input)
        frame_data = fake_map.frames.data
        return CaseRunResult(
            case=case,
            frame_time_ns=frame_time_ns,
            pose_est=pp.SE3(visual_input.pose_initial.tensor().clone()),
            velocity_est=frame_data["imu_vio_velocity_world"].clone(),
            acc_bias_est=frame_data["imu_vio_acc_bias"].clone(),
            gyro_bias_est=frame_data["imu_vio_gyro_bias"].clone(),
            frame_diagnostics=pd.DataFrame(),
            num_windows=0,
            local_ba_window_sizes=[],
            max_shift_source_bias_error=0.0,
            max_shift_source_pose_translation_error_m=0.0,
            max_shift_source_pose_rotation_error_rad=0.0,
            max_shift_source_velocity_error_m_s=0.0,
            bias_state_active=False,
        )

    if case.writeback not in {"current", "all_optimized"}:
        raise ValueError(f"Unsupported W=3 writeback mode: {case.writeback}")

    fake_map = _fake_map(config, visual_input)
    frame_data = fake_map.frames.data
    context = TwoFrame_PGO.init_context(_optimizer_config(case.writeback))
    if str(context["device"]) != "cpu":
        raise RuntimeError("Synthetic W=3 validation must execute on CPU")
    optimizer = object.__new__(TwoFrame_PGO)
    optimizer.last_pair_diagnostics = {}
    optimizer.last_breakpoint_trace = None
    optimizer.last_breakpoint_frame_indices = []

    stored_edges: dict[int, _StoredEdge] = {}
    diagnostics: list[dict[str, object]] = []
    local_ba_window_sizes: list[int] = []
    max_shift_source_bias_error = 0.0
    max_shift_source_pose_translation_error_m = 0.0
    max_shift_source_pose_rotation_error_rad = 0.0
    max_shift_source_velocity_error_m_s = 0.0
    previous_output = None
    bias_state_seen = False
    total_windows = max(0, frame_count - 2)
    run_start = time.perf_counter()

    for target_index in range(1, frame_count):
        shift_source_bias_error = 0.0
        shift_source_pose_translation_error_m = 0.0
        shift_source_pose_rotation_error_rad = 0.0
        shift_source_velocity_error_m_s = 0.0
        if previous_output is not None:
            if (
                previous_output.window_frame_indices is None
                or previous_output.window_motions is None
                or previous_output.window_velocity_world is None
                or previous_output.window_acc_bias is None
                or previous_output.window_gyro_bias is None
            ):
                raise RuntimeError("Previous production window output omitted optimized state history")
            middle_frame_index = int(
                previous_output.window_frame_indices.reshape(-1)[1].item()
            )
            if middle_frame_index != target_index - 2:
                raise RuntimeError("Overlapping W=3 windows do not share the expected middle frame")
            expected_middle_bias = torch.cat(
                [
                    previous_output.window_acc_bias.reshape(-1, 3)[1],
                    previous_output.window_gyro_bias.reshape(-1, 3)[1],
                ]
            ).float()
            stored_middle_bias = torch.cat(
                [
                    frame_data["imu_vio_acc_bias"][middle_frame_index],
                    frame_data["imu_vio_gyro_bias"][middle_frame_index],
                ]
            ).float()
            shift_source_bias_error = float(
                (expected_middle_bias - stored_middle_bias).norm().item()
            )
            expected_middle_pose = pp.SE3(
                previous_output.window_motions.reshape(-1, 7)[1].clone()
            )
            stored_middle_pose = pp.SE3(frame_data["pose"][middle_frame_index].clone())
            pose_difference = expected_middle_pose.Inv() @ stored_middle_pose
            shift_source_pose_translation_error_m = float(
                pose_difference.translation().norm().item()
            )
            shift_source_pose_rotation_error_rad = float(
                pose_difference.rotation().Log().tensor().norm().item()
            )
            shift_source_velocity_error_m_s = float(
                (
                    previous_output.window_velocity_world.reshape(-1, 3)[1]
                    - frame_data["imu_vio_velocity_world"][middle_frame_index]
                )
                .norm()
                .item()
            )
            max_shift_source_bias_error = max(
                max_shift_source_bias_error,
                shift_source_bias_error,
            )
            max_shift_source_pose_translation_error_m = max(
                max_shift_source_pose_translation_error_m,
                shift_source_pose_translation_error_m,
            )
            max_shift_source_pose_rotation_error_rad = max(
                max_shift_source_pose_rotation_error_rad,
                shift_source_pose_rotation_error_rad,
            )
            max_shift_source_velocity_error_m_s = max(
                max_shift_source_velocity_error_m_s,
                shift_source_velocity_error_m_s,
            )

        stored_edges[target_index] = _build_stored_edge(
            config,
            imu_input,
            visual_input.edges[target_index - 1],
            frame_time_ns,
            frame_data,
        )
        if target_index < 2:
            continue

        leading_stored = stored_edges[target_index - 1]
        trailing_stored = stored_edges[target_index]
        leading_edge = _rebase_window_edge(
            leading_stored,
            frame_data,
            bias_enabled=case.bias_enabled,
        )
        trailing_edge = _rebase_window_edge(
            trailing_stored,
            frame_data,
            bias_enabled=case.bias_enabled,
        )
        window_input = LocalWindowGraphInput(
            frame_indices=torch.arange(target_index - 2, target_index + 1, dtype=torch.long),
            frame_poses=frame_data["pose"][target_index - 2 : target_index + 1].clone(),
            edges=[leading_edge, trailing_edge],
            fixed_first_frame=True,
            writeback=case.writeback,
            device="cpu",
        )
        diagnostic_graph = LocalWindowInertialGraph(window_input).double()
        context, output = TwoFrame_PGO._optimize(context, window_input)
        row = _window_diagnostic_row(
            case=case,
            target_index=target_index,
            context=context,
            frame_data=frame_data,
            window_input=window_input,
            stored_edges=(leading_stored, trailing_stored),
            rebased_edges=(leading_edge, trailing_edge),
            diagnostic_graph=diagnostic_graph,
            output=output,
            shift_source_bias_error=shift_source_bias_error,
            shift_source_pose_translation_error_m=shift_source_pose_translation_error_m,
            shift_source_pose_rotation_error_rad=shift_source_pose_rotation_error_rad,
            shift_source_velocity_error_m_s=shift_source_velocity_error_m_s,
        )
        optimizer._write_local_ba_graph_data(output, fake_map)
        diagnostics.append(row)
        local_ba_window_sizes.append(int(output.local_ba_window_size))
        bias_state_seen = bias_state_seen or bool(row["bias_state_active"])
        previous_output = output

        if progress is not None:
            progress(
                {
                    "case": case.name,
                    "completed_windows": len(diagnostics),
                    "total_windows": total_windows,
                    "frame_index": target_index,
                    "elapsed_s": float(time.perf_counter() - run_start),
                }
            )

    return CaseRunResult(
        case=case,
        frame_time_ns=frame_time_ns,
        pose_est=pp.SE3(frame_data["pose"].clone()),
        velocity_est=frame_data["imu_vio_velocity_world"].clone(),
        acc_bias_est=frame_data["imu_vio_acc_bias"].clone(),
        gyro_bias_est=frame_data["imu_vio_gyro_bias"].clone(),
        frame_diagnostics=pd.DataFrame(diagnostics),
        num_windows=len(diagnostics),
        local_ba_window_sizes=local_ba_window_sizes,
        max_shift_source_bias_error=max_shift_source_bias_error,
        max_shift_source_pose_translation_error_m=max_shift_source_pose_translation_error_m,
        max_shift_source_pose_rotation_error_rad=max_shift_source_pose_rotation_error_rad,
        max_shift_source_velocity_error_m_s=max_shift_source_velocity_error_m_s,
        bias_state_active=bias_state_seen,
    )
