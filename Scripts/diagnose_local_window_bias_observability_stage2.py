#!/usr/bin/env python3
"""Validate local-window IMU bias observability with production factors.

This is a deterministic synthetic diagnostic. It does not run MACVO, start a
HoloOcean sequence, or change estimator defaults. Four camera states and three
adjacent IMU edges are generated with known poses, velocities, and biases. The
production preintegrator, local-window graph, IMU residual, random-walk factor,
and write-back path are then exercised directly.
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import sys
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pypose as pp
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Module.Map import MatchObs, PointNode
from Module.Optimization.TwoFramePGO.Graphs import (
    GraphInput,
    GraphOutput,
    LocalWindowGraphInput,
    LocalWindowInertialGraph,
)
from Module.Optimization.TwoFramePGO.Optimizer import TwoFrame_PGO
from Utility.Point import point2pixel_NED


def _load_preintegrate_imu():
    module_path = PROJECT_ROOT / "Module" / "IMUPreintegration.py"
    spec = importlib.util.spec_from_file_location("stage2_imu_preintegration", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load IMU preintegration module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.preintegrate_imu


preintegrate_imu = _load_preintegrate_imu()


CAMERA_RATE_HZ = 30.0
IMU_RATE_HZ = 100.0
GRAVITY_M_S2 = 9.8
FRAME_TIME_NS = torch.tensor([0, 33_333_333, 66_666_667, 100_000_000], dtype=torch.long)

SIGMA_ACC = 0.014125836301591877
SIGMA_GYRO = 0.0018289805264456304
SIGMA_ACC_W = 0.00038607055555878352
SIGMA_GYRO_W = 3.5786418170002551e-05

BASE_ACC_BIAS = torch.tensor([0.0040, -0.0030, 0.0020], dtype=torch.float64)
BASE_GYRO_BIAS = torch.tensor([0.00040, -0.00030, 0.00200], dtype=torch.float64)
ACC_BIAS_STEP = torch.tensor([0.00045, -0.00020, 0.00030], dtype=torch.float64)
GYRO_BIAS_STEP = torch.tensor([0.00005, -0.00003, 0.00008], dtype=torch.float64)

WORLD_VELOCITY = torch.tensor([0.80, 0.15, -0.03], dtype=torch.float64)
YAW_RATE_RAD_S = 0.12

K = torch.tensor(
    [[320.0, 0.0, 320.0], [0.0, 320.0, 240.0], [0.0, 0.0, 1.0]],
    dtype=torch.float32,
)
BASELINE_M = 0.12
BODY_POINTS = torch.tensor(
    [
        [4.0, -0.60, -0.25],
        [5.5, 0.35, 0.30],
        [6.5, -0.10, 0.55],
        [7.0, 0.80, -0.40],
    ],
    dtype=torch.float32,
)


@dataclass(frozen=True)
class FourFrameBiasProblem:
    frame_time_ns: torch.Tensor
    imu_time_ns: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    frame_poses: pp.LieTensor
    velocities_world: torch.Tensor
    acc_bias_truth: torch.Tensor
    gyro_bias_truth: torch.Tensor
    edges: tuple[GraphInput, GraphInput, GraphInput]


def _bias_truth(mode: str) -> tuple[torch.Tensor, torch.Tensor]:
    if mode == "zero":
        zeros = torch.zeros((4, 3), dtype=torch.float64)
        return zeros.clone(), zeros.clone()
    if mode == "constant":
        return BASE_ACC_BIAS.repeat(4, 1), BASE_GYRO_BIAS.repeat(4, 1)
    if mode == "drift":
        scale = torch.arange(4, dtype=torch.float64).reshape(-1, 1)
        return (
            BASE_ACC_BIAS.reshape(1, 3) + scale * ACC_BIAS_STEP.reshape(1, 3),
            BASE_GYRO_BIAS.reshape(1, 3) + scale * GYRO_BIAS_STEP.reshape(1, 3),
        )
    raise ValueError(f"Unsupported bias mode: {mode}")


def _yaw_pose(time_s: float) -> pp.LieTensor:
    yaw = YAW_RATE_RAD_S * time_s
    translation = WORLD_VELOCITY * time_s
    half = 0.5 * yaw
    tensor = torch.tensor(
        [[
            float(translation[0]),
            float(translation[1]),
            float(translation[2]),
            0.0,
            0.0,
            math.sin(half),
            math.cos(half),
        ]],
        dtype=torch.float64,
    )
    return pp.SE3(tensor)


def _interval_times(start_ns: int, end_ns: int) -> torch.Tensor:
    step_ns = int(round(1e9 / IMU_RATE_HZ))
    samples = [start_ns]
    next_ns = start_ns + step_ns
    while next_ns < end_ns:
        samples.append(next_ns)
        next_ns += step_ns
    if samples[-1] != end_ns:
        samples.append(end_ns)
    return torch.tensor(samples, dtype=torch.long)


def _make_observations(pose_i: pp.LieTensor, pose_j: pp.LieTensor) -> tuple[MatchObs, PointNode]:
    body_j = BODY_POINTS.double()
    point_w = (pose_j * body_j).reshape(-1, 3)
    body_i = (pose_i.Inv() * point_w).reshape(-1, 3)

    pixel1_uv = point2pixel_NED(body_i.float(), K)
    pixel2_uv = point2pixel_NED(body_j.float(), K)
    pixel1_disp = (K[0, 0] * BASELINE_M / body_i[:, 0].float()).reshape(-1, 1)
    pixel2_disp = (K[0, 0] * BASELINE_M / body_j[:, 0].float()).reshape(-1, 1)
    count = body_j.shape[0]

    observations = MatchObs.init(
        {
            "pixel1_uv": pixel1_uv,
            "pixel2_uv": pixel2_uv,
            "pixel1_d": body_i[:, 0:1].float(),
            "pixel2_d": body_j[:, 0:1].float(),
            "pixel1_disp": pixel1_disp,
            "pixel2_disp": pixel2_disp,
            "pixel1_disp_cov": torch.ones((count, 1), dtype=torch.float32),
            "pixel2_disp_cov": torch.ones((count, 1), dtype=torch.float32),
            "obs1_covTc": torch.eye(3, dtype=torch.float64).repeat(count, 1, 1),
            "obs2_covTc": torch.eye(3, dtype=torch.float64).repeat(count, 1, 1),
            "pixel1_uv_cov": torch.tensor([[1.0, 1.0, 0.0]], dtype=torch.float32).repeat(count, 1),
            "pixel2_uv_cov": torch.tensor([[1.0, 1.0, 0.0]], dtype=torch.float32).repeat(count, 1),
            "pixel1_d_cov": torch.ones((count, 1), dtype=torch.float32),
            "pixel2_d_cov": torch.ones((count, 1), dtype=torch.float32),
        }
    )
    points = PointNode.init(
        {
            "pos_Tw": point_w.float(),
            "cov_Tw": torch.eye(3, dtype=torch.float64).repeat(count, 1, 1),
            "color": torch.zeros((count, 3), dtype=torch.uint8),
        }
    )
    return observations, points


def _make_edge(
    frame_i: int,
    frame_j: int,
    pose_i: pp.LieTensor,
    pose_j: pp.LieTensor,
    velocity_i: torch.Tensor,
    velocity_j: torch.Tensor,
    acc_bias_i: torch.Tensor,
    gyro_bias_i: torch.Tensor,
) -> tuple[GraphInput, torch.Tensor]:
    start_ns = int(FRAME_TIME_NS[frame_i].item())
    end_ns = int(FRAME_TIME_NS[frame_j].item())
    time_ns = _interval_times(start_ns, end_ns)

    # Production VIO consumes the internal NED/body convention. Stationary
    # specific force is therefore -g on body z. The trajectory has constant
    # world velocity and yaw-only rotation, so body z remains aligned with
    # world z and no additional kinematic acceleration is present.
    acc_sample = torch.tensor([0.0, 0.0, -GRAVITY_M_S2], dtype=torch.float64) + acc_bias_i
    gyro_sample = torch.tensor([0.0, 0.0, YAW_RATE_RAD_S], dtype=torch.float64) + gyro_bias_i
    acc = acc_sample.repeat(time_ns.numel(), 1).float()
    gyro = gyro_sample.repeat(time_ns.numel(), 1).float()

    preint = preintegrate_imu(
        time_ns=time_ns,
        acc=acc,
        gyro=gyro,
        R0_world=pose_i.rotation(),
        gravity=GRAVITY_M_S2,
        sigma_acc=SIGMA_ACC,
        sigma_gyro=SIGMA_GYRO,
        sigma_acc_w=SIGMA_ACC_W,
        sigma_gyro_w=SIGMA_GYRO_W,
        acc_bias=torch.zeros(3, dtype=torch.float32),
        gyro_bias=torch.zeros(3, dtype=torch.float32),
    )
    assert preint.bias_jacobian is not None
    assert preint.bias_rw_cov is not None
    assert preint.linearized_acc_bias is not None
    assert preint.linearized_gyro_bias is not None

    observations, points = _make_observations(pose_i, pose_j)
    count = observations.data["pixel2_uv"].shape[0]
    edge = GraphInput(
        frame_idx=torch.tensor([frame_j], dtype=torch.long),
        from_idx=torch.tensor([frame_i], dtype=torch.long),
        init_motion=pose_j.float(),
        from_pose=pose_i.float(),
        baseline=torch.tensor([BASELINE_M], dtype=torch.float32),
        observations=observations,
        points=points,
        images_intrinsic=K.clone(),
        edges_index=torch.zeros((count,), dtype=torch.long),
        device="cpu",
        imu_vio_factor_enable=True,
        imu_vio_prev_velocity_world=velocity_i.float(),
        imu_vio_curr_velocity_init_world=velocity_j.float(),
        imu_vio_prev_acc_bias=torch.zeros(3, dtype=torch.float32),
        imu_vio_prev_gyro_bias=torch.zeros(3, dtype=torch.float32),
        imu_vio_curr_acc_bias_init=torch.zeros(3, dtype=torch.float32),
        imu_vio_curr_gyro_bias_init=torch.zeros(3, dtype=torch.float32),
        imu_vio_linearized_acc_bias=preint.linearized_acc_bias.float(),
        imu_vio_linearized_gyro_bias=preint.linearized_gyro_bias.float(),
        imu_vio_bias_jacobian=preint.bias_jacobian.float(),
        imu_vio_bias_rw_cov=preint.bias_rw_cov.float(),
        imu_vio_delta_rotvec=preint.delta_R.Log().tensor().reshape(3).float(),
        imu_vio_delta_v=preint.delta_v.float(),
        imu_vio_delta_p=preint.delta_p.float(),
        imu_vio_cov=preint.cov.float(),
        imu_vio_dt=torch.tensor([preint.dt_total], dtype=torch.float32),
        imu_vio_sensor_T_imu=pp.identity_SE3(1, dtype=torch.float32).tensor(),
        imu_vio_alpha_p=1.0,
        imu_vio_alpha_v=1.0,
        imu_vio_alpha_R=1.0,
    )
    return edge, time_ns


def build_four_frame_problem(mode: str) -> FourFrameBiasProblem:
    acc_bias, gyro_bias = _bias_truth(mode)
    times_s = FRAME_TIME_NS.double() * 1e-9
    poses = pp.SE3(torch.cat([_yaw_pose(float(t)).tensor() for t in times_s], dim=0))
    velocities = WORLD_VELOCITY.repeat(4, 1)
    edge_payloads = tuple(
        _make_edge(
            frame_i=i,
            frame_j=i + 1,
            pose_i=poses[i],
            pose_j=poses[i + 1],
            velocity_i=velocities[i],
            velocity_j=velocities[i + 1],
            acc_bias_i=acc_bias[i],
            gyro_bias_i=gyro_bias[i],
        )
        for i in range(3)
    )
    edges = tuple(payload[0] for payload in edge_payloads)
    imu_time_ns = tuple(payload[1] for payload in edge_payloads)
    return FourFrameBiasProblem(
        frame_time_ns=FRAME_TIME_NS.clone(),
        imu_time_ns=imu_time_ns,
        frame_poses=poses,
        velocities_world=velocities,
        acc_bias_truth=acc_bias,
        gyro_bias_truth=gyro_bias,
        edges=edges,
    )


def _window_graph(problem: FourFrameBiasProblem, start: int, size: int, writeback: str) -> LocalWindowInertialGraph:
    end = start + size
    frame_indices = torch.arange(start, end, dtype=torch.long)
    graph = LocalWindowInertialGraph(
        LocalWindowGraphInput(
            frame_indices=frame_indices,
            frame_poses=problem.frame_poses[start:end].tensor().float(),
            edges=list(problem.edges[start:end - 1]),
            fixed_first_frame=True,
            writeback=writeback,
            device="cpu",
        )
    ).double()

    # The boundary truth represents a valid prior from earlier history. Every
    # later bias starts at zero, which is the state whose observability is being
    # tested. Pose and velocity are fixed truth so they cannot absorb bias.
    with torch.no_grad():
        graph.fixed_velocity0.copy_(problem.velocities_world[start:start + 1])
        graph.fixed_acc_bias0.copy_(problem.acc_bias_truth[start:start + 1])
        graph.fixed_gyro_bias0.copy_(problem.gyro_bias_truth[start:start + 1])
        graph.velocity_window.copy_(problem.velocities_world[start + 1:end])
        graph.acc_bias_window.zero_()
        graph.gyro_bias_window.zero_()
    graph.pose_window.requires_grad_(False)
    graph.velocity_window.requires_grad_(False)
    return graph


def _information(covariance: torch.Tensor) -> torch.Tensor:
    cov = 0.5 * (covariance + covariance.mT)
    return torch.linalg.pinv(cov, hermitian=True)


def _imu_edge_energy(graph: LocalWindowInertialGraph, edge_position: int) -> torch.Tensor:
    edge = graph.edges[edge_position]
    local_i, local_j = graph._edge_local_indices(edge)
    poses = graph._all_poses()
    residual = graph._imu_edge_residual(edge, poses[local_i], poses[local_j], local_i, local_j).reshape(9)
    covariance = graph._imu_covariances[edge_position].to(residual)
    return residual @ _information(covariance) @ residual


def _bias_residual_jacobians(
    graph: LocalWindowInertialGraph,
    edge_position: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the actual graph Jacobians wrt this edge's source and terminal Bias."""
    edge = graph.edges[edge_position]
    local_i, local_j = graph._edge_local_indices(edge)
    poses = graph._all_poses()
    residual = graph._imu_edge_residual(edge, poses[local_i], poses[local_j], local_i, local_j).reshape(9)
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
        jac_acc_rows.append(
            torch.zeros_like(graph.acc_bias_window) if grad_acc is None else grad_acc
        )
        jac_gyro_rows.append(
            torch.zeros_like(graph.gyro_bias_window) if grad_gyro is None else grad_gyro
        )
    jac_acc = torch.stack(jac_acc_rows, dim=0).reshape(9, parameter_rows, 3)
    jac_gyro = torch.stack(jac_gyro_rows, dim=0).reshape(9, parameter_rows, 3)

    def jacobian_for_local_frame(local_index: int) -> torch.Tensor:
        parameter_index = local_index - 1 if graph.fixed_first_frame else local_index
        if parameter_index < 0 or parameter_index >= parameter_rows:
            return residual.new_zeros((9, 6))
        return torch.cat(
            [jac_acc[:, parameter_index, :], jac_gyro[:, parameter_index, :]],
            dim=1,
        )

    return jacobian_for_local_frame(local_i), jacobian_for_local_frame(local_j)


def _jacobian_metrics(jacobian: torch.Tensor) -> tuple[float, int, float]:
    singular = torch.linalg.svdvals(jacobian.detach())
    norm = float(jacobian.detach().norm().cpu().item())
    if singular.numel() == 0 or float(singular.max().item()) <= 0.0:
        return norm, 0, 0.0
    threshold = max(float(singular.max().item()) * 1e-8, 1e-12)
    rank = int((singular > threshold).sum().item())
    return norm, rank, float(singular.min().cpu().item())


def _optimize_source_bias_from_imu_edge(
    graph: LocalWindowInertialGraph,
    edge_position: int,
) -> tuple[float, float]:
    """Isolate Bias recovery to one production IMU main residual only."""
    initial_imu = float(_imu_edge_energy(graph, edge_position).detach().cpu().item())
    optimizer = torch.optim.LBFGS(
        [graph.acc_bias_window, graph.gyro_bias_window],
        lr=1.0,
        max_iter=100,
        tolerance_grad=1e-13,
        tolerance_change=1e-15,
        line_search_fn="strong_wolfe",
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        loss = _imu_edge_energy(graph, edge_position)
        loss.backward()
        return loss

    optimizer.step(closure)
    final_imu = float(_imu_edge_energy(graph, edge_position).detach().cpu().item())
    return initial_imu, final_imu


def _cosine(estimate: torch.Tensor, truth: torch.Tensor) -> float:
    denom = float(estimate.norm().item() * truth.norm().item())
    if denom <= 1e-30:
        return 1.0 if float((estimate - truth).norm().item()) <= 1e-12 else 0.0
    return float((estimate @ truth).item() / denom)


def _relative_error(estimate: torch.Tensor, truth: torch.Tensor) -> float:
    truth_norm = float(truth.norm().item())
    error_norm = float((estimate - truth).norm().item())
    if truth_norm <= 1e-15:
        return 0.0 if error_norm <= 1e-12 else float("nan")
    return error_norm / truth_norm


def _base_row(case: str, problem: FourFrameBiasProblem) -> dict[str, float | int | str]:
    min_cov = min(float(edge.imu_vio_cov.diagonal().min().item()) for edge in problem.edges)
    camera_rate_hz = (
        (problem.frame_time_ns.numel() - 1)
        * 1e9
        / float(problem.frame_time_ns[-1] - problem.frame_time_ns[0])
    )
    imu_dt_ns = torch.cat([times[1:] - times[:-1] for times in problem.imu_time_ns]).double()
    imu_rate_hz = 1e9 / float(torch.median(imu_dt_ns).item())
    return {
        "case": case,
        "num_camera_frames": int(problem.frame_poses.shape[0]),
        "num_imu_edges": len(problem.edges),
        "camera_rate_hz": camera_rate_hz,
        "imu_rate_hz": imu_rate_hz,
        "preintegration_cov_min_diag": min_cov,
        "source_bias_residual_jacobian_norm": float("nan"),
        "source_bias_residual_jacobian_rank": -1,
        "source_bias_residual_jacobian_min_singular": float("nan"),
        "terminal_bias_residual_jacobian_norm": float("nan"),
        "terminal_bias_residual_jacobian_rank": -1,
        "terminal_bias_residual_jacobian_min_singular": float("nan"),
        "bias_recovery_objective": "not_applicable",
        "initial_imu_energy": float("nan"),
        "final_imu_energy": float("nan"),
        "acc_bias_cosine": float("nan"),
        "gyro_bias_cosine": float("nan"),
        "acc_bias_relative_error": float("nan"),
        "gyro_bias_relative_error": float("nan"),
        "estimated_acc_bias_norm": float("nan"),
        "estimated_gyro_bias_norm": float("nan"),
        "estimated_bias_error_norm": float("nan"),
        "zero_baseline_bias_error_norm": float("nan"),
        "persistence_baseline_bias_error_norm": float("nan"),
        "intermediate_writeback_applied": -1,
        "shifted_graph_constructed": 0,
        "shift_source_bias_error_norm": float("nan"),
    }


def _w2_row(problem: FourFrameBiasProblem) -> dict[str, float | int | str]:
    graph = _window_graph(problem, start=0, size=2, writeback="current")
    _, terminal_jacobian = _bias_residual_jacobians(graph, edge_position=0)
    terminal_norm, terminal_rank, terminal_min_singular = _jacobian_metrics(terminal_jacobian)
    row = _base_row("w2_terminal_constant", problem)
    row.update(
        {
            "terminal_bias_residual_jacobian_norm": terminal_norm,
            "terminal_bias_residual_jacobian_rank": terminal_rank,
            "terminal_bias_residual_jacobian_min_singular": terminal_min_singular,
        }
    )
    return row


def _w3_recovery_row(mode: str, case: str) -> tuple[dict[str, float | int | str], LocalWindowInertialGraph]:
    problem = build_four_frame_problem(mode)
    graph = _window_graph(problem, start=0, size=3, writeback="all_optimized")
    source_jacobian, terminal_jacobian = _bias_residual_jacobians(graph, edge_position=1)
    source_norm, source_rank, source_min_singular = _jacobian_metrics(source_jacobian)
    terminal_norm, terminal_rank, terminal_min_singular = _jacobian_metrics(terminal_jacobian)
    initial_imu, final_imu = _optimize_source_bias_from_imu_edge(graph, edge_position=1)

    estimated_acc = graph.acc_bias_window[0].detach().cpu()
    estimated_gyro = graph.gyro_bias_window[0].detach().cpu()
    truth_acc = problem.acc_bias_truth[1].cpu()
    truth_gyro = problem.gyro_bias_truth[1].cpu()
    estimated_error = torch.cat([estimated_acc - truth_acc, estimated_gyro - truth_gyro]).norm()
    zero_error = torch.cat([truth_acc, truth_gyro]).norm()
    persistence_error = torch.cat(
        [
            problem.acc_bias_truth[0].cpu() - truth_acc,
            problem.gyro_bias_truth[0].cpu() - truth_gyro,
        ]
    ).norm()

    row = _base_row(case, problem)
    row.update(
        {
            "source_bias_residual_jacobian_norm": source_norm,
            "source_bias_residual_jacobian_rank": source_rank,
            "source_bias_residual_jacobian_min_singular": source_min_singular,
            "terminal_bias_residual_jacobian_norm": terminal_norm,
            "terminal_bias_residual_jacobian_rank": terminal_rank,
            "terminal_bias_residual_jacobian_min_singular": terminal_min_singular,
            "bias_recovery_objective": "imu_edge_only",
            "initial_imu_energy": initial_imu,
            "final_imu_energy": final_imu,
            "acc_bias_cosine": _cosine(estimated_acc, truth_acc),
            "gyro_bias_cosine": _cosine(estimated_gyro, truth_gyro),
            "acc_bias_relative_error": _relative_error(estimated_acc, truth_acc),
            "gyro_bias_relative_error": _relative_error(estimated_gyro, truth_gyro),
            "estimated_acc_bias_norm": float(estimated_acc.norm().item()),
            "estimated_gyro_bias_norm": float(estimated_gyro.norm().item()),
            "estimated_bias_error_norm": float(estimated_error.item()),
            "zero_baseline_bias_error_norm": float(zero_error.item()),
            "persistence_baseline_bias_error_norm": float(persistence_error.item()),
        }
    )
    return row, graph


def _fake_map(problem: FourFrameBiasProblem) -> SimpleNamespace:
    zeros = torch.zeros((4, 3), dtype=torch.float32)
    prev_velocity = zeros.clone()
    curr_velocity = problem.velocities_world.float().clone()
    for target_index in range(1, 4):
        prev_velocity[target_index] = problem.velocities_world[target_index - 1].float()
    return SimpleNamespace(
        frames=SimpleNamespace(
            data={
                "pose": problem.frame_poses.tensor().float().clone(),
                "imu_vio_prev_velocity_world": prev_velocity,
                "imu_vio_curr_velocity_init_world": curr_velocity.clone(),
                "imu_vio_velocity_world": curr_velocity.clone(),
                "imu_vio_prev_acc_bias": zeros.clone(),
                "imu_vio_prev_gyro_bias": zeros.clone(),
                "imu_vio_acc_bias": zeros.clone(),
                "imu_vio_gyro_bias": zeros.clone(),
            }
        )
    )


def _edge_from_frame_state(edge: GraphInput, frame_data: dict[str, torch.Tensor]) -> GraphInput:
    from_index = int(edge.from_idx.reshape(-1)[0].item())
    target_index = int(edge.frame_idx.reshape(-1)[0].item())
    return replace(
        edge,
        from_pose=pp.SE3(frame_data["pose"][from_index:from_index + 1]),
        init_motion=pp.SE3(frame_data["pose"][target_index:target_index + 1]),
        imu_vio_prev_velocity_world=frame_data["imu_vio_prev_velocity_world"][target_index].clone(),
        imu_vio_curr_velocity_init_world=frame_data["imu_vio_curr_velocity_init_world"][target_index].clone(),
        imu_vio_prev_acc_bias=frame_data["imu_vio_prev_acc_bias"][target_index].clone(),
        imu_vio_prev_gyro_bias=frame_data["imu_vio_prev_gyro_bias"][target_index].clone(),
        imu_vio_curr_acc_bias_init=frame_data["imu_vio_acc_bias"][target_index].clone(),
        imu_vio_curr_gyro_bias_init=frame_data["imu_vio_gyro_bias"][target_index].clone(),
    )


def _shifted_graph_from_frame_state(
    problem: FourFrameBiasProblem,
    frame_data: dict[str, torch.Tensor],
) -> LocalWindowInertialGraph:
    shifted_edges = [
        _edge_from_frame_state(problem.edges[1], frame_data),
        _edge_from_frame_state(problem.edges[2], frame_data),
    ]
    return LocalWindowInertialGraph(
        LocalWindowGraphInput(
            frame_indices=torch.tensor([1, 2, 3], dtype=torch.long),
            frame_poses=frame_data["pose"][1:4].clone(),
            edges=shifted_edges,
            fixed_first_frame=True,
            writeback="all_optimized",
            device="cpu",
        )
    ).double()


def _writeback_row(
    problem: FourFrameBiasProblem,
    optimized_graph: LocalWindowInertialGraph,
    mode: str,
) -> dict[str, float | int | str]:
    output = optimized_graph.write_back()
    output = replace(output, local_ba_writeback=mode)
    fake_map = _fake_map(problem)
    optimizer = object.__new__(TwoFrame_PGO)
    optimizer.last_pair_diagnostics = {}
    optimizer.last_breakpoint_trace = None
    optimizer.last_breakpoint_frame_indices = []
    optimizer._write_local_ba_graph_data(output, fake_map)
    shifted_graph = _shifted_graph_from_frame_state(problem, fake_map.frames.data)

    middle_acc = output.window_acc_bias.reshape(-1, 3)[1]
    middle_gyro = output.window_gyro_bias.reshape(-1, 3)[1]
    shifted_source = torch.cat(
        [
            shifted_graph.fixed_acc_bias0.reshape(3).float(),
            shifted_graph.fixed_gyro_bias0.reshape(3).float(),
        ]
    )
    expected = torch.cat([middle_acc, middle_gyro])
    frame1_stored = torch.cat(
        [
            fake_map.frames.data["imu_vio_acc_bias"][1],
            fake_map.frames.data["imu_vio_gyro_bias"][1],
        ]
    )

    case = "w3_shift_all_optimized" if mode == "all_optimized" else "w3_shift_current"
    row = _base_row(case, problem)
    row["intermediate_writeback_applied"] = int(torch.allclose(frame1_stored, expected, atol=1e-9, rtol=0.0))
    row["shifted_graph_constructed"] = 1
    row["shift_source_bias_error_norm"] = float((shifted_source - expected).norm().item())
    return row


@lru_cache(maxsize=1)
def _cached_stage2_rows() -> tuple[dict[str, float | int | str], ...]:
    torch.manual_seed(0)
    constant_problem = build_four_frame_problem("constant")
    w2 = _w2_row(constant_problem)
    constant, constant_graph = _w3_recovery_row("constant", "w3_middle_constant")
    drift, _ = _w3_recovery_row("drift", "w3_middle_drift")
    zero, _ = _w3_recovery_row("zero", "w3_zero_bias")
    shift_current = _writeback_row(constant_problem, constant_graph, "current")
    shift_all = _writeback_row(constant_problem, constant_graph, "all_optimized")
    return w2, constant, drift, zero, shift_current, shift_all


def run_stage2_cases() -> pd.DataFrame:
    return pd.DataFrame([dict(row) for row in _cached_stage2_rows()])


def _decision(rows: pd.DataFrame) -> str:
    indexed = rows.set_index("case")
    w2 = indexed.loc["w2_terminal_constant"]
    w3 = indexed.loc["w3_middle_constant"]
    drift = indexed.loc["w3_middle_drift"]
    zero = indexed.loc["w3_zero_bias"]
    shift_current = indexed.loc["w3_shift_current"]
    shift_all = indexed.loc["w3_shift_all_optimized"]
    if (
        float(w2.terminal_bias_residual_jacobian_norm) < 1e-12
        and int(w2.terminal_bias_residual_jacobian_rank) == 0
        and float(w3.source_bias_residual_jacobian_norm) > 1e-6
        and int(w3.source_bias_residual_jacobian_rank) == 6
        and float(w3.source_bias_residual_jacobian_min_singular) > 1e-6
        and float(w3.terminal_bias_residual_jacobian_norm) < 1e-12
        and int(w3.terminal_bias_residual_jacobian_rank) == 0
        and str(w3.bias_recovery_objective) == "imu_edge_only"
        and float(w3.acc_bias_cosine) > 0.99
        and float(w3.gyro_bias_cosine) > 0.99
        and float(w3.acc_bias_relative_error) < 0.10
        and float(w3.gyro_bias_relative_error) < 0.10
        and float(w3.final_imu_energy) < float(w3.initial_imu_energy) * 0.05
        and float(zero.estimated_acc_bias_norm) < 1e-6
        and float(zero.estimated_gyro_bias_norm) < 1e-6
        and float(drift.estimated_bias_error_norm)
        < float(drift.persistence_baseline_bias_error_norm) * 0.10
        and int(shift_current.shifted_graph_constructed) == 1
        and int(shift_current.intermediate_writeback_applied) == 0
        and float(shift_current.shift_source_bias_error_norm) > 1e-6
        and int(shift_all.shifted_graph_constructed) == 1
        and int(shift_all.intermediate_writeback_applied) == 1
        and float(shift_all.shift_source_bias_error_norm) < 1e-8
    ):
        return "three_frame_bias_path_and_all_writeback_confirmed"
    return "local_window_bias_path_requires_further_investigation"


def write_stage2_report(output_dir: Path) -> tuple[Path, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = run_stage2_cases()
    indexed = rows.set_index("case")
    decision = _decision(rows)

    csv_path = output_dir / "local_window_bias_observability_cases.csv"
    report_path = output_dir / "local_window_bias_observability_summary_cn.md"
    rows.to_csv(csv_path, index=False, float_format="%.12g")

    w2 = indexed.loc["w2_terminal_constant"]
    w3 = indexed.loc["w3_middle_constant"]
    drift = indexed.loc["w3_middle_drift"]
    zero = indexed.loc["w3_zero_bias"]
    current = indexed.loc["w3_shift_current"]
    all_optimized = indexed.loc["w3_shift_all_optimized"]
    table = rows[
        [
            "case",
            "source_bias_residual_jacobian_rank",
            "terminal_bias_residual_jacobian_rank",
            "initial_imu_energy",
            "final_imu_energy",
            "estimated_bias_error_norm",
            "shift_source_bias_error_norm",
        ]
    ].to_markdown(index=False, floatfmt=".6g")

    report_path.write_text(
        f"""# 局部窗口 Bias 可观测性第二阶段验证

## 结论

诊断判定：`{decision}`。

在同一套生产 IMU 预积分与残差定义下，W=2 的终点 Bias 对当前 edge 的 9x6 残差 Jacobian 范数为 `{float(w2.terminal_bias_residual_jacobian_norm):.3e}`、秩为 `{int(w2.terminal_bias_residual_jacobian_rank)}`。W=3 中间帧 Bias 成为 `j -> k` 的源状态后，9x6 Jacobian 范数为 `{float(w3.source_bias_residual_jacobian_norm):.6g}`、秩为 `{int(w3.source_bias_residual_jacobian_rank)}`、最小奇异值为 `{float(w3.source_bias_residual_jacobian_min_singular):.6g}`；同一 edge 的终点 Bias Jacobian 仍为零。

固定真值 pose 和 velocity 后，Bias 恢复只最小化 `j -> k` 的 9 维 IMU 主残差，目标记为 `imu_edge_only`，不加入 Bias random-walk 项。常值 Bias 的 IMU 能量由 `{float(w3.initial_imu_energy):.6g}` 降到 `{float(w3.final_imu_energy):.6g}`，因此该恢复不能再由真值边界的 random-walk 先验代替完成。

窗口滑移实验还表明，`current` 写回后，使用帧字段重建的 `[j,k,l]` 图得到的固定源 Bias 误差为 `{float(current.shift_source_bias_error_norm):.6g}`；`all_optimized` 写回后重建同一图，对应误差为 `{float(all_optimized.shift_source_bias_error_norm):.6g}`。

这些结果验证的是 Bias 状态的拓扑可观测性和写回连续性，不代表场景轨迹精度，也不能单独证明 normal-noise 场景一定会改善。

## 实验条件

- 四个相机状态、三条相邻 IMU edge
- 相机频率：`{CAMERA_RATE_HZ:.0f} Hz`
- IMU 频率：`{IMU_RATE_HZ:.0f} Hz`
- 使用生产 `preintegrate_imu()`、`LocalWindowInertialGraph`、IMU 残差和 9x9 预积分协方差
- 生产图仍携带 6x6 Bias random-walk 协方差；但恢复隔离明确使用 `imu_edge_only`，不把 random-walk 能量放进目标函数
- Bias 恢复时固定全部 pose 和 velocity 为真值，仅优化 `j -> k` 的源 Bias；终点 Bias 无 Jacobian，保持初值

## 结构化结果

{table}

## 控制实验

- `w3_zero_bias`：估计 acc/gyro Bias 范数分别为 `{float(zero.estimated_acc_bias_norm):.3e}` 和 `{float(zero.estimated_gyro_bias_norm):.3e}`。
- `w3_middle_drift`：估计误差 `{float(drift.estimated_bias_error_norm):.6g}`，上一帧保持基线误差 `{float(drift.persistence_baseline_bias_error_norm):.6g}`，零值基线误差 `{float(drift.zero_baseline_bias_error_norm):.6g}`。
- `w3_shift_current`：只写最后一帧；重建的移位窗口仍读取到旧的中间 Bias。
- `w3_shift_all_optimized`：写回全部可优化帧并刷新字段；重建的移位窗口读取到优化后的中间 Bias。

## 工程含义

W=2 不是“优化器完全不能估 Bias”，而是当前固定起点、只优化终点的孤立两帧拓扑没有让终点 Bias 进入 IMU 主残差。W=3 提供了最小的跨 edge 观测路径；若要让该信息在连续滑窗中保留下来，必须使用等价于 `all_optimized` 的中间状态写回，或者引入包含同等历史信息的边缘化先验。

本验证没有检查未知初始 Bias 的启动过程，也没有证明完整视觉、IMU 与 random-walk 联合目标的场景级最优性。
""",
        encoding="utf-8",
    )
    return csv_path, report_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis_local_window_bias_observability_stage2_20260710"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    csv_path, report_path = write_stage2_report(args.output_dir)
    rows = pd.read_csv(csv_path)
    columns = [
        "case",
        "source_bias_residual_jacobian_rank",
        "terminal_bias_residual_jacobian_rank",
        "initial_imu_energy",
        "final_imu_energy",
        "estimated_bias_error_norm",
        "shift_source_bias_error_norm",
    ]
    print(rows[columns].to_string(index=False))
    print(f"Wrote CSV:    {csv_path}")
    print(f"Wrote report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
