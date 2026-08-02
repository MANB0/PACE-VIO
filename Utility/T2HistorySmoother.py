"""Offline full-history smoothing for archived PACE factors.

This module deliberately consumes the factors already written to ``tensor_map.npz``.
It does not run MACVO, alter the online two-state estimator, or feed smoothed states
back to the realtime path.  The residuals and local state convention are imported
from :mod:`Utility.TwoStateVIO`, so the audit uses the production PACE mathematics.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pypose as pp
import scipy.sparse
import scipy.sparse.linalg
import torch

from Utility.TwoStateVIO import (
    PAIR_DOF,
    STATE_DOF,
    ImuPreintegrationFactor,
    LinearizedUVDPoseFactor,
    NavigationState,
    SquareRootPrior,
    _bias_residual_and_analytic_jacobian,
    _imu_residual,
    _linearized_visual_residual_and_analytic_jacobian,
    _prior_residual_and_analytic_jacobian,
    linearized_uvd_pose_factor_from_normal_equations,
    make_diagonal_prior,
    retract_state,
    visual_whitened_residuals,
)


@dataclass(frozen=True)
class ArchivedT2Edge:
    frame_i: int
    frame_j: int
    imu: ImuPreintegrationFactor
    visual: LinearizedUVDPoseFactor
    cached_hessian: torch.Tensor
    cached_gradient: torch.Tensor


@dataclass(frozen=True)
class T2HistoryArchive:
    source_path: Path
    source_sha256: str
    frame_indices: np.ndarray
    timestamps_ns: np.ndarray
    online_states: tuple[NavigationState, ...]
    initial_prior: SquareRootPrior
    edges: tuple[ArchivedT2Edge, ...]
    extrinsic_CI: torch.Tensor


@dataclass(frozen=True)
class HistoryIteration:
    iteration: int
    accepted: bool
    cost_before: float
    cost_after: float
    damping: float
    step_norm: float
    gradient_inf_norm: float
    linearize_s: float
    solve_s: float


@dataclass(frozen=True)
class HistorySmootherResult:
    states: tuple[NavigationState, ...]
    initial_cost: float
    final_cost: float
    converged: bool
    convergence_reason: str
    iterations: tuple[HistoryIteration, ...]
    final_gradient_inf_norm: float
    elapsed_s: float


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require(data: np.lib.npyio.NpzFile, name: str) -> np.ndarray:
    if name not in data.files:
        raise KeyError(f"PACE-VIO tensor archive lacks {name!r}")
    return np.asarray(data[name])


def _state_from_row(
    camera_pose: np.ndarray,
    extrinsic_CI: torch.Tensor,
    velocity: np.ndarray,
    acc_bias: np.ndarray,
    gyro_bias: np.ndarray,
) -> NavigationState:
    camera = pp.SE3(torch.as_tensor(camera_pose, dtype=torch.float64).reshape(1, 7))
    body = camera @ pp.SE3(extrinsic_CI.reshape(1, 7))
    return NavigationState(
        pose_WB=body.tensor().detach(),
        velocity_W=torch.as_tensor(velocity, dtype=torch.float64).reshape(3),
        acc_bias=torch.as_tensor(acc_bias, dtype=torch.float64).reshape(3),
        gyro_bias=torch.as_tensor(gyro_bias, dtype=torch.float64).reshape(3),
    )


def load_t2_history_archive(
    tensor_map_path: str | Path,
    *,
    start_frame: int,
    end_frame: int,
    pose_translation_std: float = 1.0e-5,
    pose_rotation_std: float = 1.0e-5,
    velocity_std: float = 0.05,
    acc_bias_std: float = 0.2,
    gyro_bias_std: float = 0.02,
    normal_eigenvalue_floor: float = 1.0e-10,
    visual_normal_equations_path: str | Path | None = None,
) -> T2HistoryArchive:
    """Load an inclusive frame range from a production ``tensor_map.npz``."""

    path = Path(tensor_map_path).expanduser().resolve()
    with np.load(path, allow_pickle=False) as data:
        poses = _require(data, "frames//pose")
        timestamps = _require(data, "frames//time_ns").astype(np.int64, copy=False)
        velocity = _require(data, "frames//imu_vio_velocity_world")
        acc_bias = _require(data, "frames//imu_vio_acc_bias")
        gyro_bias = _require(data, "frames//imu_vio_gyro_bias")
        extrinsics = _require(data, "frames//imu_vio_sensor_T_imu")

        count = int(poses.shape[0])
        if not (0 <= start_frame < end_frame < count):
            raise ValueError(
                f"invalid inclusive frame range [{start_frame}, {end_frame}] for {count} frames"
            )
        if not np.isfinite(extrinsics[start_frame : end_frame + 1]).all():
            raise FloatingPointError("camera-to-IMU extrinsic contains NaN/Inf")
        reference_extrinsic = np.asarray(extrinsics[start_frame], dtype=np.float64).reshape(7)
        if not np.allclose(
            extrinsics[start_frame : end_frame + 1],
            reference_extrinsic.reshape(1, 7),
            atol=1.0e-8,
            rtol=0.0,
        ):
            raise ValueError("camera-to-IMU extrinsic changes inside the smoothing range")
        extrinsic_CI = torch.as_tensor(reference_extrinsic, dtype=torch.float64).reshape(1, 7)

        states = tuple(
            _state_from_row(
                poses[row], extrinsic_CI, velocity[row], acc_bias[row], gyro_bias[row]
            )
            for row in range(start_frame, end_frame + 1)
        )

        delta_rot = _require(data, "frames//imu_vio_delta_rotvec")
        delta_v = _require(data, "frames//imu_vio_delta_v")
        delta_p = _require(data, "frames//imu_vio_delta_p")
        imu_cov = _require(data, "frames//imu_vio_cov")
        dt = _require(data, "frames//imu_vio_dt")
        bias_jacobian = _require(data, "frames//imu_vio_bias_jacobian")
        linearized_ba = _require(data, "frames//imu_vio_linearized_acc_bias")
        linearized_bg = _require(data, "frames//imu_vio_linearized_gyro_bias")
        bias_rw_cov = _require(data, "frames//imu_vio_bias_rw_cov")
        gravity = _require(data, "frames//imu_vio_gravity_world")
        gravity_in_residual = _require(data, "frames//imu_vio_gravity_in_residual")
        if visual_normal_equations_path is None:
            visual_reference = _require(data, "frames//visual_compressed_uvd_reference_CjCi")
            visual_hessian = _require(data, "frames//visual_compressed_uvd_hessian")
            visual_gradient = _require(data, "frames//visual_compressed_uvd_gradient")
        else:
            visual_path = Path(visual_normal_equations_path).expanduser().resolve()
            with np.load(visual_path, allow_pickle=False) as visual_data:
                visual_timestamps = _require(visual_data, "timestamps_ns").astype(
                    np.int64, copy=False
                )
                if visual_timestamps.shape[0] <= end_frame or not np.array_equal(
                    visual_timestamps[start_frame : end_frame + 1],
                    timestamps[start_frame : end_frame + 1],
                ):
                    raise ValueError("external UVD normal equations do not match frame timestamps")
                visual_reference = _require(visual_data, "reference_CjCi").copy()
                visual_hessian = _require(visual_data, "hessian").copy()
                visual_gradient = _require(visual_data, "gradient").copy()

        edges: list[ArchivedT2Edge] = []
        for frame_j in range(start_frame + 1, end_frame + 1):
            frame_i = frame_j - 1
            edge_dt = float(dt[frame_j])
            if not np.isfinite(edge_dt) or edge_dt <= 0.0:
                raise ValueError(f"edge {frame_i}->{frame_j} has invalid dt={edge_dt}")
            imu = ImuPreintegrationFactor(
                delta_rotation=torch.as_tensor(delta_rot[frame_j], dtype=torch.float64),
                delta_velocity=torch.as_tensor(delta_v[frame_j], dtype=torch.float64),
                delta_position=torch.as_tensor(delta_p[frame_j], dtype=torch.float64),
                covariance=torch.as_tensor(imu_cov[frame_j], dtype=torch.float64),
                dt=edge_dt,
                bias_jacobian=torch.as_tensor(bias_jacobian[frame_j], dtype=torch.float64),
                linearized_acc_bias=torch.as_tensor(linearized_ba[frame_j], dtype=torch.float64),
                linearized_gyro_bias=torch.as_tensor(linearized_bg[frame_j], dtype=torch.float64),
                bias_rw_covariance=torch.as_tensor(bias_rw_cov[frame_j], dtype=torch.float64),
                gravity_world=(
                    torch.as_tensor(gravity[frame_j], dtype=torch.float64)
                    if bool(gravity_in_residual[frame_j])
                    else None
                ),
                gravity_handling=(
                    "residual" if bool(gravity_in_residual[frame_j]) else "preintegration"
                ),
            )
            cached_hessian = torch.as_tensor(
                visual_hessian[frame_j], dtype=torch.float64
            ).reshape(6, 6)
            cached_gradient = torch.as_tensor(
                visual_gradient[frame_j], dtype=torch.float64
            ).reshape(6)
            visual = linearized_uvd_pose_factor_from_normal_equations(
                torch.as_tensor(visual_reference[frame_j], dtype=torch.float64).reshape(1, 7),
                cached_hessian,
                cached_gradient,
                extrinsic_CI,
                normal_eigenvalue_floor=normal_eigenvalue_floor,
            )
            edges.append(
                ArchivedT2Edge(
                    frame_i=frame_i,
                    frame_j=frame_j,
                    imu=imu,
                    visual=visual,
                    cached_hessian=cached_hessian,
                    cached_gradient=cached_gradient,
                )
            )

    initial_prior = make_diagonal_prior(
        states[0],
        pose_translation_std=pose_translation_std,
        pose_rotation_std=pose_rotation_std,
        velocity_std=velocity_std,
        acc_bias_std=acc_bias_std,
        gyro_bias_std=gyro_bias_std,
    )
    return T2HistoryArchive(
        source_path=path,
        source_sha256=_sha256(path),
        frame_indices=np.arange(start_frame, end_frame + 1, dtype=np.int64),
        timestamps_ns=timestamps[start_frame : end_frame + 1].copy(),
        online_states=states,
        initial_prior=initial_prior,
        edges=tuple(edges),
        extrinsic_CI=extrinsic_CI.detach().clone(),
    )


def compressed_factor_equivalence(archive: T2HistoryArchive) -> dict[str, float | int | bool]:
    """Check that cached H/g and reconstructed square-root factors agree."""

    max_h_abs = 0.0
    max_h_rel = 0.0
    max_g_abs = 0.0
    max_g_rel = 0.0
    for edge in archive.edges:
        a = edge.visual.sqrt_information
        c = edge.visual.residual_offset
        rebuilt_h = a.mT @ a
        rebuilt_g = a.mT @ c
        h_error = float((rebuilt_h - edge.cached_hessian).abs().max().item())
        g_error = float((rebuilt_g - edge.cached_gradient).abs().max().item())
        h_scale = max(float(edge.cached_hessian.abs().max().item()), 1.0)
        g_scale = max(float(edge.cached_gradient.abs().max().item()), 1.0)
        max_h_abs = max(max_h_abs, h_error)
        max_h_rel = max(max_h_rel, h_error / h_scale)
        max_g_abs = max(max_g_abs, g_error)
        max_g_rel = max(max_g_rel, g_error / g_scale)
    return {
        "edge_count": len(archive.edges),
        "max_hessian_absolute_error": max_h_abs,
        "max_hessian_relative_error": max_h_rel,
        "max_gradient_absolute_error": max_g_abs,
        "max_gradient_relative_error": max_g_rel,
        "has_nan_or_inf": False,
    }


def _imu_residual_and_jacobian(
    state_i: NavigationState,
    state_j: NavigationState,
    imu: ImuPreintegrationFactor,
    covariance_eigenvalue_floor: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    zero = torch.zeros(PAIR_DOF, dtype=torch.float64, requires_grad=True)

    def residual(increment: torch.Tensor) -> torch.Tensor:
        candidate_i = retract_state(state_i, increment[:STATE_DOF])
        candidate_j = retract_state(state_j, increment[STATE_DOF:])
        return _imu_residual(
            candidate_i, candidate_j, imu, covariance_eigenvalue_floor
        )

    value = residual(zero)
    jacobian = torch.autograd.functional.jacobian(
        residual, zero, create_graph=False, vectorize=True
    )
    return value.detach(), jacobian.detach()


def _edge_linearization(
    state_i: NavigationState,
    state_j: NavigationState,
    edge: ArchivedT2Edge,
    covariance_eigenvalue_floor: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    imu_r, imu_j = _imu_residual_and_jacobian(
        state_i, state_j, edge.imu, covariance_eigenvalue_floor
    )
    bias_r, bias_j = _bias_residual_and_analytic_jacobian(
        state_i, state_j, edge.imu, covariance_eigenvalue_floor
    )
    visual_r, visual_j = _linearized_visual_residual_and_analytic_jacobian(
        state_i, state_j, edge.visual
    )
    return torch.cat([imu_r, bias_r, visual_r]), torch.cat(
        [imu_j, bias_j, visual_j], dim=0
    )


def _edge_residual(
    state_i: NavigationState,
    state_j: NavigationState,
    edge: ArchivedT2Edge,
    covariance_eigenvalue_floor: float,
) -> torch.Tensor:
    imu = _imu_residual(state_i, state_j, edge.imu, covariance_eigenvalue_floor)
    bias, _ = _bias_residual_and_analytic_jacobian(
        state_i, state_j, edge.imu, covariance_eigenvalue_floor
    )
    visual = visual_whitened_residuals(
        state_i, state_j, edge.visual, covariance_eigenvalue_floor
    ).reshape(-1)
    return torch.cat([imu, bias, visual])


def _evaluate_cost(
    states: Sequence[NavigationState],
    archive: T2HistoryArchive,
    covariance_eigenvalue_floor: float,
) -> float:
    prior_r, _ = _prior_residual_and_analytic_jacobian(
        states[0], archive.initial_prior
    )
    square_sum = prior_r.square().sum()
    for local_edge, edge in enumerate(archive.edges):
        residual = _edge_residual(
            states[local_edge],
            states[local_edge + 1],
            edge,
            covariance_eigenvalue_floor,
        )
        square_sum = square_sum + residual.square().sum()
    return 0.5 * float(square_sum.detach().cpu().item())


def factor_cost_breakdown(
    states: Sequence[NavigationState],
    archive: T2HistoryArchive,
    *,
    covariance_eigenvalue_floor: float = 1.0e-12,
) -> dict[str, np.ndarray]:
    """Return unmodified quadratic costs for every archived factor.

    The visual entries are the locally compressed UVD quadratic factors.  No
    additional visual robust kernel is applied because the archived ``H`` and
    ``g`` already encode the production UVD linearization and its weights.
    """

    if len(states) != len(archive.online_states):
        raise ValueError(
            f"state count {len(states)} does not match archive count "
            f"{len(archive.online_states)}"
        )
    prior_r, _ = _prior_residual_and_analytic_jacobian(
        states[0], archive.initial_prior
    )
    imu_cost: list[float] = []
    bias_cost: list[float] = []
    visual_cost: list[float] = []
    for local_edge, edge in enumerate(archive.edges):
        state_i = states[local_edge]
        state_j = states[local_edge + 1]
        imu_r = _imu_residual(
            state_i, state_j, edge.imu, covariance_eigenvalue_floor
        )
        bias_r, _ = _bias_residual_and_analytic_jacobian(
            state_i, state_j, edge.imu, covariance_eigenvalue_floor
        )
        visual_r = visual_whitened_residuals(
            state_i, state_j, edge.visual, covariance_eigenvalue_floor
        ).reshape(-1)
        imu_cost.append(0.5 * float(imu_r.square().sum().item()))
        bias_cost.append(0.5 * float(bias_r.square().sum().item()))
        visual_cost.append(0.5 * float(visual_r.square().sum().item()))
    return {
        "prior": np.asarray([0.5 * float(prior_r.square().sum().item())]),
        "imu": np.asarray(imu_cost, dtype=np.float64),
        "bias": np.asarray(bias_cost, dtype=np.float64),
        "visual": np.asarray(visual_cost, dtype=np.float64),
    }


def _sparse_normal_equation(
    states: Sequence[NavigationState],
    archive: T2HistoryArchive,
    covariance_eigenvalue_floor: float,
) -> tuple[scipy.sparse.csr_matrix, np.ndarray, float]:
    count = len(states)
    diagonal = np.zeros((count, STATE_DOF, STATE_DOF), dtype=np.float64)
    upper = np.zeros((count - 1, STATE_DOF, STATE_DOF), dtype=np.float64)
    gradient = np.zeros((count, STATE_DOF), dtype=np.float64)
    square_sum = 0.0

    prior_r, prior_pair_j = _prior_residual_and_analytic_jacobian(
        states[0], archive.initial_prior
    )
    prior_j = prior_pair_j[:, :STATE_DOF]
    diagonal[0] += (prior_j.mT @ prior_j).cpu().numpy()
    gradient[0] += (prior_j.mT @ prior_r).cpu().numpy()
    square_sum += float(prior_r.square().sum().item())

    for local_edge, edge in enumerate(archive.edges):
        residual, jacobian = _edge_linearization(
            states[local_edge],
            states[local_edge + 1],
            edge,
            covariance_eigenvalue_floor,
        )
        hessian = (jacobian.mT @ jacobian).cpu().numpy()
        edge_gradient = (jacobian.mT @ residual).cpu().numpy()
        diagonal[local_edge] += hessian[:STATE_DOF, :STATE_DOF]
        diagonal[local_edge + 1] += hessian[STATE_DOF:, STATE_DOF:]
        upper[local_edge] += hessian[:STATE_DOF, STATE_DOF:]
        gradient[local_edge] += edge_gradient[:STATE_DOF]
        gradient[local_edge + 1] += edge_gradient[STATE_DOF:]
        square_sum += float(residual.square().sum().item())

    row_parts: list[np.ndarray] = []
    col_parts: list[np.ndarray] = []
    value_parts: list[np.ndarray] = []

    def add_block(block_row: int, block_col: int, block: np.ndarray) -> None:
        base_row = block_row * STATE_DOF
        base_col = block_col * STATE_DOF
        rr, cc = np.indices((STATE_DOF, STATE_DOF))
        row_parts.append((rr + base_row).reshape(-1))
        col_parts.append((cc + base_col).reshape(-1))
        value_parts.append(block.reshape(-1))

    for index in range(count):
        add_block(index, index, diagonal[index])
    for index in range(count - 1):
        add_block(index, index + 1, upper[index])
        add_block(index + 1, index, upper[index].T)

    dimension = count * STATE_DOF
    matrix = scipy.sparse.coo_matrix(
        (
            np.concatenate(value_parts),
            (np.concatenate(row_parts), np.concatenate(col_parts)),
        ),
        shape=(dimension, dimension),
    ).tocsr()
    return matrix, gradient.reshape(-1), 0.5 * square_sum


def smooth_t2_history(
    archive: T2HistoryArchive,
    *,
    max_iterations: int = 12,
    initial_damping: float = 1.0e-3,
    step_tolerance: float = 1.0e-7,
    cost_tolerance: float = 1.0e-8,
    covariance_eigenvalue_floor: float = 1.0e-12,
) -> HistorySmootherResult:
    """Optimize every archived 15D state without modifying the online archive."""

    started = time.perf_counter()
    states = tuple(state.detach() for state in archive.online_states)
    initial_cost = _evaluate_cost(states, archive, covariance_eigenvalue_floor)
    current_cost = initial_cost
    damping = max(float(initial_damping), 1.0e-12)
    history: list[HistoryIteration] = []
    converged = False
    reason = "iteration_limit"
    final_gradient_inf = float("inf")

    for iteration in range(max(1, int(max_iterations))):
        linearize_started = time.perf_counter()
        hessian, gradient, linearized_cost = _sparse_normal_equation(
            states, archive, covariance_eigenvalue_floor
        )
        linearize_s = time.perf_counter() - linearize_started
        if not np.isfinite(gradient).all() or not np.isfinite(hessian.data).all():
            raise FloatingPointError("history smoother normal equation contains NaN/Inf")
        final_gradient_inf = float(np.max(np.abs(gradient)))
        diagonal = np.maximum(np.abs(hessian.diagonal()), 1.0)
        system = hessian + scipy.sparse.diags(damping * diagonal)
        solve_started = time.perf_counter()
        step = scipy.sparse.linalg.spsolve(system.tocsc(), -gradient)
        solve_s = time.perf_counter() - solve_started
        if not np.isfinite(step).all():
            raise FloatingPointError("history smoother produced a non-finite step")
        step_norm = float(np.linalg.norm(step))
        if step_norm <= step_tolerance:
            history.append(
                HistoryIteration(
                    iteration=iteration + 1,
                    accepted=False,
                    cost_before=current_cost,
                    cost_after=current_cost,
                    damping=damping,
                    step_norm=step_norm,
                    gradient_inf_norm=final_gradient_inf,
                    linearize_s=linearize_s,
                    solve_s=solve_s,
                )
            )
            converged = True
            reason = "step_tolerance"
            break

        candidate = tuple(
            retract_state(
                state,
                torch.as_tensor(
                    step[index * STATE_DOF : (index + 1) * STATE_DOF],
                    dtype=torch.float64,
                ),
            ).detach()
            for index, state in enumerate(states)
        )
        candidate_cost = _evaluate_cost(
            candidate, archive, covariance_eigenvalue_floor
        )
        accepted = candidate_cost < current_cost
        history.append(
            HistoryIteration(
                iteration=iteration + 1,
                accepted=accepted,
                cost_before=current_cost,
                cost_after=candidate_cost,
                damping=damping,
                step_norm=step_norm,
                gradient_inf_norm=final_gradient_inf,
                linearize_s=linearize_s,
                solve_s=solve_s,
            )
        )
        if accepted:
            previous = current_cost
            states = candidate
            current_cost = candidate_cost
            damping = max(damping * 0.25, 1.0e-12)
            if abs(previous - current_cost) <= cost_tolerance:
                converged = True
                reason = "cost_tolerance"
                break
        else:
            damping = min(damping * 10.0, 1.0e12)

    return HistorySmootherResult(
        states=states,
        initial_cost=initial_cost,
        final_cost=current_cost,
        converged=converged,
        convergence_reason=reason,
        iterations=tuple(history),
        final_gradient_inf_norm=final_gradient_inf,
        elapsed_s=time.perf_counter() - started,
    )


def state_arrays(states: Sequence[NavigationState]) -> dict[str, np.ndarray]:
    return {
        "pose_WB_internal_ned": np.stack(
            [state.pose_WB.reshape(7).detach().cpu().numpy() for state in states]
        ),
        "velocity_W_internal_ned": np.stack(
            [state.velocity_W.detach().cpu().numpy() for state in states]
        ),
        "acc_bias_internal": np.stack(
            [state.acc_bias.detach().cpu().numpy() for state in states]
        ),
        "gyro_bias_internal": np.stack(
            [state.gyro_bias.detach().cpu().numpy() for state in states]
        ),
    }


def write_summary(path: str | Path, payload: dict) -> None:
    Path(path).write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
