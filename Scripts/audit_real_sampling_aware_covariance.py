#!/usr/bin/env python3
"""Audit real-data sampling-aware IMU covariance for the frozen normal-noise slice."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import pypose as pp
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Module.IMUPreintegration import preintegrate_imu_local_frame  # noqa: E402
from Scripts.audit_normal_noise_preintegration_nis import (  # noqa: E402
    stage_for_edge,
)
from Utility.IMUCSV import IMUCSVLoader  # noqa: E402


SCENE = "clear_stop_turn_rectangle_truth_normal_noise"
DATASET_DIR = (
    Path("/mnt/e")
    / "\u6587\u6863/holoocean/code/recordings"
    / "batch_clear_truth_paths_20260713_static63_variants"
    / SCENE
)
RESULT_DIR = (
    ROOT
    / "Results/rectangle_normal_noise_two_state_standard_full_20260715"
    / "trial_1/vio_two_state_fixed_lag_standard_full"
    / SCENE
)
DEFAULT_OUTPUT = ROOT / "analysis_normal_noise_sampling_aware_20260716"
FIRST_VALID_FRAME = 90
MAX_FRAME_EXCLUSIVE = 300
EXPECTED_EDGES = 209
IMU_RATE_HZ = 100.0
SIGMA_A = 0.0141258
SIGMA_G = 0.00182898
SIGMA_AW = 0.000386071
SIGMA_GW = 3.57864e-05
RAW_STD_A = SIGMA_A * math.sqrt(IMU_RATE_HZ)
RAW_STD_G = SIGMA_G * math.sqrt(IMU_RATE_HZ)
FLU_TO_NED = np.diag([1.0, -1.0, -1.0])
COMPONENTS = ("p_x", "p_y", "p_z", "v_x", "v_y", "v_z", "R_x", "R_y", "R_z")


@dataclass(frozen=True)
class EdgeSamplingMap:
    edge_id: int
    frame_i: int
    frame_j: int
    start_ns: int
    end_ns: int
    raw_indices: np.ndarray
    raw_times_ns: np.ndarray
    knot_times_ns: np.ndarray
    knot_from_raw: np.ndarray
    midpoint_from_knot: np.ndarray
    temporal_map: np.ndarray
    full_map: np.ndarray

    @property
    def step_count(self) -> int:
        return int(self.temporal_map.shape[0])

    @property
    def raw_count(self) -> int:
        return int(self.temporal_map.shape[1])

    @property
    def dt_s(self) -> np.ndarray:
        return np.diff(self.knot_times_ns).astype(np.float64) * 1e-9


@dataclass(frozen=True)
class EdgeCovariance:
    sampling_map: EdgeSamplingMap
    motion_stage: str
    phase_ns: int
    nominal_acc_mid: np.ndarray
    nominal_gyro_mid: np.ndarray
    jacobian_processed: np.ndarray
    jacobian_raw: np.ndarray
    p_current: np.ndarray
    p_current_measurement: np.ndarray
    p_bias_process: np.ndarray
    p_sampling_measurement: np.ndarray
    p_sampling_total: np.ndarray
    fd_max_abs: float
    fd_max_relative: float
    production_delta_difference: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DATASET_DIR)
    parser.add_argument("--result-dir", type=Path, default=RESULT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--mc-realizations", type=int, default=5000)
    parser.add_argument("--mc-chunk-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260716)
    return parser.parse_args()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def stats(values: Iterable[float]) -> dict[str, float | int]:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    if not array.size:
        return {"count": 0, "min": math.nan, "median": math.nan,
                "mean": math.nan, "p95": math.nan, "max": math.nan}
    return {
        "count": int(array.size), "min": float(array.min()),
        "median": float(np.median(array)), "mean": float(array.mean()),
        "p95": float(np.quantile(array, 0.95)), "max": float(array.max()),
    }


def axis_values(frame: pd.DataFrame, names: list[str]) -> np.ndarray:
    return frame[names].to_numpy(np.float64)


def interpolation_support(times_ns: np.ndarray, target_ns: int) -> list[tuple[int, float]]:
    index = int(np.searchsorted(times_ns, target_ns, side="left"))
    if index < len(times_ns) and int(times_ns[index]) == int(target_ns):
        return [(index, 1.0)]
    if index <= 0:
        return [(0, 1.0)]
    if index >= len(times_ns):
        return [(len(times_ns) - 1, 1.0)]
    left = index - 1
    right = index
    alpha = (int(target_ns) - int(times_ns[left])) / float(
        int(times_ns[right]) - int(times_ns[left])
    )
    return [(left, 1.0 - alpha), (right, alpha)]


def build_sampling_map(
    times_ns: np.ndarray,
    *,
    edge_id: int,
    frame_i: int,
    frame_j: int,
    start_ns: int,
    end_ns: int,
) -> EdgeSamplingMap:
    interior_left = int(np.searchsorted(times_ns, start_ns, side="right"))
    interior_right = int(np.searchsorted(times_ns, end_ns, side="left"))
    interior = np.arange(interior_left, interior_right, dtype=np.int64)
    supports: list[list[tuple[int, float]]] = [interpolation_support(times_ns, start_ns)]
    supports.extend([[(int(index), 1.0)] for index in interior])
    supports.append(interpolation_support(times_ns, end_ns))
    raw_indices = np.asarray(
        sorted({index for support in supports for index, _ in support}), dtype=np.int64
    )
    local_column = {int(index): column for column, index in enumerate(raw_indices)}
    knot_from_raw = np.zeros((len(supports), len(raw_indices)), dtype=np.float64)
    for row, support in enumerate(supports):
        for global_index, weight in support:
            knot_from_raw[row, local_column[int(global_index)]] += float(weight)
    knot_times = np.concatenate(
        [np.asarray([start_ns], np.int64), times_ns[interior], np.asarray([end_ns], np.int64)]
    )
    midpoint_from_knot = np.zeros((len(knot_times) - 1, len(knot_times)), dtype=np.float64)
    for step in range(len(knot_times) - 1):
        midpoint_from_knot[step, step] = 0.5
        midpoint_from_knot[step, step + 1] = 0.5
    temporal_map = midpoint_from_knot @ knot_from_raw

    full_map = np.zeros((temporal_map.shape[0] * 6, temporal_map.shape[1] * 6), dtype=np.float64)
    sensor_rotation = np.zeros((6, 6), dtype=np.float64)
    sensor_rotation[0:3, 0:3] = FLU_TO_NED
    sensor_rotation[3:6, 3:6] = FLU_TO_NED
    for step in range(temporal_map.shape[0]):
        for sample in range(temporal_map.shape[1]):
            full_map[step * 6:(step + 1) * 6, sample * 6:(sample + 1) * 6] = (
                temporal_map[step, sample] * sensor_rotation
            )
    return EdgeSamplingMap(
        edge_id=edge_id, frame_i=frame_i, frame_j=frame_j,
        start_ns=start_ns, end_ns=end_ns,
        raw_indices=raw_indices, raw_times_ns=times_ns[raw_indices],
        knot_times_ns=knot_times, knot_from_raw=knot_from_raw,
        midpoint_from_knot=midpoint_from_knot,
        temporal_map=temporal_map, full_map=full_map,
    )


def query_production_impulse(
    times_ns: np.ndarray,
    sampling_map: EdgeSamplingMap,
    raw_global_index: int,
    channel: int,
) -> np.ndarray:
    loader = IMUCSVLoader.__new__(IMUCSVLoader)
    loader.time_ns = torch.from_numpy(times_ns.copy()).long()
    loader.acc = torch.zeros((len(times_ns), 3), dtype=torch.float32)
    loader.gyro = torch.zeros((len(times_ns), 3), dtype=torch.float32)
    if channel < 3:
        loader.acc[raw_global_index, channel] = 1.0
    else:
        loader.gyro[raw_global_index, channel - 3] = 1.0
    _, acc, gyro = loader.query_range(sampling_map.start_ns, sampling_map.end_ns)
    acc_internal = acc.double().numpy() @ FLU_TO_NED.T
    gyro_internal = gyro.double().numpy() @ FLU_TO_NED.T
    processed = np.concatenate(
        [0.5 * (acc_internal[:-1] + acc_internal[1:]),
         0.5 * (gyro_internal[:-1] + gyro_internal[1:])],
        axis=1,
    )
    return processed.reshape(-1)


def skew(vector: torch.Tensor) -> torch.Tensor:
    x, y, z = vector.unbind(dim=-1)
    zero = torch.zeros_like(x)
    return torch.stack(
        [zero, -z, y, z, zero, -x, -y, x, zero], dim=-1
    ).reshape(*vector.shape[:-1], 3, 3)


def so3_exp(rotvec: torch.Tensor) -> torch.Tensor:
    theta = torch.linalg.vector_norm(rotvec, dim=-1, keepdim=True)
    matrix = skew(rotvec)
    a = torch.sinc(theta / math.pi)[..., None]
    b = (0.5 * torch.sinc(theta / (2.0 * math.pi)).square())[..., None]
    identity = torch.eye(3, dtype=rotvec.dtype, device=rotvec.device)
    return identity + a * matrix + b * (matrix @ matrix)


def so3_log(rotation: torch.Tensor) -> torch.Tensor:
    vee = 0.5 * torch.stack(
        [rotation[..., 2, 1] - rotation[..., 1, 2],
         rotation[..., 0, 2] - rotation[..., 2, 0],
         rotation[..., 1, 0] - rotation[..., 0, 1]],
        dim=-1,
    )
    sine = torch.linalg.vector_norm(vee, dim=-1, keepdim=True)
    cosine = ((torch.diagonal(rotation, dim1=-2, dim2=-1).sum(dim=-1, keepdim=True) - 1.0) * 0.5).clamp(-1.0, 1.0)
    theta = torch.atan2(sine, cosine)
    raw_factor = theta / sine.clamp_min(1e-12)
    series_factor = 1.0 + sine.square() / 6.0
    factor = torch.where(sine > 1e-7, raw_factor, series_factor)
    return factor * vee


def integrate_processed(
    acc_mid: torch.Tensor,
    gyro_mid: torch.Tensor,
    dt_s: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if acc_mid.ndim == 2:
        acc_mid = acc_mid.unsqueeze(0)
        gyro_mid = gyro_mid.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False
    batch = acc_mid.shape[0]
    rotation = torch.eye(3, dtype=acc_mid.dtype, device=acc_mid.device).expand(batch, 3, 3).clone()
    velocity = torch.zeros((batch, 3), dtype=acc_mid.dtype, device=acc_mid.device)
    position = torch.zeros((batch, 3), dtype=acc_mid.dtype, device=acc_mid.device)
    for step in range(acc_mid.shape[1]):
        dt = dt_s[step]
        acceleration = (rotation @ acc_mid[:, step].unsqueeze(-1)).squeeze(-1)
        position = position + velocity * dt + 0.5 * acceleration * dt * dt
        velocity = velocity + acceleration * dt
        rotation = rotation @ so3_exp(gyro_mid[:, step] * dt)
    if squeeze:
        return position[0], velocity[0], rotation[0]
    return position, velocity, rotation


def residual_from_processed_noise(
    noise_flat: torch.Tensor,
    nominal_acc_mid: torch.Tensor,
    nominal_gyro_mid: torch.Tensor,
    dt_s: torch.Tensor,
) -> torch.Tensor:
    steps = nominal_acc_mid.shape[0]
    noise = noise_flat.reshape(steps, 6)
    ref_p, ref_v, ref_r = integrate_processed(nominal_acc_mid, nominal_gyro_mid, dt_s)
    noisy_p, noisy_v, noisy_r = integrate_processed(
        nominal_acc_mid + noise[:, 0:3], nominal_gyro_mid + noise[:, 3:6], dt_s
    )
    return torch.cat([noisy_p - ref_p, noisy_v - ref_v, so3_log(ref_r.mT @ noisy_r)])


def finite_difference_jacobian(
    function,
    dimension: int,
    epsilon: float = 1e-5,
) -> np.ndarray:
    jacobian = np.zeros((9, dimension), dtype=np.float64)
    for column in range(dimension):
        plus = torch.zeros(dimension, dtype=torch.float64)
        minus = torch.zeros(dimension, dtype=torch.float64)
        plus[column] = epsilon
        minus[column] = -epsilon
        jacobian[:, column] = ((function(plus) - function(minus)) / (2.0 * epsilon)).detach().numpy()
    return jacobian


def raw_q_diagonal(raw_count: int) -> np.ndarray:
    per_sample = np.asarray([RAW_STD_A**2] * 3 + [RAW_STD_G**2] * 3, dtype=np.float64)
    return np.tile(per_sample, raw_count)


def covariance_health(covariance: np.ndarray) -> dict[str, Any]:
    covariance = 0.5 * (covariance + covariance.T)
    eigenvalues = np.linalg.eigvalsh(covariance)
    return {
        "trace": float(np.trace(covariance)),
        "eigenvalues": eigenvalues.tolist(),
        "min_eigenvalue": float(eigenvalues.min()),
        "max_eigenvalue": float(eigenvalues.max()),
        "condition_number": float(eigenvalues.max() / eigenvalues.min()),
    }


def rotation_difference(reference: pp.LieTensor, candidate_matrix: np.ndarray) -> np.ndarray:
    reference_matrix = reference.double().matrix().reshape(3, 3)
    candidate = torch.from_numpy(candidate_matrix).double().reshape(3, 3)
    return so3_log(reference_matrix.mT @ candidate).reshape(3).numpy()


def build_edge_covariance(
    sampling_map: EdgeSamplingMap,
    *,
    raw_reference_acc_flu: np.ndarray,
    raw_reference_gyro_flu: np.ndarray,
    linearized_acc_bias: np.ndarray,
    linearized_gyro_bias: np.ndarray,
    p_current_stored: np.ndarray,
    motion_stage: str,
) -> EdgeCovariance:
    raw_values = np.concatenate(
        [raw_reference_acc_flu[sampling_map.raw_indices],
         raw_reference_gyro_flu[sampling_map.raw_indices]], axis=1
    ).reshape(-1)
    processed = (sampling_map.full_map @ raw_values).reshape(sampling_map.step_count, 6)
    nominal_acc_mid = processed[:, 0:3] - linearized_acc_bias.reshape(1, 3)
    nominal_gyro_mid = processed[:, 3:6] - linearized_gyro_bias.reshape(1, 3)
    dt_s = torch.from_numpy(sampling_map.dt_s).double()
    nominal_acc_t = torch.from_numpy(nominal_acc_mid).double()
    nominal_gyro_t = torch.from_numpy(nominal_gyro_mid).double()

    def function(noise: torch.Tensor) -> torch.Tensor:
        return residual_from_processed_noise(noise, nominal_acc_t, nominal_gyro_t, dt_s)

    zero = torch.zeros(sampling_map.step_count * 6, dtype=torch.float64, requires_grad=True)
    jacobian_processed = torch.autograd.functional.jacobian(
        function, zero, create_graph=False, strict=True, vectorize=False
    ).detach().numpy()
    jacobian_fd = finite_difference_jacobian(function, zero.numel())
    difference = np.abs(jacobian_processed - jacobian_fd)
    nonzero = np.abs(jacobian_fd) > 1e-7
    relative = difference[nonzero] / np.abs(jacobian_fd[nonzero]) if np.any(nonzero) else np.asarray([0.0])
    jacobian_raw = jacobian_processed @ sampling_map.full_map
    q_diagonal = raw_q_diagonal(sampling_map.raw_count)
    p_sampling_measurement = (jacobian_raw * q_diagonal.reshape(1, -1)) @ jacobian_raw.T
    q_processed = (sampling_map.full_map * q_diagonal.reshape(1, -1)) @ sampling_map.full_map.T
    p_sampling_equivalent = jacobian_processed @ q_processed @ jacobian_processed.T
    if not np.allclose(p_sampling_measurement, p_sampling_equivalent, rtol=1e-10, atol=1e-15):
        raise AssertionError("J_raw Q_raw J_raw^T and J_processed Q_processed J_processed^T differ")

    knot_values = sampling_map.knot_from_raw @ np.concatenate(
        [raw_reference_acc_flu[sampling_map.raw_indices],
         raw_reference_gyro_flu[sampling_map.raw_indices]], axis=1
    )
    production_full = preintegrate_imu_local_frame(
        time_ns=torch.from_numpy(sampling_map.knot_times_ns.copy()).long(),
        acc=torch.from_numpy((knot_values[:, 0:3] @ FLU_TO_NED.T).copy()).float(),
        gyro=torch.from_numpy((knot_values[:, 3:6] @ FLU_TO_NED.T).copy()).float(),
        sigma_acc=SIGMA_A, sigma_gyro=SIGMA_G,
        sigma_acc_w=SIGMA_AW, sigma_gyro_w=SIGMA_GW,
        acc_bias=torch.from_numpy(linearized_acc_bias.copy()).float(),
        gyro_bias=torch.from_numpy(linearized_gyro_bias.copy()).float(),
    )
    production_measurement = preintegrate_imu_local_frame(
        time_ns=torch.from_numpy(sampling_map.knot_times_ns.copy()).long(),
        acc=torch.from_numpy((knot_values[:, 0:3] @ FLU_TO_NED.T).copy()).float(),
        gyro=torch.from_numpy((knot_values[:, 3:6] @ FLU_TO_NED.T).copy()).float(),
        sigma_acc=SIGMA_A, sigma_gyro=SIGMA_G,
        sigma_acc_w=0.0, sigma_gyro_w=0.0,
        acc_bias=torch.from_numpy(linearized_acc_bias.copy()).float(),
        gyro_bias=torch.from_numpy(linearized_gyro_bias.copy()).float(),
    )
    p_current_recomputed = production_full.cov.double().numpy()
    p_current_measurement = production_measurement.cov.double().numpy()
    p_bias_process = 0.5 * (
        (p_current_recomputed - p_current_measurement)
        + (p_current_recomputed - p_current_measurement).T
    )
    p_sampling_total = 0.5 * (
        p_sampling_measurement + p_sampling_measurement.T
    ) + p_bias_process

    diagnostic_p, diagnostic_v, diagnostic_r = integrate_processed(nominal_acc_t, nominal_gyro_t, dt_s)
    production_delta_difference = np.concatenate(
        [
            diagnostic_p.detach().numpy() - production_full.delta_p.double().numpy(),
            diagnostic_v.detach().numpy() - production_full.delta_v.double().numpy(),
            rotation_difference(production_full.delta_R, diagnostic_r.detach().numpy()),
        ]
    )
    return EdgeCovariance(
        sampling_map=sampling_map,
        motion_stage=motion_stage,
        phase_ns=normalize_phase_ns(sampling_map.start_ns % 10_000_000),
        nominal_acc_mid=nominal_acc_mid,
        nominal_gyro_mid=nominal_gyro_mid,
        jacobian_processed=jacobian_processed,
        jacobian_raw=jacobian_raw,
        p_current=p_current_stored,
        p_current_measurement=p_current_measurement,
        p_bias_process=p_bias_process,
        p_sampling_measurement=p_sampling_measurement,
        p_sampling_total=p_sampling_total,
        fd_max_abs=float(difference.max()),
        fd_max_relative=float(relative.max()),
        production_delta_difference=production_delta_difference,
    )


def normalize_phase_ns(phase_ns: int) -> int:
    return 0 if phase_ns >= 9_000_000 else int(phase_ns)


def pack_ragged(maps: list[EdgeSamplingMap], output_path: Path) -> None:
    payload: dict[str, np.ndarray] = {
        "edge_id": np.asarray([item.edge_id for item in maps], np.int64),
        "frame_i": np.asarray([item.frame_i for item in maps], np.int64),
        "frame_j": np.asarray([item.frame_j for item in maps], np.int64),
        "start_ns": np.asarray([item.start_ns for item in maps], np.int64),
        "end_ns": np.asarray([item.end_ns for item in maps], np.int64),
        "a_rows": np.asarray([item.full_map.shape[0] for item in maps], np.int64),
        "a_cols": np.asarray([item.full_map.shape[1] for item in maps], np.int64),
        "temporal_rows": np.asarray([item.temporal_map.shape[0] for item in maps], np.int64),
        "temporal_cols": np.asarray([item.temporal_map.shape[1] for item in maps], np.int64),
    }
    for name, getter in (
        ("a_values", lambda item: item.full_map.reshape(-1)),
        ("temporal_values", lambda item: item.temporal_map.reshape(-1)),
        ("raw_indices", lambda item: item.raw_indices),
        ("raw_times_ns", lambda item: item.raw_times_ns),
        ("knot_times_ns", lambda item: item.knot_times_ns),
        ("q_processed_acc_values", lambda item: processed_covariance(item, RAW_STD_A, 0).reshape(-1)),
        ("q_processed_gyro_values", lambda item: processed_covariance(item, RAW_STD_G, 3).reshape(-1)),
        ("q_processed_values", lambda item: full_processed_covariance(item).reshape(-1)),
    ):
        arrays = [np.asarray(getter(item)) for item in maps]
        offsets = np.cumsum([0] + [array.size for array in arrays], dtype=np.int64)
        payload[name] = np.concatenate(arrays) if arrays else np.asarray([], np.float64)
        payload[f"{name}_offsets"] = offsets
    np.savez_compressed(output_path, **payload)


def processed_covariance(sampling_map: EdgeSamplingMap, raw_std: float, channel_offset: int) -> np.ndarray:
    rows = np.arange(sampling_map.step_count) * 6 + channel_offset
    columns = np.concatenate(
        [np.arange(sampling_map.raw_count) * 6 + channel_offset + axis for axis in range(3)]
    )
    # Return the full 3S covariance for one sensor, preserving axes and cross-step terms.
    selected_rows = np.concatenate([rows + axis for axis in range(3)])
    a = sampling_map.full_map[selected_rows][:, columns]
    return (a * (raw_std**2)) @ a.T


def full_processed_covariance(sampling_map: EdgeSamplingMap) -> np.ndarray:
    q_diagonal = raw_q_diagonal(sampling_map.raw_count)
    return (sampling_map.full_map * q_diagonal.reshape(1, -1)) @ sampling_map.full_map.T


def flatten_matrix(row: dict[str, Any], prefix: str, matrix: np.ndarray) -> None:
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            row[f"{prefix}_{i}{j}"] = float(matrix[i, j])


def covariance_row(edge: EdgeCovariance) -> dict[str, Any]:
    current = edge.p_current
    sampling = edge.p_sampling_total
    difference = sampling - current
    row: dict[str, Any] = {
        "edge_id": edge.sampling_map.edge_id,
        "frame_i": edge.sampling_map.frame_i,
        "frame_j": edge.sampling_map.frame_j,
        "start_ns": edge.sampling_map.start_ns,
        "end_ns": edge.sampling_map.end_ns,
        "phase_ns": edge.phase_ns,
        "motion_stage": edge.motion_stage,
        "raw_sample_count": edge.sampling_map.raw_count,
        "processed_step_count": edge.sampling_map.step_count,
        "frobenius_relative_error": float(np.linalg.norm(difference) / max(np.linalg.norm(current), 1e-18)),
        "trace_ratio_sampling_over_current": float(np.trace(sampling) / np.trace(current)),
        "fd_max_abs": edge.fd_max_abs,
        "fd_max_relative": edge.fd_max_relative,
        "production_delta_difference_norm": float(np.linalg.norm(edge.production_delta_difference)),
    }
    for name, block in (("p", slice(0, 3)), ("v", slice(3, 6)), ("R", slice(6, 9))):
        row[f"{name}_block_relative_error"] = float(
            np.linalg.norm(difference[block, block]) / max(np.linalg.norm(current[block, block]), 1e-18)
        )
    for left_name, left in (("p", slice(0, 3)), ("v", slice(3, 6)), ("R", slice(6, 9))):
        for right_name, right in (("p", slice(0, 3)), ("v", slice(3, 6)), ("R", slice(6, 9))):
            row[f"cross_{left_name}{right_name}_difference_frobenius"] = float(
                np.linalg.norm(difference[left, right])
            )
    for name, matrix in (
        ("P_current", current),
        ("P_current_measurement", edge.p_current_measurement),
        ("P_bias_process", edge.p_bias_process),
        ("P_sampling_measurement", edge.p_sampling_measurement),
        ("P_sampling", sampling),
    ):
        flatten_matrix(row, name, matrix)
        health = covariance_health(matrix)
        row[f"{name}_trace"] = health["trace"]
        row[f"{name}_min_eigenvalue"] = health["min_eigenvalue"]
        row[f"{name}_max_eigenvalue"] = health["max_eigenvalue"]
        row[f"{name}_condition_number"] = health["condition_number"]
        for index, value in enumerate(health["eigenvalues"]):
            row[f"{name}_eigenvalue_{index}"] = value
    return row


def stable_cholesky(covariance: np.ndarray) -> np.ndarray:
    covariance = 0.5 * (covariance + covariance.T)
    return np.linalg.cholesky(covariance)


def monte_carlo_case(
    edge: EdgeCovariance,
    *,
    realizations: int,
    chunk_size: int,
    rng: np.random.Generator,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    errors: list[np.ndarray] = []
    nominal_acc = torch.from_numpy(edge.nominal_acc_mid).double()
    nominal_gyro = torch.from_numpy(edge.nominal_gyro_mid).double()
    dt_s = torch.from_numpy(edge.sampling_map.dt_s).double()
    ref_p, ref_v, ref_r = integrate_processed(nominal_acc, nominal_gyro, dt_s)
    raw_std = np.asarray([RAW_STD_A] * 3 + [RAW_STD_G] * 3, np.float64)
    for start in range(0, realizations, chunk_size):
        count = min(chunk_size, realizations - start)
        raw_noise = rng.normal(size=(count, edge.sampling_map.raw_count, 6)) * raw_std
        processed = np.einsum("sm,bmc->bsc", edge.sampling_map.temporal_map, raw_noise)
        # Temporal interpolation is axis-preserving; apply FLU -> NED signs.
        processed[:, :, 0:3] = processed[:, :, 0:3] @ FLU_TO_NED.T
        processed[:, :, 3:6] = processed[:, :, 3:6] @ FLU_TO_NED.T
        noisy_p, noisy_v, noisy_r = integrate_processed(
            nominal_acc.unsqueeze(0) + torch.from_numpy(processed[:, :, 0:3]).double(),
            nominal_gyro.unsqueeze(0) + torch.from_numpy(processed[:, :, 3:6]).double(),
            dt_s,
        )
        residual = torch.cat(
            [noisy_p - ref_p, noisy_v - ref_v, so3_log(ref_r.mT.unsqueeze(0) @ noisy_r)], dim=1
        )
        errors.append(residual.numpy())
    error = np.concatenate(errors, axis=0)
    p_mc = np.cov(error, rowvar=False, ddof=1)
    p_current = edge.p_current_measurement
    p_sampling = edge.p_sampling_measurement
    current_chol = stable_cholesky(p_current)
    sampling_chol = stable_cholesky(p_sampling)
    w_current = np.linalg.solve(current_chol, error.T).T
    w_sampling = np.linalg.solve(sampling_chol, error.T).T
    nis_current = np.sum(w_current * w_current, axis=1)
    nis_sampling = np.sum(w_sampling * w_sampling, axis=1)

    def block_nis(covariance: np.ndarray, block: slice) -> np.ndarray:
        chol = stable_cholesky(covariance[block, block])
        whitened = np.linalg.solve(chol, error[:, block].T).T
        return np.sum(whitened * whitened, axis=1)

    whitened_covariance = np.cov(w_sampling, rowvar=False, ddof=1)
    whitened_correlation = np.corrcoef(w_sampling, rowvar=False)
    off_diagonal = whitened_correlation - np.eye(9)
    row = {
        "edge_id": edge.sampling_map.edge_id,
        "frame_i": edge.sampling_map.frame_i,
        "frame_j": edge.sampling_map.frame_j,
        "phase_ns": edge.phase_ns,
        "motion_stage": edge.motion_stage,
        "realizations": realizations,
        "P_sampling_vs_MC_relative_frobenius": float(np.linalg.norm(p_sampling - p_mc) / np.linalg.norm(p_mc)),
        "P_current_vs_MC_relative_frobenius": float(np.linalg.norm(p_current - p_mc) / np.linalg.norm(p_mc)),
        "NIS_current_mean": float(nis_current.mean()),
        "NIS_current_median": float(np.median(nis_current)),
        "NIS_current_p95": float(np.quantile(nis_current, 0.95)),
        "NIS_sampling_mean": float(nis_sampling.mean()),
        "NIS_sampling_median": float(np.median(nis_sampling)),
        "NIS_sampling_p95": float(np.quantile(nis_sampling, 0.95)),
        "NIS_sampling_p_mean": float(block_nis(p_sampling, slice(0, 3)).mean()),
        "NIS_sampling_v_mean": float(block_nis(p_sampling, slice(3, 6)).mean()),
        "NIS_sampling_R_mean": float(block_nis(p_sampling, slice(6, 9)).mean()),
        "whitened_mean_max_abs": float(np.abs(w_sampling.mean(axis=0)).max()),
        "whitened_covariance_diag_min": float(np.diag(whitened_covariance).min()),
        "whitened_covariance_diag_max": float(np.diag(whitened_covariance).max()),
        "whitened_max_abs_offdiag_correlation": float(np.abs(off_diagonal).max()),
    }
    for index, name in enumerate(COMPONENTS):
        row[f"whitened_{name}_mean"] = float(w_sampling[:, index].mean())
        row[f"whitened_{name}_std"] = float(w_sampling[:, index].std(ddof=1))
    return row, {
        "error": error, "P_MC": p_mc, "w_sampling": w_sampling,
        "nis_current": nis_current, "nis_sampling": nis_sampling,
    }


def select_mc_edges(
    diagnostics: pd.DataFrame,
    ref_times: np.ndarray,
    ref_velocity: np.ndarray,
    ref_angular_velocity: np.ndarray,
) -> list[pd.Series]:
    candidates = diagnostics.copy()
    candidates["phase_ns"] = candidates["timestamp_i"].astype(np.int64).map(
        lambda value: normalize_phase_ns(int(value) % 10_000_000)
    )
    candidates["motion_stage"] = [
        stage_for_edge(
            start_ns=int(row.timestamp_i), end_ns=int(row.timestamp_j),
            ref_times=ref_times, ref_velocity=ref_velocity,
            ref_angular_velocity=ref_angular_velocity,
        )
        for row in candidates.itertuples(index=False)
    ]
    stages = ["stationary", "accelerating", "constant_velocity", "turning", "decelerating_or_stopping"]
    phases = [0, 3_333_333, 6_666_666]
    selected: list[pd.Series] = []
    for stage in stages:
        stage_rows = candidates[candidates["motion_stage"] == stage]
        if stage_rows.empty:
            raise AssertionError(f"No full-sequence edge is available for motion stage {stage}")
        for phase in phases:
            phase_rows = stage_rows[stage_rows["phase_ns"] == phase]
            if phase_rows.empty:
                continue
            selected.append(phase_rows.iloc[len(phase_rows) // 2])
    selected_stages = {str(row["motion_stage"]) for row in selected}
    selected_phases = {int(row["phase_ns"]) for row in selected}
    if selected_stages != set(stages) or selected_phases != set(phases):
        raise AssertionError(
            f"MC coverage incomplete: stages={selected_stages}, phases={selected_phases}"
        )
    unique: dict[tuple[int, int], pd.Series] = {}
    for row in selected:
        unique[(int(row["frame_i"]), int(row["frame_j"]))] = row
    return list(unique.values())


def markdown_report(summary: dict[str, Any]) -> str:
    gate = summary["acceptance"]
    lines = [
        "# Sampling-aware covariance Monte Carlo 报告",
        "",
        "> 本任务的目标是降低 MACVIO 对 GT 的高频误差，不是让 MACVIO 轨迹或 Bias 接近 GTSAM。",
        "",
        "## 结论",
        "",
        f"- 关卡 3 总判定：**{'通过' if gate['gate3_passed'] else '未通过'}**。",
        f"- 代表边数量：{summary['case_count']}；每边 realization：{summary['realizations_per_case']}。",
        f"- P_sampling/MC Frobenius 中位数：{summary['P_sampling_vs_MC_relative_frobenius']['median']:.6g}。",
        f"- P_sampling/MC Frobenius P95：{summary['P_sampling_vs_MC_relative_frobenius']['p95']:.6g}。",
        f"- sampling-aware NIS9 均值范围：{summary['NIS_sampling_mean']['min']:.6g}–{summary['NIS_sampling_mean']['max']:.6g}。",
        f"- whitened covariance 对角范围：{summary['whitened_covariance_diag_global_min']:.6g}–{summary['whitened_covariance_diag_global_max']:.6g}。",
        f"- 最大绝对非对角相关：{summary['whitened_max_abs_offdiag_correlation']['max']:.6g}。",
        "",
        "## 关卡规则",
        "",
        "只有本报告所有 acceptance 条件均为 true，才允许新增 sampling-aware N=2 实验模式。",
        "未通过时禁止通过调 sigma、视觉权重、LM、窗口长度或输出滤波掩盖问题。",
        "",
        "## Acceptance",
        "",
    ]
    lines.extend([f"- `{name}`: {value}" for name, value in gate.items()])
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    dataset_dir = args.dataset_dir.resolve()
    result_dir = args.result_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    decomposition = pd.read_csv(dataset_dir / "imu_truth_decomposition.csv")
    imu_export = pd.read_csv(dataset_dir / "imu_data.csv")
    reference = pd.read_csv(dataset_dir / "ref_pose.csv")
    diagnostics = pd.read_csv(result_dir / "frame_pair_diagnostics.csv")
    if not np.array_equal(
        decomposition["timestamp"].to_numpy(np.int64), imu_export["timestamp"].to_numpy(np.int64)
    ):
        raise AssertionError("IMU export and truth decomposition timestamps differ")
    times_ns = decomposition["timestamp"].to_numpy(np.int64)
    measured_acc = axis_values(decomposition, ["measured_lin_acc_x", "measured_lin_acc_y", "measured_lin_acc_z"])
    measured_gyro = axis_values(decomposition, ["measured_ang_vel_x", "measured_ang_vel_y", "measured_ang_vel_z"])
    acc_noise = axis_values(decomposition, ["acc_noise_x", "acc_noise_y", "acc_noise_z"])
    gyro_noise = axis_values(decomposition, ["gyro_noise_x", "gyro_noise_y", "gyro_noise_z"])
    reference_acc = measured_acc - acc_noise
    reference_gyro = measured_gyro - gyro_noise

    tensor_map = np.load(result_dir / "tensor_map.npz", allow_pickle=False)
    linearized_ba = tensor_map["frames//imu_vio_linearized_acc_bias"].astype(np.float64)
    linearized_bg = tensor_map["frames//imu_vio_linearized_gyro_bias"].astype(np.float64)
    stored_covariance = tensor_map["frames//imu_vio_cov"].astype(np.float64)
    frame_times = tensor_map["frames//time_ns"].astype(np.int64)

    frozen = diagnostics[
        (diagnostics["frame_i"] >= FIRST_VALID_FRAME)
        & (diagnostics["frame_j"] < MAX_FRAME_EXCLUSIVE)
    ].sort_values(["frame_i", "frame_j"]).reset_index(drop=True)
    if len(frozen) != EXPECTED_EDGES:
        raise AssertionError(f"Expected {EXPECTED_EDGES} frozen edges, found {len(frozen)}")
    maps: list[EdgeSamplingMap] = []
    impulse_rows: list[dict[str, Any]] = []
    for index, row in frozen.iterrows():
        sampling_map = build_sampling_map(
            times_ns, edge_id=int(row["pair_id"]), frame_i=int(row["frame_i"]),
            frame_j=int(row["frame_j"]), start_ns=int(row["timestamp_i"]),
            end_ns=int(row["timestamp_j"]),
        )
        maps.append(sampling_map)
        for local_index, global_index in enumerate(sampling_map.raw_indices):
            for channel in range(6):
                observed = query_production_impulse(times_ns, sampling_map, int(global_index), channel)
                expected = sampling_map.full_map[:, local_index * 6 + channel]
                difference = np.abs(observed - expected)
                impulse_rows.append(
                    {
                        "edge_id": sampling_map.edge_id,
                        "frame_i": sampling_map.frame_i,
                        "frame_j": sampling_map.frame_j,
                        "raw_local_index": local_index,
                        "raw_global_index": int(global_index),
                        "sensor": "acc" if channel < 3 else "gyro",
                        "axis": ("x", "y", "z")[channel % 3],
                        "max_abs_error": float(difference.max()),
                    }
                )
    impulse = pd.DataFrame(impulse_rows)
    impulse.to_csv(output_dir / "sampling_impulse_test.csv", index=False)
    pack_ragged(maps, output_dir / "sampling_map_per_edge.npz")
    map_audit = {
        "gate": 1,
        "edge_count": len(maps),
        "mapping": "y_processed = A_full * n_raw_FLU",
        "A_full_contents": ["endpoint linear interpolation", "interior raw samples", "midpoint averaging", "FLU-to-NED axis rotation"],
        "raw_noise_contract": {
            "independent_samples": True, "rate_hz": IMU_RATE_HZ,
            "acc_per_sample_std": RAW_STD_A, "gyro_per_sample_std": RAW_STD_G,
            "Q_raw": "blockdiag(diag([acc_std^2]*3+[gyro_std^2]*3) per raw sample)",
        },
        "raw_sample_count": stats(item.raw_count for item in maps),
        "processed_step_count": stats(item.step_count for item in maps),
        "phase_distribution": {
            str(phase): int(sum(normalize_phase_ns(item.start_ns % 10_000_000) == phase for item in maps))
            for phase in sorted({normalize_phase_ns(item.start_ns % 10_000_000) for item in maps})
        },
        "impulse_test_count": int(len(impulse)),
        "impulse_max_abs_error": float(impulse["max_abs_error"].max()),
        "impulse_float_dtype": "production IMUCSVLoader float32",
        "impulse_pass_threshold": 2e-7,
        "gate1_passed": bool(float(impulse["max_abs_error"].max()) <= 2e-7),
    }
    write_json(output_dir / "sampling_map_audit.json", map_audit)
    if not map_audit["gate1_passed"]:
        raise AssertionError("Gate 1 failed: production impulse mapping differs from A")

    ref_times = reference["timestamp"].to_numpy(np.int64)
    ref_velocity = axis_values(reference, ["vx", "vy", "vz"])
    ref_angular_velocity = axis_values(reference, ["wx", "wy", "wz"])
    edges: list[EdgeCovariance] = []
    for sampling_map, (_, row) in zip(maps, frozen.iterrows()):
        stage = stage_for_edge(
            start_ns=sampling_map.start_ns, end_ns=sampling_map.end_ns,
            ref_times=ref_times, ref_velocity=ref_velocity,
            ref_angular_velocity=ref_angular_velocity,
        )
        frame_j = sampling_map.frame_j
        edges.append(
            build_edge_covariance(
                sampling_map,
                raw_reference_acc_flu=reference_acc,
                raw_reference_gyro_flu=reference_gyro,
                linearized_acc_bias=linearized_ba[frame_j],
                linearized_gyro_bias=linearized_bg[frame_j],
                p_current_stored=stored_covariance[frame_j],
                motion_stage=stage,
            )
        )
    covariance_frame = pd.DataFrame([covariance_row(edge) for edge in edges])
    covariance_frame.to_csv(output_dir / "sampling_aware_covariance_per_edge.csv", index=False)
    gate2_summary = {
        "gate": 2,
        "edge_count": len(edges),
        "residual_order": ["p", "v", "R"],
        "rotation_error": "Log(Delta_R_reference^-1 * Delta_R_noisy)",
        "P_sampling_measurement": "J_raw * Q_raw * J_raw^T",
        "P_sampling_runtime_total": "P_sampling_measurement + unchanged production within-edge bias-process contribution",
        "frobenius_relative_error_sampling_total_vs_current": stats(covariance_frame["frobenius_relative_error"]),
        "trace_ratio_sampling_over_current": stats(covariance_frame["trace_ratio_sampling_over_current"]),
        "jacobian_fd_max_abs": float(covariance_frame["fd_max_abs"].max()),
        "jacobian_fd_max_relative_nonzero": float(covariance_frame["fd_max_relative"].max()),
        "jacobian_fd_epsilon": 1e-5,
        "jacobian_relative_entry_threshold": 1e-7,
        "production_delta_difference_norm_max": float(covariance_frame["production_delta_difference_norm"].max()),
        "gate2_passed": bool(
            covariance_frame["fd_max_abs"].max() <= 1e-7
            and covariance_frame["fd_max_relative"].max() <= 1e-4
            and covariance_frame["production_delta_difference_norm"].max() <= 2e-7
        ),
    }
    write_json(output_dir / "sampling_aware_covariance_summary.json", gate2_summary)
    if not gate2_summary["gate2_passed"]:
        raise AssertionError("Gate 2 failed: processed-noise Jacobian or diagnostic integration mismatch")

    full_diagnostics = diagnostics[
        (diagnostics["frame_i"] >= FIRST_VALID_FRAME)
        & (diagnostics["frame_j"] < len(frame_times))
    ].sort_values(["frame_i", "frame_j"]).reset_index(drop=True)
    selected_rows = select_mc_edges(full_diagnostics, ref_times, ref_velocity, ref_angular_velocity)
    rng = np.random.default_rng(args.seed)
    mc_rows: list[dict[str, Any]] = []
    mc_details: list[dict[str, np.ndarray]] = []
    for selected in selected_rows:
        frame_i = int(selected["frame_i"])
        frame_j = int(selected["frame_j"])
        sampling_map = build_sampling_map(
            times_ns, edge_id=int(selected["pair_id"]), frame_i=frame_i, frame_j=frame_j,
            start_ns=int(selected["timestamp_i"]), end_ns=int(selected["timestamp_j"]),
        )
        stage = str(selected["motion_stage"])
        edge = build_edge_covariance(
            sampling_map,
            raw_reference_acc_flu=reference_acc,
            raw_reference_gyro_flu=reference_gyro,
            linearized_acc_bias=linearized_ba[frame_j],
            linearized_gyro_bias=linearized_bg[frame_j],
            p_current_stored=stored_covariance[frame_j],
            motion_stage=stage,
        )
        row, detail = monte_carlo_case(
            edge, realizations=args.mc_realizations,
            chunk_size=args.mc_chunk_size, rng=rng,
        )
        mc_rows.append(row)
        mc_details.append(detail)
    mc_frame = pd.DataFrame(mc_rows)
    mc_frame.to_csv(output_dir / "sampling_aware_mc_per_case.csv", index=False)

    phase_means = mc_frame.groupby("phase_ns")["NIS_sampling_mean"].mean().to_dict()
    stage_means = mc_frame.groupby("motion_stage")["NIS_sampling_mean"].mean().to_dict()
    acceptance = {
        "frobenius_median_below_10_percent": bool(mc_frame["P_sampling_vs_MC_relative_frobenius"].median() < 0.10),
        "frobenius_p95_below_20_percent": bool(mc_frame["P_sampling_vs_MC_relative_frobenius"].quantile(0.95) < 0.20),
        "all_case_NIS9_means_within_9_plus_minus_15_percent": bool(mc_frame["NIS_sampling_mean"].between(7.65, 10.35).all()),
        "all_case_block_means_within_3_plus_minus_20_percent": bool(
            mc_frame[["NIS_sampling_p_mean", "NIS_sampling_v_mean", "NIS_sampling_R_mean"]].apply(
                lambda column: column.between(2.4, 3.6).all()
            ).all()
        ),
        "all_whitened_covariance_diagonals_in_0_8_to_1_2": bool(
            (mc_frame["whitened_covariance_diag_min"] >= 0.8).all()
            and (mc_frame["whitened_covariance_diag_max"] <= 1.2).all()
        ),
        "all_max_abs_offdiag_correlations_below_0_15": bool(
            (mc_frame["whitened_max_abs_offdiag_correlation"] < 0.15).all()
        ),
        "all_whitened_mean_components_below_0_05_abs": bool(
            (mc_frame["whitened_mean_max_abs"] < 0.05).all()
        ),
    }
    acceptance["gate3_passed"] = bool(all(acceptance.values()))
    mc_summary = {
        "gate": 3,
        "case_count": int(len(mc_frame)),
        "realizations_per_case": int(args.mc_realizations),
        "covered_phases_ns": sorted(int(value) for value in mc_frame["phase_ns"].unique()),
        "covered_motion_stages": sorted(str(value) for value in mc_frame["motion_stage"].unique()),
        "P_sampling_vs_MC_relative_frobenius": stats(mc_frame["P_sampling_vs_MC_relative_frobenius"]),
        "P_current_vs_MC_relative_frobenius": stats(mc_frame["P_current_vs_MC_relative_frobenius"]),
        "NIS_current_mean": stats(mc_frame["NIS_current_mean"]),
        "NIS_sampling_mean": stats(mc_frame["NIS_sampling_mean"]),
        "NIS_sampling_p_mean": stats(mc_frame["NIS_sampling_p_mean"]),
        "NIS_sampling_v_mean": stats(mc_frame["NIS_sampling_v_mean"]),
        "NIS_sampling_R_mean": stats(mc_frame["NIS_sampling_R_mean"]),
        "whitened_covariance_diag_global_min": float(mc_frame["whitened_covariance_diag_min"].min()),
        "whitened_covariance_diag_global_max": float(mc_frame["whitened_covariance_diag_max"].max()),
        "whitened_max_abs_offdiag_correlation": stats(mc_frame["whitened_max_abs_offdiag_correlation"]),
        "phase_NIS_sampling_mean": {str(key): float(value) for key, value in phase_means.items()},
        "motion_stage_NIS_sampling_mean": {str(key): float(value) for key, value in stage_means.items()},
        "acceptance": acceptance,
    }
    write_json(output_dir / "sampling_aware_mc_summary.json", mc_summary)
    (output_dir / "sampling_aware_monte_carlo_report_cn.md").write_text(
        markdown_report(mc_summary), encoding="utf-8"
    )
    print(json.dumps({"gate1": map_audit["gate1_passed"],
                      "gate2": gate2_summary["gate2_passed"],
                      "gate3": acceptance["gate3_passed"]}, indent=2))
    return 0 if acceptance["gate3_passed"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
