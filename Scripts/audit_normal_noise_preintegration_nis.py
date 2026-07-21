#!/usr/bin/env python3
"""Independent covariance/NIS audit for the HoloOcean normal-noise sequence.

This script is intentionally read-only with respect to the production estimator.
It calls ``preintegrate_imu_local_frame`` twice per frozen camera edge:

1. with the exported noisy IMU measurement;
2. with the same measurement minus the saved per-sample white-noise truth.

Both integrations retain the identical time-varying truth bias, use the same
edge-start truth bias as ``biasHat``, and share exact endpoint interpolation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pypose as pp
import torch
from scipy import signal, stats


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Module.IMUPreintegration import preintegrate_imu_local_frame


COMPONENTS_9 = ["p_x", "p_y", "p_z", "v_x", "v_y", "v_z", "R_x", "R_y", "R_z"]
SIGMA_A = 0.0141258
SIGMA_G = 0.00182898
SIGMA_AW = 0.000386071
SIGMA_GW = 3.57864e-05
CHI2_9_95 = (2.700, 19.023)
CHI2_3_95 = (0.216, 9.348)
FLU_TO_NED = np.diag([1.0, -1.0, -1.0])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--source-dataset-path", type=str, default="")
    parser.add_argument("--tensor-map", type=Path, required=True)
    parser.add_argument("--frame-diagnostics", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--first-valid-frame", type=int, default=90)
    parser.add_argument("--max-frame-exclusive", type=int, default=300)
    parser.add_argument("--expected-edges", type=int, default=209)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(json_ready(payload), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def axis_values(frame: pd.DataFrame, names: list[str]) -> np.ndarray:
    return frame[names].to_numpy(dtype=np.float64)


def interpolate_value(times: np.ndarray, values: np.ndarray, target_ns: int) -> np.ndarray:
    index = int(np.searchsorted(times, target_ns, side="left"))
    if index < len(times) and int(times[index]) == int(target_ns):
        return values[index].copy()
    if index <= 0:
        return values[0].copy()
    if index >= len(times):
        return values[-1].copy()
    left = index - 1
    right = index
    alpha = (int(target_ns) - int(times[left])) / float(int(times[right]) - int(times[left]))
    return values[left] + (values[right] - values[left]) * alpha


def query_interval(
    times: np.ndarray,
    fields: dict[str, np.ndarray],
    start_ns: int,
    end_ns: int,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    if end_ns < start_ns:
        start_ns, end_ns = end_ns, start_ns
    interior_left = int(np.searchsorted(times, start_ns, side="right"))
    interior_right = int(np.searchsorted(times, end_ns, side="left"))
    segment_times = np.concatenate(
        [
            np.asarray([start_ns], dtype=np.int64),
            times[interior_left:interior_right],
            np.asarray([end_ns], dtype=np.int64),
        ]
    )
    output: dict[str, np.ndarray] = {}
    for name, values in fields.items():
        output[name] = np.concatenate(
            [
                interpolate_value(times, values, start_ns).reshape(1, -1),
                values[interior_left:interior_right],
                interpolate_value(times, values, end_ns).reshape(1, -1),
            ],
            axis=0,
        )
    return segment_times, output


def rotate_flu_to_ned(values: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=np.float64) @ FLU_TO_NED.T


def time_weighted_mean(times_ns: np.ndarray, values: np.ndarray) -> np.ndarray:
    if len(times_ns) <= 1:
        return values.reshape(-1, values.shape[-1]).mean(axis=0)
    seconds = (times_ns - times_ns[0]).astype(np.float64) * 1e-9
    duration = float(seconds[-1])
    if duration <= 0.0:
        return values.mean(axis=0)
    return np.trapz(values, x=seconds, axis=0) / duration


def stable_whiten(covariance: np.ndarray, residual: np.ndarray) -> tuple[np.ndarray, float, dict[str, Any]]:
    covariance = 0.5 * (covariance + covariance.T)
    eigenvalues = np.linalg.eigvalsh(covariance)
    max_eigenvalue = float(np.max(eigenvalues))
    rank_tolerance = max(max_eigenvalue * covariance.shape[0] * np.finfo(np.float64).eps, 1e-18)
    positive = eigenvalues > rank_tolerance
    rank = int(np.count_nonzero(positive))
    min_positive = float(np.min(eigenvalues[positive])) if rank else math.nan
    condition = max_eigenvalue / min_positive if rank else math.inf
    cholesky_success = True
    fallback = "none"
    try:
        lower = np.linalg.cholesky(covariance)
        whitened = np.linalg.solve(lower, residual)
    except np.linalg.LinAlgError:
        cholesky_success = False
        fallback = "eigendecomposition_pseudoinverse"
        eigenvalues_full, eigenvectors = np.linalg.eigh(covariance)
        inverse_sqrt = np.zeros_like(eigenvalues_full)
        valid = eigenvalues_full > rank_tolerance
        inverse_sqrt[valid] = 1.0 / np.sqrt(eigenvalues_full[valid])
        whitened = (eigenvectors * inverse_sqrt) @ (eigenvectors.T @ residual)
    nis = float(whitened @ whitened)
    health = {
        "finite": bool(np.isfinite(covariance).all()),
        "symmetry_max_abs": float(np.max(np.abs(covariance - covariance.T))),
        "min_eigenvalue": float(np.min(eigenvalues)),
        "max_eigenvalue": max_eigenvalue,
        "rank_tolerance": rank_tolerance,
        "rank": rank,
        "condition": condition,
        "cholesky_success": cholesky_success,
        "fallback": fallback,
    }
    return whitened, nis, health


def nis_stats(values: np.ndarray, interval: tuple[float, float]) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    low, high = interval
    return {
        "count": int(values.size),
        "min": float(np.min(values)),
        "median": float(np.median(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
        "p05": float(np.percentile(values, 5)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
        "central_95_interval": [low, high],
        "fraction_inside_central_95": float(np.mean((values >= low) & (values <= high))),
        "fraction_below_central_95": float(np.mean(values < low)),
        "fraction_above_central_95": float(np.mean(values > high)),
    }


def component_autocorrelation(values: np.ndarray, lag: int) -> np.ndarray:
    if values.shape[0] <= lag:
        return np.full(values.shape[1], np.nan)
    correlations = []
    for component in range(values.shape[1]):
        left = values[:-lag, component]
        right = values[lag:, component]
        if np.std(left) <= 1e-15 or np.std(right) <= 1e-15:
            correlations.append(np.nan)
        else:
            correlations.append(float(np.corrcoef(left, right)[0, 1]))
    return np.asarray(correlations)


def stage_for_edge(
    ref_times: np.ndarray,
    ref_velocity: np.ndarray,
    ref_angular_velocity: np.ndarray,
    start_ns: int,
    end_ns: int,
) -> str:
    velocity_i = interpolate_value(ref_times, ref_velocity, start_ns)
    velocity_j = interpolate_value(ref_times, ref_velocity, end_ns)
    omega_i = interpolate_value(ref_times, ref_angular_velocity, start_ns)
    omega_j = interpolate_value(ref_times, ref_angular_velocity, end_ns)
    dt = max((end_ns - start_ns) * 1e-9, 1e-12)
    speed_i = float(np.linalg.norm(velocity_i))
    speed_j = float(np.linalg.norm(velocity_j))
    angular_speed = 0.5 * float(np.linalg.norm(omega_i) + np.linalg.norm(omega_j))
    acceleration_norm = float(np.linalg.norm(velocity_j - velocity_i) / dt)
    if max(speed_i, speed_j) < 0.01 and angular_speed < 0.02:
        return "stationary"
    if angular_speed >= 0.02:
        return "turning"
    if speed_j < speed_i - 0.01 or (speed_i >= 0.02 and speed_j < 0.01):
        return "decelerating_or_stopping"
    if speed_j > speed_i + 0.01 or acceleration_norm >= 0.05:
        return "accelerating"
    if max(speed_i, speed_j) >= 0.01 and acceleration_norm < 0.05:
        return "constant_velocity"
    return "unknown"


def tangent_convention_test() -> dict[str, Any]:
    generator = torch.Generator().manual_seed(20260715)
    max_right_error = 0.0
    max_left_expected_error = 0.0
    max_left_vs_right_difference = 0.0
    for _ in range(100):
        reference_rotvec = torch.randn(1, 3, generator=generator, dtype=torch.float64) * 0.8
        phi = torch.randn(1, 3, generator=generator, dtype=torch.float64) * 1e-6
        reference = pp.so3(reference_rotvec).Exp()
        noisy = reference @ pp.so3(phi).Exp()
        right_error = (reference.Inv() @ noisy).Log().tensor()
        left_error = (noisy @ reference.Inv()).Log().tensor()
        expected_left = reference.Act(phi)
        max_right_error = max(max_right_error, float(torch.max(torch.abs(right_error - phi)).item()))
        max_left_expected_error = max(
            max_left_expected_error,
            float(torch.max(torch.abs(left_error - expected_left)).item()),
        )
        max_left_vs_right_difference = max(
            max_left_vs_right_difference,
            float(torch.max(torch.abs(left_error - right_error)).item()),
        )
    return {
        "samples": 100,
        "perturbation": "R_noisy = R_reference * Exp(phi)",
        "audited_error": "Log(R_reference^-1 * R_noisy)",
        "production_rotation_correction": "delta_R_corrected = delta_R * Exp(delta_phi)",
        "max_abs_right_tangent_recovery_error": max_right_error,
        "max_abs_left_tangent_adjoint_error": max_left_expected_error,
        "max_abs_left_vs_right_error_difference": max_left_vs_right_difference,
        "conclusion": "The requested e_R is the right perturbation tangent used by the production bias correction.",
    }


def high_frequency_energy(values: np.ndarray, sample_rate_hz: float, cutoff_hz: float = 5.0) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    output = []
    for axis in range(values.shape[1]):
        frequencies, spectrum = signal.welch(
            values[:, axis],
            fs=sample_rate_hz,
            nperseg=min(128, values.shape[0]),
            detrend="linear",
            scaling="density",
        )
        mask = frequencies >= cutoff_hz
        energy = float(np.trapz(spectrum[mask], frequencies[mask])) if np.count_nonzero(mask) >= 2 else 0.0
        output.append(energy)
    return {"cutoff_hz": cutoff_hz, "axis_energy": output, "total_energy": float(np.sum(output))}


def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = dataset_dir / "metadata.json"
    imu_path = dataset_dir / "imu_data.csv"
    decomposition_path = dataset_dir / "imu_truth_decomposition.csv"
    reference_path = dataset_dir / "ref_pose.csv"
    for path in [metadata_path, imu_path, decomposition_path, reference_path, args.tensor_map, args.frame_diagnostics]:
        if not path.exists():
            raise FileNotFoundError(path)

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    imu_meta = metadata["imu"]
    expected_parameters = {
        "sigma_a": SIGMA_A,
        "sigma_g": SIGMA_G,
        "sigma_aw": SIGMA_AW,
        "sigma_gw": SIGMA_GW,
    }
    actual_parameters = {
        "sigma_a": float(imu_meta["AccelSigma"]) / math.sqrt(float(imu_meta["rate_hz"])),
        "sigma_g": float(imu_meta["AngVelSigma"]) / math.sqrt(float(imu_meta["rate_hz"])),
        "sigma_aw": float(imu_meta["AccelBiasSigma"]) * math.sqrt(float(imu_meta["bias_random_walk_update_hz"])),
        "sigma_gw": float(imu_meta["AngVelBiasSigma"]) * math.sqrt(float(imu_meta["bias_random_walk_update_hz"])),
    }
    for name, expected in expected_parameters.items():
        if not math.isclose(actual_parameters[name], expected, rel_tol=1e-9, abs_tol=1e-12):
            raise AssertionError(f"Runtime {name}={actual_parameters[name]} does not match {expected}")

    decomposition = pd.read_csv(decomposition_path)
    imu_export = pd.read_csv(imu_path)
    reference = pd.read_csv(reference_path)
    if not np.array_equal(decomposition["timestamp"].to_numpy(np.int64), imu_export["timestamp"].to_numpy(np.int64)):
        raise AssertionError("imu_data.csv and truth decomposition timestamps differ")

    times = decomposition["timestamp"].to_numpy(dtype=np.int64)
    fields = {
        "gyro_measured": axis_values(decomposition, ["measured_ang_vel_x", "measured_ang_vel_y", "measured_ang_vel_z"]),
        "acc_measured": axis_values(decomposition, ["measured_lin_acc_x", "measured_lin_acc_y", "measured_lin_acc_z"]),
        "gyro_noise": axis_values(decomposition, ["gyro_noise_x", "gyro_noise_y", "gyro_noise_z"]),
        "acc_noise": axis_values(decomposition, ["acc_noise_x", "acc_noise_y", "acc_noise_z"]),
        "gyro_bias": axis_values(decomposition, ["gyro_bias_x", "gyro_bias_y", "gyro_bias_z"]),
        "acc_bias": axis_values(decomposition, ["acc_bias_x", "acc_bias_y", "acc_bias_z"]),
    }
    exported_gyro = axis_values(imu_export, ["ang_vel_x", "ang_vel_y", "ang_vel_z"])
    exported_acc = axis_values(imu_export, ["lin_acc_x", "lin_acc_y", "lin_acc_z"])
    measured_consistency = {
        "max_abs_gyro_difference": float(np.max(np.abs(fields["gyro_measured"] - exported_gyro))),
        "max_abs_acc_difference": float(np.max(np.abs(fields["acc_measured"] - exported_acc))),
    }
    if measured_consistency["max_abs_gyro_difference"] > 1e-7 or measured_consistency["max_abs_acc_difference"] > 1e-7:
        raise AssertionError(f"Saved decomposition does not reproduce imu_data.csv: {measured_consistency}")

    tensor_map = np.load(args.tensor_map, allow_pickle=False)
    frame_times = tensor_map["frames//time_ns"].astype(np.int64)
    estimated_acc_bias = tensor_map["frames//imu_vio_acc_bias"].astype(np.float64)
    estimated_gyro_bias = tensor_map["frames//imu_vio_gyro_bias"].astype(np.float64)
    stored_preintegration_covariance = tensor_map["frames//imu_vio_cov"].astype(np.float64)
    stored_bias_rw_covariance = tensor_map["frames//imu_vio_bias_rw_cov"].astype(np.float64)
    diagnostics = pd.read_csv(args.frame_diagnostics)
    frozen_edges = diagnostics[
        (diagnostics["frame_i"] >= args.first_valid_frame)
        & (diagnostics["frame_j"] < args.max_frame_exclusive)
    ].copy()
    frozen_edges = frozen_edges.sort_values(["frame_i", "frame_j"]).reset_index(drop=True)
    if len(frozen_edges) != args.expected_edges:
        raise AssertionError(
            f"Expected {args.expected_edges} frozen edges, found {len(frozen_edges)}; experiment interval was not changed"
        )
    expected_i = np.arange(args.first_valid_frame, args.max_frame_exclusive - 1)
    expected_j = expected_i + 1
    if not np.array_equal(frozen_edges["frame_i"].to_numpy(int), expected_i) or not np.array_equal(
        frozen_edges["frame_j"].to_numpy(int), expected_j
    ):
        raise AssertionError("Frozen edge list is not the exact contiguous 90->91 ... 298->299 interval")
    if not np.array_equal(frozen_edges["timestamp_i"].to_numpy(np.int64), frame_times[expected_i]) or not np.array_equal(
        frozen_edges["timestamp_j"].to_numpy(np.int64), frame_times[expected_j]
    ):
        raise AssertionError("Frozen diagnostics timestamps differ from tensor_map frame timestamps")

    ref_times = reference["timestamp"].to_numpy(dtype=np.int64)
    ref_velocity = axis_values(reference, ["vx", "vy", "vz"])
    ref_angular_velocity = axis_values(reference, ["wx", "wy", "wz"])

    manifest_rows: list[dict[str, Any]] = []
    preintegration_rows: list[dict[str, Any]] = []
    bias_rows: list[dict[str, Any]] = []
    dt_small_steps: list[float] = []
    gt_bias_by_frame: dict[int, np.ndarray] = {}

    for frozen_index, edge in frozen_edges.iterrows():
        edge_id = int(edge["pair_id"]) if "pair_id" in frozen_edges.columns else int(frozen_index)
        frame_i = int(edge["frame_i"])
        frame_j = int(edge["frame_j"])
        start_ns = int(edge["timestamp_i"])
        end_ns = int(edge["timestamp_j"])
        segment_times, segment = query_interval(times, fields, start_ns, end_ns)
        segment_dt = np.diff(segment_times).astype(np.float64) * 1e-9
        dt_small_steps.extend(segment_dt.tolist())

        noisy_acc = rotate_flu_to_ned(segment["acc_measured"])
        noisy_gyro = rotate_flu_to_ned(segment["gyro_measured"])
        acc_noise = rotate_flu_to_ned(segment["acc_noise"])
        gyro_noise = rotate_flu_to_ned(segment["gyro_noise"])
        reference_acc = noisy_acc - acc_noise
        reference_gyro = noisy_gyro - gyro_noise
        acc_bias_start = rotate_flu_to_ned(segment["acc_bias"][:1]).reshape(3)
        gyro_bias_start = rotate_flu_to_ned(segment["gyro_bias"][:1]).reshape(3)
        acc_bias_end = rotate_flu_to_ned(segment["acc_bias"][-1:]).reshape(3)
        gyro_bias_end = rotate_flu_to_ned(segment["gyro_bias"][-1:]).reshape(3)
        gt_bias_by_frame[frame_i] = np.concatenate([acc_bias_start, gyro_bias_start])
        gt_bias_by_frame[frame_j] = np.concatenate([acc_bias_end, gyro_bias_end])

        preint_kwargs = {
            "time_ns": torch.from_numpy(segment_times.copy()).long(),
            "sigma_acc": SIGMA_A,
            "sigma_gyro": SIGMA_G,
            "sigma_acc_w": SIGMA_AW,
            "sigma_gyro_w": SIGMA_GW,
            "acc_bias": torch.from_numpy(acc_bias_start.copy()).float(),
            "gyro_bias": torch.from_numpy(gyro_bias_start.copy()).float(),
        }
        noisy_result = preintegrate_imu_local_frame(
            acc=torch.from_numpy(noisy_acc.copy()).float(),
            gyro=torch.from_numpy(noisy_gyro.copy()).float(),
            **preint_kwargs,
        )
        reference_result = preintegrate_imu_local_frame(
            acc=torch.from_numpy(reference_acc.copy()).float(),
            gyro=torch.from_numpy(reference_gyro.copy()).float(),
            **preint_kwargs,
        )
        measurement_only_kwargs = dict(preint_kwargs)
        measurement_only_kwargs["sigma_acc_w"] = 0.0
        measurement_only_kwargs["sigma_gyro_w"] = 0.0
        measurement_only_result = preintegrate_imu_local_frame(
            acc=torch.from_numpy(noisy_acc.copy()).float(),
            gyro=torch.from_numpy(noisy_gyro.copy()).float(),
            **measurement_only_kwargs,
        )

        error_p = (noisy_result.delta_p - reference_result.delta_p).double().cpu().numpy().reshape(3)
        error_v = (noisy_result.delta_v - reference_result.delta_v).double().cpu().numpy().reshape(3)
        error_r = (
            reference_result.delta_R.double().Inv() @ noisy_result.delta_R.double()
        ).Log().tensor().cpu().numpy().reshape(3)
        error_9 = np.concatenate([error_p, error_v, error_r])
        covariance = noisy_result.cov.double().cpu().numpy()
        measurement_only_covariance = measurement_only_result.cov.double().cpu().numpy()
        stored_covariance = stored_preintegration_covariance[frame_j]
        stored_covariance_difference = covariance - stored_covariance
        whitened, nis_9, health = stable_whiten(covariance, error_9)
        _, nis_p, _ = stable_whiten(covariance[0:3, 0:3], error_p)
        _, nis_v, _ = stable_whiten(covariance[3:6, 3:6], error_v)
        _, nis_r, _ = stable_whiten(covariance[6:9, 6:9], error_r)

        motion_stage = stage_for_edge(
            ref_times, ref_velocity, ref_angular_velocity, start_ns, end_ns
        )
        manifest_rows.append(
            {
                "edge_id": edge_id,
                "frame_i": frame_i,
                "frame_j": frame_j,
                "timestamp_i": start_ns,
                "timestamp_j": end_ns,
                "delta_t": (end_ns - start_ns) * 1e-9,
                "imu_sample_count": len(segment_times),
                "first_imu_timestamp": int(segment_times[0]),
                "last_imu_timestamp": int(segment_times[-1]),
                "frame_i_mod_3": frame_i % 3,
                "start_timestamp_mod_10ms_ns": start_ns % 10_000_000,
            }
        )
        preint_row: dict[str, Any] = dict(manifest_rows[-1])
        for name, value in zip(["ep_x", "ep_y", "ep_z"], error_p):
            preint_row[name] = value
        for name, value in zip(["ev_x", "ev_y", "ev_z"], error_v):
            preint_row[name] = value
        for name, value in zip(["er_x", "er_y", "er_z"], error_r):
            preint_row[name] = value
        for index, value in enumerate(whitened):
            preint_row[f"w{index}"] = value
        preint_row.update(
            {
                "nis_9": nis_9,
                "nis_p": nis_p,
                "nis_v": nis_v,
                "nis_R": nis_r,
            }
        )
        for row_index in range(9):
            for column_index in range(9):
                preint_row[f"P_{row_index}{column_index}"] = covariance[row_index, column_index]
        preint_row.update(
            {
                "P_min_eigenvalue": health["min_eigenvalue"],
                "P_max_eigenvalue": health["max_eigenvalue"],
                "P_condition": health["condition"],
                "P_rank": health["rank"],
                "P_symmetry_max_abs": health["symmetry_max_abs"],
                "P_cholesky_success": health["cholesky_success"],
                "P_fallback": health["fallback"],
                "P_recomputed_vs_stored_max_abs": float(np.max(np.abs(stored_covariance_difference))),
                "P_recomputed_vs_stored_relative_frobenius": float(
                    np.linalg.norm(stored_covariance_difference)
                    / max(np.linalg.norm(stored_covariance), 1e-30)
                ),
                "P_recomputed_vs_stored_trace_ratio": float(
                    np.trace(covariance) / max(np.trace(stored_covariance), 1e-30)
                ),
                "P_bias_rw_trace_fraction": float(
                    (np.trace(covariance) - np.trace(measurement_only_covariance))
                    / max(np.trace(covariance), 1e-30)
                ),
                "P_bias_rw_relative_frobenius_contribution": float(
                    np.linalg.norm(covariance - measurement_only_covariance)
                    / max(np.linalg.norm(covariance), 1e-30)
                ),
                "motion_stage": motion_stage,
            }
        )
        preintegration_rows.append(preint_row)

        truth_delta_bias = np.concatenate([acc_bias_end - acc_bias_start, gyro_bias_end - gyro_bias_start])
        estimated_delta_bias = np.concatenate(
            [
                estimated_acc_bias[frame_j] - estimated_acc_bias[frame_i],
                estimated_gyro_bias[frame_j] - estimated_gyro_bias[frame_i],
            ]
        )
        bias_covariance = noisy_result.bias_rw_cov.double().cpu().numpy()
        stored_bias_covariance = stored_bias_rw_covariance[frame_j]
        _, nis_bias_6, bias_health = stable_whiten(bias_covariance, truth_delta_bias)
        _, nis_bias_a, _ = stable_whiten(bias_covariance[0:3, 0:3], truth_delta_bias[0:3])
        _, nis_bias_g, _ = stable_whiten(bias_covariance[3:6, 3:6], truth_delta_bias[3:6])
        acc_noise_mean = time_weighted_mean(segment_times, acc_noise)
        gyro_noise_mean = time_weighted_mean(segment_times, gyro_noise)
        expected_bias_covariance = np.diag(
            np.concatenate(
                ([SIGMA_AW**2 * noisy_result.dt_total] * 3, [SIGMA_GW**2 * noisy_result.dt_total] * 3)
            )
        )
        bias_row: dict[str, Any] = dict(manifest_rows[-1])
        for prefix, values in [
            ("dba_gt", truth_delta_bias[0:3]),
            ("dbg_gt", truth_delta_bias[3:6]),
            ("dba_est", estimated_delta_bias[0:3]),
            ("dbg_est", estimated_delta_bias[3:6]),
            ("acc_white_noise_mean", acc_noise_mean),
            ("gyro_white_noise_mean", gyro_noise_mean),
        ]:
            for axis, value in zip("xyz", values):
                bias_row[f"{prefix}_{axis}"] = value
        bias_row.update(
            {
                "nis_bias_6": nis_bias_6,
                "nis_bias_acc_3": nis_bias_a,
                "nis_bias_gyro_3": nis_bias_g,
                "Q_bias_min_eigenvalue": bias_health["min_eigenvalue"],
                "Q_bias_max_eigenvalue": bias_health["max_eigenvalue"],
                "Q_bias_condition": bias_health["condition"],
                "Q_bias_rank": bias_health["rank"],
                "Q_bias_cholesky_success": bias_health["cholesky_success"],
                "Q_bias_max_abs_offdiag": float(
                    np.max(np.abs(bias_covariance - np.diag(np.diag(bias_covariance))))
                ),
                "Q_bias_max_abs_vs_sigma_squared_dt": float(
                    np.max(np.abs(bias_covariance - expected_bias_covariance))
                ),
                "Q_bias_recomputed_vs_stored_max_abs": float(
                    np.max(np.abs(bias_covariance - stored_bias_covariance))
                ),
                "motion_stage": motion_stage,
            }
        )
        for row_index in range(6):
            for column_index in range(6):
                bias_row[f"Qb_{row_index}{column_index}"] = bias_covariance[row_index, column_index]
        bias_rows.append(bias_row)

    manifest = pd.DataFrame(manifest_rows)
    preintegration = pd.DataFrame(preintegration_rows)
    bias = pd.DataFrame(bias_rows)
    manifest_path = output_dir / "macvio_normal_noise_edge_manifest.csv"
    preintegration_path = output_dir / "macvio_preintegration_nis_per_edge.csv"
    bias_path = output_dir / "macvio_bias_rw_nis_per_edge.csv"
    manifest.to_csv(manifest_path, index=False)
    preintegration.to_csv(preintegration_path, index=False)
    bias.to_csv(bias_path, index=False)

    whitened_matrix = preintegration[[f"w{index}" for index in range(9)]].to_numpy(np.float64)
    empirical_covariance = np.cov(whitened_matrix, rowvar=False, ddof=1)
    empirical_correlation = np.corrcoef(whitened_matrix, rowvar=False)
    covariance_frame = pd.DataFrame(empirical_covariance, index=COMPONENTS_9, columns=COMPONENTS_9)
    covariance_frame.index.name = "component"
    covariance_frame.to_csv(output_dir / "macvio_whitened_residual_covariance.csv")

    diagonal_mask = ~np.eye(9, dtype=bool)
    autocorrelation = {
        str(lag): component_autocorrelation(whitened_matrix, lag) for lag in [1, 2, 5, 10]
    }
    stage_summary: dict[str, Any] = {}
    for stage, group in preintegration.groupby("motion_stage"):
        stage_w = group[[f"w{index}" for index in range(9)]].to_numpy(np.float64)
        stage_summary[stage] = {
            "count": len(group),
            "whitened_mean": np.mean(stage_w, axis=0),
            "whitened_std": np.std(stage_w, axis=0, ddof=1) if len(group) > 1 else np.zeros(9),
            "whitened_covariance": np.cov(stage_w, rowvar=False, ddof=1) if len(group) > 1 else np.zeros((9, 9)),
            "nis_9": nis_stats(group["nis_9"].to_numpy(float), CHI2_9_95),
            "nis_p": nis_stats(group["nis_p"].to_numpy(float), CHI2_3_95),
            "nis_v": nis_stats(group["nis_v"].to_numpy(float), CHI2_3_95),
            "nis_R": nis_stats(group["nis_R"].to_numpy(float), CHI2_3_95),
        }
    phase_summary: dict[str, Any] = {}
    for phase, group in preintegration.groupby("frame_i_mod_3"):
        phase_w = group[[f"w{index}" for index in range(9)]].to_numpy(np.float64)
        phase_summary[str(int(phase))] = {
            "count": len(group),
            "camera_timestamp_phase": {
                "min_mod_10ms_ns": int(group["start_timestamp_mod_10ms_ns"].min()),
                "max_mod_10ms_ns": int(group["start_timestamp_mod_10ms_ns"].max()),
            },
            "whitened_mean": np.mean(phase_w, axis=0),
            "whitened_std": np.std(phase_w, axis=0, ddof=1),
            "nis_9": nis_stats(group["nis_9"].to_numpy(float), CHI2_9_95),
        }

    preintegration_summary = {
        "experiment": {
            "scene": metadata.get("dataset", {}).get("scene_name", dataset_dir.name),
            "runtime_dataset_path": str(dataset_dir),
            "source_dataset_path": args.source_dataset_path or str(dataset_dir),
            "frame_range_inclusive": [0, args.max_frame_exclusive - 1],
            "valid_edge_frame_range": [args.first_valid_frame, args.max_frame_exclusive - 1],
            "valid_edge_count": len(preintegration),
            "frame_time_range_ns": [int(frame_times[0]), int(frame_times[args.max_frame_exclusive - 1])],
            "valid_edge_time_range_ns": [
                int(manifest.iloc[0]["timestamp_i"]),
                int(manifest.iloc[-1]["timestamp_j"]),
            ],
            "metadata_sha256": sha256_file(metadata_path),
            "imu_csv_sha256": sha256_file(imu_path),
            "imu_truth_decomposition_sha256": sha256_file(decomposition_path),
            "edge_manifest_sha256": sha256_file(manifest_path),
            "tensor_map_sha256": sha256_file(args.tensor_map),
            "frame_diagnostics_sha256": sha256_file(args.frame_diagnostics),
        },
        "measured_decomposition_consistency": measured_consistency,
        "rotation_tangent_test": tangent_convention_test(),
        "whitened_error": {
            "component_order": COMPONENTS_9,
            "mean": np.mean(whitened_matrix, axis=0),
            "std": np.std(whitened_matrix, axis=0, ddof=1),
            "empirical_covariance": empirical_covariance,
            "empirical_correlation": empirical_correlation,
            "max_abs_off_diagonal_correlation": float(np.nanmax(np.abs(empirical_correlation[diagonal_mask]))),
            "autocorrelation": autocorrelation,
        },
        "nis_9": nis_stats(preintegration["nis_9"].to_numpy(float), CHI2_9_95),
        "nis_blocks": {
            "p": nis_stats(preintegration["nis_p"].to_numpy(float), CHI2_3_95),
            "v": nis_stats(preintegration["nis_v"].to_numpy(float), CHI2_3_95),
            "R": nis_stats(preintegration["nis_R"].to_numpy(float), CHI2_3_95),
        },
        "covariance_health": {
            "all_finite": bool(np.isfinite(preintegration[[f"P_{r}{c}" for r in range(9) for c in range(9)]].to_numpy(float)).all()),
            "cholesky_success_count": int(preintegration["P_cholesky_success"].sum()),
            "cholesky_failure_count": int((~preintegration["P_cholesky_success"]).sum()),
            "rank_min": int(preintegration["P_rank"].min()),
            "rank_max": int(preintegration["P_rank"].max()),
            "min_eigenvalue_min": float(preintegration["P_min_eigenvalue"].min()),
            "max_eigenvalue_max": float(preintegration["P_max_eigenvalue"].max()),
            "condition_min": float(preintegration["P_condition"].min()),
            "condition_median": float(preintegration["P_condition"].median()),
            "condition_max": float(preintegration["P_condition"].max()),
            "symmetry_max_abs": float(preintegration["P_symmetry_max_abs"].max()),
            "recomputed_vs_stored_max_abs": float(
                preintegration["P_recomputed_vs_stored_max_abs"].max()
            ),
            "recomputed_vs_stored_relative_frobenius_median": float(
                preintegration["P_recomputed_vs_stored_relative_frobenius"].median()
            ),
            "recomputed_vs_stored_relative_frobenius_max": float(
                preintegration["P_recomputed_vs_stored_relative_frobenius"].max()
            ),
            "recomputed_vs_stored_trace_ratio_min": float(
                preintegration["P_recomputed_vs_stored_trace_ratio"].min()
            ),
            "recomputed_vs_stored_trace_ratio_max": float(
                preintegration["P_recomputed_vs_stored_trace_ratio"].max()
            ),
            "bias_rw_trace_fraction_median": float(
                preintegration["P_bias_rw_trace_fraction"].median()
            ),
            "bias_rw_trace_fraction_max": float(
                preintegration["P_bias_rw_trace_fraction"].max()
            ),
            "bias_rw_relative_frobenius_contribution_max": float(
                preintegration["P_bias_rw_relative_frobenius_contribution"].max()
            ),
        },
        "by_motion_stage": stage_summary,
        "by_camera_imu_phase_frame_i_mod_3": phase_summary,
    }
    write_json(output_dir / "macvio_preintegration_nis_summary.json", preintegration_summary)

    truth_delta = bias[[f"dba_gt_{axis}" for axis in "xyz"] + [f"dbg_gt_{axis}" for axis in "xyz"]].to_numpy(float)
    estimated_delta = bias[[f"dba_est_{axis}" for axis in "xyz"] + [f"dbg_est_{axis}" for axis in "xyz"]].to_numpy(float)
    noise_mean = bias[
        [f"acc_white_noise_mean_{axis}" for axis in "xyz"]
        + [f"gyro_white_noise_mean_{axis}" for axis in "xyz"]
    ].to_numpy(float)
    same_axis_correlation = []
    for axis in range(6):
        same_axis_correlation.append(float(np.corrcoef(estimated_delta[:, axis], noise_mean[:, axis])[0, 1]))
    full_increment_noise_correlation = np.empty((6, 6), dtype=np.float64)
    for estimated_axis in range(6):
        for noise_axis in range(6):
            full_increment_noise_correlation[estimated_axis, noise_axis] = np.corrcoef(
                estimated_delta[:, estimated_axis], noise_mean[:, noise_axis]
            )[0, 1]

    frame_indices = np.arange(args.first_valid_frame, args.max_frame_exclusive)
    truth_bias_series = np.stack([gt_bias_by_frame[int(frame)] for frame in frame_indices], axis=0)
    estimated_bias_series = np.concatenate(
        [estimated_acc_bias[frame_indices], estimated_gyro_bias[frame_indices]], axis=1
    )
    estimated_hf = high_frequency_energy(estimated_bias_series, sample_rate_hz=30.0)
    truth_hf = high_frequency_energy(truth_bias_series, sample_rate_hz=30.0)
    increment_energy_est = np.mean(estimated_delta**2, axis=0)
    increment_energy_truth = np.mean(truth_delta**2, axis=0)
    bias_summary = {
        "component_order": ["ba_x", "ba_y", "ba_z", "bg_x", "bg_y", "bg_z"],
        "nis_bias_6": nis_stats(
            bias["nis_bias_6"].to_numpy(float),
            (float(stats.chi2.ppf(0.025, 6)), float(stats.chi2.ppf(0.975, 6))),
        ),
        "nis_bias_acc_3": nis_stats(bias["nis_bias_acc_3"].to_numpy(float), CHI2_3_95),
        "nis_bias_gyro_3": nis_stats(bias["nis_bias_gyro_3"].to_numpy(float), CHI2_3_95),
        "production_bias_covariance": {
            "cholesky_success_count": int(bias["Q_bias_cholesky_success"].sum()),
            "rank_min": int(bias["Q_bias_rank"].min()),
            "rank_max": int(bias["Q_bias_rank"].max()),
            "max_abs_off_diagonal": float(bias["Q_bias_max_abs_offdiag"].max()),
            "max_abs_difference_from_diag_sigma_squared_dt": float(
                bias["Q_bias_max_abs_vs_sigma_squared_dt"].max()
            ),
            "recomputed_vs_stored_max_abs": float(
                bias["Q_bias_recomputed_vs_stored_max_abs"].max()
            ),
        },
        "estimated_increment_vs_white_noise": {
            "same_axis_pearson_correlation": same_axis_correlation,
            "maximum_absolute_same_axis_correlation": float(np.max(np.abs(same_axis_correlation))),
            "full_6x6_pearson_correlation": full_increment_noise_correlation,
            "maximum_absolute_any_axis_correlation": float(
                np.max(np.abs(full_increment_noise_correlation))
            ),
        },
        "increment_energy": {
            "estimated_mean_squared_increment": increment_energy_est,
            "truth_mean_squared_increment": increment_energy_truth,
            "estimated_to_truth_ratio": increment_energy_est / np.maximum(increment_energy_truth, 1e-30),
            "estimated_total": float(np.sum(increment_energy_est)),
            "truth_total": float(np.sum(increment_energy_truth)),
            "estimated_to_truth_total_ratio": float(
                np.sum(increment_energy_est) / max(np.sum(increment_energy_truth), 1e-30)
            ),
        },
        "spectral_high_frequency_energy": {
            "estimated": estimated_hf,
            "truth": truth_hf,
            "estimated_to_truth_total_ratio": estimated_hf["total_energy"]
            / max(truth_hf["total_energy"], 1e-30),
        },
    }
    write_json(output_dir / "macvio_bias_rw_nis_summary.json", bias_summary)

    dt_array = np.asarray(dt_small_steps, dtype=np.float64)
    runtime_contract = {
        "production_path": "standard_local_frame_preintegration",
        "parameter_source": "metadata.json (frame calibration); odometry.yaml is fallback and was not used",
        "axis_model": "isotropic scalar repeated on x/y/z",
        "continuous_noise_density": actual_parameters,
        "holocean_metadata_discrete_parameters": {
            "rate_hz": imu_meta["rate_hz"],
            "bias_random_walk_update_hz": imu_meta["bias_random_walk_update_hz"],
            "AccelSigma": imu_meta["AccelSigma"],
            "AngVelSigma": imu_meta["AngVelSigma"],
            "AccelBiasSigma": imu_meta["AccelBiasSigma"],
            "AngVelBiasSigma": imu_meta["AngVelBiasSigma"],
            "sigma_unit": imu_meta["sigma_unit"],
            "bias_sigma_unit": imu_meta["bias_sigma_unit"],
        },
        "saved_white_noise_empirical_statistics": {
            "acc_mean_flu": np.mean(fields["acc_noise"], axis=0),
            "acc_std_flu": np.std(fields["acc_noise"], axis=0, ddof=0),
            "acc_configured_per_sample_std": SIGMA_A * math.sqrt(float(imu_meta["rate_hz"])),
            "gyro_mean_flu": np.mean(fields["gyro_noise"], axis=0),
            "gyro_std_flu": np.std(fields["gyro_noise"], axis=0, ddof=0),
            "gyro_configured_per_sample_std": SIGMA_G * math.sqrt(float(imu_meta["rate_hz"])),
        },
        "measurement_process_noise_per_small_step": {
            "Q_measurement_acc_dt": "diag([sigma_a^2/dt]*3)",
            "Q_measurement_gyro_dt": "diag([sigma_g^2/dt]*3)",
            "numeric_variance_density_acc": SIGMA_A**2,
            "numeric_variance_density_gyro": SIGMA_G**2,
            "G_acc_position": "0.5 * R * dt^2",
            "G_acc_velocity": "R * dt",
            "G_gyro_rotation": "I * dt",
        },
        "bias_process_noise_per_small_step": {
            "Q_bias_acc_dt": "diag([sigma_aw^2/dt]*3), injected through G_bias=I*dt",
            "Q_bias_gyro_dt": "diag([sigma_gw^2/dt]*3), injected through G_bias=I*dt",
            "effective_increment_covariance": "diag([sigma_aw^2*dt]*3 + [sigma_gw^2*dt]*3)",
            "numeric_variance_density_acc_bias": SIGMA_AW**2,
            "numeric_variance_density_gyro_bias": SIGMA_GW**2,
        },
        "per_camera_edge_bias_covariance": "Cov15[9:15,9:15], numerically equal to diag(sigma_aw^2*edge_dt*I3, sigma_gw^2*edge_dt*I3)",
        "additional_integration_covariance": {
            "enabled": False,
            "value": 0.0,
            "note": "Cov15 starts at zero; production MACVIO adds no standalone integration covariance.",
        },
        "dt_contract": {
            "source": "exact camera endpoint interpolation plus interior IMU timestamps",
            "small_step_count": int(dt_array.size),
            "min_s": float(dt_array.min()),
            "median_s": float(np.median(dt_array)),
            "max_s": float(dt_array.max()),
            "unique_rounded_s": np.unique(np.round(dt_array, 12)),
            "uses": [
                "delta_p += delta_v*dt + 0.5*a_corr*dt^2",
                "delta_v += a_corr*dt",
                "delta_R *= Exp(gyro_mid*dt)",
                "F position/velocity/rotation/bias blocks use dt or dt^2",
                "Q_sample divides continuous variance density by dt",
                "G multiplies noise by dt or dt^2",
            ],
        },
        "motion_stage_contract": {
            "stationary": "max endpoint speed < 0.01 m/s and mean angular speed < 0.02 rad/s",
            "turning": "mean angular speed >= 0.02 rad/s",
            "decelerating_or_stopping": "speed decreases by > 0.01 m/s or reaches < 0.01 m/s",
            "accelerating": "speed increases by > 0.01 m/s or endpoint acceleration norm >= 0.05 m/s^2",
            "constant_velocity": "moving and endpoint acceleration norm < 0.05 m/s^2",
            "unknown": "none of the above",
        },
        "covariance_and_error_order": ["p", "v", "R"],
        "rotation_tangent": preintegration_summary["rotation_tangent_test"],
        "code_locations": {
            "metadata_conversion": "DataLoader/Dataset/GeneralStereoIMU.py:238-280",
            "local_preintegration_entry": "Module/IMUPreintegration.py:264-294",
            "state_and_noise_propagation": "Module/IMUPreintegration.py:181-249",
            "runtime_selection": "Odometry/MACVO.py:1538-1572",
        },
        "input_hashes": preintegration_summary["experiment"],
    }
    write_json(output_dir / "macvio_runtime_noise_contract.json", runtime_contract)

    print(json.dumps({
        "output_dir": str(output_dir),
        "edges": len(preintegration),
        "nis_9_mean": preintegration_summary["nis_9"]["mean"],
        "nis_9_median": preintegration_summary["nis_9"]["median"],
        "whitened_mean": preintegration_summary["whitened_error"]["mean"].tolist(),
        "whitened_std": preintegration_summary["whitened_error"]["std"].tolist(),
        "bias_nis_6_mean": bias_summary["nis_bias_6"]["mean"],
        "bias_estimated_to_truth_increment_energy_ratio": bias_summary["increment_energy"]["estimated_to_truth_total_ratio"],
    }, indent=2))


if __name__ == "__main__":
    main()
