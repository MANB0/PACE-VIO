#!/usr/bin/env python3
"""Replay a stochastic-clone ESKF from MACVO relative-pose factors and raw IMU."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pypose as pp
import torch


ROOT = Path("/home/admin1/macvo-dev")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Utility.IMUCSV import IMUCSVLoader  # noqa: E402
from Utility.IMUKinematics import (  # noqa: E402
    imu_bias_sigma_to_continuous_random_walk_density,
    imu_sigma_to_continuous_density,
    transform_imu_samples_to_internal_frame,
)
from Utility.PoseFrame import convert_pose_frame, write_timed_se3_csv  # noqa: E402
from Utility.RelativePoseFactorCache import (  # noqa: E402
    RelativePoseFactorCacheReader,
    camera_factor_to_body_factor,
)
from Utility.StochasticCloneESKF import (  # noqa: E402
    ESKFNoiseDensities,
    ESKFNominalState,
    augment_with_current_pose_clone,
    initial_navigation_covariance,
    propagate_imu_knots,
    reclone_current_pose,
    update_relative_pose,
)


SCENE = "clear_circle_truth_normal_noise"
DATASET = Path(
    "/mnt/e/文档/holoocean/code/recordings/"
    "batch_clear_truth_paths_20260713_static63_variants"
) / SCENE
CACHE = ROOT / "VisualCache/static63_unique_visual_20260713" / SCENE
BASELINE = ROOT / (
    "Results/circle_straight_normal_noise_two_state_standard_full_20260715/"
    "trial_1/vio_two_state_fixed_lag_standard_full/"
    "clear_circle_truth_normal_noise"
)
OUTPUT = ROOT / "Results/circle_relative_pose_clone_eskf_short_20260718"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _axis_value(metadata: dict, key: str) -> float | list[float]:
    if f"{key}XYZ" in metadata:
        return metadata[f"{key}XYZ"]
    value = metadata[key]
    if isinstance(value, dict) and all(axis in value for axis in ("x", "y", "z")):
        return [value["x"], value["y"], value["z"]]
    return value


def _three_axis(value: float | tuple[float, ...] | list[float]) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=torch.float64).reshape(-1)
    if tensor.numel() == 1:
        tensor = tensor.repeat(3)
    if tensor.numel() != 3:
        raise ValueError(f"expected scalar or three-axis sigma, got {tensor.numel()}")
    return tensor


def _noise_from_metadata(metadata: dict) -> tuple[ESKFNoiseDensities, dict[str, object]]:
    imu = metadata["imu"]
    rate = float(imu["rate_hz"])
    bias_rate = float(imu.get("bias_random_walk_update_hz", rate))
    sigma_unit = str(imu.get("sigma_unit", "legacy_sqrt_rate_scaled"))
    bias_unit = str(imu.get("bias_sigma_unit", sigma_unit))
    accel = _three_axis(
        imu_sigma_to_continuous_density(
            _axis_value(imu, "AccelSigma"), rate, sigma_unit
        )
    )
    gyro = _three_axis(
        imu_sigma_to_continuous_density(
            _axis_value(imu, "AngVelSigma"), rate, sigma_unit
        )
    )
    accel_bias = _three_axis(
        imu_bias_sigma_to_continuous_random_walk_density(
            _axis_value(imu, "AccelBiasSigma"), bias_rate, bias_unit
        )
    )
    gyro_bias = _three_axis(
        imu_bias_sigma_to_continuous_random_walk_density(
            _axis_value(imu, "AngVelBiasSigma"), bias_rate, bias_unit
        )
    )
    return (
        ESKFNoiseDensities(accel, gyro, accel_bias, gyro_bias),
        {
            "measurement_rate_hz": rate,
            "bias_random_walk_update_hz": bias_rate,
            "sigma_unit": sigma_unit,
            "bias_sigma_unit": bias_unit,
            "continuous_accel_noise_density": accel.tolist(),
            "continuous_gyro_noise_density": gyro.tolist(),
            "continuous_accel_bias_rw_density": accel_bias.tolist(),
            "continuous_gyro_bias_rw_density": gyro_bias.tolist(),
        },
    )


def _pose_row(pose: torch.Tensor) -> np.ndarray:
    return pose.detach().cpu().double().reshape(7).numpy()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--cache", type=Path, default=CACHE)
    parser.add_argument("--baseline", type=Path, default=BASELINE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--seq-to", type=int, default=300)
    parser.add_argument("--finite-difference-epsilon", type=float, default=1.0e-6)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata_path = args.dataset / "metadata.json"
    imu_path = args.dataset / "imu_data.csv"
    tensor_path = args.baseline / "tensor_map.npz"
    sidecar_path = args.cache / "relative_pose_factors.npz"
    for path in (metadata_path, imu_path, tensor_path, sidecar_path):
        if not path.exists():
            raise FileNotFoundError(path)

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    noise, noise_contract = _noise_from_metadata(metadata)
    tensors = np.load(tensor_path, allow_pickle=False)
    timestamps = tensors["frames//time_ns"].astype(np.int64)
    frame_count = min(int(args.seq_to), int(timestamps.size))
    if frame_count < 3:
        raise ValueError("--seq-to must retain at least three image frames")
    dt_values = tensors["frames//imu_vio_dt"]
    point_counts = tensors["frames//visual_relative_pose_num_points"]
    valid_j = np.flatnonzero((dt_values > 0.0) & (point_counts > 0) & (np.arange(len(dt_values)) < frame_count))
    if valid_j.size == 0:
        raise RuntimeError("no valid relative-pose/IMU edges in requested frame range")
    first_j = int(valid_j[0])
    start_frame = first_j - 1
    if valid_j.tolist() != list(range(first_j, frame_count)):
        raise RuntimeError("clone ESKF requires contiguous valid image edges after initialization")

    dtype = torch.float64
    device = torch.device("cpu")
    extrinsic_CI = pp.SE3(
        torch.from_numpy(tensors["frames//imu_vio_sensor_T_imu"][first_j]).to(dtype).reshape(1, 7)
    )
    pose_WC_start = pp.SE3(
        torch.from_numpy(tensors["frames//pose"][start_frame]).to(dtype).reshape(1, 7)
    )
    nominal = ESKFNominalState(
        pose_WB=(pose_WC_start @ extrinsic_CI).tensor(),
        velocity_W=torch.from_numpy(
            tensors["frames//imu_vio_prev_velocity_world"][first_j]
        ).to(dtype),
        acc_bias=torch.from_numpy(
            tensors["frames//imu_vio_prev_acc_bias"][first_j]
        ).to(dtype),
        gyro_bias=torch.from_numpy(
            tensors["frames//imu_vio_prev_gyro_bias"][first_j]
        ).to(dtype),
    ).to(dtype=dtype, device=device)
    covariance_nav = initial_navigation_covariance(
        pose_translation_std=1.0e-5,
        pose_rotation_std=1.0e-5,
        velocity_std=0.05,
        acc_bias_std=0.2,
        gyro_bias_std=0.02,
        dtype=dtype,
        device=device,
    )
    state = augment_with_current_pose_clone(
        nominal, covariance_nav, int(timestamps[start_frame])
    )

    imu_loader = IMUCSVLoader(imu_path)
    sidecar = RelativePoseFactorCacheReader(args.cache)
    flu_to_ned = pp.SE3(
        torch.tensor([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]], dtype=dtype)
    )
    gravity_world = torch.from_numpy(
        tensors["frames//imu_vio_gravity_world"][first_j]
    ).to(dtype)

    poses_internal = np.asarray(tensors["frames//pose"][:frame_count], dtype=np.float64).copy()
    poses_internal[start_frame] = _pose_row(pose_WC_start.tensor())
    diagnostics: list[dict[str, object]] = []
    start_time = time.perf_counter()
    for frame_j in valid_j:
        frame_j = int(frame_j)
        frame_i = frame_j - 1
        knot_time, acc_raw, gyro_raw = imu_loader.query_range(
            int(timestamps[frame_i]), int(timestamps[frame_j])
        )
        if knot_time.numel() < 2:
            raise RuntimeError(f"insufficient IMU coverage for edge {frame_i}->{frame_j}")
        acc_body, gyro_body = transform_imu_samples_to_internal_frame(
            acc_raw.to(dtype), gyro_raw.to(dtype), flu_to_ned
        )
        state = propagate_imu_knots(
            state,
            knot_time,
            acc_body,
            gyro_body,
            gravity_world,
            noise,
        )
        packet = sidecar.load_pair(
            frame_i, frame_j, str(sidecar.visual_hashes[frame_i])
        )
        measurement_body, covariance_body = camera_factor_to_body_factor(
            packet.measurement_CiCj.to(dtype),
            packet.covariance.to(dtype),
            extrinsic_CI.tensor(),
        )
        state, update = update_relative_pose(
            state,
            measurement_body,
            covariance_body,
            finite_difference_epsilon=float(args.finite_difference_epsilon),
        )
        pose_WC = pp.SE3(state.nominal.pose_WB) @ extrinsic_CI.Inv()
        poses_internal[frame_j] = _pose_row(pose_WC.tensor())
        covariance_nav_now = state.covariance[:15, :15]
        nav_eigenvalues = torch.linalg.eigvalsh(covariance_nav_now)
        diagnostics.append(
            {
                "frame_i": frame_i,
                "frame_j": frame_j,
                "timestamp_ns": int(timestamps[frame_j]),
                "dt_s": float((timestamps[frame_j] - timestamps[frame_i]) * 1.0e-9),
                "imu_knot_count": int(knot_time.numel()),
                "visual_num_points": int(packet.num_points),
                "visual_num_inliers": int(packet.num_inliers),
                "visual_inlier_ratio": float(packet.num_inliers / max(packet.num_points, 1)),
                "visual_point_mean_mahalanobis_sq": float(packet.mean_mahalanobis_sq),
                "pose_nis": update.nis,
                "pose_residual_norm_before": update.residual_norm_before,
                "pose_residual_norm_after": update.residual_norm_after,
                "position_correction_norm_m": float(torch.linalg.vector_norm(update.increment[0:3]).item()),
                "velocity_correction_norm_mps": float(torch.linalg.vector_norm(update.increment[3:6]).item()),
                "rotation_correction_norm_rad": float(torch.linalg.vector_norm(update.increment[6:9]).item()),
                "acc_bias_correction_norm_mps2": float(torch.linalg.vector_norm(update.increment[9:12]).item()),
                "gyro_bias_correction_norm_radps": float(torch.linalg.vector_norm(update.increment[12:15]).item()),
                "clone_position_correction_norm_m": float(
                    torch.linalg.vector_norm(update.increment[15:18]).item()
                ),
                "clone_rotation_correction_norm_rad": float(
                    torch.linalg.vector_norm(update.increment[18:21]).item()
                ),
                "relative_position_correction_norm_m": float(
                    torch.linalg.vector_norm(update.increment[0:3] - update.increment[15:18]).item()
                ),
                "velocity_x": float(state.nominal.velocity_W[0].item()),
                "velocity_y": float(state.nominal.velocity_W[1].item()),
                "velocity_z": float(state.nominal.velocity_W[2].item()),
                "acc_bias_x": float(state.nominal.acc_bias[0].item()),
                "acc_bias_y": float(state.nominal.acc_bias[1].item()),
                "acc_bias_z": float(state.nominal.acc_bias[2].item()),
                "gyro_bias_x": float(state.nominal.gyro_bias[0].item()),
                "gyro_bias_y": float(state.nominal.gyro_bias[1].item()),
                "gyro_bias_z": float(state.nominal.gyro_bias[2].item()),
                "nav_cov_min_eigenvalue": float(nav_eigenvalues.min().item()),
                "nav_cov_max_eigenvalue": float(nav_eigenvalues.max().item()),
                "finite": bool(update.finite),
            }
        )
        state = reclone_current_pose(state)

    runtime_s = time.perf_counter() - start_time
    args.output.mkdir(parents=True, exist_ok=True)
    poses_nwu = convert_pose_frame(poses_internal, "NED", "NWU")
    write_timed_se3_csv(args.output / "poses.csv", timestamps[:frame_count], poses_nwu)
    with (args.output / "eskf_diagnostics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(diagnostics[0]))
        writer.writeheader()
        writer.writerows(diagnostics)
    manifest = {
        "method": "relative_pose_stochastic_clone_eskf",
        "scene": SCENE,
        "frame_count": frame_count,
        "first_active_edge": [start_frame, first_j],
        "initial_state_source": (
            "Relative-pose baseline state entering its first post-static edge; "
            "no later baseline states are consumed"
        ),
        "static_prefix_source": f"baseline frames [0,{start_frame}) for output only",
        "truth_used_by_estimator": False,
        "relative_pose_contract": (
            "MACVO sidecar T_CiCj, right perturbation covariance [translation,rotation], "
            "converted by camera_factor_to_body_factor"
        ),
        "imu_contract": "raw FLU CSV -> existing Rx180 FLU-to-internal-NED transform",
        "noise_contract": noise_contract,
        "runtime_s": runtime_s,
        "inputs": {
            "metadata": str(metadata_path),
            "metadata_sha256": _sha256(metadata_path),
            "imu_csv": str(imu_path),
            "imu_csv_sha256": _sha256(imu_path),
            "relative_pose_sidecar": str(sidecar_path),
            "relative_pose_sidecar_sha256": _sha256(sidecar_path),
            "baseline_tensor_map": str(tensor_path),
        },
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(args.output)


if __name__ == "__main__":
    main()
