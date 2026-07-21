#!/usr/bin/env python3
"""Run the Static63 calibrated IMU chain without visual residuals.

This is not a second strapdown implementation. It reuses the production
static initializer, IMU preintegrator, attitude propagation, calibration-unit
conversion, and camera-frame interval interpolation used by MACVO's
``staticinit_calibrated`` fusion mode. The only omitted component is visual
optimization. Bias is therefore initialized from the startup-static interval
and then held constant, because IMU-only propagation cannot observe bias drift.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pypose as pp
import torch


WORKDIR = Path("/home/admin1/macvo-dev")
if str(WORKDIR) not in sys.path:
    sys.path.insert(0, str(WORKDIR))

from Utility.IMUCSV import IMUCSVLoader  # noqa: E402
from Utility.IMUKinematics import (  # noqa: E402
    estimate_static_imu_initialization,
    imu_bias_sigma_to_continuous_random_walk_density,
    imu_sigma_to_continuous_density,
    integrate_gyro_attitude_world,
    propagate_imu_velocity_world,
)


def _load_production_preintegrator():
    module_path = WORKDIR / "Module" / "IMUPreintegration.py"
    module_name = "_macvo_production_imu_preintegration"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load production preintegrator from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module.preintegrate_imu


preintegrate_imu = _load_production_preintegrator()


DEFAULT_BATCH_ROOT = Path(
    "/mnt/e/\u6587\u6863/holoocean/code/recordings/"
    "batch_clear_truth_paths_20260713_static63_variants"
)
DEFAULT_OUTPUT_ROOT = WORKDIR / "Results" / "static63_calibrated_imu_only_four_configs_20260713"
DEFAULT_LOG = WORKDIR / "logs" / "static63_calibrated_imu_only_four_configs_20260713.log"
METHOD = "imu_only_staticinit_calibrated"
SCENES = (
    "clear_circle_truth_normal_noise",
    "clear_circle_truth_bias_no_noise",
    "clear_circle_truth_noise_no_bias",
    "clear_circle_truth_no_noise_no_bias",
    "clear_stop_turn_rectangle_truth_normal_noise",
    "clear_stop_turn_rectangle_truth_bias_no_noise",
    "clear_stop_turn_rectangle_truth_noise_no_bias",
    "clear_stop_turn_rectangle_truth_no_noise_no_bias",
    "clear_straight_truth_normal_noise",
    "clear_straight_truth_bias_no_noise",
    "clear_straight_truth_noise_no_bias",
    "clear_straight_truth_no_noise_no_bias",
)


@dataclass(frozen=True)
class CalibrationParameters:
    gravity: float
    measurement_rate_hz: float
    sigma_acc: float | tuple[float, float, float]
    sigma_gyro: float | tuple[float, float, float]
    sigma_acc_w: float | tuple[float, float, float]
    sigma_gyro_w: float | tuple[float, float, float]


@dataclass(frozen=True)
class CalibratedImuOnlyResult:
    time_ns: np.ndarray
    camera_position_w: np.ndarray
    imu_velocity_w: np.ndarray
    body_to_world: np.ndarray
    covariance_trace: np.ndarray
    static_end_ns: int
    static_acc_bias: np.ndarray
    static_gyro_bias: np.ndarray
    static_diagnostics: dict[str, object]


def _axis_value(config: dict, key: str):
    xyz_key = f"{key}XYZ"
    if xyz_key in config:
        return config[xyz_key]
    value = config[key]
    if isinstance(value, dict) and all(axis in value for axis in ("x", "y", "z")):
        return [value["x"], value["y"], value["z"]]
    return value


def _zero_density_like(value):
    tensor = torch.as_tensor(value, dtype=torch.float64).reshape(-1)
    if tensor.numel() == 1:
        return 0.0
    return (0.0, 0.0, 0.0)


def load_calibration_parameters(metadata: dict) -> CalibrationParameters:
    imu = metadata.get("imu", {})
    rate_hz = float(imu.get("rate_hz", 100.0))
    bias_rate_hz = float(imu.get("bias_random_walk_update_hz", rate_hz))
    sigma_unit = str(imu.get("sigma_unit", "legacy_sqrt_rate_scaled"))
    bias_sigma_unit = str(imu.get("bias_sigma_unit", sigma_unit))

    acc_raw = _axis_value(imu, "AccelSigma")
    gyro_raw = _axis_value(imu, "AngVelSigma")
    sigma_acc = imu_sigma_to_continuous_density(acc_raw, rate_hz, sigma_unit)
    sigma_gyro = imu_sigma_to_continuous_density(gyro_raw, rate_hz, sigma_unit)

    if "AccelBiasSigma" in imu:
        sigma_acc_w = imu_bias_sigma_to_continuous_random_walk_density(
            _axis_value(imu, "AccelBiasSigma"), bias_rate_hz, bias_sigma_unit
        )
    else:
        sigma_acc_w = _zero_density_like(sigma_acc)
    if "AngVelBiasSigma" in imu:
        sigma_gyro_w = imu_bias_sigma_to_continuous_random_walk_density(
            _axis_value(imu, "AngVelBiasSigma"), bias_rate_hz, bias_sigma_unit
        )
    else:
        sigma_gyro_w = _zero_density_like(sigma_gyro)

    return CalibrationParameters(
        gravity=float(imu.get("gravity_m_s2", 9.8)),
        measurement_rate_hz=rate_hz,
        sigma_acc=sigma_acc,
        sigma_gyro=sigma_gyro,
        sigma_acc_w=sigma_acc_w,
        sigma_gyro_w=sigma_gyro_w,
    )


def _sample_sigma_limit(
    density,
    rate_hz: float,
    *,
    floor: float,
    multiplier: float,
) -> torch.Tensor:
    values = torch.as_tensor(density, dtype=torch.float64).reshape(-1)
    if values.numel() == 1:
        values = values.repeat(3)
    sample_sigma = values * math.sqrt(max(float(rate_hz), 1e-9))
    return torch.maximum(
        sample_sigma * float(multiplier),
        torch.full_like(sample_sigma, float(floor)),
    )


def _collect_static_samples_like_fusion(
    camera_time_ns: np.ndarray,
    imu_loader: IMUCSVLoader,
    required_end_ns: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reproduce MACVO's frame-by-frame startup IMU accumulation."""
    time_chunks: list[torch.Tensor] = []
    acc_chunks: list[torch.Tensor] = []
    gyro_chunks: list[torch.Tensor] = []
    last_time_ns: int | None = None

    for index in range(1, int(camera_time_ns.size)):
        interval_time, interval_acc, interval_gyro = imu_loader.query_range(
            int(camera_time_ns[index - 1]),
            int(camera_time_ns[index]),
        )
        if last_time_ns is not None:
            keep = interval_time > int(last_time_ns)
            interval_time = interval_time[keep]
            interval_acc = interval_acc[keep]
            interval_gyro = interval_gyro[keep]
        if interval_time.numel() == 0:
            continue
        time_chunks.append(interval_time)
        acc_chunks.append(interval_acc)
        gyro_chunks.append(interval_gyro)
        last_time_ns = int(interval_time[-1].item())
        if last_time_ns >= int(required_end_ns):
            break

    if not time_chunks:
        raise RuntimeError("No IMU samples available for static initialization")
    all_time = torch.cat(time_chunks, dim=0)
    all_acc = torch.cat(acc_chunks, dim=0)
    all_gyro = torch.cat(gyro_chunks, dim=0)
    if int(all_time[-1].item()) < int(required_end_ns):
        raise RuntimeError(
            "IMU coverage does not span the configured static initialization interval"
        )

    end_index = int(
        torch.searchsorted(
            all_time,
            torch.tensor(int(required_end_ns)),
            right=False,
        ).item()
    )
    end_index = min(end_index + 1, int(all_time.numel()))
    return all_time[:end_index], all_acc[:end_index], all_gyro[:end_index]


def calibrated_imu_only_nwu(
    *,
    camera_time_ns: np.ndarray,
    imu_loader: IMUCSVLoader,
    initial_camera_position_w: np.ndarray,
    initial_body_to_world: np.ndarray,
    camera_to_imu_body: np.ndarray,
    calibration: CalibrationParameters,
    static_duration_s: float = 3.0,
    static_sigma_multiplier: float = 5.0,
    static_gyro_mean_norm_max: float = 0.03,
    static_acc_norm_error_max: float = 0.6,
) -> CalibratedImuOnlyResult:
    """Propagate the calibrated Fusion IMU branch in world NWU coordinates."""
    camera_time_ns = np.asarray(camera_time_ns, dtype=np.int64).reshape(-1)
    if camera_time_ns.size == 0:
        raise ValueError("camera_time_ns must not be empty")

    initial_R = pp.from_matrix(
        torch.as_tensor(initial_body_to_world, dtype=torch.float64).reshape(1, 3, 3),
        pp.SO3_type,
    )
    static_start_ns = max(int(camera_time_ns[0]), int(imu_loader.time_ns[0].item()))
    required_end_ns = static_start_ns + int(round(float(static_duration_s) * 1e9))
    static_time, static_acc, static_gyro = _collect_static_samples_like_fusion(
        camera_time_ns,
        imu_loader,
        required_end_ns,
    )

    static_result = estimate_static_imu_initialization(
        time_ns=static_time,
        acc_body=static_acc,
        gyro_body=static_gyro,
        initial_body_to_world=initial_R,
        gravity=-abs(float(calibration.gravity)),
        min_duration_s=float(static_duration_s),
        gyro_mean_norm_max=float(static_gyro_mean_norm_max),
        gyro_std_max=_sample_sigma_limit(
            calibration.sigma_gyro,
            calibration.measurement_rate_hz,
            floor=0.005,
            multiplier=static_sigma_multiplier,
        ),
        acc_norm_error_max=float(static_acc_norm_error_max),
        acc_std_max=_sample_sigma_limit(
            calibration.sigma_acc,
            calibration.measurement_rate_hz,
            floor=0.05,
            multiplier=static_sigma_multiplier,
        ),
    )
    if not static_result.stationary:
        raise RuntimeError(
            "Configured startup interval is not stationary: "
            f"{static_result.failure_reason}"
        )

    n = int(camera_time_ns.size)
    positions_camera = np.zeros((n, 3), dtype=np.float64)
    velocities_imu = np.zeros((n, 3), dtype=np.float64)
    rotations = np.zeros((n, 3, 3), dtype=np.float64)
    covariance_trace = np.zeros(n, dtype=np.float64)

    anchor_camera = np.asarray(initial_camera_position_w, dtype=np.float64).reshape(3)
    camera_to_imu = np.asarray(camera_to_imu_body, dtype=np.float64).reshape(3)
    R = static_result.body_to_world.double()
    R_matrix = R.matrix().reshape(3, 3).cpu().numpy()
    p_imu = anchor_camera + R_matrix @ camera_to_imu
    v_imu = torch.zeros(3, dtype=torch.float32)
    gravity_world = torch.tensor(
        [0.0, 0.0, -abs(float(calibration.gravity))], dtype=torch.float32
    )

    for index, frame_ns in enumerate(camera_time_ns):
        if int(frame_ns) <= required_end_ns:
            positions_camera[index] = anchor_camera
            velocities_imu[index] = 0.0
            rotations[index] = np.asarray(initial_body_to_world, dtype=np.float64).reshape(3, 3)
            continue

        previous_frame_ns = int(camera_time_ns[index - 1]) if index > 0 else required_end_ns
        interval_start_ns = max(previous_frame_ns, required_end_ns)
        interval_time, interval_acc, interval_gyro = imu_loader.query_range(
            interval_start_ns, int(frame_ns)
        )
        if interval_time.numel() < 2:
            raise RuntimeError(
                f"Insufficient IMU coverage for [{interval_start_ns}, {int(frame_ns)}]"
            )

        preint = preintegrate_imu(
            time_ns=interval_time,
            acc=interval_acc,
            gyro=interval_gyro,
            R0_world=R,
            gravity=-abs(float(calibration.gravity)),
            sigma_acc=calibration.sigma_acc,
            sigma_gyro=calibration.sigma_gyro,
            sigma_acc_w=calibration.sigma_acc_w,
            sigma_gyro_w=calibration.sigma_gyro_w,
            acc_bias=static_result.acc_bias,
            gyro_bias=static_result.gyro_bias,
            gravity_handling="preintegration",
        )

        R_before = R.matrix().reshape(3, 3).float()
        p_imu = (
            torch.as_tensor(p_imu, dtype=torch.float32)
            + v_imu * float(preint.dt_total)
            + R_before @ preint.delta_p.reshape(3).float()
        ).cpu().numpy()
        v_imu = propagate_imu_velocity_world(
            velocity_world=v_imu,
            delta_v_body=preint.delta_v,
            R_body_to_world=R_before,
            gravity_world=gravity_world,
            dt_total=preint.dt_total,
            gravity_handling="preintegration",
        ).float()
        R = integrate_gyro_attitude_world(
            R,
            interval_time,
            interval_gyro - static_result.gyro_bias.to(interval_gyro),
        ).double()

        R_matrix = R.matrix().reshape(3, 3).cpu().numpy()
        positions_camera[index] = p_imu - R_matrix @ camera_to_imu
        velocities_imu[index] = v_imu.cpu().numpy()
        rotations[index] = R_matrix
        covariance_trace[index] = float(preint.cov.diagonal().sum().item())

    static_diag = {
        "stationary": bool(static_result.stationary),
        "duration_s": float(static_result.duration_s),
        "sample_count": int(static_result.sample_count),
        "acc_mean": static_result.acc_mean.tolist(),
        "gyro_mean": static_result.gyro_mean.tolist(),
        "acc_std": static_result.acc_std.tolist(),
        "gyro_std": static_result.gyro_std.tolist(),
        "acc_bias": static_result.acc_bias.tolist(),
        "gyro_bias": static_result.gyro_bias.tolist(),
        "failure_reason": static_result.failure_reason,
    }
    return CalibratedImuOnlyResult(
        time_ns=camera_time_ns.copy(),
        camera_position_w=positions_camera,
        imu_velocity_w=velocities_imu,
        body_to_world=rotations,
        covariance_trace=covariance_trace,
        static_end_ns=required_end_ns,
        static_acc_bias=static_result.acc_bias.cpu().numpy(),
        static_gyro_bias=static_result.gyro_bias.cpu().numpy(),
        static_diagnostics=static_diag,
    )


def run_scene(
    scene: str,
    scene_root: Path,
    output_root: Path,
    *,
    seq_to: int | None = None,
) -> tuple[dict[str, object], pd.DataFrame]:
    from Scripts.run_clear_circle_imu_only_mechanization import (
        convert_camera_state_to_imu_initial,
        evaluate_joined,
        load_camera_to_imu_translation_nwu,
        load_ref_pose,
        matrix_to_quat_xyzw,
    )

    metadata = json.loads((scene_root / "metadata.json").read_text(encoding="utf-8"))
    ref = load_ref_pose(scene_root)
    if seq_to is not None:
        ref = ref.iloc[: int(seq_to)].copy()
    if ref.empty:
        raise ValueError(f"No camera frames selected for {scene}")
    camera_times = ref["timestamp_ns"].to_numpy(np.int64)
    first = ref.iloc[0]
    initial_position = np.array([first["x"], first["y"], first["z"]], dtype=np.float64)
    initial_quaternion = torch.tensor(
        [first["qx"], first["qy"], first["qz"], first["qw"]], dtype=torch.float64
    )
    initial_rotation = pp.SO3(initial_quaternion.reshape(1, 4)).matrix().reshape(3, 3).numpy()

    result = calibrated_imu_only_nwu(
        camera_time_ns=camera_times,
        imu_loader=IMUCSVLoader(scene_root / "imu_data.csv"),
        initial_camera_position_w=initial_position,
        initial_body_to_world=initial_rotation,
        camera_to_imu_body=load_camera_to_imu_translation_nwu(scene_root),
        calibration=load_calibration_parameters(metadata),
    )
    quaternions = np.stack([matrix_to_quat_xyzw(R) for R in result.body_to_world], axis=0)
    trajectory_path = output_root / "trajectories" / f"{scene}_{METHOD}_poses.csv"
    trajectory_path.parent.mkdir(parents=True, exist_ok=True)
    trajectory = pd.DataFrame(
        {
            "timestamp_ns": result.time_ns,
            "tx": result.camera_position_w[:, 0],
            "ty": result.camera_position_w[:, 1],
            "tz": result.camera_position_w[:, 2],
            "qx": quaternions[:, 0],
            "qy": quaternions[:, 1],
            "qz": quaternions[:, 2],
            "qw": quaternions[:, 3],
            "vx_imu": result.imu_velocity_w[:, 0],
            "vy_imu": result.imu_velocity_w[:, 1],
            "vz_imu": result.imu_velocity_w[:, 2],
            "imu_cov_trace": result.covariance_trace,
        }
    )
    trajectory.to_csv(trajectory_path, index=False)

    joined = pd.DataFrame(
        {
            "timestamp_ns": result.time_ns,
            "tx_est": result.camera_position_w[:, 0],
            "ty_est": result.camera_position_w[:, 1],
            "tz_est": result.camera_position_w[:, 2],
            "qx_est": quaternions[:, 0],
            "qy_est": quaternions[:, 1],
            "qz_est": quaternions[:, 2],
            "qw_est": quaternions[:, 3],
            "tx_gt": ref["x"].to_numpy(float),
            "ty_gt": ref["y"].to_numpy(float),
            "tz_gt": ref["z"].to_numpy(float),
            "qx_gt": ref["qx"].to_numpy(float),
            "qy_gt": ref["qy"].to_numpy(float),
            "qz_gt": ref["qz"].to_numpy(float),
            "qw_gt": ref["qw"].to_numpy(float),
        }
    )
    joined["err_m"] = np.linalg.norm(
        joined[["tx_est", "ty_est", "tz_est"]].to_numpy(float)
        - joined[["tx_gt", "ty_gt", "tz_gt"]].to_numpy(float),
        axis=1,
    )
    frame_metrics = output_root / "frame_metrics" / f"{scene}_{METHOD}_frame_metrics.csv"
    frame_metrics.parent.mkdir(parents=True, exist_ok=True)
    joined.to_csv(frame_metrics, index=False)

    diagnostic_path = output_root / "diagnostics" / f"{scene}_static_initialization.json"
    diagnostic_path.parent.mkdir(parents=True, exist_ok=True)
    diagnostic_path.write_text(
        json.dumps(result.static_diagnostics, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    label = f"{scene} / {METHOD}"
    summary = evaluate_joined(label, scene, METHOD, joined, str(trajectory_path))
    summary.update(
        {
            "static_end_ns": int(result.static_end_ns),
            "static_acc_bias_x": float(result.static_acc_bias[0]),
            "static_acc_bias_y": float(result.static_acc_bias[1]),
            "static_acc_bias_z": float(result.static_acc_bias[2]),
            "static_gyro_bias_x": float(result.static_gyro_bias[0]),
            "static_gyro_bias_y": float(result.static_gyro_bias[1]),
            "static_gyro_bias_z": float(result.static_gyro_bias[2]),
        }
    )
    return summary, joined


def _row_count(path: Path) -> int:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return max(sum(1 for _ in stream) - 1, 0)


def _write_manifest(output_root: Path, batch_root: Path, scenes: tuple[str, ...]) -> None:
    fields = ["trial", "scene", "variant", "scene_root", "result_dir", "created_at"]
    with (output_root / "run_manifest.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for scene in scenes:
            writer.writerow(
                {
                    "trial": 1,
                    "scene": scene,
                    "variant": METHOD,
                    "scene_root": batch_root / scene,
                    "result_dir": output_root,
                    "created_at": datetime.now().isoformat(timespec="seconds"),
                }
            )
    (output_root / "progress.csv").write_text(
        "trial,scene,variant,status,return_code,runtime_s,result_dir\n",
        encoding="utf-8",
    )


def _append_progress(output_root: Path, scene: str, status: str, runtime_s: str = "") -> None:
    with (output_root / "progress.csv").open("a", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow([1, scene, METHOD, status, 0 if status == "ok" else "", runtime_s, output_root])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-root", type=Path, default=DEFAULT_BATCH_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--dashboard-port", type=int, default=8765)
    parser.add_argument("--seq-to", type=int, default=None)
    parser.add_argument("--no-dashboard", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--scenes", nargs="*", choices=SCENES)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    scenes = tuple(args.scenes) if args.scenes else SCENES
    args.output_root.mkdir(parents=True, exist_ok=True)
    _write_manifest(args.output_root, args.batch_root, scenes)
    if not args.no_dashboard:
        from Scripts.run_visual_factor_cache_batch import switch_dashboard

        switch_dashboard(args.output_root, DEFAULT_LOG, port=int(args.dashboard_port))

    existing_summary_path = args.output_root / "imu_only_summary.csv"
    existing = {}
    if existing_summary_path.exists():
        existing_df = pd.read_csv(existing_summary_path)
        existing = {str(row["scene"]): row.to_dict() for _, row in existing_df.iterrows()}

    summaries: list[dict[str, object]] = []
    for index, scene in enumerate(scenes, start=1):
        scene_root = args.batch_root / scene
        trajectory = args.output_root / "trajectories" / f"{scene}_{METHOD}_poses.csv"
        total_expected = _row_count(scene_root / "ref_pose.csv")
        expected = (
            total_expected
            if args.seq_to is None
            else min(total_expected, int(args.seq_to))
        )
        if (
            not args.force
            and scene in existing
            and trajectory.exists()
            and _row_count(trajectory) == expected
        ):
            print(f"[{index}/{len(scenes)}] reused: {scene}", flush=True)
            summaries.append(existing[scene])
            _append_progress(args.output_root, scene, "ok", "reused")
            continue

        print(f"[{index}/{len(scenes)}] calibrated IMU-only: {scene}", flush=True)
        _append_progress(args.output_root, scene, "running")
        started = time.monotonic()
        summary, _ = run_scene(
            scene,
            scene_root,
            args.output_root,
            seq_to=args.seq_to,
        )
        summaries.append(summary)
        _append_progress(args.output_root, scene, "ok", f"{time.monotonic() - started:.1f}")

    pd.DataFrame(summaries).to_csv(existing_summary_path, index=False)
    (args.output_root / "trajectory_contract.txt").write_text(
        "method=imu_only_staticinit_calibrated\n"
        "world=NWU\n"
        "trajectory_point=CameraLeftSocket\n"
        "initial_gauge=first ref_pose only\n"
        "visual_residuals=disabled\n"
        "bias_after_static_initialization=held_constant\n"
        f"seq_to={args.seq_to if args.seq_to is not None else 'full'}\n",
        encoding="utf-8",
    )
    print(f"summary: {existing_summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
