from pathlib import Path

import numpy as np
import pandas as pd
import torch

from Utility.IMUCSV import IMUCSVLoader
from Scripts.run_static63_calibrated_imu_only import (
    CalibrationParameters,
    _collect_static_samples_like_fusion,
    calibrated_imu_only_nwu,
)


def _write_imu(path: Path, time_ns: np.ndarray, acc: np.ndarray, gyro: np.ndarray) -> None:
    pd.DataFrame(
        {
            "timestamp": time_ns,
            "ang_vel_x": gyro[:, 0],
            "ang_vel_y": gyro[:, 1],
            "ang_vel_z": gyro[:, 2],
            "lin_acc_x": acc[:, 0],
            "lin_acc_y": acc[:, 1],
            "lin_acc_z": acc[:, 2],
        }
    ).to_csv(path, index=False)


def _calibration() -> CalibrationParameters:
    return CalibrationParameters(
        gravity=9.8,
        measurement_rate_hz=100.0,
        sigma_acc=0.014,
        sigma_gyro=0.0018,
        sigma_acc_w=0.00038,
        sigma_gyro_w=0.000036,
    )


def test_static_calibration_removes_stationary_specific_force_and_holds_zupt(
    tmp_path: Path,
):
    time_ns = np.arange(0, 5_000_000_001, 10_000_000, dtype=np.int64)
    acc_bias = np.array([0.03, -0.02, 0.01])
    gyro_bias = np.array([0.002, -0.001, 0.003])
    acc = np.tile(np.array([0.0, 0.0, 9.8]) + acc_bias, (len(time_ns), 1))
    gyro = np.tile(gyro_bias, (len(time_ns), 1))
    imu_path = tmp_path / "imu_data.csv"
    _write_imu(imu_path, time_ns, acc, gyro)

    camera_times = np.arange(0, 5_000_000_001, 100_000_000, dtype=np.int64)
    result = calibrated_imu_only_nwu(
        camera_time_ns=camera_times,
        imu_loader=IMUCSVLoader(imu_path),
        initial_camera_position_w=np.zeros(3),
        initial_body_to_world=np.eye(3),
        camera_to_imu_body=np.zeros(3),
        calibration=_calibration(),
    )

    assert np.allclose(result.static_gyro_bias, gyro_bias, atol=1e-6)
    first_motion_index = int(np.flatnonzero(camera_times > 3_000_000_000)[0])
    corrected_acc_world = (
        result.body_to_world[first_motion_index]
        @ (acc[0] - result.static_acc_bias)
        + np.array([0.0, 0.0, -9.8])
    )
    assert np.allclose(corrected_acc_world, 0.0, atol=1e-5)
    assert np.allclose(result.camera_position_w, 0.0, atol=1e-5)
    assert np.allclose(result.imu_velocity_w, 0.0, atol=1e-5)


def test_post_static_motion_uses_frame_boundary_interpolation(tmp_path: Path):
    time_ns = np.arange(0, 5_000_000_001, 10_000_000, dtype=np.int64)
    acc = np.tile(np.array([0.0, 0.0, 9.8]), (len(time_ns), 1))
    acc[time_ns > 3_000_000_000, 0] = 1.0
    gyro = np.zeros_like(acc)
    imu_path = tmp_path / "imu_data.csv"
    _write_imu(imu_path, time_ns, acc, gyro)

    camera_times = np.arange(0, 5_000_000_001, 100_000_000, dtype=np.int64)
    result = calibrated_imu_only_nwu(
        camera_time_ns=camera_times,
        imu_loader=IMUCSVLoader(imu_path),
        initial_camera_position_w=np.zeros(3),
        initial_body_to_world=np.eye(3),
        camera_to_imu_body=np.zeros(3),
        calibration=_calibration(),
    )

    # Midpoint interpolation at the 3 s boundary gives nearly two seconds of
    # 1 m/s^2 acceleration after startup.
    assert np.isclose(result.imu_velocity_w[-1, 0], 1.995, atol=0.01)
    assert np.isclose(result.camera_position_w[-1, 0], 1.99, atol=0.03)
    assert np.allclose(result.camera_position_w[camera_times <= 3_000_000_000], 0.0)


def test_static_collection_matches_frame_by_frame_fusion_sampling(tmp_path: Path):
    time_ns = np.arange(0, 3_100_000_001, 10_000_000, dtype=np.int64)
    acc = np.tile(np.array([0.0, 0.0, 9.8]), (len(time_ns), 1))
    gyro = np.zeros_like(acc)
    imu_path = tmp_path / "imu_data.csv"
    _write_imu(imu_path, time_ns, acc, gyro)

    camera_times = np.arange(3_333_333, 3_100_000_000, 33_333_333, dtype=np.int64)
    required_end_ns = int(camera_times[0]) + 3_000_000_000
    static_time, _, _ = _collect_static_samples_like_fusion(
        camera_times,
        IMUCSVLoader(imu_path),
        required_end_ns,
    )

    assert int(static_time[0].item()) == int(camera_times[0])
    assert int(static_time[-1].item()) >= required_end_ns
    assert static_time.numel() > 301
    assert torch.unique(static_time).numel() == static_time.numel()
