from __future__ import annotations

import numpy as np

from Utility.OutputTrajectoryEKF import CausalPoseOutputEKF, OutputEKFNoise


def noise() -> OutputEKFNoise:
    return OutputEKFNoise(
        position_measurement_std=np.array([0.02, 0.02]),
        yaw_measurement_std=0.01,
        linear_acceleration_process_std=np.array([0.2, 0.2]),
        yaw_acceleration_process_std=0.1,
    )


def test_output_filter_reduces_stationary_measurement_variance():
    generator = np.random.default_rng(7)
    measurements = generator.normal(0.0, 0.02, size=(300, 2))
    output_filter = CausalPoseOutputEKF(measurements[0], 0.0, noise())
    filtered = []
    for index, measurement in enumerate(measurements):
        output_filter.step(None if index == 0 else 0.02, measurement, 0.0)
        filtered.append(output_filter.state[0:2].copy())
    filtered = np.asarray(filtered)
    assert np.var(filtered[50:], axis=0).max() < np.var(measurements[50:], axis=0).max()


def test_output_filter_handles_yaw_wrap_without_large_innovation():
    output_filter = CausalPoseOutputEKF(np.zeros(2), np.pi - 0.01, noise())
    diagnostics = output_filter.step(0.02, np.zeros(2), -np.pi + 0.01)
    assert abs(diagnostics.innovation[2]) < 0.03


def test_output_filter_does_not_mutate_measurements_and_keeps_covariance_psd():
    xy = np.array([1.0, -2.0])
    original = xy.copy()
    output_filter = CausalPoseOutputEKF(xy, 0.2, noise())
    output_filter.step(0.03, xy, 0.2)
    assert np.array_equal(xy, original)
    assert np.linalg.eigvalsh(output_filter.covariance).min() >= -1.0e-12
    assert np.isfinite(output_filter.state).all()
