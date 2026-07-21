import importlib.util
from pathlib import Path

import pypose as pp
import pytest
import torch

from Utility.IMUKinematics import (
    estimate_static_imu_initialization,
    integrate_gyro_attitude_world,
)


_PREINT_SPEC = importlib.util.spec_from_file_location(
    "imu_preintegration_calibration_test",
    Path("Module/IMUPreintegration.py"),
)
_PREINT_MODULE = importlib.util.module_from_spec(_PREINT_SPEC)
assert _PREINT_SPEC is not None and _PREINT_SPEC.loader is not None
_PREINT_SPEC.loader.exec_module(_PREINT_MODULE)
preintegrate_imu = _PREINT_MODULE.preintegrate_imu


def _static_interval(duration_s: float = 3.0, rate_hz: int = 100):
    sample_count = int(round(duration_s * rate_hz)) + 1
    time_ns = torch.arange(sample_count, dtype=torch.int64) * int(1e9 / rate_hz)
    return time_ns


def test_static_initialization_recovers_gyro_bias_and_gravity_consistent_acc_state():
    time_ns = _static_interval()
    acc_bias = torch.tensor([0.12, -0.08, 0.03])
    gyro_bias = torch.tensor([0.01, -0.006, 0.004])
    acc = torch.tensor([0.0, 0.0, -9.8]).repeat(time_ns.numel(), 1) + acc_bias
    gyro = gyro_bias.repeat(time_ns.numel(), 1)

    result = estimate_static_imu_initialization(
        time_ns=time_ns,
        acc_body=acc,
        gyro_body=gyro,
        initial_body_to_world=pp.identity_SO3(dtype=torch.float32),
        gravity=9.8,
        gyro_mean_norm_max=0.03,
        gyro_std_max=0.02,
        acc_norm_error_max=0.5,
        acc_std_max=0.2,
    )

    assert result.stationary
    assert result.duration_s == pytest.approx(3.0)
    assert torch.allclose(result.gyro_bias, gyro_bias, atol=1e-6)
    gravity_world = torch.tensor([0.0, 0.0, 9.8])
    expected_static = -result.body_to_world.Inv().Act(gravity_world).reshape(3)
    assert torch.allclose(expected_static + result.acc_bias, result.acc_mean, atol=1e-6)
    # Static accelerometer data alone cannot distinguish horizontal bias from
    # a small roll/pitch error. The observable gravity-axis component remains.
    assert abs(float(result.acc_bias[2].item()) - float(acc_bias[2].item())) < 2e-3


def test_static_initialization_rejects_rotating_interval():
    time_ns = _static_interval()
    acc = torch.tensor([0.0, 0.0, -9.8]).repeat(time_ns.numel(), 1)
    gyro = torch.tensor([0.0, 0.0, 0.2]).repeat(time_ns.numel(), 1)

    result = estimate_static_imu_initialization(
        time_ns=time_ns,
        acc_body=acc,
        gyro_body=gyro,
        initial_body_to_world=pp.identity_SO3(dtype=torch.float32),
        gravity=9.8,
    )

    assert not result.stationary
    assert "gyro mean" in result.failure_reason


def test_bias_corrected_attitude_thread_does_not_integrate_constant_gyro_bias():
    time_ns = _static_interval(duration_s=1.0)
    gyro_bias = torch.tensor([0.0, 0.0, 0.02])
    measured = gyro_bias.repeat(time_ns.numel(), 1)

    raw_attitude = integrate_gyro_attitude_world(
        pp.identity_SO3(dtype=torch.float32), time_ns, measured
    )
    corrected_attitude = integrate_gyro_attitude_world(
        pp.identity_SO3(dtype=torch.float32), time_ns, measured - gyro_bias
    )

    assert float(raw_attitude.Log().tensor().norm().item()) == pytest.approx(0.02, rel=1e-4)
    assert float(corrected_attitude.Log().tensor().norm().item()) < 1e-8


def test_preintegration_preserves_three_axis_noise_covariance():
    time_ns = _static_interval(duration_s=0.1)
    acc = torch.tensor([0.0, 0.0, -9.8]).repeat(time_ns.numel(), 1)
    gyro = torch.zeros_like(acc)

    result = preintegrate_imu(
        time_ns=time_ns,
        acc=acc,
        gyro=gyro,
        R0_world=pp.identity_SO3(dtype=torch.float32),
        gravity=9.8,
        sigma_acc=(0.01, 0.02, 0.04),
        sigma_gyro=(0.001, 0.002, 0.004),
    )

    velocity_variance = result.cov[3:6, 3:6].diagonal()
    rotation_variance = result.cov[6:9, 6:9].diagonal()
    assert velocity_variance[0] < velocity_variance[1] < velocity_variance[2]
    assert rotation_variance[0] < rotation_variance[1] < rotation_variance[2]
