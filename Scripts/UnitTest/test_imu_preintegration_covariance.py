import importlib.util
from pathlib import Path

import torch
import pytest

from Utility.IMUKinematics import covariance_information_matrix


_SPEC = importlib.util.spec_from_file_location(
    "imu_preintegration_under_test",
    Path("Module/IMUPreintegration.py"),
)
_MODULE = importlib.util.module_from_spec(_SPEC)
assert _SPEC is not None and _SPEC.loader is not None
_SPEC.loader.exec_module(_MODULE)
preintegrate_imu = _MODULE.preintegrate_imu


def test_gyro_noise_density_accumulates_as_angle_variance_over_dt():
    dt_s = 0.01
    sigma_gyro = 0.1
    result = preintegrate_imu(
        time_ns=torch.tensor([0, int(dt_s * 1e9)], dtype=torch.long),
        acc=torch.zeros((2, 3), dtype=torch.float32),
        gyro=torch.zeros((2, 3), dtype=torch.float32),
        R0_world=None,
        gravity=0.0,
        sigma_acc=0.0,
        sigma_gyro=sigma_gyro,
        sigma_acc_w=0.0,
        sigma_gyro_w=0.0,
    )

    rot_var = result.cov[6:9, 6:9].diagonal()
    expected = torch.full((3,), sigma_gyro**2 * dt_s, dtype=torch.float32)
    assert torch.allclose(rot_var, expected, rtol=0.05, atol=1e-8)


def test_preintegration_keeps_zero_noise_covariance_raw_without_fixed_floor():
    result = preintegrate_imu(
        time_ns=torch.tensor([0, 10_000_000], dtype=torch.long),
        acc=torch.zeros((2, 3), dtype=torch.float32),
        gyro=torch.zeros((2, 3), dtype=torch.float32),
        R0_world=None,
        gravity=0.0,
        sigma_acc=0.0,
        sigma_gyro=0.0,
        sigma_acc_w=0.0,
        sigma_gyro_w=0.0,
    )

    assert torch.count_nonzero(result.cov) == 0
    assert torch.count_nonzero(result.bias_rw_cov) == 0


def test_information_regularization_preserves_mixed_physical_scales():
    covariance = torch.diag(torch.tensor([1e-12, 1e-6, 1e-2], dtype=torch.float64))

    information = covariance_information_matrix(covariance)

    expected = torch.diag(torch.tensor([1e12, 1e6, 1e2], dtype=torch.float64))
    assert torch.allclose(information, expected, rtol=1e-6, atol=1e-6)


def test_information_regularization_handles_exact_singular_covariance():
    covariance = torch.diag(torch.tensor([0.0, 1e-6, 0.0], dtype=torch.float64))

    information = covariance_information_matrix(covariance)

    assert torch.isfinite(information).all()
    assert information[0, 0] == 0.0
    assert float(information[1, 1].item()) == pytest.approx(1e6, rel=2e-9)
    assert information[2, 2] == 0.0


def test_absolute_covariance_floor_reduces_overconfident_information():
    covariance = torch.diag(torch.tensor([1e-10, 1e-8, 1e-6], dtype=torch.float64))

    information = covariance_information_matrix(
        covariance,
        absolute_diagonal_floor=1e-8,
    )

    expected = torch.diag(1.0 / (covariance.diagonal() + 1e-8))
    assert torch.allclose(information, expected, rtol=1e-6, atol=1e-6)
