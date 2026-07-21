import pytest

from Utility.IMUKinematics import (
    imu_bias_sigma_to_continuous_random_walk_density,
    imu_sigma_to_continuous_density,
)


def test_continuous_noise_density_is_kept_unchanged():
    assert imu_sigma_to_continuous_density(
        sigma_value=0.02,
        rate_hz=100.0,
        sigma_unit="continuous_noise_density",
    ) == pytest.approx(0.02)


def test_per_sample_standard_deviation_converts_to_density():
    assert imu_sigma_to_continuous_density(
        sigma_value=0.02,
        rate_hz=100.0,
        sigma_unit="per-sample standard deviation",
    ) == pytest.approx(0.002)


def test_per_sample_bias_random_walk_increment_converts_to_density():
    assert imu_bias_sigma_to_continuous_random_walk_density(
        sigma_value=0.02,
        rate_hz=100.0,
        sigma_unit="per-sample standard deviation",
    ) == pytest.approx(0.2)


def test_per_tick_bias_random_walk_increment_converts_to_density():
    assert imu_bias_sigma_to_continuous_random_walk_density(
        sigma_value=0.02,
        rate_hz=100.0,
        sigma_unit="per-tick bias random-walk increment standard deviation",
    ) == pytest.approx(0.2)


def test_bias_random_walk_density_is_kept_unchanged():
    assert imu_bias_sigma_to_continuous_random_walk_density(
        sigma_value=0.02,
        rate_hz=100.0,
        sigma_unit="continuous_random_walk_density",
    ) == pytest.approx(0.02)


def test_legacy_sqrt_rate_scaling_is_explicit():
    assert imu_sigma_to_continuous_density(
        sigma_value=0.02,
        rate_hz=100.0,
        sigma_unit="legacy_sqrt_rate_scaled",
    ) == pytest.approx(0.2)


def test_unknown_sigma_unit_fails_closed():
    with pytest.raises(ValueError):
        imu_sigma_to_continuous_density(
            sigma_value=0.02,
            rate_hz=100.0,
            sigma_unit="unclear unit",
        )
