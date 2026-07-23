from types import SimpleNamespace

import pypose as pp
import pytest
import torch

from DataLoader.Dataset.GeneralStereoIMU import _continuous_imu_noise_density
from Module.IMUPreintegration import _raw_sample_periods_s
from Odometry.MACVO import MACVO
from Scripts.run_realtime_t2 import validate_static_initialization_options
from Utility.IMUKinematics import (
    estimate_static_imu_initialization,
    evaluate_adaptive_static_imu_initialization,
)


def _options(
    mode: str,
    duration: float | None = None,
    state_policy: str = "estimated",
) -> SimpleNamespace:
    return SimpleNamespace(
        static_init_mode=mode,
        static_init_state_policy=state_policy,
        static_init_duration_s=duration,
        static_init_min_duration_s=1.0,
        static_init_max_duration_s=8.0,
        static_init_window_s=0.25,
        static_init_stable_hold_s=0.75,
    )


def _static_imu(duration_s: float = 3.2, rate_hz: int = 100):
    torch.manual_seed(31)
    count = int(round(duration_s * rate_hz)) + 1
    stamps = torch.arange(count, dtype=torch.long) * int(1e9 / rate_hz)
    gyro_bias = torch.tensor([0.0015, -0.0004, 0.0007])
    acc_bias = torch.tensor([0.025, -0.018, 0.035])
    gyro = gyro_bias + 0.018 * torch.randn(count, 3)
    acc = torch.tensor([0.0, 0.0, 9.8]) + acc_bias + 0.14 * torch.randn(count, 3)
    return stamps, acc, gyro


def test_continuous_density_metadata_preserves_scalar_and_axis_values():
    assert _continuous_imu_noise_density({"NoiseAcc": 0.014}, "NoiseAcc") == 0.014
    assert _continuous_imu_noise_density(
        {"NoiseGyroXYZ": [0.001, 0.002, 0.003]}, "NoiseGyro"
    ) == (0.001, 0.002, 0.003)
    with pytest.raises(ValueError, match="finite non-negative"):
        _continuous_imu_noise_density({"AccWalk": [0.1, -0.2, 0.3]}, "AccWalk")


def test_timestamp_based_density_discretization_matches_regular_rate():
    base = 1_720_000_000_000_000_000
    stamps_100_hz = torch.tensor(
        [base, base + 10_000_000, base + 20_000_000], dtype=torch.long
    )
    density = torch.tensor([0.01, 0.02, 0.03], dtype=torch.float64)
    sample_sigma = MACVO._imu_sample_sigma(
        density, stamps_100_hz, floor=0.0, multiplier=1.0
    )
    assert torch.allclose(sample_sigma, density * 10.0, atol=1e-12, rtol=1e-12)

    _, periods_s = _raw_sample_periods_s(
        stamps_100_hz, device=torch.device("cpu"), dtype=torch.float64
    )
    expected_variance = density.square().reshape(1, 3) / periods_s.reshape(-1, 1)
    assert torch.allclose(
        expected_variance,
        density.square().reshape(1, 3).repeat(3, 1) * 100.0,
        atol=1e-12,
        rtol=1e-12,
    )


def test_irregular_timestamps_define_per_sample_support_periods():
    base = 1_720_000_000_000_000_000
    stamps = torch.tensor(
        [base, base + 10_000_000, base + 30_000_000], dtype=torch.long
    )
    _, periods_s = _raw_sample_periods_s(
        stamps, device=torch.device("cpu"), dtype=torch.float64
    )
    assert torch.allclose(
        periods_s,
        torch.tensor([0.01, 0.015, 0.02], dtype=torch.float64),
        atol=1e-15,
        rtol=1e-15,
    )


def test_static_mode_contract_requires_explicit_fixed_duration():
    with pytest.raises(ValueError, match="requires --static-init-duration-s"):
        validate_static_initialization_options(_options("fixed"))
    with pytest.raises(ValueError, match="must be > 0"):
        validate_static_initialization_options(_options("fixed", 0.0))
    validate_static_initialization_options(_options("fixed", 3.0))

    validate_static_initialization_options(_options("adaptive"))
    validate_static_initialization_options(_options("off"))
    with pytest.raises(ValueError, match="valid only"):
        validate_static_initialization_options(_options("off", 3.0))
    validate_static_initialization_options(_options("adaptive", state_policy="zero"))
    with pytest.raises(ValueError, match="requires fixed or adaptive"):
        validate_static_initialization_options(_options("off", state_policy="zero"))


def test_adaptive_mode_finishes_inside_three_second_static_prefix():
    stamps, acc, gyro = _static_imu()
    decision = None
    for count in range(4, stamps.numel() + 1, 3):
        decision = evaluate_adaptive_static_imu_initialization(
            stamps[:count],
            acc[:count],
            gyro[:count],
            pp.identity_SO3(),
            -9.8,
            min_duration_s=1.0,
            max_duration_s=8.0,
            window_s=0.25,
            stable_hold_s=0.75,
            target_gyro_bias_sem=1.2e-3,
            target_gravity_direction_sem_rad=3.0e-3,
            gyro_mean_norm_max=0.03,
            gyro_std_max=0.1,
            acc_norm_error_max=0.6,
            acc_std_max=0.8,
        )
        if decision.ready:
            break

    assert decision is not None and decision.ready
    assert not decision.timed_out
    assert 1.0 <= decision.initialization.duration_s <= 3.0
    assert decision.recent_stationary
    assert decision.gyro_bias_sem_max <= 1.2e-3
    assert decision.gravity_direction_sem_rad <= 3.0e-3
    assert torch.isfinite(decision.initialization.acc_bias).all()
    assert torch.isfinite(decision.initialization.gyro_bias).all()

    fixed_count = int(torch.searchsorted(stamps, torch.tensor(3_000_000_000)).item()) + 1
    fixed = estimate_static_imu_initialization(
        stamps[:fixed_count],
        acc[:fixed_count],
        gyro[:fixed_count],
        pp.identity_SO3(),
        -9.8,
        min_duration_s=3.0,
        gyro_mean_norm_max=0.03,
        gyro_std_max=0.1,
        acc_norm_error_max=0.6,
        acc_std_max=0.8,
    )
    assert fixed.stationary
    assert float(
        (decision.initialization.gyro_bias - fixed.gyro_bias).norm().item()
    ) < 5.0e-3
    assert float(
        (decision.initialization.body_to_world.Inv() @ fixed.body_to_world)
        .Log()
        .tensor()
        .norm()
        .item()
    ) < 1.0e-2


def test_adaptive_mode_times_out_on_nonstationary_startup():
    stamps, acc, gyro = _static_imu(duration_s=2.1)
    gyro[:, 2] += 0.2
    decision = evaluate_adaptive_static_imu_initialization(
        stamps,
        acc,
        gyro,
        pp.identity_SO3(),
        -9.8,
        min_duration_s=1.0,
        max_duration_s=2.0,
        window_s=0.25,
        stable_hold_s=0.75,
        target_gyro_bias_sem=1.2e-3,
        target_gravity_direction_sem_rad=3.0e-3,
        gyro_mean_norm_max=0.03,
        gyro_std_max=0.1,
        acc_norm_error_max=0.6,
        acc_std_max=0.8,
    )

    assert not decision.ready
    assert decision.timed_out
    assert "gyro mean indicates motion" in decision.failure_reason


def _macvo_static_state(
    mode: str,
    duration_s: float | None = None,
    state_policy: str = "estimated",
) -> MACVO:
    system = MACVO.__new__(MACVO)
    system.imu_static_initialization_mode = mode
    system.imu_static_initialization_state_policy = state_policy
    system.imu_static_initialization_enable = mode != "off"
    system.imu_static_initialization_duration_s = duration_s
    system.imu_static_adaptive_min_duration_s = 1.0
    system.imu_static_adaptive_max_duration_s = 8.0
    system.imu_static_adaptive_window_s = 0.25
    system.imu_static_adaptive_stable_hold_s = 0.75
    system.imu_static_adaptive_target_gyro_bias_sem = 1.2e-3
    system.imu_static_adaptive_target_gravity_direction_sem_rad = 3.0e-3
    system.imu_sigma_acc = 0.014
    system.imu_sigma_gyro = 0.0018
    system.imu_static_sigma_multiplier = 5.0
    system.imu_static_gyro_mean_norm_max = 0.03
    system.imu_static_acc_norm_error_max = 0.6
    system.pipeline_trace_path = None
    system.prev_keyframe = None
    system._imu_static_initialized = mode == "off"
    system._imu_static_time_chunks = []
    system._imu_static_acc_chunks = []
    system._imu_static_gyro_chunks = []
    system._imu_static_last_time_ns = None
    system._imu_static_initial_rotation = None
    system._imu_static_anchor_pose = None
    system._imu_static_zupt_active = False
    system._imu_static_init_diag = None
    system._imu_vel_w = torch.ones(3)
    system._imu_acc_bias = torch.zeros(3)
    system._imu_gyro_bias = torch.zeros(3)
    system._imu_attitude_world = None
    system._imu_last_frame_time_ns = None
    system._imu_attitude_last_time_ns = None
    return system


@pytest.mark.parametrize(
    ("mode", "duration_s"),
    (("fixed", 3.0), ("adaptive", None)),
)
def test_macvo_fixed_and_adaptive_modes_write_the_same_state_contract(
    mode: str,
    duration_s: float | None,
):
    stamps, acc, gyro = _static_imu()
    system = _macvo_static_state(mode, duration_s)
    frame = SimpleNamespace(
        imu_calib_acc_sigma=0.014,
        imu_calib_gyro_sigma=0.0018,
    )

    anchored = False
    for start in range(0, stamps.numel(), 3):
        stop = min(start + 3, stamps.numel())
        anchored = system._try_static_imu_initialization(
            frame, stamps[start:stop], acc[start:stop], gyro[start:stop], -9.8
        )
        if system._imu_static_initialized:
            break

    assert anchored
    assert system._imu_static_initialized
    assert system._imu_static_init_diag["mode"] == mode
    assert system._imu_static_init_diag["status"] == "complete"
    assert system._imu_static_init_diag["state_policy"] == "estimated"
    assert system._imu_static_init_diag["estimate_applied"]
    assert torch.equal(system._imu_vel_w, torch.zeros(3))
    assert torch.isfinite(system._imu_acc_bias).all()
    assert torch.isfinite(system._imu_gyro_bias).all()
    assert system._imu_attitude_world is not None
    assert system._imu_static_anchor_pose is not None


def test_zero_state_policy_discards_estimate_at_same_adaptive_boundary():
    stamps, acc, gyro = _static_imu()
    system = _macvo_static_state("adaptive", state_policy="zero")
    frame = SimpleNamespace(
        imu_calib_acc_sigma=0.014,
        imu_calib_gyro_sigma=0.0018,
    )

    anchored = False
    for start in range(0, stamps.numel(), 3):
        stop = min(start + 3, stamps.numel())
        anchored = system._try_static_imu_initialization(
            frame, stamps[start:stop], acc[start:stop], gyro[start:stop], -9.8
        )
        if system._imu_static_initialized:
            break

    assert anchored
    assert system._imu_static_initialized
    assert system._imu_static_init_diag["state_policy"] == "zero"
    assert not system._imu_static_init_diag["estimate_applied"]
    assert torch.linalg.vector_norm(
        torch.tensor(system._imu_static_init_diag["estimated_acc_bias"])
    ) > 1.0e-3
    assert torch.linalg.vector_norm(
        torch.tensor(system._imu_static_init_diag["estimated_gyro_bias"])
    ) > 1.0e-4
    assert torch.equal(system._imu_vel_w, torch.zeros(3))
    assert torch.equal(system._imu_acc_bias, torch.zeros(3))
    assert torch.equal(system._imu_gyro_bias, torch.zeros(3))
    assert torch.equal(
        system._imu_attitude_world.Log().tensor(), torch.zeros(3)
    )
    assert system._imu_static_anchor_pose is not None


def test_macvo_off_mode_skips_static_collection():
    stamps, acc, gyro = _static_imu(duration_s=0.1)
    system = _macvo_static_state("off")
    frame = SimpleNamespace()

    anchored = system._try_static_imu_initialization(
        frame, stamps, acc, gyro, -9.8
    )

    assert not anchored
    assert system._imu_static_initialized
    assert not system._imu_static_zupt_active
    assert system._imu_static_time_chunks == []
