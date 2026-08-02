import pytest
import torch

from Utility.NearZeroVelocityDetector import (
    TurningNearZeroVelocityDetector,
    TurningNearZeroVelocityDetectorV2,
    ZeroTranslationKinematicEvidence,
    zero_translation_kinematic_evidence,
)


def _step(
    detector: TurningNearZeroVelocityDetector,
    *,
    speed: float = 0.08,
    imu_rate: float = 0.55,
    visual_rate: float = 0.54,
):
    return detector.update(
        dt_s=1.0 / 30.0,
        estimated_speed_m_s=speed,
        imu_angular_rate_rad_s=imu_rate,
        visual_angular_rate_rad_s=visual_rate,
        visual_body_speed_m_s=0.5,
    )


def test_detector_enters_only_after_continuous_hold():
    detector = TurningNearZeroVelocityDetector(enter_hold_s=0.20)
    decisions = [_step(detector) for _ in range(6)]

    assert not any(decision.active for decision in decisions[:5])
    assert decisions[5].active
    assert decisions[5].entered


def test_fast_turn_is_not_a_zero_velocity_candidate():
    detector = TurningNearZeroVelocityDetector()
    decisions = [_step(detector, speed=0.35) for _ in range(20)]

    assert not any(decision.candidate for decision in decisions)
    assert not any(decision.active for decision in decisions)


def test_rotation_disagreement_blocks_activation():
    detector = TurningNearZeroVelocityDetector()
    decisions = [
        _step(detector, imu_rate=0.55, visual_rate=0.05)
        for _ in range(20)
    ]

    assert not any(decision.active for decision in decisions)
    assert "rotation_disagreement" in decisions[-1].reason


def test_detector_uses_release_hysteresis():
    detector = TurningNearZeroVelocityDetector(
        enter_hold_s=0.10,
        release_hold_s=0.10,
    )
    for _ in range(3):
        active = _step(detector)
    assert active.active

    first_failure = _step(detector, imu_rate=0.0, visual_rate=0.0)
    second_failure = _step(detector, imu_rate=0.0, visual_rate=0.0)
    third_failure = _step(detector, imu_rate=0.0, visual_rate=0.0)

    assert first_failure.active
    assert second_failure.active
    assert not third_failure.active
    assert third_failure.exited


def test_detector_rejects_nonfinite_input():
    detector = TurningNearZeroVelocityDetector()
    with pytest.raises(ValueError, match="NaN/Inf"):
        detector.update(
            dt_s=1.0 / 30.0,
            estimated_speed_m_s=float("nan"),
            imu_angular_rate_rad_s=0.5,
            visual_angular_rate_rad_s=0.5,
            visual_body_speed_m_s=0.0,
        )


def _v2_evidence(nis_per_dof: float = 0.5):
    return ZeroTranslationKinematicEvidence(
        nis=6.0 * nis_per_dof,
        degrees_of_freedom=6,
        nis_per_dof=nis_per_dof,
        position_residual_norm_m=1.0e-5,
        velocity_residual_norm_m_s=1.0e-3,
        corrected_imu_rotvec_body=torch.tensor(
            [0.0, 0.0, 0.55 / 30.0], dtype=torch.float64
        ),
    )


def _v2_step(
    detector: TurningNearZeroVelocityDetectorV2,
    *,
    estimated_speed: float = 10.0,
    imu_rate: float = 0.55,
    visual_rate: float = 0.54,
    visual_sign: float = 1.0,
    nis_per_dof: float = 0.5,
):
    return detector.update(
        dt_s=1.0 / 30.0,
        estimated_speed_m_s=estimated_speed,
        imu_rotvec_body=torch.tensor(
            [0.0, 0.0, imu_rate / 30.0], dtype=torch.float64
        ),
        visual_rotvec_body=torch.tensor(
            [0.0, 0.0, visual_sign * visual_rate / 30.0],
            dtype=torch.float64,
        ),
        visual_body_speed_m_s=0.5,
        zero_translation_evidence=_v2_evidence(nis_per_dof),
    )


def test_zero_translation_evidence_closes_at_static_state():
    dt_s = 1.0 / 30.0
    gravity = torch.tensor([0.0, 0.0, 9.81], dtype=torch.float64)
    evidence = zero_translation_kinematic_evidence(
        pose_WB=torch.tensor(
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            dtype=torch.float64,
        ),
        acc_bias=torch.zeros(3, dtype=torch.float64),
        gyro_bias=torch.zeros(3, dtype=torch.float64),
        delta_rotation=torch.zeros(3, dtype=torch.float64),
        delta_velocity=-gravity * dt_s,
        delta_position=-0.5 * gravity * dt_s * dt_s,
        covariance_pvr=torch.eye(9, dtype=torch.float64) * 1.0e-4,
        dt_s=dt_s,
        bias_jacobian=torch.zeros((9, 6), dtype=torch.float64),
        linearized_acc_bias=torch.zeros(3, dtype=torch.float64),
        linearized_gyro_bias=torch.zeros(3, dtype=torch.float64),
        gravity_world=gravity,
    )

    assert evidence.degrees_of_freedom == 6
    assert evidence.nis == pytest.approx(0.0, abs=1.0e-12)
    assert evidence.position_residual_norm_m == pytest.approx(
        0.0, abs=1.0e-12
    )
    assert evidence.velocity_residual_norm_m_s == pytest.approx(
        0.0, abs=1.0e-12
    )


def test_v2_does_not_gate_on_drifted_estimated_speed():
    detector = TurningNearZeroVelocityDetectorV2(enter_hold_s=0.20)
    decisions = [_v2_step(detector, estimated_speed=10.0) for _ in range(6)]

    assert decisions[-1].active
    assert decisions[-1].entered


def test_v2_rejects_continuous_circle_angular_rate():
    detector = TurningNearZeroVelocityDetectorV2()
    decisions = [
        _v2_step(detector, imu_rate=0.21, visual_rate=0.20)
        for _ in range(20)
    ]

    assert not any(decision.candidate for decision in decisions)
    assert "imu_rotation" in decisions[-1].reason


def test_v2_rejects_large_zero_translation_nis():
    detector = TurningNearZeroVelocityDetectorV2()
    decisions = [
        _v2_step(detector, nis_per_dof=4.0) for _ in range(20)
    ]

    assert not any(decision.active for decision in decisions)
    assert "zero_translation_nis" in decisions[-1].reason


def test_v2_rejects_opposite_rotation_axis():
    detector = TurningNearZeroVelocityDetectorV2()
    decisions = [
        _v2_step(detector, visual_sign=-1.0) for _ in range(20)
    ]

    assert not any(decision.active for decision in decisions)
    assert "rotation_axis" in decisions[-1].reason
