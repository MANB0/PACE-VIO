import torch

from Utility.IMUKinematics import (
    compose_imu_translation_prior,
    resolve_translation_prior_mode,
    translation_prior_semantics,
)


def test_legacy_bool_false_resolves_to_damping_mode():
    assert resolve_translation_prior_mode(None, use_velocity=False) == "damping_delta_p"


def test_legacy_bool_true_resolves_to_imu_velocity_mode():
    assert resolve_translation_prior_mode(None, use_velocity=True) == "imu_velocity_composed"


def test_explicit_mode_overrides_legacy_bool():
    assert resolve_translation_prior_mode("visual_velocity_composed", use_velocity=False) == "visual_velocity_composed"


def test_translation_semantics_are_clear_for_each_mode():
    assert translation_prior_semantics("off") == "translation_prior_disabled"
    assert translation_prior_semantics("damping_delta_p") == "delta_p_motion_damping"
    assert translation_prior_semantics("visual_velocity_composed") == "visual_velocity_composed_pose_prior"
    assert translation_prior_semantics("imu_velocity_composed") == "velocity_composed_vio_like"


def test_old_translation_semantics_keyword_interface_still_works():
    assert translation_prior_semantics(use_velocity=False) == "delta_p_motion_damping"
    assert translation_prior_semantics(use_velocity=True) == "velocity_composed_vio_like"


def test_constant_velocity_composition_adds_missing_motion_term():
    delta_p_body = torch.zeros(3)
    velocity_world = torch.tensor([1.0, 0.0, 0.0])
    prior = compose_imu_translation_prior(
        delta_p_body=delta_p_body,
        velocity_world=velocity_world,
        R_body_to_world=torch.eye(3),
        dt_total=0.1,
    )
    assert torch.allclose(prior, torch.tensor([0.1, 0.0, 0.0]), atol=1e-6)


def test_damping_mode_keeps_delta_p_without_velocity_term():
    from Utility.IMUKinematics import compose_translation_prior_by_mode

    prior = compose_translation_prior_by_mode(
        mode="damping_delta_p",
        delta_p_body=torch.zeros(3),
        imu_velocity_world=torch.tensor([3.0, 0.0, 0.0]),
        visual_velocity_world=torch.tensor([2.0, 0.0, 0.0]),
        R_body_to_world=torch.eye(3),
        dt_total=0.5,
    )
    assert torch.allclose(prior, torch.zeros(3), atol=1e-6)


def test_visual_velocity_mode_uses_visual_velocity_not_imu_velocity():
    from Utility.IMUKinematics import compose_translation_prior_by_mode

    prior = compose_translation_prior_by_mode(
        mode="visual_velocity_composed",
        delta_p_body=torch.zeros(3),
        imu_velocity_world=torch.tensor([3.0, 0.0, 0.0]),
        visual_velocity_world=torch.tensor([2.0, 0.0, 0.0]),
        R_body_to_world=torch.eye(3),
        dt_total=0.5,
    )
    assert torch.allclose(prior, torch.tensor([1.0, 0.0, 0.0]), atol=1e-6)


def test_imu_velocity_mode_uses_imu_velocity():
    from Utility.IMUKinematics import compose_translation_prior_by_mode

    prior = compose_translation_prior_by_mode(
        mode="imu_velocity_composed",
        delta_p_body=torch.zeros(3),
        imu_velocity_world=torch.tensor([3.0, 0.0, 0.0]),
        visual_velocity_world=torch.tensor([2.0, 0.0, 0.0]),
        R_body_to_world=torch.eye(3),
        dt_total=0.5,
    )
    assert torch.allclose(prior, torch.tensor([1.5, 0.0, 0.0]), atol=1e-6)
