def test_default_translation_prior_is_labeled_as_motion_damping():
    from Utility.IMUKinematics import translation_prior_semantics

    assert translation_prior_semantics(use_velocity=False) == "delta_p_motion_damping"


def test_velocity_composed_translation_prior_is_labeled_as_vio_like():
    from Utility.IMUKinematics import translation_prior_semantics

    assert translation_prior_semantics(use_velocity=True) == "velocity_composed_vio_like"
