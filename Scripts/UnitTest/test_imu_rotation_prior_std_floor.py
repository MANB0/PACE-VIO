import Utility.IMUKinematics as imu_kin


def test_rotation_prior_std_respects_configured_engineering_floor():
    assert hasattr(imu_kin, "select_rotation_prior_std")
    assert imu_kin.select_rotation_prior_std(
        preintegrated_std=1.0e-4,
        sensor_noise_std=7.0e-5,
        configured_floor=1.2e-2,
    ) == 1.2e-2


def test_rotation_prior_std_keeps_larger_preintegrated_uncertainty():
    assert imu_kin.select_rotation_prior_std(
        preintegrated_std=3.0e-2,
        sensor_noise_std=7.0e-5,
        configured_floor=1.2e-2,
    ) == 3.0e-2


def test_translation_active_rotation_floor_only_applies_when_translation_is_active():
    assert imu_kin.select_translation_active_rotation_prior_std(
        base_std=1.2e-2,
        translation_active=False,
        translation_active_floor=3.0e-2,
    ) == 1.2e-2

    assert imu_kin.select_translation_active_rotation_prior_std(
        base_std=1.2e-2,
        translation_active=True,
        translation_active_floor=3.0e-2,
    ) == 3.0e-2

    assert imu_kin.select_translation_active_rotation_prior_std(
        base_std=5.0e-2,
        translation_active=True,
        translation_active_floor=3.0e-2,
    ) == 5.0e-2
