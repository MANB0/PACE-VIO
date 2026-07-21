from Scripts import run_vio_imu_prior_mode_grid as grid
from Scripts import run_local_ba_writeback_validation as validation


def test_validation_batch_targets_rectangle_writeback_matrix():
    assert validation.VALIDATION_SCENES == [
        "clear_rectangle_zero_noise",
        "clear_rectangle_normal_noise",
    ]

    assert validation.BASELINE_VARIANTS == [
        "pure_macvo",
        "vio_preintegrated_full_imuatt_estinit",
    ]

    expected_local_ba = {
        "vio_local_ba_w2_imuatt": (2, "current"),
        "vio_local_ba_w2_imuatt_all": (2, "all_optimized"),
        "vio_local_ba_w3_imuatt": (3, "current"),
        "vio_local_ba_w3_imuatt_all": (3, "all_optimized"),
    }
    assert validation.LOCAL_BA_VARIANTS == list(expected_local_ba)
    assert validation.VALIDATION_VARIANTS == validation.BASELINE_VARIANTS + validation.LOCAL_BA_VARIANTS

    for variant_name, (window_size, writeback) in expected_local_ba.items():
        variant = grid.VARIANTS[variant_name]
        assert variant.imu_factor_mode == "local_inertial_ba"
        assert variant.force_mode == "full_imu"
        assert variant.gravity_pose_source == "imu_integrated_estinit"
        assert variant.local_ba_window_size == window_size
        assert variant.local_ba_writeback == writeback
        assert variant.local_ba_fix_first_frame is True
