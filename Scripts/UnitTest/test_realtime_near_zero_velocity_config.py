from pathlib import Path

import yaml

from Scripts.run_realtime_t2 import configure_odom
from Utility.NearZeroVelocityDetector import FROZEN_NEAR_ZERO_VELOCITY_V2


def _configure(
    path: Path,
    *,
    detector: str,
    visual_factor: str = "pace",
) -> dict:
    configure_odom(
        path,
        model=Path("/tmp/model.pth"),
        trace=Path("/tmp/pipeline_trace.csv"),
        parallel=True,
        static_mode="fixed",
        static_state_policy="estimated",
        static_duration_s=3.0,
        static_min_duration_s=1.0,
        static_max_duration_s=8.0,
        static_window_s=0.25,
        static_stable_hold_s=0.75,
        cpu_threads=4,
        vio_backend="isam2",
        visual_factor=visual_factor,
        near_zero_velocity_detector=detector,
        near_zero_velocity_prior_std_m_s=0.01,
    )
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_realtime_config_keeps_detector_disabled_by_default_path(tmp_path):
    config = _configure(tmp_path / "off.yaml", detector="off")
    optimizer = config["Odometry"]["optimizer"]["args"]

    assert optimizer["two_state_near_zero_velocity_enable"] is False


def test_realtime_config_records_frozen_v2_contract(tmp_path):
    config = _configure(tmp_path / "v2.yaml", detector="v2")
    optimizer = config["Odometry"]["optimizer"]["args"]

    assert optimizer["two_state_near_zero_velocity_enable"] is True
    assert optimizer["two_state_near_zero_velocity_detector_version"] == "v2"
    assert optimizer["two_state_near_zero_velocity_prior_std_m_s"] == 0.01
    frozen = FROZEN_NEAR_ZERO_VELOCITY_V2
    assert optimizer[
        "two_state_near_zero_velocity_v2_minimum_imu_angular_rate_rad_s"
    ] == frozen.minimum_imu_angular_rate_rad_s
    assert optimizer[
        "two_state_near_zero_velocity_v2_minimum_visual_angular_rate_rad_s"
    ] == frozen.minimum_visual_angular_rate_rad_s
    assert optimizer[
        "two_state_near_zero_velocity_v2_maximum_rotation_vector_rate_difference_rad_s"
    ] == frozen.maximum_rotation_vector_rate_difference_rad_s
    assert optimizer[
        "two_state_near_zero_velocity_v2_minimum_rotation_axis_cosine"
    ] == frozen.minimum_rotation_axis_cosine
    assert optimizer[
        "two_state_near_zero_velocity_v2_maximum_zero_translation_nis_per_dof"
    ] == frozen.maximum_zero_translation_nis_per_dof
    assert optimizer[
        "two_state_near_zero_velocity_enter_hold_s"
    ] == frozen.enter_hold_s
    assert optimizer[
        "two_state_near_zero_velocity_release_hold_s"
    ] == frozen.release_hold_s


def test_realtime_config_maps_all_visual_factor_names(tmp_path):
    expected = {
        "pose": "relative_pose",
        "uvd": "direct_uvd",
        "pace": "compressed_uvd",
    }
    for public_name, internal_name in expected.items():
        config = _configure(
            tmp_path / f"{public_name}.yaml",
            detector="off",
            visual_factor=public_name,
        )
        optimizer = config["Odometry"]["optimizer"]["args"]
        assert optimizer["two_state_visual_factor_mode"] == internal_name
