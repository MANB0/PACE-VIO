from pathlib import Path

from Scripts.audit_metadata_usage import audit_trace
from Utility.MetadataUsageTrace import generate_metadata_usage_trace


def test_metadata_usage_trace_reports_sigma_unit_conversion():
    trace = generate_metadata_usage_trace(
        seq_root=Path("scene"),
        meta={"coordinate_convention": {}, "time_synchronization": {}},
        cam_meta=None,
        imu_calib={
            "rate_hz": 100,
            "frame": "FLU",
            "acc_unit": "m/s^2",
            "gyro_unit": "rad/s",
            "sigma_unit": "per-sample standard deviation",
            "AccelSigma": 0.02,
            "AngVelSigma": 0.04,
            "AccelBiasSigma": 0.01,
            "AngVelBiasSigma": 0.03,
        },
        imu_extrinsic=None,
    )

    assert trace["imu"]["sigma_unit"]["value"] == "per-sample standard deviation"
    assert trace["imu"]["bias_sigma_unit"]["value"] == "per-sample standard deviation"
    assert "continuous_noise_density=0.002" in trace["imu"]["AccelSigma"]["note"]
    assert "continuous_noise_density=0.004" in trace["imu"]["AngVelSigma"]["note"]
    assert "continuous_random_walk_density=0.1" in trace["imu"]["AccelBiasSigma"]["note"]
    assert "continuous_random_walk_density=0.3" in trace["imu"]["AngVelBiasSigma"]["note"]


def test_metadata_usage_trace_uses_body_imu_extrinsic_for_holoocean_imu_tbs():
    trace = generate_metadata_usage_trace(
        seq_root=Path("clear_shallow"),
        meta={
            "coordinate_convention": {
                "export_world_frame": "NWU",
                "body_frame": "NWU",
                "camera_frame": "body NWU, aligned",
                "imu_measurement_frame": "FLU",
            },
            "time_synchronization": {},
        },
        cam_meta={
            "fx": 320.0,
            "fy": 320.0,
            "cx": 320.0,
            "cy": 240.0,
            "baseline_m": 0.225,
            "camera_rate_hz": 30,
        },
        imu_calib={
            "rate_hz": 300,
            "frame": "FLU",
            "acc_unit": "m/s^2",
            "gyro_unit": "rad/s",
            "acc_includes_gravity": True,
            "gravity_m_s2": 9.8,
            "noise_model": "additive_gaussian_per_sample",
            "sigma_unit": "per-sample standard deviation",
            "AccelSigma": 0.00277,
            "AngVelSigma": 0.00123,
            "AccelBiasSigma": 0.00141,
            "AngVelBiasSigma": 0.00388,
        },
        imu_extrinsic={
            "T_imu_camera": {"translation_body_nwu_m": [0.417, 0.18, 0.095]},
            "T_body_imu": {"translation_body_nwu_m": [-0.097, -0.07, 0.06]},
            "T_body_camera": {"translation_body_nwu_m": [0.32, 0.11, 0.155]},
        },
    )

    assert trace["extrinsics"]["T_body_imu"]["status"] == "USED_IN_COMPUTATION"
    assert "imu_T_BS_translation" in trace["extrinsics"]["T_body_imu"]["used_in"]
    assert "imu_vio_sensor_T_imu" in trace["extrinsics"]["T_body_imu"]["used_in"]
    assert trace["extrinsics"]["T_body_camera"]["status"] == "USED_IN_COMPUTATION"
    assert "imu_vio_sensor_T_imu" in trace["extrinsics"]["T_body_camera"]["used_in"]
    assert trace["extrinsics"]["T_imu_camera.translation"]["status"] == "READ_ONLY"
    assert audit_trace(trace)["issues"] == []


def test_metadata_usage_trace_documents_xyzw_quaternion_order_for_pypose():
    trace = generate_metadata_usage_trace(
        seq_root=Path("clear_shallow"),
        meta={
            "coordinate_convention": {
                "quaternion_convention": "xyzw, qw>=0, normalized",
            },
            "time_synchronization": {},
        },
        cam_meta=None,
        imu_calib=None,
        imu_extrinsic=None,
    )

    note = trace["coordinate_convention"]["quaternion_convention"]["note"]
    assert "xyzw order" in note
    assert "[qw,qx,qy,qz]" not in note
