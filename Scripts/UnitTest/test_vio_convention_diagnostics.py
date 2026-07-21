from Utility.VIOConventionDiagnostics import (
    FULL_VIO_CONVENTION_FIELDS,
    is_active_full_vio_row,
    validate_full_vio_convention,
)


def _valid_row(**overrides):
    row = {
        "imu_factor_mode": "preintegrated_vio",
        "vio_factor_active": "1",
        "imu_residual_rows": "3",
        "imu_source_world_frame": "NWU",
        "imu_source_measurement_frame": "FLU",
        "imu_internal_world_frame": "NED",
        "imu_internal_measurement_frame": "NED",
        "imu_acc_unit": "m/s^2",
        "imu_gyro_unit": "rad/s",
        "imu_timestamp_unit": "ns",
        "imu_time_offset_ns": "0",
        "imu_time_offset_source": "metadata.time_synchronization.camera_imu_time_offset_ns",
        "imu_gravity_source": "metadata.json",
        "imu_metadata_gravity_m_s2": "9.8",
        "imu_preintegration_gravity_z": "9.8",
    }
    row.update(overrides)
    return row


def test_full_vio_convention_fields_include_expected_contract():
    assert FULL_VIO_CONVENTION_FIELDS == {
        "imu_source_world_frame",
        "imu_source_measurement_frame",
        "imu_internal_world_frame",
        "imu_internal_measurement_frame",
        "imu_acc_unit",
        "imu_gyro_unit",
        "imu_timestamp_unit",
        "imu_time_offset_ns",
        "imu_time_offset_source",
        "imu_gravity_source",
        "imu_metadata_gravity_m_s2",
        "imu_preintegration_gravity_z",
    }


def test_validate_full_vio_convention_accepts_expected_holocean_contract():
    ok, message = validate_full_vio_convention(_valid_row())

    assert ok is True
    assert message == ""


def test_validate_full_vio_convention_rejects_wrong_source_frame():
    ok, message = validate_full_vio_convention(_valid_row(imu_source_measurement_frame="NED"))

    assert ok is False
    assert "imu_source_measurement_frame" in message


def test_is_active_full_vio_row_requires_active_factor_and_conventions():
    assert is_active_full_vio_row(_valid_row()) is True
    assert is_active_full_vio_row(_valid_row(imu_factor_mode="two_state_fixed_lag")) is True
    assert is_active_full_vio_row(_valid_row(vio_factor_active="0")) is False
    assert is_active_full_vio_row(_valid_row(imu_residual_rows="2")) is False
    assert is_active_full_vio_row(_valid_row(imu_preintegration_gravity_z="not-a-number")) is False
    assert is_active_full_vio_row(_valid_row(imu_factor_mode="legacy_pose_prior")) is False


def test_full_vio_convention_rejects_wrong_internal_ned_gravity_sign():
    ok, message = validate_full_vio_convention(_valid_row(imu_preintegration_gravity_z="-9.8"))

    assert ok is False
    assert "imu_preintegration_gravity_z" in message
    assert is_active_full_vio_row(_valid_row(imu_preintegration_gravity_z="-9.8")) is False


def test_full_vio_convention_rejects_endpoint_estimated_time_offset():
    ok, message = validate_full_vio_convention(
        _valid_row(
            imu_time_offset_ns="-15000000",
            imu_time_offset_source="auto_endpoint_average",
        )
    )

    assert ok is False
    assert "imu_time_offset_source" in message
    assert is_active_full_vio_row(
        _valid_row(
            imu_time_offset_ns="-15000000",
            imu_time_offset_source="auto_endpoint_average",
        )
    ) is False
