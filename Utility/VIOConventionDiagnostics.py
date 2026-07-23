"""Shared contract checks for preintegrated full-VIO diagnostics."""

from __future__ import annotations

import torch

FULL_VIO_CONVENTION_EXPECTED = {
    "imu_source_world_frame": "NWU",
    "imu_source_measurement_frame": "FLU",
    "imu_internal_world_frame": "NED",
    "imu_internal_measurement_frame": "FLU",
}
FULL_VIO_NUMERIC_CONVENTION_FIELDS = (
    "imu_metadata_gravity_m_s2",
    "imu_preintegration_gravity_z",
)
FULL_VIO_CONVENTION_FIELDS = set(FULL_VIO_CONVENTION_EXPECTED) | set(FULL_VIO_NUMERIC_CONVENTION_FIELDS) | {
    "imu_gravity_source",
}
ACTIVE_FULL_VIO_FACTOR_MODES = {
    "preintegrated_vio",
    "local_inertial_ba",
    "two_state_fixed_lag",
}


def truthy(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def world_nwu_vector_to_internal(
    vector_nwu: torch.Tensor,
    *,
    internal_world_frame: str,
) -> torch.Tensor:
    """Express a world-frame NWU vector in the configured internal basis."""
    vector = torch.as_tensor(vector_nwu)
    if "NED" not in str(internal_world_frame).upper():
        return vector.clone()
    return vector * vector.new_tensor([1.0, -1.0, -1.0])


def camera_velocity_to_imu_origin(
    *,
    camera_velocity_world_nwu: torch.Tensor,
    angular_velocity_body_nwu: torch.Tensor,
    camera_to_imu_body_internal: torch.Tensor,
    camera_rotation_body_to_world_internal: torch.Tensor,
    internal_world_frame: str,
) -> torch.Tensor:
    """Shift CameraLeftSocket world velocity to the IMUSocket origin."""
    camera_velocity = world_nwu_vector_to_internal(
        camera_velocity_world_nwu,
        internal_world_frame=internal_world_frame,
    )
    angular_velocity = world_nwu_vector_to_internal(
        angular_velocity_body_nwu,
        internal_world_frame=internal_world_frame,
    ).to(camera_velocity)
    lever_arm = torch.as_tensor(camera_to_imu_body_internal).reshape(3).to(camera_velocity)
    rotation = torch.as_tensor(camera_rotation_body_to_world_internal).reshape(3, 3).to(camera_velocity)
    lever_velocity_body = torch.linalg.cross(angular_velocity.reshape(3), lever_arm)
    return camera_velocity.reshape(3) + rotation @ lever_velocity_body


def validate_full_vio_convention(row: dict[str, str]) -> tuple[bool, str]:
    for key, expected in FULL_VIO_CONVENTION_EXPECTED.items():
        actual = str(row.get(key, "")).strip()
        if actual.lower() != expected.lower():
            return False, f"{key}={actual!r}, expected {expected!r}"
    if not str(row.get("imu_gravity_source", "")).strip():
        return False, "imu_gravity_source is empty"
    numeric_values: dict[str, float] = {}
    for key in FULL_VIO_NUMERIC_CONVENTION_FIELDS:
        try:
            numeric_values[key] = float(str(row.get(key, "")).strip())
        except ValueError:
            return False, f"{key} is not numeric"
    metadata_gravity = numeric_values["imu_metadata_gravity_m_s2"]
    preintegration_gravity_z = numeric_values["imu_preintegration_gravity_z"]
    if metadata_gravity <= 0.0:
        return False, f"imu_metadata_gravity_m_s2={metadata_gravity!r}, expected positive gravity magnitude"
    if preintegration_gravity_z <= 0.0:
        return False, (
            f"imu_preintegration_gravity_z={preintegration_gravity_z!r}, "
            "expected positive NED z-down gravity"
        )
    if abs(preintegration_gravity_z - abs(metadata_gravity)) > 1e-3:
        return False, (
            f"imu_preintegration_gravity_z={preintegration_gravity_z!r}, "
            f"expected to match metadata gravity {abs(metadata_gravity)!r}"
        )
    return True, ""


def is_active_full_vio_row(row: dict[str, str]) -> bool:
    try:
        residual_rows = int(float(row.get("imu_residual_rows", "0") or "0"))
    except ValueError:
        residual_rows = 0
    if str(row.get("imu_factor_mode", "")).strip().lower() not in ACTIVE_FULL_VIO_FACTOR_MODES:
        return False
    if not truthy(row.get("vio_factor_active")):
        return False
    if residual_rows < 3:
        return False
    valid_convention, _ = validate_full_vio_convention(row)
    return valid_convention
