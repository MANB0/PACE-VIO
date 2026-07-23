"""Generate a machine-readable trace of metadata fields used at runtime."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _status(value: object, used_in: list[str], note: str = "") -> dict[str, Any]:
    status = "USED_IN_COMPUTATION" if used_in else "READ_ONLY"
    entry: dict[str, Any] = {"value": value, "status": status, "used_in": used_in}
    if note:
        entry["note"] = note
    return entry


def generate_metadata_usage_trace(
    seq_root: Path,
    meta: dict | None,
    cam_meta: dict | None,
    imu_calib: dict | None,
    imu_extrinsic: dict | None,
) -> dict[str, Any]:
    warnings: list[str] = []

    camera: dict[str, Any] = {}
    if cam_meta:
        camera = {
            "fx": _status(cam_meta.get("fx"), ["K_matrix", "UVD_geometry", "visual_Jacobian"]),
            "fy": _status(cam_meta.get("fy"), ["K_matrix", "UVD_geometry", "visual_Jacobian"]),
            "cx": _status(cam_meta.get("cx"), ["K_matrix", "UVD_geometry", "visual_Jacobian"]),
            "cy": _status(cam_meta.get("cy"), ["K_matrix", "UVD_geometry", "visual_Jacobian"]),
            "baseline_m": _status(
                cam_meta.get("baseline_m"),
                ["disparity_residual", "depth_from_disparity", "visual_Jacobian"],
            ),
        }
    else:
        warnings.append("No camera metadata found")

    imu: dict[str, Any] = {}
    if imu_calib:
        density_note = "continuous-time density consumed directly; dt comes from IMU timestamps"
        imu = {
            "NoiseAcc": _status(
                imu_calib.get("NoiseAcc"),
                ["preintegration_measurement_covariance", "static_detector_threshold"],
                density_note,
            ),
            "NoiseGyro": _status(
                imu_calib.get("NoiseGyro"),
                ["preintegration_measurement_covariance", "static_detector_threshold"],
                density_note,
            ),
            "AccWalk": _status(
                imu_calib.get("AccWalk"),
                ["preintegration_bias_random_walk_covariance"],
                density_note,
            ),
            "GyroWalk": _status(
                imu_calib.get("GyroWalk"),
                ["preintegration_bias_random_walk_covariance"],
                density_note,
            ),
            "gravity_m_s2": _status(
                imu_calib.get("gravity_m_s2"),
                ["GeneralStereoIMU.gravity", "factor_gravity_world"],
                "positive magnitude; internal NED gravity is [0, 0, +g]",
            ),
        }
    else:
        warnings.append("No IMU calibration metadata found")

    extrinsics: dict[str, Any] = {}
    if imu_extrinsic:
        extrinsics["T_CI"] = _status(
            imu_extrinsic.get("T_CI"),
            [
                "camera_pose_to_imu_state",
                "imu_state_to_camera_pose",
                "visual_factor_extrinsic_adjoint",
                "IMU_center_trajectory_export",
            ],
            "sole 4x4 extrinsic; p_C = T_CI p_I; never applied to raw IMU samples",
        )
    else:
        warnings.append("No extrinsics metadata found")

    return {
        "scene": seq_root.name,
        "camera": camera,
        "imu": imu,
        "extrinsics": extrinsics,
        "runtime_frame_contract": {
            "world_internal": "NED",
            "world_export": "NWU",
            "imu_measurement_and_preintegration_frame": "raw FLU",
            "camera_internal_frame": "FRD/NED",
            "timing": "nanosecond timestamps; zero camera/IMU offset",
            "note": "fixed code/data contract, not metadata fields or estimated values",
        },
        "warnings": warnings,
    }


def write_metadata_usage_trace(seq_root: Path, saveto: Any) -> None:
    meta_path = seq_root / "metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else None
    trace = generate_metadata_usage_trace(
        seq_root,
        meta,
        meta.get("camera") if meta else None,
        meta.get("imu") if meta else None,
        meta.get("extrinsics") if meta else None,
    )
    out_path = saveto.path("metadata_usage_trace.json") if hasattr(saveto, "path") else Path(saveto) / "metadata_usage_trace.json"
    out_path.write_text(
        json.dumps(trace, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
