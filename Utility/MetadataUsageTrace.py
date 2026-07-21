"""
Generate metadata_usage_trace.json for audit.
Non-intrusive: reads metadata and documents what each field is used for.
"""
import json
from pathlib import Path
from typing import Any

from Utility.IMUKinematics import (
    imu_bias_sigma_to_continuous_random_walk_density,
    imu_sigma_to_continuous_density,
)


def _status(value, used_in: list[str], note: str = "") -> dict[str, Any]:
    """Helper: build a trace entry."""
    status = "USED_IN_COMPUTATION" if used_in else "READ_ONLY"
    entry: dict[str, Any] = {"value": value, "status": status, "used_in": used_in}
    if note:
        entry["note"] = note
    return entry


def _sigma_conversion_note(imu_calib: dict, key: str, rate_hz: float, *, bias: bool = False) -> str:
    sigma_unit = imu_calib.get(
        "bias_sigma_unit" if bias else "sigma_unit",
        imu_calib.get("sigma_unit", "legacy_sqrt_rate_scaled"),
    )
    value = imu_calib.get(key)
    if value is None:
        return f"sigma_unit={sigma_unit}; value missing"
    try:
        if bias:
            density = imu_bias_sigma_to_continuous_random_walk_density(float(value), rate_hz, sigma_unit)
        else:
            density = imu_sigma_to_continuous_density(float(value), rate_hz, sigma_unit)
    except Exception as exc:
        return f"sigma_unit={sigma_unit}; conversion failed: {exc}"
    density_name = "continuous_random_walk_density" if bias else "continuous_noise_density"
    return f"sigma_unit={sigma_unit}; {density_name}={density:.8g}"


def _metadata_uses_holoocean_nwu_flu(meta: dict | None) -> bool:
    if not meta:
        return False
    coord = meta.get("coordinate_convention", {})
    imu_meta = meta.get("imu", {})
    world_frame = str(coord.get("export_world_frame", coord.get("holocean_world_frame", ""))).upper()
    body_frame = str(coord.get("body_frame", "")).upper()
    camera_frame = str(coord.get("camera_frame", "")).upper()
    imu_frame = str(coord.get("imu_measurement_frame", imu_meta.get("frame", ""))).upper()
    return (
        "NWU" in world_frame
        and "NWU" in body_frame
        and "FLU" in imu_frame
        and ("BODY NWU" in camera_frame or "ALIGNED" in camera_frame)
    )


def generate_metadata_usage_trace(
    seq_root: Path,
    meta: dict | None,
    cam_meta: dict | None,
    imu_calib: dict | None,
    imu_extrinsic: dict | None,
) -> dict[str, Any]:
    """Generate a machine-readable metadata usage trace for audit."""
    trace: dict[str, Any] = {"scene": seq_root.name, "warnings": []}

    # ── Camera ──────────────────────────────────────────────────────
    cam_fields: dict[str, Any] = {}
    if cam_meta:
        cam_fields["fx"] = _status(cam_meta.get("fx"), ["K_matrix", "pixel2point_NED", "point2pixel_NED", "analytic_Jacobian"])
        cam_fields["fy"] = _status(cam_meta.get("fy"), ["K_matrix", "pixel2point_NED", "point2pixel_NED", "analytic_Jacobian"])
        cam_fields["cx"] = _status(cam_meta.get("cx"), ["K_matrix", "point2pixel_NED", "analytic_Jacobian"])
        cam_fields["cy"] = _status(cam_meta.get("cy"), ["K_matrix", "point2pixel_NED", "analytic_Jacobian"])
        cam_fields["baseline_m"] = _status(cam_meta.get("baseline_m"), ["disparity_residual", "disparity_analytic_Jacobian", "depth_from_disparity"])
        cam_fields["image_width"] = _status(cam_meta.get("image_width"), [], "implicit in image tensor dims")
        cam_fields["image_height"] = _status(cam_meta.get("image_height"), [], "implicit in image tensor dims")
        cam_fields["camera_model"] = _status(cam_meta.get("camera_model"), [], "pinhole assumed; no explicit check")
        cam_fields["distortion_model"] = _status(cam_meta.get("distortion_model"), [], "none assumed; no undistort applied")
        cam_fields["camera_rate_hz"] = _status(cam_meta.get("camera_rate_hz"), ["check_script_rate_validation"])
        cam_fields["left_image_folder"] = _status(cam_meta.get("left_image_folder"), [], "hardcoded 'left' (consistent)")
        cam_fields["right_image_folder"] = _status(cam_meta.get("right_image_folder"), [], "hardcoded 'right' (consistent)")
        cam_fields["image_format"] = _status(cam_meta.get("image_format"), [], "from config 'format' field (png)")
    else:
        trace["warnings"].append("No camera metadata found")

    # ── IMU ─────────────────────────────────────────────────────────
    imu_fields: dict[str, Any] = {}
    if imu_calib:
        rate_hz = float(imu_calib.get("rate_hz", 100))
        imu_fields["rate_hz"] = _status(rate_hz, ["per_sample_to_continuous_noise_conversion"])
        imu_fields["frame"] = _status(imu_calib.get("frame"), ["FLU_to_NED_conversion_trigger"], "FLU detected → R_x(180°) applied")
        imu_fields["acc_unit"] = _status(imu_calib.get("acc_unit"), ["GeneralStereoIMU_unit_validation"], "must be m/s² for preintegration")
        imu_fields["gyro_unit"] = _status(imu_calib.get("gyro_unit"), ["GeneralStereoIMU_unit_validation"], "must be rad/s for preintegration")
        imu_fields["sigma_unit"] = _status(imu_calib.get("sigma_unit", "legacy_sqrt_rate_scaled"), ["imu_sigma_to_continuous_density"])
        imu_fields["bias_sigma_unit"] = _status(
            imu_calib.get("bias_sigma_unit", imu_calib.get("sigma_unit", "legacy_sqrt_rate_scaled")),
            ["imu_bias_sigma_to_continuous_random_walk_density"],
        )
        if imu_fields["acc_unit"]["value"] != "m/s^2":
            trace["warnings"].append(f"acc_unit={imu_calib.get('acc_unit')} NOT m/s², no conversion!")
        if imu_fields["gyro_unit"]["value"] != "rad/s":
            trace["warnings"].append(f"gyro_unit={imu_calib.get('gyro_unit')} NOT rad/s, no conversion!")
        imu_fields["acc_includes_gravity"] = _status(imu_calib.get("acc_includes_gravity"), ["preintegration_gravity_correction"], "consistent with a_corr = acc + g_body")
        imu_fields["noise_model"] = _status(imu_calib.get("noise_model"), ["Q_d=G·Q_c·G^T·dt_in_preintegration"])
        imu_fields["AccelSigma"] = _status(
            imu_calib.get("AccelSigma"), ["preintegration_covariance", "rot_prior_std_floor"],
            _sigma_conversion_note(imu_calib, "AccelSigma", rate_hz))
        imu_fields["AngVelSigma"] = _status(
            imu_calib.get("AngVelSigma"), ["preintegration_covariance", "rot_prior_std_floor"],
            _sigma_conversion_note(imu_calib, "AngVelSigma", rate_hz))
        imu_fields["AccelBiasSigma"] = _status(
            imu_calib.get("AccelBiasSigma"),
            ["preintegration_covariance_bias_rw"],
            _sigma_conversion_note(imu_calib, "AccelBiasSigma", rate_hz, bias=True),
        )
        imu_fields["AngVelBiasSigma"] = _status(
            imu_calib.get("AngVelBiasSigma"),
            ["preintegration_covariance_bias_rw"],
            _sigma_conversion_note(imu_calib, "AngVelBiasSigma", rate_hz, bias=True),
        )
        imu_fields["gravity_m_s2"] = _status(
            imu_calib.get("gravity_m_s2"),
            ["GeneralStereoIMU.gravity", "preintegration_gravity_sign"],
            "metadata gravity overrides config when present",
        )
        imu_fields["imu_csv_format"] = _status(imu_calib.get("imu_csv_format"), [], "documentation only")
    else:
        trace["warnings"].append("No IMU calibration metadata found")

    # ── Extrinsics ───────────────────────────────────────────────────
    ext_fields: dict[str, Any] = {}
    if imu_extrinsic:
        t_ic = imu_extrinsic.get("T_imu_camera", {})
        t_bi = imu_extrinsic.get("T_body_imu", {})
        t_bc = imu_extrinsic.get("T_body_camera", {})
        trans_ic = t_ic.get("translation_body_nwu_m", None)
        trans_bi = t_bi.get("translation_body_nwu_m", None)
        trans_bc = t_bc.get("translation_body_nwu_m", None)
        uses_body_imu = _metadata_uses_holoocean_nwu_flu(meta) and bool(trans_bi)
        uses_camera_imu_lever = uses_body_imu and bool(trans_bc)
        body_camera_note = (
            "paired with T_body_imu to compute camera-to-IMU lever arm"
            if uses_camera_imu_lever
            else ("present but not selected for this metadata convention" if t_bc else "MISSING in metadata")
        )
        ext_fields["T_imu_camera.translation"] = _status(
            trans_ic,
            ["imu_vio_sensor_T_imu_fallback"] if (not uses_camera_imu_lever and trans_ic) else [],
            "fallback source for camera-to-IMU lever arm when T_body_camera is unavailable"
            if (not uses_camera_imu_lever and trans_ic)
            else "read-only when T_body_imu and T_body_camera are both available",
        )
        ext_fields["T_imu_camera.rotation"] = _status(
            t_ic.get("rotation", None), ["FLU_to_NED_R_x_180"],
            "hardcoded R_x(180°) consistent with metadata description")
        ext_fields["body_frame"] = _status(imu_extrinsic.get("body_frame"), [], "logged only")
        ext_fields["camera_frame"] = _status(imu_extrinsic.get("camera_frame"), [], "logged only; camera-body aligned identity")
        ext_fields["imu_frame"] = _status(imu_extrinsic.get("imu_frame"), ["FLU_to_NED_trigger"])
        ext_fields["T_body_imu"] = _status(
            t_bi if t_bi else None,
            ["imu_T_BS_translation", "imu_vio_sensor_T_imu"] if uses_camera_imu_lever else (["imu_T_BS_translation"] if uses_body_imu else []),
            "primary HoloOcean source for imu_T_BS translation and camera-to-IMU lever arm"
            if uses_camera_imu_lever
            else ("primary HoloOcean source for imu_T_BS translation" if uses_body_imu else "MISSING in metadata"),
        )
        ext_fields["T_body_camera"] = _status(
            t_bc if t_bc else None,
            ["imu_vio_sensor_T_imu"] if uses_camera_imu_lever else [],
            body_camera_note,
        )
        for k in ["T_body_imu", "T_body_camera", "T_imu_body", "T_camera_body", "T_camera_imu"]:
            if k in ext_fields:
                continue
            ext_fields[k] = _status(imu_extrinsic.get(k), [], "MISSING in metadata" if k not in imu_extrinsic else "present but UNUSED")
    else:
        trace["warnings"].append("No extrinsics metadata found")

    # ── Coordinate Convention ────────────────────────────────────────
    cc = meta.get("coordinate_convention", {}) if meta else {}
    coord_fields: dict[str, Any] = {
        "export_world_frame": _status(cc.get("export_world_frame"), [], "NWU, same as MACVO output frame"),
        "body_frame": _status(cc.get("body_frame"), [], "NWU, logged"),
        "imu_frame": _status(cc.get("imu_frame"), ["FLU_to_NED_conversion_decision"]),
        "imu_measurement_frame": _status(cc.get("imu_measurement_frame"), ["FLU_to_NED_input"]),
        "ref_pose_position_frame": _status(cc.get("ref_pose_position_frame"), ["GT_diagnostics_frame_annotation"]),
        "R_holocean_to_export": _status(cc.get("R_holocean_to_export"), [], "identity → no extra rotation needed"),
        "R_sensor_to_flu": _status(cc.get("R_sensor_to_flu"), ["FLU_to_NED_inverse_applied"], "inverse operation hardcoded as R_x(180°)"),
        "quaternion_convention": _status(cc.get("quaternion_convention"), ["GT_quat_parsing"], "xyzw order; PyPose SO3 stores xyzw"),
    }

    # ── Time Synchronization ─────────────────────────────────────────
    ts = meta.get("time_synchronization", {}) if meta else {}
    time_fields: dict[str, Any] = {
        "timestamp_unit": _status(ts.get("timestamp_unit"), ["GeneralStereoIMU_timestamp_unit_validation"], "must be ns"),
        "camera_rate_hz": _status(ts.get("camera_rate_hz"), ["check_script_validation"]),
        "imu_rate_hz": _status(ts.get("imu_rate_hz"), ["per_sample_to_continuous_sigma", "dt_validation"]),
        "camera_ref_pose_aligned": _status(ts.get("camera_ref_pose_aligned"), ["timestamp_matching_in_diagnostics"]),
        "camera_imu_time_offset_ns": _status(
            ts.get("camera_imu_time_offset_ns"),
            ["GeneralStereoIMU_time_offset_precedence"],
            "metadata offset takes precedence; endpoint estimate is used only when metadata has no offset",
        ),
        "imu_samples_per_camera_frame_expected": _status(ts.get("imu_samples_per_camera_frame_expected"), ["IMU_window_selection"], "~10 per frame at 300Hz/30fps"),
    }

    return {
        "scene": seq_root.name,
        "camera": cam_fields,
        "imu": imu_fields,
        "extrinsics": ext_fields,
        "coordinate_convention": coord_fields,
        "time_synchronization": time_fields,
        "warnings": trace["warnings"],
    }


def write_metadata_usage_trace(seq_root: Path, saveto: Any) -> None:
    """Write metadata_usage_trace.json to results directory.
    saveto: Sandbox object with .path() method or Path.
    """
    import json as _json
    from pathlib import Path as _Path

    meta = None
    meta_path = seq_root / "metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = _json.load(f)

    cam_meta = meta.get("camera", None) if meta else None
    imu_calib = meta.get("imu", None) if meta else None
    imu_extrinsic = meta.get("extrinsics", None) if meta else None

    trace = generate_metadata_usage_trace(seq_root, meta, cam_meta, imu_calib, imu_extrinsic)

    if hasattr(saveto, 'path'):
        out_path = saveto.path("metadata_usage_trace.json")
    else:
        out_path = _Path(saveto) / "metadata_usage_trace.json"

    out_path.write_text(_json.dumps(trace, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
