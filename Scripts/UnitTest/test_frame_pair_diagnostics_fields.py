import csv
import importlib
from pathlib import Path

import torch

from Utility.FramePairDiagnostics import CSV_HEADER, FramePairDiagnosticsWriter


def test_imu_translation_prior_diagnostics_are_exported():
    required = [
        "imu_trans_prior_mode",
        "imu_trans_prior_std_x",
        "imu_trans_prior_std_y",
        "imu_trans_prior_std_z",
        "imu_noise_sigma_unit",
        "imu_noise_source",
    ]
    for field in required:
        assert field in CSV_HEADER


def test_imu_translation_prior_diagnostics_live_with_imu_fields():
    assert CSV_HEADER.index("imu_trans_prior_mode") > CSV_HEADER.index("imu_dt")
    assert CSV_HEADER.index("imu_trans_prior_std_z") < CSV_HEADER.index("delta_p_over_est")
    assert CSV_HEADER.index("imu_noise_source") < CSV_HEADER.index("delta_p_over_est")


def test_visual_input_fingerprint_is_exported_with_visual_stats():
    assert "visual_input_sha256" in CSV_HEADER
    assert CSV_HEADER.index("visual_input_sha256") < CSV_HEADER.index("visual_loss_raw_sum")


def test_vio_factor_activity_diagnostics_are_exported_with_switches():
    required = [
        "imu_factor_mode",
        "vio_factor_active",
        "imu_residual_rows",
    ]
    for field in required:
        assert field in CSV_HEADER

    assert CSV_HEADER.index("imu_factor_mode") > CSV_HEADER.index("autodiff_enabled")
    assert CSV_HEADER.index("imu_residual_rows") < CSV_HEADER.index("est_delta_x")


def test_vio_detailed_residual_diagnostics_are_exported_with_losses():
    required = [
        "r_v_whitened_norm",
        "imu_vio_whitened_norm",
        "imu_vio_raw_norm",
        "imu_vel_loss",
        "imu_vio_cov_trace",
        "imu_vio_weight_trace",
        "imu_vio_weight_diag_min",
        "imu_vio_weight_diag_max",
        "imu_vio_acc_bias_norm",
        "imu_vio_gyro_bias_norm",
        "imu_vio_acc_bias_x",
        "imu_vio_acc_bias_y",
        "imu_vio_acc_bias_z",
        "imu_vio_gyro_bias_x",
        "imu_vio_gyro_bias_y",
        "imu_vio_gyro_bias_z",
        "imu_vel_loss_ratio",
    ]
    for field in required:
        assert field in CSV_HEADER

    assert CSV_HEADER.index("r_v_whitened_norm") > CSV_HEADER.index("r_p_whitened_norm")
    assert CSV_HEADER.index("imu_vel_loss") > CSV_HEADER.index("imu_trans_loss")
    assert CSV_HEADER.index("imu_vio_weight_diag_max") < CSV_HEADER.index("visual_loss_ratio")


def test_vio_alpha_diagnostics_are_exported_with_imu_residual_fields():
    required = [
        "imu_vio_alpha_p",
        "imu_vio_alpha_v",
        "imu_vio_alpha_R",
    ]
    for field in required:
        assert field in CSV_HEADER

    assert CSV_HEADER.index("imu_vio_alpha_p") > CSV_HEADER.index("imu_vio_gyro_bias_z")
    assert CSV_HEADER.index("imu_vio_alpha_R") < CSV_HEADER.index("total_loss")


def test_vio_cost_scale_audit_fields_are_exported_after_alpha_fields():
    required = [
        "energy_visual_weighted",
        "energy_p_weighted",
        "energy_v_weighted",
        "energy_R_weighted",
        "energy_pv_weighted",
        "energy_imu_diag_weighted",
        "energy_imu_weighted",
        "energy_imu_to_visual_ratio",
        "energy_pv_to_visual_ratio",
        "energy_R_to_visual_ratio",
    ]
    for field in required:
        assert field in CSV_HEADER

    assert CSV_HEADER.index("energy_visual_weighted") > CSV_HEADER.index("imu_vio_alpha_R")
    assert CSV_HEADER.index("energy_R_to_visual_ratio") < CSV_HEADER.index("total_loss")


def test_causal_diagnostic_fields_are_grouped_and_written(tmp_path):
    causal_fields = [
        "init_delta_x",
        "init_delta_y",
        "init_delta_z",
        "init_delta_t_norm",
        "init_delta_R_angle",
        "init_velocity_j_x",
        "init_velocity_j_y",
        "init_velocity_j_z",
        "initial_energy_visual_weighted",
        "initial_energy_p_weighted",
        "initial_energy_v_weighted",
        "initial_energy_R_weighted",
        "initial_energy_pv_weighted",
        "initial_energy_imu_diag_weighted",
        "initial_energy_imu_weighted",
        "initial_energy_imu_to_visual_ratio",
        "initial_energy_pv_to_visual_ratio",
        "initial_energy_R_to_visual_ratio",
        "initial_total_loss",
        "update_pose_translation_norm",
        "update_pose_rotation_norm",
        "update_velocity_norm",
        "update_acc_bias_norm",
        "update_gyro_bias_norm",
        "influence_visual_grad_norm",
        "influence_imu_grad_norm",
        "influence_grad_cosine",
        "influence_visual_hessian_trace",
        "influence_imu_hessian_trace",
        "influence_imu_to_visual_grad_ratio",
        "influence_imu_to_visual_hessian_ratio",
        "influence_p_grad_norm",
        "influence_v_grad_norm",
        "influence_R_grad_norm",
        "influence_sampled",
        "energy_visual_change",
        "energy_imu_change",
        "energy_p_change",
        "energy_v_change",
        "energy_R_change",
        "counterfactual_visual_step_norm",
        "counterfactual_imu_step_norm",
        "counterfactual_full_step_norm",
        "counterfactual_visual_to_imu_cosine",
        "actual_to_visual_step_cosine",
        "actual_to_imu_step_cosine",
        "actual_to_full_step_cosine",
        "predicted_visual_change_on_actual_step",
        "predicted_imu_change_on_actual_step",
    ]

    for field in causal_fields:
        assert field in CSV_HEADER

    final_energy_start = CSV_HEADER.index("energy_visual_weighted")
    assert all(CSV_HEADER.index(field) < final_energy_start for field in causal_fields)

    path = tmp_path / "frame_pair_diagnostics.csv"
    writer = FramePairDiagnosticsWriter(path)
    writer.write_row(**{field: index + 0.5 for index, field in enumerate(causal_fields)})
    writer.close()

    with path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))

    assert rows[0] == CSV_HEADER
    for index, field in enumerate(causal_fields):
        assert rows[1][CSV_HEADER.index(field)] == f"{index + 0.5:.9g}"


def test_source_to_gt_fields_are_exported_before_causal_energies():
    required = [
        "init_over_gt_translation_ratio",
        "cos_init_gt_translation",
        "init_rotation_error_angle",
        "imu_rotation_error_angle",
        "velocity_reference_point",
        "gt_velocity_i_x",
        "gt_velocity_i_y",
        "gt_velocity_i_z",
        "gt_velocity_j_x",
        "gt_velocity_j_y",
        "gt_velocity_j_z",
        "gt_delta_velocity_norm",
        "est_velocity_j_x",
        "est_velocity_j_y",
        "est_velocity_j_z",
        "init_velocity_error_norm",
        "est_velocity_error_norm",
    ]
    for field in required:
        assert field in CSV_HEADER

    assert CSV_HEADER.index("init_over_gt_translation_ratio") > CSV_HEADER.index("rotation_error_angle")
    assert CSV_HEADER.index("est_velocity_error_norm") < CSV_HEADER.index("initial_energy_visual_weighted")


def test_world_nwu_vector_conversion_uses_internal_world_basis():
    module = importlib.import_module("Utility.VIOConventionDiagnostics")
    convert = getattr(module, "world_nwu_vector_to_internal")
    vector = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)

    converted_ned = convert(vector, internal_world_frame="NED")
    converted_nwu = convert(vector, internal_world_frame="NWU")

    assert converted_ned.dtype == vector.dtype
    assert torch.equal(converted_ned, torch.tensor([1.0, -2.0, -3.0], dtype=torch.float64))
    assert torch.equal(converted_nwu, vector)


def test_camera_velocity_is_shifted_to_imu_origin_with_lever_arm_rate():
    module = importlib.import_module("Utility.VIOConventionDiagnostics")
    convert = getattr(module, "camera_velocity_to_imu_origin")

    velocity = convert(
        camera_velocity_world_nwu=torch.tensor([1.0, 0.0, 0.0]),
        angular_velocity_body_nwu=torch.tensor([0.0, 0.0, 1.0]),
        camera_to_imu_body_internal=torch.tensor([1.0, 0.0, 0.0]),
        camera_rotation_body_to_world_internal=torch.eye(3),
        internal_world_frame="NED",
    )

    assert torch.allclose(velocity, torch.tensor([1.0, -1.0, 0.0]))


def test_causal_fields_are_forwarded_from_optimizer_to_csv_writer():
    root = Path(__file__).resolve().parents[2]
    optimizer_source = (root / "Module" / "Optimization" / "TwoFramePGO" / "Optimizer.py").read_text(
        encoding="utf-8"
    )
    odometry_source = (root / "Odometry" / "MACVO.py").read_text(encoding="utf-8")
    fields = [
        "init_delta_x",
        "init_delta_y",
        "init_delta_z",
        "init_delta_t_norm",
        "init_delta_R_angle",
        "init_velocity_j_x",
        "init_velocity_j_y",
        "init_velocity_j_z",
        "initial_energy_visual_weighted",
        "energy_visual_change",
        "energy_imu_change",
        "influence_visual_grad_norm",
        "influence_imu_grad_norm",
        "actual_to_visual_step_cosine",
        "actual_to_imu_step_cosine",
    ]
    for field in fields:
        assert f'"{field}": result.{field}' in optimizer_source
        assert f"{field}=opt_diag.get(" in odometry_source


def test_gt_velocity_is_loaded_and_attached_to_odometry():
    root = Path(__file__).resolve().parents[2]
    main_source = (root / "MACVO.py").read_text(encoding="utf-8")
    odometry_source = (root / "Odometry" / "MACVO.py").read_text(encoding="utf-8")

    assert "gt_velocities: dict = {}" in main_source
    assert "gt_angular_velocities: dict = {}" in main_source
    for field in ("vx", "vy", "vz"):
        assert f'_row.get("{field}"' in main_source
    for field in ("wx", "wy", "wz"):
        assert f'_row.get("{field}"' in main_source
    assert "system.set_gt_positions(" in main_source
    assert "gt_velocities=gt_velocities" in main_source
    assert "def set_gt_positions(" in odometry_source
    assert "gt_velocities: dict | None = None" in odometry_source
    assert "self._gt_velocities = gt_velocities" in odometry_source
    assert "gt_angular_velocities: dict | None = None" in odometry_source
    assert "self._gt_angular_velocities = gt_angular_velocities" in odometry_source


def test_local_ba_profile_diagnostics_are_exported_with_switches():
    required = [
        "local_ba_window_size",
        "local_ba_writeback",
        "local_ba_num_frames",
        "local_ba_num_edges",
        "local_ba_num_visual_residual_blocks",
        "local_ba_graph_build_s",
        "local_ba_lm_s",
        "local_ba_refine_s",
        "local_ba_optimize_total_s",
    ]
    for field in required:
        assert field in CSV_HEADER

    assert CSV_HEADER.index("local_ba_window_size") > CSV_HEADER.index("imu_residual_rows")
    assert CSV_HEADER.index("local_ba_optimize_total_s") < CSV_HEADER.index("est_delta_x")


def test_macvo_frame_pair_writer_passes_local_ba_profile_fields():
    root = Path(__file__).resolve().parents[2]
    source = (root / "Odometry" / "MACVO.py").read_text(encoding="utf-8")
    required = [
        "local_ba_window_size",
        "local_ba_writeback",
        "local_ba_num_frames",
        "local_ba_num_edges",
        "local_ba_num_visual_residual_blocks",
        "local_ba_graph_build_s",
        "local_ba_lm_s",
        "local_ba_refine_s",
        "local_ba_optimize_total_s",
    ]
    for field in required:
        assert f"{field}=opt_diag.get(" in source


def test_imu_metadata_convention_diagnostics_are_exported_with_imu_fields():
    required = [
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
        "imu_vio_gravity_pose_source",
        "imu_attitude_source_active",
        "imu_attitude_source_angle_to_est_rad",
        "imu_gravity_rp_active",
        "imu_gravity_rp_angle_rad",
        "imu_gravity_rp_acc_norm",
    ]
    for field in required:
        assert field in CSV_HEADER

    assert CSV_HEADER.index("imu_source_world_frame") > CSV_HEADER.index("imu_noise_source")
    assert CSV_HEADER.index("imu_time_offset_source") > CSV_HEADER.index("imu_timestamp_unit")
    assert CSV_HEADER.index("imu_preintegration_gravity_z") < CSV_HEADER.index("delta_p_over_est")
    assert CSV_HEADER.index("imu_attitude_source_angle_to_est_rad") < CSV_HEADER.index("delta_p_over_est")
    assert CSV_HEADER.index("imu_gravity_rp_acc_norm") < CSV_HEADER.index("delta_p_over_est")


def test_writer_rotates_incompatible_existing_header(tmp_path):
    path = tmp_path / "frame_pair_diagnostics.csv"
    legacy_header = [field for field in CSV_HEADER if field != "vio_factor_active"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(legacy_header)
        writer.writerow(["legacy" for _ in legacy_header])

    diag_writer = FramePairDiagnosticsWriter(path)
    diag_writer.write_row(pair_id=1, imu_factor_mode="preintegrated_vio", vio_factor_active=1, imu_residual_rows=3)
    diag_writer.close()

    with path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))

    assert rows[0] == CSV_HEADER
    assert rows[1][CSV_HEADER.index("vio_factor_active")] == "1"
    backups = list(tmp_path.glob("frame_pair_diagnostics.csv.legacy*"))
    assert len(backups) == 1
