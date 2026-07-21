"""
FramePairDiagnosticsWriter: appends per-frame-pair diagnostic scalars to a CSV file.

Writes alongside poses.csv in the same result directory.
All fields are scalars; missing values are written as empty (interpreted as NaN).
"""

from __future__ import annotations

import csv
from pathlib import Path


CSV_HEADER = [
    # ── Identity ──
    "scene", "method", "pair_id", "frame_i", "frame_j",
    "timestamp_i", "timestamp_j", "dt",
    # ── Switches ──
    "use_imu_rotation", "use_imu_translation", "autodiff_enabled",
    "imu_factor_mode", "vio_factor_active", "imu_residual_rows",
    "local_ba_window_size", "local_ba_writeback",
    "local_ba_num_frames", "local_ba_num_edges", "local_ba_num_visual_residual_blocks",
    "local_ba_graph_build_s", "local_ba_lm_s", "local_ba_refine_s", "local_ba_optimize_total_s",
    "two_state_solver_converged", "two_state_solver_iterations",
    "two_state_solver_convergence_reason", "two_state_final_step_norm",
    "two_state_final_gradient_inf_norm", "two_state_solver_accepted_steps",
    "two_state_solver_rejected_steps",
    # ── Estimated inter-frame motion ──
    "est_delta_x", "est_delta_y", "est_delta_z",
    "est_delta_t_norm", "est_delta_R_angle",
    # Optimizer initial state before LM. It may include configured IMU attitude
    # initialization, so these columns are deliberately named init, not visual.
    "init_delta_x", "init_delta_y", "init_delta_z",
    "init_delta_t_norm", "init_delta_R_angle",
    "init_velocity_j_x", "init_velocity_j_y", "init_velocity_j_z",
    # ── GT inter-frame motion (NaN when unavailable) ──
    "gt_delta_t_norm", "gt_delta_R_angle",
    "est_over_gt_translation_ratio", "cos_est_gt_translation",
    "rotation_error_angle",
    "init_over_gt_translation_ratio", "cos_init_gt_translation",
    "init_rotation_error_angle", "imu_rotation_error_angle",
    # GT and optimized latent velocity in the optimizer's internal world frame.
    "velocity_reference_point",
    "gt_velocity_i_x", "gt_velocity_i_y", "gt_velocity_i_z",
    "gt_velocity_j_x", "gt_velocity_j_y", "gt_velocity_j_z",
    "gt_delta_velocity_norm",
    "est_velocity_j_x", "est_velocity_j_y", "est_velocity_j_z",
    "init_velocity_error_norm", "est_velocity_error_norm",
    # ── IMU preintegration quantities ──
    "imu_delta_R_angle", "imu_delta_p_x", "imu_delta_p_y", "imu_delta_p_z",
    "imu_delta_p_norm", "imu_delta_v_norm",
    "num_imu_samples", "imu_dt",
    "imu_trans_prior_mode",
    "imu_trans_prior_std_x", "imu_trans_prior_std_y", "imu_trans_prior_std_z",
    "imu_noise_sigma_unit", "imu_noise_source",
    "imu_source_world_frame", "imu_source_measurement_frame",
    "imu_internal_world_frame", "imu_internal_measurement_frame",
    "imu_acc_unit", "imu_gyro_unit", "imu_timestamp_unit",
    "imu_time_offset_ns", "imu_time_offset_source",
    "imu_gravity_source", "imu_metadata_gravity_m_s2",
    "imu_preintegration_gravity_z",
    "imu_vio_gravity_pose_source",
    "imu_vio_gravity_handling",
    "imu_vio_gravity_world_x", "imu_vio_gravity_world_y", "imu_vio_gravity_world_z",
    "imu_attitude_source_active", "imu_attitude_source_angle_to_est_rad",
    "imu_gravity_rp_active",
    "imu_gravity_rp_angle_rad", "imu_gravity_rp_acc_norm",
    # ── Δp_IMU vs est / GT ──
    "delta_p_over_est", "delta_p_over_gt",
    "cos_delta_p_est", "cos_delta_p_gt",
    # ── Visual minimal stats ──
    "num_valid_points", "num_visual_residuals",
    "median_flow_cov", "median_depth_cov",
    "visual_input_sha256",
    "visual_loss_raw_sum", "visual_loss_per_residual",
    # ── Frontend covariance stats (from flow + depth keypoint retrieval) ──
    "median_flow_u_cov", "p90_flow_u_cov", "mean_flow_u_cov",
    "median_flow_v_cov", "p90_flow_v_cov", "mean_flow_v_cov",
    "median_kp0_depth_cov", "p90_kp0_depth_cov", "mean_kp0_depth_cov",
    "median_kp1_depth_cov", "p90_kp1_depth_cov", "mean_kp1_depth_cov",
    "valid_depth_ratio", "num_selected_keypoints",
    # ── IMU residual / loss ──
    "r_R_whitened_norm", "r_p_whitened_norm", "r_v_whitened_norm",
    "imu_vio_whitened_norm", "imu_vio_raw_norm",
    "imu_rot_loss", "imu_trans_loss", "imu_vel_loss",
    "imu_vio_cov_trace", "imu_vio_weight_trace",
    "imu_vio_weight_diag_min", "imu_vio_weight_diag_max",
    "imu_vio_sa_v2_sampling_noise_cost",
    "imu_vio_sa_v2_cross_covariance_frobenius_norm",
    "imu_vio_sa_v2_incoming_sample_count",
    "imu_vio_sa_v2_outgoing_sample_count",
    "imu_vio_sa_v2_prior_reset",
    "imu_vio_sa_v2_unique_cov_min_eigenvalue",
    "imu_vio_sa_v2_unique_cov_max_eigenvalue",
    "imu_vio_sa_v2_unique_cov_effective_rank",
    "imu_vio_sa_v2_unique_cov_dimension",
    "imu_vio_sa_v2_unique_cov_condition_number",
    "imu_vio_sa_v2_prior_i_min_eigenvalue",
    "imu_vio_sa_v2_prior_i_max_eigenvalue",
    "imu_vio_sa_v2_prior_i_effective_rank",
    "imu_vio_sa_v2_prior_i_dimension",
    "imu_vio_sa_v2_prior_i_condition_number",
    "imu_vio_sa_v2_h_mm_min_eigenvalue",
    "imu_vio_sa_v2_h_mm_max_eigenvalue",
    "imu_vio_sa_v2_h_mm_effective_rank",
    "imu_vio_sa_v2_h_mm_dimension",
    "imu_vio_sa_v2_h_mm_condition_number",
    "imu_vio_sa_v2_prior_j_min_eigenvalue",
    "imu_vio_sa_v2_prior_j_max_eigenvalue",
    "imu_vio_sa_v2_prior_j_effective_rank",
    "imu_vio_sa_v2_prior_j_dimension",
    "imu_vio_sa_v2_prior_j_condition_number",
    "imu_vio_sa_v2_discarded_h_mm_dimensions",
    "imu_vio_sa_v2_discarded_prior_dimensions",
    "imu_vio_sa_v2_schur_quadratic_relative_error",
    "imu_vio_sa_v2_state_i_translation_update_norm",
    "imu_vio_sa_v2_state_i_rotation_update_norm",
    "imu_vio_sa_v2_state_j_translation_update_norm",
    "imu_vio_sa_v2_state_j_rotation_update_norm",
    "imu_vio_sa_v2_common_translation_update_world_x",
    "imu_vio_sa_v2_common_translation_update_world_y",
    "imu_vio_sa_v2_common_translation_update_world_z",
    "imu_vio_sa_v2_common_translation_update_world_norm",
    "imu_vio_sa_v2_differential_translation_update_world_norm",
    "imu_vio_sa_v2_rank_aware_imu_whitening",
    "imu_vio_sa_v2_rank_aware_fallback_active",
    "imu_vio_sa_v2_imu_residual_dimension",
    "imu_vio_acc_bias_norm", "imu_vio_gyro_bias_norm",
    "imu_vio_acc_bias_x", "imu_vio_acc_bias_y", "imu_vio_acc_bias_z",
    "imu_vio_gyro_bias_x", "imu_vio_gyro_bias_y", "imu_vio_gyro_bias_z",
    "imu_vio_alpha_p", "imu_vio_alpha_v", "imu_vio_alpha_R",
    # Opt-in causal audit: graph state before LM, net state update, and
    # linearized visual/IMU influence. Empty when the audit is disabled.
    "initial_energy_visual_weighted",
    "initial_energy_p_weighted", "initial_energy_v_weighted", "initial_energy_R_weighted",
    "initial_energy_pv_weighted", "initial_energy_imu_diag_weighted", "initial_energy_imu_weighted",
    "initial_energy_imu_to_visual_ratio",
    "initial_energy_pv_to_visual_ratio",
    "initial_energy_R_to_visual_ratio",
    "initial_total_loss",
    "update_pose_translation_norm", "update_pose_rotation_norm",
    "update_velocity_norm", "update_acc_bias_norm", "update_gyro_bias_norm",
    "influence_visual_grad_norm", "influence_imu_grad_norm", "influence_grad_cosine",
    "influence_visual_hessian_trace", "influence_imu_hessian_trace",
    "influence_imu_to_visual_grad_ratio", "influence_imu_to_visual_hessian_ratio",
    "influence_p_grad_norm", "influence_v_grad_norm", "influence_R_grad_norm",
    "influence_sampled",
    "energy_visual_change", "energy_imu_change",
    "energy_p_change", "energy_v_change", "energy_R_change",
    "counterfactual_visual_step_norm", "counterfactual_imu_step_norm", "counterfactual_full_step_norm",
    "counterfactual_visual_to_imu_cosine",
    "actual_to_visual_step_cosine", "actual_to_imu_step_cosine", "actual_to_full_step_cosine",
    "predicted_visual_change_on_actual_step", "predicted_imu_change_on_actual_step",
    "energy_visual_weighted",
    "energy_p_weighted", "energy_v_weighted", "energy_R_weighted",
    "energy_pv_weighted", "energy_imu_diag_weighted", "energy_imu_weighted",
    "energy_imu_to_visual_ratio",
    "energy_pv_to_visual_ratio",
    "energy_R_to_visual_ratio",
    "total_loss",
    # ── Loss ratios ──
    "visual_loss_ratio", "imu_rot_loss_ratio", "imu_trans_loss_ratio", "imu_vel_loss_ratio",
    # ── Coordinate frame annotations ──
    "est_pose_frame", "gt_pose_frame", "imu_meas_frame", "imu_delta_frame",
    # ── Adaptive mode annotations ──
    "adaptive_mode", "adaptive_use_rotation", "adaptive_use_translation",
    "adaptive_reason", "visual_health_score", "degeneracy_score", "motion_abnormal_score",
]


class FramePairDiagnosticsWriter:
    """Append-only CSV writer for per-frame-pair diagnostic scalars."""

    def __init__(self, filepath: Path, scene: str = "", method: str = "") -> None:
        self._filepath = filepath
        self._scene = scene
        self._method = method
        self._rotate_incompatible_existing_file()
        self._needs_header = not filepath.exists()
        self._fh = open(filepath, "a", newline="", encoding="utf-8")
        self._writer = csv.writer(self._fh)
        if self._needs_header:
            self._writer.writerow(CSV_HEADER)
            self._fh.flush()

    def _rotate_incompatible_existing_file(self) -> None:
        if not self._filepath.exists():
            return
        try:
            with self._filepath.open("r", newline="", encoding="utf-8") as f:
                header = next(csv.reader(f), [])
        except Exception:
            return
        if header == CSV_HEADER:
            return

        idx = 0
        while True:
            suffix = ".legacy" if idx == 0 else f".legacy{idx}"
            backup = self._filepath.with_name(self._filepath.name + suffix)
            if not backup.exists():
                self._filepath.rename(backup)
                return
            idx += 1

    # ── write one row ──────────────────────────────────────────────────────
    def write_row(self, **kwargs: float | int | str | None) -> None:
        row: list[str] = []
        for col in CSV_HEADER:
            val = kwargs.get(col)
            if val is None:
                row.append("")
            elif isinstance(val, float):
                row.append(f"{val:.9g}")
            elif isinstance(val, int):
                row.append(str(val))
            else:
                row.append(str(val))
        self._writer.writerow(row)

    def flush(self) -> None:
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()

    @property
    def scene(self) -> str:
        return self._scene

    @property
    def method(self) -> str:
        return self._method
