import torch
import pypose as pp
import typing as T
from dataclasses import dataclass

from Module.Map import MatchObs, PointNode
from Utility.Point import pixel2point_NED, point2pixel_NED
from Utility.IMUKinematics import (
    build_weight_matrix_from_covariances,
    vio_bias_random_walk_residual,
    vio_preintegrated_covariance_blocks,
    vio_preintegrated_covariance_matrix,
    vio_preintegrated_imu_residual,
)
from ..PyposeOptimizers import AnalyticModule, FactorGraph


@dataclass
class GraphInput:
    frame_idx         : torch.Tensor
    from_idx          : torch.Tensor
    init_motion       : pp.LieTensor
    from_pose         : pp.LieTensor
    baseline          : torch.Tensor
    observations      : MatchObs
    points            : PointNode
    images_intrinsic  : torch.Tensor
    edges_index       : torch.Tensor
    device            : str
    imu_rotvec_prior  : torch.Tensor | None = None
    imu_rot_prior_std : torch.Tensor | None = None
    # Translation prior from IMU preintegration: delta_p in from_pose body frame
    imu_trans_prior   : torch.Tensor | None = None   # (3,) float32
    imu_trans_cov     : torch.Tensor | None = None   # (3,3) float32
    # Mean visual observation covariance (scalar, from obs2_covTc trace mean)
    # Used for adaptive visual-IMU fusion weight instead of fixed constant
    visual_obs_cov_mean: float | None = None
    # Number of valid visual observations (matched keypoints) for quality assessment
    num_observations: int = 0
    # Online visual geometry quality signals used by adaptive post-fusion.
    visual_keypoint_coverage: float | None = None
    visual_depth_spread: float | None = None
    # Optional compressed MACVO pose factor. T_CiCj maps a point from the
    # current camera frame Cj into the previous camera frame Ci.
    visual_relative_pose_CiCj: torch.Tensor | None = None
    visual_relative_pose_cov: torch.Tensor | None = None
    visual_relative_pose_num_points: int = 0
    visual_relative_pose_num_inliers: int = 0
    visual_relative_pose_mean_mahalanobis_sq: float | None = None
    # Optional local compression of the native UVD objective. The normal
    # equations use a right perturbation of T_CjCi ordered as [t, r].
    visual_compressed_uvd_reference_CjCi: torch.Tensor | None = None
    visual_compressed_uvd_hessian: torch.Tensor | None = None
    visual_compressed_uvd_gradient: torch.Tensor | None = None
    visual_compressed_uvd_robust_cost: float | None = None
    visual_compressed_uvd_num_points: int = 0
    visual_compressed_uvd_num_inliers: int = 0
    visual_compressed_uvd_mean_mahalanobis_sq: float | None = None
    visual_compressed_uvd_huber_delta: float | None = None
    # Optional deployable VIO IMU factor. Visual residuals still come from MACVO;
    # these fields only add a standard preintegrated inertial residual.
    imu_vio_factor_enable: bool = False
    imu_vio_prev_velocity_world: torch.Tensor | None = None
    imu_vio_curr_velocity_init_world: torch.Tensor | None = None
    imu_vio_prev_acc_bias: torch.Tensor | None = None
    imu_vio_prev_gyro_bias: torch.Tensor | None = None
    imu_vio_curr_acc_bias_init: torch.Tensor | None = None
    imu_vio_curr_gyro_bias_init: torch.Tensor | None = None
    imu_vio_linearized_acc_bias: torch.Tensor | None = None
    imu_vio_linearized_gyro_bias: torch.Tensor | None = None
    imu_vio_bias_jacobian: torch.Tensor | None = None
    imu_vio_bias_rw_cov: torch.Tensor | None = None
    imu_vio_delta_rotvec: torch.Tensor | None = None
    imu_vio_delta_v: torch.Tensor | None = None
    imu_vio_delta_p: torch.Tensor | None = None
    imu_vio_cov: torch.Tensor | None = None
    imu_vio_sa_v2_unique_cov: torch.Tensor | None = None
    imu_vio_sa_v2_incoming_raw_time_ns: torch.Tensor | None = None
    imu_vio_sa_v2_outgoing_raw_time_ns: torch.Tensor | None = None
    imu_vio_sa_v2_incoming_sensitivity: torch.Tensor | None = None
    imu_vio_sa_v2_outgoing_sensitivity: torch.Tensor | None = None
    imu_vio_dt: torch.Tensor | None = None
    imu_vio_sensor_T_imu: torch.Tensor | None = None
    imu_vio_gravity_world: torch.Tensor | None = None
    imu_vio_gravity_in_residual: bool = False
    imu_vio_alpha_p: float = 1.0
    imu_vio_alpha_v: float = 1.0
    imu_vio_alpha_R: float = 1.0


@dataclass
class GraphOutput:
    motion   : torch.Tensor
    from_idx : torch.Tensor
    frame_idx: torch.Tensor
    visual_obs_cov_mean: float | None = None
    num_observations: int = 0
    visual_keypoint_coverage: float | None = None
    visual_depth_spread: float | None = None
    visual_pose_inlier_ratio: float | None = None
    visual_pose_mean_mahalanobis_sq: float | None = None
    visual_pose_whitened_residual_norm: float | None = None
    visual_pose_covariance_inflation: float | None = None
    visual_pose_gate_action: str | None = None
    velocity_world: torch.Tensor | None = None
    acc_bias: torch.Tensor | None = None
    gyro_bias: torch.Tensor | None = None
    # ── Per-pair diagnostics (populated by _optimize) ──
    final_loss: float | None = None
    visual_loss: float | None = None
    imu_rot_loss: float | None = None
    imu_vel_loss: float | None = None
    imu_trans_loss: float | None = None
    r_R_whitened_norm: float | None = None
    r_v_whitened_norm: float | None = None
    r_p_whitened_norm: float | None = None
    imu_vio_whitened_norm: float | None = None
    imu_vio_raw_norm: float | None = None
    imu_vio_cov_trace: float | None = None
    imu_vio_weight_trace: float | None = None
    imu_vio_weight_diag_min: float | None = None
    imu_vio_weight_diag_max: float | None = None
    imu_vio_sa_v2_sampling_noise_cost: float | None = None
    imu_vio_sa_v2_cross_covariance_frobenius_norm: float | None = None
    imu_vio_sa_v2_incoming_sample_count: int | None = None
    imu_vio_sa_v2_outgoing_sample_count: int | None = None
    imu_vio_sa_v2_prior_reset: bool | None = None
    imu_vio_sa_v2_unique_cov_min_eigenvalue: float | None = None
    imu_vio_sa_v2_unique_cov_max_eigenvalue: float | None = None
    imu_vio_sa_v2_unique_cov_effective_rank: int | None = None
    imu_vio_sa_v2_unique_cov_dimension: int | None = None
    imu_vio_sa_v2_unique_cov_condition_number: float | None = None
    imu_vio_sa_v2_prior_i_min_eigenvalue: float | None = None
    imu_vio_sa_v2_prior_i_max_eigenvalue: float | None = None
    imu_vio_sa_v2_prior_i_effective_rank: int | None = None
    imu_vio_sa_v2_prior_i_dimension: int | None = None
    imu_vio_sa_v2_prior_i_condition_number: float | None = None
    imu_vio_sa_v2_h_mm_min_eigenvalue: float | None = None
    imu_vio_sa_v2_h_mm_max_eigenvalue: float | None = None
    imu_vio_sa_v2_h_mm_effective_rank: int | None = None
    imu_vio_sa_v2_h_mm_dimension: int | None = None
    imu_vio_sa_v2_h_mm_condition_number: float | None = None
    imu_vio_sa_v2_prior_j_min_eigenvalue: float | None = None
    imu_vio_sa_v2_prior_j_max_eigenvalue: float | None = None
    imu_vio_sa_v2_prior_j_effective_rank: int | None = None
    imu_vio_sa_v2_prior_j_dimension: int | None = None
    imu_vio_sa_v2_prior_j_condition_number: float | None = None
    imu_vio_sa_v2_discarded_h_mm_dimensions: int | None = None
    imu_vio_sa_v2_discarded_prior_dimensions: int | None = None
    imu_vio_sa_v2_schur_quadratic_relative_error: float | None = None
    imu_vio_sa_v2_state_i_translation_update_norm: float | None = None
    imu_vio_sa_v2_state_i_rotation_update_norm: float | None = None
    imu_vio_sa_v2_state_j_translation_update_norm: float | None = None
    imu_vio_sa_v2_state_j_rotation_update_norm: float | None = None
    imu_vio_sa_v2_common_translation_update_world_x: float | None = None
    imu_vio_sa_v2_common_translation_update_world_y: float | None = None
    imu_vio_sa_v2_common_translation_update_world_z: float | None = None
    imu_vio_sa_v2_common_translation_update_world_norm: float | None = None
    imu_vio_sa_v2_differential_translation_update_world_norm: float | None = None
    imu_vio_sa_v2_rank_aware_imu_whitening: bool | None = None
    imu_vio_sa_v2_rank_aware_fallback_active: bool | None = None
    imu_vio_sa_v2_imu_residual_dimension: int | None = None
    imu_vio_acc_bias_norm: float | None = None
    imu_vio_gyro_bias_norm: float | None = None
    imu_vio_acc_bias_x: float | None = None
    imu_vio_acc_bias_y: float | None = None
    imu_vio_acc_bias_z: float | None = None
    imu_vio_gyro_bias_x: float | None = None
    imu_vio_gyro_bias_y: float | None = None
    imu_vio_gyro_bias_z: float | None = None
    imu_vio_alpha_p: float = 1.0
    imu_vio_alpha_v: float = 1.0
    imu_vio_alpha_R: float = 1.0
    # Opt-in causal diagnostics. Initial fields describe the graph state
    # before LM; update fields describe the net optimizer state change.
    initial_motion: torch.Tensor | None = None
    init_delta_x: float | None = None
    init_delta_y: float | None = None
    init_delta_z: float | None = None
    init_delta_t_norm: float | None = None
    init_delta_R_angle: float | None = None
    init_velocity_j_x: float | None = None
    init_velocity_j_y: float | None = None
    init_velocity_j_z: float | None = None
    initial_energy_visual_weighted: float | None = None
    initial_energy_p_weighted: float | None = None
    initial_energy_v_weighted: float | None = None
    initial_energy_R_weighted: float | None = None
    initial_energy_pv_weighted: float | None = None
    initial_energy_imu_diag_weighted: float | None = None
    initial_energy_imu_weighted: float | None = None
    initial_energy_imu_to_visual_ratio: float | None = None
    initial_energy_pv_to_visual_ratio: float | None = None
    initial_energy_R_to_visual_ratio: float | None = None
    initial_total_loss: float | None = None
    update_pose_translation_norm: float | None = None
    update_pose_rotation_norm: float | None = None
    update_velocity_norm: float | None = None
    update_acc_bias_norm: float | None = None
    update_gyro_bias_norm: float | None = None
    # Exact visual/full-IMU linearized influence plus diagnostic IMU blocks.
    influence_visual_grad_norm: float | None = None
    influence_imu_grad_norm: float | None = None
    influence_grad_cosine: float | None = None
    influence_visual_hessian_trace: float | None = None
    influence_imu_hessian_trace: float | None = None
    influence_imu_to_visual_grad_ratio: float | None = None
    influence_imu_to_visual_hessian_ratio: float | None = None
    influence_p_grad_norm: float | None = None
    influence_v_grad_norm: float | None = None
    influence_R_grad_norm: float | None = None
    influence_sampled: int = 0
    energy_visual_change: float | None = None
    energy_imu_change: float | None = None
    energy_p_change: float | None = None
    energy_v_change: float | None = None
    energy_R_change: float | None = None
    counterfactual_visual_step_norm: float | None = None
    counterfactual_imu_step_norm: float | None = None
    counterfactual_full_step_norm: float | None = None
    counterfactual_visual_to_imu_cosine: float | None = None
    actual_to_visual_step_cosine: float | None = None
    actual_to_imu_step_cosine: float | None = None
    actual_to_full_step_cosine: float | None = None
    predicted_visual_change_on_actual_step: float | None = None
    predicted_imu_change_on_actual_step: float | None = None
    energy_visual_weighted: float | None = None
    energy_p_weighted: float | None = None
    energy_v_weighted: float | None = None
    energy_R_weighted: float | None = None
    energy_pv_weighted: float | None = None
    energy_imu_diag_weighted: float | None = None
    energy_imu_weighted: float | None = None
    energy_imu_to_visual_ratio: float | None = None
    energy_pv_to_visual_ratio: float | None = None
    energy_R_to_visual_ratio: float | None = None
    num_visual_residuals: int = 0
    imu_factor_mode: str = "legacy_pose_prior"
    vio_factor_active: bool = False
    vio_bias_state_active: bool = False
    imu_residual_rows: int = 0
    use_imu_rotation: bool = False
    use_imu_translation: bool = False
    window_frame_indices: torch.Tensor | None = None
    window_motions: torch.Tensor | None = None
    window_velocity_world: torch.Tensor | None = None
    window_acc_bias: torch.Tensor | None = None
    window_gyro_bias: torch.Tensor | None = None
    local_ba_window_size: int = 0
    local_ba_writeback: str = "current"
    local_ba_num_frames: int = 0
    local_ba_num_edges: int = 0
    local_ba_num_visual_residual_blocks: int = 0
    local_ba_graph_build_s: float | None = None
    local_ba_lm_s: float | None = None
    local_ba_refine_s: float | None = None
    local_ba_optimize_total_s: float | None = None
    two_state_solver_converged: bool | None = None
    two_state_solver_iterations: int | None = None
    two_state_solver_convergence_reason: str | None = None
    two_state_final_step_norm: float | None = None
    two_state_final_gradient_inf_norm: float | None = None
    two_state_solver_accepted_steps: int | None = None
    two_state_solver_rejected_steps: int | None = None
    vio_backend: str = "two_state"
    isam2_update_ms: float | None = None
    isam2_state_count: int | None = None
    isam2_history_revision: bool = False
    isam2_initial_pose_mismatch_norm: float | None = None
    isam2_initial_velocity_mismatch_norm: float | None = None
    isam2_initial_bias_mismatch_norm: float | None = None
    debug_trace: dict[str, T.Any] | None = None


@dataclass
class LocalWindowGraphInput:
    frame_indices: torch.Tensor
    frame_poses: torch.Tensor | pp.LieTensor
    edges: list[GraphInput]
    fixed_first_frame: bool = True
    writeback: str = "current"
    device: str = "cpu"


############## Optimization Graphs

def _skew3(vec: torch.Tensor) -> torch.Tensor:
    vec = vec.reshape(3)
    zero = torch.zeros((), device=vec.device, dtype=vec.dtype)
    x, y, z = vec[0], vec[1], vec[2]
    return torch.stack(
        [
            torch.stack([zero, -z, y]),
            torch.stack([z, zero, -x]),
            torch.stack([-y, x, zero]),
        ]
    )


def _so3_right_jacobian_inverse(phi: torch.Tensor) -> torch.Tensor:
    """Right Jacobian inverse for Log(R_prior^-1 R_est) wrt local SO(3) perturbation."""
    phi = phi.reshape(3)
    eye = torch.eye(3, device=phi.device, dtype=phi.dtype)
    skew = _skew3(phi)
    skew2 = skew @ skew
    theta = torch.linalg.norm(phi)
    if float(theta.detach().cpu()) < 1e-8:
        return eye + 0.5 * skew + (1.0 / 12.0) * skew2
    coeff = (1.0 / (theta * theta)) - ((1.0 + torch.cos(theta)) / (2.0 * theta * torch.sin(theta)))
    return eye + 0.5 * skew + coeff * skew2


class ICP_TwoframePGO(FactorGraph):
    def __init__(self, graph_data: GraphInput) -> None:
        super().__init__()
        self.device                = graph_data.device
        self.init_motion           = graph_data.init_motion
        self.from_idx              = graph_data.from_idx
        self.frame_idx             = graph_data.frame_idx
        
        self.pose2opt       = pp.Parameter(pp.SE3(self.init_motion))
        self.edges_index    = graph_data.edges_index
        
        # ICP-based residual
        self.pts = graph_data.points
        self.obs = graph_data.observations
        
        self.register_buffer("K", graph_data.images_intrinsic)
        self.register_buffer("points_Tc",
            pixel2point_NED(self.obs.data["pixel2_uv"], self.obs.data["pixel2_d"].squeeze(-1), graph_data.images_intrinsic)
        )
        self.points_Tc: torch.Tensor
        self.register_buffer("points_Tw", self.pts.data["pos_Tw"])
        self.register_buffer("obs_covTc", self.obs.data["obs2_covTc"])
        self.register_buffer("pts_covTw", self.pts.data["cov_Tw"])
        self.register_buffer("from_pose", graph_data.from_pose.tensor())
        if graph_data.imu_vio_sensor_T_imu is not None and graph_data.imu_vio_sensor_T_imu.numel() == 7:
            sensor_T_imu = graph_data.imu_vio_sensor_T_imu.reshape(1, 7).float()
        else:
            sensor_T_imu = pp.identity_SE3(1, dtype=torch.float32).tensor()
        self.register_buffer("imu_sensor_T_imu", sensor_T_imu)

        self.use_imu_rot_prior = (
            graph_data.imu_rotvec_prior is not None and graph_data.imu_rot_prior_std is not None
        )
        if self.use_imu_rot_prior:
            self.register_buffer("imu_rotvec_prior", graph_data.imu_rotvec_prior.reshape(3))
            sigma = float(graph_data.imu_rot_prior_std.reshape(-1)[0].item())
            self.register_buffer("imu_rot_cov", torch.eye(3) * (sigma ** 2))

        self.use_imu_trans_prior = (
            graph_data.imu_trans_prior is not None and graph_data.imu_trans_cov is not None
        )
        if self.use_imu_trans_prior:
            self.register_buffer("imu_trans_prior", graph_data.imu_trans_prior.reshape(3))
            self.register_buffer("imu_trans_cov", graph_data.imu_trans_cov.reshape(3, 3))

        self._visual_obs_cov_mean: float | None = graph_data.visual_obs_cov_mean
        self._num_observations: int = getattr(graph_data, "num_observations", 0)
        self._visual_keypoint_coverage: float | None = graph_data.visual_keypoint_coverage
        self._visual_depth_spread: float | None = graph_data.visual_depth_spread

    def forward(self) -> torch.Tensor:
        frame_pose = T.cast(pp.LieTensor, self.pose2opt[self.edges_index])
        residual = frame_pose.Act(self.points_Tc) - self.points_Tw
        if self.use_imu_rot_prior:
            residual = torch.cat([residual, self._imu_rot_residual().to(residual)], dim=0)
        if self.use_imu_trans_prior:
            residual = torch.cat([residual, self._imu_trans_residual().to(residual)], dim=0)
        return residual

    def _imu_rot_residual(self) -> torch.Tensor:
        rel_est = self._imu_relative_estimate()
        rot_est = rel_est.rotation()
        rot_prior = pp.so3(self.imu_rotvec_prior).Exp()
        rot_err = rot_prior.Inv() @ rot_est
        return rot_err.Log().tensor().reshape(1, 3)

    def _imu_trans_residual(self) -> torch.Tensor:
        rel_est = self._imu_relative_estimate()
        trans_est = rel_est.translation()
        return (trans_est - self.imu_trans_prior).reshape(1, 3)

    def _imu_relative_estimate(self) -> pp.LieTensor:
        from_pose = pp.SE3(self.from_pose)
        sensor_T_imu = pp.SE3(self.imu_sensor_T_imu).to(
            device=self.pose2opt.tensor().device,
            dtype=self.pose2opt.tensor().dtype,
        )
        from_imu = from_pose @ sensor_T_imu
        to_imu = self.pose2opt @ sensor_T_imu
        return from_imu.Inv() @ to_imu

    @torch.no_grad()
    @torch.inference_mode()
    def covariance_array(self) -> torch.Tensor:
        frame_pose = T.cast(pp.LieTensor, self.pose2opt[self.edges_index])
        R  = frame_pose.rotation().matrix()
        RT = R.transpose(-2, -1)
        cov = (R @ self.obs_covTc @ RT) + self.pts_covTw # type: ignore
        if self.use_imu_rot_prior:
            cov = torch.cat([cov, self.imu_rot_cov.unsqueeze(0).to(cov)], dim=0)
        if self.use_imu_trans_prior:
            cov = torch.cat([cov, self.imu_trans_cov.unsqueeze(0).to(cov)], dim=0)
        return cov

    @torch.no_grad()
    @torch.inference_mode()
    def write_back(self) -> GraphOutput:
        return GraphOutput(motion=self.pose2opt, frame_idx=self.frame_idx, from_idx=self.from_idx,
                           visual_obs_cov_mean=getattr(self, "_visual_obs_cov_mean", None),
                           num_observations=getattr(self, "_num_observations", 0),
                           visual_keypoint_coverage=getattr(self, "_visual_keypoint_coverage", None),
                           visual_depth_spread=getattr(self, "_visual_depth_spread", None))


class Reproj_TwoFramePGO(FactorGraph):
    def __init__(self, graph_data: GraphInput) -> None:
        super().__init__()
        self.from_idx : torch.Tensor = graph_data.from_idx
        self.frame_idx: torch.Tensor = graph_data.frame_idx
        self.init_motion:  pp.LieTensor = graph_data.init_motion
        
        self.pose2opt       = pp.Parameter(pp.SE3(self.init_motion))
        self.edges_index    = graph_data.edges_index
        
        self.pts     = graph_data.points
        self.obs     = graph_data.observations

        self.pos_Tc: torch.Tensor
        self.pos_Tw: torch.Tensor
        self.K: torch.Tensor
        self.register_buffer("K", graph_data.images_intrinsic)
        self.register_buffer("pos_Tw" , self.pts.data["pos_Tw"])
        self.register_buffer("cov_Tw" , self.pts.data["cov_Tw"])
        self.register_buffer("kp2"    , self.obs.data["pixel2_uv"])
        
        N = self.obs.data["pixel2_uv_cov"].size(0)
        cov_kp2 = torch.empty((N, 2, 2))
        cov_kp2[:, 0, 0] = self.obs.data["pixel2_uv_cov"][:, 0]
        cov_kp2[:, 1, 1] = self.obs.data["pixel2_uv_cov"][:, 1]
        cov_kp2[:, 0, 1] = self.obs.data["pixel2_uv_cov"][:, 2]
        cov_kp2[:, 1, 0] = self.obs.data["pixel2_uv_cov"][:, 2]
        self.register_buffer("cov_kp2", cov_kp2)

        self._visual_obs_cov_mean: float | None = graph_data.visual_obs_cov_mean
        self._num_observations: int = getattr(graph_data, "num_observations", 0)
        self._visual_keypoint_coverage: float | None = graph_data.visual_keypoint_coverage
        self._visual_depth_spread: float | None = graph_data.visual_depth_spread

    def forward(self) -> torch.Tensor:
        self.pos_Tc = self.pose2opt.Inv().Act(self.pos_Tw)
        return point2pixel_NED(self.pos_Tc, self.K) - self.kp2

    @torch.no_grad()
    @torch.inference_mode()
    def covariance_array(self) -> torch.Tensor:
        return T.cast(torch.Tensor, self.cov_kp2)

    @torch.no_grad()
    @torch.inference_mode()
    def write_back(self) -> GraphOutput:
        with torch.no_grad():
            return GraphOutput(motion=self.pose2opt, frame_idx=self.frame_idx, from_idx=self.from_idx,
                               visual_obs_cov_mean=getattr(self, "_visual_obs_cov_mean", None),
                               num_observations=getattr(self, "_num_observations", 0),
                               visual_keypoint_coverage=getattr(self, "_visual_keypoint_coverage", None),
                               visual_depth_spread=getattr(self, "_visual_depth_spread", None))


class ReprojDisp_TwoFramePGO(Reproj_TwoFramePGO):
    def __init__(self, graph_data: GraphInput) -> None:
        super().__init__(graph_data)
        self.register_buffer("from_pose", graph_data.from_pose.tensor())
        if graph_data.imu_vio_sensor_T_imu is not None and graph_data.imu_vio_sensor_T_imu.numel() == 7:
            sensor_T_imu = graph_data.imu_vio_sensor_T_imu.reshape(1, 7).float()
        else:
            sensor_T_imu = pp.identity_SE3(1, dtype=torch.float32).tensor()
        self.register_buffer("imu_sensor_T_imu", sensor_T_imu)

        self.use_vio_imu_factor = bool(
            graph_data.imu_vio_factor_enable
            and graph_data.imu_vio_prev_velocity_world is not None
            and graph_data.imu_vio_curr_velocity_init_world is not None
            and graph_data.imu_vio_delta_rotvec is not None
            and graph_data.imu_vio_delta_v is not None
            and graph_data.imu_vio_delta_p is not None
            and graph_data.imu_vio_cov is not None
            and graph_data.imu_vio_dt is not None
        )
        if self.use_vio_imu_factor:
            self.velocity2opt = torch.nn.Parameter(
                graph_data.imu_vio_curr_velocity_init_world.reshape(3).clone().float()
            )
            self.register_buffer("imu_vio_prev_velocity_world", graph_data.imu_vio_prev_velocity_world.reshape(3).float())
            self.register_buffer("imu_vio_delta_rotvec", graph_data.imu_vio_delta_rotvec.reshape(3).float())
            self.register_buffer("imu_vio_delta_v", graph_data.imu_vio_delta_v.reshape(3).float())
            self.register_buffer("imu_vio_delta_p", graph_data.imu_vio_delta_p.reshape(3).float())
            self.register_buffer("imu_vio_dt", graph_data.imu_vio_dt.reshape(-1)[0:1].float())
            gravity_world = (
                graph_data.imu_vio_gravity_world.reshape(3).float()
                if graph_data.imu_vio_gravity_world is not None
                else torch.zeros(3, dtype=torch.float32)
            )
            self.register_buffer("imu_vio_gravity_world", gravity_world)
            self.imu_vio_gravity_in_residual = bool(graph_data.imu_vio_gravity_in_residual)
            self.register_buffer(
                "imu_vio_cov_blocks",
                vio_preintegrated_covariance_blocks(graph_data.imu_vio_cov.reshape(9, 9).float()),
            )
            self.register_buffer(
                "imu_vio_cov_matrix",
                vio_preintegrated_covariance_matrix(graph_data.imu_vio_cov.reshape(9, 9).float()),
            )
            self.imu_vio_alpha_p = max(float(graph_data.imu_vio_alpha_p), 0.0)
            self.imu_vio_alpha_v = max(float(graph_data.imu_vio_alpha_v), 0.0)
            self.imu_vio_alpha_R = max(float(graph_data.imu_vio_alpha_R), 0.0)
            row_scale = torch.tensor(
                [
                    [self.imu_vio_alpha_p ** 0.5],
                    [self.imu_vio_alpha_v ** 0.5],
                    [self.imu_vio_alpha_R ** 0.5],
                ],
                dtype=torch.float32,
            )
            self.register_buffer("imu_vio_residual_row_scale", row_scale)
            self.use_vio_bias_state = bool(
                graph_data.imu_vio_prev_acc_bias is not None
                and graph_data.imu_vio_prev_gyro_bias is not None
                and graph_data.imu_vio_curr_acc_bias_init is not None
                and graph_data.imu_vio_curr_gyro_bias_init is not None
                and graph_data.imu_vio_linearized_acc_bias is not None
                and graph_data.imu_vio_linearized_gyro_bias is not None
                and graph_data.imu_vio_bias_jacobian is not None
                and graph_data.imu_vio_bias_rw_cov is not None
            )
            if self.use_vio_bias_state:
                self.acc_bias2opt = torch.nn.Parameter(
                    graph_data.imu_vio_curr_acc_bias_init.reshape(3).clone().float()
                )
                self.gyro_bias2opt = torch.nn.Parameter(
                    graph_data.imu_vio_curr_gyro_bias_init.reshape(3).clone().float()
                )
                self.register_buffer("imu_vio_prev_acc_bias", graph_data.imu_vio_prev_acc_bias.reshape(3).float())
                self.register_buffer("imu_vio_prev_gyro_bias", graph_data.imu_vio_prev_gyro_bias.reshape(3).float())
                self.register_buffer(
                    "imu_vio_linearized_acc_bias",
                    graph_data.imu_vio_linearized_acc_bias.reshape(3).float(),
                )
                self.register_buffer(
                    "imu_vio_linearized_gyro_bias",
                    graph_data.imu_vio_linearized_gyro_bias.reshape(3).float(),
                )
                self.register_buffer("imu_vio_bias_jacobian", graph_data.imu_vio_bias_jacobian.reshape(9, 6).float())
                self.register_buffer("imu_vio_bias_rw_cov", graph_data.imu_vio_bias_rw_cov.reshape(6, 6).float())
        else:
            self.use_vio_bias_state = False

        self.use_imu_rot_prior = (
            graph_data.imu_rotvec_prior is not None and graph_data.imu_rot_prior_std is not None
            and not self.use_vio_imu_factor
        )
        if self.use_imu_rot_prior:
            self.register_buffer("imu_rotvec_prior", graph_data.imu_rotvec_prior.reshape(3))
            sigma = float(graph_data.imu_rot_prior_std.reshape(-1)[0].item())
            self.register_buffer("imu_rot_cov", torch.eye(3) * (sigma ** 2))

        self.use_imu_trans_prior = (
            graph_data.imu_trans_prior is not None and graph_data.imu_trans_cov is not None
            and not self.use_vio_imu_factor
        )
        if self.use_imu_trans_prior:
            self.register_buffer("imu_trans_prior", graph_data.imu_trans_prior.reshape(3))
            self.register_buffer("imu_trans_cov", graph_data.imu_trans_cov.reshape(3, 3))

        self.register_buffer("baseline", graph_data.baseline)
        self.baseline: torch.Tensor
        self.register_buffer("kp2_disparity", graph_data.observations.data["pixel2_disp"])

        cov_kp2 = T.cast(torch.Tensor, self.cov_kp2)

        N = cov_kp2.size(0)
        cov = torch.zeros((N, 3, 3))
        cov[:, :2, :2] = cov_kp2
        cov[:, 2, 2] = graph_data.observations.data["pixel2_disp_cov"].squeeze(-1)
        self.register_buffer("cov", cov)

    def forward(self) -> torch.Tensor:
        self.pos_Tc = self.pose2opt.Inv() * self.pos_Tw
        K = T.cast(torch.Tensor, self.K)
        bl = T.cast(torch.Tensor, self.baseline)

        reproj_err = point2pixel_NED(self.pos_Tc, K) - T.cast(torch.Tensor, self.kp2)
        depth_err = (self.pos_Tc[:, 0:1].reciprocal() * (K[0, 0] * bl)) - self.kp2_disparity
        residual = torch.cat((reproj_err, depth_err), dim=-1)
        if self.use_vio_imu_factor:
            residual = torch.cat([residual, self._imu_vio_residual().to(residual)], dim=0)
            if self.use_vio_bias_state:
                residual = torch.cat([residual, self._imu_vio_bias_residual().to(residual)], dim=0)
        else:
            if self.use_imu_rot_prior:
                residual = torch.cat([residual, self._imu_rot_residual().to(residual)], dim=0)
            if self.use_imu_trans_prior:
                residual = torch.cat([residual, self._imu_trans_residual().to(residual)], dim=0)
        return residual

    def _imu_vio_residual(self) -> torch.Tensor:
        residual = vio_preintegrated_imu_residual(
            from_pose=pp.SE3(self.from_pose),
            to_pose=self.pose2opt,
            prev_velocity_world=self.imu_vio_prev_velocity_world,
            curr_velocity_world=self.velocity2opt,
            delta_R=self.imu_vio_delta_rotvec,
            delta_v=self.imu_vio_delta_v,
            delta_p=self.imu_vio_delta_p,
            dt_total=float(self.imu_vio_dt.reshape(-1)[0].detach().cpu().item()),
            prev_acc_bias=self.imu_vio_prev_acc_bias if self.use_vio_bias_state else None,
            prev_gyro_bias=self.imu_vio_prev_gyro_bias if self.use_vio_bias_state else None,
            curr_acc_bias=self.acc_bias2opt if self.use_vio_bias_state else None,
            curr_gyro_bias=self.gyro_bias2opt if self.use_vio_bias_state else None,
            linearized_acc_bias=self.imu_vio_linearized_acc_bias if self.use_vio_bias_state else None,
            linearized_gyro_bias=self.imu_vio_linearized_gyro_bias if self.use_vio_bias_state else None,
            bias_jacobian=self.imu_vio_bias_jacobian if self.use_vio_bias_state else None,
            sensor_T_imu=pp.SE3(self.imu_sensor_T_imu),
            gravity_world=self.imu_vio_gravity_world,
            gravity_handling="residual" if self.imu_vio_gravity_in_residual else "preintegration",
        )
        return residual * self.imu_vio_residual_row_scale.to(residual)

    def _imu_vio_bias_residual(self) -> torch.Tensor:
        return vio_bias_random_walk_residual(
            prev_acc_bias=self.imu_vio_prev_acc_bias,
            prev_gyro_bias=self.imu_vio_prev_gyro_bias,
            curr_acc_bias=self.acc_bias2opt,
            curr_gyro_bias=self.gyro_bias2opt,
        )

    def _imu_rot_residual(self) -> torch.Tensor:
        rel_est = self._imu_relative_estimate()
        rot_est = rel_est.rotation()
        rot_prior = pp.so3(self.imu_rotvec_prior).Exp()
        rot_err = rot_prior.Inv() @ rot_est
        return rot_err.Log().tensor().reshape(1, 3)

    def _imu_trans_residual(self) -> torch.Tensor:
        rel_est = self._imu_relative_estimate()
        trans_est = rel_est.translation()
        return (trans_est - self.imu_trans_prior).reshape(1, 3)

    def _imu_relative_estimate(self) -> pp.LieTensor:
        from_pose = pp.SE3(self.from_pose)
        sensor_T_imu = pp.SE3(self.imu_sensor_T_imu).to(
            device=self.pose2opt.tensor().device,
            dtype=self.pose2opt.tensor().dtype,
        )
        from_imu = from_pose @ sensor_T_imu
        to_imu = self.pose2opt @ sensor_T_imu
        return from_imu.Inv() @ to_imu

    @torch.no_grad()
    @torch.inference_mode()
    def covariance_array(self) -> torch.Tensor:
        cov = T.cast(torch.Tensor, self.cov)
        if self.use_vio_imu_factor:
            cov = torch.cat([cov, self.imu_vio_cov_blocks.to(cov)], dim=0)
            if self.use_vio_bias_state:
                bias_cov = self.imu_vio_bias_rw_cov.reshape(6, 6).to(cov)
                bias_blocks = torch.stack(
                    [
                        bias_cov[0:3, 0:3],
                        bias_cov[3:6, 3:6],
                    ],
                    dim=0,
                )
                cov = torch.cat([cov, bias_blocks], dim=0)
        else:
            if self.use_imu_rot_prior:
                cov = torch.cat([cov, self.imu_rot_cov.unsqueeze(0).to(cov)], dim=0)
            if self.use_imu_trans_prior:
                cov = torch.cat([cov, self.imu_trans_cov.unsqueeze(0).to(cov)], dim=0)
        return cov

    @torch.no_grad()
    @torch.inference_mode()
    def weight_matrix(self) -> torch.Tensor:
        cov = T.cast(torch.Tensor, self.cov)
        if self.use_vio_imu_factor:
            full_covariances: list[torch.Tensor] = [self.imu_vio_cov_matrix.to(cov)]
            if self.use_vio_bias_state:
                full_covariances.append(self.imu_vio_bias_rw_cov.reshape(6, 6).to(cov))
            return build_weight_matrix_from_covariances(
                cov,
                full_covariances=full_covariances,
            )
        return build_weight_matrix_from_covariances(self.covariance_array().to(cov))

    @torch.no_grad()
    @torch.inference_mode()
    def write_back(self) -> GraphOutput:
        output = super().write_back()
        if self.use_vio_imu_factor:
            output.velocity_world = self.velocity2opt.detach().reshape(3).cpu()
            if self.use_vio_bias_state:
                output.acc_bias = self.acc_bias2opt.detach().reshape(3).cpu()
                output.gyro_bias = self.gyro_bias2opt.detach().reshape(3).cpu()
            output.imu_factor_mode = "preintegrated_vio"
            output.vio_factor_active = True
            output.vio_bias_state_active = bool(self.use_vio_bias_state)
            output.imu_residual_rows = 5 if self.use_vio_bias_state else 3
            output.use_imu_rotation = True
            output.use_imu_translation = True
            output.imu_vio_alpha_p = float(self.imu_vio_alpha_p)
            output.imu_vio_alpha_v = float(self.imu_vio_alpha_v)
            output.imu_vio_alpha_R = float(self.imu_vio_alpha_R)
        else:
            output.imu_factor_mode = "legacy_pose_prior"
            output.vio_factor_active = False
            output.vio_bias_state_active = False
            output.imu_residual_rows = int(self.use_imu_rot_prior) + int(self.use_imu_trans_prior)
            output.use_imu_rotation = bool(self.use_imu_rot_prior)
            output.use_imu_translation = bool(self.use_imu_trans_prior)
        return output


class LocalWindowInertialGraph(FactorGraph):
    def __init__(self, graph_data: LocalWindowGraphInput) -> None:
        super().__init__()
        self.frame_indices = graph_data.frame_indices.reshape(-1).long()
        self.edges = [self._coerce_edge(edge) for edge in graph_data.edges]
        self.fixed_first_frame = bool(graph_data.fixed_first_frame)
        self.local_ba_writeback = str(graph_data.writeback)

        frame_poses = pp.SE3(graph_data.frame_poses).tensor().reshape(-1, 7).float()
        if frame_poses.shape[0] != self.frame_indices.numel():
            raise ValueError("frame_poses and frame_indices must have matching lengths")
        if len(self.edges) != max(0, self.frame_indices.numel() - 1):
            raise ValueError("local window expects one adjacent edge per frame transition")

        if self.fixed_first_frame:
            self.register_buffer("fixed_pose0", frame_poses[0:1])
            self.pose_window = pp.Parameter(pp.SE3(frame_poses[1:].clone()))
        else:
            self.register_buffer("fixed_pose0", frame_poses.new_zeros((0, 7)))
            self.pose_window = pp.Parameter(pp.SE3(frame_poses.clone()))

        velocity, acc_bias, gyro_bias = self._initial_vector_states(frame_poses)
        if self.fixed_first_frame:
            self.register_buffer("fixed_velocity0", velocity[0:1])
            self.register_buffer("fixed_acc_bias0", acc_bias[0:1])
            self.register_buffer("fixed_gyro_bias0", gyro_bias[0:1])
            self.velocity_window = torch.nn.Parameter(velocity[1:].clone())
            self.acc_bias_window = torch.nn.Parameter(acc_bias[1:].clone())
            self.gyro_bias_window = torch.nn.Parameter(gyro_bias[1:].clone())
        else:
            self.register_buffer("fixed_velocity0", velocity.new_zeros((0, 3)))
            self.register_buffer("fixed_acc_bias0", acc_bias.new_zeros((0, 3)))
            self.register_buffer("fixed_gyro_bias0", gyro_bias.new_zeros((0, 3)))
            self.velocity_window = torch.nn.Parameter(velocity.clone())
            self.acc_bias_window = torch.nn.Parameter(acc_bias.clone())
            self.gyro_bias_window = torch.nn.Parameter(gyro_bias.clone())

        self._visual_covariances = self._build_visual_covariances()
        self._imu_covariances = [self._edge_imu_cov(edge) for edge in self.edges]
        self._bias_covariances = [self._edge_bias_cov(edge) for edge in self.edges if self._edge_has_bias(edge)]

    @staticmethod
    def _coerce_edge(edge: GraphInput | dict) -> GraphInput:
        if isinstance(edge, GraphInput):
            return edge
        if isinstance(edge, dict):
            return GraphInput(**edge)
        raise TypeError(f"Unsupported local window edge payload type: {type(edge)!r}")

    def _initial_vector_states(self, frame_poses: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        n = frame_poses.shape[0]
        velocity = frame_poses.new_zeros((n, 3))
        acc_bias = frame_poses.new_zeros((n, 3))
        gyro_bias = frame_poses.new_zeros((n, 3))
        index_to_local = {int(idx.item()): i for i, idx in enumerate(self.frame_indices)}

        for edge in self.edges:
            from_local = index_to_local[int(edge.from_idx.reshape(-1)[0].item())]
            to_local = index_to_local[int(edge.frame_idx.reshape(-1)[0].item())]
            if edge.imu_vio_prev_velocity_world is not None:
                velocity[from_local] = edge.imu_vio_prev_velocity_world.reshape(3).float()
            if edge.imu_vio_curr_velocity_init_world is not None:
                velocity[to_local] = edge.imu_vio_curr_velocity_init_world.reshape(3).float()
            if edge.imu_vio_prev_acc_bias is not None:
                acc_bias[from_local] = edge.imu_vio_prev_acc_bias.reshape(3).float()
            if edge.imu_vio_prev_gyro_bias is not None:
                gyro_bias[from_local] = edge.imu_vio_prev_gyro_bias.reshape(3).float()
            if edge.imu_vio_curr_acc_bias_init is not None:
                acc_bias[to_local] = edge.imu_vio_curr_acc_bias_init.reshape(3).float()
            if edge.imu_vio_curr_gyro_bias_init is not None:
                gyro_bias[to_local] = edge.imu_vio_curr_gyro_bias_init.reshape(3).float()
        return velocity, acc_bias, gyro_bias

    def _all_poses(self) -> pp.LieTensor:
        if self.fixed_first_frame:
            return pp.SE3(torch.cat([self.fixed_pose0.to(self.pose_window.tensor()), self.pose_window.tensor()], dim=0))
        return pp.SE3(self.pose_window.tensor())

    def _all_velocity(self) -> torch.Tensor:
        if self.fixed_first_frame:
            return torch.cat([self.fixed_velocity0.to(self.velocity_window), self.velocity_window], dim=0)
        return self.velocity_window

    def _all_acc_bias(self) -> torch.Tensor:
        if self.fixed_first_frame:
            return torch.cat([self.fixed_acc_bias0.to(self.acc_bias_window), self.acc_bias_window], dim=0)
        return self.acc_bias_window

    def _all_gyro_bias(self) -> torch.Tensor:
        if self.fixed_first_frame:
            return torch.cat([self.fixed_gyro_bias0.to(self.gyro_bias_window), self.gyro_bias_window], dim=0)
        return self.gyro_bias_window

    def _edge_local_indices(self, edge: GraphInput) -> tuple[int, int]:
        from_idx = int(edge.from_idx.reshape(-1)[0].item())
        to_idx = int(edge.frame_idx.reshape(-1)[0].item())
        matches = {int(idx.item()): i for i, idx in enumerate(self.frame_indices)}
        return matches[from_idx], matches[to_idx]

    @staticmethod
    def _edge_has_vio(edge: GraphInput) -> bool:
        return bool(
            edge.imu_vio_factor_enable
            and edge.imu_vio_delta_rotvec is not None
            and edge.imu_vio_delta_v is not None
            and edge.imu_vio_delta_p is not None
            and edge.imu_vio_cov is not None
            and edge.imu_vio_dt is not None
        )

    @staticmethod
    def _edge_has_bias(edge: GraphInput) -> bool:
        return bool(
            edge.imu_vio_prev_acc_bias is not None
            and edge.imu_vio_prev_gyro_bias is not None
            and edge.imu_vio_curr_acc_bias_init is not None
            and edge.imu_vio_curr_gyro_bias_init is not None
            and edge.imu_vio_linearized_acc_bias is not None
            and edge.imu_vio_linearized_gyro_bias is not None
            and edge.imu_vio_bias_jacobian is not None
            and edge.imu_vio_bias_rw_cov is not None
        )

    @staticmethod
    def _edge_imu_cov(edge: GraphInput) -> torch.Tensor:
        cov = edge.imu_vio_cov if edge.imu_vio_cov is not None else torch.eye(9, dtype=torch.float32) * 1e6
        return vio_preintegrated_covariance_matrix(cov.reshape(9, 9).float())

    @staticmethod
    def _edge_bias_cov(edge: GraphInput) -> torch.Tensor:
        cov = edge.imu_vio_bias_rw_cov if edge.imu_vio_bias_rw_cov is not None else torch.eye(6, dtype=torch.float32) * 1e6
        return cov.reshape(6, 6).float()

    def _build_visual_covariances(self) -> torch.Tensor:
        covs: list[torch.Tensor] = []
        for edge in self.edges:
            obs = edge.observations
            cov_kp2 = obs.data["pixel2_uv_cov"]
            n = cov_kp2.size(0)
            cov = torch.zeros((n, 3, 3), dtype=torch.float32)
            if cov_kp2.shape[-1] == 3:
                cov[:, 0, 0] = cov_kp2[:, 0].float().clamp(min=1e-12)
                cov[:, 1, 1] = cov_kp2[:, 1].float().clamp(min=1e-12)
                cov[:, 0, 1] = cov_kp2[:, 2].float()
                cov[:, 1, 0] = cov_kp2[:, 2].float()
            else:
                cov[:, :2, :2] = cov_kp2.float()
            cov[:, 2, 2] = obs.data["pixel2_disp_cov"].reshape(-1).float().clamp(min=1e-12)
            covs.append(cov)
        if not covs:
            return torch.zeros((0, 3, 3), dtype=torch.float32)
        return torch.cat(covs, dim=0)

    def _visual_edge_residual(self, edge: GraphInput, pose_i: pp.LieTensor, pose_j: pp.LieTensor) -> torch.Tensor:
        obs = edge.observations
        dtype = self.pose_window.tensor().dtype
        device = self.pose_window.tensor().device
        K = edge.images_intrinsic.to(device=device, dtype=dtype)
        baseline = edge.baseline.reshape(-1)[0].to(device=device, dtype=dtype)
        point_w = edge.points.data["pos_Tw"].to(device=device, dtype=dtype)
        point_j = pose_j.Inv() * point_w
        reproj_err = point2pixel_NED(point_j, K) - obs.data["pixel2_uv"].to(device=device, dtype=dtype)
        disp_pred = point_j[:, 0:1].reciprocal() * (K[0, 0] * baseline)
        disp_err = disp_pred - obs.data["pixel2_disp"].to(device=device, dtype=dtype)
        return torch.cat([reproj_err, disp_err], dim=-1)

    def _imu_edge_residual(
        self,
        edge: GraphInput,
        pose_i: pp.LieTensor,
        pose_j: pp.LieTensor,
        local_i: int,
        local_j: int,
    ) -> torch.Tensor:
        velocity = self._all_velocity()
        acc_bias = self._all_acc_bias()
        gyro_bias = self._all_gyro_bias()
        residual = vio_preintegrated_imu_residual(
            from_pose=pose_i,
            to_pose=pose_j,
            prev_velocity_world=velocity[local_i],
            curr_velocity_world=velocity[local_j],
            delta_R=edge.imu_vio_delta_rotvec if edge.imu_vio_delta_rotvec is not None else torch.zeros(3),
            delta_v=edge.imu_vio_delta_v if edge.imu_vio_delta_v is not None else torch.zeros(3),
            delta_p=edge.imu_vio_delta_p if edge.imu_vio_delta_p is not None else torch.zeros(3),
            dt_total=float(edge.imu_vio_dt.reshape(-1)[0].detach().cpu().item()) if edge.imu_vio_dt is not None else 0.0,
            prev_acc_bias=acc_bias[local_i] if self._edge_has_bias(edge) else None,
            prev_gyro_bias=gyro_bias[local_i] if self._edge_has_bias(edge) else None,
            curr_acc_bias=acc_bias[local_j] if self._edge_has_bias(edge) else None,
            curr_gyro_bias=gyro_bias[local_j] if self._edge_has_bias(edge) else None,
            linearized_acc_bias=edge.imu_vio_linearized_acc_bias if self._edge_has_bias(edge) else None,
            linearized_gyro_bias=edge.imu_vio_linearized_gyro_bias if self._edge_has_bias(edge) else None,
            bias_jacobian=edge.imu_vio_bias_jacobian if self._edge_has_bias(edge) else None,
            sensor_T_imu=pp.SE3(edge.imu_vio_sensor_T_imu.reshape(1, 7)) if edge.imu_vio_sensor_T_imu is not None else None,
            gravity_world=edge.imu_vio_gravity_world,
            gravity_handling="residual" if edge.imu_vio_gravity_in_residual else "preintegration",
        )
        row_scale = torch.tensor(
            [
                [max(float(edge.imu_vio_alpha_p), 0.0) ** 0.5],
                [max(float(edge.imu_vio_alpha_v), 0.0) ** 0.5],
                [max(float(edge.imu_vio_alpha_R), 0.0) ** 0.5],
            ],
            dtype=residual.dtype,
            device=residual.device,
        )
        return residual * row_scale

    def _bias_edge_residual(self, local_i: int, local_j: int) -> torch.Tensor:
        acc_bias = self._all_acc_bias()
        gyro_bias = self._all_gyro_bias()
        return vio_bias_random_walk_residual(
            prev_acc_bias=acc_bias[local_i],
            prev_gyro_bias=gyro_bias[local_i],
            curr_acc_bias=acc_bias[local_j],
            curr_gyro_bias=gyro_bias[local_j],
        )

    def forward(self) -> torch.Tensor:
        poses = self._all_poses()
        visual_rows: list[torch.Tensor] = []
        imu_rows: list[torch.Tensor] = []
        bias_rows: list[torch.Tensor] = []
        for edge in self.edges:
            local_i, local_j = self._edge_local_indices(edge)
            pose_i = poses[local_i]
            pose_j = poses[local_j]
            visual_rows.append(self._visual_edge_residual(edge, pose_i, pose_j))
            if self._edge_has_vio(edge):
                imu_rows.append(self._imu_edge_residual(edge, pose_i, pose_j, local_i, local_j))
                if self._edge_has_bias(edge):
                    bias_rows.append(self._bias_edge_residual(local_i, local_j))

        rows = visual_rows + imu_rows + bias_rows
        if not rows:
            return self.pose_window.tensor().new_zeros((0, 3))
        return torch.cat([row.reshape(-1, 3) for row in rows], dim=0)

    @torch.no_grad()
    @torch.inference_mode()
    def covariance_array(self) -> torch.Tensor:
        blocks = [self._visual_covariances]
        for cov in self._imu_covariances:
            blocks.append(vio_preintegrated_covariance_blocks(cov.reshape(9, 9)))
        for cov in self._bias_covariances:
            eye = torch.eye(3, dtype=cov.dtype, device=cov.device) * 1e-12
            blocks.append(torch.stack([cov[0:3, 0:3] + eye, cov[3:6, 3:6] + eye], dim=0))
        return torch.cat(blocks, dim=0) if blocks else torch.zeros((0, 3, 3), dtype=torch.float32)

    @torch.no_grad()
    @torch.inference_mode()
    def weight_matrix(self) -> torch.Tensor:
        return build_weight_matrix_from_covariances(
            self._visual_covariances.to(self.pose_window.tensor()),
            full_covariances=[cov.to(self.pose_window.tensor()) for cov in (self._imu_covariances + self._bias_covariances)],
        )

    @torch.no_grad()
    @torch.inference_mode()
    def write_back(self) -> GraphOutput:
        poses = self._all_poses().tensor().detach().cpu().float()
        velocity = self._all_velocity().detach().cpu().float()
        acc_bias = self._all_acc_bias().detach().cpu().float()
        gyro_bias = self._all_gyro_bias().detach().cpu().float()
        return GraphOutput(
            motion=poses[-1:].clone(),
            from_idx=self.frame_indices[-2:-1].clone(),
            frame_idx=self.frame_indices[-1:].clone(),
            velocity_world=velocity[-1].clone(),
            acc_bias=acc_bias[-1].clone(),
            gyro_bias=gyro_bias[-1].clone(),
            imu_factor_mode="local_inertial_ba",
            vio_factor_active=True,
            vio_bias_state_active=True,
            imu_residual_rows=5 * len(self.edges),
            use_imu_rotation=True,
            use_imu_translation=True,
            window_frame_indices=self.frame_indices.detach().cpu().clone(),
            window_motions=poses,
            window_velocity_world=velocity,
            window_acc_bias=acc_bias,
            window_gyro_bias=gyro_bias,
            local_ba_window_size=int(self.frame_indices.numel()),
            local_ba_writeback=self.local_ba_writeback,
            local_ba_num_frames=int(self.frame_indices.numel()),
            local_ba_num_edges=len(self.edges),
            local_ba_num_visual_residual_blocks=sum(
                int(edge.observations.data["pixel2_uv"].shape[0]) for edge in self.edges
            ),
        )


class Analytic_ICP_TwoframePGO(ICP_TwoframePGO, AnalyticModule):
    def __init__(self, graph_data: GraphInput) -> None:
        super().__init__(graph_data)

    @torch.no_grad()
    def build_jacobian(self) -> torch.Tensor:
        frame_pose = T.cast(pp.LieTensor, self.pose2opt[self.edges_index])
        R = frame_pose.rotation().matrix()
        p = self.points_Tc
        E = p.shape[0]

        J = torch.zeros((E, 3, 7), device=p.device, dtype=p.dtype)

        I3 = torch.eye(3, device=p.device, dtype=p.dtype).unsqueeze(0)
        J[..., 0:3] = I3
        J[..., 3:6] = -pp.vec2skew(frame_pose.Act(p))

        J_flat = J.view(-1, 7)
        if self.use_imu_rot_prior:
            J_imu = torch.zeros(3, 7, device=J.device, dtype=J.dtype)
            J_imu[:, 3:6] = _so3_right_jacobian_inverse(self._imu_rot_residual().reshape(3)).to(J)
            J_flat = torch.cat([J_flat, J_imu], dim=0)
        if self.use_imu_trans_prior:
            J_imu_t = torch.zeros(3, 7, device=J.device, dtype=J.dtype)
            J_imu_t[:, 0:3] = torch.eye(3, device=J.device, dtype=J.dtype)
            J_flat = torch.cat([J_flat, J_imu_t], dim=0)
        return J_flat


class Analytic_Reproj_TwoFramePGO(Reproj_TwoFramePGO, AnalyticModule):
    def __init__(self, graph_data: GraphInput) -> None:
        super().__init__(graph_data)

    @torch.no_grad()
    def build_jacobian(self) -> torch.Tensor:
        assert self.pos_Tc is not None, "pos_Tc not found, need to call forward() before building jacobian."
        fx = self.K[0, 0]
        fy = self.K[1, 1]
        assert self.K[0, 1] == 0, "K[0, 1] non-zero is currently not supported"
        # s = self.K[0, 1] # TODO: add this feature later!

        x, y, z = self.pos_Tc[:, 0], self.pos_Tc[:, 1], self.pos_Tc[:, 2]
        x_square = x ** 2
        J_homoKS = torch.zeros(self.pos_Tc.shape[0], 2, 3, device=self.pos_Tc.device, dtype=self.pos_Tc.dtype)
        J_homoKS[:, 0, 0] = -fx * y / x_square
        J_homoKS[:, 0, 1] = fx / x
        J_homoKS[:, 1, 0] = -fy * z / x_square
        J_homoKS[:, 1, 2] = fy / x

        R = self.pose2opt.rotation().matrix()
        R_T = R.transpose(-2, -1)
        J_Tinv_p = torch.zeros(self.pos_Tc.shape[0], 3, 7, device=self.pos_Tc.device,
                               dtype=self.pos_Tc.dtype)  # 7 width because of pypose implementation, last column is useless
        J_Tinv_p[..., :3] = -R_T
        J_Tinv_p[..., 3:6] = R_T @ pp.vec2skew(self.pos_Tw)
        J = (J_homoKS @ J_Tinv_p).view(-1, 7)
        if self.use_imu_rot_prior:
            J_imu = torch.zeros(3, 7, device=J.device, dtype=J.dtype)
            J_imu[:, 3:6] = _so3_right_jacobian_inverse(self._imu_rot_residual().reshape(3)).to(J)
            J = torch.cat([J, J_imu], dim=0)
        if self.use_imu_trans_prior:
            J_imu_t = torch.zeros(3, 7, device=J.device, dtype=J.dtype)
            J_imu_t[:, 0:3] = torch.eye(3, device=J.device, dtype=J.dtype)
            J = torch.cat([J, J_imu_t], dim=0)
        return J


class Analytic_ReprojDisp_TwoFramePGO(ReprojDisp_TwoFramePGO, AnalyticModule):
    def __init__(self, graph_data: GraphInput) -> None:
        super().__init__(graph_data)

    @torch.no_grad()
    def build_jacobian(self) -> torch.Tensor:
        assert self.pos_Tc is not None, "pos_Tc not found, need to call forward() before building jacobian."
        fx = self.K[0, 0]
        fy = self.K[1, 1]
        cx = self.K[0, 2]
        cy = self.K[1, 2]
        assert self.K[0, 1] == 0, "K[0, 1] non-zero is currently not supported"
        # s = self.K[0, 1] # TODO: add this feature later!

        x, y, z = self.pos_Tc[:, 0], self.pos_Tc[:, 1], self.pos_Tc[:, 2]
        x_square = x ** 2
        J_homoKS = torch.zeros(self.pos_Tc.shape[0], 2, 3, device=self.pos_Tc.device, dtype=self.pos_Tc.dtype)
        J_homoKS[:, 0, 0] = -fx * y / x_square
        J_homoKS[:, 0, 1] = fx / x
        J_homoKS[:, 1, 0] = -fy * z / x_square
        J_homoKS[:, 1, 2] = fy / x
        R = self.pose2opt.rotation().matrix()
        R_T = R.transpose(-2, -1)
        J_Tinv_p = torch.zeros(self.pos_Tc.shape[0], 3, 7, device=self.pos_Tc.device,
                               dtype=self.pos_Tc.dtype)  # 7 width because of pypose implementation, last column is useless
        J_Tinv_p[..., :3] = -R_T
        J_Tinv_p[..., 3:6] = R_T @ pp.vec2skew(self.pos_Tw)
        J_reproj = (J_homoKS @ J_Tinv_p)
        J_disp = (-(self.baseline * fx) / x_square).view(-1, 1, 1) * J_Tinv_p[:, 0:1, :]
        J = torch.cat((J_reproj, J_disp), dim=1).view(-1, 7)
        if self.use_imu_rot_prior:
            J_imu = torch.zeros(3, 7, device=J.device, dtype=J.dtype)
            J_imu[:, 3:6] = _so3_right_jacobian_inverse(self._imu_rot_residual().reshape(3)).to(J)
            J = torch.cat([J, J_imu], dim=0)
        if self.use_imu_trans_prior:
            J_imu_t = torch.zeros(3, 7, device=J.device, dtype=J.dtype)
            J_imu_t[:, 0:3] = torch.eye(3, device=J.device, dtype=J.dtype)
            J = torch.cat([J, J_imu_t], dim=0)
        return J
