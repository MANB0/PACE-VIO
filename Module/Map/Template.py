import typing as T
from .Graph import TensorBundle, AutoScalingBundle

# Define storage of interest
FrameFeature = T.Literal[
    "K",            # Nx3x3 , dtype=float32
    "baseline",     # Nx1   , dtype=float32
    "pose",         # Nx7   , dtype=float32, pose of sensor under world frame.
    "T_BS",         # Nx7   , dtype=float32, body-to-sensor SE3 transformation.
    "need_interp",  # Nx1   , dtype=bool
    "time_ns",      # Nx1   , dtype=long
    "imu_rotvec_prior",   # Nx3   , dtype=float32, relative rotation prior from IMU gyro integration.
    "imu_rot_prior_std",  # Nx1   , dtype=float32, std (rad) for IMU rotation soft constraint.
    "imu_trans_prior",    # Nx3   , dtype=float32, relative translation prior from IMU preintegration.
    "imu_trans_cov",      # Nx3x3 , dtype=float32, covariance of IMU translation prior.
    "imu_vio_prev_velocity_world",      # Nx3   , dtype=float32, previous world-frame velocity for VIO factor.
    "imu_vio_curr_velocity_init_world", # Nx3   , dtype=float32, current velocity initialization for VIO factor.
    "imu_vio_velocity_world",           # Nx3   , dtype=float32, optimized current world-frame velocity.
    "imu_vio_prev_acc_bias",            # Nx3   , dtype=float32, previous accelerometer bias state.
    "imu_vio_prev_gyro_bias",           # Nx3   , dtype=float32, previous gyroscope bias state.
    "imu_vio_acc_bias",                 # Nx3   , dtype=float32, optimized/current accelerometer bias state.
    "imu_vio_gyro_bias",                # Nx3   , dtype=float32, optimized/current gyroscope bias state.
    "imu_vio_linearized_acc_bias",      # Nx3   , dtype=float32, accelerometer bias used to build this edge preintegration.
    "imu_vio_linearized_gyro_bias",     # Nx3   , dtype=float32, gyroscope bias used to build this edge preintegration.
    "imu_vio_bias_jacobian",            # Nx9x6 , dtype=float32, preintegration bias Jacobian.
    "imu_vio_bias_rw_cov",              # Nx6x6 , dtype=float32, bias random-walk covariance.
    "imu_vio_delta_rotvec",             # Nx3   , dtype=float32, preintegrated rotation vector in body_i.
    "imu_vio_delta_v",                  # Nx3   , dtype=float32, preintegrated velocity delta in body_i.
    "imu_vio_delta_p",                  # Nx3   , dtype=float32, preintegrated position delta in body_i.
    "imu_vio_cov",                      # Nx9x9 , dtype=float32, covariance of [delta_p, delta_v, delta_phi].
    "imu_vio_sa_v2_unique_cov",         # Nx9x9 , dtype=float32, covariance excluding endpoint-shared raw samples.
    "imu_vio_sa_v2_incoming_raw_time_ns", # Nx2, dtype=long, padded raw timestamps supporting t_i.
    "imu_vio_sa_v2_outgoing_raw_time_ns", # Nx2, dtype=long, padded raw timestamps supporting t_j.
    "imu_vio_sa_v2_incoming_count",     # Nx1, dtype=long, number of valid incoming raw timestamps.
    "imu_vio_sa_v2_outgoing_count",     # Nx1, dtype=long, number of valid outgoing raw timestamps.
    "imu_vio_sa_v2_incoming_sensitivity", # Nx9x12, dtype=float32, standardized endpoint noise sensitivity.
    "imu_vio_sa_v2_outgoing_sensitivity", # Nx9x12, dtype=float32, standardized endpoint noise sensitivity.
    "imu_vio_dt",                       # Nx1   , dtype=float32, preintegration interval in seconds.
    "imu_vio_sensor_T_imu",             # Nx7   , dtype=float32, optimized sensor/camera to IMU SE3 for VIO residuals.
    "imu_vio_gravity_world",             # Nx3   , dtype=float32, signed world-frame gravity vector.
    "imu_vio_gravity_in_residual",       # Nx1   , dtype=bool, true for standard VIO residual gravity handling.
    "visual_relative_pose_CiCj",          # Nx7   , dtype=float32, cached T_CiCj visual pose measurement.
    "visual_relative_pose_cov",           # Nx6x6 , dtype=float32, covariance of the visual pose measurement.
    "visual_relative_pose_num_points",    # Nx1   , dtype=long, point count used by the pose sidecar.
    "visual_relative_pose_num_inliers",   # Nx1   , dtype=long, robust inlier count used by the sidecar.
    "visual_relative_pose_mean_mahalanobis_sq", # Nx1, dtype=float32, mean point residual Mahalanobis squared.
    "visual_compressed_uvd_reference_CjCi", # Nx7, dtype=float64, Cj<-Ci linearization pose.
    "visual_compressed_uvd_hessian",        # Nx6x6, dtype=float64, right-tangent [t,r] Hessian.
    "visual_compressed_uvd_gradient",       # Nx6, dtype=float64, right-tangent [t,r] gradient.
    "visual_compressed_uvd_robust_cost",    # Nx1, dtype=float64, robust visual cost at the reference.
    "visual_compressed_uvd_num_points",     # Nx1, dtype=long, point count used by the cached factor.
    "visual_compressed_uvd_num_inliers",    # Nx1, dtype=long, robust inlier count at the reference.
    "visual_compressed_uvd_mean_mahalanobis_sq", # Nx1, dtype=float64, mean raw UVD Mahalanobis squared.
    "visual_compressed_uvd_huber_delta",    # Nx1, dtype=float64, Huber threshold used before compression.
    "fusion_visual_quality",     # Nx1, dtype=float32, online visual quality score in [0, 1].
    "fusion_degrade_score",      # Nx1, dtype=float32, online visual degradation score in [0, 1].
    "fusion_trans_switch",       # Nx1, dtype=float32, robust switch for IMU translation.
    "fusion_rot_switch",         # Nx1, dtype=float32, robust switch for IMU rotation.
    "fusion_xy_weight",          # Nx1, dtype=float32, fused IMU XY contribution ratio.
    "fusion_z_weight",           # Nx1, dtype=float32, fused IMU Z contribution ratio.
    "fusion_rot_weight",         # Nx1, dtype=float32, fused IMU rotation contribution ratio.
    "fusion_gate_flags",         # Nx4, dtype=float32, [active, xy_enabled, z_enabled, rot_enabled].
]
MatchingFeature = T.Literal[
    "pixel1_uv",    # Nx2   , dtype=float32
    "pixel1_d",     # Nx1   , dtype=float32
    "pixel2_uv",    # Nx2   , dtype=float32
    "pixel2_d",     # Nx1   , dtype=float32
    "pixel1_disp",  # Nx1   , dtype=float32
    "pixel2_disp",  # Nx1   , dtype=float32
    "pixel1_uv_cov",# Nx3   , dtype=float32, (\sigma_uu, \sigma_vv, \sigma_uv)
    "pixel2_uv_cov",# Nx3   , dtype=float32, (\sigma_uu, \sigma_vv, \sigma_uv)
    "pixel1_d_cov" ,# Nx1   , dtype=float32
    "pixel2_d_cov" ,# Nx1   , dtype=float32
    "pixel1_disp_cov",    # Nx1   , dtype=float32
    "pixel2_disp_cov",    # Nx1   , dtype=float32
    "obs1_covTc",   # Nx3x3 , dtype=float64
    "obs2_covTc",   # Nx3x3 , dtype=float64
]
PointFeature = T.Literal[
    "pos_Tc",       # Nx3   , dtype=float32, exact source-frame point used to build pos_Tw
    "pos_Tw",       # Nx3   , dtype=float32
    "cov_Tw",       # Nx3x3 , dtype=float64
    "color" ,       # Nx3   , dtype=uint8
]


FrameNode    = TensorBundle[FrameFeature]
FrameStore   = AutoScalingBundle[FrameFeature]

MatchObs     = TensorBundle[MatchingFeature]
MatchStore   = AutoScalingBundle[MatchingFeature]

PointNode    = TensorBundle[PointFeature]
PointStore   = AutoScalingBundle[PointFeature]
