import typing as T
import torch
import numpy as np
from typing_extensions import Self

from Utility.Extensions import AutoScalingTensor
from .Graph import Scaling_DenseEdge_Multi, Scaling_SparseEdge_Multi, Scaling_SingleEdge

# Define storage of interest
from .Template   import (
    FrameStore, MatchStore , PointStore,
    FrameNode , MatchObs, PointNode ,
)

class VisualMap:
    def __init__(self) -> None:
        self.init_size: T.Final[int]  = 1024
        self.max_pt_obs: T.Final[int] = 5
        self.max_frame_range: T.Final[int] = 2
        
        self.frames = FrameStore(
            index=AutoScalingTensor((self.init_size,), grow_on=0, dtype=torch.long),
            data={
                "K"          : AutoScalingTensor((self.init_size, 3, 3), grow_on=0, dtype=torch.float32),
                "baseline"   : AutoScalingTensor((self.init_size,     ), grow_on=0, dtype=torch.float32),
                "pose"       : AutoScalingTensor((self.init_size, 7   ), grow_on=0, dtype=torch.float32),
                "T_BS"       : AutoScalingTensor((self.init_size, 7   ), grow_on=0, dtype=torch.float32),
                "need_interp": AutoScalingTensor((self.init_size,     ), grow_on=0, dtype=torch.bool),
                "time_ns"    : AutoScalingTensor((self.init_size,     ), grow_on=0, dtype=torch.long),
                "imu_rotvec_prior": AutoScalingTensor((self.init_size, 3  ), grow_on=0, dtype=torch.float32),
                "imu_rot_prior_std": AutoScalingTensor((self.init_size,   ), grow_on=0, dtype=torch.float32),
                "imu_trans_prior"  : AutoScalingTensor((self.init_size, 3  ), grow_on=0, dtype=torch.float32),
                "imu_trans_cov"    : AutoScalingTensor((self.init_size, 3, 3), grow_on=0, dtype=torch.float32),
                "imu_vio_prev_velocity_world": AutoScalingTensor((self.init_size, 3), grow_on=0, dtype=torch.float32),
                "imu_vio_curr_velocity_init_world": AutoScalingTensor((self.init_size, 3), grow_on=0, dtype=torch.float32),
                "imu_vio_velocity_world": AutoScalingTensor((self.init_size, 3), grow_on=0, dtype=torch.float32),
                "imu_vio_prev_acc_bias": AutoScalingTensor((self.init_size, 3), grow_on=0, dtype=torch.float32),
                "imu_vio_prev_gyro_bias": AutoScalingTensor((self.init_size, 3), grow_on=0, dtype=torch.float32),
                "imu_vio_acc_bias": AutoScalingTensor((self.init_size, 3), grow_on=0, dtype=torch.float32),
                "imu_vio_gyro_bias": AutoScalingTensor((self.init_size, 3), grow_on=0, dtype=torch.float32),
                "imu_vio_linearized_acc_bias": AutoScalingTensor((self.init_size, 3), grow_on=0, dtype=torch.float32),
                "imu_vio_linearized_gyro_bias": AutoScalingTensor((self.init_size, 3), grow_on=0, dtype=torch.float32),
                "imu_vio_bias_jacobian": AutoScalingTensor((self.init_size, 9, 6), grow_on=0, dtype=torch.float32),
                "imu_vio_bias_rw_cov": AutoScalingTensor((self.init_size, 6, 6), grow_on=0, dtype=torch.float32),
                "imu_vio_delta_rotvec": AutoScalingTensor((self.init_size, 3), grow_on=0, dtype=torch.float32),
                "imu_vio_delta_v": AutoScalingTensor((self.init_size, 3), grow_on=0, dtype=torch.float32),
                "imu_vio_delta_p": AutoScalingTensor((self.init_size, 3), grow_on=0, dtype=torch.float32),
                "imu_vio_cov": AutoScalingTensor((self.init_size, 9, 9), grow_on=0, dtype=torch.float32),
                "imu_vio_sa_v2_unique_cov": AutoScalingTensor((self.init_size, 9, 9), grow_on=0, dtype=torch.float32),
                "imu_vio_sa_v2_incoming_raw_time_ns": AutoScalingTensor(
                    (self.init_size, 2), grow_on=0, dtype=torch.long, init_val=-1
                ),
                "imu_vio_sa_v2_outgoing_raw_time_ns": AutoScalingTensor(
                    (self.init_size, 2), grow_on=0, dtype=torch.long, init_val=-1
                ),
                "imu_vio_sa_v2_incoming_count": AutoScalingTensor(
                    (self.init_size,), grow_on=0, dtype=torch.long, init_val=0
                ),
                "imu_vio_sa_v2_outgoing_count": AutoScalingTensor(
                    (self.init_size,), grow_on=0, dtype=torch.long, init_val=0
                ),
                "imu_vio_sa_v2_incoming_sensitivity": AutoScalingTensor(
                    (self.init_size, 9, 12), grow_on=0, dtype=torch.float32, init_val=0.0
                ),
                "imu_vio_sa_v2_outgoing_sensitivity": AutoScalingTensor(
                    (self.init_size, 9, 12), grow_on=0, dtype=torch.float32, init_val=0.0
                ),
                "imu_vio_dt": AutoScalingTensor((self.init_size,), grow_on=0, dtype=torch.float32),
                "imu_vio_sensor_T_imu": AutoScalingTensor((self.init_size, 7), grow_on=0, dtype=torch.float32),
                "imu_vio_gravity_world": AutoScalingTensor((self.init_size, 3), grow_on=0, dtype=torch.float32),
                "imu_vio_gravity_in_residual": AutoScalingTensor((self.init_size,), grow_on=0, dtype=torch.bool),
                "visual_relative_pose_CiCj": AutoScalingTensor((self.init_size, 7), grow_on=0, dtype=torch.float32),
                "visual_relative_pose_cov": AutoScalingTensor((self.init_size, 6, 6), grow_on=0, dtype=torch.float32),
                "visual_relative_pose_num_points": AutoScalingTensor(
                    (self.init_size,), grow_on=0, dtype=torch.long, init_val=0
                ),
                "visual_relative_pose_num_inliers": AutoScalingTensor(
                    (self.init_size,), grow_on=0, dtype=torch.long, init_val=0
                ),
                "visual_relative_pose_mean_mahalanobis_sq": AutoScalingTensor(
                    (self.init_size,), grow_on=0, dtype=torch.float32, init_val=-1.0
                ),
                "visual_compressed_uvd_reference_CjCi": AutoScalingTensor(
                    (self.init_size, 7), grow_on=0, dtype=torch.float64
                ),
                "visual_compressed_uvd_hessian": AutoScalingTensor(
                    (self.init_size, 6, 6), grow_on=0, dtype=torch.float64
                ),
                "visual_compressed_uvd_gradient": AutoScalingTensor(
                    (self.init_size, 6), grow_on=0, dtype=torch.float64
                ),
                "visual_compressed_uvd_robust_cost": AutoScalingTensor(
                    (self.init_size,), grow_on=0, dtype=torch.float64, init_val=-1.0
                ),
                "visual_compressed_uvd_num_points": AutoScalingTensor(
                    (self.init_size,), grow_on=0, dtype=torch.long, init_val=0
                ),
                "visual_compressed_uvd_num_inliers": AutoScalingTensor(
                    (self.init_size,), grow_on=0, dtype=torch.long, init_val=0
                ),
                "visual_compressed_uvd_mean_mahalanobis_sq": AutoScalingTensor(
                    (self.init_size,), grow_on=0, dtype=torch.float64, init_val=-1.0
                ),
                "visual_compressed_uvd_huber_delta": AutoScalingTensor(
                    (self.init_size,), grow_on=0, dtype=torch.float64, init_val=0.1
                ),
                "fusion_visual_quality": AutoScalingTensor((self.init_size,), grow_on=0, dtype=torch.float32, init_val=-1.0),
                "fusion_degrade_score": AutoScalingTensor((self.init_size,), grow_on=0, dtype=torch.float32, init_val=-1.0),
                "fusion_trans_switch": AutoScalingTensor((self.init_size,), grow_on=0, dtype=torch.float32, init_val=-1.0),
                "fusion_rot_switch": AutoScalingTensor((self.init_size,), grow_on=0, dtype=torch.float32, init_val=-1.0),
                "fusion_xy_weight": AutoScalingTensor((self.init_size,), grow_on=0, dtype=torch.float32, init_val=0.0),
                "fusion_z_weight": AutoScalingTensor((self.init_size,), grow_on=0, dtype=torch.float32, init_val=0.0),
                "fusion_rot_weight": AutoScalingTensor((self.init_size,), grow_on=0, dtype=torch.float32, init_val=0.0),
                "fusion_gate_flags": AutoScalingTensor((self.init_size, 4), grow_on=0, dtype=torch.float32, init_val=0.0),
            }
        )
        
        self.points = PointStore(
            index=AutoScalingTensor((self.init_size,), grow_on=0, dtype=torch.long),
            data={
                "pos_Tc" : AutoScalingTensor((self.init_size, 3   ), grow_on=0, dtype=torch.float32),
                "pos_Tw" : AutoScalingTensor((self.init_size, 3   ), grow_on=0, dtype=torch.float32),
                "cov_Tw" : AutoScalingTensor((self.init_size, 3, 3), grow_on=0, dtype=torch.float64),
                "color"  : AutoScalingTensor((self.init_size, 3   ), grow_on=0, dtype=torch.uint8)
            }
        )
        
        self.map_points = PointStore(
            index=AutoScalingTensor((self.init_size,), grow_on=0, dtype=torch.long),
            data={
                "pos_Tc" : AutoScalingTensor((self.init_size, 3   ), grow_on=0, dtype=torch.float32),
                "pos_Tw" : AutoScalingTensor((self.init_size, 3   ), grow_on=0, dtype=torch.float32),
                "cov_Tw" : AutoScalingTensor((self.init_size, 3, 3), grow_on=0, dtype=torch.float64),
                "color"  : AutoScalingTensor((self.init_size, 3   ), grow_on=0, dtype=torch.uint8)
            }
        )

        self.match = MatchStore(
            index=AutoScalingTensor((self.init_size,), grow_on=0, dtype=torch.long),
            data={
                "pixel1_uv"      : AutoScalingTensor((self.init_size, 2   ), grow_on=0, dtype=torch.float32),
                "pixel2_uv"      : AutoScalingTensor((self.init_size, 2   ), grow_on=0, dtype=torch.float32),
                "pixel1_d"       : AutoScalingTensor((self.init_size, 1   ), grow_on=0, dtype=torch.float32),
                "pixel2_d"       : AutoScalingTensor((self.init_size, 1   ), grow_on=0, dtype=torch.float32),
                "pixel1_disp"    : AutoScalingTensor((self.init_size, 1   ), grow_on=0, dtype=torch.float32),
                "pixel2_disp"    : AutoScalingTensor((self.init_size, 1   ), grow_on=0, dtype=torch.float32),
                "pixel1_disp_cov": AutoScalingTensor((self.init_size, 1   ), grow_on=0, dtype=torch.float32),
                "pixel2_disp_cov": AutoScalingTensor((self.init_size, 1   ), grow_on=0, dtype=torch.float32),
                "obs1_covTc"     : AutoScalingTensor((self.init_size, 3, 3), grow_on=0, dtype=torch.float64),
                "obs2_covTc"     : AutoScalingTensor((self.init_size, 3, 3), grow_on=0, dtype=torch.float64),
                "pixel1_uv_cov"  : AutoScalingTensor((self.init_size, 3   ), grow_on=0, dtype=torch.float32),
                "pixel2_uv_cov"  : AutoScalingTensor((self.init_size, 3   ), grow_on=0, dtype=torch.float32),
                "pixel1_d_cov"   : AutoScalingTensor((self.init_size, 1   ), grow_on=0, dtype=torch.float32),
                "pixel2_d_cov"   : AutoScalingTensor((self.init_size, 1   ), grow_on=0, dtype=torch.float32)
            }
        )

        self.frame2match  = Scaling_DenseEdge_Multi(self.init_size, self.max_frame_range)
        self.frame2map    = Scaling_DenseEdge_Multi(self.init_size, self.max_frame_range)
        self.match2frame1 = Scaling_SingleEdge(self.init_size)
        self.match2frame2 = Scaling_SingleEdge(self.init_size)
        self.match2point  = Scaling_SingleEdge(self.init_size)
        self.point2match  = Scaling_SparseEdge_Multi(self.init_size, self.max_pt_obs)
        
        self.frames.register_edge(self.frame2map)
        self.frames.register_edge(self.frame2match)
        self.points.register_edge(self.point2match)
        self.match.register_edge(self.match2point)
        self.match.register_edge(self.match2frame1)
        self.match.register_edge(self.match2frame2)
        

    def get_frame2match(self, frame: FrameNode) -> MatchObs:
        return self.match[self.frame2match.project(frame.index)]

    def get_match2point(self, match: MatchObs) -> PointNode:
        return self.points[self.match2point.project(match.index)]
    
    def get_point2match(self, point: PointNode) -> MatchObs:
        return self.match[self.point2match.project(point.index)]
    
    def get_match2frame1(self, match: MatchObs) -> FrameNode:
        return self.frames[self.match2frame1.project(match.index)]
    
    def get_match2frame2(self, match: MatchObs) -> FrameNode:
        return self.frames[self.match2frame2.project(match.index)]
    
    def get_frame2map(self, frame: FrameNode) -> PointNode:
        return self.map_points[self.frame2map.project(frame.index)]

    def serialize(self) -> dict[str, np.ndarray]:
        return (
            self.frames.serialize("frames/")
          | self.points.serialize("points/")
          | self.match.serialize("match/")
          | self.frame2match.serialize("edge/frame2match")
          | self.point2match.serialize("edge/point2match")
          | self.match2point.serialize("edge/match2point")
          | self.match2frame1.serialize("edge/match2frame1")
          | self.match2frame2.serialize("edge/match2frame2")
          | self.frame2map.serialize("edge/frame2map")
        )
    
    @classmethod
    def deserialize(cls, value: dict[str, np.ndarray]) -> Self:
        map = cls()
        map.frames = map.frames.deserialize("frames/", value)
        map.match  = map.match.deserialize("match/", value)
        map.points = map.points.deserialize("points/", value)
        
        map.frame2match  = map.frame2match.deserialize("edge/frame2match", value)
        map.point2match  = map.point2match.deserialize("edge/point2match", value)
        map.match2point  = map.match2point .deserialize("edge/match2point", value)
        map.match2frame1 = map.match2frame1.deserialize("edge/match2frame1", value)
        map.match2frame2 = map.match2frame2.deserialize("edge/match2frame2", value)
        map.frame2map    = map.frame2map.deserialize("edge/frame2map", value)
        return map

    def __repr__(self) -> str:
        return f"VisualMap(#frame={len(self.frames)}, #point={len(self.points)}, #map={len(self.map_points)})"
