import torch
import pypose as pp
import pypose.module as pm

from abc import ABC, abstractmethod
from pathlib import Path

from typing import Generic, cast
from types import SimpleNamespace
from Utility.Extensions import ConfigTestableSubclass,TensorQueue
from Utility.PrettyPrint import Logger
from DataLoader import StereoFrame, StereoInertialFrame, T_Data
from Utility.Timer import Timer


def _camera_delta_from_raw_imu_delta(frame: StereoInertialFrame, delta_I: pp.LieTensor) -> pp.LieTensor:
    """Convert an IMU-frame relative motion to the MACVO camera frame."""
    extrinsic = getattr(frame, "imu_vio_sensor_T_imu", None)
    if extrinsic is None:
        return delta_I
    extrinsic_CI = pp.SE3(extrinsic).to(device=delta_I.device, dtype=delta_I.dtype)
    return extrinsic_CI @ delta_I @ extrinsic_CI.Inv()


class IMotionModel(ABC, Generic[T_Data], ConfigTestableSubclass):
    """
    A motion model class receives informations (e.g. frames, estimated flow and depth) and produce an
    initial guess to the pose of incoming frame **under global coordinate**.
    """
    def __init__(self, config: SimpleNamespace):
        self.config : SimpleNamespace = config

    @abstractmethod
    def predict(self, frame: T_Data, flow: torch.Tensor | None, depth: torch.Tensor | None) -> pp.LieTensor:
        """
        Estimate the pose of next frame given current frame, estimated depth and flow.

        NOTE: returned pose should be under global coordinate!

        Returns
        *   pose  - 7, shaped pypose.LieTensor (SE3 ltype) under world coordinate
                  predicted pose of next frame.
        """
        ...

    @abstractmethod
    def update(self, pose: pp.LieTensor) -> None:
        """
        Receive a feedback (optimized pose) and may (or may not) use this method to refine next prediction.
        """
        ...


class GTMotionwithNoise(IMotionModel[StereoFrame]):
    """
    Apply GT motion with noise (can be disabled by setting `noise_std` to 0.0 in config) on previous optimized pose to predict next pose.
    """
    def __init__(self, config: SimpleNamespace):
        super().__init__(config)
        self.prev_pose: pp.LieTensor | None = None
        self.prev_gt_pose: pp.LieTensor | None = None

    def init_context(self) -> None: return None

    def _stableNoiseModel(self) -> pp.LieTensor:
        if self.config.noise_std == 0.0:
            return pp.identity_SE3()
        noise: pp.LieTensor = pp.randn_SE3(sigma=self.noise_std)    #type:ignore
        return noise

    def predict(self, frame: StereoFrame, flow: torch.Tensor | None, depth: torch.Tensor | None) -> pp.LieTensor:
        assert frame.gt_pose is not None
        frame_gtpose = cast(pp.LieTensor, frame.gt_pose.squeeze(0))

        if self.prev_pose is None or self.prev_gt_pose is None:
            self.prev_pose    = pp.identity_SE3()
            self.prev_gt_pose = frame_gtpose
            return pp.identity_SE3()

        gtMotion = self.prev_gt_pose.Inv() @ frame_gtpose
        gtMotion_w_noise = gtMotion @ self._stableNoiseModel()
        predict = self.prev_pose @ gtMotion_w_noise

        self.prev_pose = predict
        self.prev_gt_pose = frame_gtpose

        return predict

    def update(self, pose: pp.LieTensor) -> None:
        self.prev_pose = pose

    @classmethod
    def is_valid_config(cls, config: SimpleNamespace | None) -> None:
        cls._enforce_config_spec(config, {
            "noise_std": lambda noise: isinstance(noise, (int, float)) and noise >= 0.0
        })


class TartanMotionNet(IMotionModel[StereoFrame]):
    """
    Apply motion estimated by MotionNet adapted from TartanVO on previously optimized pose to predict next pose.
    """
    def __init__(self, config: SimpleNamespace):
        from .Network.TartanVOStereo import TartanStereoVOMotion

        super().__init__(config)
        self.model = TartanStereoVOMotion(self.config.weight, True, self.config.device)
        self.prev_pose = None

    @Timer.cpu_timeit("MotionModel")
    @Timer.gpu_timeit("MotionModel")
    @torch.inference_mode()
    def predict(self, frame: StereoFrame, flow: torch.Tensor | None, depth: torch.Tensor | None) -> pp.LieTensor:
        if self.prev_pose is None:
            self.prev_pose = pp.identity_SE3(device=self.config.device)
            return pp.identity_SE3(device=self.config.device)

        assert flow is not None and depth is not None, "Motion model requires flow and depth to predict motion"
        motion_se3: torch.Tensor = self.model.inference(frame, flow, depth)
        new_pose = self.prev_pose @ pp.se3(motion_se3).Exp()
        self.prev_pose = new_pose
        return new_pose

    def update(self, pose: pp.LieTensor) -> None:
        self.prev_pose = pose.to(self.prev_pose.device)    #type: ignore

    @classmethod
    def is_valid_config(cls, config: SimpleNamespace | None) -> None:
        cls._enforce_config_spec(config, {
            "weight": lambda f: isinstance(f, str),
            "device": lambda dev: isinstance(dev, str) and (("cuda" in dev) or (dev == "cpu"))
        })


class StaticMotionModel(IMotionModel[StereoFrame]):
    """
    Assumes the camera is static and simply record and returns the pose of previous frame.
    """
    def __init__(self, config: SimpleNamespace):
        super().__init__(config)
        self.prev_pose: pp.LieTensor | None = None

    def predict(self, frame: StereoFrame, flow: torch.Tensor | None, depth: torch.Tensor | None) -> pp.LieTensor:
        if self.prev_pose is None:
            self.prev_pose = pp.identity_SE3()
            return pp.identity_SE3()
        return self.prev_pose

    def update(self, pose: pp.LieTensor) -> None:
        assert self.prev_pose is not None
        self.prev_pose = pose.to(self.prev_pose.device) # type: ignore

    @classmethod
    def is_valid_config(cls, config: SimpleNamespace | None) -> None: return


class IMUGyroMotionModel(IMotionModel[StereoInertialFrame]):
    """
    Use IMU gyroscope integration to predict relative rotation between consecutive frames.

    This model is intentionally lightweight: it estimates only rotational motion from IMU,
    while keeping translation unchanged from previous optimized pose. It is useful in
    underwater low-texture / low-light segments where visual rotation can drift.
    """

    def __init__(self, config: SimpleNamespace):
        super().__init__(config)
        self.device = getattr(config, "device", "cpu")
        self.prev_pose: pp.LieTensor | None = None

        gyro_bias = getattr(config, "gyro_bias", [0.0, 0.0, 0.0])
        self.gyro_bias = torch.tensor(gyro_bias, dtype=torch.float32, device=self.device)

        self.max_delta_angle_rad = float(getattr(config, "max_delta_angle_rad", 0.8))
        self.fallback_dt_s = float(getattr(config, "fallback_dt_s", 0.01))

    @Timer.cpu_timeit("MotionModel")
    @Timer.gpu_timeit("MotionModel")
    @torch.inference_mode()
    def predict(self, frame: StereoFrame, flow: torch.Tensor | None, depth: torch.Tensor | None) -> pp.LieTensor:
        if self.prev_pose is None:
            self.prev_pose = pp.identity_SE3(device=self.device)
            return self.prev_pose

        delta_pose = self._integrate_gyro_delta(frame)
        if delta_pose is None:
            return self.prev_pose

        predict = self.prev_pose @ delta_pose
        self.prev_pose = predict
        return predict

    def _integrate_gyro_delta(self, frame: StereoFrame) -> pp.LieTensor | None:
        if not isinstance(frame, StereoInertialFrame):
            return None

        imu = frame.imu
        if imu.gyro.numel() == 0 or imu.time_ns.numel() == 0:
            return None

        gyro = imu.gyro.reshape(-1, 3).to(device=self.device, dtype=torch.float32) - self.gyro_bias
        imu_time_ns = imu.time_ns.reshape(-1).to(device=self.device, dtype=torch.float64)

        if gyro.size(0) == 1:
            omega_int = gyro[0] * self.fallback_dt_s
        else:
            dt_s = (imu_time_ns[1:] - imu_time_ns[:-1]).clamp(min=1.0) * 1e-9
            gyro_mid = 0.5 * (gyro[:-1] + gyro[1:])
            omega_int = (gyro_mid.to(torch.float64) * dt_s.unsqueeze(-1)).sum(dim=0).to(torch.float32)

        angle = torch.linalg.norm(omega_int)
        if angle > self.max_delta_angle_rad:
            omega_int = omega_int * (self.max_delta_angle_rad / angle)

        delta_twist = torch.cat([torch.zeros(3, device=self.device), omega_int], dim=0)
        delta_I = pp.se3(delta_twist).Exp()
        return _camera_delta_from_raw_imu_delta(frame, delta_I)

    def update(self, pose: pp.LieTensor) -> None:
        self.prev_pose = pose.to(self.device)

    @classmethod
    def is_valid_config(cls, config: SimpleNamespace | None) -> None:
        cls._enforce_config_spec(config, {
            "device": lambda dev: isinstance(dev, str) and (("cuda" in dev) or (dev == "cpu")),
            "gyro_bias": lambda v: isinstance(v, list) and len(v) == 3 and all(isinstance(x, (float, int)) for x in v),
            "max_delta_angle_rad": lambda v: isinstance(v, (float, int)) and v > 0.0,
            "fallback_dt_s": lambda v: isinstance(v, (float, int)) and v > 0.0,
        }, allow_excessive_cfg=True)


class VisualHealthIMUGyroMotionModel(IMUGyroMotionModel):
    """
    Health-gated gyro motion initialization.

    The base MACVO static initialization is usually safest when stereo/flow is
    healthy. This model routes to gyro-only rotation only when online visual
    cues suggest degeneracy, which mirrors the health-monitoring idea used in
    robust underwater estimator switching without introducing scene labels.
    """

    def __init__(self, config: SimpleNamespace):
        super().__init__(config)
        self.min_gyro_angle_rad = float(getattr(config, "min_gyro_angle_rad", 0.025))
        self.min_depth_valid_ratio = float(getattr(config, "min_depth_valid_ratio", 0.35))
        self.min_depth_spread = float(getattr(config, "min_depth_spread", 0.08))
        self.low_flow_median_px = float(getattr(config, "low_flow_median_px", 0.35))
        self.force_static_when_healthy = bool(getattr(config, "force_static_when_healthy", True))

    @staticmethod
    def _flatten_flow(flow: torch.Tensor) -> torch.Tensor | None:
        f = flow.detach().float()
        if f.numel() == 0:
            return None
        if f.shape[-1] == 2:
            return f.reshape(-1, 2)
        if f.dim() >= 3 and f.shape[-3] == 2:
            return f.movedim(-3, -1).reshape(-1, 2)
        if f.dim() >= 2 and f.shape[0] == 2:
            return f.movedim(0, -1).reshape(-1, 2)
        if f.dim() >= 3 and f.shape[1] == 2:
            return f.movedim(1, -1).reshape(-1, 2)
        return None

    def _visual_health(self, flow: torch.Tensor | None, depth: torch.Tensor | None) -> tuple[bool, dict[str, float]]:
        depth_valid_ratio = 1.0
        depth_spread = 1.0
        if depth is not None and depth.numel() > 0:
            d = depth.detach().float().reshape(-1)
            valid = d[torch.isfinite(d) & (d > 1e-3)]
            depth_valid_ratio = float(valid.numel()) / max(float(d.numel()), 1.0)
            if valid.numel() > 1:
                depth_spread = float((valid.std() / valid.mean().clamp(min=1e-3)).clamp(0.0, 3.0).item())

        flow_median = 1.0
        if flow is not None:
            flow_flat = self._flatten_flow(flow)
            if flow_flat is not None and flow_flat.numel() > 0:
                norm = torch.linalg.norm(flow_flat, dim=-1)
                norm = norm[torch.isfinite(norm)]
                if norm.numel() > 0:
                    flow_median = float(norm.median().item())

        degraded = (
            depth_valid_ratio < self.min_depth_valid_ratio
            or depth_spread < self.min_depth_spread
            or flow_median < self.low_flow_median_px
        )
        return degraded, {
            "depth_valid_ratio": depth_valid_ratio,
            "depth_spread": depth_spread,
            "flow_median": flow_median,
        }

    @Timer.cpu_timeit("MotionModel")
    @Timer.gpu_timeit("MotionModel")
    @torch.inference_mode()
    def predict(self, frame: StereoFrame, flow: torch.Tensor | None, depth: torch.Tensor | None) -> pp.LieTensor:
        if self.prev_pose is None:
            self.prev_pose = pp.identity_SE3(device=self.device)
            return self.prev_pose

        delta_pose = self._integrate_gyro_delta(frame)
        if delta_pose is None:
            return self.prev_pose

        gyro_angle = float(torch.linalg.norm(delta_pose.Log().tensor().reshape(-1)[3:6]).item())
        degraded, _ = self._visual_health(flow, depth)
        use_gyro = gyro_angle >= self.min_gyro_angle_rad and (degraded or not self.force_static_when_healthy)
        if not use_gyro:
            return self.prev_pose

        predict = self.prev_pose @ delta_pose
        self.prev_pose = predict
        return predict

    @classmethod
    def is_valid_config(cls, config: SimpleNamespace | None) -> None:
        super().is_valid_config(config)
        cls._enforce_config_spec(config, {
            "min_gyro_angle_rad": lambda v: isinstance(v, (float, int)) and v >= 0.0,
            "min_depth_valid_ratio": lambda v: isinstance(v, (float, int)) and 0.0 <= float(v) <= 1.0,
            "min_depth_spread": lambda v: isinstance(v, (float, int)) and v >= 0.0,
            "low_flow_median_px": lambda v: isinstance(v, (float, int)) and v >= 0.0,
            "force_static_when_healthy": lambda v: isinstance(v, bool),
        }, allow_excessive_cfg=True)


class IMUConstVelMotionModel(IMotionModel[StereoInertialFrame]):
    """
    Hybrid motion model: IMU gyro for rotation + constant velocity for translation.

    Maintains a running estimate of linear velocity from recent visual pose differences.
    Uses IMU gyro integration for rotation prediction (no drift accumulation since
    rotation is per-frame delta). Translation predicts via constant velocity model.

    Falls back to static model when visual tracking is poor or IMU data is missing.
    """

    def __init__(self, config: SimpleNamespace):
        super().__init__(config)
        self.device = getattr(config, "device", "cpu")
        self.prev_pose: pp.LieTensor | None = None
        self.prev_velocity: torch.Tensor | None = None  # (3,) in world frame
        self.vel_window = int(getattr(config, "vel_window", 5))
        self.vel_history: list[torch.Tensor] = []

        gyro_bias = getattr(config, "gyro_bias", [0.0, 0.0, 0.0])
        self.gyro_bias = torch.tensor(gyro_bias, dtype=torch.float32, device=self.device)
        self.max_delta_angle_rad = float(getattr(config, "max_delta_angle_rad", 0.8))
        self.fallback_dt_s = float(getattr(config, "fallback_dt_s", 0.01))

    def _integrate_gyro(self, frame: StereoInertialFrame) -> pp.LieTensor | None:
        """Integrate raw gyro and express the relative motion in the camera frame."""
        imu = frame.imu
        if imu.gyro.numel() == 0 or imu.time_ns.numel() == 0:
            return None
        gyro = imu.gyro.reshape(-1, 3).to(device=self.device, dtype=torch.float32) - self.gyro_bias
        imu_time_ns = imu.time_ns.reshape(-1).to(device=self.device, dtype=torch.float64)

        if gyro.size(0) == 1:
            omega_int = gyro[0] * self.fallback_dt_s
        else:
            dt_s = (imu_time_ns[1:] - imu_time_ns[:-1]).clamp(min=1.0) * 1e-9
            gyro_mid = 0.5 * (gyro[:-1] + gyro[1:])
            omega_int = (gyro_mid.to(torch.float64) * dt_s.unsqueeze(-1)).sum(dim=0).to(torch.float32)

        angle = torch.linalg.norm(omega_int)
        if angle > self.max_delta_angle_rad:
            omega_int = omega_int * (self.max_delta_angle_rad / angle)
        delta_I = pp.se3(torch.cat([torch.zeros(3, device=self.device), omega_int], dim=0)).Exp()
        return _camera_delta_from_raw_imu_delta(frame, delta_I)

    @Timer.cpu_timeit("MotionModel")
    @Timer.gpu_timeit("MotionModel")
    @torch.inference_mode()
    def predict(self, frame: StereoFrame, flow: torch.Tensor | None, depth: torch.Tensor | None) -> pp.LieTensor:
        if self.prev_pose is None:
            self.prev_pose = pp.identity_SE3(device=self.device)
            return self.prev_pose

        # Default: repeat last pose (StaticMotionModel fallback)
        imu_delta_camera = None
        if isinstance(frame, StereoInertialFrame):
            imu_delta_camera = self._integrate_gyro(frame)

        if imu_delta_camera is None:
            # No IMU data → pure static
            return self.prev_pose

        # Translation from constant velocity model
        if self.prev_velocity is not None and self.prev_velocity.norm().item() > 1e-6:
            # Use the frame's timestamp difference if available, else fallback
            delta_t = self.fallback_dt_s
            delta_twist_trans = self.prev_velocity * delta_t
        else:
            delta_twist_trans = torch.zeros(3, device=self.device)

        delta_translation = pp.se3(torch.cat([
            delta_twist_trans,
            torch.zeros(3, device=self.device),
        ], dim=0)).Exp()
        delta_pose = delta_translation @ imu_delta_camera

        predict = self.prev_pose @ delta_pose
        self.prev_pose = predict
        return predict

    def update(self, pose: pp.LieTensor) -> None:
        """Update velocity estimate from pose change."""
        if self.prev_pose is not None:
            prev = self.prev_pose.to(self.device)
            cur = pose.to(self.device)
            rel = pp.SE3(prev.Inv() @ cur)
            trans = rel.translation().reshape(3)
            # Approximate dt from motion model frame rate
            dt = self.fallback_dt_s
            if dt > 1e-6:
                vel = trans / dt
                self.vel_history.append(vel)
                if len(self.vel_history) > self.vel_window:
                    self.vel_history.pop(0)
                if len(self.vel_history) > 0:
                    self.prev_velocity = torch.stack(self.vel_history).mean(dim=0)
        self.prev_pose = pose.to(self.device)

    @classmethod
    def is_valid_config(cls, config: SimpleNamespace | None) -> None:
        cls._enforce_config_spec(config, {
            "device": lambda dev: isinstance(dev, str) and (("cuda" in dev) or (dev == "cpu")),
            "gyro_bias": lambda v: isinstance(v, list) and len(v) == 3 and all(isinstance(x, (float, int)) for x in v),
            "max_delta_angle_rad": lambda v: isinstance(v, (float, int)) and v > 0.0,
            "fallback_dt_s": lambda v: isinstance(v, (float, int)) and v > 0.0,
            "vel_window": lambda v: isinstance(v, int) and v >= 1,
        }, allow_excessive_cfg=True)


class ReadPoseFile(IMotionModel[StereoFrame]):
    """
    Use an external file of Nx7 SE3 poses as motion model output poses.

    NOTE: Specifically, the module will *not* output these poses directly but calculate the motion
    and apply motion on modified poses (potentially by optimizer) iteratively.
    """
    def __init__(self, config: SimpleNamespace):
        super().__init__(config)
        self.prev_pose: None | pp.LieTensor = None
        self.prev_gt_pose: None | pp.LieTensor = None
        self.poses: pp.LieTensor = self.load_poses()

    def load_poses(self) -> pp.LieTensor:
        pose_file = Path(self.config.pose_file)
        if not pose_file.exists():
            Logger.write("error", f"Cannot read pose file at {pose_file} - File Not Exist!")
            raise FileNotFoundError(f"Cannot read pose file at {pose_file} - File Not Exist!")

        poses: pp.LieTensor
        match pose_file.suffix:
            case ".npy":
                import numpy as np
                poses_data: torch.Tensor = torch.from_numpy(np.load(str(pose_file)))
            case ".pt" | ".pth":
                poses_data: torch.Tensor = torch.load(str(pose_file), weights_only=False)
            case ".txt":
                import numpy as np
                poses_data: torch.Tensor = torch.from_numpy(np.loadtxt(str(pose_file)))
            case suffix:
                raise NameError(f"Cannot handle a file with suffix '{suffix}'. Consider change it to .npy/.pt/.pth/.txt or write a custom loader.")
        assert poses_data.ndim == 2 and poses_data.shape[1] == 7
        poses = pp.SE3(poses_data)
        return poses

    def predict(self, frame: StereoFrame, flow: torch.Tensor | None, depth: torch.Tensor | None) -> pp.LieTensor:
        if self.prev_pose is None or self.prev_gt_pose is None:
            self.prev_pose = pp.identity_SE3()
            self.prev_gt_pose = pp.SE3(self.poses[frame.frame_idx])
            return pp.identity_SE3()

        motion = self.prev_gt_pose.Inv() @ pp.SE3(self.poses[frame.frame_idx])
        predict = self.prev_pose @ motion

        self.prev_pose = predict
        self.prev_gt_pose = pp.SE3(self.poses[frame.frame_idx])
        return predict

    def update(self, pose: pp.LieTensor) -> None:
        self.prev_pose = pose

    @classmethod
    def is_valid_config(cls, config: SimpleNamespace | None) -> None:
        cls._enforce_config_spec(config, {
            "pose_file": lambda s: isinstance(s, str)
        })
