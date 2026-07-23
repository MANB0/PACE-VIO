import csv
import json
import time
import traceback
import torch
import pypose as pp
import typing as T
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

from rich.columns import Columns
from rich.panel import Panel
from typing import Callable

import Module
from DataLoader import StereoFrame, StereoInertialFrame
from Module.Map import VisualMap, FrameNode, MatchObs, PointNode
from Module.IMUPreintegration import (
    CURRENT_INDEPENDENT_STEP,
    SAMPLING_AWARE,
    SAMPLING_AWARE_CROSS_EDGE,
    build_sampling_aware_covariance_components,
    normalize_preintegration_covariance_mode,
    preintegrate_imu,
    preintegrate_imu_local_frame,
    replace_with_sampling_aware_covariance,
)
from Utility.Point import filterPointsInRange, pixel2point_NED
from Utility.IMUKinematics import (
    compose_adaptive_fallback_pose,
    compose_translation_prior_by_mode,
    estimate_static_imu_initialization,
    evaluate_adaptive_static_imu_initialization,
    gravity_for_world_frame,
    gravity_roll_pitch_aligned_rotation,
    gravity_is_standard_local_frame,
    imu_sigma_rms,
    integrate_gyro_attitude_world,
    is_valid_imu_sigma,
    normalize_gravity_handling,
    propagate_imu_velocity_world,
    resolve_translation_prior_mode,
    select_rotation_prior_std,
    select_translation_active_rotation_prior_std,
    should_enable_preintegrated_vio_factor,
    translation_prior_semantics,
)
from Utility.PrettyPrint import Logger, GlobalConsole
from Utility.QuaternionConvention import nwu_xyzw_quaternion_to_internal_so3
from Utility.VIOConventionDiagnostics import (
    camera_velocity_to_imu_origin,
    world_nwu_vector_to_internal,
)
from Utility.PoseFrame import write_timed_se3_csv, convert_pose_world_frame_only
from Utility.TrajectoryReference import compose_camera_to_imu_poses
from Utility.Timer import Timer
from Utility.Visualize import fig_plt
from Utility.VisualFactorCache import (
    FRONTEND_COVARIANCE_DIAGNOSTIC_FIELDS,
    VisualFactorCacheError,
    VisualFactorCacheReader,
)
from Utility.RelativePoseFactorCache import RelativePoseFactorCacheReader
from Utility.CompressedUVDFactorCache import CompressedUVDFactorCacheReader
from Utility.VisualInputFingerprint import visual_input_sha256
from Utility.TwoStateVIO import (
    UVDFactor,
    linearize_uvd_relative_pose_factor,
    solve_uvd_relative_pose_visual_only,
)
from Utility.Extensions import ConfigTestable
from Module.Frontend.Frontend import ReplayFrontend
from Module.KeyframeSelector import AllKeyframe
from Module.MotionModel import StaticMotionModel

from .Interface import IOdometry

T_SensorFrame = T.TypeVar("T_SensorFrame", bound=StereoFrame)


class MACVO(IOdometry[T_SensorFrame], ConfigTestable):
    # Type alias of callback hooks for MAC-VO system. Will be called by the system on
    # certain event occurs (optimization finish, for instance.)
    T_SYSHOOK = Callable[["MACVO",], None]

    def __init__(
        self,
        device, num_point, edgewidth, match_cov_default, profile, mapping,
        frontend        : Module.IFrontend,
        motion_model    : Module.IMotionModel[T_SensorFrame],
        kp_selector     : Module.IKeypointSelector,
        map_selector    : Module.IKeypointSelector,
        obs_filter      : Module.IObservationFilter,
        obs_covmodel    : Module.ICovariance2to3,
        post_process    : Module.IMapProcessor,
        kf_selector     : Module.IKeyframeSelector[T_SensorFrame],
        optimizer       : Module.IOptimizer,
        imu_rot_prior_std=0.12,
        imu_sigma_acc=0.3,
        imu_sigma_gyro=0.01,
        imu_rot_prior_std_when_translation=None,
        imu_sigma_acc_w=0.0,
        imu_sigma_gyro_w=0.0,
        imu_gravity_rp_correction_gain=0.0,
        imu_gravity_rp_acc_tol=0.15,
        imu_gravity_rp_window=24,
        imu_trans_prior_enable=True,
        imu_trans_prior_use_velocity=False,
        imu_trans_prior_mode=None,
        imu_rot_prior_enable=True,
        imu_pose_fusion_enable=False,
        imu_pose_fusion_alpha=0.7,
        imu_legacy_gyro_prior_enable=False,
        imu_legacy_gyro_sign=None,
        imu_vio_gravity_pose_source="estimated",
        imu_vio_gravity_handling="preintegration",
        two_state_covariance_mode=CURRENT_INDEPENDENT_STEP,
        imu_vio_velocity_feedback_enable=True,
        imu_vio_bias_feedback_enable=True,
        imu_static_initialization_enable=False,
        imu_static_initialization_mode=None,
        imu_static_initialization_state_policy="estimated",
        imu_static_initialization_duration_s=None,
        imu_static_adaptive_min_duration_s=1.0,
        imu_static_adaptive_max_duration_s=8.0,
        imu_static_adaptive_window_s=0.25,
        imu_static_adaptive_stable_hold_s=0.75,
        imu_static_adaptive_target_gyro_bias_sem=1.2e-3,
        imu_static_adaptive_target_gravity_direction_sem_rad=3.0e-3,
        imu_static_sigma_multiplier=5.0,
        imu_static_gyro_mean_norm_max=0.03,
        imu_static_acc_norm_error_max=0.6,
        visual_cache_mode: str = "off",
        visual_cache_path: str | None = None,
        pipeline_trace_path: str | None = None,
        **_excessive_args,
    ) -> None:
        super().__init__(profile=profile)
        if len(_excessive_args) > 0:
            Logger.write("warn", f"Receive excessive arguments for __init__ {_excessive_args}, update/clean up your config!")

        self.graph = VisualMap()
        self.device = device
        self.mapping: bool = mapping
        self.match_cov_default: float = match_cov_default
        self.imu_rot_prior_std: float = float(imu_rot_prior_std)
        self.imu_rot_prior_std_when_translation = (
            None
            if imu_rot_prior_std_when_translation is None
            else float(imu_rot_prior_std_when_translation)
        )
        self.imu_sigma_acc = imu_sigma_acc
        self.imu_sigma_gyro = imu_sigma_gyro
        self.imu_sigma_acc_w = imu_sigma_acc_w
        self.imu_sigma_gyro_w = imu_sigma_gyro_w
        self.imu_gravity_rp_correction_gain: float = float(imu_gravity_rp_correction_gain)
        self.imu_gravity_rp_acc_tol: float = float(imu_gravity_rp_acc_tol)
        self.imu_gravity_rp_window: int = int(imu_gravity_rp_window)
        self.imu_trans_prior_enable: bool = bool(imu_trans_prior_enable)
        self.imu_trans_prior_use_velocity: bool = bool(imu_trans_prior_use_velocity)
        self.imu_trans_prior_mode: str = resolve_translation_prior_mode(
            imu_trans_prior_mode,
            self.imu_trans_prior_use_velocity,
        )
        self.imu_translation_semantics: str = translation_prior_semantics(self.imu_trans_prior_mode)
        self.imu_rot_prior_enable: bool = bool(imu_rot_prior_enable)
        self.imu_pose_fusion_enable: bool = bool(imu_pose_fusion_enable)
        self.imu_pose_fusion_alpha: float = float(imu_pose_fusion_alpha)
        self.imu_legacy_gyro_prior_enable: bool = bool(imu_legacy_gyro_prior_enable)
        sign = [1.0, 1.0, 1.0] if imu_legacy_gyro_sign is None else list(imu_legacy_gyro_sign)
        self.imu_legacy_gyro_sign = torch.tensor(sign, dtype=torch.float32)
        self.imu_vio_gravity_pose_source: str = str(imu_vio_gravity_pose_source).strip().lower()
        self.imu_vio_gravity_handling: str = normalize_gravity_handling(imu_vio_gravity_handling)
        self.two_state_covariance_mode = normalize_preintegration_covariance_mode(
            two_state_covariance_mode
        )
        self.imu_vio_velocity_feedback_enable: bool = bool(imu_vio_velocity_feedback_enable)
        self.imu_vio_bias_feedback_enable: bool = bool(imu_vio_bias_feedback_enable)
        if imu_static_initialization_mode is None:
            static_mode = "fixed" if bool(imu_static_initialization_enable) else "off"
        else:
            static_mode = str(imu_static_initialization_mode).strip().lower()
        if static_mode not in {"fixed", "adaptive", "off"}:
            raise ValueError(
                "imu_static_initialization_mode must be fixed, adaptive or off; "
                f"got {imu_static_initialization_mode!r}"
            )
        static_state_policy = str(imu_static_initialization_state_policy).strip().lower()
        if static_state_policy not in {"estimated", "zero"}:
            raise ValueError(
                "imu_static_initialization_state_policy must be estimated or zero; "
                f"got {imu_static_initialization_state_policy!r}"
            )
        if static_state_policy == "zero" and static_mode == "off":
            raise ValueError(
                "zero static initialization state policy requires fixed or adaptive mode"
            )
        if static_mode == "fixed":
            if imu_static_initialization_duration_s is None:
                raise ValueError(
                    "fixed static initialization requires "
                    "imu_static_initialization_duration_s"
                )
            if float(imu_static_initialization_duration_s) <= 0.0:
                raise ValueError("fixed static initialization duration must be > 0")
        if static_mode == "adaptive":
            if float(imu_static_adaptive_min_duration_s) <= 0.0:
                raise ValueError("adaptive static minimum duration must be > 0")
            if float(imu_static_adaptive_max_duration_s) < float(
                imu_static_adaptive_min_duration_s
            ):
                raise ValueError(
                    "adaptive static maximum duration must be >= minimum duration"
                )
            if float(imu_static_adaptive_window_s) <= 0.0:
                raise ValueError("adaptive static window must be > 0")
            if float(imu_static_adaptive_stable_hold_s) < 2.0 * float(
                imu_static_adaptive_window_s
            ):
                raise ValueError(
                    "adaptive stable hold must be at least two detection windows"
                )
        self.imu_static_initialization_mode = static_mode
        self.imu_static_initialization_state_policy = static_state_policy
        self.imu_static_initialization_enable = static_mode != "off"
        self.imu_static_initialization_duration_s = (
            None
            if imu_static_initialization_duration_s is None
            else float(imu_static_initialization_duration_s)
        )
        self.imu_static_adaptive_min_duration_s = float(
            imu_static_adaptive_min_duration_s
        )
        self.imu_static_adaptive_max_duration_s = float(
            imu_static_adaptive_max_duration_s
        )
        self.imu_static_adaptive_window_s = float(imu_static_adaptive_window_s)
        self.imu_static_adaptive_stable_hold_s = float(
            imu_static_adaptive_stable_hold_s
        )
        self.imu_static_adaptive_target_gyro_bias_sem = float(
            imu_static_adaptive_target_gyro_bias_sem
        )
        self.imu_static_adaptive_target_gravity_direction_sem_rad = float(
            imu_static_adaptive_target_gravity_direction_sem_rad
        )
        self.imu_static_sigma_multiplier = float(imu_static_sigma_multiplier)
        self.imu_static_gyro_mean_norm_max = float(imu_static_gyro_mean_norm_max)
        self.imu_static_acc_norm_error_max = float(imu_static_acc_norm_error_max)
        self.visual_cache_mode = str(visual_cache_mode).strip().lower()
        self.visual_cache_path = visual_cache_path
        self.pipeline_trace_path = pipeline_trace_path
        self._pipeline_trace_stream = None
        self._pipeline_trace_writer = None
        self._pipeline_pending: dict | None = None
        if pipeline_trace_path:
            self._pipeline_trace_stream = open(pipeline_trace_path, "w", newline="", encoding="utf-8")
            self._pipeline_trace_writer = csv.DictWriter(
                self._pipeline_trace_stream,
                fieldnames=[
                    "frame_i", "frame_j", "timestamp_i_ns", "timestamp_j_ns",
                    "frontend_ms", "backend_solver_ms", "backend_wait_ms", "commit_ms",
                    "backend_submitted", "static_initialization_active",
                ],
            )
            self._pipeline_trace_writer.writeheader()
            self._pipeline_trace_stream.flush()
        self._visual_cache_reader: VisualFactorCacheReader | None = None
        self._relative_pose_factor_reader: RelativePoseFactorCacheReader | None = None
        self._compressed_uvd_factor_reader: CompressedUVDFactorCacheReader | None = None
        self._visual_cache_sequence_frame_count: int | None = None
        self._visual_cache_consumed_pairs: list[tuple[int, int]] = []
        # Pure-visual live sidecar.  This is intentionally separate from the
        # graph pose: the latter is VIO/T2 state and must never be reused as
        # the MACVO raw trajectory.
        self._live_macvo_raw_poses: dict[int, torch.Tensor] = {}
        self._live_macvo_raw_last_diagnostics: dict[str, T.Any] = {}

        if self.visual_cache_mode not in {"off", "replay"}:
            raise ValueError(f"visual_cache_mode must be 'off' or 'replay', got {visual_cache_mode!r}")
        if self.visual_cache_mode == "replay":
            if not isinstance(visual_cache_path, str) or not visual_cache_path:
                raise ValueError("visual_cache_path is required when visual_cache_mode='replay'")
            if mapping is not False:
                raise ValueError("visual cache replay requires mapping=False")
            if not isinstance(frontend, ReplayFrontend):
                raise ValueError("visual cache replay requires ReplayFrontend")
            if not isinstance(motion_model, StaticMotionModel):
                raise ValueError("visual cache replay requires StaticMotionModel")
            if not isinstance(kf_selector, AllKeyframe):
                raise ValueError("visual cache replay requires AllKeyframe")
            self._visual_cache_reader = VisualFactorCacheReader(visual_cache_path)

        # IMU velocity state: maintained across keyframes to enable acc preintegration
        # Velocity state is in MACVO's internal world frame (NED for the current visual graph), m/s.
        self._imu_vel_w: torch.Tensor = torch.zeros(3, dtype=torch.float32)
        self._imu_acc_bias: torch.Tensor = torch.zeros(3, dtype=torch.float32)
        self._imu_gyro_bias: torch.Tensor = torch.zeros(3, dtype=torch.float32)
        self._imu_last_frame_time_ns: int | None = None
        self._imu_attitude_world: pp.LieTensor | None = None
        self._imu_attitude_last_time_ns: int | None = None
        self._pending_imu_vio_factor: dict | None = None
        self._imu_static_initialized = self.imu_static_initialization_mode == "off"
        self._imu_static_time_chunks: list[torch.Tensor] = []
        self._imu_static_acc_chunks: list[torch.Tensor] = []
        self._imu_static_gyro_chunks: list[torch.Tensor] = []
        self._imu_static_last_time_ns: int | None = None
        self._imu_static_initial_rotation: pp.LieTensor | None = None
        self._imu_static_anchor_pose: pp.LieTensor | None = None
        self._imu_static_zupt_active = False
        self._imu_static_init_diag: dict | None = (
            {
                "mode": "off",
                "stationary": False,
                "duration_s": 0.0,
                "sample_count": 0,
                "status": "disabled",
            }
            if self._imu_static_initialized
            else None
        )

        # Modules
        self.Frontend = frontend
        self.MotionEstimator = motion_model
        self.KeypointSelector = kp_selector
        self.MappointSelector = map_selector
        self.OutlierFilter = obs_filter
        self.ObsCovModel = obs_covmodel
        self.MapRefiner = post_process
        self.KeyframeSelector = kf_selector
        self.Optimizer = optimizer
        optimizer_mode = str(
            getattr(self.Optimizer.config, "imu_factor_mode", "")
        ).strip().lower()
        two_state_visual_mode = str(
            getattr(self.Optimizer.config, "two_state_visual_factor_mode", "relative_pose")
        ).strip().lower()
        two_state_warm_start = str(
            getattr(self.Optimizer.config, "two_state_warm_start", "macvo_pose")
        ).strip().lower()
        if (
            optimizer_mode == "two_state_fixed_lag"
            and self._visual_cache_reader is None
            and two_state_visual_mode != "compressed_uvd"
        ):
            raise ValueError(
                "online two_state_fixed_lag requires two_state_visual_factor_mode='compressed_uvd'; "
                "relative_pose/direct_uvd currently require visual replay inputs"
            )
        needs_relative_pose_sidecar = (
            two_state_visual_mode == "relative_pose"
            or (
                two_state_warm_start == "macvo_pose"
                and two_state_visual_mode != "compressed_uvd"
            )
        )
        if (
            optimizer_mode == "two_state_fixed_lag"
            and self._visual_cache_reader is not None
            and needs_relative_pose_sidecar
        ):
            self._relative_pose_factor_reader = RelativePoseFactorCacheReader(visual_cache_path)
        if (
            optimizer_mode == "two_state_fixed_lag"
            and self._visual_cache_reader is not None
            and two_state_visual_mode == "compressed_uvd"
        ):
            self._compressed_uvd_factor_reader = CompressedUVDFactorCacheReader(
                visual_cache_path
            )
        if self.two_state_covariance_mode in {
            SAMPLING_AWARE,
            SAMPLING_AWARE_CROSS_EDGE,
        }:
            if optimizer_mode != "two_state_fixed_lag":
                raise ValueError("sampling-aware covariance is currently audited only for two_state_fixed_lag")
            if not gravity_is_standard_local_frame(self.imu_vio_gravity_handling):
                raise ValueError("sampling-aware covariance requires standard local-frame preintegration")
        # end

        self.min_num_point = 10
        self.num_point = num_point
        self.edge_width = edgewidth
        self.isinitiated = False

        # Context for tracking
        # [0] - Frame Source Data
        # [1] - Frame index (in visual map)
        # [2] - Frame stereo depth
        self.prev_keyframe: tuple[T_SensorFrame, int, Module.IStereoDepth.Output | None] | None = None

        # Hooks
        self.on_optimize_writeback: list[MACVO.T_SYSHOOK] = []

        # Per-pair diagnostics
        self._pending_imu_diag: dict | None = None   # IMU preint diag for current pair
        self._pending_cov_diag: dict | None = None    # frontend covariance stats for current pair
        self._pending_tracking_state: dict | None = None  # optimizer tracking state for current pair
        self._last_opt_diag: dict | None = None       # optimization diag from last finished pair
        self._diag_writer = None                       # FramePairDiagnosticsWriter, set by caller
        self._visual_factor_diag_stream = None
        self._visual_factor_diag_writer = None
        self._visual_factor_diag_written_pairs: set[tuple[int, int]] = set()
        self._pair_counter: int = 0
        self._last_written_frame_idx: int = -1          # avoid duplicate rows
        self._optimizer_finalize_summary: dict | None = None
        self._scene_name: str = ""
        self._method_name: str = ""
        self._gt_positions: dict | None = None           # {timestamp_ns: (x_nwu, y_nwu, z_nwu)}
        self._gt_quaternions: dict | None = None          # {timestamp_ns: (qx, qy, qz, qw)} or None
        self._gt_velocities: dict | None = None           # {timestamp_ns: world-NWU velocity (m/s)}
        self._gt_angular_velocities: dict | None = None   # {timestamp_ns: body-NWU angular velocity (rad/s)}

        # ── Adaptive v1 ───────────────────────────────────────────────────
        self._adaptive_enabled: bool = False
        self._adaptive_gate = None      # VisualHealthGate instance
        self._adaptive_decision: Any | None = None  # current pair decision
        self._adaptive_decisions_writer: Any | None = None  # CSV writer

        self.report_config()

    @classmethod
    def from_config(cls, cfg: SimpleNamespace):
        odomcfg = cfg.Odometry
        # Initialize modules for VO
        visual_cache_mode = str(getattr(odomcfg.args, "visual_cache_mode", "off")).strip().lower()
        Frontend = (
            ReplayFrontend(SimpleNamespace())
            if visual_cache_mode == "replay"
            else Module.IFrontend.instantiate(odomcfg.frontend.type, odomcfg.frontend.args)
        )
        MotionEstimator     = Module.IMotionModel[T_SensorFrame].instantiate(odomcfg.motion.type, odomcfg.motion.args)
        KeypointSelector    = Module.IKeypointSelector.instantiate(odomcfg.keypoint.type, odomcfg.keypoint.args)
        MappointSelector    = Module.IKeypointSelector.instantiate(odomcfg.mappoint.type, odomcfg.mappoint.args)
        ObservationFilter   = Module.IObservationFilter.instantiate(odomcfg.outlier.type, odomcfg.outlier.args)
        ObserveCovModel     = Module.ICovariance2to3.instantiate(odomcfg.cov.obs.type, odomcfg.cov.obs.args)
        MapRefiner          = Module.IMapProcessor.instantiate(odomcfg.postprocess.type, odomcfg.postprocess.args)
        KeyframeSelector    = Module.IKeyframeSelector[T_SensorFrame].instantiate(odomcfg.keyframe.type, odomcfg.keyframe.args)
        Optimizer           = Module.IOptimizer.instantiate(odomcfg.optimizer.type, odomcfg.optimizer.args)

        return cls(
            frontend=Frontend,
            motion_model=MotionEstimator,
            kp_selector=KeypointSelector,
            map_selector=MappointSelector,
            obs_filter=ObservationFilter,
            obs_covmodel=ObserveCovModel,
            post_process=MapRefiner,
            kf_selector=KeyframeSelector,
            optimizer=Optimizer,
            **vars(odomcfg.args),
        )

    def _frame_imu_vio_sensor_T_imu(self, frame: T_SensorFrame) -> pp.LieTensor:
        sensor_T_imu = getattr(frame, "imu_vio_sensor_T_imu", None)
        if sensor_T_imu is None:
            return pp.identity_SE3(1, dtype=torch.float32)
        return pp.SE3(sensor_T_imu).float()

    def report_config(self):
        # Cute fine-print boxes
        box1 = Panel.fit(
            "\n".join(
                [
                    f"DepthEstimator cov: {self.Frontend.provide_cov[0]}",
                    f"MatchEstimator cov: {self.Frontend.provide_cov[1]}",
                    f"Observation cov:    {self.ObsCovModel.__class__.__name__}",
                ]
            ),
            title="Odometry Covariance",
            title_align="left",
        )
        box2 = Panel.fit(
            "\n".join(
                [
                    f"Optimizer       -'{self.Optimizer       .__class__.__name__}'",
                    f"Frontend        -'{self.Frontend        .__class__.__name__}'",
                    f"MotionEstimator -'{self.MotionEstimator .__class__.__name__}'",
                    f"KeypointSelector-'{self.KeypointSelector.__class__.__name__}'",
                    f"MappointSelector-'{self.MappointSelector.__class__.__name__}'",
                    f"OutlierFilter   -'{self.OutlierFilter   .__class__.__name__}'",
                    f"MapRefiner      -'{self.MapRefiner      .__class__.__name__}'",
                ]
            ),
            title="Odometry Modules",
            title_align="left",
        )
        GlobalConsole.print(Columns([box1, box2]))

    @classmethod
    def is_valid_config(cls, config: SimpleNamespace | None) -> None:
        assert config is not None
        Module.IKeyframeSelector.is_valid_config(config.keyframe)
        Module.IMapProcessor.is_valid_config(config.postprocess)
        Module.IObservationFilter.is_valid_config(config.outlier)
        Module.IMotionModel.is_valid_config(config.motion)
        Module.IKeypointSelector.is_valid_config(config.keypoint)
        Module.ICovariance2to3.is_valid_config(config.cov.obs)
        Module.IFrontend.is_valid_config(config.frontend)
        Module.IOptimizer.is_valid_config(config.optimizer)

        cls._enforce_config_spec(config.args, {
            "device"            : lambda s: isinstance(s, str) and (("cuda" in s) or (s == "cpu")),
            "num_point"         : lambda b: isinstance(b, int) and b > 0,
            "edgewidth"         : lambda b: isinstance(b, int) and b > 0,
            "match_cov_default" : lambda b: isinstance(b, (float, int)) and b > 0.0,
            "profile"           : lambda b: isinstance(b, bool),
            "mapping"           : lambda b: isinstance(b, bool),
        }, allow_excessive_cfg=True)

        optional_args_spec = {
            "imu_rot_prior_std" : lambda b: isinstance(b, (float, int)) and b > 0.0,
            "imu_sigma_acc": is_valid_imu_sigma,
            "imu_sigma_gyro": is_valid_imu_sigma,
            "imu_rot_prior_std_when_translation": lambda b: b is None or (isinstance(b, (float, int)) and b > 0.0),
            "imu_sigma_acc_w": is_valid_imu_sigma,
            "imu_sigma_gyro_w": is_valid_imu_sigma,
            "imu_gravity_rp_correction_gain": lambda b: isinstance(b, (float, int)) and 0.0 <= float(b) <= 1.0,
            "imu_gravity_rp_acc_tol": lambda b: isinstance(b, (float, int)) and 0.0 <= float(b) <= 1.0,
            "imu_gravity_rp_window": lambda b: isinstance(b, int) and b >= 1,
            "imu_trans_prior_enable": lambda b: isinstance(b, bool),
            "imu_trans_prior_use_velocity": lambda b: isinstance(b, bool),
            "imu_pose_fusion_enable": lambda b: isinstance(b, bool),
            "imu_pose_fusion_alpha": lambda b: isinstance(b, (float, int)) and 0.0 <= float(b) <= 1.0,
            "imu_trans_prior_mode": lambda s: s is None or str(s).strip().lower() in {"", "off", "damping_delta_p", "visual_velocity_composed", "imu_velocity_composed"},
            "imu_rot_prior_enable": lambda b: isinstance(b, bool),
            "imu_legacy_gyro_prior_enable": lambda b: isinstance(b, bool),
            "imu_legacy_gyro_sign": lambda v: isinstance(v, list) and len(v) == 3 and all(isinstance(x, (float, int)) for x in v),
            "imu_vio_gravity_pose_source": lambda s: str(s).strip().lower() in {"estimated", "gt_ref", "imu_gravity_rp", "imu_integrated_estinit"},
            "imu_vio_gravity_handling": lambda s: str(s).strip().lower() in {
                "preintegration",
                "residual",
                "legacy_external_attitude_gravity_compensation",
                "standard_local_frame_preintegration",
            },
            "two_state_covariance_mode": lambda s: str(s).strip().lower() in {
                CURRENT_INDEPENDENT_STEP,
                SAMPLING_AWARE,
                SAMPLING_AWARE_CROSS_EDGE,
            },
            "imu_vio_velocity_feedback_enable": lambda b: isinstance(b, bool),
            "imu_vio_bias_feedback_enable": lambda b: isinstance(b, bool),
            "imu_static_initialization_enable": lambda b: isinstance(b, bool),
            "imu_static_initialization_mode": lambda s: str(s).strip().lower() in {"fixed", "adaptive", "off"},
            "imu_static_initialization_state_policy": lambda s: str(s).strip().lower() in {"estimated", "zero"},
            "imu_static_initialization_duration_s": lambda b: b is None or (isinstance(b, (float, int)) and float(b) > 0.0),
            "imu_static_adaptive_min_duration_s": lambda b: isinstance(b, (float, int)) and float(b) > 0.0,
            "imu_static_adaptive_max_duration_s": lambda b: isinstance(b, (float, int)) and float(b) > 0.0,
            "imu_static_adaptive_window_s": lambda b: isinstance(b, (float, int)) and float(b) > 0.0,
            "imu_static_adaptive_stable_hold_s": lambda b: isinstance(b, (float, int)) and float(b) > 0.0,
            "imu_static_adaptive_target_gyro_bias_sem": lambda b: isinstance(b, (float, int)) and float(b) > 0.0,
            "imu_static_adaptive_target_gravity_direction_sem_rad": lambda b: isinstance(b, (float, int)) and float(b) > 0.0,
            "imu_static_sigma_multiplier": lambda b: isinstance(b, (float, int)) and float(b) > 0.0,
            "imu_static_gyro_mean_norm_max": lambda b: isinstance(b, (float, int)) and float(b) >= 0.0,
            "imu_static_acc_norm_error_max": lambda b: isinstance(b, (float, int)) and float(b) >= 0.0,
            "visual_cache_mode": lambda s: isinstance(s, str) and s.strip().lower() in {"off", "replay"},
            "visual_cache_path": lambda p: p is None or isinstance(p, str),
        }
        for key, test_fn in optional_args_spec.items():
            if key in config.args.__dict__ and not test_fn(config.args.__dict__[key]):
                raise ValueError(f"Config does not match specification! ({key}={config.args.__dict__[key]!r})")

    def initialize(self, frame0: T_SensorFrame):
        if self._visual_cache_reader is not None:
            self._initialize_visual_cache_replay(frame0)
            return

        depth0          = self.Frontend.estimate_depth(frame0.stereo)
        est_pose        = self.MotionEstimator.predict(frame0, None, depth0.depth).unsqueeze(0)
        # Keep the frontend pose before any VIO fusion for the live MACVO-raw track.
        self._live_macvo_raw_pose = est_pose.tensor().detach().clone()

        if isinstance(frame0, StereoInertialFrame):
            self._imu_last_frame_time_ns = frame0.stereo.frame_ns

        frame_idx = self.graph.frames.push(FrameNode.init({
            "pose"        : est_pose,
            "T_BS"        : frame0.stereo.T_BS,
            "need_interp" : torch.tensor([0], dtype=torch.bool),
            "time_ns"     : torch.tensor([frame0.stereo.frame_ns], dtype=torch.long),
            "K"           : frame0.stereo.K,
            "baseline"    : frame0.stereo.baseline,
            "imu_rotvec_prior": torch.zeros((1, 3), dtype=torch.float32),
            "imu_rot_prior_std": torch.tensor([1e6], dtype=torch.float32),
            "imu_trans_prior": torch.zeros((1, 3), dtype=torch.float32),
            "imu_trans_cov"  : torch.eye(3, dtype=torch.float32).unsqueeze(0) * 1e6,
            "imu_vio_prev_velocity_world": torch.zeros((1, 3), dtype=torch.float32),
            "imu_vio_curr_velocity_init_world": torch.zeros((1, 3), dtype=torch.float32),
            "imu_vio_velocity_world": torch.zeros((1, 3), dtype=torch.float32),
            "imu_vio_prev_acc_bias": torch.zeros((1, 3), dtype=torch.float32),
            "imu_vio_prev_gyro_bias": torch.zeros((1, 3), dtype=torch.float32),
            "imu_vio_acc_bias": torch.zeros((1, 3), dtype=torch.float32),
            "imu_vio_gyro_bias": torch.zeros((1, 3), dtype=torch.float32),
            "imu_vio_linearized_acc_bias": torch.zeros((1, 3), dtype=torch.float32),
            "imu_vio_linearized_gyro_bias": torch.zeros((1, 3), dtype=torch.float32),
            "imu_vio_bias_jacobian": torch.zeros((1, 9, 6), dtype=torch.float32),
            "imu_vio_bias_rw_cov": torch.eye(6, dtype=torch.float32).unsqueeze(0) * 1e6,
            "imu_vio_delta_rotvec": torch.zeros((1, 3), dtype=torch.float32),
            "imu_vio_delta_v": torch.zeros((1, 3), dtype=torch.float32),
            "imu_vio_delta_p": torch.zeros((1, 3), dtype=torch.float32),
            "imu_vio_cov": torch.eye(9, dtype=torch.float32).unsqueeze(0) * 1e6,
            "imu_vio_sa_v2_unique_cov": torch.eye(9, dtype=torch.float32).unsqueeze(0) * 1e6,
            "imu_vio_sa_v2_incoming_raw_time_ns": torch.full((1, 2), -1, dtype=torch.long),
            "imu_vio_sa_v2_outgoing_raw_time_ns": torch.full((1, 2), -1, dtype=torch.long),
            "imu_vio_sa_v2_incoming_count": torch.zeros(1, dtype=torch.long),
            "imu_vio_sa_v2_outgoing_count": torch.zeros(1, dtype=torch.long),
            "imu_vio_sa_v2_incoming_sensitivity": torch.zeros((1, 9, 12), dtype=torch.float32),
            "imu_vio_sa_v2_outgoing_sensitivity": torch.zeros((1, 9, 12), dtype=torch.float32),
            "imu_vio_dt": torch.tensor([0.0], dtype=torch.float32),
            "imu_vio_sensor_T_imu": self._frame_imu_vio_sensor_T_imu(frame0),
            "imu_vio_gravity_world": torch.zeros((1, 3), dtype=torch.float32),
            "imu_vio_gravity_in_residual": torch.tensor([False], dtype=torch.bool),
            "visual_relative_pose_CiCj": pp.identity_SE3(1, dtype=torch.float32).tensor(),
            "visual_relative_pose_cov": torch.eye(6, dtype=torch.float32).unsqueeze(0) * 1e6,
            "visual_relative_pose_num_points": torch.tensor([0], dtype=torch.long),
            "visual_relative_pose_num_inliers": torch.tensor([0], dtype=torch.long),
            "visual_relative_pose_mean_mahalanobis_sq": torch.tensor([-1.0], dtype=torch.float32),
            "visual_compressed_uvd_reference_CjCi": pp.identity_SE3(1, dtype=torch.float64).tensor(),
            "visual_compressed_uvd_hessian": torch.zeros((1, 6, 6), dtype=torch.float64),
            "visual_compressed_uvd_gradient": torch.zeros((1, 6), dtype=torch.float64),
            "visual_compressed_uvd_robust_cost": torch.tensor([-1.0], dtype=torch.float64),
            "visual_compressed_uvd_num_points": torch.tensor([0], dtype=torch.long),
            "visual_compressed_uvd_num_inliers": torch.tensor([0], dtype=torch.long),
            "visual_compressed_uvd_mean_mahalanobis_sq": torch.tensor([-1.0], dtype=torch.float64),
            "visual_compressed_uvd_huber_delta": torch.tensor([0.1], dtype=torch.float64),
            "fusion_visual_quality": torch.tensor([-1.0], dtype=torch.float32),
            "fusion_degrade_score": torch.tensor([-1.0], dtype=torch.float32),
            "fusion_trans_switch": torch.tensor([-1.0], dtype=torch.float32),
            "fusion_rot_switch": torch.tensor([-1.0], dtype=torch.float32),
            "fusion_xy_weight": torch.tensor([0.0], dtype=torch.float32),
            "fusion_z_weight": torch.tensor([0.0], dtype=torch.float32),
            "fusion_rot_weight": torch.tensor([0.0], dtype=torch.float32),
            "fusion_gate_flags": torch.zeros((1, 4), dtype=torch.float32),
        }))
        self.OutlierFilter.set_meta(frame0.stereo)
        self.prev_keyframe = (frame0, int(frame_idx.item()), depth0)
        self._live_macvo_raw_poses[int(frame_idx.item())] = (
            est_pose.tensor().detach().clone()
        )

    def _update_live_macvo_raw_pose(self, graph_data: T.Any) -> None:
        """Run the pure-visual MACVO sidecar for one already-built pair.

        ``graph_data`` is used only for the current pair's source-frame UVD
        observations.  No pose, velocity, bias, IMU factor, prior, or T2
        result enters this solve.  The solved camera relative motion is then
        chained in an independent camera trajectory.  In online T2 mode the
        same visual optimum is also the local UVD compression point and pose
        warm-start; it is not added as a second pose measurement.
        """
        points_Ci = graph_data.points.data.get("pos_Tc")
        target_uv = graph_data.observations.data.get("pixel2_uv")
        target_disparity = graph_data.observations.data.get("pixel2_disp")
        target_uv_cov = graph_data.observations.data.get("pixel2_uv_cov")
        target_disparity_cov = graph_data.observations.data.get("pixel2_disp_cov")
        fields = (
            points_Ci,
            target_uv,
            target_disparity,
            target_uv_cov,
            target_disparity_cov,
        )
        if any(value is None for value in fields):
            self._live_macvo_raw_last_diagnostics = {
                "available": False,
                "reason": "missing_online_uvd_fields",
            }
            return

        # The UVD whitening path intentionally evaluates the normal equations
        # in float64.  Promote the observations here as well so the residual
        # and its autodiff Jacobian have the same scalar type.
        points_Ci = points_Ci.reshape(-1, 3).double()
        target_uv = target_uv.reshape(-1, 2).to(points_Ci)
        target_disparity = target_disparity.reshape(-1, 1).to(points_Ci)
        target_uv_cov = target_uv_cov.reshape(-1, 3).to(points_Ci)
        target_disparity_cov = target_disparity_cov.reshape(-1).to(points_Ci)
        count = int(points_Ci.shape[0])
        if count < 3 or any(int(value.shape[0]) != count for value in (
            target_uv, target_disparity, target_uv_cov, target_disparity_cov.reshape(-1, 1)
        )):
            self._live_macvo_raw_last_diagnostics = {
                "available": False,
                "reason": "online_uvd_row_mismatch",
                "point_count": count,
            }
            return

        covariance = torch.zeros(
            (count, 3, 3), dtype=points_Ci.dtype, device=points_Ci.device
        )
        covariance[:, 0, 0] = target_uv_cov[:, 0]
        covariance[:, 1, 1] = target_uv_cov[:, 1]
        covariance[:, 0, 1] = target_uv_cov[:, 2]
        covariance[:, 1, 0] = target_uv_cov[:, 2]
        covariance[:, 2, 2] = target_disparity_cov
        if not bool(torch.isfinite(torch.cat([
            points_Ci.reshape(-1), target_uv.reshape(-1),
            target_disparity.reshape(-1), covariance.reshape(-1),
        ])).all()):
            self._live_macvo_raw_last_diagnostics = {
                "available": False,
                "reason": "nonfinite_online_uvd_fields",
                "point_count": count,
            }
            return

        identity_extrinsic = pp.identity_SE3(
            1, dtype=points_Ci.dtype, device=points_Ci.device
        ).tensor()
        visual = UVDFactor(
            points_Ci=points_Ci,
            target_uv=target_uv,
            target_disparity=target_disparity,
            covariance_uvd=covariance,
            intrinsic=graph_data.images_intrinsic.to(points_Ci),
            baseline=float(graph_data.baseline.reshape(-1)[0].item()),
            extrinsic_CI=identity_extrinsic,
            huber_delta=float(getattr(
                self.Optimizer.config, "two_state_uvd_huber_delta", 0.1
            )),
        )
        try:
            relative_CjCi, diagnostics = solve_uvd_relative_pose_visual_only(
                visual,
                max_iterations=int(getattr(
                    self.Optimizer.config, "two_state_max_iterations", 20
                )),
                damping=1.0e-6,
            )
            visual_mode = str(getattr(
                self.Optimizer.config,
                "two_state_visual_factor_mode",
                "relative_pose",
            )).strip().lower()
            optimizer_mode = str(getattr(
                self.Optimizer.config,
                "imu_factor_mode",
                "",
            )).strip().lower()
            if (
                optimizer_mode == "two_state_fixed_lag"
                and visual_mode == "compressed_uvd"
            ):
                compression = linearize_uvd_relative_pose_factor(
                    relative_CjCi,
                    visual,
                    marginal_mode="full",
                    normal_eigenvalue_floor=float(getattr(
                        self.Optimizer.config,
                        "two_state_marginalization_eigenvalue_floor",
                        1.0e-10,
                    )),
                )
                graph_data.visual_compressed_uvd_reference_CjCi = (
                    relative_CjCi.detach().clone()
                )
                graph_data.visual_compressed_uvd_hessian = (
                    compression.full_hessian.detach().clone()
                )
                graph_data.visual_compressed_uvd_gradient = (
                    compression.full_gradient.detach().clone()
                )
                graph_data.visual_compressed_uvd_robust_cost = float(
                    diagnostics["robust_cost"]
                )
                graph_data.visual_compressed_uvd_num_points = int(
                    diagnostics["num_points"]
                )
                graph_data.visual_compressed_uvd_num_inliers = int(
                    diagnostics["num_inliers"]
                )
                graph_data.visual_compressed_uvd_mean_mahalanobis_sq = float(
                    diagnostics["mean_mahalanobis_sq"]
                )
                graph_data.visual_compressed_uvd_huber_delta = float(
                    visual.huber_delta
                )
                if str(getattr(
                    self.Optimizer.config,
                    "two_state_warm_start",
                    "macvo_pose",
                )).strip().lower() == "macvo_pose":
                    relative_for_graph = relative_CjCi.to(
                        device=graph_data.from_pose.device,
                        dtype=graph_data.from_pose.dtype,
                    )
                    graph_data.init_motion = (
                        pp.SE3(graph_data.from_pose)
                        @ pp.SE3(relative_for_graph).Inv()
                    )
                diagnostics["t2_compression_source"] = "online_visual_optimum"
            from_idx = int(graph_data.from_idx.reshape(-1)[0].item())
            frame_idx = int(graph_data.frame_idx.reshape(-1)[0].item())
            previous_raw = self._live_macvo_raw_poses.get(from_idx)
            if previous_raw is None:
                # When startup IMU initialization suppresses the first visual
                # edges, seed the independent raw track at the active source
                # frame.  This keeps raw and T2 on the same post-static clock.
                if 0 <= from_idx < len(self.graph.frames):
                    previous_raw = self.graph.frames.data["pose"][from_idx].detach().clone()
                    self._live_macvo_raw_poses[from_idx] = previous_raw
                else:
                    self._live_macvo_raw_last_diagnostics = {
                        "available": False,
                        "reason": "raw_source_pose_missing",
                        "from_idx": from_idx,
                        "frame_idx": frame_idx,
                    }
                    return
            # The visual sidecar follows the same independent camera-frame
            # convention as the exported raw trajectory.  Keep the chained
            # pose and the newly solved relative pose in one dtype before
            # composing them; the live graph may store the seed as float32
            # while the visual solver can be configured for another dtype.
            relative_CjCi = relative_CjCi.to(
                device=previous_raw.device,
                dtype=previous_raw.dtype,
            )
            raw_next = pp.SE3(previous_raw) @ pp.SE3(relative_CjCi).Inv()
            raw_next_tensor = raw_next.tensor().detach().clone()
            if not bool(torch.isfinite(raw_next_tensor).all()):
                raise FloatingPointError("pure visual raw pose is non-finite")
            self._live_macvo_raw_poses[frame_idx] = raw_next_tensor
            self._live_macvo_raw_pose = raw_next_tensor
            self._live_macvo_raw_last_diagnostics = {
                "available": True,
                "from_idx": from_idx,
                "frame_idx": frame_idx,
                "point_count": count,
                "relative_CjCi": relative_CjCi.detach().cpu(),
                **diagnostics,
            }
        except Exception as error:
            self._live_macvo_raw_last_diagnostics = {
                "available": False,
                "reason": f"visual_only_solver_failed:{type(error).__name__}",
                "error": str(error)[:240],
                "point_count": count,
            }
            Logger.write(
                "warn",
                "MACVO raw visual-only sidecar failed: "
                f"{type(error).__name__}: {str(error)[:240]}\n"
                f"{traceback.format_exc(limit=8)}",
            )

    def _visual_cache_scene(self, frame: T_SensorFrame) -> str:
        scene = self._scene_name or getattr(frame, "scene", None) or getattr(frame, "scene_name", None)
        if not isinstance(scene, str) or not scene:
            raise VisualFactorCacheError("visual cache replay requires a scene name")
        return scene

    def _validate_visual_cache_sequence_metadata(self, sequence) -> None:
        reader = self._visual_cache_reader
        if reader is None:
            return

        try:
            sequence_frame_count = int(len(sequence))
        except Exception as error:
            raise VisualFactorCacheError("unable to read replay sequence frame_count") from error
        if sequence_frame_count != reader.manifest.frame_count:
            raise VisualFactorCacheError("replay sequence frame_count does not cover the complete cache range")
        self._visual_cache_sequence_frame_count = sequence_frame_count

        sequence_indices = getattr(sequence, "indices", None)
        if sequence_indices is None:
            return
        try:
            if hasattr(sequence_indices, "tolist"):
                actual_indices = [int(value) for value in sequence_indices.tolist()]
            else:
                actual_indices = [int(value) for value in sequence_indices]
        except Exception as error:
            raise VisualFactorCacheError("unable to read replay sequence source indices") from error
        expected_indices = list(range(reader.manifest.frame_count))
        if actual_indices != expected_indices:
            raise VisualFactorCacheError("replay sequence source indices do not cover the complete cache range")

    def receive_frames(self, sequence, saveto, on_frame_finished=None):
        self._validate_visual_cache_sequence_metadata(sequence)
        result = super().receive_frames(sequence, saveto, on_frame_finished)
        self._export_live_macvo_raw_trajectory(saveto, sequence)
        return result

    def _export_live_macvo_raw_trajectory(self, saveto, sequence) -> None:
        """Persist the same independent raw trajectory that the dashboard shows."""
        if not self._live_macvo_raw_poses:
            return
        ordered = sorted(self._live_macvo_raw_poses.items())
        if not ordered:
            return
        frame_indices = [index for index, _ in ordered]
        poses_camera = torch.cat([pose.reshape(1, 7) for _, pose in ordered], dim=0)
        time_ns = self.graph.frames.data["time_ns"][frame_indices].detach().cpu().numpy()
        poses_camera_np = poses_camera.detach().cpu().double().numpy()
        write_timed_se3_csv(
            saveto.path("macvo_raw_poses_camera.csv"),
            time_ns,
            poses_camera_np,
        )
        if "imu_vio_sensor_T_imu" not in self.graph.frames.data:
            return
        ext = self.graph.frames.data["imu_vio_sensor_T_imu"][frame_indices]
        ext_np = ext.detach().cpu().double().numpy()
        raw_imu_internal = compose_camera_to_imu_poses(poses_camera_np, ext_np)
        output_frame = str(getattr(sequence, "pose_output_frame", "NED")).upper()
        raw_imu_output = convert_pose_world_frame_only(raw_imu_internal, "NED", output_frame)
        write_timed_se3_csv(
            saveto.path("macvo_raw_poses_imu.csv"),
            time_ns,
            raw_imu_output,
        )

    def _validate_visual_cache_initial_frame(self, frame: T_SensorFrame) -> None:
        reader = self._visual_cache_reader
        if reader is None:
            raise RuntimeError("visual cache reader is unavailable in replay mode")

        source_frame_idx = int(frame.frame_idx)
        if source_frame_idx < 0 or source_frame_idx >= reader.manifest.frame_count:
            raise VisualFactorCacheError("initial source frame index differs from cache manifest")
        first_source_frame_idx = int(reader.manifest.pairs[0]["frame_i"])
        if source_frame_idx != first_source_frame_idx:
            raise VisualFactorCacheError("replay must start from the first cached source frame")
        if reader.manifest.timestamps_ns[source_frame_idx] != int(frame.stereo.frame_ns):
            raise VisualFactorCacheError("initial frame timestamp differs from cache manifest")

        reader.validate_run(
            scene=self._visual_cache_scene(frame),
            frame_count=self._visual_cache_sequence_frame_count or reader.manifest.frame_count,
            timestamps_ns=list(reader.manifest.timestamps_ns),
            K=frame.stereo.frame_K,
            baseline_m=frame.stereo.frame_baseline,
        )

    def _initialize_visual_cache_replay(self, frame0: T_SensorFrame) -> None:
        self._validate_visual_cache_initial_frame(frame0)
        est_pose = self.MotionEstimator.predict(frame0, None, None).unsqueeze(0)
        # Keep the frontend pose before any VIO fusion for the live MACVO-raw track.
        self._live_macvo_raw_pose = est_pose.tensor().detach().clone()

        if isinstance(frame0, StereoInertialFrame):
            self._imu_last_frame_time_ns = frame0.stereo.frame_ns

        frame_idx = self.graph.frames.push(FrameNode.init({
            "pose"        : est_pose,
            "T_BS"        : frame0.stereo.T_BS,
            "need_interp" : torch.tensor([0], dtype=torch.bool),
            "time_ns"     : torch.tensor([frame0.stereo.frame_ns], dtype=torch.long),
            "K"           : frame0.stereo.K,
            "baseline"    : frame0.stereo.baseline,
            "imu_rotvec_prior": torch.zeros((1, 3), dtype=torch.float32),
            "imu_rot_prior_std": torch.tensor([1e6], dtype=torch.float32),
            "imu_trans_prior": torch.zeros((1, 3), dtype=torch.float32),
            "imu_trans_cov"  : torch.eye(3, dtype=torch.float32).unsqueeze(0) * 1e6,
            "imu_vio_prev_velocity_world": torch.zeros((1, 3), dtype=torch.float32),
            "imu_vio_curr_velocity_init_world": torch.zeros((1, 3), dtype=torch.float32),
            "imu_vio_velocity_world": torch.zeros((1, 3), dtype=torch.float32),
            "imu_vio_prev_acc_bias": torch.zeros((1, 3), dtype=torch.float32),
            "imu_vio_prev_gyro_bias": torch.zeros((1, 3), dtype=torch.float32),
            "imu_vio_acc_bias": torch.zeros((1, 3), dtype=torch.float32),
            "imu_vio_gyro_bias": torch.zeros((1, 3), dtype=torch.float32),
            "imu_vio_linearized_acc_bias": torch.zeros((1, 3), dtype=torch.float32),
            "imu_vio_linearized_gyro_bias": torch.zeros((1, 3), dtype=torch.float32),
            "imu_vio_bias_jacobian": torch.zeros((1, 9, 6), dtype=torch.float32),
            "imu_vio_bias_rw_cov": torch.eye(6, dtype=torch.float32).unsqueeze(0) * 1e6,
            "imu_vio_delta_rotvec": torch.zeros((1, 3), dtype=torch.float32),
            "imu_vio_delta_v": torch.zeros((1, 3), dtype=torch.float32),
            "imu_vio_delta_p": torch.zeros((1, 3), dtype=torch.float32),
            "imu_vio_cov": torch.eye(9, dtype=torch.float32).unsqueeze(0) * 1e6,
            "imu_vio_sa_v2_unique_cov": torch.eye(9, dtype=torch.float32).unsqueeze(0) * 1e6,
            "imu_vio_sa_v2_incoming_raw_time_ns": torch.full((1, 2), -1, dtype=torch.long),
            "imu_vio_sa_v2_outgoing_raw_time_ns": torch.full((1, 2), -1, dtype=torch.long),
            "imu_vio_sa_v2_incoming_count": torch.zeros(1, dtype=torch.long),
            "imu_vio_sa_v2_outgoing_count": torch.zeros(1, dtype=torch.long),
            "imu_vio_sa_v2_incoming_sensitivity": torch.zeros((1, 9, 12), dtype=torch.float32),
            "imu_vio_sa_v2_outgoing_sensitivity": torch.zeros((1, 9, 12), dtype=torch.float32),
            "imu_vio_dt": torch.tensor([0.0], dtype=torch.float32),
            "imu_vio_sensor_T_imu": self._frame_imu_vio_sensor_T_imu(frame0),
            "imu_vio_gravity_world": torch.zeros((1, 3), dtype=torch.float32),
            "imu_vio_gravity_in_residual": torch.tensor([False], dtype=torch.bool),
            "visual_relative_pose_CiCj": pp.identity_SE3(1, dtype=torch.float32).tensor(),
            "visual_relative_pose_cov": torch.eye(6, dtype=torch.float32).unsqueeze(0) * 1e6,
            "visual_relative_pose_num_points": torch.tensor([0], dtype=torch.long),
            "visual_relative_pose_num_inliers": torch.tensor([0], dtype=torch.long),
            "visual_relative_pose_mean_mahalanobis_sq": torch.tensor([-1.0], dtype=torch.float32),
            "visual_compressed_uvd_reference_CjCi": pp.identity_SE3(1, dtype=torch.float64).tensor(),
            "visual_compressed_uvd_hessian": torch.zeros((1, 6, 6), dtype=torch.float64),
            "visual_compressed_uvd_gradient": torch.zeros((1, 6), dtype=torch.float64),
            "visual_compressed_uvd_robust_cost": torch.tensor([-1.0], dtype=torch.float64),
            "visual_compressed_uvd_num_points": torch.tensor([0], dtype=torch.long),
            "visual_compressed_uvd_num_inliers": torch.tensor([0], dtype=torch.long),
            "visual_compressed_uvd_mean_mahalanobis_sq": torch.tensor([-1.0], dtype=torch.float64),
            "visual_compressed_uvd_huber_delta": torch.tensor([0.1], dtype=torch.float64),
            "fusion_visual_quality": torch.tensor([-1.0], dtype=torch.float32),
            "fusion_degrade_score": torch.tensor([-1.0], dtype=torch.float32),
            "fusion_trans_switch": torch.tensor([-1.0], dtype=torch.float32),
            "fusion_rot_switch": torch.tensor([-1.0], dtype=torch.float32),
            "fusion_xy_weight": torch.tensor([0.0], dtype=torch.float32),
            "fusion_z_weight": torch.tensor([0.0], dtype=torch.float32),
            "fusion_rot_weight": torch.tensor([0.0], dtype=torch.float32),
            "fusion_gate_flags": torch.zeros((1, 4), dtype=torch.float32),
        }))
        self.prev_keyframe = (frame0, int(frame_idx.item()), None)
        self._live_macvo_raw_poses[int(frame_idx.item())] = (
            est_pose.tensor().detach().clone()
        )

    def _sync_optimized_vio_velocity_from_map(self) -> None:
        if self.prev_keyframe is None:
            return
        mode = str(getattr(self.Optimizer.config, "imu_factor_mode", "legacy_pose_prior")).strip().lower()
        if mode not in {"preintegrated_vio", "local_inertial_ba", "two_state_fixed_lag"}:
            return
        if not (self.imu_vio_velocity_feedback_enable or self.imu_vio_bias_feedback_enable):
            return
        frame_idx = int(self.prev_keyframe[1])
        if frame_idx < 0 or frame_idx >= len(self.graph.frames):
            return
        if self.imu_vio_velocity_feedback_enable and "imu_vio_velocity_world" in self.graph.frames.data:
            velocity = self.graph.frames.data["imu_vio_velocity_world"][frame_idx].reshape(3).float()
            if torch.isfinite(velocity).all():
                self._imu_vel_w = velocity.clone()
        if self.imu_vio_bias_feedback_enable:
            if "imu_vio_acc_bias" in self.graph.frames.data:
                acc_bias = self.graph.frames.data["imu_vio_acc_bias"][frame_idx].reshape(3).float()
                if torch.isfinite(acc_bias).all():
                    self._imu_acc_bias = acc_bias.clone()
            if "imu_vio_gyro_bias" in self.graph.frames.data:
                gyro_bias = self.graph.frames.data["imu_vio_gyro_bias"][frame_idx].reshape(3).float()
                if torch.isfinite(gyro_bias).all():
                    self._imu_gyro_bias = gyro_bias.clone()

    def _run_adaptive_d2_rerun(self) -> None:
        decision = self._adaptive_decision
        if decision is None or not getattr(decision, "d2_rerun_triggered", False):
            return

        pair_id = getattr(decision, "d2_rerun_pair_id", -1)
        frame_idx = self.prev_keyframe[1] if self.prev_keyframe else -1
        if frame_idx < 0 or frame_idx >= len(self.graph.frames):
            return

        imu_diag = self._pending_imu_diag or {}
        trans_prior_full = imu_diag.get("trans_prior_full")
        trans_cov_full = imu_diag.get("trans_cov_full")
        if trans_prior_full is None or trans_cov_full is None:
            decision.d2_rerun_failed = True
            decision.d2_rerun_failure_reason = "full IMU priors not available"
            Logger.write(
                "warn",
                f"D2 rerun SKIPPED pair={pair_id}: full IMU priors not in pending diag",
            )
            return

        try:
            pre_diag = getattr(self.Optimizer, "last_pair_diagnostics", {}) or {}
            decision.d2_pre_rerun_est_delta_t_norm = pre_diag.get("r_p_whitened_norm", float("nan"))
            decision.d2_pre_rerun_r_p_whitened_norm = pre_diag.get("r_p_whitened_norm", float("nan"))
            decision.d2_pre_rerun_num_vis_res = pre_diag.get("num_visual_residuals", 0)
            decision.d2_pre_rerun_mode = getattr(
                decision,
                "d2_pre_rerun_mode",
                pre_diag.get("mode", "unknown"),
            )

            frame_data = self.graph.frames[frame_idx].data
            frame_data["imu_trans_prior"] = trans_prior_full
            frame_data["imu_trans_cov"] = trans_cov_full

            graph_data = self.Optimizer.get_graph_data(
                self.graph,
                torch.tensor([frame_idx], dtype=torch.long),
            )
            result = self.Optimizer.sequential_optimize(graph_data)
            self.Optimizer.write_graph_data(result, self.graph)

            post_diag = getattr(self.Optimizer, "last_pair_diagnostics", {}) or {}
            decision.d2_post_rerun_est_delta_t_norm = float(
                post_diag.get("est_delta_t_norm", post_diag.get("r_p_whitened_norm", float("nan")))
            )
            decision.d2_post_rerun_r_p_whitened_norm = float(
                post_diag.get("r_p_whitened_norm", float("nan"))
            )
            decision.d2_post_rerun_num_vis_res = int(post_diag.get("num_visual_residuals", 0))
            decision.d2_post_rerun_mode = "full_imu_active"
            decision.d2_committed_result_source = "full_imu_rerun"
            decision.d2_rerun_failed = False

            Logger.write(
                "info",
                f"D2 rerun: pair={pair_id}, frame={frame_idx}, "
                f"pre_r_p={decision.d2_pre_rerun_r_p_whitened_norm:.3f}, "
                f"post_r_p={decision.d2_post_rerun_r_p_whitened_norm:.3f}, "
                f"pre_vis={decision.d2_pre_rerun_num_vis_res}, "
                f"post_vis={decision.d2_post_rerun_num_vis_res}",
            )
        except Exception as error:
            decision.d2_rerun_failed = True
            decision.d2_rerun_failure_reason = str(error)[:200]
            decision.d2_committed_result_source = "original"
            Logger.write(
                "warn",
                f"D2 rerun FAILED pair={pair_id}: {error}. "
                "Falling back to original rotation_only result.",
            )

    def _commit_previous_backend_result(
        self,
        frame0: T_SensorFrame,
        frame1: T_SensorFrame,
    ) -> None:
        commit_start = time.perf_counter()
        self.Optimizer.write_map(self.graph)
        commit_ms = (time.perf_counter() - commit_start) * 1000.0
        self._sync_optimized_vio_velocity_from_map()
        for func in self.on_optimize_writeback:
            func(self)
        self._update_adaptive_gate(frame0, frame1)
        self._run_adaptive_d2_rerun()
        self._write_adaptive_decision_csv()
        self._write_frame_pair_diagnostics(frame0, frame1)
        self._flush_pipeline_trace_pending(commit_ms)

    def _flush_pipeline_trace_pending(self, commit_ms: float) -> None:
        """Record the backend result paired with the most recent frontend call."""
        pending = self._pipeline_pending
        if pending is not None and self._pipeline_trace_writer is not None:
            now = time.perf_counter()
            opt_diag = getattr(self.Optimizer, "last_pair_diagnostics", {}) or {}
            solver_s = opt_diag.get("local_ba_optimize_total_s")
            self._pipeline_trace_writer.writerow({
                "frame_i": pending["frame_i"],
                "frame_j": pending["frame_j"],
                "timestamp_i_ns": pending["timestamp_i_ns"],
                "timestamp_j_ns": pending["timestamp_j_ns"],
                "frontend_ms": pending["frontend_ms"],
                "backend_solver_ms": "" if solver_s is None else float(solver_s) * 1000.0,
                "backend_wait_ms": (now - pending["submitted_at"]) * 1000.0,
                "commit_ms": commit_ms,
                "backend_submitted": 1,
                "static_initialization_active": 0,
            })
            self._pipeline_trace_stream.flush()
        self._pipeline_pending = None

    def _complete_tracking_step(
        self,
        *,
        frame1: T_SensorFrame,
        frame_idx: torch.Tensor,
        match_idx: torch.Tensor,
        match_obs: MatchObs,
        candidate_count: int,
        prev_pose: pp.LieTensor,
        est_pose: pp.LieTensor,
        imu_rel_pose: pp.LieTensor | None,
        suppress_optimizer: bool = False,
    ) -> bool:
        match_count = int(match_idx.size(0))
        stored_match_fields = {
            name: values[match_idx]
            for name, values in self.graph.match.data.items()
        }
        self._pending_tracking_state = {
            "optimizer_skipped": bool(match_count < self.min_num_point or suppress_optimizer),
            "static_zupt_active": bool(suppress_optimizer),
            "match_idx_size_after_filter": match_count,
            "num_keypoints_candidate": int(candidate_count),
            "min_num_point": int(self.min_num_point),
            "visual_input_sha256": visual_input_sha256(stored_match_fields),
        }
        if match_count >= self.min_num_point:
            graph_data = self.Optimizer.get_graph_data(self.graph, frame_idx)
            # Raw MACVO deliberately keeps its own visual history, including
            # the first three seconds.  Only T2 is held by static IMU
            # initialization, so the dashboard can expose the startup drift.
            self._update_live_macvo_raw_pose(graph_data)
            if suppress_optimizer:
                return False
            self.Optimizer.start_optimize(graph_data)
            return False

        if suppress_optimizer:
            return False

        fallback_applied = False
        fallback_reason = "disabled"
        decision = self._adaptive_decision
        if self._adaptive_enabled and decision is not None and imu_rel_pose is not None:
            use_fallback_rotation = bool(getattr(decision, "use_imu_rotation", False))
            use_fallback_translation = bool(getattr(decision, "use_imu_translation", False))
            if use_fallback_rotation or use_fallback_translation:
                fallback_pose = compose_adaptive_fallback_pose(
                    prev_pose=prev_pose,
                    visual_pose=est_pose,
                    imu_rel_pose=imu_rel_pose,
                    use_imu_rotation=use_fallback_rotation,
                    use_imu_translation=use_fallback_translation,
                    sensor_T_imu=self._frame_imu_vio_sensor_T_imu(frame1),
                )
                self.graph.frames.data["pose"][frame_idx] = fallback_pose.float()
                fallback_applied = True
                fallback_reason = (
                    "imu_rotation_translation" if use_fallback_translation else "imu_rotation_only"
                )

        self._pending_tracking_state["lost_track_fallback_applied"] = fallback_applied
        self._pending_tracking_state["lost_track_fallback_reason"] = fallback_reason
        Logger.write(
            "warn",
            f"VOLostTrack @ {frame1.frame_idx} - only get {match_count} observations; "
            f"fallback={fallback_reason}",
        )
        self.graph.frames.data["need_interp"][frame_idx] = True
        return True

    def _record_visual_factor_diagnostics(
        self,
        frame_i: int,
        frame_j: int,
    ) -> None:
        writer = self._visual_factor_diag_writer
        covariance = self._pending_cov_diag
        pair = (int(frame_i), int(frame_j))
        if writer is None or covariance is None or pair in self._visual_factor_diag_written_pairs:
            return
        writer.writerow({
            "frame_i": pair[0],
            "frame_j": pair[1],
            "timestamp_i": int(self.graph.frames.data["time_ns"][pair[0]].item()),
            "timestamp_j": int(self.graph.frames.data["time_ns"][pair[1]].item()),
            "visual_input_sha256": str(
                (self._pending_tracking_state or {}).get("visual_input_sha256", "")
            ),
            **{
                name: covariance.get(name, float("nan"))
                for name in FRONTEND_COVARIANCE_DIAGNOSTIC_FIELDS
            },
        })
        self._visual_factor_diag_stream.flush()
        self._visual_factor_diag_written_pairs.add(pair)

    def run_pair(self, frame0: T_SensorFrame, frame1: T_SensorFrame) -> None:
        if self._visual_cache_reader is not None:
            self._run_pair_visual_cache(frame0, frame1)
            return

        assert self.prev_keyframe is not None

        # Check if current frame is the keyframe ########################################
        if not self.KeyframeSelector.isKeyframe(frame1):
            self.push_keyframe(frame1, self.graph.frames.data["pose"][self.prev_keyframe[1]].unsqueeze(0), need_interp=True)
            return

        depth0          = self.prev_keyframe[2]
        frontend_start = time.perf_counter()
        depth1, match01 = self.Frontend.estimate_pair(frame0.stereo, frame1.stereo)
        frontend_ms = (time.perf_counter() - frontend_start) * 1000.0

        # Receive optimization result from previous step (if exists) ####################
        # NOTE: should always writeback optimized pose to global map before selecting new
        # keypoints (register new 3D point) on that frame.
        self._commit_previous_backend_result(frame0, frame1)

        # Motion model provide an initial guess to the pose of frame1 ###################
        # Update motion model (this must be after write_back to get latest result)
        # NOTE: I assume the motion estimator works on stereo camera frame (not body frame)
        self.MotionEstimator.update(pp.SE3(self.graph.frames.data["pose"][self.prev_keyframe[1]]))
        est_pose = self.MotionEstimator.predict(frame1, match01.flow, depth1.depth).unsqueeze(0)
        # Save before static anchoring and before the optional IMU pre-fusion.
        imu_rotvec_prior, imu_rot_prior_std, imu_trans_prior, imu_trans_cov, imu_rel_pose = self._estimate_imu_priors(frame1)
        if self._imu_static_zupt_active and self._imu_static_anchor_pose is not None:
            est_pose = pp.SE3(
                self._imu_static_anchor_pose.tensor().to(est_pose.tensor())
            ).unsqueeze(0)
        use_post_imu_fusion = bool(getattr(self.Optimizer.config, "post_imu_fusion_enable", False))
        use_pre_imu_fusion = (not use_post_imu_fusion) or bool(
            getattr(self.Optimizer.config, "post_imu_fusion_prepose_enable", True)
        )
        if use_pre_imu_fusion:
            est_pose = self._fuse_visual_imu_pose(est_pose, imu_rel_pose, self._frame_imu_vio_sensor_T_imu(frame1))

        # Generate Keypoints for frame 0 and 1 ##########################################
        kp0_uv  = self.KeypointSelector.select_point(frame0.stereo, self.num_point, depth0, depth1, match01)
        kp1_uv  = kp0_uv + self.Frontend.retrieve_pixels(kp0_uv, match01.flow).T

        inbound_mask= filterPointsInRange(
            kp1_uv,
            (self.edge_width, frame1.stereo.width - self.edge_width),
            (self.edge_width, frame1.stereo.height - self.edge_width)
        )
        kp0_uv  = kp0_uv[inbound_mask]
        kp1_uv  = kp1_uv[inbound_mask]

        # Retrieve depth and depth cov for kp on frame 0 and 1 ##########################
        kp0_d               = self.Frontend.retrieve_pixels(kp0_uv, depth0.depth).squeeze(0)
        kp0_disparity       = self.Frontend.retrieve_pixels(kp0_uv, depth0.disparity)
        kp0_sigma_disparity = self.Frontend.retrieve_pixels(kp0_uv, depth0.disparity_uncertainty)
        kp0_sigma_dd        = self.Frontend.retrieve_pixels(kp0_uv, depth0.cov)
        kp0_sigma_dd        = kp0_sigma_dd.squeeze(0) if kp0_sigma_dd is not None else None

        kp1_d               = self.Frontend.retrieve_pixels(kp1_uv, depth1.depth).squeeze(0)
        kp1_disparity       = self.Frontend.retrieve_pixels(kp1_uv, depth1.disparity)
        kp1_sigma_disparity = self.Frontend.retrieve_pixels(kp1_uv, depth1.disparity_uncertainty)
        kp1_sigma_dd        = self.Frontend.retrieve_pixels(kp1_uv, depth1.cov)
        kp1_sigma_dd        = kp1_sigma_dd.squeeze(0) if kp1_sigma_dd is not None else None


        # Retrieve match cov for kp on frame 0 and 1    #################################
        num_kp = kp0_uv.size(0)

        # kp 0 has a fake sigma uv as it is manually selected pixels. This UV
        # represents the uncertainty introduced by the quantization process when
        # taking photo with discrete pixels.
        kp0_sigma_uv = torch.ones((num_kp, 3), device=self.device) * self.match_cov_default
        kp0_sigma_uv[..., 2] = 0.   # No sigma_uv off-diag term.

        kp1_sigma_uv = self.Frontend.retrieve_pixels(kp0_uv, match01.cov)
        kp1_sigma_uv = kp1_sigma_uv.T if kp1_sigma_uv is not None else None

        # ── Extract frontend covariance statistics for diagnostics ──────
        self._pending_cov_diag = _compute_frontend_cov_stats(
            depth0.cov, depth1.cov, match01.cov,
            kp0_sigma_dd, kp1_sigma_dd, kp0_sigma_uv, kp1_sigma_uv, num_kp,
            depth0_depth=depth0.depth, depth1_depth=depth1.depth)

        # Record color of keypoints (for visualization) #################################
        kp0_uv_cpu = kp0_uv.cpu()
        kp0_color  = frame0.stereo.imageL[..., kp0_uv_cpu[..., 1], kp0_uv_cpu[..., 0]].squeeze(0).T
        kp0_color  = (kp0_color * 255).to(torch.uint8)

        # Project from 2D -> 3D #########################################################
        pos0_Tc = pixel2point_NED(kp0_uv, kp0_d, frame0.stereo.frame_K).cpu()
        pos0_covTc  = self.ObsCovModel.estimate(frame0.stereo, kp0_uv, depth0, kp0_sigma_dd, kp0_sigma_uv)
        pos1_covTc  = self.ObsCovModel.estimate(frame1.stereo, kp1_uv, depth1, kp1_sigma_dd, kp1_sigma_uv)


        # Run Outlier Filter ############################################################
        match_obs = MatchObs.init({
            "pixel1_uv"      : kp0_uv_cpu,
            "pixel2_uv"      : kp1_uv.cpu(),

            "pixel1_d"       : kp0_d.unsqueeze(-1).cpu(),
            "pixel2_d"       : kp1_d.unsqueeze(-1).cpu(),

            "pixel1_disp"    : torch.empty((num_kp, 1)).fill_(-1) if kp0_disparity is None else kp0_disparity.T.cpu(),
            "pixel2_disp"    : torch.empty((num_kp, 1)).fill_(-1) if kp1_disparity is None else kp1_disparity.T.cpu(),

            "pixel1_disp_cov": torch.empty((num_kp, 1)).fill_(-1) if kp0_sigma_disparity is None else kp0_sigma_disparity.T.cpu(),
            "pixel2_disp_cov": torch.empty((num_kp, 1)).fill_(-1) if kp1_sigma_disparity is None else kp1_sigma_disparity.T.cpu(),

            "pixel1_d_cov"   : torch.empty((num_kp, 1)).fill_(-1) if kp0_sigma_dd is None else kp0_sigma_dd.unsqueeze(-1).cpu(),
            "pixel2_d_cov"   : torch.empty((num_kp, 1)).fill_(-1) if kp1_sigma_dd is None else kp1_sigma_dd.unsqueeze(-1).cpu(),

            "pixel1_uv_cov"  : torch.empty((num_kp, 3)).fill_(-1) if kp0_sigma_uv is None else kp0_sigma_uv,
            "pixel2_uv_cov"  : torch.empty((num_kp, 3)).fill_(-1) if kp1_sigma_uv is None else kp1_sigma_uv,

            "obs1_covTc"     : pos0_covTc,
            "obs2_covTc"     : pos1_covTc,
        })
        assert self.OutlierFilter.verify_shape(match_obs), "The provided MatchFactor does not contain all data for outlier filter."
        mask = self.OutlierFilter.filter(match_obs, torch.device("cpu"))
        match_obs = match_obs[mask]

        # Register the factor graph #####################################################
        prev_pose       = pp.SE3(self.graph.frames.data["pose"][self.prev_keyframe[1]])
        prev_rot        = prev_pose.rotation().matrix().repeat((num_kp, 1, 1)).to(torch.float64)
        num_match_orig  = len(self.graph.match)

        point_idx = self.graph.points.push(PointNode.init({
            "pos_Tc": pos0_Tc,
            "pos_Tw": pp.SE3_type.Act(prev_pose, pos0_Tc)[..., :3],  # NOTE: Refer to https://github.com/pypose/pypose/issues/342
            "cov_Tw": torch.bmm(torch.bmm(prev_rot, pos0_covTc), prev_rot.transpose(1, 2)),
            "color" : kp0_color
        })[mask])
        imu_vio_factor = self._pending_imu_vio_factor or {}
        frame_idx      = self.push_keyframe(
            frame1, est_pose,
            imu_rotvec_prior=imu_rotvec_prior,
            imu_rot_prior_std=imu_rot_prior_std,
            imu_trans_prior=imu_trans_prior,
            imu_trans_cov=imu_trans_cov,
            imu_vio_prev_velocity_world=imu_vio_factor.get("prev_velocity_world"),
            imu_vio_curr_velocity_init_world=imu_vio_factor.get("curr_velocity_init_world"),
            imu_vio_prev_acc_bias=imu_vio_factor.get("prev_acc_bias"),
            imu_vio_prev_gyro_bias=imu_vio_factor.get("prev_gyro_bias"),
            imu_vio_acc_bias=imu_vio_factor.get("curr_acc_bias_init"),
            imu_vio_gyro_bias=imu_vio_factor.get("curr_gyro_bias_init"),
            imu_vio_linearized_acc_bias=imu_vio_factor.get("linearized_acc_bias"),
            imu_vio_linearized_gyro_bias=imu_vio_factor.get("linearized_gyro_bias"),
            imu_vio_bias_jacobian=imu_vio_factor.get("bias_jacobian"),
            imu_vio_bias_rw_cov=imu_vio_factor.get("bias_rw_cov"),
            imu_vio_delta_rotvec=imu_vio_factor.get("delta_rotvec"),
            imu_vio_delta_v=imu_vio_factor.get("delta_v"),
            imu_vio_delta_p=imu_vio_factor.get("delta_p"),
            imu_vio_cov=imu_vio_factor.get("cov"),
            imu_vio_sa_v2_unique_cov=imu_vio_factor.get("sa_v2_unique_cov"),
            imu_vio_sa_v2_incoming_raw_time_ns=imu_vio_factor.get("sa_v2_incoming_raw_time_ns"),
            imu_vio_sa_v2_outgoing_raw_time_ns=imu_vio_factor.get("sa_v2_outgoing_raw_time_ns"),
            imu_vio_sa_v2_incoming_count=imu_vio_factor.get("sa_v2_incoming_count"),
            imu_vio_sa_v2_outgoing_count=imu_vio_factor.get("sa_v2_outgoing_count"),
            imu_vio_sa_v2_incoming_sensitivity=imu_vio_factor.get("sa_v2_incoming_sensitivity"),
            imu_vio_sa_v2_outgoing_sensitivity=imu_vio_factor.get("sa_v2_outgoing_sensitivity"),
            imu_vio_dt=imu_vio_factor.get("dt"),
            imu_vio_gravity_world=imu_vio_factor.get("gravity_world"),
            imu_vio_gravity_in_residual=imu_vio_factor.get("gravity_in_residual"),
        )
        prev_frame_idx = torch.tensor([self.prev_keyframe[1]], dtype=torch.long)
        match_idx      = self.graph.match.push(match_obs)

        num_match_kp = len(match_obs)
        self.graph.point2match.add(point_idx, match_idx)    # Associate point -> match
        self.graph.match2point.set(match_idx, point_idx)    # Associate match -> point
        self.graph.frame2match.add(prev_frame_idx, torch.tensor([num_match_orig], dtype=torch.long), torch.tensor([num_match_kp], dtype=torch.long))   # Associate frame -> match
        self.graph.frame2match.add(frame_idx     , torch.tensor([num_match_orig], dtype=torch.long), torch.tensor([num_match_kp], dtype=torch.long))   # Associate frame -> match
        self.graph.match2frame1.set(match_idx    , torch.empty((num_match_kp,), dtype=torch.long).fill_(prev_frame_idx.item()))    # Associate match -> frame1
        self.graph.match2frame2.set(match_idx    , torch.empty((num_match_kp,), dtype=torch.long).fill_(frame_idx.item()     ))    # Associate match -> frame2

        # Visualization #################################################################
        fig_plt.plot_imatcher("matching", match01, frame0, frame1)
        fig_plt.plot_istereo ("stereo_d", depth1 , frame1)
        fig_plt.plot_macvo   ("macvo_kp", match_obs, depth1, match01, frame0, frame1)

        # Update the tracking context ###################################################
        self.prev_keyframe = (frame1, int(frame_idx.item()), depth1)

        # Launch Optimization task  #####################################################
        lost_track = self._complete_tracking_step(
            frame1=frame1,
            frame_idx=frame_idx,
            match_idx=match_idx,
            match_obs=match_obs,
            candidate_count=int(kp0_uv.size(0)),
            prev_pose=prev_pose,
            est_pose=est_pose,
            imu_rel_pose=imu_rel_pose,
            suppress_optimizer=self._imu_static_zupt_active,
        )
        if (
            not lost_track
            and self._pipeline_trace_writer is not None
            and self._pipeline_pending is None
            and not self._imu_static_zupt_active
        ):
            self._pipeline_pending = {
                "frame_i": int(frame0.frame_idx),
                "frame_j": int(frame1.frame_idx),
                "timestamp_i_ns": int(frame0.stereo.frame_ns),
                "timestamp_j_ns": int(frame1.stereo.frame_ns),
                "frontend_ms": frontend_ms,
                "submitted_at": time.perf_counter(),
            }
        self._record_visual_factor_diagnostics(prev_frame_idx.item(), frame_idx.item())
        if lost_track:
            # NOTE: if lost track, we do not do mapping since the pose is not reliable anyway.
            return

        # Add (dense) mapping points to the map #########################################
        if self.mapping:
            map0_uv       = self.MappointSelector.select_point(frame0.stereo, 2000, depth0, depth1, match01)
            num_kp        = map0_uv.size(0)
            map0_d        = self.Frontend.retrieve_pixels(map0_uv, depth0.depth).squeeze(0)
            map0_Tc       = pixel2point_NED(map0_uv, map0_d, frame0.stereo.frame_K).cpu()

            map0_sigma_dd = self.Frontend.retrieve_pixels(map0_uv, depth0.cov)
            map0_sigma_dd = map0_sigma_dd.squeeze(0) if (map0_sigma_dd is not None) else None
            map0_sigma_uv = torch.ones((num_kp, 3), device=self.device) * self.match_cov_default
            map0_sigma_uv[..., 2] = 0.   # No sigma_uv off-diag term.
            map0_Tc_cov = self.ObsCovModel.estimate(frame0.stereo, map0_uv, depth0, map0_sigma_dd, map0_sigma_uv)

            map0_uv_cpu = map0_uv.cpu()
            map0_color  = frame0.stereo.imageL[..., map0_uv_cpu[..., 1], map0_uv_cpu[..., 0]].squeeze(0).T
            map0_color  = (map0_color * 255).to(torch.uint8)

            num_map_orig  = len(self.graph.map_points)
            num_mappoint  = map0_Tc.size(0)
            map_idx = self.graph.map_points.push(PointNode.init({
                "pos_Tc": map0_Tc,
                "pos_Tw": pp.SE3_type.Act(prev_pose, map0_Tc)[..., :3],
                "cov_Tw": map0_Tc_cov,
                "color" : map0_color,
            }))
            self.graph.frame2map.add(frame_idx, torch.tensor([num_map_orig], dtype=torch.long), torch.tensor([num_mappoint], dtype=torch.long))   # Associate frame -> map

    def _validate_visual_cache_packet_calibration(
        self,
        packet,
        frame0: T_SensorFrame,
        frame1: T_SensorFrame,
    ) -> None:
        packet_K = packet.K.detach().to(device="cpu", dtype=torch.float64)
        for frame in (frame0, frame1):
            frame_K = torch.as_tensor(frame.stereo.frame_K).detach().to(device="cpu", dtype=torch.float64)
            if not torch.equal(frame_K, packet_K):
                raise VisualFactorCacheError("K differs from cached visual packet")
            if float(frame.stereo.frame_baseline) != float(packet.baseline_m):
                raise VisualFactorCacheError("baseline differs from cached visual packet")

    def _run_pair_visual_cache(self, frame0: T_SensorFrame, frame1: T_SensorFrame) -> None:
        assert self.prev_keyframe is not None
        reader = self._visual_cache_reader
        if reader is None:
            raise RuntimeError("visual cache reader is unavailable in replay mode")

        # Commit the previous backend result before using it as this packet's world frame.
        self._commit_previous_backend_result(frame0, frame1)

        packet = reader.load_pair(
            int(frame0.frame_idx),
            int(frame1.frame_idx),
            int(frame0.stereo.frame_ns),
            int(frame1.stereo.frame_ns),
        )
        self._validate_visual_cache_packet_calibration(packet, frame0, frame1)
        self._pending_cov_diag = dict(packet.covariance_diagnostics)

        prev_frame_idx = torch.tensor([self.prev_keyframe[1]], dtype=torch.long)
        prev_pose = pp.SE3(self.graph.frames.data["pose"][self.prev_keyframe[1]])
        visual_relative_pose_CiCj = None
        visual_relative_pose_cov = None
        visual_relative_pose_num_points = 0
        visual_relative_pose_num_inliers = 0
        visual_relative_pose_mean_mahalanobis_sq = None
        visual_compressed_uvd_reference_CjCi = None
        visual_compressed_uvd_hessian = None
        visual_compressed_uvd_gradient = None
        visual_compressed_uvd_robust_cost = None
        visual_compressed_uvd_num_points = 0
        visual_compressed_uvd_num_inliers = 0
        visual_compressed_uvd_mean_mahalanobis_sq = None
        visual_compressed_uvd_huber_delta = None
        if self._relative_pose_factor_reader is not None:
            pose_factor = self._relative_pose_factor_reader.load_pair(
                int(frame0.frame_idx),
                int(frame1.frame_idx),
                packet.visual_sha256,
            )
            relative_pose_init = pp.SE3(pose_factor.measurement_CiCj).to(
                device=prev_pose.tensor().device,
                dtype=prev_pose.tensor().dtype,
            )
            visual_relative_pose_CiCj = pose_factor.measurement_CiCj.float()
            visual_relative_pose_cov = pose_factor.covariance.unsqueeze(0).float()
            visual_relative_pose_num_points = int(pose_factor.num_points)
            visual_relative_pose_num_inliers = int(pose_factor.num_inliers)
            visual_relative_pose_mean_mahalanobis_sq = float(pose_factor.mean_mahalanobis_sq)
        elif self._compressed_uvd_factor_reader is not None:
            compressed = self._compressed_uvd_factor_reader.load_pair(
                int(frame0.frame_idx),
                int(frame1.frame_idx),
                packet.visual_sha256,
            )
            reference_CjCi = pp.SE3(compressed.reference_CjCi).to(
                device=prev_pose.tensor().device,
                dtype=prev_pose.tensor().dtype,
            )
            relative_pose_init = reference_CjCi.Inv()
            visual_compressed_uvd_reference_CjCi = compressed.reference_CjCi
            visual_compressed_uvd_hessian = compressed.hessian
            visual_compressed_uvd_gradient = compressed.gradient
            visual_compressed_uvd_robust_cost = float(compressed.robust_cost)
            visual_compressed_uvd_num_points = int(compressed.num_points)
            visual_compressed_uvd_num_inliers = int(compressed.num_inliers)
            visual_compressed_uvd_mean_mahalanobis_sq = float(
                compressed.mean_mahalanobis_sq
            )
            visual_compressed_uvd_huber_delta = float(compressed.huber_delta)
        else:
            relative_pose_init = pp.from_matrix(packet.relative_pose_init, pp.SE3_type).to(
                device=prev_pose.tensor().device,
                dtype=prev_pose.tensor().dtype,
            )
        est_pose = pp.SE3(
            (prev_pose @ relative_pose_init).tensor().reshape(1, 7)
        )
        imu_rotvec_prior, imu_rot_prior_std, imu_trans_prior, imu_trans_cov, imu_rel_pose = self._estimate_imu_priors(frame1)
        if self._imu_static_zupt_active and self._imu_static_anchor_pose is not None:
            est_pose = pp.SE3(
                self._imu_static_anchor_pose.tensor().to(est_pose.tensor())
            ).unsqueeze(0)
        use_post_imu_fusion = bool(getattr(self.Optimizer.config, "post_imu_fusion_enable", False))
        use_pre_imu_fusion = (not use_post_imu_fusion) or bool(
            getattr(self.Optimizer.config, "post_imu_fusion_prepose_enable", True)
        )
        if use_pre_imu_fusion:
            est_pose = self._fuse_visual_imu_pose(est_pose, imu_rel_pose, self._frame_imu_vio_sensor_T_imu(frame1))

        match_obs = MatchObs.init(packet.match_fields)
        num_match_orig = len(self.graph.match)
        num_match_kp = len(match_obs)
        point_pose = pp.SE3(prev_pose.tensor().to(dtype=packet.points_local.dtype))
        points_local = packet.points_local.to(
            device=point_pose.tensor().device,
            dtype=packet.points_local.dtype,
        )
        points_cov_local = packet.points_cov_local.to(
            device=point_pose.tensor().device,
            dtype=packet.points_cov_local.dtype,
        )
        prev_rot = prev_pose.rotation().matrix().to(
            device=points_cov_local.device,
            dtype=points_cov_local.dtype,
        ).repeat((num_match_kp, 1, 1))
        point_idx = self.graph.points.push(PointNode.init({
            "pos_Tc": points_local,
            "pos_Tw": pp.SE3_type.Act(point_pose, points_local)[..., :3],
            "cov_Tw": torch.bmm(torch.bmm(prev_rot, points_cov_local), prev_rot.transpose(1, 2)),
            "color": packet.point_colors,
        }))

        imu_vio_factor = self._pending_imu_vio_factor or {}
        frame_idx = self.push_keyframe(
            frame1, est_pose,
            imu_rotvec_prior=imu_rotvec_prior,
            imu_rot_prior_std=imu_rot_prior_std,
            imu_trans_prior=imu_trans_prior,
            imu_trans_cov=imu_trans_cov,
            imu_vio_prev_velocity_world=imu_vio_factor.get("prev_velocity_world"),
            imu_vio_curr_velocity_init_world=imu_vio_factor.get("curr_velocity_init_world"),
            imu_vio_prev_acc_bias=imu_vio_factor.get("prev_acc_bias"),
            imu_vio_prev_gyro_bias=imu_vio_factor.get("prev_gyro_bias"),
            imu_vio_acc_bias=imu_vio_factor.get("curr_acc_bias_init"),
            imu_vio_gyro_bias=imu_vio_factor.get("curr_gyro_bias_init"),
            imu_vio_linearized_acc_bias=imu_vio_factor.get("linearized_acc_bias"),
            imu_vio_linearized_gyro_bias=imu_vio_factor.get("linearized_gyro_bias"),
            imu_vio_bias_jacobian=imu_vio_factor.get("bias_jacobian"),
            imu_vio_bias_rw_cov=imu_vio_factor.get("bias_rw_cov"),
            imu_vio_delta_rotvec=imu_vio_factor.get("delta_rotvec"),
            imu_vio_delta_v=imu_vio_factor.get("delta_v"),
            imu_vio_delta_p=imu_vio_factor.get("delta_p"),
            imu_vio_cov=imu_vio_factor.get("cov"),
            imu_vio_sa_v2_unique_cov=imu_vio_factor.get("sa_v2_unique_cov"),
            imu_vio_sa_v2_incoming_raw_time_ns=imu_vio_factor.get("sa_v2_incoming_raw_time_ns"),
            imu_vio_sa_v2_outgoing_raw_time_ns=imu_vio_factor.get("sa_v2_outgoing_raw_time_ns"),
            imu_vio_sa_v2_incoming_count=imu_vio_factor.get("sa_v2_incoming_count"),
            imu_vio_sa_v2_outgoing_count=imu_vio_factor.get("sa_v2_outgoing_count"),
            imu_vio_sa_v2_incoming_sensitivity=imu_vio_factor.get("sa_v2_incoming_sensitivity"),
            imu_vio_sa_v2_outgoing_sensitivity=imu_vio_factor.get("sa_v2_outgoing_sensitivity"),
            imu_vio_dt=imu_vio_factor.get("dt"),
            imu_vio_gravity_world=imu_vio_factor.get("gravity_world"),
            imu_vio_gravity_in_residual=imu_vio_factor.get("gravity_in_residual"),
            visual_relative_pose_CiCj=visual_relative_pose_CiCj,
            visual_relative_pose_cov=visual_relative_pose_cov,
            visual_relative_pose_num_points=visual_relative_pose_num_points,
            visual_relative_pose_num_inliers=visual_relative_pose_num_inliers,
            visual_relative_pose_mean_mahalanobis_sq=visual_relative_pose_mean_mahalanobis_sq,
            visual_compressed_uvd_reference_CjCi=visual_compressed_uvd_reference_CjCi,
            visual_compressed_uvd_hessian=visual_compressed_uvd_hessian,
            visual_compressed_uvd_gradient=visual_compressed_uvd_gradient,
            visual_compressed_uvd_robust_cost=visual_compressed_uvd_robust_cost,
            visual_compressed_uvd_num_points=visual_compressed_uvd_num_points,
            visual_compressed_uvd_num_inliers=visual_compressed_uvd_num_inliers,
            visual_compressed_uvd_mean_mahalanobis_sq=visual_compressed_uvd_mean_mahalanobis_sq,
            visual_compressed_uvd_huber_delta=visual_compressed_uvd_huber_delta,
        )
        match_idx = self.graph.match.push(match_obs)

        self.graph.point2match.add(point_idx, match_idx)
        self.graph.match2point.set(match_idx, point_idx)
        self.graph.frame2match.add(prev_frame_idx, torch.tensor([num_match_orig], dtype=torch.long), torch.tensor([num_match_kp], dtype=torch.long))
        self.graph.frame2match.add(frame_idx, torch.tensor([num_match_orig], dtype=torch.long), torch.tensor([num_match_kp], dtype=torch.long))
        self.graph.match2frame1.set(match_idx, torch.empty((num_match_kp,), dtype=torch.long).fill_(prev_frame_idx.item()))
        self.graph.match2frame2.set(match_idx, torch.empty((num_match_kp,), dtype=torch.long).fill_(frame_idx.item()))

        self.prev_keyframe = (frame1, int(frame_idx.item()), None)
        cached_candidate_count = packet.covariance_diagnostics.get(
            "num_selected_keypoints",
            num_match_kp,
        )
        try:
            candidate_count = int(cached_candidate_count)
        except (TypeError, ValueError, OverflowError):
            candidate_count = num_match_kp
        self._complete_tracking_step(
            frame1=frame1,
            frame_idx=frame_idx,
            match_idx=match_idx,
            match_obs=match_obs,
            candidate_count=candidate_count,
            prev_pose=prev_pose,
            est_pose=est_pose,
            imu_rel_pose=imu_rel_pose,
            suppress_optimizer=self._imu_static_zupt_active,
        )
        self._visual_cache_consumed_pairs.append((packet.frame_i, packet.frame_j))

    def push_keyframe(
        self,
        frame: T_SensorFrame,
        est_pose: pp.LieTensor | torch.Tensor,
        need_interp: bool=False,
        imu_rotvec_prior: torch.Tensor | None = None,
        imu_rot_prior_std: torch.Tensor | None = None,
        imu_trans_prior: torch.Tensor | None = None,
        imu_trans_cov: torch.Tensor | None = None,
        imu_vio_prev_velocity_world: torch.Tensor | None = None,
        imu_vio_curr_velocity_init_world: torch.Tensor | None = None,
        imu_vio_prev_acc_bias: torch.Tensor | None = None,
        imu_vio_prev_gyro_bias: torch.Tensor | None = None,
        imu_vio_acc_bias: torch.Tensor | None = None,
        imu_vio_gyro_bias: torch.Tensor | None = None,
        imu_vio_linearized_acc_bias: torch.Tensor | None = None,
        imu_vio_linearized_gyro_bias: torch.Tensor | None = None,
        imu_vio_bias_jacobian: torch.Tensor | None = None,
        imu_vio_bias_rw_cov: torch.Tensor | None = None,
        imu_vio_delta_rotvec: torch.Tensor | None = None,
        imu_vio_delta_v: torch.Tensor | None = None,
        imu_vio_delta_p: torch.Tensor | None = None,
        imu_vio_cov: torch.Tensor | None = None,
        imu_vio_sa_v2_unique_cov: torch.Tensor | None = None,
        imu_vio_sa_v2_incoming_raw_time_ns: torch.Tensor | None = None,
        imu_vio_sa_v2_outgoing_raw_time_ns: torch.Tensor | None = None,
        imu_vio_sa_v2_incoming_count: torch.Tensor | None = None,
        imu_vio_sa_v2_outgoing_count: torch.Tensor | None = None,
        imu_vio_sa_v2_incoming_sensitivity: torch.Tensor | None = None,
        imu_vio_sa_v2_outgoing_sensitivity: torch.Tensor | None = None,
        imu_vio_dt: torch.Tensor | None = None,
        imu_vio_gravity_world: torch.Tensor | None = None,
        imu_vio_gravity_in_residual: torch.Tensor | None = None,
        visual_relative_pose_CiCj: torch.Tensor | None = None,
        visual_relative_pose_cov: torch.Tensor | None = None,
        visual_relative_pose_num_points: int = 0,
        visual_relative_pose_num_inliers: int = 0,
        visual_relative_pose_mean_mahalanobis_sq: float | None = None,
        visual_compressed_uvd_reference_CjCi: torch.Tensor | None = None,
        visual_compressed_uvd_hessian: torch.Tensor | None = None,
        visual_compressed_uvd_gradient: torch.Tensor | None = None,
        visual_compressed_uvd_robust_cost: float | None = None,
        visual_compressed_uvd_num_points: int = 0,
        visual_compressed_uvd_num_inliers: int = 0,
        visual_compressed_uvd_mean_mahalanobis_sq: float | None = None,
        visual_compressed_uvd_huber_delta: float | None = None,
    ) -> torch.Tensor:
        if imu_rotvec_prior is None:
            imu_rotvec_prior = torch.zeros((1, 3), dtype=torch.float32)
        if imu_rot_prior_std is None:
            imu_rot_prior_std = torch.tensor([1e6], dtype=torch.float32)
        if imu_trans_prior is None:
            imu_trans_prior = torch.zeros((1, 3), dtype=torch.float32)
        if imu_trans_cov is None:
            imu_trans_cov = torch.eye(3, dtype=torch.float32).unsqueeze(0) * 1e6
        if imu_vio_prev_velocity_world is None:
            imu_vio_prev_velocity_world = torch.zeros((1, 3), dtype=torch.float32)
        if imu_vio_curr_velocity_init_world is None:
            imu_vio_curr_velocity_init_world = torch.zeros((1, 3), dtype=torch.float32)
        current_acc_bias = (
            getattr(self, "_imu_acc_bias", torch.zeros(3, dtype=torch.float32))
            .detach()
            .reshape(1, 3)
            .cpu()
            .float()
            .clone()
        )
        current_gyro_bias = (
            getattr(self, "_imu_gyro_bias", torch.zeros(3, dtype=torch.float32))
            .detach()
            .reshape(1, 3)
            .cpu()
            .float()
            .clone()
        )
        if imu_vio_prev_acc_bias is None:
            imu_vio_prev_acc_bias = current_acc_bias.clone()
        if imu_vio_prev_gyro_bias is None:
            imu_vio_prev_gyro_bias = current_gyro_bias.clone()
        if imu_vio_acc_bias is None:
            imu_vio_acc_bias = current_acc_bias.clone()
        if imu_vio_gyro_bias is None:
            imu_vio_gyro_bias = current_gyro_bias.clone()
        if imu_vio_linearized_acc_bias is None:
            imu_vio_linearized_acc_bias = current_acc_bias.clone()
        if imu_vio_linearized_gyro_bias is None:
            imu_vio_linearized_gyro_bias = current_gyro_bias.clone()
        if imu_vio_bias_jacobian is None:
            imu_vio_bias_jacobian = torch.zeros((1, 9, 6), dtype=torch.float32)
        if imu_vio_bias_rw_cov is None:
            imu_vio_bias_rw_cov = torch.eye(6, dtype=torch.float32).unsqueeze(0) * 1e6
        if imu_vio_delta_rotvec is None:
            imu_vio_delta_rotvec = torch.zeros((1, 3), dtype=torch.float32)
        if imu_vio_delta_v is None:
            imu_vio_delta_v = torch.zeros((1, 3), dtype=torch.float32)
        if imu_vio_delta_p is None:
            imu_vio_delta_p = torch.zeros((1, 3), dtype=torch.float32)
        if imu_vio_cov is None:
            imu_vio_cov = torch.eye(9, dtype=torch.float32).unsqueeze(0) * 1e6
        if imu_vio_sa_v2_unique_cov is None:
            imu_vio_sa_v2_unique_cov = torch.eye(9, dtype=torch.float32).unsqueeze(0) * 1e6
        if imu_vio_sa_v2_incoming_raw_time_ns is None:
            imu_vio_sa_v2_incoming_raw_time_ns = torch.full((1, 2), -1, dtype=torch.long)
        if imu_vio_sa_v2_outgoing_raw_time_ns is None:
            imu_vio_sa_v2_outgoing_raw_time_ns = torch.full((1, 2), -1, dtype=torch.long)
        if imu_vio_sa_v2_incoming_count is None:
            imu_vio_sa_v2_incoming_count = torch.zeros(1, dtype=torch.long)
        if imu_vio_sa_v2_outgoing_count is None:
            imu_vio_sa_v2_outgoing_count = torch.zeros(1, dtype=torch.long)
        if imu_vio_sa_v2_incoming_sensitivity is None:
            imu_vio_sa_v2_incoming_sensitivity = torch.zeros((1, 9, 12), dtype=torch.float32)
        if imu_vio_sa_v2_outgoing_sensitivity is None:
            imu_vio_sa_v2_outgoing_sensitivity = torch.zeros((1, 9, 12), dtype=torch.float32)
        if imu_vio_dt is None:
            imu_vio_dt = torch.tensor([0.0], dtype=torch.float32)
        if imu_vio_gravity_world is None:
            imu_vio_gravity_world = torch.zeros((1, 3), dtype=torch.float32)
        if imu_vio_gravity_in_residual is None:
            imu_vio_gravity_in_residual = torch.tensor([False], dtype=torch.bool)
        if visual_relative_pose_CiCj is None:
            visual_relative_pose_CiCj = pp.identity_SE3(1, dtype=torch.float32).tensor()
        if visual_relative_pose_cov is None:
            visual_relative_pose_cov = torch.eye(6, dtype=torch.float32).unsqueeze(0) * 1e6
        if visual_relative_pose_mean_mahalanobis_sq is None:
            visual_relative_pose_mean_mahalanobis_sq = -1.0
        if visual_compressed_uvd_reference_CjCi is None:
            visual_compressed_uvd_reference_CjCi = pp.identity_SE3(
                1, dtype=torch.float64
            ).tensor()
        if visual_compressed_uvd_hessian is None:
            visual_compressed_uvd_hessian = torch.zeros(
                (1, 6, 6), dtype=torch.float64
            )
        if visual_compressed_uvd_gradient is None:
            visual_compressed_uvd_gradient = torch.zeros(
                (1, 6), dtype=torch.float64
            )
        if visual_compressed_uvd_robust_cost is None:
            visual_compressed_uvd_robust_cost = -1.0
        if visual_compressed_uvd_mean_mahalanobis_sq is None:
            visual_compressed_uvd_mean_mahalanobis_sq = -1.0
        if visual_compressed_uvd_huber_delta is None:
            visual_compressed_uvd_huber_delta = 0.1

        frame_idx = self.graph.frames.push(FrameNode.init({
            "pose"        : est_pose,
            "T_BS"        : frame.stereo.T_BS,
            "need_interp" : torch.tensor([need_interp], dtype=torch.bool),
            "time_ns"     : torch.tensor([frame.stereo.frame_ns], dtype=torch.long),
            "K"           : frame.stereo.K,
            "baseline"    : frame.stereo.baseline,
            "imu_rotvec_prior": imu_rotvec_prior,
            "imu_rot_prior_std": imu_rot_prior_std,
            "imu_trans_prior": imu_trans_prior,
            "imu_trans_cov"  : imu_trans_cov,
            "imu_vio_prev_velocity_world": imu_vio_prev_velocity_world,
            "imu_vio_curr_velocity_init_world": imu_vio_curr_velocity_init_world,
            "imu_vio_velocity_world": imu_vio_curr_velocity_init_world,
            "imu_vio_prev_acc_bias": imu_vio_prev_acc_bias,
            "imu_vio_prev_gyro_bias": imu_vio_prev_gyro_bias,
            "imu_vio_acc_bias": imu_vio_acc_bias,
            "imu_vio_gyro_bias": imu_vio_gyro_bias,
            "imu_vio_linearized_acc_bias": imu_vio_linearized_acc_bias,
            "imu_vio_linearized_gyro_bias": imu_vio_linearized_gyro_bias,
            "imu_vio_bias_jacobian": imu_vio_bias_jacobian,
            "imu_vio_bias_rw_cov": imu_vio_bias_rw_cov,
            "imu_vio_delta_rotvec": imu_vio_delta_rotvec,
            "imu_vio_delta_v": imu_vio_delta_v,
            "imu_vio_delta_p": imu_vio_delta_p,
            "imu_vio_cov": imu_vio_cov,
            "imu_vio_sa_v2_unique_cov": imu_vio_sa_v2_unique_cov,
            "imu_vio_sa_v2_incoming_raw_time_ns": imu_vio_sa_v2_incoming_raw_time_ns,
            "imu_vio_sa_v2_outgoing_raw_time_ns": imu_vio_sa_v2_outgoing_raw_time_ns,
            "imu_vio_sa_v2_incoming_count": imu_vio_sa_v2_incoming_count,
            "imu_vio_sa_v2_outgoing_count": imu_vio_sa_v2_outgoing_count,
            "imu_vio_sa_v2_incoming_sensitivity": imu_vio_sa_v2_incoming_sensitivity,
            "imu_vio_sa_v2_outgoing_sensitivity": imu_vio_sa_v2_outgoing_sensitivity,
            "imu_vio_dt": imu_vio_dt,
            "imu_vio_sensor_T_imu": self._frame_imu_vio_sensor_T_imu(frame),
            "imu_vio_gravity_world": imu_vio_gravity_world,
            "imu_vio_gravity_in_residual": imu_vio_gravity_in_residual,
            "visual_relative_pose_CiCj": visual_relative_pose_CiCj,
            "visual_relative_pose_cov": visual_relative_pose_cov,
            "visual_relative_pose_num_points": torch.tensor(
                [int(visual_relative_pose_num_points)], dtype=torch.long
            ),
            "visual_relative_pose_num_inliers": torch.tensor(
                [int(visual_relative_pose_num_inliers)], dtype=torch.long
            ),
            "visual_relative_pose_mean_mahalanobis_sq": torch.tensor(
                [float(visual_relative_pose_mean_mahalanobis_sq)], dtype=torch.float32
            ),
            "visual_compressed_uvd_reference_CjCi": visual_compressed_uvd_reference_CjCi.reshape(1, 7).double(),
            "visual_compressed_uvd_hessian": visual_compressed_uvd_hessian.reshape(1, 6, 6).double(),
            "visual_compressed_uvd_gradient": visual_compressed_uvd_gradient.reshape(1, 6).double(),
            "visual_compressed_uvd_robust_cost": torch.tensor(
                [float(visual_compressed_uvd_robust_cost)], dtype=torch.float64
            ),
            "visual_compressed_uvd_num_points": torch.tensor(
                [int(visual_compressed_uvd_num_points)], dtype=torch.long
            ),
            "visual_compressed_uvd_num_inliers": torch.tensor(
                [int(visual_compressed_uvd_num_inliers)], dtype=torch.long
            ),
            "visual_compressed_uvd_mean_mahalanobis_sq": torch.tensor(
                [float(visual_compressed_uvd_mean_mahalanobis_sq)], dtype=torch.float64
            ),
            "visual_compressed_uvd_huber_delta": torch.tensor(
                [float(visual_compressed_uvd_huber_delta)], dtype=torch.float64
            ),
            "fusion_visual_quality": torch.tensor([-1.0], dtype=torch.float32),
            "fusion_degrade_score": torch.tensor([-1.0], dtype=torch.float32),
            "fusion_trans_switch": torch.tensor([-1.0], dtype=torch.float32),
            "fusion_rot_switch": torch.tensor([-1.0], dtype=torch.float32),
            "fusion_xy_weight": torch.tensor([0.0], dtype=torch.float32),
            "fusion_z_weight": torch.tensor([0.0], dtype=torch.float32),
            "fusion_rot_weight": torch.tensor([0.0], dtype=torch.float32),
            "fusion_gate_flags": torch.zeros((1, 4), dtype=torch.float32),
        }))
        return frame_idx

    @staticmethod
    def _imu_sample_sigma(
        density,
        time_ns: torch.Tensor,
        *,
        floor: float,
        multiplier: float,
    ) -> torch.Tensor:
        values = torch.as_tensor(density, dtype=torch.float64).reshape(-1)
        if values.numel() == 1:
            values = values.repeat(3)
        if values.numel() != 3:
            raise ValueError("IMU calibration sigma must be scalar or three-axis")
        stamps = torch.as_tensor(time_ns, dtype=torch.long).reshape(-1)
        intervals_s = (stamps[1:] - stamps[:-1]).to(torch.float64) * 1e-9
        intervals_s = intervals_s[intervals_s > 0.0]
        if intervals_s.numel() > 0:
            sample_sigma = values / torch.sqrt(torch.median(intervals_s))
        else:
            sample_sigma = values
        return torch.maximum(
            sample_sigma * float(multiplier),
            torch.full_like(sample_sigma, float(floor)),
        )

    def _try_static_imu_initialization(
        self,
        frame: T_SensorFrame,
        time_ns: torch.Tensor,
        acc_body: torch.Tensor,
        gyro_body: torch.Tensor,
        gravity: float,
    ) -> bool:
        """Accumulate the configured startup interval and initialize IMU state.

        Returns True while the current frame must remain anchored by startup
        ZUPT. During this interval visual packets are still consumed and checked,
        but pose/state optimization is paused. The first post-initialization IMU
        edge therefore starts after the static interval.
        """
        if self._imu_static_initialized:
            self._imu_static_zupt_active = False
            return False
        if self.imu_static_initialization_mode == "off":
            self._imu_static_initialized = True
            self._imu_static_zupt_active = False
            return False
        self._imu_static_zupt_active = True

        stamps = time_ns.reshape(-1).long().detach().cpu()
        acc = acc_body.reshape(-1, 3).float().detach().cpu()
        gyro = gyro_body.reshape(-1, 3).float().detach().cpu()
        if self._imu_static_last_time_ns is not None:
            keep = stamps > int(self._imu_static_last_time_ns)
            stamps = stamps[keep]
            acc = acc[keep]
            gyro = gyro[keep]
        if stamps.numel() > 0:
            self._imu_static_time_chunks.append(stamps)
            self._imu_static_acc_chunks.append(acc)
            self._imu_static_gyro_chunks.append(gyro)
            self._imu_static_last_time_ns = int(stamps[-1].item())

        if self._imu_static_initial_rotation is None:
            extrinsic_CI = self._frame_imu_vio_sensor_T_imu(frame)
            if self.prev_keyframe is not None:
                pose_WC = pp.SE3(self.graph.frames.data["pose"][self.prev_keyframe[1]])
                pose_WI = pose_WC @ extrinsic_CI.to(device=pose_WC.device, dtype=pose_WC.dtype)
                self._imu_static_anchor_pose = pp.SE3(pose_WC.tensor().detach().clone()).float()
                self._imu_static_initial_rotation = pp.SO3(
                    pose_WI.rotation().tensor().detach().cpu().float().reshape(4)
                )
            else:
                self._imu_static_anchor_pose = pp.identity_SE3(dtype=torch.float32)
                self._imu_static_initial_rotation = pp.SO3(
                    extrinsic_CI.rotation().tensor().detach().cpu().float().reshape(4)
                )

        if not self._imu_static_time_chunks:
            return True
        all_time = torch.cat(self._imu_static_time_chunks, dim=0)
        all_acc = torch.cat(self._imu_static_acc_chunks, dim=0)
        all_gyro = torch.cat(self._imu_static_gyro_chunks, dim=0)
        acc_density = getattr(frame, "imu_calib_acc_sigma", self.imu_sigma_acc)
        gyro_density = getattr(frame, "imu_calib_gyro_sigma", self.imu_sigma_gyro)
        acc_std_limit = self._imu_sample_sigma(
            acc_density,
            all_time,
            floor=0.05,
            multiplier=self.imu_static_sigma_multiplier,
        )
        gyro_std_limit = self._imu_sample_sigma(
            gyro_density,
            all_time,
            floor=0.005,
            multiplier=self.imu_static_sigma_multiplier,
        )

        adaptive_diag: dict[str, T.Any] = {}
        if self.imu_static_initialization_mode == "fixed":
            assert self.imu_static_initialization_duration_s is not None
            required_end = int(all_time[0].item()) + int(
                round(self.imu_static_initialization_duration_s * 1e9)
            )
            if int(all_time[-1].item()) < required_end:
                self._imu_static_init_diag = {
                    "mode": "fixed",
                    "status": "collecting",
                    "stationary": None,
                    "duration_s": float((all_time[-1] - all_time[0]).item()) * 1e-9,
                    "required_duration_s": self.imu_static_initialization_duration_s,
                    "sample_count": int(all_time.numel()),
                }
                return True

            end_index = int(
                torch.searchsorted(all_time, torch.tensor(required_end), right=False).item()
            )
            end_index = min(end_index + 1, int(all_time.numel()))
            init_time = all_time[:end_index]
            init_acc = all_acc[:end_index]
            init_gyro = all_gyro[:end_index]
            result = estimate_static_imu_initialization(
                time_ns=init_time,
                acc_body=init_acc,
                gyro_body=init_gyro,
                initial_body_to_world=self._imu_static_initial_rotation,
                gravity=gravity,
                min_duration_s=self.imu_static_initialization_duration_s,
                gyro_mean_norm_max=self.imu_static_gyro_mean_norm_max,
                gyro_std_max=gyro_std_limit,
                acc_norm_error_max=self.imu_static_acc_norm_error_max,
                acc_std_max=acc_std_limit,
            )
        else:
            decision = evaluate_adaptive_static_imu_initialization(
                time_ns=all_time,
                acc_body=all_acc,
                gyro_body=all_gyro,
                initial_body_to_world=self._imu_static_initial_rotation,
                gravity=gravity,
                min_duration_s=self.imu_static_adaptive_min_duration_s,
                max_duration_s=self.imu_static_adaptive_max_duration_s,
                window_s=self.imu_static_adaptive_window_s,
                stable_hold_s=self.imu_static_adaptive_stable_hold_s,
                target_gyro_bias_sem=self.imu_static_adaptive_target_gyro_bias_sem,
                target_gravity_direction_sem_rad=(
                    self.imu_static_adaptive_target_gravity_direction_sem_rad
                ),
                gyro_mean_norm_max=self.imu_static_gyro_mean_norm_max,
                gyro_std_max=gyro_std_limit,
                acc_norm_error_max=self.imu_static_acc_norm_error_max,
                acc_std_max=acc_std_limit,
            )
            result = decision.initialization
            adaptive_diag = {
                "ready": bool(decision.ready),
                "timed_out": bool(decision.timed_out),
                "recent_stationary": bool(decision.recent_stationary),
                "gyro_bias_sem_max": float(decision.gyro_bias_sem_max),
                "gravity_direction_sem_rad": float(decision.gravity_direction_sem_rad),
                "gyro_mean_drift_norm": float(decision.gyro_mean_drift_norm),
                "gravity_direction_drift_rad": float(
                    decision.gravity_direction_drift_rad
                ),
            }
            if not decision.ready:
                self._imu_static_init_diag = {
                    "mode": "adaptive",
                    "status": "collecting" if not decision.timed_out else "failed",
                    "stationary": bool(result.stationary),
                    "duration_s": float(result.duration_s),
                    "sample_count": int(result.sample_count),
                    "failure_reason": decision.failure_reason,
                    **adaptive_diag,
                }
                if decision.timed_out:
                    if self.pipeline_trace_path:
                        init_path = Path(self.pipeline_trace_path).with_name(
                            "static_initialization.json"
                        )
                        init_path.write_text(
                            json.dumps(self._imu_static_init_diag, indent=2),
                            encoding="utf-8",
                        )
                    raise RuntimeError(
                        "Adaptive static IMU initialization timed out: "
                        f"{decision.failure_reason}; diagnostics={self._imu_static_init_diag}"
                    )
                return True
            init_time = all_time
            init_acc = all_acc
            init_gyro = all_gyro

        estimate_applied = self.imu_static_initialization_state_policy == "estimated"
        applied_acc_bias = (
            result.acc_bias.clone().float() if estimate_applied else torch.zeros(3, dtype=torch.float32)
        )
        applied_gyro_bias = (
            result.gyro_bias.clone().float() if estimate_applied else torch.zeros(3, dtype=torch.float32)
        )
        applied_attitude = (
            result.body_to_world.clone().float()
            if estimate_applied
            else self._imu_static_initial_rotation.clone().float()
        )
        self._imu_static_init_diag = {
            "mode": self.imu_static_initialization_mode,
            "state_policy": self.imu_static_initialization_state_policy,
            "estimate_applied": estimate_applied,
            "status": "complete",
            "stationary": bool(result.stationary),
            "duration_s": float(result.duration_s),
            "sample_count": int(result.sample_count),
            "estimated_acc_bias": result.acc_bias.tolist(),
            "estimated_gyro_bias": result.gyro_bias.tolist(),
            "estimated_body_to_world_rotvec": result.body_to_world.Log().tensor().tolist(),
            "applied_acc_bias": applied_acc_bias.tolist(),
            "applied_gyro_bias": applied_gyro_bias.tolist(),
            "applied_body_to_world_rotvec": applied_attitude.Log().tensor().tolist(),
            "acc_std": result.acc_std.tolist(),
            "gyro_std": result.gyro_std.tolist(),
            "failure_reason": result.failure_reason,
            **adaptive_diag,
        }
        if self.pipeline_trace_path:
            init_path = Path(self.pipeline_trace_path).with_name("static_initialization.json")
            init_path.write_text(
                json.dumps(self._imu_static_init_diag, indent=2),
                encoding="utf-8",
            )
        if not result.stationary:
            raise RuntimeError(
                "Configured static IMU initialization interval is not stationary: "
                f"{result.failure_reason}; diagnostics={self._imu_static_init_diag}"
            )

        self._imu_acc_bias = applied_acc_bias
        self._imu_gyro_bias = applied_gyro_bias
        self._imu_vel_w = torch.zeros(3, dtype=torch.float32)
        self._imu_attitude_world = applied_attitude
        self._imu_last_frame_time_ns = int(init_time[-1].item())
        self._imu_attitude_last_time_ns = int(init_time[-1].item())
        self._imu_static_initialized = True
        Logger.write(
            "info",
            "Static IMU initialization complete: "
            f"mode={self.imu_static_initialization_mode}, "
            f"state_policy={self.imu_static_initialization_state_policy}, "
            f"duration={result.duration_s:.3f}s, samples={result.sample_count}, "
            f"estimated_acc_bias={result.acc_bias.tolist()}, "
            f"estimated_gyro_bias={result.gyro_bias.tolist()}, "
            f"applied_acc_bias={applied_acc_bias.tolist()}, "
            f"applied_gyro_bias={applied_gyro_bias.tolist()}",
        )
        return True

    def _estimate_imu_priors(
        self, frame: T_SensorFrame
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None, pp.LieTensor | None]:
        """
        Returns (rot_prior_vec, rot_prior_std, trans_prior, trans_cov, imu_rel_pose).

        Uses full 15-state IMU preintegration (Forster et al. TRO 2017) with:
          - Proper continuous-time noise density → sampled-noise covariance conversion
          - Velocity state propagation across keyframes
          - Gravity applied only in the factor residual
          - Raw IMU measurement-frame deltas, without camera/world pre-rotation

        rot_prior_vec: (1,3) preintegrated rotation vector in body_i frame
        rot_prior_std: (1,)  scalar std (diag mean of rot covariance)
        trans_prior:   (1,3) preintegrated translation in body_i frame, or None
        trans_cov:     (1,3,3) translation covariance (3×3 sub-block of 9×9 cov), or None
        imu_rel_pose:  SE3 relative motion from full preintegration, or None
        """
        no_rot  = torch.zeros((1, 3), dtype=torch.float32)
        no_std  = torch.tensor([1e6], dtype=torch.float32)
        sampling_v2_components = None

        if not isinstance(frame, StereoInertialFrame):
            self._pending_imu_diag = None
            self._pending_imu_vio_factor = None
            return no_rot, no_std, None, None, None

        imu = frame.imu
        if imu.time_ns.numel() == 0 or imu.gyro.numel() == 0:
            self._pending_imu_diag = None
            self._pending_imu_vio_factor = None
            return no_rot, no_std, None, None, None

        time_ns = imu.time_ns.reshape(-1).long()
        acc_raw = imu.acc.reshape(-1, 3).float()
        gyro_raw = imu.gyro.reshape(-1, 3).float()

        gravity = float(getattr(imu, "gravity", [9.81])[0]) if hasattr(imu, "gravity") else 9.81
        imu_world_frame = str(getattr(frame, "imu_world_frame", "NED"))
        preintegration_gravity = gravity_for_world_frame(gravity, imu_world_frame)
        if self._try_static_imu_initialization(
            frame,
            time_ns,
            acc_raw,
            gyro_raw,
            preintegration_gravity,
        ):
            self._pending_imu_diag = None
            self._pending_imu_vio_factor = None
            return no_rot, no_std, None, None, None

        vio_prev_velocity_world = self._imu_vel_w.clone().float()
        vio_prev_acc_bias = self._imu_acc_bias.clone().float()
        vio_prev_gyro_bias = self._imu_gyro_bias.clone().float()
        acc_unbiased = acc_raw - vio_prev_acc_bias.to(acc_raw)
        gyro_unbiased = gyro_raw - vio_prev_gyro_bias.to(gyro_raw)
        gravity_pose_source = self.imu_vio_gravity_pose_source
        gravity_handling = self.imu_vio_gravity_handling
        gravity_in_residual = gravity_is_standard_local_frame(gravity_handling)
        gravity_rp_active = False
        gravity_rp_angle = 0.0
        gravity_rp_acc_norm = float("nan")
        attitude_source_active = False
        attitude_source_angle_to_est = float("nan")

        if bool(getattr(self, "imu_legacy_gyro_prior_enable", False)):
            gyro_legacy = gyro_unbiased * self.imu_legacy_gyro_sign.to(gyro_unbiased)
            if gyro_legacy.size(0) == 1:
                omega_int = gyro_legacy[0] * 0.01
            else:
                dt_s = (time_ns[1:] - time_ns[:-1]).clamp(min=1).double() * 1e-9
                gyro_mid = 0.5 * (gyro_legacy[:-1].double() + gyro_legacy[1:].double())
                omega_int = (gyro_mid * dt_s.unsqueeze(-1)).sum(dim=0).float()
            rot_prior_vec = omega_int.reshape(1, 3)
            rot_prior_std = torch.tensor([self.imu_rot_prior_std], dtype=torch.float32)
            self._pending_imu_diag = None
            self._pending_imu_vio_factor = None
            return rot_prior_vec, rot_prior_std, None, None, None

        # ── Current world-frame orientation for gravity alignment
        R0_world = None
        if self.prev_keyframe is not None:
            prev_pose_WC = pp.SE3(self.graph.frames.data["pose"][self.prev_keyframe[1]])
            extrinsic_CI = self._frame_imu_vio_sensor_T_imu(frame).to(
                device=prev_pose_WC.device,
                dtype=prev_pose_WC.dtype,
            )
            R0_world = (prev_pose_WC @ extrinsic_CI).rotation()
            estimated_R0_world = R0_world
            if gravity_in_residual:
                gravity_pose_source = "optimized_state"
            elif gravity_pose_source == "gt_ref":
                gt_rotation = self._gt_internal_rotation_for_timestamp(
                    int(self.prev_keyframe[0].stereo.frame_ns),
                    internal_world_frame=str(getattr(frame, "imu_world_frame", "NED")),
                )
                if gt_rotation is not None:
                    R0_world = gt_rotation
            elif gravity_pose_source == "imu_integrated_estinit":
                if self._imu_attitude_world is None:
                    self._imu_attitude_world = pp.SO3(R0_world.tensor().detach().clone()).float()
                R0_world = pp.SO3(self._imu_attitude_world.tensor().detach().clone()).float()
                attitude_source_active = True
                attitude_source_angle_to_est = float(
                    (R0_world.Inv() @ estimated_R0_world.float()).Log().tensor().norm().item()
                )
            elif gravity_pose_source == "imu_gravity_rp":
                alignment = gravity_roll_pitch_aligned_rotation(
                    estimated_body_to_world=R0_world,
                    acc_body=acc_unbiased,
                    gravity=preintegration_gravity,
                    correction_gain=self.imu_gravity_rp_correction_gain,
                    acc_norm_tol=self.imu_gravity_rp_acc_tol,
                    window=self.imu_gravity_rp_window,
                )
                R0_world = alignment.rotation.float()
                gravity_rp_active = bool(alignment.active)
                gravity_rp_angle = float(alignment.correction_angle_rad)
                gravity_rp_acc_norm = float(alignment.acc_norm)

        # The dataset loader exposes continuous-time IMU noise densities directly.
        has_calib = hasattr(frame, "imu_calib_acc_sigma") and hasattr(frame, "imu_calib_gyro_sigma")
        if has_calib:
            sigma_acc  = getattr(frame, "imu_calib_acc_sigma",  self.imu_sigma_acc)
            sigma_gyro = getattr(frame, "imu_calib_gyro_sigma", self.imu_sigma_gyro)
            sigma_source = str(getattr(frame, "imu_calib_source", "frame_calibration"))
        else:
            sigma_acc  = self.imu_sigma_acc
            sigma_gyro = self.imu_sigma_gyro
            sigma_source = "MACVO_config"

        # Bias random walk from calibration (if available)
        if hasattr(frame, "imu_calib_acc_w_sigma") and hasattr(frame, "imu_calib_gyro_w_sigma"):
            sigma_acc_w  = getattr(frame, "imu_calib_acc_w_sigma",  self.imu_sigma_acc_w)
            sigma_gyro_w = getattr(frame, "imu_calib_gyro_w_sigma", self.imu_sigma_gyro_w)
        else:
            sigma_acc_w  = self.imu_sigma_acc_w
            sigma_gyro_w = self.imu_sigma_gyro_w

        try:
            if gravity_in_residual:
                preint = preintegrate_imu_local_frame(
                    time_ns=time_ns,
                    acc=acc_raw,
                    gyro=gyro_raw,
                    sigma_acc=sigma_acc,
                    sigma_gyro=sigma_gyro,
                    sigma_acc_w=sigma_acc_w,
                    sigma_gyro_w=sigma_gyro_w,
                    acc_bias=vio_prev_acc_bias,
                    gyro_bias=vio_prev_gyro_bias,
                )
                if self.two_state_covariance_mode in {
                    SAMPLING_AWARE,
                    SAMPLING_AWARE_CROSS_EDGE,
                }:
                    knot_from_raw = getattr(frame, "imu_sampling_knot_from_raw", None)
                    if knot_from_raw is None:
                        raise ValueError(
                            "sampling-aware covariance requires the raw-to-knot interpolation map"
                        )
                    raw_time_ns = getattr(frame, "imu_sampling_raw_time_ns", None)
                    if raw_time_ns is None:
                        raise ValueError(
                            "sampling-aware covariance requires raw IMU timestamps"
                        )
                    sensor_to_internal_rotation = torch.eye(
                        3,
                        dtype=torch.float64,
                        device=acc_raw.device,
                    )
                    if self.two_state_covariance_mode == SAMPLING_AWARE:
                        preint = replace_with_sampling_aware_covariance(
                            preint,
                            time_ns=time_ns,
                            acc_internal=acc_raw,
                            gyro_internal=gyro_raw,
                            knot_from_raw=knot_from_raw,
                            sensor_to_internal_rotation=sensor_to_internal_rotation,
                            raw_time_ns=raw_time_ns,
                            sigma_acc=sigma_acc,
                            sigma_gyro=sigma_gyro,
                            acc_bias=vio_prev_acc_bias,
                            gyro_bias=vio_prev_gyro_bias,
                        )
                    else:
                        sampling_v2_components = build_sampling_aware_covariance_components(
                            preint,
                            time_ns=time_ns,
                            acc_internal=acc_raw,
                            gyro_internal=gyro_raw,
                            knot_from_raw=knot_from_raw,
                            sensor_to_internal_rotation=sensor_to_internal_rotation,
                            raw_time_ns=raw_time_ns,
                            sigma_acc=sigma_acc,
                            sigma_gyro=sigma_gyro,
                            acc_bias=vio_prev_acc_bias,
                            gyro_bias=vio_prev_gyro_bias,
                        )
                        incoming_times = sampling_v2_components.incoming_raw_time_ns
                        outgoing_times = sampling_v2_components.outgoing_raw_time_ns
                        if incoming_times.numel() > 2 or outgoing_times.numel() > 2:
                            raise ValueError(
                                "SA-v2 storage supports at most two raw samples per endpoint"
                            )
                        if bool(
                            (incoming_times.reshape(-1, 1) == outgoing_times.reshape(1, -1))
                            .any()
                            .item()
                        ):
                            raise ValueError(
                                "SA-v2 requires disjoint start/end endpoint raw-sample supports"
                            )
                        preint = replace(
                            preint,
                            cov=sampling_v2_components.total_covariance.float(),
                            measurement_cov=(
                                sampling_v2_components.measurement_covariance.float()
                            ),
                        )
            else:
                preint = preintegrate_imu(
                    time_ns=time_ns,
                    acc=acc_raw,
                    gyro=gyro_raw,
                    R0_world=R0_world,
                    gravity=preintegration_gravity,
                    sigma_acc=sigma_acc,
                    sigma_gyro=sigma_gyro,
                    sigma_acc_w=sigma_acc_w,
                    sigma_gyro_w=sigma_gyro_w,
                    acc_bias=vio_prev_acc_bias,
                    gyro_bias=vio_prev_gyro_bias,
                    gravity_handling=gravity_handling,
                )
            if (
                not gravity_in_residual
                and gravity_pose_source == "imu_integrated_estinit"
                and attitude_source_active
                and R0_world is not None
            ):
                self._imu_attitude_world = integrate_gyro_attitude_world(
                    R0_world,
                    time_ns,
                    gyro_unbiased,
                ).float()
                self._imu_attitude_last_time_ns = int(time_ns[-1].item())
        except Exception as error:
            Logger.write(
                "warn",
                "IMU preintegration/attitude propagation failed; using gyro-only "
                f"fallback for this frame: {type(error).__name__}: {error}",
            )
            # Fallback to simple gyro-only integration
            if gyro_unbiased.size(0) == 1:
                omega_int = gyro_unbiased[0] * 0.01
            else:
                dt_s = (time_ns[1:] - time_ns[:-1]).clamp(min=1.0).double() * 1e-9
                gyro_mid = 0.5 * (gyro_unbiased[:-1].double() + gyro_unbiased[1:].double())
                omega_int = (gyro_mid * dt_s.unsqueeze(-1)).sum(dim=0).float()
            rot_prior_vec = omega_int.reshape(1, 3)
            rot_prior_std = torch.tensor([self.imu_rot_prior_std], dtype=torch.float32)
            self._pending_imu_diag = None
            self._pending_imu_vio_factor = None
            return rot_prior_vec, rot_prior_std, None, None, None

        # ── Extract rotation prior: delta_R in body_i frame, as rotation vector
        delta_R = preint.delta_R  # SO3
        preint_rotvec = delta_R.Log().tensor().reshape(1, 3).float()
        rot_prior_vec = preint_rotvec.clone()

        # ── Rotation uncertainty: use calibration noise density as floor
        rot_cov = preint.cov[6:9, 6:9]  # (3,3)
        rot_var_mean = float(rot_cov.diagonal().clamp(min=1e-12).mean().item())
        rot_prior_std_preint = max(float(rot_var_mean ** 0.5), 1e-6)
        rot_prior_std_value = select_rotation_prior_std(
            preintegrated_std=rot_prior_std_preint,
            sensor_noise_std=imu_sigma_rms(sigma_gyro),
            configured_floor=self.imu_rot_prior_std,
        )
        rot_prior_std = torch.tensor([rot_prior_std_value], dtype=torch.float32)

        # ── Translation prior in body_i frame.
        # Modes keep the adaptive-state-machine direction unchanged while making
        # the actual pose prior explicit and testable.
        delta_p = preint.delta_p  # (3,)
        visual_velocity_world = self._estimate_visual_velocity_world()
        if self.prev_keyframe is not None and R0_world is not None:
            R_world = R0_world.matrix().float()
            if R_world.dim() == 3:
                R_world = R_world.squeeze(0)
        else:
            R_world = torch.eye(3, dtype=torch.float32)
        trans_prior_body = compose_translation_prior_by_mode(
            mode=self.imu_trans_prior_mode,
            delta_p_body=delta_p.float(),
            imu_velocity_world=self._imu_vel_w.to(delta_p),
            visual_velocity_world=visual_velocity_world.to(delta_p) if visual_velocity_world is not None else None,
            R_body_to_world=R_world.to(delta_p),
            dt_total=preint.dt_total,
        )
        gravity_world = torch.tensor(
            [0.0, 0.0, preintegration_gravity],
            dtype=delta_p.dtype,
            device=delta_p.device,
        )
        if gravity_in_residual:
            gravity_body = R_world.to(delta_p).T @ gravity_world
            trans_prior_body = trans_prior_body + 0.5 * gravity_body * (float(preint.dt_total) ** 2)
        trans_prior = trans_prior_body.reshape(1, 3).float()

        # ── Translation: IMU position integration is inherently unreliable
        # over short baselines. Use a fixed covariance floor and report the
        # optimizer-scaled std in diagnostics so the actual strength is visible.
        trans_cov = preint.cov[0:3, 0:3].float()  # (3,3)
        trans_floor = 0.05  # m, fixed conservative floor
        trans_var_mean = float(trans_cov.diagonal().clamp(min=1e-12).mean().item())
        trans_std_preint = max(float(trans_var_mean ** 0.5), 1e-6)
        trans_std = max(trans_std_preint, trans_floor)
        trans_cov = trans_cov + torch.eye(3) * (trans_std**2 - trans_std_preint**2)
        trans_cov = trans_cov.unsqueeze(0)  # (1, 3, 3)
        trans_scale = max(float(getattr(self.Optimizer.config, "imu_trans_prior_scale", 1.0)), 1e-6)
        trans_prior_std_diag = (trans_cov.reshape(3, 3) * trans_scale).diagonal().clamp(min=0).sqrt().float()

        # ── Assemble SE3 relative pose (translation + rotation)
        imu_rel_pose = pp.SE3(torch.cat([
            trans_prior_body.reshape(1, 3).float(),
            delta_R.tensor().reshape(1, 4).float(),
        ], dim=-1))

        # ── D2: Save full IMU priors BEFORE adaptive gate nullification ─
        _d2_trans_prior_full = trans_prior.clone() if trans_prior is not None else None
        _d2_trans_cov_full = trans_cov.clone() if trans_cov is not None else None
        vio_use_imu_rotation = bool(getattr(self, "imu_rot_prior_enable", True))
        vio_use_imu_translation = bool(
            self.imu_trans_prior_mode != "off"
            and self.imu_trans_prior_enable
            and trans_prior is not None
            and trans_cov is not None
        )

        # ── Ablation gate: disable translation prior when flag is false
        if self.imu_trans_prior_mode == "off" or not self.imu_trans_prior_enable:
            vio_use_imu_translation = False
            trans_prior = None
            trans_cov = None

        # ── Ablation gate: disable rotation prior when flag is false
        if not getattr(self, "imu_rot_prior_enable", True):
            vio_use_imu_rotation = False
            rot_prior_vec = torch.zeros((1, 3), dtype=torch.float32)
            rot_prior_std = torch.tensor([1e6], dtype=torch.float32)

        # ── adaptive_v1: override ablation gates with gate decision ──
        if self._adaptive_enabled and self._adaptive_decision is not None:
            if not self._adaptive_decision.use_imu_rotation:
                vio_use_imu_rotation = False
                rot_prior_vec = torch.zeros((1, 3), dtype=torch.float32)
                rot_prior_std = torch.tensor([1e6], dtype=torch.float32)
            if not self._adaptive_decision.use_imu_translation:
                vio_use_imu_translation = False
                trans_prior = None
                trans_cov = None

        if trans_prior is not None and trans_cov is not None:
            rot_prior_std_value = select_translation_active_rotation_prior_std(
                base_std=float(rot_prior_std.reshape(-1)[0].item()),
                translation_active=True,
                translation_active_floor=self.imu_rot_prior_std_when_translation,
            )
            rot_prior_std = torch.tensor([rot_prior_std_value], dtype=torch.float32)

        # ── Update IMU velocity state (world frame) for next frame
        vio_curr_velocity_init_world = vio_prev_velocity_world.clone()
        if self.prev_keyframe is not None:
            if R0_world is not None:
                R_world = R0_world.matrix().float()  # (1, 3, 3) or (3, 3)
            else:
                prev_pose_se3 = pp.SE3(self.graph.frames.data["pose"][self.prev_keyframe[1]].float())
                R_world = prev_pose_se3.rotation().matrix().float()  # (1, 3, 3) or (3, 3)
            if R_world.dim() == 3:
                R_world = R_world.squeeze(0)
            delta_v_body = preint.delta_v.float()  # (3,)
            vio_curr_velocity_init_world = propagate_imu_velocity_world(
                velocity_world=self._imu_vel_w,
                delta_v_body=delta_v_body,
                R_body_to_world=R_world,
                gravity_world=gravity_world,
                dt_total=preint.dt_total,
                gravity_handling=gravity_handling,
            ).float()
            if self.imu_vio_velocity_feedback_enable:
                self._imu_vel_w = vio_curr_velocity_init_world.clone()

        self._imu_last_frame_time_ns = int(time_ns[-1].item())
        vio_factor_allowed = should_enable_preintegrated_vio_factor(
            use_imu_rotation=vio_use_imu_rotation,
            use_imu_translation=vio_use_imu_translation,
            dt_total=float(preint.dt_total),
        )
        sa_v2_unique_cov = torch.eye(9, dtype=torch.float32).unsqueeze(0) * 1e6
        sa_v2_incoming_times = torch.full((1, 2), -1, dtype=torch.long)
        sa_v2_outgoing_times = torch.full((1, 2), -1, dtype=torch.long)
        sa_v2_incoming_count = torch.zeros(1, dtype=torch.long)
        sa_v2_outgoing_count = torch.zeros(1, dtype=torch.long)
        sa_v2_incoming_sensitivity = torch.zeros((1, 9, 12), dtype=torch.float32)
        sa_v2_outgoing_sensitivity = torch.zeros((1, 9, 12), dtype=torch.float32)
        if sampling_v2_components is not None:
            incoming_count = int(sampling_v2_components.incoming_raw_time_ns.numel())
            outgoing_count = int(sampling_v2_components.outgoing_raw_time_ns.numel())
            sa_v2_unique_cov = sampling_v2_components.unique_covariance.reshape(
                1, 9, 9
            ).float()
            sa_v2_incoming_times[0, :incoming_count] = (
                sampling_v2_components.incoming_raw_time_ns.cpu()
            )
            sa_v2_outgoing_times[0, :outgoing_count] = (
                sampling_v2_components.outgoing_raw_time_ns.cpu()
            )
            sa_v2_incoming_count[0] = incoming_count
            sa_v2_outgoing_count[0] = outgoing_count
            sa_v2_incoming_sensitivity[0, :, : incoming_count * 6] = (
                sampling_v2_components.incoming_sensitivity.cpu().float()
            )
            sa_v2_outgoing_sensitivity[0, :, : outgoing_count * 6] = (
                sampling_v2_components.outgoing_sensitivity.cpu().float()
            )
        self._pending_imu_vio_factor = (
            {
                "prev_velocity_world": vio_prev_velocity_world.reshape(1, 3).float(),
                "curr_velocity_init_world": vio_curr_velocity_init_world.reshape(1, 3).float(),
                "prev_acc_bias": vio_prev_acc_bias.reshape(1, 3).float(),
                "prev_gyro_bias": vio_prev_gyro_bias.reshape(1, 3).float(),
                "curr_acc_bias_init": vio_prev_acc_bias.reshape(1, 3).float(),
                "curr_gyro_bias_init": vio_prev_gyro_bias.reshape(1, 3).float(),
                "linearized_acc_bias": (
                    preint.linearized_acc_bias.reshape(1, 3).float()
                    if preint.linearized_acc_bias is not None
                    else vio_prev_acc_bias.reshape(1, 3).float()
                ),
                "linearized_gyro_bias": (
                    preint.linearized_gyro_bias.reshape(1, 3).float()
                    if preint.linearized_gyro_bias is not None
                    else vio_prev_gyro_bias.reshape(1, 3).float()
                ),
                "bias_jacobian": (
                    preint.bias_jacobian.reshape(1, 9, 6).float()
                    if preint.bias_jacobian is not None
                    else torch.zeros((1, 9, 6), dtype=torch.float32)
                ),
                "bias_rw_cov": (
                    preint.bias_rw_cov.reshape(1, 6, 6).float()
                    if preint.bias_rw_cov is not None
                    else torch.eye(6, dtype=torch.float32).unsqueeze(0) * 1e6
                ),
                "delta_rotvec": preint_rotvec.reshape(1, 3).float(),
                "delta_v": preint.delta_v.reshape(1, 3).float(),
                "delta_p": preint.delta_p.reshape(1, 3).float(),
                "cov": preint.cov.reshape(1, 9, 9).float(),
                "sa_v2_unique_cov": sa_v2_unique_cov,
                "sa_v2_incoming_raw_time_ns": sa_v2_incoming_times,
                "sa_v2_outgoing_raw_time_ns": sa_v2_outgoing_times,
                "sa_v2_incoming_count": sa_v2_incoming_count,
                "sa_v2_outgoing_count": sa_v2_outgoing_count,
                "sa_v2_incoming_sensitivity": sa_v2_incoming_sensitivity,
                "sa_v2_outgoing_sensitivity": sa_v2_outgoing_sensitivity,
                "dt": torch.tensor([float(preint.dt_total)], dtype=torch.float32),
                "gravity_world": gravity_world.reshape(1, 3).float(),
                "gravity_in_residual": torch.tensor([gravity_in_residual], dtype=torch.bool),
            }
            if vio_factor_allowed
            else None
        )

        # ── Store IMU preintegration diagnostics for frame_pair_diagnostics.csv ──
        self._pending_imu_diag = {
            "delta_R": preint.delta_R.float(),
            "delta_p": preint.delta_p.float(),
            "delta_v": preint.delta_v.float(),
            "dt_total": preint.dt_total,
            "num_imu_samples": int(time_ns.numel()),
            "translation_semantics": self.imu_translation_semantics,
            "translation_prior_mode": self.imu_trans_prior_mode,
            "translation_prior_std_diag": (
                trans_prior_std_diag
                if self.imu_trans_prior_mode != "off" and self.imu_trans_prior_enable
                else None
            ),
            "noise_source": sigma_source,
            "imu_world_frame": imu_world_frame,
            "imu_source_world_frame": str(getattr(frame, "imu_source_world_frame", "")),
            "imu_source_measurement_frame": str(getattr(frame, "imu_source_measurement_frame", "")),
            "imu_internal_world_frame": str(getattr(frame, "imu_internal_world_frame", imu_world_frame)),
            "imu_internal_measurement_frame": str(getattr(frame, "imu_internal_measurement_frame", imu_world_frame)),
            "imu_gravity_source": str(getattr(frame, "imu_gravity_source", "")),
            "imu_metadata_gravity_m_s2": getattr(frame, "imu_metadata_gravity_m_s2", None),
            "preintegration_gravity_z": preintegration_gravity,
            "gravity_world": gravity_world.detach().cpu().float(),
            "gravity_pose_source": gravity_pose_source,
            "gravity_handling": gravity_handling,
            "attitude_source_active": attitude_source_active,
            "attitude_source_angle_to_est_rad": attitude_source_angle_to_est,
            "gravity_rp_active": gravity_rp_active,
            "gravity_rp_angle_rad": gravity_rp_angle,
            "gravity_rp_acc_norm": gravity_rp_acc_norm,
            # ── D2: Full IMU priors before nullification ──────────────────
            "trans_prior_full": _d2_trans_prior_full,
            "trans_cov_full": _d2_trans_cov_full,
        }

        return rot_prior_vec, rot_prior_std, trans_prior, trans_cov, imu_rel_pose

    def _estimate_visual_velocity_world(self) -> torch.Tensor | None:
        if self.prev_keyframe is None:
            return None
        curr_idx = int(self.prev_keyframe[1])
        prev_idx = curr_idx - 1
        if prev_idx < 0 or curr_idx >= len(self.graph.frames):
            return None
        try:
            pose_prev = pp.SE3(self.graph.frames.data["pose"][prev_idx].float())
            pose_curr = pp.SE3(self.graph.frames.data["pose"][curr_idx].float())
            time_prev = int(self.graph.frames.data["time_ns"][prev_idx].item())
            time_curr = int(self.graph.frames.data["time_ns"][curr_idx].item())
        except Exception:
            return None
        dt = (time_curr - time_prev) * 1e-9
        if dt <= 1e-6:
            return None
        delta_world = pose_curr.translation().reshape(3) - pose_prev.translation().reshape(3)
        return (delta_world / dt).float()

    def _fuse_visual_imu_pose(
        self,
        visual_pose: pp.LieTensor | torch.Tensor,
        imu_rel_pose: pp.LieTensor | None,
        sensor_T_imu: pp.LieTensor | torch.Tensor | None = None,
    ) -> pp.LieTensor | torch.Tensor:
        """Fuse visual and IMU adjacent-frame motion in se(3) to correct initial pose."""
        if (not self.imu_pose_fusion_enable) or (imu_rel_pose is None) or (self.prev_keyframe is None):
            return visual_pose

        # Ensure all tensors on same device (use device of visual_pose)
        device = visual_pose.device if hasattr(visual_pose, 'device') else torch.device('cpu')

        prev_pose_tensor = self.graph.frames.data["pose"][self.prev_keyframe[1]].to(device)
        visual_pose_to = visual_pose.to(device) if hasattr(visual_pose, 'to') else visual_pose
        imu_rel_pose_to = imu_rel_pose.to(device) if hasattr(imu_rel_pose, 'to') else imu_rel_pose

        prev_pose = pp.SE3(prev_pose_tensor.unsqueeze(0) if prev_pose_tensor.dim() == 1 else prev_pose_tensor)
        visual_pose_se3 = pp.SE3(visual_pose_to)

        rel_visual = prev_pose.Inv() @ visual_pose_se3
        rel_imu = pp.SE3(imu_rel_pose_to)
        if sensor_T_imu is not None:
            extrinsic = pp.SE3(sensor_T_imu).to(device=device, dtype=rel_imu.tensor().dtype)
            rel_imu = extrinsic @ rel_imu @ extrinsic.Inv()

        alpha = min(max(float(self.imu_pose_fusion_alpha), 0.0), 1.0)
        xi_visual = rel_visual.Log().tensor()
        xi_imu = rel_imu.Log().tensor()
        xi_fused = alpha * xi_visual + (1.0 - alpha) * xi_imu

        rel_fused = pp.se3(xi_fused).Exp()
        return (prev_pose @ rel_fused).float()

    @Timer.cpu_timeit("Odom_Runtime")
    @Timer.gpu_timeit("Odom_Runtime")
    def run(self, frame: T_SensorFrame) -> None:
        """
        The main process that continuously running to manage different modules in MAC-VO.
        The multi-threading part will be managed in this function.
        Args:
            frame (T_SensorFrame): The current stereo frame to be processed.
        Returns:
            None
        """

        if not self.isinitiated:
            self.initialize(frame)
            self.isinitiated = True
            return

        assert self.prev_keyframe is not None
        self.run_pair(self.prev_keyframe[0], frame)

    def get_map(self) -> VisualMap:
        return self.graph

    def _validate_visual_cache_consumption_complete(self) -> None:
        reader = self._visual_cache_reader
        if reader is None:
            return
        expected_pairs = tuple(
            (int(pair["frame_i"]), int(pair["frame_j"]))
            for pair in reader.manifest.pairs
        )
        consumed_pairs = tuple(self._visual_cache_consumed_pairs)
        if consumed_pairs != expected_pairs:
            raise VisualFactorCacheError(
                "replay terminated without consuming the complete manifest pair range: "
                f"expected={expected_pairs}, consumed={consumed_pairs}"
            )

    def validate_completion(self) -> None:
        self._validate_visual_cache_consumption_complete()

    def terminate(self) -> None:
        if self.terminated:
            return
        super().terminate()
        if self.prev_keyframe is not None:
            commit_start = time.perf_counter()
            self.Optimizer.write_map(self.graph)
            commit_ms = (time.perf_counter() - commit_start) * 1000.0
            self._flush_pipeline_trace_pending(commit_ms)
            last_frame = self.prev_keyframe[0]
            self._write_frame_pair_diagnostics(last_frame, last_frame)
            final_result = self.Optimizer.finalize_map(self.graph)
            if final_result is not None:
                final_frames = getattr(final_result, "window_frame_indices", None)
                final_frame_list = (
                    [int(value) for value in final_frames.reshape(-1).tolist()]
                    if isinstance(final_frames, torch.Tensor)
                    else []
                )
                self._optimizer_finalize_summary = {
                    "schema_version": 1,
                    "backend": str(getattr(final_result, "vio_backend", "unknown")),
                    "reason": str(
                        getattr(
                            final_result,
                            "two_state_solver_convergence_reason",
                            "unknown",
                        )
                    ),
                    "history_revision": bool(
                        getattr(final_result, "isam2_history_revision", False)
                    ),
                    "writeback": str(
                        getattr(final_result, "local_ba_writeback", "unknown")
                    ),
                    "state_count": int(
                        getattr(final_result, "isam2_state_count", 0) or 0
                    ),
                    "first_frame": final_frame_list[0] if final_frame_list else None,
                    "last_frame": final_frame_list[-1] if final_frame_list else None,
                    "history_frame_count": len(final_frame_list),
                    "snapshot_build_ms": 1000.0
                    * float(
                        getattr(final_result, "local_ba_optimize_total_s", 0.0)
                        or 0.0
                    ),
                }
                self._sync_optimized_vio_velocity_from_map()
                for func in self.on_optimize_writeback:
                    func(self)
        self.Optimizer.terminate()
        self.MapRefiner.elaborate_map(self.graph.frames)
        # Close diagnostics writer
        if self._diag_writer is not None:
            self._diag_writer.close()
            self._diag_writer = None
        self._close_visual_factor_diagnostics()
        if self._pipeline_trace_stream is not None:
            self._pipeline_trace_stream.close()
            self._pipeline_trace_stream = None
            self._pipeline_trace_writer = None

    def export_diagnostics(self, saveto) -> None:
        logs = getattr(self.Optimizer, "fusion_logs", None)
        if logs:
            saveto.path("fusion_log.json").write_text(json.dumps(logs, indent=2), encoding="utf-8")
        if self._optimizer_finalize_summary is not None:
            saveto.path("optimizer_finalize_summary.json").write_text(
                json.dumps(
                    self._optimizer_finalize_summary,
                    indent=2,
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )

    def _close_visual_factor_diagnostics(self) -> None:
        if self._visual_factor_diag_stream is not None:
            self._visual_factor_diag_stream.close()
        self._visual_factor_diag_stream = None
        self._visual_factor_diag_writer = None

    def set_diagnostics_writer(self, writer, scene: str = "", method: str = "") -> None:
        """Attach a FramePairDiagnosticsWriter for per-pair CSV logging."""
        self._close_visual_factor_diagnostics()
        self._diag_writer = writer
        self._scene_name = scene
        self._method_name = method
        self._visual_factor_diag_written_pairs.clear()
        if writer is None or self._visual_cache_reader is not None:
            return
        diagnostics_path = getattr(writer, "_filepath", None)
        if diagnostics_path is None:
            return
        visual_path = Path(diagnostics_path).with_name("visual_factor_diagnostics.csv")
        self._visual_factor_diag_stream = visual_path.open("w", newline="", encoding="utf-8")
        fieldnames = [
            "frame_i",
            "frame_j",
            "timestamp_i",
            "timestamp_j",
            "visual_input_sha256",
            *FRONTEND_COVARIANCE_DIAGNOSTIC_FIELDS,
        ]
        self._visual_factor_diag_writer = csv.DictWriter(
            self._visual_factor_diag_stream,
            fieldnames=fieldnames,
        )
        self._visual_factor_diag_writer.writeheader()
        self._visual_factor_diag_stream.flush()

    def set_gt_positions(
        self,
        gt_positions: dict,
        gt_quaternions: dict | None = None,
        gt_velocities: dict | None = None,
        gt_angular_velocities: dict | None = None,
    ) -> None:
        """Provide GT positions dict {timestamp_ns: (x_nwu, y_nwu, z_nwu)} and
        optional quaternions and world-NWU velocities for diagnostics."""
        self._gt_positions = gt_positions
        self._gt_quaternions = gt_quaternions
        self._gt_velocities = gt_velocities
        self._gt_angular_velocities = gt_angular_velocities

    def _write_optimizer_breakpoint_trace(self, trace: dict | None, frame_indices: list[int], global_map: VisualMap) -> None:
        """Append optional optimizer breakpoint snapshots to a JSONL file.

        The snapshots are controlled by TwoFrame_PGO's trace flag and are meant
        for debugging only. They do not affect optimization or map writeback.
        """
        if not trace or self._diag_writer is None:
            return

        diag_path = getattr(self._diag_writer, "_filepath", None)
        if diag_path is None:
            return

        try:
            map_after_writeback = []
            for frame_idx in frame_indices:
                if frame_idx < 0 or frame_idx >= len(global_map.frames):
                    continue
                frame_state = {
                    "frame_idx": frame_idx,
                    "pose": global_map.frames.data["pose"][frame_idx].detach().cpu().float().tolist(),
                }
                if "time_ns" in global_map.frames.data:
                    frame_state["timestamp_ns"] = int(global_map.frames.data["time_ns"][frame_idx].item())
                if "imu_vio_velocity_world" in global_map.frames.data:
                    frame_state["velocity_world"] = (
                        global_map.frames.data["imu_vio_velocity_world"][frame_idx]
                        .detach().cpu().float().tolist()
                    )
                if "imu_vio_acc_bias" in global_map.frames.data:
                    frame_state["acc_bias"] = (
                        global_map.frames.data["imu_vio_acc_bias"][frame_idx]
                        .detach().cpu().float().tolist()
                    )
                if "imu_vio_gyro_bias" in global_map.frames.data:
                    frame_state["gyro_bias"] = (
                        global_map.frames.data["imu_vio_gyro_bias"][frame_idx]
                        .detach().cpu().float().tolist()
                    )
                map_after_writeback.append(frame_state)

            payload = {
                "scene": self._scene_name,
                "method": self._method_name,
                "from_idx": frame_indices[0] if frame_indices else -1,
                "frame_idx": frame_indices[-1] if frame_indices else -1,
                "trace": trace,
                "map_after_writeback": map_after_writeback,
            }
            trace_path = diag_path.with_name("optimization_breakpoint_trace.jsonl")
            with trace_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        except Exception as exc:
            Logger.write("warn", f"Failed to write optimizer breakpoint trace: {exc}")

    def _gt_internal_rotation_for_timestamp(
        self,
        timestamp_ns: int,
        *,
        internal_world_frame: str,
    ) -> pp.LieTensor | None:
        if self._gt_quaternions is None:
            return None
        q = self._gt_quaternions.get(int(timestamp_ns))
        if q is None:
            return None
        rotation_WC = nwu_xyzw_quaternion_to_internal_so3(
            q, internal_world_frame=internal_world_frame
        )
        try:
            rotation_CI = pp.SE3(
                self.graph.frames.data["imu_vio_sensor_T_imu"][0].float()
            ).rotation()
            return rotation_WC @ rotation_CI
        except Exception:
            return rotation_WC

    def enable_adaptive(self, gate, decisions_csv_path: str = "", version: str = "v1") -> None:
        """Enable adaptive mode with the given VisualHealthGate instance.
        version: 'v1', 'v2', 'v3a', or 'v3b' — determines CSV header and signal passing."""
        self._adaptive_enabled = True
        self._adaptive_gate = gate
        self._adaptive_version = version
        if decisions_csv_path:
            import csv as _csv
            _f = open(decisions_csv_path, "w", newline="", encoding="utf-8")
            _w = _csv.writer(_f)
            if version == "v3b":
                _w.writerow(["scene","pair_id","frame_i","frame_j","timestamp_i","timestamp_j",
                    "gate_version","adaptive_mode","previous_mode","state_name",
                    "use_imu_rotation","use_imu_translation","imu_translation_semantics","adaptive_reason",
                    "num_visual_residuals","median_flow_cov",
                    "r_R_whitened_norm","r_p_whitened_norm",
                    "imu_trans_loss","visual_loss_per_residual","total_loss",
                    "visual_collapse_raw","visual_collapse_triggered","visual_collapse_counter",
                    "full_divergence_raw","full_divergence_triggered","full_divergence_counter",
                    "rot_harm_raw","rot_harm_triggered","rot_harm_counter",
                    "rot_harm_cooldown_remaining","full_divergence_cooldown_remaining",
                    "probation_counter","cooldown_reason",
                    "visual_collapse_sustain_config",
                    "vc_mode","visual_collapse_trigger_source","visual_collapse_reason",
                    "severe_vc_raw","severe_vc_counter","severe_vc_triggered",
                    "severe_vc_threshold","severe_vc_sustain_config",
                    "mild_vc_raw","mild_vc_counter","mild_vc_triggered",
                    "mild_vc_threshold","mild_vc_sustain_config",
                    "velocity_reset_enabled","velocity_reset_triggered","velocity_reset_strategy",
                    "velocity_before_reset_norm","velocity_after_reset_norm",
                    "velocity_reset_pair","velocity_reset_reason",
                    "mode_transition_caused_velocity_reset",
                    "d2_rerun_enabled","d2_rerun_triggered","d2_rerun_pair_id",
                    "d2_rerun_reason","d2_committed_result_source",
                    "d2_pre_rerun_mode","d2_post_rerun_mode",
                    "d2_pre_rerun_est_delta_t_norm","d2_post_rerun_est_delta_t_norm",
                    "d2_pre_rerun_r_p_whitened_norm","d2_post_rerun_r_p_whitened_norm",
                    "d2_pre_rerun_num_vis_res","d2_post_rerun_num_vis_res",
                    "d2_rerun_failed","d2_rerun_failure_reason",
                    # ── FD-E fields (V3b++ Phase 1a) ──────────────────────
                    "fd_grace_enabled","fd_grace_period_config",
                    "full_imu_episode_frame_idx","fd_check_suppressed_by_grace",
                    "fd_grace_remaining","full_divergence_reason",
                    # ── Cooldown config (V3b++ Phase 1b) ──────────────────
                    "fd_cooldown_config","rot_harm_cooldown_config",
                    "cooldown_active"])
            elif version == "v3a":
                _w.writerow(["scene","frame_i","frame_j","timestamp_i","timestamp_j",
                    "gate_version","mode","use_imu_rotation","use_imu_translation","reason",
                    "median_flow_cov","running_median_flow_cov",
                    "last_valid_flow_cov","last_valid_running_median_flow_cov",
                    "flow_cov_window_size","flow_cov_enter_full_threshold","flow_cov_exit_full_threshold",
                    "cov_missing","cov_missing_counter","max_cov_bridge_frames","cov_missing_bridge_active",
                    "num_visual_residuals","num_selected_keypoints",
                    "visual_lost_track","lost_track_counter","lost_track_enter_frames",
                    "optimizer_skipped","match_idx_size_after_filter",
                    "num_keypoints_candidate","min_num_point",
                    "stable_visual_counter","stable_visual_exit_frames",
                    "full_hold_counter","current_mode_duration",
                    "median_depth_cov","p90_depth_cov",
                    "visual_loss_per_residual",
                    "r_R_whitened_norm","r_p_whitened_norm",
                    "imu_rot_loss","imu_trans_loss",
                    "est_delta_t_norm",
                    "fallback_triggered","hysteresis_state"])
            elif version == "v2":
                _w.writerow(["scene","frame_i","frame_j","gate_version","mode",
                    "use_imu_rotation","use_imu_translation",
                    "median_flow_cov","p90_flow_u_cov","mean_flow_u_cov",
                    "median_depth_cov","p90_depth_cov",
                    "visual_loss_raw_sum","visual_loss_per_residual",
                    "num_selected_keypoints","num_visual_residuals",
                    "cov_degraded_candidate","cov_degraded_counter",
                    "cov_recovery_counter","translation_hold_counter",
                    "r_R_whitened_norm","r_p_whitened_norm",
                    "reason","hysteresis_state"])
            else:
                _w.writerow(["scene","frame_i","frame_j","timestamp_i","timestamp_j",
                    "mode","use_imu_rotation","use_imu_translation",
                    "visual_health_score","degeneracy_score","motion_abnormal_score",
                    "num_visual_residuals","num_valid_points","visual_loss",
                    "est_delta_t_norm","est_delta_R_angle",
                    "r_R_whitened_norm","r_p_whitened_norm","reason","hysteresis_state"])
            _f.flush()
            self._adaptive_decisions_writer = (_f, _w)

    def _update_adaptive_gate(self, frame0: T_SensorFrame, frame1: T_SensorFrame) -> None:
        """Feed previous pair's diagnostics to VisualHealthGate, get decision for current pair."""
        if not self._adaptive_enabled or self._adaptive_gate is None:
            return
        # Save frame refs for adaptive CSV write (called after D2 block)
        self._last_gate_frame0 = frame0
        self._last_gate_frame1 = frame1

        # Gate decision

        version = getattr(self, "_adaptive_version", "v1")

        # Gather runtime-available signals from last optimization
        opt_diag = getattr(self.Optimizer, "last_pair_diagnostics", None)
        signals: dict = {
            "num_visual_residuals": opt_diag.get("num_visual_residuals", 0) if opt_diag else 0,
            "num_valid_points": opt_diag.get("num_observations", 0) if opt_diag else 0,
            "visual_loss": opt_diag.get("visual_loss_raw_sum", opt_diag.get("visual_loss", float("inf"))) if opt_diag else float("inf"),
            "visual_loss_raw_sum": opt_diag.get("visual_loss_raw_sum", opt_diag.get("visual_loss", float("inf"))) if opt_diag else float("inf"),
            "r_R_whitened_norm": opt_diag.get("r_R_whitened_norm", float("nan")) if opt_diag else float("nan"),
            "r_p_whitened_norm": opt_diag.get("r_p_whitened_norm", float("nan")) if opt_diag else float("nan"),
            "imu_samples_available": self._pending_imu_diag is not None,
            "metadata_ok": True,
        }

        # ── v2/v3a/v3b: Add frontend covariance signals ──────────────────
        cov_diag = self._pending_cov_diag or {}
        signals["median_flow_cov"] = cov_diag.get("median_flow_u_cov", float("nan"))
        signals["p90_flow_u_cov"] = cov_diag.get("p90_flow_u_cov", float("nan"))
        signals["mean_flow_u_cov"] = cov_diag.get("mean_flow_u_cov", float("nan"))
        signals["median_depth_cov"] = cov_diag.get("median_kp0_depth_cov", float("nan"))
        signals["p90_depth_cov"] = cov_diag.get("p90_kp0_depth_cov", float("nan"))
        signals["num_selected_keypoints"] = cov_diag.get("num_selected_keypoints", signals["num_valid_points"])

        # ── v3a/v3b: Add visual_loss_per_residual and IMU losses ────────
        if version in ("v3a", "v3b"):
            v_loss_raw = signals.get("visual_loss_raw_sum", float("inf"))
            n_vis = signals.get("num_visual_residuals", 0)
            signals["visual_loss_per_residual"] = v_loss_raw / max(n_vis, 1) if n_vis > 0 and v_loss_raw < float("inf") else float("inf")
            signals["imu_rot_loss"] = opt_diag.get("imu_rot_loss", float("nan")) if opt_diag else float("nan")
            signals["imu_trans_loss"] = opt_diag.get("imu_trans_loss", float("nan")) if opt_diag else float("nan")

        # ── v3a/v3b: Add optimizer tracking state from PREVIOUS pair ────
        if version in ("v3a", "v3b"):
            track_state = self._pending_tracking_state
            if track_state is not None:
                signals["optimizer_skipped"] = track_state.get("optimizer_skipped", False)
                signals["match_idx_size_after_filter"] = track_state.get("match_idx_size_after_filter", 0)
                signals["num_keypoints_candidate"] = track_state.get("num_keypoints_candidate", 0)
                signals["min_num_point"] = track_state.get("min_num_point", 10)
                signals["has_tracking_state"] = True
            else:
                signals["optimizer_skipped"] = False
                signals["match_idx_size_after_filter"] = 0
                signals["num_keypoints_candidate"] = 0
                signals["min_num_point"] = 10
                signals["has_tracking_state"] = False

        # Estimate current frame-pair motion (if prev pair data available)
        try:
            prev_idx = self.prev_keyframe[1] if self.prev_keyframe else -1
            if prev_idx > 0:
                pose_prev = pp.SE3(self.graph.frames.data["pose"][prev_idx].float())
                pose_curr = pp.SE3(self.graph.frames.data["pose"][prev_idx + 1].float()) if prev_idx + 1 < len(self.graph.frames) else pose_prev
                rel = pose_prev.Inv() @ pose_curr
                signals["est_delta_t_norm"] = float(rel.translation().norm().item())
                signals["est_delta_R_angle"] = float(rel.rotation().Log().tensor().norm().item())
        except Exception:
            pass

        # Save signals for adaptive CSV write (used in _write_adaptive_decision_csv)
        self._last_gate_signals = signals

        # Gate decision
        if version == "v3b":
            pair_id = self._pair_counter + 1  # current pair being set up
            decision = self._adaptive_gate.update(signals, pair_id=pair_id)
        else:
            decision = self._adaptive_gate.update(signals)
        self._adaptive_decision = decision

        # ── Ablation: velocity reset on full_imu entry ───────────────
        if (version == "v3b" and getattr(decision, "velocity_reset_triggered", False)):
            vel_before = float(torch.norm(self._imu_vel_w).item())
            self._imu_vel_w = torch.zeros(3, dtype=torch.float32)
            vel_after = 0.0

            # Update decision with actual values
            decision.velocity_before_reset_norm = vel_before
            decision.velocity_after_reset_norm = vel_after
            decision.velocity_reset_pair = pair_id

            Logger.write("info",
                f"V3b velocity reset: mode={decision.mode}, "
                f"pair={pair_id}, strategy={decision.velocity_reset_strategy}, "
                f"vel_before={vel_before:.4f} m/s, vel_after={vel_after:.4f} m/s")

    def _write_adaptive_decision_csv(self, gate_signals: dict | None = None):
        """Write one row to adaptive_decisions.csv using the current decision."""
        if not self._adaptive_decisions_writer:
            return
        _f, _w = self._adaptive_decisions_writer
        version = getattr(self, "_adaptive_version", "v1")
        decision = self._adaptive_decision
        if decision is None:
            return

        opt_diag = getattr(self.Optimizer, "last_pair_diagnostics", None)
        # Fallback: use stored signals from last _update_adaptive_gate call
        sig = gate_signals if gate_signals is not None else getattr(self, "_last_gate_signals", {})
        frame0, frame1 = None, None  # optional, only for v3b rows
        # Restore frame references from last pair if available
        if hasattr(self, "_last_gate_frame0") and hasattr(self, "_last_gate_frame1"):
            frame0 = self._last_gate_frame0
            frame1 = self._last_gate_frame1

        try:
            if version == "v3b":
                d = decision
                def _fmt(v):
                    if isinstance(v, float) and (v != v or v == float("inf") or v == float("-inf")):
                        return "nan"
                    return str(v)
                _w.writerow([
                    self._scene_name,
                    str(self._pair_counter + 1),
                    str(opt_diag.get("from_idx", -1)) if opt_diag else "-1",
                    str(opt_diag.get("frame_idx", -1)) if opt_diag else "-1",
                    str(frame0.timestamp_ns) if frame0 is not None and hasattr(frame0, "timestamp_ns") else "",
                    str(frame1.timestamp_ns) if frame1 is not None and hasattr(frame1, "timestamp_ns") else "",
                    "v3b", d.mode, d.previous_mode, d.state_name,
                    "1" if d.use_imu_rotation else "0",
                    "1" if d.use_imu_translation else "0",
                    self.imu_translation_semantics,
                    d.reason,
                    str(d.num_visual_residuals),
                    _fmt(d.median_flow_cov),
                    _fmt(d.r_R_whitened_norm),
                    _fmt(d.r_p_whitened_norm),
                    _fmt(d.imu_trans_loss),
                    _fmt(d.visual_loss_per_residual),
                    _fmt(d.total_loss),
                    "1" if d.visual_collapse_raw else "0",
                    "1" if d.visual_collapse_triggered else "0",
                    str(d.visual_collapse_counter),
                    "1" if d.full_divergence_raw else "0",
                    "1" if d.full_divergence_triggered else "0",
                    str(d.full_divergence_counter),
                    "1" if d.rot_harm_raw else "0",
                    "1" if d.rot_harm_triggered else "0",
                    str(d.rot_harm_counter),
                    str(d.rot_harm_cooldown_remaining),
                    str(d.full_divergence_cooldown_remaining),
                    str(d.probation_counter),
                    d.cooldown_reason,
                    str(d.visual_collapse_sustain_config),
                    d.vc_mode,
                    d.visual_collapse_trigger_source,
                    d.visual_collapse_reason,
                    "1" if d.severe_vc_raw else "0",
                    str(d.severe_vc_counter),
                    "1" if d.severe_vc_triggered else "0",
                    str(d.severe_vc_threshold),
                    str(d.severe_vc_sustain_config),
                    "1" if d.mild_vc_raw else "0",
                    str(d.mild_vc_counter),
                    "1" if d.mild_vc_triggered else "0",
                    str(d.mild_vc_threshold),
                    str(d.mild_vc_sustain_config),
                    "1" if d.velocity_reset_enabled else "0",
                    "1" if d.velocity_reset_triggered else "0",
                    d.velocity_reset_strategy,
                    _fmt(d.velocity_before_reset_norm),
                    _fmt(d.velocity_after_reset_norm),
                    str(d.velocity_reset_pair),
                    d.velocity_reset_reason,
                    d.mode_transition_caused_velocity_reset,
                    "1" if d.d2_rerun_enabled else "0",
                    "1" if d.d2_rerun_triggered else "0",
                    str(d.d2_rerun_pair_id),
                    d.d2_rerun_reason,
                    d.d2_committed_result_source,
                    d.d2_pre_rerun_mode,
                    d.d2_post_rerun_mode,
                    _fmt(d.d2_pre_rerun_est_delta_t_norm),
                    _fmt(d.d2_post_rerun_est_delta_t_norm),
                    _fmt(d.d2_pre_rerun_r_p_whitened_norm),
                    _fmt(d.d2_post_rerun_r_p_whitened_norm),
                    str(d.d2_pre_rerun_num_vis_res),
                    str(d.d2_post_rerun_num_vis_res),
                    "1" if d.d2_rerun_failed else "0",
                    d.d2_rerun_failure_reason,
                    # ── FD-E fields (V3b++ Phase 1a) ──────────────────────
                    "1" if d.fd_grace_enabled else "0",
                    str(d.fd_grace_period_config),
                    str(d.full_imu_episode_frame_idx),
                    "1" if d.fd_check_suppressed_by_grace else "0",
                    str(d.fd_grace_remaining),
                    d.full_divergence_reason,
                    # ── Cooldown config (V3b++ Phase 1b) ──────────────────
                    str(d.fd_cooldown_config),
                    str(d.rot_harm_cooldown_config),
                    "1" if d.cooldown_active else "0",
                ])
            elif version == "v3a":
                d = decision
                def _fmt(v):
                    if isinstance(v, float) and (v != v or v == float("inf") or v == float("-inf")):
                        return "nan"
                    return str(v)
                _w.writerow([
                    self._scene_name,
                    str(opt_diag.get("from_idx", -1)) if opt_diag else "-1",
                    str(opt_diag.get("frame_idx", -1)) if opt_diag else "-1",
                    str(frame0.timestamp_ns) if frame0 is not None and hasattr(frame0, "timestamp_ns") else "",
                    str(frame1.timestamp_ns) if frame1 is not None and hasattr(frame1, "timestamp_ns") else "",
                    "v3a", d.mode,
                    "1" if d.use_imu_rotation else "0",
                    "1" if d.use_imu_translation else "0",
                    d.reason,
                    _fmt(d.median_flow_cov),
                    _fmt(d.running_median_flow_cov),
                    _fmt(d.last_valid_flow_cov),
                    _fmt(d.last_valid_running_median_flow_cov),
                    str(d.flow_cov_window_size),
                    str(d.flow_cov_enter_full_threshold),
                    str(d.flow_cov_exit_full_threshold),
                    "1" if d.cov_missing else "0",
                    str(d.cov_missing_counter),
                    str(d.max_cov_bridge_frames),
                    "1" if d.cov_missing_bridge_active else "0",
                    str(d.num_visual_residuals),
                    str(d.num_selected_keypoints),
                    "1" if d.visual_lost_track else "0",
                    str(d.lost_track_counter),
                    str(d.lost_track_enter_frames),
                    "1" if sig.get("optimizer_skipped", False) else "0",
                    str(sig.get("match_idx_size_after_filter", 0)),
                    str(sig.get("num_keypoints_candidate", 0)),
                    str(sig.get("min_num_point", 10)),
                    str(d.stable_visual_counter),
                    str(d.stable_visual_exit_frames),
                    str(d.full_hold_counter),
                    str(d.current_mode_duration),
                    _fmt(sig.get("median_depth_cov", float("nan"))),
                    _fmt(sig.get("p90_depth_cov", float("nan"))),
                    _fmt(sig.get("visual_loss_per_residual", float("nan"))),
                    _fmt(sig.get("r_R_whitened_norm", float("nan"))),
                    _fmt(sig.get("r_p_whitened_norm", float("nan"))),
                    _fmt(sig.get("imu_rot_loss", float("nan"))),
                    _fmt(sig.get("imu_trans_loss", float("nan"))),
                    _fmt(sig.get("est_delta_t_norm", float("nan"))),
                    "1" if d.fallback_triggered else "0",
                    d.hysteresis_state,
                ])
            elif version == "v2":
                gs = getattr(self._adaptive_gate, "state", None)
                _w.writerow([
                    self._scene_name,
                    str(opt_diag.get("from_idx", -1)) if opt_diag else "-1",
                    str(opt_diag.get("frame_idx", -1)) if opt_diag else "-1",
                    "v2", decision.mode,
                    "1" if decision.use_imu_rotation else "0",
                    "1" if decision.use_imu_translation else "0",
                    sig.get("median_flow_cov", "nan"),
                    sig.get("p90_flow_u_cov", "nan"),
                    sig.get("mean_flow_u_cov", "nan"),
                    sig.get("median_depth_cov", "nan"),
                    sig.get("p90_depth_cov", "nan"),
                    sig.get("visual_loss_raw_sum", "nan"),
                    sig.get("visual_loss", "nan"),
                    sig.get("num_selected_keypoints", 0),
                    sig.get("num_visual_residuals", 0),
                    str(getattr(gs, "cov_degraded_counter", "0")),
                    str(getattr(gs, "cov_recovery_counter", "0")),
                    str(getattr(gs, "translation_hold_counter", "0")),
                    signals.get("r_R_whitened_norm", "nan"),
                    signals.get("r_p_whitened_norm", "nan"),
                    decision.reason, decision.hysteresis_state
                ])
            else:
                _w.writerow([
                    self._scene_name,
                    str(opt_diag.get("from_idx", -1)) if opt_diag else "-1",
                    str(opt_diag.get("frame_idx", -1)) if opt_diag else "-1",
                    "", "", decision.mode,
                    "1" if decision.use_imu_rotation else "0",
                    "1" if decision.use_imu_translation else "0",
                    f"{decision.visual_health_score:.4f}",
                    f"{decision.degeneracy_score:.4f}",
                    f"{decision.motion_abnormal_score:.4f}",
                    signals.get("num_visual_residuals", 0),
                    signals.get("num_valid_points", 0),
                    "nan",
                    str(signals.get("est_delta_t_norm", "nan")),
                    str(signals.get("est_delta_R_angle", "nan")),
                    str(signals.get("r_R_whitened_norm", "nan")),
                    str(signals.get("r_p_whitened_norm", "nan")),
                    decision.reason, decision.hysteresis_state
                ])
            _f.flush()
        except Exception as e:
            import traceback
            Logger.write("warn", f"Adaptive decision CSV write error: {e}\n{traceback.format_exc()}")

    def _write_frame_pair_diagnostics(self, frame0: T_SensorFrame, frame1: T_SensorFrame) -> None:
        """Best-effort: write one row to frame_pair_diagnostics.csv for the pair
        that just finished optimization (the PREVIOUS pair, not frame0->frame1)."""
        if self._diag_writer is None:
            return

        opt_diag = getattr(self.Optimizer, "last_pair_diagnostics", None)
        if opt_diag is None:
            return  # no optimization has finished yet

        from_idx = opt_diag.get("from_idx", -1)
        to_idx = opt_diag.get("frame_idx", -1)
        if from_idx < 0 or to_idx < 0 or to_idx >= len(self.graph.frames):
            return
        if to_idx == self._last_written_frame_idx:
            return  # already wrote diagnostics for this pair
        self._last_written_frame_idx = to_idx

        imu_diag = self._pending_imu_diag  # IMU diag for the pair that just finished

        try:
            pose_i = pp.SE3(self.graph.frames.data["pose"][from_idx].float())
            pose_j = pp.SE3(self.graph.frames.data["pose"][to_idx].float())
            ts_i = int(self.graph.frames.data["time_ns"][from_idx].item())
            ts_j = int(self.graph.frames.data["time_ns"][to_idx].item())
        except Exception:
            return

        # ── Estimated inter-frame motion ──────────────────────────────────
        rel_pose = pose_i.Inv() @ pose_j
        delta_t = rel_pose.translation()
        delta_R = rel_pose.rotation()
        est_delta_x = float(delta_t[0].item())
        est_delta_y = float(delta_t[1].item())
        est_delta_z = float(delta_t[2].item())
        est_delta_t_norm = float(delta_t.norm().item())
        est_delta_R_angle = float(delta_R.Log().tensor().norm().item())
        imu_est_delta_t = delta_t
        imu_est_delta_t_norm = est_delta_t_norm
        try:
            sensor_T_imu = pp.SE3(self.graph.frames.data["imu_vio_sensor_T_imu"][from_idx].float()).to(
                device=pose_i.tensor().device,
                dtype=pose_i.tensor().dtype,
            )
            rel_pose_imu = (pose_i @ sensor_T_imu).Inv() @ (pose_j @ sensor_T_imu)
            imu_est_delta_t = rel_pose_imu.translation().reshape(3)
            imu_est_delta_t_norm = float(imu_est_delta_t.norm().item())
        except Exception:
            pass

        # ── Switches ──────────────────────────────────────────────────────
        # Optimizer state at graph construction, before LM. The configured
        # method may already use IMU attitude initialization, so this is not
        # labelled as a pure visual pose.
        init_delta_x = float("nan")
        init_delta_y = float("nan")
        init_delta_z = float("nan")
        init_delta_t_norm = float("nan")
        init_delta_R_angle = float("nan")
        init_delta_t: torch.Tensor | None = None
        init_delta_R: pp.LieTensor | None = None
        initial_motion = opt_diag.get("initial_motion")
        if isinstance(initial_motion, torch.Tensor) and initial_motion.numel() == 7:
            try:
                init_pose_j = pp.SE3(initial_motion.reshape(1, 7).float())
                init_rel_pose = pose_i.Inv() @ init_pose_j
                init_delta_t = init_rel_pose.translation().reshape(3)
                init_delta_R = init_rel_pose.rotation()
                init_delta_x = float(init_delta_t[0].item())
                init_delta_y = float(init_delta_t[1].item())
                init_delta_z = float(init_delta_t[2].item())
                init_delta_t_norm = float(init_delta_t.norm().item())
                init_delta_R_angle = float(init_delta_R.Log().tensor().norm().item())
            except Exception:
                pass
        def _optional_diag_float(name: str) -> float:
            value = opt_diag.get(name)
            return float(value) if value is not None else float("nan")

        if init_delta_t is None:
            init_delta_x = _optional_diag_float("init_delta_x")
            init_delta_y = _optional_diag_float("init_delta_y")
            init_delta_z = _optional_diag_float("init_delta_z")
            init_delta_t_norm = _optional_diag_float("init_delta_t_norm")
            init_delta_R_angle = _optional_diag_float("init_delta_R_angle")

        init_velocity_j = torch.tensor([
            _optional_diag_float("init_velocity_j_x"),
            _optional_diag_float("init_velocity_j_y"),
            _optional_diag_float("init_velocity_j_z"),
        ], dtype=torch.float32)
        est_velocity_j = torch.full((3,), float("nan"), dtype=torch.float32)
        try:
            est_velocity_j = (
                self.graph.frames.data["imu_vio_velocity_world"][to_idx]
                .detach().cpu().float().reshape(3)
            )
        except Exception:
            pass

        use_imu_rotation = bool(opt_diag.get("use_imu_rotation", getattr(self, "imu_rot_prior_enable", False)))
        use_imu_translation = bool(opt_diag.get("use_imu_translation", getattr(self, "imu_trans_prior_enable", False)))
        autodiff_enabled = bool(getattr(self.Optimizer.config, "autodiff", False))
        imu_factor_mode = str(opt_diag.get("imu_factor_mode", "legacy_pose_prior"))
        vio_factor_active = bool(opt_diag.get("vio_factor_active", False))
        imu_residual_rows = int(opt_diag.get("imu_residual_rows", 0))
        imu_diag = self._pending_imu_diag  # IMU diag for the pair that just finished
        internal_world_frame = str(
            (imu_diag or {}).get(
                "imu_internal_world_frame",
                (imu_diag or {}).get("imu_world_frame", getattr(frame1, "imu_internal_world_frame", getattr(frame1, "imu_world_frame", "NED"))),
            )
        ).upper()
        source_measurement_frame = str(
            (imu_diag or {}).get("imu_source_measurement_frame", getattr(frame1, "imu_source_measurement_frame", ""))
        ).upper()
        source_world_frame = str(
            (imu_diag or {}).get("imu_source_world_frame", getattr(frame1, "imu_source_world_frame", ""))
        ).upper()
        internal_measurement_frame = str(
            (imu_diag or {}).get(
                "imu_internal_measurement_frame",
                getattr(frame1, "imu_internal_measurement_frame", internal_world_frame),
            )
        ).upper()
        imu_gravity_source = str((imu_diag or {}).get("imu_gravity_source", getattr(frame1, "imu_gravity_source", "")))
        imu_metadata_gravity_m_s2 = (imu_diag or {}).get(
            "imu_metadata_gravity_m_s2",
            getattr(frame1, "imu_metadata_gravity_m_s2", None),
        )
        imu_preintegration_gravity_z = (imu_diag or {}).get("preintegration_gravity_z", float("nan"))
        imu_vio_gravity_pose_source = str((imu_diag or {}).get(
            "gravity_pose_source",
            getattr(self, "imu_vio_gravity_pose_source", ""),
        ))
        imu_vio_gravity_handling = str((imu_diag or {}).get(
            "gravity_handling",
            getattr(self, "imu_vio_gravity_handling", "preintegration"),
        ))
        stored_gravity_world = torch.as_tensor(
            (imu_diag or {}).get("gravity_world", [0.0, 0.0, imu_preintegration_gravity_z]),
            dtype=torch.float32,
        ).reshape(3)
        imu_gravity_rp_active = int(bool((imu_diag or {}).get("gravity_rp_active", False)))
        imu_gravity_rp_angle_rad = float((imu_diag or {}).get("gravity_rp_angle_rad", float("nan")))
        imu_gravity_rp_acc_norm = float((imu_diag or {}).get("gravity_rp_acc_norm", float("nan")))
        imu_attitude_source_active = int(bool((imu_diag or {}).get("attitude_source_active", False)))
        imu_attitude_source_angle_to_est_rad = float(
            (imu_diag or {}).get("attitude_source_angle_to_est_rad", float("nan"))
        )
        visual_input_sha = str((self._pending_tracking_state or {}).get("visual_input_sha256", ""))

        def _nwu_vector_to_internal(vec_nwu: torch.Tensor) -> torch.Tensor:
            return world_nwu_vector_to_internal(
                vec_nwu.float(),
                internal_world_frame=internal_world_frame,
            )

        def _nwu_rotation_to_internal(rotation_nwu: pp.LieTensor) -> pp.LieTensor:
            if "NED" in internal_world_frame:
                nwu_to_ned = torch.diag(torch.tensor([1.0, -1.0, -1.0], dtype=torch.float32))
                rot_mat = rotation_nwu.matrix().float().reshape(3, 3)
                return pp.from_matrix((nwu_to_ned @ rot_mat @ nwu_to_ned).reshape(1, 3, 3), pp.SO3_type).float()
            return rotation_nwu.float()

        # ── GT inter-frame motion (from ref_pose.csv if available) ────────
        gt_delta_t_norm = float("nan")
        gt_delta_R_angle = float("nan")
        est_over_gt_translation_ratio = float("nan")
        cos_est_gt_translation = float("nan")
        rotation_error_angle = float("nan")
        init_over_gt_translation_ratio = float("nan")
        cos_init_gt_translation = float("nan")
        init_rotation_error_angle = float("nan")
        imu_rotation_error_angle = float("nan")
        velocity_reference_point = ""
        gt_velocity_i = torch.full((3,), float("nan"), dtype=torch.float32)
        gt_velocity_j = torch.full((3,), float("nan"), dtype=torch.float32)
        gt_delta_velocity_norm = float("nan")
        init_velocity_error_norm = float("nan")
        est_velocity_error_norm = float("nan")
        gt_delta_R: pp.LieTensor | None = None
        gt_delta_R_imu: pp.LieTensor | None = None
        d_gt_camera_body: torch.Tensor | None = None  # GT camera-origin delta in body_i frame
        d_gt_imu_body: torch.Tensor | None = None     # GT IMU-origin delta in body_i frame

        if self._gt_positions is not None:
            gt_i = self._gt_positions.get(ts_i)
            gt_j = self._gt_positions.get(ts_j)
            if gt_i is not None and gt_j is not None:
                # GT is in NWU world frame: (x, y, z)_nwu
                gt_i_t = torch.tensor(gt_i, dtype=torch.float32)
                gt_j_t = torch.tensor(gt_j, dtype=torch.float32)
                d_gt_nwu = gt_j_t - gt_i_t  # camera-origin world delta in NWU

                gt_delta_t_norm = float(d_gt_nwu.norm().item())
                d_gt_internal = _nwu_vector_to_internal(d_gt_nwu)
                R_gt_i: pp.LieTensor | None = None
                R_gt_j: pp.LieTensor | None = None

                # ── GT rotation (if quaternion available) ──────────────
                if self._gt_quaternions is not None:
                    q_i = self._gt_quaternions.get(ts_i)
                    q_j = self._gt_quaternions.get(ts_j)
                    if q_i is not None and q_j is not None:
                        # Normalize quaternions
                        def _norm_quat(qx, qy, qz, qw):
                            n = (qx**2 + qy**2 + qz**2 + qw**2) ** 0.5
                            if n < 1e-12:
                                return (0.0, 0.0, 0.0, 1.0)
                            return (qx/n, qy/n, qz/n, qw/n)
                        qi_n = _norm_quat(*q_i)
                        qj_n = _norm_quat(*q_j)
                        R_gt_i = nwu_xyzw_quaternion_to_internal_so3(
                            qi_n,
                            internal_world_frame=internal_world_frame,
                        )
                        R_gt_j = nwu_xyzw_quaternion_to_internal_so3(
                            qj_n,
                            internal_world_frame=internal_world_frame,
                        )
                        # GT relative rotation: R_gt_i^T @ R_gt_j in MACVO internal frame
                        gt_delta_R = R_gt_i.Inv() @ R_gt_j
                        try:
                            rotation_CI = pp.SE3(
                                self.graph.frames.data["imu_vio_sensor_T_imu"][from_idx].float()
                            ).rotation()
                            gt_delta_R_imu = rotation_CI.Inv() @ gt_delta_R @ rotation_CI
                        except Exception:
                            gt_delta_R_imu = None
                        gt_delta_R_angle = float(gt_delta_R.Log().tensor().norm().item())
                        rot_err = gt_delta_R.Inv() @ delta_R
                        rotation_error_angle = float(rot_err.Log().tensor().norm().item())

                # Rotate to body_i frame: R_i^T @ d_gt_internal.
                R_i = pose_i.rotation().matrix().float().squeeze(0)
                d_gt_camera_body = R_i.T @ d_gt_internal
                if est_delta_t_norm > 1e-9 and gt_delta_t_norm > 1e-9:
                    est_over_gt_translation_ratio = est_delta_t_norm / gt_delta_t_norm
                    cos_est_gt_translation = float(
                        torch.dot(delta_t, d_gt_camera_body).item()
                        / max(est_delta_t_norm * float(d_gt_camera_body.norm().item()), 1e-9)
                    )
                if init_delta_t is not None and init_delta_t_norm > 1e-9 and gt_delta_t_norm > 1e-9:
                    init_over_gt_translation_ratio = init_delta_t_norm / gt_delta_t_norm
                    cos_init_gt_translation = float(
                        torch.dot(init_delta_t, d_gt_camera_body).item()
                        / max(init_delta_t_norm * float(d_gt_camera_body.norm().item()), 1e-9)
                    )

                if gt_delta_R is not None and init_delta_R is not None:
                    init_rot_err = gt_delta_R.Inv() @ init_delta_R
                    init_rotation_error_angle = float(
                        init_rot_err.Log().tensor().norm().item()
                    )

                if (
                    self._gt_velocities is not None
                    and self._gt_angular_velocities is not None
                    and R_gt_i is not None
                    and R_gt_j is not None
                ):
                    velocity_i_nwu = self._gt_velocities.get(ts_i)
                    velocity_j_nwu = self._gt_velocities.get(ts_j)
                    angular_i_nwu = self._gt_angular_velocities.get(ts_i)
                    angular_j_nwu = self._gt_angular_velocities.get(ts_j)
                    if (
                        velocity_i_nwu is not None
                        and velocity_j_nwu is not None
                        and angular_i_nwu is not None
                        and angular_j_nwu is not None
                    ):
                        try:
                            camera_to_imu_body = pp.SE3(
                                self.graph.frames.data["imu_vio_sensor_T_imu"][from_idx].float()
                            ).translation().reshape(3)
                        except Exception:
                            camera_to_imu_body = torch.zeros(3, dtype=torch.float32)
                        gt_velocity_i = camera_velocity_to_imu_origin(
                            camera_velocity_world_nwu=torch.tensor(velocity_i_nwu, dtype=torch.float32),
                            angular_velocity_body_nwu=torch.tensor(angular_i_nwu, dtype=torch.float32),
                            camera_to_imu_body_internal=camera_to_imu_body,
                            camera_rotation_body_to_world_internal=R_gt_i.matrix().float(),
                            internal_world_frame=internal_world_frame,
                        )
                        gt_velocity_j = camera_velocity_to_imu_origin(
                            camera_velocity_world_nwu=torch.tensor(velocity_j_nwu, dtype=torch.float32),
                            angular_velocity_body_nwu=torch.tensor(angular_j_nwu, dtype=torch.float32),
                            camera_to_imu_body_internal=camera_to_imu_body,
                            camera_rotation_body_to_world_internal=R_gt_j.matrix().float(),
                            internal_world_frame=internal_world_frame,
                        )
                        velocity_reference_point = "IMUSocket"
                        gt_delta_velocity_norm = float((gt_velocity_j - gt_velocity_i).norm().item())
                        if torch.isfinite(init_velocity_j).all():
                            init_velocity_error_norm = float((init_velocity_j - gt_velocity_j).norm().item())
                        if torch.isfinite(est_velocity_j).all():
                            est_velocity_error_norm = float((est_velocity_j - gt_velocity_j).norm().item())

                if R_gt_i is not None and R_gt_j is not None:
                    try:
                        sensor_T_imu = pp.SE3(self.graph.frames.data["imu_vio_sensor_T_imu"][from_idx].float())
                        camera_to_imu_body = sensor_T_imu.translation().reshape(3).float()
                        rotation_CI_mat = sensor_T_imu.rotation().matrix().float().reshape(3, 3)
                    except Exception:
                        camera_to_imu_body = torch.zeros(3, dtype=torch.float32)
                        rotation_CI_mat = torch.eye(3, dtype=torch.float32)

                    p_cam_i_internal = _nwu_vector_to_internal(gt_i_t)
                    p_cam_j_internal = _nwu_vector_to_internal(gt_j_t)
                    R_gt_i_mat = R_gt_i.matrix().float().reshape(3, 3)
                    R_gt_j_mat = R_gt_j.matrix().float().reshape(3, 3)
                    p_imu_i_internal = p_cam_i_internal + R_gt_i_mat @ camera_to_imu_body
                    p_imu_j_internal = p_cam_j_internal + R_gt_j_mat @ camera_to_imu_body
                    d_gt_imu_internal = p_imu_j_internal - p_imu_i_internal
                    R_gt_imu_i_mat = R_gt_i_mat @ rotation_CI_mat
                    d_gt_imu_body = R_gt_imu_i_mat.T @ d_gt_imu_internal
                else:
                    # Without GT attitude, the lever-arm correction cannot be formed.
                    # Fall back to the camera-origin delta so legacy datasets still log.
                    d_gt_imu_body = d_gt_camera_body

        # ── IMU preintegration quantities ──────────────────────────────────
        imu_delta_R_angle = float("nan")
        imu_delta_p_x = float("nan")
        imu_delta_p_y = float("nan")
        imu_delta_p_z = float("nan")
        imu_delta_p_norm = float("nan")
        imu_delta_v_norm = float("nan")
        num_imu_samples = 0
        imu_dt = float("nan")
        imu_trans_prior_mode = ""
        imu_trans_prior_std_x = float("nan")
        imu_trans_prior_std_y = float("nan")
        imu_trans_prior_std_z = float("nan")
        imu_noise_source = ""
        delta_p_over_est = float("nan")
        delta_p_over_gt = float("nan")
        cos_delta_p_est = float("nan")
        cos_delta_p_gt = float("nan")

        if imu_diag is not None:
            try:
                dr = imu_diag["delta_R"]
                dp = imu_diag["delta_p"].reshape(3)
                dv = imu_diag["delta_v"].reshape(3)
                imu_delta_R_angle = float(dr.Log().tensor().norm().item())
                if gt_delta_R_imu is not None:
                    imu_rot_err = gt_delta_R_imu.Inv() @ dr
                    imu_rotation_error_angle = float(
                        imu_rot_err.Log().tensor().norm().item()
                    )
                imu_delta_p_x = float(dp[0].item())
                imu_delta_p_y = float(dp[1].item())
                imu_delta_p_z = float(dp[2].item())
                imu_delta_p_norm = float(dp.norm().item())
                imu_delta_v_norm = float(dv.norm().item())
                num_imu_samples = int(imu_diag.get("num_imu_samples", 0))
                imu_dt = float(imu_diag.get("dt_total", float("nan")))
                imu_trans_prior_mode = str(imu_diag.get("translation_prior_mode", ""))
                imu_noise_source = str(imu_diag.get("noise_source", ""))
                prior_std_diag = imu_diag.get("translation_prior_std_diag")
                if prior_std_diag is not None:
                    prior_std_diag = prior_std_diag.reshape(3)
                    imu_trans_prior_std_x = float(prior_std_diag[0].item())
                    imu_trans_prior_std_y = float(prior_std_diag[1].item())
                    imu_trans_prior_std_z = float(prior_std_diag[2].item())

                # Compare IMU Δp (body frame) with estimated IMU-origin Δp.
                if imu_delta_p_norm > 1e-9 and imu_est_delta_t_norm > 1e-9:
                    delta_p_over_est = imu_delta_p_norm / imu_est_delta_t_norm
                    cos_delta_p_est = float(
                        torch.dot(dp, imu_est_delta_t).item() / max(imu_delta_p_norm * imu_est_delta_t_norm, 1e-9)
                    )
                # Compare IMU Δp (body frame) with GT IMU-origin Δp when GT attitude
                # is available. This avoids comparing an IMU lever-arm trajectory
                # against CameraLeftSocket motion.
                if imu_delta_p_norm > 1e-9 and d_gt_imu_body is not None:
                    gt_body_norm = float(d_gt_imu_body.norm().item())
                    if gt_body_norm > 1e-9:
                        delta_p_over_gt = imu_delta_p_norm / gt_body_norm
                        cos_delta_p_gt = float(
                            torch.dot(dp, d_gt_imu_body).item() / max(imu_delta_p_norm * gt_body_norm, 1e-9)
                        )
            except Exception:
                pass

        # ── Losses ────────────────────────────────────────────────────────
        final_loss = opt_diag.get("final_loss")
        visual_loss = opt_diag.get("visual_loss")
        imu_rot_loss = opt_diag.get("imu_rot_loss")
        imu_trans_loss = opt_diag.get("imu_trans_loss")
        imu_vel_loss = opt_diag.get("imu_vel_loss")
        r_R_wn = opt_diag.get("r_R_whitened_norm")
        r_p_wn = opt_diag.get("r_p_whitened_norm")
        r_v_wn = opt_diag.get("r_v_whitened_norm")
        num_visual_residuals = int(opt_diag.get("num_visual_residuals", 0))
        energy_visual_weighted = opt_diag.get("energy_visual_weighted")
        energy_p_weighted = opt_diag.get("energy_p_weighted")
        energy_v_weighted = opt_diag.get("energy_v_weighted")
        energy_R_weighted = opt_diag.get("energy_R_weighted")
        energy_pv_weighted = opt_diag.get("energy_pv_weighted")
        energy_imu_diag_weighted = opt_diag.get("energy_imu_diag_weighted")
        energy_imu_weighted = opt_diag.get("energy_imu_weighted")
        energy_imu_to_visual_ratio = opt_diag.get("energy_imu_to_visual_ratio")
        energy_pv_to_visual_ratio = opt_diag.get("energy_pv_to_visual_ratio")
        energy_R_to_visual_ratio = opt_diag.get("energy_R_to_visual_ratio")

        # ── Total loss and ratios ─────────────────────────────────────────
        total_loss = float("nan")
        vis_ratio = float("nan")
        rot_ratio = float("nan")
        trans_ratio = float("nan")
        vel_ratio = float("nan")
        if final_loss is not None:
            total_loss = float(final_loss)
        vl = visual_loss if visual_loss is not None else 0.0
        rl = imu_rot_loss if imu_rot_loss is not None else 0.0
        tl = imu_trans_loss if imu_trans_loss is not None else 0.0
        il = imu_vel_loss if imu_vel_loss is not None else 0.0
        raw_total = vl + rl + tl + il
        if raw_total > 1e-12:
            vis_ratio = vl / raw_total
            rot_ratio = rl / raw_total
            trans_ratio = tl / raw_total
            vel_ratio = il / raw_total

        vl_raw = visual_loss if visual_loss is not None else float("nan")
        vl_per = vl_raw / max(num_visual_residuals, 1) if num_visual_residuals > 0 else float("nan")

        # ── Frontend covariance stats ─────────────────────────────────
        cov_diag = self._pending_cov_diag or {}

        # ── Write row ─────────────────────────────────────────────────────
        self._pair_counter += 1
        self._diag_writer.write_row(
            scene=self._scene_name,
            method=self._method_name,
            pair_id=self._pair_counter,
            frame_i=from_idx,
            frame_j=to_idx,
            timestamp_i=ts_i,
            timestamp_j=ts_j,
            dt=(ts_j - ts_i) * 1e-9 if ts_j > ts_i else float("nan"),
            use_imu_rotation=use_imu_rotation,
            use_imu_translation=use_imu_translation,
            autodiff_enabled=autodiff_enabled,
            imu_factor_mode=imu_factor_mode,
            vio_factor_active=1 if vio_factor_active else 0,
            imu_residual_rows=imu_residual_rows,
            local_ba_window_size=opt_diag.get("local_ba_window_size", ""),
            local_ba_writeback=opt_diag.get("local_ba_writeback", ""),
            local_ba_num_frames=opt_diag.get("local_ba_num_frames", ""),
            local_ba_num_edges=opt_diag.get("local_ba_num_edges", ""),
            local_ba_num_visual_residual_blocks=opt_diag.get("local_ba_num_visual_residual_blocks", ""),
            local_ba_graph_build_s=opt_diag.get("local_ba_graph_build_s", ""),
            local_ba_lm_s=opt_diag.get("local_ba_lm_s", ""),
            local_ba_refine_s=opt_diag.get("local_ba_refine_s", ""),
            local_ba_optimize_total_s=opt_diag.get("local_ba_optimize_total_s", ""),
            two_state_solver_converged=opt_diag.get("two_state_solver_converged", ""),
            two_state_solver_iterations=opt_diag.get("two_state_solver_iterations", ""),
            two_state_solver_convergence_reason=opt_diag.get(
                "two_state_solver_convergence_reason", ""
            ),
            two_state_final_step_norm=opt_diag.get("two_state_final_step_norm", ""),
            two_state_final_gradient_inf_norm=opt_diag.get(
                "two_state_final_gradient_inf_norm", ""
            ),
            two_state_solver_accepted_steps=opt_diag.get(
                "two_state_solver_accepted_steps", ""
            ),
            two_state_solver_rejected_steps=opt_diag.get(
                "two_state_solver_rejected_steps", ""
            ),
            vio_backend=opt_diag.get("vio_backend", "two_state"),
            isam2_update_ms=opt_diag.get("isam2_update_ms", ""),
            isam2_state_count=opt_diag.get("isam2_state_count", ""),
            isam2_history_revision=int(bool(opt_diag.get("isam2_history_revision", False))),
            isam2_initial_pose_mismatch_norm=opt_diag.get(
                "isam2_initial_pose_mismatch_norm", ""
            ),
            isam2_initial_velocity_mismatch_norm=opt_diag.get(
                "isam2_initial_velocity_mismatch_norm", ""
            ),
            isam2_initial_bias_mismatch_norm=opt_diag.get(
                "isam2_initial_bias_mismatch_norm", ""
            ),
            est_delta_x=est_delta_x,
            est_delta_y=est_delta_y,
            est_delta_z=est_delta_z,
            est_delta_t_norm=est_delta_t_norm,
            est_delta_R_angle=est_delta_R_angle,
            init_delta_x=opt_diag.get("init_delta_x", init_delta_x),
            init_delta_y=opt_diag.get("init_delta_y", init_delta_y),
            init_delta_z=opt_diag.get("init_delta_z", init_delta_z),
            init_delta_t_norm=opt_diag.get("init_delta_t_norm", init_delta_t_norm),
            init_delta_R_angle=opt_diag.get("init_delta_R_angle", init_delta_R_angle),
            init_velocity_j_x=opt_diag.get("init_velocity_j_x", float(init_velocity_j[0].item())),
            init_velocity_j_y=opt_diag.get("init_velocity_j_y", float(init_velocity_j[1].item())),
            init_velocity_j_z=opt_diag.get("init_velocity_j_z", float(init_velocity_j[2].item())),
            # GT fields
            gt_delta_t_norm=gt_delta_t_norm,
            gt_delta_R_angle=gt_delta_R_angle,
            est_over_gt_translation_ratio=est_over_gt_translation_ratio,
            cos_est_gt_translation=cos_est_gt_translation,
            rotation_error_angle=rotation_error_angle,
            init_over_gt_translation_ratio=init_over_gt_translation_ratio,
            cos_init_gt_translation=cos_init_gt_translation,
            init_rotation_error_angle=init_rotation_error_angle,
            imu_rotation_error_angle=imu_rotation_error_angle,
            velocity_reference_point=velocity_reference_point,
            gt_velocity_i_x=float(gt_velocity_i[0].item()),
            gt_velocity_i_y=float(gt_velocity_i[1].item()),
            gt_velocity_i_z=float(gt_velocity_i[2].item()),
            gt_velocity_j_x=float(gt_velocity_j[0].item()),
            gt_velocity_j_y=float(gt_velocity_j[1].item()),
            gt_velocity_j_z=float(gt_velocity_j[2].item()),
            gt_delta_velocity_norm=gt_delta_velocity_norm,
            est_velocity_j_x=float(est_velocity_j[0].item()),
            est_velocity_j_y=float(est_velocity_j[1].item()),
            est_velocity_j_z=float(est_velocity_j[2].item()),
            init_velocity_error_norm=init_velocity_error_norm,
            est_velocity_error_norm=est_velocity_error_norm,
            # IMU preintegration
            imu_delta_R_angle=imu_delta_R_angle,
            imu_delta_p_x=imu_delta_p_x,
            imu_delta_p_y=imu_delta_p_y,
            imu_delta_p_z=imu_delta_p_z,
            imu_delta_p_norm=imu_delta_p_norm,
            imu_delta_v_norm=imu_delta_v_norm,
            num_imu_samples=num_imu_samples,
            imu_dt=imu_dt,
            imu_trans_prior_mode=imu_trans_prior_mode,
            imu_trans_prior_std_x=imu_trans_prior_std_x,
            imu_trans_prior_std_y=imu_trans_prior_std_y,
            imu_trans_prior_std_z=imu_trans_prior_std_z,
            imu_noise_source=imu_noise_source,
            imu_source_world_frame=source_world_frame,
            imu_source_measurement_frame=source_measurement_frame,
            imu_internal_world_frame=internal_world_frame,
            imu_internal_measurement_frame=internal_measurement_frame,
            imu_gravity_source=imu_gravity_source,
            imu_metadata_gravity_m_s2=imu_metadata_gravity_m_s2,
            imu_preintegration_gravity_z=imu_preintegration_gravity_z,
            imu_vio_gravity_pose_source=imu_vio_gravity_pose_source,
            imu_vio_gravity_handling=imu_vio_gravity_handling,
            imu_vio_gravity_world_x=float(stored_gravity_world[0].item()),
            imu_vio_gravity_world_y=float(stored_gravity_world[1].item()),
            imu_vio_gravity_world_z=float(stored_gravity_world[2].item()),
            imu_attitude_source_active=imu_attitude_source_active,
            imu_attitude_source_angle_to_est_rad=imu_attitude_source_angle_to_est_rad,
            imu_gravity_rp_active=imu_gravity_rp_active,
            imu_gravity_rp_angle_rad=imu_gravity_rp_angle_rad,
            imu_gravity_rp_acc_norm=imu_gravity_rp_acc_norm,
            delta_p_over_est=delta_p_over_est,
            delta_p_over_gt=delta_p_over_gt,
            cos_delta_p_est=cos_delta_p_est,
            cos_delta_p_gt=cos_delta_p_gt,
            # Visual stats
            num_valid_points=int(opt_diag.get("num_observations", 0)),
            num_visual_residuals=num_visual_residuals,
            median_flow_cov=cov_diag.get("median_flow_u_cov", float("nan")),
            median_depth_cov=cov_diag.get("median_kp0_depth_cov", float("nan")),
            visual_input_sha256=visual_input_sha,
            visual_loss_raw_sum=vl_raw,
            visual_loss_per_residual=vl_per,
            # Frontend covariance stats
            **{k: cov_diag.get(k, float("nan")) for k in [
                "median_flow_u_cov", "p90_flow_u_cov", "mean_flow_u_cov",
                "median_flow_v_cov", "p90_flow_v_cov", "mean_flow_v_cov",
                "median_kp0_depth_cov", "p90_kp0_depth_cov", "mean_kp0_depth_cov",
                "median_kp1_depth_cov", "p90_kp1_depth_cov", "mean_kp1_depth_cov",
                "valid_depth_ratio", "num_selected_keypoints",
            ] if k not in ["median_flow_cov", "median_depth_cov"]},
            # IMU residual / loss
            r_R_whitened_norm=r_R_wn if r_R_wn is not None else float("nan"),
            r_p_whitened_norm=r_p_wn if r_p_wn is not None else float("nan"),
            r_v_whitened_norm=r_v_wn if r_v_wn is not None else float("nan"),
            imu_vio_whitened_norm=opt_diag.get("imu_vio_whitened_norm", float("nan")),
            imu_vio_raw_norm=opt_diag.get("imu_vio_raw_norm", float("nan")),
            imu_rot_loss=imu_rot_loss if imu_rot_loss is not None else float("nan"),
            imu_trans_loss=imu_trans_loss if imu_trans_loss is not None else float("nan"),
            imu_vel_loss=imu_vel_loss if imu_vel_loss is not None else float("nan"),
            imu_vio_cov_trace=opt_diag.get("imu_vio_cov_trace", float("nan")),
            imu_vio_weight_trace=opt_diag.get("imu_vio_weight_trace", float("nan")),
            imu_vio_weight_diag_min=opt_diag.get("imu_vio_weight_diag_min", float("nan")),
            imu_vio_weight_diag_max=opt_diag.get("imu_vio_weight_diag_max", float("nan")),
            imu_vio_sa_v2_sampling_noise_cost=opt_diag.get(
                "imu_vio_sa_v2_sampling_noise_cost", float("nan")
            ),
            imu_vio_sa_v2_cross_covariance_frobenius_norm=opt_diag.get(
                "imu_vio_sa_v2_cross_covariance_frobenius_norm", float("nan")
            ),
            imu_vio_sa_v2_incoming_sample_count=opt_diag.get(
                "imu_vio_sa_v2_incoming_sample_count", float("nan")
            ),
            imu_vio_sa_v2_outgoing_sample_count=opt_diag.get(
                "imu_vio_sa_v2_outgoing_sample_count", float("nan")
            ),
            **{
                name: opt_diag.get(name, float("nan"))
                for name in (
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
                )
            },
            imu_vio_acc_bias_norm=opt_diag.get("imu_vio_acc_bias_norm", float("nan")),
            imu_vio_gyro_bias_norm=opt_diag.get("imu_vio_gyro_bias_norm", float("nan")),
            imu_vio_acc_bias_x=opt_diag.get("imu_vio_acc_bias_x", float("nan")),
            imu_vio_acc_bias_y=opt_diag.get("imu_vio_acc_bias_y", float("nan")),
            imu_vio_acc_bias_z=opt_diag.get("imu_vio_acc_bias_z", float("nan")),
            imu_vio_gyro_bias_x=opt_diag.get("imu_vio_gyro_bias_x", float("nan")),
            imu_vio_gyro_bias_y=opt_diag.get("imu_vio_gyro_bias_y", float("nan")),
            imu_vio_gyro_bias_z=opt_diag.get("imu_vio_gyro_bias_z", float("nan")),
            imu_vio_alpha_p=opt_diag.get("imu_vio_alpha_p", float("nan")),
            imu_vio_alpha_v=opt_diag.get("imu_vio_alpha_v", float("nan")),
            imu_vio_alpha_R=opt_diag.get("imu_vio_alpha_R", float("nan")),
            initial_energy_visual_weighted=opt_diag.get("initial_energy_visual_weighted", float("nan")),
            initial_energy_p_weighted=opt_diag.get("initial_energy_p_weighted", float("nan")),
            initial_energy_v_weighted=opt_diag.get("initial_energy_v_weighted", float("nan")),
            initial_energy_R_weighted=opt_diag.get("initial_energy_R_weighted", float("nan")),
            initial_energy_pv_weighted=opt_diag.get("initial_energy_pv_weighted", float("nan")),
            initial_energy_imu_diag_weighted=opt_diag.get("initial_energy_imu_diag_weighted", float("nan")),
            initial_energy_imu_weighted=opt_diag.get("initial_energy_imu_weighted", float("nan")),
            initial_energy_imu_to_visual_ratio=opt_diag.get("initial_energy_imu_to_visual_ratio", float("nan")),
            initial_energy_pv_to_visual_ratio=opt_diag.get("initial_energy_pv_to_visual_ratio", float("nan")),
            initial_energy_R_to_visual_ratio=opt_diag.get("initial_energy_R_to_visual_ratio", float("nan")),
            initial_total_loss=opt_diag.get("initial_total_loss", float("nan")),
            update_pose_translation_norm=opt_diag.get("update_pose_translation_norm", float("nan")),
            update_pose_rotation_norm=opt_diag.get("update_pose_rotation_norm", float("nan")),
            update_velocity_norm=opt_diag.get("update_velocity_norm", float("nan")),
            update_acc_bias_norm=opt_diag.get("update_acc_bias_norm", float("nan")),
            update_gyro_bias_norm=opt_diag.get("update_gyro_bias_norm", float("nan")),
            influence_visual_grad_norm=opt_diag.get("influence_visual_grad_norm", float("nan")),
            influence_imu_grad_norm=opt_diag.get("influence_imu_grad_norm", float("nan")),
            influence_grad_cosine=opt_diag.get("influence_grad_cosine", float("nan")),
            influence_visual_hessian_trace=opt_diag.get("influence_visual_hessian_trace", float("nan")),
            influence_imu_hessian_trace=opt_diag.get("influence_imu_hessian_trace", float("nan")),
            influence_imu_to_visual_grad_ratio=opt_diag.get("influence_imu_to_visual_grad_ratio", float("nan")),
            influence_imu_to_visual_hessian_ratio=opt_diag.get("influence_imu_to_visual_hessian_ratio", float("nan")),
            influence_p_grad_norm=opt_diag.get("influence_p_grad_norm", float("nan")),
            influence_v_grad_norm=opt_diag.get("influence_v_grad_norm", float("nan")),
            influence_R_grad_norm=opt_diag.get("influence_R_grad_norm", float("nan")),
            influence_sampled=opt_diag.get("influence_sampled", 0),
            energy_visual_change=opt_diag.get("energy_visual_change", float("nan")),
            energy_imu_change=opt_diag.get("energy_imu_change", float("nan")),
            energy_p_change=opt_diag.get("energy_p_change", float("nan")),
            energy_v_change=opt_diag.get("energy_v_change", float("nan")),
            energy_R_change=opt_diag.get("energy_R_change", float("nan")),
            counterfactual_visual_step_norm=opt_diag.get("counterfactual_visual_step_norm", float("nan")),
            counterfactual_imu_step_norm=opt_diag.get("counterfactual_imu_step_norm", float("nan")),
            counterfactual_full_step_norm=opt_diag.get("counterfactual_full_step_norm", float("nan")),
            counterfactual_visual_to_imu_cosine=opt_diag.get("counterfactual_visual_to_imu_cosine", float("nan")),
            actual_to_visual_step_cosine=opt_diag.get("actual_to_visual_step_cosine", float("nan")),
            actual_to_imu_step_cosine=opt_diag.get("actual_to_imu_step_cosine", float("nan")),
            actual_to_full_step_cosine=opt_diag.get("actual_to_full_step_cosine", float("nan")),
            predicted_visual_change_on_actual_step=opt_diag.get("predicted_visual_change_on_actual_step", float("nan")),
            predicted_imu_change_on_actual_step=opt_diag.get("predicted_imu_change_on_actual_step", float("nan")),
            energy_visual_weighted=energy_visual_weighted if energy_visual_weighted is not None else float("nan"),
            energy_p_weighted=energy_p_weighted if energy_p_weighted is not None else float("nan"),
            energy_v_weighted=energy_v_weighted if energy_v_weighted is not None else float("nan"),
            energy_R_weighted=energy_R_weighted if energy_R_weighted is not None else float("nan"),
            energy_pv_weighted=energy_pv_weighted if energy_pv_weighted is not None else float("nan"),
            energy_imu_diag_weighted=energy_imu_diag_weighted if energy_imu_diag_weighted is not None else float("nan"),
            energy_imu_weighted=energy_imu_weighted if energy_imu_weighted is not None else float("nan"),
            energy_imu_to_visual_ratio=energy_imu_to_visual_ratio if energy_imu_to_visual_ratio is not None else float("nan"),
            energy_pv_to_visual_ratio=energy_pv_to_visual_ratio if energy_pv_to_visual_ratio is not None else float("nan"),
            energy_R_to_visual_ratio=energy_R_to_visual_ratio if energy_R_to_visual_ratio is not None else float("nan"),
            total_loss=total_loss,
            visual_loss_ratio=vis_ratio,
            imu_rot_loss_ratio=rot_ratio,
            imu_trans_loss_ratio=trans_ratio,
            imu_vel_loss_ratio=vel_ratio,
            # Coordinate frame annotations
            est_pose_frame=f"{internal_world_frame}_internal",
            gt_pose_frame="NWU_world",
            imu_meas_frame=(
                f"{source_measurement_frame}->{internal_measurement_frame} via T_BS"
                if source_measurement_frame and internal_measurement_frame and source_measurement_frame != internal_measurement_frame
                else (internal_measurement_frame or internal_world_frame)
            ),
            imu_delta_frame=f"{internal_measurement_frame}_i",
            # Adaptive mode annotations
            adaptive_mode=(self._adaptive_decision.mode if self._adaptive_decision else ""),
            adaptive_use_rotation=(self._adaptive_decision.use_imu_rotation if self._adaptive_decision else ""),
            adaptive_use_translation=(self._adaptive_decision.use_imu_translation if self._adaptive_decision else ""),
            adaptive_reason=(self._adaptive_decision.reason if self._adaptive_decision else ""),
            visual_health_score=(f"{getattr(self._adaptive_decision, 'visual_health_score', float('nan')):.4f}"
                                 if self._adaptive_decision and hasattr(self._adaptive_decision, 'visual_health_score') else ""),
            degeneracy_score=(f"{getattr(self._adaptive_decision, 'degeneracy_score', float('nan')):.4f}"
                              if self._adaptive_decision and hasattr(self._adaptive_decision, 'degeneracy_score') else ""),
            motion_abnormal_score=(f"{getattr(self._adaptive_decision, 'motion_abnormal_score', float('nan')):.4f}"
                                   if self._adaptive_decision and hasattr(self._adaptive_decision, 'motion_abnormal_score') else ""),
        )
        self._diag_writer.flush()
        self._write_optimizer_breakpoint_trace(
            getattr(self.Optimizer, "last_breakpoint_trace", None),
            list(getattr(self.Optimizer, "last_breakpoint_frame_indices", [from_idx, to_idx])),
            self.graph,
        )

    def register_on_optimize_finish(self, func: T_SYSHOOK):
        """
        Install a callback hook when optimization result is written back to the map
        """
        self.on_optimize_writeback.append(func)


def _compute_frontend_cov_stats(depth0_cov, depth1_cov, match01_cov,
                                 kp0_sigma_dd, kp1_sigma_dd,
                                 kp0_sigma_uv, kp1_sigma_uv, num_kp,
                                 depth0_depth=None, depth1_depth=None) -> dict:
    """Compute per-frame-pair frontend covariance statistics for diagnostics.
    All inputs are optional; missing → NaN in output.
    """
    import numpy as np
    stats = {}

    # ── Depth covariance (kp0_sigma_dd is [N] or None) ──────────────
    for name, sigma in [("depth_cov_kp0", kp0_sigma_dd), ("depth_cov_kp1", kp1_sigma_dd)]:
        if sigma is not None:
            arr = sigma.detach().cpu().float().numpy().ravel()
            arr = arr[~np.isnan(arr)]
            if len(arr) > 0:
                prefix = name.replace("depth_cov_", "")
                stats[f"median_{prefix}_depth_cov"] = float(np.median(arr))
                stats[f"p90_{prefix}_depth_cov"] = float(np.percentile(arr, 90))
                stats[f"mean_{prefix}_depth_cov"] = float(np.mean(arr))

    # ── Flow/match covariance (kp1_sigma_uv is [N,3] or None) ──────
    if kp1_sigma_uv is not None:
        arr = kp1_sigma_uv.detach().cpu().float().numpy()
        # columns: [var_u, var_v, cov_uv]
        var_u = arr[:, 0]; var_v = arr[:, 1]
        for name, vec in [("flow_var_u", var_u), ("flow_var_v", var_v)]:
            v = vec[~np.isnan(vec)]
            if len(v) > 0:
                prefix = name.replace("flow_var_", "")
                stats[f"median_flow_{prefix}_cov"] = float(np.median(v))
                stats[f"p90_flow_{prefix}_cov"] = float(np.percentile(v, 90))
                stats[f"mean_flow_{prefix}_cov"] = float(np.mean(v))

    # ── valid depth ratio from depth maps ─────────────────────────
    valid_depth_ratio = float("nan")
    if depth0_depth is not None:
        try:
            d0 = depth0_depth.detach().cpu().float().numpy()
            valid = (~np.isnan(d0)) & (d0 > 0) & (np.isfinite(d0))
            valid_depth_ratio = float(np.mean(valid))
        except Exception:
            pass
    stats["valid_depth_ratio"] = valid_depth_ratio

    # ── Keypoint covariance statistics ──────────────────────────────
    stats["num_selected_keypoints"] = int(num_kp)

    return stats
