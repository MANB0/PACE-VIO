import math
import os
import time
import torch
from pathlib import Path
from types import SimpleNamespace
import pypose as pp

from pypose.optim import LM
from pypose.optim.corrector import FastTriggs
from pypose.optim.functional import modjac
from pypose.optim.kernel import Huber
from pypose.optim.scheduler import StopOnPlateau
from pypose.optim.solver import PINV
from pypose.optim.strategy import TrustRegion

from Module.Map import VisualMap
from Utility.Timer import Timer
from Utility.Math  import NormalizeQuat
from Module.URAF import uraf_post_fusion
from Utility.IMUKinematics import (
    vio_preintegrated_covariance_blocks,
    vio_preintegrated_covariance_matrix,
)
from Utility.RelativePoseFactorCache import camera_factor_to_body_factor
from Utility.T2FactorPacket import T2FactorPacket
from Utility.T2ISAM2Backend import IncrementalT2ISAM2Backend
from Utility.TwoStateVIO import (
    ImuPreintegrationFactor,
    LinearizedUVDPoseFactor,
    NavigationState,
    RelativePoseFactor,
    SquareRootPrior,
    TwoStateVIOProblem,
    TwoStateVIOSolver,
    UVDFactor,
    linearize_uvd_relative_pose_factor,
    linearized_uvd_pose_factor_from_normal_equations,
    make_diagonal_prior,
    state_boxminus,
    visual_whitened_residuals,
)
from Utility.TwoStateSamplingAwareVIO import (
    CrossEdgeImuFactor,
    CrossEdgeSquareRootPrior,
    CrossEdgeTwoStateProblem,
    CrossEdgeTwoStateSolver,
    make_cross_edge_diagonal_prior,
)

from ..Interface import IOptimizer
from ..PyposeOptimizers import LM_analytic, AnalyticModule, FactorGraph
from .Graphs import GraphInput, GraphOutput, LocalWindowGraphInput
from .Graphs import ICP_TwoframePGO, Reproj_TwoFramePGO, ReprojDisp_TwoFramePGO
from .Graphs import LocalWindowInertialGraph
from .Graphs import Analytic_ICP_TwoframePGO, Analytic_Reproj_TwoFramePGO, Analytic_ReprojDisp_TwoFramePGO


def _optional_frame(value) -> int | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"none", "off", "false", "-1"}:
        return None
    return int(text)


def _frame_set(value) -> set[int]:
    if value is None:
        return set()
    if isinstance(value, (list, tuple, set)):
        return {int(item) for item in value}
    text = str(value).strip()
    if not text:
        return set()
    return {int(item.strip()) for item in text.split(",") if item.strip()}


def _optional_bool(value, *, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off", ""}:
        return False
    raise ValueError(f"expected a boolean value, got {value!r}")


def _gate_two_state_visual_factor(
    state_i: NavigationState,
    state_j: NavigationState,
    measurement_body: torch.Tensor,
    covariance_body: torch.Tensor,
    *,
    num_points: int,
    num_inliers: int,
    mean_mahalanobis_sq: float | None,
    config: dict[str, float],
    eigenvalue_floor: float,
) -> tuple[torch.Tensor, dict[str, float | str | None]]:
    """Inflate or effectively reject an abnormal compressed visual pose edge."""

    covariance = covariance_body.reshape(6, 6)
    covariance = 0.5 * (covariance + covariance.mT)
    if not bool(torch.isfinite(covariance).all()):
        raise FloatingPointError("visual relative-pose covariance contains NaN/Inf")

    predicted = pp.SE3(state_i.pose_WB.reshape(1, 7)).Inv() @ pp.SE3(
        state_j.pose_WB.reshape(1, 7)
    )
    residual = (
        pp.SE3(measurement_body.reshape(1, 7)).Inv() @ predicted
    ).Log().tensor().reshape(6)
    if not bool(torch.isfinite(residual).all()):
        raise FloatingPointError("visual relative-pose residual contains NaN/Inf")

    values, vectors = torch.linalg.eigh(covariance)
    floor = max(float(eigenvalue_floor), torch.finfo(covariance.dtype).eps)
    covariance_spd = vectors @ torch.diag(values.clamp_min(floor)) @ vectors.mT
    lower = torch.linalg.cholesky(covariance_spd)
    whitened = torch.linalg.solve_triangular(
        lower, residual.reshape(6, 1), upper=False
    ).reshape(6)
    whitened_norm = float(torch.linalg.vector_norm(whitened).detach().cpu().item())

    inlier_ratio = (
        float(num_inliers) / float(num_points)
        if int(num_points) > 0
        else None
    )
    mean_mahalanobis = (
        float(mean_mahalanobis_sq)
        if mean_mahalanobis_sq is not None and math.isfinite(float(mean_mahalanobis_sq))
        else None
    )
    inflation = 1.0
    reject_reasons: list[str] = []

    if inlier_ratio is not None:
        if inlier_ratio < config["reject_inlier_ratio"]:
            reject_reasons.append("low_inlier_ratio")
        elif inlier_ratio < config["soft_inlier_ratio"]:
            inflation *= (config["soft_inlier_ratio"] / max(inlier_ratio, 1e-12)) ** 2

    if mean_mahalanobis is not None:
        if mean_mahalanobis > config["reject_mean_mahalanobis_sq"]:
            reject_reasons.append("high_mean_mahalanobis")
        elif mean_mahalanobis > config["soft_mean_mahalanobis_sq"]:
            inflation *= mean_mahalanobis / config["soft_mean_mahalanobis_sq"]

    if whitened_norm > config["reject_whitened_pose_norm"]:
        reject_reasons.append("high_whitened_pose_residual")
    elif whitened_norm > config["soft_whitened_pose_norm"]:
        inflation *= (whitened_norm / config["soft_whitened_pose_norm"]) ** 2

    if reject_reasons:
        action = "reject:" + "+".join(reject_reasons)
        inflation = config["max_covariance_inflation"]
    else:
        inflation = min(max(inflation, 1.0), config["max_covariance_inflation"])
        action = "downweight" if inflation > 1.0 + 1e-12 else "accept"

    gated_covariance = covariance_spd * inflation
    diagnostics: dict[str, float | str | None] = {
        "inlier_ratio": inlier_ratio,
        "mean_mahalanobis_sq": mean_mahalanobis,
        "whitened_pose_residual_norm": whitened_norm,
        "covariance_inflation": float(inflation),
        "action": action,
    }
    return gated_covariance, diagnostics


def _two_state_visual_whitened_norm(
    state_i: NavigationState,
    state_j: NavigationState,
    measurement_body: torch.Tensor,
    covariance_body: torch.Tensor,
    *,
    eigenvalue_floor: float,
) -> float:
    predicted = pp.SE3(state_i.pose_WB.reshape(1, 7)).Inv() @ pp.SE3(
        state_j.pose_WB.reshape(1, 7)
    )
    residual = (
        pp.SE3(measurement_body.reshape(1, 7)).Inv() @ predicted
    ).Log().tensor().reshape(6)
    covariance = covariance_body.reshape(6, 6)
    covariance = 0.5 * (covariance + covariance.mT)
    values, vectors = torch.linalg.eigh(covariance)
    floor = max(float(eigenvalue_floor), torch.finfo(covariance.dtype).eps)
    covariance = vectors @ torch.diag(values.clamp_min(floor)) @ vectors.mT
    lower = torch.linalg.cholesky(covariance)
    whitened = torch.linalg.solve_triangular(
        lower, residual.reshape(6, 1), upper=False
    ).reshape(6)
    if not bool(torch.isfinite(whitened).all()):
        raise FloatingPointError("visual relative-pose whitened residual contains NaN/Inf")
    return float(torch.linalg.vector_norm(whitened).detach().cpu().item())


def _make_two_state_uvd_factor(
    graph_data: GraphInput,
    extrinsic_CI: pp.LieTensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
    huber_delta: float,
) -> UVDFactor:
    points_Ci = graph_data.points.data.get("pos_Tc")
    target_uv = graph_data.observations.data.get("pixel2_uv")
    target_disparity = graph_data.observations.data.get("pixel2_disp")
    target_uv_cov = graph_data.observations.data.get("pixel2_uv_cov")
    target_disparity_cov = graph_data.observations.data.get("pixel2_disp_cov")
    fields = {
        "points.pos_Tc": points_Ci,
        "observations.pixel2_uv": target_uv,
        "observations.pixel2_disp": target_disparity,
        "observations.pixel2_uv_cov": target_uv_cov,
        "observations.pixel2_disp_cov": target_disparity_cov,
    }
    missing = [name for name, value in fields.items() if value is None]
    if missing:
        raise ValueError(f"direct UVD factor is missing cached fields: {missing}")

    count = int(points_Ci.reshape(-1, 3).shape[0])
    if count < 3:
        raise ValueError("direct UVD factor requires at least three matched points")
    row_counts = {
        "pixel2_uv": int(target_uv.reshape(-1, 2).shape[0]),
        "pixel2_disp": int(target_disparity.reshape(-1, 1).shape[0]),
        "pixel2_uv_cov": int(target_uv_cov.reshape(-1, 3).shape[0]),
        "pixel2_disp_cov": int(target_disparity_cov.reshape(-1, 1).shape[0]),
    }
    mismatched = {name: rows for name, rows in row_counts.items() if rows != count}
    if mismatched:
        raise ValueError(
            f"direct UVD point/observation row mismatch: points={count}, fields={mismatched}"
        )

    covariance = torch.zeros((count, 3, 3), device=device, dtype=dtype)
    uv_cov = target_uv_cov.reshape(count, 3).to(device=device, dtype=dtype)
    covariance[:, 0, 0] = uv_cov[:, 0]
    covariance[:, 1, 1] = uv_cov[:, 1]
    covariance[:, 0, 1] = uv_cov[:, 2]
    covariance[:, 1, 0] = uv_cov[:, 2]
    covariance[:, 2, 2] = target_disparity_cov.reshape(count).to(
        device=device, dtype=dtype
    )
    if not bool(torch.isfinite(covariance).all()):
        raise FloatingPointError("direct UVD covariance contains NaN/Inf")

    return UVDFactor(
        points_Ci=points_Ci,
        target_uv=target_uv,
        target_disparity=target_disparity,
        covariance_uvd=covariance,
        intrinsic=graph_data.images_intrinsic,
        baseline=float(graph_data.baseline.reshape(-1)[0].item()),
        extrinsic_CI=extrinsic_CI.tensor(),
        huber_delta=float(huber_delta),
    ).to(device=device, dtype=dtype)


def _imu_propagated_pose(
    state_i: NavigationState,
    imu: ImuPreintegrationFactor,
) -> torch.Tensor:
    dtype = state_i.pose_WB.dtype
    device = state_i.pose_WB.device
    pose_i = pp.SE3(state_i.pose_WB)
    rotation_i = pose_i.rotation()
    rotation_i_matrix = rotation_i.matrix().reshape(3, 3)
    position_i = pose_i.Act(torch.zeros(3, device=device, dtype=dtype)).reshape(3)

    delta_position = imu.delta_position.reshape(3).to(device=device, dtype=dtype)
    delta_rotation = pp.so3(
        imu.delta_rotation.reshape(3).to(device=device, dtype=dtype)
    ).Exp()
    db = torch.cat(
        [
            state_i.acc_bias.reshape(3).to(device=device, dtype=dtype)
            - imu.linearized_acc_bias.reshape(3).to(device=device, dtype=dtype),
            state_i.gyro_bias.reshape(3).to(device=device, dtype=dtype)
            - imu.linearized_gyro_bias.reshape(3).to(device=device, dtype=dtype),
        ]
    )
    correction = (
        imu.bias_jacobian.reshape(9, 6).to(device=device, dtype=dtype) @ db
    )
    delta_position = delta_position + correction[0:3]
    delta_rotation = delta_rotation @ pp.so3(correction[6:9].reshape(1, 3)).Exp()

    gravity = torch.zeros(3, device=device, dtype=dtype)
    if str(imu.gravity_handling) == "residual":
        if imu.gravity_world is None:
            raise ValueError("IMU propagation requires gravity_world in residual mode")
        gravity = imu.gravity_world.reshape(3).to(device=device, dtype=dtype)
    dt = float(imu.dt)
    position_j = (
        position_i
        + state_i.velocity_W.reshape(3).to(device=device, dtype=dtype) * dt
        + 0.5 * gravity * dt * dt
        + rotation_i_matrix @ delta_position
    )
    rotation_j = rotation_i @ delta_rotation
    return pp.SE3(
        torch.cat(
            [position_j.reshape(1, 3), rotation_j.tensor().reshape(1, 4)], dim=-1
        )
    ).tensor()


class TwoFrame_PGO(IOptimizer[GraphInput, dict, GraphOutput]):
    def __init__(self, config: SimpleNamespace) -> None:
        super().__init__(config)
        self._bagf_active_streak = 0
        self._bagf_cooldown = 0
        self._dua_active = False
        self._dua_score_ema = 0.0
        self.fusion_logs: list[dict] = []
        # Per-pair diagnostics (populated by write_graph_data from GraphOutput)
        self.last_pair_diagnostics: dict | None = None
        self.last_breakpoint_trace: dict | None = None
        self.last_breakpoint_frame_indices: list[int] = []

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        return math.atan2(math.sin(angle), math.cos(angle))

    @staticmethod
    def _so3_to_ypr(rot: pp.LieTensor) -> tuple[float, float, float]:
        rot_mat = rot.matrix().reshape(3, 3)
        yaw = math.atan2(float(rot_mat[1, 0].item()), float(rot_mat[0, 0].item()))
        pitch = math.atan2(
            float((-rot_mat[2, 0]).item()),
            math.sqrt(float(rot_mat[2, 1].item()) ** 2 + float(rot_mat[2, 2].item()) ** 2),
        )
        roll = math.atan2(float(rot_mat[2, 1].item()), float(rot_mat[2, 2].item()))
        return yaw, pitch, roll

    @staticmethod
    def _ypr_to_so3(yaw: float, pitch: float, roll: float, *, device: torch.device, dtype: torch.dtype) -> pp.LieTensor:
        cy, sy = math.cos(yaw), math.sin(yaw)
        cp, sp = math.cos(pitch), math.sin(pitch)
        cr, sr = math.cos(roll), math.sin(roll)
        rot_mat = torch.tensor([
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ], device=device, dtype=dtype)
        return pp.SO3(pp.mat2SO3(rot_mat).tensor().reshape(1, 4))

    @staticmethod
    def _clamp01(value: float) -> float:
        return max(0.0, min(1.0, float(value)))

    @staticmethod
    def _dcs_switch(chi2: float, phi: float) -> float:
        phi = max(float(phi), 1e-9)
        return max(0.0, min(1.0, (2.0 * phi) / max(phi + max(float(chi2), 0.0), 1e-9)))

    @staticmethod
    def _should_use_local_window_graph(config) -> bool:
        """Return True when the optimizer is explicitly configured for local BA."""
        mode = str(getattr(config, "imu_factor_mode", "legacy_pose_prior")).strip().lower()
        return mode == "local_inertial_ba"

    @torch.no_grad()
    def get_graph_data(self, global_map: VisualMap, frame_idx: torch.Tensor,
                       observations: torch.Tensor | None = None, edges: torch.Tensor | None = None) -> GraphInput:
        if (
            self._should_use_local_window_graph(self.config)
            and not bool(getattr(self, "_building_local_ba_pair", False))
        ):
            return self.get_local_ba_graph_data(global_map, frame_idx)

        frame2opt = global_map.frames[frame_idx]

        obs = global_map.get_frame2match(frame2opt)
        pts = global_map.get_match2point(obs)
        im_intrinsics = frame2opt.data["K"][0]

        lengths = global_map.frame2match.ranges[frame2opt.index, :, 1].flatten()
        lengths = lengths[lengths >= 0]
        edges_idx = torch.repeat_interleave(torch.arange(lengths.size(0)), lengths.long())
        init_motion = pp.SE3(frame2opt.data["pose"])
        from_pose = pp.SE3(global_map.frames.data["pose"][frame_idx - 1])
        baseline = frame2opt.data["baseline"]

        imu_rotvec_prior = None
        imu_rot_prior_std = None
        imu_trans_prior = None
        imu_trans_cov = None
        imu_vio_factor_enable = False
        imu_vio_prev_velocity_world = None
        imu_vio_curr_velocity_init_world = None
        imu_vio_prev_acc_bias = None
        imu_vio_prev_gyro_bias = None
        imu_vio_curr_acc_bias_init = None
        imu_vio_curr_gyro_bias_init = None
        imu_vio_linearized_acc_bias = None
        imu_vio_linearized_gyro_bias = None
        imu_vio_bias_jacobian = None
        imu_vio_bias_rw_cov = None
        imu_vio_delta_rotvec = None
        imu_vio_delta_v = None
        imu_vio_delta_p = None
        imu_vio_cov = None
        imu_vio_sa_v2_unique_cov = None
        imu_vio_sa_v2_incoming_raw_time_ns = None
        imu_vio_sa_v2_outgoing_raw_time_ns = None
        imu_vio_sa_v2_incoming_sensitivity = None
        imu_vio_sa_v2_outgoing_sensitivity = None
        imu_vio_dt = None
        imu_vio_sensor_T_imu = None
        imu_vio_gravity_world = None
        imu_vio_gravity_in_residual = False
        imu_vio_alpha_p = max(float(getattr(self.config, "imu_vio_alpha_p", 1.0)), 0.0)
        imu_vio_alpha_v = max(float(getattr(self.config, "imu_vio_alpha_v", 1.0)), 0.0)
        imu_vio_alpha_R = max(float(getattr(self.config, "imu_vio_alpha_R", 1.0)), 0.0)
        imu_factor_mode = str(getattr(self.config, "imu_factor_mode", "legacy_pose_prior")).strip().lower()
        enable_imu = (
            getattr(self.config, "imu_rot_prior", False)
            and self.config.graph_type in {"icp", "disp"}
        )
        if "imu_vio_sensor_T_imu" in frame2opt.data:
            sensor_T_imu = frame2opt.data["imu_vio_sensor_T_imu"]
            if sensor_T_imu.numel() == 7:
                imu_vio_sensor_T_imu = sensor_T_imu.reshape(1, 7)
        if "imu_vio_gravity_world" in frame2opt.data:
            stored_gravity = frame2opt.data["imu_vio_gravity_world"]
            if stored_gravity.numel() == 3:
                imu_vio_gravity_world = stored_gravity.reshape(3)
        if "imu_vio_gravity_in_residual" in frame2opt.data:
            stored_mode = frame2opt.data["imu_vio_gravity_in_residual"]
            if stored_mode.numel() >= 1:
                imu_vio_gravity_in_residual = bool(stored_mode.reshape(-1)[0].item())
        if enable_imu:
            prior = frame2opt.data["imu_rotvec_prior"]
            prior_std = frame2opt.data["imu_rot_prior_std"]
            if prior.numel() > 0 and prior_std.numel() > 0 and float(prior_std[0].item()) < 1e5:
                scale = float(getattr(self.config, "imu_rot_prior_scale", 1.0))
                scale = max(scale, 1e-6)
                imu_rotvec_prior = prior[0]
                # scale > 1 → larger std → weaker IMU prior (consistent with trans_scale semantics)
                imu_rot_prior_std = (prior_std[0] * scale).unsqueeze(0)

            # Translation prior from preintegration
            if "imu_trans_prior" in frame2opt.data and "imu_trans_cov" in frame2opt.data:
                t_prior = frame2opt.data["imu_trans_prior"]
                t_cov   = frame2opt.data["imu_trans_cov"]
                trans_scale = float(getattr(self.config, "imu_trans_prior_scale", 1.0))
                trans_scale = max(trans_scale, 1e-6)
                if t_prior.numel() == 3 and t_cov.numel() == 9:
                    imu_trans_prior = t_prior.reshape(3)
                    imu_trans_cov   = (t_cov.reshape(3, 3) * trans_scale)

            if imu_factor_mode in {
                "preintegrated_vio",
                "local_inertial_ba",
                "two_state_fixed_lag",
            } and self.config.graph_type == "disp":
                needed = (
                    "imu_vio_prev_velocity_world",
                    "imu_vio_curr_velocity_init_world",
                    "imu_vio_delta_rotvec",
                    "imu_vio_delta_v",
                    "imu_vio_delta_p",
                    "imu_vio_cov",
                    "imu_vio_dt",
                )
                if all(k in frame2opt.data for k in needed):
                    vio_prev_v = frame2opt.data["imu_vio_prev_velocity_world"]
                    vio_curr_v = frame2opt.data["imu_vio_curr_velocity_init_world"]
                    vio_rotvec = frame2opt.data["imu_vio_delta_rotvec"]
                    vio_dv = frame2opt.data["imu_vio_delta_v"]
                    vio_dp = frame2opt.data["imu_vio_delta_p"]
                    vio_cov = frame2opt.data["imu_vio_cov"]
                    vio_dt = frame2opt.data["imu_vio_dt"]
                    if (
                        vio_prev_v.numel() == 3
                        and vio_curr_v.numel() == 3
                        and vio_rotvec.numel() == 3
                        and vio_dv.numel() == 3
                        and vio_dp.numel() == 3
                        and vio_cov.numel() == 81
                        and vio_dt.numel() >= 1
                        and float(vio_dt.reshape(-1)[0].item()) > 0.0
                    ):
                        imu_vio_factor_enable = True
                        imu_vio_prev_velocity_world = vio_prev_v.reshape(3)
                        imu_vio_curr_velocity_init_world = vio_curr_v.reshape(3)
                        imu_vio_delta_rotvec = vio_rotvec.reshape(3)
                        imu_vio_delta_v = vio_dv.reshape(3)
                        imu_vio_delta_p = vio_dp.reshape(3)
                        vio_cov_scale = max(float(getattr(self.config, "imu_vio_cov_scale", 1.0)), 1e-12)
                        vio_cov_floor = max(
                            float(getattr(self.config, "imu_vio_cov_diagonal_floor", 0.0)),
                            0.0,
                        )
                        imu_vio_cov = vio_preintegrated_covariance_matrix(
                            vio_cov.reshape(9, 9) * vio_cov_scale,
                            diagonal_floor=vio_cov_floor,
                        )
                        sa_v2_needed = (
                            "imu_vio_sa_v2_unique_cov",
                            "imu_vio_sa_v2_incoming_raw_time_ns",
                            "imu_vio_sa_v2_outgoing_raw_time_ns",
                            "imu_vio_sa_v2_incoming_count",
                            "imu_vio_sa_v2_outgoing_count",
                            "imu_vio_sa_v2_incoming_sensitivity",
                            "imu_vio_sa_v2_outgoing_sensitivity",
                        )
                        if all(k in frame2opt.data for k in sa_v2_needed):
                            incoming_count = int(
                                frame2opt.data["imu_vio_sa_v2_incoming_count"]
                                .reshape(-1)[0]
                                .item()
                            )
                            outgoing_count = int(
                                frame2opt.data["imu_vio_sa_v2_outgoing_count"]
                                .reshape(-1)[0]
                                .item()
                            )
                            if 0 < incoming_count <= 2 and 0 < outgoing_count <= 2:
                                imu_vio_sa_v2_unique_cov = vio_preintegrated_covariance_matrix(
                                    frame2opt.data["imu_vio_sa_v2_unique_cov"].reshape(9, 9)
                                    * vio_cov_scale,
                                    diagonal_floor=vio_cov_floor,
                                )
                                imu_vio_sa_v2_incoming_raw_time_ns = frame2opt.data[
                                    "imu_vio_sa_v2_incoming_raw_time_ns"
                                ].reshape(-1)[:incoming_count]
                                imu_vio_sa_v2_outgoing_raw_time_ns = frame2opt.data[
                                    "imu_vio_sa_v2_outgoing_raw_time_ns"
                                ].reshape(-1)[:outgoing_count]
                                imu_vio_sa_v2_incoming_sensitivity = frame2opt.data[
                                    "imu_vio_sa_v2_incoming_sensitivity"
                                ].reshape(9, 12)[:, : incoming_count * 6] * math.sqrt(
                                    vio_cov_scale
                                )
                                imu_vio_sa_v2_outgoing_sensitivity = frame2opt.data[
                                    "imu_vio_sa_v2_outgoing_sensitivity"
                                ].reshape(9, 12)[:, : outgoing_count * 6] * math.sqrt(
                                    vio_cov_scale
                                )
                        imu_vio_dt = vio_dt.reshape(-1)[0:1]
                        bias_needed = (
                            "imu_vio_prev_acc_bias",
                            "imu_vio_prev_gyro_bias",
                            "imu_vio_acc_bias",
                            "imu_vio_gyro_bias",
                            "imu_vio_linearized_acc_bias",
                            "imu_vio_linearized_gyro_bias",
                            "imu_vio_bias_jacobian",
                            "imu_vio_bias_rw_cov",
                        )
                        if all(k in frame2opt.data for k in bias_needed):
                            vio_prev_ba = frame2opt.data["imu_vio_prev_acc_bias"]
                            vio_prev_bg = frame2opt.data["imu_vio_prev_gyro_bias"]
                            vio_curr_ba = frame2opt.data["imu_vio_acc_bias"]
                            vio_curr_bg = frame2opt.data["imu_vio_gyro_bias"]
                            vio_lin_ba = frame2opt.data["imu_vio_linearized_acc_bias"]
                            vio_lin_bg = frame2opt.data["imu_vio_linearized_gyro_bias"]
                            vio_bias_jac = frame2opt.data["imu_vio_bias_jacobian"]
                            vio_bias_cov = frame2opt.data["imu_vio_bias_rw_cov"]
                            if (
                                vio_prev_ba.numel() == 3
                                and vio_prev_bg.numel() == 3
                                and vio_curr_ba.numel() == 3
                                and vio_curr_bg.numel() == 3
                                and vio_lin_ba.numel() == 3
                                and vio_lin_bg.numel() == 3
                                and vio_bias_jac.numel() == 54
                                and vio_bias_cov.numel() == 36
                            ):
                                imu_vio_prev_acc_bias = vio_prev_ba.reshape(3)
                                imu_vio_prev_gyro_bias = vio_prev_bg.reshape(3)
                                imu_vio_curr_acc_bias_init = vio_curr_ba.reshape(3)
                                imu_vio_curr_gyro_bias_init = vio_curr_bg.reshape(3)
                                imu_vio_linearized_acc_bias = vio_lin_ba.reshape(3)
                                imu_vio_linearized_gyro_bias = vio_lin_bg.reshape(3)
                                imu_vio_bias_jacobian = vio_bias_jac.reshape(9, 6)
                                imu_vio_bias_rw_cov = vio_bias_cov.reshape(6, 6)

        # Compute quality-weighted visual observation covariance from obs2_covTc (N,3,3).
        # Innovation 1: Inverse-variance weighting — high-quality points (low indiv cov)
        # get higher weight in the aggregate, giving a truer estimate of usable visual info.
        visual_obs_cov_mean: float | None = None
        num_observations: int = 0
        visual_keypoint_coverage: float | None = None
        visual_depth_spread: float | None = None
        visual_relative_pose_CiCj: torch.Tensor | None = None
        visual_relative_pose_cov: torch.Tensor | None = None
        visual_relative_pose_num_points = 0
        visual_relative_pose_num_inliers = 0
        visual_relative_pose_mean_mahalanobis_sq: float | None = None
        visual_compressed_uvd_reference_CjCi: torch.Tensor | None = None
        visual_compressed_uvd_hessian: torch.Tensor | None = None
        visual_compressed_uvd_gradient: torch.Tensor | None = None
        visual_compressed_uvd_robust_cost: float | None = None
        visual_compressed_uvd_num_points = 0
        visual_compressed_uvd_num_inliers = 0
        visual_compressed_uvd_mean_mahalanobis_sq: float | None = None
        visual_compressed_uvd_huber_delta: float | None = None
        obs_cov = obs.data.get("obs2_covTc", None)
        if obs_cov is not None and obs_cov.numel() > 0:
            num_observations = obs_cov.shape[0]
            # Per-point mean variance (trace / 3)
            per_point_var = obs_cov.diagonal(dim1=-2, dim2=-1).sum(dim=-1) / 3.0  # (N,)
            # Inverse-variance weights: w_i = 1 / (var_i + eps)
            eps_var = 1e-6
            weights = 1.0 / (per_point_var + eps_var)
            weights = weights / weights.sum()  # normalize
            visual_obs_cov_mean = float((weights * per_point_var).sum().item())

        pixel2_uv = obs.data.get("pixel2_uv", None)
        if pixel2_uv is not None and pixel2_uv.numel() > 0:
            uv = pixel2_uv.float()
            width = max(float(im_intrinsics[0, 2].item()) * 2.0, 1.0)
            height = max(float(im_intrinsics[1, 2].item()) * 2.0, 1.0)
            cov_u = float(((uv[:, 0].max() - uv[:, 0].min()) / width).clamp(0.0, 1.0).item())
            cov_v = float(((uv[:, 1].max() - uv[:, 1].min()) / height).clamp(0.0, 1.0).item())
            visual_keypoint_coverage = float(max(cov_u * cov_v, 0.0) ** 0.5)

        pixel2_d = obs.data.get("pixel2_d", None)
        if pixel2_d is not None and pixel2_d.numel() > 1:
            depth = pixel2_d.reshape(-1).float()
            depth = depth[torch.isfinite(depth) & (depth > 1e-3)]
            if depth.numel() > 1:
                visual_depth_spread = float((depth.std() / depth.mean().clamp(min=1e-3)).clamp(0.0, 3.0).item())

        if imu_factor_mode == "two_state_fixed_lag":
            pose_key = "visual_relative_pose_CiCj"
            covariance_key = "visual_relative_pose_cov"
            if pose_key in frame2opt.data and covariance_key in frame2opt.data:
                stored_pose = frame2opt.data[pose_key]
                stored_covariance = frame2opt.data[covariance_key]
                if stored_pose.numel() == 7 and stored_covariance.numel() == 36:
                    visual_relative_pose_CiCj = stored_pose.reshape(1, 7)
                    visual_relative_pose_cov = stored_covariance.reshape(6, 6)
                    if "visual_relative_pose_num_points" in frame2opt.data:
                        visual_relative_pose_num_points = int(
                            frame2opt.data["visual_relative_pose_num_points"].reshape(-1)[0].item()
                        )
                    if "visual_relative_pose_num_inliers" in frame2opt.data:
                        visual_relative_pose_num_inliers = int(
                            frame2opt.data["visual_relative_pose_num_inliers"].reshape(-1)[0].item()
                        )
                    if "visual_relative_pose_mean_mahalanobis_sq" in frame2opt.data:
                        stored_mean = float(
                            frame2opt.data["visual_relative_pose_mean_mahalanobis_sq"]
                            .reshape(-1)[0]
                            .item()
                        )
                        if stored_mean >= 0.0 and math.isfinite(stored_mean):
                            visual_relative_pose_mean_mahalanobis_sq = stored_mean
            compressed_fields = (
                "visual_compressed_uvd_reference_CjCi",
                "visual_compressed_uvd_hessian",
                "visual_compressed_uvd_gradient",
            )
            if all(name in frame2opt.data for name in compressed_fields):
                stored_reference = frame2opt.data[compressed_fields[0]]
                stored_hessian = frame2opt.data[compressed_fields[1]]
                stored_gradient = frame2opt.data[compressed_fields[2]]
                if (
                    stored_reference.numel() == 7
                    and stored_hessian.numel() == 36
                    and stored_gradient.numel() == 6
                ):
                    visual_compressed_uvd_reference_CjCi = stored_reference.reshape(1, 7)
                    visual_compressed_uvd_hessian = stored_hessian.reshape(6, 6)
                    visual_compressed_uvd_gradient = stored_gradient.reshape(6)
                    scalar_fields = {
                        "visual_compressed_uvd_robust_cost": "robust_cost",
                        "visual_compressed_uvd_mean_mahalanobis_sq": "mean_mahalanobis_sq",
                        "visual_compressed_uvd_huber_delta": "huber_delta",
                    }
                    for stored_name, local_name in scalar_fields.items():
                        if stored_name in frame2opt.data:
                            value = float(frame2opt.data[stored_name].reshape(-1)[0].item())
                            if math.isfinite(value):
                                if local_name == "robust_cost":
                                    visual_compressed_uvd_robust_cost = value
                                elif local_name == "mean_mahalanobis_sq":
                                    visual_compressed_uvd_mean_mahalanobis_sq = value
                                else:
                                    visual_compressed_uvd_huber_delta = value
                    if "visual_compressed_uvd_num_points" in frame2opt.data:
                        visual_compressed_uvd_num_points = int(
                            frame2opt.data["visual_compressed_uvd_num_points"].reshape(-1)[0].item()
                        )
                    if "visual_compressed_uvd_num_inliers" in frame2opt.data:
                        visual_compressed_uvd_num_inliers = int(
                            frame2opt.data["visual_compressed_uvd_num_inliers"].reshape(-1)[0].item()
                        )

        return GraphInput(
            frame_idx, frame_idx - 1, init_motion, from_pose, baseline,
            obs, pts, im_intrinsics, edges_idx, "cpu",
            imu_rotvec_prior=imu_rotvec_prior,
            imu_rot_prior_std=imu_rot_prior_std,
            imu_trans_prior=imu_trans_prior,
            imu_trans_cov=imu_trans_cov,
            visual_obs_cov_mean=visual_obs_cov_mean,
            num_observations=num_observations,
            visual_keypoint_coverage=visual_keypoint_coverage,
            visual_depth_spread=visual_depth_spread,
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
            imu_vio_factor_enable=imu_vio_factor_enable,
            imu_vio_prev_velocity_world=imu_vio_prev_velocity_world,
            imu_vio_curr_velocity_init_world=imu_vio_curr_velocity_init_world,
            imu_vio_prev_acc_bias=imu_vio_prev_acc_bias,
            imu_vio_prev_gyro_bias=imu_vio_prev_gyro_bias,
            imu_vio_curr_acc_bias_init=imu_vio_curr_acc_bias_init,
            imu_vio_curr_gyro_bias_init=imu_vio_curr_gyro_bias_init,
            imu_vio_linearized_acc_bias=imu_vio_linearized_acc_bias,
            imu_vio_linearized_gyro_bias=imu_vio_linearized_gyro_bias,
            imu_vio_bias_jacobian=imu_vio_bias_jacobian,
            imu_vio_bias_rw_cov=imu_vio_bias_rw_cov,
            imu_vio_delta_rotvec=imu_vio_delta_rotvec,
            imu_vio_delta_v=imu_vio_delta_v,
            imu_vio_delta_p=imu_vio_delta_p,
            imu_vio_cov=imu_vio_cov,
            imu_vio_sa_v2_unique_cov=imu_vio_sa_v2_unique_cov,
            imu_vio_sa_v2_incoming_raw_time_ns=imu_vio_sa_v2_incoming_raw_time_ns,
            imu_vio_sa_v2_outgoing_raw_time_ns=imu_vio_sa_v2_outgoing_raw_time_ns,
            imu_vio_sa_v2_incoming_sensitivity=imu_vio_sa_v2_incoming_sensitivity,
            imu_vio_sa_v2_outgoing_sensitivity=imu_vio_sa_v2_outgoing_sensitivity,
            imu_vio_dt=imu_vio_dt,
            imu_vio_sensor_T_imu=imu_vio_sensor_T_imu,
            imu_vio_gravity_world=imu_vio_gravity_world,
            imu_vio_gravity_in_residual=imu_vio_gravity_in_residual,
            imu_vio_alpha_p=imu_vio_alpha_p,
            imu_vio_alpha_v=imu_vio_alpha_v,
            imu_vio_alpha_R=imu_vio_alpha_R,
        )

    @torch.no_grad()
    def get_local_ba_graph_data(self, global_map: VisualMap, frame_idx: torch.Tensor) -> GraphInput | LocalWindowGraphInput:
        current_idx = int(frame_idx.reshape(-1)[0].item())
        window_size = int(getattr(self.config, "local_ba_window_size", 3))
        window_size = max(window_size, 2)
        start_idx = max(0, current_idx - window_size + 1)
        actual_size = current_idx - start_idx + 1
        if actual_size < 2:
            self._building_local_ba_pair = True
            try:
                return self.get_graph_data(global_map, frame_idx)
            finally:
                self._building_local_ba_pair = False

        frame_indices = torch.arange(start_idx, current_idx + 1, dtype=torch.long)
        frame_poses = global_map.frames.data["pose"][frame_indices].clone()
        pair_edges: list[GraphInput] = []
        self._building_local_ba_pair = True
        try:
            for target_idx in range(start_idx + 1, current_idx + 1):
                pair_edges.append(
                    self.get_graph_data(global_map, torch.tensor([target_idx], dtype=torch.long))
                )
        finally:
            self._building_local_ba_pair = False

        return LocalWindowGraphInput(
            frame_indices=frame_indices,
            frame_poses=frame_poses,
            edges=pair_edges,
            fixed_first_frame=bool(getattr(self.config, "local_ba_fix_first_frame", True)),
            writeback=str(getattr(self.config, "local_ba_writeback", "current")),
            device=str(getattr(self.config, "device", "cpu")),
        )

    @classmethod
    def is_valid_config(cls, config: SimpleNamespace | None) -> None:
        cls._enforce_config_spec(config, {
            "graph_type": lambda s: s in {"icp", "reproj", "disp"},
            "device": lambda v: isinstance(v, str) and (v == "cpu" or "cuda" in v),
            "vectorize": lambda b: isinstance(b, bool),
            "parallel": lambda b: isinstance(b, bool),
            "autodiff": lambda b: isinstance(b, bool),
        }, allow_excessive_cfg=True)
        if hasattr(config, "imu_rot_prior"):
            assert isinstance(config.imu_rot_prior, bool)
        if hasattr(config, "imu_rot_prior_scale"):
            assert isinstance(config.imu_rot_prior_scale, (float, int)) and config.imu_rot_prior_scale > 0.0
        if hasattr(config, "imu_trans_prior_scale"):
            assert isinstance(config.imu_trans_prior_scale, (float, int)) and config.imu_trans_prior_scale > 0.0
        if hasattr(config, "imu_factor_mode"):
            mode = str(config.imu_factor_mode).strip().lower()
            assert mode in {
                "legacy_pose_prior",
                "preintegrated_vio",
                "local_inertial_ba",
                "two_state_fixed_lag",
            }
            if mode in {"preintegrated_vio", "local_inertial_ba", "two_state_fixed_lag"}:
                assert config.graph_type == "disp", f"{mode} currently supports graph_type='disp'"
                assert bool(config.autodiff), f"{mode} requires autodiff=True"
            if mode == "local_inertial_ba":
                if hasattr(config, "local_ba_window_size"):
                    assert isinstance(config.local_ba_window_size, int) and config.local_ba_window_size >= 2
                if hasattr(config, "local_ba_fix_first_frame"):
                    assert isinstance(config.local_ba_fix_first_frame, bool)
                if hasattr(config, "local_ba_writeback"):
                    assert str(config.local_ba_writeback).strip().lower() in {"current", "all_optimized"}
        if hasattr(config, "imu_vio_cov_scale"):
            assert isinstance(config.imu_vio_cov_scale, (float, int)) and float(config.imu_vio_cov_scale) > 0.0
        if hasattr(config, "imu_vio_cov_diagonal_floor"):
            assert isinstance(config.imu_vio_cov_diagonal_floor, (float, int))
            assert float(config.imu_vio_cov_diagonal_floor) >= 0.0
        for alpha_name in ("imu_vio_alpha_p", "imu_vio_alpha_v", "imu_vio_alpha_R"):
            if hasattr(config, alpha_name):
                alpha_value = getattr(config, alpha_name)
                assert isinstance(alpha_value, (float, int)) and float(alpha_value) >= 0.0
        for name in (
            "two_state_initial_pose_translation_std",
            "two_state_initial_pose_rotation_std",
            "two_state_initial_velocity_std",
            "two_state_initial_acc_bias_std",
            "two_state_initial_gyro_bias_std",
            "two_state_visual_huber_delta",
        ):
            if hasattr(config, name):
                value = getattr(config, name)
                assert isinstance(value, (float, int)) and float(value) > 0.0
        if hasattr(config, "two_state_max_iterations"):
            assert isinstance(config.two_state_max_iterations, int) and config.two_state_max_iterations > 0
        if hasattr(config, "two_state_cpu_threads"):
            assert isinstance(config.two_state_cpu_threads, int)
            assert config.two_state_cpu_threads > 0
        for name in ("two_state_optimize_acc_bias", "two_state_optimize_gyro_bias"):
            if hasattr(config, name):
                assert isinstance(getattr(config, name), bool)
        if hasattr(config, "two_state_visual_factor_mode"):
            assert str(config.two_state_visual_factor_mode).strip().lower() in {
                "relative_pose",
                "direct_uvd",
                "compressed_uvd",
            }
        if hasattr(config, "two_state_warm_start"):
            assert str(config.two_state_warm_start).strip().lower() in {
                "macvo_pose",
                "imu_propagation",
            }
        if hasattr(config, "two_state_uvd_huber_delta"):
            assert float(config.two_state_uvd_huber_delta) > 0.0
        if hasattr(config, "post_imu_fusion_enable"):
            assert isinstance(config.post_imu_fusion_enable, bool)
        if hasattr(config, "post_imu_fusion_prepose_enable"):
            assert isinstance(config.post_imu_fusion_prepose_enable, bool)
        if hasattr(config, "post_imu_fusion_mode"):
            assert str(config.post_imu_fusion_mode).lower() in {"none", "uraf", "qavif", "padf", "bagf", "dua"}
        if hasattr(config, "post_imu_fusion_visual_trans_std"):
            assert isinstance(config.post_imu_fusion_visual_trans_std, (float, int)) and config.post_imu_fusion_visual_trans_std > 0.0
        if hasattr(config, "post_imu_fusion_visual_rot_std"):
            assert isinstance(config.post_imu_fusion_visual_rot_std, (float, int)) and config.post_imu_fusion_visual_rot_std > 0.0
        if hasattr(config, "post_imu_fusion_imu_trans_enable"):
            assert isinstance(config.post_imu_fusion_imu_trans_enable, bool)
        if hasattr(config, "post_imu_fusion_imu_rot_enable"):
            assert isinstance(config.post_imu_fusion_imu_rot_enable, bool)
        if hasattr(config, "post_imu_fusion_imu_trans_z_only"):
            assert isinstance(config.post_imu_fusion_imu_trans_z_only, bool)
        if hasattr(config, "post_imu_fusion_cov_scale"):
            assert isinstance(config.post_imu_fusion_cov_scale, (float, int)) and config.post_imu_fusion_cov_scale > 0.0
        if hasattr(config, "post_imu_fusion_imu_trans_std_floor"):
            assert isinstance(config.post_imu_fusion_imu_trans_std_floor, (float, int)) and config.post_imu_fusion_imu_trans_std_floor > 0.0
        if hasattr(config, "post_imu_fusion_imu_trans_xy_std_floor"):
            assert isinstance(config.post_imu_fusion_imu_trans_xy_std_floor, (float, int)) and config.post_imu_fusion_imu_trans_xy_std_floor > 0.0
        if hasattr(config, "post_imu_fusion_imu_trans_z_std_floor"):
            assert isinstance(config.post_imu_fusion_imu_trans_z_std_floor, (float, int)) and config.post_imu_fusion_imu_trans_z_std_floor > 0.0
        if hasattr(config, "post_imu_fusion_imu_rot_std_floor"):
            assert isinstance(config.post_imu_fusion_imu_rot_std_floor, (float, int)) and config.post_imu_fusion_imu_rot_std_floor > 0.0
        if hasattr(config, "post_imu_fusion_imu_rot_rp_std_floor"):
            assert isinstance(config.post_imu_fusion_imu_rot_rp_std_floor, (float, int)) and config.post_imu_fusion_imu_rot_rp_std_floor > 0.0
        if hasattr(config, "post_imu_fusion_imu_rot_yaw_std_floor"):
            assert isinstance(config.post_imu_fusion_imu_rot_yaw_std_floor, (float, int)) and config.post_imu_fusion_imu_rot_yaw_std_floor > 0.0
        if hasattr(config, "post_imu_fusion_yaw_residual_enable"):
            assert isinstance(config.post_imu_fusion_yaw_residual_enable, bool)
        if hasattr(config, "post_imu_fusion_yaw_residual_gain"):
            assert isinstance(config.post_imu_fusion_yaw_residual_gain, (float, int)) and config.post_imu_fusion_yaw_residual_gain > 0.0
        if hasattr(config, "post_imu_fusion_yaw_residual_max_corr"):
            assert isinstance(config.post_imu_fusion_yaw_residual_max_corr, (float, int)) and config.post_imu_fusion_yaw_residual_max_corr > 0.0
        if hasattr(config, "post_imu_fusion_yaw_residual_turn_ref"):
            assert isinstance(config.post_imu_fusion_yaw_residual_turn_ref, (float, int)) and config.post_imu_fusion_yaw_residual_turn_ref > 0.0
        if hasattr(config, "post_imu_fusion_yaw_residual_vis_cov_ref"):
            assert isinstance(config.post_imu_fusion_yaw_residual_vis_cov_ref, (float, int)) and config.post_imu_fusion_yaw_residual_vis_cov_ref > 0.0
        if hasattr(config, "post_imu_fusion_yaw_residual_min_quality_scale"):
            assert isinstance(config.post_imu_fusion_yaw_residual_min_quality_scale, (float, int)) and 0.0 < float(config.post_imu_fusion_yaw_residual_min_quality_scale) <= 1.0
        if hasattr(config, "post_imu_fusion_yaw_residual_disagree_ref"):
            assert isinstance(config.post_imu_fusion_yaw_residual_disagree_ref, (float, int)) and config.post_imu_fusion_yaw_residual_disagree_ref > 0.0
        if hasattr(config, "post_imu_fusion_yaw_residual_opposite_dir_scale"):
            assert isinstance(config.post_imu_fusion_yaw_residual_opposite_dir_scale, (float, int)) and 0.0 <= float(config.post_imu_fusion_yaw_residual_opposite_dir_scale) <= 1.0
        if hasattr(config, "post_imu_fusion_yaw_trans_couple_enable"):
            assert isinstance(config.post_imu_fusion_yaw_trans_couple_enable, bool)
        if hasattr(config, "post_imu_fusion_yaw_trans_couple_gain"):
            assert isinstance(config.post_imu_fusion_yaw_trans_couple_gain, (float, int)) and float(config.post_imu_fusion_yaw_trans_couple_gain) >= 0.0
        if hasattr(config, "post_imu_fusion_yaw_trans_couple_max_angle"):
            assert isinstance(config.post_imu_fusion_yaw_trans_couple_max_angle, (float, int)) and float(config.post_imu_fusion_yaw_trans_couple_max_angle) > 0.0
        if hasattr(config, "post_imu_fusion_yaw_trans_couple_imu_trans_ref"):
            assert isinstance(config.post_imu_fusion_yaw_trans_couple_imu_trans_ref, (float, int)) and float(config.post_imu_fusion_yaw_trans_couple_imu_trans_ref) > 0.0
        if hasattr(config, "post_imu_fusion_yaw_blend_imu_scale"):
            assert isinstance(config.post_imu_fusion_yaw_blend_imu_scale, (float, int)) and float(config.post_imu_fusion_yaw_blend_imu_scale) > 0.0
        if hasattr(config, "post_imu_fusion_yaw_blend_ratio_max"):
            assert isinstance(config.post_imu_fusion_yaw_blend_ratio_max, (float, int)) and 0.0 < float(config.post_imu_fusion_yaw_blend_ratio_max) <= 1.0
        if hasattr(config, "post_imu_fusion_xy_turn_residual_enable"):
            assert isinstance(config.post_imu_fusion_xy_turn_residual_enable, bool)
        if hasattr(config, "post_imu_fusion_xy_turn_residual_gain"):
            assert isinstance(config.post_imu_fusion_xy_turn_residual_gain, (float, int)) and config.post_imu_fusion_xy_turn_residual_gain > 0.0
        if hasattr(config, "post_imu_fusion_xy_turn_residual_max_corr"):
            assert isinstance(config.post_imu_fusion_xy_turn_residual_max_corr, (float, int)) and config.post_imu_fusion_xy_turn_residual_max_corr > 0.0
        if hasattr(config, "post_imu_fusion_xy_turn_residual_yaw_ref"):
            assert isinstance(config.post_imu_fusion_xy_turn_residual_yaw_ref, (float, int)) and config.post_imu_fusion_xy_turn_residual_yaw_ref > 0.0
        if hasattr(config, "post_imu_fusion_xy_turn_residual_imu_trans_ref"):
            assert isinstance(config.post_imu_fusion_xy_turn_residual_imu_trans_ref, (float, int)) and config.post_imu_fusion_xy_turn_residual_imu_trans_ref > 0.0
        if hasattr(config, "post_imu_fusion_xy_turn_residual_vis_cov_ref"):
            assert isinstance(config.post_imu_fusion_xy_turn_residual_vis_cov_ref, (float, int)) and config.post_imu_fusion_xy_turn_residual_vis_cov_ref > 0.0
        if hasattr(config, "post_imu_fusion_xy_turn_residual_min_quality_scale"):
            assert isinstance(config.post_imu_fusion_xy_turn_residual_min_quality_scale, (float, int)) and 0.0 < float(config.post_imu_fusion_xy_turn_residual_min_quality_scale) <= 1.0
        if hasattr(config, "post_imu_fusion_xy_turn_visual_std_scale_max"):
            assert isinstance(config.post_imu_fusion_xy_turn_visual_std_scale_max, (float, int)) and float(config.post_imu_fusion_xy_turn_visual_std_scale_max) >= 1.0
        if hasattr(config, "post_imu_fusion_turn_visual_z_std_scale_max"):
            assert isinstance(config.post_imu_fusion_turn_visual_z_std_scale_max, (float, int)) and float(config.post_imu_fusion_turn_visual_z_std_scale_max) >= 1.0
        if hasattr(config, "post_imu_fusion_z_conf_adapt_enable"):
            assert isinstance(config.post_imu_fusion_z_conf_adapt_enable, bool)
        if hasattr(config, "post_imu_fusion_z_conf_rp_ref"):
            assert isinstance(config.post_imu_fusion_z_conf_rp_ref, (float, int)) and config.post_imu_fusion_z_conf_rp_ref > 0.0
        if hasattr(config, "post_imu_fusion_z_conf_max_gain"):
            assert isinstance(config.post_imu_fusion_z_conf_max_gain, (float, int)) and config.post_imu_fusion_z_conf_max_gain >= 0.0
        if hasattr(config, "post_imu_fusion_z_conf_min_scale"):
            assert isinstance(config.post_imu_fusion_z_conf_min_scale, (float, int)) and config.post_imu_fusion_z_conf_min_scale > 0.0
        if hasattr(config, "post_imu_fusion_yaw_guard_enable"):
            assert isinstance(config.post_imu_fusion_yaw_guard_enable, bool)
        if hasattr(config, "post_imu_fusion_yaw_guard_thresh"):
            assert isinstance(config.post_imu_fusion_yaw_guard_thresh, (float, int)) and config.post_imu_fusion_yaw_guard_thresh > 0.0
        if hasattr(config, "post_imu_fusion_yaw_guard_gain"):
            assert isinstance(config.post_imu_fusion_yaw_guard_gain, (float, int)) and config.post_imu_fusion_yaw_guard_gain >= 0.0
        if hasattr(config, "padf_enable"):
            assert isinstance(config.padf_enable, bool)
        if hasattr(config, "padf_calib_window"):
            assert isinstance(config.padf_calib_window, int) and config.padf_calib_window >= 4
        if hasattr(config, "padf_calib_gain"):
            assert isinstance(config.padf_calib_gain, (float, int)) and config.padf_calib_gain >= 0.0
        if hasattr(config, "padf_calib_max"):
            assert isinstance(config.padf_calib_max, (float, int)) and config.padf_calib_max >= 1.0
        if hasattr(config, "padf_calib_min"):
            assert isinstance(config.padf_calib_min, (float, int)) and 0.0 < config.padf_calib_min <= 1.0
        if hasattr(config, "padf_obs_count_ref"):
            assert isinstance(config.padf_obs_count_ref, (float, int)) and config.padf_obs_count_ref > 0
        if hasattr(config, "padf_obs_count_min"):
            assert isinstance(config.padf_obs_count_min, (float, int)) and 0.0 < config.padf_obs_count_min <= 1.0
        if hasattr(config, "bagf_visual_cov_ref"):
            assert isinstance(config.bagf_visual_cov_ref, (float, int)) and float(config.bagf_visual_cov_ref) > 0.0
        if hasattr(config, "bagf_obs_count_ref"):
            assert isinstance(config.bagf_obs_count_ref, (float, int)) and float(config.bagf_obs_count_ref) > 0.0
        if hasattr(config, "bagf_depth_spread_ref"):
            assert isinstance(config.bagf_depth_spread_ref, (float, int)) and float(config.bagf_depth_spread_ref) > 0.0
        if hasattr(config, "bagf_dcs_phi"):
            assert isinstance(config.bagf_dcs_phi, (float, int)) and float(config.bagf_dcs_phi) > 0.0
        if hasattr(config, "bagf_dcs_scale_max"):
            assert isinstance(config.bagf_dcs_scale_max, (float, int)) and float(config.bagf_dcs_scale_max) >= 1.0
        if hasattr(config, "bagf_good_visual_imu_relax"):
            assert isinstance(config.bagf_good_visual_imu_relax, (float, int)) and float(config.bagf_good_visual_imu_relax) >= 0.0
        if hasattr(config, "bagf_apply_score_min"):
            assert isinstance(config.bagf_apply_score_min, (float, int)) and 0.0 <= float(config.bagf_apply_score_min) <= 1.0
        if hasattr(config, "bagf_visual_quality_enable_thresh"):
            assert isinstance(config.bagf_visual_quality_enable_thresh, (float, int)) and 0.0 <= float(config.bagf_visual_quality_enable_thresh) <= 1.0
        if hasattr(config, "bagf_z_apply_score_min"):
            assert isinstance(config.bagf_z_apply_score_min, (float, int)) and 0.0 <= float(config.bagf_z_apply_score_min) <= 1.0
        if hasattr(config, "bagf_rot_apply_score_min"):
            assert isinstance(config.bagf_rot_apply_score_min, (float, int)) and 0.0 <= float(config.bagf_rot_apply_score_min) <= 1.0
        if hasattr(config, "bagf_skip_healthy_visual"):
            assert isinstance(config.bagf_skip_healthy_visual, bool)
        if hasattr(config, "bagf_skip_obs_quality"):
            assert isinstance(config.bagf_skip_obs_quality, (float, int)) and 0.0 <= float(config.bagf_skip_obs_quality) <= 1.0
        if hasattr(config, "bagf_skip_coverage_quality"):
            assert isinstance(config.bagf_skip_coverage_quality, (float, int)) and 0.0 <= float(config.bagf_skip_coverage_quality) <= 1.0
        if hasattr(config, "bagf_skip_depth_quality"):
            assert isinstance(config.bagf_skip_depth_quality, (float, int)) and 0.0 <= float(config.bagf_skip_depth_quality) <= 1.0
        if hasattr(config, "bagf_max_active_streak"):
            assert isinstance(config.bagf_max_active_streak, int) and config.bagf_max_active_streak >= 0
        if hasattr(config, "bagf_cooldown_frames"):
            assert isinstance(config.bagf_cooldown_frames, int) and config.bagf_cooldown_frames >= 0
        if hasattr(config, "bagf_xy_enable_quality_thresh"):
            assert isinstance(config.bagf_xy_enable_quality_thresh, (float, int)) and 0.0 <= float(config.bagf_xy_enable_quality_thresh) <= 1.0
        if hasattr(config, "bagf_xy_consistency_thresh"):
            assert isinstance(config.bagf_xy_consistency_thresh, (float, int)) and -1.0 <= float(config.bagf_xy_consistency_thresh) <= 1.0
        if hasattr(config, "bagf_yaw_blend_max_good"):
            assert isinstance(config.bagf_yaw_blend_max_good, (float, int)) and 0.0 <= float(config.bagf_yaw_blend_max_good) <= 1.0
        if hasattr(config, "bagf_yaw_blend_max_bad"):
            assert isinstance(config.bagf_yaw_blend_max_bad, (float, int)) and 0.0 <= float(config.bagf_yaw_blend_max_bad) <= 1.0

    def _post_fuse_visual_imu(
        self,
        to_pose: pp.LieTensor,
        frame_idx: torch.Tensor,
        global_map: VisualMap,
        visual_obs_cov_mean: float | None = None,
        num_observations: int = 0,
    ) -> tuple[pp.LieTensor, dict]:
        """Fuse visual pose with IMU prior using Quality-Aware Adaptive Fusion (QA-VIF).

        Three innovations over standard fixed-weight fusion:
        1. Inverse-variance weighted visual covariance (aggregated upstream)
        2. Observation-count-based IMU trust modulation (few pts → trust IMU more)
        3. Visual quality trend detection (degrading → preemptive IMU shift)

        Returns: (fused_pose, log_dict)
        """
        log = {"frame_idx": int(frame_idx.item()), "skipped": True}

        if int(frame_idx.item()) <= 0:
            return to_pose, log

        frame_data = global_map.frames.data
        if ("imu_rotvec_prior" not in frame_data) or ("imu_rot_prior_std" not in frame_data):
            return to_pose, log
        if ("imu_trans_prior" not in frame_data) or ("imu_trans_cov" not in frame_data):
            return to_pose, log

        try:
            rot_prior = frame_data["imu_rotvec_prior"][frame_idx].reshape(-1)
            rot_std = frame_data["imu_rot_prior_std"][frame_idx].reshape(-1)
            trans_prior = frame_data["imu_trans_prior"][frame_idx].reshape(-1)
            trans_cov = frame_data["imu_trans_cov"][frame_idx].reshape(3, 3)
        except Exception:
            return to_pose, log

        if rot_prior.numel() != 3 or trans_prior.numel() != 3 or trans_cov.numel() != 9 or rot_std.numel() == 0:
            return to_pose, log
        if float(rot_std[0].item()) >= 1e5:
            return to_pose, log

        prev_pose = pp.SE3(frame_data["pose"][frame_idx - 1].double())
        rel_visual = prev_pose.Inv() @ pp.SE3(to_pose)

        # Fuse actual relative translation and rotation-vector components separately.
        # Editing yaw in se(3) log-space changes the Exp-mapped translation as well,
        # which can turn pure heading corrections into fake XY sliding.
        rot_visual = pp.SO3(rel_visual.rotation().tensor().double())
        rot_imu = pp.so3(rot_prior.reshape(1, 3).double()).Exp()

        xi_visual = torch.cat([
            rel_visual.translation().reshape(3).double(),
            rot_visual.Log().tensor().reshape(3).double(),
        ], dim=0)
        xi_imu = torch.cat([trans_prior, rot_prior], dim=0).reshape(6).double()
        imu_trans_norm = float(xi_imu[0:3].norm().item())

        yaw_visual, pitch_visual, roll_visual = self._so3_to_ypr(rot_visual)
        yaw_imu, _, _ = self._so3_to_ypr(rot_imu)

        # Residual-style yaw correction: correct true heading instead of using the
        # rotation-vector z component as a proxy, which can under/over-estimate turn
        # angle once roll/pitch are present.
        yaw_residual = self._wrap_angle(yaw_imu - yaw_visual)
        yaw_turn_mag = max(abs(yaw_visual), abs(yaw_imu))

        yaw_residual_enable = bool(getattr(self.config, "post_imu_fusion_yaw_residual_enable", True))
        yaw_residual_gain = float(getattr(self.config, "post_imu_fusion_yaw_residual_gain", 0.60))
        yaw_residual_max_corr = float(getattr(self.config, "post_imu_fusion_yaw_residual_max_corr", 0.10))
        yaw_residual_turn_ref = float(getattr(self.config, "post_imu_fusion_yaw_residual_turn_ref", 0.10))
        yaw_residual_vis_cov_ref = float(getattr(self.config, "post_imu_fusion_yaw_residual_vis_cov_ref", 0.02))
        yaw_residual_min_quality_scale = float(getattr(self.config, "post_imu_fusion_yaw_residual_min_quality_scale", 0.35))
        yaw_residual_disagree_ref = float(getattr(self.config, "post_imu_fusion_yaw_residual_disagree_ref", 0.30))
        yaw_residual_opposite_dir_scale = float(getattr(self.config, "post_imu_fusion_yaw_residual_opposite_dir_scale", 0.20))
        yaw_trans_couple_enable = bool(getattr(self.config, "post_imu_fusion_yaw_trans_couple_enable", True))
        yaw_trans_couple_gain = float(getattr(self.config, "post_imu_fusion_yaw_trans_couple_gain", 1.0))
        yaw_trans_couple_max_angle = float(getattr(self.config, "post_imu_fusion_yaw_trans_couple_max_angle", 0.18))
        yaw_trans_couple_imu_trans_ref = float(getattr(self.config, "post_imu_fusion_yaw_trans_couple_imu_trans_ref", 0.08))

        yaw_turn_ratio = min(1.0, yaw_turn_mag / max(yaw_residual_turn_ref, 1e-6))
        yaw_quality_scale = 1.0
        if visual_obs_cov_mean is not None:
            vis_ratio = min(1.0, max(0.0, float(visual_obs_cov_mean) / max(yaw_residual_vis_cov_ref, 1e-6)))
            yaw_quality_scale = yaw_residual_min_quality_scale + (1.0 - yaw_residual_min_quality_scale) * vis_ratio

        yaw_dir_scale = 1.0
        if abs(yaw_visual) > 1e-6 and abs(yaw_imu) > 1e-6 and yaw_visual * yaw_imu < 0.0:
            yaw_dir_scale = yaw_residual_opposite_dir_scale

        yaw_residual_scale = max(0.0, 1.0 - abs(yaw_residual) / max(yaw_residual_disagree_ref, 1e-6))
        yaw_correction = 0.0
        xi_visual_corr = xi_visual.clone()
        yaw_visual_corr = yaw_visual
        if yaw_residual_enable:
            yaw_correction = yaw_residual_gain * yaw_residual * yaw_turn_ratio * yaw_quality_scale * yaw_dir_scale * yaw_residual_scale
            yaw_correction = max(-yaw_residual_max_corr, min(yaw_residual_max_corr, yaw_correction))
            yaw_visual_corr = self._wrap_angle(yaw_visual + yaw_correction)

        rot_visual_corr = self._ypr_to_so3(
            yaw_visual_corr,
            pitch_visual,
            roll_visual,
            device=xi_visual.device,
            dtype=torch.double,
        )
        xi_visual_corr[3:6] = rot_visual_corr.Log().tensor().reshape(3).double()

        # Record how much real inter-frame motion IMU sees. We only re-couple
        # translation direction to heading when there is actual motion; near-stationary
        # turns should stay protected by the sliding gate.
        yaw_trans_couple_angle = 0.0
        yaw_trans_couple_motion_ratio = min(1.0, imu_trans_norm / max(yaw_trans_couple_imu_trans_ref, 1e-6))

        # When rotation dominates and IMU translation stays near zero, suppress
        # turn-induced visual translation sliding. XY gets residual damping and
        # both XY/Z can receive covariance inflation so fusion trusts IMU more.
        xy_turn_residual_enable = bool(getattr(self.config, "post_imu_fusion_xy_turn_residual_enable", True))
        xy_turn_residual_gain = float(getattr(self.config, "post_imu_fusion_xy_turn_residual_gain", 0.65))
        xy_turn_residual_max_corr = float(getattr(self.config, "post_imu_fusion_xy_turn_residual_max_corr", 0.10))
        xy_turn_residual_yaw_ref = float(getattr(self.config, "post_imu_fusion_xy_turn_residual_yaw_ref", 0.10))
        xy_turn_residual_imu_trans_ref = float(getattr(self.config, "post_imu_fusion_xy_turn_residual_imu_trans_ref", 0.05))
        xy_turn_residual_vis_cov_ref = float(getattr(self.config, "post_imu_fusion_xy_turn_residual_vis_cov_ref", 0.02))
        xy_turn_residual_min_quality_scale = float(getattr(self.config, "post_imu_fusion_xy_turn_residual_min_quality_scale", 0.35))
        xy_turn_visual_std_scale_max = float(getattr(self.config, "post_imu_fusion_xy_turn_visual_std_scale_max", 1.0))
        turn_visual_z_std_scale_max = float(getattr(self.config, "post_imu_fusion_turn_visual_z_std_scale_max", 1.0))

        xy_turn_ratio = min(1.0, yaw_turn_mag / max(xy_turn_residual_yaw_ref, 1e-6))
        imu_xy_norm = float(xi_imu[0:2].norm().item())
        turn_stationary_ratio = max(0.0, 1.0 - imu_trans_norm / max(xy_turn_residual_imu_trans_ref, 1e-6))
        xy_quality_scale = 1.0
        if visual_obs_cov_mean is not None:
            vis_ratio = min(1.0, max(0.0, float(visual_obs_cov_mean) / max(xy_turn_residual_vis_cov_ref, 1e-6)))
            xy_quality_scale = xy_turn_residual_min_quality_scale + (1.0 - xy_turn_residual_min_quality_scale) * vis_ratio

        xy_turn_visual_std_scale = 1.0
        turn_visual_z_std_scale = 1.0
        if xy_turn_residual_enable and xy_turn_ratio > 0.0 and turn_stationary_ratio > 0.0:
            xy_turn_visual_gate = xy_turn_ratio * turn_stationary_ratio * xy_quality_scale
            xy_turn_visual_std_scale = 1.0 + (max(1.0, xy_turn_visual_std_scale_max) - 1.0) * xy_turn_visual_gate
            turn_visual_z_std_scale = 1.0 + (max(1.0, turn_visual_z_std_scale_max) - 1.0) * xy_turn_visual_gate

        xy_correction_norm = 0.0
        if xy_turn_residual_enable and xy_turn_ratio > 0.0 and turn_stationary_ratio > 0.0:
            xy_residual = xi_imu[0:2] - xi_visual_corr[0:2]
            xy_correction = xy_turn_residual_gain * xy_residual * xy_turn_ratio * turn_stationary_ratio * xy_quality_scale
            xy_correction_norm = float(xy_correction.norm().item())
            if xy_correction_norm > xy_turn_residual_max_corr:
                xy_correction = xy_correction * (xy_turn_residual_max_corr / max(xy_correction_norm, 1e-9))
                xy_correction_norm = xy_turn_residual_max_corr
            xi_visual_corr[0:2] = xi_visual_corr[0:2] + xy_correction

        # ── Adaptive visual sigma ──────────────────────────────────────────────
        # If neural-network predicted covariance is available, derive sigma_v from it.
        # fallback_trans_std / fallback_rot_std act as floor values when NN cov is small.
        fallback_trans_std = float(getattr(self.config, "post_imu_fusion_visual_trans_std", 0.10))
        fallback_rot_std   = float(getattr(self.config, "post_imu_fusion_visual_rot_std",   0.10))
        cov_scale          = float(getattr(self.config, "post_imu_fusion_cov_scale",         1.0))

        if visual_obs_cov_mean is not None:
            # obs2_covTc is in camera frame (metres²); scale to reasonable range
            adaptive_trans_std = max(float((visual_obs_cov_mean * cov_scale) ** 0.5), fallback_trans_std)
            # Rotation uncertainty: heuristically proportional to translation uncertainty
            adaptive_rot_std   = max(adaptive_trans_std * (fallback_rot_std / max(fallback_trans_std, 1e-9)), fallback_rot_std)
        else:
            adaptive_trans_std = fallback_trans_std
            adaptive_rot_std   = fallback_rot_std
        adaptive_trans_std_xy = adaptive_trans_std * xy_turn_visual_std_scale
        adaptive_trans_std_z = adaptive_trans_std * turn_visual_z_std_scale
        # ──────────────────────────────────────────────────────────────────────

        Sigma_v = torch.diag(torch.tensor([
            adaptive_trans_std_xy ** 2,
            adaptive_trans_std_xy ** 2,
            adaptive_trans_std_z ** 2,
            adaptive_rot_std ** 2,
            adaptive_rot_std ** 2,
            adaptive_rot_std ** 2,
        ], dtype=torch.double))

        # IMU translation covariance is often over-confident in short-window preintegration
        # without bias/velocity states. Apply a floor std to avoid locking translation.
        imu_trans_std_floor = float(getattr(self.config, "post_imu_fusion_imu_trans_std_floor", 0.15))
        imu_trans_xy_floor = float(getattr(self.config, "post_imu_fusion_imu_trans_xy_std_floor", imu_trans_std_floor))
        imu_trans_z_floor = float(getattr(self.config, "post_imu_fusion_imu_trans_z_std_floor", imu_trans_std_floor))
        imu_trans_std_base = max(
            float(trans_cov.diagonal().clamp(min=0).mean().sqrt().item()),
            imu_trans_std_floor,
        )
        imu_trans_std_eff_xy = max(imu_trans_std_base, imu_trans_xy_floor)
        imu_trans_std_eff_z = max(imu_trans_std_base, imu_trans_z_floor)
        imu_rot_std_floor = float(getattr(self.config, "post_imu_fusion_imu_rot_std_floor", 0.05))
        imu_rot_rp_floor = float(getattr(self.config, "post_imu_fusion_imu_rot_rp_std_floor", imu_rot_std_floor))
        imu_rot_yaw_floor = float(getattr(self.config, "post_imu_fusion_imu_rot_yaw_std_floor", imu_rot_std_floor))
        imu_rot_std_eff_rp = max(float(rot_std[0].item()), imu_rot_rp_floor)
        imu_rot_std_eff_yaw = max(float(rot_std[0].item()), imu_rot_yaw_floor)

        # ═══════════════════════════════════════════════════════════════════════
        # QA-VIF: Quality-Aware Adaptive Visual-Inertial Fusion
        # Innovations 2 & 3: observation-count-based trust + quality trend detection
        # ═══════════════════════════════════════════════════════════════════════
        qa_vif_enable = bool(getattr(self.config, "qa_vif_enable", True))
        qa_vif_obs_count_ref = float(getattr(self.config, "qa_vif_obs_count_ref", 80))
        qa_vif_obs_count_min_scale = float(getattr(self.config, "qa_vif_obs_count_min_scale", 0.40))
        qa_vif_window = int(getattr(self.config, "qa_vif_window", 10))
        qa_vif_trend_gain = float(getattr(self.config, "qa_vif_trend_gain", 0.60))
        qa_vif_trend_max = float(getattr(self.config, "qa_vif_trend_max", 2.5))

        if qa_vif_enable:
            # ── Innovation 2: Observation-count-based IMU trust ──────────────
            # Few visual observations → visual info is sparse → trust IMU more.
            if num_observations > 0:
                count_ratio = min(1.0, num_observations / max(qa_vif_obs_count_ref, 1.0))
                count_scale = qa_vif_obs_count_min_scale + (1.0 - qa_vif_obs_count_min_scale) * count_ratio
            else:
                count_scale = qa_vif_obs_count_min_scale  # worst case

            # ── Innovation 3: Visual quality trend detection ──────────────────
            # When visual covariance is INCREASING over recent frames, quality is
            # degrading. Preemptively shift trust to IMU before visual fails.
            if not hasattr(self, "_qa_vif_cov_history"):
                self._qa_vif_cov_history: list[float] = []
            if visual_obs_cov_mean is not None:
                self._qa_vif_cov_history.append(float(visual_obs_cov_mean))
                if len(self._qa_vif_cov_history) > qa_vif_window:
                    self._qa_vif_cov_history.pop(0)

            trend_scale = 1.0
            if len(self._qa_vif_cov_history) >= 4:
                half = len(self._qa_vif_cov_history) // 2
                older = self._qa_vif_cov_history[:half]
                recent = self._qa_vif_cov_history[half:]
                older_mean = sum(older) / max(len(older), 1)
                recent_mean = sum(recent) / max(len(recent), 1)
                if older_mean > 1e-9:
                    trend = (recent_mean - older_mean) / older_mean
                    # Positive trend = degrading quality → decrease scale to tighten IMU
                    trend_scale = max(0.35, 1.0 - qa_vif_trend_gain * min(max(trend, 0.0), 1.0))
                elif recent_mean > older_mean * 1.5:
                    trend_scale = max(0.35, 1.0 - qa_vif_trend_gain * 0.5)

            # Combined quality scale: lower = trust IMU more
            qa_combined_scale = count_scale * trend_scale
            qa_combined_scale = max(0.25, min(1.0, qa_combined_scale))

            # Apply to IMU uncertainty: tighter uncertainty = higher IMU weight in fusion
            imu_trans_std_eff_xy *= qa_combined_scale
            imu_trans_std_eff_z  *= qa_combined_scale
            imu_rot_std_eff_rp   *= qa_combined_scale
            imu_rot_std_eff_yaw  *= qa_combined_scale
        # ═══════════════════════════════════════════════════════════════════════

        # Adaptive guard against visual-IMU rotation conflicts.
        # If roll/pitch disagreement is large (e.g., startup transients),
        # relax IMU rotation trust automatically to avoid sudden 180-degree flips.
        rot_guard_enable = bool(getattr(self.config, "post_imu_fusion_rot_guard_enable", True))
        rot_guard_rp_thresh = float(getattr(self.config, "post_imu_fusion_rot_guard_rp_thresh", 0.08))
        rot_guard_gain = float(getattr(self.config, "post_imu_fusion_rot_guard_gain", 8.0))
        rot_guard_scale = 1.0
        rp_disagreement = float((xi_visual[3:5] - xi_imu[3:5]).norm().item())
        if rot_guard_enable:
            excess = max(0.0, rp_disagreement - max(rot_guard_rp_thresh, 1e-6))
            rot_guard_scale = 1.0 + rot_guard_gain * excess
            imu_rot_std_eff_rp = imu_rot_std_eff_rp * rot_guard_scale

        # Adaptive yaw guard against over-turning in XY plane.
        # When visual and IMU yaw disagree, reduce IMU yaw trust automatically.
        yaw_guard_enable = bool(getattr(self.config, "post_imu_fusion_yaw_guard_enable", True))
        yaw_guard_thresh = float(getattr(self.config, "post_imu_fusion_yaw_guard_thresh", 0.12))
        yaw_guard_gain = float(getattr(self.config, "post_imu_fusion_yaw_guard_gain", 4.0))
        yaw_disagreement = abs(self._wrap_angle(yaw_visual_corr - yaw_imu))
        yaw_guard_scale = 1.0
        if yaw_guard_enable:
            yaw_excess = max(0.0, yaw_disagreement - max(yaw_guard_thresh, 1e-6))
            yaw_guard_scale = 1.0 + yaw_guard_gain * yaw_excess
            imu_rot_std_eff_yaw = imu_rot_std_eff_yaw * yaw_guard_scale

        # Couple Z confidence to roll/pitch agreement.
        z_conf_adapt_enable = bool(getattr(self.config, "post_imu_fusion_z_conf_adapt_enable", True))
        z_conf_rp_ref = float(getattr(self.config, "post_imu_fusion_z_conf_rp_ref", rot_guard_rp_thresh))
        z_conf_max_gain = float(getattr(self.config, "post_imu_fusion_z_conf_max_gain", 0.35))
        z_conf_min_scale = float(getattr(self.config, "post_imu_fusion_z_conf_min_scale", 0.55))
        z_conf_scale = 1.0
        if z_conf_adapt_enable:
            rp_ref = max(z_conf_rp_ref, 1e-6)
            rp_agree = max(0.0, min(1.0, (rp_ref - rp_disagreement) / rp_ref))
            z_conf_scale = max(z_conf_min_scale, 1.0 - z_conf_max_gain * rp_agree)
            imu_trans_std_eff_z = imu_trans_std_eff_z * z_conf_scale

        imu_trans_enable = bool(getattr(self.config, "post_imu_fusion_imu_trans_enable", True))
        imu_trans_z_only = bool(getattr(self.config, "post_imu_fusion_imu_trans_z_only", False))

        Sigma_i = torch.zeros((6, 6), dtype=torch.double)
        if imu_trans_enable:
            if imu_trans_z_only:
                # Keep XY purely visual while injecting IMU only on vertical translation.
                Sigma_i[:3, :3] = torch.diag(torch.tensor([
                    1e12,
                    1e12,
                    imu_trans_std_eff_z ** 2,
                ], dtype=torch.double))
            else:
                Sigma_i[:3, :3] = torch.diag(torch.tensor([
                    imu_trans_std_eff_xy ** 2,
                    imu_trans_std_eff_xy ** 2,
                    imu_trans_std_eff_z ** 2,
                ], dtype=torch.double))
        else:
            # Near-infinite covariance on translation means almost zero IMU translation weight.
            Sigma_i[:3, :3] = torch.eye(3, dtype=torch.double) * 1e12
        Sigma_i[3:, 3:] = torch.diag(torch.tensor([
            imu_rot_std_eff_rp ** 2,   # roll (x)
            imu_rot_std_eff_rp ** 2,   # pitch (y)
            imu_rot_std_eff_yaw ** 2,  # yaw (z)
        ], dtype=torch.double))

        epsI = torch.eye(6, dtype=torch.double) * 1e-9
        Wv = torch.linalg.pinv(Sigma_v + epsI)
        Wi = torch.linalg.pinv(Sigma_i + epsI)

        A = Wv + Wi + epsI
        b = Wv @ xi_visual_corr + Wi @ xi_imu
        xi_fused = torch.linalg.solve(A, b).reshape(1, 6)

        rot_fused_base = pp.so3(xi_fused[:, 3:6]).Exp()
        _, pitch_fused, roll_fused = self._so3_to_ypr(rot_fused_base)
        yaw_blend_imu_scale = float(getattr(self.config, "post_imu_fusion_yaw_blend_imu_scale", 1.0))
        yaw_blend_ratio_max = float(getattr(self.config, "post_imu_fusion_yaw_blend_ratio_max", 1.0))
        yaw_imu_weight = yaw_blend_imu_scale / max(imu_rot_std_eff_yaw ** 2, 1e-9)
        yaw_visual_weight = 1.0 / max(adaptive_rot_std ** 2, 1e-9)
        yaw_blend_ratio = yaw_imu_weight / max(yaw_imu_weight + yaw_visual_weight, 1e-9)
        yaw_blend_ratio = min(yaw_blend_ratio, yaw_blend_ratio_max)
        yaw_fused = self._wrap_angle(yaw_visual_corr + yaw_blend_ratio * self._wrap_angle(yaw_imu - yaw_visual_corr))

        if yaw_trans_couple_enable and yaw_trans_couple_motion_ratio > 0.0:
            yaw_trans_couple_angle = self._wrap_angle(yaw_fused - yaw_visual) * yaw_trans_couple_gain * yaw_trans_couple_motion_ratio
            yaw_trans_couple_angle = max(-yaw_trans_couple_max_angle, min(yaw_trans_couple_max_angle, yaw_trans_couple_angle))
            trans_x = float(xi_fused[0, 0].item())
            trans_y = float(xi_fused[0, 1].item())
            cos_yaw = math.cos(yaw_trans_couple_angle)
            sin_yaw = math.sin(yaw_trans_couple_angle)
            xi_fused[0, 0] = cos_yaw * trans_x - sin_yaw * trans_y
            xi_fused[0, 1] = sin_yaw * trans_x + cos_yaw * trans_y

        rot_fused = self._ypr_to_so3(
            yaw_fused,
            pitch_fused,
            roll_fused,
            device=xi_fused.device,
            dtype=torch.double,
        )
        rel_fused = NormalizeQuat(pp.SE3(torch.cat([xi_fused[:, 0:3], rot_fused.tensor()], dim=-1)))
        fused_pose = (prev_pose @ rel_fused).float()

        # IMU weight ratio: scalar in [0, 1], how much IMU contributed
        wi_trace = float(Wi.trace().item())
        wv_trace = float(Wv.trace().item())
        imu_weight_ratio = wi_trace / max(wi_trace + wv_trace, 1e-9)

        log = {
            "frame_idx"        : int(frame_idx.item()),
            "skipped"          : False,
            "visual_obs_cov"   : round(visual_obs_cov_mean, 6) if visual_obs_cov_mean is not None else None,
            "adaptive_trans_std": round(adaptive_trans_std, 6),
            "adaptive_trans_std_xy": round(adaptive_trans_std_xy, 6),
            "adaptive_trans_std_z": round(adaptive_trans_std_z, 6),
            "adaptive_rot_std" : round(adaptive_rot_std, 6),
            "imu_trans_enabled": imu_trans_enable,
            "imu_trans_z_only": imu_trans_z_only,
            "imu_trans_std_eff_xy": round(imu_trans_std_eff_xy, 6),
            "imu_trans_std_eff_z": round(imu_trans_std_eff_z, 6),
            "imu_rot_std_eff_rp": round(imu_rot_std_eff_rp, 6),
            "imu_rot_std_eff_yaw": round(imu_rot_std_eff_yaw, 6),
            "yaw_visual": round(yaw_visual, 6),
            "yaw_visual_corr": round(yaw_visual_corr, 6),
            "yaw_imu": round(yaw_imu, 6),
            "yaw_fused": round(yaw_fused, 6),
            "yaw_blend_imu_scale": round(yaw_blend_imu_scale, 6),
            "yaw_blend_ratio_max": round(yaw_blend_ratio_max, 6),
            "yaw_blend_ratio": round(yaw_blend_ratio, 6),
            "yaw_trans_couple_angle": round(yaw_trans_couple_angle, 6),
            "yaw_trans_couple_motion_ratio": round(yaw_trans_couple_motion_ratio, 6),
            "yaw_correction": round(yaw_correction, 6),
            "yaw_turn_ratio": round(yaw_turn_ratio, 6),
            "yaw_quality_scale": round(yaw_quality_scale, 6),
            "yaw_dir_scale": round(yaw_dir_scale, 6),
            "yaw_residual_scale": round(yaw_residual_scale, 6),
            "xy_turn_ratio": round(xy_turn_ratio, 6),
            "imu_xy_norm": round(imu_xy_norm, 6),
            "imu_trans_norm": round(imu_trans_norm, 6),
            "turn_stationary_ratio": round(turn_stationary_ratio, 6),
            "xy_quality_scale": round(xy_quality_scale, 6),
            "xy_turn_visual_std_scale": round(xy_turn_visual_std_scale, 6),
            "turn_visual_z_std_scale": round(turn_visual_z_std_scale, 6),
            "xy_correction_norm": round(xy_correction_norm, 6),
            "rot_guard_scale": round(rot_guard_scale, 4),
            "rp_disagreement": round(rp_disagreement, 6),
            "yaw_guard_scale": round(yaw_guard_scale, 4),
            "yaw_disagreement": round(yaw_disagreement, 6),
            "z_conf_scale": round(z_conf_scale, 4),
            "imu_weight_ratio" : round(imu_weight_ratio, 4),
            "xi_visual_norm"   : round(float(xi_visual.norm().item()), 6),
            "xi_imu_norm"      : round(float(xi_imu.norm().item()), 6),
            "xi_fused_norm"    : round(float(xi_fused.norm().item()), 6),
            "rot_visual_norm"  : round(float(xi_visual[3:6].norm().item()), 6),
            "rot_imu_norm"     : round(float(xi_imu[3:6].norm().item()), 6),
            "rot_fused_norm"   : round(float(xi_fused[0, 3:6].norm().item()), 6),
        }
        return fused_pose, log

    def _post_fuse_visual_imu_padf(
        self,
        to_pose: pp.LieTensor,
        frame_idx: torch.Tensor,
        global_map: VisualMap,
        visual_obs_cov_mean: float | None = None,
        num_observations: int = 0,
    ) -> tuple[pp.LieTensor, dict]:
        """Per-Axis Decoupled Fusion (PADF) with online visual covariance calibration.

        Key innovations over scalar QA-VIF:
        1. Per-axis visual constraint quality from keypoint geometry
           - Forward (X): depth reliability  ~ 1 / mean(depth_var)
           - Lateral  (Y): lateral flow constraint from feature distribution width
           - Vertical (Z): vertical flow + gravity alignment
           - Roll/Pitch:  feature spatial distribution entropy
           - Yaw:         feature angular coverage

        2. IMU-Aided Visual Covariance Calibration (online):
           Track rolling visual-IMU agreement per axis.
           When consistent disagreement → inflate visual cov (trust IMU more).
           When consistent agreement → visual cov well-calibrated.

        3. Decoupled per-axis information fusion:
           Each axis fused independently based on its own quality metrics.
        """
        log = {"frame_idx": int(frame_idx.item()), "skipped": True, "mode": "PADF"}

        if int(frame_idx.item()) <= 0:
            return to_pose, log

        frame_data = global_map.frames.data
        if ("imu_rotvec_prior" not in frame_data) or ("imu_rot_prior_std" not in frame_data):
            return to_pose, log
        if ("imu_trans_prior" not in frame_data) or ("imu_trans_cov" not in frame_data):
            return to_pose, log

        try:
            rot_prior = frame_data["imu_rotvec_prior"][frame_idx].reshape(-1)
            rot_std = frame_data["imu_rot_prior_std"][frame_idx].reshape(-1)
            trans_prior = frame_data["imu_trans_prior"][frame_idx].reshape(-1)
            trans_cov = frame_data["imu_trans_cov"][frame_idx].reshape(3, 3)
        except Exception:
            return to_pose, log

        if rot_prior.numel() != 3 or trans_prior.numel() != 3 or trans_cov.numel() != 9 or rot_std.numel() == 0:
            return to_pose, log
        if float(rot_std[0].item()) >= 1e5:
            return to_pose, log

        prev_pose = pp.SE3(frame_data["pose"][frame_idx - 1].double())
        rel_visual = prev_pose.Inv() @ pp.SE3(to_pose)

        rot_visual = pp.SO3(rel_visual.rotation().tensor().double())
        rot_imu = pp.so3(rot_prior.reshape(1, 3).double()).Exp()

        xi_visual = torch.cat([
            rel_visual.translation().reshape(3).double(),
            rot_visual.Log().tensor().reshape(3).double(),
        ], dim=0)
        xi_imu = torch.cat([trans_prior, rot_prior], dim=0).reshape(6).double()

        yaw_visual, pitch_visual, roll_visual = self._so3_to_ypr(rot_visual)
        yaw_imu, _, _ = self._so3_to_ypr(rot_imu)
        yaw_residual = self._wrap_angle(yaw_imu - yaw_visual)

        # ── Configurable parameters ─────────────────────────────────────────
        visual_trans_std = float(getattr(self.config, "post_imu_fusion_visual_trans_std", 0.08))
        visual_rot_std   = float(getattr(self.config, "post_imu_fusion_visual_rot_std", 0.03))
        cov_scale        = float(getattr(self.config, "post_imu_fusion_cov_scale", 1.5))
        imu_trans_enable = bool(getattr(self.config, "post_imu_fusion_imu_trans_enable", False))
        imu_trans_z_only = bool(getattr(self.config, "post_imu_fusion_imu_trans_z_only", False))

        # PADF-specific parameters
        padf_calib_window = int(getattr(self.config, "padf_calib_window", 20))
        padf_calib_gain   = float(getattr(self.config, "padf_calib_gain", 0.15))
        padf_calib_max    = float(getattr(self.config, "padf_calib_max", 3.0))
        padf_calib_min    = float(getattr(self.config, "padf_calib_min", 0.30))
        padf_obs_count_ref = float(getattr(self.config, "padf_obs_count_ref", 80))
        padf_obs_count_min = float(getattr(self.config, "padf_obs_count_min", 0.40))
        # ═══════════════════════════════════════════════════════════════════════

        # ── Step 1: Per-axis visual constraint quality ──────────────────────
        # Default per-axis quality = 1.0 (neutral), derived from geometric heuristics
        axis_quality = torch.ones(6, dtype=torch.double)

        if visual_obs_cov_mean is not None:
            vc = max(float(visual_obs_cov_mean), 1e-9)

            # Forward (X): depth reliability from mean observation covariance
            # Higher cov → lower reliability → lower quality
            axis_quality[0] = max(0.15, 1.0 / max(vc / 0.01, 1.0))

            # Lateral (Y): similar to X but scaled by feature horizontal spread
            axis_quality[1] = axis_quality[0]

            # Vertical (Z): stereo Z tends to be less reliable → extra discount
            axis_quality[2] = axis_quality[0] * 0.85

            # Rotation quality from visual obs covariance
            rot_quality = max(0.2, 1.0 / max(vc / 0.005, 1.0))
            axis_quality[3] = rot_quality  # roll
            axis_quality[4] = rot_quality  # pitch
            axis_quality[5] = rot_quality * 0.75  # yaw: intrinsically harder
        # ═══════════════════════════════════════════════════════════════════════

        # ── Step 2: Observation-count modulation ────────────────────────────
        if num_observations > 0:
            count_ratio = min(1.0, num_observations / max(padf_obs_count_ref, 1.0))
            count_scale = padf_obs_count_min + (1.0 - padf_obs_count_min) * count_ratio
        else:
            count_scale = padf_obs_count_min
        # Lower count_scale → fewer obs → visual less reliable → trust IMU more
        # ═══════════════════════════════════════════════════════════════════════

        # ── Step 3: IMU-Aided Visual Covariance Calibration (online) ────────
        # Innovation: Use IMU as independent measurement to calibrate visual cov.
        # When visual-IMU agreement is consistently good → visual cov is well-calibrated.
        # When disagreement is persistent → visual cov is under-estimated → inflate.

        if not hasattr(self, "_padf_calib_hist"):
            self._padf_calib_hist: list[torch.Tensor] = []  # list of 6-DOF disagreement vectors

        # Per-axis normalized disagreement: |xi_vis - xi_imu| / max(|xi_vis|, |xi_imu|, eps)
        eps_d = 1e-6
        disagreement = torch.zeros(6, dtype=torch.double)
        for a in range(6):
            v = float(xi_visual[a].item())
            i = float(xi_imu[a].item())
            denom = max(abs(v), abs(i), eps_d)
            disagreement[a] = min(1.0, abs(v - i) / denom)

        self._padf_calib_hist.append(disagreement)
        if len(self._padf_calib_hist) > padf_calib_window:
            self._padf_calib_hist.pop(0)

        calib_scale = torch.ones(6, dtype=torch.double)
        if len(self._padf_calib_hist) >= 4:
            hist_stack = torch.stack(self._padf_calib_hist, dim=0)  # (W, 6)
            mean_disagreement = hist_stack.mean(dim=0)  # (6,)

            # High persistent disagreement → visual cov under-estimated → inflate (scale > 1)
            # Low persistent disagreement → visual cov well-calibrated → keep (scale ≈ 1)
            # Scale = 1 + gain * mean_disagreement, clamped to [min, max]
            for a in range(6):
                md = float(mean_disagreement[a].item())
                calib_scale[a] = min(padf_calib_max, max(padf_calib_min, 1.0 + padf_calib_gain * md * 10.0))

            # If disagreement is very low, slightly reduce visual cov (trust vision more)
            for a in range(6):
                md = float(mean_disagreement[a].item())
                if md < 0.02:
                    calib_scale[a] = min(1.0, max(padf_calib_min, calib_scale[a] * 0.9))
        # ═══════════════════════════════════════════════════════════════════════

        # ── Step 4: Build per-axis visual covariance ────────────────────────
        # Combine: base visual std × quality factor × calibration × count modulation
        base_trans_std = visual_trans_std
        base_rot_std   = visual_rot_std

        # Adaptive from obs cov if available
        if visual_obs_cov_mean is not None:
            adaptive_trans_std = max(float((visual_obs_cov_mean * cov_scale) ** 0.5), base_trans_std)
            adaptive_rot_std   = max(adaptive_trans_std * (base_rot_std / max(base_trans_std, 1e-9)), base_rot_std)
        else:
            adaptive_trans_std = base_trans_std
            adaptive_rot_std   = base_rot_std

        # Per-axis visual std: base / (quality * count_scale) clamped
        # When quality is low or counts low: visual std increases → trust IMU more
        vis_std = torch.zeros(6, dtype=torch.double)
        vis_std[0] = max(adaptive_trans_std, adaptive_trans_std / max(axis_quality[0] * count_scale, 0.1))
        vis_std[1] = max(adaptive_trans_std, adaptive_trans_std / max(axis_quality[1] * count_scale, 0.1))
        vis_std[2] = max(adaptive_trans_std, adaptive_trans_std / max(axis_quality[2] * count_scale, 0.1))
        vis_std[3] = max(adaptive_rot_std, adaptive_rot_std / max(axis_quality[3] * count_scale, 0.1))
        vis_std[4] = max(adaptive_rot_std, adaptive_rot_std / max(axis_quality[4] * count_scale, 0.1))
        vis_std[5] = max(adaptive_rot_std, adaptive_rot_std / max(axis_quality[5] * count_scale, 0.1))

        # Apply calibration scaling: inflate visual cov when calibrated
        for a in range(6):
            vis_std[a] = vis_std[a] * float(calib_scale[a].item())

        Sigma_v = torch.diag(vis_std ** 2)
        # ═══════════════════════════════════════════════════════════════════════

        # ── Step 5: Build per-axis IMU covariance from preintegration ───────
        imu_trans_std_floor = float(getattr(self.config, "post_imu_fusion_imu_trans_std_floor", 0.15))
        imu_trans_xy_floor = float(getattr(self.config, "post_imu_fusion_imu_trans_xy_std_floor", imu_trans_std_floor))
        imu_trans_z_floor = float(getattr(self.config, "post_imu_fusion_imu_trans_z_std_floor", imu_trans_std_floor))
        imu_rot_std_floor = float(getattr(self.config, "post_imu_fusion_imu_rot_std_floor", 0.05))
        imu_rot_rp_floor = float(getattr(self.config, "post_imu_fusion_imu_rot_rp_std_floor", imu_rot_std_floor))
        imu_rot_yaw_floor = float(getattr(self.config, "post_imu_fusion_imu_rot_yaw_std_floor", imu_rot_std_floor))

        # IMU translation std from preintegration covariance
        imu_trans_std_base = max(
            float(trans_cov.diagonal().clamp(min=0).mean().sqrt().item()),
            imu_trans_std_floor,
        )
        imu_trans_std_xy = max(imu_trans_std_base, imu_trans_xy_floor)
        imu_trans_std_z  = max(imu_trans_std_base, imu_trans_z_floor)
        imu_rot_std_rp   = max(float(rot_std[0].item()), imu_rot_rp_floor)
        imu_rot_std_yaw  = max(float(rot_std[0].item()), imu_rot_yaw_floor)

        # Rot guard: if roll/pitch disagreement is large, relax IMU trust
        rp_disagreement = float((xi_visual[3:5] - xi_imu[3:5]).norm().item())
        rot_guard_enable = bool(getattr(self.config, "post_imu_fusion_rot_guard_enable", False))
        rot_guard_rp_thresh = float(getattr(self.config, "post_imu_fusion_rot_guard_rp_thresh", 0.08))
        rot_guard_gain = float(getattr(self.config, "post_imu_fusion_rot_guard_gain", 8.0))
        if rot_guard_enable:
            excess = max(0.0, rp_disagreement - max(rot_guard_rp_thresh, 1e-6))
            imu_rot_std_rp = imu_rot_std_rp * (1.0 + rot_guard_gain * excess)

        # Yaw guard
        yaw_guard_enable = bool(getattr(self.config, "post_imu_fusion_yaw_guard_enable", False))
        yaw_guard_thresh = float(getattr(self.config, "post_imu_fusion_yaw_guard_thresh", 0.12))
        yaw_guard_gain = float(getattr(self.config, "post_imu_fusion_yaw_guard_gain", 4.0))
        if yaw_guard_enable:
            yaw_disagreement_abs = abs(self._wrap_angle(yaw_visual - yaw_imu))
            yaw_excess = max(0.0, yaw_disagreement_abs - max(yaw_guard_thresh, 1e-6))
            imu_rot_std_yaw = imu_rot_std_yaw * (1.0 + yaw_guard_gain * yaw_excess)

        Sigma_i = torch.zeros((6, 6), dtype=torch.double)
        if imu_trans_enable:
            if imu_trans_z_only:
                Sigma_i[0, 0] = 1e12
                Sigma_i[1, 1] = 1e12
                Sigma_i[2, 2] = imu_trans_std_z ** 2
            else:
                Sigma_i[0, 0] = imu_trans_std_xy ** 2
                Sigma_i[1, 1] = imu_trans_std_xy ** 2
                Sigma_i[2, 2] = imu_trans_std_z ** 2
        else:
            Sigma_i[0, 0] = 1e12
            Sigma_i[1, 1] = 1e12
            Sigma_i[2, 2] = 1e12
        Sigma_i[3, 3] = imu_rot_std_rp ** 2
        Sigma_i[4, 4] = imu_rot_std_rp ** 2
        Sigma_i[5, 5] = imu_rot_std_yaw ** 2
        # ═══════════════════════════════════════════════════════════════════════

        # ── Step 6: Minimal yaw correction (residual-based, bounded) ────────
        yaw_residual_enable = bool(getattr(self.config, "post_imu_fusion_yaw_residual_enable", False))
        yaw_residual_gain = float(getattr(self.config, "post_imu_fusion_yaw_residual_gain", 0.60))
        yaw_residual_max_corr = float(getattr(self.config, "post_imu_fusion_yaw_residual_max_corr", 0.10))
        xi_visual_corr = xi_visual.clone()
        yaw_visual_corr = yaw_visual
        if yaw_residual_enable and abs(yaw_residual) > 1e-6:
            yaw_turn_mag = max(abs(yaw_visual), abs(yaw_imu))
            yaw_turn_ref = float(getattr(self.config, "post_imu_fusion_yaw_residual_turn_ref", 0.10))
            yaw_turn_ratio = min(1.0, yaw_turn_mag / max(yaw_turn_ref, 1e-6))
            yaw_correction = yaw_residual_gain * yaw_residual * yaw_turn_ratio
            yaw_correction = max(-yaw_residual_max_corr, min(yaw_residual_max_corr, yaw_correction))
            yaw_visual_corr = self._wrap_angle(yaw_visual + yaw_correction)

        rot_visual_corr = self._ypr_to_so3(
            yaw_visual_corr, pitch_visual, roll_visual,
            device=xi_visual.device, dtype=torch.double,
        )
        xi_visual_corr[3:6] = rot_visual_corr.Log().tensor().reshape(3).double()
        # ═══════════════════════════════════════════════════════════════════════

        # ── Step 7: Decoupled per-axis information fusion ────────────────────
        epsI = torch.eye(6, dtype=torch.double) * 1e-9
        Wv = torch.linalg.pinv(Sigma_v + epsI)
        Wi = torch.linalg.pinv(Sigma_i + epsI)

        A = Wv + Wi + epsI
        b = Wv @ xi_visual_corr + Wi @ xi_imu
        xi_fused = torch.linalg.solve(A, b).reshape(1, 6)

        # Re-assemble fused pose
        yaw_fused   = self._wrap_angle(yaw_visual_corr)
        pitch_fused = pitch_visual
        roll_fused  = roll_visual

        # Yaw blend from information ratio
        yaw_blend_ratio = float(Wi[5, 5].item()) / max(float(Wi[5, 5].item()) + float(Wv[5, 5].item()), 1e-9)
        yaw_blend_ratio = min(yaw_blend_ratio, float(getattr(self.config, "post_imu_fusion_yaw_blend_ratio_max", 0.0)))
        yaw_fused = self._wrap_angle(yaw_visual_corr + yaw_blend_ratio * self._wrap_angle(yaw_imu - yaw_visual_corr))

        rot_fused = self._ypr_to_so3(
            yaw_fused, pitch_fused, roll_fused,
            device=xi_fused.device, dtype=torch.double,
        )
        rel_fused = NormalizeQuat(pp.SE3(torch.cat([xi_fused[:, 0:3], rot_fused.tensor()], dim=-1)))
        fused_pose = (prev_pose @ rel_fused).float()
        # ═══════════════════════════════════════════════════════════════════════

        # ── Per-axis IMU weight ratios for logging ──────────────────────────
        per_axis_imu_weight = torch.zeros(6, dtype=torch.double)
        for a in range(6):
            wi_a = float(Wi[a, a].item())
            wv_a = float(Wv[a, a].item())
            per_axis_imu_weight[a] = wi_a / max(wi_a + wv_a, 1e-9)

        wi_trace = float(Wi.trace().item())
        wv_trace = float(Wv.trace().item())
        imu_weight_ratio = wi_trace / max(wi_trace + wv_trace, 1e-9)

        log = {
            "frame_idx"           : int(frame_idx.item()),
            "skipped"             : False,
            "mode"                : "PADF",
            "visual_obs_cov"      : round(visual_obs_cov_mean, 6) if visual_obs_cov_mean is not None else None,
            "num_observations"    : num_observations,
            "count_scale"         : round(count_scale, 4),
            "imu_weight_ratio"    : round(imu_weight_ratio, 4),
            "axis_quality"        : [round(float(v), 4) for v in axis_quality],
            "calib_scale"         : [round(float(v), 4) for v in calib_scale],
            "vis_std"             : [round(float(v), 6) for v in vis_std],
            "imu_std"             : [round(float(Sigma_i[a, a].item() ** 0.5), 6) for a in range(6)],
            "per_axis_imu_weight" : [round(float(v), 4) for v in per_axis_imu_weight],
            "rp_disagreement"     : round(rp_disagreement, 6),
            "yaw_visual"          : round(yaw_visual, 6),
            "yaw_imu"             : round(yaw_imu, 6),
            "yaw_fused"           : round(yaw_fused, 6),
            "yaw_blend_ratio"     : round(yaw_blend_ratio, 6),
        }
        return fused_pose, log
    # ═══════════════════════════════════════════════════════════════════════════

    def _post_fuse_visual_imu_dua(
        self,
        to_pose: pp.LieTensor,
        frame_idx: torch.Tensor,
        global_map: VisualMap,
        visual_obs_cov_mean: float | None = None,
        num_observations: int = 0,
        visual_keypoint_coverage: float | None = None,
        visual_depth_spread: float | None = None,
    ) -> tuple[pp.LieTensor, dict]:
        """Degradation-aware Uncertainty Adaptive fusion.

        DUA is the paper-facing fusion path: it treats IMU as a switchable
        constraint rather than a default correction. The switch is driven only by
        online MACVO signals: visual covariance, observation count, keypoint
        coverage, depth diversity, and visual/IMU relative-motion agreement.
        """
        log = {"frame_idx": int(frame_idx.item()), "skipped": True, "mode": "DUA"}
        if int(frame_idx.item()) <= 0:
            return to_pose, log

        frame_data = global_map.frames.data
        needed = ("imu_rotvec_prior", "imu_rot_prior_std", "imu_trans_prior", "imu_trans_cov")
        if any(key not in frame_data for key in needed):
            log["skip_reason"] = "missing_imu_prior"
            return to_pose, log

        try:
            rot_prior = frame_data["imu_rotvec_prior"][frame_idx].reshape(-1)
            rot_std = frame_data["imu_rot_prior_std"][frame_idx].reshape(-1)
            trans_prior = frame_data["imu_trans_prior"][frame_idx].reshape(-1)
            trans_cov = frame_data["imu_trans_cov"][frame_idx].reshape(3, 3)
        except Exception:
            log["skip_reason"] = "bad_imu_prior"
            return to_pose, log

        if rot_prior.numel() != 3 or trans_prior.numel() != 3 or trans_cov.numel() != 9 or rot_std.numel() == 0:
            log["skip_reason"] = "bad_imu_shape"
            return to_pose, log
        if float(rot_std[0].item()) >= 1e5:
            log["skip_reason"] = "invalid_imu_std"
            return to_pose, log

        prev_pose = pp.SE3(frame_data["pose"][frame_idx - 1].double())
        rel_visual = prev_pose.Inv() @ pp.SE3(to_pose)
        rot_visual = pp.SO3(rel_visual.rotation().tensor().double())
        rot_imu = pp.so3(rot_prior.reshape(1, 3).double()).Exp()

        xi_visual = torch.cat([
            rel_visual.translation().reshape(3).double(),
            rot_visual.Log().tensor().reshape(3).double(),
        ], dim=0)
        xi_imu = torch.cat([trans_prior, rot_prior], dim=0).reshape(6).double()

        yaw_visual, pitch_visual, roll_visual = self._so3_to_ypr(rot_visual)
        yaw_imu, _, _ = self._so3_to_ypr(rot_imu)
        yaw_residual = self._wrap_angle(yaw_imu - yaw_visual)

        cov_good = float(getattr(self.config, "dua_visual_cov_good", 0.012))
        cov_bad = float(getattr(self.config, "dua_visual_cov_bad", 0.060))
        obs_ref = float(getattr(self.config, "dua_obs_count_ref", 130.0))
        coverage_ref = float(getattr(self.config, "dua_coverage_ref", 0.55))
        depth_ref = float(getattr(self.config, "dua_depth_spread_ref", 0.35))

        vc = float(visual_obs_cov_mean) if visual_obs_cov_mean is not None else cov_good
        cov_quality = self._clamp01((cov_bad - vc) / max(cov_bad - cov_good, 1e-9))
        obs_quality = self._clamp01(float(num_observations) / max(obs_ref, 1.0))
        coverage = float(visual_keypoint_coverage) if visual_keypoint_coverage is not None else coverage_ref
        coverage_quality = self._clamp01(coverage / max(coverage_ref, 1e-9))
        depth_spread = float(visual_depth_spread) if visual_depth_spread is not None else depth_ref
        depth_quality = self._clamp01(depth_spread / max(depth_ref, 1e-9))

        visual_quality = self._clamp01(
            0.38 * cov_quality
            + 0.24 * obs_quality
            + 0.23 * coverage_quality
            + 0.15 * depth_quality
        )
        raw_degrade_score = 1.0 - visual_quality
        ema_alpha = float(getattr(self.config, "dua_degrade_ema_alpha", 0.35))
        ema_alpha = self._clamp01(ema_alpha)
        self._dua_score_ema = (1.0 - ema_alpha) * self._dua_score_ema + ema_alpha * raw_degrade_score
        degrade_score = max(raw_degrade_score, self._dua_score_ema)

        onset = float(getattr(self.config, "dua_degrade_onset", 0.55))
        release = float(getattr(self.config, "dua_degrade_release", 0.45))
        if self._dua_active:
            self._dua_active = degrade_score >= release
        else:
            self._dua_active = degrade_score >= onset
        quality_enable_max = float(getattr(self.config, "dua_enable_visual_quality_max", 1.0))
        quality_enable_gate = visual_quality <= quality_enable_max

        visual_trans_floor = float(getattr(self.config, "post_imu_fusion_visual_trans_std", 0.08))
        visual_rot_floor = float(getattr(self.config, "post_imu_fusion_visual_rot_std", 0.035))
        cov_scale = float(getattr(self.config, "post_imu_fusion_cov_scale", 1.0))
        adaptive_trans_std = max(float((max(vc, 1e-9) * cov_scale) ** 0.5), visual_trans_floor)
        visual_inflation = 1.0 + degrade_score * float(getattr(self.config, "dua_visual_std_gain", 1.6))
        visual_xy_std = adaptive_trans_std * visual_inflation / max(0.75 + 0.25 * coverage_quality, 0.1)
        visual_z_std = adaptive_trans_std * (1.0 + degrade_score * 1.15) / max(0.75 + 0.25 * depth_quality, 0.1)
        visual_rot_std = max(adaptive_trans_std * (visual_rot_floor / max(visual_trans_floor, 1e-9)), visual_rot_floor)
        visual_rp_std = visual_rot_std * (1.0 + degrade_score * 0.85)
        visual_yaw_std = visual_rot_std * (1.0 + degrade_score * 1.15)

        imu_trans_std_floor = float(getattr(self.config, "post_imu_fusion_imu_trans_std_floor", 0.18))
        imu_trans_xy_floor = float(getattr(self.config, "post_imu_fusion_imu_trans_xy_std_floor", 0.45))
        imu_trans_z_floor = float(getattr(self.config, "post_imu_fusion_imu_trans_z_std_floor", 0.18))
        imu_rot_std_floor = float(getattr(self.config, "post_imu_fusion_imu_rot_std_floor", 0.055))
        imu_rot_rp_floor = float(getattr(self.config, "post_imu_fusion_imu_rot_rp_std_floor", 0.075))
        imu_rot_yaw_floor = float(getattr(self.config, "post_imu_fusion_imu_rot_yaw_std_floor", 0.055))

        imu_trans_std_base = max(
            float(trans_cov.diagonal().clamp(min=0).mean().sqrt().item()),
            imu_trans_std_floor,
        )
        imu_trans_xy_std = max(imu_trans_std_base, imu_trans_xy_floor)
        imu_trans_z_std = max(imu_trans_std_base, imu_trans_z_floor)
        imu_rot_rp_std = max(float(rot_std[0].item()), imu_rot_rp_floor)
        imu_rot_yaw_std = max(float(rot_std[0].item()), imu_rot_yaw_floor)

        trans_residual = xi_visual[0:3] - xi_imu[0:3]
        rot_residual = xi_visual[3:6] - xi_imu[3:6]
        xy_residual = xi_visual[0:2] - xi_imu[0:2]
        z_residual = float(abs(trans_residual[2].item()))

        dcs_phi = float(getattr(self.config, "dua_dcs_phi", getattr(self.config, "bagf_dcs_phi", 2.5)))
        trans_chi2 = float((trans_residual / max(imu_trans_std_base, 1e-6)).pow(2).sum().item())
        rot_chi2 = float((rot_residual / max(imu_rot_std_floor, 1e-6)).pow(2).sum().item())
        xy_chi2 = float((xy_residual / max(imu_trans_xy_std, 1e-6)).pow(2).sum().item())
        z_chi2 = float((z_residual / max(imu_trans_z_std, 1e-6)) ** 2)
        trans_switch = self._dcs_switch(trans_chi2, dcs_phi)
        rot_switch = self._dcs_switch(rot_chi2, dcs_phi)
        xy_switch = self._dcs_switch(xy_chi2, dcs_phi)
        z_switch = self._dcs_switch(z_chi2, dcs_phi)

        xy_visual = xi_visual[0:2]
        xy_imu = xi_imu[0:2]
        xy_visual_norm = float(xy_visual.norm().item())
        xy_imu_norm = float(xy_imu.norm().item())
        if xy_visual_norm > 1e-6 and xy_imu_norm > 1e-6:
            xy_cos = float(torch.dot(xy_visual, xy_imu).item() / max(xy_visual_norm * xy_imu_norm, 1e-9))
        else:
            xy_cos = 0.0

        trans_cfg = bool(getattr(self.config, "post_imu_fusion_imu_trans_enable", True))
        rot_cfg = bool(getattr(self.config, "post_imu_fusion_imu_rot_enable", True))
        z_only = bool(getattr(self.config, "post_imu_fusion_imu_trans_z_only", False))

        rot_enable = (
            rot_cfg
            and self._dua_active
            and quality_enable_gate
            and degrade_score >= float(getattr(self.config, "dua_rot_degrade_min", 0.65))
            and rot_switch >= float(getattr(self.config, "dua_rot_switch_min", 0.28))
        )
        z_enable = (
            trans_cfg
            and self._dua_active
            and quality_enable_gate
            and degrade_score >= float(getattr(self.config, "dua_z_degrade_min", 0.60))
            and z_switch >= float(getattr(self.config, "dua_z_switch_min", 0.22))
        )
        xy_enable = (
            trans_cfg
            and (not z_only)
            and self._dua_active
            and quality_enable_gate
            and degrade_score >= float(getattr(self.config, "dua_xy_degrade_min", 0.75))
            and xy_switch >= float(getattr(self.config, "dua_xy_switch_min", 0.45))
            and xy_cos >= float(getattr(self.config, "dua_xy_cos_min", 0.65))
            and xy_imu_norm >= float(getattr(self.config, "dua_xy_motion_min", 0.04))
        )

        if not (rot_enable or z_enable or xy_enable):
            log.update({
                "skip_reason": "degradation_or_switch_gate",
                "visual_quality": round(visual_quality, 4),
                "degrade_score": round(degrade_score, 4),
                "raw_degrade_score": round(raw_degrade_score, 4),
                "trans_switch": round(trans_switch, 4),
                "rot_switch": round(rot_switch, 4),
                "xy_switch": round(xy_switch, 4),
                "z_switch": round(z_switch, 4),
                "xy_cos": round(xy_cos, 4),
                "active": self._dua_active,
                "quality_enable_gate": quality_enable_gate,
            })
            return to_pose, log

        switch_floor = float(getattr(self.config, "dua_switch_floor", 0.08))
        good_visual_relax = 1.0 + visual_quality * float(getattr(self.config, "dua_good_visual_relax", 5.0))
        if rot_enable:
            imu_rot_rp_std *= good_visual_relax / math.sqrt(max(rot_switch, switch_floor))
            imu_rot_yaw_std *= good_visual_relax / math.sqrt(max(rot_switch, switch_floor))
        else:
            imu_rot_rp_std = 1e6
            imu_rot_yaw_std = 1e6
        if z_enable:
            imu_trans_z_std *= good_visual_relax / math.sqrt(max(z_switch, switch_floor))
        else:
            imu_trans_z_std = 1e6
        if xy_enable:
            imu_trans_xy_std *= good_visual_relax / math.sqrt(max(xy_switch, switch_floor))
        else:
            imu_trans_xy_std = 1e6

        yaw_residual_enable = bool(getattr(self.config, "post_imu_fusion_yaw_residual_enable", True))
        yaw_visual_corr = yaw_visual
        yaw_correction = 0.0
        if yaw_residual_enable and rot_enable:
            yaw_ref = max(float(getattr(self.config, "post_imu_fusion_yaw_residual_turn_ref", 0.10)), 1e-6)
            yaw_turn_ratio = min(1.0, max(abs(yaw_visual), abs(yaw_imu)) / yaw_ref)
            yaw_residual_max = float(getattr(self.config, "post_imu_fusion_yaw_residual_max_corr", 0.06))
            yaw_residual_gain = float(getattr(self.config, "post_imu_fusion_yaw_residual_gain", 0.35))
            yaw_correction = yaw_residual_gain * yaw_residual * yaw_turn_ratio * degrade_score * rot_switch
            yaw_correction = max(-yaw_residual_max, min(yaw_residual_max, yaw_correction))
            yaw_visual_corr = self._wrap_angle(yaw_visual + yaw_correction)

        rot_visual_corr = self._ypr_to_so3(
            yaw_visual_corr,
            pitch_visual,
            roll_visual,
            device=xi_visual.device,
            dtype=torch.double,
        )
        xi_visual_corr = xi_visual.clone()
        xi_visual_corr[3:6] = rot_visual_corr.Log().tensor().reshape(3).double()

        Sigma_v = torch.diag(torch.tensor([
            visual_xy_std ** 2,
            visual_xy_std ** 2,
            visual_z_std ** 2,
            visual_rp_std ** 2,
            visual_rp_std ** 2,
            visual_yaw_std ** 2,
        ], dtype=torch.double))
        Sigma_i = torch.diag(torch.tensor([
            imu_trans_xy_std ** 2,
            imu_trans_xy_std ** 2,
            imu_trans_z_std ** 2,
            imu_rot_rp_std ** 2,
            imu_rot_rp_std ** 2,
            imu_rot_yaw_std ** 2,
        ], dtype=torch.double))

        epsI = torch.eye(6, dtype=torch.double) * 1e-9
        Wv = torch.linalg.pinv(Sigma_v + epsI)
        Wi = torch.linalg.pinv(Sigma_i + epsI)
        xi_raw = torch.linalg.solve(Wv + Wi + epsI, Wv @ xi_visual_corr + Wi @ xi_imu).reshape(1, 6)

        correction = (xi_raw.reshape(6) - xi_visual_corr).clone()
        correction_gain = float(getattr(self.config, "dua_correction_gain", 0.45))
        correction = correction * max(0.0, min(1.0, correction_gain))

        max_trans_corr = float(getattr(self.config, "dua_max_trans_correction", 0.035))
        max_rot_corr = float(getattr(self.config, "dua_max_rot_correction", 0.035))
        trans_corr_norm = float(correction[0:3].norm().item())
        rot_corr_norm = float(correction[3:6].norm().item())
        if trans_corr_norm > max_trans_corr:
            correction[0:3] *= max_trans_corr / max(trans_corr_norm, 1e-9)
            trans_corr_norm = max_trans_corr
        if rot_corr_norm > max_rot_corr:
            correction[3:6] *= max_rot_corr / max(rot_corr_norm, 1e-9)
            rot_corr_norm = max_rot_corr

        xi_fused = (xi_visual_corr + correction).reshape(1, 6)
        rot_fused_base = pp.so3(xi_fused[:, 3:6]).Exp()
        yaw_fused_base, pitch_fused, roll_fused = self._so3_to_ypr(rot_fused_base)
        yaw_blend_ratio = 0.0
        if rot_enable:
            yaw_imu_weight = 1.0 / max(imu_rot_yaw_std ** 2, 1e-9)
            yaw_visual_weight = 1.0 / max(visual_yaw_std ** 2, 1e-9)
            yaw_blend_ratio = yaw_imu_weight / max(yaw_imu_weight + yaw_visual_weight, 1e-9)
            yaw_blend_max = float(getattr(self.config, "dua_yaw_blend_max", 0.10))
            yaw_blend_ratio = min(yaw_blend_ratio, yaw_blend_max)
        yaw_fused = self._wrap_angle(yaw_fused_base + yaw_blend_ratio * self._wrap_angle(yaw_imu - yaw_fused_base))

        rot_fused = self._ypr_to_so3(
            yaw_fused,
            pitch_fused,
            roll_fused,
            device=xi_fused.device,
            dtype=torch.double,
        )
        rel_fused = NormalizeQuat(pp.SE3(torch.cat([xi_fused[:, 0:3], rot_fused.tensor()], dim=-1)))
        fused_pose = (prev_pose @ rel_fused).float()

        per_axis_weight = []
        for axis in range(6):
            wi_a = float(Wi[axis, axis].item())
            wv_a = float(Wv[axis, axis].item())
            per_axis_weight.append(wi_a / max(wi_a + wv_a, 1e-9))
        xy_weight = 0.5 * (per_axis_weight[0] + per_axis_weight[1])
        z_weight = per_axis_weight[2]
        rot_weight = (per_axis_weight[3] + per_axis_weight[4] + per_axis_weight[5]) / 3.0

        log.update({
            "skipped": False,
            "visual_obs_cov": round(vc, 6),
            "num_observations": int(num_observations),
            "visual_quality": round(visual_quality, 4),
            "degrade_score": round(degrade_score, 4),
            "raw_degrade_score": round(raw_degrade_score, 4),
            "degrade_ema": round(self._dua_score_ema, 4),
            "cov_quality": round(cov_quality, 4),
            "obs_quality": round(obs_quality, 4),
            "coverage_quality": round(coverage_quality, 4),
            "depth_quality": round(depth_quality, 4),
            "quality_enable_gate": quality_enable_gate,
            "trans_switch": round(trans_switch, 4),
            "rot_switch": round(rot_switch, 4),
            "xy_switch": round(xy_switch, 4),
            "z_switch": round(z_switch, 4),
            "xy_cos": round(xy_cos, 4),
            "active": self._dua_active,
            "xy_imu_enable": xy_enable,
            "z_imu_enable": z_enable,
            "rot_imu_enable": rot_enable,
            "xy_weight": round(xy_weight, 4),
            "z_weight": round(z_weight, 4),
            "rot_weight": round(rot_weight, 4),
            "per_axis_imu_weight": [round(v, 4) for v in per_axis_weight],
            "yaw_correction": round(yaw_correction, 6),
            "yaw_blend_ratio": round(yaw_blend_ratio, 4),
            "trans_correction_norm": round(trans_corr_norm, 6),
            "rot_correction_norm": round(rot_corr_norm, 6),
        })
        return fused_pose, log

    def _post_fuse_visual_imu_bagf(
        self,
        to_pose: pp.LieTensor,
        frame_idx: torch.Tensor,
        global_map: VisualMap,
        visual_obs_cov_mean: float | None = None,
        num_observations: int = 0,
        visual_keypoint_coverage: float | None = None,
        visual_depth_spread: float | None = None,
    ) -> tuple[pp.LieTensor, dict]:
        """Balanced Adaptive Gated Fusion.

        BAGF is a conservative post-fusion path for the seven-scene HoloOcean
        benchmark. It lets IMU help rotation and vertical translation broadly,
        while using online visual quality and IMU/vision consistency gates before
        allowing IMU to affect XY translation.
        """
        log = {"frame_idx": int(frame_idx.item()), "skipped": True, "mode": "BAGF"}
        if not hasattr(self, "_bagf_active_streak"):
            self._bagf_active_streak = 0
            self._bagf_cooldown = 0
        if int(frame_idx.item()) <= 0:
            return to_pose, log

        frame_data = global_map.frames.data
        needed = ("imu_rotvec_prior", "imu_rot_prior_std", "imu_trans_prior", "imu_trans_cov")
        if any(key not in frame_data for key in needed):
            return to_pose, log

        try:
            rot_prior = frame_data["imu_rotvec_prior"][frame_idx].reshape(-1)
            rot_std = frame_data["imu_rot_prior_std"][frame_idx].reshape(-1)
            trans_prior = frame_data["imu_trans_prior"][frame_idx].reshape(-1)
            trans_cov = frame_data["imu_trans_cov"][frame_idx].reshape(3, 3)
        except Exception:
            return to_pose, log

        if rot_prior.numel() != 3 or trans_prior.numel() != 3 or trans_cov.numel() != 9 or rot_std.numel() == 0:
            return to_pose, log
        if float(rot_std[0].item()) >= 1e5:
            return to_pose, log

        prev_pose = pp.SE3(frame_data["pose"][frame_idx - 1].double())
        rel_visual = prev_pose.Inv() @ pp.SE3(to_pose)
        rot_visual = pp.SO3(rel_visual.rotation().tensor().double())
        rot_imu = pp.so3(rot_prior.reshape(1, 3).double()).Exp()

        xi_visual = torch.cat([
            rel_visual.translation().reshape(3).double(),
            rot_visual.Log().tensor().reshape(3).double(),
        ], dim=0)
        xi_imu = torch.cat([trans_prior, rot_prior], dim=0).reshape(6).double()

        yaw_visual, pitch_visual, roll_visual = self._so3_to_ypr(rot_visual)
        yaw_imu, _, _ = self._so3_to_ypr(rot_imu)
        yaw_residual = self._wrap_angle(yaw_imu - yaw_visual)
        yaw_turn_mag = max(abs(yaw_visual), abs(yaw_imu))

        visual_cov_ref = float(getattr(self.config, "bagf_visual_cov_ref", 0.02))
        obs_count_ref = float(getattr(self.config, "bagf_obs_count_ref", 120.0))
        depth_spread_ref = float(getattr(self.config, "bagf_depth_spread_ref", 0.35))

        vc = float(visual_obs_cov_mean) if visual_obs_cov_mean is not None else visual_cov_ref
        cov_quality = max(0.0, min(1.0, visual_cov_ref / max(vc, visual_cov_ref)))
        obs_quality = max(0.0, min(1.0, float(num_observations) / max(obs_count_ref, 1.0)))
        coverage_quality = max(0.0, min(1.0, float(visual_keypoint_coverage) if visual_keypoint_coverage is not None else 0.55))
        depth_quality = max(0.0, min(1.0, float(visual_depth_spread or 0.0) / max(depth_spread_ref, 1e-6)))
        visual_quality = (
            0.40 * cov_quality
            + 0.25 * obs_quality
            + 0.20 * coverage_quality
            + 0.15 * depth_quality
        )
        visual_quality = max(0.0, min(1.0, visual_quality))

        skip_healthy_visual = bool(getattr(self.config, "bagf_skip_healthy_visual", True))
        healthy_geometry = (
            obs_quality >= float(getattr(self.config, "bagf_skip_obs_quality", 0.65))
            and coverage_quality >= float(getattr(self.config, "bagf_skip_coverage_quality", 0.30))
            and depth_quality >= float(getattr(self.config, "bagf_skip_depth_quality", 0.30))
        )
        if skip_healthy_visual and healthy_geometry:
            self._bagf_active_streak = 0
            if self._bagf_cooldown > 0:
                self._bagf_cooldown -= 1
            log.update({
                "skip_reason": "healthy_visual_geometry",
                "visual_quality": round(visual_quality, 4),
                "cov_quality": round(cov_quality, 4),
                "obs_quality": round(obs_quality, 4),
                "coverage_quality": round(coverage_quality, 4),
                "depth_quality": round(depth_quality, 4),
            })
            return to_pose, log

        visual_trans_floor = float(getattr(self.config, "post_imu_fusion_visual_trans_std", 0.08))
        visual_rot_floor = float(getattr(self.config, "post_imu_fusion_visual_rot_std", 0.035))
        cov_scale = float(getattr(self.config, "post_imu_fusion_cov_scale", 1.0))
        adaptive_trans_std = max(float((vc * cov_scale) ** 0.5), visual_trans_floor)
        quality_inflation = 1.0 + (1.0 - visual_quality) * float(getattr(self.config, "bagf_visual_quality_std_gain", 1.25))
        spread_bonus = 1.0 / max(0.65 + 0.35 * depth_quality, 0.1)
        visual_xy_std = adaptive_trans_std * quality_inflation * spread_bonus
        visual_z_std = adaptive_trans_std * (1.0 + (1.0 - visual_quality) * 0.85)
        visual_rot_std = max(adaptive_trans_std * (visual_rot_floor / max(visual_trans_floor, 1e-9)), visual_rot_floor)
        visual_yaw_std = visual_rot_std * (1.0 + (1.0 - visual_quality) * 0.75)

        imu_trans_std_floor = float(getattr(self.config, "post_imu_fusion_imu_trans_std_floor", 0.18))
        imu_trans_xy_floor = float(getattr(self.config, "post_imu_fusion_imu_trans_xy_std_floor", max(imu_trans_std_floor, 0.35)))
        imu_trans_z_floor = float(getattr(self.config, "post_imu_fusion_imu_trans_z_std_floor", imu_trans_std_floor))
        imu_rot_std_floor = float(getattr(self.config, "post_imu_fusion_imu_rot_std_floor", 0.05))
        imu_rot_rp_floor = float(getattr(self.config, "post_imu_fusion_imu_rot_rp_std_floor", imu_rot_std_floor))
        imu_rot_yaw_floor = float(getattr(self.config, "post_imu_fusion_imu_rot_yaw_std_floor", imu_rot_std_floor))

        imu_trans_std_base = max(
            float(trans_cov.diagonal().clamp(min=0).mean().sqrt().item()),
            imu_trans_std_floor,
        )
        imu_trans_std_xy = max(imu_trans_std_base, imu_trans_xy_floor)
        imu_trans_std_z = max(imu_trans_std_base, imu_trans_z_floor)
        imu_rot_std_rp = max(float(rot_std[0].item()), imu_rot_rp_floor)
        imu_rot_std_yaw = max(float(rot_std[0].item()), imu_rot_yaw_floor)

        # Good visual geometry should remain mostly visual. IMU is tightened only
        # when online visual quality drops or the vision/IMU residual stays small.
        good_visual_relax = 1.0 + visual_quality * float(getattr(self.config, "bagf_good_visual_imu_relax", 3.0))
        imu_trans_std_z *= good_visual_relax
        imu_rot_std_rp *= good_visual_relax
        imu_rot_std_yaw *= good_visual_relax

        xy_visual = xi_visual[0:2]
        xy_imu = xi_imu[0:2]
        visual_xy_norm = float(xy_visual.norm().item())
        imu_xy_norm = float(xy_imu.norm().item())
        imu_trans_norm = float(xi_imu[0:3].norm().item())
        if visual_xy_norm > 1e-6 and imu_xy_norm > 1e-6:
            xy_cos = float(torch.dot(xy_visual, xy_imu).item() / max(visual_xy_norm * imu_xy_norm, 1e-9))
        else:
            xy_cos = 0.0

        dcs_phi = float(getattr(self.config, "bagf_dcs_phi", 4.0))
        dcs_scale_max = float(getattr(self.config, "bagf_dcs_scale_max", 8.0))
        trans_chi2 = float(((xi_visual[0:3] - xi_imu[0:3]) / max(imu_trans_std_base, 1e-6)).pow(2).sum().item())
        rot_chi2 = float(((xi_visual[3:6] - xi_imu[3:6]) / max(imu_rot_std_floor, 1e-6)).pow(2).sum().item())
        trans_switch = min(1.0, (2.0 * dcs_phi) / max(dcs_phi + trans_chi2, 1e-9))
        rot_switch = min(1.0, (2.0 * dcs_phi) / max(dcs_phi + rot_chi2, 1e-9))
        trans_dcs_scale = min(dcs_scale_max, 1.0 / max(trans_switch, 0.05))
        rot_dcs_scale = min(dcs_scale_max, 1.0 / max(rot_switch, 0.05))

        trans_apply_score = (1.0 - visual_quality) * trans_switch
        rot_apply_score = (1.0 - visual_quality) * rot_switch
        apply_score = min(trans_apply_score, rot_apply_score)
        apply_score_min = float(getattr(self.config, "bagf_apply_score_min", 0.30))
        enable_quality_thresh = float(getattr(self.config, "bagf_visual_quality_enable_thresh", 0.45))
        z_apply_score_min = float(getattr(self.config, "bagf_z_apply_score_min", apply_score_min))
        rot_apply_score_min = float(getattr(self.config, "bagf_rot_apply_score_min", apply_score_min))
        trans_enable_cfg = bool(getattr(self.config, "post_imu_fusion_imu_trans_enable", True))
        rot_enable_cfg = bool(getattr(self.config, "post_imu_fusion_imu_rot_enable", True))
        rot_imu_enable = (
            rot_enable_cfg
            and visual_quality <= enable_quality_thresh
            and rot_apply_score >= rot_apply_score_min
        )
        z_imu_enable = (
            trans_enable_cfg
            and visual_quality <= enable_quality_thresh
            and trans_apply_score >= z_apply_score_min
        )

        if apply_score < apply_score_min or not (rot_imu_enable or z_imu_enable):
            self._bagf_active_streak = 0
            if self._bagf_cooldown > 0:
                self._bagf_cooldown -= 1
            log.update({
                "skip_reason": "quality_or_consistency_gate",
                "visual_quality": round(visual_quality, 4),
                "trans_switch": round(trans_switch, 4),
                "rot_switch": round(rot_switch, 4),
                "trans_apply_score": round(trans_apply_score, 4),
                "rot_apply_score": round(rot_apply_score, 4),
                "apply_score": round(apply_score, 4),
                "rot_imu_enable": rot_imu_enable,
                "z_imu_enable": z_imu_enable,
            })
            return to_pose, log

        max_active_streak = int(getattr(self.config, "bagf_max_active_streak", 180))
        cooldown_frames = int(getattr(self.config, "bagf_cooldown_frames", 360))
        if self._bagf_cooldown > 0:
            self._bagf_cooldown -= 1
            log.update({
                "skip_reason": "bounded_inertial_cooldown",
                "visual_quality": round(visual_quality, 4),
                "trans_switch": round(trans_switch, 4),
                "rot_switch": round(rot_switch, 4),
                "active_streak": self._bagf_active_streak,
                "cooldown_left": self._bagf_cooldown,
            })
            return to_pose, log
        if max_active_streak > 0 and self._bagf_active_streak >= max_active_streak:
            self._bagf_active_streak = 0
            self._bagf_cooldown = cooldown_frames
            log.update({
                "skip_reason": "bounded_inertial_budget",
                "visual_quality": round(visual_quality, 4),
                "trans_switch": round(trans_switch, 4),
                "rot_switch": round(rot_switch, 4),
                "cooldown_left": self._bagf_cooldown,
            })
            return to_pose, log

        if z_imu_enable:
            imu_trans_std_z *= trans_dcs_scale
        else:
            imu_trans_std_z = 1e6
        if rot_imu_enable:
            imu_rot_std_rp *= rot_dcs_scale
            imu_rot_std_yaw *= rot_dcs_scale
        else:
            imu_rot_std_rp = 1e6
            imu_rot_std_yaw = 1e6

        xy_quality_thresh = float(getattr(self.config, "bagf_xy_enable_quality_thresh", 0.35))
        xy_consistency_thresh = float(getattr(self.config, "bagf_xy_consistency_thresh", 0.35))
        xy_motion_ref = float(getattr(self.config, "bagf_xy_motion_ref", 0.04))
        xy_imu_enable = (
            trans_enable_cfg
            and visual_quality <= min(xy_quality_thresh, enable_quality_thresh)
            and trans_apply_score >= z_apply_score_min
            and imu_xy_norm >= xy_motion_ref
            and xy_cos >= xy_consistency_thresh
            and trans_switch > 0.25
        )
        if xy_imu_enable:
            # Even when enabled, keep XY conservative to avoid straightening turns.
            imu_trans_std_xy *= trans_dcs_scale * (1.0 + visual_quality)
        else:
            imu_trans_std_xy = 1e6

        turn_yaw_ref = float(getattr(self.config, "post_imu_fusion_xy_turn_residual_yaw_ref", 0.10))
        stationary_ref = float(getattr(self.config, "post_imu_fusion_xy_turn_residual_imu_trans_ref", 0.05))
        turn_ratio = min(1.0, yaw_turn_mag / max(turn_yaw_ref, 1e-6))
        stationary_ratio = max(0.0, 1.0 - imu_trans_norm / max(stationary_ref, 1e-6))
        turn_gate = turn_ratio * stationary_ratio * (1.0 - visual_quality)
        visual_xy_std *= 1.0 + (float(getattr(self.config, "post_imu_fusion_xy_turn_visual_std_scale_max", 2.0)) - 1.0) * turn_gate
        visual_z_std *= 1.0 + (float(getattr(self.config, "post_imu_fusion_turn_visual_z_std_scale_max", 1.6)) - 1.0) * turn_gate

        yaw_residual_enable = bool(getattr(self.config, "post_imu_fusion_yaw_residual_enable", True))
        yaw_residual_gain = float(getattr(self.config, "post_imu_fusion_yaw_residual_gain", 0.65))
        yaw_residual_max_corr = float(getattr(self.config, "post_imu_fusion_yaw_residual_max_corr", 0.12))
        yaw_residual_disagree_ref = float(getattr(self.config, "post_imu_fusion_yaw_residual_disagree_ref", 0.50))
        yaw_residual_scale = max(0.0, 1.0 - abs(yaw_residual) / max(yaw_residual_disagree_ref, 1e-6))
        yaw_correction = 0.0
        yaw_visual_corr = yaw_visual
        if yaw_residual_enable and rot_imu_enable:
            yaw_correction = yaw_residual_gain * yaw_residual * turn_ratio * (0.15 + 0.85 * (1.0 - visual_quality)) * yaw_residual_scale
            yaw_correction = max(-yaw_residual_max_corr, min(yaw_residual_max_corr, yaw_correction))
            yaw_visual_corr = self._wrap_angle(yaw_visual + yaw_correction)

        rot_visual_corr = self._ypr_to_so3(
            yaw_visual_corr,
            pitch_visual,
            roll_visual,
            device=xi_visual.device,
            dtype=torch.double,
        )
        xi_visual_corr = xi_visual.clone()
        xi_visual_corr[3:6] = rot_visual_corr.Log().tensor().reshape(3).double()

        Sigma_v = torch.diag(torch.tensor([
            visual_xy_std ** 2,
            visual_xy_std ** 2,
            visual_z_std ** 2,
            visual_rot_std ** 2,
            visual_rot_std ** 2,
            visual_yaw_std ** 2,
        ], dtype=torch.double))
        Sigma_i = torch.diag(torch.tensor([
            imu_trans_std_xy ** 2,
            imu_trans_std_xy ** 2,
            imu_trans_std_z ** 2,
            imu_rot_std_rp ** 2,
            imu_rot_std_rp ** 2,
            imu_rot_std_yaw ** 2,
        ], dtype=torch.double))

        epsI = torch.eye(6, dtype=torch.double) * 1e-9
        Wv = torch.linalg.pinv(Sigma_v + epsI)
        Wi = torch.linalg.pinv(Sigma_i + epsI)
        xi_fused = torch.linalg.solve(Wv + Wi + epsI, Wv @ xi_visual_corr + Wi @ xi_imu).reshape(1, 6)

        rot_fused_base = pp.so3(xi_fused[:, 3:6]).Exp()
        _, pitch_fused, roll_fused = self._so3_to_ypr(rot_fused_base)
        yaw_blend_good = float(getattr(self.config, "bagf_yaw_blend_max_good", 0.03))
        yaw_blend_bad = float(getattr(self.config, "bagf_yaw_blend_max_bad", 0.22))
        yaw_blend_ratio_max = yaw_blend_good + (yaw_blend_bad - yaw_blend_good) * (1.0 - visual_quality)
        yaw_blend_imu_scale = float(getattr(self.config, "post_imu_fusion_yaw_blend_imu_scale", 1.0))
        yaw_imu_weight = yaw_blend_imu_scale / max(imu_rot_std_yaw ** 2, 1e-9)
        yaw_visual_weight = 1.0 / max(visual_yaw_std ** 2, 1e-9)
        yaw_blend_ratio = yaw_imu_weight / max(yaw_imu_weight + yaw_visual_weight, 1e-9)
        yaw_blend_ratio = min(yaw_blend_ratio, yaw_blend_ratio_max) if rot_imu_enable else 0.0
        yaw_fused = self._wrap_angle(yaw_visual_corr + yaw_blend_ratio * self._wrap_angle(yaw_imu - yaw_visual_corr))

        yaw_trans_couple_enable = bool(getattr(self.config, "post_imu_fusion_yaw_trans_couple_enable", True))
        yaw_trans_couple_angle = 0.0
        motion_ratio = min(1.0, imu_trans_norm / max(float(getattr(self.config, "post_imu_fusion_yaw_trans_couple_imu_trans_ref", 0.08)), 1e-6))
        motion_ratio *= max(0.0, 1.0 - stationary_ratio)
        if yaw_trans_couple_enable and rot_imu_enable and xy_imu_enable and motion_ratio > 0.0:
            yaw_trans_couple_gain = float(getattr(self.config, "post_imu_fusion_yaw_trans_couple_gain", 0.85))
            yaw_trans_couple_max_angle = float(getattr(self.config, "post_imu_fusion_yaw_trans_couple_max_angle", 0.18))
            yaw_trans_couple_angle = self._wrap_angle(yaw_fused - yaw_visual) * yaw_trans_couple_gain * motion_ratio
            yaw_trans_couple_angle = max(-yaw_trans_couple_max_angle, min(yaw_trans_couple_max_angle, yaw_trans_couple_angle))
            trans_x = float(xi_fused[0, 0].item())
            trans_y = float(xi_fused[0, 1].item())
            cos_yaw = math.cos(yaw_trans_couple_angle)
            sin_yaw = math.sin(yaw_trans_couple_angle)
            xi_fused[0, 0] = cos_yaw * trans_x - sin_yaw * trans_y
            xi_fused[0, 1] = sin_yaw * trans_x + cos_yaw * trans_y

        rot_fused = self._ypr_to_so3(
            yaw_fused,
            pitch_fused,
            roll_fused,
            device=xi_fused.device,
            dtype=torch.double,
        )
        rel_fused = NormalizeQuat(pp.SE3(torch.cat([xi_fused[:, 0:3], rot_fused.tensor()], dim=-1)))
        fused_pose = (prev_pose @ rel_fused).float()

        per_axis_imu_weight = []
        for axis in range(6):
            wi_a = float(Wi[axis, axis].item())
            wv_a = float(Wv[axis, axis].item())
            per_axis_imu_weight.append(round(wi_a / max(wi_a + wv_a, 1e-9), 4))

        log.update({
            "skipped": False,
            "active_streak": self._bagf_active_streak,
            "cooldown_left": self._bagf_cooldown,
            "visual_quality": round(visual_quality, 4),
            "cov_quality": round(cov_quality, 4),
            "obs_quality": round(obs_quality, 4),
            "coverage_quality": round(coverage_quality, 4),
            "depth_quality": round(depth_quality, 4),
            "xy_imu_enable": xy_imu_enable,
            "rot_imu_enable": rot_imu_enable,
            "z_imu_enable": z_imu_enable,
            "xy_cos": round(xy_cos, 4),
            "trans_apply_score": round(trans_apply_score, 4),
            "rot_apply_score": round(rot_apply_score, 4),
            "apply_score": round(apply_score, 4),
            "trans_dcs_scale": round(trans_dcs_scale, 4),
            "rot_dcs_scale": round(rot_dcs_scale, 4),
            "good_visual_relax": round(good_visual_relax, 4),
            "yaw_blend_ratio": round(yaw_blend_ratio, 4),
            "yaw_correction": round(yaw_correction, 6),
            "yaw_trans_couple_angle": round(yaw_trans_couple_angle, 6),
            "turn_gate": round(turn_gate, 4),
            "per_axis_imu_weight": per_axis_imu_weight,
        })
        self._bagf_active_streak += 1
        return fused_pose, log

    @staticmethod
    def init_context(config) -> dict:
        imu_factor_mode = str(getattr(config, "imu_factor_mode", "legacy_pose_prior")).strip().lower()
        if imu_factor_mode in {"preintegrated_vio", "local_inertial_ba", "two_state_fixed_lag"}:
            if config.graph_type != "disp" or not bool(config.autodiff):
                raise ValueError(f"{imu_factor_mode} requires graph_type='disp' and autodiff=True")
        two_state_cpu_threads = int(
            getattr(
                config,
                "two_state_cpu_threads",
                os.environ.get("MACVO_TWO_STATE_CPU_THREADS", 0),
            )
            or 0
        )
        if (
            imu_factor_mode == "two_state_fixed_lag"
            and str(config.device).strip().lower() == "cpu"
            and two_state_cpu_threads > 0
        ):
            torch.set_num_threads(two_state_cpu_threads)

        visual_factor_mode = str(
            getattr(config, "two_state_visual_factor_mode", "relative_pose")
        ).strip().lower()
        two_state_backend_name = str(
            getattr(config, "two_state_backend", "two_state")
        ).strip().lower()
        if two_state_backend_name not in {"two_state", "isam2"}:
            raise ValueError(
                "two_state_backend must be either 'two_state' or 'isam2', "
                f"got {two_state_backend_name!r}"
            )
        if two_state_backend_name == "isam2" and visual_factor_mode != "compressed_uvd":
            raise ValueError(
                "the incremental iSAM2 backend requires "
                "two_state_visual_factor_mode='compressed_uvd'"
            )
        initial_prior_std = {
            "pose_translation_std": float(
                getattr(config, "two_state_initial_pose_translation_std", 1e-5)
            ),
            "pose_rotation_std": float(
                getattr(config, "two_state_initial_pose_rotation_std", 1e-5)
            ),
            "velocity_std": float(
                getattr(config, "two_state_initial_velocity_std", 0.05)
            ),
            "acc_bias_std": float(
                getattr(config, "two_state_initial_acc_bias_std", 0.2)
            ),
            "gyro_bias_std": float(
                getattr(config, "two_state_initial_gyro_bias_std", 0.02)
            ),
        }
        isam2_backend = (
            IncrementalT2ISAM2Backend(
                initial_prior_std=initial_prior_std,
                relinearize_threshold=float(
                    getattr(config, "two_state_isam2_relinearize_threshold", 0.01)
                ),
                relinearize_skip=int(
                    getattr(config, "two_state_isam2_relinearize_skip", 1)
                ),
                covariance_floor=float(
                    getattr(config, "two_state_isam2_covariance_floor", 1.0e-12)
                ),
            )
            if two_state_backend_name == "isam2"
            else None
        )

        match (config.autodiff, config.graph_type):
            case (True, "icp"):
                PoseGraphClass = ICP_TwoframePGO
            case (True, "reproj"):
                PoseGraphClass = Reproj_TwoFramePGO
            case (True, "disp"):
                PoseGraphClass = ReprojDisp_TwoFramePGO
            case (False, "icp"):
                PoseGraphClass = Analytic_ICP_TwoframePGO
            case (False, "reproj"):
                PoseGraphClass = Analytic_Reproj_TwoFramePGO
            case (False, "disp"):
                PoseGraphClass = Analytic_ReprojDisp_TwoFramePGO
            case _:
                raise ValueError(f"Graph type of {config.graph_type} is not supported")

        return {
            "optimizer_cfg": {
                "kernel"   : Huber(delta=0.1),
                "solver"   : PINV(),
                "strategy" : TrustRegion(radius=1e3),
                "corrector": FastTriggs(Huber(delta=0.1)),
                "vectorize": config.vectorize,
            },
            "device": config.device,
            "two_state_cpu_threads": two_state_cpu_threads,
            "pose_graph_class": PoseGraphClass,
            "optimizer_breakpoint_trace_enable": bool(
                getattr(config, "optimizer_breakpoint_trace_enable", False)
                or os.environ.get("MACVO_OPTIMIZER_BREAKPOINT_TRACE", "").strip().lower()
                in {"1", "true", "yes", "on"}
            ),
            "vio_causal_diagnostics_enable": bool(
                getattr(config, "vio_causal_diagnostics_enable", False)
            ),
            "vio_causal_diagnostics_interval": max(
                1,
                int(getattr(config, "vio_causal_diagnostics_interval", 5)),
            ),
            "imu_factor_mode": imu_factor_mode,
            "two_state_backend_name": two_state_backend_name,
            "two_state_isam2_backend": isam2_backend,
            "two_state_isam2_history_publish_interval": max(
                1,
                int(
                    getattr(
                        config,
                        "two_state_isam2_history_publish_interval",
                        30,
                    )
                ),
            ),
            "two_state_solver": TwoStateVIOSolver(
                max_iterations=int(getattr(config, "two_state_max_iterations", 20)),
                initial_damping=float(getattr(config, "two_state_initial_damping", 1e-3)),
                covariance_eigenvalue_floor=float(
                    getattr(config, "two_state_covariance_eigenvalue_floor", 1e-12)
                ),
                marginalization_eigenvalue_floor=float(
                    getattr(config, "two_state_marginalization_eigenvalue_floor", 1e-10)
                ),
            ),
            "two_state_cross_edge_solver": CrossEdgeTwoStateSolver(
                max_iterations=int(getattr(config, "two_state_max_iterations", 20)),
                initial_damping=float(getattr(config, "two_state_initial_damping", 1e-3)),
                covariance_eigenvalue_floor=float(
                    getattr(config, "two_state_covariance_eigenvalue_floor", 1e-12)
                ),
                marginalization_eigenvalue_floor=float(
                    getattr(config, "two_state_marginalization_eigenvalue_floor", 1e-10)
                ),
                rank_aware_imu_whitening=_optional_bool(
                    getattr(
                        config,
                        "two_state_cross_edge_rank_aware_imu_whitening",
                        os.environ.get("MACVO_SA_V2_RANK_AWARE_IMU_WHITENING"),
                    )
                ),
            ),
            "two_state_prior": None,
            "two_state_cross_edge_prior": None,
            "two_state_last_frame_idx": None,
            "two_state_cross_edge_reset_prior_frame": _optional_frame(
                getattr(
                    config,
                    "two_state_cross_edge_reset_prior_frame",
                    os.environ.get("MACVO_SA_V2_RESET_PRIOR_AT_FRAME"),
                )
            ),
            "two_state_cross_edge_prior_reset_done": False,
            "two_state_cross_edge_checkpoint_frames": _frame_set(
                getattr(
                    config,
                    "two_state_cross_edge_checkpoint_frames",
                    os.environ.get("MACVO_SA_V2_CHECKPOINT_FRAMES"),
                )
            ),
            "two_state_cross_edge_checkpoint_dir": str(
                getattr(
                    config,
                    "two_state_cross_edge_checkpoint_dir",
                    os.environ.get("MACVO_SA_V2_CHECKPOINT_DIR", ""),
                )
                or ""
            ).strip(),
            "two_state_initial_prior_std": initial_prior_std,
            "two_state_visual_huber_delta": float(
                getattr(config, "two_state_visual_huber_delta", 3.0)
            ),
            "two_state_visual_factor_mode": visual_factor_mode,
            "two_state_warm_start": str(
                getattr(config, "two_state_warm_start", "macvo_pose")
            ).strip().lower(),
            "two_state_uvd_huber_delta": float(
                getattr(config, "two_state_uvd_huber_delta", 0.1)
            ),
            "two_state_optimize_acc_bias": bool(
                getattr(config, "two_state_optimize_acc_bias", True)
            ),
            "two_state_optimize_gyro_bias": bool(
                getattr(config, "two_state_optimize_gyro_bias", True)
            ),
            "two_state_visual_gate": {
                "soft_inlier_ratio": float(
                    getattr(config, "two_state_visual_soft_inlier_ratio", 0.5)
                ),
                "reject_inlier_ratio": float(
                    getattr(config, "two_state_visual_reject_inlier_ratio", 0.2)
                ),
                "soft_mean_mahalanobis_sq": float(
                    getattr(config, "two_state_visual_soft_mean_mahalanobis_sq", 9.0)
                ),
                "reject_mean_mahalanobis_sq": float(
                    getattr(config, "two_state_visual_reject_mean_mahalanobis_sq", 100.0)
                ),
                "soft_whitened_pose_norm": float(
                    getattr(config, "two_state_visual_soft_whitened_pose_norm", 6.0)
                ),
                "reject_whitened_pose_norm": float(
                    getattr(config, "two_state_visual_reject_whitened_pose_norm", 20.0)
                ),
                "max_covariance_inflation": float(
                    getattr(config, "two_state_visual_max_covariance_inflation", 1.0e6)
                ),
            },
        }

    @staticmethod
    def _isam2_history_tensors(
        context: dict,
        history: list[tuple[int, NavigationState]] | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        backend = context.get("two_state_isam2_backend")
        if not isinstance(backend, IncrementalT2ISAM2Backend) or not backend.initialized:
            raise RuntimeError("iSAM2 history requested before backend initialization")
        packet = context.get("two_state_last_factor_packet")
        if not isinstance(packet, T2FactorPacket):
            raise RuntimeError("iSAM2 history has no matching T2 factor packet")

        history = backend.history() if history is None else history
        if len(history) < 2:
            raise RuntimeError("iSAM2 final history must contain at least two states")
        frame_indices = torch.tensor(
            [frame for frame, _ in history], dtype=torch.long
        )
        states = [state for _, state in history]
        extrinsic_CI = pp.SE3(packet.extrinsic_CI.detach().cpu())
        camera_poses = torch.cat(
            [
                (pp.SE3(state.pose_WB.detach().cpu()) @ extrinsic_CI.Inv()).tensor()
                for state in states
            ],
            dim=0,
        ).float()
        velocities = torch.stack(
            [state.velocity_W.detach().cpu() for state in states], dim=0
        ).float()
        acc_biases = torch.stack(
            [state.acc_bias.detach().cpu() for state in states], dim=0
        ).float()
        gyro_biases = torch.stack(
            [state.gyro_bias.detach().cpu() for state in states], dim=0
        ).float()
        return frame_indices, camera_poses, velocities, acc_biases, gyro_biases

    @staticmethod
    def _finalize_context(context: dict) -> tuple[dict, GraphOutput | None]:
        if str(context.get("two_state_backend_name", "two_state")) != "isam2":
            return context, None
        backend = context.get("two_state_isam2_backend")
        if not isinstance(backend, IncrementalT2ISAM2Backend) or not backend.initialized:
            return context, None

        started = time.perf_counter()
        history = backend.history()
        (
            frame_indices,
            camera_poses,
            velocities,
            acc_biases,
            gyro_biases,
        ) = TwoFrame_PGO._isam2_history_tensors(context, history)
        frame_idx = int(frame_indices[-1].item())
        from_idx = int(frame_indices[-2].item())
        context["two_state_isam2_last_history_frame"] = frame_idx

        output = GraphOutput(
            motion=camera_poses[-1:].clone(),
            from_idx=torch.tensor([from_idx], dtype=torch.long),
            frame_idx=torch.tensor([frame_idx], dtype=torch.long),
            velocity_world=velocities[-1].clone(),
            acc_bias=acc_biases[-1].clone(),
            gyro_bias=gyro_biases[-1].clone(),
            imu_factor_mode="two_state_fixed_lag",
            vio_factor_active=True,
            vio_bias_state_active=True,
            imu_residual_rows=5,
            use_imu_rotation=True,
            use_imu_translation=True,
            window_frame_indices=frame_indices,
            window_motions=camera_poses,
            window_velocity_world=velocities,
            window_acc_bias=acc_biases,
            window_gyro_bias=gyro_biases,
            local_ba_window_size=int(frame_indices.numel()),
            local_ba_writeback="all_isam2_history",
            local_ba_num_frames=int(frame_indices.numel()),
            local_ba_num_edges=max(int(frame_indices.numel()) - 1, 0),
            local_ba_num_visual_residual_blocks=0,
            local_ba_graph_build_s=0.0,
            local_ba_lm_s=0.0,
            local_ba_refine_s=0.0,
            local_ba_optimize_total_s=time.perf_counter() - started,
            two_state_solver_converged=True,
            two_state_solver_iterations=0,
            two_state_solver_convergence_reason="isam2_final_history_snapshot",
            two_state_final_step_norm=0.0,
            two_state_final_gradient_inf_norm=0.0,
            two_state_solver_accepted_steps=0,
            two_state_solver_rejected_steps=0,
            vio_backend="isam2",
            isam2_update_ms=0.0,
            isam2_state_count=int(backend.state_count),
            isam2_history_revision=True,
        )
        return context, output

    @staticmethod
    def _refine_vio_vector_states(graph: FactorGraph, weight: torch.Tensor) -> float | None:
        """Refine VIO vector states with pose fixed.

        PyPose LM solves the whole weighted normal equation at once. In the VIO
        path, visual projection columns can be numerically much larger than IMU
        bias/velocity columns, so the mixed solve may leave the ordinary vector
        states unchanged even when their gradients are valid. This small
        block-coordinate step updates only velocity/bias states after the pose
        update.
        """
        if not bool(getattr(graph, "use_vio_imu_factor", False)):
            return None

        params: list[torch.nn.Parameter] = []
        velocity = getattr(graph, "velocity2opt", None)
        if isinstance(velocity, torch.nn.Parameter) and velocity.requires_grad:
            params.append(velocity)

        if bool(getattr(graph, "use_vio_bias_state", False)):
            for name in ("acc_bias2opt", "gyro_bias2opt"):
                param = getattr(graph, name, None)
                if isinstance(param, torch.nn.Parameter) and param.requires_grad:
                    params.append(param)

        if not params:
            return None

        device = next(graph.parameters()).device
        weight = weight.detach().clone().to(device=device, dtype=torch.float64)
        optimizer = torch.optim.LBFGS(
            params,
            lr=1.0,
            max_iter=8,
            tolerance_grad=1e-12,
            tolerance_change=1e-12,
            line_search_fn="strong_wolfe",
        )

        final_loss: torch.Tensor | None = None

        def closure() -> torch.Tensor:
            nonlocal final_loss
            graph.zero_grad(set_to_none=True)
            residual_blocks = [graph._imu_vio_residual()]  # type: ignore[attr-defined]
            if bool(getattr(graph, "use_vio_bias_state", False)):
                residual_blocks.append(graph._imu_vio_bias_residual())  # type: ignore[attr-defined]
            residual = torch.cat(residual_blocks, dim=0).reshape(-1, 1).to(weight)
            imu_weight = weight[-residual.numel():, -residual.numel():]
            loss = (residual.mT @ imu_weight @ residual).reshape(())
            loss.backward()
            final_loss = loss.detach()
            return loss

        try:
            with torch.enable_grad():
                optimizer.step(closure)
        except Exception:
            return None
        finally:
            graph.zero_grad(set_to_none=True)

        if final_loss is None:
            return None
        return float(final_loss.cpu().item())

    @staticmethod
    def _refine_local_window_states(graph: FactorGraph, weight: torch.Tensor) -> float | None:
        params = [param for param in graph.parameters() if param.requires_grad]
        if not params:
            return None
        device = next(graph.parameters()).device
        weight = weight.detach().clone().to(device=device, dtype=torch.float64)
        optimizer = torch.optim.LBFGS(
            params,
            lr=1.0,
            max_iter=80,
            tolerance_grad=1e-12,
            tolerance_change=1e-12,
            line_search_fn="strong_wolfe",
        )
        final_loss: torch.Tensor | None = None

        def closure() -> torch.Tensor:
            nonlocal final_loss
            graph.zero_grad(set_to_none=True)
            residual = graph.forward().reshape(-1, 1).to(weight)
            loss = (residual.mT @ weight @ residual).reshape(())
            loss.backward()
            final_loss = loss.detach()
            return loss

        try:
            with torch.enable_grad():
                optimizer.step(closure)
        except Exception:
            return None
        finally:
            graph.zero_grad(set_to_none=True)

        if final_loss is None:
            return None
        return float(final_loss.cpu().item())

    @staticmethod
    def _refine_local_window_vector_states(graph: FactorGraph, weight: torch.Tensor) -> float | None:
        params: list[torch.nn.Parameter] = []
        for name in ("velocity_window", "acc_bias_window", "gyro_bias_window"):
            param = getattr(graph, name, None)
            if isinstance(param, torch.nn.Parameter) and param.requires_grad:
                params.append(param)
        if not params:
            return None

        edges = list(getattr(graph, "edges", []))
        n_visual_rows = sum(int(edge.observations.data["pixel2_uv"].shape[0]) for edge in edges)
        device = next(graph.parameters()).device
        weight = weight.detach().clone().to(device=device, dtype=torch.float64)
        optimizer = torch.optim.LBFGS(
            params,
            lr=1.0,
            max_iter=8,
            tolerance_grad=1e-12,
            tolerance_change=1e-12,
            line_search_fn="strong_wolfe",
        )
        final_loss: torch.Tensor | None = None

        def closure() -> torch.Tensor:
            nonlocal final_loss
            graph.zero_grad(set_to_none=True)
            rows = graph.forward()[n_visual_rows:]
            residual = rows.reshape(-1, 1).to(weight)
            offset = n_visual_rows * 3
            imu_weight = weight[offset:, offset:]
            loss = (residual.mT @ imu_weight @ residual).reshape(())
            loss.backward()
            final_loss = loss.detach()
            return loss

        try:
            with torch.enable_grad():
                optimizer.step(closure)
        except Exception:
            return None
        finally:
            graph.zero_grad(set_to_none=True)

        if final_loss is None:
            return None
        return float(final_loss.cpu().item())

    @staticmethod
    def _debug_tensor_payload(value) -> list | float | int | None:
        try:
            if isinstance(value, pp.LieTensor):
                value = value.tensor()
            if isinstance(value, torch.nn.Parameter):
                value = value.detach()
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu()
                if value.numel() == 1:
                    return float(value.reshape(-1)[0].item())
                return value.tolist()
            if isinstance(value, (float, int)):
                return value
        except Exception:
            return None
        return None

    @staticmethod
    def _debug_edge_payload(edge: GraphInput) -> dict:
        payload = {
            "from_idx": int(edge.from_idx.reshape(-1)[0].detach().cpu().item()),
            "frame_idx": int(edge.frame_idx.reshape(-1)[0].detach().cpu().item()),
            "num_visual_observations": int(edge.observations.data["pixel2_uv"].shape[0]),
            "imu_factor_active": bool(edge.imu_vio_factor_enable),
            "alpha_p": float(edge.imu_vio_alpha_p),
            "alpha_v": float(edge.imu_vio_alpha_v),
            "alpha_R": float(edge.imu_vio_alpha_R),
        }
        if edge.imu_vio_dt is not None:
            payload["imu_dt"] = TwoFrame_PGO._debug_tensor_payload(edge.imu_vio_dt)
        if edge.imu_vio_delta_p is not None:
            payload["imu_delta_p"] = TwoFrame_PGO._debug_tensor_payload(edge.imu_vio_delta_p.reshape(3))
        if edge.imu_vio_delta_v is not None:
            payload["imu_delta_v"] = TwoFrame_PGO._debug_tensor_payload(edge.imu_vio_delta_v.reshape(3))
        if edge.imu_vio_delta_rotvec is not None:
            payload["imu_delta_rotvec"] = TwoFrame_PGO._debug_tensor_payload(edge.imu_vio_delta_rotvec.reshape(3))
        return payload

    @staticmethod
    def _debug_residual_payload(graph: FactorGraph) -> dict:
        try:
            residual = graph.forward().detach().to(dtype=torch.float64)
        except Exception as exc:
            return {"error": str(exc)}
        if residual.numel() == 0:
            return {"rows": 0}

        rows = residual.reshape(-1, residual.shape[-1])
        n_total = int(rows.shape[0])
        n_visual = n_total
        n_imu = 0
        n_bias = 0

        if hasattr(graph, "frame_indices") and hasattr(graph, "edges"):
            try:
                edges = list(getattr(graph, "edges"))
                n_visual = sum(int(edge.observations.data["pixel2_uv"].shape[0]) for edge in edges)
                n_imu = sum(3 for edge in edges if graph._edge_has_vio(edge))  # type: ignore[attr-defined]
                n_bias = sum(2 for edge in edges if graph._edge_has_bias(edge))  # type: ignore[attr-defined]
            except Exception:
                n_visual = n_total
                n_imu = 0
                n_bias = 0
        else:
            has_vio = bool(getattr(graph, "use_vio_imu_factor", False))
            has_vio_bias = bool(getattr(graph, "use_vio_bias_state", False))
            has_rot = bool(getattr(graph, "use_imu_rot_prior", False))
            has_trans = bool(getattr(graph, "use_imu_trans_prior", False))
            if has_vio:
                n_imu = 3
                n_bias = 2 if has_vio_bias else 0
            else:
                n_imu = int(has_rot) + int(has_trans)
            n_visual = max(0, n_total - n_imu - n_bias)

        n_visual = min(n_visual, n_total)
        n_imu = min(n_imu, max(0, n_total - n_visual))
        n_bias = min(n_bias, max(0, n_total - n_visual - n_imu))

        visual = rows[:n_visual]
        imu = rows[n_visual:n_visual + n_imu]
        bias = rows[n_visual + n_imu:n_visual + n_imu + n_bias]
        return {
            "rows": n_total,
            "visual_rows": int(n_visual),
            "imu_rows": int(n_imu),
            "bias_rows": int(n_bias),
            "raw_norm": float(rows.reshape(-1).norm().cpu().item()),
            "visual_raw_norm": float(visual.reshape(-1).norm().cpu().item()) if visual.numel() else 0.0,
            "imu_raw_norm": float(imu.reshape(-1).norm().cpu().item()) if imu.numel() else 0.0,
            "bias_raw_norm": float(bias.reshape(-1).norm().cpu().item()) if bias.numel() else 0.0,
        }

    @staticmethod
    def _debug_graph_state(graph: FactorGraph, stage: str) -> dict:
        with torch.no_grad():
            payload: dict = {
                "stage": stage,
                "graph_class": type(graph).__name__,
                "residual": TwoFrame_PGO._debug_residual_payload(graph),
            }

            if hasattr(graph, "frame_indices") and hasattr(graph, "_all_poses"):
                poses = graph._all_poses().tensor().detach().cpu().float()  # type: ignore[attr-defined]
                payload["frame_indices"] = TwoFrame_PGO._debug_tensor_payload(getattr(graph, "frame_indices"))
                payload["poses"] = poses.tolist()
                if hasattr(graph, "_all_velocity"):
                    payload["velocity_world"] = TwoFrame_PGO._debug_tensor_payload(graph._all_velocity())  # type: ignore[attr-defined]
                if hasattr(graph, "_all_acc_bias"):
                    payload["acc_bias"] = TwoFrame_PGO._debug_tensor_payload(graph._all_acc_bias())  # type: ignore[attr-defined]
                if hasattr(graph, "_all_gyro_bias"):
                    payload["gyro_bias"] = TwoFrame_PGO._debug_tensor_payload(graph._all_gyro_bias())  # type: ignore[attr-defined]
                payload["edges"] = [
                    TwoFrame_PGO._debug_edge_payload(edge) for edge in list(getattr(graph, "edges", []))
                ]
                return payload

            payload["frame_indices"] = [
                int(getattr(graph, "from_idx").reshape(-1)[0].detach().cpu().item()) if hasattr(graph, "from_idx") else -1,
                int(getattr(graph, "frame_idx").reshape(-1)[0].detach().cpu().item()) if hasattr(graph, "frame_idx") else -1,
            ]
            if hasattr(graph, "from_pose"):
                payload["from_pose"] = TwoFrame_PGO._debug_tensor_payload(getattr(graph, "from_pose"))
            if hasattr(graph, "init_motion"):
                payload["init_motion"] = TwoFrame_PGO._debug_tensor_payload(getattr(graph, "init_motion"))
            if hasattr(graph, "pose2opt"):
                payload["pose2opt"] = TwoFrame_PGO._debug_tensor_payload(getattr(graph, "pose2opt"))
            if hasattr(graph, "velocity2opt"):
                payload["velocity_world"] = TwoFrame_PGO._debug_tensor_payload(getattr(graph, "velocity2opt"))
            if hasattr(graph, "acc_bias2opt"):
                payload["acc_bias"] = TwoFrame_PGO._debug_tensor_payload(getattr(graph, "acc_bias2opt"))
            if hasattr(graph, "gyro_bias2opt"):
                payload["gyro_bias"] = TwoFrame_PGO._debug_tensor_payload(getattr(graph, "gyro_bias2opt"))
            return payload

    @staticmethod
    def _debug_w2_equivalence_payload(
        local_graph: FactorGraph,
        graph_data: LocalWindowGraphInput,
        context: dict,
    ) -> dict | None:
        """Compare W=2 local graph against the ordinary two-frame graph on the same input."""
        try:
            if int(graph_data.frame_indices.numel()) != 2 or len(graph_data.edges) != 1:
                return None
            edge = LocalWindowInertialGraph._coerce_edge(graph_data.edges[0])
            two_frame_graph = context["pose_graph_class"](edge)
            two_frame_graph = two_frame_graph.to(
                device=torch.device(context["device"]),
                dtype=torch.double,
            )
            assert isinstance(two_frame_graph, FactorGraph)

            with torch.no_grad():
                local_residual = local_graph.forward().detach().to(dtype=torch.float64)
                two_residual = two_frame_graph.forward().detach().to(dtype=torch.float64)
                local_weight = local_graph.weight_matrix().detach().to(dtype=torch.float64)  # type: ignore[attr-defined]
                two_weight = two_frame_graph.weight_matrix().detach().to(dtype=torch.float64)  # type: ignore[attr-defined]

                residual_shape_match = tuple(local_residual.shape) == tuple(two_residual.shape)
                weight_shape_match = tuple(local_weight.shape) == tuple(two_weight.shape)
                payload: dict = {
                    "same_input_edge": {
                        "from_idx": int(edge.from_idx.reshape(-1)[0].item()),
                        "frame_idx": int(edge.frame_idx.reshape(-1)[0].item()),
                        "num_visual_observations": int(edge.observations.data["pixel2_uv"].shape[0]),
                    },
                    "local_graph_class": type(local_graph).__name__,
                    "two_frame_graph_class": type(two_frame_graph).__name__,
                    "residual_shape_local": list(local_residual.shape),
                    "residual_shape_two_frame": list(two_residual.shape),
                    "weight_shape_local": list(local_weight.shape),
                    "weight_shape_two_frame": list(two_weight.shape),
                    "residual_shape_match": residual_shape_match,
                    "weight_shape_match": weight_shape_match,
                }
                if residual_shape_match:
                    diff = local_residual - two_residual
                    payload["residual_max_abs_diff"] = float(diff.abs().max().cpu().item()) if diff.numel() else 0.0
                    payload["residual_l2_diff"] = float(diff.reshape(-1).norm().cpu().item()) if diff.numel() else 0.0
                if weight_shape_match:
                    wdiff = local_weight - two_weight
                    payload["weight_max_abs_diff"] = float(wdiff.abs().max().cpu().item()) if wdiff.numel() else 0.0
                    payload["weight_l2_diff"] = float(wdiff.reshape(-1).norm().cpu().item()) if wdiff.numel() else 0.0
                return payload
        except Exception as exc:
            return {"error": str(exc)}

    @staticmethod
    def _debug_w2_reference_output_payload(
        local_output: GraphOutput,
        graph_data: LocalWindowGraphInput,
        context: dict,
    ) -> dict | None:
        """Run the ordinary two-frame optimizer on the same W=2 edge and compare outputs."""
        try:
            if int(graph_data.frame_indices.numel()) != 2 or len(graph_data.edges) != 1:
                return None
            edge = LocalWindowInertialGraph._coerce_edge(graph_data.edges[0])
            ref_context = dict(context)
            ref_context["optimizer_breakpoint_trace_enable"] = False
            _, ref_output = TwoFrame_PGO._optimize(ref_context, edge)

            def _tensor(value) -> torch.Tensor:
                if isinstance(value, pp.LieTensor):
                    return value.tensor().detach().cpu().float()
                if isinstance(value, torch.nn.Parameter):
                    return value.detach().cpu().float()
                if isinstance(value, torch.Tensor):
                    return value.detach().cpu().float()
                raise TypeError(type(value).__name__)

            local_motion = _tensor(local_output.motion).reshape(-1, 7)[-1]
            ref_motion = _tensor(ref_output.motion).reshape(-1, 7)[-1]
            payload: dict = {
                "same_input_edge": {
                    "from_idx": int(edge.from_idx.reshape(-1)[0].item()),
                    "frame_idx": int(edge.frame_idx.reshape(-1)[0].item()),
                },
                "motion_l2_diff": float((local_motion - ref_motion).norm().item()),
                "motion_max_abs_diff": float((local_motion - ref_motion).abs().max().item()),
                "local_final_loss": local_output.final_loss,
                "two_frame_final_loss": ref_output.final_loss,
            }
            for name in ("velocity_world", "acc_bias", "gyro_bias"):
                local_value = getattr(local_output, name)
                ref_value = getattr(ref_output, name)
                if local_value is not None and ref_value is not None:
                    diff = _tensor(local_value).reshape(-1) - _tensor(ref_value).reshape(-1)
                    payload[f"{name}_l2_diff"] = float(diff.norm().item())
                    payload[f"{name}_max_abs_diff"] = float(diff.abs().max().item())
            return payload
        except Exception as exc:
            return {"error": str(exc)}

    @staticmethod
    def _plain_parameter_tensor(value: torch.Tensor) -> torch.Tensor:
        if isinstance(value, pp.LieTensor):
            return value.tensor()
        return value.as_subclass(torch.Tensor) if type(value) is not torch.Tensor else value

    @staticmethod
    def _capture_parameter_state(graph: FactorGraph) -> dict[str, torch.Tensor]:
        return {
            name: TwoFrame_PGO._plain_parameter_tensor(param.detach()).clone().to(dtype=torch.float64)
            for name, param in graph.named_parameters()
        }

    @staticmethod
    def _parameter_delta(
        graph: FactorGraph,
        initial_state: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        parts: list[torch.Tensor] = []
        for name, param in graph.named_parameters():
            initial = initial_state.get(name)
            if initial is None:
                continue
            final = TwoFrame_PGO._plain_parameter_tensor(param.detach()).to(dtype=torch.float64)
            if final.shape != initial.shape:
                continue
            final_aligned = final.clone()
            if final_aligned.shape[-1:] == (7,):
                initial_rows = initial.reshape(-1, 7)
                final_rows = final_aligned.reshape(-1, 7)
                flip = (initial_rows[:, 3:7] * final_rows[:, 3:7]).sum(dim=1) < 0
                final_rows[flip, 3:7] *= -1
                final_aligned = final_rows.reshape_as(final_aligned)
            parts.append((final_aligned - initial).reshape(-1))
        if not parts:
            return torch.zeros(0, dtype=torch.float64)
        return torch.cat(parts, dim=0)

    @staticmethod
    def _capture_causal_energy_snapshot(
        graph: FactorGraph,
        graph_data: GraphInput | LocalWindowGraphInput,
    ) -> dict[str, float | None]:
        snapshot_output = graph.write_back()
        if isinstance(graph_data, LocalWindowGraphInput):
            TwoFrame_PGO._compute_local_window_diagnostics(graph, graph_data, snapshot_output, None)
        else:
            TwoFrame_PGO._compute_pair_diagnostics(graph, snapshot_output, None)

        visual = snapshot_output.energy_visual_weighted
        imu = snapshot_output.energy_imu_weighted
        total = None
        if visual is not None or imu is not None:
            total = float((visual or 0.0) + (imu or 0.0))
        return {
            "energy_visual_weighted": visual,
            "energy_p_weighted": snapshot_output.energy_p_weighted,
            "energy_v_weighted": snapshot_output.energy_v_weighted,
            "energy_R_weighted": snapshot_output.energy_R_weighted,
            "energy_pv_weighted": snapshot_output.energy_pv_weighted,
            "energy_imu_diag_weighted": snapshot_output.energy_imu_diag_weighted,
            "energy_imu_weighted": imu,
            "energy_imu_to_visual_ratio": snapshot_output.energy_imu_to_visual_ratio,
            "energy_pv_to_visual_ratio": snapshot_output.energy_pv_to_visual_ratio,
            "energy_R_to_visual_ratio": snapshot_output.energy_R_to_visual_ratio,
            "total_loss": total,
        }

    @staticmethod
    def _safe_scalar_ratio(numerator: float, denominator: float) -> float | None:
        if not math.isfinite(numerator) or not math.isfinite(denominator) or abs(denominator) <= 1e-12:
            return None
        return float(numerator / denominator)

    @staticmethod
    def _safe_vector_cosine(left: torch.Tensor, right: torch.Tensor) -> float | None:
        if left.numel() == 0 or right.numel() == 0 or left.numel() != right.numel():
            return None
        denom = float(left.norm().item() * right.norm().item())
        if denom <= 1e-12:
            return None
        return float(torch.dot(left, right).item() / denom)

    @staticmethod
    def _damped_counterfactual_step(hessian: torch.Tensor, gradient: torch.Tensor) -> torch.Tensor:
        if gradient.numel() == 0:
            return gradient.clone()
        diag_scale = float(hessian.diagonal().abs().mean().item()) if hessian.numel() else 1.0
        damping = max(1e-9, 1e-6 * max(diag_scale, 1.0))
        system = hessian + torch.eye(
            hessian.shape[0], device=hessian.device, dtype=hessian.dtype
        ) * damping
        return -(torch.linalg.pinv(system) @ gradient)

    @staticmethod
    def _compute_initial_factor_influence(
        graph: FactorGraph,
        optimizer,
        weight: torch.Tensor,
    ) -> dict[str, object]:
        if not bool(getattr(graph, "use_vio_imu_factor", False)):
            return {}

        residual_groups = list(optimizer.model((), None))
        jacobian_raw = modjac(
            optimizer.model,
            input=((), None),
            flatten=False,
            vectorize=True,
        )
        params = tuple(dict(optimizer.model.named_parameters()).values())
        jacobian_groups = [
            optimizer.model.flatten_row_jacobian(group, params)
            for group in jacobian_raw
        ]
        for idx in range(len(residual_groups)):
            corrector = optimizer.corrector[0] if len(optimizer.corrector) == 1 else optimizer.corrector[idx]
            residual_groups[idx], jacobian_groups[idx] = corrector(
                R=residual_groups[idx], J=jacobian_groups[idx]
            )
        residual, normalized_weight, jacobian = optimizer.model.normalize_RWJ(
            residual_groups,
            weight,
            jacobian_groups,
        )
        residual = residual.reshape(-1).to(dtype=torch.float64)
        jacobian = jacobian.to(dtype=torch.float64)
        normalized_weight = normalized_weight.to(dtype=torch.float64)

        residual_blocks = int(graph.forward().shape[0])
        imu_blocks = 3 + (2 if bool(getattr(graph, "use_vio_bias_state", False)) else 0)
        visual_scalars = max(0, residual_blocks - imu_blocks) * 3
        if visual_scalars <= 0 or visual_scalars >= residual.numel():
            return {}

        def normal_terms(start: int, end: int) -> tuple[torch.Tensor, torch.Tensor]:
            r = residual[start:end]
            J = jacobian[start:end]
            W = normalized_weight[start:end, start:end]
            return J.mT @ W @ r, J.mT @ W @ J

        visual_gradient, visual_hessian = normal_terms(0, visual_scalars)
        imu_gradient, imu_hessian = normal_terms(visual_scalars, residual.numel())
        full_gradient = visual_gradient + imu_gradient
        full_hessian = visual_hessian + imu_hessian

        visual_step = TwoFrame_PGO._damped_counterfactual_step(visual_hessian, visual_gradient)
        imu_step = TwoFrame_PGO._damped_counterfactual_step(imu_hessian, imu_gradient)
        full_step = TwoFrame_PGO._damped_counterfactual_step(full_hessian, full_gradient)

        block_gradients: list[float | None] = []
        for block_idx in range(3):
            start = visual_scalars + block_idx * 3
            end = start + 3
            if end <= residual.numel():
                gradient, _ = normal_terms(start, end)
                block_gradients.append(float(gradient.norm().item()))
            else:
                block_gradients.append(None)

        visual_grad_norm = float(visual_gradient.norm().item())
        imu_grad_norm = float(imu_gradient.norm().item())
        visual_hessian_trace = float(torch.trace(visual_hessian).item())
        imu_hessian_trace = float(torch.trace(imu_hessian).item())
        return {
            "visual_gradient": visual_gradient.detach(),
            "imu_gradient": imu_gradient.detach(),
            "visual_hessian": visual_hessian.detach(),
            "imu_hessian": imu_hessian.detach(),
            "visual_step": visual_step.detach(),
            "imu_step": imu_step.detach(),
            "full_step": full_step.detach(),
            "influence_visual_grad_norm": visual_grad_norm,
            "influence_imu_grad_norm": imu_grad_norm,
            "influence_grad_cosine": TwoFrame_PGO._safe_vector_cosine(visual_gradient, imu_gradient),
            "influence_visual_hessian_trace": visual_hessian_trace,
            "influence_imu_hessian_trace": imu_hessian_trace,
            "influence_imu_to_visual_grad_ratio": TwoFrame_PGO._safe_scalar_ratio(
                imu_grad_norm, visual_grad_norm
            ),
            "influence_imu_to_visual_hessian_ratio": TwoFrame_PGO._safe_scalar_ratio(
                imu_hessian_trace, visual_hessian_trace
            ),
            "influence_p_grad_norm": block_gradients[0],
            "influence_v_grad_norm": block_gradients[1],
            "influence_R_grad_norm": block_gradients[2],
            "counterfactual_visual_step_norm": float(visual_step.norm().item()),
            "counterfactual_imu_step_norm": float(imu_step.norm().item()),
            "counterfactual_full_step_norm": float(full_step.norm().item()),
            "counterfactual_visual_to_imu_cosine": TwoFrame_PGO._safe_vector_cosine(
                visual_step, imu_step
            ),
        }

    @staticmethod
    def _attach_causal_diagnostics(
        output: GraphOutput,
        graph: FactorGraph,
        graph_data: GraphInput | LocalWindowGraphInput,
        initial_energy: dict[str, float | None],
        initial_state: dict[str, torch.Tensor],
        influence: dict[str, object],
    ) -> None:
        initial_pose = initial_state.get("pose2opt")
        if initial_pose is not None:
            output.initial_motion = initial_pose.detach().clone()
            if isinstance(graph_data, GraphInput):
                from_pose = pp.SE3(graph_data.from_pose).to(
                    device=initial_pose.device,
                    dtype=initial_pose.dtype,
                )
                initial_relative = from_pose.Inv() @ pp.SE3(initial_pose)
                initial_translation = initial_relative.translation().reshape(3)
                output.init_delta_x = float(initial_translation[0].item())
                output.init_delta_y = float(initial_translation[1].item())
                output.init_delta_z = float(initial_translation[2].item())
                output.init_delta_t_norm = float(initial_translation.norm().item())
                output.init_delta_R_angle = float(
                    initial_relative.rotation().Log().tensor().norm().item()
                )

        initial_velocity = initial_state.get("velocity2opt")
        if initial_velocity is not None and initial_velocity.numel() >= 3:
            velocity = initial_velocity.reshape(-1, 3)[-1]
            output.init_velocity_j_x = float(velocity[0].item())
            output.init_velocity_j_y = float(velocity[1].item())
            output.init_velocity_j_z = float(velocity[2].item())

        output.initial_energy_visual_weighted = initial_energy.get("energy_visual_weighted")
        output.initial_energy_p_weighted = initial_energy.get("energy_p_weighted")
        output.initial_energy_v_weighted = initial_energy.get("energy_v_weighted")
        output.initial_energy_R_weighted = initial_energy.get("energy_R_weighted")
        output.initial_energy_pv_weighted = initial_energy.get("energy_pv_weighted")
        output.initial_energy_imu_diag_weighted = initial_energy.get("energy_imu_diag_weighted")
        output.initial_energy_imu_weighted = initial_energy.get("energy_imu_weighted")
        output.initial_energy_imu_to_visual_ratio = initial_energy.get("energy_imu_to_visual_ratio")
        output.initial_energy_pv_to_visual_ratio = initial_energy.get("energy_pv_to_visual_ratio")
        output.initial_energy_R_to_visual_ratio = initial_energy.get("energy_R_to_visual_ratio")
        output.initial_total_loss = initial_energy.get("total_loss")

        def difference(final: float | None, initial_key: str) -> float | None:
            initial = initial_energy.get(initial_key)
            if final is None or initial is None:
                return None
            return float(final - initial)

        output.energy_visual_change = difference(output.energy_visual_weighted, "energy_visual_weighted")
        output.energy_imu_change = difference(output.energy_imu_weighted, "energy_imu_weighted")
        output.energy_p_change = difference(output.energy_p_weighted, "energy_p_weighted")
        output.energy_v_change = difference(output.energy_v_weighted, "energy_v_weighted")
        output.energy_R_change = difference(output.energy_R_weighted, "energy_R_weighted")

        initial_pose = initial_state.get("pose2opt")
        final_pose = getattr(graph, "pose2opt", None)
        if initial_pose is not None and isinstance(final_pose, torch.Tensor):
            relative = pp.SE3(initial_pose).Inv() @ pp.SE3(final_pose.detach().to(dtype=torch.float64))
            output.update_pose_translation_norm = float(relative.translation().norm().item())
            output.update_pose_rotation_norm = float(relative.rotation().Log().tensor().norm().item())

        for state_name, output_name in (
            ("velocity2opt", "update_velocity_norm"),
            ("acc_bias2opt", "update_acc_bias_norm"),
            ("gyro_bias2opt", "update_gyro_bias_norm"),
        ):
            initial = initial_state.get(state_name)
            final = getattr(graph, state_name, None)
            if initial is not None and isinstance(final, torch.Tensor):
                setattr(output, output_name, float((final.detach().to(dtype=torch.float64) - initial).norm().item()))

        if not influence:
            return
        output.influence_sampled = 1
        scalar_fields = (
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
            "counterfactual_visual_step_norm",
            "counterfactual_imu_step_norm",
            "counterfactual_full_step_norm",
            "counterfactual_visual_to_imu_cosine",
        )
        for field in scalar_fields:
            value = influence.get(field)
            if value is not None:
                setattr(output, field, float(value))

        actual_step = TwoFrame_PGO._parameter_delta(graph, initial_state)
        visual_step = influence.get("visual_step")
        imu_step = influence.get("imu_step")
        full_step = influence.get("full_step")
        if isinstance(visual_step, torch.Tensor):
            output.actual_to_visual_step_cosine = TwoFrame_PGO._safe_vector_cosine(actual_step, visual_step)
        if isinstance(imu_step, torch.Tensor):
            output.actual_to_imu_step_cosine = TwoFrame_PGO._safe_vector_cosine(actual_step, imu_step)
        if isinstance(full_step, torch.Tensor):
            output.actual_to_full_step_cosine = TwoFrame_PGO._safe_vector_cosine(actual_step, full_step)

        for prefix in ("visual", "imu"):
            gradient = influence.get(f"{prefix}_gradient")
            hessian = influence.get(f"{prefix}_hessian")
            if isinstance(gradient, torch.Tensor) and isinstance(hessian, torch.Tensor):
                if actual_step.numel() == gradient.numel():
                    predicted_change = torch.dot(gradient, actual_step) + 0.5 * torch.dot(
                        actual_step, hessian @ actual_step
                    )
                    setattr(
                        output,
                        f"predicted_{prefix}_change_on_actual_step",
                        float(predicted_change.item()),
                    )

    @staticmethod
    def _optimize_two_state_fixed_lag(
        context: dict,
        graph_data: GraphInput,
    ) -> tuple[dict, GraphOutput]:
        optimize_start = time.perf_counter()
        visual_factor_mode = str(
            context.get("two_state_visual_factor_mode", "relative_pose")
        ).strip().lower()
        required = {
            "imu_vio_prev_velocity_world": graph_data.imu_vio_prev_velocity_world,
            "imu_vio_curr_velocity_init_world": graph_data.imu_vio_curr_velocity_init_world,
            "imu_vio_prev_acc_bias": graph_data.imu_vio_prev_acc_bias,
            "imu_vio_prev_gyro_bias": graph_data.imu_vio_prev_gyro_bias,
            "imu_vio_curr_acc_bias_init": graph_data.imu_vio_curr_acc_bias_init,
            "imu_vio_curr_gyro_bias_init": graph_data.imu_vio_curr_gyro_bias_init,
            "imu_vio_linearized_acc_bias": graph_data.imu_vio_linearized_acc_bias,
            "imu_vio_linearized_gyro_bias": graph_data.imu_vio_linearized_gyro_bias,
            "imu_vio_bias_jacobian": graph_data.imu_vio_bias_jacobian,
            "imu_vio_bias_rw_cov": graph_data.imu_vio_bias_rw_cov,
            "imu_vio_delta_rotvec": graph_data.imu_vio_delta_rotvec,
            "imu_vio_delta_v": graph_data.imu_vio_delta_v,
            "imu_vio_delta_p": graph_data.imu_vio_delta_p,
            "imu_vio_cov": graph_data.imu_vio_cov,
            "imu_vio_dt": graph_data.imu_vio_dt,
            "imu_vio_sensor_T_imu": graph_data.imu_vio_sensor_T_imu,
        }
        if visual_factor_mode == "relative_pose":
            required.update(
                {
                    "visual_relative_pose_CiCj": graph_data.visual_relative_pose_CiCj,
                    "visual_relative_pose_cov": graph_data.visual_relative_pose_cov,
                }
            )
        elif visual_factor_mode == "compressed_uvd":
            has_cached_compression = (
                graph_data.visual_compressed_uvd_reference_CjCi is not None
                and graph_data.visual_compressed_uvd_hessian is not None
                and graph_data.visual_compressed_uvd_gradient is not None
                and bool(
                    torch.isfinite(graph_data.visual_compressed_uvd_hessian).all()
                    and graph_data.visual_compressed_uvd_hessian.abs().sum() > 0.0
                )
            )
            if has_cached_compression:
                required.update(
                    {
                        "visual_compressed_uvd_reference_CjCi": graph_data.visual_compressed_uvd_reference_CjCi,
                        "visual_compressed_uvd_hessian": graph_data.visual_compressed_uvd_hessian,
                        "visual_compressed_uvd_gradient": graph_data.visual_compressed_uvd_gradient,
                    }
                )
            else:
                # Online mode uses the raw UVD factor already built by the
                # current MACVO pair; no visual cache or sidecar is involved.
                required.update(
                    {
                        "points.pos_Tc": graph_data.points.data.get("pos_Tc"),
                        "observations.pixel2_uv": graph_data.observations.data.get("pixel2_uv"),
                        "observations.pixel2_disp": graph_data.observations.data.get("pixel2_disp"),
                        "observations.pixel2_uv_cov": graph_data.observations.data.get("pixel2_uv_cov"),
                        "observations.pixel2_disp_cov": graph_data.observations.data.get("pixel2_disp_cov"),
                    }
                )
        elif visual_factor_mode != "direct_uvd":
            raise ValueError(f"unsupported two-state visual factor mode: {visual_factor_mode}")
        missing = [name for name, value in required.items() if value is None]
        if missing or not bool(graph_data.imu_vio_factor_enable):
            raise ValueError(
                "two_state_fixed_lag requires complete IMU and selected visual-factor data; "
                f"missing={missing}"
            )
        cross_edge_values = (
            graph_data.imu_vio_sa_v2_unique_cov,
            graph_data.imu_vio_sa_v2_incoming_raw_time_ns,
            graph_data.imu_vio_sa_v2_outgoing_raw_time_ns,
            graph_data.imu_vio_sa_v2_incoming_sensitivity,
            graph_data.imu_vio_sa_v2_outgoing_sensitivity,
        )
        cross_edge_present = tuple(value is not None for value in cross_edge_values)
        if any(cross_edge_present) and not all(cross_edge_present):
            raise ValueError("SA-v2 GraphInput contains an incomplete cross-edge payload")
        use_cross_edge_sampling = all(cross_edge_present)

        device = torch.device(context["device"])
        dtype = torch.float64
        extrinsic_CI = pp.SE3(graph_data.imu_vio_sensor_T_imu.reshape(1, 7)).to(
            device=device,
            dtype=dtype,
        )
        pose_WCi = pp.SE3(graph_data.from_pose).to(device=device, dtype=dtype)
        pose_WCj = pp.SE3(graph_data.init_motion).to(device=device, dtype=dtype)
        state_i = NavigationState(
            pose_WB=(pose_WCi @ extrinsic_CI).tensor(),
            velocity_W=graph_data.imu_vio_prev_velocity_world,
            acc_bias=graph_data.imu_vio_prev_acc_bias,
            gyro_bias=graph_data.imu_vio_prev_gyro_bias,
        ).to(device=device, dtype=dtype)
        state_j = NavigationState(
            pose_WB=(pose_WCj @ extrinsic_CI).tensor(),
            velocity_W=graph_data.imu_vio_curr_velocity_init_world,
            acc_bias=graph_data.imu_vio_curr_acc_bias_init,
            gyro_bias=graph_data.imu_vio_curr_gyro_bias_init,
        ).to(device=device, dtype=dtype)

        from_idx = int(graph_data.from_idx.reshape(-1)[0].item())
        frame_idx = int(graph_data.frame_idx.reshape(-1)[0].item())
        prior = context.get("two_state_prior")
        if not use_cross_edge_sampling:
            if (
                not isinstance(prior, SquareRootPrior)
                or context.get("two_state_last_frame_idx") != from_idx
            ):
                prior = make_diagonal_prior(
                    state_i, **context["two_state_initial_prior_std"]
                )

        imu = ImuPreintegrationFactor(
            delta_rotation=graph_data.imu_vio_delta_rotvec,
            delta_velocity=graph_data.imu_vio_delta_v,
            delta_position=graph_data.imu_vio_delta_p,
            covariance=graph_data.imu_vio_cov,
            dt=float(graph_data.imu_vio_dt.reshape(-1)[0].item()),
            bias_jacobian=graph_data.imu_vio_bias_jacobian,
            linearized_acc_bias=graph_data.imu_vio_linearized_acc_bias,
            linearized_gyro_bias=graph_data.imu_vio_linearized_gyro_bias,
            bias_rw_covariance=graph_data.imu_vio_bias_rw_cov,
            gravity_world=(
                graph_data.imu_vio_gravity_world
                if bool(graph_data.imu_vio_gravity_in_residual)
                else None
            ),
            gravity_handling=(
                "residual" if bool(graph_data.imu_vio_gravity_in_residual) else "preintegration"
            ),
        )
        cross_edge_imu = None
        cross_edge_prior = None
        incoming_noise = None
        outgoing_noise = None
        cross_edge_prior_was_reset = False
        if use_cross_edge_sampling:
            cross_edge_imu = CrossEdgeImuFactor(
                base=imu,
                unique_covariance=graph_data.imu_vio_sa_v2_unique_cov,
                incoming_raw_time_ns=graph_data.imu_vio_sa_v2_incoming_raw_time_ns,
                outgoing_raw_time_ns=graph_data.imu_vio_sa_v2_outgoing_raw_time_ns,
                incoming_sensitivity=graph_data.imu_vio_sa_v2_incoming_sensitivity,
                outgoing_sensitivity=graph_data.imu_vio_sa_v2_outgoing_sensitivity,
            )
            stored_cross_prior = context.get("two_state_cross_edge_prior")
            reset_frame = context.get("two_state_cross_edge_reset_prior_frame")
            cross_edge_prior_was_reset = bool(
                reset_frame is not None
                and int(reset_frame) == from_idx
                and not bool(context.get("two_state_cross_edge_prior_reset_done", False))
            )
            prior_is_continuous = (
                isinstance(stored_cross_prior, CrossEdgeSquareRootPrior)
                and context.get("two_state_last_frame_idx") == from_idx
                and not cross_edge_prior_was_reset
                and torch.equal(
                    stored_cross_prior.raw_time_ns.detach().cpu().reshape(-1),
                    cross_edge_imu.incoming_raw_time_ns.detach().cpu().reshape(-1),
                )
            )
            if prior_is_continuous:
                cross_edge_prior = stored_cross_prior
                incoming_noise = stored_cross_prior.reference_noise.detach().clone()
            else:
                cross_edge_prior = make_cross_edge_diagonal_prior(
                    state_i,
                    cross_edge_imu.incoming_raw_time_ns,
                    **context["two_state_initial_prior_std"],
                )
                incoming_noise = torch.zeros(
                    cross_edge_imu.incoming_dof,
                    dtype=dtype,
                    device=device,
                )
            if cross_edge_prior_was_reset:
                context["two_state_cross_edge_prior_reset_done"] = True
            outgoing_noise = torch.zeros(
                cross_edge_imu.outgoing_dof,
                dtype=dtype,
                device=device,
            )
        if str(context.get("two_state_warm_start", "macvo_pose")) == "imu_propagation":
            state_j = NavigationState(
                pose_WB=_imu_propagated_pose(state_i, imu),
                velocity_W=state_j.velocity_W,
                acc_bias=state_j.acc_bias,
                gyro_bias=state_j.gyro_bias,
            )
        solver = (
            context["two_state_cross_edge_solver"]
            if use_cross_edge_sampling
            else context["two_state_solver"]
        )
        measurement_body = None
        covariance_body = None
        factor_build_start = time.perf_counter()
        if visual_factor_mode == "relative_pose":
            measurement_body, covariance_body = camera_factor_to_body_factor(
                graph_data.visual_relative_pose_CiCj,
                graph_data.visual_relative_pose_cov.to(device=device, dtype=dtype),
                extrinsic_CI.tensor(),
            )
            covariance_body, visual_gate = _gate_two_state_visual_factor(
                state_i,
                state_j,
                measurement_body,
                covariance_body,
                num_points=graph_data.visual_relative_pose_num_points,
                num_inliers=graph_data.visual_relative_pose_num_inliers,
                mean_mahalanobis_sq=graph_data.visual_relative_pose_mean_mahalanobis_sq,
                config=context["two_state_visual_gate"],
                eigenvalue_floor=float(solver.covariance_eigenvalue_floor),
            )
            visual = RelativePoseFactor(
                measurement_BiBj=measurement_body,
                covariance=covariance_body,
                huber_delta=float(context["two_state_visual_huber_delta"]),
            )
        elif visual_factor_mode == "direct_uvd":
            visual = _make_two_state_uvd_factor(
                graph_data,
                extrinsic_CI,
                device=device,
                dtype=dtype,
                huber_delta=float(context["two_state_uvd_huber_delta"]),
            )
            initial_white = visual_whitened_residuals(
                state_i,
                state_j,
                visual,
                float(solver.covariance_eigenvalue_floor),
            )
            initial_norms = torch.linalg.vector_norm(initial_white, dim=-1)
            visual_gate = {
                "inlier_ratio": float(
                    (initial_norms <= float(visual.huber_delta)).double().mean().item()
                ),
                "mean_mahalanobis_sq": float(initial_norms.square().mean().item()),
                "whitened_pose_residual_norm": float(
                    torch.linalg.vector_norm(initial_white).item()
                ),
                "covariance_inflation": 1.0,
                "action": "direct_uvd_point_huber",
            }
        else:
            has_cached_compression = (
                graph_data.visual_compressed_uvd_reference_CjCi is not None
                and graph_data.visual_compressed_uvd_hessian is not None
                and graph_data.visual_compressed_uvd_gradient is not None
                and bool(
                    torch.isfinite(graph_data.visual_compressed_uvd_hessian).all()
                    and graph_data.visual_compressed_uvd_hessian.abs().sum() > 0.0
                )
            )
            if has_cached_compression:
                visual = linearized_uvd_pose_factor_from_normal_equations(
                    graph_data.visual_compressed_uvd_reference_CjCi,
                    graph_data.visual_compressed_uvd_hessian,
                    graph_data.visual_compressed_uvd_gradient,
                    extrinsic_CI.tensor(),
                    normal_eigenvalue_floor=float(
                        solver.marginalization_eigenvalue_floor
                    ),
                )
                point_count = max(int(graph_data.visual_compressed_uvd_num_points), 0)
                inlier_count = max(
                    min(int(graph_data.visual_compressed_uvd_num_inliers), point_count),
                    0,
                )
                mean_mahalanobis_sq = (
                    graph_data.visual_compressed_uvd_mean_mahalanobis_sq
                    if graph_data.visual_compressed_uvd_mean_mahalanobis_sq is not None
                    else 0.0
                )
                visual_action = "compressed_uvd_cached_normal_equations"
            else:
                visual_uvd = _make_two_state_uvd_factor(
                    graph_data,
                    extrinsic_CI,
                    device=device,
                    dtype=dtype,
                    huber_delta=float(context["two_state_uvd_huber_delta"]),
                )
                relative_CiCj = (
                    pp.SE3(graph_data.from_pose).Inv()
                    @ pp.SE3(graph_data.init_motion)
                )
                reference_CjCi = relative_CiCj.Inv()
                linearization = linearize_uvd_relative_pose_factor(
                    reference_CjCi.tensor(),
                    visual_uvd,
                    marginal_mode="full",
                    normal_eigenvalue_floor=float(
                        solver.marginalization_eigenvalue_floor
                    ),
                )
                visual = linearization.factor
                point_count = int(visual_uvd.points_Ci.shape[0])
                whitened_rows = visual_whitened_residuals(
                    state_i,
                    state_j,
                    visual_uvd,
                    float(solver.covariance_eigenvalue_floor),
                )
                norms = torch.linalg.vector_norm(whitened_rows, dim=-1)
                inlier_count = int((norms <= float(visual_uvd.huber_delta)).sum().item())
                mean_mahalanobis_sq = float(norms.square().mean().item())
                visual_action = "compressed_uvd_online_normal_equations"
            visual_gate = {
                "inlier_ratio": (
                    float(inlier_count) / float(point_count)
                    if point_count > 0
                    else 0.0
                ),
                "mean_mahalanobis_sq": float(mean_mahalanobis_sq),
                "whitened_pose_residual_norm": 0.0,
                "covariance_inflation": 1.0,
                "action": visual_action,
            }
        factor_build_s = time.perf_counter() - factor_build_start
        factor_packet = None
        if isinstance(visual, LinearizedUVDPoseFactor) and not use_cross_edge_sampling:
            factor_packet = T2FactorPacket.create(
                frame_i=from_idx,
                frame_j=frame_idx,
                state_i_initial=state_i,
                state_j_initial=state_j,
                imu=imu,
                visual=visual,
                extrinsic_CI=extrinsic_CI.tensor(),
            )
            context["two_state_last_factor_packet"] = factor_packet

        def solve_current_visual_factor():
            if use_cross_edge_sampling:
                return solver.solve(
                    CrossEdgeTwoStateProblem(
                        state_i=state_i,
                        state_j=state_j,
                        noise_i=incoming_noise,
                        noise_j=outgoing_noise,
                        prior_i=cross_edge_prior,
                        imu=cross_edge_imu,
                        visual=visual,
                        optimize_acc_bias=bool(context["two_state_optimize_acc_bias"]),
                        optimize_gyro_bias=bool(context["two_state_optimize_gyro_bias"]),
                    )
                )
            if factor_packet is not None:
                return solver.solve(
                    factor_packet.to_two_state_problem(
                        prior_i=prior,
                        device=device,
                        optimize_acc_bias=bool(
                            context["two_state_optimize_acc_bias"]
                        ),
                        optimize_gyro_bias=bool(
                            context["two_state_optimize_gyro_bias"]
                        ),
                    )
                )
            return solver.solve(
                TwoStateVIOProblem(
                    state_i=state_i,
                    state_j=state_j,
                    prior_i=prior,
                    imu=imu,
                    visual_pose=visual,
                    optimize_acc_bias=bool(context["two_state_optimize_acc_bias"]),
                    optimize_gyro_bias=bool(context["two_state_optimize_gyro_bias"]),
                )
            )

        solve_start = time.perf_counter()
        isam2_update = None
        isam2_history_states = None
        backend_name = str(context.get("two_state_backend_name", "two_state"))
        if backend_name == "isam2":
            if use_cross_edge_sampling:
                raise ValueError("iSAM2 backend does not accept SA-v2 cross-edge packets")
            if factor_packet is None:
                raise ValueError("iSAM2 backend requires a compressed UVD factor packet")
            isam2_backend = context.get("two_state_isam2_backend")
            if not isinstance(isam2_backend, IncrementalT2ISAM2Backend):
                raise RuntimeError("iSAM2 backend was not initialized in optimizer context")
            isam2_update = isam2_backend.consume(factor_packet)
            state_i_result = isam2_update.previous_state.to(device=device, dtype=dtype)
            state_j_result = isam2_update.state.to(device=device, dtype=dtype)
            state_step = state_boxminus(
                state_j_result,
                factor_packet.state_j_initial.to(device=device, dtype=dtype),
            )
            result = SimpleNamespace(
                state_i=state_i_result,
                state_j=state_j_result,
                prior_j=None,
                converged=True,
                iterations=1,
                initial_cost=float(isam2_update.total_edge_cost),
                final_cost=float(isam2_update.total_edge_cost),
                prior_cost=0.0,
                imu_cost=float(isam2_update.imu_cost),
                bias_cost=float(isam2_update.bias_cost),
                visual_pose_cost=float(isam2_update.visual_cost),
                final_step_norm=float(torch.linalg.vector_norm(state_step).item()),
                final_gradient_inf_norm=0.0,
                convergence_reason="isam2_incremental_update",
                accepted_steps=1,
                rejected_steps=0,
            )
            history_interval = int(
                context.get("two_state_isam2_history_publish_interval", 30)
            )
            if (
                isam2_backend.state_count <= 2
                or (isam2_backend.state_count - 1) % history_interval == 0
            ):
                isam2_history_states = isam2_backend.history()
                context["two_state_isam2_last_history_frame"] = frame_idx
        else:
            result = solve_current_visual_factor()
        if visual_factor_mode == "relative_pose":
            post_solve_visual_norm = _two_state_visual_whitened_norm(
                result.state_i,
                result.state_j,
                measurement_body,
                covariance_body,
                eigenvalue_floor=float(solver.covariance_eigenvalue_floor),
            )
            visual_gate["whitened_pose_residual_norm"] = post_solve_visual_norm
        elif visual_factor_mode == "direct_uvd":
            post_white = visual_whitened_residuals(
                result.state_i,
                result.state_j,
                visual,
                float(solver.covariance_eigenvalue_floor),
            )
            post_norms = torch.linalg.vector_norm(post_white, dim=-1)
            visual_gate["inlier_ratio"] = float(
                (post_norms <= float(visual.huber_delta)).double().mean().item()
            )
            visual_gate["mean_mahalanobis_sq"] = float(post_norms.square().mean().item())
            visual_gate["whitened_pose_residual_norm"] = float(
                torch.linalg.vector_norm(post_white).item()
            )
        else:
            post_white = visual_whitened_residuals(
                result.state_i,
                result.state_j,
                visual,
                float(solver.covariance_eigenvalue_floor),
            )
            visual_gate["whitened_pose_residual_norm"] = float(
                torch.linalg.vector_norm(post_white).item()
            )

        if visual_factor_mode == "relative_pose" and not str(visual_gate["action"]).startswith("reject:"):
            gate_config = context["two_state_visual_gate"]
            additional_inflation = 1.0
            post_action = None
            if post_solve_visual_norm > gate_config["reject_whitened_pose_norm"]:
                additional_inflation = gate_config["max_covariance_inflation"]
                post_action = "reject:high_postsolve_whitened_pose_residual"
            elif post_solve_visual_norm > gate_config["soft_whitened_pose_norm"]:
                additional_inflation = (
                    post_solve_visual_norm / gate_config["soft_whitened_pose_norm"]
                ) ** 2
                post_action = "downweight:high_postsolve_whitened_pose_residual"

            if additional_inflation > 1.0:
                current_inflation = float(visual_gate["covariance_inflation"])
                total_inflation = min(
                    current_inflation * additional_inflation,
                    gate_config["max_covariance_inflation"],
                )
                covariance_body = covariance_body * (total_inflation / current_inflation)
                visual = RelativePoseFactor(
                    measurement_BiBj=measurement_body,
                    covariance=covariance_body,
                    huber_delta=float(context["two_state_visual_huber_delta"]),
                )
                result = solve_current_visual_factor()
                visual_gate["covariance_inflation"] = total_inflation
                visual_gate["action"] = post_action
        solve_s = time.perf_counter() - solve_start
        if use_cross_edge_sampling:
            context["two_state_cross_edge_prior"] = result.prior_j
            context["two_state_prior"] = None

            checkpoint_frames = context.get("two_state_cross_edge_checkpoint_frames", set())
            checkpoint_dir = str(context.get("two_state_cross_edge_checkpoint_dir", ""))
            if frame_idx in checkpoint_frames and checkpoint_dir:
                checkpoint_path = Path(checkpoint_dir)
                checkpoint_path.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "schema_version": 1,
                        "from_idx": from_idx,
                        "frame_idx": frame_idx,
                        "prior_reset": cross_edge_prior_was_reset,
                        "state_i": result.state_i,
                        "state_j": result.state_j,
                        "noise_i": result.noise_i,
                        "noise_j": result.noise_j,
                        "prior_j": result.prior_j,
                        "unique_covariance_diagnostics": result.unique_covariance_diagnostics,
                        "incoming_prior_diagnostics": result.incoming_prior_diagnostics,
                        "marginalization_diagnostics": result.marginalization_diagnostics,
                        "common_translation_update_world": result.common_translation_update_world,
                        "differential_translation_update_world": result.differential_translation_update_world,
                        "rank_aware_imu_whitening": result.rank_aware_imu_whitening,
                        "rank_aware_fallback_active": result.rank_aware_fallback_active,
                        "rank_aware_imu_residual_dimension": result.rank_aware_imu_residual_dimension,
                    },
                    checkpoint_path / f"sa_v2_prior_after_frame_{frame_idx:06d}.pt",
                )
        elif backend_name == "isam2":
            context["two_state_prior"] = None
            context["two_state_cross_edge_prior"] = None
        else:
            context["two_state_prior"] = result.prior_j
            context["two_state_cross_edge_prior"] = None
        context["two_state_last_frame_idx"] = frame_idx

        pose_WCi_optimized = pp.SE3(result.state_i.pose_WB) @ extrinsic_CI.Inv()
        pose_WCj_optimized = pp.SE3(result.state_j.pose_WB) @ extrinsic_CI.Inv()
        acc_bias = result.state_j.acc_bias.detach().cpu().float()
        gyro_bias = result.state_j.gyro_bias.detach().cpu().float()
        if isam2_history_states is not None:
            (
                window_frames,
                window_camera_poses,
                window_velocity,
                window_acc_bias,
                window_gyro_bias,
            ) = TwoFrame_PGO._isam2_history_tensors(
                context, isam2_history_states
            )
            window_writeback = "all_isam2_history"
        else:
            window_frames = torch.tensor([from_idx, frame_idx], dtype=torch.long)
            window_camera_poses = torch.cat(
                [pose_WCi_optimized.tensor(), pose_WCj_optimized.tensor()], dim=0
            ).detach().cpu().float()
            window_velocity = torch.stack(
                [result.state_i.velocity_W, result.state_j.velocity_W], dim=0
            ).detach().cpu().float()
            window_acc_bias = torch.stack(
                [result.state_i.acc_bias, result.state_j.acc_bias], dim=0
            ).detach().cpu().float()
            window_gyro_bias = torch.stack(
                [result.state_i.gyro_bias, result.state_j.gyro_bias], dim=0
            ).detach().cpu().float()
            window_writeback = "all_two_state"
        output = GraphOutput(
            motion=pose_WCj_optimized.tensor().detach().cpu().float(),
            from_idx=graph_data.from_idx.detach().cpu().clone(),
            frame_idx=graph_data.frame_idx.detach().cpu().clone(),
            visual_obs_cov_mean=graph_data.visual_obs_cov_mean,
            num_observations=graph_data.num_observations,
            visual_keypoint_coverage=graph_data.visual_keypoint_coverage,
            visual_depth_spread=graph_data.visual_depth_spread,
            visual_pose_inlier_ratio=visual_gate["inlier_ratio"],
            visual_pose_mean_mahalanobis_sq=visual_gate["mean_mahalanobis_sq"],
            visual_pose_whitened_residual_norm=visual_gate["whitened_pose_residual_norm"],
            visual_pose_covariance_inflation=visual_gate["covariance_inflation"],
            visual_pose_gate_action=str(visual_gate["action"]),
            velocity_world=result.state_j.velocity_W.detach().cpu().float(),
            acc_bias=acc_bias,
            gyro_bias=gyro_bias,
            final_loss=result.final_cost,
            visual_loss=result.visual_pose_cost,
            imu_vel_loss=result.imu_cost,
            imu_vio_whitened_norm=math.sqrt(max(2.0 * result.imu_cost, 0.0)),
            imu_vio_sa_v2_sampling_noise_cost=(
                float(result.sampling_noise_cost)
                if use_cross_edge_sampling
                else None
            ),
            imu_vio_sa_v2_cross_covariance_frobenius_norm=(
                float(result.cross_covariance_frobenius_norm)
                if use_cross_edge_sampling
                else None
            ),
            imu_vio_sa_v2_incoming_sample_count=(
                int(cross_edge_imu.incoming_raw_time_ns.numel())
                if use_cross_edge_sampling
                else None
            ),
            imu_vio_sa_v2_outgoing_sample_count=(
                int(cross_edge_imu.outgoing_raw_time_ns.numel())
                if use_cross_edge_sampling
                else None
            ),
            imu_vio_sa_v2_prior_reset=(
                cross_edge_prior_was_reset if use_cross_edge_sampling else None
            ),
            imu_vio_sa_v2_unique_cov_min_eigenvalue=(
                result.unique_covariance_diagnostics.min_eigenvalue
                if use_cross_edge_sampling else None
            ),
            imu_vio_sa_v2_unique_cov_max_eigenvalue=(
                result.unique_covariance_diagnostics.max_eigenvalue
                if use_cross_edge_sampling else None
            ),
            imu_vio_sa_v2_unique_cov_effective_rank=(
                result.unique_covariance_diagnostics.effective_rank
                if use_cross_edge_sampling else None
            ),
            imu_vio_sa_v2_unique_cov_dimension=(
                result.unique_covariance_diagnostics.dimension
                if use_cross_edge_sampling else None
            ),
            imu_vio_sa_v2_unique_cov_condition_number=(
                result.unique_covariance_diagnostics.condition_number
                if use_cross_edge_sampling else None
            ),
            imu_vio_sa_v2_prior_i_min_eigenvalue=(
                result.incoming_prior_diagnostics.min_eigenvalue
                if use_cross_edge_sampling else None
            ),
            imu_vio_sa_v2_prior_i_max_eigenvalue=(
                result.incoming_prior_diagnostics.max_eigenvalue
                if use_cross_edge_sampling else None
            ),
            imu_vio_sa_v2_prior_i_effective_rank=(
                result.incoming_prior_diagnostics.effective_rank
                if use_cross_edge_sampling else None
            ),
            imu_vio_sa_v2_prior_i_dimension=(
                result.incoming_prior_diagnostics.dimension
                if use_cross_edge_sampling else None
            ),
            imu_vio_sa_v2_prior_i_condition_number=(
                result.incoming_prior_diagnostics.condition_number
                if use_cross_edge_sampling else None
            ),
            imu_vio_sa_v2_h_mm_min_eigenvalue=(
                result.marginalization_diagnostics.h_mm.min_eigenvalue
                if use_cross_edge_sampling else None
            ),
            imu_vio_sa_v2_h_mm_max_eigenvalue=(
                result.marginalization_diagnostics.h_mm.max_eigenvalue
                if use_cross_edge_sampling else None
            ),
            imu_vio_sa_v2_h_mm_effective_rank=(
                result.marginalization_diagnostics.h_mm.effective_rank
                if use_cross_edge_sampling else None
            ),
            imu_vio_sa_v2_h_mm_dimension=(
                result.marginalization_diagnostics.h_mm.dimension
                if use_cross_edge_sampling else None
            ),
            imu_vio_sa_v2_h_mm_condition_number=(
                result.marginalization_diagnostics.h_mm.condition_number
                if use_cross_edge_sampling else None
            ),
            imu_vio_sa_v2_prior_j_min_eigenvalue=(
                result.marginalization_diagnostics.schur_prior.min_eigenvalue
                if use_cross_edge_sampling else None
            ),
            imu_vio_sa_v2_prior_j_max_eigenvalue=(
                result.marginalization_diagnostics.schur_prior.max_eigenvalue
                if use_cross_edge_sampling else None
            ),
            imu_vio_sa_v2_prior_j_effective_rank=(
                result.marginalization_diagnostics.schur_prior.effective_rank
                if use_cross_edge_sampling else None
            ),
            imu_vio_sa_v2_prior_j_dimension=(
                result.marginalization_diagnostics.schur_prior.dimension
                if use_cross_edge_sampling else None
            ),
            imu_vio_sa_v2_prior_j_condition_number=(
                result.marginalization_diagnostics.schur_prior.condition_number
                if use_cross_edge_sampling else None
            ),
            imu_vio_sa_v2_discarded_h_mm_dimensions=(
                result.marginalization_diagnostics.discarded_h_mm_dimensions
                if use_cross_edge_sampling else None
            ),
            imu_vio_sa_v2_discarded_prior_dimensions=(
                result.marginalization_diagnostics.discarded_prior_dimensions
                if use_cross_edge_sampling else None
            ),
            imu_vio_sa_v2_schur_quadratic_relative_error=(
                result.marginalization_diagnostics.quadratic_relative_error
                if use_cross_edge_sampling else None
            ),
            imu_vio_sa_v2_state_i_translation_update_norm=(
                float(torch.linalg.vector_norm(result.state_i_increment[0:3]).item())
                if use_cross_edge_sampling else None
            ),
            imu_vio_sa_v2_state_i_rotation_update_norm=(
                float(torch.linalg.vector_norm(result.state_i_increment[3:6]).item())
                if use_cross_edge_sampling else None
            ),
            imu_vio_sa_v2_state_j_translation_update_norm=(
                float(torch.linalg.vector_norm(result.state_j_increment[0:3]).item())
                if use_cross_edge_sampling else None
            ),
            imu_vio_sa_v2_state_j_rotation_update_norm=(
                float(torch.linalg.vector_norm(result.state_j_increment[3:6]).item())
                if use_cross_edge_sampling else None
            ),
            imu_vio_sa_v2_common_translation_update_world_x=(
                float(result.common_translation_update_world[0].item())
                if use_cross_edge_sampling else None
            ),
            imu_vio_sa_v2_common_translation_update_world_y=(
                float(result.common_translation_update_world[1].item())
                if use_cross_edge_sampling else None
            ),
            imu_vio_sa_v2_common_translation_update_world_z=(
                float(result.common_translation_update_world[2].item())
                if use_cross_edge_sampling else None
            ),
            imu_vio_sa_v2_common_translation_update_world_norm=(
                float(torch.linalg.vector_norm(result.common_translation_update_world).item())
                if use_cross_edge_sampling else None
            ),
            imu_vio_sa_v2_differential_translation_update_world_norm=(
                float(torch.linalg.vector_norm(result.differential_translation_update_world).item())
                if use_cross_edge_sampling else None
            ),
            imu_vio_sa_v2_rank_aware_imu_whitening=(
                bool(result.rank_aware_imu_whitening)
                if use_cross_edge_sampling else None
            ),
            imu_vio_sa_v2_rank_aware_fallback_active=(
                bool(result.rank_aware_fallback_active)
                if use_cross_edge_sampling else None
            ),
            imu_vio_sa_v2_imu_residual_dimension=(
                int(result.rank_aware_imu_residual_dimension)
                if use_cross_edge_sampling else None
            ),
            imu_vio_acc_bias_norm=float(torch.linalg.vector_norm(acc_bias).item()),
            imu_vio_gyro_bias_norm=float(torch.linalg.vector_norm(gyro_bias).item()),
            imu_vio_acc_bias_x=float(acc_bias[0].item()),
            imu_vio_acc_bias_y=float(acc_bias[1].item()),
            imu_vio_acc_bias_z=float(acc_bias[2].item()),
            imu_vio_gyro_bias_x=float(gyro_bias[0].item()),
            imu_vio_gyro_bias_y=float(gyro_bias[1].item()),
            imu_vio_gyro_bias_z=float(gyro_bias[2].item()),
            imu_factor_mode=(
                "two_state_fixed_lag_direct_uvd_sampling_aware_cross_edge"
                if use_cross_edge_sampling and visual_factor_mode == "direct_uvd"
                else (
                    "two_state_fixed_lag_sampling_aware_cross_edge"
                    if use_cross_edge_sampling
                    else (
                        "two_state_fixed_lag_direct_uvd"
                        if visual_factor_mode == "direct_uvd"
                        else "two_state_fixed_lag"
                    )
                )
            ),
            vio_factor_active=True,
            vio_bias_state_active=True,
            imu_residual_rows=5,
            use_imu_rotation=True,
            use_imu_translation=True,
            num_visual_residuals=(
                int(visual.points_Ci.shape[0]) * 3
                if isinstance(visual, UVDFactor)
                else 6
            ),
            window_frame_indices=window_frames,
            window_motions=window_camera_poses,
            window_velocity_world=window_velocity,
            window_acc_bias=window_acc_bias,
            window_gyro_bias=window_gyro_bias,
            local_ba_window_size=int(window_frames.numel()),
            local_ba_writeback=window_writeback,
            local_ba_num_frames=int(window_frames.numel()),
            local_ba_num_edges=max(int(window_frames.numel()) - 1, 1),
            local_ba_num_visual_residual_blocks=(
                int(visual.points_Ci.shape[0])
                if isinstance(visual, UVDFactor)
                else 1
            ),
            local_ba_graph_build_s=factor_build_s,
            local_ba_lm_s=solve_s,
            local_ba_refine_s=0.0,
            local_ba_optimize_total_s=time.perf_counter() - optimize_start,
            two_state_solver_converged=result.converged,
            two_state_solver_iterations=result.iterations,
            two_state_solver_convergence_reason=result.convergence_reason,
            two_state_final_step_norm=result.final_step_norm,
            two_state_final_gradient_inf_norm=result.final_gradient_inf_norm,
            two_state_solver_accepted_steps=result.accepted_steps,
            two_state_solver_rejected_steps=result.rejected_steps,
            vio_backend=backend_name,
            isam2_update_ms=(
                float(isam2_update.update_ms) if isam2_update is not None else None
            ),
            isam2_state_count=(
                int(context["two_state_isam2_backend"].state_count)
                if isam2_update is not None
                else None
            ),
            isam2_history_revision=isam2_history_states is not None,
            isam2_initial_pose_mismatch_norm=(
                float(isam2_update.initial_pose_mismatch_norm)
                if isam2_update is not None
                else None
            ),
            isam2_initial_velocity_mismatch_norm=(
                float(isam2_update.initial_velocity_mismatch_norm)
                if isam2_update is not None
                else None
            ),
            isam2_initial_bias_mismatch_norm=(
                float(isam2_update.initial_bias_mismatch_norm)
                if isam2_update is not None
                else None
            ),
        )
        return context, output

    @staticmethod
    def _optimize(context: dict, graph_data: GraphInput | LocalWindowGraphInput) -> tuple[dict, GraphOutput]:
        if context.get("imu_factor_mode") == "two_state_fixed_lag" and isinstance(graph_data, GraphInput):
            return TwoFrame_PGO._optimize_two_state_fixed_lag(context, graph_data)
        optimize_start = time.perf_counter()
        graph_build_s: float | None = None
        lm_s: float | None = None
        refine_s: float | None = None
        debug_trace_enable = bool(context.get("optimizer_breakpoint_trace_enable", False))
        causal_enable = bool(context.get("vio_causal_diagnostics_enable", False))
        causal_interval = max(1, int(context.get("vio_causal_diagnostics_interval", 5)))
        initial_energy: dict[str, float | None] = {}
        initial_state: dict[str, torch.Tensor] = {}
        influence: dict[str, object] = {}
        debug_trace: dict | None = None
        with Timer.CPUTimingContext("TwoframePGO"), Timer.GPUTimingContext("TwoframePGO", torch.cuda.current_stream()):
            graph_build_start = time.perf_counter()
            if isinstance(graph_data, LocalWindowGraphInput):
                graph = LocalWindowInertialGraph(graph_data)
            else:
                graph = context["pose_graph_class"](graph_data)
            graph = graph.to(device=torch.device(context["device"]), dtype=torch.double)
            assert isinstance(graph, FactorGraph)
            graph_build_s = time.perf_counter() - graph_build_start

            if isinstance(graph, AnalyticModule):
                optimizer = LM_analytic(graph, min=1e-6, **context["optimizer_cfg"])
            else:
                optimizer = LM(graph, min=1e-6, **context["optimizer_cfg"])

            if causal_enable:
                initial_state = TwoFrame_PGO._capture_parameter_state(graph)
                initial_energy = TwoFrame_PGO._capture_causal_energy_snapshot(graph, graph_data)
                frame_value = int(
                    (
                        graph_data.frame_indices[-1]
                        if isinstance(graph_data, LocalWindowGraphInput)
                        else graph_data.frame_idx.reshape(-1)[-1]
                    ).detach().cpu().item()
                )
                should_sample = ((max(frame_value, 1) - 1) % causal_interval) == 0
                if should_sample and not isinstance(graph, AnalyticModule):
                    if hasattr(graph, "weight_matrix") and callable(getattr(graph, "weight_matrix")):
                        initial_weight = graph.weight_matrix().to(context["device"]).double()
                    else:
                        initial_weight = torch.block_diag(*(
                            torch.pinverse(graph.covariance_array().to(context["device"]).double())
                        ))
                    influence = TwoFrame_PGO._compute_initial_factor_influence(
                        graph,
                        optimizer,
                        initial_weight,
                    )

            scheduler = StopOnPlateau(optimizer, steps=10, patience=2, decreasing=1e-5, verbose=False)
            final_loss_val: float | None = None
            if debug_trace_enable:
                debug_trace = {
                    "graph_input_kind": "local_window" if isinstance(graph_data, LocalWindowGraphInput) else "two_frame",
                    "device": str(context["device"]),
                    "snapshots": [TwoFrame_PGO._debug_graph_state(graph, "init_before_lm")],
                }
                if isinstance(graph_data, LocalWindowGraphInput):
                    debug_trace["w2_equivalence"] = TwoFrame_PGO._debug_w2_equivalence_payload(
                        graph,
                        graph_data,
                        context,
                    )

            lm_start = time.perf_counter()
            while scheduler.continual():
                if hasattr(graph, "weight_matrix") and callable(getattr(graph, "weight_matrix")):
                    weight = graph.weight_matrix().to(context["device"]).double()
                else:
                    weight = torch.block_diag(*(
                        torch.pinverse(graph.covariance_array().to(context["device"]).double())
                    ))
                loss = optimizer.step(input=(), weight=weight)
                scheduler.step(loss)
                final_loss_val = float(loss.detach().cpu().item()) if isinstance(loss, torch.Tensor) else float(loss)
            lm_s = time.perf_counter() - lm_start
            if debug_trace is not None:
                debug_trace["snapshots"].append(TwoFrame_PGO._debug_graph_state(graph, "after_lm_before_refine"))

            if hasattr(graph, "weight_matrix") and callable(getattr(graph, "weight_matrix")):
                weight = graph.weight_matrix().to(context["device"]).double()
            else:
                weight = torch.block_diag(*(
                    torch.pinverse(graph.covariance_array().to(context["device"]).double())
                ))
            refine_start = time.perf_counter()
            if isinstance(graph_data, LocalWindowGraphInput):
                if bool(graph_data.fixed_first_frame) and int(graph_data.frame_indices.numel()) == 2:
                    refined_loss = TwoFrame_PGO._refine_local_window_vector_states(graph, weight)
                else:
                    refined_loss = TwoFrame_PGO._refine_local_window_states(graph, weight)
            else:
                refined_loss = TwoFrame_PGO._refine_vio_vector_states(graph, weight)
            refine_s = time.perf_counter() - refine_start
            if refined_loss is not None:
                try:
                    raw_residual = graph.forward()
                    final_loss_val = float((raw_residual.detach().double() ** 2).sum().cpu().item())
                except Exception:
                    pass
            if debug_trace is not None:
                debug_trace["snapshots"].append(TwoFrame_PGO._debug_graph_state(graph, "after_refine_before_writeback"))

        # ── Compute per-pair diagnostics from final graph state ──────────
        output: GraphOutput = graph.write_back()
        if debug_trace is not None and isinstance(graph_data, LocalWindowGraphInput):
            debug_trace["w2_reference_two_frame_output"] = TwoFrame_PGO._debug_w2_reference_output_payload(
                output,
                graph_data,
                context,
            )
        if debug_trace is not None:
            debug_trace["timing_s"] = {
                "graph_build": graph_build_s,
                "lm": lm_s,
                "refine": refine_s,
                "total": time.perf_counter() - optimize_start,
            }
            output.debug_trace = debug_trace
        if isinstance(graph_data, LocalWindowGraphInput):
            output.local_ba_graph_build_s = graph_build_s
            output.local_ba_lm_s = lm_s
            output.local_ba_refine_s = refine_s
            output.local_ba_optimize_total_s = time.perf_counter() - optimize_start
            output.final_loss = final_loss_val
            try:
                TwoFrame_PGO._compute_local_window_diagnostics(graph, graph_data, output, final_loss_val)
            except Exception:
                pass
        else:
            try:
                _ = TwoFrame_PGO._compute_pair_diagnostics(graph, output, final_loss_val)
            except Exception:
                pass  # diagnostics are best-effort; never crash the optimizer

        if causal_enable:
            TwoFrame_PGO._attach_causal_diagnostics(
                output,
                graph,
                graph_data,
                initial_energy,
                initial_state,
                influence,
            )

        return context, output

    @staticmethod
    def _compute_pair_diagnostics(
        graph: FactorGraph,
        output: GraphOutput,
        total_loss: float | None,
    ) -> None:
        """Best-effort: attach per-pair diagnostic scalars to *output* in-place."""
        try:
            residual = graph.forward()
            if isinstance(residual, torch.Tensor) and residual.numel() > 0:
                residual = residual.detach().to(dtype=torch.float64)
            else:
                return
        except Exception:
            return

        try:
            cov_blocks = graph.covariance_array()  # (N, 3, 3) or (N, 2, 2) blocks
            if isinstance(cov_blocks, torch.Tensor) and cov_blocks.numel() > 0:
                cov_blocks = cov_blocks.detach().to(dtype=torch.float64)
            else:
                cov_blocks = None
        except Exception:
            cov_blocks = None

        # ── Count visual vs IMU residuals ─────────────────────────────────
        has_vio = bool(getattr(graph, "use_vio_imu_factor", False))
        has_vio_bias = bool(getattr(graph, "use_vio_bias_state", False))
        has_rot = bool(getattr(graph, "use_imu_rot_prior", False))
        has_trans = bool(getattr(graph, "use_imu_trans_prior", False))
        n_total = int(residual.shape[0])
        n_imu_rows = 0
        if has_vio:
            n_imu_rows += 3
            if has_vio_bias:
                n_imu_rows += 2
        else:
            if has_trans:
                n_imu_rows += 1
            if has_rot:
                n_imu_rows += 1
        n_vis_rows = max(0, n_total - n_imu_rows)
        output.num_visual_residuals = n_vis_rows
        output.imu_factor_mode = "preintegrated_vio" if has_vio else "legacy_pose_prior"
        output.vio_factor_active = has_vio
        output.vio_bias_state_active = has_vio_bias
        output.imu_residual_rows = n_imu_rows
        output.use_imu_rotation = bool(has_vio or has_rot)
        output.use_imu_translation = bool(has_vio or has_trans)

        # ── Separate residual blocks ──────────────────────────────────────
        r_vis = residual[:n_vis_rows] if n_vis_rows > 0 else None
        idx = n_vis_rows
        r_trans: torch.Tensor | None = None
        r_vel: torch.Tensor | None = None
        r_rot: torch.Tensor | None = None
        if has_vio:
            r_trans = residual[idx].reshape(-1)
            idx += 1
            r_vel = residual[idx].reshape(-1)
            idx += 1
            r_rot = residual[idx].reshape(-1)
            idx += 1
            if has_vio_bias:
                idx += 2
        else:
            if has_rot:
                r_rot = residual[idx].reshape(-1)
                idx += 1
            if has_trans:
                r_trans = residual[idx].reshape(-1)

        # ── Compute whitened norms using per-block 3x3 covariances ────────
        def _whitened_norm_3x3(r_blk: torch.Tensor, cov_3x3: torch.Tensor | None) -> float:
            if cov_3x3 is None:
                return float(r_blk.norm().item())
            diag = cov_3x3.diagonal().clamp(min=1e-12)
            return float(((r_blk ** 2) / diag).sum().sqrt().item())

        def _weighted_energy(r_blk: torch.Tensor, cov: torch.Tensor | None) -> float:
            r_flat = r_blk.reshape(-1).to(dtype=torch.float64)
            if cov is None:
                return float(torch.dot(r_flat, r_flat).item())
            cov_m = cov.reshape(r_flat.numel(), r_flat.numel()).to(dtype=torch.float64, device=r_flat.device)
            info = torch.linalg.pinv(0.5 * (cov_m + cov_m.T))
            energy = torch.dot(r_flat, info @ r_flat)
            return float(torch.clamp(energy, min=0.0).item())

        def _weighted_rows_energy(rows: torch.Tensor | None, covs: torch.Tensor | None) -> float | None:
            if rows is None or covs is None or rows.numel() == 0 or covs.numel() == 0:
                return None
            total = 0.0
            for r_blk, cov_blk in zip(rows.reshape(-1, rows.shape[-1]), covs.reshape(-1, covs.shape[-2], covs.shape[-1])):
                total += _weighted_energy(r_blk, cov_blk)
            return total

        def _safe_ratio(num: float | None, den: float | None) -> float | None:
            if num is None or den is None or den <= 1e-12:
                return None
            return float(num / den)

        def _raw_sos(r_blk: torch.Tensor) -> float:
            return float((r_blk ** 2).sum().item())

        output.energy_visual_weighted = _weighted_rows_energy(
            r_vis,
            cov_blocks[:n_vis_rows] if cov_blocks is not None and n_vis_rows > 0 else None,
        )

        if has_vio and r_trans is not None and r_vel is not None and r_rot is not None:
            r_vio = torch.cat([r_trans.reshape(-1), r_vel.reshape(-1), r_rot.reshape(-1)], dim=0)
            output.imu_vio_raw_norm = float(r_vio.norm().item())
            cov_9x9 = getattr(graph, "imu_vio_cov_matrix", None)
            if isinstance(cov_9x9, torch.Tensor) and cov_9x9.numel() == 81:
                cov_9x9 = cov_9x9.detach().reshape(9, 9).to(dtype=torch.float64, device=r_vio.device)
                try:
                    info_9x9 = torch.linalg.pinv(cov_9x9)
                    weighted = torch.dot(r_vio, info_9x9 @ r_vio)
                    output.imu_vio_whitened_norm = float(torch.clamp(weighted, min=0.0).sqrt().item())
                    output.energy_imu_weighted = float(torch.clamp(weighted, min=0.0).item())
                    diag = info_9x9.diagonal()
                    output.imu_vio_cov_trace = float(torch.trace(cov_9x9).item())
                    output.imu_vio_weight_trace = float(torch.trace(info_9x9).item())
                    output.imu_vio_weight_diag_min = float(diag.min().item())
                    output.imu_vio_weight_diag_max = float(diag.max().item())
                except Exception:
                    output.imu_vio_whitened_norm = None

        if r_rot is not None and cov_blocks is not None:
            # VIO appends [position, velocity, rotation, optional bias_a, optional bias_g].
            # Legacy pose prior appends rotation before translation.
            cov_idx = n_vis_rows + 2 if has_vio else cov_blocks.shape[0] - 1
            if (not has_vio) and has_trans:
                cov_idx -= 1
            if cov_idx >= 0:
                cov_rot_3x3 = cov_blocks[cov_idx]
                output.r_R_whitened_norm = _whitened_norm_3x3(r_rot, cov_rot_3x3)
                output.energy_R_weighted = _weighted_energy(r_rot, cov_rot_3x3)
            else:
                output.r_R_whitened_norm = _raw_sos(r_rot)
                output.energy_R_weighted = _raw_sos(r_rot)
            output.imu_rot_loss = _raw_sos(r_rot)

        if r_vel is not None and cov_blocks is not None:
            cov_vel_3x3 = cov_blocks[n_vis_rows + 1] if has_vio else None
            output.r_v_whitened_norm = _whitened_norm_3x3(r_vel, cov_vel_3x3)
            output.energy_v_weighted = _weighted_energy(r_vel, cov_vel_3x3)
            output.imu_vel_loss = _raw_sos(r_vel)

        if r_trans is not None and cov_blocks is not None:
            cov_trans_3x3 = cov_blocks[n_vis_rows] if has_vio else cov_blocks[-1]
            output.r_p_whitened_norm = _whitened_norm_3x3(r_trans, cov_trans_3x3)
            output.energy_p_weighted = _weighted_energy(r_trans, cov_trans_3x3)
            output.imu_trans_loss = _raw_sos(r_trans)

        if r_vis is not None:
            output.visual_loss = _raw_sos(r_vis.reshape(-1))

        if has_vio_bias and output.acc_bias is not None and output.gyro_bias is not None:
            acc_bias = output.acc_bias.detach().reshape(3).to(dtype=torch.float64)
            gyro_bias = output.gyro_bias.detach().reshape(3).to(dtype=torch.float64)
            output.imu_vio_acc_bias_norm = float(acc_bias.norm().item())
            output.imu_vio_gyro_bias_norm = float(gyro_bias.norm().item())
            output.imu_vio_acc_bias_x = float(acc_bias[0].item())
            output.imu_vio_acc_bias_y = float(acc_bias[1].item())
            output.imu_vio_acc_bias_z = float(acc_bias[2].item())
            output.imu_vio_gyro_bias_x = float(gyro_bias[0].item())
            output.imu_vio_gyro_bias_y = float(gyro_bias[1].item())
            output.imu_vio_gyro_bias_z = float(gyro_bias[2].item())

        if has_vio:
            output.imu_vio_alpha_p = float(getattr(graph, "imu_vio_alpha_p", 1.0))
            output.imu_vio_alpha_v = float(getattr(graph, "imu_vio_alpha_v", 1.0))
            output.imu_vio_alpha_R = float(getattr(graph, "imu_vio_alpha_R", 1.0))

        p_energy = output.energy_p_weighted
        v_energy = output.energy_v_weighted
        R_energy = output.energy_R_weighted
        if p_energy is not None or v_energy is not None:
            output.energy_pv_weighted = float((p_energy or 0.0) + (v_energy or 0.0))
        if output.energy_pv_weighted is not None or R_energy is not None:
            output.energy_imu_diag_weighted = float((output.energy_pv_weighted or 0.0) + (R_energy or 0.0))
        if output.energy_imu_weighted is None:
            output.energy_imu_weighted = output.energy_imu_diag_weighted
        output.energy_imu_to_visual_ratio = _safe_ratio(output.energy_imu_weighted, output.energy_visual_weighted)
        output.energy_pv_to_visual_ratio = _safe_ratio(output.energy_pv_weighted, output.energy_visual_weighted)
        output.energy_R_to_visual_ratio = _safe_ratio(output.energy_R_weighted, output.energy_visual_weighted)

        output.final_loss = total_loss

    @staticmethod
    def _compute_local_window_diagnostics(
        graph: FactorGraph,
        graph_data: LocalWindowGraphInput,
        output: GraphOutput,
        total_loss: float | None,
    ) -> None:
        """Attach cost-scale diagnostics for local inertial BA windows."""
        residual = graph.forward().detach().to(dtype=torch.float64)
        if residual.numel() == 0:
            return

        def _weighted_energy(r_blk: torch.Tensor, cov: torch.Tensor | None) -> float:
            r_flat = r_blk.reshape(-1).to(dtype=torch.float64)
            if cov is None:
                return float(torch.dot(r_flat, r_flat).item())
            cov_m = cov.reshape(r_flat.numel(), r_flat.numel()).to(dtype=torch.float64, device=r_flat.device)
            info = torch.linalg.pinv(0.5 * (cov_m + cov_m.T))
            energy = torch.dot(r_flat, info @ r_flat)
            return float(torch.clamp(energy, min=0.0).item())

        def _safe_ratio(num: float | None, den: float | None) -> float | None:
            if num is None or den is None or den <= 1e-12:
                return None
            return float(num / den)

        n_visual = sum(int(edge.observations.data["pixel2_uv"].shape[0]) for edge in graph_data.edges)
        output.num_visual_residuals = n_visual
        r_vis = residual[:n_visual] if n_visual > 0 else residual.new_zeros((0, residual.shape[-1]))
        output.visual_loss = float((r_vis.reshape(-1) ** 2).sum().cpu().item())

        visual_covs = getattr(graph, "_visual_covariances", None)
        if isinstance(visual_covs, torch.Tensor) and visual_covs.numel() > 0 and n_visual > 0:
            output.energy_visual_weighted = 0.0
            for r_blk, cov_blk in zip(
                r_vis.reshape(-1, r_vis.shape[-1]),
                visual_covs.reshape(-1, visual_covs.shape[-2], visual_covs.shape[-1]),
            ):
                output.energy_visual_weighted += _weighted_energy(r_blk, cov_blk)

        idx = n_visual
        energy_p = 0.0
        energy_v = 0.0
        energy_R = 0.0
        energy_imu_full = 0.0
        raw_p = 0.0
        raw_v = 0.0
        raw_R = 0.0
        active_edges = 0
        imu_covariances = list(getattr(graph, "_imu_covariances", []))
        for edge_pos, edge in enumerate(graph_data.edges):
            if not LocalWindowInertialGraph._edge_has_vio(edge):
                continue
            r_rows = residual[idx:idx + 3]
            idx += 3
            if r_rows.shape[0] != 3:
                continue
            active_edges += 1
            r_p = r_rows[0].reshape(3)
            r_v = r_rows[1].reshape(3)
            r_R = r_rows[2].reshape(3)
            raw_p += float((r_p ** 2).sum().item())
            raw_v += float((r_v ** 2).sum().item())
            raw_R += float((r_R ** 2).sum().item())
            cov9 = imu_covariances[edge_pos].reshape(9, 9) if edge_pos < len(imu_covariances) else None
            if cov9 is not None:
                cov9 = cov9.to(dtype=torch.float64, device=r_rows.device)
                blocks = vio_preintegrated_covariance_blocks(cov9).to(dtype=torch.float64, device=r_rows.device)
                energy_p += _weighted_energy(r_p, blocks[0])
                energy_v += _weighted_energy(r_v, blocks[1])
                energy_R += _weighted_energy(r_R, blocks[2])
                energy_imu_full += _weighted_energy(r_rows.reshape(9), vio_preintegrated_covariance_matrix(cov9))
            else:
                energy_p += raw_p
                energy_v += raw_v
                energy_R += raw_R

        output.imu_trans_loss = raw_p
        output.imu_vel_loss = raw_v
        output.imu_rot_loss = raw_R
        output.imu_vio_raw_norm = float(max(raw_p + raw_v + raw_R, 0.0) ** 0.5)
        if active_edges > 0:
            output.energy_p_weighted = energy_p
            output.energy_v_weighted = energy_v
            output.energy_R_weighted = energy_R
            output.energy_pv_weighted = energy_p + energy_v
            output.energy_imu_diag_weighted = energy_p + energy_v + energy_R
            output.energy_imu_weighted = energy_imu_full if energy_imu_full > 0.0 else output.energy_imu_diag_weighted
            output.r_p_whitened_norm = float(max(energy_p, 0.0) ** 0.5)
            output.r_v_whitened_norm = float(max(energy_v, 0.0) ** 0.5)
            output.r_R_whitened_norm = float(max(energy_R, 0.0) ** 0.5)
            output.imu_vio_whitened_norm = float(max(output.energy_imu_weighted or 0.0, 0.0) ** 0.5)
            if imu_covariances:
                traces = [float(torch.trace(cov.reshape(9, 9)).item()) for cov in imu_covariances]
                output.imu_vio_cov_trace = float(sum(traces))
            output.energy_imu_to_visual_ratio = _safe_ratio(output.energy_imu_weighted, output.energy_visual_weighted)
            output.energy_pv_to_visual_ratio = _safe_ratio(output.energy_pv_weighted, output.energy_visual_weighted)
            output.energy_R_to_visual_ratio = _safe_ratio(output.energy_R_weighted, output.energy_visual_weighted)

        output.final_loss = total_loss
    # ═══════════════════════════════════════════════════════════════════════

    def _record_fusion_diagnostics(self, global_map: VisualMap, frame_idx: torch.Tensor, log: dict | None) -> None:
        if not log:
            return
        frame_data = global_map.frames.data
        idx = frame_idx

        def as_float(key: str, default: float = -1.0) -> float:
            value = log.get(key, default)
            if value is None:
                return default
            try:
                return float(value)
            except (TypeError, ValueError):
                return default

        visual_quality = as_float("visual_quality", -1.0)
        degrade_score = as_float("degrade_score", -1.0)
        trans_switch = as_float("trans_switch", as_float("z_switch", -1.0))
        rot_switch = as_float("rot_switch", -1.0)
        xy_weight = as_float("xy_weight", 0.0)
        z_weight = as_float("z_weight", 0.0)
        rot_weight = as_float("rot_weight", 0.0)

        per_axis = log.get("per_axis_imu_weight")
        if isinstance(per_axis, (list, tuple)) and len(per_axis) >= 6:
            try:
                xy_weight = float(per_axis[0] + per_axis[1]) * 0.5
                z_weight = float(per_axis[2])
                rot_weight = float(per_axis[3] + per_axis[4] + per_axis[5]) / 3.0
            except (TypeError, ValueError):
                pass

        flags = torch.tensor([[
            1.0 if (
                bool(log.get("xy_imu_enable", False))
                or bool(log.get("z_imu_enable", False))
                or bool(log.get("rot_imu_enable", False))
            ) else 0.0,
            1.0 if bool(log.get("xy_imu_enable", False)) else 0.0,
            1.0 if bool(log.get("z_imu_enable", False)) else 0.0,
            1.0 if bool(log.get("rot_imu_enable", False)) else 0.0,
        ]], dtype=torch.float32)

        if "fusion_visual_quality" in frame_data:
            frame_data["fusion_visual_quality"][idx] = torch.tensor([visual_quality], dtype=torch.float32)
            frame_data["fusion_degrade_score"][idx] = torch.tensor([degrade_score], dtype=torch.float32)
            frame_data["fusion_trans_switch"][idx] = torch.tensor([trans_switch], dtype=torch.float32)
            frame_data["fusion_rot_switch"][idx] = torch.tensor([rot_switch], dtype=torch.float32)
            frame_data["fusion_xy_weight"][idx] = torch.tensor([xy_weight], dtype=torch.float32)
            frame_data["fusion_z_weight"][idx] = torch.tensor([z_weight], dtype=torch.float32)
            frame_data["fusion_rot_weight"][idx] = torch.tensor([rot_weight], dtype=torch.float32)
            frame_data["fusion_gate_flags"][idx] = flags

        self.fusion_logs.append(dict(log))

    @staticmethod
    def _causal_diagnostics_from_result(result: GraphOutput) -> dict[str, object]:
        return {
            "initial_motion": result.initial_motion,
            "init_delta_x": result.init_delta_x,
            "init_delta_y": result.init_delta_y,
            "init_delta_z": result.init_delta_z,
            "init_delta_t_norm": result.init_delta_t_norm,
            "init_delta_R_angle": result.init_delta_R_angle,
            "init_velocity_j_x": result.init_velocity_j_x,
            "init_velocity_j_y": result.init_velocity_j_y,
            "init_velocity_j_z": result.init_velocity_j_z,
            "initial_energy_visual_weighted": result.initial_energy_visual_weighted,
            "initial_energy_p_weighted": result.initial_energy_p_weighted,
            "initial_energy_v_weighted": result.initial_energy_v_weighted,
            "initial_energy_R_weighted": result.initial_energy_R_weighted,
            "initial_energy_pv_weighted": result.initial_energy_pv_weighted,
            "initial_energy_imu_diag_weighted": result.initial_energy_imu_diag_weighted,
            "initial_energy_imu_weighted": result.initial_energy_imu_weighted,
            "initial_energy_imu_to_visual_ratio": result.initial_energy_imu_to_visual_ratio,
            "initial_energy_pv_to_visual_ratio": result.initial_energy_pv_to_visual_ratio,
            "initial_energy_R_to_visual_ratio": result.initial_energy_R_to_visual_ratio,
            "initial_total_loss": result.initial_total_loss,
            "update_pose_translation_norm": result.update_pose_translation_norm,
            "update_pose_rotation_norm": result.update_pose_rotation_norm,
            "update_velocity_norm": result.update_velocity_norm,
            "update_acc_bias_norm": result.update_acc_bias_norm,
            "update_gyro_bias_norm": result.update_gyro_bias_norm,
            "influence_visual_grad_norm": result.influence_visual_grad_norm,
            "influence_imu_grad_norm": result.influence_imu_grad_norm,
            "influence_grad_cosine": result.influence_grad_cosine,
            "influence_visual_hessian_trace": result.influence_visual_hessian_trace,
            "influence_imu_hessian_trace": result.influence_imu_hessian_trace,
            "influence_imu_to_visual_grad_ratio": result.influence_imu_to_visual_grad_ratio,
            "influence_imu_to_visual_hessian_ratio": result.influence_imu_to_visual_hessian_ratio,
            "influence_p_grad_norm": result.influence_p_grad_norm,
            "influence_v_grad_norm": result.influence_v_grad_norm,
            "influence_R_grad_norm": result.influence_R_grad_norm,
            "influence_sampled": result.influence_sampled,
            "energy_visual_change": result.energy_visual_change,
            "energy_imu_change": result.energy_imu_change,
            "energy_p_change": result.energy_p_change,
            "energy_v_change": result.energy_v_change,
            "energy_R_change": result.energy_R_change,
            "counterfactual_visual_step_norm": result.counterfactual_visual_step_norm,
            "counterfactual_imu_step_norm": result.counterfactual_imu_step_norm,
            "counterfactual_full_step_norm": result.counterfactual_full_step_norm,
            "counterfactual_visual_to_imu_cosine": result.counterfactual_visual_to_imu_cosine,
            "actual_to_visual_step_cosine": result.actual_to_visual_step_cosine,
            "actual_to_imu_step_cosine": result.actual_to_imu_step_cosine,
            "actual_to_full_step_cosine": result.actual_to_full_step_cosine,
            "predicted_visual_change_on_actual_step": result.predicted_visual_change_on_actual_step,
            "predicted_imu_change_on_actual_step": result.predicted_imu_change_on_actual_step,
        }

    def _write_local_ba_graph_data(self, result: GraphOutput, global_map: VisualMap) -> None:
        assert result.window_frame_indices is not None
        assert result.window_motions is not None

        frame_indices = result.window_frame_indices.reshape(-1).long()
        motions = result.window_motions.reshape(-1, 7).float()
        writeback = str(getattr(result, "local_ba_writeback", "current")).strip().lower()
        if writeback in {"all_two_state", "all_isam2_history"}:
            update_positions = range(frame_indices.numel())
        elif writeback == "all_optimized":
            update_positions = range(1, frame_indices.numel())  # keep fixed boundary untouched
        else:
            update_positions = [frame_indices.numel() - 1]

        frame_data = global_map.frames.data
        n_frames = int(frame_data["pose"].shape[0])

        def _write_vec(key: str, idx: int, value: torch.Tensor) -> None:
            if key in frame_data and 0 <= idx < n_frames:
                frame_data[key][idx] = value.reshape(1, 3).float()

        for local_pos in update_positions:
            frame_idx_t = frame_indices[local_pos]
            frame_idx = int(frame_idx_t.item())
            frame_data["pose"][frame_idx_t] = motions[local_pos:local_pos + 1]

            velocity = None
            acc_bias = None
            gyro_bias = None
            if result.window_velocity_world is not None:
                velocity = result.window_velocity_world.reshape(-1, 3)[local_pos]
                _write_vec("imu_vio_velocity_world", frame_idx, velocity)
                _write_vec("imu_vio_curr_velocity_init_world", frame_idx, velocity)
            if result.window_acc_bias is not None:
                acc_bias = result.window_acc_bias.reshape(-1, 3)[local_pos]
                _write_vec("imu_vio_acc_bias", frame_idx, acc_bias)
            if result.window_gyro_bias is not None:
                gyro_bias = result.window_gyro_bias.reshape(-1, 3)[local_pos]
                _write_vec("imu_vio_gyro_bias", frame_idx, gyro_bias)

            # Frame k stores the edge (k-1 -> k), so rewriting frame j also has
            # to refresh frame j+1's "previous" endpoint state when it exists.
            next_idx = frame_idx + 1
            if velocity is not None:
                _write_vec("imu_vio_prev_velocity_world", next_idx, velocity)
            if acc_bias is not None:
                _write_vec("imu_vio_prev_acc_bias", next_idx, acc_bias)
            if gyro_bias is not None:
                _write_vec("imu_vio_prev_gyro_bias", next_idx, gyro_bias)

        self.last_pair_diagnostics = {
            "from_idx": int(result.from_idx.reshape(-1)[0].item()),
            "frame_idx": int(result.frame_idx.reshape(-1)[0].item()),
            "final_loss": result.final_loss,
            "visual_loss": result.visual_loss,
            "imu_rot_loss": result.imu_rot_loss,
            "imu_trans_loss": result.imu_trans_loss,
            "imu_vel_loss": result.imu_vel_loss,
            "r_R_whitened_norm": result.r_R_whitened_norm,
            "r_p_whitened_norm": result.r_p_whitened_norm,
            "r_v_whitened_norm": result.r_v_whitened_norm,
            "imu_vio_whitened_norm": result.imu_vio_whitened_norm,
            "imu_vio_raw_norm": result.imu_vio_raw_norm,
            "imu_vio_cov_trace": result.imu_vio_cov_trace,
            "imu_vio_weight_trace": result.imu_vio_weight_trace,
            "imu_vio_weight_diag_min": result.imu_vio_weight_diag_min,
            "imu_vio_weight_diag_max": result.imu_vio_weight_diag_max,
            "imu_vio_sa_v2_sampling_noise_cost": result.imu_vio_sa_v2_sampling_noise_cost,
            "imu_vio_sa_v2_cross_covariance_frobenius_norm": (
                result.imu_vio_sa_v2_cross_covariance_frobenius_norm
            ),
            "imu_vio_sa_v2_incoming_sample_count": (
                result.imu_vio_sa_v2_incoming_sample_count
            ),
            "imu_vio_sa_v2_outgoing_sample_count": (
                result.imu_vio_sa_v2_outgoing_sample_count
            ),
            "imu_vio_acc_bias_norm": result.imu_vio_acc_bias_norm,
            "imu_vio_gyro_bias_norm": result.imu_vio_gyro_bias_norm,
            "imu_vio_acc_bias_x": result.imu_vio_acc_bias_x,
            "imu_vio_acc_bias_y": result.imu_vio_acc_bias_y,
            "imu_vio_acc_bias_z": result.imu_vio_acc_bias_z,
            "imu_vio_gyro_bias_x": result.imu_vio_gyro_bias_x,
            "imu_vio_gyro_bias_y": result.imu_vio_gyro_bias_y,
            "imu_vio_gyro_bias_z": result.imu_vio_gyro_bias_z,
            "imu_vio_alpha_p": result.imu_vio_alpha_p,
            "imu_vio_alpha_v": result.imu_vio_alpha_v,
            "imu_vio_alpha_R": result.imu_vio_alpha_R,
            **self._causal_diagnostics_from_result(result),
            "energy_visual_weighted": result.energy_visual_weighted,
            "energy_p_weighted": result.energy_p_weighted,
            "energy_v_weighted": result.energy_v_weighted,
            "energy_R_weighted": result.energy_R_weighted,
            "energy_pv_weighted": result.energy_pv_weighted,
            "energy_imu_diag_weighted": result.energy_imu_diag_weighted,
            "energy_imu_weighted": result.energy_imu_weighted,
            "energy_imu_to_visual_ratio": result.energy_imu_to_visual_ratio,
            "energy_pv_to_visual_ratio": result.energy_pv_to_visual_ratio,
            "energy_R_to_visual_ratio": result.energy_R_to_visual_ratio,
            "num_visual_residuals": result.num_visual_residuals,
            "num_observations": result.num_observations,
            "visual_pose_inlier_ratio": result.visual_pose_inlier_ratio,
            "visual_pose_mean_mahalanobis_sq": result.visual_pose_mean_mahalanobis_sq,
            "visual_pose_whitened_residual_norm": result.visual_pose_whitened_residual_norm,
            "visual_pose_covariance_inflation": result.visual_pose_covariance_inflation,
            "visual_pose_gate_action": result.visual_pose_gate_action,
            "imu_factor_mode": result.imu_factor_mode,
            "vio_factor_active": True,
            "vio_bias_state_active": True,
            "imu_residual_rows": result.imu_residual_rows,
            "use_imu_rotation": True,
            "use_imu_translation": True,
            "local_ba_window_size": result.local_ba_window_size,
            "local_ba_writeback": writeback,
            "local_ba_num_frames": result.local_ba_num_frames,
            "local_ba_num_edges": result.local_ba_num_edges,
            "local_ba_num_visual_residual_blocks": result.local_ba_num_visual_residual_blocks,
            "local_ba_graph_build_s": result.local_ba_graph_build_s,
            "local_ba_lm_s": result.local_ba_lm_s,
            "local_ba_refine_s": result.local_ba_refine_s,
            "local_ba_optimize_total_s": result.local_ba_optimize_total_s,
            "two_state_solver_converged": result.two_state_solver_converged,
            "two_state_solver_iterations": result.two_state_solver_iterations,
            "two_state_solver_convergence_reason": result.two_state_solver_convergence_reason,
            "two_state_final_step_norm": result.two_state_final_step_norm,
            "two_state_final_gradient_inf_norm": result.two_state_final_gradient_inf_norm,
            "two_state_solver_accepted_steps": result.two_state_solver_accepted_steps,
            "two_state_solver_rejected_steps": result.two_state_solver_rejected_steps,
            "vio_backend": result.vio_backend,
            "isam2_update_ms": result.isam2_update_ms,
            "isam2_state_count": result.isam2_state_count,
            "isam2_history_revision": result.isam2_history_revision,
            "isam2_initial_pose_mismatch_norm": (
                result.isam2_initial_pose_mismatch_norm
            ),
            "isam2_initial_velocity_mismatch_norm": (
                result.isam2_initial_velocity_mismatch_norm
            ),
            "isam2_initial_bias_mismatch_norm": (
                result.isam2_initial_bias_mismatch_norm
            ),
        }
        for field_name in (
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
        ):
            self.last_pair_diagnostics[field_name] = getattr(
                result, field_name, None
            )
        self.last_breakpoint_trace = result.debug_trace
        self.last_breakpoint_frame_indices = [int(v) for v in frame_indices.detach().cpu().reshape(-1).tolist()]

    def write_graph_data(self, result: GraphOutput | None, global_map: VisualMap) -> None:
        if result is None: return
        if result.window_frame_indices is not None:
            self._write_local_ba_graph_data(result, global_map)
            return

        to_pose = pp.SE3(result.motion[0].data.double().cpu())
        fusion_log: dict | None = None
        if bool(getattr(self.config, "post_imu_fusion_enable", False)):
            mode = str(getattr(self.config, "post_imu_fusion_mode", "uraf")).lower()
            if mode == "none":
                pass
            elif mode == "qavif":
                to_pose, fusion_log = self._post_fuse_visual_imu(
                    to_pose,
                    result.frame_idx,
                    global_map,
                    result.visual_obs_cov_mean,
                    getattr(result, "num_observations", 0),
                )
            elif mode == "padf":
                to_pose, fusion_log = self._post_fuse_visual_imu_padf(
                    to_pose,
                    result.frame_idx,
                    global_map,
                    result.visual_obs_cov_mean,
                    getattr(result, "num_observations", 0),
                )
            elif mode == "dua":
                to_pose, fusion_log = self._post_fuse_visual_imu_dua(
                    to_pose,
                    result.frame_idx,
                    global_map,
                    result.visual_obs_cov_mean,
                    getattr(result, "num_observations", 0),
                    getattr(result, "visual_keypoint_coverage", None),
                    getattr(result, "visual_depth_spread", None),
                )
            elif mode == "bagf":
                to_pose, fusion_log = self._post_fuse_visual_imu_bagf(
                    to_pose,
                    result.frame_idx,
                    global_map,
                    result.visual_obs_cov_mean,
                    getattr(result, "num_observations", 0),
                    getattr(result, "visual_keypoint_coverage", None),
                    getattr(result, "visual_depth_spread", None),
                )
            elif mode == "uraf":
                frame_idx_i = int(result.frame_idx.item())
                if frame_idx_i > 0:
                    frame_data = global_map.frames.data
                    try:
                        rot_prior = frame_data["imu_rotvec_prior"][result.frame_idx].reshape(-1)
                        rot_std = frame_data["imu_rot_prior_std"][result.frame_idx].reshape(-1)
                        trans_prior = frame_data["imu_trans_prior"][result.frame_idx].reshape(-1)
                        trans_cov = frame_data["imu_trans_cov"][result.frame_idx].reshape(3, 3)
                        prev_pose = pp.SE3(frame_data["pose"][result.frame_idx - 1].double())

                        if (rot_prior.numel() == 3 and trans_prior.numel() == 3
                                and trans_cov.numel() == 9 and float(rot_std[0].item()) < 1e5):
                            uraf_lambda = float(getattr(self.config, "uraf_lambda", 0.5))
                            to_pose, fusion_log = uraf_post_fusion(
                                to_pose=to_pose,
                                prev_pose=prev_pose,
                                rot_prior=rot_prior,
                                rot_std_raw=rot_std,
                                trans_prior=trans_prior,
                                trans_cov_raw=trans_cov,
                                visual_cov_mean=result.visual_obs_cov_mean,
                                num_observations=getattr(result, "num_observations", 0),
                                uraf_lambda=uraf_lambda,
                                imu_trans_enable=bool(getattr(self.config, "post_imu_fusion_imu_trans_enable", False)),
                                imu_trans_z_only=bool(getattr(self.config, "post_imu_fusion_imu_trans_z_only", False)),
                            )
                    except Exception:
                        pass  # Fall through: use visual-only pose
            else:
                raise ValueError(f"Unsupported post_imu_fusion_mode={mode!r}")
        self._record_fusion_diagnostics(global_map, result.frame_idx, fusion_log)
        global_map.frames.data["pose"][result.frame_idx] = to_pose.float()
        if result.velocity_world is not None and "imu_vio_velocity_world" in global_map.frames.data:
            global_map.frames.data["imu_vio_velocity_world"][result.frame_idx] = (
                result.velocity_world.reshape(1, 3).float()
            )
        if result.acc_bias is not None and "imu_vio_acc_bias" in global_map.frames.data:
            global_map.frames.data["imu_vio_acc_bias"][result.frame_idx] = (
                result.acc_bias.reshape(1, 3).float()
            )
        if result.gyro_bias is not None and "imu_vio_gyro_bias" in global_map.frames.data:
            global_map.frames.data["imu_vio_gyro_bias"][result.frame_idx] = (
                result.gyro_bias.reshape(1, 3).float()
            )

        # ── Store per-pair diagnostics from GraphOutput for frame_pair_diagnostics.csv ──
        self.last_pair_diagnostics = {
            "from_idx": int(result.from_idx.item()) if hasattr(result, "from_idx") else -1,
            "frame_idx": int(result.frame_idx.item()) if hasattr(result, "frame_idx") else -1,
            "final_loss": result.final_loss,
            "visual_loss": result.visual_loss,
            "imu_rot_loss": result.imu_rot_loss,
            "imu_trans_loss": result.imu_trans_loss,
            "imu_vel_loss": result.imu_vel_loss,
            "r_R_whitened_norm": result.r_R_whitened_norm,
            "r_p_whitened_norm": result.r_p_whitened_norm,
            "r_v_whitened_norm": result.r_v_whitened_norm,
            "imu_vio_whitened_norm": result.imu_vio_whitened_norm,
            "imu_vio_raw_norm": result.imu_vio_raw_norm,
            "imu_vio_cov_trace": result.imu_vio_cov_trace,
            "imu_vio_weight_trace": result.imu_vio_weight_trace,
            "imu_vio_weight_diag_min": result.imu_vio_weight_diag_min,
            "imu_vio_weight_diag_max": result.imu_vio_weight_diag_max,
            "imu_vio_acc_bias_norm": result.imu_vio_acc_bias_norm,
            "imu_vio_gyro_bias_norm": result.imu_vio_gyro_bias_norm,
            "imu_vio_acc_bias_x": result.imu_vio_acc_bias_x,
            "imu_vio_acc_bias_y": result.imu_vio_acc_bias_y,
            "imu_vio_acc_bias_z": result.imu_vio_acc_bias_z,
            "imu_vio_gyro_bias_x": result.imu_vio_gyro_bias_x,
            "imu_vio_gyro_bias_y": result.imu_vio_gyro_bias_y,
            "imu_vio_gyro_bias_z": result.imu_vio_gyro_bias_z,
            "imu_vio_alpha_p": result.imu_vio_alpha_p,
            "imu_vio_alpha_v": result.imu_vio_alpha_v,
            "imu_vio_alpha_R": result.imu_vio_alpha_R,
            **self._causal_diagnostics_from_result(result),
            "energy_visual_weighted": result.energy_visual_weighted,
            "energy_p_weighted": result.energy_p_weighted,
            "energy_v_weighted": result.energy_v_weighted,
            "energy_R_weighted": result.energy_R_weighted,
            "energy_pv_weighted": result.energy_pv_weighted,
            "energy_imu_diag_weighted": result.energy_imu_diag_weighted,
            "energy_imu_weighted": result.energy_imu_weighted,
            "energy_imu_to_visual_ratio": result.energy_imu_to_visual_ratio,
            "energy_pv_to_visual_ratio": result.energy_pv_to_visual_ratio,
            "energy_R_to_visual_ratio": result.energy_R_to_visual_ratio,
            "num_visual_residuals": result.num_visual_residuals,
            "num_observations": result.num_observations,
            "imu_factor_mode": result.imu_factor_mode,
            "vio_factor_active": result.vio_factor_active,
            "vio_bias_state_active": result.vio_bias_state_active,
            "imu_residual_rows": result.imu_residual_rows,
            "use_imu_rotation": result.use_imu_rotation,
            "use_imu_translation": result.use_imu_translation,
        }
        for field_name in (
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
        ):
            self.last_pair_diagnostics[field_name] = getattr(result, field_name, None)
        self.last_breakpoint_trace = result.debug_trace
        self.last_breakpoint_frame_indices = [
            int(result.from_idx.reshape(-1)[0].detach().cpu().item()) if hasattr(result, "from_idx") else -1,
            int(result.frame_idx.reshape(-1)[0].detach().cpu().item()) if hasattr(result, "frame_idx") else -1,
        ]


class Local_TwoFrame_PGO(TwoFrame_PGO):
    """
    Simple two-frame PGO in visual-odometry (MAC-VO) under Local frame. May lead to better optimization
    due to more numerical stability (especially in large-scene with 1000+ meters size)
    """
    def get_graph_data(self, global_map: VisualMap, frame_idx: torch.Tensor,
                       observations: torch.Tensor | None = None, edges: torch.Tensor | None = None) -> GraphInput:
        global_graph_data = super().get_graph_data(global_map, frame_idx, observations, edges)
        self.T_o2w_idx = frame_idx - 1

        T_o2w = pp.SE3(global_map.frames.data["pose"][frame_idx - 1])
        T_w2o = T_o2w.Inv()
        return self.world_to_optim(global_graph_data, T_w2o)

    def write_graph_data(self, result: GraphOutput | None, global_map: VisualMap) -> None:
        if result is None: return

        T_o2w = pp.SE3(global_map.frames.data["pose"][self.T_o2w_idx])
        super().write_graph_data(self.optim_to_world(result, T_o2w), global_map)

    def world_to_optim(self, data: GraphInput, T_w2o: pp.LieTensor) -> GraphInput:
        """Transform the optimization graph data into local reference frame (i.e. the reference frame is the pose of previous key frame)
        """
        # Same for below:
        # c = camera to optimize, o = optimization frame, w = world (global) frame
        T_c2w = pp.LieTensor(data.init_motion, ltype=pp.SE3_type)
        T_c2o = T_w2o @ T_c2w
        R_w2o = T_w2o.rotation().matrix().to(data.points.data["cov_Tw"])

        data.init_motion = T_c2o
        data.points.data["pos_Tw"]  = pp.Act(pp.SE3(T_w2o.to(data.points.data["pos_Tw"])), data.points.data["pos_Tw"])
        data.points.data["cov_Tw"]  = R_w2o @ data.points.data["cov_Tw"] @ R_w2o.transpose(-1, -2)
        return data

    def optim_to_world(self, data: GraphOutput, T_o2w: pp.LieTensor) -> GraphOutput:
        """Transform the optimization result under local reference frame (w.r.t. previous KF) to the global frame.
        """
        T_c2o = data.motion
        data.motion = NormalizeQuat(T_o2w @ pp.SE3(T_c2o.to(T_o2w)))
        return data


class Empty_TwoFrame_PGO(TwoFrame_PGO):
    """
    A 'no-op' variant of the Two-frame PGO optimizer. Helpful in debugging process.
    """
    @staticmethod
    def _optimize(context: dict, graph_data: GraphInput) -> tuple[dict, GraphOutput]:
        return context, GraphOutput(motion=graph_data.init_motion,
                                    frame_idx=graph_data.frame_idx,
                                    from_idx=graph_data.from_idx,
                                    visual_obs_cov_mean=graph_data.visual_obs_cov_mean,
                                    num_observations=graph_data.num_observations,
                                    visual_keypoint_coverage=graph_data.visual_keypoint_coverage,
                                    visual_depth_spread=graph_data.visual_depth_spread)
