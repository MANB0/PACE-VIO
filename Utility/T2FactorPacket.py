from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import pypose as pp
import torch

from Utility.TwoStateVIO import (
    ImuPreintegrationFactor,
    LinearizedUVDPoseFactor,
    NavigationState,
    RelativePoseFactor,
    SquareRootPrior,
    TwoStateVIOProblem,
    UVDFactor,
    VisualFactor,
)


def _normalized_pose(value: torch.Tensor) -> torch.Tensor:
    pose = value.detach().reshape(7).to(device="cpu", dtype=torch.float64).clone()
    norm = pose[3:7].norm()
    if not bool(torch.isfinite(norm)) or float(norm.item()) < 1.0e-15:
        raise ValueError("SE(3) quaternion is zero or nonfinite")
    pose[3:7] = pose[3:7] / norm
    return pose.reshape(1, 7)


def _cpu_state(state: NavigationState) -> NavigationState:
    value = state.to(device=torch.device("cpu"), dtype=torch.float64).detach()
    return NavigationState(
        pose_WB=_normalized_pose(value.pose_WB),
        velocity_W=value.velocity_W.clone(),
        acc_bias=value.acc_bias.clone(),
        gyro_bias=value.gyro_bias.clone(),
    )


def _cpu_imu(factor: ImuPreintegrationFactor) -> ImuPreintegrationFactor:
    value = factor.to(device=torch.device("cpu"), dtype=torch.float64)
    return ImuPreintegrationFactor(
        delta_rotation=value.delta_rotation.detach().clone(),
        delta_velocity=value.delta_velocity.detach().clone(),
        delta_position=value.delta_position.detach().clone(),
        covariance=value.covariance.detach().clone(),
        dt=float(value.dt),
        bias_jacobian=value.bias_jacobian.detach().clone(),
        linearized_acc_bias=value.linearized_acc_bias.detach().clone(),
        linearized_gyro_bias=value.linearized_gyro_bias.detach().clone(),
        bias_rw_covariance=value.bias_rw_covariance.detach().clone(),
        gravity_world=(
            None if value.gravity_world is None else value.gravity_world.detach().clone()
        ),
        gravity_handling=str(value.gravity_handling),
    )


def _cpu_visual(factor: VisualFactor) -> VisualFactor:
    value = factor.to(device=torch.device("cpu"), dtype=torch.float64)
    if isinstance(value, RelativePoseFactor):
        return RelativePoseFactor(
            measurement_BiBj=_normalized_pose(value.measurement_BiBj),
            covariance=value.covariance.detach().clone(),
            huber_delta=float(value.huber_delta),
        )
    if isinstance(value, UVDFactor):
        return UVDFactor(
            points_Ci=value.points_Ci.detach().clone(),
            target_uv=value.target_uv.detach().clone(),
            target_disparity=value.target_disparity.detach().clone(),
            covariance_uvd=value.covariance_uvd.detach().clone(),
            intrinsic=value.intrinsic.detach().clone(),
            baseline=float(value.baseline),
            extrinsic_CI=_normalized_pose(value.extrinsic_CI),
            huber_delta=float(value.huber_delta),
            optimization_mode=str(value.optimization_mode),
            anchor_relative_CjCi=(
                None
                if value.anchor_relative_CjCi is None
                else _normalized_pose(value.anchor_relative_CjCi)
            ),
        )
    if isinstance(value, LinearizedUVDPoseFactor):
        return LinearizedUVDPoseFactor(
            reference_relative_CjCi=_normalized_pose(
                value.reference_relative_CjCi
            ),
            sqrt_information=value.sqrt_information.detach().clone(),
            residual_offset=value.residual_offset.detach().clone(),
            extrinsic_CI=_normalized_pose(value.extrinsic_CI),
            marginal_mode=str(value.marginal_mode),
            huber_delta=float(value.huber_delta),
        )
    raise TypeError(f"unsupported visual factor packet type: {type(value).__name__}")


def _all_finite(name: str, value: torch.Tensor) -> None:
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} contains NaN/Inf")


def _validate_covariance(name: str, value: torch.Tensor, size: int) -> None:
    covariance = value.reshape(size, size)
    _all_finite(name, covariance)
    asymmetry = float((covariance - covariance.T).abs().max().item())
    scale = max(float(covariance.abs().max().item()), 1.0)
    if asymmetry > 1.0e-10 * scale:
        raise ValueError(f"{name} is not symmetric: max asymmetry={asymmetry:.3e}")
    minimum = float(torch.linalg.eigvalsh(0.5 * (covariance + covariance.T)).min().item())
    if minimum < -1.0e-10 * scale:
        raise ValueError(f"{name} is not positive semidefinite: min eig={minimum:.3e}")


@dataclass(frozen=True)
class PACEFactorPacket:
    """Backend-neutral measurement packet for one PACE-VIO edge.

    Contract:
      * state pose: ``T_WB`` (body/IMU to world);
      * state tangent: ``[p, phi, v, ba, bg]``;
      * IMU residual/covariance: ``[p, v, R]`` in the frame-i body tangent;
      * visual factor: Pose, point-level UVD, or compressed PACE;
      * extrinsic: ``T_CI`` mapping raw IMU frame I to MACVO camera frame C,
        with ``p_C = T_CI p_I``.

    The packet is materialized as detached CPU float64 data. Both the existing
    two-state solver and the incremental iSAM2 backend must consume this exact
    object. The incremental backend preserves each visual representation's
    native residual rather than silently converting Pose or UVD into PACE.
    """

    frame_i: int
    frame_j: int
    state_i_initial: NavigationState
    state_j_initial: NavigationState
    imu: ImuPreintegrationFactor
    visual: VisualFactor
    extrinsic_CI: torch.Tensor

    @classmethod
    def create(
        cls,
        *,
        frame_i: int,
        frame_j: int,
        state_i_initial: NavigationState,
        state_j_initial: NavigationState,
        imu: ImuPreintegrationFactor,
        visual: VisualFactor,
        extrinsic_CI: torch.Tensor,
    ) -> "PACEFactorPacket":
        packet = cls(
            frame_i=int(frame_i),
            frame_j=int(frame_j),
            state_i_initial=_cpu_state(state_i_initial),
            state_j_initial=_cpu_state(state_j_initial),
            imu=_cpu_imu(imu),
            visual=_cpu_visual(visual),
            extrinsic_CI=_normalized_pose(extrinsic_CI),
        )
        packet.validate()
        return packet

    def validate(self) -> None:
        if self.frame_i < 0 or self.frame_j <= self.frame_i:
            raise ValueError(
                f"invalid PACE-VIO edge order: frame_i={self.frame_i}, frame_j={self.frame_j}"
            )
        if not self.imu.dt > 0.0:
            raise ValueError(f"IMU dt must be positive, got {self.imu.dt}")

        tensors = {
            "state_i.pose_WB": self.state_i_initial.pose_WB.reshape(7),
            "state_i.velocity_W": self.state_i_initial.velocity_W.reshape(3),
            "state_i.acc_bias": self.state_i_initial.acc_bias.reshape(3),
            "state_i.gyro_bias": self.state_i_initial.gyro_bias.reshape(3),
            "state_j.pose_WB": self.state_j_initial.pose_WB.reshape(7),
            "state_j.velocity_W": self.state_j_initial.velocity_W.reshape(3),
            "state_j.acc_bias": self.state_j_initial.acc_bias.reshape(3),
            "state_j.gyro_bias": self.state_j_initial.gyro_bias.reshape(3),
            "imu.delta_rotation": self.imu.delta_rotation.reshape(-1),
            "imu.delta_velocity": self.imu.delta_velocity.reshape(3),
            "imu.delta_position": self.imu.delta_position.reshape(3),
            "imu.bias_jacobian": self.imu.bias_jacobian.reshape(9, 6),
            "imu.linearized_acc_bias": self.imu.linearized_acc_bias.reshape(3),
            "imu.linearized_gyro_bias": self.imu.linearized_gyro_bias.reshape(3),
            "extrinsic_CI": self.extrinsic_CI.reshape(7),
        }
        if isinstance(self.visual, RelativePoseFactor):
            tensors.update(
                {
                    "visual.measurement_BiBj": self.visual.measurement_BiBj.reshape(7),
                    "visual.covariance": self.visual.covariance.reshape(6, 6),
                }
            )
        elif isinstance(self.visual, UVDFactor):
            count = int(self.visual.points_Ci.reshape(-1, 3).shape[0])
            if count < 3:
                raise ValueError("direct UVD packet requires at least three points")
            if str(self.visual.optimization_mode).strip().lower() != "full":
                raise ValueError(
                    "incremental iSAM2 supports only the full direct UVD objective"
                )
            expected = {
                "target_uv": int(self.visual.target_uv.reshape(-1, 2).shape[0]),
                "target_disparity": int(
                    self.visual.target_disparity.reshape(-1, 1).shape[0]
                ),
                "covariance_uvd": int(
                    self.visual.covariance_uvd.reshape(-1, 3, 3).shape[0]
                ),
            }
            mismatch = {name: rows for name, rows in expected.items() if rows != count}
            if mismatch:
                raise ValueError(
                    f"direct UVD packet row mismatch: points={count}, fields={mismatch}"
                )
            tensors.update(
                {
                    "visual.points_Ci": self.visual.points_Ci.reshape(count, 3),
                    "visual.target_uv": self.visual.target_uv.reshape(count, 2),
                    "visual.target_disparity": self.visual.target_disparity.reshape(count, 1),
                    "visual.covariance_uvd": self.visual.covariance_uvd.reshape(count, 3, 3),
                    "visual.intrinsic": self.visual.intrinsic.reshape(3, 3),
                    "visual.extrinsic_CI": self.visual.extrinsic_CI.reshape(7),
                }
            )
            if not math.isfinite(float(self.visual.baseline)) or self.visual.baseline <= 0.0:
                raise ValueError("direct UVD packet baseline must be finite and positive")
        elif isinstance(self.visual, LinearizedUVDPoseFactor):
            tensors.update(
                {
                    "visual.reference_relative_CjCi": self.visual.reference_relative_CjCi.reshape(7),
                    "visual.sqrt_information": self.visual.sqrt_information.reshape(-1, 6),
                    "visual.residual_offset": self.visual.residual_offset.reshape(-1),
                    "visual.extrinsic_CI": self.visual.extrinsic_CI.reshape(7),
                }
            )
        else:
            raise TypeError(
                f"unsupported visual factor packet type: {type(self.visual).__name__}"
            )
        if self.imu.gravity_world is not None:
            tensors["imu.gravity_world"] = self.imu.gravity_world.reshape(3)
        for name, value in tensors.items():
            _all_finite(name, value)

        delta_rotation_size = int(self.imu.delta_rotation.numel())
        if delta_rotation_size not in {3, 4}:
            raise ValueError(
                "imu.delta_rotation must be a rotvec (3) or SO(3) quaternion (4), "
                f"got {delta_rotation_size} values"
            )
        if isinstance(self.visual, LinearizedUVDPoseFactor):
            rows = int(self.visual.sqrt_information.reshape(-1, 6).shape[0])
            if rows < 1 or rows > 6:
                raise ValueError(f"compressed visual rank must be in [1, 6], got {rows}")
            if int(self.visual.residual_offset.numel()) != rows:
                raise ValueError("visual sqrt-information rows and residual offset disagree")

        _validate_covariance("imu.covariance", self.imu.covariance, 9)
        _validate_covariance("imu.bias_rw_covariance", self.imu.bias_rw_covariance, 6)
        if isinstance(self.visual, RelativePoseFactor):
            _validate_covariance("visual.covariance", self.visual.covariance, 6)
        elif isinstance(self.visual, UVDFactor):
            for index, covariance in enumerate(
                self.visual.covariance_uvd.reshape(-1, 3, 3)
            ):
                _validate_covariance(f"visual.covariance_uvd[{index}]", covariance, 3)

        for name, pose in (
            ("state_i.pose_WB", self.state_i_initial.pose_WB),
            ("state_j.pose_WB", self.state_j_initial.pose_WB),
            ("extrinsic_CI", self.extrinsic_CI),
        ):
            quaternion_norm = float(pose.reshape(7)[3:7].norm().item())
            if abs(quaternion_norm - 1.0) > 1.0e-10:
                raise ValueError(f"{name} quaternion is not normalized: {quaternion_norm}")

        if isinstance(self.visual, RelativePoseFactor):
            visual_poses = (
                ("visual.measurement_BiBj", self.visual.measurement_BiBj),
            )
        elif isinstance(self.visual, UVDFactor):
            visual_poses = (("visual.extrinsic_CI", self.visual.extrinsic_CI),)
        else:
            visual_poses = (
                ("visual.reference_relative_CjCi", self.visual.reference_relative_CjCi),
                ("visual.extrinsic_CI", self.visual.extrinsic_CI),
            )
        for name, pose in visual_poses:
            quaternion_norm = float(pose.reshape(7)[3:7].norm().item())
            if abs(quaternion_norm - 1.0) > 1.0e-10:
                raise ValueError(f"{name} quaternion is not normalized: {quaternion_norm}")

        if not isinstance(self.visual, RelativePoseFactor):
            extrinsic_error = (
                pp.SE3(self.visual.extrinsic_CI).Inv() @ pp.SE3(self.extrinsic_CI)
            ).Log().tensor().reshape(6)
            if float(extrinsic_error.abs().max().item()) > 1.0e-10:
                raise ValueError("visual factor and packet use different T_CI extrinsics")

    def to_two_state_problem(
        self,
        *,
        prior_i: SquareRootPrior,
        device: torch.device,
        optimize_acc_bias: bool,
        optimize_gyro_bias: bool,
    ) -> TwoStateVIOProblem:
        dtype = torch.float64
        return TwoStateVIOProblem(
            state_i=self.state_i_initial.to(device=device, dtype=dtype),
            state_j=self.state_j_initial.to(device=device, dtype=dtype),
            prior_i=prior_i.to(device=device, dtype=dtype),
            imu=self.imu.to(device=device, dtype=dtype),
            visual_pose=self.visual.to(device=device, dtype=dtype),
            optimize_acc_bias=bool(optimize_acc_bias),
            optimize_gyro_bias=bool(optimize_gyro_bias),
        )

    def incremental_payload(self) -> dict[str, Any]:
        """Return detached arrays for the in-process C++ incremental backend."""

        delta_rotation = self.imu.delta_rotation.reshape(-1)
        if int(delta_rotation.numel()) == 4:
            delta_rotation = pp.SO3(delta_rotation.reshape(1, 4)).Log().tensor().reshape(3)
        payload = {
            "frame_i": self.frame_i,
            "frame_j": self.frame_j,
            "state_i_pose_WB": self.state_i_initial.pose_WB.reshape(7).numpy(),
            "state_i_velocity_W": self.state_i_initial.velocity_W.reshape(3).numpy(),
            "state_i_acc_bias": self.state_i_initial.acc_bias.reshape(3).numpy(),
            "state_i_gyro_bias": self.state_i_initial.gyro_bias.reshape(3).numpy(),
            "state_j_pose_WB": self.state_j_initial.pose_WB.reshape(7).numpy(),
            "state_j_velocity_W": self.state_j_initial.velocity_W.reshape(3).numpy(),
            "state_j_acc_bias": self.state_j_initial.acc_bias.reshape(3).numpy(),
            "state_j_gyro_bias": self.state_j_initial.gyro_bias.reshape(3).numpy(),
            "imu_delta_rotvec": delta_rotation.numpy(),
            "imu_delta_velocity": self.imu.delta_velocity.reshape(3).numpy(),
            "imu_delta_position": self.imu.delta_position.reshape(3).numpy(),
            "imu_covariance_pvr": self.imu.covariance.reshape(9, 9).numpy(),
            "imu_dt": float(self.imu.dt),
            "imu_bias_jacobian_pvr_babg": self.imu.bias_jacobian.reshape(9, 6).numpy(),
            "imu_linearized_acc_bias": self.imu.linearized_acc_bias.reshape(3).numpy(),
            "imu_linearized_gyro_bias": self.imu.linearized_gyro_bias.reshape(3).numpy(),
            "bias_rw_covariance_babg": self.imu.bias_rw_covariance.reshape(6, 6).numpy(),
            "gravity_world": (
                None
                if self.imu.gravity_world is None
                else self.imu.gravity_world.reshape(3).numpy()
            ),
            "gravity_handling": str(self.imu.gravity_handling),
            "extrinsic_CI": self.extrinsic_CI.reshape(7).numpy(),
        }
        if isinstance(self.visual, RelativePoseFactor):
            payload.update(
                {
                    "visual_factor_mode": "relative_pose",
                    "visual_measurement_BiBj": self.visual.measurement_BiBj.reshape(7).numpy(),
                    "visual_covariance_tr": self.visual.covariance.reshape(6, 6).numpy(),
                    "visual_huber_delta": float(self.visual.huber_delta),
                }
            )
        elif isinstance(self.visual, UVDFactor):
            count = int(self.visual.points_Ci.reshape(-1, 3).shape[0])
            target_uvd = torch.cat(
                [
                    self.visual.target_uv.reshape(count, 2),
                    self.visual.target_disparity.reshape(count, 1),
                ],
                dim=-1,
            )
            payload.update(
                {
                    "visual_factor_mode": "direct_uvd",
                    "visual_points_Ci": self.visual.points_Ci.reshape(count, 3).numpy(),
                    "visual_target_uvd": target_uvd.numpy(),
                    "visual_covariance_uvd_flat": self.visual.covariance_uvd.reshape(count, 9).numpy(),
                    "visual_intrinsic": self.visual.intrinsic.reshape(3, 3).numpy(),
                    "visual_baseline": float(self.visual.baseline),
                    "visual_huber_delta": float(self.visual.huber_delta),
                }
            )
        elif isinstance(self.visual, LinearizedUVDPoseFactor):
            payload.update(
                {
                    "visual_factor_mode": "compressed_uvd",
                    "visual_reference_CjCi": self.visual.reference_relative_CjCi.reshape(7).numpy(),
                    "visual_sqrt_information": self.visual.sqrt_information.reshape(-1, 6).numpy(),
                    "visual_residual_offset": self.visual.residual_offset.reshape(-1).numpy(),
                }
            )
        else:
            raise TypeError(
                f"unsupported visual factor packet type: {type(self.visual).__name__}"
            )
        return payload


# Backward-compatible API for archived configs, scripts, and result replays.
T2FactorPacket = PACEFactorPacket
