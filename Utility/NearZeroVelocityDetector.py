from __future__ import annotations

from dataclasses import dataclass
import math

import pypose as pp
import torch


@dataclass(frozen=True)
class NearZeroVelocityDecision:
    active: bool
    candidate: bool
    entered: bool
    exited: bool
    estimated_speed_m_s: float
    imu_angular_rate_rad_s: float
    visual_angular_rate_rad_s: float
    visual_body_speed_m_s: float
    angular_rate_disagreement_rad_s: float
    candidate_duration_s: float
    release_duration_s: float
    reason: str


class TurningNearZeroVelocityDetector:
    """Causal detector for stop-turn motion without ground truth.

    The detector deliberately allows non-zero angular velocity. It declares a
    candidate only when the current navigation speed is low and both the IMU
    and visual relative rotation independently indicate a turn. Hysteresis
    prevents a single noisy edge from inserting a velocity factor.
    """

    def __init__(
        self,
        *,
        maximum_speed_m_s: float = 0.18,
        minimum_imu_angular_rate_rad_s: float = 0.15,
        minimum_visual_angular_rate_rad_s: float = 0.12,
        maximum_angular_rate_disagreement_rad_s: float = 0.35,
        enter_hold_s: float = 0.20,
        release_hold_s: float = 0.10,
    ) -> None:
        values = {
            "maximum_speed_m_s": maximum_speed_m_s,
            "minimum_imu_angular_rate_rad_s": minimum_imu_angular_rate_rad_s,
            "minimum_visual_angular_rate_rad_s": minimum_visual_angular_rate_rad_s,
            "maximum_angular_rate_disagreement_rad_s": (
                maximum_angular_rate_disagreement_rad_s
            ),
            "enter_hold_s": enter_hold_s,
            "release_hold_s": release_hold_s,
        }
        if not all(math.isfinite(float(value)) for value in values.values()):
            raise ValueError("near-zero-velocity detector thresholds must be finite")
        if maximum_speed_m_s <= 0.0:
            raise ValueError("maximum_speed_m_s must be positive")
        if minimum_imu_angular_rate_rad_s < 0.0:
            raise ValueError("minimum_imu_angular_rate_rad_s must be nonnegative")
        if minimum_visual_angular_rate_rad_s < 0.0:
            raise ValueError("minimum_visual_angular_rate_rad_s must be nonnegative")
        if maximum_angular_rate_disagreement_rad_s < 0.0:
            raise ValueError(
                "maximum_angular_rate_disagreement_rad_s must be nonnegative"
            )
        if enter_hold_s <= 0.0 or release_hold_s <= 0.0:
            raise ValueError("detector hold durations must be positive")

        self.maximum_speed_m_s = float(maximum_speed_m_s)
        self.minimum_imu_angular_rate_rad_s = float(
            minimum_imu_angular_rate_rad_s
        )
        self.minimum_visual_angular_rate_rad_s = float(
            minimum_visual_angular_rate_rad_s
        )
        self.maximum_angular_rate_disagreement_rad_s = float(
            maximum_angular_rate_disagreement_rad_s
        )
        self.enter_hold_s = float(enter_hold_s)
        self.release_hold_s = float(release_hold_s)
        self.reset()

    def reset(self) -> None:
        self._active = False
        self._candidate_duration_s = 0.0
        self._release_duration_s = 0.0

    @property
    def active(self) -> bool:
        return bool(self._active)

    def update(
        self,
        *,
        dt_s: float,
        estimated_speed_m_s: float,
        imu_angular_rate_rad_s: float,
        visual_angular_rate_rad_s: float,
        visual_body_speed_m_s: float,
    ) -> NearZeroVelocityDecision:
        values = (
            dt_s,
            estimated_speed_m_s,
            imu_angular_rate_rad_s,
            visual_angular_rate_rad_s,
            visual_body_speed_m_s,
        )
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("near-zero-velocity detector input contains NaN/Inf")
        if dt_s <= 0.0:
            raise ValueError("near-zero-velocity detector dt must be positive")

        disagreement = abs(
            float(imu_angular_rate_rad_s)
            - float(visual_angular_rate_rad_s)
        )
        low_speed = float(estimated_speed_m_s) <= self.maximum_speed_m_s
        imu_turning = (
            float(imu_angular_rate_rad_s)
            >= self.minimum_imu_angular_rate_rad_s
        )
        visual_turning = (
            float(visual_angular_rate_rad_s)
            >= self.minimum_visual_angular_rate_rad_s
        )
        rotation_agrees = (
            disagreement <= self.maximum_angular_rate_disagreement_rad_s
        )
        candidate = low_speed and imu_turning and visual_turning and rotation_agrees

        entered = False
        exited = False
        if candidate:
            self._candidate_duration_s += float(dt_s)
            self._release_duration_s = 0.0
            if (
                not self._active
                and self._candidate_duration_s + 1.0e-12 >= self.enter_hold_s
            ):
                self._active = True
                entered = True
        else:
            self._candidate_duration_s = 0.0
            if self._active:
                self._release_duration_s += float(dt_s)
                if self._release_duration_s + 1.0e-12 >= self.release_hold_s:
                    self._active = False
                    self._release_duration_s = 0.0
                    exited = True
            else:
                self._release_duration_s = 0.0

        failed = []
        if not low_speed:
            failed.append("speed")
        if not imu_turning:
            failed.append("imu_rotation")
        if not visual_turning:
            failed.append("visual_rotation")
        if not rotation_agrees:
            failed.append("rotation_disagreement")
        if self._active:
            reason = "active"
        elif candidate:
            reason = "candidate_hold"
        else:
            reason = "rejected:" + ",".join(failed)

        return NearZeroVelocityDecision(
            active=bool(self._active),
            candidate=bool(candidate),
            entered=bool(entered),
            exited=bool(exited),
            estimated_speed_m_s=float(estimated_speed_m_s),
            imu_angular_rate_rad_s=float(imu_angular_rate_rad_s),
            visual_angular_rate_rad_s=float(visual_angular_rate_rad_s),
            visual_body_speed_m_s=float(visual_body_speed_m_s),
            angular_rate_disagreement_rad_s=float(disagreement),
            candidate_duration_s=float(self._candidate_duration_s),
            release_duration_s=float(self._release_duration_s),
            reason=reason,
        )


@dataclass(frozen=True)
class ZeroTranslationKinematicEvidence:
    nis: float
    degrees_of_freedom: int
    nis_per_dof: float
    position_residual_norm_m: float
    velocity_residual_norm_m_s: float
    corrected_imu_rotvec_body: torch.Tensor


def zero_translation_kinematic_evidence(
    *,
    pose_WB: torch.Tensor,
    acc_bias: torch.Tensor,
    gyro_bias: torch.Tensor,
    delta_rotation: torch.Tensor,
    delta_velocity: torch.Tensor,
    delta_position: torch.Tensor,
    covariance_pvr: torch.Tensor,
    dt_s: float,
    bias_jacobian: torch.Tensor,
    linearized_acc_bias: torch.Tensor,
    linearized_gyro_bias: torch.Tensor,
    gravity_world: torch.Tensor,
    covariance_eigenvalue_floor: float = 1.0e-12,
) -> ZeroTranslationKinematicEvidence:
    """Test the zero-translation hypothesis using the production IMU contract.

    The residual uses the same [position, velocity] rows, bias correction, and
    residual-side gravity convention as the PACE cached IMU factor. It does not
    use the optimized velocity or ground truth.
    """

    dt_s = float(dt_s)
    if not math.isfinite(dt_s) or dt_s <= 0.0:
        raise ValueError("zero-translation evidence dt must be finite and positive")
    tensors = (
        pose_WB,
        acc_bias,
        gyro_bias,
        delta_rotation,
        delta_velocity,
        delta_position,
        covariance_pvr,
        bias_jacobian,
        linearized_acc_bias,
        linearized_gyro_bias,
        gravity_world,
    )
    if not all(bool(torch.isfinite(torch.as_tensor(value)).all()) for value in tensors):
        raise ValueError("zero-translation evidence input contains NaN/Inf")

    pose = pp.SE3(torch.as_tensor(pose_WB).reshape(1, 7))
    dtype = pose.tensor().dtype
    device = pose.tensor().device
    delta_bias = torch.cat(
        [
            torch.as_tensor(acc_bias, device=device, dtype=dtype).reshape(3)
            - torch.as_tensor(
                linearized_acc_bias, device=device, dtype=dtype
            ).reshape(3),
            torch.as_tensor(gyro_bias, device=device, dtype=dtype).reshape(3)
            - torch.as_tensor(
                linearized_gyro_bias, device=device, dtype=dtype
            ).reshape(3),
        ]
    )
    correction = (
        torch.as_tensor(
            bias_jacobian, device=device, dtype=dtype
        ).reshape(9, 6)
        @ delta_bias
    )
    corrected_position = (
        torch.as_tensor(
            delta_position, device=device, dtype=dtype
        ).reshape(3)
        + correction[0:3]
    )
    corrected_velocity = (
        torch.as_tensor(
            delta_velocity, device=device, dtype=dtype
        ).reshape(3)
        + correction[3:6]
    )
    raw_delta_rotation = torch.as_tensor(
        delta_rotation, device=device, dtype=dtype
    ).reshape(-1)
    if raw_delta_rotation.numel() == 3:
        corrected_rotation = (
            pp.so3(raw_delta_rotation.reshape(1, 3)).Exp()
            @ pp.so3(correction[6:9].reshape(1, 3)).Exp()
        )
    elif raw_delta_rotation.numel() == 4:
        corrected_rotation = (
            pp.SO3(raw_delta_rotation.reshape(1, 4))
            @ pp.so3(correction[6:9].reshape(1, 3)).Exp()
        )
    else:
        raise ValueError("delta_rotation must contain 3 or 4 values")

    rotation_i = pose.rotation().matrix().reshape(3, 3)
    gravity_body = rotation_i.mT @ torch.as_tensor(
        gravity_world, device=device, dtype=dtype
    ).reshape(3)
    residual = torch.cat(
        [
            -0.5 * gravity_body * dt_s * dt_s - corrected_position,
            -gravity_body * dt_s - corrected_velocity,
        ]
    )
    covariance = torch.as_tensor(
        covariance_pvr, device=device, dtype=dtype
    ).reshape(9, 9)[:6, :6]
    covariance = 0.5 * (covariance + covariance.mT)
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    scale = max(float(eigenvalues.abs().max().detach().cpu().item()), 1.0)
    threshold = max(
        float(covariance_eigenvalue_floor),
        torch.finfo(dtype).eps * scale * 6,
    )
    positive = eigenvalues > threshold
    rank = int(positive.sum().detach().cpu().item())
    inverse_values = torch.where(
        positive,
        eigenvalues.reciprocal(),
        torch.zeros_like(eigenvalues),
    )
    information = eigenvectors @ torch.diag(inverse_values) @ eigenvectors.mT
    nis = float((residual @ information @ residual).detach().cpu().item())
    return ZeroTranslationKinematicEvidence(
        nis=nis,
        degrees_of_freedom=rank,
        nis_per_dof=nis / max(rank, 1),
        position_residual_norm_m=float(
            residual[:3].norm().detach().cpu().item()
        ),
        velocity_residual_norm_m_s=float(
            residual[3:6].norm().detach().cpu().item()
        ),
        corrected_imu_rotvec_body=(
            corrected_rotation.Log().tensor().reshape(3).detach().clone()
        ),
    )


@dataclass(frozen=True)
class NearZeroVelocityThresholdsV2:
    minimum_imu_angular_rate_rad_s: float = 0.40
    minimum_visual_angular_rate_rad_s: float = 0.40
    maximum_rotation_vector_rate_difference_rad_s: float = 0.05
    minimum_rotation_axis_cosine: float = 0.995
    maximum_zero_translation_nis_per_dof: float = 1.25
    enter_hold_s: float = 0.20
    release_hold_s: float = 0.10


# Frozen from motion-label calibration on detector evidence only. Downstream
# APE/RPE and optimized trajectory quality are deliberately excluded.
FROZEN_NEAR_ZERO_VELOCITY_V2 = NearZeroVelocityThresholdsV2()


@dataclass(frozen=True)
class NearZeroVelocityDecisionV2:
    active: bool
    candidate: bool
    entered: bool
    exited: bool
    estimated_speed_m_s: float
    imu_angular_rate_rad_s: float
    visual_angular_rate_rad_s: float
    visual_body_speed_m_s: float
    rotation_vector_rate_difference_rad_s: float
    rotation_axis_cosine: float
    zero_translation_nis: float
    zero_translation_dof: int
    zero_translation_nis_per_dof: float
    zero_translation_position_residual_norm_m: float
    zero_translation_velocity_residual_norm_m_s: float
    candidate_duration_s: float
    release_duration_s: float
    reason: str


class TurningNearZeroVelocityDetectorV2:
    """Causal stop-turn detector independent of optimized speed.

    High, mutually consistent IMU/visual rotation is the primary discriminator.
    The zero-translation NIS is retained as a statistical rejection gate. The
    estimated speed and visual translation are diagnostics only.
    """

    def __init__(
        self,
        *,
        minimum_imu_angular_rate_rad_s: float = (
            FROZEN_NEAR_ZERO_VELOCITY_V2.minimum_imu_angular_rate_rad_s
        ),
        minimum_visual_angular_rate_rad_s: float = (
            FROZEN_NEAR_ZERO_VELOCITY_V2.minimum_visual_angular_rate_rad_s
        ),
        maximum_rotation_vector_rate_difference_rad_s: float = (
            FROZEN_NEAR_ZERO_VELOCITY_V2.maximum_rotation_vector_rate_difference_rad_s
        ),
        minimum_rotation_axis_cosine: float = (
            FROZEN_NEAR_ZERO_VELOCITY_V2.minimum_rotation_axis_cosine
        ),
        maximum_zero_translation_nis_per_dof: float = (
            FROZEN_NEAR_ZERO_VELOCITY_V2.maximum_zero_translation_nis_per_dof
        ),
        enter_hold_s: float = FROZEN_NEAR_ZERO_VELOCITY_V2.enter_hold_s,
        release_hold_s: float = FROZEN_NEAR_ZERO_VELOCITY_V2.release_hold_s,
    ) -> None:
        values = (
            minimum_imu_angular_rate_rad_s,
            minimum_visual_angular_rate_rad_s,
            maximum_rotation_vector_rate_difference_rad_s,
            minimum_rotation_axis_cosine,
            maximum_zero_translation_nis_per_dof,
            enter_hold_s,
            release_hold_s,
        )
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("V2 detector thresholds must be finite")
        if minimum_imu_angular_rate_rad_s < 0.0:
            raise ValueError("minimum IMU angular rate must be nonnegative")
        if minimum_visual_angular_rate_rad_s < 0.0:
            raise ValueError("minimum visual angular rate must be nonnegative")
        if maximum_rotation_vector_rate_difference_rad_s < 0.0:
            raise ValueError("maximum rotation-vector disagreement must be nonnegative")
        if not -1.0 <= minimum_rotation_axis_cosine <= 1.0:
            raise ValueError("minimum rotation-axis cosine must be in [-1, 1]")
        if maximum_zero_translation_nis_per_dof <= 0.0:
            raise ValueError("maximum zero-translation NIS/DoF must be positive")
        if enter_hold_s <= 0.0 or release_hold_s <= 0.0:
            raise ValueError("V2 detector hold durations must be positive")

        self.minimum_imu_angular_rate_rad_s = float(
            minimum_imu_angular_rate_rad_s
        )
        self.minimum_visual_angular_rate_rad_s = float(
            minimum_visual_angular_rate_rad_s
        )
        self.maximum_rotation_vector_rate_difference_rad_s = float(
            maximum_rotation_vector_rate_difference_rad_s
        )
        self.minimum_rotation_axis_cosine = float(
            minimum_rotation_axis_cosine
        )
        self.maximum_zero_translation_nis_per_dof = float(
            maximum_zero_translation_nis_per_dof
        )
        self.enter_hold_s = float(enter_hold_s)
        self.release_hold_s = float(release_hold_s)
        self.reset()

    def reset(self) -> None:
        self._active = False
        self._candidate_duration_s = 0.0
        self._release_duration_s = 0.0

    @property
    def active(self) -> bool:
        return bool(self._active)

    def update(
        self,
        *,
        dt_s: float,
        estimated_speed_m_s: float,
        imu_rotvec_body: torch.Tensor,
        visual_rotvec_body: torch.Tensor,
        visual_body_speed_m_s: float,
        zero_translation_evidence: ZeroTranslationKinematicEvidence,
    ) -> NearZeroVelocityDecisionV2:
        dt_s = float(dt_s)
        estimated_speed_m_s = float(estimated_speed_m_s)
        visual_body_speed_m_s = float(visual_body_speed_m_s)
        imu_rotvec = torch.as_tensor(imu_rotvec_body).reshape(3)
        visual_rotvec = torch.as_tensor(visual_rotvec_body).reshape(3).to(
            imu_rotvec
        )
        if (
            not math.isfinite(dt_s)
            or dt_s <= 0.0
            or not math.isfinite(estimated_speed_m_s)
            or not math.isfinite(visual_body_speed_m_s)
            or not bool(torch.isfinite(imu_rotvec).all())
            or not bool(torch.isfinite(visual_rotvec).all())
            or not math.isfinite(zero_translation_evidence.nis_per_dof)
        ):
            raise ValueError("V2 detector input contains NaN/Inf or invalid dt")

        imu_norm = float(imu_rotvec.norm().detach().cpu().item())
        visual_norm = float(visual_rotvec.norm().detach().cpu().item())
        imu_rate = imu_norm / dt_s
        visual_rate = visual_norm / dt_s
        vector_difference_rate = float(
            (imu_rotvec - visual_rotvec).norm().detach().cpu().item()
        ) / dt_s
        axis_cosine = float(
            torch.dot(imu_rotvec, visual_rotvec).detach().cpu().item()
            / max(imu_norm * visual_norm, 1.0e-12)
        )

        imu_turning = imu_rate >= self.minimum_imu_angular_rate_rad_s
        visual_turning = (
            visual_rate >= self.minimum_visual_angular_rate_rad_s
        )
        vector_agrees = (
            vector_difference_rate
            <= self.maximum_rotation_vector_rate_difference_rad_s
        )
        axis_agrees = axis_cosine >= self.minimum_rotation_axis_cosine
        translation_agrees = (
            zero_translation_evidence.nis_per_dof
            <= self.maximum_zero_translation_nis_per_dof
        )
        candidate = (
            imu_turning
            and visual_turning
            and vector_agrees
            and axis_agrees
            and translation_agrees
        )

        entered = False
        exited = False
        if candidate:
            self._candidate_duration_s += dt_s
            self._release_duration_s = 0.0
            if (
                not self._active
                and self._candidate_duration_s + 1.0e-12 >= self.enter_hold_s
            ):
                self._active = True
                entered = True
        else:
            self._candidate_duration_s = 0.0
            if self._active:
                self._release_duration_s += dt_s
                if self._release_duration_s + 1.0e-12 >= self.release_hold_s:
                    self._active = False
                    self._release_duration_s = 0.0
                    exited = True
            else:
                self._release_duration_s = 0.0

        failed = []
        if not imu_turning:
            failed.append("imu_rotation")
        if not visual_turning:
            failed.append("visual_rotation")
        if not vector_agrees:
            failed.append("rotation_vector_disagreement")
        if not axis_agrees:
            failed.append("rotation_axis")
        if not translation_agrees:
            failed.append("zero_translation_nis")
        if self._active:
            reason = "active"
        elif candidate:
            reason = "candidate_hold"
        else:
            reason = "rejected:" + ",".join(failed)

        return NearZeroVelocityDecisionV2(
            active=bool(self._active),
            candidate=bool(candidate),
            entered=bool(entered),
            exited=bool(exited),
            estimated_speed_m_s=estimated_speed_m_s,
            imu_angular_rate_rad_s=imu_rate,
            visual_angular_rate_rad_s=visual_rate,
            visual_body_speed_m_s=visual_body_speed_m_s,
            rotation_vector_rate_difference_rad_s=vector_difference_rate,
            rotation_axis_cosine=axis_cosine,
            zero_translation_nis=float(zero_translation_evidence.nis),
            zero_translation_dof=int(
                zero_translation_evidence.degrees_of_freedom
            ),
            zero_translation_nis_per_dof=float(
                zero_translation_evidence.nis_per_dof
            ),
            zero_translation_position_residual_norm_m=float(
                zero_translation_evidence.position_residual_norm_m
            ),
            zero_translation_velocity_residual_norm_m_s=float(
                zero_translation_evidence.velocity_residual_norm_m_s
            ),
            candidate_duration_s=float(self._candidate_duration_s),
            release_duration_s=float(self._release_duration_s),
            reason=reason,
        )
