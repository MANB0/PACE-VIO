from __future__ import annotations

from dataclasses import dataclass

import pypose as pp
import torch
from pypose.lietensor.operation import se3_Jl_inv

from Utility.IMUKinematics import (
    vio_bias_random_walk_residual,
    vio_preintegrated_imu_residual,
)
from Utility.Point import point2pixel_NED


STATE_DOF = 15
PAIR_DOF = 2 * STATE_DOF


@dataclass(frozen=True)
class NavigationState:
    """A body-to-world navigation state at one image timestamp."""

    pose_WB: torch.Tensor
    velocity_W: torch.Tensor
    acc_bias: torch.Tensor
    gyro_bias: torch.Tensor

    def to(self, *, device: torch.device, dtype: torch.dtype) -> "NavigationState":
        return NavigationState(
            pose_WB=self.pose_WB.reshape(1, 7).to(device=device, dtype=dtype),
            velocity_W=self.velocity_W.reshape(3).to(device=device, dtype=dtype),
            acc_bias=self.acc_bias.reshape(3).to(device=device, dtype=dtype),
            gyro_bias=self.gyro_bias.reshape(3).to(device=device, dtype=dtype),
        )

    def detach(self) -> "NavigationState":
        return NavigationState(
            pose_WB=self.pose_WB.detach().clone(),
            velocity_W=self.velocity_W.detach().clone(),
            acc_bias=self.acc_bias.detach().clone(),
            gyro_bias=self.gyro_bias.detach().clone(),
        )


@dataclass(frozen=True)
class ImuPreintegrationFactor:
    """Preintegrated IMU data ordered as [delta_p, delta_v, delta_phi]."""

    delta_rotation: torch.Tensor
    delta_velocity: torch.Tensor
    delta_position: torch.Tensor
    covariance: torch.Tensor
    dt: float
    bias_jacobian: torch.Tensor
    linearized_acc_bias: torch.Tensor
    linearized_gyro_bias: torch.Tensor
    bias_rw_covariance: torch.Tensor
    gravity_world: torch.Tensor | None = None
    gravity_handling: str = "preintegration"

    def to(self, *, device: torch.device, dtype: torch.dtype) -> "ImuPreintegrationFactor":
        return ImuPreintegrationFactor(
            delta_rotation=self.delta_rotation.to(device=device, dtype=dtype),
            delta_velocity=self.delta_velocity.reshape(3).to(device=device, dtype=dtype),
            delta_position=self.delta_position.reshape(3).to(device=device, dtype=dtype),
            covariance=self.covariance.reshape(9, 9).to(device=device, dtype=dtype),
            dt=float(self.dt),
            bias_jacobian=self.bias_jacobian.reshape(9, 6).to(device=device, dtype=dtype),
            linearized_acc_bias=self.linearized_acc_bias.reshape(3).to(device=device, dtype=dtype),
            linearized_gyro_bias=self.linearized_gyro_bias.reshape(3).to(device=device, dtype=dtype),
            bias_rw_covariance=self.bias_rw_covariance.reshape(6, 6).to(device=device, dtype=dtype),
            gravity_world=(
                None
                if self.gravity_world is None
                else self.gravity_world.reshape(3).to(device=device, dtype=dtype)
            ),
            gravity_handling=str(self.gravity_handling),
        )


@dataclass(frozen=True)
class RelativePoseFactor:
    """MACVO relative body-pose measurement and its SE(3) covariance."""

    measurement_BiBj: torch.Tensor
    covariance: torch.Tensor
    huber_delta: float = 3.0

    def to(self, *, device: torch.device, dtype: torch.dtype) -> "RelativePoseFactor":
        return RelativePoseFactor(
            measurement_BiBj=self.measurement_BiBj.reshape(1, 7).to(device=device, dtype=dtype),
            covariance=self.covariance.reshape(6, 6).to(device=device, dtype=dtype),
            huber_delta=float(self.huber_delta),
        )


@dataclass(frozen=True)
class UVDFactor:
    """MACVO source-frame 3D points and target-frame UVD observations."""

    points_Ci: torch.Tensor
    target_uv: torch.Tensor
    target_disparity: torch.Tensor
    covariance_uvd: torch.Tensor
    intrinsic: torch.Tensor
    baseline: float
    extrinsic_CI: torch.Tensor
    huber_delta: float = 0.1
    optimization_mode: str = "full"
    anchor_relative_CjCi: torch.Tensor | None = None

    def to(self, *, device: torch.device, dtype: torch.dtype) -> "UVDFactor":
        count = int(self.points_Ci.reshape(-1, 3).shape[0])
        return UVDFactor(
            points_Ci=self.points_Ci.reshape(count, 3).to(device=device, dtype=dtype),
            target_uv=self.target_uv.reshape(count, 2).to(device=device, dtype=dtype),
            target_disparity=self.target_disparity.reshape(count, 1).to(
                device=device, dtype=dtype
            ),
            covariance_uvd=self.covariance_uvd.reshape(count, 3, 3).to(
                device=device, dtype=dtype
            ),
            intrinsic=self.intrinsic.reshape(3, 3).to(device=device, dtype=dtype),
            baseline=float(self.baseline),
            extrinsic_CI=self.extrinsic_CI.reshape(1, 7).to(device=device, dtype=dtype),
            huber_delta=float(self.huber_delta),
            optimization_mode=str(self.optimization_mode),
            anchor_relative_CjCi=(
                None
                if self.anchor_relative_CjCi is None
                else self.anchor_relative_CjCi.reshape(1, 7).to(
                    device=device, dtype=dtype
                )
            ),
        )


@dataclass(frozen=True)
class LinearizedUVDPoseFactor:
    """Fixed-linearization UVD pose factor in the Cj<-Ci right tangent."""

    reference_relative_CjCi: torch.Tensor
    sqrt_information: torch.Tensor
    residual_offset: torch.Tensor
    extrinsic_CI: torch.Tensor
    marginal_mode: str
    huber_delta: float = 1.0e12

    def to(
        self, *, device: torch.device, dtype: torch.dtype
    ) -> "LinearizedUVDPoseFactor":
        rows = int(self.sqrt_information.reshape(-1, 6).shape[0])
        return LinearizedUVDPoseFactor(
            reference_relative_CjCi=self.reference_relative_CjCi.reshape(1, 7).to(
                device=device, dtype=dtype
            ),
            sqrt_information=self.sqrt_information.reshape(rows, 6).to(
                device=device, dtype=dtype
            ),
            residual_offset=self.residual_offset.reshape(rows).to(
                device=device, dtype=dtype
            ),
            extrinsic_CI=self.extrinsic_CI.reshape(1, 7).to(
                device=device, dtype=dtype
            ),
            marginal_mode=str(self.marginal_mode),
            huber_delta=float(self.huber_delta),
        )


@dataclass(frozen=True)
class UVDPoseLinearization:
    """Auditable local normal equations used to build a UVD pose factor."""

    factor: LinearizedUVDPoseFactor
    robust_residual: torch.Tensor
    relative_jacobian: torch.Tensor
    full_hessian: torch.Tensor
    full_gradient: torch.Tensor
    reduced_hessian: torch.Tensor
    reduced_gradient: torch.Tensor
    retained_indices: torch.Tensor
    nuisance_indices: torch.Tensor


VisualFactor = RelativePoseFactor | UVDFactor | LinearizedUVDPoseFactor


@dataclass(frozen=True)
class SquareRootPrior:
    """Linearized history prior r = A * boxminus(x, x0) + c."""

    reference: NavigationState
    sqrt_information: torch.Tensor
    residual_offset: torch.Tensor

    def to(self, *, device: torch.device, dtype: torch.dtype) -> "SquareRootPrior":
        return SquareRootPrior(
            reference=self.reference.to(device=device, dtype=dtype),
            sqrt_information=self.sqrt_information.to(device=device, dtype=dtype),
            residual_offset=self.residual_offset.reshape(-1).to(device=device, dtype=dtype),
        )


@dataclass(frozen=True)
class TwoStateVIOProblem:
    state_i: NavigationState
    state_j: NavigationState
    prior_i: SquareRootPrior
    imu: ImuPreintegrationFactor
    visual_pose: VisualFactor
    optimize_acc_bias: bool = True
    optimize_gyro_bias: bool = True


@dataclass(frozen=True)
class TwoStateVIOResult:
    state_i: NavigationState
    state_j: NavigationState
    prior_j: SquareRootPrior
    converged: bool
    iterations: int
    initial_cost: float
    final_cost: float
    prior_cost: float
    imu_cost: float
    bias_cost: float
    visual_pose_cost: float
    hessian: torch.Tensor
    gradient: torch.Tensor
    final_step_norm: float
    final_gradient_inf_norm: float
    convergence_reason: str
    accepted_steps: int
    rejected_steps: int


def make_diagonal_prior(
    reference: NavigationState,
    *,
    pose_translation_std: float,
    pose_rotation_std: float,
    velocity_std: float,
    acc_bias_std: float,
    gyro_bias_std: float,
) -> SquareRootPrior:
    std = torch.tensor(
        [pose_translation_std] * 3
        + [pose_rotation_std] * 3
        + [velocity_std] * 3
        + [acc_bias_std] * 3
        + [gyro_bias_std] * 3,
        dtype=reference.pose_WB.dtype,
        device=reference.pose_WB.device,
    )
    if bool((std <= 0).any()):
        raise ValueError("all prior standard deviations must be positive")
    return SquareRootPrior(
        reference=reference.detach(),
        sqrt_information=torch.diag(std.reciprocal()),
        residual_offset=torch.zeros(STATE_DOF, dtype=std.dtype, device=std.device),
    )


def state_boxminus(state: NavigationState, reference: NavigationState) -> torch.Tensor:
    pose_error = (
        pp.SE3(reference.pose_WB.reshape(1, 7)).Inv()
        @ pp.SE3(state.pose_WB.reshape(1, 7))
    ).Log().tensor().reshape(6)
    return torch.cat(
        [
            pose_error,
            state.velocity_W.reshape(3) - reference.velocity_W.reshape(3),
            state.acc_bias.reshape(3) - reference.acc_bias.reshape(3),
            state.gyro_bias.reshape(3) - reference.gyro_bias.reshape(3),
        ]
    )


def retract_state(state: NavigationState, increment: torch.Tensor) -> NavigationState:
    increment = increment.reshape(STATE_DOF)
    pose = pp.SE3(state.pose_WB.reshape(1, 7)) @ pp.se3(increment[0:6].reshape(1, 6)).Exp()
    return NavigationState(
        pose_WB=pose.tensor(),
        velocity_W=state.velocity_W.reshape(3) + increment[6:9],
        acc_bias=state.acc_bias.reshape(3) + increment[9:12],
        gyro_bias=state.gyro_bias.reshape(3) + increment[12:15],
    )


def _covariance_cholesky(covariance: torch.Tensor, eigenvalue_floor: float) -> torch.Tensor:
    covariance = covariance.to(dtype=torch.float64) if covariance.dtype == torch.float32 else covariance
    covariance = 0.5 * (covariance + covariance.mT)
    values, vectors = torch.linalg.eigh(covariance)
    scale = max(float(values.abs().max().detach().cpu().item()), 1.0)
    floor = max(float(eigenvalue_floor), torch.finfo(covariance.dtype).eps * scale)
    stabilized = vectors @ torch.diag(values.clamp_min(floor)) @ vectors.mT
    return torch.linalg.cholesky(0.5 * (stabilized + stabilized.mT))


def _whiten(residual: torch.Tensor, covariance: torch.Tensor, eigenvalue_floor: float) -> torch.Tensor:
    lower = _covariance_cholesky(covariance, eigenvalue_floor)
    return torch.linalg.solve_triangular(
        lower,
        residual.reshape(-1, 1).to(lower),
        upper=False,
    ).reshape(-1)


def _whiten_rows(
    residual: torch.Tensor,
    covariance: torch.Tensor,
    eigenvalue_floor: float,
) -> torch.Tensor:
    covariance = covariance.to(dtype=torch.float64) if covariance.dtype == torch.float32 else covariance
    covariance = 0.5 * (covariance + covariance.transpose(-1, -2))
    values, vectors = torch.linalg.eigh(covariance)
    scale = values.abs().amax(dim=-1).clamp_min(1.0)
    floor = torch.maximum(
        torch.full_like(scale, float(eigenvalue_floor)),
        torch.as_tensor(torch.finfo(covariance.dtype).eps, device=scale.device) * scale,
    )
    stabilized = (
        vectors
        @ torch.diag_embed(torch.maximum(values, floor.unsqueeze(-1)))
        @ vectors.transpose(-1, -2)
    )
    lower = torch.linalg.cholesky(0.5 * (stabilized + stabilized.transpose(-1, -2)))
    return torch.linalg.solve_triangular(
        lower,
        residual.reshape(-1, 3, 1).to(lower),
        upper=False,
    ).reshape(-1, 3)


def _camera_relative_CjCi(
    state_i: NavigationState,
    state_j: NavigationState,
    extrinsic_CI: torch.Tensor,
) -> pp.LieTensor:
    extrinsic = pp.SE3(extrinsic_CI)
    pose_WCi = pp.SE3(state_i.pose_WB) @ extrinsic.Inv()
    pose_WCj = pp.SE3(state_j.pose_WB) @ extrinsic.Inv()
    return pose_WCj.Inv() @ pose_WCi


def _uvd_whitened_rows_from_relative(
    relative_CjCi: pp.LieTensor,
    visual: UVDFactor,
    covariance_eigenvalue_floor: float,
) -> torch.Tensor:
    predicted_Cj = relative_CjCi.Act(visual.points_Ci)
    predicted_uv = point2pixel_NED(predicted_Cj, visual.intrinsic)
    predicted_disparity = (
        visual.intrinsic[0, 0] * float(visual.baseline)
        / predicted_Cj[:, 0:1]
    )
    raw = torch.cat(
        [
            predicted_uv - visual.target_uv,
            predicted_disparity - visual.target_disparity,
        ],
        dim=-1,
    )
    if not bool(torch.isfinite(raw).all()):
        raise FloatingPointError("direct UVD residual contains NaN/Inf")
    return _whiten_rows(raw, visual.covariance_uvd, covariance_eigenvalue_floor)


def uvd_whitened_rows_from_relative(
    relative_CjCi: torch.Tensor,
    visual: UVDFactor,
    covariance_eigenvalue_floor: float = 1.0e-12,
) -> torch.Tensor:
    """Evaluate the production UVD residual at a supplied Cj<-Ci pose."""

    relative = pp.SE3(
        relative_CjCi.reshape(1, 7).to(
            device=visual.points_Ci.device,
            dtype=visual.points_Ci.dtype,
        )
    )
    return _uvd_whitened_rows_from_relative(
        relative,
        visual,
        covariance_eigenvalue_floor,
    )


def _symmetric_pseudoinverse(
    matrix: torch.Tensor,
    eigenvalue_floor: float,
) -> torch.Tensor:
    matrix = 0.5 * (matrix + matrix.mT)
    values, vectors = torch.linalg.eigh(matrix)
    scale = max(float(values.abs().max().detach().cpu().item()), 1.0)
    threshold = max(
        float(eigenvalue_floor),
        torch.finfo(matrix.dtype).eps * scale * max(int(matrix.shape[0]), 1),
    )
    inverse_values = torch.where(
        values > threshold,
        values.reciprocal(),
        torch.zeros_like(values),
    )
    return vectors @ torch.diag(inverse_values) @ vectors.mT


def _sqrt_factor_from_normal_equations(
    hessian: torch.Tensor,
    gradient: torch.Tensor,
    *,
    full_dimension: int,
    retained_indices: torch.Tensor,
    eigenvalue_floor: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    hessian = 0.5 * (hessian + hessian.mT)
    values, vectors = torch.linalg.eigh(hessian)
    scale = max(float(values.abs().max().detach().cpu().item()), 1.0)
    threshold = max(
        float(eigenvalue_floor),
        torch.finfo(hessian.dtype).eps * scale * max(int(hessian.shape[0]), 1),
    )
    positive = values > threshold
    if not bool(positive.any()):
        raise RuntimeError("UVD marginalization produced an empty visual factor")
    active_values = values[positive]
    active_vectors = vectors[:, positive]
    sqrt_reduced = torch.diag(active_values.sqrt()) @ active_vectors.mT
    residual_offset = (
        torch.diag(active_values.rsqrt()) @ active_vectors.mT @ gradient
    )
    sqrt_information = torch.zeros(
        (sqrt_reduced.shape[0], full_dimension),
        dtype=hessian.dtype,
        device=hessian.device,
    )
    sqrt_information[:, retained_indices] = sqrt_reduced
    return sqrt_information, residual_offset


def linearized_uvd_pose_factor_from_normal_equations(
    reference_relative_CjCi: torch.Tensor,
    hessian: torch.Tensor,
    gradient: torch.Tensor,
    extrinsic_CI: torch.Tensor,
    *,
    normal_eigenvalue_floor: float = 1.0e-10,
) -> LinearizedUVDPoseFactor:
    """Rebuild a fixed UVD factor from cached right-tangent normal equations.

    ``hessian`` and ``gradient`` must use the 6D ``[translation, rotation]``
    right tangent of ``T_CjCi`` at ``reference_relative_CjCi``.
    """

    reference = reference_relative_CjCi.reshape(1, 7)
    hessian = hessian.reshape(6, 6).to(reference)
    gradient = gradient.reshape(6).to(reference)
    if not bool(
        torch.isfinite(reference).all()
        and torch.isfinite(hessian).all()
        and torch.isfinite(gradient).all()
    ):
        raise FloatingPointError("cached UVD normal equations contain NaN/Inf")
    symmetry_error = float((hessian - hessian.mT).abs().max().detach().cpu().item())
    if symmetry_error > 1.0e-8:
        raise ValueError("cached UVD Hessian is not symmetric")
    hessian = 0.5 * (hessian + hessian.mT)
    retained = torch.arange(6, device=hessian.device)
    sqrt_information, residual_offset = _sqrt_factor_from_normal_equations(
        hessian,
        gradient,
        full_dimension=6,
        retained_indices=retained,
        eigenvalue_floor=normal_eigenvalue_floor,
    )
    return LinearizedUVDPoseFactor(
        reference_relative_CjCi=reference.detach().clone(),
        sqrt_information=sqrt_information.detach(),
        residual_offset=residual_offset.detach(),
        extrinsic_CI=extrinsic_CI.reshape(1, 7).to(reference).detach().clone(),
        marginal_mode="full",
    )


def linearize_uvd_relative_pose_factor(
    reference_relative_CjCi: torch.Tensor,
    visual: UVDFactor,
    *,
    marginal_mode: str,
    covariance_eigenvalue_floor: float = 1.0e-12,
    normal_eigenvalue_floor: float = 1.0e-10,
) -> UVDPoseLinearization:
    """Compress the UVD objective at a supplied Cj<-Ci right-tangent pose."""

    mode = str(marginal_mode).strip().lower()
    if mode not in {"full", "translation", "rotation"}:
        raise ValueError(f"unsupported UVD marginal mode: {marginal_mode}")
    if visual.points_Ci.numel() == 0:
        raise ValueError("UVD pose linearization requires at least one point")
    reference = pp.SE3(
        reference_relative_CjCi.reshape(1, 7).to(
            device=visual.points_Ci.device,
            dtype=visual.points_Ci.dtype,
        )
    ).detach()
    base_rows = _uvd_whitened_rows_from_relative(
        reference, visual, covariance_eigenvalue_floor
    )
    base_norms = torch.linalg.vector_norm(base_rows, dim=-1)
    delta = max(float(visual.huber_delta), 1.0e-12)
    robust_weight = torch.where(
        base_norms <= delta,
        torch.ones_like(base_norms),
        torch.as_tensor(
            delta, dtype=base_norms.dtype, device=base_norms.device
        )
        / base_norms.clamp_min(1.0e-12),
    ).detach()
    sqrt_weight = robust_weight.sqrt().unsqueeze(-1)
    zero = torch.zeros(
        6,
        dtype=visual.points_Ci.dtype,
        device=visual.points_Ci.device,
        requires_grad=True,
    )

    def robust_residual(increment: torch.Tensor) -> torch.Tensor:
        candidate = reference @ pp.se3(increment.reshape(1, 6)).Exp()
        rows = _uvd_whitened_rows_from_relative(
            candidate, visual, covariance_eigenvalue_floor
        )
        return (sqrt_weight * rows).reshape(-1)

    residual = robust_residual(zero)
    jacobian = torch.autograd.functional.jacobian(
        robust_residual,
        zero,
        create_graph=False,
        vectorize=True,
    )
    full_hessian = 0.5 * (
        jacobian.mT @ jacobian + (jacobian.mT @ jacobian).mT
    )
    full_gradient = jacobian.mT @ residual.detach()

    if mode == "full":
        retained = torch.arange(6, device=full_hessian.device)
        nuisance = torch.empty(0, dtype=torch.long, device=full_hessian.device)
        reduced_hessian = full_hessian
        reduced_gradient = full_gradient
    else:
        retained = torch.arange(
            0 if mode == "translation" else 3,
            3 if mode == "translation" else 6,
            device=full_hessian.device,
        )
        nuisance = torch.arange(
            3 if mode == "translation" else 0,
            6 if mode == "translation" else 3,
            device=full_hessian.device,
        )
        h_rr = full_hessian[retained][:, retained]
        h_rn = full_hessian[retained][:, nuisance]
        h_nr = full_hessian[nuisance][:, retained]
        h_nn = full_hessian[nuisance][:, nuisance]
        g_r = full_gradient[retained]
        g_n = full_gradient[nuisance]
        h_nn_inverse = _symmetric_pseudoinverse(
            h_nn, normal_eigenvalue_floor
        )
        reduced_hessian = h_rr - h_rn @ h_nn_inverse @ h_nr
        reduced_hessian = 0.5 * (reduced_hessian + reduced_hessian.mT)
        reduced_gradient = g_r - h_rn @ h_nn_inverse @ g_n

    sqrt_information, residual_offset = _sqrt_factor_from_normal_equations(
        reduced_hessian,
        reduced_gradient,
        full_dimension=6,
        retained_indices=retained,
        eigenvalue_floor=normal_eigenvalue_floor,
    )
    factor = LinearizedUVDPoseFactor(
        reference_relative_CjCi=reference.tensor().detach(),
        sqrt_information=sqrt_information.detach(),
        residual_offset=residual_offset.detach(),
        extrinsic_CI=visual.extrinsic_CI.detach().clone(),
        marginal_mode=mode,
    )
    return UVDPoseLinearization(
        factor=factor,
        robust_residual=residual.detach(),
        relative_jacobian=jacobian.detach(),
        full_hessian=full_hessian.detach(),
        full_gradient=full_gradient.detach(),
        reduced_hessian=reduced_hessian.detach(),
        reduced_gradient=reduced_gradient.detach(),
        retained_indices=retained.detach(),
        nuisance_indices=nuisance.detach(),
    )


def linearize_uvd_pose_factor(
    state_i: NavigationState,
    state_j: NavigationState,
    visual: UVDFactor,
    *,
    marginal_mode: str,
    covariance_eigenvalue_floor: float = 1.0e-12,
    normal_eigenvalue_floor: float = 1.0e-10,
) -> UVDPoseLinearization:
    """Build a fixed local UVD factor at the current two-state camera motion."""

    reference = _camera_relative_CjCi(
        state_i, state_j, visual.extrinsic_CI
    ).detach()
    return linearize_uvd_relative_pose_factor(
        reference.tensor(),
        visual,
        marginal_mode=marginal_mode,
        covariance_eigenvalue_floor=covariance_eigenvalue_floor,
        normal_eigenvalue_floor=normal_eigenvalue_floor,
    )


def solve_uvd_relative_pose_visual_only(
    visual: UVDFactor,
    *,
    initial_relative_CjCi: torch.Tensor | None = None,
    max_iterations: int = 8,
    damping: float = 1.0e-6,
    normal_eigenvalue_floor: float = 1.0e-10,
    step_tolerance: float = 1.0e-6,
) -> tuple[torch.Tensor, dict[str, float | int | bool | str]]:
    """Solve one MACVO UVD pair without IMU, prior, or backend state.

    The returned pose maps ``C_i`` coordinates to ``C_j`` coordinates.  This
    is deliberately a small visual-only sidecar for live comparison; it is
    not used as a warm start by the VIO optimizer.  The objective and right
    ``[translation, rotation]`` tangent are the same as the deployable UVD
    factor, so the sidecar is an actual pure-visual result rather than the
    motion-model prediction.
    """
    if initial_relative_CjCi is None:
        current = pp.identity_SE3(
            1,
            dtype=visual.points_Ci.dtype,
            device=visual.points_Ci.device,
        )
    else:
        current = pp.SE3(initial_relative_CjCi.reshape(1, 7).to(
            device=visual.points_Ci.device,
            dtype=visual.points_Ci.dtype,
        ))

    iterations = 0
    final_step_norm = float("inf")
    converged = False
    reason = "iteration_limit"
    for iteration in range(max(int(max_iterations), 1)):
        linearization = linearize_uvd_relative_pose_factor(
            current.tensor(),
            visual,
            marginal_mode="full",
            normal_eigenvalue_floor=normal_eigenvalue_floor,
        )
        hessian = 0.5 * (
            linearization.full_hessian + linearization.full_hessian.mT
        )
        diagonal_scale = float(
            hessian.diagonal().abs().max().detach().cpu().item()
        )
        damping_value = max(float(damping) * max(diagonal_scale, 1.0), 1.0e-12)
        damped = hessian + damping_value * torch.eye(
            6, dtype=hessian.dtype, device=hessian.device
        )
        step = -_symmetric_pseudoinverse(
            damped, normal_eigenvalue_floor
        ) @ linearization.full_gradient
        final_step_norm = float(torch.linalg.vector_norm(step).detach().cpu().item())
        current = current @ pp.se3(step.reshape(1, 6)).Exp()
        iterations = iteration + 1
        if not bool(torch.isfinite(current.tensor()).all()):
            reason = "nonfinite"
            break
        if final_step_norm <= float(step_tolerance):
            converged = True
            reason = "step_tolerance"
            break

    final_rows = _uvd_whitened_rows_from_relative(
        current, visual, covariance_eigenvalue_floor=1.0e-12
    )
    final_norms = torch.linalg.vector_norm(final_rows, dim=-1)
    huber_delta = max(float(visual.huber_delta), 1.0e-12)
    robust_cost = torch.where(
        final_norms <= huber_delta,
        0.5 * final_norms.square(),
        huber_delta * (final_norms - 0.5 * huber_delta),
    ).sum()
    return current.tensor().detach(), {
        "iterations": iterations,
        "final_step_norm": final_step_norm,
        "converged": converged,
        "reason": reason,
        "robust_cost": float(robust_cost.detach().cpu().item()),
        "num_points": int(final_norms.numel()),
        "num_inliers": int((final_norms <= huber_delta).sum().detach().cpu().item()),
        "mean_mahalanobis_sq": float(
            final_norms.square().mean().detach().cpu().item()
        ),
    }


def visual_whitened_residuals(
    state_i: NavigationState,
    state_j: NavigationState,
    visual: VisualFactor,
    covariance_eigenvalue_floor: float,
) -> torch.Tensor:
    if isinstance(visual, RelativePoseFactor):
        predicted = pp.SE3(state_i.pose_WB).Inv() @ pp.SE3(state_j.pose_WB)
        visual_raw = (
            pp.SE3(visual.measurement_BiBj).Inv() @ predicted
        ).Log().tensor().reshape(6)
        return _whiten(
            visual_raw, visual.covariance, covariance_eigenvalue_floor
        ).reshape(1, 6)

    if isinstance(visual, LinearizedUVDPoseFactor):
        current = _camera_relative_CjCi(
            state_i, state_j, visual.extrinsic_CI
        )
        local_increment = (
            pp.SE3(visual.reference_relative_CjCi).Inv() @ current
        ).Log().tensor().reshape(6)
        residual = (
            visual.sqrt_information @ local_increment
            + visual.residual_offset
        )
        if not bool(torch.isfinite(residual).all()):
            raise FloatingPointError(
                "linearized UVD pose residual contains NaN/Inf"
            )
        return residual.reshape(-1, 1)

    mode = str(visual.optimization_mode).strip().lower()
    if mode == "no_visual":
        return torch.zeros(
            (0, 3),
            dtype=state_i.pose_WB.dtype,
            device=state_i.pose_WB.device,
        )
    if mode not in {"full", "rotation_only", "translation_only"}:
        raise ValueError(f"unsupported direct UVD optimization mode: {mode}")
    if visual.points_Ci.numel() == 0:
        raise ValueError("direct UVD factor requires at least one point")
    relative_CjCi = _camera_relative_CjCi(
        state_i, state_j, visual.extrinsic_CI
    )
    if mode == "full":
        return _uvd_whitened_rows_from_relative(
            relative_CjCi, visual, covariance_eigenvalue_floor
        )
    else:
        if visual.anchor_relative_CjCi is None:
            raise ValueError(f"direct UVD {mode} mode requires an IMU anchor")
        anchor = pp.SE3(visual.anchor_relative_CjCi)
        zero = torch.zeros(
            3,
            dtype=visual.points_Ci.dtype,
            device=visual.points_Ci.device,
        )
        current_rotation = relative_CjCi.rotation().matrix().reshape(3, 3)
        current_translation = relative_CjCi.Act(zero).reshape(3)
        anchor_rotation = anchor.rotation().matrix().reshape(3, 3).detach()
        anchor_translation = anchor.Act(zero).reshape(3).detach()
        rotation = current_rotation if mode == "rotation_only" else anchor_rotation
        translation = (
            anchor_translation if mode == "rotation_only" else current_translation
        )
        predicted_Cj = torch.einsum(
            "ij,nj->ni", rotation, visual.points_Ci
        ) + translation.unsqueeze(0)
        predicted_uv = point2pixel_NED(predicted_Cj, visual.intrinsic)
        predicted_disparity = (
            visual.intrinsic[0, 0] * float(visual.baseline)
            / predicted_Cj[:, 0:1]
        )
        raw = torch.cat(
            [
                predicted_uv - visual.target_uv,
                predicted_disparity - visual.target_disparity,
            ],
            dim=-1,
        )
        if not bool(torch.isfinite(raw).all()):
            raise FloatingPointError("direct UVD residual contains NaN/Inf")
        return _whiten_rows(
            raw, visual.covariance_uvd, covariance_eigenvalue_floor
        )


def _prior_residual(
    state_i: NavigationState,
    prior: SquareRootPrior,
) -> torch.Tensor:
    return (
        prior.sqrt_information @ state_boxminus(state_i, prior.reference)
        + prior.residual_offset
    )


def _imu_residual(
    state_i: NavigationState,
    state_j: NavigationState,
    imu: ImuPreintegrationFactor,
    covariance_eigenvalue_floor: float,
) -> torch.Tensor:
    raw = vio_preintegrated_imu_residual(
        from_pose=pp.SE3(state_i.pose_WB),
        to_pose=pp.SE3(state_j.pose_WB),
        prev_velocity_world=state_i.velocity_W,
        curr_velocity_world=state_j.velocity_W,
        delta_R=imu.delta_rotation,
        delta_v=imu.delta_velocity,
        delta_p=imu.delta_position,
        dt_total=imu.dt,
        prev_acc_bias=state_i.acc_bias,
        prev_gyro_bias=state_i.gyro_bias,
        curr_acc_bias=state_j.acc_bias,
        curr_gyro_bias=state_j.gyro_bias,
        linearized_acc_bias=imu.linearized_acc_bias,
        linearized_gyro_bias=imu.linearized_gyro_bias,
        bias_jacobian=imu.bias_jacobian,
        sensor_T_imu=None,
        gravity_world=imu.gravity_world,
        gravity_handling=imu.gravity_handling,
    ).reshape(9)
    return _whiten(raw, imu.covariance, covariance_eigenvalue_floor)


def _bias_residual(
    state_i: NavigationState,
    state_j: NavigationState,
    imu: ImuPreintegrationFactor,
    covariance_eigenvalue_floor: float,
) -> torch.Tensor:
    raw = vio_bias_random_walk_residual(
        prev_acc_bias=state_i.acc_bias,
        prev_gyro_bias=state_i.gyro_bias,
        curr_acc_bias=state_j.acc_bias,
        curr_gyro_bias=state_j.gyro_bias,
    ).reshape(6)
    return _whiten(raw, imu.bias_rw_covariance, covariance_eigenvalue_floor)


def _robust_visual_residual(
    state_i: NavigationState,
    state_j: NavigationState,
    visual: VisualFactor,
    covariance_eigenvalue_floor: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    white_rows = visual_whitened_residuals(
        state_i, state_j, visual, covariance_eigenvalue_floor
    )
    norms = torch.linalg.vector_norm(white_rows, dim=-1)
    delta = max(float(visual.huber_delta), 1e-12)
    weight = torch.where(
        norms <= delta,
        torch.ones_like(norms),
        torch.as_tensor(delta, dtype=norms.dtype, device=norms.device)
        / norms.clamp_min(1e-12),
    ).detach()
    return (weight.sqrt().unsqueeze(-1) * white_rows).reshape(-1), white_rows


def _skew(vector: torch.Tensor) -> torch.Tensor:
    x, y, z = vector.reshape(3).unbind()
    zero = torch.zeros((), dtype=vector.dtype, device=vector.device)
    return torch.stack(
        [
            torch.stack([zero, -z, y]),
            torch.stack([z, zero, -x]),
            torch.stack([-y, x, zero]),
        ]
    )


def _se3_adjoint_matrix(transform: pp.LieTensor | torch.Tensor) -> torch.Tensor:
    """Adjoint for the local ``[translation, rotation]`` se(3) ordering."""

    pose = pp.SE3(transform)
    dtype = pose.tensor().dtype
    device = pose.tensor().device
    basis = pp.se3(torch.eye(6, dtype=dtype, device=device))
    return pose.Adj(basis).tensor().reshape(6, 6).mT


def _se3_right_jacobian_inverse(tangent: torch.Tensor) -> torch.Tensor:
    # Jr(x) = Jl(-x), using the same closed form as PyPose Log backward.
    return se3_Jl_inv(-tangent.reshape(1, 6)).reshape(6, 6)


def _prior_residual_and_analytic_jacobian(
    state_i: NavigationState,
    prior: SquareRootPrior,
) -> tuple[torch.Tensor, torch.Tensor]:
    local = state_boxminus(state_i, prior.reference)
    local_jacobian = torch.eye(
        STATE_DOF, dtype=local.dtype, device=local.device
    )
    local_jacobian[0:6, 0:6] = _se3_right_jacobian_inverse(local[0:6])
    pair_jacobian = torch.zeros(
        (prior.sqrt_information.shape[0], PAIR_DOF),
        dtype=local.dtype,
        device=local.device,
    )
    pair_jacobian[:, 0:STATE_DOF] = prior.sqrt_information @ local_jacobian
    return prior.sqrt_information @ local + prior.residual_offset, pair_jacobian


def _linearized_visual_residual_and_analytic_jacobian(
    state_i: NavigationState,
    state_j: NavigationState,
    visual: LinearizedUVDPoseFactor,
) -> tuple[torch.Tensor, torch.Tensor]:
    current = _camera_relative_CjCi(state_i, state_j, visual.extrinsic_CI)
    local = (
        pp.SE3(visual.reference_relative_CjCi).Inv() @ current
    ).Log().tensor().reshape(6)
    residual = visual.sqrt_information @ local + visual.residual_offset

    body_to_camera_tangent = _se3_adjoint_matrix(pp.SE3(visual.extrinsic_CI))
    relative_tangent_jacobian = torch.cat(
        [
            body_to_camera_tangent,
            -_se3_adjoint_matrix(current.Inv()) @ body_to_camera_tangent,
        ],
        dim=1,
    )
    local_jacobian = (
        _se3_right_jacobian_inverse(local) @ relative_tangent_jacobian
    )
    pose_jacobian = visual.sqrt_information @ local_jacobian

    norms = residual.abs()
    delta = max(float(visual.huber_delta), 1e-12)
    weights = torch.where(
        norms <= delta,
        torch.ones_like(norms),
        torch.as_tensor(delta, dtype=norms.dtype, device=norms.device)
        / norms.clamp_min(1e-12),
    ).detach()
    sqrt_weights = weights.sqrt()
    residual = sqrt_weights * residual
    pose_jacobian = sqrt_weights.unsqueeze(-1) * pose_jacobian

    pair_jacobian = torch.zeros(
        (residual.numel(), PAIR_DOF),
        dtype=residual.dtype,
        device=residual.device,
    )
    pair_jacobian[:, 0:6] = pose_jacobian[:, 0:6]
    pair_jacobian[:, STATE_DOF : STATE_DOF + 6] = pose_jacobian[:, 6:12]
    return residual.reshape(-1), pair_jacobian


def _bias_residual_and_analytic_jacobian(
    state_i: NavigationState,
    state_j: NavigationState,
    imu: ImuPreintegrationFactor,
    covariance_eigenvalue_floor: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    raw = vio_bias_random_walk_residual(
        prev_acc_bias=state_i.acc_bias,
        prev_gyro_bias=state_i.gyro_bias,
        curr_acc_bias=state_j.acc_bias,
        curr_gyro_bias=state_j.gyro_bias,
    ).reshape(6)
    raw_jacobian = torch.zeros(
        (6, PAIR_DOF), dtype=raw.dtype, device=raw.device
    )
    identity = torch.eye(3, dtype=raw.dtype, device=raw.device)
    raw_jacobian[0:3, 9:12] = -identity
    raw_jacobian[0:3, STATE_DOF + 9 : STATE_DOF + 12] = identity
    raw_jacobian[3:6, 12:15] = -identity
    raw_jacobian[3:6, STATE_DOF + 12 : STATE_DOF + 15] = identity
    lower = _covariance_cholesky(
        imu.bias_rw_covariance, covariance_eigenvalue_floor
    )
    residual = torch.linalg.solve_triangular(
        lower, raw.reshape(6, 1).to(lower), upper=False
    ).reshape(6)
    jacobian = torch.linalg.solve_triangular(
        lower, raw_jacobian.to(lower), upper=False
    )
    return residual, jacobian


def _factor_residuals(
    state_i: NavigationState,
    state_j: NavigationState,
    prior: SquareRootPrior,
    imu: ImuPreintegrationFactor,
    visual: VisualFactor,
    *,
    covariance_eigenvalue_floor: float,
    robust_visual: bool,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    prior_r = _prior_residual(state_i, prior)
    imu_r = _imu_residual(
        state_i, state_j, imu, covariance_eigenvalue_floor
    )
    bias_r = _bias_residual(
        state_i, state_j, imu, covariance_eigenvalue_floor
    )
    visual_white_rows = visual_whitened_residuals(
        state_i, state_j, visual, covariance_eigenvalue_floor
    )
    if robust_visual:
        norms = torch.linalg.vector_norm(visual_white_rows, dim=-1)
        delta = max(float(visual.huber_delta), 1e-12)
        weight = torch.where(
            norms <= delta,
            torch.ones_like(norms),
            torch.as_tensor(delta, dtype=norms.dtype, device=norms.device)
            / norms.clamp_min(1e-12),
        ).detach()
        visual_r = (
            weight.sqrt().unsqueeze(-1) * visual_white_rows
        ).reshape(-1)
    else:
        visual_r = visual_white_rows.reshape(-1)

    blocks = {
        "prior": prior_r,
        "imu": imu_r,
        "bias": bias_r,
        "visual_pose": visual_r,
        "visual_pose_unweighted": visual_white_rows.reshape(-1),
        "visual_group_norms": torch.linalg.vector_norm(visual_white_rows, dim=-1),
    }
    return torch.cat([prior_r, imu_r, bias_r, visual_r]), blocks


def _true_cost(blocks: dict[str, torch.Tensor], huber_delta: float) -> torch.Tensor:
    base = 0.5 * (
        blocks["prior"].square().sum()
        + blocks["imu"].square().sum()
        + blocks["bias"].square().sum()
    )
    visual_norms = blocks["visual_group_norms"]
    delta = torch.as_tensor(
        max(float(huber_delta), 1e-12),
        dtype=visual_norms.dtype,
        device=visual_norms.device,
    )
    visual_costs = torch.where(
        visual_norms <= delta,
        0.5 * visual_norms.square(),
        delta * visual_norms - 0.5 * delta.square(),
    )
    return base + visual_costs.sum()


def _linearize_joint_autograd(
    state_i: NavigationState,
    state_j: NavigationState,
    prior: SquareRootPrior,
    imu: ImuPreintegrationFactor,
    visual: VisualFactor,
    covariance_eigenvalue_floor: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    zero = torch.zeros(
        PAIR_DOF,
        dtype=state_i.pose_WB.dtype,
        device=state_i.pose_WB.device,
        requires_grad=True,
    )

    def residual_from_increment(increment: torch.Tensor) -> torch.Tensor:
        candidate_i = retract_state(state_i, increment[0:STATE_DOF])
        candidate_j = retract_state(state_j, increment[STATE_DOF:PAIR_DOF])
        residual, _ = _factor_residuals(
            candidate_i,
            candidate_j,
            prior,
            imu,
            visual,
            covariance_eigenvalue_floor=covariance_eigenvalue_floor,
            robust_visual=True,
        )
        return residual

    residual = residual_from_increment(zero)
    jacobian = torch.autograd.functional.jacobian(
        residual_from_increment,
        zero,
        create_graph=False,
        vectorize=True,
    )
    hessian = jacobian.mT @ jacobian
    gradient = jacobian.mT @ residual.detach()
    _, blocks = _factor_residuals(
        state_i,
        state_j,
        prior,
        imu,
        visual,
        covariance_eigenvalue_floor=covariance_eigenvalue_floor,
        robust_visual=False,
    )
    cost = _true_cost(blocks, visual.huber_delta)
    return residual.detach(), jacobian.detach(), hessian.detach(), gradient.detach(), blocks


def _linearize_blockwise_autograd(
    state_i: NavigationState,
    state_j: NavigationState,
    prior: SquareRootPrior,
    imu: ImuPreintegrationFactor,
    visual: VisualFactor,
    covariance_eigenvalue_floor: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Linearize independent factors separately and assemble one normal equation.

    The prior, IMU and compressed visual factors retain their existing PyPose
    reverse-mode derivatives. Bias random walk is exactly linear in the local
    additive bias increments, so its Jacobian is assembled analytically.
    """

    zero = torch.zeros(
        PAIR_DOF,
        dtype=state_i.pose_WB.dtype,
        device=state_i.pose_WB.device,
        requires_grad=True,
    )

    def candidate_states(increment: torch.Tensor) -> tuple[NavigationState, NavigationState]:
        return (
            retract_state(state_i, increment[0:STATE_DOF]),
            retract_state(state_j, increment[STATE_DOF:PAIR_DOF]),
        )

    def prior_from_increment(increment: torch.Tensor) -> torch.Tensor:
        candidate_i, _ = candidate_states(increment)
        return _prior_residual(candidate_i, prior)

    def imu_from_increment(increment: torch.Tensor) -> torch.Tensor:
        candidate_i, candidate_j = candidate_states(increment)
        return _imu_residual(
            candidate_i,
            candidate_j,
            imu,
            covariance_eigenvalue_floor,
        )

    def visual_from_increment(increment: torch.Tensor) -> torch.Tensor:
        candidate_i, candidate_j = candidate_states(increment)
        residual, _ = _robust_visual_residual(
            candidate_i,
            candidate_j,
            visual,
            covariance_eigenvalue_floor,
        )
        return residual

    prior_r = prior_from_increment(zero)
    prior_j = torch.autograd.functional.jacobian(
        prior_from_increment,
        zero,
        create_graph=False,
        vectorize=True,
    )
    imu_r = imu_from_increment(zero)
    imu_j = torch.autograd.functional.jacobian(
        imu_from_increment,
        zero,
        create_graph=False,
        vectorize=True,
    )
    visual_r = visual_from_increment(zero)
    visual_j = torch.autograd.functional.jacobian(
        visual_from_increment,
        zero,
        create_graph=False,
        vectorize=True,
    )

    bias_raw = vio_bias_random_walk_residual(
        prev_acc_bias=state_i.acc_bias,
        prev_gyro_bias=state_i.gyro_bias,
        curr_acc_bias=state_j.acc_bias,
        curr_gyro_bias=state_j.gyro_bias,
    ).reshape(6)
    bias_lower = _covariance_cholesky(
        imu.bias_rw_covariance, covariance_eigenvalue_floor
    )
    bias_r = torch.linalg.solve_triangular(
        bias_lower,
        bias_raw.reshape(6, 1).to(bias_lower),
        upper=False,
    ).reshape(6)
    bias_raw_j = torch.zeros(
        (6, PAIR_DOF),
        dtype=state_i.pose_WB.dtype,
        device=state_i.pose_WB.device,
    )
    identity = torch.eye(3, dtype=bias_raw_j.dtype, device=bias_raw_j.device)
    bias_raw_j[0:3, 9:12] = -identity
    bias_raw_j[0:3, STATE_DOF + 9 : STATE_DOF + 12] = identity
    bias_raw_j[3:6, 12:15] = -identity
    bias_raw_j[3:6, STATE_DOF + 12 : STATE_DOF + 15] = identity
    bias_j = torch.linalg.solve_triangular(
        bias_lower,
        bias_raw_j.to(bias_lower),
        upper=False,
    )

    residual = torch.cat([prior_r, imu_r, bias_r, visual_r])
    jacobian = torch.cat([prior_j, imu_j, bias_j, visual_j], dim=0)
    hessian = jacobian.mT @ jacobian
    gradient = jacobian.mT @ residual.detach()
    _, blocks = _factor_residuals(
        state_i,
        state_j,
        prior,
        imu,
        visual,
        covariance_eigenvalue_floor=covariance_eigenvalue_floor,
        robust_visual=False,
    )
    return (
        residual.detach(),
        jacobian.detach(),
        hessian.detach(),
        gradient.detach(),
        blocks,
    )


def _joint_residual_from_increment(
    increment: torch.Tensor,
    state_i: NavigationState,
    state_j: NavigationState,
    prior: SquareRootPrior,
    imu: ImuPreintegrationFactor,
    visual: VisualFactor,
    covariance_eigenvalue_floor: float,
) -> torch.Tensor:
    candidate_i = retract_state(state_i, increment[0:STATE_DOF])
    candidate_j = retract_state(state_j, increment[STATE_DOF:PAIR_DOF])
    residual, _ = _factor_residuals(
        candidate_i,
        candidate_j,
        prior,
        imu,
        visual,
        covariance_eigenvalue_floor=covariance_eigenvalue_floor,
        robust_visual=True,
    )
    return residual


_joint_residual_jacrev = pp.func.jacrev(
    _joint_residual_from_increment,
    argnums=0,
)


def _linearize_func_jacrev(
    state_i: NavigationState,
    state_j: NavigationState,
    prior: SquareRootPrior,
    imu: ImuPreintegrationFactor,
    visual: VisualFactor,
    covariance_eigenvalue_floor: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Joint reverse-mode linearization using PyPose's Lie-aware jacrev."""

    zero = torch.zeros(
        PAIR_DOF,
        dtype=state_i.pose_WB.dtype,
        device=state_i.pose_WB.device,
    )

    residual = _joint_residual_from_increment(
        zero,
        state_i,
        state_j,
        prior,
        imu,
        visual,
        covariance_eigenvalue_floor,
    )
    jacobian = _joint_residual_jacrev(
        zero,
        state_i,
        state_j,
        prior,
        imu,
        visual,
        covariance_eigenvalue_floor,
    )
    hessian = jacobian.mT @ jacobian
    gradient = jacobian.mT @ residual.detach()
    _, blocks = _factor_residuals(
        state_i,
        state_j,
        prior,
        imu,
        visual,
        covariance_eigenvalue_floor=covariance_eigenvalue_floor,
        robust_visual=False,
    )
    return (
        residual.detach(),
        jacobian.detach(),
        hessian.detach(),
        gradient.detach(),
        blocks,
    )


def _linearize(
    state_i: NavigationState,
    state_j: NavigationState,
    prior: SquareRootPrior,
    imu: ImuPreintegrationFactor,
    visual: VisualFactor,
    covariance_eigenvalue_floor: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    return _linearize_joint_autograd(
        state_i,
        state_j,
        prior,
        imu,
        visual,
        covariance_eigenvalue_floor,
    )


def marginalize_source_state(
    optimized_i: NavigationState,
    optimized_j: NavigationState,
    hessian: torch.Tensor,
    gradient: torch.Tensor,
    *,
    eigenvalue_floor: float = 1e-10,
    active_state_mask: torch.Tensor | None = None,
) -> SquareRootPrior:
    hessian = 0.5 * (hessian + hessian.mT)
    if active_state_mask is None:
        active_state_mask = torch.ones(
            STATE_DOF, dtype=torch.bool, device=hessian.device
        )
    else:
        active_state_mask = active_state_mask.reshape(STATE_DOF).to(
            device=hessian.device, dtype=torch.bool
        )
    active_local = torch.nonzero(active_state_mask, as_tuple=False).reshape(-1)
    if active_local.numel() == 0:
        raise ValueError("at least one state degree of freedom must remain active")
    active_i = active_local
    active_j = active_local + STATE_DOF
    h_ii = hessian[active_i][:, active_i]
    h_ij = hessian[active_i][:, active_j]
    h_ji = hessian[active_j][:, active_i]
    h_jj = hessian[active_j][:, active_j]
    g_i = gradient[active_i]
    g_j = gradient[active_j]

    values_i, vectors_i = torch.linalg.eigh(0.5 * (h_ii + h_ii.mT))
    scale_i = max(float(values_i.abs().max().detach().cpu().item()), 1.0)
    floor_i = max(float(eigenvalue_floor), torch.finfo(hessian.dtype).eps * scale_i)
    h_ii_inverse = vectors_i @ torch.diag(values_i.clamp_min(floor_i).reciprocal()) @ vectors_i.mT

    prior_hessian = h_jj - h_ji @ h_ii_inverse @ h_ij
    prior_gradient = g_j - h_ji @ h_ii_inverse @ g_i
    prior_hessian = 0.5 * (prior_hessian + prior_hessian.mT)

    values, vectors = torch.linalg.eigh(prior_hessian)
    scale = max(float(values.abs().max().detach().cpu().item()), 1.0)
    threshold = max(float(eigenvalue_floor), torch.finfo(hessian.dtype).eps * scale)
    positive = values > threshold
    if not bool(positive.any()):
        raise RuntimeError("marginalization produced an empty prior")
    active_values = values[positive]
    active_vectors = vectors[:, positive]
    sqrt_information_active = torch.diag(active_values.sqrt()) @ active_vectors.mT
    sqrt_information = torch.zeros(
        (sqrt_information_active.shape[0], STATE_DOF),
        dtype=hessian.dtype,
        device=hessian.device,
    )
    sqrt_information[:, active_local] = sqrt_information_active
    residual_offset = torch.diag(active_values.rsqrt()) @ active_vectors.mT @ prior_gradient
    return SquareRootPrior(
        reference=optimized_j.detach(),
        sqrt_information=sqrt_information.detach(),
        residual_offset=residual_offset.detach(),
    )


class TwoStateVIOSolver:
    """Levenberg-Marquardt solver for a two-state fixed-lag VIO problem."""

    def __init__(
        self,
        *,
        max_iterations: int = 20,
        initial_damping: float = 1e-3,
        step_tolerance: float = 1e-8,
        cost_tolerance: float = 1e-10,
        covariance_eigenvalue_floor: float = 1e-12,
        marginalization_eigenvalue_floor: float = 1e-10,
    ) -> None:
        self.max_iterations = max(1, int(max_iterations))
        self.initial_damping = max(float(initial_damping), 1e-12)
        self.step_tolerance = max(float(step_tolerance), 0.0)
        self.cost_tolerance = max(float(cost_tolerance), 0.0)
        self.covariance_eigenvalue_floor = max(float(covariance_eigenvalue_floor), 0.0)
        self.marginalization_eigenvalue_floor = max(float(marginalization_eigenvalue_floor), 0.0)

    def solve(self, problem: TwoStateVIOProblem) -> TwoStateVIOResult:
        dtype = torch.float64
        device = problem.state_i.pose_WB.device
        state_i = problem.state_i.to(device=device, dtype=dtype)
        state_j = problem.state_j.to(device=device, dtype=dtype)
        prior = problem.prior_i.to(device=device, dtype=dtype)
        imu = problem.imu.to(device=device, dtype=dtype)
        visual = problem.visual_pose.to(device=device, dtype=dtype)
        active_state_mask = torch.ones(STATE_DOF, dtype=torch.bool, device=device)
        active_state_mask[9:12] = bool(problem.optimize_acc_bias)
        active_state_mask[12:15] = bool(problem.optimize_gyro_bias)
        active_pair_mask = torch.cat([active_state_mask, active_state_mask], dim=0)
        active_pair_indices = torch.nonzero(active_pair_mask, as_tuple=False).reshape(-1)

        _, blocks = _factor_residuals(
            state_i,
            state_j,
            prior,
            imu,
            visual,
            covariance_eigenvalue_floor=self.covariance_eigenvalue_floor,
            robust_visual=False,
        )
        initial_cost = float(_true_cost(blocks, visual.huber_delta).detach().cpu().item())
        current_cost = initial_cost
        damping = self.initial_damping
        converged = False
        iterations = 0
        final_step_norm = float("inf")
        convergence_reason = "iteration_limit"
        accepted_steps = 0
        rejected_steps = 0
        linearization_current = False

        for iteration in range(self.max_iterations):
            iterations = iteration + 1
            if not linearization_current:
                _, _, hessian, gradient, blocks = _linearize(
                    state_i,
                    state_j,
                    prior,
                    imu,
                    visual,
                    self.covariance_eigenvalue_floor,
                )
                linearization_current = True
            active_hessian = hessian[active_pair_indices][:, active_pair_indices]
            active_gradient = gradient[active_pair_indices]
            diagonal = active_hessian.diagonal().abs().clamp_min(1.0)
            system = active_hessian + damping * torch.diag(diagonal)
            try:
                active_step = torch.linalg.solve(system, -active_gradient)
            except torch.linalg.LinAlgError:
                active_step = torch.linalg.pinv(system) @ (-active_gradient)
            step = torch.zeros(PAIR_DOF, dtype=dtype, device=device)
            step[active_pair_indices] = active_step

            if not bool(torch.isfinite(step).all()):
                raise FloatingPointError("two-state LM produced a non-finite step")
            final_step_norm = float(torch.linalg.vector_norm(step).detach().cpu().item())
            if final_step_norm <= self.step_tolerance:
                converged = True
                convergence_reason = "step_tolerance"
                break

            candidate_i = retract_state(state_i, step[:STATE_DOF])
            candidate_j = retract_state(state_j, step[STATE_DOF:])
            _, candidate_blocks = _factor_residuals(
                candidate_i,
                candidate_j,
                prior,
                imu,
                visual,
                covariance_eigenvalue_floor=self.covariance_eigenvalue_floor,
                robust_visual=False,
            )
            candidate_cost = float(_true_cost(candidate_blocks, visual.huber_delta).detach().cpu().item())

            if candidate_cost < current_cost:
                accepted_steps += 1
                previous_cost = current_cost
                state_i = candidate_i.detach()
                state_j = candidate_j.detach()
                current_cost = candidate_cost
                linearization_current = False
                damping = max(damping * 0.25, 1e-12)
                if abs(previous_cost - current_cost) <= self.cost_tolerance:
                    converged = True
                    convergence_reason = "cost_tolerance"
                    break
            else:
                rejected_steps += 1
                damping = min(damping * 10.0, 1e12)

        if not linearization_current:
            _, _, hessian, gradient, blocks = _linearize(
                state_i,
                state_j,
                prior,
                imu,
                visual,
                self.covariance_eigenvalue_floor,
            )
        prior_j = marginalize_source_state(
            state_i,
            state_j,
            hessian,
            gradient,
            eigenvalue_floor=self.marginalization_eigenvalue_floor,
            active_state_mask=active_state_mask,
        )
        visual_norms = blocks["visual_group_norms"]
        final_gradient_inf_norm = float(
            gradient[active_pair_indices].abs().max().detach().cpu().item()
        )
        delta = max(float(visual.huber_delta), 1e-12)
        visual_cost = torch.where(
            visual_norms <= delta,
            0.5 * visual_norms.square(),
            delta * visual_norms - 0.5 * delta * delta,
        ).sum()

        return TwoStateVIOResult(
            state_i=state_i.detach(),
            state_j=state_j.detach(),
            prior_j=prior_j,
            converged=converged,
            iterations=iterations,
            initial_cost=initial_cost,
            final_cost=float(_true_cost(blocks, visual.huber_delta).detach().cpu().item()),
            prior_cost=0.5 * float(blocks["prior"].square().sum().detach().cpu().item()),
            imu_cost=0.5 * float(blocks["imu"].square().sum().detach().cpu().item()),
            bias_cost=0.5 * float(blocks["bias"].square().sum().detach().cpu().item()),
            visual_pose_cost=float(visual_cost.detach().cpu().item()),
            hessian=hessian.detach(),
            gradient=gradient.detach(),
            final_step_norm=final_step_norm,
            final_gradient_inf_norm=final_gradient_inf_norm,
            convergence_reason=convergence_reason,
            accepted_steps=accepted_steps,
            rejected_steps=rejected_steps,
        )
