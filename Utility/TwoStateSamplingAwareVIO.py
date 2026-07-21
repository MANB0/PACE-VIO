from __future__ import annotations

from dataclasses import dataclass, replace

import pypose as pp
import torch

from Utility.IMUKinematics import (
    vio_bias_random_walk_residual,
    vio_preintegrated_imu_residual,
)
from Utility.TwoStateVIO import (
    STATE_DOF,
    ImuPreintegrationFactor,
    NavigationState,
    VisualFactor,
    _true_cost,
    _whiten,
    retract_state,
    state_boxminus,
    visual_whitened_residuals,
)


@dataclass(frozen=True)
class CrossEdgeImuFactor:
    """IMU factor conditioned on standardized endpoint raw-sample noise."""

    base: ImuPreintegrationFactor
    unique_covariance: torch.Tensor
    incoming_raw_time_ns: torch.Tensor
    outgoing_raw_time_ns: torch.Tensor
    incoming_sensitivity: torch.Tensor
    outgoing_sensitivity: torch.Tensor

    def to(self, *, device: torch.device, dtype: torch.dtype) -> "CrossEdgeImuFactor":
        incoming_times = self.incoming_raw_time_ns.reshape(-1).to(
            device=device, dtype=torch.long
        )
        outgoing_times = self.outgoing_raw_time_ns.reshape(-1).to(
            device=device, dtype=torch.long
        )
        return CrossEdgeImuFactor(
            base=self.base.to(device=device, dtype=dtype),
            unique_covariance=self.unique_covariance.reshape(9, 9).to(
                device=device, dtype=dtype
            ),
            incoming_raw_time_ns=incoming_times,
            outgoing_raw_time_ns=outgoing_times,
            incoming_sensitivity=self.incoming_sensitivity.reshape(
                9, incoming_times.numel() * 6
            ).to(device=device, dtype=dtype),
            outgoing_sensitivity=self.outgoing_sensitivity.reshape(
                9, outgoing_times.numel() * 6
            ).to(device=device, dtype=dtype),
        )

    @property
    def incoming_dof(self) -> int:
        return int(self.incoming_raw_time_ns.numel()) * 6

    @property
    def outgoing_dof(self) -> int:
        return int(self.outgoing_raw_time_ns.numel()) * 6


@dataclass(frozen=True)
class CrossEdgeSquareRootPrior:
    """History prior over one navigation state and shared raw-sample noise."""

    reference_state: NavigationState
    reference_noise: torch.Tensor
    raw_time_ns: torch.Tensor
    sqrt_information: torch.Tensor
    residual_offset: torch.Tensor
    retained_measurement_sensitivity: torch.Tensor | None = None

    def to(
        self, *, device: torch.device, dtype: torch.dtype
    ) -> "CrossEdgeSquareRootPrior":
        noise = self.reference_noise.reshape(-1).to(device=device, dtype=dtype)
        retained = self.retained_measurement_sensitivity
        return CrossEdgeSquareRootPrior(
            reference_state=self.reference_state.to(device=device, dtype=dtype),
            reference_noise=noise,
            raw_time_ns=self.raw_time_ns.reshape(-1).to(device=device, dtype=torch.long),
            sqrt_information=self.sqrt_information.to(device=device, dtype=dtype),
            residual_offset=self.residual_offset.reshape(-1).to(
                device=device, dtype=dtype
            ),
            retained_measurement_sensitivity=(
                None
                if retained is None
                else retained.reshape(9, noise.numel()).to(device=device, dtype=dtype)
            ),
        )


@dataclass(frozen=True)
class CrossEdgeTwoStateProblem:
    state_i: NavigationState
    state_j: NavigationState
    noise_i: torch.Tensor
    noise_j: torch.Tensor
    prior_i: CrossEdgeSquareRootPrior
    imu: CrossEdgeImuFactor
    visual: VisualFactor
    optimize_acc_bias: bool = True
    optimize_gyro_bias: bool = True


@dataclass(frozen=True)
class SymmetricMatrixDiagnostics:
    min_eigenvalue: float
    max_eigenvalue: float
    effective_rank: int
    dimension: int
    condition_number: float
    threshold: float


@dataclass(frozen=True)
class CrossEdgeMarginalizationDiagnostics:
    h_mm: SymmetricMatrixDiagnostics
    schur_prior: SymmetricMatrixDiagnostics
    discarded_h_mm_dimensions: int
    discarded_prior_dimensions: int
    quadratic_relative_error: float


@dataclass(frozen=True)
class CrossEdgeTwoStateResult:
    state_i: NavigationState
    state_j: NavigationState
    noise_i: torch.Tensor
    noise_j: torch.Tensor
    prior_j: CrossEdgeSquareRootPrior
    converged: bool
    iterations: int
    initial_cost: float
    final_cost: float
    prior_cost: float
    imu_cost: float
    bias_cost: float
    visual_pose_cost: float
    sampling_noise_cost: float
    hessian: torch.Tensor
    gradient: torch.Tensor
    final_step_norm: float
    final_gradient_inf_norm: float
    convergence_reason: str
    accepted_steps: int
    rejected_steps: int
    cross_covariance_frobenius_norm: float
    unique_covariance_diagnostics: SymmetricMatrixDiagnostics
    incoming_prior_diagnostics: SymmetricMatrixDiagnostics
    marginalization_diagnostics: CrossEdgeMarginalizationDiagnostics
    state_i_increment: torch.Tensor
    state_j_increment: torch.Tensor
    common_translation_update_world: torch.Tensor
    differential_translation_update_world: torch.Tensor
    rank_aware_imu_whitening: bool
    rank_aware_fallback_active: bool
    rank_aware_imu_residual_dimension: int


def symmetric_matrix_diagnostics(
    matrix: torch.Tensor,
    *,
    eigenvalue_floor: float,
) -> SymmetricMatrixDiagnostics:
    symmetric = 0.5 * (matrix + matrix.mT)
    values = torch.linalg.eigvalsh(symmetric)
    max_abs = max(float(values.abs().max().detach().cpu().item()), 1.0)
    threshold = max(
        float(eigenvalue_floor),
        torch.finfo(symmetric.dtype).eps * max_abs,
    )
    active = values > threshold
    rank = int(active.sum().detach().cpu().item())
    if rank:
        active_values = values[active]
        condition = float(
            (active_values.max() / active_values.min()).detach().cpu().item()
        )
    else:
        condition = float("inf")
    return SymmetricMatrixDiagnostics(
        min_eigenvalue=float(values.min().detach().cpu().item()),
        max_eigenvalue=float(values.max().detach().cpu().item()),
        effective_rank=rank,
        dimension=int(values.numel()),
        condition_number=condition,
        threshold=threshold,
    )


def cross_edge_prior_hessian(prior: CrossEdgeSquareRootPrior) -> torch.Tensor:
    return prior.sqrt_information.mT @ prior.sqrt_information


def make_cross_edge_diagonal_prior(
    reference: NavigationState,
    incoming_raw_time_ns: torch.Tensor,
    *,
    pose_translation_std: float,
    pose_rotation_std: float,
    velocity_std: float,
    acc_bias_std: float,
    gyro_bias_std: float,
) -> CrossEdgeSquareRootPrior:
    raw_time_ns = incoming_raw_time_ns.reshape(-1).to(dtype=torch.long)
    noise_dof = int(raw_time_ns.numel()) * 6
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
    sqrt_information = torch.zeros(
        (STATE_DOF + noise_dof, STATE_DOF + noise_dof),
        dtype=std.dtype,
        device=std.device,
    )
    sqrt_information[:STATE_DOF, :STATE_DOF] = torch.diag(std.reciprocal())
    sqrt_information[STATE_DOF:, STATE_DOF:] = torch.eye(
        noise_dof, dtype=std.dtype, device=std.device
    )
    return CrossEdgeSquareRootPrior(
        reference_state=reference.detach(),
        reference_noise=torch.zeros(noise_dof, dtype=std.dtype, device=std.device),
        raw_time_ns=raw_time_ns.detach().clone(),
        sqrt_information=sqrt_information,
        residual_offset=torch.zeros(
            STATE_DOF + noise_dof, dtype=std.dtype, device=std.device
        ),
    )


def _raw_imu_residual(
    state_i: NavigationState,
    state_j: NavigationState,
    imu: ImuPreintegrationFactor,
) -> torch.Tensor:
    return vio_preintegrated_imu_residual(
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


def cross_edge_factor_residuals(
    state_i: NavigationState,
    state_j: NavigationState,
    noise_i: torch.Tensor,
    noise_j: torch.Tensor,
    prior: CrossEdgeSquareRootPrior,
    imu: CrossEdgeImuFactor,
    visual: VisualFactor,
    *,
    covariance_eigenvalue_floor: float,
    robust_visual: bool,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    prior_delta = torch.cat(
        [
            state_boxminus(state_i, prior.reference_state),
            noise_i.reshape(-1) - prior.reference_noise.reshape(-1),
        ]
    )
    prior_r = prior.sqrt_information @ prior_delta + prior.residual_offset

    imu_raw = _raw_imu_residual(state_i, state_j, imu.base)
    conditioned_imu_raw = (
        imu_raw
        + imu.incoming_sensitivity @ noise_i.reshape(-1)
        + imu.outgoing_sensitivity @ noise_j.reshape(-1)
    )
    imu_r = _whiten(
        conditioned_imu_raw,
        imu.unique_covariance,
        covariance_eigenvalue_floor,
    )

    bias_raw = vio_bias_random_walk_residual(
        prev_acc_bias=state_i.acc_bias,
        prev_gyro_bias=state_i.gyro_bias,
        curr_acc_bias=state_j.acc_bias,
        curr_gyro_bias=state_j.gyro_bias,
    ).reshape(6)
    bias_r = _whiten(
        bias_raw,
        imu.base.bias_rw_covariance,
        covariance_eigenvalue_floor,
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
        visual_r = (weight.sqrt().unsqueeze(-1) * visual_white_rows).reshape(-1)
    else:
        visual_r = visual_white_rows.reshape(-1)

    outgoing_noise_r = noise_j.reshape(-1)
    blocks = {
        "prior": prior_r,
        "imu": imu_r,
        "bias": bias_r,
        "visual_pose": visual_r,
        "visual_pose_unweighted": visual_white_rows.reshape(-1),
        "visual_group_norms": torch.linalg.vector_norm(visual_white_rows, dim=-1),
        "sampling_noise": outgoing_noise_r,
        "conditioned_imu_raw": conditioned_imu_raw,
    }
    return (
        torch.cat([prior_r, imu_r, bias_r, visual_r, outgoing_noise_r]),
        blocks,
    )


def _cross_edge_true_cost(
    blocks: dict[str, torch.Tensor], huber_delta: float
) -> torch.Tensor:
    return _true_cost(blocks, huber_delta) + 0.5 * blocks[
        "sampling_noise"
    ].square().sum()


def cross_edge_linearize(
    state_i: NavigationState,
    state_j: NavigationState,
    noise_i: torch.Tensor,
    noise_j: torch.Tensor,
    prior: CrossEdgeSquareRootPrior,
    imu: CrossEdgeImuFactor,
    visual: VisualFactor,
    covariance_eigenvalue_floor: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    incoming_dof = int(noise_i.numel())
    outgoing_dof = int(noise_j.numel())
    pair_dof = 2 * STATE_DOF + incoming_dof + outgoing_dof
    zero = torch.zeros(
        pair_dof,
        dtype=state_i.pose_WB.dtype,
        device=state_i.pose_WB.device,
        requires_grad=True,
    )

    def residual_from_increment(increment: torch.Tensor) -> torch.Tensor:
        candidate_i = retract_state(state_i, increment[:STATE_DOF])
        candidate_j = retract_state(
            state_j, increment[STATE_DOF : 2 * STATE_DOF]
        )
        incoming_start = 2 * STATE_DOF
        outgoing_start = incoming_start + incoming_dof
        candidate_noise_i = noise_i + increment[
            incoming_start:outgoing_start
        ]
        candidate_noise_j = noise_j + increment[outgoing_start:]
        residual, _ = cross_edge_factor_residuals(
            candidate_i,
            candidate_j,
            candidate_noise_i,
            candidate_noise_j,
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
    _, blocks = cross_edge_factor_residuals(
        state_i,
        state_j,
        noise_i,
        noise_j,
        prior,
        imu,
        visual,
        covariance_eigenvalue_floor=covariance_eigenvalue_floor,
        robust_visual=False,
    )
    return residual.detach(), jacobian.detach(), hessian.detach(), gradient.detach(), blocks


def marginalize_cross_edge_source(
    optimized_i: NavigationState,
    optimized_j: NavigationState,
    optimized_noise_j: torch.Tensor,
    outgoing_raw_time_ns: torch.Tensor,
    outgoing_sensitivity: torch.Tensor,
    hessian: torch.Tensor,
    gradient: torch.Tensor,
    incoming_dof: int,
    *,
    eigenvalue_floor: float = 1e-10,
    active_state_mask: torch.Tensor | None = None,
) -> tuple[CrossEdgeSquareRootPrior, CrossEdgeMarginalizationDiagnostics]:
    hessian = 0.5 * (hessian + hessian.mT)
    outgoing_dof = int(optimized_noise_j.numel())
    if active_state_mask is None:
        active_state_mask = torch.ones(
            STATE_DOF, dtype=torch.bool, device=hessian.device
        )
    else:
        active_state_mask = active_state_mask.reshape(STATE_DOF).to(
            device=hessian.device, dtype=torch.bool
        )
    active_local = torch.nonzero(active_state_mask, as_tuple=False).reshape(-1)
    incoming_start = 2 * STATE_DOF
    outgoing_start = incoming_start + int(incoming_dof)
    marginal_indices = torch.cat(
        [
            active_local,
            torch.arange(
                incoming_start,
                outgoing_start,
                device=hessian.device,
            ),
        ]
    )
    retained_indices = torch.cat(
        [
            active_local + STATE_DOF,
            torch.arange(
                outgoing_start,
                outgoing_start + outgoing_dof,
                device=hessian.device,
            ),
        ]
    )
    h_mm = hessian[marginal_indices][:, marginal_indices]
    h_mr = hessian[marginal_indices][:, retained_indices]
    h_rm = hessian[retained_indices][:, marginal_indices]
    h_rr = hessian[retained_indices][:, retained_indices]
    g_m = gradient[marginal_indices]
    g_r = gradient[retained_indices]

    values_m, vectors_m = torch.linalg.eigh(0.5 * (h_mm + h_mm.mT))
    scale_m = max(float(values_m.abs().max().detach().cpu().item()), 1.0)
    threshold_m = max(
        float(eigenvalue_floor), torch.finfo(hessian.dtype).eps * scale_m
    )
    inverse_values = torch.where(
        values_m > threshold_m,
        values_m.reciprocal(),
        torch.zeros_like(values_m),
    )
    h_mm_inverse = vectors_m @ torch.diag(inverse_values) @ vectors_m.mT

    prior_hessian = h_rr - h_rm @ h_mm_inverse @ h_mr
    prior_gradient = g_r - h_rm @ h_mm_inverse @ g_m
    prior_hessian = 0.5 * (prior_hessian + prior_hessian.mT)
    values, vectors = torch.linalg.eigh(prior_hessian)
    scale = max(float(values.abs().max().detach().cpu().item()), 1.0)
    threshold = max(
        float(eigenvalue_floor), torch.finfo(hessian.dtype).eps * scale
    )
    positive = values > threshold
    if not bool(positive.any()):
        raise RuntimeError("cross-edge marginalization produced an empty prior")
    active_values = values[positive]
    active_vectors = vectors[:, positive]
    sqrt_information_active = torch.diag(active_values.sqrt()) @ active_vectors.mT
    retained_dof = STATE_DOF + outgoing_dof
    sqrt_information = torch.zeros(
        (sqrt_information_active.shape[0], retained_dof),
        dtype=hessian.dtype,
        device=hessian.device,
    )
    sqrt_information[:, active_local] = sqrt_information_active[
        :, : active_local.numel()
    ]
    sqrt_information[:, STATE_DOF:] = sqrt_information_active[
        :, active_local.numel() :
    ]
    residual_offset = (
        torch.diag(active_values.rsqrt()) @ active_vectors.mT @ prior_gradient
    )
    prior = CrossEdgeSquareRootPrior(
        reference_state=optimized_j.detach(),
        reference_noise=optimized_noise_j.detach().clone(),
        raw_time_ns=outgoing_raw_time_ns.detach().clone(),
        sqrt_information=sqrt_information.detach(),
        residual_offset=residual_offset.detach(),
        retained_measurement_sensitivity=outgoing_sensitivity.detach().clone(),
    )

    # Verify the Schur quadratic independently at a deterministic retained
    # increment. This catches index/order mistakes that a symmetry/SPD check
    # cannot detect.
    retained_probe = torch.linspace(
        -0.25,
        0.25,
        retained_indices.numel(),
        dtype=hessian.dtype,
        device=hessian.device,
    )
    marginal_probe = -h_mm_inverse @ (h_mr @ retained_probe + g_m)
    full_probe = torch.cat([marginal_probe, retained_probe])
    ordered_hessian = torch.cat(
        [
            torch.cat([h_mm, h_mr], dim=1),
            torch.cat([h_rm, h_rr], dim=1),
        ],
        dim=0,
    )
    ordered_gradient = torch.cat([g_m, g_r])
    full_hessian_term = 0.5 * full_probe @ ordered_hessian @ full_probe
    full_gradient_term = ordered_gradient @ full_probe
    full_quadratic = full_hessian_term + full_gradient_term
    reduced_hessian_term = 0.5 * retained_probe @ prior_hessian @ retained_probe
    reduced_gradient_term = prior_gradient @ retained_probe
    reduced_constant = -0.5 * g_m @ h_mm_inverse @ g_m
    reduced_quadratic = (
        reduced_hessian_term + reduced_gradient_term + reduced_constant
    )
    quadratic_scale = max(
        abs(float(full_hessian_term.detach().cpu().item())),
        abs(float(full_gradient_term.detach().cpu().item())),
        abs(float(reduced_hessian_term.detach().cpu().item())),
        abs(float(reduced_gradient_term.detach().cpu().item())),
        abs(float(reduced_constant.detach().cpu().item())),
        1.0,
    )
    quadratic_relative_error = abs(
        float((full_quadratic - reduced_quadratic).detach().cpu().item())
    ) / quadratic_scale
    diagnostics = CrossEdgeMarginalizationDiagnostics(
        h_mm=symmetric_matrix_diagnostics(
            h_mm, eigenvalue_floor=eigenvalue_floor
        ),
        schur_prior=symmetric_matrix_diagnostics(
            prior_hessian, eigenvalue_floor=eigenvalue_floor
        ),
        discarded_h_mm_dimensions=int((~(values_m > threshold_m)).sum().item()),
        discarded_prior_dimensions=int((~positive).sum().item()),
        quadratic_relative_error=quadratic_relative_error,
    )
    return prior, diagnostics


class CrossEdgeTwoStateSolver:
    """LM solver that carries adjacent-edge raw-sample correlation in its prior."""

    def __init__(
        self,
        *,
        max_iterations: int = 20,
        initial_damping: float = 1e-3,
        step_tolerance: float = 1e-8,
        cost_tolerance: float = 1e-10,
        covariance_eigenvalue_floor: float = 1e-12,
        marginalization_eigenvalue_floor: float = 1e-10,
        rank_aware_imu_whitening: bool = False,
    ) -> None:
        self.max_iterations = max(1, int(max_iterations))
        self.initial_damping = max(float(initial_damping), 1e-12)
        self.step_tolerance = max(float(step_tolerance), 0.0)
        self.cost_tolerance = max(float(cost_tolerance), 0.0)
        self.covariance_eigenvalue_floor = max(
            float(covariance_eigenvalue_floor), 0.0
        )
        self.marginalization_eigenvalue_floor = max(
            float(marginalization_eigenvalue_floor), 0.0
        )
        self.rank_aware_imu_whitening = bool(rank_aware_imu_whitening)

    def solve(self, problem: CrossEdgeTwoStateProblem) -> CrossEdgeTwoStateResult:
        dtype = torch.float64
        device = problem.state_i.pose_WB.device
        state_i = problem.state_i.to(device=device, dtype=dtype)
        state_j = problem.state_j.to(device=device, dtype=dtype)
        prior = problem.prior_i.to(device=device, dtype=dtype)
        imu = problem.imu.to(device=device, dtype=dtype)
        visual = problem.visual.to(device=device, dtype=dtype)
        noise_i = problem.noise_i.reshape(-1).to(device=device, dtype=dtype)
        noise_j = problem.noise_j.reshape(-1).to(device=device, dtype=dtype)
        initial_state_i = state_i.detach()
        initial_state_j = state_j.detach()
        if not torch.equal(prior.raw_time_ns, imu.incoming_raw_time_ns):
            raise ValueError("SA-v2 prior/raw-sample timestamps are discontinuous")
        if prior.reference_noise.numel() != noise_i.numel():
            raise ValueError("SA-v2 incoming latent dimension does not match prior")
        if noise_i.numel() != imu.incoming_dof or noise_j.numel() != imu.outgoing_dof:
            raise ValueError("SA-v2 latent dimensions do not match endpoint supports")

        original_unique_covariance_diagnostics = symmetric_matrix_diagnostics(
            imu.unique_covariance,
            eigenvalue_floor=self.covariance_eigenvalue_floor,
        )
        rank_aware_fallback_active = bool(
            self.rank_aware_imu_whitening
            and original_unique_covariance_diagnostics.effective_rank
            < original_unique_covariance_diagnostics.dimension
        )
        if rank_aware_fallback_active:
            # A singular conditional covariance represents exact null-space
            # constraints. Hard flooring invents a finite but enormous weight;
            # pseudoinverse whitening would instead discard those constraints.
            # Use the complete per-edge covariance and break latent correlation
            # at this edge, which is the statistically valid SA-v1 fallback.
            imu = replace(
                imu,
                unique_covariance=imu.base.covariance,
                incoming_sensitivity=torch.zeros_like(imu.incoming_sensitivity),
                outgoing_sensitivity=torch.zeros_like(imu.outgoing_sensitivity),
            )

        active_state_mask = torch.ones(STATE_DOF, dtype=torch.bool, device=device)
        active_state_mask[9:12] = bool(problem.optimize_acc_bias)
        active_state_mask[12:15] = bool(problem.optimize_gyro_bias)
        active_local = torch.nonzero(active_state_mask, as_tuple=False).reshape(-1)
        incoming_start = 2 * STATE_DOF
        outgoing_start = incoming_start + noise_i.numel()
        active_indices = torch.cat(
            [
                active_local,
                active_local + STATE_DOF,
                torch.arange(incoming_start, outgoing_start, device=device),
                torch.arange(
                    outgoing_start,
                    outgoing_start + noise_j.numel(),
                    device=device,
                ),
            ]
        )
        pair_dof = 2 * STATE_DOF + noise_i.numel() + noise_j.numel()

        _, _, hessian, gradient, blocks = cross_edge_linearize(
            state_i,
            state_j,
            noise_i,
            noise_j,
            prior,
            imu,
            visual,
            self.covariance_eigenvalue_floor,
        )
        initial_cost = float(
            _cross_edge_true_cost(blocks, visual.huber_delta).detach().cpu().item()
        )
        current_cost = initial_cost
        damping = self.initial_damping
        converged = False
        iterations = 0
        final_step_norm = float("inf")
        convergence_reason = "iteration_limit"
        accepted_steps = 0
        rejected_steps = 0

        for iteration in range(self.max_iterations):
            iterations = iteration + 1
            _, _, hessian, gradient, _ = cross_edge_linearize(
                state_i,
                state_j,
                noise_i,
                noise_j,
                prior,
                imu,
                visual,
                self.covariance_eigenvalue_floor,
            )
            active_hessian = hessian[active_indices][:, active_indices]
            active_gradient = gradient[active_indices]
            diagonal = active_hessian.diagonal().abs().clamp_min(1.0)
            system = active_hessian + damping * torch.diag(diagonal)
            try:
                active_step = torch.linalg.solve(system, -active_gradient)
            except torch.linalg.LinAlgError:
                active_step = torch.linalg.pinv(system) @ (-active_gradient)
            step = torch.zeros(pair_dof, dtype=dtype, device=device)
            step[active_indices] = active_step
            if not bool(torch.isfinite(step).all()):
                raise FloatingPointError("SA-v2 LM produced a non-finite step")
            final_step_norm = float(
                torch.linalg.vector_norm(step).detach().cpu().item()
            )
            if final_step_norm <= self.step_tolerance:
                converged = True
                convergence_reason = "step_tolerance"
                break

            candidate_i = retract_state(state_i, step[:STATE_DOF])
            candidate_j = retract_state(
                state_j, step[STATE_DOF : 2 * STATE_DOF]
            )
            candidate_noise_i = noise_i + step[incoming_start:outgoing_start]
            candidate_noise_j = noise_j + step[outgoing_start:]
            _, candidate_blocks = cross_edge_factor_residuals(
                candidate_i,
                candidate_j,
                candidate_noise_i,
                candidate_noise_j,
                prior,
                imu,
                visual,
                covariance_eigenvalue_floor=self.covariance_eigenvalue_floor,
                robust_visual=False,
            )
            candidate_cost = float(
                _cross_edge_true_cost(candidate_blocks, visual.huber_delta)
                .detach()
                .cpu()
                .item()
            )
            if candidate_cost < current_cost:
                accepted_steps += 1
                previous_cost = current_cost
                state_i = candidate_i.detach()
                state_j = candidate_j.detach()
                noise_i = candidate_noise_i.detach()
                noise_j = candidate_noise_j.detach()
                current_cost = candidate_cost
                damping = max(damping * 0.25, 1e-12)
                if abs(previous_cost - current_cost) <= self.cost_tolerance:
                    converged = True
                    convergence_reason = "cost_tolerance"
                    break
            else:
                rejected_steps += 1
                damping = min(damping * 10.0, 1e12)

        _, _, hessian, gradient, blocks = cross_edge_linearize(
            state_i,
            state_j,
            noise_i,
            noise_j,
            prior,
            imu,
            visual,
            self.covariance_eigenvalue_floor,
        )
        prior_j, marginalization_diagnostics = marginalize_cross_edge_source(
            state_i,
            state_j,
            noise_j,
            imu.outgoing_raw_time_ns,
            imu.outgoing_sensitivity,
            hessian,
            gradient,
            int(noise_i.numel()),
            eigenvalue_floor=self.marginalization_eigenvalue_floor,
            active_state_mask=active_state_mask,
        )
        if (
            self.rank_aware_imu_whitening
            and marginalization_diagnostics.quadratic_relative_error > 1.0e-8
        ):
            raise FloatingPointError(
                "SA-v2 rank-aware marginalization failed its quadratic "
                "consistency check: relative_error="
                f"{marginalization_diagnostics.quadratic_relative_error:.3e}"
            )
        visual_norms = blocks["visual_group_norms"]
        delta = max(float(visual.huber_delta), 1e-12)
        visual_cost = torch.where(
            visual_norms <= delta,
            0.5 * visual_norms.square(),
            delta * visual_norms - 0.5 * delta * delta,
        ).sum()
        final_gradient_inf_norm = float(
            gradient[active_indices].abs().max().detach().cpu().item()
        )
        cross_covariance_norm = 0.0
        if prior.retained_measurement_sensitivity is not None:
            cross_covariance = (
                prior.retained_measurement_sensitivity
                @ imu.incoming_sensitivity.mT
            )
            cross_covariance_norm = float(
                torch.linalg.matrix_norm(cross_covariance).detach().cpu().item()
            )

        state_i_increment = state_boxminus(state_i, initial_state_i).detach()
        state_j_increment = state_boxminus(state_j, initial_state_j).detach()
        translation_i_before = pp.SE3(initial_state_i.pose_WB).translation().reshape(3)
        translation_j_before = pp.SE3(initial_state_j.pose_WB).translation().reshape(3)
        translation_i_after = pp.SE3(state_i.pose_WB).translation().reshape(3)
        translation_j_after = pp.SE3(state_j.pose_WB).translation().reshape(3)
        translation_update_i = translation_i_after - translation_i_before
        translation_update_j = translation_j_after - translation_j_before

        return CrossEdgeTwoStateResult(
            state_i=state_i.detach(),
            state_j=state_j.detach(),
            noise_i=noise_i.detach(),
            noise_j=noise_j.detach(),
            prior_j=prior_j,
            converged=converged,
            iterations=iterations,
            initial_cost=initial_cost,
            final_cost=float(
                _cross_edge_true_cost(blocks, visual.huber_delta)
                .detach()
                .cpu()
                .item()
            ),
            prior_cost=0.5
            * float(blocks["prior"].square().sum().detach().cpu().item()),
            imu_cost=0.5
            * float(blocks["imu"].square().sum().detach().cpu().item()),
            bias_cost=0.5
            * float(blocks["bias"].square().sum().detach().cpu().item()),
            visual_pose_cost=float(visual_cost.detach().cpu().item()),
            sampling_noise_cost=0.5
            * float(blocks["sampling_noise"].square().sum().detach().cpu().item()),
            hessian=hessian.detach(),
            gradient=gradient.detach(),
            final_step_norm=final_step_norm,
            final_gradient_inf_norm=final_gradient_inf_norm,
            convergence_reason=convergence_reason,
            accepted_steps=accepted_steps,
            rejected_steps=rejected_steps,
            cross_covariance_frobenius_norm=cross_covariance_norm,
            unique_covariance_diagnostics=symmetric_matrix_diagnostics(
                problem.imu.unique_covariance.to(device=device, dtype=dtype),
                eigenvalue_floor=self.covariance_eigenvalue_floor,
            ),
            incoming_prior_diagnostics=symmetric_matrix_diagnostics(
                cross_edge_prior_hessian(prior),
                eigenvalue_floor=self.marginalization_eigenvalue_floor,
            ),
            marginalization_diagnostics=marginalization_diagnostics,
            state_i_increment=state_i_increment,
            state_j_increment=state_j_increment,
            common_translation_update_world=(
                0.5 * (translation_update_i + translation_update_j)
            ).detach(),
            differential_translation_update_world=(
                translation_update_j - translation_update_i
            ).detach(),
            rank_aware_imu_whitening=self.rank_aware_imu_whitening,
            rank_aware_fallback_active=rank_aware_fallback_active,
            rank_aware_imu_residual_dimension=int(blocks["imu"].numel()),
        )
