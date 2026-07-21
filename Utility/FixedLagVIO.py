from __future__ import annotations

from dataclasses import dataclass

import pypose as pp
import torch

from Utility.IMUKinematics import vio_bias_random_walk_residual, vio_preintegrated_imu_residual
from Utility.TwoStateVIO import (
    STATE_DOF,
    ImuPreintegrationFactor,
    NavigationState,
    RelativePoseFactor,
    SquareRootPrior,
    TwoStateVIOProblem,
    _factor_residuals,
    _linearize,
    _whiten,
    marginalize_source_state,
    retract_state,
    state_boxminus,
)


@dataclass(frozen=True)
class FixedLagVIOProblem:
    states: tuple[NavigationState, ...]
    prior_first: SquareRootPrior
    imu_factors: tuple[ImuPreintegrationFactor, ...]
    visual_factors: tuple[RelativePoseFactor, ...]
    optimize_acc_bias: bool = True
    optimize_gyro_bias: bool = True
    shared_acc_bias: bool = False


@dataclass(frozen=True)
class FixedLagVIOResult:
    states: tuple[NavigationState, ...]
    prior_next: SquareRootPrior
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
    shared_acc_bias: bool
    acc_bias_marginal_information_eigenvalues: torch.Tensor | None


def _shared_acc_bias_projection(
    state_count: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    optimize_acc_bias: bool,
    optimize_gyro_bias: bool,
) -> torch.Tensor:
    """Map reduced window increments to full per-state 15-DoF increments.

    The reduced ordering is one ``[pose(6), velocity(3), gyro_bias(3)]``
    block per state followed by one shared accelerometer-bias block. When
    accelerometer bias is fixed, the final shared block is omitted.
    """
    if state_count < 2:
        raise ValueError("shared accelerometer bias requires at least two states")
    per_state_dof = 12 if optimize_gyro_bias else 9
    shared_dof = 3 if optimize_acc_bias else 0
    projection = torch.zeros(
        (STATE_DOF * state_count, per_state_dof * state_count + shared_dof),
        device=device,
        dtype=dtype,
    )
    for index in range(state_count):
        full = index * STATE_DOF
        reduced = index * per_state_dof
        projection[full : full + 9, reduced : reduced + 9] = torch.eye(
            9, device=device, dtype=dtype
        )
        if optimize_gyro_bias:
            projection[full + 12 : full + 15, reduced + 9 : reduced + 12] = torch.eye(
                3, device=device, dtype=dtype
            )
        if optimize_acc_bias:
            shared = per_state_dof * state_count
            projection[full + 9 : full + 12, shared : shared + 3] = torch.eye(
                3, device=device, dtype=dtype
            )
    return projection


def _marginal_information_block(
    hessian: torch.Tensor,
    retained_indices: torch.Tensor,
    *,
    eigenvalue_floor: float,
) -> torch.Tensor:
    """Return Schur-complement information for selected coordinates."""
    dimension = hessian.shape[0]
    retained_indices = retained_indices.reshape(-1).long().to(hessian.device)
    keep = torch.zeros(dimension, dtype=torch.bool, device=hessian.device)
    keep[retained_indices] = True
    nuisance_indices = torch.nonzero(~keep, as_tuple=False).reshape(-1)
    hessian = 0.5 * (hessian + hessian.mT)
    h_rr = hessian[retained_indices][:, retained_indices]
    if nuisance_indices.numel() == 0:
        return h_rr
    h_nn = hessian[nuisance_indices][:, nuisance_indices]
    h_nr = hessian[nuisance_indices][:, retained_indices]
    h_rn = hessian[retained_indices][:, nuisance_indices]
    values, vectors = torch.linalg.eigh(0.5 * (h_nn + h_nn.mT))
    scale = max(float(values.abs().max().detach().cpu().item()), 1.0)
    floor = max(float(eigenvalue_floor), torch.finfo(hessian.dtype).eps * scale)
    inverse = vectors @ torch.diag(values.clamp_min(floor).reciprocal()) @ vectors.mT
    return 0.5 * ((h_rr - h_rn @ inverse @ h_nr) + (h_rr - h_rn @ inverse @ h_nr).mT)


def marginalize_source_state_shared_acc_bias(
    optimized_i: NavigationState,
    optimized_j: NavigationState,
    hessian: torch.Tensor,
    gradient: torch.Tensor,
    *,
    optimize_gyro_bias: bool,
    eigenvalue_floor: float = 1e-10,
) -> SquareRootPrior:
    """Marginalize state i while retaining its shared ba as state j's ba.

    The two full ba increments are tied before the Schur complement. This is
    equivalent to a persistent, piecewise-constant accelerometer-bias state.
    """
    projection = _shared_acc_bias_projection(
        2,
        device=hessian.device,
        dtype=hessian.dtype,
        optimize_acc_bias=True,
        optimize_gyro_bias=optimize_gyro_bias,
    )
    reduced_hessian = projection.mT @ (0.5 * (hessian + hessian.mT)) @ projection
    reduced_gradient = projection.mT @ gradient
    per_state_dof = 12 if optimize_gyro_bias else 9
    source_dof = per_state_dof
    shared_slice = slice(2 * per_state_dof, 2 * per_state_dof + 3)
    source = torch.arange(source_dof, device=hessian.device)
    target_pose_velocity = torch.arange(
        per_state_dof, per_state_dof + 9, device=hessian.device
    )
    if optimize_gyro_bias:
        target_gyro = torch.arange(
            per_state_dof + 9, per_state_dof + 12, device=hessian.device
        )
    else:
        target_gyro = torch.empty(0, dtype=torch.long, device=hessian.device)
    shared = torch.arange(shared_slice.start, shared_slice.stop, device=hessian.device)
    retained = torch.cat([target_pose_velocity, shared, target_gyro])

    h_mm = reduced_hessian[source][:, source]
    h_mr = reduced_hessian[source][:, retained]
    h_rm = reduced_hessian[retained][:, source]
    h_rr = reduced_hessian[retained][:, retained]
    g_m = reduced_gradient[source]
    g_r = reduced_gradient[retained]
    values_m, vectors_m = torch.linalg.eigh(0.5 * (h_mm + h_mm.mT))
    scale_m = max(float(values_m.abs().max().detach().cpu().item()), 1.0)
    floor_m = max(float(eigenvalue_floor), torch.finfo(hessian.dtype).eps * scale_m)
    inverse_m = vectors_m @ torch.diag(values_m.clamp_min(floor_m).reciprocal()) @ vectors_m.mT
    prior_hessian_active = 0.5 * (
        (h_rr - h_rm @ inverse_m @ h_mr) + (h_rr - h_rm @ inverse_m @ h_mr).mT
    )
    prior_gradient_active = g_r - h_rm @ inverse_m @ g_m

    active_local = torch.cat(
        [
            torch.arange(0, 9, device=hessian.device),
            torch.arange(9, 12, device=hessian.device),
            torch.arange(12, 15, device=hessian.device)
            if optimize_gyro_bias
            else torch.empty(0, dtype=torch.long, device=hessian.device),
        ]
    )
    values, vectors = torch.linalg.eigh(prior_hessian_active)
    scale = max(float(values.abs().max().detach().cpu().item()), 1.0)
    threshold = max(float(eigenvalue_floor), torch.finfo(hessian.dtype).eps * scale)
    positive = values > threshold
    if not bool(positive.any()):
        raise RuntimeError("shared-ba marginalization produced an empty prior")
    active_values = values[positive]
    active_vectors = vectors[:, positive]
    sqrt_active = torch.diag(active_values.sqrt()) @ active_vectors.mT
    sqrt_information = torch.zeros(
        (sqrt_active.shape[0], STATE_DOF), dtype=hessian.dtype, device=hessian.device
    )
    sqrt_information[:, active_local] = sqrt_active
    residual_offset = (
        torch.diag(active_values.rsqrt()) @ active_vectors.mT @ prior_gradient_active
    )
    return SquareRootPrior(
        reference=optimized_j.detach(),
        sqrt_information=sqrt_information.detach(),
        residual_offset=residual_offset.detach(),
    )


def propagate_prior_acc_bias_random_walk(
    prior: SquareRootPrior,
    covariance: torch.Tensor,
    *,
    eigenvalue_floor: float = 1e-12,
) -> SquareRootPrior:
    """Predict a persistent ba prior through one segment random-walk step."""
    dtype = prior.sqrt_information.dtype
    device = prior.sqrt_information.device
    covariance = covariance.reshape(3, 3).to(device=device, dtype=dtype)
    covariance = 0.5 * (covariance + covariance.mT)
    if not bool(torch.isfinite(covariance).all()) or bool((torch.linalg.eigvalsh(covariance) < 0).any()):
        raise ValueError("accelerometer-bias random-walk covariance must be finite PSD")
    information = prior.sqrt_information.mT @ prior.sqrt_information
    information = 0.5 * (information + information.mT)
    values, vectors = torch.linalg.eigh(information)
    scale = max(float(values.abs().max().detach().cpu().item()), 1.0)
    threshold = max(float(eigenvalue_floor), torch.finfo(dtype).eps * scale)
    if not bool((values > threshold).all()):
        raise ValueError("cannot propagate random walk through a rank-deficient state prior")
    state_covariance = vectors @ torch.diag(values.reciprocal()) @ vectors.mT
    state_covariance[9:12, 9:12] += covariance
    state_covariance = 0.5 * (state_covariance + state_covariance.mT)
    predicted_information = torch.linalg.inv(state_covariance)
    predicted_information = 0.5 * (predicted_information + predicted_information.mT)

    old_gradient = prior.sqrt_information.mT @ prior.residual_offset
    mean_increment = -torch.linalg.solve(information, old_gradient)
    predicted_gradient = -predicted_information @ mean_increment
    new_values, new_vectors = torch.linalg.eigh(predicted_information)
    new_values = new_values.clamp_min(threshold)
    sqrt_information = torch.diag(new_values.sqrt()) @ new_vectors.mT
    residual_offset = torch.diag(new_values.rsqrt()) @ new_vectors.mT @ predicted_gradient
    return SquareRootPrior(
        reference=prior.reference.detach(),
        sqrt_information=sqrt_information.detach(),
        residual_offset=residual_offset.detach(),
    )


def _visual_residual(
    state_i: NavigationState,
    state_j: NavigationState,
    visual: RelativePoseFactor,
    covariance_eigenvalue_floor: float,
    *,
    robust: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    predicted = pp.SE3(state_i.pose_WB).Inv() @ pp.SE3(state_j.pose_WB)
    raw = (pp.SE3(visual.measurement_BiBj).Inv() @ predicted).Log().tensor().reshape(6)
    white = _whiten(raw, visual.covariance, covariance_eigenvalue_floor)
    if not robust:
        return white, white
    norm = torch.linalg.vector_norm(white)
    delta = max(float(visual.huber_delta), 1e-12)
    weight = torch.where(
        norm <= delta,
        torch.ones_like(norm),
        torch.as_tensor(delta, dtype=norm.dtype, device=norm.device) / norm.clamp_min(1e-12),
    ).detach()
    return weight.sqrt() * white, white


def fixed_lag_factor_residuals(
    states: tuple[NavigationState, ...],
    prior: SquareRootPrior,
    imus: tuple[ImuPreintegrationFactor, ...],
    visuals: tuple[RelativePoseFactor, ...],
    *,
    covariance_eigenvalue_floor: float,
    robust_visual: bool,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if len(states) < 2 or len(imus) != len(states) - 1 or len(visuals) != len(imus):
        raise ValueError("fixed-lag factors must form one contiguous pairwise chain")
    prior_r = prior.sqrt_information @ state_boxminus(states[0], prior.reference) + prior.residual_offset
    imu_rows, bias_rows, visual_rows, visual_white_rows = [], [], [], []
    for state_i, state_j, imu, visual in zip(states[:-1], states[1:], imus, visuals):
        imu_raw = vio_preintegrated_imu_residual(
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
        imu_rows.append(_whiten(imu_raw, imu.covariance, covariance_eigenvalue_floor))
        bias_raw = vio_bias_random_walk_residual(
            prev_acc_bias=state_i.acc_bias,
            prev_gyro_bias=state_i.gyro_bias,
            curr_acc_bias=state_j.acc_bias,
            curr_gyro_bias=state_j.gyro_bias,
        ).reshape(6)
        bias_rows.append(_whiten(bias_raw, imu.bias_rw_covariance, covariance_eigenvalue_floor))
        visual_r, visual_white = _visual_residual(
            state_i, state_j, visual, covariance_eigenvalue_floor, robust=robust_visual
        )
        visual_rows.append(visual_r)
        visual_white_rows.append(visual_white)
    blocks = {
        "prior": prior_r,
        "imu": torch.cat(imu_rows),
        "bias": torch.cat(bias_rows),
        "visual_pose": torch.cat(visual_rows),
        "visual_pose_unweighted": torch.stack(visual_white_rows),
    }
    residual = torch.cat([blocks["prior"], blocks["imu"], blocks["bias"], blocks["visual_pose"]])
    return residual, blocks


def fixed_lag_true_cost(blocks: dict[str, torch.Tensor], huber_deltas: torch.Tensor) -> torch.Tensor:
    cost = 0.5 * (
        blocks["prior"].square().sum()
        + blocks["imu"].square().sum()
        + blocks["bias"].square().sum()
    )
    norms = torch.linalg.vector_norm(blocks["visual_pose_unweighted"], dim=1)
    deltas = huber_deltas.to(norms).clamp_min(1e-12)
    visual = torch.where(norms <= deltas, 0.5 * norms.square(), deltas * norms - 0.5 * deltas.square())
    return cost + visual.sum()


def _linearize_window(
    states: tuple[NavigationState, ...],
    prior: SquareRootPrior,
    imus: tuple[ImuPreintegrationFactor, ...],
    visuals: tuple[RelativePoseFactor, ...],
    covariance_eigenvalue_floor: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    dof = STATE_DOF * len(states)
    zero = torch.zeros(dof, dtype=states[0].pose_WB.dtype, device=states[0].pose_WB.device, requires_grad=True)

    def residual_from_increment(increment: torch.Tensor) -> torch.Tensor:
        candidates = tuple(
            retract_state(state, increment[index * STATE_DOF : (index + 1) * STATE_DOF])
            for index, state in enumerate(states)
        )
        residual, _ = fixed_lag_factor_residuals(
            candidates,
            prior,
            imus,
            visuals,
            covariance_eigenvalue_floor=covariance_eigenvalue_floor,
            robust_visual=True,
        )
        return residual

    residual = residual_from_increment(zero)
    jacobian = torch.autograd.functional.jacobian(
        residual_from_increment, zero, create_graph=False, vectorize=True
    )
    hessian = jacobian.mT @ jacobian
    gradient = jacobian.mT @ residual.detach()
    _, blocks = fixed_lag_factor_residuals(
        states,
        prior,
        imus,
        visuals,
        covariance_eigenvalue_floor=covariance_eigenvalue_floor,
        robust_visual=False,
    )
    return hessian.detach(), gradient.detach(), residual.detach(), blocks


class FixedLagVIOSolver:
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

    def solve(self, problem: FixedLagVIOProblem) -> FixedLagVIOResult:
        dtype = torch.float64
        device = problem.states[0].pose_WB.device
        states = tuple(state.to(device=device, dtype=dtype) for state in problem.states)
        if problem.shared_acc_bias:
            reference_ba = states[0].acc_bias.reshape(3)
            if any(
                not torch.allclose(state.acc_bias.reshape(3), reference_ba, atol=1e-12, rtol=1e-12)
                for state in states[1:]
            ):
                raise ValueError("shared_acc_bias requires identical initial ba for every active state")
        prior = problem.prior_first.to(device=device, dtype=dtype)
        imus = tuple(factor.to(device=device, dtype=dtype) for factor in problem.imu_factors)
        visuals = tuple(factor.to(device=device, dtype=dtype) for factor in problem.visual_factors)
        huber_deltas = torch.tensor([factor.huber_delta for factor in visuals], dtype=dtype, device=device)
        state_mask = torch.ones(STATE_DOF, dtype=torch.bool, device=device)
        state_mask[9:12] = bool(problem.optimize_acc_bias)
        state_mask[12:15] = bool(problem.optimize_gyro_bias)
        projection = None
        if problem.shared_acc_bias:
            projection = _shared_acc_bias_projection(
                len(states),
                device=device,
                dtype=dtype,
                optimize_acc_bias=bool(problem.optimize_acc_bias),
                optimize_gyro_bias=bool(problem.optimize_gyro_bias),
            )
            active_indices = torch.arange(projection.shape[1], device=device)
        else:
            active_indices = torch.nonzero(state_mask.repeat(len(states)), as_tuple=False).reshape(-1)

        hessian, gradient, _, blocks = _linearize_window(
            states, prior, imus, visuals, self.covariance_eigenvalue_floor
        )
        initial_cost = float(fixed_lag_true_cost(blocks, huber_deltas).item())
        current_cost = initial_cost
        damping = self.initial_damping
        converged = False
        iterations = accepted = rejected = 0
        final_step_norm = float("inf")
        reason = "iteration_limit"
        for iteration in range(self.max_iterations):
            iterations = iteration + 1
            hessian, gradient, _, _ = _linearize_window(
                states, prior, imus, visuals, self.covariance_eigenvalue_floor
            )
            if projection is None:
                active_h = hessian[active_indices][:, active_indices]
                active_g = gradient[active_indices]
            else:
                active_h = projection.mT @ hessian @ projection
                active_g = projection.mT @ gradient
            diagonal = active_h.diagonal().abs().clamp_min(1.0)
            system = active_h + damping * torch.diag(diagonal)
            try:
                active_step = torch.linalg.solve(system, -active_g)
            except torch.linalg.LinAlgError:
                active_step = torch.linalg.pinv(system) @ (-active_g)
            if projection is None:
                step = torch.zeros(STATE_DOF * len(states), dtype=dtype, device=device)
                step[active_indices] = active_step
            else:
                step = projection @ active_step
            if not bool(torch.isfinite(step).all()):
                raise FloatingPointError("fixed-lag LM produced a non-finite step")
            final_step_norm = float(torch.linalg.vector_norm(step).item())
            if final_step_norm <= self.step_tolerance:
                converged, reason = True, "step_tolerance"
                break
            candidates = tuple(
                retract_state(state, step[index * STATE_DOF : (index + 1) * STATE_DOF])
                for index, state in enumerate(states)
            )
            _, candidate_blocks = fixed_lag_factor_residuals(
                candidates,
                prior,
                imus,
                visuals,
                covariance_eigenvalue_floor=self.covariance_eigenvalue_floor,
                robust_visual=False,
            )
            candidate_cost = float(fixed_lag_true_cost(candidate_blocks, huber_deltas).item())
            if candidate_cost < current_cost:
                accepted += 1
                previous = current_cost
                states = tuple(state.detach() for state in candidates)
                current_cost = candidate_cost
                damping = max(damping * 0.25, 1e-12)
                if abs(previous - current_cost) <= self.cost_tolerance:
                    converged, reason = True, "cost_tolerance"
                    break
            else:
                rejected += 1
                damping = min(damping * 10.0, 1e12)

        hessian, gradient, _, blocks = _linearize_window(
            states, prior, imus, visuals, self.covariance_eigenvalue_floor
        )
        pair_problem = TwoStateVIOProblem(
            state_i=states[0],
            state_j=states[1],
            prior_i=prior,
            imu=imus[0],
            visual_pose=visuals[0],
            optimize_acc_bias=problem.optimize_acc_bias,
            optimize_gyro_bias=problem.optimize_gyro_bias,
        )
        _, _, pair_h, pair_g, _ = _linearize(
            pair_problem.state_i,
            pair_problem.state_j,
            pair_problem.prior_i,
            pair_problem.imu,
            pair_problem.visual_pose,
            self.covariance_eigenvalue_floor,
        )
        if problem.shared_acc_bias:
            prior_next = marginalize_source_state_shared_acc_bias(
                states[0],
                states[1],
                pair_h,
                pair_g,
                optimize_gyro_bias=bool(problem.optimize_gyro_bias),
                eigenvalue_floor=self.marginalization_eigenvalue_floor,
            )
            information_projection = _shared_acc_bias_projection(
                len(states),
                device=device,
                dtype=dtype,
                optimize_acc_bias=True,
                optimize_gyro_bias=bool(problem.optimize_gyro_bias),
            )
            reduced_hessian = information_projection.mT @ hessian @ information_projection
            shared_start = reduced_hessian.shape[0] - 3
            ba_information = _marginal_information_block(
                reduced_hessian,
                torch.arange(shared_start, shared_start + 3, device=device),
                eigenvalue_floor=self.marginalization_eigenvalue_floor,
            )
            ba_information_eigenvalues = torch.linalg.eigvalsh(ba_information).detach()
        else:
            prior_next = marginalize_source_state(
                states[0], states[1], pair_h, pair_g,
                eigenvalue_floor=self.marginalization_eigenvalue_floor,
                active_state_mask=state_mask,
            )
            ba_information_eigenvalues = None
        visual_norms = torch.linalg.vector_norm(blocks["visual_pose_unweighted"], dim=1)
        visual_costs = torch.where(
            visual_norms <= huber_deltas,
            0.5 * visual_norms.square(),
            huber_deltas * visual_norms - 0.5 * huber_deltas.square(),
        )
        return FixedLagVIOResult(
            states=tuple(state.detach() for state in states),
            prior_next=prior_next,
            converged=converged,
            iterations=iterations,
            initial_cost=initial_cost,
            final_cost=float(fixed_lag_true_cost(blocks, huber_deltas).item()),
            prior_cost=0.5 * float(blocks["prior"].square().sum().item()),
            imu_cost=0.5 * float(blocks["imu"].square().sum().item()),
            bias_cost=0.5 * float(blocks["bias"].square().sum().item()),
            visual_pose_cost=float(visual_costs.sum().item()),
            hessian=hessian,
            gradient=gradient,
            final_step_norm=final_step_norm,
            final_gradient_inf_norm=float(
                (
                    gradient[active_indices]
                    if projection is None
                    else projection.mT @ gradient
                ).abs().max().item()
            ),
            convergence_reason=reason,
            accepted_steps=accepted,
            rejected_steps=rejected,
            shared_acc_bias=bool(problem.shared_acc_bias),
            acc_bias_marginal_information_eigenvalues=ba_information_eigenvalues,
        )
