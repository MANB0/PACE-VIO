#!/usr/bin/env python3
"""Profile the frozen T2 N=2 linearization without changing production math."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Callable

import pypose as pp
import torch
from pypose.lietensor.operation import se3_Jl_inv


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Scripts.run_u1_counterfactual_branches import clone_problem  # noqa: E402
from Utility.CompressedUVDFactorCache import CompressedUVDFactorCacheReader  # noqa: E402
from Utility.IMUKinematics import (  # noqa: E402
    vio_bias_random_walk_residual,
    vio_preintegrated_imu_residual,
)
from Utility.TwoStateVIO import (  # noqa: E402
    PAIR_DOF,
    STATE_DOF,
    LinearizedUVDPoseFactor,
    NavigationState,
    TwoStateVIOProblem,
    _factor_residuals,
    _bias_residual_and_analytic_jacobian,
    _camera_relative_CjCi,
    _linearize,
    _linearize_blockwise_autograd,
    _linearize_joint_autograd,
    _linearized_visual_residual_and_analytic_jacobian,
    _prior_residual_and_analytic_jacobian,
    _se3_adjoint_matrix,
    _true_cost,
    _whiten,
    linearized_uvd_pose_factor_from_normal_equations,
    marginalize_source_state,
    retract_state,
    visual_whitened_residuals,
)


DEFAULT_PACKET = ROOT / (
    "analysis_rectangle_uvd_schur_marginal_20260719/"
    "captured_rectangle_u1_problems.pt"
)
DEFAULT_CACHE = ROOT / (
    "VisualCache/static63_unique_visual_20260713/"
    "clear_stop_turn_rectangle_truth_normal_noise"
)
DEFAULT_OUTPUT = ROOT / "analysis_pure_u1_t2_pareto_20260720/t2_profile"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, default=DEFAULT_PACKET)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--profile-edges", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--cpu-threads", type=int, default=4)
    return parser.parse_args()


def median_ms(action: Callable[[], object], repeats: int) -> tuple[float, object]:
    action()
    values: list[float] = []
    result: object = None
    for _ in range(repeats):
        start = time.perf_counter()
        result = action()
        values.append((time.perf_counter() - start) * 1000.0)
    return statistics.median(values), result


def candidates(
    problem: TwoStateVIOProblem, increment: torch.Tensor
) -> tuple[NavigationState, NavigationState]:
    return (
        retract_state(problem.state_i, increment[:STATE_DOF]),
        retract_state(problem.state_j, increment[STATE_DOF:]),
    )


def prior_residual(problem: TwoStateVIOProblem, increment: torch.Tensor) -> torch.Tensor:
    state_i, _ = candidates(problem, increment)
    prior = problem.prior_i
    from Utility.TwoStateVIO import state_boxminus

    return (
        prior.sqrt_information @ state_boxminus(state_i, prior.reference)
        + prior.residual_offset
    )


def imu_residual(problem: TwoStateVIOProblem, increment: torch.Tensor) -> torch.Tensor:
    state_i, state_j = candidates(problem, increment)
    imu = problem.imu
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
    return _whiten(raw, imu.covariance, 1.0e-12)


def bias_residual(problem: TwoStateVIOProblem, increment: torch.Tensor) -> torch.Tensor:
    state_i, state_j = candidates(problem, increment)
    imu = problem.imu
    raw = vio_bias_random_walk_residual(
        prev_acc_bias=state_i.acc_bias,
        prev_gyro_bias=state_i.gyro_bias,
        curr_acc_bias=state_j.acc_bias,
        curr_gyro_bias=state_j.gyro_bias,
    ).reshape(6)
    return _whiten(raw, imu.bias_rw_covariance, 1.0e-12)


def visual_residual(problem: TwoStateVIOProblem, increment: torch.Tensor) -> torch.Tensor:
    state_i, state_j = candidates(problem, increment)
    rows = visual_whitened_residuals(state_i, state_j, problem.visual_pose, 1.0e-12)
    norms = torch.linalg.vector_norm(rows, dim=-1)
    delta = max(float(problem.visual_pose.huber_delta), 1.0e-12)
    weights = torch.where(
        norms <= delta,
        torch.ones_like(norms),
        torch.as_tensor(delta, dtype=norms.dtype, device=norms.device)
        / norms.clamp_min(1.0e-12),
    ).detach()
    return (weights.sqrt().unsqueeze(-1) * rows).reshape(-1)


def jacobian(
    function: Callable[[torch.Tensor], torch.Tensor],
    zero: torch.Tensor,
    strategy: str,
) -> torch.Tensor:
    return torch.autograd.functional.jacobian(
        function,
        zero,
        create_graph=False,
        vectorize=True,
        strategy=strategy,
    )


def build_problem(edge: dict, packet) -> TwoStateVIOProblem:
    incoming = clone_problem(edge["problem"])
    visual = incoming.visual_pose
    factor = linearized_uvd_pose_factor_from_normal_equations(
        packet.reference_CjCi,
        packet.hessian,
        packet.gradient,
        visual.extrinsic_CI,
    )
    if not isinstance(factor, LinearizedUVDPoseFactor):
        raise TypeError("expected a compressed T2 visual factor")
    return replace(incoming, visual_pose=factor)


def main() -> int:
    args = parse_args()
    torch.set_num_threads(int(args.cpu_threads))
    payload = torch.load(args.packet, map_location="cpu", weights_only=False)
    start = int(payload.get("active_start_frame", 0))
    all_edges = [edge for edge in payload["edges"] if int(edge["frame_i"]) >= start]
    count = min(max(int(args.profile_edges), 1), len(all_edges))
    indices = sorted(set(round(value) for value in torch.linspace(0, len(all_edges) - 1, count).tolist()))
    reader = CompressedUVDFactorCacheReader(args.cache)
    rows: list[dict[str, float | int]] = []
    for sample_id, edge_index in enumerate(indices):
        edge = all_edges[edge_index]
        packet = reader.load_pair(
            int(edge["frame_i"]),
            int(edge["frame_j"]),
            str(reader.visual_hashes[int(edge["frame_i"])]),
        )
        problem = build_problem(edge, packet)
        problem = TwoStateVIOProblem(
            state_i=problem.state_i.to(device=torch.device("cpu"), dtype=torch.float64),
            state_j=problem.state_j.to(device=torch.device("cpu"), dtype=torch.float64),
            prior_i=problem.prior_i.to(device=torch.device("cpu"), dtype=torch.float64),
            imu=problem.imu.to(device=torch.device("cpu"), dtype=torch.float64),
            visual_pose=problem.visual_pose.to(device=torch.device("cpu"), dtype=torch.float64),
            optimize_acc_bias=problem.optimize_acc_bias,
            optimize_gyro_bias=problem.optimize_gyro_bias,
        )
        zero = torch.zeros(PAIR_DOF, dtype=torch.float64, requires_grad=True)
        block_functions = {
            "prior": lambda x, p=problem: prior_residual(p, x),
            "imu": lambda x, p=problem: imu_residual(p, x),
            "bias": lambda x, p=problem: bias_residual(p, x),
            "visual": lambda x, p=problem: visual_residual(p, x),
        }
        row: dict[str, float | int] = {
            "sample_id": sample_id,
            "edge_index": edge_index,
            "frame_i": int(edge["frame_i"]),
            "frame_j": int(edge["frame_j"]),
        }
        autodiff_blocks: dict[str, torch.Tensor] = {}
        for name, function in block_functions.items():
            elapsed, block_j = median_ms(
                lambda fn=function: jacobian(fn, zero, "reverse-mode"),
                int(args.repeats),
            )
            row[f"{name}_jacobian_reverse_ms"] = elapsed
            row[f"{name}_rows"] = int(block_j.shape[0])
            autodiff_blocks[name] = block_j

        analytic_prior_r, analytic_prior_j = _prior_residual_and_analytic_jacobian(
            problem.state_i, problem.prior_i
        )
        analytic_bias_r, analytic_bias_j = _bias_residual_and_analytic_jacobian(
            problem.state_i,
            problem.state_j,
            problem.imu,
            1.0e-12,
        )
        analytic_visual_r, analytic_visual_j = (
            _linearized_visual_residual_and_analytic_jacobian(
                problem.state_i,
                problem.state_j,
                problem.visual_pose,
            )
        )
        row["prior_analytic_jacobian_max_abs"] = float(
            (analytic_prior_j - autodiff_blocks["prior"]).abs().max().item()
        )
        row["bias_analytic_jacobian_max_abs"] = float(
            (analytic_bias_j - autodiff_blocks["bias"]).abs().max().item()
        )
        row["visual_analytic_jacobian_max_abs"] = float(
            (analytic_visual_j - autodiff_blocks["visual"]).abs().max().item()
        )
        row["prior_analytic_residual_max_abs"] = float(
            (analytic_prior_r - block_functions["prior"](zero)).abs().max().item()
        )
        row["bias_analytic_residual_max_abs"] = float(
            (analytic_bias_r - block_functions["bias"](zero)).abs().max().item()
        )
        row["visual_analytic_residual_max_abs"] = float(
            (analytic_visual_r - block_functions["visual"](zero)).abs().max().item()
        )
        pose_zero = torch.zeros(6, dtype=torch.float64, requires_grad=True)
        prior_relative = (
            pp.SE3(problem.prior_i.reference.pose_WB).Inv()
            @ pp.SE3(problem.state_i.pose_WB)
        )
        prior_local = prior_relative.Log().tensor().reshape(6)

        def prior_pose_local(increment: torch.Tensor) -> torch.Tensor:
            candidate = pp.SE3(problem.state_i.pose_WB) @ pp.se3(
                increment.reshape(1, 6)
            ).Exp()
            return (
                pp.SE3(problem.prior_i.reference.pose_WB).Inv() @ candidate
            ).Log().tensor().reshape(6)

        prior_local_j = torch.autograd.functional.jacobian(
            prior_pose_local, pose_zero, vectorize=True
        )
        row["prior_local_vs_jr_inv_max_abs"] = float(
            (
                prior_local_j
                - se3_Jl_inv(-prior_local.reshape(1, 6)).reshape(6, 6)
            )
            .abs()
            .max()
            .item()
        )
        row["prior_local_vs_jl_inv_max_abs"] = float(
            (
                prior_local_j
                - se3_Jl_inv(prior_local.reshape(1, 6)).reshape(6, 6)
            )
            .abs()
            .max()
            .item()
        )
        relative_zero = torch.zeros(12, dtype=torch.float64, requires_grad=True)
        current_relative = _camera_relative_CjCi(
            problem.state_i,
            problem.state_j,
            problem.visual_pose.extrinsic_CI,
        ).detach()

        def relative_local_from_body(increment: torch.Tensor) -> torch.Tensor:
            candidate_i = retract_state(problem.state_i, torch.cat([increment[0:6], torch.zeros(9, dtype=increment.dtype)]))
            candidate_j = retract_state(problem.state_j, torch.cat([increment[6:12], torch.zeros(9, dtype=increment.dtype)]))
            candidate_relative = _camera_relative_CjCi(
                candidate_i,
                candidate_j,
                problem.visual_pose.extrinsic_CI,
            )
            return (
                current_relative.Inv() @ candidate_relative
            ).Log().tensor().reshape(6)

        relative_body_j = torch.autograd.functional.jacobian(
            relative_local_from_body, relative_zero, vectorize=True
        )
        extrinsic_adjoint = _se3_adjoint_matrix(
            pp.SE3(problem.visual_pose.extrinsic_CI)
        )
        relative_body_analytic = torch.cat(
            [
                extrinsic_adjoint,
                -_se3_adjoint_matrix(current_relative.Inv()) @ extrinsic_adjoint,
            ],
            dim=1,
        )
        row["relative_body_map_analytic_max_abs"] = float(
            (relative_body_j - relative_body_analytic).abs().max().item()
        )

        def full_blockwise():
            return _linearize_blockwise_autograd(
                problem.state_i,
                problem.state_j,
                problem.prior_i,
                problem.imu,
                problem.visual_pose,
                1.0e-12,
            )

        blockwise_ms, blockwise_result = median_ms(
            full_blockwise, int(args.repeats)
        )
        row["blockwise_linearize_ms"] = blockwise_ms
        blockwise_r, blockwise_j, blockwise_h, blockwise_g, _ = blockwise_result

        def full_joint_autograd():
            return _linearize_joint_autograd(
                problem.state_i,
                problem.state_j,
                problem.prior_i,
                problem.imu,
                problem.visual_pose,
                1.0e-12,
            )

        joint_ms, joint_result = median_ms(
            full_joint_autograd, int(args.repeats)
        )
        row["joint_autograd_linearize_ms"] = joint_ms
        joint_r, joint_j, joint_h, joint_g, _ = joint_result
        row["blockwise_joint_residual_max_abs"] = float(
            (blockwise_r - joint_r).abs().max().item()
        )
        row["blockwise_joint_jacobian_max_abs"] = float(
            (blockwise_j - joint_j).abs().max().item()
        )
        row["blockwise_joint_hessian_max_abs"] = float(
            (blockwise_h - joint_h).abs().max().item()
        )
        row["blockwise_joint_gradient_max_abs"] = float(
            (blockwise_g - joint_g).abs().max().item()
        )

        def full_residual(increment: torch.Tensor) -> torch.Tensor:
            state_i, state_j = candidates(problem, increment)
            residual, _ = _factor_residuals(
                state_i,
                state_j,
                problem.prior_i,
                problem.imu,
                problem.visual_pose,
                covariance_eigenvalue_floor=1.0e-12,
                robust_visual=True,
            )
            return residual

        try:
            jacrev_function = pp.func.jacrev(full_residual)
            jacrev_ms, jacrev_j = median_ms(
                lambda: jacrev_function(zero.detach()),
                int(args.repeats),
            )
            row["pypose_func_jacrev_ms"] = jacrev_ms
            row["pypose_func_jacrev_max_abs"] = float(
                (jacrev_j - joint_j).abs().max().item()
            )
            row["pypose_func_jacrev_supported"] = 1
        except (NotImplementedError, RuntimeError):
            row["pypose_func_jacrev_ms"] = float("nan")
            row["pypose_func_jacrev_max_abs"] = float("nan")
            row["pypose_func_jacrev_supported"] = 0

        try:
            forward_ms, forward_j = median_ms(
                lambda: jacobian(full_residual, zero, "forward-mode"),
                int(args.repeats),
            )
            row["joint_jacobian_forward_ms"] = forward_ms
            row["forward_reverse_jacobian_max_abs"] = float(
                (forward_j - joint_j).abs().max().item()
            )
            row["forward_mode_supported"] = 1
        except NotImplementedError:
            # PyPose 0.6.x custom LieTensor Functions do not implement JVP.
            row["joint_jacobian_forward_ms"] = float("nan")
            row["forward_reverse_jacobian_max_abs"] = float("nan")
            row["forward_mode_supported"] = 0

        active = torch.arange(PAIR_DOF)
        diagonal = blockwise_h.diagonal().abs().clamp_min(1.0)
        system = blockwise_h + 1.0e-3 * torch.diag(diagonal)
        solve_ms, _ = median_ms(
            lambda: torch.linalg.solve(system[active][:, active], -blockwise_g[active]),
            max(int(args.repeats), 5),
        )
        row["dense_solve_ms"] = solve_ms
        marginal_ms, _ = median_ms(
            lambda: marginalize_source_state(
                problem.state_i,
                problem.state_j,
                blockwise_h,
                blockwise_g,
            ),
            max(int(args.repeats), 5),
        )
        row["marginalization_ms"] = marginal_ms
        rows.append(row)
        print(
            f"[{sample_id + 1}/{len(indices)}] edge={edge_index} "
            f"joint={joint_ms:.2f}ms blockwise={blockwise_ms:.2f}ms "
            f"imu={row['imu_jacobian_reverse_ms']:.2f}ms "
            f"visual={row['visual_jacobian_reverse_ms']:.2f}ms",
            flush=True,
        )

    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "factor_profile_per_edge.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    numeric_fields = [
        key for key in rows[0] if key.endswith("_ms") or key.endswith("_max_abs")
    ]
    summary = {
        "contract": {
            "packet": str(args.packet.resolve()),
            "cache": str(args.cache.resolve()),
            "profile_edges": len(rows),
            "cpu_threads": torch.get_num_threads(),
            "dtype": "torch.float64",
            "production_math_changed": False,
        },
        "median": {
            key: statistics.median(float(row[key]) for row in rows)
            for key in numeric_fields
        },
        "forward_mode_supported": all(
            int(row["forward_mode_supported"]) == 1 for row in rows
        ),
        "max_forward_reverse_jacobian_abs": (
            max(
                float(row["forward_reverse_jacobian_max_abs"])
                for row in rows
                if int(row["forward_mode_supported"]) == 1
            )
            if any(int(row["forward_mode_supported"]) == 1 for row in rows)
            else None
        ),
    }
    (args.output / "factor_profile_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
