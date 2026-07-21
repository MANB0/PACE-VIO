#!/usr/bin/env python3
"""Run the short-circle V3 backend information-strength diagnostics.

This is an experiment-only runner. It reuses the production two-state backend without
changing its formulas, LM settings, marginalization, IMU factors, or state definition.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pypose as pp
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import Scripts.run_circle_translation_oracles as base  # noqa: E402
from Utility.TwoStateVIO import (  # noqa: E402
    PAIR_DOF,
    STATE_DOF,
    NavigationState,
    TwoStateVIOProblem,
    TwoStateVIOResult,
    TwoStateVIOSolver,
    _factor_residuals,
    _linearize,
    _true_cost,
    marginalize_source_state,
    retract_state,
)


OUT = ROOT / "analysis_circle_v3_backend_pointlevel_20260716/information_strength"
CASES: dict[str, dict[str, Any]] = {
    "C0": {"description": "current V3", "covariance_scale": 1.0, "huber_delta": 3.0},
    "C1": {"description": "Huber disabled", "covariance_scale": 1.0, "huber_delta": 1.0e12},
    "C2": {"description": "visual covariance x0.3", "covariance_scale": 0.3, "huber_delta": 3.0},
    "C3": {"description": "visual covariance x0.1", "covariance_scale": 0.1, "huber_delta": 3.0},
    "C4": {"description": "visual covariance x0.03", "covariance_scale": 0.03, "huber_delta": 3.0},
    "C5": {"description": "near-hard GT pose factor", "covariance_scale": 1.0e-6, "huber_delta": 3.0},
    "C6": {"description": "GT poses fixed; optimize v/ba/bg", "covariance_scale": 1.0, "huber_delta": 3.0},
}

GATE_ROWS: list[dict[str, Any]] = []
FIXED_POSES: dict[str, torch.Tensor | None] = {"i": None, "j": None}


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(base.jsonify(payload), indent=2, ensure_ascii=False), encoding="utf-8")


def configure_base() -> None:
    base.OUT = OUT
    base.MODES = {
        case: {"rotation": "gt", "translation": "gt", "opt_ba": True, "opt_bg": True}
        for case in CASES
    }


def prepare_case(case: str) -> dict[str, Any]:
    configure_base()
    contract = {
        "cache": base.prepare_cache("short", case),
        "config": base.prepare_config("short", case),
        "case": CASES[case],
    }
    cache_path = base.cache_root("short", case) / "relative_pose_factors.npz"
    with np.load(cache_path, allow_pickle=False) as stream:
        arrays = {name: stream[name].copy() for name in stream.files}
    source_covariance = arrays["covariance"].copy()
    arrays["covariance"] *= float(CASES[case]["covariance_scale"])
    with cache_path.open("wb") as stream:
        np.savez_compressed(stream, **arrays)

    config_path = base.config_path("short", case)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    optimizer = config["Odometry"]["optimizer"]["args"]
    optimizer["two_state_visual_huber_delta"] = float(CASES[case]["huber_delta"])
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    source_eigenvalues = np.linalg.eigvalsh(source_covariance)
    scaled_eigenvalues = np.linalg.eigvalsh(arrays["covariance"])
    contract["proofs"] = {
        "measurement_mean_is_GT": True,
        "covariance_scale_exact_max_abs_error": float(
            np.max(np.abs(arrays["covariance"] - source_covariance * CASES[case]["covariance_scale"]))
        ),
        "source_covariance_min_eigenvalue": float(source_eigenvalues.min()),
        "scaled_covariance_min_eigenvalue": float(scaled_eigenvalues.min()),
        "huber_delta_written": float(optimizer["two_state_visual_huber_delta"]),
        "locked_factor_mode": optimizer["imu_factor_mode"],
        "locked_max_iterations": int(optimizer["two_state_max_iterations"]),
        "locked_covariance_mode": config["Odometry"]["args"]["two_state_covariance_mode"],
        "locked_gravity_handling": config["Odometry"]["args"]["imu_vio_gravity_handling"],
    }
    write_json(base.mode_root("short", case) / "experiment_contract.json", contract)
    return contract


def prepare_all() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    contracts = {case: prepare_case(case) for case in CASES}
    write_json(OUT / "information_strength_preparation_contract.json", contracts)
    create_c7_gt_chain()


def create_c7_gt_chain() -> None:
    """Construct the no-IMU GT relative-pose chain and prove the composition identity."""

    root = base.mode_root("short", "C7")
    root.mkdir(parents=True, exist_ok=True)
    ref, truth = base.load_truth(base.DEFAULT_DATASET)
    truth = truth[: base.SHORT_FRAME_LIMIT]
    start = base.ACTIVE_START_FRAME
    chained = [truth[start].copy()]
    for frame_i in range(start, len(truth) - 1):
        z_gt = base.invert_transform(truth[frame_i]) @ truth[frame_i + 1]
        chained.append(chained[-1] @ z_gt)
    chained_array = np.stack(chained)
    truth_active = truth[start:]
    max_abs = float(np.max(np.abs(chained_array - truth_active)))
    metrics = base.trajectory_metrics(chained_array, truth_active)
    rows = []
    for offset, (estimate, target) in enumerate(zip(chained_array, truth_active)):
        frame = start + offset
        rotation_error = base.rotation_log(target[:3, :3].T @ estimate[:3, :3])
        row = {
            "case": "C7",
            "frame": frame,
            "timestamp_ns": int(ref.iloc[frame]["timestamp"]),
        }
        for axis, value in zip("xyz", estimate[:3, 3]):
            row[f"position_est_{axis}"] = float(value)
        for axis, value in zip("xyz", target[:3, 3]):
            row[f"position_gt_{axis}"] = float(value)
        for axis, value in zip("xyz", estimate[:3, 3] - target[:3, 3]):
            row[f"position_error_{axis}"] = float(value)
        for axis, value in zip("xyz", rotation_error):
            row[f"rotation_error_{axis}"] = float(value)
        rows.append(row)
    state_path = root / "gt_relative_pose_chain_per_frame.csv"
    pd.DataFrame(rows).to_csv(state_path, index=False)
    summary = {
        "scope": "short",
        "mode": "C7",
        "description": "pure GT relative-pose chaining; no IMU and no optimizer",
        "metrics": metrics,
        "composition_max_abs_transform_error": max_abs,
        "velocity_bias_metrics_applicable": False,
        "factor_costs_applicable": False,
        "artifacts": {"state_per_frame": str(state_path)},
    }
    write_json(root / "summary.json", summary)


def run_case(case: str, *, force: bool) -> None:
    configure_base()
    optimizer_module = importlib.import_module("Module.Optimization.TwoFramePGO.Optimizer")
    original_gate = optimizer_module._gate_two_state_visual_factor
    GATE_ROWS.clear()

    def traced_gate(*args, **kwargs):
        covariance, diagnostics = original_gate(*args, **kwargs)
        row = {
            "case": case,
            "frame_i": int(base.CURRENT_EDGE["i"]),
            "frame_j": int(base.CURRENT_EDGE["j"]),
            "solver_call_before_gate": int(base.CURRENT_EDGE["call"]),
        }
        row.update(diagnostics)
        row["effective_covariance_trace"] = float(covariance.detach().cpu().trace().item())
        row["effective_covariance_logdet"] = float(
            np.linalg.slogdet(covariance.detach().cpu().numpy())[1]
        )
        GATE_ROWS.append(row)
        return covariance, diagnostics

    optimizer_module._gate_two_state_visual_factor = traced_gate
    original_base_solve = base.ORIGINAL_SOLVE
    original_base_optimize = base.ORIGINAL_OPTIMIZE
    if case == "C6":
        base.ORIGINAL_SOLVE = fixed_pose_solve
        base.ORIGINAL_OPTIMIZE = make_gt_pose_optimize(original_base_optimize)
    try:
        base.run_one("short", case, force=force)
    finally:
        optimizer_module._gate_two_state_visual_factor = original_gate
        base.ORIGINAL_SOLVE = original_base_solve
        base.ORIGINAL_OPTIMIZE = original_base_optimize
        TwoStateVIOSolver.solve = original_base_solve
        base.TwoFrame_PGO._optimize_two_state_fixed_lag = staticmethod(original_base_optimize)
    pd.DataFrame(GATE_ROWS).to_csv(base.mode_root("short", case) / "visual_gate_runtime.csv", index=False)
    summarize_all()


def make_gt_pose_optimize(production_optimize):
    _, truth_pose = base.load_truth(base.DEFAULT_DATASET)

    def optimize_with_gt_pose(context, graph_data):
        frame_i = int(graph_data.from_idx.reshape(-1)[0].item())
        frame_j = int(graph_data.frame_idx.reshape(-1)[0].item())
        dtype = graph_data.init_motion.dtype
        device = graph_data.init_motion.device
        pose_i = torch.from_numpy(base.se3_to_xyzw(truth_pose[frame_i])).to(device=device, dtype=dtype).reshape(1, 7)
        pose_j = torch.from_numpy(base.se3_to_xyzw(truth_pose[frame_j])).to(device=device, dtype=dtype).reshape(1, 7)
        graph_data.from_pose = pp.SE3(pose_i)
        graph_data.init_motion = pp.SE3(pose_j)
        extrinsic_ci = pp.SE3(
            graph_data.imu_vio_sensor_T_imu.reshape(1, 7).to(device=device, dtype=dtype)
        )
        FIXED_POSES["i"] = (pp.SE3(pose_i) @ extrinsic_ci).tensor().double()
        FIXED_POSES["j"] = (pp.SE3(pose_j) @ extrinsic_ci).tensor().double()
        return production_optimize(context, graph_data)

    return optimize_with_gt_pose


def fixed_pose_solve(self: TwoStateVIOSolver, problem: TwoStateVIOProblem) -> TwoStateVIOResult:
    """Experiment-only LM solve with both pose blocks removed from the active variables."""

    dtype = torch.float64
    device = problem.state_i.pose_WB.device
    fixed_i = FIXED_POSES["i"]
    fixed_j = FIXED_POSES["j"]
    if fixed_i is None or fixed_j is None:
        raise RuntimeError("C6 fixed GT poses were not initialized for the current edge")
    state_i = NavigationState(
        pose_WB=fixed_i.to(device=device, dtype=dtype),
        velocity_W=problem.state_i.velocity_W.to(device=device, dtype=dtype),
        acc_bias=problem.state_i.acc_bias.to(device=device, dtype=dtype),
        gyro_bias=problem.state_i.gyro_bias.to(device=device, dtype=dtype),
    )
    state_j = NavigationState(
        pose_WB=fixed_j.to(device=device, dtype=dtype),
        velocity_W=problem.state_j.velocity_W.to(device=device, dtype=dtype),
        acc_bias=problem.state_j.acc_bias.to(device=device, dtype=dtype),
        gyro_bias=problem.state_j.gyro_bias.to(device=device, dtype=dtype),
    )
    prior = problem.prior_i.to(device=device, dtype=dtype)
    imu = problem.imu.to(device=device, dtype=dtype)
    visual = problem.visual_pose.to(device=device, dtype=dtype)
    active_state_mask = torch.zeros(STATE_DOF, dtype=torch.bool, device=device)
    active_state_mask[6:9] = True
    active_state_mask[9:12] = bool(problem.optimize_acc_bias)
    active_state_mask[12:15] = bool(problem.optimize_gyro_bias)
    active_pair_mask = torch.cat([active_state_mask, active_state_mask])
    active_pair_indices = torch.nonzero(active_pair_mask, as_tuple=False).reshape(-1)

    _, _, hessian, gradient, blocks = _linearize(
        state_i, state_j, prior, imu, visual, self.covariance_eigenvalue_floor
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
    for iteration in range(self.max_iterations):
        iterations = iteration + 1
        _, _, hessian, gradient, _ = _linearize(
            state_i, state_j, prior, imu, visual, self.covariance_eigenvalue_floor
        )
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
            raise FloatingPointError("C6 fixed-pose LM produced a non-finite step")
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
            damping = max(damping * 0.25, 1.0e-12)
            if abs(previous_cost - current_cost) <= self.cost_tolerance:
                converged = True
                convergence_reason = "cost_tolerance"
                break
        else:
            rejected_steps += 1
            damping = min(damping * 10.0, 1.0e12)

    _, _, hessian, gradient, blocks = _linearize(
        state_i, state_j, prior, imu, visual, self.covariance_eigenvalue_floor
    )
    prior_j = marginalize_source_state(
        state_i,
        state_j,
        hessian,
        gradient,
        eigenvalue_floor=self.marginalization_eigenvalue_floor,
        active_state_mask=active_state_mask,
    )
    visual_white = blocks["visual_pose_unweighted"]
    visual_norm = float(torch.linalg.vector_norm(visual_white).detach().cpu().item())
    final_gradient_inf_norm = float(
        gradient[active_pair_indices].abs().max().detach().cpu().item()
    )
    delta = max(float(visual.huber_delta), 1.0e-12)
    visual_cost = (
        0.5 * visual_norm**2
        if visual_norm <= delta
        else delta * visual_norm - 0.5 * delta**2
    )
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
        visual_pose_cost=float(visual_cost),
        hessian=hessian.detach(),
        gradient=gradient.detach(),
        final_step_norm=final_step_norm,
        final_gradient_inf_norm=final_gradient_inf_norm,
        convergence_reason=convergence_reason,
        accepted_steps=accepted_steps,
        rejected_steps=rejected_steps,
    )


def finite_distribution(values: np.ndarray) -> dict[str, float | None]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {name: None for name in ("min", "median", "mean", "p95", "max")}
    return {
        "min": float(values.min()),
        "median": float(np.median(values)),
        "mean": float(values.mean()),
        "p95": float(np.quantile(values, 0.95)),
        "max": float(values.max()),
    }


def summarize_all() -> None:
    rows: list[dict[str, Any]] = []
    payload: dict[str, Any] = {}
    for case, spec in CASES.items():
        root = base.mode_root("short", case)
        summary_path = root / "summary.json"
        if not summary_path.exists():
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        gate_path = root / "visual_gate_runtime.csv"
        gate = pd.read_csv(gate_path) if gate_path.exists() else pd.DataFrame()
        diagnostics_path = Path(summary["artifacts"]["diagnostics"])
        diagnostics = pd.read_csv(diagnostics_path)
        diagnostics = diagnostics[
            (diagnostics["frame_i"] >= base.ACTIVE_START_FRAME)
            & (diagnostics["frame_j"] < base.SHORT_FRAME_LIMIT)
        ]
        metrics = summary["metrics"]
        correction = summary["visual_relative_pose_correction"]
        row = {
            "case": case,
            "description": spec["description"],
            "covariance_scale": spec["covariance_scale"],
            "huber_delta": spec["huber_delta"],
            **metrics,
            "visual_correction_translation_median_m": correction["translation_norm_m"]["median"],
            "visual_correction_translation_p95_m": correction["translation_norm_m"]["p95"],
            "visual_correction_rotation_median_rad": correction["rotation_norm_rad"]["median"],
            "iterations_mean": summary["convergence"]["iterations"]["mean"],
            "converged_rate": summary["convergence"]["converged_rate"],
        }
        for factor in ("prior", "imu", "bias", "pose"):
            row[f"{factor}_cost_sum_after"] = summary["factor_costs"][factor]["sum_after"]
        if not gate.empty:
            counts = gate["action"].astype(str).value_counts()
            row["gate_accept_count"] = int(counts.get("accept", 0))
            row["gate_downweight_count"] = int(counts.get("downweight", 0))
            row["gate_reject_count"] = int(sum(value for key, value in counts.items() if key.startswith("reject:")))
            row["gate_inflation_median"] = float(gate["covariance_inflation"].median())
            row["gate_inflation_p95"] = float(gate["covariance_inflation"].quantile(0.95))
        rows.append(row)
        payload[case] = {
            "spec": spec,
            "summary": summary,
            "runtime_gate_action_counts": (
                gate["action"].astype(str).value_counts().to_dict() if not gate.empty else {}
            ),
            "runtime_gate_inflation": (
                finite_distribution(gate["covariance_inflation"].to_numpy()) if not gate.empty else {}
            ),
            "postsolve_gate_action_counts": (
                diagnostics["visual_pose_gate_action"].astype(str).value_counts().to_dict()
                if "visual_pose_gate_action" in diagnostics else {}
            ),
        }
    c7_path = base.mode_root("short", "C7") / "summary.json"
    if c7_path.exists():
        c7 = json.loads(c7_path.read_text(encoding="utf-8"))
        rows.append(
            {
                "case": "C7",
                "description": c7["description"],
                "covariance_scale": math.nan,
                "huber_delta": math.nan,
                **c7["metrics"],
                "velocity_truth_rmse_mps": math.nan,
                "acc_bias_truth_rmse_mps2": math.nan,
                "gyro_bias_truth_rmse_radps": math.nan,
                "visual_correction_translation_median_m": 0.0,
                "visual_correction_translation_p95_m": 0.0,
                "visual_correction_rotation_median_rad": 0.0,
                "iterations_mean": 0.0,
                "converged_rate": 1.0,
                "prior_cost_sum_after": 0.0,
                "imu_cost_sum_after": 0.0,
                "bias_cost_sum_after": 0.0,
                "pose_cost_sum_after": 0.0,
            }
        )
        payload["C7"] = c7
    if rows:
        pd.DataFrame(rows).sort_values("case").to_csv(OUT / "backend_information_strength_summary.csv", index=False)
    write_json(OUT / "backend_information_strength_summary.json", payload)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--prepare", action="store_true")
    action.add_argument("--run-case", choices=sorted(CASES))
    action.add_argument("--summarize", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.prepare:
        prepare_all()
    elif args.run_case:
        run_case(args.run_case, force=args.force)
    else:
        summarize_all()


if __name__ == "__main__":
    main()
