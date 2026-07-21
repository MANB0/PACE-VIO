#!/usr/bin/env python3
"""Offline counterfactual audit for direct-UVD U1 visual update modes.

The production replay is executed once to capture the exact incoming two-state
problem at every active edge. Candidate solves are then run from cloned problem
snapshots. Candidate priors never flow back into the production replay.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import io
import json
import math
import multiprocessing as mp
import runpy
import shutil
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pypose as pp
import torch
import yaml
from scipy.spatial.transform import Rotation


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import Scripts.audit_circle_translation_oracle as truth_tools  # noqa: E402
import Scripts.run_circle_translation_oracles as replay_base  # noqa: E402
import Scripts.run_direct_uvd_short_experiments as direct_uvd  # noqa: E402
from Module.Optimization.TwoFramePGO.Optimizer import (  # noqa: E402
    TwoFrame_PGO,
    _imu_propagated_pose,
)
from Utility.TwoStateVIO import (  # noqa: E402
    ImuPreintegrationFactor,
    NavigationState,
    SquareRootPrior,
    TwoStateVIOProblem,
    TwoStateVIOResult,
    TwoStateVIOSolver,
    UVDFactor,
    _whiten_rows,
    state_boxminus,
    visual_whitened_residuals,
)
from Utility.Point import point2pixel_NED  # noqa: E402


OUTPUT = ROOT / "analysis_u1_counterfactual_branches_20260719"
PACKET_PATH = OUTPUT / "captured_u1_problems.pt"
ACTIVE_START_FRAME = truth_tools.ACTIVE_START_FRAME
FRAME_LIMIT = replay_base.SHORT_FRAME_LIMIT
MODES = ("full", "rotation_only", "translation_only", "no_visual", "alt_rt")
NED_TO_NWU = truth_tools.NWU_TO_NED

ORIGINAL_SOLVE = TwoStateVIOSolver.solve
ORIGINAL_OPTIMIZE = TwoFrame_PGO._optimize_two_state_fixed_lag
CAPTURED: list[dict[str, Any]] = []
CURRENT_EDGE: dict[str, Any] = {}
SOLVER_SETTINGS: dict[str, float | int] = {}
WORKER_SOLVER: TwoStateVIOSolver | None = None
WORKER_EDGES: list[dict[str, Any]] = []


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force-capture", action="store_true")
    parser.add_argument("--skip-capture", action="store_true")
    parser.add_argument("--max-edges", type=int, default=300)
    parser.add_argument("--lookahead", type=int, default=5)
    parser.add_argument("--lookahead-stride", type=int, default=10)
    parser.add_argument("--top-seeds-per-mode", type=int, default=5)
    parser.add_argument("--skip-lookahead", action="store_true")
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--immediate-batch-edges", type=int, default=30)
    return parser.parse_args()


def jsonify(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): jsonify(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonify(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(jsonify(value), indent=2, ensure_ascii=False), encoding="utf-8"
    )


def serialize_torch(value: Any) -> bytes:
    stream = io.BytesIO()
    torch.save(value, stream)
    return stream.getvalue()


def deserialize_torch(payload: bytes) -> Any:
    return torch.load(io.BytesIO(payload), map_location="cpu", weights_only=False)


def clone_state(state: NavigationState) -> NavigationState:
    return state.to(device=torch.device("cpu"), dtype=torch.float64).detach()


def clone_prior(prior: SquareRootPrior) -> SquareRootPrior:
    return prior.to(device=torch.device("cpu"), dtype=torch.float64)


def clone_imu(imu: ImuPreintegrationFactor) -> ImuPreintegrationFactor:
    return imu.to(device=torch.device("cpu"), dtype=torch.float64)


def clone_visual(visual: UVDFactor) -> UVDFactor:
    if not isinstance(visual, UVDFactor):
        raise TypeError("counterfactual capture requires a direct-UVD factor")
    return visual.to(device=torch.device("cpu"), dtype=torch.float64)


def clone_problem(problem: TwoStateVIOProblem) -> TwoStateVIOProblem:
    return TwoStateVIOProblem(
        state_i=clone_state(problem.state_i),
        state_j=clone_state(problem.state_j),
        prior_i=clone_prior(problem.prior_i),
        imu=clone_imu(problem.imu),
        visual_pose=clone_visual(problem.visual_pose),
        optimize_acc_bias=bool(problem.optimize_acc_bias),
        optimize_gyro_bias=bool(problem.optimize_gyro_bias),
    )


def result_payload(result: TwoStateVIOResult) -> dict[str, Any]:
    return {
        "state_i": clone_state(result.state_i),
        "state_j": clone_state(result.state_j),
        "prior_j": clone_prior(result.prior_j),
        "converged": bool(result.converged),
        "iterations": int(result.iterations),
        "initial_cost": float(result.initial_cost),
        "final_cost": float(result.final_cost),
        "prior_cost": float(result.prior_cost),
        "imu_cost": float(result.imu_cost),
        "bias_cost": float(result.bias_cost),
        "visual_pose_cost": float(result.visual_pose_cost),
        "final_step_norm": float(result.final_step_norm),
        "final_gradient_inf_norm": float(result.final_gradient_inf_norm),
        "convergence_reason": str(result.convergence_reason),
    }


def capture_optimize(context: dict, graph_data):
    CURRENT_EDGE.clear()
    CURRENT_EDGE.update(
        {
            "frame_i": int(graph_data.from_idx.reshape(-1)[0].item()),
            "frame_j": int(graph_data.frame_idx.reshape(-1)[0].item()),
            "num_observations": int(graph_data.num_observations),
            "coverage": graph_data.visual_keypoint_coverage,
            "depth_spread": graph_data.visual_depth_spread,
            "visual_obs_cov_mean": graph_data.visual_obs_cov_mean,
            "solver_calls": 0,
        }
    )
    return ORIGINAL_OPTIMIZE(context, graph_data)


def capture_solve(self: TwoStateVIOSolver, problem: TwoStateVIOProblem):
    CURRENT_EDGE["solver_calls"] = int(CURRENT_EDGE.get("solver_calls", 0)) + 1
    normalized = clone_problem(problem)
    result = ORIGINAL_SOLVE(self, problem)
    frame_i = int(CURRENT_EDGE.get("frame_i", -1))
    frame_j = int(CURRENT_EDGE.get("frame_j", -1))
    if frame_i >= ACTIVE_START_FRAME and CURRENT_EDGE["solver_calls"] == 1:
        if not SOLVER_SETTINGS:
            SOLVER_SETTINGS.update(
                {
                    "max_iterations": int(self.max_iterations),
                    "initial_damping": float(self.initial_damping),
                    "step_tolerance": float(self.step_tolerance),
                    "cost_tolerance": float(self.cost_tolerance),
                    "covariance_eigenvalue_floor": float(
                        self.covariance_eigenvalue_floor
                    ),
                    "marginalization_eigenvalue_floor": float(
                        self.marginalization_eigenvalue_floor
                    ),
                }
            )
        CAPTURED.append(
            {
                "frame_i": frame_i,
                "frame_j": frame_j,
                "problem": normalized,
                "baseline": result_payload(result),
                "quality": {
                    key: CURRENT_EDGE.get(key)
                    for key in (
                        "num_observations",
                        "coverage",
                        "depth_spread",
                        "visual_obs_cov_mean",
                    )
                },
            }
        )
    return result


def prepare_capture_config() -> Path:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load(direct_uvd.BASELINE_CONFIG.read_text(encoding="utf-8"))
    optimizer = config["Odometry"]["optimizer"]["args"]
    optimizer["two_state_visual_factor_mode"] = "direct_uvd"
    optimizer["two_state_warm_start"] = "macvo_pose"
    optimizer["two_state_uvd_huber_delta"] = 0.1
    odometry = config["Odometry"]["args"]
    odometry["mapping"] = False
    odometry["two_state_covariance_mode"] = "current_independent_step"
    path = OUTPUT / "effective_u1_odometry.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def capture_production_problems(*, force: bool) -> list[dict[str, Any]]:
    if PACKET_PATH.exists() and not force:
        payload = torch.load(PACKET_PATH, map_location="cpu", weights_only=False)
        SOLVER_SETTINGS.update(payload["solver_settings"])
        return payload["edges"]

    run_root = OUTPUT / "capture_run_result"
    if run_root.exists():
        if not force:
            raise FileExistsError(f"{run_root} exists without {PACKET_PATH}")
        shutil.rmtree(run_root)
    CAPTURED.clear()
    SOLVER_SETTINGS.clear()
    cache = direct_uvd.prepare_cache()
    config = prepare_capture_config()
    TwoStateVIOSolver.solve = capture_solve
    TwoFrame_PGO._optimize_two_state_fixed_lag = staticmethod(capture_optimize)
    started = time.perf_counter()
    try:
        sys.argv = [
            str(ROOT / "MACVO.py"),
            "--odom",
            str(config),
            "--data",
            str(replay_base.DATA_CONFIG_SOURCE),
            "--resultRoot",
            str(run_root),
            "--visual-cache-mode",
            "replay",
            "--visual-cache-path",
            str(cache),
            "--seq_to",
            str(FRAME_LIMIT),
        ]
        runpy.run_path(str(ROOT / "MACVO.py"), run_name="__main__")
    finally:
        TwoStateVIOSolver.solve = ORIGINAL_SOLVE
        TwoFrame_PGO._optimize_two_state_fixed_lag = staticmethod(ORIGINAL_OPTIMIZE)
    elapsed = time.perf_counter() - started
    if len(CAPTURED) != FRAME_LIMIT - ACTIVE_START_FRAME - 1:
        raise RuntimeError(
            f"expected {FRAME_LIMIT - ACTIVE_START_FRAME - 1} captured edges, "
            f"got {len(CAPTURED)}"
        )
    payload = {
        "schema_version": 1,
        "active_start_frame": ACTIVE_START_FRAME,
        "frame_limit": FRAME_LIMIT,
        "solver_settings": dict(SOLVER_SETTINGS),
        "capture_runtime_seconds": elapsed,
        "edges": CAPTURED,
    }
    torch.save(payload, PACKET_PATH)
    write_json(
        OUTPUT / "capture_manifest.json",
        {
            key: value
            for key, value in payload.items()
            if key not in {"edges"}
        }
        | {"edge_count": len(CAPTURED), "packet_path": str(PACKET_PATH)},
    )
    return CAPTURED


def make_solver() -> TwoStateVIOSolver:
    if not SOLVER_SETTINGS:
        raise RuntimeError("solver settings were not captured")
    return TwoStateVIOSolver(**SOLVER_SETTINGS)


def initialize_worker(
    settings: dict[str, float | int],
    edges_payload: bytes | None = None,
) -> None:
    global WORKER_SOLVER, WORKER_EDGES
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    WORKER_SOLVER = TwoStateVIOSolver(**settings)
    WORKER_EDGES = [] if edges_payload is None else deserialize_torch(edges_payload)


def imu_anchor_relative_camera(problem: TwoStateVIOProblem) -> torch.Tensor:
    visual = clone_visual(problem.visual_pose)
    pose_WBj = pp.SE3(_imu_propagated_pose(problem.state_i, problem.imu))
    extrinsic_CI = pp.SE3(visual.extrinsic_CI)
    pose_WCi = pp.SE3(problem.state_i.pose_WB) @ extrinsic_CI.Inv()
    pose_WCj = pose_WBj @ extrinsic_CI.Inv()
    return (pose_WCj.Inv() @ pose_WCi).tensor().detach()


def mode_factor(problem: TwoStateVIOProblem, mode: str) -> UVDFactor:
    visual = clone_visual(problem.visual_pose)
    if mode == "full":
        return replace(
            visual, optimization_mode="full", anchor_relative_CjCi=None
        )
    if mode == "no_visual":
        return replace(
            visual, optimization_mode="no_visual", anchor_relative_CjCi=None
        )
    if mode not in {"rotation_only", "translation_only"}:
        raise ValueError(f"unsupported single-stage mode: {mode}")
    return replace(
        visual,
        optimization_mode=mode,
        anchor_relative_CjCi=imu_anchor_relative_camera(problem),
    )


def solve_mode(
    solver: TwoStateVIOSolver, problem: TwoStateVIOProblem, mode: str
) -> tuple[TwoStateVIOResult, dict[str, Any]]:
    problem = clone_problem(problem)
    if mode != "alt_rt":
        result = solver.solve(
            replace(problem, visual_pose=mode_factor(problem, mode))
        )
        return result, {"stage_iterations": [int(result.iterations)]}

    stage_results: list[TwoStateVIOResult] = []
    stage_problem = problem
    for stage in ("rotation_only", "translation_only", "full"):
        stage_problem = replace(
            problem,
            state_i=stage_problem.state_i,
            state_j=stage_problem.state_j,
            visual_pose=mode_factor(problem, stage),
        )
        result = solver.solve(stage_problem)
        stage_results.append(result)
        stage_problem = replace(
            problem,
            state_i=result.state_i,
            state_j=result.state_j,
        )
    return stage_results[-1], {
        "stage_iterations": [int(result.iterations) for result in stage_results],
        "stage_converged": [bool(result.converged) for result in stage_results],
    }


def immediate_worker(
    payload: tuple[int, str, bytes]
) -> tuple[int, str, bytes, dict[str, Any], dict[str, float] | None]:
    if WORKER_SOLVER is None:
        raise RuntimeError("counterfactual worker solver was not initialized")
    index, mode, problem_payload = payload
    problem = deserialize_torch(problem_payload)
    features = visual_information_features(problem) if mode == "full" else None
    result, stages = solve_mode(WORKER_SOLVER, problem, mode)
    return index, mode, serialize_torch(result), stages, features


def bounded_process_map(
    worker,
    jobs: list[Any],
    *,
    workers: int,
    initargs: tuple[Any, ...],
    label: str,
    report_every: int,
    started: float,
) -> list[Any]:
    completed: list[Any] = []
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=workers,
        mp_context=mp.get_context("spawn"),
        initializer=initialize_worker,
        initargs=initargs,
    ) as executor:
        iterator = iter(jobs)
        pending: set[concurrent.futures.Future] = set()
        for _ in range(min(len(jobs), workers * 2)):
            try:
                pending.add(executor.submit(worker, next(iterator)))
            except StopIteration:
                break
        while pending:
            done, pending = concurrent.futures.wait(
                pending,
                return_when=concurrent.futures.FIRST_COMPLETED,
            )
            for future in done:
                completed.append(future.result())
                try:
                    pending.add(executor.submit(worker, next(iterator)))
                except StopIteration:
                    pass
                count = len(completed)
                if count % report_every == 0 or count == len(jobs):
                    print(
                        f"[{label}] {count}/{len(jobs)} branches "
                        f"({time.perf_counter() - started:.1f}s)",
                        flush=True,
                    )
    return completed


def bias_corrected_delta_velocity(
    state_i: NavigationState, imu: ImuPreintegrationFactor
) -> torch.Tensor:
    db = torch.cat(
        [
            state_i.acc_bias.reshape(3) - imu.linearized_acc_bias.reshape(3),
            state_i.gyro_bias.reshape(3) - imu.linearized_gyro_bias.reshape(3),
        ]
    )
    correction = imu.bias_jacobian.reshape(9, 6) @ db
    return imu.delta_velocity.reshape(3) + correction[3:6]


def future_problem(
    template: TwoStateVIOProblem,
    previous: TwoStateVIOResult,
) -> TwoStateVIOProblem:
    template = clone_problem(template)
    state_i = clone_state(previous.state_j)
    template_relative = pp.SE3(template.state_i.pose_WB).Inv() @ pp.SE3(
        template.state_j.pose_WB
    )
    pose_j = (pp.SE3(state_i.pose_WB) @ template_relative).tensor()
    rotation_i = pp.SE3(state_i.pose_WB).rotation().matrix().reshape(3, 3)
    gravity = torch.zeros(3, dtype=torch.float64)
    if str(template.imu.gravity_handling) == "residual":
        if template.imu.gravity_world is None:
            raise ValueError("future propagation requires gravity_world")
        gravity = template.imu.gravity_world.reshape(3)
    velocity_j = (
        state_i.velocity_W
        + gravity * float(template.imu.dt)
        + rotation_i @ bias_corrected_delta_velocity(state_i, template.imu)
    )
    state_j = NavigationState(
        pose_WB=pose_j,
        velocity_W=velocity_j,
        acc_bias=state_i.acc_bias.clone(),
        gyro_bias=state_i.gyro_bias.clone(),
    )
    return replace(
        template,
        state_i=state_i,
        state_j=state_j,
        prior_i=clone_prior(previous.prior_j),
        visual_pose=replace(
            clone_visual(template.visual_pose),
            optimization_mode="full",
            anchor_relative_CjCi=None,
        ),
    )


def pose_matrix(state: NavigationState) -> np.ndarray:
    return (
        pp.SE3(state.pose_WB)
        .matrix()
        .reshape(4, 4)
        .detach()
        .cpu()
        .numpy()
        .astype(np.float64)
    )


def invert(transform: np.ndarray) -> np.ndarray:
    return truth_tools.invert_transform(transform)


def relative_error(
    estimate_i: np.ndarray,
    estimate_j: np.ndarray,
    truth_i: np.ndarray,
    truth_j: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    estimate = invert(estimate_i) @ estimate_j
    truth = invert(truth_i) @ truth_j
    error = invert(truth) @ estimate
    return error[:3, 3], truth_tools.rotation_log(error[:3, :3]), error


def load_truth_contract() -> dict[str, Any]:
    dataset = truth_tools.DEFAULT_DATASET
    metadata = json.loads((dataset / "metadata.json").read_text(encoding="utf-8"))
    ref, truth_body = truth_tools.load_truth(dataset)
    lever_nwu = np.asarray(
        metadata["extrinsics"]["T_body_imu"]["translation_body_nwu_m"],
        dtype=np.float64,
    )
    lever_internal = truth_tools.NWU_TO_NED @ lever_nwu
    transform_BI = truth_tools.make_transform(np.eye(3), lever_internal)
    truth_imu = np.stack([pose @ transform_BI for pose in truth_body])
    timestamps = ref["timestamp"].to_numpy(np.int64)
    velocity = (
        ref[["vx", "vy", "vz"]].to_numpy(np.float64)
        @ replay_base.FLU_TO_NED.T
    )
    decomposition = pd.read_csv(dataset / "imu_truth_decomposition.csv")
    imu_time = decomposition["timestamp"].to_numpy(np.int64)
    ba = replay_base.interpolate_rows(
        imu_time,
        decomposition[["acc_bias_x", "acc_bias_y", "acc_bias_z"]].to_numpy(
            np.float64
        ),
        timestamps,
    ) @ replay_base.FLU_TO_NED.T
    bg = replay_base.interpolate_rows(
        imu_time,
        decomposition[["gyro_bias_x", "gyro_bias_y", "gyro_bias_z"]].to_numpy(
            np.float64
        ),
        timestamps,
    ) @ replay_base.FLU_TO_NED.T
    return {
        "dataset": str(dataset),
        "timestamps": timestamps,
        "pose": truth_imu,
        "velocity": velocity,
        "ba": ba,
        "bg": bg,
        "lever_body_to_imu_nwu": lever_nwu,
    }


def visual_information_features(problem: TwoStateVIOProblem) -> dict[str, float]:
    visual = clone_visual(problem.visual_pose)
    white = visual_whitened_residuals(
        problem.state_i, problem.state_j, visual, 1.0e-12
    )
    norms = torch.linalg.vector_norm(white, dim=-1)
    relative = (
        pp.SE3(problem.state_j.pose_WB)
        @ pp.SE3(visual.extrinsic_CI).Inv()
    ).Inv() @ (
        pp.SE3(problem.state_i.pose_WB) @ pp.SE3(visual.extrinsic_CI).Inv()
    )
    zero = torch.zeros(6, dtype=torch.float64, requires_grad=True)

    def residual(increment: torch.Tensor) -> torch.Tensor:
        candidate = relative @ pp.se3(increment.reshape(1, 6)).Exp()
        predicted = candidate.Act(visual.points_Ci)
        predicted_uv = point2pixel_NED(predicted, visual.intrinsic)
        predicted_disp = (
            visual.intrinsic[0, 0] * float(visual.baseline) / predicted[:, 0:1]
        )
        raw = torch.cat(
            [predicted_uv - visual.target_uv, predicted_disp - visual.target_disparity],
            dim=-1,
        )
        whitened = _whiten_rows(raw, visual.covariance_uvd, 1.0e-12)
        row_norm = torch.linalg.vector_norm(whitened, dim=-1)
        delta = float(visual.huber_delta)
        weight = torch.where(
            row_norm <= delta,
            torch.ones_like(row_norm),
            torch.as_tensor(delta, dtype=torch.float64)
            / row_norm.clamp_min(1.0e-12),
        ).detach()
        return (weight.sqrt().unsqueeze(-1) * whitened).reshape(-1)

    jacobian = torch.autograd.functional.jacobian(
        residual, zero, create_graph=False, vectorize=True
    ).detach()
    hessian = 0.5 * (jacobian.mT @ jacobian + (jacobian.mT @ jacobian).mT)
    h_tt = hessian[:3, :3]
    h_rr = hessian[3:, 3:]
    h_tr = hessian[:3, 3:]

    def eigen_stats(matrix: torch.Tensor) -> tuple[float, float, float]:
        values = torch.linalg.eigvalsh(matrix).clamp_min(0.0)
        minimum = float(values.min().item())
        maximum = float(values.max().item())
        condition = maximum / max(minimum, 1.0e-15)
        return minimum, maximum, condition

    t_min, t_max, t_condition = eigen_stats(h_tt)
    r_min, r_max, r_condition = eigen_stats(h_rr)
    t_values, t_vectors = torch.linalg.eigh(h_tt)
    r_values, r_vectors = torch.linalg.eigh(h_rr)
    t_inv_sqrt = t_vectors @ torch.diag(
        t_values.clamp_min(1.0e-12).rsqrt()
    ) @ t_vectors.mT
    r_inv_sqrt = r_vectors @ torch.diag(
        r_values.clamp_min(1.0e-12).rsqrt()
    ) @ r_vectors.mT
    coupling = float(
        torch.linalg.svdvals(t_inv_sqrt @ h_tr @ r_inv_sqrt).max().item()
    )
    return {
        "point_count": int(visual.points_Ci.shape[0]),
        "initial_inlier_ratio": float((norms <= visual.huber_delta).double().mean()),
        "initial_mean_mahalanobis_sq": float(norms.square().mean()),
        "initial_whitened_norm": float(torch.linalg.vector_norm(white)),
        "h_tt_min_eigenvalue": t_min,
        "h_tt_max_eigenvalue": t_max,
        "h_tt_condition": t_condition,
        "h_rr_min_eigenvalue": r_min,
        "h_rr_max_eigenvalue": r_max,
        "h_rr_condition": r_condition,
        "h_tr_normalized_max_singular": coupling,
    }


def outcome_row(
    *,
    edge: dict[str, Any],
    mode: str,
    result: TwoStateVIOResult,
    stages: dict[str, Any],
    truth: dict[str, Any],
    features: dict[str, Any],
    baseline_replay_error: float | None = None,
) -> dict[str, Any]:
    frame_i = int(edge["frame_i"])
    frame_j = int(edge["frame_j"])
    truth_pose = truth["pose"]
    translation, rotation, _ = relative_error(
        pose_matrix(result.state_i),
        pose_matrix(result.state_j),
        truth_pose[frame_i],
        truth_pose[frame_j],
    )
    state_j = result.state_j
    velocity_error = state_j.velocity_W.detach().numpy() - truth["velocity"][frame_j]
    ba_error = state_j.acc_bias.detach().numpy() - truth["ba"][frame_j]
    bg_error = state_j.gyro_bias.detach().numpy() - truth["bg"][frame_j]
    row: dict[str, Any] = {
        "frame_i": frame_i,
        "frame_j": frame_j,
        "timestamp_j_ns": int(truth["timestamps"][frame_j]),
        "mode": mode,
        "translation_error_x": float(translation[0]),
        "translation_error_y": float(translation[1]),
        "translation_error_z": float(translation[2]),
        "translation_error_norm": float(np.linalg.norm(translation)),
        "rotation_error_x": float(rotation[0]),
        "rotation_error_y": float(rotation[1]),
        "rotation_error_z": float(rotation[2]),
        "rotation_error_norm": float(np.linalg.norm(rotation)),
        "velocity_error_norm": float(np.linalg.norm(velocity_error)),
        "acc_bias_error_norm": float(np.linalg.norm(ba_error)),
        "gyro_bias_error_norm": float(np.linalg.norm(bg_error)),
        "initial_cost": float(result.initial_cost),
        "final_cost": float(result.final_cost),
        "prior_cost": float(result.prior_cost),
        "imu_cost": float(result.imu_cost),
        "bias_cost": float(result.bias_cost),
        "visual_cost": float(result.visual_pose_cost),
        "converged": bool(result.converged),
        "iterations": int(result.iterations),
        "stage_iterations": ",".join(map(str, stages["stage_iterations"])),
        "final_step_norm": float(result.final_step_norm),
        "final_gradient_inf_norm": float(result.final_gradient_inf_norm),
        "convergence_reason": str(result.convergence_reason),
        "baseline_full_replay_state_error": baseline_replay_error,
        **edge["quality"],
        **features,
    }
    return row


def run_immediate(
    edges: list[dict[str, Any]],
    truth: dict[str, Any],
    *,
    workers: int,
    batch_edges: int,
) -> tuple[pd.DataFrame, dict[tuple[int, str], TwoStateVIOResult]]:
    rows: list[dict[str, Any]] = []
    results: dict[tuple[int, str], TwoStateVIOResult] = {}
    stages_by_key: dict[tuple[int, str], dict[str, Any]] = {}
    started = time.perf_counter()
    features_by_index: dict[int, dict[str, float]] = {}
    worker_count = max(1, int(workers))
    batch_size = max(1, int(batch_edges))
    if worker_count == 1:
        initialize_worker(dict(SOLVER_SETTINGS))
    for batch_start in range(0, len(edges), batch_size):
        batch_stop = min(batch_start + batch_size, len(edges))
        batch_jobs: list[tuple[int, str, bytes]] = []
        for index in range(batch_start, batch_stop):
            problem = clone_problem(edges[index]["problem"])
            problem_payload = serialize_torch(problem)
            for mode in MODES:
                batch_jobs.append((index, mode, problem_payload))
        if worker_count == 1:
            completed = [immediate_worker(job) for job in batch_jobs]
        else:
            completed = bounded_process_map(
                immediate_worker,
                batch_jobs,
                workers=worker_count,
                initargs=(dict(SOLVER_SETTINGS), None),
                label=f"immediate edges {batch_start}:{batch_stop}",
                report_every=50,
                started=started,
            )
        for index, mode, result_payload, stages, features in completed:
            result = deserialize_torch(result_payload)
            results[(index, mode)] = result
            stages_by_key[(index, mode)] = stages
            if features is not None:
                features_by_index[index] = features
        print(
            f"[immediate] completed edges {batch_start}:{batch_stop} "
            f"({time.perf_counter() - started:.1f}s)",
            flush=True,
        )

    for index, edge in enumerate(edges):
        for mode in MODES:
            result = results[(index, mode)]
            replay_error = None
            if mode == "full":
                baseline_state = edge["baseline"]["state_j"]
                replay_error = float(
                    torch.linalg.vector_norm(
                        state_boxminus(result.state_j, baseline_state)
                    ).item()
                )
            rows.append(
                outcome_row(
                    edge=edge,
                    mode=mode,
                    result=result,
                    stages=stages_by_key[(index, mode)],
                    truth=truth,
                    features=features_by_index[index],
                    baseline_replay_error=replay_error,
                )
            )
    frame = pd.DataFrame(rows)
    frame.to_csv(OUTPUT / "immediate_counterfactual_per_edge.csv", index=False)
    return frame, results


def select_lookahead_seeds(
    immediate: pd.DataFrame,
    *,
    stride: int,
    top_per_mode: int,
) -> list[int]:
    pivot_t = immediate.pivot(
        index="frame_i", columns="mode", values="translation_error_norm"
    )
    pivot_r = immediate.pivot(
        index="frame_i", columns="mode", values="rotation_error_norm"
    )
    frame_indices = list(map(int, pivot_t.index))
    selected = set(frame_indices[:: max(int(stride), 1)])
    t_scale = max(float(pivot_t["full"].median()), 1.0e-12)
    r_scale = max(float(pivot_r["full"].median()), 1.0e-12)
    for mode in ("rotation_only", "translation_only", "no_visual", "alt_rt"):
        score = (
            (pivot_t["full"] - pivot_t[mode]) / t_scale
            + (pivot_r["full"] - pivot_r[mode]) / r_scale
        )
        for frame_i in score.nlargest(max(int(top_per_mode), 0)).index:
            selected.add(int(frame_i))
    return sorted(selected)


def run_lookahead(
    edges: list[dict[str, Any]],
    immediate_results: dict[tuple[int, str], TwoStateVIOResult],
    truth: dict[str, Any],
    seeds: list[int],
    horizon: int,
    workers: int,
) -> pd.DataFrame:
    global WORKER_EDGES
    WORKER_EDGES = edges
    index_by_frame = {int(edge["frame_i"]): index for index, edge in enumerate(edges)}
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    jobs = [
        (
            index_by_frame[frame_i],
            mode,
            serialize_torch(immediate_results[(index_by_frame[frame_i], mode)]),
            horizon,
        )
        for frame_i in seeds
        for mode in MODES
    ]
    worker_count = max(1, int(workers))
    if worker_count == 1:
        initialize_worker(dict(SOLVER_SETTINGS), serialize_torch(edges))
        completed = [lookahead_worker(job) for job in jobs]
    else:
        completed = bounded_process_map(
            lookahead_worker,
            jobs,
            workers=worker_count,
            initargs=(dict(SOLVER_SETTINGS), serialize_torch(edges)),
            label="lookahead",
            report_every=20,
            started=started,
        )
    for seed_index, mode, result_payload, actual_horizon, future_iterations in completed:
            result = deserialize_torch(result_payload)
            frame_i = int(edges[seed_index]["frame_i"])
            final_frame = int(edges[seed_index + actual_horizon]["frame_j"])
            translation, rotation, _ = relative_error(
                pose_matrix(immediate_results[(seed_index, mode)].state_i),
                pose_matrix(result.state_j),
                truth["pose"][frame_i],
                truth["pose"][final_frame],
            )
            rows.append(
                {
                    "seed_frame_i": frame_i,
                    "first_frame_j": int(edges[seed_index]["frame_j"]),
                    "final_frame_j": final_frame,
                    "mode": mode,
                    "requested_horizon": int(horizon),
                    "actual_horizon": actual_horizon,
                    "translation_error_norm": float(np.linalg.norm(translation)),
                    "rotation_error_norm": float(np.linalg.norm(rotation)),
                    "velocity_error_norm": float(
                        np.linalg.norm(
                            result.state_j.velocity_W.detach().numpy()
                            - truth["velocity"][final_frame]
                        )
                    ),
                    "acc_bias_error_norm": float(
                        np.linalg.norm(
                            result.state_j.acc_bias.detach().numpy()
                            - truth["ba"][final_frame]
                        )
                    ),
                    "gyro_bias_error_norm": float(
                        np.linalg.norm(
                            result.state_j.gyro_bias.detach().numpy()
                            - truth["bg"][final_frame]
                        )
                    ),
                    "future_iterations": future_iterations,
                    "converged": bool(result.converged),
                    "final_cost": float(result.final_cost),
                    "prior_cost": float(result.prior_cost),
                    "imu_cost": float(result.imu_cost),
                    "bias_cost": float(result.bias_cost),
                    "visual_cost": float(result.visual_pose_cost),
                }
            )
    frame = pd.DataFrame(rows)
    frame.to_csv(OUTPUT / "lookahead_counterfactual_per_seed.csv", index=False)
    return frame


def lookahead_worker(
    payload: tuple[int, str, bytes, int]
) -> tuple[int, str, bytes, int, int]:
    if WORKER_SOLVER is None:
        raise RuntimeError("counterfactual worker solver was not initialized")
    seed_index, mode, result_payload, horizon = payload
    result = deserialize_torch(result_payload)
    actual_horizon = 0
    future_iterations = 0
    for offset in range(1, horizon + 1):
        future_index = seed_index + offset
        if future_index >= len(WORKER_EDGES):
            break
        problem = future_problem(WORKER_EDGES[future_index]["problem"], result)
        result, _ = solve_mode(WORKER_SOLVER, problem, "full")
        future_iterations += int(result.iterations)
        actual_horizon += 1
    return (
        seed_index,
        mode,
        serialize_torch(result),
        actual_horizon,
        future_iterations,
    )


def strict_dominance_counts(frame: pd.DataFrame) -> dict[str, int]:
    pivot_t = frame.pivot(
        index=frame.columns[0], columns="mode", values="translation_error_norm"
    )
    pivot_r = frame.pivot(
        index=frame.columns[0], columns="mode", values="rotation_error_norm"
    )
    counts: dict[str, int] = {}
    for mode in MODES:
        if mode == "full":
            continue
        no_worse = (pivot_t[mode] <= pivot_t["full"] + 1.0e-12) & (
            pivot_r[mode] <= pivot_r["full"] + 1.0e-12
        )
        strictly_better = (pivot_t[mode] < pivot_t["full"] - 1.0e-12) | (
            pivot_r[mode] < pivot_r["full"] - 1.0e-12
        )
        counts[mode] = int((no_worse & strictly_better).sum())
    return counts


def distribution(values: pd.Series) -> dict[str, float]:
    array = values.to_numpy(np.float64)
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p95": float(np.quantile(array, 0.95)),
        "max": float(np.max(array)),
    }


def summarize(
    immediate: pd.DataFrame,
    lookahead: pd.DataFrame | None,
    *,
    edge_count: int,
    seeds: list[int],
) -> dict[str, Any]:
    mode_summary: dict[str, Any] = {}
    for mode in MODES:
        subset = immediate[immediate["mode"] == mode]
        mode_summary[mode] = {
            "edge_count": int(len(subset)),
            "translation_error_norm": distribution(subset["translation_error_norm"]),
            "rotation_error_norm": distribution(subset["rotation_error_norm"]),
            "velocity_error_norm": distribution(subset["velocity_error_norm"]),
            "acc_bias_error_norm": distribution(subset["acc_bias_error_norm"]),
            "gyro_bias_error_norm": distribution(subset["gyro_bias_error_norm"]),
            "convergence_rate": float(subset["converged"].mean()),
            "iterations": distribution(subset["iterations"]),
            "costs": {
                name: distribution(subset[name])
                for name in ("prior_cost", "imu_cost", "bias_cost", "visual_cost")
            },
        }
    summary: dict[str, Any] = {
        "contract": {
            "visual_baseline": "Direct UVD U1",
            "preintegration": "standard local-frame, captured from production replay",
            "state_prior": "full 15D incoming Schur prior retained for every candidate",
            "candidate_prior_isolation": True,
            "truth_usage": "offline scoring and seed selection only",
            "active_start_frame": ACTIVE_START_FRAME,
            "captured_edge_count": edge_count,
            "evaluated_edge_count": int(immediate["frame_i"].nunique()),
            "lookahead_seed_count": len(seeds),
        },
        "full_reproduction": {
            "max_state_boxminus_norm": float(
                immediate.loc[
                    immediate["mode"] == "full",
                    "baseline_full_replay_state_error",
                ].max()
            )
        },
        "immediate": {
            "modes": mode_summary,
            "strictly_dominates_full_count": strict_dominance_counts(immediate),
        },
    }
    if lookahead is not None and not lookahead.empty:
        delayed_modes: dict[str, Any] = {}
        for mode in MODES:
            subset = lookahead[lookahead["mode"] == mode]
            delayed_modes[mode] = {
                "seed_count": int(len(subset)),
                "translation_error_norm": distribution(
                    subset["translation_error_norm"]
                ),
                "rotation_error_norm": distribution(subset["rotation_error_norm"]),
                "velocity_error_norm": distribution(subset["velocity_error_norm"]),
                "acc_bias_error_norm": distribution(subset["acc_bias_error_norm"]),
                "gyro_bias_error_norm": distribution(subset["gyro_bias_error_norm"]),
                "convergence_rate": float(subset["converged"].mean()),
            }
        summary["lookahead"] = {
            "modes": delayed_modes,
            "strictly_dominates_full_count": strict_dominance_counts(lookahead),
        }
    write_json(OUTPUT / "u1_counterfactual_summary.json", summary)
    return summary


def build_report(summary: dict[str, Any]) -> None:
    immediate = summary["immediate"]
    delayed = summary.get("lookahead")
    lines = [
        "# U1 直接UVD五分支反事实报告",
        "",
        "## 审计契约",
        "",
        "- 基线：Direct UVD U1，standard local-frame preintegration。",
        "- 每个候选从完全相同的两状态、IMU因子和完整15维incoming prior开始。",
        "- 候选prior只在离线分支中传播，不写回生产回放。",
        "- GT仅用于离线评分和延迟种子选择。",
        f"- FULL复现最大状态差：`{summary['full_reproduction']['max_state_boxminus_norm']:.3e}`。",
        "",
        "## 即时结果",
        "",
        "| 模式 | t误差中位数(m) | r误差中位数(rad) | 速度误差中位数(m/s) | 收敛率 | 严格支配FULL边数 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    dominance = immediate["strictly_dominates_full_count"]
    for mode in MODES:
        item = immediate["modes"][mode]
        lines.append(
            "| {mode} | {t:.6g} | {r:.6g} | {v:.6g} | {c:.1%} | {d} |".format(
                mode=mode,
                t=item["translation_error_norm"]["median"],
                r=item["rotation_error_norm"]["median"],
                v=item["velocity_error_norm"]["median"],
                c=item["convergence_rate"],
                d=dominance.get(mode, 0),
            )
        )
    if delayed:
        lines.extend(
            [
                "",
                "## 五边延迟结果",
                "",
                "| 模式 | 累计t误差中位数(m) | 累计r误差中位数(rad) | 速度误差中位数(m/s) | 严格支配FULL种子数 |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        delayed_dominance = delayed["strictly_dominates_full_count"]
        for mode in MODES:
            item = delayed["modes"][mode]
            lines.append(
                "| {mode} | {t:.6g} | {r:.6g} | {v:.6g} | {d} |".format(
                    mode=mode,
                    t=item["translation_error_norm"]["median"],
                    r=item["rotation_error_norm"]["median"],
                    v=item["velocity_error_norm"]["median"],
                    d=delayed_dominance.get(mode, 0),
                )
            )
    lines.extend(
        [
            "",
            "## 解释边界",
            "",
            "- rotation_only固定的是视觉变换中的相对平移；IMU、bias和prior仍可更新完整状态。",
            "- translation_only固定视觉相对旋转。由于相对平移在目标相机坐标系表达，它对绝对姿态仍存在坐标耦合，不能解释成视觉旋转Jacobian严格为零。",
            "- alt_rt的中间阶段prior全部丢弃，最终FULL阶段只边缘化一次，避免重复计数当前measurement。",
            "- 本报告验证模式是否存在；是否能由非GT量在线预测需要在结果成立后另做shadow selector。",
            "",
            "## 原始产物",
            "",
            "- `immediate_counterfactual_per_edge.csv`",
            "- `lookahead_counterfactual_per_seed.csv`",
            "- `lookahead_seed_selection.csv`",
            "- `u1_counterfactual_summary.json`",
            "- `captured_u1_problems.pt`",
        ]
    )
    (OUTPUT / "u1_counterfactual_report_cn.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> int:
    args = parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if args.skip_capture and not PACKET_PATH.exists():
        raise FileNotFoundError(f"--skip-capture requested but {PACKET_PATH} is absent")
    edges = capture_production_problems(force=bool(args.force_capture))
    edges = edges[: max(1, min(int(args.max_edges), len(edges)))]
    truth = load_truth_contract()
    write_json(
        OUTPUT / "truth_contract.json",
        {
            "dataset": truth["dataset"],
            "body_to_imu_translation_nwu_m": truth["lever_body_to_imu_nwu"],
            "truth_reference": "ref_pose body/root origin translated to IMU origin",
            "estimate_reference": "NavigationState pose_WB at IMU/body origin",
            "alignment": "none; edge errors are gauge invariant",
        },
    )
    immediate, results = run_immediate(
        edges,
        truth,
        workers=int(args.workers),
        batch_edges=int(args.immediate_batch_edges),
    )
    seeds: list[int] = []
    lookahead: pd.DataFrame | None = None
    if not args.skip_lookahead:
        seeds = select_lookahead_seeds(
            immediate,
            stride=int(args.lookahead_stride),
            top_per_mode=int(args.top_seeds_per_mode),
        )
        pd.DataFrame({"seed_frame_i": seeds}).to_csv(
            OUTPUT / "lookahead_seed_selection.csv", index=False
        )
        lookahead = run_lookahead(
            edges,
            results,
            truth,
            seeds,
            max(int(args.lookahead), 0),
            int(args.workers),
        )
    summary = summarize(
        immediate,
        lookahead,
        edge_count=len(edges),
        seeds=seeds,
    )
    build_report(summary)
    print(OUTPUT / "u1_counterfactual_report_cn.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
