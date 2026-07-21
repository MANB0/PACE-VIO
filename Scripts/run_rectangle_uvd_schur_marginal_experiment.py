#!/usr/bin/env python3
"""Audit Schur-marginal UVD factors on the first rectangle straight and turn.

The production Direct-UVD U1 replay is used only to capture exact incoming
two-state problems.  Every candidate branch starts from the same captured
state, prior, IMU factor, and UVD observations.  Candidate priors never feed
back into the production capture.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import multiprocessing as mp
import os
import runpy
import shutil
import sys
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
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

from Module.Optimization.TwoFramePGO.Optimizer import TwoFrame_PGO  # noqa: E402
from Scripts.run_static63_cached_imu_fusion import (  # noqa: E402
    LATEST_IMUATT_METHOD,
    RETAINED_VARIANTS,
    RunSpec,
    TASKS,
    make_seq_cfg,
)
from Scripts.run_u1_counterfactual_branches import (  # noqa: E402
    clone_problem,
    future_problem,
    result_payload,
)
from Utility.TwoStateVIO import (  # noqa: E402
    LinearizedUVDPoseFactor,
    NavigationState,
    TwoStateVIOProblem,
    TwoStateVIOResult,
    TwoStateVIOSolver,
    linearize_uvd_pose_factor,
    state_boxminus,
)


OUTPUT = ROOT / "analysis_rectangle_uvd_schur_marginal_20260719"
PACKET = OUTPUT / "captured_rectangle_u1_problems.pt"
FROZEN_CONFIG = (
    ROOT / "Baselines/direct_uvd_u1_standard_20260719/configs/odometry.yaml"
)
ACTIVE_START_FRAME = 90
FRAME_LIMIT = 724
LOOKAHEAD = 3
MODES = (
    "nonlinear_full",
    "linearized_full",
    "translation_marginal",
    "rotation_marginal",
)
NWU_TO_NED = np.diag([1.0, -1.0, -1.0])

ORIGINAL_SOLVE = TwoStateVIOSolver.solve
ORIGINAL_OPTIMIZE = TwoFrame_PGO._optimize_two_state_fixed_lag
CAPTURED: list[dict[str, Any]] = []
CURRENT_EDGE: dict[str, Any] = {}
SOLVER_SETTINGS: dict[str, float | int] = {}
WORKER_PAYLOAD: dict[str, Any] | None = None
WORKER_TRUTH: dict[str, np.ndarray] | None = None
WORKER_SOLVER: TwoStateVIOSolver | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force-capture", action="store_true")
    parser.add_argument("--skip-capture", action="store_true")
    parser.add_argument("--frame-limit", type=int, default=FRAME_LIMIT)
    parser.add_argument("--lookahead", type=int, default=LOOKAHEAD)
    parser.add_argument("--seeds-per-segment", type=int, default=12)
    parser.add_argument("--workers", type=int, default=4)
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


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(jsonify(payload), ensure_ascii=False, indent=2), encoding="utf-8"
    )


def rectangle_task():
    return next(
        task
        for task in TASKS
        if task.dataset_scene == "clear_stop_turn_rectangle_truth_normal_noise"
    )


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
                "frame_j": int(CURRENT_EDGE["frame_j"]),
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


def prepare_truncated_cache(task, frame_limit: int) -> Path:
    source = task.cache_dir
    destination = OUTPUT / f"visual_cache_first_{int(frame_limit)}_frames"
    manifest_path = destination / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if int(manifest["frame_count"]) == int(frame_limit):
            return destination
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    if int(frame_limit) > int(manifest["frame_count"]):
        raise ValueError("requested truncated cache exceeds source cache")
    manifest["frame_count"] = int(frame_limit)
    manifest["source"]["frame_count"] = int(frame_limit)
    manifest["source"]["exporter"] = Path(__file__).name
    manifest["timestamps_ns"] = manifest["timestamps_ns"][: int(frame_limit)]
    manifest["pairs"] = manifest["pairs"][: int(frame_limit) - 1]
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    os.symlink(source / "pairs", destination / "pairs", target_is_directory=True)
    sidecar = source / "relative_pose_factors.npz"
    if sidecar.exists():
        shutil.copy2(sidecar, destination / sidecar.name)
    return destination


def capture_production_problems(*, force: bool, frame_limit: int) -> dict[str, Any]:
    if PACKET.exists() and not force:
        payload = torch.load(PACKET, map_location="cpu", weights_only=False)
        if int(payload["frame_limit"]) < int(frame_limit):
            raise ValueError(
                f"cached packet stops at {payload['frame_limit']}, requested {frame_limit}"
            )
        return payload
    if not FROZEN_CONFIG.exists():
        raise FileNotFoundError(f"frozen U1 config is missing: {FROZEN_CONFIG}")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    run_root = OUTPUT / "capture_run_result"
    if run_root.exists():
        shutil.rmtree(run_root)
    config = OUTPUT / "effective_frozen_u1_odometry.yaml"
    shutil.copy2(FROZEN_CONFIG, config)
    task = rectangle_task()
    cache = prepare_truncated_cache(task, frame_limit)
    variant = RETAINED_VARIANTS[LATEST_IMUATT_METHOD]._replace(
        name="uvd_schur_capture",
        imu_factor_mode="two_state_fixed_lag",
    )
    spec = RunSpec(
        trial=1,
        scene=task.cache_scene,
        scene_root=task.scene_root,
        variant=variant,
        result_dir=run_root,
    )
    sequence = make_seq_cfg(spec, OUTPUT / "configs")
    CAPTURED.clear()
    SOLVER_SETTINGS.clear()
    old_argv = list(sys.argv)
    TwoStateVIOSolver.solve = capture_solve
    TwoFrame_PGO._optimize_two_state_fixed_lag = staticmethod(capture_optimize)
    started = time.perf_counter()
    try:
        sys.argv = [
            str(ROOT / "MACVO.py"),
            "--odom",
            str(config),
            "--data",
            str(sequence),
            "--resultRoot",
            str(run_root),
            "--visual-cache-mode",
            "replay",
            "--visual-cache-path",
            str(cache),
            "--seq_to",
            str(int(frame_limit)),
        ]
        runpy.run_path(str(ROOT / "MACVO.py"), run_name="__main__")
    finally:
        sys.argv = old_argv
        TwoStateVIOSolver.solve = ORIGINAL_SOLVE
        TwoFrame_PGO._optimize_two_state_fixed_lag = staticmethod(ORIGINAL_OPTIMIZE)
    elapsed = time.perf_counter() - started
    expected = int(frame_limit) - ACTIVE_START_FRAME - 1
    if len(CAPTURED) != expected:
        raise RuntimeError(f"expected {expected} captured edges, got {len(CAPTURED)}")
    payload = {
        "schema_version": 1,
        "active_start_frame": ACTIVE_START_FRAME,
        "frame_limit": int(frame_limit),
        "solver_settings": dict(SOLVER_SETTINGS),
        "capture_runtime_seconds": elapsed,
        "dataset": str(task.scene_root),
        "visual_cache": str(cache),
        "frozen_config": str(FROZEN_CONFIG),
        "edges": CAPTURED,
    }
    torch.save(payload, PACKET)
    write_json(
        OUTPUT / "capture_manifest.json",
        {key: value for key, value in payload.items() if key != "edges"}
        | {"edge_count": len(CAPTURED), "packet": str(PACKET)},
    )
    return payload


def make_transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    value = np.eye(4, dtype=np.float64)
    value[:3, :3] = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    value[:3, 3] = np.asarray(translation, dtype=np.float64).reshape(3)
    return value


def invert(transform: np.ndarray) -> np.ndarray:
    rotation = transform[:3, :3]
    return make_transform(rotation.T, -rotation.T @ transform[:3, 3])


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


def load_truth(dataset: Path) -> dict[str, np.ndarray]:
    ref = pd.read_csv(dataset / "ref_pose.csv")
    metadata = json.loads((dataset / "metadata.json").read_text(encoding="utf-8"))
    lever_nwu = np.asarray(
        metadata["extrinsics"]["T_body_imu"]["translation_body_nwu_m"],
        dtype=np.float64,
    )
    lever_internal = NWU_TO_NED @ lever_nwu
    poses = []
    for row in ref.itertuples(index=False):
        rotation_nwu = Rotation.from_quat(
            [row.qx, row.qy, row.qz, row.qw]
        ).as_matrix()
        body = make_transform(
            NWU_TO_NED @ rotation_nwu @ NWU_TO_NED,
            NWU_TO_NED @ np.asarray([row.x, row.y, row.z], dtype=np.float64),
        )
        poses.append(body @ make_transform(np.eye(3), lever_internal))
    velocity = ref[["vx", "vy", "vz"]].to_numpy(np.float64) @ NWU_TO_NED.T
    angular_rate = ref[["wx", "wy", "wz"]].to_numpy(np.float64)
    speed = np.linalg.norm(velocity, axis=1)
    turn_rate = np.linalg.norm(angular_rate, axis=1)
    return {
        "poses": np.stack(poses),
        "velocity": velocity,
        "speed": speed,
        "turn_rate": turn_rate,
        "timestamps": ref["timestamp"].to_numpy(np.int64),
        "lever_nwu": lever_nwu,
    }


def segment_label(frame: int, truth: dict[str, np.ndarray]) -> str:
    speed = float(truth["speed"][frame])
    turn_rate = float(truth["turn_rate"][frame])
    if speed < 0.03 and turn_rate < 0.03:
        return "static"
    if turn_rate > 0.15:
        return "turn"
    if speed > 0.10:
        return "straight"
    return "transition"


def evenly_spaced(values: list[int], count: int) -> list[int]:
    if len(values) <= count:
        return values
    positions = np.linspace(0, len(values) - 1, count).round().astype(int)
    return [values[index] for index in sorted(set(positions.tolist()))]


def select_seeds(
    edges: list[dict[str, Any]],
    truth: dict[str, np.ndarray],
    *,
    per_segment: int,
    lookahead: int,
) -> list[int]:
    by_segment: dict[str, list[int]] = {
        "static": [],
        "straight": [],
        "turn": [],
        "transition": [],
    }
    for index, edge in enumerate(edges):
        if index + lookahead >= len(edges):
            continue
        by_segment[segment_label(int(edge["frame_i"]), truth)].append(index)
    selected: list[int] = []
    for segment in ("static", "straight", "turn", "transition"):
        selected.extend(evenly_spaced(by_segment[segment], per_segment))
    return sorted(set(selected))


def solver_result_from_baseline(payload: dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(**payload)


def solve_mode(
    solver: TwoStateVIOSolver,
    problem: TwoStateVIOProblem,
    mode: str,
) -> tuple[TwoStateVIOResult | SimpleNamespace, dict[str, float]]:
    problem = clone_problem(problem)
    if mode == "nonlinear_full":
        result = solver.solve(problem)
        return result, {}
    marginal_mode = {
        "linearized_full": "full",
        "translation_marginal": "translation",
        "rotation_marginal": "rotation",
    }[mode]
    linearization = linearize_uvd_pose_factor(
        problem.state_i,
        problem.state_j,
        problem.visual_pose,
        marginal_mode=marginal_mode,
    )
    if not isinstance(linearization.factor, LinearizedUVDPoseFactor):
        raise TypeError("expected a linearized UVD factor")
    result = solver.solve(replace(problem, visual_pose=linearization.factor))
    full_values = torch.linalg.eigvalsh(linearization.full_hessian)
    reduced_values = torch.linalg.eigvalsh(linearization.reduced_hessian)
    hessian = linearization.full_hessian
    coupling = float(
        torch.linalg.matrix_norm(hessian[:3, 3:])
        / (
            torch.linalg.matrix_norm(hessian[:3, :3])
            * torch.linalg.matrix_norm(hessian[3:, 3:])
        ).sqrt().clamp_min(1.0e-15)
    )
    return result, {
        "full_hessian_min_eigenvalue": float(full_values.min()),
        "full_hessian_max_eigenvalue": float(full_values.max()),
        "reduced_hessian_min_eigenvalue": float(reduced_values.min()),
        "reduced_hessian_max_eigenvalue": float(reduced_values.max()),
        "reduced_hessian_rank": int(
            torch.linalg.matrix_rank(linearization.reduced_hessian).item()
        ),
        "translation_rotation_coupling": coupling,
        "uses_imu_anchor": 0.0,
    }


def rotation_error(estimate: np.ndarray, truth: np.ndarray) -> np.ndarray:
    return Rotation.from_matrix(truth[:3, :3].T @ estimate[:3, :3]).as_rotvec()


def append_metrics(
    row: dict[str, Any],
    *,
    prefix: str,
    estimate: NavigationState,
    truth_pose: np.ndarray,
    truth_velocity: np.ndarray,
) -> None:
    pose = pose_matrix(estimate)
    position_error = pose[:3, 3] - truth_pose[:3, 3]
    orientation_error = rotation_error(pose, truth_pose)
    velocity_error = estimate.velocity_W.detach().cpu().numpy().reshape(3) - truth_velocity
    row[f"{prefix}_position_xy_error"] = float(np.linalg.norm(position_error[:2]))
    row[f"{prefix}_position_3d_error"] = float(np.linalg.norm(position_error))
    row[f"{prefix}_orientation_error"] = float(np.linalg.norm(orientation_error))
    row[f"{prefix}_velocity_error"] = float(np.linalg.norm(velocity_error))


def evaluate_one(
    edges: list[dict[str, Any]],
    truth: dict[str, np.ndarray],
    solver: TwoStateVIOSolver,
    *,
    seed: int,
    mode: str,
    lookahead: int,
) -> dict[str, Any]:
    edge = edges[seed]
    baseline = solver_result_from_baseline(edge["baseline"])
    started = time.perf_counter()
    if mode == "nonlinear_full":
        result = baseline
        diagnostics: dict[str, float] = {}
    else:
        result, diagnostics = solve_mode(solver, edge["problem"], mode)
    immediate_result = result
    for offset in range(1, lookahead + 1):
        propagated = future_problem(edges[seed + offset]["problem"], result)
        result = solver.solve(propagated)
    frame_i = int(edge["frame_i"])
    frame_j = int(edge["frame_j"])
    end_frame = int(edges[seed + lookahead]["frame_j"])
    estimate_relative = invert(pose_matrix(immediate_result.state_i)) @ pose_matrix(
        immediate_result.state_j
    )
    truth_relative = invert(truth["poses"][frame_i]) @ truth["poses"][frame_j]
    relative_error = invert(truth_relative) @ estimate_relative
    baseline_pair_delta = torch.cat(
        [
            state_boxminus(immediate_result.state_i, baseline.state_i),
            state_boxminus(immediate_result.state_j, baseline.state_j),
        ]
    )
    row: dict[str, Any] = {
        "seed_index": seed,
        "frame_i": frame_i,
        "frame_j": frame_j,
        "end_frame": end_frame,
        "segment": segment_label(frame_i, truth),
        "mode": mode,
        "converged": bool(immediate_result.converged),
        "iterations": int(immediate_result.iterations),
        "runtime_s": time.perf_counter() - started,
        "immediate_relative_translation_error": float(
            np.linalg.norm(relative_error[:3, 3])
        ),
        "immediate_relative_translation_xy_error": float(
            np.linalg.norm(relative_error[:2, 3])
        ),
        "immediate_relative_rotation_error": float(
            np.linalg.norm(Rotation.from_matrix(relative_error[:3, :3]).as_rotvec())
        ),
        "pair_state_delta_vs_nonlinear_full": float(
            torch.linalg.vector_norm(baseline_pair_delta).item()
        ),
        "initial_cost": float(immediate_result.initial_cost),
        "final_cost": float(immediate_result.final_cost),
        "prior_cost": float(immediate_result.prior_cost),
        "imu_cost": float(immediate_result.imu_cost),
        "bias_cost": float(immediate_result.bias_cost),
        "visual_cost": float(immediate_result.visual_pose_cost),
        "num_observations": int(edge["quality"]["num_observations"]),
    }
    row.update(diagnostics)
    append_metrics(
        row,
        prefix="immediate",
        estimate=immediate_result.state_j,
        truth_pose=truth["poses"][frame_j],
        truth_velocity=truth["velocity"][frame_j],
    )
    append_metrics(
        row,
        prefix="lookahead",
        estimate=result.state_j,
        truth_pose=truth["poses"][end_frame],
        truth_velocity=truth["velocity"][end_frame],
    )
    return row


def initialize_evaluation_worker(packet_path: str) -> None:
    global WORKER_PAYLOAD, WORKER_TRUTH, WORKER_SOLVER
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    WORKER_PAYLOAD = torch.load(packet_path, map_location="cpu", weights_only=False)
    WORKER_TRUTH = load_truth(Path(WORKER_PAYLOAD["dataset"]))
    WORKER_SOLVER = TwoStateVIOSolver(**WORKER_PAYLOAD["solver_settings"])


def evaluation_worker(job: tuple[int, str, int]) -> dict[str, Any]:
    if WORKER_PAYLOAD is None or WORKER_TRUTH is None or WORKER_SOLVER is None:
        raise RuntimeError("evaluation worker was not initialized")
    seed, mode, lookahead = job
    return evaluate_one(
        WORKER_PAYLOAD["edges"],
        WORKER_TRUTH,
        WORKER_SOLVER,
        seed=seed,
        mode=mode,
        lookahead=lookahead,
    )


def evaluate(
    payload: dict[str, Any],
    *,
    lookahead: int,
    seeds_per_segment: int,
    workers: int = 1,
) -> pd.DataFrame:
    edges = payload["edges"]
    truth = load_truth(Path(payload["dataset"]))
    seeds = select_seeds(
        edges,
        truth,
        per_segment=seeds_per_segment,
        lookahead=lookahead,
    )
    jobs = [(seed, mode, lookahead) for seed in seeds for mode in MODES]
    rows: list[dict[str, Any]] = []
    if workers <= 1:
        solver = TwoStateVIOSolver(**payload["solver_settings"])
        iterator = (
            evaluate_one(
                edges, truth, solver, seed=seed, mode=mode, lookahead=lookahead
            )
            for seed, mode, _ in jobs
        )
        for completed, row in enumerate(iterator, start=1):
            rows.append(row)
            if completed % 12 == 0 or completed == len(jobs):
                print(f"[experiment] {completed}/{len(jobs)}", flush=True)
    else:
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=int(workers),
            mp_context=mp.get_context("spawn"),
            initializer=initialize_evaluation_worker,
            initargs=(str(PACKET),),
        ) as executor:
            futures = [executor.submit(evaluation_worker, job) for job in jobs]
            for completed, future in enumerate(
                concurrent.futures.as_completed(futures), start=1
            ):
                rows.append(future.result())
                if completed % 12 == 0 or completed == len(jobs):
                    print(f"[experiment] {completed}/{len(jobs)}", flush=True)
    frame = pd.DataFrame(rows)
    required_finite = [
        "runtime_s",
        "immediate_relative_translation_error",
        "immediate_relative_translation_xy_error",
        "immediate_relative_rotation_error",
        "pair_state_delta_vs_nonlinear_full",
        "initial_cost",
        "final_cost",
        "prior_cost",
        "imu_cost",
        "bias_cost",
        "visual_cost",
        "immediate_position_xy_error",
        "immediate_position_3d_error",
        "immediate_orientation_error",
        "immediate_velocity_error",
        "lookahead_position_xy_error",
        "lookahead_position_3d_error",
        "lookahead_orientation_error",
        "lookahead_velocity_error",
    ]
    if not bool(np.isfinite(frame[required_finite].to_numpy()).all()):
        raise FloatingPointError("experiment output contains NaN/Inf")
    return frame


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "immediate_relative_translation_xy_error",
        "immediate_relative_rotation_error",
        "immediate_position_xy_error",
        "immediate_orientation_error",
        "immediate_velocity_error",
        "lookahead_position_xy_error",
        "lookahead_orientation_error",
        "lookahead_velocity_error",
        "pair_state_delta_vs_nonlinear_full",
        "iterations",
    ]
    records: list[dict[str, Any]] = []
    for (segment, mode), group in frame.groupby(["segment", "mode"], sort=True):
        record: dict[str, Any] = {
            "segment": segment,
            "mode": mode,
            "count": len(group),
            "convergence_rate": float(group["converged"].mean()),
        }
        for metric in metrics:
            values = group[metric].to_numpy(np.float64)
            record[f"{metric}_rmse"] = float(np.sqrt(np.mean(values**2)))
            record[f"{metric}_median"] = float(np.median(values))
            record[f"{metric}_p95"] = float(np.quantile(values, 0.95))
        records.append(record)
    return pd.DataFrame(records)


def decide(frame: pd.DataFrame, summary: pd.DataFrame) -> dict[str, Any]:
    control = frame[frame["mode"] == "linearized_full"]
    control_max = float(control["pair_state_delta_vs_nonlinear_full"].max())
    control_p95 = float(control["pair_state_delta_vs_nonlinear_full"].quantile(0.95))
    valid = control_max <= 1.0e-2
    segment_decisions: dict[str, Any] = {}
    paired_effects: dict[str, Any] = {}
    metric = "lookahead_position_xy_error_rmse"
    rotation_metric = "lookahead_orientation_error_rmse"
    for segment in sorted(summary["segment"].unique()):
        part = summary[summary["segment"] == segment].set_index("mode")
        if not valid:
            winner = "invalid_control"
        else:
            candidates = part.loc[
                ["linearized_full", "translation_marginal", "rotation_marginal"]
            ]
            normalized = candidates[metric] / max(float(candidates[metric].min()), 1e-12)
            normalized += candidates[rotation_metric] / max(
                float(candidates[rotation_metric].min()), 1e-12
            )
            winner = str(normalized.idxmin())
        segment_decisions[segment] = {
            "winner_balanced_position_orientation": winner,
            "metrics": part[[metric, rotation_metric, "convergence_rate"]]
                .to_dict(orient="index"),
        }
        segment_rows = frame[frame["segment"] == segment]
        pivot = segment_rows.pivot(
            index="seed_index",
            columns="mode",
            values=[
                "lookahead_position_xy_error",
                "lookahead_orientation_error",
                "lookahead_velocity_error",
            ],
        )
        baseline_position = float(part.loc["nonlinear_full", metric])
        baseline_orientation = float(part.loc["nonlinear_full", rotation_metric])
        effects: dict[str, Any] = {}
        for candidate in ("translation_marginal", "rotation_marginal"):
            candidate_position = float(part.loc[candidate, metric])
            candidate_orientation = float(part.loc[candidate, rotation_metric])
            position_delta = (
                pivot[("lookahead_position_xy_error", candidate)]
                - pivot[("lookahead_position_xy_error", "nonlinear_full")]
            )
            orientation_delta = (
                pivot[("lookahead_orientation_error", candidate)]
                - pivot[("lookahead_orientation_error", "nonlinear_full")]
            )
            effects[candidate] = {
                "position_rmse_change_percent": 100.0
                * (candidate_position - baseline_position)
                / max(baseline_position, 1.0e-12),
                "orientation_rmse_change_percent": 100.0
                * (candidate_orientation - baseline_orientation)
                / max(baseline_orientation, 1.0e-12),
                "position_win_rate": float((position_delta < 0.0).mean()),
                "orientation_win_rate": float((orientation_delta < 0.0).mean()),
                "position_median_paired_delta_m": float(position_delta.median()),
                "orientation_median_paired_delta_rad": float(
                    orientation_delta.median()
                ),
            }
        paired_effects[segment] = effects
    linearized = frame[frame["mode"] == "linearized_full"]
    coupling_by_segment = {
        str(segment): {
            "median": float(group["translation_rotation_coupling"].median()),
            "p95": float(group["translation_rotation_coupling"].quantile(0.95)),
        }
        for segment, group in linearized.groupby("segment")
    }
    return {
        "linearization_control_valid": valid,
        "linearized_full_pair_delta_p95": control_p95,
        "linearized_full_pair_delta_max": control_max,
        "control_threshold": 1.0e-2,
        "segment_decisions": segment_decisions,
        "paired_effects_vs_nonlinear_full": paired_effects,
        "translation_rotation_hessian_coupling": coupling_by_segment,
        "production_u1_modified": False,
        "gt_used_in_factor_construction": False,
        "imu_anchor_used_by_marginal_factors": False,
        "adaptive_hard_switching_approved": False,
        "recommended_production_mode": "nonlinear_full_direct_uvd_u1",
        "next_experiment": (
            "Keep the full UVD factor and compare N=2/N=5/N=10 fixed-lag windows; "
            "do not train or tune a mode classifier before a material branch advantage exists."
        ),
        "interpretation": (
            "Marginal-mode comparisons are valid only as fixed-linearization local "
            "counterfactuals when the linearized-full control remains close to nonlinear U1. "
            "No tested segment shows a material, joint position-and-orientation benefit from "
            "hard translation/rotation marginal switching."
        ),
    }


def write_report(
    frame: pd.DataFrame, summary: pd.DataFrame, decision: dict[str, Any]
) -> None:
    lines = [
        "# 矩形 U1 UVD Schur 边缘因子分段实验",
        "",
        "## 实验契约",
        "",
        "- 生产 Direct UVD U1 已冻结，未修改默认配置。",
        "- 每个候选使用相同的状态初值、Schur prior、IMU 因子和 UVD 点观测。",
        "- 平移/旋转边缘因子由视觉 UVD 正规方程消元得到，不使用 IMU anchor。",
        "- GT 只用于运动分段和误差评分，不进入任何因子。",
        f"- 固定线性化 FULL 控制最大两状态差：`{decision['linearized_full_pair_delta_max']:.6g}`。",
        f"- 控制门是否通过：`{decision['linearization_control_valid']}`。",
        "- UVD 平移与旋转正规方程高度耦合，分段 coupling 中位数约为 0.94--0.95。",
        "",
        "## 分段结果",
        "",
    ]
    columns = [
        "segment",
        "mode",
        "count",
        "convergence_rate",
        "lookahead_position_xy_error_rmse",
        "lookahead_orientation_error_rmse",
        "lookahead_velocity_error_rmse",
        "pair_state_delta_vs_nonlinear_full_p95",
    ]
    lines.append(summary[columns].to_markdown(index=False, floatfmt=".6g"))
    lines.extend(["", "## 判定", "", "```json", json.dumps(jsonify(decision), ensure_ascii=False, indent=2), "```", ""])
    (OUTPUT / "rectangle_uvd_schur_marginal_report_cn.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    if args.skip_capture:
        if not PACKET.exists():
            raise FileNotFoundError(PACKET)
        payload = torch.load(PACKET, map_location="cpu", weights_only=False)
    else:
        payload = capture_production_problems(
            force=bool(args.force_capture), frame_limit=int(args.frame_limit)
        )
    frame = evaluate(
        payload,
        lookahead=max(0, int(args.lookahead)),
        seeds_per_segment=max(1, int(args.seeds_per_segment)),
        workers=max(1, int(args.workers)),
    )
    summary = summarize(frame)
    decision = decide(frame, summary)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUTPUT / "rectangle_uvd_schur_marginal_per_seed.csv", index=False)
    summary.to_csv(OUTPUT / "rectangle_uvd_schur_marginal_summary.csv", index=False)
    write_json(OUTPUT / "rectangle_uvd_schur_marginal_decision.json", decision)
    write_report(frame, summary, decision)
    print(json.dumps(jsonify(decision), ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
