#!/usr/bin/env python3
"""Benchmark the frozen N=2 U1/T2 backends and assemble a Pure/U1/T2 Pareto report.

The microbenchmark consumes the same captured production U1 problems for every
backend branch. T2 is measured both with online UVD compression and with the
prebuilt compression cache. Pure MACVO's historical full-pipeline wall time is
reported separately because the captured problems start after Pure MACVO has
already performed network inference and visual pose optimization.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import statistics
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Scripts.run_u1_counterfactual_branches import clone_problem, future_problem  # noqa: E402
from Utility.CompressedUVDFactorCache import CompressedUVDFactorCacheReader  # noqa: E402
from Utility.TwoStateVIO import (  # noqa: E402
    TwoStateVIOProblem,
    TwoStateVIOResult,
    TwoStateVIOSolver,
    UVDFactor,
    linearize_uvd_relative_pose_factor,
    linearized_uvd_pose_factor_from_normal_equations,
)


DEFAULT_PACKET = ROOT / (
    "analysis_rectangle_uvd_schur_marginal_20260719/"
    "captured_rectangle_u1_problems.pt"
)
DEFAULT_CACHE = ROOT / (
    "VisualCache/static63_unique_visual_20260713/"
    "clear_stop_turn_rectangle_truth_normal_noise"
)
DEFAULT_OUTPUT = ROOT / "analysis_pure_u1_t2_pareto_20260720"
PURE_PROGRESS = ROOT / (
    "Results/visual_factor_cache_static63_unique_source_20260713/progress.csv"
)
ACCURACY_ALL = ROOT / (
    "analysis_imu_center_all_methods_20260719/imu_center_accuracy_metrics.csv"
)
ACCURACY_T2 = ROOT / (
    "analysis_t0_u1_t2_full_imu_center_20260720/"
    "t0_u1_t2_full_imu_center_metrics.csv"
)
METHODS = ("u1_direct_uvd", "t2_online_compression", "t2_cached_compression")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, default=DEFAULT_PACKET)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--edge-count", type=int, default=209)
    parser.add_argument("--warmup-edges", type=int, default=3)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    return parser.parse_args()


def distribution(values: list[float]) -> dict[str, float]:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("cannot summarize an empty distribution")
    return {
        "min": ordered[0],
        "median": statistics.median(ordered),
        "mean": statistics.fmean(ordered),
        "p95": ordered[int(round(0.95 * (len(ordered) - 1)))],
        "max": ordered[-1],
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def solver_from(settings: dict[str, Any]) -> TwoStateVIOSolver:
    names = (
        "max_iterations",
        "initial_damping",
        "step_tolerance",
        "cost_tolerance",
        "covariance_eigenvalue_floor",
        "marginalization_eigenvalue_floor",
    )
    return TwoStateVIOSolver(**{name: settings[name] for name in names})


def prepare_visual(method: str, incoming: TwoStateVIOProblem, packet):
    visual = incoming.visual_pose
    if not isinstance(visual, UVDFactor):
        raise TypeError("captured problem does not contain a direct UVD factor")
    if method == "u1_direct_uvd":
        return visual
    if method == "t2_online_compression":
        return linearize_uvd_relative_pose_factor(
            packet.reference_CjCi,
            visual,
            marginal_mode="full",
        ).factor
    if method == "t2_cached_compression":
        return linearized_uvd_pose_factor_from_normal_equations(
            packet.reference_CjCi,
            packet.hessian,
            packet.gradient,
            visual.extrinsic_CI,
        )
    raise ValueError(method)


def run_method(
    method: str,
    edges: list[dict[str, Any]],
    packets: list[Any],
    settings: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[TwoStateVIOResult]]:
    solver = solver_from(settings)
    previous: TwoStateVIOResult | None = None
    rows: list[dict[str, Any]] = []
    results: list[TwoStateVIOResult] = []
    for edge_id, (edge, packet) in enumerate(zip(edges, packets, strict=True)):
        total_start = time.perf_counter()
        incoming = (
            clone_problem(edge["problem"])
            if previous is None
            else future_problem(edge["problem"], previous)
        )
        factor_start = time.perf_counter()
        visual = prepare_visual(method, incoming, packet)
        factor_ms = (time.perf_counter() - factor_start) * 1000.0
        problem = replace(incoming, visual_pose=visual)
        solve_start = time.perf_counter()
        result = solver.solve(problem)
        solve_ms = (time.perf_counter() - solve_start) * 1000.0
        total_ms = (time.perf_counter() - total_start) * 1000.0
        rows.append(
            {
                "method": method,
                "edge_id": edge_id,
                "frame_i": int(edge["frame_i"]),
                "frame_j": int(edge["frame_j"]),
                "point_count": int(incoming.visual_pose.points_Ci.shape[0]),
                "factor_prepare_ms": factor_ms,
                "solve_ms": solve_ms,
                "backend_total_ms": total_ms,
                "iterations": int(result.iterations),
                "converged": int(result.converged),
                "final_cost": float(result.final_cost),
                "final_step_norm": float(result.final_step_norm),
                "final_gradient_inf_norm": float(result.final_gradient_inf_norm),
            }
        )
        results.append(result)
        previous = result
        print(
            f"[{method}] {edge_id + 1}/{len(edges)} "
            f"prepare={factor_ms:.2f} solve={solve_ms:.2f} total={total_ms:.2f} ms",
            flush=True,
        )
    return rows, results


def historical_pure_runtime() -> dict[str, Any]:
    with PURE_PROGRESS.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    selected = next(
        row
        for row in rows
        if row["scene"] == "clear_stop_turn_rectangle_truth_normal_noise"
        and row["variant"] == "pure_macvo"
        and row["status"] == "ok"
    )
    frame_count = 1890
    runtime_s = float(selected["runtime_s"])
    return {
        "scene": selected["scene"],
        "runtime_s": runtime_s,
        "frame_count": frame_count,
        "wall_ms_per_frame": runtime_s * 1000.0 / frame_count,
        "scope": "historical full pipeline: image IO + MACVO networks + native visual optimization",
        "directly_comparable_to_backend_microbenchmark": False,
    }


def accuracy_rows() -> list[dict[str, Any]]:
    with ACCURACY_ALL.open(encoding="utf-8", newline="") as stream:
        all_rows = list(csv.DictReader(stream))
    with ACCURACY_T2.open(encoding="utf-8", newline="") as stream:
        t2_rows = list(csv.DictReader(stream))
    output: list[dict[str, Any]] = []
    for scene in ("circle", "rectangle", "straight"):
        pure = next(
            row
            for row in all_rows
            if row["scene"] == scene
            and row["optimizer"] == "pure_macvo"
            and row["output_method"] == "A_raw"
        )
        u1 = next(
            row
            for row in all_rows
            if row["scene"] == scene
            and row["optimizer"] == "u1"
            and row["output_method"] == "A_raw"
        )
        t2 = next(
            row for row in t2_rows if row["scene"] == scene and row["method"] == "t2"
        )
        for method, row in (("pure_macvo", pure), ("u1", u1), ("t2", t2)):
            output.append(
                {
                    "scene": scene,
                    "method": method,
                    "frame_count": int(row["frame_count"]),
                    "xy_ate_rmse_m": float(row["xy_ate_rmse_m"]),
                    "xyz_ate_rmse_m": float(row["xyz_ate_rmse_m"]),
                    "orientation_rmse_rad": float(row["orientation_rmse_rad"]),
                    "translation_rpe_rmse_m": float(row["translation_rpe_rmse_m"]),
                    "rotation_rpe_rmse_rad": float(row["rotation_rpe_rmse_rad"]),
                }
            )
    return output


def main() -> int:
    args = parse_args()
    if args.cpu_threads <= 0:
        raise ValueError("--cpu-threads must be positive")
    torch.set_num_threads(int(args.cpu_threads))
    args.output.mkdir(parents=True, exist_ok=True)
    payload = torch.load(args.packet, map_location="cpu", weights_only=False)
    start = int(payload.get("active_start_frame", 0))
    edges = [edge for edge in payload["edges"] if int(edge["frame_i"]) >= start][
        : int(args.edge_count)
    ]
    if len(edges) != int(args.edge_count):
        raise RuntimeError(f"requested {args.edge_count} edges, found {len(edges)}")
    reader = CompressedUVDFactorCacheReader(args.cache)
    packets = [
        reader.load_pair(
            int(edge["frame_i"]),
            int(edge["frame_j"]),
            str(reader.visual_hashes[int(edge["frame_i"])]),
        )
        for edge in edges
    ]

    method_rows: dict[str, list[dict[str, Any]]] = {}
    method_results: dict[str, list[TwoStateVIOResult]] = {}
    selected_methods = tuple(args.methods)
    for method in selected_methods:
        rows, results = run_method(method, edges, packets, payload["solver_settings"])
        method_rows[method] = rows
        method_results[method] = results
    all_timing_rows = [row for method in selected_methods for row in method_rows[method]]
    write_csv(args.output / "backend_timing_per_edge.csv", all_timing_rows)

    warmup = min(max(int(args.warmup_edges), 0), len(edges) - 1)
    summary: dict[str, Any] = {
        "contract": {
            "packet": str(args.packet.resolve()),
            "cache": str(args.cache.resolve()),
            "edge_count": len(edges),
            "active_start_frame": start,
            "warmup_edges_excluded": warmup,
            "cpu_threads": torch.get_num_threads(),
            "dtype": "torch.float64",
            "python": platform.python_version(),
            "torch": torch.__version__,
            "same_inputs_for_u1_and_t2": True,
            "methods": list(selected_methods),
            "pure_runtime_scope_warning": (
                "Pure MACVO historical wall time includes network inference; "
                "the backend microbenchmark begins from captured visual observations."
            ),
        },
        "pure_macvo_historical_runtime": historical_pure_runtime(),
        "backend_timing": {},
    }
    for method in selected_methods:
        rows = method_rows[method][warmup:]
        summary["backend_timing"][method] = {
            field: distribution([float(row[field]) for row in rows])
            for field in ("factor_prepare_ms", "solve_ms", "backend_total_ms")
        }
        summary["backend_timing"][method]["converged_rate"] = statistics.fmean(
            int(row["converged"]) for row in method_rows[method]
        )
        summary["backend_timing"][method]["iterations"] = distribution(
            [float(row["iterations"]) for row in method_rows[method]]
        )

    if {
        "t2_online_compression",
        "t2_cached_compression",
    }.issubset(method_rows):
        cached = method_rows["t2_cached_compression"]
        online = method_rows["t2_online_compression"]
        summary["t2_online_vs_cached_equivalence"] = {
            "max_final_cost_abs_difference": max(
                abs(float(a["final_cost"]) - float(b["final_cost"]))
                for a, b in zip(online, cached, strict=True)
            ),
            "all_iterations_equal": all(
                int(a["iterations"]) == int(b["iterations"])
                for a, b in zip(online, cached, strict=True)
            ),
            "all_convergence_equal": all(
                int(a["converged"]) == int(b["converged"])
                for a, b in zip(online, cached, strict=True)
            ),
        }
    (args.output / "runtime_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_csv(args.output / "accuracy_pareto.csv", accuracy_rows())
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
