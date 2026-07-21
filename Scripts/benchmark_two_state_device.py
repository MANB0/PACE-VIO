#!/usr/bin/env python3
"""Benchmark the frozen T2 two-state solver on CPU and CUDA.

The benchmark consumes the captured U1 problem packet and the cached T2 normal
equations. It does not run MACVO or alter any optimizer formula or setting.
Each device follows its own fixed-lag prior chain. Results are copied back to
CPU between edges to include the transfer required by the current pipeline.
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

import pypose as pp
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Scripts.run_u1_counterfactual_branches import (  # noqa: E402
    clone_problem,
    future_problem,
)
from Utility.CompressedUVDFactorCache import (  # noqa: E402
    CompressedUVDFactorCacheReader,
)
from Utility.TwoStateVIO import (  # noqa: E402
    NavigationState,
    SquareRootPrior,
    TwoStateVIOProblem,
    TwoStateVIOResult,
    TwoStateVIOSolver,
    UVDFactor,
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
DEFAULT_OUTPUT = ROOT / "analysis_two_state_device_benchmark_20260720"
DTYPE = torch.float64


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, default=DEFAULT_PACKET)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--edge-count", type=int, default=209)
    parser.add_argument(
        "--devices",
        nargs="+",
        default=("cpu", "cuda"),
        choices=("cpu", "cuda"),
    )
    parser.add_argument("--warmup-edges", type=int, default=3)
    return parser.parse_args()


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def problem_to(
    problem: TwoStateVIOProblem,
    *,
    device: torch.device,
) -> TwoStateVIOProblem:
    return TwoStateVIOProblem(
        state_i=problem.state_i.to(device=device, dtype=DTYPE),
        state_j=problem.state_j.to(device=device, dtype=DTYPE),
        prior_i=problem.prior_i.to(device=device, dtype=DTYPE),
        imu=problem.imu.to(device=device, dtype=DTYPE),
        visual_pose=problem.visual_pose.to(device=device, dtype=DTYPE),
        optimize_acc_bias=bool(problem.optimize_acc_bias),
        optimize_gyro_bias=bool(problem.optimize_gyro_bias),
    )


def state_to_cpu(state: NavigationState) -> NavigationState:
    return state.to(device=torch.device("cpu"), dtype=DTYPE).detach()


def prior_to_cpu(prior: SquareRootPrior) -> SquareRootPrior:
    return prior.to(device=torch.device("cpu"), dtype=DTYPE)


def result_to_cpu(result: TwoStateVIOResult) -> TwoStateVIOResult:
    return replace(
        result,
        state_i=state_to_cpu(result.state_i),
        state_j=state_to_cpu(result.state_j),
        prior_j=prior_to_cpu(result.prior_j),
        hessian=result.hessian.detach().cpu(),
        gradient=result.gradient.detach().cpu(),
    )


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


def cached_factor(
    incoming: TwoStateVIOProblem,
    reader: CompressedUVDFactorCacheReader,
    frame_i: int,
    frame_j: int,
):
    visual = incoming.visual_pose
    if not isinstance(visual, UVDFactor):
        raise TypeError("captured problem does not contain a UVD factor")
    packet = reader.load_pair(
        frame_i,
        frame_j,
        str(reader.visual_hashes[frame_i]),
    )
    return linearized_uvd_pose_factor_from_normal_equations(
        packet.reference_CjCi,
        packet.hessian,
        packet.gradient,
        visual.extrinsic_CI,
    )


def run_device(
    *,
    name: str,
    edges: list[dict[str, Any]],
    reader: CompressedUVDFactorCacheReader,
    settings: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[TwoStateVIOResult]]:
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA benchmark requested but CUDA is unavailable")
    solver = solver_from(settings)
    previous_cpu: TwoStateVIOResult | None = None
    rows: list[dict[str, Any]] = []
    results: list[TwoStateVIOResult] = []

    for edge_id, edge in enumerate(edges):
        total_start = time.perf_counter()
        incoming_cpu = (
            clone_problem(edge["problem"])
            if previous_cpu is None
            else future_problem(edge["problem"], previous_cpu)
        )
        frame_i = int(edge["frame_i"])
        frame_j = int(edge["frame_j"])
        visual = cached_factor(incoming_cpu, reader, frame_i, frame_j)
        incoming_cpu = replace(incoming_cpu, visual_pose=visual)

        synchronize(device)
        transfer_in_start = time.perf_counter()
        problem = problem_to(incoming_cpu, device=device)
        synchronize(device)
        transfer_in_ms = (time.perf_counter() - transfer_in_start) * 1000.0

        solve_start = time.perf_counter()
        result = solver.solve(problem)
        synchronize(device)
        solve_ms = (time.perf_counter() - solve_start) * 1000.0

        transfer_out_start = time.perf_counter()
        result_cpu = result_to_cpu(result)
        synchronize(device)
        transfer_out_ms = (time.perf_counter() - transfer_out_start) * 1000.0
        total_ms = (time.perf_counter() - total_start) * 1000.0

        rows.append(
            {
                "device": name,
                "edge_id": edge_id,
                "frame_i": frame_i,
                "frame_j": frame_j,
                "transfer_in_ms": transfer_in_ms,
                "solve_ms": solve_ms,
                "transfer_out_ms": transfer_out_ms,
                "end_to_end_ms": total_ms,
                "iterations": int(result_cpu.iterations),
                "converged": int(result_cpu.converged),
                "final_cost": float(result_cpu.final_cost),
                "final_step_norm": float(result_cpu.final_step_norm),
                "final_gradient_inf_norm": float(result_cpu.final_gradient_inf_norm),
            }
        )
        results.append(result_cpu)
        previous_cpu = result_cpu
        print(
            f"[{name}] {edge_id + 1}/{len(edges)} "
            f"solve={solve_ms:.2f} ms total={total_ms:.2f} ms",
            flush=True,
        )
    return rows, results


def distribution(values: list[float]) -> dict[str, float]:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("cannot summarize an empty distribution")
    p95_index = int(round(0.95 * (len(ordered) - 1)))
    return {
        "min": ordered[0],
        "median": statistics.median(ordered),
        "mean": statistics.fmean(ordered),
        "p95": ordered[p95_index],
        "max": ordered[-1],
    }


def state_difference(
    reference: NavigationState,
    candidate: NavigationState,
) -> dict[str, float]:
    pose = (
        pp.SE3(reference.pose_WB).Inv() @ pp.SE3(candidate.pose_WB)
    ).Log().tensor().reshape(6)
    return {
        "pose_max_abs": float(pose.abs().max().item()),
        "velocity_max_abs": float(
            (reference.velocity_W - candidate.velocity_W).abs().max().item()
        ),
        "acc_bias_max_abs": float(
            (reference.acc_bias - candidate.acc_bias).abs().max().item()
        ),
        "gyro_bias_max_abs": float(
            (reference.gyro_bias - candidate.gyro_bias).abs().max().item()
        ),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    payload = torch.load(args.packet, map_location="cpu", weights_only=False)
    edges = list(payload["edges"][: int(args.edge_count)])
    if len(edges) != int(args.edge_count):
        raise ValueError("captured packet contains fewer edges than requested")
    reader = CompressedUVDFactorCacheReader(args.cache)

    device_rows: dict[str, list[dict[str, Any]]] = {}
    device_results: dict[str, list[TwoStateVIOResult]] = {}
    for name in args.devices:
        rows, results = run_device(
            name=name,
            edges=edges,
            reader=reader,
            settings=payload["solver_settings"],
        )
        device_rows[name] = rows
        device_results[name] = results

    all_rows = [row for name in args.devices for row in device_rows[name]]
    write_csv(args.output / "device_timing_per_edge.csv", all_rows)

    warmup = min(max(int(args.warmup_edges), 0), max(len(edges) - 1, 0))
    summary: dict[str, Any] = {
        "contract": {
            "packet": str(args.packet.resolve()),
            "cache": str(args.cache.resolve()),
            "edge_count": len(edges),
            "warmup_edges_excluded": warmup,
            "dtype": str(DTYPE),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "torch_num_threads": torch.get_num_threads(),
            "torch_num_interop_threads": torch.get_num_interop_threads(),
            "gpu": (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            ),
            "devices": list(args.devices),
            "solver_settings": payload["solver_settings"],
        },
        "timing": {},
    }
    for name, rows in device_rows.items():
        steady = rows[warmup:]
        summary["timing"][name] = {
            field: distribution([float(row[field]) for row in steady])
            for field in (
                "transfer_in_ms",
                "solve_ms",
                "transfer_out_ms",
                "end_to_end_ms",
            )
        }
        summary["timing"][name]["converged_rate"] = statistics.fmean(
            int(row["converged"]) for row in rows
        )
        summary["timing"][name]["iterations"] = distribution(
            [float(row["iterations"]) for row in rows]
        )

    comparison_rows: list[dict[str, Any]] = []
    if "cpu" in device_results and "cuda" in device_results:
        for edge, cpu, cuda in zip(
            edges, device_results["cpu"], device_results["cuda"]
        ):
            state_i = state_difference(cpu.state_i, cuda.state_i)
            state_j = state_difference(cpu.state_j, cuda.state_j)
            comparison_rows.append(
                {
                    "frame_i": int(edge["frame_i"]),
                    "frame_j": int(edge["frame_j"]),
                    **{f"state_i_{key}": value for key, value in state_i.items()},
                    **{f"state_j_{key}": value for key, value in state_j.items()},
                    "final_cost_abs_difference": abs(
                        float(cpu.final_cost) - float(cuda.final_cost)
                    ),
                    "iterations_equal": int(cpu.iterations == cuda.iterations),
                    "converged_equal": int(cpu.converged == cuda.converged),
                }
            )
        write_csv(args.output / "cpu_vs_cuda_per_edge.csv", comparison_rows)
        numeric = [
            key
            for key in comparison_rows[0]
            if key.endswith("max_abs") or key.endswith("abs_difference")
        ]
        summary["cpu_vs_cuda"] = {
            "maximum_difference": max(
                float(row[key]) for row in comparison_rows for key in numeric
            ),
            "per_field_maximum": {
                key: max(float(row[key]) for row in comparison_rows)
                for key in numeric
            },
            "all_iterations_equal": all(
                bool(row["iterations_equal"]) for row in comparison_rows
            ),
            "all_convergence_equal": all(
                bool(row["converged_equal"]) for row in comparison_rows
            ),
            "median_end_to_end_speedup_cuda_over_cpu": (
                summary["timing"]["cpu"]["end_to_end_ms"]["median"]
                / summary["timing"]["cuda"]["end_to_end_ms"]["median"]
            ),
        }

    (args.output / "device_benchmark_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
