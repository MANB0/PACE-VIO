#!/usr/bin/env python3
"""Compare rebuilt and cached UVD factors on identical captured IMU edges."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
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
DEFAULT_OUTPUT = ROOT / "analysis_compressed_uvd_production_equivalence_20260720"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, default=DEFAULT_PACKET)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--edge-count", type=int, default=209)
    return parser.parse_args()


def _solver(settings: dict) -> TwoStateVIOSolver:
    return TwoStateVIOSolver(**{key: settings[key] for key in (
        "max_iterations",
        "initial_damping",
        "step_tolerance",
        "cost_tolerance",
        "covariance_eigenvalue_floor",
        "marginalization_eigenvalue_floor",
    )})


def _factor(
    incoming: TwoStateVIOProblem,
    reader: CompressedUVDFactorCacheReader,
    frame_i: int,
    frame_j: int,
    *,
    cached: bool,
):
    visual = incoming.visual_pose
    if not isinstance(visual, UVDFactor):
        raise TypeError("captured problem does not contain UVD observations")
    visual_hash = reader.visual_hashes[frame_i]
    packet = reader.load_pair(frame_i, frame_j, str(visual_hash))
    if cached:
        return (
            linearized_uvd_pose_factor_from_normal_equations(
                packet.reference_CjCi,
                packet.hessian,
                packet.gradient,
                visual.extrinsic_CI,
            ),
            packet.hessian,
            packet.gradient,
        )
    linearization = linearize_uvd_relative_pose_factor(
        packet.reference_CjCi,
        visual,
        marginal_mode="full",
    )
    return linearization.factor, linearization.full_hessian, linearization.full_gradient


def _run_branch(
    edges: list[dict],
    reader: CompressedUVDFactorCacheReader,
    settings: dict,
    *,
    cached: bool,
) -> tuple[list[TwoStateVIOResult], list[dict]]:
    solver = _solver(settings)
    previous: TwoStateVIOResult | None = None
    results: list[TwoStateVIOResult] = []
    rows: list[dict] = []
    for edge_id, edge in enumerate(edges):
        incoming = (
            clone_problem(edge["problem"])
            if previous is None
            else future_problem(edge["problem"], previous)
        )
        build_start = time.perf_counter()
        factor, hessian, gradient = _factor(
            incoming,
            reader,
            int(edge["frame_i"]),
            int(edge["frame_j"]),
            cached=cached,
        )
        build_ms = (time.perf_counter() - build_start) * 1000.0
        problem = replace(incoming, visual_pose=factor)
        solve_start = time.perf_counter()
        result = solver.solve(problem)
        solve_ms = (time.perf_counter() - solve_start) * 1000.0
        rows.append({
            "branch": "cached" if cached else "rebuilt",
            "edge_id": edge_id,
            "frame_i": int(edge["frame_i"]),
            "frame_j": int(edge["frame_j"]),
            "build_ms": build_ms,
            "solve_ms": solve_ms,
            "iterations": int(result.iterations),
            "converged": bool(result.converged),
            "final_cost": float(result.final_cost),
            "hessian_norm": float(torch.linalg.matrix_norm(hessian).item()),
            "gradient_norm": float(torch.linalg.vector_norm(gradient).item()),
        })
        results.append(result)
        previous = result
    return results, rows


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    payload = torch.load(args.packet, map_location="cpu", weights_only=False)
    edges = list(payload["edges"][: int(args.edge_count)])
    if len(edges) != int(args.edge_count):
        raise ValueError("captured packet does not contain the requested edge count")
    reader = CompressedUVDFactorCacheReader(args.cache)

    rebuilt_results, rebuilt_rows = _run_branch(
        edges, reader, payload["solver_settings"], cached=False
    )
    cached_results, cached_rows = _run_branch(
        edges, reader, payload["solver_settings"], cached=True
    )

    comparison: list[dict] = []
    for edge, rebuilt, cached in zip(edges, rebuilt_results, cached_results):
        pose_i_error = (
            pp.SE3(rebuilt.state_i.pose_WB).Inv() @ pp.SE3(cached.state_i.pose_WB)
        ).Log().tensor().reshape(6)
        pose_j_error = (
            pp.SE3(rebuilt.state_j.pose_WB).Inv() @ pp.SE3(cached.state_j.pose_WB)
        ).Log().tensor().reshape(6)
        comparison.append({
            "frame_i": int(edge["frame_i"]),
            "frame_j": int(edge["frame_j"]),
            "pose_i_tangent_max_abs": float(pose_i_error.abs().max().item()),
            "pose_j_tangent_max_abs": float(pose_j_error.abs().max().item()),
            "velocity_i_max_abs": float((rebuilt.state_i.velocity_W - cached.state_i.velocity_W).abs().max().item()),
            "velocity_j_max_abs": float((rebuilt.state_j.velocity_W - cached.state_j.velocity_W).abs().max().item()),
            "acc_bias_i_max_abs": float((rebuilt.state_i.acc_bias - cached.state_i.acc_bias).abs().max().item()),
            "acc_bias_j_max_abs": float((rebuilt.state_j.acc_bias - cached.state_j.acc_bias).abs().max().item()),
            "gyro_bias_i_max_abs": float((rebuilt.state_i.gyro_bias - cached.state_i.gyro_bias).abs().max().item()),
            "gyro_bias_j_max_abs": float((rebuilt.state_j.gyro_bias - cached.state_j.gyro_bias).abs().max().item()),
            "final_cost_abs_difference": abs(float(rebuilt.final_cost) - float(cached.final_cost)),
            "iterations_equal": int(rebuilt.iterations == cached.iterations),
            "converged_equal": int(rebuilt.converged == cached.converged),
        })

    args.output.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output / "factor_branch_timing_per_edge.csv", rebuilt_rows + cached_rows)
    _write_csv(args.output / "cached_vs_rebuilt_state_per_edge.csv", comparison)
    rebuilt_build = np.asarray([row["build_ms"] for row in rebuilt_rows])
    cached_build = np.asarray([row["build_ms"] for row in cached_rows])
    numeric_fields = [
        key for key in comparison[0]
        if key.endswith("max_abs") or key.endswith("abs_difference")
    ]
    summary = {
        "edge_count": len(edges),
        "all_converged_rebuilt": all(row["converged"] for row in rebuilt_rows),
        "all_converged_cached": all(row["converged"] for row in cached_rows),
        "all_iterations_equal": all(row["iterations_equal"] for row in comparison),
        "maximum_numeric_difference": max(
            max(float(row[key]) for row in comparison) for key in numeric_fields
        ),
        "per_field_maximum_difference": {
            key: max(float(row[key]) for row in comparison) for key in numeric_fields
        },
        "rebuilt_factor_build_ms_median": float(np.median(rebuilt_build)),
        "cached_factor_build_ms_median": float(np.median(cached_build)),
        "factor_build_speedup": float(np.median(rebuilt_build) / np.median(cached_build)),
    }
    (args.output / "equivalence_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    report = f"""# 压缩 UVD 生产接口等价性审计

- 固定输入：同一 {len(edges)} 条 IMU edge、同一初值、同一 prior 传播、同一 LM 参数。
- 现场重建分支：每边从点级 UVD 重新计算 Jacobian/Hessian/gradient。
- 缓存分支：每边只读取 sidecar 的参考位姿、Hessian 和 gradient。
- 最大状态/代价数值差：{summary['maximum_numeric_difference']:.6g}。
- 迭代次数逐边完全相同：{summary['all_iterations_equal']}。
- 两分支全部收敛：{summary['all_converged_rebuilt'] and summary['all_converged_cached']}。
- factor build 中位耗时：{summary['rebuilt_factor_build_ms_median']:.3f} ms -> {summary['cached_factor_build_ms_median']:.3f} ms。
- 仅 factor build 加速：{summary['factor_build_speedup']:.2f}x。

结论：此测试只证明压缩接口在固定 measurement/factor 输入下等价；它不证明在线 bias feedback 后生成的后续 IMU factors 与旧 U1 捕获包相同。
"""
    (args.output / "compressed_uvd_production_equivalence_report_cn.md").write_text(
        report, encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
