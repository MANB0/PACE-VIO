#!/usr/bin/env python3
"""Analyze optimizer breakpoint traces against GT, pure MACVO, and IMU-only.

This script is intentionally read-only with respect to run results. It parses
``optimization_breakpoint_trace.jsonl`` files produced by the optimizer trace
hook and emits frame-by-frame comparison tables.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


STAGES = (
    "init_before_lm",
    "after_lm_before_refine",
    "after_refine_before_writeback",
    "map_after_writeback",
)


def _float(value: str | None) -> float:
    if value in (None, ""):
        return float("nan")
    return float(value)


def _vec3(row: dict[str, str], prefix: str) -> tuple[float, float, float]:
    return (_float(row[f"{prefix}x"]), _float(row[f"{prefix}y"]), _float(row[f"{prefix}z"]))


def _quat(row: dict[str, str], prefix: str) -> tuple[float, float, float, float]:
    return (
        _float(row[f"{prefix}x"]),
        _float(row[f"{prefix}y"]),
        _float(row[f"{prefix}z"]),
        _float(row[f"{prefix}w"]),
    )


def norm3(v: tuple[float, float, float]) -> float:
    return math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])


def sub3(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def dot3(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def quat_angle_deg(
    q1: tuple[float, float, float, float], q2: tuple[float, float, float, float]
) -> float:
    dot = abs(q1[0] * q2[0] + q1[1] * q2[1] + q1[2] * q2[2] + q1[3] * q2[3])
    dot = max(-1.0, min(1.0, dot))
    return math.degrees(2.0 * math.acos(dot))


def internal_pose_to_nwu(pose: list[float]) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    # Optimizer pose trace is in internal NED. Joined result CSVs are exported in NWU.
    return (pose[0], -pose[1], -pose[2]), (pose[3], -pose[4], -pose[5], pose[6])


def load_joined(path: Path) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    with path.open(newline="") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            ts = int(row["timestamp_ns"])
            rows[ts] = {
                "est_p": (_float(row["tx_est"]), _float(row["ty_est"]), _float(row["tz_est"])),
                "est_q": (_float(row["qx_est"]), _float(row["qy_est"]), _float(row["qz_est"]), _float(row["qw_est"])),
                "gt_p": (_float(row["tx_gt"]), _float(row["ty_gt"]), _float(row["tz_gt"])),
                "gt_q": (_float(row["qx_gt"]), _float(row["qy_gt"]), _float(row["qz_gt"]), _float(row["qw_gt"])),
                "err_m": _float(row.get("err_m")),
            }
    return rows


def residual_fields(snapshot: dict[str, Any]) -> dict[str, float]:
    residual = snapshot.get("residual") or {}
    return {
        "raw_norm": float(residual.get("raw_norm", float("nan"))),
        "visual_raw_norm": float(residual.get("visual_raw_norm", float("nan"))),
        "imu_raw_norm": float(residual.get("imu_raw_norm", float("nan"))),
        "bias_raw_norm": float(residual.get("bias_raw_norm", float("nan"))),
        "visual_rows": float(residual.get("visual_rows", float("nan"))),
        "imu_rows": float(residual.get("imu_rows", float("nan"))),
        "bias_rows": float(residual.get("bias_rows", float("nan"))),
    }


def stage_pose(snapshot: dict[str, Any]) -> list[float]:
    if "poses" in snapshot:
        return snapshot["poses"][-1]
    pose = snapshot["pose2opt"]
    if pose and isinstance(pose[0], list):
        return pose[-1]
    return pose


def nearest_basis_projection(
    origin: tuple[float, float, float],
    target: tuple[float, float, float],
    moved: tuple[float, float, float],
) -> float:
    basis = sub3(target, origin)
    denom = dot3(basis, basis)
    if denom <= 1e-18:
        return float("nan")
    return dot3(moved, basis) / denom


def trace_files(results_root: Path, scene: str) -> list[Path]:
    return sorted(results_root.glob(f"trial_*/*/{scene}/optimization_breakpoint_trace.jsonl"))


def analyze_trace_file(
    path: Path,
    scene: str,
    method: str,
    method_joined: dict[int, dict[str, Any]],
    pure: dict[int, dict[str, Any]],
    imu: dict[int, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    long_rows: list[dict[str, Any]] = []
    wide_rows: list[dict[str, Any]] = []

    with path.open() as fp:
        for line_no, line in enumerate(fp, start=1):
            record = json.loads(line)
            frame_idx = int(record["frame_idx"])
            map_entries = record.get("map_after_writeback") or []
            if not map_entries:
                continue
            map_last = map_entries[-1]
            ts = int(map_last["timestamp_ns"])
            if ts not in method_joined:
                continue

            gt_p = method_joined[ts]["gt_p"]
            gt_q = method_joined[ts]["gt_q"]
            method_p = method_joined[ts]["est_p"]
            pure_p = pure.get(ts, {}).get("est_p", (float("nan"), float("nan"), float("nan")))
            imu_p = imu.get(ts, {}).get("est_p", (float("nan"), float("nan"), float("nan")))
            pure_q = pure.get(ts, {}).get("est_q", (float("nan"), float("nan"), float("nan"), float("nan")))
            imu_q = imu.get(ts, {}).get("est_q", (float("nan"), float("nan"), float("nan"), float("nan")))

            stage_data: dict[str, dict[str, Any]] = {}
            for snapshot in record["trace"]["snapshots"]:
                stage = snapshot["stage"]
                p, q = internal_pose_to_nwu(stage_pose(snapshot))
                res = residual_fields(snapshot)
                row = {
                    "scene": scene,
                    "method": method,
                    "frame_idx": frame_idx,
                    "timestamp_ns": ts,
                    "stage": stage,
                    "x": p[0],
                    "y": p[1],
                    "z": p[2],
                    "qx": q[0],
                    "qy": q[1],
                    "qz": q[2],
                    "qw": q[3],
                    "gt_x": gt_p[0],
                    "gt_y": gt_p[1],
                    "gt_z": gt_p[2],
                    "err_to_gt_m": norm3(sub3(p, gt_p)),
                    "err_rot_to_gt_deg": quat_angle_deg(q, gt_q),
                    "dist_to_pure_m": norm3(sub3(p, pure_p)),
                    "dist_to_imu_m": norm3(sub3(p, imu_p)),
                    "rot_dist_to_pure_deg": quat_angle_deg(q, pure_q),
                    "rot_dist_to_imu_deg": quat_angle_deg(q, imu_q),
                    **res,
                }
                long_rows.append(row)
                stage_data[stage] = row

            p_map, q_map = internal_pose_to_nwu(map_last["pose"])
            map_row = {
                "scene": scene,
                "method": method,
                "frame_idx": frame_idx,
                "timestamp_ns": ts,
                "stage": "map_after_writeback",
                "x": p_map[0],
                "y": p_map[1],
                "z": p_map[2],
                "qx": q_map[0],
                "qy": q_map[1],
                "qz": q_map[2],
                "qw": q_map[3],
                "gt_x": gt_p[0],
                "gt_y": gt_p[1],
                "gt_z": gt_p[2],
                "err_to_gt_m": norm3(sub3(p_map, gt_p)),
                "err_rot_to_gt_deg": quat_angle_deg(q_map, gt_q),
                "dist_to_pure_m": norm3(sub3(p_map, pure_p)),
                "dist_to_imu_m": norm3(sub3(p_map, imu_p)),
                "rot_dist_to_pure_deg": quat_angle_deg(q_map, pure_q),
                "rot_dist_to_imu_deg": quat_angle_deg(q_map, imu_q),
                "raw_norm": float("nan"),
                "visual_raw_norm": float("nan"),
                "imu_raw_norm": float("nan"),
                "bias_raw_norm": float("nan"),
                "visual_rows": float("nan"),
                "imu_rows": float("nan"),
                "bias_rows": float("nan"),
            }
            long_rows.append(map_row)
            stage_data["map_after_writeback"] = map_row

            init = stage_data["init_before_lm"]
            final = stage_data["after_refine_before_writeback"]
            map_stage = stage_data["map_after_writeback"]
            init_p = (init["x"], init["y"], init["z"])
            final_p = (final["x"], final["y"], final["z"])
            moved = sub3(final_p, init_p)
            wide_rows.append(
                {
                    "scene": scene,
                    "method": method,
                    "frame_idx": frame_idx,
                    "timestamp_ns": ts,
                    "gt_x": gt_p[0],
                    "gt_y": gt_p[1],
                    "gt_z": gt_p[2],
                    "pure_x": pure_p[0],
                    "pure_y": pure_p[1],
                    "pure_z": pure_p[2],
                    "imu_x": imu_p[0],
                    "imu_y": imu_p[1],
                    "imu_z": imu_p[2],
                    "method_output_x": method_p[0],
                    "method_output_y": method_p[1],
                    "method_output_z": method_p[2],
                    "init_x": init["x"],
                    "init_y": init["y"],
                    "init_z": init["z"],
                    "final_x": final["x"],
                    "final_y": final["y"],
                    "final_z": final["z"],
                    "map_x": map_stage["x"],
                    "map_y": map_stage["y"],
                    "map_z": map_stage["z"],
                    "init_err_gt_m": init["err_to_gt_m"],
                    "final_err_gt_m": final["err_to_gt_m"],
                    "map_err_gt_m": map_stage["err_to_gt_m"],
                    "pure_err_gt_m": pure.get(ts, {}).get("err_m", float("nan")),
                    "imu_err_gt_m": imu.get(ts, {}).get("err_m", float("nan")),
                    "init_dist_to_pure_m": init["dist_to_pure_m"],
                    "init_dist_to_imu_m": init["dist_to_imu_m"],
                    "final_dist_to_pure_m": final["dist_to_pure_m"],
                    "final_dist_to_imu_m": final["dist_to_imu_m"],
                    "map_dist_to_pure_m": map_stage["dist_to_pure_m"],
                    "map_dist_to_imu_m": map_stage["dist_to_imu_m"],
                    "lm_move_m": norm3(
                        sub3(
                            (
                                stage_data["after_lm_before_refine"]["x"],
                                stage_data["after_lm_before_refine"]["y"],
                                stage_data["after_lm_before_refine"]["z"],
                            ),
                            init_p,
                        )
                    ),
                    "refine_move_m": norm3(
                        sub3(
                            final_p,
                            (
                                stage_data["after_lm_before_refine"]["x"],
                                stage_data["after_lm_before_refine"]["y"],
                                stage_data["after_lm_before_refine"]["z"],
                            ),
                        )
                    ),
                    "final_move_from_init_m": norm3(moved),
                    "final_error_change_vs_init_m": final["err_to_gt_m"] - init["err_to_gt_m"],
                    "final_projection_toward_imu": nearest_basis_projection(init_p, imu_p, moved),
                    "final_projection_toward_pure": nearest_basis_projection(init_p, pure_p, moved),
                    "final_projection_toward_gt": nearest_basis_projection(init_p, gt_p, moved),
                    "init_closer_to_imu_than_pure": init["dist_to_imu_m"] < init["dist_to_pure_m"],
                    "final_closer_to_imu_than_pure": final["dist_to_imu_m"] < final["dist_to_pure_m"],
                    "map_closer_to_imu_than_pure": map_stage["dist_to_imu_m"] < map_stage["dist_to_pure_m"],
                    "raw_norm_init": init["raw_norm"],
                    "raw_norm_final": final["raw_norm"],
                    "visual_norm_init": init["visual_raw_norm"],
                    "visual_norm_final": final["visual_raw_norm"],
                    "imu_norm_init": init["imu_raw_norm"],
                    "imu_norm_final": final["imu_raw_norm"],
                    "bias_norm_final": final["bias_raw_norm"],
                    "line_no": line_no,
                }
            )
    return long_rows, wide_rows


def finite_values(rows: list[dict[str, Any]], key: str) -> list[float]:
    values = []
    for row in rows:
        value = row.get(key, float("nan"))
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)) and math.isfinite(value):
            values.append(float(value))
    return values


def safe_mean(rows: list[dict[str, Any]], key: str) -> float:
    values = finite_values(rows, key)
    return mean(values) if values else float("nan")


def pct(condition_count: int, total: int) -> float:
    return 100.0 * condition_count / total if total else float("nan")


def summarize(wide_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in wide_rows:
        grouped[row["method"]].append(row)

    summary_rows = []
    for method, rows in grouped.items():
        total = len(rows)
        summary_rows.append(
            {
                "method": method,
                "frames": total,
                "mean_pure_err_gt_m": safe_mean(rows, "pure_err_gt_m"),
                "mean_imu_err_gt_m": safe_mean(rows, "imu_err_gt_m"),
                "mean_init_err_gt_m": safe_mean(rows, "init_err_gt_m"),
                "mean_final_err_gt_m": safe_mean(rows, "final_err_gt_m"),
                "mean_map_err_gt_m": safe_mean(rows, "map_err_gt_m"),
                "mean_final_error_change_vs_init_m": safe_mean(rows, "final_error_change_vs_init_m"),
                "mean_final_dist_to_pure_m": safe_mean(rows, "final_dist_to_pure_m"),
                "mean_final_dist_to_imu_m": safe_mean(rows, "final_dist_to_imu_m"),
                "mean_map_dist_to_pure_m": safe_mean(rows, "map_dist_to_pure_m"),
                "mean_map_dist_to_imu_m": safe_mean(rows, "map_dist_to_imu_m"),
                "mean_lm_move_m": safe_mean(rows, "lm_move_m"),
                "mean_refine_move_m": safe_mean(rows, "refine_move_m"),
                "mean_final_move_from_init_m": safe_mean(rows, "final_move_from_init_m"),
                "mean_final_projection_toward_imu": safe_mean(rows, "final_projection_toward_imu"),
                "mean_final_projection_toward_pure": safe_mean(rows, "final_projection_toward_pure"),
                "mean_final_projection_toward_gt": safe_mean(rows, "final_projection_toward_gt"),
                "pct_init_closer_to_imu_than_pure": pct(
                    sum(1 for row in rows if row["init_closer_to_imu_than_pure"]), total
                ),
                "pct_final_closer_to_imu_than_pure": pct(
                    sum(1 for row in rows if row["final_closer_to_imu_than_pure"]), total
                ),
                "pct_map_closer_to_imu_than_pure": pct(
                    sum(1 for row in rows if row["map_closer_to_imu_than_pure"]), total
                ),
                "pct_final_improves_gt_vs_init": pct(
                    sum(1 for row in rows if row["final_error_change_vs_init_m"] < 0.0), total
                ),
                "mean_visual_norm_init": safe_mean(rows, "visual_norm_init"),
                "mean_visual_norm_final": safe_mean(rows, "visual_norm_final"),
                "mean_imu_norm_init": safe_mean(rows, "imu_norm_init"),
                "mean_imu_norm_final": safe_mean(rows, "imu_norm_final"),
            }
        )
    return summary_rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: Any, digits: int = 4) -> str:
    if isinstance(value, bool):
        return "是" if value else "否"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            return "nan"
        return f"{value:.{digits}f}"
    return str(value)


def write_markdown(
    path: Path,
    scene: str,
    results_root: Path,
    analysis_root: Path,
    summary_rows: list[dict[str, Any]],
    wide_rows: list[dict[str, Any]],
) -> None:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in wide_rows:
        grouped[row["method"]].append(row)

    lines = [
        f"# {scene} 逐帧断点调试结论",
        "",
        "## 数据来源",
        "",
        f"- 断点 trace：`{results_root}`",
        f"- 轨迹对比输出：`{analysis_root}`",
        "- 断点阶段：`init_before_lm`、`after_lm_before_refine`、`after_refine_before_writeback`、`map_after_writeback`。",
        "- 坐标统一：优化器 trace 内部 pose 从 NED 转成 NWU 后再和 GT / pure MACVO / IMU-only 比较。",
        "",
        "## 方法汇总",
        "",
        "| 方法 | 帧数 | pure 平均误差 m | IMU-only 平均误差 m | 初值平均误差 m | 优化后平均误差 m | 写回平均误差 m | 优化后距 pure m | 优化后距 IMU m | 写回距 pure m | 写回距 IMU m | 优化后更靠近 IMU 的帧比例 | 优化后相对初值改善 GT 的帧比例 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["method"],
                    fmt(row["frames"], 0),
                    fmt(row["mean_pure_err_gt_m"]),
                    fmt(row["mean_imu_err_gt_m"]),
                    fmt(row["mean_init_err_gt_m"]),
                    fmt(row["mean_final_err_gt_m"]),
                    fmt(row["mean_map_err_gt_m"]),
                    fmt(row["mean_final_dist_to_pure_m"]),
                    fmt(row["mean_final_dist_to_imu_m"]),
                    fmt(row["mean_map_dist_to_pure_m"]),
                    fmt(row["mean_map_dist_to_imu_m"]),
                    fmt(row["pct_final_closer_to_imu_than_pure"], 1) + "%",
                    fmt(row["pct_final_improves_gt_vs_init"], 1) + "%",
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## 断点阶段解释",
            "",
            "- `init_before_lm`：该帧进入优化器前的初始状态。如果这里已经贴近 IMU-only，问题发生在优化初值/预测阶段，而不是 LM 求解阶段。",
            "- `after_lm_before_refine`：LM 主优化结束后的状态。它和初值的差值说明优化器本身把状态往哪里拉。",
            "- `after_refine_before_writeback`：refine 后、写回地图前的最终优化状态。它和 `map_after_writeback` 的一致性说明写回链路是否额外引入偏差。",
            "- `map_after_writeback`：真正写回地图并导出轨迹的状态。",
            "",
            "## 关键逐帧证据",
            "",
        ]
    )

    for method, rows in grouped.items():
        worst = sorted(rows, key=lambda row: row["map_err_gt_m"], reverse=True)[:8]
        lines.extend(
            [
                f"### {method}",
                "",
                "| frame | GT x | GT y | pure y | IMU y | init y | final y | map y | final-GT m | final 距 pure m | final 距 IMU m | LM 移动 m | refine 移动 m | visual norm 初/后 | IMU norm 初/后 |",
                "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in worst:
            lines.append(
                "| "
                + " | ".join(
                    [
                        fmt(row["frame_idx"], 0),
                        fmt(row["gt_x"]),
                        fmt(row["gt_y"]),
                        fmt(row["pure_y"]),
                        fmt(row["imu_y"]),
                        fmt(row["init_y"]),
                        fmt(row["final_y"]),
                        fmt(row["map_y"]),
                        fmt(row["final_err_gt_m"]),
                        fmt(row["final_dist_to_pure_m"]),
                        fmt(row["final_dist_to_imu_m"]),
                        fmt(row["lm_move_m"]),
                        fmt(row["refine_move_m"], 6),
                        f"{fmt(row['visual_norm_init'])}/{fmt(row['visual_norm_final'])}",
                        f"{fmt(row['imu_norm_init'], 6)}/{fmt(row['imu_norm_final'], 6)}",
                    ]
                )
                + " |"
            )
        lines.append("")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", default="clear_rectangle_normal_noise")
    parser.add_argument("--results-root", type=Path, default=Path("Results/frame_debug_trace_150_20260708"))
    parser.add_argument(
        "--analysis-root", type=Path, default=Path("analysis_frame_debug_trace_150_20260708")
    )
    parser.add_argument(
        "--reference-analysis-root",
        type=Path,
        default=Path("analysis_local_ba_writeback_validation_20260708"),
    )
    args = parser.parse_args()

    scene_dir = args.analysis_root / args.scene
    traj_dir = scene_dir / "trajectories"
    ref_traj_dir = args.reference_analysis_root / args.scene / "trajectories"

    pure = load_joined(ref_traj_dir / f"{args.scene}_pure_macvo_joined.csv")
    imu = load_joined(ref_traj_dir / f"{args.scene}_imu_only_mechanization_joined.csv")

    long_rows: list[dict[str, Any]] = []
    wide_rows: list[dict[str, Any]] = []
    for trace_path in trace_files(args.results_root, args.scene):
        method = trace_path.parts[-3]
        method_joined_path = traj_dir / f"{args.scene}_{method}_joined.csv"
        if not method_joined_path.exists():
            raise FileNotFoundError(method_joined_path)
        method_joined = load_joined(method_joined_path)
        method_long, method_wide = analyze_trace_file(
            trace_path, args.scene, method, method_joined, pure, imu
        )
        long_rows.extend(method_long)
        wide_rows.extend(method_wide)

    summary_rows = summarize(wide_rows)
    write_csv(scene_dir / "frame_breakpoint_stage_long.csv", long_rows)
    write_csv(scene_dir / "frame_breakpoint_comparison.csv", wide_rows)
    write_csv(scene_dir / "frame_breakpoint_summary.csv", summary_rows)
    write_markdown(
        scene_dir / "frame_breakpoint_summary.md",
        args.scene,
        args.results_root,
        scene_dir,
        summary_rows,
        wide_rows,
    )

    print(f"Wrote {scene_dir / 'frame_breakpoint_stage_long.csv'}")
    print(f"Wrote {scene_dir / 'frame_breakpoint_comparison.csv'}")
    print(f"Wrote {scene_dir / 'frame_breakpoint_summary.csv'}")
    print(f"Wrote {scene_dir / 'frame_breakpoint_summary.md'}")


if __name__ == "__main__":
    main()
