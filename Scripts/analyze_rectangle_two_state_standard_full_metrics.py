#!/usr/bin/env python3
"""Compute absolute and relative trajectory metrics for the full rectangle run."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


WORKDIR = Path("/home/admin1/macvo-dev")
SCENE = "clear_stop_turn_rectangle_truth_bias_no_noise"
GT_PATH = (
    Path("/mnt/e/文档/holoocean/code/recordings")
    / "batch_clear_truth_paths_20260713_static63_variants"
    / SCENE
    / "ref_pose.csv"
)
TRAJECTORIES = {
    "pure_macvo": (
        WORKDIR
        / "Results/visual_factor_cache_static63_unique_source_20260713/trial_1/pure_macvo"
        / "clear_stop_turn_rectangle_truth_normal_noise/poses.csv"
    ),
    "two_state_D_standard_full": (
        WORKDIR
        / "Results/rectangle_bias_no_noise_two_state_standard_full_20260715"
        / "trial_1/vio_two_state_fixed_lag_standard_full"
        / SCENE
        / "poses.csv"
    ),
}
OUTDIR = WORKDIR / "analysis_rectangle_two_state_standard_full_20260715"
RPE_LAGS = (1, 30, 150)


@dataclass(frozen=True)
class PoseSeries:
    timestamps_ns: np.ndarray
    positions: np.ndarray
    rotations: np.ndarray


def read_pose_series(path: Path) -> PoseSeries:
    timestamps: list[int] = []
    positions: list[list[float]] = []
    quaternions: list[list[float]] = []
    with path.open("r", newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            timestamp_key = "timestamp_ns" if "timestamp_ns" in row else "timestamp"
            position_keys = ("tx", "ty", "tz") if "tx" in row else ("x", "y", "z")
            timestamps.append(int(float(row[timestamp_key])))
            positions.append([float(row[key]) for key in position_keys])
            quaternions.append(
                [float(row["qx"]), float(row["qy"]), float(row["qz"]), float(row["qw"])]
            )
    if not timestamps:
        raise ValueError(f"empty pose file: {path}")
    quaternion_array = np.asarray(quaternions, dtype=np.float64)
    quaternion_norm = np.linalg.norm(quaternion_array, axis=1)
    if np.any(quaternion_norm <= 0.0):
        raise ValueError(f"zero quaternion in {path}")
    quaternion_array /= quaternion_norm[:, None]
    return PoseSeries(
        timestamps_ns=np.asarray(timestamps, dtype=np.int64),
        positions=np.asarray(positions, dtype=np.float64),
        rotations=Rotation.from_quat(quaternion_array).as_matrix(),
    )


def synchronize(gt: PoseSeries, estimate: PoseSeries) -> tuple[PoseSeries, PoseSeries]:
    gt_index = {int(timestamp): index for index, timestamp in enumerate(gt.timestamps_ns)}
    est_index = {int(timestamp): index for index, timestamp in enumerate(estimate.timestamps_ns)}
    common = sorted(set(gt_index) & set(est_index))
    if len(common) < 2:
        raise ValueError("fewer than two exactly matched timestamps")
    gt_indices = np.asarray([gt_index[timestamp] for timestamp in common], dtype=np.int64)
    est_indices = np.asarray([est_index[timestamp] for timestamp in common], dtype=np.int64)
    timestamps = np.asarray(common, dtype=np.int64)
    return (
        PoseSeries(timestamps, gt.positions[gt_indices], gt.rotations[gt_indices]),
        PoseSeries(timestamps, estimate.positions[est_indices], estimate.rotations[est_indices]),
    )


def scalar_stats(values: np.ndarray, prefix: str) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    return {
        f"{prefix}_rmse": float(np.sqrt(np.mean(np.square(values)))),
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_median": float(np.median(values)),
        f"{prefix}_p95": float(np.percentile(values, 95.0)),
        f"{prefix}_max": float(np.max(values)),
        f"{prefix}_final": float(values[-1]),
    }


def vector_axis_stats(values: np.ndarray, prefix: str) -> dict[str, float]:
    result: dict[str, float] = {}
    for axis, index in zip("xyz", range(3)):
        component = values[:, index]
        result[f"{prefix}_{axis}_rmse"] = float(np.sqrt(np.mean(np.square(component))))
        result[f"{prefix}_{axis}_mean"] = float(np.mean(component))
        result[f"{prefix}_{axis}_max_abs"] = float(np.max(np.abs(component)))
        result[f"{prefix}_{axis}_final"] = float(component[-1])
    return result


def rigid_align(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    source_center = np.mean(source, axis=0)
    target_center = np.mean(target, axis=0)
    centered_source = source - source_center
    centered_target = target - target_center
    u, _, vt = np.linalg.svd(centered_source.T @ centered_target)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0.0:
        vt[-1, :] *= -1.0
        rotation = vt.T @ u.T
    translation = target_center - rotation @ source_center
    return rotation, translation


def relative_errors(
    gt: PoseSeries,
    estimate: PoseSeries,
    lag: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if lag <= 0 or lag >= len(gt.timestamps_ns):
        raise ValueError(f"invalid RPE lag {lag} for {len(gt.timestamps_ns)} poses")
    gt_r_i = gt.rotations[:-lag]
    gt_r_j = gt.rotations[lag:]
    est_r_i = estimate.rotations[:-lag]
    est_r_j = estimate.rotations[lag:]
    gt_relative_r = np.einsum("nji,njk->nik", gt_r_i, gt_r_j)
    est_relative_r = np.einsum("nji,njk->nik", est_r_i, est_r_j)

    gt_world_delta = gt.positions[lag:] - gt.positions[:-lag]
    est_world_delta = estimate.positions[lag:] - estimate.positions[:-lag]
    gt_relative_t = np.einsum("nji,nj->ni", gt_r_i, gt_world_delta)
    est_relative_t = np.einsum("nji,nj->ni", est_r_i, est_world_delta)
    translation_error = np.linalg.norm(est_relative_t - gt_relative_t, axis=1)

    rotation_error_matrix = np.einsum("nji,njk->nik", est_relative_r, gt_relative_r)
    rotation_error_deg = np.degrees(
        np.linalg.norm(Rotation.from_matrix(rotation_error_matrix).as_rotvec(), axis=1)
    )
    delta_time_s = (gt.timestamps_ns[lag:] - gt.timestamps_ns[:-lag]) * 1e-9
    return translation_error, rotation_error_deg, delta_time_s


def evaluate_trajectory(method: str, gt: PoseSeries, estimate: PoseSeries) -> tuple[dict, list[dict], dict]:
    gt, estimate = synchronize(gt, estimate)
    position_error_vector = estimate.positions - gt.positions
    position_error = np.linalg.norm(position_error_vector, axis=1)
    absolute_rotation_error = np.einsum("nji,njk->nik", gt.rotations, estimate.rotations)
    absolute_rotation_rotvec_deg = np.degrees(
        Rotation.from_matrix(absolute_rotation_error).as_rotvec()
    )
    absolute_rotation_error_deg = np.linalg.norm(absolute_rotation_rotvec_deg, axis=1)

    alignment_r, alignment_t = rigid_align(estimate.positions, gt.positions)
    aligned_positions = (alignment_r @ estimate.positions.T).T + alignment_t
    aligned_position_error = np.linalg.norm(aligned_positions - gt.positions, axis=1)

    gt_path_length = float(np.sum(np.linalg.norm(np.diff(gt.positions, axis=0), axis=1)))
    est_path_length = float(np.sum(np.linalg.norm(np.diff(estimate.positions, axis=0), axis=1)))
    closure_error = float(np.linalg.norm(estimate.positions[-1] - estimate.positions[0]))
    duration_s = float((gt.timestamps_ns[-1] - gt.timestamps_ns[0]) * 1e-9)

    summary = {
        "method": method,
        "matched_frames": len(gt.timestamps_ns),
        "duration_s": duration_s,
        **scalar_stats(position_error, "ate_translation_m"),
        **vector_axis_stats(position_error_vector, "position_error_m"),
        **scalar_stats(absolute_rotation_error_deg, "absolute_rotation_error_deg"),
        **vector_axis_stats(absolute_rotation_rotvec_deg, "rotation_error_rotvec_deg"),
        "se3_aligned_translation_ate_rmse_m": float(
            np.sqrt(np.mean(np.square(aligned_position_error)))
        ),
        "gt_path_length_m": gt_path_length,
        "estimate_path_length_m": est_path_length,
        "path_length_ratio": est_path_length / gt_path_length if gt_path_length > 0.0 else math.nan,
        "closure_error_m": closure_error,
        "closure_error_pct_gt_path": (
            closure_error / gt_path_length * 100.0 if gt_path_length > 0.0 else math.nan
        ),
    }

    rpe_rows: list[dict] = []
    for lag in RPE_LAGS:
        translation_error, rotation_error_deg, delta_time_s = relative_errors(gt, estimate, lag)
        row = {
            "method": method,
            "lag_frames": lag,
            "mean_delta_time_s": float(np.mean(delta_time_s)),
            "pairs": len(translation_error),
            **scalar_stats(translation_error, "translation_rpe_m"),
            **scalar_stats(rotation_error_deg, "rotation_rpe_deg"),
            "translation_rpe_rate_rmse_m_s": float(
                np.sqrt(np.mean(np.square(translation_error / delta_time_s)))
            ),
            "rotation_rpe_rate_rmse_deg_s": float(
                np.sqrt(np.mean(np.square(rotation_error_deg / delta_time_s)))
            ),
        }
        rpe_rows.append(row)

    framewise = {
        "timestamps_ns": gt.timestamps_ns,
        "time_s": (gt.timestamps_ns - gt.timestamps_ns[0]) * 1e-9,
        "position_error_vector": position_error_vector,
        "position_error": position_error,
        "rotation_error_rotvec_deg": absolute_rotation_rotvec_deg,
        "rotation_error_deg": absolute_rotation_error_deg,
    }
    return summary, rpe_rows, framewise


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_framewise_csv(path: Path, framewise_by_method: dict[str, dict]) -> None:
    methods = list(framewise_by_method)
    first = framewise_by_method[methods[0]]
    fields = ["timestamp_ns", "time_s"]
    for method in methods:
        fields.extend(
            [
                f"{method}_position_error_x_m",
                f"{method}_position_error_y_m",
                f"{method}_position_error_z_m",
                f"{method}_position_error_norm_m",
                f"{method}_rotation_error_x_deg",
                f"{method}_rotation_error_y_deg",
                f"{method}_rotation_error_z_deg",
                f"{method}_rotation_error_norm_deg",
            ]
        )
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for index in range(len(first["timestamps_ns"])):
            row = {
                "timestamp_ns": int(first["timestamps_ns"][index]),
                "time_s": float(first["time_s"][index]),
            }
            for method in methods:
                data = framewise_by_method[method]
                p = data["position_error_vector"][index]
                r = data["rotation_error_rotvec_deg"][index]
                row.update(
                    {
                        f"{method}_position_error_x_m": float(p[0]),
                        f"{method}_position_error_y_m": float(p[1]),
                        f"{method}_position_error_z_m": float(p[2]),
                        f"{method}_position_error_norm_m": float(data["position_error"][index]),
                        f"{method}_rotation_error_x_deg": float(r[0]),
                        f"{method}_rotation_error_y_deg": float(r[1]),
                        f"{method}_rotation_error_z_deg": float(r[2]),
                        f"{method}_rotation_error_norm_deg": float(data["rotation_error_deg"][index]),
                    }
                )
            writer.writerow(row)


def write_markdown(path: Path, summaries: list[dict], rpe_rows: list[dict]) -> None:
    lines = [
        "# 矩形 Bias-only 全序列轨迹统计",
        "",
        "- 1890 帧按纳秒时间戳精确匹配，坐标统一为 NWU。",
        "- 主 ATE 为不做对齐的绝对位置误差 RMSE。",
        "- `SE(3)-aligned ATE` 仅用于区分整体刚体偏移与轨迹形状误差，不作为主结果。",
        "- RPE 使用 `T_i^{-1}T_j` 的局部平移差和相对旋转角误差，与项目 MACVO 相邻帧指标一致。",
        "",
        "## 绝对误差与路径统计",
        "",
        "| 方法 | ATE RMSE / m | ATE median / m | ATE p95 / m | ATE max / m | 终点误差 / m | 旋转 RMSE / deg | SE(3)-aligned ATE / m | 路径长度 / m | 长度比 | 闭环误差 / m |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            "| {method} | {ate_translation_m_rmse:.6f} | {ate_translation_m_median:.6f} | "
            "{ate_translation_m_p95:.6f} | {ate_translation_m_max:.6f} | {ate_translation_m_final:.6f} | "
            "{absolute_rotation_error_deg_rmse:.6f} | {se3_aligned_translation_ate_rmse_m:.6f} | "
            "{estimate_path_length_m:.6f} | {path_length_ratio:.6f} | {closure_error_m:.6f} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "## 逐轴绝对误差",
            "",
            "| 方法 | x RMSE / m | y RMSE / m | z RMSE / m | x final / m | y final / m | z final / m |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summaries:
        lines.append(
            "| {method} | {position_error_m_x_rmse:.6f} | {position_error_m_y_rmse:.6f} | "
            "{position_error_m_z_rmse:.6f} | {position_error_m_x_final:.6f} | "
            "{position_error_m_y_final:.6f} | {position_error_m_z_final:.6f} |".format(**row)
        )
    lines.extend(
        [
            "",
            "## RPE",
            "",
            "| 方法 | 间隔 | 平均时间 / s | 平移 RPE RMSE / m | 平移 p95 / m | 旋转 RPE RMSE / deg | 旋转 p95 / deg |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rpe_rows:
        lines.append(
            "| {method} | {lag_frames} 帧 | {mean_delta_time_s:.6f} | "
            "{translation_rpe_m_rmse:.6f} | {translation_rpe_m_p95:.6f} | "
            "{rotation_rpe_deg_rmse:.6f} | {rotation_rpe_deg_p95:.6f} |".format(**row)
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    gt = read_pose_series(GT_PATH)
    summaries: list[dict] = []
    rpe_rows: list[dict] = []
    framewise_by_method: dict[str, dict] = {}
    for method, trajectory_path in TRAJECTORIES.items():
        summary, method_rpe, framewise = evaluate_trajectory(
            method, gt, read_pose_series(trajectory_path)
        )
        summary["trajectory_path"] = str(trajectory_path)
        summaries.append(summary)
        rpe_rows.extend(method_rpe)
        framewise_by_method[method] = framewise

    write_csv(OUTDIR / "trajectory_metrics.csv", summaries)
    write_csv(OUTDIR / "rpe_metrics.csv", rpe_rows)
    write_framewise_csv(OUTDIR / "framewise_trajectory_errors.csv", framewise_by_method)
    write_markdown(OUTDIR / "trajectory_metrics_summary_cn.md", summaries, rpe_rows)
    print(OUTDIR / "trajectory_metrics.csv")
    print(OUTDIR / "rpe_metrics.csv")
    print(OUTDIR / "framewise_trajectory_errors.csv")
    print(OUTDIR / "trajectory_metrics_summary_cn.md")


if __name__ == "__main__":
    main()
