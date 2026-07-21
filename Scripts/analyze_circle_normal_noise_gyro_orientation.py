#!/usr/bin/env python3
"""Diagnose gyro bias and relative orientation on the full circle normal-noise run.

This is an offline reader only. It does not run or modify the optimizer.
All vectors and rotations are reported in the optimizer's internal NED frame.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation


WORKDIR = Path("/home/admin1/macvo-dev")
SCENE = "clear_circle_truth_normal_noise"
RESULT = (
    WORKDIR
    / "Results/circle_straight_normal_noise_two_state_standard_full_20260715"
    / "trial_1/vio_two_state_fixed_lag_standard_full"
    / SCENE
)
DATASET = (
    Path("/mnt/e")
    / "\u6587\u6863"
    / "holoocean/code/recordings/batch_clear_truth_paths_20260713_static63_variants"
    / SCENE
)
OUTDIR = WORKDIR / "analysis_circle_normal_noise_gyro_orientation_20260716"
AXES = ("x", "y", "z")
D_FLU_TO_NED = np.diag([1.0, -1.0, -1.0])
FIRST_ACTIVE_FRAME = 90


def vec(frame: pd.DataFrame, prefix: str) -> np.ndarray:
    return frame[[f"{prefix}_{axis}" for axis in AXES]].to_numpy(np.float64)


def flu_to_ned(values: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=np.float64) @ D_FLU_TO_NED.T


def interpolate(time_src: np.ndarray, values: np.ndarray, time_dst: np.ndarray) -> np.ndarray:
    return np.column_stack(
        [np.interp(time_dst.astype(np.float64), time_src.astype(np.float64), values[:, axis])
         for axis in range(values.shape[1])]
    )


def hold_previous(time_src: np.ndarray, values: np.ndarray, time_dst: np.ndarray) -> np.ndarray:
    index = np.searchsorted(time_src, time_dst, side="right") - 1
    index = np.clip(index, 0, time_src.size - 1)
    return values[index]


def rotations_from_pose(values: np.ndarray) -> np.ndarray:
    return Rotation.from_quat(np.asarray(values)[:, 3:7]).as_matrix()


def relative_rotations(world_from_body: np.ndarray, frame_i: np.ndarray) -> np.ndarray:
    return np.einsum(
        "nij,njk->nik",
        world_from_body[frame_i].transpose(0, 2, 1),
        world_from_body[frame_i + 1],
    )


def rotation_vectors(rotation: np.ndarray) -> np.ndarray:
    return Rotation.from_matrix(rotation).as_rotvec()


def rotation_error(estimate: np.ndarray, truth: np.ndarray) -> np.ndarray:
    error = np.einsum("nij,njk->nik", estimate.transpose(0, 2, 1), truth)
    return rotation_vectors(error)


def cumulative_rotations(deltas: np.ndarray) -> np.ndarray:
    output = np.empty((deltas.shape[0] + 1, 3, 3), dtype=np.float64)
    output[0] = np.eye(3)
    for index, delta in enumerate(deltas):
        output[index + 1] = output[index] @ delta
    return output


def relative_yaw(rotation: np.ndarray) -> np.ndarray:
    return np.unwrap(np.arctan2(rotation[:, 1, 0], rotation[:, 0, 0]))


def rmse_axis(error: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean(np.square(error), axis=0))


def rmse_norm(error: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum(np.square(error), axis=1))))


def sustained_interval(mask: np.ndarray, time_s: np.ndarray, minimum_count: int) -> tuple[float, float]:
    padded = np.r_[False, mask, False].astype(np.int8)
    change = np.diff(padded)
    starts = np.flatnonzero(change == 1)
    ends = np.flatnonzero(change == -1) - 1
    valid = [(start, end) for start, end in zip(starts, ends) if end - start + 1 >= minimum_count]
    if not valid:
        return math.nan, math.nan
    start, end = max(valid, key=lambda item: item[1] - item[0])
    return float(time_s[start]), float(time_s[end])


def event_contract(
    frame_time: np.ndarray,
    ref_pose: pd.DataFrame,
    imu_time: np.ndarray,
    gyro_truth: np.ndarray,
) -> dict[str, float]:
    frame_s = frame_time.astype(np.float64) * 1e-9
    ref_s = ref_pose["timestamp"].to_numpy(np.float64) * 1e-9
    speed = np.linalg.norm(ref_pose[["vx", "vy", "vz"]].to_numpy(np.float64), axis=1)
    first_motion_start, first_motion_end = sustained_interval(speed > 1.0e-3, ref_s, 5)
    effective_motion_start, effective_motion_end = sustained_interval(speed > 2.0e-2, ref_s, 5)
    turn_start, turn_end = sustained_interval(
        np.abs(gyro_truth[:, 2]) > 2.0e-2,
        imu_time.astype(np.float64) * 1e-9,
        10,
    )
    return {
        "static_initialization_end_s": float(frame_s[FIRST_ACTIVE_FRAME]),
        "first_effective_visual_frame_s": float(frame_s[FIRST_ACTIVE_FRAME]),
        "first_motion_s": first_motion_start,
        "effective_motion_s": effective_motion_start,
        "turn_start_s": turn_start,
        "turn_end_s": turn_end,
        "effective_motion_end_s": effective_motion_end,
        "stopped_s": first_motion_end,
        "first_motion_speed_threshold_m_s": 1.0e-3,
        "effective_motion_speed_threshold_m_s": 2.0e-2,
        "turn_rate_threshold_rad_s": 2.0e-2,
    }


def phase_at(time_s: float, events: dict[str, float]) -> str:
    if time_s < events["static_initialization_end_s"]:
        return "static_initialization"
    if time_s < events["first_motion_s"]:
        return "post_init_near_static"
    if time_s < events["turn_start_s"]:
        return "motion_ramp_up"
    if time_s <= events["turn_end_s"]:
        return "turning"
    if time_s <= events["stopped_s"]:
        return "motion_ramp_down"
    return "stopped"


def add_event_marks(axis: plt.Axes, events: dict[str, float]) -> None:
    axis.axvspan(0.0, events["static_initialization_end_s"], color="#d9dee5", alpha=0.42)
    axis.axvspan(events["turn_start_s"], events["turn_end_s"], color="#f6c453", alpha=0.10)
    marks = (
        ("static_initialization_end_s", "3 s init", "#475569"),
        ("first_motion_s", "motion", "#0f766e"),
        ("turn_start_s", "turn", "#b45309"),
        ("turn_end_s", "turn end", "#b45309"),
        ("stopped_s", "stop", "#7c3aed"),
    )
    for key, label, color in marks:
        value = events[key]
        if math.isfinite(value):
            axis.axvline(value, color=color, linewidth=0.9, linestyle="--", alpha=0.72)
            if axis is axis.figure.axes[0]:
                axis.text(value, 1.01, label, color=color, fontsize=8, rotation=90,
                          va="bottom", ha="right", transform=axis.get_xaxis_transform())


def save_axes_plot(
    path: Path,
    title: str,
    time_s: np.ndarray,
    series: list[tuple[str, np.ndarray, str, str, float]],
    ylabel: str,
    events: dict[str, float],
) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(15, 9), sharex=True, constrained_layout=True)
    for axis_index, axis in enumerate(axes):
        add_event_marks(axis, events)
        for label, values, color, style, alpha in series:
            axis.plot(time_s, values[:, axis_index], label=label, color=color,
                      linestyle=style, linewidth=1.15, alpha=alpha)
        axis.set_ylabel(f"{AXES[axis_index]} / {ylabel}")
        axis.grid(True, color="#d9e0e7", linewidth=0.65)
        if axis_index == 0:
            axis.legend(loc="upper right", ncol=min(4, len(series)), fontsize=8)
    axes[-1].set_xlabel("time / s")
    fig.suptitle(title, fontsize=15)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_rows(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def append_xyz(row: dict[str, object], prefix: str, value: np.ndarray) -> None:
    for axis, item in zip(AXES, value):
        row[f"{prefix}_{axis}"] = float(item)


def write_imu_samples(
    imu_time: np.ndarray,
    events: dict[str, float],
    signals: dict[str, np.ndarray],
) -> None:
    prefixes = [
        "gyro_meas", "gyro_truth", "bg_truth", "bg_static_applied",
        "bg_static_unique_mean", "bg_est_before", "bg_est_after",
        "gyro_corrected_before", "gyro_corrected_after",
        "gyro_corrected_error_before", "gyro_corrected_error_after",
    ]
    fields = ["sample", "timestamp_ns", "time_s", "phase"]
    fields += [f"{prefix}_{axis}" for prefix in prefixes for axis in AXES]
    rows: list[dict[str, object]] = []
    for index, timestamp in enumerate(imu_time):
        time_s = float(timestamp * 1e-9)
        row: dict[str, object] = {
            "sample": index,
            "timestamp_ns": int(timestamp),
            "time_s": time_s,
            "phase": phase_at(time_s, events),
        }
        for prefix in prefixes:
            append_xyz(row, prefix, signals[prefix][index])
        rows.append(row)
    write_rows(OUTDIR / "circle_gyro_diagnostics_imu_samples.csv", fields, rows)


def write_frame_bias(
    frame_time: np.ndarray,
    events: dict[str, float],
    static_applied: np.ndarray,
    before: np.ndarray,
    after: np.ndarray,
    truth: np.ndarray,
) -> None:
    prefixes = ["bg_static_applied", "bg_est_before", "bg_est_after", "bg_truth",
                "bg_error_before", "bg_error_after", "bg_update"]
    fields = ["frame", "timestamp_ns", "time_s", "phase"]
    fields += [f"{prefix}_{axis}" for prefix in prefixes for axis in AXES]
    fields += ["bg_update_norm", "bg_truth_step_norm"]
    rows: list[dict[str, object]] = []
    for index, timestamp in enumerate(frame_time):
        time_s = float(timestamp * 1e-9)
        values = {
            "bg_static_applied": static_applied,
            "bg_est_before": before[index],
            "bg_est_after": after[index],
            "bg_truth": truth[index],
            "bg_error_before": before[index] - truth[index],
            "bg_error_after": after[index] - truth[index],
            "bg_update": after[index] - before[index],
        }
        row: dict[str, object] = {
            "frame": index,
            "timestamp_ns": int(timestamp),
            "time_s": time_s,
            "phase": phase_at(time_s, events),
            "bg_update_norm": float(np.linalg.norm(after[index] - before[index])),
            "bg_truth_step_norm": float(np.linalg.norm(truth[index] - truth[max(index - 1, 0)])),
        }
        for prefix, value in values.items():
            append_xyz(row, prefix, value)
        rows.append(row)
    write_rows(OUTDIR / "circle_gyro_bias_framewise.csv", fields, rows)


def write_edges(
    frame_time: np.ndarray,
    events: dict[str, float],
    frame_i: np.ndarray,
    imu_raw_vec: np.ndarray,
    imu_vec: np.ndarray,
    macvo_vec: np.ndarray,
    optimized_vec: np.ndarray,
    truth_vec: np.ndarray,
    delta_errors: dict[str, np.ndarray],
    yaw: dict[str, np.ndarray],
) -> None:
    prefixes = [
        "imu_delta_rot_raw", "imu_delta_rot", "macvo_delta_rot",
        "optimized_delta_rot", "gt_delta_rot", "imu_delta_error",
        "macvo_delta_error", "optimized_delta_error",
    ]
    fields = ["edge_id", "frame_i", "frame_j", "timestamp_i_ns", "timestamp_j_ns", "time_s", "phase"]
    fields += [f"{prefix}_{axis}" for prefix in prefixes for axis in AXES]
    fields += ["imu_delta_error_norm", "macvo_delta_error_norm", "optimized_delta_error_norm",
               "imu_relative_yaw", "macvo_relative_yaw", "optimized_relative_yaw", "gt_relative_yaw"]
    errors = {
        "imu_delta_error": delta_errors["imu"],
        "macvo_delta_error": delta_errors["macvo"],
        "optimized_delta_error": delta_errors["optimized"],
    }
    rows: list[dict[str, object]] = []
    for edge_id, source in enumerate(frame_i):
        target = source + 1
        time_s = float(frame_time[target] * 1e-9)
        values = {
            "imu_delta_rot_raw": imu_raw_vec[edge_id],
            "imu_delta_rot": imu_vec[edge_id],
            "macvo_delta_rot": macvo_vec[edge_id],
            "optimized_delta_rot": optimized_vec[edge_id],
            "gt_delta_rot": truth_vec[edge_id],
            **{key: value[edge_id] for key, value in errors.items()},
        }
        row: dict[str, object] = {
            "edge_id": edge_id,
            "frame_i": int(source),
            "frame_j": int(target),
            "timestamp_i_ns": int(frame_time[source]),
            "timestamp_j_ns": int(frame_time[target]),
            "time_s": time_s,
            "phase": phase_at(time_s, events),
            "imu_delta_error_norm": float(np.linalg.norm(errors["imu_delta_error"][edge_id])),
            "macvo_delta_error_norm": float(np.linalg.norm(errors["macvo_delta_error"][edge_id])),
            "optimized_delta_error_norm": float(np.linalg.norm(errors["optimized_delta_error"][edge_id])),
            "imu_relative_yaw": float(yaw["imu"][edge_id + 1]),
            "macvo_relative_yaw": float(yaw["macvo"][edge_id + 1]),
            "optimized_relative_yaw": float(yaw["optimized"][edge_id + 1]),
            "gt_relative_yaw": float(yaw["truth"][edge_id + 1]),
        }
        for prefix, value in values.items():
            append_xyz(row, prefix, value)
        rows.append(row)
    write_rows(OUTDIR / "circle_rotation_diagnostics_per_edge.csv", fields, rows)


def build_summary(
    events: dict[str, float],
    static_applied: np.ndarray,
    static_unique_mean: np.ndarray,
    frame_truth: np.ndarray,
    before: np.ndarray,
    after: np.ndarray,
    imu_time: np.ndarray,
    gyro_truth: np.ndarray,
    gyro_measured: np.ndarray,
    corrected_after: np.ndarray,
    delta_errors: dict[str, np.ndarray],
    yaw: dict[str, np.ndarray],
) -> dict[str, object]:
    active = slice(FIRST_ACTIVE_FRAME, -1)
    after_error = after[active] - frame_truth[active]
    before_error = before[active] - frame_truth[active]
    after_step = np.diff(after[FIRST_ACTIVE_FRAME:-1], axis=0)
    truth_step = np.diff(frame_truth[FIRST_ACTIVE_FRAME:-1], axis=0)
    post = imu_time >= int(round(events["static_initialization_end_s"] * 1e9))
    raw_error = gyro_measured[post] - gyro_truth[post]
    corrected_error = corrected_after[post] - gyro_truth[post]
    integrated_error = np.trapz(corrected_error, imu_time[post].astype(np.float64) * 1e-9, axis=0)
    yaw_error = {name: values - yaw["truth"] for name, values in yaw.items() if name != "truth"}

    output: dict[str, object] = {
        "scene": SCENE,
        "coordinate_contract": {
            "vectors": "optimizer internal NED",
            "pose": "T_WB body-to-world after fixed camera-to-IMU extrinsic",
            "relative_rotation": "R_i^T R_j, mapping body j into body i",
            "rotation_vector": "SO(3) Log in the source body frame",
            "yaw": "unwrap(atan2(R[1,0], R[0,0])) of accumulated relative rotation",
        },
        "events": events,
        "static_bias": {
            "applied_from_saved_state_rad_s": static_applied.tolist(),
            "unique_raw_imu_mean_rad_s": static_unique_mean.tolist(),
            "truth_at_initialization_rad_s": frame_truth[FIRST_ACTIVE_FRAME].tolist(),
            "applied_error_rad_s": (static_applied - frame_truth[FIRST_ACTIVE_FRAME]).tolist(),
            "first_online_after_rad_s": after[FIRST_ACTIVE_FRAME].tolist(),
            "first_online_update_rad_s": (after[FIRST_ACTIVE_FRAME] - before[FIRST_ACTIVE_FRAME]).tolist(),
            "production_static_sample_count": 361,
            "note": "Production initialization used the deduplicated union of raw IMU knots and interpolated camera-boundary knots; it is not the simple unique-row CSV mean.",
        },
        "online_bias": {
            "before_error_axis_rmse_rad_s": rmse_axis(before_error).tolist(),
            "after_error_axis_rmse_rad_s": rmse_axis(after_error).tolist(),
            "after_error_vector_rmse_rad_s": rmse_norm(after_error),
            "after_error_axis_mean_rad_s": np.mean(after_error, axis=0).tolist(),
            "estimate_step_axis_rms_rad_s": np.sqrt(np.mean(np.square(after_step), axis=0)).tolist(),
            "truth_step_axis_rms_rad_s": np.sqrt(np.mean(np.square(truth_step), axis=0)).tolist(),
            "estimate_over_truth_step_rms_ratio": (
                np.sqrt(np.mean(np.square(after_step), axis=0))
                / np.maximum(np.sqrt(np.mean(np.square(truth_step), axis=0)), 1e-15)
            ).tolist(),
        },
        "corrected_gyro": {
            "raw_error_axis_rmse_rad_s": rmse_axis(raw_error).tolist(),
            "raw_error_vector_rmse_rad_s": rmse_norm(raw_error),
            "corrected_error_axis_rmse_rad_s": rmse_axis(corrected_error).tolist(),
            "corrected_error_vector_rmse_rad_s": rmse_norm(corrected_error),
            "corrected_error_axis_mean_rad_s": np.mean(corrected_error, axis=0).tolist(),
            "corrected_error_integral_rad": integrated_error.tolist(),
        },
        "rotation_increment": {},
        "cumulative_yaw": {},
    }
    for name, error in delta_errors.items():
        output["rotation_increment"][name] = {
            "error_axis_rmse_rad": rmse_axis(error).tolist(),
            "error_vector_rmse_rad": rmse_norm(error),
            "error_axis_mean_rad": np.mean(error, axis=0).tolist(),
        }
    for name, values in yaw.items():
        record = {"final_rad": float(values[-1])}
        if name != "truth":
            record.update({
                "final_error_rad": float(yaw_error[name][-1]),
                "error_rmse_rad": float(np.sqrt(np.mean(np.square(yaw_error[name])))),
            })
        output["cumulative_yaw"][name] = record
    output["evidence"] = {
        "integrated_corrected_z_rate_error_rad": float(integrated_error[2]),
        "optimized_final_yaw_error_rad": float(yaw_error["optimized"][-1]),
        "imu_final_yaw_error_rad": float(yaw_error["imu"][-1]),
        "macvo_final_yaw_error_rad": float(yaw_error["macvo"][-1]),
    }
    return output


def write_report(summary: dict[str, object]) -> None:
    static = summary["static_bias"]
    bias = summary["online_bias"]
    gyro = summary["corrected_gyro"]
    yaw = summary["cumulative_yaw"]
    evidence = summary["evidence"]
    ratio = bias["estimate_over_truth_step_rms_ratio"]
    lines = [
        "# 圆形 Normal-noise：陀螺 Bias 与朝向诊断",
        "",
        "## 结论",
        "",
        "当前圆形轨迹的朝向偏差不是单一的初始 yaw 坐标系问题。在线 gyro bias 和 MACVO 视觉相对旋转都存在同向的累计少转，融合结果位于二者之间，并更接近 IMU。",
        "",
        f"- 在线 `b_g,z - b_g,z^GT` 的均值为 `{bias['after_error_axis_mean_rad_s'][2]:.6e} rad/s`。由于校正为 `omega_m - b_g`，它使校正角速度 z 轴平均偏差变为 `{gyro['corrected_error_axis_mean_rad_s'][2]:.6e} rad/s`。",
        f"- 该 z 轴角速度误差积分为 `{evidence['integrated_corrected_z_rate_error_rad']:.6f} rad`（`{math.degrees(evidence['integrated_corrected_z_rate_error_rad']):.3f} deg`）。",
        f"- 最终累计 yaw 误差：IMU `{yaw['imu']['final_error_rad']:.6f} rad`，MACVO `{yaw['macvo']['final_error_rad']:.6f} rad`，VIO `{yaw['optimized']['final_error_rad']:.6f} rad`。VIO 与 gyro 误差积分的量级高度一致。",
        f"- 在线 bias 帧间变化 RMS 相对 GT random walk 分别为 x `{ratio[0]:.1f}x`、y `{ratio[1]:.1f}x`、z `{ratio[2]:.1f}x`，说明 bias 明显追随了短期残差/噪声。",
        f"- 第一条有效边把静止 bias 更新了 `{static['first_online_update_rad_s']}` rad/s；其中 y/z 突变远大于真实 bias random walk。",
        "",
        "因此，`b_g,z` 在线偏移是圆形朝向少转的重要直接原因，但不是唯一原因：纯 MACVO 累计 yaw 本身也少转。当前证据支持先处理 bias 可观性/更新活跃度，再复核视觉相对旋转的系统偏差；不能只旋转整条轨迹来掩盖。",
        "",
        "## 数据解释",
        "",
        "- 所有三轴量均已从 HoloOcean FLU 转为优化器内部 NED。",
        "- `bg_est_before` 是该状态成为当前边起点之前保存的 bias；`bg_est_after` 是本次两状态优化后写回的同一状态 bias。",
        "- 高频 corrected gyro 使用相机帧 bias 的零阶保持。预积分旋转增量则使用保存的 bias Jacobian 对该边起点的优化后 bias 做一阶修正。",
        "- MACVO、IMU、VIO、GT 的每帧旋转均统一为 body `R_i^T R_j` 后比较；累计 yaw 使用 SO(3) 连乘，不是逐轴 Euler 角速度相加。",
        "- 静止初始化生产路径使用 361 个去重后的 raw/interpolated knot，故保存的静止 bias 与 300 个唯一 CSV 行直接均值不完全相同。",
        "",
        "## 产物",
        "",
        "- `circle_gyro_diagnostics_imu_samples.csv`：逐 IMU 样本 raw/truth/bias/corrected。",
        "- `circle_gyro_bias_framewise.csv`：逐相机帧 bias before/after/truth。",
        "- `circle_rotation_diagnostics_per_edge.csv`：逐边 IMU/MACVO/VIO/GT 旋转增量和累计 yaw。",
        "- `circle_gyro_orientation_summary.json`：完整数值汇总和坐标契约。",
    ]
    (OUTDIR / "circle_gyro_orientation_report_cn.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_html(summary: dict[str, object]) -> None:
    evidence = summary["evidence"]
    bias = summary["online_bias"]
    yaw = summary["cumulative_yaw"]
    images = [
        ("原始三轴角速度", "01_raw_gyro_axes.png"),
        ("三轴 gyro bias：静止值、GT、优化前和优化后", "02_gyro_bias_axes.png"),
        ("去 bias 后角速度与 GT", "03_corrected_gyro_axes.png"),
        ("每帧旋转增量：IMU / MACVO / VIO / GT", "04_rotation_increment_axes.png"),
        ("累计相对 yaw", "05_cumulative_relative_yaw.png"),
        ("z-bias 误差积分与 yaw 误差", "06_z_bias_integral_vs_yaw_error.png"),
    ]
    tabs = "".join(
        f"<button class='tab{' active' if index == 0 else ''}' data-target='p{index}'>{title}</button>"
        for index, (title, _) in enumerate(images)
    )
    panels = "".join(
        f"<section id='p{index}' class='panel{' active' if index == 0 else ''}'><h2>{title}</h2><a href='{name}'><img src='{name}' alt='{title}'></a></section>"
        for index, (title, name) in enumerate(images)
    )
    page = f"""<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>
<title>圆形 Normal-noise：gyro 与朝向诊断</title><style>
:root{{--ink:#26323f;--muted:#667483;--line:#d6dde5;--blue:#1769aa;--bg:#eef1f4}}*{{box-sizing:border-box}}body{{margin:0;padding:16px;background:var(--bg);font-family:Arial,'Microsoft YaHei',sans-serif;color:var(--ink)}}main{{max-width:1500px;margin:auto;background:#fff;border:1px solid var(--line)}}header{{padding:18px;border-bottom:1px solid var(--line)}}h1{{font-size:23px;margin:0 0 8px}}p{{color:var(--muted);line-height:1.55}}.cards{{display:grid;grid-template-columns:repeat(4,minmax(210px,1fr));gap:8px}}.card{{border:1px solid var(--line);padding:11px}}.card b{{display:block;font-size:19px;margin-top:5px}}nav{{display:flex;gap:6px;flex-wrap:wrap;padding:10px;border-bottom:1px solid var(--line)}}button{{padding:8px 11px;border:1px solid #aeb9c5;background:#f8fafc;cursor:pointer}}button.active{{background:var(--blue);color:#fff;border-color:var(--blue)}}.panel{{display:none;padding:12px 16px}}.panel.active{{display:block}}.panel h2{{font-size:17px}}img{{display:block;width:100%;height:auto;border:1px solid var(--line)}}.files{{padding:14px 18px;border-top:1px solid var(--line)}}.files a{{margin-right:16px}}@media(max-width:900px){{.cards{{grid-template-columns:1fr 1fr}}body{{padding:5px}}}}
</style></head><body><main><header><h1>圆形 Normal-noise：陀螺 Bias 与累计朝向诊断</h1>
<p>只读取既有 full-run 产物；未修改优化器、未重新运行序列。全部量统一到内部 NED，旋转增量统一为 body <code>R_i^T R_j</code>。</p>
<div class='cards'><article class='card'>校正 z 角速度误差积分<b>{evidence['integrated_corrected_z_rate_error_rad']:.4f} rad</b></article>
<article class='card'>IMU 最终 yaw 误差<b>{yaw['imu']['final_error_rad']:.4f} rad</b></article><article class='card'>MACVO 最终 yaw 误差<b>{yaw['macvo']['final_error_rad']:.4f} rad</b></article><article class='card'>VIO 最终 yaw 误差<b>{yaw['optimized']['final_error_rad']:.4f} rad</b></article></div>
<p>在线 bias 步长相对 GT random walk：x {bias['estimate_over_truth_step_rms_ratio'][0]:.1f}x，y {bias['estimate_over_truth_step_rms_ratio'][1]:.1f}x，z {bias['estimate_over_truth_step_rms_ratio'][2]:.1f}x。曲线中的阴影/竖线标出 3 秒初始化、运动、转弯和停止。</p></header><nav>{tabs}</nav>{panels}
<section class='files'><a href='circle_gyro_orientation_report_cn.md'>中文报告</a><a href='circle_gyro_orientation_summary.json'>JSON 汇总</a><a href='circle_gyro_diagnostics_imu_samples.csv'>逐 IMU CSV</a><a href='circle_gyro_bias_framewise.csv'>逐帧 Bias CSV</a><a href='circle_rotation_diagnostics_per_edge.csv'>逐边旋转 CSV</a></section>
</main><script>document.querySelectorAll('.tab').forEach(b=>b.onclick=()=>{{document.querySelectorAll('.tab,.panel').forEach(x=>x.classList.remove('active'));b.classList.add('active');document.getElementById(b.dataset.target).classList.add('active')}});</script></body></html>"""
    (OUTDIR / "interactive_circle_gyro_orientation_diagnostics.html").write_text(page, encoding="utf-8")


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    tensor = np.load(RESULT / "tensor_map.npz", allow_pickle=False)
    frame_time = tensor["frames//time_ns"].astype(np.int64)
    decomposition = pd.read_csv(DATASET / "imu_truth_decomposition.csv")
    ref_pose = pd.read_csv(DATASET / "ref_pose.csv")
    imu_time = decomposition["timestamp"].to_numpy(np.int64)
    ref_time = ref_pose["timestamp"].to_numpy(np.int64)
    if not np.array_equal(frame_time, ref_time):
        raise AssertionError("tensor-map and ref_pose frame timestamps differ")

    gyro_measured = flu_to_ned(vec(decomposition, "measured_ang_vel"))
    gyro_truth = flu_to_ned(vec(decomposition, "true_ang_vel"))
    bg_truth_imu = flu_to_ned(vec(decomposition, "gyro_bias"))
    bg_truth_frame = interpolate(imu_time, bg_truth_imu, frame_time)
    events = event_contract(frame_time, ref_pose, imu_time, gyro_truth)

    bg_before = tensor["frames//imu_vio_prev_gyro_bias"].astype(np.float64)
    bg_after = tensor["frames//imu_vio_gyro_bias"].astype(np.float64)
    static_applied = bg_before[FIRST_ACTIVE_FRAME].copy()
    static_unique_mask = imu_time <= frame_time[FIRST_ACTIVE_FRAME]
    static_unique_mean = gyro_measured[static_unique_mask].mean(axis=0)
    static_union_time = np.unique(
        np.concatenate(
            [imu_time[static_unique_mask], frame_time[: FIRST_ACTIVE_FRAME + 1]]
        )
    )
    static_union_mean = interpolate(imu_time, gyro_measured, static_union_time).mean(axis=0)
    if static_union_time.size != 361 or not np.allclose(
        static_union_mean, static_applied, atol=2.0e-9, rtol=0.0
    ):
        raise AssertionError(
            "production static-knot reconstruction does not match saved gyro bias"
        )

    bg_before_hold = hold_previous(frame_time, bg_before, imu_time)
    bg_after_hold = hold_previous(frame_time, bg_after, imu_time)
    before_valid = imu_time >= frame_time[FIRST_ACTIVE_FRAME]
    bg_before_hold[~before_valid] = static_applied
    bg_after_hold[~before_valid] = static_applied
    corrected_before = gyro_measured - bg_before_hold
    corrected_after = gyro_measured - bg_after_hold
    static_rows = np.repeat(static_applied.reshape(1, 3), imu_time.size, axis=0)
    unique_mean_rows = np.repeat(static_unique_mean.reshape(1, 3), imu_time.size, axis=0)
    signals = {
        "gyro_meas": gyro_measured,
        "gyro_truth": gyro_truth,
        "bg_truth": bg_truth_imu,
        "bg_static_applied": static_rows,
        "bg_static_unique_mean": unique_mean_rows,
        "bg_est_before": bg_before_hold,
        "bg_est_after": bg_after_hold,
        "gyro_corrected_before": corrected_before,
        "gyro_corrected_after": corrected_after,
        "gyro_corrected_error_before": corrected_before - gyro_truth,
        "gyro_corrected_error_after": corrected_after - gyro_truth,
    }

    frame_i = np.arange(FIRST_ACTIVE_FRAME, frame_time.size - 1, dtype=np.int64)
    frame_j = frame_i + 1
    diagnostics = pd.read_csv(RESULT / "frame_pair_diagnostics.csv")
    if diagnostics.shape[0] != frame_i.size or not np.array_equal(
        diagnostics["frame_i"].to_numpy(np.int64), frame_i
    ) or not np.array_equal(diagnostics["frame_j"].to_numpy(np.int64), frame_j):
        raise AssertionError("saved frame-pair diagnostics do not match active edge range")
    pose_camera_rotation = rotations_from_pose(tensor["frames//pose"])
    extrinsic_rotation = rotations_from_pose(tensor["frames//imu_vio_sensor_T_imu"])
    body_rotation = np.einsum("nij,njk->nik", pose_camera_rotation, extrinsic_rotation)

    ref_rotation_nwu = Rotation.from_quat(
        ref_pose[["qx", "qy", "qz", "qw"]].to_numpy(np.float64)
    ).as_matrix()
    ref_rotation_ned = np.einsum("ab,nbc,cd->nad", D_FLU_TO_NED, ref_rotation_nwu, D_FLU_TO_NED)
    gt_body_rotation = np.einsum("nij,njk->nik", ref_rotation_ned, extrinsic_rotation)
    optimized_delta = relative_rotations(body_rotation, frame_i)
    truth_delta = relative_rotations(gt_body_rotation, frame_i)

    macvo_camera_delta = Rotation.from_quat(
        tensor["frames//visual_relative_pose_CiCj"][frame_j, 3:7]
    ).as_matrix()
    r_ci = extrinsic_rotation[frame_j]
    macvo_delta = np.einsum(
        "nij,njk,nkl->nil", r_ci.transpose(0, 2, 1), macvo_camera_delta, r_ci
    )

    imu_raw_vec = tensor["frames//imu_vio_delta_rotvec"][frame_j].astype(np.float64)
    jacobian = tensor["frames//imu_vio_bias_jacobian"][frame_j].astype(np.float64)
    linearized_bg = tensor["frames//imu_vio_linearized_gyro_bias"][frame_j].astype(np.float64)
    correction_vec = np.einsum(
        "nij,nj->ni", jacobian[:, 6:9, 3:6], bg_after[frame_i] - linearized_bg
    )
    imu_delta = np.einsum(
        "nij,njk->nik",
        Rotation.from_rotvec(imu_raw_vec).as_matrix(),
        Rotation.from_rotvec(correction_vec).as_matrix(),
    )

    imu_vec = rotation_vectors(imu_delta)
    macvo_vec = rotation_vectors(macvo_delta)
    optimized_vec = rotation_vectors(optimized_delta)
    truth_vec = rotation_vectors(truth_delta)
    delta_errors = {
        "imu": rotation_error(imu_delta, truth_delta),
        "macvo": rotation_error(macvo_delta, truth_delta),
        "optimized": rotation_error(optimized_delta, truth_delta),
    }

    relative_optimized = np.einsum(
        "ij,njk->nik", body_rotation[FIRST_ACTIVE_FRAME].T, body_rotation[FIRST_ACTIVE_FRAME:]
    )
    relative_truth = np.einsum(
        "ij,njk->nik", gt_body_rotation[FIRST_ACTIVE_FRAME].T, gt_body_rotation[FIRST_ACTIVE_FRAME:]
    )
    yaw = {
        "imu": relative_yaw(cumulative_rotations(imu_delta)),
        "macvo": relative_yaw(cumulative_rotations(macvo_delta)),
        "optimized": relative_yaw(relative_optimized),
        "truth": relative_yaw(relative_truth),
    }

    write_imu_samples(imu_time, events, signals)
    write_frame_bias(frame_time, events, static_applied, bg_before, bg_after, bg_truth_frame)
    write_edges(frame_time, events, frame_i, imu_raw_vec, imu_vec, macvo_vec,
                optimized_vec, truth_vec, delta_errors, yaw)
    event_rows = [{"event": key, "value": value} for key, value in events.items()]
    write_rows(OUTDIR / "circle_motion_events.csv", ["event", "value"], event_rows)

    imu_s = imu_time.astype(np.float64) * 1e-9
    frame_s = frame_time.astype(np.float64) * 1e-9
    edge_s = frame_time[frame_j].astype(np.float64) * 1e-9
    save_axes_plot(
        OUTDIR / "01_raw_gyro_axes.png", "Raw gyro and clean truth (NED)", imu_s,
        [("measured", gyro_measured, "#2563eb", "-", 0.38),
         ("truth", gyro_truth, "#111827", "-", 0.95)], "rad/s", events,
    )
    static_frame = np.repeat(static_applied.reshape(1, 3), frame_time.size, axis=0)
    save_axes_plot(
        OUTDIR / "02_gyro_bias_axes.png", "Gyro bias: static, GT, before and after optimization", frame_s,
        [("GT", bg_truth_frame, "#111827", "-", 1.0),
         ("static applied", static_frame, "#7c3aed", "--", 0.95),
         ("before", bg_before, "#0f766e", ":", 0.78),
         ("after", bg_after, "#dc2626", "-", 0.90)], "rad/s", events,
    )
    corrected_rolling = pd.DataFrame(corrected_after).rolling(101, center=True, min_periods=1).mean().to_numpy()
    truth_rolling = pd.DataFrame(gyro_truth).rolling(101, center=True, min_periods=1).mean().to_numpy()
    save_axes_plot(
        OUTDIR / "03_corrected_gyro_axes.png", "Bias-corrected gyro versus truth (1 s rolling means emphasized)", imu_s,
        [("corrected samples", corrected_after, "#60a5fa", "-", 0.22),
         ("corrected 1 s mean", corrected_rolling, "#2563eb", "-", 0.95),
         ("GT 1 s mean", truth_rolling, "#111827", "--", 0.95)], "rad/s", events,
    )
    save_axes_plot(
        OUTDIR / "04_rotation_increment_axes.png", "Per-camera-frame body rotation increments", edge_s,
        [("IMU corrected", imu_vec, "#dc2626", "-", 0.72),
         ("MACVO", macvo_vec, "#f97316", "-", 0.82),
         ("VIO optimized", optimized_vec, "#2563eb", "-", 0.82),
         ("GT", truth_vec, "#111827", "--", 0.96)], "rad", events,
    )

    cumulative_time = frame_time[FIRST_ACTIVE_FRAME:].astype(np.float64) * 1e-9
    fig, axis = plt.subplots(figsize=(15, 6), constrained_layout=True)
    add_event_marks(axis, events)
    colors = {"truth": "#111827", "imu": "#dc2626", "macvo": "#f97316", "optimized": "#2563eb"}
    labels = {"truth": "GT", "imu": "IMU corrected", "macvo": "MACVO", "optimized": "VIO optimized"}
    for name in ("truth", "imu", "macvo", "optimized"):
        axis.plot(cumulative_time, yaw[name], color=colors[name], label=labels[name], linewidth=2.0)
    axis.set(title="Accumulated relative yaw from first active frame", xlabel="time / s", ylabel="relative yaw / rad")
    axis.grid(True, color="#d9e0e7"); axis.legend()
    fig.savefig(OUTDIR / "05_cumulative_relative_yaw.png", dpi=160); plt.close(fig)

    post = imu_time >= frame_time[FIRST_ACTIVE_FRAME]
    z_error = (corrected_after - gyro_truth)[:, 2]
    post_time = imu_s[post]
    dt = np.diff(post_time, prepend=post_time[0])
    z_integral = np.cumsum(z_error[post] * dt)
    fig, axis = plt.subplots(figsize=(15, 6), constrained_layout=True)
    add_event_marks(axis, events)
    axis.plot(post_time, z_integral, color="#dc2626", label="integral(corrected gyro z error)", linewidth=2)
    for name, color in (("imu", "#b91c1c"), ("macvo", "#f97316"), ("optimized", "#2563eb")):
        axis.plot(cumulative_time, yaw[name] - yaw["truth"], color=color, label=f"{name} yaw error", linewidth=1.6)
    axis.set(title="Integrated z-rate error and accumulated yaw errors", xlabel="time / s", ylabel="angle error / rad")
    axis.grid(True, color="#d9e0e7"); axis.legend()
    fig.savefig(OUTDIR / "06_z_bias_integral_vs_yaw_error.png", dpi=160); plt.close(fig)

    summary = build_summary(
        events, static_applied, static_unique_mean, bg_truth_frame, bg_before, bg_after,
        imu_time, gyro_truth, gyro_measured, corrected_after, delta_errors, yaw,
    )
    summary["static_bias"]["reconstructed_production_union_mean_rad_s"] = static_union_mean.tolist()
    summary["static_bias"]["production_static_sample_count"] = int(static_union_time.size)
    numeric_arrays = [
        gyro_measured, gyro_truth, bg_truth_imu, bg_before, bg_after,
        imu_raw_vec, imu_vec, macvo_vec, optimized_vec, truth_vec,
        *delta_errors.values(), *yaw.values(),
    ]
    if not all(np.isfinite(values).all() for values in numeric_arrays):
        raise AssertionError("diagnostic inputs or derived rotations contain NaN/Inf")
    (OUTDIR / "circle_gyro_orientation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_report(summary)
    write_html(summary)
    print(OUTDIR / "interactive_circle_gyro_orientation_diagnostics.html")


if __name__ == "__main__":
    main()
