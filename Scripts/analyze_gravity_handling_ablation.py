#!/usr/bin/env python3
"""Analyze the gravity-handling ablation without starting MACVO runs.

The report intentionally separates observed trajectory quality from causal
attribution. Independent MACVO frontend runs are nondeterministic, so the
visual-input identity audit is carried into every report and HTML page.
"""

from __future__ import annotations

import argparse
import csv
import html
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd


WORKDIR = Path("/home/admin1/macvo-dev")
if str(WORKDIR) not in sys.path:
    sys.path.insert(0, str(WORKDIR))

from Scripts import analyse_clear_circle_pair_vio as pair_analysis
from Scripts.compare_bias_linearization_fix_closed_paths import (
    qxyzw_to_rot,
    relative_metrics,
    rot_angle_deg,
)


DEFAULT_RESULT_ROOT = WORKDIR / "Results" / "gravity_handling_ablation_20260711"
DEFAULT_HISTORICAL_ROOT = WORKDIR / "Results" / "closed_paths_latest_20260706_timeinterp"
DEFAULT_OUT_ROOT = WORKDIR / "analysis_gravity_handling_ablation_20260711"

SCENES = [
    "clear_rectangle_zero_noise",
    "clear_rectangle_normal_noise",
]

METHODS = [
    {
        "method": "pure_macvo_reused",
        "source": "historical",
        "variant": "pure_macvo",
    },
    {
        "method": "imuatt_previous_reused",
        "source": "historical",
        "variant": "vio_preintegrated_full_imuatt_estinit",
    },
    {
        "method": "imuatt_current_batch",
        "source": "current",
        "variant": "vio_preintegrated_full_imuatt_estinit",
    },
    {
        "method": "residual_gravity_current",
        "source": "current",
        "variant": "vio_preintegrated_full_residual_gravity",
    },
]

DIAGNOSTIC_COLUMNS = [
    "est_velocity_error_norm",
    "rotation_error_angle",
    "r_p_whitened_norm",
    "r_v_whitened_norm",
    "r_R_whitened_norm",
    "imu_vio_whitened_norm",
    "energy_imu_to_visual_ratio",
    "energy_pv_to_visual_ratio",
    "energy_R_to_visual_ratio",
    "imu_vio_cov_trace",
    "imu_vio_weight_diag_min",
    "imu_vio_weight_diag_max",
    "imu_vio_acc_bias_norm",
    "imu_vio_gyro_bias_norm",
]


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def select_manifest_row(
    rows: list[dict[str, str]], scene: str, variant: str
) -> dict[str, str]:
    matches = [row for row in rows if row.get("scene") == scene and row.get("variant") == variant]
    if len(matches) != 1:
        raise ValueError(f"Expected one row for {scene}/{variant}, found {len(matches)}")
    return matches[0]


def absolute_rotation_errors_deg(joined: pd.DataFrame) -> np.ndarray:
    errors: list[float] = []
    for row in joined.itertuples(index=False):
        est = qxyzw_to_rot(row.qx_est, row.qy_est, row.qz_est, row.qw_est)
        gt = qxyzw_to_rot(row.qx_gt, row.qy_gt, row.qz_gt, row.qw_gt)
        errors.append(rot_angle_deg(est.T @ gt))
    return np.asarray(errors, dtype=np.float64)


def summarize_trajectory(
    scene: str,
    method: str,
    source_path: Path,
    joined: pd.DataFrame,
) -> dict[str, object]:
    err = joined["err_m"].to_numpy(np.float64)
    est_pos = joined[["tx_est", "ty_est", "tz_est"]].to_numpy(np.float64)
    gt_pos = joined[["tx_gt", "ty_gt", "tz_gt"]].to_numpy(np.float64)
    abs_rot = absolute_rotation_errors_deg(joined)
    est_path = float(np.linalg.norm(np.diff(est_pos, axis=0), axis=1).sum())
    gt_path = float(np.linalg.norm(np.diff(gt_pos, axis=0), axis=1).sum())
    closure_delta = (est_pos[-1] - est_pos[0]) - (gt_pos[-1] - gt_pos[0])
    row: dict[str, object] = {
        "scene": scene,
        "method": method,
        "matched_frames": int(len(joined)),
        "ate_rmse_m": float(np.sqrt(np.mean(err * err))),
        "ate_median_m": float(np.median(err)),
        "ate_final_m": float(err[-1]),
        "ate_max_m": float(np.max(err)),
        "absolute_rotation_rmse_deg": float(np.sqrt(np.mean(abs_rot * abs_rot))),
        "absolute_rotation_median_deg": float(np.median(abs_rot)),
        "absolute_rotation_final_deg": float(abs_rot[-1]),
        "absolute_rotation_max_deg": float(np.max(abs_rot)),
        "estimated_path_length_m": est_path,
        "gt_path_length_m": gt_path,
        "path_length_ratio": est_path / gt_path if gt_path > 0.0 else math.nan,
        "closure_error_m": float(np.linalg.norm(closure_delta)),
        "source_poses": str(source_path),
    }
    row.update(relative_metrics(joined))
    return row


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def diagnostic_summary(
    result_root: Path,
    scene: str,
    variant: str,
) -> list[dict[str, object]]:
    path = result_root / "trial_1" / variant / scene / "frame_pair_diagnostics.csv"
    frame = pd.read_csv(path)
    rows: list[dict[str, object]] = []
    for column in DIAGNOSTIC_COLUMNS:
        values = pd.to_numeric(frame[column], errors="coerce").dropna()
        if values.empty:
            continue
        rows.append(
            {
                "scene": scene,
                "variant": variant,
                "metric": column,
                "count": int(len(values)),
                "mean": float(values.mean()),
                "median": float(values.median()),
                "p90": float(values.quantile(0.9)),
                "max": float(values.max()),
            }
        )
    return rows


def first_crossing_frame(path: Path, column: str, threshold: float) -> int | None:
    frame = pd.read_csv(path, usecols=["frame_j", column])
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(np.float64)
    indices = np.flatnonzero(values > threshold)
    return int(frame.iloc[int(indices[0])]["frame_j"]) if len(indices) else None


def add_page_context(page: Path, scene: str, exact_visual_match: bool) -> None:
    text = page.read_text(encoding="utf-8")
    title = f"{scene} gravity-handling comparison"
    text = text.replace("Clear-circle VIO interactive trajectory", title)
    warning = (
        "严格视觉输入一致性检查通过，可用于隔离重力处理差异。"
        if exact_visual_match
        else "警告：两次运行的视觉输入指纹 898/898 不一致；轨迹可评价，但不能把差异严格归因于重力处理方式。"
    )
    banner = (
        '<div style="padding:10px 14px;background:#fff4ce;border-bottom:1px solid #d8b04c;'
        'font-size:13px;line-height:1.5"><strong>因果性说明：</strong>'
        + html.escape(warning)
        + "</div>"
    )
    text = text.replace('<div class="panel">', '<div class="panel">' + banner, 1)
    page.write_text(text, encoding="utf-8")


def write_scene_readme(
    scene_dir: Path,
    scene: str,
    rows: list[dict[str, object]],
    exact_visual_match: bool,
) -> None:
    lines = [
        f"# {scene}",
        "",
        "本页为严格起点一致、无 SE(3)/Sim(3) 对齐的轨迹评估。",
        "",
        f"- 视觉输入严格一致：`{exact_visual_match}`",
        "- 若为 `False`，只能比较本次运行的实际表现，不能做严格的单变量因果归因。",
        "",
        "| method | ATE RMSE m | final m | abs rot final deg | t_rel m/frame | r_rel deg/frame | t_vel m/s | r_vel deg/s |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["method"]),
                    f"{float(row['ate_rmse_m']):.6f}",
                    f"{float(row['ate_final_m']):.6f}",
                    f"{float(row['absolute_rotation_final_deg']):.6f}",
                    f"{float(row['t_rel_m_per_frame']):.6f}",
                    f"{float(row['r_rel_deg_per_frame']):.6f}",
                    f"{float(row['t_vel_m_s']):.6f}",
                    f"{float(row['r_vel_deg_s']):.6f}",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "- `interactive_trajectory_gt_vs_est.html`: 交互轨迹与误差曲线。",
            "- `trajectory_summary.csv`: 完整指标。",
            "- `trajectories/`: 时间戳对齐后的逐帧数据。",
        ]
    )
    (scene_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_index(
    out_root: Path,
    summaries: list[dict[str, object]],
    pages: dict[str, Path],
    identity: dict[str, bool],
) -> None:
    lines = [
        "<!doctype html>",
        '<html lang="zh-CN"><head><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        "<title>重力处理消融结果</title>",
        "<style>",
        "body{font-family:Arial,Helvetica,sans-serif;margin:24px;color:#1f2933;background:#f5f7fa}",
        "main{max-width:1280px;margin:auto;background:white;border:1px solid #d8dee6;padding:18px}",
        ".warning{background:#fff4ce;border:1px solid #d8b04c;padding:10px 12px;line-height:1.55}",
        "table{border-collapse:collapse;width:100%;font-size:12px;margin-top:12px}",
        "th,td{border:1px solid #d8dee6;padding:6px 7px;text-align:right}",
        "th:first-child,td:first-child,th:nth-child(2),td:nth-child(2){text-align:left}",
        "th{background:#eef2f7}a{color:#1f6feb;text-decoration:none}",
        "</style></head><body><main>",
        "<h1>重力处理消融结果</h1>",
        '<p class="warning"><strong>重要限制：</strong>当前两种重力处理运行的视觉输入指纹在两个场景中均为 898/898 帧对不一致。'
        "因此本页可用于评价每条轨迹的实际质量，也可发现严重失效；但不能把数值差异严格归因于重力处理这一项。</p>",
        "<h2>交互页面</h2><ul>",
    ]
    for scene, page in pages.items():
        rel = page.relative_to(out_root)
        status = "一致" if identity[scene] else "不一致"
        lines.append(
            f'<li><a href="{html.escape(str(rel))}">{html.escape(scene)}</a> '
            f"(视觉输入：{status})</li>"
        )
    lines.extend(
        [
            "</ul><h2>完整指标</h2>",
            "<table><thead><tr>",
            "<th>scene</th><th>method</th><th>ATE RMSE m</th><th>final m</th>",
            "<th>abs rot final deg</th><th>t_rel m/frame</th><th>r_rel deg/frame</th>",
            "<th>t_vel m/s</th><th>r_vel deg/s</th><th>path ratio</th><th>closure m</th>",
            "</tr></thead><tbody>",
        ]
    )
    for row in summaries:
        lines.append(
            "<tr>"
            f"<td>{html.escape(str(row['scene']))}</td>"
            f"<td>{html.escape(str(row['method']))}</td>"
            f"<td>{float(row['ate_rmse_m']):.6f}</td>"
            f"<td>{float(row['ate_final_m']):.6f}</td>"
            f"<td>{float(row['absolute_rotation_final_deg']):.6f}</td>"
            f"<td>{float(row['t_rel_m_per_frame']):.6f}</td>"
            f"<td>{float(row['r_rel_deg_per_frame']):.6f}</td>"
            f"<td>{float(row['t_vel_m_s']):.6f}</td>"
            f"<td>{float(row['r_vel_deg_s']):.6f}</td>"
            f"<td>{float(row['path_length_ratio']):.6f}</td>"
            f"<td>{float(row['closure_error_m']):.6f}</td>"
            "</tr>"
        )
    lines.extend(["</tbody></table></main></body></html>"])
    (out_root / "index.html").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_analysis_report(
    out_root: Path,
    summaries: list[dict[str, object]],
    result_root: Path,
) -> None:
    by_key = {(str(row["scene"]), str(row["method"])): row for row in summaries}

    def ratio(scene: str, metric: str) -> float:
        residual = float(by_key[(scene, "residual_gravity_current")][metric])
        legacy = float(by_key[(scene, "imuatt_current_batch")][metric])
        return residual / legacy if legacy != 0.0 else math.inf

    zero_diag = result_root / "trial_1" / "vio_preintegrated_full_residual_gravity" / SCENES[0] / "frame_pair_diagnostics.csv"
    normal_diag = result_root / "trial_1" / "vio_preintegrated_full_residual_gravity" / SCENES[1] / "frame_pair_diagnostics.csv"
    lines = [
        "# 重力处理消融分析",
        "",
        "## 结论摘要",
        "",
        "1. 当前 `residual_gravity` 版本在两个场景都发生严重发散，不能作为后续算法基线。",
        "2. 无噪声场景中，旧 `imuatt` 明显改善 pure MACVO；normal-noise 中则明显劣于 pure MACVO。这再次说明当前系统的主要问题是有噪 IMU 下的融合、状态估计与不确定性链路，而不是 IMU 在所有条件下都无效。",
        "3. 新版本首先出现速度状态失真，随后位置残差和绝对姿态误差累积。其低速度残差不代表速度正确，因为当前帧速度是自由变量，可被优化器直接调整来满足单条 IMU edge。",
        "4. 两次运行的视觉输入指纹 898/898 不同，因此这不是严格的单变量消融；但失效幅度远大于历史与当前旧版之间的重复运行差异，足以判定当前 residual-gravity 链路不可用。",
        "",
        "## 关键倍率",
        "",
        f"- zero-noise：residual-gravity 的 ATE RMSE 是当前旧版的 `{ratio(SCENES[0], 'ate_rmse_m'):.2f}x`，`t_vel` 是 `{ratio(SCENES[0], 't_vel_m_s'):.2f}x`。",
        f"- normal-noise：residual-gravity 的 ATE RMSE 是当前旧版的 `{ratio(SCENES[1], 'ate_rmse_m'):.2f}x`，`t_vel` 是 `{ratio(SCENES[1], 't_vel_m_s'):.2f}x`，`r_vel` 是 `{ratio(SCENES[1], 'r_vel_deg_s'):.2f}x`。",
        "",
        "## 发散时间线",
        "",
        f"- zero-noise residual-gravity：速度误差首次超过 0.1/1/5 m/s 的帧分别为 `{first_crossing_frame(zero_diag, 'est_velocity_error_norm', 0.1)}`、`{first_crossing_frame(zero_diag, 'est_velocity_error_norm', 1.0)}`、`{first_crossing_frame(zero_diag, 'est_velocity_error_norm', 5.0)}`。",
        f"- normal-noise residual-gravity：速度误差首次超过 0.1/1/5 m/s 的帧分别为 `{first_crossing_frame(normal_diag, 'est_velocity_error_norm', 0.1)}`、`{first_crossing_frame(normal_diag, 'est_velocity_error_norm', 1.0)}`、`{first_crossing_frame(normal_diag, 'est_velocity_error_norm', 5.0)}`。",
        "",
        "## 不能据此声称的内容",
        "",
        "- 不能严格声称全部差异只由 `gravity_handling` 造成；视觉观测未锁定。",
        "- 不能因 `r_v` 很小就声称速度正确；实际 GT 速度误差已经很大。",
        "- 不能据此开始调 alpha 或鲁棒核；当前首先缺少可重复的视觉输入和统计一致的惯性协方差。",
        "",
        "## 下一阶段项目顺序",
        "",
        "1. 实现 MACVO 视觉观测缓存/回放，使不同后端方法消费完全相同的 GraphInput。",
        "2. 在相同视觉输入上重做 legacy-vs-residual gravity 消融，定位是公式、速度反馈还是两帧状态结构导致发散。",
        "3. 将已验证的 sampling/interpolation-aware IMU covariance 和自适应数值正则化接入生产链路，并以 NIS≈9、白化标准差≈1、float32 SPD 为验收门槛。",
        "4. 在固定视觉输入下做 `gravity mode x covariance mode` 的 2x2 序列实验，保留 pure MACVO、历史旧版和当前旧版轨迹。",
        "5. 核心统计链路通过后，再实现静止段 Bias 初始化/标定 Bias 先验，并用非零 Bias 合成测试验证。",
        "6. Bias 能被可靠恢复后，再讨论 W3、边缘化先验、共享 MapPoint BA；alpha/鲁棒核最后作为异常保护，不作为标定替代品。",
    ]
    (out_root / "analysis_report_cn.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--historical-root", type=Path, default=DEFAULT_HISTORICAL_ROOT)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out_root.mkdir(parents=True, exist_ok=True)
    manifests = {
        "current": read_manifest(args.result_root / "run_manifest.csv"),
        "historical": read_manifest(args.historical_root / "run_manifest.csv"),
    }
    identity_rows = read_manifest(args.result_root / "visual_input_identity.csv")
    identity = {
        row["scene"]: str(row.get("exact_match", "0")).strip().lower() in {"1", "true", "yes"}
        for row in identity_rows
    }

    summaries: list[dict[str, object]] = []
    diagnostic_rows: list[dict[str, object]] = []
    pages: dict[str, Path] = {}
    for scene in SCENES:
        scene_dir = args.out_root / scene
        trajectory_dir = scene_dir / "trajectories"
        trajectory_dir.mkdir(parents=True, exist_ok=True)
        trajectories: dict[str, pd.DataFrame] = {}
        scene_rows: list[dict[str, object]] = []

        for spec in METHODS:
            method = str(spec["method"])
            source = str(spec["source"])
            variant = str(spec["variant"])
            manifest_row = select_manifest_row(manifests[source], scene, variant)
            _, joined = pair_analysis.evaluate_run(manifest_row)
            joined_path = trajectory_dir / f"{scene}_{method}_joined.csv"
            joined.to_csv(joined_path, index=False)
            trajectories[f"{scene} / {method}"] = joined
            summary = summarize_trajectory(
                scene=scene,
                method=method,
                source_path=Path(manifest_row["result_dir"]) / "poses.csv",
                joined=joined,
            )
            scene_rows.append(summary)
            summaries.append(summary)

        pair_analysis.plot_xy(trajectories, scene_dir)
        pair_analysis.plot_xy_gt_region(trajectories, scene_dir)
        pair_analysis.plot_xz(trajectories, scene_dir)
        pair_analysis.plot_error(trajectories, scene_dir)
        pair_analysis.write_interactive_html(trajectories, scene_dir)
        page = scene_dir / "interactive_trajectory_gt_vs_est.html"
        add_page_context(page, scene, identity.get(scene, False))
        pages[scene] = page
        write_csv(scene_dir / "trajectory_summary.csv", scene_rows)
        write_scene_readme(scene_dir, scene, scene_rows, identity.get(scene, False))

        for variant in (
            "vio_preintegrated_full_imuatt_estinit",
            "vio_preintegrated_full_residual_gravity",
        ):
            diagnostic_rows.extend(diagnostic_summary(args.result_root, scene, variant))

    write_csv(args.out_root / "trajectory_summary.csv", summaries)
    write_csv(args.out_root / "diagnostic_summary.csv", diagnostic_rows)
    write_index(args.out_root, summaries, pages, identity)
    write_analysis_report(args.out_root, summaries, args.result_root)

    print(f"Index: {args.out_root / 'index.html'}")
    for scene, page in pages.items():
        print(f"{scene}: {page}")
    print(f"Summary: {args.out_root / 'trajectory_summary.csv'}")
    print(f"Diagnostics: {args.out_root / 'diagnostic_summary.csv'}")
    print(f"Report: {args.out_root / 'analysis_report_cn.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
