#!/usr/bin/env python3
"""Synthesize the V3 backend and production-UVD point-level audits."""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis_circle_v3_backend_pointlevel_20260716"
INFO = OUT / "information_strength"
WINDOW = OUT / "window_ablation"
POINT_3D = OUT / "point_level"
POINT_UVD = OUT / "point_level_uvd"
ORACLE = ROOT / "analysis_circle_translation_oracle_20260716"
MODES = tuple(f"P{i}" for i in range(8))


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def finite_correlation(x: pd.Series, y: pd.Series) -> dict[str, float | int | None]:
    values = pd.DataFrame({"x": pd.to_numeric(x, errors="coerce"), "y": pd.to_numeric(y, errors="coerce")}).dropna()
    if len(values) < 3 or values.x.std() == 0 or values.y.std() == 0:
        return {"count": int(len(values)), "pearson": None, "spearman": None}
    return {
        "count": int(len(values)),
        "pearson": float(stats.pearsonr(values.x, values.y).statistic),
        "spearman": float(stats.spearmanr(values.x, values.y).statistic),
    }


def enrich_contract() -> None:
    path = OUT / "v3_runtime_visual_contract.json"
    contract = load_json(path)
    contract["runtime_gate_thresholds"] = {
        "soft_inlier_ratio": 0.5,
        "reject_inlier_ratio": 0.2,
        "soft_mean_mahalanobis_sq": 9.0,
        "reject_mean_mahalanobis_sq": 100.0,
        "soft_whitened_pose_norm": 6.0,
        "reject_whitened_pose_norm": 20.0,
        "max_covariance_inflation": 1.0e6,
    }
    contract["important_scope_note"] = (
        "The production two-state loss consumes only the compressed relative-pose mean/covariance. "
        "Point/UVD fields are replayed by the frontend but do not enter TwoStateVIOProblem."
    )
    write_json(path, contract)


def visual_information_summary() -> dict[str, Any]:
    table = pd.read_csv(INFO / "backend_information_strength_summary.csv").set_index("case")
    full = load_json(ORACLE / "oracles/full/V3/summary.json")
    accumulation = load_json(OUT / "v3_pose_correction_accumulation.json")
    cases = {case: table.loc[case].replace({np.nan: None}).to_dict() for case in table.index}
    c0, c1, c2, c3, c4, c5, c6, c7 = (table.loc[f"C{i}"] for i in range(8))
    payload = {
        "scope": {
            "C0_C7": "same 391-frame short segment",
            "full_V3": "1890 frames, 1799 active two-state edges",
            "oracle_warning": "C2-C6 are sensitivity/oracle experiments, not deployable covariance calibration.",
        },
        "full_v3_metrics": full["metrics"],
        "cases": cases,
        "correction_accumulation": accumulation,
        "tests": {
            "C1_equals_C0_max_selected_metric_difference": float(
                max(
                    abs(float(c1[key]) - float(c0[key]))
                    for key in (
                        "ate_xy_rmse_m_no_alignment", "translation_rpe_rmse_m",
                        "velocity_truth_rmse_mps", "acc_bias_truth_rmse_mps2",
                        "gyro_bias_truth_rmse_radps",
                    )
                )
            ),
            "C2_C4_ate_monotonic_decrease": bool(
                c0.ate_xy_rmse_m_no_alignment > c2.ate_xy_rmse_m_no_alignment
                > c3.ate_xy_rmse_m_no_alignment > c4.ate_xy_rmse_m_no_alignment
            ),
            "C5_near_hard_ate_xy_m": float(c5.ate_xy_rmse_m_no_alignment),
            "C6_fixed_pose_ate_xy_m": float(c6.ate_xy_rmse_m_no_alignment),
            "C7_gt_chain_ate_xy_m": float(c7.ate_xy_rmse_m_no_alignment),
        },
        "decision": {
            "translation_dominates_full_v3_departure": True,
            "short_segment_huber_is_causal_source": False,
            "original_MACVO_pose_covariance_is_weak_for_GT_mean_oracle": True,
            "covariance_is_statistically_recalibrated_by_this_test": False,
            "free_pose_navigation_coupling_is_material": True,
            "evidence": (
                "C0->C4 monotonically reduces pose departure as visual information is strengthened; "
                "C6/C7 reproduce GT pose while C6 retains nonzero navigation-state costs."
            ),
        },
    }
    write_json(OUT / "v3_visual_information_oracle_summary.json", payload)
    return payload


def factor_and_bias_summary() -> dict[str, Any]:
    factor_source = INFO / "factor_hessian_gradient_per_edge.csv"
    factor_target = OUT / "v3_factor_information_per_edge.csv"
    shutil.copyfile(factor_source, factor_target)
    factor = pd.read_csv(factor_source)
    lifecycle = pd.read_csv(INFO / "bias_lifecycle_per_frame.csv")
    trace = pd.read_csv(INFO / "oracles/short/C0/factor_trace.csv")
    trace = (
        trace.sort_values(["frame_i", "solver_call"])
        .groupby(["frame_i", "frame_j"], as_index=False).tail(1)
    )
    update_rows = []
    for row in trace.itertuples(index=False):
        values = row._asdict()
        update_rows.append(
            {
                "frame_state": int(values["frame_i"]),
                "pose_i_translation_update_norm": float(np.linalg.norm([values[f"state_i_update_{k}"] for k in range(3)])),
                "pose_i_rotation_update_norm": float(np.linalg.norm([values[f"state_i_update_{k}"] for k in range(3, 6)])),
                "velocity_i_update_norm": float(np.linalg.norm([values[f"state_i_update_{k}"] for k in range(6, 9)])),
                "ba_i_update_norm": float(np.linalg.norm([values[f"state_i_update_{k}"] for k in range(9, 12)])),
                "bg_i_update_norm": float(np.linalg.norm([values[f"state_i_update_{k}"] for k in range(12, 15)])),
            }
        )
    update = pd.DataFrame(update_rows)
    initial = factor[factor.stage.eq("initial")]
    imu = initial[initial.factor.eq("imu")][["frame_i", "residual_norm", "cost", "gradient_norm"]].rename(
        columns={"frame_i": "frame_state", "residual_norm": "imu_initial_residual_norm", "cost": "imu_initial_cost", "gradient_norm": "imu_initial_gradient_norm"}
    )
    prior = initial[initial.factor.eq("prior")][["frame_i", "residual_norm", "cost", "gradient_norm"]].rename(
        columns={"frame_i": "frame_state", "residual_norm": "prior_initial_residual_norm", "cost": "prior_initial_cost", "gradient_norm": "prior_initial_gradient_norm"}
    )
    pose = pd.read_csv(OUT / "v3_pose_factor_runtime_per_edge.csv")
    pose = pose[["frame_i", "raw_rho_x", "raw_rho_y", "raw_rho_z", "raw_phi_x", "raw_phi_y", "raw_phi_z", "whitened_norm"]]
    pose["gt_pose_translation_correction_norm"] = np.linalg.norm(pose[["raw_rho_x", "raw_rho_y", "raw_rho_z"]], axis=1)
    pose["gt_pose_rotation_correction_norm"] = np.linalg.norm(pose[["raw_phi_x", "raw_phi_y", "raw_phi_z"]], axis=1)
    pose = pose.rename(columns={"frame_i": "frame_state", "whitened_norm": "gt_pose_whitened_correction_norm"})
    enriched = lifecycle.merge(update, on="frame_state", how="left").merge(imu, on="frame_state", how="left").merge(prior, on="frame_state", how="left").merge(
        pose[["frame_state", "gt_pose_translation_correction_norm", "gt_pose_rotation_correction_norm", "gt_pose_whitened_correction_norm"]], on="frame_state", how="left"
    )
    enriched.to_csv(OUT / "v3_bias_lifecycle_per_frame.csv", index=False)
    correlation_targets = {
        "imu_initial_residual_norm": "IMU residual",
        "gt_pose_translation_correction_norm": "GT-pose translation correction",
        "velocity_i_update_norm": "velocity correction",
        "prior_initial_gradient_norm": "Schur-prior gradient",
    }
    correlations = {}
    for bias in ("ba", "bg"):
        target = enriched[f"{bias}_next_reoptimization_norm"]
        correlations[bias] = {
            label: finite_correlation(target, enriched[column])
            for column, label in correlation_targets.items()
        }
    final_factor = factor[factor.stage.eq("final")]
    factor_summary = {}
    for name, group in final_factor.groupby("factor"):
        factor_summary[name] = {
            "cost_mean": float(group.cost.mean()),
            "hessian_trace_mean": float(group.hessian_trace.mean()),
            "gradient_norm_mean": float(group.gradient_norm.mean()),
            "pose_velocity_cross_fro_mean": float(group.H_pose_velocity_cross_fro.mean()),
            "pose_acc_bias_cross_fro_mean": float(group.H_pose_acc_bias_cross_fro.mean()),
            "pose_gyro_bias_cross_fro_mean": float(group.H_pose_gyro_bias_cross_fro.mean()),
            "velocity_acc_bias_cross_fro_mean": float(group.H_velocity_acc_bias_cross_fro.mean()),
            "rotation_gyro_bias_cross_fro_mean": float(group.H_rotation_gyro_bias_cross_fro.mean()),
        }
    checks = pd.read_csv(INFO / "factor_information_reconstruction_checks.csv")
    lifecycle_raw = load_json(INFO / "bias_lifecycle_summary.json")
    payload = {
        "factor_information": factor_summary,
        "linearization_reconstruction": {
            "hessian_sum_max_abs_error": float(checks.hessian_sum_max_abs_error.max()),
            "gradient_sum_max_abs_error": float(checks.gradient_sum_max_abs_error.max()),
        },
        "bias_lifecycle": lifecycle_raw,
        "bias_reoptimization_correlations": correlations,
        "interpretation": (
            "Bias carry-over is continuous to numerical precision, but the same state is reoptimized "
            "when it becomes the next source state. IMU and prior carry material pose/velocity/bias cross information."
        ),
    }
    write_json(OUT / "v3_factor_bias_summary.json", payload)
    return payload


def window_summary() -> dict[str, Any]:
    table = pd.read_csv(WINDOW / "v3_window_summary.csv").set_index("window")
    n2, n3, n5 = table.loc[2], table.loc[3], table.loc[5]
    payload = {
        "factor_contract": load_json(WINDOW / "N2/summary.json")["factor_contract"],
        "windows": {str(int(window)): row.replace({np.nan: None}).to_dict() for window, row in table.iterrows()},
        "N5_vs_N2": {
            "xy_ate_reduction_fraction": float(1.0 - n5.ate_xy_rmse_m_no_alignment / n2.ate_xy_rmse_m_no_alignment),
            "translation_rpe_reduction_fraction": float(1.0 - n5.translation_rpe_rmse_m / n2.translation_rpe_rmse_m),
            "velocity_rmse_reduction_fraction": float(1.0 - n5.velocity_truth_rmse_mps / n2.velocity_truth_rmse_mps),
            "acc_bias_rmse_change_fraction": float(n5.acc_bias_truth_rmse_mps2 / n2.acc_bias_truth_rmse_mps2 - 1.0),
            "gyro_bias_rmse_reduction_fraction": float(1.0 - n5.gyro_bias_truth_rmse_radps / n2.gyro_bias_truth_rmse_radps),
        },
        "decision": (
            "N=5 modestly reduces RPE/velocity/bg error, but changes XY ATE by less than 1% and does not improve ba. "
            "The N=2 window is a secondary amplifier, not the dominant V3 residual source."
        ),
        "production_promotion_approved": False,
    }
    write_json(OUT / "v3_window_diagnostic_summary.json", payload)
    return payload


def point_level_summary() -> dict[str, Any]:
    uvd = load_json(POINT_UVD / "macvo_uvd_translation_counterfactual_summary.json")
    sidecar = load_json(POINT_3D / "macvo_translation_counterfactual_summary.json")
    aggregate = {row["mode"]: row for row in uvd["aggregate"]}
    points = pd.read_csv(
        POINT_UVD / "macvo_uvd_point_contributions.csv",
        usecols=[
            "mode", "frame_i", "selected", "gt_uvd_residual_u", "gt_uvd_residual_v",
            "gt_uvd_residual_disp", "gt_uvd_whitened_norm", "gt_translation_gradient_lateral",
            "depth_cov_sum", "flow_u_cov", "flow_v_cov", "disp_cov",
        ],
    )
    points = points[points["mode"].eq("P0")].copy()
    points["flow_cov_trace"] = points.flow_u_cov + points.flow_v_cov
    points["gt_flow_residual_norm"] = np.hypot(points.gt_uvd_residual_u, points.gt_uvd_residual_v)
    moving = points[np.isfinite(points.gt_translation_gradient_lateral)]
    point_correlations = {
        name: finite_correlation(moving.gt_translation_gradient_lateral, moving[column])
        for name, column in {
            "signed_disparity_residual": "gt_uvd_residual_disp",
            "flow_residual_norm": "gt_flow_residual_norm",
            "depth_covariance": "depth_cov_sum",
            "flow_covariance": "flow_cov_trace",
            "disparity_covariance": "disp_cov",
        }.items()
    }
    edge_gradient = moving.groupby("frame_i", as_index=False).agg(
        gt_lateral_gradient_sum=("gt_translation_gradient_lateral", "sum"),
        gt_disp_residual_mean=("gt_uvd_residual_disp", "mean"),
        gt_flow_residual_mean=("gt_flow_residual_norm", "mean"),
    )
    per_edge = pd.read_csv(POINT_UVD / "macvo_uvd_translation_counterfactual_per_edge.csv")
    p0 = per_edge[per_edge["mode"].eq("P0")][["frame_i", "e_lateral"]]
    edge_gradient = edge_gradient.merge(p0, on="frame_i", how="left")
    edge_correlations = {
        "lateral_pose_error_vs_GT_lateral_cost_gradient": finite_correlation(edge_gradient.e_lateral, edge_gradient.gt_lateral_gradient_sum),
        "lateral_pose_error_vs_mean_disparity_residual": finite_correlation(edge_gradient.e_lateral, edge_gradient.gt_disp_residual_mean),
        "lateral_pose_error_vs_mean_flow_residual": finite_correlation(edge_gradient.e_lateral, edge_gradient.gt_flow_residual_mean),
    }
    refit = pd.read_csv(POINT_UVD / "macvo_uvd_original_objective_refit.csv")
    refit_summary = {
        "translation_error_mean_before_m": float(refit.translation_error_before_m.mean()),
        "translation_error_mean_after_m": float(refit.translation_error_after_refit_m.mean()),
        "rotation_error_mean_before_rad": float(refit.rotation_error_before_rad.mean()),
        "rotation_error_mean_after_rad": float(refit.rotation_error_after_refit_rad.mean()),
        "improved_translation_edge_count": int((refit.translation_error_after_refit_m < refit.translation_error_before_m).sum()),
        "edge_count": int(len(refit)),
    }
    p0_value = aggregate["P0"]
    improvements = {
        mode: {
            "translation_error_reduction_fraction": float(
                1.0 - aggregate[mode]["translation_error_norm_mean_m"] / p0_value["translation_error_norm_mean_m"]
            ),
            "absolute_lateral_bias_reduction_fraction": float(
                1.0 - abs(aggregate[mode]["lateral_error_mean_m"]) / abs(p0_value["lateral_error_mean_m"])
            ),
        }
        for mode in ("P1", "P2", "P3", "P4", "P5", "P6", "P7")
    }
    payload = {
        "scope": "7 representative edges; causal diagnostic, not a full-sequence production correction",
        "production_UVD_contract": uvd["production_contract"],
        "self_checks": uvd["self_checks"],
        "production_UVD_counterfactuals": uvd["aggregate"],
        "improvement_vs_P0": improvements,
        "production_objective_continued_refit": refit_summary,
        "point_level_correlations": point_correlations,
        "representative_edge_correlations": edge_correlations,
        "sidecar_3D3D_audit_is_separate": {
            "reason": (
                "relative_pose_information_from_packet uses symmetric 3D-3D residuals only to compute the 6x6 "
                "sidecar covariance at the pure-MACVO mean. The pure-MACVO mean itself came from graph_type=disp."
            ),
            "results": sidecar["aggregate"],
        },
        "decision_B": {
            "dominant_evidence": "rotation/translation coupling in the production UVD objective",
            "support": (
                "P1 fixes GT rotation and reduces mean translation error/lateral bias strongly; P2 keeps MACVO "
                "rotation and reoptimizes translation but is effectively unchanged."
            ),
            "depth_role": "material secondary contributor; P5 outperforms equal-size low-flow P4",
            "learned_covariance_role": "contributes jointly with inlier selection; P3 improves but changes two mechanisms",
            "robust_gate_role": "secondary; stricter P6 helps less than P1/P5",
            "spatial_distribution_role": "naive sparse stratification P7 worsens the representative result",
            "frontend_early_stop_role": (
                "secondary/mixed; continued production-objective refit improves only a subset of edges and does not fix the worst edges"
            ),
        },
    }
    write_json(OUT / "macvo_translation_counterfactual_summary.json", payload)
    return payload


def reports(visual: dict[str, Any], factor: dict[str, Any], window: dict[str, Any], point: dict[str, Any]) -> None:
    agg = {row["mode"]: row for row in point["production_UVD_counterfactuals"]}
    lines = [
        "# MACVO 相对平移点级根因报告", "",
        "> 本报告只解释原始 MACVO `t_mac`；V3 的 GT-mean 后端残余见综合报告。所有 GT 只用于离线诊断。", "",
        "## 契约", "",
        "原始纯 MACVO 使用 `graph_type: disp`。每个点的生产残差是预测当前帧 `[u,v,disparity]` 减观测值；上一帧局部 3D 点固定，权重来自当前帧 UV 与 disparity covariance，鲁棒核为 Huber `delta=0.1`。", "",
        "sidecar 的对称 3D-3D 模型只负责在 MACVO mean 处计算 6x6 pose covariance，不负责生成 MACVO mean，因此两套点级结果已分开保存。", "",
        "## 代表边 P0-P7", "",
        "| 模式 | 平均平移误差 (m) | 平均横向误差 (m) | 平均旋转误差 (rad) | 平均点数 |", "|---|---:|---:|---:|---:|",
    ]
    for mode in MODES:
        row = agg[mode]
        lines.append(
            f"| {mode} | {row['translation_error_norm_mean_m']:.6g} | {row['lateral_error_mean_m']:.6g} | "
            f"{row['rotation_error_mean_rad']:.6g} | {row['selected_points_mean']:.1f} |"
        )
    lines += [
        "", "## 结论", "",
        "1. **R/t 耦合是最强证据。** P1 固定 GT rotation 后，平均平移误差和横向偏置显著下降；P2 保留 MACVO rotation、只重解 translation，结果几乎不变。", "",
        "2. **深度质量是重要的次级因素。** 在相同平均点数下，P5 低 depth-cov 点明显优于 P4 低 flow-cov 点，尤其是横向偏置。", "",
        "3. **learned covariance、点筛选和鲁棒门控共同作用，但不是单一根因。** P3 同时做 inlier 等权，不能把改善单独归因于 covariance；P6 更严格门控的收益小于 P1/P5。", "",
        "4. **简单空间/深度均衡采样没有解决问题。** P7 点数少且误差更大，说明不能用稀疏分层采样直接替代现有选择器。", "",
        "5. **继续迭代原 UVD 目标不是充分修复。** 它只改善部分代表边，P95/最大误差边仍不改善，因此前端提前停止是次要因素。", "",
        "原始逐边和逐点数据分别见 `point_level_uvd/macvo_uvd_translation_counterfactual_per_edge.csv` 与 `point_level_uvd/macvo_uvd_point_contributions.csv`。", "",
    ]
    (OUT / "macvo_translation_point_level_report_cn.md").write_text("\n".join(lines), encoding="utf-8")

    c0 = visual["cases"]["C0"]
    c4 = visual["cases"]["C4"]
    combined = [
        "# 完整圆形 V3 后端与 MACVO 点级诊断", "",
        "## 首页结论", "",
        "**结论 A：V3 的 1.021444 m XY ATE 不是 GT relative-pose mean 或 SE(3) 连乘错误。** 它由后端逐边偏离 GT mean 累积，几乎全部来自 translation correction。原 MACVO pose covariance 在 GT-mean oracle 中给视觉的信息量偏弱，允许 IMU、velocity、bias 和 Schur prior 重新分配冲突。", "",
        "**结论 B：原始 MACVO 横向相对平移偏置的最强证据是生产 UVD 目标中的 rotation/translation 耦合，depth 质量是重要次级因素。** 这与 V3 后端残余是两条独立主线。", "",
        "## 结论 A 的证据", "",
        f"- 完整 V3：XY ATE `{visual['full_v3_metrics']['ate_xy_rmse_m_no_alignment']:.6f} m`。逐边修正可按 SE(3) 精确重建轨迹，最大矩阵误差 `{visual['correction_accumulation']['reconstruction_max_abs_transform_error']:.3e}`。", "",
        f"- 只累计 rotation correction 时 XY RMSE `{visual['correction_accumulation']['rotation_component_only_metrics']['ate_xy_rmse_m']:.6f} m`；只累计 translation correction 时 `{visual['correction_accumulation']['translation_component_only_metrics']['ate_xy_rmse_m']:.6f} m`。", "",
        f"- 391 帧 C0→C4 将视觉 covariance 从 1.0 缩到 0.03，XY ATE 从 `{c0['ate_xy_rmse_m_no_alignment']:.6f}` 单调降到 `{c4['ate_xy_rmse_m_no_alignment']:.6f} m`。这只是信息灵敏度证据，不是正式 covariance 标定。", "",
        "- C1 关闭 Huber 与 C0 完全一致，因此短片段 Huber 不是根因；完整序列后段存在 Huber 触发，尚未做完整 C1 因果实验。", "",
        "- C6 固定 GT pose 后，pose 数值闭合且 velocity RMSE 约 0.00132 m/s，但 IMU/prior cost 不能再靠移动 pose 吸收，证明 navigation states 与 GT pose 之间存在实际冲突。", "",
        "- Bias 没有断链：上一边终点到下一边起点误差在 1e-9/1e-10 量级；但同一 bias 成为下一条 IMU edge 起点后会再次被优化。", "",
        f"- N=2→N=5 只使 XY ATE 改善 `{window['N5_vs_N2']['xy_ate_reduction_fraction']*100:.2f}%`、平移 RPE 改善 `{window['N5_vs_N2']['translation_rpe_reduction_fraction']*100:.2f}%`。窗口过短是次级放大器。", "",
        "## 尚不能宣称", "",
        "- 不能把 C4/C5 的 covariance 倍率用于生产；它们使用 GT mean，只是 oracle。", "",
        "- 本轮没有单独关闭 Schur prior，因此不能把 prior 宣布为单一根因。", "",
        "- P0-P7 只有代表边，不能直接作为完整序列修复参数。", "",
        "## 下一步", "",
        "1. 在不使用 GT 的前提下，对 MACVO rotation 做可部署的质量诊断，重点检查 turn 段微小 rotation error 如何通过深度投影放大成 lateral translation。", "",
        "2. 用留出数据校准 relative-pose 6x6 covariance，而不是采用 oracle 缩放倍率；应检查 NIS 分方向和 eigenbasis。", "",
        "3. 设计只改变一个机制的前端实验：固定点集比较 depth covariance、flow covariance 与 rotation 初值，随后再考虑更长窗口。", "",
    ]
    (OUT / "circle_v3_backend_and_macvo_pointlevel_report_cn.md").write_text("\n".join(combined), encoding="utf-8")


def main() -> None:
    enrich_contract()
    visual = visual_information_summary()
    factor = factor_and_bias_summary()
    window = window_summary()
    point = point_level_summary()
    reports(visual, factor, window, point)
    print(json.dumps({
        "visual_information": visual["decision"],
        "window": window["decision"],
        "point_level": point["decision_B"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
