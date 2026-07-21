#!/usr/bin/env python3
"""Turn U1 counterfactual CSVs into a conservative mode-selection decision."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr, wilcoxon


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "analysis_u1_counterfactual_branches_20260719"
MODES = ("rotation_only", "translation_only", "no_visual", "alt_rt")


def distribution(values: pd.Series) -> dict[str, float]:
    array = values.to_numpy(np.float64)
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p95": float(np.quantile(array, 0.95)),
        "max": float(np.max(array)),
    }


def direction_free_auc(labels: pd.Series, values: pd.Series) -> dict[str, float]:
    y = labels.to_numpy(np.int64)
    x = values.to_numpy(np.float64)
    valid = np.isfinite(x)
    y = y[valid]
    x = x[valid]
    positive = int(y.sum())
    negative = int(len(y) - positive)
    if positive == 0 or negative == 0:
        return {"auc": float("nan"), "raw_auc": float("nan")}
    ranks = rankdata(x)
    raw = float(
        (ranks[y == 1].sum() - positive * (positive + 1) / 2)
        / (positive * negative)
    )
    return {"auc": max(raw, 1.0 - raw), "raw_auc": raw}


def paired_mode_stats(
    frame: pd.DataFrame,
    index: str,
    mode: str,
    *,
    translation_threshold: float,
    rotation_threshold: float,
) -> dict[str, object]:
    full = frame[frame["mode"] == "full"].set_index(index)
    candidate = frame[frame["mode"] == mode].set_index(index)
    common = full.index.intersection(candidate.index)
    full = full.loc[common]
    candidate = candidate.loc[common]
    delta_t = candidate["translation_error_norm"] - full["translation_error_norm"]
    delta_r = candidate["rotation_error_norm"] - full["rotation_error_norm"]
    delta_v = candidate["velocity_error_norm"] - full["velocity_error_norm"]
    practical = (delta_t < -translation_threshold) & (
        delta_r < -rotation_threshold
    )
    raw_pareto = (delta_t < 0.0) & (delta_r < 0.0)
    t_wilcoxon = float(wilcoxon(delta_t).pvalue) if np.any(delta_t != 0.0) else 1.0
    r_wilcoxon = float(wilcoxon(delta_r).pvalue) if np.any(delta_r != 0.0) else 1.0
    return {
        "count": int(len(common)),
        "delta_translation": distribution(delta_t),
        "delta_rotation": distribution(delta_r),
        "delta_velocity": distribution(delta_v),
        "raw_pareto_better_count": int(raw_pareto.sum()),
        "practical_pareto_better_count": int(practical.sum()),
        "translation_wilcoxon_p": t_wilcoxon,
        "rotation_wilcoxon_p": r_wilcoxon,
    }


def predictor_analysis(
    immediate: pd.DataFrame, lookahead_uniform: pd.DataFrame
) -> dict[str, object]:
    delayed_full = lookahead_uniform[
        lookahead_uniform["mode"] == "full"
    ].set_index("seed_frame_i")
    delayed_t = lookahead_uniform[
        lookahead_uniform["mode"] == "translation_only"
    ].set_index("seed_frame_i")
    common = delayed_full.index.intersection(delayed_t.index)
    delayed_full = delayed_full.loc[common]
    delayed_t = delayed_t.loc[common]
    delta_t = delayed_t["translation_error_norm"] - delayed_full[
        "translation_error_norm"
    ]
    delta_r = delayed_t["rotation_error_norm"] - delayed_full[
        "rotation_error_norm"
    ]
    labels = ((delta_t < -1.0e-3) & (delta_r < -1.0e-4)).astype(np.int64)
    benefit = -(
        delta_t / max(float(delayed_full["translation_error_norm"].median()), 1e-12)
        + delta_r
        / max(float(delayed_full["rotation_error_norm"].median()), 1e-12)
    )

    current_full = immediate[immediate["mode"] == "full"].set_index("frame_i")
    current_t = immediate[
        immediate["mode"] == "translation_only"
    ].set_index("frame_i")
    current_full = current_full.loc[common]
    current_t = current_t.loc[common]
    feature_map: dict[str, pd.Series] = {}
    for column in (
        "initial_inlier_ratio",
        "initial_mean_mahalanobis_sq",
        "initial_whitened_norm",
        "h_tt_min_eigenvalue",
        "h_tt_condition",
        "h_rr_min_eigenvalue",
        "h_rr_condition",
        "h_tr_normalized_max_singular",
        "coverage",
        "depth_spread",
        "visual_obs_cov_mean",
        "prior_cost",
        "imu_cost",
        "visual_cost",
        "iterations",
        "final_gradient_inf_norm",
    ):
        feature_map[f"full_{column}"] = current_full[column]
    for column in (
        "final_cost",
        "prior_cost",
        "imu_cost",
        "visual_cost",
        "iterations",
        "final_gradient_inf_norm",
    ):
        feature_map[f"translation_only_{column}"] = current_t[column]
        feature_map[f"delta_{column}"] = current_t[column] - current_full[column]

    rows: list[dict[str, float | str]] = []
    for name, values in feature_map.items():
        valid = np.isfinite(values.to_numpy(np.float64))
        rho, p_value = spearmanr(values.to_numpy(np.float64)[valid], benefit[valid])
        auc = direction_free_auc(labels[valid], values[valid])
        rows.append(
            {
                "feature": name,
                "direction_free_auc": float(auc["auc"]),
                "raw_auc": float(auc["raw_auc"]),
                "spearman_rho": float(rho),
                "spearman_p": float(p_value),
            }
        )
    rows.sort(
        key=lambda row: (
            -np.nan_to_num(float(row["direction_free_auc"]), nan=-1.0),
            -abs(np.nan_to_num(float(row["spearman_rho"]), nan=0.0)),
        )
    )
    return {
        "uniform_seed_count": int(len(common)),
        "practical_translation_only_benefit_count": int(labels.sum()),
        "positive_seed_frames": [int(value) for value in labels[labels == 1].index],
        "features_ranked": rows,
    }


def main() -> int:
    immediate = pd.read_csv(OUTPUT / "immediate_counterfactual_per_edge.csv")
    lookahead = pd.read_csv(OUTPUT / "lookahead_counterfactual_per_seed.csv")
    uniform = lookahead[lookahead["seed_frame_i"] % 10 == 0].copy()

    immediate_stats = {
        mode: paired_mode_stats(
            immediate,
            "frame_i",
            mode,
            translation_threshold=1.0e-4,
            rotation_threshold=1.0e-5,
        )
        for mode in MODES
    }
    delayed_uniform_stats = {
        mode: paired_mode_stats(
            uniform,
            "seed_frame_i",
            mode,
            translation_threshold=1.0e-3,
            rotation_threshold=1.0e-4,
        )
        for mode in MODES
    }
    predictor = predictor_analysis(immediate, uniform)
    decision = {
        "hard_per_edge_switch_approved": False,
        "production_default": "full direct-UVD U1",
        "discard_as_adaptive_candidates": [
            "rotation_only",
            "no_visual",
            "alt_rt",
        ],
        "retain_for_shadow_validation_only": ["translation_only"],
        "reason": (
            "R-only improvement is transient and sub-millimetric; alternating "
            "returns the FULL optimum at roughly triple stage work; T-only has "
            "a real but inconsistent five-edge benefit with an accuracy tradeoff."
        ),
        "next_experiment": (
            "Evaluate FULL versus translation-only on all valid five-edge seeds "
            "for circle, rectangle, and straight; add causal visual/IMU rotation "
            "innovation NIS features and validate a shadow selector by leaving one "
            "scene out before any production switch is allowed."
        ),
    }
    payload = {
        "thresholds": {
            "immediate_translation_m": 1.0e-4,
            "immediate_rotation_rad": 1.0e-5,
            "lookahead_translation_m": 1.0e-3,
            "lookahead_rotation_rad": 1.0e-4,
        },
        "immediate": immediate_stats,
        "lookahead_uniform_stride_10": delayed_uniform_stats,
        "predictor": predictor,
        "decision": decision,
        "limitations": [
            "The five-edge unbiased subset has 30 seeds from one circle scene.",
            "The additional 15 delayed seeds were GT-selected and are excluded from the causal decision.",
            "No mode classifier has yet been validated on an unseen scene.",
        ],
    }
    json_path = OUTPUT / "adaptive_mode_decision_analysis.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# U1自适应模式决策",
        "",
        "## 结论",
        "",
        "当前不批准把R-only/T-only/FULL硬切换接入生产。生产默认继续使用完整Direct UVD U1。",
        "",
        "- R-only的即时变化只有微小量级，五边后反而略差。",
        "- T-only在无偏30个种子中有9个达到实用双改善，但总体以旋转和速度变差换取平移改善。",
        "- R->T->FULL最终回到与FULL相同的解，却需要三阶段求解，不值得保留。",
        "- 无视觉会明显损失旋转约束。",
        "",
        "## 无偏五边结果",
        "",
        "| 模式 | 平移差中位数(m) | 旋转差中位数(rad) | 速度差中位数(m/s) | 实用双改善/30 |",
        "|---|---:|---:|---:|---:|",
    ]
    for mode in MODES:
        item = delayed_uniform_stats[mode]
        lines.append(
            f"| {mode} | {item['delta_translation']['median']:.6g} | "
            f"{item['delta_rotation']['median']:.6g} | "
            f"{item['delta_velocity']['median']:.6g} | "
            f"{item['practical_pareto_better_count']}/30 |"
        )
    top = predictor["features_ranked"][0]
    lines.extend(
        [
            "",
            "## 可预测性",
            "",
            f"现有最强单一非GT指标是`{top['feature']}`，方向无关AUC为"
            f"`{top['direction_free_auc']:.3f}`；样本只有30个，尚不足以形成生产门限。",
            "",
            "## 下一步",
            "",
            "只保留FULL与T-only两个shadow候选，在圆形、矩形、直线的全部有效五边种子上评估；补充视觉/IMU旋转innovation NIS，并采用留一场景验证。只有未见场景仍能稳定获益，才实现带滞回的在线选择器。",
        ]
    )
    report_path = OUTPUT / "u1_adaptive_mode_decision_cn.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(report_path)
    print(json_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
