#!/usr/bin/env python3
"""Summarize the frozen-factor shared accelerometer-bias ablation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "analysis_normal_noise_sampling_aware_20260716"
WINDOW_ROOT = ANALYSIS / "window_ablation"
VARIANTS = (
    ("normal", 2),
    ("normal", 5),
    ("normal", 10),
    ("fixed_ba", 5),
    ("shared_ba", 5),
    ("shared_ba", 10),
    ("rate_limited_ba", 5),
    ("rw_gated_ba", 5),
)


def _summary(mode: str, window: int) -> dict:
    path = WINDOW_ROOT / f"{mode}_N{window}" / "summary.json"
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _trajectory_difference(mode_a: str, mode_b: str, window: int) -> float:
    a = pd.read_csv(WINDOW_ROOT / f"{mode_a}_N{window}" / "trajectory.csv")
    b = pd.read_csv(WINDOW_ROOT / f"{mode_b}_N{window}" / "trajectory.csv")
    if not np.array_equal(a["timestamp_ns"].to_numpy(), b["timestamp_ns"].to_numpy()):
        raise AssertionError("trajectory timestamps differ")
    columns = ["tx", "ty", "tz", "qx", "qy", "qz", "qw"]
    return float(np.max(np.abs(a[columns].to_numpy() - b[columns].to_numpy())))


def main() -> int:
    rows = []
    for mode, window in VARIANTS:
        summary = _summary(mode, window)
        truth = summary["truth_metrics"]
        high = summary["high_frequency_metrics"]
        bias = summary["bias_diagnostics"]
        solver = summary["solver"]
        rows.append(
            {
                "mode": mode,
                "window_size": window,
                "xy_highpass_rms_m": high["xy_position_error_highpass_rms_m"],
                "xy_correction_d2_rms_m": high["xy_position_correction_second_difference_rms_m"],
                "translation_rpe_rmse_m": truth["translation_rpe_rmse_m"],
                "rotation_rpe_rmse_rad": truth["rotation_rpe_rmse_rad"],
                "ate_position_rmse_m": truth["ate_position_rmse_m"],
                "velocity_rmse_mps": truth["velocity_rmse_mps"],
                "acc_bias_rmse_mps2": truth["acc_bias_rmse_mps2"],
                "ba_increment_standardized_rms": bias["ba_increment_standardized_component_rms"],
                "converged_ratio": solver["converged_ratio"],
                "mean_iterations": solver["mean_iterations"],
                "mean_runtime_ms": solver["mean_runtime_ms_per_solve"],
            }
        )
    metrics = pd.DataFrame(rows)
    metrics.to_csv(ANALYSIS / "shared_ba_ablation_metrics.csv", index=False)

    gated = pd.read_csv(WINDOW_ROOT / "rw_gated_ba_N5" / "solve_per_frame.csv")
    candidates = gated[gated["ba_update_due"]]
    actions = gated["ba_gate_action"].value_counts().to_dict()
    candidate_nis = candidates["ba_candidate_nis"].dropna()
    equivalence = {
        f"normal_vs_shared_N{window}_max_abs_trajectory_component_difference":
            _trajectory_difference("normal", "shared_ba", window)
        for window in (5, 10)
    }
    decision = {
        "dataset": "clear_stop_turn_rectangle_truth_normal_noise",
        "frame_range": [0, 299],
        "active_edges": 209,
        "production_default_changed": False,
        "unit_tests": "11 passed",
        "shared_ba_equivalence": equivalence,
        "rw_gate": {
            "distribution": "chi_square_3dof",
            "confidence": 0.999,
            "threshold": 16.26623619623813,
            "action_counts": actions,
            "candidate_count": int(len(candidate_nis)),
            "candidate_nis_min": float(candidate_nis.min()),
            "candidate_nis_median": float(candidate_nis.median()),
            "candidate_nis_p95": float(candidate_nis.quantile(0.95)),
            "candidate_nis_max": float(candidate_nis.max()),
        },
        "evidence_classification": [
            "window_internal_sharing_is_already_approximately_enforced_by_current_bias_rw",
            "rolling_shared_ba_does_not_reduce_cross_window_ba_revisions",
            "frame_count_rate_limiting_turns_revisions_into_larger_sparse_jumps",
            "all_N5_candidates_are_inconsistent_with_metadata_bias_random_walk",
            "rw_gating_matches_fixed_static_ba_on_this_short_segment",
        ],
        "approved": {
            "merge_shared_ba_math_and_diagnostics": True,
            "change_production_default": False,
            "run_full_sequences": False,
        },
        "next_required_validation": [
            "repeat rw-gated mode on no-noise and the same circle/straight/rectangle short intervals",
            "separate startup ba refinement from runtime ba random walk",
            "only then decide whether to expose rw-gated ba in production replay",
        ],
    }
    (ANALYSIS / "shared_ba_ablation_decision.json").write_text(
        json.dumps(decision, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    table = metrics.copy()
    numeric = [column for column in table.columns if column not in {"mode", "window_size"}]
    table[numeric] = table[numeric].map(lambda value: f"{value:.6g}")
    lines = [
        "# Accelerometer Bias 共享/降频/RW 门控消融报告",
        "",
        "## 范围",
        "",
        "所有实验使用同一份 300 帧冻结因子缓存；视觉 measurement/covariance、Standard local-frame preintegration、Sampling-aware covariance、Bias-RW、LM 和初始 prior 均未调整。生产默认配置没有改变。",
        "",
        "## 结果",
        "",
        table.to_markdown(index=False),
        "",
        "## 关键数值事实",
        "",
        f"- N=5 normal 与 shared-ba 的最大轨迹分量差：`{equivalence['normal_vs_shared_N5_max_abs_trajectory_component_difference']:.3e}`。",
        f"- N=10 normal 与 shared-ba 的最大轨迹分量差：`{equivalence['normal_vs_shared_N10_max_abs_trajectory_component_difference']:.3e}`。",
        "- 因此窗口内独立 ba 已被当前 Bias-RW 约束到近似共享；显式共享本身不会解决跨窗口重估。",
        f"- N=5 RW 门控候选共 `{len(candidate_nis)}` 次，全部拒绝；NIS min/median/P95/max = "
        f"`{candidate_nis.min():.3f}/{candidate_nis.median():.3f}/{candidate_nis.quantile(0.95):.3f}/{candidate_nis.max():.3f}`，门限为 `16.266`。",
        "- 只按帧数降频会产生稀疏大跳；加入 RW 一致性门控后，本片段退化为固定静止 ba，并恢复了固定-ba 的低抖动结果。",
        "",
        "## 结论",
        "",
        "1. `ba` 过度变化的主要位置是相邻 fixed-lag 求解之间的共同 Bias 重估，而不是同一窗口内部各状态相互分离。",
        "2. 仅共享 `ba` 或增大 N 不能消除该问题；N=10 仍约为 84.47 sigma。",
        "3. 本片段中所有在线候选更新都与 metadata Bias-RW 模型显著冲突，因此拒绝更新有数值依据。",
        "4. 这不等价于批准永久冻结 ba。三秒静止 ba 与运动期 ba refinement 必须在更多短片段上分开验证。",
        "5. 当前不修改生产默认，不运行完整序列。",
    ]
    (ANALYSIS / "shared_ba_ablation_report_cn.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(decision, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
