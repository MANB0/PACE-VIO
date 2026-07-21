#!/usr/bin/env python3
"""Build the Chinese Gate 0-6 evidence report from frozen JSON artifacts."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis_normal_noise_sampling_aware_20260716"


def read(name: str):
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def percent(change: float) -> str:
    return f"{change:+.1f}%"


def main() -> int:
    baseline = read("baseline_normal_noise_summary.json")
    covariance = read("sampling_aware_covariance_summary.json")
    mc = read("sampling_aware_mc_summary.json")
    n2 = read("sampling_covariance_n2_comparison.json")
    bias = read("macvio_bias_ablation_summary.json")
    decision = read("macvio_window_final_recommendation.json")
    windows = {}
    for mode in ("normal", "fixed_ba"):
        for window in (2, 3, 5, 10):
            windows[(mode, window)] = json.loads(
                (OUT / f"window_ablation/{mode}_N{window}/summary.json").read_text(encoding="utf-8")
            )

    lines = [
        "# MACVIO Normal-noise Sampling-aware / Bias / Window 最终报告",
        "",
        "> 本任务的目标是降低 MACVIO 对 GT 的高频误差，不是让 MACVIO 轨迹或 Bias 接近 GTSAM。",
        "",
        "## 首页结论",
        "",
        "1. **Sampling-aware covariance 数学上通过。** 原始独立采样噪声经过端点插值与 midpoint 共享后，Monte Carlo 的 NIS9 均值为 "
        f"{mc['NIS_sampling_mean']['mean']:.4f}，P/V/R 分块均值分别为 "
        f"{mc['NIS_sampling_p_mean']['mean']:.4f}/{mc['NIS_sampling_v_mean']['mean']:.4f}/{mc['NIS_sampling_R_mean']['mean']:.4f}。",
        "2. **它不是抖动修复。** 在 N=2 中替换为统计正确的 P 后，XY 高频 RMS 和 correction 二阶差分均恶化约 10.6%，因此不能为了轨迹更平滑回退到错误 covariance。",
        "3. **加速度计 Bias 过度活跃是当前主因。** 正常 B1 的 ba increment 为理论 RW 尺度的约 "
        f"{bias['B1']['bias_diagnostics']['ba_increment_standardized_component_rms']:.1f} 倍；固定 ba、继续优化 bg 的 B5 令 XY 高频、二阶差分和速度高频分别下降 "
        f"{-bias['B5']['high_frequency_change_vs_B1_percent']['xy_position_error_highpass_rms_m']:.1f}%/"
        f"{-bias['B5']['high_frequency_change_vs_B1_percent']['xy_position_correction_second_difference_rms_m']:.1f}%/"
        f"{-bias['B5']['high_frequency_change_vs_B1_percent']['velocity_truth_error_highpass_rms_mps']:.1f}%。",
        "4. **窗口变长有帮助，但不是完整修复。** normal N=10 相对 N=2 的两项高频指标达到 21.1%/20.6% 改善，RPE/ATE 也改善；但 ba increment 仍约 84.5 倍理论 RW。固定 ba 后 N=10 仅改善 14.5%/12.7%，未达到 20% 门槛。",
        "5. **当前不批准完整序列。** 没有任何窗口同时通过高频、GT 精度、Bias 活跃度和计算成本门槛。最终分类为 **B（Bias 过度活跃主导）+ E（多因素非加性共同作用）**。",
        "",
        "## 受控输入",
        "",
        f"- 场景：`{baseline['scene']}`；帧 0-299；有效边 90->91 至 298->299，共 {baseline['edge_count']} 条。",
        "- 预积分：`standard_local_frame_preintegration`；残差/协方差顺序 `[p,v,R]`。",
        "- 固定连续时间参数：sigma_a=0.0141258、sigma_g=0.00182898、sigma_aw=0.000386071、sigma_gw=3.57864e-05。",
        "- 所有窗口共用同一冻结因子 manifest；Delta、P_sampling、Bias Jacobian、Bias RW、视觉相对位姿与 6x6 covariance 的数组哈希不变。",
        "- 209 条视觉边均为 accept，covariance inflation=1；未改 LM、Huber、门控、初始 prior 或 sigma。",
        "- 300 帧片段结束于第一次矩形转弯/停车之前，因此本报告不能证明转弯和停车局部峰值已修复。",
        "",
        "## 关卡结果",
        "",
        "| 关卡 | 证据 | 结果 |",
        "|---|---|---|",
        f"| 0 冻结基线 | 209/209 收敛；ATE {baseline['truth_metrics_valid_frames']['ate_position_rmse_m']:.6g} m | 通过 |",
        "| 1 Sampling 映射 | 6270 个 impulse；最大误差 1.35e-8 | 通过 |",
        f"| 2 P 传播 | Sampling/current trace ratio median {covariance['trace_ratio_sampling_over_current']['median']:.6g} | 通过 |",
        f"| 3 Monte Carlo | P_sampling/MC Frobenius median {mc['P_sampling_vs_MC_relative_frobenius']['median']*100:.2f}%，P95 {mc['P_sampling_vs_MC_relative_frobenius']['p95']*100:.2f}% | 通过 |",
        f"| 4 N=2 P 替换 | XY 高频 {percent(n2['high_frequency_metrics']['xy_position_error_highpass_rms_m']['relative_change_percent'])} | 统计正确，但未降抖 |",
        "| 5 Bias 消融 | B5 高频三项显著改善；B4 明显恶化 | ba 过度活跃为主因 |",
        "| 6 窗口 | N=10 部分达标，但 Bias/成本门槛失败 | 不批准生产窗口 |",
        "",
        "## Sampling-aware Covariance",
        "",
        f"`P_sampling` 对 Monte Carlo 的完整 9x9 Frobenius 相对误差中位数为 {mc['P_sampling_vs_MC_relative_frobenius']['median']:.4f}，旧 P 为 {mc['P_current_vs_MC_relative_frobenius']['median']:.4f}。",
        f"Sampling 白化 covariance 对角线范围为 [{mc['whitened_covariance_diag_global_min']:.4f}, {mc['whitened_covariance_diag_global_max']:.4f}]，最大绝对非对角相关为 {mc['whitened_max_abs_offdiag_correlation']['max']:.4f}。",
        "这些量证明 covariance 恢复了真实采样相关性；它给 IMU 更高而不是更低的信息量，所以 N=2 轨迹略抖并不矛盾。",
        "",
        "## Bias 消融",
        "",
        "| 模式 | XY 高频 RMS (m) | correction 二阶差分 (m) | velocity 高频 (m/s) | 平移 RPE (m) | ATE (m) | ba RMSE |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ("B1", "B2", "B3", "B4", "B5"):
        h, t = bias[name]["high_frequency_metrics"], bias[name]["truth_metrics"]
        lines.append(
            f"| {name} {bias[name]['label']} | {h['xy_position_error_highpass_rms_m']:.6g} | "
            f"{h['xy_position_correction_second_difference_rms_m']:.6g} | "
            f"{h['velocity_truth_error_highpass_rms_mps']:.6g} | {t['translation_rpe_rmse_m']:.6g} | "
            f"{t['ate_position_rmse_m']:.6g} | {t['acc_bias_rmse_mps2']:.6g} |"
        )
    lines += [
        "",
        "B2 固定全部静止 Bias 虽降低高频，却因静止 bg 不准而造成明显低频/姿态恶化；它不是最终方案。B3 是 GT oracle，仅证明正确 Bias 可显著压低速度高频。B4 只开放 ba 最差，B5 固定 ba、开放 bg 最好，因此问题集中在 ba 与平移/速度的弱可观耦合。",
        "",
        "## 窗口消融",
        "",
        "| 模式 | N | XY 高频 | correction 二阶差分 | velocity 高频 | 平移 RPE | 旋转 RPE | ATE | ba RW 标准化 RMS | ms/solve |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for mode in ("normal", "fixed_ba"):
        for window in (2, 3, 5, 10):
            item = windows[(mode, window)]
            h, t, b, solver = item["high_frequency_metrics"], item["truth_metrics"], item["bias_diagnostics"], item["solver"]
            lines.append(
                f"| {mode} | {window} | {h['xy_position_error_highpass_rms_m']:.6g} | "
                f"{h['xy_position_correction_second_difference_rms_m']:.6g} | "
                f"{h['velocity_truth_error_highpass_rms_mps']:.6g} | "
                f"{t['translation_rpe_rmse_m']:.6g} | {t['rotation_rpe_rmse_rad']:.6g} | "
                f"{t['ate_position_rmse_m']:.6g} | {b['ba_increment_standardized_component_rms']:.3g} | "
                f"{solver['mean_runtime_ms_per_solve']:.1f} |"
            )
    lines += [
        "",
        "normal N=10 确实达到两项 20% 高频改善，而且没有牺牲 RPE/ATE；但 ba 的标准化 increment 只从 85.8 降至 84.5，远离正确 RW 的 O(1) 量级。固定 ba 的诊断模式从 N=2 到 N=10 只改善 14.5%/12.7%，说明窗口长度主要是在缓和错误的 Bias 自由度，而非独立消除视觉/IMU 抖动。",
        "",
        f"计算开销从 normal N=2 的 {windows[('normal',2)]['solver']['mean_runtime_ms_per_solve']:.1f} ms/solve 增至 N=10 的 {windows[('normal',10)]['solver']['mean_runtime_ms_per_solve']:.1f} ms/solve。全部 8 组均 209/209 收敛且无 NaN/Inf；normal prior rank=15，fixed-ba rank=12。",
        "",
        "## 最终因果判断",
        "",
        "- **不是 A：Sampling covariance 不是主要抖动问题。** 它通过统计验证，但 N=2 抖动反而约增加 10%。",
        "- **支持 B：Bias 过度活跃是主要问题。** 固定 ba 带来的 52%-69% 高频改善远大于单纯加窗；B1 ba increment 约为理论 RW 的 85.7 倍。",
        "- **不支持把 C 单独作为主因。** N=10 有约 20% 改善，但 N=3/5 远未达标，且 Bias 仍异常；窗口是次级缓解。",
        "- **D 尚未充分证明。** 固定 ba、N=10 后仍有残余高频误差，但本片段不含转弯/停车，不能据此宣布视觉统计模型是主限制。",
        "- **最终为 E 的非加性组合，主次顺序为 Bias > 窗口 > Sampling covariance 对观感的影响。** Sampling 修复贡献的是统计正确性，不是正向平滑收益。",
        "",
        "## 下一步",
        "",
        "1. 暂不跑完整 12 序列，也不把 N=10 提升为生产默认；其 4.3 s/solve 且 Bias gate 失败。",
        "2. 下一项受控实验应只改 ba 参数化：在窗口内使用一个共享 ba，或低频更新的分段常量 ba；保持 Sampling-aware P、四个 sigma、视觉因子和 LM 不变。这样保留在线估计能力，同时移除每帧 ba 吸收白噪声/速度误差的自由度。",
        "3. 共享-ba 通过后，再对 N=2/3/5 做最短窗口复验；只有满足相同门槛才进入包含首个转弯/停车的短片段。",
        "4. GT Bias 继续只作离线评价，不进入生产 loss；B5 也只作为诊断参考，不能当最终算法。",
        "",
        "## 测试与限制",
        "",
        "- `test_two_state_vio.py` 与 PyPose translation accessor 回归：7 passed。",
        "- `test_sampling_aware_runtime.py`：3 条真实边全部断言通过，Delta 完全不变，Sampling covariance SPD。",
        "- Gate 1 impulse：6270 项；Gate 3 Monte Carlo：15 case x 5000 realization，全部验收条件通过。",
        "- 扩展 pytest 集合受仓库既有 `jaxtyping/typeguard` 对字符串 shape 注解 `B 2 H W` 的 SyntaxError 阻塞；这发生在测试收集阶段，与本次因子数值测试无关。",
        "- 尚未运行完整 12 序列，也未验证第一次转弯/停车局部峰值。",
        "",
        "## 本任务修改文件",
        "",
        "- `Utility/IMUCSV.py`、`DataLoader/Dataset/GeneralStereoIMU.py`：保留真实原始采样到插值 knot 的线性映射。",
        "- `Module/IMUPreintegration.py`、`Odometry/MACVO.py`：新增默认关闭的 Sampling-aware covariance 模式；Delta 计算不变。",
        "- `Utility/TwoStateVIO.py`、`Module/Optimization/TwoFramePGO/Optimizer.py`：增加 ba/bg 自由度消融开关，并从正规方程和 prior 中严格移除固定自由度。",
        "- `Utility/FixedLagVIO.py`：通用 N 状态离线固定滞后求解与仅旧因子边缘化。",
        "- `Scripts/audit_real_sampling_aware_covariance.py`、`Scripts/test_sampling_aware_runtime.py`、`Scripts/run_sampling_aware_n2_comparison.py`、`Scripts/run_sampling_aware_bias_ablation.py`、`Scripts/run_sampling_aware_window_ablation.py`：关卡审计与回放。",
        "- `Scripts/UnitTest/test_two_state_vio.py`：N=2 等价、三状态链与固定 Bias 回归测试。",
        "",
        "## 产物索引",
        "",
        "- `sampling_map_audit.json`、`sampling_map_per_edge.npz`、`sampling_impulse_test.csv`",
        "- `sampling_aware_covariance_per_edge.csv`、`sampling_aware_covariance_summary.json`",
        "- `sampling_aware_monte_carlo_report_cn.md`、`sampling_aware_mc_summary.json`",
        "- `sampling_covariance_n2_comparison.json`",
        "- `macvio_bias_ablation_report_cn.md`、`macvio_bias_lifecycle_per_frame.csv`",
        "- `macvio_window_metrics.csv`、`macvio_window_final_recommendation.json`",
        "- `window_ablation/frozen_factor_manifest.json` 与每组逐帧/逐 solve 原始 CSV。",
    ]
    destination = OUT / "macvio_sampling_aware_covariance_report_cn.md"
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")
    window_lines = [
        "# MACVIO 最短有效窗口消融报告", "",
        "> 本实验评价 MACVIO 对 GT 的误差，不以 GTSAM 轨迹为目标。", "",
        "## 实验契约", "",
        "N=2/3/5/10 全部读取 `window_ablation/frozen_factor_manifest.json` 指定的同一组 209 条因子。Delta、Sampling-aware P、视觉测量/covariance、Bias RW、LM、Huber、初始 prior 均不变。Bias 模式仅为正常 ba/bg 和 Gate 5 选出的固定 ba/优化 bg 诊断模式。", "",
        "## 结果", "",
        "| 模式 | N | XY 高频 | 二阶差分 | velocity 高频 | 平移 RPE | ATE | ba RW 标准化 RMS | ms/solve |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for mode in ("normal", "fixed_ba"):
        for window in (2, 3, 5, 10):
            item = windows[(mode, window)]
            h, t, b, solver = item["high_frequency_metrics"], item["truth_metrics"], item["bias_diagnostics"], item["solver"]
            window_lines.append(
                f"| {mode} | {window} | {h['xy_position_error_highpass_rms_m']:.6g} | "
                f"{h['xy_position_correction_second_difference_rms_m']:.6g} | "
                f"{h['velocity_truth_error_highpass_rms_mps']:.6g} | {t['translation_rpe_rmse_m']:.6g} | "
                f"{t['ate_position_rmse_m']:.6g} | {b['ba_increment_standardized_component_rms']:.3g} | "
                f"{solver['mean_runtime_ms_per_solve']:.1f} |"
            )
    window_lines += [
        "", "## 门槛判定", "",
        "- normal N=3/5 没有两项高频指标改善 20%。",
        "- normal N=10 的 XY 高频和二阶差分改善 21.1%/20.6%，RPE/ATE 未恶化，但 ba RW 标准化 RMS 仍为 84.5，Bias gate 失败；平均耗时 4.32 s/solve。",
        "- fixed-ba N=10 的两项改善仅 14.5%/12.7%，仍未达到 20%。",
        "- 8 组均 209/209 收敛且无 NaN/Inf；normal prior rank=15，fixed-ba rank=12。", "",
        "## 决策", "",
        "没有窗口满足全部条件，不提升生产默认窗口，也不批准完整 12 序列。窗口是次级缓解；下一步应先验证共享或分段常量 ba 参数化，再重跑 N=2/3/5。", "",
        "逐项判定见 `macvio_window_final_recommendation.json`；逐帧状态、solve cost、迭代数和 prior 统计位于 `window_ablation/*/`。",
    ]
    window_destination = OUT / "macvio_window_ablation_report_cn.md"
    window_destination.write_text("\n".join(window_lines) + "\n", encoding="utf-8")
    print(destination)
    print(window_destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
