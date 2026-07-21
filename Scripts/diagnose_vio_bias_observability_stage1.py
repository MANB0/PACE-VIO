#!/usr/bin/env python3
"""Run a real-scale one-edge VIO bias observability isolation.

This diagnostic intentionally does not run MACVO or change the production
estimator. It reuses the production preintegrator and IMU residual while
exposing different endpoint state subsets to determine why the two-frame VIO
bias states remain zero on noisy sequences.
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pypose as pp
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Utility.IMUKinematics import (
    vio_bias_random_walk_residual,
    vio_preintegrated_imu_residual,
)


def _load_preintegrate_imu():
    module_path = PROJECT_ROOT / "Module" / "IMUPreintegration.py"
    spec = importlib.util.spec_from_file_location("stage1_imu_preintegration", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load IMU preintegration module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.preintegrate_imu


preintegrate_imu = _load_preintegrate_imu()


DT_S = 1.0 / 30.0
TIME_NS = torch.tensor([0, 10_000_000, 20_000_000, 30_000_000, 33_333_333], dtype=torch.long)
GRAVITY_M_S2 = 9.8

# These are deterministic accumulated-bias probes, not additional white-noise
# samples. Their scale is representative of several standard deviations of the
# configured 100 Hz random walk over a tens-of-seconds sequence.
TRUE_ACC_BIAS = torch.tensor([0.0040, -0.0030, 0.0020], dtype=torch.float64)
TRUE_GYRO_BIAS = torch.tensor([0.00040, -0.00030, 0.00020], dtype=torch.float64)

SIGMA_ACC = 0.0141258
SIGMA_GYRO = 0.00182898
SIGMA_ACC_W = 0.000386071
SIGMA_GYRO_W = 3.57864e-05

VISUAL_TRANSLATION_STD_M = 0.08
VISUAL_ROTATION_STD_RAD = 0.035
RELAXED_RANDOM_WALK_SCALE = 1e12

CASES = (
    "current_two_frame",
    "relaxed_random_walk",
    "fixed_pose_terminal_states",
    "fixed_pose_velocity_start_bias",
)


@dataclass(frozen=True)
class SyntheticBiasProblem:
    delta_R: pp.LieTensor
    delta_v: torch.Tensor
    delta_p: torch.Tensor
    covariance: torch.Tensor
    bias_jacobian: torch.Tensor
    bias_rw_covariance: torch.Tensor
    linearized_acc_bias: torch.Tensor
    linearized_gyro_bias: torch.Tensor


@dataclass(frozen=True)
class CaseSpec:
    name: str
    optimize_pose: bool
    optimize_velocity: bool
    optimize_terminal_bias: bool
    optimize_start_bias: bool
    random_walk_cov_scale: float
    use_visual_pose_prior: bool


def build_real_scale_problem() -> SyntheticBiasProblem:
    acc = torch.tensor([0.0, 0.0, -GRAVITY_M_S2], dtype=torch.float64) + TRUE_ACC_BIAS
    gyro = TRUE_GYRO_BIAS.clone()
    acc_samples = acc.repeat(TIME_NS.numel(), 1).float()
    gyro_samples = gyro.repeat(TIME_NS.numel(), 1).float()

    result = preintegrate_imu(
        time_ns=TIME_NS,
        acc=acc_samples,
        gyro=gyro_samples,
        R0_world=pp.identity_SO3(dtype=torch.float64),
        gravity=GRAVITY_M_S2,
        sigma_acc=SIGMA_ACC,
        sigma_gyro=SIGMA_GYRO,
        sigma_acc_w=SIGMA_ACC_W,
        sigma_gyro_w=SIGMA_GYRO_W,
        acc_bias=torch.zeros(3, dtype=torch.float32),
        gyro_bias=torch.zeros(3, dtype=torch.float32),
    )
    assert result.bias_jacobian is not None
    assert result.bias_rw_cov is not None
    assert result.linearized_acc_bias is not None
    assert result.linearized_gyro_bias is not None

    return SyntheticBiasProblem(
        delta_R=pp.SO3(result.delta_R.tensor().double()),
        delta_v=result.delta_v.double(),
        delta_p=result.delta_p.double(),
        covariance=result.cov.double(),
        bias_jacobian=result.bias_jacobian.double(),
        bias_rw_covariance=result.bias_rw_cov.double(),
        linearized_acc_bias=result.linearized_acc_bias.double(),
        linearized_gyro_bias=result.linearized_gyro_bias.double(),
    )


def _symmetric_information(covariance: torch.Tensor) -> torch.Tensor:
    covariance = 0.5 * (covariance + covariance.mT)
    return torch.linalg.pinv(covariance, hermitian=True)


def _pose_from_delta(delta: torch.Tensor) -> pp.LieTensor:
    return pp.se3(delta.reshape(1, 6)).Exp()


def _imu_residual(
    problem: SyntheticBiasProblem,
    pose_delta: torch.Tensor,
    velocity_j: torch.Tensor,
    start_acc_bias: torch.Tensor,
    start_gyro_bias: torch.Tensor,
    terminal_acc_bias: torch.Tensor,
    terminal_gyro_bias: torch.Tensor,
) -> torch.Tensor:
    return vio_preintegrated_imu_residual(
        from_pose=pp.identity_SE3(1, dtype=torch.float64),
        to_pose=_pose_from_delta(pose_delta),
        prev_velocity_world=torch.zeros(3, dtype=torch.float64),
        curr_velocity_world=velocity_j,
        delta_R=problem.delta_R,
        delta_v=problem.delta_v,
        delta_p=problem.delta_p,
        dt_total=DT_S,
        prev_acc_bias=start_acc_bias,
        prev_gyro_bias=start_gyro_bias,
        curr_acc_bias=terminal_acc_bias,
        curr_gyro_bias=terminal_gyro_bias,
        linearized_acc_bias=problem.linearized_acc_bias,
        linearized_gyro_bias=problem.linearized_gyro_bias,
        bias_jacobian=problem.bias_jacobian,
    ).double()


def _energy_terms(
    problem: SyntheticBiasProblem,
    pose_delta: torch.Tensor,
    velocity_j: torch.Tensor,
    start_acc_bias: torch.Tensor,
    start_gyro_bias: torch.Tensor,
    terminal_acc_bias: torch.Tensor,
    terminal_gyro_bias: torch.Tensor,
    *,
    random_walk_cov_scale: float,
    use_visual_pose_prior: bool,
) -> dict[str, torch.Tensor]:
    residual = _imu_residual(
        problem,
        pose_delta,
        velocity_j,
        start_acc_bias,
        start_gyro_bias,
        terminal_acc_bias,
        terminal_gyro_bias,
    )
    r_flat = residual.reshape(9)
    imu_information = _symmetric_information(problem.covariance)
    imu_energy = r_flat @ imu_information @ r_flat

    rw_residual = vio_bias_random_walk_residual(
        prev_acc_bias=start_acc_bias,
        prev_gyro_bias=start_gyro_bias,
        curr_acc_bias=terminal_acc_bias,
        curr_gyro_bias=terminal_gyro_bias,
    ).double().reshape(6)
    rw_information = _symmetric_information(problem.bias_rw_covariance * float(random_walk_cov_scale))
    rw_energy = rw_residual @ rw_information @ rw_residual

    if use_visual_pose_prior:
        visual_energy = (
            pose_delta[0:3].square().sum() / (VISUAL_TRANSLATION_STD_M ** 2)
            + pose_delta[3:6].square().sum() / (VISUAL_ROTATION_STD_RAD ** 2)
        )
    else:
        visual_energy = pose_delta.new_zeros(())

    return {
        "total": imu_energy + rw_energy + visual_energy,
        "imu": imu_energy,
        "random_walk": rw_energy,
        "visual": visual_energy,
        "residual": residual,
    }


def _gradient_norms(problem: SyntheticBiasProblem) -> tuple[float, float]:
    pose_delta = torch.zeros(6, dtype=torch.float64)
    velocity_j = torch.zeros(3, dtype=torch.float64)
    start_acc = torch.zeros(3, dtype=torch.float64, requires_grad=True)
    start_gyro = torch.zeros(3, dtype=torch.float64, requires_grad=True)
    terminal_acc = torch.zeros(3, dtype=torch.float64, requires_grad=True)
    terminal_gyro = torch.zeros(3, dtype=torch.float64, requires_grad=True)

    residual = _imu_residual(
        problem,
        pose_delta,
        velocity_j,
        start_acc,
        start_gyro,
        terminal_acc,
        terminal_gyro,
    ).reshape(9)
    energy = residual @ _symmetric_information(problem.covariance) @ residual
    grads = torch.autograd.grad(
        energy,
        (start_acc, start_gyro, terminal_acc, terminal_gyro),
        allow_unused=True,
    )

    def norm_or_zero(items: tuple[torch.Tensor | None, ...]) -> float:
        tensors = [item.reshape(-1) for item in items if item is not None]
        if not tensors:
            return 0.0
        return float(torch.cat(tensors).norm().detach().cpu().item())

    return norm_or_zero((grads[0], grads[1])), norm_or_zero((grads[2], grads[3]))


def _case_spec(name: str) -> CaseSpec:
    if name == "current_two_frame":
        return CaseSpec(name, True, True, True, False, 1.0, True)
    if name == "relaxed_random_walk":
        return CaseSpec(name, True, True, True, False, RELAXED_RANDOM_WALK_SCALE, True)
    if name == "fixed_pose_terminal_states":
        return CaseSpec(name, False, True, True, False, 1.0, False)
    if name == "fixed_pose_velocity_start_bias":
        return CaseSpec(name, False, False, False, True, 1.0, False)
    raise ValueError(f"Unknown Stage 1 case: {name}")


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    denom = float(a.norm().item() * b.norm().item())
    if denom <= 0.0:
        return float("nan")
    return float((a @ b).item() / denom)


def _relative_error(estimate: torch.Tensor, truth: torch.Tensor) -> float:
    denom = max(float(truth.norm().item()), 1e-15)
    return float((estimate - truth).norm().item() / denom)


def _run_case(problem: SyntheticBiasProblem, spec: CaseSpec) -> dict[str, float | int | str]:
    pose_parameter = torch.nn.Parameter(torch.zeros(6, dtype=torch.float64)) if spec.optimize_pose else None
    velocity_parameter = torch.nn.Parameter(torch.zeros(3, dtype=torch.float64)) if spec.optimize_velocity else None
    terminal_acc_parameter = (
        torch.nn.Parameter(torch.zeros(3, dtype=torch.float64)) if spec.optimize_terminal_bias else None
    )
    terminal_gyro_parameter = (
        torch.nn.Parameter(torch.zeros(3, dtype=torch.float64)) if spec.optimize_terminal_bias else None
    )
    start_acc_parameter = torch.nn.Parameter(torch.zeros(3, dtype=torch.float64)) if spec.optimize_start_bias else None
    start_gyro_parameter = torch.nn.Parameter(torch.zeros(3, dtype=torch.float64)) if spec.optimize_start_bias else None

    parameters = [
        item
        for item in (
            pose_parameter,
            velocity_parameter,
            terminal_acc_parameter,
            terminal_gyro_parameter,
            start_acc_parameter,
            start_gyro_parameter,
        )
        if item is not None
    ]

    def state() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        pose = pose_parameter if pose_parameter is not None else torch.zeros(6, dtype=torch.float64)
        velocity = velocity_parameter if velocity_parameter is not None else torch.zeros(3, dtype=torch.float64)
        start_acc = start_acc_parameter if start_acc_parameter is not None else torch.zeros(3, dtype=torch.float64)
        start_gyro = start_gyro_parameter if start_gyro_parameter is not None else torch.zeros(3, dtype=torch.float64)
        if spec.optimize_start_bias:
            terminal_acc = start_acc
            terminal_gyro = start_gyro
        else:
            terminal_acc = (
                terminal_acc_parameter
                if terminal_acc_parameter is not None
                else torch.zeros(3, dtype=torch.float64)
            )
            terminal_gyro = (
                terminal_gyro_parameter
                if terminal_gyro_parameter is not None
                else torch.zeros(3, dtype=torch.float64)
            )
        return pose, velocity, start_acc, start_gyro, terminal_acc, terminal_gyro

    def evaluate() -> dict[str, torch.Tensor]:
        pose, velocity, start_acc, start_gyro, terminal_acc, terminal_gyro = state()
        return _energy_terms(
            problem,
            pose,
            velocity,
            start_acc,
            start_gyro,
            terminal_acc,
            terminal_gyro,
            random_walk_cov_scale=spec.random_walk_cov_scale,
            use_visual_pose_prior=spec.use_visual_pose_prior,
        )

    initial = evaluate()
    initial_residual = initial["residual"].detach().clone()
    initial_total = float(initial["total"].detach().cpu().item())
    initial_imu = float(initial["imu"].detach().cpu().item())

    optimizer = torch.optim.LBFGS(
        parameters,
        lr=1.0,
        max_iter=100,
        tolerance_grad=1e-14,
        tolerance_change=1e-14,
        line_search_fn="strong_wolfe",
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        loss = evaluate()["total"]
        loss.backward()
        return loss

    optimizer.step(closure)
    final = evaluate()
    final_residual = final["residual"].detach().clone()
    pose, velocity, start_acc, start_gyro, terminal_acc, terminal_gyro = [item.detach() for item in state()]
    start_grad_norm, curr_grad_norm = _gradient_norms(problem)
    optimized_bias_role = "start_i" if spec.optimize_start_bias else "terminal_j"
    optimized_bias_grad_norm = start_grad_norm if spec.optimize_start_bias else curr_grad_norm
    optimized_acc_bias = start_acc if spec.optimize_start_bias else terminal_acc
    optimized_gyro_bias = start_gyro if spec.optimize_start_bias else terminal_gyro

    row: dict[str, float | int | str] = {
        "case": spec.name,
        "dt_s": DT_S,
        "num_imu_samples": int(TIME_NS.numel()),
        "random_walk_cov_scale": spec.random_walk_cov_scale,
        "preintegration_cov_min_diag": float(problem.covariance.diagonal().min().item()),
        "preintegration_cov_max_diag": float(problem.covariance.diagonal().max().item()),
        "bias_rw_cov_min_diag": float(problem.bias_rw_covariance.diagonal().min().item()),
        "bias_rw_cov_max_diag": float(problem.bias_rw_covariance.diagonal().max().item()),
        "initial_total_energy": initial_total,
        "final_total_energy": float(final["total"].detach().cpu().item()),
        "initial_imu_energy": initial_imu,
        "final_imu_energy": float(final["imu"].detach().cpu().item()),
        "initial_random_walk_energy": float(initial["random_walk"].detach().cpu().item()),
        "final_random_walk_energy": float(final["random_walk"].detach().cpu().item()),
        "initial_visual_energy": float(initial["visual"].detach().cpu().item()),
        "final_visual_energy": float(final["visual"].detach().cpu().item()),
        "initial_position_residual_norm": float(initial_residual[0].norm().item()),
        "final_position_residual_norm": float(final_residual[0].norm().item()),
        "initial_velocity_residual_norm": float(initial_residual[1].norm().item()),
        "final_velocity_residual_norm": float(final_residual[1].norm().item()),
        "initial_rotation_residual_norm": float(initial_residual[2].norm().item()),
        "final_rotation_residual_norm": float(final_residual[2].norm().item()),
        "initial_start_bias_imu_grad_norm": start_grad_norm,
        "initial_curr_bias_imu_grad_norm": curr_grad_norm,
        "optimized_bias_role": optimized_bias_role,
        "optimized_bias_imu_grad_norm": optimized_bias_grad_norm,
        "optimized_acc_bias_norm": float(optimized_acc_bias.norm().item()),
        "optimized_gyro_bias_norm": float(optimized_gyro_bias.norm().item()),
        "pose_translation_update_norm": float(pose[0:3].norm().item()),
        "pose_rotation_update_norm": float(pose[3:6].norm().item()),
        "velocity_update_norm": float(velocity.norm().item()),
        "estimated_curr_acc_bias_norm": float(terminal_acc.norm().item()),
        "estimated_curr_gyro_bias_norm": float(terminal_gyro.norm().item()),
        "estimated_start_acc_bias_norm": float(start_acc.norm().item()),
        "estimated_start_gyro_bias_norm": float(start_gyro.norm().item()),
        "estimated_start_acc_bias_cosine": _cosine(start_acc, TRUE_ACC_BIAS),
        "estimated_start_gyro_bias_cosine": _cosine(start_gyro, TRUE_GYRO_BIAS),
        "start_acc_bias_relative_error": _relative_error(start_acc, TRUE_ACC_BIAS),
        "start_gyro_bias_relative_error": _relative_error(start_gyro, TRUE_GYRO_BIAS),
    }
    for axis, index in zip("xyz", range(3)):
        row[f"true_acc_bias_{axis}"] = float(TRUE_ACC_BIAS[index].item())
        row[f"true_gyro_bias_{axis}"] = float(TRUE_GYRO_BIAS[index].item())
        row[f"estimated_start_acc_bias_{axis}"] = float(start_acc[index].item())
        row[f"estimated_start_gyro_bias_{axis}"] = float(start_gyro[index].item())
        row[f"estimated_curr_acc_bias_{axis}"] = float(terminal_acc[index].item())
        row[f"estimated_curr_gyro_bias_{axis}"] = float(terminal_gyro[index].item())
    return row


def run_stage1_cases() -> pd.DataFrame:
    torch.manual_seed(0)
    problem = build_real_scale_problem()
    return pd.DataFrame([_run_case(problem, _case_spec(name)) for name in CASES])


def _decision(rows: pd.DataFrame) -> str:
    indexed = rows.set_index("case")
    terminal_cases = indexed.loc[
        ["current_two_frame", "relaxed_random_walk", "fixed_pose_terminal_states"]
    ]
    terminal_unobservable = bool(
        (terminal_cases["initial_curr_bias_imu_grad_norm"] < 1e-12).all()
        and (terminal_cases["estimated_curr_acc_bias_norm"] < 1e-10).all()
        and (terminal_cases["estimated_curr_gyro_bias_norm"] < 1e-10).all()
    )
    start = indexed.loc["fixed_pose_velocity_start_bias"]
    start_recovered = bool(
        start["initial_start_bias_imu_grad_norm"] > 0.0
        and start["estimated_start_acc_bias_cosine"] > 0.99
        and start["estimated_start_gyro_bias_cosine"] > 0.99
        and start["start_acc_bias_relative_error"] < 0.10
        and start["start_gyro_bias_relative_error"] < 0.10
    )
    if terminal_unobservable and start_recovered:
        return "two_frame_start_bias_anchoring_confirmed"
    if not start_recovered:
        return "start_bias_optimizer_or_jacobian_requires_investigation"
    return "terminal_bias_random_walk_weight_requires_investigation"


def write_stage1_report(output_dir: Path) -> tuple[Path, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = run_stage1_cases()
    decision = _decision(rows)

    csv_path = output_dir / "bias_observability_cases.csv"
    report_path = output_dir / "bias_observability_summary_cn.md"
    rows.to_csv(csv_path, index=False, float_format="%.12g")

    indexed = rows.set_index("case")
    current = indexed.loc["current_two_frame"]
    relaxed = indexed.loc["relaxed_random_walk"]
    fixed_pose = indexed.loc["fixed_pose_terminal_states"]
    start = indexed.loc["fixed_pose_velocity_start_bias"]

    report_path.write_text(
        f"""# VIO Bias 可观测性第一阶段诊断

## 结论

诊断判定：`{decision}`。

生产残差对终点 bias `b_j` 的 IMU 梯度为零；将 random-walk 协方差放大到 `{RELAXED_RANDOM_WALK_SCALE:.0e}` 后，终点 bias 仍然不移动。相反，在固定正确 pose 和 velocity、直接优化起点 bias `b_i` 时，生产 bias Jacobian 能恢复注入 bias 的方向并显著降低 IMU 能量。

这说明当前全序列中 bias 始终为零的首要原因不是普通向量参数无法更新，也不是单纯 random-walk 权重过强，而是两帧图固定了真正能够修正 `i -> j` 预积分的起点 bias，只优化了不参与该边 IMU 主残差的终点 bias。

## 实验条件

- 相机间隔：`{DT_S:.9f} s`
- IMU 样本数：`{TIME_NS.numel()}`
- IMU 频率：`100 Hz`
- 注入 acc bias：`{TRUE_ACC_BIAS.tolist()}` m/s^2
- 注入 gyro bias：`{TRUE_GYRO_BIAS.tolist()}` rad/s
- 使用生产 `preintegrate_imu()` 和 `vio_preintegrated_imu_residual()`。

## 四种隔离条件

| case | 被优化 bias | 该 bias 的 IMU 梯度 | acc bias 范数 | gyro bias 范数 | 速度更新 | IMU 能量初值 | IMU 能量终值 |
|---|---|---:|---:|---:|---:|---:|---:|
| current_two_frame | {current.optimized_bias_role} | {current.optimized_bias_imu_grad_norm:.3e} | {current.optimized_acc_bias_norm:.3e} | {current.optimized_gyro_bias_norm:.3e} | {current.velocity_update_norm:.3e} | {current.initial_imu_energy:.6g} | {current.final_imu_energy:.6g} |
| relaxed_random_walk | {relaxed.optimized_bias_role} | {relaxed.optimized_bias_imu_grad_norm:.3e} | {relaxed.optimized_acc_bias_norm:.3e} | {relaxed.optimized_gyro_bias_norm:.3e} | {relaxed.velocity_update_norm:.3e} | {relaxed.initial_imu_energy:.6g} | {relaxed.final_imu_energy:.6g} |
| fixed_pose_terminal_states | {fixed_pose.optimized_bias_role} | {fixed_pose.optimized_bias_imu_grad_norm:.3e} | {fixed_pose.optimized_acc_bias_norm:.3e} | {fixed_pose.optimized_gyro_bias_norm:.3e} | {fixed_pose.velocity_update_norm:.3e} | {fixed_pose.initial_imu_energy:.6g} | {fixed_pose.final_imu_energy:.6g} |
| fixed_pose_velocity_start_bias | {start.optimized_bias_role} | {start.optimized_bias_imu_grad_norm:.3e} | {start.optimized_acc_bias_norm:.3e} | {start.optimized_gyro_bias_norm:.3e} | {start.velocity_update_norm:.3e} | {start.initial_imu_energy:.6g} | {start.final_imu_energy:.6g} |

## 起点 bias 恢复结果

- 起点 bias IMU 梯度范数：`{start.initial_start_bias_imu_grad_norm:.6g}`
- acc bias 方向余弦：`{start.estimated_start_acc_bias_cosine:.9f}`
- gyro bias 方向余弦：`{start.estimated_start_gyro_bias_cosine:.9f}`
- acc bias 相对误差：`{start.start_acc_bias_relative_error:.3%}`
- gyro bias 相对误差：`{start.start_gyro_bias_relative_error:.3%}`
- IMU 能量下降比例：`{start.final_imu_energy / max(start.initial_imu_energy, 1e-30):.6g}`

## 下一阶段门槛

下一步不应继续调 alpha 或只放松 random-walk。需要让多个相邻 edge 共同优化其起点 bias，或者在窗口中把公共/缓慢变化的 bias 状态设为可优化变量，并为窗口首 bias 提供合理而非“固定为零”的先验。进入场景实验前，应先建立最小多帧 synthetic 测试，验证共享 bias 能从零移动并降低所有 edge 的残差。
""",
        encoding="utf-8",
    )
    return csv_path, report_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis_vio_bias_observability_stage1_20260710"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    csv_path, report_path = write_stage1_report(args.output_dir)
    rows = pd.read_csv(csv_path)
    columns = [
        "case",
        "optimized_bias_role",
        "optimized_bias_imu_grad_norm",
        "optimized_acc_bias_norm",
        "optimized_gyro_bias_norm",
        "start_acc_bias_relative_error",
        "start_gyro_bias_relative_error",
        "initial_imu_energy",
        "final_imu_energy",
    ]
    print(rows[columns].to_string(index=False))
    print(f"Wrote CSV:    {csv_path}")
    print(f"Wrote report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
