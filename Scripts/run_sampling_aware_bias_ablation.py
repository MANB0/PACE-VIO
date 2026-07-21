#!/usr/bin/env python3
"""Gate 5: sampling-aware N=2 accelerometer/gyro bias ablations."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Scripts import freeze_normal_noise_sampling_baseline as baseline  # noqa: E402
from Module.Optimization.TwoFramePGO.Optimizer import TwoFrame_PGO  # noqa: E402
from Odometry.MACVO import MACVO as MACVOImplementation  # noqa: E402


OUT = baseline.DEFAULT_OUTPUT
ABLATION_ROOT = OUT / "bias_ablation"
SIGMA_ACC_W = 0.000386071
SIGMA_GYRO_W = 3.57864e-05
VARIANTS = {
    "B1": {"label": "optimize_ba_bg", "opt_ba": True, "opt_bg": True, "reference": "online"},
    "B2": {"label": "fixed_static_ba_bg", "opt_ba": False, "opt_bg": False, "reference": "static"},
    "B3": {"label": "fixed_gt_ba_bg_oracle", "opt_ba": False, "opt_bg": False, "reference": "gt"},
    "B4": {"label": "optimize_ba_fixed_static_bg", "opt_ba": True, "opt_bg": False, "reference": "static"},
    "B5": {"label": "fixed_static_ba_optimize_bg", "opt_ba": False, "opt_bg": True, "reference": "static"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=sorted(VARIANTS))
    parser.add_argument("--only", nargs="+", choices=sorted(VARIANTS))
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _variant_dir(name: str) -> Path:
    return ABLATION_ROOT / f"{name}_{VARIANTS[name]['label']}"


def _frame_truth_bias() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ref = pd.read_csv(baseline.DATASET_DIR / "ref_pose.csv").iloc[: baseline.FRAME_LIMIT]
    frame_time = ref["timestamp"].to_numpy(np.int64)
    truth = pd.read_csv(baseline.DATASET_DIR / "imu_truth_decomposition.csv")
    imu_time = truth["timestamp"].to_numpy(np.int64)
    ba_flu = truth[["acc_bias_x", "acc_bias_y", "acc_bias_z"]].to_numpy(np.float64)
    bg_flu = truth[["gyro_bias_x", "gyro_bias_y", "gyro_bias_z"]].to_numpy(np.float64)
    ba = baseline.interpolate_rows(imu_time, ba_flu, frame_time) @ baseline.FLU_TO_NED.T
    bg = baseline.interpolate_rows(imu_time, bg_flu, frame_time) @ baseline.FLU_TO_NED.T
    return frame_time, ba, bg


def _static_bias() -> tuple[np.ndarray, np.ndarray]:
    source = OUT / "sampling_aware_n2/sampling_n2_state_per_frame.csv"
    if not source.exists():
        raise FileNotFoundError("Gate 4 sampling-aware N=2 output is required before Gate 5")
    frame = pd.read_csv(source)
    row = frame.loc[frame["frame"] == baseline.FIRST_VALID_FRAME].iloc[0]
    ba = row[["ba_est_x", "ba_est_y", "ba_est_z"]].to_numpy(np.float64)
    bg = row[["bg_est_x", "bg_est_y", "bg_est_z"]].to_numpy(np.float64)
    return ba, bg


def _install_bias_reference_control(variant: str) -> None:
    spec = VARIANTS[variant]
    if spec["reference"] == "online":
        return
    _, gt_ba, gt_bg = _frame_truth_bias()
    static_ba, static_bg = _static_bias()
    original_commit = MACVOImplementation._commit_previous_backend_result
    original_optimize = baseline.replay.ORIGINAL_OPTIMIZE

    def reference(frame_idx: int, kind: str) -> torch.Tensor:
        if spec["reference"] == "gt":
            values = gt_ba if kind == "ba" else gt_bg
            return torch.from_numpy(values[frame_idx]).float()
        values = static_ba if kind == "ba" else static_bg
        return torch.from_numpy(values).float()

    def controlled_commit(self, frame0, frame1):
        result = original_commit(self, frame0, frame1)
        frame_i = int(frame0.frame_idx)
        if not bool(spec["opt_ba"]):
            self._imu_acc_bias = reference(frame_i, "ba")
        if not bool(spec["opt_bg"]):
            self._imu_gyro_bias = reference(frame_i, "bg")
        return result

    def controlled_optimize(context, graph_data):
        frame_i = int(graph_data.from_idx.reshape(-1)[0].item())
        frame_j = int(graph_data.frame_idx.reshape(-1)[0].item())
        if not bool(spec["opt_ba"]):
            graph_data.imu_vio_prev_acc_bias = reference(frame_i, "ba")
            graph_data.imu_vio_curr_acc_bias_init = reference(frame_j, "ba")
        if not bool(spec["opt_bg"]):
            graph_data.imu_vio_prev_gyro_bias = reference(frame_i, "bg")
            graph_data.imu_vio_curr_gyro_bias_init = reference(frame_j, "bg")
        return original_optimize(context, graph_data)

    MACVOImplementation._commit_previous_backend_result = controlled_commit
    baseline.replay.ORIGINAL_OPTIMIZE = controlled_optimize


def run_variant(variant: str, *, force: bool) -> int:
    spec = VARIANTS[variant]
    run_out = _variant_dir(variant)
    if run_out.exists():
        if not force:
            raise FileExistsError(f"{run_out} exists; pass --force")
        if ABLATION_ROOT.resolve() not in run_out.resolve().parents:
            raise RuntimeError("refusing to clear output outside bias ablation root")
        shutil.rmtree(run_out)
    run_out.mkdir(parents=True)

    config = yaml.safe_load(baseline.ODOM_SOURCE.read_text(encoding="utf-8"))
    args = config["Odometry"]["args"]
    args["two_state_covariance_mode"] = "sampling_aware"
    optimizer_args = config["Odometry"]["optimizer"]["args"]
    optimizer_args["two_state_optimize_acc_bias"] = bool(spec["opt_ba"])
    optimizer_args["two_state_optimize_gyro_bias"] = bool(spec["opt_bg"])
    odom_source = run_out / "odometry_source.yaml"
    odom_source.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    replay = baseline.replay
    replay.OUT = run_out
    replay.FRAME_LIMIT = baseline.FRAME_LIMIT
    replay.ITERATION_LIMIT = baseline.ITERATION_LIMIT
    replay.SOURCE_CACHE = baseline.SOURCE_CACHE
    replay.PREFIX_CACHE = run_out / "cache_rectangle_300"
    replay.SOURCE_RESULT = baseline.SOURCE_RESULT
    replay.ODOM_SOURCE = odom_source
    replay.GT_PATH = baseline.DATASET_DIR / "ref_pose.csv"
    replay.RUN_RESULT_ROOT = run_out / "run_result"
    replay.TRACE_PATH = run_out / "factor_per_edge.csv"
    replay.ODOM_CONFIG = run_out / "odometry.yaml"
    replay.DATA_CONFIG = run_out / "data.yaml"
    replay.TRACE_FIELDS = baseline.TRACE_FIELDS
    replay.audited_solve = baseline.audited_solve
    replay.summarize_run = lambda: baseline.summarize_baseline(run_out)
    replay.GATE_BY_EDGE.clear()
    _install_bias_reference_control(variant)
    status = replay.main()

    trace = pd.read_csv(run_out / "factor_per_edge.csv")
    if not bool(spec["opt_ba"]):
        update = trace[[f"state_i_update_{index}" for index in range(9, 12)]].to_numpy()
        update_j = trace[[f"state_j_update_{index}" for index in range(9, 12)]].to_numpy()
        if float(np.max(np.abs(np.concatenate([update, update_j], axis=0)))) > 1e-12:
            raise AssertionError(f"{variant}: fixed accelerometer bias changed")
    if not bool(spec["opt_bg"]):
        update = trace[[f"state_i_update_{index}" for index in range(12, 15)]].to_numpy()
        update_j = trace[[f"state_j_update_{index}" for index in range(12, 15)]].to_numpy()
        if float(np.max(np.abs(np.concatenate([update, update_j], axis=0)))) > 1e-12:
            raise AssertionError(f"{variant}: fixed gyro bias changed")
    if spec["reference"] == "gt":
        state = pd.read_csv(run_out / "baseline_state_per_frame.csv").query("vio_active == True")
        errors = state[[f"{sensor}_error_{axis}" for sensor in ("ba", "bg") for axis in "xyz"]].to_numpy()
        if float(np.max(np.abs(errors))) > 1e-6:
            raise AssertionError(f"{variant}: GT bias oracle was not held at truth")

    summary_path = run_out / "baseline_normal_noise_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary.update(
        gate=5,
        variant=variant,
        label=spec["label"],
        covariance_mode="sampling_aware",
        optimize_acc_bias=bool(spec["opt_ba"]),
        optimize_gyro_bias=bool(spec["opt_bg"]),
        bias_reference=spec["reference"],
    )
    (run_out / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return status


def _pearson(a: np.ndarray, b: np.ndarray) -> float | None:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    finite = np.isfinite(a) & np.isfinite(b)
    a, b = a[finite], b[finite]
    if a.size < 3 or np.std(a) < 1e-15 or np.std(b) < 1e-15:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def _edge_noise_means(frame_time: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    truth = pd.read_csv(baseline.DATASET_DIR / "imu_truth_decomposition.csv")
    time = truth["timestamp"].to_numpy(np.int64)
    acc = truth[["acc_noise_x", "acc_noise_y", "acc_noise_z"]].to_numpy(np.float64)
    gyro = truth[["gyro_noise_x", "gyro_noise_y", "gyro_noise_z"]].to_numpy(np.float64)
    acc_means, gyro_means = [], []
    for frame_i in range(baseline.FIRST_VALID_FRAME, baseline.FRAME_LIMIT - 1):
        mask = (time >= frame_time[frame_i]) & (time <= frame_time[frame_i + 1])
        acc_means.append(acc[mask].mean(axis=0) @ baseline.FLU_TO_NED.T)
        gyro_means.append(gyro[mask].mean(axis=0) @ baseline.FLU_TO_NED.T)
    return np.asarray(acc_means), np.asarray(gyro_means)


def _lifecycle_rows(variant: str, trace: pd.DataFrame, state: pd.DataFrame) -> list[dict]:
    rows: list[dict] = []
    state_by_frame = state.set_index("frame")
    previous = None
    for _, edge in trace.iterrows():
        frame = int(edge["frame_i"])
        row = {"variant": variant, "frame": frame, "next_frame": int(edge["frame_j"])}
        for sensor in ("ba", "bg"):
            stages = {
                "before_current_window": [edge[f"{sensor}_i_before_{a}"] for a in "xyz"],
                "after_current_window": [edge[f"{sensor}_i_after_{a}"] for a in "xyz"],
                "as_previous_endpoint": (
                    [previous[f"{sensor}_j_after_{a}"] for a in "xyz"]
                    if previous is not None else [math.nan] * 3
                ),
                "as_next_start_initial": [edge[f"{sensor}_i_before_{a}"] for a in "xyz"],
                "as_next_start_optimized": [edge[f"{sensor}_i_after_{a}"] for a in "xyz"],
            }
            for stage, values in stages.items():
                for axis, value in zip("xyz", values):
                    row[f"{sensor}_{stage}_{axis}"] = float(value)
            update = np.asarray(stages["after_current_window"]) - np.asarray(stages["before_current_window"])
            continuity = np.asarray(stages["as_next_start_initial"]) - np.asarray(stages["as_previous_endpoint"])
            row[f"{sensor}_same_state_update_norm"] = float(np.linalg.norm(update))
            row[f"{sensor}_endpoint_to_next_start_norm"] = float(np.linalg.norm(continuity))
            for axis in "xyz":
                row[f"{sensor}_truth_{axis}"] = float(state_by_frame.loc[frame, f"{sensor}_gt_{axis}"])
                row[f"{sensor}_error_{axis}"] = float(state_by_frame.loc[frame, f"{sensor}_error_{axis}"])
        rows.append(row)
        previous = edge
    return rows


def aggregate() -> None:
    frame_time, _, _ = _frame_truth_bias()
    acc_noise, gyro_noise = _edge_noise_means(frame_time)
    all_lifecycle: list[dict] = []
    result: dict[str, dict] = {}
    for variant, spec in VARIANTS.items():
        run_out = _variant_dir(variant)
        summary = json.loads((run_out / "summary.json").read_text(encoding="utf-8"))
        trace = pd.read_csv(run_out / "factor_per_edge.csv")
        state = pd.read_csv(run_out / "baseline_state_per_frame.csv")
        active = state.query("vio_active == True").copy()
        all_lifecycle.extend(_lifecycle_rows(variant, trace, state))

        ba = active[[f"ba_est_{a}" for a in "xyz"]].to_numpy(np.float64)
        bg = active[[f"bg_est_{a}" for a in "xyz"]].to_numpy(np.float64)
        ba_gt = active[[f"ba_gt_{a}" for a in "xyz"]].to_numpy(np.float64)
        bg_gt = active[[f"bg_gt_{a}" for a in "xyz"]].to_numpy(np.float64)
        dt = np.diff(active["timestamp_ns"].to_numpy(np.float64)) * 1e-9
        dba, dbg = np.diff(ba, axis=0), np.diff(bg, axis=0)
        dba_gt, dbg_gt = np.diff(ba_gt, axis=0), np.diff(bg_gt, axis=0)
        norm_dba = dba / (SIGMA_ACC_W * np.sqrt(dt)[:, None])
        norm_dbg = dbg / (SIGMA_GYRO_W * np.sqrt(dt)[:, None])
        ba_update = trace[[f"ba_i_after_{a}" for a in "xyz"]].to_numpy() - trace[[f"ba_i_before_{a}" for a in "xyz"]].to_numpy()
        bg_update = trace[[f"bg_i_after_{a}" for a in "xyz"]].to_numpy() - trace[[f"bg_i_before_{a}" for a in "xyz"]].to_numpy()
        velocity_update = trace[[f"velocity_i_after_{a}" for a in "xyz"]].to_numpy() - trace[[f"velocity_i_before_{a}" for a in "xyz"]].to_numpy()
        xy_update = trace[["state_i_update_0", "state_i_update_1"]].to_numpy()
        visual_t = trace[[f"pose_correction_after_{i}" for i in range(3)]].to_numpy()
        visual_r = trace[[f"pose_correction_after_{i}" for i in range(3, 6)]].to_numpy()

        result[variant] = {
            "label": spec["label"],
            "truth_metrics": summary["truth_metrics_valid_frames"],
            "high_frequency_metrics": summary["high_frequency_metrics_valid_frames"],
            "solver": summary["solver"],
            "factor_cost_statistics": summary["factor_cost_statistics"],
            "bias_diagnostics": {
                "ba_increment_absolute_rms": float(np.sqrt(np.mean(dba * dba))),
                "bg_increment_absolute_rms": float(np.sqrt(np.mean(dbg * dbg))),
                "ba_increment_standardized_component_rms": float(np.sqrt(np.mean(norm_dba * norm_dba))),
                "bg_increment_standardized_component_rms": float(np.sqrt(np.mean(norm_dbg * norm_dbg))),
                "ba_increment_energy_over_truth": float(np.sum(dba * dba) / max(np.sum(dba_gt * dba_gt), 1e-30)),
                "bg_increment_energy_over_truth": float(np.sum(dbg * dbg) / max(np.sum(dbg_gt * dbg_gt), 1e-30)),
                "ba_update_vs_acc_white_noise_correlation": _pearson(ba_update, acc_noise),
                "bg_update_vs_gyro_white_noise_correlation": _pearson(bg_update, gyro_noise),
                "ba_update_norm_vs_velocity_correction_norm_correlation": _pearson(np.linalg.norm(ba_update, axis=1), np.linalg.norm(velocity_update, axis=1)),
                "bg_update_norm_vs_velocity_correction_norm_correlation": _pearson(np.linalg.norm(bg_update, axis=1), np.linalg.norm(velocity_update, axis=1)),
                "ba_update_norm_vs_xy_pose_correction_norm_correlation": _pearson(np.linalg.norm(ba_update, axis=1), np.linalg.norm(xy_update, axis=1)),
                "bg_update_norm_vs_xy_pose_correction_norm_correlation": _pearson(np.linalg.norm(bg_update, axis=1), np.linalg.norm(xy_update, axis=1)),
                "ba_update_vs_visual_translation_residual_correlation": _pearson(ba_update, visual_t),
                "bg_update_vs_visual_rotation_residual_correlation": _pearson(bg_update, visual_r),
            },
        }

    pd.DataFrame(all_lifecycle).to_csv(
        OUT / "macvio_bias_lifecycle_per_frame.csv", index=False
    )
    b1 = result["B1"]["high_frequency_metrics"]
    for name, values in result.items():
        values["high_frequency_change_vs_B1_percent"] = {
            key: (float(value) / float(b1[key]) - 1.0) * 100.0
            for key, value in values["high_frequency_metrics"].items()
            if isinstance(value, (int, float)) and key in b1 and float(b1[key]) != 0.0
        }
    (OUT / "macvio_bias_ablation_summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    xy_hp = {name: values["high_frequency_metrics"]["xy_position_error_highpass_rms_m"] for name, values in result.items()}
    pos_dd = {name: values["high_frequency_metrics"]["xy_position_correction_second_difference_rms_m"] for name, values in result.items()}
    lines = [
        "# MACVIO Bias 专项消融报告",
        "",
        "> 本任务的目标是降低 MACVIO 对 GT 的高频误差，不是让 MACVIO 轨迹或 Bias 接近 GTSAM。",
        "",
        "所有实验固定为 N=2、Sampling-aware measurement covariance、相同视觉 sidecar、Bias RW、LM、Huber 与 prior。GT Bias 只用于 B3 离线 oracle。",
        "",
        "## 核心结果",
        "",
        "| 模式 | XY 高频误差 RMS (m) | XY correction 二阶差分 RMS (m) | 平移 RPE (m) | 速度 RMSE (m/s) | ba RMSE | bg RMSE |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, values in result.items():
        t = values["truth_metrics"]
        h = values["high_frequency_metrics"]
        lines.append(
            f"| {name} {values['label']} | {h['xy_position_error_highpass_rms_m']:.6g} | "
            f"{h['xy_position_correction_second_difference_rms_m']:.6g} | "
            f"{t['translation_rpe_rmse_m']:.6g} | {t['velocity_rmse_mps']:.6g} | "
            f"{t['acc_bias_rmse_mps2']:.6g} | {t['gyro_bias_rmse_radps']:.6g} |"
        )
    best_fixed = min(("B2", "B3"), key=lambda name: xy_hp[name])
    lines += [
        "",
        "## 判读",
        "",
        f"B1 的 XY 高频误差为 {xy_hp['B1']:.6g} m。固定 Bias 中最低的是 {best_fixed}，为 {xy_hp[best_fixed]:.6g} m。",
        f"只优化 ba 的 B4 为 {xy_hp['B4']:.6g} m；只优化 bg 的 B5 为 {xy_hp['B5']:.6g} m。",
        "",
        "Bias 是否为主要来源，按固定 Bias 相对 B1 是否显著降低至少两项高频指标判断；完整数值、相关性和标准化 increment 见 JSON 与 lifecycle CSV。",
    ]
    (OUT / "macvio_bias_ablation_report_cn.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps({"xy_highpass": xy_hp, "position_second_difference": pos_dd}, indent=2))


def orchestrate(force: bool, variants: list[str] | None = None) -> int:
    ABLATION_ROOT.mkdir(parents=True, exist_ok=True)
    for variant in (variants or list(VARIANTS)):
        run_out = _variant_dir(variant)
        if (run_out / "summary.json").exists() and not force:
            print(f"[{variant}] reusing completed output")
            continue
        print(f"[{variant}] starting {VARIANTS[variant]['label']}", flush=True)
        log_out = ABLATION_ROOT / f"{variant}_stdout.log"
        log_err = ABLATION_ROOT / f"{variant}_stderr.log"
        with log_out.open("w", encoding="utf-8") as stdout, log_err.open("w", encoding="utf-8") as stderr:
            command = [sys.executable, str(Path(__file__).resolve()), "--variant", variant]
            if force:
                command.append("--force")
            completed = subprocess.run(command, cwd=ROOT, stdout=stdout, stderr=stderr)
        if completed.returncode != 0:
            raise RuntimeError(f"{variant} failed; inspect {log_err}")
        print(f"[{variant}] complete", flush=True)
    aggregate()
    return 0


def main() -> int:
    args = parse_args()
    if args.variant:
        return run_variant(args.variant, force=args.force)
    return orchestrate(args.force, variants=args.only)


if __name__ == "__main__":
    raise SystemExit(main())
