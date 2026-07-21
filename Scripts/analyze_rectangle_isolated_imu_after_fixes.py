#!/usr/bin/env python3
"""Analyze the corrected rectangle runs with isolated IMU bias and noise."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path

import numpy as np


WORKDIR = Path("/home/admin1/macvo-dev")
if str(WORKDIR) not in sys.path:
    sys.path.insert(0, str(WORKDIR))

from Scripts.evaluate_macvo_relative_metrics import (  # noqa: E402
    RunSpec,
    evaluate_run,
    mat_mul,
    mat_t,
    read_poses,
    rot_angle_rad,
)
from Scripts.plot_static63_gt_macvo import HTML_TEMPLATE, metrics, read_xyz  # noqa: E402


DATA_ROOT = Path(
    "/mnt/e/\u6587\u6863/holoocean/code/recordings/"
    "batch_clear_truth_paths_20260713_static63_variants"
)
NEW_RESULT_ROOT = (
    WORKDIR
    / "Results/rectangle_isolated_imu_after_fixes_20260713/trial_1"
    / "vio_preintegrated_full_imuatt_staticinit_calibrated_biasfix_floor_1e-8"
)
OLD_ESTINIT_ROOT = (
    WORKDIR
    / "Results/static63_cached_imu_fusion_four_configs_20260713/trial_1"
    / "vio_preintegrated_full_imuatt_estinit"
)
OLD_STATICINIT_ROOT = (
    WORKDIR
    / "Results/static63_cached_imu_fusion_staticinit_calibrated_20260713/trial_1"
    / "vio_preintegrated_full_imuatt_staticinit_calibrated"
)
PURE_MACVO_PATH = (
    WORKDIR
    / "Results/visual_factor_cache_static63_unique_source_20260713/trial_1/pure_macvo"
    / "clear_stop_turn_rectangle_truth_normal_noise/poses.csv"
)
IMU_ONLY_ROOT = (
    WORKDIR / "Results/static63_calibrated_imu_only_four_configs_20260713/trajectories"
)
OUTDIR = WORKDIR / "analysis_rectangle_isolated_imu_after_fixes_20260714"

CONFIGS = (
    (
        "bias_no_noise",
        "Bias only",
        "clear_stop_turn_rectangle_truth_bias_no_noise",
    ),
    (
        "noise_no_bias",
        "White noise only",
        "clear_stop_turn_rectangle_truth_noise_no_bias",
    ),
)

METHODS = (
    ("previous_imuatt_estinit", "Previous imuatt_estinit", "#7c3aed", "8 5"),
    ("staticinit_before_bias_fix", "Staticinit before Bias fix", "#64748b", "4 4"),
    (
        "latest_biasfix_floor_1e-8",
        "Bias persistence fix + floor=1e-8",
        "#2563eb",
        "",
    ),
)
AXES = ("x", "y", "z")
FLU_TO_NED_SIGN = np.asarray([1.0, -1.0, -1.0], dtype=np.float64)


def method_path(method: str, scene: str) -> Path:
    if method == "previous_imuatt_estinit":
        return OLD_ESTINIT_ROOT / scene / "poses.csv"
    if method == "staticinit_before_bias_fix":
        return OLD_STATICINIT_ROOT / scene / "poses.csv"
    if method == "latest_biasfix_floor_1e-8":
        return NEW_RESULT_ROOT / scene / "poses.csv"
    if method == "calibrated_imu_only":
        return IMU_ONLY_ROOT / f"{scene}_imu_only_staticinit_calibrated_poses.csv"
    if method == "pure_macvo":
        return PURE_MACVO_PATH
    raise KeyError(method)


def diagnostic_path(method: str, scene: str) -> Path:
    if method == "previous_imuatt_estinit":
        return OLD_ESTINIT_ROOT / scene / "frame_pair_diagnostics.csv"
    if method == "staticinit_before_bias_fix":
        return OLD_STATICINIT_ROOT / scene / "frame_pair_diagnostics.csv"
    if method == "latest_biasfix_floor_1e-8":
        return NEW_RESULT_ROOT / scene / "frame_pair_diagnostics.csv"
    raise KeyError(method)


def xyz(rows: list[tuple[int, float, float, float]]) -> list[list[float]]:
    return [[x, y, z] for _, x, y, z in rows]


def position_errors(gt, est) -> list[float]:
    return [
        math.sqrt(
            (est[index][1] - gt[index][1]) ** 2
            + (est[index][2] - gt[index][2]) ** 2
            + (est[index][3] - gt[index][3]) ** 2
        )
        for index in range(min(len(gt), len(est)))
    ]


def path_length(rows) -> float:
    return sum(
        math.sqrt(
            (second[1] - first[1]) ** 2
            + (second[2] - first[2]) ** 2
            + (second[3] - first[3]) ** 2
        )
        for first, second in zip(rows, rows[1:])
    )


def rpe_translation_1s(gt_path: Path, est_path: Path) -> tuple[float, float, int]:
    gt = read_poses(gt_path)
    est = read_poses(est_path)
    stamps = sorted(set(gt) & set(est))
    errors: list[float] = []
    for index, stamp_i in enumerate(stamps):
        target = stamp_i + 1_000_000_000
        candidates = stamps[index + 1 :]
        if not candidates:
            continue
        stamp_j = min(candidates, key=lambda value: abs(value - target))
        if abs(stamp_j - target) > 20_000_000:
            continue
        g0, g1 = gt[stamp_i], gt[stamp_j]
        e0, e1 = est[stamp_i], est[stamp_j]
        gt_delta = tuple(g1.p[k] - g0.p[k] for k in range(3))
        est_delta = tuple(e1.p[k] - e0.p[k] for k in range(3))
        rotated = tuple(
            sum(
                g0.r[row][middle]
                * sum(e0.r[col][middle] * est_delta[col] for col in range(3))
                for middle in range(3)
            )
            for row in range(3)
        )
        errors.append(math.sqrt(sum((gt_delta[k] - rotated[k]) ** 2 for k in range(3))))
    if not errors:
        return math.nan, math.nan, 0
    return (
        math.sqrt(sum(value * value for value in errors) / len(errors)),
        statistics.median(errors),
        len(errors),
    )


def summarize_trajectory(config: str, scene: str, method: str, path: Path) -> dict[str, object]:
    gt_path = DATA_ROOT / scene / "ref_pose.csv"
    gt_rows = read_xyz(gt_path)
    est_rows = read_xyz(path)
    count = min(len(gt_rows), len(est_rows))
    gt_cut = gt_rows[:count]
    est_cut = est_rows[:count]
    absolute = metrics(gt_cut, est_cut)
    relative = evaluate_run(
        RunSpec("static63", scene, method, "isolated_imu_after_fixes", "1", path, gt_path)
    )
    gt_length = path_length(gt_cut)
    estimated_length = path_length(est_cut)
    rpe_rmse, rpe_median, rpe_pairs = rpe_translation_1s(gt_path, path)
    gt_poses = read_poses(gt_path)
    estimated_poses = read_poses(path)
    common_timestamps = sorted(set(gt_poses) & set(estimated_poses))
    rotation_errors_deg = [
        math.degrees(
            rot_angle_rad(
                mat_mul(
                    mat_t(estimated_poses[timestamp].r),
                    gt_poses[timestamp].r,
                )
            )
        )
        for timestamp in common_timestamps
    ]
    return {
        "config": config,
        "scene": scene,
        "method": method,
        "matched_frames": count,
        "ate_rmse_m": absolute["rmse_m"],
        "ate_mean_m": absolute["mean_m"],
        "ate_final_m": absolute["final_m"],
        "ate_max_m": absolute["max_m"],
        "rpe_translation_rmse_1s_m": rpe_rmse,
        "rpe_translation_median_1s_m": rpe_median,
        "rpe_num_pairs": rpe_pairs,
        "estimated_path_length_m": estimated_length,
        "gt_path_length_m": gt_length,
        "path_length_ratio": estimated_length / gt_length,
        "rotation_rmse_deg": math.sqrt(
            sum(value * value for value in rotation_errors_deg) / len(rotation_errors_deg)
        ),
        "rotation_final_deg": rotation_errors_deg[-1],
        "rotation_max_deg": max(rotation_errors_deg),
        "t_rel_m_per_frame": relative["t_rel_m_per_frame"],
        "r_rel_deg_per_frame": relative["r_rel_deg_per_frame"],
        "t_vel_m_s": relative["t_vel_m_s"],
        "r_vel_deg_s": relative["r_vel_deg_s"],
        "source": str(path),
    }


def read_dicts(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def finite_float(row: dict[str, str], column: str) -> float | None:
    try:
        value = float(row[column])
    except (KeyError, TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def diagnostic_summary() -> list[dict[str, object]]:
    columns = (
        "r_R_whitened_norm",
        "r_p_whitened_norm",
        "r_v_whitened_norm",
        "imu_vio_whitened_norm",
        "imu_vio_cov_trace",
        "imu_vio_weight_trace",
        "imu_vio_weight_diag_min",
        "imu_vio_weight_diag_max",
        "energy_imu_to_visual_ratio",
        "est_velocity_error_norm",
        "update_acc_bias_norm",
        "update_gyro_bias_norm",
    )
    output: list[dict[str, object]] = []
    for config, _, scene in CONFIGS:
        for method, _, _, _ in METHODS:
            path = diagnostic_path(method, scene)
            rows = read_dicts(path)
            for column in columns:
                values = [value for row in rows if (value := finite_float(row, column)) is not None]
                if not values:
                    continue
                ordered = sorted(values)
                output.append(
                    {
                        "config": config,
                        "method": method,
                        "metric": column,
                        "count": len(values),
                        "mean": statistics.fmean(values),
                        "median": statistics.median(values),
                        "p90": ordered[min(len(ordered) - 1, int(0.9 * len(ordered)))],
                        "max": max(values),
                    }
                )
    return output


def load_truth(path: Path) -> np.ndarray:
    return np.genfromtxt(path, delimiter=",", names=True)


def truth_columns(data: np.ndarray, prefix: str) -> np.ndarray:
    source_flu = np.stack(
        [data[f"{prefix}_{axis}"] for axis in AXES], axis=1
    ).astype(np.float64)
    # Saved truth decomposition is FLU; optimizer Bias states are internal NED.
    return source_flu * FLU_TO_NED_SIGN.reshape(1, 3)


def interpolate(time_src: np.ndarray, values: np.ndarray, time_dst: np.ndarray) -> np.ndarray:
    result = np.empty((time_dst.size, 3), dtype=np.float64)
    for axis in range(3):
        result[:, axis] = np.interp(time_dst, time_src, values[:, axis])
    return result


def vector_rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum(values * values, axis=1))))


def bias_truth_summary() -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for config, _, scene in CONFIGS:
        truth = load_truth(DATA_ROOT / scene / "imu_truth_decomposition.csv")
        truth_time = truth["timestamp"].astype(np.int64)
        static = truth_time <= 3_003_333_333
        diagnostics = read_dicts(diagnostic_path("latest_biasfix_floor_1e-8", scene))
        diagnostic_time = np.asarray([int(float(row["timestamp_j"])) for row in diagnostics])
        active = np.asarray([row.get("vio_factor_active", "0") == "1" for row in diagnostics])
        diagnostic_time = diagnostic_time[active]

        for sensor, truth_prefix, measured_prefix in (
            ("gyro", "gyro", "ang_vel"),
            ("acc", "acc", "lin_acc"),
        ):
            states = np.asarray(
                [
                    [float(row[f"imu_vio_{sensor}_bias_{axis}"] or 0.0) for axis in AXES]
                    for row, enabled in zip(diagnostics, active)
                    if enabled
                ],
                dtype=np.float64,
            )
            true_bias_all = truth_columns(truth, f"{truth_prefix}_bias")
            truth_noise_all = truth_columns(truth, f"{truth_prefix}_noise")
            measured_all = truth_columns(truth, f"measured_{measured_prefix}")
            true_signal_all = truth_columns(truth, f"true_{measured_prefix}")
            static_true_bias_mean = true_bias_all[static].mean(axis=0)
            static_noise_mean = truth_noise_all[static].mean(axis=0)
            static_apparent_offset = (measured_all[static] - true_signal_all[static]).mean(axis=0)
            true_bias_at_frames = interpolate(truth_time, true_bias_all, diagnostic_time)
            first_estimate = states[0]
            fixed_error = np.broadcast_to(first_estimate, states.shape) - true_bias_at_frames
            optimized_error = states - true_bias_at_frames

            post = truth_time >= diagnostic_time[0]
            post_time = truth_time[post]
            estimated_bias_at_imu = interpolate(diagnostic_time, states, post_time)
            measured = measured_all[post]
            true_signal = true_signal_all[post]
            true_bias = true_bias_all[post]
            truth_noise = truth_noise_all[post]
            raw_error = measured - true_signal
            corrected_error = measured - estimated_bias_at_imu - true_signal
            oracle_error = measured - true_bias - true_signal

            optimized_rmse = vector_rms(optimized_error)
            fixed_rmse = vector_rms(fixed_error)
            row: dict[str, object] = {
                "config": config,
                "scene": scene,
                "sensor": sensor,
                "diagnostic_frames": int(states.shape[0]),
                "first_estimate_norm": float(np.linalg.norm(first_estimate)),
                "true_bias_at_first_norm": float(np.linalg.norm(true_bias_at_frames[0])),
                "first_estimate_error_norm": float(
                    np.linalg.norm(first_estimate - true_bias_at_frames[0])
                ),
                "static_true_bias_mean_norm": float(np.linalg.norm(static_true_bias_mean)),
                "static_noise_mean_norm": float(np.linalg.norm(static_noise_mean)),
                "static_apparent_offset_mean_norm": float(
                    np.linalg.norm(static_apparent_offset)
                ),
                "first_estimate_vs_static_apparent_offset_error_norm": float(
                    np.linalg.norm(first_estimate - static_apparent_offset)
                ),
                "true_bias_drift_norm": float(
                    np.linalg.norm(true_bias_at_frames[-1] - true_bias_at_frames[0])
                ),
                "estimated_bias_state_drift_norm": float(np.linalg.norm(states[-1] - states[0])),
                "estimated_bias_state_max_change_norm": float(
                    np.linalg.norm(states - states[0], axis=1).max()
                ),
                "optimized_bias_rmse_norm": optimized_rmse,
                "fixed_initial_bias_rmse_norm": fixed_rmse,
                "optimized_over_fixed_rmse_ratio": (
                    optimized_rmse / fixed_rmse if fixed_rmse > 0.0 else math.nan
                ),
                "optimized_final_bias_error_norm": float(np.linalg.norm(optimized_error[-1])),
                "raw_measurement_error_vector_rms": vector_rms(raw_error),
                "after_estimated_bias_error_vector_rms": vector_rms(corrected_error),
                "after_true_bias_error_vector_rms": vector_rms(oracle_error),
                "truth_noise_vector_rms": vector_rms(truth_noise),
                "estimated_correction_over_raw_rms_ratio": (
                    vector_rms(corrected_error) / vector_rms(raw_error)
                    if vector_rms(raw_error) > 0.0
                    else math.nan
                ),
            }
            for axis, axis_name in enumerate(AXES):
                row[f"first_estimate_{axis_name}"] = float(first_estimate[axis])
                row[f"static_true_bias_mean_{axis_name}"] = float(
                    static_true_bias_mean[axis]
                )
                row[f"static_noise_mean_{axis_name}"] = float(static_noise_mean[axis])
                row[f"true_bias_at_first_{axis_name}"] = float(true_bias_at_frames[0, axis])
                row[f"optimized_final_{axis_name}"] = float(states[-1, axis])
                row[f"true_bias_at_final_{axis_name}"] = float(true_bias_at_frames[-1, axis])
            output.append(row)
    return output


def visual_fingerprint_audit() -> dict[str, object]:
    fingerprints: dict[str, dict[int, str]] = {}
    for config, _, scene in CONFIGS:
        for method, _, _, _ in METHODS:
            key = f"{config}/{method}"
            rows = read_dicts(diagnostic_path(method, scene))
            fingerprints[key] = {
                int(float(row["frame_j"])): row["visual_input_sha256"] for row in rows
            }
    reference_key = "bias_no_noise/latest_biasfix_floor_1e-8"
    reference = fingerprints[reference_key]
    comparisons: dict[str, object] = {}
    for key, values in fingerprints.items():
        common = sorted(set(reference) & set(values))
        comparisons[key] = {
            "common_frames": len(common),
            "all_common_equal": all(reference[frame] == values[frame] for frame in common),
        }
    return {"reference": reference_key, "comparisons": comparisons}


def pairwise_trajectory_summary() -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    comparisons = (
        ("pure_macvo", PURE_MACVO_PATH),
        ("calibrated_imu_only", None),
        ("previous_imuatt_estinit", None),
        ("staticinit_before_bias_fix", None),
    )
    for config, _, scene in CONFIGS:
        latest = read_xyz(method_path("latest_biasfix_floor_1e-8", scene))
        for method, shared_path in comparisons:
            path = shared_path or method_path(method, scene)
            other = read_xyz(path)
            count = min(len(latest), len(other))
            distances = position_errors(latest[:count], other[:count])
            output.append(
                {
                    "config": config,
                    "latest_compared_to": method,
                    "matched_frames": count,
                    "position_rmse_m": math.sqrt(
                        sum(value * value for value in distances) / len(distances)
                    ),
                    "position_mean_m": statistics.fmean(distances),
                    "position_final_m": distances[-1],
                    "position_max_m": max(distances),
                    "other_source": str(path),
                }
            )
    return output


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_scene_payload() -> tuple[dict[str, object], list[dict[str, object]]]:
    _, _, reference_scene = CONFIGS[0]
    gt_path = DATA_ROOT / reference_scene / "ref_pose.csv"
    gt_rows = read_xyz(gt_path)
    pure_rows = read_xyz(PURE_MACVO_PATH)
    all_paths = [
        method_path(method, scene)
        for _, _, scene in CONFIGS
        for method, _, _, _ in METHODS
    ] + [method_path("calibrated_imu_only", scene) for _, _, scene in CONFIGS]
    frame_count = min(len(gt_rows), len(pure_rows), *(len(read_xyz(path)) for path in all_paths))
    gt_cut = gt_rows[:frame_count]
    pure_cut = pure_rows[:frame_count]
    t0 = gt_cut[0][0]
    fusion: list[dict[str, object]] = []
    imu_only: list[dict[str, object]] = []
    trajectory_rows: list[dict[str, object]] = []

    trajectory_rows.append(
        summarize_trajectory("shared_visual", reference_scene, "pure_macvo", PURE_MACVO_PATH)
    )
    for config, config_label, scene in CONFIGS:
        for method, method_label, color, dasharray in METHODS:
            path = method_path(method, scene)
            estimate = read_xyz(path)[:frame_count]
            fusion.append(
                {
                    "key": f"{config}_{method}",
                    "source": method_label,
                    "config": config,
                    "label": f"{config_label} / {method_label}",
                    "color": color,
                    "dasharray": dasharray,
                    "scene": scene,
                    "xyz": xyz(estimate),
                    "error_m": position_errors(gt_cut, estimate),
                    "metrics": metrics(gt_cut, estimate),
                    "path": str(path),
                }
            )
            trajectory_rows.append(summarize_trajectory(config, scene, method, path))

        imu_path = method_path("calibrated_imu_only", scene)
        imu_rows = read_xyz(imu_path)[:frame_count]
        imu_only.append(
            {
                "key": f"{config}_calibrated_imu_only",
                "config": config,
                "label": f"{config_label} / Staticinit calibrated",
                "color": "#059669" if config == "bias_no_noise" else "#d97706",
                "scene": scene,
                "xyz": xyz(imu_rows),
                "error_m": position_errors(gt_cut, imu_rows),
                "metrics": metrics(gt_cut, imu_rows),
                "path": str(imu_path),
            }
        )
        trajectory_rows.append(
            summarize_trajectory(config, scene, "calibrated_imu_only", imu_path)
        )

    payload = {
        "scene": "Stop-turn rectangle / isolated IMU errors",
        "gt": xyz(gt_cut),
        "macvo": xyz(pure_cut),
        "time_s": [(row[0] - t0) / 1e9 for row in gt_cut],
        "error_m": position_errors(gt_cut, pure_cut),
        "metrics": metrics(gt_cut, pure_cut),
        "fusion": fusion,
        "imu_only": imu_only,
        "gt_path": str(gt_path),
        "macvo_path": str(PURE_MACVO_PATH),
    }
    return payload, trajectory_rows


def main() -> None:
    required = [PURE_MACVO_PATH]
    for _, _, scene in CONFIGS:
        required.extend(
            [
                DATA_ROOT / scene / "ref_pose.csv",
                DATA_ROOT / scene / "imu_truth_decomposition.csv",
                method_path("calibrated_imu_only", scene),
            ]
        )
        for method, _, _, _ in METHODS:
            required.extend([method_path(method, scene), diagnostic_path(method, scene)])
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing isolated-IMU inputs:\n" + "\n".join(missing))

    reference_hashes = {
        scene: sha256(DATA_ROOT / scene / "ref_pose.csv") for _, _, scene in CONFIGS
    }
    if len(set(reference_hashes.values())) != 1:
        raise RuntimeError(f"The two GT files are not identical: {reference_hashes}")

    OUTDIR.mkdir(parents=True, exist_ok=True)
    scene_payload, trajectory_rows = build_scene_payload()
    diagnostic_rows = diagnostic_summary()
    bias_rows = bias_truth_summary()
    pairwise_rows = pairwise_trajectory_summary()
    visual_audit = visual_fingerprint_audit()

    write_csv(OUTDIR / "trajectory_metrics.csv", trajectory_rows)
    write_csv(OUTDIR / "optimizer_diagnostic_summary.csv", diagnostic_rows)
    write_csv(OUTDIR / "bias_noise_truth_summary.csv", bias_rows)
    write_csv(OUTDIR / "latest_pairwise_trajectory_distance.csv", pairwise_rows)
    (OUTDIR / "visual_input_audit.json").write_text(
        json.dumps(visual_audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (OUTDIR / "gt_hash_audit.json").write_text(
        json.dumps(reference_hashes, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    html = HTML_TEMPLATE.replace("{{", "{").replace("}}", "}")
    html = html.replace(
        "Circle, stop-turn rectangle, and straight trajectory comparison",
        "Rectangle isolated IMU error comparison",
    )
    html = html.replace(
        "__METHOD_SCOPE__",
        "GT, Pure MACVO, historical fusion, corrected fusion, and calibrated IMU-only",
    )
    html = html.replace(
        "__LINE_NOTE__",
        "Bias-only and noise-only runs share identical GT and cached visual observations.",
    )
    html = html.replace("__DATA__", json.dumps({"scenes": [scene_payload]}, ensure_ascii=False))
    page = OUTDIR / "interactive_trajectory_gt_vs_est.html"
    page.write_text(html, encoding="utf-8")

    print(page)
    print(OUTDIR / "trajectory_metrics.csv")
    print(OUTDIR / "optimizer_diagnostic_summary.csv")
    print(OUTDIR / "bias_noise_truth_summary.csv")
    print(OUTDIR / "latest_pairwise_trajectory_distance.csv")
    print(json.dumps(visual_audit, ensure_ascii=False))


if __name__ == "__main__":
    main()
