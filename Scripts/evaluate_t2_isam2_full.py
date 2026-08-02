#!/usr/bin/env python3
"""Evaluate full-sequence PACE-VIO-iSAM2 accuracy, smoothness, and timing growth."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Scripts.evaluate_t2_history_smoother import (  # noqa: E402
    evaluate_method,
    interpolate_rows,
    select_truth,
)
from Scripts.evaluate_t2_isam2 import convert_isam2_states  # noqa: E402


SCENES = {
    "circle": "clear_circle_truth_normal_noise",
    "rectangle": "clear_stop_turn_rectangle_truth_normal_noise",
    "straight": "clear_straight_truth_normal_noise",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-root", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--gt-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def rmse_norm(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum(np.square(values), axis=1))))


def smoothness_metrics(estimate: pd.DataFrame, truth: pd.DataFrame) -> dict[str, float]:
    position = estimate[["tx_nwu", "ty_nwu", "tz_nwu"]].to_numpy(np.float64)
    gt_position = truth[["tx", "ty", "tz"]].to_numpy(np.float64)
    position_error = (position - position[0]) - (gt_position - gt_position[0])
    rotation = Rotation.from_quat(
        estimate[["qx_nwu", "qy_nwu", "qz_nwu", "qw_nwu"]].to_numpy(np.float64)
    )
    gt_rotation = Rotation.from_quat(
        truth[["qx", "qy", "qz", "qw"]].to_numpy(np.float64)
    )
    rotation_error = (gt_rotation.inv() * rotation).as_rotvec()
    return {
        "position_error_step_rms_m": rmse_norm(np.diff(position_error, axis=0)),
        "position_error_second_difference_rms_m": rmse_norm(
            np.diff(position_error, n=2, axis=0)
        ),
        "rotation_error_step_rms_deg": float(
            np.rad2deg(rmse_norm(np.diff(rotation_error, axis=0)))
        ),
        "rotation_error_second_difference_rms_deg": float(
            np.rad2deg(rmse_norm(np.diff(rotation_error, n=2, axis=0)))
        ),
    }


def state_difference(first: pd.DataFrame, second: pd.DataFrame) -> dict[str, float]:
    if not np.array_equal(first.timestamp_ns, second.timestamp_ns):
        raise AssertionError("state comparison timestamps differ")
    first_position = first[["tx_nwu", "ty_nwu", "tz_nwu"]].to_numpy(np.float64)
    second_position = second[["tx_nwu", "ty_nwu", "tz_nwu"]].to_numpy(np.float64)
    first_rotation = Rotation.from_quat(
        first[["qx_nwu", "qy_nwu", "qz_nwu", "qw_nwu"]].to_numpy(np.float64)
    )
    second_rotation = Rotation.from_quat(
        second[["qx_nwu", "qy_nwu", "qz_nwu", "qw_nwu"]].to_numpy(np.float64)
    )
    return {
        "position_rmse_m": rmse_norm(second_position - first_position),
        "position_max_m": float(
            np.max(np.linalg.norm(second_position - first_position, axis=1))
        ),
        "rotation_rmse_deg": float(
            np.rad2deg(rmse_norm((first_rotation.inv() * second_rotation).as_rotvec()))
        ),
        "rotation_max_deg": float(
            np.rad2deg(np.max((first_rotation.inv() * second_rotation).magnitude()))
        ),
    }


def timing_rows(scene: str, method: str, path: Path) -> list[dict[str, float | int | str]]:
    timing = pd.read_csv(path)
    timing["bin"] = ((timing.local_j - 1) // 300).astype(np.int64)
    result: list[dict[str, float | int | str]] = []
    for bin_index, group in timing.groupby("bin", sort=True):
        result.append(
            {
                "scene": scene,
                "method": method,
                "bin": int(bin_index),
                "edge_start": int(group.local_j.min()),
                "edge_end": int(group.local_j.max()),
                "edge_count": int(len(group)),
                "median_ms": float(group.update_ms.median()),
                "p95_ms": float(group.update_ms.quantile(0.95)),
                "max_ms": float(group.update_ms.max()),
            }
        )
    return result


def write_plot(
    output: Path,
    trajectories: dict[str, dict[str, tuple[np.ndarray, str]]],
) -> None:
    views = ((0, 1, "XY"), (0, 2, "XZ"), (1, 2, "YZ"))
    fig, axes = plt.subplots(3, 3, figsize=(15, 14), constrained_layout=True)
    for row, scene in enumerate(SCENES):
        for column, (first, second, label) in enumerate(views):
            axis = axes[row, column]
            for name, (points, color) in trajectories[scene].items():
                axis.plot(
                    points[:, first], points[:, second], color=color,
                    linewidth=2.4 if name == "GT" else 1.5, label=name,
                )
            axis.set_title(f"{scene.title()} / {label}")
            axis.set_xlabel("m (NWU)")
            axis.set_ylabel("m (NWU)")
            axis.axis("equal")
            axis.grid(alpha=0.25)
        axes[row, 0].legend(fontsize=8)
    fig.suptitle("Full-sequence PACE-VIO-2S vs PACE-VIO-iSAM2, IMU center")
    fig.savefig(output / "t2_isam2_full_three_scenes.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    analysis_root = args.analysis_root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    metric_rows: list[dict[str, float | int | str]] = []
    smoothness_rows: list[dict[str, float | int | str]] = []
    timing: list[dict[str, float | int | str]] = []
    summaries: dict[str, object] = {}
    trajectories: dict[str, dict[str, tuple[np.ndarray, str]]] = {}

    for scene, dataset_name in SCENES.items():
        scene_root = analysis_root / scene
        online = convert_isam2_states(scene_root / "bundle" / "states.csv")
        numerical = convert_isam2_states(
            scene_root / "isam2_result" / "isam2_states_internal.csv"
        )
        analytic = convert_isam2_states(
            scene_root / "isam2_result_analytic" / "isam2_states_internal.csv"
        )
        timestamps = analytic.timestamp_ns.to_numpy(np.int64)
        truth = select_truth(
            args.gt_root / scene / "gt_imu_rebased.csv", timestamps
        )
        dataset = args.dataset_root / dataset_name
        velocity_truth = interpolate_rows(
            dataset / "ref_pose.csv", timestamps, "timestamp", ["vx", "vy", "vz"]
        )
        acc_bias_truth = interpolate_rows(
            dataset / "imu_truth_decomposition.csv",
            timestamps,
            "timestamp",
            ["acc_bias_x", "acc_bias_y", "acc_bias_z"],
        )
        gyro_bias_truth = interpolate_rows(
            dataset / "imu_truth_decomposition.csv",
            timestamps,
            "timestamp",
            ["gyro_bias_x", "gyro_bias_y", "gyro_bias_z"],
        )

        estimates = {
            "online_t2": online,
            "t2_isam2_analytic": analytic,
        }
        for method, estimate in estimates.items():
            row = evaluate_method(
                method, estimate, truth, velocity_truth, acc_bias_truth, gyro_bias_truth
            )
            row["scene"] = scene
            metric_rows.append(row)
            smoothness_rows.append(
                {"scene": scene, "method": method, **smoothness_metrics(estimate, truth)}
            )

        numerical_summary = json.loads(
            (scene_root / "isam2_result" / "isam2_summary.json").read_text(encoding="utf-8")
        )
        analytic_summary = json.loads(
            (scene_root / "isam2_result_analytic" / "isam2_summary.json").read_text(
                encoding="utf-8"
            )
        )
        audit = json.loads(
            (scene_root / "isam2_result_analytic" / "factor_equivalence_summary.json").read_text(
                encoding="utf-8"
            )
        )
        timing.extend(
            timing_rows(
                scene,
                "numerical_imu_jacobian",
                scene_root / "isam2_result" / "isam2_update_timing.csv",
            )
        )
        timing.extend(
            timing_rows(
                scene,
                "analytic_imu_jacobian",
                scene_root / "isam2_result_analytic" / "isam2_update_timing.csv",
            )
        )
        summaries[scene] = {
            "numerical_runtime": numerical_summary,
            "analytic_runtime": analytic_summary,
            "analytic_factor_audit": audit,
            "analytic_vs_numerical_final_state": state_difference(numerical, analytic),
        }

        gt_points = truth[["tx", "ty", "tz"]].to_numpy(np.float64)
        online_points = online[["tx_nwu", "ty_nwu", "tz_nwu"]].to_numpy(np.float64)
        analytic_points = analytic[["tx_nwu", "ty_nwu", "tz_nwu"]].to_numpy(np.float64)
        trajectories[scene] = {
            "GT": (gt_points - gt_points[0], "#111827"),
            "Online PACE-VIO-2S": (online_points - online_points[0], "#dc2626"),
            "PACE-VIO-iSAM2": (analytic_points - analytic_points[0], "#059669"),
        }
        analytic.to_csv(
            output / f"{scene}_t2_isam2_states_nwu_imu.csv",
            index=False,
            float_format="%.17g",
        )

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(output / "t2_isam2_full_accuracy_metrics.csv", index=False)
    smoothness = pd.DataFrame(smoothness_rows)
    smoothness.to_csv(output / "t2_isam2_full_smoothness_metrics.csv", index=False)
    pd.DataFrame(timing).to_csv(output / "t2_isam2_timing_by_300_edges.csv", index=False)
    payload = {
        "schema_version": 1,
        "coordinate_contract": {
            "reference_point": "IMU center",
            "world": "NWU",
            "alignment": "translation rebase at frame 90 only",
            "se3_fit": False,
            "scale_fit": False,
        },
        "scenes": summaries,
        "metrics": metric_rows,
        "smoothness": smoothness_rows,
    }
    (output / "t2_isam2_full_summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    write_plot(output, trajectories)


if __name__ == "__main__":
    main()
