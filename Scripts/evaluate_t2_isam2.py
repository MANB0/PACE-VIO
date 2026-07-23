#!/usr/bin/env python3
"""Compare the isolated T2-iSAM2 prototype with existing T2 estimates."""

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
    load_estimate,
    navigation_states,
    select_truth,
)
from Utility.PoseFrame import convert_pose_world_frame_only  # noqa: E402
from Utility.T2HistorySmoother import (  # noqa: E402
    factor_cost_breakdown,
    load_t2_history_archive,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--isam2-result-dir", required=True, type=Path)
    parser.add_argument("--history-result-dir", required=True, type=Path)
    parser.add_argument("--tensor-map", required=True, type=Path)
    parser.add_argument("--gt-imu-csv", required=True, type=Path)
    parser.add_argument("--ref-pose", required=True, type=Path)
    parser.add_argument("--bias-truth", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def convert_isam2_states(path: Path) -> pd.DataFrame:
    source = pd.read_csv(path)
    required = {
        "frame", "timestamp_ns", "tx", "ty", "tz",
        "qx", "qy", "qz", "qw", "vx", "vy", "vz",
        "ba_x", "ba_y", "ba_z", "bg_x", "bg_y", "bg_z",
    }
    missing = sorted(required - set(source.columns))
    if missing:
        raise ValueError(f"{path} lacks columns {missing}")
    numeric = source[list(required)].select_dtypes(include=[np.number]).to_numpy()
    if not np.isfinite(numeric).all():
        raise FloatingPointError(f"{path} contains NaN/Inf")

    pose_internal = source[
        ["tx", "ty", "tz", "qx", "qy", "qz", "qw"]
    ].to_numpy(np.float64)
    pose_nwu = convert_pose_world_frame_only(pose_internal, "NED", "NWU")
    velocity_nwu = source[["vx", "vy", "vz"]].to_numpy(np.float64)
    velocity_nwu[:, 1:3] *= -1.0
    return pd.DataFrame(
        {
            "frame": source.frame.to_numpy(np.int64),
            "timestamp_ns": source.timestamp_ns.to_numpy(np.int64),
            "tx_nwu": pose_nwu[:, 0],
            "ty_nwu": pose_nwu[:, 1],
            "tz_nwu": pose_nwu[:, 2],
            "qx_nwu": pose_nwu[:, 3],
            "qy_nwu": pose_nwu[:, 4],
            "qz_nwu": pose_nwu[:, 5],
            "qw_nwu": pose_nwu[:, 6],
            "vx_nwu": velocity_nwu[:, 0],
            "vy_nwu": velocity_nwu[:, 1],
            "vz_nwu": velocity_nwu[:, 2],
            "ba_x_internal": source.ba_x,
            "ba_y_internal": source.ba_y,
            "ba_z_internal": source.ba_z,
            "bg_x_internal": source.bg_x,
            "bg_y_internal": source.bg_y,
            "bg_z_internal": source.bg_z,
        }
    )


def vector_norm_stats(values: np.ndarray) -> dict[str, float]:
    norms = np.linalg.norm(values, axis=1)
    return {
        "rmse_norm": float(np.sqrt(np.mean(np.square(norms)))),
        "median_norm": float(np.median(norms)),
        "max_norm": float(np.max(norms)),
        "max_abs_component": float(np.max(np.abs(values))),
    }


def compare_states(reference: pd.DataFrame, candidate: pd.DataFrame) -> dict[str, object]:
    if not np.array_equal(reference.frame, candidate.frame):
        raise AssertionError("reference and candidate frame indices differ")
    if not np.array_equal(reference.timestamp_ns, candidate.timestamp_ns):
        raise AssertionError("reference and candidate timestamps differ")

    p_ref = reference[["tx_nwu", "ty_nwu", "tz_nwu"]].to_numpy(np.float64)
    p = candidate[["tx_nwu", "ty_nwu", "tz_nwu"]].to_numpy(np.float64)
    p_ref_rebased = p_ref - p_ref[0]
    p_rebased = p - p[0]
    r_ref = Rotation.from_quat(
        reference[["qx_nwu", "qy_nwu", "qz_nwu", "qw_nwu"]].to_numpy(np.float64)
    )
    r = Rotation.from_quat(
        candidate[["qx_nwu", "qy_nwu", "qz_nwu", "qw_nwu"]].to_numpy(np.float64)
    )
    rotation_error = (r_ref.inv() * r).as_rotvec()
    velocity_ref = reference[["vx_nwu", "vy_nwu", "vz_nwu"]].to_numpy(np.float64)
    velocity = candidate[["vx_nwu", "vy_nwu", "vz_nwu"]].to_numpy(np.float64)
    ba_ref = reference[
        ["ba_x_internal", "ba_y_internal", "ba_z_internal"]
    ].to_numpy(np.float64)
    ba = candidate[
        ["ba_x_internal", "ba_y_internal", "ba_z_internal"]
    ].to_numpy(np.float64)
    bg_ref = reference[
        ["bg_x_internal", "bg_y_internal", "bg_z_internal"]
    ].to_numpy(np.float64)
    bg = candidate[
        ["bg_x_internal", "bg_y_internal", "bg_z_internal"]
    ].to_numpy(np.float64)
    return {
        "position_raw_m": vector_norm_stats(p - p_ref),
        "position_rebased_m": vector_norm_stats(p_rebased - p_ref_rebased),
        "rotation_rad": vector_norm_stats(rotation_error),
        "rotation_deg": vector_norm_stats(np.rad2deg(rotation_error)),
        "velocity_mps": vector_norm_stats(velocity - velocity_ref),
        "acc_bias_mps2": vector_norm_stats(ba - ba_ref),
        "gyro_bias_radps": vector_norm_stats(bg - bg_ref),
    }


def summarize_costs(costs: dict[str, np.ndarray]) -> dict[str, float]:
    summary = {key: float(value.sum()) for key, value in costs.items()}
    summary["total"] = float(sum(summary.values()))
    return summary


def write_plot(
    output: Path,
    truth: pd.DataFrame,
    estimates: dict[str, pd.DataFrame],
) -> None:
    gt = truth[["tx", "ty", "tz"]].to_numpy(np.float64)
    curves: dict[str, tuple[np.ndarray, str]] = {
        "GT": (gt - gt[0], "#111827"),
    }
    colors = {
        "Online T2": "#dc2626",
        "Python full-history 15D": "#2563eb",
        "T2-iSAM2": "#059669",
    }
    for name, frame in estimates.items():
        points = frame[["tx_nwu", "ty_nwu", "tz_nwu"]].to_numpy(np.float64)
        curves[name] = (points - points[0], colors[name])

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    views = ((0, 1, "XY"), (0, 2, "XZ"), (1, 2, "YZ"))
    for axis, (first, second, title) in zip(axes, views):
        for name, (points, color) in curves.items():
            axis.plot(
                points[:, first], points[:, second],
                color=color, linewidth=2.4 if name == "GT" else 1.7,
                label=name,
            )
        axis.set_title(title)
        axis.set_xlabel("m (NWU)")
        axis.set_ylabel("m (NWU)")
        axis.axis("equal")
        axis.grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    fig.suptitle("T2 solver comparison, frames 90-299, IMU center")
    fig.savefig(output / "t2_isam2_vs_existing.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    isam2_dir = args.isam2_result_dir.expanduser().resolve()
    history_dir = args.history_result_dir.expanduser().resolve()

    isam2 = convert_isam2_states(isam2_dir / "isam2_states_internal.csv")
    isam2.to_csv(output / "t2_isam2_states.csv", index=False, float_format="%.17g")
    online = load_estimate(history_dir / "t2_online_states.csv")
    history = load_estimate(history_dir / "t2_smoothed_states.csv")
    isam2 = load_estimate(output / "t2_isam2_states.csv")
    for candidate in (online, history):
        if not np.array_equal(candidate.timestamp_ns, isam2.timestamp_ns):
            raise AssertionError("all estimates must cover the same timestamps")

    timestamps = isam2.timestamp_ns.to_numpy(np.int64)
    truth = select_truth(args.gt_imu_csv, timestamps)
    velocity_truth = interpolate_rows(
        args.ref_pose, timestamps, "timestamp", ["vx", "vy", "vz"]
    )
    acc_bias_truth = interpolate_rows(
        args.bias_truth,
        timestamps,
        "timestamp",
        ["acc_bias_x", "acc_bias_y", "acc_bias_z"],
    )
    gyro_bias_truth = interpolate_rows(
        args.bias_truth,
        timestamps,
        "timestamp",
        ["gyro_bias_x", "gyro_bias_y", "gyro_bias_z"],
    )

    estimates = {
        "Online T2": online,
        "Python full-history 15D": history,
        "T2-iSAM2": isam2,
    }
    metrics = [
        evaluate_method(
            name,
            frame,
            truth,
            velocity_truth,
            acc_bias_truth,
            gyro_bias_truth,
        )
        for name, frame in estimates.items()
    ]
    pd.DataFrame(metrics).to_csv(output / "t2_isam2_accuracy_metrics.csv", index=False)

    archive = load_t2_history_archive(
        args.tensor_map,
        start_frame=int(isam2.frame.iloc[0]),
        end_frame=int(isam2.frame.iloc[-1]),
    )
    factor_costs = {
        name: summarize_costs(factor_cost_breakdown(navigation_states(frame), archive))
        for name, frame in estimates.items()
    }
    runtime_summary = json.loads(
        (isam2_dir / "isam2_summary.json").read_text(encoding="utf-8")
    )
    equivalence_summary = json.loads(
        (isam2_dir / "factor_equivalence_summary.json").read_text(encoding="utf-8")
    )
    summary = {
        "schema_version": 1,
        "coordinate_contract": {
            "physical_point": "IMU center",
            "evaluation_frame": "NWU",
            "translation_alignment": "rebase each trajectory at frame 90 only",
            "se3_fit": False,
            "scale_fit": False,
            "isam2_source_frame": "internal NED",
        },
        "metrics": metrics,
        "factor_costs_recomputed_by_python": factor_costs,
        "isam2_runtime": runtime_summary,
        "factor_equivalence": equivalence_summary,
        "isam2_vs_python_full_history": compare_states(history, isam2),
    }
    (output / "t2_isam2_comparison_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_plot(output, truth, estimates)


if __name__ == "__main__":
    main()
