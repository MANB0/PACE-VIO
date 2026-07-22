#!/usr/bin/env python3
"""Evaluate the real-frontend incremental iSAM2 runs on three scenes."""

from __future__ import annotations

import argparse
import csv
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
    FLU_TO_NED,
    evaluate_method,
    interpolate_rows,
    select_truth,
)
from Scripts.evaluate_t2_isam2 import compare_states  # noqa: E402
from Scripts.evaluate_t2_isam2_full import smoothness_metrics  # noqa: E402


SCENES = {
    "circle": "clear_circle_truth_normal_noise",
    "rectangle": "clear_stop_turn_rectangle_truth_normal_noise",
    "straight": "clear_straight_truth_normal_noise",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--realtime-root", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--gt-root", required=True, type=Path)
    parser.add_argument("--cached-comparison", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def find_result(scene_root: Path) -> Path:
    candidates = list(scene_root.glob("**/poses_imu.csv"))
    if len(candidates) != 1:
        raise ValueError(f"expected one poses_imu.csv below {scene_root}, got {candidates}")
    return candidates[0].parent


def edge_range(result: Path) -> tuple[int, int, pd.DataFrame]:
    diagnostics = pd.read_csv(result / "frame_pair_diagnostics.csv")
    if diagnostics.empty:
        raise ValueError(f"no diagnostics in {result}")
    if set(diagnostics.vio_backend.astype(str)) != {"isam2"}:
        raise ValueError(f"non-iSAM2 edge found in {result}")
    start = int(diagnostics.frame_i.iloc[0])
    end = int(diagnostics.frame_j.iloc[-1])
    expected_edges = end - start
    if len(diagnostics) != expected_edges:
        raise ValueError(
            f"non-contiguous diagnostics in {result}: {len(diagnostics)} != {expected_edges}"
        )
    return start, end, diagnostics


def load_live_states(result: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    start, end, diagnostics = edge_range(result)
    poses = pd.read_csv(result / "poses_imu.csv")
    tensor = np.load(result / "tensor_map.npz", allow_pickle=False)
    selection = np.arange(start, end + 1, dtype=np.int64)
    if len(poses) <= end or tensor["frames//time_ns"].shape[0] <= end:
        raise ValueError(f"state files do not cover frame {end}")
    coordinate_frame = (result / "pose_coordinate_frame.txt").read_text(
        encoding="utf-8"
    ).strip().upper()
    if coordinate_frame != "NWU":
        raise ValueError(
            f"poses_imu.csv must be exported in NWU, got {coordinate_frame!r}"
        )
    pose_nwu = poses.loc[
        selection, ["tx", "ty", "tz", "qx", "qy", "qz", "qw"]
    ].to_numpy(np.float64)
    timestamps = tensor["frames//time_ns"][selection].astype(np.int64)
    if not np.array_equal(timestamps, poses.loc[selection, "timestamp_ns"].to_numpy(np.int64)):
        raise AssertionError("poses_imu.csv and tensor_map timestamps differ")
    velocity = tensor["frames//imu_vio_velocity_world"][selection].astype(np.float64)
    velocity[:, 1:3] *= -1.0
    ba = tensor["frames//imu_vio_acc_bias"][selection].astype(np.float64)
    bg = tensor["frames//imu_vio_gyro_bias"][selection].astype(np.float64)
    state = pd.DataFrame({
        "frame": selection,
        "timestamp_ns": timestamps,
        "tx_nwu": pose_nwu[:, 0], "ty_nwu": pose_nwu[:, 1], "tz_nwu": pose_nwu[:, 2],
        "qx_nwu": pose_nwu[:, 3], "qy_nwu": pose_nwu[:, 4],
        "qz_nwu": pose_nwu[:, 5], "qw_nwu": pose_nwu[:, 6],
        "vx_nwu": velocity[:, 0], "vy_nwu": velocity[:, 1], "vz_nwu": velocity[:, 2],
        "ba_x_internal": ba[:, 0], "ba_y_internal": ba[:, 1], "ba_z_internal": ba[:, 2],
        "bg_x_internal": bg[:, 0], "bg_y_internal": bg[:, 1], "bg_z_internal": bg[:, 2],
    })
    if not np.isfinite(state.select_dtypes(include=[np.number]).to_numpy()).all():
        raise FloatingPointError(f"live state contains NaN/Inf: {result}")
    return state, diagnostics


def runtime_summary(
    scene_root: Path,
    result_dir: Path,
    diagnostics: pd.DataFrame,
) -> dict[str, float | int | bool | str]:
    pipeline = pd.read_csv(scene_root / "pipeline_trace.csv")
    finalize = json.loads(
        (result_dir / "optimizer_finalize_summary.json").read_text(encoding="utf-8")
    )
    first_frame = int(diagnostics.frame_i.iloc[0])
    last_frame = int(diagnostics.frame_j.iloc[-1])
    expected_states = last_frame - first_frame + 1
    expected_finalize = {
        "backend": "isam2",
        "reason": "isam2_final_history_snapshot",
        "history_revision": True,
        "writeback": "all_isam2_history",
        "first_frame": first_frame,
        "last_frame": last_frame,
        "history_frame_count": expected_states,
        "state_count": expected_states,
    }
    for key, expected in expected_finalize.items():
        if finalize.get(key) != expected:
            raise ValueError(
                f"invalid final iSAM2 snapshot {result_dir}: "
                f"{key}={finalize.get(key)!r}, expected {expected!r}"
            )

    result: dict[str, float | int | bool | str] = {
        "edge_count": int(len(diagnostics)),
        "state_count": int(diagnostics.isam2_state_count.iloc[-1]),
        "history_revision_count": int(diagnostics.isam2_history_revision.astype(bool).sum()),
        "final_snapshot_verified": True,
        "final_snapshot_build_ms": float(finalize["snapshot_build_ms"]),
        "final_snapshot_first_frame": first_frame,
        "final_snapshot_last_frame": last_frame,
        "final_snapshot_state_count": expected_states,
    }
    for key, values in {
        "isam2_update": diagnostics.isam2_update_ms,
        "frontend": pipeline.frontend_ms,
        "backend_total": pipeline.backend_solver_ms,
        "commit": pipeline.commit_ms,
    }.items():
        values = pd.to_numeric(values, errors="coerce").dropna()
        result[f"{key}_median_ms"] = float(values.median())
        result[f"{key}_p95_ms"] = float(values.quantile(0.95))
        result[f"{key}_max_ms"] = float(values.max())
    return result


def plot_trajectories(
    output: Path,
    trajectories: dict[str, dict[str, np.ndarray]],
) -> None:
    views = ((0, 1, "XY"), (0, 2, "XZ"), (1, 2, "YZ"))
    colors = {"GT": "#111827", "Cached-factor iSAM2": "#2563eb", "Real-frontend iSAM2": "#059669"}
    fig, axes = plt.subplots(3, 3, figsize=(15, 14), constrained_layout=True)
    for row, scene in enumerate(SCENES):
        for column, (first, second, label) in enumerate(views):
            axis = axes[row, column]
            for method, points in trajectories[scene].items():
                rebased = points - points[0]
                axis.plot(
                    rebased[:, first], rebased[:, second],
                    color=colors[method], linewidth=2.4 if method == "GT" else 1.5,
                    linestyle="-", label=method,
                )
            axis.set_title(f"{scene.title()} / {label}")
            axis.set_xlabel("m (NWU)")
            axis.set_ylabel("m (NWU)")
            axis.axis("equal")
            axis.grid(alpha=0.25)
        axes[row, 0].legend(fontsize=8)
    fig.suptitle("T2-iSAM2: cached-factor validation vs real MACVO frontend, IMU center")
    fig.savefig(output / "realtime_t2_isam2_three_scenes.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    metric_rows: list[dict[str, object]] = []
    smoothness_rows: list[dict[str, object]] = []
    runtime_rows: list[dict[str, object]] = []
    comparison_rows: list[dict[str, object]] = []
    trajectories: dict[str, dict[str, np.ndarray]] = {}

    for scene, dataset_name in SCENES.items():
        scene_root = args.realtime_root / dataset_name
        result = find_result(scene_root)
        live, diagnostics = load_live_states(result)
        timestamps = live.timestamp_ns.to_numpy(np.int64)
        truth = select_truth(args.gt_root / scene / "gt_imu_rebased.csv", timestamps)
        dataset = args.dataset_root / dataset_name
        velocity_truth = interpolate_rows(
            dataset / "ref_pose.csv", timestamps, "timestamp", ["vx", "vy", "vz"]
        )
        acc_bias_truth = interpolate_rows(
            dataset / "imu_truth_decomposition.csv", timestamps, "timestamp",
            ["acc_bias_x", "acc_bias_y", "acc_bias_z"],
        ) @ FLU_TO_NED.T
        gyro_bias_truth = interpolate_rows(
            dataset / "imu_truth_decomposition.csv", timestamps, "timestamp",
            ["gyro_bias_x", "gyro_bias_y", "gyro_bias_z"],
        ) @ FLU_TO_NED.T

        metric = evaluate_method(
            "real_frontend_t2_isam2", live, truth,
            velocity_truth, acc_bias_truth, gyro_bias_truth,
        )
        metric["scene"] = scene
        metric_rows.append(metric)
        smoothness_rows.append({
            "scene": scene,
            "method": "real_frontend_t2_isam2",
            **smoothness_metrics(live, truth),
        })
        runtime_rows.append({
            "scene": scene,
            **runtime_summary(scene_root, result, diagnostics),
        })

        cached = pd.read_csv(
            args.cached_comparison / f"{scene}_t2_isam2_states_nwu_imu.csv"
        )
        cached = cached.set_index("timestamp_ns").reindex(timestamps).reset_index()
        if cached.isna().any().any():
            raise ValueError(f"cached comparison does not cover {scene}")
        comparison_rows.append({
            "scene": scene,
            **{
                f"live_vs_cached_{group}_{key}": value
                for group, values in compare_states(cached, live).items()
                for key, value in values.items()
            },
        })
        live.to_csv(output / f"{scene}_realtime_t2_isam2_states_nwu_imu.csv", index=False)
        trajectories[scene] = {
            "GT": truth[["tx", "ty", "tz"]].to_numpy(np.float64),
            "Cached-factor iSAM2": cached[["tx_nwu", "ty_nwu", "tz_nwu"]].to_numpy(np.float64),
            "Real-frontend iSAM2": live[["tx_nwu", "ty_nwu", "tz_nwu"]].to_numpy(np.float64),
        }

    metrics = pd.DataFrame(metric_rows)
    smoothness = pd.DataFrame(smoothness_rows)
    runtime = pd.DataFrame(runtime_rows)
    comparison = pd.DataFrame(comparison_rows)
    metrics.to_csv(output / "realtime_t2_isam2_accuracy_metrics.csv", index=False)
    smoothness.to_csv(output / "realtime_t2_isam2_smoothness_metrics.csv", index=False)
    runtime.to_csv(output / "realtime_t2_isam2_runtime_metrics.csv", index=False)
    comparison.to_csv(output / "realtime_vs_cached_state_difference.csv", index=False)
    plot_trajectories(output, trajectories)
    (output / "realtime_t2_isam2_summary.json").write_text(
        json.dumps({
            "schema_version": 1,
            "coordinate_contract": {
                "world": "NWU",
                "reference_point": "IMU center",
                "alignment": "translation rebase at frame 90 only",
                "se3_fit": False,
                "scale_fit": False,
            },
            "metrics": metric_rows,
            "smoothness": smoothness_rows,
            "runtime": runtime_rows,
            "live_vs_cached": comparison_rows,
        }, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
