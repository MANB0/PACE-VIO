#!/usr/bin/env python3
"""Plot the short relative-pose clone-ESKF replay against frozen baselines."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation


ROOT = Path("/home/admin1/macvo-dev")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Scripts.plot_circle_direct_uvd_u1_vs_pose_factor import (  # noqa: E402
    BATCH_ROOT,
    HTML_TEMPLATE,
    MACVO_POSE,
    POSE_FACTOR_POSE,
    SCENE,
    metrics,
    position_errors,
    read_forward_axes,
    read_xyz,
    xy_metrics,
    xyz,
)


ESKF = ROOT / "Results/circle_relative_pose_clone_eskf_short_20260718/poses.csv"
ESKF_DIAGNOSTICS = ESKF.parent / "eskf_diagnostics.csv"
OUTPUT = ROOT / "analysis_circle_relative_pose_clone_eskf_short_20260718"


def _relative_metrics(gt_path: Path, estimate_path: Path, count: int) -> dict[str, float]:
    gt = pd.read_csv(gt_path).iloc[:count]
    estimate = pd.read_csv(estimate_path).iloc[:count]
    gt_time = gt["timestamp"].to_numpy(np.int64)
    estimate_time = estimate["timestamp_ns"].to_numpy(np.int64)
    if not np.array_equal(gt_time, estimate_time):
        raise AssertionError(f"timestamp mismatch: {estimate_path}")

    def matrices(frame: pd.DataFrame, positions: list[str]) -> np.ndarray:
        result = np.tile(np.eye(4), (len(frame), 1, 1))
        result[:, :3, :3] = Rotation.from_quat(
            frame[["qx", "qy", "qz", "qw"]].to_numpy()
        ).as_matrix()
        result[:, :3, 3] = frame[positions].to_numpy(np.float64)
        return result

    gt_pose = matrices(gt, ["x", "y", "z"])
    estimate_pose = matrices(estimate, ["tx", "ty", "tz"])
    gt_relative = np.linalg.inv(gt_pose[:-1]) @ gt_pose[1:]
    estimate_relative = np.linalg.inv(estimate_pose[:-1]) @ estimate_pose[1:]
    error = np.linalg.inv(gt_relative) @ estimate_relative
    translation = np.linalg.norm(error[:, :3, 3], axis=1)
    rotation = Rotation.from_matrix(error[:, :3, :3]).magnitude()
    return {
        "translation_rpe_rmse_m": float(np.sqrt(np.mean(np.square(translation)))),
        "rotation_rpe_rmse_rad": float(np.sqrt(np.mean(np.square(rotation)))),
    }


def _xy_error_smoothness(
    gt_path: Path, estimate_path: Path, count: int, active_from: int
) -> dict[str, float]:
    gt = pd.read_csv(gt_path).iloc[:count][["x", "y"]].to_numpy(np.float64)
    estimate = (
        pd.read_csv(estimate_path).iloc[:count][["tx", "ty"]].to_numpy(np.float64)
    )
    error = (estimate - gt)[active_from:]
    first = np.diff(error, axis=0)
    second = np.diff(error, n=2, axis=0)
    return {
        "active_xy_error_rmse_m": float(
            np.sqrt(np.mean(np.sum(np.square(error), axis=1)))
        ),
        "active_xy_error_first_difference_rmse_m": float(
            np.sqrt(np.mean(np.sum(np.square(first), axis=1)))
        ),
        "active_xy_error_second_difference_rmse_m": float(
            np.sqrt(np.mean(np.sum(np.square(second), axis=1)))
        ),
    }


def _diagnostic_summary(path: Path) -> dict[str, object]:
    frame = pd.read_csv(path)
    columns = [
        "pose_nis",
        "pose_residual_norm_before",
        "pose_residual_norm_after",
        "position_correction_norm_m",
        "relative_position_correction_norm_m",
        "velocity_correction_norm_mps",
        "rotation_correction_norm_rad",
        "acc_bias_correction_norm_mps2",
        "gyro_bias_correction_norm_radps",
        "nav_cov_min_eigenvalue",
        "nav_cov_max_eigenvalue",
    ]
    summary = {
        column: {
            "min": float(frame[column].min()),
            "median": float(frame[column].median()),
            "p95": float(frame[column].quantile(0.95)),
            "max": float(frame[column].max()),
        }
        for column in columns
    }
    return {
        "edge_count": int(len(frame)),
        "finite_all": bool(frame["finite"].all()),
        "pose_nis_chi2_6d_95_coverage": float(
            (frame["pose_nis"] <= 12.591587243743977).mean()
        ),
        "statistics": summary,
    }


def main() -> None:
    gt_path = BATCH_ROOT / SCENE / "ref_pose.csv"
    paths = {
        "pure_macvo": MACVO_POSE,
        "pose_factor": POSE_FACTOR_POSE,
        "clone_eskf": ESKF,
    }
    for path in (gt_path, *paths.values(), ESKF_DIAGNOSTICS):
        if not path.exists():
            raise FileNotFoundError(path)
    count = len(pd.read_csv(ESKF))
    trajectories = {"GT": read_xyz(gt_path)[:count]}
    trajectories.update({key: read_xyz(path)[:count] for key, path in paths.items()})
    forwards = {"GT": read_forward_axes(gt_path)[:count]}
    forwards.update({key: read_forward_axes(path)[:count] for key, path in paths.items()})
    timestamps = [row[0] for row in trajectories["GT"]]
    for key, rows in trajectories.items():
        if [row[0] for row in rows] != timestamps:
            raise AssertionError(f"timestamp mismatch for {key}")

    labels = {
        "pose_factor": "Relative pose T_ij factor graph",
        "clone_eskf": "Relative pose stochastic-clone ESKF",
    }
    colors = {"pose_factor": "#f97316", "clone_eskf": "#2563eb"}
    metric_payload: dict[str, object] = {}
    for key in ("pure_macvo", "pose_factor", "clone_eskf"):
        metric_payload[key] = {
            **metrics(trajectories["GT"], trajectories[key]),
            **xy_metrics(trajectories["GT"], trajectories[key]),
            **_relative_metrics(gt_path, paths[key], count),
            **_xy_error_smoothness(gt_path, paths[key], count, active_from=90),
        }

    payload = {
        "scene": f"Circle / Normal noise / First {count} frames",
        "gt": xyz(trajectories["GT"]),
        "gt_forward": forwards["GT"],
        "macvo": xyz(trajectories["pure_macvo"]),
        "macvo_forward": forwards["pure_macvo"],
        "time_s": [(value - timestamps[0]) * 1.0e-9 for value in timestamps],
        "error_m": position_errors(
            trajectories["GT"], trajectories["pure_macvo"], xy_only=False
        ),
        "metrics": metrics(trajectories["GT"], trajectories["pure_macvo"]),
        "fusion": [
            {
                "key": key,
                "source": key,
                "config": "normal_noise",
                "label": labels[key],
                "color": colors[key],
                "dasharray": "",
                "scene": SCENE,
                "xyz": xyz(trajectories[key]),
                "forward": forwards[key],
                "error_m": position_errors(
                    trajectories["GT"], trajectories[key], xy_only=False
                ),
                "metrics": metrics(trajectories["GT"], trajectories[key]),
                "path": str(paths[key]),
            }
            for key in ("pose_factor", "clone_eskf")
        ],
        "imu_only": [],
        "gt_path": str(gt_path),
        "macvo_path": str(MACVO_POSE),
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    template = HTML_TEMPLATE.replace("{{", "{").replace("}}", "}")
    template = template.replace(
        "Circle, stop-turn rectangle, and straight trajectory comparison",
        "Relative-pose factor graph vs stochastic-clone ESKF",
    )
    template = template.replace(
        "__METHOD_SCOPE__",
        "GT, Pure MACVO, Relative-pose T_ij factor graph, and clone ESKF",
    )
    template = template.replace(
        "__LINE_NOTE__",
        "Timestamp-matched NWU; no alignment, fitting, or scale correction. XY is primary.",
    )
    (OUTPUT / "interactive_relative_pose_clone_eskf_short.html").write_text(
        template.replace(
            "__DATA__", json.dumps({"scenes": [payload]}, ensure_ascii=False)
        ),
        encoding="utf-8",
    )
    (OUTPUT / "metrics.json").write_text(
        json.dumps(metric_payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (OUTPUT / "diagnostics_summary.json").write_text(
        json.dumps(_diagnostic_summary(ESKF_DIAGNOSTICS), indent=2),
        encoding="utf-8",
    )

    figure, axis = plt.subplots(figsize=(10, 7))
    axis.plot(
        [row[1] for row in trajectories["GT"]],
        [row[2] for row in trajectories["GT"]],
        color="#111827",
        linewidth=2.5,
        label="GT",
    )
    for key in ("pose_factor", "clone_eskf"):
        axis.plot(
            [row[1] for row in trajectories[key]],
            [row[2] for row in trajectories[key]],
            color=colors[key],
            linewidth=1.8,
            label=labels[key],
        )
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("x / m (NWU)")
    axis.set_ylabel("y / m (NWU)")
    axis.grid(True, alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(OUTPUT / "relative_pose_clone_eskf_short_xy.png", dpi=180)
    plt.close(figure)
    print(OUTPUT)


if __name__ == "__main__":
    main()
