import csv
import math
from pathlib import Path

import pandas as pd
import pytest

from Scripts.evaluate_macvo_relative_metrics import RunSpec, evaluate_run
from Scripts.run_clear_circle_imu_only_mechanization import evaluate_joined


def _yaw_quat(deg: float) -> tuple[float, float, float, float]:
    half = math.radians(deg) * 0.5
    return (0.0, 0.0, math.sin(half), math.cos(half))


def _write_pose_csv(path: Path, positions: list[tuple[float, float, float]], yaws_deg: list[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    timestamps = [0, 1_000_000_000, 2_000_000_000]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp_ns", "tx", "ty", "tz", "qx", "qy", "qz", "qw"])
        for ts, pos, yaw in zip(timestamps, positions, yaws_deg):
            writer.writerow([ts, *pos, *_yaw_quat(yaw)])


def test_relative_metrics_include_velocity_normalized_errors(tmp_path: Path) -> None:
    est_path = tmp_path / "est.csv"
    gt_path = tmp_path / "gt.csv"
    _write_pose_csv(gt_path, [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0)], [0.0, 0.0, 0.0])
    _write_pose_csv(est_path, [(0.0, 0.0, 0.0), (0.9, 0.0, 0.0), (1.8, 0.0, 0.0)], [0.0, 0.0, 0.0])

    row = evaluate_run(
        RunSpec(
            dataset="synthetic",
            scene="scene",
            method="method",
            source="source",
            trial="trial",
            poses_path=est_path,
            gt_path=gt_path,
        )
    )

    assert row["t_rel_m_per_frame"] == pytest.approx(0.1)
    assert row["r_rel_deg_per_frame"] == pytest.approx(0.0)
    assert row["t_vel_m_s"] == pytest.approx(0.1)
    assert row["r_vel_deg_s"] == pytest.approx(0.0)


def test_evaluate_joined_includes_velocity_normalized_errors() -> None:
    joined = pd.DataFrame(
        {
            "timestamp_ns": [0, 1_000_000_000, 2_000_000_000],
            "tx_gt": [0.0, 0.0, 0.0],
            "ty_gt": [0.0, 0.0, 0.0],
            "tz_gt": [0.0, 0.0, 0.0],
            "tx_est": [0.0, 0.0, 0.0],
            "ty_est": [0.0, 0.0, 0.0],
            "tz_est": [0.0, 0.0, 0.0],
            "qx_gt": [0.0, 0.0, 0.0],
            "qy_gt": [0.0, 0.0, 0.0],
            "qz_gt": [_yaw_quat(0.0)[2], _yaw_quat(90.0)[2], _yaw_quat(180.0)[2]],
            "qw_gt": [_yaw_quat(0.0)[3], _yaw_quat(90.0)[3], _yaw_quat(180.0)[3]],
            "qx_est": [0.0, 0.0, 0.0],
            "qy_est": [0.0, 0.0, 0.0],
            "qz_est": [_yaw_quat(0.0)[2], _yaw_quat(45.0)[2], _yaw_quat(90.0)[2]],
            "qw_est": [_yaw_quat(0.0)[3], _yaw_quat(45.0)[3], _yaw_quat(90.0)[3]],
        }
    )

    row = evaluate_joined("scene / method", "scene", "method", joined, "source")

    assert row["t_vel_m_s"] == pytest.approx(0.0)
    assert row["r_vel_deg_s"] == pytest.approx(45.0)
