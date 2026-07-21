"""Audit a real-image/IMU T2 run without using visual cache data."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def poses(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = read_csv(path)
    ts = np.asarray([int(row["timestamp_ns"]) for row in rows], dtype=np.int64)
    p = np.asarray(
        [[float(row[key]) for key in ("tx", "ty", "tz")] for row in rows],
        dtype=np.float64,
    )
    q = np.asarray(
        [[float(row[key]) for key in ("qx", "qy", "qz", "qw")] for row in rows],
        dtype=np.float64,
    )
    q /= np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-15)
    return ts, p, q


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ax, ay, az, aw = np.moveaxis(a, -1, 0)
    bx, by, bz, bw = np.moveaxis(b, -1, 0)
    return np.stack(
        (
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ),
        axis=-1,
    )


def quat_inv(q: np.ndarray) -> np.ndarray:
    result = q.copy()
    result[..., :3] *= -1.0
    return result / np.maximum(np.sum(q * q, axis=-1, keepdims=True), 1e-15)


def quat_to_rot(q: np.ndarray) -> np.ndarray:
    x, y, z, w = np.moveaxis(q, -1, 0)
    return np.stack(
        (
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        ),
        axis=-1,
    ).reshape((-1, 3, 3))


def angle_from_quat(q: np.ndarray) -> np.ndarray:
    q = q / np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), 1e-15)
    return 2.0 * np.arctan2(np.linalg.norm(q[..., :3], axis=-1), np.abs(q[..., 3]))


def rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values))))


def percentile(values: np.ndarray, q: float) -> float:
    return float(np.percentile(values, q))


def metric_block(values: np.ndarray) -> dict[str, float]:
    return {
        "rmse": rms(values),
        "mean_abs": float(np.mean(np.abs(values))),
        "p95_abs": percentile(np.abs(values), 95),
        "max_abs": float(np.max(np.abs(values))),
    }


def trajectory_metrics(p: np.ndarray, q: np.ndarray, gt_p: np.ndarray, gt_q: np.ndarray) -> dict:
    pos_error = p - gt_p
    rot_error_q = quat_mul(quat_inv(gt_q), q)
    rot_error = angle_from_quat(rot_error_q)
    if len(p) > 1:
        est_dp = p[1:] - p[:-1]
        gt_dp = gt_p[1:] - gt_p[:-1]
        rpe_t = np.linalg.norm(est_dp - gt_dp, axis=1)
        est_dq = quat_mul(quat_inv(q[:-1]), q[1:])
        gt_dq = quat_mul(quat_inv(gt_q[:-1]), gt_q[1:])
        rpe_r = angle_from_quat(quat_mul(quat_inv(gt_dq), est_dq))
    else:
        rpe_t = np.zeros(0)
        rpe_r = np.zeros(0)
    return {
        "position_axis": {axis: metric_block(pos_error[:, i]) for i, axis in enumerate(("x", "y", "z"))},
        "position_norm": metric_block(np.linalg.norm(pos_error, axis=1)),
        "orientation_rad": metric_block(rot_error),
        "orientation_deg": metric_block(np.rad2deg(rot_error)),
        "rpe_translation": metric_block(rpe_t),
        "rpe_rotation_rad": metric_block(rpe_r),
        "rpe_rotation_deg": metric_block(np.rad2deg(rpe_r)),
    }


def load_gt_imu_center(path: Path, imu_to_camera: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = read_csv(path)
    ts = np.asarray([int(row["timestamp"]) for row in rows], dtype=np.int64)
    p_camera = np.asarray(
        [[float(row[key]) for key in ("x", "y", "z")] for row in rows],
        dtype=np.float64,
    )
    q_camera = np.asarray(
        [[float(row[key]) for key in ("qx", "qy", "qz", "qw")] for row in rows],
        dtype=np.float64,
    )
    q_camera /= np.maximum(np.linalg.norm(q_camera, axis=1, keepdims=True), 1e-15)
    # metadata defines p_C = R_CI p_I + t_CI, so p_I = p_C - R_WC t_CI.
    p_imu = p_camera - np.einsum("nij,j->ni", quat_to_rot(q_camera), imu_to_camera)
    return ts, p_imu, q_camera


def trace_metrics(path: Path) -> dict:
    rows = read_csv(path)
    if not rows:
        return {"rows": 0}
    solver = np.asarray([float(row["backend_solver_ms"]) for row in rows if row["backend_solver_ms"]], dtype=float)
    frontend = np.asarray([float(row["frontend_ms"]) for row in rows if row["frontend_ms"]], dtype=float)
    wait = np.asarray([float(row["backend_wait_ms"]) for row in rows if row["backend_wait_ms"]], dtype=float)
    commit = np.asarray([float(row["commit_ms"]) for row in rows if row["commit_ms"]], dtype=float)
    return {
        "rows": len(rows),
        "first_edge": [int(rows[0]["frame_i"]), int(rows[0]["frame_j"])],
        "last_edge": [int(rows[-1]["frame_i"]), int(rows[-1]["frame_j"])],
        "frontend_ms_median": float(np.median(frontend)),
        "frontend_ms_p95": percentile(frontend, 95),
        "backend_solver_ms_median": float(np.median(solver)) if solver.size else None,
        "backend_solver_ms_p95": percentile(solver, 95) if solver.size else None,
        "backend_wait_ms_median": float(np.median(wait)) if wait.size else None,
        "commit_ms_median": float(np.median(commit)) if commit.size else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--gt", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    with args.metadata.open(encoding="utf-8") as stream:
        metadata = json.load(stream)
    t_imu_camera = np.asarray(metadata["extrinsics"]["T_imu_camera"]["translation_body_nwu_m"], dtype=float)
    serial_dir = args.root / "serial"
    pipeline_dir = args.root / "pipeline"
    serial_pose = args.root / "serial_poses_imu.csv"
    pipeline_pose = args.root / "pipeline_poses_imu.csv"
    gt_ts, gt_p_all, gt_q_all = load_gt_imu_center(args.gt, t_imu_camera)
    serial_ts, serial_p, serial_q = poses(serial_pose)
    pipeline_ts, pipeline_p, pipeline_q = poses(pipeline_pose)
    common_ts = np.intersect1d(np.intersect1d(serial_ts, pipeline_ts), gt_ts)
    def select(ts, values):
        index = {int(t): i for i, t in enumerate(ts)}
        return values[[index[int(t)] for t in common_ts]]
    s_p, s_q = select(serial_ts, serial_p), select(serial_ts, serial_q)
    p_p, p_q = select(pipeline_ts, pipeline_p), select(pipeline_ts, pipeline_q)
    g_p, g_q = select(gt_ts, gt_p_all), select(gt_ts, gt_q_all)

    with (args.out / "serial_pipeline_per_frame.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = ["timestamp_ns", "serial_x", "serial_y", "serial_z", "pipeline_x", "pipeline_y", "pipeline_z", "gt_imu_x", "gt_imu_y", "gt_imu_z", "serial_pipeline_translation_diff", "serial_pipeline_rotation_diff_deg", "serial_position_error_norm", "pipeline_position_error_norm", "serial_orientation_error_deg", "pipeline_orientation_error_deg"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        sp_dq = angle_from_quat(quat_mul(quat_inv(s_q), p_q))
        for i, timestamp in enumerate(common_ts):
            writer.writerow({
                "timestamp_ns": int(timestamp),
                "serial_x": s_p[i, 0], "serial_y": s_p[i, 1], "serial_z": s_p[i, 2],
                "pipeline_x": p_p[i, 0], "pipeline_y": p_p[i, 1], "pipeline_z": p_p[i, 2],
                "gt_imu_x": g_p[i, 0], "gt_imu_y": g_p[i, 1], "gt_imu_z": g_p[i, 2],
                "serial_pipeline_translation_diff": float(np.linalg.norm(s_p[i] - p_p[i])),
                "serial_pipeline_rotation_diff_deg": float(np.rad2deg(sp_dq[i])),
                "serial_position_error_norm": float(np.linalg.norm(s_p[i] - g_p[i])),
                "pipeline_position_error_norm": float(np.linalg.norm(p_p[i] - g_p[i])),
                "serial_orientation_error_deg": float(np.rad2deg(angle_from_quat(quat_mul(quat_inv(g_q[i]), s_q[i])))),
                "pipeline_orientation_error_deg": float(np.rad2deg(angle_from_quat(quat_mul(quat_inv(g_q[i]), p_q[i])))),
            })

    serial_trace = serial_dir / "clear_stop_turn_rectangle_truth_normal_noise" / "pipeline_trace.csv"
    pipeline_trace = pipeline_dir / "clear_stop_turn_rectangle_truth_normal_noise" / "pipeline_trace.csv"
    static_serial = serial_dir / "clear_stop_turn_rectangle_truth_normal_noise" / "static_initialization.json"
    static_pipeline = pipeline_dir / "clear_stop_turn_rectangle_truth_normal_noise" / "static_initialization.json"
    summary = {
        "dataset": str(args.gt),
        "metadata_t_imu_camera_m": t_imu_camera.tolist(),
        "common_pose_rows": int(common_ts.size),
        "first_timestamp_ns": int(common_ts[0]),
        "last_timestamp_ns": int(common_ts[-1]),
        "serial_vs_pipeline": {
            "translation_diff": metric_block(np.linalg.norm(s_p - p_p, axis=1)),
            "rotation_diff_deg": metric_block(np.rad2deg(angle_from_quat(quat_mul(quat_inv(s_q), p_q)))),
            "all_finite": bool(np.isfinite(s_p).all() and np.isfinite(p_p).all() and np.isfinite(s_q).all() and np.isfinite(p_q).all()),
        },
        "serial_truth_metrics": trajectory_metrics(s_p, s_q, g_p, g_q),
        "pipeline_truth_metrics": trajectory_metrics(p_p, p_q, g_p, g_q),
        "serial_trace": trace_metrics(serial_trace),
        "pipeline_trace": trace_metrics(pipeline_trace),
        "static_initialization_serial": json.loads(static_serial.read_text(encoding="utf-8")),
        "static_initialization_pipeline": json.loads(static_pipeline.read_text(encoding="utf-8")),
        "contract_checks": {
            "first_active_edge_expected": [90, 91],
            "first_active_edge_serial": trace_metrics(serial_trace).get("first_edge"),
            "first_active_edge_pipeline": trace_metrics(pipeline_trace).get("first_edge"),
            "expected_active_edge_count": 209,
            "serial_edge_count": trace_metrics(serial_trace).get("rows"),
            "pipeline_edge_count": trace_metrics(pipeline_trace).get("rows"),
            "trajectory_reference": "IMU center via poses_imu.csv; GT camera-left converted with T_imu_camera",
        },
    }
    (args.out / "real_t2_pipeline_validation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
