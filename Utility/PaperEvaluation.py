"""Create a reproducible, full-sequence paper-evaluation bundle."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterable

import numpy as np


POSE_FIELDS = ("tx", "ty", "tz", "qx", "qy", "qz", "qw")
GT_POSITION_FIELDS = (("x", "y", "z"), ("x_m", "y_m", "z_m"))


@dataclass(frozen=True)
class PoseSeries:
    timestamps_ns: np.ndarray
    poses: np.ndarray
    frame_indices: np.ndarray

    def subset(self, indices: np.ndarray) -> "PoseSeries":
        return PoseSeries(
            self.timestamps_ns[indices],
            self.poses[indices],
            self.frame_indices[indices],
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _finite_or_none(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _integer(value: Any) -> int:
    """Parse decimal/scientific integer text without losing nanosecond bits."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(Decimal(str(value)))


def _summary(values: Iterable[float]) -> dict[str, float | int | None]:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {"count": 0, "median": None, "p95": None, "mean": None, "rmse": None}
    return {
        "count": int(array.size),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95.0)),
        "mean": float(np.mean(array)),
        "rmse": float(np.sqrt(np.mean(np.square(array)))),
    }


def _normalize_quaternions(quaternions: np.ndarray) -> np.ndarray:
    values = np.asarray(quaternions, dtype=np.float64)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norms <= 1.0e-15) or not np.isfinite(values).all():
        raise ValueError("trajectory contains invalid quaternions")
    return values / norms


def _quat_conjugate(q: np.ndarray) -> np.ndarray:
    result = np.asarray(q, dtype=np.float64).copy()
    result[..., :3] *= -1.0
    return result


def _quat_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    lx, ly, lz, lw = np.moveaxis(left, -1, 0)
    rx, ry, rz, rw = np.moveaxis(right, -1, 0)
    return np.stack(
        (
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
            lw * rw - lx * rx - ly * ry - lz * rz,
        ),
        axis=-1,
    )


def _quat_matrix(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    x, y, z, w = np.moveaxis(q, -1, 0)
    return np.stack(
        (
            1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
            2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
            2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
        ),
        axis=-1,
    ).reshape(q.shape[:-1] + (3, 3))


def _rotation_angle_deg(quaternions: np.ndarray) -> np.ndarray:
    q = _normalize_quaternions(np.asarray(quaternions).reshape(-1, 4))
    return np.degrees(2.0 * np.arccos(np.clip(np.abs(q[:, 3]), 0.0, 1.0)))


def _anchor(series: PoseSeries) -> PoseSeries:
    if series.poses.shape[0] == 0:
        return series
    poses = series.poses.copy()
    rotation0 = _quat_matrix(poses[0:1, 3:7])[0]
    poses[:, :3] = (rotation0.T @ (poses[:, :3] - poses[0, :3]).T).T
    poses[:, 3:7] = _quat_multiply(
        np.broadcast_to(_quat_conjugate(poses[0, 3:7]), poses[:, 3:7].shape),
        poses[:, 3:7],
    )
    poses[:, 3:7] = _normalize_quaternions(poses[:, 3:7])
    return PoseSeries(series.timestamps_ns.copy(), poses, series.frame_indices.copy())


def _relative(poses: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rotations = _quat_matrix(poses[:, 3:7])
    translation = np.einsum(
        "nij,nj->ni",
        np.swapaxes(rotations[:-1], 1, 2),
        poses[1:, :3] - poses[:-1, :3],
    )
    rotation = _quat_multiply(_quat_conjugate(poses[:-1, 3:7]), poses[1:, 3:7])
    return translation, _normalize_quaternions(rotation)


def _read_pose_csv(path: Path) -> PoseSeries:
    timestamps: list[int] = []
    poses: list[list[float]] = []
    frame_indices: list[int] = []
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        fields = set(reader.fieldnames or [])
        timestamp_field = "timestamp_ns" if "timestamp_ns" in fields else "timestamp"
        if timestamp_field not in fields:
            raise ValueError(f"{path} has no timestamp column")
        pose_fields = POSE_FIELDS
        if not set(pose_fields).issubset(fields):
            position_fields = next(
                (candidate for candidate in GT_POSITION_FIELDS if set(candidate).issubset(fields)),
                None,
            )
            if position_fields is None or not {"qx", "qy", "qz", "qw"}.issubset(fields):
                raise ValueError(f"{path} does not contain a complete pose")
            pose_fields = (*position_fields, "qx", "qy", "qz", "qw")
        for row_index, row in enumerate(reader):
            timestamps.append(_integer(row[timestamp_field]))
            poses.append([float(row[name]) for name in pose_fields])
            frame_indices.append(int(row.get("frame_idx", row_index)))
    pose_array = np.asarray(poses, dtype=np.float64).reshape(-1, 7)
    if pose_array.size:
        pose_array[:, 3:7] = _normalize_quaternions(pose_array[:, 3:7])
    return PoseSeries(
        np.asarray(timestamps, dtype=np.int64),
        pose_array,
        np.asarray(frame_indices, dtype=np.int64),
    )


def _write_pose_csv(path: Path, series: PoseSeries) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("frame_idx", "timestamp_ns", *POSE_FIELDS))
        for frame_idx, timestamp, pose in zip(
            series.frame_indices, series.timestamps_ns, series.poses
        ):
            writer.writerow(
                [int(frame_idx), int(timestamp), *[f"{float(value):.17g}" for value in pose]]
            )


def _newest_output_bundle(result_root: Path) -> Path:
    candidates = [
        path.parent
        for path in result_root.rglob("poses.csv")
        if path.parent.joinpath("pose_coordinate_frame.txt").is_file()
    ]
    if result_root.joinpath("poses.csv").is_file() and result_root.joinpath(
        "pose_coordinate_frame.txt"
    ).is_file():
        candidates.append(result_root)
    if not candidates:
        raise FileNotFoundError(f"No completed pose bundle found below {result_root}")
    with_diagnostics = [
        directory for directory in candidates
        if directory.joinpath("frame_pair_diagnostics.csv").is_file()
    ]
    pool = with_diagnostics or candidates
    return max(pool, key=lambda directory: directory.joinpath("poses.csv").stat().st_mtime_ns)


def _active_timestamp_range(bundle: Path, final: PoseSeries) -> tuple[int, int, str]:
    diagnostics = bundle / "frame_pair_diagnostics.csv"
    if diagnostics.is_file():
        valid: list[tuple[int, int]] = []
        with diagnostics.open("r", newline="", encoding="utf-8-sig") as stream:
            for row in csv.DictReader(stream):
                try:
                    frame_i = int(row["frame_i"])
                    frame_j = int(row["frame_j"])
                    timestamp_i = _integer(row.get("timestamp_i", ""))
                    timestamp_j = _integer(row.get("timestamp_j", ""))
                except (KeyError, TypeError, ValueError):
                    continue
                if frame_j > frame_i and timestamp_j > timestamp_i:
                    valid.append((timestamp_i, timestamp_j))
        if valid:
            return valid[0][0], valid[-1][1], "first_to_last_valid_diagnostic_edge"
    if final.timestamps_ns.size == 0:
        raise ValueError("final trajectory is empty")
    return int(final.timestamps_ns[0]), int(final.timestamps_ns[-1]), "complete_pose_export"


def _filter_range(series: PoseSeries, start_ns: int, end_ns: int) -> PoseSeries:
    indices = np.flatnonzero(
        (series.timestamps_ns >= int(start_ns)) & (series.timestamps_ns <= int(end_ns))
    )
    return series.subset(indices)


def _load_alignment(path: Path | None) -> dict[str, Any]:
    defaults = {
        "mode": "first_pose",
        "time_offset_ns": 0,
        "max_association_error_ns": 0,
        "fixed_rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
        "fixed_translation_m": [0.0, 0.0, 0.0],
        "scale": 1.0,
    }
    if path is not None:
        supplied = json.loads(Path(path).read_text(encoding="utf-8"))
        defaults.update(supplied)
    if defaults["mode"] not in {"first_pose", "none", "fixed_se3"}:
        raise ValueError("alignment mode must be first_pose, none or fixed_se3")
    if float(defaults.get("scale", 1.0)) != 1.0:
        raise ValueError("trajectory scale changes are forbidden")
    return defaults


def _apply_fixed_se3(series: PoseSeries, alignment: dict[str, Any]) -> PoseSeries:
    q = _normalize_quaternions(
        np.asarray(alignment["fixed_rotation_xyzw"], dtype=np.float64).reshape(1, 4)
    )[0]
    t = np.asarray(alignment["fixed_translation_m"], dtype=np.float64).reshape(3)
    rotation = _quat_matrix(q.reshape(1, 4))[0]
    poses = series.poses.copy()
    poses[:, :3] = (rotation @ poses[:, :3].T).T + t
    poses[:, 3:7] = _quat_multiply(
        np.broadcast_to(q, poses[:, 3:7].shape), poses[:, 3:7]
    )
    return PoseSeries(series.timestamps_ns.copy(), poses, series.frame_indices.copy())


def _associate(
    estimate: PoseSeries,
    ground_truth: PoseSeries,
    alignment: dict[str, Any],
) -> tuple[PoseSeries, PoseSeries, np.ndarray]:
    offset = int(alignment.get("time_offset_ns", 0))
    tolerance = int(alignment.get("max_association_error_ns", 0))
    gt_times = ground_truth.timestamps_ns
    if gt_times.size == 0:
        raise ValueError("ground-truth trajectory is empty")
    estimate_indices: list[int] = []
    gt_indices: list[int] = []
    errors: list[int] = []
    for index, timestamp in enumerate(estimate.timestamps_ns):
        target = int(timestamp) + offset
        insertion = int(np.searchsorted(gt_times, target))
        candidates = [candidate for candidate in (insertion - 1, insertion) if 0 <= candidate < gt_times.size]
        if not candidates:
            continue
        best = min(candidates, key=lambda candidate: abs(int(gt_times[candidate]) - target))
        error = int(gt_times[best]) - target
        if abs(error) <= tolerance:
            estimate_indices.append(index)
            gt_indices.append(best)
            errors.append(error)
    if len(estimate_indices) < 2:
        raise ValueError(
            "fewer than two estimate/GT timestamps were associated; provide an explicit "
            "time_offset_ns and max_association_error_ns when clocks are not identical"
        )
    return (
        estimate.subset(np.asarray(estimate_indices, dtype=np.int64)),
        ground_truth.subset(np.asarray(gt_indices, dtype=np.int64)),
        np.asarray(errors, dtype=np.int64),
    )


def _segment_ids(timestamps_ns: np.ndarray) -> np.ndarray:
    timestamps = np.asarray(timestamps_ns, dtype=np.int64)
    if timestamps.size == 0:
        return np.empty((0,), dtype=np.int64)
    if timestamps.size == 1:
        return np.zeros((1,), dtype=np.int64)
    intervals = np.diff(timestamps)
    positive = intervals[intervals > 0]
    if positive.size == 0:
        return np.arange(timestamps.size, dtype=np.int64)
    threshold = max(int(3.0 * float(np.median(positive))), 1)
    starts = np.concatenate(([False], intervals > threshold))
    return np.cumsum(starts, dtype=np.int64)


def _trajectory_metrics(
    estimate: PoseSeries,
    ground_truth: PoseSeries,
    association_errors_ns: np.ndarray,
    *,
    output_dir: Path,
    prefix: str,
    anchor_first: bool,
) -> dict[str, Any]:
    estimate_evaluated = _anchor(estimate) if anchor_first else estimate
    ground_truth_evaluated = _anchor(ground_truth) if anchor_first else ground_truth
    position_error = estimate_evaluated.poses[:, :3] - ground_truth_evaluated.poses[:, :3]
    xy_error = np.linalg.norm(position_error[:, :2], axis=1)
    xyz_error = np.linalg.norm(position_error, axis=1)
    rotation_error_q = _quat_multiply(
        _quat_conjugate(ground_truth_evaluated.poses[:, 3:7]),
        estimate_evaluated.poses[:, 3:7],
    )
    rotation_error_deg = _rotation_angle_deg(rotation_error_q)
    segment_ids = _segment_ids(ground_truth.timestamps_ns)

    state_path = output_dir / f"{prefix}_metrics_per_state.csv"
    with state_path.open("w", newline="", encoding="utf-8") as stream:
        fields = (
            "state_index", "frame_idx", "timestamp_est_ns", "timestamp_gt_ns",
            "association_error_ns", "segment_id", "error_x_m", "error_y_m", "error_z_m",
            "xy_error_m", "translation_error_m", "rotation_error_deg",
        )
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for index in range(estimate.poses.shape[0]):
            writer.writerow({
                "state_index": index,
                "frame_idx": int(estimate.frame_indices[index]),
                "timestamp_est_ns": int(estimate.timestamps_ns[index]),
                "timestamp_gt_ns": int(ground_truth.timestamps_ns[index]),
                "association_error_ns": int(association_errors_ns[index]),
                "segment_id": int(segment_ids[index]),
                "error_x_m": f"{position_error[index, 0]:.17g}",
                "error_y_m": f"{position_error[index, 1]:.17g}",
                "error_z_m": f"{position_error[index, 2]:.17g}",
                "xy_error_m": f"{xy_error[index]:.17g}",
                "translation_error_m": f"{xyz_error[index]:.17g}",
                "rotation_error_deg": f"{rotation_error_deg[index]:.17g}",
            })

    estimate_dt, estimate_dq = _relative(estimate_evaluated.poses)
    gt_dt, gt_dq = _relative(ground_truth_evaluated.poses)
    gt_delta_rotation = _quat_matrix(gt_dq)
    translation_rpe = np.einsum(
        "nij,nj->ni",
        np.swapaxes(gt_delta_rotation, 1, 2),
        estimate_dt - gt_dt,
    )
    translation_rpe_norm = np.linalg.norm(translation_rpe, axis=1)
    rotation_rpe_q = _quat_multiply(_quat_conjugate(gt_dq), estimate_dq)
    rotation_rpe_deg = _rotation_angle_deg(rotation_rpe_q)
    valid_edges = segment_ids[1:] == segment_ids[:-1]
    valid_edge_indices = np.flatnonzero(valid_edges)
    edge_path = output_dir / f"{prefix}_metrics_per_edge.csv"
    with edge_path.open("w", newline="", encoding="utf-8") as stream:
        fields = (
            "edge_index", "frame_i", "frame_j", "timestamp_i_ns", "timestamp_j_ns",
            "translation_rpe_m", "rotation_rpe_deg",
        )
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for output_index, index in enumerate(valid_edge_indices):
            writer.writerow({
                "edge_index": output_index,
                "frame_i": int(estimate.frame_indices[index]),
                "frame_j": int(estimate.frame_indices[index + 1]),
                "timestamp_i_ns": int(estimate.timestamps_ns[index]),
                "timestamp_j_ns": int(estimate.timestamps_ns[index + 1]),
                "translation_rpe_m": f"{translation_rpe_norm[index]:.17g}",
                "rotation_rpe_deg": f"{rotation_rpe_deg[index]:.17g}",
            })

    second_difference = (
        estimate_evaluated.poses[2:, :3]
        - 2.0 * estimate_evaluated.poses[1:-1, :3]
        + estimate_evaluated.poses[:-2, :3]
    )
    valid_triplets = (
        (segment_ids[2:] == segment_ids[1:-1])
        & (segment_ids[1:-1] == segment_ids[:-2])
    )
    second_difference_norm = np.linalg.norm(second_difference[valid_triplets], axis=1)
    return {
        "state_count": int(estimate.poses.shape[0]),
        "edge_count": int(valid_edge_indices.size),
        "segment_count": int(segment_ids[-1] + 1) if segment_ids.size else 0,
        "ape": {
            "xy_m": _summary(xy_error),
            "translation_m": _summary(xyz_error),
            "rotation_deg": _summary(rotation_error_deg),
        },
        "rpe": {
            "translation_m": _summary(translation_rpe_norm[valid_edges]),
            "rotation_deg": _summary(rotation_rpe_deg[valid_edges]),
        },
        "position_second_difference_rms_m": (
            None
            if second_difference_norm.size == 0
            else float(np.sqrt(np.mean(np.square(second_difference_norm))))
        ),
        "association_error_ns": _summary(np.abs(association_errors_ns)),
        "per_state_csv": state_path.name,
        "per_edge_csv": edge_path.name,
    }


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def _edge_key(row: dict[str, str]) -> tuple[int, int] | None:
    try:
        return int(row["frame_i"]), int(row["frame_j"])
    except (KeyError, TypeError, ValueError):
        return None


def _timing_and_solver_outputs(bundle: Path, result_root: Path, output_dir: Path) -> dict[str, Any]:
    diagnostic_rows = _read_csv_rows(bundle / "frame_pair_diagnostics.csv")
    pipeline_rows = _read_csv_rows(result_root / "pipeline_trace.csv")
    pipeline_by_edge = {_edge_key(row): row for row in pipeline_rows if _edge_key(row) is not None}
    timing_fields = (
        "edge_index", "frame_i", "frame_j", "timestamp_i_ns", "timestamp_j_ns",
        "frontend_ms", "factor_build_ms", "backend_update_ms", "backend_total_ms",
        "backend_commit_latency_ms", "commit_ms", "total_compute_ms",
        "backend", "converged",
    )
    timing_rows: list[dict[str, Any]] = []
    solver_rows: list[dict[str, Any]] = []
    for row in diagnostic_rows:
        key = _edge_key(row)
        if key is None or key[1] <= key[0]:
            continue
        pipeline = pipeline_by_edge.get(key, {})
        factor_build = _finite_or_none(row.get("local_ba_graph_build_s"))
        factor_build = None if factor_build is None else factor_build * 1000.0
        backend = str(row.get("vio_backend", "unknown"))
        if backend == "isam2":
            backend_update = _finite_or_none(row.get("isam2_update_ms"))
        else:
            backend_update_s = _finite_or_none(row.get("local_ba_lm_s"))
            backend_update = None if backend_update_s is None else backend_update_s * 1000.0
        backend_total_s = _finite_or_none(row.get("local_ba_optimize_total_s"))
        backend_total = None if backend_total_s is None else backend_total_s * 1000.0
        frontend = _finite_or_none(pipeline.get("frontend_ms"))
        wait = _finite_or_none(pipeline.get("backend_wait_ms"))
        commit = _finite_or_none(pipeline.get("commit_ms"))
        # backend_total is the complete backend call and already contains graph
        # construction and solver work.  Summing factor_build and update again
        # would double count them.  backend_wait_ms is an asynchronous commit
        # latency spanning the following frontend call, not another compute stage.
        compute_parts = (frontend, backend_total, commit)
        compute_total = sum(value for value in compute_parts if value is not None)
        converged_text = str(row.get("two_state_solver_converged", "")).strip().lower()
        converged = 1 if converged_text in {"1", "true", "yes"} else 0
        timing_rows.append({
            "edge_index": len(timing_rows),
            "frame_i": key[0],
            "frame_j": key[1],
            "timestamp_i_ns": row.get("timestamp_i", pipeline.get("timestamp_i_ns", "")),
            "timestamp_j_ns": row.get("timestamp_j", pipeline.get("timestamp_j_ns", "")),
            "frontend_ms": frontend,
            "factor_build_ms": factor_build,
            "backend_update_ms": backend_update,
            "backend_total_ms": backend_total,
            "backend_commit_latency_ms": wait,
            "commit_ms": commit,
            "total_compute_ms": compute_total,
            "backend": backend,
            "converged": converged,
        })
        solver_rows.append({
            "edge_index": len(solver_rows),
            "frame_i": key[0],
            "frame_j": key[1],
            "backend": backend,
            "converged": converged,
            "iterations": row.get("two_state_solver_iterations", ""),
            "reason": row.get("two_state_solver_convergence_reason", ""),
            "accepted_steps": row.get("two_state_solver_accepted_steps", ""),
            "rejected_steps": row.get("two_state_solver_rejected_steps", ""),
            "history_revision": row.get("isam2_history_revision", ""),
            "state_count": row.get("isam2_state_count", ""),
        })

    for filename, fields, rows in (
        ("timing_per_edge.csv", timing_fields, timing_rows),
        ("solver_status.csv", tuple(solver_rows[0]) if solver_rows else (
            "edge_index", "frame_i", "frame_j", "backend", "converged", "iterations",
            "reason", "accepted_steps", "rejected_steps", "history_revision", "state_count",
        ), solver_rows),
    ):
        with (output_dir / filename).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    timing_summary: dict[str, Any] = {}
    for field in timing_fields:
        if not field.endswith("_ms"):
            continue
        timing_summary[field] = _summary(
            float(row[field]) for row in timing_rows if row.get(field) is not None
        )
    timing_summary["edge_count"] = len(timing_rows)
    backends = sorted({str(row["backend"]) for row in timing_rows})
    timing_summary["backend"] = backends[0] if len(backends) == 1 else ",".join(backends)
    timing_summary["convergence_rate"] = (
        None if not timing_rows else float(np.mean([row["converged"] for row in timing_rows]))
    )
    execution_path = result_root / "run_execution.json"
    if execution_path.is_file():
        execution = json.loads(execution_path.read_text(encoding="utf-8"))
        runtime_s = _finite_or_none(execution.get("process_wall_runtime_s"))
        timing_summary["process_wall_runtime_s"] = runtime_s
        timing_summary["process_wall_time_per_active_edge_ms"] = (
            None
            if runtime_s is None or not timing_rows
            else 1000.0 * runtime_s / len(timing_rows)
        )
        timing_summary["effective_active_edge_rate_hz"] = (
            None
            if runtime_s is None or runtime_s <= 0.0
            else len(timing_rows) / runtime_s
        )
    return timing_summary


def _motion_outputs(
    bundle: Path,
    output_dir: Path,
    reference_path: Path | None,
) -> dict[str, Any]:
    rows = _read_csv_rows(bundle / "frame_pair_diagnostics.csv")
    fields = (
        "frame_i", "frame_j", "timestamp_i_ns", "timestamp_j_ns", "candidate", "active",
        "entered", "exited", "reason", "estimated_speed_m_s", "imu_angular_rate_rad_s",
        "visual_angular_rate_rad_s", "zero_translation_nis_per_dof", "reference_active",
    )
    reference_by_frame: dict[int, int] = {}
    reference_by_time: dict[int, int] = {}
    if reference_path is not None:
        for row in _read_csv_rows(reference_path):
            label = row.get("reference_active", row.get("label", row.get("active", "")))
            if str(label).strip().lower() in {"1", "true", "yes"}:
                value = 1
            elif str(label).strip().lower() in {"0", "false", "no"}:
                value = 0
            else:
                continue
            if row.get("frame_j") not in (None, ""):
                reference_by_frame[int(row["frame_j"])] = value
            if row.get("timestamp_j_ns") not in (None, ""):
                reference_by_time[_integer(row["timestamp_j_ns"])] = value

    output_rows: list[dict[str, Any]] = []
    for row in rows:
        key = _edge_key(row)
        if key is None or key[1] <= key[0]:
            continue
        timestamp_j = _integer(row.get("timestamp_j", 0) or 0)
        reference = reference_by_frame.get(key[1], reference_by_time.get(timestamp_j))
        output_rows.append({
            "frame_i": key[0],
            "frame_j": key[1],
            "timestamp_i_ns": row.get("timestamp_i", ""),
            "timestamp_j_ns": row.get("timestamp_j", ""),
            "candidate": row.get("near_zero_velocity_candidate", ""),
            "active": row.get("near_zero_velocity_active", ""),
            "entered": row.get("near_zero_velocity_entered", ""),
            "exited": row.get("near_zero_velocity_exited", ""),
            "reason": row.get("near_zero_velocity_reason", ""),
            "estimated_speed_m_s": row.get("near_zero_velocity_estimated_speed_m_s", ""),
            "imu_angular_rate_rad_s": row.get("near_zero_velocity_imu_angular_rate_rad_s", ""),
            "visual_angular_rate_rad_s": row.get("near_zero_velocity_visual_angular_rate_rad_s", ""),
            "zero_translation_nis_per_dof": row.get("near_zero_velocity_zero_translation_nis_per_dof", ""),
            "reference_active": "" if reference is None else reference,
        })
    with (output_dir / "motion_detection.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output_rows)

    paired = [row for row in output_rows if row["reference_active"] != ""]
    if not paired:
        confusion = {
            "available": False,
            "reason": "no explicit reference motion labels were supplied",
            "tp": None, "fn": None, "fp": None, "tn": None,
            "precision": None, "recall": None,
        }
    else:
        truth = np.asarray([int(row["reference_active"]) for row in paired])
        prediction = np.asarray([
            1 if str(row["active"]).strip().lower() in {"1", "true", "yes"} else 0
            for row in paired
        ])
        tp = int(np.sum((truth == 1) & (prediction == 1)))
        fn = int(np.sum((truth == 1) & (prediction == 0)))
        fp = int(np.sum((truth == 0) & (prediction == 1)))
        tn = int(np.sum((truth == 0) & (prediction == 0)))
        confusion = {
            "available": True,
            "tp": tp, "fn": fn, "fp": fp, "tn": tn,
            "precision": None if tp + fp == 0 else tp / (tp + fp),
            "recall": None if tp + fn == 0 else tp / (tp + fn),
        }
    _json(output_dir / "confusion_matrix.json", confusion)
    return confusion


def _dataset_manifest(dataset: Path) -> dict[str, Any]:
    metadata_path = dataset / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.is_file() else {}
    imu_path = dataset / "imu_data.csv"
    if not imu_path.is_file():
        imu_path = dataset / "imu.csv"
    gt_path = dataset / "ref_pose.csv"
    left_images = sorted(path for path in (dataset / "left").glob("*") if path.is_file())
    right_images = sorted(path for path in (dataset / "right").glob("*") if path.is_file())
    images = [*left_images, *right_images]
    image_digest = hashlib.sha256()
    for path in images:
        image_digest.update(path.relative_to(dataset).as_posix().encode("utf-8"))
        image_digest.update(bytes.fromhex(_sha256(path)))
    gt = _read_pose_csv(gt_path) if gt_path.is_file() else None

    def timestamp_summary(path: Path) -> dict[str, float | int | None]:
        if not path.is_file():
            return {"sample_count": 0, "duration_s": None, "median_frequency_hz": None}
        timestamps: list[int] = []
        with path.open("r", newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            fields = set(reader.fieldnames or ())
            timestamp_field = next(
                (name for name in ("timestamp_ns", "timestamp", "time_ns") if name in fields),
                None,
            )
            if timestamp_field is None:
                return {"sample_count": 0, "duration_s": None, "median_frequency_hz": None}
            for row in reader:
                value = row.get(timestamp_field, "")
                if value != "":
                    timestamps.append(_integer(value))
        values = np.asarray(timestamps, dtype=np.int64)
        positive_dt = np.diff(values)
        positive_dt = positive_dt[positive_dt > 0]
        return {
            "sample_count": int(values.size),
            "duration_s": (
                None if values.size < 2 else float((values[-1] - values[0]) * 1.0e-9)
            ),
            "median_frequency_hz": (
                None if positive_dt.size == 0
                else float(1.0e9 / np.median(positive_dt.astype(np.float64)))
            ),
        }

    imu_summary = timestamp_summary(imu_path)
    ground_truth_summary = timestamp_summary(gt_path)
    return {
        "schema_version": 2,
        "dataset_path": str(dataset),
        "metadata_sha256": _sha256(metadata_path) if metadata_path.is_file() else None,
        "imu_sha256": _sha256(imu_path) if imu_path.is_file() else None,
        "ground_truth_sha256": _sha256(gt_path) if gt_path.is_file() else None,
        "left_image_count": len(left_images),
        "right_image_count": len(right_images),
        "stereo_pair_count": min(len(left_images), len(right_images)),
        "stereo_image_file_count": len(images),
        "stereo_image_content_sha256": image_digest.hexdigest(),
        "camera_duration_s": ground_truth_summary["duration_s"],
        "camera_median_frequency_hz": ground_truth_summary["median_frequency_hz"],
        "imu_sample_count": imu_summary["sample_count"],
        "imu_duration_s": imu_summary["duration_s"],
        "imu_median_frequency_hz": imu_summary["median_frequency_hz"],
        "ground_truth_state_count": None if gt is None else int(gt.poses.shape[0]),
        "ground_truth_path_length_m": (
            None if gt is None or gt.poses.shape[0] < 2
            else float(np.linalg.norm(np.diff(gt.poses[:, :3], axis=0), axis=1).sum())
        ),
        "metadata_contract": {
            "ground_truth": metadata.get("ground_truth"),
            "camera": metadata.get("camera"),
            "imu": metadata.get("imu"),
            "extrinsics": metadata.get("extrinsics"),
        },
    }


def _run_manifest(root: Path, result_root: Path, bundle: Path) -> dict[str, Any]:
    def command(args: list[str]) -> str | None:
        try:
            return subprocess.check_output(args, cwd=root, text=True, stderr=subprocess.DEVNULL).strip()
        except Exception:
            return None

    try:
        import torch
        torch_info = {
            "version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }
    except Exception:
        torch_info = None
    config_hashes = {
        str(path.relative_to(result_root)): _sha256(path)
        for path in sorted(result_root.rglob("*.yaml"))
    }
    contract = result_root / "runtime_contract.json"
    return {
        "schema_version": 1,
        "project_root": str(root),
        "result_root": str(result_root),
        "source_bundle": str(bundle),
        "git_commit": command(["git", "rev-parse", "HEAD"]),
        "git_dirty": command(["git", "status", "--porcelain"]) not in (None, ""),
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "torch": torch_info,
        "runtime_contract": json.loads(contract.read_text(encoding="utf-8")) if contract.is_file() else None,
        "runtime_config_sha256": config_hashes,
    }


def export_paper_evaluation(
    *,
    project_root: Path,
    dataset_root: Path,
    result_root: Path,
    evaluation_dir: Path | None = None,
    alignment_path: Path | None = None,
    motion_reference_path: Path | None = None,
) -> Path:
    """Export final/online/raw trajectories and all paper-facing statistics."""
    project_root = Path(project_root).resolve()
    dataset_root = Path(dataset_root).resolve()
    result_root = Path(result_root).resolve()
    bundle = _newest_output_bundle(result_root)
    output_dir = (
        Path(evaluation_dir).resolve()
        if evaluation_dir is not None
        else result_root / "paper_evaluation"
    )
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    final = _read_pose_csv(bundle / "poses.csv")
    start_ns, end_ns, range_source = _active_timestamp_range(bundle, final)
    final = _filter_range(final, start_ns, end_ns)
    ground_truth_path = dataset_root / "ref_pose.csv"
    if not ground_truth_path.is_file():
        raise FileNotFoundError(f"Paper evaluation requires ground truth: {ground_truth_path}")
    ground_truth_all = _read_pose_csv(ground_truth_path)
    alignment = _load_alignment(alignment_path)
    if alignment["mode"] == "fixed_se3":
        final = _apply_fixed_se3(final, alignment)
    final_paired, gt_paired, association_error = _associate(final, ground_truth_all, alignment)
    if alignment["mode"] == "first_pose":
        evaluation_note = "estimate and GT independently anchored at first evaluated state"
    elif alignment["mode"] == "fixed_se3":
        evaluation_note = "explicit fixed SE(3) applied; scale fixed to one"
    else:
        evaluation_note = "no spatial alignment"

    _write_pose_csv(output_dir / "poses_final.csv", final)
    _write_pose_csv(output_dir / "ground_truth.csv", ground_truth_all)
    variants: dict[str, PoseSeries] = {"final": final}
    source_candidates = {
        "online": bundle / "poses_online.csv",
        "macvo_raw": bundle / "macvo_raw_poses_imu.csv",
    }
    for name, path in source_candidates.items():
        if not path.is_file():
            continue
        series = _filter_range(_read_pose_csv(path), start_ns, end_ns)
        if alignment["mode"] == "fixed_se3":
            series = _apply_fixed_se3(series, alignment)
        variants[name] = series
        _write_pose_csv(output_dir / f"poses_{name}.csv", series)

    trajectory_summaries: dict[str, Any] = {}
    anchor_first = alignment["mode"] == "first_pose"
    for name, series in variants.items():
        paired_estimate, paired_gt, errors = _associate(series, ground_truth_all, alignment)
        evaluated_estimate = _anchor(paired_estimate) if anchor_first else paired_estimate
        evaluated_gt = _anchor(paired_gt) if anchor_first else paired_gt
        _write_pose_csv(output_dir / f"poses_{name}_evaluated.csv", evaluated_estimate)
        _write_pose_csv(output_dir / f"ground_truth_{name}_evaluated.csv", evaluated_gt)
        trajectory_summaries[name] = _trajectory_metrics(
            paired_estimate,
            paired_gt,
            errors,
            output_dir=output_dir,
            prefix=name,
            anchor_first=anchor_first,
        )
    shutil.copy2(output_dir / "poses_final_evaluated.csv", output_dir / "trajectory_evaluated.csv")
    shutil.copy2(
        output_dir / "ground_truth_final_evaluated.csv",
        output_dir / "ground_truth_evaluated.csv",
    )
    shutil.copy2(output_dir / "final_metrics_per_state.csv", output_dir / "metrics_per_state.csv")
    shutil.copy2(output_dir / "final_metrics_per_edge.csv", output_dir / "metrics_per_edge.csv")

    with (output_dir / "evaluation_pairs.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = (
            "state_index", "frame_idx", "timestamp_est_ns", "timestamp_gt_ns",
            "association_error_ns", "segment_id", "anchor_state",
        )
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        final_segment_ids = _segment_ids(gt_paired.timestamps_ns)
        for index in range(final_paired.poses.shape[0]):
            writer.writerow({
                "state_index": index,
                "frame_idx": int(final_paired.frame_indices[index]),
                "timestamp_est_ns": int(final_paired.timestamps_ns[index]),
                "timestamp_gt_ns": int(gt_paired.timestamps_ns[index]),
                "association_error_ns": int(association_error[index]),
                "segment_id": int(final_segment_ids[index]),
                "anchor_state": int(index == 0),
            })

    timing = _timing_and_solver_outputs(bundle, result_root, output_dir)
    confusion = _motion_outputs(bundle, output_dir, motion_reference_path)
    dataset_manifest = _dataset_manifest(dataset_root)
    run_manifest = _run_manifest(project_root, result_root, bundle)
    evaluation_contract = {
        "schema_version": 2,
        "active_range": {
            "start_timestamp_ns": start_ns,
            "end_timestamp_ns": end_ns,
            "duration_s": float((end_ns - start_ns) * 1.0e-9),
            "first_frame_idx": int(final_paired.frame_indices[0]),
            "last_frame_idx": int(final_paired.frame_indices[-1]),
            "evaluated_state_count": int(final_paired.poses.shape[0]),
            "evaluated_edge_count": int(trajectory_summaries["final"]["edge_count"]),
            "source": range_source,
            "hard_coded_edge_count": False,
        },
        "alignment": alignment,
        "evaluation_note": evaluation_note,
        "canonical_reference_point": "IMU center",
        "canonical_world_frame": (bundle / "pose_coordinate_frame.txt").read_text(encoding="utf-8").strip(),
        "metrics_trajectory": "complete-sequence final export",
        "history_revision_policy": (
            "final export preserves each backend's native history-revision mechanism; "
            "poses_online.csv records the causal committed state"
        ),
    }
    metrics_summary = {
        "schema_version": 1,
        "trajectories": trajectory_summaries,
        "timing": timing,
        "motion_detection": confusion,
    }
    _json(output_dir / "dataset_manifest.json", dataset_manifest)
    _json(output_dir / "run_manifest.json", run_manifest)
    _json(output_dir / "evaluation_alignment.json", evaluation_contract)
    _json(output_dir / "metrics_summary.json", metrics_summary)
    flat_fields = (
        "trajectory", "state_count", "edge_count", "xy_ape_rmse_m", "xy_ape_p95_m",
        "translation_ape_rmse_m", "translation_ape_p95_m", "rotation_ape_rmse_deg",
        "rotation_ape_p95_deg", "translation_rpe_rmse_m", "translation_rpe_p95_m",
        "rotation_rpe_rmse_deg", "rotation_rpe_p95_deg",
        "position_second_difference_rms_m",
    )
    with (output_dir / "metrics_summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=flat_fields)
        writer.writeheader()
        for name, values in trajectory_summaries.items():
            writer.writerow({
                "trajectory": name,
                "state_count": values["state_count"],
                "edge_count": values["edge_count"],
                "xy_ape_rmse_m": values["ape"]["xy_m"]["rmse"],
                "xy_ape_p95_m": values["ape"]["xy_m"]["p95"],
                "translation_ape_rmse_m": values["ape"]["translation_m"]["rmse"],
                "translation_ape_p95_m": values["ape"]["translation_m"]["p95"],
                "rotation_ape_rmse_deg": values["ape"]["rotation_deg"]["rmse"],
                "rotation_ape_p95_deg": values["ape"]["rotation_deg"]["p95"],
                "translation_rpe_rmse_m": values["rpe"]["translation_m"]["rmse"],
                "translation_rpe_p95_m": values["rpe"]["translation_m"]["p95"],
                "rotation_rpe_rmse_deg": values["rpe"]["rotation_deg"]["rmse"],
                "rotation_rpe_p95_deg": values["rpe"]["rotation_deg"]["p95"],
                "position_second_difference_rms_m": values["position_second_difference_rms_m"],
            })
    run_fields = (
        "backend", "edge_count", "frontend_median_ms", "frontend_p95_ms",
        "factor_build_median_ms", "factor_build_p95_ms", "backend_update_median_ms",
        "backend_update_p95_ms", "backend_total_median_ms", "backend_total_p95_ms",
        "total_compute_median_ms", "total_compute_p95_ms", "convergence_rate",
        "process_wall_runtime_s", "effective_active_edge_rate_hz", "detector_available",
        "tp", "fn", "fp", "tn", "precision", "recall",
    )

    def timing_value(field: str, statistic: str) -> float | int | None:
        values = timing.get(field, {})
        return values.get(statistic) if isinstance(values, dict) else None

    with (output_dir / "run_summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=run_fields)
        writer.writeheader()
        writer.writerow({
            "backend": timing.get("backend"),
            "edge_count": timing.get("edge_count"),
            "frontend_median_ms": timing_value("frontend_ms", "median"),
            "frontend_p95_ms": timing_value("frontend_ms", "p95"),
            "factor_build_median_ms": timing_value("factor_build_ms", "median"),
            "factor_build_p95_ms": timing_value("factor_build_ms", "p95"),
            "backend_update_median_ms": timing_value("backend_update_ms", "median"),
            "backend_update_p95_ms": timing_value("backend_update_ms", "p95"),
            "backend_total_median_ms": timing_value("backend_total_ms", "median"),
            "backend_total_p95_ms": timing_value("backend_total_ms", "p95"),
            "total_compute_median_ms": timing_value("total_compute_ms", "median"),
            "total_compute_p95_ms": timing_value("total_compute_ms", "p95"),
            "convergence_rate": timing.get("convergence_rate"),
            "process_wall_runtime_s": timing.get("process_wall_runtime_s"),
            "effective_active_edge_rate_hz": timing.get("effective_active_edge_rate_hz"),
            "detector_available": confusion.get("available"),
            "tp": confusion.get("tp"),
            "fn": confusion.get("fn"),
            "fp": confusion.get("fp"),
            "tn": confusion.get("tn"),
            "precision": confusion.get("precision"),
            "recall": confusion.get("recall"),
        })
    (output_dir / "README.md").write_text(
        "# PACE-VIO paper evaluation bundle\n\n"
        "All state and edge counts are derived from the complete active sequence. "
        "`poses_final.csv` is the post-sequence trajectory used for APE/RPE; "
        "`poses_online.csv` is the causal committed trajectory when available. "
        "No scale change is permitted. See `evaluation_alignment.json` for the exact "
        "time and spatial alignment contract.\n",
        encoding="utf-8",
    )
    (result_root / "paper_evaluation_path.txt").write_text(str(output_dir) + "\n", encoding="utf-8")
    return output_dir
