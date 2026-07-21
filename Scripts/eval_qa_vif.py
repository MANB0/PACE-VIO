#!/usr/bin/env python3
"""Evaluate MACVO trajectories against HoloOcean reference poses."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from Utility.PoseFrame import convert_pose_frame


def _load_pose_table(path: str | Path) -> np.ndarray:
    path = Path(path)
    if path.suffix == ".npy":
        data = np.load(path, allow_pickle=True)
    elif path.suffix == ".csv":
        try:
            data = np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.float64)
        except ValueError:
            raw = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8")
            if raw.shape == ():
                raw = np.array([raw])
            if raw.dtype.names is None:
                raise
            cols = [raw[name].astype(np.float64) for name in raw.dtype.names[1:]]
            data = np.column_stack([np.arange(len(raw), dtype=np.float64), *cols])
    else:
        data = np.loadtxt(path, dtype=np.float64)
    if isinstance(data, np.ndarray) and data.ndim == 1:
        data = data.reshape(1, -1)
    return data


def _load_timed_pose_csv(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load timestamp_ns as int64 and pose columns as float64 from a CSV pose table."""
    raw = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    if raw.shape == ():
        raw = np.array([raw])
    if raw.dtype.names is None or len(raw.dtype.names) < 8:
        raise ValueError(f"Expected headered pose CSV with timestamp + 7 pose columns: {path}")
    time_ns = raw[raw.dtype.names[0]].astype(np.int64)
    poses = np.column_stack([raw[name].astype(np.float64) for name in raw.dtype.names[1:8]])
    return time_ns, poses


def infer_macvo_pose_frame(vo_result: str | Path) -> str:
    path = Path(vo_result)
    sidecar = path.parent / "pose_coordinate_frame.txt"
    if sidecar.exists():
        return sidecar.read_text(encoding="utf-8").strip().upper()
    if path.suffix == ".npy":
        return "NED"
    return "NWU"


def load_macvo_poses(
    vo_result: str | Path,
    source_frame: str = "auto",
    target_frame: str = "NWU",
) -> np.ndarray:
    """Load MACVO output poses as [tx, ty, tz, qx, qy, qz, qw] in target_frame."""
    data = _load_pose_table(vo_result)
    if isinstance(data, np.ndarray) and data.ndim == 2 and data.shape[1] >= 7:
        if data.shape[1] == 8:
            poses = data[:, 1:].astype(np.float64)
        else:
            poses = data[:, :7].astype(np.float64)
        source = infer_macvo_pose_frame(vo_result) if source_frame == "auto" else source_frame
        return convert_pose_frame(poses, source, target_frame)
    raise ValueError(f"Cannot parse MACVO result from {vo_result}. Shape: {getattr(data, 'shape', 'unknown')}")


def load_macvo_timed_poses(
    vo_result: str | Path,
    source_frame: str = "auto",
    target_frame: str = "NWU",
) -> tuple[np.ndarray, np.ndarray]:
    """Load MACVO poses with timestamps. Returns (timestamp_ns, poses_SE3)."""
    path = Path(vo_result)
    if path.suffix == ".csv":
        time_ns, poses = _load_timed_pose_csv(path)
        source = infer_macvo_pose_frame(vo_result) if source_frame == "auto" else source_frame
        return time_ns, convert_pose_frame(poses, source, target_frame)

    data = _load_pose_table(path)
    if not isinstance(data, np.ndarray) or data.ndim != 2 or data.shape[1] < 7:
        raise ValueError(f"Cannot parse MACVO result from {vo_result}. Shape: {getattr(data, 'shape', 'unknown')}")
    if data.shape[1] == 8:
        time_ns = data[:, 0].astype(np.int64)
        poses = data[:, 1:].astype(np.float64)
    else:
        time_ns = np.arange(data.shape[0], dtype=np.int64)
        poses = data[:, :7].astype(np.float64)
    source = infer_macvo_pose_frame(vo_result) if source_frame == "auto" else source_frame
    return time_ns, convert_pose_frame(poses, source, target_frame)


def load_ref_poses(csv_path: str | Path, target_frame: str = "NWU") -> np.ndarray:
    """Load HoloOcean ref_pose.csv. Native frame is NWU.
    Supports both old format (x_m, y_m, z_m) and new format (x, y, z)."""
    raw = np.genfromtxt(csv_path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    poses = np.zeros((len(raw), 7), dtype=np.float64)
    # Auto-detect field names: new format has 'x', old has 'x_m'
    names_lower = {n.lower(): n for n in raw.dtype.names}
    for axis, idx in [("x", 0), ("y", 1), ("z", 2)]:
        for candidate in [axis, f"{axis}_m"]:
            if candidate in names_lower:
                poses[:, idx] = raw[names_lower[candidate]]
                break
    poses[:, 6] = 1.0  # qw = 1 (identity quaternion for position-only GT)

    # Try to load GT orientation quaternion if available
    has_quat = all(names_lower.get(c) for c in ["qx", "qy", "qz", "qw"])
    if has_quat:
        qx = raw[names_lower["qx"]].astype(np.float64)
        qy = raw[names_lower["qy"]].astype(np.float64)
        qz = raw[names_lower["qz"]].astype(np.float64)
        qw = raw[names_lower["qw"]].astype(np.float64)
        # Normalize quaternions
        norms = np.sqrt(qx**2 + qy**2 + qz**2 + qw**2)
        qx /= norms; qy /= norms; qz /= norms; qw /= norms
        poses[:, 3] = qx
        poses[:, 4] = qy
        poses[:, 5] = qz
        poses[:, 6] = qw

    return convert_pose_frame(poses, "NWU", target_frame)


def load_ref_timed_poses(csv_path: str | Path, target_frame: str = "NWU") -> tuple[np.ndarray, np.ndarray]:
    """Load HoloOcean ref_pose.csv with timestamps. Native frame is NWU."""
    raw = np.genfromtxt(csv_path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    time_ns = raw["timestamp"].astype(np.int64) if "timestamp" in raw.dtype.names else np.arange(len(raw), dtype=np.int64)
    return time_ns, load_ref_poses(csv_path, target_frame)


def match_by_timestamp_or_index(
    est_time_ns: np.ndarray,
    est_poses: np.ndarray,
    ref_time_ns: np.ndarray,
    ref_poses: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Match trajectories by exact timestamp when possible; otherwise truncate by index."""
    common, est_idx, ref_idx = np.intersect1d(est_time_ns, ref_time_ns, assume_unique=False, return_indices=True)
    if common.size > 0:
        order = np.argsort(est_idx)
        est_idx = est_idx[order]
        ref_idx = ref_idx[order]
        common = common[order]
        return est_poses[est_idx], ref_poses[ref_idx], {
            "mode": "exact_timestamp",
            "n_matched": int(common.size),
            "first_timestamp_ns": int(common[0]),
            "last_timestamp_ns": int(common[-1]),
        }

    n_min = min(est_poses.shape[0], ref_poses.shape[0])
    return est_poses[:n_min], ref_poses[:n_min], {
        "mode": "index_truncate",
        "n_matched": int(n_min),
        "first_timestamp_ns": None,
        "last_timestamp_ns": None,
    }


def se3_to_matrix(pose: np.ndarray) -> np.ndarray:
    tx, ty, tz = pose[0], pose[1], pose[2]
    qx, qy, qz, qw = pose[3], pose[4], pose[5], pose[6]
    R = np.array([
        [1 - 2 * qy**2 - 2 * qz**2, 2 * qx * qy - 2 * qz * qw, 2 * qx * qz + 2 * qy * qw],
        [2 * qx * qy + 2 * qz * qw, 1 - 2 * qx**2 - 2 * qz**2, 2 * qy * qz - 2 * qx * qw],
        [2 * qx * qz - 2 * qy * qw, 2 * qy * qz + 2 * qx * qw, 1 - 2 * qx**2 - 2 * qy**2],
    ])
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [tx, ty, tz]
    return T


def compute_ate(est_poses: np.ndarray, gt_poses: np.ndarray) -> dict[str, float]:
    assert est_poses.shape == gt_poses.shape, f"Shape mismatch: {est_poses.shape} vs {gt_poses.shape}"
    est_xyz = est_poses[:, :3]
    gt_xyz = gt_poses[:, :3]

    est_mean = est_xyz.mean(axis=0)
    gt_mean = gt_xyz.mean(axis=0)
    est_centered = est_xyz - est_mean
    gt_centered = gt_xyz - gt_mean

    H = est_centered.T @ gt_centered
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    denom = np.trace(est_centered.T @ est_centered)
    scale = np.trace(gt_centered.T @ est_centered @ R) / denom if abs(denom) > 1e-12 else 1.0
    t = gt_mean - scale * R @ est_mean
    est_aligned = (scale * R @ est_xyz.T).T + t
    errors = np.linalg.norm(est_aligned - gt_xyz, axis=1)

    return {
        "ate_rmse": float(np.sqrt(np.mean(errors**2))),
        "ate_mean": float(np.mean(errors)),
        "ate_std": float(np.std(errors)),
        "ate_max": float(np.max(errors)),
        "scale": float(scale),
        "alignment": "umeyama_sim3",
    }


def compute_direct_ate(est_poses: np.ndarray, gt_poses: np.ndarray) -> dict[str, float | None | str]:
    """Absolute position error without rotation/translation/scale alignment."""
    assert est_poses.shape == gt_poses.shape, f"Shape mismatch: {est_poses.shape} vs {gt_poses.shape}"
    errors = np.linalg.norm(est_poses[:, :3] - gt_poses[:, :3], axis=1)
    return {
        "ate_rmse": float(np.sqrt(np.mean(errors**2))),
        "ate_mean": float(np.mean(errors)),
        "ate_std": float(np.std(errors)),
        "ate_max": float(np.max(errors)),
        "scale": None,
        "alignment": "none",
    }


def compute_rpe(est_poses: np.ndarray, gt_poses: np.ndarray, delta: int = 1) -> dict[str, float]:
    trans_errors = []
    rot_errors = []
    for i in range(est_poses.shape[0] - delta):
        T_est_rel = np.linalg.inv(se3_to_matrix(est_poses[i])) @ se3_to_matrix(est_poses[i + delta])
        T_gt_rel = np.linalg.inv(se3_to_matrix(gt_poses[i])) @ se3_to_matrix(gt_poses[i + delta])
        T_err = np.linalg.inv(T_est_rel) @ T_gt_rel
        trans_errors.append(np.linalg.norm(T_err[:3, 3]))
        rot_errors.append(np.arccos(max(-1.0, min(1.0, (np.trace(T_err[:3, :3]) - 1) / 2))))

    trans_errors = np.array(trans_errors)
    rot_errors = np.array(rot_errors)
    return {
        "rpe_trans_rmse": float(np.sqrt(np.mean(trans_errors**2))),
        "rpe_trans_mean": float(np.mean(trans_errors)),
        "rpe_rot_rmse_deg": float(np.sqrt(np.mean(rot_errors**2)) * 180 / np.pi),
        "rpe_rot_mean_deg": float(np.mean(rot_errors) * 180 / np.pi),
    }


def evaluate_trajectory(
    vo_result: str | Path,
    ref_pose: str | Path,
    name: str = "MACVO",
    vo_frame: str = "auto",
    eval_frame: str = "NWU",
) -> dict:
    est_poses = load_macvo_poses(vo_result, source_frame=vo_frame, target_frame=eval_frame)
    ref_poses = load_ref_poses(ref_pose, target_frame=eval_frame)
    n_min = min(est_poses.shape[0], ref_poses.shape[0])
    est_poses = est_poses[:n_min]
    ref_poses = ref_poses[:n_min]
    return {
        "name": name,
        "ate": compute_ate(est_poses, ref_poses),
        "rpe": compute_rpe(est_poses, ref_poses),
        "n_frames": int(n_min),
        "coordinate_frame": eval_frame.upper(),
        "vo_source_frame": infer_macvo_pose_frame(vo_result) if vo_frame == "auto" else vo_frame.upper(),
    }


def evaluate_trajectory_direct(
    vo_result: str | Path,
    ref_pose: str | Path,
    name: str = "MACVO",
    vo_frame: str = "auto",
    eval_frame: str = "NWU",
) -> dict:
    est_time, est_poses = load_macvo_timed_poses(vo_result, source_frame=vo_frame, target_frame=eval_frame)
    ref_time, ref_poses = load_ref_timed_poses(ref_pose, target_frame=eval_frame)
    est_poses, ref_poses, match_info = match_by_timestamp_or_index(est_time, est_poses, ref_time, ref_poses)
    return {
        "name": name,
        "ate": compute_direct_ate(est_poses, ref_poses),
        "rpe": compute_rpe(est_poses, ref_poses),
        "n_frames": int(est_poses.shape[0]),
        "coordinate_frame": eval_frame.upper(),
        "vo_source_frame": infer_macvo_pose_frame(vo_result) if vo_frame == "auto" else vo_frame.upper(),
        "alignment": "none",
        "matching": match_info,
    }


def save_plot(
    vo_result: str | Path,
    ref_pose: str | Path,
    name: str,
    plot_out: str | Path,
    vo_frame: str = "auto",
    eval_frame: str = "NWU",
) -> None:
    import matplotlib.pyplot as plt

    est_poses = load_macvo_poses(vo_result, source_frame=vo_frame, target_frame=eval_frame)
    ref_poses = load_ref_poses(ref_pose, target_frame=eval_frame)
    n_min = min(est_poses.shape[0], ref_poses.shape[0])
    est_poses = est_poses[:n_min]
    ref_poses = ref_poses[:n_min]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(ref_poses[:, 0], ref_poses[:, 1], "k-", linewidth=2, label="Reference")
    axes[0].plot(est_poses[:, 0], est_poses[:, 1], "r--", linewidth=1.5, label=name)
    axes[0].set_xlabel("X (m)")
    axes[0].set_ylabel("Y (m)")
    axes[0].set_title("Top-Down Trajectory (XY)")
    axes[0].legend()
    axes[0].axis("equal")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(ref_poses[:, 0], ref_poses[:, 2], "k-", linewidth=2, label="Reference")
    axes[1].plot(est_poses[:, 0], est_poses[:, 2], "r--", linewidth=1.5, label=name)
    axes[1].set_xlabel("X (m)")
    axes[1].set_ylabel("Z (m)")
    axes[1].set_title("Side View Trajectory (XZ)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_out, dpi=150)


def print_metrics(metrics: dict) -> None:
    ate = metrics["ate"]
    rpe = metrics["rpe"]
    print(f"\n{'=' * 60}")
    print(f"  QA-VIF Evaluation: {metrics['name']}")
    print(f"{'=' * 60}")
    print(f"\n  Coordinate frame: {metrics.get('coordinate_frame', 'unknown')}")
    print(f"  VO source frame:  {metrics.get('vo_source_frame', 'unknown')}")
    print("\n  Absolute Trajectory Error (ATE):")
    print(f"    RMSE:  {ate['ate_rmse']:.4f} m")
    print(f"    Mean:  {ate['ate_mean']:.4f} m")
    print(f"    Std:   {ate['ate_std']:.4f} m")
    print(f"    Max:   {ate['ate_max']:.4f} m")
    print(f"    Alignment: {ate.get('alignment', metrics.get('alignment', 'unknown'))}")
    if ate.get("scale") is not None:
        print(f"    Scale: {ate['scale']:.4f}")
    print("\n  Relative Pose Error (RPE, delta=1):")
    print(f"    Trans RMSE:  {rpe['rpe_trans_rmse']:.4f} m")
    print(f"    Trans Mean:  {rpe['rpe_trans_mean']:.4f} m")
    print(f"    Rot RMSE:    {rpe['rpe_rot_rmse_deg']:.4f} deg")
    print(f"    Rot Mean:    {rpe['rpe_rot_mean_deg']:.4f} deg")


def main() -> None:
    parser = argparse.ArgumentParser(description="HoloOcean MACVO trajectory evaluation")
    parser.add_argument("--vo_result", type=str, required=True, help="Path to MACVO poses.csv or legacy poses.npy")
    parser.add_argument("--ref_pose", type=str, required=True, help="Path to ref_pose.csv")
    parser.add_argument("--name", type=str, default="MACVO", help="Experiment name")
    parser.add_argument(
        "--vo_frame",
        type=str,
        default="auto",
        choices=["auto", "NED", "NWU", "ned", "nwu"],
        help="Coordinate frame of MACVO poses; auto uses pose_coordinate_frame.txt, or NED for legacy .npy, NWU for .csv.",
    )
    parser.add_argument(
        "--eval_frame",
        type=str,
        default="NWU",
        choices=["NED", "NWU", "ned", "nwu"],
        help="Coordinate frame used for metric computation. HoloOcean ref_pose.csv is native NWU.",
    )
    parser.add_argument(
        "--alignment",
        type=str,
        default="umeyama",
        choices=["umeyama", "direct"],
        help="ATE alignment mode: umeyama uses Sim(3), direct uses no rotation/translation/scale alignment.",
    )
    parser.add_argument("--metrics_out", type=str, default=None, help="Optional JSON metrics output path")
    parser.add_argument("--plot", action="store_true", help="Generate trajectory plot")
    parser.add_argument("--plot_out", type=str, default=None, help="Optional plot output path")
    args = parser.parse_args()

    if args.alignment == "direct":
        metrics = evaluate_trajectory_direct(args.vo_result, args.ref_pose, args.name, args.vo_frame, args.eval_frame)
    else:
        metrics = evaluate_trajectory(args.vo_result, args.ref_pose, args.name, args.vo_frame, args.eval_frame)
    print_metrics(metrics)

    if args.metrics_out:
        out_path = Path(args.metrics_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(f"\n  Metrics saved to: {out_path}")

    if args.plot:
        plot_out = Path(args.plot_out) if args.plot_out else Path(f"eval_{args.name}.png")
        save_plot(args.vo_result, args.ref_pose, args.name, plot_out, args.vo_frame, args.eval_frame)
        print(f"\n  Plot saved to: {plot_out}")


if __name__ == "__main__":
    main()
