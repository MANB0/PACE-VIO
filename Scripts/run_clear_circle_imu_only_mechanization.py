#!/usr/bin/env python3
"""Run IMU-only strapdown mechanization on clear-circle recordings.

The experiment initializes pose and velocity from the first GT frame, then
propagates the full sequence using only imu_data.csv.  It is intentionally
separate from MACVO and the optimizer.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

WORKDIR = Path("/home/admin1/macvo-dev")
if str(WORKDIR) not in sys.path:
    sys.path.insert(0, str(WORKDIR))

import Scripts.analyse_clear_circle_pair_vio as pair_analysis
import Scripts.diagnose_imu_vio_residuals as diag
import Scripts.evaluate_macvo_relative_metrics as rel_metrics


DEFAULT_OUTPUT_ROOT = WORKDIR / "analysis_clear_circle_imu_only_mechanization_20260704"
DEFAULT_SCENE_ROOTS = {
    "clear_circle_normal_noise": Path(
        "/mnt/e/文档/holoocean/code/recordings/"
        "batch_clear_circle_pair_20260704/normal_noise/clear_circle_path"
    ),
    "clear_circle_zero_noise": Path(
        "/mnt/e/文档/holoocean/code/recordings/"
        "batch_clear_circle_pair_20260704/zero_noise/clear_circle_path"
    ),
}
DEFAULT_EXISTING_RESULT_ROOTS = [
    WORKDIR / "Results" / "clear_circle_pair_vio_20260704_fixed",
    WORKDIR / "Results" / "clear_circle_pair_pure_macvo_20260704",
]

METHOD_COLORS = {
    "clear_circle_normal_noise / imu_only": "#e31a1c",
    "clear_circle_zero_noise / imu_only": "#fb9a99",
    "clear_circle_normal_noise / vio_preintegrated_full": "#d95f02",
    "clear_circle_zero_noise / vio_preintegrated_full": "#1b9e77",
    "clear_circle_normal_noise / pure_macvo": "#9467bd",
    "clear_circle_zero_noise / pure_macvo": "#1f78b4",
}


@dataclass
class MechanizedStates:
    time_ns: np.ndarray
    position_w: np.ndarray
    velocity_w: np.ndarray
    R_bw: np.ndarray


def _orthonormalize_rotation(R: np.ndarray) -> np.ndarray:
    U, _, Vt = np.linalg.svd(R)
    out = U @ Vt
    if np.linalg.det(out) < 0.0:
        U[:, -1] *= -1.0
        out = U @ Vt
    return out


def matrix_to_quat_xyzw(R: np.ndarray) -> np.ndarray:
    """Convert a rotation matrix to xyzw quaternion."""
    m = np.asarray(R, dtype=np.float64).reshape(3, 3)
    trace = float(np.trace(m))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (m[2, 1] - m[1, 2]) / s
        qy = (m[0, 2] - m[2, 0]) / s
        qz = (m[1, 0] - m[0, 1]) / s
    else:
        idx = int(np.argmax(np.diag(m)))
        if idx == 0:
            s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
            qw = (m[2, 1] - m[1, 2]) / s
            qx = 0.25 * s
            qy = (m[0, 1] + m[1, 0]) / s
            qz = (m[0, 2] + m[2, 0]) / s
        elif idx == 1:
            s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
            qw = (m[0, 2] - m[2, 0]) / s
            qx = (m[0, 1] + m[1, 0]) / s
            qy = 0.25 * s
            qz = (m[1, 2] + m[2, 1]) / s
        else:
            s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
            qw = (m[1, 0] - m[0, 1]) / s
            qx = (m[0, 2] + m[2, 0]) / s
            qy = (m[1, 2] + m[2, 1]) / s
            qz = 0.25 * s
    q = np.array([qx, qy, qz, qw], dtype=np.float64)
    if q[3] < 0.0:
        q *= -1.0
    return q / max(np.linalg.norm(q), 1e-15)


def mechanize_imu_nwu(
    *,
    time_ns: np.ndarray,
    acc_b: np.ndarray,
    gyro_b: np.ndarray,
    p0: np.ndarray,
    R0_bw: np.ndarray,
    v0_w: np.ndarray,
    gravity: float = 9.8,
) -> MechanizedStates:
    """Propagate NWU pose/velocity with raw body-frame IMU samples.

    HoloOcean records a specific-force accelerometer reading.  In NWU, gravity
    is ``[0, 0, -g]`` in world coordinates, so world acceleration is
    ``R_bw @ acc_b + g_w``.
    """
    time_ns = np.asarray(time_ns, dtype=np.int64)
    acc_b = np.asarray(acc_b, dtype=np.float64).reshape(-1, 3)
    gyro_b = np.asarray(gyro_b, dtype=np.float64).reshape(-1, 3)
    if len(time_ns) != len(acc_b) or len(time_ns) != len(gyro_b):
        raise ValueError("time_ns, acc_b, and gyro_b must have the same length")
    if len(time_ns) == 0:
        raise ValueError("At least one IMU sample is required")

    n = len(time_ns)
    positions = np.zeros((n, 3), dtype=np.float64)
    velocities = np.zeros((n, 3), dtype=np.float64)
    rotations = np.zeros((n, 3, 3), dtype=np.float64)

    p = np.asarray(p0, dtype=np.float64).reshape(3).copy()
    v = np.asarray(v0_w, dtype=np.float64).reshape(3).copy()
    R = _orthonormalize_rotation(np.asarray(R0_bw, dtype=np.float64).reshape(3, 3))
    g_w = np.array([0.0, 0.0, -abs(float(gravity))], dtype=np.float64)

    positions[0] = p
    velocities[0] = v
    rotations[0] = R

    for k in range(n - 1):
        dt = max(float(time_ns[k + 1] - time_ns[k]) * 1e-9, 0.0)
        gyro_mid = 0.5 * (gyro_b[k] + gyro_b[k + 1])
        acc_mid = 0.5 * (acc_b[k] + acc_b[k + 1])
        dR = diag.so3_exp(gyro_mid * dt)
        R_mid = R @ diag.so3_exp(gyro_mid * (0.5 * dt))
        a_w = R_mid @ acc_mid + g_w

        p = p + v * dt + 0.5 * a_w * dt * dt
        v = v + a_w * dt
        R = _orthonormalize_rotation(R @ dR)

        positions[k + 1] = p
        velocities[k + 1] = v
        rotations[k + 1] = R

    return MechanizedStates(time_ns=time_ns.copy(), position_w=positions, velocity_w=velocities, R_bw=rotations)


def load_camera_to_imu_translation_nwu(scene_root: Path) -> np.ndarray:
    meta_path = scene_root / "metadata.json"
    if not meta_path.exists():
        return np.zeros(3, dtype=np.float64)
    with meta_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)
    extrinsics = metadata.get("extrinsics", {})
    body_imu = extrinsics.get("T_body_imu", {}).get("translation_body_nwu_m")
    body_camera = extrinsics.get("T_body_camera", {}).get("translation_body_nwu_m")
    if body_imu and body_camera and len(body_imu) == 3 and len(body_camera) == 3:
        return np.asarray(body_imu, dtype=np.float64) - np.asarray(body_camera, dtype=np.float64)
    imu_camera = extrinsics.get("T_imu_camera", {}).get("translation_body_nwu_m")
    if imu_camera and len(imu_camera) == 3:
        return -np.asarray(imu_camera, dtype=np.float64)
    return np.zeros(3, dtype=np.float64)


def convert_camera_state_to_imu_initial(
    *,
    camera_position_w: np.ndarray,
    camera_velocity_w: np.ndarray,
    R_bw: np.ndarray,
    omega_body: np.ndarray,
    camera_to_imu_body: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    r_b = np.asarray(camera_to_imu_body, dtype=np.float64).reshape(3)
    R = np.asarray(R_bw, dtype=np.float64).reshape(3, 3)
    omega_b = np.asarray(omega_body, dtype=np.float64).reshape(3)
    p_imu_w = np.asarray(camera_position_w, dtype=np.float64).reshape(3) + R @ r_b
    v_imu_w = np.asarray(camera_velocity_w, dtype=np.float64).reshape(3) + R @ np.cross(omega_b, r_b)
    return p_imu_w, v_imu_w


def convert_imu_states_to_camera_point(
    states: MechanizedStates,
    camera_to_imu_body: np.ndarray,
) -> MechanizedStates:
    r_b = np.asarray(camera_to_imu_body, dtype=np.float64).reshape(3)
    positions = states.position_w - np.einsum("nij,j->ni", states.R_bw, r_b)
    velocities = states.velocity_w.copy()
    if len(states.time_ns) > 1:
        t_s = states.time_ns.astype(np.float64) * 1e-9
        velocities = np.gradient(positions, t_s, axis=0, edge_order=1)
    return MechanizedStates(
        time_ns=states.time_ns.copy(),
        position_w=positions,
        velocity_w=velocities,
        R_bw=states.R_bw.copy(),
    )


def load_imu_csv(scene_root: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    imu = np.genfromtxt(scene_root / "imu_data.csv", delimiter=",", names=True)
    time_ns = imu["timestamp"].astype(np.int64)
    acc = np.stack([imu["lin_acc_x"], imu["lin_acc_y"], imu["lin_acc_z"]], axis=1).astype(np.float64)
    gyro = np.stack([imu["ang_vel_x"], imu["ang_vel_y"], imu["ang_vel_z"]], axis=1).astype(np.float64)
    return time_ns, acc, gyro


def load_ref_pose(scene_root: Path) -> pd.DataFrame:
    df = pd.read_csv(scene_root / "ref_pose.csv")
    if "timestamp" in df.columns:
        df = df.rename(columns={"timestamp": "timestamp_ns"})
    return df


def sample_states_at_times(states: MechanizedStates, target_time_ns: np.ndarray) -> MechanizedStates:
    target_time_ns = np.asarray(target_time_ns, dtype=np.int64)
    positions = np.zeros((len(target_time_ns), 3), dtype=np.float64)
    velocities = np.zeros((len(target_time_ns), 3), dtype=np.float64)
    rotations = np.zeros((len(target_time_ns), 3, 3), dtype=np.float64)

    for i, ts in enumerate(target_time_ns):
        idx = int(np.searchsorted(states.time_ns, ts, side="left"))
        if idx < len(states.time_ns) and int(states.time_ns[idx]) == int(ts):
            positions[i] = states.position_w[idx]
            velocities[i] = states.velocity_w[idx]
            rotations[i] = states.R_bw[idx]
            continue
        if idx <= 0:
            positions[i] = states.position_w[0]
            velocities[i] = states.velocity_w[0]
            rotations[i] = states.R_bw[0]
            continue
        if idx >= len(states.time_ns):
            positions[i] = states.position_w[-1]
            velocities[i] = states.velocity_w[-1]
            rotations[i] = states.R_bw[-1]
            continue
        t0 = float(states.time_ns[idx - 1])
        t1 = float(states.time_ns[idx])
        alpha = (float(ts) - t0) / max(t1 - t0, 1.0)
        positions[i] = (1.0 - alpha) * states.position_w[idx - 1] + alpha * states.position_w[idx]
        velocities[i] = (1.0 - alpha) * states.velocity_w[idx - 1] + alpha * states.velocity_w[idx]
        rotations[i] = states.R_bw[idx if alpha >= 0.5 else idx - 1]

    return MechanizedStates(target_time_ns.copy(), positions, velocities, rotations)


def write_poses_csv(path: Path, states: MechanizedStates) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp_ns", "tx", "ty", "tz", "qx", "qy", "qz", "qw", "vx", "vy", "vz"])
        for ts, p, v, R in zip(states.time_ns, states.position_w, states.velocity_w, states.R_bw):
            q = matrix_to_quat_xyzw(R)
            writer.writerow([int(ts), *p.tolist(), *q.tolist(), *v.tolist()])


def evaluate_joined(label: str, scene: str, method: str, joined: pd.DataFrame, source: str) -> dict[str, object]:
    est = joined[["tx_est", "ty_est", "tz_est"]].to_numpy(float)
    gt = joined[["tx_gt", "ty_gt", "tz_gt"]].to_numpy(float)
    err = np.linalg.norm(est - gt, axis=1)
    path_len = float(np.linalg.norm(np.diff(gt, axis=0), axis=1).sum()) if len(gt) > 1 else 0.0
    row = {
        "scene": scene,
        "method": method,
        "label": label,
        "matched_frames": int(len(joined)),
        "ate_rmse_m": float(np.sqrt(np.mean(err**2))),
        "ate_median_m": float(np.median(err)),
        "ate_final_m": float(err[-1]),
        "ate_max_m": float(np.max(err)),
        "gt_path_length_m": path_len,
        "source": source,
    }
    row.update(relative_velocity_metrics_from_joined(joined))
    return row


def _stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": math.nan, "median": math.nan, "max": math.nan}
    arr = np.asarray(values, dtype=np.float64)
    return {"mean": float(np.mean(arr)), "median": float(np.median(arr)), "max": float(np.max(arr))}


def relative_velocity_metrics_from_joined(joined: pd.DataFrame) -> dict[str, float]:
    est_pos = joined[["tx_est", "ty_est", "tz_est"]].to_numpy(float)
    gt_pos = joined[["tx_gt", "ty_gt", "tz_gt"]].to_numpy(float)
    timestamps = joined["timestamp_ns"].to_numpy(np.int64)
    quat_cols = {"qx_est", "qy_est", "qz_est", "qw_est", "qx_gt", "qy_gt", "qz_gt", "qw_gt"}
    has_quat = quat_cols.issubset(joined.columns)

    t_errors: list[float] = []
    r_errors_deg: list[float] = []
    t_vel_errors: list[float] = []
    r_vel_errors_deg_s: list[float] = []

    for i in range(len(joined) - 1):
        dt_s = float(timestamps[i + 1] - timestamps[i]) * 1e-9
        if dt_s <= 0.0:
            continue

        gt_delta = tuple((gt_pos[i + 1] - gt_pos[i]).tolist())
        est_delta = tuple((est_pos[i + 1] - est_pos[i]).tolist())
        if has_quat:
            g0_r = rel_metrics.qxyzw_to_rot(
                float(joined["qx_gt"].iloc[i]),
                float(joined["qy_gt"].iloc[i]),
                float(joined["qz_gt"].iloc[i]),
                float(joined["qw_gt"].iloc[i]),
            )
            e0_r = rel_metrics.qxyzw_to_rot(
                float(joined["qx_est"].iloc[i]),
                float(joined["qy_est"].iloc[i]),
                float(joined["qz_est"].iloc[i]),
                float(joined["qw_est"].iloc[i]),
            )
            est_delta_in_gt_world = rel_metrics.mat_vec(g0_r, rel_metrics.mat_vec(rel_metrics.mat_t(e0_r), est_delta))
            t_err = rel_metrics.vec_norm(rel_metrics.vec_sub(gt_delta, est_delta_in_gt_world))

            g1_r = rel_metrics.qxyzw_to_rot(
                float(joined["qx_gt"].iloc[i + 1]),
                float(joined["qy_gt"].iloc[i + 1]),
                float(joined["qz_gt"].iloc[i + 1]),
                float(joined["qw_gt"].iloc[i + 1]),
            )
            e1_r = rel_metrics.qxyzw_to_rot(
                float(joined["qx_est"].iloc[i + 1]),
                float(joined["qy_est"].iloc[i + 1]),
                float(joined["qz_est"].iloc[i + 1]),
                float(joined["qw_est"].iloc[i + 1]),
            )
            gt_rel = rel_metrics.mat_mul(rel_metrics.mat_t(g0_r), g1_r)
            est_rel = rel_metrics.mat_mul(rel_metrics.mat_t(e0_r), e1_r)
            r_err = rel_metrics.mat_mul(rel_metrics.mat_t(est_rel), gt_rel)
            r_err_deg = math.degrees(rel_metrics.rot_angle_rad(r_err))
            r_errors_deg.append(r_err_deg)
            r_vel_errors_deg_s.append(r_err_deg / dt_s)
        else:
            t_err = float(np.linalg.norm(np.asarray(gt_delta) - np.asarray(est_delta)))

        t_errors.append(t_err)
        t_vel_errors.append(t_err / dt_s)

    t_rel = _stats(t_errors)
    r_rel = _stats(r_errors_deg)
    t_vel = _stats(t_vel_errors)
    r_vel = _stats(r_vel_errors_deg_s)
    return {
        "t_rel_m_per_frame": t_rel["mean"],
        "t_rel_m_per_frame_median": t_rel["median"],
        "t_rel_m_per_frame_max": t_rel["max"],
        "r_rel_deg_per_frame": r_rel["mean"],
        "r_rel_deg_per_frame_median": r_rel["median"],
        "r_rel_deg_per_frame_max": r_rel["max"],
        "t_vel_m_s": t_vel["mean"],
        "t_vel_m_s_median": t_vel["median"],
        "t_vel_m_s_max": t_vel["max"],
        "r_vel_deg_s": r_vel["mean"],
        "r_vel_deg_s_median": r_vel["median"],
        "r_vel_deg_s_max": r_vel["max"],
    }


def run_imu_only_for_scene(scene: str, scene_root: Path, output_root: Path) -> tuple[dict[str, object], pd.DataFrame]:
    ref = load_ref_pose(scene_root)
    time_ns, acc, gyro = load_imu_csv(scene_root)

    first = ref.iloc[0]
    p0_camera = np.array([first["x"], first["y"], first["z"]], dtype=np.float64)
    R0 = diag.quat_xyzw_to_R(np.array([first["qx"], first["qy"], first["qz"], first["qw"]], dtype=np.float64))
    v0_camera = np.array([first["vx"], first["vy"], first["vz"]], dtype=np.float64)
    omega0 = np.array([first.get("wx", 0.0), first.get("wy", 0.0), first.get("wz", 0.0)], dtype=np.float64)

    gravity = 9.8
    meta_path = scene_root / "metadata.json"
    if meta_path.exists():
        with meta_path.open("r", encoding="utf-8") as f:
            metadata = json.load(f)
        gravity = float(metadata.get("imu", {}).get("gravity_m_s2", gravity))
    camera_to_imu_body = load_camera_to_imu_translation_nwu(scene_root)
    p0, v0 = convert_camera_state_to_imu_initial(
        camera_position_w=p0_camera,
        camera_velocity_w=v0_camera,
        R_bw=R0,
        omega_body=omega0,
        camera_to_imu_body=camera_to_imu_body,
    )

    states = mechanize_imu_nwu(
        time_ns=time_ns,
        acc_b=acc,
        gyro_b=gyro,
        p0=p0,
        R0_bw=R0,
        v0_w=v0,
        gravity=gravity,
    )
    camera_times = ref["timestamp_ns"].to_numpy(np.int64)
    sampled_imu = sample_states_at_times(states, camera_times)
    sampled = convert_imu_states_to_camera_point(sampled_imu, camera_to_imu_body)

    pose_path = output_root / "trajectories" / f"{scene}_imu_only_poses.csv"
    write_poses_csv(pose_path, sampled)
    q_est = np.stack([matrix_to_quat_xyzw(R) for R in sampled.R_bw], axis=0)

    joined = pd.DataFrame(
        {
            "timestamp_ns": camera_times,
            "tx_est": sampled.position_w[:, 0],
            "ty_est": sampled.position_w[:, 1],
            "tz_est": sampled.position_w[:, 2],
            "qx_est": q_est[:, 0],
            "qy_est": q_est[:, 1],
            "qz_est": q_est[:, 2],
            "qw_est": q_est[:, 3],
            "tx_gt": ref["x"].to_numpy(float),
            "ty_gt": ref["y"].to_numpy(float),
            "tz_gt": ref["z"].to_numpy(float),
            "qx_gt": ref["qx"].to_numpy(float),
            "qy_gt": ref["qy"].to_numpy(float),
            "qz_gt": ref["qz"].to_numpy(float),
            "qw_gt": ref["qw"].to_numpy(float),
        }
    )
    joined["err_m"] = np.linalg.norm(
        joined[["tx_est", "ty_est", "tz_est"]].to_numpy(float)
        - joined[["tx_gt", "ty_gt", "tz_gt"]].to_numpy(float),
        axis=1,
    )
    label = f"{scene} / imu_only"
    summary = evaluate_joined(label, scene, "imu_only", joined, str(pose_path))
    return summary, joined


def load_existing_runs(result_roots: list[Path]) -> tuple[list[dict[str, object]], dict[str, pd.DataFrame]]:
    rows = pair_analysis.read_manifests(result_roots)
    summaries: list[dict[str, object]] = []
    trajectories: dict[str, pd.DataFrame] = {}
    for row in rows:
        summary, joined = pair_analysis.evaluate_run(row)
        label = str(summary["label"])
        method = str(summary["variant"])
        scene = str(summary["scene"])
        summaries.append(
            evaluate_joined(
                label=label,
                scene=scene,
                method=method,
                joined=joined,
                source=str(summary["poses_path"]),
            )
        )
        trajectories[label] = joined
    return summaries, trajectories


def color_for_label(label: str, index: int) -> str:
    return METHOD_COLORS.get(label, pair_analysis.TRACE_COLORS[index % len(pair_analysis.TRACE_COLORS)])


def plot_xy(trajs: dict[str, pd.DataFrame], outdir: Path, *, gt_region: bool = False) -> Path:
    fig, ax = plt.subplots(figsize=(8.0, 5.8), dpi=180)
    first = next(iter(trajs.values()))
    ax.plot(first["tx_gt"], first["ty_gt"], color="black", lw=2.8, label="GT")
    for idx, (label, df) in enumerate(trajs.items()):
        ax.plot(df["tx_est"], df["ty_est"], lw=1.6, color=color_for_label(label, idx), label=label)
    if gt_region:
        gt_x = first["tx_gt"].to_numpy(float)
        gt_y = first["ty_gt"].to_numpy(float)
        ax.set_xlim(float(gt_x.min()) - 1.5, float(gt_x.max()) + 1.5)
        ax.set_ylim(float(gt_y.min()) - 1.5, float(gt_y.max()) + 1.5)
    ax.set_aspect("equal", adjustable="box" if gt_region else "datalim")
    ax.set_xlabel("x / m (NWU)")
    ax.set_ylabel("y / m (NWU)")
    ax.grid(True, linewidth=0.35, alpha=0.5)
    ax.legend(fontsize=7)
    fig.tight_layout()
    name = "trajectory_xy_gt_region.png" if gt_region else "trajectory_xy_full_range.png"
    path = outdir / name
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_xz(trajs: dict[str, pd.DataFrame], outdir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8.0, 4.8), dpi=180)
    first = next(iter(trajs.values()))
    ax.plot(first["tx_gt"], first["tz_gt"], color="black", lw=2.8, label="GT")
    for idx, (label, df) in enumerate(trajs.items()):
        ax.plot(df["tx_est"], df["tz_est"], lw=1.6, color=color_for_label(label, idx), label=label)
    ax.set_xlabel("x / m (NWU)")
    ax.set_ylabel("z / m (NWU)")
    ax.grid(True, linewidth=0.35, alpha=0.5)
    ax.legend(fontsize=7)
    fig.tight_layout()
    path = outdir / "trajectory_xz_full_range.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_error(trajs: dict[str, pd.DataFrame], outdir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8.0, 4.8), dpi=180)
    for idx, (label, df) in enumerate(trajs.items()):
        t = (df["timestamp_ns"].to_numpy(float) - float(df["timestamp_ns"].iloc[0])) * 1e-9
        ax.plot(t, df["err_m"], lw=1.5, color=color_for_label(label, idx), label=label)
    ax.set_xlabel("time / s")
    ax.set_ylabel("position error / m")
    ax.grid(True, linewidth=0.35, alpha=0.5)
    ax.legend(fontsize=7)
    fig.tight_layout()
    path = outdir / "position_error_over_time.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_imu_only_focus(trajs: dict[str, pd.DataFrame], outdir: Path) -> Path:
    selected = {k: v for k, v in trajs.items() if k.endswith("/ imu_only")}
    first = next(iter(selected.values()))
    fig, ax = plt.subplots(figsize=(7.4, 5.4), dpi=180)
    ax.plot(first["tx_gt"], first["ty_gt"], color="black", lw=2.8, label="GT")
    for idx, (label, df) in enumerate(selected.items()):
        ax.plot(df["tx_est"], df["ty_est"], lw=1.8, color=color_for_label(label, idx), label=label)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("x / m (NWU)")
    ax.set_ylabel("y / m (NWU)")
    ax.grid(True, linewidth=0.35, alpha=0.5)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = outdir / "imu_only_xy_gt_vs_est.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def write_summary(path: Path, summaries: list[dict[str, object]]) -> None:
    fieldnames = [
        "scene",
        "method",
        "label",
        "matched_frames",
        "ate_rmse_m",
        "ate_median_m",
        "ate_final_m",
        "ate_max_m",
        "gt_path_length_m",
        "t_rel_m_per_frame",
        "t_rel_m_per_frame_median",
        "t_rel_m_per_frame_max",
        "r_rel_deg_per_frame",
        "r_rel_deg_per_frame_median",
        "r_rel_deg_per_frame_max",
        "t_vel_m_s",
        "t_vel_m_s_median",
        "t_vel_m_s_max",
        "r_vel_deg_s",
        "r_vel_deg_s_median",
        "r_vel_deg_s_max",
        "source",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted(summaries, key=lambda r: (str(r["scene"]), str(r["method"]))):
            writer.writerow(row)


def write_report(output_root: Path, summaries: list[dict[str, object]], figure_paths: list[Path]) -> None:
    lines = [
        "# Clear-circle IMU-only mechanization",
        "",
        "Purpose: propagate pose using only imu_data.csv after initializing from the first GT frame.",
        "No MACVO visual output, optimizer, factor graph, or GT reset is used after initialization.",
        "",
        "| scene | method | RMSE m | median m | final m | max m | t_vel m/s | r_vel deg/s |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(summaries, key=lambda r: (str(r["scene"]), str(r["method"]))):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["scene"]),
                    str(row["method"]),
                    f"{float(row['ate_rmse_m']):.6f}",
                    f"{float(row['ate_median_m']):.6f}",
                    f"{float(row['ate_final_m']):.6f}",
                    f"{float(row['ate_max_m']):.6f}",
                    f"{float(row['t_vel_m_s']):.6f}",
                    f"{float(row['r_vel_deg_s']):.6f}",
                ]
            )
            + " |"
        )

    zero = [r for r in summaries if r["scene"] == "clear_circle_zero_noise" and r["method"] == "imu_only"]
    if zero:
        z = zero[0]
        lines.extend(
            [
                "",
                "Automatic interpretation:",
                f"- zero-noise IMU-only final error is {float(z['ate_final_m']):.6f} m "
                f"and RMSE is {float(z['ate_rmse_m']):.6f} m.",
                "- This checks long-horizon inertial propagation, not local per-frame IMU/GT consistency.",
            ]
        )

    lines.extend(["", "Figures:"])
    for path in figure_paths:
        lines.append(f"- `{path.relative_to(output_root)}`")
    lines.extend(["", "Trajectory CSV files:", "- `trajectories/*_imu_only_poses.csv`"])
    (output_root / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--existing-result-root",
        type=Path,
        action="append",
        default=None,
        help="Existing MACVO/VIO result root with run_manifest.csv. Can be repeated.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_root: Path = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    existing_roots = args.existing_result_root or DEFAULT_EXISTING_RESULT_ROOTS
    summaries, trajectories = load_existing_runs(existing_roots)

    for scene, root in DEFAULT_SCENE_ROOTS.items():
        summary, joined = run_imu_only_for_scene(scene, root, output_root)
        summaries.append(summary)
        trajectories[str(summary["label"])] = joined

    write_summary(output_root / "trajectory_summary.csv", summaries)
    figure_paths = [
        plot_xy(trajectories, output_root, gt_region=False),
        plot_xy(trajectories, output_root, gt_region=True),
        plot_xz(trajectories, output_root),
        plot_error(trajectories, output_root),
        plot_imu_only_focus(trajectories, output_root),
    ]
    pair_analysis.write_interactive_html(trajectories, output_root)
    figure_paths.append(output_root / "interactive_trajectory_gt_vs_est.html")
    write_report(output_root, summaries, figure_paths)

    print(f"Wrote {output_root}")
    print(f"Summary rows: {len(summaries)}")
    print(f"Figures: {len([p for p in figure_paths if p.suffix.lower() == '.png'])}")
    print(f"Interactive: {output_root / 'interactive_trajectory_gt_vs_est.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
