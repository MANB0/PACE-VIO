#!/usr/bin/env python3
"""
Diagnose whether HoloOcean IMU measurements are consistent with GT motion.

This script is intentionally independent from the MACVO runtime.  It reads
imu_data.csv and ref_pose.csv directly and evaluates two IMU residual protocols:

1. standard_vio:
   Forster/GTSAM-style preintegration of specific force without adding gravity
   inside the integral.  GT residuals use the usual gravity terms:
       R_i^T (v_j - v_i - g dt) - Delta_v
       R_i^T (p_j - p_i - v_i dt - 0.5 g dt^2) - Delta_p

2. macvo_kinematic:
   MACVO's current style where gravity is added during integration, producing
   kinematic Delta_v/Delta_p in the body-i frame.  GT residuals omit the gravity
   terms:
       R_i^T (v_j - v_i) - Delta_v
       R_i^T (p_j - p_i - v_i dt) - Delta_p

Coordinate hypotheses:
  - raw_nwu: use metadata coordinates directly (x forward, y left, z up).
  - rx180_to_ned: consistently transform IMU, GT body, and GT world axes by
    diag(1,-1,-1).  This should be numerically equivalent to raw_nwu.
  - imu_rx180_only_gt_nwu: transform IMU only while leaving GT in NWU.  This is
    a deliberate mismatch detector for accidental mixed-frame pipelines.

Outputs are written to analysis_imu_vio_residual_diagnostics/.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


WORKDIR = Path("/home/admin1/macvo-dev")
OUTPUT_ROOT = WORKDIR / "analysis_imu_vio_residual_diagnostics"
FIG_ROOT = OUTPUT_ROOT / "figures"

SCENE_ROOTS = {
    "turbid_harbor": Path("/mnt/e/文档/holoocean/code/recordings/batch_20260515_155653/turbid_harbor"),
    "clear_shallow": Path("/mnt/e/文档/holoocean/code/recordings/batch_20260515_155653/clear_shallow"),
    "deep_dark": Path("/mnt/e/文档/holoocean/code/recordings/batch_20260515_155653/deep_dark"),
    "caustic_shallow": Path("/mnt/e/文档/holoocean/code/recordings/batch_20260515_155653/caustic_shallow"),
    "dam_inspection": Path("/mnt/e/文档/holoocean/code/recordings/batch_20260515_155653/dam_inspection"),
    "murky_coast": Path("/mnt/e/文档/holoocean/code/recordings/batch_20260515_155653/murky_coast"),
    "open_water": Path("/mnt/e/文档/holoocean/code/recordings/batch_20260515_155653/open_water"),
    "moderate_turbidity": Path("/mnt/e/文档/holoocean/code/recordings/batch_20260528_203401/moderate_turbidity"),
    "open_water_overcast": Path("/mnt/e/文档/holoocean/code/recordings/batch_20260528_203401/open_water_overcast"),
    "twilight_coast": Path("/mnt/e/文档/holoocean/code/recordings/batch_20260528_203401/twilight_coast"),
    "validation_moderate_harbor": Path("/mnt/e/文档/holoocean/code/recordings/batch_20260601_162707/validation_moderate_harbor"),
    "validation_transient_dropout": Path("/mnt/e/文档/holoocean/code/recordings/batch_20260601_162707/validation_transient_dropout"),
    "validation_twilight_structure": Path("/mnt/e/文档/holoocean/code/recordings/batch_20260601_162707/validation_twilight_structure"),
    "locked_murky_entry_help": Path("/mnt/e/文档/holoocean/code/recordings/batch_20260618_110853/locked_murky_entry_help"),
    "locked_clear_imu_harm": Path("/mnt/e/文档/holoocean/code/recordings/batch_20260618_110853/locked_clear_imu_harm"),
    "locked_quality_degrade_no_dropout": Path("/mnt/e/文档/holoocean/code/recordings/batch_20260618_110853/locked_quality_degrade_no_dropout"),
}

MAIN12 = {
    "turbid_harbor",
    "clear_shallow",
    "deep_dark",
    "caustic_shallow",
    "dam_inspection",
    "murky_coast",
    "open_water_overcast",
    "validation_transient_dropout",
    "validation_twilight_structure",
    "locked_murky_entry_help",
    "locked_clear_imu_harm",
    "locked_quality_degrade_no_dropout",
}

RX180 = np.diag([1.0, -1.0, -1.0])


@dataclass
class Trajectory:
    ts: np.ndarray
    p_w: np.ndarray
    q_xyzw: np.ndarray
    R_bw: np.ndarray
    v_csv: np.ndarray
    v_b: np.ndarray
    v_w: np.ndarray
    w_b: np.ndarray
    velocity_frame_used: str


@dataclass
class ImuSeries:
    ts: np.ndarray
    acc_b: np.ndarray
    gyro_b: np.ndarray


@dataclass
class PairResidual:
    scene: str
    variant: str
    protocol: str
    pair_idx: int
    t_i_ns: int
    t_j_ns: int
    dt: float
    n_imu: int
    rot_err_deg: float
    vel_err_norm: float
    pos_err_norm: float
    gt_delta_p_norm: float
    imu_delta_p_norm: float
    gt_delta_v_norm: float
    imu_delta_v_norm: float
    gyro_delta_angle_deg: float
    gt_delta_angle_deg: float


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def quat_xyzw_to_R(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    q = q / max(np.linalg.norm(q), 1e-15)
    x, y, z, w = q
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array(
        [
            [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
            [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
            [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)],
        ],
        dtype=np.float64,
    )


def skew(v: np.ndarray) -> np.ndarray:
    x, y, z = v
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64)


def so3_exp(phi: np.ndarray) -> np.ndarray:
    phi = np.asarray(phi, dtype=np.float64).reshape(3)
    theta = float(np.linalg.norm(phi))
    K = skew(phi)
    if theta < 1e-12:
        return np.eye(3) + K + 0.5 * K @ K
    return np.eye(3) + (math.sin(theta) / theta) * K + ((1.0 - math.cos(theta)) / (theta * theta)) * (K @ K)


def so3_log(R: np.ndarray) -> np.ndarray:
    cos_theta = max(-1.0, min(1.0, (float(np.trace(R)) - 1.0) * 0.5))
    theta = math.acos(cos_theta)
    if theta < 1e-12:
        return np.array(
            [
                0.5 * (R[2, 1] - R[1, 2]),
                0.5 * (R[0, 2] - R[2, 0]),
                0.5 * (R[1, 0] - R[0, 1]),
            ],
            dtype=np.float64,
        )
    return (theta / (2.0 * math.sin(theta))) * np.array(
        [R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]],
        dtype=np.float64,
    )


def detect_velocity_frame(ts: np.ndarray, p_w: np.ndarray, R_bw: np.ndarray, v_csv: np.ndarray) -> tuple[str, float, float]:
    if len(ts) < 2:
        return "world", float("nan"), float("nan")
    dt = np.maximum((ts[1:] - ts[:-1]).astype(np.float64) * 1e-9, 1e-12)
    v_fd_w = (p_w[1:] - p_w[:-1]) / dt[:, None]
    v_world_err = np.linalg.norm(v_fd_w - v_csv[:-1], axis=1)
    v_body_as_world = np.einsum("nij,nj->ni", R_bw[:-1], v_csv[:-1])
    v_body_err = np.linalg.norm(v_fd_w - v_body_as_world, axis=1)
    world_med = float(np.median(v_world_err))
    body_med = float(np.median(v_body_err))
    return ("world" if world_med <= body_med else "body", world_med, body_med)


def metadata_velocity_frame(scene_root: Path) -> str | None:
    meta_path = scene_root / "metadata.json"
    if not meta_path.exists():
        return None
    with meta_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)
    candidates = [
        meta.get("ground_truth", {}).get("velocity_frame"),
        meta.get("coordinate_convention", {}).get("ref_pose_velocity_frame"),
    ]
    for value in candidates:
        text = str(value or "").upper()
        if "WORLD" in text or "GLOBAL" in text:
            return "world"
        if "BODY" in text:
            return "body"
    return None


def load_camera_to_imu_translation_nwu(scene_root: Path) -> np.ndarray:
    meta_path = scene_root / "metadata.json"
    if not meta_path.exists():
        return np.zeros(3, dtype=np.float64)
    with meta_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)
    extrinsics = meta.get("extrinsics", {})
    body_imu = extrinsics.get("T_body_imu", {}).get("translation_body_nwu_m")
    body_camera = extrinsics.get("T_body_camera", {}).get("translation_body_nwu_m")
    if body_imu and body_camera and len(body_imu) == 3 and len(body_camera) == 3:
        return np.asarray(body_imu, dtype=np.float64) - np.asarray(body_camera, dtype=np.float64)
    imu_camera = extrinsics.get("T_imu_camera", {}).get("translation_body_nwu_m")
    if imu_camera and len(imu_camera) == 3:
        return -np.asarray(imu_camera, dtype=np.float64)
    return np.zeros(3, dtype=np.float64)


def load_gt(scene_root: Path) -> Trajectory:
    rows = read_csv_rows(scene_root / "ref_pose.csv")
    ts = np.asarray([int(float(r["timestamp"])) for r in rows], dtype=np.int64)
    p_w = np.asarray([[float(r["x"]), float(r["y"]), float(r["z"])] for r in rows], dtype=np.float64)
    q = np.asarray([[float(r["qx"]), float(r["qy"]), float(r["qz"]), float(r["qw"])] for r in rows], dtype=np.float64)
    R_bw = np.stack([quat_xyzw_to_R(qq) for qq in q], axis=0)
    v_csv = np.asarray([[float(r["vx"]), float(r["vy"]), float(r["vz"])] for r in rows], dtype=np.float64)
    w_b = np.asarray([[float(r["wx"]), float(r["wy"]), float(r["wz"])] for r in rows], dtype=np.float64)
    detected_velocity_frame, _, _ = detect_velocity_frame(ts, p_w, R_bw, v_csv)
    velocity_frame = metadata_velocity_frame(scene_root) or detected_velocity_frame
    if velocity_frame == "world":
        v_w = v_csv.copy()
        v_b = np.einsum("nji,nj->ni", R_bw, v_w)
    else:
        v_b = v_csv.copy()
        v_w = np.einsum("nij,nj->ni", R_bw, v_b)
    return Trajectory(
        ts=ts,
        p_w=p_w,
        q_xyzw=q,
        R_bw=R_bw,
        v_csv=v_csv,
        v_b=v_b,
        v_w=v_w,
        w_b=w_b,
        velocity_frame_used=velocity_frame,
    )


def load_imu(scene_root: Path) -> ImuSeries:
    rows = read_csv_rows(scene_root / "imu_data.csv")
    ts = np.asarray([int(float(r["timestamp"])) for r in rows], dtype=np.int64)
    gyro = np.asarray([[float(r["ang_vel_x"]), float(r["ang_vel_y"]), float(r["ang_vel_z"])] for r in rows], dtype=np.float64)
    acc = np.asarray([[float(r["lin_acc_x"]), float(r["lin_acc_y"]), float(r["lin_acc_z"])] for r in rows], dtype=np.float64)
    return ImuSeries(ts=ts, acc_b=acc, gyro_b=gyro)


def transform_traj(traj: Trajectory, S: np.ndarray) -> Trajectory:
    p_w = (S @ traj.p_w.T).T
    w_b = (S @ traj.w_b.T).T
    R_bw = np.einsum("ab,nbc,dc->nad", S, traj.R_bw, S)
    v_w = (S @ traj.v_w.T).T
    v_b = np.einsum("nji,nj->ni", R_bw, v_w)
    v_csv = (S @ traj.v_csv.T).T if traj.velocity_frame_used == "world" else (S @ traj.v_b.T).T
    return Trajectory(
        ts=traj.ts.copy(),
        p_w=p_w,
        q_xyzw=traj.q_xyzw.copy(),
        v_csv=v_csv,
        R_bw=R_bw,
        v_b=v_b,
        v_w=v_w,
        w_b=w_b,
        velocity_frame_used=traj.velocity_frame_used,
    )


def transform_imu(imu: ImuSeries, S: np.ndarray) -> ImuSeries:
    return ImuSeries(
        ts=imu.ts.copy(),
        acc_b=(S @ imu.acc_b.T).T,
        gyro_b=(S @ imu.gyro_b.T).T,
    )


def camera_to_imu_translation_for_variant(scene_root: Path, variant: str) -> np.ndarray:
    r_nwu = load_camera_to_imu_translation_nwu(scene_root)
    if variant == "raw_nwu" or variant == "imu_rx180_only_gt_nwu":
        return r_nwu
    if variant == "rx180_to_ned":
        return RX180 @ r_nwu
    raise ValueError(variant)


def load_gravity(scene_root: Path, variant: str) -> np.ndarray:
    meta_path = scene_root / "metadata.json"
    gravity = 9.8
    if meta_path.exists():
        with meta_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)
        gravity = float(meta.get("imu", {}).get("gravity_m_s2", gravity))
    if variant == "raw_nwu" or variant == "imu_rx180_only_gt_nwu":
        return np.array([0.0, 0.0, -gravity], dtype=np.float64)
    if variant == "rx180_to_ned":
        return np.array([0.0, 0.0, gravity], dtype=np.float64)
    raise ValueError(variant)


def query_imu_interval(imu: ImuSeries, start_ns: int, end_ns: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    i0 = int(np.searchsorted(imu.ts, start_ns, side="left"))
    i1 = int(np.searchsorted(imu.ts, end_ns, side="right"))
    i0 = max(0, min(i0, len(imu.ts)))
    i1 = max(0, min(i1, len(imu.ts)))
    return imu.ts[i0:i1], imu.acc_b[i0:i1], imu.gyro_b[i0:i1]


def preintegrate_standard(ts: np.ndarray, acc: np.ndarray, gyro: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    if len(ts) < 2:
        return np.eye(3), np.zeros(3), np.zeros(3), 0.0
    dR = np.eye(3)
    dv = np.zeros(3)
    dp = np.zeros(3)
    dt_total = 0.0
    for k in range(len(ts) - 1):
        dt = max(float(ts[k + 1] - ts[k]) * 1e-9, 1e-12)
        acc_mid = 0.5 * (acc[k] + acc[k + 1])
        gyro_mid = 0.5 * (gyro[k] + gyro[k + 1])
        a_i = dR @ acc_mid
        dp = dp + dv * dt + 0.5 * a_i * dt * dt
        dv = dv + a_i * dt
        dR = dR @ so3_exp(gyro_mid * dt)
        dt_total += dt
    return dR, dv, dp, dt_total


def preintegrate_macvo_kinematic(
    ts: np.ndarray, acc: np.ndarray, gyro: np.ndarray, R_i_bw: np.ndarray, g_w: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    if len(ts) < 2:
        return np.eye(3), np.zeros(3), np.zeros(3), 0.0
    dR = np.eye(3)
    dv = np.zeros(3)
    dp = np.zeros(3)
    dt_total = 0.0
    g_body_i = R_i_bw.T @ g_w
    for k in range(len(ts) - 1):
        dt = max(float(ts[k + 1] - ts[k]) * 1e-9, 1e-12)
        acc_mid = 0.5 * (acc[k] + acc[k + 1])
        gyro_mid = 0.5 * (gyro[k] + gyro[k + 1])
        a_kin_i = dR @ acc_mid + g_body_i
        dp = dp + dv * dt + 0.5 * a_kin_i * dt * dt
        dv = dv + a_kin_i * dt
        dR = dR @ so3_exp(gyro_mid * dt)
        dt_total += dt
    return dR, dv, dp, dt_total


def evaluate_scene_variant(
    scene: str,
    scene_root: Path,
    variant: str,
    max_pairs: int | None = None,
) -> list[PairResidual]:
    gt_raw = load_gt(scene_root)
    imu_raw = load_imu(scene_root)
    if variant == "raw_nwu":
        gt = gt_raw
        imu = imu_raw
    elif variant == "rx180_to_ned":
        gt = transform_traj(gt_raw, RX180)
        imu = transform_imu(imu_raw, RX180)
    elif variant == "imu_rx180_only_gt_nwu":
        gt = gt_raw
        imu = transform_imu(imu_raw, RX180)
    else:
        raise ValueError(variant)

    g_w = load_gravity(scene_root, variant)
    camera_to_imu_body = camera_to_imu_translation_for_variant(scene_root, variant)
    n_pairs = len(gt.ts) - 1 if max_pairs is None else min(len(gt.ts) - 1, max_pairs)
    records: list[PairResidual] = []

    for i in range(n_pairs):
        t_i = int(gt.ts[i])
        t_j = int(gt.ts[i + 1])
        ts, acc, gyro = query_imu_interval(imu, t_i, t_j)
        if len(ts) < 2:
            continue

        R_i = gt.R_bw[i]
        R_j = gt.R_bw[i + 1]
        dR_gt = R_i.T @ R_j
        dt = float(t_j - t_i) * 1e-9
        r_b = camera_to_imu_body
        p_i = gt.p_w[i] + R_i @ r_b
        p_j = gt.p_w[i + 1] + R_j @ r_b
        v_i = gt.v_w[i] + R_i @ np.cross(gt.w_b[i], r_b)
        v_j = gt.v_w[i + 1] + R_j @ np.cross(gt.w_b[i + 1], r_b)

        dR_s, dv_s, dp_s, imu_dt = preintegrate_standard(ts, acc, gyro)
        gt_dv_s = R_i.T @ (v_j - v_i - g_w * dt)
        gt_dp_s = R_i.T @ (p_j - p_i - v_i * dt - 0.5 * g_w * dt * dt)
        records.append(
            make_record(
                scene, variant, "standard_vio", i, t_i, t_j, dt, len(ts),
                dR_s, dv_s, dp_s, dR_gt, gt_dv_s, gt_dp_s,
            )
        )

        dR_m, dv_m, dp_m, _ = preintegrate_macvo_kinematic(ts, acc, gyro, R_i, g_w)
        gt_dv_m = R_i.T @ (v_j - v_i)
        gt_dp_m = R_i.T @ (p_j - p_i - v_i * dt)
        records.append(
            make_record(
                scene, variant, "macvo_kinematic", i, t_i, t_j, dt, len(ts),
                dR_m, dv_m, dp_m, dR_gt, gt_dv_m, gt_dp_m,
            )
        )

    return records


def make_record(
    scene: str,
    variant: str,
    protocol: str,
    pair_idx: int,
    t_i: int,
    t_j: int,
    dt: float,
    n_imu: int,
    dR_imu: np.ndarray,
    dv_imu: np.ndarray,
    dp_imu: np.ndarray,
    dR_gt: np.ndarray,
    dv_gt: np.ndarray,
    dp_gt: np.ndarray,
) -> PairResidual:
    rot_err = so3_log(dR_imu.T @ dR_gt)
    return PairResidual(
        scene=scene,
        variant=variant,
        protocol=protocol,
        pair_idx=pair_idx,
        t_i_ns=t_i,
        t_j_ns=t_j,
        dt=dt,
        n_imu=n_imu,
        rot_err_deg=math.degrees(float(np.linalg.norm(rot_err))),
        vel_err_norm=float(np.linalg.norm(dv_gt - dv_imu)),
        pos_err_norm=float(np.linalg.norm(dp_gt - dp_imu)),
        gt_delta_p_norm=float(np.linalg.norm(dp_gt)),
        imu_delta_p_norm=float(np.linalg.norm(dp_imu)),
        gt_delta_v_norm=float(np.linalg.norm(dv_gt)),
        imu_delta_v_norm=float(np.linalg.norm(dv_imu)),
        gyro_delta_angle_deg=math.degrees(float(np.linalg.norm(so3_log(dR_imu)))),
        gt_delta_angle_deg=math.degrees(float(np.linalg.norm(so3_log(dR_gt)))),
    )


def percentile(values: Iterable[float], q: float) -> float:
    arr = np.asarray(list(values), dtype=np.float64)
    if len(arr) == 0:
        return float("nan")
    return float(np.percentile(arr, q))


def median(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=np.float64)
    if len(arr) == 0:
        return float("nan")
    return float(np.median(arr))


def mean(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=np.float64)
    if len(arr) == 0:
        return float("nan")
    return float(np.mean(arr))


def summarize(records: list[PairResidual]) -> list[dict[str, object]]:
    by_key: dict[tuple[str, str, str], list[PairResidual]] = {}
    for r in records:
        by_key.setdefault((r.scene, r.variant, r.protocol), []).append(r)
    rows: list[dict[str, object]] = []
    for key in sorted(by_key):
        scene, variant, protocol = key
        rs = by_key[key]
        gt_p = [r.gt_delta_p_norm for r in rs]
        imu_p = [r.imu_delta_p_norm for r in rs]
        gt_v = [r.gt_delta_v_norm for r in rs]
        imu_v = [r.imu_delta_v_norm for r in rs]
        rows.append(
            {
                "scene": scene,
                "paper_role": "main" if scene in MAIN12 else "excluded_from_main",
                "variant": variant,
                "protocol": protocol,
                "n_pairs": len(rs),
                "rot_err_deg_median": median(r.rot_err_deg for r in rs),
                "rot_err_deg_p95": percentile((r.rot_err_deg for r in rs), 95),
                "rot_err_deg_mean": mean(r.rot_err_deg for r in rs),
                "vel_err_median": median(r.vel_err_norm for r in rs),
                "vel_err_p95": percentile((r.vel_err_norm for r in rs), 95),
                "vel_err_mean": mean(r.vel_err_norm for r in rs),
                "pos_err_median": median(r.pos_err_norm for r in rs),
                "pos_err_p95": percentile((r.pos_err_norm for r in rs), 95),
                "pos_err_mean": mean(r.pos_err_norm for r in rs),
                "gt_delta_p_median": median(gt_p),
                "imu_delta_p_median": median(imu_p),
                "imu_over_gt_delta_p_median": median(
                    (i / g for i, g in zip(imu_p, gt_p) if g > 1e-12)
                ),
                "gt_delta_v_median": median(gt_v),
                "imu_delta_v_median": median(imu_v),
                "imu_over_gt_delta_v_median": median(
                    (i / g for i, g in zip(imu_v, gt_v) if g > 1e-12)
                ),
                "gt_delta_angle_deg_median": median(r.gt_delta_angle_deg for r in rs),
                "imu_delta_angle_deg_median": median(r.gyro_delta_angle_deg for r in rs),
            }
        )
    return rows


def build_velocity_frame_checks() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for scene, root in SCENE_ROOTS.items():
        rows_raw = read_csv_rows(root / "ref_pose.csv")
        ts = np.asarray([int(float(r["timestamp"])) for r in rows_raw], dtype=np.int64)
        p_w = np.asarray([[float(r["x"]), float(r["y"]), float(r["z"])] for r in rows_raw], dtype=np.float64)
        q = np.asarray([[float(r["qx"]), float(r["qy"]), float(r["qz"]), float(r["qw"])] for r in rows_raw], dtype=np.float64)
        R_bw = np.stack([quat_xyzw_to_R(qq) for qq in q], axis=0)
        v_csv = np.asarray([[float(r["vx"]), float(r["vy"]), float(r["vz"])] for r in rows_raw], dtype=np.float64)
        detected, world_med, body_med = detect_velocity_frame(ts, p_w, R_bw, v_csv)
        declared = metadata_velocity_frame(root)
        used = declared or detected
        rows.append(
            {
                "scene": scene,
                "paper_role": "main" if scene in MAIN12 else "excluded_from_main",
                "metadata_velocity_frame": declared or "",
                "finite_difference_best_frame": detected,
                "velocity_frame_used": used,
                "metadata_matches_finite_difference": "" if declared is None else declared == detected,
                "finite_difference_error_if_world_mps_median": world_med,
                "finite_difference_error_if_body_mps_median": body_med,
                "body_over_world_error_ratio": body_med / max(world_med, 1e-12),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def fmt(x: object, nd: int = 4) -> str:
    try:
        val = float(x)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return ""
    if not np.isfinite(val):
        return ""
    return f"{val:.{nd}f}"


def write_markdown(summary_rows: list[dict[str, object]]) -> None:
    selected = [
        r
        for r in summary_rows
        if r["variant"] in {"raw_nwu", "imu_rx180_only_gt_nwu"} and r["protocol"] == "standard_vio"
    ]
    header = [
        "Scene",
        "Role",
        "Variant",
        "Rot med deg",
        "Vel med m/s",
        "Pos med m",
        "IMU/GT Δp",
        "IMU/GT Δv",
    ]
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    for r in selected:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(r["scene"]),
                    str(r["paper_role"]),
                    str(r["variant"]),
                    fmt(r["rot_err_deg_median"], 5),
                    fmt(r["vel_err_median"], 5),
                    fmt(r["pos_err_median"], 5),
                    fmt(r["imu_over_gt_delta_p_median"], 3),
                    fmt(r["imu_over_gt_delta_v_median"], 3),
                ]
            )
            + " |"
        )
    (OUTPUT_ROOT / "standard_vio_key_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_summary(summary_rows: list[dict[str, object]]) -> None:
    rows = [
        r for r in summary_rows
        if r["variant"] == "raw_nwu" and r["protocol"] == "standard_vio"
    ]
    rows = sorted(rows, key=lambda r: str(r["scene"]))
    labels = [str(r["scene"]) for r in rows]
    x = np.arange(len(rows))
    fig, axes = plt.subplots(3, 1, figsize=(13, 9), constrained_layout=True)
    metrics = [
        ("rot_err_deg_median", "Rotation residual median [deg]"),
        ("vel_err_median", "Velocity residual median [m/s]"),
        ("pos_err_median", "Position residual median [m]"),
    ]
    for ax, (key, title) in zip(axes, metrics):
        vals = [float(r[key]) for r in rows]
        colors = ["#2C6DB2" if r["paper_role"] == "main" else "#999999" for r in rows]
        ax.bar(x, vals, color=colors)
        ax.set_ylabel(title)
        ax.grid(axis="y", alpha=0.25)
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    fig.suptitle("Raw NWU standard-VIO IMU-vs-GT residuals")
    fig.savefig(FIG_ROOT / "raw_nwu_standard_vio_residual_summary.png", dpi=180)
    plt.close(fig)


def plot_scene_timeseries(records: list[PairResidual], scene: str, variant: str = "raw_nwu", protocol: str = "standard_vio") -> None:
    rs = [r for r in records if r.scene == scene and r.variant == variant and r.protocol == protocol]
    if not rs:
        return
    t = np.asarray([r.pair_idx for r in rs])
    fig, axes = plt.subplots(3, 1, figsize=(12, 7), constrained_layout=True, sharex=True)
    axes[0].plot(t, [r.rot_err_deg for r in rs], color="#2C6DB2")
    axes[0].set_ylabel("rot err [deg]")
    axes[1].plot(t, [r.vel_err_norm for r in rs], color="#D55E00")
    axes[1].set_ylabel("vel err [m/s]")
    axes[2].plot(t, [r.pos_err_norm for r in rs], color="#009E73")
    axes[2].set_ylabel("pos err [m]")
    axes[2].set_xlabel("camera pair index")
    for ax in axes:
        ax.grid(True, alpha=0.25)
    fig.suptitle(f"{scene}: {variant}, {protocol}")
    fig.savefig(FIG_ROOT / f"{scene}_{variant}_{protocol}_timeseries.png", dpi=180)
    plt.close(fig)


def write_report(
    summary_rows: list[dict[str, object]],
    records: list[PairResidual],
    velocity_checks: list[dict[str, object]],
) -> None:
    raw_std = [r for r in summary_rows if r["variant"] == "raw_nwu" and r["protocol"] == "standard_vio"]
    mixed_std = [
        r for r in summary_rows if r["variant"] == "imu_rx180_only_gt_nwu" and r["protocol"] == "standard_vio"
    ]
    raw_main = [r for r in raw_std if r["paper_role"] == "main"]
    metadata_world_count = sum(1 for r in velocity_checks if r.get("metadata_velocity_frame") == "world")
    fd_world_count = sum(1 for r in velocity_checks if r.get("finite_difference_best_frame") == "world")
    metadata_fd_mismatch_count = sum(
        1 for r in velocity_checks if r.get("metadata_matches_finite_difference") is False
    )

    def agg(rows: list[dict[str, object]], key: str) -> float:
        return median(float(r[key]) for r in rows)

    lines = [
        "# IMU VIO residual diagnostics",
        "",
        "Purpose: verify whether raw IMU measurements are consistent with GT motion before changing fusion logic.",
        "",
        "Protocols:",
        "- `standard_vio`: known VIO/Forster-style residual with gravity outside the IMU integral.",
        "- `macvo_kinematic`: MACVO-style residual with gravity compensation inside the integral.",
        "",
        "Coordinate variants:",
        "- `raw_nwu`: metadata-native NWU/FLU convention.",
        "- `rx180_to_ned`: consistently transformed by Rx(180); should match raw_nwu if conversion is coherent.",
        "- `imu_rx180_only_gt_nwu`: deliberate mixed-frame diagnostic.",
        "",
        "Aggregate medians for main scenes under `raw_nwu + standard_vio`:",
        f"- rotation residual: {agg(raw_main, 'rot_err_deg_median'):.6f} deg",
        f"- velocity residual: {agg(raw_main, 'vel_err_median'):.6f} m/s",
        f"- position residual: {agg(raw_main, 'pos_err_median'):.6f} m",
        f"- IMU/GT delta-p norm ratio: {agg(raw_main, 'imu_over_gt_delta_p_median'):.3f}",
        "",
        "GT velocity-frame check:",
        f"- metadata declares world-frame ref_pose velocity for {metadata_world_count}/{len(velocity_checks)} scenes.",
        f"- finite-difference check independently prefers world-frame velocity for {fd_world_count}/{len(velocity_checks)} scenes.",
        f"- metadata/finite-difference mismatches: {metadata_fd_mismatch_count}.",
        "- The finite-difference check compares position-derived world velocity against the CSV velocity directly and after body-to-world rotation.",
        "",
        "Mixed-frame check under `imu_rx180_only_gt_nwu + standard_vio`:",
        f"- rotation residual median across all scenes: {agg(mixed_std, 'rot_err_deg_median'):.6f} deg",
        f"- velocity residual median across all scenes: {agg(mixed_std, 'vel_err_median'):.6f} m/s",
        f"- position residual median across all scenes: {agg(mixed_std, 'pos_err_median'):.6f} m",
        "",
        "Files:",
        "- `imu_vio_pair_residuals.csv`: per camera-pair residuals.",
        "- `imu_vio_scene_summary.csv`: scene-level residual summaries.",
        "- `velocity_frame_check.csv`: finite-difference check for the GT velocity columns.",
        "- `standard_vio_key_table.md`: compact table for raw and mixed coordinate variants.",
        "- `figures/raw_nwu_standard_vio_residual_summary.png`: scene summary plot.",
        "",
        "Interpretation notes:",
        "- Small raw_nwu residuals mean the IMU measurements are internally consistent with GT under a known VIO equation.",
        "- Large mixed-frame residuals are expected and indicate the diagnostic can detect coordinate mistakes.",
        "- If raw_nwu is good but MACVO fusion performs poorly, the likely issue is the fusion/state model rather than the raw IMU calculation.",
        "- Downstream diagnostics use metadata velocity-frame declarations when present and use finite-difference checks as an independent consistency audit.",
    ]
    (OUTPUT_ROOT / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    FIG_ROOT.mkdir(parents=True, exist_ok=True)

    all_records: list[PairResidual] = []
    variants = ["raw_nwu", "rx180_to_ned", "imu_rx180_only_gt_nwu"]
    for scene, root in SCENE_ROOTS.items():
        for variant in variants:
            all_records.extend(evaluate_scene_variant(scene, root, variant))

    pair_rows = [r.__dict__ for r in all_records]
    write_csv(OUTPUT_ROOT / "imu_vio_pair_residuals.csv", pair_rows, list(pair_rows[0].keys()))
    summary_rows = summarize(all_records)
    write_csv(OUTPUT_ROOT / "imu_vio_scene_summary.csv", summary_rows, list(summary_rows[0].keys()))
    velocity_checks = build_velocity_frame_checks()
    write_csv(OUTPUT_ROOT / "velocity_frame_check.csv", velocity_checks, list(velocity_checks[0].keys()))
    write_markdown(summary_rows)
    plot_summary(summary_rows)
    for scene in ["turbid_harbor", "murky_coast", "locked_murky_entry_help", "locked_clear_imu_harm"]:
        plot_scene_timeseries(all_records, scene)
    write_report(summary_rows, all_records, velocity_checks)

    print(f"Wrote {OUTPUT_ROOT}")
    print(f"Pair residual rows: {len(pair_rows)}")
    print(f"Scene summary rows: {len(summary_rows)}")
    print(f"Figures: {len(list(FIG_ROOT.glob('*.png')))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
