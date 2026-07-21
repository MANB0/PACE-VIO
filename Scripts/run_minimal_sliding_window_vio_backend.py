#!/usr/bin/env python3
"""Run an offline minimal sliding-window VIO backend on closed-path scenes.

This is a diagnostic prototype, not a full ORB-SLAM3 backend.  It uses MACVO
poses as visual relative-pose constraints and raw IMU preintegration as
adjacent inertial constraints in a local fixed-lag optimizer.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pypose as pp
import torch
from scipy.spatial.transform import Rotation

WORKDIR = Path("/home/admin1/macvo-dev")
if str(WORKDIR) not in sys.path:
    sys.path.insert(0, str(WORKDIR))

from Module.IMUPreintegration import preintegrate_imu
from Scripts import analyse_clear_circle_pair_vio as pair_analysis
from Scripts import run_clear_circle_imu_only_mechanization as imu_mech
from Scripts import run_latest_closed_paths_methods as closed_paths
from Utility.RunOutputBundle import find_output_bundle
from Utility.SlidingWindowVIO import (
    SlidingWindowConfig,
    SlidingWindowEdge,
    SlidingWindowSequence,
    optimize_sliding_window_sequence,
)


BATCH_ROOT = Path("/mnt/e/文档/holoocean/code/recordings/batch_zed100_closed_paths_smooth_20260705")
SCENE_ROOTS = {
    "clear_circle_normal_noise": BATCH_ROOT / "normal_noise" / "clear_circle_path",
    "clear_rectangle_normal_noise": BATCH_ROOT / "normal_noise" / "clear_rectangle_path",
    "clear_circle_zero_noise": BATCH_ROOT / "zero_noise" / "clear_circle_path",
    "clear_rectangle_zero_noise": BATCH_ROOT / "zero_noise" / "clear_rectangle_path",
}
DEFAULT_RESULT_ROOT = WORKDIR / "Results/minimal_sliding_window_vio_20260708"
DEFAULT_COMPARE_ROOT = WORKDIR / "Results/minimal_sliding_window_vio_comparison_20260708"
DEFAULT_OUTPUT_ROOT = WORKDIR / "analysis_minimal_sliding_window_vio_20260708"
DEFAULT_VISUAL_MANIFEST = WORKDIR / "Results/vector_refine_comparison_20260707/run_manifest.csv"
DEFAULT_ALPHA_MANIFEST = WORKDIR / "Results/vio_alpha_ablation_comparison_20260707/run_manifest.csv"


@dataclass(frozen=True)
class BackendVariant:
    name: str
    config: SlidingWindowConfig


VARIANTS = {
    "swvio_pv001_w10": BackendVariant(
        name="swvio_pv001_w10",
        config=SlidingWindowConfig(
            window_size=8,
            stride=8,
            max_nfev=4,
            visual_position_weight=0.5,
            visual_rotation_weight=0.5,
            imu_position_weight=1.0,
            imu_velocity_weight=1.0,
            imu_rotation_weight=10.0,
            anchor_position_weight=100.0,
            anchor_rotation_weight=100.0,
            anchor_velocity_weight=2.0,
            velocity_damping_weight=1e-3,
        ),
    ),
    "swvio_Ronly_w10": BackendVariant(
        name="swvio_Ronly_w10",
        config=SlidingWindowConfig(
            window_size=8,
            stride=8,
            max_nfev=4,
            visual_position_weight=1.0,
            visual_rotation_weight=0.5,
            imu_position_weight=0.0,
            imu_velocity_weight=0.0,
            imu_rotation_weight=10.0,
            anchor_position_weight=100.0,
            anchor_rotation_weight=100.0,
            anchor_velocity_weight=2.0,
            velocity_damping_weight=1e-3,
        ),
    ),
    "swvio_pvstrong_w8": BackendVariant(
        name="swvio_pvstrong_w8",
        config=SlidingWindowConfig(
            window_size=8,
            stride=8,
            max_nfev=4,
            visual_position_weight=0.1,
            visual_rotation_weight=0.3,
            imu_position_weight=20.0,
            imu_velocity_weight=5.0,
            imu_rotation_weight=10.0,
            anchor_position_weight=100.0,
            anchor_rotation_weight=100.0,
            anchor_velocity_weight=2.0,
            velocity_damping_weight=1e-3,
        ),
    ),
}


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def pick_manifest_row(rows: list[dict[str, str]], scene: str, variant: str) -> dict[str, str]:
    matches = [row for row in rows if row.get("scene") == scene and row.get("variant") == variant]
    if len(matches) != 1:
        raise RuntimeError(f"expected one manifest row for {scene} / {variant}, got {len(matches)}")
    return matches[0]


def quat_xyzw_to_matrix(q: np.ndarray) -> np.ndarray:
    return Rotation.from_quat(np.asarray(q, dtype=np.float64).reshape(4)).as_matrix()


def matrix_to_quat_xyzw(R: np.ndarray) -> np.ndarray:
    q = Rotation.from_matrix(np.asarray(R, dtype=np.float64).reshape(3, 3)).as_quat()
    if q[3] < 0.0:
        q *= -1.0
    return q / max(float(np.linalg.norm(q)), 1e-15)


def load_visual_states(row: dict[str, str]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    bundle = find_output_bundle(Path(row["result_dir"]))
    frame = pair_analysis.read_pose_frame(bundle.bundle_dir)
    if frame != "NWU":
        raise ValueError(f"minimal sliding-window backend expects NWU visual poses, got {frame}")
    poses = pd.read_csv(bundle.poses_path)
    time_ns = poses["timestamp_ns"].to_numpy(np.int64)
    position = poses[["tx", "ty", "tz"]].to_numpy(np.float64)
    quat = poses[["qx", "qy", "qz", "qw"]].to_numpy(np.float64)
    rotation = np.stack([quat_xyzw_to_matrix(q) for q in quat], axis=0)
    t_s = time_ns.astype(np.float64) * 1e-9
    velocity = np.gradient(position, t_s, axis=0, edge_order=1)
    return time_ns, position, rotation, velocity


def load_imu(scene_root: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    imu = pd.read_csv(scene_root / "imu_data.csv")
    time_ns = imu["timestamp"].to_numpy(np.int64)
    acc = imu[["lin_acc_x", "lin_acc_y", "lin_acc_z"]].to_numpy(np.float64)
    gyro = imu[["ang_vel_x", "ang_vel_y", "ang_vel_z"]].to_numpy(np.float64)
    return time_ns, acc, gyro


def interpolate_samples(
    time_ns: np.ndarray,
    values: np.ndarray,
    target_time_ns: np.ndarray,
) -> np.ndarray:
    out = np.zeros((len(target_time_ns), values.shape[1]), dtype=np.float64)
    t = time_ns.astype(np.float64)
    targets = target_time_ns.astype(np.float64)
    for axis in range(values.shape[1]):
        out[:, axis] = np.interp(targets, t, values[:, axis])
    return out


def interval_imu_samples(
    imu_time_ns: np.ndarray,
    acc: np.ndarray,
    gyro: np.ndarray,
    t0: int,
    t1: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lo = min(int(t0), int(t1))
    hi = max(int(t0), int(t1))
    mask = (imu_time_ns > lo) & (imu_time_ns < hi)
    stamps = np.concatenate([[lo], imu_time_ns[mask], [hi]]).astype(np.int64)
    stamps = np.unique(stamps)
    acc_i = interpolate_samples(imu_time_ns, acc, stamps)
    gyro_i = interpolate_samples(imu_time_ns, gyro, stamps)
    return stamps, acc_i, gyro_i


def preintegrate_interval(
    *,
    imu_time_ns: np.ndarray,
    acc: np.ndarray,
    gyro: np.ndarray,
    t0: int,
    t1: int,
    R0_bw: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    stamps, acc_i, gyro_i = interval_imu_samples(imu_time_ns, acc, gyro, t0, t1)
    q0 = matrix_to_quat_xyzw(R0_bw)
    result = preintegrate_imu(
        time_ns=torch.as_tensor(stamps, dtype=torch.int64),
        acc=torch.as_tensor(acc_i, dtype=torch.float32),
        gyro=torch.as_tensor(gyro_i, dtype=torch.float32),
        R0_world=pp.SO3(torch.as_tensor(q0, dtype=torch.float32).reshape(1, 4)),
        gravity=-9.8,
        sigma_acc=0.01,
        sigma_gyro=0.001,
    )
    return (
        result.delta_p.detach().cpu().numpy().reshape(3).astype(np.float64),
        result.delta_v.detach().cpu().numpy().reshape(3).astype(np.float64),
        result.delta_R.matrix().detach().cpu().numpy().reshape(3, 3).astype(np.float64),
        float(result.dt_total),
    )


def build_sequence(scene_root: Path, visual_row: dict[str, str]) -> SlidingWindowSequence:
    time_ns, position, rotation, velocity = load_visual_states(visual_row)
    imu_time_ns, acc, gyro = load_imu(scene_root)
    edges: list[SlidingWindowEdge] = []
    for i in range(len(time_ns) - 1):
        R_i = rotation[i]
        R_j = rotation[i + 1]
        visual_delta_p = R_i.T @ (position[i + 1] - position[i])
        visual_delta_R = R_i.T @ R_j
        imu_delta_p, imu_delta_v, imu_delta_R, dt = preintegrate_interval(
            imu_time_ns=imu_time_ns,
            acc=acc,
            gyro=gyro,
            t0=int(time_ns[i]),
            t1=int(time_ns[i + 1]),
            R0_bw=R_i,
        )
        edges.append(
            SlidingWindowEdge(
                i=i,
                j=i + 1,
                dt=dt,
                visual_delta_p_body=visual_delta_p,
                visual_delta_R=visual_delta_R,
                imu_delta_p_body=imu_delta_p,
                imu_delta_v_body=imu_delta_v,
                imu_delta_R=imu_delta_R,
            )
        )
    return SlidingWindowSequence(
        time_ns=time_ns,
        position_w=position,
        rotation_bw=rotation,
        velocity_w=velocity,
        edges=edges,
    )


def write_result(
    *,
    result_dir: Path,
    scene_root: Path,
    variant: BackendVariant,
    result,
) -> None:
    result_dir.mkdir(parents=True, exist_ok=True)
    with (result_dir / "poses.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp_ns", "tx", "ty", "tz", "qx", "qy", "qz", "qw"])
        for ts, p, R in zip(result.time_ns, result.position_w, result.rotation_bw):
            q = matrix_to_quat_xyzw(R)
            writer.writerow([int(ts), *[float(x) for x in p], *[float(x) for x in q]])
    (result_dir / "pose_coordinate_frame.txt").write_text("NWU\n", encoding="utf-8")
    meta = {
        "method": variant.name,
        "scene_root": str(scene_root),
        "window_size": variant.config.window_size,
        "stride": variant.config.stride,
        "max_nfev": variant.config.max_nfev,
        "num_windows": result.num_windows,
        "total_nfev": result.total_nfev,
        "total_cost": result.total_cost,
    }
    (result_dir / "sliding_window_backend.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def run_backend(args: argparse.Namespace) -> Path:
    visual_rows = read_manifest(args.visual_manifest)
    manifest_rows: list[dict[str, str]] = []
    started = datetime.now().isoformat(timespec="seconds")
    for scene in args.scenes:
        scene_root = SCENE_ROOTS[scene]
        visual_row = pick_manifest_row(visual_rows, scene, "pure_macvo_baseline")
        print(f"-- Build sequence: {scene}", flush=True)
        sequence = build_sequence(scene_root, visual_row)
        for variant_name in args.variants:
            variant = VARIANTS[variant_name]
            print(f"-- Optimize: scene={scene} variant={variant.name}", flush=True)
            result = optimize_sliding_window_sequence(sequence, variant.config)
            result_dir = args.result_root / "trial_1" / variant.name / scene
            write_result(
                result_dir=result_dir,
                scene_root=scene_root,
                variant=variant,
                result=result,
            )
            manifest_rows.append(
                {
                    "trial": "1",
                    "scene": scene,
                    "variant": variant.name,
                    "imu_trans_prior_mode": "sliding_window_backend",
                    "imu_trans_prior_scale": "",
                    "imu_rot_prior_std": "",
                    "imu_rot_prior_std_when_translation": "",
                    "imu_factor_mode": "minimal_sliding_window_vio",
                    "force_mode": "postprocess",
                    "rot_enabled": "1",
                    "trans_enabled": "1",
                    "scene_root": str(scene_root),
                    "result_dir": str(result_dir),
                    "seq_to": "",
                    "autodiff": "0",
                    "metadata_camera_imu_time_offset_ns": "0",
                    "metadata_time_offset_source": "metadata.time_synchronization.camera_imu_time_offset_ns",
                    "args": f"--minimal-sliding-window-vio --variant {variant.name}",
                    "created_at": started,
                }
            )

    args.result_root.mkdir(parents=True, exist_ok=True)
    manifest_path = args.result_root / "run_manifest.csv"
    fieldnames = [
        "trial",
        "scene",
        "variant",
        "imu_trans_prior_mode",
        "imu_trans_prior_scale",
        "imu_rot_prior_std",
        "imu_rot_prior_std_when_translation",
        "imu_factor_mode",
        "force_mode",
        "rot_enabled",
        "trans_enabled",
        "scene_root",
        "result_dir",
        "seq_to",
        "autodiff",
        "metadata_camera_imu_time_offset_ns",
        "metadata_time_offset_source",
        "args",
        "created_at",
    ]
    with manifest_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)
    return manifest_path


def add_rows_from_manifest(
    rows: list[dict[str, str]],
    source_manifest: Path,
    scenes: list[str],
    variants: list[str],
) -> None:
    source_rows = read_manifest(source_manifest)
    for scene in scenes:
        for variant in variants:
            try:
                rows.append(pick_manifest_row(source_rows, scene, variant))
            except RuntimeError:
                continue


def write_comparison_manifest(args: argparse.Namespace) -> Path:
    rows: list[dict[str, str]] = []
    add_rows_from_manifest(
        rows,
        args.visual_manifest,
        args.scenes,
        ["pure_macvo_baseline", "imuatt_vector_refine"],
    )
    add_rows_from_manifest(
        rows,
        args.alpha_manifest,
        [s for s in args.scenes if "rectangle" in s],
        ["alpha_R_only", "alpha_pv_0p01", "alpha_pv_0p1"],
    )
    add_rows_from_manifest(rows, args.result_root / "run_manifest.csv", args.scenes, list(args.variants))

    args.compare_root.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    manifest_path = args.compare_root / "run_manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})
    return manifest_path


def analyze(args: argparse.Namespace) -> int:
    write_comparison_manifest(args)
    return closed_paths.analyze_batch(
        argparse.Namespace(
            result_root=args.compare_root,
            output_root=args.output_root,
            scenes=args.scenes,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--compare-root", type=Path, default=DEFAULT_COMPARE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--visual-manifest", type=Path, default=DEFAULT_VISUAL_MANIFEST)
    parser.add_argument("--alpha-manifest", type=Path, default=DEFAULT_ALPHA_MANIFEST)
    parser.add_argument("--scenes", nargs="*", default=list(SCENE_ROOTS))
    parser.add_argument("--variants", nargs="*", default=list(VARIANTS))
    parser.add_argument("--run-only", action="store_true")
    parser.add_argument("--analyze-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.analyze_only:
        manifest = run_backend(args)
        print(f"Wrote sliding-window manifest: {manifest}")
    if args.run_only:
        return 0
    rc = analyze(args)
    if rc == 0:
        print(f"Wrote analysis: {args.output_root}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
