#!/usr/bin/env python3
"""Replay a full PACE archive with moving-window initialization.

This is an offline validation driver. It reuses the production compressed
visual and IMU factors, never reads ground truth, and replays from frame zero so
the initialization window remains present in the output trajectory.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import time

import numpy as np
import pypose as pp
import torch
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Utility.MotionWindowInitializer import (
    _transform_pose_world,
    estimate_motion_window_initialization,
)
from Utility.PACEFactorPacket import PACEFactorPacket
from Utility.PACEISAM2Backend import IncrementalPACEISAM2Backend
from Utility.PoseFrame import convert_pose_world_frame_only
from Utility.T2HistorySmoother import load_t2_history_archive
from Utility.TwoStateVIO import (
    ImuPreintegrationFactor,
    LinearizedUVDPoseFactor,
    NavigationState,
)
from Utility.UVDArchiveReconstruction import reconstruct_uvd_normal_equations


PRIOR_STD = {
    "pose_translation_std": 1.0e-5,
    "pose_rotation_std": 1.0e-5,
    "velocity_std": 0.05,
    "acc_bias_std": 0.2,
    "gyro_bias_std": 0.02,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tensor-map", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--window-duration-s", type=float, default=5.0)
    parser.add_argument("--raw-camera-poses", type=Path, default=None)
    parser.add_argument("--uvd-cache", type=Path, default=None)
    parser.add_argument("--estimate-acc-bias", action="store_true")
    parser.add_argument("--fix-gyro-bias", action="store_true")
    parser.add_argument("--end-frame", type=int, default=-1)
    parser.add_argument("--progress-every", type=int, default=500)
    parser.add_argument("--torch-threads", type=int, default=1)
    return parser.parse_args()


def _frame_count(path: Path) -> int:
    with np.load(path, allow_pickle=False) as data:
        return int(data["frames//pose"].shape[0])


def _load_replay_payload(tensor_map: Path, uvd_cache: Path, end_frame: int):
    names = (
        "frames//pose",
        "frames//time_ns",
        "frames//imu_vio_velocity_world",
        "frames//imu_vio_sensor_T_imu",
        "frames//imu_vio_delta_rotvec",
        "frames//imu_vio_delta_v",
        "frames//imu_vio_delta_p",
        "frames//imu_vio_cov",
        "frames//imu_vio_dt",
        "frames//imu_vio_bias_jacobian",
        "frames//imu_vio_linearized_acc_bias",
        "frames//imu_vio_linearized_gyro_bias",
        "frames//imu_vio_bias_rw_cov",
        "frames//imu_vio_gravity_world",
        "frames//imu_vio_gravity_in_residual",
    )
    with np.load(tensor_map, allow_pickle=False) as data:
        payload = {name: np.asarray(data[name])[: end_frame + 1].copy() for name in names}
    with np.load(uvd_cache, allow_pickle=False) as data:
        payload["visual_reference"] = np.asarray(data["reference_CjCi"])[
            : end_frame + 1
        ].copy()
        hessian = np.asarray(data["hessian"])[1 : end_frame + 1].copy()
        gradient = np.asarray(data["gradient"])[1 : end_frame + 1].copy()
    hessian = 0.5 * (hessian + np.swapaxes(hessian, -1, -2))
    values, vectors = np.linalg.eigh(hessian)
    if not np.isfinite(values).all() or np.any(values <= 1.0e-10):
        raise ValueError("streaming replay requires full-rank finite UVD Hessians")
    payload["visual_sqrt_information"] = (
        np.sqrt(values)[:, :, None] * np.swapaxes(vectors, -1, -2)
    )
    projected_gradient = np.einsum("nrc,nr->nc", vectors, gradient)
    payload["visual_residual_offset"] = projected_gradient / np.sqrt(values)
    return payload


def _replay_states(payload, initialization) -> tuple[NavigationState, ...]:
    camera_pose = pp.SE3(torch.as_tensor(payload["frames//pose"], dtype=torch.float64))
    extrinsic_ci = pp.SE3(
        torch.as_tensor(payload["frames//imu_vio_sensor_T_imu"][0], dtype=torch.float64)
        .reshape(1, 7)
    )
    body_pose = (camera_pose @ extrinsic_ci).tensor()
    aligned_pose = _transform_pose_world(
        body_pose, initialization.world_alignment_rotation
    )
    velocity = torch.as_tensor(
        payload["frames//imu_vio_velocity_world"], dtype=torch.float64
    )
    aligned_velocity = (
        initialization.world_alignment_rotation.double() @ velocity.mT
    ).mT
    states: list[NavigationState] = []
    for index in range(int(aligned_pose.shape[0])):
        if index == 0:
            states.append(initialization.initial_state)
        else:
            states.append(
                NavigationState(
                    pose_WB=aligned_pose[index].reshape(1, 7),
                    velocity_W=aligned_velocity[index],
                    acc_bias=initialization.initial_state.acc_bias.detach().clone(),
                    gyro_bias=initialization.initial_state.gyro_bias.detach().clone(),
                )
            )
    return tuple(states)


def _run_backend_stream(payload, states, *, progress_every: int):
    backend = IncrementalPACEISAM2Backend(initial_prior_std=PRIOR_STD)
    timing: list[float] = []
    started = time.perf_counter()
    edge_count = len(states) - 1
    extrinsic_ci = torch.as_tensor(
        payload["frames//imu_vio_sensor_T_imu"][0], dtype=torch.float64
    ).reshape(1, 7)
    for edge_index in range(edge_count):
        frame_j = edge_index + 1
        imu = ImuPreintegrationFactor(
            delta_rotation=torch.as_tensor(
                payload["frames//imu_vio_delta_rotvec"][frame_j], dtype=torch.float64
            ),
            delta_velocity=torch.as_tensor(
                payload["frames//imu_vio_delta_v"][frame_j], dtype=torch.float64
            ),
            delta_position=torch.as_tensor(
                payload["frames//imu_vio_delta_p"][frame_j], dtype=torch.float64
            ),
            covariance=torch.as_tensor(
                payload["frames//imu_vio_cov"][frame_j], dtype=torch.float64
            ),
            dt=float(payload["frames//imu_vio_dt"][frame_j]),
            bias_jacobian=torch.as_tensor(
                payload["frames//imu_vio_bias_jacobian"][frame_j], dtype=torch.float64
            ),
            linearized_acc_bias=torch.as_tensor(
                payload["frames//imu_vio_linearized_acc_bias"][frame_j], dtype=torch.float64
            ),
            linearized_gyro_bias=torch.as_tensor(
                payload["frames//imu_vio_linearized_gyro_bias"][frame_j], dtype=torch.float64
            ),
            bias_rw_covariance=torch.as_tensor(
                payload["frames//imu_vio_bias_rw_cov"][frame_j], dtype=torch.float64
            ),
            gravity_world=(
                torch.as_tensor(
                    payload["frames//imu_vio_gravity_world"][frame_j], dtype=torch.float64
                )
                if bool(payload["frames//imu_vio_gravity_in_residual"][frame_j])
                else None
            ),
            gravity_handling=(
                "residual"
                if bool(payload["frames//imu_vio_gravity_in_residual"][frame_j])
                else "preintegration"
            ),
        )
        visual = LinearizedUVDPoseFactor(
            reference_relative_CjCi=torch.as_tensor(
                payload["visual_reference"][frame_j], dtype=torch.float64
            ).reshape(1, 7),
            sqrt_information=torch.as_tensor(
                payload["visual_sqrt_information"][edge_index], dtype=torch.float64
            ),
            residual_offset=torch.as_tensor(
                payload["visual_residual_offset"][edge_index], dtype=torch.float64
            ),
            extrinsic_CI=extrinsic_ci,
            marginal_mode="full",
        )
        packet = PACEFactorPacket.create(
            frame_i=edge_index,
            frame_j=frame_j,
            state_i_initial=states[edge_index],
            state_j_initial=states[frame_j],
            imu=imu,
            visual=visual,
            extrinsic_CI=extrinsic_ci,
        )
        update = backend.consume(packet)
        timing.append(update.update_ms)
        if progress_every > 0 and (
            edge_index == 0
            or (edge_index + 1) % progress_every == 0
            or edge_index + 1 == edge_count
        ):
            print(
                f"replay {edge_index + 1}/{edge_count} edges; "
                f"last iSAM2 update {update.update_ms:.3f} ms",
                flush=True,
            )
    return backend.history(), np.asarray(timing), time.perf_counter() - started


def _write_state_rows(path: Path, frames: np.ndarray, timestamps: np.ndarray, history) -> None:
    rows = []
    for expected_frame, timestamp, (frame, state) in zip(frames, timestamps, history):
        if int(frame) != int(expected_frame):
            raise AssertionError("replayed frame order changed")
        pose = state.pose_WB.reshape(7).detach().cpu().numpy()
        rows.append(
            [int(frame), int(timestamp)]
            + pose.tolist()
            + state.velocity_W.reshape(3).detach().cpu().numpy().tolist()
            + state.acc_bias.reshape(3).detach().cpu().numpy().tolist()
            + state.gyro_bias.reshape(3).detach().cpu().numpy().tolist()
        )
    header = [
        "frame", "timestamp_ns", "tx", "ty", "tz", "qx", "qy", "qz", "qw",
        "vx", "vy", "vz", "ba_x", "ba_y", "ba_z", "bg_x", "bg_y", "bg_z",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)

    poses_internal = np.asarray([row[2:9] for row in rows], dtype=np.float64)
    poses_nwu = convert_pose_world_frame_only(poses_internal, "NED", "NWU")
    with path.with_name(path.stem + "_nwu.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["frame", "timestamp_ns", "tx", "ty", "tz", "qx", "qy", "qz", "qw"])
        for frame, timestamp, pose in zip(frames, timestamps, poses_nwu):
            writer.writerow([int(frame), int(timestamp), *pose.tolist()])


def main() -> int:
    args = parse_args()
    torch.set_num_threads(max(1, int(args.torch_threads)))
    tensor_map = args.tensor_map.expanduser().resolve()
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    count = _frame_count(tensor_map)
    end_frame = count - 1 if args.end_frame < 0 else min(int(args.end_frame), count - 1)
    raw_camera_poses = (
        args.raw_camera_poses.expanduser().resolve()
        if args.raw_camera_poses is not None
        else tensor_map.with_name("macvo_raw_poses_camera.csv")
    )
    uvd_cache = (
        args.uvd_cache.expanduser().resolve()
        if args.uvd_cache is not None
        else output / "reconstructed_uvd_normal_equations.npz"
    )
    reconstruct_uvd_normal_equations(
        tensor_map,
        raw_camera_poses,
        uvd_cache,
        end_frame=end_frame,
        progress_every=args.progress_every,
    )
    with np.load(tensor_map, allow_pickle=False) as data:
        timestamps = np.asarray(data["frames//time_ns"], dtype=np.int64)[: end_frame + 1]
    initialization_cutoff = int(
        timestamps[0] + round(float(args.window_duration_s) * 1.0e9)
    )
    initialization_end = min(
        max(int(np.searchsorted(timestamps, initialization_cutoff, side="right")) - 1, 2),
        end_frame,
    )
    initialization_archive = load_t2_history_archive(
        tensor_map,
        start_frame=0,
        end_frame=initialization_end,
        visual_normal_equations_path=uvd_cache,
        **{
            "pose_translation_std": PRIOR_STD["pose_translation_std"],
            "pose_rotation_std": PRIOR_STD["pose_rotation_std"],
            "velocity_std": PRIOR_STD["velocity_std"],
            "acc_bias_std": PRIOR_STD["acc_bias_std"],
            "gyro_bias_std": PRIOR_STD["gyro_bias_std"],
        },
    )
    initialization = estimate_motion_window_initialization(
        initialization_archive,
        window_duration_s=args.window_duration_s,
        estimate_acc_bias=bool(args.estimate_acc_bias),
        estimate_gyro_bias=not bool(args.fix_gyro_bias),
    )
    payload = _load_replay_payload(tensor_map, uvd_cache, end_frame)
    replay_states = _replay_states(payload, initialization)
    history, timing, elapsed = _run_backend_stream(
        payload,
        replay_states,
        progress_every=args.progress_every,
    )
    _write_state_rows(
        output / "motion_initialized_states_internal.csv",
        np.arange(end_frame + 1, dtype=np.int64),
        timestamps,
        history,
    )
    summary = {
        "schema_version": 1,
        "method": "moving visual-IMU window initialization plus full factor replay",
        "uses_ground_truth": False,
        "source_tensor_map": str(tensor_map),
        "frame_start": 0,
        "frame_end": int(end_frame),
        "state_count": len(history),
        "edge_count": int(end_frame),
        "initialization_window_retained_in_output": len(history) == end_frame + 1,
        "replay_elapsed_s": float(elapsed),
        "isam2_update_median_ms": float(np.median(timing)),
        "isam2_update_p95_ms": float(np.percentile(timing, 95.0)),
        "isam2_update_max_ms": float(np.max(timing)),
        "initialization": initialization.diagnostics,
    }
    (output / "motion_initialization_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
