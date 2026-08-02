"""Persist the causal pose committed after every backend update."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np

from Utility.PoseFrame import convert_pose_world_frame_only
from Utility.TrajectoryReference import compose_camera_to_imu_poses


ONLINE_POSE_HEADER = (
    "update_index",
    "frame_idx",
    "timestamp_ns",
    "tx",
    "ty",
    "tz",
    "qx",
    "qy",
    "qz",
    "qw",
    "backend",
    "history_revision",
    "state_count",
)


class OnlinePoseRecorder:
    """Record each state once, before later iSAM2 history revisions.

    The canonical final trajectory is written by :class:`IOdometry` after the
    complete sequence.  This sidecar records the state that was available to a
    real-time consumer immediately after each backend writeback.
    """

    def __init__(self, path: Path, *, output_world_frame: str = "NWU") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.output_world_frame = str(output_world_frame).strip().upper()
        self._stream = self.path.open("w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._stream, fieldnames=ONLINE_POSE_HEADER)
        self._writer.writeheader()
        self._written_frames: set[int] = set()
        self._update_index = 0
        self._closed = False

    def __call__(self, system: Any) -> None:
        if self._closed or not hasattr(system, "graph"):
            return
        frames = system.graph.frames
        diagnostics = getattr(system.Optimizer, "last_pair_diagnostics", {}) or {}
        latest = diagnostics.get("frame_idx")
        if latest is None:
            return
        latest = int(latest)
        if latest < 0 or latest >= len(frames):
            return

        # The first callback also persists the anchor state.  Subsequent calls
        # normally add exactly one new state, while delayed writeback may add a
        # small contiguous range.
        pending = [idx for idx in range(latest + 1) if idx not in self._written_frames]
        if not pending:
            return

        data = frames.data
        required = ("pose", "time_ns", "imu_vio_sensor_T_imu")
        if any(name not in data for name in required):
            return
        indices = np.asarray(pending, dtype=np.int64)
        camera = data["pose"].tensor[indices].detach().cpu().double().numpy()
        extrinsic = (
            data["imu_vio_sensor_T_imu"].tensor[indices]
            .detach().cpu().double().numpy()
        )
        timestamps = data["time_ns"].tensor[indices].detach().cpu().numpy()
        imu_internal = compose_camera_to_imu_poses(camera, extrinsic)
        imu_output = convert_pose_world_frame_only(
            imu_internal,
            "NED",
            self.output_world_frame,
        )
        backend = str(diagnostics.get("vio_backend", "unknown"))
        history_revision = int(bool(diagnostics.get("isam2_history_revision", False)))
        state_count = diagnostics.get("isam2_state_count")
        state_count = "" if state_count is None else int(state_count)

        for frame_idx, timestamp, pose in zip(pending, timestamps, imu_output):
            self._writer.writerow({
                "update_index": self._update_index,
                "frame_idx": frame_idx,
                "timestamp_ns": int(timestamp),
                "tx": f"{float(pose[0]):.17g}",
                "ty": f"{float(pose[1]):.17g}",
                "tz": f"{float(pose[2]):.17g}",
                "qx": f"{float(pose[3]):.17g}",
                "qy": f"{float(pose[4]):.17g}",
                "qz": f"{float(pose[5]):.17g}",
                "qw": f"{float(pose[6]):.17g}",
                "backend": backend,
                "history_revision": history_revision,
                "state_count": state_count,
            })
            self._written_frames.add(frame_idx)
            self._update_index += 1
        self._stream.flush()

    def close(self) -> None:
        if self._closed:
            return
        self._stream.close()
        self._closed = True
