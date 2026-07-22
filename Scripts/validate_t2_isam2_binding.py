#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
import pypose as pp
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Utility.T2FactorPacket import T2FactorPacket
from Utility.T2HistorySmoother import load_t2_history_archive
from Utility.T2ISAM2Backend import IncrementalT2ISAM2Backend


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tensor-map", required=True, type=Path)
    parser.add_argument("--start-frame", type=int, default=90)
    parser.add_argument("--end-frame", type=int, default=299)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--reference-states", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    archive = load_t2_history_archive(
        args.tensor_map,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
    )
    sigma = torch.diag(archive.initial_prior.sqrt_information).reciprocal()
    backend = IncrementalT2ISAM2Backend(
        initial_prior_std={
            "pose_translation_std": float(sigma[0].item()),
            "pose_rotation_std": float(sigma[3].item()),
            "velocity_std": float(sigma[6].item()),
            "acc_bias_std": float(sigma[9].item()),
            "gyro_bias_std": float(sigma[12].item()),
        }
    )
    timing_rows = []
    for local_edge, edge in enumerate(archive.edges):
        packet = T2FactorPacket.create(
            frame_i=edge.frame_i,
            frame_j=edge.frame_j,
            state_i_initial=archive.online_states[local_edge],
            state_j_initial=archive.online_states[local_edge + 1],
            imu=edge.imu,
            visual=edge.visual,
            extrinsic_CI=archive.extrinsic_CI,
        )
        update = backend.consume(packet)
        timing_rows.append(
            {
                "frame_i": edge.frame_i,
                "frame_j": edge.frame_j,
                "update_ms": update.update_ms,
                "initial_pose_mismatch_norm": update.initial_pose_mismatch_norm,
                "initial_velocity_mismatch_norm": update.initial_velocity_mismatch_norm,
                "initial_bias_mismatch_norm": update.initial_bias_mismatch_norm,
            }
        )

    history = backend.history()
    rows = []
    for local, (frame, state) in enumerate(history):
        pose = state.pose_WB.reshape(7).numpy()
        rows.append(
            [local, frame]
            + pose.tolist()
            + state.velocity_W.numpy().tolist()
            + state.acc_bias.numpy().tolist()
            + state.gyro_bias.numpy().tolist()
        )
    state_path = output / "binding_states_internal.csv"
    with state_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["local_index", "frame", "tx", "ty", "tz", "qx", "qy", "qz", "qw"]
            + ["vx", "vy", "vz", "ba_x", "ba_y", "ba_z", "bg_x", "bg_y", "bg_z"]
        )
        writer.writerows(rows)
    with (output / "binding_timing.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(timing_rows[0]))
        writer.writeheader()
        writer.writerows(timing_rows)

    summary = {
        "state_count": len(history),
        "edge_count": len(timing_rows),
        "update_median_ms": float(np.median([row["update_ms"] for row in timing_rows])),
        "update_p95_ms": float(np.percentile([row["update_ms"] for row in timing_rows], 95)),
        "update_max_ms": float(np.max([row["update_ms"] for row in timing_rows])),
        "finite": bool(np.isfinite(np.asarray(rows, dtype=np.float64)).all()),
    }
    if args.reference_states is not None:
        reference = np.genfromtxt(
            args.reference_states.expanduser().resolve(),
            delimiter=",",
            names=True,
        )
        if len(reference) != len(rows):
            raise ValueError("binding/reference state counts differ")
        binding = np.asarray(rows, dtype=np.float64)
        reference_pose = np.column_stack(
            [reference[name] for name in ("tx", "ty", "tz", "qx", "qy", "qz", "qw")]
        )
        binding_pose = binding[:, 2:9]
        pose_error = (
            pp.SE3(torch.as_tensor(reference_pose, dtype=torch.float64)).Inv()
            @ pp.SE3(torch.as_tensor(binding_pose, dtype=torch.float64))
        ).Log().tensor().numpy()
        reference_vectors = np.column_stack(
            [
                reference[name]
                for name in (
                    "vx", "vy", "vz", "ba_x", "ba_y", "ba_z", "bg_x", "bg_y", "bg_z"
                )
            ]
        )
        vector_error = binding[:, 9:18] - reference_vectors
        summary.update(
            {
                "reference_states": str(args.reference_states.resolve()),
                "pose_tangent_max_abs": float(np.max(np.abs(pose_error))),
                "pose_tangent_rmse": float(np.sqrt(np.mean(pose_error ** 2))),
                "vector_max_abs": float(np.max(np.abs(vector_error))),
                "vector_rmse": float(np.sqrt(np.mean(vector_error ** 2))),
            }
        )
    (output / "binding_validation_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
