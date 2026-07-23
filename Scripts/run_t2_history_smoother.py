#!/usr/bin/env python3
"""Run the read-only T2 full-history smoother on a production tensor archive."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Utility.PoseFrame import convert_pose_world_frame_only
from Utility.T2HistorySmoother import (
    compressed_factor_equivalence,
    load_t2_history_archive,
    smooth_t2_history,
    state_arrays,
)
from Utility.TwoStateVIO import state_boxminus


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tensor-map", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--start-frame", type=int, default=90)
    parser.add_argument("--end-frame", type=int, default=299)
    parser.add_argument("--max-iterations", type=int, default=12)
    parser.add_argument("--initial-damping", type=float, default=1.0e-3)
    return parser.parse_args()


def write_states(
    path: Path,
    frame_indices: np.ndarray,
    timestamps_ns: np.ndarray,
    arrays: dict[str, np.ndarray],
) -> None:
    pose_internal = arrays["pose_WB_internal_ned"]
    pose_nwu = convert_pose_world_frame_only(pose_internal, "NED", "NWU")
    velocity_internal = arrays["velocity_W_internal_ned"]
    velocity_nwu = velocity_internal.copy()
    velocity_nwu[:, 1:3] *= -1.0
    header = [
        "frame", "timestamp_ns",
        "tx_nwu", "ty_nwu", "tz_nwu", "qx_nwu", "qy_nwu", "qz_nwu", "qw_nwu",
        "vx_nwu", "vy_nwu", "vz_nwu",
        "ba_x_internal", "ba_y_internal", "ba_z_internal",
        "bg_x_internal", "bg_y_internal", "bg_z_internal",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        for row in range(len(frame_indices)):
            writer.writerow(
                [int(frame_indices[row]), int(timestamps_ns[row])]
                + pose_nwu[row].tolist()
                + velocity_nwu[row].tolist()
                + arrays["acc_bias_internal"][row].tolist()
                + arrays["gyro_bias_internal"][row].tolist()
            )


def main() -> None:
    args = parse_args()
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    archive = load_t2_history_archive(
        args.tensor_map,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
    )
    equivalence = compressed_factor_equivalence(archive)
    result = smooth_t2_history(
        archive,
        max_iterations=args.max_iterations,
        initial_damping=args.initial_damping,
    )

    online_arrays = state_arrays(archive.online_states)
    smoothed_arrays = state_arrays(result.states)
    write_states(
        output / "t2_online_states.csv",
        archive.frame_indices,
        archive.timestamps_ns,
        online_arrays,
    )
    write_states(
        output / "t2_smoothed_states.csv",
        archive.frame_indices,
        archive.timestamps_ns,
        smoothed_arrays,
    )

    with (output / "t2_smoothing_corrections_per_frame.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "frame", "timestamp_ns", "dtx", "dty", "dtz", "drx", "dry", "drz",
                "dvx", "dvy", "dvz", "dba_x", "dba_y", "dba_z", "dbg_x", "dbg_y", "dbg_z",
            ]
        )
        for frame, timestamp, online, smoothed in zip(
            archive.frame_indices,
            archive.timestamps_ns,
            archive.online_states,
            result.states,
        ):
            correction = state_boxminus(smoothed, online).detach().cpu().numpy()
            writer.writerow([int(frame), int(timestamp), *correction.tolist()])

    with (output / "t2_smoother_iterations.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=list(result.iterations[0].__dataclass_fields__)
            if result.iterations
            else ["iteration"],
        )
        writer.writeheader()
        for iteration in result.iterations:
            writer.writerow(iteration.__dict__)

    corrections = np.stack(
        [
            state_boxminus(smoothed, online).detach().cpu().numpy()
            for online, smoothed in zip(archive.online_states, result.states)
        ]
    )
    summary = {
        "schema_version": 1,
        "mode": "read_only_full_history_no_feedback",
        "tensor_map": str(archive.source_path),
        "tensor_map_sha256": archive.source_sha256,
        "frame_start": int(archive.frame_indices[0]),
        "frame_end": int(archive.frame_indices[-1]),
        "state_count": len(archive.online_states),
        "edge_count": len(archive.edges),
        "coordinate_contract": {
            "optimizer_pose": "T_WI",
            "optimizer_world": "MACVO internal NED",
            "csv_pose_world": "NWU",
            "reference_point": "IMU origin",
            "state_tangent": "right [translation, rotation, velocity, acc_bias, gyro_bias]",
        },
        "compressed_factor_equivalence": equivalence,
        "solver": {
            "initial_cost": result.initial_cost,
            "final_cost": result.final_cost,
            "cost_reduction": result.initial_cost - result.final_cost,
            "converged": result.converged,
            "convergence_reason": result.convergence_reason,
            "iterations": len(result.iterations),
            "final_gradient_inf_norm": result.final_gradient_inf_norm,
            "elapsed_s": result.elapsed_s,
        },
        "online_immutability": {
            "online_archive_was_written": False,
            "smoother_returns_new_state_objects": True,
        },
        "correction": {
            "translation_norm_median": float(np.median(np.linalg.norm(corrections[:, :3], axis=1))),
            "translation_norm_p95": float(np.percentile(np.linalg.norm(corrections[:, :3], axis=1), 95)),
            "translation_norm_max": float(np.max(np.linalg.norm(corrections[:, :3], axis=1))),
            "rotation_norm_rad_median": float(np.median(np.linalg.norm(corrections[:, 3:6], axis=1))),
            "rotation_norm_rad_p95": float(np.percentile(np.linalg.norm(corrections[:, 3:6], axis=1), 95)),
            "rotation_norm_rad_max": float(np.max(np.linalg.norm(corrections[:, 3:6], axis=1))),
        },
    }
    (output / "t2_history_smoother_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
