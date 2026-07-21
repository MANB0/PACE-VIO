#!/usr/bin/env python3
"""Plot the fixed-input Gate-5 T0/U1/T2 trajectories at the IMU center."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Scripts.plot_static63_gt_macvo import HTML_TEMPLATE  # noqa: E402
from Scripts.run_uvd_compressed_pose_factor_gate5 import (  # noqa: E402
    DEFAULT_PACKET,
    T0,
    T2,
    interpolate_bias_truth,
    load_truth,
    state_rows,
)


DEFAULT_OUTPUT = ROOT / "analysis_uvd_compressed_pose_factor_gate5_20260720"
DEFAULT_STATE_CSV = DEFAULT_OUTPUT / "gate5_state_truth_errors.csv"
U1 = "U1_direct_uvd_baseline"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, default=DEFAULT_PACKET)
    parser.add_argument("--state-csv", type=Path, default=DEFAULT_STATE_CSV)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--edge-count", type=int, default=209)
    return parser.parse_args()


def extract_u1_states(edges: list[dict], edge_count: int) -> dict[int, object]:
    selected = edges[:edge_count]
    if len(selected) != edge_count:
        raise ValueError(f"packet has {len(selected)} edges, expected {edge_count}")
    states: dict[int, object] = {}
    for index, edge in enumerate(selected):
        frame_i = int(edge["frame_i"])
        frame_j = int(edge["frame_j"])
        if index and frame_i != int(selected[index - 1]["frame_j"]):
            raise ValueError(f"non-contiguous U1 edge at {frame_i}->{frame_j}")
        baseline = edge["baseline"]
        states[frame_i] = baseline["state_i"]
        if index == len(selected) - 1:
            states[frame_j] = baseline["state_j"]
    return states


def summarize(state_frame: pd.DataFrame) -> dict[str, dict[str, float | int]]:
    summary: dict[str, dict[str, float | int]] = {}
    for method in (T0, U1, T2):
        state = state_frame[state_frame.method == method].sort_values("frame")
        if state.empty:
            raise ValueError(f"missing state rows for {method}")
        summary[method] = {
            "state_count": int(len(state)),
            "frame_start": int(state.frame.iloc[0]),
            "frame_end": int(state.frame.iloc[-1]),
            "xy_ate_rmse_m": float(
                np.sqrt(np.mean(np.square(state.position_error_xy_norm_m)))
            ),
            "position_3d_rmse_m": float(
                np.sqrt(np.mean(np.square(state.position_error_3d_norm_m)))
            ),
            "orientation_rmse_deg": float(
                np.degrees(
                    np.sqrt(np.mean(np.square(state.orientation_error_norm_rad)))
                )
            ),
            "velocity_rmse_mps": float(
                np.sqrt(np.mean(np.square(state.velocity_error_norm_mps)))
            ),
        }
    return summary


def plot_html(output: Path, state_frame: pd.DataFrame) -> Path:
    def ned_to_nwu(values: np.ndarray) -> list[list[float]]:
        converted = np.asarray(values, dtype=np.float64).copy()
        converted[:, 1:] *= -1.0
        return converted.tolist()

    def method_payload(method: str) -> tuple[pd.DataFrame, list, list, list]:
        state = state_frame[state_frame.method == method].sort_values("frame")
        xyz = ned_to_nwu(
            state[["position_est_x", "position_est_y", "position_est_z"]].to_numpy()
        )
        forward = ned_to_nwu(
            state[["forward_est_x", "forward_est_y", "forward_est_z"]].to_numpy()
        )
        error = state.position_error_3d_norm_m.astype(float).tolist()
        return state, xyz, forward, error

    t0_state, t0_xyz, t0_forward, t0_error = method_payload(T0)
    _, u1_xyz, u1_forward, u1_error = method_payload(U1)
    _, t2_xyz, t2_forward, t2_error = method_payload(T2)
    gt_xyz = ned_to_nwu(
        t0_state[["position_gt_x", "position_gt_y", "position_gt_z"]].to_numpy()
    )
    gt_forward = ned_to_nwu(
        t0_state[["forward_gt_x", "forward_gt_y", "forward_gt_z"]].to_numpy()
    )

    def fusion_entry(
        *, key: str, label: str, color: str, xyz: list, forward: list, error: list
    ) -> dict:
        return {
            "key": key,
            "source": key,
            "config": "normal_noise",
            "label": label,
            "color": color,
            "dasharray": "",
            "scene": "clear_stop_turn_rectangle_truth_normal_noise",
            "xyz": xyz,
            "forward": forward,
            "error_m": error,
            "metrics": {
                "frames": int(len(error)),
                "rmse_m": float(np.sqrt(np.mean(np.square(error)))),
            },
            "path": str(output / "t0_u1_t2_state_truth_errors.csv"),
        }

    payload = {
        "scene": "Rectangle normal-noise / frames 90-299 / IMU center",
        "gt": gt_xyz,
        "gt_forward": gt_forward,
        "macvo": t0_xyz,
        "macvo_forward": t0_forward,
        "time_s": (
            (t0_state.timestamp.to_numpy() - t0_state.timestamp.iloc[0]) * 1.0e-9
        ).tolist(),
        "error_m": t0_error,
        "metrics": {
            "frames": int(len(t0_error)),
            "rmse_m": float(np.sqrt(np.mean(np.square(t0_error)))),
        },
        "fusion": [
            fusion_entry(
                key="u1_direct_uvd",
                label="U1 · Direct UVD factor",
                color="#16a34a",
                xyz=u1_xyz,
                forward=u1_forward,
                error=u1_error,
            ),
            fusion_entry(
                key="t2_uvd_compressed",
                label="T2 · Compressed UVD pose factor",
                color="#2563eb",
                xyz=t2_xyz,
                forward=t2_forward,
                error=t2_error,
            ),
        ],
        "imu_only": [],
        "gt_path": "in-memory IMU-center truth",
        "macvo_path": str(output / "t0_u1_t2_state_truth_errors.csv"),
    }
    template = HTML_TEMPLATE.replace("{{", "{").replace("}}", "}")
    template = template.replace(
        "Circle, stop-turn rectangle, and straight trajectory comparison",
        "Gate 5: T0 vs U1 vs T2 on identical captured inputs",
    )
    template = template.replace(
        "__METHOD_SCOPE__",
        "T0 relative-pose factor, U1 direct UVD factor, and T2 locally compressed UVD factor",
    )
    template = template.replace(
        "__LINE_NOTE__",
        "Frames 90-299. All estimates and truth are at the IMU center and translation-rebased at frame 90; no rotation, yaw, scale, or SE(3) fitting.",
    )
    template = template.replace("Pure MACVO</span>", "T0 · Relative-pose factor</span>")
    html = template.replace(
        "__DATA__", json.dumps({"scenes": [payload]}, ensure_ascii=False)
    )
    path = output / "interactive_t0_u1_t2_300frames.html"
    path.write_text(html, encoding="utf-8")
    return path


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    payload = torch.load(args.packet.resolve(), map_location="cpu", weights_only=False)
    u1_states = extract_u1_states(payload["edges"], args.edge_count)

    truth = load_truth(Path(payload["dataset"]))
    ref = pd.read_csv(Path(payload["dataset"]) / "ref_pose.csv")
    timestamps = ref["timestamp"].to_numpy(np.int64)
    ba_truth, bg_truth = interpolate_bias_truth(Path(payload["dataset"]), timestamps)
    u1_rows = state_rows(
        method=U1,
        states=u1_states,
        truth=truth,
        timestamps=timestamps,
        ba_truth=ba_truth,
        bg_truth=bg_truth,
    )

    existing = pd.read_csv(args.state_csv.resolve())
    existing = existing[existing.method.isin((T0, T2))]
    combined = pd.concat([existing, pd.DataFrame(u1_rows)], ignore_index=True)
    expected_frames = set(range(90, 300))
    for method in (T0, U1, T2):
        frames = set(combined.loc[combined.method == method, "frame"].astype(int))
        if frames != expected_frames:
            raise AssertionError(
                f"{method} frame mismatch: missing={sorted(expected_frames - frames)[:5]}, "
                f"extra={sorted(frames - expected_frames)[:5]}"
            )
    numeric = combined.select_dtypes(include=[np.number]).to_numpy()
    if not np.isfinite(numeric).all():
        raise FloatingPointError("T0/U1/T2 state table contains NaN/Inf")

    csv_path = output / "t0_u1_t2_state_truth_errors.csv"
    combined.to_csv(csv_path, index=False)
    summary = {
        "comparison_contract": {
            "dataset": payload["dataset"],
            "frame_range": [90, 299],
            "state_count_per_method": 210,
            "reference_point": "IMU center",
            "comparison_origin": "translation-only rebase at frame 90",
            "alignment": "none",
            "u1_source": "baseline state_i/state_j stored in the same captured U1 packet used by Gate-5 T0/T2 replay",
        },
        "methods": summarize(combined),
    }
    summary_path = output / "t0_u1_t2_trajectory_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    html_path = plot_html(output, combined)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"CSV: {csv_path}")
    print(f"Summary: {summary_path}")
    print(f"HTML: {html_path}")


if __name__ == "__main__":
    main()
