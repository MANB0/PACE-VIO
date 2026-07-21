#!/usr/bin/env python3
"""Apply the A-E output-filter study to rectangle and straight scenes."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path("/home/admin1/macvo-dev")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Scripts.plot_circle_direct_uvd_u1_vs_pose_factor import (  # noqa: E402
    BATCH_ROOT,
    HTML_TEMPLATE,
    metrics,
    position_errors,
    read_forward_axes,
    read_xyz,
    relative_metrics,
    xy_metrics,
    xyz,
)
from Scripts.plot_output_eskf3d_offline_ablation import (  # noqa: E402
    ABLATIONS,
    METHODS,
    add_two_dimensional_filter_logic,
    detailed_metrics,
    manifest_metrics,
    replace_filter_controls,
)


EKF2D_ROOT = ROOT / "Results/rectangle_straight_output_ekf2d_ablation_20260718"
ESKF3D_ROOT = ROOT / "Results/rectangle_straight_output_eskf3d_ablation_20260718"
OUTPUT = ROOT / "analysis_rectangle_straight_output_eskf3d_ablation_20260718"
PURE_ROOT = ROOT / (
    "Results/visual_factor_cache_static63_unique_source_20260713/"
    "trial_1/pure_macvo"
)


SCENES = {
    "rectangle": {
        "label": "Stop-turn rectangle / Normal noise / Full 63 s",
        "dataset": "clear_stop_turn_rectangle_truth_normal_noise",
        "frame_count": 1890,
        "raw": {
            "t_factor": ROOT / (
                "Results/rectangle_normal_noise_two_state_standard_full_20260715/"
                "trial_1/vio_two_state_fixed_lag_standard_full/"
                "clear_stop_turn_rectangle_truth_normal_noise/poses.csv"
            ),
            "u1": ROOT / (
                "Results/normal_noise_direct_uvd_u1_missing_full_scenes_20260717/"
                "trial_1/vio_two_state_direct_uvd_u1_standard_full/"
                "clear_stop_turn_rectangle_truth_normal_noise/poses.csv"
            ),
            "sa_v1": ROOT / (
                "Results/normal_noise_sa_v1_full_three_scenes_20260717/"
                "trial_1/vio_two_state_direct_uvd_sampling_aware_v1_full/"
                "clear_stop_turn_rectangle_truth_normal_noise/poses.csv"
            ),
            "sa_v2": ROOT / (
                "Results/normal_noise_sa_v2_full_three_scenes_20260717/"
                "trial_1/vio_two_state_direct_uvd_sampling_aware_v2_full/"
                "clear_stop_turn_rectangle_truth_normal_noise/poses.csv"
            ),
        },
    },
    "straight": {
        "label": "Straight / Normal noise / Full 21 s",
        "dataset": "clear_straight_truth_normal_noise",
        "frame_count": 630,
        "raw": {
            "t_factor": ROOT / (
                "Results/circle_straight_normal_noise_two_state_standard_full_20260715/"
                "trial_1/vio_two_state_fixed_lag_standard_full/"
                "clear_straight_truth_normal_noise/poses.csv"
            ),
            "u1": ROOT / (
                "Results/normal_noise_direct_uvd_u1_missing_full_scenes_20260717/"
                "trial_1/vio_two_state_direct_uvd_u1_standard_full/"
                "clear_straight_truth_normal_noise/poses.csv"
            ),
            "sa_v1": ROOT / (
                "Results/normal_noise_sa_v1_full_three_scenes_20260717/"
                "trial_1/vio_two_state_direct_uvd_sampling_aware_v1_full/"
                "clear_straight_truth_normal_noise/poses.csv"
            ),
            "sa_v2": ROOT / (
                "Results/normal_noise_sa_v2_full_three_scenes_20260717/"
                "trial_1/vio_two_state_direct_uvd_sampling_aware_v2_full/"
                "clear_straight_truth_normal_noise/poses.csv"
            ),
        },
    },
}


def trajectory_paths(scene: str, spec: dict[str, object]) -> dict[str, Path]:
    raw_paths = spec["raw"]
    assert isinstance(raw_paths, dict)
    result: dict[str, Path] = {}
    for ablation in ABLATIONS:
        for method in METHODS:
            key = f"{ablation}_{method}"
            if ablation == "A_raw":
                result[key] = Path(raw_paths[method])
            elif ablation == "B_ekf2d":
                result[key] = EKF2D_ROOT / scene / method / "poses.csv"
            else:
                mode = {
                    "C_eskf3d_no_gate": "no_gate",
                    "D_eskf3d_gate": "gate",
                    "E_eskf3d_gate_adaptive": "gate_adaptive",
                }[ablation]
                result[key] = ESKF3D_ROOT / scene / mode / method / "poses.csv"
    return result


def build_scene_payload(
    scene: str, spec: dict[str, object]
) -> tuple[dict[str, object], list[dict[str, object]], dict[str, object]]:
    dataset = str(spec["dataset"])
    expected_frames = int(spec["frame_count"])
    gt_path = BATCH_ROOT / dataset / "ref_pose.csv"
    pure_path = PURE_ROOT / dataset / "poses.csv"
    paths = trajectory_paths(scene, spec)
    for path in (gt_path, pure_path, *paths.values()):
        if not path.exists():
            raise FileNotFoundError(path)

    trajectories = {"GT": read_xyz(gt_path), "pure_macvo": read_xyz(pure_path)}
    trajectories.update({key: read_xyz(path) for key, path in paths.items()})
    lengths = {key: len(rows) for key, rows in trajectories.items()}
    if len(set(lengths.values())) != 1 or lengths["GT"] != expected_frames:
        raise AssertionError(
            f"{scene}: expected aligned {expected_frames}-frame trajectories: {lengths}"
        )
    timestamps = [row[0] for row in trajectories["GT"]]
    for key, rows in trajectories.items():
        if [row[0] for row in rows] != timestamps:
            raise AssertionError(f"{scene}: timestamp mismatch: GT vs {key}")

    forwards = {"GT": read_forward_axes(gt_path), "pure_macvo": read_forward_axes(pure_path)}
    forwards.update({key: read_forward_axes(path) for key, path in paths.items()})
    summary_rows: list[dict[str, object]] = []
    fusion: list[dict[str, object]] = []
    for ablation, ablation_spec in ABLATIONS.items():
        for method, method_spec in METHODS.items():
            key = f"{ablation}_{method}"
            fusion.append(
                {
                    "key": f"{scene}_{key}",
                    "source": key,
                    "config": ablation,
                    "optimizer": method,
                    "label": (
                        f"{method_spec['label']} / {ablation_spec['label']}"
                    ),
                    "color": method_spec["color"],
                    "dasharray": ablation_spec["dasharray"],
                    "scene": dataset,
                    "xyz": xyz(trajectories[key]),
                    "forward": forwards[key],
                    "error_m": position_errors(
                        trajectories["GT"], trajectories[key], xy_only=False
                    ),
                    "metrics": metrics(trajectories["GT"], trajectories[key]),
                    "path": str(paths[key]),
                }
            )
            summary_rows.append(
                {
                    "scene": scene,
                    "frame_count": expected_frames,
                    "ablation": ablation,
                    "method": method,
                    **metrics(trajectories["GT"], trajectories[key]),
                    **xy_metrics(trajectories["GT"], trajectories[key]),
                    **relative_metrics(gt_path, paths[key]),
                    **detailed_metrics(gt_path, paths[key]),
                    **manifest_metrics(paths[key]),
                    "estimate_path": str(paths[key]),
                }
            )

    gt = trajectories["GT"]
    payload = {
        "scene": str(spec["label"]),
        "gt": xyz(gt),
        "gt_forward": forwards["GT"],
        "macvo": xyz(trajectories["pure_macvo"]),
        "macvo_forward": forwards["pure_macvo"],
        "time_s": [(value - timestamps[0]) * 1.0e-9 for value in timestamps],
        "error_m": position_errors(gt, trajectories["pure_macvo"], xy_only=False),
        "metrics": metrics(gt, trajectories["pure_macvo"]),
        "fusion": fusion,
        "imu_only": [],
        "gt_path": str(gt_path),
        "macvo_path": str(pure_path),
    }
    alignment = {
        "scene": scene,
        "dataset": dataset,
        "frame_count": expected_frames,
        "timestamp_start_ns": int(timestamps[0]),
        "timestamp_end_ns": int(timestamps[-1]),
        "all_lengths": lengths,
        "all_timestamps_equal": True,
    }
    return payload, summary_rows, alignment


def main() -> None:
    payloads: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    alignments: list[dict[str, object]] = []
    for scene, spec in SCENES.items():
        payload, rows, alignment = build_scene_payload(scene, spec)
        payloads.append(payload)
        summary_rows.extend(rows)
        alignments.append(alignment)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    metrics_path = OUTPUT / "offline_ablation_metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    (OUTPUT / "offline_ablation_metrics.json").write_text(
        json.dumps(summary_rows, indent=2), encoding="utf-8"
    )
    (OUTPUT / "input_alignment.json").write_text(
        json.dumps(alignments, indent=2), encoding="utf-8"
    )

    template = replace_filter_controls(HTML_TEMPLATE)
    template = template.replace("{{", "{").replace("}}", "}")
    template = add_two_dimensional_filter_logic(template)
    template = template.replace(
        "Circle, stop-turn rectangle, and straight trajectory comparison",
        "Rectangle and straight 3D output-only ESKF offline A-E validation",
    )
    template = template.replace(
        "__METHOD_SCOPE__",
        "T_factor, U1, SA-v1 and SA-v2 on rectangle and straight scenes",
    )
    template = template.replace(
        "__LINE_NOTE__",
        (
            "A is raw VIO, B is the existing XY/yaw EKF, and C-E are the 3D "
            "ESKF modes. Every filter consumes completed pose CSVs only and never "
            "feeds back to VIO. Each scene uses its complete native frame range."
        ),
    )
    html_path = OUTPUT / "interactive_rectangle_straight_output_eskf3d_offline_ablation.html"
    html_path.write_text(
        template.replace(
            "__DATA__", json.dumps({"scenes": payloads}, ensure_ascii=False)
        ),
        encoding="utf-8",
    )

    figure, axes = plt.subplots(1, 2, figsize=(16, 7), constrained_layout=True)
    for axis, (scene, spec) in zip(axes, SCENES.items()):
        dataset = str(spec["dataset"])
        gt = read_xyz(BATCH_ROOT / dataset / "ref_pose.csv")
        axis.plot(
            [row[1] for row in gt],
            [row[2] for row in gt],
            color="#111827",
            linewidth=2.8,
            label="GT",
        )
        paths = trajectory_paths(scene, spec)
        for method, method_spec in METHODS.items():
            rows = read_xyz(paths[f"E_eskf3d_gate_adaptive_{method}"])
            axis.plot(
                [row[1] for row in rows],
                [row[2] for row in rows],
                color=method_spec["color"],
                linewidth=1.7,
                label=f"{method_spec['label']} / E",
            )
        axis.set_title(str(spec["label"]))
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlabel("x / m (NWU)")
        axis.set_ylabel("y / m (NWU)")
        axis.grid(True, alpha=0.25)
    axes[0].legend(fontsize=8)
    figure.savefig(OUTPUT / "rectangle_straight_mode_e_xy.png", dpi=180)
    plt.close(figure)
    print(html_path)
    print(metrics_path)


if __name__ == "__main__":
    main()
