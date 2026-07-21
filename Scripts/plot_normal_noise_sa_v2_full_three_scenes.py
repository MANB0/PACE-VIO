#!/usr/bin/env python3
"""Compare full T-pose, U1, SA-v1, and SA-v2 normal-noise trajectories."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


WORKDIR = Path("/home/admin1/macvo-dev")
if str(WORKDIR) not in sys.path:
    sys.path.insert(0, str(WORKDIR))

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


MACVO_ROOT = (
    WORKDIR
    / "Results/visual_factor_cache_static63_unique_source_20260713"
    / "trial_1/pure_macvo"
)
SA_V2_ROOT = WORKDIR / "Results/normal_noise_sa_v2_full_three_scenes_20260717"
SA_V1_ROOT = WORKDIR / "Results/normal_noise_sa_v1_full_three_scenes_20260717"
OUTPUT_DIR = WORKDIR / "analysis_normal_noise_u1_sa_v2_full_three_scenes_20260718"
SA_V1_VARIANT = "vio_two_state_direct_uvd_sampling_aware_v1_full"
SA_V2_VARIANT = "vio_two_state_direct_uvd_sampling_aware_v2_full"

POSE_FACTOR_ROOTS = {
    "clear_circle_truth_normal_noise": (
        WORKDIR / "Results/circle_straight_normal_noise_two_state_standard_full_20260715"
    ),
    "clear_stop_turn_rectangle_truth_normal_noise": (
        WORKDIR / "Results/rectangle_normal_noise_two_state_standard_full_20260715"
    ),
    "clear_straight_truth_normal_noise": (
        WORKDIR / "Results/circle_straight_normal_noise_two_state_standard_full_20260715"
    ),
}
POSE_FACTOR_VARIANT = "vio_two_state_fixed_lag_standard_full"

U1_ROOTS = {
    "clear_circle_truth_normal_noise": (
        WORKDIR / "Results/circle_normal_noise_direct_uvd_u1_full_20260716"
    ),
    "clear_stop_turn_rectangle_truth_normal_noise": (
        WORKDIR / "Results/normal_noise_direct_uvd_u1_missing_full_scenes_20260717"
    ),
    "clear_straight_truth_normal_noise": (
        WORKDIR / "Results/normal_noise_direct_uvd_u1_missing_full_scenes_20260717"
    ),
}
U1_VARIANT = "vio_two_state_direct_uvd_u1_standard_full"

SCENES = (
    ("clear_circle_truth_normal_noise", "Circle / Normal noise / Full 63 s"),
    (
        "clear_stop_turn_rectangle_truth_normal_noise",
        "Stop-turn rectangle / Normal noise / Full 63 s",
    ),
    ("clear_straight_truth_normal_noise", "Straight / Normal noise / Full 21 s"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sa-v1-root", type=Path, default=SA_V1_ROOT)
    parser.add_argument("--sa-v2-root", type=Path, default=SA_V2_ROOT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def result_pose(root: Path, variant: str, scene: str) -> Path:
    return root / "trial_1" / variant / scene / "poses.csv"


def assert_aligned(scene: str, trajectories: dict[str, list]) -> int:
    lengths = {name: len(rows) for name, rows in trajectories.items()}
    if not lengths or min(lengths.values()) == 0:
        raise AssertionError(f"{scene}: empty trajectory: {lengths}")
    if len(set(lengths.values())) != 1:
        raise AssertionError(f"{scene}: trajectory lengths differ: {lengths}")
    reference_name = next(iter(trajectories))
    reference = [row[0] for row in trajectories[reference_name]]
    for name, rows in trajectories.items():
        timestamps = [row[0] for row in rows]
        if timestamps != reference:
            mismatch = next(
                index
                for index, values in enumerate(zip(reference, timestamps))
                if values[0] != values[1]
            )
            raise AssertionError(
                f"{scene}: timestamp mismatch at {mismatch}: "
                f"{reference_name}={reference[mismatch]}, {name}={timestamps[mismatch]}"
            )
    return lengths[reference_name]


def scene_paths(
    scene: str,
    sa_v1_root: Path,
    sa_v2_root: Path,
) -> dict[str, Path]:
    paths = {
        "pure_macvo": MACVO_ROOT / scene / "poses.csv",
        "pose_factor_Tij": result_pose(
            POSE_FACTOR_ROOTS[scene], POSE_FACTOR_VARIANT, scene
        ),
        "sampling_aware_v1": result_pose(sa_v1_root, SA_V1_VARIANT, scene),
        "sampling_aware_v2": result_pose(sa_v2_root, SA_V2_VARIANT, scene),
    }
    if scene in U1_ROOTS:
        paths["direct_uvd_u1"] = result_pose(U1_ROOTS[scene], U1_VARIANT, scene)
    return paths


def build_scene(
    scene: str,
    title: str,
    sa_v1_root: Path,
    sa_v2_root: Path,
) -> tuple[dict, list[dict[str, object]], dict[str, list], dict[str, list]]:
    gt_path = BATCH_ROOT / scene / "ref_pose.csv"
    paths = scene_paths(scene, sa_v1_root, sa_v2_root)
    for path in (gt_path, *paths.values()):
        if not path.exists():
            raise FileNotFoundError(path)

    trajectories = {"GT": read_xyz(gt_path)}
    trajectories.update({name: read_xyz(path) for name, path in paths.items()})
    count = assert_aligned(scene, trajectories)
    forwards = {"GT": read_forward_axes(gt_path)}
    forwards.update({name: read_forward_axes(path) for name, path in paths.items()})
    if any(len(values) != count for values in forwards.values()):
        raise AssertionError(f"{scene}: pose and orientation row counts differ")

    specs = [
        (
            "pose_factor_Tij",
            "Relative pose T_ij factor",
            "#dc2626",
            "8 5",
            "6D_Tij",
        ),
    ]
    if "direct_uvd_u1" in paths:
        specs.append(
            (
                "direct_uvd_u1",
                "Direct UVD factor / U1",
                "#2563eb",
                "",
                "direct_UVD_U1",
            )
        )
    specs.append(
        (
            "sampling_aware_v1",
            "Direct UVD / Sampling-aware v1",
            "#7c3aed",
            "6 3",
            "direct_UVD_SA_v1",
        )
    )
    specs.append(
        (
            "sampling_aware_v2",
            "Direct UVD / Sampling-aware v2",
            "#059669",
            "",
            "direct_UVD_SA_v2",
        )
    )

    gt = trajectories["GT"]
    time_zero = gt[0][0]
    payload = {
        "scene": title,
        "gt": xyz(gt),
        "gt_forward": forwards["GT"],
        "macvo": xyz(trajectories["pure_macvo"]),
        "macvo_forward": forwards["pure_macvo"],
        "time_s": [(row[0] - time_zero) / 1e9 for row in gt],
        "error_m": position_errors(gt, trajectories["pure_macvo"], xy_only=False),
        "metrics": metrics(gt, trajectories["pure_macvo"]),
        "fusion": [
            {
                "key": f"{scene}_{key}",
                "source": source,
                "config": "normal_noise",
                "label": label,
                "color": color,
                "dasharray": dasharray,
                "scene": scene,
                "xyz": xyz(trajectories[key]),
                "forward": forwards[key],
                "error_m": position_errors(gt, trajectories[key], xy_only=False),
                "metrics": metrics(gt, trajectories[key]),
                "path": str(paths[key]),
            }
            for key, label, color, dasharray, source in specs
        ],
        "imu_only": [],
        "gt_path": str(gt_path),
        "macvo_path": str(paths["pure_macvo"]),
    }

    summary = []
    for method, path in paths.items():
        rows = trajectories[method]
        summary.append(
            {
                "scene": scene,
                "method": method,
                **metrics(gt, rows),
                **xy_metrics(gt, rows),
                **relative_metrics(gt_path, path),
                "estimate_path": str(path),
            }
        )
    return payload, summary, trajectories, forwards


def write_static_plot(
    output_path: Path,
    plotted: list[tuple[str, dict[str, list], dict[str, list]]],
) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(18, 6.2), constrained_layout=True)
    method_styles = {
        "GT": ("GT", "#202833", "-", 2.8),
        "pure_macvo": ("Pure MACVO", "#f97316", "-", 1.5),
        "pose_factor_Tij": ("Relative pose T_ij factor", "#dc2626", "--", 1.7),
        "direct_uvd_u1": ("Direct UVD factor / U1", "#2563eb", "-", 2.0),
        "sampling_aware_v1": ("Direct UVD / SA-v1", "#7c3aed", "--", 2.0),
        "sampling_aware_v2": ("Direct UVD / SA-v2", "#059669", "-", 2.2),
    }
    for axis, (title, trajectories, forwards) in zip(axes, plotted):
        for key, rows in trajectories.items():
            label, color, linestyle, width = method_styles[key]
            values = np.asarray([[row[1], row[2]] for row in rows], dtype=np.float64)
            axis.plot(
                values[:, 0],
                values[:, 1],
                label=label,
                color=color,
                linestyle=linestyle,
                linewidth=width,
            )
            forward = np.asarray(forwards[key], dtype=np.float64)
            arrow_stride = max(1, len(values) // 10)
            arrow_index = np.arange(arrow_stride, len(values), arrow_stride)
            axis.quiver(
                values[arrow_index, 0],
                values[arrow_index, 1],
                forward[arrow_index, 0],
                forward[arrow_index, 1],
                color=color,
                angles="xy",
                scale_units="xy",
                scale=5.0,
                width=0.003,
                headwidth=4.5,
                headlength=5.5,
            )
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlabel("x / m (NWU)")
        axis.set_ylabel("y / m (NWU)")
        axis.set_title(title)
        axis.grid(True, color="#dbe2ea", linewidth=0.8)
    handles, labels = axes[0].get_legend_handles_labels()
    for axis in axes[1:]:
        axis_handles, axis_labels = axis.get_legend_handles_labels()
        for handle, label in zip(axis_handles, axis_labels):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    figure.legend(handles, labels, loc="outside lower center", ncol=len(labels))
    figure.suptitle("Normal-noise full trajectories: T-pose, U1, SA-v1 and SA-v2")
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    payloads = []
    summary_rows: list[dict[str, object]] = []
    plotted = []
    for scene, title in SCENES:
        payload, summary, trajectories, forwards = build_scene(
            scene, title, args.sa_v1_root, args.sa_v2_root
        )
        payloads.append(payload)
        summary_rows.extend(summary)
        plotted.append((title, trajectories, forwards))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "trajectory_metrics.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)

    html_template = HTML_TEMPLATE.replace("{{", "{").replace("}}", "}")
    html_template = html_template.replace(
        "Circle, stop-turn rectangle, and straight trajectory comparison",
        "Normal-noise full trajectories: T-pose, U1, SA-v1 and SA-v2",
    )
    html_template = html_template.replace(
        "__METHOD_SCOPE__",
        (
            "GT, Pure MACVO, original 6D T_ij pose factor, Direct-UVD U1, "
            "Direct-UVD sampling-aware v1, and sampling-aware v2"
        ),
    )
    html_template = html_template.replace(
        "__LINE_NOTE__",
        (
            "Timestamp-matched NWU trajectories; no alignment, fitting or scale "
            "correction. XY is the primary evaluation plane. All five methods are "
            "available for all three scenes."
        ),
    )
    html = html_template.replace(
        "__DATA__", json.dumps({"scenes": payloads}, ensure_ascii=False)
    )
    html_path = args.output_dir / "interactive_sa_v2_vs_baselines_three_scenes.html"
    html_path.write_text(html, encoding="utf-8")
    all_methods_html_path = args.output_dir / "interactive_all_methods_three_scenes.html"
    all_methods_html_path.write_text(html, encoding="utf-8")

    static_path = args.output_dir / "trajectory_xy_all_methods_three_scenes.png"
    write_static_plot(static_path, plotted)

    print(html_path)
    print(all_methods_html_path)
    print(summary_path)
    print(static_path)


if __name__ == "__main__":
    main()
