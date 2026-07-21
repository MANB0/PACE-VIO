#!/usr/bin/env python3
"""Plot full-circle SA-v1 and SA-v2 with Current U1 and existing baselines."""

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
    MACVO_POSE,
    POSE_FACTOR_POSE,
    SCENE,
    metrics,
    position_errors,
    read_forward_axes,
    read_xyz,
    relative_metrics,
    xy_metrics,
    xyz,
)


DEFAULT_RESULT_ROOT = (
    WORKDIR / "Results/circle_direct_uvd_sampling_aware_v2_full_20260717"
)
DEFAULT_OUTPUT = (
    WORKDIR / "analysis_circle_direct_uvd_sampling_aware_v2_full_20260717"
)
DEFAULT_CURRENT_ROOT = (
    WORKDIR / "Results/circle_normal_noise_direct_uvd_u1_full_20260716"
)
CURRENT_VARIANT = "vio_two_state_direct_uvd_u1_standard_full"
SA_V1_VARIANT = "vio_two_state_direct_uvd_sampling_aware_v1_full"
SA_V2_VARIANT = "vio_two_state_direct_uvd_sampling_aware_v2_full"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--current-result-root",
        type=Path,
        default=DEFAULT_CURRENT_ROOT,
    )
    parser.add_argument(
        "--sa-v1-result-root",
        type=Path,
        default=DEFAULT_RESULT_ROOT / "sampling_aware_v1",
    )
    parser.add_argument(
        "--sa-v2-result-root",
        type=Path,
        default=DEFAULT_RESULT_ROOT / "sampling_aware_v2",
    )
    parser.add_argument("--current-variant", default=CURRENT_VARIANT)
    parser.add_argument("--sa-v1-variant", default=SA_V1_VARIANT)
    parser.add_argument("--sa-v2-variant", default=SA_V2_VARIANT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--omit-sa-v1",
        action="store_true",
        help="Plot SA-v2 against the existing baselines without requiring SA-v1 output.",
    )
    return parser.parse_args()


def result_pose(root: Path, variant: str) -> Path:
    return root / "trial_1" / variant / SCENE / "poses.csv"


def main() -> None:
    args = parse_args()
    gt_path = BATCH_ROOT / SCENE / "ref_pose.csv"
    paths: dict[str, Path] = {
        "pure_macvo": MACVO_POSE,
        "pose_factor_Tij": POSE_FACTOR_POSE,
        "direct_uvd_current_u1": result_pose(
            args.current_result_root, args.current_variant
        ),
        "direct_uvd_sampling_aware_v2": result_pose(
            args.sa_v2_result_root, args.sa_v2_variant
        ),
    }
    if not args.omit_sa_v1:
        paths["direct_uvd_sampling_aware_v1"] = result_pose(
            args.sa_v1_result_root, args.sa_v1_variant
        )
    for path in (gt_path, *paths.values()):
        if not path.exists():
            raise FileNotFoundError(path)

    trajectories = {"GT": read_xyz(gt_path)}
    trajectories.update({name: read_xyz(path) for name, path in paths.items()})
    lengths = {name: len(rows) for name, rows in trajectories.items()}
    if len(set(lengths.values())) != 1:
        raise AssertionError(f"trajectory lengths differ: {lengths}")
    if next(iter(lengths.values())) != 1890:
        raise AssertionError(f"expected full 1890-frame trajectories, got {lengths}")

    reference_time = [row[0] for row in trajectories["GT"]]
    for name, rows in trajectories.items():
        if [row[0] for row in rows] != reference_time:
            raise AssertionError(f"timestamp mismatch: GT vs {name}")

    forwards = {"GT": read_forward_axes(gt_path)}
    forwards.update({name: read_forward_axes(path) for name, path in paths.items()})
    gt = trajectories["GT"]
    time_zero = gt[0][0]
    fusion_specs = [
        (
            "pose_factor_Tij",
            "Relative pose T_ij factor",
            "#dc2626",
            "8 5",
            "6D_Tij",
        ),
        (
            "direct_uvd_current_u1",
            "Direct UVD factor · U1 (existing)",
            "#2563eb",
            "",
            "direct_UVD_current",
        ),
        (
            "direct_uvd_sampling_aware_v2",
            "Direct UVD / Sampling-aware v2",
            "#059669",
            "",
            "direct_UVD_SA_v2",
        ),
    ]
    if not args.omit_sa_v1:
        fusion_specs.insert(
            -1,
            (
                "direct_uvd_sampling_aware_v1",
                "Direct UVD / Sampling-aware v1",
                "#7c3aed",
                "",
                "direct_UVD_SA_v1",
            ),
        )
    payload = {
        "scene": "Circle / Normal noise / Full 63 s",
        "gt": xyz(gt),
        "gt_forward": forwards["GT"],
        "macvo": xyz(trajectories["pure_macvo"]),
        "macvo_forward": forwards["pure_macvo"],
        "time_s": [(timestamp - time_zero) / 1e9 for timestamp in reference_time],
        "error_m": position_errors(gt, trajectories["pure_macvo"], xy_only=False),
        "metrics": metrics(gt, trajectories["pure_macvo"]),
        "fusion": [
            {
                "key": key,
                "source": source,
                "config": "normal_noise",
                "label": label,
                "color": color,
                "dasharray": dasharray,
                "scene": SCENE,
                "xyz": xyz(trajectories[key]),
                "forward": forwards[key],
                "error_m": position_errors(gt, trajectories[key], xy_only=False),
                "metrics": metrics(gt, trajectories[key]),
                "path": str(paths[key]),
            }
            for key, label, color, dasharray, source in fusion_specs
        ],
        "imu_only": [],
        "gt_path": str(gt_path),
        "macvo_path": str(MACVO_POSE),
    }

    summary_rows = []
    for method, path in paths.items():
        rows = trajectories[method]
        summary_rows.append(
            {
                "method": method,
                **metrics(gt, rows),
                **xy_metrics(gt, rows),
                **relative_metrics(gt_path, path),
                "estimate_path": str(path),
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "trajectory_metrics.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)

    html_template = HTML_TEMPLATE.replace("{{", "{").replace("}}", "}")
    html_template = html_template.replace(
        "Circle, stop-turn rectangle, and straight trajectory comparison",
        (
            "Full-circle sampling-aware v2 vs baselines"
            if args.omit_sa_v1
            else "Full-circle sampling-aware v1 vs v2"
        ),
    )
    html_template = html_template.replace(
        "__METHOD_SCOPE__",
        (
            "GT, Pure MACVO, original 6D T_ij pose factor, Current direct-UVD "
            + (
                "U1 and sampling-aware v2"
                if args.omit_sa_v1
                else "U1, sampling-aware v1 and sampling-aware v2"
            )
        ),
    )
    html_template = html_template.replace(
        "__LINE_NOTE__",
        (
            "Timestamp-matched NWU trajectories; no alignment, fitting or scale "
            "correction. XY is the primary evaluation plane."
        ),
    )
    html = html_template.replace(
        "__DATA__", json.dumps({"scenes": [payload]}, ensure_ascii=False)
    )
    html_path = args.output_dir / (
        "interactive_full_sampling_aware_v2_vs_baselines.html"
        if args.omit_sa_v1
        else "interactive_full_sampling_aware_v1_vs_v2.html"
    )
    html_path.write_text(html, encoding="utf-8")

    figure, axis = plt.subplots(figsize=(12.5, 8), constrained_layout=True)
    plot_specs = [
        ("GT", "GT", "#202833", "-", 2.8),
        ("pure_macvo", "Pure MACVO", "#f97316", "-", 1.5),
        ("pose_factor_Tij", "Relative pose T_ij factor", "#dc2626", "--", 1.7),
        (
            "direct_uvd_current_u1",
            "Direct UVD factor · U1 (existing)",
            "#2563eb",
            "-",
            2.0,
        ),
        (
            "direct_uvd_sampling_aware_v2",
            "Direct UVD / Sampling-aware v2",
            "#059669",
            "-",
            2.2,
        ),
    ]
    if not args.omit_sa_v1:
        plot_specs.insert(
            -1,
            (
                "direct_uvd_sampling_aware_v1",
                "Direct UVD / Sampling-aware v1",
                "#7c3aed",
                "-",
                2.0,
            ),
        )
    for key, label, color, linestyle, width in plot_specs:
        rows = trajectories[key]
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
        arrow_index = np.arange(90, len(values), 180)
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
    axis.set_title(
        "Full circle normal-noise: sampling-aware v2 vs baselines"
        if args.omit_sa_v1
        else "Full circle normal-noise: sampling-aware v1 vs v2"
    )
    axis.grid(True, color="#dbe2ea", linewidth=0.8)
    axis.legend(loc="best")
    static_path = args.output_dir / (
        "trajectory_xy_sampling_aware_v2_vs_baselines.png"
        if args.omit_sa_v1
        else "trajectory_xy_comparison.png"
    )
    figure.savefig(static_path, dpi=180)
    plt.close(figure)

    print(html_path)
    print(summary_path)
    print(static_path)


if __name__ == "__main__":
    main()
