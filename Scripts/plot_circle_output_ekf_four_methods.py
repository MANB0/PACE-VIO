#!/usr/bin/env python3
"""Compare raw and output-EKF trajectories for four circle VIO methods."""

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
from Scripts.plot_relative_pose_clone_eskf_short import (  # noqa: E402
    _xy_error_smoothness,
)


U1_POSE = ROOT / (
    "Results/circle_normal_noise_direct_uvd_u1_full_20260716/trial_1/"
    "vio_two_state_direct_uvd_u1_standard_full/"
    "clear_circle_truth_normal_noise/poses.csv"
)
SA_V1_POSE = ROOT / (
    "Results/normal_noise_sa_v1_full_three_scenes_20260717/trial_1/"
    "vio_two_state_direct_uvd_sampling_aware_v1_full/"
    "clear_circle_truth_normal_noise/poses.csv"
)
SA_V2_POSE = ROOT / (
    "Results/normal_noise_sa_v2_full_three_scenes_20260717/trial_1/"
    "vio_two_state_direct_uvd_sampling_aware_v2_full/"
    "clear_circle_truth_normal_noise/poses.csv"
)
FILTER_ROOT = ROOT / "Results/circle_output_ekf_four_methods_20260718"
OUTPUT = ROOT / "analysis_circle_output_ekf_four_methods_20260718"


METHODS = {
    "t_factor": {
        "label": "T_factor",
        "color": "#dc2626",
        "raw": POSE_FACTOR_POSE,
    },
    "u1": {
        "label": "U1",
        "color": "#2563eb",
        "raw": U1_POSE,
    },
    "sa_v1": {
        "label": "SA-v1",
        "color": "#7c3aed",
        "raw": SA_V1_POSE,
    },
    "sa_v2": {
        "label": "SA-v2",
        "color": "#059669",
        "raw": SA_V2_POSE,
    },
}


def filtered_pose(strength: str, method: str) -> Path:
    return FILTER_ROOT / strength / method / "poses.csv"


def replace_view_selector(template: str) -> str:
    old = """      <label>IMU data
        <select data-config-filter aria-label=\"Filter trajectories by IMU data configuration\">
          <option value=\"all\">All configurations</option>
          <option value=\"normal_noise\">Normal noise + bias</option>
          <option value=\"bias_no_noise\">Bias only</option>
          <option value=\"noise_no_bias\">White noise only</option>
          <option value=\"no_noise_no_bias\">No bias / no noise</option>
        </select>
      </label>"""
    new = """      <label>Trajectory view
        <select data-config-filter aria-label=\"Select smoothing strength\">
          <option value=\"smoothed_strong\" selected>Strong output EKF</option>
          <option value=\"smoothed_base\">Base output EKF</option>
          <option value=\"raw\">Raw factor-graph outputs</option>
          <option value=\"all\">All trajectories</option>
        </select>
      </label>"""
    if old not in template:
        raise RuntimeError("HTML trajectory selector block was not found")
    return template.replace(old, new)


def main() -> None:
    gt_path = BATCH_ROOT / SCENE / "ref_pose.csv"
    paths: dict[str, Path] = {"pure_macvo": MACVO_POSE}
    trace_specs: list[dict[str, str]] = []
    for method, spec in METHODS.items():
        paths[f"raw_{method}"] = Path(spec["raw"])
        paths[f"base_{method}"] = filtered_pose("base", method)
        paths[f"strong_{method}"] = filtered_pose("strong", method)
        trace_specs.extend(
            [
                {
                    "key": f"raw_{method}",
                    "label": f"{spec['label']} / raw",
                    "color": str(spec["color"]),
                    "dasharray": "8 5",
                    "config": "raw",
                },
                {
                    "key": f"base_{method}",
                    "label": f"{spec['label']} / base output EKF",
                    "color": str(spec["color"]),
                    "dasharray": "3 4",
                    "config": "smoothed_base",
                },
                {
                    "key": f"strong_{method}",
                    "label": f"{spec['label']} / strong output EKF",
                    "color": str(spec["color"]),
                    "dasharray": "",
                    "config": "smoothed_strong",
                },
            ]
        )

    for path in (gt_path, *paths.values()):
        if not path.exists():
            raise FileNotFoundError(path)

    trajectories = {"GT": read_xyz(gt_path)}
    trajectories.update({key: read_xyz(path) for key, path in paths.items()})
    lengths = {key: len(rows) for key, rows in trajectories.items()}
    if len(set(lengths.values())) != 1 or lengths["GT"] != 1890:
        raise AssertionError(f"expected aligned 1890-frame trajectories: {lengths}")
    timestamps = [row[0] for row in trajectories["GT"]]
    for key, rows in trajectories.items():
        if [row[0] for row in rows] != timestamps:
            raise AssertionError(f"timestamp mismatch: GT vs {key}")

    forwards = {"GT": read_forward_axes(gt_path)}
    forwards.update({key: read_forward_axes(path) for key, path in paths.items()})
    gt = trajectories["GT"]
    payload = {
        "scene": "Circle / Normal noise / Full 63 s / Output smoothing",
        "gt": xyz(gt),
        "gt_forward": forwards["GT"],
        "macvo": xyz(trajectories["pure_macvo"]),
        "macvo_forward": forwards["pure_macvo"],
        "time_s": [(value - timestamps[0]) * 1.0e-9 for value in timestamps],
        "error_m": position_errors(
            gt, trajectories["pure_macvo"], xy_only=False
        ),
        "metrics": metrics(gt, trajectories["pure_macvo"]),
        "fusion": [
            {
                **spec,
                "source": spec["key"],
                "scene": SCENE,
                "xyz": xyz(trajectories[spec["key"]]),
                "forward": forwards[spec["key"]],
                "error_m": position_errors(
                    gt, trajectories[spec["key"]], xy_only=False
                ),
                "metrics": metrics(gt, trajectories[spec["key"]]),
                "path": str(paths[spec["key"]]),
            }
            for spec in trace_specs
        ],
        "imu_only": [],
        "gt_path": str(gt_path),
        "macvo_path": str(MACVO_POSE),
    }

    summary_rows = []
    for spec in trace_specs:
        key = spec["key"]
        summary_rows.append(
            {
                "method": key,
                "view": spec["config"],
                **metrics(gt, trajectories[key]),
                **xy_metrics(gt, trajectories[key]),
                **relative_metrics(gt_path, paths[key]),
                **_xy_error_smoothness(gt_path, paths[key], 1890, active_from=90),
                "estimate_path": str(paths[key]),
            }
        )

    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / "trajectory_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)

    template = replace_view_selector(HTML_TEMPLATE)
    template = template.replace("{{", "{").replace("}}", "}")
    template = template.replace(
        "Circle, stop-turn rectangle, and straight trajectory comparison",
        "Four VIO methods with output-only trajectory smoothing",
    )
    template = template.replace(
        "__METHOD_SCOPE__",
        "T_factor, U1, SA-v1 and SA-v2 under identical output-EKF settings",
    )
    template = template.replace(
        "__LINE_NOTE__",
        (
            "Strong uses measurement-std x4 and process-std x0.25; base uses x1/x1. "
            "The filter is output-only and never feeds back into VIO. XY is primary."
        ),
    )
    html_path = OUTPUT / "interactive_output_ekf_four_methods_full.html"
    html_path.write_text(
        template.replace(
            "__DATA__", json.dumps({"scenes": [payload]}, ensure_ascii=False)
        ),
        encoding="utf-8",
    )

    figure, axis = plt.subplots(figsize=(11, 8), constrained_layout=True)
    axis.plot(
        [row[1] for row in gt],
        [row[2] for row in gt],
        color="#111827",
        linewidth=2.8,
        label="GT",
    )
    for method, method_spec in METHODS.items():
        rows = trajectories[f"strong_{method}"]
        axis.plot(
            [row[1] for row in rows],
            [row[2] for row in rows],
            color=str(method_spec["color"]),
            linewidth=1.8,
            label=f"{method_spec['label']} / strong output EKF",
        )
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("x / m (NWU)")
    axis.set_ylabel("y / m (NWU)")
    axis.grid(True, alpha=0.25)
    axis.legend()
    figure.savefig(OUTPUT / "output_ekf_four_methods_strong_xy.png", dpi=180)
    plt.close(figure)

    print(html_path)
    print(OUTPUT / "trajectory_metrics.csv")


if __name__ == "__main__":
    main()
