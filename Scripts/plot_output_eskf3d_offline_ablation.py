#!/usr/bin/env python3
"""Plot A-E offline output-smoothing ablations for four circle VIO methods."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation


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
EKF2D_ROOT = ROOT / "Results/circle_output_ekf_four_methods_20260718/strong"
ESKF3D_ROOT = ROOT / "Results/circle_output_eskf3d_ablation_20260718"
OUTPUT = ROOT / "analysis_circle_output_eskf3d_ablation_20260718"


METHODS = {
    "t_factor": {"label": "T_factor", "color": "#dc2626", "raw": POSE_FACTOR_POSE},
    "u1": {"label": "U1", "color": "#2563eb", "raw": U1_POSE},
    "sa_v1": {"label": "SA-v1", "color": "#7c3aed", "raw": SA_V1_POSE},
    "sa_v2": {"label": "SA-v2", "color": "#059669", "raw": SA_V2_POSE},
}
ABLATIONS = {
    "A_raw": {
        "label": "A / Raw VIO",
        "dasharray": "2 3",
        "path": lambda method: METHODS[method]["raw"],
    },
    "B_ekf2d": {
        "label": "B / Existing strong XY+yaw EKF",
        "dasharray": "10 4",
        "path": lambda method: EKF2D_ROOT / method / "poses.csv",
    },
    "C_eskf3d_no_gate": {
        "label": "C / 3D ESKF, no gate",
        "dasharray": "8 3 2 3",
        "path": lambda method: ESKF3D_ROOT / "no_gate" / method / "poses.csv",
    },
    "D_eskf3d_gate": {
        "label": "D / 3D ESKF, block gate",
        "dasharray": "5 3",
        "path": lambda method: ESKF3D_ROOT / "gate" / method / "poses.csv",
    },
    "E_eskf3d_gate_adaptive": {
        "label": "E / 3D ESKF, block gate + adaptive Q",
        "dasharray": "",
        "path": lambda method: ESKF3D_ROOT / "gate_adaptive" / method / "poses.csv",
    },
}


def replace_filter_controls(template: str) -> str:
    old = """      <label>IMU data
        <select data-config-filter aria-label=\"Filter trajectories by IMU data configuration\">
          <option value=\"all\">All configurations</option>
          <option value=\"normal_noise\">Normal noise + bias</option>
          <option value=\"bias_no_noise\">Bias only</option>
          <option value=\"noise_no_bias\">White noise only</option>
          <option value=\"no_noise_no_bias\">No bias / no noise</option>
        </select>
      </label>"""
    ekf_options = "\n".join(
        f'          <option value="{key}"'
        + (" selected" if key == "E_eskf3d_gate_adaptive" else "")
        + f">{spec['label']}</option>"
        for key, spec in ABLATIONS.items()
    )
    optimizer_options = "\n".join(
        f'          <option value="{key}">{spec["label"]}</option>'
        for key, spec in METHODS.items()
    )
    new = (
        "      <label>Optimizer method\n"
        '        <select data-method-filter aria-label="Select optimizer method">\n'
        '          <option value="all" selected>ALL</option>\n'
        f"{optimizer_options}\n"
        "        </select>\n"
        "      </label>\n"
        "      <label>EKF method\n"
        '        <select data-config-filter aria-label="Select EKF method">\n'
        '          <option value="all">ALL</option>\n'
        f"{ekf_options}\n"
        "        </select>\n"
        "      </label>"
    )
    if old not in template:
        raise RuntimeError("HTML configuration selector was not found")
    template = template.replace(old, new)
    fusion_checkboxes = (
        '      ${{sceneData.fusion.map(item => `<label data-trace-config="${{item.config}}">'
        '<input type="checkbox" data-show-fusion="${{item.key}}" checked> Fusion · '
        '${{item.label}}</label>`).join("")}}\n'
    )
    if fusion_checkboxes not in template:
        raise RuntimeError("fusion checkbox controls were not found")
    return template.replace(fusion_checkboxes, "")


def add_two_dimensional_filter_logic(template: str) -> str:
    replacements = {
        '      <label><input type="checkbox" data-show-macvo checked> MACVO</label>': (
            '      <label><input type="checkbox" data-show-macvo checked> '
            "MACVO_Pure</label>"
        ),
        '      <span><span class="swatch" style="border-color:#e8590c"></span>Pure MACVO</span>': (
            '      <span><span class="swatch" style="border-color:#e8590c"></span>'
            "MACVO_Pure</span>"
        ),
        '  const configFilter = card.querySelector("[data-config-filter]");': (
            '  const methodFilter = card.querySelector("[data-method-filter]");\n'
            '  const configFilter = card.querySelector("[data-config-filter]");'
        ),
        '    configFilter: "all",': (
            '    methodFilter: "all", configFilter: "E_eskf3d_gate_adaptive",'
        ),
        "    return state.configFilter === \"all\" || item.config === state.configFilter;": (
            '    const methodMatches = state.methodFilter === "all" || '
            "item.optimizer === state.methodFilter;\n"
            '    const ekfMatches = state.configFilter === "all" || '
            "item.config === state.configFilter;\n"
            "    return methodMatches && ekfMatches;"
        ),
        '''    card.querySelectorAll("[data-trace-config]").forEach(node => {
      node.hidden = state.configFilter !== "all" && node.dataset.traceConfig !== state.configFilter;
    });
    card.querySelectorAll("[data-legend-config]").forEach(node => {
      node.hidden = state.configFilter !== "all" && node.dataset.legendConfig !== state.configFilter;
    });''': '''    card.querySelectorAll("[data-legend-config]").forEach(node => {
      const methodMatches = state.methodFilter === "all" ||
        node.dataset.legendMethod === state.methodFilter;
      const ekfMatches = state.configFilter === "all" ||
        node.dataset.legendConfig === state.configFilter;
      node.hidden = !(methodMatches && ekfMatches);
    });''',
        '''      const checkbox = card.querySelector(`[data-show-fusion="${item.key}"]`);
      if (!checkbox || !checkbox.checked) continue;
''': "",
        '''  configFilter.onchange = () => {
    state.configFilter = configFilter.value;
    state.xlim = state.ylim = null;
    updateFilterVisibility();
    render();
  };''': '''  methodFilter.onchange = () => {
    state.methodFilter = methodFilter.value;
    state.xlim = state.ylim = null;
    updateFilterVisibility();
    render();
  };
  configFilter.onchange = () => {
    state.configFilter = configFilter.value;
    state.xlim = state.ylim = null;
    updateFilterVisibility();
    render();
  };''',
        '  card.querySelectorAll("[data-show-fusion]").forEach(input => input.onchange = render);\n': "",
    }
    for old, new in replacements.items():
        if old not in template:
            raise RuntimeError(f"two-filter template fragment was not found: {old[:60]}")
        template = template.replace(old, new)
    old_legend = (
        'data-legend-config="${item.config}"><span class="swatch"'
    )
    new_legend = (
        'data-legend-config="${item.config}" data-legend-method="${item.optimizer}">'
        '<span class="swatch"'
    )
    if old_legend not in template:
        raise RuntimeError("fusion legend metadata was not found")
    return template.replace(old_legend, new_legend)


def load_pose(path: Path) -> tuple[pd.DataFrame, np.ndarray, Rotation]:
    frame = pd.read_csv(path)
    position = frame[["tx", "ty", "tz"]].to_numpy(np.float64)
    rotation = Rotation.from_quat(
        frame[["qx", "qy", "qz", "qw"]].to_numpy(np.float64)
    )
    return frame, position, rotation


def detailed_metrics(
    gt_path: Path, estimate_path: Path, *, active_from: int = 90
) -> dict[str, float | int]:
    gt_frame = pd.read_csv(gt_path)
    estimate_frame, position, rotation = load_pose(estimate_path)
    gt_position = gt_frame[["x", "y", "z"]].to_numpy(np.float64)
    gt_rotation = Rotation.from_quat(
        gt_frame[["qx", "qy", "qz", "qw"]].to_numpy(np.float64)
    )
    gt_timestamp = gt_frame["timestamp"].to_numpy(np.int64)
    timestamp = estimate_frame["timestamp_ns"].to_numpy(np.int64)
    if not np.array_equal(gt_timestamp, timestamp):
        raise AssertionError(f"timestamp mismatch: {estimate_path}")
    error = position - gt_position
    orientation_error = (gt_rotation.inv() * rotation).magnitude()
    active_position = position[active_from:]
    active_rotation = rotation[active_from:]
    first_difference = np.diff(active_position, axis=0)
    second_difference = np.diff(active_position, n=2, axis=0)
    rotation_increment = (
        active_rotation[:-1].inv() * active_rotation[1:]
    ).as_rotvec()
    rotation_second_difference = np.diff(rotation_increment, axis=0)
    result: dict[str, float | int] = {
        "position_rmse_m": float(np.sqrt(np.mean(np.sum(error * error, axis=1)))),
        "orientation_rmse_rad": float(np.sqrt(np.mean(orientation_error**2))),
        "position_first_difference_rmse_m": float(
            np.sqrt(np.mean(np.sum(first_difference**2, axis=1)))
        ),
        "position_second_difference_rmse_m": float(
            np.sqrt(np.mean(np.sum(second_difference**2, axis=1)))
        ),
        "rotation_increment_rmse_rad": float(
            np.sqrt(np.mean(np.sum(rotation_increment**2, axis=1)))
        ),
        "rotation_second_difference_rmse_rad": float(
            np.sqrt(np.mean(np.sum(rotation_second_difference**2, axis=1)))
        ),
    }
    for axis, index in zip("xyz", range(3)):
        result[f"position_{axis}_rmse_m"] = float(
            np.sqrt(np.mean(error[:, index] ** 2))
        )
        result[f"position_{axis}_second_difference_rmse_m"] = float(
            np.sqrt(np.mean(second_difference[:, index] ** 2))
        )
    return result


def manifest_metrics(path: Path) -> dict[str, float | int]:
    manifest_path = path.parent / "manifest.json"
    if not manifest_path.exists():
        return {
            "position_inflate_count": 0,
            "position_reject_count": 0,
            "rotation_inflate_count": 0,
            "rotation_reject_count": 0,
            "mean_filter_runtime_us": 0.0,
        }
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    actions = manifest.get("action_counts", {})
    return {
        "position_inflate_count": int(actions.get("position", {}).get("inflate", 0)),
        "position_reject_count": int(actions.get("position", {}).get("reject", 0)),
        "rotation_inflate_count": int(actions.get("rotation", {}).get("inflate", 0)),
        "rotation_reject_count": int(actions.get("rotation", {}).get("reject", 0)),
        "mean_filter_runtime_us": float(
            manifest.get("runtime", {}).get("mean_us_per_frame", 0.0)
        ),
    }


def main() -> None:
    gt_path = BATCH_ROOT / SCENE / "ref_pose.csv"
    paths = {
        f"{ablation}_{method}": Path(spec["path"](method))
        for ablation, spec in ABLATIONS.items()
        for method in METHODS
    }
    for path in (gt_path, MACVO_POSE, *paths.values()):
        if not path.exists():
            raise FileNotFoundError(path)

    trajectories = {"GT": read_xyz(gt_path), "pure_macvo": read_xyz(MACVO_POSE)}
    trajectories.update({key: read_xyz(path) for key, path in paths.items()})
    lengths = {key: len(rows) for key, rows in trajectories.items()}
    if len(set(lengths.values())) != 1 or lengths["GT"] != 1890:
        raise AssertionError(f"expected aligned 1890-frame trajectories: {lengths}")
    timestamps = [row[0] for row in trajectories["GT"]]
    for key, rows in trajectories.items():
        if [row[0] for row in rows] != timestamps:
            raise AssertionError(f"timestamp mismatch: GT vs {key}")

    forwards = {"GT": read_forward_axes(gt_path), "pure_macvo": read_forward_axes(MACVO_POSE)}
    forwards.update({key: read_forward_axes(path) for key, path in paths.items()})
    trace_specs = []
    summary_rows = []
    for ablation, ablation_spec in ABLATIONS.items():
        for method, method_spec in METHODS.items():
            key = f"{ablation}_{method}"
            trace_specs.append(
                {
                    "key": key,
                    "source": key,
                    "config": ablation,
                    "optimizer": method,
                    "label": f"{method_spec['label']} / {ablation_spec['label']}",
                    "color": method_spec["color"],
                    "dasharray": ablation_spec["dasharray"],
                }
            )
            summary_rows.append(
                {
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
        "scene": "Circle / Normal noise / Full 63 s / 3D output smoothing A-E",
        "gt": xyz(gt),
        "gt_forward": forwards["GT"],
        "macvo": xyz(trajectories["pure_macvo"]),
        "macvo_forward": forwards["pure_macvo"],
        "time_s": [(value - timestamps[0]) * 1.0e-9 for value in timestamps],
        "error_m": position_errors(gt, trajectories["pure_macvo"], xy_only=False),
        "metrics": metrics(gt, trajectories["pure_macvo"]),
        "fusion": [
            {
                **spec,
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

    OUTPUT.mkdir(parents=True, exist_ok=True)
    metrics_path = OUTPUT / "offline_ablation_metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    (OUTPUT / "offline_ablation_metrics.json").write_text(
        json.dumps(summary_rows, indent=2), encoding="utf-8"
    )

    template = replace_filter_controls(HTML_TEMPLATE)
    template = template.replace("{{", "{").replace("}}", "}")
    template = add_two_dimensional_filter_logic(template)
    template = template.replace(
        "Circle, stop-turn rectangle, and straight trajectory comparison",
        "3D output-only ESKF offline A-E validation",
    )
    template = template.replace(
        "__METHOD_SCOPE__",
        "T_factor, U1, SA-v1 and SA-v2; raw, 2D EKF and three 3D ESKF modes",
    )
    template = template.replace(
        "__LINE_NOTE__",
        (
            "All filters consume completed pose CSVs only and never feed back to VIO. "
            "The page supports XY, XZ, YZ and full 3D position-error views."
        ),
    )
    html_path = OUTPUT / "interactive_output_eskf3d_offline_ablation.html"
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
    selected = "E_eskf3d_gate_adaptive"
    for method, method_spec in METHODS.items():
        rows = trajectories[f"{selected}_{method}"]
        axis.plot(
            [row[1] for row in rows],
            [row[2] for row in rows],
            color=method_spec["color"],
            linewidth=1.8,
            label=f"{method_spec['label']} / E",
        )
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("x / m (NWU)")
    axis.set_ylabel("y / m (NWU)")
    axis.grid(True, alpha=0.25)
    axis.legend()
    figure.savefig(OUTPUT / "output_eskf3d_mode_e_xy.png", dpi=180)
    plt.close(figure)
    print(html_path)
    print(metrics_path)


if __name__ == "__main__":
    main()
