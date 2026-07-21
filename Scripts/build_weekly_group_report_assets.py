#!/usr/bin/env python3
"""Build figures for the 2026-07-13 to 2026-07-18 MACVO-VIO group report."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path("/home/admin1/macvo-dev")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Scripts.plot_circle_direct_uvd_u1_vs_pose_factor import (  # noqa: E402
    BATCH_ROOT,
    read_forward_axes,
    read_xyz,
)
from Scripts.plot_normal_noise_sa_v2_full_three_scenes import (  # noqa: E402
    SCENES,
    SA_V1_ROOT,
    SA_V2_ROOT,
    scene_paths,
)


OUTPUT = ROOT / "analysis_weekly_group_report_20260718"
FIGURES = OUTPUT / "figures"
METRICS = (
    ROOT
    / "analysis_normal_noise_u1_sa_v2_full_three_scenes_20260718"
    / "trajectory_metrics.csv"
)
BIAS_SUMMARY = (
    ROOT
    / "analysis_normal_noise_sampling_aware_20260716"
    / "macvio_bias_ablation_summary.json"
)
WINDOW_METRICS = (
    ROOT
    / "analysis_normal_noise_sampling_aware_20260716"
    / "macvio_window_metrics.csv"
)

ORDER = [
    "GT",
    "pure_macvo",
    "pose_factor_Tij",
    "direct_uvd_u1",
    "sampling_aware_v1",
    "sampling_aware_v2",
]
STYLES = {
    "GT": ("GT", "#111827", "-", 3.0),
    "pure_macvo": ("Pure MACVO", "#f97316", "-", 1.8),
    "pose_factor_Tij": ("T-pose factor", "#dc2626", "--", 2.0),
    "direct_uvd_u1": ("U1 direct UVD", "#2563eb", "-", 2.2),
    "sampling_aware_v1": ("SA-v1", "#7c3aed", "--", 2.1),
    "sampling_aware_v2": ("SA-v2", "#059669", "-", 2.3),
}
SCENE_NAMES = {
    "clear_circle_truth_normal_noise": "Circle / normal noise / 63 s",
    "clear_stop_turn_rectangle_truth_normal_noise": (
        "Stop-turn rectangle / normal noise / 63 s"
    ),
    "clear_straight_truth_normal_noise": "Straight / normal noise / 21 s",
}
SCENE_SHORT = {
    "clear_circle_truth_normal_noise": "Circle",
    "clear_stop_turn_rectangle_truth_normal_noise": "Rectangle",
    "clear_straight_truth_normal_noise": "Straight",
}


def load_scene(scene: str) -> tuple[dict[str, list], dict[str, list]]:
    paths = scene_paths(scene, SA_V1_ROOT, SA_V2_ROOT)
    trajectories = {"GT": read_xyz(BATCH_ROOT / scene / "ref_pose.csv")}
    trajectories.update({name: read_xyz(path) for name, path in paths.items()})
    forwards = {"GT": read_forward_axes(BATCH_ROOT / scene / "ref_pose.csv")}
    forwards.update({name: read_forward_axes(path) for name, path in paths.items()})
    timestamps = [row[0] for row in trajectories["GT"]]
    for method, rows in trajectories.items():
        if [row[0] for row in rows] != timestamps:
            raise AssertionError(f"{scene}: timestamp mismatch for {method}")
    return trajectories, forwards


def trajectory_figure(scene: str, output: Path) -> None:
    trajectories, forwards = load_scene(scene)
    is_straight = scene == "clear_straight_truth_normal_noise"
    if is_straight:
        figure, axis = plt.subplots(figsize=(13.4, 3.8))
        figure.subplots_adjust(left=0.08, right=0.985, top=0.84, bottom=0.32)
    else:
        figure, axis = plt.subplots(figsize=(13.4, 7.4), constrained_layout=True)
    for method in ORDER:
        rows = trajectories[method]
        label, color, linestyle, width = STYLES[method]
        xy = np.asarray([[row[1], row[2]] for row in rows], dtype=np.float64)
        axis.plot(
            xy[:, 0],
            xy[:, 1],
            color=color,
            linestyle=linestyle,
            linewidth=width,
            label=label,
        )
        forward = np.asarray(forwards[method], dtype=np.float64)
        stride = max(1, len(xy) // 9)
        indices = np.arange(stride, len(xy), stride)
        axis.quiver(
            xy[indices, 0],
            xy[indices, 1],
            forward[indices, 0],
            forward[indices, 1],
            color=color,
            angles="xy",
            scale_units="xy",
            scale=5.5,
            width=0.003,
            headwidth=4.5,
            headlength=5.5,
            alpha=0.92,
        )
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("x / m (NWU)")
    axis.set_ylabel("y / m (NWU)")
    axis.set_title(SCENE_NAMES[scene], fontsize=17, weight="bold", pad=12)
    axis.grid(True, color="#dbe2ea", linewidth=0.8)
    axis.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.42 if is_straight else -0.10),
        ncol=6,
        frameon=False,
    )
    figure.savefig(output, dpi=210, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def metrics_figure(output: Path) -> None:
    frame = pd.read_csv(METRICS)
    method_labels = {
        "pure_macvo": "Pure MACVO",
        "pose_factor_Tij": "T-pose",
        "direct_uvd_u1": "U1",
        "sampling_aware_v1": "SA-v1",
        "sampling_aware_v2": "SA-v2",
    }
    scene_order = [scene for scene, _ in SCENES]
    method_order = list(method_labels)
    colors = [STYLES[method][1] for method in method_order]
    fields = [
        ("xy_rmse_m", "XY ATE RMSE", "m"),
        ("translation_rpe_rmse_m", "Translation RPE RMSE", "m"),
        ("rotation_rpe_rmse_rad", "Rotation RPE RMSE", "rad"),
    ]
    figure, axes = plt.subplots(1, 3, figsize=(16.8, 5.4), constrained_layout=True)
    x = np.arange(len(scene_order), dtype=np.float64)
    bar_width = 0.16
    for axis, (field, title, unit) in zip(axes, fields):
        for index, (method, color) in enumerate(zip(method_order, colors)):
            values = []
            for scene in scene_order:
                value = frame[(frame.scene == scene) & (frame.method == method)][field]
                if len(value) != 1:
                    raise AssertionError(f"missing metric: {scene}/{method}/{field}")
                values.append(float(value.iloc[0]))
            offset = (index - (len(method_order) - 1) / 2) * bar_width
            axis.bar(
                x + offset,
                values,
                width=bar_width,
                color=color,
                label=method_labels[method],
            )
        axis.set_xticks(x, [SCENE_SHORT[scene] for scene in scene_order])
        axis.set_ylabel(unit)
        axis.set_title(title, weight="bold")
        axis.grid(True, axis="y", alpha=0.25)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside lower center", ncol=5, frameon=False)
    figure.suptitle(
        "Normal-noise cross-scene accuracy (no alignment / no scale fitting)",
        fontsize=16,
        weight="bold",
    )
    figure.savefig(output, dpi=210, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def bias_window_figure(output: Path) -> None:
    bias = json.loads(BIAS_SUMMARY.read_text(encoding="utf-8"))
    bias_order = ["B1", "B2", "B3", "B4", "B5"]
    bias_labels = [
        "opt ba/bg",
        "fix static ba/bg",
        "GT bias oracle",
        "opt ba only",
        "fix ba / opt bg",
    ]
    high_frequency = [
        bias[key]["high_frequency_metrics"]["xy_position_error_highpass_rms_m"]
        for key in bias_order
    ]
    window = pd.read_csv(WINDOW_METRICS)
    normal = window[window["mode"] == "normal"].sort_values("window_size")

    figure, axes = plt.subplots(1, 2, figsize=(15.8, 5.5), constrained_layout=True)
    axes[0].bar(
        np.arange(len(bias_order)),
        high_frequency,
        color=["#2563eb", "#94a3b8", "#16a34a", "#dc2626", "#7c3aed"],
    )
    axes[0].set_xticks(np.arange(len(bias_order)), bias_labels, rotation=14)
    axes[0].set_ylabel("XY high-pass RMS / m")
    axes[0].set_title("Bias ablation (300-frame rectangle slice)", weight="bold")
    axes[0].grid(True, axis="y", alpha=0.25)

    axes[1].plot(
        normal["window_size"],
        normal["xy_position_error_highpass_rms_m"],
        marker="o",
        color="#2563eb",
        linewidth=2.3,
        label="XY high-pass RMS",
    )
    axes[1].set_xlabel("Fixed-lag window size N")
    axes[1].set_ylabel("XY high-pass RMS / m", color="#2563eb")
    axes[1].tick_params(axis="y", labelcolor="#2563eb")
    axes[1].set_xticks(normal["window_size"])
    runtime_axis = axes[1].twinx()
    runtime_axis.plot(
        normal["window_size"],
        normal["solver_mean_runtime_ms_per_solve"],
        marker="s",
        color="#f97316",
        linewidth=2.0,
        label="Runtime",
    )
    runtime_axis.set_ylabel("Mean runtime / ms per solve", color="#f97316")
    runtime_axis.tick_params(axis="y", labelcolor="#f97316")
    axes[1].set_title("Window length: smoother but rapidly more expensive", weight="bold")
    axes[1].grid(True, alpha=0.25)
    figure.suptitle(
        "The dominant short-slice issue is bias freedom; window length is secondary",
        fontsize=15,
        weight="bold",
    )
    figure.savefig(output, dpi=210, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def equation_figure(output: Path) -> None:
    """Render the small set of equations kept in the simplified report."""
    figure, axis = plt.subplots(figsize=(12.5, 5.1))
    axis.axis("off")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.text(0.03, 0.92, "State", fontsize=14, weight="bold", color="#2563eb")
    axis.text(
        0.30,
        0.92,
        r"$\mathbf{x}_k=\{\mathbf{R}_k,\mathbf{p}_k,\mathbf{v}_k,\mathbf{b}_{a,k},\mathbf{b}_{g,k}\}$",
        fontsize=17,
        va="center",
    )
    axis.text(0.03, 0.76, "IMU residuals", fontsize=14, weight="bold", color="#059669")
    axis.text(
        0.30,
        0.76,
        r"$\mathbf{r}_v=\mathbf{R}_i^{\mathsf{T}}(\mathbf{v}_j-\mathbf{v}_i-\mathbf{g}\Delta t)-\Delta\mathbf{v}_{ij}$",
        fontsize=15,
        va="center",
    )
    axis.text(
        0.30,
        0.60,
        r"$\mathbf{r}_p=\mathbf{R}_i^{\mathsf{T}}(\mathbf{p}_j-\mathbf{p}_i-\mathbf{v}_i\Delta t-\frac{1}{2}\mathbf{g}\Delta t^2)-\Delta\mathbf{p}_{ij}$",
        fontsize=15,
        va="center",
    )
    axis.text(
        0.30,
        0.44,
        r"$\mathbf{r}_R=\mathrm{Log}(\Delta\mathbf{R}_{ij}^{-1}\mathbf{R}_i^{\mathsf{T}}\mathbf{R}_j)$",
        fontsize=15,
        va="center",
    )
    axis.text(0.03, 0.25, "Visual residuals", fontsize=14, weight="bold", color="#f97316")
    axis.text(
        0.30,
        0.25,
        r"$\mathbf{r}_{pose}=\mathrm{Log}(\widehat{\mathbf{T}}_{ij}^{-1}\mathbf{T}_{ij}(\mathbf{x}_i,\mathbf{x}_j))$",
        fontsize=15,
        va="center",
    )
    axis.text(
        0.30,
        0.09,
        r"$\mathbf{r}_{uvd,k}=\mathbf{z}_{j,k}-\pi(\mathbf{T}_{ji}\,\pi^{-1}(\mathbf{z}_{i,k}))$",
        fontsize=15,
        va="center",
    )
    axis.add_patch(
        plt.Rectangle((0.01, 0.01), 0.98, 0.98, fill=False, edgecolor="#cbd5e1", linewidth=1.2)
    )
    figure.savefig(output, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    for scene, _ in SCENES:
        trajectory_figure(scene, FIGURES / f"trajectory_{SCENE_SHORT[scene].lower()}.png")
    metrics_figure(FIGURES / "cross_scene_metrics.png")
    bias_window_figure(FIGURES / "bias_window_ablation.png")
    equation_figure(FIGURES / "core_equations.png")
    print(FIGURES)


if __name__ == "__main__":
    main()
