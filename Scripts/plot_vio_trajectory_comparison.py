#!/usr/bin/env python3
"""Plot GT / pure MACVO / AIM-VO trajectory comparisons for VIO diagnostics."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

WORKDIR = Path(__file__).resolve().parents[1]
if str(WORKDIR) not in sys.path:
    sys.path.insert(0, str(WORKDIR))

from Scripts.analyse_vio_imu_prior_mode_grid import KEPT12_SCENES, SCENE_ROOTS


VARIANT_LABELS = {
    "pure_macvo": "Pure MACVO",
    "aimvo_damping_s005": "AIM-VO",
}


def read_pose_xyz(path: Path) -> np.ndarray:
    data = np.genfromtxt(path, delimiter=",", dtype=float, skip_header=1)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] < 4:
        raise ValueError(f"Pose file has fewer than 4 columns: {path}")
    return data[:, 1:4]


def direct_ate(est_xyz: np.ndarray, gt_xyz: np.ndarray) -> float:
    n = min(len(est_xyz), len(gt_xyz))
    if n <= 0:
        return float("nan")
    delta = est_xyz[:n] - gt_xyz[:n]
    return float(np.sqrt(np.mean(np.sum(delta * delta, axis=1))))


def complete_trials(result_root: Path, scene: str, trials: int) -> list[int]:
    complete = []
    for trial in range(1, trials + 1):
        pure = result_root / f"trial_{trial}" / "pure_macvo" / scene / "poses.csv"
        aimvo = result_root / f"trial_{trial}" / "aimvo_damping_s005" / scene / "poses.csv"
        gt = SCENE_ROOTS[scene] / "ref_pose.csv"
        if pure.exists() and aimvo.exists() and gt.exists():
            complete.append(trial)
    return complete


def choose_representative_trial(result_root: Path, scene: str, trials: int) -> tuple[int | None, dict[int, dict[str, float]]]:
    gt_path = SCENE_ROOTS[scene] / "ref_pose.csv"
    if not gt_path.exists():
        return None, {}
    gt = read_pose_xyz(gt_path)

    per_trial: dict[int, dict[str, float]] = {}
    for trial in complete_trials(result_root, scene, trials):
        row: dict[str, float] = {}
        for variant in ("pure_macvo", "aimvo_damping_s005"):
            pose_path = result_root / f"trial_{trial}" / variant / scene / "poses.csv"
            row[variant] = direct_ate(read_pose_xyz(pose_path), gt)
        per_trial[trial] = row

    if not per_trial:
        return None, {}

    aimvo_values = np.array([row["aimvo_damping_s005"] for row in per_trial.values()], dtype=float)
    median_aimvo = float(np.median(aimvo_values))
    representative = min(
        per_trial,
        key=lambda trial: abs(per_trial[trial]["aimvo_damping_s005"] - median_aimvo),
    )
    return representative, per_trial


def set_equal_2d_limits(ax, curves: list[np.ndarray], i: int, j: int, pad_ratio: float = 0.08) -> None:
    pts = np.concatenate([curve[:, [i, j]] for curve in curves if len(curve)], axis=0)
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    center = (mins + maxs) / 2.0
    span = float(max(maxs[0] - mins[0], maxs[1] - mins[1]))
    if span <= 1e-9:
        span = 1.0
    half = span * (0.5 + pad_ratio)
    ax.set_xlim(center[0] - half, center[0] + half)
    ax.set_ylim(center[1] - half, center[1] + half)
    ax.set_aspect("equal", adjustable="box")


def plot_scene(result_root: Path, outdir: Path, scene: str, trial: int) -> dict:
    gt_path = SCENE_ROOTS[scene] / "ref_pose.csv"
    pure_path = result_root / f"trial_{trial}" / "pure_macvo" / scene / "poses.csv"
    aimvo_path = result_root / f"trial_{trial}" / "aimvo_damping_s005" / scene / "poses.csv"

    gt = read_pose_xyz(gt_path)
    pure = read_pose_xyz(pure_path)
    aimvo = read_pose_xyz(aimvo_path)
    n = min(len(gt), len(pure), len(aimvo))
    gt = gt[:n]
    pure = pure[:n]
    aimvo = aimvo[:n]

    pure_ate = direct_ate(pure, gt)
    aimvo_ate = direct_ate(aimvo, gt)

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6), dpi=180)
    colors = {
        "gt": "#222222",
        "pure": "#d95f02",
        "aimvo": "#1b9e77",
    }

    panels = [
        (axes[0], 0, 1, "Top view", "x [m]", "y [m]"),
        (axes[1], 0, 2, "Vertical profile", "x [m]", "z [m]"),
    ]
    for ax, i, j, title, xlabel, ylabel in panels:
        ax.plot(gt[:, i], gt[:, j], color=colors["gt"], lw=2.4, label="GT")
        ax.plot(pure[:, i], pure[:, j], color=colors["pure"], lw=1.8, label=f"Pure MACVO ({pure_ate:.3f})")
        ax.plot(aimvo[:, i], aimvo[:, j], color=colors["aimvo"], lw=1.8, label=f"AIM-VO ({aimvo_ate:.3f})")
        ax.scatter(gt[0, i], gt[0, j], color="#000000", marker="o", s=22, zorder=5)
        ax.scatter(gt[-1, i], gt[-1, j], color="#000000", marker="x", s=34, zorder=5)
        ax.grid(True, alpha=0.25, linewidth=0.7)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        set_equal_2d_limits(ax, [gt, pure, aimvo], i, j)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.suptitle(f"{scene} - representative trial {trial} (direct ATE, no alignment)", y=0.98)
    fig.tight_layout(rect=(0, 0.08, 1, 0.94))

    out_path = outdir / f"{scene}_trial{trial}_gt_pure_aimvo.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return {
        "scene": scene,
        "trial": trial,
        "frames": n,
        "pure_ate": pure_ate,
        "aimvo_ate": aimvo_ate,
        "delta_aimvo_minus_pure": aimvo_ate - pure_ate,
        "plot": str(out_path),
    }


def write_contact_sheet(outdir: Path, rows: list[dict]) -> Path | None:
    if not rows:
        return None
    images = []
    for row in rows:
        img = plt.imread(row["plot"])
        images.append((row["scene"], img))

    cols = 2
    rows_n = int(math.ceil(len(images) / cols))
    fig, axes = plt.subplots(rows_n, cols, figsize=(12, 4.2 * rows_n), dpi=150)
    axes = np.array(axes).reshape(rows_n, cols)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, (scene, img) in zip(axes.ravel(), images):
        ax.imshow(img)
        ax.set_title(scene, fontsize=10)
        ax.axis("off")
    fig.tight_layout()
    out_path = outdir / "trajectory_comparison_contact_sheet.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, default=Path("Results/vio_imu_kept12_main3_3trial_300f"))
    parser.add_argument("--outdir", type=Path, default=Path("analysis_vio_imu_kept12_main3_trajectory_compare"))
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--scenes", nargs="*", default=KEPT12_SCENES)
    parser.add_argument("--include-partial", action="store_true")
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    rows = []
    skipped = []
    for scene in args.scenes:
        trial, per_trial = choose_representative_trial(args.result_root, scene, args.trials)
        if trial is None:
            skipped.append((scene, "missing GT or complete pure/AIM-VO trial"))
            continue
        complete_count = len(per_trial)
        if complete_count < args.trials and not args.include_partial:
            skipped.append((scene, f"only {complete_count}/{args.trials} complete trials"))
            continue
        row = plot_scene(args.result_root, args.outdir, scene, trial)
        row["complete_trials"] = complete_count
        rows.append(row)

    summary_path = args.outdir / "trajectory_comparison_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "scene",
            "trial",
            "complete_trials",
            "frames",
            "pure_ate",
            "aimvo_ate",
            "delta_aimvo_minus_pure",
            "plot",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    skipped_path = args.outdir / "trajectory_comparison_skipped.csv"
    with skipped_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["scene", "reason"])
        writer.writerows(skipped)

    contact = write_contact_sheet(args.outdir, rows)
    print(f"Wrote {len(rows)} plots to {args.outdir}")
    print(f"Wrote summary: {summary_path}")
    print(f"Wrote skipped: {skipped_path}")
    if contact is not None:
        print(f"Wrote contact sheet: {contact}")
    if skipped:
        print("Skipped:")
        for scene, reason in skipped:
            print(f"  {scene}: {reason}")


if __name__ == "__main__":
    main()
