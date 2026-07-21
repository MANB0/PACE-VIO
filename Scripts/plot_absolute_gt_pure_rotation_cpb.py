#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml


WORKDIR = Path(__file__).resolve().parents[1]
MAIN12_CSV = WORKDIR / "analysis_v3bpp_final_paper_results_v1" / "main12_cpb_vs_pure.csv"
DEFAULT_OUTDIR = WORKDIR / "analysis_absolute_trajectory_gt_pure_rotation_cpb"

FIXED_BASELINE_ROOTS = [
    WORKDIR / "Results" / "baseline_7x4_3runs_20260518_234150",
    WORKDIR / "Results" / "holdout_validation" / "fixed_baseline",
    WORKDIR / "Results" / "v3bpp_validation_3scene_54runs" / "fixed_baseline",
    WORKDIR / "Results" / "v3bpp_locked_3scene_54runs" / "fixed_baseline",
]

CPB_ROOTS = [
    WORKDIR / "Results" / "v3bpp_latest_cpb_fdonly_early7_3x",
    WORKDIR / "Results" / "v3bpp_phase1b_cpb_fdonly_5scene_3x",
    WORKDIR / "Results" / "v3bpp_validation_3scene_54runs" / "cpb_fd_only",
    WORKDIR / "Results" / "v3bpp_locked_3scene_54runs" / "cpb_fd_only",
]

METHOD_LABELS = {
    "pure_macvo": "Pure MACVO",
    "rotation_only": "Rotation-only",
    "cpb": "CPB",
}

METHOD_COLORS = {
    "gt": "#111111",
    "pure_macvo": "#377eb8",
    "rotation_only": "#4daf4a",
    "cpb": "#e41a1c",
}


@dataclass(frozen=True)
class RunRecord:
    method: str
    scene: str
    trial: int | None
    poses_csv: Path
    config_yaml: Path
    data_root: Path
    source_root: str


def load_main_scenes() -> list[str]:
    if not MAIN12_CSV.exists():
        raise FileNotFoundError(f"Missing main-scene table: {MAIN12_CSV}")
    scenes: list[str] = []
    with MAIN12_CSV.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            scene = row.get("scene", "").strip()
            if scene:
                scenes.append(scene)
    return scenes


def _nested_get(mapping: dict, keys: Iterable[str], default=None):
    cur = mapping
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def parse_run_config(config_yaml: Path) -> tuple[str, Path, bool, bool]:
    with config_yaml.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    data_args = _nested_get(cfg, ["Data", "args", "args"], {}) or {}
    root_raw = data_args.get("root") or ""
    data_root = Path(root_raw)
    # Some stored configs kept scene: clear_shallow while root points to the
    # actual validation scene. The root directory name is therefore authoritative.
    scene = data_root.name or str(data_args.get("scene", "")).strip()

    odom_args = _nested_get(cfg, ["Odometry", "args"], {}) or {}
    rot = bool(odom_args.get("imu_rot_prior_enable", False))
    trans = bool(odom_args.get("imu_trans_prior_enable", False))
    return scene, data_root, rot, trans


def classify_fixed_method(rot: bool, trans: bool) -> str | None:
    if not rot and not trans:
        return "pure_macvo"
    if rot and not trans:
        return "rotation_only"
    return None


def parse_trial_from_path(path: Path) -> int | None:
    for part in path.parts:
        if part.startswith("trial_"):
            suffix = part.split("trial_", 1)[1]
            if suffix.isdigit():
                return int(suffix)
    return None


def scan_runs(main_scenes: set[str]) -> list[RunRecord]:
    records: list[RunRecord] = []

    def add_from_root(root: Path, forced_method: str | None) -> None:
        if not root.exists():
            return
        for poses_csv in sorted(root.rglob("poses.csv")):
            config_yaml = poses_csv.with_name("config.yaml")
            if not config_yaml.exists():
                continue
            try:
                scene, data_root, rot, trans = parse_run_config(config_yaml)
            except Exception:
                continue
            if scene not in main_scenes:
                continue
            method = forced_method if forced_method else classify_fixed_method(rot, trans)
            if method not in {"pure_macvo", "rotation_only", "cpb"}:
                continue
            records.append(
                RunRecord(
                    method=method,
                    scene=scene,
                    trial=parse_trial_from_path(poses_csv),
                    poses_csv=poses_csv,
                    config_yaml=config_yaml,
                    data_root=data_root,
                    source_root=str(root.relative_to(WORKDIR)),
                )
            )

    for root in FIXED_BASELINE_ROOTS:
        add_from_root(root, forced_method=None)
    for root in CPB_ROOTS:
        add_from_root(root, forced_method="cpb")
    return records


def assign_trials(records: list[RunRecord]) -> dict[tuple[str, str, int], RunRecord]:
    grouped: dict[tuple[str, str], list[RunRecord]] = {}
    for rec in records:
        grouped.setdefault((rec.scene, rec.method), []).append(rec)

    assigned: dict[tuple[str, str, int], RunRecord] = {}
    for (scene, method), items in grouped.items():
        def sort_key(rec: RunRecord):
            explicit = rec.trial if rec.trial is not None else 10_000
            return (explicit, str(rec.poses_csv))

        for idx, rec in enumerate(sorted(items, key=sort_key), start=1):
            trial = rec.trial if rec.trial is not None else idx
            # Keep the first run for duplicate trial numbers from the same source rule.
            assigned.setdefault((scene, method, trial), rec)
    return assigned


def read_xyz_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    if data.shape == ():
        data = data.reshape(1)
    names = data.dtype.names or ()
    time_name = "timestamp_ns" if "timestamp_ns" in names else "timestamp"
    x_name = "tx" if "tx" in names else "x"
    y_name = "ty" if "ty" in names else "y"
    z_name = "tz" if "tz" in names else "z"
    t = np.asarray(data[time_name], dtype=np.int64)
    xyz = np.column_stack([
        np.asarray(data[x_name], dtype=float),
        np.asarray(data[y_name], dtype=float),
        np.asarray(data[z_name], dtype=float),
    ])
    mask = np.isfinite(xyz).all(axis=1)
    return t[mask], xyz[mask]


def crop_gt_to_estimate_span(
    gt_t: np.ndarray,
    gt_xyz: np.ndarray,
    estimate_times: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    nonempty = [t for t in estimate_times if t.size]
    if not nonempty:
        return gt_t, gt_xyz
    t_min = min(int(t.min()) for t in nonempty)
    t_max = max(int(t.max()) for t in nonempty)
    mask = (gt_t >= t_min) & (gt_t <= t_max)
    if mask.sum() >= 2:
        return gt_t[mask], gt_xyz[mask]
    return gt_t, gt_xyz


def nearest_gt_rmse(gt_t: np.ndarray, gt_xyz: np.ndarray, est_t: np.ndarray, est_xyz: np.ndarray) -> float:
    if gt_t.size == 0 or est_t.size == 0:
        return float("nan")
    order = np.argsort(gt_t)
    gt_t_sorted = gt_t[order]
    gt_xyz_sorted = gt_xyz[order]
    idx = np.searchsorted(gt_t_sorted, est_t)
    idx0 = np.clip(idx - 1, 0, len(gt_t_sorted) - 1)
    idx1 = np.clip(idx, 0, len(gt_t_sorted) - 1)
    choose_1 = np.abs(gt_t_sorted[idx1] - est_t) < np.abs(gt_t_sorted[idx0] - est_t)
    nearest = np.where(choose_1, idx1, idx0)
    n = min(len(nearest), len(est_xyz))
    err = est_xyz[:n] - gt_xyz_sorted[nearest[:n]]
    return float(np.sqrt(np.mean(np.sum(err * err, axis=1))))


def set_equal_2d(ax, arrays: list[np.ndarray], dims: tuple[int, int]) -> None:
    pts = [arr[:, list(dims)] for arr in arrays if arr.size]
    if not pts:
        return
    all_pts = np.vstack(pts)
    lo = np.nanmin(all_pts, axis=0)
    hi = np.nanmax(all_pts, axis=0)
    center = (lo + hi) / 2.0
    span = max(float(np.max(hi - lo)), 1.0)
    pad = span * 0.06
    ax.set_xlim(center[0] - span / 2 - pad, center[0] + span / 2 + pad)
    ax.set_ylim(center[1] - span / 2 - pad, center[1] + span / 2 + pad)
    ax.set_aspect("equal", adjustable="box")


def plot_one(
    scene: str,
    trial: int,
    runs: dict[str, RunRecord],
    out_png: Path,
    *,
    crop_gt: bool,
) -> dict[str, float | str | int]:
    trajectories: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for method, rec in runs.items():
        trajectories[method] = read_xyz_csv(rec.poses_csv)

    ref_path = runs["cpb"].data_root / "ref_pose.csv"
    if not ref_path.exists():
        # The fixed baselines and CPB should point to the same recording root.
        for rec in runs.values():
            candidate = rec.data_root / "ref_pose.csv"
            if candidate.exists():
                ref_path = candidate
                break
    gt_t, gt_xyz_full = read_xyz_csv(ref_path)
    if crop_gt:
        _, gt_xyz = crop_gt_to_estimate_span(
            gt_t,
            gt_xyz_full,
            [t for t, _ in trajectories.values()],
        )
    else:
        gt_xyz = gt_xyz_full

    rmses = {
        method: nearest_gt_rmse(gt_t, gt_xyz_full, t, xyz)
        for method, (t, xyz) in trajectories.items()
    }

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.8), constrained_layout=True)
    panels = [
        (axes[0], (0, 1), "Top view: X-Y", "x [m]", "y [m]"),
        (axes[1], (0, 2), "Vertical view: X-Z", "x [m]", "z [m]"),
    ]
    for ax, dims, title, xlabel, ylabel in panels:
        ax.plot(
            gt_xyz[:, dims[0]],
            gt_xyz[:, dims[1]],
            color=METHOD_COLORS["gt"],
            linewidth=2.4,
            label="GT",
            zorder=10,
        )
        for method in ["pure_macvo", "rotation_only", "cpb"]:
            _, xyz = trajectories[method]
            ax.plot(
                xyz[:, dims[0]],
                xyz[:, dims[1]],
                color=METHOD_COLORS[method],
                linewidth=1.55,
                alpha=0.9,
                label=f"{METHOD_LABELS[method]} ({rmses[method]:.2f} m)",
            )
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle=":", linewidth=0.55, alpha=0.6)
        set_equal_2d(ax, [gt_xyz] + [xyz for _, xyz in trajectories.values()], dims)
    axes[0].legend(loc="best", fontsize=8.5, frameon=True)
    fig.suptitle(f"{scene} | trial {trial} | absolute trajectory, no alignment", fontsize=12)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=220)
    plt.close(fig)

    return {
        "scene": scene,
        "trial": trial,
        "plot_path": str(out_png.relative_to(WORKDIR)),
        "ref_pose": str(ref_path),
        "pure_rmse_direct": rmses["pure_macvo"],
        "rotation_rmse_direct": rmses["rotation_only"],
        "cpb_rmse_direct": rmses["cpb"],
        "pure_run": str(runs["pure_macvo"].poses_csv.relative_to(WORKDIR)),
        "rotation_run": str(runs["rotation_only"].poses_csv.relative_to(WORKDIR)),
        "cpb_run": str(runs["cpb"].poses_csv.relative_to(WORKDIR)),
    }


def plot_contact_sheet(summary_rows: list[dict[str, object]], out_png: Path, *, crop_gt: bool) -> None:
    trial1 = [row for row in summary_rows if int(row["trial"]) == 1]
    if not trial1:
        return
    ncols = 3
    nrows = int(np.ceil(len(trial1) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 4.2 * nrows), constrained_layout=True)
    axes_arr = np.atleast_1d(axes).reshape(nrows, ncols)

    for ax in axes_arr.ravel():
        ax.axis("off")

    for ax, row in zip(axes_arr.ravel(), trial1):
        scene = str(row["scene"])
        runs = {
            "pure_macvo": Path(WORKDIR / str(row["pure_run"])),
            "rotation_only": Path(WORKDIR / str(row["rotation_run"])),
            "cpb": Path(WORKDIR / str(row["cpb_run"])),
        }
        ref_path = Path(str(row["ref_pose"]))
        gt_t, gt_xyz_full = read_xyz_csv(ref_path)
        trajs = {method: read_xyz_csv(path) for method, path in runs.items()}
        if crop_gt:
            _, gt_xyz = crop_gt_to_estimate_span(gt_t, gt_xyz_full, [t for t, _ in trajs.values()])
        else:
            gt_xyz = gt_xyz_full
        ax.axis("on")
        ax.plot(gt_xyz[:, 0], gt_xyz[:, 1], color=METHOD_COLORS["gt"], linewidth=2.0, label="GT")
        for method in ["pure_macvo", "rotation_only", "cpb"]:
            _, xyz = trajs[method]
            ax.plot(xyz[:, 0], xyz[:, 1], color=METHOD_COLORS[method], linewidth=1.2, label=METHOD_LABELS[method])
        ax.set_title(scene, fontsize=10)
        ax.grid(True, linestyle=":", linewidth=0.45, alpha=0.5)
        set_equal_2d(ax, [gt_xyz] + [xyz for _, xyz in trajs.values()], (0, 1))
    handles, labels = axes_arr.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
    fig.suptitle("Trial 1 absolute trajectories (X-Y), no alignment", fontsize=14)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=190)
    plt.close(fig)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot absolute GT/Pure MACVO/Rotation-only/CPB trajectories without alignment."
    )
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--max-trials", type=int, default=3)
    parser.add_argument(
        "--crop-gt-to-estimate-span",
        action="store_true",
        help="Only crop the displayed GT by timestamp span; no coordinate alignment is ever applied.",
    )
    args = parser.parse_args()

    outdir = args.outdir if args.outdir.is_absolute() else WORKDIR / args.outdir
    figures_dir = outdir / "figures"
    main_scenes = load_main_scenes()
    main_scene_set = set(main_scenes)

    records = scan_runs(main_scene_set)
    assigned = assign_trials(records)

    summary_rows: list[dict[str, object]] = []
    skipped_rows: list[dict[str, object]] = []
    for scene in main_scenes:
        for trial in range(1, args.max_trials + 1):
            runs: dict[str, RunRecord] = {}
            missing: list[str] = []
            for method in ["pure_macvo", "rotation_only", "cpb"]:
                rec = assigned.get((scene, method, trial))
                if rec is None:
                    missing.append(method)
                else:
                    runs[method] = rec
            if missing:
                skipped_rows.append({"scene": scene, "trial": trial, "missing_methods": ";".join(missing)})
                continue
            out_png = figures_dir / f"{scene}_trial{trial}_absolute_gt_pure_rotation_cpb.png"
            try:
                summary_rows.append(
                    plot_one(
                        scene,
                        trial,
                        runs,
                        out_png,
                        crop_gt=bool(args.crop_gt_to_estimate_span),
                    )
                )
            except Exception as exc:
                skipped_rows.append({"scene": scene, "trial": trial, "missing_methods": f"plot_error:{exc}"})

    write_csv(outdir / "absolute_trajectory_plot_summary.csv", summary_rows)
    if skipped_rows:
        write_csv(outdir / "absolute_trajectory_plot_skipped.csv", skipped_rows)
    plot_contact_sheet(
        summary_rows,
        outdir / "absolute_trajectory_contact_sheet_trial1.png",
        crop_gt=bool(args.crop_gt_to_estimate_span),
    )

    readme = outdir / "README.md"
    readme.write_text(
        "\n".join(
            [
                "# Absolute GT/Pure/Rotation/CPB Trajectory Plots",
                "",
                "These figures compare GT, Pure MACVO, Rotation-only, and CPB trajectories.",
                "",
                "- No SE(3) alignment.",
                "- No Sim(3) alignment.",
                "- No translation recentering.",
                "- No scale correction.",
                "- GT is plotted as recorded by default, without timestamp cropping.",
                "- Optional `--crop-gt-to-estimate-span` affects display range only; it does not align coordinates.",
                "",
                f"Generated plots: {len(summary_rows)}",
                f"Skipped combinations: {len(skipped_rows)}",
                "",
                "Main output:",
                "- `figures/*_absolute_gt_pure_rotation_cpb.png`",
                "- `absolute_trajectory_contact_sheet_trial1.png`",
                "- `absolute_trajectory_plot_summary.csv`",
                "- `absolute_trajectory_plot_skipped.csv` if any combination is incomplete",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"Wrote {len(summary_rows)} plots to {figures_dir}")
    print(f"Wrote summary to {outdir / 'absolute_trajectory_plot_summary.csv'}")
    if skipped_rows:
        print(f"Skipped {len(skipped_rows)} combinations; see {outdir / 'absolute_trajectory_plot_skipped.csv'}")


if __name__ == "__main__":
    main()
