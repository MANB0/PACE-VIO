#!/usr/bin/env python3
"""
Re-evaluate paper-facing AIM-VO results with aligned ATE protocols.

The historical paper tables used a strict direct-origin ATE: estimated and
ground-truth trajectories are compared in their metric coordinate frame after
only subtracting the first position.  This script keeps that value as a sanity
check and additionally reports two common trajectory-evaluation protocols:

  * SE(3)-aligned ATE: best-fit rotation and translation, no scale correction.
  * Sim(3)-aligned ATE: best-fit similarity transform, including scale.

Outputs are written to analysis_aligned_ate_aimvo_paper_v1/.
"""

from __future__ import annotations

import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

try:
    import yaml
except Exception:  # pragma: no cover - only used for a small path classifier.
    yaml = None

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


WORKDIR = Path("/home/admin1/macvo-dev")
FINAL_RESULTS_CSV = WORKDIR / "analysis_v3bpp_final_paper_results_v1/final_cpb_vs_pure_16scene.csv"
VARIABILITY_CSV = WORKDIR / "analysis_v3bpp_discussion_package/table_6_all_16scene_all_method_variability.csv"
OUTPUT_ROOT = WORKDIR / "analysis_aligned_ate_aimvo_paper_v1"
FIG_ROOT = OUTPUT_ROOT / "figures"
PER_SCENE_FIG_ROOT = FIG_ROOT / "per_scene"

SCENE_ROOTS = {
    "turbid_harbor": Path("/mnt/e/文档/holoocean/code/recordings/batch_20260515_155653/turbid_harbor"),
    "clear_shallow": Path("/mnt/e/文档/holoocean/code/recordings/batch_20260515_155653/clear_shallow"),
    "deep_dark": Path("/mnt/e/文档/holoocean/code/recordings/batch_20260515_155653/deep_dark"),
    "caustic_shallow": Path("/mnt/e/文档/holoocean/code/recordings/batch_20260515_155653/caustic_shallow"),
    "dam_inspection": Path("/mnt/e/文档/holoocean/code/recordings/batch_20260515_155653/dam_inspection"),
    "murky_coast": Path("/mnt/e/文档/holoocean/code/recordings/batch_20260515_155653/murky_coast"),
    "open_water": Path("/mnt/e/文档/holoocean/code/recordings/batch_20260515_155653/open_water"),
    "moderate_turbidity": Path("/mnt/e/文档/holoocean/code/recordings/batch_20260528_203401/moderate_turbidity"),
    "open_water_overcast": Path("/mnt/e/文档/holoocean/code/recordings/batch_20260528_203401/open_water_overcast"),
    "twilight_coast": Path("/mnt/e/文档/holoocean/code/recordings/batch_20260528_203401/twilight_coast"),
    "validation_moderate_harbor": Path("/mnt/e/文档/holoocean/code/recordings/batch_20260601_162707/validation_moderate_harbor"),
    "validation_transient_dropout": Path("/mnt/e/文档/holoocean/code/recordings/batch_20260601_162707/validation_transient_dropout"),
    "validation_twilight_structure": Path("/mnt/e/文档/holoocean/code/recordings/batch_20260601_162707/validation_twilight_structure"),
    "locked_murky_entry_help": Path("/mnt/e/文档/holoocean/code/recordings/batch_20260618_110853/locked_murky_entry_help"),
    "locked_clear_imu_harm": Path("/mnt/e/文档/holoocean/code/recordings/batch_20260618_110853/locked_clear_imu_harm"),
    "locked_quality_degrade_no_dropout": Path("/mnt/e/文档/holoocean/code/recordings/batch_20260618_110853/locked_quality_degrade_no_dropout"),
}

EARLY_SCENES = {
    "turbid_harbor",
    "clear_shallow",
    "deep_dark",
    "caustic_shallow",
    "dam_inspection",
    "murky_coast",
    "open_water",
}
HOLDOUT_SCENES = {"moderate_turbidity", "open_water_overcast", "twilight_coast"}
VALIDATION_SCENES = {
    "validation_moderate_harbor",
    "validation_transient_dropout",
    "validation_twilight_structure",
}
LOCKED_SCENES = {
    "locked_murky_entry_help",
    "locked_clear_imu_harm",
    "locked_quality_degrade_no_dropout",
}


@dataclass
class SceneMeta:
    scene: str
    scene_group: str
    paper_role: str
    cpb_source: str
    old_pure_median: float
    old_aimvo_median: float


@dataclass
class RunRecord:
    scene: str
    method: str
    source: str
    trial: str
    poses_path: Path
    old_direct_ate_report: float | None


@dataclass
class EvalRecord:
    scene: str
    paper_role: str
    method: str
    source: str
    trial: str
    old_direct_ate_report: float | None
    direct_origin_ate: float
    direct_raw_ate: float
    se3_ate: float
    sim3_ate: float
    sim3_scale: float
    n_matched: int
    poses_path: Path
    gt_path: Path
    match_mode: str


@dataclass
class PlotTrajectory:
    scene: str
    method: str
    se3_ate: float
    sim3_ate: float
    sim3_scale: float
    xyz_se3: np.ndarray
    xyz_sim3: np.ndarray


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def parse_trial_values(value: str | None) -> list[float]:
    if not value:
        return []
    out: list[float] = []
    for part in value.split(";"):
        part = part.strip()
        if not part:
            continue
        try:
            out.append(float(part))
        except ValueError:
            pass
    return out


def parse_float(value: str | None, default: float = math.nan) -> float:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


def load_scene_meta() -> tuple[list[str], dict[str, SceneMeta], dict[str, list[float]], dict[str, list[float]]]:
    rows = read_csv(FINAL_RESULTS_CSV)
    order: list[str] = []
    meta: dict[str, SceneMeta] = {}
    aim_targets: dict[str, list[float]] = {}
    pure_targets: dict[str, list[float]] = {}

    for row in rows:
        scene = row["scene"]
        order.append(scene)
        meta[scene] = SceneMeta(
            scene=scene,
            scene_group=row["scene_group"],
            paper_role=row["paper_role"],
            cpb_source=row["cpb_source"],
            old_pure_median=parse_float(row["pure_median_ATE"]),
            old_aimvo_median=parse_float(row["cpb_median_ATE"]),
        )
        aim_targets[scene] = parse_trial_values(row.get("cpb_trial_ATEs"))

    for row in read_csv(VARIABILITY_CSV):
        scene = row["scene"]
        if scene in meta and row["method"] == "pure_macvo" and scene not in pure_targets:
            pure_targets[scene] = parse_trial_values(row.get("trial_ATEs"))

    return order, meta, aim_targets, pure_targets


def gt_path(scene: str) -> Path:
    return SCENE_ROOTS[scene] / "ref_pose.csv"


def prefix_from_trial(trial: str) -> str:
    # Trial ids in older reports are shortened, e.g. 05_29_132204_batc.
    return trial.replace("_batc", "")


def first_match(pattern: str) -> Path:
    matches = sorted(WORKDIR.glob(pattern))
    if not matches:
        raise FileNotFoundError(pattern)
    if len(matches) > 1:
        poses = [p for p in matches if p.name == "poses.csv"]
        if poses:
            return poses[0]
    return matches[0]


def locate_trial_by_prefix(root: Path, trial: str, scene: str) -> Path:
    prefix = prefix_from_trial(trial)
    matches = sorted((root / "MACVO-HoloOcean-IMU@holoocean_imu").glob(f"{prefix}*{scene}/poses.csv"))
    if not matches:
        raise FileNotFoundError(f"{root}: {trial} {scene}")
    if len(matches) > 1:
        exact_scene = [p for p in matches if p.parent.name.endswith(f"_{scene}")]
        if exact_scene:
            return exact_scene[0]
    return matches[0]


def locate_early_baseline(scene: str, trial: int) -> Path:
    root = WORKDIR / "Results/baseline_7x4_3runs_20260518_234150"
    direct_path = root / f"trial_{trial}" / scene / "pure_macvo" / "poses.csv"
    if direct_path.exists():
        return direct_path
    matches = sorted((root / f"trial_{trial}" / scene / "pure_macvo" / "MACVO-HoloOcean-IMU@holoocean_imu").glob("*/poses.csv"))
    if not matches:
        raise FileNotFoundError(f"early pure {scene} trial {trial}")
    return matches[0]


def read_yaml(path: Path) -> dict:
    if yaml is None:
        raise RuntimeError("PyYAML is required to classify holdout fixed-baseline runs.")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def classify_fixed_config(config: dict) -> str:
    od = config.get("Odometry", {})
    args = od.get("args", {})
    opt_args = od.get("optimizer", {}).get("args", {})
    mapping = args.get("mapping", True)
    post_fusion = opt_args.get("post_imu_fusion_enable", True)
    if not mapping and not post_fusion:
        return "ruleB"
    rot = bool(args.get("imu_rot_prior_enable", False))
    trans = bool(args.get("imu_trans_prior_enable", False))
    if not rot and not trans:
        return "pure_macvo"
    if rot and not trans:
        return "rotation_only"
    if not rot and trans:
        return "translation_only"
    if rot and trans:
        return "full_imu"
    return "unknown"


def collect_holdout_pure_runs(targets: dict[str, list[float]]) -> list[RunRecord]:
    root = WORKDIR / "Results/holdout_validation/fixed_baseline/MACVO-HoloOcean-IMU@holoocean_imu"
    grouped: dict[str, list[Path]] = {scene: [] for scene in HOLDOUT_SCENES}
    for run_dir in sorted(root.glob("*")):
        if not run_dir.is_dir():
            continue
        config_path = run_dir / "config.yaml"
        poses_path = run_dir / "poses.csv"
        if not config_path.exists() or not poses_path.exists():
            continue
        scene = next((s for s in HOLDOUT_SCENES if s in run_dir.name), None)
        if scene is None:
            continue
        try:
            method = classify_fixed_config(read_yaml(config_path))
        except Exception:
            continue
        if method == "pure_macvo":
            grouped[scene].append(poses_path)

    records: list[RunRecord] = []
    for scene, paths in grouped.items():
        old_values = targets.get(scene, [])
        for idx, path in enumerate(sorted(paths), start=1):
            old = old_values[idx - 1] if idx - 1 < len(old_values) else None
            trial = "_".join(path.parent.name.split("_")[:3])
            records.append(RunRecord(scene, "pure_macvo", "holdout_validation_fixed_baseline", trial, path, old))
    return records


def collect_runs(
    aim_targets: dict[str, list[float]], pure_targets: dict[str, list[float]]
) -> tuple[list[RunRecord], list[str]]:
    records: list[RunRecord] = []
    warnings: list[str] = []

    def add(scene: str, method: str, source: str, trial: str, path: Path, old: float | None) -> None:
        if not path.exists():
            warnings.append(f"missing poses: {scene} {method} {trial}: {path}")
            return
        records.append(RunRecord(scene, method, source, trial, path, old))

    # AIM-VO / CP-B-FD-only, paper-facing version.
    for row in read_csv(WORKDIR / "analysis_v3bpp_latest_cpb_fdonly_early7_report/early7_latest_cpb_trial_summary.csv"):
        scene = row["scene"]
        add(scene, "aimvo", "latest_cpb_early7", row["trial"], Path(row["poses_path"]), parse_float(row.get("ATE"), math.nan))

    phase_root = WORKDIR / "Results/v3bpp_phase1b_cpb_fdonly_5scene_3x"
    for row in read_csv(WORKDIR / "analysis_v3bpp_phase1b_cpb_report/phase1b_trial_summary.csv"):
        if row["exp"] != "CP-B-FD-only":
            continue
        scene = row["scene"]
        if scene not in HOLDOUT_SCENES:
            continue
        try:
            path = locate_trial_by_prefix(phase_root, row["trial"], scene)
        except FileNotFoundError as exc:
            warnings.append(str(exc))
            continue
        add(scene, "aimvo", "phase1b_cpb", row["trial"], path, parse_float(row.get("ATE"), math.nan))

    validation_root = WORKDIR / "Results/v3bpp_validation_3scene_54runs/cpb_fd_only"
    validation_fixed_root = WORKDIR / "Results/v3bpp_validation_3scene_54runs/fixed_baseline"
    for row in read_csv(WORKDIR / "analysis_v3bpp_validation_3scene_report/validation_trial_summary.csv"):
        scene = row["scene"]
        method = row["method"]
        if method == "cpb_fd_only":
            try:
                path = locate_trial_by_prefix(validation_root, row["trial"], scene)
            except FileNotFoundError as exc:
                warnings.append(str(exc))
                continue
            add(scene, "aimvo", "validation_3scene", row["trial"], path, parse_float(row.get("ATE"), math.nan))
        elif method == "pure_macvo":
            try:
                path = locate_trial_by_prefix(validation_fixed_root, row["trial"], scene)
            except FileNotFoundError as exc:
                warnings.append(str(exc))
                continue
            add(scene, "pure_macvo", "validation_fixed_baseline", row["trial"], path, parse_float(row.get("ATE"), math.nan))

    for row in read_csv(WORKDIR / "analysis_v3bpp_locked_3scene_report/locked_trial_summary.csv"):
        scene = row["scene"]
        method = row["method"]
        if method == "cpb_fd_only":
            add(scene, "aimvo", "locked_3scene", row["trial"], Path(row["run_dir"]) / "poses.csv", parse_float(row.get("ATE"), math.nan))
        elif method == "pure_macvo":
            add(scene, "pure_macvo", "locked_fixed_baseline", row["trial"], Path(row["run_dir"]) / "poses.csv", parse_float(row.get("ATE"), math.nan))

    # Pure MACVO for early fixed-baseline scenes.
    for scene in sorted(EARLY_SCENES):
        old_values = pure_targets.get(scene, [])
        for trial in (1, 2, 3):
            try:
                path = locate_early_baseline(scene, trial)
            except FileNotFoundError as exc:
                warnings.append(str(exc))
                continue
            old = old_values[trial - 1] if trial - 1 < len(old_values) else None
            add(scene, "pure_macvo", "baseline_7x4_3runs", str(trial), path, old)

    records.extend(collect_holdout_pure_runs(pure_targets))

    # Keep only the paper scenes and the two methods needed for this comparison.
    keep_scenes = set(SCENE_ROOTS.keys())
    records = [r for r in records if r.scene in keep_scenes and r.method in {"pure_macvo", "aimvo"}]

    return records, warnings


def load_xyz_with_timestamps(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        timestamps: list[int] = []
        xyz: list[list[float]] = []
        for row in reader:
            ts_key = "timestamp_ns" if "timestamp_ns" in row else "timestamp"
            x_key = "tx" if "tx" in row else "x"
            y_key = "ty" if "ty" in row else "y"
            z_key = "tz" if "tz" in row else "z"
            try:
                timestamps.append(int(float(row[ts_key])))
                xyz.append([float(row[x_key]), float(row[y_key]), float(row[z_key])])
            except (KeyError, TypeError, ValueError):
                continue
    if not xyz:
        raise ValueError(f"No trajectory rows in {path}")
    return np.asarray(timestamps, dtype=np.int64), np.asarray(xyz, dtype=np.float64)


def match_trajectories(
    est_ts: np.ndarray, est_xyz: np.ndarray, gt_ts: np.ndarray, gt_xyz: np.ndarray
) -> tuple[np.ndarray, np.ndarray, str]:
    gt_index = {int(ts): i for i, ts in enumerate(gt_ts)}
    est_indices: list[int] = []
    gt_indices: list[int] = []
    for i, ts in enumerate(est_ts):
        j = gt_index.get(int(ts))
        if j is not None:
            est_indices.append(i)
            gt_indices.append(j)
    if len(est_indices) >= 10:
        return est_xyz[est_indices], gt_xyz[gt_indices], "exact_timestamp"

    n = min(len(est_xyz), len(gt_xyz))
    if n < 10:
        raise ValueError(f"Too few matched poses: {n}")
    return est_xyz[:n], gt_xyz[:n], "index_prefix"


def rmse(a: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum(a * a, axis=1))))


def direct_origin_ate(est: np.ndarray, gt: np.ndarray) -> float:
    return rmse((est - est[0]) - (gt - gt[0]))


def direct_raw_ate(est: np.ndarray, gt: np.ndarray) -> float:
    return rmse(est - gt)


def umeyama_align(src: np.ndarray, dst: np.ndarray, with_scale: bool) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    if src.shape != dst.shape or src.shape[0] < 3:
        raise ValueError("source and destination trajectories must have matching shape with at least 3 samples")
    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    src_c = src - mu_src
    dst_c = dst - mu_dst
    cov = (dst_c.T @ src_c) / src.shape[0]
    u, d, vt = np.linalg.svd(cov)
    s = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        s[-1, -1] = -1
    rot = u @ s @ vt
    if with_scale:
        var_src = float(np.sum(src_c * src_c) / src.shape[0])
        scale = float(np.trace(np.diag(d) @ s) / var_src) if var_src > 0 else 1.0
    else:
        scale = 1.0
    trans = mu_dst - scale * (rot @ mu_src)
    aligned = (scale * (rot @ src.T)).T + trans
    return aligned, scale, rot, trans


def evaluate_run(record: RunRecord, scene_meta: SceneMeta) -> tuple[EvalRecord, PlotTrajectory]:
    gpath = gt_path(record.scene)
    est_ts, est_xyz_all = load_xyz_with_timestamps(record.poses_path)
    gt_ts, gt_xyz_all = load_xyz_with_timestamps(gpath)
    est_xyz, gt_xyz, match_mode = match_trajectories(est_ts, est_xyz_all, gt_ts, gt_xyz_all)

    est_se3, _, _, _ = umeyama_align(est_xyz, gt_xyz, with_scale=False)
    est_sim3, sim3_scale, _, _ = umeyama_align(est_xyz, gt_xyz, with_scale=True)

    eval_record = EvalRecord(
        scene=record.scene,
        paper_role=scene_meta.paper_role,
        method=record.method,
        source=record.source,
        trial=record.trial,
        old_direct_ate_report=record.old_direct_ate_report,
        direct_origin_ate=direct_origin_ate(est_xyz, gt_xyz),
        direct_raw_ate=direct_raw_ate(est_xyz, gt_xyz),
        se3_ate=rmse(est_se3 - gt_xyz),
        sim3_ate=rmse(est_sim3 - gt_xyz),
        sim3_scale=sim3_scale,
        n_matched=int(len(est_xyz)),
        poses_path=record.poses_path,
        gt_path=gpath,
        match_mode=match_mode,
    )
    plot_traj = PlotTrajectory(
        scene=record.scene,
        method=record.method,
        se3_ate=eval_record.se3_ate,
        sim3_ate=eval_record.sim3_ate,
        sim3_scale=sim3_scale,
        xyz_se3=est_se3,
        xyz_sim3=est_sim3,
    )
    return eval_record, plot_traj


def median(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=np.float64)
    return float(np.median(arr)) if len(arr) else math.nan


def stat(values: Iterable[float], fn: str) -> float:
    arr = np.asarray(list(values), dtype=np.float64)
    if len(arr) == 0:
        return math.nan
    if fn == "min":
        return float(np.min(arr))
    if fn == "max":
        return float(np.max(arr))
    if fn == "std":
        return float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    if fn == "mean":
        return float(np.mean(arr))
    raise ValueError(fn)


def fmt(value: float, ndigits: int = 3) -> str:
    if value is None or not np.isfinite(value):
        return ""
    return f"{value:.{ndigits}f}"


def fmt_interval(med: float, mn: float, mx: float, ndigits: int = 3) -> str:
    if not np.isfinite(med):
        return ""
    return f"{med:.{ndigits}f} [{mn:.{ndigits}f}-{mx:.{ndigits}f}]"


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def method_summary_rows(eval_records: list[EvalRecord], scene_order: list[str], meta: dict[str, SceneMeta]) -> list[dict[str, object]]:
    by_key: dict[tuple[str, str], list[EvalRecord]] = {}
    for rec in eval_records:
        by_key.setdefault((rec.scene, rec.method), []).append(rec)

    rows: list[dict[str, object]] = []
    for scene in scene_order:
        for method in ("pure_macvo", "aimvo"):
            records = by_key.get((scene, method), [])
            direct = [r.direct_origin_ate for r in records]
            se3 = [r.se3_ate for r in records]
            sim3 = [r.sim3_ate for r in records]
            scales = [r.sim3_scale for r in records]
            rows.append(
                {
                    "scene": scene,
                    "paper_role": meta[scene].paper_role,
                    "method": method,
                    "n_trials": len(records),
                    "direct_origin_median": median(direct),
                    "direct_origin_min": stat(direct, "min"),
                    "direct_origin_max": stat(direct, "max"),
                    "direct_origin_std": stat(direct, "std"),
                    "se3_median": median(se3),
                    "se3_min": stat(se3, "min"),
                    "se3_max": stat(se3, "max"),
                    "se3_std": stat(se3, "std"),
                    "sim3_median": median(sim3),
                    "sim3_min": stat(sim3, "min"),
                    "sim3_max": stat(sim3, "max"),
                    "sim3_std": stat(sim3, "std"),
                    "sim3_scale_median": median(scales),
                    "sim3_scale_min": stat(scales, "min"),
                    "sim3_scale_max": stat(scales, "max"),
                }
            )
    return rows


def comparison_rows(method_rows: list[dict[str, object]], scene_order: list[str], meta: dict[str, SceneMeta]) -> list[dict[str, object]]:
    by_key = {(r["scene"], r["method"]): r for r in method_rows}
    rows: list[dict[str, object]] = []
    for scene in scene_order:
        pure = by_key.get((scene, "pure_macvo"), {})
        aim = by_key.get((scene, "aimvo"), {})
        pure_se3 = parse_float(str(pure.get("se3_median", "")))
        aim_se3 = parse_float(str(aim.get("se3_median", "")))
        pure_sim3 = parse_float(str(pure.get("sim3_median", "")))
        aim_sim3 = parse_float(str(aim.get("sim3_median", "")))

        def gain(pure_val: float, aim_val: float) -> float:
            if not np.isfinite(pure_val) or abs(pure_val) < 1e-12 or not np.isfinite(aim_val):
                return math.nan
            return 100.0 * (pure_val - aim_val) / pure_val

        rows.append(
            {
                "scene": scene,
                "paper_role": meta[scene].paper_role,
                "n_pure": pure.get("n_trials", 0),
                "n_aimvo": aim.get("n_trials", 0),
                "pure_direct_origin": pure.get("direct_origin_median", math.nan),
                "aimvo_direct_origin": aim.get("direct_origin_median", math.nan),
                "pure_se3": pure_se3,
                "aimvo_se3": aim_se3,
                "se3_delta_aimvo_minus_pure": aim_se3 - pure_se3 if np.isfinite(pure_se3) and np.isfinite(aim_se3) else math.nan,
                "se3_gain_pct": gain(pure_se3, aim_se3),
                "pure_sim3": pure_sim3,
                "aimvo_sim3": aim_sim3,
                "sim3_delta_aimvo_minus_pure": aim_sim3 - pure_sim3 if np.isfinite(pure_sim3) and np.isfinite(aim_sim3) else math.nan,
                "sim3_gain_pct": gain(pure_sim3, aim_sim3),
                "pure_sim3_scale": pure.get("sim3_scale_median", math.nan),
                "aimvo_sim3_scale": aim.get("sim3_scale_median", math.nan),
                "old_pure_direct_median": meta[scene].old_pure_median,
                "old_aimvo_direct_median": meta[scene].old_aimvo_median,
            }
        )
    return rows


def write_markdown_table(path: Path, rows: list[dict[str, object]], include_excluded: bool = True) -> None:
    filtered = [r for r in rows if include_excluded or r["paper_role"] == "main"]
    headers = [
        "Scene",
        "Role",
        "Pure SE(3)",
        "AIM-VO SE(3)",
        "SE(3) gain",
        "Pure Sim(3)",
        "AIM-VO Sim(3)",
        "Sim(3) gain",
        "AIM-VO scale",
    ]
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in filtered:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["scene"]),
                    str(row["paper_role"]),
                    fmt(row["pure_se3"]),
                    fmt(row["aimvo_se3"]),
                    fmt(row["se3_gain_pct"], 1) + "%" if np.isfinite(row["se3_gain_pct"]) else "",
                    fmt(row["pure_sim3"]),
                    fmt(row["aimvo_sim3"]),
                    fmt(row["sim3_gain_pct"], 1) + "%" if np.isfinite(row["sim3_gain_pct"]) else "",
                    fmt(row["aimvo_sim3_scale"], 4),
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def best_plot_trajectory(
    scene: str, method: str, eval_records: list[EvalRecord], plot_records: dict[tuple[str, str, str], PlotTrajectory], metric: str
) -> PlotTrajectory | None:
    candidates = [r for r in eval_records if r.scene == scene and r.method == method]
    if not candidates:
        return None
    values = np.asarray([getattr(r, f"{metric}_ate") for r in candidates], dtype=np.float64)
    med = float(np.median(values))
    selected = min(candidates, key=lambda r: abs(getattr(r, f"{metric}_ate") - med))
    return plot_records.get((selected.scene, selected.method, selected.trial))


def set_equal_axes(ax: plt.Axes, arrays: list[np.ndarray], dims: tuple[int, int]) -> None:
    pts = [a[:, list(dims)] for a in arrays if a is not None and len(a)]
    if not pts:
        return
    xy = np.vstack(pts)
    mn = xy.min(axis=0)
    mx = xy.max(axis=0)
    center = (mn + mx) / 2.0
    radius = max(float(np.max(mx - mn)) / 2.0, 1e-3)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_aspect("equal", adjustable="box")


def plot_scene(scene: str, eval_records: list[EvalRecord], plot_records: dict[tuple[str, str, str], PlotTrajectory], align_mode: str) -> Path:
    gt_ts, gt_xyz = load_xyz_with_timestamps(gt_path(scene))
    pure = best_plot_trajectory(scene, "pure_macvo", eval_records, plot_records, align_mode)
    aim = best_plot_trajectory(scene, "aimvo", eval_records, plot_records, align_mode)
    attr = f"xyz_{align_mode}"

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), constrained_layout=True)
    for ax, dims, xlabel, ylabel, title in [
        (axes[0], (0, 1), "x [m]", "y [m]", "Top-down XY"),
        (axes[1], (0, 2), "x [m]", "z [m]", "Vertical XZ"),
    ]:
        ax.plot(gt_xyz[:, dims[0]], gt_xyz[:, dims[1]], color="0.15", lw=2.0, label="GT")
        arrays = [gt_xyz]
        if pure is not None:
            xyz = getattr(pure, attr)
            arrays.append(xyz)
            ate = pure.se3_ate if align_mode == "se3" else pure.sim3_ate
            ax.plot(xyz[:, dims[0]], xyz[:, dims[1]], color="#2C6DB2", lw=1.7, label=f"Pure MACVO ({ate:.2f} m)")
        if aim is not None:
            xyz = getattr(aim, attr)
            arrays.append(xyz)
            ate = aim.se3_ate if align_mode == "se3" else aim.sim3_ate
            ax.plot(xyz[:, dims[0]], xyz[:, dims[1]], color="#D55E00", lw=1.7, label=f"AIM-VO ({ate:.2f} m)")
        set_equal_axes(ax, arrays, dims)
        ax.grid(True, alpha=0.25, lw=0.6)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
    fig.suptitle(f"{scene} - {align_mode.upper()} aligned trajectory comparison", fontsize=12)
    axes[0].legend(loc="best", fontsize=8)
    out = PER_SCENE_FIG_ROOT / f"{scene}_{align_mode}_trajectory.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def plot_contact_sheet(scene_order: list[str], eval_records: list[EvalRecord], plot_records: dict[tuple[str, str, str], PlotTrajectory], align_mode: str) -> Path:
    fig, axes = plt.subplots(4, 4, figsize=(14, 13), constrained_layout=True)
    attr = f"xyz_{align_mode}"
    for ax, scene in zip(axes.ravel(), scene_order):
        _, gt_xyz = load_xyz_with_timestamps(gt_path(scene))
        pure = best_plot_trajectory(scene, "pure_macvo", eval_records, plot_records, align_mode)
        aim = best_plot_trajectory(scene, "aimvo", eval_records, plot_records, align_mode)
        ax.plot(gt_xyz[:, 0], gt_xyz[:, 1], color="0.15", lw=1.4, label="GT")
        arrays = [gt_xyz]
        if pure is not None:
            xyz = getattr(pure, attr)
            arrays.append(xyz)
            ax.plot(xyz[:, 0], xyz[:, 1], color="#2C6DB2", lw=1.0, label="Pure")
        if aim is not None:
            xyz = getattr(aim, attr)
            arrays.append(xyz)
            ax.plot(xyz[:, 0], xyz[:, 1], color="#D55E00", lw=1.0, label="AIM-VO")
        set_equal_axes(ax, arrays, (0, 1))
        ax.set_title(scene, fontsize=8)
        ax.grid(True, alpha=0.2, lw=0.5)
        ax.tick_params(labelsize=6)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=10)
    fig.suptitle(f"{align_mode.upper()} aligned trajectory comparison, XY view", fontsize=14)
    out = FIG_ROOT / f"trajectory_contact_sheet_{align_mode}.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def write_report(
    comparison: list[dict[str, object]],
    method_rows: list[dict[str, object]],
    eval_records: list[EvalRecord],
    warnings: list[str],
    contact_paths: list[Path],
) -> None:
    main = [r for r in comparison if r["paper_role"] == "main"]
    all_rows = comparison

    def aggregate(rows: list[dict[str, object]], key: str) -> float:
        return median([parse_float(str(r[key])) for r in rows if np.isfinite(parse_float(str(r[key])))])

    lines = [
        "# AIM-VO aligned ATE re-evaluation",
        "",
        "Protocol:",
        "- Direct-origin ATE is retained only as a sanity check against earlier strict tables.",
        "- SE(3)-aligned ATE applies the best rigid transform between estimated and ground-truth trajectories, without scale correction.",
        "- Sim(3)-aligned ATE applies the best similarity transform and therefore absorbs global scale error.",
        "",
        "Key aggregate medians (scene-wise medians):",
        f"- Main scenes, SE(3): Pure MACVO {aggregate(main, 'pure_se3'):.3f} m, AIM-VO {aggregate(main, 'aimvo_se3'):.3f} m.",
        f"- Main scenes, Sim(3): Pure MACVO {aggregate(main, 'pure_sim3'):.3f} m, AIM-VO {aggregate(main, 'aimvo_sim3'):.3f} m.",
        f"- All 16 scenes, SE(3): Pure MACVO {aggregate(all_rows, 'pure_se3'):.3f} m, AIM-VO {aggregate(all_rows, 'aimvo_se3'):.3f} m.",
        f"- All 16 scenes, Sim(3): Pure MACVO {aggregate(all_rows, 'pure_sim3'):.3f} m, AIM-VO {aggregate(all_rows, 'aimvo_sim3'):.3f} m.",
        "",
        "Generated trajectory figures:",
    ]
    for path in contact_paths:
        lines.append(f"- {path.relative_to(WORKDIR)}")
    if warnings:
        lines.extend(["", "Warnings:"])
        lines.extend(f"- {w}" for w in warnings)
    else:
        lines.extend(["", "Warnings: none."])

    lines.extend(
        [
            "",
            "Files:",
            "- aligned_ate_trial_summary.csv: one row per run.",
            "- aligned_ate_method_summary.csv: one row per scene and method.",
            "- aligned_ate_comparison_summary.csv/md: Pure MACVO vs AIM-VO per scene.",
            "- aligned_ate_main12_comparison.md: same table after removing the four excluded scenes.",
        ]
    )
    (OUTPUT_ROOT / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    FIG_ROOT.mkdir(parents=True, exist_ok=True)
    PER_SCENE_FIG_ROOT.mkdir(parents=True, exist_ok=True)

    scene_order, meta, aim_targets, pure_targets = load_scene_meta()
    records, warnings = collect_runs(aim_targets, pure_targets)

    eval_records: list[EvalRecord] = []
    plot_records: dict[tuple[str, str, str], PlotTrajectory] = {}
    for record in records:
        try:
            eval_record, plot_record = evaluate_run(record, meta[record.scene])
        except Exception as exc:
            warnings.append(f"eval failed: {record.scene} {record.method} {record.trial}: {exc}")
            continue
        eval_records.append(eval_record)
        plot_records[(record.scene, record.method, record.trial)] = plot_record

    trial_fields = [
        "scene",
        "paper_role",
        "method",
        "source",
        "trial",
        "old_direct_ate_report",
        "direct_origin_ate",
        "direct_raw_ate",
        "se3_ate",
        "sim3_ate",
        "sim3_scale",
        "n_matched",
        "match_mode",
        "poses_path",
        "gt_path",
    ]
    trial_rows = [
        {
            "scene": r.scene,
            "paper_role": r.paper_role,
            "method": r.method,
            "source": r.source,
            "trial": r.trial,
            "old_direct_ate_report": fmt(r.old_direct_ate_report) if r.old_direct_ate_report is not None else "",
            "direct_origin_ate": fmt(r.direct_origin_ate),
            "direct_raw_ate": fmt(r.direct_raw_ate),
            "se3_ate": fmt(r.se3_ate),
            "sim3_ate": fmt(r.sim3_ate),
            "sim3_scale": fmt(r.sim3_scale, 6),
            "n_matched": r.n_matched,
            "match_mode": r.match_mode,
            "poses_path": str(r.poses_path),
            "gt_path": str(r.gt_path),
        }
        for r in sorted(eval_records, key=lambda x: (scene_order.index(x.scene), x.method, str(x.trial)))
    ]
    write_csv(OUTPUT_ROOT / "aligned_ate_trial_summary.csv", trial_rows, trial_fields)

    method_rows = method_summary_rows(eval_records, scene_order, meta)
    method_fields = list(method_rows[0].keys()) if method_rows else []
    write_csv(OUTPUT_ROOT / "aligned_ate_method_summary.csv", method_rows, method_fields)

    comparison = comparison_rows(method_rows, scene_order, meta)
    comparison_fields = list(comparison[0].keys()) if comparison else []
    write_csv(OUTPUT_ROOT / "aligned_ate_comparison_summary.csv", comparison, comparison_fields)
    write_markdown_table(OUTPUT_ROOT / "aligned_ate_comparison_summary.md", comparison, include_excluded=True)
    write_markdown_table(OUTPUT_ROOT / "aligned_ate_main12_comparison.md", comparison, include_excluded=False)

    per_scene_paths = []
    for scene in scene_order:
        for align_mode in ("se3", "sim3"):
            per_scene_paths.append(plot_scene(scene, eval_records, plot_records, align_mode))
    contact_paths = [
        plot_contact_sheet(scene_order, eval_records, plot_records, "se3"),
        plot_contact_sheet(scene_order, eval_records, plot_records, "sim3"),
    ]

    write_report(comparison, method_rows, eval_records, warnings, contact_paths)

    counts: dict[tuple[str, str], int] = {}
    for r in eval_records:
        counts[(r.scene, r.method)] = counts.get((r.scene, r.method), 0) + 1
    incomplete = [(scene, method, counts.get((scene, method), 0)) for scene in scene_order for method in ("pure_macvo", "aimvo") if counts.get((scene, method), 0) != 3]
    print(f"Wrote {OUTPUT_ROOT}")
    print(f"Evaluated runs: {len(eval_records)}")
    print(f"Per-scene figures: {len(per_scene_paths)}")
    print(f"Contact sheets: {len(contact_paths)}")
    if incomplete:
        print("Incomplete scene/method trial counts:")
        for scene, method, n in incomplete:
            print(f"  {scene} {method}: {n}")
    if warnings:
        print("Warnings:")
        for w in warnings:
            print(f"  {w}")
    return 0 if not incomplete else 2


if __name__ == "__main__":
    raise SystemExit(main())
