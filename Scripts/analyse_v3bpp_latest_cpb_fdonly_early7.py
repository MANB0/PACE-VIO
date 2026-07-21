#!/usr/bin/env python3
"""
Analyze the latest CP-B-FD-only early-7 rerun.

Expected input after the manual run:
    Results/v3bpp_latest_cpb_fdonly_early7_3x/trial_{1,2,3}/{scene}/poses.csv

Outputs:
    analysis_v3bpp_latest_cpb_fdonly_early7_report/
      - early7_latest_cpb_trial_summary.csv
      - early7_latest_cpb_scene_summary.csv
      - early7_latest_cpb_report.md

Usage:
    cd /home/admin1/macvo-dev
    conda activate macvo
    python Scripts/analyse_v3bpp_latest_cpb_fdonly_early7.py
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np

WORKDIR = Path("/home/admin1/macvo-dev")
SOURCE_NAME = "latest_cpb_early7"
METHOD_NAME = "CP-B-FD-only"

BATCH_ROOT = Path("/mnt/e/文档/holoocean/code/recordings/batch_20260515_155653")
RESULT_ROOT = WORKDIR / "Results" / "v3bpp_latest_cpb_fdonly_early7_3x"
OUTDIR = WORKDIR / "analysis_v3bpp_latest_cpb_fdonly_early7_report"
REFERENCE_LONG = WORKDIR / "analysis_v3bpp_10scene_full_accuracy_long.csv"

SCENES = [
    "turbid_harbor",
    "clear_shallow",
    "deep_dark",
    "caustic_shallow",
    "dam_inspection",
    "murky_coast",
    "open_water",
]
TRIALS = 3
FIXED_METHODS = ("pure_macvo", "rotation_only", "translation_only", "full_imu")


def as_float(value) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def format_median_range(median: float, min_value: float, max_value: float) -> str:
    return f"{median:.2f} [{min_value:.2f}-{max_value:.2f}]"


def evaluate_direct_ate(poses_path: Path, ref_path: Path) -> float:
    est = np.genfromtxt(poses_path, delimiter=",", dtype=float, skip_header=1)
    gt = np.genfromtxt(ref_path, delimiter=",", dtype=float, skip_header=1)
    if est.ndim == 1:
        est = est.reshape(1, -1)
    if gt.ndim == 1:
        gt = gt.reshape(1, -1)
    n = min(len(est), len(gt))
    if n <= 0:
        return float("nan")
    delta = est[:n, 1:4] - gt[:n, 1:4]
    return float(np.sqrt(np.mean(np.sum(delta * delta, axis=1))))


def summarize_scene(
    *,
    scene: str,
    cpb_ates: list[float],
    pure_median: float,
    fixed_best_method: str,
    fixed_best_median: float,
) -> dict:
    vals = [value for value in cpb_ates if not math.isnan(value)]
    arr = np.array(vals, dtype=float)
    if len(arr) == 0:
        median = mean = std = min_value = max_value = cv = float("nan")
    else:
        median = float(np.median(arr))
        mean = float(np.mean(arr))
        std = float(np.std(arr))
        min_value = float(np.min(arr))
        max_value = float(np.max(arr))
        cv = std / mean if mean > 0 else float("nan")

    delta_vs_pure = median - pure_median if not math.isnan(pure_median) else float("nan")
    relative_gain = (
        (pure_median - median) / pure_median * 100.0
        if pure_median and not math.isnan(pure_median) and not math.isnan(median)
        else float("nan")
    )
    gap_to_fixed = (
        median / fixed_best_median
        if fixed_best_median and not math.isnan(fixed_best_median) and not math.isnan(median)
        else float("nan")
    )
    return {
        "source": SOURCE_NAME,
        "scene": scene,
        "method": METHOD_NAME,
        "n_trials": len(vals),
        "trial_1": cpb_ates[0] if len(cpb_ates) > 0 else float("nan"),
        "trial_2": cpb_ates[1] if len(cpb_ates) > 1 else float("nan"),
        "trial_3": cpb_ates[2] if len(cpb_ates) > 2 else float("nan"),
        "median_ATE": median,
        "mean_ATE": mean,
        "std_ATE": std,
        "cv": cv,
        "min_ATE": min_value,
        "max_ATE": max_value,
        "median_range": format_median_range(median, min_value, max_value)
        if not math.isnan(median)
        else "",
        "pure_median_ATE": pure_median,
        "delta_vs_pure": delta_vs_pure,
        "relative_gain_vs_pure_pct": relative_gain,
        "fixed_best_method": fixed_best_method,
        "fixed_best_median_ATE": fixed_best_median,
        "gap_to_fixed_best_x": gap_to_fixed,
    }


def load_reference_medians(path: Path) -> dict[str, dict]:
    refs: dict[str, dict] = {
        scene: {
            "pure_median": float("nan"),
            "fixed_best_method": "",
            "fixed_best_median": float("nan"),
        }
        for scene in SCENES
    }
    if not path.exists():
        return refs

    fixed_by_scene: dict[str, list[tuple[str, float]]] = {scene: [] for scene in SCENES}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            scene = row.get("scene", "")
            method = row.get("method", "")
            if scene not in refs or method not in FIXED_METHODS:
                continue
            median = as_float(row.get("median_ATE"))
            if method == "pure_macvo":
                refs[scene]["pure_median"] = median
            fixed_by_scene[scene].append((method, median))

    for scene, items in fixed_by_scene.items():
        valid = [(method, median) for method, median in items if not math.isnan(median)]
        if not valid:
            continue
        best_method, best_median = min(valid, key=lambda item: item[1])
        refs[scene]["fixed_best_method"] = best_method
        refs[scene]["fixed_best_median"] = best_median
    return refs


def read_adaptive_stats(result_dir: Path) -> dict:
    path = result_dir / "adaptive_decisions.csv"
    stats = {
        "total_pairs": 0,
        "full_imu_frames": 0,
        "full_imu_pct": float("nan"),
        "cooldown_frames": 0,
        "severe_vc_triggers": 0,
        "mild_vc_triggers": 0,
        "fd_triggers": 0,
    }
    if not path.exists():
        return stats

    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    rows = [row for row in rows if int(float(row.get("pair_id", "-1") or -1)) > 0]
    stats["total_pairs"] = len(rows)
    for row in rows:
        state = row.get("state_name", "")
        if "full_imu" in state:
            stats["full_imu_frames"] += 1
        if "cooldown" in state:
            stats["cooldown_frames"] += 1
        if row.get("severe_vc_triggered", "0") == "1":
            stats["severe_vc_triggers"] += 1
        if row.get("mild_vc_triggered", "0") == "1":
            stats["mild_vc_triggers"] += 1
        if row.get("full_divergence_triggered", "0") == "1":
            stats["fd_triggers"] += 1
    if stats["total_pairs"] > 0:
        stats["full_imu_pct"] = stats["full_imu_frames"] / stats["total_pairs"] * 100.0
    return stats


def collect_trials(result_root: Path, batch_root: Path) -> tuple[list[dict], list[str]]:
    rows = []
    missing = []
    for trial in range(1, TRIALS + 1):
        for scene in SCENES:
            result_dir = result_root / f"trial_{trial}" / scene
            poses_path = result_dir / "poses.csv"
            ref_path = batch_root / scene / "ref_pose.csv"
            if not poses_path.exists():
                nested = sorted(result_dir.rglob("poses.csv")) if result_dir.exists() else []
                poses_path = nested[0] if nested else poses_path
            if not poses_path.exists():
                missing.append(f"trial={trial} scene={scene} poses.csv")
                ate = float("nan")
            elif not ref_path.exists():
                missing.append(f"trial={trial} scene={scene} ref_pose.csv")
                ate = float("nan")
            else:
                ate = evaluate_direct_ate(poses_path, ref_path)
            stats = read_adaptive_stats(poses_path.parent if poses_path.exists() else result_dir)
            rows.append(
                {
                    "source": SOURCE_NAME,
                    "method": METHOD_NAME,
                    "trial": trial,
                    "scene": scene,
                    "ATE": ate,
                    "result_dir": str(result_dir),
                    "poses_path": str(poses_path) if poses_path.exists() else "",
                    **stats,
                }
            )
    return rows, missing


def write_trial_summary(outdir: Path, rows: list[dict]) -> None:
    cols = [
        "source",
        "method",
        "trial",
        "scene",
        "ATE",
        "full_imu_pct",
        "full_imu_frames",
        "cooldown_frames",
        "severe_vc_triggers",
        "mild_vc_triggers",
        "fd_triggers",
        "result_dir",
        "poses_path",
    ]
    with (outdir / "early7_latest_cpb_trial_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for row in rows:
            serial = row.copy()
            serial["ATE"] = f"{serial['ATE']:.4f}" if not math.isnan(serial["ATE"]) else ""
            serial["full_imu_pct"] = (
                f"{serial['full_imu_pct']:.2f}" if not math.isnan(serial["full_imu_pct"]) else ""
            )
            writer.writerow({key: serial.get(key, "") for key in cols})


def write_scene_summary(outdir: Path, summaries: list[dict]) -> None:
    cols = [
        "source",
        "scene",
        "method",
        "n_trials",
        "trial_1",
        "trial_2",
        "trial_3",
        "median_ATE",
        "min_ATE",
        "max_ATE",
        "mean_ATE",
        "std_ATE",
        "cv",
        "median_range",
        "pure_median_ATE",
        "delta_vs_pure",
        "relative_gain_vs_pure_pct",
        "fixed_best_method",
        "fixed_best_median_ATE",
        "gap_to_fixed_best_x",
    ]
    with (outdir / "early7_latest_cpb_scene_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for row in summaries:
            serial = row.copy()
            for key in [
                "trial_1",
                "trial_2",
                "trial_3",
                "median_ATE",
                "min_ATE",
                "max_ATE",
                "mean_ATE",
                "std_ATE",
                "cv",
                "pure_median_ATE",
                "delta_vs_pure",
                "relative_gain_vs_pure_pct",
                "fixed_best_median_ATE",
                "gap_to_fixed_best_x",
            ]:
                serial[key] = f"{serial[key]:.4f}" if not math.isnan(serial[key]) else ""
            writer.writerow({key: serial.get(key, "") for key in cols})


def write_report(outdir: Path, summaries: list[dict], missing: list[str]) -> None:
    with (outdir / "early7_latest_cpb_report.md").open("w", encoding="utf-8") as f:
        f.write("# Latest CP-B-FD-only Early-7 Rerun Report\n\n")
        f.write(f"Source name: `{SOURCE_NAME}`  \n")
        f.write(f"Method: `{METHOD_NAME}`  \n")
        f.write("Each result is reported as `median [min-max]` in meters.\n\n")

        f.write("## Scene Summary\n\n")
        f.write(
            "| Scene | CP-B latest | Pure MACVO | Delta vs pure | Relative gain | Fixed best | Gap to fixed best |\n"
        )
        f.write("|---|---:|---:|---:|---:|---|---:|\n")
        for row in summaries:
            pure = row["pure_median_ATE"]
            pure_cell = f"{pure:.2f}" if not math.isnan(pure) else ""
            delta = row["delta_vs_pure"]
            delta_cell = f"{delta:+.2f}" if not math.isnan(delta) else ""
            gain = row["relative_gain_vs_pure_pct"]
            gain_cell = f"{gain:+.1f}%" if not math.isnan(gain) else ""
            fixed = (
                f"{row['fixed_best_method']} {row['fixed_best_median_ATE']:.2f}"
                if row["fixed_best_method"] and not math.isnan(row["fixed_best_median_ATE"])
                else ""
            )
            gap = row["gap_to_fixed_best_x"]
            gap_cell = f"{gap:.2f}x" if not math.isnan(gap) else ""
            f.write(
                f"| `{row['scene']}` | {row['median_range']} | {pure_cell} | "
                f"{delta_cell} | {gain_cell} | {fixed} | {gap_cell} |\n"
            )

        f.write("\n## Completeness\n\n")
        if missing:
            f.write(f"- Missing or unevaluable files: {len(missing)}\n")
            for item in missing:
                f.write(f"  - {item}\n")
        else:
            f.write("- All 7 scenes have 3 evaluable trials.\n")

        f.write("\n## Paper Use\n\n")
        f.write("- Use this source as the paper-facing early-scene CP-B-FD-only rerun.\n")
        f.write("- Keep older `CP-B C7` rows as historical or reproducibility checks, not as the final main-table source.\n")
        f.write("- If this rerun differs from the old C7 batch, describe the difference as batch-level reproducibility evidence.\n")
        f.write("- Do not drop `open_water`; if it remains weak, move it to supplementary/stress-test discussion.\n")


def build_summaries(trial_rows: list[dict], refs: dict[str, dict]) -> list[dict]:
    summaries = []
    for scene in SCENES:
        ates = [row["ATE"] for row in trial_rows if row["scene"] == scene]
        ref = refs.get(scene, {})
        summaries.append(
            summarize_scene(
                scene=scene,
                cpb_ates=ates,
                pure_median=ref.get("pure_median", float("nan")),
                fixed_best_method=ref.get("fixed_best_method", ""),
                fixed_best_median=ref.get("fixed_best_median", float("nan")),
            )
        )
    return summaries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze latest CP-B-FD-only early-7 rerun.")
    parser.add_argument("--result-root", type=Path, default=RESULT_ROOT)
    parser.add_argument("--batch-root", type=Path, default=BATCH_ROOT)
    parser.add_argument("--outdir", type=Path, default=OUTDIR)
    parser.add_argument("--reference-long", type=Path, default=REFERENCE_LONG)
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    trial_rows, missing = collect_trials(args.result_root, args.batch_root)
    refs = load_reference_medians(args.reference_long)
    summaries = build_summaries(trial_rows, refs)

    write_trial_summary(args.outdir, trial_rows)
    write_scene_summary(args.outdir, summaries)
    write_report(args.outdir, summaries, missing)

    print(f"Wrote {args.outdir / 'early7_latest_cpb_trial_summary.csv'}")
    print(f"Wrote {args.outdir / 'early7_latest_cpb_scene_summary.csv'}")
    print(f"Wrote {args.outdir / 'early7_latest_cpb_report.md'}")
    if missing:
        print(f"Missing files: {len(missing)}")
        for item in missing:
            print(f"  - {item}")
        if not args.allow_incomplete:
            print("Use --allow-incomplete to keep partial outputs without a failing exit code.")
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
