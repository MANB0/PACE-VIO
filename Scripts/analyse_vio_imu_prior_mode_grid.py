#!/usr/bin/env python3
"""
Analyze the VIO/IMU translation-prior diagnostic grid.

Expected result layout:
    Results/vio_imu_prior_mode_grid/trial_1/<variant>/<scene>/poses.csv

Outputs:
    analysis_vio_imu_prior_mode_grid/
      - imu_prior_mode_trial_summary.csv
      - imu_prior_mode_scene_variant_summary.csv
      - imu_prior_mode_report.md
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

WORKDIR = Path("/home/admin1/macvo-dev")
RESULT_ROOT = WORKDIR / "Results" / "vio_imu_prior_mode_grid"
OUTDIR = WORKDIR / "analysis_vio_imu_prior_mode_grid"

SCENE_ROOTS = {
    "turbid_harbor": Path("/mnt/e/文档/holoocean/code/recordings/batch_20260515_155653/turbid_harbor"),
    "clear_shallow": Path("/mnt/e/文档/holoocean/code/recordings/batch_20260515_155653/clear_shallow"),
    "deep_dark": Path("/mnt/e/文档/holoocean/code/recordings/batch_20260515_155653/deep_dark"),
    "caustic_shallow": Path("/mnt/e/文档/holoocean/code/recordings/batch_20260515_155653/caustic_shallow"),
    "dam_inspection": Path("/mnt/e/文档/holoocean/code/recordings/batch_20260515_155653/dam_inspection"),
    "murky_coast": Path("/mnt/e/文档/holoocean/code/recordings/batch_20260515_155653/murky_coast"),
    "open_water_overcast": Path("/mnt/e/文档/holoocean/code/recordings/batch_20260528_203401/open_water_overcast"),
    "validation_transient_dropout": Path("/mnt/e/文档/holoocean/code/recordings/batch_20260601_162707/validation_transient_dropout"),
    "validation_twilight_structure": Path("/mnt/e/文档/holoocean/code/recordings/batch_20260601_162707/validation_twilight_structure"),
    "locked_clear_imu_harm": Path("/mnt/e/文档/holoocean/code/recordings/batch_20260618_110853/locked_clear_imu_harm"),
    "locked_murky_entry_help": Path("/mnt/e/文档/holoocean/code/recordings/batch_20260618_110853/locked_murky_entry_help"),
    "locked_quality_degrade_no_dropout": Path("/mnt/e/文档/holoocean/code/recordings/batch_20260618_110853/locked_quality_degrade_no_dropout"),
}
SMOKE3_SCENES = ["clear_shallow", "murky_coast", "locked_murky_entry_help"]
KEPT12_SCENES = [
    "turbid_harbor",
    "clear_shallow",
    "deep_dark",
    "caustic_shallow",
    "dam_inspection",
    "murky_coast",
    "open_water_overcast",
    "validation_transient_dropout",
    "validation_twilight_structure",
    "locked_clear_imu_harm",
    "locked_murky_entry_help",
    "locked_quality_degrade_no_dropout",
]
SCENE_PRESETS = {
    "smoke3": SMOKE3_SCENES,
    "kept12": KEPT12_SCENES,
}
DEFAULT_SCENES = SMOKE3_SCENES
DEFAULT_VARIANTS = [
    "pure_macvo",
    "rotation_only",
    "aimvo_damping_s005",
    "aimvo_damping_s1",
    "aimvo_visualvel_s1",
    "aimvo_imuvel_s1",
]
MAIN3_VARIANTS = [
    "pure_macvo",
    "rotation_only",
    "aimvo_damping_s005",
]
ROTFLOOR_VARIANTS = [
    "pure_macvo",
    "rotation_only",
    "rotation_only_rotstd02",
    "rotation_only_rotstd03",
    "aimvo_damping_s005",
    "aimvo_damping_s005_rotstd02",
    "aimvo_damping_s005_rotstd03",
    "aimvo_damping_s005_fullrotstd02",
    "aimvo_damping_s005_fullrotstd03",
]
VARIANT_PRESETS = {
    "default": DEFAULT_VARIANTS,
    "main3": MAIN3_VARIANTS,
    "rotfloor": ROTFLOOR_VARIANTS,
}


def as_float(value) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def boolish(value) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def find_file(root: Path, name: str) -> Path | None:
    direct = root / name
    if direct.exists():
        return direct
    matches = sorted(root.rglob(name)) if root.exists() else []
    return matches[0] if matches else None


def evaluate_direct_ate(poses_path: Path, ref_path: Path) -> tuple[float, int]:
    est = np.genfromtxt(poses_path, delimiter=",", dtype=float, skip_header=1)
    gt = np.genfromtxt(ref_path, delimiter=",", dtype=float, skip_header=1)
    if est.ndim == 1:
        est = est.reshape(1, -1)
    if gt.ndim == 1:
        gt = gt.reshape(1, -1)
    n = min(len(est), len(gt))
    if n <= 0:
        return float("nan"), 0
    delta = est[:n, 1:4] - gt[:n, 1:4]
    return float(np.sqrt(np.mean(np.sum(delta * delta, axis=1)))), n


def summarize_frame_pair_diagnostics(result_dir: Path) -> dict:
    path = find_file(result_dir, "frame_pair_diagnostics.csv")
    out = {
        "diag_rows": 0,
        "diag_mode_top": "",
        "adaptive_translation_pct": float("nan"),
        "median_prior_std_m": float("nan"),
        "median_r_p_whitened": float("nan"),
        "median_imu_trans_loss": float("nan"),
    }
    if path is None:
        return out

    rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
    out["diag_rows"] = len(rows)
    if not rows:
        return out

    modes = Counter(row.get("imu_trans_prior_mode", "") for row in rows if row.get("imu_trans_prior_mode", ""))
    if modes:
        out["diag_mode_top"] = modes.most_common(1)[0][0]

    trans_flags = [boolish(row.get("adaptive_use_translation", "")) for row in rows]
    if trans_flags:
        out["adaptive_translation_pct"] = sum(trans_flags) / len(trans_flags) * 100.0

    prior_stds = []
    rps = []
    losses = []
    for row in rows:
        std_vals = [
            as_float(row.get("imu_trans_prior_std_x")),
            as_float(row.get("imu_trans_prior_std_y")),
            as_float(row.get("imu_trans_prior_std_z")),
        ]
        valid_std = [value for value in std_vals if not math.isnan(value)]
        if valid_std:
            prior_stds.append(float(np.mean(valid_std)))
        r_p = as_float(row.get("r_p_whitened_norm"))
        if not math.isnan(r_p):
            rps.append(r_p)
        loss = as_float(row.get("imu_trans_loss"))
        if not math.isnan(loss):
            losses.append(loss)

    if prior_stds:
        out["median_prior_std_m"] = float(np.median(prior_stds))
    if rps:
        out["median_r_p_whitened"] = float(np.median(rps))
    if losses:
        out["median_imu_trans_loss"] = float(np.median(losses))
    return out


def summarize_adaptive_decisions(result_dir: Path) -> dict:
    path = find_file(result_dir, "adaptive_decisions.csv")
    out = {
        "adaptive_rows": 0,
        "full_imu_pct": float("nan"),
        "pure_macvo_pct": float("nan"),
        "rotation_only_pct": float("nan"),
        "fd_triggers": 0,
        "severe_vc_triggers": 0,
        "mild_vc_triggers": 0,
    }
    if path is None:
        return out

    rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
    rows = [row for row in rows if int(float(row.get("pair_id", "-1") or -1)) > 0]
    out["adaptive_rows"] = len(rows)
    if not rows:
        return out

    mode_counts = Counter(
        row.get("adaptive_mode") or row.get("mode") or row.get("state_name", "")
        for row in rows
    )
    total = len(rows)
    out["full_imu_pct"] = sum(count for mode, count in mode_counts.items() if "full_imu" in mode) / total * 100.0
    out["pure_macvo_pct"] = sum(count for mode, count in mode_counts.items() if "pure_macvo" in mode) / total * 100.0
    out["rotation_only_pct"] = sum(count for mode, count in mode_counts.items() if "rotation_only" in mode) / total * 100.0
    out["fd_triggers"] = sum(1 for row in rows if row.get("full_divergence_triggered", "0") == "1")
    out["severe_vc_triggers"] = sum(1 for row in rows if row.get("severe_vc_triggered", "0") == "1")
    out["mild_vc_triggers"] = sum(1 for row in rows if row.get("mild_vc_triggered", "0") == "1")
    return out


def collect_trials(result_root: Path, scenes: list[str], variants: list[str], trials: int) -> tuple[list[dict], list[str]]:
    rows = []
    missing = []
    for trial in range(1, trials + 1):
        for variant in variants:
            for scene in scenes:
                result_dir = result_root / f"trial_{trial}" / variant / scene
                poses_path = find_file(result_dir, "poses.csv")
                ref_path = SCENE_ROOTS[scene] / "ref_pose.csv"
                if poses_path is None:
                    ate, n_frames = float("nan"), 0
                    missing.append(f"trial={trial} variant={variant} scene={scene} poses.csv")
                elif not ref_path.exists():
                    ate, n_frames = float("nan"), 0
                    missing.append(f"trial={trial} variant={variant} scene={scene} ref_pose.csv")
                else:
                    ate, n_frames = evaluate_direct_ate(poses_path, ref_path)

                rows.append(
                    {
                        "trial": trial,
                        "scene": scene,
                        "variant": variant,
                        "ATE": ate,
                        "n_frames": n_frames,
                        "result_dir": str(result_dir),
                        "poses_path": str(poses_path) if poses_path else "",
                        **summarize_adaptive_decisions(result_dir),
                        **summarize_frame_pair_diagnostics(result_dir),
                    }
                )
    return rows, missing


def summarize_scene_variant(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["scene"], row["variant"])].append(row)

    out = []
    for (scene, variant), items in sorted(grouped.items()):
        ates = np.array([row["ATE"] for row in items if not math.isnan(row["ATE"])], dtype=float)
        if len(ates):
            median = float(np.median(ates))
            min_value = float(np.min(ates))
            max_value = float(np.max(ates))
            std = float(np.std(ates))
        else:
            median = min_value = max_value = std = float("nan")
        out.append(
            {
                "scene": scene,
                "variant": variant,
                "n_trials": len(ates),
                "median_ATE": median,
                "min_ATE": min_value,
                "max_ATE": max_value,
                "std_ATE": std,
                "median_full_imu_pct": median_of(items, "full_imu_pct"),
                "median_prior_std_m": median_of(items, "median_prior_std_m"),
                "median_r_p_whitened": median_of(items, "median_r_p_whitened"),
                "median_imu_trans_loss": median_of(items, "median_imu_trans_loss"),
            }
        )
    return out


def summarize_best_by_scene(summary_rows: list[dict], variants: list[str]) -> list[dict]:
    grouped: dict[str, dict[str, dict]] = defaultdict(dict)
    for row in summary_rows:
        grouped[row["scene"]][row["variant"]] = row

    out = []
    for scene in sorted(grouped):
        scene_rows = grouped[scene]
        valid_rows = [
            row
            for variant, row in scene_rows.items()
            if variant in variants and not math.isnan(row["median_ATE"])
        ]
        if not valid_rows:
            continue
        best = min(valid_rows, key=lambda row: row["median_ATE"])
        pure = scene_rows.get("pure_macvo")
        rotation = scene_rows.get("rotation_only")
        damping = scene_rows.get("aimvo_damping_s005")

        pure_ate = pure["median_ATE"] if pure else float("nan")
        rotation_ate = rotation["median_ATE"] if rotation else float("nan")
        damping_ate = damping["median_ATE"] if damping else float("nan")
        out.append(
            {
                "scene": scene,
                "best_variant": best["variant"],
                "best_median_ATE": best["median_ATE"],
                "pure_median_ATE": pure_ate,
                "rotation_median_ATE": rotation_ate,
                "damping_median_ATE": damping_ate,
                "damping_minus_pure": safe_diff(damping_ate, pure_ate),
                "damping_vs_pure_pct": safe_pct(damping_ate, pure_ate),
                "damping_minus_rotation": safe_diff(damping_ate, rotation_ate),
                "damping_vs_rotation_pct": safe_pct(damping_ate, rotation_ate),
                "damping_full_imu_pct": damping["median_full_imu_pct"] if damping else float("nan"),
                "n_trials_best": best["n_trials"],
            }
        )
    return out


def safe_diff(value: float, baseline: float) -> float:
    if math.isnan(value) or math.isnan(baseline):
        return float("nan")
    return value - baseline


def safe_pct(value: float, baseline: float) -> float:
    if math.isnan(value) or math.isnan(baseline) or abs(baseline) < 1e-12:
        return float("nan")
    return (value - baseline) / baseline * 100.0


def median_of(rows: list[dict], key: str) -> float:
    vals = [as_float(row.get(key)) for row in rows]
    vals = [value for value in vals if not math.isnan(value)]
    return float(np.median(vals)) if vals else float("nan")


def fmt(value: float, digits: int = 3) -> str:
    return "" if math.isnan(value) else f"{value:.{digits}f}"


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_report(outdir: Path, summary_rows: list[dict], missing: list[str]) -> None:
    lines = [
        "# AIM-VO IMU Translation-Prior Diagnostic Grid",
        "",
        "This report is a code-side diagnostic artifact. It should not be used as a final paper table.",
        "",
        "## Scene-Variant Summary",
        "",
        "| scene | variant | n | median ATE | range | full IMU % | prior std m | r_p whitened | trans loss |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        range_text = ""
        if not math.isnan(row["min_ATE"]):
            range_text = f"{row['min_ATE']:.3f}-{row['max_ATE']:.3f}"
        lines.append(
            "| {scene} | {variant} | {n_trials} | {median} | {range_text} | {full_pct} | {prior_std} | {rp} | {loss} |".format(
                scene=row["scene"],
                variant=row["variant"],
                n_trials=row["n_trials"],
                median=fmt(row["median_ATE"]),
                range_text=range_text,
                full_pct=fmt(row["median_full_imu_pct"], 2),
                prior_std=fmt(row["median_prior_std_m"], 4),
                rp=fmt(row["median_r_p_whitened"], 3),
                loss=fmt(row["median_imu_trans_loss"], 3),
            )
        )

    lines += ["", "## Missing Inputs", ""]
    if missing:
        lines.extend(f"- {item}" for item in missing)
    else:
        lines.append("- none")
    lines.append("")
    (outdir / "imu_prior_mode_report.md").write_text("\n".join(lines), encoding="utf-8")


def write_best_report(outdir: Path, best_rows: list[dict], missing: list[str]) -> None:
    counts = Counter(row["best_variant"] for row in best_rows)
    min_trials = min((int(row["n_trials_best"]) for row in best_rows), default=0)
    lines = [
        "# AIM-VO IMU Prior Mode Best Summary",
        "",
        "This table summarizes the best median ATE per scene for the requested diagnostic variants.",
        "",
        f"Minimum trials represented by the winning entries: {min_trials}",
        "",
        "| scene | best | best median ATE | pure | rotation | damping | damping vs pure | damping vs rotation | damping full IMU % |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in best_rows:
        lines.append(
            "| {scene} | {best} | {best_ate} | {pure} | {rotation} | {damping} | {dvp} | {dvr} | {full} |".format(
                scene=row["scene"],
                best=row["best_variant"],
                best_ate=fmt(row["best_median_ATE"]),
                pure=fmt(row["pure_median_ATE"]),
                rotation=fmt(row["rotation_median_ATE"]),
                damping=fmt(row["damping_median_ATE"]),
                dvp=fmt(row["damping_vs_pure_pct"], 1) + "%" if not math.isnan(row["damping_vs_pure_pct"]) else "",
                dvr=fmt(row["damping_vs_rotation_pct"], 1) + "%" if not math.isnan(row["damping_vs_rotation_pct"]) else "",
                full=fmt(row["damping_full_imu_pct"], 2),
            )
        )

    lines += ["", "Best-variant counts:", ""]
    if counts:
        for variant, count in sorted(counts.items()):
            lines.append(f"- {variant}: {count}")
    else:
        lines.append("- none")

    lines += ["", "Missing Inputs", ""]
    if missing:
        lines.extend(f"- {item}" for item in missing)
    else:
        lines.append("- none")
    lines.append("")
    (outdir / "imu_prior_mode_best_summary.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze AIM-VO IMU prior mode diagnostic grid.")
    parser.add_argument("--result-root", type=Path, default=RESULT_ROOT)
    parser.add_argument("--outdir", type=Path, default=OUTDIR)
    parser.add_argument(
        "--preset",
        choices=sorted(SCENE_PRESETS),
        default="smoke3",
        help="Scene preset. Use kept12 for the retained paper-scene diagnostic set.",
    )
    parser.add_argument("--scenes", nargs="*", default=None, help="Override preset scenes explicitly.")
    parser.add_argument(
        "--variant-preset",
        choices=sorted(VARIANT_PRESETS),
        default="default",
        help="Variant preset. Use main3 for pure/rotation/damping repeated paper-facing diagnostics.",
    )
    parser.add_argument("--variants", nargs="*", default=None, help="Override variant preset explicitly.")
    parser.add_argument("--trials", type=int, default=1)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    scenes = list(args.scenes) if args.scenes is not None else list(SCENE_PRESETS[args.preset])
    variants = list(args.variants) if args.variants is not None else list(VARIANT_PRESETS[args.variant_preset])
    for scene in scenes:
        if scene not in SCENE_ROOTS:
            raise KeyError(f"Unknown scene {scene!r}. Known scenes: {', '.join(sorted(SCENE_ROOTS))}")

    args.outdir.mkdir(parents=True, exist_ok=True)
    trial_rows, missing = collect_trials(args.result_root, scenes, variants, int(args.trials))
    summary_rows = summarize_scene_variant(trial_rows)
    best_rows = summarize_best_by_scene(summary_rows, variants)

    write_csv(args.outdir / "imu_prior_mode_trial_summary.csv", trial_rows)
    write_csv(args.outdir / "imu_prior_mode_scene_variant_summary.csv", summary_rows)
    write_csv(args.outdir / "imu_prior_mode_best_summary.csv", best_rows)
    write_report(args.outdir, summary_rows, missing)
    write_best_report(args.outdir, best_rows, missing)

    print(f"Wrote: {args.outdir / 'imu_prior_mode_trial_summary.csv'}")
    print(f"Wrote: {args.outdir / 'imu_prior_mode_scene_variant_summary.csv'}")
    print(f"Wrote: {args.outdir / 'imu_prior_mode_best_summary.csv'}")
    print(f"Wrote: {args.outdir / 'imu_prior_mode_report.md'}")
    print(f"Wrote: {args.outdir / 'imu_prior_mode_best_summary.md'}")
    if missing:
        print(f"Missing entries: {len(missing)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
