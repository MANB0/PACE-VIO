#!/usr/bin/env python3
"""
Build a full 7+3 scene accuracy table from existing analysis outputs.

The 7+3 set excludes validation scenes and locked held-out scenes:
  - original 7 scenes
  - 3 development stress scenes
"""

from __future__ import annotations

import csv
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

WORKDIR = Path("/home/admin1/macvo-dev")

SCENES_7 = [
    "turbid_harbor",
    "clear_shallow",
    "deep_dark",
    "caustic_shallow",
    "dam_inspection",
    "murky_coast",
    "open_water",
]

SCENES_3 = [
    "moderate_turbidity",
    "open_water_overcast",
    "twilight_coast",
]

SCENES_10 = SCENES_7 + SCENES_3

METHOD_ORDER = [
    "pure_macvo",
    "rotation_only",
    "translation_only",
    "full_imu",
    "original_v3b",
    "v3bplus_d4",
    "ruleB",
    "FD-E only",
    "FD-E+CP-B",
    "CP-B-FD-only (candidate7)",
    "CP-B-FD-only (phase1b)",
]

OUT_LONG = WORKDIR / "analysis_v3bpp_10scene_full_accuracy_long.csv"
OUT_WIDE = WORKDIR / "analysis_v3bpp_10scene_full_accuracy_wide.csv"
OUT_MD = WORKDIR / "analysis_v3bpp_10scene_full_accuracy_report.md"


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def as_float(value) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def stats_from_values(values: list[float]) -> dict:
    vals = [v for v in values if not math.isnan(v)]
    if not vals:
        return {
            "n_trials": 0,
            "median_ATE": float("nan"),
            "mean_ATE": float("nan"),
            "std_ATE": float("nan"),
            "min_ATE": float("nan"),
            "max_ATE": float("nan"),
            "cv": float("nan"),
            "trial_ATEs": "",
        }
    arr = np.array(vals, dtype=float)
    mean = float(np.mean(arr))
    std = float(np.std(arr))
    return {
        "n_trials": len(vals),
        "median_ATE": float(np.median(arr)),
        "mean_ATE": mean,
        "std_ATE": std,
        "min_ATE": float(np.min(arr)),
        "max_ATE": float(np.max(arr)),
        "cv": std / mean if mean > 0 else float("nan"),
        "trial_ATEs": ";".join(f"{v:.4f}" for v in vals),
    }


def add_entry(rows: list[dict], source: str, scene: str, method: str, values: list[float], notes: str = ""):
    if scene not in SCENES_10:
        return
    stats = stats_from_values(values)
    rows.append(
        {
            "scene": scene,
            "method": method,
            "source": source,
            **stats,
            "notes": notes,
        }
    )


def add_grouped_entries(
    rows: list[dict],
    source: str,
    csv_path: Path,
    scene_col: str,
    method: str | None,
    value_col: str,
    method_col: str | None = None,
    notes: str = "",
) -> None:
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in read_csv(csv_path):
        scene = row[scene_col]
        method_name = row[method_col] if method_col else method
        if method_name is None:
            raise ValueError(f"method is required for {csv_path}")
        grouped[(scene, method_name)].append(as_float(row[value_col]))

    for (scene, method_name), values in grouped.items():
        add_entry(rows, source, scene, method_name, values, notes=notes)


def build_rows() -> list[dict]:
    rows: list[dict] = []

    # 7-scene fixed baselines: use true per-trial ATE values.
    add_grouped_entries(
        rows,
        "baseline_7x4_3runs",
        WORKDIR / "Results/baseline_7x4_3runs_20260518_234150/all_runs_ate.csv",
        scene_col="scene",
        method=None,
        method_col="method",
        value_col="direct_ATE",
        notes="actual per-trial direct ATE",
    )

    # 3-scene fixed baselines + Rule B.
    new3 = read_csv(WORKDIR / "analysis_ruleB_new_scene_stress_test/new_scene_ate_summary.csv")
    for row in new3:
        add_entry(
            rows,
            "new_scene_stress_test",
            row["scene"],
            row["method"],
            [as_float(row["trial_1_ATE"]), as_float(row["trial_2_ATE"]), as_float(row["trial_3_ATE"])],
            notes=row.get("notes", ""),
        )

    # 7-scene adaptive variants: use true per-trial ATE values.
    add_grouped_entries(
        rows,
        "v3b_7x3",
        WORKDIR / "analysis_v3b_7x3_report/v3b_7x3_all_runs_ate.csv",
        scene_col="scene",
        method="original_v3b",
        value_col="v3b_direct_ATE",
        notes="actual per-trial direct ATE",
    )
    add_grouped_entries(
        rows,
        "v3bplus_d4_7x3",
        WORKDIR / "analysis_v3bplus_d4_7x3_report/v3bplus_d4_7x3_all_runs_ate.csv",
        scene_col="scene",
        method="v3bplus_d4",
        value_col="v3bplus_d4_direct_ATE",
        notes="actual per-trial direct ATE",
    )
    add_grouped_entries(
        rows,
        "ruleB_7x3",
        WORKDIR / "analysis_v3bplus_ruleB_7x3_report/v3bplus_ruleB_7x3_all_runs_ate.csv",
        scene_col="scene",
        method="ruleB",
        value_col="v3bplus_ruleB_direct_ATE",
        notes="actual per-trial direct ATE",
    )

    # FD-E only.
    fde = read_csv(WORKDIR / "analysis_v3bpp_phase1a_fd_e_grace30_5scene_3x_report/phase1a_fd_e_summary.csv")
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in fde:
        grouped[row["scene"]].append(as_float(row["ATE"]))
    for scene, values in grouped.items():
        add_entry(rows, "phase1a_fd_e", scene, "FD-E only", values)

    # CP-B candidate 7-scene.
    candidate = read_csv(WORKDIR / "analysis_v3bpp_candidate_cpb_fdonly_7scene_report/candidate_trial_summary.csv")
    grouped = defaultdict(list)
    for row in candidate:
        grouped[row["scene"]].append(as_float(row["ATE"]))
    for scene, values in grouped.items():
        add_entry(rows, "candidate7_cpb_fdonly", scene, "CP-B-FD-only (candidate7)", values)

    # CP-B phase1b and FD-E+CP-B.
    phase1b = read_csv(WORKDIR / "analysis_v3bpp_phase1b_cpb_report/phase1b_trial_summary.csv")
    grouped2: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in phase1b:
        grouped2[(row["scene"], row["exp"])].append(as_float(row["ATE"]))
    for (scene, exp), values in grouped2.items():
        method = "CP-B-FD-only (phase1b)" if exp == "CP-B-FD-only" else exp
        add_entry(rows, "phase1b_cpb", scene, method, values)

    return sorted(
        rows,
        key=lambda r: (
            SCENES_10.index(r["scene"]) if r["scene"] in SCENES_10 else 999,
            METHOD_ORDER.index(r["method"]) if r["method"] in METHOD_ORDER else 999,
            r["source"],
        ),
    )


def fmt_float(value: float) -> str:
    if value is None or math.isnan(value):
        return ""
    return f"{value:.4f}"


def write_long(rows: list[dict]) -> None:
    cols = [
        "scene",
        "method",
        "source",
        "n_trials",
        "median_ATE",
        "mean_ATE",
        "std_ATE",
        "min_ATE",
        "max_ATE",
        "cv",
        "trial_ATEs",
        "notes",
    ]
    with OUT_LONG.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for row in rows:
            out = row.copy()
            for key in ["median_ATE", "mean_ATE", "std_ATE", "min_ATE", "max_ATE", "cv"]:
                out[key] = fmt_float(out[key])
            writer.writerow(out)


def write_wide(rows: list[dict]) -> None:
    best_by_scene_method = {}
    for row in rows:
        key = (row["scene"], row["method"])
        # Prefer candidate7 for original 7 CP-B and phase1b for development 3, but keep both as separate methods.
        best_by_scene_method[key] = row

    cols = ["scene"] + METHOD_ORDER + ["oracle_mode", "oracle_ATE", "best_method", "best_ATE"]
    with OUT_WIDE.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for scene in SCENES_10:
            out = {"scene": scene}
            values = {}
            for method in METHOD_ORDER:
                row = best_by_scene_method.get((scene, method))
                if row:
                    out[method] = fmt_float(row["median_ATE"])
                    values[method] = row["median_ATE"]
                else:
                    out[method] = ""
            fixed_values = {
                m: v for m, v in values.items()
                if m in ("pure_macvo", "rotation_only", "translation_only", "full_imu")
            }
            if fixed_values:
                oracle_mode, oracle_ate = min(fixed_values.items(), key=lambda kv: kv[1])
                out["oracle_mode"] = oracle_mode
                out["oracle_ATE"] = fmt_float(oracle_ate)
            ranked_values = {m: v for m, v in values.items() if not math.isnan(v)}
            if ranked_values:
                best_method, best_ate = min(ranked_values.items(), key=lambda kv: kv[1])
                out["best_method"] = best_method
                out["best_ATE"] = fmt_float(best_ate)
            writer.writerow(out)


def write_report(rows: list[dict]) -> None:
    by_scene = defaultdict(list)
    for row in rows:
        by_scene[row["scene"]].append(row)
    by_key = {(row["scene"], row["method"]): row for row in rows}
    coverage = Counter(row["method"] for row in rows)
    incomplete = [row for row in rows if row["n_trials"] != 3]
    high_cv = sorted(
        [row for row in rows if not math.isnan(row["cv"]) and row["cv"] >= 0.15],
        key=lambda row: row["cv"],
        reverse=True,
    )

    with OUT_MD.open("w") as f:
        f.write("# V3b++ 7+3 Scene Full Accuracy Summary\n\n")
        f.write("Scope: original 7 scenes + 3 development stress scenes. Validation and locked held-out scenes are excluded.\n\n")
        f.write("## Median ATE Table\n\n")
        header = ["Scene"] + METHOD_ORDER
        f.write("| " + " | ".join(header) + " |\n")
        f.write("|" + "|".join(["---"] + ["---:"] * len(METHOD_ORDER)) + "|\n")
        for scene in SCENES_10:
            row_items = {row["method"]: row for row in by_scene[scene]}
            vals = [scene]
            for method in METHOD_ORDER:
                item = row_items.get(method)
                vals.append(f"{item['median_ATE']:.2f}" if item else "")
            f.write("| " + " | ".join(vals) + " |\n")

        f.write("\n## Notes\n\n")
        f.write("- Values are direct ATE medians in meters unless otherwise noted.\n")
        f.write("- Long CSV includes mean/std/min/max/CV/trial ATEs for every available method.\n")
        f.write("- `CP-B-FD-only (candidate7)` and `CP-B-FD-only (phase1b)` are kept separate because open_water is bimodal across batches.\n")
        f.write("- Blank cells mean that method was not run for that scene in the available experiment set.\n")

        f.write("\n## Completeness Check\n\n")
        f.write(f"- Scene-method rows: {len(rows)}.\n")
        f.write("- Per-row trial count: all rows have 3 trials.\n" if not incomplete else f"- Rows with missing trials: {len(incomplete)}.\n")
        f.write("- Method coverage by scene count:\n")
        for method in METHOD_ORDER:
            if coverage[method]:
                f.write(f"  - `{method}`: {coverage[method]}/10 scenes\n")

        f.write("\n## Paper Baseline Framing\n\n")
        f.write("- The paper-facing baseline is `pure_macvo`, because the contribution is adaptive IMU mode selection added on top of MACVO.\n")
        f.write("- `Rule B` is not an external or primary baseline. It is an in-family predecessor used for ablation.\n")
        f.write("- `rotation_only`, `translation_only`, and `full_imu` are fixed-mode diagnostic controls used to identify oracle behavior.\n")
        f.write("- Therefore the main performance question is CP-B-FD-only vs `pure_macvo`; CP-B vs Rule B is an internal mechanism question.\n")

        f.write("\n## CP-B vs Pure MACVO\n\n")
        f.write("| Scene | Pure MACVO | CP-B source | CP-B | Delta CP-B-Pure | Relative gain | Interpretation |\n")
        f.write("|---|---:|---|---:|---:|---:|---|\n")
        for scene in SCENES_10:
            pure = by_key.get((scene, "pure_macvo"))
            cpb = by_key.get((scene, "CP-B-FD-only (phase1b)"))
            source = "phase1b"
            if cpb is None:
                cpb = by_key.get((scene, "CP-B-FD-only (candidate7)"))
                source = "candidate7"
            if pure is None or cpb is None:
                continue
            delta = cpb["median_ATE"] - pure["median_ATE"]
            rel_gain = (pure["median_ATE"] - cpb["median_ATE"]) / pure["median_ATE"] * 100.0
            if delta < -0.5:
                interp = "meaningful improvement"
            elif delta > 0.5:
                interp = "meaningful regression"
            else:
                interp = "near neutral"
            f.write(
                f"| {scene} | {pure['median_ATE']:.2f} | {source} | {cpb['median_ATE']:.2f} | "
                f"{delta:+.2f} | {rel_gain:+.1f}% | {interp} |\n"
            )

        f.write("\n## Internal Ablation: CP-B vs Rule B\n\n")
        f.write("| Scene | Rule B | CP-B source | CP-B | Delta CP-B-RuleB | Relative gain |\n")
        f.write("|---|---:|---|---:|---:|---:|\n")
        for scene in SCENES_10:
            ruleb = by_key.get((scene, "ruleB"))
            cpb = by_key.get((scene, "CP-B-FD-only (phase1b)"))
            source = "phase1b"
            if cpb is None:
                cpb = by_key.get((scene, "CP-B-FD-only (candidate7)"))
                source = "candidate7"
            if ruleb is None or cpb is None:
                continue
            delta = cpb["median_ATE"] - ruleb["median_ATE"]
            rel_gain = (ruleb["median_ATE"] - cpb["median_ATE"]) / ruleb["median_ATE"] * 100.0
            f.write(
                f"| {scene} | {ruleb['median_ATE']:.2f} | {source} | "
                f"{cpb['median_ATE']:.2f} | {delta:+.2f} | {rel_gain:+.1f}% |\n"
            )

        f.write("\n## High Randomness Rows\n\n")
        f.write("Rows with CV >= 0.15, sorted by CV.\n\n")
        f.write("| Scene | Method | Median | CV | Trial ATEs |\n")
        f.write("|---|---|---:|---:|---|\n")
        for row in high_cv:
            f.write(
                f"| {row['scene']} | {row['method']} | {row['median_ATE']:.2f} | "
                f"{row['cv']:.3f} | {row['trial_ATEs']} |\n"
            )

        f.write("\n## Key Readout\n\n")
        f.write("- CP-B-FD-only has a clear median gain over Rule B only on moderate_turbidity when using the phase1b run set.\n")
        f.write("- open_water CP-B should be described as inconsistent/bimodal, not a clean success.\n")
        f.write("- open_water_overcast is a safety case: adaptive methods stay near pure/rotation performance.\n")
        f.write("- twilight_coast remains a candidate-set limitation: translation_only is the fixed oracle but is not in the adaptive set.\n")


def main() -> None:
    rows = build_rows()
    write_long(rows)
    write_wide(rows)
    write_report(rows)
    print(f"Wrote {OUT_LONG}")
    print(f"Wrote {OUT_WIDE}")
    print(f"Wrote {OUT_MD}")
    print(f"Rows: {len(rows)}")


if __name__ == "__main__":
    main()
