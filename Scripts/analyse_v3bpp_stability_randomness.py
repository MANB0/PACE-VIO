#!/usr/bin/env python3
"""
Audit run-to-run stability for V3b++ paper-facing tables.

This script uses already-generated per-trial CSVs. It does not rerun MACVO.
"""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

WORKDIR = Path("/home/admin1/macvo-dev")
OUT_CSV = WORKDIR / "analysis_v3bpp_stability_randomness_audit.csv"
OUT_MD = WORKDIR / "analysis_v3bpp_stability_randomness_audit.md"
DISCUSSION_CSV = (
    WORKDIR
    / "analysis_v3bpp_discussion_package"
    / "table_6_all_16scene_all_method_variability.csv"
)
DISCUSSION_MATRIX = (
    WORKDIR
    / "analysis_v3bpp_discussion_package"
    / "table_6_all_16scene_all_method_variability_matrix.md"
)

DEV_LONG = WORKDIR / "analysis_v3bpp_10scene_full_accuracy_long.csv"
RAW_TRIAL_SOURCES = [
    {
        "source": "validation_3scene",
        "path": WORKDIR / "analysis_v3bpp_validation_3scene_report" / "validation_trial_summary.csv",
        "scene": "scene",
        "method": "method",
        "ate": "ATE",
    },
    {
        "source": "locked_3scene",
        "path": WORKDIR / "analysis_v3bpp_locked_3scene_report" / "locked_trial_summary.csv",
        "scene": "scene",
        "method": "method",
        "ate": "ATE",
    },
]

SCENES = [
    ("开发", "turbid_harbor"),
    ("开发", "clear_shallow"),
    ("开发", "deep_dark"),
    ("开发", "caustic_shallow"),
    ("开发", "dam_inspection"),
    ("开发", "murky_coast"),
    ("开发", "open_water"),
    ("开发", "moderate_turbidity"),
    ("开发", "open_water_overcast"),
    ("开发", "twilight_coast"),
    ("验证", "validation_moderate_harbor"),
    ("验证", "validation_transient_dropout"),
    ("验证", "validation_twilight_structure"),
    ("锁定", "locked_murky_entry_help"),
    ("锁定", "locked_clear_imu_harm"),
    ("锁定", "locked_quality_degrade_no_dropout"),
]


def _parse_float(value: str) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def _norm_method(method: str) -> str:
    if method in ("cpb_fd_only", "CP-B-FD-only"):
        return "CP-B-FD-only"
    if method == "FD-E+CP-B":
        return "FD-E+CP-B"
    if method == "FD-E only":
        return "FD-E only"
    if method == "ruleB":
        return "Rule B"
    return method


def load_trials() -> list[dict]:
    rows = []

    if DEV_LONG.exists():
        with DEV_LONG.open(newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                source = row.get("source", "development_10")
                scene = row.get("scene", "")
                method = _norm_method(row.get("method", ""))
                for trial_idx, value in enumerate(row.get("trial_ATEs", "").split(";"), start=1):
                    ate = _parse_float(value)
                    if math.isnan(ate):
                        continue
                    rows.append(
                        {
                            "source": source,
                            "scene": scene,
                            "method": method,
                            "trial": str(trial_idx),
                            "ate": ate,
                        }
                    )

    for spec in RAW_TRIAL_SOURCES:
        path = spec["path"]
        if not path.exists():
            print(f"[WARN] Missing source: {path}")
            continue
        with path.open(newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                ate = _parse_float(row.get(spec["ate"], ""))
                if math.isnan(ate):
                    continue
                method = row.get(spec.get("method", ""), "")
                rows.append(
                    {
                        "source": spec["source"],
                        "scene": row.get(spec["scene"], ""),
                        "method": _norm_method(method),
                        "trial": row.get("trial", row.get("trial_id", "")),
                        "ate": ate,
                    }
                )
    return rows


def classify_stability(ates: list[float]) -> tuple[str, str]:
    n = len(ates)
    if n < 3:
        return "INSUFFICIENT", "fewer than 3 trials"

    arr = np.array(ates, dtype=float)
    median = float(np.median(arr))
    mean = float(np.mean(arr))
    std = float(np.std(arr))
    min_v = float(np.min(arr))
    max_v = float(np.max(arr))
    spread = max_v - min_v
    cv = std / mean if mean > 0 else float("nan")
    ratio = max_v / min_v if min_v > 0 else float("inf")

    if cv >= 0.25 or ratio >= 1.5 or spread >= max(10.0, 0.25 * median):
        return "UNSTABLE", f"cv={cv:.3f}, range={spread:.2f}m, max/min={ratio:.2f}"
    if cv >= 0.10 or ratio >= 1.2 or spread >= max(3.0, 0.10 * median):
        return "MODERATE", f"cv={cv:.3f}, range={spread:.2f}m, max/min={ratio:.2f}"
    return "STABLE", f"cv={cv:.3f}, range={spread:.2f}m, max/min={ratio:.2f}"


def split_for_scene(scene: str) -> str:
    if scene.startswith("validation_"):
        return "validation_3"
    if scene.startswith("locked_"):
        return "locked_3"
    return "development_10"


def summarize_trials(rows: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["source"], row["scene"], row["method"])].append(row)

    out = []
    for (source, scene, method), items in sorted(grouped.items()):
        ates = [x["ate"] for x in items]
        arr = np.array(ates, dtype=float)
        flag, reason = classify_stability(ates)
        out.append(
            {
                "split": split_for_scene(scene),
                "source": source,
                "scene": scene,
                "method": method,
                "n": len(ates),
                "median_ATE": float(np.median(arr)),
                "mean_ATE": float(np.mean(arr)),
                "std_ATE": float(np.std(arr)),
                "cv": float(np.std(arr) / np.mean(arr)) if np.mean(arr) > 0 else float("nan"),
                "min_ATE": float(np.min(arr)),
                "max_ATE": float(np.max(arr)),
                "range_ATE": float(np.max(arr) - np.min(arr)),
                "stability": flag,
                "reason": reason,
                "trial_ATEs": ";".join(f"{x:.4f}" for x in ates),
            }
        )
    return out


def cross_source_repro(rows: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        method = row["method"]
        if method.startswith("CP-B-FD-only"):
            method = "CP-B-FD-only"
        grouped[(row["scene"], method, row["source"])].append(row["ate"])

    by_scene_method = defaultdict(list)
    for (scene, method, source), ates in grouped.items():
        by_scene_method[(scene, method)].append((source, float(np.median(ates))))

    out = []
    for (scene, method), medians in sorted(by_scene_method.items()):
        if len(medians) < 2:
            continue
        vals = [x[1] for x in medians]
        min_v = min(vals)
        max_v = max(vals)
        delta = max_v - min_v
        rel = delta / min_v if min_v > 0 else float("inf")
        if delta >= 10.0 or rel >= 0.05:
            out.append(
                {
                    "scene": scene,
                    "method": method,
                    "source_medians": ", ".join(f"{s}={v:.2f}" for s, v in medians),
                    "delta": delta,
                    "relative_delta": rel,
                }
            )
    return out


def write_csv(rows: list[dict]) -> None:
    cols = [
        "split",
        "source",
        "scene",
        "method",
        "n",
        "median_ATE",
        "mean_ATE",
        "std_ATE",
        "cv",
        "min_ATE",
        "max_ATE",
        "range_ATE",
        "stability",
        "reason",
        "trial_ATEs",
    ]
    for path in (OUT_CSV, DISCUSSION_CSV):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=cols)
            writer.writeheader()
            for row in rows:
                serial = row.copy()
                for key in ["median_ATE", "mean_ATE", "std_ATE", "cv", "min_ATE", "max_ATE", "range_ATE"]:
                    serial[key] = f"{serial[key]:.4f}"
                writer.writerow(serial)


def write_variability_matrix(rows: list[dict]) -> None:
    by_scene_method = {(row["scene"], row["method"]): row for row in rows}
    methods = [
        ("Pure", "pure_macvo"),
        ("Rot", "rotation_only"),
        ("Trans", "translation_only"),
        ("Full", "full_imu"),
        ("V3b", "original_v3b"),
        ("V3b+D4", "v3bplus_d4"),
        ("Rule B", "Rule B"),
        ("FD-E", "FD-E only"),
        ("FD-E+CP-B", "FD-E+CP-B"),
        ("CP-B C7", "CP-B-FD-only (candidate7)"),
        ("CP-B P1B", "CP-B-FD-only (phase1b)"),
        ("CP-B V/L", "CP-B-FD-only"),
    ]
    fixed = [
        ("Pure", "pure_macvo"),
        ("Rot", "rotation_only"),
        ("Trans", "translation_only"),
        ("Full", "full_imu"),
    ]
    markers = {"MODERATE": "△", "UNSTABLE": "▲", "INSUFFICIENT": "?"}

    def cell(row: dict | None) -> str:
        if row is None:
            return "—"
        marker = markers.get(row["stability"], "")
        return f"{row['median_ATE']:.2f} [{row['min_ATE']:.2f}-{row['max_ATE']:.2f}]{marker}"

    with DISCUSSION_MATRIX.open("w") as f:
        f.write("# 16 场景全部方法的精度与波动总表\n\n")
        f.write("每格为中位数 [最小值-最大值]，单位为米；△ 表示中等波动，▲ 表示高波动。\n\n")
        headers = ["数据", "场景"] + [label for label, _ in methods] + ["固定模式最佳"]
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("|---|---|" + "---:|" * len(methods) + "---|\n")
        for split_label, scene in SCENES:
            values = [cell(by_scene_method.get((scene, method))) for _, method in methods]
            fixed_rows = [
                (label, by_scene_method.get((scene, method))) for label, method in fixed
            ]
            fixed_rows = [(label, row) for label, row in fixed_rows if row is not None]
            best_label, best_row = min(fixed_rows, key=lambda item: item[1]["median_ATE"])
            best = f"{best_label} {best_row['median_ATE']:.2f}"
            f.write(
                f"| {split_label} | `{scene}` | "
                + " | ".join(values)
                + f" | {best} |\n"
            )


def write_report(rows: list[dict], cross_rows: list[dict]) -> None:
    risky = [r for r in rows if r["stability"] in ("UNSTABLE", "MODERATE")]
    risky.sort(key=lambda r: (0 if r["stability"] == "UNSTABLE" else 1, -r["range_ATE"]))

    with OUT_MD.open("w") as f:
        f.write("# V3b++ Stability / Randomness Audit\n\n")
        f.write("This audit uses existing per-trial result CSVs only; MACVO was not rerun.\n\n")

        f.write("## Summary\n\n")
        counts = defaultdict(int)
        for row in rows:
            counts[row["stability"]] += 1
        f.write(f"- Total scene-method-source groups: {len(rows)}\n")
        f.write(f"- Stable: {counts['STABLE']}\n")
        f.write(f"- Moderate variability: {counts['MODERATE']}\n")
        f.write(f"- Unstable: {counts['UNSTABLE']}\n")
        f.write(f"- Insufficient trials: {counts['INSUFFICIENT']}\n\n")

        f.write("## Reviewer-Critical Variability\n\n")
        f.write("| Stability | Source | Scene | Method | Median | Std | CV | Min-Max | Trial ATEs |\n")
        f.write("|---|---|---|---|---:|---:|---:|---:|---|\n")
        for row in risky:
            f.write(
                f"| {row['stability']} | {row['source']} | {row['scene']} | {row['method']} | "
                f"{row['median_ATE']:.2f} | {row['std_ATE']:.2f} | {row['cv']:.3f} | "
                f"{row['min_ATE']:.2f}-{row['max_ATE']:.2f} | {row['trial_ATEs']} |\n"
            )

        if cross_rows:
            f.write("\n## Cross-Source Reproducibility Checks\n\n")
            f.write("| Scene | Method | Source medians | Delta | Relative delta |\n")
            f.write("|---|---|---|---:|---:|\n")
            for row in cross_rows:
                f.write(
                    f"| {row['scene']} | {row['method']} | {row['source_medians']} | "
                    f"{row['delta']:.2f} | {row['relative_delta']:.1%} |\n"
                )

        f.write("\n## Paper Implications\n\n")
        f.write("- Report median ATE with trial spread for every paper-facing table.\n")
        f.write("- Treat 3 trials as a stability check, not an independent test set.\n")
        f.write("- Any improvement smaller than the run-to-run range should be called marginal or inconclusive.\n")
        f.write("- For high-variance rows, avoid method-ranking claims unless the effect size clearly exceeds the trial spread.\n")


def main() -> None:
    trials = load_trials()
    rows = summarize_trials(trials)
    cross_rows = cross_source_repro(trials)
    write_csv(rows)
    write_variability_matrix(rows)
    write_report(rows, cross_rows)
    print(f"Loaded {len(trials)} per-trial rows.")
    print(f"Wrote {OUT_CSV}")
    print(f"Wrote {DISCUSSION_CSV}")
    print(f"Wrote {DISCUSSION_MATRIX}")
    print(f"Wrote {OUT_MD}")


if __name__ == "__main__":
    main()
