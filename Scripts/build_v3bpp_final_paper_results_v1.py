#!/usr/bin/env python3
from __future__ import annotations

import csv
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "analysis_v3bpp_final_paper_results_v1"

TABLE6_PATH = ROOT / "analysis_v3bpp_discussion_package" / "table_6_all_16scene_all_method_variability.csv"
EARLY7_PATH = ROOT / "analysis_v3bpp_latest_cpb_fdonly_early7_report" / "early7_latest_cpb_scene_summary.csv"
PHASE1B_PATH = ROOT / "analysis_v3bpp_phase1b_cpb_report" / "phase1b_trial_summary.csv"
VALIDATION_PATH = ROOT / "analysis_v3bpp_validation_3scene_report" / "validation_trial_summary.csv"
LOCKED_PATH = ROOT / "analysis_v3bpp_locked_3scene_report" / "locked_trial_summary.csv"

SCENE_ORDER = [
    "turbid_harbor",
    "clear_shallow",
    "deep_dark",
    "caustic_shallow",
    "dam_inspection",
    "murky_coast",
    "open_water",
    "moderate_turbidity",
    "open_water_overcast",
    "twilight_coast",
    "validation_moderate_harbor",
    "validation_transient_dropout",
    "validation_twilight_structure",
    "locked_murky_entry_help",
    "locked_clear_imu_harm",
    "locked_quality_degrade_no_dropout",
]

SCENE_GROUP_BY_SCENE = {
    "turbid_harbor": "early7",
    "clear_shallow": "early7",
    "deep_dark": "early7",
    "caustic_shallow": "early7",
    "dam_inspection": "early7",
    "murky_coast": "early7",
    "open_water": "early7",
    "moderate_turbidity": "post_early_development",
    "open_water_overcast": "post_early_development",
    "twilight_coast": "post_early_development",
    "validation_moderate_harbor": "validation",
    "validation_transient_dropout": "validation",
    "validation_twilight_structure": "validation",
    "locked_murky_entry_help": "locked",
    "locked_clear_imu_harm": "locked",
    "locked_quality_degrade_no_dropout": "locked",
}

FINAL_CPB_SOURCE_BY_SCENE = {
    "turbid_harbor": "latest_cpb_early7",
    "clear_shallow": "latest_cpb_early7",
    "deep_dark": "latest_cpb_early7",
    "caustic_shallow": "latest_cpb_early7",
    "dam_inspection": "latest_cpb_early7",
    "murky_coast": "latest_cpb_early7",
    "open_water": "latest_cpb_early7",
    "moderate_turbidity": "phase1b_cpb",
    "open_water_overcast": "phase1b_cpb",
    "twilight_coast": "phase1b_cpb",
    "validation_moderate_harbor": "validation_3scene",
    "validation_transient_dropout": "validation_3scene",
    "validation_twilight_structure": "validation_3scene",
    "locked_murky_entry_help": "locked_3scene",
    "locked_clear_imu_harm": "locked_3scene",
    "locked_quality_degrade_no_dropout": "locked_3scene",
}

EXCLUDED_FROM_MAIN_SCENES = {
    "open_water",
    "moderate_turbidity",
    "twilight_coast",
    "validation_moderate_harbor",
}

EXCLUSION_REASON_BY_SCENE = {
    "open_water": "degraded_vs_pure_and_large_absolute_ate",
    "moderate_turbidity": "large_absolute_ate_or_unstable",
    "twilight_coast": "large_absolute_ate",
    "validation_moderate_harbor": "degraded_vs_pure",
}

EXCLUSION_REASON_CN = {
    "": "",
    "degraded_vs_pure_and_large_absolute_ate": "相对 Pure MACVO 退化，且 ATE 绝对值偏大",
    "large_absolute_ate_or_unstable": "ATE 绝对值偏大且 trial 波动较大",
    "large_absolute_ate": "ATE 绝对值偏大，不适合作主效果叙事",
    "degraded_vs_pure": "相对 Pure MACVO 明显退化",
}

FIXED_METHODS = {"pure_macvo", "rotation_only", "translation_only", "full_imu"}

INTERPRETATION_BY_SCENE = {
    "turbid_harbor": "near_tie_or_safe",
    "clear_shallow": "near_tie_or_safe",
    "deep_dark": "near_tie_slight_worse",
    "caustic_shallow": "near_tie_slight_worse",
    "dam_inspection": "degradation_or_failure",
    "murky_coast": "clear_improvement",
    "open_water": "degradation_or_failure",
    "moderate_turbidity": "clear_improvement",
    "open_water_overcast": "near_tie_or_safe",
    "twilight_coast": "clear_improvement",
    "validation_moderate_harbor": "degradation_or_failure",
    "validation_transient_dropout": "near_tie_slight_worse",
    "validation_twilight_structure": "near_tie_slight_worse",
    "locked_murky_entry_help": "clear_improvement",
    "locked_clear_imu_harm": "near_tie_slight_worse",
    "locked_quality_degrade_no_dropout": "near_tie_or_safe",
}

INTERPRETATION_CN = {
    "clear_improvement": "明确改善",
    "near_tie_or_safe": "接近持平/安全",
    "near_tie_slight_worse": "接近持平但略差",
    "degradation_or_failure": "退化/失败场景",
}


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def parse_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"non-finite float: {value!r}")
    return number


def format_median_range(median: float, min_value: float, max_value: float) -> str:
    return f"{median:.2f} [{min_value:.2f}-{max_value:.2f}]"


def format_pct(value: float) -> str:
    return f"{value:+.1f}%"


def stats_from_values(values: list[float], source: str, method: str, trial_ids: list[str] | None = None) -> dict[str, object]:
    if not values:
        raise ValueError(f"no ATE values for {source}/{method}")
    mean = statistics.mean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    median = statistics.median(values)
    min_value = min(values)
    max_value = max(values)
    cv = std / mean if mean else 0.0
    return {
        "source": source,
        "method": method,
        "n": len(values),
        "median": median,
        "mean": mean,
        "std": std,
        "cv": cv,
        "min": min_value,
        "max": max_value,
        "median_range": format_median_range(median, min_value, max_value),
        "trial_values": values,
        "trial_ids": trial_ids or [],
    }


def stats_from_summary_row(row: dict[str, str], source: str, method: str) -> dict[str, object]:
    n_key = "n_trials" if "n_trials" in row else "n"
    stat = {
        "source": source,
        "method": method,
        "n": int(row[n_key]),
        "median": parse_float(row["median_ATE"]),
        "mean": parse_float(row["mean_ATE"]) if row.get("mean_ATE") else float("nan"),
        "std": parse_float(row["std_ATE"]) if row.get("std_ATE") else float("nan"),
        "cv": parse_float(row["cv"]) if row.get("cv") else float("nan"),
        "min": parse_float(row["min_ATE"]),
        "max": parse_float(row["max_ATE"]),
        "median_range": row.get("median_range") or format_median_range(
            parse_float(row["median_ATE"]),
            parse_float(row["min_ATE"]),
            parse_float(row["max_ATE"]),
        ),
        "trial_values": [],
        "trial_ids": [],
    }
    if row.get("trial_ATEs"):
        stat["trial_values"] = [parse_float(v) for v in row["trial_ATEs"].split(";") if v]
    else:
        stat["trial_values"] = [
            parse_float(row[key])
            for key in ("trial_1", "trial_2", "trial_3")
            if row.get(key)
        ]
    return stat


def normalize_method_name(value: str) -> str:
    return value.strip().lower().replace("-", "_").replace(" ", "_")


def is_cpb_fd_only_method(value: str) -> bool:
    norm = normalize_method_name(value)
    return norm in {"cp_b_fd_only", "cpb_fd_only", "adaptive_cpb_fdonly", "adaptive_v3b_fdonly"}


def load_fixed_method_stats() -> dict[str, dict[str, dict[str, object]]]:
    by_scene: dict[str, dict[str, dict[str, object]]] = defaultdict(dict)
    for row in read_csv_rows(TABLE6_PATH):
        scene = row["scene"]
        method = normalize_method_name(row["method"])
        if scene not in SCENE_ORDER or method not in FIXED_METHODS:
            continue
        by_scene[scene][method] = stats_from_summary_row(row, row["source"], method)
    return dict(by_scene)


def load_early7_cpb_stats() -> dict[str, dict[str, object]]:
    stats: dict[str, dict[str, object]] = {}
    for row in read_csv_rows(EARLY7_PATH):
        scene = row["scene"]
        if scene in SCENE_ORDER:
            stats[scene] = stats_from_summary_row(row, "latest_cpb_early7", "CP-B-FD-only")
    return stats


def load_trial_cpb_stats(path: Path, source: str, scene_filter: set[str], method_field: str) -> dict[str, dict[str, object]]:
    grouped_values: dict[str, list[float]] = defaultdict(list)
    grouped_trials: dict[str, list[str]] = defaultdict(list)
    for row in read_csv_rows(path):
        scene = row["scene"]
        if scene not in scene_filter:
            continue
        if not is_cpb_fd_only_method(row[method_field]):
            continue
        grouped_values[scene].append(parse_float(row["ATE"]))
        grouped_trials[scene].append(row.get("trial") or row.get("run_name") or "")
    return {
        scene: stats_from_values(values, source, "CP-B-FD-only", grouped_trials[scene])
        for scene, values in grouped_values.items()
    }


def load_final_cpb_stats_by_source() -> dict[str, dict[str, dict[str, object]]]:
    return {
        "latest_cpb_early7": load_early7_cpb_stats(),
        "phase1b_cpb": load_trial_cpb_stats(
            PHASE1B_PATH,
            "phase1b_cpb",
            {"moderate_turbidity", "open_water_overcast", "twilight_coast"},
            "exp",
        ),
        "validation_3scene": load_trial_cpb_stats(
            VALIDATION_PATH,
            "validation_3scene",
            {"validation_moderate_harbor", "validation_transient_dropout", "validation_twilight_structure"},
            "method",
        ),
        "locked_3scene": load_trial_cpb_stats(
            LOCKED_PATH,
            "locked_3scene",
            {"locked_murky_entry_help", "locked_clear_imu_harm", "locked_quality_degrade_no_dropout"},
            "method",
        ),
    }


def classify_scene(scene: str, relative_gain_vs_pure_pct: float) -> str:
    if scene in INTERPRETATION_BY_SCENE:
        return INTERPRETATION_BY_SCENE[scene]
    if relative_gain_vs_pure_pct >= 5.0:
        return "clear_improvement"
    if relative_gain_vs_pure_pct <= -5.0:
        return "degradation_or_failure"
    if relative_gain_vs_pure_pct < 0.0:
        return "near_tie_slight_worse"
    return "near_tie_or_safe"


def paper_role_for_scene(scene: str) -> str:
    return "excluded_from_main" if scene in EXCLUDED_FROM_MAIN_SCENES else "main"


def exclusion_reason_for_scene(scene: str) -> str:
    return EXCLUSION_REASON_BY_SCENE.get(scene, "")


def fixed_best_for_scene(fixed_stats: dict[str, dict[str, object]]) -> tuple[str, dict[str, object]]:
    return min(fixed_stats.items(), key=lambda item: float(item[1]["median"]))


def build_final_rows() -> list[dict[str, object]]:
    fixed_by_scene = load_fixed_method_stats()
    cpb_by_source = load_final_cpb_stats_by_source()
    rows: list[dict[str, object]] = []

    for scene in SCENE_ORDER:
        if scene not in fixed_by_scene:
            raise KeyError(f"missing fixed-method summary for scene: {scene}")
        if "pure_macvo" not in fixed_by_scene[scene]:
            raise KeyError(f"missing pure_macvo summary for scene: {scene}")

        source = FINAL_CPB_SOURCE_BY_SCENE[scene]
        cpb_stats = cpb_by_source.get(source, {}).get(scene)
        if cpb_stats is None:
            raise KeyError(f"missing final CP-B stats for scene/source: {scene}/{source}")

        pure_stats = fixed_by_scene[scene]["pure_macvo"]
        fixed_best_method, fixed_best_stats = fixed_best_for_scene(fixed_by_scene[scene])
        pure_median = float(pure_stats["median"])
        cpb_median = float(cpb_stats["median"])
        fixed_best_median = float(fixed_best_stats["median"])
        delta = cpb_median - pure_median
        gain_pct = (pure_median - cpb_median) / pure_median * 100.0
        gap_to_fixed = cpb_median / fixed_best_median if fixed_best_median else float("inf")
        interpretation = classify_scene(scene, gain_pct)
        exclusion_reason = exclusion_reason_for_scene(scene)

        rows.append(
            {
                "scene_group": SCENE_GROUP_BY_SCENE[scene],
                "scene": scene,
                "paper_role": paper_role_for_scene(scene),
                "exclusion_reason": exclusion_reason,
                "exclusion_reason_cn": EXCLUSION_REASON_CN[exclusion_reason],
                "cpb_source": source,
                "n_cpb_trials": int(cpb_stats["n"]),
                "pure_macvo": pure_stats["median_range"],
                "cpb_fd_only": cpb_stats["median_range"],
                "cpb_median_ATE": cpb_median,
                "cpb_min_ATE": float(cpb_stats["min"]),
                "cpb_max_ATE": float(cpb_stats["max"]),
                "cpb_std_ATE": float(cpb_stats["std"]),
                "cpb_cv": float(cpb_stats["cv"]),
                "pure_median_ATE": pure_median,
                "delta_cpb_minus_pure": delta,
                "relative_gain_vs_pure_pct": gain_pct,
                "fixed_best_method": fixed_best_method,
                "fixed_best": fixed_best_stats["median_range"],
                "fixed_best_median_ATE": fixed_best_median,
                "gap_to_fixed_best_x": gap_to_fixed,
                "interpretation": interpretation,
                "interpretation_cn": INTERPRETATION_CN[interpretation],
                "cpb_trial_ATEs": ";".join(f"{value:.4f}" for value in cpb_stats["trial_values"]),
                "cpb_trial_ids": ";".join(str(value) for value in cpb_stats["trial_ids"]),
            }
        )
    return rows


def as_csv_value(value: object) -> object:
    if isinstance(value, float):
        return f"{value:.4f}"
    return value


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "scene_group",
        "scene",
        "paper_role",
        "exclusion_reason",
        "exclusion_reason_cn",
        "cpb_source",
        "n_cpb_trials",
        "pure_macvo",
        "cpb_fd_only",
        "cpb_median_ATE",
        "cpb_min_ATE",
        "cpb_max_ATE",
        "cpb_std_ATE",
        "cpb_cv",
        "pure_median_ATE",
        "delta_cpb_minus_pure",
        "relative_gain_vs_pure_pct",
        "fixed_best_method",
        "fixed_best",
        "fixed_best_median_ATE",
        "gap_to_fixed_best_x",
        "interpretation",
        "interpretation_cn",
        "cpb_trial_ATEs",
        "cpb_trial_ids",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: as_csv_value(row[key]) for key in fieldnames})


def markdown_table(headers: list[str], body_rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in body_rows)
    return "\n".join(lines)


def write_cpb_vs_pure_markdown(path: Path, rows: list[dict[str, object]]) -> None:
    table_rows = [
        [
            str(row["paper_role"]),
            str(row["scene_group"]),
            str(row["scene"]),
            str(row["cpb_source"]),
            str(row["pure_macvo"]),
            str(row["cpb_fd_only"]),
            format_pct(float(row["relative_gain_vs_pure_pct"])),
            str(row["fixed_best_method"]),
            str(row["fixed_best"]),
            f"{float(row['gap_to_fixed_best_x']):.2f}x",
            str(row["interpretation_cn"]),
            str(row["exclusion_reason_cn"]),
        ]
        for row in rows
    ]
    content = "\n\n".join(
        [
            "# Final CP-B-FD-only vs Pure MACVO (16 scenes)",
            "This table keeps all 16 scenes for traceability. The paper_role column marks whether a scene enters the main paper table.",
            markdown_table(
                [
                    "paper role",
                    "group",
                    "scene",
                    "CP-B source",
                    "Pure MACVO",
                    "CP-B-FD-only",
                    "gain vs pure",
                    "fixed best",
                    "fixed best ATE",
                    "gap",
                    "reading",
                    "exclusion reason",
                ],
                table_rows,
            ),
            "Positive gain means lower ATE than Pure MACVO. Values are median [min-max] over three trials.",
        ]
    )
    path.write_text(content + "\n", encoding="utf-8")


def write_main_result_table(path: Path, rows: list[dict[str, object]]) -> None:
    table_rows = [
        [
            str(row["scene"]),
            str(row["pure_macvo"]),
            str(row["cpb_fd_only"]),
            format_pct(float(row["relative_gain_vs_pure_pct"])),
            str(row["interpretation_cn"]),
        ]
        for row in rows
    ]
    content = "\n\n".join(
        [
            "# Candidate Main Result Table (12 scenes)",
            "This compact table excludes open_water, moderate_turbidity, twilight_coast, and validation_moderate_harbor from the main-effect table while retaining them in the full traceability and excluded-scene audit files.",
            markdown_table(
                ["scene", "Pure MACVO", "CP-B-FD-only", "gain vs pure", "interpretation"],
                table_rows,
            ),
            "Note: all CP-B-FD-only entries are the final parameterization. Values are median [min-max] over three trials.",
        ]
    )
    path.write_text(content + "\n", encoding="utf-8")


def write_excluded_scene_audit(path: Path, rows: list[dict[str, object]]) -> None:
    table_rows = [
        [
            str(row["scene"]),
            str(row["pure_macvo"]),
            str(row["cpb_fd_only"]),
            format_pct(float(row["relative_gain_vs_pure_pct"])),
            str(row["interpretation_cn"]),
            str(row["exclusion_reason_cn"]),
        ]
        for row in rows
    ]
    content = "\n\n".join(
        [
            "# Excluded Scene Audit (4 scenes)",
            "These scenes are excluded from the main-effect table, not deleted from the experiment record. They should be mentioned in limitations or supplementary material.",
            markdown_table(
                ["scene", "Pure MACVO", "CP-B-FD-only", "gain vs pure", "reading", "exclusion reason"],
                table_rows,
            ),
        ]
    )
    path.write_text(content + "\n", encoding="utf-8")


def write_interpretation(path: Path, rows: list[dict[str, object]]) -> None:
    main_rows = [row for row in rows if row["paper_role"] == "main"]
    excluded_rows = [row for row in rows if row["paper_role"] == "excluded_from_main"]
    main_counts = Counter(str(row["interpretation"]) for row in main_rows)
    excluded_counts = Counter(str(row["interpretation"]) for row in excluded_rows)

    def scene_list(selected: list[dict[str, object]]) -> str:
        return "、".join(str(row["scene"]) for row in selected)

    clear_main = [row for row in main_rows if row["interpretation"] == "clear_improvement"]
    near_main = [row for row in main_rows if str(row["interpretation"]).startswith("near_tie")]
    fail_main = [row for row in main_rows if row["interpretation"] == "degradation_or_failure"]

    content = f"""# 最终结果解释草稿

## 当前主文口径

按照新的场景选择，主文结果表保留 12 个场景，排除 `open_water`、`moderate_turbidity`、`twilight_coast`、`validation_moderate_harbor`。这 4 个场景不从实验记录中删除，而是进入 excluded-scene audit / supplementary material。

这个选择需要如实表述：`open_water` 和 `validation_moderate_harbor` 是相对 Pure MACVO 退化；`moderate_turbidity` 和 `twilight_coast` 虽然相对 Pure MACVO 有改善，但 ATE 绝对值偏大，尤其 `moderate_turbidity` trial 波动也较大，因此不适合作主效果叙事。

## 主文 12 场景摘要

- 明确改善：{main_counts['clear_improvement']} 个，分别是 {scene_list(clear_main)}。
- 接近持平或小幅波动：{len(near_main)} 个，分别是 {scene_list(near_main)}。
- 退化/失败但仍保留讨论：{main_counts['degradation_or_failure']} 个，分别是 {scene_list(fail_main)}。

## 排除 4 场景摘要

- 排除场景：{scene_list(excluded_rows)}。
- 排除场景中的相对退化/失败：{excluded_counts['degradation_or_failure']} 个。
- 排除场景中的相对改善但 ATE 绝对值过大：{excluded_counts['clear_improvement']} 个。

## 推荐写法

可以写：“为避免由极端轨迹尺度或失败场景主导主效果表，我们将 4 个场景从主文精度汇总中排除，并在补充材料中完整报告。排除标准包括相对 Pure MACVO 明显退化、ATE 绝对值异常偏大或 trial 波动较大。其余 12 个场景仍用于讨论最终 CP-B-FD-only 配置的行为。”

不建议写成“删除了失败场景”。更稳妥的说法是“excluded from the main-effect table, retained for limitation analysis”。
"""
    path.write_text(content, encoding="utf-8")


def write_readme(path: Path, rows: list[dict[str, object]]) -> None:
    source_counts = Counter(str(row["cpb_source"]) for row in rows)
    source_lines = "\n".join(f"- `{source}`: {count} scenes" for source, count in sorted(source_counts.items()))
    main_count = sum(1 for row in rows if row["paper_role"] == "main")
    excluded_count = sum(1 for row in rows if row["paper_role"] == "excluded_from_main")
    content = f"""# V3b++ Final Paper Results v1

This directory is a paper-facing aggregation layer. It does not launch experiments and does not rewrite source analysis reports.

## Files

- `final_cpb_vs_pure_16scene.csv`: machine-readable 16-scene table with source traceability and paper-role labels.
- `final_cpb_vs_pure_16scene.md`: full readable 16-scene table.
- `main12_cpb_vs_pure.csv`: machine-readable main-effect table after excluding 4 scenes.
- `main12_cpb_vs_pure.md`: readable 12-scene main-effect table.
- `final_main_result_table.md`: same 12-scene main-effect table, kept as a stable paper-draft filename.
- `excluded4_scene_audit.csv`: machine-readable audit of excluded scenes.
- `excluded4_scene_audit.md`: readable audit of excluded scenes and exclusion reasons.
- `final_result_interpretation_cn.md`: Chinese interpretation notes for paper drafting.

## Scene Selection

- Main-effect table: {main_count} scenes.
- Excluded from main-effect table: {excluded_count} scenes.
- Excluded scenes: `open_water`, `moderate_turbidity`, `twilight_coast`, `validation_moderate_harbor`.

The excluded scenes are retained in the full 16-scene table and audit files. They should be treated as limitation/supplementary evidence, not removed from the experiment record.

## CP-B Source Rule

{source_lines}

The paper-facing method name is `CP-B-FD-only` for all 16 scenes. The source column remains visible because the results come from validated batches rather than one single all-scene rerun directory.

## Source Reports

- `{EARLY7_PATH.relative_to(ROOT)}`
- `{PHASE1B_PATH.relative_to(ROOT)}`
- `{VALIDATION_PATH.relative_to(ROOT)}`
- `{LOCKED_PATH.relative_to(ROOT)}`
- `{TABLE6_PATH.relative_to(ROOT)}`
"""
    path.write_text(content, encoding="utf-8")


def write_outputs(rows: list[dict[str, object]]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    main_rows = [row for row in rows if row["paper_role"] == "main"]
    excluded_rows = [row for row in rows if row["paper_role"] == "excluded_from_main"]

    write_csv(OUT_DIR / "final_cpb_vs_pure_16scene.csv", rows)
    write_csv(OUT_DIR / "main12_cpb_vs_pure.csv", main_rows)
    write_csv(OUT_DIR / "excluded4_scene_audit.csv", excluded_rows)
    write_cpb_vs_pure_markdown(OUT_DIR / "final_cpb_vs_pure_16scene.md", rows)
    write_main_result_table(OUT_DIR / "main12_cpb_vs_pure.md", main_rows)
    write_main_result_table(OUT_DIR / "final_main_result_table.md", main_rows)
    write_excluded_scene_audit(OUT_DIR / "excluded4_scene_audit.md", excluded_rows)
    write_interpretation(OUT_DIR / "final_result_interpretation_cn.md", rows)
    write_readme(OUT_DIR / "README.md", rows)


def main() -> None:
    rows = build_final_rows()
    write_outputs(rows)
    main_count = sum(1 for row in rows if row["paper_role"] == "main")
    excluded_count = sum(1 for row in rows if row["paper_role"] == "excluded_from_main")
    print(f"Wrote {len(rows)} rows to {OUT_DIR} ({main_count} main, {excluded_count} excluded)")


if __name__ == "__main__":
    main()