#!/usr/bin/env python3
"""
Build a compact discussion package for the V3b++ adaptive-IMU results.

This package is meant for collaborators who need to understand:
  1. what the method is compared against,
  2. where it improves over pure MACVO,
  3. what fixed-mode controls reveal,
  4. what the locked L1 failure means.
"""

from __future__ import annotations

import csv
import math
from collections import Counter
from pathlib import Path

WORKDIR = Path("/home/admin1/macvo-dev")
OUTDIR = WORKDIR / "analysis_v3bpp_discussion_package"

DEV_WIDE = WORKDIR / "analysis_v3bpp_10scene_full_accuracy_wide.csv"
VAL_ADAPT = WORKDIR / "analysis_v3bpp_validation_3scene_report/validation_adaptive_comparison.csv"
VAL_ORACLE = WORKDIR / "analysis_v3bpp_validation_3scene_report/validation_fixed_oracle_by_scene.csv"
LOCKED_WIDE = WORKDIR / "analysis_v3bpp_locked_3scene_report/locked_accuracy_wide.csv"
LOCKED_SCENE_SUMMARY = WORKDIR / "analysis_v3bpp_locked_3scene_report/locked_scene_summary.csv"
L1_SEGMENT = WORKDIR / "analysis_v3bpp_locked_l1_oracle_recovery_audit/l1_segment_error_summary.csv"
L1_CHECKPOINT = WORKDIR / "analysis_v3bpp_locked_l1_oracle_recovery_audit/l1_checkpoint_error_summary.csv"
L1_PAIRWISE = WORKDIR / "analysis_v3bpp_locked_l1_oracle_recovery_audit/l1_pairwise_method_distance_summary.csv"


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def fnum(value: str | float | int | None) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def fmt(value: float, ndigits: int = 4) -> str:
    return "" if math.isnan(value) else f"{value:.{ndigits}f}"


def classify(delta: float) -> str:
    if delta < -0.5:
        return "improved_vs_pure"
    if delta > 0.5:
        return "regressed_vs_pure"
    return "near_neutral_vs_pure"


def cpb_from_dev_row(row: dict) -> tuple[float, str]:
    phase = row.get("CP-B-FD-only (phase1b)", "")
    cand = row.get("CP-B-FD-only (candidate7)", "")
    if phase:
        return fnum(phase), "phase1b"
    if cand:
        return fnum(cand), "candidate7"
    return float("nan"), ""


def build_main_rows() -> list[dict]:
    rows = []

    for row in read_csv(DEV_WIDE):
        pure = fnum(row["pure_macvo"])
        cpb, source = cpb_from_dev_row(row)
        delta = cpb - pure
        gain = (pure - cpb) / pure * 100 if pure > 0 else float("nan")
        oracle = fnum(row["oracle_ATE"])
        rows.append(
            {
                "split": "development_10",
                "scene": row["scene"],
                "pure_macvo": pure,
                "cpb_ours": cpb,
                "cpb_source": source,
                "delta_cpb_minus_pure": delta,
                "relative_gain_pct": gain,
                "oracle_method": row["oracle_mode"],
                "oracle_ATE": oracle,
                "cpb_vs_oracle_ratio": cpb / oracle if oracle > 0 else float("nan"),
                "interpretation": classify(delta),
            }
        )

    val_oracle = {row["scene"]: row for row in read_csv(VAL_ORACLE)}
    for row in read_csv(VAL_ADAPT):
        scene = row["scene"]
        oracle_row = val_oracle[scene]
        pure = fnum(oracle_row["pure_macvo"])
        cpb = fnum(row["cpb_median_ATE"])
        delta = cpb - pure
        gain = (pure - cpb) / pure * 100 if pure > 0 else float("nan")
        oracle = fnum(row["oracle_ATE"])
        rows.append(
            {
                "split": "validation_3",
                "scene": scene,
                "pure_macvo": pure,
                "cpb_ours": cpb,
                "cpb_source": "validation",
                "delta_cpb_minus_pure": delta,
                "relative_gain_pct": gain,
                "oracle_method": row["oracle_method"],
                "oracle_ATE": oracle,
                "cpb_vs_oracle_ratio": fnum(row["cpb_vs_oracle_ratio"]),
                "interpretation": classify(delta),
            }
        )

    for row in read_csv(LOCKED_WIDE):
        pure = fnum(row["pure_macvo"])
        cpb = fnum(row["cpb_fd_only"])
        delta = cpb - pure
        gain = (pure - cpb) / pure * 100 if pure > 0 else float("nan")
        oracle = fnum(row["oracle_ATE"])
        rows.append(
            {
                "split": "locked_3",
                "scene": row["scene"],
                "pure_macvo": pure,
                "cpb_ours": cpb,
                "cpb_source": "locked",
                "delta_cpb_minus_pure": delta,
                "relative_gain_pct": gain,
                "oracle_method": row["oracle_method"],
                "oracle_ATE": oracle,
                "cpb_vs_oracle_ratio": cpb / oracle if oracle > 0 else float("nan"),
                "interpretation": classify(delta),
            }
        )

    return rows


def build_oracle_rows() -> list[dict]:
    rows = []

    for row in read_csv(DEV_WIDE):
        rows.append(
            {
                "split": "development_10",
                "scene": row["scene"],
                "pure_macvo": fnum(row["pure_macvo"]),
                "rotation_only": fnum(row["rotation_only"]),
                "translation_only": fnum(row["translation_only"]),
                "full_imu": fnum(row["full_imu"]),
                "oracle_method": row["oracle_mode"],
                "oracle_ATE": fnum(row["oracle_ATE"]),
                "diagnostic_role": "fixed-mode oracle/reference controls",
            }
        )

    for row in read_csv(VAL_ORACLE):
        rows.append(
            {
                "split": "validation_3",
                "scene": row["scene"],
                "pure_macvo": fnum(row["pure_macvo"]),
                "rotation_only": fnum(row["rotation_only"]),
                "translation_only": fnum(row["translation_only"]),
                "full_imu": fnum(row["full_imu"]),
                "oracle_method": row["oracle_method"],
                "oracle_ATE": fnum(row["oracle_median_ATE"]),
                "diagnostic_role": "fixed-mode oracle/reference controls",
            }
        )

    for row in read_csv(LOCKED_WIDE):
        rows.append(
            {
                "split": "locked_3",
                "scene": row["scene"],
                "pure_macvo": fnum(row["pure_macvo"]),
                "rotation_only": fnum(row["rotation_only"]),
                "translation_only": fnum(row["translation_only"]),
                "full_imu": fnum(row["full_imu"]),
                "oracle_method": row["oracle_method"],
                "oracle_ATE": fnum(row["oracle_ATE"]),
                "diagnostic_role": "fixed-mode oracle/reference controls",
            }
        )

    return rows


def build_internal_ablation_rows() -> list[dict]:
    rows = []
    for row in read_csv(DEV_WIDE):
        ruleb = fnum(row["ruleB"])
        cpb, source = cpb_from_dev_row(row)
        delta = cpb - ruleb
        gain = (ruleb - cpb) / ruleb * 100 if ruleb > 0 else float("nan")
        rows.append(
            {
                "split": "development_10",
                "scene": row["scene"],
                "ruleB_internal_predecessor": ruleb,
                "cpb_ours": cpb,
                "cpb_source": source,
                "delta_cpb_minus_ruleB": delta,
                "relative_gain_pct": gain,
                "role": "internal ablation only, not paper-facing baseline",
            }
        )

    for row in read_csv(VAL_ADAPT):
        ruleb = fnum(row["ruleB_median_ATE"])
        cpb = fnum(row["cpb_median_ATE"])
        delta = cpb - ruleb
        gain = (ruleb - cpb) / ruleb * 100 if ruleb > 0 else float("nan")
        rows.append(
            {
                "split": "validation_3",
                "scene": row["scene"],
                "ruleB_internal_predecessor": ruleb,
                "cpb_ours": cpb,
                "cpb_source": "validation",
                "delta_cpb_minus_ruleB": delta,
                "relative_gain_pct": gain,
                "role": "internal ablation only, not paper-facing baseline",
            }
        )

    for row in read_csv(LOCKED_WIDE):
        ruleb = fnum(row["ruleB"])
        cpb = fnum(row["cpb_fd_only"])
        delta = cpb - ruleb
        gain = (ruleb - cpb) / ruleb * 100 if ruleb > 0 else float("nan")
        rows.append(
            {
                "split": "locked_3",
                "scene": row["scene"],
                "ruleB_internal_predecessor": ruleb,
                "cpb_ours": cpb,
                "cpb_source": "locked",
                "delta_cpb_minus_ruleB": delta,
                "relative_gain_pct": gain,
                "role": "internal ablation only, not paper-facing baseline",
            }
        )
    return rows


def lookup(rows: list[dict], **keys) -> dict:
    for row in rows:
        if all(row.get(k) == v for k, v in keys.items()):
            return row
    raise KeyError(keys)


def build_l1_failure_rows() -> list[dict]:
    scene_rows = read_csv(LOCKED_SCENE_SUMMARY)
    seg_rows = read_csv(L1_SEGMENT)
    cp_rows = read_csv(L1_CHECKPOINT)
    pair_rows = read_csv(L1_PAIRWISE)

    cpb_scene = lookup(scene_rows, scene="locked_murky_entry_help", method="cpb_fd_only")
    ruleb_scene = lookup(scene_rows, scene="locked_murky_entry_help", method="ruleB")

    def seg(method: str, segment: str) -> float:
        return fnum(lookup(seg_rows, method=method, segment=segment)["median_rmse"])

    def cp(method: str, frame: str) -> float:
        return fnum(lookup(cp_rows, method=method, frame_idx=frame)["median_position_error"])

    def pair(left: str, right: str, segment: str) -> float:
        return fnum(
            lookup(pair_rows, left_method=left, right_method=right, segment=segment)[
                "median_position_rmse_between_methods"
            ]
        )

    return [
        {
            "finding": "CP-B enters full_imu early",
            "evidence": "first_full_row",
            "ruleB_value": fnum(ruleb_scene["median_first_full_imu_row"]),
            "cpb_value": fnum(cpb_scene["median_first_full_imu_row"]),
            "interpretation": "not an entry-miss failure on locked L1",
        },
        {
            "finding": "CP-B uses full_imu for most of L1",
            "evidence": "full_imu_usage_pct",
            "ruleB_value": fnum(ruleb_scene["median_full_imu_pct"]),
            "cpb_value": fnum(cpb_scene["median_full_imu_pct"]),
            "interpretation": "CP-B increases full_imu usage but still fails oracle recovery",
        },
        {
            "finding": "CP-B temporarily recovers before FD exit",
            "evidence": "position_error_at_row_1162",
            "ruleB_value": cp("ruleB", "1162"),
            "cpb_value": cp("cpb_fd_only", "1162"),
            "fixed_full_imu_reference": cp("full_imu", "1162"),
            "interpretation": "entry can work locally; failure is later",
        },
        {
            "finding": "FD exit/cooldown is the proximate collapse",
            "evidence": "position_error_at_row_1377",
            "ruleB_value": cp("ruleB", "1377"),
            "cpb_value": cp("cpb_fd_only", "1377"),
            "fixed_full_imu_reference": cp("full_imu", "1377"),
            "interpretation": "error explodes after full_divergence exit/cooldown",
        },
        {
            "finding": "CP-B is nearly identical to Rule B before cooldown",
            "evidence": "pairwise_rmse_first_full_episode",
            "ruleB_value": float("nan"),
            "cpb_value": pair("cpb_fd_only", "ruleB", "first_full_episode"),
            "interpretation": "CP-B does not change pre-cooldown behavior",
        },
        {
            "finding": "CP-B does not recover full_imu oracle",
            "evidence": "all_sequence_ATE",
            "ruleB_value": seg("ruleB", "all"),
            "cpb_value": seg("cpb_fd_only", "all"),
            "fixed_full_imu_reference": seg("full_imu", "all"),
            "interpretation": "exit/recovery failure after successful full_imu entry",
        },
    ]


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            out = {}
            for field in fields:
                value = row.get(field, "")
                if isinstance(value, float):
                    out[field] = fmt(value)
                else:
                    out[field] = value
            writer.writerow(out)


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |"]
    out.append("|" + "|".join(["---"] + ["---:"] * (len(headers) - 1)) + "|")
    for row in rows:
        out.append("| " + " | ".join(row) + " |")
    return "\n".join(out) + "\n"


def write_readme(main_rows: list[dict], l1_rows: list[dict]) -> None:
    counts = Counter(row["interpretation"] for row in main_rows)
    by_split = Counter((row["split"], row["interpretation"]) for row in main_rows)

    highlight_scenes = [
        "murky_coast",
        "moderate_turbidity",
        "locked_murky_entry_help",
        "locked_clear_imu_harm",
        "locked_quality_degrade_no_dropout",
        "open_water",
        "validation_moderate_harbor",
    ]
    main_lookup = {row["scene"]: row for row in main_rows}
    highlight_rows = []
    for scene in highlight_scenes:
        row = main_lookup.get(scene)
        if not row:
            continue
        highlight_rows.append(
            [
                row["split"],
                scene,
                f"{row['pure_macvo']:.2f}",
                f"{row['cpb_ours']:.2f}",
                f"{row['relative_gain_pct']:+.1f}%",
                row["oracle_method"],
                f"{row['cpb_vs_oracle_ratio']:.2f}x",
                row["interpretation"],
            ]
        )

    with (OUTDIR / "README.md").open("w", encoding="utf-8") as f:
        f.write("# V3b++ Discussion Package\n\n")
        f.write("Purpose: give collaborators a compact, readable view of what the adaptive-IMU method does and what the current results support.\n\n")

        f.write("## Baseline Framing\n\n")
        f.write("- Paper-facing baseline: **pure MACVO**.\n")
        f.write("- Proposed/candidate method: **CP-B-FD-only adaptive IMU mode selection**.\n")
        f.write("- `rotation_only`, `translation_only`, and `full_imu` are **fixed-mode diagnostic controls / oracle references**, not the main baseline.\n")
        f.write("- `Rule B` is an **in-family predecessor for internal ablation**, not the paper-facing baseline.\n\n")

        f.write("## One-Sentence Result\n\n")
        f.write(
            "Compared with pure MACVO, the adaptive controller improves several visually degraded scenes while staying near-neutral on safe scenes; "
            "fixed-mode controls show that it still does not reliably recover the full_imu oracle, with locked L1 exposing an exit/recovery failure after successful entry.\n\n"
        )

        f.write("## Main Result Counts: CP-B vs Pure MACVO\n\n")
        f.write(f"- Improved vs pure: {counts['improved_vs_pure']} scenes.\n")
        f.write(f"- Near-neutral vs pure: {counts['near_neutral_vs_pure']} scenes.\n")
        f.write(f"- Regressed vs pure: {counts['regressed_vs_pure']} scenes.\n\n")
        f.write("By split:\n\n")
        for split in ["development_10", "validation_3", "locked_3"]:
            f.write(
                f"- `{split}`: improved={by_split[(split, 'improved_vs_pure')]}, "
                f"neutral={by_split[(split, 'near_neutral_vs_pure')]}, "
                f"regressed={by_split[(split, 'regressed_vs_pure')]}\n"
            )

        f.write("\n## Key Scenes\n\n")
        f.write(
            md_table(
                ["Split", "Scene", "Pure", "CP-B", "Gain vs Pure", "Oracle", "CP-B/Oracle", "Interpretation"],
                highlight_rows,
            )
        )

        f.write("## Locked L1 Mechanism\n\n")
        f.write("`locked_murky_entry_help` is the key failure-analysis scene. CP-B improves over pure MACVO but fails to recover fixed full_imu oracle.\n\n")
        f.write("| Finding | Evidence | Rule B | CP-B | Fixed full_imu ref | Interpretation |\n")
        f.write("|---|---|---:|---:|---:|---|\n")
        for row in l1_rows:
            f.write(
                f"| {row['finding']} | {row['evidence']} | {fmt(row.get('ruleB_value', float('nan')))} | "
                f"{fmt(row.get('cpb_value', float('nan')))} | {fmt(row.get('fixed_full_imu_reference', float('nan')))} | "
                f"{row['interpretation']} |\n"
            )

        f.write("\n## Files in This Package\n\n")
        f.write("- `table_1_main_results_vs_pure_macvo.csv`: main paper-facing table; CP-B vs pure MACVO plus oracle gap.\n")
        f.write("- `table_2_fixed_mode_oracle_controls.csv`: pure/rotation/translation/full fixed controls and oracle method.\n")
        f.write("- `table_3_internal_ablation_ruleB.csv`: CP-B vs Rule B; internal ablation only.\n")
        f.write("- `table_4_locked_l1_failure_mechanism.csv`: compact L1 mechanism evidence.\n")
        f.write("- `claims_and_limitations.md`: safe claims, unsafe claims, and recommended table layout.\n\n")

        f.write("## Suggested Discussion Question\n\n")
        f.write(
            "Is the paper best framed as a diagnostic ablation study of adaptive IMU mode selection, with pure MACVO as the main baseline and fixed-mode controls as oracle references? "
            "The current evidence strongly supports that framing.\n"
        )


def write_claims() -> None:
    with (OUTDIR / "claims_and_limitations.md").open("w", encoding="utf-8") as f:
        f.write("# Claims and Limitations for Discussion\n\n")
        f.write("## Safe Claims\n\n")
        f.write("- The method adds adaptive IMU mode selection on top of MACVO.\n")
        f.write("- The paper-facing baseline is pure MACVO.\n")
        f.write("- Compared with pure MACVO, CP-B-FD-only improves several visually degraded scenes, including `murky_coast`, `moderate_turbidity`, and locked L1.\n")
        f.write("- Fixed-mode controls show that full_imu is highly scene-dependent: sometimes oracle, sometimes harmful.\n")
        f.write("- Locked L2 shows safety: the gate avoids harmful full_imu and remains near pure MACVO.\n")
        f.write("- Locked L1 exposes a remaining failure mode: exit/recovery failure after successful full_imu entry.\n\n")

        f.write("## Unsafe Claims\n\n")
        f.write("- Do not claim CP-B-FD-only generally recovers the full_imu oracle.\n")
        f.write("- Do not call Rule B the main baseline; it is an internal predecessor.\n")
        f.write("- Do not claim broad SOTA SLAM/VIO performance without external baselines such as ORB-SLAM3/VINS/DROID.\n")
        f.write("- Do not hide fixed-mode controls; they are necessary to show why adaptive mode selection is nontrivial.\n\n")

        f.write("## Recommended Paper Table Layout\n\n")
        f.write("Main table:\n\n")
        f.write("- Scene\n")
        f.write("- Pure MACVO\n")
        f.write("- CP-B-FD-only\n")
        f.write("- Delta / relative gain vs pure MACVO\n")
        f.write("- Best fixed mode\n")
        f.write("- CP-B / oracle ratio\n")
        f.write("- diagnostic label\n\n")

        f.write("Secondary diagnostic table:\n\n")
        f.write("- Pure MACVO\n")
        f.write("- Rotation-only\n")
        f.write("- Translation-only\n")
        f.write("- Full-IMU\n")
        f.write("- Fixed-mode oracle\n\n")

        f.write("Internal ablation table:\n\n")
        f.write("- Rule B\n")
        f.write("- FD-E only\n")
        f.write("- FD-E+CP-B\n")
        f.write("- CP-B-FD-only\n\n")

        f.write("## Recommended Title Direction\n\n")
        f.write("`When Should Underwater Visual Odometry Trust IMU Priors? A Diagnostic Study of Adaptive Mode Selection`\n")


def write_figures() -> None:
    figures_md = OUTDIR / "figures_for_discussion.md"
    with figures_md.open("w", encoding="utf-8") as f:
        f.write("# Discussion Figures\n\n")
        f.write("These are Mermaid diagrams for quick discussion. They are not final camera-ready figures, but they make the method logic and result interpretation clear.\n\n")

        f.write("## Figure 1: Method Flow\n\n")
        f.write("```mermaid\n")
        f.write("""flowchart TD
    A[Stereo image pair + IMU stream] --> B[MACVO frontend and two-frame optimization]
    B --> C[Online visual diagnostics]
    C --> C1[n_vis / visual residual count]
    C --> C2[median flow covariance]
    C --> C3[IMU residual diagnostics]
    C1 --> D[Two-level visual collapse gate]
    C2 --> D
    D -->|No collapse| E[Default rotation-only IMU prior]
    D -->|Severe or sustained mild collapse| F[Enter full_imu prior]
    F --> G[Full-divergence monitor]
    G -->|Stable| H[Continue full_imu]
    G -->|Divergence triggered| I[Exit to pure / cooldown]
    I --> D
    E --> J[Estimated trajectory]
    H --> J
    I --> J
""")
        f.write("```\n\n")

        f.write("## Figure 2: Correct Baseline Framing\n\n")
        f.write("```mermaid\n")
        f.write("""flowchart LR
    P[pure MACVO<br/>paper-facing baseline] --> Q[CP-B-FD-only<br/>adaptive IMU mode selection]
    Q --> R[Main question:<br/>Does adaptive IMU improve over pure MACVO?]

    subgraph Diagnostics[Fixed-mode diagnostic controls / oracle references]
        RO[rotation-only]
        TO[translation-only]
        FI[full-IMU]
    end
    Diagnostics --> O[Best fixed-mode oracle per scene]
    O --> S[Does CP-B approach oracle?]

    RB[Rule B<br/>in-family predecessor] --> AB[Internal ablation only]
    Q --> AB

    EXT[ORB-SLAM3 / VINS / DROID] -. absent in current experiments .-> Scope[Scope limitation:<br/>not a SOTA SLAM claim]
""")
        f.write("```\n\n")

        f.write("## Figure 3: Result Interpretation Flow\n\n")
        f.write("```mermaid\n")
        f.write("""flowchart TD
    A[Scene result] --> B{CP-B vs pure MACVO}
    B -->|Better| C[Adaptive IMU helps this scene]
    B -->|Near equal| D[Safety / neutrality case]
    B -->|Worse| E[Regression case]

    C --> F{CP-B vs fixed-mode oracle}
    D --> F
    E --> F

    F -->|Near oracle| G[Successful mode selection]
    F -->|Far from oracle + no full_imu entry| H[Entry coverage failure]
    F -->|Far from oracle + full_imu entered| I[Exit / recovery failure]
    F -->|Oracle is translation-only| J[Candidate-set limitation]

    G --> K[Safe positive claim]
    H --> L[Diagnostic limitation]
    I --> L
    J --> L
""")
        f.write("```\n\n")

        f.write("## Figure 4: Locked L1 Failure Timeline\n\n")
        f.write("```mermaid\n")
        f.write("""timeline
    title locked_murky_entry_help: CP-B Oracle-Recovery Failure
    rows 1-16 : default rotation-only : initial offset appears : error at row 16 approx 4.87m
    row 17 : full_imu entry : entry succeeds early
    rows 17-1161 : long full_imu episode : partial recovery : error at row 1162 approx 1.66m
    row 1162 : full_divergence exit : controller leaves working full_imu
    rows 1162-1377 : FD cooldown / pure recovery : trajectory collapses : error at row 1377 approx 83.91m
    rows 1378-1799 : re-entry after cooldown : full_imu usage resumes but oracle not recovered : final ATE approx 40.57m
""")
        f.write("```\n\n")

        f.write("## Figure 5: Evidence Map\n\n")
        f.write("```mermaid\n")
        f.write("""flowchart TD
    A[Evidence] --> B[CP-B improves over pure MACVO on degraded scenes]
    A --> C[CP-B remains safe / neutral on clear or full_imu-harmful scenes]
    A --> D[Fixed-mode controls reveal oracle gaps]
    A --> E[Locked L1 reveals exit/recovery failure]

    B --> F[Supported claim:<br/>adaptive IMU can help underwater VO under degradation]
    C --> G[Supported claim:<br/>gate can avoid harmful full_imu]
    D --> H[Limitation:<br/>not always oracle recovery]
    E --> H

    H --> I[Paper framing:<br/>diagnostic ablation study, not final SOTA method]
""")
        f.write("```\n")

    mermaid_files = {
        "figure_1_method_flow.mmd": """flowchart TD
    A[Stereo image pair + IMU stream] --> B[MACVO frontend and two-frame optimization]
    B --> C[Online visual diagnostics]
    C --> C1[n_vis / visual residual count]
    C --> C2[median flow covariance]
    C --> C3[IMU residual diagnostics]
    C1 --> D[Two-level visual collapse gate]
    C2 --> D
    D -->|No collapse| E[Default rotation-only IMU prior]
    D -->|Severe or sustained mild collapse| F[Enter full_imu prior]
    F --> G[Full-divergence monitor]
    G -->|Stable| H[Continue full_imu]
    G -->|Divergence triggered| I[Exit to pure / cooldown]
    I --> D
    E --> J[Estimated trajectory]
    H --> J
    I --> J
""",
        "figure_2_baseline_framing.mmd": """flowchart LR
    P[pure MACVO<br/>paper-facing baseline] --> Q[CP-B-FD-only<br/>adaptive IMU mode selection]
    Q --> R[Main question:<br/>Does adaptive IMU improve over pure MACVO?]
    subgraph Diagnostics[Fixed-mode diagnostic controls / oracle references]
        RO[rotation-only]
        TO[translation-only]
        FI[full-IMU]
    end
    Diagnostics --> O[Best fixed-mode oracle per scene]
    O --> S[Does CP-B approach oracle?]
    RB[Rule B<br/>in-family predecessor] --> AB[Internal ablation only]
    Q --> AB
    EXT[ORB-SLAM3 / VINS / DROID] -. absent in current experiments .-> Scope[Scope limitation:<br/>not a SOTA SLAM claim]
""",
        "figure_3_result_interpretation.mmd": """flowchart TD
    A[Scene result] --> B{CP-B vs pure MACVO}
    B -->|Better| C[Adaptive IMU helps this scene]
    B -->|Near equal| D[Safety / neutrality case]
    B -->|Worse| E[Regression case]
    C --> F{CP-B vs fixed-mode oracle}
    D --> F
    E --> F
    F -->|Near oracle| G[Successful mode selection]
    F -->|Far from oracle + no full_imu entry| H[Entry coverage failure]
    F -->|Far from oracle + full_imu entered| I[Exit / recovery failure]
    F -->|Oracle is translation-only| J[Candidate-set limitation]
    G --> K[Safe positive claim]
    H --> L[Diagnostic limitation]
    I --> L
    J --> L
""",
        "figure_4_locked_l1_timeline.mmd": """timeline
    title locked_murky_entry_help: CP-B Oracle-Recovery Failure
    rows 1-16 : default rotation-only : initial offset appears : error at row 16 approx 4.87m
    row 17 : full_imu entry : entry succeeds early
    rows 17-1161 : long full_imu episode : partial recovery : error at row 1162 approx 1.66m
    row 1162 : full_divergence exit : controller leaves working full_imu
    rows 1162-1377 : FD cooldown / pure recovery : trajectory collapses : error at row 1377 approx 83.91m
    rows 1378-1799 : re-entry after cooldown : full_imu usage resumes but oracle not recovered : final ATE approx 40.57m
""",
    }
    for name, text in mermaid_files.items():
        (OUTDIR / name).write_text(text, encoding="utf-8")

    with (OUTDIR / "external_ai_figure_prompts.md").open("w", encoding="utf-8") as f:
        f.write("# External AI Figure Prompts\n\n")
        f.write("Use these prompts if you want polished publication-style figures from an external design/image model.\n\n")
        f.write("## Prompt 1: Method Pipeline Figure\n\n")
        f.write(
            "Create a clean academic vector-style flowchart for a robotics paper. Topic: adaptive IMU mode selection on top of MACVO for underwater stereo visual odometry. "
            "Show inputs: stereo image pair and IMU stream. Then MACVO frontend and two-frame optimization. Then online visual diagnostics: n_vis, median flow covariance, IMU residual diagnostics. "
            "Then a two-level visual collapse gate. If no collapse, use rotation-only IMU prior. If severe/sustained collapse, enter full_imu prior. Then full-divergence monitor: stable continues full_imu, divergence exits to pure/cooldown and loops back to gate. "
            "Output estimated trajectory. Use restrained blue/green/gray colors, no decorative background, paper-ready labels, horizontal layout, vector diagram style.\n\n"
        )
        f.write("## Prompt 2: Baseline Framing Figure\n\n")
        f.write(
            "Create a paper-ready conceptual diagram showing baseline hierarchy. Center: CP-B-FD-only adaptive IMU mode selection. Left: pure MACVO, labeled paper-facing baseline. "
            "Top/right: fixed-mode diagnostic controls: rotation-only, translation-only, full-IMU, feeding into best fixed-mode oracle per scene. Bottom: Rule B, labeled in-family predecessor / internal ablation only. "
            "Dashed side note: external baselines such as ORB-SLAM3/VINS/DROID are absent and treated as scope limitation. Make the diagram clean, explicit, non-marketing, suitable for a technical paper.\n\n"
        )
        f.write("## Prompt 3: Locked L1 Failure Timeline\n\n")
        f.write(
            "Create a horizontal timeline figure for locked_murky_entry_help. It should show CP-B behavior across rows: rows 1-16 default rotation-only with initial offset, row 17 full_imu entry, rows 17-1161 long full_imu episode with recovery, row 1162 full_divergence exit, rows 1162-1377 cooldown/pure recovery with trajectory collapse, rows 1378-1799 re-entry but oracle not recovered. "
            "Annotate key errors: row 1162 CP-B approx 1.66 m, fixed full_imu approx 3.81 m; row 1377 CP-B approx 83.91 m, fixed full_imu approx 7.08 m; final CP-B ATE approx 40.57 m, fixed full_imu oracle approx 5.99 m. "
            "Use a clean technical style with red highlighting only for the collapse interval and neutral colors elsewhere.\n\n"
        )
        f.write("## Prompt 4: Result Interpretation Decision Tree\n\n")
        f.write(
            "Create a compact decision-tree diagram for interpreting adaptive IMU mode selection results. Start with scene result. First branch: CP-B vs pure MACVO: better, near equal, worse. Then branch by CP-B vs fixed-mode oracle: near oracle means successful mode selection; far from oracle with no full_imu entry means entry coverage failure; far from oracle with full_imu entered means exit/recovery failure; oracle is translation-only means candidate-set limitation. "
            "Use clean academic diagram style, grayscale with small accent colors, no background illustration.\n"
        )


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)

    main_rows = build_main_rows()
    oracle_rows = build_oracle_rows()
    ablation_rows = build_internal_ablation_rows()
    l1_rows = build_l1_failure_rows()

    write_csv(
        OUTDIR / "table_1_main_results_vs_pure_macvo.csv",
        main_rows,
        [
            "split",
            "scene",
            "pure_macvo",
            "cpb_ours",
            "cpb_source",
            "delta_cpb_minus_pure",
            "relative_gain_pct",
            "oracle_method",
            "oracle_ATE",
            "cpb_vs_oracle_ratio",
            "interpretation",
        ],
    )
    write_csv(
        OUTDIR / "table_2_fixed_mode_oracle_controls.csv",
        oracle_rows,
        [
            "split",
            "scene",
            "pure_macvo",
            "rotation_only",
            "translation_only",
            "full_imu",
            "oracle_method",
            "oracle_ATE",
            "diagnostic_role",
        ],
    )
    write_csv(
        OUTDIR / "table_3_internal_ablation_ruleB.csv",
        ablation_rows,
        [
            "split",
            "scene",
            "ruleB_internal_predecessor",
            "cpb_ours",
            "cpb_source",
            "delta_cpb_minus_ruleB",
            "relative_gain_pct",
            "role",
        ],
    )
    write_csv(
        OUTDIR / "table_4_locked_l1_failure_mechanism.csv",
        l1_rows,
        ["finding", "evidence", "ruleB_value", "cpb_value", "fixed_full_imu_reference", "interpretation"],
    )
    write_readme(main_rows, l1_rows)
    write_claims()
    write_figures()

    print(f"Wrote discussion package: {OUTDIR}")
    for path in sorted(OUTDIR.glob("*")):
        print(f"  {path.name}")


if __name__ == "__main__":
    main()
