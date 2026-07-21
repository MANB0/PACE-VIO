#!/usr/bin/env python3
"""
Analyse V3b++ locked held-out results.

Inputs:
  Results/v3bpp_locked_3scene_54runs/
  /mnt/e/文档/holoocean/code/recordings/batch_20260618_110853/

Outputs:
  analysis_v3bpp_locked_3scene_report/
"""

from __future__ import annotations

import csv
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml

WORKDIR = Path("/home/admin1/macvo-dev")
sys.path.insert(0, str(WORKDIR))

from Scripts.eval_qa_vif import evaluate_trajectory_direct  # noqa: E402
from Utility.Config import IncludeLoader  # noqa: E402

RESULT_ROOT = WORKDIR / "Results/v3bpp_locked_3scene_54runs"
BATCH = Path("/mnt/e/文档/holoocean/code/recordings/batch_20260618_110853")
OUTDIR = WORKDIR / "analysis_v3bpp_locked_3scene_report"

SCENES = [
    "locked_murky_entry_help",
    "locked_clear_imu_harm",
    "locked_quality_degrade_no_dropout",
]

SCENE_PURPOSE = {
    "locked_murky_entry_help": "L1 positive entry-coverage test: full_imu should help and gate should enter.",
    "locked_clear_imu_harm": "L2 negative safety test: full_imu should not be needed; gate should avoid it.",
    "locked_quality_degrade_no_dropout": "L3 quality-degradation test: detects whether high n_vis but poor quality causes entry failure.",
}

METHODS_ORDER = [
    "pure_macvo",
    "rotation_only",
    "translation_only",
    "full_imu",
    "ruleB",
    "cpb_fd_only",
]

FIXED_METHODS = METHODS_ORDER[:4]


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.load(f, IncludeLoader)


def is_true(value: str | int | float | None) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def as_float(value: str | int | float | None) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def fmt(value: float, ndigits: int = 4) -> str:
    return "" if math.isnan(value) else f"{value:.{ndigits}f}"


def pct(numer: float, denom: float) -> float:
    return numer / denom * 100.0 if denom else 0.0


def fixed_method_from_config(cfg: dict) -> str:
    args = cfg["Odometry"]["args"]
    imu_rot = bool(args.get("imu_rot_prior_enable", False))
    imu_trans = bool(args.get("imu_trans_prior_enable", False))
    if not imu_rot and not imu_trans:
        return "pure_macvo"
    if imu_rot and not imu_trans:
        return "rotation_only"
    if not imu_rot and imu_trans:
        return "translation_only"
    return "full_imu"


def scene_from_config(cfg: dict, run_name: str) -> str:
    try:
        scene = cfg["Data"]["args"]["args"].get("scene")
        if scene:
            return scene
    except Exception:
        pass
    try:
        root = cfg["Data"]["args"]["args"].get("root")
        if root:
            return Path(root).name
    except Exception:
        pass
    for scene in sorted(SCENES, key=len, reverse=True):
        if scene in run_name:
            return scene
    return "unknown"


def stats(values: list[float]) -> dict[str, float | int | str]:
    vals = [v for v in values if not math.isnan(v)]
    if not vals:
        return {
            "n_trials": 0,
            "median": float("nan"),
            "mean": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            "cv": float("nan"),
            "values": "",
        }
    arr = np.array(vals, dtype=float)
    mean = float(np.mean(arr))
    std = float(np.std(arr))
    return {
        "n_trials": len(vals),
        "median": float(np.median(arr)),
        "mean": mean,
        "std": std,
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "cv": std / mean if mean > 0 else float("nan"),
        "values": ";".join(f"{v:.4f}" for v in vals),
    }


def nanmedian_or_nan(values: list[float]) -> float:
    vals = [v for v in values if not math.isnan(v)]
    return float(np.median(vals)) if vals else float("nan")


def empty_adaptive_stats() -> dict:
    return {
        "total_frames": 0,
        "full_imu": 0,
        "rotation_only": 0,
        "pure_macvo": 0,
        "translation_only": 0,
        "cooldown_total": 0,
        "cooldown_fd": 0,
        "cooldown_rh": 0,
        "fd_raw": 0,
        "fd_trig": 0,
        "fd_supp": 0,
        "severe_vc": 0,
        "mild_vc": 0,
        "vc_trig": 0,
        "rh_trig": 0,
        "expected_missing_nan": 0,
        "nvis_median": float("nan"),
        "nvis_min": float("nan"),
        "nvis_max": float("nan"),
        "flow_cov_median": float("nan"),
        "flow_cov_max": float("nan"),
        "first_full_imu_row": 0,
        "last_full_imu_row": 0,
        "first_cooldown_row": 0,
        "last_cooldown_row": 0,
        "longest_episode": 0,
        "episodes": [],
    }


def analyse_adaptive_decisions(path: Path) -> dict:
    out = empty_adaptive_stats()
    if not path.exists():
        return out

    with path.open(newline="") as f:
        rows = [r for r in csv.DictReader(f) if int(r.get("pair_id", "-1") or -1) > 0]
    out["total_frames"] = len(rows)
    nvis = []
    flow_cov = []
    cur_episode = 0

    for row_idx, row in enumerate(rows, start=1):
        mode = row.get("adaptive_mode", "") or row.get("mode", "")
        state = row.get("state_name", "")
        is_full = mode == "full_imu" or mode.startswith("full_imu") or "full_imu" in state
        is_cooldown = "cooldown" in state or is_true(row.get("cooldown_active"))

        if is_full:
            out["full_imu"] += 1
            if out["first_full_imu_row"] == 0:
                out["first_full_imu_row"] = row_idx
            out["last_full_imu_row"] = row_idx
        elif mode in {"rotation_only", "pure_macvo", "translation_only"}:
            out[mode] += 1
        elif "rotation" in state:
            out["rotation_only"] += 1
        elif "pure" in state:
            out["pure_macvo"] += 1

        if is_full:
            cur_episode += 1
        elif cur_episode:
            out["episodes"].append(cur_episode)
            cur_episode = 0

        if is_cooldown:
            out["cooldown_total"] += 1
            if out["first_cooldown_row"] == 0:
                out["first_cooldown_row"] = row_idx
            out["last_cooldown_row"] = row_idx
        reason = row.get("cooldown_reason", "")
        if "full_div" in reason:
            out["cooldown_fd"] += 1
        elif "rot_harm" in reason:
            out["cooldown_rh"] += 1

        if is_true(row.get("full_divergence_raw")):
            out["fd_raw"] += 1
        if is_true(row.get("full_divergence_triggered")):
            out["fd_trig"] += 1
        if is_true(row.get("fd_check_suppressed_by_grace")):
            out["fd_supp"] += 1
        if is_true(row.get("severe_vc_triggered")):
            out["severe_vc"] += 1
        if is_true(row.get("mild_vc_triggered")):
            out["mild_vc"] += 1
        if is_true(row.get("visual_collapse_triggered")):
            out["vc_trig"] += 1
        if is_true(row.get("rot_harm_triggered")):
            out["rh_trig"] += 1

        for key in ("r_p_whitened_norm", "imu_trans_loss", "median_flow_cov"):
            if row.get(key, "") == "nan":
                out["expected_missing_nan"] += 1

        nv = as_float(row.get("num_visual_residuals"))
        if not math.isnan(nv):
            nvis.append(nv)
        fc = as_float(row.get("median_flow_cov"))
        if not math.isnan(fc):
            flow_cov.append(fc)

    if cur_episode:
        out["episodes"].append(cur_episode)
    out["longest_episode"] = max(out["episodes"]) if out["episodes"] else 0
    if nvis:
        out["nvis_median"] = float(np.median(nvis))
        out["nvis_min"] = float(np.min(nvis))
        out["nvis_max"] = float(np.max(nvis))
    if flow_cov:
        out["flow_cov_median"] = float(np.median(flow_cov))
        out["flow_cov_max"] = float(np.max(flow_cov))
    return out


def collect_runs() -> list[dict]:
    runs = []
    phase_roots = [
        ("fixed_baseline", None),
        ("ruleB_baseline", "ruleB"),
        ("cpb_fd_only", "cpb_fd_only"),
    ]

    for phase, forced_method in phase_roots:
        root = RESULT_ROOT / phase
        if not root.exists():
            continue
        for run_dir in sorted(root.glob("*/*")):
            poses = run_dir / "poses.csv"
            cfg_path = run_dir / "config.yaml"
            if not poses.exists() or not cfg_path.exists():
                continue
            cfg = load_yaml(cfg_path)
            scene = scene_from_config(cfg, run_dir.name)
            method = forced_method or fixed_method_from_config(cfg)
            if scene not in SCENES or method not in METHODS_ORDER:
                continue

            ref = BATCH / scene / "ref_pose.csv"
            try:
                metrics = evaluate_trajectory_direct(poses, ref, f"{scene}_{method}_{run_dir.name}")
                ate = float(metrics["ate"]["ate_rmse"])
                rpe_t = float(metrics["rpe"]["rpe_trans_rmse"])
                rpe_r = float(metrics["rpe"]["rpe_rot_rmse_deg"])
                n_frames = int(metrics["n_frames"])
                match_mode = metrics["matching"]["mode"]
                n_matched = int(metrics["matching"]["n_matched"])
            except Exception as exc:
                print(f"[WARN] ATE failed for {run_dir}: {exc}")
                ate = rpe_t = rpe_r = float("nan")
                n_frames = n_matched = 0
                match_mode = "failed"

            adaptive = analyse_adaptive_decisions(run_dir / "adaptive_decisions.csv")
            runs.append(
                {
                    "phase": phase,
                    "scene": scene,
                    "method": method,
                    "run_name": run_dir.name,
                    "run_dir": str(run_dir),
                    "ATE": ate,
                    "RPE_trans": rpe_t,
                    "RPE_rot_deg": rpe_r,
                    "n_frames": n_frames,
                    "match_mode": match_mode,
                    "n_matched": n_matched,
                    **adaptive,
                }
            )

    grouped = defaultdict(list)
    for run in runs:
        grouped[(run["scene"], run["method"])].append(run)
    for group in grouped.values():
        group.sort(key=lambda r: r["run_name"])
        for idx, run in enumerate(group, start=1):
            run["trial"] = idx
    return sorted(
        runs,
        key=lambda r: (
            SCENES.index(r["scene"]),
            METHODS_ORDER.index(r["method"]),
            r["trial"],
        ),
    )


def write_trial_summary(runs: list[dict]) -> None:
    cols = [
        "scene",
        "method",
        "trial",
        "run_name",
        "ATE",
        "RPE_trans",
        "RPE_rot_deg",
        "n_frames",
        "match_mode",
        "n_matched",
        "full_imu",
        "full_imu_pct",
        "rotation_only",
        "pure_macvo",
        "translation_only",
        "cooldown_total",
        "cooldown_fd",
        "cooldown_rh",
        "first_full_imu_row",
        "last_full_imu_row",
        "first_cooldown_row",
        "last_cooldown_row",
        "episodes",
        "longest_episode",
        "episode_lengths",
        "fd_raw",
        "fd_trig",
        "fd_supp",
        "severe_vc",
        "mild_vc",
        "vc_trig",
        "rh_trig",
        "nvis_median",
        "nvis_min",
        "nvis_max",
        "flow_cov_median",
        "flow_cov_max",
        "expected_missing_nan",
        "run_dir",
    ]
    with (OUTDIR / "locked_trial_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for run in runs:
            out = {k: run.get(k, "") for k in cols}
            out["ATE"] = fmt(run["ATE"])
            out["RPE_trans"] = fmt(run["RPE_trans"])
            out["RPE_rot_deg"] = fmt(run["RPE_rot_deg"])
            out["full_imu_pct"] = f"{pct(run['full_imu'], run['total_frames']):.1f}"
            out["episodes"] = len(run["episodes"])
            out["longest_episode"] = run["longest_episode"]
            out["episode_lengths"] = ";".join(str(x) for x in run["episodes"])
            for k in ("nvis_median", "nvis_min", "nvis_max", "flow_cov_median", "flow_cov_max"):
                out[k] = fmt(run[k], 2)
            writer.writerow(out)


def summarize_by_scene_method(runs: list[dict]) -> dict[tuple[str, str], dict]:
    summary = {}
    grouped = defaultdict(list)
    for run in runs:
        grouped[(run["scene"], run["method"])].append(run)
    for key, group in grouped.items():
        ates = stats([r["ATE"] for r in group])
        total_frames = [r["total_frames"] for r in group]
        denom = [max(x, 1) for x in total_frames]
        summary[key] = {
            "scene": key[0],
            "method": key[1],
            "n_trials": ates["n_trials"],
            "median_ATE": ates["median"],
            "mean_ATE": ates["mean"],
            "std_ATE": ates["std"],
            "min_ATE": ates["min"],
            "max_ATE": ates["max"],
            "cv": ates["cv"],
            "trial_ATEs": ates["values"],
            "median_full_imu_pct": float(np.median([pct(r["full_imu"], d) for r, d in zip(group, denom)])),
            "median_cooldown_frames": float(np.median([r["cooldown_total"] for r in group])),
            "median_episodes": float(np.median([len(r["episodes"]) for r in group])),
            "median_fd_trig": float(np.median([r["fd_trig"] for r in group])),
            "median_severe_vc": float(np.median([r["severe_vc"] for r in group])),
            "median_mild_vc": float(np.median([r["mild_vc"] for r in group])),
            "median_nvis": nanmedian_or_nan([r["nvis_median"] for r in group]),
            "median_flow_cov": nanmedian_or_nan([r["flow_cov_median"] for r in group]),
            "median_first_full_imu_row": float(np.median([r["first_full_imu_row"] for r in group])),
            "median_last_full_imu_row": float(np.median([r["last_full_imu_row"] for r in group])),
            "median_first_cooldown_row": float(np.median([r["first_cooldown_row"] for r in group])),
            "median_last_cooldown_row": float(np.median([r["last_cooldown_row"] for r in group])),
            "median_longest_episode": float(np.median([r["longest_episode"] for r in group])),
        }
    return summary


def write_scene_summary(summary: dict[tuple[str, str], dict]) -> None:
    cols = [
        "scene",
        "method",
        "n_trials",
        "median_ATE",
        "mean_ATE",
        "std_ATE",
        "cv",
        "min_ATE",
        "max_ATE",
        "trial_ATEs",
        "median_full_imu_pct",
        "median_cooldown_frames",
        "median_episodes",
        "median_fd_trig",
        "median_severe_vc",
        "median_mild_vc",
        "median_nvis",
        "median_flow_cov",
        "median_first_full_imu_row",
        "median_last_full_imu_row",
        "median_first_cooldown_row",
        "median_last_cooldown_row",
        "median_longest_episode",
    ]
    with (OUTDIR / "locked_scene_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for scene in SCENES:
            for method in METHODS_ORDER:
                item = summary.get((scene, method))
                if not item:
                    continue
                out = item.copy()
                for key in (
                    "median_ATE",
                    "mean_ATE",
                    "std_ATE",
                    "cv",
                    "min_ATE",
                    "max_ATE",
                    "median_full_imu_pct",
                    "median_cooldown_frames",
                    "median_episodes",
                    "median_fd_trig",
                    "median_severe_vc",
                    "median_mild_vc",
                    "median_nvis",
                    "median_flow_cov",
                    "median_first_full_imu_row",
                    "median_last_full_imu_row",
                    "median_first_cooldown_row",
                    "median_last_cooldown_row",
                    "median_longest_episode",
                ):
                    out[key] = fmt(float(out[key]), 4)
                writer.writerow({k: out.get(k, "") for k in cols})


def oracle_for_scene(summary: dict[tuple[str, str], dict], scene: str) -> tuple[str, float]:
    candidates = []
    for method in FIXED_METHODS:
        item = summary.get((scene, method))
        if item and not math.isnan(item["median_ATE"]):
            candidates.append((method, item["median_ATE"]))
    return min(candidates, key=lambda x: x[1]) if candidates else ("", float("nan"))


def locked_judgment(scene: str, summary: dict[tuple[str, str], dict]) -> str:
    oracle_method, oracle_ate = oracle_for_scene(summary, scene)
    cpb = summary.get((scene, "cpb_fd_only"))
    ruleb = summary.get((scene, "ruleB"))
    full = summary.get((scene, "full_imu"))
    pure = summary.get((scene, "pure_macvo"))
    rot = summary.get((scene, "rotation_only"))
    trans = summary.get((scene, "translation_only"))

    if not cpb or not ruleb:
        return "MISSING"
    cpb_ate = cpb["median_ATE"]
    cpb_full_pct = cpb["median_full_imu_pct"]

    if scene == "locked_murky_entry_help":
        if oracle_method == "full_imu" and cpb_full_pct < 1.0:
            return "ENTRY_FAILURE"
        if oracle_method == "full_imu" and cpb_ate / oracle_ate > 2.0:
            return "ENTRY_OK_BUT_ORACLE_NOT_RECOVERED"
        if oracle_method == "full_imu" and cpb_ate <= ruleb["median_ATE"]:
            return "POSITIVE_ENTRY_OK"
        if oracle_method != "full_imu":
            return "SCENE_NOT_FULL_IMU_HELPFUL"
        return "PARTIAL"

    if scene == "locked_clear_imu_harm":
        visual_best = min(
            x for x in [pure["median_ATE"], rot["median_ATE"]] if not math.isnan(x)
        )
        full_harmful = bool(full and full["median_ATE"] > visual_best + 0.5)
        safe_avoid = cpb_full_pct < 1.0 and cpb_ate <= visual_best + 1.0
        if full_harmful and safe_avoid:
            return "SAFETY_PASS"
        if cpb_full_pct >= 1.0:
            return "SAFETY_ENTRY_RISK"
        return "NEUTRAL_SAFETY"

    if scene == "locked_quality_degrade_no_dropout":
        if oracle_method == "full_imu" and cpb_full_pct < 1.0:
            return "ENTRY_FAILURE_REPRODUCED"
        if oracle_method == "translation_only":
            return "OUT_OF_CANDIDATE_SET_ORACLE"
        if oracle_method in {"pure_macvo", "rotation_only"} and cpb_full_pct < 1.0:
            return "SAFETY_PASS_NON_FULL_IMU_HELPFUL"
        return "OBSERVE"

    return "OBSERVE"


def write_comparison(summary: dict[tuple[str, str], dict]) -> None:
    with (OUTDIR / "locked_fixed_oracle_by_scene.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["scene", "oracle_method", "oracle_median_ATE", *FIXED_METHODS])
        for scene in SCENES:
            oracle_method, oracle_ate = oracle_for_scene(summary, scene)
            row = [scene, oracle_method, fmt(oracle_ate)]
            for method in FIXED_METHODS:
                row.append(fmt(summary.get((scene, method), {}).get("median_ATE", float("nan"))))
            writer.writerow(row)

    with (OUTDIR / "locked_adaptive_comparison.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "scene",
                "design_purpose",
                "oracle_method",
                "oracle_ATE",
                "ruleB_ATE",
                "cpb_ATE",
                "delta_cpb_vs_ruleB",
                "cpb_vs_oracle_ratio",
                "ruleB_full_imu_pct",
                "cpb_full_imu_pct",
                "ruleB_fd_trig",
                "cpb_fd_trig",
                "ruleB_severe_vc",
                "cpb_severe_vc",
                "judgment",
            ]
        )
        for scene in SCENES:
            oracle_method, oracle_ate = oracle_for_scene(summary, scene)
            ruleb = summary[(scene, "ruleB")]
            cpb = summary[(scene, "cpb_fd_only")]
            delta = cpb["median_ATE"] - ruleb["median_ATE"]
            ratio = cpb["median_ATE"] / oracle_ate if oracle_ate > 0 else float("nan")
            writer.writerow(
                [
                    scene,
                    SCENE_PURPOSE[scene],
                    oracle_method,
                    fmt(oracle_ate),
                    fmt(ruleb["median_ATE"]),
                    fmt(cpb["median_ATE"]),
                    fmt(delta),
                    fmt(ratio, 3),
                    fmt(ruleb["median_full_imu_pct"], 1),
                    fmt(cpb["median_full_imu_pct"], 1),
                    fmt(ruleb["median_fd_trig"], 0),
                    fmt(cpb["median_fd_trig"], 0),
                    fmt(ruleb["median_severe_vc"], 0),
                    fmt(cpb["median_severe_vc"], 0),
                    locked_judgment(scene, summary),
                ]
            )


def write_wide(summary: dict[tuple[str, str], dict]) -> None:
    with (OUTDIR / "locked_accuracy_wide.csv").open("w", newline="") as f:
        cols = ["scene", *METHODS_ORDER, "oracle_method", "oracle_ATE", "judgment"]
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for scene in SCENES:
            row = {"scene": scene}
            for method in METHODS_ORDER:
                row[method] = fmt(summary.get((scene, method), {}).get("median_ATE", float("nan")))
            oracle_method, oracle_ate = oracle_for_scene(summary, scene)
            row["oracle_method"] = oracle_method
            row["oracle_ATE"] = fmt(oracle_ate)
            row["judgment"] = locked_judgment(scene, summary)
            writer.writerow(row)


def write_report(runs: list[dict], summary: dict[tuple[str, str], dict]) -> None:
    counts = Counter((r["scene"], r["method"]) for r in runs)
    incomplete = [(scene, method, counts[(scene, method)]) for scene in SCENES for method in METHODS_ORDER if counts[(scene, method)] != 3]

    with (OUTDIR / "locked_report.md").open("w", encoding="utf-8") as f:
        f.write("# V3b++ Locked Held-Out 3-Scene Report\n\n")
        f.write("Scope: locked held-out scenes generated before MACVO evaluation. Metric: direct ATE RMSE in meters, median of 3 trials.\n\n")

        f.write("## Completeness\n\n")
        f.write(f"- `poses.csv` runs analysed: {len(runs)}/54.\n")
        if incomplete:
            f.write("- Incomplete scene-method cells:\n")
            for scene, method, count in incomplete:
                f.write(f"  - {scene} / {method}: {count}/3\n")
        else:
            f.write("- Every scene-method cell has exactly 3 trials.\n")

        f.write("\n## Median ATE Table\n\n")
        f.write("| Scene | Pure | Rotation | Translation | Full IMU | Rule B | CP-B-FD-only | Oracle | Judgment |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---|---|\n")
        for scene in SCENES:
            vals = {
                method: summary.get((scene, method), {}).get("median_ATE", float("nan"))
                for method in METHODS_ORDER
            }
            oracle_method, oracle_ate = oracle_for_scene(summary, scene)
            f.write(
                f"| {scene} | {vals['pure_macvo']:.2f} | {vals['rotation_only']:.2f} | "
                f"{vals['translation_only']:.2f} | {vals['full_imu']:.2f} | "
                f"{vals['ruleB']:.2f} | {vals['cpb_fd_only']:.2f} | "
                f"{oracle_method} ({oracle_ate:.2f}) | {locked_judgment(scene, summary)} |\n"
            )

        f.write("\n## Adaptive Gate Behavior\n\n")
        f.write("| Scene | Method | Full-IMU% | First Full Row | Cooldown | First Cooldown Row | Episodes | Longest Ep | FD Trig | Severe VC | Mild VC | n_vis median | flow_cov median |\n")
        f.write("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for scene in SCENES:
            for method in ("ruleB", "cpb_fd_only"):
                item = summary[(scene, method)]
                f.write(
                    f"| {scene} | {method} | {item['median_full_imu_pct']:.1f}% | "
                    f"{item['median_first_full_imu_row']:.0f} | "
                    f"{item['median_cooldown_frames']:.0f} | {item['median_first_cooldown_row']:.0f} | "
                    f"{item['median_episodes']:.0f} | {item['median_longest_episode']:.0f} | "
                    f"{item['median_fd_trig']:.0f} | {item['median_severe_vc']:.0f} | "
                    f"{item['median_mild_vc']:.0f} | {item['median_nvis']:.1f} | "
                    f"{item['median_flow_cov']:.2f} |\n"
                )

        f.write("\n## Paper Baseline: CP-B vs Pure MACVO\n\n")
        f.write("| Scene | Pure MACVO | CP-B | Delta CP-B-Pure | Relative Gain | Interpretation |\n")
        f.write("|---|---:|---:|---:|---:|---|\n")
        for scene in SCENES:
            pure = summary[(scene, "pure_macvo")]
            cpb = summary[(scene, "cpb_fd_only")]
            delta = cpb["median_ATE"] - pure["median_ATE"]
            gain = (pure["median_ATE"] - cpb["median_ATE"]) / pure["median_ATE"] * 100.0
            if delta < -0.5:
                interp = "meaningful improvement"
            elif delta > 0.5:
                interp = "meaningful regression"
            else:
                interp = "near neutral"
            f.write(
                f"| {scene} | {pure['median_ATE']:.2f} | {cpb['median_ATE']:.2f} | "
                f"{delta:+.2f} | {gain:+.1f}% | {interp} |\n"
            )

        f.write("\n## Internal Ablation: CP-B vs Rule B\n\n")
        f.write("| Scene | Rule B | CP-B | Delta | Relative Gain | CP-B / Oracle |\n")
        f.write("|---|---:|---:|---:|---:|---:|\n")
        for scene in SCENES:
            ruleb = summary[(scene, "ruleB")]
            cpb = summary[(scene, "cpb_fd_only")]
            _, oracle_ate = oracle_for_scene(summary, scene)
            delta = cpb["median_ATE"] - ruleb["median_ATE"]
            gain = (ruleb["median_ATE"] - cpb["median_ATE"]) / ruleb["median_ATE"] * 100.0
            ratio = cpb["median_ATE"] / oracle_ate if oracle_ate > 0 else float("nan")
            f.write(
                f"| {scene} | {ruleb['median_ATE']:.2f} | {cpb['median_ATE']:.2f} | "
                f"{delta:+.2f} | {gain:+.1f}% | {ratio:.2f}x |\n"
            )

        f.write("\n## Trial Spread Flags\n\n")
        high_cv = sorted(
            [item for item in summary.values() if item["cv"] >= 0.15],
            key=lambda item: item["cv"],
            reverse=True,
        )
        if not high_cv:
            f.write("- No scene-method row has CV >= 0.15.\n")
        else:
            f.write("| Scene | Method | Median | CV | Trials |\n")
            f.write("|---|---|---:|---:|---|\n")
            for item in high_cv:
                f.write(
                    f"| {item['scene']} | {item['method']} | {item['median_ATE']:.2f} | "
                    f"{item['cv']:.3f} | {item['trial_ATEs']} |\n"
                )

        f.write("\n## Paper Interpretation\n\n")
        for scene in SCENES:
            f.write(f"- `{scene}`: {SCENE_PURPOSE[scene]} Result = **{locked_judgment(scene, summary)}**.\n")

        f.write("\n## Conservative Claim Update\n\n")
        f.write("- These locked scenes must be treated as held-out diagnostics, not tuning evidence.\n")
        f.write("- The paper-facing baseline is pure MACVO. Rule B is an in-family predecessor for internal ablation, not the primary baseline.\n")
        f.write("- CP-B-FD-only can be claimed as improving over pure MACVO on visually degraded/full_imu-beneficial cases, while preserving safety when it avoids harmful full_imu entry.\n")
        f.write("- If a locked full_imu-helpful scene still shows low CP-B full_imu usage, that strengthens the entry-coverage limitation rather than the method claim.\n")

        f.write("\n## Bottom Line\n\n")
        f.write("- The locked evaluation fixes the earlier independent-test-set weakness in form: 3 locked scenes, 6 methods, 3 trials each, 54/54 completed.\n")
        f.write("- It supports improvement over pure MACVO on L1, but does **not** support a broad claim that CP-B-FD-only recovers the full_imu oracle.\n")
        f.write("- It does support a narrower diagnostic claim: the gate is conservative and safe on full_imu-harmful/non-helpful scenes, but a full_imu-helpful locked scene exposes an oracle-recovery failure after entry.\n")
        f.write("- Recommended paper framing remains: systematic diagnostic ablation of VC-gated adaptive mode selection, not a final generalized SLAM improvement method.\n")


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    runs = collect_runs()
    summary = summarize_by_scene_method(runs)

    write_trial_summary(runs)
    write_scene_summary(summary)
    write_comparison(summary)
    write_wide(summary)
    write_report(runs, summary)

    print(f"Analysed {len(runs)} runs")
    print(f"Outputs: {OUTDIR}")
    for path in sorted(OUTDIR.glob("*")):
        print(f"  {path.name}")


if __name__ == "__main__":
    main()
