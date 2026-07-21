#!/usr/bin/env python3
"""
L1 oracle-recovery audit for V3b++ locked_murky_entry_help.

This is a post-hoc diagnostic analysis only. It does not tune parameters or
change method outputs. It decomposes direct trajectory error over the adaptive
timeline to explain why CP-B enters full_imu but does not recover the fixed
full_imu oracle.
"""

from __future__ import annotations

import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

WORKDIR = Path("/home/admin1/macvo-dev")
sys.path.insert(0, str(WORKDIR))

from Scripts.eval_qa_vif import (  # noqa: E402
    load_macvo_timed_poses,
    load_ref_timed_poses,
    match_by_timestamp_or_index,
)
from Utility.Config import IncludeLoader  # noqa: E402

SCENE = "locked_murky_entry_help"
RESULT_ROOT = WORKDIR / "Results/v3bpp_locked_3scene_54runs"
BATCH = Path("/mnt/e/文档/holoocean/code/recordings/batch_20260618_110853")
OUTDIR = WORKDIR / "analysis_v3bpp_locked_l1_oracle_recovery_audit"

METHODS_ORDER = [
    "pure_macvo",
    "rotation_only",
    "translation_only",
    "full_imu",
    "ruleB",
    "cpb_fd_only",
]

# Row numbers are 1-based adaptive_decisions rows. For frame-indexed trajectory
# errors we use the same integer as the nearest frame index; this is adequate
# for segment-level diagnosis and keeps labels consistent with decision rows.
SEGMENTS = [
    ("all", 0, 1799, "all frames"),
    ("pre_entry", 0, 16, "before first adaptive full_imu row"),
    ("first_full_episode", 17, 1161, "initial long full_imu episode"),
    ("fd_cooldown_window", 1162, 1377, "CP-B FD cooldown window"),
    ("post_cooldown_reentry", 1378, 1799, "CP-B re-entry after cooldown"),
    ("ruleB_long_cooldown_tail", 1162, 1799, "Rule-B cooldown-dominated tail"),
    ("after_entry", 17, 1799, "all frames after adaptive entry"),
]

CHECKPOINTS = [0, 16, 17, 100, 300, 600, 900, 1161, 1162, 1377, 1378, 1500, 1799]


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.load(f, IncludeLoader)


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


def fmt(value: float, ndigits: int = 4) -> str:
    return "" if value is None or math.isnan(value) else f"{value:.{ndigits}f}"


def rmse(values: np.ndarray) -> float:
    if values.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(values.astype(float) ** 2)))


def stats(values: list[float]) -> dict:
    vals = [v for v in values if not math.isnan(v)]
    if not vals:
        return {
            "n": 0,
            "median": float("nan"),
            "mean": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            "values": "",
        }
    arr = np.array(vals, dtype=float)
    return {
        "n": len(vals),
        "median": float(np.median(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "values": ";".join(f"{v:.4f}" for v in vals),
    }


def collect_l1_runs() -> list[dict]:
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
        for run_dir in sorted(root.glob(f"*/*{SCENE}")):
            poses = run_dir / "poses.csv"
            cfg_path = run_dir / "config.yaml"
            if not poses.exists() or not cfg_path.exists():
                continue
            cfg = load_yaml(cfg_path)
            method = forced_method or fixed_method_from_config(cfg)
            if method not in METHODS_ORDER:
                continue
            runs.append(
                {
                    "phase": phase,
                    "method": method,
                    "run_name": run_dir.name,
                    "run_dir": run_dir,
                    "poses": poses,
                    "config": cfg,
                }
            )

    grouped = defaultdict(list)
    for run in runs:
        grouped[run["method"]].append(run)
    for group in grouped.values():
        group.sort(key=lambda r: r["run_name"])
        for trial, run in enumerate(group, start=1):
            run["trial"] = trial

    return sorted(runs, key=lambda r: (METHODS_ORDER.index(r["method"]), r["trial"]))


def load_error_trace(poses_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    est_time, est_poses = load_macvo_timed_poses(poses_path)
    ref_time, ref_poses = load_ref_timed_poses(BATCH / SCENE / "ref_pose.csv")
    est, ref, _ = match_by_timestamp_or_index(est_time, est_poses, ref_time, ref_poses)
    errors = np.linalg.norm(est[:, :3] - ref[:, :3], axis=1)
    return errors, est[:, :3], ref[:, :3]


def load_adaptive_summary(run_dir: Path) -> dict:
    path = run_dir / "adaptive_decisions.csv"
    if not path.exists():
        return {}
    with path.open(newline="") as f:
        rows = [r for r in csv.DictReader(f) if int(r.get("pair_id", "-1") or -1) > 0]
    full_rows = []
    cooldown_rows = []
    reasons = defaultdict(int)
    episodes = []
    current_len = 0
    current_start = None
    for idx, row in enumerate(rows, start=1):
        mode = row.get("adaptive_mode", "")
        state = row.get("state_name", "")
        is_full = mode == "full_imu" or mode.startswith("full_imu") or "full_imu" in state
        is_cooldown = "cooldown" in state or str(row.get("cooldown_active", "")).strip() == "1"
        if is_full:
            full_rows.append(idx)
            if current_len == 0:
                current_start = idx
            current_len += 1
        elif current_len:
            episodes.append((current_start, idx - 1, current_len))
            current_start = None
            current_len = 0
        if is_cooldown:
            cooldown_rows.append(idx)
        reasons[row.get("adaptive_reason", "")] += 1
    if current_len:
        episodes.append((current_start, len(rows), current_len))
    return {
        "full_rows": full_rows,
        "cooldown_rows": cooldown_rows,
        "episodes": episodes,
        "reasons": dict(sorted(reasons.items(), key=lambda kv: -kv[1])),
    }


def build_traces(runs: list[dict]) -> list[dict]:
    traces = []
    for run in runs:
        errors, est_xyz, ref_xyz = load_error_trace(run["poses"])
        adaptive = load_adaptive_summary(run["run_dir"])
        traces.append({**run, "errors": errors, "est_xyz": est_xyz, "ref_xyz": ref_xyz, "adaptive": adaptive})
    return traces


def segment_rows(traces: list[dict]) -> list[dict]:
    rows = []
    for trace in traces:
        errors = trace["errors"]
        for seg_name, start, end, desc in SEGMENTS:
            s = max(0, start)
            e = min(end, len(errors) - 1)
            val = rmse(errors[s : e + 1])
            rows.append(
                {
                    "method": trace["method"],
                    "trial": trace["trial"],
                    "run_name": trace["run_name"],
                    "segment": seg_name,
                    "frame_start": s,
                    "frame_end": e,
                    "segment_desc": desc,
                    "rmse": val,
                }
            )
    return rows


def checkpoint_rows(traces: list[dict]) -> list[dict]:
    rows = []
    for trace in traces:
        errors = trace["errors"]
        for cp in CHECKPOINTS:
            idx = min(cp, len(errors) - 1)
            rows.append(
                {
                    "method": trace["method"],
                    "trial": trace["trial"],
                    "run_name": trace["run_name"],
                    "frame_idx": idx,
                    "position_error": float(errors[idx]),
                }
            )
    return rows


def pairwise_oracle_distance_rows(traces: list[dict]) -> list[dict]:
    """Pair adaptive and fixed trajectories by trial order, then compare positions."""
    by_method = defaultdict(list)
    for trace in traces:
        by_method[trace["method"]].append(trace)
    for group in by_method.values():
        group.sort(key=lambda r: r["trial"])

    pairs = [
        ("cpb_fd_only", "full_imu"),
        ("cpb_fd_only", "translation_only"),
        ("cpb_fd_only", "ruleB"),
        ("ruleB", "full_imu"),
        ("ruleB", "translation_only"),
        ("translation_only", "full_imu"),
    ]
    rows = []
    for left, right in pairs:
        if left not in by_method or right not in by_method:
            continue
        for l_trace, r_trace in zip(by_method[left], by_method[right]):
            n = min(len(l_trace["est_xyz"]), len(r_trace["est_xyz"]))
            dist = np.linalg.norm(l_trace["est_xyz"][:n] - r_trace["est_xyz"][:n], axis=1)
            for seg_name, start, end, desc in SEGMENTS:
                s = max(0, start)
                e = min(end, n - 1)
                rows.append(
                    {
                        "left_method": left,
                        "right_method": right,
                        "trial": l_trace["trial"],
                        "segment": seg_name,
                        "frame_start": s,
                        "frame_end": e,
                        "segment_desc": desc,
                        "position_rmse_between_methods": rmse(dist[s : e + 1]),
                    }
                )
    return rows


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out = row.copy()
            for key, value in list(out.items()):
                if isinstance(value, float):
                    out[key] = fmt(value)
            writer.writerow(out)


def summarize_segments(rows: list[dict]) -> dict[tuple[str, str], dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["method"], row["segment"])].append(row["rmse"])
    return {key: stats(values) for key, values in grouped.items()}


def summarize_checkpoints(rows: list[dict]) -> dict[tuple[str, int], dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["method"], row["frame_idx"])].append(row["position_error"])
    return {key: stats(values) for key, values in grouped.items()}


def summarize_pairwise(rows: list[dict]) -> dict[tuple[str, str, str], dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["left_method"], row["right_method"], row["segment"])].append(
            row["position_rmse_between_methods"]
        )
    return {key: stats(values) for key, values in grouped.items()}


def write_summary_tables(
    seg_summary: dict[tuple[str, str], dict],
    cp_summary: dict[tuple[str, int], dict],
    pair_summary: dict[tuple[str, str, str], dict],
) -> None:
    seg_rows = []
    for method in METHODS_ORDER:
        for seg_name, start, end, desc in SEGMENTS:
            item = seg_summary.get((method, seg_name))
            if not item:
                continue
            seg_rows.append(
                {
                    "method": method,
                    "segment": seg_name,
                    "frame_start": start,
                    "frame_end": end,
                    "median_rmse": item["median"],
                    "mean_rmse": item["mean"],
                    "std_rmse": item["std"],
                    "trial_values": item["values"],
                    "segment_desc": desc,
                }
            )
    write_csv(
        OUTDIR / "l1_segment_error_summary.csv",
        seg_rows,
        [
            "method",
            "segment",
            "frame_start",
            "frame_end",
            "median_rmse",
            "mean_rmse",
            "std_rmse",
            "trial_values",
            "segment_desc",
        ],
    )

    cp_rows = []
    for method in METHODS_ORDER:
        for cp in CHECKPOINTS:
            item = cp_summary.get((method, cp))
            if not item:
                continue
            cp_rows.append(
                {
                    "method": method,
                    "frame_idx": cp,
                    "median_position_error": item["median"],
                    "mean_position_error": item["mean"],
                    "std_position_error": item["std"],
                    "trial_values": item["values"],
                }
            )
    write_csv(
        OUTDIR / "l1_checkpoint_error_summary.csv",
        cp_rows,
        ["method", "frame_idx", "median_position_error", "mean_position_error", "std_position_error", "trial_values"],
    )

    pair_rows = []
    for left, right in [
        ("cpb_fd_only", "full_imu"),
        ("cpb_fd_only", "translation_only"),
        ("cpb_fd_only", "ruleB"),
        ("ruleB", "full_imu"),
        ("ruleB", "translation_only"),
        ("translation_only", "full_imu"),
    ]:
        for seg_name, start, end, desc in SEGMENTS:
            item = pair_summary.get((left, right, seg_name))
            if not item:
                continue
            pair_rows.append(
                {
                    "left_method": left,
                    "right_method": right,
                    "segment": seg_name,
                    "frame_start": start,
                    "frame_end": end,
                    "median_position_rmse_between_methods": item["median"],
                    "trial_values": item["values"],
                    "segment_desc": desc,
                }
            )
    write_csv(
        OUTDIR / "l1_pairwise_method_distance_summary.csv",
        pair_rows,
        [
            "left_method",
            "right_method",
            "segment",
            "frame_start",
            "frame_end",
            "median_position_rmse_between_methods",
            "trial_values",
            "segment_desc",
        ],
    )


def config_comparison(traces: list[dict]) -> list[dict]:
    selected = {}
    for method in ("full_imu", "cpb_fd_only", "ruleB", "translation_only"):
        for trace in traces:
            if trace["method"] == method and trace["trial"] == 1:
                selected[method] = trace["config"]
                break

    keys = [
        ("Odometry.args.imu_rot_prior_enable", lambda c: c["Odometry"]["args"].get("imu_rot_prior_enable")),
        ("Odometry.args.imu_trans_prior_enable", lambda c: c["Odometry"]["args"].get("imu_trans_prior_enable")),
        ("Odometry.args.mapping", lambda c: c["Odometry"]["args"].get("mapping")),
        ("Odometry.optimizer.args.imu_rot_prior", lambda c: c["Odometry"]["optimizer"]["args"].get("imu_rot_prior")),
        (
            "Odometry.optimizer.args.imu_trans_prior_scale",
            lambda c: c["Odometry"]["optimizer"]["args"].get("imu_trans_prior_scale"),
        ),
        (
            "Odometry.optimizer.args.post_imu_fusion_enable",
            lambda c: c["Odometry"]["optimizer"]["args"].get("post_imu_fusion_enable"),
        ),
        ("Odometry.optimizer.args.autodiff", lambda c: c["Odometry"]["optimizer"]["args"].get("autodiff")),
    ]

    rows = []
    for key_name, getter in keys:
        row = {"config_key": key_name}
        for method in ("full_imu", "cpb_fd_only", "ruleB", "translation_only"):
            try:
                row[method] = str(getter(selected[method]))
            except Exception:
                row[method] = ""
        rows.append(row)
    write_csv(
        OUTDIR / "l1_config_key_comparison.csv",
        rows,
        ["config_key", "full_imu", "cpb_fd_only", "ruleB", "translation_only"],
    )
    return rows


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |"]
    out.append("|" + "|".join(["---"] + ["---:"] * (len(headers) - 1)) + "|")
    for row in rows:
        out.append("| " + " | ".join(row) + " |")
    return "\n".join(out) + "\n"


def write_report(
    traces: list[dict],
    seg_summary: dict[tuple[str, str], dict],
    cp_summary: dict[tuple[str, int], dict],
    pair_summary: dict[tuple[str, str, str], dict],
    config_rows: list[dict],
) -> None:
    def seg(method: str, segment: str) -> float:
        return seg_summary[(method, segment)]["median"]

    def cp(method: str, frame_idx: int) -> float:
        return cp_summary[(method, frame_idx)]["median"]

    full_ate = seg("full_imu", "all")
    trans_ate = seg("translation_only", "all")
    ruleb_ate = seg("ruleB", "all")
    cpb_ate = seg("cpb_fd_only", "all")
    cpb_first = seg("cpb_fd_only", "first_full_episode")
    cpb_post = seg("cpb_fd_only", "post_cooldown_reentry")
    cpb_pre = seg("cpb_fd_only", "pre_entry")
    full_pre = seg("full_imu", "pre_entry")
    cpb_at_1162 = cp("cpb_fd_only", 1162)
    cpb_at_1377 = cp("cpb_fd_only", 1377)
    full_at_1162 = cp("full_imu", 1162)
    full_at_1377 = cp("full_imu", 1377)
    cpb_full_distance = pair_summary[("cpb_fd_only", "full_imu", "first_full_episode")]["median"]
    cpb_trans_distance = pair_summary[("cpb_fd_only", "translation_only", "first_full_episode")]["median"]
    cpb_ruleb_first_distance = pair_summary[("cpb_fd_only", "ruleB", "first_full_episode")]["median"]
    cpb_ruleb_post_distance = pair_summary[("cpb_fd_only", "ruleB", "post_cooldown_reentry")]["median"]

    adaptive_rows = []
    for method in ("ruleB", "cpb_fd_only"):
        method_traces = [t for t in traces if t["method"] == method]
        full_pcts = []
        first_full = []
        first_cd = []
        longest = []
        reasons = defaultdict(list)
        for trace in method_traces:
            adaptive = trace["adaptive"]
            n = len(trace["errors"]) - 1
            full_rows = adaptive.get("full_rows", [])
            cooldown_rows = adaptive.get("cooldown_rows", [])
            full_pcts.append(len(full_rows) / max(n, 1) * 100.0)
            first_full.append(full_rows[0] if full_rows else 0)
            first_cd.append(cooldown_rows[0] if cooldown_rows else 0)
            longest.append(max([ep[2] for ep in adaptive.get("episodes", [])], default=0))
            for reason, count in adaptive.get("reasons", {}).items():
                reasons[reason].append(count)
        adaptive_rows.append(
            [
                method,
                f"{np.median(full_pcts):.1f}%",
                f"{np.median(first_full):.0f}",
                f"{np.median(first_cd):.0f}",
                f"{np.median(longest):.0f}",
                ", ".join(f"{k}={int(np.median(v))}" for k, v in sorted(reasons.items())),
            ]
        )

    segment_table_rows = []
    for method in METHODS_ORDER:
        segment_table_rows.append(
            [
                method,
                f"{seg(method, 'all'):.2f}",
                f"{seg(method, 'pre_entry'):.2f}",
                f"{seg(method, 'first_full_episode'):.2f}",
                f"{seg(method, 'fd_cooldown_window'):.2f}",
                f"{seg(method, 'post_cooldown_reentry'):.2f}",
            ]
        )

    checkpoint_table_rows = []
    for frame_idx in CHECKPOINTS:
        checkpoint_table_rows.append(
            [
                str(frame_idx),
                f"{cp('full_imu', frame_idx):.2f}",
                f"{cp('translation_only', frame_idx):.2f}",
                f"{cp('ruleB', frame_idx):.2f}",
                f"{cp('cpb_fd_only', frame_idx):.2f}",
            ]
        )

    pair_table_rows = []
    for left, right in [
        ("cpb_fd_only", "full_imu"),
        ("cpb_fd_only", "translation_only"),
        ("cpb_fd_only", "ruleB"),
        ("ruleB", "full_imu"),
        ("ruleB", "translation_only"),
    ]:
        pair_table_rows.append(
            [
                f"{left} vs {right}",
                f"{pair_summary[(left, right, 'all')]['median']:.2f}",
                f"{pair_summary[(left, right, 'first_full_episode')]['median']:.2f}",
                f"{pair_summary[(left, right, 'post_cooldown_reentry')]['median']:.2f}",
            ]
        )

    with (OUTDIR / "analysis_v3bpp_locked_l1_oracle_recovery_audit.md").open("w", encoding="utf-8") as f:
        f.write("# Locked L1 Oracle-Recovery Causality Audit\n\n")
        f.write("Scene: `locked_murky_entry_help`. Metric: direct position error in meters, median over 3 trials.\n\n")

        f.write("## Executive Finding\n\n")
        f.write(
            "CP-B does **not** fail because it misses the full_imu entry. It enters full_imu at decision row 17 "
            "and stays in full_imu for most of the sequence. The failure is an **oracle-recovery failure after entry**: "
            f"fixed full_imu is {full_ate:.2f}m, while CP-B is {cpb_ate:.2f}m.\n\n"
        )
        f.write(
            "The timeline is sharper than a simple entry-failure story: CP-B is initially contaminated by the first "
            f"rotation-only rows, then partially recovers during the first long full_imu episode. At row 1162 its "
            f"instantaneous error is {cpb_at_1162:.2f}m, even lower than fixed full_imu at the same row "
            f"({full_at_1162:.2f}m). The collapse happens after the FD exit/cooldown window: by row 1377 CP-B is "
            f"{cpb_at_1377:.2f}m while fixed full_imu is {full_at_1377:.2f}m.\n\n"
        )
        f.write(
            "So the immediate mechanism is not lack of entry. It is harmful exit/recovery behavior after the first "
            "full_divergence trigger: the controller leaves a working full_imu trajectory and the subsequent pure/full "
            "switching does not recover the oracle trajectory.\n\n"
        )

        f.write("## Accuracy by Timeline Segment\n\n")
        f.write(
            md_table(
                ["Method", "All", "Pre-Entry 0-16", "First Full 17-1161", "FD Window 1162-1377", "Post Re-Entry 1378-1799"],
                segment_table_rows,
            )
        )

        f.write("## Checkpoint Position Error\n\n")
        f.write(md_table(["Frame", "full_imu", "translation_only", "ruleB", "cpb_fd_only"], checkpoint_table_rows))

        f.write("## Adaptive Timeline\n\n")
        f.write(md_table(["Method", "Full-IMU%", "First Full Row", "First Cooldown Row", "Longest Episode", "Reason Counts"], adaptive_rows))

        f.write("## Trajectory Similarity Between Methods\n\n")
        f.write(
            "This table compares estimated positions between methods, not against GT. Large CP-B vs full_imu distance during "
            "the first full episode means CP-B's full_imu mode is not following the fixed full_imu trajectory.\n\n"
        )
        f.write(md_table(["Pair", "All", "First Full 17-1161", "Post Re-Entry 1378-1799"], pair_table_rows))

        f.write("## Static Config Check\n\n")
        f.write("| Config Key | full_imu | cpb_fd_only | ruleB | translation_only |\n")
        f.write("|---|---:|---:|---:|---:|\n")
        for row in config_rows:
            f.write(
                f"| {row['config_key']} | {row['full_imu']} | {row['cpb_fd_only']} | "
                f"{row['ruleB']} | {row['translation_only']} |\n"
            )

        f.write("\n## Hypothesis Tests\n\n")
        f.write(
            f"- **H1 early rotation-only contamination:** supported as an initial offset, but not sufficient as the final "
            f"cause. Pre-entry CP-B RMSE is {cpb_pre:.2f}m versus fixed full_imu {full_pre:.2f}m. However CP-B "
            f"recovers to {cpb_at_1162:.2f}m by row 1162, so the early offset is not by itself irreversible.\n"
        )
        f.write(
            f"- **H2 FD exit/cooldown is the proximate collapse:** supported. Error jumps from {cpb_at_1162:.2f}m at "
            f"row 1162 to {cpb_at_1377:.2f}m at row 1377. This means the full_divergence exit policy is globally "
            "harmful on L1, even if the internal residual test is locally triggered.\n"
        )
        f.write(
            f"- **H3 CP-B changes pre-cooldown behavior relative to Rule B:** rejected. CP-B and Rule B are almost identical "
            f"during the first full episode (pairwise RMSE {cpb_ruleb_first_distance:.3f}m). CP-B only meaningfully "
            f"diverges from Rule B after cooldown/re-entry (post-cooldown pairwise RMSE {cpb_ruleb_post_distance:.2f}m).\n"
        )
        f.write(
            f"- **H4 shortened FD cooldown fixes L1:** rejected by current evidence. CP-B reduces cooldown and increases "
            f"full_imu usage, but changes Rule B {ruleb_ate:.2f}m to CP-B {cpb_ate:.2f}m, a slight degradation.\n"
        )
        f.write(
            "- **H5 fixed/adaptive config equivalence:** mostly supported for IMU prior keys, but not perfectly clean. "
            "`imu_rot_prior_enable`, `imu_trans_prior_enable`, `imu_rot_prior`, and `imu_trans_prior_scale` match between "
            "fixed full_imu and CP-B. However `mapping` differs in the stored configs (`full_imu=True`, adaptive=False); "
            "this should be documented or controlled with one L1 fixed_full_imu_mapping_false run if the paper needs a "
            "strict config-equivalence claim.\n"
        )

        f.write("\n## Paper Consequence\n\n")
        f.write(
            "The L1 locked result should be written as a new failure mode: **post-entry oracle-recovery failure**. "
            "More specifically, it is an **exit/recovery failure after a successful full_imu entry**. The method can safely "
            "avoid harmful full_imu on L2/L3, but on a full_imu-beneficial scene, entering full_imu is not enough; the "
            "controller must also know when not to leave full_imu. This supports a diagnostic ablation paper, not a final "
            "generalized controller claim.\n"
        )

        f.write("\n## Recommended Next Experiment\n\n")
        f.write(
            "Run one L1-only post-hoc diagnostic: `force_full_imu_after_row_17_no_exit`. This is the cleanest next test. "
            "If it approaches fixed full_imu, the failure is caused by FD exit/cooldown/re-entry logic. If it remains near "
            "40m, adaptive initialization/state reuse differs from fixed full_imu in a deeper way. A secondary hygiene run "
            "is `fixed_full_imu_mapping_false` to remove the stored-config mapping confound.\n"
        )


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    runs = collect_l1_runs()
    traces = build_traces(runs)

    seg = segment_rows(traces)
    checkpoints = checkpoint_rows(traces)
    pairwise = pairwise_oracle_distance_rows(traces)

    write_csv(
        OUTDIR / "l1_segment_error_by_trial.csv",
        seg,
        ["method", "trial", "run_name", "segment", "frame_start", "frame_end", "segment_desc", "rmse"],
    )
    write_csv(
        OUTDIR / "l1_checkpoint_error_by_trial.csv",
        checkpoints,
        ["method", "trial", "run_name", "frame_idx", "position_error"],
    )
    write_csv(
        OUTDIR / "l1_pairwise_method_distance_by_trial.csv",
        pairwise,
        [
            "left_method",
            "right_method",
            "trial",
            "segment",
            "frame_start",
            "frame_end",
            "segment_desc",
            "position_rmse_between_methods",
        ],
    )

    seg_summary = summarize_segments(seg)
    cp_summary = summarize_checkpoints(checkpoints)
    pair_summary = summarize_pairwise(pairwise)
    write_summary_tables(seg_summary, cp_summary, pair_summary)
    cfg_rows = config_comparison(traces)
    write_report(traces, seg_summary, cp_summary, pair_summary, cfg_rows)

    print(f"Analysed {len(traces)} L1 runs")
    print(f"Outputs: {OUTDIR}")
    for path in sorted(OUTDIR.glob("*")):
        print(f"  {path.name}")


if __name__ == "__main__":
    main()
