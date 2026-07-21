#!/usr/bin/env python3
"""Compare legacy SA-v2, a one-time prior reset, and rank-aware SA-v2."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


ROOT = Path("/home/admin1/macvo-dev")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Scripts.analyze_sa_v2_late_outlier import audit_run, percentile_summary  # noqa: E402
from Scripts.plot_circle_direct_uvd_u1_vs_pose_factor import (  # noqa: E402
    BATCH_ROOT,
    HTML_TEMPLATE,
    MACVO_POSE,
    SCENE,
    metrics,
    position_errors,
    read_forward_axes,
    read_xyz,
    relative_metrics,
    xy_metrics,
    xyz,
)


LEGACY = ROOT / (
    "Results/circle_direct_uvd_sampling_aware_v2_full_20260717/"
    "sampling_aware_v2/trial_1/"
    "vio_two_state_direct_uvd_sampling_aware_v2_full/"
    "clear_circle_truth_normal_noise"
)
RESET = ROOT / (
    "Results/circle_sa_v2_prior_reset_audit_20260717/trial_1/"
    "vio_two_state_direct_uvd_sa_v2_legacy_reset1794/"
    "clear_circle_truth_normal_noise"
)
RANK_AWARE = ROOT / (
    "Results/circle_sa_v2_rank_aware_full_20260717/trial_1/"
    "vio_two_state_direct_uvd_sa_v2_rank_aware/"
    "clear_circle_truth_normal_noise"
)
U1 = ROOT / (
    "Results/circle_normal_noise_direct_uvd_u1_full_20260716/trial_1/"
    "vio_two_state_direct_uvd_u1_standard_full/"
    "clear_circle_truth_normal_noise"
)
OUTPUT = ROOT / "analysis_circle_sa_v2_prior_rank_aware_20260717"


def finite_summary(values: pd.Series) -> dict[str, float | int]:
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=np.float64)
    numeric = numeric[np.isfinite(numeric)]
    if not numeric.size:
        return {"count": 0}
    return {
        "count": int(numeric.size),
        "min": float(np.min(numeric)),
        "median": float(np.median(numeric)),
        "p95": float(np.percentile(numeric, 95)),
        "max": float(np.max(numeric)),
    }


def saved_unique_covariance_summary(run_dir: Path) -> dict[str, object]:
    tensor_map = np.load(run_dir / "tensor_map.npz", allow_pickle=True)
    covariance = np.asarray(
        tensor_map["frames//imu_vio_sa_v2_unique_cov"], dtype=np.float64
    )
    incoming_count = np.asarray(
        tensor_map["frames//imu_vio_sa_v2_incoming_count"], dtype=np.int64
    )
    outgoing_count = np.asarray(
        tensor_map["frames//imu_vio_sa_v2_outgoing_count"], dtype=np.int64
    )
    valid = (incoming_count > 0) | (outgoing_count > 0)
    covariance = 0.5 * (
        covariance[valid] + np.swapaxes(covariance[valid], -1, -2)
    )
    eigenvalues = np.linalg.eigvalsh(covariance)
    ranks = np.count_nonzero(eigenvalues > 1.0e-12, axis=1)
    unique, counts = np.unique(ranks, return_counts=True)
    rank_counts = {
        str(int(rank)): int(count) for rank, count in zip(unique, counts)
    }
    return {
        "valid_edge_count": int(valid.sum()),
        "rank_threshold": 1.0e-12,
        "rank_counts": rank_counts,
        "rank_deficient_edge_count": int(np.count_nonzero(ranks < 9)),
        "minimum_eigenvalue": float(eigenvalues.min()),
        "maximum_eigenvalue": float(eigenvalues.max()),
        "source": "saved tensor_map P_unique with nonzero sampling counts",
    }


def trajectory_difference_summary(
    reference_dir: Path,
    candidate_dir: Path,
) -> dict[str, float | int | None]:
    reference = read_xyz(reference_dir / "poses.csv")
    candidate = read_xyz(candidate_dir / "poses.csv")
    reference_position = np.asarray([row[1:4] for row in reference])
    candidate_position = np.asarray([row[1:4] for row in candidate])
    difference = np.linalg.norm(candidate_position - reference_position, axis=1)
    changed = np.flatnonzero(difference > 1.0e-9)
    first = int(changed[0]) if changed.size else None
    return {
        "position_difference_threshold_m": 1.0e-9,
        "first_different_frame": first,
        "first_difference_m": float(difference[first]) if first is not None else 0.0,
        "difference_at_frame_1794_m": float(difference[1794]),
        "difference_at_frame_1795_m": float(difference[1795]),
        "difference_at_frame_1850_m": float(difference[1850]),
        "maximum_difference_m": float(difference.max()),
        "maximum_difference_frame": int(np.argmax(difference)),
    }


def diagnostics(run_dir: Path, method: str) -> pd.DataFrame:
    frame = pd.read_csv(run_dir / "frame_pair_diagnostics.csv")
    if "method" in frame:
        frame["method"] = method
    else:
        frame.insert(0, "method", method)
    return frame


def column_or_nan(frame: pd.DataFrame, name: str) -> pd.Series:
    if name in frame:
        return frame[name]
    return pd.Series(np.full(len(frame), np.nan), index=frame.index, name=name)


def checkpoint_diagnostics(run_dir: Path, method: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for path in sorted((run_dir / "sa_v2_checkpoints").glob("*.pt")):
        value = torch.load(path, map_location="cpu")
        marginal = value["marginalization_diagnostics"]
        unique = value["unique_covariance_diagnostics"]
        rows.append(
            {
                "method": method,
                "frame_j": int(value["frame_idx"]),
                "prior_reset": bool(value.get("prior_reset", False)),
                "unique_cov_effective_rank": int(unique.effective_rank),
                "unique_cov_min_eigenvalue": float(unique.min_eigenvalue),
                "h_mm_condition_number": float(marginal.h_mm.condition_number),
                "prior_j_condition_number": float(
                    marginal.schur_prior.condition_number
                ),
                "schur_quadratic_relative_error": float(
                    marginal.quadratic_relative_error
                ),
                "common_translation_update_world_norm": float(
                    torch.linalg.vector_norm(
                        value["common_translation_update_world"]
                    )
                ),
                "differential_translation_update_world_norm": float(
                    torch.linalg.vector_norm(
                        value["differential_translation_update_world"]
                    )
                ),
                "rank_aware": bool(value.get("rank_aware_imu_whitening", False)),
                "rank_aware_fallback_active": bool(
                    value.get("rank_aware_fallback_active", False)
                ),
                "imu_residual_dimension": int(
                    value.get("rank_aware_imu_residual_dimension", 9)
                ),
                "checkpoint_path": str(path),
            }
        )
    return pd.DataFrame(rows)


def trajectory_step_summary(run_dir: Path) -> dict[str, float | int]:
    rows = read_xyz(run_dir / "poses.csv")
    positions = np.asarray([[row[1], row[2], row[3]] for row in rows])
    time_s = (np.asarray([row[0] for row in rows]) - rows[0][0]) * 1.0e-9
    step_xy = np.linalg.norm(np.diff(positions[:, :2], axis=0), axis=1)
    late = step_xy[time_s[1:] >= 59.8]
    index = int(np.argmax(step_xy)) + 1
    return {
        "maximum_xy_step_m": float(step_xy[index - 1]),
        "maximum_xy_step_frame": index,
        "maximum_xy_step_time_s": float(time_s[index]),
        "late_static_median_xy_step_m": float(np.median(late)),
        "late_static_p95_xy_step_m": float(np.percentile(late, 95)),
        "late_static_maximum_xy_step_m": float(np.max(late)),
        "late_static_count_over_0p1_m": int(np.count_nonzero(late > 0.1)),
    }


def write_interactive(
    runs: dict[str, Path],
    labels: dict[str, str],
    output: Path,
) -> None:
    gt_path = BATCH_ROOT / SCENE / "ref_pose.csv"
    trajectories = {"GT": read_xyz(gt_path), "pure_macvo": read_xyz(MACVO_POSE)}
    trajectories.update({key: read_xyz(path / "poses.csv") for key, path in runs.items()})
    forwards = {"GT": read_forward_axes(gt_path), "pure_macvo": read_forward_axes(MACVO_POSE)}
    forwards.update(
        {key: read_forward_axes(path / "poses.csv") for key, path in runs.items()}
    )
    timestamps = [row[0] for row in trajectories["GT"]]
    for key, rows in trajectories.items():
        if [row[0] for row in rows] != timestamps:
            raise AssertionError(f"timestamp mismatch for {key}")
    colors = {
        "legacy": "#059669",
        "reset": "#7c3aed",
        "rank_aware": "#dc2626",
        "u1": "#2563eb",
    }
    gt = trajectories["GT"]
    payload = {
        "scene": "Circle / Normal noise / Full 63 s",
        "gt": xyz(gt),
        "gt_forward": forwards["GT"],
        "macvo": xyz(trajectories["pure_macvo"]),
        "macvo_forward": forwards["pure_macvo"],
        "time_s": [(timestamp - timestamps[0]) * 1.0e-9 for timestamp in timestamps],
        "error_m": position_errors(gt, trajectories["pure_macvo"], xy_only=False),
        "metrics": metrics(gt, trajectories["pure_macvo"]),
        "fusion": [
            {
                "key": key,
                "source": key,
                "config": "normal_noise",
                "label": labels[key],
                "color": colors[key],
                "dasharray": "" if key != "reset" else "7 4",
                "scene": SCENE,
                "xyz": xyz(trajectories[key]),
                "forward": forwards[key],
                "error_m": position_errors(gt, trajectories[key], xy_only=False),
                "metrics": metrics(gt, trajectories[key]),
                "path": str(runs[key] / "poses.csv"),
            }
            for key in runs
        ],
        "imu_only": [],
        "gt_path": str(gt_path),
        "macvo_path": str(MACVO_POSE),
    }
    template = HTML_TEMPLATE.replace("{{", "{").replace("}}", "}")
    template = template.replace(
        "Circle, stop-turn rectangle, and straight trajectory comparison",
        "SA-v2 prior reset and rank-aware covariance comparison",
    )
    template = template.replace(
        "__METHOD_SCOPE__",
        "GT, Pure MACVO, Direct-UVD U1, legacy SA-v2, prior-reset audit, and rank-aware SA-v2",
    )
    template = template.replace(
        "__LINE_NOTE__",
        "Timestamp-matched NWU; no alignment, fitting, or scale correction. XY is primary.",
    )
    output.write_text(
        template.replace("__DATA__", json.dumps({"scenes": [payload]}, ensure_ascii=False)),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy", type=Path, default=LEGACY)
    parser.add_argument("--reset", type=Path, default=RESET)
    parser.add_argument("--rank-aware", type=Path, default=RANK_AWARE)
    parser.add_argument("--u1", type=Path, default=U1)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    runs = {
        "legacy": args.legacy,
        "reset": args.reset,
        "rank_aware": args.rank_aware,
    }
    labels = {
        "legacy": "SA-v2 legacy / continuous prior",
        "reset": "SA-v2 legacy / prior reset at 1794",
        "rank_aware": "SA-v2 rank-aware / continuous prior",
        "u1": "Direct UVD factor / U1",
    }
    for run_dir in runs.values():
        for name in ("poses.csv", "frame_pair_diagnostics.csv"):
            if not (run_dir / name).exists():
                raise FileNotFoundError(run_dir / name)
    if not (args.u1 / "poses.csv").exists():
        raise FileNotFoundError(args.u1 / "poses.csv")
    args.output.mkdir(parents=True, exist_ok=True)

    gt_path = BATCH_ROOT / SCENE / "ref_pose.csv"
    gt = read_xyz(gt_path)
    summaries: dict[str, object] = {}
    all_diagnostics: list[pd.DataFrame] = []
    all_checkpoint_diagnostics: list[pd.DataFrame] = []
    rewrite_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []
    for key, run_dir in runs.items():
        frame = diagnostics(run_dir, key)
        all_diagnostics.append(frame)
        checkpoints = checkpoint_diagnostics(run_dir, key)
        all_checkpoint_diagnostics.append(checkpoints)
        rewrite = audit_run(run_dir, key)
        rewrite_rows.extend(rewrite)
        poses = read_xyz(run_dir / "poses.csv")
        metric_rows.append(
            {
                "method": key,
                **metrics(gt, poses),
                **xy_metrics(gt, poses),
                **relative_metrics(gt_path, run_dir / "poses.csv"),
                "estimate_path": str(run_dir / "poses.csv"),
            }
        )
        late_rewrite = percentile_summary(rewrite, 59.8, 63.0)
        summaries[key] = {
            "trajectory_steps": trajectory_step_summary(run_dir),
            "late_static_rewrite": late_rewrite,
            "unique_rank": finite_summary(
                column_or_nan(frame, "imu_vio_sa_v2_unique_cov_effective_rank")
            ),
            "rank_aware_fallback_count": int(
                column_or_nan(
                    frame, "imu_vio_sa_v2_rank_aware_fallback_active"
                ).astype(str).str.lower().isin({"true", "1", "1.0"}).sum()
            ),
            "h_mm_condition": finite_summary(
                column_or_nan(frame, "imu_vio_sa_v2_h_mm_condition_number")
            ),
            "prior_j_condition": finite_summary(
                column_or_nan(frame, "imu_vio_sa_v2_prior_j_condition_number")
            ),
            "schur_quadratic_relative_error": finite_summary(
                column_or_nan(
                    frame, "imu_vio_sa_v2_schur_quadratic_relative_error"
                )
            ),
            "common_translation_update_norm": finite_summary(
                column_or_nan(
                    frame, "imu_vio_sa_v2_common_translation_update_world_norm"
                )
            ),
            "checkpoint_h_mm_condition": finite_summary(
                checkpoints["h_mm_condition_number"]
                if not checkpoints.empty
                else pd.Series(dtype=float)
            ),
            "checkpoint_schur_quadratic_relative_error": finite_summary(
                checkpoints["schur_quadratic_relative_error"]
                if not checkpoints.empty
                else pd.Series(dtype=float)
            ),
            "saved_unique_covariance": saved_unique_covariance_summary(run_dir),
        }

    summaries["reset_effect_vs_legacy"] = trajectory_difference_summary(
        runs["legacy"], runs["reset"]
    )
    summaries["rank_aware_effect_vs_legacy"] = trajectory_difference_summary(
        runs["legacy"], runs["rank_aware"]
    )
    u1_poses = read_xyz(args.u1 / "poses.csv")
    metric_rows.append(
        {
            "method": "u1",
            **metrics(gt, u1_poses),
            **xy_metrics(gt, u1_poses),
            **relative_metrics(gt_path, args.u1 / "poses.csv"),
            "estimate_path": str(args.u1 / "poses.csv"),
        }
    )
    summaries["u1"] = {
        "trajectory_steps": trajectory_step_summary(args.u1),
        "role": "trajectory-only baseline; no SA-v2 prior audit",
    }

    combined = pd.concat(all_diagnostics, ignore_index=True)
    combined.to_csv(args.output / "sa_v2_prior_rank_diagnostics_per_edge.csv", index=False)
    checkpoint_combined = pd.concat(
        all_checkpoint_diagnostics, ignore_index=True
    )
    checkpoint_combined.to_csv(
        args.output / "sa_v2_prior_rank_checkpoint_diagnostics.csv", index=False
    )
    pd.DataFrame(rewrite_rows).to_csv(
        args.output / "sa_v2_prior_rank_rewrite_per_edge.csv", index=False
    )
    with (args.output / "trajectory_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(metric_rows[0]))
        writer.writeheader()
        writer.writerows(metric_rows)
    (args.output / "summary.json").write_text(
        json.dumps(summaries, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    figure, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True)
    for key, color in (("legacy", "#059669"), ("reset", "#7c3aed"), ("rank_aware", "#dc2626")):
        frame = combined[combined["method"] == key]
        checkpoints = checkpoint_combined[
            checkpoint_combined["method"] == key
        ]
        common_values = pd.to_numeric(
            column_or_nan(
                frame, "imu_vio_sa_v2_common_translation_update_world_norm"
            ),
            errors="coerce",
        )
        condition_values = pd.to_numeric(
            column_or_nan(frame, "imu_vio_sa_v2_h_mm_condition_number"),
            errors="coerce",
        )
        if not np.isfinite(common_values).any() and not checkpoints.empty:
            common_x = checkpoints["frame_j"]
            common_values = checkpoints["common_translation_update_world_norm"]
        else:
            common_x = frame["frame_j"]
        if not np.isfinite(condition_values).any() and not checkpoints.empty:
            condition_x = checkpoints["frame_j"]
            condition_values = checkpoints["h_mm_condition_number"]
        else:
            condition_x = frame["frame_j"]
        axes[0].plot(
            common_x,
            common_values,
            label=labels[key],
            color=color,
        )
        axes[1].plot(
            condition_x,
            condition_values,
            label=labels[key],
            color=color,
        )
    axes[0].set_ylabel("common translation update (m)")
    axes[1].set_ylabel("condition(H_mm)")
    axes[1].set_xlabel("frame")
    for axis in axes:
        axis.set_yscale("log")
        axis.axvline(1794, color="#555", linestyle="--", linewidth=1)
        axis.axvline(1850, color="#111", linestyle=":", linewidth=1)
        axis.grid(True, alpha=0.25)
        axis.legend(loc="upper left")
    figure.tight_layout()
    figure.savefig(args.output / "prior_condition_and_common_update.png", dpi=180)
    plt.close(figure)

    write_interactive(
        {**runs, "u1": args.u1},
        labels,
        args.output / "interactive_sa_v2_prior_reset_rank_aware.html",
    )
    print(args.output)


if __name__ == "__main__":
    main()
