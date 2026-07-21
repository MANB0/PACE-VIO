#!/usr/bin/env python3
"""Compute MAC-VO relative translation/rotation metrics for stored runs.

The metrics follow the MAC-VO paper's adjacent-frame relative error:

    t_rel = mean ||(p_{t+1}-p_t) - R_t Rhat_t^T (phat_{t+1}-phat_t)||
    r_rel = mean angle(Rhat_{t,t+1}^T R_{t,t+1})

Both are computed on exact timestamp matches between poses.csv and ref_pose.csv.
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


OLD16_SOURCE = Path("analysis_aligned_ate_aimvo_paper_v1/aligned_ate_trial_summary.csv")
CLOSED_PATHS_MANIFEST = Path("Results/closed_paths_latest_20260706/run_manifest.csv")
DEFAULT_OUTPUT = Path("analysis_macvo_relative_metrics_20260706")


@dataclass(frozen=True)
class Pose:
    timestamp: int
    p: tuple[float, float, float]
    r: tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]


@dataclass(frozen=True)
class RunSpec:
    dataset: str
    scene: str
    method: str
    source: str
    trial: str
    poses_path: Path
    gt_path: Path
    paper_role: str = ""


def qxyzw_to_rot(qx: float, qy: float, qz: float, qw: float):
    norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if norm == 0:
        raise ValueError("zero quaternion")
    x, y, z, w = qx / norm, qy / norm, qz / norm, qw / norm
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return (
        (1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)),
        (2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)),
        (2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)),
    )


def mat_t(a):
    return (
        (a[0][0], a[1][0], a[2][0]),
        (a[0][1], a[1][1], a[2][1]),
        (a[0][2], a[1][2], a[2][2]),
    )


def mat_mul(a, b):
    return tuple(
        tuple(sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3))
        for i in range(3)
    )


def mat_vec(a, v):
    return tuple(sum(a[i][k] * v[k] for k in range(3)) for i in range(3))


def vec_sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def vec_norm(v):
    return math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])


def rot_angle_rad(r):
    tr = r[0][0] + r[1][1] + r[2][2]
    cos_theta = max(-1.0, min(1.0, (tr - 1.0) * 0.5))
    return math.acos(cos_theta)


def read_poses(path: Path) -> dict[int, Pose]:
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        rows: dict[int, Pose] = {}
        for row in reader:
            ts_key = "timestamp_ns" if "timestamp_ns" in row else "timestamp"
            ts = int(float(row[ts_key]))
            px = float(row["tx"] if "tx" in row else row["x"])
            py = float(row["ty"] if "ty" in row else row["y"])
            pz = float(row["tz"] if "tz" in row else row["z"])
            r = qxyzw_to_rot(float(row["qx"]), float(row["qy"]), float(row["qz"]), float(row["qw"]))
            rows[ts] = Pose(ts, (px, py, pz), r)
    return rows


def evaluate_run(spec: RunSpec) -> dict[str, object]:
    if not spec.poses_path.exists():
        raise FileNotFoundError(spec.poses_path)
    if not spec.gt_path.exists():
        raise FileNotFoundError(spec.gt_path)

    est = read_poses(spec.poses_path)
    gt = read_poses(spec.gt_path)
    stamps = sorted(set(est) & set(gt))
    t_errors: list[float] = []
    r_errors_deg: list[float] = []
    t_vel_errors: list[float] = []
    r_vel_errors_deg_s: list[float] = []

    for a, b in zip(stamps, stamps[1:]):
        g0, g1 = gt[a], gt[b]
        e0, e1 = est[a], est[b]
        dt_s = float(b - a) * 1e-9
        if dt_s <= 0.0:
            continue

        gt_delta = vec_sub(g1.p, g0.p)
        est_delta = vec_sub(e1.p, e0.p)
        est_delta_in_gt_world = mat_vec(g0.r, mat_vec(mat_t(e0.r), est_delta))
        t_err = vec_norm(vec_sub(gt_delta, est_delta_in_gt_world))
        t_errors.append(t_err)
        t_vel_errors.append(t_err / dt_s)

        gt_rel = mat_mul(mat_t(g0.r), g1.r)
        est_rel = mat_mul(mat_t(e0.r), e1.r)
        r_err = mat_mul(mat_t(est_rel), gt_rel)
        r_err_deg = math.degrees(rot_angle_rad(r_err))
        r_errors_deg.append(r_err_deg)
        r_vel_errors_deg_s.append(r_err_deg / dt_s)

    if not t_errors:
        raise ValueError(f"not enough matched timestamps for {spec.poses_path}")

    return {
        "dataset": spec.dataset,
        "scene": spec.scene,
        "paper_role": spec.paper_role,
        "method": spec.method,
        "source": spec.source,
        "trial": spec.trial,
        "n_matched": len(stamps),
        "n_pairs": len(t_errors),
        "t_rel_m_per_frame": statistics.fmean(t_errors),
        "t_rel_m_per_frame_median": statistics.median(t_errors),
        "t_rel_m_per_frame_max": max(t_errors),
        "r_rel_deg_per_frame": statistics.fmean(r_errors_deg),
        "r_rel_deg_per_frame_median": statistics.median(r_errors_deg),
        "r_rel_deg_per_frame_max": max(r_errors_deg),
        "t_vel_m_s": statistics.fmean(t_vel_errors),
        "t_vel_m_s_median": statistics.median(t_vel_errors),
        "t_vel_m_s_max": max(t_vel_errors),
        "r_vel_deg_s": statistics.fmean(r_vel_errors_deg_s),
        "r_vel_deg_s_median": statistics.median(r_vel_errors_deg_s),
        "r_vel_deg_s_max": max(r_vel_errors_deg_s),
        "poses_path": str(spec.poses_path),
        "gt_path": str(spec.gt_path),
    }


def read_old16_specs(path: Path) -> list[RunSpec]:
    specs: list[RunSpec] = []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            specs.append(
                RunSpec(
                    dataset="old16_aimvo_paper_v1",
                    scene=row["scene"],
                    paper_role=row.get("paper_role", ""),
                    method=row["method"],
                    source=row["source"],
                    trial=row["trial"],
                    poses_path=Path(row["poses_path"]),
                    gt_path=Path(row["gt_path"]),
                )
            )
    return specs


def read_closed_path_specs(path: Path) -> list[RunSpec]:
    specs: list[RunSpec] = []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            specs.append(
                RunSpec(
                    dataset="closed_paths_latest_20260706",
                    scene=row["scene"],
                    paper_role="diagnostic",
                    method=row["variant"],
                    source="closed_paths_latest_20260706",
                    trial=row["trial"],
                    poses_path=Path(row["result_dir"]) / "poses.csv",
                    gt_path=Path(row["scene_root"]) / "ref_pose.csv",
                )
            )
    return specs


def write_csv(path: Path, rows: list[dict[str, object]], fields: Iterable[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = rows[0].keys() if rows else []
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fields))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, object]], group_fields: tuple[str, ...]) -> list[dict[str, object]]:
    groups: dict[tuple[object, ...], list[dict[str, object]]] = {}
    for row in rows:
        key = tuple(row[field] for field in group_fields)
        groups.setdefault(key, []).append(row)

    out: list[dict[str, object]] = []
    for key, items in sorted(groups.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        metric_values = {
            "t_rel_m_per_frame": [float(r["t_rel_m_per_frame"]) for r in items],
            "r_rel_deg_per_frame": [float(r["r_rel_deg_per_frame"]) for r in items],
            "t_vel_m_s": [float(r["t_vel_m_s"]) for r in items],
            "r_vel_deg_s": [float(r["r_vel_deg_s"]) for r in items],
        }
        row = {field: value for field, value in zip(group_fields, key)}
        row["n_trials"] = len(items)
        for metric, values in metric_values.items():
            row[f"{metric}_median"] = statistics.median(values)
            row[f"{metric}_mean"] = statistics.fmean(values)
            row[f"{metric}_min"] = min(values)
            row[f"{metric}_max"] = max(values)
            row[f"{metric}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
        out.append(row)
    return out


def markdown_table(rows: list[dict[str, object]], fields: list[str]) -> str:
    lines = ["| " + " | ".join(fields) + " |", "| " + " | ".join(["---"] * len(fields)) + " |"]
    for row in rows:
        vals = []
        for field in fields:
            value = row.get(field, "")
            if isinstance(value, float):
                vals.append(f"{value:.6f}")
            else:
                vals.append(str(value))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines) + "\n"


def write_markdown(path: Path, title: str, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"# {title}\n\n" + markdown_table(rows, fields), encoding="utf-8")


def compare_against_baseline(
    rows: list[dict[str, object]],
    baseline_method: str,
    target_method: str,
    label: str,
) -> list[dict[str, object]]:
    by_scene_method = {(r["scene"], r["method"]): r for r in rows}
    scenes = sorted({str(r["scene"]) for r in rows})
    out: list[dict[str, object]] = []
    for scene in scenes:
        base = by_scene_method.get((scene, baseline_method))
        target = by_scene_method.get((scene, target_method))
        if not base or not target:
            continue
        base_t = float(base["t_rel_m_per_frame_median"])
        target_t = float(target["t_rel_m_per_frame_median"])
        base_r = float(base["r_rel_deg_per_frame_median"])
        target_r = float(target["r_rel_deg_per_frame_median"])
        base_t_vel = float(base["t_vel_m_s_median"])
        target_t_vel = float(target["t_vel_m_s_median"])
        base_r_vel = float(base["r_vel_deg_s_median"])
        target_r_vel = float(target["r_vel_deg_s_median"])
        out.append(
            {
                "comparison": label,
                "scene": scene,
                "paper_role": target.get("paper_role", ""),
                "baseline_method": baseline_method,
                "target_method": target_method,
                "baseline_t_rel_m_per_frame": base_t,
                "target_t_rel_m_per_frame": target_t,
                "delta_t_rel_target_minus_baseline": target_t - base_t,
                "gain_t_rel_pct": (base_t - target_t) / base_t * 100.0 if base_t else 0.0,
                "baseline_r_rel_deg_per_frame": base_r,
                "target_r_rel_deg_per_frame": target_r,
                "delta_r_rel_target_minus_baseline": target_r - base_r,
                "gain_r_rel_pct": (base_r - target_r) / base_r * 100.0 if base_r else 0.0,
                "baseline_t_vel_m_s": base_t_vel,
                "target_t_vel_m_s": target_t_vel,
                "delta_t_vel_target_minus_baseline": target_t_vel - base_t_vel,
                "gain_t_vel_pct": (base_t_vel - target_t_vel) / base_t_vel * 100.0 if base_t_vel else 0.0,
                "baseline_r_vel_deg_s": base_r_vel,
                "target_r_vel_deg_s": target_r_vel,
                "delta_r_vel_target_minus_baseline": target_r_vel - base_r_vel,
                "gain_r_vel_pct": (base_r_vel - target_r_vel) / base_r_vel * 100.0 if base_r_vel else 0.0,
            }
        )
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old16-source", type=Path, default=OLD16_SOURCE)
    parser.add_argument("--closed-paths-manifest", type=Path, default=CLOSED_PATHS_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    specs: list[RunSpec] = []
    specs.extend(read_old16_specs(args.old16_source))
    specs.extend(read_closed_path_specs(args.closed_paths_manifest))

    trial_rows: list[dict[str, object]] = []
    errors: list[dict[str, object]] = []
    for spec in specs:
        try:
            trial_rows.append(evaluate_run(spec))
        except Exception as exc:  # Keep batch evaluation auditable.
            errors.append(
                {
                    "dataset": spec.dataset,
                    "scene": spec.scene,
                    "method": spec.method,
                    "source": spec.source,
                    "trial": spec.trial,
                    "poses_path": str(spec.poses_path),
                    "gt_path": str(spec.gt_path),
                    "error": repr(exc),
                }
            )

    out = args.output
    trial_fields = [
        "dataset",
        "scene",
        "paper_role",
        "method",
        "source",
        "trial",
        "n_matched",
        "n_pairs",
        "t_rel_m_per_frame",
        "t_rel_m_per_frame_median",
        "t_rel_m_per_frame_max",
        "r_rel_deg_per_frame",
        "r_rel_deg_per_frame_median",
        "r_rel_deg_per_frame_max",
        "t_vel_m_s",
        "t_vel_m_s_median",
        "t_vel_m_s_max",
        "r_vel_deg_s",
        "r_vel_deg_s_median",
        "r_vel_deg_s_max",
        "poses_path",
        "gt_path",
    ]
    write_csv(out / "all_trial_relative_metrics.csv", trial_rows, trial_fields)
    write_csv(out / "errors.csv", errors, errors[0].keys() if errors else ["dataset", "scene", "method", "source", "trial", "poses_path", "gt_path", "error"])

    old_rows = [r for r in trial_rows if r["dataset"] == "old16_aimvo_paper_v1"]
    closed_rows = [r for r in trial_rows if r["dataset"] == "closed_paths_latest_20260706"]

    old_summary = summarize(old_rows, ("dataset", "scene", "paper_role", "method", "source"))
    closed_summary = summarize(closed_rows, ("dataset", "scene", "paper_role", "method", "source"))
    all_summary = summarize(trial_rows, ("dataset", "scene", "paper_role", "method", "source"))

    summary_fields = [
        "dataset",
        "scene",
        "paper_role",
        "method",
        "source",
        "n_trials",
        "t_rel_m_per_frame_median",
        "t_rel_m_per_frame_mean",
        "t_rel_m_per_frame_min",
        "t_rel_m_per_frame_max",
        "t_rel_m_per_frame_std",
        "r_rel_deg_per_frame_median",
        "r_rel_deg_per_frame_mean",
        "r_rel_deg_per_frame_min",
        "r_rel_deg_per_frame_max",
        "r_rel_deg_per_frame_std",
        "t_vel_m_s_median",
        "t_vel_m_s_mean",
        "t_vel_m_s_min",
        "t_vel_m_s_max",
        "t_vel_m_s_std",
        "r_vel_deg_s_median",
        "r_vel_deg_s_mean",
        "r_vel_deg_s_min",
        "r_vel_deg_s_max",
        "r_vel_deg_s_std",
    ]
    write_csv(out / "old16_method_summary.csv", old_summary, summary_fields)
    write_csv(out / "closed_paths_method_summary.csv", closed_summary, summary_fields)
    write_csv(out / "all_method_summary.csv", all_summary, summary_fields)

    old_comparison = compare_against_baseline(old_summary, "pure_macvo", "aimvo", "old16_aimvo_vs_pure")
    closed_comparison = compare_against_baseline(
        closed_summary,
        "pure_macvo",
        "vio_preintegrated_full_imuatt_estinit",
        "closed_paths_imuatt_vs_pure",
    )
    comparison_fields = [
        "comparison",
        "scene",
        "paper_role",
        "baseline_method",
        "target_method",
        "baseline_t_rel_m_per_frame",
        "target_t_rel_m_per_frame",
        "delta_t_rel_target_minus_baseline",
        "gain_t_rel_pct",
        "baseline_r_rel_deg_per_frame",
        "target_r_rel_deg_per_frame",
        "delta_r_rel_target_minus_baseline",
        "gain_r_rel_pct",
        "baseline_t_vel_m_s",
        "target_t_vel_m_s",
        "delta_t_vel_target_minus_baseline",
        "gain_t_vel_pct",
        "baseline_r_vel_deg_s",
        "target_r_vel_deg_s",
        "delta_r_vel_target_minus_baseline",
        "gain_r_vel_pct",
    ]
    write_csv(out / "old16_pure_vs_aimvo_comparison.csv", old_comparison, comparison_fields)
    write_csv(out / "closed_paths_pure_vs_imuatt_comparison.csv", closed_comparison, comparison_fields)

    md_fields = [
        "scene",
        "paper_role",
        "method",
        "source",
        "n_trials",
        "t_rel_m_per_frame_median",
        "r_rel_deg_per_frame_median",
        "t_vel_m_s_median",
        "r_vel_deg_s_median",
    ]
    write_markdown(out / "old16_method_summary.md", "Old 16 Scene MAC-VO Relative Metrics", old_summary, md_fields)
    write_markdown(out / "closed_paths_method_summary.md", "Closed Path MAC-VO Relative Metrics", closed_summary, md_fields)
    comparison_md_fields = [
        "scene",
        "paper_role",
        "baseline_t_rel_m_per_frame",
        "target_t_rel_m_per_frame",
        "gain_t_rel_pct",
        "baseline_r_rel_deg_per_frame",
        "target_r_rel_deg_per_frame",
        "gain_r_rel_pct",
        "baseline_t_vel_m_s",
        "target_t_vel_m_s",
        "gain_t_vel_pct",
        "baseline_r_vel_deg_s",
        "target_r_vel_deg_s",
        "gain_r_vel_pct",
    ]
    write_markdown(out / "old16_pure_vs_aimvo_comparison.md", "Old 16 Scene AIM-VO vs Pure MACVO", old_comparison, comparison_md_fields)
    write_markdown(out / "closed_paths_pure_vs_imuatt_comparison.md", "Closed Path IMUAtt vs Pure MACVO", closed_comparison, comparison_md_fields)

    readme = f"""# MAC-VO Relative Metrics Re-evaluation

Generated by `Scripts/evaluate_macvo_relative_metrics.py`.

## Scope

- `old16_aimvo_paper_v1`: `{args.old16_source}`.
  This is the complete 16-scene path manifest currently available for paper-facing `pure_macvo` and `aimvo`/CPB runs.
- `closed_paths_latest_20260706`: `{args.closed_paths_manifest}`.
  This covers the latest generated closed-path diagnostic scenes and their currently completed methods.

## Metrics

- `t_rel_m_per_frame`: mean adjacent-frame relative translation error, in m/frame.
- `r_rel_deg_per_frame`: mean adjacent-frame relative rotation error, in degree/frame.
- `t_vel_m_s`: adjacent-frame relative translation error divided by frame interval, in m/s.
- `r_vel_deg_s`: adjacent-frame relative rotation error divided by frame interval, in degree/s.

Both metrics use exact timestamp matches and no global trajectory alignment.

## Outputs

- `all_trial_relative_metrics.csv`: per-run raw metrics.
- `old16_method_summary.csv` / `.md`: old 16-scene scene-method summaries.
- `closed_paths_method_summary.csv` / `.md`: latest closed-path scene-method summaries.
- `old16_pure_vs_aimvo_comparison.csv` / `.md`: AIM-VO/CPB relative-metric deltas versus pure MACVO.
- `closed_paths_pure_vs_imuatt_comparison.csv` / `.md`: latest IMUAtt relative-metric deltas versus pure MACVO.
- `all_method_summary.csv`: combined summaries.
- `errors.csv`: missing or invalid run records.

## Run Count

- Evaluated runs: {len(trial_rows)}
- Errors: {len(errors)}
"""
    (out / "README.md").write_text(readme, encoding="utf-8")

    print(f"Wrote {out}")
    print(f"Evaluated runs: {len(trial_rows)}")
    print(f"Errors: {len(errors)}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
