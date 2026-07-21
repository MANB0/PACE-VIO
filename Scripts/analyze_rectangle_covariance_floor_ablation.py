#!/usr/bin/env python3
"""Compare corrected static-init fusion with and without a covariance floor."""

from __future__ import annotations

import csv
import json
import math
import statistics
import sys
from pathlib import Path


WORKDIR = Path("/home/admin1/macvo-dev")
if str(WORKDIR) not in sys.path:
    sys.path.insert(0, str(WORKDIR))

from Scripts.evaluate_macvo_relative_metrics import RunSpec, evaluate_run, read_poses  # noqa: E402
from Scripts.plot_static63_gt_macvo import HTML_TEMPLATE, metrics, read_xyz  # noqa: E402


SCENE = "clear_stop_turn_rectangle_truth_no_noise_no_bias"
VISUAL_SCENE = "clear_stop_turn_rectangle_truth_normal_noise"
BATCH_ROOT = Path(
    "/mnt/e/\u6587\u6863/holoocean/code/recordings/"
    "batch_clear_truth_paths_20260713_static63_variants"
)
RESULT_ROOT = WORKDIR / "Results/rectangle_covariance_floor_ablation_20260713"
OUTDIR = WORKDIR / "analysis_rectangle_covariance_floor_ablation_20260713"

GT_PATH = BATCH_ROOT / SCENE / "ref_pose.csv"
PATHS = {
    "pure_macvo": (
        WORKDIR
        / "Results/visual_factor_cache_static63_unique_source_20260713/trial_1/pure_macvo"
        / VISUAL_SCENE
        / "poses.csv"
    ),
    "previous_imuatt_estinit": (
        WORKDIR
        / "Results/static63_cached_imu_fusion_four_configs_20260713/trial_1"
        / "vio_preintegrated_full_imuatt_estinit"
        / SCENE
        / "poses.csv"
    ),
    "staticinit_before_bias_fix": (
        WORKDIR
        / "Results/static63_cached_imu_fusion_staticinit_calibrated_20260713/trial_1"
        / "vio_preintegrated_full_imuatt_staticinit_calibrated"
        / SCENE
        / "poses.csv"
    ),
    "floor_0": (
        RESULT_ROOT
        / "trial_1/vio_preintegrated_full_imuatt_staticinit_calibrated_floor_0"
        / SCENE
        / "poses.csv"
    ),
    "floor_1e-8": (
        RESULT_ROOT
        / "trial_1/vio_preintegrated_full_imuatt_staticinit_calibrated_floor_1e-8"
        / SCENE
        / "poses.csv"
    ),
    "calibrated_imu_only": (
        WORKDIR
        / "Results/static63_calibrated_imu_only_four_configs_20260713/trajectories"
        / f"{SCENE}_imu_only_staticinit_calibrated_poses.csv"
    ),
}

DIAGNOSTICS = {
    "floor_0": RESULT_ROOT
    / "trial_1/vio_preintegrated_full_imuatt_staticinit_calibrated_floor_0"
    / SCENE
    / "frame_pair_diagnostics.csv",
    "floor_1e-8": RESULT_ROOT
    / "trial_1/vio_preintegrated_full_imuatt_staticinit_calibrated_floor_1e-8"
    / SCENE
    / "frame_pair_diagnostics.csv",
}

TRACE_SPECS = (
    ("previous_imuatt_estinit", "Previous imuatt_estinit", "#7c3aed", "8 5"),
    ("staticinit_before_bias_fix", "Staticinit before Bias fix", "#64748b", "4 4"),
    ("floor_0", "Corrected staticinit / floor=0", "#dc2626", ""),
    ("floor_1e-8", "Corrected staticinit / floor=1e-8", "#2563eb", ""),
)


def xyz(rows: list[tuple[int, float, float, float]]) -> list[list[float]]:
    return [[x, y, z] for _, x, y, z in rows]


def position_errors(
    gt: list[tuple[int, float, float, float]],
    est: list[tuple[int, float, float, float]],
) -> list[float]:
    return [
        math.sqrt(
            (est[i][1] - gt[i][1]) ** 2
            + (est[i][2] - gt[i][2]) ** 2
            + (est[i][3] - gt[i][3]) ** 2
        )
        for i in range(min(len(gt), len(est)))
    ]


def path_length(rows: list[tuple[int, float, float, float]]) -> float:
    return sum(
        math.sqrt(
            (b[1] - a[1]) ** 2 + (b[2] - a[2]) ** 2 + (b[3] - a[3]) ** 2
        )
        for a, b in zip(rows, rows[1:])
    )


def rpe_translation_1s(gt_path: Path, est_path: Path) -> tuple[float, float, int]:
    gt = read_poses(gt_path)
    est = read_poses(est_path)
    stamps = sorted(set(gt) & set(est))
    errors: list[float] = []
    for index, stamp_i in enumerate(stamps):
        target = stamp_i + 1_000_000_000
        candidates = stamps[index + 1 :]
        if not candidates:
            continue
        stamp_j = min(candidates, key=lambda value: abs(value - target))
        if abs(stamp_j - target) > 20_000_000:
            continue
        g0, g1 = gt[stamp_i], gt[stamp_j]
        e0, e1 = est[stamp_i], est[stamp_j]
        gt_delta = tuple(g1.p[k] - g0.p[k] for k in range(3))
        est_delta = tuple(e1.p[k] - e0.p[k] for k in range(3))
        # Match the MACVO relative-translation convention used by the project.
        rotated = tuple(
            sum(
                g0.r[row][middle]
                * sum(e0.r[col][middle] * est_delta[col] for col in range(3))
                for middle in range(3)
            )
            for row in range(3)
        )
        errors.append(math.sqrt(sum((gt_delta[k] - rotated[k]) ** 2 for k in range(3))))
    if not errors:
        return math.nan, math.nan, 0
    return (
        math.sqrt(sum(value * value for value in errors) / len(errors)),
        statistics.median(errors),
        len(errors),
    )


def summarize_trajectory(method: str, path: Path, gt_rows) -> dict[str, object]:
    est_rows = read_xyz(path)
    count = min(len(gt_rows), len(est_rows))
    gt_cut = gt_rows[:count]
    est_cut = est_rows[:count]
    absolute = metrics(gt_cut, est_cut)
    relative = evaluate_run(
        RunSpec("static63", SCENE, method, "floor_ablation", "1", path, GT_PATH)
    )
    gt_length = path_length(gt_cut)
    est_length = path_length(est_cut)
    rpe_rmse, rpe_median, rpe_pairs = rpe_translation_1s(GT_PATH, path)
    return {
        "method": method,
        "matched_frames": count,
        "ate_rmse_m": absolute["rmse_m"],
        "ate_mean_m": absolute["mean_m"],
        "ate_final_m": absolute["final_m"],
        "ate_max_m": absolute["max_m"],
        "rpe_translation_rmse_1s_m": rpe_rmse,
        "rpe_translation_median_1s_m": rpe_median,
        "rpe_num_pairs": rpe_pairs,
        "estimated_path_length_m": est_length,
        "gt_path_length_m": gt_length,
        "path_length_ratio": est_length / gt_length,
        "t_rel_m_per_frame": relative["t_rel_m_per_frame"],
        "r_rel_deg_per_frame": relative["r_rel_deg_per_frame"],
        "t_vel_m_s": relative["t_vel_m_s"],
        "r_vel_deg_s": relative["r_vel_deg_s"],
        "source": str(path),
    }


def finite_values(path: Path, column: str) -> list[float]:
    values: list[float] = []
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            try:
                value = float(row[column])
            except (KeyError, TypeError, ValueError):
                continue
            if math.isfinite(value):
                values.append(value)
    return values


def diagnostic_summary() -> list[dict[str, object]]:
    columns = (
        "r_R_whitened_norm",
        "r_p_whitened_norm",
        "r_v_whitened_norm",
        "imu_vio_whitened_norm",
        "imu_vio_cov_trace",
        "imu_vio_weight_trace",
        "imu_vio_weight_diag_min",
        "imu_vio_weight_diag_max",
        "energy_imu_to_visual_ratio",
        "est_velocity_error_norm",
    )
    rows: list[dict[str, object]] = []
    for method, path in DIAGNOSTICS.items():
        for column in columns:
            values = finite_values(path, column)
            if not values:
                continue
            ordered = sorted(values)
            rows.append(
                {
                    "method": method,
                    "metric": column,
                    "count": len(values),
                    "mean": statistics.fmean(values),
                    "median": statistics.median(values),
                    "p90": ordered[min(len(ordered) - 1, int(0.9 * len(ordered)))],
                    "max": max(values),
                }
            )
    return rows


def visual_fingerprint_audit() -> dict[str, object]:
    fingerprints: dict[str, list[str]] = {}
    for method, path in DIAGNOSTICS.items():
        with path.open(newline="", encoding="utf-8") as stream:
            fingerprints[method] = [
                row["visual_input_sha256"] for row in csv.DictReader(stream)
            ]
    return {
        "same_length": len(fingerprints["floor_0"]) == len(fingerprints["floor_1e-8"]),
        "all_equal": fingerprints["floor_0"] == fingerprints["floor_1e-8"],
        "pair_count": min(len(values) for values in fingerprints.values()),
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    required = [GT_PATH, *PATHS.values(), *DIAGNOSTICS.values()]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing floor-ablation inputs:\n" + "\n".join(missing))

    OUTDIR.mkdir(parents=True, exist_ok=True)
    gt_rows = read_xyz(GT_PATH)
    rows = {method: read_xyz(path) for method, path in PATHS.items()}
    frame_count = min(len(gt_rows), *(len(value) for value in rows.values()))
    gt_cut = gt_rows[:frame_count]
    t0 = gt_cut[0][0]

    fusion = []
    for key, label, color, dasharray in TRACE_SPECS:
        estimate = rows[key][:frame_count]
        fusion.append(
            {
                "key": key,
                "source": label,
                "config": "no_noise_no_bias",
                "label": label,
                "color": color,
                "dasharray": dasharray,
                "scene": SCENE,
                "xyz": xyz(estimate),
                "error_m": position_errors(gt_cut, estimate),
                "metrics": metrics(gt_cut, estimate),
                "path": str(PATHS[key]),
            }
        )

    imu_only_rows = rows["calibrated_imu_only"][:frame_count]
    scene_payload = {
        "scene": "Stop-turn rectangle / No bias / no noise",
        "gt": xyz(gt_cut),
        "macvo": xyz(rows["pure_macvo"][:frame_count]),
        "time_s": [(row[0] - t0) / 1e9 for row in gt_cut],
        "error_m": position_errors(gt_cut, rows["pure_macvo"][:frame_count]),
        "metrics": metrics(gt_cut, rows["pure_macvo"][:frame_count]),
        "fusion": fusion,
        "imu_only": [
            {
                "key": "calibrated_imu_only",
                "config": "no_noise_no_bias",
                "label": "IMU-only staticinit calibrated",
                "color": "#059669",
                "scene": SCENE,
                "xyz": xyz(imu_only_rows),
                "error_m": position_errors(gt_cut, imu_only_rows),
                "metrics": metrics(gt_cut, imu_only_rows),
                "path": str(PATHS["calibrated_imu_only"]),
            }
        ],
        "gt_path": str(GT_PATH),
        "macvo_path": str(PATHS["pure_macvo"]),
    }

    trajectory_rows = [
        summarize_trajectory(method, path, gt_rows) for method, path in PATHS.items()
    ]
    diagnostic_rows = diagnostic_summary()
    audit = visual_fingerprint_audit()
    write_csv(OUTDIR / "trajectory_metrics.csv", trajectory_rows)
    write_csv(OUTDIR / "floor_diagnostic_summary.csv", diagnostic_rows)
    (OUTDIR / "visual_input_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    html = HTML_TEMPLATE.replace("{{", "{").replace("}}", "}")
    html = html.replace(
        "Circle, stop-turn rectangle, and straight trajectory comparison",
        "Rectangle covariance-floor ablation",
    )
    html = html.replace(
        "__METHOD_SCOPE__",
        "GT, Pure MACVO, historical fusion, corrected floor variants, and calibrated IMU-only",
    )
    html = html.replace(
        "__LINE_NOTE__",
        "Both floor runs replay the same cached visual factors. ",
    )
    html = html.replace("__DATA__", json.dumps({"scenes": [scene_payload]}, ensure_ascii=False))
    page = OUTDIR / "interactive_trajectory_gt_vs_est.html"
    page.write_text(html, encoding="utf-8")

    print(page)
    print(OUTDIR / "trajectory_metrics.csv")
    print(OUTDIR / "floor_diagnostic_summary.csv")
    print(json.dumps(audit, ensure_ascii=False))


if __name__ == "__main__":
    main()
