#!/usr/bin/env python3
"""Build an interactive before/after comparison for the 9x9 IMU weight change."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

WORKDIR = Path("/home/admin1/macvo-dev")
if str(WORKDIR) not in sys.path:
    sys.path.insert(0, str(WORKDIR))

from Scripts import run_latest_closed_paths_methods as closed_paths


FIELDS = [
    "trial",
    "scene",
    "variant",
    "imu_trans_prior_mode",
    "imu_trans_prior_scale",
    "imu_rot_prior_std",
    "imu_rot_prior_std_when_translation",
    "imu_factor_mode",
    "force_mode",
    "rot_enabled",
    "trans_enabled",
    "scene_root",
    "result_dir",
    "seq_to",
    "autodiff",
    "metadata_camera_imu_time_offset_ns",
    "metadata_time_offset_source",
    "args",
    "created_at",
]

SCENES = [
    "clear_circle_normal_noise",
    "clear_rectangle_normal_noise",
    "clear_circle_zero_noise",
    "clear_rectangle_zero_noise",
]


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def pick_row(rows: list[dict[str, str]], scene: str, variant: str) -> dict[str, str]:
    matches = [r for r in rows if r.get("scene") == scene and r.get("variant") == variant]
    if len(matches) != 1:
        raise RuntimeError(f"expected one row for {scene} / {variant}, got {len(matches)}")
    return matches[0]


def write_comparison_manifest(
    *,
    before_manifest: Path,
    after_manifest: Path,
    result_root: Path,
) -> Path:
    before_rows = read_manifest(before_manifest)
    after_rows = read_manifest(after_manifest)
    result_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    for scene in SCENES:
        specs = [
            (before_rows, "pure_macvo", "pure_macvo_baseline"),
            (before_rows, "vio_preintegrated_full_imuatt_estinit", "imuatt_before_9x9"),
            (after_rows, "vio_preintegrated_full_imuatt_estinit", "imuatt_after_9x9"),
        ]
        for source_rows, source_variant, display_variant in specs:
            source = pick_row(source_rows, scene, source_variant)
            row = {field: source.get(field, "") for field in FIELDS}
            row["variant"] = display_variant
            rows.append(row)

    manifest_path = result_root / "run_manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return manifest_path


def print_before_after_delta(summary_path: Path) -> None:
    with summary_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    by_key = {(row["scene"], row["method"]): row for row in rows}

    print("\nBefore/after 9x9 delta:")
    print(
        "scene,before_ate,after_ate,delta_ate,delta_pct,"
        "before_tvel,after_tvel,before_rvel,after_rvel"
    )
    for scene in SCENES:
        before = by_key[(scene, "imuatt_before_9x9")]
        after = by_key[(scene, "imuatt_after_9x9")]
        before_ate = float(before["ate_rmse_m"])
        after_ate = float(after["ate_rmse_m"])
        delta = after_ate - before_ate
        print(
            f"{scene},{before_ate:.6f},{after_ate:.6f},{delta:.6f},"
            f"{delta / before_ate * 100:.2f}%,"
            f"{float(before['t_vel_m_s']):.6f},{float(after['t_vel_m_s']):.6f},"
            f"{float(before['r_vel_deg_s']):.6f},{float(after['r_vel_deg_s']):.6f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--before-manifest",
        type=Path,
        default=WORKDIR / "Results/closed_paths_latest_20260706_timeinterp/run_manifest.csv",
    )
    parser.add_argument(
        "--after-manifest",
        type=Path,
        default=WORKDIR / "Results/closed_paths_orbstyle_combined_20260706/run_manifest.csv",
    )
    parser.add_argument(
        "--result-root",
        type=Path,
        default=WORKDIR / "Results/orbstyle_before_after_9x9_20260707",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=WORKDIR / "analysis_orbstyle_before_after_9x9_20260707",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = write_comparison_manifest(
        before_manifest=args.before_manifest,
        after_manifest=args.after_manifest,
        result_root=args.result_root,
    )
    print(f"Wrote manifest: {manifest}")
    analyze_args = argparse.Namespace(
        result_root=args.result_root,
        output_root=args.output_root,
        scenes=SCENES,
    )
    rc = closed_paths.analyze_batch(analyze_args)
    if rc == 0:
        print_before_after_delta(args.output_root / "trajectory_summary.csv")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
