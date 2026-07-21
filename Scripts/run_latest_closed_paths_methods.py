#!/usr/bin/env python3
"""Run and analyze the latest four closed-path HoloOcean scenes.

The batch covers:
  - clear_circle_path / clear_rectangle_path
  - normal_noise / zero_noise

It reuses the current MACVO/AIM-VO runner variants and writes one interactive
HTML trajectory viewer per scene plus an index page.
"""

from __future__ import annotations

import argparse
import csv
import html
import sys
import tempfile
import time
from pathlib import Path

WORKDIR = Path("/home/admin1/macvo-dev")
if str(WORKDIR) not in sys.path:
    sys.path.insert(0, str(WORKDIR))

from Scripts import analyse_clear_circle_pair_vio as pair_analysis
from Scripts import run_clear_circle_imu_only_mechanization as imu_mech
from Scripts import run_vio_imu_prior_mode_grid as grid


BATCH_ROOT = Path("/mnt/e/文档/holoocean/code/recordings/batch_zed100_closed_paths_smooth_20260705")
SCENE_ROOTS = {
    "clear_circle_normal_noise": BATCH_ROOT / "normal_noise" / "clear_circle_path",
    "clear_rectangle_normal_noise": BATCH_ROOT / "normal_noise" / "clear_rectangle_path",
    "clear_circle_zero_noise": BATCH_ROOT / "zero_noise" / "clear_circle_path",
    "clear_rectangle_zero_noise": BATCH_ROOT / "zero_noise" / "clear_rectangle_path",
}
DEFAULT_VARIANTS = [
    "pure_macvo",
    "vio_preintegrated_full",
    "vio_preintegrated_full_gtgravity",
    "vio_preintegrated_full_imuatt_estinit",
]
DEFAULT_RESULT_ROOT = WORKDIR / "Results" / "closed_paths_latest_20260706"
DEFAULT_OUTPUT_ROOT = WORKDIR / "analysis_closed_paths_latest_20260706"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--variants", nargs="*", default=DEFAULT_VARIANTS)
    parser.add_argument("--scenes", nargs="*", default=list(SCENE_ROOTS))
    parser.add_argument("--seq-to", type=int, default=None)
    parser.add_argument("--timeout", type=int, default=grid.RUN_TIMEOUT_S)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--run-only", action="store_true")
    parser.add_argument("--analyze-only", action="store_true")
    parser.add_argument("--overwrite-manifest", action="store_true")
    return parser.parse_args()


def run_batch(args: argparse.Namespace) -> int:
    grid.SCENE_ROOTS.update(SCENE_ROOTS)
    specs = grid.build_specs(
        scenes=list(args.scenes),
        variants=list(args.variants),
        trials=1,
        result_root=args.result_root,
    )

    print("=" * 78)
    print("  Latest closed-path MACVO/VIO batch")
    print(f"  Results:  {args.result_root}")
    print(f"  Scenes:   {', '.join(args.scenes)}")
    print(f"  Variants: {', '.join(args.variants)}")
    print(f"  Seq-to:   {args.seq_to if args.seq_to is not None else 'full sequence'}")
    print(f"  Jobs:     {max(1, int(args.jobs))}")
    print(f"  Runs:     {len(specs)}")
    print("=" * 78)

    if not grid.sanity_check(specs):
        return 1

    if args.dry_run:
        print("\nDRY RUN - no MACVO process will be started.")
        for idx, spec in enumerate(specs, start=1):
            marker = "SKIP" if grid.has_completed_run(spec) else "RUN"
            print(
                f"  [{idx:02d}/{len(specs):02d}] {marker} "
                f"scene={spec.scene} variant={spec.variant.name} -> {spec.result_dir}"
            )
        return 0

    args.result_root.mkdir(parents=True, exist_ok=True)
    existing_manifest = args.result_root / "run_manifest.csv"
    if existing_manifest.exists() and not args.overwrite_manifest:
        if not grid.manifest_matches_specs(
            args.result_root,
            specs,
            args.seq_to,
            autodiff=False,
        ):
            print(f"ERROR: existing manifest does not match requested schedule: {existing_manifest}")
            print("Use --overwrite-manifest or choose a fresh --result-root.")
            return 1
    grid.write_manifest_guarded(
        args.result_root,
        specs,
        args.seq_to,
        autodiff=False,
        overwrite=bool(args.overwrite_manifest),
    )
    print(f"\nManifest: {args.result_root / 'run_manifest.csv'}")

    tmpdir = Path(tempfile.mkdtemp(prefix="closed_paths_latest_"))
    print(f"Temp config dir: {tmpdir}")
    odom_cfgs = {
        variant: grid.make_odom_cfg(grid.VARIANTS[variant], tmpdir, autodiff=False)
        for variant in args.variants
    }

    started = time.time()
    failures = grid.execute_run_schedule(
        specs,
        odom_cfgs,
        tmpdir,
        args.result_root,
        timeout_s=int(args.timeout),
        seq_to=args.seq_to,
        jobs=int(args.jobs),
    )
    elapsed = time.time() - started
    print("\n" + "=" * 78)
    print(f"  Attempted schedule in {elapsed / 60:.1f} min")
    print(f"  Results:  {args.result_root}")
    print(f"  Progress: {args.result_root / 'progress.csv'}")
    if failures:
        print(f"  Failures: {len(failures)}")
        for spec, rc in failures:
            print(f"    - scene={spec.scene} variant={spec.variant.name} rc={rc}")
    else:
        print("  No failed return codes in this run.")
    print("=" * 78)
    return 1 if failures else 0


def write_scene_artifacts(
    *,
    output_root: Path,
    scene: str,
    summaries: list[dict[str, object]],
    trajectories: dict[str, object],
) -> Path:
    scene_dir = output_root / scene
    traj_dir = scene_dir / "trajectories"
    traj_dir.mkdir(parents=True, exist_ok=True)

    for label, joined in trajectories.items():
        method = label.split(" / ", 1)[-1].replace("/", "_")
        joined.to_csv(traj_dir / f"{scene}_{method}_joined.csv", index=False)

    imu_mech.write_summary(scene_dir / "trajectory_summary.csv", summaries)
    figure_paths = [
        imu_mech.plot_xy(trajectories, scene_dir, gt_region=False),
        imu_mech.plot_xy(trajectories, scene_dir, gt_region=True),
        imu_mech.plot_xz(trajectories, scene_dir),
        imu_mech.plot_error(trajectories, scene_dir),
    ]
    pair_analysis.write_interactive_html(trajectories, scene_dir)
    figure_paths.append(scene_dir / "interactive_trajectory_gt_vs_est.html")

    lines = [
        f"# {scene}",
        "",
        "| method | frames | RMSE m | median m | final m | max m | t_vel m/s | r_vel deg/s |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(summaries, key=lambda r: str(r["method"])):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["method"]),
                    str(row["matched_frames"]),
                    f"{float(row['ate_rmse_m']):.6f}",
                    f"{float(row['ate_median_m']):.6f}",
                    f"{float(row['ate_final_m']):.6f}",
                    f"{float(row['ate_max_m']):.6f}",
                    f"{float(row['t_vel_m_s']):.6f}",
                    f"{float(row['r_vel_deg_s']):.6f}",
                ]
            )
            + " |"
        )
    lines.extend(["", "Artifacts:"])
    for path in figure_paths:
        lines.append(f"- `{path.relative_to(scene_dir)}`")
    (scene_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return scene_dir / "interactive_trajectory_gt_vs_est.html"


def write_index(
    *,
    output_root: Path,
    summaries: list[dict[str, object]],
    scene_pages: dict[str, Path],
    skipped: list[tuple[str, str, str]],
) -> Path:
    rows = sorted(summaries, key=lambda r: (str(r["scene"]), str(r["method"])))
    lines = [
        "<!doctype html>",
        '<html lang="en"><head><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        "<title>Latest closed-path trajectory results</title>",
        "<style>",
        "body{font-family:Arial,Helvetica,sans-serif;margin:24px;color:#1f2933;background:#f5f7fa}",
        "main{max-width:1180px;margin:auto;background:white;border:1px solid #d8dee6;padding:18px}",
        "table{border-collapse:collapse;width:100%;font-size:13px}",
        "th,td{border:1px solid #d8dee6;padding:6px 8px;text-align:right}",
        "th:first-child,td:first-child,th:nth-child(2),td:nth-child(2){text-align:left}",
        "th{background:#eef2f7}",
        "a{color:#1f6feb;text-decoration:none}",
        "</style></head><body><main>",
        "<h1>Latest closed-path trajectory results</h1>",
        "<p>Each scene page contains an interactive XY/XZ/error viewer.</p>",
        "<h2>Scene pages</h2><ul>",
    ]
    for scene, page in sorted(scene_pages.items()):
        rel = page.relative_to(output_root)
        lines.append(f'<li><a href="{html.escape(str(rel))}">{html.escape(scene)}</a></li>')
    lines.extend(
        [
            "</ul>",
            "<h2>Summary</h2>",
            "<table><thead><tr>",
            "<th>scene</th><th>method</th><th>frames</th><th>RMSE m</th><th>median m</th><th>final m</th><th>max m</th><th>t_vel m/s</th><th>r_vel deg/s</th>",
            "</tr></thead><tbody>",
        ]
    )
    for row in rows:
        lines.append(
            "<tr>"
            f"<td>{html.escape(str(row['scene']))}</td>"
            f"<td>{html.escape(str(row['method']))}</td>"
            f"<td>{int(row['matched_frames'])}</td>"
            f"<td>{float(row['ate_rmse_m']):.6f}</td>"
            f"<td>{float(row['ate_median_m']):.6f}</td>"
            f"<td>{float(row['ate_final_m']):.6f}</td>"
            f"<td>{float(row['ate_max_m']):.6f}</td>"
            f"<td>{float(row['t_vel_m_s']):.6f}</td>"
            f"<td>{float(row['r_vel_deg_s']):.6f}</td>"
            "</tr>"
        )
    lines.append("</tbody></table>")
    if skipped:
        lines.extend(["<h2>Skipped or failed loads</h2><ul>"])
        for scene, variant, reason in skipped:
            lines.append(
                f"<li>{html.escape(scene)} / {html.escape(variant)}: {html.escape(reason)}</li>"
            )
        lines.append("</ul>")
    lines.append("</main></body></html>")

    path = output_root / "index.html"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def analyze_batch(args: argparse.Namespace) -> int:
    output_root: Path = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    rows = pair_analysis.read_manifest(args.result_root)

    all_summaries: list[dict[str, object]] = []
    by_scene_summaries: dict[str, list[dict[str, object]]] = {scene: [] for scene in args.scenes}
    by_scene_trajs: dict[str, dict[str, object]] = {scene: {} for scene in args.scenes}
    skipped: list[tuple[str, str, str]] = []

    for row in rows:
        scene = row["scene"]
        if scene not in by_scene_trajs:
            continue
        try:
            summary, joined = pair_analysis.evaluate_run(row)
        except Exception as exc:
            skipped.append((scene, row.get("variant", ""), str(exc)))
            continue
        converted = imu_mech.evaluate_joined(
            label=str(summary["label"]),
            scene=scene,
            method=str(summary["variant"]),
            joined=joined,
            source=str(summary["poses_path"]),
        )
        all_summaries.append(converted)
        by_scene_summaries[scene].append(converted)
        by_scene_trajs[scene][str(summary["label"])] = joined

    with (output_root / "trajectory_summary.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "scene",
            "method",
            "label",
            "matched_frames",
            "ate_rmse_m",
            "ate_median_m",
            "ate_final_m",
            "ate_max_m",
            "gt_path_length_m",
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
            "source",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sorted(all_summaries, key=lambda r: (str(r["scene"]), str(r["method"]))))

    scene_pages: dict[str, Path] = {}
    for scene in args.scenes:
        if by_scene_trajs[scene]:
            scene_pages[scene] = write_scene_artifacts(
                output_root=output_root,
                scene=scene,
                summaries=by_scene_summaries[scene],
                trajectories=by_scene_trajs[scene],
            )

    index_path = write_index(
        output_root=output_root,
        summaries=all_summaries,
        scene_pages=scene_pages,
        skipped=skipped,
    )
    print(f"Wrote summary: {output_root / 'trajectory_summary.csv'}")
    print(f"Wrote index:   {index_path}")
    for scene, page in sorted(scene_pages.items()):
        print(f"  {scene}: {page}")
    if skipped:
        print("Skipped/failed loads:")
        for scene, variant, reason in skipped:
            print(f"  - {scene} / {variant}: {reason}")
    return 0 if scene_pages else 1


def main() -> int:
    args = parse_args()
    if args.analyze_only and args.run_only:
        print("ERROR: --analyze-only and --run-only cannot be combined.")
        return 1

    rc = 0
    if not args.analyze_only:
        rc = run_batch(args)
        if rc != 0 or args.run_only or args.dry_run:
            return rc
    return analyze_batch(args)


if __name__ == "__main__":
    raise SystemExit(main())
