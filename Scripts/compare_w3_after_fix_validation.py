#!/usr/bin/env python3
"""Build the corrected W=3 validation comparison page.

This is a post-processing script only. It reuses existing joined trajectory CSVs
and does not start MACVO runs.
"""

from __future__ import annotations

import argparse
import html
import sys
from pathlib import Path

import pandas as pd

WORKDIR = Path("/home/admin1/macvo-dev")
if str(WORKDIR) not in sys.path:
    sys.path.insert(0, str(WORKDIR))

from Scripts import analyse_clear_circle_pair_vio as pair_analysis
from Scripts.compare_bias_linearization_fix_closed_paths import (
    load_joined,
    summarize_joined,
    write_summary_csv,
)


SCENE = "clear_rectangle_normal_noise"
DEFAULT_OUT_ROOT = WORKDIR / "analysis_w3_after_fix_validation_comparison_20260709"

METHODS = [
    {
        "method": "pure_macvo_reused",
        "root": WORKDIR / "analysis_closed_paths_latest_20260706_timeinterp",
        "source_variant": "pure_macvo",
    },
    {
        "method": "two_frame_imuatt_reused",
        "root": WORKDIR / "analysis_w2_equivalence_biaslinfix_20260709",
        "source_variant": "vio_preintegrated_full_imuatt_estinit",
    },
    {
        "method": "w2_local_ba_current",
        "root": WORKDIR / "analysis_w2_equivalence_biaslinfix_20260709",
        "source_variant": "vio_local_ba_w2_imuatt",
    },
    {
        "method": "w3_local_ba_current",
        "root": WORKDIR / "analysis_w3_after_fix_validation_20260709",
        "source_variant": "vio_local_ba_w3_imuatt",
    },
    {
        "method": "w3_local_ba_all_optimized",
        "root": WORKDIR / "analysis_w3_after_fix_validation_20260709",
        "source_variant": "vio_local_ba_w3_imuatt_all",
    },
]


def write_scene_readme(scene_dir: Path, rows: list[dict[str, object]]) -> None:
    lines = [
        f"# {SCENE}",
        "",
        "本页面用于验证 bias linearization 和状态同步修正后，W=3 Local BA 是否相对两帧 VIO / W=2 有实际收益。",
        "",
        "| method | frames | RMSE m | median m | final m | max m | t_vel m/s | r_vel deg/s |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
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
    lines.extend(
        [
            "",
            "Artifacts:",
            "- `interactive_trajectory_gt_vs_est.html`",
            "- `trajectory_xy_full_range.png`",
            "- `trajectory_xy_gt_region.png`",
            "- `trajectory_xz_full_range.png`",
            "- `position_error_over_time.png`",
        ]
    )
    (scene_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_index(out_root: Path, rows: list[dict[str, object]], scene_page: Path) -> None:
    lines = [
        "<!doctype html>",
        '<html lang="zh-CN"><head><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        "<title>Corrected W=3 Local BA validation</title>",
        "<style>",
        "body{font-family:Arial,Helvetica,sans-serif;margin:24px;color:#1f2933;background:#f5f7fa}",
        "main{max-width:1200px;margin:auto;background:white;border:1px solid #d8dee6;padding:18px}",
        "table{border-collapse:collapse;width:100%;font-size:13px}",
        "th,td{border:1px solid #d8dee6;padding:6px 8px;text-align:right}",
        "th:first-child,td:first-child,th:nth-child(2),td:nth-child(2){text-align:left}",
        "th{background:#eef2f7}",
        "a{color:#1f6feb;text-decoration:none}",
        "</style></head><body><main>",
        "<h1>Corrected W=3 Local BA validation</h1>",
        "<p>该页面复用已有结果，将 pure MACVO、两帧 imuatt、W=2、W=3 current、W=3 all optimized 放在同一交互轨迹图中。</p>",
        f'<p><a href="{html.escape(str(scene_page.relative_to(out_root)))}">打开交互式轨迹对比</a></p>',
        "<table><thead><tr>",
        "<th>scene</th><th>method</th><th>frames</th><th>RMSE m</th><th>median m</th><th>final m</th><th>max m</th><th>t_vel m/s</th><th>r_vel deg/s</th>",
        "</tr></thead><tbody>",
    ]
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
    lines.extend(["</tbody></table></main></body></html>"])
    (out_root / "index.html").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    scene_dir = args.out_root / SCENE
    traj_dir = scene_dir / "trajectories"
    traj_dir.mkdir(parents=True, exist_ok=True)

    trajectories: dict[str, pd.DataFrame] = {}
    rows: list[dict[str, object]] = []
    for spec in METHODS:
        method = str(spec["method"])
        root = Path(spec["root"])
        source_variant = str(spec["source_variant"])
        df = load_joined(root, SCENE, source_variant)
        source_csv = root / SCENE / "trajectories" / f"{SCENE}_{source_variant}_joined.csv"
        copied_csv = traj_dir / f"{SCENE}_{method}_joined.csv"
        df.to_csv(copied_csv, index=False)
        label = f"{SCENE} / {method}"
        trajectories[label] = df
        row = summarize_joined(SCENE, method, source_csv, df)
        rows.append(row)

    write_summary_csv(scene_dir / "trajectory_summary.csv", rows)
    write_summary_csv(args.out_root / "trajectory_summary.csv", rows)
    pair_analysis.plot_xy(trajectories, scene_dir)
    pair_analysis.plot_xy_gt_region(trajectories, scene_dir)
    pair_analysis.plot_xz(trajectories, scene_dir)
    pair_analysis.plot_error(trajectories, scene_dir)
    pair_analysis.write_interactive_html(trajectories, scene_dir)
    write_scene_readme(scene_dir, rows)
    scene_page = scene_dir / "interactive_trajectory_gt_vs_est.html"
    write_index(args.out_root, rows, scene_page)

    print(f"Wrote {args.out_root / 'index.html'}")
    print(f"{SCENE}: {scene_page}")
    print(f"Summary: {args.out_root / 'trajectory_summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
