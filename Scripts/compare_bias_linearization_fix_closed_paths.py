#!/usr/bin/env python3
"""Build interactive before/after trajectory pages for the bias-linearization fix.

The script reuses existing joined trajectory CSVs where possible:

* pure MACVO baseline from the previous time-interpolation batch
* old ``vio_preintegrated_full_imuatt_estinit`` from the same batch
* newly rerun ``vio_preintegrated_full_imuatt_estinit`` after per-edge bias
  linearization was fixed

No MACVO sequence is started here.
"""

from __future__ import annotations

import argparse
import csv
import html
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

WORKDIR = Path("/home/admin1/macvo-dev")
if str(WORKDIR) not in sys.path:
    sys.path.insert(0, str(WORKDIR))

from Scripts import analyse_clear_circle_pair_vio as pair_analysis


DEFAULT_OLD_ROOT = WORKDIR / "analysis_closed_paths_latest_20260706_timeinterp"
DEFAULT_NEW_ROOT = WORKDIR / "analysis_closed_paths_bias_linearized_fix_20260709"
DEFAULT_OUT_ROOT = WORKDIR / "analysis_closed_paths_bias_linearized_fix_comparison_20260709"

SCENES = [
    "clear_circle_normal_noise",
    "clear_rectangle_normal_noise",
    "clear_circle_zero_noise",
    "clear_rectangle_zero_noise",
]

METHODS = [
    {
        "method": "pure_macvo_reused",
        "root_key": "old",
        "source_variant": "pure_macvo",
        "description": "previous pure MACVO result reused from the time-interpolation batch",
    },
    {
        "method": "imuatt_before_biaslin",
        "root_key": "old",
        "source_variant": "vio_preintegrated_full_imuatt_estinit",
        "description": "previous imuatt result before per-edge bias linearization fix",
    },
    {
        "method": "imuatt_biaslin_fix",
        "root_key": "new",
        "source_variant": "vio_preintegrated_full_imuatt_estinit",
        "description": "rerun after per-edge bias linearization fix",
    },
]


def qxyzw_to_rot(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if norm <= 0.0:
        raise ValueError("zero quaternion")
    x, y, z, w = qx / norm, qy / norm, qz / norm, qw / norm
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float64,
    )


def rot_angle_deg(rot: np.ndarray) -> float:
    cos_theta = max(-1.0, min(1.0, 0.5 * (float(np.trace(rot)) - 1.0)))
    return math.degrees(math.acos(cos_theta))


def joined_csv_path(root: Path, scene: str, source_variant: str) -> Path:
    return root / scene / "trajectories" / f"{scene}_{source_variant}_joined.csv"


def load_joined(root: Path, scene: str, source_variant: str) -> pd.DataFrame:
    path = joined_csv_path(root, scene, source_variant)
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    required = {
        "timestamp_ns",
        "tx_est",
        "ty_est",
        "tz_est",
        "qx_est",
        "qy_est",
        "qz_est",
        "qw_est",
        "tx_gt",
        "ty_gt",
        "tz_gt",
        "qx_gt",
        "qy_gt",
        "qz_gt",
        "qw_gt",
        "err_m",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} lacks columns: {sorted(missing)}")
    return df


def relative_metrics(df: pd.DataFrame) -> dict[str, float]:
    stamps = df["timestamp_ns"].to_numpy(np.float64)
    est_pos = df[["tx_est", "ty_est", "tz_est"]].to_numpy(np.float64)
    gt_pos = df[["tx_gt", "ty_gt", "tz_gt"]].to_numpy(np.float64)

    t_errors: list[float] = []
    r_errors_deg: list[float] = []
    t_vel_errors: list[float] = []
    r_vel_errors: list[float] = []

    for idx in range(len(df) - 1):
        dt_s = float(stamps[idx + 1] - stamps[idx]) * 1e-9
        if dt_s <= 0.0:
            continue

        est_r0 = qxyzw_to_rot(
            float(df["qx_est"].iat[idx]),
            float(df["qy_est"].iat[idx]),
            float(df["qz_est"].iat[idx]),
            float(df["qw_est"].iat[idx]),
        )
        est_r1 = qxyzw_to_rot(
            float(df["qx_est"].iat[idx + 1]),
            float(df["qy_est"].iat[idx + 1]),
            float(df["qz_est"].iat[idx + 1]),
            float(df["qw_est"].iat[idx + 1]),
        )
        gt_r0 = qxyzw_to_rot(
            float(df["qx_gt"].iat[idx]),
            float(df["qy_gt"].iat[idx]),
            float(df["qz_gt"].iat[idx]),
            float(df["qw_gt"].iat[idx]),
        )
        gt_r1 = qxyzw_to_rot(
            float(df["qx_gt"].iat[idx + 1]),
            float(df["qy_gt"].iat[idx + 1]),
            float(df["qz_gt"].iat[idx + 1]),
            float(df["qw_gt"].iat[idx + 1]),
        )

        gt_delta = gt_pos[idx + 1] - gt_pos[idx]
        est_delta = est_pos[idx + 1] - est_pos[idx]
        est_delta_in_gt_world = gt_r0 @ est_r0.T @ est_delta
        t_err = float(np.linalg.norm(gt_delta - est_delta_in_gt_world))
        t_errors.append(t_err)
        t_vel_errors.append(t_err / dt_s)

        gt_rel = gt_r0.T @ gt_r1
        est_rel = est_r0.T @ est_r1
        r_err = est_rel.T @ gt_rel
        r_deg = rot_angle_deg(r_err)
        r_errors_deg.append(r_deg)
        r_vel_errors.append(r_deg / dt_s)

    if not t_errors:
        raise ValueError("not enough adjacent pose pairs")

    return {
        "t_rel_m_per_frame": float(np.mean(t_errors)),
        "t_rel_m_per_frame_median": float(np.median(t_errors)),
        "t_rel_m_per_frame_max": float(np.max(t_errors)),
        "r_rel_deg_per_frame": float(np.mean(r_errors_deg)),
        "r_rel_deg_per_frame_median": float(np.median(r_errors_deg)),
        "r_rel_deg_per_frame_max": float(np.max(r_errors_deg)),
        "t_vel_m_s": float(np.mean(t_vel_errors)),
        "t_vel_m_s_median": float(np.median(t_vel_errors)),
        "t_vel_m_s_max": float(np.max(t_vel_errors)),
        "r_vel_deg_s": float(np.mean(r_vel_errors)),
        "r_vel_deg_s_median": float(np.median(r_vel_errors)),
        "r_vel_deg_s_max": float(np.max(r_vel_errors)),
    }


def summarize_joined(scene: str, method: str, source_csv: Path, df: pd.DataFrame) -> dict[str, object]:
    err = df["err_m"].to_numpy(np.float64)
    summary: dict[str, object] = {
        "scene": scene,
        "method": method,
        "matched_frames": int(len(df)),
        "ate_rmse_m": float(np.sqrt(np.mean(err * err))),
        "ate_median_m": float(np.median(err)),
        "ate_final_m": float(err[-1]),
        "ate_max_m": float(np.max(err)),
        "source_csv": str(source_csv),
    }
    summary.update(relative_metrics(df))
    return summary


def write_summary_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "scene",
        "method",
        "matched_frames",
        "ate_rmse_m",
        "ate_median_m",
        "ate_final_m",
        "ate_max_m",
        "t_rel_m_per_frame",
        "r_rel_deg_per_frame",
        "t_vel_m_s",
        "r_vel_deg_s",
        "source_csv",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_scene_readme(scene_dir: Path, scene: str, rows: list[dict[str, object]]) -> None:
    lines = [
        f"# {scene}",
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
            "- `trajectory_xy_gt_vs_est.png`",
            "- `trajectory_xy_gt_region.png`",
            "- `trajectory_xz_gt_vs_est.png`",
            "- `position_error_over_time.png`",
        ]
    )
    (scene_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_index(out_root: Path, rows: list[dict[str, object]], pages: dict[str, Path]) -> None:
    lines = [
        "<!doctype html>",
        '<html lang="zh-CN"><head><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        "<title>Bias linearization fix trajectory comparison</title>",
        "<style>",
        "body{font-family:Arial,Helvetica,sans-serif;margin:24px;color:#1f2933;background:#f5f7fa}",
        "main{max-width:1200px;margin:auto;background:white;border:1px solid #d8dee6;padding:18px}",
        "table{border-collapse:collapse;width:100%;font-size:13px}",
        "th,td{border:1px solid #d8dee6;padding:6px 8px;text-align:right}",
        "th:first-child,td:first-child,th:nth-child(2),td:nth-child(2){text-align:left}",
        "th{background:#eef2f7}",
        "a{color:#1f6feb;text-decoration:none}",
        "</style></head><body><main>",
        "<h1>Bias linearization fix trajectory comparison</h1>",
        "<p>每个场景页面包含 GT、旧 pure MACVO、旧 imuatt、新 bias-linearized imuatt 的交互式轨迹对比。</p>",
        "<h2>Scene pages</h2><ul>",
    ]
    for scene, page in pages.items():
        rel = page.relative_to(out_root)
        lines.append(f'<li><a href="{html.escape(str(rel))}">{html.escape(scene)}</a></li>')
    lines.extend(
        [
            "</ul><h2>Summary</h2>",
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
    lines.extend(["</tbody></table></main></body></html>"])
    (out_root / "index.html").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-root", type=Path, default=DEFAULT_OLD_ROOT)
    parser.add_argument("--new-root", type=Path, default=DEFAULT_NEW_ROOT)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out_root.mkdir(parents=True, exist_ok=True)
    roots = {"old": args.old_root, "new": args.new_root}

    all_rows: list[dict[str, object]] = []
    scene_pages: dict[str, Path] = {}
    for scene in SCENES:
        scene_dir = args.out_root / scene
        traj_dir = scene_dir / "trajectories"
        traj_dir.mkdir(parents=True, exist_ok=True)
        trajectories: dict[str, pd.DataFrame] = {}
        scene_rows: list[dict[str, object]] = []

        for spec in METHODS:
            method = str(spec["method"])
            source_variant = str(spec["source_variant"])
            source_root = roots[str(spec["root_key"])]
            source_csv = joined_csv_path(source_root, scene, source_variant)
            df = load_joined(source_root, scene, source_variant)
            label = f"{scene} / {method}"
            trajectories[label] = df
            df.to_csv(traj_dir / f"{scene}_{method}_joined.csv", index=False)
            row = summarize_joined(scene, method, source_csv, df)
            scene_rows.append(row)
            all_rows.append(row)

        write_summary_csv(scene_dir / "trajectory_summary.csv", scene_rows)
        pair_analysis.plot_xy(trajectories, scene_dir)
        pair_analysis.plot_xy_gt_region(trajectories, scene_dir)
        pair_analysis.plot_xz(trajectories, scene_dir)
        pair_analysis.plot_error(trajectories, scene_dir)
        pair_analysis.write_interactive_html(trajectories, scene_dir)
        write_scene_readme(scene_dir, scene, scene_rows)
        scene_pages[scene] = scene_dir / "interactive_trajectory_gt_vs_est.html"

    write_summary_csv(args.out_root / "trajectory_summary.csv", all_rows)
    write_index(args.out_root, all_rows, scene_pages)
    print(f"Wrote {args.out_root / 'index.html'}")
    for scene, page in scene_pages.items():
        print(f"{scene}: {page}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
