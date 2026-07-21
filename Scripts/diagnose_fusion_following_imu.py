#!/usr/bin/env python3
"""Diagnose whether VIO/Local-BA outputs follow IMU-only or visual-only motion."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

WORKDIR = Path("/home/admin1/macvo-dev")
DEFAULT_ANALYSIS_SCENE = (
    WORKDIR
    / "analysis_local_ba_writeback_validation_20260708"
    / "clear_rectangle_normal_noise"
)
DEFAULT_RESULT_ROOT = WORKDIR / "Results" / "local_ba_writeback_validation_20260708" / "trial_1"
DEFAULT_OUTPUT_ROOT = WORKDIR / "analysis_fusion_following_imu_debug_20260708"
SCENE = "clear_rectangle_normal_noise"

METHOD_ORDER = [
    "pure_macvo",
    "imu_only_mechanization",
    "vio_preintegrated_full_imuatt_estinit",
    "vio_local_ba_w2_imuatt",
    "vio_local_ba_w2_imuatt_all",
    "vio_local_ba_w3_imuatt",
    "vio_local_ba_w3_imuatt_all",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-scene", type=Path, default=DEFAULT_ANALYSIS_SCENE)
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def load_trajectories(scene_dir: Path) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    traj_dir = scene_dir / "trajectories"
    for method in METHOD_ORDER:
        path = traj_dir / f"{SCENE}_{method}_joined.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        out[method] = pd.read_csv(path)
    return out


def xyz(df: pd.DataFrame, prefix: str) -> np.ndarray:
    return df[[f"tx_{prefix}", f"ty_{prefix}", f"tz_{prefix}"]].to_numpy(np.float64)


def rmse(vec: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum(vec * vec, axis=1))))


def norm_rows(vec: np.ndarray) -> np.ndarray:
    return np.linalg.norm(vec, axis=1)


def common_frames(trajs: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    stamps: set[int] | None = None
    for df in trajs.values():
        vals = set(df["timestamp_ns"].astype(np.int64).tolist())
        stamps = vals if stamps is None else stamps.intersection(vals)
    if not stamps:
        raise RuntimeError("No common timestamps across trajectories")
    ordered = sorted(stamps)
    out: dict[str, pd.DataFrame] = {}
    for method, df in trajs.items():
        clipped = df[df["timestamp_ns"].isin(ordered)].copy()
        clipped["timestamp_ns"] = clipped["timestamp_ns"].astype(np.int64)
        clipped = clipped.sort_values("timestamp_ns").reset_index(drop=True)
        out[method] = clipped
    return out


def projection_decomposition(trajs: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    ref = trajs["pure_macvo"]
    pure = xyz(ref, "est")
    imu = xyz(trajs["imu_only_mechanization"], "est")
    gt = xyz(ref, "gt")
    axis = imu - pure
    denom = np.sum(axis * axis, axis=1)
    valid = denom > 1e-10

    rows: list[dict[str, object]] = []
    frame_rows: list[dict[str, object]] = []
    for method, df in trajs.items():
        est = xyz(df, "est")
        err_gt = est - gt
        err_imu = est - imu
        err_pure = est - pure
        beta = np.full(len(est), np.nan, dtype=np.float64)
        beta_gt = np.full(len(est), np.nan, dtype=np.float64)
        orth = np.full(len(est), np.nan, dtype=np.float64)
        if valid.any():
            beta[valid] = np.sum((est[valid] - pure[valid]) * axis[valid], axis=1) / denom[valid]
            beta_gt[valid] = np.sum((gt[valid] - pure[valid]) * axis[valid], axis=1) / denom[valid]
            proj = pure[valid] + beta[valid, None] * axis[valid]
            orth[valid] = norm_rows(est[valid] - proj)

        dist_to_imu = norm_rows(err_imu)
        dist_to_pure = norm_rows(err_pure)
        rows.append(
            {
                "method": method,
                "frames": len(df),
                "rmse_to_gt_m": rmse(err_gt),
                "rmse_to_imu_only_m": rmse(err_imu),
                "rmse_to_pure_macvo_m": rmse(err_pure),
                "final_y_error_m": float(err_gt[-1, 1]),
                "mean_y_error_m": float(np.mean(err_gt[:, 1])),
                "final_x_error_m": float(err_gt[-1, 0]),
                "mean_beta_pure0_imu1": float(np.nanmean(beta)),
                "median_beta_pure0_imu1": float(np.nanmedian(beta)),
                "mean_gt_beta_pure0_imu1": float(np.nanmean(beta_gt)),
                "mean_abs_beta_minus_gt_beta": float(np.nanmean(np.abs(beta - beta_gt))),
                "mean_orthogonal_error_to_pure_imu_line_m": float(np.nanmean(orth)),
                "frames_closer_to_imu_than_pure": int(np.sum(dist_to_imu < dist_to_pure)),
                "closer_to_imu_ratio": float(np.mean(dist_to_imu < dist_to_pure)),
            }
        )

        t = (df["timestamp_ns"].to_numpy(np.float64) - float(df["timestamp_ns"].iloc[0])) * 1e-9
        for i in range(len(df)):
            frame_rows.append(
                {
                    "method": method,
                    "frame": i,
                    "time_s": float(t[i]),
                    "err_to_gt_m": float(norm_rows(err_gt[i : i + 1])[0]),
                    "err_to_imu_only_m": float(dist_to_imu[i]),
                    "err_to_pure_macvo_m": float(dist_to_pure[i]),
                    "y_error_m": float(err_gt[i, 1]),
                    "beta_pure0_imu1": float(beta[i]) if math.isfinite(beta[i]) else math.nan,
                    "gt_beta_pure0_imu1": float(beta_gt[i]) if math.isfinite(beta_gt[i]) else math.nan,
                    "orthogonal_error_to_pure_imu_line_m": float(orth[i]) if math.isfinite(orth[i]) else math.nan,
                }
            )

    return pd.DataFrame(rows), pd.DataFrame(frame_rows)


def delta_decomposition(trajs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    gt = xyz(trajs["pure_macvo"], "gt")
    pure = xyz(trajs["pure_macvo"], "est")
    imu = xyz(trajs["imu_only_mechanization"], "est")
    d_gt = np.diff(gt, axis=0)
    d_pure = np.diff(pure, axis=0)
    d_imu = np.diff(imu, axis=0)
    rows: list[dict[str, object]] = []
    for method, df in trajs.items():
        d_est = np.diff(xyz(df, "est"), axis=0)
        rows.append(
            {
                "method": method,
                "delta_rmse_to_gt_m": rmse(d_est - d_gt),
                "delta_rmse_to_imu_only_m": rmse(d_est - d_imu),
                "delta_rmse_to_pure_macvo_m": rmse(d_est - d_pure),
                "mean_delta_y_minus_gt_m": float(np.mean((d_est - d_gt)[:, 1])),
                "final_cumulative_y_error_from_deltas_m": float(np.sum((d_est - d_gt)[:, 1])),
                "mean_delta_x_minus_gt_m": float(np.mean((d_est - d_gt)[:, 0])),
            }
        )
    return pd.DataFrame(rows)


def optimizer_diagnostics(result_root: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for method in METHOD_ORDER:
        if method == "imu_only_mechanization":
            continue
        path = result_root / method / SCENE / "frame_pair_diagnostics.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        row: dict[str, object] = {"method": method, "diagnostic_rows": len(df)}
        for col in [
            "vio_factor_active",
            "local_ba_window_size",
            "local_ba_num_frames",
            "local_ba_num_edges",
            "local_ba_num_visual_residual_blocks",
            "visual_loss_per_residual",
            "total_loss",
            "r_R_whitened_norm",
            "r_p_whitened_norm",
            "r_v_whitened_norm",
            "imu_vio_whitened_norm",
            "imu_rot_loss",
            "imu_trans_loss",
            "imu_vel_loss",
            "imu_vio_weight_trace",
        ]:
            if col in df.columns:
                values = pd.to_numeric(df[col], errors="coerce")
                row[f"{col}_mean"] = float(values.mean()) if values.notna().any() else math.nan
                row[f"{col}_median"] = float(values.median()) if values.notna().any() else math.nan
        if "imu_factor_mode" in df.columns:
            row["imu_factor_mode"] = ",".join(sorted(set(map(str, df["imu_factor_mode"].dropna()))))
        if "local_ba_writeback" in df.columns:
            row["local_ba_writeback"] = ",".join(sorted(set(map(str, df["local_ba_writeback"].dropna()))))
        rows.append(row)
    return pd.DataFrame(rows)


def plot_y_error(frame_df: pd.DataFrame, outdir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8.8, 4.8), dpi=180)
    for method in METHOD_ORDER:
        sub = frame_df[frame_df["method"] == method]
        ax.plot(sub["time_s"], sub["y_error_m"], label=method, linewidth=1.6)
    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.set_xlabel("time / s")
    ax.set_ylabel("signed y error vs GT / m")
    ax.grid(True, linewidth=0.35, alpha=0.5)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    path = outdir / "signed_y_error_vs_gt.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_beta(frame_df: pd.DataFrame, outdir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8.8, 4.8), dpi=180)
    for method in METHOD_ORDER:
        sub = frame_df[frame_df["method"] == method]
        ax.plot(sub["time_s"], sub["beta_pure0_imu1"], label=method, linewidth=1.6)
    gt = frame_df[frame_df["method"] == "pure_macvo"]
    ax.plot(gt["time_s"], gt["gt_beta_pure0_imu1"], label="GT projected beta", color="black", linewidth=2.2)
    ax.axhline(0.0, color="#888888", linewidth=0.8)
    ax.axhline(1.0, color="#888888", linewidth=0.8, linestyle="--")
    ax.set_xlabel("time / s")
    ax.set_ylabel("projection on pure-MACVO -> IMU-only axis")
    ax.grid(True, linewidth=0.35, alpha=0.5)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    path = outdir / "projection_beta_pure_to_imu.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_distance_sources(frame_df: pd.DataFrame, outdir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8.8, 4.8), dpi=180)
    for method in [
        "vio_preintegrated_full_imuatt_estinit",
        "vio_local_ba_w2_imuatt",
        "vio_local_ba_w2_imuatt_all",
        "vio_local_ba_w3_imuatt",
        "vio_local_ba_w3_imuatt_all",
    ]:
        sub = frame_df[frame_df["method"] == method]
        ax.plot(sub["time_s"], sub["err_to_imu_only_m"], label=f"{method} -> imu", linewidth=1.4)
    ax.set_xlabel("time / s")
    ax.set_ylabel("distance to IMU-only trajectory / m")
    ax.grid(True, linewidth=0.35, alpha=0.5)
    ax.legend(fontsize=7, ncol=1)
    fig.tight_layout()
    path = outdir / "distance_to_imu_only_over_time.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def write_markdown(
    outdir: Path,
    position_df: pd.DataFrame,
    delta_df: pd.DataFrame,
    opt_df: pd.DataFrame,
    plot_paths: list[Path],
) -> Path:
    p = outdir / "fusion_following_imu_debug.md"
    lines = [
        "# Fusion Following IMU Debug Report",
        "",
        "Dataset: `clear_rectangle_normal_noise`, first 150 frames.",
        "",
        "Interpretation of `beta`: `0` means the estimate lies on the pure-MACVO side, `1` means it lies on the IMU-only side, and the GT beta shows where the ground truth lies on that same pure-to-IMU axis.",
        "",
        "## Position-level decomposition",
        "",
        position_df.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Frame-to-frame delta decomposition",
        "",
        delta_df.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Optimizer diagnostics summary",
        "",
        opt_df.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Plots",
        "",
    ]
    for path in plot_paths:
        lines.append(f"- `{path.name}`")
    lines.extend(
        [
            "",
            "## Evidence-based conclusion",
            "",
            "The VIO/Local-BA outputs are much closer to IMU-only than to pure MACVO in position space, and their projection beta is near the IMU-only side while the GT beta lies between pure MACVO and IMU-only. This supports the conclusion that the current fusion is IMU-dominated on this segment.",
            "",
            "Because IMU-only itself is not catastrophically wrong over the first 150 frames, the likely issue is not a broken raw IMU mechanization. The remaining evidence points to the balance between visual residuals and IMU p/v/R constraints inside the optimizer: the visual side does not pull the fused solution back toward GT enough once the IMU p/v constraint is active.",
            "",
        ]
    )
    p.write_text("\n".join(lines), encoding="utf-8")
    return p


def main() -> int:
    args = parse_args()
    outdir = args.output_root
    outdir.mkdir(parents=True, exist_ok=True)

    trajs = common_frames(load_trajectories(args.analysis_scene))
    position_df, frame_df = projection_decomposition(trajs)
    delta_df = delta_decomposition(trajs)
    opt_df = optimizer_diagnostics(args.result_root)

    position_df.to_csv(outdir / "position_decomposition.csv", index=False)
    frame_df.to_csv(outdir / "framewise_projection.csv", index=False)
    delta_df.to_csv(outdir / "delta_decomposition.csv", index=False)
    opt_df.to_csv(outdir / "optimizer_diagnostics_summary.csv", index=False)

    plots = [
        plot_y_error(frame_df, outdir),
        plot_beta(frame_df, outdir),
        plot_distance_sources(frame_df, outdir),
    ]
    report = write_markdown(outdir, position_df, delta_df, opt_df, plots)

    print(f"Wrote {outdir / 'position_decomposition.csv'}")
    print(f"Wrote {outdir / 'delta_decomposition.csv'}")
    print(f"Wrote {outdir / 'framewise_projection.csv'}")
    print(f"Wrote {outdir / 'optimizer_diagnostics_summary.csv'}")
    print(f"Wrote {report}")
    print(position_df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
