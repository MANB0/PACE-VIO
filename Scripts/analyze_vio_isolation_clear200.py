from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation


ROOT = Path("Results/vio_isolation_clear200_20260703")
OUT = Path("analysis_vio_isolation_clear200_20260703")
SCENE_ROOT = Path("/mnt/e/文档/holoocean/code/recordings/batch_20260515_155653/clear_shallow")

VARIANTS = [
    "vio_preintegrated_full",
    "vio_preintegrated_full_gtgravity",
    "vio_preintegrated_full_no_velfb",
    "vio_preintegrated_full_cov1000",
]

CONTEXT_RUNS = {
    "existing_pure_macvo_300f_trial1": Path(
        "Results/vio_imu_kept12_main3_3trial_300f/trial_1/pure_macvo/clear_shallow/poses.csv"
    ),
    "existing_rotation_only_300f_trial1": Path(
        "Results/vio_imu_kept12_main3_3trial_300f/trial_1/rotation_only/clear_shallow/poses.csv"
    ),
    "existing_latest_cpb_trial1": Path(
        "Results/v3bpp_latest_cpb_fdonly_early7_3x/trial_1/clear_shallow/poses.csv"
    ),
}


def read_pose_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "timestamp" in df.columns and "timestamp_ns" not in df.columns:
        df = df.rename(columns={"timestamp": "timestamp_ns", "x": "tx", "y": "ty", "z": "tz"})
    return df


def normalize_quat(q: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(q, axis=1, keepdims=True)
    return q / np.maximum(norm, 1e-12)


def first_crossing(series: pd.Series, threshold: float) -> int | str:
    hit = series[series > threshold]
    if hit.empty:
        return ""
    return int(hit.index[0])


def pose_metrics(est: pd.DataFrame, gt: pd.DataFrame) -> dict[str, float]:
    joined = est.merge(gt, on="timestamp_ns", suffixes=("_est", "_gt"))
    est_pos = joined[["tx_est", "ty_est", "tz_est"]].to_numpy(float)
    gt_pos = joined[["tx_gt", "ty_gt", "tz_gt"]].to_numpy(float)
    est_rel = est_pos - est_pos[0]
    gt_rel = gt_pos - gt_pos[0]
    err = est_rel - gt_rel
    err_norm = np.linalg.norm(err, axis=1)

    q_est = normalize_quat(joined[["qx_est", "qy_est", "qz_est", "qw_est"]].to_numpy(float))
    q_gt = normalize_quat(joined[["qx_gt", "qy_gt", "qz_gt", "qw_gt"]].to_numpy(float))
    rot_est = Rotation.from_quat(q_est)
    rot_gt = Rotation.from_quat(q_gt)
    rot_err_deg = (rot_gt.inv() * rot_est).magnitude() * 180.0 / np.pi

    return {
        "matched_frames": len(joined),
        "ate_rmse_m": float(np.sqrt(np.mean(err_norm**2))),
        "ate_median_m": float(np.median(err_norm)),
        "ate_final_m": float(err_norm[-1]),
        "ate_max_m": float(np.max(err_norm)),
        "rot_median_deg": float(np.median(rot_err_deg)),
        "rot_final_deg": float(rot_err_deg[-1]),
        "rot_max_deg": float(np.max(rot_err_deg)),
    }


def finite_norm_stats(arr: np.ndarray) -> tuple[float, float]:
    arr = np.asarray(arr, dtype=float)
    if arr.ndim == 1:
        vals = np.abs(arr)
    else:
        vals = np.linalg.norm(arr, axis=1)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return float("nan"), float("nan")
    return float(vals[-1]), float(vals.max())


def main() -> None:
    OUT.mkdir(exist_ok=True)
    gt = read_pose_csv(SCENE_ROOT / "ref_pose.csv")

    rows: list[dict[str, object]] = []
    trajs: dict[str, pd.DataFrame] = {}

    for variant in VARIANTS:
        run_dir = ROOT / "trial_1" / variant / "clear_shallow"
        est = read_pose_csv(run_dir / "poses.csv")
        diag = pd.read_csv(run_dir / "frame_pair_diagnostics.csv")
        npz = np.load(run_dir / "tensor_map.npz", allow_pickle=True)

        row: dict[str, object] = {
            "variant": variant,
            "pose_frame": (run_dir / "pose_coordinate_frame.txt").read_text().strip(),
            "poses": len(est),
            "diag_pairs": len(diag),
            "vio_active_pairs": int(diag["vio_factor_active"].fillna(0).sum()),
            "rot_pairs": int(diag["use_imu_rotation"].astype(bool).sum()),
            "trans_pairs": int(diag["use_imu_translation"].astype(bool).sum()),
        }
        row.update(pose_metrics(est, gt))

        ratio = diag["est_over_gt_translation_ratio"].replace([np.inf, -np.inf], np.nan)
        ratio = ratio.dropna().reset_index(drop=True)
        row.update(
            {
                "ratio_median": float(ratio.median()) if len(ratio) else float("nan"),
                "ratio_p90": float(ratio.quantile(0.9)) if len(ratio) else float("nan"),
                "ratio_max": float(ratio.max()) if len(ratio) else float("nan"),
                "first_ratio_gt_2_pair": first_crossing(ratio, 2.0),
                "first_ratio_gt_5_pair": first_crossing(ratio, 5.0),
                "first_ratio_gt_10_pair": first_crossing(ratio, 10.0),
                "imu_delta_v_median": float(diag["imu_delta_v_norm"].median()),
                "imu_delta_v_p90": float(diag["imu_delta_v_norm"].quantile(0.9)),
                "imu_delta_v_max": float(diag["imu_delta_v_norm"].max()),
                "r_p_whitened_max": float(diag["r_p_whitened_norm"].max()),
                "r_R_whitened_max": float(diag["r_R_whitened_norm"].max()),
            }
        )

        for key, label in [
            ("frames//imu_vio_velocity_world", "vel_opt"),
            ("frames//imu_vio_curr_velocity_init_world", "vel_init"),
            ("frames//imu_vio_prev_velocity_world", "vel_prev"),
            ("frames//imu_vio_delta_v", "delta_v_state"),
        ]:
            if key in npz.files:
                final_val, max_val = finite_norm_stats(npz[key])
                row[f"{label}_final_norm"] = final_val
                row[f"{label}_max_norm"] = max_val

        rows.append(row)
        trajs[variant] = est

    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "summary.csv", index=False)

    context_rows: list[dict[str, object]] = []
    for name, pose_path in CONTEXT_RUNS.items():
        if not pose_path.exists():
            continue
        est = read_pose_csv(pose_path).iloc[:200].copy()
        row: dict[str, object] = {
            "variant": name,
            "poses": len(est),
            "source": str(pose_path),
        }
        row.update(pose_metrics(est, gt))
        context_rows.append(row)
    context = pd.DataFrame(context_rows)
    if len(context):
        context.to_csv(OUT / "context_existing_methods_200f.csv", index=False)

    gt_short = gt.iloc[: max(len(v) for v in trajs.values())]
    fig, ax = plt.subplots(figsize=(7.2, 5.4), dpi=180)
    ax.plot(gt_short["tx"], gt_short["ty"], color="black", lw=2.0, label="GT")
    for variant, est in trajs.items():
        label = variant.replace("vio_preintegrated_full", "full")
        ax.plot(est["tx"], est["ty"], lw=1.4, label=label)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("x / m (NWU)")
    ax.set_ylabel("y / m (NWU)")
    ax.grid(True, linewidth=0.35, alpha=0.5)
    ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(OUT / "trajectory_xy.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 5.4), dpi=180)
    ax.plot(gt_short["tx"], gt_short["ty"], color="black", lw=2.0, label="GT")
    zoom_variants = [variant for variant in VARIANTS if variant != "vio_preintegrated_full"]
    xs = [gt_short["tx"].to_numpy(float)]
    ys = [gt_short["ty"].to_numpy(float)]
    for variant in zoom_variants:
        est = trajs[variant]
        label = variant.replace("vio_preintegrated_full", "full")
        ax.plot(est["tx"], est["ty"], lw=1.4, label=label)
        xs.append(est["tx"].to_numpy(float))
        ys.append(est["ty"].to_numpy(float))
    all_x = np.concatenate(xs)
    all_y = np.concatenate(ys)
    pad_x = max((float(all_x.max() - all_x.min())) * 0.15, 0.25)
    pad_y = max((float(all_y.max() - all_y.min())) * 0.15, 0.25)
    ax.set_xlim(float(all_x.min() - pad_x), float(all_x.max() + pad_x))
    ax.set_ylim(float(all_y.min() - pad_y), float(all_y.max() + pad_y))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x / m (NWU)")
    ax.set_ylabel("y / m (NWU)")
    ax.grid(True, linewidth=0.35, alpha=0.5)
    ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(OUT / "trajectory_xy_zoom.png")
    plt.close(fig)

    cols = [
        "variant",
        "ate_rmse_m",
        "ate_final_m",
        "ate_max_m",
        "rot_median_deg",
        "rot_final_deg",
        "vel_opt_final_norm",
        "vel_opt_max_norm",
        "delta_v_state_max_norm",
        "ratio_max",
        "first_ratio_gt_10_pair",
        "r_p_whitened_max",
        "r_R_whitened_max",
    ]
    md = [
        "# VIO isolation analysis: clear_shallow first 200 frames",
        "",
        "All variants force full IMU translation and rotation with the preintegrated VIO factor.",
        "",
        summary[cols].to_markdown(index=False, floatfmt=".6g"),
        "",
    ]
    if len(context):
        context_cols = [
            "variant",
            "ate_rmse_m",
            "ate_final_m",
            "rot_median_deg",
            "rot_final_deg",
            "source",
        ]
        md.extend(
            [
                "## Existing-method context, same first 200 frames",
                "",
                context[context_cols].to_markdown(index=False, floatfmt=".6g"),
                "",
            ]
        )
    md.extend(
        [
        f"- Summary CSV: `{OUT / 'summary.csv'}`",
        f"- Existing-method context CSV: `{OUT / 'context_existing_methods_200f.csv'}`",
        f"- XY trajectory: `{OUT / 'trajectory_xy.png'}`",
        f"- Zoomed XY trajectory: `{OUT / 'trajectory_xy_zoom.png'}`",
        ]
    )
    (OUT / "summary.md").write_text("\n".join(md) + "\n")

    print(summary[cols].to_string(index=False))
    print(f"Wrote {OUT / 'summary.csv'}")
    print(f"Wrote {OUT / 'trajectory_xy.png'}")


if __name__ == "__main__":
    main()
