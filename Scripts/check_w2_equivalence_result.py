#!/usr/bin/env python3
"""Compare preintegrated VIO and W=2 local inertial BA joined trajectories."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--analysis-root",
        type=Path,
        default=Path("analysis_w2_equivalence_biaslinfix_20260709"),
    )
    parser.add_argument("--scene", default="clear_rectangle_normal_noise")
    parser.add_argument("--preint-method", default="vio_preintegrated_full_imuatt_estinit")
    parser.add_argument("--w2-method", default="vio_local_ba_w2_imuatt")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    traj_root = args.analysis_root / args.scene / "trajectories"
    preint_path = traj_root / f"{args.scene}_{args.preint_method}_joined.csv"
    w2_path = traj_root / f"{args.scene}_{args.w2_method}_joined.csv"
    if not preint_path.exists():
        raise FileNotFoundError(preint_path)
    if not w2_path.exists():
        raise FileNotFoundError(w2_path)

    preint = pd.read_csv(preint_path)
    w2 = pd.read_csv(w2_path)
    cols = ["timestamp_ns", "tx_est", "ty_est", "tz_est", "qx_est", "qy_est", "qz_est", "qw_est"]
    merged = preint[cols].merge(w2[cols], on="timestamp_ns", suffixes=("_preint", "_w2"))
    if merged.empty:
        raise ValueError("No matched timestamps between compared trajectories")

    pos_preint = merged[["tx_est_preint", "ty_est_preint", "tz_est_preint"]].to_numpy(float)
    pos_w2 = merged[["tx_est_w2", "ty_est_w2", "tz_est_w2"]].to_numpy(float)
    pos_diff = np.linalg.norm(pos_preint - pos_w2, axis=1)

    quat_preint = merged[["qx_est_preint", "qy_est_preint", "qz_est_preint", "qw_est_preint"]].to_numpy(float)
    quat_w2 = merged[["qx_est_w2", "qy_est_w2", "qz_est_w2", "qw_est_w2"]].to_numpy(float)
    quat_diff = np.linalg.norm(quat_preint - quat_w2, axis=1)

    summary = pd.read_csv(args.analysis_root / "trajectory_summary.csv")
    print(f"scene={args.scene}")
    print(f"matched_frames={len(merged)}")
    print(f"position_diff_rmse_m={float(np.sqrt(np.mean(pos_diff**2))):.12g}")
    print(f"position_diff_median_m={float(np.median(pos_diff)):.12g}")
    print(f"position_diff_max_m={float(np.max(pos_diff)):.12g}")
    print(f"quat_l2_diff_max={float(np.max(quat_diff)):.12g}")
    print(f"first_max_pos_diff_frame={int(np.argmax(pos_diff))}")
    print("")
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
