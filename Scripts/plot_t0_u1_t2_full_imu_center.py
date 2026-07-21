#!/usr/bin/env python3
"""Compare full normal-noise T0/U1/T2 trajectories at the IMU center."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Scripts.build_imu_center_historical_comparison import (  # noqa: E402
    BATCH_ROOT,
    HTML_TEMPLATE,
    accuracy_metrics,
    load_poses,
    metadata_levers,
    metrics,
    position_errors,
    read_forward_axes,
    read_xyz,
    write_poses,
    xyz,
)
from Utility.TrajectoryReference import (  # noqa: E402
    rebase_pose_positions,
    translate_pose_reference_point,
)


OUTPUT = ROOT / "analysis_t0_u1_t2_full_imu_center_20260720"
PURE_ROOT = ROOT / (
    "Results/visual_factor_cache_static63_unique_source_20260713/"
    "trial_1/pure_macvo"
)
T2_ROOT = ROOT / (
    "Results/normal_noise_compressed_uvd_t2_full_three_scenes_20260720/"
    "trial_1/vio_two_state_compressed_uvd_t2_full"
)

METHODS = {
    "t0": {
        "label": "T0 - Relative-pose factor",
        "color": "#dc2626",
    },
    "u1": {
        "label": "U1 - Direct UVD factor",
        "color": "#16a34a",
    },
    "t2": {
        "label": "T2 - Compressed UVD pose factor",
        "color": "#2563eb",
    },
}

SCENES = {
    "circle": {
        "label": "Circle / Normal noise / Full 63 s / IMU center",
        "dataset": "clear_circle_truth_normal_noise",
        "frames": 1890,
        "t0": ROOT / (
            "Results/circle_straight_normal_noise_two_state_standard_full_20260715/"
            "trial_1/vio_two_state_fixed_lag_standard_full/"
            "clear_circle_truth_normal_noise/poses.csv"
        ),
        "u1": ROOT / (
            "Results/circle_normal_noise_direct_uvd_u1_full_20260716/"
            "trial_1/vio_two_state_direct_uvd_u1_standard_full/"
            "clear_circle_truth_normal_noise/poses.csv"
        ),
    },
    "rectangle": {
        "label": "Stop-turn rectangle / Normal noise / Full 63 s / IMU center",
        "dataset": "clear_stop_turn_rectangle_truth_normal_noise",
        "frames": 1890,
        "t0": ROOT / (
            "Results/rectangle_normal_noise_two_state_standard_full_20260715/"
            "trial_1/vio_two_state_fixed_lag_standard_full/"
            "clear_stop_turn_rectangle_truth_normal_noise/poses.csv"
        ),
        "u1": ROOT / (
            "Results/normal_noise_direct_uvd_u1_missing_full_scenes_20260717/"
            "trial_1/vio_two_state_direct_uvd_u1_standard_full/"
            "clear_stop_turn_rectangle_truth_normal_noise/poses.csv"
        ),
    },
    "straight": {
        "label": "Straight / Normal noise / Full 21 s / IMU center",
        "dataset": "clear_straight_truth_normal_noise",
        "frames": 630,
        "t0": ROOT / (
            "Results/circle_straight_normal_noise_two_state_standard_full_20260715/"
            "trial_1/vio_two_state_fixed_lag_standard_full/"
            "clear_straight_truth_normal_noise/poses.csv"
        ),
        "u1": ROOT / (
            "Results/normal_noise_direct_uvd_u1_missing_full_scenes_20260717/"
            "trial_1/vio_two_state_direct_uvd_u1_standard_full/"
            "clear_straight_truth_normal_noise/poses.csv"
        ),
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def assert_finite(name: str, values: np.ndarray) -> None:
    if not np.isfinite(values).all():
        raise FloatingPointError(f"{name} contains NaN/Inf")


def make_trace(
    *,
    scene: str,
    method: str,
    path: Path,
    gt_rows: list[tuple[int, float, float, float]],
) -> dict[str, object]:
    rows = read_xyz(path)
    spec = METHODS[method]
    return {
        "key": f"{scene}_{method}",
        "source": method,
        "config": "normal_noise",
        "optimizer": method,
        "label": spec["label"],
        "color": spec["color"],
        "dasharray": "",
        "scene": scene,
        "xyz": xyz(rows),
        "forward": read_forward_axes(path),
        "error_m": position_errors(gt_rows, rows, xy_only=False),
        "metrics": metrics(gt_rows, rows),
        "path": str(path),
    }


def validate_t2_config(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    required = {
        "imu_factor_mode": "two_state_fixed_lag",
        "two_state_visual_factor_mode": "compressed_uvd",
        "two_state_warm_start": "macvo_pose",
        "two_state_covariance_mode": "current_independent_step",
        "imu_vio_gravity_handling": "standard_local_frame_preintegration",
        "imu_static_initialization_enable": "true",
        "imu_static_initialization_duration_s": "3.0",
    }
    for key, value in required.items():
        token = f"{key}: {value}"
        if token not in text:
            raise AssertionError(f"missing T2 config contract: {token}")


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    payloads: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []
    manifest_rows: list[dict[str, object]] = []
    scene_contracts: dict[str, object] = {}

    for scene, spec in SCENES.items():
        dataset = str(spec["dataset"])
        expected_frames = int(spec["frames"])
        dataset_dir = BATCH_ROOT / dataset
        metadata_path = dataset_dir / "metadata.json"
        gt_path = dataset_dir / "ref_pose.csv"
        t2_dir = T2_ROOT / dataset
        pure_path = PURE_ROOT / dataset / "poses.csv"
        sources = {
            "t0": Path(spec["t0"]),
            "u1": Path(spec["u1"]),
            "t2": t2_dir / "poses.csv",
        }
        required_paths = [
            metadata_path,
            gt_path,
            pure_path,
            t2_dir / "config.yaml",
            t2_dir / "poses_imu.csv",
            *sources.values(),
        ]
        for path in required_paths:
            if not path.is_file():
                raise FileNotFoundError(path)
        validate_t2_config(t2_dir / "config.yaml")

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        levers = metadata_levers(metadata)
        gt_time, gt_body = load_poses(gt_path)
        if len(gt_time) != expected_frames:
            raise AssertionError(f"{scene}: GT has {len(gt_time)} rows, expected {expected_frames}")
        assert_finite(f"{scene}/gt", gt_body)
        gt_imu = translate_pose_reference_point(gt_body, levers["body_to_imu_nwu"])
        gt_rebased = rebase_pose_positions(gt_imu)

        converted_dir = OUTPUT / "converted" / scene
        gt_converted = converted_dir / "gt_imu_rebased.csv"
        write_poses(gt_converted, gt_time, gt_rebased)
        gt_rows = read_xyz(gt_converted)
        converted: dict[str, Path] = {}

        for method, source in sources.items():
            timestamps, camera_pose = load_poses(source)
            if len(timestamps) != expected_frames:
                raise AssertionError(
                    f"{scene}/{method}: {len(timestamps)} rows, expected {expected_frames}"
                )
            if not np.array_equal(timestamps, gt_time):
                raise AssertionError(f"{scene}/{method}: timestamps differ from GT")
            assert_finite(f"{scene}/{method}", camera_pose)
            imu_pose = translate_pose_reference_point(
                camera_pose, levers["camera_to_imu_nwu"]
            )
            imu_rebased = rebase_pose_positions(imu_pose)
            assert_finite(f"{scene}/{method}/imu", imu_rebased)
            converted_path = converted_dir / f"{method}_imu_rebased.csv"
            write_poses(converted_path, timestamps, imu_rebased)
            converted[method] = converted_path

            values = accuracy_metrics(gt_rebased, imu_rebased, active_from=90)
            metric_rows.append(
                {
                    "scene": scene,
                    "method": method,
                    **values,
                    "source_path": str(source),
                    "converted_path": str(converted_path),
                }
            )
            manifest_rows.append(
                {
                    "scene": scene,
                    "method": method,
                    "source_path": str(source),
                    "source_sha256": sha256(source),
                    "converted_path": str(converted_path),
                    "frame_count": len(timestamps),
                    "timestamp_start_ns": int(timestamps[0]),
                    "timestamp_end_ns": int(timestamps[-1]),
                }
            )

        # Pure MACVO is the independent visual-only baseline.  Keep it separate
        # from T0: T0 is already a VIO factor-graph result.
        pure_timestamps, pure_camera_pose = load_poses(pure_path)
        if len(pure_timestamps) != expected_frames:
            raise AssertionError(
                f"{scene}/pure_macvo: {len(pure_timestamps)} rows, expected {expected_frames}"
            )
        if not np.array_equal(pure_timestamps, gt_time):
            raise AssertionError(f"{scene}/pure_macvo: timestamps differ from GT")
        assert_finite(f"{scene}/pure_macvo", pure_camera_pose)
        pure_imu_pose = translate_pose_reference_point(
            pure_camera_pose, levers["camera_to_imu_nwu"]
        )
        pure_imu_rebased = rebase_pose_positions(pure_imu_pose)
        assert_finite(f"{scene}/pure_macvo/imu", pure_imu_rebased)
        pure_converted_path = converted_dir / "pure_macvo_imu_rebased.csv"
        write_poses(pure_converted_path, pure_timestamps, pure_imu_rebased)
        converted["pure_macvo"] = pure_converted_path
        pure_values = accuracy_metrics(gt_rebased, pure_imu_rebased, active_from=90)
        metric_rows.append(
            {
                "scene": scene,
                "method": "pure_macvo",
                **pure_values,
                "source_path": str(pure_path),
                "converted_path": str(pure_converted_path),
            }
        )
        manifest_rows.append(
            {
                "scene": scene,
                "method": "pure_macvo",
                "source_path": str(pure_path),
                "source_sha256": sha256(pure_path),
                "converted_path": str(pure_converted_path),
                "frame_count": len(pure_timestamps),
                "timestamp_start_ns": int(pure_timestamps[0]),
                "timestamp_end_ns": int(pure_timestamps[-1]),
            }
        )

        # Production emits an unrebased IMU-center trajectory as an independent check.
        production_time, production_t2_imu = load_poses(t2_dir / "poses_imu.csv")
        if not np.array_equal(production_time, gt_time):
            raise AssertionError(f"{scene}/T2 poses_imu timestamps differ from GT")
        _, raw_t2_camera = load_poses(sources["t2"])
        independently_converted_t2 = translate_pose_reference_point(
            raw_t2_camera, levers["camera_to_imu_nwu"]
        )
        t2_production_max_error = float(
            np.max(np.abs(independently_converted_t2 - production_t2_imu))
        )
        if t2_production_max_error > 1.0e-7:
            raise AssertionError(
                f"{scene}: T2 IMU-center conversion mismatch {t2_production_max_error}"
            )

        scene_contracts[scene] = {
            "dataset": dataset,
            "frame_count": expected_frames,
            "body_to_imu_nwu_m": levers["body_to_imu_nwu"].tolist(),
            "body_to_camera_nwu_m": levers["body_to_camera_nwu"].tolist(),
            "camera_to_imu_nwu_m": levers["camera_to_imu_nwu"].tolist(),
            "timestamps_identical": True,
            "all_values_finite": True,
            "t2_independent_vs_production_imu_max_abs_error": t2_production_max_error,
        }

        pure_rows = read_xyz(converted["pure_macvo"])
        payloads.append(
            {
                "scene": str(spec["label"]),
                "gt": xyz(gt_rows),
                "gt_forward": read_forward_axes(gt_converted),
                "macvo": xyz(pure_rows),
                "macvo_forward": read_forward_axes(converted["pure_macvo"]),
                "time_s": ((gt_time - gt_time[0]) * 1.0e-9).tolist(),
                "error_m": position_errors(gt_rows, pure_rows, xy_only=False),
                "metrics": metrics(gt_rows, pure_rows),
                "fusion": [
                    make_trace(
                        scene=dataset,
                        method=method,
                        path=converted[method],
                        gt_rows=gt_rows,
                    )
                    for method in ("t0", "u1", "t2")
                ],
                "imu_only": [],
                "gt_path": str(gt_converted),
                "macvo_path": str(converted["pure_macvo"]),
            }
        )

    metrics_path = OUTPUT / "t0_u1_t2_full_imu_center_metrics.csv"
    pd.DataFrame(metric_rows).to_csv(metrics_path, index=False, float_format="%.12g")
    manifest_path = OUTPUT / "trajectory_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)

    contract = {
        "schema_version": 1,
        "world_frame": "NWU",
        "estimate_input_reference": "left camera center / T_WC",
        "estimate_output_reference": "IMU center / T_WI",
        "estimate_conversion": "p_WI = p_WC + R_WC * (t_BI - t_BC)",
        "gt_input_reference": "body/root rotation origin / T_WB",
        "gt_output_reference": "IMU center / T_WI",
        "gt_conversion": "p_WI = p_WB + R_WB * t_BI",
        "rebase": "translation-only subtraction of each converted first position",
        "alignment": "none; no SE(3), yaw, rotation, scale, or trajectory fitting",
        "line_style": "solid for GT, T0, U1, and T2",
        "scenes": scene_contracts,
    }
    contract_path = OUTPUT / "reference_contract.json"
    contract_path.write_text(
        json.dumps(contract, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    template = HTML_TEMPLATE.replace("{{", "{").replace("}}", "}")
    template = template.replace(
        "Circle, stop-turn rectangle, and straight trajectory comparison",
        "Full normal-noise T0 / U1 / T2 comparison at the IMU center",
    )
    template = template.replace(
        "__METHOD_SCOPE__",
        "GT, Pure MACVO, T0 relative-pose factor, U1 direct UVD factor, and T2 compressed UVD pose factor",
    )
    template = template.replace(
        "__LINE_NOTE__",
        "All trajectories use solid lines. Estimates and GT are converted to the IMU center in NWU and independently translation-rebased at frame 0; no fitting is applied.",
    )
    template = template.replace("> MACVO</label>", "> Pure MACVO</label>")
    template = template.replace("MACVO RMSE", "Pure MACVO RMSE")
    html_path = OUTPUT / "interactive_t0_u1_t2_full_imu_center.html"
    html_path.write_text(
        template.replace(
            "__DATA__", json.dumps({"scenes": payloads}, ensure_ascii=False)
        ),
        encoding="utf-8",
    )

    figure, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)
    for axis, (scene, spec) in zip(axes, SCENES.items()):
        gt = np.asarray(payloads[list(SCENES).index(scene)]["gt"], dtype=np.float64)
        axis.plot(gt[:, 0], gt[:, 1], color="#111827", lw=2.5, label="GT")
        pure_path = OUTPUT / "converted" / scene / "pure_macvo_imu_rebased.csv"
        pure_points = np.asarray(xyz(read_xyz(pure_path)), dtype=np.float64)
        axis.plot(
            pure_points[:, 0],
            pure_points[:, 1],
            color="#e8590c",
            lw=1.4,
            linestyle="-",
            label="Pure MACVO",
        )
        for method in ("t0", "u1", "t2"):
            path = OUTPUT / "converted" / scene / f"{method}_imu_rebased.csv"
            points = np.asarray(xyz(read_xyz(path)), dtype=np.float64)
            axis.plot(
                points[:, 0],
                points[:, 1],
                color=METHODS[method]["color"],
                lw=1.4,
                linestyle="-",
                label=METHODS[method]["label"],
            )
        axis.set_title(str(spec["label"]))
        axis.set_xlabel("x / m (NWU)")
        axis.set_ylabel("y / m (NWU)")
        axis.set_aspect("equal", adjustable="datalim")
        axis.grid(alpha=0.25)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=4)
    figure.suptitle("T0 / U1 / T2 full trajectories at the IMU center")
    png_path = OUTPUT / "t0_u1_t2_full_imu_center_xy.png"
    figure.savefig(png_path, dpi=180, bbox_inches="tight")
    plt.close(figure)

    print(pd.DataFrame(metric_rows).to_string(index=False))
    print(f"HTML: {html_path}")
    print(f"PNG: {png_path}")
    print(f"Metrics: {metrics_path}")
    print(f"Contract: {contract_path}")


if __name__ == "__main__":
    main()
